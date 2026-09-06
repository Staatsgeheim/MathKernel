# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Certificate-aware constrained finite-horizon control.

The solver searches a sparse-in-meaning, explicit trajectory QP. Original-data
KKT/Farkas substitution is authoritative. Feasibility, optimality, terminal-set
invariance, recursive feasibility and feedback stability remain distinct claims.
"""
from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import StrictInt, model_validator

from .control_design import _copy_system
from .control_systems import _matrix_tuple, _system_trust, stability
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust,
                          checked_result, numeric_array, sympy_samples,
                          validate_scalars)
from .optimization import (OptimizationCertificate, OptimizationProblem,
                           exact_psd, solve as solve_optimization,
                           verify_certificate)
from .riccati import _data, _pd

Matrix = tuple[tuple[sp.Expr, ...], ...]


def _entries(*groups):
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (tuple, list)):
            for item in group:
                if isinstance(item, (tuple, list)):
                    yield from _entries(item)
                elif item is not None:
                    yield item
        else:
            yield group


def _bounds(values, size, name):
    if values is None:
        return (None,) * size
    if len(values) != size:
        raise ValueError(f'{name} must have {size} entries')
    validate_scalars((value for value in values if value is not None), real=True)
    if any(value is not None and value.free_symbols for value in values):
        raise ValueError(f'{name} must contain concrete real values or null')
    return tuple(values)


def _ordered(lower, upper, name):
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo is not None and hi is not None and (hi-lo).is_nonnegative is not True:
            raise ValueError(f'{name}[{i}] lower bound exceeds upper bound')


def _within(value, lo, hi):
    lower = True if lo is None else (value-lo).is_nonnegative is True
    upper = True if hi is None else (hi-value).is_nonnegative is True
    return lower and upper


class MPCPlan(EngineeringModel):
    system_id: str
    problem: OptimizationProblem
    certificate: OptimizationCertificate | None = None
    horizon: StrictInt
    state_dimension: StrictInt
    input_dimension: StrictInt
    states: tuple[tuple[sp.Expr, ...], ...]
    controls: tuple[tuple[sp.Expr, ...], ...]
    cost: sp.Expr
    terminal_gain: Matrix = ()
    backend: Literal['exact_certificate', 'numeric_candidate']
    input_trust: str = 'exact'

    @model_validator(mode='after')
    def validate_plan(self):
        n, m, N = self.state_dimension, self.input_dimension, self.horizon
        if n < 1 or m < 1 or N < 1 or len(self.states) != N+1 or len(self.controls) != N:
            raise ValueError('invalid MPC trajectory dimensions')
        if any(len(x) != n for x in self.states) or any(len(u) != m for u in self.controls):
            raise ValueError('invalid MPC state/control width')
        validate_scalars((*_entries(self.states, self.controls), self.cost), real=True)
        if self.terminal_gain:
            if len(self.terminal_gain) != m or any(len(row) != n for row in self.terminal_gain):
                raise ValueError('terminal gain must be inputs by states')
            validate_scalars(_entries(self.terminal_gain), real=True)
        cap_trust(self.input_trust)
        return self


def build_problem(system, Q, R, terminal, initial, horizon, *,
                  state_lower=None, state_upper=None, input_lower=None,
                  input_upper=None, terminal_lower=None, terminal_upper=None,
                  input_trust='exact'):
    if system.time_domain != 'discrete':
        raise ValueError('MPC requires a discrete-time system')
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('MPC horizon must be a positive integer')
    A, B = sp.Matrix(system.A), sp.Matrix(system.B)
    n, m = A.rows, B.cols
    _, _, q, r = _data(system, Q, R, False)
    F = sp.Matrix(terminal)
    if F.shape != (n, n) or F != F.T:
        raise ValueError('terminal weight must be symmetric and match the state dimension')
    validate_scalars((*F, *initial), real=True)
    if any(v.free_symbols for v in (*F, *initial)) or len(initial) != n:
        raise ValueError('initial state and terminal weight must be concrete and dimensionally valid')
    if any(v.has(sp.Float) for v in (*q, *r, *F)):
        import numpy as np
        qn,rn,fn=(numeric_array(tuple(M),real=True).reshape(M.shape) for M in (q,r,F))
        scale=lambda M:max(1.,float(np.linalg.norm(M,ord=np.inf)))
        valid=(float(np.linalg.eigvalsh(qn)[0])>=-1e-10*scale(qn) and
               float(np.linalg.eigvalsh(fn)[0])>=-1e-10*scale(fn) and
               float(np.linalg.eigvalsh(rn)[0])>1e-12*scale(rn))
    else:
        valid=exact_psd(q)[0] is True and _pd(r) and exact_psd(F)[0] is True
    if not valid:
        raise ValueError('Q/terminal must be PSD and R positive definite')

    sl, su = _bounds(state_lower, n, 'state_lower'), _bounds(state_upper, n, 'state_upper')
    ul, uu = _bounds(input_lower, m, 'input_lower'), _bounds(input_upper, m, 'input_upper')
    tl = _bounds(terminal_lower, n, 'terminal_lower') if terminal_lower is not None else sl
    tu = _bounds(terminal_upper, n, 'terminal_upper') if terminal_upper is not None else su
    _ordered(sl, su, 'state bounds'); _ordered(ul, uu, 'input bounds'); _ordered(tl, tu, 'terminal bounds')
    for i in range(n):
        if tl[i] is not None and sl[i] is not None and (tl[i]-sl[i]).is_nonnegative is not True:
            raise ValueError('terminal lower bounds must lie inside state bounds')
        if tu[i] is not None and su[i] is not None and (su[i]-tu[i]).is_nonnegative is not True:
            raise ValueError('terminal upper bounds must lie inside state bounds')
    if all(value is None for value in (*sl, *su, *ul, *uu, *tl, *tu)):
        raise ValueError('MPC requires at least one finite state, input or terminal constraint')

    d = (horizon+1)*n+horizon*m
    H = sp.zeros(d); c = (sp.S.Zero,)*d
    for t in range(horizon):
        H[t*n:(t+1)*n, t*n:(t+1)*n] = 2*q
    H[horizon*n:(horizon+1)*n, horizon*n:(horizon+1)*n] = 2*F
    offset = (horizon+1)*n
    for t in range(horizon):
        H[offset+t*m:offset+(t+1)*m, offset+t*m:offset+(t+1)*m] = 2*r

    rows, rhs = [], []
    row = [sp.S.Zero]*d
    for i in range(n):
        fixed = row.copy(); fixed[i] = sp.S.One; rows.append(tuple(fixed)); rhs.append(initial[i])
    for t in range(horizon):
        for i in range(n):
            dynamic = row.copy(); dynamic[(t+1)*n+i] = sp.S.One
            for j in range(n): dynamic[t*n+j] -= A[i,j]
            for j in range(m): dynamic[offset+t*m+j] -= B[i,j]
            rows.append(tuple(dynamic)); rhs.append(sp.S.Zero)
    lower, upper = [], []
    for t in range(horizon+1):
        lower.extend(tl if t == horizon else sl)
        upper.extend(tu if t == horizon else su)
    for _ in range(horizon): lower.extend(ul); upper.extend(uu)
    variables = tuple([f'x_{t}_{i}' for t in range(horizon+1) for i in range(n)] +
                      [f'u_{t}_{j}' for t in range(horizon) for j in range(m)])
    trust = cap_trust(_system_trust(system), input_trust,
                      arithmetic_trust(_entries(Q,R,terminal,initial,sl,su,ul,uu,tl,tu), input_trust))
    problem = OptimizationProblem(variables=variables, c=c, Q=_matrix_tuple(H),
        A_eq=tuple(rows), b_eq=tuple(rhs), lower=tuple(lower), upper=tuple(upper),
        input_trust=trust)
    metadata = {'n':n, 'm':m, 'horizon':horizon, 'state_lower':sl, 'state_upper':su,
                'input_lower':ul, 'input_upper':uu, 'terminal_lower':tl,
                'terminal_upper':tu, 'Q':_matrix_tuple(q), 'R':_matrix_tuple(r),
                'terminal':_matrix_tuple(F), 'initial':tuple(initial)}
    return problem, metadata


def _unpack(primal, n, m, horizon):
    offset = (horizon+1)*n
    states = tuple(tuple(primal[t*n+i] for i in range(n)) for t in range(horizon+1))
    controls = tuple(tuple(primal[offset+t*m+j] for j in range(m)) for t in range(horizon))
    return states, controls


def _numeric_values(values):
    return sympy_samples(values)


def _plan_checks(plan, system, tolerance=1e-9):
    from .optimization import canonical
    Q, c, G, h, Aeq, b = canonical(plan.problem)
    z = sp.Matrix((*_entries(plan.states), *_entries(plan.controls)))
    exact = plan.input_trust == 'exact' and not any(v.has(sp.Float) for v in (*Q,*c,*G,*h,*Aeq,*b,*z))
    if exact:
        equality = all(sp.cancel(v) == 0 for v in Aeq*z-b)
        inequality = all(v.is_nonnegative is True for v in h-G*z)
        objective = sp.cancel(plan.cost-((z.T*Q*z)[0]/2+(c.T*z)[0])) == 0
        residuals = None
    else:
        import numpy as np
        arr = lambda M: numeric_array(tuple(M), real=True).reshape(M.shape)
        q, cv, g, hv, ae, bv, zv = (arr(M) for M in (Q,c,G,h,Aeq,b,z))
        eq = float(np.max(np.abs(ae@zv-bv), initial=0))
        violation = float(np.max(g@zv-hv, initial=0))
        calculated = float((zv.T@q@zv)[0,0]/2+(cv.T@zv)[0,0])
        cost_error = abs(float(plan.cost)-calculated)
        scale = max(1.,abs(calculated))
        equality, inequality, objective = eq <= tolerance, violation <= tolerance, cost_error <= tolerance*scale
        residuals = {'dynamics_and_initial':eq, 'constraint_violation':max(0.,violation),
                     'cost_relative':cost_error/scale, 'tolerance':tolerance, 'error_bound':False}
    checks = {'initial_and_dynamics':bool(equality), 'trajectory_constraints':bool(inequality),
              'objective_identity':bool(objective)}
    return checks, residuals, z


def _terminal_analysis(plan, system, metadata):
    if not plan.terminal_gain:
        return {'provided':False, 'terminal_set_invariant':None,
                'terminal_control_admissible':None, 'terminal_feedback_stable':None,
                'recursive_feasibility_certified':False}, None
    K = sp.Matrix(plan.terminal_gain); A, B = sp.Matrix(system.A), sp.Matrix(system.B)
    closed = A-B*K; lo, hi = metadata['terminal_lower'], metadata['terminal_upper']
    ul, uu = metadata['input_lower'], metadata['input_upper']
    finite_box = all(value is not None for value in (*lo,*hi))
    exact = plan.input_trust == 'exact' and not any(v.has(sp.Float) for v in (*closed,*K,*_entries(lo,hi,ul,uu)))

    def image_bounds(M):
        lows, highs = [], []
        for i in range(M.rows):
            low = high = sp.S.Zero
            for j in range(M.cols):
                coefficient = M[i,j]
                if coefficient.is_nonnegative is True:
                    low += coefficient*lo[j]; high += coefficient*hi[j]
                elif coefficient.is_nonpositive is True:
                    low += coefficient*hi[j]; high += coefficient*lo[j]
                else:
                    return None, None
            lows.append(sp.cancel(low)); highs.append(sp.cancel(high))
        return tuple(lows), tuple(highs)

    invariant = admissible = False
    image_lo = image_hi = control_lo = control_hi = None
    if finite_box:
        image_lo, image_hi = image_bounds(closed)
        control_lo, control_hi = image_bounds(-K)
        invariant = image_lo is not None and all(_within(image_lo[i],lo[i],hi[i]) and
                    _within(image_hi[i],lo[i],hi[i]) for i in range(len(lo)))
        admissible = control_lo is not None and all(_within(control_lo[i],ul[i],uu[i]) and
                     _within(control_hi[i],ul[i],uu[i]) for i in range(len(ul)))
    probe = _copy_system(system, closed, sp.zeros(A.rows,1), sp.zeros(1,A.rows), sp.zeros(1,1),
                         input_unit='', output_unit='', input_units=(), output_units=(),
                         state_units=(), input_trust=plan.input_trust)
    stable = stability(probe).value is True
    terminal_state = plan.states[-1]
    terminal_reached = all(_within(terminal_state[i],lo[i],hi[i]) for i in range(len(lo)))
    recursive = bool(exact and terminal_reached and invariant and admissible)
    details = {'provided':True, 'finite_terminal_box':finite_box,
        'terminal_set_invariant':bool(invariant), 'terminal_control_admissible':bool(admissible),
        'terminal_feedback_stable':bool(stable), 'terminal_state_in_set':bool(terminal_reached),
        'terminal_feedback_stability_certified':bool(exact and stable),
        'recursive_feasibility_certified':recursive,
        'closed_loop_mpc_stability_claimed':False, 'analysis_exact':exact,
        'terminal_image_lower':image_lo, 'terminal_image_upper':image_hi,
        'terminal_control_lower':control_lo, 'terminal_control_upper':control_hi,
        'scope':'box invariance/admissibility for u=-Kx; stability applies to terminal feedback only'}
    terminal_system = _copy_system(system, closed, sp.Matrix(system.B),
        sp.Matrix(system.C)-sp.Matrix(system.D)*K, sp.Matrix(system.D),
        input_trust=plan.input_trust)
    return details, terminal_system


def verify_plan(plan, system, *, tolerance=1e-9):
    # Metadata is deterministically recoverable from the generated variable bounds.
    n,m,N = plan.state_dimension,plan.input_dimension,plan.horizon
    lower,upper=plan.problem.lower,plan.problem.upper
    metadata = {'terminal_lower':lower[N*n:(N+1)*n], 'terminal_upper':upper[N*n:(N+1)*n],
        'input_lower':lower[(N+1)*n:(N+1)*n+m], 'input_upper':upper[(N+1)*n:(N+1)*n+m]}
    checks,residuals,z = _plan_checks(plan,system,tolerance)
    trust=cap_trust(plan.input_trust,_system_trust(system),plan.problem.input_trust)
    exact_feasible=trust=='exact' and all(checks.values())
    result=checked_result('verify_mpc',plan.cost,method='original_trajectory_constraints',trust=trust,
        checks={'feasible_plan':True} if exact_feasible else {},candidate=not exact_feasible,
        residual=residuals,precision=53 if residuals else None,
        details={'accepted':exact_feasible,'feasibility_certified':exact_feasible,
            'feasibility_checks':checks,'feasibility_residuals':residuals,
            'objective_candidate':plan.cost,'optimality_certified':False,
            'infeasibility_certified':False,'claim_scope':'this finite-horizon constrained trajectory'})
    feasibility_bundle=result.claim_evidence.pop('verify_mpc');result.claim_evidence={'feasibility':feasibility_bundle}
    if plan.certificate is not None:
        optimality=verify_certificate(plan.problem,plan.certificate,certificate_trust=plan.input_trust)
        problem_optimal=bool(optimality.details.get('accepted'))
        result.details['optimality_checks']=optimality.details.get('certificate_checks',{})
        matches=len(plan.certificate.primal)==len(z) and all(sp.cancel(a-b)==0 for a,b in zip(z,plan.certificate.primal))
        result.details['problem_optimum_certified']=problem_optimal
        result.details['plan_matches_certificate']=matches
        result.details['optimality_certified']=bool(problem_optimal and matches and exact_feasible)
        for name,bundle in optimality.claim_evidence.items():result.claim_evidence['optimality' if name=='optimality' else name]=bundle
    terminal,terminal_system=_terminal_analysis(plan,system,metadata)
    result.details['terminal_analysis']=terminal
    if terminal.get('provided'):
        terminal_report=checked_result('terminal_set_invariance',terminal.get('terminal_set_invariant'),
            method='exact_box_linear_image',trust=trust,
            checks={'invariant_and_admissible':True} if terminal.get('analysis_exact') and
                   terminal.get('terminal_set_invariant') and terminal.get('terminal_control_admissible') else {},
            candidate=not bool(terminal.get('analysis_exact') and terminal.get('terminal_set_invariant') and terminal.get('terminal_control_admissible')),
            details=terminal)
        result.claim_evidence['terminal_invariance']=terminal_report.claim_evidence['terminal_set_invariance']
    outputs={'terminal_controller':terminal_system} if terminal_system is not None else {}
    return result,outputs


def synthesize(system,system_id,Q,R,terminal,initial,horizon,*,mode='exact',
               state_lower=None,state_upper=None,input_lower=None,input_upper=None,
               terminal_lower=None,terminal_upper=None,terminal_gain=None,
               certificate=None,certificate_trust='exact',input_trust='exact',
               tolerance=1e-9,max_iterations=1000,time_limit=30.,reconstruction_denominator=1_000_000):
    problem,metadata=build_problem(system,Q,R,terminal,initial,horizon,state_lower=state_lower,
        state_upper=state_upper,input_lower=input_lower,input_upper=input_upper,
        terminal_lower=terminal_lower,terminal_upper=terminal_upper,input_trust=input_trust)
    gain=()
    if terminal_gain is not None:
        gain=tuple(tuple(row) for row in terminal_gain)
        if len(gain)!=metadata['m'] or any(len(row)!=metadata['n'] for row in gain):
            raise ValueError('terminal_gain must be inputs by states')
        validate_scalars(_entries(gain),real=True)
        if any(v.free_symbols for v in _entries(gain)):
            raise ValueError('terminal_gain must contain concrete real values')
    if mode=='exact':
        if certificate is None:
            raise NotImplementedError('exact MPC requires certificate_id; use numeric mode for search')
        outcome=verify_certificate(problem,certificate,certificate_trust=certificate_trust)
        if not outcome.details.get('accepted'):
            outcome.operation='mpc';outcome.details.update(plan_created=False,
                feasibility_certified=False,optimality_certified=False)
            return outcome,{'problem':problem,'certificate':certificate}
    elif mode=='numeric':
        if certificate is not None:
            raise ValueError('certificate_id is for exact MPC replay, not numeric search')
        outcome=solve_optimization(problem,mode='numeric',max_iterations=max_iterations,
            tolerance=tolerance,time_limit=time_limit,reconstruction_denominator=reconstruction_denominator)
        certificate=outcome.details.get('certificate_model') or outcome.details.get('candidate_certificate_model')
    else:raise ValueError('mode must be exact or numeric')
    outputs={'problem':problem}
    if certificate is not None:outputs['certificate']=certificate
    if outcome.details.get('accepted') and getattr(certificate,'kind',None) in {'infeasible','unbounded'}:
        outcome.operation='mpc';outcome.details.update(mpc_horizon=horizon,plan_created=False,
            feasibility_certified=False,optimality_certified=False)
        return outcome,outputs
    if outcome.details.get('accepted') and certificate is not None:
        primal=certificate.primal;backend='exact_certificate';trust=outcome.trust
    else:
        raw=outcome.details.get('primal')
        if raw is None:
            outcome.operation='mpc';outcome.details.update(plan_created=False,
                feasibility_certified=False,optimality_certified=False)
            return outcome,outputs
        primal=_numeric_values(raw);backend='numeric_candidate';trust=cap_trust(outcome.trust,'numeric')
    states,controls=_unpack(primal,metadata['n'],metadata['m'],horizon)
    z=sp.Matrix(primal);H=sp.Matrix(problem.Q);cost=sp.cancel((z.T*H*z)[0]/2)
    plan=MPCPlan(system_id=system_id,problem=problem,certificate=certificate,horizon=horizon,
        state_dimension=metadata['n'],input_dimension=metadata['m'],states=states,controls=controls,
        cost=cost,terminal_gain=gain,backend=backend,input_trust=trust)
    verified,extra=verify_plan(plan,system,tolerance=tolerance)
    verified.operation='mpc';verified.details.update(search=outcome.details,
        plan_created=all(verified.details['feasibility_checks'].values()),
        mode=mode,control_law='apply first control, then re-solve from the measured next state',
        recursive_feasibility_scope='proved only from exact terminal box invariance/admissibility')
    # Retain the optimizer's separate proof/candidate evidence.
    for name,bundle in outcome.claim_evidence.items():
        verified.claim_evidence['optimality' if name=='optimality' else 'search_'+name]=bundle
    if all(verified.details['feasibility_checks'].values()):outputs={'output':plan,**outputs,**extra}
    return verified,outputs


def first_control(plan,system,tolerance=1e-9):
    report,_=verify_plan(plan,system,tolerance=tolerance)
    report.operation='first_control';report.value=plan.controls[0]
    report.details.update(control=plan.controls[0],requires_reoptimization=True,
        claim_scope='first move of this stored feasible plan; not an autonomous feedback-law proof')
    report.claim_evidence['first_control']=report.claim_evidence.pop('feasibility')
    return report
