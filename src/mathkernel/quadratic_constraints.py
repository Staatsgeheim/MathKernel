# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Quadratic constraints and global quadratic-Lagrangian witnesses.

A PSD Lagrangian Hessian, stationarity, feasibility and complementarity suffice
for global optimality, even if individual quadratics are nonconvex. Failing that
sufficient test does not disprove optimality. No generic nonlinear claim is made.
"""
from __future__ import annotations
import sympy as sp
from pydantic import model_validator
from .engineering import EngineeringModel, EngineeringResult, arithmetic_trust, cap_trust, checked_result, numeric_array, validate_scalars
from .optimization import OptimizationProblem, canonical, exact_psd, _rational_candidate


class QuadraticConstraint(EngineeringModel):
    Q: tuple[tuple[sp.Expr, ...], ...]
    a: tuple[sp.Expr, ...]
    r: sp.Expr = sp.S.Zero

    @model_validator(mode='after')
    def validate_quadratic(self):
        n = len(self.a)
        if not n or len(self.Q) != n or any(len(row) != n for row in self.Q):
            raise ValueError('quadratic constraint Q must be square and match a')
        if any(self.Q[i][j] != self.Q[j][i] for i in range(n) for j in range(i)):
            raise ValueError('quadratic constraint Q must be explicitly symmetric')
        entries = (*self.a, self.r, *(v for row in self.Q for v in row))
        validate_scalars(entries, real=True)
        if any(v.free_symbols for v in entries):
            raise ValueError('quadratic constraint coefficients must be concrete')
        return self


class QuadraticallyConstrainedProblem(OptimizationProblem):
    quadratics: tuple[QuadraticConstraint, ...]

    @model_validator(mode='after')
    def validate_quadratics(self):
        if any(self.integrality):
            raise ValueError('integer quadratic constraints are not implemented')
        if not self.quadratics or any(len(q.a) != len(self.c) for q in self.quadratics):
            raise ValueError('quadratics must be nonempty and match the objective dimension')
        return self

    @property
    def family(self):
        return 'qcqp'


class QuadraticCertificate(EngineeringModel):
    primal: tuple[sp.Expr, ...]
    inequality_dual: tuple[sp.Expr, ...] = ()
    equality_dual: tuple[sp.Expr, ...] = ()
    quadratic_dual: tuple[sp.Expr, ...]

    @model_validator(mode='after')
    def validate_witness(self):
        values = (*self.primal, *self.inequality_dual, *self.equality_dual, *self.quadratic_dual)
        validate_scalars(values, real=True)
        if any(v.free_symbols for v in values):
            raise ValueError('quadratic witness must be concrete')
        return self


def problem_entries(problem):
    for matrix in canonical(problem):
        yield from matrix
    for q in problem.quadratics:
        yield from q.a
        yield q.r
        for row in q.Q:
            yield from row


def verify_certificate(problem, certificate, *, certificate_trust="exact"):
    Q, c, G, h, E, f = canonical(problem)
    n, k = len(problem.c), len(problem.quadratics)
    if (len(certificate.primal) != n or len(certificate.inequality_dual) != G.rows or
        len(certificate.equality_dual) != E.rows or len(certificate.quadratic_dual) != k):
        raise ValueError('quadratic certificate dimensions must match all constraints including bounds')
    x = sp.Matrix(certificate.primal)
    lam = sp.Matrix(G.rows, 1, certificate.inequality_dual)
    nu = sp.Matrix(E.rows, 1, certificate.equality_dual)
    H, linear = Q.copy(), c+G.T*lam+E.T*nu
    constant = -(h.T*lam)[0]-(f.T*nu)[0]
    values = []
    for q, multiplier in zip(problem.quadratics, certificate.quadratic_dual):
        matrix, a = sp.Matrix(q.Q), sp.Matrix(q.a)
        values.append((x.T*matrix*x)[0]/2+(a.T*x)[0]+q.r)
        H += multiplier*matrix
        linear += multiplier*a
        constant += multiplier*q.r
    stationarity = H*x+linear
    slack = h-G*x
    psd, pivots = exact_psd(H)
    primal = (x.T*Q*x)[0]/2+(c.T*x)[0]
    dual = constant-(x.T*H*x)[0]/2
    checks = {'linear_feasibility': all(v.is_nonnegative is True for v in slack),
        'equality_feasibility': all(v == 0 for v in E*x-f),
        'quadratic_feasibility': all(v.is_nonpositive is True for v in values),
        'dual_nonnegative': all(v.is_nonnegative is True for v in (*lam, *certificate.quadratic_dual)),
        'stationarity': all(sp.cancel(v) == 0 for v in stationarity),
        'linear_complementarity': all(l*s == 0 for l, s in zip(lam, slack)),
        'quadratic_complementarity': all(v*l == 0 for v, l in zip(values, certificate.quadratic_dual)),
        'lagrangian_psd': psd is True, 'zero_duality_gap': sp.cancel(primal-dual) == 0}
    entries = (*problem_entries(problem), *certificate.primal, *certificate.inequality_dual,
               *certificate.equality_dual, *certificate.quadratic_dual)
    trust = cap_trust(arithmetic_trust(entries, problem.input_trust), certificate_trust)
    accepted = all(checks.values()) and trust == 'exact' and all(v.is_Rational for v in entries)
    sign = 1 if problem.sense == 'min' else -1
    return checked_result('verify_certificate', sign*primal, method='global_quadratic_lagrangian',
        trust=trust, checks={'certificate_accepted': True} if accepted else {},
        witness=certificate.model_dump(mode='json'), candidate=not accepted,
        details={'accepted': accepted, 'conclusion': 'certified_global_optimum' if accepted else 'uncertified_candidate',
                 'certificate_checks': checks, 'primal': certificate.primal, 'primal_value': sign*primal,
                 'dual_value': sign*dual, 'duality_gap': sp.cancel(primal-dual), 'quadratic_values': values,
                 'lagrangian_hessian': H.tolist(), 'psd_pivots': pivots,
                 'stationarity_residual': tuple(stationarity), 'certificate': certificate.model_dump(mode='json'),
                 'criterion': 'sufficient global Lagrangian certificate; failure is inconclusive'})


def _isolated_qcqp_solve(problem, *, initial=None, max_iterations=1000, tolerance=1e-9, reconstruction_denominator=1000000):
    import numpy as np
    from scipy.optimize import minimize, lsq_linear
    Q, c, G, h, E, f = canonical(problem)
    n = len(problem.c)
    def arr(matrix):
        return numeric_array(tuple(matrix), real=True).reshape(matrix.shape)
    q, cv, g, hv, e, fv = [arr(m) for m in (Q, c, G, h, E, f)]
    cv, hv, fv = cv.ravel(), hv.ravel(), fv.ravel()
    # Parse/convert once; analytic batched derivatives avoid symbolic callbacks.
    qs = np.stack([arr(sp.Matrix(item.Q)) for item in problem.quadratics])
    aa = np.stack([numeric_array(item.a, real=True) for item in problem.quadratics])
    rr = numeric_array(tuple(item.r for item in problem.quadratics), real=True)
    def quadratic_values(x):
        return .5*np.einsum('i,kij,j->k', x, qs, x)+aa@x+rr
    def quadratic_jacobian(x):
        return np.einsum('kij,j->ki', qs, x)+aa
    constraints = [{'type': 'ineq', 'fun': lambda x: -quadratic_values(x),
                    'jac': lambda x: -quadratic_jacobian(x)}]
    if G.rows:
        constraints.append({'type': 'ineq', 'fun': lambda x: hv-g@x, 'jac': lambda x: -g})
    if E.rows:
        constraints.append({'type': 'eq', 'fun': lambda x: e@x-fv, 'jac': lambda x: e})
    x0 = np.zeros(n) if initial is None else numeric_array(initial, real=True)
    if x0.shape != (n,):
        raise ValueError('initial point must match objective dimension')
    raw = minimize(lambda x: float(x@q@x/2+cv@x), x0, jac=lambda x: q@x+cv,
                   method='SLSQP', constraints=constraints,
                   options={'maxiter': max_iterations, 'ftol': tolerance})
    x = np.asarray(raw.x)
    if not np.all(np.isfinite(x)):
        return EngineeringResult(operation='solve', status='unknown', trust='unknown',
            details={'conclusion': 'nonfinite_solver_candidate', 'solver_message': str(raw.message)})
    vals, slack = quadratic_values(x), hv-g@x
    # A permissive active-set guess aids witness reconstruction after an
    # imperfect numerical termination; exact complementarity remains authority.
    active_tolerance = max(10*tolerance, np.sqrt(tolerance))
    active_l = np.flatnonzero(abs(slack) <= active_tolerance*(1+abs(hv)))
    active_q = np.flatnonzero(abs(vals) <= active_tolerance*(1+abs(rr)))
    basis = np.column_stack((g[active_l].T, quadratic_jacobian(x)[active_q].T, e.T))
    constrained = len(active_l)+len(active_q)
    if basis.shape[1]:
        multipliers = lsq_linear(basis, -(q@x+cv),
            bounds=(np.r_[np.zeros(constrained), np.full(E.rows, -np.inf)], np.full(basis.shape[1], np.inf)),
            max_iter=max_iterations, tol=tolerance).x
    else:
        multipliers = np.zeros(0)
    lam = np.zeros(G.rows);mu = np.zeros(len(problem.quadratics))
    lam[active_l] = multipliers[:len(active_l)]
    mu[active_q] = multipliers[len(active_l):constrained]
    nu = multipliers[constrained:]
    residual = {'primal': float(max(0, np.max(-slack, initial=0), np.max(vals, initial=0), np.max(abs(e@x-fv), initial=0))),
                'stationarity': float(np.max(abs(q@x+cv+g.T@lam+quadratic_jacobian(x).T@mu+e.T@nu))),
                'complementarity': float(max(np.max(abs(slack*lam), initial=0), np.max(abs(vals*mu), initial=0)))}
    details = {'solver': 'scipy_slsqp_analytic_qcqp', 'solver_status': int(raw.status), 'solver_message': str(raw.message),
               'primal': x.tolist(), 'residuals': residual, 'tolerance': tolerance, 'active_set_relative_tolerance': float(active_tolerance),
               'global_optimality_certified': False, 'conclusion': 'numerical_candidate'}
    report = checked_result('solve', float(x@q@x/2+cv@x)*(1 if problem.sense == 'min' else -1),
        method='scipy_slsqp_analytic_qcqp', trust='numeric', candidate=True, residual=residual, precision=53, details=details)
    certificate = None
    for denominator in sorted({min(reconstruction_denominator, d) for d in (100, 10000, reconstruction_denominator)}):
        certificate = QuadraticCertificate(primal=_rational_candidate(x, denominator),
            inequality_dual=_rational_candidate(lam, denominator), equality_dual=_rational_candidate(nu, denominator),
            quadratic_dual=_rational_candidate(mu, denominator))
        result = verify_certificate(problem, certificate)
        if result.details['accepted']:
            result.operation = 'solve'
            result.details.update(search_diagnostics=details, reconstruction_max_denominator=denominator)
            # Optional numerical search evidence is not required exact-proof ancestry.
            diagnostic = report.claim_evidence['solve']
            for item in (*diagnostic.computation, *diagnostic.numerical):
                item.role = 'diagnostic'
            bundle = result.claim_evidence['verify_certificate']
            bundle.computation.extend(diagnostic.computation);bundle.numerical.extend(diagnostic.numerical)
            return result, certificate
    report.details.update(certificate=certificate.model_dump(mode='json'), accepted=False)
    return report, certificate


def solve(problem, *, initial=None, max_iterations=1000, tolerance=1e-9,
          reconstruction_denominator=1000000, time_limit=30.0):
    from .engines import run_in_subprocess
    output = run_in_subprocess(_isolated_qcqp_solve, time_limit, problem, initial=initial,
        max_iterations=max_iterations, tolerance=tolerance,
        reconstruction_denominator=reconstruction_denominator)
    result = output[0] if isinstance(output, tuple) else output
    result.details['process_isolation'] = 'fresh_interpreter_hard_killable'
    return output
