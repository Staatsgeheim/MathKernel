# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Finite-horizon policies, immutable Kalman states and checked LQG composition.

Exact checks use original equations. Float64 checks are diagnostics, never proofs.
Matrices are small per-stage blocks; no horizon-sized dense problem is formed.
"""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import StrictInt, model_validator
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust, checked_result,
                          numeric_array, sympy_samples, validate_scalars)
from .control_systems import _matrix_tuple, _system_trust
from .control_design import _copy_system, channel_units
from .riccati import _data, _pd, synthesize
from .optimization import exact_psd

Matrix = tuple[tuple[sp.Expr, ...], ...]


def _concrete(entries):
    entries = tuple(entries)
    validate_scalars(entries, real=True)
    if any(v.free_symbols for v in entries):
        raise ValueError('sequential control requires concrete real data')


def _square(values, n):
    if len(values) != n or any(len(row) != n for row in values):
        raise ValueError('matrix dimension mismatch')
    _concrete(v for row in values for v in row)
    if sp.Matrix(values) != sp.Matrix(values).T:
        raise ValueError('covariance/weight must be explicitly symmetric')


class FiniteHorizonLQR(EngineeringModel):
    system_id: str
    Q: Matrix
    R: Matrix
    terminal: Matrix
    gains: tuple[Matrix, ...]
    values: tuple[Matrix, ...]
    backend: Literal['exact', 'numeric']
    input_trust: str = 'exact'

    @model_validator(mode='after')
    def dimensions(self):
        n, m = len(self.Q), len(self.R)
        if not n or not m or len(self.values) != len(self.gains)+1:
            raise ValueError('invalid finite-horizon policy dimensions')
        for matrix in (self.Q, self.terminal, *self.values):
            _square(matrix, n)
        _square(self.R, m)
        for gain in self.gains:
            if len(gain) != m or any(len(row) != n for row in gain):
                raise ValueError('gain dimension mismatch')
            _concrete(v for row in gain for v in row)
        cap_trust(self.input_trust)
        return self


class KalmanState(EngineeringModel):
    system_id: str
    mean: tuple[sp.Expr, ...]
    covariance: Matrix
    W: Matrix
    V: Matrix
    phase: Literal['prior', 'posterior'] = 'prior'
    index: StrictInt = 0
    pending_control: tuple[sp.Expr, ...] = ()
    backend: Literal['exact', 'numeric']
    input_trust: str = 'exact'

    @model_validator(mode='after')
    def dimensions(self):
        n = len(self.mean)
        if not n or self.index < 0 or not self.V:
            raise ValueError('invalid Kalman state dimension/index')
        _concrete((*self.mean, *self.pending_control))
        for matrix in (self.covariance, self.W):
            _square(matrix, n)
        _square(self.V, len(self.V))
        if (self.phase == 'prior' and self.pending_control) or (self.phase == 'posterior' and not self.pending_control):
            raise ValueError('pending control must occur exactly in posterior states')
        cap_trust(self.input_trust)
        return self


class _Arithmetic:
    """Shared small-matrix backend; Cholesky factors serve all gain RHS columns."""
    def __init__(self, mode):
        if mode not in {'exact', 'numeric'}:
            raise ValueError('mode must be exact or numeric')
        self.residuals = []
        self.numeric = mode == 'numeric'
        if self.numeric:
            import numpy as np
            self.np = np

    def matrix(self, values):
        M = sp.Matrix(values)
        _concrete(M)
        return numeric_array(tuple(M), real=True).reshape(M.shape) if self.numeric else M

    def eye(self, n):
        return self.np.eye(n) if self.numeric else sp.eye(n)

    def solve(self, S, rhs):
        if self.numeric:
            from scipy.linalg import cho_factor, cho_solve
            return cho_solve(cho_factor(S, lower=True), rhs)
        return S.solve(rhs, method='GJ')

    def clean(self, M):
        if self.numeric:
            if not self.np.all(self.np.isfinite(M)):
                raise ValueError('control calculation exceeded float64 range')
            return M
        return M.applyfunc(sp.cancel)

    def symmetric(self, M):
        return self.clean((M+M.T)/2)

    def pack(self, M):
        self.clean(M)
        if self.numeric:
            M = sp.Matrix(*M.shape, sympy_samples(M.ravel()))
        return _matrix_tuple(M)

    def psd(self, M, strict=False):
        if self.numeric:
            if not self.np.all(self.np.isfinite(M)) or not self.np.array_equal(M, M.T):
                return False
            smallest = float(self.np.linalg.eigvalsh(M)[0])
            return smallest > 0 if strict else smallest >= -1e-10*max(1., float(self.np.linalg.norm(M, ord=self.np.inf)))
        return _pd(M) if strict else exact_psd(M)[0] is True

    def equal(self, left, right):
        if self.numeric:
            norm = lambda x: float(self.np.linalg.norm(x, ord=self.np.inf))
            scale = max(1., norm(left), norm(right))
            residual = norm(left-right)
            self.residuals.append(residual/scale if self.np.isfinite(scale) and self.np.isfinite(residual) else float('inf'))
            return bool(self.np.isfinite(scale) and self.np.isfinite(residual) and residual <= 1e-9*scale)
        return all(sp.cancel(v) == 0 for v in left-right)

    def diagnostics(self):
        return {'relative_residuals': self.residuals, 'relative_tolerance': 1e-9,
                'psd_relative_tolerance': 1e-10, 'error_bound': False} if self.numeric else {}


def _report(operation, value, trust, checks, details, witness=None):
    accepted = trust == 'exact' and bool(checks) and all(checks.values())
    return checked_result(operation, value, method=operation+'_identities', trust=trust,
        checks=checks if accepted else {}, candidate=not accepted, witness=witness,
        residual=details.get('relative_residuals') or None, precision=53 if trust == 'numeric' else None,
        details={'accepted': accepted, 'checks': checks, 'checks_are_rigorous': trust == 'exact', **details})


def _discrete(system):
    if system.time_domain != 'discrete':
        raise ValueError('this operation requires a discrete-time system')


def _trust(system, mode, input_trust, entries=()):
    return cap_trust(_system_trust(system), arithmetic_trust(entries, input_trust),
                     'numeric' if mode == 'numeric' else 'exact')


def finite_lqr(system, system_id, Q, R, terminal, horizon, *, mode='exact', input_trust='exact'):
    _discrete(system)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')
    a, b, q, r = _data(system, Q, R, False)
    _square(terminal, a.rows)
    ar = _Arithmetic(mode)
    A, B, Qn, Rn, F = map(ar.matrix, (a, b, q, r, terminal))
    if not ar.psd(Qn) or not ar.psd(Rn, True) or not ar.psd(F):
        raise ValueError('Q/terminal must be PSD and R positive definite')
    values, gains = [None]*(horizon+1), [None]*horizon
    values[-1] = F
    for t in range(horizon-1, -1, -1):
        P = values[t+1]
        S = ar.symmetric(Rn+B.T@P@B)
        K = ar.solve(S, B.T@P@A)
        # Candidate generation uses the subtractive recurrence. Verification below
        # uses the closed-loop completion-of-squares identity independently.
        values[t] = ar.symmetric(Qn+A.T@P@A-A.T@P@B@K)
        gains[t] = ar.clean(K)
    trust = _trust(system, mode, input_trust, (*q, *r, *sp.Matrix(terminal)))
    policy = FiniteHorizonLQR(system_id=system_id, Q=_matrix_tuple(q), R=_matrix_tuple(r),
        terminal=terminal, gains=tuple(ar.pack(K) for K in gains), values=tuple(ar.pack(P) for P in values),
        backend=mode, input_trust=trust)
    report = verify_policy(policy, system, operation='finite_lqr')
    return report, {'output': policy} if all(report.details['checks'].values()) else None


def verify_policy(policy, system, *, operation='verify'):
    _discrete(system)
    a, b, q, r = _data(system, policy.Q, policy.R, False)
    if len(policy.terminal) != a.rows:
        raise ValueError('policy and system dimensions differ')
    ar = _Arithmetic(policy.backend)
    A, B, Q, R, F = map(ar.matrix, (a, b, q, r, policy.terminal))
    P = tuple(map(ar.matrix, policy.values)); K = tuple(map(ar.matrix, policy.gains))
    checks = {'Q_psd': ar.psd(Q), 'R_pd': ar.psd(R, True), 'terminal_psd': ar.psd(F),
              'terminal_identity': ar.equal(P[-1], F)}
    for t, gain in enumerate(K):
        S = ar.symmetric(R+B.T@P[t+1]@B)
        closed = A-B@gain
        checks[f'stage_{t}_S_pd'] = ar.psd(S, True)
        checks[f'stage_{t}_gain'] = ar.equal(S@gain, B.T@P[t+1]@A)
        checks[f'stage_{t}_bellman'] = ar.equal(P[t], Q+gain.T@R@gain+closed.T@P[t+1]@closed)
        checks[f'stage_{t}_value_psd'] = ar.psd(P[t])
    trust = _trust(system, policy.backend, policy.input_trust,
                   (v for M in (policy.Q, policy.R, policy.terminal, *policy.values, *policy.gains) for row in M for v in row))
    return _report(operation, policy.values[0], trust, checks,
        {'horizon': len(K), 'claim_scope': 'unconstrained discrete finite-horizon quadratic optimal control',
         'cost': 'sum(t=0..N-1, x[t].T*Q*x[t]+u[t].T*R*u[t])+x[N].T*terminal*x[N]',
         'control_law': 'u[t]=-gains[t]*x[t]', 'stability_claimed': False,
         'backend': policy.backend, **ar.diagnostics()}, witness={'gains': policy.gains, 'values': policy.values})


def policy_control(policy, state, step, *, input_trust='exact'):
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < len(policy.gains):
        raise ValueError('step must index a control stage')
    if len(state) != len(policy.Q):
        raise ValueError('state dimension mismatch')
    ar = _Arithmetic(policy.backend)
    x, K, P = map(ar.matrix, (state, policy.gains[step], policy.values[step]))
    u = ar.clean(-K@x)
    trust = cap_trust(policy.input_trust, arithmetic_trust(state, input_trust), 'numeric' if ar.numeric else 'exact')
    return _report('control', tuple(row[0] for row in ar.pack(u)), trust,
        {'control_identity': ar.equal(u, -K@x)}, {'step': step, **ar.diagnostics(), 'value_to_go': ar.pack(x.T@P@x)[0][0],
        'claim_scope': 'evaluation of the stored policy; optimality belongs to policy verification'})


def rollout(policy, system, initial, *, input_trust='exact'):
    _discrete(system)
    if len(initial) != len(system.A):
        raise ValueError('initial state dimension mismatch')
    ar = _Arithmetic(policy.backend)
    A, B, Q, R, F, x = map(ar.matrix, (system.A, system.B, policy.Q, policy.R, policy.terminal, initial))
    states, controls = [ar.pack(x)], []
    cost = ar.matrix([[sp.S.Zero]])
    for gain in policy.gains:
        u = ar.clean(-ar.matrix(gain)@x)
        cost = ar.clean(cost+x.T@Q@x+u.T@R@u)
        x = ar.clean(A@x+B@u)
        controls.append(ar.pack(u)); states.append(ar.pack(x))
    cost = ar.clean(cost+x.T@F@x)
    x0 = ar.matrix(initial)
    expected = x0.T@ar.matrix(policy.values[0])@x0
    trust = _trust(system, policy.backend, cap_trust(policy.input_trust, input_trust), initial)
    return _report('rollout', states, trust, {'cost_identity': ar.equal(cost, expected)},
        {'controls': controls, **ar.diagnostics(), 'total_cost': ar.pack(cost)[0][0], 'value_to_go': ar.pack(expected)[0][0],
         'claim_scope': 'stored policy trajectory and telescoping cost identity'})


_NOISE = {'noise_assumptions': ['zero-mean uncorrelated white process and measurement noise',
    'known deterministic control', 'W in state coordinates; covariances per sample'],
    'claim_scope': 'Kalman mean/covariance algebra under the stated model',
    'gaussianity_verified': False, 'model_validated': False}


def kalman_state(system, system_id, mean, covariance, W, V, *, mode='exact', index=0, input_trust='exact'):
    _discrete(system)
    _data(system, W, V, True)
    if len(mean) != len(system.A):
        raise ValueError('mean dimension mismatch')
    _square(covariance, len(mean))
    ar = _Arithmetic(mode)
    m, P, Wn, Vn = map(ar.matrix, (mean, covariance, W, V))
    checks = {'covariance_psd': ar.psd(P), 'W_psd': ar.psd(Wn), 'V_pd': ar.psd(Vn, True)}
    if not all(checks.values()):
        raise ValueError('covariance/W must be PSD and V positive definite')
    trust = _trust(system, mode, input_trust, (*sp.Matrix(mean), *sp.Matrix(covariance), *sp.Matrix(W), *sp.Matrix(V)))
    state = KalmanState(system_id=system_id, mean=tuple(row[0] for row in ar.pack(m)),
        covariance=ar.pack(P), W=W, V=V, index=index, backend=mode, input_trust=trust)
    return _report('kalman_state', state, trust, checks, {**_NOISE, **ar.diagnostics(), 'phase': 'prior', 'index': index}), {'output': state}


def kalman_step(state, system, operation, *, measurement=(), control=(), input_trust='exact'):
    _discrete(system)
    _data(system, state.W, state.V, True)
    if len(state.mean) != len(system.A):
        raise ValueError('state and system dimensions differ')
    ar = _Arithmetic(state.backend)
    A, B, C, D, m, P, W, V = map(ar.matrix, (system.A, system.B, system.C, system.D,
                                           state.mean, state.covariance, state.W, state.V))
    checks = {'input_covariance_psd': ar.psd(P), 'W_psd': ar.psd(W), 'V_pd': ar.psd(V, True)}
    details = dict(_NOISE)
    if operation == 'update':
        if state.phase != 'prior':
            raise ValueError('update requires a prior state; predict after the last update')
        if len(measurement) != C.shape[0] or len(control) != B.shape[1]:
            raise ValueError('measurement/control dimension mismatch')
        y, u = map(ar.matrix, (measurement, control))
        S = ar.symmetric(C@P@C.T+V)
        H = ar.solve(S, C@P).T
        innovation = ar.clean(y-C@m-D@u)
        next_m = ar.clean(m+H@innovation)
        J = ar.eye(A.shape[0])-H@C
        next_P = ar.symmetric(J@P@J.T+H@V@H.T)  # Joseph form, stable under roundoff.
        checks.update(innovation_pd=ar.psd(S, True), gain_identity=ar.equal(S@H.T, C@P),
                      covariance_identity=ar.equal(next_P, P-H@C@P))
        details.update(innovation=ar.pack(innovation), gain=ar.pack(H), innovation_covariance=ar.pack(S),
                       timing='condition x[k] on y[k]=C*x[k]+D*u[k]+v[k]; predict reuses this u[k]')
        updates = {'phase': 'posterior', 'pending_control': tuple(control)}
    elif operation == 'predict':
        if state.phase != 'posterior':
            raise ValueError('predict requires a posterior state; update first')
        if len(state.pending_control) != B.shape[1]:
            raise ValueError('stored control dimension mismatch')
        u = ar.matrix(state.pending_control)
        next_m = ar.clean(A@m+B@u)
        next_P = ar.symmetric(A@P@A.T+W)
        checks['prediction_covariance_identity'] = ar.equal(next_P-W, A@P@A.T)
        updates = {'phase': 'prior', 'index': state.index+1, 'pending_control': ()}
        details['timing'] = 'predict x[k+1]=A*x[k]+B*u[k]+w[k] after conditioning on y[k]'
    else:
        raise ValueError('unknown Kalman state operation')
    checks['output_covariance_psd'] = ar.psd(next_P)
    trust = _trust(system, state.backend, cap_trust(state.input_trust, input_trust), (*measurement, *control))
    new = KalmanState(**{**state.model_dump(mode='python'), **updates,
        'mean': tuple(row[0] for row in ar.pack(next_m)), 'covariance': ar.pack(next_P), 'input_trust': trust})
    details.update(phase=new.phase, index=new.index, **ar.diagnostics())
    report = _report(operation, new, trust, checks, details)
    return report, {'output': new} if all(checks.values()) else None


def _isolated_lqg(system, Q, R, W, V, *, mode='exact', input_trust='exact',
        lqr_certificate=None, kalman_certificate=None, lqr_trust='exact', kalman_trust='exact', max_exact_order=8,
        time_limit=30.0):
    if (lqr_certificate is None) != (kalman_certificate is None):
        raise ValueError('supply both Riccati certificate IDs or neither')
    regulator, ro = synthesize(system, Q, R, mode=mode, input_trust=input_trust,
        certificate=lqr_certificate, certificate_trust=lqr_trust, max_exact_order=max_exact_order,
        time_limit=time_limit/2, _isolate=False)
    estimator, eo = synthesize(system, W, V, kalman=True, mode=mode, input_trust=input_trust,
        certificate=kalman_certificate, certificate_trust=kalman_trust, max_exact_order=max_exact_order,
        time_limit=time_limit/2, _isolate=False)
    trust = cap_trust(regulator.trust, estimator.trust)
    outputs = {'lqr_certificate': ro['certificate'], 'kalman_certificate': eo['certificate']}
    details = {'lqr': regulator.details, 'kalman': estimator.details,
        'claim_scope': 'internally stabilizing regulator/predictor composition and separation identity',
        'stochastic_optimality_claimed': False,
        'controller_input': 'measured y', 'controller_output': 'u=-K*xhat',
        'closed_loop_coordinates': '[x,xhat]', 'closed_loop_input': '[process noise w, measurement noise v]',
        'closed_loop_output': 'plant state x', 'separated_coordinates': '[x,e=x-xhat]'}
    checks = {'regulator_available': 'output' in ro, 'estimator_available': 'output' in eo}
    if not all(checks.values()):
        return _report('lqg', None, trust, checks, details), outputs
    ar = _Arithmetic('exact' if trust == 'exact' else 'numeric')
    A, B, C, D, K, L = map(ar.matrix, (system.A, system.B, system.C, system.D,
                                      regulator.details['gain'], estimator.details['gain']))
    n, m, p = len(system.A), len(system.D[0]), len(system.C)
    # Use symbolic block assembly only at the object boundary.
    sm = lambda M: sp.Matrix(ar.pack(M))
    Ac, BK, LC = sm(A-B@K-L@C+L@D@K), sm(B@K), sm(L@C)
    As = sm(A); Ks = sm(K); Ls = sm(L)
    augmented = As.row_join(-BK).col_join(LC.row_join(As-BK-LC))
    T = sp.eye(n).row_join(sp.zeros(n)).col_join(sp.eye(n).row_join(-sp.eye(n)))
    separated = (As-BK).row_join(BK).col_join(sp.zeros(n).row_join(As-LC))
    checks['separation_identity'] = ar.equal(ar.matrix(T*augmented*T), ar.matrix(separated))
    checks['feedthrough_cancellation'] = ar.equal(ar.matrix(Ac-Ls*sp.Matrix(system.D)*Ks), ar.matrix(As-BK-LC))
    inputs, measurements = channel_units(system)
    states = system.state_units or ('',)*n
    if all(checks.values()):
        outputs['output'] = _copy_system(system, Ac, Ls, -Ks, sp.zeros(m, p),
            input_units=measurements, output_units=inputs, input_unit='', output_unit='', input_trust=trust)
        outputs['closed_loop'] = _copy_system(system, augmented,
            sp.eye(n).row_join(sp.zeros(n,p)).col_join(sp.zeros(n).row_join(Ls)),
            sp.eye(n).row_join(sp.zeros(n)), sp.zeros(n,n+p), state_units=(*states,*states),
            input_units=(*(tuple(f'{unit}/s' if unit else 's^-1' for unit in states) if system.time_domain == 'continuous' else states),*measurements), output_units=states, input_unit='', output_unit='', input_trust=trust)
    details.update(**ar.diagnostics(), K=ar.pack(K), L=ar.pack(L), separated_A=_matrix_tuple(separated),
                   controller_stability_claimed=False)
    return _report('lqg', _matrix_tuple(Ac), trust, checks, details), outputs


def lqg(system, Q, R, W, V, *, mode='exact', input_trust='exact',
        lqr_certificate=None, kalman_certificate=None, lqr_trust='exact', kalman_trust='exact', max_exact_order=8,
        time_limit=30.0):
    options = dict(mode=mode, input_trust=input_trust, lqr_certificate=lqr_certificate,
        kalman_certificate=kalman_certificate, lqr_trust=lqr_trust, kalman_trust=kalman_trust,
        max_exact_order=max_exact_order, time_limit=time_limit*.9)
    if mode != 'numeric' or lqr_certificate is not None or kalman_certificate is not None:
        return _isolated_lqg(system,Q,R,W,V,**options)
    from .engines import run_in_subprocess
    output = run_in_subprocess(_isolated_lqg, time_limit, system,Q,R,W,V,**options)
    output[0].details['process_isolation'] = 'fresh_interpreter_hard_killable'
    return output
