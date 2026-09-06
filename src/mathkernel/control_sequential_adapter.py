# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded MathIR boundary for derived control policies and estimator states."""
from . import control_sequential as seq

OPERATIONS = {
    'StateSpaceSystem': {
        'lqg': {**{key: 'MathIR[][]' for key in ('Q','R','W','V')},
                'lqr_certificate_id': 'RiccatiCertificate object id?',
                'kalman_certificate_id': 'RiccatiCertificate object id?'},
        'finite_lqr': {**{key: 'MathIR[][]' for key in ('Q','R','terminal')}, 'horizon': 'nonnegative integer'},
        'kalman_state': {**{key: 'MathIR[][]' for key in ('covariance','W','V')},
                         'mean': 'MathIR[]', 'index': 'nonnegative integer'},
    },
    'FiniteHorizonLQR': {'verify': {}, 'control': {'state': 'MathIR[]', 'step': 'zero-based integer'},
                         'rollout': {'initial': 'MathIR[]'}},
    'KalmanState': {'update': {'measurement': 'MathIR[]', 'control': 'MathIR[]'}, 'predict': {}},
}


def _index(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f'{name} must be an integer in [0,{maximum}]')
    return value


def apply(kernel, value, object_type, operation, p, reference, object_id):
    from .engineering_adapter import _parse_tree
    s = kernel.settings
    system = value
    if object_type != 'StateSpaceSystem':
        record = kernel._get_math_object_record(value.system_id)
        if record is None or record['object_type'] != 'StateSpaceSystem':
            raise ValueError('source state-space system is unavailable')
        system = record['value']
    mode = p.get('mode', 'exact') if object_type == 'StateSpaceSystem' else value.backend
    n, m, channels = len(system.A), len(system.B[0]), len(system.C)
    order = max(n,m,channels)
    if order > s.max_control_order or (mode == 'exact' and n > s.max_exact_control_order):
        raise ValueError('sequential operation exceeds control order limit')
    horizon = (_index(p.get('horizon'), s.max_control_horizon, 'horizon') if operation == 'finite_lqr'
               else len(value.gains) if object_type == 'FiniteHorizonLQR' else 1)
    if horizon > s.max_control_horizon or (mode == 'exact' and horizon > s.max_exact_control_horizon):
        raise ValueError('horizon exceeds configured control horizon limit')
    if max(1,horizon)*order**3 > s.max_engineering_work:
        raise ValueError('sequential control exceeds max_engineering_work')
    if operation == 'lqg' and max(2*n,n+channels) > s.max_control_order:
        raise ValueError('augmented LQG output exceeds max_control_order')
    schema = OPERATIONS[object_type][operation]
    fields = {key: p[key] for key, kind in schema.items() if kind.startswith('MathIR')}
    for key, item in fields.items():
        if not isinstance(item, (list,tuple)) or len(item) > s.max_control_order:
            raise ValueError(f'{key} exceeds control dimension limit')
        if '[][]' in schema[key] and any(not isinstance(row,(list,tuple)) or len(row)>s.max_control_order for row in item):
            raise ValueError(f'{key} exceeds control dimension limit')
    parsed, trust = _parse_tree(kernel, fields, p.get('context_id'))
    if operation == 'finite_lqr':
        return seq.finite_lqr(system, object_id, **parsed, horizon=horizon, mode=mode, input_trust=trust.value)
    if operation == 'kalman_state':
        return seq.kalman_state(system, object_id, **parsed, mode=mode,
            index=_index(p.get('index',0),2**53-1,'index'), input_trust=trust.value)
    if object_type == 'KalmanState':
        if operation == 'predict' and value.index >= 2**53-1:
            raise ValueError('Kalman sample index limit reached')
        return seq.kalman_step(value, system, operation, **parsed, input_trust=trust.value)
    if operation == 'verify':
        return seq.verify_policy(value, system)
    if operation == 'control':
        return seq.policy_control(value, **parsed,
            step=_index(p.get('step'), horizon-1, 'step'), input_trust=trust.value)
    if operation == 'rollout':
        return seq.rollout(value, system, **parsed, input_trust=trust.value)
    kwargs = {}
    for prefix in ('lqr','kalman'):
        key = prefix+'_certificate_id'
        if key in p:
            kwargs[prefix+'_certificate'] = reference(key, 'RiccatiCertificate')
            kwargs[prefix+'_trust'] = kernel._get_math_object_record(str(p[key]))['input_trust'].value
    return seq.lqg(system, **parsed, mode=mode, input_trust=trust.value,
                   max_exact_order=s.max_exact_control_order,
                   time_limit=s.solver_timeout_seconds*.9, **kwargs)
