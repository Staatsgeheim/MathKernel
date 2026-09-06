# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted MathIR boundary and resource gates for conic/QCQP objects."""
from .engineering import cap_trust, positive_integer
from .models import TrustLevel

TYPES = {
    'conicproblem': 'ConicProblem', 'conic_problem': 'ConicProblem',
    'coniccertificate': 'ConicCertificate', 'conic_certificate': 'ConicCertificate',
    'quadraticallyconstrainedproblem': 'QuadraticallyConstrainedProblem', 'qcqp': 'QuadraticallyConstrainedProblem',
    'quadraticcertificate': 'QuadraticCertificate', 'quadratic_certificate': 'QuadraticCertificate',
}
def witness_schema(*fields):
    return {'type': 'object', 'properties': {name: 'MathIR[]' for name in fields}}

SOLVE = {'max_iterations': 'positive integer', 'tolerance': 'positive finite float',
         'reconstruction_denominator': 'positive integer'}
OPERATIONS = {
    'ConicProblem': {'solve': SOLVE, 'verify_certificate': {'certificate_id': 'ConicCertificate object id?', 'certificate': witness_schema('primal', 'cone_dual', 'equality_dual', 'ray')}},
    'QuadraticallyConstrainedProblem': {'solve': {**SOLVE, 'initial': 'MathIR[] starting point?'},
        'verify_certificate': {'certificate_id': 'QuadraticCertificate object id?', 'certificate': witness_schema('primal', 'inequality_dual', 'equality_dual', 'quadratic_dual')}},
}


def construct(kernel, kind, definition):
    from .engineering_adapter import _parse_tree
    from .conic import ConeBlock, ConicProblem, ConicCertificate
    from .quadratic_constraints import QuadraticConstraint, QuadraticallyConstrainedProblem, QuadraticCertificate
    typ = TYPES[kind];data = dict(definition);s = kernel.settings
    context = data.pop('context_id', None)
    certificate = typ.endswith('Certificate')
    if certificate and 'input_trust' in data:
        raise ValueError('certificate trust is derived from its entries')
    def vector(name, maximum):
        if not isinstance(data.get(name, ()), (list, tuple)) or len(data.get(name, ())) > maximum:
            raise ValueError(f'{name} exceeds configured optimization dimensions')
    def matrix(value, rows, columns):
        if not isinstance(value, (list, tuple)) or len(value) > rows or any(not isinstance(row, (list, tuple)) or len(row) > columns for row in value):
            raise ValueError('matrix exceeds configured optimization dimensions')
    if certificate:
        fields = ('primal', 'cone_dual', 'equality_dual', 'ray') if typ == 'ConicCertificate' else ('primal', 'inequality_dual', 'equality_dual', 'quadratic_dual')
        for name in fields:
            vector(name, s.max_optimization_constraints+2*s.max_optimization_variables)
    else:
        vector('c', s.max_optimization_variables)
        vector('variables', s.max_optimization_variables)
        n = len(data.get('c', ()))
        for name in ('A', 'A_eq', 'A_ub', 'Q'):
            if name in data:
                matrix(data[name], s.max_optimization_constraints, s.max_optimization_variables)
        for name in ('b', 'b_eq', 'b_ub', 'lower', 'upper'):
            vector(name, s.max_optimization_constraints+2*s.max_optimization_variables)
        if typ == 'ConicProblem':
            raw = data.get('cones', ())
            if not isinstance(raw, (list, tuple)) or len(raw) > s.max_optimization_constraints:
                raise ValueError('too many cone blocks')
            blocks = tuple(ConeBlock(**block) for block in raw)
            if sum(block.size for block in blocks) > s.max_optimization_constraints:
                raise ValueError('cone rows exceed max_optimization_constraints')
            if any(block.kind == 'psd' and block.dimension > s.max_psd_cone_order for block in blocks):
                raise ValueError('PSD block exceeds max_psd_cone_order')
            data['cones'] = blocks
            fields = ('c', 'A', 'b', 'A_eq', 'b_eq')
        else:
            quadratics = data.get('quadratics', ())
            if not isinstance(quadratics, (list, tuple)) or len(quadratics) > s.max_quadratic_constraints:
                raise ValueError('quadratics exceed max_quadratic_constraints')
            if len(quadratics)*(n*n+n+1) > s.max_engineering_work:
                raise ValueError('quadratic coefficient work exceeds max_engineering_work')
            for item in quadratics:
                if not isinstance(item, dict) or set(item)-{'Q', 'a', 'r'}:
                    raise ValueError('quadratic constraint accepts only Q, a and r')
                matrix(item.get('Q', ()), n, n)
                if len(item.get('a', ())) > n:
                    raise ValueError('quadratic linear term exceeds variable dimension')
            fields = ('c', 'Q', 'A_ub', 'b_ub', 'A_eq', 'b_eq', 'lower', 'upper', 'quadratics')
    parsed, trust = _parse_tree(kernel, {key: data[key] for key in fields if key in data}, context)
    data.update(parsed)
    if not certificate:
        trust = TrustLevel(cap_trust(trust.value, data.get('input_trust', 'exact')))
        data['input_trust'] = trust.value
    if typ == 'QuadraticallyConstrainedProblem':
        data['quadratics'] = tuple(QuadraticConstraint(**item) for item in data.get('quadratics', ()))
    cls = {'ConicProblem': ConicProblem, 'ConicCertificate': ConicCertificate,
           'QuadraticallyConstrainedProblem': QuadraticallyConstrainedProblem, 'QuadraticCertificate': QuadraticCertificate}[typ]
    return typ, cls(**data), trust, []


def apply(kernel, problem, object_type, operation, parameters, reference):
    import math
    from .engineering_adapter import _parse_tree
    from . import conic, quadratic_constraints
    p, s = parameters, kernel.settings
    module = conic if object_type == 'ConicProblem' else quadratic_constraints
    if operation == 'verify_certificate':
        if ('certificate_id' in p) == ('certificate' in p):
            raise ValueError('supply exactly one certificate_id or inline certificate')
        name = 'ConicCertificate' if object_type == 'ConicProblem' else 'QuadraticCertificate'
        if 'certificate_id' in p:
            cert = reference('certificate_id', name)
            cert_trust = kernel._get_math_object_record(str(p['certificate_id']))['input_trust']
        else:
            definition = dict(p['certificate'])
            if p.get('context_id') is not None:
                definition.setdefault('context_id', p['context_id'])
            _, cert, cert_trust, _ = construct(kernel, name.lower(), definition)
        return module.verify_certificate(problem, cert, certificate_trust=cert_trust.value)
    if p.get('mode', 'exact') != 'numeric':
        raise NotImplementedError('conic/QCQP search requires explicit numeric mode; exact witnesses use verify_certificate')
    tolerance = p.get('tolerance', 1e-9)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('tolerance must be finite and in (0,1)')
    options = dict(max_iterations=positive_integer(p.get('max_iterations', s.max_iterations), 'max_iterations', s.max_iterations),
        tolerance=tolerance, reconstruction_denominator=positive_integer(p.get('reconstruction_denominator', 1000000), 'reconstruction_denominator', 1000000000))
    if object_type == 'ConicProblem':
        from .conic_search import solve
        return solve(problem, time_limit=s.solver_timeout_seconds*.9, **options)
    if 'initial' in p:
        if not isinstance(p['initial'], (tuple, list)) or len(p['initial']) != len(problem.c):
            raise ValueError('initial point must match objective dimension')
        options['initial'], _ = _parse_tree(kernel, p['initial'], p.get('context_id'))
    return quadratic_constraints.solve(problem, time_limit=s.solver_timeout_seconds*.9, **options)
