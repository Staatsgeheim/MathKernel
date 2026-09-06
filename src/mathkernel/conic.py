# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Self-dual product-cone programs with original-data rational witnesses.

Minimize c'x subject to b-Ax in K and E x=f. Variables are free. PSD blocks
use full row-major symmetric matrices and the trace inner product, never svec
scaling in the public mathematical model. Search is isolated in conic_search.
"""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import StrictInt, model_validator
from .engineering import EngineeringModel, arithmetic_trust, cap_trust, checked_result, validate_scalars
from .optimization import exact_psd


class ConeBlock(EngineeringModel):
    kind: Literal['nonnegative', 'second_order', 'psd']
    dimension: StrictInt

    @model_validator(mode='after')
    def validate_dimension(self):
        if self.dimension < (2 if self.kind == 'second_order' else 1):
            raise ValueError('cone dimension must be positive; second_order needs at least 2')
        return self

    @property
    def size(self):
        return self.dimension**2 if self.kind == 'psd' else self.dimension


class ConicProblem(EngineeringModel):
    variables: tuple[str, ...]
    c: tuple[sp.Expr, ...]
    A: tuple[tuple[sp.Expr, ...], ...] = ()
    b: tuple[sp.Expr, ...] = ()
    cones: tuple[ConeBlock, ...] = ()
    A_eq: tuple[tuple[sp.Expr, ...], ...] = ()
    b_eq: tuple[sp.Expr, ...] = ()
    sense: Literal['min', 'max'] = 'min'
    input_trust: str = 'exact'

    @model_validator(mode='after')
    def validate_problem(self):
        n = len(self.c)
        if not n or len(self.variables) != n or len(set(self.variables)) != n:
            raise ValueError('unique variables must match objective dimension')
        for matrix, rhs in ((self.A, self.b), (self.A_eq, self.b_eq)):
            if len(matrix) != len(rhs) or any(len(row) != n for row in matrix):
                raise ValueError('constraint dimensions do not match objective')
        if sum(block.size for block in self.cones) != len(self.b):
            raise ValueError('cone block sizes must partition all A/b rows')
        entries = tuple(problem_entries(self))
        validate_scalars(entries, real=True)
        if any(v.free_symbols for v in entries):
            raise ValueError('conic data must be concrete finite real coefficients')
        offset = 0
        for block in self.cones:
            if block.kind == 'psd':
                d = block.dimension
                for i in range(d):
                    for j in range(i):
                        p, q = offset+i*d+j, offset+j*d+i
                        if self.b[p] != self.b[q] or self.A[p] != self.A[q]:
                            raise ValueError('every PSD affine coefficient matrix must be explicitly symmetric')
            offset += block.size
        cap_trust(self.input_trust)
        return self


class ConicCertificate(EngineeringModel):
    kind: Literal['optimal', 'bound', 'infeasible', 'unbounded'] = 'optimal'
    primal: tuple[sp.Expr, ...] = ()
    cone_dual: tuple[sp.Expr, ...] = ()
    equality_dual: tuple[sp.Expr, ...] = ()
    ray: tuple[sp.Expr, ...] = ()

    @model_validator(mode='after')
    def validate_witness(self):
        validate_scalars(certificate_entries(self), real=True)
        if any(v.free_symbols for v in certificate_entries(self)):
            raise ValueError('certificate entries must be concrete')
        return self


def problem_entries(problem):
    yield from problem.c
    yield from problem.b
    yield from problem.b_eq
    for matrix in (problem.A, problem.A_eq):
        for row in matrix:
            yield from row


def certificate_entries(certificate):
    return (*certificate.primal, *certificate.cone_dual, *certificate.equality_dual, *certificate.ray)


def matrices(problem):
    n = len(problem.c)
    return (sp.Matrix(problem.c)*(1 if problem.sense == 'min' else -1),
            sp.Matrix(len(problem.b), n, [v for row in problem.A for v in row]),
            sp.Matrix(len(problem.b), 1, problem.b),
            sp.Matrix(len(problem.b_eq), n, [v for row in problem.A_eq for v in row]),
            sp.Matrix(len(problem.b_eq), 1, problem.b_eq))


def cone_membership(values, blocks):
    """Independent exact sign/squared-norm/Schur checks, including boundaries."""
    if len(values) != sum(block.size for block in blocks):
        raise ValueError('cone vector dimension mismatch')
    records = []
    offset = 0
    for block in blocks:
        part = list(values[offset:offset+block.size])
        if block.kind == 'nonnegative':
            ok = all(v.is_nonnegative is True for v in part)
            record = {'nonnegative': ok}
        elif block.kind == 'second_order':
            margin = sp.cancel(part[0]**2-sum(v*v for v in part[1:]))
            ok = part[0].is_nonnegative is True and margin.is_nonnegative is True
            record = {'nonnegative_axis': part[0].is_nonnegative is True, 'squared_margin': margin}
        else:
            matrix = sp.Matrix(block.dimension, block.dimension, part)
            symmetric = matrix == matrix.T
            psd, pivots = exact_psd(matrix) if symmetric else (False, [])
            ok = symmetric and psd is True
            record = {'symmetric': symmetric, 'psd': psd, 'pivots': pivots}
        records.append({'kind': block.kind, 'dimension': block.dimension, 'accepted': ok, **record})
        offset += block.size
    return all(r['accepted'] for r in records), records


def verify_certificate(problem, certificate, *, certificate_trust="exact"):
    c, A, b, E, f = matrices(problem)
    n, m, p = len(problem.c), len(problem.b), len(problem.b_eq)
    kind = certificate.kind
    requires_primal = kind in {'optimal', 'unbounded'}
    requires_dual = kind != 'unbounded'
    if (len(certificate.primal) != (n if requires_primal else 0) or
        len(certificate.cone_dual) != (m if requires_dual else 0) or
        len(certificate.equality_dual) != (p if requires_dual else 0) or
        len(certificate.ray) != (n if kind == 'unbounded' else 0)):
        raise ValueError('certificate fields/dimensions do not match its kind and conic problem')
    details = {}; checks = {}; value = None
    sign = 1 if problem.sense == 'min' else -1
    if requires_primal:
        x = sp.Matrix(certificate.primal)
        slack = b-A*x
        member, records = cone_membership(tuple(slack), problem.cones)
        checks.update(primal_cone=member, primal_equalities=all(v == 0 for v in E*x-f))
        details.update(primal=certificate.primal, slack=tuple(slack), primal_blocks=records)
    if requires_dual:
        z = sp.Matrix(m, 1, certificate.cone_dual)
        y = sp.Matrix(p, 1, certificate.equality_dual)
        member, records = cone_membership(tuple(z), problem.cones)
        stationarity = A.T*z+E.T*y+(c if kind != 'infeasible' else sp.zeros(n, 1))
        checks.update(dual_cone=member, stationarity=all(v == 0 for v in stationarity))
        pairing = (b.T*z)[0]+(f.T*y)[0]
        details.update(dual_blocks=records, stationarity_residual=tuple(stationarity))
        if kind == 'infeasible':
            checks['strict_contradiction'] = pairing.is_negative is True
            details['contradiction'] = pairing
        else:
            value = -sign*pairing
            details['dual_value'] = value
            if kind == 'optimal':
                primal = (c.T*x)[0]
                gap = sp.cancel(primal+pairing)
                checks['zero_duality_gap'] = gap == 0
                checks['complementarity'] = (slack.T*z)[0] == 0
                value = sign*primal
                details.update(primal_value=value, duality_gap=gap)
    if kind == 'unbounded':
        d = sp.Matrix(certificate.ray)
        member, records = cone_membership(tuple(-A*d), problem.cones)
        improvement = (c.T*d)[0]
        checks.update(recession_cone=member, recession_equalities=all(v == 0 for v in E*d),
                      strict_improvement=improvement.is_negative is True)
        details.update(ray_blocks=records, canonical_directional_objective=improvement)
    entries = (*problem_entries(problem), *certificate_entries(certificate))
    trust = cap_trust(arithmetic_trust(entries, problem.input_trust), certificate_trust)
    accepted = all(checks.values()) and trust == 'exact' and all(v.is_Rational for v in entries)
    conclusion = {'optimal': 'certified_global_optimum', 'infeasible': 'certified_infeasible',
                  'unbounded': 'certified_unbounded',
                  'bound': 'certified_lower_bound' if sign == 1 else 'certified_upper_bound'}[kind]
    result = checked_result('verify_certificate', value, method='original_data_product_cone_witness',
        trust=trust, checks={'certificate_accepted': True} if accepted else {},
        witness=certificate.model_dump(mode='json'), candidate=not accepted,
        details={**details, 'accepted': accepted, 'conclusion': conclusion if accepted else 'uncertified_candidate',
                 'certificate_checks': checks, 'certificate': certificate.model_dump(mode='json'),
                 'psd_representation': 'full_row_major_trace_inner_product'})
    if accepted and kind in {'infeasible', 'unbounded'}:
        result.status = kind
    return result


def from_linear_problem(problem, *, max_rows=512):
    from .optimization import canonical, _problem_trust
    Q, c, G, h, E, f = canonical(problem)
    if any(problem.integrality) or any(v != 0 for v in Q):
        raise NotImplementedError('to_conic currently converts continuous linear problems only')
    if G.rows > max_rows:
        raise ValueError('canonical bounds/rows exceed conic max_optimization_constraints')
    # canonical c is normalized; preserve the original objective and sense.
    derived = ConicProblem(variables=problem.variables, c=problem.c, A=tuple(map(tuple, G.tolist())),
        b=tuple(h), A_eq=tuple(map(tuple, E.tolist())), b_eq=tuple(f), sense=problem.sense,
        cones=(ConeBlock(kind='nonnegative', dimension=G.rows),) if G.rows else (),
        input_trust=_problem_trust(problem))
    return checked_result('to_conic', derived, method='canonical_lp_to_nonnegative_cone',
        trust=derived.input_trust, checks={'constraint_equivalence': True, 'objective_identity': True},
        details={'bounds_included': True, 'variables_are_free': True}), derived
