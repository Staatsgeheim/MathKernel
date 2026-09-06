# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed exact local differential geometry with checked Levi-Civita identities.

Public text is parsed by the adapter's restricted MathIR boundary.  This module
only accepts already parsed SymPy objects and never parses user strings.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from typing import Any, Literal

import sympy as sp
from sympy.core.relational import Relational
from pydantic import model_validator

from .engineering import (EngineeringModel, arithmetic_trust, cap_trust,
                          checked_result, exact_zero, validate_scalars)


class Manifold(EngineeringModel):
    name: str
    dimension: int
    orientable: bool | None = None
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_manifold(self):
        if not self.name.strip():
            raise ValueError("manifold name must be non-empty")
        if self.dimension < 1:
            raise ValueError("manifold dimension must be positive")
        cap_trust(self.input_trust)
        return self


class Chart(EngineeringModel):
    name: str
    manifold_id: str
    manifold_name: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    orientation: Literal[-1, 1] | None = None
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_chart(self):
        if not self.name.strip() or not self.manifold_id:
            raise ValueError("chart name and manifold reference are required")
        if not self.coordinates or any(not isinstance(x, sp.Symbol) for x in self.coordinates):
            raise ValueError("chart coordinates must be symbols")
        if len(set(self.coordinates)) != len(self.coordinates):
            raise ValueError("chart coordinates must be unique")
        if any(not isinstance(condition, Relational) for condition in self.domain):
            raise ValueError("chart domain conditions must be parsed mathematical relations")
        cap_trust(self.input_trust)
        return self


class Metric(EngineeringModel):
    name: str = "g"
    chart_id: str
    manifold_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    components: tuple[tuple[sp.Expr, ...], ...]
    signature: tuple[int, int] | None = None
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_metric(self):
        n = len(self.coordinates)
        if not n or len(self.components) != n or any(len(row) != n for row in self.components):
            raise ValueError("metric must be square with the chart dimension")
        values = tuple(x for row in self.components for x in row)
        validate_scalars(values)
        if any(value.is_real is False for value in values):
            raise ValueError("metric components must be real-valued on the chart")
        for i in range(n):
            for j in range(i):
                if exact_zero(self.components[i][j] - self.components[j][i]) is not True:
                    raise ValueError("metric must be explicitly symmetric")
        if exact_zero(sp.det(sp.Matrix(self.components))) is True:
            raise ValueError("metric is degenerate")
        if (self.signature is not None and
                (len(self.signature) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                                  for x in self.signature)
                 or sum(self.signature) != n)):
            raise ValueError("metric signature must be (positive, negative) and sum to the dimension")
        cap_trust(self.input_trust)
        return self


class GeometryTensor(EngineeringModel):
    kind: Literal["inverse_metric", "riemann", "ricci", "einstein"]
    chart_id: str
    metric_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    variance: tuple[Literal["up", "down"], ...]
    components: tuple[Any, ...]
    convention: str
    input_trust: str = "symbolic"


class Connection(EngineeringModel):
    chart_id: str
    metric_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    coefficients: tuple[tuple[tuple[sp.Expr, ...], ...], ...]
    convention: str = "Gamma^rho_(mu nu); torsion-free Levi-Civita connection"
    input_trust: str = "symbolic"


class GeodesicSystem(EngineeringModel):
    chart_id: str
    metric_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    velocity_symbols: tuple[sp.Symbol, ...]
    accelerations: tuple[sp.Expr, ...]
    parameter_convention: str = "affine parameter; q''^rho = acceleration[rho]"
    input_trust: str = "symbolic"


class CoordinateMap(EngineeringModel):
    name: str = "F"
    source_chart_id: str
    target_chart_id: str
    manifold_id: str
    source_coordinates: tuple[sp.Symbol, ...]
    target_coordinates: tuple[sp.Symbol, ...]
    forward: tuple[sp.Expr, ...]
    inverse: tuple[sp.Expr, ...] | None = None
    source_domain: tuple[Any, ...] = ()
    target_domain: tuple[Any, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_map(self):
        if not self.name.strip():
            raise ValueError("coordinate-map name must be non-empty")
        if len(self.forward) != len(self.target_coordinates):
            raise ValueError("forward map must define every target coordinate")
        if self.inverse is not None and len(self.inverse) != len(self.source_coordinates):
            raise ValueError("inverse map must define every source coordinate")
        validate_scalars(self.forward)
        if self.inverse is not None:
            validate_scalars(self.inverse)
        cap_trust(self.input_trust)
        return self


class JacobianMap(EngineeringModel):
    map_id: str
    source_chart_id: str
    target_chart_id: str
    source_coordinates: tuple[sp.Symbol, ...]
    target_coordinates: tuple[sp.Symbol, ...]
    components: tuple[tuple[sp.Expr, ...], ...]
    determinant: sp.Expr | None = None
    convention: str = "J[a,i] = d(target coordinate a)/d(source coordinate i)"
    input_trust: str = "symbolic"


class TensorField(EngineeringModel):
    name: str = "T"
    chart_id: str
    manifold_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    variance: tuple[Literal["up", "down"], ...]
    components: Any
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_tensor(self):
        n = len(self.coordinates)
        def check(value, depth):
            if depth == 0:
                if isinstance(value, (tuple, list)):
                    raise ValueError("tensor component rank does not match variance")
                validate_scalars((value,))
                return
            if not isinstance(value, (tuple, list)) or len(value) != n:
                raise ValueError("every tensor axis must equal the chart dimension")
            for item in value:
                check(item, depth - 1)
        check(self.components, len(self.variance))
        cap_trust(self.input_trust)
        return self


class FormTerm(EngineeringModel):
    indices: tuple[int, ...]
    coefficient: sp.Expr


class DifferentialForm(EngineeringModel):
    name: str = "omega"
    chart_id: str
    manifold_id: str
    coordinates: tuple[sp.Symbol, ...]
    domain: tuple[Any, ...] = ()
    degree: int
    terms: tuple[FormTerm, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_form(self):
        n = len(self.coordinates)
        if self.degree < 0 or (self.degree > n and self.terms):
            raise ValueError("nonzero form degree must be within the chart dimension")
        seen = set()
        for term in self.terms:
            if len(term.indices) != self.degree:
                raise ValueError("form term index count must equal its degree")
            if tuple(sorted(term.indices)) != term.indices or len(set(term.indices)) != len(term.indices):
                raise ValueError("form indices must be strictly increasing")
            if any(index < 0 or index >= n for index in term.indices):
                raise ValueError("form index is outside the coordinate range")
            if term.indices in seen:
                raise ValueError("form terms must have unique index tuples")
            seen.add(term.indices)
            validate_scalars((term.coefficient,))
        cap_trust(self.input_trust)
        return self


def _simplify(value):
    # Apply FU identities before and after rational normalization.  Calling the
    # generic simplifier last can expand a trigonometric zero back into an
    # undecidable quotient (notably in polar/spherical charts).
    value = sp.trigsimp(value, method="fu")
    value = sp.cancel(sp.together(value))
    return sp.factor(sp.trigsimp(value, method="fu"))


def _all_zero(values):
    states = [exact_zero(_simplify(value)) for value in values]
    if any(state is False for state in states):
        return False
    if all(state is True for state in states):
        return True
    return None


@lru_cache(maxsize=64)
def _inverse_cached(components):
    matrix = sp.Matrix(components)
    inverse = matrix.inv()
    return tuple(tuple(_simplify(inverse[i, j]) for j in range(matrix.rows))
                 for i in range(matrix.rows))


@lru_cache(maxsize=64)
def _connection_cached(coordinates, components):
    n = len(coordinates)
    inverse = _inverse_cached(components)
    gamma = []
    for rho in range(n):
        upper = []
        for mu in range(n):
            row = []
            for nu in range(n):
                value = sum(inverse[rho][sigma] * (
                    sp.diff(components[sigma][nu], coordinates[mu])
                    + sp.diff(components[sigma][mu], coordinates[nu])
                    - sp.diff(components[mu][nu], coordinates[sigma]))
                    for sigma in range(n)) / 2
                row.append(_simplify(value))
            upper.append(tuple(row))
        gamma.append(tuple(upper))
    return tuple(gamma)


@lru_cache(maxsize=32)
def _curvature_cached(coordinates, components):
    n = len(coordinates)
    inverse = _inverse_cached(components)
    gamma = _connection_cached(coordinates, components)
    riemann = []
    for rho in range(n):
        first = []
        for sigma in range(n):
            second = []
            for mu in range(n):
                row = []
                for nu in range(n):
                    value = (sp.diff(gamma[rho][nu][sigma], coordinates[mu])
                             - sp.diff(gamma[rho][mu][sigma], coordinates[nu]))
                    value += sum(gamma[rho][mu][lam] * gamma[lam][nu][sigma]
                                 - gamma[rho][nu][lam] * gamma[lam][mu][sigma]
                                 for lam in range(n))
                    row.append(_simplify(value))
                second.append(tuple(row))
            first.append(tuple(second))
        riemann.append(tuple(first))
    riemann = tuple(riemann)
    ricci = tuple(tuple(_simplify(sum(riemann[rho][sigma][rho][nu]
                                       for rho in range(n)))
                        for nu in range(n)) for sigma in range(n))
    scalar = _simplify(sum(inverse[i][j] * ricci[i][j]
                           for i in range(n) for j in range(n)))
    einstein = tuple(tuple(_simplify(ricci[i][j] - components[i][j] * scalar / 2)
                              for j in range(n)) for i in range(n))
    return riemann, ricci, scalar, einstein


@lru_cache(maxsize=128)
def _jacobian_cached(source_coordinates, forward):
    return tuple(tuple(_simplify(sp.diff(expression, coordinate))
                       for coordinate in source_coordinates)
                 for expression in forward)


def _trust(metric):
    return arithmetic_trust(
        (value for row in metric.components for value in row), metric.input_trust)


def _condition(metric):
    determinant = _simplify(sp.det(sp.Matrix(metric.components)))
    conditions = [sp.sstr(condition) for condition in metric.domain]
    if determinant.is_zero is not False:
        conditions.append(sp.sstr(sp.Ne(determinant, 0)))
    return determinant, tuple(dict.fromkeys(conditions))


def inverse_metric(metric, metric_id):
    n = len(metric.coordinates)
    inverse = _inverse_cached(metric.components)
    product = sp.Matrix(metric.components) * sp.Matrix(inverse)
    checks = {"left_inverse": _all_zero(product[i, j] - (1 if i == j else 0)
                                          for i in range(n) for j in range(n))}
    determinant, conditions = _condition(metric)
    trust = _trust(metric)
    tensor = GeometryTensor(kind="inverse_metric", chart_id=metric.chart_id,
        metric_id=metric_id, coordinates=metric.coordinates, variance=("up", "up"),
        domain=metric.domain, components=inverse,
        convention="g^(i j), coordinate basis", input_trust=trust)
    result = checked_result("inverse_metric", inverse, method="matrix_inverse_identity",
        trust=trust, checks=checks, conditions=conditions,
        details={"determinant": determinant, "identity_checks": checks,
                 "coordinate_order": metric.coordinates})
    return result, tensor


def christoffel(metric, metric_id):
    n = len(metric.coordinates)
    gamma = _connection_cached(metric.coordinates, metric.components)
    torsion_free = _all_zero(gamma[rho][mu][nu] - gamma[rho][nu][mu]
        for rho in range(n) for mu in range(n) for nu in range(n))
    compatibility = _all_zero(
        sp.diff(metric.components[i][j], metric.coordinates[k])
        - sum(gamma[ell][k][i] * metric.components[ell][j]
              + gamma[ell][k][j] * metric.components[i][ell] for ell in range(n))
        for k in range(n) for i in range(n) for j in range(n))
    checks = {"torsion_free": torsion_free, "metric_compatible": compatibility}
    _, conditions = _condition(metric)
    trust = _trust(metric)
    connection = Connection(chart_id=metric.chart_id, metric_id=metric_id,
        coordinates=metric.coordinates, domain=metric.domain,
        coefficients=gamma, input_trust=trust)
    result = checked_result("christoffel", gamma, method="levi_civita_formula",
        trust=trust, checks=checks, conditions=conditions,
        details={"identity_checks": checks, "coordinate_order": metric.coordinates})
    return result, connection


def _curvature_checks(metric, riemann):
    n = len(metric.coordinates)
    antisymmetry = _all_zero(riemann[rho][sigma][mu][nu]
        + riemann[rho][sigma][nu][mu]
        for rho in range(n) for sigma in range(n) for mu in range(n) for nu in range(n))
    first_bianchi = _all_zero(riemann[rho][sigma][mu][nu]
        + riemann[rho][mu][nu][sigma] + riemann[rho][nu][sigma][mu]
        for rho in range(n) for sigma in range(n) for mu in range(n) for nu in range(n))
    lowered = [[[[ _simplify(sum(metric.components[a][rho] * riemann[rho][b][c][d]
                            for rho in range(n)))
                    for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
    lowered_symmetries = _all_zero(
        value for a in range(n) for b in range(n) for c in range(n) for d in range(n)
        for value in (lowered[a][b][c][d] + lowered[b][a][c][d],
                      lowered[a][b][c][d] + lowered[a][b][d][c],
                      lowered[a][b][c][d] - lowered[c][d][a][b]))
    return {"last_pair_antisymmetry": antisymmetry,
            "first_bianchi": first_bianchi,
            "lowered_riemann_symmetries": lowered_symmetries}


def curvature(metric, metric_id, operation):
    riemann, ricci, scalar, einstein = _curvature_cached(
        metric.coordinates, metric.components)
    checks = _curvature_checks(metric, riemann)
    checks["ricci_symmetric"] = _all_zero(
        ricci[i][j] - ricci[j][i] for i in range(len(ricci)) for j in range(len(ricci)))
    if operation == "einstein":
        n = len(metric.coordinates)
        inverse = _inverse_cached(metric.components)
        gamma = _connection_cached(metric.coordinates, metric.components)
        checks["contracted_bianchi"] = _all_zero(
            sum(inverse[i][k] * (
                sp.diff(einstein[i][j], metric.coordinates[k])
                - sum(gamma[ell][k][i] * einstein[ell][j]
                      + gamma[ell][k][j] * einstein[i][ell]
                      for ell in range(n)))
                for i in range(n) for k in range(n))
            for j in range(n))
    _, conditions = _condition(metric)
    trust = _trust(metric)
    common = {"identity_checks": checks, "coordinate_order": metric.coordinates,
              "riemann_convention": "R^rho_(sigma mu nu)"}
    if operation == "scalar_curvature":
        return checked_result(operation, scalar, method="levi_civita_curvature",
            trust=trust, checks=checks, conditions=conditions, details=common)
    values = {"riemann": riemann, "ricci": ricci, "einstein": einstein}
    variances = {"riemann": ("up", "down", "down", "down"),
                 "ricci": ("down", "down"), "einstein": ("down", "down")}
    tensor = GeometryTensor(kind=operation, chart_id=metric.chart_id,
        metric_id=metric_id, coordinates=metric.coordinates,
        domain=metric.domain, variance=variances[operation], components=values[operation],
        convention=common["riemann_convention"], input_trust=trust)
    result = checked_result(operation, values[operation], method="levi_civita_curvature",
        trust=trust, checks=checks, conditions=conditions, details=common)
    return result, tensor


def geodesic_equations(metric, metric_id):
    gamma = _connection_cached(metric.coordinates, metric.components)
    velocities = tuple(sp.Symbol(f"d_{coordinate.name}", real=True) for coordinate in metric.coordinates)
    accelerations = tuple(_simplify(-sum(gamma[rho][mu][nu] * velocities[mu] * velocities[nu]
        for mu in range(len(velocities)) for nu in range(len(velocities))))
        for rho in range(len(velocities)))
    checks = {"connection_lower_indices_symmetric": _all_zero(
        gamma[rho][mu][nu] - gamma[rho][nu][mu]
        for rho in range(len(velocities)) for mu in range(len(velocities))
        for nu in range(len(velocities)))}
    _, conditions = _condition(metric)
    trust = _trust(metric)
    system = GeodesicSystem(chart_id=metric.chart_id, metric_id=metric_id,
        coordinates=metric.coordinates, velocity_symbols=velocities,
        domain=metric.domain, accelerations=accelerations, input_trust=trust)
    result = checked_result("geodesic_equations", accelerations,
        method="affine_geodesic_equation", trust=trust, checks=checks,
        conditions=conditions, details={"velocity_symbols": velocities,
            "equation_convention": system.parameter_convention})
    return result, system


def _permutation_sign(values):
    if len(set(values)) != len(values):
        return 0
    return -1 if sum(values[i] > values[j]
                     for i in range(len(values)) for j in range(i + 1, len(values))) % 2 else 1


def _form_dict(form):
    return {term.indices: term.coefficient for term in form.terms}


def _form_component(terms, indices):
    sign = _permutation_sign(indices)
    return sp.S.Zero if sign == 0 else sign * terms.get(tuple(sorted(indices)), sp.S.Zero)


def _make_form(source, *, name, degree, terms, trust=None, chart_id=None,
               coordinates=None, domain=None, manifold_id=None):
    canonical = []
    for indices, coefficient in sorted(terms.items()):
        value = _simplify(coefficient)
        if exact_zero(value) is not True:
            canonical.append(FormTerm(indices=indices, coefficient=value))
    return DifferentialForm(name=name, chart_id=chart_id or source.chart_id,
        manifold_id=manifold_id or source.manifold_id,
        coordinates=coordinates or source.coordinates,
        domain=source.domain if domain is None else domain, degree=degree,
        terms=tuple(canonical), input_trust=trust or source.input_trust)


def coordinate_jacobian(mapping, map_id):
    matrix = _jacobian_cached(mapping.source_coordinates, mapping.forward)
    determinant = (_simplify(sp.det(sp.Matrix(matrix)))
                   if len(mapping.source_coordinates) == len(mapping.target_coordinates) else None)
    checks = {"jacobian_nonsingular": None if determinant is None else determinant.is_zero is False}
    conditions = [sp.sstr(condition) for condition in mapping.source_domain]
    if determinant is not None and determinant.is_zero is not False:
        conditions.append(sp.sstr(sp.Ne(determinant, 0)))
    jacobian = JacobianMap(map_id=map_id,
        source_chart_id=mapping.source_chart_id, target_chart_id=mapping.target_chart_id,
        source_coordinates=mapping.source_coordinates,
        target_coordinates=mapping.target_coordinates, components=matrix,
        determinant=determinant, input_trust=mapping.input_trust)
    result = checked_result("jacobian", matrix, method="symbolic_differentiation",
        trust=mapping.input_trust, checks=checks,
        conditions=tuple(dict.fromkeys(conditions)),
        details={"determinant": determinant, "identity_checks": checks,
                 "row_coordinates": mapping.target_coordinates,
                 "column_coordinates": mapping.source_coordinates})
    return result, jacobian


def verify_coordinate_map(mapping):
    jacobian = sp.Matrix(_jacobian_cached(mapping.source_coordinates, mapping.forward))
    determinant = _simplify(jacobian.det())
    checks = {"jacobian_nonsingular": (True if determinant.is_zero is False
                                        else False if determinant.is_zero is True else None)}
    if mapping.inverse is not None:
        forward_sub = dict(zip(mapping.target_coordinates, mapping.forward))
        inverse_sub = dict(zip(mapping.source_coordinates, mapping.inverse))
        checks["inverse_after_forward"] = _all_zero(
            expression.xreplace(forward_sub) - coordinate
            for expression, coordinate in zip(mapping.inverse, mapping.source_coordinates))
        checks["forward_after_inverse"] = _all_zero(
            expression.xreplace(inverse_sub) - coordinate
            for expression, coordinate in zip(mapping.forward, mapping.target_coordinates))
    conditions = [sp.sstr(condition) for condition in mapping.source_domain]
    if determinant.is_zero is not False:
        conditions.append(sp.sstr(sp.Ne(determinant, 0)))
    return checked_result("verify", checks, method="coordinate_composition_and_jacobian",
        trust=mapping.input_trust, checks=checks,
        conditions=tuple(dict.fromkeys(conditions)),
        details={"determinant": determinant, "identity_checks": checks})


def _tensor_get(components, indices):
    value = components
    for index in indices:
        value = value[index]
    return value


def _tensor_build(n, rank, function, prefix=()):
    if rank == 0:
        return _simplify(function(prefix))
    return tuple(_tensor_build(n, rank - 1, function, prefix + (index,))
                 for index in range(n))


def covariant_derivative(tensor, tensor_id, metric, metric_id):
    n, rank = len(tensor.coordinates), len(tensor.variance)
    gamma = _connection_cached(metric.coordinates, metric.components)
    def component(indices):
        base, derivative_index = indices[:-1], indices[-1]
        value = sp.diff(_tensor_get(tensor.components, base), tensor.coordinates[derivative_index])
        for slot, variance in enumerate(tensor.variance):
            original = base[slot]
            for contracted in range(n):
                changed = base[:slot] + (contracted,) + base[slot + 1:]
                if variance == "up":
                    value += gamma[original][derivative_index][contracted] * _tensor_get(tensor.components, changed)
                else:
                    value -= gamma[contracted][derivative_index][original] * _tensor_get(tensor.components, changed)
        return value
    components = _tensor_build(n, rank + 1, component)
    trust = cap_trust(tensor.input_trust, metric.input_trust, "symbolic")
    derived = TensorField(name=f"nabla({tensor.name})", chart_id=tensor.chart_id,
        manifold_id=tensor.manifold_id, coordinates=tensor.coordinates,
        domain=tensor.domain, variance=tensor.variance + ("down",),
        components=components, input_trust=trust)
    checks = {"connection_metric_compatible": _all_zero(
        sp.diff(metric.components[i][j], metric.coordinates[k])
        - sum(gamma[ell][k][i] * metric.components[ell][j]
              + gamma[ell][k][j] * metric.components[i][ell] for ell in range(n))
        for k in range(n) for i in range(n) for j in range(n))}
    return checked_result("covariant_derivative", components,
        method="levi_civita_coordinate_formula", trust=trust, checks=checks,
        conditions=_condition(metric)[1], details={"identity_checks": checks,
        "variance": derived.variance, "metric_id": metric_id}), derived


def lie_derivative(tensor, tensor_id, vector, vector_id):
    n, rank = len(tensor.coordinates), len(tensor.variance)
    def component(indices):
        value = sum(_tensor_get(vector.components, (c,))
                    * sp.diff(_tensor_get(tensor.components, indices), tensor.coordinates[c])
                    for c in range(n))
        for slot, variance in enumerate(tensor.variance):
            original = indices[slot]
            for c in range(n):
                changed = indices[:slot] + (c,) + indices[slot + 1:]
                derivative = (sp.diff(_tensor_get(vector.components, (original,)), tensor.coordinates[c])
                              if variance == "up" else
                              sp.diff(_tensor_get(vector.components, (c,)), tensor.coordinates[original]))
                value += (-1 if variance == "up" else 1) * derivative * _tensor_get(tensor.components, changed)
        return value
    components = _tensor_build(n, rank, component)
    trust = cap_trust(tensor.input_trust, vector.input_trust, "symbolic")
    derived = TensorField(name=f"L_{vector.name}({tensor.name})", chart_id=tensor.chart_id,
        manifold_id=tensor.manifold_id, coordinates=tensor.coordinates,
        domain=tensor.domain, variance=tensor.variance,
        components=components, input_trust=trust)
    checks = {"vector_field_type": vector.variance == ("up",)}
    return checked_result("lie_derivative", components,
        method="coordinate_lie_derivative", trust=trust, checks=checks,
        details={"identity_checks": checks, "vector_field_id": vector_id}), derived


def wedge(left, left_id, right, right_id):
    degree = left.degree + right.degree
    if degree > len(left.coordinates):
        terms = {}
    else:
        terms = {}
        for left_indices, a in _form_dict(left).items():
            for right_indices, b in _form_dict(right).items():
                joined = left_indices + right_indices
                sign = _permutation_sign(joined)
                if sign:
                    key = tuple(sorted(joined))
                    terms[key] = terms.get(key, sp.S.Zero) + sign * a * b
    trust = cap_trust(left.input_trust, right.input_trust, "symbolic")
    derived = _make_form(left, name=f"{left.name} wedge {right.name}",
                         degree=degree, terms=terms, trust=trust)
    reverse = {}
    for ri, b in _form_dict(right).items():
        for li, a in _form_dict(left).items():
            sign = _permutation_sign(ri + li)
            if sign:
                key = tuple(sorted(ri + li))
                reverse[key] = reverse.get(key, sp.S.Zero) + sign * b * a
    checks = {"graded_commutativity": _all_zero(
        terms.get(key, 0) - (-1)**(left.degree * right.degree) * reverse.get(key, 0)
        for key in set(terms) | set(reverse))}
    return checked_result("wedge", tuple((term.indices, term.coefficient) for term in derived.terms),
        method="antisymmetric_shuffle_product", trust=trust, checks=checks,
        details={"identity_checks": checks, "left_id": left_id, "right_id": right_id}), derived


def _exterior_terms(form):
    out = {}
    for indices, coefficient in _form_dict(form).items():
        for j, coordinate in enumerate(form.coordinates):
            joined = (j,) + indices
            sign = _permutation_sign(joined)
            if sign:
                key = tuple(sorted(joined))
                out[key] = out.get(key, sp.S.Zero) + sign * sp.diff(coefficient, coordinate)
    return out


def exterior_derivative(form, form_id):
    terms = _exterior_terms(form) if form.degree < len(form.coordinates) else {}
    derived = _make_form(form, name=f"d({form.name})", degree=form.degree + 1,
                         terms=terms, trust=form.input_trust)
    second = _exterior_terms(derived) if derived.degree < len(form.coordinates) else {}
    checks = {"d_squared_zero": _all_zero(second.values())}
    return checked_result("exterior_derivative",
        tuple((term.indices, term.coefficient) for term in derived.terms),
        method="coordinate_exterior_derivative", trust=form.input_trust,
        checks=checks, details={"identity_checks": checks, "source_form_id": form_id}), derived


def interior_product(form, form_id, vector, vector_id):
    if form.degree == 0:
        raise ValueError("interior product is undefined for a zero-form")
    source = _form_dict(form)
    terms = {}
    for remaining in combinations(range(len(form.coordinates)), form.degree - 1):
        terms[remaining] = sum(_tensor_get(vector.components, (i,))
                               * _form_component(source, (i,) + remaining)
                               for i in range(len(form.coordinates)))
    trust = cap_trust(form.input_trust, vector.input_trust, "symbolic")
    derived = _make_form(form, name=f"i_{vector.name}({form.name})",
                         degree=form.degree - 1, terms=terms, trust=trust)
    checks = {"degree_reduced_by_one": derived.degree == form.degree - 1,
              "vector_field_type": vector.variance == ("up",)}
    return checked_result("interior_product",
        tuple((term.indices, term.coefficient) for term in derived.terms),
        method="coordinate_contraction", trust=trust, checks=checks,
        details={"identity_checks": checks, "form_id": form_id,
                 "vector_field_id": vector_id}), derived


def pullback_form(form, form_id, mapping, map_id):
    source_n = len(mapping.source_coordinates)
    jacobian = sp.Matrix(_jacobian_cached(mapping.source_coordinates, mapping.forward))
    substitutions = dict(zip(mapping.target_coordinates, mapping.forward))
    source_terms = _form_dict(form)
    terms = {}
    for columns in combinations(range(source_n), form.degree):
        value = sp.S.Zero
        for rows, coefficient in source_terms.items():
            minor = jacobian.extract(rows, columns).det() if rows else sp.S.One
            value += coefficient.xreplace(substitutions) * minor
        terms[columns] = value
    trust = cap_trust(form.input_trust, mapping.input_trust, "symbolic")
    derived = _make_form(form, name=f"{mapping.name}*({form.name})",
        degree=form.degree, terms=terms, trust=trust,
        chart_id=mapping.source_chart_id, coordinates=mapping.source_coordinates,
        domain=mapping.source_domain, manifold_id=mapping.manifold_id)
    left = _exterior_terms(derived) if derived.degree < source_n else {}
    right_form = None
    if form.degree < len(form.coordinates):
        dform = _make_form(form, name=f"d({form.name})", degree=form.degree + 1,
                           terms=_exterior_terms(form), trust=form.input_trust)
        right_form, _ = _pullback_terms_only(dform, mapping, trust)
    right = _form_dict(right_form) if right_form is not None else {}
    checks = {"pullback_commutes_with_d": _all_zero(
        left.get(key, 0) - right.get(key, 0) for key in set(left) | set(right))}
    return checked_result("pullback",
        tuple((term.indices, term.coefficient) for term in derived.terms),
        method="jacobian_minor_pullback", trust=trust, checks=checks,
        conditions=tuple(sp.sstr(c) for c in mapping.source_domain),
        details={"identity_checks": checks, "form_id": form_id, "map_id": map_id}), derived


def _pullback_terms_only(form, mapping, trust):
    jacobian = sp.Matrix(_jacobian_cached(mapping.source_coordinates, mapping.forward))
    substitutions = dict(zip(mapping.target_coordinates, mapping.forward))
    terms = {}
    for columns in combinations(range(len(mapping.source_coordinates)), form.degree):
        terms[columns] = sum(coefficient.xreplace(substitutions)
            * (jacobian.extract(rows, columns).det() if rows else sp.S.One)
            for rows, coefficient in _form_dict(form).items())
    return _make_form(form, name=f"{mapping.name}*({form.name})", degree=form.degree,
        terms=terms, trust=trust, chart_id=mapping.source_chart_id,
        coordinates=mapping.source_coordinates, domain=mapping.source_domain,
        manifold_id=mapping.manifold_id), terms


def _hodge_terms(form, metric, orientation):
    n, k = len(form.coordinates), form.degree
    inverse = _inverse_cached(metric.components)
    determinant = _simplify(sp.det(sp.Matrix(metric.components)))
    volume = sp.sqrt(sp.Abs(determinant))
    source = _form_dict(form)
    raised = {}
    for upper in combinations(range(n), k):
        value = sp.S.Zero
        for lower in combinations(range(n), k):
            value += sp.det(sp.Matrix([[inverse[i][j] for j in lower] for i in upper])) * source.get(lower, 0)
        raised[upper] = _simplify(value)
    terms = {}
    for complement in combinations(range(n), n - k):
        terms[complement] = orientation * volume * sum(
            _permutation_sign(upper + complement) * value
            for upper, value in raised.items())
    return terms


def hodge_star(form, form_id, metric, metric_id, orientation):
    n, k = len(form.coordinates), form.degree
    if k > n:
        raise ValueError("Hodge star is only defined through the chart dimension")
    determinant = _simplify(sp.det(sp.Matrix(metric.components)))
    terms = _hodge_terms(form, metric, orientation)
    trust = cap_trust(form.input_trust, metric.input_trust, "symbolic")
    derived = _make_form(form, name=f"star({form.name})", degree=n-k,
                         terms=terms, trust=trust)
    nondegenerate = (True if determinant.is_zero is False
                     else False if determinant.is_zero is True else None)
    checks = {"metric_nondegenerate": nondegenerate,
              "orientation_explicit": orientation in (-1, 1)}
    conditions = list(_condition(metric)[1])
    if metric.signature is None:
        checks["double_star"] = None
        conditions.append("metric signature required to certify the Hodge double-star sign")
    else:
        twice = _hodge_terms(derived, metric, orientation)
        expected_sign = (-1) ** (k * (n - k) + metric.signature[1])
        original = _form_dict(form)
        checks["double_star"] = _all_zero(
            twice.get(indices, 0) - expected_sign * original.get(indices, 0)
            for indices in set(twice) | set(original))
    return checked_result("hodge_star",
        tuple((term.indices, term.coefficient) for term in derived.terms),
        method="metric_volume_and_raised_components", trust=trust, checks=checks,
        conditions=tuple(dict.fromkeys(conditions)),
        details={"identity_checks": checks, "form_id": form_id,
                 "metric_id": metric_id, "orientation": orientation,
                 "metric_determinant": determinant,
                 "metric_signature": metric.signature}), derived


def cache_info():
    return {"inverse": _inverse_cached.cache_info()._asdict(),
            "connection": _connection_cached.cache_info()._asdict(),
            "curvature": _curvature_cached.cache_info()._asdict(),
            "jacobian": _jacobian_cached.cache_info()._asdict()}


def clear_caches():
    _inverse_cached.cache_clear()
    _connection_cached.cache_clear()
    _curvature_cached.cache_clear()
    _jacobian_cached.cache_clear()
