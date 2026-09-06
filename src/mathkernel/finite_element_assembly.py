# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Phase G.4 sparse P1 assembly and replayable algebraic solves."""
from __future__ import annotations

from math import isfinite
from typing import Literal

import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import (ComputationEvidence, EvidenceBundle,
                                  ModelEvidence, NumericalEvidence)

from .engineering import EngineeringModel, EngineeringResult, cap_trust
from .finite_elements import (BasisFunctionSet, FEMMesh, FiniteElementSpace,
                              QuadratureRule)
from .partial_differential_equations import PDEProblem, _expression
from .weak_forms import WeakForm


class SparseMatrixEntry(EngineeringModel):
    row: int
    column: int
    value: sp.Expr

    @model_validator(mode="after")
    def validate_entry(self):
        if self.row < 0 or self.column < 0:
            raise ValueError("sparse entry indices must be nonnegative")
        _expression(self.value, "sparse entry")
        if self.value.free_symbols:
            raise ValueError("assembled sparse entries must be concrete")
        return self


class LocalElementContribution(EngineeringModel):
    cell_index: int
    global_dofs: tuple[int, ...]
    matrix: tuple[tuple[sp.Expr, ...], ...]
    vector: tuple[sp.Expr, ...]
    jacobian_determinant: sp.Expr
    quadrature_exact: bool

    @model_validator(mode="after")
    def validate_local(self):
        size = len(self.global_dofs)
        if (self.cell_index < 0 or not size or len(self.matrix) != size or
                any(len(row) != size for row in self.matrix) or
                len(self.vector) != size):
            raise ValueError("local element contribution shape is invalid")
        for value in (*self.vector, self.jacobian_determinant,
                      *(value for row in self.matrix for value in row)):
            _expression(value, "local contribution")
            if value.free_symbols:
                raise ValueError("local contributions must be concrete")
        return self


class NaturalBoundaryContribution(EngineeringModel):
    boundary_condition_index: int
    facet: tuple[int, ...]
    cell_index: int
    global_dofs: tuple[int, ...]
    matrix: tuple[tuple[sp.Expr, ...], ...]
    vector: tuple[sp.Expr, ...]
    quadrature_exact: bool

    @model_validator(mode="after")
    def validate_boundary(self):
        size = len(self.global_dofs)
        if (self.boundary_condition_index < 0 or self.cell_index < 0 or
                not self.facet or len(self.matrix) != size or
                any(len(row) != size for row in self.matrix) or
                len(self.vector) != size):
            raise ValueError("natural-boundary contribution shape is invalid")
        for value in (*self.vector, *(value for row in self.matrix for value in row)):
            _expression(value, "natural-boundary contribution")
            if value.free_symbols:
                raise ValueError("natural-boundary contributions must be concrete")
        return self


class EssentialConstraint(EngineeringModel):
    dof: int
    value: sp.Expr
    boundary_condition_indices: tuple[int, ...]

    @model_validator(mode="after")
    def validate_constraint(self):
        if self.dof < 0 or not self.boundary_condition_indices:
            raise ValueError("essential constraint provenance is invalid")
        _expression(self.value, "essential value")
        if self.value.free_symbols:
            raise ValueError("essential values must be concrete")
        return self


class AssembledSystem(EngineeringModel):
    finite_element_space_id: str
    weak_form_id: str
    mesh_id: str
    basis_id: str
    quadrature_id: str
    field: str
    substitutions: tuple[tuple[str, sp.Expr], ...]
    dof_count: int
    local_contributions: tuple[LocalElementContribution, ...]
    natural_contributions: tuple[NaturalBoundaryContribution, ...]
    raw_matrix_entries: tuple[SparseMatrixEntry, ...]
    raw_rhs: tuple[sp.Expr, ...]
    essential_constraints: tuple[EssentialConstraint, ...]
    matrix_entries: tuple[SparseMatrixEntry, ...]
    rhs: tuple[sp.Expr, ...]
    boundary_method: Literal["symmetric_elimination"] = "symmetric_elimination"
    quadrature_exact: bool
    raw_matrix_symmetric: bool
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_system(self):
        if not all((self.finite_element_space_id, self.weak_form_id,
                    self.mesh_id, self.basis_id, self.quadrature_id)):
            raise ValueError("assembled-system source provenance is invalid")
        if self.dof_count < 1 or len(self.raw_rhs) != self.dof_count or len(self.rhs) != self.dof_count:
            raise ValueError("assembled-system vector shape is invalid")
        for entries in (self.raw_matrix_entries, self.matrix_entries):
            if any(entry.row >= self.dof_count or entry.column >= self.dof_count
                   for entry in entries):
                raise ValueError("sparse entry lies outside the matrix")
            if len({(entry.row, entry.column) for entry in entries}) != len(entries):
                raise ValueError("sparse entries must be coalesced")
        if len({name for name, _ in self.substitutions}) != len(self.substitutions):
            raise ValueError("assembly substitutions must be unique")
        for _, value in self.substitutions:
            _expression(value, "assembly substitution")
            if value.free_symbols:
                raise ValueError("assembly substitutions must be concrete")
        for value in (*self.raw_rhs, *self.rhs):
            _expression(value, "assembled vector entry")
            if value.free_symbols:
                raise ValueError("assembled vectors must be concrete")
        return self


class FEMSolution(EngineeringModel):
    assembled_system_id: str
    solve_status: Literal[
        "unique", "ill_conditioned", "singular_inconsistent",
        "singular_underdetermined", "singular_least_squares",
    ]
    method: Literal["exact", "numeric"]
    values: tuple[sp.Expr, ...]
    residual: tuple[sp.Expr, ...]
    residual_norm: sp.Expr | None
    rank: int | None
    augmented_rank: int | None
    condition_number: float | None
    tolerance: float
    condition_limit: float
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_solution(self):
        if not self.assembled_system_id:
            raise ValueError("solution requires assembled-system provenance")
        if self.rank is not None and self.rank < 0:
            raise ValueError("matrix rank must be nonnegative")
        if self.augmented_rank is not None and self.augmented_rank < 0:
            raise ValueError("augmented rank must be nonnegative")
        if (not isfinite(self.tolerance) or self.tolerance <= 0 or
                not isfinite(self.condition_limit) or self.condition_limit <= 1):
            raise ValueError("solver thresholds must be finite and positive")
        if self.condition_number is not None and (
                not isfinite(self.condition_number) or self.condition_number < 0):
            raise ValueError("finite condition diagnostics are required")
        for value in (*self.values, *self.residual):
            _expression(value, "solution value")
            if value.free_symbols:
                raise ValueError("solution values and residuals must be concrete")
        if self.residual_norm is not None:
            _expression(self.residual_norm, "residual norm")
        return self


def _bundle(method: str, trust: str, diagnostics=(), *, residual=None):
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(engine="finite_element_assembly",
            method=method, arithmetic=trust, deterministic=True, trust=trust)],
        model=[ModelEvidence(role="diagnostic", trust="unknown",
            assumptions=[], diagnostics=[*diagnostics,
                "assembled algebraic evidence does not prove a continuous PDE solution",
                "existence, uniqueness, regularity, well-posedness, and continuum error are not established"],
            metadata={"continuous_solution": "not_claimed"})],
        justified_trust=trust)
    if residual is not None:
        bundle.numerical.append(NumericalEvidence(
            precision=53, residual=str(residual), trust="numeric",
            metadata={"error_bound": False}))
    return bundle


def _coalesce(values: dict[tuple[int, int], sp.Expr]):
    return tuple(SparseMatrixEntry(row=row, column=column, value=reduced)
                 for (row, column), value in sorted(values.items())
                 if (reduced := sp.simplify(value)) != 0)


def _entry_dict(entries):
    return {(entry.row, entry.column): entry.value for entry in entries}


def _substitute(expr, substitutions, coordinate_map):
    value = sp.simplify(expr.subs(substitutions).subs(coordinate_map))
    if value.free_symbols:
        raise ValueError(f"assembly expression retains unresolved symbols: {sorted(map(str, value.free_symbols))[0]}")
    _expression(value, "assembled scalar")
    return value


def _derivative_factor(derivative, basis_index, basis_values, gradients,
                       problem_variables, integration_variables):
    orders = {name: derivative[index] for index, name in enumerate(problem_variables)}
    if any(order for name, order in orders.items() if name not in integration_variables):
        raise ValueError("G.4 stationary assembly does not support derivatives outside the integration variables")
    active = [(integration_variables.index(name), order) for name, order in orders.items() if order]
    if not active:
        return basis_values[basis_index]
    if len(active) != 1 or active[0][1] != 1:
        raise ValueError("G.4 P1 assembly supports only zero- or first-order weak factors")
    return gradients[basis_index][active[0][0]]


def _quadrature_exact(expression, symbols, degree):
    try:
        polynomial = sp.Poly(sp.expand(expression), *symbols)
    except sp.PolynomialError:
        return False
    return polynomial.total_degree() <= degree


def _integrate_reference(expression, symbols, point_maps, weights,
                         reference_measure, degree):
    """Apply the declared rule, with an exact constant-integrand shortcut."""
    exact = _quadrature_exact(expression, symbols, degree)
    if not expression.free_symbols:
        return sp.simplify(expression * reference_measure), exact
    return sp.simplify(sum(
        weight * expression.subs(point_map)
        for point_map, weight in zip(point_maps, weights)
    )), exact


def _facet_rule(dimension, degree):
    if dimension == 1:
        return ((),), (sp.S.One,)
    if dimension == 2:
        if degree == 1:
            return ((sp.Rational(1, 2),),), (sp.S.One,)
        offset = sp.sqrt(3) / 6
        return ((sp.Rational(1, 2) - offset,),
                (sp.Rational(1, 2) + offset,)), (sp.Rational(1, 2),) * 2
    if degree == 1:
        return ((sp.Rational(1, 3), sp.Rational(1, 3)),), (sp.Rational(1, 2),)
    return ((sp.Rational(1, 6), sp.Rational(1, 6)),
            (sp.Rational(2, 3), sp.Rational(1, 6)),
            (sp.Rational(1, 6), sp.Rational(2, 3))), (sp.Rational(1, 6),) * 3


def _facet_reference_point(vertices, parameter):
    if len(vertices[0]) == 1:
        return vertices[0]
    if len(vertices[0]) == 2:
        t = parameter[0]
        return tuple(sp.simplify((1-t)*vertices[0][axis] + t*vertices[1][axis])
                     for axis in range(2))
    s, t = parameter
    return tuple(sp.simplify(vertices[0][axis] +
        s*(vertices[1][axis]-vertices[0][axis]) +
        t*(vertices[2][axis]-vertices[0][axis])) for axis in range(3))


def _facet_measure(points):
    if len(points[0]) == 1:
        return sp.S.One
    if len(points[0]) == 2:
        return sp.sqrt(sum((points[1][axis]-points[0][axis])**2 for axis in range(2)))
    left = sp.Matrix([points[1][axis]-points[0][axis] for axis in range(3)])
    right = sp.Matrix([points[2][axis]-points[0][axis] for axis in range(3)])
    return sp.sqrt(left.cross(right).dot(left.cross(right)))


def _condition_facets(problem, mesh, condition):
    axis = problem.independent_variables.index(condition.coordinate)
    bounds = {name: (lower, upper) for name, lower, upper in problem.domain}
    target = bounds[condition.coordinate][condition.side == "upper"]
    return {facet for facet in mesh.boundary_facets if all(
        sp.simplify(mesh.points[index][axis] - target) == 0 for index in facet)}


def _essential_constraints(problem, weak, mesh, space, substitutions):
    by_dof: dict[int, tuple[sp.Expr, list[int]]] = {}
    variables = tuple(sp.Symbol(name) for name in problem.independent_variables)
    for boundary_index in weak.essential_boundary_indices:
        condition = problem.boundary_conditions[boundary_index]
        if condition.field != space.field:
            continue
        axis = problem.independent_variables.index(condition.coordinate)
        bounds = {name: (lower, upper) for name, lower, upper in problem.domain}
        target = bounds[condition.coordinate][condition.side == "upper"]
        for dof, point in enumerate(mesh.points):
            if sp.simplify(point[axis] - target) != 0:
                continue
            coordinate_map = dict(zip(variables, point))
            value = _substitute(condition.value, substitutions, coordinate_map)
            if dof in by_dof:
                previous, indices = by_dof[dof]
                equal = sp.simplify(previous - value)
                if equal != 0:
                    raise ValueError("conflicting essential values meet at one finite-element DOF")
                indices.append(boundary_index)
            else:
                by_dof[dof] = (value, [boundary_index])
    if set(by_dof) != set(space.essential_dofs):
        raise ValueError("essential DOFs do not reconcile with the weak-form boundary partition")
    return tuple(EssentialConstraint(dof=dof, value=value,
        boundary_condition_indices=tuple(indices))
        for dof, (value, indices) in sorted(by_dof.items()))


def assemble_system(problem: PDEProblem, weak: WeakForm, mesh: FEMMesh,
                    space: FiniteElementSpace, space_id: str,
                    basis: BasisFunctionSet, quadrature: QuadratureRule,
                    quadrature_id: str, substitutions: tuple[tuple[str, sp.Expr], ...],
                    trust: str):
    if len(weak.trial_spaces) != 1 or weak.trial_spaces[0].field != space.field:
        raise ValueError("G.4 assembly supports one scalar trial field")
    if (mesh.weak_form_id != space.weak_form_id or
            basis.reference_element_id != space.reference_element_id or
            quadrature.reference_element_id != space.reference_element_id):
        raise ValueError("finite-element assembly source ancestry is inconsistent")
    if weak.periodic_boundary_indices:
        raise ValueError("G.4 assembly does not yet support periodic constraints")
    if basis.polynomial_degree != 1 or space.family != "P1_lagrange":
        raise ValueError("G.4 assembly requires the P1 Lagrange space")
    if quadrature.cell_type != mesh.cell_type or quadrature.dimension != mesh.dimension:
        raise ValueError("quadrature and mesh cell types do not match")
    substitution_map = {sp.Symbol(name): value for name, value in substitutions}
    xi = tuple(sp.Symbol(name) for name in basis.coordinate_names)
    xi_set = set(xi)
    quadrature_point_maps = tuple(
        dict(zip(xi, point)) for point in quadrature.points)
    physical_symbols = tuple(sp.Symbol(name) for name in problem.independent_variables)
    if tuple(weak.integration_variables) != tuple(problem.independent_variables):
        raise ValueError("G.4 stationary assembly requires all PDE variables to be integrated")
    size = len(basis.functions)
    prepared_volume_terms = []
    needs_physical_map = False
    physical_symbol_set = set(physical_symbols)
    for term in weak.volume_terms:
        coefficient = sp.simplify(term.coefficient.subs(substitution_map))
        unresolved = coefficient.free_symbols - physical_symbol_set
        if unresolved:
            raise ValueError(f"assembly coefficient retains unresolved symbol: {sorted(map(str, unresolved))[0]}")
        coordinate_dependent = bool(coefficient.free_symbols & physical_symbol_set)
        needs_physical_map |= coordinate_dependent
        prepared_volume_terms.append((term, coefficient, coordinate_dependent))
    raw_matrix: dict[tuple[int, int], sp.Expr] = {}
    raw_rhs = [sp.S.Zero] * space.dof_count
    local_outputs = []
    all_exact = True

    for cell_index, cell in enumerate(mesh.cells):
        physical = [mesh.points[index] for index in cell]
        base = sp.Matrix(physical[0])
        jacobian = sp.Matrix.hstack(*(sp.Matrix(point)-base for point in physical[1:]))
        determinant = sp.simplify(jacobian.det())
        inverse_transpose = jacobian.inv().T
        gradients = tuple(tuple(sp.simplify(value) for value in inverse_transpose * sp.Matrix(row))
                          for row in basis.gradients)
        coordinate_map = {}
        if needs_physical_map:
            physical_map = tuple(base[axis] + sum(
                jacobian[axis, column] * xi[column] for column in range(mesh.dimension))
                for axis in range(mesh.dimension))
            coordinate_map = dict(zip(physical_symbols, physical_map))
        local_matrix = [[sp.S.Zero for _ in range(size)] for _ in range(size)]
        local_vector = [sp.S.Zero for _ in range(size)]
        cell_exact = True
        for term, prepared_coefficient, coordinate_dependent in prepared_volume_terms:
            coefficient = (sp.simplify(prepared_coefficient.subs(coordinate_map))
                           if coordinate_dependent else prepared_coefficient)
            unresolved = coefficient.free_symbols - xi_set
            if unresolved:
                raise ValueError(f"assembly coefficient retains unresolved symbol: {sorted(map(str, unresolved))[0]}")
            for test_index in range(size):
                test_factor = _derivative_factor(term.test_derivative, test_index,
                    basis.functions, gradients, problem.independent_variables,
                    weak.integration_variables)
                if term.side == "rhs" and term.field is None:
                    integrand = sp.simplify(coefficient * test_factor * determinant)
                    integral, integral_exact = _integrate_reference(
                        integrand, xi, quadrature_point_maps, quadrature.weights,
                        quadrature.reference_measure, quadrature.degree_exact)
                    cell_exact &= integral_exact
                    local_vector[test_index] += integral
                    continue
                if term.side != "lhs" or term.field != space.field or term.field_power != 1:
                    raise ValueError("G.4 assembly supports scalar linear LHS terms and field-free RHS terms")
                for trial_index in range(size):
                    trial_factor = _derivative_factor(term.field_derivative, trial_index,
                        basis.functions, gradients, problem.independent_variables,
                        weak.integration_variables)
                    integrand = sp.simplify(coefficient * trial_factor * test_factor * determinant)
                    integral, integral_exact = _integrate_reference(
                        integrand, xi, quadrature_point_maps, quadrature.weights,
                        quadrature.reference_measure, quadrature.degree_exact)
                    cell_exact &= integral_exact
                    local_matrix[test_index][trial_index] += integral
        matrix_tuple = tuple(tuple(sp.simplify(value) for value in row) for row in local_matrix)
        vector_tuple = tuple(sp.simplify(value) for value in local_vector)
        local_outputs.append(LocalElementContribution(cell_index=cell_index,
            global_dofs=cell, matrix=matrix_tuple, vector=vector_tuple,
            jacobian_determinant=determinant, quadrature_exact=cell_exact))
        all_exact &= cell_exact
        for local_row, global_row in enumerate(cell):
            raw_rhs[global_row] += vector_tuple[local_row]
            for local_column, global_column in enumerate(cell):
                raw_matrix[(global_row, global_column)] = raw_matrix.get(
                    (global_row, global_column), sp.S.Zero) + matrix_tuple[local_row][local_column]

    natural_outputs = []
    owner_by_facet = dict(zip(mesh.boundary_facets, mesh.boundary_facet_owners))
    for term in weak.boundary_terms:
        if term.vanishes_by_trace:
            continue
        if term.boundary_condition_index is None:
            raise ValueError("unresolved weak boundary flux prevents assembly")
        boundary_index = term.boundary_condition_index
        condition = problem.boundary_conditions[boundary_index]
        if condition.kind not in {"neumann", "robin"}:
            raise ValueError("unsupported nonessential boundary condition")
        if condition.kind == "robin" and sp.simplify(condition.beta.subs(substitution_map)) == 0:
            raise ValueError("Robin assembly requires nonzero beta")
        for facet in sorted(_condition_facets(problem, mesh, condition)):
            cell_index, local_facet, _ = owner_by_facet[facet]
            cell = mesh.cells[cell_index]
            ref_vertices = tuple(tuple(sp.S.Zero if vertex == 0 else
                sp.S.One if axis == vertex-1 else sp.S.Zero for axis in range(mesh.dimension))
                for vertex in range(mesh.dimension+1))
            face_ref_vertices = tuple(ref_vertices[index] for index in range(len(cell))
                                      if index != local_facet)
            face_physical = tuple(mesh.points[cell[index]] for index in range(len(cell))
                                  if index != local_facet)
            parameters, weights = _facet_rule(mesh.dimension, quadrature.degree_exact)
            measure = _facet_measure(face_physical)
            local_matrix = [[sp.S.Zero for _ in range(size)] for _ in range(size)]
            local_vector = [sp.S.Zero for _ in range(size)]
            face_exact = True
            for parameter, weight in zip(parameters, weights):
                point = _facet_reference_point(face_ref_vertices, parameter)
                basis_values = tuple(function.subs(dict(zip(xi, point))) for function in basis.functions)
                physical_point = tuple(sum(basis_values[index] * mesh.points[cell[index]][axis]
                                           for index in range(size))
                                       for axis in range(mesh.dimension))
                coordinate_map = dict(zip(physical_symbols, physical_point))
                coefficient = _substitute(
                    term.coefficient * term.measure.outward_orientation,
                    substitution_map, coordinate_map)
                value = _substitute(condition.value, substitution_map, coordinate_map)
                if condition.kind == "neumann":
                    for row in range(size):
                        local_vector[row] += -weight * measure * coefficient * value * basis_values[row]
                else:
                    alpha = _substitute(condition.alpha, substitution_map, coordinate_map)
                    beta = _substitute(condition.beta, substitution_map, coordinate_map)
                    if beta == 0:
                        raise ValueError("Robin beta evaluates to zero on a boundary facet")
                    for row in range(size):
                        local_vector[row] += -weight * measure * coefficient * value / beta * basis_values[row]
                        for column in range(size):
                            local_matrix[row][column] += (-weight * measure * coefficient * alpha /
                                beta * basis_values[row] * basis_values[column])
            # Facet exactness follows the rule degree for constant represented data;
            # nonconstant data is conservatively marked non-exact.
            face_symbols = set(condition.value.free_symbols | condition.alpha.free_symbols |
                               condition.beta.free_symbols | term.coefficient.free_symbols)
            required_degree = 2 if condition.kind == "robin" else 1
            face_exact = (quadrature.degree_exact >= required_degree and
                          face_symbols <= {sp.Symbol(condition.coordinate)} | set(substitution_map))
            matrix_tuple = tuple(tuple(sp.simplify(value) for value in row) for row in local_matrix)
            vector_tuple = tuple(sp.simplify(value) for value in local_vector)
            natural_outputs.append(NaturalBoundaryContribution(
                boundary_condition_index=boundary_index, facet=facet,
                cell_index=cell_index, global_dofs=cell, matrix=matrix_tuple,
                vector=vector_tuple, quadrature_exact=face_exact))
            all_exact &= face_exact
            for local_row, global_row in enumerate(cell):
                raw_rhs[global_row] += vector_tuple[local_row]
                for local_column, global_column in enumerate(cell):
                    raw_matrix[(global_row, global_column)] = raw_matrix.get(
                        (global_row, global_column), sp.S.Zero) + matrix_tuple[local_row][local_column]

    raw_rhs = tuple(sp.simplify(value) for value in raw_rhs)
    constraints = _essential_constraints(problem, weak, mesh, space, substitution_map)
    constrained_matrix = {key: sp.simplify(value) for key, value in raw_matrix.items()}
    constrained_rhs = list(raw_rhs)
    for constraint in constraints:
        dof, value = constraint.dof, constraint.value
        for row in range(space.dof_count):
            if row != dof:
                constrained_rhs[row] = sp.simplify(constrained_rhs[row] -
                    constrained_matrix.get((row, dof), sp.S.Zero) * value)
        constrained_matrix = {key: item for key, item in constrained_matrix.items()
                              if key[0] != dof and key[1] != dof}
        constrained_matrix[(dof, dof)] = sp.S.One
        constrained_rhs[dof] = value
    raw_entries = _coalesce(raw_matrix)
    entries = _coalesce(constrained_matrix)
    raw_dictionary = _entry_dict(raw_entries)
    symmetric = all(sp.simplify(value - raw_dictionary.get((column, row), 0)) == 0
                    for (row, column), value in raw_dictionary.items())
    output = AssembledSystem(finite_element_space_id=space_id,
        weak_form_id=space.weak_form_id, mesh_id=space.mesh_id,
        basis_id=space.basis_id, quadrature_id=quadrature_id,
        field=space.field, substitutions=substitutions, dof_count=space.dof_count,
        local_contributions=tuple(local_outputs),
        natural_contributions=tuple(natural_outputs), raw_matrix_entries=raw_entries,
        raw_rhs=raw_rhs, essential_constraints=constraints, matrix_entries=entries,
        rhs=tuple(sp.simplify(value) for value in constrained_rhs),
        quadrature_exact=all_exact, raw_matrix_symmetric=symmetric,
        input_trust=trust)
    checks = {"local_to_global_accumulation": True,
              "essential_constraints_reconciled": True,
              "symmetric_elimination_replayed": True,
              "sparse_entries_coalesced": True}
    return EngineeringResult(operation="assemble", status="verified", trust=trust,
        value={"dof_count": output.dof_count, "raw_nnz": len(raw_entries),
               "constrained_nnz": len(entries),
               "essential_constraint_count": len(constraints),
               "natural_contribution_count": len(natural_outputs),
               "quadrature_exact": all_exact}, verification=checks,
        details={"boundary_method": "symmetric_elimination",
                 "system_scope": "quadrature_defined_finite_dimensional_system",
                 "quadrature_integral_exact": all_exact,
                 "continuous_solution": "not_claimed"},
        claim_evidence={"assemble": _bundle("sparse_local_to_global_assembly", trust,
            [] if all_exact else ["quadrature defines the discrete system but does not exactly integrate every represented term"])}), output


def verify_system(problem, weak, mesh, space, basis, quadrature, result):
    verdict, candidate = assemble_system(problem, weak, mesh, space,
        result.finite_element_space_id, basis, quadrature, result.quadrature_id,
        result.substitutions, result.input_trust)
    checks = {name: getattr(candidate, name) == getattr(result, name) for name in (
        "local_contributions", "natural_contributions", "raw_matrix_entries",
        "raw_rhs", "essential_constraints", "matrix_entries", "rhs",
        "quadrature_exact", "raw_matrix_symmetric")}
    trust = cap_trust(candidate.input_trust, result.input_trust)
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted", trust=trust,
        value={"checks": checks}, verification=checks, details=verdict.details,
        claim_evidence={"verify": _bundle("assembled_system_full_replay", trust)})


def _dense(system):
    matrix = sp.MutableDenseMatrix.zeros(system.dof_count, system.dof_count)
    for entry in system.matrix_entries:
        matrix[entry.row, entry.column] = entry.value
    return sp.ImmutableDenseMatrix(matrix), sp.ImmutableDenseMatrix(system.rhs)


def solve_system(system: AssembledSystem, system_id: str, method: str,
                 tolerance: float, condition_limit: float):
    has_float = any(entry.value.has(sp.Float) for entry in system.matrix_entries) or any(
        value.has(sp.Float) for value in system.rhs)
    selected = "numeric" if method == "numeric" or (method == "auto" and has_float) else "exact"
    if method == "exact" and has_float:
        raise ValueError("exact solve refuses floating assembled coefficients")
    if selected == "exact":
        matrix, rhs = _dense(system)
        rank = int(matrix.rank()); augmented_rank = int(matrix.row_join(rhs).rank())
        condition = None
        if rank < augmented_rank:
            status, values, residual, norm = "singular_inconsistent", (), (), None
        elif rank < system.dof_count:
            status, values, residual, norm = "singular_underdetermined", (), (), None
        else:
            values_matrix = matrix.LUsolve(rhs)
            values = tuple(sp.simplify(value) for value in values_matrix)
            residual = tuple(sp.simplify(value) for value in matrix*values_matrix-rhs)
            norm = sp.sqrt(sum(value**2 for value in residual))
            try:
                import numpy as np
                condition = float(np.linalg.cond(np.asarray(matrix.evalf(17), dtype=np.float64)))
            except (TypeError, ValueError, OverflowError):
                condition = None
            status = "unique"
    else:
        import numpy as np
        from scipy.sparse import csc_matrix
        from scipy.sparse.linalg import LinearOperator, lsqr, onenormest, splu
        rows = [entry.row for entry in system.matrix_entries]
        columns = [entry.column for entry in system.matrix_entries]
        data = [float(entry.value) for entry in system.matrix_entries]
        sparse = csc_matrix((data, (rows, columns)), shape=(system.dof_count, system.dof_count))
        vector = np.asarray([float(value) for value in system.rhs], dtype=np.float64)
        dense_for_diagnostics = sparse.toarray() if system.dof_count <= 512 else None
        rank = int(np.linalg.matrix_rank(dense_for_diagnostics, tol=tolerance)) if dense_for_diagnostics is not None else None
        augmented_rank = int(np.linalg.matrix_rank(
            np.column_stack((dense_for_diagnostics, vector)), tol=tolerance)) if dense_for_diagnostics is not None else None
        condition = float(np.linalg.cond(dense_for_diagnostics)) if dense_for_diagnostics is not None else None
        factor = None
        try:
            factor = splu(sparse)
        except RuntimeError:
            pass
        if factor is not None and dense_for_diagnostics is None:
            pivots = np.abs(factor.U.diagonal())
            scale = max(float(np.max(pivots)), 1.0)
            rank = int(np.count_nonzero(pivots > tolerance * scale))
            if rank == system.dof_count:
                inverse = LinearOperator(sparse.shape, matvec=factor.solve,
                    rmatvec=lambda item: factor.solve(item, "T"), dtype=np.float64)
                condition = float(onenormest(sparse) * onenormest(inverse))
                augmented_rank = rank
        singular = factor is None or rank is None or rank < system.dof_count
        if singular:
            candidate = lsqr(sparse, vector, atol=tolerance, btol=tolerance)[0]
            preliminary = sparse @ candidate - vector
            inconsistent = float(np.linalg.norm(preliminary)) > tolerance
            if rank is not None:
                augmented_rank = min(system.dof_count, rank + (1 if inconsistent else 0))
            status = "singular_inconsistent" if inconsistent else "singular_least_squares"
        else:
            candidate = factor.solve(vector)
            status = "ill_conditioned" if condition is not None and condition > condition_limit else "unique"
        residual_array = sparse @ candidate - vector
        values = tuple(sp.Float(float(value), 17) for value in candidate)
        residual = tuple(sp.Float(float(value), 17) for value in residual_array)
        norm = sp.Float(float(np.linalg.norm(residual_array)), 17)
    trust = cap_trust(system.input_trust, "numeric" if selected == "numeric" else "exact")
    output = FEMSolution(assembled_system_id=system_id, solve_status=status,
        method=selected, values=values, residual=residual, residual_norm=norm,
        rank=rank, augmented_rank=augmented_rank, condition_number=condition,
        tolerance=tolerance, condition_limit=condition_limit, input_trust=trust)
    residual_ok = (norm == 0 if selected == "exact" and norm is not None else
                   float(norm) <= tolerance if norm is not None else None)
    checks = {"solver_outcome_classified": True}
    if status == "unique":
        checks["algebraic_residual_within_tolerance"] = residual_ok
    result_status = "verified"
    return EngineeringResult(operation="solve", status=result_status, trust=trust,
        value={"solve_status": status, "values": values, "residual_norm": norm,
               "rank": rank, "augmented_rank": augmented_rank,
               "condition_number": condition}, verification=checks,
        details={"system_scope": "assembled_finite_dimensional_system",
                 "continuous_solution": "not_claimed",
                 "condition_number_kind": "dense_2_norm" if condition is not None else "not_computed"},
        claim_evidence={"solve": _bundle(
            "exact_rank_and_solve" if selected == "exact" else "scipy_sparse_solve_with_residual",
            trust, [f"solver outcome: {status}"], residual=norm if selected == "numeric" else None)}), output


def verify_solution(system, result):
    verdict, candidate = solve_system(system, result.assembled_system_id,
        result.method, result.tolerance, result.condition_limit)
    checks = {name: getattr(candidate, name) == getattr(result, name) for name in (
        "solve_status", "method", "values", "residual", "residual_norm",
        "rank", "augmented_rank", "condition_number")}
    trust = cap_trust(candidate.input_trust, result.input_trust)
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted", trust=trust,
        value={"checks": checks}, verification=checks, details=verdict.details,
        claim_evidence={"verify": _bundle("fem_solution_full_replay", trust)})
