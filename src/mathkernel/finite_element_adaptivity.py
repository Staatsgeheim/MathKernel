# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Phase G.5 residual indicators, marking, refinement, and observed rates."""
from __future__ import annotations

from collections import deque
from math import isfinite, log
from typing import Literal

import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import (ComputationEvidence, EvidenceBundle,
                                  EmpiricalEvidence, ModelEvidence)

from .engineering import EngineeringModel, EngineeringResult, cap_trust
from .finite_elements import FEMMesh, FiniteElementSpace, QuadratureRule, build_mesh
from .finite_element_assembly import AssembledSystem, FEMSolution
from .partial_differential_equations import PDEProblem, _expression
from .weak_forms import WeakForm


class CellErrorIndicator(EngineeringModel):
    cell_index: int
    diameter_squared: sp.Expr
    volume_residual_squared: sp.Expr
    interior_jump_squared: sp.Expr
    natural_boundary_residual_squared: sp.Expr = sp.S.Zero
    total_indicator_squared: sp.Expr

    @model_validator(mode="after")
    def validate_indicator(self):
        if self.cell_index < 0:
            raise ValueError("cell indicator index must be nonnegative")
        values = (self.diameter_squared, self.volume_residual_squared,
                  self.interior_jump_squared,
                  self.natural_boundary_residual_squared,
                  self.total_indicator_squared)
        for value in values:
            _expression(value, "cell error-indicator component")
            if value.free_symbols or value.is_nonnegative is False:
                raise ValueError("cell error-indicator components must be concrete and nonnegative")
        if sp.simplify(sum(values[1:4]) - self.total_indicator_squared) != 0:
            raise ValueError("cell error-indicator components do not reconcile")
        return self


class FEMErrorEstimate(EngineeringModel):
    solution_id: str
    assembled_system_id: str
    finite_element_space_id: str
    mesh_id: str
    estimator_kind: Literal["p1_residual_jump_diagonal_diffusion"]
    cell_indicators: tuple[CellErrorIndicator, ...]
    global_estimator_squared: sp.Expr
    global_estimator: sp.Expr
    algebraic_residual_norm: sp.Expr | None
    quadrature_exact: bool
    rigorous_error_bound: Literal[False] = False
    reliability_constant_established: Literal[False] = False
    efficiency_constant_established: Literal[False] = False
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_estimate(self):
        if not all((self.solution_id, self.assembled_system_id,
                    self.finite_element_space_id, self.mesh_id)):
            raise ValueError("error estimate requires complete solution ancestry")
        if not self.cell_indicators or tuple(item.cell_index for item in self.cell_indicators) != tuple(range(len(self.cell_indicators))):
            raise ValueError("cell error indicators must be complete and ordered")
        _expression(self.global_estimator_squared, "global estimator squared")
        _expression(self.global_estimator, "global estimator")
        if (self.global_estimator_squared.free_symbols or self.global_estimator.free_symbols or
                sp.simplify(sum(item.total_indicator_squared for item in self.cell_indicators) - self.global_estimator_squared) != 0 or
                sp.simplify(self.global_estimator ** 2 - self.global_estimator_squared) != 0):
            raise ValueError("global estimator does not reconcile with local indicators")
        return self


class RefinementMarking(EngineeringModel):
    error_estimate_id: str
    mesh_id: str
    strategy: Literal["dorfler", "maximum"]
    theta: float
    marked_cells: tuple[int, ...]
    marked_indicator_squared: sp.Expr
    total_indicator_squared: sp.Expr
    achieved_fraction: float
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_marking(self):
        if (not self.error_estimate_id or not self.mesh_id or
                not isfinite(self.theta) or not 0 < self.theta <= 1 or
                not self.marked_cells or tuple(sorted(set(self.marked_cells))) != self.marked_cells or
                not isfinite(self.achieved_fraction) or not 0 <= self.achieved_fraction <= 1):
            raise ValueError("refinement marking is invalid")
        return self


class RefinedMesh(FEMMesh):
    marking_id: str
    parent_mesh_id: str
    requested_marked_cells: tuple[int, ...]
    closure_refined_cells: tuple[int, ...]
    parent_cell_indices: tuple[int, ...]
    child_local_indices: tuple[int, ...]
    refinement_rule: Literal["triangle_red_with_conforming_closure"]

    @model_validator(mode="after")
    def validate_refinement(self):
        if (not self.marking_id or not self.parent_mesh_id or
                len(self.parent_cell_indices) != len(self.cells) or
                len(self.child_local_indices) != len(self.cells)):
            raise ValueError("refinement lineage is incomplete")
        return self


class MeshTransfer(EngineeringModel):
    marking_id: str
    parent_mesh_id: str
    parent_point_count: int
    refined_point_count: int
    interpolation_rows: tuple[tuple[tuple[int, sp.Expr], ...], ...]
    transfer_kind: Literal["nested_P1_nodal_interpolation"] = "nested_P1_nodal_interpolation"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_transfer(self):
        if (not self.marking_id or not self.parent_mesh_id or self.parent_point_count < 1 or
                self.refined_point_count != len(self.interpolation_rows)):
            raise ValueError("mesh transfer shape is invalid")
        for row in self.interpolation_rows:
            if not row or sp.simplify(sum(weight for _, weight in row) - 1) != 0:
                raise ValueError("mesh transfer rows must be affine combinations")
            if any(index < 0 or index >= self.parent_point_count for index, _ in row):
                raise ValueError("mesh transfer references an invalid parent DOF")
        return self


class FEMConvergenceObservation(EngineeringModel):
    fine_estimate_id: str
    coarse_estimate_id: str
    coarse_mesh_id: str
    fine_mesh_id: str
    dimension: int
    coarse_cell_count: int
    fine_cell_count: int
    coarse_estimator: sp.Expr
    fine_estimator: sp.Expr
    estimator_ratio: float | None
    observed_estimator_rate: float | None
    interpretation: Literal["empirical_estimator_sequence"] = "empirical_estimator_sequence"
    convergence_theorem: Literal[False] = False
    continuum_error_bound: Literal[False] = False
    input_trust: str = "empirical"

    @model_validator(mode="after")
    def validate_observation(self):
        if (not self.fine_estimate_id or not self.coarse_estimate_id or
                self.dimension < 1 or self.coarse_cell_count < 1 or
                self.fine_cell_count <= self.coarse_cell_count):
            raise ValueError("convergence observation requires a strict refinement pair")
        for value in (self.estimator_ratio, self.observed_estimator_rate):
            if value is not None and not isfinite(value):
                raise ValueError("observed convergence values must be finite")
        return self


def _bundle(method: str, trust: str, *, empirical=False):
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(engine="finite_element_adaptivity",
            method=method, arithmetic=trust, deterministic=True, trust=trust)],
        model=[ModelEvidence(role="diagnostic", trust="unknown", assumptions=[],
            diagnostics=["the residual indicator is not a rigorous continuum error bound",
                         "reliability and efficiency constants are not established",
                         "an observed estimator rate is not a convergence theorem"],
            metadata={"continuum_error_bound": False, "convergence_theorem": False})],
        justified_trust=trust)
    if empirical:
        bundle.empirical.append(EmpiricalEvidence(sampling_method="two_mesh_estimator_rate",
            sample_size=2, role="diagnostic", trust="empirical",
            metadata={"convergence_theorem": False, "error_bound": False}))
    return bundle


def _cell_geometry(mesh, cell):
    points = [mesh.points[index] for index in cell]
    base = sp.Matrix(points[0])
    jacobian = sp.Matrix.hstack(*(sp.Matrix(point) - base for point in points[1:]))
    return points, jacobian, sp.simplify(jacobian.det())


def _gradients(mesh, space, values):
    result = []
    reference_gradients = sp.Matrix.hstack(
        sp.Matrix([-1] * mesh.dimension),
        *[sp.eye(mesh.dimension)[:, i] for i in range(mesh.dimension)])
    for cell in mesh.cells:
        _, jacobian, _ = _cell_geometry(mesh, cell)
        local = sp.Matrix([values[index] for index in cell])
        result.append(tuple(sp.simplify(v) for v in jacobian.inv().T * reference_gradients * local))
    return tuple(result)


def _diffusion_coefficients(problem, system):
    equation = problem.equations[0]
    if len(problem.fields) != 1 or len(problem.equations) != 1 or not equation.linear:
        raise ValueError("G.5 estimator requires one scalar linear equation")
    substitutions = {sp.Symbol(name): value for name, value in system.substitutions}
    substitutions.update({sp.Symbol(name): value for name, value in problem.parameters if value is not None})
    coefficients = [None] * len(problem.independent_variables)
    for term in equation.terms:
        nonzero = [i for i, order in enumerate(term.derivative) if order]
        if (term.field != problem.fields[0] or term.power != 1 or len(nonzero) != 1 or
                term.derivative[nonzero[0]] != 2 or sum(term.derivative) != 2 or
                coefficients[nonzero[0]] is not None):
            raise ValueError("G.5 estimator supports only pure second-order diagonal diffusion terms")
        value = sp.simplify(term.coefficient.subs(substitutions))
        if value.free_symbols:
            raise ValueError("G.5 diffusion coefficients must be concrete constants")
        coefficients[nonzero[0]] = value
    if any(value is None or value == 0 for value in coefficients):
        raise ValueError("G.5 estimator requires one nonzero diffusion coefficient per coordinate")
    source = sp.simplify(equation.source.subs(substitutions))
    unresolved = source.free_symbols - {sp.Symbol(name) for name in problem.independent_variables}
    if unresolved:
        raise ValueError("G.5 source retains unresolved parameters")
    return tuple(coefficients), source


def estimate_error(problem: PDEProblem, weak: WeakForm, mesh: FEMMesh,
                   space: FiniteElementSpace, system: AssembledSystem,
                   solution: FEMSolution, solution_id: str,
                   quadrature: QuadratureRule):
    if mesh.cell_type != "triangle" or mesh.dimension != 2:
        raise ValueError("G.5 residual-jump estimation currently requires triangle P1 meshes")
    if any(condition.kind != "dirichlet" for condition in problem.boundary_conditions):
        raise ValueError("G.5 estimator currently requires an essential-only boundary partition")
    if solution.solve_status not in {"unique", "ill_conditioned"} or len(solution.values) != space.dof_count:
        raise ValueError("error estimation requires a complete algebraic solution")
    coefficients, source = _diffusion_coefficients(problem, system)
    gradients = _gradients(mesh, space, solution.values)
    physical_symbols = tuple(sp.Symbol(name) for name in problem.independent_variables)
    xi = sp.symbols("xi0:2")
    volume = [sp.S.Zero] * len(mesh.cells)
    diameters = []
    exact = True
    constant_source = not source.free_symbols
    for cell_index, cell in enumerate(mesh.cells):
        points, jacobian, determinant = _cell_geometry(mesh, cell)
        if constant_source:
            # The reference weights already integrate constants to the simplex
            # measure. Avoid constructing/substituting an affine polynomial.
            integral = sp.simplify(source ** 2 * determinant * quadrature.reference_measure)
        else:
            physical = tuple(points[0][axis] + sum(
                jacobian[axis, j] * xi[j] for j in range(2)) for axis in range(2))
            residual = sp.simplify(-source.subs(dict(zip(physical_symbols, physical))))
            integrand = sp.expand(residual ** 2 * determinant)
            try:
                exact &= sp.Poly(integrand, *xi).total_degree() <= quadrature.degree_exact
            except sp.PolynomialError:
                exact = False
            integral = sp.simplify(sum(weight * integrand.subs(dict(zip(xi, point)))
                                       for point, weight in zip(quadrature.points, quadrature.weights)))
        distance_squared = [sp.simplify(sum((left[k] - right[k]) ** 2 for k in range(2)))
                            for i, left in enumerate(points) for right in points[i + 1:]]
        diameter_squared = max(distance_squared, key=lambda v: float(sp.N(v, 17)))
        diameters.append(diameter_squared)
        volume[cell_index] = sp.simplify(diameter_squared * integral)
    incidence = {}
    for cell_index, cell in enumerate(mesh.cells):
        for omit in range(3):
            facet = tuple(sorted(cell[i] for i in range(3) if i != omit))
            incidence.setdefault(facet, []).append((cell_index, omit))
    jumps = [sp.S.Zero] * len(mesh.cells)
    for facet, owners in incidence.items():
        if len(owners) != 2:
            continue
        scaled_fluxes = []
        a, b = (mesh.points[index] for index in facet)
        dx, dy = b[0] - a[0], b[1] - a[1]
        for cell_index, omit in owners:
            third = mesh.points[mesh.cells[cell_index][omit]]
            raw = sp.Matrix([dy, -dx])
            midpoint = sp.Matrix([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
            if sp.simplify(raw.dot(sp.Matrix(third) - midpoint)).is_positive:
                raw = -raw
            flux = sp.Matrix([coefficients[i] * gradients[cell_index][i] for i in range(2)])
            scaled_fluxes.append(sp.simplify(flux.dot(raw)))
        contribution = sp.simplify((sum(scaled_fluxes)) ** 2)
        for cell_index, _ in owners:
            jumps[cell_index] += sp.Rational(1, 2) * contribution
    cells = tuple(CellErrorIndicator(cell_index=i, diameter_squared=diameters[i],
        volume_residual_squared=sp.simplify(volume[i]),
        interior_jump_squared=sp.simplify(jumps[i]),
        total_indicator_squared=sp.simplify(volume[i] + jumps[i]))
        for i in range(len(mesh.cells)))
    total = sp.simplify(sum(item.total_indicator_squared for item in cells))
    output = FEMErrorEstimate(solution_id=solution_id,
        assembled_system_id=solution.assembled_system_id,
        finite_element_space_id=system.finite_element_space_id,
        mesh_id=space.mesh_id, estimator_kind="p1_residual_jump_diagonal_diffusion",
        cell_indicators=cells, global_estimator_squared=total,
        global_estimator=sp.sqrt(total), algebraic_residual_norm=solution.residual_norm,
        quadrature_exact=bool(exact), input_trust=cap_trust(solution.input_trust,
                                                            system.input_trust))
    checks = {"solution_ancestry": True, "local_components_recomputed": True,
              "global_norm_reconciled": True, "rigorous_error_bound": None}
    return EngineeringResult(operation="estimate_error", status="verified",
        trust=output.input_trust,
        value={"global_estimator": output.global_estimator,
               "global_estimator_squared": total, "quadrature_exact": bool(exact),
               "rigorous_error_bound": False}, verification=checks,
        details={"estimator_not_bound": True, "algebraic_residual_separate": True},
        claim_evidence={"estimate_error": _bundle("p1_residual_jump_recomputation",
                                                   output.input_trust)}), output


def verify_estimate(problem, weak, mesh, space, system, solution, estimate, quadrature):
    _, candidate = estimate_error(problem, weak, mesh, space, system, solution,
                                  estimate.solution_id, quadrature)
    checks = {name: getattr(candidate, name) == getattr(estimate, name) for name in (
        "assembled_system_id", "finite_element_space_id", "mesh_id",
        "cell_indicators", "global_estimator_squared", "global_estimator",
        "algebraic_residual_norm", "quadrature_exact")}
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted",
        trust=cap_trust(candidate.input_trust, estimate.input_trust), value={"checks": checks},
        verification=checks, details={"rigorous_error_bound": False},
        claim_evidence={"verify": _bundle("error_estimate_replay", estimate.input_trust)})


def mark(estimate: FEMErrorEstimate, estimate_id: str, strategy: str, theta: float):
    if strategy not in {"dorfler", "maximum"}:
        raise ValueError("marking strategy must be dorfler or maximum")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not isfinite(float(theta)) or not 0 < float(theta) <= 1:
        raise ValueError("theta must be finite and in (0, 1]")
    values = [(item.cell_index, item.total_indicator_squared) for item in estimate.cell_indicators]
    if all(value == 0 for _, value in values):
        raise ValueError("a zero estimator has no refinement marking")
    ordered = sorted(values, key=lambda item: (-float(sp.N(item[1], 17)), item[0]))
    total = estimate.global_estimator_squared
    if strategy == "dorfler":
        selected, accumulated = [], sp.S.Zero
        for index, value in ordered:
            selected.append(index); accumulated += value
            if float(sp.N(accumulated / total, 17)) >= float(theta):
                break
    else:
        threshold = float(theta) * float(sp.N(ordered[0][1], 17))
        selected = [index for index, value in ordered if float(sp.N(value, 17)) >= threshold]
        accumulated = sum((values[index][1] for index in selected), sp.S.Zero)
    selected = tuple(sorted(selected)); fraction = float(sp.N(accumulated / total, 17))
    output = RefinementMarking(error_estimate_id=estimate_id, mesh_id=estimate.mesh_id,
        strategy=strategy, theta=float(theta), marked_cells=selected,
        marked_indicator_squared=sp.simplify(accumulated), total_indicator_squared=total,
        achieved_fraction=fraction, input_trust=estimate.input_trust)
    checks = {"policy_recomputed": True, "threshold_met": fraction + 1e-15 >= float(theta) if strategy == "dorfler" else True}
    return EngineeringResult(operation="mark", status="verified", trust=estimate.input_trust,
        value={"marked_cells": selected, "achieved_fraction": fraction}, verification=checks,
        claim_evidence={"mark": _bundle(f"{strategy}_marking", estimate.input_trust)}), output


def verify_marking(estimate, marking):
    _, candidate = mark(estimate, marking.error_estimate_id, marking.strategy, marking.theta)
    checks = {name: getattr(candidate, name) == getattr(marking, name) for name in (
        "mesh_id", "marked_cells", "marked_indicator_squared", "total_indicator_squared",
        "achieved_fraction")}
    return EngineeringResult(operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=marking.input_trust, value={"checks": checks}, verification=checks,
        claim_evidence={"verify": _bundle("marking_policy_replay", marking.input_trust)})


def refine(weak: WeakForm, mesh: FEMMesh, marking: RefinementMarking, marking_id: str):
    if mesh.cell_type != "triangle":
        raise ValueError("G.5 conforming refinement currently requires triangle meshes")
    incidence = {}
    for cell_index, cell in enumerate(mesh.cells):
        for omit in range(3):
            edge = tuple(sorted(cell[i] for i in range(3) if i != omit))
            incidence.setdefault(edge, []).append(cell_index)
    adjacency = [set() for _ in mesh.cells]
    for owners in incidence.values():
        if len(owners) == 2:
            left, right = owners
            adjacency[left].add(right); adjacency[right].add(left)
    closure = set(marking.marked_cells)
    pending = deque(marking.marked_cells)
    while pending:
        for neighbour in adjacency[pending.popleft()]:
            if neighbour not in closure:
                closure.add(neighbour); pending.append(neighbour)
    points = list(mesh.points)
    rows = [((i, sp.S.One),) for i in range(len(points))]
    midpoint = {}
    for cell_index in sorted(closure):
        cell = mesh.cells[cell_index]
        for edge in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])):
            key = tuple(sorted(edge))
            if key not in midpoint:
                midpoint[key] = len(points)
                points.append(tuple(sp.simplify((mesh.points[key[0]][axis] + mesh.points[key[1]][axis]) / 2)
                                    for axis in range(2)))
                rows.append(((key[0], sp.Rational(1, 2)), (key[1], sp.Rational(1, 2))))
    cells, parents, child_local = [], [], []
    for parent, (a, b, c) in enumerate(mesh.cells):
        if parent not in closure:
            cells.append((a, b, c)); parents.append(parent); child_local.append(0)
            continue
        ab, bc, ca = midpoint[tuple(sorted((a, b)))], midpoint[tuple(sorted((b, c)))], midpoint[tuple(sorted((c, a)))]
        children = ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
        cells.extend(children); parents.extend((parent,) * 4); child_local.extend(range(4))
    _, base = build_mesh(weak, mesh.weak_form_id, tuple(points), tuple(cells),
                         "triangle", mesh.input_trust)
    refined = RefinedMesh(**base.model_dump(), marking_id=marking_id,
        parent_mesh_id=marking.mesh_id, requested_marked_cells=marking.marked_cells,
        closure_refined_cells=tuple(sorted(closure)), parent_cell_indices=tuple(parents),
        child_local_indices=tuple(child_local), refinement_rule="triangle_red_with_conforming_closure")
    transfer = MeshTransfer(marking_id=marking_id, parent_mesh_id=marking.mesh_id,
        parent_point_count=len(mesh.points), refined_point_count=len(points),
        interpolation_rows=tuple(rows), input_trust=mesh.input_trust)
    checks = {"parent_child_map_complete": True, "conforming_closure_recomputed": True,
              "nested_interpolation_rows_recomputed": True}
    return EngineeringResult(operation="refine", status="verified", trust=mesh.input_trust,
        value={"point_count": len(points), "cell_count": len(cells),
               "requested_cells": marking.marked_cells,
               "closure_cells": tuple(sorted(closure))}, verification=checks,
        claim_evidence={"refine": _bundle("conforming_red_refinement", mesh.input_trust)}), {
            "mesh": refined, "transfer": transfer}


def verify_refined_mesh(weak, parent, marking, refined):
    _, outputs = refine(weak, parent, marking, refined.marking_id)
    candidate = outputs["mesh"]
    checks = {name: getattr(candidate, name) == getattr(refined, name) for name in (
        "points", "cells", "boundary_facets", "interior_facets", "cell_determinants",
        "requested_marked_cells", "closure_refined_cells", "parent_cell_indices",
        "child_local_indices")}
    return EngineeringResult(operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=refined.input_trust, value={"checks": checks}, verification=checks,
        claim_evidence={"verify": _bundle("refinement_lineage_replay", refined.input_trust)})


def verify_transfer(weak, parent, marking, transfer):
    _, outputs = refine(weak, parent, marking, transfer.marking_id)
    candidate = outputs["transfer"]
    checks = {name: getattr(candidate, name) == getattr(transfer, name) for name in (
        "parent_point_count", "refined_point_count", "interpolation_rows")}
    return EngineeringResult(operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=transfer.input_trust, value={"checks": checks}, verification=checks,
        claim_evidence={"verify": _bundle("mesh_transfer_replay", transfer.input_trust)})


def compare(fine: FEMErrorEstimate, fine_id: str, coarse: FEMErrorEstimate,
            coarse_id: str, fine_mesh: FEMMesh, coarse_mesh: FEMMesh):
    if fine.mesh_id == coarse.mesh_id or not isinstance(fine_mesh, RefinedMesh) or fine_mesh.parent_mesh_id != coarse.mesh_id:
        raise ValueError("convergence comparison requires a direct parent-to-child refinement pair")
    coarse_value, fine_value = float(sp.N(coarse.global_estimator, 17)), float(sp.N(fine.global_estimator, 17))
    ratio = None if coarse_value == 0 else fine_value / coarse_value
    h_ratio = (len(coarse_mesh.cells) / len(fine_mesh.cells)) ** (1 / fine_mesh.dimension)
    rate = None
    if coarse_value > 0 and fine_value > 0 and h_ratio != 1:
        rate = log(fine_value / coarse_value) / log(h_ratio)
    output = FEMConvergenceObservation(fine_estimate_id=fine_id,
        coarse_estimate_id=coarse_id, coarse_mesh_id=coarse.mesh_id,
        fine_mesh_id=fine.mesh_id, dimension=fine_mesh.dimension,
        coarse_cell_count=len(coarse_mesh.cells), fine_cell_count=len(fine_mesh.cells),
        coarse_estimator=coarse.global_estimator, fine_estimator=fine.global_estimator,
        estimator_ratio=ratio, observed_estimator_rate=rate, input_trust="empirical")
    checks = {"direct_refinement_lineage": True, "estimator_ratio_recomputed": True,
              "observed_rate_recomputed": True, "convergence_theorem": None}
    return EngineeringResult(operation="compare", status="verified", trust="empirical",
        value={"estimator_ratio": ratio, "observed_estimator_rate": rate,
               "convergence_theorem": False}, verification=checks,
        claim_evidence={"compare": _bundle("observed_estimator_rate", "empirical", empirical=True)}), output


def verify_observation(fine, coarse, fine_mesh, coarse_mesh, observation):
    _, candidate = compare(fine, observation.fine_estimate_id, coarse,
                           observation.coarse_estimate_id, fine_mesh, coarse_mesh)
    checks = {name: getattr(candidate, name) == getattr(observation, name) for name in (
        "coarse_mesh_id", "fine_mesh_id", "coarse_cell_count", "fine_cell_count",
        "coarse_estimator", "fine_estimator", "estimator_ratio", "observed_estimator_rate")}
    return EngineeringResult(operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="empirical", value={"checks": checks}, verification=checks,
        details={"convergence_theorem": False, "continuum_error_bound": False},
        claim_evidence={"verify": _bundle("observed_rate_replay", "empirical", empirical=True)})
