# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed Phase G.3 simplex meshes, P1 spaces, bases, and quadrature."""
from __future__ import annotations

from itertools import product
from typing import Literal

import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import ComputationEvidence, EvidenceBundle, ModelEvidence

from .engineering import EngineeringModel, EngineeringResult, cap_trust
from .partial_differential_equations import PDEProblem, _expression
from .weak_forms import WeakForm


CellType = Literal["interval", "triangle", "tetrahedron"]


def _simplex_size(cell_type: CellType) -> tuple[int, int]:
    return {"interval": (1, 2), "triangle": (2, 3), "tetrahedron": (3, 4)}[cell_type]


def _simplex_determinant(points, cell):
    base = points[cell[0]]
    matrix = sp.Matrix([
        [points[cell[column]][row] - base[row]
         for column in range(1, len(cell))]
        for row in range(len(base))
    ])
    return sp.simplify(matrix.det())


def _sign(value: sp.Expr, scale: float = 1.0) -> int | None:
    if value.has(sp.Float):
        numeric = float(sp.N(value, 17))
        if abs(numeric) <= 128 * 2.220446049250313e-16 * max(1.0, scale):
            return None
        return 1 if numeric > 0 else -1
    if value.is_positive is True: return 1
    if value.is_negative is True: return -1
    if value.is_zero is True or value == 0: return 0
    return None


def _facets(cell):
    return tuple(tuple(cell[index] for index in range(len(cell)) if index != omit)
                 for omit in range(len(cell)))


class FEMMesh(EngineeringModel):
    weak_form_id: str
    triangulation_id: str | None = None
    dimension: int
    cell_type: CellType
    points: tuple[tuple[sp.Expr, ...], ...]
    cells: tuple[tuple[int, ...], ...]
    boundary_facets: tuple[tuple[int, ...], ...]
    boundary_facet_owners: tuple[tuple[int, int, int], ...]
    interior_facets: tuple[tuple[int, ...], ...]
    cell_determinants: tuple[sp.Expr, ...]
    orientation_flips: tuple[int, ...] = ()
    orientation_status: Literal["positive", "ambiguous"] = "positive"
    combinatorially_conforming: bool = True
    cell_component_count: int
    domain_coverage: Literal["not_established"] = "not_established"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_mesh(self):
        expected_dimension, nodes = _simplex_size(self.cell_type)
        if not self.weak_form_id or self.dimension != expected_dimension:
            raise ValueError("mesh dimension and weak-form provenance are invalid")
        if not self.points or not self.cells:
            raise ValueError("mesh requires points and cells")
        for point in self.points:
            if len(point) != self.dimension:
                raise ValueError("mesh point dimension is invalid")
            for value in point:
                _expression(value, "mesh coordinate")
                if value.free_symbols:
                    raise ValueError("mesh coordinates must be concrete")
        if len(set(self.points)) != len(self.points):
            raise ValueError("mesh points must be unique")
        for cell in self.cells:
            if len(cell) != nodes or len(set(cell)) != nodes or any(
                    isinstance(index, bool) or not isinstance(index, int) or
                    index < 0 or index >= len(self.points) for index in cell):
                raise ValueError("mesh cell connectivity is invalid")
        if len({frozenset(cell) for cell in self.cells}) != len(self.cells):
            raise ValueError("mesh cells must be unique")
        if len(self.cell_determinants) != len(self.cells):
            raise ValueError("mesh determinant count is invalid")
        for value in self.cell_determinants:
            _expression(value, "cell determinant")
            if _sign(value) == 0:
                raise ValueError("mesh cells must be nondegenerate")
        if any(index < 0 or index >= len(self.cells) for index in self.orientation_flips):
            raise ValueError("mesh orientation-flip provenance is invalid")
        if self.cell_component_count < 1 or self.cell_component_count > len(self.cells):
            raise ValueError("mesh cell-component count is invalid")
        if len(self.boundary_facet_owners) != len(self.boundary_facets):
            raise ValueError("boundary-facet ownership count is invalid")
        for cell_index, local_facet, orientation in self.boundary_facet_owners:
            if (cell_index < 0 or cell_index >= len(self.cells) or
                    local_facet < 0 or local_facet >= nodes or orientation not in {-1, 1}):
                raise ValueError("boundary-facet ownership is invalid")
        if not self.combinatorially_conforming:
            raise ValueError("nonmanifold simplex incidence is unsupported")
        cap_trust(self.input_trust)
        return self


class ReferenceElement(EngineeringModel):
    mesh_id: str
    cell_type: CellType
    dimension: int
    polynomial_degree: Literal[1] = 1
    coordinate_names: tuple[str, ...]
    vertices: tuple[tuple[sp.Expr, ...], ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_reference(self):
        dimension, nodes = _simplex_size(self.cell_type)
        if (not self.mesh_id or self.dimension != dimension or
                len(self.coordinate_names) != dimension or len(self.vertices) != nodes):
            raise ValueError("reference-element shape is invalid")
        if any(not name.isidentifier() for name in self.coordinate_names):
            raise ValueError("reference coordinates must be identifiers")
        cap_trust(self.input_trust)
        return self


class BasisFunctionSet(EngineeringModel):
    reference_element_id: str
    family: Literal["lagrange"] = "lagrange"
    polynomial_degree: Literal[1] = 1
    coordinate_names: tuple[str, ...]
    nodes: tuple[tuple[sp.Expr, ...], ...]
    functions: tuple[sp.Expr, ...]
    gradients: tuple[tuple[sp.Expr, ...], ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_basis(self):
        if (not self.reference_element_id or len(self.functions) != len(self.nodes) or
                len(self.gradients) != len(self.functions)):
            raise ValueError("basis cardinality is invalid")
        if any(len(gradient) != len(self.coordinate_names) for gradient in self.gradients):
            raise ValueError("basis gradients have the wrong dimension")
        for value in (*self.functions, *(item for row in self.gradients for item in row)):
            _expression(value, "basis expression")
        cap_trust(self.input_trust)
        return self


class QuadratureRule(EngineeringModel):
    reference_element_id: str
    cell_type: CellType
    dimension: int
    degree_exact: Literal[1, 2]
    points: tuple[tuple[sp.Expr, ...], ...]
    weights: tuple[sp.Expr, ...]
    reference_measure: sp.Expr
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_rule(self):
        if not self.reference_element_id or not self.points or len(self.points) != len(self.weights):
            raise ValueError("quadrature points and weights are invalid")
        if any(len(point) != self.dimension for point in self.points):
            raise ValueError("quadrature point dimension is invalid")
        for value in (*self.weights, self.reference_measure,
                      *(entry for point in self.points for entry in point)):
            _expression(value, "quadrature scalar")
            if value.free_symbols:
                raise ValueError("quadrature data must be concrete")
        cap_trust(self.input_trust)
        return self


class FiniteElementSpace(EngineeringModel):
    mesh_id: str
    reference_element_id: str
    basis_id: str
    weak_form_id: str
    field: str
    family: Literal["P1_lagrange"] = "P1_lagrange"
    continuity: Literal["C0"] = "C0"
    local_to_global: tuple[tuple[int, ...], ...]
    dof_coordinates: tuple[tuple[sp.Expr, ...], ...]
    essential_dofs: tuple[int, ...]
    dof_count: int
    conformity_scope: Literal["vertex_P1_combinatorial"] = "vertex_P1_combinatorial"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_space(self):
        if not all((self.mesh_id, self.reference_element_id, self.basis_id,
                    self.weak_form_id)) or not self.field.isidentifier():
            raise ValueError("finite-element space provenance is invalid")
        if self.dof_count != len(self.dof_coordinates):
            raise ValueError("finite-element DOF count does not reconcile")
        if any(index < 0 or index >= self.dof_count for index in self.essential_dofs):
            raise ValueError("essential DOF is outside the global range")
        cap_trust(self.input_trust)
        return self


def _cell_components(cell_count: int, incidence: dict) -> int:
    adjacency = [set() for _ in range(cell_count)]
    for owners in incidence.values():
        if len(owners) == 2:
            left, right = owners[0][0], owners[1][0]
            adjacency[left].add(right); adjacency[right].add(left)
    remaining = set(range(cell_count)); count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            for neighbour in adjacency[stack.pop()] & remaining:
                remaining.remove(neighbour); stack.append(neighbour)
    return count


def _bundle(method: str, trust: str, diagnostics: list[str]) -> EvidenceBundle:
    return EvidenceBundle(
        computation=[ComputationEvidence(engine="finite_elements", method=method,
                                         arithmetic=trust, deterministic=True,
                                         trust=trust)],
        model=[ModelEvidence(assumptions=[], diagnostics=[*diagnostics,
            "mesh coverage and geometric non-overlap are not established",
            "mesh quality does not prove approximation convergence",
            "no assembly, algebraic solve, continuous solution, or discrete solution is claimed"],
            role="diagnostic", trust="unknown",
            metadata={"assembly": "not_performed", "solution": "not_claimed"})],
        justified_trust=trust)


def build_mesh(weak: WeakForm, weak_form_id: str, points, cells,
               cell_type: CellType, trust: str, triangulation_id=None):
    dimension, nodes = _simplex_size(cell_type)
    if dimension != len(weak.integration_variables):
        raise ValueError("mesh dimension must equal the weak-form integration dimension")
    if any(len(cell) != nodes for cell in cells):
        raise ValueError("cell connectivity does not match cell_type")
    if any(len(set(cell)) != nodes or any(index < 0 or index >= len(points)
                                         for index in cell) for cell in cells):
        raise ValueError("mesh cell connectivity is invalid")
    oriented = []
    determinants = []
    flips = []
    ambiguous = False
    for index, raw_cell in enumerate(cells):
        cell = tuple(raw_cell)
        determinant = _simplex_determinant(points, cell)
        sign = _sign(determinant)
        if sign == 0:
            raise ValueError("degenerate mesh cell")
        if sign is None:
            ambiguous = True
        elif sign < 0:
            cell = (*cell[:-2], cell[-1], cell[-2])
            determinant = sp.simplify(-determinant)
            flips.append(index)
        oriented.append(cell); determinants.append(determinant)
    incidence = {}
    for cell_index, cell in enumerate(oriented):
        for local_facet, facet in enumerate(_facets(cell)):
            key = tuple(sorted(facet))
            incidence.setdefault(key, []).append((cell_index, local_facet,
                                                  1 if local_facet % 2 == 0 else -1))
    if any(len(owners) > 2 for owners in incidence.values()):
        raise ValueError("nonmanifold facet incidence exceeds two cells")
    boundary = tuple(sorted(key for key, owners in incidence.items() if len(owners) == 1))
    interior = tuple(sorted(key for key, owners in incidence.items() if len(owners) == 2))
    owners = tuple(incidence[key][0] for key in boundary)
    components = _cell_components(len(oriented), incidence)
    mesh = FEMMesh(
        weak_form_id=weak_form_id, triangulation_id=triangulation_id,
        dimension=dimension, cell_type=cell_type, points=tuple(points),
        cells=tuple(oriented), boundary_facets=boundary,
        boundary_facet_owners=owners, interior_facets=interior,
        cell_determinants=tuple(determinants), orientation_flips=tuple(flips),
        orientation_status="ambiguous" if ambiguous else "positive",
        cell_component_count=components, input_trust=trust)
    status = "unknown" if ambiguous else "verified"
    checks = {"connectivity_in_range": True, "cells_nondegenerate": None if ambiguous else True,
              "positive_orientation": None if ambiguous else True,
              "facet_incidence_at_most_two": True,
              "cell_components_recomputed": True,
              "domain_coverage": None}
    return EngineeringResult(
        operation="construct_mesh", status=status, trust=trust,
        value={"dimension": dimension, "point_count": len(points),
               "cell_count": len(cells), "boundary_facet_count": len(boundary),
               "interior_facet_count": len(interior),
               "cell_component_count": components,
               "orientation_flips": flips, "orientation_status": mesh.orientation_status},
        details={"conformity": "combinatorial_only",
                 "domain_coverage": "not_established", "assembly": "not_performed"},
        verification=checks, claim_evidence={"construct_mesh": _bundle(
            "simplex_connectivity_determinant_and_facet_incidence", trust,
            ["numeric near-degeneracy remains ambiguous"])}), mesh


def verify_mesh(mesh: FEMMesh):
    incidence = {}
    determinants = []
    for cell_index, cell in enumerate(mesh.cells):
        determinants.append(_simplex_determinant(mesh.points, cell))
        for local_facet, facet in enumerate(_facets(cell)):
            key = tuple(sorted(facet))
            incidence.setdefault(key, []).append((cell_index, local_facet,
                                                  1 if local_facet % 2 == 0 else -1))
    boundary = tuple(sorted(key for key, owners in incidence.items() if len(owners) == 1))
    interior = tuple(sorted(key for key, owners in incidence.items() if len(owners) == 2))
    owners = tuple(incidence[key][0] for key in boundary)
    components = _cell_components(len(mesh.cells), incidence)
    signs = [_sign(value) for value in determinants]
    checks = {"determinants_recomputed": tuple(determinants) == mesh.cell_determinants,
              "boundary_facets_recomputed": boundary == mesh.boundary_facets,
              "boundary_owners_recomputed": owners == mesh.boundary_facet_owners,
              "interior_facets_recomputed": interior == mesh.interior_facets,
              "facet_incidence_at_most_two": all(len(value) <= 2 for value in incidence.values()),
              "cell_components_recomputed": components == mesh.cell_component_count,
              "positive_orientation": None if None in signs else all(value == 1 for value in signs),
              "domain_coverage": None}
    failed = any(value is False for value in checks.values())
    unknown = any(value is None for value in checks.values())
    status = "refuted" if failed else "unknown" if unknown else "verified"
    return EngineeringResult(
        operation="verify", status=status, trust=mesh.input_trust,
        value={"checks": checks}, verification=checks,
        details={"conformity": "combinatorial_only", "domain_coverage": "not_established"},
        claim_evidence={"verify": _bundle("mesh_replay", mesh.input_trust, [])})


def make_reference(mesh: FEMMesh, mesh_id: str):
    if mesh.orientation_status != "positive":
        raise ValueError("ambiguous mesh orientation cannot produce a reference element")
    dimension, _ = _simplex_size(mesh.cell_type)
    names = ("xi", "eta", "zeta")[:dimension]
    origin = (sp.S.Zero,) * dimension
    vertices = [origin]
    for axis in range(dimension):
        vertices.append(tuple(sp.S.One if index == axis else sp.S.Zero
                              for index in range(dimension)))
    output = ReferenceElement(mesh_id=mesh_id, cell_type=mesh.cell_type,
        dimension=dimension, coordinate_names=names, vertices=tuple(vertices),
        input_trust=mesh.input_trust)
    return EngineeringResult(
        operation="reference_element", status="verified", trust=mesh.input_trust,
        value={"cell_type": mesh.cell_type, "dimension": dimension,
               "polynomial_degree": 1},
        verification={"canonical_simplex": True},
        details={"physical_mapping": "affine_per_cell", "assembly": "not_performed"},
        claim_evidence={"reference_element": _bundle(
            "canonical_unit_simplex", mesh.input_trust, [])}), output


def verify_reference(mesh: FEMMesh, result: ReferenceElement):
    _, candidate = make_reference(mesh, result.mesh_id)
    checks = {"cell_type_recomputed": candidate.cell_type == result.cell_type,
              "coordinates_recomputed": candidate.coordinate_names == result.coordinate_names,
              "vertices_recomputed": candidate.vertices == result.vertices,
              "degree_recomputed": result.polynomial_degree == 1}
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted",
        trust=cap_trust(mesh.input_trust, result.input_trust),
        value={"checks": checks}, verification=checks,
        details={"assembly": "not_performed"},
        claim_evidence={"verify": _bundle("reference_element_replay",
            cap_trust(mesh.input_trust, result.input_trust), [])})


def make_basis(reference: ReferenceElement, reference_id: str):
    symbols = tuple(sp.Symbol(name) for name in reference.coordinate_names)
    functions = (sp.simplify(1 - sum(symbols)), *symbols)
    gradients = tuple(tuple(sp.diff(function, symbol) for symbol in symbols)
                      for function in functions)
    output = BasisFunctionSet(reference_element_id=reference_id,
        coordinate_names=reference.coordinate_names, nodes=reference.vertices,
        functions=functions, gradients=gradients, input_trust=reference.input_trust)
    matrix = tuple(tuple(sp.simplify(function.subs(dict(zip(symbols, node))))
                         for node in reference.vertices) for function in functions)
    checks = {"basis_cardinality": len(functions) == len(reference.vertices),
              "nodal_kronecker_property": matrix == tuple(tuple(
                  sp.S.One if row == column else sp.S.Zero
                  for column in range(len(functions))) for row in range(len(functions))),
              "partition_of_unity": sp.simplify(sum(functions) - 1) == 0,
              "gradient_partition": all(sp.simplify(sum(row)) == 0
                  for row in zip(*gradients))}
    return EngineeringResult(operation="basis", status="verified", trust=reference.input_trust,
        value={"basis_count": len(functions), "functions": functions,
               "gradients": gradients}, verification=checks,
        details={"basis": "nodal P1 Lagrange", "assembly": "not_performed"},
        claim_evidence={"basis": _bundle("symbolic_p1_nodal_basis",
            reference.input_trust, [])}), output


def verify_basis(reference: ReferenceElement, result: BasisFunctionSet):
    verdict, candidate = make_basis(reference, result.reference_element_id)
    checks = {"nodes_recomputed": candidate.nodes == result.nodes,
              "functions_recomputed": candidate.functions == result.functions,
              "gradients_recomputed": candidate.gradients == result.gradients}
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted",
        trust=cap_trust(reference.input_trust, result.input_trust),
        value={"checks": checks}, verification=checks,
        details=verdict.details,
        claim_evidence={"verify": _bundle("basis_replay",
            cap_trust(reference.input_trust, result.input_trust), [])})


def _quadrature_data(cell_type: CellType, degree: int):
    if cell_type == "interval":
        if degree == 1:
            return ((sp.Rational(1, 2),),), (sp.S.One,), sp.S.One
        offset = sp.sqrt(3) / 6
        return (((sp.Rational(1, 2) - offset),),
                ((sp.Rational(1, 2) + offset),)), (sp.Rational(1, 2),) * 2, sp.S.One
    if cell_type == "triangle":
        if degree == 1:
            return ((sp.Rational(1, 3), sp.Rational(1, 3)),), (sp.Rational(1, 2),), sp.Rational(1, 2)
        return ((sp.Rational(1, 6), sp.Rational(1, 6)),
                (sp.Rational(2, 3), sp.Rational(1, 6)),
                (sp.Rational(1, 6), sp.Rational(2, 3))), (sp.Rational(1, 6),) * 3, sp.Rational(1, 2)
    if degree != 1:
        raise ValueError("tetrahedron G.3 quadrature supports exact degree 1 only")
    return ((sp.Rational(1, 4),) * 3,), (sp.Rational(1, 6),), sp.Rational(1, 6)


def _monomial_integral(exponents):
    numerator = sp.prod(sp.factorial(value) for value in exponents)
    return sp.simplify(numerator / sp.factorial(sum(exponents) + len(exponents)))


def make_quadrature(reference: ReferenceElement, reference_id: str, degree: int):
    points, weights, measure = _quadrature_data(reference.cell_type, degree)
    output = QuadratureRule(reference_element_id=reference_id,
        cell_type=reference.cell_type, dimension=reference.dimension,
        degree_exact=degree, points=points, weights=weights,
        reference_measure=measure, input_trust=reference.input_trust)
    exact = True
    for exponents in product(range(degree + 1), repeat=reference.dimension):
        if sum(exponents) > degree: continue
        estimate = sum(weight * sp.prod(point[index] ** exponent
            for index, exponent in enumerate(exponents))
            for point, weight in zip(points, weights))
        exact &= sp.simplify(estimate - _monomial_integral(exponents)) == 0
    checks = {"weights_sum_to_reference_measure": sp.simplify(sum(weights) - measure) == 0,
              "monomials_through_requested_degree_exact": exact,
              "weights_positive": all(weight.is_positive is True for weight in weights)}
    return EngineeringResult(operation="quadrature", status="verified" if all(checks.values()) else "refuted",
        trust=reference.input_trust, value={"point_count": len(points),
            "degree_exact": degree, "reference_measure": measure}, verification=checks,
        details={"reference_cell_only": True, "assembly": "not_performed"},
        claim_evidence={"quadrature": _bundle("exact_monomial_moment_replay",
            reference.input_trust, [])}), output


def verify_quadrature(reference: ReferenceElement, result: QuadratureRule):
    _, candidate = make_quadrature(reference, result.reference_element_id, result.degree_exact)
    checks = {"points_recomputed": candidate.points == result.points,
              "weights_recomputed": candidate.weights == result.weights,
              "measure_recomputed": candidate.reference_measure == result.reference_measure}
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted",
        trust=cap_trust(reference.input_trust, result.input_trust),
        value={"checks": checks}, verification=checks,
        details={"reference_cell_only": True, "assembly": "not_performed"},
        claim_evidence={"verify": _bundle("quadrature_replay",
            cap_trust(reference.input_trust, result.input_trust), [])})


def make_space(mesh: FEMMesh, mesh_id: str, weak: WeakForm,
               reference: ReferenceElement, reference_id: str,
               basis: BasisFunctionSet, basis_id: str, field: str,
               problem: PDEProblem):
    if mesh.orientation_status != "positive":
        raise ValueError("ambiguous mesh orientation cannot produce a finite-element space")
    if reference.mesh_id != mesh_id or basis.reference_element_id != reference_id:
        raise ValueError("finite-element source ancestry is inconsistent")
    if field not in {space.field for space in weak.trial_spaces}:
        raise ValueError("finite-element field is absent from the weak trial spaces")
    essential = set()
    bounds = {name: (lower, upper) for name, lower, upper in problem.domain}
    axis_for = {name: index for index, name in enumerate(weak.integration_variables)}
    for boundary_index in weak.essential_boundary_indices:
        condition = problem.boundary_conditions[boundary_index]
        if condition.field != field or condition.coordinate not in axis_for: continue
        axis = axis_for[condition.coordinate]
        target = bounds[condition.coordinate][condition.side == "upper"]
        for index, point in enumerate(mesh.points):
            if sp.simplify(point[axis] - target) == 0:
                essential.add(index)
    output = FiniteElementSpace(mesh_id=mesh_id,
        reference_element_id=reference_id, basis_id=basis_id,
        weak_form_id=mesh.weak_form_id, field=field,
        local_to_global=mesh.cells, dof_coordinates=mesh.points,
        essential_dofs=tuple(sorted(essential)), dof_count=len(mesh.points),
        input_trust=cap_trust(mesh.input_trust, reference.input_trust,
                              basis.input_trust, weak.input_trust))
    checks = {"one_vertex_dof_per_p1_basis_node": all(
                  len(row) == len(basis.functions) for row in output.local_to_global),
              "global_dofs_in_range": all(0 <= index < output.dof_count
                  for row in output.local_to_global for index in row),
              "essential_dofs_recomputed": True,
              "combinatorial_c0_conformity": True,
              "geometric_nonoverlap": None}
    return EngineeringResult(operation="finite_element_space", status="unknown",
        trust=output.input_trust, value={"dof_count": output.dof_count,
            "essential_dofs": output.essential_dofs,
            "local_to_global": output.local_to_global}, verification=checks,
        details={"conformity": "vertex_P1_combinatorial",
                 "geometric_nonoverlap": "not_established", "assembly": "not_performed"},
        claim_evidence={"finite_element_space": _bundle(
            "p1_vertex_dof_numbering", output.input_trust,
            ["C0 conformity is combinatorial on the supplied simplex complex"])}), output


def verify_space(mesh: FEMMesh, weak: WeakForm, reference: ReferenceElement,
                 basis: BasisFunctionSet, problem: PDEProblem,
                 result: FiniteElementSpace):
    _, candidate = make_space(mesh, result.mesh_id, weak, reference,
        result.reference_element_id, basis, result.basis_id, result.field, problem)
    checks = {"local_to_global_recomputed": candidate.local_to_global == result.local_to_global,
              "dof_coordinates_recomputed": candidate.dof_coordinates == result.dof_coordinates,
              "essential_dofs_recomputed": candidate.essential_dofs == result.essential_dofs,
              "dof_count_recomputed": candidate.dof_count == result.dof_count,
              "source_chain_recomputed": candidate.weak_form_id == result.weak_form_id}
    return EngineeringResult(operation="verify",
        status="verified" if all(checks.values()) else "refuted",
        trust=cap_trust(candidate.input_trust, result.input_trust),
        value={"checks": checks}, verification=checks,
        details={"conformity": "vertex_P1_combinatorial",
                 "geometric_nonoverlap": "not_established", "assembly": "not_performed"},
        claim_evidence={"verify": _bundle("finite_element_space_replay",
            cap_trust(candidate.input_trust, result.input_trust), [])})
