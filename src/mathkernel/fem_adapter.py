# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted adapter for Phase G.3 finite-element representation artifacts."""
from __future__ import annotations

import sympy as sp

from .engineering import arithmetic_trust, cap_trust
from .models import TrustLevel


TYPES = {"femmesh": "FEMMesh", "fem_mesh": "FEMMesh"}

OPERATIONS = {
    "FEMMesh": {
        "verify": {},
        "reference_element": {},
        "finite_element_space": {
            "reference_element_id": "ReferenceElement object id",
            "basis_id": "BasisFunctionSet object id",
            "field": "declared weak-form field",
        },
    },
    "ReferenceElement": {
        "verify": {}, "basis": {},
        "quadrature": {"degree_exact": "integer 1 or 2 (default 2)"},
    },
    "BasisFunctionSet": {"verify": {}},
    "QuadratureRule": {"verify": {}},
    "FiniteElementSpace": {
        "verify": {},
        "assemble": {
            "quadrature_id": "QuadratureRule object id",
            "substitutions": "concrete values for unresolved PDE parameters (default {})",
        },
    },
    "AssembledSystem": {
        "verify": {},
        "solve": {
            "method": "auto, exact, or numeric (default auto)",
            "tolerance": "positive finite residual/rank tolerance (default 1e-10)",
            "condition_limit": "positive finite ill-conditioning threshold (default 1e12)",
        },
    },
    "FEMSolution": {"verify": {}, "estimate_error": {}},
    "FEMErrorEstimate": {
        "verify": {},
        "mark": {
            "strategy": "dorfler or maximum (default dorfler)",
            "theta": "finite fraction in (0, 1] (default 0.5)",
        },
        "compare": {"coarse_estimate_id": "FEMErrorEstimate object id"},
    },
    "RefinementMarking": {"verify": {}, "refine": {}},
    "RefinedMesh": {
        "verify": {},
        "reference_element": {},
        "finite_element_space": {
            "reference_element_id": "ReferenceElement object id",
            "basis_id": "BasisFunctionSet object id",
            "field": "declared weak-form field",
        },
    },
    "MeshTransfer": {"verify": {}},
    "FEMConvergenceObservation": {"verify": {}},
}

DERIVED_OUTPUTS = {
    "reference_element": "ReferenceElement",
    "basis": "BasisFunctionSet",
    "quadrature": "QuadratureRule",
    "finite_element_space": "FiniteElementSpace",
    "assemble": "AssembledSystem",
    "solve": "FEMSolution",
    "estimate_error": "FEMErrorEstimate",
    "mark": "RefinementMarking",
    "refine": "RefinedMesh",
    "compare": "FEMConvergenceObservation",
}


def _record(kernel, object_id, expected):
    record = kernel._get_math_object_record(str(object_id))
    if record is None or record["object_type"] != expected:
        raise ValueError(f"{object_id} must reference a stored {expected}")
    return record


def _mesh_record(kernel, object_id):
    record = kernel._get_math_object_record(str(object_id))
    if record is None or record["object_type"] not in {"FEMMesh", "RefinedMesh"}:
        raise ValueError(f"{object_id} must reference a stored FEMMesh or RefinedMesh")
    return record


def _weak_problem(kernel, weak_id):
    weak_record = _record(kernel, weak_id, "WeakForm")
    weak = weak_record["value"]
    problem_record = _record(kernel, weak.problem_id, "PDEProblem")
    return weak, problem_record["value"]


def _check_mesh_limits(kernel, mesh):
    work = len(mesh.cells) * (len(mesh.cells[0]) ** 3 + len(mesh.cells[0]))
    if (len(mesh.points) > kernel.settings.max_fem_points or
            len(mesh.cells) > kernel.settings.max_fem_cells or
            work > kernel.settings.max_fem_work):
        raise ValueError("mesh operation exceeds current FEM limits")


def _assembly_sources(kernel, space):
    weak, problem = _weak_problem(kernel, space.weak_form_id)
    mesh = _mesh_record(kernel, space.mesh_id)["value"]
    reference = _record(kernel, space.reference_element_id, "ReferenceElement")["value"]
    basis = _record(kernel, space.basis_id, "BasisFunctionSet")["value"]
    return problem, weak, mesh, reference, basis


def _parse_substitutions(kernel, problem, raw):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or any(not isinstance(name, str) for name in raw):
        raise ValueError("substitutions must map PDE parameter names to MathIR scalars")
    unresolved = {name for name, value in problem.parameters if value is None}
    unknown = set(raw) - unresolved
    if unknown:
        raise ValueError(f"assembly substitution is not an unresolved PDE parameter: {sorted(unknown)[0]}")
    texts = [str(raw[name]) for name in sorted(raw)]
    if sum(map(len, texts)) > kernel.settings.max_input_length:
        raise ValueError("assembly substitutions exceed max_input_length")
    if not texts:
        return (), TrustLevel.EXACT
    _, values, parsed_trust = kernel._typed_parse_many(texts, None)
    if any(value.free_symbols for value in values):
        raise ValueError("assembly parameter substitutions must be concrete")
    trust = TrustLevel(cap_trust(parsed_trust.value,
        arithmetic_trust(values, parsed_trust.value)))
    return tuple(zip(sorted(raw), values)), trust


def construct(kernel, kind, definition):
    from .finite_elements import build_mesh
    from .weak_forms import verify_weak_form

    data = dict(definition)
    if data.pop("context_id", None) is not None:
        raise ValueError("FEMMesh uses the weak form's declared scope")
    allowed = {"weak_form_id", "cell_type", "points", "cells", "triangulation_id"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown FEMMesh field: {sorted(unknown)[0]}")
    weak_id = data.get("weak_form_id")
    if not isinstance(weak_id, str):
        raise ValueError("weak_form_id must be a stored WeakForm id")
    weak_record = _record(kernel, weak_id, "WeakForm")
    weak, problem = _weak_problem(kernel, weak_id)
    if verify_weak_form(problem, weak).status != "verified":
        raise ValueError("weak-form source failed exact replay verification")
    cell_type = data.get("cell_type")
    if not isinstance(cell_type, str) or cell_type.lower() not in {"interval", "triangle", "tetrahedron"}:
        raise ValueError("cell_type must be interval, triangle, or tetrahedron")
    cell_type = cell_type.lower()
    triangulation_id = data.get("triangulation_id")
    sources = [weak_id]
    if triangulation_id is not None:
        if data.get("points") is not None or data.get("cells") is not None:
            raise ValueError("supply triangulation_id or points/cells, not both")
        if cell_type != "triangle":
            raise ValueError("Triangulation sources produce triangle meshes")
        tri_record = _record(kernel, triangulation_id, "Triangulation")
        triangulation = tri_record["value"]
        from .computational_geometry import verify_triangulation
        if verify_triangulation(triangulation).status != "verified":
            raise ValueError("triangulation source failed exact replay verification")
        points = triangulation.points; cells = triangulation.triangles
        trust = TrustLevel(cap_trust(weak_record["input_trust"].value,
                                     tri_record["input_trust"].value))
        sources.append(triangulation_id)
    else:
        raw_points, raw_cells = data.get("points"), data.get("cells")
        if (not isinstance(raw_points, (list, tuple)) or not raw_points or
                len(raw_points) > kernel.settings.max_fem_points or
                not isinstance(raw_cells, (list, tuple)) or not raw_cells or
                len(raw_cells) > kernel.settings.max_fem_cells):
            raise ValueError("mesh points/cells are empty or exceed configured limits")
        dimension = {"interval": 1, "triangle": 2, "tetrahedron": 3}[cell_type]
        if any(not isinstance(point, (list, tuple)) or len(point) != dimension
               for point in raw_points):
            raise ValueError("mesh point dimension does not match cell_type")
        if any(not isinstance(cell, (list, tuple)) or any(
                isinstance(index, bool) or not isinstance(index, int) for index in cell)
               for cell in raw_cells):
            raise ValueError("mesh cells must contain integer indices")
        raw = [str(value) for point in raw_points for value in point]
        if sum(len(value) for value in raw) > kernel.settings.max_input_length:
            raise ValueError("mesh coordinate input exceeds max_input_length")
        _, parsed, parsed_trust = kernel._typed_parse_many(raw, None)
        iterator = iter(parsed)
        points = tuple(tuple(next(iterator) for _ in range(dimension)) for _ in raw_points)
        cells = tuple(tuple(cell) for cell in raw_cells)
        trust = TrustLevel(cap_trust(
            weak_record["input_trust"].value, parsed_trust.value,
            arithmetic_trust(parsed, parsed_trust.value)))
    if len(points) > kernel.settings.max_fem_points or len(cells) > kernel.settings.max_fem_cells:
        raise ValueError("mesh source exceeds configured point/cell limits")
    work = len(cells) * (len(cells[0]) ** 3 + len(cells[0]))
    if work > kernel.settings.max_fem_work:
        raise ValueError("mesh validation exceeds max_fem_work")
    verdict, mesh = build_mesh(weak, weak_id, tuple(points), tuple(cells),
        cell_type, trust.value, triangulation_id=triangulation_id)
    # Points must lie inside the rectangular PDE coordinates used by the weak measure.
    domain = {name: (lower, upper) for name, lower, upper in problem.domain}
    for point in mesh.points:
        for axis, name in enumerate(weak.integration_variables):
            lower, upper = domain[name]
            left = sp.simplify(point[axis] - lower)
            right = sp.simplify(upper - point[axis])
            if left.is_negative is True or right.is_negative is True:
                raise ValueError("mesh point lies outside the represented PDE domain")
    return "FEMMesh", mesh, trust, sources


def apply(kernel, value, operation, parameters, object_id):
    from . import finite_elements as fem

    object_type = type(value).__name__
    allowed = set(OPERATIONS[object_type][operation]) | {"context_id"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unsupported operation parameter: {sorted(unknown)[0]}")
    if parameters.get("context_id") is not None:
        raise ValueError("finite-element operations use stored source scopes")
    if object_type in {"FEMMesh", "RefinedMesh"}:
        weak, problem = _weak_problem(kernel, value.weak_form_id)
        _check_mesh_limits(kernel, value)
        if operation == "verify":
            if object_type == "FEMMesh":
                return fem.verify_mesh(value)
            from . import finite_element_adaptivity as adapt
            marking = _record(kernel, value.marking_id, "RefinementMarking")["value"]
            parent = _mesh_record(kernel, value.parent_mesh_id)["value"]
            if marking.mesh_id != value.parent_mesh_id:
                raise ValueError("refined-mesh parent and marking sources do not reconcile")
            return adapt.verify_refined_mesh(weak, parent, marking, value)
        if operation == "reference_element": return fem.make_reference(value, object_id)
        reference_id = parameters.get("reference_element_id")
        basis_id = parameters.get("basis_id")
        field = parameters.get("field")
        if not all(isinstance(item, str) for item in (reference_id, basis_id, field)):
            raise ValueError("finite_element_space requires reference_element_id, basis_id, and field")
        if len(value.points) > kernel.settings.max_fem_dofs:
            raise ValueError("finite-element space exceeds max_fem_dofs")
        reference = _record(kernel, reference_id, "ReferenceElement")["value"]
        basis = _record(kernel, basis_id, "BasisFunctionSet")["value"]
        return fem.make_space(value, object_id, weak, reference, reference_id,
                              basis, basis_id, field, problem)
    if object_type == "ReferenceElement":
        mesh = _mesh_record(kernel, value.mesh_id)["value"]
        _check_mesh_limits(kernel, mesh)
        if operation == "verify": return fem.verify_reference(mesh, value)
        if operation == "basis": return fem.make_basis(value, object_id)
        degree = parameters.get("degree_exact", 2)
        if isinstance(degree, bool) or not isinstance(degree, int) or degree not in {1, 2}:
            raise ValueError("degree_exact must be 1 or 2")
        return fem.make_quadrature(value, object_id, degree)
    if object_type == "BasisFunctionSet":
        reference = _record(kernel, value.reference_element_id, "ReferenceElement")["value"]
        mesh = _mesh_record(kernel, reference.mesh_id)["value"]
        _check_mesh_limits(kernel, mesh)
        return fem.verify_basis(reference, value)
    if object_type == "QuadratureRule":
        reference = _record(kernel, value.reference_element_id, "ReferenceElement")["value"]
        mesh = _mesh_record(kernel, reference.mesh_id)["value"]
        _check_mesh_limits(kernel, mesh)
        return fem.verify_quadrature(reference, value)
    if object_type == "AssembledSystem":
        from . import finite_element_assembly as assembly
        space = _record(kernel, value.finite_element_space_id, "FiniteElementSpace")["value"]
        problem, weak, mesh, reference, basis = _assembly_sources(kernel, space)
        quadrature = _record(kernel, value.quadrature_id, "QuadratureRule")["value"]
        _check_mesh_limits(kernel, mesh)
        if len(value.matrix_entries) > kernel.settings.max_fem_assembly_nnz:
            raise ValueError("assembled system exceeds current sparse-entry limits")
        if operation == "verify":
            return assembly.verify_system(problem, weak, mesh, space, basis,
                                          quadrature, value)
        method = parameters.get("method", "auto")
        tolerance = parameters.get("tolerance", 1e-10)
        condition_limit = parameters.get("condition_limit", 1e12)
        if method not in {"auto", "exact", "numeric"}:
            raise ValueError("method must be auto, exact, or numeric")
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
                not 0 < float(tolerance) < float("inf")):
            raise ValueError("tolerance must be positive and finite")
        if (isinstance(condition_limit, bool) or not isinstance(condition_limit, (int, float)) or
                not 1 < float(condition_limit) < float("inf")):
            raise ValueError("condition_limit must be finite and greater than one")
        has_float = any(entry.value.has(sp.Float) for entry in value.matrix_entries) or any(
            item.has(sp.Float) for item in value.rhs)
        selected = "numeric" if method == "numeric" or (method == "auto" and has_float) else "exact"
        maximum = (kernel.settings.max_fem_exact_solve_dofs if selected == "exact"
                   else kernel.settings.max_fem_numeric_solve_dofs)
        if value.dof_count > maximum:
            raise ValueError(f"{selected} FEM solve exceeds its configured DOF limit")
        return assembly.solve_system(value, object_id, method, float(tolerance),
                                     float(condition_limit))
    if object_type == "FEMSolution":
        from . import finite_element_assembly as assembly
        system = _record(kernel, value.assembled_system_id, "AssembledSystem")["value"]
        maximum = (kernel.settings.max_fem_exact_solve_dofs if value.method == "exact"
                   else kernel.settings.max_fem_numeric_solve_dofs)
        if system.dof_count > maximum or len(system.matrix_entries) > kernel.settings.max_fem_assembly_nnz:
            raise ValueError("FEM solution replay exceeds current limits")
        if operation == "verify":
            return assembly.verify_solution(system, value)
        from . import finite_element_adaptivity as adapt
        space = _record(kernel, system.finite_element_space_id, "FiniteElementSpace")["value"]
        problem, weak, mesh, _, _ = _assembly_sources(kernel, space)
        quadrature = _record(kernel, system.quadrature_id, "QuadratureRule")["value"]
        work = len(mesh.cells) * (len(quadrature.points) + len(mesh.interior_facets) + 1)
        if work > kernel.settings.max_fem_estimator_work:
            raise ValueError("error estimation exceeds max_fem_estimator_work")
        return adapt.estimate_error(problem, weak, mesh, space, system, value,
                                    object_id, quadrature)
    if object_type == "FEMErrorEstimate":
        from . import finite_element_adaptivity as adapt
        solution = _record(kernel, value.solution_id, "FEMSolution")["value"]
        system = _record(kernel, value.assembled_system_id, "AssembledSystem")["value"]
        space = _record(kernel, value.finite_element_space_id, "FiniteElementSpace")["value"]
        problem, weak, mesh, _, _ = _assembly_sources(kernel, space)
        quadrature = _record(kernel, system.quadrature_id, "QuadratureRule")["value"]
        if (solution.assembled_system_id != value.assembled_system_id or
                system.finite_element_space_id != value.finite_element_space_id or
                space.mesh_id != value.mesh_id):
            raise ValueError("error-estimate solution/system/space/mesh ancestry does not reconcile")
        if operation == "verify":
            return adapt.verify_estimate(problem, weak, mesh, space, system,
                                         solution, value, quadrature)
        if operation == "mark":
            return adapt.mark(value, object_id, parameters.get("strategy", "dorfler"),
                              parameters.get("theta", 0.5))
        coarse_id = parameters.get("coarse_estimate_id")
        if not isinstance(coarse_id, str):
            raise ValueError("compare requires coarse_estimate_id")
        coarse = _record(kernel, coarse_id, "FEMErrorEstimate")["value"]
        fine_mesh = _mesh_record(kernel, value.mesh_id)["value"]
        coarse_mesh = _mesh_record(kernel, coarse.mesh_id)["value"]
        return adapt.compare(value, object_id, coarse, coarse_id, fine_mesh, coarse_mesh)
    if object_type == "RefinementMarking":
        from . import finite_element_adaptivity as adapt
        estimate = _record(kernel, value.error_estimate_id, "FEMErrorEstimate")["value"]
        mesh = _mesh_record(kernel, value.mesh_id)["value"]
        if estimate.mesh_id != value.mesh_id:
            raise ValueError("refinement marking does not belong to its mesh")
        if operation == "verify":
            return adapt.verify_marking(estimate, value)
        projected_cells = len(mesh.cells) + 3 * len(mesh.cells)
        projected_points = len(mesh.points) + 3 * len(mesh.cells)
        if (projected_cells > kernel.settings.max_fem_refined_cells or
                projected_points > kernel.settings.max_fem_points):
            raise ValueError("refinement exceeds current FEM output limits")
        weak, _ = _weak_problem(kernel, mesh.weak_form_id)
        return adapt.refine(weak, mesh, value, object_id)
    if object_type == "MeshTransfer":
        from . import finite_element_adaptivity as adapt
        marking = _record(kernel, value.marking_id, "RefinementMarking")["value"]
        parent = _mesh_record(kernel, value.parent_mesh_id)["value"]
        weak, _ = _weak_problem(kernel, parent.weak_form_id)
        return adapt.verify_transfer(weak, parent, marking, value)
    if object_type == "FEMConvergenceObservation":
        from . import finite_element_adaptivity as adapt
        fine = _record(kernel, value.fine_estimate_id, "FEMErrorEstimate")["value"]
        coarse = _record(kernel, value.coarse_estimate_id, "FEMErrorEstimate")["value"]
        fine_mesh = _mesh_record(kernel, value.fine_mesh_id)["value"]
        coarse_mesh = _mesh_record(kernel, value.coarse_mesh_id)["value"]
        return adapt.verify_observation(fine, coarse, fine_mesh, coarse_mesh, value)
    mesh = _mesh_record(kernel, value.mesh_id)["value"]
    _check_mesh_limits(kernel, mesh)
    weak, problem = _weak_problem(kernel, value.weak_form_id)
    reference = _record(kernel, value.reference_element_id, "ReferenceElement")["value"]
    basis = _record(kernel, value.basis_id, "BasisFunctionSet")["value"]
    if (len(value.dof_coordinates) > kernel.settings.max_fem_dofs or
            len(value.local_to_global) > kernel.settings.max_fem_cells):
        raise ValueError("finite-element-space replay exceeds current limits")
    if operation == "verify":
        return fem.verify_space(mesh, weak, reference, basis, problem, value)
    from . import finite_element_assembly as assembly
    quadrature_id = parameters.get("quadrature_id")
    if not isinstance(quadrature_id, str):
        raise ValueError("assemble requires quadrature_id")
    quadrature_record = _record(kernel, quadrature_id, "QuadratureRule")
    quadrature = quadrature_record["value"]
    if quadrature.reference_element_id != value.reference_element_id:
        raise ValueError("quadrature does not belong to the finite-element reference element")
    substitutions, substitution_trust = _parse_substitutions(
        kernel, problem, parameters.get("substitutions", {}))
    local_size = len(basis.functions)
    quadrature_points = len(quadrature.points)
    projected_work = (len(mesh.cells) * local_size * local_size *
                      max(1, len(weak.volume_terms)) * quadrature_points)
    projected_nnz = len(mesh.cells) * local_size * local_size
    if projected_work > kernel.settings.max_fem_assembly_work:
        raise ValueError("assembly exceeds max_fem_assembly_work")
    if projected_nnz > kernel.settings.max_fem_assembly_nnz:
        raise ValueError("assembly exceeds max_fem_assembly_nnz")
    source_trust = cap_trust(value.input_trust,
        quadrature_record["input_trust"].value, substitution_trust.value)
    return assembly.assemble_system(problem, weak, mesh, value, object_id,
        basis, quadrature, quadrature_id, substitutions, source_trust)
