# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
import asyncio
import pytest

import sympy as sp

from mathkernel import AssembledSystem, FEMSolution, MathKernel, Settings


POINTS = [[0, 0], [1, 0], [1, 1], [0, 1], ["1/2", "1/2"]]
CELLS = [[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]]


def problem_definition(*, source="f", conditions=None, coefficient=-1):
    return {
        "fields": ["u"], "independent_variables": ["x", "y"],
        "domain": {"x": [0, 1], "y": [0, 1]},
        "parameters": {"f": None},
        "equations": [{"terms": [
            {"coefficient": coefficient, "field": "u", "derivative": {"x": 2}},
            {"coefficient": -1, "field": "u", "derivative": {"y": 2}},
        ], "source": source}],
        "boundary_conditions": conditions if conditions is not None else [
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "upper", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "upper", "value": 0},
        ],
    }


def weak_parameters(conditions):
    essential = [index for index, item in enumerate(conditions) if item["kind"] == "dirichlet"]
    family = "H1_0" if len(essential) == len(conditions) else "H1_D" if essential else "H1"
    return {
        "integration_variables": ["x", "y"],
        "trial_spaces": [{"name": "U", "field": "u", "family": family,
            "regularity_order": 1, "trace_boundary_indices": essential}],
        "test_space": {"name": "v", "field": "u", "family": family,
            "regularity_order": 1, "trace_boundary_indices": essential},
        "integration_by_parts": [{"term_index": 0, "coordinate": "x"},
                                  {"term_index": 1, "coordinate": "y"}],
    }


def discrete_chain(kernel, *, definition=None, quadrature_degree=2):
    definition = definition or problem_definition()
    problem = kernel.object_create("PDEProblem", definition)
    assert problem.ok, problem.errors
    weak = kernel.apply(problem.data["object_id"], "derive_weak_form",
                        weak_parameters(definition["boundary_conditions"]))
    assert weak.ok, weak.errors
    mesh = kernel.object_create("FEMMesh", {"weak_form_id": weak.data["object_id"],
        "cell_type": "triangle", "points": POINTS, "cells": CELLS})
    reference = kernel.apply(mesh.data["object_id"], "reference_element")
    basis = kernel.apply(reference.data["object_id"], "basis")
    quadrature = kernel.apply(reference.data["object_id"], "quadrature",
                              {"degree_exact": quadrature_degree})
    space = kernel.apply(mesh.data["object_id"], "finite_element_space", {
        "reference_element_id": reference.data["object_id"],
        "basis_id": basis.data["object_id"], "field": "u"})
    return problem, weak, mesh, reference, basis, quadrature, space


def assemble(kernel, *, definition=None, substitutions=None, degree=2):
    chain = discrete_chain(kernel, definition=definition, quadrature_degree=degree)
    quadrature, space = chain[-2:]
    result = kernel.apply(space.data["object_id"], "assemble", {
        "quadrature_id": quadrature.data["object_id"],
        "substitutions": substitutions or {}})
    return (*chain, result)


def test_exact_poisson_sparse_assembly_and_solution():
    kernel = MathKernel(); *_, system = assemble(kernel, substitutions={"f": 1})
    assert system.ok and system.status == "verified"
    value = kernel.math_objects[system.data["object_id"]]["value"]
    assert isinstance(value, AssembledSystem)
    assert value.raw_rhs == (sp.Rational(1, 6),) * 4 + (sp.Rational(1, 3),)
    assert {(entry.row, entry.column): entry.value for entry in value.raw_matrix_entries} == {
        (0, 0): 1, (0, 4): -1, (1, 1): 1, (1, 4): -1,
        (2, 2): 1, (2, 4): -1, (3, 3): 1, (3, 4): -1,
        (4, 0): -1, (4, 1): -1, (4, 2): -1, (4, 3): -1, (4, 4): 4}
    assert value.rhs == (0, 0, 0, 0, sp.Rational(1, 3))
    assert value.quadrature_exact and value.raw_matrix_symmetric
    solution = kernel.apply(system.data["object_id"], "solve")
    assert solution.ok and solution.status == "verified"
    solved = kernel.math_objects[solution.data["object_id"]]["value"]
    assert isinstance(solved, FEMSolution)
    assert solved.solve_status == "unique" and solved.values == (0, 0, 0, 0, sp.Rational(1, 12))
    assert solved.residual_norm == 0 and solved.rank == solved.augmented_rank == 5


def test_assembly_and_solution_have_complete_direct_and_transitive_ancestry():
    kernel = MathKernel(); *chain, system = assemble(kernel, substitutions={"f": 1})
    quadrature, space = chain[-2:]
    system_id = system.data["object_id"]
    assert kernel.math_objects[system_id]["sources"] == [
        space.data["object_id"], quadrature.data["object_id"]]
    solution = kernel.apply(system_id, "solve")
    assert kernel.math_objects[solution.data["object_id"]]["sources"] == [system_id]
    assert kernel.apply(system_id, "verify").status == "verified"
    assert kernel.apply(solution.data["object_id"], "verify").status == "verified"


def test_inhomogeneous_dirichlet_values_are_symmetrically_eliminated():
    conditions = [
        {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
        {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "upper", "value": 1},
        {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "lower", "value": "x"},
        {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "upper", "value": "x"},
    ]
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source=0, conditions=conditions))
    solved = kernel.apply(system.data["object_id"], "solve")
    values = kernel.math_objects[solved.data["object_id"]]["value"].values
    assert values == (0, 1, 1, 0, sp.Rational(1, 2))


def test_neumann_facets_are_retained_and_integrated():
    conditions = [
        {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "x", "side": "upper", "value": 1},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "upper", "value": 0},
    ]
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source=0, conditions=conditions))
    value = kernel.math_objects[system.data["object_id"]]["value"]
    assert len(value.natural_contributions) == 3
    solved = kernel.apply(system.data["object_id"], "solve")
    assert kernel.math_objects[solved.data["object_id"]]["value"].values == (
        0, 1, 1, 0, sp.Rational(1, 2))


def test_robin_boundary_adds_matrix_and_requires_sufficient_quadrature():
    conditions = [
        {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
        {"field": "u", "kind": "robin", "coordinate": "x", "side": "upper",
         "value": 2, "alpha": 1, "beta": 1},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "upper", "value": 0},
    ]
    definition = problem_definition(source=0, conditions=conditions)
    kernel = MathKernel(); *_, system = assemble(kernel, definition=definition)
    value = kernel.math_objects[system.data["object_id"]]["value"]
    assert value.quadrature_exact
    solved = kernel.apply(system.data["object_id"], "solve")
    assert kernel.math_objects[solved.data["object_id"]]["value"].values == (
        0, 1, 1, 0, sp.Rational(1, 2))
    kernel = MathKernel(); *_, low_order = assemble(kernel, definition=definition, degree=1)
    assert low_order.ok and low_order.status == "verified"
    assert not kernel.math_objects[low_order.data["object_id"]]["value"].quadrature_exact


def test_unresolved_parameters_and_boundary_fluxes_fail_closed_atomically():
    kernel = MathKernel(); *chain, _ = assemble(kernel, substitutions={"f": 1})
    quadrature, space = chain[-2:]
    before = len(kernel.math_objects)
    unresolved = kernel.apply(space.data["object_id"], "assemble", {
        "quadrature_id": quadrature.data["object_id"]})
    assert not unresolved.ok and "unresolved" in unresolved.errors[0]
    assert len(kernel.math_objects) == before
    definition = problem_definition(source=0, conditions=[])
    kernel = MathKernel(); *_, result = assemble(kernel, definition=definition)
    assert not result.ok and "boundary flux" in result.errors[0]


def test_stationary_linear_scope_refuses_strong_second_derivatives_and_periodic_constraints():
    kernel = MathKernel(); definition = problem_definition(source=0)
    problem = kernel.object_create("PDEProblem", definition)
    parameters = weak_parameters(definition["boundary_conditions"])
    parameters["integration_by_parts"] = []
    parameters["trial_spaces"][0]["regularity_order"] = 2
    parameters["trial_spaces"][0]["family"] = "H2"
    weak = kernel.apply(problem.data["object_id"], "derive_weak_form", parameters)
    mesh = kernel.object_create("FEMMesh", {"weak_form_id": weak.data["object_id"],
        "cell_type": "triangle", "points": POINTS, "cells": CELLS})
    ref = kernel.apply(mesh.data["object_id"], "reference_element")
    basis = kernel.apply(ref.data["object_id"], "basis")
    quad = kernel.apply(ref.data["object_id"], "quadrature", {"degree_exact": 2})
    space = kernel.apply(mesh.data["object_id"], "finite_element_space", {
        "reference_element_id": ref.data["object_id"], "basis_id": basis.data["object_id"], "field": "u"})
    result = kernel.apply(space.data["object_id"], "assemble", {"quadrature_id": quad.data["object_id"]})
    assert not result.ok and "first-order" in result.errors[0]
    periodic = [
        {"field": "u", "kind": "periodic", "coordinate": "x", "side": "lower", "value": 0,
         "paired_side": "upper"},
        {"field": "u", "kind": "periodic", "coordinate": "x", "side": "upper", "value": 0,
         "paired_side": "lower"},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "upper", "value": 0},
    ]
    kernel = MathKernel(); *_, result = assemble(kernel,
        definition=problem_definition(source=0, conditions=periodic))
    assert not result.ok and "periodic" in result.errors[0]


def test_cross_reference_quadrature_is_refused_atomically():
    kernel = MathKernel(); *first, _ = assemble(kernel, substitutions={"f": 1})
    *_, other_quadrature, _ = discrete_chain(kernel)
    space = first[-1]
    before = len(kernel.math_objects)
    result = kernel.apply(space.data["object_id"], "assemble", {
        "quadrature_id": other_quadrature.data["object_id"], "substitutions": {"f": 1}})
    assert not result.ok and "reference element" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_exact_singular_outcomes_are_explicit_artifacts():
    conditions = [
        {"field": "u", "kind": "neumann", "coordinate": "x", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "x", "side": "upper", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "y", "side": "upper", "value": 0},
    ]
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source=0, conditions=conditions))
    solution = kernel.apply(system.data["object_id"], "solve")
    value = kernel.math_objects[solution.data["object_id"]]["value"]
    assert solution.ok and solution.status == "verified"
    assert value.solve_status == "singular_underdetermined"
    assert value.rank == 4 and value.augmented_rank == 4
    conditions[1] = {**conditions[1], "value": 1}
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source=0, conditions=conditions))
    solution = kernel.apply(system.data["object_id"], "solve")
    value = kernel.math_objects[solution.data["object_id"]]["value"]
    assert value.solve_status == "singular_inconsistent"
    assert value.augmented_rank > value.rank


def test_numeric_sparse_solve_reports_residual_and_conditioning():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source="1.0", coefficient="-1.0"))
    solution = kernel.apply(system.data["object_id"], "solve", {
        "method": "numeric", "condition_limit": 2.0, "tolerance": 1e-10})
    value = kernel.math_objects[solution.data["object_id"]]["value"]
    assert value.method == "numeric" and value.solve_status == "ill_conditioned"
    assert float(value.residual_norm) <= 1e-10
    assert value.condition_number is not None and value.condition_number > 2


def test_output_only_types_tamper_and_source_substitution_fail_closed():
    kernel = MathKernel()
    assert not kernel.object_create("AssembledSystem", {}).ok
    assert not kernel.object_create("FEMSolution", {}).ok
    *_, system = assemble(kernel, substitutions={"f": 1})
    solution = kernel.apply(system.data["object_id"], "solve")
    system_id = system.data["object_id"]
    original = kernel.math_objects[system_id]["value"]
    bad_rhs = (*original.rhs[:-1], sp.Integer(99))
    kernel.math_objects[system_id]["value"] = original.model_copy(update={"rhs": bad_rhs})
    assert kernel.apply(system_id, "verify").status == "refuted"
    solution_id = solution.data["object_id"]
    original_solution = kernel.math_objects[solution_id]["value"]
    kernel.math_objects[solution_id]["value"] = original_solution.model_copy(
        update={"values": (sp.Integer(99),) * 5})
    assert kernel.apply(solution_id, "verify").status == "refuted"


def test_assembled_source_envelope_substitution_is_rejected_before_execution():
    kernel = MathKernel(); *_, system = assemble(kernel, substitutions={"f": 1})
    system_id = system.data["object_id"]
    kernel.math_objects[system_id]["sources"] = list(reversed(
        kernel.math_objects[system_id]["sources"]))
    before = len(kernel.math_objects)
    result = kernel.apply(system_id, "verify")
    assert not result.ok and "ancestry" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_persistence_restarts_assembly_and_solution_replay(tmp_path):
    settings = Settings(store_path=str(tmp_path / "g4.sqlite"))
    kernel = MathKernel(settings); *_, system = assemble(kernel, substitutions={"f": 1})
    solution = kernel.apply(system.data["object_id"], "solve")
    restarted = MathKernel(settings)
    assert restarted.apply(system.data["object_id"], "verify").status == "verified"
    assert restarted.apply(solution.data["object_id"], "verify").status == "verified"


def test_current_assembly_and_solve_limits_are_atomic():
    kernel = MathKernel(Settings(max_fem_assembly_nnz=1))
    *_, system = assemble(kernel, substitutions={"f": 1})
    assert not system.ok and "max_fem_assembly_nnz" in system.errors[0]
    kernel = MathKernel(Settings(max_fem_exact_solve_dofs=4))
    *_, system = assemble(kernel, substitutions={"f": 1})
    before = len(kernel.math_objects)
    solution = kernel.apply(system.data["object_id"], "solve")
    assert not solution.ok and len(kernel.math_objects) == before


def test_capabilities_expose_assembly_scope_and_limits():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="pde")
    assert manifest["count"] >= 20
    names = {item["name"] for item in manifest["capabilities"]}
    assert {"pde.FiniteElementSpace.assemble", "pde.AssembledSystem.verify",
            "pde.AssembledSystem.solve", "pde.FEMSolution.verify"} <= names
    overview = kernel.capabilities()["pde"]
    assert overview["max_fem_assembly_nnz"] == 2_000_000
    assert "continuous PDE solution" in overview["claims_excluded"]


def test_generic_mcp_tools_execute_assembly_and_solve():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        kernel = MathKernel(); *chain, system = assemble(kernel, substitutions={"f": 1})
        # MCP uses its own kernel, so recreate through generic calls compactly.
        definition = problem_definition()
        async with Client(mcp) as client:
            p = await client.call_tool("math_object_create", {"object_type": "PDEProblem", "definition": definition})
            w = await client.call_tool("math_apply", {"object_id": p.data["data"]["object_id"],
                "operation": "derive_weak_form", "parameters": weak_parameters(definition["boundary_conditions"])})
            m = await client.call_tool("math_object_create", {"object_type": "FEMMesh", "definition": {
                "weak_form_id": w.data["data"]["object_id"], "cell_type": "triangle",
                "points": POINTS, "cells": CELLS}})
            r = await client.call_tool("math_apply", {"object_id": m.data["data"]["object_id"],
                "operation": "reference_element", "parameters": {}})
            b = await client.call_tool("math_apply", {"object_id": r.data["data"]["object_id"],
                "operation": "basis", "parameters": {}})
            q = await client.call_tool("math_apply", {"object_id": r.data["data"]["object_id"],
                "operation": "quadrature", "parameters": {"degree_exact": 2}})
            s = await client.call_tool("math_apply", {"object_id": m.data["data"]["object_id"],
                "operation": "finite_element_space", "parameters": {
                    "reference_element_id": r.data["data"]["object_id"],
                    "basis_id": b.data["data"]["object_id"], "field": "u"}})
            a = await client.call_tool("math_apply", {"object_id": s.data["data"]["object_id"],
                "operation": "assemble", "parameters": {
                    "quadrature_id": q.data["data"]["object_id"], "substitutions": {"f": 1}}})
            sol = await client.call_tool("math_apply", {"object_id": a.data["data"]["object_id"],
                "operation": "solve", "parameters": {}})
            return a.data, sol.data
    system, solution = asyncio.run(scenario())
    assert system["ok"] and solution["ok"] and solution["status"] == "verified"

