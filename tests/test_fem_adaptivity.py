# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
import asyncio
from dataclasses import replace

import sympy as sp

from mathkernel import (FEMConvergenceObservation, FEMErrorEstimate, MathKernel,
                        MeshTransfer, RefinedMesh, RefinementMarking, Settings)
from test_fem_assembly import assemble


def adaptive_chain(kernel):
    *_, system = assemble(kernel, substitutions={"f": 1})
    solution = kernel.apply(system.data["object_id"], "solve")
    estimate = kernel.apply(solution.data["object_id"], "estimate_error")
    marking = kernel.apply(estimate.data["object_id"], "mark", {"theta": 0.5})
    refinement = kernel.apply(marking.data["object_id"], "refine")
    return system, solution, estimate, marking, refinement


def fine_chain(kernel, refinement):
    mesh_id = refinement.data["object_ids"]["mesh"]
    reference = kernel.apply(mesh_id, "reference_element")
    basis = kernel.apply(reference.data["object_id"], "basis")
    quadrature = kernel.apply(reference.data["object_id"], "quadrature", {"degree_exact": 2})
    space = kernel.apply(mesh_id, "finite_element_space", {
        "reference_element_id": reference.data["object_id"],
        "basis_id": basis.data["object_id"], "field": "u"})
    system = kernel.apply(space.data["object_id"], "assemble", {
        "quadrature_id": quadrature.data["object_id"], "substitutions": {"f": 1}})
    solution = kernel.apply(system.data["object_id"], "solve")
    estimate = kernel.apply(solution.data["object_id"], "estimate_error")
    return reference, basis, quadrature, space, system, solution, estimate


def test_exact_local_residual_jump_estimator_and_semantic_boundaries():
    kernel = MathKernel(); _, solution, estimate, _, _ = adaptive_chain(kernel)
    assert estimate.ok and estimate.status == "verified"
    value = kernel.math_objects[estimate.data["object_id"]]["value"]
    assert isinstance(value, FEMErrorEstimate)
    assert value.global_estimator_squared == sp.Rational(10, 9)
    assert value.global_estimator == sp.sqrt(10) / 3
    assert value.algebraic_residual_norm == 0
    assert value.quadrature_exact
    assert not value.rigorous_error_bound
    assert not value.reliability_constant_established
    assert sum(x.total_indicator_squared for x in value.cell_indicators) == value.global_estimator_squared
    assert kernel.math_objects[estimate.data["object_id"]]["sources"] == [solution.data["object_id"]]


def test_dorfler_and_maximum_marking_replay_exact_policy():
    kernel = MathKernel(); _, _, estimate, marking, _ = adaptive_chain(kernel)
    value = kernel.math_objects[marking.data["object_id"]]["value"]
    assert isinstance(value, RefinementMarking)
    assert value.marked_cells == (0, 1) and value.achieved_fraction == 0.5
    assert kernel.apply(marking.data["object_id"], "verify").status == "verified"
    maximum = kernel.apply(estimate.data["object_id"], "mark", {
        "strategy": "maximum", "theta": 1})
    assert kernel.math_objects[maximum.data["object_id"]]["value"].marked_cells == (0, 1, 2, 3)


def test_refinement_has_complete_parent_child_and_transfer_lineage():
    kernel = MathKernel(); _, _, _, marking, refinement = adaptive_chain(kernel)
    assert refinement.ok and refinement.data["object_type"] == "RefinedMesh"
    mesh_id = refinement.data["object_ids"]["mesh"]
    transfer_id = refinement.data["object_ids"]["transfer"]
    mesh = kernel.math_objects[mesh_id]["value"]
    transfer = kernel.math_objects[transfer_id]["value"]
    assert isinstance(mesh, RefinedMesh) and isinstance(transfer, MeshTransfer)
    assert len(mesh.cells) == 16 and len(mesh.parent_cell_indices) == 16
    assert mesh.requested_marked_cells == (0, 1)
    assert mesh.closure_refined_cells == (0, 1, 2, 3)
    assert set(mesh.parent_cell_indices) == {0, 1, 2, 3}
    assert len(transfer.interpolation_rows) == len(mesh.points) == 13
    assert kernel.math_objects[mesh_id]["sources"] == [marking.data["object_id"]]
    assert kernel.math_objects[transfer_id]["sources"] == [marking.data["object_id"]]
    assert kernel.apply(mesh_id, "verify").status == "verified"
    assert kernel.apply(transfer_id, "verify").status == "verified"


def test_refined_mesh_runs_full_discrete_assembly_solve_estimate_chain():
    kernel = MathKernel(); *_, refinement = adaptive_chain(kernel)
    *_, estimate = fine_chain(kernel, refinement)
    assert estimate.ok and kernel.apply(estimate.data["object_id"], "verify").status == "verified"


def test_observed_rate_is_empirical_and_not_theorem_or_bound():
    kernel = MathKernel(); _, _, coarse, _, refinement = adaptive_chain(kernel)
    *_, fine = fine_chain(kernel, refinement)
    result = kernel.apply(fine.data["object_id"], "compare", {
        "coarse_estimate_id": coarse.data["object_id"]})
    assert result.ok
    value = kernel.math_objects[result.data["object_id"]]["value"]
    assert isinstance(value, FEMConvergenceObservation)
    assert value.estimator_ratio < 1 and value.observed_estimator_rate > 0
    assert value.interpretation == "empirical_estimator_sequence"
    assert not value.convergence_theorem and not value.continuum_error_bound
    assert kernel.math_objects[result.data["object_id"]]["sources"] == [
        fine.data["object_id"], coarse.data["object_id"]]
    assert kernel.apply(result.data["object_id"], "verify").status == "verified"


def test_output_only_types_and_invalid_marking_fail_atomically():
    kernel = MathKernel()
    for name in ("FEMErrorEstimate", "RefinementMarking", "RefinedMesh",
                 "MeshTransfer", "FEMConvergenceObservation"):
        assert not kernel.object_create(name, {}).ok
    _, _, estimate, _, _ = adaptive_chain(kernel)
    before = len(kernel.math_objects)
    for params in ({"theta": 0}, {"theta": float("nan")}, {"strategy": "mystery"}):
        result = kernel.apply(estimate.data["object_id"], "mark", params)
        assert not result.ok and len(kernel.math_objects) == before


def test_estimator_scope_refuses_natural_boundary_and_nondiagonal_terms():
    conditions = [
        {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
        {"field": "u", "kind": "neumann", "coordinate": "x", "side": "upper", "value": 0},
        {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "lower", "value": 0},
        {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "upper", "value": 0},
    ]
    from test_fem_assembly import problem_definition
    kernel = MathKernel(); *_, system = assemble(kernel,
        definition=problem_definition(source=1, conditions=conditions))
    solution = kernel.apply(system.data["object_id"], "solve")
    before = len(kernel.math_objects)
    result = kernel.apply(solution.data["object_id"], "estimate_error")
    assert not result.ok and "essential-only" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_cross_estimate_comparison_and_tampered_source_envelopes_fail_closed():
    kernel = MathKernel(); _, _, estimate, marking, refinement = adaptive_chain(kernel)
    other = MathKernel(); _, _, other_estimate, _, _ = adaptive_chain(other)
    result = kernel.apply(estimate.data["object_id"], "compare", {
        "coarse_estimate_id": estimate.data["object_id"]})
    assert not result.ok and "parent-to-child" in result.errors[0]
    estimate_id = estimate.data["object_id"]
    kernel.math_objects[estimate_id]["sources"] = [marking.data["object_id"]]
    assert not kernel.apply(estimate_id, "verify").ok


def test_current_estimator_and_refinement_limits_apply_on_replay():
    kernel = MathKernel(); _, solution, _, marking, _ = adaptive_chain(kernel)
    limited = MathKernel(replace(kernel.settings, max_fem_estimator_work=1))
    limited.math_objects = dict(kernel.math_objects)
    assert not limited.apply(solution.data["object_id"], "estimate_error").ok
    limited = MathKernel(replace(kernel.settings, max_fem_refined_cells=3))
    limited.math_objects = dict(kernel.math_objects)
    assert not limited.apply(marking.data["object_id"], "refine").ok


def test_persistence_restart_replays_full_adaptive_lineage(tmp_path):
    settings = Settings(store_path=str(tmp_path / "g5.sqlite"))
    kernel = MathKernel(settings); _, _, coarse, marking, refinement = adaptive_chain(kernel)
    *_, fine = fine_chain(kernel, refinement)
    observation = kernel.apply(fine.data["object_id"], "compare", {
        "coarse_estimate_id": coarse.data["object_id"]})
    restarted = MathKernel(settings)
    ids = [coarse.data["object_id"], marking.data["object_id"],
           refinement.data["object_ids"]["mesh"], refinement.data["object_ids"]["transfer"],
           fine.data["object_id"], observation.data["object_id"]]
    assert all(restarted.apply(item, "verify").status == "verified" for item in ids)


def test_pde_capability_inventory_and_version():
    kernel = MathKernel(); caps = kernel.capabilities()
    assert caps["version"] == __import__("mathkernel").__version__
    names = {item["name"] for item in kernel.capability_query(domain="pde")["capabilities"]}
    assert {"pde.FEMSolution.estimate_error", "pde.FEMErrorEstimate.mark",
            "pde.RefinementMarking.refine", "pde.FEMErrorEstimate.compare",
            "pde.MeshTransfer.verify"} <= names
    assert len(names) == 31


def test_generic_mcp_exposes_and_executes_adaptivity_chain():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    from test_fem_assembly import (CELLS, POINTS,
                                                  problem_definition,
                                                  weak_parameters)

    async def scenario():
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
            b = await client.call_tool("math_apply", {"object_id": r.data["data"]["object_id"], "operation": "basis", "parameters": {}})
            q = await client.call_tool("math_apply", {"object_id": r.data["data"]["object_id"], "operation": "quadrature", "parameters": {"degree_exact": 2}})
            s = await client.call_tool("math_apply", {"object_id": m.data["data"]["object_id"], "operation": "finite_element_space", "parameters": {
                "reference_element_id": r.data["data"]["object_id"], "basis_id": b.data["data"]["object_id"], "field": "u"}})
            a = await client.call_tool("math_apply", {"object_id": s.data["data"]["object_id"], "operation": "assemble", "parameters": {
                "quadrature_id": q.data["data"]["object_id"], "substitutions": {"f": 1}}})
            sol = await client.call_tool("math_apply", {"object_id": a.data["data"]["object_id"], "operation": "solve", "parameters": {}})
            est = await client.call_tool("math_apply", {"object_id": sol.data["data"]["object_id"], "operation": "estimate_error", "parameters": {}})
            mark = await client.call_tool("math_apply", {"object_id": est.data["data"]["object_id"], "operation": "mark", "parameters": {"theta": 0.5}})
            refine = await client.call_tool("math_apply", {"object_id": mark.data["data"]["object_id"], "operation": "refine", "parameters": {}})
            return est.data, mark.data, refine.data

    estimate, marking, refinement = asyncio.run(scenario())
    assert estimate["ok"] and marking["ok"] and refinement["ok"]
    assert refinement["data"]["object_type"] == "RefinedMesh"
