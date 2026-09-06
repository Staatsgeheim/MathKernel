# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
import asyncio

import sympy as sp

from mathkernel import (BasisFunctionSet, FEMMesh, FiniteElementSpace,
                        MathKernel, QuadratureRule, ReferenceElement, Settings)
from test_weak_forms import poisson_definition, weak_parameters


POINTS = [[0, 0], [1, 0], [1, 1], [0, 1]]
CELLS = [[0, 2, 1], [0, 2, 3]]


def chain(kernel, *, points=POINTS, cells=CELLS, triangulation_id=None):
    problem = kernel.object_create("PDEProblem", poisson_definition())
    assert problem.ok, problem.errors
    weak = kernel.apply(problem.data["object_id"], "derive_weak_form", weak_parameters())
    assert weak.ok, weak.errors
    definition = {"weak_form_id": weak.data["object_id"], "cell_type": "triangle"}
    if triangulation_id is None:
        definition.update(points=points, cells=cells)
    else:
        definition["triangulation_id"] = triangulation_id
    mesh = kernel.object_create("FEMMesh", definition)
    return problem, weak, mesh


def discrete_chain(kernel):
    problem, weak, mesh = chain(kernel)
    assert mesh.ok, mesh.errors
    reference = kernel.apply(mesh.data["object_id"], "reference_element")
    basis = kernel.apply(reference.data["object_id"], "basis")
    quadrature = kernel.apply(reference.data["object_id"], "quadrature",
                              {"degree_exact": 2})
    space = kernel.apply(mesh.data["object_id"], "finite_element_space", {
        "reference_element_id": reference.data["object_id"],
        "basis_id": basis.data["object_id"], "field": "u"})
    return problem, weak, mesh, reference, basis, quadrature, space


def test_mesh_canonicalizes_orientation_and_records_topology():
    kernel = MathKernel(); _, weak, made = chain(kernel)
    assert made.ok
    mesh = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(mesh, FEMMesh)
    assert mesh.cells == ((0, 1, 2), (0, 2, 3))
    assert mesh.orientation_flips == (0,)
    assert mesh.cell_determinants == (1, 1)
    assert mesh.interior_facets == ((0, 2),)
    assert len(mesh.boundary_facets) == len(mesh.boundary_facet_owners) == 4
    assert mesh.cell_component_count == 1
    assert kernel.math_objects[made.data["object_id"]]["sources"] == [weak.data["object_id"]]


def test_mesh_verification_is_conservative_about_domain_coverage():
    kernel = MathKernel(); _, _, mesh = chain(kernel)
    replay = kernel.apply(mesh.data["object_id"], "verify")
    assert replay.ok and replay.status == "unknown"
    assert replay.data["verification"]["positive_orientation"] is True
    assert replay.data["verification"]["cell_components_recomputed"] is True
    assert replay.data["verification"]["domain_coverage"] is None


def test_disconnected_cells_are_counted_without_inventing_invalidity():
    points = [[0, 0], [1, 0], [0, 1], [1, 1], [1, 0], [1, 1]]
    # Duplicate points are rejected before component analysis.
    kernel = MathKernel(); _, _, made = chain(kernel, points=points, cells=[[0, 1, 2]])
    assert not made.ok and "unique" in made.errors[0]
    points = [[0, 0], [1, 0], [0, 1], ["1/2", "1/2"], [1, "1/2"], ["1/2", 1]]
    kernel = MathKernel(); _, _, made = chain(kernel, points=points,
                                               cells=[[0, 1, 2], [3, 4, 5]])
    assert made.ok
    assert kernel.math_objects[made.data["object_id"]]["value"].cell_component_count == 2


def test_invalid_meshes_are_refused_atomically():
    bad = [
        ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]], "degenerate"),
        (POINTS, [[0, 1, 9]], "connectivity"),
        ([[0, 0], [1, 0], [0, 1], [0, -1], [1, 1]],
         [[0, 1, 2], [0, 1, 3], [0, 1, 4]], "nonmanifold"),
        ([[0, 0], [2, 0], [0, 1]], [[0, 1, 2]], "outside"),
    ]
    for points, cells, message in bad:
        kernel = MathKernel(); before = len(kernel.math_objects)
        _, _, made = chain(kernel, points=points, cells=cells)
        assert not made.ok and message in made.errors[0]
        assert len(kernel.math_objects) == before + 2


def test_numeric_near_degeneracy_cannot_advance_to_reference_element():
    kernel = MathKernel(); _, _, mesh = chain(
        kernel, points=[["0.0", "0.0"], ["1.0", "0.0"],
                        ["0.0", "0.00000000000000000001"]], cells=[[0, 1, 2]])
    assert mesh.ok
    value = kernel.math_objects[mesh.data["object_id"]]["value"]
    assert value.orientation_status == "ambiguous"
    blocked = kernel.apply(mesh.data["object_id"], "reference_element")
    assert not blocked.ok and "ambiguous" in blocked.errors[0]


def test_reference_basis_quadrature_and_space_form_a_replayable_chain():
    kernel = MathKernel(); _, _, mesh, reference, basis, quadrature, space = discrete_chain(kernel)
    assert all(item.ok for item in (reference, basis, quadrature, space))
    ref_value = kernel.math_objects[reference.data["object_id"]]["value"]
    basis_value = kernel.math_objects[basis.data["object_id"]]["value"]
    quad_value = kernel.math_objects[quadrature.data["object_id"]]["value"]
    space_value = kernel.math_objects[space.data["object_id"]]["value"]
    assert isinstance(ref_value, ReferenceElement)
    assert ref_value.vertices == ((0, 0), (1, 0), (0, 1))
    assert isinstance(basis_value, BasisFunctionSet)
    xi, eta = sp.symbols("xi eta")
    assert basis_value.functions == (1 - eta - xi, xi, eta)
    assert isinstance(quad_value, QuadratureRule)
    assert quad_value.degree_exact == 2 and sum(quad_value.weights) == sp.Rational(1, 2)
    assert isinstance(space_value, FiniteElementSpace)
    assert space_value.local_to_global == ((0, 1, 2), (0, 2, 3))
    assert space_value.dof_count == 4 and space_value.essential_dofs == (0, 1, 2, 3)
    assert space.status == "unknown"
    assert kernel.math_objects[space.data["object_id"]]["sources"] == [
        mesh.data["object_id"], reference.data["object_id"], basis.data["object_id"]]
    for item in (reference, basis, quadrature, space):
        assert kernel.apply(item.data["object_id"], "verify").status == "verified"


def test_mismatched_multi_source_space_is_refused_without_output():
    kernel = MathKernel(); *_, mesh1, ref1, basis1, _, _ = discrete_chain(kernel)
    *_, mesh2, ref2, basis2, _, _ = discrete_chain(kernel)
    before = len(kernel.math_objects)
    result = kernel.apply(mesh1.data["object_id"], "finite_element_space", {
        "reference_element_id": ref2.data["object_id"],
        "basis_id": basis2.data["object_id"], "field": "u"})
    assert not result.ok and len(kernel.math_objects) == before


def test_output_only_types_and_tamper_replay_are_safe():
    kernel = MathKernel()
    for name in ("ReferenceElement", "BasisFunctionSet", "QuadratureRule", "FiniteElementSpace"):
        assert not kernel.object_create(name, {}).ok
    _, _, _, _, basis, quadrature, space = discrete_chain(kernel)
    for result, field, replacement in (
        (basis, "functions", (sp.Integer(9),) * 3),
        (quadrature, "weights", (sp.Integer(9),) * 3),
        (space, "dof_count", 99),
    ):
        object_id = result.data["object_id"]
        original = kernel.math_objects[object_id]["value"]
        kernel.math_objects[object_id]["value"] = original.model_copy(update={field: replacement})
        assert kernel.apply(object_id, "verify").status == "refuted"


def test_triangulation_source_is_verified_and_preserves_ancestry():
    kernel = MathKernel()
    tri = kernel.object_create("Triangulation", {"points": POINTS,
        "triangles": [[0, 1, 2], [0, 2, 3]]})
    _, weak, mesh = chain(kernel, triangulation_id=tri.data["object_id"])
    assert mesh.ok
    assert kernel.math_objects[mesh.data["object_id"]]["sources"] == [
        weak.data["object_id"], tri.data["object_id"]]


def test_persistence_restarts_the_complete_finite_element_chain(tmp_path):
    settings = Settings(store_path=str(tmp_path / "g3.sqlite"))
    kernel = MathKernel(settings); *_, space = discrete_chain(kernel)
    object_id = space.data["object_id"]
    restarted = MathKernel(settings)
    assert restarted.object_get(object_id).ok
    assert restarted.apply(object_id, "verify").status == "verified"


def test_current_limits_block_replay_and_creation_atomically():
    kernel = MathKernel(Settings(max_fem_points=3))
    _, _, mesh = chain(kernel)
    assert not mesh.ok and "limits" in mesh.errors[0]
    kernel = MathKernel(Settings(max_fem_dofs=3)); _, _, mesh = chain(kernel)
    assert mesh.ok
    ref = kernel.apply(mesh.data["object_id"], "reference_element")
    basis = kernel.apply(ref.data["object_id"], "basis")
    before = len(kernel.math_objects)
    space = kernel.apply(mesh.data["object_id"], "finite_element_space", {
        "reference_element_id": ref.data["object_id"],
        "basis_id": basis.data["object_id"], "field": "u"})
    assert not space.ok and len(kernel.math_objects) == before


def test_capabilities_preserve_the_complete_finite_element_surface():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="pde")
    assert manifest["count"] >= 16
    names = {item["name"] for item in manifest["capabilities"]}
    for name in ("pde.FEMMesh.verify", "pde.FEMMesh.reference_element",
                 "pde.ReferenceElement.basis", "pde.ReferenceElement.quadrature",
                 "pde.FEMMesh.finite_element_space"):
        assert name in names
    overview = kernel.capabilities()["pde"]
    assert overview["max_fem_work"] == 20_000_000


def test_generic_mcp_tools_execute_a_finite_element_chain():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            p = await client.call_tool("math_object_create", {
                "object_type": "PDEProblem", "definition": poisson_definition()})
            w = await client.call_tool("math_apply", {"object_id": p.data["data"]["object_id"],
                "operation": "derive_weak_form", "parameters": weak_parameters()})
            mesh = await client.call_tool("math_object_create", {"object_type": "FEMMesh",
                "definition": {"weak_form_id": w.data["data"]["object_id"],
                    "cell_type": "triangle", "points": POINTS, "cells": CELLS}})
            ref = await client.call_tool("math_apply", {
                "object_id": mesh.data["data"]["object_id"],
                "operation": "reference_element", "parameters": {}})
            basis = await client.call_tool("math_apply", {
                "object_id": ref.data["data"]["object_id"],
                "operation": "basis", "parameters": {}})
            return mesh.data, ref.data, basis.data
    mesh, ref, basis = asyncio.run(scenario())
    assert mesh["ok"] and ref["ok"] and basis["ok"]


