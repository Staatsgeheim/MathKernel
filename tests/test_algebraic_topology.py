import asyncio

from mathkernel import MathKernel, Settings, TrustLevel


TRIANGLE_CIRCLE = {
    "vertices": ["a", "b", "c"],
    "maximal_simplices": [[0, 1], [1, 2], [0, 2]],
}


def create(kernel, kind, definition):
    result = kernel.object_create(kind, definition)
    assert result.ok, result.errors
    return result.data["object_id"]


def groups(kernel, result):
    assert result.ok and result.status == "verified", result.errors
    value = kernel.math_objects[result.data["object_id"]]["value"]
    return {group.degree: group for group in value.groups}


def test_typed_simplicial_cubical_and_chain_sources_are_exact():
    k = MathKernel(Settings())
    simplex = k.object_create("SimplicialComplex", TRIANGLE_CIRCLE)
    cube = k.object_create("CubicalComplex", {"ambient_dimension": 1,
        "maximal_cells": [[[0, 1]]]})
    chain = k.object_create("ChainComplex", {"ranks": [1, 1],
        "boundaries": [[[0]]], "basis_labels": [["v"], ["e"]]})
    assert all(result.ok and result.trust == TrustLevel.EXACT
               for result in (simplex, cube, chain))
    assert not k.object_create("Homology", {}).ok


def test_simplicial_closure_and_oriented_boundary_are_canonical():
    k = MathKernel(); object_id = create(k, "SimplicialComplex", {
        "vertices": ["a", "b", "c"], "maximal_simplices": [[2, 0, 1]]})
    value = k.math_objects[object_id]["value"]
    assert tuple(map(len, value.simplices)) == (3, 3, 1)
    verify = k.apply(object_id, "verify")
    boundary = k.apply(object_id, "boundary_matrix", {"degree": 2})
    assert verify.ok and verify.status == "verified"
    assert boundary.data["value"]["matrix"] == [[1], [-1], [1]]


def test_chain_complex_conversion_is_derived_and_keeps_source_ancestry():
    k = MathKernel(); source = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    result = k.apply(source, "chain_complex")
    assert result.ok and result.data["object_type"] == "ChainComplex"
    derived = k.math_objects[result.data["object_id"]]
    assert derived["sources"] == [source]
    assert derived["value"].ranks == (3, 3)


def test_circle_homology_over_z_q_and_gf2():
    k = MathKernel(); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    for parameters, expected in (({}, ("Z", "Z")),
            ({"coefficient": "Q"}, ("Q", "Q")),
            ({"coefficient": "GF(p)", "prime": 2}, ("GF(2)", "GF(2)"))):
        result = k.apply(circle, "homology", parameters)
        found = groups(k, result)
        assert (found[0].structure, found[1].structure) == expected
        assert result.data["details"]["identity_checks"]["euler_poincare"] is True


def test_filled_triangle_is_contractible():
    k = MathKernel(); disk = create(k, "SimplicialComplex", {
        "vertices": ["a", "b", "c"], "maximal_simplices": [[0, 1, 2]]})
    found = groups(k, k.apply(disk, "homology"))
    assert [found[i].structure for i in range(3)] == ["Z", "0", "0"]


def test_tetrahedron_boundary_has_two_sphere_homology():
    k = MathKernel(); sphere = create(k, "SimplicialComplex", {
        "vertices": ["a", "b", "c", "d"],
        "maximal_simplices": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]})
    found = groups(k, k.apply(sphere, "homology"))
    assert [found[i].betti_number for i in range(3)] == [1, 0, 1]


def test_disconnected_complex_has_component_rank_in_h0():
    k = MathKernel(); points = create(k, "SimplicialComplex", {
        "vertices": ["a", "b", "c"], "maximal_simplices": [[0], [1], [2]]})
    found = groups(k, k.apply(points, "homology"))
    assert found[0].betti_number == 3 and found[0].structure == "Z^3"


def test_integer_homology_reports_projective_plane_torsion():
    k = MathKernel(); rp2 = create(k, "ChainComplex", {
        "ranks": [1, 1, 1], "boundaries": [[[0]], [[2]]]})
    found = groups(k, k.apply(rp2, "homology"))
    assert found[1].betti_number == 0
    assert found[1].torsion_coefficients == (2,)
    assert found[1].structure == "Z/2Z"
    assert found[1].torsion_representatives == ((1,),)


def test_field_homology_detects_mod_p_torsion_classes():
    k = MathKernel(); rp2 = create(k, "ChainComplex", {
        "ranks": [1, 1, 1], "boundaries": [[[0]], [[2]]]})
    q = groups(k, k.apply(rp2, "homology", {"coefficient": "Q"}))
    mod2 = groups(k, k.apply(rp2, "homology", {"coefficient": "GF(p)", "prime": 2}))
    mod3 = groups(k, k.apply(rp2, "homology", {"coefficient": "GF(p)", "prime": 3}))
    assert [q[i].betti_number for i in range(3)] == [1, 0, 0]
    assert [mod2[i].betti_number for i in range(3)] == [1, 1, 1]
    assert [mod3[i].betti_number for i in range(3)] == [1, 0, 0]


def test_invalid_boundary_composition_is_refuted_and_homology_blocked():
    k = MathKernel(); invalid = create(k, "ChainComplex", {
        "ranks": [1, 1, 1], "boundaries": [[[2]], [[3]]]})
    verify = k.apply(invalid, "verify")
    homology = k.apply(invalid, "homology")
    assert verify.ok and verify.status == "refuted"
    assert verify.data["verification"]["boundary_squared_zero_degree_2"] is False
    assert not homology.ok and "boundary squared zero" in homology.errors[0]


def test_cubical_filled_square_is_contractible_and_boundary_squared_zero():
    k = MathKernel(); square = create(k, "CubicalComplex", {
        "ambient_dimension": 2,
        "maximal_cells": [[[0, 1], [0, 1]]]})
    value = k.math_objects[square]["value"]
    assert tuple(map(len, value.cells)) == (4, 4, 1)
    assert k.apply(square, "verify").status == "verified"
    found = groups(k, k.apply(square, "homology"))
    assert [found[i].structure for i in range(3)] == ["Z", "0", "0"]


def test_cubical_square_boundary_has_one_cycle():
    k = MathKernel(); loop = create(k, "CubicalComplex", {
        "ambient_dimension": 2, "maximal_cells": [
            [[0, 1], [0, 0]], [[0, 1], [1, 1]],
            [[0, 0], [0, 1]], [[1, 1], [0, 1]]]})
    found = groups(k, k.apply(loop, "homology"))
    assert found[0].betti_number == found[1].betti_number == 1


def test_degree_selection_returns_only_requested_homology_group():
    k = MathKernel(); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    result = k.apply(circle, "homology", {"degree": 1})
    value = k.math_objects[result.data["object_id"]]["value"]
    assert len(value.groups) == 1 and value.groups[0].degree == 1
    assert value.euler_characteristic is None


def test_cycle_representative_is_annihilated_by_boundary():
    k = MathKernel(); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    chain_result = k.apply(circle, "chain_complex")
    chain = k.math_objects[chain_result.data["object_id"]]["value"]
    group = groups(k, k.apply(circle, "homology"))[1]
    cycle = group.cycle_representatives[0]
    assert all(sum(row[i] * cycle[i] for i in range(chain.ranks[1])) == 0
               for row in chain.boundaries[0])


def test_euler_characteristic_uses_alternating_chain_ranks():
    k = MathKernel(); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    chain = k.apply(circle, "chain_complex")
    result = k.apply(chain.data["object_id"], "euler_characteristic")
    assert result.ok and result.data["value"] == 0


def test_prime_and_parameter_validation_fail_closed():
    k = MathKernel(); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    assert not k.apply(circle, "homology", {"coefficient": "GF(p)", "prime": 4}).ok
    assert not k.apply(circle, "homology", {"coefficient": "Q", "prime": 2}).ok
    assert not k.apply(circle, "homology", {"coefficient": "R"}).ok
    assert not k.apply(circle, "homology", {"mystery": 1}).ok


def test_topology_resource_limits_cover_closure_entries_bits_and_snf():
    limited = MathKernel(Settings(max_topology_cells=3))
    closure = limited.object_create("SimplicialComplex", {
        "vertices": ["a", "b", "c"], "maximal_simplices": [[0, 1, 2]]})
    assert not closure.ok and "closure" in closure.errors[0]
    bits = MathKernel(Settings(max_topology_entry_bits=2))
    too_large = bits.object_create("ChainComplex", {
        "ranks": [1, 1], "boundaries": [[[8]]]})
    assert not too_large.ok and "entry" in too_large.errors[0]
    snf = MathKernel(Settings(max_normal_form_dim=1, max_topology_work=1000))
    chain = create(snf, "ChainComplex", {"ranks": [2], "boundaries": []})
    blocked = snf.apply(chain, "homology")
    assert not blocked.ok and "max_normal_form_dim" in blocked.errors[0]


def test_topology_objects_and_homology_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "e4.sqlite"))
    k = MathKernel(settings); circle = create(k, "SimplicialComplex", TRIANGLE_CIRCLE)
    chain = k.apply(circle, "chain_complex"); homology = k.apply(circle, "homology")
    restarted = MathKernel(settings)
    for result, expected in ((chain, "ChainComplex"), (homology, "Homology")):
        restored = restarted.object_get(result.data["object_id"])
        assert restored.ok and restored.data["object_type"] == expected
        assert restarted.math_objects[result.data["object_id"]]["sources"] == [circle]


def test_topology_capabilities_and_limits(monkeypatch):
    monkeypatch.setenv("MATHKERNEL_MAX_TOPOLOGY_CELLS", "77")
    settings = Settings.from_env()
    assert settings.max_topology_cells == 77
    k = MathKernel(Settings())
    manifest = k.capability_query(domain="geometry")
    assert manifest["count"] == 44
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert "smith_kernel_quotient" in entries[("ChainComplex", "homology")]["verification_methods"]
    assert entries[("SimplicialComplex", "homology")]["trust_levels"] == ["exact"]
    assert k.capabilities()["geometry"]["limits"]["max_topology_cells"] == 10_000


def test_live_mcp_algebraic_topology_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            source = (await client.call_tool("math_object_create", {
                "object_type": "SimplicialComplex", "definition": TRIANGLE_CIRCLE})).data
            return (await client.call_tool("math_apply", {
                "object_id": source["data"]["object_id"], "operation": "homology",
                "parameters": {"coefficient": "GF(p)", "prime": 2}})).data
    result = asyncio.run(run())
    assert result["ok"] and result["data"]["object_type"] == "Homology"


def test_nonclosed_or_malformed_topological_definitions_are_rejected():
    k = MathKernel()
    duplicate_vertices = k.object_create("SimplicialComplex", {
        "vertices": ["a", "a"], "maximal_simplices": [[0, 1]]})
    bad_cube = k.object_create("CubicalComplex", {"ambient_dimension": 1,
        "maximal_cells": [[[0, 2]]]})
    wrong_shape = k.object_create("ChainComplex", {"ranks": [2, 1],
        "boundaries": [[[1]]]})
    assert not duplicate_vertices.ok and not bad_cube.ok and not wrong_shape.ok
