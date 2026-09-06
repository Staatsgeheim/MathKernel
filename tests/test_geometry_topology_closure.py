import asyncio

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.algebraic_topology import cache_info, clear_caches


POINTS = [[0, 0], [4, 0], [3, 4], [0, 3], [1, 1]]


def create(kernel, kind, definition):
    result = kernel.object_create(kind, definition)
    assert result.ok, result.errors
    return result.data["object_id"]


def test_exact_geometry_to_topology_to_homology_composes_with_recursive_lineage():
    k = MathKernel()
    points = create(k, "PointSet", {"dimension": 2, "points": POINTS})
    delaunay = k.apply(points, "delaunay")
    simplicial = k.apply(delaunay.data["object_id"], "to_simplicial_complex")
    homology = k.apply(simplicial.data["object_id"], "homology")
    assert all(result.ok and result.status == "verified"
               for result in (delaunay, simplicial, homology))
    groups = k.math_objects[homology.data["object_id"]]["value"].groups
    assert [group.structure for group in groups] == ["Z", "0", "0"]
    assert k.math_objects[delaunay.data["object_id"]]["sources"] == [points]
    assert k.math_objects[simplicial.data["object_id"]]["sources"] == [delaunay.data["object_id"]]
    assert k.math_objects[homology.data["object_id"]]["sources"] == [simplicial.data["object_id"]]


def test_geometry_topology_chain_survives_restart_with_every_link(tmp_path):
    settings = Settings(store_path=str(tmp_path / "geometry.sqlite"))
    k = MathKernel(settings)
    points = create(k, "PointSet", {"dimension": 2, "points": POINTS})
    delaunay = k.apply(points, "delaunay")
    simplicial = k.apply(delaunay.data["object_id"], "to_simplicial_complex")
    homology = k.apply(simplicial.data["object_id"], "homology")
    restarted = MathKernel(settings)
    chain = [points, delaunay.data["object_id"], simplicial.data["object_id"],
             homology.data["object_id"]]
    assert all(restarted.object_get(object_id).ok for object_id in chain)
    assert restarted.math_objects[chain[1]]["sources"] == [chain[0]]
    assert restarted.math_objects[chain[2]]["sources"] == [chain[1]]
    assert restarted.math_objects[chain[3]]["sources"] == [chain[2]]


def test_numeric_triangulation_cannot_launder_into_exact_topology():
    k = MathKernel(); triangulation = create(k, "Triangulation", {
        "points": [["0.0", 0], [1, 0], [0, 1]], "triangles": [[0, 1, 2]]})
    before = len(k.math_objects)
    result = k.apply(triangulation, "to_simplicial_complex")
    assert not result.ok and "requires exact coordinates" in result.errors[0]
    assert len(k.math_objects) == before


def test_refuted_triangulation_cannot_cross_the_combinatorial_boundary():
    k = MathKernel(); triangulation = create(k, "Triangulation", {
        "points": [[0, 0], [3, 0], [2, 2], [0, 2]],
        "triangles": [[0, 1, 2], [0, 1, 3]]})
    assert k.apply(triangulation, "verify").status == "refuted"
    result = k.apply(triangulation, "to_simplicial_complex")
    assert not result.ok and "verified nonoverlapping" in result.errors[0]


def test_closure_expansion_stops_at_configured_cell_limit():
    k = MathKernel(Settings(max_topology_cells=10))
    simplex = k.object_create("SimplicialComplex", {
        "vertices": list("abcde"), "maximal_simplices": [[0, 1, 2, 3, 4]]})
    cube = k.object_create("CubicalComplex", {"ambient_dimension": 3,
        "maximal_cells": [[[0, 1], [0, 1], [0, 1]]]})
    assert not simplex.ok and "closure exceeds" in simplex.errors[0]
    assert not cube.ok and "closure exceeds" in cube.errors[0]


def test_maximal_simplex_input_count_is_bounded_before_closure():
    k = MathKernel(Settings(max_topology_cells=2))
    result = k.object_create("SimplicialComplex", {
        "vertices": ["a", "b"], "maximal_simplices": [[0], [1], [0, 1]]})
    assert not result.ok and "bounded sequence" in result.errors[0]


def test_complex_boundary_allocation_is_preflighted_before_construction():
    k = MathKernel(Settings(max_topology_matrix_entries=2))
    complex_id = create(k, "SimplicialComplex", {
        "vertices": ["a", "b", "c"], "maximal_simplices": [[0, 1, 2]]})
    result = k.apply(complex_id, "verify")
    assert not result.ok and "max_topology_matrix_entries" in result.errors[0]


def test_triangulation_bridge_respects_topology_matrix_preflight():
    k = MathKernel(Settings(max_topology_matrix_entries=2))
    triangulation = create(k, "Triangulation", {
        "points": [[0, 0], [1, 0], [0, 1]], "triangles": [[0, 1, 2]]})
    before = len(k.math_objects)
    result = k.apply(triangulation, "to_simplicial_complex")
    assert not result.ok and "max_topology_matrix_entries" in result.errors[0]
    assert len(k.math_objects) == before


def test_cross_chart_tensor_form_operations_fail_without_derived_objects():
    k = MathKernel(); manifold = create(k, "Manifold", {"name": "M", "dimension": 2})
    left = create(k, "Chart", {"name": "left", "manifold_id": manifold,
        "coordinates": ["x", "y"], "orientation": 1})
    right = create(k, "Chart", {"name": "right", "manifold_id": manifold,
        "coordinates": ["u", "v"], "orientation": 1})
    form = create(k, "DifferentialForm", {"chart_id": left, "degree": 1,
        "terms": [{"indices": [0], "coefficient": "x"}]})
    wrong_metric = create(k, "Metric", {"chart_id": right,
        "components": [[1, 0], [0, 1]], "signature": [2, 0]})
    wrong_form = create(k, "DifferentialForm", {"chart_id": right, "degree": 1,
        "terms": [{"indices": [0], "coefficient": "u"}]})
    before = len(k.math_objects)
    assert not k.apply(form, "hodge_star", {"metric_id": wrong_metric}).ok
    assert not k.apply(form, "wedge", {"other_id": wrong_form}).ok
    assert len(k.math_objects) == before


def test_wrong_direction_pullback_is_rejected_without_trust_or_source_leakage():
    k = MathKernel(); manifold = create(k, "Manifold", {"name": "M", "dimension": 1})
    x = create(k, "Chart", {"name": "x", "manifold_id": manifold, "coordinates": ["x"]})
    u = create(k, "Chart", {"name": "u", "manifold_id": manifold, "coordinates": ["u"]})
    mapping = create(k, "CoordinateMap", {"source_chart_id": u,
        "target_chart_id": x, "forward": ["u"]})
    source_form = create(k, "DifferentialForm", {"chart_id": u, "degree": 1,
        "terms": [{"indices": [0], "coefficient": 1}]})
    result = k.apply(source_form, "pullback", {"map_id": mapping})
    assert not result.ok and "map target" in result.errors[0]


def test_ambiguous_delaunay_does_not_materialize_a_topology_object():
    k = MathKernel(); square = create(k, "PointSet", {"dimension": 2,
        "points": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    before = len(k.math_objects)
    result = k.apply(square, "delaunay")
    assert result.ok and result.status == "candidate"
    assert result.data["value"]["classification"] == "ambiguous"
    assert "object_id" not in result.data and len(k.math_objects) == before


def test_invalid_chain_blocks_z_q_and_every_finite_field_claim():
    k = MathKernel(); chain = create(k, "ChainComplex", {
        "ranks": [1, 1, 1], "boundaries": [[[2]], [[3]]]})
    for parameters in ({}, {"coefficient": "Q"},
                       {"coefficient": "GF(p)", "prime": 2},
                       {"coefficient": "GF(p)", "prime": 5}):
        result = k.apply(chain, "homology", parameters)
        assert not result.ok and "boundary squared zero" in result.errors[0]


def test_smith_cache_reuses_immutable_certificates_without_changing_results():
    from mathkernel.engines import run_with_timeout
    run_with_timeout(clear_caches, 5); k = MathKernel(); chain = create(k, "ChainComplex", {
        "ranks": [1, 1, 1], "boundaries": [[[0]], [[6]]]})
    first = k.apply(chain, "homology")
    after_first = run_with_timeout(cache_info, 5)["smith_normal_forms"].copy()
    second = k.apply(chain, "homology")
    after_second = run_with_timeout(cache_info, 5)["smith_normal_forms"]
    assert first.data["value"] == second.data["value"]
    assert after_second["hits"] > after_first["hits"]
    group = k.math_objects[second.data["object_id"]]["value"].groups[1]
    assert group.structure == "Z/6Z" and group.torsion_representatives == ((1,),)


def test_cell_orientation_changes_representatives_not_homology_structure():
    k = MathKernel()
    left = create(k, "SimplicialComplex", {"vertices": ["a", "b", "c"],
        "maximal_simplices": [[0, 1], [1, 2], [2, 0]]})
    right = create(k, "SimplicialComplex", {"vertices": ["c", "b", "a"],
        "maximal_simplices": [[2, 1], [1, 0], [0, 2]]})
    structures = []
    for source in (left, right):
        result = k.apply(source, "homology")
        structures.append([group["structure"] for group in result.data["object"]["groups"]])
    assert structures == [["Z", "Z"], ["Z", "Z"]]


def test_geometry_derived_types_remain_output_only():
    k = MathKernel()
    for kind in ("GeometryTensor", "Connection", "GeodesicSystem", "JacobianMap",
                 "VoronoiDiagram", "Homology", "HomologyGroup"):
        assert not k.object_create(kind, {}).ok


def test_geometry_capability_and_resource_surfaces():
    k = MathKernel(Settings())
    manifest = k.capability_query(domain="geometry")
    assert manifest["count"] == 44
    bridge = next(item for item in manifest["capabilities"]
                  if item["operation"] == "to_simplicial_complex")
    assert bridge["input_types"] == ["Triangulation"]
    assert bridge["output_types"] == ["EngineeringResult", "SimplicialComplex"]
    assert bridge["trust_levels"] == ["exact"]


def test_live_mcp_geometry_topology_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            triangulation = (await client.call_tool("math_object_create", {
                "object_type": "Triangulation", "definition": {
                    "points": [[0, 0], [1, 0], [0, 1]],
                    "triangles": [[0, 1, 2]]}})).data["data"]["object_id"]
            simplicial = (await client.call_tool("math_apply", {
                "object_id": triangulation, "operation": "to_simplicial_complex",
                "parameters": {}})).data["data"]["object_id"]
            return (await client.call_tool("math_apply", {"object_id": simplicial,
                "operation": "homology", "parameters": {}})).data
    result = asyncio.run(run())
    assert result["ok"] and [group["structure"] for group in
        result["data"]["object"]["groups"]] == ["Z", "0", "0"]

