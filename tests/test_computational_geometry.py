import asyncio

import sympy as sp

from mathkernel import MathKernel, Settings, TrustLevel


GENERIC_POINTS = [[0, 0], [4, 0], [3, 4], [0, 3], [1, 1]]


def point_set(kernel, points=GENERIC_POINTS, dimension=2):
    result = kernel.object_create("PointSet", {"dimension": dimension, "points": points})
    assert result.ok, result.errors
    return result.data["object_id"]


def test_t_point_pointset_polygon_polytope_and_triangulation_are_typed_sources():
    k = MathKernel()
    point = k.object_create("Point", {"coordinates": [0, 1, 2]})
    points = k.object_create("PointSet", {"dimension": 2, "points": [[0, 0], [1, 0]]})
    polygon = k.object_create("Polygon", {"vertices": [[0, 0], [1, 0], [0, 1]]})
    polytope = k.object_create("Polytope", {"dimension": 2, "halfspaces": [
        {"normal": [1, 0], "bound": 1}, {"normal": [-1, 0], "bound": 0},
        {"normal": [0, 1], "bound": 1}, {"normal": [0, -1], "bound": 0}]})
    triangulation = k.object_create("Triangulation", {"points": [[0, 0], [1, 0], [0, 1]],
        "triangles": [[0, 1, 2]]})
    assert all(result.ok for result in (point, points, polygon, polytope, triangulation))
    assert not k.object_create("VoronoiDiagram", {}).ok
    symbolic = k.object_create("Point", {"coordinates": ["x", 0]})
    assert not symbolic.ok and "finite real" in symbolic.errors[0]


def test_orientation_is_exact_in_two_and_three_dimensions():
    k = MathKernel(); planar = point_set(k, [[0, 0], [2, 0], [0, 3]])
    positive = k.apply(planar, "orientation", {"indices": [0, 1, 2]})
    assert positive.ok and positive.status == "verified"
    assert positive.data["value"] == {"sign": 1, "classification": "positive", "determinant": "6"}
    spatial = point_set(k, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 3)
    volume = k.apply(spatial, "orientation", {"indices": [0, 1, 2, 3]})
    assert volume.ok and volume.data["value"]["determinant"] == "1"


def test_near_degenerate_decimal_orientation_is_explicitly_ambiguous():
    k = MathKernel(); points = point_set(k,
        [[0, 0], [1, 1], [2, "2.0000000000000001"]])
    result = k.apply(points, "orientation", {"indices": [0, 1, 2]})
    assert result.ok and result.data["status"] == "available"
    assert result.trust == TrustLevel.NUMERIC
    assert result.data["value"]["classification"] == "ambiguous"
    assert result.data["verification"]["predicate_decided"] is None


def test_incircle_is_orientation_normalized_and_reports_cocircularity():
    k = MathKernel(); points = point_set(k, [[0, 0], [2, 0], [0, 2], [1, 1], [2, 2]])
    inside = k.apply(points, "incircle", {"indices": [0, 1, 2, 3]})
    reversed_order = k.apply(points, "incircle", {"indices": [0, 2, 1, 3]})
    circular = k.apply(points, "incircle", {"indices": [0, 1, 2, 4]})
    assert inside.data["value"]["classification"] == "inside"
    assert reversed_order.data["value"]["classification"] == "inside"
    assert circular.data["value"]["classification"] == "cocircular"


def test_segment_intersection_distinguishes_proper_touching_and_disjoint():
    k = MathKernel(); points = point_set(k,
        [[0, 0], [2, 2], [0, 2], [2, 0], [3, 0], [3, 2]])
    proper = k.apply(points, "segment_intersection", {"indices": [0, 1, 2, 3]})
    touching = k.apply(points, "segment_intersection", {"indices": [0, 3, 3, 5]})
    disjoint = k.apply(points, "segment_intersection", {"indices": [0, 2, 4, 5]})
    assert proper.data["value"]["classification"] == "proper"
    assert touching.data["value"]["classification"] == "touching"
    assert disjoint.data["value"] == {"classification": "disjoint", "intersects": False}


def test_monotone_chain_hull_returns_ccw_polygon_and_witness():
    k = MathKernel(); points = point_set(k, [[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])
    result = k.apply(points, "convex_hull")
    assert result.ok and result.status == "verified" and result.data["object_type"] == "Polygon"
    polygon = k.math_objects[result.data["object_id"]]["value"]
    assert polygon.vertices == ((0, 0), (2, 0), (2, 2), (0, 2))
    assert result.data["details"]["identity_checks"] == {
        "counterclockwise": True, "all_points_contained": True}
    assert k.math_objects[result.data["object_id"]]["sources"] == [points]


def test_nearest_neighbor_uses_exact_squared_distances_and_exposes_ties():
    k = MathKernel(); points = point_set(k, [[0, 0], [2, 0], [1, 2]])
    unique = k.apply(points, "nearest_neighbor", {"point": ["7/4", "1/4"]})
    tie = k.apply(points, "nearest_neighbor", {"point": [1, 0]})
    assert unique.ok and unique.data["value"]["index"] == 1
    assert unique.data["value"]["squared_distance"] == "1/8"
    assert tie.ok and tie.data["value"]["index"] is None
    assert tie.data["value"]["indices"] == [0, 1]


def test_point_distance_is_compositional_and_ancestry_aware():
    k = MathKernel(); a = k.object_create("Point", {"coordinates": [0, 0]})
    b = k.object_create("Point", {"coordinates": [3, 4]})
    result = k.apply(a.data["object_id"], "distance_to", {"other_id": b.data["object_id"]})
    assert result.ok and result.data["value"] == {"squared_distance": "25", "distance": "5"}
    required = result.data["provenance"]["required_object_inputs"]
    assert {item["object_id"] for item in required} == {a.data["object_id"], b.data["object_id"]}


def test_polygon_verification_and_general_point_containment():
    k = MathKernel(); polygon = k.object_create("Polygon", {
        "vertices": [[0, 0], [3, 0], [3, 3], [1, 1], [0, 3]]})
    verify = k.apply(polygon.data["object_id"], "verify")
    inside = k.apply(polygon.data["object_id"], "contains", {"point": ["1/2", "1/2"]})
    outside = k.apply(polygon.data["object_id"], "contains", {"point": [1, 2]})
    boundary = k.apply(polygon.data["object_id"], "contains", {"point": [1, 1]})
    assert verify.ok and verify.data["value"]["simple"] is True
    assert verify.data["value"]["convex"] is False
    assert inside.data["value"]["classification"] == "inside"
    assert outside.data["value"]["classification"] == "outside"
    assert boundary.data["value"]["classification"] == "boundary"


def test_convex_polygon_intersection_is_exact_and_retains_both_sources():
    k = MathKernel()
    left = k.object_create("Polygon", {"vertices": [[0, 0], [3, 0], [3, 2], [0, 2]]})
    right = k.object_create("Polygon", {"vertices": [[1, -1], [2, -1], [2, 3], [1, 3]]})
    result = k.apply(left.data["object_id"], "intersection", {"other_id": right.data["object_id"]})
    assert result.ok and result.data["object_type"] == "Polygon"
    vertices = set(k.math_objects[result.data["object_id"]]["value"].vertices)
    assert vertices == {(1, 0), (2, 0), (2, 2), (1, 2)}
    assert set(k.math_objects[result.data["object_id"]]["sources"]) == {
        left.data["object_id"], right.data["object_id"]}


def test_concave_polygon_intersection_fails_closed_as_unsupported():
    k = MathKernel()
    concave = k.object_create("Polygon", {"vertices": [[0, 0], [2, 0], [1, 1], [2, 2], [0, 2]]})
    square = k.object_create("Polygon", {"vertices": [[0, 0], [3, 0], [3, 3], [0, 3]]})
    result = k.apply(concave.data["object_id"], "intersection", {"other_id": square.data["object_id"]})
    assert not result.ok and "verified convex" in result.errors[0]


def test_ear_clipping_triangulates_concave_polygon_with_area_certificate():
    k = MathKernel(); polygon = k.object_create("Polygon", {
        "vertices": [[0, 0], [3, 0], [3, 3], [1, 1], [0, 3]]})
    result = k.apply(polygon.data["object_id"], "triangulate")
    assert result.ok and result.data["object_type"] == "Triangulation"
    assert len(k.math_objects[result.data["object_id"]]["value"].triangles) == 3
    assert all(result.data["details"]["identity_checks"].values())


def test_exact_delaunay_and_cocircular_nonuniqueness():
    k = MathKernel(); generic = point_set(k)
    result = k.apply(generic, "delaunay")
    assert result.ok and result.status == "verified"
    assert len(k.math_objects[result.data["object_id"]]["value"].triangles) == 4
    assert result.data["details"]["identity_checks"] == {
        "empty_circumcircle": True, "planar_triangle_count": True}
    square = point_set(k, [[0, 0], [1, 0], [1, 1], [0, 1]])
    ambiguous = k.apply(square, "delaunay")
    assert ambiguous.ok and ambiguous.data["status"] == "candidate"
    assert ambiguous.data["value"]["classification"] == "ambiguous"
    assert "cocircular" in ambiguous.data["details"]["ambiguity"]


def test_voronoi_is_a_finite_exact_dual_with_unbounded_rays():
    k = MathKernel(); points = point_set(k)
    result = k.apply(points, "voronoi")
    assert result.ok and result.status == "verified" and result.data["object_type"] == "VoronoiDiagram"
    diagram = k.math_objects[result.data["object_id"]]["value"]
    assert len(diagram.vertices) == len(diagram.delaunay_triangles) == 4
    assert len(diagram.edges) == 4 and len(diagram.rays) == 4


def test_halfspace_polytope_contains_and_verifies_listed_vertices():
    k = MathKernel(); cube = k.object_create("Polytope", {"dimension": 3,
        "halfspaces": [{"normal": [1, 0, 0], "bound": 1},
                       {"normal": [-1, 0, 0], "bound": 0},
                       {"normal": [0, 1, 0], "bound": 1},
                       {"normal": [0, -1, 0], "bound": 0},
                       {"normal": [0, 0, 1], "bound": 1},
                       {"normal": [0, 0, -1], "bound": 0}],
        "vertices": [[0, 0, 0], [1, 1, 1]]})
    assert cube.ok
    verify = k.apply(cube.data["object_id"], "verify")
    inside = k.apply(cube.data["object_id"], "contains", {"point": ["1/2", "1/2", "1/2"]})
    outside = k.apply(cube.data["object_id"], "contains", {"point": [2, 0, 0]})
    assert verify.ok and all(verify.data["details"]["identity_checks"].values())
    assert inside.data["value"]["classification"] == "inside"
    assert outside.data["value"]["classification"] == "outside"


def test_triangulation_verifier_checks_orientation_and_edge_incidence():
    k = MathKernel(); triangulation = k.object_create("Triangulation", {
        "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "triangles": [[0, 1, 2], [0, 2, 3]]})
    result = k.apply(triangulation.data["object_id"], "verify")
    assert result.ok and result.status == "verified"
    assert result.data["details"]["identity_checks"] == {
        "predicates_decided": True, "nondegenerate_ccw": True,
        "manifold_edge_incidence": True, "nonoverlapping_interiors": True}


def test_triangulation_verifier_rejects_crossing_and_nested_faces():
    k = MathKernel()
    crossing = k.object_create("Triangulation", {
        "points": [[0, 0], [3, 0], [2, 2], [0, 2]],
        "triangles": [[0, 1, 2], [0, 1, 3]]})
    nested = k.object_create("Triangulation", {
        "points": [[0, 0], [4, 0], [0, 4], [1, 1], [2, 1], [1, 2]],
        "triangles": [[0, 1, 2], [3, 4, 5]]})
    for source in (crossing, nested):
        result = k.apply(source.data["object_id"], "verify")
        assert result.ok and result.status == "refuted"
        assert result.data["verification"]["nonoverlapping_interiors"] is False


def test_duplicate_triangulation_faces_are_rejected_at_construction():
    k = MathKernel()
    result = k.object_create("Triangulation", {
        "points": [[0, 0], [1, 0], [0, 1]],
        "triangles": [[0, 1, 2], [2, 0, 1]]})
    assert not result.ok and "duplicate triangles" in result.errors[0]


def test_computational_geometry_objects_and_duals_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "e3.sqlite"))
    k = MathKernel(settings); points = point_set(k)
    hull = k.apply(points, "convex_hull"); delaunay = k.apply(points, "delaunay")
    voronoi = k.apply(points, "voronoi")
    restarted = MathKernel(settings)
    for result, typ in ((hull, "Polygon"), (delaunay, "Triangulation"),
                        (voronoi, "VoronoiDiagram")):
        restored = restarted.object_get(result.data["object_id"])
        assert restored.ok and restored.data["object_type"] == typ
        assert restarted.math_objects[result.data["object_id"]]["sources"] == [points]


def test_computational_geometry_resource_limits_and_capability_discovery():
    k = MathKernel(Settings(max_geometry_points=3))
    too_many = k.object_create("PointSet", {"dimension": 2,
        "points": [[0, 0], [1, 0], [0, 1], [1, 1]]})
    assert not too_many.ok and "max_geometry_points" in too_many.errors[0]
    k = MathKernel(Settings(max_geometry_work=100))
    points = point_set(k, GENERIC_POINTS)
    blocked = k.apply(points, "delaunay")
    assert not blocked.ok and "max_geometry_work" in blocked.errors[0]
    manifest = MathKernel().capability_query(domain="geometry")
    assert manifest["count"] == 44
    operations = {(entry["input_types"][0], entry["operation"]): entry
                  for entry in manifest["capabilities"]}
    assert "adaptive_incircle_filter" in operations[("PointSet", "incircle")]["verification_methods"]
    assert "area_partition" in operations[("Polygon", "triangulate")]["verification_methods"]


def test_decimal_geometry_never_upgrades_to_exact():
    k = MathKernel(); points = point_set(k, [["0.0", 0], [2, 0], [0, 2]])
    hull = k.apply(points, "convex_hull")
    assert hull.ok and hull.trust == TrustLevel.NUMERIC
    assert k.math_objects[hull.data["object_id"]]["input_trust"] == TrustLevel.NUMERIC


def test_live_mcp_computational_geometry_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            points = (await client.call_tool("math_object_create", {"object_type": "PointSet",
                "definition": {"dimension": 2, "points": GENERIC_POINTS}})).data
            return (await client.call_tool("math_apply", {
                "object_id": points["data"]["object_id"], "operation": "delaunay",
                "parameters": {}})).data
    result = asyncio.run(run())
    assert result["ok"] and result["data"]["object_type"] == "Triangulation"
