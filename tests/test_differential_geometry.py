import asyncio

import pytest
import sympy as sp

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.differential_geometry import Metric, _all_zero, cache_info, curvature


def create_geometry(kernel, *, name="plane", coordinates=("r", "theta"),
                    components=((1, 0), (0, "r^2")), domain=("r>0",)):
    manifold = kernel.object_create("Manifold", {"name": name, "dimension": len(coordinates)})
    assert manifold.ok, manifold.errors
    chart = kernel.object_create("Chart", {"name": "chart", "manifold_id": manifold.data["object_id"],
        "coordinates": list(coordinates), "domain": list(domain)})
    assert chart.ok, chart.errors
    metric = kernel.object_create("Metric", {"chart_id": chart.data["object_id"],
        "components": [list(row) for row in components]})
    assert metric.ok, metric.errors
    return manifold.data["object_id"], chart.data["object_id"], metric.data["object_id"]


def tensor_entries(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_entries(item)
    else:
        yield value


def test_typed_manifold_chart_metric_construction_and_ancestry():
    k = MathKernel()
    manifold, chart, metric = create_geometry(k)
    assert k.math_objects[chart]["sources"] == [manifold]
    assert k.math_objects[metric]["sources"] == [chart]
    before = k.object_get(metric).model_dump(mode="json")
    result = k.apply(metric, "inverse_metric")
    assert result.ok and result.data["object_type"] == "GeometryTensor"
    assert k.math_objects[result.data["object_id"]]["sources"] == [metric]
    assert k.object_get(metric).model_dump(mode="json") == before
    forbidden = k.object_create("GeometryTensor", {"components": []})
    assert not forbidden.ok


def test_polar_connection_is_metric_compatible_and_torsion_free():
    k = MathKernel(); _, _, metric = create_geometry(k)
    result = k.apply(metric, "christoffel")
    assert result.ok and result.status == "verified"
    checks = result.data["details"]["identity_checks"]
    assert checks == {"torsion_free": True, "metric_compatible": True}
    gamma = k.math_objects[result.data["object_id"]]["value"].coefficients
    r = sp.Symbol("r", real=True)
    assert gamma[0][1][1] == -r
    assert gamma[1][0][1] == gamma[1][1][0] == 1/r


def test_flat_polar_metric_has_zero_riemann_tensor():
    k = MathKernel(); _, _, metric = create_geometry(k)
    result = k.apply(metric, "riemann")
    assert result.ok and result.trust == TrustLevel.SYMBOLIC
    tensor = k.math_objects[result.data["object_id"]]["value"]
    assert all(sp.simplify(value) == 0 for value in tensor_entries(tensor.components))
    assert all(result.data["details"]["identity_checks"].values())


def test_unit_sphere_curvature_and_bianchi_identities():
    k = MathKernel()
    _, _, metric = create_geometry(k, name="S2", coordinates=("theta", "phi"),
        components=((1, 0), (0, "sin(theta)^2")), domain=("theta>0", "theta<pi"))
    ricci = k.apply(metric, "ricci")
    scalar = k.apply(metric, "scalar_curvature")
    einstein = k.apply(metric, "einstein")
    assert ricci.ok and scalar.ok and einstein.ok
    assert scalar.data["value"] == "2"
    ricci_tensor = k.math_objects[ricci.data["object_id"]]["value"]
    theta = sp.Symbol("theta", real=True)
    assert ricci_tensor.components == ((1, 0), (0, sp.sin(theta)**2))
    einstein_tensor = k.math_objects[einstein.data["object_id"]]["value"]
    assert all(value == 0 for value in tensor_entries(einstein_tensor.components))
    assert einstein.data["details"]["identity_checks"]["contracted_bianchi"]
    assert ricci.data["details"]["identity_checks"]["first_bianchi"]
    assert ricci.data["details"]["identity_checks"]["lowered_riemann_symmetries"]


def test_conformal_plane_matches_independent_closed_curvature_formula():
    k = MathKernel()
    factor = "exp(2*(x^2+y^2))"
    _, _, metric = create_geometry(k, name="conformal", coordinates=("x", "y"),
        components=((factor, 0), (0, factor)), domain=())
    result = k.apply(metric, "scalar_curvature")
    x, y = sp.symbols("x y", real=True)
    expected = -8*sp.exp(-2*(x**2+y**2))
    assert result.ok and sp.simplify(k.math_objects[metric]["value"].coordinates[0]-x) == 0
    raw = curvature(k.math_objects[metric]["value"], metric, "scalar_curvature")
    assert sp.simplify(raw.value - expected) == 0


def test_sphere_geodesic_equations_use_explicit_coordinate_velocity_order():
    k = MathKernel()
    _, _, metric = create_geometry(k, name="S2", coordinates=("theta", "phi"),
        components=((1, 0), (0, "sin(theta)^2")), domain=("theta>0", "theta<pi"))
    result = k.apply(metric, "geodesic_equations")
    system = k.math_objects[result.data["object_id"]]["value"]
    theta = sp.Symbol("theta", real=True); dtheta = sp.Symbol("d_theta", real=True); dphi = sp.Symbol("d_phi", real=True)
    assert system.coordinates == (theta, sp.Symbol("phi", real=True))
    assert system.velocity_symbols == (dtheta, dphi)
    assert sp.trigsimp(system.accelerations[0] - dphi**2*sp.sin(2*theta)/2) == 0
    assert sp.trigsimp(system.accelerations[1] + 2*dphi*dtheta/sp.tan(theta)) == 0


def test_inverse_metric_records_nondegeneracy_condition_and_identity():
    k = MathKernel(); _, _, metric = create_geometry(k)
    result = k.apply(metric, "inverse_metric")
    assert result.ok and result.data["details"]["identity_checks"]["left_inverse"]
    assert result.side_conditions == ["r > 0", "Ne(r**2, 0)"]
    tensor = k.math_objects[result.data["object_id"]]["value"]
    assert tensor.variance == ("up", "up")
    assert tensor.components == ((1, 0), (0, 1/sp.Symbol("r", real=True)**2))


def test_geometry_objects_and_derived_tensors_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path/"geometry.sqlite"))
    k = MathKernel(settings); manifold, chart, metric = create_geometry(k)
    curvature = k.apply(metric, "ricci")
    restarted = MathKernel(settings)
    for object_id, typ in ((manifold, "Manifold"), (chart, "Chart"), (metric, "Metric"),
                           (curvature.data["object_id"], "GeometryTensor")):
        restored = restarted.object_get(object_id)
        assert restored.ok and restored.data["object_type"] == typ
    assert restarted.math_objects[metric]["sources"] == [chart]
    replay = restarted.apply(metric, "scalar_curvature")
    assert replay.ok and replay.data["value"] == "0"


def test_decimal_metric_ancestry_cannot_become_symbolic_or_exact():
    k = MathKernel()
    _, _, metric = create_geometry(k, components=(("1.0", 0), (0, "r^2")))
    result = k.apply(metric, "scalar_curvature")
    assert result.ok and result.trust == TrustLevel.NUMERIC


@pytest.mark.parametrize("components,message", [
    (((1, 1), (0, 1)), "symmetric"),
    (((1, 0), (0, 0)), "degenerate"),
])
def test_invalid_metrics_fail_before_curvature_work(components, message):
    k = MathKernel()
    manifold = k.object_create("Manifold", {"name": "bad", "dimension": 2})
    chart = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                      "coordinates": ["x", "y"]})
    metric = k.object_create("Metric", {"chart_id": chart.data["object_id"],
                                         "components": components})
    assert not metric.ok and message in metric.errors[0]


def test_chart_dimension_and_coordinate_uniqueness_are_enforced():
    k = MathKernel(); manifold = k.object_create("Manifold", {"name": "M", "dimension": 2})
    short = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                      "coordinates": ["x"]})
    duplicate = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                          "coordinates": ["x", "x"]})
    assert not short.ok and "coordinate count" in short.errors[0]
    assert not duplicate.ok and "unique" in duplicate.errors[0]


def test_chart_domains_and_metric_reality_are_validated():
    k = MathKernel(); manifold = k.object_create("Manifold", {"name": "M", "dimension": 2})
    bad_domain = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                           "coordinates": ["x", "y"], "domain": ["x"]})
    assert not bad_domain.ok and "domain conditions" in bad_domain.errors[0]
    chart = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                      "coordinates": ["x", "y"]})
    complex_metric = k.object_create("Metric", {"chart_id": chart.data["object_id"],
                                                 "components": [["sqrt(-1)", 0], [0, 1]]})
    assert not complex_metric.ok and "real-valued" in complex_metric.errors[0]


def test_undecidable_symbolic_zero_is_unknown_not_refuted():
    assert _all_zero([sp.Symbol("undecided")]) is None


def test_geometry_resource_limits_fail_closed():
    k = MathKernel(Settings(max_geometry_dimension=1))
    result = k.object_create("Manifold", {"name": "too-large", "dimension": 2})
    assert not result.ok and "max_geometry_dimension" in result.errors[0]
    k = MathKernel(Settings(max_geometry_work=20))
    manifold = k.object_create("Manifold", {"name": "M", "dimension": 2})
    chart = k.object_create("Chart", {"name": "c", "manifold_id": manifold.data["object_id"],
                                      "coordinates": ["x", "y"]})
    metric = k.object_create("Metric", {"chart_id": chart.data["object_id"],
                                         "components": [[1, 0], [0, 1]]})
    assert not metric.ok and "max_geometry_work" in metric.errors[0]


def test_geometry_capabilities_are_truthful_and_typed():
    k = MathKernel()
    manifest = k.capability_query(domain="geometry", input_type="Metric")
    assert manifest["count"] == 7
    by_operation = {item["operation"]: item for item in manifest["capabilities"]}
    assert "GeometryTensor" in by_operation["riemann"]["output_types"]
    assert "Connection" in by_operation["christoffel"]["output_types"]
    assert by_operation["scalar_curvature"]["output_types"] == ["EngineeringResult"]
    assert "first_bianchi" in by_operation["riemann"]["verification_methods"]


def test_cached_curvature_reuses_exact_symbolic_bundle():
    k = MathKernel()
    _, _, metric = create_geometry(k, name="S2", coordinates=("theta", "phi"),
        components=((1, 0), (0, "sin(theta)^2")), domain=("theta>0", "theta<pi"))
    from mathkernel.engines import run_with_timeout
    before = run_with_timeout(cache_info, 5)["curvature"]["hits"]
    assert k.apply(metric, "ricci").ok
    assert k.apply(metric, "scalar_curvature").ok
    after = run_with_timeout(cache_info, 5)["curvature"]["hits"]
    assert after >= before + 1


def test_live_mcp_geometry_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            manifold = (await client.call_tool("math_object_create", {"object_type": "Manifold",
                "definition": {"name": "S2", "dimension": 2}})).data
            chart = (await client.call_tool("math_object_create", {"object_type": "Chart",
                "definition": {"name": "spherical", "manifold_id": manifold["data"]["object_id"],
                               "coordinates": ["theta", "phi"]}})).data
            metric = (await client.call_tool("math_object_create", {"object_type": "Metric",
                "definition": {"chart_id": chart["data"]["object_id"],
                               "components": [[1, 0], [0, "sin(theta)^2"]]}})).data
            return (await client.call_tool("math_apply", {"object_id": metric["data"]["object_id"],
                "operation": "scalar_curvature", "parameters": {}})).data
    result = asyncio.run(run())
    assert result["ok"] and result["data"]["value"] == "2"
