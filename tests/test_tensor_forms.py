import asyncio

import sympy as sp

from mathkernel import MathKernel, Settings, TrustLevel


def setup_plane(kernel, *, second_chart=True):
    manifold = kernel.object_create("Manifold", {"name": "R2", "dimension": 2})
    xy = kernel.object_create("Chart", {"name": "xy", "manifold_id": manifold.data["object_id"],
        "coordinates": ["x", "y"], "orientation": 1})
    uv = None
    if second_chart:
        uv = kernel.object_create("Chart", {"name": "uv", "manifold_id": manifold.data["object_id"],
            "coordinates": ["u", "v"], "orientation": 1})
    metric = kernel.object_create("Metric", {"chart_id": xy.data["object_id"],
        "components": [[1, 0], [0, 1]], "signature": [2, 0]})
    return manifold.data["object_id"], xy.data["object_id"], (uv.data["object_id"] if uv else None), metric.data["object_id"]


def form_terms(kernel, object_id):
    return {term.indices: term.coefficient
            for term in kernel.math_objects[object_id]["value"].terms}


def test_coordinate_map_has_explicit_jacobian_inverse_checks_and_ancestry():
    k = MathKernel(); _, xy, uv, _ = setup_plane(k)
    mapping = k.object_create("CoordinateMap", {"name": "F", "source_chart_id": uv,
        "target_chart_id": xy, "forward": ["u+v", "u-v"],
        "inverse": ["(x+y)/2", "(x-y)/2"]})
    assert mapping.ok and k.math_objects[mapping.data["object_id"]]["sources"] == [uv, xy]
    jacobian = k.apply(mapping.data["object_id"], "jacobian")
    verify = k.apply(mapping.data["object_id"], "verify")
    assert jacobian.ok and jacobian.data["details"]["determinant"] == "-2"
    assert k.math_objects[jacobian.data["object_id"]]["value"].components == ((1, 1), (1, -1))
    assert all(verify.data["details"]["identity_checks"].values())


def test_coordinate_maps_reject_cross_manifold_and_report_singular_condition():
    k = MathKernel(); _, xy, uv, _ = setup_plane(k)
    other = k.object_create("Manifold", {"name": "N", "dimension": 2})
    other_chart = k.object_create("Chart", {"name": "z", "manifold_id": other.data["object_id"],
        "coordinates": ["a", "b"]})
    cross = k.object_create("CoordinateMap", {"source_chart_id": uv,
        "target_chart_id": other_chart.data["object_id"], "forward": ["u", "v"]})
    assert not cross.ok and "same manifold" in cross.errors[0]
    singular = k.object_create("CoordinateMap", {"source_chart_id": uv,
        "target_chart_id": xy, "forward": ["u", "u"]})
    result = k.apply(singular.data["object_id"], "verify")
    assert result.ok and result.data["details"]["identity_checks"]["jacobian_nonsingular"] is False


def test_covariant_derivative_uses_variance_signs_and_metric_connection():
    k = MathKernel(); manifold = k.object_create("Manifold", {"name": "polar", "dimension": 2})
    chart = k.object_create("Chart", {"name": "polar", "manifold_id": manifold.data["object_id"],
        "coordinates": ["r", "theta"], "domain": ["r>0"]})
    metric = k.object_create("Metric", {"chart_id": chart.data["object_id"],
        "components": [[1, 0], [0, "r^2"]]})
    covector = k.object_create("TensorField", {"name": "dr", "chart_id": chart.data["object_id"],
        "variance": ["down"], "components": [1, 0]})
    result = k.apply(covector.data["object_id"], "covariant_derivative",
                     {"metric_id": metric.data["object_id"]})
    assert result.ok and result.data["details"]["identity_checks"]["connection_metric_compatible"]
    tensor = k.math_objects[result.data["object_id"]]["value"]
    r = sp.Symbol("r", real=True)
    assert tensor.variance == ("down", "down")
    assert tensor.components == ((0, 0), (0, r))


def test_lie_derivative_of_covector_matches_coordinate_formula():
    k = MathKernel(); _, xy, _, _ = setup_plane(k, second_chart=False)
    vector = k.object_create("TensorField", {"name": "X", "chart_id": xy,
        "variance": ["up"], "components": ["x", "-y"]})
    covector = k.object_create("TensorField", {"name": "a", "chart_id": xy,
        "variance": ["down"], "components": ["y", "x"]})
    result = k.apply(covector.data["object_id"], "lie_derivative",
                     {"vector_field_id": vector.data["object_id"]})
    assert result.ok
    assert k.math_objects[result.data["object_id"]]["value"].components == (0, 0)


def test_wedge_exterior_derivative_and_d_squared_zero():
    k = MathKernel(); _, xy, _, _ = setup_plane(k, second_chart=False)
    alpha = k.object_create("DifferentialForm", {"name": "alpha", "chart_id": xy,
        "degree": 1, "terms": [{"indices": [0], "coefficient": "x*y"},
                                 {"indices": [1], "coefficient": "x^2"}]})
    beta = k.object_create("DifferentialForm", {"name": "beta", "chart_id": xy,
        "degree": 1, "terms": [{"indices": [0], "coefficient": "y"},
                                 {"indices": [1], "coefficient": 1}]})
    wedge = k.apply(alpha.data["object_id"], "wedge", {"other_id": beta.data["object_id"]})
    derivative = k.apply(alpha.data["object_id"], "exterior_derivative")
    assert wedge.ok and wedge.data["details"]["identity_checks"]["graded_commutativity"]
    x, y = sp.symbols("x y", real=True)
    assert sp.expand(form_terms(k, wedge.data["object_id"])[(0, 1)] - (x*y - x**2*y)) == 0
    assert form_terms(k, derivative.data["object_id"])[(0, 1)] == x
    assert derivative.data["details"]["identity_checks"]["d_squared_zero"] is True
    second = k.apply(derivative.data["object_id"], "exterior_derivative")
    assert second.ok and not k.math_objects[second.data["object_id"]]["value"].terms


def test_interior_product_contracts_a_form_with_a_vector():
    k = MathKernel(); _, xy, _, _ = setup_plane(k, second_chart=False)
    volume = k.object_create("DifferentialForm", {"name": "vol", "chart_id": xy,
        "degree": 2, "terms": [{"indices": [0, 1], "coefficient": 1}]})
    vector = k.object_create("TensorField", {"name": "X", "chart_id": xy,
        "variance": ["up"], "components": ["x", "y"]})
    result = k.apply(volume.data["object_id"], "interior_product",
                     {"vector_field_id": vector.data["object_id"]})
    x, y = sp.symbols("x y", real=True)
    assert result.ok and form_terms(k, result.data["object_id"]) == {(0,): -y, (1,): x}


def test_form_pullback_uses_jacobian_minors_and_commutes_with_d():
    k = MathKernel(); _, xy, uv, _ = setup_plane(k)
    mapping = k.object_create("CoordinateMap", {"source_chart_id": uv, "target_chart_id": xy,
        "forward": ["u*v", "v"]})
    form = k.object_create("DifferentialForm", {"name": "omega", "chart_id": xy,
        "degree": 1, "terms": [{"indices": [0], "coefficient": "y"},
                                 {"indices": [1], "coefficient": "x"}]})
    result = k.apply(form.data["object_id"], "pullback", {"map_id": mapping.data["object_id"]})
    u, v = sp.symbols("u v", real=True)
    assert result.ok and form_terms(k, result.data["object_id"]) == {(0,): v**2, (1,): 2*u*v}
    assert result.data["details"]["identity_checks"]["pullback_commutes_with_d"] is True
    assert set(k.math_objects[result.data["object_id"]]["sources"]) == {form.data["object_id"], mapping.data["object_id"]}


def test_hodge_star_tracks_orientation_signature_and_double_star():
    k = MathKernel(); _, xy, _, metric = setup_plane(k, second_chart=False)
    form = k.object_create("DifferentialForm", {"name": "alpha", "chart_id": xy,
        "degree": 1, "terms": [{"indices": [0], "coefficient": "x"},
                                 {"indices": [1], "coefficient": "y"}]})
    first = k.apply(form.data["object_id"], "hodge_star", {"metric_id": metric})
    x, y = sp.symbols("x y", real=True)
    assert first.ok and form_terms(k, first.data["object_id"]) == {(0,): -y, (1,): x}
    assert first.data["details"]["identity_checks"]["double_star"] is True
    second = k.apply(first.data["object_id"], "hodge_star", {"metric_id": metric})
    assert form_terms(k, second.data["object_id"]) == {(0,): -x, (1,): -y}


def test_form_objects_persist_with_all_dependency_sources(tmp_path):
    settings = Settings(store_path=str(tmp_path / "e2.sqlite"))
    k = MathKernel(settings); _, xy, uv, metric = setup_plane(k)
    mapping = k.object_create("CoordinateMap", {"source_chart_id": uv, "target_chart_id": xy,
        "forward": ["u", "v"], "inverse": ["x", "y"]})
    form = k.object_create("DifferentialForm", {"chart_id": xy, "degree": 1,
        "terms": [{"indices": [0], "coefficient": "x"}]})
    pulled = k.apply(form.data["object_id"], "pullback", {"map_id": mapping.data["object_id"]})
    starred = k.apply(form.data["object_id"], "hodge_star", {"metric_id": metric})
    restarted = MathKernel(settings)
    for result, typ in ((mapping, "CoordinateMap"), (form, "DifferentialForm"),
                        (pulled, "DifferentialForm"), (starred, "DifferentialForm")):
        restored = restarted.object_get(result.data["object_id"])
        assert restored.ok and restored.data["object_type"] == typ
    assert set(restarted.math_objects[pulled.data["object_id"]]["sources"]) == {
        form.data["object_id"], mapping.data["object_id"]}


def test_form_limits_validation_capabilities_and_derived_construction_boundary():
    k = MathKernel(Settings(max_geometry_rank=1)); _, xy, _, _ = setup_plane(k, second_chart=False)
    too_high = k.object_create("TensorField", {"chart_id": xy,
        "variance": ["up", "down"], "components": [[1, 0], [0, 1]]})
    assert not too_high.ok and "max_geometry_rank" in too_high.errors[0]
    bad_form = k.object_create("DifferentialForm", {"chart_id": xy, "degree": 1,
        "terms": [{"indices": [1, 0], "coefficient": 1}]})
    assert not bad_form.ok and "index count" in bad_form.errors[0]
    assert not k.object_create("JacobianMap", {}).ok
    manifest = k.capability_query(domain="geometry")
    assert manifest["count"] == 44
    operations = {entry["operation"]: entry for entry in manifest["capabilities"]}
    assert "d_squared_zero" in operations["exterior_derivative"]["verification_methods"]
    assert "pullback_commutes_with_d" in operations["pullback"]["verification_methods"]
    invalid_orientation = k.object_create("Chart", {"name": "bad",
        "manifold_id": k.math_objects[xy]["value"].manifold_id,
        "coordinates": ["u", "v"], "orientation": True})
    assert not invalid_orientation.ok and "orientation" in invalid_orientation.errors[0]


def test_decimal_form_and_map_ancestry_cannot_upgrade_to_symbolic():
    k = MathKernel(); _, xy, uv, _ = setup_plane(k)
    mapping = k.object_create("CoordinateMap", {"source_chart_id": uv, "target_chart_id": xy,
        "forward": ["1.0*u", "v"]})
    form = k.object_create("DifferentialForm", {"chart_id": xy, "degree": 1,
        "terms": [{"indices": [0], "coefficient": "x"}]})
    result = k.apply(form.data["object_id"], "pullback", {"map_id": mapping.data["object_id"]})
    assert result.ok and result.trust == TrustLevel.NUMERIC


def test_live_mcp_form_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            manifold = (await client.call_tool("math_object_create", {"object_type": "Manifold",
                "definition": {"name": "R2", "dimension": 2}})).data["data"]["object_id"]
            chart = (await client.call_tool("math_object_create", {"object_type": "Chart",
                "definition": {"name": "xy", "manifold_id": manifold,
                               "coordinates": ["x", "y"]}})).data["data"]["object_id"]
            form = (await client.call_tool("math_object_create", {"object_type": "DifferentialForm",
                "definition": {"chart_id": chart, "degree": 1,
                               "terms": [{"indices": [0], "coefficient": "x*y"}]}})).data["data"]["object_id"]
            return (await client.call_tool("math_apply", {"object_id": form,
                "operation": "exterior_derivative", "parameters": {}})).data
    result = asyncio.run(run())
    assert result["ok"] and result["data"]["details"]["identity_checks"]["d_squared_zero"]


