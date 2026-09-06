import asyncio

import pytest
import sympy as sp

from mathkernel import (
    GLMFit, GeneralizedLinearModel, MathKernel, Settings, StatisticalSample,
    TrustLevel,
)


def sample(kernel, rows, variables=("x", "y"), **metadata):
    result = kernel.object_create("StatisticalSample", {
        "variables": list(variables), "observations": rows, **metadata})
    assert result.ok, result.errors
    return result.data["object_id"]


def model(kernel, sample_id, family="gaussian", link="identity", **extra):
    result = kernel.object_create("GeneralizedLinearModel", {
        "sample_id": sample_id, "response": "y", "predictors": ["x"],
        "family": family, "link": link, **extra})
    assert result.ok, result.errors
    return result.data["object_id"]


def fit(kernel, model_id, parameters=None):
    result = kernel.apply(model_id, "fit", parameters or {})
    assert result.ok, result.errors
    return result, result.data["object_id"]


def test_exact_gaussian_identity_fit_has_exact_coefficients_covariance_and_deviance():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [[0, 1], [1, 3], [2, 5], [3, 7]]))
    result, fit_id = fit(kernel, model_id)
    fitted = kernel.math_objects[fit_id]["value"]
    assert isinstance(fitted, GLMFit)
    assert result.trust == TrustLevel.EXACT
    assert fitted.coefficient_names == ("intercept", "x")
    assert fitted.coefficients == (1, 2)
    assert fitted.deviance == fitted.null_deviance * 0 == 0
    assert fitted.covariance == ((0, 0), (0, 0))
    assert result.data["details"]["model_validity"] == "not_established"


def test_gaussian_normal_equations_and_fit_can_be_reverified_from_source_data():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [[0, 1], [1, 2], [2, 2], [3, 5], [4, 4]]))
    _, fit_id = fit(kernel, model_id)
    verification = kernel.apply(fit_id, "verify")
    assert verification.ok and verification.status == "verified"
    assert verification.data["verification"] == {
        "normal_equations": True, "deviance_recomputed": True,
        "coefficient_names_match": True}


def test_binomial_logit_irls_records_numeric_convergence_score_and_model_evidence():
    kernel = MathKernel()
    rows = [[0, 0], [1, 0], [2, 1], [3, 0], [4, 1],
            [5, 1], [6, 1], [7, 0], [8, 1], [9, 1]]
    model_id = model(kernel, sample(kernel, rows), "binomial", "logit",
                     model_assumptions=["observations are conditionally independent"])
    result, fit_id = fit(kernel, model_id)
    fitted = kernel.math_objects[fit_id]["value"]
    assert result.trust == TrustLevel.NUMERIC and fitted.converged
    assert fitted.iterations <= 100 and float(fitted.score_residual) < 1e-7
    assert all(0 < float(item) < 1 for item in fitted.fitted_means)
    bundle = result.claim_evidence["fit"]
    assert bundle.numerical[0].convergence["converged"] is True
    assert bundle.model[0].role == "diagnostic"
    assert bundle.model[0].metadata["causal_claim"] == "not_established"


def test_poisson_log_irls_records_deviance_covariance_and_predictions():
    kernel = MathKernel()
    rows = [[0, 1], [1, 1], [2, 2], [3, 2],
            [4, 4], [5, 5], [6, 8], [7, 9]]
    model_id = model(kernel, sample(kernel, rows), "poisson", "log")
    result, fit_id = fit(kernel, model_id)
    fitted = kernel.math_objects[fit_id]["value"]
    assert result.trust == TrustLevel.NUMERIC
    assert 0 <= float(fitted.deviance) < float(fitted.null_deviance)
    assert fitted.covariance[0][1] == fitted.covariance[1][0]
    prediction = kernel.apply(fit_id, "predict", {"rows": [[8], [10]]})
    means = [float(item) for item in prediction.data["value"]["conditional_means"]]
    assert prediction.ok and prediction.trust == TrustLevel.NUMERIC
    assert 0 < means[0] < means[1]
    assert prediction.data["details"]["prediction_scope"] == "conditional_mean_only"


@pytest.mark.parametrize("definition,message", [
    ({"response": "missing", "predictors": ["x"], "family": "gaussian",
      "link": "identity"}, "response"),
    ({"response": "y", "predictors": ["missing"], "family": "gaussian",
      "link": "identity"}, "predictor"),
    ({"response": "y", "predictors": ["x", "x"], "family": "gaussian",
      "link": "identity"}, "unique"),
    ({"response": "y", "predictors": ["y"], "family": "gaussian",
      "link": "identity"}, "also be a predictor"),
    ({"response": "y", "predictors": ["x"], "family": "binomial",
      "link": "identity"}, "only logit"),
    ({"response": "y", "predictors": ["x"], "family": "poisson",
      "link": "log", "include_intercept": 1}, "boolean"),
])
def test_invalid_glm_specifications_fail_closed(definition, message):
    kernel = MathKernel(); sample_id = sample(kernel, [[0, 1], [1, 2], [2, 3]])
    result = kernel.object_create("GLM", {"sample_id": sample_id, **definition})
    assert not result.ok and message in result.errors[0]


@pytest.mark.parametrize("family,link,rows", [
    ("binomial", "logit", [[0, 0], [1, 2], [2, 1]]),
    ("binomial", "logit", [[0, 1], [1, 1], [2, 1]]),
    ("poisson", "log", [[0, 1], [1, -1], [2, 2]]),
    ("poisson", "log", [[0, 0], [1, 0], [2, 0]]),
    ("poisson", "log", [[0, 1], [1, "1.5"], [2, 2]]),
])
def test_family_response_domains_and_boundary_mles_are_refused(family, link, rows):
    kernel = MathKernel(); sample_id = sample(kernel, rows)
    result = kernel.object_create("GLM", {
        "sample_id": sample_id, "response": "y", "predictors": ["x"],
        "family": family, "link": link})
    assert not result.ok and ("domain" in result.errors[0] or "degenerate" in result.errors[0])


def test_rank_deficiency_is_refuted_and_fit_creates_no_derived_object():
    kernel = MathKernel()
    sample_id = sample(kernel, [[0, 0, 1], [1, 2, 2], [2, 4, 3], [3, 6, 4]],
                       variables=("x", "twox", "y"))
    created = kernel.object_create("GLM", {
        "sample_id": sample_id, "response": "y", "predictors": ["x", "twox"],
        "family": "gaussian", "link": "identity"})
    assert created.ok
    model_id = created.data["object_id"]
    verified = kernel.apply(model_id, "verify")
    assert verified.status == "refuted"
    assert verified.data["verification"]["design_full_column_rank"] is False
    before = len(kernel.math_objects)
    failed = kernel.apply(model_id, "fit")
    assert not failed.ok and "design_full_column_rank" in failed.errors[0]
    assert len(kernel.math_objects) == before


def test_complete_separation_is_refused_without_persisting_a_fit():
    kernel = MathKernel()
    rows = [[-3, 0], [-2, 0], [-1, 0], [1, 1], [2, 1], [3, 1]]
    model_id = model(kernel, sample(kernel, rows), "binomial", "logit")
    before = len(kernel.math_objects)
    result = kernel.apply(model_id, "fit")
    assert not result.ok and "separation" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_iteration_and_work_limits_fail_before_creating_fit_objects():
    kernel = MathKernel(Settings(max_glm_iterations=1))
    model_id = model(kernel, sample(kernel, [[0, 1], [1, 1], [2, 2], [3, 4]]),
                     "poisson", "log")
    before = len(kernel.math_objects)
    result = kernel.apply(model_id, "fit", {"max_iterations": 2})
    assert not result.ok and "max_glm_iterations" in result.errors[0]
    assert len(kernel.math_objects) == before
    constrained = MathKernel(Settings(max_glm_work=1))
    mid = model(constrained, sample(constrained, [[0, 1], [1, 2], [2, 3]]))
    assert not constrained.apply(mid, "fit").ok


def test_actual_irls_nonconvergence_is_refused_without_a_fit():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [
        [0, 1], [1, 1], [2, 2], [3, 2], [4, 4], [5, 5], [6, 8], [7, 9]]),
        "poisson", "log")
    before = len(kernel.math_objects)
    result = kernel.apply(model_id, "fit", {"max_iterations": 1})
    assert not result.ok and "did not converge" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_intercept_only_models_and_empty_prediction_rows_are_explicitly_supported():
    kernel = MathKernel()
    sample_id = sample(kernel, [[0, 0], [1, 1], [2, 0], [3, 1]])
    made = kernel.object_create("GLM", {
        "sample_id": sample_id, "response": "y", "predictors": [],
        "family": "binomial", "link": "logit"})
    assert made.ok
    _, fit_id = fit(kernel, made.data["object_id"])
    prediction = kernel.apply(fit_id, "predict", {"rows": [[], []]})
    assert prediction.ok
    assert [float(item) for item in prediction.data["value"]["conditional_means"]] == [.5, .5]


def test_decimal_ancestry_cannot_upgrade_through_numeric_gaussian_fit():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [["0.0", "1.1"], ["1.0", "2.9"],
                                                    ["2.0", "5.2"], ["3.0", "6.8"]]))
    result, fit_id = fit(kernel, model_id)
    assert result.trust == TrustLevel.NUMERIC
    assert kernel.math_objects[fit_id]["input_trust"] == TrustLevel.NUMERIC


def test_exact_gaussian_prediction_preserves_exact_mathir_and_rejects_bad_rows():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [[0, 1], [1, 3], [2, 5], [3, 7]]))
    _, fit_id = fit(kernel, model_id)
    prediction = kernel.apply(fit_id, "predict", {"rows": [["1/2"], [4]]})
    assert prediction.trust == TrustLevel.EXACT
    assert prediction.data["value"]["conditional_means"] == ["2", "9"]
    before = len(kernel.math_objects)
    assert not kernel.apply(fit_id, "predict", {"rows": [[1, 2]]}).ok
    assert not kernel.apply(fit_id, "predict", {"rows": [[True]]}).ok
    assert len(kernel.math_objects) == before


def test_poisson_prediction_refuses_float64_exponential_overflow_range():
    kernel = MathKernel()
    model_id = model(kernel, sample(kernel, [
        [0, 1], [1, 1], [2, 2], [3, 2], [4, 4], [5, 5], [6, 8], [7, 9]]),
        "poisson", "log")
    _, fit_id = fit(kernel, model_id)
    result = kernel.apply(fit_id, "predict", {"rows": [[1_000_000]]})
    assert not result.ok and "exponential range" in result.errors[0]


def test_glm_diagnostics_are_computational_not_model_validity_proof():
    kernel = MathKernel()
    _, fit_id = fit(kernel, model(kernel, sample(
        kernel, [[0, 1], [1, 2], [2, 2], [3, 5], [4, 4]])))
    result = kernel.apply(fit_id, "diagnostics")
    assert result.ok and result.data["value"]["model_validity"] == "not_established"
    assert result.data["details"]["diagnostics_do_not_prove_model_validity"] is True
    assert result.claim_evidence["diagnostics"].model[0].trust == "unknown"


def test_glm_model_fit_and_multihop_lineage_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "glm.sqlite"))
    kernel = MathKernel(settings)
    sample_id = sample(kernel, [[0, 1], [1, 3], [2, 5], [3, 7]])
    model_id = model(kernel, sample_id)
    _, fit_id = fit(kernel, model_id)
    restarted = MathKernel(settings)
    assert restarted.object_get(model_id).data["sources"] == [sample_id]
    assert restarted.object_get(fit_id).data["sources"] == [model_id]
    assert restarted.apply(fit_id, "verify").status == "verified"


def test_glm_derived_type_is_output_only_and_public_types_are_immutable():
    kernel = MathKernel()
    assert not kernel.object_create("GLMFit", {}).ok
    model_id = model(kernel, sample(kernel, [[0, 1], [1, 3], [2, 5]]))
    value = kernel.math_objects[model_id]["value"]
    assert isinstance(value, GeneralizedLinearModel)
    with pytest.raises(Exception):
        value.response = "x"
    assert all(item is not None for item in (StatisticalSample, GeneralizedLinearModel, GLMFit))


def test_glm_capabilities_limits_and_version_are_truthful():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    names = {(item["input_types"][0], item["operation"]): item
             for item in manifest["capabilities"]}
    assert names[("GeneralizedLinearModel", "fit")]["output_types"] == [
        "EngineeringResult", "GLMFit"]
    assert "separation_check" in names[("GeneralizedLinearModel", "fit")]["verification_methods"]
    assert names[("GLMFit", "predict")]["parameter_schema"]["rows"].startswith("MathIR")
    capabilities = kernel.capabilities()
    assert capabilities["version"] == __import__("mathkernel").__version__
    for name in ("max_glm_parameters", "max_glm_iterations",
                 "max_glm_prediction_rows", "max_glm_work"):
        assert name in Settings().limits()



def test_live_mcp_glm_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def run():
        async with Client(mcp) as client:
            made = await client.call_tool("math_object_create", {
                "object_type": "StatisticalSample", "definition": {
                    "variables": ["x", "y"],
                    "observations": [[0, 1], [1, 3], [2, 5], [3, 7]]}})
            sample_id = made.data["data"]["object_id"]
            made = await client.call_tool("math_object_create", {
                "object_type": "GeneralizedLinearModel", "definition": {
                    "sample_id": sample_id, "response": "y", "predictors": ["x"],
                    "family": "gaussian", "link": "identity"}})
            return (await client.call_tool("math_apply", {
                "object_id": made.data["data"]["object_id"],
                "operation": "fit", "parameters": {}})).data

    result = asyncio.run(run())
    assert result["ok"] and result["trust"] == "exact"
    assert result["data"]["value"]["coefficients"] == {"intercept": "1", "x": "2"}
