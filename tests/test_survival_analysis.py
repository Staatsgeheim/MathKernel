import asyncio
from pathlib import Path

import pytest
import numpy as np
import sympy as sp

from mathkernel import (
    CoxPHFit, KaplanMeierEstimate, MathKernel, Settings, SurvivalDataset,
    TrustLevel,
)
from mathkernel.survival_analysis import _concordance, _cox_components


def make_sample(kernel, variables, rows, **metadata):
    result = kernel.object_create("StatisticalSample", {
        "variables": variables, "observations": rows, **metadata})
    assert result.ok, result.errors
    return result.data["object_id"]


def make_dataset(kernel, rows=None, **definition):
    rows = rows or [[1, 1], [2, 1], [2, 0], [3, 1]]
    sample_id = make_sample(kernel, ["time", "event"], rows)
    result = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        **definition})
    assert result.ok, result.errors
    return sample_id, result.data["object_id"]


def cox_rows():
    rows = []
    for index in range(40):
        predictor = index % 4
        duration = max(20 - index // 4 - 2 * predictor + index % 3, 1)
        event = 0 if index % 7 == 0 else 1
        rows.append([duration, event, predictor])
    return rows


def make_cox(kernel, tie_method="efron", rows=None):
    sample_id = make_sample(kernel, ["time", "event", "x"], rows or cox_rows())
    dataset = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event"})
    assert dataset.ok, dataset.errors
    model = kernel.object_create("CoxPHModel", {
        "dataset_id": dataset.data["object_id"], "predictors": ["x"],
        "tie_method": tie_method})
    assert model.ok, model.errors
    return sample_id, dataset.data["object_id"], model.data["object_id"]


def test_survival_dataset_is_source_linked_and_verified():
    kernel = MathKernel(); sample_id, dataset_id = make_dataset(kernel)
    stored = kernel.math_objects[dataset_id]["value"]
    assert isinstance(stored, SurvivalDataset)
    assert kernel.object_get(dataset_id).data["sources"] == [sample_id]
    verified = kernel.apply(dataset_id, "verify")
    assert verified.ok and verified.status == "verified"
    assert verified.data["value"] == {
        "observations": 4, "events": 3, "censored": 1,
        "strata": 1, "delayed_entry": False}
    assert verified.data["details"]["independent_censoring"] == "asserted_not_verified"


def test_kaplan_meier_exact_product_limit_and_counts():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel)
    result = kernel.apply(dataset_id, "kaplan_meier")
    assert result.ok, result.errors
    output = kernel.math_objects[result.data["object_id"]]["value"]
    assert isinstance(output, KaplanMeierEstimate)
    assert output.timeline == (1, 2, 3)
    assert output.at_risk == (4, 3, 1)
    assert output.events == (1, 1, 1)
    assert output.censored == (0, 1, 0)
    assert output.survival == (sp.Rational(3, 4), sp.Rational(1, 2), 0)
    assert output.median_survival == 2
    assert result.data["details"]["point_estimate_arithmetic"] == "exact"
    assert result.trust == TrustLevel.NUMERIC_HIGH_PRECISION


def test_greenwood_log_log_intervals_are_ordered_and_explicit():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel)
    result = kernel.apply(dataset_id, "kaplan_meier", {
        "confidence_level": "9/10"})
    output = kernel.math_objects[result.data["object_id"]]["value"]
    assert output.confidence_method == "greenwood_log_log"
    assert output.confidence_level == sp.Float("0.9", 17)
    assert all(0 <= float(lo) <= float(s) <= float(hi) <= 1
               for lo, s, hi in zip(output.confidence_lower, output.survival,
                                    output.confidence_upper))
    evidence = result.claim_evidence["kaplan_meier"]
    assert evidence.numerical[0].role == "required"
    assert evidence.model[0].role == "diagnostic"


def test_delayed_entry_changes_constructed_risk_sets():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["time", "event", "entry"], [
        [2, 1, 0], [3, 1, 1], [4, 0, 3]])
    dataset = kernel.object_create("SurvivalData", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        "entry": "entry"})
    result = kernel.apply(dataset.data["object_id"], "kaplan_meier")
    output = kernel.math_objects[result.data["object_id"]]["value"]
    assert output.at_risk == (2, 2, 1)
    assert output.entered == (0, 1, 0)
    assert output.survival == (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4))


def test_stratified_kaplan_meier_requires_and_records_selection():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["time", "event", "group"], [
        [1, 1, 0], [3, 0, 0], [2, 1, 1], [4, 1, 1]])
    dataset = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        "strata": "group"})
    refused = kernel.apply(dataset.data["object_id"], "kaplan_meier")
    assert not refused.ok and "stratum is required" in refused.errors[0]
    selected = kernel.apply(dataset.data["object_id"], "kaplan_meier", {"stratum": 1})
    assert selected.ok
    output = kernel.math_objects[selected.data["object_id"]]["value"]
    assert output.stratum == 1 and output.at_risk == (2, 1)


def test_strata_limit_uses_numeric_equality_not_literal_spelling():
    kernel = MathKernel(Settings(max_survival_strata=1))
    sample_id = make_sample(kernel, ["time", "event", "group"], [
        [1, 1, 1], [2, 0, "1.0"]])
    dataset = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        "strata": "group"})
    assert dataset.ok, dataset.errors


def test_survival_at_uses_right_continuous_step_curve():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel)
    curve = kernel.apply(dataset_id, "kaplan_meier")
    curve_id = curve.data["object_id"]
    before = kernel.apply(curve_id, "survival_at", {"time": "1/2"})
    at = kernel.apply(curve_id, "survival_at", {"time": 2})
    assert before.data["value"]["survival"] == "1"
    assert at.data["value"]["survival"] == "1/2"


def test_kaplan_meier_replay_recomputes_risk_sets_curve_and_uncertainty():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel)
    curve = kernel.apply(dataset_id, "kaplan_meier")
    replay = kernel.apply(curve.data["object_id"], "verify")
    assert replay.ok and replay.status == "verified"
    assert all(replay.data["verification"].values())


def test_decimal_survival_ancestry_cannot_upgrade_to_exact():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel, [
        ["1.0", 1], [2, 0], [3, 1]])
    result = kernel.apply(dataset_id, "kaplan_meier")
    assert result.ok and result.trust == TrustLevel.NUMERIC
    assert result.data["details"]["point_estimate_arithmetic"] == "numeric"


@pytest.mark.parametrize("rows,message", [
    ([[1, 2], [2, 1]], "exact 0 or 1"),
    ([[-1, 1], [2, 0]], "0 <= entry"),
])
def test_invalid_survival_domains_fail_at_source_construction(rows, message):
    kernel = MathKernel(); sample_id = make_sample(kernel, ["time", "event"], rows)
    result = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event"})
    assert not result.ok and message in result.errors[0]


def test_invalid_delayed_entry_is_refused():
    kernel = MathKernel(); sample_id = make_sample(kernel, ["time", "event", "entry"], [
        [1, 1, 2], [3, 0, 0]])
    result = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        "entry": "entry"})
    assert not result.ok and "entry <= duration" in result.errors[0]


def test_all_censored_data_are_representable_but_do_not_support_cox_fit():
    kernel = MathKernel(); sample_id = make_sample(kernel, ["time", "event", "x"], [
        [1, 0, 0], [2, 0, 1], [3, 0, 2]])
    dataset = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event"})
    assert dataset.ok
    curve = kernel.apply(dataset.data["object_id"], "kaplan_meier")
    output = kernel.math_objects[curve.data["object_id"]]["value"]
    assert output.survival == (1, 1, 1) and output.median_survival is None
    model = kernel.object_create("CoxPHModel", {
        "dataset_id": dataset.data["object_id"], "predictors": ["x"]})
    checked = kernel.apply(model.data["object_id"], "verify")
    assert checked.status == "refuted"
    assert checked.data["verification"]["events_exceed_parameters"] is False


@pytest.mark.parametrize("parameters,message", [
    ({"confidence_level": 1}, "in (0,1)"),
    ({"confidence_level": "x"}, "concrete"),
])
def test_invalid_kaplan_meier_parameters_fail_without_output(parameters, message):
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel)
    before = len(kernel.math_objects)
    result = kernel.apply(dataset_id, "kaplan_meier", parameters)
    assert not result.ok and message in result.errors[0]
    assert len(kernel.math_objects) == before


def test_survival_resource_limits_preflight():
    kernel = MathKernel(Settings(max_survival_timeline_points=2))
    _, dataset_id = make_dataset(kernel)
    result = kernel.apply(dataset_id, "kaplan_meier")
    assert not result.ok and "max_survival_timeline_points" in result.errors[0]


def test_cox_model_verification_and_efron_fit():
    kernel = MathKernel(); _, _, model_id = make_cox(kernel)
    verification = kernel.apply(model_id, "verify")
    assert verification.ok and verification.status == "verified"
    fit = kernel.apply(model_id, "fit")
    assert fit.ok, fit.errors
    output = kernel.math_objects[fit.data["object_id"]]["value"]
    assert isinstance(output, CoxPHFit)
    assert output.tie_method == "efron" and output.converged
    assert output.coefficients[0] > 0 and output.hazard_ratios[0] > 1
    assert float(output.score_residual) < 1e-7
    assert all(b >= 0 for b in output.baseline_hazard)
    assert all(output.baseline_cumulative_hazard[index] >=
               output.baseline_cumulative_hazard[index - 1]
               for index in range(1, len(output.baseline_cumulative_hazard)))


def test_cox_breslow_and_efron_ties_are_explicit_and_distinct():
    kernel = MathKernel(); _, _, efron_id = make_cox(kernel, "efron")
    efron = kernel.apply(efron_id, "fit")
    dataset_id = kernel.math_objects[efron_id]["value"].dataset_id
    breslow_model = kernel.object_create("CoxPHModel", {
        "dataset_id": dataset_id, "predictors": ["x"], "tie_method": "breslow"})
    breslow = kernel.apply(breslow_model.data["object_id"], "fit")
    assert breslow.ok and efron.ok
    left = kernel.math_objects[efron.data["object_id"]]["value"]
    right = kernel.math_objects[breslow.data["object_id"]]["value"]
    assert left.tie_method == "efron" and right.tie_method == "breslow"
    assert left.coefficients != right.coefficients


@pytest.mark.parametrize("tie_method", ["breslow", "efron"])
def test_cox_score_and_information_match_finite_differences(tie_method):
    x = np.asarray([[0.0], [1.0], [2.0], [3.0], [1.5]])
    start = np.zeros(5)
    stop = np.asarray([1.0, 2.0, 2.0, 3.0, 4.0])
    status = np.asarray([1, 1, 1, 0, 1])
    beta = np.asarray([0.2])
    _, score, information = _cox_components(
        beta, x, start, stop, status, tie_method)
    step = 1e-5
    plus_ll, plus_score, _ = _cox_components(
        beta + step, x, start, stop, status, tie_method)
    minus_ll, minus_score, _ = _cox_components(
        beta - step, x, start, stop, status, tie_method)
    numerical_score = (plus_ll - minus_ll) / (2 * step)
    numerical_information = -(plus_score[0] - minus_score[0]) / (2 * step)
    assert score[0] == pytest.approx(numerical_score, rel=1e-8, abs=1e-8)
    assert information[0, 0] == pytest.approx(
        numerical_information, rel=1e-8, abs=1e-8)


def test_cox_concordance_excludes_not_yet_entered_pairs():
    beta = np.asarray([1.0])
    x = np.asarray([[1.0], [0.0], [2.0]])
    start = np.asarray([0.0, 0.0, 3.0])
    stop = np.asarray([2.0, 4.0, 5.0])
    status = np.asarray([1, 0, 0])
    assert _concordance(beta, x, start, stop, status) == 1.0


def test_cox_fit_replay_diagnostics_and_prediction():
    kernel = MathKernel(); _, _, model_id = make_cox(kernel)
    fit = kernel.apply(model_id, "fit"); fit_id = fit.data["object_id"]
    replay = kernel.apply(fit_id, "verify")
    diagnostics = kernel.apply(fit_id, "diagnostics")
    prediction = kernel.apply(fit_id, "predict_partial_hazard", {
        "rows": [[0], [2]]})
    assert replay.ok and replay.status == "verified"
    assert diagnostics.ok and diagnostics.data["details"]["proportional_hazards"] == "not_established_by_diagnostics"
    assert prediction.ok and prediction.data["value"]["partial_hazards"][0] == "1.0000000000000000"
    assert float(prediction.data["value"]["partial_hazards"][1]) > 1


def test_cox_evidence_does_not_claim_model_population_or_causality():
    kernel = MathKernel(); _, _, model_id = make_cox(kernel)
    fit = kernel.apply(model_id, "fit")
    assert fit.trust == TrustLevel.NUMERIC
    assert fit.data["details"]["proportional_hazards"] == "asserted_not_verified"
    assert fit.data["details"]["population_generalization"] == "not_established"
    assert fit.data["details"]["causal_effect"] == "not_established"
    assert all(item.role == "diagnostic" for item in fit.claim_evidence["fit"].model)


def test_rank_deficient_and_event_sparse_cox_models_are_refuted():
    kernel = MathKernel()
    rows = [[1, 1, 4], [2, 0, 4], [3, 1, 4], [4, 0, 4]]
    _, _, model_id = make_cox(kernel, rows=rows)
    result = kernel.apply(model_id, "verify")
    assert result.status == "refuted"
    assert result.data["verification"]["predictor_rank_full"] is False
    fit = kernel.apply(model_id, "fit")
    assert not fit.ok and "verification failed" in fit.errors[0]


def test_stratified_cox_is_explicitly_outside_initial_fit():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["time", "event", "x", "group"], [
        [1, 1, 0, 0], [2, 1, 1, 0], [3, 0, 0, 1], [4, 1, 1, 1]])
    dataset = kernel.object_create("SurvivalDataset", {
        "sample_id": sample_id, "duration": "time", "event": "event",
        "strata": "group"})
    model = kernel.object_create("CoxPHModel", {
        "dataset_id": dataset.data["object_id"], "predictors": ["x"]})
    result = kernel.apply(model.data["object_id"], "verify")
    assert not result.ok and "unstratified" in result.errors[0]


def test_cox_work_limit_and_nonconvergence_fail_without_fit_object():
    limited = MathKernel(Settings(max_survival_work=100))
    _, _, model_id = make_cox(limited)
    before = len(limited.math_objects)
    result = limited.apply(model_id, "fit")
    assert not result.ok and "max_survival_work" in result.errors[0]
    assert len(limited.math_objects) == before


def test_survival_derived_objects_are_output_only_and_persistent(tmp_path):
    settings = Settings(store_path=str(tmp_path / "survival.sqlite"))
    kernel = MathKernel(settings); _, _, model_id = make_cox(kernel)
    dataset_id = kernel.math_objects[model_id]["value"].dataset_id
    curve = kernel.apply(dataset_id, "kaplan_meier")
    fit = kernel.apply(model_id, "fit")
    assert not kernel.object_create("KaplanMeierEstimate", {}).ok
    assert not kernel.object_create("CoxPHFit", {}).ok
    restarted = MathKernel(settings)
    assert restarted.apply(curve.data["object_id"], "verify").status == "verified"
    assert restarted.apply(fit.data["object_id"], "verify").status == "verified"
    assert restarted.object_get(model_id).data["sources"] == [dataset_id]


def test_survival_capabilities_limits_and_version():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert entries[("SurvivalDataset", "kaplan_meier")]["output_types"] == [
        "EngineeringResult", "KaplanMeierEstimate"]
    assert entries[("CoxProportionalHazardsModel", "fit")]["output_types"] == [
        "EngineeringResult", "CoxPHFit"]
    assert entries[("CoxProportionalHazardsModel", "fit")]["engines"] == [
        "survival_analysis", "numpy"]
    assert entries[("CoxProportionalHazardsModel", "fit")]["trust_levels"] == [
        "numeric"]
    assert kernel.capabilities()["version"] == __import__("mathkernel").__version__
    for setting in ("max_survival_strata", "max_survival_timeline_points",
                    "max_cox_parameters", "max_cox_iterations",
                    "max_cox_prediction_rows", "max_cox_information_condition",
                    "max_survival_work"):
        assert setting in Settings().limits()
