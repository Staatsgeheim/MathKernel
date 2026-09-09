import asyncio
import math
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from mathkernel import (MathKernel, Settings, TimeSeriesAnalysis,
                        TimeSeriesDataset, TimeSeriesFit,
                        TimeSeriesForecast, TimeSeriesModel, TrustLevel)


def make_sample(kernel, values, times=None):
    times = list(range(len(values))) if times is None else times
    made = kernel.object_create("StatisticalSample", {
        "variables": ["time", "value"],
        "observations": [[time, value] for time, value in zip(times, values)]})
    assert made.ok, made.errors
    return made.data["object_id"]


def make_dataset(kernel, values, times=None):
    sample_id = make_sample(kernel, values, times)
    made = kernel.object_create("TimeSeriesDataset", {
        "sample_id": sample_id, "time": "time", "value": "value"})
    assert made.ok, made.errors
    return sample_id, made.data["object_id"]


def deterministic_series(count=100):
    values = []
    current = 0.0
    for index in range(count):
        current = 1 + 0.55 * current + 0.2 * math.sin(index * 1.7)
        values.append(str(current))
    return values


def make_model(kernel, family="ar", p=1, d=0, q=0, values=None):
    sample_id, dataset_id = make_dataset(kernel, values or deterministic_series())
    made = kernel.object_create("TimeSeriesModel", {
        "dataset_id": dataset_id, "family": family, "p": p, "d": d, "q": q})
    assert made.ok, made.errors
    return sample_id, dataset_id, made.data["object_id"]


def test_time_series_dataset_preserves_strict_order_and_spacing():
    kernel = MathKernel(); sample_id, dataset_id = make_dataset(kernel, [1, 2, 3, 4])
    stored = kernel.math_objects[dataset_id]["value"]
    assert isinstance(stored, TimeSeriesDataset)
    assert kernel.object_get(dataset_id).data["sources"] == [sample_id]
    checked = kernel.apply(dataset_id, "verify")
    assert checked.status == "verified"
    assert checked.data["value"] == {
        "observations": 4, "regular_spacing": True, "spacing": "1"}


@pytest.mark.parametrize("times", [[0, 2, 1], [0, 1, 1]])
def test_unsorted_or_duplicate_times_are_refused_at_construction(times):
    kernel = MathKernel(); sample_id = make_sample(kernel, [1, 2, 3], times)
    made = kernel.object_create("TimeSeriesDataset", {
        "sample_id": sample_id, "time": "time", "value": "value"})
    assert not made.ok and "strictly increasing" in made.errors[0]


def test_irregular_series_is_analyzable_but_model_is_refuted():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel, [1, 2, 4, 8], [0, 1, 3, 6])
    assert kernel.apply(dataset_id, "acf", {"max_lag": 2}).ok
    model = kernel.object_create("TimeSeriesModel", {
        "dataset_id": dataset_id, "family": "ar", "p": 1})
    checked = kernel.apply(model.data["object_id"], "verify")
    assert checked.status == "refuted"
    assert checked.data["verification"]["regular_spacing"] is False


def test_exact_acf_and_pacf_have_known_values_and_replay():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel, [1, 2, 3, 4])
    acf = kernel.apply(dataset_id, "acf", {"max_lag": 3})
    pacf = kernel.apply(dataset_id, "pacf", {"max_lag": 2})
    assert acf.ok and pacf.ok and acf.trust == TrustLevel.EXACT
    acf_output = kernel.math_objects[acf.data["object_id"]]["value"]
    pacf_output = kernel.math_objects[pacf.data["object_id"]]["value"]
    assert isinstance(acf_output, TimeSeriesAnalysis)
    assert acf_output.values == (1, sp.Rational(1, 4), sp.Rational(-3, 10), sp.Rational(-9, 20))
    assert pacf_output.values == (1, sp.Rational(1, 4), sp.Rational(-29, 75))
    assert kernel.apply(acf.data["object_id"], "verify").status == "verified"
    assert kernel.apply(pacf.data["object_id"], "verify").status == "verified"


def test_decimal_acf_ancestry_stays_numeric_and_constant_series_refuses():
    kernel = MathKernel(); _, numeric_id = make_dataset(kernel, ["1.0", 2, 3, 4])
    assert kernel.apply(numeric_id, "acf", {"max_lag": 2}).trust == TrustLevel.NUMERIC
    _, constant_id = make_dataset(kernel, [2, 2, 2, 2])
    refused = kernel.apply(constant_id, "acf", {"max_lag": 1})
    assert not refused.ok and "constant" in refused.errors[0]


def test_adf_stationarity_diagnostic_is_explicitly_asymptotic():
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel, deterministic_series(80))
    result = kernel.apply(dataset_id, "stationarity_test", {"method": "adf", "max_lag": 1})
    assert result.ok and result.trust == TrustLevel.NUMERIC
    output = kernel.math_objects[result.data["object_id"]]["value"]
    assert output.analysis == "adf" and output.selected_lag == 1
    assert result.data["details"]["critical_value_scope"] == "asymptotic_approximation"
    assert result.data["details"]["stationarity"] == "diagnostic_not_established"
    assert kernel.apply(result.data["object_id"], "verify").status == "verified"


@pytest.mark.parametrize("family,p,d,q", [
    ("ar", 1, 0, 0), ("ma", 0, 0, 1), ("arma", 1, 0, 1),
    ("arima", 1, 1, 0), ("arima", 0, 1, 0),
])
def test_ar_ma_arma_arima_models_fit_verify_and_forecast(family, p, d, q):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    kernel = MathKernel(); _, _, model_id = make_model(kernel, family, p, d, q)
    assert kernel.apply(model_id, "verify").status == "verified"
    fitted = kernel.apply(model_id, "fit", {"max_iterations": 300, "tolerance": 1e-9})
    assert fitted.ok, fitted.errors
    fit = kernel.math_objects[fitted.data["object_id"]]["value"]
    assert isinstance(fit, TimeSeriesFit) and fit.converged
    assert fit.family == family and fit.order == (p, d, q)
    assert kernel.apply(fitted.data["object_id"], "verify").status == "verified"
    diagnostics = kernel.apply(fitted.data["object_id"], "diagnostics", {"max_lag": 5})
    assert diagnostics.ok and 0 <= float(diagnostics.data["value"]["ljung_box_p_value"]) <= 1
    forecast = kernel.apply(fitted.data["object_id"], "forecast", {"horizon": 4})
    assert forecast.ok, forecast.errors
    output = kernel.math_objects[forecast.data["object_id"]]["value"]
    assert isinstance(output, TimeSeriesForecast) and output.times == (100, 101, 102, 103)
    assert kernel.apply(forecast.data["object_id"], "verify").status == "verified"


def test_garch_fit_enforces_positive_stationary_variance_and_forecasts():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    rng = np.random.default_rng(20260904)
    values = []; variance = 1.0; residual = 0.0
    for _ in range(240):
        variance = .08 + .12 * residual ** 2 + .8 * variance
        residual = float(rng.normal()) * math.sqrt(variance)
        values.append(str(residual))
    kernel = MathKernel(); _, _, model_id = make_model(
        kernel, "garch", 1, 0, 1, values)
    fitted = kernel.apply(model_id, "fit", {"max_iterations": 400})
    assert fitted.ok, fitted.errors
    fit = kernel.math_objects[fitted.data["object_id"]]["value"]
    assert fit.variance_intercept > 0
    assert all(value >= 0 for value in fit.arch_coefficients + fit.garch_coefficients)
    assert sum(fit.arch_coefficients + fit.garch_coefficients) < 1
    assert all(value > 0 for value in fit.conditional_variance)
    assert kernel.apply(fitted.data["object_id"], "verify").status == "verified"
    forecast = kernel.apply(fitted.data["object_id"], "forecast", {"horizon": 5})
    assert forecast.ok and all(float(value) > 0 for value in
                               kernel.math_objects[forecast.data["object_id"]]["value"].standard_errors)


@pytest.mark.parametrize("definition", [
    {"family": "ar", "p": 0}, {"family": "ma", "p": 1, "q": 1},
    {"family": "arma", "p": 1, "q": 0}, {"family": "garch", "p": 0, "q": 1},
])
def test_model_family_order_contract_is_strict(definition):
    kernel = MathKernel(); _, dataset_id = make_dataset(kernel, deterministic_series())
    made = kernel.object_create("TimeSeriesModel", {"dataset_id": dataset_id, **definition})
    assert not made.ok and "orders" in made.errors[0]


def test_model_and_forecast_evidence_make_no_validity_or_coverage_claim():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    kernel = MathKernel(); _, _, model_id = make_model(kernel)
    fitted = kernel.apply(model_id, "fit")
    assert fitted.data["details"]["model_validity"] == "not_established"
    assert fitted.data["details"]["population_generalization"] == "not_established"
    assert fitted.data["details"]["causal_effect"] == "not_established"
    forecast = kernel.apply(fitted.data["object_id"], "forecast", {"horizon": 2})
    assert forecast.data["details"]["forecast_coverage"] == "conditional_not_established"
    assert all(item.role == "diagnostic" for item in fitted.claim_evidence["fit"].model)


def test_time_series_limits_preflight_without_output_objects():
    kernel = MathKernel(Settings(max_time_series_lag=1))
    _, dataset_id = make_dataset(kernel, list(range(10)))
    before = len(kernel.math_objects)
    result = kernel.apply(dataset_id, "acf", {"max_lag": 2})
    assert not result.ok and "max_time_series_lag" in result.errors[0]
    assert len(kernel.math_objects) == before
    limited = MathKernel(Settings(max_time_series_parameters=1))
    _, limited_id = make_dataset(limited, deterministic_series())
    model = limited.object_create("TimeSeriesModel", {
        "dataset_id": limited_id, "family": "arma", "p": 1, "q": 1})
    assert not model.ok and "max_time_series_parameters" in model.errors[0]


def test_invalid_fit_and_forecast_parameters_create_no_outputs():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    kernel = MathKernel(); _, _, model_id = make_model(kernel)
    before = len(kernel.math_objects)
    refused = kernel.apply(model_id, "fit", {"max_iterations": 0})
    assert not refused.ok and len(kernel.math_objects) == before
    fitted = kernel.apply(model_id, "fit")
    before = len(kernel.math_objects)
    bad = kernel.apply(fitted.data["object_id"], "forecast", {"horizon": 0})
    assert not bad.ok and len(kernel.math_objects) == before


def test_derived_time_series_objects_are_output_only_and_persistent(tmp_path):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    settings = Settings(store_path=str(tmp_path / "time-series.sqlite"))
    kernel = MathKernel(settings); _, _, model_id = make_model(kernel)
    dataset_id = kernel.math_objects[model_id]["value"].dataset_id
    analysis = kernel.apply(dataset_id, "acf", {"max_lag": 3})
    fitted = kernel.apply(model_id, "fit")
    forecast = kernel.apply(fitted.data["object_id"], "forecast", {"horizon": 3})
    for name in ("TimeSeriesAnalysis", "TimeSeriesFit", "TimeSeriesForecast"):
        assert not kernel.object_create(name, {}).ok
    restarted = MathKernel(settings)
    assert restarted.apply(analysis.data["object_id"], "verify").status == "verified"
    assert restarted.apply(fitted.data["object_id"], "verify").status == "verified"
    assert restarted.apply(forecast.data["object_id"], "verify").status == "verified"


def test_time_series_capabilities_version_and_limits():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert entries[("TimeSeriesDataset", "acf")]["output_types"] == [
        "EngineeringResult", "TimeSeriesAnalysis"]
    assert entries[("TimeSeriesModel", "fit")]["output_types"] == [
        "EngineeringResult", "TimeSeriesFit"]
    assert entries[("TimeSeriesFit", "forecast")]["output_types"] == [
        "EngineeringResult", "TimeSeriesForecast"]
    assert kernel.capabilities()["version"] == __import__("mathkernel").__version__
    for name in ("max_time_series_lag", "max_time_series_difference",
                 "max_time_series_parameters", "max_time_series_iterations",
                 "max_time_series_forecast_steps", "max_time_series_work"):
        assert name in Settings().limits()
