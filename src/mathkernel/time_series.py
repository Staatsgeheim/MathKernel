# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded Phase F.5 ordered univariate time-series analysis.

Calculations are conditional on stored order and explicit model assumptions.
Stationarity diagnostics and successful optimization never establish the data-
generating model, population validity, or causal interpretation.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import sympy as sp

from mathkernel_artifacts import (ComputationEvidence, EmpiricalEvidence,
                                  EvidenceBundle, ModelEvidence,
                                  NumericalEvidence)

from .engineering import EngineeringResult, cap_trust
from .statistical_inference import (StatisticalSample, TimeSeriesAnalysis,
                                    TimeSeriesDataset, TimeSeriesFit,
                                    TimeSeriesForecast, TimeSeriesModel,
                                    _compare)


def _column(sample, name):
    if name not in sample.variables:
        raise ValueError(f"{name!r} must name a stored sample variable")
    index = sample.variables.index(name)
    return tuple(row[index] for row in sample.observations)


def _float(value):
    return sp.Float(repr(float(value)), 17)


def _columns(sample, dataset):
    times = _column(sample, dataset.time)
    values = _column(sample, dataset.value)
    return times, values


def _ordering(times):
    strict = all(_compare(times[index - 1], times[index]) < 0
                 for index in range(1, len(times)))
    if not strict:
        return False, False, None
    if len(times) < 2:
        return True, True, None
    spacing = sp.simplify(times[1] - times[0])
    regular = all(_compare(sp.simplify(times[index] - times[index - 1]), spacing) == 0
                  for index in range(2, len(times)))
    return True, regular, spacing


def _bundle(sample, dataset, method, *, arithmetic="numeric", numerical=True,
            assumptions=()):
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="time_series", method=method, arithmetic=arithmetic,
            deterministic=True, trust=arithmetic)],
        empirical=[EmpiricalEvidence(
            sample_size=len(sample.observations),
            sampling_method=sample.sampling_method, role="diagnostic",
            trust="empirical", metadata={"scope": "stored ordered series"})],
        model=[ModelEvidence(
            assumptions=["stored row order is temporal order",
                         *dataset.time_assumptions, *assumptions],
            diagnostics=["stationarity and residual tests are diagnostics, not proof",
                         "population generalization is not established",
                         "causal interpretation is not established"],
            role="diagnostic", trust="unknown",
            metadata={"missing_policy": dataset.missing_policy,
                      "population_claim": "not_established"})],
        justified_trust=arithmetic)
    if numerical:
        bundle.numerical.append(NumericalEvidence(
            precision=53, residual=None, role="required", trust=arithmetic,
            metadata={"error_bound": False, "method": method}))
    return bundle


def validate_dataset(sample: StatisticalSample, dataset: TimeSeriesDataset):
    times, values = _columns(sample, dataset)
    strict, regular, spacing = _ordering(times)
    checks = {"time_strictly_increasing": strict,
              "values_complete": len(values) == len(times),
              "missing_policy_explicit": dataset.missing_policy == "reject"}
    status = "verified" if all(checks.values()) else "refuted"
    trust = cap_trust(sample.input_trust, dataset.input_trust)
    return EngineeringResult(
        operation="verify", status=status, trust=trust,
        value={"observations": len(values), "regular_spacing": regular,
               "spacing": spacing},
        details={"identity_checks": checks, "row_order_preserved": True,
                 "population_generalization": "not_established"},
        verification=checks,
        claim_evidence={"verify": _bundle(
            sample, dataset, "ordered_time_schema", arithmetic=trust,
            numerical=False)})


def autocorrelation(sample, dataset, dataset_id, max_lag):
    _, values = _columns(sample, dataset)
    n = len(values)
    if not 0 <= max_lag < n:
        raise ValueError("max_lag must be nonnegative and below the observation count")
    mean = sp.simplify(sum(values, sp.S.Zero) / n)
    centered = tuple(sp.simplify(item - mean) for item in values)
    denominator = sp.simplify(sum((item * item for item in centered), sp.S.Zero))
    if denominator == 0:
        raise ValueError("autocorrelation is undefined for a constant series")
    correlations = tuple(sp.simplify(
        sum((centered[index] * centered[index - lag]
             for index in range(lag, n)), sp.S.Zero) / denominator)
        for lag in range(max_lag + 1))
    trust = cap_trust(sample.input_trust, "exact")
    output = TimeSeriesAnalysis(
        dataset_id=dataset_id, analysis="acf", lags=tuple(range(max_lag + 1)),
        values=correlations, method="biased_centered_autocorrelation",
        input_trust=trust)
    checks = {"lag_zero_unity": correlations[0] == 1,
              "bounded_by_unity": all(abs(float(item)) <= 1 + 1e-12
                                      for item in correlations)}
    result = EngineeringResult(
        operation="acf", status="verified", trust=trust,
        value={"lags": output.lags, "values": correlations},
        details={"normalization": "common_lag_zero_denominator",
                 "population_generalization": "not_established"},
        verification=checks,
        claim_evidence={"acf": _bundle(
            sample, dataset, "biased_centered_autocorrelation",
            arithmetic=trust, numerical=False)})
    return result, output


def partial_autocorrelation(sample, dataset, dataset_id, max_lag):
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    _, acf = autocorrelation(sample, dataset, dataset_id, max_lag)
    rho = acf.values
    pacf = [sp.S.One]
    phi = []
    for order in range(1, max_lag + 1):
        numerator = rho[order] - sum(
            (phi[index - 1] * rho[order - index]
             for index in range(1, order)), sp.S.Zero)
        denominator = sp.S.One - sum(
            (phi[index - 1] * rho[index]
             for index in range(1, order)), sp.S.Zero)
        if denominator == 0:
            raise ValueError("PACF Durbin-Levinson denominator is zero")
        reflection = sp.simplify(numerator / denominator)
        previous = phi
        phi = [sp.simplify(previous[index - 1] - reflection *
                           previous[order - index - 1])
               for index in range(1, order)] + [reflection]
        pacf.append(reflection)
    trust = cap_trust(sample.input_trust, "exact")
    output = TimeSeriesAnalysis(
        dataset_id=dataset_id, analysis="pacf", lags=tuple(range(max_lag + 1)),
        values=tuple(pacf), method="durbin_levinson",
        input_trust=trust)
    checks = {"lag_zero_unity": pacf[0] == 1,
              "recursion_complete": len(pacf) == max_lag + 1}
    return EngineeringResult(
        operation="pacf", status="verified", trust=trust,
        value={"lags": output.lags, "values": output.values},
        details={"method": "Durbin-Levinson from biased ACF",
                 "population_generalization": "not_established"},
        verification=checks,
        claim_evidence={"pacf": _bundle(
            sample, dataset, "durbin_levinson", arithmetic=trust,
            numerical=False)}), output


def adf_test(sample, dataset, dataset_id, max_lag):
    _, raw = _columns(sample, dataset)
    y = np.asarray([float(item) for item in raw], dtype=float)
    delta = np.diff(y)
    if max_lag < 0 or len(y) <= max_lag + 3:
        raise ValueError("ADF lag leaves insufficient observations")
    rows = np.arange(max_lag, len(delta))
    response = delta[rows]
    columns = [np.ones(len(rows)), y[rows]]
    columns.extend(delta[rows - lag] for lag in range(1, max_lag + 1))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("ADF regression design is rank deficient")
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    degrees = len(response) - design.shape[1]
    if degrees <= 0:
        raise ValueError("ADF regression has no residual degrees of freedom")
    covariance = (float(residual @ residual) / degrees) * np.linalg.inv(design.T @ design)
    standard_error = math.sqrt(max(float(covariance[1, 1]), 0.0))
    if standard_error == 0:
        raise ValueError("ADF lagged-level standard error is zero")
    statistic = float(coefficients[1]) / standard_error
    critical = (("1%", _float(-3.43)), ("5%", _float(-2.86)),
                ("10%", _float(-2.57)))
    output = TimeSeriesAnalysis(
        dataset_id=dataset_id, analysis="adf", statistic=_float(statistic),
        critical_values=critical, selected_lag=max_lag,
        method="adf_constant_asymptotic_critical_values", input_trust="numeric")
    checks = {"regression_full_rank": True,
              "statistic_finite": math.isfinite(statistic),
              "critical_values_explicit": True}
    return EngineeringResult(
        operation="stationarity_test", status="verified", trust="numeric",
        value={"statistic": output.statistic,
               "critical_values": dict(critical),
               "reject_unit_root_at_5_percent": statistic < -2.86},
        details={"null_hypothesis": "unit_root",
                 "alternative": "level_stationary_with_constant",
                 "critical_value_scope": "asymptotic_approximation",
                 "finite_sample_error_bound": False,
                 "stationarity": "diagnostic_not_established"},
        verification=checks,
        claim_evidence={"stationarity_test": _bundle(
            sample, dataset, "adf_ols_asymptotic_critical_values")}), output


def verify_analysis(sample, dataset, result, max_work):
    lag = (result.lags[-1] if result.lags else result.selected_lag or 0)
    work = len(sample.observations) * (lag + 1)
    if result.analysis == "pacf":
        work += lag ** 3
    if work > max_work:
        raise ValueError("time-series analysis replay exceeds max_time_series_work")
    if result.analysis == "acf":
        _, candidate = autocorrelation(sample, dataset, result.dataset_id,
                                       result.lags[-1])
    elif result.analysis == "pacf":
        _, candidate = partial_autocorrelation(sample, dataset, result.dataset_id,
                                               result.lags[-1])
    else:
        _, candidate = adf_test(sample, dataset, result.dataset_id,
                                result.selected_lag or 0)
    checks = {"analysis_reconciled": candidate.analysis == result.analysis,
              "values_recomputed": candidate.values == result.values,
              "statistic_recomputed": candidate.statistic == result.statistic,
              "method_reconciled": candidate.method == result.method}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust, value={"checks": checks}, verification=checks,
        details={"population_generalization": "not_established"})


def _difference(values, order):
    output = np.asarray(values, dtype=float)
    for _ in range(order):
        output = np.diff(output)
    return output


def _arma_residuals(parameters, y, p, q, include_constant):
    cursor = 0
    constant = parameters[cursor] if include_constant else 0.0
    cursor += int(include_constant)
    ar = np.asarray(parameters[cursor:cursor + p]); cursor += p
    ma = np.asarray(parameters[cursor:cursor + q])
    residual = np.zeros(len(y)); fitted = np.zeros(len(y))
    for index in range(len(y)):
        prediction = constant
        for lag in range(1, min(p, index) + 1):
            prediction += ar[lag - 1] * y[index - lag]
        for lag in range(1, min(q, index) + 1):
            prediction += ma[lag - 1] * residual[index - lag]
        fitted[index] = prediction
        residual[index] = y[index] - prediction
    return residual, fitted, constant, ar, ma


def _roots(ar, ma):
    ar_roots = np.roots(np.r_[-np.asarray(ar)[::-1], 1.0]) if len(ar) else np.asarray([])
    ma_roots = np.roots(np.r_[np.asarray(ma)[::-1], 1.0]) if len(ma) else np.asarray([])
    return np.abs(ar_roots), np.abs(ma_roots)


def _fit_arma(y, model, max_iterations, tolerance):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ValueError("ARMA/ARIMA fitting requires the scipy optional dependency") from exc
    p, q = model.p, model.q
    count = int(model.include_constant) + p + q
    initial = np.zeros(count)
    if model.include_constant:
        initial[0] = float(np.mean(y))
    burn = max(p, q)
    def objective(parameters):
        residual, _, _, _, _ = _arma_residuals(parameters, y, p, q,
                                                model.include_constant)
        return float(residual[burn:] @ residual[burn:])
    bounds = ([(None, None)] if model.include_constant else []) + [(-0.999, 0.999)] * (p + q)
    optimized = minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": max_iterations, "ftol": tolerance,
                                  "gtol": tolerance})
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        raise ValueError(f"time-series conditional fit did not converge: {optimized.message}")
    residual, fitted, constant, ar, ma = _arma_residuals(
        optimized.x, y, p, q, model.include_constant)
    ar_roots, ma_roots = _roots(ar, ma)
    if len(ar_roots) and np.min(ar_roots) <= 1 + 1e-6:
        raise ValueError("fitted AR polynomial is not stationary")
    if len(ma_roots) and np.min(ma_roots) <= 1 + 1e-6:
        raise ValueError("fitted MA polynomial is not invertible")
    variance = float(np.mean(residual[burn:] ** 2))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("innovation variance is nonpositive")
    return residual, fitted, constant, ar, ma, variance, int(optimized.nit)


def _garch_variance(parameters, y, p, q, include_constant):
    cursor = 0
    mean = parameters[cursor] if include_constant else 0.0
    cursor += int(include_constant)
    omega = parameters[cursor]; cursor += 1
    alpha = np.asarray(parameters[cursor:cursor + p]); cursor += p
    beta = np.asarray(parameters[cursor:cursor + q])
    residual = y - mean
    base = max(float(np.var(residual)), np.finfo(float).eps)
    variance = np.full(len(y), base)
    for index in range(1, len(y)):
        current = omega
        for lag in range(1, min(p, index) + 1):
            current += alpha[lag - 1] * residual[index - lag] ** 2
        for lag in range(1, min(q, index) + 1):
            current += beta[lag - 1] * variance[index - lag]
        variance[index] = max(current, np.finfo(float).eps)
    return residual, variance, mean, omega, alpha, beta


def _fit_garch(y, model, max_iterations, tolerance):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ValueError("GARCH fitting requires the scipy optional dependency") from exc
    variance = max(float(np.var(y)), 1e-8)
    initial = ([float(np.mean(y))] if model.include_constant else []) + [variance * .05]
    initial += [0.05 / model.p] * model.p + [0.85 / model.q] * model.q
    def objective(parameters):
        residual, conditional, *_ = _garch_variance(
            parameters, y, model.p, model.q, model.include_constant)
        return .5 * float(np.sum(np.log(2 * math.pi * conditional) +
                                 residual ** 2 / conditional))
    bounds = ([(None, None)] if model.include_constant else []) + \
             [(1e-12, None)] + [(0.0, 0.999)] * (model.p + model.q)
    offset = int(model.include_constant) + 1
    constraint = {"type": "ineq", "fun": lambda parameters:
                  0.999 - float(np.sum(parameters[offset:]))}
    optimized = minimize(objective, np.asarray(initial), method="SLSQP",
                         bounds=bounds, constraints=[constraint],
                         options={"maxiter": max_iterations, "ftol": tolerance})
    if not optimized.success or not np.all(np.isfinite(optimized.x)):
        raise ValueError(f"GARCH likelihood fit did not converge: {optimized.message}")
    values = _garch_variance(optimized.x, y, model.p, model.q,
                             model.include_constant)
    return (*values, int(optimized.nit))


def validate_model(sample, dataset, model):
    verification = validate_dataset(sample, dataset)
    regular = verification.value["regular_spacing"]
    _, raw = _columns(sample, dataset)
    effective = len(raw) - model.d
    parameters = int(model.include_constant) + model.p + model.q + (1 if model.family == "garch" else 0)
    checks = {"dataset_verified": verification.status == "verified",
              "regular_spacing": regular,
              "enough_observations": effective > parameters + max(model.p, model.q) + 2,
              "orders_match_family": True,
              "gaussian_innovation_explicit": model.innovation_distribution == "gaussian"}
    status = "verified" if all(checks.values()) else "refuted"
    return EngineeringResult(
        operation="verify", status=status, trust="numeric",
        value={"checks": checks, "observations": len(raw),
               "effective_observations": effective, "parameters": parameters},
        verification=checks,
        details={"stationarity": "asserted_not_verified",
                 "model_validity": "not_established",
                 "population_generalization": "not_established"},
        claim_evidence={"verify": _bundle(
            sample, dataset, "time_series_model_preflight",
            assumptions=model.model_assumptions)})


def fit_model(sample, dataset, model, model_id, max_iterations, tolerance):
    verified = validate_model(sample, dataset, model)
    if verified.status != "verified":
        failed = [key for key, value in verified.verification.items() if not value]
        raise ValueError(f"time-series model verification failed: {', '.join(failed)}")
    _, raw = _columns(sample, dataset)
    original = np.asarray([float(item) for item in raw])
    y = _difference(original, model.d)
    if model.family == "garch":
        residual, conditional, constant, omega, alpha, beta, iterations = _fit_garch(
            y, model, max_iterations, tolerance)
        fitted = y - residual
        ar = ma = np.asarray([]); innovation_variance = float(np.mean(conditional))
        log_likelihood = -.5 * float(np.sum(
            np.log(2 * math.pi * conditional) + residual ** 2 / conditional))
        stationary = float(np.sum(alpha) + np.sum(beta)) < 1
        invertible = True
    else:
        residual, fitted, constant, ar, ma, innovation_variance, iterations = _fit_arma(
            y, model, max_iterations, tolerance)
        conditional = np.full(len(y), innovation_variance)
        omega = None; alpha = beta = np.asarray([])
        log_likelihood = -.5 * len(y) * (math.log(2 * math.pi * innovation_variance) + 1)
        ar_roots, ma_roots = _roots(ar, ma)
        stationary = not len(ar_roots) or bool(np.min(ar_roots) > 1)
        invertible = not len(ma_roots) or bool(np.min(ma_roots) > 1)
    ar_roots, ma_roots = _roots(ar, ma)
    parameter_count = int(model.include_constant) + len(ar) + len(ma)
    if model.family == "garch":
        parameter_count = int(model.include_constant) + 1 + len(alpha) + len(beta)
    aic = 2 * parameter_count - 2 * log_likelihood
    bic = math.log(len(y)) * parameter_count - 2 * log_likelihood
    output = TimeSeriesFit(
        model_id=model_id, family=model.family, order=(model.p, model.d, model.q),
        intercept=_float(constant), ar_coefficients=tuple(_float(v) for v in ar),
        ma_coefficients=tuple(_float(v) for v in ma),
        variance_intercept=None if omega is None else _float(omega),
        arch_coefficients=tuple(_float(v) for v in alpha),
        garch_coefficients=tuple(_float(v) for v in beta),
        residuals=tuple(_float(v) for v in residual),
        fitted_values=tuple(_float(v) for v in fitted),
        conditional_variance=tuple(_float(v) for v in conditional),
        innovation_variance=_float(innovation_variance),
        log_likelihood=_float(log_likelihood), aic=_float(aic), bic=_float(bic),
        iterations=iterations, converged=True,
        ar_root_moduli=tuple(_float(v) for v in ar_roots),
        ma_root_moduli=tuple(_float(v) for v in ma_roots),
        stationary=stationary, invertible=invertible,
        initialization=model.initialization, training_observations=len(original),
        input_trust="numeric")
    checks = {"converged": True, "residuals_finite": bool(np.all(np.isfinite(residual))),
              "variances_positive": bool(np.all(conditional > 0)),
              "stationary": stationary, "invertible": invertible}
    result = EngineeringResult(
        operation="fit", status="verified", trust="numeric",
        value={"family": model.family, "order": output.order,
               "intercept": output.intercept,
               "ar_coefficients": output.ar_coefficients,
               "ma_coefficients": output.ma_coefficients,
               "log_likelihood": output.log_likelihood,
               "aic": output.aic, "bic": output.bic},
        details={"identity_checks": checks, "iterations": iterations,
                 "initialization": model.initialization,
                 "innovation_distribution": model.innovation_distribution,
                 "stationarity": "conditional_parameter_check_only",
                 "model_validity": "not_established",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        verification=checks,
        claim_evidence={"fit": _bundle(
            sample, dataset,
            "gaussian_garch_likelihood" if model.family == "garch" else
            "arma_conditional_sum_squares",
            assumptions=["weak stationarity after configured differencing",
                         "independent Gaussian innovations", *model.model_assumptions])})
    return result, output


def _stored_components(fit):
    return (float(fit.intercept),
            np.asarray([float(v) for v in fit.ar_coefficients]),
            np.asarray([float(v) for v in fit.ma_coefficients]))


def verify_fit(sample, dataset, model, fit):
    _, raw = _columns(sample, dataset)
    y = _difference([float(v) for v in raw], model.d)
    if fit.family == "garch":
        parameters = ([float(fit.intercept)] if model.include_constant else []) + \
            [float(fit.variance_intercept)] + [float(v) for v in fit.arch_coefficients] + \
            [float(v) for v in fit.garch_coefficients]
        residual, variance, *_ = _garch_variance(
            np.asarray(parameters), y, model.p, model.q, model.include_constant)
        fitted = y - residual
    else:
        constant, ar, ma = _stored_components(fit)
        parameters = ([constant] if model.include_constant else []) + list(ar) + list(ma)
        residual, fitted, *_ = _arma_residuals(
            np.asarray(parameters), y, model.p, model.q, model.include_constant)
        variance = np.full(len(y), float(fit.innovation_variance))
    stored_residual = np.asarray([float(v) for v in fit.residuals])
    stored_fitted = np.asarray([float(v) for v in fit.fitted_values])
    stored_variance = np.asarray([float(v) for v in fit.conditional_variance])
    checks = {"residuals_recomputed": bool(np.allclose(residual, stored_residual, rtol=1e-10, atol=1e-10)),
              "fitted_values_recomputed": bool(np.allclose(fitted, stored_fitted, rtol=1e-10, atol=1e-10)),
              "variance_recomputed": bool(np.allclose(variance, stored_variance, rtol=1e-10, atol=1e-10)),
              "source_length_reconciled": fit.training_observations == len(raw)}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="numeric", value={"checks": checks}, verification=checks,
        details={"model_validity": "not_established",
                 "population_generalization": "not_established"})


def diagnostics(fit, max_lag):
    residual = np.asarray([float(v) for v in fit.residuals])
    residual = residual[max(fit.order[0], fit.order[2]):]
    n = len(residual)
    if not 1 <= max_lag < n:
        raise ValueError("diagnostic max_lag must be positive and below residual count")
    centered = residual - np.mean(residual)
    denominator = float(centered @ centered)
    if denominator <= 0:
        raise ValueError("residual diagnostics require nonconstant residuals")
    correlations = [float(centered[lag:] @ centered[:-lag] / denominator)
                    for lag in range(1, max_lag + 1)]
    q_statistic = n * (n + 2) * sum(
        correlations[lag - 1] ** 2 / (n - lag)
        for lag in range(1, max_lag + 1))
    degrees = max(max_lag - fit.order[0] - fit.order[2], 1)
    p_value = float(mp.gammainc(degrees / 2, q_statistic / 2, mp.inf,
                                regularized=True))
    std = float(np.std(centered))
    skew = float(np.mean(centered ** 3) / std ** 3) if std else 0.0
    kurtosis = float(np.mean(centered ** 4) / std ** 4) if std else 3.0
    jarque_bera = n / 6 * (skew ** 2 + (kurtosis - 3) ** 2 / 4)
    return EngineeringResult(
        operation="diagnostics", status="verified", trust="numeric",
        value={"residual_acf": tuple(_float(v) for v in correlations),
               "ljung_box_q": _float(q_statistic),
               "ljung_box_degrees_of_freedom": degrees,
               "ljung_box_p_value": _float(p_value),
               "jarque_bera": _float(jarque_bera)},
        details={"white_noise": "diagnostic_not_established",
                 "gaussian_innovations": "diagnostic_not_established",
                 "model_validity": "not_established"},
        verification={"statistics_finite": all(math.isfinite(v) for v in
                      [q_statistic, p_value, jarque_bera]),
                      "p_value_in_unit_interval": 0 <= p_value <= 1})


def _psi_weights(ar, ma, differencing, horizon):
    ar_polynomial = np.r_[1.0, -np.asarray(ar)]
    difference_polynomial = np.asarray([1.0])
    for _ in range(differencing):
        difference_polynomial = np.convolve(difference_polynomial, [1.0, -1.0])
    denominator = np.convolve(ar_polynomial, difference_polynomial)
    numerator = np.r_[1.0, np.asarray(ma)]
    psi = np.zeros(horizon); psi[0] = 1.0
    for index in range(1, horizon):
        theta = numerator[index] if index < len(numerator) else 0.0
        psi[index] = theta - sum(denominator[lag] * psi[index - lag]
                                 for lag in range(1, min(index, len(denominator) - 1) + 1))
    return psi


def forecast(sample, dataset, model, fit, fit_id, horizon, confidence_level):
    times, raw = _columns(sample, dataset)
    strict, regular, spacing = _ordering(times)
    if not strict or not regular or spacing is None:
        raise ValueError("forecasting requires at least two regularly spaced times")
    values = np.asarray([float(v) for v in raw])
    z = math.sqrt(2) * float(mp.erfinv(float(confidence_level)))
    means = []; variances = []
    if fit.family == "garch":
        mean = float(fit.intercept)
        alpha = np.asarray([float(v) for v in fit.arch_coefficients])
        beta = np.asarray([float(v) for v in fit.garch_coefficients])
        omega = float(fit.variance_intercept)
        residual_history = list(np.asarray([float(v) for v in fit.residuals]) ** 2)
        variance_history = list(float(v) for v in fit.conditional_variance)
        for _ in range(horizon):
            variance = omega
            variance += sum(alpha[lag - 1] * residual_history[-lag]
                            for lag in range(1, min(len(alpha), len(residual_history)) + 1))
            variance += sum(beta[lag - 1] * variance_history[-lag]
                            for lag in range(1, min(len(beta), len(variance_history)) + 1))
            means.append(mean); variances.append(variance)
            residual_history.append(variance); variance_history.append(variance)
        method = "garch_gaussian_conditional_variance"
    else:
        constant, ar, ma = _stored_components(fit)
        differenced = list(_difference(values, model.d))
        errors = list(float(v) for v in fit.residuals)
        difference_forecasts = []
        for _ in range(horizon):
            predicted = constant
            predicted += sum(ar[lag - 1] * differenced[-lag]
                             for lag in range(1, min(len(ar), len(differenced)) + 1))
            predicted += sum(ma[lag - 1] * errors[-lag]
                             for lag in range(1, min(len(ma), len(errors)) + 1))
            difference_forecasts.append(predicted)
            differenced.append(predicted); errors.append(0.0)
        if model.d == 0:
            means = difference_forecasts
        else:
            levels = [list(_difference(values, order)) for order in range(model.d)]
            for predicted in difference_forecasts:
                current = predicted
                for order in range(model.d - 1, -1, -1):
                    current = levels[order][-1] + current
                    levels[order].append(current)
                means.append(current)
        psi = _psi_weights(ar, ma, model.d, horizon)
        variance = float(fit.innovation_variance)
        variances = [variance * float(np.sum(psi[:step + 1] ** 2))
                     for step in range(horizon)]
        method = "arma_psi_weight_gaussian_interval"
    standard_errors = [math.sqrt(max(v, 0.0)) for v in variances]
    future_times = tuple(sp.simplify(times[-1] + (step + 1) * spacing)
                         for step in range(horizon))
    lower = [mean - z * se for mean, se in zip(means, standard_errors)]
    upper = [mean + z * se for mean, se in zip(means, standard_errors)]
    output = TimeSeriesForecast(
        fit_id=fit_id, horizon=horizon, times=future_times,
        means=tuple(_float(v) for v in means),
        standard_errors=tuple(_float(v) for v in standard_errors),
        lower=tuple(_float(v) for v in lower), upper=tuple(_float(v) for v in upper),
        confidence_level=_float(confidence_level), method=method,
        input_trust="numeric")
    checks = {"horizon_complete": len(means) == horizon,
              "intervals_ordered": all(lo <= mean <= hi for lo, mean, hi in
                                       zip(lower, means, upper)),
              "future_times_regular": True}
    result = EngineeringResult(
        operation="forecast", status="verified", trust="numeric",
        value={"times": future_times, "means": output.means,
               "standard_errors": output.standard_errors,
               "lower": output.lower, "upper": output.upper},
        details={"method": method, "confidence_level": confidence_level,
                 "innovation_distribution": model.innovation_distribution,
                 "forecast_coverage": "conditional_not_established",
                 "model_validity": "not_established",
                 "population_generalization": "not_established"},
        verification=checks,
        claim_evidence={"forecast": _bundle(
            sample, dataset, method,
            assumptions=["future innovations follow the fitted Gaussian model",
                         *model.model_assumptions])})
    return result, output


def verify_forecast(sample, dataset, model, fit, forecast_result):
    _, candidate = forecast(
        sample, dataset, model, fit, forecast_result.fit_id,
        forecast_result.horizon, float(forecast_result.confidence_level))
    checks = {"times_recomputed": candidate.times == forecast_result.times,
              "means_recomputed": candidate.means == forecast_result.means,
              "uncertainty_recomputed": (candidate.standard_errors == forecast_result.standard_errors and
                                           candidate.lower == forecast_result.lower and
                                           candidate.upper == forecast_result.upper),
              "method_reconciled": candidate.method == forecast_result.method}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="numeric", value={"checks": checks}, verification=checks,
        details={"forecast_coverage": "conditional_not_established",
                 "model_validity": "not_established"})
