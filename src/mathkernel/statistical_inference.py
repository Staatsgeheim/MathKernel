# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed Phase F statistical samples with claim-scoped evidence.

Exact arithmetic can establish a statistic of the stored observations.  It
cannot establish that the sample is representative, iid, unbiased, or drawn
from a proposed population.  Those scopes remain explicit in every result.
"""
from __future__ import annotations

from functools import cmp_to_key
import math
from typing import Literal

import numpy as np
import sympy as sp
from pydantic import Field, model_validator

from mathkernel_artifacts import (ComputationEvidence, EmpiricalEvidence,
                                  EvidenceBundle, ModelEvidence,
                                  NumericalEvidence)

from .engineering import EngineeringModel, EngineeringResult, cap_trust


SamplingMethod = Literal[
    "unspecified", "simple_random", "stratified", "cluster", "systematic",
    "convenience", "census", "experiment", "fixed_design",
]
GLMFamily = Literal["gaussian", "binomial", "poisson"]
GLMLink = Literal["identity", "logit", "log"]
NonparametricTestName = Literal[
    "mann_whitney", "wilcoxon", "kruskal_wallis", "ks_2samp",
    "spearman", "kendall",
]


def _check_scalar(value: sp.Expr) -> None:
    if not isinstance(value, sp.Expr) or value.free_symbols:
        raise ValueError("statistical observations must be concrete MathIR scalars")
    if value.has(sp.nan, sp.oo, -sp.oo, sp.zoo) or value.is_finite is not True:
        raise ValueError("statistical observations must be finite")
    if value.is_real is not True:
        raise ValueError("statistical observations must be real")


class StatisticalSample(EngineeringModel):
    variables: tuple[str, ...]
    observations: tuple[tuple[sp.Expr, ...], ...]
    observation_ids: tuple[str, ...] = ()
    sampling_method: SamplingMethod = "unspecified"
    population: str | None = None
    design_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_sample(self):
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("variables must be nonempty and unique")
        if any(not name or not isinstance(name, str) for name in self.variables):
            raise ValueError("variable names must be nonempty strings")
        if not self.observations:
            raise ValueError("statistical sample must contain observations")
        if any(len(row) != len(self.variables) for row in self.observations):
            raise ValueError("every observation must match the variable count")
        for row in self.observations:
            for value in row:
                _check_scalar(value)
        if self.observation_ids and (
                len(self.observation_ids) != len(self.observations)
                or len(set(self.observation_ids)) != len(self.observation_ids)
                or any(not item for item in self.observation_ids)):
            raise ValueError("observation_ids must be unique and match all rows")
        if self.population is not None and not self.population.strip():
            raise ValueError("population must be nonempty when supplied")
        if any(not item.strip() for item in self.design_assumptions):
            raise ValueError("design assumptions must be nonempty strings")
        if self.input_trust not in {"exact", "numeric", "numeric_high_precision"}:
            raise ValueError("statistical sample arithmetic trust must be exact or numeric")
        return self


class VariableSummary(EngineeringModel):
    variable: str
    count: int
    mean: sp.Expr
    variance_population: sp.Expr
    variance_sample: sp.Expr | None = None
    minimum: sp.Expr
    q1: sp.Expr
    median: sp.Expr
    q3: sp.Expr
    maximum: sp.Expr


class DescriptiveSummary(EngineeringModel):
    summaries: tuple[VariableSummary, ...]
    inference_scope: Literal["sample_only"] = "sample_only"
    input_trust: str = "exact"


class CovarianceMatrix(EngineeringModel):
    variables: tuple[str, ...]
    normalization: Literal["population", "sample"]
    matrix: tuple[tuple[sp.Expr, ...], ...]
    inference_scope: Literal["sample_only"] = "sample_only"
    input_trust: str = "exact"


class EmpiricalDistribution(EngineeringModel):
    variable: str
    support: tuple[sp.Expr, ...]
    counts: tuple[int, ...]
    probabilities: tuple[sp.Rational, ...]
    sample_size: int
    inference_scope: Literal["sample_empirical_distribution"] = (
        "sample_empirical_distribution")
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_distribution(self):
        if not self.support or not (
                len(self.support) == len(self.counts) == len(self.probabilities)):
            raise ValueError("empirical support, counts, and probabilities must align")
        if sum(self.counts) != self.sample_size or sum(self.probabilities) != 1:
            raise ValueError("empirical distribution must normalize to the sample size")
        return self


class GeneralizedLinearModel(EngineeringModel):
    """A canonical-link GLM specification over one immutable sample."""

    sample_id: str
    response: str
    predictors: tuple[str, ...]
    family: GLMFamily
    link: GLMLink
    include_intercept: bool = True
    model_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_model(self):
        canonical = {"gaussian": "identity", "binomial": "logit",
                     "poisson": "log"}
        if not self.sample_id:
            raise ValueError("sample_id must be nonempty")
        if not self.response or len(set(self.predictors)) != len(self.predictors):
            raise ValueError("response must be nonempty and predictors unique")
        if self.response in self.predictors:
            raise ValueError("response cannot also be a predictor")
        if self.link != canonical[self.family]:
            raise ValueError(
                f"Phase F.2 supports only {canonical[self.family]} link for {self.family}")
        if any(not item.strip() for item in self.model_assumptions):
            raise ValueError("model assumptions must be nonempty strings")
        return self


class GLMFit(EngineeringModel):
    """Derived coefficient and convergence evidence for a fitted GLM."""

    model_id: str
    family: GLMFamily
    link: GLMLink
    response: str
    predictors: tuple[str, ...]
    coefficient_names: tuple[str, ...]
    coefficients: tuple[sp.Expr, ...]
    covariance: tuple[tuple[sp.Expr, ...], ...]
    standard_errors: tuple[sp.Expr, ...]
    linear_predictors: tuple[sp.Expr, ...]
    fitted_means: tuple[sp.Expr, ...]
    deviance: sp.Expr
    null_deviance: sp.Expr
    log_likelihood: sp.Expr | None = None
    dispersion: sp.Expr
    iterations: int
    converged: bool
    design_rank: int
    score_residual: sp.Expr
    hessian_condition: sp.Expr | None = None
    inference_scope: Literal["conditional_model_fit"] = "conditional_model_fit"
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_fit(self):
        size = len(self.coefficient_names)
        if not size or not (size == len(self.coefficients) ==
                            len(self.standard_errors) == len(self.covariance)):
            raise ValueError("GLM coefficient and covariance dimensions must align")
        if any(len(row) != size for row in self.covariance):
            raise ValueError("GLM covariance must be square")
        if len(self.linear_predictors) != len(self.fitted_means):
            raise ValueError("GLM fitted vectors must align")
        return self


class NonparametricTestResult(EngineeringModel):
    """Derived exact-enumeration or explicitly asymptotic test result."""

    test: NonparametricTestName
    statistic: sp.Expr
    p_value: sp.Expr
    alternative: Literal["two_sided", "less", "greater"] = "two_sided"
    method: Literal[
        "exact_permutation", "asymptotic_normal", "asymptotic_chi_square",
        "asymptotic_kolmogorov", "asymptotic_student_t",
    ]
    variables: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    enumeration_count: int | None = None
    tie_correction: sp.Expr | None = None
    configuration: dict = Field(default_factory=dict)
    inference_scope: Literal["conditional_test_result"] = "conditional_test_result"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_test_result(self):
        _check_scalar(self.statistic); _check_scalar(self.p_value)
        if self.p_value.is_nonnegative is not True or (1 - self.p_value).is_nonnegative is not True:
            raise ValueError("test p-value must be in [0, 1]")
        if not self.variables or not self.sample_sizes or any(n < 1 for n in self.sample_sizes):
            raise ValueError("test variables and sample sizes must be nonempty")
        if self.method == "exact_permutation" and not self.enumeration_count:
            raise ValueError("exact permutation results require an enumeration count")
        return self


class ResamplingResult(EngineeringModel):
    """Derived exact or seeded Monte Carlo/percentile resampling result."""

    procedure: Literal["permutation_test", "bootstrap"]
    statistic_name: Literal["difference_in_means", "difference_in_medians",
                            "mean", "median"]
    observed_statistic: sp.Expr
    method: Literal["exact_enumeration", "monte_carlo", "percentile_bootstrap"]
    variables: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    p_value: sp.Expr | None = None
    interval: tuple[sp.Expr, sp.Expr] | None = None
    standard_error: sp.Expr | None = None
    confidence_level: sp.Expr | None = None
    resamples: int
    seed: int | None = None
    rng_algorithm: str | None = None
    extreme_count: int | None = None
    configuration: dict = Field(default_factory=dict)
    inference_scope: Literal["conditional_resampling_result"] = (
        "conditional_resampling_result")
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_resampling_result(self):
        _check_scalar(self.observed_statistic)
        if self.procedure == "permutation_test":
            if self.p_value is None or self.interval is not None:
                raise ValueError("permutation result requires only a p-value")
        elif self.interval is None or self.confidence_level is None:
            raise ValueError("bootstrap result requires interval and confidence level")
        if self.method != "exact_enumeration" and (
                self.seed is None or self.rng_algorithm != "PCG64"):
            raise ValueError("simulated resampling requires PCG64 seed provenance")
        return self


class SurvivalDataset(EngineeringModel):
    """Right-censored survival-data specification over an immutable sample."""

    sample_id: str
    duration: str
    event: str
    entry: str | None = None
    strata: str | None = None
    censoring: Literal["right"] = "right"
    survival_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_survival_dataset(self):
        if not self.sample_id:
            raise ValueError("sample_id must be nonempty")
        columns = [self.duration, self.event]
        if self.entry is not None:
            columns.append(self.entry)
        if self.strata is not None:
            columns.append(self.strata)
        if any(not item for item in columns) or len(set(columns)) != len(columns):
            raise ValueError("survival columns must be nonempty and distinct")
        if any(not item.strip() for item in self.survival_assumptions):
            raise ValueError("survival assumptions must be nonempty strings")
        return self


class KaplanMeierEstimate(EngineeringModel):
    """Derived product-limit curve with Greenwood/log-log uncertainty."""

    dataset_id: str
    stratum: sp.Expr | None = None
    timeline: tuple[sp.Expr, ...]
    at_risk: tuple[int, ...]
    events: tuple[int, ...]
    censored: tuple[int, ...]
    entered: tuple[int, ...]
    survival: tuple[sp.Expr, ...]
    standard_errors: tuple[sp.Expr, ...]
    confidence_lower: tuple[sp.Expr, ...]
    confidence_upper: tuple[sp.Expr, ...]
    confidence_level: sp.Expr
    confidence_method: Literal["greenwood_log_log"] = "greenwood_log_log"
    median_survival: sp.Expr | None = None
    inference_scope: Literal["conditional_product_limit_estimate"] = (
        "conditional_product_limit_estimate")
    input_trust: str = "numeric_high_precision"

    @model_validator(mode="after")
    def validate_kaplan_meier(self):
        lengths = {len(self.timeline), len(self.at_risk), len(self.events),
                   len(self.censored), len(self.entered), len(self.survival),
                   len(self.standard_errors), len(self.confidence_lower),
                   len(self.confidence_upper)}
        if lengths != {len(self.timeline)} or not self.timeline:
            raise ValueError("Kaplan-Meier curve fields must align and be nonempty")
        if any(n < 0 for values in (self.at_risk, self.events, self.censored,
                                    self.entered) for n in values):
            raise ValueError("survival counts must be nonnegative")
        for s, se, lower, upper in zip(
                self.survival, self.standard_errors,
                self.confidence_lower, self.confidence_upper):
            for item in (s, se, lower, upper):
                _check_scalar(item)
            if not (0 <= float(lower) <= float(s) <= float(upper) <= 1):
                raise ValueError("Kaplan-Meier estimates and intervals must be ordered in [0,1]")
        return self


class CoxProportionalHazardsModel(EngineeringModel):
    """Conditional Cox model specification with an explicit tie method."""

    dataset_id: str
    predictors: tuple[str, ...]
    tie_method: Literal["breslow", "efron"] = "efron"
    model_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_cox_model(self):
        if not self.dataset_id or not self.predictors:
            raise ValueError("dataset_id and at least one predictor are required")
        if len(set(self.predictors)) != len(self.predictors):
            raise ValueError("Cox predictors must be unique")
        if any(not item or not isinstance(item, str) for item in self.predictors):
            raise ValueError("Cox predictors must be nonempty strings")
        if any(not item.strip() for item in self.model_assumptions):
            raise ValueError("model assumptions must be nonempty strings")
        return self


class CoxPHFit(EngineeringModel):
    """Derived checked numerical Cox partial-likelihood fit."""

    model_id: str
    predictors: tuple[str, ...]
    tie_method: Literal["breslow", "efron"]
    coefficients: tuple[sp.Expr, ...]
    hazard_ratios: tuple[sp.Expr, ...]
    covariance: tuple[tuple[sp.Expr, ...], ...]
    standard_errors: tuple[sp.Expr, ...]
    log_partial_likelihood: sp.Expr
    null_log_partial_likelihood: sp.Expr
    score_residual: sp.Expr
    iterations: int
    converged: bool
    information_condition: sp.Expr
    event_count: int
    baseline_timeline: tuple[sp.Expr, ...]
    baseline_hazard: tuple[sp.Expr, ...]
    baseline_cumulative_hazard: tuple[sp.Expr, ...]
    concordance_index: sp.Expr | None = None
    schoenfeld_time_correlations: tuple[sp.Expr, ...] = ()
    inference_scope: Literal["conditional_cox_partial_likelihood_fit"] = (
        "conditional_cox_partial_likelihood_fit")
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_cox_fit(self):
        p = len(self.predictors)
        if not p or not (len(self.coefficients) == len(self.hazard_ratios) ==
                         len(self.standard_errors) == len(self.covariance) == p):
            raise ValueError("Cox coefficient fields must align")
        if any(len(row) != p for row in self.covariance):
            raise ValueError("Cox covariance must be square")
        if not (len(self.baseline_timeline) == len(self.baseline_hazard) ==
                len(self.baseline_cumulative_hazard)):
            raise ValueError("Cox baseline hazard fields must align")
        if self.schoenfeld_time_correlations and len(
                self.schoenfeld_time_correlations) != p:
            raise ValueError("Schoenfeld diagnostics must align with predictors")
        return self


class TimeSeriesDataset(EngineeringModel):
    """Ordered univariate observations over an immutable sample."""

    sample_id: str
    time: str
    value: str
    missing_policy: Literal["reject"] = "reject"
    time_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_time_series_dataset(self):
        if not self.sample_id or not self.time or not self.value:
            raise ValueError("sample_id, time, and value are required")
        if self.time == self.value:
            raise ValueError("time and value columns must be distinct")
        if any(not item.strip() for item in self.time_assumptions):
            raise ValueError("time assumptions must be nonempty strings")
        return self


class TimeSeriesAnalysis(EngineeringModel):
    """Derived ACF, PACF, or ADF diagnostic with replay provenance."""

    dataset_id: str
    analysis: Literal["acf", "pacf", "adf"]
    lags: tuple[int, ...] = ()
    values: tuple[sp.Expr, ...] = ()
    statistic: sp.Expr | None = None
    critical_values: tuple[tuple[str, sp.Expr], ...] = ()
    selected_lag: int | None = None
    method: str
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_time_series_analysis(self):
        if not self.dataset_id or not self.method:
            raise ValueError("analysis source and method are required")
        if self.analysis in {"acf", "pacf"}:
            if not self.values or len(self.lags) != len(self.values):
                raise ValueError("correlation lags and values must align")
        elif self.statistic is None or not self.critical_values:
            raise ValueError("ADF analysis requires statistic and critical values")
        return self


class TimeSeriesModel(EngineeringModel):
    """Explicit univariate AR/MA/ARMA/ARIMA/GARCH specification."""

    dataset_id: str
    family: Literal["ar", "ma", "arma", "arima", "garch"]
    p: int = 0
    d: int = 0
    q: int = 0
    include_constant: bool = True
    innovation_distribution: Literal["gaussian"] = "gaussian"
    initialization: Literal["conditional_sum_squares"] = "conditional_sum_squares"
    model_assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_time_series_model(self):
        if not self.dataset_id or min(self.p, self.d, self.q) < 0:
            raise ValueError("dataset_id and nonnegative orders are required")
        valid = ((self.family == "ar" and self.p > 0 and self.d == self.q == 0) or
                 (self.family == "ma" and self.q > 0 and self.p == self.d == 0) or
                 (self.family == "arma" and self.p > 0 and self.q > 0 and self.d == 0) or
                 (self.family == "arima" and self.d > 0 and self.p + self.q >= 0) or
                 (self.family == "garch" and self.p > 0 and self.q > 0 and self.d == 0))
        if not valid:
            raise ValueError("orders do not match the selected time-series family")
        if any(not item.strip() for item in self.model_assumptions):
            raise ValueError("model assumptions must be nonempty strings")
        return self


class TimeSeriesFit(EngineeringModel):
    """Derived checked conditional time-series fit."""

    model_id: str
    family: Literal["ar", "ma", "arma", "arima", "garch"]
    order: tuple[int, int, int]
    intercept: sp.Expr
    ar_coefficients: tuple[sp.Expr, ...] = ()
    ma_coefficients: tuple[sp.Expr, ...] = ()
    variance_intercept: sp.Expr | None = None
    arch_coefficients: tuple[sp.Expr, ...] = ()
    garch_coefficients: tuple[sp.Expr, ...] = ()
    residuals: tuple[sp.Expr, ...]
    fitted_values: tuple[sp.Expr, ...]
    conditional_variance: tuple[sp.Expr, ...]
    innovation_variance: sp.Expr
    log_likelihood: sp.Expr
    aic: sp.Expr
    bic: sp.Expr
    iterations: int
    converged: bool
    ar_root_moduli: tuple[sp.Expr, ...] = ()
    ma_root_moduli: tuple[sp.Expr, ...] = ()
    stationary: bool
    invertible: bool
    initialization: str
    training_observations: int
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_time_series_fit(self):
        if not self.model_id or self.training_observations <= 0:
            raise ValueError("fit source and training count are required")
        n = len(self.residuals)
        if not n or len(self.fitted_values) != n or len(self.conditional_variance) != n:
            raise ValueError("fit series fields must align")
        if not self.converged:
            raise ValueError("stored time-series fits must be converged")
        return self


class TimeSeriesForecast(EngineeringModel):
    """Derived analytic conditional-mean forecast and Gaussian interval."""

    fit_id: str
    horizon: int
    times: tuple[sp.Expr, ...]
    means: tuple[sp.Expr, ...]
    standard_errors: tuple[sp.Expr, ...]
    lower: tuple[sp.Expr, ...]
    upper: tuple[sp.Expr, ...]
    confidence_level: sp.Expr
    method: str
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_time_series_forecast(self):
        lengths = {len(self.times), len(self.means), len(self.standard_errors),
                   len(self.lower), len(self.upper)}
        if self.horizon <= 0 or lengths != {self.horizon}:
            raise ValueError("forecast fields must match the positive horizon")
        return self


def _compare(left: sp.Expr, right: sp.Expr) -> int:
    difference = sp.simplify(left - right)
    if difference == 0:
        return 0
    if difference.is_negative is True:
        return -1
    if difference.is_positive is True:
        return 1
    if left.has(sp.Float) or right.has(sp.Float):
        left_float, right_float = float(left), float(right)
        return (left_float > right_float) - (left_float < right_float)
    raise ValueError("observation ordering is not decidable exactly")


def _sorted(values: tuple[sp.Expr, ...]) -> tuple[sp.Expr, ...]:
    return tuple(sorted(values, key=cmp_to_key(_compare)))


def _quantile(values: tuple[sp.Expr, ...], numerator: int, denominator: int) -> sp.Expr:
    position = sp.Rational(numerator, denominator) * (len(values) - 1)
    lower = int(sp.floor(position)); upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return sp.simplify(values[lower] * (1 - fraction) + values[upper] * fraction)


def _columns(sample: StatisticalSample) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(tuple(row[index] for row in sample.observations)
                 for index in range(len(sample.variables)))


def _evidence(sample: StatisticalSample, operation: str, method: str,
              trust: str) -> EvidenceBundle:
    diagnostics = ["population generalization is not established"]
    if sample.sampling_method == "unspecified":
        diagnostics.append("sampling method is unspecified")
    return EvidenceBundle(
        computation=[ComputationEvidence(
            engine="statistical_inference", method=method,
            arithmetic=trust, deterministic=True, trust=trust)],
        empirical=[EmpiricalEvidence(
            sample_size=len(sample.observations),
            sampling_method=sample.sampling_method,
            trust="empirical", role="diagnostic",
            metadata={"scope": "stored observations only"})],
        model=[ModelEvidence(
            assumptions=list(sample.design_assumptions),
            diagnostics=diagnostics, trust="unknown", role="diagnostic",
            metadata={"population": sample.population,
                      "population_claim": "not_established"})],
        justified_trust=trust,
    )


def _result(sample: StatisticalSample, operation: str, value, *, method: str,
            checks: dict[str, bool], details: dict | None = None):
    trust = cap_trust(sample.input_trust, "exact")
    bundle = _evidence(sample, operation, method, trust)
    return EngineeringResult(
        operation=operation, status="verified", trust=trust, value=value,
        details={
            "identity_checks": checks,
            "claim_scope": "stored_sample_only",
            "population_generalization": "not_established",
            "sampling_method_asserted_not_verified": sample.sampling_method,
            **(details or {}),
        },
        verification=checks,
        claim_evidence={operation: bundle},
    )


def describe(sample: StatisticalSample):
    summaries = []
    for name, column in zip(sample.variables, _columns(sample)):
        ordered = _sorted(column); count = len(column)
        total = sum(column, sp.S.Zero)
        mean = sp.simplify(total / count)
        # The exact path uses the one-pass sufficient-statistic identity.  The
        # numeric path retains centered summation to avoid avoidable cancellation.
        if sample.input_trust == "exact":
            variance = sp.simplify(
                (sum((value * value for value in column), sp.S.Zero)
                 - total * total / count) / count)
        else:
            variance = sp.simplify(
                sum((value - mean) ** 2 for value in column) / count)
        summaries.append(VariableSummary(
            variable=name, count=count, mean=mean,
            variance_population=variance,
            variance_sample=(sp.simplify(variance * count / (count - 1))
                             if count > 1 else None),
            minimum=ordered[0], q1=_quantile(ordered, 1, 4),
            median=_quantile(ordered, 1, 2), q3=_quantile(ordered, 3, 4),
            maximum=ordered[-1]))
    output = DescriptiveSummary(summaries=tuple(summaries),
                                input_trust=sample.input_trust)
    checks = {"count_reconciled": all(item.count == len(sample.observations)
                                       for item in summaries),
              "ordered_five_number_summary": all(
                  item.minimum <= item.q1 <= item.median <= item.q3 <= item.maximum
                  for item in summaries)}
    return _result(sample, "describe", {"variables": sample.variables},
                   method="exact_sample_moments_and_order_statistics",
                   checks=checks), output


def covariance(sample: StatisticalSample, normalization: str = "sample"):
    if normalization not in {"sample", "population"}:
        raise ValueError("normalization must be sample or population")
    count = len(sample.observations)
    if normalization == "sample" and count < 2:
        raise ValueError("sample covariance requires at least two observations")
    columns = _columns(sample)
    totals = tuple(sum(column, sp.S.Zero) for column in columns)
    means = tuple(sp.simplify(total / count) for total in totals)
    denominator = count - 1 if normalization == "sample" else count
    size = len(columns)
    mutable = [[sp.S.Zero] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            if sample.input_trust == "exact":
                numerator = (sum((left * right for left, right in
                                  zip(columns[i], columns[j])), sp.S.Zero)
                             - totals[i] * totals[j] / count)
            else:
                numerator = sum(((left - means[i]) * (right - means[j])
                                 for left, right in zip(columns[i], columns[j])),
                                sp.S.Zero)
            mutable[i][j] = mutable[j][i] = sp.simplify(numerator / denominator)
    matrix = tuple(tuple(row) for row in mutable)
    output = CovarianceMatrix(variables=sample.variables,
        normalization=normalization, matrix=matrix, input_trust=sample.input_trust)
    checks = {"symmetric": all(matrix[i][j] == matrix[j][i]
                                for i in range(len(matrix)) for j in range(len(matrix))),
              "diagonal_nonnegative": all(value.is_nonnegative is True
                                            for value in (matrix[i][i]
                                                          for i in range(len(matrix))))}
    return _result(sample, "covariance", {"matrix": matrix},
                   method="centered_cross_product", checks=checks,
                   details={"normalization": normalization}), output


def empirical_distribution(sample: StatisticalSample, variable: str):
    if variable not in sample.variables:
        raise ValueError("variable must name a stored sample variable")
    column = _sorted(_columns(sample)[sample.variables.index(variable)])
    support: list[sp.Expr] = []
    counts: list[int] = []
    for value in column:
        if support and _compare(support[-1], value) == 0:
            counts[-1] += 1
        else:
            support.append(value); counts.append(1)
    count = len(column)
    probabilities = tuple(sp.Rational(item, count) for item in counts)
    output = EmpiricalDistribution(variable=variable, support=tuple(support),
        counts=tuple(counts), probabilities=probabilities, sample_size=count,
        input_trust=sample.input_trust)
    checks = {"counts_reconcile": sum(counts) == count,
              "probabilities_normalize": sum(probabilities) == 1,
              "support_strictly_ordered": all(_compare(left, right) < 0
                                               for left, right in zip(
                                                   support, support[1:]))}
    return _result(sample, "empirical_distribution",
                   {"support_size": len(support)},
                   method="exact_frequency_table", checks=checks), output


def evidence_profile(sample: StatisticalSample):
    checks = {"observation_schema_valid": True,
              "arithmetic_scope_separated": True,
              "population_claim_not_inferred": True}
    value = {
        "sample_size": len(sample.observations),
        "variable_count": len(sample.variables),
        "arithmetic_trust": sample.input_trust,
        "sampling_method": sample.sampling_method,
        "population": sample.population,
        "design_assumptions": sample.design_assumptions,
        "sample_statistic_scope": "exact_or_numeric_given_stored_observations",
        "population_generalization": "not_established",
        "model_validity": "not_established",
    }
    return _result(sample, "evidence_profile", value,
                   method="statistical_evidence_scope_audit", checks=checks)


def _glm_columns(sample: StatisticalSample, model: GeneralizedLinearModel):
    names = (("intercept",) if model.include_intercept else ()) + model.predictors
    variable_index = {name: index for index, name in enumerate(sample.variables)}
    if model.response not in variable_index:
        raise ValueError("response must name a stored sample variable")
    if any(name not in variable_index for name in model.predictors):
        raise ValueError("every predictor must name a stored sample variable")
    rows = tuple(
        ((sp.S.One,) if model.include_intercept else ()) +
        tuple(row[variable_index[name]] for name in model.predictors)
        for row in sample.observations)
    response = tuple(row[variable_index[model.response]]
                     for row in sample.observations)
    if not names:
        raise ValueError("a GLM needs an intercept or at least one predictor")
    return names, rows, response


def _response_domain(sample: StatisticalSample, model: GeneralizedLinearModel,
                     response: tuple[sp.Expr, ...]) -> bool:
    if model.family == "gaussian":
        return True
    numeric = tuple(float(value) for value in response)
    if model.family == "binomial":
        return all(value in {0.0, 1.0} for value in numeric) and len(set(numeric)) == 2
    integral = all(abs(value - round(value)) <= 1e-12 and value >= 0
                   for value in numeric)
    return integral and any(value > 0 for value in numeric)


def _exact_crossproducts(rows, response=None):
    width = len(rows[0])
    information = sp.zeros(width, width)
    rhs = sp.zeros(width, 1) if response is not None else None
    for row_index, row in enumerate(rows):
        for i in range(width):
            if rhs is not None:
                rhs[i] += row[i] * response[row_index]
            for j in range(i, width):
                information[i, j] += row[i] * row[j]
    for i in range(width):
        for j in range(i):
            information[i, j] = information[j, i]
    return information, rhs


def validate_glm_specification(sample: StatisticalSample,
                               model: GeneralizedLinearModel):
    names, rows, response = _glm_columns(sample, model)
    exact_design = not any(value.has(sp.Float) for row in rows for value in row)
    rank = (_exact_crossproducts(rows)[0].rank() if exact_design else
            int(np.linalg.matrix_rank(np.asarray(rows, dtype=float))))
    checks = {
        "variables_resolved": True,
        "response_domain": _response_domain(sample, model, response),
        "design_full_column_rank": rank == len(names),
        "positive_residual_degrees_of_freedom": len(rows) > len(names),
        "canonical_family_link": True,
    }
    trust = cap_trust(sample.input_trust, "exact")
    bundle = _glm_evidence(sample, model, "glm_specification_audit", trust)
    status = "verified" if all(checks.values()) else "refuted"
    return EngineeringResult(
        operation="verify", status=status, trust=trust,
        value={"family": model.family, "link": model.link,
               "coefficient_names": names, "observations": len(rows),
               "design_rank": rank},
        details={"identity_checks": checks,
                 "population_generalization": "not_established",
                 "model_validity": "not_established",
                 "causal_effect": "not_established"},
        verification=checks, claim_evidence={"verify": bundle})


def _glm_assumptions(model: GeneralizedLinearModel) -> list[str]:
    standard = {
        "gaussian": ["conditional mean is linear", "constant conditional variance"],
        "binomial": ["Bernoulli conditional response", "log-odds are linear"],
        "poisson": ["conditional variance equals conditional mean",
                    "log conditional mean is linear"],
    }
    return [*standard[model.family], *model.model_assumptions]


def _glm_evidence(sample: StatisticalSample, model: GeneralizedLinearModel,
                  method: str, trust: str, *, numerical: dict | None = None,
                  goodness_of_fit=None) -> EvidenceBundle:
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="statistical_inference", method=method,
            arithmetic=trust, deterministic=True, trust=trust)],
        empirical=[EmpiricalEvidence(
            sample_size=len(sample.observations),
            sampling_method=sample.sampling_method, trust="empirical",
            role="diagnostic", metadata={"scope": "stored observations only"})],
        model=[ModelEvidence(
            assumptions=_glm_assumptions(model),
            diagnostics=["model validity is not established",
                         "population generalization is not established",
                         "causal interpretation is not established"],
            goodness_of_fit=goodness_of_fit, trust="unknown", role="diagnostic",
            metadata={"family": model.family, "link": model.link,
                      "population_claim": "not_established",
                      "causal_claim": "not_established"})],
        justified_trust=trust)
    if numerical is not None:
        bundle.numerical.append(NumericalEvidence(
            precision=53, residual=numerical.get("score_residual"),
            convergence={"converged": numerical.get("converged"),
                         "iterations": numerical.get("iterations")},
            conditioning=numerical.get("hessian_condition"), trust="numeric",
            role="required", metadata={"error_bound": False,
                                        "arithmetic": "IEEE-754 float64"}))
    return bundle


def _sp_float(value: float) -> sp.Float:
    if not math.isfinite(float(value)):
        raise ValueError("GLM numerical fitting produced a non-finite value")
    return sp.Float(repr(float(value)), 15)


def _exact_gaussian_fit(sample: StatisticalSample,
                        model: GeneralizedLinearModel, model_id: str,
                        names, rows, response, information=None, rhs=None):
    if information is None or rhs is None:
        information, rhs = _exact_crossproducts(rows, response)
    coefficients = information.inv() * rhs
    linear = tuple(sum((row[index] * coefficients[index]
                        for index in range(len(names))), sp.S.Zero)
                   for row in rows)
    residual = tuple(target - fitted
                     for target, fitted in zip(response, linear))
    deviance = sp.simplify(sum((item * item for item in residual), sp.S.Zero))
    mean = sp.simplify(sum(response, sp.S.Zero) / len(response))
    null_deviance = sp.simplify(sum((item - mean) ** 2 for item in response))
    degrees = len(response) - len(names)
    dispersion = sp.simplify(deviance / degrees)
    covariance = sp.simplify(dispersion * information.inv())
    covariance_rows = tuple(tuple(sp.simplify(covariance[i, j])
                                  for j in range(covariance.cols))
                            for i in range(covariance.rows))
    standard_errors = tuple(sp.sqrt(covariance[i, i])
                            for i in range(covariance.rows))
    score = information * coefficients - rhs
    score_residual = max((abs(item) for item in score), default=sp.S.Zero)
    fit = GLMFit(
        model_id=model_id, family=model.family, link=model.link,
        response=model.response, predictors=model.predictors,
        coefficient_names=names, coefficients=tuple(coefficients),
        covariance=covariance_rows, standard_errors=standard_errors,
        linear_predictors=linear, fitted_means=linear,
        deviance=deviance, null_deviance=null_deviance,
        dispersion=dispersion, iterations=1, converged=True,
        design_rank=len(names), score_residual=score_residual,
        input_trust=sample.input_trust)
    checks = {"normal_equations": all(sp.simplify(item) == 0 for item in score),
              "covariance_symmetric": covariance == covariance.T,
              "deviance_nonnegative": deviance.is_nonnegative is True,
              "design_full_column_rank": True}
    bundle = _glm_evidence(sample, model, "exact_normal_equations", "exact",
                           goodness_of_fit={"deviance": sp.sstr(deviance),
                                            "null_deviance": sp.sstr(null_deviance)})
    result = EngineeringResult(
        operation="fit", status="verified", trust="exact",
        value={"family": model.family, "link": model.link,
               "coefficients": dict(zip(names, tuple(coefficients))),
               "deviance": deviance, "null_deviance": null_deviance,
               "iterations": 1, "converged": True},
        details={"identity_checks": checks, "dispersion": dispersion,
                 "model_validity": "not_established",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        verification=checks, claim_evidence={"fit": bundle})
    return result, fit


def _logistic_mean(linear: np.ndarray) -> np.ndarray:
    output = np.empty_like(linear)
    positive = linear >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-linear[positive]))
    exponential = np.exp(linear[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _log_likelihood(family: str, response: np.ndarray,
                    mean: np.ndarray) -> float:
    if family == "binomial":
        bounded = np.clip(mean, 1e-15, 1 - 1e-15)
        return float(np.sum(response * np.log(bounded) +
                            (1 - response) * np.log1p(-bounded)))
    return float(np.sum(response * np.log(np.clip(mean, 1e-300, None)) -
                        mean - np.vectorize(math.lgamma)(response + 1)))


def _deviance(family: str, response: np.ndarray, mean: np.ndarray) -> float:
    if family == "binomial":
        return -2.0 * _log_likelihood(family, response, mean)
    positive = response > 0
    terms = np.array(mean, copy=True)
    terms[positive] = (response[positive] *
                       np.log(response[positive] / mean[positive]) -
                       (response[positive] - mean[positive]))
    return float(2 * np.sum(terms))


def _numeric_glm_fit(sample: StatisticalSample,
                     model: GeneralizedLinearModel, model_id: str,
                     names, rows, response, *, max_iterations: int,
                     tolerance: float):
    design = np.asarray(rows, dtype=float); target = np.asarray(response, dtype=float)
    parameters = design.shape[1]
    coefficients = np.zeros(parameters, dtype=float)
    average = float(np.mean(target))
    if model.include_intercept:
        if model.family == "binomial":
            coefficients[0] = math.log(average / (1 - average))
        else:
            coefficients[0] = math.log(average)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        linear = design @ coefficients
        if model.family == "binomial":
            mean = _logistic_mean(linear)
            weights = np.clip(mean * (1 - mean), 1e-12, None)
        else:
            mean = np.exp(np.clip(linear, -30, 30))
            weights = np.clip(mean, 1e-12, None)
        working = linear + (target - mean) / weights
        information = design.T @ (weights[:, None] * design)
        rhs = design.T @ (weights * working)
        try:
            updated = np.linalg.solve(information, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("GLM weighted design became singular") from exc
        if not np.all(np.isfinite(updated)):
            raise ValueError("GLM iteration produced non-finite coefficients")
        delta = float(np.linalg.norm(updated - coefficients, ord=np.inf))
        coefficients = updated
        if delta <= tolerance * (1 + float(np.linalg.norm(coefficients, ord=np.inf))):
            converged = True
            break
        if model.family == "binomial":
            trial_mean = _logistic_mean(design @ coefficients)
            classified = np.all((trial_mean >= .5) == (target == 1))
            if classified and (float(np.linalg.norm(coefficients, ord=np.inf)) > 25 or
                               float(np.min(trial_mean * (1 - trial_mean))) < 1e-11):
                raise ValueError("complete or quasi separation suspected; finite logit MLE not established")
    if not converged:
        raise ValueError("GLM IRLS did not converge within max_iterations")
    linear = design @ coefficients
    mean = (_logistic_mean(linear) if model.family == "binomial" else
            np.exp(np.clip(linear, -30, 30)))
    if model.family == "binomial":
        classified = np.all((mean >= .5) == (target == 1))
        if classified and (float(np.linalg.norm(coefficients, ord=np.inf)) > 25 or
                           float(np.min(mean * (1 - mean))) < 1e-11):
            raise ValueError("complete or quasi separation suspected; finite logit MLE not established")
    elif np.any(np.abs(linear) > 30):
        raise ValueError("Poisson fit left the verified exponential range")
    weights = (mean * (1 - mean) if model.family == "binomial" else mean)
    information = design.T @ (weights[:, None] * design)
    condition = float(np.linalg.cond(information))
    if not math.isfinite(condition) or condition > 1e14:
        raise ValueError("GLM observed information is singular or ill-conditioned")
    score = design.T @ (target - mean)
    score_residual = float(np.linalg.norm(score, ord=np.inf))
    covariance = np.linalg.inv(information)
    covariance = (covariance + covariance.T) / 2
    deviance = _deviance(model.family, target, mean)
    null_mean = np.full_like(target, average)
    null_deviance = _deviance(model.family, target, null_mean)
    log_likelihood = _log_likelihood(model.family, target, mean)
    fit = GLMFit(
        model_id=model_id, family=model.family, link=model.link,
        response=model.response, predictors=model.predictors,
        coefficient_names=names,
        coefficients=tuple(_sp_float(item) for item in coefficients),
        covariance=tuple(tuple(_sp_float(item) for item in row) for row in covariance),
        standard_errors=tuple(_sp_float(math.sqrt(item)) for item in np.diag(covariance)),
        linear_predictors=tuple(_sp_float(item) for item in linear),
        fitted_means=tuple(_sp_float(item) for item in mean),
        deviance=_sp_float(deviance), null_deviance=_sp_float(null_deviance),
        log_likelihood=_sp_float(log_likelihood), dispersion=sp.S.One,
        iterations=iteration, converged=True, design_rank=parameters,
        score_residual=_sp_float(score_residual),
        hessian_condition=_sp_float(condition), input_trust="numeric")
    score_limit = max(1e-7, tolerance * 100 * max(1.0, len(target)))
    checks = {"irls_converged": True,
              "score_residual_within_tolerance": score_residual <= score_limit,
              "observed_information_invertible": True,
              "design_full_column_rank": True}
    if not all(checks.values()):
        raise ValueError("GLM converged step failed the independent score residual check")
    numerical = {"score_residual": score_residual, "converged": True,
                 "iterations": iteration, "hessian_condition": condition}
    bundle = _glm_evidence(
        sample, model, "float64_irls", "numeric", numerical=numerical,
        goodness_of_fit={"deviance": deviance, "null_deviance": null_deviance})
    result = EngineeringResult(
        operation="fit", status="verified", trust="numeric",
        value={"family": model.family, "link": model.link,
               "coefficients": dict(zip(names, fit.coefficients)),
               "deviance": fit.deviance, "null_deviance": fit.null_deviance,
               "iterations": iteration, "converged": True},
        details={"identity_checks": checks, "score_residual": fit.score_residual,
                 "hessian_condition": fit.hessian_condition,
                 "model_validity": "not_established",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        verification=checks, claim_evidence={"fit": bundle})
    return result, fit


def fit_glm(sample: StatisticalSample, model: GeneralizedLinearModel,
            model_id: str, *, max_iterations: int = 100,
            tolerance: float = 1e-9):
    names, rows, response = _glm_columns(sample, model)
    exact = sample.input_trust == "exact" and not any(
        value.has(sp.Float) for row in rows for value in row) and not any(
        value.has(sp.Float) for value in response)
    if not _response_domain(sample, model, response):
        raise ValueError("GLM specification failed: response_domain")
    if len(rows) <= len(names):
        raise ValueError("GLM specification failed: positive_residual_degrees_of_freedom")
    if model.family == "gaussian" and exact:
        information, rhs = _exact_crossproducts(rows, response)
        if information.rank() != len(names):
            raise ValueError("GLM specification failed: design_full_column_rank")
        return _exact_gaussian_fit(sample, model, model_id, names, rows, response,
                                   information, rhs)
    numeric_design = np.asarray(rows, dtype=float)
    if int(np.linalg.matrix_rank(numeric_design)) != len(names):
        raise ValueError("GLM specification failed: design_full_column_rank")
    if model.family == "gaussian":
        # Numeric Gaussian identity is solved as one weighted least-squares step.
        design = numeric_design; target = np.asarray(response, dtype=float)
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank != len(names):
            raise ValueError("GLM design is rank deficient")
        fitted = design @ coefficients; residual = target - fitted
        degrees = len(target) - len(names)
        dispersion = float((residual @ residual) / degrees)
        information = design.T @ design
        covariance = dispersion * np.linalg.inv(information)
        covariance = (covariance + covariance.T) / 2
        condition = float(np.linalg.cond(information))
        if not math.isfinite(condition) or condition > 1e14:
            raise ValueError("Gaussian information matrix is singular or ill-conditioned")
        score = design.T @ residual
        score_residual = float(np.linalg.norm(score, ord=np.inf))
        mean = float(np.mean(target))
        deviance = float(residual @ residual)
        null_deviance = float(np.sum((target - mean) ** 2))
        fit = GLMFit(
            model_id=model_id, family=model.family, link=model.link,
            response=model.response, predictors=model.predictors,
            coefficient_names=names,
            coefficients=tuple(_sp_float(item) for item in coefficients),
            covariance=tuple(tuple(_sp_float(item) for item in row) for row in covariance),
            standard_errors=tuple(_sp_float(math.sqrt(max(0.0, item)))
                                  for item in np.diag(covariance)),
            linear_predictors=tuple(_sp_float(item) for item in fitted),
            fitted_means=tuple(_sp_float(item) for item in fitted),
            deviance=_sp_float(deviance), null_deviance=_sp_float(null_deviance),
            dispersion=_sp_float(dispersion), iterations=1, converged=True,
            design_rank=len(names), score_residual=_sp_float(score_residual),
            hessian_condition=_sp_float(condition), input_trust="numeric")
        checks = {"normal_equations_within_tolerance": score_residual <=
                  max(1e-8, tolerance * 100 * len(target)),
                  "covariance_symmetric": bool(np.allclose(covariance, covariance.T)),
                  "design_full_column_rank": True}
        if not all(checks.values()):
            raise ValueError("Gaussian fit failed its residual verification")
        numerical = {"score_residual": score_residual, "converged": True,
                     "iterations": 1, "hessian_condition": condition}
        bundle = _glm_evidence(sample, model, "float64_least_squares", "numeric",
                               numerical=numerical,
                               goodness_of_fit={"deviance": deviance,
                                                "null_deviance": null_deviance})
        result = EngineeringResult(
            operation="fit", status="verified", trust="numeric",
            value={"family": model.family, "link": model.link,
                   "coefficients": dict(zip(names, fit.coefficients)),
                   "deviance": fit.deviance, "null_deviance": fit.null_deviance,
                   "iterations": 1, "converged": True},
            details={"identity_checks": checks, "dispersion": fit.dispersion,
                     "model_validity": "not_established",
                     "population_generalization": "not_established",
                     "causal_effect": "not_established"},
            verification=checks, claim_evidence={"fit": bundle})
        return result, fit
    return _numeric_glm_fit(sample, model, model_id, names, rows, response,
                            max_iterations=max_iterations, tolerance=tolerance)


def verify_glm_fit(sample: StatisticalSample, model: GeneralizedLinearModel,
                   fit: GLMFit):
    names, rows, response = _glm_columns(sample, model)
    coefficients = sp.Matrix(fit.coefficients)
    if model.family == "gaussian" and fit.input_trust == "exact":
        information, rhs = _exact_crossproducts(rows, response)
        score = information * coefficients - rhs
        linear = tuple(sum((row[index] * coefficients[index]
                            for index in range(len(names))), sp.S.Zero)
                       for row in rows)
        residual = tuple(target - fitted
                         for target, fitted in zip(response, linear))
        checks = {"normal_equations": all(sp.simplify(item) == 0 for item in score),
                  "deviance_recomputed": sp.simplify(
                      sum((item * item for item in residual), sp.S.Zero) -
                      fit.deviance) == 0,
                  "coefficient_names_match": names == fit.coefficient_names}
        trust = "exact"
        numerical = None
    else:
        array = np.asarray(rows, dtype=float); target = np.asarray(response, dtype=float)
        beta = np.asarray(fit.coefficients, dtype=float); linear = array @ beta
        exponential_range = True
        if model.family == "gaussian":
            mean = linear
        elif model.family == "binomial":
            mean = _logistic_mean(linear)
        else:
            if np.any(np.abs(linear) > 700):
                exponential_range = False
                mean = np.exp(np.clip(linear, -700, 700))
            else:
                mean = np.exp(linear)
        score = array.T @ (target - mean)
        residual = float(np.linalg.norm(score, ord=np.inf))
        checks = {"score_recomputed": exponential_range and residual <= max(1e-7, float(fit.score_residual) * 10 + 1e-12),
                  "coefficient_names_match": names == fit.coefficient_names,
                  "convergence_recorded": fit.converged}
        trust = "numeric"
        numerical = {"score_residual": residual, "converged": fit.converged,
                     "iterations": fit.iterations,
                     "hessian_condition": (float(fit.hessian_condition)
                                             if fit.hessian_condition else None)}
    bundle = _glm_evidence(sample, model, "independent_fit_recomputation", trust,
                           numerical=numerical,
                           goodness_of_fit={"deviance": sp.sstr(fit.deviance)})
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust, value={"checks": checks}, verification=checks,
        details={"model_validity": "not_established",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        claim_evidence={"verify": bundle})


def glm_diagnostics(sample: StatisticalSample, model: GeneralizedLinearModel,
                    fit: GLMFit):
    checks = {"converged": fit.converged,
              "full_column_rank": fit.design_rank == len(fit.coefficients),
              "covariance_symmetric": all(
                  sp.simplify(fit.covariance[i][j] - fit.covariance[j][i]) == 0
                  for i in range(len(fit.covariance))
                  for j in range(len(fit.covariance)))}
    value = {"family": fit.family, "link": fit.link,
             "deviance": fit.deviance, "null_deviance": fit.null_deviance,
             "dispersion": fit.dispersion, "iterations": fit.iterations,
             "score_residual": fit.score_residual,
             "hessian_condition": fit.hessian_condition,
             "coefficient_covariance": fit.covariance,
             "standard_errors": fit.standard_errors,
             "model_validity": "not_established"}
    numerical = None if fit.input_trust == "exact" else {
        "score_residual": float(fit.score_residual), "converged": fit.converged,
        "iterations": fit.iterations,
        "hessian_condition": (float(fit.hessian_condition)
                                if fit.hessian_condition else None)}
    bundle = _glm_evidence(sample, model, "glm_diagnostic_report",
                           fit.input_trust, numerical=numerical,
                           goodness_of_fit={"deviance": sp.sstr(fit.deviance),
                                            "null_deviance": sp.sstr(fit.null_deviance)})
    return EngineeringResult(
        operation="diagnostics", status="verified", trust=fit.input_trust,
        value=value, verification=checks,
        details={"diagnostics_do_not_prove_model_validity": True,
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        claim_evidence={"diagnostics": bundle})


def glm_predict(sample: StatisticalSample, model: GeneralizedLinearModel,
                fit: GLMFit, rows: tuple[tuple[sp.Expr, ...], ...],
                row_trust: str):
    if any(len(row) != len(model.predictors) for row in rows):
        raise ValueError("every prediction row must match the predictor count")
    design = tuple(((sp.S.One,) if model.include_intercept else ()) + row
                   for row in rows)
    trust = cap_trust(fit.input_trust, row_trust)
    if fit.family == "gaussian" and trust == "exact":
        predictions = tuple(sp.simplify(value) for value in
                            sp.Matrix(design) * sp.Matrix(fit.coefficients))
    else:
        linear = np.asarray(design, dtype=float) @ np.asarray(fit.coefficients, dtype=float)
        if fit.family == "gaussian":
            means = linear
        elif fit.family == "binomial":
            means = _logistic_mean(linear)
        else:
            if np.any(np.abs(linear) > 700):
                raise ValueError("Poisson prediction exceeds the finite float64 exponential range")
            means = np.exp(linear)
        predictions = tuple(_sp_float(item) for item in means)
        trust = "numeric"
    checks = {"row_dimensions_match": True,
              "finite_conditional_means": all(value.is_finite is True
                                                for value in predictions)}
    bundle = _glm_evidence(sample, model, "conditional_mean_prediction", trust)
    return EngineeringResult(
        operation="predict", status="verified", trust=trust,
        value={"conditional_means": predictions}, verification=checks,
        conditions=["predictions are conditional on the fitted model specification"],
        details={"prediction_scope": "conditional_mean_only",
                 "model_validity": "not_established",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        claim_evidence={"predict": bundle})
