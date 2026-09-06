# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded Phase F.4 right-censored survival analysis.

Product-limit arithmetic, numerical uncertainty, and conditional Cox fitting
are deliberately separate from censoring/model/population validity claims.
"""
from __future__ import annotations

from functools import cmp_to_key
import math

import mpmath as mp
import numpy as np
import sympy as sp

from mathkernel_artifacts import (
    ComputationEvidence, EmpiricalEvidence, EvidenceBundle, ModelEvidence,
    NumericalEvidence,
)

from .engineering import EngineeringResult, cap_trust
from .statistical_inference import (
    CoxPHFit, CoxProportionalHazardsModel, KaplanMeierEstimate,
    StatisticalSample, SurvivalDataset, _compare, _sorted,
)


def _column(sample: StatisticalSample, name: str) -> tuple[sp.Expr, ...]:
    if name not in sample.variables:
        raise ValueError(f"{name!r} must name a stored sample variable")
    index = sample.variables.index(name)
    return tuple(row[index] for row in sample.observations)


def _eq(left: sp.Expr, right: sp.Expr) -> bool:
    return _compare(left, right) == 0


def _le(left: sp.Expr, right: sp.Expr) -> bool:
    return _compare(left, right) <= 0


def _unique_sorted(values) -> tuple[sp.Expr, ...]:
    output = []
    for value in _sorted(tuple(values)):
        if not output or not _eq(output[-1], value):
            output.append(value)
    return tuple(output)


def _float(value, digits=17) -> sp.Float:
    return sp.Float(repr(float(value)), digits)


def _dataset_columns(sample: StatisticalSample, dataset: SurvivalDataset):
    duration = _column(sample, dataset.duration)
    event_raw = _column(sample, dataset.event)
    entry = (_column(sample, dataset.entry) if dataset.entry is not None
             else tuple(sp.S.Zero for _ in duration))
    strata = (_column(sample, dataset.strata) if dataset.strata is not None
              else tuple(sp.S.Zero for _ in duration))
    events = []
    for value in event_raw:
        if value == 0:
            events.append(0)
        elif value == 1:
            events.append(1)
        else:
            raise ValueError("event indicator must contain only exact 0 or 1")
    for start, stop in zip(entry, duration):
        if not _le(0, start) or not _le(start, stop):
            raise ValueError("survival times require 0 <= entry <= duration")
    return duration, tuple(events), entry, strata


def _survival_bundle(sample, dataset, operation, method, trust, *, numerical=False):
    assumptions = [
        "right censoring is non-informative conditional on represented covariates",
        "event and censoring times are recorded without silent row deletion",
        *dataset.survival_assumptions,
    ]
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="survival_analysis", method=method, arithmetic=trust,
            deterministic=True, trust=trust)],
        empirical=[EmpiricalEvidence(
            sample_size=len(sample.observations),
            sampling_method=sample.sampling_method, role="diagnostic",
            trust="empirical", metadata={"scope": "stored survival records"})],
        model=[ModelEvidence(
            assumptions=assumptions,
            diagnostics=["independent censoring is not established",
                         "population generalization is not established",
                         "causal interpretation is not established"],
            role="diagnostic", trust="unknown",
            metadata={"censoring": dataset.censoring,
                      "population_claim": "not_established",
                      "causal_claim": "not_established"})],
        justified_trust=trust)
    if numerical:
        bundle.numerical.append(NumericalEvidence(
            precision=53, residual=None, role="required", trust=trust,
            metadata={"error_bound": False, "method": method}))
    return bundle


def validate_survival_dataset(sample: StatisticalSample, dataset: SurvivalDataset):
    duration, events, entry, strata = _dataset_columns(sample, dataset)
    strata_count = len(_unique_sorted(strata)) if dataset.strata else 1
    checks = {
        "time_domain_valid": all(_le(0, start) and _le(start, stop)
                                 for start, stop in zip(entry, duration)),
        "event_indicator_binary": all(item in {0, 1} for item in events),
        "event_indicator_complete": len(events) == len(duration),
        "row_count_reconciled": len(duration) == len(sample.observations),
    }
    trust = cap_trust(sample.input_trust, dataset.input_trust)
    bundle = _survival_bundle(sample, dataset, "verify",
                              "survival_schema_and_risk_set_validation", trust)
    return EngineeringResult(
        operation="verify", status="verified", trust=trust,
        value={"observations": len(duration), "events": sum(events),
               "censored": len(events) - sum(events), "strata": strata_count,
               "delayed_entry": dataset.entry is not None},
        details={"identity_checks": checks,
                 "independent_censoring": "asserted_not_verified",
                 "population_generalization": "not_established"},
        verification=checks, claim_evidence={"verify": bundle})


def kaplan_meier(sample: StatisticalSample, dataset: SurvivalDataset,
                 dataset_id: str, *, stratum: sp.Expr | None = None,
                 confidence_level=0.95, max_timeline=100_000,
                 max_work=20_000_000, parameter_trust="exact"):
    duration, event, entry, strata = _dataset_columns(sample, dataset)
    if dataset.strata is None and stratum is not None:
        raise ValueError("stratum requires a configured strata column")
    if dataset.strata is not None:
        available = _unique_sorted(strata)
        if stratum is None and len(available) > 1:
            raise ValueError("stratum is required for a multi-stratum dataset")
        if stratum is not None and not any(_eq(stratum, item) for item in available):
            raise ValueError("requested stratum is not present in the survival data")
        selected = [index for index, item in enumerate(strata)
                    if _eq(item, stratum)]
    else:
        selected = list(range(len(duration)))
    if not selected:
        raise ValueError("selected survival stratum is empty")
    times = _unique_sorted(duration[index] for index in selected)
    if len(times) > max_timeline:
        raise ValueError("survival timeline exceeds max_survival_timeline_points")
    if len(times) * len(selected) > max_work:
        raise ValueError("Kaplan-Meier risk-set work exceeds max_survival_work")
    if not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level must be in (0,1)")
    z = math.sqrt(2) * float(mp.erfinv(float(confidence_level)))
    survival = sp.S.One
    greenwood = sp.S.Zero
    at_risk = []; event_counts = []; censor_counts = []; entered = []
    estimates = []; standard_errors = []; lowers = []; uppers = []
    median = None
    starts = _sorted(tuple(entry[index] for index in selected))
    stop_indices = sorted(
        selected, key=cmp_to_key(
            lambda left, right: _compare(duration[left], duration[right])))
    start_cursor = stop_cursor = exited_before = 0
    for time in times:
        arrivals = 0
        while start_cursor < len(starts) and _le(starts[start_cursor], time):
            arrivals += int(_eq(starts[start_cursor], time))
            start_cursor += 1
        group_end = stop_cursor
        deaths = censored = 0
        while (group_end < len(stop_indices) and
               _eq(duration[stop_indices[group_end]], time)):
            if event[stop_indices[group_end]]:
                deaths += 1
            else:
                censored += 1
            group_end += 1
        risk = start_cursor - exited_before
        if deaths > risk:
            raise ValueError("event count exceeds the constructed risk set")
        if deaths:
            survival = sp.simplify(survival * sp.Rational(risk - deaths, risk))
            if risk > deaths:
                greenwood += sp.Rational(deaths, risk * (risk - deaths))
        se = _float(float(survival) * math.sqrt(float(greenwood)))
        if survival == 0:
            lower = upper = sp.Float(0, 17)
        elif survival == 1:
            lower = upper = sp.Float(1, 17)
        else:
            log_survival = math.log(float(survival))
            loglog = math.log(-log_survival)
            transform_se = math.sqrt(float(greenwood)) / abs(log_survival)
            lower = _float(math.exp(-math.exp(loglog + z * transform_se)))
            upper = _float(math.exp(-math.exp(loglog - z * transform_se)))
        if median is None and float(survival) <= 0.5:
            median = time
        at_risk.append(risk); event_counts.append(deaths)
        censor_counts.append(censored); entered.append(arrivals)
        estimates.append(survival); standard_errors.append(se)
        lowers.append(lower); uppers.append(upper)
        exited_before += group_end - stop_cursor
        stop_cursor = group_end
    trust = cap_trust(sample.input_trust, parameter_trust,
                      "numeric_high_precision")
    output = KaplanMeierEstimate(
        dataset_id=dataset_id, stratum=stratum, timeline=times,
        at_risk=tuple(at_risk), events=tuple(event_counts),
        censored=tuple(censor_counts), entered=tuple(entered),
        survival=tuple(estimates), standard_errors=tuple(standard_errors),
        confidence_lower=tuple(lowers), confidence_upper=tuple(uppers),
        confidence_level=_float(confidence_level), median_survival=median,
        input_trust=trust)
    checks = {
        "risk_sets_nonnegative": all(n >= d + c for n, d, c in zip(
            at_risk, event_counts, censor_counts)),
        "survival_nonincreasing": all(estimates[index] <= estimates[index - 1]
                                      for index in range(1, len(estimates))),
        "intervals_ordered": all(float(lo) <= float(s) <= float(hi)
                                 for lo, s, hi in zip(lowers, estimates, uppers)),
        "counts_reconciled": sum(event_counts) + sum(censor_counts) == len(selected),
    }
    bundle = _survival_bundle(sample, dataset, "kaplan_meier",
                              "exact_product_limit_with_greenwood_log_log",
                              trust, numerical=True)
    result = EngineeringResult(
        operation="kaplan_meier", status="verified", trust=trust,
        value={"timeline": times, "survival": tuple(estimates),
               "standard_errors": tuple(standard_errors),
               "confidence_lower": tuple(lowers),
               "confidence_upper": tuple(uppers),
               "median_survival": median},
        details={"identity_checks": checks, "confidence_method": "greenwood_log_log",
                 "confidence_level": confidence_level,
                 "point_estimate_arithmetic": cap_trust(sample.input_trust, "exact"),
                 "independent_censoring": "asserted_not_verified",
                 "population_generalization": "not_established"},
        verification=checks, claim_evidence={"kaplan_meier": bundle})
    return result, output


def survival_at(estimate: KaplanMeierEstimate, time: sp.Expr):
    if time.free_symbols or time.is_real is not True:
        raise ValueError("query time must be a concrete real scalar")
    index = None
    for current, point in enumerate(estimate.timeline):
        if _le(point, time):
            index = current
        else:
            break
    if index is None:
        value, lower, upper, se = sp.S.One, sp.S.One, sp.S.One, sp.S.Zero
    else:
        value = estimate.survival[index]
        lower = estimate.confidence_lower[index]
        upper = estimate.confidence_upper[index]
        se = estimate.standard_errors[index]
    checks = {"step_function_lookup": True,
              "interval_contains_estimate": float(lower) <= float(value) <= float(upper)}
    return EngineeringResult(
        operation="survival_at", status="verified", trust=estimate.input_trust,
        value={"time": time, "survival": value, "standard_error": se,
               "confidence_interval": (lower, upper),
               "confidence_level": estimate.confidence_level},
        details={"right_continuous_step_function": True,
                 "population_generalization": "not_established"},
        verification=checks)


def verify_kaplan_meier(sample, dataset, result, max_timeline, max_work):
    recomputed_result, candidate = kaplan_meier(
        sample, dataset, result.dataset_id, stratum=result.stratum,
        confidence_level=float(result.confidence_level),
        max_timeline=max_timeline, max_work=max_work)
    checks = {
        "timeline_recomputed": candidate.timeline == result.timeline,
        "risk_sets_recomputed": candidate.at_risk == result.at_risk,
        "events_recomputed": candidate.events == result.events,
        "censoring_recomputed": candidate.censored == result.censored,
        "survival_recomputed": candidate.survival == result.survival,
        "uncertainty_recomputed": (candidate.standard_errors == result.standard_errors and
                                   candidate.confidence_lower == result.confidence_lower and
                                   candidate.confidence_upper == result.confidence_upper),
    }
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust, value={"checks": checks}, verification=checks,
        details={"independent_censoring": "asserted_not_verified",
                 "population_generalization": "not_established"},
        claim_evidence={"verify": recomputed_result.claim_evidence["kaplan_meier"]})


def _cox_arrays(sample, dataset, model):
    duration, event, entry, _ = _dataset_columns(sample, dataset)
    if dataset.strata is not None:
        raise ValueError("Phase F.4 Cox fitting requires an unstratified SurvivalDataset")
    forbidden = {dataset.duration, dataset.event, dataset.entry, dataset.strata}
    if any(name in forbidden or name not in sample.variables for name in model.predictors):
        raise ValueError("Cox predictors must name distinct non-survival sample columns")
    x = np.asarray([[float(row[sample.variables.index(name)])
                     for name in model.predictors] for row in sample.observations], dtype=float)
    stop = np.asarray([float(item) for item in duration], dtype=float)
    start = np.asarray([float(item) for item in entry], dtype=float)
    status = np.asarray(event, dtype=int)
    return x, start, stop, status


def validate_cox_model(sample, dataset, model):
    x, start, stop, status = _cox_arrays(sample, dataset, model)
    rank = int(np.linalg.matrix_rank(x - np.mean(x, axis=0)))
    events = int(np.sum(status))
    checks = {
        "predictor_rank_full": rank == x.shape[1],
        "events_exceed_parameters": events > x.shape[1],
        "finite_design": bool(np.all(np.isfinite(x))),
        "risk_intervals_valid": bool(np.all(start <= stop)),
        "tie_method_explicit": model.tie_method in {"breslow", "efron"},
    }
    status_text = "verified" if all(checks.values()) else "refuted"
    trust = cap_trust(sample.input_trust, "numeric")
    bundle = _survival_bundle(sample, dataset, "verify",
                              "cox_design_and_event_validation", trust, numerical=True)
    bundle.model.append(ModelEvidence(
        assumptions=["proportional hazards", "log-linear covariate effects"],
        diagnostics=["proportional hazards is not established"],
        role="diagnostic", trust="unknown"))
    return EngineeringResult(
        operation="verify", status=status_text, trust=trust,
        value={"checks": checks, "rank": rank, "parameters": x.shape[1],
               "events": events}, verification=checks,
        details={"proportional_hazards": "asserted_not_verified",
                 "independent_censoring": "asserted_not_verified",
                 "population_generalization": "not_established"},
        claim_evidence={"verify": bundle})


def _cox_components(beta, x, start, stop, status, tie_method):
    eta = x @ beta
    log_likelihood = 0.0
    score = np.zeros(x.shape[1])
    information = np.zeros((x.shape[1], x.shape[1]))
    event_times = np.unique(stop[status == 1])
    for time in event_times:
        event_indices = np.flatnonzero((stop == time) & (status == 1))
        risk_indices = np.flatnonzero((start <= time) & (stop >= time))
        d = len(event_indices)
        eta_risk = eta[risk_indices]
        shift = float(np.max(eta_risk))
        weights = np.exp(eta_risk - shift)
        risk_x = x[risk_indices]
        s0 = float(np.sum(weights))
        s1 = np.sum(weights[:, None] * risk_x, axis=0)
        s2 = np.einsum("i,ij,ik->jk", weights, risk_x, risk_x)
        event_weights = np.exp(eta[event_indices] - shift)
        event_x = x[event_indices]
        e0 = float(np.sum(event_weights))
        e1 = np.sum(event_weights[:, None] * event_x, axis=0)
        e2 = np.einsum("i,ij,ik->jk", event_weights, event_x, event_x)
        log_likelihood += float(np.sum(eta[event_indices]))
        score += np.sum(event_x, axis=0)
        for tied_index in range(d):
            fraction = tied_index / d if tie_method == "efron" else 0.0
            denominator = s0 - fraction * e0
            if not math.isfinite(denominator) or denominator <= 0:
                raise ValueError("Cox risk denominator is nonpositive")
            first = s1 - fraction * e1
            second = s2 - fraction * e2
            mean = first / denominator
            log_likelihood -= shift + math.log(denominator)
            score -= mean
            information += second / denominator - np.outer(mean, mean)
    return log_likelihood, score, information


def _cox_baseline(beta, x, start, stop, status, tie_method):
    eta = np.clip(x @ beta, -700, 700)
    weights = np.exp(eta)
    times = np.unique(stop[status == 1])
    increments = []
    cumulative = []; total = 0.0
    for time in times:
        event_indices = np.flatnonzero((stop == time) & (status == 1))
        risk_indices = np.flatnonzero((start <= time) & (stop >= time))
        d = len(event_indices); s0 = float(np.sum(weights[risk_indices]))
        e0 = float(np.sum(weights[event_indices]))
        increment = sum(1 / (s0 - (index / d if tie_method == "efron" else 0) * e0)
                        for index in range(d))
        total += increment; increments.append(increment); cumulative.append(total)
    return times, np.asarray(increments), np.asarray(cumulative)


def _concordance(beta, x, start, stop, status):
    risk = x @ beta; concordant = comparable = 0.0
    for left in range(len(stop)):
        for right in range(left + 1, len(stop)):
            if (stop[left] < stop[right] and status[left] == 1 and
                    start[right] <= stop[left]):
                earlier, later = left, right
            elif (stop[right] < stop[left] and status[right] == 1 and
                  start[left] <= stop[right]):
                earlier, later = right, left
            else:
                continue
            comparable += 1
            concordant += (1 if risk[earlier] > risk[later] else
                           0.5 if risk[earlier] == risk[later] else 0)
    return None if comparable == 0 else concordant / comparable


def _schoenfeld_correlations(beta, x, start, stop, status, tie_method):
    eta = np.clip(x @ beta, -700, 700); weights = np.exp(eta)
    residuals = []; log_times = []
    for time in np.unique(stop[status == 1]):
        risk = np.flatnonzero((start <= time) & (stop >= time))
        event = np.flatnonzero((stop == time) & (status == 1))
        s0 = np.sum(weights[risk])
        s1 = np.sum(weights[risk, None] * x[risk], axis=0)
        e0 = np.sum(weights[event])
        e1 = np.sum(weights[event, None] * x[event], axis=0)
        expected = np.mean([
            (s1 - (index / len(event) if tie_method == "efron" else 0) * e1) /
            (s0 - (index / len(event) if tie_method == "efron" else 0) * e0)
            for index in range(len(event))
        ], axis=0)
        for index in event:
            residuals.append(x[index] - expected)
            log_times.append(math.log(max(time, np.finfo(float).tiny)))
    residuals = np.asarray(residuals); log_times = np.asarray(log_times)
    output = []
    for column in range(x.shape[1]):
        values = residuals[:, column]
        if len(values) < 3 or np.std(values) == 0 or np.std(log_times) == 0:
            output.append(0.0)
        else:
            output.append(float(np.corrcoef(values, log_times)[0, 1]))
    return tuple(output)


def fit_cox(sample, dataset, model, model_id, *, max_iterations=100,
            tolerance=1e-9, max_condition=1e12):
    validation = validate_cox_model(sample, dataset, model)
    if validation.status != "verified":
        failed = [name for name, accepted in validation.verification.items() if not accepted]
        raise ValueError(f"Cox model verification failed: {', '.join(failed)}")
    x, start, stop, status = _cox_arrays(sample, dataset, model)
    beta = np.zeros(x.shape[1]); null_ll, _, _ = _cox_components(
        beta, x, start, stop, status, model.tie_method)
    converged = False; condition = math.inf
    for iteration in range(1, max_iterations + 1):
        ll, score, information = _cox_components(
            beta, x, start, stop, status, model.tie_method)
        condition = float(np.linalg.cond(information))
        if not math.isfinite(condition) or condition > max_condition:
            raise ValueError("Cox observed information is singular or ill-conditioned")
        score_norm = float(np.linalg.norm(score, ord=np.inf))
        if score_norm <= tolerance:
            converged = True
            break
        step = np.linalg.solve(information, score)
        step_scale = min(1.0, 5.0 / max(float(np.linalg.norm(step)), 5.0))
        accepted = False
        for _ in range(30):
            candidate = beta + step_scale * step
            candidate_ll, _, _ = _cox_components(
                candidate, x, start, stop, status, model.tie_method)
            if math.isfinite(candidate_ll) and candidate_ll >= ll - 1e-12:
                beta = candidate; accepted = True; break
            step_scale *= 0.5
        if not accepted:
            raise ValueError("Cox Newton line search failed")
        if float(np.linalg.norm(step_scale * step, ord=np.inf)) <= tolerance * (
                1 + float(np.linalg.norm(beta, ord=np.inf))):
            _, final_score, _ = _cox_components(
                beta, x, start, stop, status, model.tie_method)
            converged = float(np.linalg.norm(final_score, ord=np.inf)) <= math.sqrt(tolerance)
            if converged:
                break
    ll, score, information = _cox_components(beta, x, start, stop, status, model.tie_method)
    condition = float(np.linalg.cond(information))
    score_norm = float(np.linalg.norm(score, ord=np.inf))
    if not converged or score_norm > max(math.sqrt(tolerance), 1e-7):
        raise ValueError("Cox partial-likelihood fit did not converge")
    if np.max(np.abs(beta)) > 50:
        raise ValueError("Cox coefficients indicate monotone likelihood or separation")
    covariance = np.linalg.inv(information)
    if np.any(np.diag(covariance) < 0):
        raise ValueError("Cox covariance has a negative diagonal")
    standard_errors = np.sqrt(np.diag(covariance))
    times, hazard, cumulative = _cox_baseline(
        beta, x, start, stop, status, model.tie_method)
    concordance = _concordance(beta, x, start, stop, status)
    schoenfeld = _schoenfeld_correlations(
        beta, x, start, stop, status, model.tie_method)
    trust = cap_trust(sample.input_trust, "numeric")
    output = CoxPHFit(
        model_id=model_id, predictors=model.predictors, tie_method=model.tie_method,
        coefficients=tuple(_float(item) for item in beta),
        hazard_ratios=tuple(_float(math.exp(item)) for item in beta),
        covariance=tuple(tuple(_float(item) for item in row) for row in covariance),
        standard_errors=tuple(_float(item) for item in standard_errors),
        log_partial_likelihood=_float(ll), null_log_partial_likelihood=_float(null_ll),
        score_residual=_float(score_norm), iterations=iteration, converged=True,
        information_condition=_float(condition), event_count=int(np.sum(status)),
        baseline_timeline=tuple(_float(item) for item in times),
        baseline_hazard=tuple(_float(item) for item in hazard),
        baseline_cumulative_hazard=tuple(_float(item) for item in cumulative),
        concordance_index=None if concordance is None else _float(concordance),
        schoenfeld_time_correlations=tuple(_float(item) for item in schoenfeld),
        input_trust=trust)
    checks = {
        "converged": True,
        "score_residual_small": score_norm <= max(math.sqrt(tolerance), 1e-7),
        "information_positive_definite": bool(np.all(np.linalg.eigvalsh(information) > 0)),
        "baseline_hazard_nonnegative": bool(np.all(hazard >= 0)),
        "baseline_cumulative_monotone": bool(np.all(np.diff(cumulative) >= 0)),
    }
    bundle = _survival_bundle(sample, dataset, "fit",
                              f"cox_partial_likelihood_{model.tie_method}", trust,
                              numerical=True)
    bundle.model.append(ModelEvidence(
        assumptions=["proportional hazards", "log-linear covariate effects",
                     *model.model_assumptions],
        diagnostics=["Schoenfeld time correlations are diagnostics, not proof",
                     "proportional hazards is not established"],
        role="diagnostic", trust="unknown",
        metadata={"tie_method": model.tie_method}))
    result = EngineeringResult(
        operation="fit", status="verified", trust=trust,
        value={"coefficients": output.coefficients,
               "hazard_ratios": output.hazard_ratios,
               "log_partial_likelihood": output.log_partial_likelihood,
               "score_residual": output.score_residual},
        details={"identity_checks": checks, "iterations": iteration,
                 "tie_method": model.tie_method,
                 "proportional_hazards": "asserted_not_verified",
                 "independent_censoring": "asserted_not_verified",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        verification=checks, claim_evidence={"fit": bundle})
    return result, output


def verify_cox_fit(sample, dataset, model, result, *, max_iterations,
                   tolerance, max_condition):
    recomputed, candidate = fit_cox(
        sample, dataset, model, result.model_id, max_iterations=max_iterations,
        tolerance=tolerance, max_condition=max_condition)
    coefficient_error = max(abs(float(a) - float(b)) for a, b in zip(
        candidate.coefficients, result.coefficients))
    checks = {
        "coefficients_recomputed": coefficient_error <= max(1e-10, tolerance * 10),
        "tie_method_reconciled": candidate.tie_method == result.tie_method,
        "event_count_reconciled": candidate.event_count == result.event_count,
        "baseline_hazard_recomputed": candidate.baseline_timeline == result.baseline_timeline and
                                      candidate.baseline_hazard == result.baseline_hazard,
    }
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust, value={"checks": checks}, verification=checks,
        details={"coefficient_max_absolute_error": coefficient_error,
                 "proportional_hazards": "asserted_not_verified",
                 "population_generalization": "not_established"},
        claim_evidence={"verify": recomputed.claim_evidence["fit"]})


def cox_diagnostics(result: CoxPHFit):
    checks = {
        "fit_converged": result.converged,
        "score_finite": math.isfinite(float(result.score_residual)),
        "covariance_symmetric": all(abs(float(result.covariance[i][j] -
                                                   result.covariance[j][i])) <= 1e-10
                                    for i in range(len(result.predictors))
                                    for j in range(len(result.predictors))),
        "schoenfeld_diagnostics_present": len(result.schoenfeld_time_correlations) ==
                                           len(result.predictors),
    }
    return EngineeringResult(
        operation="diagnostics", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust,
        value={"concordance_index": result.concordance_index,
               "schoenfeld_time_correlations": result.schoenfeld_time_correlations,
               "score_residual": result.score_residual,
               "information_condition": result.information_condition},
        details={"proportional_hazards": "not_established_by_diagnostics",
                 "diagnostics_can_refute_or_question_but_not_prove_the_model": True,
                 "population_generalization": "not_established"},
        verification=checks)


def predict_partial_hazard(result: CoxPHFit, rows, row_trust):
    x = np.asarray([[float(item) for item in row] for row in rows], dtype=float)
    beta = np.asarray([float(item) for item in result.coefficients])
    eta = x @ beta
    if np.any(eta > 700) or np.any(eta < -700):
        raise ValueError("Cox prediction exceeds finite exponential range")
    hazards = np.exp(eta)
    trust = cap_trust(result.input_trust, row_trust, "numeric")
    checks = {"row_dimension_reconciled": x.shape[1] == len(beta),
              "partial_hazards_positive": bool(np.all(hazards > 0)),
              "finite_prediction": bool(np.all(np.isfinite(hazards)))}
    return EngineeringResult(
        operation="predict_partial_hazard", status="verified", trust=trust,
        value={"linear_predictors": tuple(_float(item) for item in eta),
               "partial_hazards": tuple(_float(item) for item in hazards)},
        details={"baseline_hazard_not_applied": True,
                 "conditional_model_prediction": True,
                 "proportional_hazards": "asserted_not_verified",
                 "population_generalization": "not_established"},
        verification=checks)
