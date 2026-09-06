# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded Phase F.6 stochastic-process objects and checked queries."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import (ComputationEvidence, EvidenceBundle,
                                  ModelEvidence, NumericalEvidence)

from .engineering import EngineeringModel, EngineeringResult, cap_trust
from .statistical_inference import _check_scalar, _compare


class PoissonProcess(EngineeringModel):
    rate: sp.Expr
    start_time: sp.Expr = sp.S.Zero
    assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_process(self):
        _check_scalar(self.rate); _check_scalar(self.start_time)
        if _compare(self.rate, sp.S.Zero) <= 0:
            raise ValueError("Poisson rate must be positive")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be nonempty strings")
        return self


class WienerProcess(EngineeringModel):
    drift: sp.Expr = sp.S.Zero
    diffusion: sp.Expr = sp.S.One
    initial: sp.Expr = sp.S.Zero
    start_time: sp.Expr = sp.S.Zero
    assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_process(self):
        for item in (self.drift, self.diffusion, self.initial, self.start_time):
            _check_scalar(item)
        if _compare(self.diffusion, sp.S.Zero) <= 0:
            raise ValueError("Wiener diffusion must be positive")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be nonempty strings")
        return self


class GaussianProcess(EngineeringModel):
    kernel: Literal["rbf", "matern32", "linear", "brownian"]
    mean: sp.Expr = sp.S.Zero
    variance: sp.Expr = sp.S.One
    length_scale: sp.Expr = sp.S.One
    observation_noise: sp.Expr = sp.S.Zero
    start_time: sp.Expr = sp.S.Zero
    assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_process(self):
        for item in (self.mean, self.variance, self.length_scale,
                     self.observation_noise, self.start_time):
            _check_scalar(item)
        if _compare(self.variance, 0) <= 0 or _compare(self.length_scale, 0) <= 0:
            raise ValueError("GP variance and length_scale must be positive")
        if _compare(self.observation_noise, 0) < 0:
            raise ValueError("GP observation_noise must be nonnegative")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be nonempty strings")
        return self


class ContinuousTimeMarkovChain(EngineeringModel):
    states: tuple[str, ...]
    generator: tuple[tuple[sp.Expr, ...], ...]
    initial_distribution: tuple[sp.Expr, ...]
    assumptions: tuple[str, ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_process(self):
        n = len(self.states)
        if not n or len(set(self.states)) != n or any(not state for state in self.states):
            raise ValueError("CTMC states must be nonempty and unique")
        if len(self.generator) != n or any(len(row) != n for row in self.generator):
            raise ValueError("CTMC generator must be square and match states")
        if len(self.initial_distribution) != n:
            raise ValueError("initial_distribution must match states")
        for row in self.generator:
            for item in row:
                _check_scalar(item)
        for item in self.initial_distribution:
            _check_scalar(item)
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("assumptions must be nonempty strings")
        return self


class FiniteDimensionalDistribution(EngineeringModel):
    process_id: str
    family: Literal["poisson_increment", "wiener", "wiener_increment", "gaussian_process"]
    query: Literal["finite_dimensional", "increment_distribution"]
    times: tuple[sp.Expr, ...]
    means: tuple[sp.Expr, ...]
    covariance: tuple[tuple[sp.Expr, ...], ...]
    increment_parameters: tuple[sp.Expr, ...] = ()
    independent_increments: bool | None = None
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_distribution(self):
        n = len(self.means)
        if not self.process_id or not self.times or not n:
            raise ValueError("finite-dimensional distribution must be nonempty")
        if len(self.covariance) != n or any(len(row) != n for row in self.covariance):
            raise ValueError("distribution covariance must match means")
        return self


class GaussianProcessPosterior(EngineeringModel):
    process_id: str
    observation_times: tuple[sp.Expr, ...]
    observation_values: tuple[sp.Expr, ...]
    prediction_times: tuple[sp.Expr, ...]
    means: tuple[sp.Expr, ...]
    covariance: tuple[tuple[sp.Expr, ...], ...]
    condition_number: sp.Expr
    jitter: sp.Expr
    method: Literal["cholesky_gaussian_conditioning"] = "cholesky_gaussian_conditioning"
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_posterior(self):
        if len(self.observation_times) != len(self.observation_values):
            raise ValueError("GP observations must align")
        n = len(self.prediction_times)
        if not n or len(self.means) != n or len(self.covariance) != n or any(
                len(row) != n for row in self.covariance):
            raise ValueError("GP posterior prediction fields must align")
        return self


class CTMCTransition(EngineeringModel):
    process_id: str
    time: sp.Expr
    matrix: tuple[tuple[sp.Expr, ...], ...]
    semigroup_residual: sp.Expr
    method: Literal["scipy_matrix_exponential"] = "scipy_matrix_exponential"
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_transition(self):
        n = len(self.matrix)
        if not self.process_id or not n or any(len(row) != n for row in self.matrix):
            raise ValueError("CTMC transition matrix must be nonempty and square")
        return self


def _float(value):
    return sp.Float(repr(float(value)), 17)


def _bundle(method, trust, assumptions, *, numerical=False):
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="stochastic_processes", method=method, arithmetic=trust,
            deterministic=True, trust=trust)],
        model=[ModelEvidence(
            assumptions=list(assumptions),
            diagnostics=["process law is asserted, not empirically established",
                         "population generalization is not established",
                         "causal interpretation is not established"],
            role="diagnostic", trust="unknown",
            metadata={"model_validity": "not_established"})],
        justified_trust=trust)
    if numerical:
        bundle.numerical.append(NumericalEvidence(
            precision=53, residual=None, role="required", trust=trust,
            metadata={"error_bound": False, "method": method}))
    return bundle


def _time_after(value, start, name="time"):
    _check_scalar(value)
    if _compare(value, start) < 0:
        raise ValueError(f"{name} must be at or after process start_time")


def _ordered_times(times, start):
    if not times:
        raise ValueError("times must be nonempty")
    for time in times:
        _time_after(time, start)
    if any(_compare(times[i - 1], times[i]) >= 0 for i in range(1, len(times))):
        raise ValueError("times must be strictly increasing")


def _min_time(left, right):
    """Return the exact earlier scalar without converting it to binary float."""
    return left if _compare(left, right) <= 0 else right


def verify_poisson(process):
    checks = {"rate_positive": _compare(process.rate, 0) > 0,
              "start_time_concrete": not process.start_time.free_symbols}
    trust = cap_trust(process.input_trust, "exact")
    return EngineeringResult(
        operation="verify", status="verified", trust=trust,
        value={"rate": process.rate, "start_time": process.start_time},
        details={"independent_stationary_increments": "asserted_not_verified",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"verify": _bundle(
            "poisson_parameter_domain", trust,
            ["homogeneous rate", "independent stationary increments", *process.assumptions])})


def poisson_pmf(process, time, count):
    _time_after(time, process.start_time)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    parameter = sp.simplify(process.rate * (time - process.start_time))
    probability = sp.simplify(sp.exp(-parameter) * parameter ** count / sp.factorial(count))
    trust = cap_trust(process.input_trust, "symbolic")
    return EngineeringResult(
        operation="pmf", status="verified", trust=trust,
        value={"time": time, "count": count, "parameter": parameter,
               "probability": probability},
        details={"distribution": "Poisson", "process_law": "asserted_not_verified"},
        verification={"count_domain": True, "parameter_nonnegative": _compare(parameter, 0) >= 0},
        claim_evidence={"pmf": _bundle(
            "poisson_pmf", trust,
            ["homogeneous Poisson process", *process.assumptions])})


def poisson_moments(process, time):
    _time_after(time, process.start_time)
    parameter = sp.simplify(process.rate * (time - process.start_time))
    trust = cap_trust(process.input_trust, "exact")
    return EngineeringResult(
        operation="moments", status="verified", trust=trust,
        value={"time": time, "mean": parameter, "variance": parameter},
        details={"process_law": "asserted_not_verified"},
        verification={"mean_equals_variance": True},
        claim_evidence={"moments": _bundle(
            "poisson_mean_variance", trust,
            ["homogeneous Poisson process", *process.assumptions])})


def poisson_increment(process, process_id, start, end):
    _time_after(start, process.start_time, "start")
    _time_after(end, process.start_time, "end")
    if _compare(start, end) >= 0:
        raise ValueError("increment requires start < end")
    parameter = sp.simplify(process.rate * (end - start))
    trust = cap_trust(process.input_trust, "exact")
    output = FiniteDimensionalDistribution(
        process_id=process_id, family="poisson_increment",
        query="increment_distribution", times=(start, end), means=(parameter,),
        covariance=((parameter,),), increment_parameters=(parameter,),
        independent_increments=True, input_trust=trust)
    result = EngineeringResult(
        operation="increment_distribution", status="verified", trust=trust,
        value={"start": start, "end": end, "parameter": parameter,
               "family": "Poisson"},
        details={"stationary_independent_increments": "asserted_not_verified",
                 "model_validity": "not_established"},
        verification={"positive_interval": True, "parameter_positive": _compare(parameter, 0) > 0},
        claim_evidence={"increment_distribution": _bundle(
            "poisson_increment_parameter", trust,
            ["homogeneous Poisson process", "independent stationary increments",
             *process.assumptions])})
    return result, output


def verify_wiener(process):
    checks = {"diffusion_positive": _compare(process.diffusion, 0) > 0,
              "parameters_concrete": True}
    trust = cap_trust(process.input_trust, "exact")
    return EngineeringResult(
        operation="verify", status="verified", trust=trust,
        value={"drift": process.drift, "diffusion": process.diffusion,
               "initial": process.initial, "start_time": process.start_time},
        details={"continuous_gaussian_independent_increments": "asserted_not_verified",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"verify": _bundle(
            "wiener_parameter_domain", trust,
            ["continuous paths", "Gaussian independent increments", *process.assumptions])})


def wiener_finite(process, process_id, times):
    _ordered_times(times, process.start_time)
    means = tuple(sp.simplify(process.initial + process.drift *
                              (time - process.start_time)) for time in times)
    covariance = tuple(tuple(sp.simplify(process.diffusion ** 2 *
        (_min_time(left, right) - process.start_time))
        for right in times) for left in times)
    trust = cap_trust(process.input_trust, "exact")
    output = FiniteDimensionalDistribution(
        process_id=process_id, family="wiener", query="finite_dimensional",
        times=times, means=means, covariance=covariance,
        independent_increments=True, input_trust=trust)
    checks = {"covariance_symmetric": all(covariance[i][j] == covariance[j][i]
              for i in range(len(times)) for j in range(len(times))),
              "diagonal_nonnegative": all(_compare(covariance[i][i], 0) >= 0
                                           for i in range(len(times)))}
    return EngineeringResult(
        operation="finite_dimensional", status="verified", trust=trust,
        value={"family": "multivariate_normal", "times": times,
               "means": means, "covariance": covariance},
        details={"continuous_paths": "asserted_not_verified",
                 "process_law": "asserted_not_verified"}, verification=checks,
        claim_evidence={"finite_dimensional": _bundle(
            "wiener_gaussian_finite_dimensional_law", trust,
            ["Wiener process with configured drift/diffusion", *process.assumptions])}), output


def wiener_increment(process, process_id, start, end):
    _time_after(start, process.start_time, "start"); _time_after(end, process.start_time, "end")
    if _compare(start, end) >= 0:
        raise ValueError("increment requires start < end")
    delta = sp.simplify(end - start)
    mean = sp.simplify(process.drift * delta)
    variance = sp.simplify(process.diffusion ** 2 * delta)
    trust = cap_trust(process.input_trust, "exact")
    output = FiniteDimensionalDistribution(
        process_id=process_id, family="wiener_increment",
        query="increment_distribution", times=(start, end), means=(mean,),
        covariance=((variance,),), independent_increments=True, input_trust=trust)
    return EngineeringResult(
        operation="increment_distribution", status="verified", trust=trust,
        value={"family": "normal", "mean": mean, "variance": variance},
        details={"independent_stationary_increments": "asserted_not_verified"},
        verification={"positive_interval": True, "variance_positive": _compare(variance, 0) > 0},
        claim_evidence={"increment_distribution": _bundle(
            "wiener_increment_normal_law", trust,
            ["Wiener process", "independent stationary increments", *process.assumptions])}), output


def _kernel(process, left, right):
    variance = process.variance
    if process.kernel == "rbf":
        return sp.simplify(variance * sp.exp(-(left - right) ** 2 /
                                             (2 * process.length_scale ** 2)))
    if process.kernel == "matern32":
        distance = sp.Abs(left - right)
        scaled = sp.sqrt(3) * distance / process.length_scale
        return sp.simplify(variance * (1 + scaled) * sp.exp(-scaled))
    if process.kernel == "linear":
        return sp.simplify(variance * (left - process.start_time) *
                           (right - process.start_time))
    return sp.simplify(variance * (_min_time(left, right) -
                                   process.start_time))


def verify_gp(process):
    checks = {"variance_positive": _compare(process.variance, 0) > 0,
              "length_scale_positive": _compare(process.length_scale, 0) > 0,
              "observation_noise_nonnegative": _compare(process.observation_noise, 0) >= 0,
              "kernel_supported": process.kernel in {"rbf", "matern32", "linear", "brownian"}}
    trust = cap_trust(process.input_trust, "symbolic")
    return EngineeringResult(
        operation="verify", status="verified", trust=trust,
        value={"kernel": process.kernel, "mean": process.mean,
               "variance": process.variance},
        details={"positive_semidefinite_kernel_family": "defining_family_identity",
                 "sample_path_properties": "not_established",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"verify": _bundle(
            "gp_kernel_parameter_domain", trust,
            [f"{process.kernel} covariance kernel", *process.assumptions])})


def gp_finite(process, process_id, times):
    if process.kernel == "brownian":
        _ordered_times(times, process.start_time)
    elif not times:
        raise ValueError("times must be nonempty")
    else:
        for time in times: _check_scalar(time)
    means = tuple(process.mean for _ in times)
    covariance = tuple(tuple(_kernel(process, left, right) for right in times)
                       for left in times)
    numeric = np.asarray([[float(item) for item in row] for row in covariance])
    minimum = float(np.min(np.linalg.eigvalsh((numeric + numeric.T) / 2)))
    if minimum < -1e-9 * max(1.0, float(np.max(np.abs(numeric)))):
        raise ValueError("GP covariance is not positive semidefinite at requested times")
    trust = cap_trust(process.input_trust, "symbolic")
    output = FiniteDimensionalDistribution(
        process_id=process_id, family="gaussian_process", query="finite_dimensional",
        times=times, means=means, covariance=covariance, input_trust=trust)
    checks = {"covariance_symmetric": all(covariance[i][j] == covariance[j][i]
              for i in range(len(times)) for j in range(len(times))),
              "numerical_psd_check": minimum >= -1e-9}
    return EngineeringResult(
        operation="finite_dimensional", status="verified", trust=trust,
        value={"family": "multivariate_normal", "times": times,
               "means": means, "covariance": covariance},
        details={"kernel": process.kernel,
                 "sample_path_properties": "not_established",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"finite_dimensional": _bundle(
            "gp_kernel_matrix", trust,
            [f"{process.kernel} Gaussian process", *process.assumptions])}), output


def gp_condition(process, process_id, observation_times, observation_values,
                 prediction_times, max_condition, jitter):
    if not observation_times or len(observation_times) != len(observation_values):
        raise ValueError("GP observation times and values must be nonempty and align")
    if not prediction_times:
        raise ValueError("GP prediction_times must be nonempty")
    for item in (*observation_times, *observation_values, *prediction_times):
        _check_scalar(item)
    if process.kernel == "brownian":
        for item in (*observation_times, *prediction_times):
            _time_after(item, process.start_time)
    xo = tuple(observation_times); xp = tuple(prediction_times)
    koo = np.asarray([[float(_kernel(process, a, b)) for b in xo] for a in xo])
    kop = np.asarray([[float(_kernel(process, a, b)) for b in xp] for a in xo])
    kpp = np.asarray([[float(_kernel(process, a, b)) for b in xp] for a in xp])
    noise = float(process.observation_noise)
    matrix = koo + (noise + jitter) * np.eye(len(xo))
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > max_condition:
        raise ValueError("GP observation covariance is singular or ill-conditioned")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("GP observation covariance is not positive definite") from exc
    centered = np.asarray([float(value - process.mean) for value in observation_values])
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, centered))
    means = float(process.mean) + kop.T @ alpha
    projection = np.linalg.solve(factor, kop)
    covariance = (kpp - projection.T @ projection)
    covariance = (covariance + covariance.T) / 2
    minimum = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum < -1e-8:
        raise ValueError("GP posterior covariance failed the PSD check")
    output = GaussianProcessPosterior(
        process_id=process_id, observation_times=xo,
        observation_values=tuple(observation_values), prediction_times=xp,
        means=tuple(_float(value) for value in means),
        covariance=tuple(tuple(_float(value) for value in row) for row in covariance),
        condition_number=_float(condition), jitter=_float(jitter), input_trust="numeric")
    checks = {"cholesky_succeeded": True, "posterior_covariance_symmetric": True,
              "posterior_covariance_psd": minimum >= -1e-8}
    return EngineeringResult(
        operation="condition", status="verified", trust="numeric",
        value={"prediction_times": xp, "means": output.means,
               "covariance": output.covariance},
        details={"kernel": process.kernel, "condition_number": condition,
                 "jitter": jitter, "hyperparameters_fitted": False,
                 "model_validity": "not_established",
                 "population_generalization": "not_established"}, verification=checks,
        claim_evidence={"condition": _bundle(
            "cholesky_gaussian_conditioning", "numeric",
            [f"{process.kernel} Gaussian process", "Gaussian observation model",
             *process.assumptions], numerical=True)}), output


def verify_ctmc(process):
    n = len(process.states)
    row_sums = tuple(sp.simplify(sum(row, sp.S.Zero)) for row in process.generator)
    checks = {"row_sums_zero": all(value == 0 for value in row_sums),
              "off_diagonal_nonnegative": all(_compare(process.generator[i][j], 0) >= 0
                  for i in range(n) for j in range(n) if i != j),
              "diagonal_nonpositive": all(_compare(process.generator[i][i], 0) <= 0
                                            for i in range(n)),
              "initial_nonnegative": all(_compare(value, 0) >= 0
                                           for value in process.initial_distribution),
              "initial_normalized": sp.simplify(sum(process.initial_distribution) - 1) == 0}
    status = "verified" if all(checks.values()) else "refuted"
    trust = cap_trust(process.input_trust, "exact")
    return EngineeringResult(
        operation="verify", status=status, trust=trust,
        value={"states": process.states, "row_sums": row_sums},
        details={"generator_interpretation": "asserted_time_homogeneous_ctmc",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"verify": _bundle(
            "ctmc_generator_axioms", trust,
            ["time-homogeneous CTMC generator", *process.assumptions])})


def ctmc_transition(process, process_id, time):
    _check_scalar(time)
    if _compare(time, 0) < 0:
        raise ValueError("CTMC transition time must be nonnegative")
    if verify_ctmc(process).status != "verified":
        raise ValueError("CTMC generator verification failed")
    try:
        from scipy.linalg import expm
    except ImportError as exc:
        raise ValueError("CTMC transition matrices require the scipy optional dependency") from exc
    q = np.asarray([[float(value) for value in row] for row in process.generator])
    t = float(time); matrix = expm(q * t); half = expm(q * (t / 2))
    residual = float(np.linalg.norm(matrix - half @ half, ord=np.inf))
    tolerance = 1e-10 * max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    if (np.min(matrix) < -tolerance or
            np.max(np.abs(np.sum(matrix, axis=1) - 1)) > tolerance):
        raise ValueError("CTMC matrix exponential failed stochasticity checks")
    matrix[np.abs(matrix) < tolerance] = 0.0
    output = CTMCTransition(
        process_id=process_id, time=time,
        matrix=tuple(tuple(_float(value) for value in row) for row in matrix),
        semigroup_residual=_float(residual), input_trust="numeric")
    checks = {"entries_nonnegative": bool(np.min(matrix) >= 0),
              "rows_normalized": bool(np.allclose(np.sum(matrix, axis=1), 1, atol=tolerance)),
              "semigroup_residual_small": residual <= tolerance}
    return EngineeringResult(
        operation="transition_matrix", status="verified", trust="numeric",
        value={"time": time, "matrix": output.matrix},
        details={"method": output.method, "semigroup_residual": residual,
                 "generator_model_validity": "not_established"}, verification=checks,
        claim_evidence={"transition_matrix": _bundle(
            "scipy_matrix_exponential", "numeric",
            ["time-homogeneous CTMC generator", *process.assumptions], numerical=True)}), output


def ctmc_distribution(process, time):
    result, transition = ctmc_transition(process, "query_only", time)
    initial = np.asarray([float(value) for value in process.initial_distribution])
    matrix = np.asarray([[float(value) for value in row] for row in transition.matrix])
    distribution = initial @ matrix
    checks = {"nonnegative": bool(np.min(distribution) >= -1e-12),
              "normalized": abs(float(np.sum(distribution)) - 1) <= 1e-10}
    return EngineeringResult(
        operation="distribution", status="verified", trust="numeric",
        value={"time": time, "states": process.states,
               "probabilities": tuple(_float(value) for value in distribution)},
        details={"generator_model_validity": "not_established"}, verification=checks,
        claim_evidence={"distribution": result.claim_evidence["transition_matrix"]})


def ctmc_stationary(process):
    verified = verify_ctmc(process)
    if verified.status != "verified":
        raise ValueError("CTMC generator verification failed")
    q = sp.Matrix(process.generator)
    variables = sp.symbols(f"p0:{len(process.states)}")
    equations = list(sp.Matrix(1, len(variables), variables) * q)
    solution = sp.linsolve([*equations, sum(variables) - 1], variables)
    candidates = list(solution)
    if len(candidates) != 1 or any(value.free_symbols for value in candidates[0]):
        return EngineeringResult(
            operation="stationary_distribution", status="unknown", trust="unknown",
            value={"reason": "stationary distribution is not unique"},
            details={"ergodicity": "not_established"},
            verification={"unique_solution": False})
    probabilities = tuple(sp.simplify(value) for value in candidates[0])
    nonnegative = all(_compare(value, 0) >= 0 for value in probabilities)
    checks = {"left_null_vector": all(value == 0 for value in
              (sp.Matrix(1, len(probabilities), probabilities) * q)),
              "normalized": sp.simplify(sum(probabilities) - 1) == 0,
              "nonnegative": nonnegative}
    trust = cap_trust(process.input_trust, "exact")
    return EngineeringResult(
        operation="stationary_distribution",
        status="verified" if all(checks.values()) else "refuted", trust=trust,
        value={"states": process.states, "probabilities": probabilities},
        details={"uniqueness": "linear_system_unique",
                 "convergence_to_stationarity": "not_established"}, verification=checks,
        claim_evidence={"stationary_distribution": _bundle(
            "exact_generator_left_nullspace", trust,
            ["time-homogeneous CTMC generator", *process.assumptions])})


def verify_distribution(process, distribution):
    if distribution.family == "poisson_increment":
        _, candidate = poisson_increment(process, distribution.process_id,
                                         distribution.times[0], distribution.times[1])
    elif distribution.family == "wiener_increment":
        _, candidate = wiener_increment(process, distribution.process_id,
                                        distribution.times[0], distribution.times[1])
    elif distribution.family == "wiener":
        _, candidate = wiener_finite(process, distribution.process_id,
                                     distribution.times)
    else:
        _, candidate = gp_finite(process, distribution.process_id,
                                 distribution.times)
    checks = {"family_reconciled": candidate.family == distribution.family,
              "means_recomputed": candidate.means == distribution.means,
              "covariance_recomputed": candidate.covariance == distribution.covariance,
              "increments_recomputed": candidate.increment_parameters == distribution.increment_parameters}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=distribution.input_trust, value={"checks": checks}, verification=checks,
        details={"process_law": "asserted_not_verified"})


def verify_posterior(process, posterior, max_condition):
    _, candidate = gp_condition(
        process, posterior.process_id, posterior.observation_times,
        posterior.observation_values, posterior.prediction_times,
        max_condition, float(posterior.jitter))
    checks = {"means_recomputed": candidate.means == posterior.means,
              "covariance_recomputed": candidate.covariance == posterior.covariance,
              "method_reconciled": candidate.method == posterior.method}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="numeric", value={"checks": checks}, verification=checks,
        details={"model_validity": "not_established"})


def verify_transition(process, transition):
    _, candidate = ctmc_transition(process, transition.process_id, transition.time)
    checks = {"matrix_recomputed": candidate.matrix == transition.matrix,
              "method_reconciled": candidate.method == transition.method,
              "semigroup_residual_recomputed": candidate.semigroup_residual == transition.semigroup_residual}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="numeric", value={"checks": checks}, verification=checks,
        details={"generator_model_validity": "not_established"})
