# =============================================================================
# MathKernel - Statistical inference for observed finite phylogenetic tensors
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Statistically valid inference for observed finite phylogenetic tensors.

This module is the numerical/statistical companion to
:mod:`mathkernel.phylogenetic_tensor`. The exact module proves how local
observation channels transform latent joint laws and edge flattenings. This
module addresses the finite-sample problem that exact left-inverse recovery
does not solve:

* probability-simplex constrained recovery through a known channel;
* likelihood and ridge regularisation without negative probability cells;
* multinomial covariance and Fisher-information propagation;
* nonnegative-rank likelihood fits for General-Markov edge flattenings;
* covariance-aware (Wald) diagnostics for rank constraints; and
* predeclared quartet split scoring from count tensors.

Only NumPy and mpmath are required. Public estimators always return valid
probability vectors or matrices. Algebraic exactness remains the responsibility
of :mod:`mathkernel.phylogenetic_tensor`; routines here are explicitly
floating-point and report convergence and conditioning diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Iterable, Mapping, Sequence

import mpmath as mp
import numpy as np

from .phylogenetic_tensor import FiniteObservationChannel


ArrayLike = Sequence[float] | np.ndarray


def _as_float_vector(values: ArrayLike, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _as_count_vector(values: ArrayLike, *, name: str = "counts") -> np.ndarray:
    counts = _as_float_vector(values, name=name)
    if np.any(counts < 0):
        raise ValueError(f"{name} must be nonnegative")
    if float(counts.sum()) <= 0:
        raise ValueError(f"{name} must contain positive total mass")
    return counts


def _as_channel_operator(operator: ArrayLike | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("channel operator must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("channel operator must be finite and nonnegative")
    column_sums = matrix.sum(axis=0)
    if not np.allclose(column_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("every channel-operator column must sum to one")
    return matrix


def normalize_probability_vector(
    values: ArrayLike,
    *,
    tolerance: float = 1e-12,
    name: str = "probabilities",
) -> np.ndarray:
    """Validate and normalize a floating-point probability vector.

    Tiny negative roundoff within ``tolerance`` is clipped. Materially
    negative entries are rejected rather than silently repaired.
    """

    vector = _as_float_vector(values, name=name).copy()
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if float(vector.min()) < -tolerance:
        raise ValueError(f"{name} must be nonnegative")
    vector[vector < 0] = 0.0
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive total mass")
    return vector / total


def project_probability_simplex(values: ArrayLike, *, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum(x) = total}``."""

    vector = _as_float_vector(values, name="values")
    if not isfinite(total) or total <= 0:
        raise ValueError("simplex total must be finite and positive")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - total
    indices = np.arange(1, vector.size + 1, dtype=float)
    active = ordered - cumulative / indices > 0
    if not np.any(active):
        return np.full(vector.size, total / vector.size, dtype=float)
    rho = int(np.nonzero(active)[0][-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    error = total - float(projected.sum())
    if abs(error) > 8 * np.finfo(float).eps * max(1.0, total):
        positive = np.flatnonzero(projected > 0)
        if positive.size:
            projected[positive] += error / positive.size
            projected = np.maximum(projected, 0.0)
    projected *= total / float(projected.sum())
    return projected


def local_observation_operator(
    channels: Sequence[FiniteObservationChannel | Sequence[Sequence[float]]],
) -> np.ndarray:
    """Return the Kronecker probability-vector operator for local channels.

    :class:`FiniteObservationChannel` stores rows ``P(Y=y | X=x)``. Its
    probability-vector operator is the transpose, with columns summing to one.
    Plain matrices are interpreted directly as probability-vector operators.
    """

    if not channels:
        raise ValueError("at least one observation channel is required")
    result = np.ones((1, 1), dtype=float)
    for channel in channels:
        raw = channel.operator if isinstance(channel, FiniteObservationChannel) else channel
        result = np.kron(result, _as_channel_operator(raw))
    return result


def multinomial_covariance(
    probabilities: ArrayLike,
    sample_size: int,
    *,
    for_counts: bool = False,
) -> np.ndarray:
    """Return multinomial covariance for counts or empirical proportions."""

    p = normalize_probability_vector(probabilities)
    n = int(sample_size)
    if n <= 0:
        raise ValueError("sample_size must be positive")
    covariance = np.diag(p) - np.outer(p, p)
    return covariance * n if for_counts else covariance / n


def propagate_linear_covariance(
    covariance: Sequence[Sequence[float]] | np.ndarray,
    operator: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Propagate covariance through a linear map ``x -> operator @ x``."""

    sigma = np.asarray(covariance, dtype=float)
    transform = np.asarray(operator, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be square")
    if transform.ndim != 2 or transform.shape[1] != sigma.shape[0]:
        raise ValueError("operator and covariance dimensions do not agree")
    if not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(transform)):
        raise ValueError("operator and covariance must be finite")
    result = transform @ sigma @ transform.T
    return (result + result.T) / 2.0


def _helmert_tangent_basis(size: int) -> np.ndarray:
    """Orthonormal basis of vectors whose coordinates sum to zero."""

    if size < 2:
        return np.zeros((size, 0), dtype=float)
    basis = np.zeros((size, size - 1), dtype=float)
    for column in range(size - 1):
        count = column + 1
        scale = np.sqrt(count * (count + 1.0))
        basis[:count, column] = 1.0 / scale
        basis[count, column] = -count / scale
    return basis


@dataclass(frozen=True, slots=True)
class FisherCovarianceResult:
    covariance: np.ndarray
    tangent_information: np.ndarray
    tangent_rank: int
    condition_number: float
    observed_probabilities: np.ndarray
    sample_size: int


@dataclass(frozen=True, slots=True)
class ChannelRecoveryResult:
    method: str
    probabilities: np.ndarray
    fitted_observed_probabilities: np.ndarray
    log_likelihood: float
    objective: float
    iterations: int
    converged: bool
    sample_size: float
    prior_strength: float
    l1_step: float
    channel_rank: int
    channel_condition_number: float
    simplex_violation: float


@dataclass(frozen=True, slots=True)
class RegularizationSelection:
    selected_strength: float
    selected_result: ChannelRecoveryResult
    validation_log_likelihoods: tuple[tuple[float, float], ...]
    results: tuple[ChannelRecoveryResult, ...]


@dataclass(frozen=True, slots=True)
class LatentClassFit:
    rank_bound: int
    probabilities: np.ndarray
    weights: np.ndarray
    left_components: np.ndarray
    right_components: np.ndarray
    log_likelihood: float
    deviance: float
    iterations: int
    converged: bool
    restart: int
    restarts: int
    effective_parameter_count: int
    aic: float
    bic: float
    sample_size: float
    numerical_rank: int


@dataclass(frozen=True, slots=True)
class RankWaldResult:
    statistic: float
    degrees_of_freedom: int
    p_value: float
    residual_norm: float
    covariance_rank: int
    covariance_condition_number: float
    rank_bound: int
    fitted_rank: int
    applicable: bool
    warning: str | None


@dataclass(frozen=True, slots=True)
class QuartetSplitScore:
    split: tuple[int, int]
    complement: tuple[int, int]
    matrix_shape: tuple[int, int]
    sample_size: float
    raw_tail_frobenius: float
    relative_tail_frobenius: float
    likelihood_deviance: float
    deviance_per_site: float
    log_likelihood: float
    wald_statistic: float
    wald_degrees_of_freedom: int
    wald_p_value: float
    fit_converged: bool
    fit_iterations: int
    fitted_numerical_rank: int


def known_channel_fisher_covariance(
    latent_probabilities: ArrayLike,
    channel_operator: Sequence[Sequence[float]] | np.ndarray,
    sample_size: int,
    *,
    rcond: float = 1e-12,
    probability_floor: float = 1e-15,
) -> FisherCovarianceResult:
    """Approximate latent covariance from observed multinomial Fisher information."""

    p = normalize_probability_vector(latent_probabilities)
    operator = _as_channel_operator(channel_operator)
    if operator.shape[1] != p.size:
        raise ValueError("channel and latent probability dimensions disagree")
    n = int(sample_size)
    if n <= 0:
        raise ValueError("sample_size must be positive")
    if rcond <= 0 or probability_floor <= 0:
        raise ValueError("rcond and probability_floor must be positive")
    q = operator @ p
    reachable = q > probability_floor
    weighted = operator[reachable, :] / np.sqrt(q[reachable, None])
    information = weighted.T @ weighted
    tangent = _helmert_tangent_basis(p.size)
    tangent_information = n * (tangent.T @ information @ tangent)
    if tangent_information.size == 0:
        covariance = np.zeros((p.size, p.size), dtype=float)
        rank = 0
        condition = 1.0
    else:
        singular = np.linalg.svd(tangent_information, compute_uv=False)
        cutoff = rcond * singular[0] if singular.size and singular[0] > 0 else 0.0
        positive = singular[singular > cutoff]
        rank = int(positive.size)
        condition = (
            float(positive[0] / positive[-1])
            if positive.size == tangent_information.shape[0]
            else float("inf")
        )
        covariance_tangent = np.linalg.pinv(tangent_information, rcond=rcond)
        covariance = tangent @ covariance_tangent @ tangent.T
        covariance = (covariance + covariance.T) / 2.0
    return FisherCovarianceResult(
        covariance=covariance,
        tangent_information=tangent_information,
        tangent_rank=rank,
        condition_number=condition,
        observed_probabilities=q,
        sample_size=n,
    )


def _channel_condition(operator: np.ndarray, *, rcond: float = 1e-12) -> tuple[int, float]:
    singular = np.linalg.svd(operator, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0, float("inf")
    rank = int(np.count_nonzero(singular > rcond * singular[0]))
    if rank < operator.shape[1]:
        return rank, float("inf")
    return rank, float(singular[0] / singular[-1])


def _multinomial_log_likelihood(counts: np.ndarray, probabilities: np.ndarray) -> float:
    positive = counts > 0
    if np.any(probabilities[positive] <= 0):
        return float("-inf")
    return float(np.dot(counts[positive], np.log(probabilities[positive])))


def fit_known_channel_distribution(
    counts: ArrayLike,
    channel_operator: Sequence[Sequence[float]] | np.ndarray,
    *,
    prior: ArrayLike | None = None,
    prior_strength: float = 0.0,
    initial: ArrayLike | None = None,
    max_iterations: int = 20_000,
    tolerance: float = 1e-11,
    probability_floor: float = 1e-15,
) -> ChannelRecoveryResult:
    """Fit a latent simplex distribution through a known stochastic channel.

    Uses EM for a finite mixture with known emissions. ``prior_strength``
    supplies Dirichlet pseudo-count mass distributed according to ``prior``.
    Every iterate is nonnegative and sums to one.
    """

    c = _as_count_vector(counts)
    operator = _as_channel_operator(channel_operator)
    if operator.shape[0] != c.size:
        raise ValueError("counts and channel output dimensions disagree")
    if not isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("prior_strength must be finite and nonnegative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0 or probability_floor <= 0:
        raise ValueError("tolerance and probability_floor must be positive")
    impossible = (operator.max(axis=1) == 0) & (c > 0)
    if np.any(impossible):
        raise ValueError("positive counts occur in an impossible observed state")

    latent_size = operator.shape[1]
    prior_vector = (
        np.full(latent_size, 1.0 / latent_size, dtype=float)
        if prior is None
        else normalize_probability_vector(prior, name="prior")
    )
    if prior_vector.size != latent_size:
        raise ValueError("prior has the wrong latent dimension")
    if initial is None:
        backprojection = operator.T @ (c / float(c.sum()))
        p = normalize_probability_vector(backprojection + 1e-12 * prior_vector)
    else:
        p = normalize_probability_vector(initial, name="initial")
        if p.size != latent_size:
            raise ValueError("initial has the wrong latent dimension")

    sample_size = float(c.sum())
    converged = False
    l1_step = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        q = operator @ p
        positive = c > 0
        if np.any(q[positive] <= probability_floor):
            p = normalize_probability_vector((1.0 - 1e-10) * p + 1e-10 * prior_vector)
            q = operator @ p
            if np.any(q[positive] <= 0):
                raise ValueError("current latent support cannot explain positive counts")
        expected_latent = p * (operator.T @ (c / np.maximum(q, probability_floor)))
        updated = (expected_latent + prior_strength * prior_vector) / (
            sample_size + prior_strength
        )
        updated = normalize_probability_vector(updated)
        l1_step = float(np.linalg.norm(updated - p, ord=1))
        p = updated
        if l1_step <= tolerance:
            converged = True
            break

    fitted = operator @ p
    log_likelihood = _multinomial_log_likelihood(c, fitted)
    if prior_strength > 0:
        prior_positive = prior_vector > 0
        penalty = (
            float("-inf")
            if np.any(p[prior_positive] <= 0)
            else float(
                prior_strength
                * np.dot(prior_vector[prior_positive], np.log(p[prior_positive]))
            )
        )
    else:
        penalty = 0.0
    rank, condition = _channel_condition(operator)
    simplex_violation = max(abs(float(p.sum()) - 1.0), max(0.0, -float(p.min())))
    return ChannelRecoveryResult(
        method="known-channel-em",
        probabilities=p,
        fitted_observed_probabilities=fitted,
        log_likelihood=log_likelihood,
        objective=log_likelihood + penalty,
        iterations=iteration,
        converged=converged,
        sample_size=sample_size,
        prior_strength=float(prior_strength),
        l1_step=l1_step,
        channel_rank=rank,
        channel_condition_number=condition,
        simplex_violation=simplex_violation,
    )


def fit_ridge_channel_distribution(
    counts: ArrayLike,
    channel_operator: Sequence[Sequence[float]] | np.ndarray,
    *,
    ridge: float,
    prior: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    max_iterations: int = 20_000,
    tolerance: float = 1e-11,
) -> ChannelRecoveryResult:
    """Simplex-constrained ridge recovery by accelerated projected gradient."""

    c = _as_count_vector(counts)
    operator = _as_channel_operator(channel_operator)
    if operator.shape[0] != c.size:
        raise ValueError("counts and channel output dimensions disagree")
    if not isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    latent_size = operator.shape[1]
    prior_vector = (
        np.full(latent_size, 1.0 / latent_size, dtype=float)
        if prior is None
        else normalize_probability_vector(prior, name="prior")
    )
    if prior_vector.size != latent_size:
        raise ValueError("prior has the wrong latent dimension")
    if initial is None:
        p = prior_vector.copy()
    else:
        p = normalize_probability_vector(initial, name="initial")
        if p.size != latent_size:
            raise ValueError("initial has the wrong latent dimension")
    y = c / float(c.sum())
    lipschitz = float(np.linalg.norm(operator, ord=2) ** 2 + ridge)
    if lipschitz <= 0:
        raise ValueError("channel operator has zero Lipschitz constant")
    step = 1.0 / lipschitz
    accelerated = p.copy()
    momentum = 1.0
    converged = False
    l1_step = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        gradient = operator.T @ (operator @ accelerated - y)
        if ridge:
            gradient += ridge * (accelerated - prior_vector)
        updated = project_probability_simplex(accelerated - step * gradient)
        l1_step = float(np.linalg.norm(updated - p, ord=1))
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        extrapolated = updated + ((momentum - 1.0) / next_momentum) * (updated - p)
        if float(np.dot(updated - p, extrapolated - updated)) > 0:
            accelerated = updated.copy()
            next_momentum = 1.0
        else:
            accelerated = extrapolated
        p = updated
        momentum = next_momentum
        if l1_step <= tolerance:
            converged = True
            break
    fitted = operator @ p
    residual = fitted - y
    objective = 0.5 * float(residual @ residual) + 0.5 * ridge * float(
        (p - prior_vector) @ (p - prior_vector)
    )
    rank, condition = _channel_condition(operator)
    return ChannelRecoveryResult(
        method="simplex-ridge",
        probabilities=p,
        fitted_observed_probabilities=fitted,
        log_likelihood=_multinomial_log_likelihood(c, fitted),
        objective=objective,
        iterations=iteration,
        converged=converged,
        sample_size=float(c.sum()),
        prior_strength=float(ridge),
        l1_step=l1_step,
        channel_rank=rank,
        channel_condition_number=condition,
        simplex_violation=max(abs(float(p.sum()) - 1.0), max(0.0, -float(p.min()))),
    )


def select_known_channel_regularization(
    training_counts: ArrayLike,
    validation_counts: ArrayLike,
    channel_operator: Sequence[Sequence[float]] | np.ndarray,
    strengths: Iterable[float],
    *,
    prior: ArrayLike | None = None,
    max_iterations: int = 20_000,
    tolerance: float = 1e-11,
) -> RegularizationSelection:
    """Select Dirichlet pseudo-count strength using held-out likelihood."""

    train = _as_count_vector(training_counts, name="training_counts")
    validation = _as_count_vector(validation_counts, name="validation_counts")
    if train.size != validation.size:
        raise ValueError("training and validation counts must have equal length")
    candidates = tuple(float(value) for value in strengths)
    if not candidates or any(not isfinite(value) or value < 0 for value in candidates):
        raise ValueError("strengths must be a nonempty family of finite nonnegative values")
    operator = _as_channel_operator(channel_operator)
    results: list[ChannelRecoveryResult] = []
    scores: list[tuple[float, float]] = []
    warm: np.ndarray | None = None
    for strength in candidates:
        result = fit_known_channel_distribution(
            train,
            operator,
            prior=prior,
            prior_strength=strength,
            initial=warm,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        warm = result.probabilities
        score = _multinomial_log_likelihood(validation, result.fitted_observed_probabilities)
        results.append(result)
        scores.append((strength, score))
    winner = max(range(len(scores)), key=lambda index: (scores[index][1], -scores[index][0]))
    return RegularizationSelection(
        selected_strength=candidates[winner],
        selected_result=results[winner],
        validation_log_likelihoods=tuple(scores),
        results=tuple(results),
    )


def _validate_count_matrix(counts: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(counts, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("counts must be a nonempty matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("counts must be finite and nonnegative")
    if float(matrix.sum()) <= 0:
        raise ValueError("counts must contain positive total mass")
    return matrix


def _latent_class_parameters(rows: int, columns: int, rank_bound: int) -> int:
    return (rank_bound - 1) + rank_bound * (rows - 1) + rank_bound * (columns - 1)


def fit_nonnegative_rank(
    counts: Sequence[Sequence[float]] | np.ndarray,
    rank_bound: int,
    *,
    restarts: int = 8,
    seed: int = 0,
    max_iterations: int = 5_000,
    tolerance: float = 1e-9,
    component_floor: float = 1e-12,
    pseudo_count: float = 0.0,
) -> LatentClassFit:
    """Fit ``P_ij=sum_h w_h a_hi b_hj`` by multinomial EM.

    This is the observed-domain likelihood model naturally induced by a hidden
    boundary state of cardinality ``rank_bound``. It enforces nonnegative rank
    rather than only truncating ordinary singular values.
    """

    c = _validate_count_matrix(counts)
    rows, columns = c.shape
    rank = int(rank_bound)
    if rank < 1 or rank > min(rows, columns):
        raise ValueError("rank_bound must lie between one and min(matrix shape)")
    if restarts < 1 or max_iterations < 1:
        raise ValueError("restarts and max_iterations must be positive")
    if tolerance <= 0 or component_floor <= 0 or pseudo_count < 0:
        raise ValueError("tolerances must be positive and pseudo_count nonnegative")
    total = float(c.sum())
    rng = np.random.default_rng(seed)
    best: LatentClassFit | None = None

    for restart in range(restarts):
        if restart == 0:
            row_margin = c.sum(axis=1) + component_floor
            row_margin /= row_margin.sum()
            column_margin = c.sum(axis=0) + component_floor
            column_margin /= column_margin.sum()
            weights = np.full(rank, 1.0 / rank, dtype=float)
            left = np.vstack(
                [normalize_probability_vector(row_margin + 0.03 * rng.random(rows)) for _ in range(rank)]
            )
            right = np.vstack(
                [normalize_probability_vector(column_margin + 0.03 * rng.random(columns)) for _ in range(rank)]
            )
        else:
            weights = rng.dirichlet(np.ones(rank))
            left = rng.dirichlet(np.ones(rows), size=rank)
            right = rng.dirichlet(np.ones(columns), size=rank)

        previous = float("-inf")
        converged = False
        iteration = 0
        for iteration in range(1, max_iterations + 1):
            components = weights[:, None, None] * left[:, :, None] * right[:, None, :]
            fitted = components.sum(axis=0)
            fitted = np.maximum(fitted, component_floor)
            responsibilities = components / fitted[None, :, :]
            expected = responsibilities * c[None, :, :]
            component_totals = expected.sum(axis=(1, 2))
            if pseudo_count:
                component_totals += pseudo_count
            collapsed = component_totals <= component_floor
            if np.any(collapsed):
                for index in np.flatnonzero(collapsed):
                    expected[index] = component_floor
                    expected[index] += rng.random((rows, columns)) * component_floor
                component_totals = expected.sum(axis=(1, 2))
            weights = component_totals / component_totals.sum()
            left = expected.sum(axis=2) + pseudo_count / max(rows, 1)
            right = expected.sum(axis=1) + pseudo_count / max(columns, 1)
            left /= left.sum(axis=1, keepdims=True)
            right /= right.sum(axis=1, keepdims=True)

            probabilities = np.einsum("h,hi,hj->ij", weights, left, right, optimize=True)
            log_likelihood = _multinomial_log_likelihood(c.ravel(), probabilities.ravel())
            if iteration > 1 and abs(log_likelihood - previous) <= tolerance * (
                1.0 + abs(previous)
            ):
                converged = True
                break
            previous = log_likelihood

        probabilities = np.einsum("h,hi,hj->ij", weights, left, right, optimize=True)
        probabilities /= probabilities.sum()
        log_likelihood = _multinomial_log_likelihood(c.ravel(), probabilities.ravel())
        empirical = c / total
        positive = c > 0
        deviance = 2.0 * float(
            np.sum(c[positive] * np.log(empirical[positive] / probabilities[positive]))
        )
        parameter_count = _latent_class_parameters(rows, columns, rank)
        spectral_norm = float(np.linalg.norm(probabilities, 2))
        numerical_rank = int(
            np.linalg.matrix_rank(probabilities, tol=1e-10 * spectral_norm)
        )
        fit = LatentClassFit(
            rank_bound=rank,
            probabilities=probabilities,
            weights=weights,
            left_components=left,
            right_components=right,
            log_likelihood=log_likelihood,
            deviance=max(0.0, deviance),
            iterations=iteration,
            converged=converged,
            restart=restart,
            restarts=restarts,
            effective_parameter_count=parameter_count,
            aic=2.0 * parameter_count - 2.0 * log_likelihood,
            bic=log(total) * parameter_count - 2.0 * log_likelihood,
            sample_size=total,
            numerical_rank=numerical_rank,
        )
        if best is None or fit.log_likelihood > best.log_likelihood:
            best = fit
    assert best is not None
    return best


def rank_tail_frobenius(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    rank_bound: int,
    *,
    relative: bool = False,
) -> float:
    """Frobenius distance to ordinary matrix rank at most ``rank_bound``."""

    source = np.asarray(matrix, dtype=float)
    if source.ndim != 2 or not np.all(np.isfinite(source)):
        raise ValueError("matrix must be finite and two-dimensional")
    rank = int(rank_bound)
    if rank < 0:
        raise ValueError("rank_bound must be nonnegative")
    singular = np.linalg.svd(source, compute_uv=False)
    tail = float(np.sqrt(np.sum(singular[rank:] ** 2)))
    if not relative:
        return tail
    norm = float(np.linalg.norm(source, ord="fro"))
    return 0.0 if norm == 0 else tail / norm


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return float("nan")
    if statistic < 0 or not isfinite(statistic):
        return 0.0 if statistic == float("inf") else float("nan")
    value = mp.gammainc(
        mp.mpf(degrees_of_freedom) / 2,
        mp.mpf(statistic) / 2,
        mp.inf,
        regularized=True,
    )
    return float(value)


def rank_wald_test(
    counts: Sequence[Sequence[float]] | np.ndarray,
    rank_bound: int,
    *,
    fit: LatentClassFit | None = None,
    restarts: int = 8,
    seed: int = 0,
    rcond: float = 1e-10,
) -> RankWaldResult:
    """Covariance-weighted normal-space residual for a rank constraint.

    The null fit is the nonnegative-rank MLE. The ordinary rank-manifold normal
    space is spanned by outer products of singular vectors beyond the rank
    bound. Multinomial covariance is projected into that space and
    pseudoinverted. The chi-square reference is explicitly asymptotic.
    """

    c = _validate_count_matrix(counts)
    rank = int(rank_bound)
    fitted = fit or fit_nonnegative_rank(c, rank, restarts=restarts, seed=seed)
    if fitted.rank_bound != rank or fitted.probabilities.shape != c.shape:
        raise ValueError("fit does not match counts or rank_bound")
    total = float(c.sum())
    empirical = c / total
    u, singular, vt = np.linalg.svd(fitted.probabilities, full_matrices=True)
    fitted_rank = int(np.count_nonzero(singular > rcond * singular[0])) if singular[0] else 0
    effective_rank = min(rank, fitted_rank)
    u_perp = u[:, effective_rank:]
    v_perp = vt.T[:, effective_rank:]
    basis_vectors = [
        np.outer(u_perp[:, i], v_perp[:, j]).ravel(order="C")
        for i in range(u_perp.shape[1])
        for j in range(v_perp.shape[1])
    ]
    if not basis_vectors:
        return RankWaldResult(
            statistic=0.0,
            degrees_of_freedom=0,
            p_value=float("nan"),
            residual_norm=0.0,
            covariance_rank=0,
            covariance_condition_number=1.0,
            rank_bound=rank,
            fitted_rank=fitted_rank,
            applicable=False,
            warning="rank bound saturates the matrix dimensions",
        )
    basis = np.column_stack(basis_vectors)
    residual_coordinates = basis.T @ (empirical - fitted.probabilities).ravel(order="C")
    covariance = multinomial_covariance(fitted.probabilities.ravel(), int(round(total)))
    projected = basis.T @ covariance @ basis
    projected = (projected + projected.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(projected)
    cutoff = rcond * eigenvalues[-1] if eigenvalues.size and eigenvalues[-1] > 0 else 0.0
    positive = eigenvalues[eigenvalues > cutoff]
    covariance_rank = int(positive.size)
    condition = (
        float(positive[-1] / positive[0])
        if positive.size == projected.shape[0]
        else float("inf")
    )
    inverse = np.linalg.pinv(projected, rcond=rcond)
    statistic = max(0.0, float(residual_coordinates @ inverse @ residual_coordinates))
    minimum_expected = float(total * fitted.probabilities.min())
    warning: str | None = None
    applicable = covariance_rank > 0
    if minimum_expected < 1.0:
        warning = "asymptotic chi-square calibration is fragile: a fitted expected cell is below one"
    if not fitted.converged:
        warning = "null likelihood fit did not reach the requested convergence tolerance"
    if covariance_rank < projected.shape[0]:
        rank_warning = "projected multinomial covariance is rank deficient"
        warning = rank_warning if warning is None else f"{warning}; {rank_warning}"
    return RankWaldResult(
        statistic=statistic,
        degrees_of_freedom=covariance_rank,
        p_value=_chi_square_survival(statistic, covariance_rank),
        residual_norm=float(np.linalg.norm(residual_coordinates)),
        covariance_rank=covariance_rank,
        covariance_condition_number=condition,
        rank_bound=rank,
        fitted_rank=fitted_rank,
        applicable=applicable,
        warning=warning,
    )


def flatten_count_tensor(
    count_tensor: np.ndarray | Sequence[float],
    left_coordinates: Sequence[int],
) -> np.ndarray:
    """Flatten a count tensor with selected coordinates on matrix rows."""

    tensor = np.asarray(count_tensor, dtype=float)
    if tensor.ndim < 2 or any(size <= 0 for size in tensor.shape):
        raise ValueError("count_tensor must have at least two nonempty dimensions")
    if not np.all(np.isfinite(tensor)) or np.any(tensor < 0):
        raise ValueError("count_tensor must be finite and nonnegative")
    left = tuple(int(index) for index in left_coordinates)
    if not left or len(left) == tensor.ndim or len(set(left)) != len(left):
        raise ValueError("left_coordinates must be a unique nontrivial subset")
    if any(index < 0 or index >= tensor.ndim for index in left):
        raise ValueError("left coordinate out of range")
    right = tuple(index for index in range(tensor.ndim) if index not in left)
    reordered = np.transpose(tensor, left + right)
    rows = int(np.prod([tensor.shape[index] for index in left], dtype=int))
    columns = int(np.prod([tensor.shape[index] for index in right], dtype=int))
    return reordered.reshape(rows, columns)


def score_quartet_splits(
    count_tensor: np.ndarray | Sequence[float],
    *,
    rank_bound: int,
    splits: Sequence[Sequence[int]] = ((0, 1), (0, 2), (0, 3)),
    restarts: int = 8,
    seed: int = 0,
    max_iterations: int = 5_000,
    tolerance: float = 1e-9,
) -> tuple[QuartetSplitScore, ...]:
    """Score predeclared quartet splits without inspecting a reference tree."""

    tensor = np.asarray(count_tensor, dtype=float)
    if tensor.ndim != 4:
        raise ValueError("quartet scoring requires a four-dimensional count tensor")
    if float(tensor.sum()) <= 0:
        raise ValueError("count_tensor must contain positive total mass")
    results: list[QuartetSplitScore] = []
    for offset, raw_split in enumerate(splits):
        split = tuple(int(index) for index in raw_split)
        if len(split) != 2 or len(set(split)) != 2:
            raise ValueError("each quartet split must contain two distinct coordinates")
        complement = tuple(index for index in range(4) if index not in split)
        if len(complement) != 2:
            raise ValueError("quartet split coordinate out of range")
        matrix = flatten_count_tensor(tensor, split)
        empirical = matrix / matrix.sum()
        fit = fit_nonnegative_rank(
            matrix,
            rank_bound,
            restarts=restarts,
            seed=seed + 104729 * offset,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        wald = rank_wald_test(matrix, rank_bound, fit=fit)
        raw_tail = rank_tail_frobenius(empirical, rank_bound)
        results.append(
            QuartetSplitScore(
                split=split,
                complement=complement,
                matrix_shape=matrix.shape,
                sample_size=float(matrix.sum()),
                raw_tail_frobenius=raw_tail,
                relative_tail_frobenius=rank_tail_frobenius(empirical, rank_bound, relative=True),
                likelihood_deviance=fit.deviance,
                deviance_per_site=fit.deviance / float(matrix.sum()),
                log_likelihood=fit.log_likelihood,
                wald_statistic=wald.statistic,
                wald_degrees_of_freedom=wald.degrees_of_freedom,
                wald_p_value=wald.p_value,
                fit_converged=fit.converged,
                fit_iterations=fit.iterations,
                fitted_numerical_rank=fit.numerical_rank,
            )
        )
    return tuple(results)


def split_score_minimizers(
    scores: Sequence[QuartetSplitScore],
    *,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-10,
) -> Mapping[str, tuple[tuple[int, int], ...]]:
    """Return every tied minimizer for each lower-is-better diagnostic.

    Reporting the complete minimizer set prevents insertion-order bias when a
    rank-destroying observation or extreme regularization makes two or more
    quartet splits exactly indistinguishable.
    """

    if not scores:
        raise ValueError("at least one split score is required")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("score tolerances must be nonnegative")
    attributes = (
        "raw_tail_frobenius",
        "relative_tail_frobenius",
        "likelihood_deviance",
        "deviance_per_site",
        "wald_statistic",
    )
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for attribute in attributes:
        values = np.asarray([getattr(score, attribute) for score in scores], dtype=float)
        minimum = float(values.min())
        tolerance = max(
            absolute_tolerance,
            relative_tolerance * max(1.0, abs(minimum)),
        )
        result[attribute] = tuple(
            score.split
            for score, value in zip(scores, values, strict=True)
            if float(value) <= minimum + tolerance
        )
    return result


def split_score_winners(
    scores: Sequence[QuartetSplitScore],
    *,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-10,
) -> Mapping[str, tuple[int, int]]:
    """Return unique minimizers, rejecting tied diagnostics explicitly."""

    minimizers = split_score_minimizers(
        scores,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    tied = {name: values for name, values in minimizers.items() if len(values) != 1}
    if tied:
        details = ", ".join(f"{name}={values}" for name, values in tied.items())
        raise ValueError(f"quartet split score has no unique winner: {details}")
    return {name: values[0] for name, values in minimizers.items()}
