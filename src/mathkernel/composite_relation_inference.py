# =============================================================================
# MathKernel - Composite inference for observable relation subspaces
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Direction-agnostic tests, nuisance geometry, and invariant uncertainty.

This module is the composite-inference layer above
:mod:`mathkernel.information_geometry`.  It keeps the finite-dimensional,
fail-closed conventions of MathKernel while adding five operations that are
not supplied by one-direction testing bounds alone:

* a single score-norm test for an entire visible relation subspace;
* a conservative finite-sample guarantee with explicit relation dimension;
* Fisher-orthogonal removal of nuisance tangent directions;
* confidence regions for eigenspaces rather than unstable eigenvectors; and
* dependence-aware/studentized uncertainty and sandwich score tests.

For a :class:`~mathkernel.information_geometry.SimplexTangentGeometry`, let
``lambda_1,...,lambda_r`` be its positive generalized information eigenvalues.
The corresponding observed score columns, divided by ``sqrt(lambda_k)``, are
centered and have identity covariance under the nominal observed law.  The
composite statistic

``T_N = N * ||mean(Z_1,...,Z_N)||_2**2``

therefore has the usual chi-square limit under the null.  Along a local
latent-Fisher-unit perturbation of radius ``epsilon`` its noncentrality is
``N * epsilon**2 * directional_information``.  A bounded-score argument also
gives one direction-agnostic finite-sample guarantee over the complete unit
sphere:

``N >= 8*r*B**2*log(2*r/delta) / (epsilon**2*lambda_min)``.

The bound is intentionally conservative, but unlike the pointwise bound in
Phase 8 it belongs to one declared composite norm test and exposes the cost of
relation-space dimension and weak conditioning explicitly.

Nuisance adjustment uses Schur complements of latent and observed Fisher Gram
matrices.  A zero nuisance-adjusted eigenvalue is an exact confounding
certificate: the target direction is visible in isolation but indistinguishable
from a nuisance perturbation after observation.

All chi-square tail calculations are imported lazily from SciPy.  Geometry,
statistics, conservative bounds, and bootstrap resampling continue to work
without SciPy; only p-values and noncentral-power inversion require it.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, isfinite, log, sqrt
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np

from .information_geometry import (
    BootstrapMethod,
    SimplexTangentGeometry,
    _channel_rows,
    _matrix,
    _probability_vector,
    _resample_indices,
    _symmetrize,
    score_subspace_geometry,
)
from .relation_subspace import newey_west_long_run_covariance, relation_subspace_from_partition


ArrayLike = Sequence[float] | np.ndarray
CovarianceMode = Literal["nominal", "empirical", "hac"]


def _generalized_spectrum(
    latent_metric: np.ndarray,
    observed_metric: np.ndarray,
    *,
    rank_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Generalized spectrum with latent-metric-normalized directions."""

    latent = _symmetrize(np.asarray(latent_metric, dtype=float))
    observed = _symmetrize(np.asarray(observed_metric, dtype=float))
    if latent.shape != observed.shape or latent.ndim != 2 or latent.shape[0] != latent.shape[1]:
        raise ValueError("latent and observed metrics must be equally sized square matrices")
    latent_values, latent_vectors = np.linalg.eigh(latent)
    scale = max(1.0, float(np.max(np.abs(latent_values))))
    active = latent_values > rank_tolerance * scale
    latent_rank = int(np.count_nonzero(active))
    if latent_rank == 0:
        raise ValueError("declared target subspace has zero efficient Fisher rank")
    whitening = latent_vectors[:, active] / np.sqrt(latent_values[active])[None, :]
    whitened = _symmetrize(whitening.T @ observed @ whitening)
    values, vectors = np.linalg.eigh(whitened)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    residual = max(0.0, -float(np.min(values)), float(np.max(values)) - 1.0)
    if residual > 1e3 * rank_tolerance:
        raise RuntimeError("nuisance-adjusted geometry violates Fisher data processing")
    values = np.clip(values, 0.0, 1.0)
    directions = whitening @ vectors
    observed_rank = int(
        np.count_nonzero(values > rank_tolerance * max(1.0, float(np.max(values))))
    )
    return values, directions, latent_rank, observed_rank, residual


def _chi_square_survival(value: float, degrees_of_freedom: int) -> float:
    try:
        from scipy.stats import chi2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("SciPy is required for chi-square p-values") from exc
    return float(chi2.sf(float(value), int(degrees_of_freedom)))


def _noncentral_power(
    sample_size: int,
    per_sample_noncentrality: float,
    degrees_of_freedom: int,
    alpha: float,
) -> float:
    try:
        from scipy.stats import chi2, ncx2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("SciPy is required for noncentral chi-square power") from exc
    critical = float(chi2.ppf(1.0 - alpha, degrees_of_freedom))
    return float(ncx2.sf(critical, degrees_of_freedom, sample_size * per_sample_noncentrality))


@dataclass(frozen=True, slots=True)
class CompositeScoreGeometry:
    """Whitened observed score system for one relation subspace."""

    observed_law: np.ndarray
    standardized_scores: np.ndarray
    active_output_indices: tuple[int, ...]
    information_eigenvalues: np.ndarray
    visible_information_eigenvalues: np.ndarray
    visible_generalized_directions: np.ndarray
    relation_rank: int
    visible_rank: int
    minimum_positive_information: float
    maximum_absolute_score: float
    maximum_score_norm: float
    covariance_residual: float


@dataclass(frozen=True, slots=True)
class CompositeScoreTestResult:
    """One direction-agnostic score-norm test result."""

    statistic: float
    degrees_of_freedom: int
    p_value: float | None
    reject: bool | None
    alpha: float | None
    sample_size: int
    score_mean: np.ndarray
    covariance: np.ndarray
    covariance_rank: int
    covariance_mode: str
    max_lag: int | None


@dataclass(frozen=True, slots=True)
class CompositeDetectionBounds:
    """Dimension-aware guarantees for the declared composite norm test."""

    epsilon: float
    max_error: float
    relation_dimension: int
    weakest_information_retention: float
    weakest_visibility: float
    per_sample_worst_direction_noncentrality: float
    maximum_absolute_standardized_score: float
    chi_square_necessary_samples: int | float | None
    asymptotic_composite_samples: int | float | None
    conservative_composite_sufficient_samples: int | float | None
    exact_blind_direction: bool
    resolution_status: str = "resolved"
    asymptotic_search_status: str = "target_attained"
    requested_power: float = 0.90
    power_at_search_bound: float | None = None
    sample_search_limit: int = 2**60
    rank_certificate_scope: str | None = None


@dataclass(frozen=True, slots=True)
class NuisanceAdjustedGeometry:
    """Target Fisher geometry after projection away from nuisance tangents."""

    latent_law: np.ndarray
    observed_law: np.ndarray
    channel: np.ndarray
    target_tangent_frame: np.ndarray
    nuisance_tangent_frame: np.ndarray
    pushed_target_frame: np.ndarray
    pushed_nuisance_frame: np.ndarray
    latent_efficient_target_frame: np.ndarray
    observed_efficient_target_frame: np.ndarray
    latent_efficient_metric: np.ndarray
    observed_efficient_metric: np.ndarray
    generalized_information_eigenvalues: np.ndarray
    transfer_singular_values: np.ndarray
    generalized_target_directions: np.ndarray
    latent_efficient_rank: int
    observed_efficient_rank: int
    minimum_information_retention: float
    information_condition_number: float
    data_processing_residual: float

    @property
    def fully_identifiable_after_nuisance(self) -> bool:
        return self.observed_efficient_rank == self.latent_efficient_rank


@dataclass(frozen=True, slots=True)
class RobustCompositeDetectionBound:
    """Score-mean separation bound under a norm-bounded model drift."""

    nominal_separation: float
    misspecification_radius: float
    effective_separation: float
    relation_dimension: int
    maximum_absolute_standardized_score: float
    max_error: float
    conservative_sufficient_samples: int | float
    robustly_separated: bool


@dataclass(frozen=True, slots=True)
class EigenspaceClusterRegion:
    """Bootstrap confidence radius for one invariant eigenvalue cluster."""

    indices: tuple[int, ...]
    point_eigenvalues: np.ndarray
    principal_sine_radius: float
    frobenius_projector_radius: float
    median_principal_sine_distance: float
    median_frobenius_projector_distance: float
    principal_sine_distances: np.ndarray
    frobenius_projector_distances: np.ndarray


@dataclass(frozen=True, slots=True)
class EigenspaceBootstrapUncertainty:
    confidence_level: float
    method: str
    sample_size: int
    requested_replicates: int
    successful_replicates: int
    failed_replicates: int
    block_length: int | None
    clusters: tuple[EigenspaceClusterRegion, ...]


@dataclass(frozen=True, slots=True)
class StudentizedSpectrumBootstrap:
    """Studentized intervals for ordered information eigenvalues."""

    point_information_eigenvalues: np.ndarray
    point_standard_errors: np.ndarray
    lower_information_eigenvalues: np.ndarray
    upper_information_eigenvalues: np.ndarray
    bootstrap_information_eigenvalues: np.ndarray
    studentized_statistics: np.ndarray
    confidence_level: float
    method: str
    sample_size: int
    requested_replicates: int
    successful_replicates: int
    failed_replicates: int
    jackknife_groups: int
    block_length: int | None


def composite_score_geometry(
    geometry: SimplexTangentGeometry,
    *,
    rank_tolerance: float = 1e-11,
) -> CompositeScoreGeometry:
    """Whiten positive-information generalized observed scores.

    Under ``geometry.observed_law`` the returned score columns have zero mean
    and identity covariance, up to numerical roundoff.
    """

    values = np.asarray(geometry.generalized_information_eigenvalues, dtype=float)
    visible = values > rank_tolerance * max(1.0, float(np.max(values)))
    positive = values[visible]
    directions = geometry.generalized_coefficient_directions[:, visible]
    tangents = geometry.pushed_tangent_frame @ directions
    q = np.asarray(geometry.observed_law, dtype=float)
    active = np.asarray(geometry.active_output_indices, dtype=np.int64)
    scores = np.zeros((q.size, positive.size), dtype=float)
    if positive.size:
        scores[active] = tangents[active] / q[active, None]
        scores[:, :] /= np.sqrt(positive)[None, :]
    mean = q @ scores
    covariance = _symmetrize(scores.T @ (q[:, None] * scores))
    residual = float(
        max(
            np.max(np.abs(mean)) if mean.size else 0.0,
            np.max(np.abs(covariance - np.eye(positive.size))) if positive.size else 0.0,
        )
    )
    return CompositeScoreGeometry(
        observed_law=q.copy(),
        standardized_scores=scores,
        active_output_indices=tuple(int(i) for i in active),
        information_eigenvalues=values.copy(),
        visible_information_eigenvalues=positive.copy(),
        visible_generalized_directions=directions.copy(),
        relation_rank=geometry.latent_rank,
        visible_rank=int(positive.size),
        minimum_positive_information=float(np.min(positive)) if positive.size else 0.0,
        maximum_absolute_score=float(np.max(np.abs(scores))) if scores.size else 0.0,
        maximum_score_norm=float(np.max(np.linalg.norm(scores, axis=1))) if scores.size else 0.0,
        covariance_residual=residual,
    )


def nuisance_composite_score_geometry(
    geometry: NuisanceAdjustedGeometry,
    *,
    rank_tolerance: float = 1e-11,
) -> CompositeScoreGeometry:
    """Construct standardized efficient scores after nuisance adjustment."""

    values = np.asarray(geometry.generalized_information_eigenvalues, dtype=float)
    visible = values > rank_tolerance * max(1.0, float(np.max(values)))
    positive = values[visible]
    directions = geometry.generalized_target_directions[:, visible]
    tangents = geometry.observed_efficient_target_frame @ directions
    q = geometry.observed_law
    active = np.flatnonzero(q > 1e-15)
    scores = np.zeros((q.size, positive.size), dtype=float)
    if positive.size:
        scores[active] = tangents[active] / q[active, None]
        scores /= np.sqrt(positive)[None, :]
    mean = q @ scores
    covariance = _symmetrize(scores.T @ (q[:, None] * scores))
    residual = float(
        max(
            np.max(np.abs(mean)) if mean.size else 0.0,
            np.max(np.abs(covariance - np.eye(positive.size))) if positive.size else 0.0,
        )
    )
    return CompositeScoreGeometry(
        observed_law=q.copy(),
        standardized_scores=scores,
        active_output_indices=tuple(int(i) for i in active),
        information_eigenvalues=values.copy(),
        visible_information_eigenvalues=positive.copy(),
        visible_generalized_directions=directions.copy(),
        relation_rank=geometry.latent_efficient_rank,
        visible_rank=int(positive.size),
        minimum_positive_information=float(np.min(positive)) if positive.size else 0.0,
        maximum_absolute_score=float(np.max(np.abs(scores))) if scores.size else 0.0,
        maximum_score_norm=float(np.max(np.linalg.norm(scores, axis=1))) if scores.size else 0.0,
        covariance_residual=residual,
    )


def _validate_labels(labels: Sequence[int] | np.ndarray, output_states: int) -> np.ndarray:
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("observed_labels must be a nonempty one-dimensional array")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValueError("observed_labels must contain finite integers")
    result = raw.astype(np.int64)
    if np.any(result < 0) or np.any(result >= output_states):
        raise ValueError("observed label lies outside the declared output alphabet")
    return result


def composite_score_test(
    observed_labels: Sequence[int] | np.ndarray,
    geometry: CompositeScoreGeometry,
    *,
    covariance_mode: CovarianceMode = "nominal",
    max_lag: int | None = None,
    alpha: float | None = 0.05,
    rank_tolerance: float = 1e-11,
) -> CompositeScoreTestResult:
    """Apply one direction-agnostic quadratic score test.

    ``nominal`` uses the identity covariance guaranteed by score whitening.
    ``empirical`` uses the ordinary sample covariance.  ``hac`` uses the
    Bartlett/Newey-West long-run covariance and is appropriate for an ordered
    dependent record sequence.  The p-value is asymptotic in all three modes.
    """

    labels = _validate_labels(observed_labels, geometry.observed_law.size)
    if geometry.visible_rank == 0:
        raise ValueError("composite test has no observable relation directions")
    rows = geometry.standardized_scores[labels]
    n = labels.size
    mean = rows.mean(axis=0)
    if covariance_mode == "nominal":
        covariance = np.eye(geometry.visible_rank)
        lag = None
    elif covariance_mode == "empirical":
        centered = rows - mean
        covariance = _symmetrize(centered.T @ centered / n)
        lag = None
    elif covariance_mode == "hac":
        lag = int(max_lag) if max_lag is not None else max(1, int(round(n ** (1.0 / 3.0))))
        covariance = newey_west_long_run_covariance(rows, max_lag=lag, center=True)
    else:
        raise ValueError("covariance_mode must be 'nominal', 'empirical', or 'hac'")
    values = np.linalg.eigvalsh(_symmetrize(covariance))
    scale = max(1.0, float(np.max(np.abs(values))))
    rank = int(np.count_nonzero(values > rank_tolerance * scale))
    if rank == 0:
        raise ValueError("estimated score covariance has zero rank")
    inverse = np.linalg.pinv(covariance, rcond=rank_tolerance)
    statistic = float(n * mean @ inverse @ mean)
    if alpha is None:
        p_value = None
        reject = None
        alpha_value = None
    else:
        alpha_value = float(alpha)
        if not 0 < alpha_value < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        p_value = _chi_square_survival(statistic, rank)
        reject = bool(p_value < alpha_value)
    return CompositeScoreTestResult(
        statistic=statistic,
        degrees_of_freedom=rank,
        p_value=p_value,
        reject=reject,
        alpha=alpha_value,
        sample_size=n,
        score_mean=mean,
        covariance=covariance,
        covariance_rank=rank,
        covariance_mode=covariance_mode,
        max_lag=lag,
    )


def composite_detection_bounds(
    geometry: SimplexTangentGeometry,
    epsilon: float,
    *,
    max_error: float = 0.10,
    alpha: float = 0.05,
    target_power: float = 0.90,
    tolerance: float = 1e-14,
    max_samples: int = 2**60,
) -> CompositeDetectionBounds:
    """Bounds for one score-norm test over all latent-Fisher-unit directions.

    A capped or unresolved asymptotic search returns None and a status, never
    the cap as if it attained the target. Infinity is reserved for certified
    rank loss. Numerical rank thresholds are not exact blindness certificates.
    """

    eps = float(epsilon)
    delta = float(max_error)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    if eps > geometry.uniform_latent_positivity_radius + 1e-13:
        raise ValueError("epsilon exceeds the subspace-uniform positivity radius")
    if not 0 < delta < 0.5:
        raise ValueError("max_error must lie in (0, 1/2)")
    if not 0 < alpha < 1 or not 0 < target_power < 1:
        raise ValueError("alpha and target_power must lie in (0, 1)")
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    from math import log1p
    from .information_geometry import exact_observation_rank_loss
    composite = composite_score_geometry(geometry, rank_tolerance=tolerance)
    weakest = float(geometry.generalized_information_eigenvalues[-1])
    deficient = composite.visible_rank < geometry.latent_rank
    rank_loss = exact_observation_rank_loss(geometry) if deficient else False
    blind = rank_loss is True
    per_sample = eps * eps * max(0.0, weakest)
    unresolved = (deficient or per_sample <= 0) and not blind
    required_chi = 4.0 * (1.0 - 2.0 * delta) ** 2
    necessary = (inf if blind else (None if per_sample <= 0 else
                 int(ceil(log1p(required_chi) / log1p(per_sample)))))
    achieved = None
    if blind:
        asymptotic, conservative = inf, inf
        search_status = "exact_rank_loss"
    elif unresolved or per_sample <= 0:
        # Infinity is not an impossibility certificate when numerical rank or
        # underflow prevents us from resolving a positive-information direction.
        asymptotic, conservative = None, None
        search_status = "numerically_unresolved"
    else:
        low, high = 1, 1
        achieved = _noncentral_power(high, per_sample, composite.visible_rank, alpha)
        while achieved < target_power and high < max_samples:
            high = min(high * 2, max_samples)
            achieved = _noncentral_power(high, per_sample, composite.visible_rank, alpha)
        if not isfinite(achieved):
            asymptotic, search_status = None, "numerically_unresolved"
        elif achieved < target_power:
            asymptotic, search_status = None, "search_limit_reached"
        else:
            while low < high:
                mid = (low + high) // 2
                power = _noncentral_power(mid, per_sample, composite.visible_rank, alpha)
                if not isfinite(power):
                    raise ValueError("noncentral power calculation is not finite")
                if power >= target_power:
                    high = mid
                else:
                    low = mid + 1
            asymptotic = int(low)
            achieved = _noncentral_power(asymptotic, per_sample, composite.visible_rank, alpha)
            search_status = "target_attained"
        b = composite.maximum_absolute_score
        conservative = int(ceil(8.0 * composite.visible_rank * b * b *
                                log(2.0 * composite.visible_rank / delta) / per_sample))
    return CompositeDetectionBounds(
        epsilon=eps,
        max_error=delta,
        relation_dimension=composite.visible_rank,
        weakest_information_retention=weakest,
        weakest_visibility=sqrt(max(0.0, weakest)),
        per_sample_worst_direction_noncentrality=per_sample,
        maximum_absolute_standardized_score=composite.maximum_absolute_score,
        chi_square_necessary_samples=necessary,
        asymptotic_composite_samples=asymptotic,
        conservative_composite_sufficient_samples=conservative,
        exact_blind_direction=blind,
        resolution_status="exact_rank_loss" if blind else ("numerically_unresolved" if unresolved else "resolved"),
        asymptotic_search_status=search_status,
        requested_power=target_power,
        power_at_search_bound=achieved,
        sample_search_limit=max_samples,
        rank_certificate_scope=("exact rational rank of declared binary-float channel and tangent frame" if blind else None),
    )


def nuisance_adjusted_tangent_geometry(
    latent_law: ArrayLike,
    channel: ArrayLike,
    target_tangent_frame: ArrayLike,
    nuisance_tangent_frame: ArrayLike,
    *,
    rank_tolerance: float = 1e-11,
) -> NuisanceAdjustedGeometry:
    """Compute efficient target information after removing nuisance tangents."""

    p = _probability_vector(latent_law)
    rows = _channel_rows(channel)
    if rows.shape[0] != p.size:
        raise ValueError("channel latent alphabet does not match the law")
    target = _matrix(target_tangent_frame, rows=p.size, name="target_tangent_frame")
    nuisance = _matrix(nuisance_tangent_frame, rows=p.size, name="nuisance_tangent_frame")
    if not np.allclose(target.sum(axis=0), 0.0, atol=rank_tolerance, rtol=0.0):
        raise ValueError("every target tangent must sum to zero")
    if not np.allclose(nuisance.sum(axis=0), 0.0, atol=rank_tolerance, rtol=0.0):
        raise ValueError("every nuisance tangent must sum to zero")
    q = p @ rows
    active = q > 1e-15
    pushed_target = rows.T @ target
    pushed_nuisance = rows.T @ nuisance

    def efficient(
        target_frame: np.ndarray,
        nuisance_frame: np.ndarray,
        weights: np.ndarray,
        support: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if support is not None:
            t = target_frame[support]
            n = nuisance_frame[support]
            w = weights[support]
        else:
            t, n, w = target_frame, nuisance_frame, weights
        wt = t / w[:, None]
        wn = n / w[:, None]
        gtt = _symmetrize(t.T @ wt)
        gnn = _symmetrize(n.T @ wn)
        gtn = t.T @ wn
        coefficients = np.linalg.pinv(gnn, rcond=rank_tolerance) @ gtn.T
        residual = target_frame - nuisance_frame @ coefficients
        efficient_metric = _symmetrize(gtt - gtn @ np.linalg.pinv(gnn, rcond=rank_tolerance) @ gtn.T)
        return residual, efficient_metric

    latent_residual, latent_metric = efficient(target, nuisance, p)
    observed_residual, observed_metric = efficient(
        pushed_target, pushed_nuisance, q, support=active
    )
    values, directions, latent_rank, observed_rank, residual = _generalized_spectrum(
        latent_metric, observed_metric, rank_tolerance=rank_tolerance
    )
    positive = values[values > rank_tolerance * max(1.0, float(np.max(values)))]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else inf
    return NuisanceAdjustedGeometry(
        latent_law=p,
        observed_law=q,
        channel=rows,
        target_tangent_frame=target,
        nuisance_tangent_frame=nuisance,
        pushed_target_frame=pushed_target,
        pushed_nuisance_frame=pushed_nuisance,
        latent_efficient_target_frame=latent_residual,
        observed_efficient_target_frame=observed_residual,
        latent_efficient_metric=latent_metric,
        observed_efficient_metric=observed_metric,
        generalized_information_eigenvalues=values,
        transfer_singular_values=np.sqrt(values),
        generalized_target_directions=directions,
        latent_efficient_rank=latent_rank,
        observed_efficient_rank=observed_rank,
        minimum_information_retention=float(values[-1]),
        information_condition_number=condition,
        data_processing_residual=residual,
    )


def nuisance_adjusted_score_geometry(
    latent_law: ArrayLike,
    channel: ArrayLike,
    target_scores: ArrayLike,
    nuisance_scores: ArrayLike,
    *,
    center: bool = True,
    rank_tolerance: float = 1e-11,
) -> NuisanceAdjustedGeometry:
    """Score-coordinate convenience wrapper for nuisance adjustment."""

    p = _probability_vector(latent_law)
    target = _matrix(target_scores, rows=p.size, name="target_scores").copy()
    nuisance = _matrix(nuisance_scores, rows=p.size, name="nuisance_scores").copy()
    if center:
        target -= np.outer(np.ones(p.size), p @ target)
        nuisance -= np.outer(np.ones(p.size), p @ nuisance)
    return nuisance_adjusted_tangent_geometry(
        p,
        channel,
        p[:, None] * target,
        p[:, None] * nuisance,
        rank_tolerance=rank_tolerance,
    )


def robust_composite_detection_bound(
    geometry: CompositeScoreGeometry,
    nominal_separation: float,
    misspecification_radius: float,
    *,
    max_error: float = 0.10,
) -> RobustCompositeDetectionBound:
    """Conservative score-mean guarantee for two uncertainty balls.

    The null and alternative score means may each move by at most
    ``misspecification_radius`` in Euclidean norm.  Their worst-case remaining
    separation is therefore ``max(0, nominal_separation - 2*radius)``.
    """

    separation = float(nominal_separation)
    radius = float(misspecification_radius)
    delta = float(max_error)
    if not isfinite(separation) or separation < 0:
        raise ValueError("nominal_separation must be finite and nonnegative")
    if not isfinite(radius) or radius < 0:
        raise ValueError("misspecification_radius must be finite and nonnegative")
    if not 0 < delta < 0.5:
        raise ValueError("max_error must lie in (0, 1/2)")
    effective = max(0.0, separation - 2.0 * radius)
    if effective <= 1e-14:
        effective = 0.0
    if effective == 0.0 or geometry.visible_rank == 0:
        samples: int | float = inf
    else:
        samples = int(
            ceil(
                8.0
                * geometry.visible_rank
                * geometry.maximum_absolute_score**2
                * log(2.0 * geometry.visible_rank / delta)
                / (effective * effective)
            )
        )
    return RobustCompositeDetectionBound(
        nominal_separation=separation,
        misspecification_radius=radius,
        effective_separation=effective,
        relation_dimension=geometry.visible_rank,
        maximum_absolute_standardized_score=geometry.maximum_absolute_score,
        max_error=delta,
        conservative_sufficient_samples=samples,
        robustly_separated=effective > 0,
    )


def _metric_principal_distances(
    first: np.ndarray,
    second: np.ndarray,
    metric: np.ndarray,
    *,
    tolerance: float,
) -> tuple[float, float]:
    """Largest sine and Frobenius projector distance in a common metric."""

    def orthonormalize(frame: np.ndarray) -> np.ndarray:
        gram = _symmetrize(frame.T @ metric @ frame)
        values, vectors = np.linalg.eigh(gram)
        active = values > tolerance * max(1.0, float(np.max(np.abs(values))))
        if not np.any(active):
            raise ValueError("eigenspace cluster is metric-degenerate")
        return frame @ (vectors[:, active] / np.sqrt(values[active])[None, :])

    q1 = orthonormalize(first)
    q2 = orthonormalize(second)
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("eigenspace clusters have different dimensions")
    singular = np.linalg.svd(q1.T @ metric @ q2, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    sin_squared = np.maximum(0.0, 1.0 - singular * singular)
    return float(np.sqrt(np.max(sin_squared))), float(np.sqrt(2.0 * np.sum(sin_squared)))


def _normalize_clusters(
    clusters: Sequence[Sequence[int]] | None,
    eigenvalues: np.ndarray,
    *,
    relative_gap: float,
) -> tuple[tuple[int, ...], ...]:
    if clusters is not None:
        normalized: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for cluster in clusters:
            current = tuple(sorted(int(index) for index in cluster))
            if not current:
                raise ValueError("eigenspace cluster must be nonempty")
            if current[0] < 0 or current[-1] >= eigenvalues.size:
                raise ValueError("eigenspace cluster index is out of range")
            if any(index in seen for index in current):
                raise ValueError("eigenspace clusters must be disjoint")
            seen.update(current)
            normalized.append(current)
        return tuple(normalized)
    result: list[list[int]] = [[0]]
    for index in range(1, eigenvalues.size):
        scale = max(1.0, abs(float(eigenvalues[index - 1])), abs(float(eigenvalues[index])))
        if abs(float(eigenvalues[index - 1] - eigenvalues[index])) <= relative_gap * scale:
            result[-1].append(index)
        else:
            result.append([index])
    return tuple(tuple(cluster) for cluster in result)


def bootstrap_eigenspace_regions(
    latent_score_rows: ArrayLike,
    observed_labels: Sequence[int] | np.ndarray,
    *,
    clusters: Sequence[Sequence[int]] | None = None,
    relative_gap: float = 0.03,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    method: BootstrapMethod = "iid",
    block_length: int | None = None,
    seed: int | np.random.Generator = 20260906,
    rank_tolerance: float = 1e-11,
) -> EigenspaceBootstrapUncertainty:
    """Bootstrap invariant confidence radii for generalized eigenspaces."""

    scores = _matrix(latent_score_rows, name="latent_score_rows")
    labels = _validate_labels(observed_labels, int(np.max(observed_labels)) + 1)
    if labels.size != scores.shape[0]:
        raise ValueError("observed_labels must have one entry per score row")
    requested = int(n_bootstrap)
    if requested < 20:
        raise ValueError("at least 20 bootstrap replicates are required")
    confidence = float(confidence_level)
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    n = scores.shape[0]
    point = relation_subspace_from_partition(
        np.full(n, 1.0 / n), labels, scores, rank_tolerance=rank_tolerance
    )
    cluster_indices = _normalize_clusters(
        clusters,
        point.generalized_information_eigenvalues,
        relative_gap=relative_gap,
    )
    metric = point.latent_gram
    point_directions = point.generalized_directions
    principal: list[list[float]] = [[] for _ in cluster_indices]
    frobenius: list[list[float]] = [[] for _ in cluster_indices]
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    failures = 0
    for _ in range(requested):
        indices = _resample_indices(
            rng, n, method, block_length=block_length, clusters=None
        )
        try:
            replicate = relation_subspace_from_partition(
                np.full(indices.size, 1.0 / indices.size),
                labels[indices],
                scores[indices],
                rank_tolerance=rank_tolerance,
            )
            if replicate.generalized_directions.shape[1] < point_directions.shape[1]:
                raise ValueError("bootstrap replicate lost relation rank")
            for position, cluster in enumerate(cluster_indices):
                first = point_directions[:, cluster]
                second = replicate.generalized_directions[:, cluster]
                dmax, dfrob = _metric_principal_distances(
                    first, second, metric, tolerance=rank_tolerance
                )
                principal[position].append(dmax)
                frobenius[position].append(dfrob)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
    successful = requested - failures
    if successful < max(20, requested // 2):
        raise RuntimeError("too few successful eigenspace bootstrap replicates")
    regions: list[EigenspaceClusterRegion] = []
    for cluster, dmax_values, dfrob_values in zip(cluster_indices, principal, frobenius):
        dmax_array = np.asarray(dmax_values, dtype=float)
        dfrob_array = np.asarray(dfrob_values, dtype=float)
        regions.append(
            EigenspaceClusterRegion(
                indices=cluster,
                point_eigenvalues=point.generalized_information_eigenvalues[list(cluster)],
                principal_sine_radius=float(np.quantile(dmax_array, confidence)),
                frobenius_projector_radius=float(np.quantile(dfrob_array, confidence)),
                median_principal_sine_distance=float(np.median(dmax_array)),
                median_frobenius_projector_distance=float(np.median(dfrob_array)),
                principal_sine_distances=dmax_array,
                frobenius_projector_distances=dfrob_array,
            )
        )
    return EigenspaceBootstrapUncertainty(
        confidence_level=confidence,
        method=method,
        sample_size=n,
        requested_replicates=requested,
        successful_replicates=successful,
        failed_replicates=failures,
        block_length=block_length if method == "moving_block" else None,
        clusters=tuple(regions),
    )


def estimate_circular_block_length(
    rows: ArrayLike,
    *,
    max_lag: int | None = None,
    minimum: int = 2,
    maximum: int | None = None,
) -> int:
    """Dependence-informed circular-block length heuristic.

    The rule estimates an initial-positive integrated correlation time from the
    trace autocorrelation and returns ``ceil((2*tau**2*n)**(1/3))``.  It is a
    transparent heuristic, not a claim of asymptotically optimal block length.
    """

    matrix = _matrix(rows, name="rows")
    n = matrix.shape[0]
    if n < 8:
        raise ValueError("at least eight ordered rows are required")
    centered = matrix - matrix.mean(axis=0)
    gamma0 = float(np.sum(centered * centered) / n)
    if gamma0 <= 1e-15:
        return int(min(maximum or n, max(minimum, 2)))
    limit = min(n // 4, int(max_lag) if max_lag is not None else max(5, int(round(n ** 0.5))))
    positive_sum = 0.0
    for lag in range(1, limit + 1):
        rho = float(np.sum(centered[:-lag] * centered[lag:]) / ((n - lag) * gamma0))
        if rho <= 0.0:
            break
        positive_sum += rho
    tau = max(1.0, 1.0 + 2.0 * positive_sum)
    length = int(ceil((2.0 * tau * tau * n) ** (1.0 / 3.0)))
    upper = n // 4 if maximum is None else int(maximum)
    return int(min(max(minimum, length), max(minimum, upper)))


def estimate_relation_block_length(
    latent_score_rows: ArrayLike,
    observed_labels: Sequence[int] | np.ndarray,
    *,
    max_lag: int | None = None,
) -> int:
    """Estimate a block length from both latent scores and observed labels."""

    scores = _matrix(latent_score_rows, name="latent_score_rows")
    labels = _validate_labels(observed_labels, int(np.max(observed_labels)) + 1)
    if labels.size != scores.shape[0]:
        raise ValueError("observed_labels must have one entry per score row")
    one_hot = np.eye(int(labels.max()) + 1, dtype=float)[labels]
    return estimate_circular_block_length(
        np.column_stack((scores, one_hot)), max_lag=max_lag
    )


def _spectrum_estimate(
    scores: np.ndarray,
    labels: np.ndarray,
    target_rank: int,
    rank_tolerance: float,
) -> np.ndarray:
    transfer = relation_subspace_from_partition(
        np.full(scores.shape[0], 1.0 / scores.shape[0]),
        labels,
        scores,
        rank_tolerance=rank_tolerance,
    )
    result = np.zeros(target_rank, dtype=float)
    available = min(target_rank, transfer.generalized_information_eigenvalues.size)
    result[:available] = transfer.generalized_information_eigenvalues[:available]
    return result


def _delete_group_standard_error(
    scores: np.ndarray,
    labels: np.ndarray,
    target_rank: int,
    groups: int,
    rank_tolerance: float,
) -> np.ndarray:
    n = scores.shape[0]
    group_count = min(int(groups), max(2, n // max(3, target_rank + 1)))
    if group_count < 2:
        raise ValueError("not enough rows for delete-group jackknife")
    pieces = np.array_split(np.arange(n), group_count)
    estimates: list[np.ndarray] = []
    for piece in pieces:
        keep = np.ones(n, dtype=bool)
        keep[piece] = False
        estimates.append(
            _spectrum_estimate(scores[keep], labels[keep], target_rank, rank_tolerance)
        )
    values = np.vstack(estimates)
    mean = values.mean(axis=0)
    variance = (group_count - 1.0) / group_count * np.sum((values - mean) ** 2, axis=0)
    return np.sqrt(np.maximum(variance, 0.0))


def studentized_bootstrap_relation_spectrum(
    latent_score_rows: ArrayLike,
    observed_labels: Sequence[int] | np.ndarray,
    *,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    method: BootstrapMethod = "iid",
    block_length: int | None = None,
    jackknife_groups: int = 8,
    seed: int | np.random.Generator = 20260906,
    rank_tolerance: float = 1e-11,
) -> StudentizedSpectrumBootstrap:
    """Studentized ordered-spectrum intervals using delete-group standard errors."""

    scores = _matrix(latent_score_rows, name="latent_score_rows")
    labels = _validate_labels(observed_labels, int(np.max(observed_labels)) + 1)
    if labels.size != scores.shape[0]:
        raise ValueError("observed_labels must have one entry per score row")
    requested = int(n_bootstrap)
    if requested < 20:
        raise ValueError("at least 20 bootstrap replicates are required")
    confidence = float(confidence_level)
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must lie in (0, 1)")
    groups = int(jackknife_groups)
    if groups < 3:
        raise ValueError("jackknife_groups must be at least three")
    n = scores.shape[0]
    if method == "moving_block" and block_length is None:
        block_length = estimate_relation_block_length(scores, labels)
    point_transfer = relation_subspace_from_partition(
        np.full(n, 1.0 / n), labels, scores, rank_tolerance=rank_tolerance
    )
    point = point_transfer.generalized_information_eigenvalues.copy()
    target_rank = point_transfer.latent_rank
    point_se = _delete_group_standard_error(
        scores, labels, target_rank, groups, rank_tolerance
    )
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    statistics: list[np.ndarray] = []
    failures = 0
    for _ in range(requested):
        indices = _resample_indices(
            rng, n, method, block_length=block_length, clusters=None
        )
        try:
            estimate = _spectrum_estimate(
                scores[indices], labels[indices], target_rank, rank_tolerance
            )
            se = _delete_group_standard_error(
                scores[indices], labels[indices], target_rank, groups, rank_tolerance
            )
            valid = se > 1e-14
            t = np.zeros(target_rank, dtype=float)
            t[valid] = (estimate[valid] - point[valid]) / se[valid]
            t[~valid] = 0.0
            estimates.append(estimate)
            statistics.append(t)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
    if len(estimates) < max(20, requested // 2):
        raise RuntimeError("too few successful studentized bootstrap replicates")
    values = np.vstack(estimates)
    t_values = np.vstack(statistics)
    alpha = (1.0 - confidence) / 2.0
    lower_t = np.quantile(t_values, alpha, axis=0)
    upper_t = np.quantile(t_values, 1.0 - alpha, axis=0)
    lower = np.clip(point - upper_t * point_se, 0.0, 1.0)
    upper = np.clip(point - lower_t * point_se, 0.0, 1.0)
    swap = lower > upper
    if np.any(swap):
        old = lower.copy()
        lower[swap] = upper[swap]
        upper[swap] = old[swap]
    return StudentizedSpectrumBootstrap(
        point_information_eigenvalues=point,
        point_standard_errors=point_se,
        lower_information_eigenvalues=lower,
        upper_information_eigenvalues=upper,
        bootstrap_information_eigenvalues=values,
        studentized_statistics=t_values,
        confidence_level=confidence,
        method=method,
        sample_size=n,
        requested_replicates=requested,
        successful_replicates=values.shape[0],
        failed_replicates=failures,
        jackknife_groups=groups,
        block_length=block_length if method == "moving_block" else None,
    )


__all__ = [
    "CompositeScoreGeometry",
    "CompositeScoreTestResult",
    "CompositeDetectionBounds",
    "NuisanceAdjustedGeometry",
    "RobustCompositeDetectionBound",
    "EigenspaceClusterRegion",
    "EigenspaceBootstrapUncertainty",
    "StudentizedSpectrumBootstrap",
    "composite_score_geometry",
    "nuisance_composite_score_geometry",
    "composite_score_test",
    "composite_detection_bounds",
    "nuisance_adjusted_tangent_geometry",
    "nuisance_adjusted_score_geometry",
    "robust_composite_detection_bound",
    "bootstrap_eigenspace_regions",
    "estimate_circular_block_length",
    "estimate_relation_block_length",
    "studentized_bootstrap_relation_spectrum",
]
