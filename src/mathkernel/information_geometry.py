# =============================================================================
# MathKernel - Coordinate-invariant information geometry of observation transfer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Finite-simplex tangent geometry, local testing limits, and uncertainty.

The relation-subspace module represents local perturbations with centered score
functions.  This module records the same object intrinsically as tangent
vectors on the interior of a finite probability simplex.

For an interior law ``p``, a tangent vector ``u`` satisfies ``sum(u) == 0`` and
has Fisher metric

``g_p(u, v) = sum_x u[x] * v[x] / p[x]``.

A row-stochastic observation channel ``K[y|x]`` pushes the tangent to
``K_* u = u K``.  If ``q = p K``, data processing is the matrix inequality

``B.T diag(1/p) B >= (K.T B).T diag(1/q) (K.T B)``

for every tangent-frame matrix ``B``.  Its generalized eigenvalues are
coordinate-independent retained-information fractions.  This is exactly the
score-space construction in :mod:`mathkernel.relation_subspace`, because the
score corresponding to ``u`` is ``u / p``.

The module also turns the weakest retained-information eigenvalue into local
hypothesis-testing bounds.  For a latent-Fisher-unit direction ``u`` and
``p_epsilon = p + epsilon*u``, the observed Pearson divergence is exactly

``chi2(q_epsilon || q) = epsilon**2 * lambda(u)``.

This gives an estimator-independent necessary sample bound.  A finite
Bhattacharyya calculation supplies an exact sufficient bound for the optimal
likelihood-ratio test, while a spectrum-wide version gives local minimax
bounds over the entire declared tangent subspace.

Finally, percentile bootstrap intervals are provided for empirical relation
spectra.  Ordinary, circular moving-block, and cluster resampling are
supported.  Ordered eigenvalue intervals are appropriate for separated
spectra; individual eigenvectors are intentionally not assigned uncertainty
intervals when eigenvalues can collide.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, isfinite, log, log1p, sqrt
from statistics import NormalDist
from typing import Literal, Sequence

import numpy as np

from .phylogenetic_tensor import FiniteObservationChannel
from .relation_subspace import relation_subspace_from_partition


ArrayLike = Sequence[float] | np.ndarray
ChannelLike = FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray
BootstrapMethod = Literal["iid", "moving_block", "cluster"]


def _probability_vector(values: ArrayLike, *, full_support: bool = True) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("law must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("law must be finite and nonnegative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("law must have positive mass")
    vector = vector / total
    if full_support and np.any(vector <= 0):
        raise ValueError("law must have full support")
    return vector


def _channel_rows(channel: ChannelLike) -> np.ndarray:
    raw = channel.conditional if isinstance(channel, FiniteObservationChannel) else channel
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("channel must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("channel entries must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-13, rtol=1e-11):
        raise ValueError("every channel row must sum to one")
    return matrix


def _matrix(values: ArrayLike, *, rows: int | None = None, name: str = "matrix") -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if rows is not None and matrix.shape[0] != rows:
        raise ValueError(f"{name} has the wrong number of rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _ordered_generalized_spectrum(
    latent_metric: np.ndarray,
    observed_metric: np.ndarray,
    *,
    rank_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """Whiten ``latent_metric`` and diagonalize ``observed_metric``.

    Returns eigenvalues, coefficient directions, latent nullspace, blind
    directions, latent rank, observed rank, and data-processing residual.
    """

    latent_metric = _symmetrize(latent_metric)
    observed_metric = _symmetrize(observed_metric)
    latent_values, latent_vectors = np.linalg.eigh(latent_metric)
    scale = max(1.0, float(np.max(np.abs(latent_values))))
    active = latent_values > rank_tolerance * scale
    latent_rank = int(np.count_nonzero(active))
    if latent_rank == 0:
        raise ValueError("declared tangent frame has zero Fisher rank")
    latent_nullspace = latent_vectors[:, ~active]
    whitening = latent_vectors[:, active] / np.sqrt(latent_values[active])[None, :]
    whitened = _symmetrize(whitening.T @ observed_metric @ whitening)
    values, vectors = np.linalg.eigh(whitened)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    residual = max(0.0, -float(np.min(values)), float(np.max(values)) - 1.0)
    if residual > 1e3 * rank_tolerance:
        raise RuntimeError("observation transfer violates Fisher data processing")
    values = np.clip(values, 0.0, 1.0)
    directions = whitening @ vectors
    visible = values > rank_tolerance * max(1.0, float(np.max(values)))
    observed_rank = int(np.count_nonzero(visible))
    blind = directions[:, ~visible]
    return (
        values,
        directions,
        latent_nullspace,
        blind,
        latent_rank,
        observed_rank,
        residual,
    )


@dataclass(frozen=True, slots=True)
class SimplexTangentGeometry:
    """Intrinsic Fisher geometry of a tangent subspace through a channel."""

    latent_law: np.ndarray
    observed_law: np.ndarray
    channel: np.ndarray
    tangent_frame: np.ndarray
    pushed_tangent_frame: np.ndarray
    latent_metric: np.ndarray
    observed_metric: np.ndarray
    generalized_information_eigenvalues: np.ndarray
    transfer_singular_values: np.ndarray
    generalized_coefficient_directions: np.ndarray
    generalized_latent_tangents: np.ndarray
    generalized_observed_tangents: np.ndarray
    latent_nullspace: np.ndarray
    observation_blind_coefficient_directions: np.ndarray
    active_output_indices: tuple[int, ...]
    latent_rank: int
    observed_rank: int
    min_information_retention: float
    min_positive_information_retention: float
    max_information_retention: float
    information_condition_number: float
    uniform_latent_positivity_radius: float
    uniform_observed_score_bound: float
    data_processing_residual: float

    @property
    def fully_identifiable(self) -> bool:
        return self.observed_rank == self.latent_rank


@dataclass(frozen=True, slots=True)
class DirectionTestingBounds:
    """Exact local information and finite-sample bounds for one direction."""

    normalized_coefficients: np.ndarray
    latent_tangent: np.ndarray
    observed_tangent: np.ndarray
    observed_score: np.ndarray
    epsilon: float
    max_error: float
    information_retention: float
    visibility: float
    chi_square_per_sample: float
    bhattacharyya_coefficient: float
    bhattacharyya_information: float
    chi_square_necessary_samples: int | float
    bhattacharyya_sufficient_samples: int | float
    retention_only_sufficient_samples: int | float
    local_asymptotic_samples: int | float
    direction_positivity_radius: float
    standardized_observed_score_abs_bound: float


@dataclass(frozen=True, slots=True)
class SubspaceMinimaxBounds:
    """Worst-direction local limits for a declared tangent subspace.

    ``chi_square_necessary_samples`` is a genuine lower bound for any test
    required to work uniformly over the declared alternatives, because the
    weakest generalized direction belongs to that set.  The field named
    ``uniform_retention_sufficient_samples`` is deliberately only a *uniform
    pointwise* bound: for each fixed known direction there exists a matched
    simple likelihood-ratio test with that sample count.  It is not, by
    itself, an upper bound for one direction-agnostic composite test.
    """

    epsilon: float
    max_error: float
    weakest_information_retention: float
    weakest_visibility: float
    weakest_coefficients: np.ndarray
    weakest_latent_tangent: np.ndarray
    weakest_observed_tangent: np.ndarray
    chi_square_necessary_samples: int | float | None
    uniform_retention_sufficient_samples: int | float | None
    local_asymptotic_samples: int | float | None
    uniform_latent_positivity_radius: float
    uniform_observed_score_bound: float
    exact_blind_direction: bool
    resolution_status: str = "resolved"


@dataclass(frozen=True, slots=True)
class VisibilitySpectrumBootstrap:
    """Ordered generalized-eigenvalue uncertainty from empirical resampling."""

    point_information_eigenvalues: np.ndarray
    point_visibility: np.ndarray
    lower_information_eigenvalues: np.ndarray
    median_information_eigenvalues: np.ndarray
    upper_information_eigenvalues: np.ndarray
    standard_error_information_eigenvalues: np.ndarray
    lower_visibility: np.ndarray
    median_visibility: np.ndarray
    upper_visibility: np.ndarray
    bootstrap_information_eigenvalues: np.ndarray
    confidence_level: float
    method: str
    sample_size: int
    requested_replicates: int
    successful_replicates: int
    failed_replicates: int
    block_length: int | None
    cluster_count: int | None


def simplex_fisher_inner(law: ArrayLike, first: ArrayLike, second: ArrayLike) -> float:
    """Return the Fisher inner product of two finite-simplex tangents."""

    p = _probability_vector(law)
    u = np.asarray(first, dtype=float)
    v = np.asarray(second, dtype=float)
    if u.shape != p.shape or v.shape != p.shape:
        raise ValueError("tangents must match the law")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
        raise ValueError("tangents must be finite")
    tolerance = 1e-11 * max(1.0, float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    if abs(float(u.sum())) > tolerance or abs(float(v.sum())) > tolerance:
        raise ValueError("simplex tangents must sum to zero")
    return float(np.dot(u / p, v))


def scores_to_tangents(
    law: ArrayLike,
    scores: ArrayLike,
    *,
    center: bool = True,
    tolerance: float = 1e-11,
) -> np.ndarray:
    """Convert score columns to intrinsic tangent columns ``diag(p) R``."""

    p = _probability_vector(law)
    matrix = _matrix(scores, rows=p.size, name="scores").copy()
    if center:
        matrix -= np.outer(np.ones(p.size), p @ matrix)
    elif not np.allclose(p @ matrix, 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("score columns must be centered when center=False")
    return p[:, None] * matrix


def tangents_to_scores(
    law: ArrayLike,
    tangents: ArrayLike,
    *,
    tolerance: float = 1e-11,
) -> np.ndarray:
    """Convert intrinsic tangent columns to centered score columns."""

    p = _probability_vector(law)
    matrix = _matrix(tangents, rows=p.size, name="tangents")
    if not np.allclose(matrix.sum(axis=0), 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("every simplex tangent column must sum to zero")
    return matrix / p[:, None]


def push_simplex_tangents(channel: ChannelLike, tangents: ArrayLike) -> np.ndarray:
    """Push tangent columns through a row-stochastic channel."""

    rows = _channel_rows(channel)
    matrix = _matrix(tangents, rows=rows.shape[0], name="tangents")
    pushed = rows.T @ matrix
    if not np.allclose(pushed.sum(axis=0), 0.0, atol=2e-11, rtol=0.0):
        raise RuntimeError("a stochastic channel failed to preserve tangent mass")
    return pushed


def mixture_chart_jacobian(state_count: int, *, reference_state: int = -1) -> np.ndarray:
    """Jacobian of the affine mixture chart using one state as reference."""

    n = int(state_count)
    if n < 2:
        raise ValueError("at least two states are required")
    ref = int(reference_state)
    if ref < 0:
        ref += n
    if not 0 <= ref < n:
        raise ValueError("reference_state is out of range")
    columns: list[np.ndarray] = []
    for state in range(n):
        if state == ref:
            continue
        vector = np.zeros(n, dtype=float)
        vector[state] = 1.0
        vector[ref] = -1.0
        columns.append(vector)
    return np.column_stack(columns)


def softmax_chart_jacobian(law: ArrayLike, *, reference_state: int = -1) -> np.ndarray:
    """Jacobian of log-odds coordinates ``log(p_i/p_reference)`` at ``law``."""

    p = _probability_vector(law)
    n = p.size
    ref = int(reference_state)
    if ref < 0:
        ref += n
    if not 0 <= ref < n:
        raise ValueError("reference_state is out of range")
    full = np.diag(p) - np.outer(p, p)
    keep = [state for state in range(n) if state != ref]
    return full[:, keep]


def simplex_tangent_geometry(
    latent_law: ArrayLike,
    channel: ChannelLike,
    tangent_frame: ArrayLike,
    *,
    support_tolerance: float = 1e-15,
    rank_tolerance: float = 1e-11,
) -> SimplexTangentGeometry:
    """Compute the coordinate-invariant observation geometry of a tangent frame."""

    p = _probability_vector(latent_law)
    rows = _channel_rows(channel)
    if rows.shape[0] != p.size:
        raise ValueError("channel latent alphabet does not match the law")
    frame = _matrix(tangent_frame, rows=p.size, name="tangent_frame")
    if not np.allclose(frame.sum(axis=0), 0.0, atol=rank_tolerance, rtol=0.0):
        raise ValueError("every tangent-frame column must sum to zero")
    q_full = p @ rows
    active = np.flatnonzero(q_full > support_tolerance)
    if active.size == 0:
        raise ValueError("channel has no observed support")
    q = q_full[active]
    pushed_full = rows.T @ frame
    pushed = pushed_full[active]
    latent_metric = _symmetrize(frame.T @ (frame / p[:, None]))
    observed_metric = _symmetrize(pushed.T @ (pushed / q[:, None]))
    (
        values,
        directions,
        latent_nullspace,
        blind,
        latent_rank,
        observed_rank,
        residual,
    ) = _ordered_generalized_spectrum(
        latent_metric,
        observed_metric,
        rank_tolerance=rank_tolerance,
    )
    generalized_latent = frame @ directions
    generalized_observed = pushed_full @ directions
    visible = values > rank_tolerance * max(1.0, float(np.max(values)))
    positive = values[visible]
    min_positive = float(np.min(positive)) if positive.size else 0.0
    condition = (
        float(np.max(positive) / np.min(positive)) if positive.size else inf
    )

    latent_inverse = np.linalg.pinv(latent_metric, rcond=rank_tolerance)
    latent_row_radius = np.sqrt(
        np.maximum(0.0, np.einsum("ij,jk,ik->i", frame, latent_inverse, frame))
    )
    finite_radii = p[latent_row_radius > rank_tolerance] / latent_row_radius[
        latent_row_radius > rank_tolerance
    ]
    uniform_radius = float(np.min(finite_radii)) if finite_radii.size else inf

    output_scores = np.zeros_like(pushed_full)
    output_scores[active] = pushed / q[:, None]
    output_score_radius = np.sqrt(
        np.maximum(
            0.0,
            np.einsum("ij,jk,ik->i", output_scores, latent_inverse, output_scores),
        )
    )
    output_bound = float(np.max(output_score_radius)) if output_score_radius.size else 0.0

    return SimplexTangentGeometry(
        latent_law=p,
        observed_law=q_full,
        channel=rows,
        tangent_frame=frame,
        pushed_tangent_frame=pushed_full,
        latent_metric=latent_metric,
        observed_metric=observed_metric,
        generalized_information_eigenvalues=values,
        transfer_singular_values=np.sqrt(values),
        generalized_coefficient_directions=directions,
        generalized_latent_tangents=generalized_latent,
        generalized_observed_tangents=generalized_observed,
        latent_nullspace=latent_nullspace,
        observation_blind_coefficient_directions=blind,
        active_output_indices=tuple(int(index) for index in active),
        latent_rank=latent_rank,
        observed_rank=observed_rank,
        min_information_retention=float(values[-1]),
        min_positive_information_retention=min_positive,
        max_information_retention=float(values[0]),
        information_condition_number=condition,
        uniform_latent_positivity_radius=uniform_radius,
        uniform_observed_score_bound=output_bound,
        data_processing_residual=residual,
    )


def score_subspace_geometry(
    latent_law: ArrayLike,
    channel: ChannelLike,
    scores: ArrayLike,
    *,
    center: bool = True,
    support_tolerance: float = 1e-15,
    rank_tolerance: float = 1e-11,
) -> SimplexTangentGeometry:
    """Convenience bridge from centered score coordinates to tangent geometry."""

    tangents = scores_to_tangents(latent_law, scores, center=center)
    return simplex_tangent_geometry(
        latent_law,
        channel,
        tangents,
        support_tolerance=support_tolerance,
        rank_tolerance=rank_tolerance,
    )


def tangent_information_retention(
    geometry: SimplexTangentGeometry,
    coefficients: ArrayLike,
    *,
    tolerance: float = 1e-14,
) -> float:
    """Return the Fisher information ratio for one frame coefficient vector."""

    vector = np.asarray(coefficients, dtype=float)
    if vector.ndim != 1 or vector.size != geometry.tangent_frame.shape[1]:
        raise ValueError("coefficient direction has the wrong shape")
    latent = float(vector @ geometry.latent_metric @ vector)
    if latent <= tolerance:
        raise ValueError("coefficient direction is tangent-degenerate")
    observed = float(vector @ geometry.observed_metric @ vector)
    value = observed / latent
    if value < -1e-10 or value > 1.0 + 1e-10:
        raise RuntimeError("directional information violates data processing")
    return float(np.clip(value, 0.0, 1.0))


def _direction_positivity_radius(p: np.ndarray, tangent: np.ndarray) -> float:
    negative = tangent < 0
    if not np.any(negative):
        return inf
    return float(np.min(p[negative] / (-tangent[negative])))


def _integer_ceil_ratio(numerator: float, denominator: float) -> int | float:
    if denominator <= 0 or not isfinite(denominator):
        return inf
    return int(ceil(numerator / denominator))


def direction_testing_bounds(
    geometry: SimplexTangentGeometry,
    coefficients: ArrayLike,
    epsilon: float,
    *,
    max_error: float = 0.10,
    asymptotic_type1: float | None = None,
    asymptotic_type2: float | None = None,
    tolerance: float = 1e-14,
) -> DirectionTestingBounds:
    """Return exact divergence and finite-sample limits for one local direction.

    The comparison is ``H0: p`` versus ``H1: p + epsilon*u`` after normalizing
    ``u`` to unit latent Fisher norm.  Bounds use equal prior probabilities.
    The necessary bound applies whenever the optimal Bayes error is at most
    ``max_error``.  The Bhattacharyya bound is a sufficient sample count for
    that simple likelihood-ratio problem.
    """

    delta = float(max_error)
    if not 0 < delta < 0.5:
        raise ValueError("max_error must lie strictly between zero and one half")
    eps = float(epsilon)
    if not isfinite(eps) or eps == 0:
        raise ValueError("epsilon must be finite and nonzero")
    vector = np.asarray(coefficients, dtype=float)
    if vector.ndim != 1 or vector.size != geometry.tangent_frame.shape[1]:
        raise ValueError("coefficient direction has the wrong shape")
    latent_norm_squared = float(vector @ geometry.latent_metric @ vector)
    if latent_norm_squared <= tolerance:
        raise ValueError("coefficient direction is tangent-degenerate")
    normalized = vector / sqrt(latent_norm_squared)
    tangent = geometry.tangent_frame @ normalized
    pushed = geometry.pushed_tangent_frame @ normalized
    radius = _direction_positivity_radius(geometry.latent_law, np.sign(eps) * tangent)
    if abs(eps) > radius + 1e-13:
        raise ValueError("epsilon leaves the finite-simplex probability region")
    alternative = geometry.latent_law + eps * tangent
    if np.min(alternative) < -1e-12:
        raise ValueError("epsilon leaves the finite-simplex probability region")
    alternative = np.maximum(alternative, 0.0)
    alternative /= alternative.sum()

    q = geometry.observed_law
    q_alternative = q + eps * pushed
    if np.min(q_alternative) < -1e-12:
        raise RuntimeError("pushed alternative left the observed probability simplex")
    q_alternative = np.maximum(q_alternative, 0.0)
    q_alternative /= q_alternative.sum()
    active = q > tolerance
    if np.any((~active) & (q_alternative > tolerance)):
        raise RuntimeError("local alternative created unsupported output mass")
    score = np.zeros_like(q)
    score[active] = pushed[active] / q[active]
    information = float(np.sum(pushed[active] ** 2 / q[active]))
    information = float(np.clip(information, 0.0, 1.0))
    chi_square = eps * eps * information
    required_chi = 4.0 * (1.0 - 2.0 * delta) ** 2
    necessary = (
        inf
        if chi_square <= tolerance
        else int(ceil(log1p(required_chi) / log1p(chi_square)))
    )

    bc = float(np.sum(np.sqrt(q[active] * q_alternative[active])))
    bc = float(np.clip(bc, 0.0, 1.0))
    if bc <= tolerance:
        sufficient = 1
        bc_information = inf
    elif 1.0 - bc <= tolerance:
        sufficient = inf
        bc_information = 0.0
    else:
        bc_information = -log(bc)
        sufficient = int(ceil(log(1.0 / (2.0 * delta)) / bc_information))

    score_bound = float(np.max(np.abs(score)))
    a = abs(eps) * score_bound
    retained_hellinger = (
        eps * eps * information / (2.0 * (1.0 + sqrt(1.0 + a)) ** 2)
    )
    if retained_hellinger <= tolerance:
        retention_sufficient = inf
    else:
        retained_hellinger = min(retained_hellinger, 1.0 - 1e-15)
        retention_sufficient = int(
            ceil(log(1.0 / (2.0 * delta)) / (-log(1.0 - retained_hellinger)))
        )

    alpha = delta if asymptotic_type1 is None else float(asymptotic_type1)
    beta = delta if asymptotic_type2 is None else float(asymptotic_type2)
    if not (0 < alpha < 0.5 and 0 < beta < 0.5):
        raise ValueError("asymptotic error probabilities must lie in (0, 1/2)")
    if chi_square <= tolerance:
        local_samples = inf
    else:
        z = NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(1.0 - beta)
        local_samples = int(ceil(z * z / chi_square))

    return DirectionTestingBounds(
        normalized_coefficients=normalized,
        latent_tangent=tangent,
        observed_tangent=pushed,
        observed_score=score,
        epsilon=eps,
        max_error=delta,
        information_retention=information,
        visibility=sqrt(information),
        chi_square_per_sample=chi_square,
        bhattacharyya_coefficient=bc,
        bhattacharyya_information=bc_information,
        chi_square_necessary_samples=necessary,
        bhattacharyya_sufficient_samples=sufficient,
        retention_only_sufficient_samples=retention_sufficient,
        local_asymptotic_samples=local_samples,
        direction_positivity_radius=radius,
        standardized_observed_score_abs_bound=score_bound,
    )


def subspace_minimax_bounds(
    geometry: SimplexTangentGeometry,
    epsilon: float,
    *,
    max_error: float = 0.10,
    tolerance: float = 1e-14,
) -> SubspaceMinimaxBounds:
    """Local minimax bounds over all latent-Fisher-unit subspace directions.

    The lower bound follows because the weakest generalized direction is part
    of the composite alternative set.  The reported retention-based upper
    count is *pointwise*: it controls each fixed, known direction using the
    weakest retained information and largest possible observed score, provided
    ``epsilon`` lies within the uniform positivity radius.  Establishing one
    direction-agnostic composite test needs an additional dimension/covering
    argument and is intentionally not claimed here.
    """

    eps = float(epsilon)
    delta = float(max_error)
    if not 0 < delta < 0.5:
        raise ValueError("max_error must lie strictly between zero and one half")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    if eps > geometry.uniform_latent_positivity_radius + 1e-13:
        raise ValueError("epsilon exceeds the subspace-uniform positivity radius")

    values = geometry.generalized_information_eigenvalues
    coefficients = geometry.generalized_coefficient_directions[:, -1]
    tangent = geometry.generalized_latent_tangents[:, -1]
    pushed = geometry.generalized_observed_tangents[:, -1]
    weakest = float(values[-1])
    blind = bool(exact_observation_rank_loss(geometry)) if weakest <= tolerance else False
    per_sample = eps * eps * weakest
    unresolved = per_sample <= 0 and not blind
    required_chi = 4.0 * (1.0 - 2.0 * delta) ** 2
    necessary = (
        inf
        if per_sample <= 0
        else int(ceil(log1p(required_chi) / log1p(per_sample)))
    )

    output_bound = geometry.uniform_observed_score_bound
    a = eps * output_bound
    retained_hellinger = (
        eps * eps * weakest / (2.0 * (1.0 + sqrt(1.0 + a)) ** 2)
    )
    if retained_hellinger <= 0:
        uniform_sufficient = inf
    else:
        retained_hellinger = min(retained_hellinger, 1.0 - 1e-15)
        uniform_sufficient = int(
            ceil(log(1.0 / (2.0 * delta)) / (-log1p(-retained_hellinger)))
        )

    if per_sample <= 0:
        local_samples = inf
    else:
        z = 2.0 * NormalDist().inv_cdf(1.0 - delta)
        local_samples = int(ceil(z * z / per_sample))

    if blind:
        necessary = uniform_sufficient = local_samples = inf
    elif unresolved:
        necessary = uniform_sufficient = local_samples = None

    return SubspaceMinimaxBounds(
        epsilon=eps,
        max_error=delta,
        weakest_information_retention=weakest,
        weakest_visibility=sqrt(weakest),
        weakest_coefficients=coefficients,
        weakest_latent_tangent=tangent,
        weakest_observed_tangent=pushed,
        chi_square_necessary_samples=necessary,
        uniform_retention_sufficient_samples=uniform_sufficient,
        local_asymptotic_samples=local_samples,
        uniform_latent_positivity_radius=geometry.uniform_latent_positivity_radius,
        uniform_observed_score_bound=output_bound,
        exact_blind_direction=blind,
        resolution_status="exact_rank_loss" if blind else ("numerically_unresolved" if unresolved else "resolved"),
    )


def exact_observation_rank_loss(geometry: SimplexTangentGeometry, *, max_cells: int = 4096) -> bool | None:
    """Check rank loss over the exact rational values of the supplied floats.

    This is an algebraic certificate for the *declared finite model*, not a
    statement about measurement accuracy. Large matrices return None instead
    of turning an SVD tolerance into an exact nullspace certificate.
    """
    import sympy as sp
    a = np.asarray(geometry.channel, dtype=float)
    t = np.asarray(geometry.tangent_frame, dtype=float)
    if a.size + t.size > max_cells or max(*a.shape, *t.shape) > 64:
        return None
    rational = lambda x: sp.Rational(*float(x).as_integer_ratio())
    channel = sp.Matrix([[rational(x) for x in row] for row in a])
    tangent = sp.Matrix([[rational(x) for x in row] for row in t])
    return bool((channel.T * tangent).rank() < tangent.rank())


def _resample_indices(
    rng: np.random.Generator,
    n: int,
    method: BootstrapMethod,
    *,
    block_length: int | None,
    clusters: np.ndarray | None,
) -> np.ndarray:
    if method == "iid":
        return rng.integers(0, n, size=n, dtype=np.int64)
    if method == "moving_block":
        if block_length is None:
            raise ValueError("block_length is required for moving_block bootstrap")
        length = int(block_length)
        if not 1 <= length <= n:
            raise ValueError("block_length must lie in [1, sample_size]")
        block_count = int(ceil(n / length))
        starts = rng.integers(0, n, size=block_count)
        offsets = np.arange(length, dtype=np.int64)
        return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
    if method == "cluster":
        if clusters is None:
            raise ValueError("clusters are required for cluster bootstrap")
        unique = np.unique(clusters)
        if unique.size < 2:
            raise ValueError("cluster bootstrap requires at least two clusters")
        selected = rng.choice(unique, size=unique.size, replace=True)
        pieces = [np.flatnonzero(clusters == value) for value in selected]
        return np.concatenate(pieces).astype(np.int64, copy=False)
    raise ValueError("method must be 'iid', 'moving_block', or 'cluster'")


def bootstrap_empirical_relation_spectrum(
    latent_score_rows: ArrayLike,
    observed_labels: Sequence[int] | np.ndarray,
    *,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    method: BootstrapMethod = "iid",
    block_length: int | None = None,
    clusters: Sequence[object] | np.ndarray | None = None,
    seed: int | np.random.Generator = 20260906,
    rank_tolerance: float = 1e-11,
) -> VisibilitySpectrumBootstrap:
    """Estimate relation-spectrum uncertainty from paired empirical records.

    Each row carries latent score values and an observed category.  Treating
    rows as equally weighted finite latent states makes the deterministic
    partition transfer equal to the empirical conditional-expectation
    estimate.  Resampling rows therefore bootstraps the complete Fisher Gram
    calculation without estimating a dense channel matrix first.
    """

    scores = _matrix(latent_score_rows, name="latent_score_rows")
    n = scores.shape[0]
    raw_labels = np.asarray(observed_labels)
    if raw_labels.ndim != 1 or raw_labels.size != n:
        raise ValueError("observed_labels must have one entry per score row")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        if not np.all(np.isfinite(raw_labels)) or not np.all(raw_labels == np.floor(raw_labels)):
            raise ValueError("observed_labels must be finite integers")
    labels = raw_labels.astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("observed_labels must be nonnegative")
    requested = int(n_bootstrap)
    if requested < 20:
        raise ValueError("at least 20 bootstrap replicates are required")
    confidence = float(confidence_level)
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    cluster_array = None if clusters is None else np.asarray(clusters)
    if cluster_array is not None and (cluster_array.ndim != 1 or cluster_array.size != n):
        raise ValueError("clusters must have one entry per score row")

    point = relation_subspace_from_partition(
        np.full(n, 1.0 / n),
        labels,
        scores,
        rank_tolerance=rank_tolerance,
    )
    target_rank = point.latent_rank
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    values: list[np.ndarray] = []
    failures = 0
    for _ in range(requested):
        indices = _resample_indices(
            rng,
            n,
            method,
            block_length=block_length,
            clusters=cluster_array,
        )
        if indices.size < max(3, target_rank + 1):
            failures += 1
            continue
        try:
            transfer = relation_subspace_from_partition(
                np.full(indices.size, 1.0 / indices.size),
                labels[indices],
                scores[indices],
                rank_tolerance=rank_tolerance,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            failures += 1
            continue
        replicate = np.zeros(target_rank, dtype=float)
        available = min(target_rank, transfer.generalized_information_eigenvalues.size)
        replicate[:available] = transfer.generalized_information_eigenvalues[:available]
        values.append(replicate)
    if len(values) < max(20, requested // 2):
        raise RuntimeError("too few successful bootstrap replicates")
    samples = np.vstack(values)
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(samples, alpha, axis=0)
    median = np.quantile(samples, 0.5, axis=0)
    upper = np.quantile(samples, 1.0 - alpha, axis=0)
    standard_error = np.std(samples, axis=0, ddof=1)
    point_values = point.generalized_information_eigenvalues.copy()
    return VisibilitySpectrumBootstrap(
        point_information_eigenvalues=point_values,
        point_visibility=np.sqrt(point_values),
        lower_information_eigenvalues=lower,
        median_information_eigenvalues=median,
        upper_information_eigenvalues=upper,
        standard_error_information_eigenvalues=standard_error,
        lower_visibility=np.sqrt(np.maximum(0.0, lower)),
        median_visibility=np.sqrt(np.maximum(0.0, median)),
        upper_visibility=np.sqrt(np.maximum(0.0, upper)),
        bootstrap_information_eigenvalues=samples,
        confidence_level=confidence,
        method=method,
        sample_size=n,
        requested_replicates=requested,
        successful_replicates=samples.shape[0],
        failed_replicates=failures,
        block_length=block_length if method == "moving_block" else None,
        cluster_count=(int(np.unique(cluster_array).size) if method == "cluster" and cluster_array is not None else None),
    )


__all__ = [
    "SimplexTangentGeometry",
    "DirectionTestingBounds",
    "SubspaceMinimaxBounds",
    "VisibilitySpectrumBootstrap",
    "simplex_fisher_inner",
    "scores_to_tangents",
    "tangents_to_scores",
    "push_simplex_tangents",
    "mixture_chart_jacobian",
    "softmax_chart_jacobian",
    "simplex_tangent_geometry",
    "score_subspace_geometry",
    "tangent_information_retention",
    "direction_testing_bounds",
    "subspace_minimax_bounds",
    "bootstrap_empirical_relation_spectrum",
]
