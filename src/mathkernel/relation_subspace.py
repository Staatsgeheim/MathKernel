# =============================================================================
# MathKernel - Multi-relation observation transfer and experiment design
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Finite multi-relation visibility, Fisher geometry, and sensor design.

This module is the multi-parameter counterpart of :mod:`relation_detection`.
Let ``p`` be a baseline law on a finite latent state space and let the columns
of ``R`` be centered latent score directions.  For a row-stochastic channel
``K[y|x]`` define ``q = p K`` and

``H[y,a] = E_p[R[X,a] | Y=y]``.

The latent and observed Fisher Gram matrices are

``G_lat = R.T diag(p) R`` and ``G_obs = H.T diag(q) H``.

For the locally linear family

``p_theta(x) = p(x) * (1 + R[x,:] @ theta)``,

these identities are exact whenever the displayed probabilities are
nonnegative:

``q_theta(y) = q(y) * (1 + H[y,:] @ theta)``

and

``chi2(q_theta || q) = theta.T G_obs theta``.

The generalized eigenvalues of ``(G_obs, G_lat)`` lie in ``[0, 1]`` and are
the retained-information factors of orthogonal relation directions.  Their
square roots are relation-subspace visibility singular values.  A zero value
is an exact observation-blind combination, not merely a badly estimated one.

The module also provides exhaustive finite sensor-subset optimization and a
long-run-covariance correction for dependent score observations.  It does not
claim that arbitrary nonlinear or globally parameterized statistical models
are exhausted by their tangent-space Fisher geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import inf, isfinite, log
from typing import Mapping, Sequence

import numpy as np

from .phylogenetic_tensor import FiniteObservationChannel
from .relation_detection import joint_conditionally_independent_channel


ArrayLike = Sequence[float] | np.ndarray
ChannelLike = FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray


def _probability_vector(values: ArrayLike, *, full_support: bool = True) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("baseline law must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("baseline law must be finite and nonnegative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("baseline law must have positive total mass")
    vector = vector / total
    if full_support and np.any(vector <= 0):
        raise ValueError("baseline law must have full support")
    return vector


def _channel_rows(channel: ChannelLike) -> np.ndarray:
    raw = channel.conditional if isinstance(channel, FiniteObservationChannel) else channel
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("channel must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("channel entries must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-11, atol=1e-13):
        raise ValueError("every channel row must sum to one")
    return matrix


def _score_matrix(scores: ArrayLike, n_states: int) -> np.ndarray:
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n_states or matrix.shape[1] == 0:
        raise ValueError(
            "scores must have shape (latent_states, relation_directions)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("scores must be finite")
    return matrix


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _finite_channel(channel: ChannelLike, *, name: str) -> FiniteObservationChannel:
    if isinstance(channel, FiniteObservationChannel):
        return channel
    return FiniteObservationChannel.from_rows(_channel_rows(channel).tolist(), name=name)


@dataclass(frozen=True, slots=True)
class RelationSubspaceTransfer:
    """Complete Fisher-geometry description of a relation subspace."""

    latent_law: np.ndarray
    observed_law: np.ndarray
    latent_scores: np.ndarray
    observed_scores: np.ndarray
    latent_gram: np.ndarray
    observed_gram: np.ndarray
    generalized_information_eigenvalues: np.ndarray
    transfer_singular_values: np.ndarray
    generalized_directions: np.ndarray
    latent_nullspace: np.ndarray
    observation_blind_directions: np.ndarray
    active_output_indices: tuple[int, ...]
    latent_rank: int
    observed_rank: int
    max_visibility: float
    min_visibility: float
    min_positive_visibility: float
    mean_information_retention: float
    information_condition_number: float
    transfer_condition_number: float
    logdet_information_retention: float
    pseudo_logdet_information_retention: float
    data_processing_residual: float

    @property
    def fully_identifiable(self) -> bool:
        """Whether observation preserves every nondegenerate latent direction."""

        return self.observed_rank == self.latent_rank


@dataclass(frozen=True, slots=True)
class RelationCollisionCertificate:
    """Two distinct latent laws with the same observed law along a blind direction."""

    coefficient_direction: np.ndarray
    latent_score_direction: np.ndarray
    epsilon: float
    latent_minus: np.ndarray
    latent_plus: np.ndarray
    observed_minus: np.ndarray
    observed_plus: np.ndarray
    latent_l1_separation: float
    observed_l1_separation: float
    exact_within_tolerance: bool


@dataclass(frozen=True, slots=True)
class SensorSubsetEvaluation:
    """Relation-space quality of one conditionally independent sensor subset."""

    sensors: tuple[str, ...]
    total_cost: float
    output_states: int
    latent_rank: int
    observed_rank: int
    fully_identifiable: bool
    min_visibility: float
    min_positive_visibility: float
    mean_information_retention: float
    information_trace: float
    logdet_information_retention: float
    pseudo_logdet_information_retention: float
    information_condition_number: float
    transfer_condition_number: float
    objective: str
    objective_value: float


@dataclass(frozen=True, slots=True)
class SensorOptimizationResult:
    objective: str
    best_value: float
    best_subsets: tuple[SensorSubsetEvaluation, ...]
    evaluations: tuple[SensorSubsetEvaluation, ...]


@dataclass(frozen=True, slots=True)
class DependentRelationInformation:
    """Fisher geometry after correcting score means for serial dependence."""

    iid_information: np.ndarray
    long_run_covariance: np.ndarray
    effective_information: np.ndarray
    iid_rank: int
    effective_rank: int
    generalized_information_eigenvalues: np.ndarray
    sample_inflation_eigenvalues: np.ndarray
    worst_case_sample_inflation: float
    best_case_sample_inflation: float


def _assemble_relation_subspace(
    p: np.ndarray,
    q_full: np.ndarray,
    relation_scores: np.ndarray,
    observed_scores: np.ndarray,
    active: np.ndarray,
    *,
    rank_tolerance: float,
) -> RelationSubspaceTransfer:
    q = q_full[active]
    g_lat = _symmetrize(relation_scores.T @ (p[:, None] * relation_scores))
    g_obs = _symmetrize(observed_scores.T @ (q[:, None] * observed_scores))

    latent_values, latent_vectors = np.linalg.eigh(g_lat)
    latent_scale = max(1.0, float(np.max(np.abs(latent_values))))
    latent_active = latent_values > rank_tolerance * latent_scale
    latent_rank = int(np.count_nonzero(latent_active))
    if latent_rank == 0:
        raise ValueError("declared score family has zero latent Fisher rank")
    latent_nullspace = latent_vectors[:, ~latent_active]

    whitening = latent_vectors[:, latent_active] / np.sqrt(
        latent_values[latent_active]
    )[None, :]
    whitened_observed = _symmetrize(whitening.T @ g_obs @ whitening)
    info_values, info_vectors = np.linalg.eigh(whitened_observed)
    order = np.argsort(info_values)[::-1]
    info_values = info_values[order]
    info_vectors = info_vectors[:, order]

    # Numerical roundoff may move a theoretically valid value a few ulps
    # outside [0,1].  Preserve genuine violations in the residual before
    # clipping and expose that residual to callers.
    data_processing_residual = max(
        0.0,
        -float(np.min(info_values)),
        float(np.max(info_values)) - 1.0,
    )
    if data_processing_residual > 1e3 * rank_tolerance:
        raise RuntimeError("observed Fisher information violates data processing")
    info_values = np.clip(info_values, 0.0, 1.0)
    directions = whitening @ info_vectors

    information_scale = max(1.0, float(np.max(info_values)))
    visible = info_values > rank_tolerance * information_scale
    observed_rank = int(np.count_nonzero(visible))
    blind_directions = directions[:, ~visible]
    singular = np.sqrt(info_values)
    positive_singular = singular[visible]

    max_visibility = float(singular[0]) if singular.size else 0.0
    min_visibility = float(singular[-1]) if singular.size else 0.0
    min_positive_visibility = (
        float(np.min(positive_singular)) if positive_singular.size else 0.0
    )
    if observed_rank == 0:
        information_condition = inf
        transfer_condition = inf
        pseudo_logdet = -inf
    else:
        positive_info = info_values[visible]
        information_condition = float(np.max(positive_info) / np.min(positive_info))
        transfer_condition = float(np.sqrt(information_condition))
        pseudo_logdet = float(np.sum(np.log(positive_info)))
    full_logdet = (
        float(np.sum(np.log(info_values)))
        if observed_rank == latent_rank
        else -inf
    )

    return RelationSubspaceTransfer(
        latent_law=p,
        observed_law=q_full,
        latent_scores=relation_scores,
        observed_scores=observed_scores,
        latent_gram=g_lat,
        observed_gram=g_obs,
        generalized_information_eigenvalues=info_values,
        transfer_singular_values=singular,
        generalized_directions=directions,
        latent_nullspace=latent_nullspace,
        observation_blind_directions=blind_directions,
        active_output_indices=tuple(int(value) for value in active),
        latent_rank=latent_rank,
        observed_rank=observed_rank,
        max_visibility=max_visibility,
        min_visibility=min_visibility,
        min_positive_visibility=min_positive_visibility,
        mean_information_retention=float(np.mean(info_values)),
        information_condition_number=information_condition,
        transfer_condition_number=transfer_condition,
        logdet_information_retention=full_logdet,
        pseudo_logdet_information_retention=pseudo_logdet,
        data_processing_residual=data_processing_residual,
    )


def relation_subspace_transfer(
    baseline_law: ArrayLike,
    channel: ChannelLike,
    scores: ArrayLike,
    *,
    center: bool = True,
    support_tolerance: float = 1e-15,
    rank_tolerance: float = 1e-11,
) -> RelationSubspaceTransfer:
    """Push a finite latent score subspace through a stochastic channel.

    ``scores`` has one row per latent state and one column per declared
    relation direction.  The returned generalized directions are columns in
    coefficient space, normalized so that ``c.T G_lat c == 1``.
    """

    p = _probability_vector(baseline_law)
    rows = _channel_rows(channel)
    if rows.shape[0] != p.size:
        raise ValueError("channel latent alphabet does not match baseline law")
    relation_scores = _score_matrix(scores, p.size).copy()
    if center:
        relation_scores -= np.outer(np.ones(p.size), p @ relation_scores)
    elif not np.allclose(p @ relation_scores, 0.0, atol=rank_tolerance, rtol=0.0):
        raise ValueError("score columns must be centered when center=False")

    q_full = p @ rows
    active = np.flatnonzero(q_full > support_tolerance)
    if active.size == 0:
        raise ValueError("channel has no observed support under the baseline law")
    q = q_full[active]
    active_rows = rows[:, active]
    numerator = active_rows.T @ (p[:, None] * relation_scores)
    observed_scores = numerator / q[:, None]
    return _assemble_relation_subspace(
        p,
        q_full,
        relation_scores,
        observed_scores,
        active,
        rank_tolerance=rank_tolerance,
    )


def relation_subspace_from_partition(
    baseline_law: ArrayLike,
    labels: Sequence[int] | np.ndarray,
    scores: ArrayLike,
    *,
    center: bool = True,
    support_tolerance: float = 1e-15,
    rank_tolerance: float = 1e-11,
) -> RelationSubspaceTransfer:
    """Efficient deterministic-observation transfer from integer labels.

    This is mathematically identical to a one-hot deterministic channel but
    avoids materializing an ``n_states x n_outputs`` dense matrix.  It is
    especially useful when the latent states are empirical records and the
    observation is a discretized sensor family.
    """

    p = _probability_vector(baseline_law)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or raw_labels.size != p.size:
        raise ValueError("labels must be one-dimensional with one value per latent state")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        if not np.all(np.isfinite(raw_labels)) or not np.all(raw_labels == np.floor(raw_labels)):
            raise ValueError("labels must contain finite integers")
    integer_labels = raw_labels.astype(np.int64)
    if np.any(integer_labels < 0):
        raise ValueError("labels must be nonnegative")
    _, compact = np.unique(integer_labels, return_inverse=True)
    output_count = int(compact.max()) + 1

    relation_scores = _score_matrix(scores, p.size).copy()
    if center:
        relation_scores -= np.outer(np.ones(p.size), p @ relation_scores)
    elif not np.allclose(p @ relation_scores, 0.0, atol=rank_tolerance, rtol=0.0):
        raise ValueError("score columns must be centered when center=False")

    q_full = np.bincount(compact, weights=p, minlength=output_count).astype(float)
    numerator = np.zeros((output_count, relation_scores.shape[1]), dtype=float)
    np.add.at(numerator, compact, p[:, None] * relation_scores)
    active = np.flatnonzero(q_full > support_tolerance)
    if active.size == 0:
        raise ValueError("partition has no observed support")
    observed_scores = numerator[active] / q_full[active, None]
    return _assemble_relation_subspace(
        p,
        q_full,
        relation_scores,
        observed_scores,
        active,
        rank_tolerance=rank_tolerance,
    )

def direction_information_retention(
    transfer: RelationSubspaceTransfer,
    coefficients: ArrayLike,
    *,
    tolerance: float = 1e-14,
) -> float:
    """Return ``c.T G_obs c / (c.T G_lat c)`` for one relation direction."""

    vector = np.asarray(coefficients, dtype=float)
    if vector.ndim != 1 or vector.size != transfer.latent_gram.shape[0]:
        raise ValueError("coefficient direction has the wrong shape")
    latent = float(vector @ transfer.latent_gram @ vector)
    if latent <= tolerance:
        raise ValueError("coefficient direction is latent-degenerate")
    observed = float(vector @ transfer.observed_gram @ vector)
    ratio = observed / latent
    if ratio < -1e-10 or ratio > 1.0 + 1e-10:
        raise RuntimeError("directional information retention violates data processing")
    return float(np.clip(ratio, 0.0, 1.0))


def direction_visibility(
    transfer: RelationSubspaceTransfer,
    coefficients: ArrayLike,
) -> float:
    """Square root of :func:`direction_information_retention`."""

    return float(np.sqrt(direction_information_retention(transfer, coefficients)))


def linear_relation_law(
    baseline_law: ArrayLike,
    scores: ArrayLike,
    parameters: ArrayLike,
    *,
    center: bool = True,
    nonnegative_tolerance: float = 1e-13,
) -> np.ndarray:
    """Construct ``p * (1 + R theta)`` after validating nonnegativity."""

    p = _probability_vector(baseline_law)
    relation_scores = _score_matrix(scores, p.size).copy()
    if center:
        relation_scores -= np.outer(np.ones(p.size), p @ relation_scores)
    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 1 or theta.size != relation_scores.shape[1]:
        raise ValueError("parameter vector has the wrong shape")
    if not np.all(np.isfinite(theta)):
        raise ValueError("parameters must be finite")
    law = p * (1.0 + relation_scores @ theta)
    if float(np.min(law)) < -nonnegative_tolerance:
        raise ValueError("parameters leave the nonnegative probability region")
    law = np.maximum(law, 0.0)
    total = float(law.sum())
    if total <= 0:
        raise RuntimeError("constructed law has no mass")
    return law / total


def push_finite_law(law: ArrayLike, channel: ChannelLike) -> np.ndarray:
    """Push one finite probability vector through a row-stochastic channel."""

    p = _probability_vector(law, full_support=False)
    rows = _channel_rows(channel)
    if rows.shape[0] != p.size:
        raise ValueError("channel latent alphabet does not match law")
    return p @ rows


def chi_square_divergence(law: ArrayLike, reference: ArrayLike) -> float:
    """Finite Pearson chi-square divergence with support validation."""

    p = _probability_vector(law, full_support=False)
    q = _probability_vector(reference, full_support=False)
    if p.shape != q.shape:
        raise ValueError("law and reference must have the same shape")
    if np.any((q == 0) & (p > 0)):
        raise ValueError("law is not absolutely continuous with respect to reference")
    active = q > 0
    return float(np.sum((p[active] - q[active]) ** 2 / q[active]))


def relation_chi_square(
    transfer: RelationSubspaceTransfer,
    parameters: ArrayLike,
) -> float:
    """Exact tangent-family output chi-square ``theta.T G_obs theta``."""

    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 1 or theta.size != transfer.observed_gram.shape[0]:
        raise ValueError("parameter vector has the wrong shape")
    return float(theta @ transfer.observed_gram @ theta)


def collision_certificate(
    baseline_law: ArrayLike,
    channel: ChannelLike,
    scores: ArrayLike,
    coefficient_direction: ArrayLike,
    *,
    epsilon: float | None = None,
    tolerance: float = 1e-11,
) -> RelationCollisionCertificate:
    """Construct a plus/minus pair along a declared observation-blind direction."""

    p = _probability_vector(baseline_law)
    rows = _channel_rows(channel)
    relation_scores = _score_matrix(scores, p.size).copy()
    relation_scores -= np.outer(np.ones(p.size), p @ relation_scores)
    coefficients = np.asarray(coefficient_direction, dtype=float)
    if coefficients.ndim != 1 or coefficients.size != relation_scores.shape[1]:
        raise ValueError("coefficient direction has the wrong shape")
    score = relation_scores @ coefficients
    scale = float(np.max(np.abs(score)))
    if scale <= tolerance:
        raise ValueError("coefficient direction is latent-degenerate")
    maximum = 0.95 / scale
    chosen = 0.5 * maximum if epsilon is None else float(epsilon)
    if not isfinite(chosen) or chosen <= 0 or chosen > maximum:
        raise ValueError("epsilon must be positive and stay inside the probability region")
    minus = linear_relation_law(p, relation_scores, -chosen * coefficients, center=False)
    plus = linear_relation_law(p, relation_scores, chosen * coefficients, center=False)
    observed_minus = push_finite_law(minus, rows)
    observed_plus = push_finite_law(plus, rows)
    observed_distance = float(np.sum(np.abs(observed_plus - observed_minus)))
    return RelationCollisionCertificate(
        coefficient_direction=coefficients,
        latent_score_direction=score,
        epsilon=chosen,
        latent_minus=minus,
        latent_plus=plus,
        observed_minus=observed_minus,
        observed_plus=observed_plus,
        latent_l1_separation=float(np.sum(np.abs(plus - minus))),
        observed_l1_separation=observed_distance,
        exact_within_tolerance=observed_distance <= tolerance,
    )


def _objective_value(transfer: RelationSubspaceTransfer, objective: str) -> float:
    normalized = objective.strip().lower().replace("-", "_")
    if normalized in {"rank", "identifiable_rank"}:
        return float(transfer.observed_rank)
    if normalized in {"e", "e_optimal", "min_visibility"}:
        return float(transfer.min_visibility**2)
    if normalized in {"trace", "a_trace", "mean_information"}:
        return float(np.sum(transfer.generalized_information_eigenvalues))
    if normalized in {"d", "d_optimal", "logdet"}:
        return float(transfer.logdet_information_retention)
    if normalized in {"pseudo_logdet", "pseudo_d"}:
        return float(transfer.pseudo_logdet_information_retention)
    raise ValueError(
        "objective must be one of rank, e_optimal, trace, d_optimal, pseudo_logdet"
    )


def evaluate_sensor_subset(
    baseline_law: ArrayLike,
    scores: ArrayLike,
    sensors: Mapping[str, ChannelLike],
    selected: Sequence[str],
    *,
    costs: Mapping[str, float] | None = None,
    objective: str = "e_optimal",
) -> SensorSubsetEvaluation:
    """Evaluate one subset as conditionally independent observations given state."""

    names = tuple(str(name) for name in selected)
    if not names:
        raise ValueError("at least one sensor must be selected")
    if len(set(names)) != len(names):
        raise ValueError("selected sensor names must be unique")
    missing = [name for name in names if name not in sensors]
    if missing:
        raise KeyError(f"unknown sensor(s): {', '.join(missing)}")
    finite = tuple(_finite_channel(sensors[name], name=name) for name in names)
    latent_states = finite[0].latent_states
    if any(sensor.latent_states != latent_states for sensor in finite):
        raise ValueError("all sensors must share the same latent state space")
    joint = joint_conditionally_independent_channel(finite, name=" + ".join(names))
    transfer = relation_subspace_transfer(baseline_law, joint, scores)
    total_cost = float(sum((costs or {}).get(name, 1.0) for name in names))
    if not isfinite(total_cost) or total_cost < 0:
        raise ValueError("sensor costs must be finite and nonnegative")
    normalized_objective = objective.strip().lower().replace("-", "_")
    return SensorSubsetEvaluation(
        sensors=names,
        total_cost=total_cost,
        output_states=joint.output_states,
        latent_rank=transfer.latent_rank,
        observed_rank=transfer.observed_rank,
        fully_identifiable=transfer.fully_identifiable,
        min_visibility=transfer.min_visibility,
        min_positive_visibility=transfer.min_positive_visibility,
        mean_information_retention=transfer.mean_information_retention,
        information_trace=float(np.sum(transfer.generalized_information_eigenvalues)),
        logdet_information_retention=transfer.logdet_information_retention,
        pseudo_logdet_information_retention=transfer.pseudo_logdet_information_retention,
        information_condition_number=transfer.information_condition_number,
        transfer_condition_number=transfer.transfer_condition_number,
        objective=normalized_objective,
        objective_value=_objective_value(transfer, normalized_objective),
    )


def optimize_sensor_subsets(
    baseline_law: ArrayLike,
    scores: ArrayLike,
    sensors: Mapping[str, ChannelLike],
    *,
    costs: Mapping[str, float] | None = None,
    max_sensors: int | None = None,
    max_total_cost: float | None = None,
    objective: str = "e_optimal",
    tie_tolerance: float = 1e-12,
) -> SensorOptimizationResult:
    """Exhaustively optimize finite sensor subsets under cardinality/cost limits."""

    names = tuple(sorted(str(name) for name in sensors))
    if not names:
        raise ValueError("sensor mapping must be nonempty")
    limit = len(names) if max_sensors is None else int(max_sensors)
    if limit < 1:
        raise ValueError("max_sensors must be positive")
    limit = min(limit, len(names))
    if max_total_cost is not None and (
        not isfinite(float(max_total_cost)) or float(max_total_cost) < 0
    ):
        raise ValueError("max_total_cost must be finite and nonnegative")

    evaluations: list[SensorSubsetEvaluation] = []
    for size in range(1, limit + 1):
        for subset in combinations(names, size):
            cost = float(sum((costs or {}).get(name, 1.0) for name in subset))
            if max_total_cost is not None and cost > float(max_total_cost) + 1e-15:
                continue
            evaluations.append(
                evaluate_sensor_subset(
                    baseline_law,
                    scores,
                    sensors,
                    subset,
                    costs=costs,
                    objective=objective,
                )
            )
    if not evaluations:
        raise ValueError("no sensor subset satisfies the declared constraints")
    best = max(item.objective_value for item in evaluations)
    if best == -inf:
        winners = tuple(item for item in evaluations if item.objective_value == -inf)
    else:
        scale = max(1.0, abs(best))
        winners = tuple(
            item
            for item in evaluations
            if abs(item.objective_value - best) <= tie_tolerance * scale
        )
    ordered = tuple(
        sorted(
            evaluations,
            key=lambda item: (
                -item.objective_value,
                item.total_cost,
                len(item.sensors),
                item.sensors,
            ),
        )
    )
    return SensorOptimizationResult(
        objective=objective.strip().lower().replace("-", "_"),
        best_value=float(best),
        best_subsets=winners,
        evaluations=ordered,
    )


def newey_west_long_run_covariance(
    score_samples: ArrayLike,
    max_lag: int,
    *,
    center: bool = True,
    finite_sample: bool = False,
) -> np.ndarray:
    """Bartlett/Newey-West long-run covariance of a score-vector series."""

    samples = np.asarray(score_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] == 0:
        raise ValueError("score_samples must have shape (time, score_dimension)")
    if not np.all(np.isfinite(samples)):
        raise ValueError("score_samples must be finite")
    lag_limit = int(max_lag)
    if lag_limit < 0 or lag_limit >= samples.shape[0]:
        raise ValueError("max_lag must lie between zero and sample_count - 1")
    values = samples - samples.mean(axis=0, keepdims=True) if center else samples.copy()
    n = values.shape[0]
    denominator0 = n - 1 if finite_sample else n
    omega = (values.T @ values) / denominator0
    for lag in range(1, lag_limit + 1):
        weight = 1.0 - lag / (lag_limit + 1.0)
        denominator = n - lag - (1 if finite_sample else 0)
        if denominator <= 0:
            break
        gamma = (values[lag:].T @ values[:-lag]) / denominator
        omega += weight * (gamma + gamma.T)
    return _symmetrize(omega)


def dependent_relation_information(
    iid_information: ArrayLike,
    long_run_covariance: ArrayLike,
    *,
    rank_tolerance: float = 1e-11,
) -> DependentRelationInformation:
    """Correct score-mean information for serial dependence.

    If the observed null score has per-observation Fisher Gram ``G`` and
    long-run covariance ``Omega``, its local mean shift is ``G theta`` and the
    asymptotic information per sample is ``G Omega^+ G``.  For independent
    observations ``Omega == G`` and the usual Fisher information is recovered.
    """

    g = np.asarray(iid_information, dtype=float)
    omega = np.asarray(long_run_covariance, dtype=float)
    if g.ndim == 1:
        g = np.diag(g)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape != omega.shape:
        raise ValueError("iid information and long-run covariance must be equal square matrices")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(omega)):
        raise ValueError("information matrices must be finite")
    g = _symmetrize(g)
    omega = _symmetrize(omega)
    g_values = np.linalg.eigvalsh(g)
    o_values = np.linalg.eigvalsh(omega)
    if float(np.min(g_values)) < -1e-10 or float(np.min(o_values)) < -1e-10:
        raise ValueError("information and covariance matrices must be positive semidefinite")
    omega_pinv = np.linalg.pinv(omega, rcond=rank_tolerance, hermitian=True)
    effective = _symmetrize(g @ omega_pinv @ g)

    g_values, g_vectors = np.linalg.eigh(g)
    active = g_values > rank_tolerance * max(1.0, float(np.max(np.abs(g_values))))
    iid_rank = int(np.count_nonzero(active))
    if iid_rank == 0:
        raise ValueError("iid information has zero rank")
    whitening = g_vectors[:, active] / np.sqrt(g_values[active])[None, :]
    relative = _symmetrize(whitening.T @ effective @ whitening)
    values = np.linalg.eigvalsh(relative)[::-1]
    values = np.maximum(values, 0.0)
    visible = values > rank_tolerance * max(1.0, float(np.max(values)))
    inflation = np.full_like(values, inf)
    inflation[visible] = 1.0 / values[visible]
    finite_inflation = inflation[np.isfinite(inflation)]
    return DependentRelationInformation(
        iid_information=g,
        long_run_covariance=omega,
        effective_information=effective,
        iid_rank=iid_rank,
        effective_rank=int(np.count_nonzero(visible)),
        generalized_information_eigenvalues=values,
        sample_inflation_eigenvalues=inflation,
        worst_case_sample_inflation=(
            float(np.max(inflation)) if inflation.size else inf
        ),
        best_case_sample_inflation=(
            float(np.min(finite_inflation)) if finite_inflation.size else inf
        ),
    )


def analytic_ar1_long_run_covariance(
    covariance: ArrayLike,
    autoregression: float,
) -> np.ndarray:
    """Long-run covariance for a common scalar AR(1) correlation factor."""

    gamma0 = np.asarray(covariance, dtype=float)
    if gamma0.ndim == 1:
        gamma0 = np.diag(gamma0)
    if gamma0.ndim != 2 or gamma0.shape[0] != gamma0.shape[1]:
        raise ValueError("covariance must be square")
    phi = float(autoregression)
    if not isfinite(phi) or abs(phi) >= 1:
        raise ValueError("autoregression must lie strictly between -1 and 1")
    return _symmetrize(gamma0) * ((1.0 + phi) / (1.0 - phi))
