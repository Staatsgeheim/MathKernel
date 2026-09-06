# =============================================================================
# MathKernel - Observation visibility and connected-relation detection
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Quantitative visibility and sample cost for finite connected relations.

The basis-contraction framework separates a latent connected coefficient from
local observation-transfer coefficients.  This module turns that algebraic
separation into a finite statistical experiment.

For strictly positive finite marginals ``mu_j`` and centered, unit-``L2``
latent modes ``psi_j``, define the pure order-``d`` interaction family

``p_theta(x) = prod_j mu_j(x_j) * (1 + theta * prod_j psi_j(x_j))``.

Every proper marginal remains the product marginal, so the sole nontrivial
connected coefficient of the declared modes is ``theta``.  Under independent
local channels ``K_j(y|x)``, let

``h_j(y) = E[psi_j(X_j) | Y_j=y]``

and ``rho_j = ||h_j||_{L2(nu_j)}``.  The observed family is exactly

``q_theta(y) = prod_j nu_j(y_j) * (1 + theta * prod_j h_j(y_j))``.

Consequently:

* the best normalized product observable has connected amplitude
  ``theta * prod_j rho_j``;
* the output chi-square separation from the product null is exactly
  ``theta**2 * prod_j rho_j**2``;
* the Fisher information at ``theta=0`` is the same product; and
* if any ``rho_j`` vanishes, the declared relation is statistically
  unidentifiable through those observations.

The functions below provide exact rational certificates when possible,
weighted conditional-expectation singular systems, finite-sample lower and
upper bounds, and numerical law checks.  They do not claim that generic
multi-parameter latent models reduce to this one-parameter family.  The pure
interaction family is a controlled local model for one declared connected
relation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from math import ceil, inf, isfinite, log, log1p, prod, sqrt
from typing import Iterable, Sequence

import numpy as np

from .phylogenetic_tensor import FiniteObservationChannel, transform_joint_law
from .stochastic_koopman import FiniteJointLaw


ArrayLike = Sequence[float] | np.ndarray
RationalLike = Fraction | int | str | float


def _as_fraction(value: RationalLike) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _as_probability_vector(values: ArrayLike, *, strictly_positive: bool = True) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("probability vector must be nonempty and one-dimensional")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0):
        raise ValueError("probability vector must be finite and nonnegative")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError("probability vector must have positive mass")
    vector = vector / total
    if strictly_positive and np.any(vector <= 0):
        raise ValueError("the declared marginal must have full support")
    return vector


def _as_channel_rows(
    channel: FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    raw = channel.conditional if isinstance(channel, FiniteObservationChannel) else channel
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("channel must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("channel entries must be finite and nonnegative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-11, atol=1e-13):
        raise ValueError("every channel row must sum to one")
    return matrix


def _weighted_moments(values: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    mean = float(np.dot(probabilities, values))
    norm_squared = float(np.dot(probabilities, values * values))
    return mean, norm_squared


def standardize_centered_mode(
    mode: ArrayLike,
    marginal: ArrayLike,
    *,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Center and normalize a real finite mode in ``L2(marginal)``."""

    mu = _as_probability_vector(marginal)
    values = np.asarray(mode, dtype=float)
    if values.ndim != 1 or values.shape != mu.shape or not np.all(np.isfinite(values)):
        raise ValueError("mode must be a finite vector matching the marginal")
    centered = values - float(np.dot(mu, values))
    norm_squared = float(np.dot(mu, centered * centered))
    if norm_squared <= tolerance:
        raise ValueError("mode is constant on the marginal support")
    return centered / sqrt(norm_squared)


def _orthogonal_complement(unit_vector: np.ndarray) -> np.ndarray:
    if unit_vector.ndim != 1:
        raise ValueError("unit_vector must be one-dimensional")
    norm = float(np.linalg.norm(unit_vector))
    if not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("unit_vector must have Euclidean norm one")
    # The rows of V^T after the first span the null space of unit_vector^T.
    _, _, vt = np.linalg.svd(unit_vector.reshape(1, -1), full_matrices=True)
    return vt[1:].T


@dataclass(frozen=True, slots=True)
class ConditionalExpectationSpectrum:
    """Weighted singular system of ``psi -> E[psi(X)|Y]`` on centered modes."""

    latent_marginal: np.ndarray
    observed_marginal: np.ndarray
    centered_singular_values: np.ndarray
    latent_modes: np.ndarray
    observed_modes: np.ndarray
    maximal_correlation: float
    active_output_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ModeTransfer:
    """Mode-specific stochastic observation transfer."""

    latent_marginal: np.ndarray
    observed_marginal: np.ndarray
    latent_mode: np.ndarray
    transferred_mode: np.ndarray
    standardized_output_mode: np.ndarray
    visibility: float
    visibility_squared: float
    transfer_coefficient: float
    maximal_correlation: float
    blind: bool


@dataclass(frozen=True, slots=True)
class ExactModeTransfer:
    observed_marginal: tuple[Fraction, ...]
    transferred_mode: tuple[Fraction, ...]
    visibility_squared: Fraction


@dataclass(frozen=True, slots=True)
class ExactPureInteractionCertificate:
    order: int
    theta: Fraction
    latent_law: FiniteJointLaw
    observed_law: FiniteJointLaw
    observed_formula_law: FiniteJointLaw
    null_latent_law: FiniteJointLaw
    null_observed_law: FiniteJointLaw
    local_transfers: tuple[ExactModeTransfer, ...]
    latent_connected_cumulant: Fraction
    latent_chi_square: Fraction
    observed_chi_square: Fraction
    predicted_observed_chi_square: Fraction
    formula_matches: bool
    proper_latent_marginals_preserved: bool
    proper_observed_marginals_preserved: bool


@dataclass(frozen=True, slots=True)
class PureInteractionInformation:
    order: int
    theta: float
    local_transfers: tuple[ModeTransfer, ...]
    visibilities: tuple[float, ...]
    visibility_product: float
    connected_amplitude: float
    latent_chi_square: float
    observed_chi_square: float
    information_retention: float
    latent_fisher_at_null: float
    observed_fisher_at_null: float
    standardized_score_abs_bound: float
    blind: bool
    theta_interval: tuple[float, float]
    law_formula_error: float
    proper_latent_marginal_error: float
    proper_observed_marginal_error: float


@dataclass(frozen=True, slots=True)
class DetectionSampleBounds:
    max_error: float
    chi_square_lower_samples: int | float
    hoeffding_upper_samples: int | float
    inverse_information_scale: float
    connected_amplitude: float
    observed_information_per_sample: float
    score_abs_bound: float
    two_sided: bool


def conditional_expectation_spectrum(
    marginal: ArrayLike,
    channel: FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray,
    *,
    support_tolerance: float = 1e-15,
) -> ConditionalExpectationSpectrum:
    """Return the centered weighted singular system of a finite channel.

    If ``K[x,y] = P(Y=y|X=x)``, the weighted matrix is
    ``B[y,x] = K[x,y] * sqrt(mu[x] / nu[y])``.  On centered subspaces its
    largest singular value is the Hirschfeld-Gebelein-Renyi maximal
    correlation for the source-channel pair.
    """

    mu = _as_probability_vector(marginal)
    rows = _as_channel_rows(channel)
    if rows.shape[0] != mu.size:
        raise ValueError("channel latent alphabet does not match marginal")
    nu = mu @ rows
    active = np.flatnonzero(nu > support_tolerance)
    if active.size == 0:
        raise ValueError("channel has no output support under the marginal")
    nu_active = nu[active]
    rows_active = rows[:, active]
    weighted = (rows_active.T * np.sqrt(mu)[None, :]) / np.sqrt(nu_active)[:, None]

    latent_centered = _orthogonal_complement(np.sqrt(mu))
    output_centered = _orthogonal_complement(np.sqrt(nu_active))
    if latent_centered.shape[1] == 0 or output_centered.shape[1] == 0:
        singular = np.empty(0, dtype=float)
        latent_modes = np.empty((mu.size, 0), dtype=float)
        observed_modes_active = np.empty((active.size, 0), dtype=float)
    else:
        centered_operator = output_centered.T @ weighted @ latent_centered
        left, singular, right_t = np.linalg.svd(centered_operator, full_matrices=False)
        latent_weighted_modes = latent_centered @ right_t.T
        observed_weighted_modes = output_centered @ left
        latent_modes = latent_weighted_modes / np.sqrt(mu)[:, None]
        observed_modes_active = observed_weighted_modes / np.sqrt(nu_active)[:, None]

    observed_modes = np.zeros((rows.shape[1], observed_modes_active.shape[1]), dtype=float)
    observed_modes[active, :] = observed_modes_active
    maximal = float(singular[0]) if singular.size else 0.0
    return ConditionalExpectationSpectrum(
        latent_marginal=mu,
        observed_marginal=nu,
        centered_singular_values=singular,
        latent_modes=latent_modes,
        observed_modes=observed_modes,
        maximal_correlation=maximal,
        active_output_indices=tuple(int(index) for index in active),
    )


def mode_transfer(
    marginal: ArrayLike,
    channel: FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray,
    mode: ArrayLike,
    *,
    standardize: bool = True,
    tolerance: float = 1e-12,
) -> ModeTransfer:
    """Compute ``E[psi(X)|Y]`` and its mode-specific visibility."""

    mu = _as_probability_vector(marginal)
    rows = _as_channel_rows(channel)
    if rows.shape[0] != mu.size:
        raise ValueError("channel latent alphabet does not match marginal")
    psi = np.asarray(mode, dtype=float)
    if psi.ndim != 1 or psi.shape != mu.shape or not np.all(np.isfinite(psi)):
        raise ValueError("mode must be a finite vector matching the marginal")
    if standardize:
        psi = standardize_centered_mode(psi, mu)
    else:
        mean, norm_squared = _weighted_moments(psi, mu)
        if abs(mean) > tolerance or abs(norm_squared - 1.0) > tolerance:
            raise ValueError("mode must be centered and unit-normalized")

    nu = mu @ rows
    numerator = (mu * psi) @ rows
    transferred = np.zeros(rows.shape[1], dtype=float)
    active = nu > tolerance
    transferred[active] = numerator[active] / nu[active]
    visibility_squared = max(0.0, float(np.dot(nu, transferred * transferred)))
    visibility = sqrt(visibility_squared)
    blind = visibility <= tolerance
    standardized_output = (
        np.zeros_like(transferred) if blind else transferred / visibility
    )
    pullback = rows @ standardized_output
    coefficient = 0.0 if blind else float(np.dot(mu, psi * pullback))
    spectrum = conditional_expectation_spectrum(mu, rows)
    return ModeTransfer(
        latent_marginal=mu,
        observed_marginal=nu,
        latent_mode=psi,
        transferred_mode=transferred,
        standardized_output_mode=standardized_output,
        visibility=visibility,
        visibility_squared=visibility_squared,
        transfer_coefficient=coefficient,
        maximal_correlation=spectrum.maximal_correlation,
        blind=blind,
    )


def _fraction_probability(values: Sequence[RationalLike]) -> tuple[Fraction, ...]:
    result = tuple(_as_fraction(value) for value in values)
    if not result or any(value < 0 for value in result):
        raise ValueError("marginal must be a nonempty nonnegative vector")
    if sum(result, Fraction(0)) != 1:
        raise ValueError("marginal probabilities must sum exactly to one")
    if any(value == 0 for value in result):
        raise ValueError("the declared exact marginal must have full support")
    return result


def _validate_exact_mode(
    marginal: tuple[Fraction, ...], mode: Sequence[RationalLike]
) -> tuple[Fraction, ...]:
    psi = tuple(_as_fraction(value) for value in mode)
    if len(psi) != len(marginal):
        raise ValueError("mode and marginal cardinalities do not match")
    mean = sum((marginal[i] * psi[i] for i in range(len(psi))), Fraction(0))
    norm = sum((marginal[i] * psi[i] * psi[i] for i in range(len(psi))), Fraction(0))
    if mean != 0 or norm != 1:
        raise ValueError("exact modes must be centered and unit-normalized")
    return psi


def exact_mode_transfer(
    marginal: Sequence[RationalLike],
    channel: FiniteObservationChannel,
    mode: Sequence[RationalLike],
) -> ExactModeTransfer:
    """Exact rational conditional expectation and squared visibility."""

    mu = _fraction_probability(marginal)
    if channel.latent_states != len(mu):
        raise ValueError("channel latent alphabet does not match marginal")
    psi = _validate_exact_mode(mu, mode)
    nu = tuple(
        sum((mu[x] * channel.conditional[x][y] for x in range(len(mu))), Fraction(0))
        for y in range(channel.output_states)
    )
    transferred = tuple(
        Fraction(0)
        if nu[y] == 0
        else sum(
            (mu[x] * channel.conditional[x][y] * psi[x] for x in range(len(mu))),
            Fraction(0),
        )
        / nu[y]
        for y in range(channel.output_states)
    )
    visibility_squared = sum(
        (nu[y] * transferred[y] * transferred[y] for y in range(len(nu))),
        Fraction(0),
    )
    return ExactModeTransfer(nu, transferred, visibility_squared)


def _exact_product_law(marginals: Sequence[tuple[Fraction, ...]]) -> FiniteJointLaw:
    sizes = tuple(len(mu) for mu in marginals)
    masses = {
        state: prod((marginals[j][state[j]] for j in range(len(sizes))), start=Fraction(1))
        for state in product(*(range(size) for size in sizes))
    }
    return FiniteJointLaw(sizes, masses)


def _exact_theta_interval(modes: Sequence[tuple[Fraction, ...]]) -> tuple[Fraction | None, Fraction | None]:
    lower: Fraction | None = None
    upper: Fraction | None = None
    for state in product(*(range(len(mode)) for mode in modes)):
        value = prod((modes[j][state[j]] for j in range(len(modes))), start=Fraction(1))
        if value > 0:
            candidate = -Fraction(1, 1) / value
            lower = candidate if lower is None or candidate > lower else lower
        elif value < 0:
            candidate = -Fraction(1, 1) / value
            upper = candidate if upper is None or candidate < upper else upper
    return lower, upper


def exact_pure_interaction_law(
    marginals: Sequence[Sequence[RationalLike]],
    modes: Sequence[Sequence[RationalLike]],
    theta: RationalLike,
) -> FiniteJointLaw:
    """Construct the exact rational pure connected-interaction law."""

    mus = tuple(_fraction_probability(mu) for mu in marginals)
    if not mus or len(mus) != len(modes):
        raise ValueError("one mode is required for every nonempty marginal")
    psis = tuple(_validate_exact_mode(mu, mode) for mu, mode in zip(mus, modes, strict=True))
    parameter = _as_fraction(theta)
    lower, upper = _exact_theta_interval(psis)
    if lower is not None and parameter < lower:
        raise ValueError("theta lies below the nonnegativity interval")
    if upper is not None and parameter > upper:
        raise ValueError("theta lies above the nonnegativity interval")
    sizes = tuple(len(mu) for mu in mus)
    masses: dict[tuple[int, ...], Fraction] = {}
    for state in product(*(range(size) for size in sizes)):
        base = prod((mus[j][state[j]] for j in range(len(sizes))), start=Fraction(1))
        interaction = prod((psis[j][state[j]] for j in range(len(sizes))), start=Fraction(1))
        mass = base * (1 + parameter * interaction)
        if mass < 0:
            raise RuntimeError("internal nonnegativity failure")
        if mass:
            masses[state] = mass
    return FiniteJointLaw(sizes, masses)


def _exact_observed_formula_law(
    transfers: Sequence[ExactModeTransfer], theta: Fraction
) -> FiniteJointLaw:
    sizes = tuple(len(item.observed_marginal) for item in transfers)
    masses: dict[tuple[int, ...], Fraction] = {}
    for state in product(*(range(size) for size in sizes)):
        base = prod(
            (transfers[j].observed_marginal[state[j]] for j in range(len(sizes))),
            start=Fraction(1),
        )
        interaction = prod(
            (transfers[j].transferred_mode[state[j]] for j in range(len(sizes))),
            start=Fraction(1),
        )
        mass = base * (1 + theta * interaction)
        if mass:
            masses[state] = mass
    return FiniteJointLaw(sizes, masses)


def exact_chi_square(law: FiniteJointLaw, reference: FiniteJointLaw) -> Fraction:
    """Pearson chi-square divergence ``sum (p-q)^2/q`` exactly."""

    if law.state_sizes != reference.state_sizes:
        raise ValueError("laws must share state cardinalities")
    result = Fraction(0)
    for state in product(*(range(size) for size in law.state_sizes)):
        p = law.probabilities.get(state, Fraction(0))
        q = reference.probabilities.get(state, Fraction(0))
        if q == 0:
            if p:
                raise ValueError("law is not absolutely continuous with respect to reference")
            continue
        result += (p - q) * (p - q) / q
    return result


def _law_equal(left: FiniteJointLaw, right: FiniteJointLaw) -> bool:
    return left.state_sizes == right.state_sizes and left.probabilities == right.probabilities


def _proper_marginals_match_product(
    law: FiniteJointLaw,
    marginals: Sequence[tuple[Fraction, ...]],
) -> bool:
    dimension = law.dimension
    if dimension < 2:
        return True
    for size in range(1, dimension):
        for coordinates in combinations(range(dimension), size):
            expected = _exact_product_law(tuple(marginals[index] for index in coordinates))
            if not _law_equal(law.marginal(coordinates), expected):
                return False
    return True


def exact_pure_interaction_certificate(
    marginals: Sequence[Sequence[RationalLike]],
    modes: Sequence[Sequence[RationalLike]],
    channels: Sequence[FiniteObservationChannel],
    theta: RationalLike,
) -> ExactPureInteractionCertificate:
    """Build an exact certificate for transfer, connectedness, and chi-square loss."""

    mus = tuple(_fraction_probability(mu) for mu in marginals)
    if len(channels) != len(mus) or len(modes) != len(mus):
        raise ValueError("marginals, modes, and channels must have equal length")
    psis = tuple(_validate_exact_mode(mu, mode) for mu, mode in zip(mus, modes, strict=True))
    parameter = _as_fraction(theta)
    latent = exact_pure_interaction_law(mus, psis, parameter)
    latent_null = _exact_product_law(mus)
    transfers = tuple(
        exact_mode_transfer(mu, channel, psi)
        for mu, channel, psi in zip(mus, channels, psis, strict=True)
    )
    observed = transform_joint_law(latent, channels)
    observed_null = transform_joint_law(latent_null, channels)
    observed_formula = _exact_observed_formula_law(transfers, parameter)
    observed_marginals = tuple(item.observed_marginal for item in transfers)
    latent_chi = exact_chi_square(latent, latent_null)
    observed_chi = exact_chi_square(observed, observed_null)
    predicted = parameter * parameter * prod(
        (item.visibility_squared for item in transfers), start=Fraction(1)
    )
    return ExactPureInteractionCertificate(
        order=len(mus),
        theta=parameter,
        latent_law=latent,
        observed_law=observed,
        observed_formula_law=observed_formula,
        null_latent_law=latent_null,
        null_observed_law=observed_null,
        local_transfers=transfers,
        latent_connected_cumulant=latent.cumulant(psis),
        latent_chi_square=latent_chi,
        observed_chi_square=observed_chi,
        predicted_observed_chi_square=predicted,
        formula_matches=_law_equal(observed, observed_formula),
        proper_latent_marginals_preserved=_proper_marginals_match_product(latent, mus),
        proper_observed_marginals_preserved=_proper_marginals_match_product(
            observed, observed_marginals
        ),
    )


def _numeric_theta_interval(modes: Sequence[np.ndarray]) -> tuple[float, float]:
    lower = -inf
    upper = inf
    for state in product(*(range(mode.size) for mode in modes)):
        value = float(prod((modes[j][state[j]] for j in range(len(modes)))))
        if value > 0:
            lower = max(lower, -1.0 / value)
        elif value < 0:
            upper = min(upper, -1.0 / value)
    return lower, upper


def _outer_product_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray:
    result = np.asarray(1.0)
    for vector in vectors:
        result = np.multiply.outer(result, vector)
    return np.asarray(result)


def numerical_pure_interaction_law(
    marginals: Sequence[ArrayLike],
    modes: Sequence[ArrayLike],
    theta: float,
) -> np.ndarray:
    """Construct a floating-point pure interaction tensor."""

    mus = tuple(_as_probability_vector(mu) for mu in marginals)
    if not mus or len(mus) != len(modes):
        raise ValueError("one mode is required for every nonempty marginal")
    psis = tuple(
        standardize_centered_mode(mode, mu)
        for mu, mode in zip(mus, modes, strict=True)
    )
    if not isfinite(theta):
        raise ValueError("theta must be finite")
    lower, upper = _numeric_theta_interval(psis)
    if theta < lower - 1e-12 or theta > upper + 1e-12:
        raise ValueError("theta lies outside the nonnegativity interval")
    base = _outer_product_vectors(mus)
    interaction = _outer_product_vectors(psis)
    law = base * (1.0 + float(theta) * interaction)
    if float(law.min()) < -1e-12:
        raise RuntimeError("internal nonnegativity failure")
    law = np.maximum(law, 0.0)
    law /= float(law.sum())
    return law


def push_joint_law_through_local_channels(
    law: np.ndarray,
    channels: Sequence[FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray],
) -> np.ndarray:
    """Push a dense joint-law tensor through independent local channels."""

    result = np.asarray(law, dtype=float)
    if result.ndim != len(channels) or result.ndim == 0:
        raise ValueError("one channel is required per law coordinate")
    if not np.all(np.isfinite(result)) or np.any(result < -1e-14):
        raise ValueError("law must be finite and nonnegative")
    for axis, channel in enumerate(channels):
        rows = _as_channel_rows(channel)
        if rows.shape[0] != result.shape[axis]:
            raise ValueError("channel latent alphabet does not match law coordinate")
        result = np.tensordot(result, rows, axes=([axis], [0]))
        result = np.moveaxis(result, -1, axis)
    result = np.maximum(result, 0.0)
    result /= float(result.sum())
    return result


def numerical_chi_square(law: np.ndarray, reference: np.ndarray) -> float:
    p = np.asarray(law, dtype=float)
    q = np.asarray(reference, dtype=float)
    if p.shape != q.shape or p.ndim == 0:
        raise ValueError("laws must be nonempty tensors of equal shape")
    if np.any(q <= 0) or np.any(p < 0):
        raise ValueError("reference must be strictly positive and law nonnegative")
    return float(np.sum((p - q) ** 2 / q))


def _proper_marginal_error(law: np.ndarray, marginals: Sequence[np.ndarray]) -> float:
    dimension = law.ndim
    maximum = 0.0
    axes = tuple(range(dimension))
    for size in range(1, dimension):
        for coordinates in combinations(axes, size):
            summed = tuple(axis for axis in axes if axis not in coordinates)
            actual = law.sum(axis=summed) if summed else law
            expected = _outer_product_vectors(tuple(marginals[index] for index in coordinates))
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return maximum


def pure_interaction_information(
    marginals: Sequence[ArrayLike],
    modes: Sequence[ArrayLike],
    channels: Sequence[FiniteObservationChannel | Sequence[Sequence[float]] | np.ndarray],
    theta: float,
    *,
    verify_law: bool = True,
) -> PureInteractionInformation:
    """Analyze one pure connected relation under independent local channels."""

    mus = tuple(_as_probability_vector(mu) for mu in marginals)
    if not mus or len(modes) != len(mus) or len(channels) != len(mus):
        raise ValueError("marginals, modes, and channels must have equal nonzero length")
    psis = tuple(
        standardize_centered_mode(mode, mu)
        for mu, mode in zip(mus, modes, strict=True)
    )
    if not isfinite(float(theta)):
        raise ValueError("theta must be finite")
    interval = _numeric_theta_interval(psis)
    if theta < interval[0] - 1e-12 or theta > interval[1] + 1e-12:
        raise ValueError("theta lies outside the nonnegativity interval")
    transfers = tuple(
        mode_transfer(mu, channel, psi, standardize=False)
        for mu, channel, psi in zip(mus, channels, psis, strict=True)
    )
    visibilities = tuple(item.visibility for item in transfers)
    visibility_product = float(prod(visibilities))
    blind = any(item.blind for item in transfers)
    amplitude = float(theta) * visibility_product
    latent_chi = float(theta) ** 2
    observed_chi = latent_chi * visibility_product * visibility_product
    score_bound = (
        inf
        if blind
        else float(
            prod(
                float(np.max(np.abs(item.standardized_output_mode)))
                for item in transfers
            )
        )
    )

    formula_error = 0.0
    latent_marginal_error = 0.0
    observed_marginal_error = 0.0
    if verify_law:
        latent = numerical_pure_interaction_law(mus, psis, theta)
        latent_null = _outer_product_vectors(mus)
        observed = push_joint_law_through_local_channels(latent, channels)
        nus = tuple(item.observed_marginal for item in transfers)
        hs = tuple(item.transferred_mode for item in transfers)
        observed_null = _outer_product_vectors(nus)
        observed_formula = observed_null * (1.0 + float(theta) * _outer_product_vectors(hs))
        formula_error = float(np.max(np.abs(observed - observed_formula)))
        latent_marginal_error = _proper_marginal_error(latent, mus)
        observed_marginal_error = _proper_marginal_error(observed, nus)
        direct_chi = numerical_chi_square(observed, observed_null)
        if not np.isclose(direct_chi, observed_chi, rtol=2e-10, atol=2e-12):
            raise RuntimeError("observed chi-square factorization failed")
        if not np.isclose(numerical_chi_square(latent, latent_null), latent_chi, rtol=2e-10, atol=2e-12):
            raise RuntimeError("latent chi-square normalization failed")

    return PureInteractionInformation(
        order=len(mus),
        theta=float(theta),
        local_transfers=transfers,
        visibilities=visibilities,
        visibility_product=visibility_product,
        connected_amplitude=amplitude,
        latent_chi_square=latent_chi,
        observed_chi_square=observed_chi,
        information_retention=visibility_product * visibility_product,
        latent_fisher_at_null=1.0,
        observed_fisher_at_null=visibility_product * visibility_product,
        standardized_score_abs_bound=score_bound,
        blind=blind,
        theta_interval=interval,
        law_formula_error=formula_error,
        proper_latent_marginal_error=latent_marginal_error,
        proper_observed_marginal_error=observed_marginal_error,
    )


def pure_relation_sample_bounds(
    information: PureInteractionInformation,
    *,
    max_error: float = 0.05,
    two_sided: bool = False,
) -> DetectionSampleBounds:
    """Necessary chi-square and sufficient Hoeffding sample bounds.

    The lower bound applies to any test with both type-I and type-II error at
    most ``max_error``.  The upper bound applies to the empirical normalized
    product score, with known sign unless ``two_sided`` is true.
    """

    delta = float(max_error)
    if not 0 < delta < 0.5:
        raise ValueError("max_error must lie strictly between zero and one half")
    v = float(information.observed_chi_square)
    amplitude = abs(float(information.connected_amplitude))
    lower: int | float
    if v == 0:
        lower = inf
    else:
        required_chi = 4.0 * (1.0 - 2.0 * delta) ** 2
        lower = int(ceil(log1p(required_chi) / log1p(v)))
    inverse_scale = inf if v == 0 else 1.0 / v

    bound = float(information.standardized_score_abs_bound)
    upper: int | float
    if amplitude == 0 or not isfinite(bound):
        upper = inf
    else:
        logarithm = log((2.0 if two_sided else 1.0) / delta)
        upper = int(ceil(8.0 * bound * bound * logarithm / (amplitude * amplitude)))
    return DetectionSampleBounds(
        max_error=delta,
        chi_square_lower_samples=lower,
        hoeffding_upper_samples=upper,
        inverse_information_scale=inverse_scale,
        connected_amplitude=float(information.connected_amplitude),
        observed_information_per_sample=float(information.observed_fisher_at_null),
        score_abs_bound=bound,
        two_sided=bool(two_sided),
    )


def binary_symmetric_channel(error_probability: RationalLike) -> FiniteObservationChannel:
    """Return a binary symmetric observation channel."""

    error = _as_fraction(error_probability)
    if error < 0 or error > 1:
        raise ValueError("error probability must lie in [0,1]")
    return FiniteObservationChannel.from_rows(
        ((1 - error, error), (error, 1 - error)),
        name=f"BSC({error})",
    )


def binary_parity_information(
    order: int,
    theta: float,
    *,
    error_probability: float | Sequence[float] = 0.0,
    verify_law: bool = True,
) -> PureInteractionInformation:
    """Pure parity interaction observed through local binary symmetric channels."""

    d = int(order)
    if d < 2:
        raise ValueError("order must be at least two")
    if isinstance(error_probability, (str, bytes)) or np.isscalar(error_probability):
        errors = (float(error_probability),) * d
    else:
        try:
            errors = tuple(float(value) for value in error_probability)
        except TypeError as exc:
            raise ValueError("error_probability must be a scalar or finite sequence") from exc
        if len(errors) != d:
            raise ValueError("one error probability is required per coordinate")
    channels = tuple(binary_symmetric_channel(value) for value in errors)
    return pure_interaction_information(
        ((0.5, 0.5),) * d,
        ((-1.0, 1.0),) * d,
        channels,
        theta,
        verify_law=verify_law,
    )


def joint_conditionally_independent_channel(
    channels: Sequence[FiniteObservationChannel],
    *,
    name: str = "joint observation",
) -> FiniteObservationChannel:
    """Public convenience wrapper for conditionally independent sensor fusion."""

    return FiniteObservationChannel.jointly_conditionally_independent(channels, name=name)
