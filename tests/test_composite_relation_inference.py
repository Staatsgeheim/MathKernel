from __future__ import annotations

from math import inf

import numpy as np
import pytest

from mathkernel.composite_relation_inference import (
    bootstrap_eigenspace_regions,
    composite_detection_bounds,
    composite_score_geometry,
    composite_score_test,
    estimate_circular_block_length,
    estimate_relation_block_length,
    nuisance_adjusted_score_geometry,
    nuisance_composite_score_geometry,
    robust_composite_detection_bound,
    studentized_bootstrap_relation_spectrum,
)
from mathkernel.information_geometry import score_subspace_geometry


def _two_bit_scores() -> np.ndarray:
    return np.asarray(
        (
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
        )
    )


def _two_bit_channel(error: float = 0.1) -> np.ndarray:
    reliability = 1.0 - 2.0 * error
    # Independent symmetric bit errors in the state order 00,01,10,11.
    rows = np.zeros((4, 4), dtype=float)
    for state in range(4):
        b1, b2 = divmod(state, 2)
        for output in range(4):
            y1, y2 = divmod(output, 2)
            p1 = 1.0 - error if y1 == b1 else error
            p2 = 1.0 - error if y2 == b2 else error
            rows[state, output] = p1 * p2
    assert reliability > 0
    return rows


def test_composite_scores_are_centered_and_whitened() -> None:
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    composite = composite_score_geometry(geometry)
    assert composite.visible_rank == 3
    assert composite.covariance_residual < 1e-12
    assert np.allclose(composite.observed_law @ composite.standardized_scores, 0.0)
    covariance = composite.standardized_scores.T @ (
        composite.observed_law[:, None] * composite.standardized_scores
    )
    assert np.allclose(covariance, np.eye(3), atol=1e-12)


def test_nominal_composite_test_matches_score_norm() -> None:
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    composite = composite_score_geometry(geometry)
    labels = np.asarray((0, 0, 1, 2, 3, 3, 3, 2, 1, 0), dtype=np.int64)
    result = composite_score_test(labels, composite, covariance_mode="nominal")
    expected = labels.size * np.dot(result.score_mean, result.score_mean)
    assert result.statistic == pytest.approx(expected)
    assert result.degrees_of_freedom == 3
    assert result.p_value is not None


def test_empirical_and_hac_tests_return_valid_results() -> None:
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    rng = np.random.default_rng(7)
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    composite = composite_score_geometry(geometry)
    labels = rng.choice(4, size=1200, p=composite.observed_law)
    empirical = composite_score_test(labels, composite, covariance_mode="empirical")
    hac = composite_score_test(labels, composite, covariance_mode="hac", max_lag=15)
    assert empirical.statistic >= 0
    assert hac.statistic >= 0
    assert empirical.covariance_rank == hac.covariance_rank == 3
    assert hac.max_lag == 15


def test_composite_bound_is_finite_and_brackets_worst_direction_scale() -> None:
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    result = composite_detection_bounds(geometry, 0.15)
    assert result.relation_dimension == 3
    assert result.weakest_information_retention == pytest.approx(0.4096)
    assert result.chi_square_necessary_samples < result.asymptotic_composite_samples
    assert result.asymptotic_composite_samples < result.conservative_composite_sufficient_samples


def test_composite_bound_detects_blind_direction() -> None:
    p = np.full(4, 0.25)
    bit1 = np.asarray(((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)))
    geometry = score_subspace_geometry(p, bit1, _two_bit_scores())
    result = composite_detection_bounds(geometry, 0.1)
    assert result.exact_blind_direction
    assert result.chi_square_necessary_samples == inf
    assert result.asymptotic_composite_samples == inf
    assert result.conservative_composite_sufficient_samples == inf


def test_dimension_aware_bound_increases_with_extra_visible_dimension() -> None:
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    p = np.full(4, 0.25)
    full = score_subspace_geometry(p, _two_bit_channel(), _two_bit_scores())
    single = score_subspace_geometry(p, _two_bit_channel(), _two_bit_scores()[:, :1])
    full_bound = composite_detection_bounds(full, 0.15)
    single_bound = composite_detection_bounds(single, 0.15)
    assert full_bound.conservative_composite_sufficient_samples > single_bound.conservative_composite_sufficient_samples


def test_nuisance_adjustment_removes_observed_confounding_exactly() -> None:
    p = np.full(4, 0.25)
    # Channel exposes state 0 versus the merged set {1,2,3}.
    channel = np.asarray(((1.0, 0.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    target = np.asarray((1.0, 1.0, -1.0, -1.0))[:, None]
    nuisance = np.asarray((1.0, -1.0, 1.0, -1.0))[:, None]
    isolated = score_subspace_geometry(p, channel, target)
    adjusted = nuisance_adjusted_score_geometry(p, channel, target, nuisance)
    assert isolated.generalized_information_eigenvalues[0] > 0
    assert adjusted.latent_efficient_rank == 1
    assert adjusted.observed_efficient_rank == 0
    assert adjusted.minimum_information_retention == pytest.approx(0.0, abs=1e-14)


def test_nuisance_adjustment_is_invariant_to_nuisance_scaling() -> None:
    rng = np.random.default_rng(12)
    p = rng.dirichlet(np.ones(6))
    channel = rng.dirichlet(np.ones(4), size=6)
    target = rng.normal(size=(6, 2))
    nuisance = rng.normal(size=(6, 2))
    first = nuisance_adjusted_score_geometry(p, channel, target, nuisance)
    second = nuisance_adjusted_score_geometry(p, channel, target, nuisance @ np.asarray(((2.0, 0.5), (0.0, -1.5))))
    assert np.allclose(
        first.generalized_information_eigenvalues,
        second.generalized_information_eigenvalues,
        atol=5e-11,
    )


def test_nuisance_composite_scores_whiten_efficient_information() -> None:
    p = np.full(4, 0.25)
    target = _two_bit_scores()[:, :2]
    nuisance = _two_bit_scores()[:, 2:]
    adjusted = nuisance_adjusted_score_geometry(p, _two_bit_channel(), target, nuisance)
    composite = nuisance_composite_score_geometry(adjusted)
    assert composite.visible_rank == 2
    assert composite.covariance_residual < 1e-12


def test_robust_bound_becomes_infinite_when_uncertainty_balls_overlap() -> None:
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    composite = composite_score_geometry(geometry)
    separated = robust_composite_detection_bound(composite, 0.4, 0.1)
    overlapping = robust_composite_detection_bound(composite, 0.4, 0.2)
    assert separated.robustly_separated
    assert separated.conservative_sufficient_samples < inf
    assert not overlapping.robustly_separated
    assert overlapping.conservative_sufficient_samples == inf


def test_eigenspace_region_handles_repeated_top_cluster() -> None:
    rng = np.random.default_rng(91)
    n = 2500
    latent = rng.integers(0, 4, size=n)
    channel = _two_bit_channel()
    cumulative = np.cumsum(channel, axis=1)
    observed = np.sum(rng.random(n)[:, None] > cumulative[latent], axis=1)
    scores = _two_bit_scores()[latent]
    result = bootstrap_eigenspace_regions(
        scores,
        observed,
        clusters=((0, 1), (2,)),
        n_bootstrap=60,
        seed=8,
    )
    assert result.successful_replicates == 60
    assert result.clusters[0].indices == (0, 1)
    assert 0 <= result.clusters[0].principal_sine_radius <= 1
    assert result.clusters[0].frobenius_projector_radius >= result.clusters[0].principal_sine_radius


def test_eigenspace_region_is_invariant_to_relation_basis_change() -> None:
    rng = np.random.default_rng(19)
    n = 1800
    latent = rng.integers(0, 4, size=n)
    channel = _two_bit_channel()
    observed = np.sum(rng.random(n)[:, None] > np.cumsum(channel, axis=1)[latent], axis=1)
    scores = _two_bit_scores()[latent]
    change = np.asarray(((1.0, 0.2, 0.0), (0.0, 1.3, 0.1), (0.1, 0.0, 0.8)))
    first = bootstrap_eigenspace_regions(
        scores, observed, clusters=((0, 1),), n_bootstrap=40, seed=22
    )
    second = bootstrap_eigenspace_regions(
        scores @ change, observed, clusters=((0, 1),), n_bootstrap=40, seed=22
    )
    assert first.clusters[0].principal_sine_radius == pytest.approx(
        second.clusters[0].principal_sine_radius, abs=2e-8
    )


def test_block_length_responds_to_serial_dependence() -> None:
    rng = np.random.default_rng(4)
    iid = rng.normal(size=(5000, 2))
    dependent = np.empty((5000, 2))
    dependent[0] = rng.normal(size=2)
    for i in range(1, dependent.shape[0]):
        dependent[i] = 0.9 * dependent[i - 1] + rng.normal(scale=np.sqrt(1 - 0.9**2), size=2)
    assert estimate_circular_block_length(dependent) > estimate_circular_block_length(iid)


def test_relation_block_length_uses_labels_and_scores() -> None:
    rng = np.random.default_rng(5)
    labels = np.empty(1200, dtype=np.int64)
    labels[0] = 0
    for i in range(1, labels.size):
        labels[i] = labels[i - 1] if rng.random() < 0.9 else 1 - labels[i - 1]
    scores = np.where(labels[:, None] == 0, -1.0, 1.0)
    length = estimate_relation_block_length(scores, labels)
    assert 2 <= length <= labels.size // 4


def test_studentized_bootstrap_returns_ordered_bounded_intervals() -> None:
    rng = np.random.default_rng(2)
    n = 1200
    latent = rng.integers(0, 2, size=n)
    observed = np.bitwise_xor(latent, (rng.random(n) < 0.2).astype(np.int64))
    scores = np.where(latent[:, None] == 0, -1.0, 1.0)
    result = studentized_bootstrap_relation_spectrum(
        scores,
        observed,
        n_bootstrap=40,
        jackknife_groups=6,
        seed=3,
    )
    assert result.successful_replicates == 40
    assert np.all(result.lower_information_eigenvalues >= 0)
    assert np.all(result.upper_information_eigenvalues <= 1)
    assert np.all(result.lower_information_eigenvalues <= result.upper_information_eigenvalues)


def test_studentized_moving_block_chooses_automatic_length() -> None:
    rng = np.random.default_rng(21)
    n = 900
    latent = np.empty(n, dtype=np.int64)
    latent[0] = rng.integers(0, 2)
    for i in range(1, n):
        latent[i] = latent[i - 1] if rng.random() < 0.9 else 1 - latent[i - 1]
    observed = np.bitwise_xor(latent, (rng.random(n) < 0.2).astype(np.int64))
    scores = np.where(latent[:, None] == 0, -1.0, 1.0)
    result = studentized_bootstrap_relation_spectrum(
        scores,
        observed,
        n_bootstrap=30,
        method="moving_block",
        jackknife_groups=5,
        seed=10,
    )
    assert result.block_length is not None
    assert result.block_length > 2


def test_composite_score_test_rejects_out_of_range_labels() -> None:
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    with pytest.raises(ValueError, match="outside"):
        composite_score_test((0, 1, 4), composite_score_geometry(geometry))


def test_invalid_robust_bound_fails_closed() -> None:
    geometry = score_subspace_geometry(
        np.full(4, 0.25), _two_bit_channel(), _two_bit_scores()
    )
    composite = composite_score_geometry(geometry)
    with pytest.raises(ValueError, match="nonnegative"):
        robust_composite_detection_bound(composite, -0.1, 0.0)


def test_invalid_nuisance_tangent_fails_closed() -> None:
    with pytest.raises(ValueError, match="target tangent"):
        from mathkernel.composite_relation_inference import nuisance_adjusted_tangent_geometry

        nuisance_adjusted_tangent_geometry(
            (0.5, 0.5), np.eye(2), ((1.0,), (0.0,)), ((-0.5,), (0.5,))
        )
