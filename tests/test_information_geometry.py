from __future__ import annotations

from math import comb, inf

import numpy as np
import pytest

from mathkernel import (
    binary_symmetric_channel,
    bootstrap_empirical_relation_spectrum,
    direction_testing_bounds,
    mixture_chart_jacobian,
    relation_subspace_transfer,
    score_subspace_geometry,
    scores_to_tangents,
    simplex_fisher_inner,
    simplex_tangent_geometry,
    softmax_chart_jacobian,
    subspace_minimax_bounds,
    tangent_information_retention,
    tangents_to_scores,
)


def _random_channel(rng: np.random.Generator, n: int, m: int) -> np.ndarray:
    return rng.dirichlet(np.ones(m), size=n)


def _bayes_error_bernoulli_product(p: float, q: float, n: int) -> float:
    total = 0.0
    for k in range(n + 1):
        left = comb(n, k) * p**k * (1 - p) ** (n - k)
        right = comb(n, k) * q**k * (1 - q) ** (n - k)
        total += min(left, right)
    return 0.5 * total


def _minimum_n_for_error(p: float, q: float, target: float, limit: int = 100_000) -> int:
    for n in range(1, limit + 1):
        if _bayes_error_bernoulli_product(p, q, n) <= target:
            return n
    raise AssertionError("search limit too small")


def test_score_and_tangent_geometries_are_identical() -> None:
    rng = np.random.default_rng(20260906)
    p = rng.dirichlet(np.ones(6))
    channel = _random_channel(rng, 6, 4)
    scores = rng.normal(size=(6, 3))
    score_transfer = relation_subspace_transfer(p, channel, scores)
    tangent_geometry = score_subspace_geometry(p, channel, scores)
    assert np.allclose(score_transfer.latent_gram, tangent_geometry.latent_metric)
    assert np.allclose(score_transfer.observed_gram, tangent_geometry.observed_metric)
    assert np.allclose(
        score_transfer.generalized_information_eigenvalues,
        tangent_geometry.generalized_information_eigenvalues,
    )


def test_score_tangent_round_trip() -> None:
    p = np.asarray((0.1, 0.2, 0.3, 0.4))
    scores = np.asarray(((1.0, 2.0), (-1.0, 0.5), (0.2, -0.4), (0.7, 1.2)))
    tangents = scores_to_tangents(p, scores)
    restored = tangents_to_scores(p, tangents)
    centered = scores - np.outer(np.ones(4), p @ scores)
    assert np.allclose(restored, centered)
    assert np.allclose(tangents.sum(axis=0), 0.0)


def test_fisher_inner_matches_score_variance() -> None:
    p = np.asarray((0.2, 0.3, 0.5))
    score = np.asarray((-1.2, 0.4, 0.24))
    score -= p @ score
    tangent = p * score
    assert simplex_fisher_inner(p, tangent, tangent) == pytest.approx(
        float(np.dot(p, score * score))
    )


def test_mixture_and_softmax_charts_give_same_intrinsic_spectrum() -> None:
    p = np.asarray((0.11, 0.19, 0.27, 0.43))
    channel = np.asarray(
        (
            (0.75, 0.15, 0.10),
            (0.20, 0.65, 0.15),
            (0.10, 0.20, 0.70),
            (0.35, 0.25, 0.40),
        )
    )
    mixture = simplex_tangent_geometry(p, channel, mixture_chart_jacobian(4))
    softmax = simplex_tangent_geometry(p, channel, softmax_chart_jacobian(p))
    assert np.allclose(
        mixture.generalized_information_eigenvalues,
        softmax.generalized_information_eigenvalues,
        atol=2e-12,
    )
    assert mixture.latent_rank == softmax.latent_rank == 3


def test_tangent_basis_change_is_coordinate_invariant() -> None:
    rng = np.random.default_rng(11)
    p = rng.dirichlet(np.ones(5))
    channel = _random_channel(rng, 5, 4)
    frame = mixture_chart_jacobian(5)
    change = rng.normal(size=(4, 4))
    while abs(np.linalg.det(change)) < 0.1:
        change = rng.normal(size=(4, 4))
    first = simplex_tangent_geometry(p, channel, frame)
    second = simplex_tangent_geometry(p, channel, frame @ change)
    assert np.allclose(
        first.generalized_information_eigenvalues,
        second.generalized_information_eigenvalues,
        atol=3e-12,
    )


def test_full_state_permutation_preserves_spectrum() -> None:
    rng = np.random.default_rng(8)
    p = rng.dirichlet(np.ones(5))
    channel = _random_channel(rng, 5, 3)
    frame = mixture_chart_jacobian(5)
    permutation = np.asarray((3, 0, 4, 1, 2))
    first = simplex_tangent_geometry(p, channel, frame)
    second = simplex_tangent_geometry(
        p[permutation], channel[permutation], frame[permutation]
    )
    assert np.allclose(
        first.generalized_information_eigenvalues,
        second.generalized_information_eigenvalues,
        atol=2e-12,
    )


def test_data_processing_holds_for_random_tangent_subspaces() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        n = int(rng.integers(2, 9))
        m = int(rng.integers(2, 8))
        rank = int(rng.integers(1, n))
        p = rng.dirichlet(np.ones(n))
        channel = _random_channel(rng, n, m)
        frame = rng.normal(size=(n, rank))
        frame -= np.outer(np.ones(n), frame.mean(axis=0))
        geometry = simplex_tangent_geometry(p, channel, frame)
        assert np.min(geometry.generalized_information_eigenvalues) >= -1e-13
        assert np.max(geometry.generalized_information_eigenvalues) <= 1 + 1e-13
        difference = geometry.latent_metric - geometry.observed_metric
        assert np.min(np.linalg.eigvalsh(0.5 * (difference + difference.T))) > -2e-10


def test_directional_chi_square_is_exact() -> None:
    p = np.asarray((0.5, 0.5))
    channel = np.asarray(((0.8, 0.2), (0.2, 0.8)))
    frame = scores_to_tangents(p, (-1.0, 1.0))
    geometry = simplex_tangent_geometry(p, channel, frame)
    result = direction_testing_bounds(geometry, (1.0,), 0.3)
    alternative = p + result.epsilon * result.latent_tangent
    observed = alternative @ channel
    baseline = p @ channel
    direct = float(np.sum((observed - baseline) ** 2 / baseline))
    assert direct == pytest.approx(result.chi_square_per_sample, abs=1e-15)
    assert result.information_retention == pytest.approx(0.6**2)
    assert tangent_information_retention(geometry, (2.0,)) == pytest.approx(0.6**2)


def test_finite_sample_lower_and_upper_bounds_bracket_exact_requirement() -> None:
    p = np.asarray((0.5, 0.5))
    channel = np.asarray(((0.8, 0.2), (0.2, 0.8)))
    frame = scores_to_tangents(p, (-1.0, 1.0))
    geometry = simplex_tangent_geometry(p, channel, frame)
    result = direction_testing_bounds(geometry, (1.0,), 0.4, max_error=0.1)
    q0 = float(geometry.observed_law[1])
    q1 = float((geometry.observed_law + result.epsilon * result.observed_tangent)[1])
    exact_n = _minimum_n_for_error(q0, q1, 0.1)
    assert result.chi_square_necessary_samples <= exact_n
    assert exact_n <= result.bhattacharyya_sufficient_samples
    assert exact_n <= result.retention_only_sufficient_samples


def test_blind_direction_has_infinite_sample_requirement() -> None:
    p = np.full(4, 0.25)
    bit1 = np.asarray(((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)))
    scores = np.asarray(((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)))
    geometry = score_subspace_geometry(p, bit1, scores)
    bounds = subspace_minimax_bounds(geometry, 0.25)
    assert bounds.exact_blind_direction
    assert bounds.weakest_information_retention == pytest.approx(0.0)
    assert bounds.chi_square_necessary_samples == inf
    assert bounds.uniform_retention_sufficient_samples == inf


def test_minimax_bound_uses_smallest_generalized_eigenvalue() -> None:
    p = np.full(4, 0.25)
    channel = np.asarray(
        (
            (0.81, 0.09, 0.09, 0.01),
            (0.09, 0.81, 0.01, 0.09),
            (0.09, 0.01, 0.81, 0.09),
            (0.01, 0.09, 0.09, 0.81),
        )
    )
    scores = np.asarray(((-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
    geometry = score_subspace_geometry(p, channel, scores)
    bounds = subspace_minimax_bounds(geometry, 0.2)
    assert bounds.weakest_information_retention == pytest.approx(
        geometry.generalized_information_eigenvalues[-1]
    )
    assert bounds.weakest_information_retention == pytest.approx(0.4096)
    assert bounds.chi_square_necessary_samples < inf


def test_uniform_positivity_radius_is_chart_invariant() -> None:
    p = np.asarray((0.15, 0.25, 0.60))
    channel = np.eye(3)
    first = simplex_tangent_geometry(p, channel, mixture_chart_jacobian(3))
    second = simplex_tangent_geometry(p, channel, softmax_chart_jacobian(p))
    assert first.uniform_latent_positivity_radius == pytest.approx(
        second.uniform_latent_positivity_radius
    )


def test_bootstrap_empirical_spectrum_recovers_known_channel() -> None:
    rng = np.random.default_rng(20260906)
    n = 12_000
    latent = rng.integers(0, 2, size=n)
    noise = rng.random(n) < 0.2
    observed = np.bitwise_xor(latent, noise.astype(np.int64))
    scores = np.where(latent[:, None] == 0, -1.0, 1.0)
    result = bootstrap_empirical_relation_spectrum(
        scores,
        observed,
        n_bootstrap=160,
        confidence_level=0.95,
        seed=19,
    )
    true_information = 0.6**2
    assert result.point_information_eigenvalues[0] == pytest.approx(true_information, abs=0.025)
    assert result.lower_information_eigenvalues[0] < true_information
    assert result.upper_information_eigenvalues[0] > true_information - 0.01
    assert result.successful_replicates == 160


def test_moving_block_bootstrap_reflects_serial_dependence() -> None:
    rng = np.random.default_rng(101)
    n = 5_000
    latent = np.empty(n, dtype=np.int64)
    latent[0] = rng.integers(0, 2)
    for index in range(1, n):
        latent[index] = latent[index - 1] if rng.random() < 0.95 else 1 - latent[index - 1]
    observed = np.bitwise_xor(latent, (rng.random(n) < 0.2).astype(np.int64))
    scores = np.where(latent[:, None] == 0, -1.0, 1.0)
    iid = bootstrap_empirical_relation_spectrum(
        scores, observed, n_bootstrap=120, method="iid", seed=7
    )
    block = bootstrap_empirical_relation_spectrum(
        scores,
        observed,
        n_bootstrap=120,
        method="moving_block",
        block_length=80,
        seed=7,
    )
    iid_width = iid.upper_information_eigenvalues[0] - iid.lower_information_eigenvalues[0]
    block_width = block.upper_information_eigenvalues[0] - block.lower_information_eigenvalues[0]
    assert block_width > iid_width


def test_cluster_bootstrap_reports_cluster_count() -> None:
    rng = np.random.default_rng(4)
    clusters = np.repeat(np.arange(25), 40)
    latent = rng.integers(0, 2, size=clusters.size)
    observed = np.bitwise_xor(latent, (rng.random(clusters.size) < 0.1).astype(np.int64))
    scores = np.where(latent[:, None] == 0, -1.0, 1.0)
    result = bootstrap_empirical_relation_spectrum(
        scores,
        observed,
        n_bootstrap=80,
        method="cluster",
        clusters=clusters,
        seed=5,
    )
    assert result.cluster_count == 25
    assert result.successful_replicates == 80


def test_softmax_chart_is_full_rank_and_tangent() -> None:
    p = np.asarray((0.12, 0.18, 0.27, 0.43))
    jacobian = softmax_chart_jacobian(p)
    assert jacobian.shape == (4, 3)
    assert np.linalg.matrix_rank(jacobian) == 3
    assert np.allclose(jacobian.sum(axis=0), 0.0)


def test_invalid_tangent_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        simplex_tangent_geometry((0.5, 0.5), np.eye(2), ((1.0,), (0.0,)))
    with pytest.raises(ValueError, match="full support"):
        score_subspace_geometry((1.0, 0.0), np.eye(2), ((-1.0,), (1.0,)))
    with pytest.raises(ValueError, match="epsilon leaves"):
        geometry = score_subspace_geometry(
            (0.5, 0.5), np.eye(2), ((-1.0,), (1.0,))
        )
        direction_testing_bounds(geometry, (1.0,), 2.0)


def test_bootstrap_validation_rejects_bad_configuration() -> None:
    scores = np.asarray(((-1.0,), (1.0,), (-1.0,), (1.0,)))
    labels = np.asarray((0, 1, 0, 1))
    with pytest.raises(ValueError, match="at least 20"):
        bootstrap_empirical_relation_spectrum(scores, labels, n_bootstrap=10)
    with pytest.raises(ValueError, match="block_length"):
        bootstrap_empirical_relation_spectrum(
            scores, labels, n_bootstrap=20, method="moving_block"
        )
