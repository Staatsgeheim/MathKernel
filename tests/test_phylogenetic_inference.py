# =============================================================================
# MathKernel - Tests for statistical phylogenetic tensor inference
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mathkernel.phylogenetic_inference import (
    fit_known_channel_distribution,
    fit_nonnegative_rank,
    fit_ridge_channel_distribution,
    flatten_count_tensor,
    known_channel_fisher_covariance,
    local_observation_operator,
    multinomial_covariance,
    normalize_probability_vector,
    project_probability_simplex,
    propagate_linear_covariance,
    rank_tail_frobenius,
    rank_wald_test,
    QuartetSplitScore,
    score_quartet_splits,
    select_known_channel_regularization,
    split_score_minimizers,
    split_score_winners,
)
from mathkernel.phylogenetic_tensor import FiniteMarkovTree, FiniteObservationChannel


def test_probability_simplex_projection_and_validation() -> None:
    projected = project_probability_simplex((-1.0, 2.0, 0.5))
    assert np.allclose(projected, (0.0, 1.0, 0.0))
    assert projected.sum() == pytest.approx(1.0)
    assert np.all(projected >= 0)

    source = np.asarray((0.8, -0.2, 0.7))
    result = project_probability_simplex(source)
    alternatives = np.asarray([(x, 1.0 - x, 0.0) for x in np.linspace(0.0, 1.0, 1001)])
    assert np.linalg.norm(result - source) <= np.min(
        np.linalg.norm(alternatives - source, axis=1)
    ) + 1e-3

    assert np.allclose(normalize_probability_vector((2, 3, 5)), (0.2, 0.3, 0.5))
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_probability_vector((0.5, -0.1, 0.6))


def test_local_observation_operator_matches_kronecker_action() -> None:
    first = FiniteObservationChannel.from_rows(
        ((Fraction(3, 4), Fraction(1, 4)), (Fraction(1, 5), Fraction(4, 5)))
    )
    second = FiniteObservationChannel.deterministic((0, 1, 1), output_states=2)
    operator = local_observation_operator((first, second))
    assert operator.shape == (4, 6)
    assert np.allclose(operator.sum(axis=0), 1.0)

    latent = np.asarray((0.05, 0.10, 0.15, 0.20, 0.25, 0.25))
    manual = np.zeros(4)
    for x0 in range(2):
        for x1 in range(3):
            mass = latent[x0 * 3 + x1]
            for y0 in range(2):
                for y1 in range(2):
                    manual[y0 * 2 + y1] += mass * float(
                        first.conditional[x0][y0] * second.conditional[x1][y1]
                    )
    assert np.allclose(operator @ latent, manual)


def test_multinomial_covariance_and_linear_propagation() -> None:
    probabilities = np.asarray((0.2, 0.3, 0.5))
    covariance = multinomial_covariance(probabilities, 100)
    assert np.allclose(covariance.sum(axis=0), 0.0)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1e-14)
    count_covariance = multinomial_covariance(probabilities, 100, for_counts=True)
    assert np.allclose(count_covariance, covariance * 10000)

    transform = np.asarray(((1.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    propagated = propagate_linear_covariance(covariance, transform)
    assert propagated.shape == (2, 2)
    assert np.allclose(propagated, propagated.T)


def test_known_channel_em_identity_and_full_rank_recovery() -> None:
    counts = np.asarray((10.0, 20.0, 30.0))
    identity = fit_known_channel_distribution(counts, np.eye(3))
    assert identity.converged
    assert identity.iterations == 1
    assert np.allclose(identity.probabilities, counts / counts.sum())
    assert identity.simplex_violation < 1e-14

    channel = np.asarray(
        ((0.82, 0.08, 0.10), (0.10, 0.80, 0.10), (0.08, 0.12, 0.80))
    )
    latent = np.asarray((0.2, 0.3, 0.5))
    observed = channel @ latent
    result = fit_known_channel_distribution(
        observed * 1_000_000,
        channel,
        max_iterations=50_000,
        tolerance=1e-13,
    )
    assert result.converged
    assert np.linalg.norm(result.probabilities - latent, ord=1) < 2e-6
    assert np.all(result.probabilities >= 0)
    assert result.probabilities.sum() == pytest.approx(1.0)


def test_known_channel_em_handles_rank_loss_without_invalid_probabilities() -> None:
    channel = np.asarray(((1.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    result = fit_known_channel_distribution((70, 30), channel)
    assert np.all(result.probabilities >= 0)
    assert result.probabilities.sum() == pytest.approx(1.0)
    assert np.allclose(channel @ result.probabilities, (0.7, 0.3), atol=1e-9)
    assert result.channel_rank == 2
    assert np.isinf(result.channel_condition_number)

    impossible = np.asarray(((1.0, 1.0), (0.0, 0.0)))
    with pytest.raises(ValueError, match="impossible"):
        fit_known_channel_distribution((10, 1), impossible)


def test_simplex_ridge_is_valid_and_improves_its_objective() -> None:
    channel = np.asarray(
        ((0.55, 0.45, 0.35), (0.30, 0.35, 0.40), (0.15, 0.20, 0.25))
    )
    counts = np.asarray((520.0, 320.0, 160.0))
    ridge = 0.05
    result = fit_ridge_channel_distribution(counts, channel, ridge=ridge)
    assert result.converged
    assert np.all(result.probabilities >= 0)
    assert result.probabilities.sum() == pytest.approx(1.0)

    prior = np.full(3, 1.0 / 3.0)
    y = counts / counts.sum()
    baseline = 0.5 * np.linalg.norm(channel @ prior - y) ** 2
    assert result.objective <= baseline + 1e-12


def test_known_channel_fisher_covariance_respects_simplex_tangent() -> None:
    channel = np.asarray(
        ((0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9))
    )
    result = known_channel_fisher_covariance((0.2, 0.3, 0.5), channel, 1000)
    assert result.tangent_rank == 2
    assert np.isfinite(result.condition_number)
    assert np.allclose(result.covariance.sum(axis=0), 0.0, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(result.covariance) >= -1e-14)

    collapsed = known_channel_fisher_covariance(
        (0.2, 0.3, 0.5), ((1, 1, 0), (0, 0, 1)), 1000
    )
    assert collapsed.tangent_rank < 2
    assert np.isinf(collapsed.condition_number)


def test_regularization_selection_uses_only_held_out_likelihood() -> None:
    channel = np.asarray(
        ((0.70, 0.20, 0.10), (0.20, 0.65, 0.20), (0.10, 0.15, 0.70))
    )
    selection = select_known_channel_regularization(
        (40, 35, 25),
        (20, 16, 14),
        channel,
        (0.0, 1.0, 10.0),
    )
    assert selection.selected_strength in (0.0, 1.0, 10.0)
    assert len(selection.results) == 3
    assert len(selection.validation_log_likelihoods) == 3
    best = max(selection.validation_log_likelihoods, key=lambda item: (item[1], -item[0]))
    assert selection.selected_strength == best[0]


def _rank_two_probability_matrix() -> np.ndarray:
    weights = np.asarray((0.4, 0.6))
    left = np.asarray(((0.7, 0.2, 0.1), (0.1, 0.3, 0.6)))
    right = np.asarray(((0.6, 0.3, 0.1), (0.2, 0.3, 0.5)))
    return np.einsum("h,hi,hj->ij", weights, left, right)


def test_nonnegative_rank_likelihood_fit_and_wald_diagnostic() -> None:
    probability = _rank_two_probability_matrix()
    counts = probability * 100_000
    fit = fit_nonnegative_rank(
        counts,
        2,
        restarts=8,
        seed=12,
        max_iterations=8000,
        tolerance=1e-10,
    )
    assert fit.converged
    assert fit.numerical_rank <= 2
    assert fit.deviance < 0.01
    assert np.max(np.abs(fit.probabilities - probability)) < 1e-4

    wald = rank_wald_test(counts, 2, fit=fit)
    assert wald.applicable
    assert wald.statistic < 0.1
    assert wald.p_value > 0.9

    alternative = np.asarray(
        ((0.15, 0.02, 0.03), (0.01, 0.20, 0.04), (0.05, 0.10, 0.40))
    )
    alternative /= alternative.sum()
    alternative_fit = fit_nonnegative_rank(
        alternative * 100_000, 2, restarts=10, seed=13
    )
    alternative_wald = rank_wald_test(
        alternative * 100_000, 2, fit=alternative_fit
    )
    assert alternative_fit.deviance > 100
    assert alternative_wald.statistic > 100
    assert alternative_wald.p_value < 1e-8


def _binary_quartet_counts(sample_size: int = 1_000_000) -> np.ndarray:
    tree = FiniteMarkovTree.from_edges(
        (2, 2, 2, 2, 2, 2, 2),
        root=0,
        root_distribution=(Fraction(1, 2), Fraction(1, 2)),
        edges=(
            (0, 1, ((Fraction(9, 10), Fraction(1, 10)), (Fraction(1, 5), Fraction(4, 5)))),
            (0, 2, ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 10), Fraction(9, 10)))),
            (1, 3, ((Fraction(19, 20), Fraction(1, 20)), (Fraction(1, 10), Fraction(9, 10)))),
            (1, 4, ((Fraction(9, 10), Fraction(1, 10)), (Fraction(1, 20), Fraction(19, 20)))),
            (2, 5, ((Fraction(17, 20), Fraction(3, 20)), (Fraction(1, 10), Fraction(9, 10)))),
            (2, 6, ((Fraction(4, 5), Fraction(1, 5)), (Fraction(1, 20), Fraction(19, 20)))),
        ),
        leaf_order=(3, 4, 5, 6),
    )
    law = tree.leaf_joint_law()
    probabilities = np.zeros((2, 2, 2, 2), dtype=float)
    for state, mass in law.probabilities.items():
        probabilities[state] = float(mass)
    return probabilities * sample_size


def test_quartet_scoring_prefers_true_general_markov_edge() -> None:
    counts = _binary_quartet_counts()
    scores = score_quartet_splits(
        counts,
        rank_bound=2,
        restarts=10,
        seed=17,
        max_iterations=8000,
        tolerance=1e-10,
    )
    winners = split_score_winners(scores)
    assert winners["raw_tail_frobenius"] == (0, 1)
    assert winners["likelihood_deviance"] == (0, 1)
    assert winners["wald_statistic"] == (0, 1)
    true = next(score for score in scores if score.split == (0, 1))
    assert true.likelihood_deviance < 0.1
    assert true.fitted_numerical_rank <= 2


def test_flatten_count_tensor_order_and_rank_tail() -> None:
    tensor = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    matrix = flatten_count_tensor(tensor, (0, 2))
    manual = np.transpose(tensor, (0, 2, 1, 3)).reshape(4, 4)
    assert np.array_equal(matrix, manual)
    assert rank_tail_frobenius(np.eye(4), 4) == 0.0
    assert rank_tail_frobenius(np.eye(4), 2) == pytest.approx(np.sqrt(2.0))
    assert rank_tail_frobenius(np.eye(4), 2, relative=True) == pytest.approx(
        1.0 / np.sqrt(2.0)
    )



def test_tied_quartet_scores_are_not_resolved_by_input_order() -> None:
    def item(split: tuple[int, int], raw: float) -> QuartetSplitScore:
        complement = tuple(index for index in range(4) if index not in split)
        return QuartetSplitScore(
            split=split,
            complement=complement,
            matrix_shape=(16, 16),
            sample_size=100.0,
            raw_tail_frobenius=raw,
            relative_tail_frobenius=raw,
            likelihood_deviance=raw,
            deviance_per_site=raw,
            log_likelihood=-raw,
            wald_statistic=raw,
            wald_degrees_of_freedom=1,
            wald_p_value=0.5,
            fit_converged=True,
            fit_iterations=1,
            fitted_numerical_rank=1,
        )

    scores = (item((0, 1), 0.0), item((0, 2), 0.0), item((0, 3), 1.0))
    minimizers = split_score_minimizers(scores)
    assert minimizers["raw_tail_frobenius"] == ((0, 1), (0, 2))
    with pytest.raises(ValueError, match="no unique winner"):
        split_score_winners(scores)

def test_validation_guards() -> None:
    with pytest.raises(ValueError, match="rank_bound"):
        fit_nonnegative_rank(np.ones((2, 2)), 3)
    with pytest.raises(ValueError, match="four-dimensional"):
        score_quartet_splits(np.ones((2, 2, 2)), rank_bound=2)
    with pytest.raises(ValueError, match="column"):
        fit_known_channel_distribution((1, 2), ((0.5, 0.5), (0.4, 0.4)))
