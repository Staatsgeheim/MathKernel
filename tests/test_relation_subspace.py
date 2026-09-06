from __future__ import annotations

from math import inf

import numpy as np
import pytest

from mathkernel import (
    FiniteObservationChannel,
    analytic_ar1_long_run_covariance,
    chi_square_divergence,
    collision_certificate,
    dependent_relation_information,
    direction_information_retention,
    direction_visibility,
    evaluate_sensor_subset,
    linear_relation_law,
    mode_transfer,
    newey_west_long_run_covariance,
    optimize_sensor_subsets,
    push_finite_law,
    relation_chi_square,
    relation_subspace_transfer,
)


MARGINAL4 = np.full(4, 0.25)
SCORES4 = np.asarray(
    [
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
    ]
)
BIT1 = FiniteObservationChannel.deterministic((0, 0, 1, 1), name="bit1")
BIT2 = FiniteObservationChannel.deterministic((0, 1, 0, 1), name="bit2")
PARITY = FiniteObservationChannel.deterministic((0, 1, 1, 0), name="parity")


def noisy_bit_channel(bit: int, error: float) -> FiniteObservationChannel:
    rows = []
    for state in range(4):
        value = (state >> (1 - bit)) & 1
        rows.append((1 - error, error) if value == 0 else (error, 1 - error))
    return FiniteObservationChannel.from_rows(rows, name=f"bit{bit + 1}-e{error}")


def noisy_parity_channel(error: float) -> FiniteObservationChannel:
    mapping = (0, 1, 1, 0)
    return FiniteObservationChannel.from_rows(
        [
            (1 - error, error) if value == 0 else (error, 1 - error)
            for value in mapping
        ],
        name=f"parity-e{error}",
    )


def test_one_dimensional_transfer_recovers_mode_visibility() -> None:
    marginal = np.asarray((0.2, 0.3, 0.5))
    channel = np.asarray(((0.8, 0.2), (0.4, 0.6), (0.1, 0.9)))
    score = np.asarray((1.0, -2.0, 0.8))
    subspace = relation_subspace_transfer(marginal, channel, score)
    scalar = mode_transfer(marginal, channel, score)
    assert subspace.latent_rank == 1
    assert subspace.observed_rank == int(not scalar.blind)
    assert subspace.transfer_singular_values[0] == pytest.approx(scalar.visibility)
    assert subspace.generalized_information_eigenvalues[0] == pytest.approx(
        scalar.visibility_squared
    )


def test_complementary_deterministic_sensors_restore_full_relation_space() -> None:
    first = relation_subspace_transfer(MARGINAL4, BIT1, SCORES4)
    second = relation_subspace_transfer(MARGINAL4, BIT2, SCORES4)
    fused = evaluate_sensor_subset(
        MARGINAL4, SCORES4, {"bit1": BIT1, "bit2": BIT2}, ("bit1", "bit2")
    )
    assert first.latent_rank == 3 and first.observed_rank == 1
    assert second.observed_rank == 1
    assert first.observation_blind_directions.shape[1] == 2
    assert fused.fully_identifiable
    assert fused.observed_rank == 3
    assert fused.min_visibility == pytest.approx(1.0)
    assert fused.logdet_information_retention == pytest.approx(0.0)


def test_noisy_complementary_sensors_have_product_parity_visibility() -> None:
    sensors = {"bit1": noisy_bit_channel(0, 0.1), "bit2": noisy_bit_channel(1, 0.1)}
    result = evaluate_sensor_subset(MARGINAL4, SCORES4, sensors, ("bit1", "bit2"))
    # Reliabilities are 0.8, 0.8 and their product 0.64 for parity.
    assert result.fully_identifiable
    assert result.min_visibility == pytest.approx(0.64)
    assert result.mean_information_retention == pytest.approx(
        (0.8**2 + 0.8**2 + 0.64**2) / 3
    )


def test_generalized_eigenvalues_are_basis_covariant() -> None:
    channel = noisy_bit_channel(0, 0.15)
    original = relation_subspace_transfer(MARGINAL4, channel, SCORES4)
    change = np.asarray(((2.0, 1.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0)))
    rotated = relation_subspace_transfer(MARGINAL4, channel, SCORES4 @ change)
    assert np.allclose(
        original.generalized_information_eigenvalues,
        rotated.generalized_information_eigenvalues,
        atol=2e-12,
    )
    assert original.latent_rank == rotated.latent_rank
    assert original.observed_rank == rotated.observed_rank


def test_exact_blind_combination_produces_collision_certificate() -> None:
    transfer = relation_subspace_transfer(MARGINAL4, BIT1, SCORES4)
    assert transfer.observation_blind_directions.shape[1] == 2
    direction = np.asarray((0.0, 1.0, 0.0))
    assert direction_information_retention(transfer, direction) == pytest.approx(0.0)
    certificate = collision_certificate(MARGINAL4, BIT1, SCORES4, direction)
    assert certificate.latent_l1_separation > 0.1
    assert certificate.observed_l1_separation < 1e-13
    assert certificate.exact_within_tolerance


def test_linear_family_chi_square_is_exact_quadratic_form() -> None:
    channel = noisy_bit_channel(0, 0.2)
    transfer = relation_subspace_transfer(MARGINAL4, channel, SCORES4)
    theta = np.asarray((0.2, -0.1, 0.08))
    latent = linear_relation_law(MARGINAL4, SCORES4, theta)
    observed = push_finite_law(latent, channel)
    direct = chi_square_divergence(observed, transfer.observed_law)
    predicted = relation_chi_square(transfer, theta)
    assert direct == pytest.approx(predicted, abs=2e-14)


def test_direction_visibility_respects_declared_mode_geometry() -> None:
    transfer = relation_subspace_transfer(MARGINAL4, noisy_bit_channel(0, 0.1), SCORES4)
    assert direction_visibility(transfer, (1.0, 0.0, 0.0)) == pytest.approx(0.8)
    assert direction_visibility(transfer, (0.0, 1.0, 0.0)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="latent-degenerate"):
        direction_visibility(transfer, (0.0, 0.0, 0.0))


def test_data_processing_spectrum_is_bounded_for_random_channels() -> None:
    rng = np.random.default_rng(20260906)
    for _ in range(100):
        n = int(rng.integers(2, 8))
        m = int(rng.integers(2, 7))
        r = int(rng.integers(1, min(5, n)))
        p = rng.dirichlet(np.ones(n))
        channel = rng.dirichlet(np.ones(m), size=n)
        scores = rng.normal(size=(n, r))
        transfer = relation_subspace_transfer(p, channel, scores)
        assert np.min(transfer.generalized_information_eigenvalues) >= -1e-13
        assert np.max(transfer.generalized_information_eigenvalues) <= 1 + 1e-13
        assert transfer.data_processing_residual < 2e-12


def test_sensor_optimizer_selects_strong_complementary_pair() -> None:
    sensors = {
        "bit1": noisy_bit_channel(0, 0.1),
        "bit2": noisy_bit_channel(1, 0.1),
        "parity": noisy_parity_channel(0.25),
    }
    result = optimize_sensor_subsets(
        MARGINAL4,
        SCORES4,
        sensors,
        max_sensors=2,
        objective="e_optimal",
    )
    assert result.best_subsets
    assert {item.sensors for item in result.best_subsets} == {("bit1", "bit2")}
    assert result.best_value == pytest.approx(0.64**2)


def test_d_optimal_requires_full_relation_rank() -> None:
    sensors = {"bit1": BIT1, "bit2": BIT2}
    result = optimize_sensor_subsets(
        MARGINAL4, SCORES4, sensors, max_sensors=2, objective="d_optimal"
    )
    assert result.best_subsets[0].sensors == ("bit1", "bit2")
    singles = [item for item in result.evaluations if len(item.sensors) == 1]
    assert all(item.objective_value == -inf for item in singles)


def test_cost_constraint_is_respected() -> None:
    sensors = {"bit1": BIT1, "bit2": BIT2, "parity": PARITY}
    costs = {"bit1": 2.0, "bit2": 2.0, "parity": 10.0}
    result = optimize_sensor_subsets(
        MARGINAL4,
        SCORES4,
        sensors,
        costs=costs,
        max_sensors=2,
        max_total_cost=4.0,
        objective="rank",
    )
    assert all(item.total_cost <= 4.0 for item in result.evaluations)
    assert result.best_subsets[0].sensors == ("bit1", "bit2")


def test_ar1_dependence_has_exact_sample_inflation() -> None:
    iid = np.eye(3)
    omega = analytic_ar1_long_run_covariance(iid, 0.6)
    result = dependent_relation_information(iid, omega)
    assert np.allclose(omega, 4.0 * np.eye(3))
    assert np.allclose(result.effective_information, 0.25 * np.eye(3))
    assert np.allclose(result.generalized_information_eigenvalues, 0.25)
    assert np.allclose(result.sample_inflation_eigenvalues, 4.0)
    assert result.worst_case_sample_inflation == pytest.approx(4.0)


def test_negative_serial_correlation_can_improve_mean_information() -> None:
    iid = np.eye(2)
    omega = analytic_ar1_long_run_covariance(iid, -0.5)
    result = dependent_relation_information(iid, omega)
    assert np.allclose(omega, np.eye(2) / 3.0)
    assert np.allclose(result.generalized_information_eigenvalues, 3.0)
    assert result.best_case_sample_inflation == pytest.approx(1 / 3)


def test_newey_west_estimates_scalar_ar1_long_run_variance() -> None:
    rng = np.random.default_rng(20260906)
    phi = 0.7
    n = 150_000
    innovations = rng.normal(scale=np.sqrt(1 - phi**2), size=n)
    values = np.empty(n)
    values[0] = innovations[0]
    for index in range(1, n):
        values[index] = phi * values[index - 1] + innovations[index]
    estimate = newey_west_long_run_covariance(values, 250)[0, 0]
    target = (1 + phi) / (1 - phi)
    assert estimate == pytest.approx(target, rel=0.08)


def test_dependent_information_exposes_unrecoverable_direction() -> None:
    iid = np.eye(2)
    omega = np.diag((1.0, 0.0))
    result = dependent_relation_information(iid, omega)
    assert result.effective_rank == 1
    assert np.isinf(result.worst_case_sample_inflation)


def test_linearly_dependent_scores_report_latent_nullspace() -> None:
    scores = np.column_stack((SCORES4[:, 0], 2 * SCORES4[:, 0], SCORES4[:, 1]))
    transfer = relation_subspace_transfer(MARGINAL4, BIT1, scores)
    assert transfer.latent_rank == 2
    assert transfer.latent_nullspace.shape == (3, 1)
    assert transfer.observed_rank == 1


def test_invalid_probability_and_channel_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="full support"):
        relation_subspace_transfer((1.0, 0.0), ((1.0,), (1.0,)), ((-1.0,), (1.0,)))
    with pytest.raises(ValueError, match="row must sum"):
        relation_subspace_transfer((0.5, 0.5), ((0.8, 0.3), (0.2, 0.8)), ((-1.0,), (1.0,)))
    with pytest.raises(ValueError, match="zero latent Fisher rank"):
        relation_subspace_transfer((0.5, 0.5), ((1.0,), (1.0,)), ((1.0,), (1.0,)))


def test_linear_family_rejects_parameters_outside_probability_region() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        linear_relation_law(MARGINAL4, SCORES4, (2.0, 2.0, 2.0))


def test_sensor_ties_are_returned_explicitly() -> None:
    sensors = {"bit1a": BIT1, "bit1b": BIT1}
    result = optimize_sensor_subsets(
        MARGINAL4, SCORES4[:, :1], sensors, max_sensors=1, objective="e_optimal"
    )
    assert {item.sensors for item in result.best_subsets} == {("bit1a",), ("bit1b",)}


def test_partition_api_matches_dense_deterministic_channel() -> None:
    from mathkernel import relation_subspace_from_partition

    labels = np.asarray((0, 0, 1, 1))
    partitioned = relation_subspace_from_partition(MARGINAL4, labels, SCORES4)
    dense = relation_subspace_transfer(MARGINAL4, BIT1, SCORES4)
    assert np.allclose(partitioned.observed_gram, dense.observed_gram)
    assert np.allclose(
        partitioned.generalized_information_eigenvalues,
        dense.generalized_information_eigenvalues,
    )


def test_partition_api_compacts_sparse_labels() -> None:
    from mathkernel import relation_subspace_from_partition

    result = relation_subspace_from_partition(
        MARGINAL4, np.asarray((10, 10, 99, 99)), SCORES4
    )
    assert result.observed_law.shape == (2,)
    assert result.observed_rank == 1


def test_adding_conditionally_independent_sensor_cannot_reduce_information() -> None:
    sensors = {
        "bit1": noisy_bit_channel(0, 0.1),
        "bit2": noisy_bit_channel(1, 0.1),
        "parity": noisy_parity_channel(0.25),
    }
    first = evaluate_sensor_subset(
        MARGINAL4, SCORES4, sensors, ("bit1",), objective="trace"
    )
    pair = evaluate_sensor_subset(
        MARGINAL4, SCORES4, sensors, ("bit1", "bit2"), objective="trace"
    )
    triple = evaluate_sensor_subset(
        MARGINAL4,
        SCORES4,
        sensors,
        ("bit1", "bit2", "parity"),
        objective="trace",
    )
    assert first.information_trace <= pair.information_trace + 1e-12
    assert pair.information_trace <= triple.information_trace + 1e-12
    assert first.observed_rank <= pair.observed_rank <= triple.observed_rank
    assert pair.min_visibility <= triple.min_visibility + 1e-12
