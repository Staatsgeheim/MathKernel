# =============================================================================
# MathKernel - Tests for observable connected-relation detection
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from fractions import Fraction
from math import inf

import numpy as np
import pytest

from mathkernel.phylogenetic_tensor import FiniteObservationChannel
from mathkernel.relation_detection import (
    binary_parity_information,
    binary_symmetric_channel,
    conditional_expectation_spectrum,
    exact_chi_square,
    exact_pure_interaction_certificate,
    exact_pure_interaction_law,
    joint_conditionally_independent_channel,
    mode_transfer,
    numerical_pure_interaction_law,
    pure_interaction_information,
    pure_relation_sample_bounds,
    push_joint_law_through_local_channels,
    standardize_centered_mode,
)
from mathkernel.stochastic_koopman import FiniteJointLaw


def test_standardize_centered_mode_has_zero_mean_and_unit_norm() -> None:
    marginal = np.asarray((0.2, 0.3, 0.5))
    mode = standardize_centered_mode((4.0, -1.0, 2.0), marginal)
    assert abs(float(np.dot(marginal, mode))) < 1e-14
    assert abs(float(np.dot(marginal, mode * mode)) - 1.0) < 1e-14
    with pytest.raises(ValueError, match="constant"):
        standardize_centered_mode((1.0, 1.0, 1.0), marginal)


def test_bsc_centered_singular_value_is_reliability() -> None:
    for error in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        spectrum = conditional_expectation_spectrum(
            (0.5, 0.5), binary_symmetric_channel(error)
        )
        assert spectrum.centered_singular_values.shape == (1,)
        assert spectrum.centered_singular_values[0] == pytest.approx(abs(1 - 2 * error))
        assert spectrum.maximal_correlation == pytest.approx(abs(1 - 2 * error))


def test_singular_modes_saturate_mode_visibility() -> None:
    marginal = (0.1, 0.2, 0.3, 0.4)
    channel = np.asarray(
        (
            (0.70, 0.20, 0.10),
            (0.15, 0.65, 0.20),
            (0.20, 0.20, 0.60),
            (0.55, 0.10, 0.35),
        )
    )
    spectrum = conditional_expectation_spectrum(marginal, channel)
    assert spectrum.centered_singular_values.size == 2
    for index, singular in enumerate(spectrum.centered_singular_values):
        transfer = mode_transfer(
            marginal,
            channel,
            spectrum.latent_modes[:, index],
            standardize=False,
        )
        assert transfer.visibility == pytest.approx(singular, rel=1e-11, abs=1e-12)
        assert abs(transfer.transfer_coefficient - transfer.visibility) < 1e-11
        if singular > 1e-12:
            assert np.max(
                np.abs(
                    transfer.standardized_output_mode
                    - spectrum.observed_modes[:, index]
                )
            ) < 1e-10


def test_arbitrary_mode_visibility_is_bounded_by_maximal_correlation() -> None:
    marginal = (0.15, 0.20, 0.25, 0.40)
    channel = np.asarray(
        (
            (0.6, 0.3, 0.1),
            (0.2, 0.6, 0.2),
            (0.1, 0.3, 0.6),
            (0.5, 0.2, 0.3),
        )
    )
    rng = np.random.default_rng(20260906)
    spectrum = conditional_expectation_spectrum(marginal, channel)
    for _ in range(50):
        transfer = mode_transfer(marginal, channel, rng.normal(size=4))
        assert transfer.visibility <= spectrum.maximal_correlation + 2e-12


def test_deterministic_observation_matches_projection_visibility() -> None:
    channel = FiniteObservationChannel.deterministic((0, 0, 1, 1))
    visible = mode_transfer((0.25,) * 4, channel, (-1, -1, 1, 1), standardize=False)
    blind = mode_transfer((0.25,) * 4, channel, (-1, 1, -1, 1), standardize=False)
    assert visible.visibility == pytest.approx(1.0)
    assert visible.transfer_coefficient == pytest.approx(1.0)
    assert blind.visibility == pytest.approx(0.0, abs=1e-14)
    assert blind.blind


def test_zero_probability_output_symbols_are_removed_from_spectrum() -> None:
    channel = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    spectrum = conditional_expectation_spectrum((0.5, 0.5), channel)
    assert spectrum.active_output_indices == (0, 1)
    assert spectrum.observed_modes.shape == (3, 1)
    assert spectrum.observed_modes[2, 0] == 0.0
    assert spectrum.centered_singular_values[0] == pytest.approx(1.0)


def test_exact_pure_interaction_certificate_proves_all_identities() -> None:
    errors = (Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 8))
    theta = Fraction(2, 5)
    channels = tuple(binary_symmetric_channel(error) for error in errors)
    certificate = exact_pure_interaction_certificate(
        ((Fraction(1, 2), Fraction(1, 2)),) * 4,
        ((-1, 1),) * 4,
        channels,
        theta,
    )
    reliability_product = np.prod([Fraction(1) - 2 * error for error in errors])
    expected_chi = theta * theta * reliability_product * reliability_product
    assert certificate.formula_matches
    assert certificate.proper_latent_marginals_preserved
    assert certificate.proper_observed_marginals_preserved
    assert certificate.latent_connected_cumulant == theta
    assert certificate.latent_chi_square == theta * theta
    assert certificate.observed_chi_square == expected_chi
    assert certificate.predicted_observed_chi_square == expected_chi
    assert certificate.observed_law.cumulant(((-1, 1),) * 4) == theta * reliability_product


def test_exact_blind_channel_erases_the_relation() -> None:
    certificate = exact_pure_interaction_certificate(
        ((Fraction(1, 2), Fraction(1, 2)),) * 3,
        ((-1, 1),) * 3,
        (
            binary_symmetric_channel(Fraction(1, 10)),
            binary_symmetric_channel(Fraction(1, 2)),
            binary_symmetric_channel(Fraction(1, 5)),
        ),
        Fraction(1, 2),
    )
    assert certificate.local_transfers[1].visibility_squared == 0
    assert certificate.observed_chi_square == 0
    assert certificate.observed_law.probabilities == certificate.null_observed_law.probabilities


def test_pure_interaction_rejects_invalid_parameter() -> None:
    with pytest.raises(ValueError, match="nonnegativity"):
        exact_pure_interaction_law(
            ((Fraction(1, 2), Fraction(1, 2)),) * 3,
            ((-1, 1),) * 3,
            Fraction(11, 10),
        )
    with pytest.raises(ValueError, match="nonnegativity"):
        numerical_pure_interaction_law(
            ((0.5, 0.5),) * 2,
            ((-1.0, 1.0),) * 2,
            1.1,
        )


def test_numerical_general_channel_formula_and_chi_square_factorization() -> None:
    marginals = ((0.2, 0.3, 0.5), (0.1, 0.2, 0.3, 0.4), (0.5, 0.5))
    modes = ((1.0, -1.0, 0.0), (1.0, -1.0, 1.0, -1.0), (-1.0, 1.0))
    channels = (
        ((0.7, 0.2, 0.1), (0.2, 0.5, 0.3), (0.1, 0.3, 0.6)),
        ((0.8, 0.2), (0.6, 0.4), (0.3, 0.7), (0.1, 0.9)),
        ((0.9, 0.1), (0.25, 0.75)),
    )
    information = pure_interaction_information(marginals, modes, channels, 0.2)
    assert information.law_formula_error < 2e-15
    assert information.proper_latent_marginal_error < 2e-15
    assert information.proper_observed_marginal_error < 2e-15
    assert information.observed_chi_square == pytest.approx(
        information.theta**2 * np.prod(np.square(information.visibilities)),
        rel=2e-12,
    )
    assert information.connected_amplitude == pytest.approx(
        information.theta * np.prod(information.visibilities)
    )


def test_binary_parity_information_scales_as_reliability_to_order() -> None:
    theta = 0.6
    error = 0.2
    reliability = 1 - 2 * error
    for order in range(2, 9):
        information = binary_parity_information(
            order, theta, error_probability=error, verify_law=False
        )
        assert information.visibility_product == pytest.approx(reliability**order)
        assert information.connected_amplitude == pytest.approx(theta * reliability**order)
        assert information.information_retention == pytest.approx(reliability ** (2 * order))
        assert information.observed_chi_square == pytest.approx(
            theta**2 * reliability ** (2 * order)
        )


def test_heterogeneous_binary_channels_multiply_modewise() -> None:
    errors = (0.05, 0.10, 0.20, 0.35, 0.45)
    information = binary_parity_information(
        5, 0.4, error_probability=errors, verify_law=False
    )
    expected = np.prod([abs(1 - 2 * value) for value in errors])
    assert information.visibility_product == pytest.approx(expected)
    assert information.information_retention == pytest.approx(expected * expected)


def test_sample_bounds_are_finite_and_monotone_under_information_loss() -> None:
    clear = binary_parity_information(4, 0.5, error_probability=0.05, verify_law=False)
    noisy = binary_parity_information(4, 0.5, error_probability=0.30, verify_law=False)
    clear_bounds = pure_relation_sample_bounds(clear, max_error=0.05)
    noisy_bounds = pure_relation_sample_bounds(noisy, max_error=0.05)
    assert clear_bounds.chi_square_lower_samples < noisy_bounds.chi_square_lower_samples
    assert clear_bounds.hoeffding_upper_samples < noisy_bounds.hoeffding_upper_samples
    assert clear_bounds.inverse_information_scale < noisy_bounds.inverse_information_scale
    assert clear_bounds.chi_square_lower_samples <= clear_bounds.hoeffding_upper_samples
    assert noisy_bounds.chi_square_lower_samples <= noisy_bounds.hoeffding_upper_samples


def test_blind_relation_has_infinite_detection_cost() -> None:
    information = binary_parity_information(
        4,
        0.5,
        error_probability=(0.1, 0.2, 0.5, 0.1),
    )
    bounds = pure_relation_sample_bounds(information)
    assert information.blind
    assert information.observed_chi_square == 0
    assert bounds.chi_square_lower_samples == inf
    assert bounds.hoeffding_upper_samples == inf
    assert bounds.inverse_information_scale == inf


def test_complementary_sensor_fusion_restores_an_invisible_mode() -> None:
    # State labels are (bit_1, bit_2) in lexicographic order.  The declared
    # mode is their parity/product, invisible to either bit alone.
    marginal = (0.25, 0.25, 0.25, 0.25)
    parity_mode = (1.0, -1.0, -1.0, 1.0)
    bit_one = FiniteObservationChannel.deterministic((0, 0, 1, 1), name="bit one")
    bit_two = FiniteObservationChannel.deterministic((0, 1, 0, 1), name="bit two")
    fused = joint_conditionally_independent_channel((bit_one, bit_two))
    first = mode_transfer(marginal, bit_one, parity_mode, standardize=False)
    second = mode_transfer(marginal, bit_two, parity_mode, standardize=False)
    together = mode_transfer(marginal, fused, parity_mode, standardize=False)
    assert first.blind and second.blind
    assert together.visibility == pytest.approx(1.0)

    invisible = pure_interaction_information(
        (marginal,) * 3,
        (parity_mode,) * 3,
        (bit_one,) * 3,
        0.4,
    )
    visible = pure_interaction_information(
        (marginal,) * 3,
        (parity_mode,) * 3,
        (fused,) * 3,
        0.4,
    )
    assert invisible.observed_chi_square == 0
    assert visible.observed_chi_square == pytest.approx(0.16)


def test_push_joint_law_matches_expected_shapes() -> None:
    law = numerical_pure_interaction_law(
        ((0.5, 0.5),) * 3,
        ((-1.0, 1.0),) * 3,
        0.25,
    )
    observed = push_joint_law_through_local_channels(
        law,
        (
            ((0.8, 0.2), (0.1, 0.9)),
            ((0.7, 0.2, 0.1), (0.2, 0.3, 0.5)),
            ((1.0,), (1.0,)),
        ),
    )
    assert observed.shape == (2, 3, 1)
    assert observed.sum() == pytest.approx(1.0)



def test_binary_parity_accepts_numpy_error_vector() -> None:
    errors = np.asarray((0.05, 0.10, 0.20, 0.25))
    information = binary_parity_information(
        4, 0.3, error_probability=errors, verify_law=False
    )
    expected = np.prod(np.abs(1.0 - 2.0 * errors))
    assert information.visibility_product == pytest.approx(expected)


def test_pure_interaction_rejects_nonfinite_theta() -> None:
    for theta in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            pure_interaction_information(
                ((0.5, 0.5),) * 2,
                ((-1.0, 1.0),) * 2,
                (binary_symmetric_channel(0.1),) * 2,
                theta,
            )

def test_exact_chi_square_rejects_missing_reference_support() -> None:
    law = FiniteJointLaw.from_dense((2,), (Fraction(1, 2), Fraction(1, 2)))
    reference = FiniteJointLaw.from_dense((2,), (1, 0))
    with pytest.raises(ValueError, match="absolutely continuous"):
        exact_chi_square(law, reference)
