# =============================================================================
# MathKernel - Tests for stochastic observation transfer / Koopman dilations
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from fractions import Fraction
from itertools import product

import pytest

from mathkernel.koopman import FiniteSystem, observed_statistic
from mathkernel.stochastic_koopman import (
    FiniteJointLaw,
    FiniteMarkovKernel,
    RationalMarkovDilation,
    arbitrary_joint_contraction,
    compressed_transport_cumulant,
    compressed_transport_moment,
    deterministic_koopman_equivalence,
    markov_observation_contraction,
    observation_transfer_coefficients,
    pullback_observable,
    rational_basis_coordinates,
)


HADAMARD_2 = (
    (Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(-1)),
)


def _dense_binary_law(order: int) -> FiniteJointLaw:
    weights = list(range(1, (1 << order) + 1))
    total = sum(weights)
    return FiniteJointLaw.from_dense(
        (2,) * order, [Fraction(weight, total) for weight in weights]
    )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_theorem_a_arbitrary_joint_contraction_exact(order: int) -> None:
    law = _dense_binary_law(order)
    observables = []
    for index in range(order):
        observables.append(
            (
                Fraction(2 * index - 3, index + 2),
                Fraction(3 * index + 5, index + 3),
            )
        )
    direct = law.cumulant(observables)
    contracted = arbitrary_joint_contraction(
        law, observables, (HADAMARD_2,) * order
    )
    assert contracted == direct


def test_theorem_a_allows_heterogeneous_latent_spaces_and_observation_maps() -> None:
    law = FiniteJointLaw(
        (2, 3, 2),
        {
            (0, 0, 0): Fraction(1, 10),
            (0, 1, 1): Fraction(1, 5),
            (0, 2, 0): Fraction(1, 10),
            (1, 0, 1): Fraction(1, 4),
            (1, 1, 0): Fraction(3, 20),
            (1, 2, 1): Fraction(1, 5),
        },
    )
    observations = ((0, 1), (0, 1, 1), (1, 0))
    output_functions = (
        (Fraction(-2), Fraction(5)),
        (Fraction(7, 3), Fraction(-1, 2)),
        (Fraction(4), Fraction(9)),
    )
    pulled = tuple(
        pullback_observable(observation, output_function)
        for observation, output_function in zip(
            observations, output_functions, strict=True
        )
    )
    ternary_basis = (
        (Fraction(1), Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(-1)),
        (Fraction(1), Fraction(-2), Fraction(1)),
    )
    bases = (HADAMARD_2, ternary_basis, HADAMARD_2)
    assert arbitrary_joint_contraction(law, pulled, bases) == law.cumulant(pulled)


def test_exact_rational_basis_and_observation_transfer() -> None:
    basis = (
        (Fraction(1), Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(-1)),
        (Fraction(1), Fraction(-2), Fraction(1)),
    )
    function = (Fraction(4), Fraction(1), Fraction(7))
    coefficients = rational_basis_coordinates(function, basis)
    reconstructed = tuple(
        sum(coefficients[a] * basis[a][x] for a in range(3)) for x in range(3)
    )
    assert reconstructed == function
    assert observation_transfer_coefficients((0, 1, 1), (4, 7), basis) == \
        rational_basis_coordinates((4, 7, 7), basis)


def _stochastic_kernel() -> FiniteMarkovKernel:
    return FiniteMarkovKernel.from_rows(
        (
            (Fraction(3, 4), Fraction(1, 4)),
            (Fraction(1, 3), Fraction(2, 3)),
        )
    )


def test_markov_ordered_product_formula_matches_independent_joint_law() -> None:
    kernel = _stochastic_kernel()
    initial = (Fraction(2, 5), Fraction(3, 5))
    times = (0, 1, 3, 3)
    functions = (
        (Fraction(-1), Fraction(2)),
        (Fraction(3), Fraction(5)),
        (Fraction(0), Fraction(7)),
        (Fraction(4), Fraction(-2)),
    )
    law = kernel.path_joint_law(initial, times)
    assert kernel.path_moment(initial, times, functions) == law.moment(functions)
    assert kernel.path_cumulant(initial, times, functions) == law.cumulant(functions)


def test_markov_observation_contraction_is_exact() -> None:
    kernel = _stochastic_kernel()
    initial = (Fraction(2, 5), Fraction(3, 5))
    times = (0, 1, 2)
    observables = (
        (Fraction(3), Fraction(-1)),
        (Fraction(5, 2), Fraction(7, 3)),
        (Fraction(-4), Fraction(2)),
    )
    direct = kernel.path_cumulant(initial, times, observables)
    via = markov_observation_contraction(
        kernel, initial, times, observables, HADAMARD_2
    )
    assert via == direct


def test_deterministic_markov_path_reduces_to_existing_koopman_system() -> None:
    successor = (1, 2, 0)
    initial = (Fraction(1, 3),) * 3
    kernel = FiniteMarkovKernel.deterministic(successor)
    system = FiniteSystem(list(initial), list(successor))
    functions = [
        [Fraction(-1), Fraction(2), Fraction(4)],
        [Fraction(3), Fraction(0), Fraction(-2)],
        [Fraction(1), Fraction(5), Fraction(7)],
    ]
    times = [0, 2, 5]
    direct_existing = observed_statistic(
        system, functions, [0, 1, 2], times, connected=True
    )
    assert kernel.path_cumulant(initial, times, functions) == direct_existing
    equivalence = deterministic_koopman_equivalence(
        kernel, initial, times, functions
    )
    assert equivalence["moment_equal"]
    assert equivalence["cumulant_equal"]


def test_naive_independent_markov_transport_fails_for_genuine_stochasticity() -> None:
    kernel = _stochastic_kernel()
    initial = (Fraction(1, 2), Fraction(1, 2))
    function = (Fraction(-1), Fraction(1))
    times = (1, 2)
    true_moment = kernel.path_moment(initial, times, (function, function))
    naive_moment = compressed_transport_moment(
        kernel, initial, times, (function, function)
    )
    assert true_moment == Fraction(61, 144)
    assert naive_moment == Fraction(71, 864)
    assert true_moment != naive_moment
    assert kernel.path_cumulant(initial, times, (function, function)) != \
        compressed_transport_cumulant(kernel, initial, times, (function, function))


def test_multiplicativity_defect_is_conditional_covariance() -> None:
    kernel = _stochastic_kernel()
    f = (Fraction(-2), Fraction(3))
    g = (Fraction(5), Fraction(7))
    defect = kernel.multiplicativity_defect(f, g)
    manual = []
    for row in kernel.transition:
        ef = sum(row[y] * f[y] for y in range(kernel.n))
        eg = sum(row[y] * g[y] for y in range(kernel.n))
        efg = sum(row[y] * f[y] * g[y] for y in range(kernel.n))
        manual.append(efg - ef * eg)
    assert defect == tuple(manual)
    assert any(value != 0 for value in defect)


def test_algebra_homomorphism_iff_kernel_is_deterministic_exhaustive_small_grid() -> None:
    probabilities = (Fraction(0), Fraction(1, 2), Fraction(1))
    for p, q in product(probabilities, repeat=2):
        kernel = FiniteMarkovKernel.from_rows(((p, 1 - p), (q, 1 - q)))
        certificate = kernel.algebra_homomorphism_certificate()
        expected = p in (0, 1) and q in (0, 1)
        assert kernel.is_deterministic() == expected
        assert certificate["multiplicative_on_point_indicators"] == expected
        assert certificate["equivalence_verified"]
        if expected:
            for f in product((-1, 0, 2), repeat=2):
                for g in product((-3, 1, 4), repeat=2):
                    assert kernel.multiplicativity_defect(f, g) == (0, 0)
        else:
            assert certificate["witnesses"]


def test_rational_dilation_reproduces_kernel_compression_and_path_law() -> None:
    kernel = _stochastic_kernel()
    dilation = RationalMarkovDilation.from_kernel(kernel)
    assert dilation.noise_size == 12
    assert dilation.induced_kernel() == kernel
    function = (Fraction(-7, 3), Fraction(11, 5))
    assert dilation.compression_apply(function) == kernel.apply(function)

    initial = (Fraction(2, 5), Fraction(3, 5))
    times = (0, 1, 2, 4)
    markov_law = kernel.path_joint_law(initial, times)
    lifted_law = dilation.path_joint_law(initial, times)
    assert lifted_law == markov_law

    observables = (
        (Fraction(-1), Fraction(1)),
        (Fraction(0), Fraction(3)),
        (Fraction(5), Fraction(-2)),
        (Fraction(4), Fraction(7)),
    )
    assert lifted_law.moment(observables) == kernel.path_moment(
        initial, times, observables
    )
    assert lifted_law.cumulant(observables) == kernel.path_cumulant(
        initial, times, observables
    )


def test_repeated_time_exposes_failure_of_multiplicativity_maximally() -> None:
    fair_resampling = FiniteMarkovKernel.from_rows(
        ((Fraction(1, 2), Fraction(1, 2)),) * 2
    )
    initial = (Fraction(1, 2), Fraction(1, 2))
    sign = (Fraction(-1), Fraction(1))
    # Both arguments refer to the same random variable X_1.
    assert fair_resampling.path_moment(initial, (1, 1), (sign, sign)) == 1
    assert compressed_transport_moment(
        fair_resampling, initial, (1, 1), (sign, sign)
    ) == 0


def test_validation_guards() -> None:
    with pytest.raises(ValueError, match="sum exactly"):
        FiniteJointLaw((2,), {(0,): Fraction(1, 3)})
    with pytest.raises(ValueError, match="row"):
        FiniteMarkovKernel.from_rows(((Fraction(1, 3), Fraction(1, 3)),) * 2)
    with pytest.raises(ValueError, match="nondecreasing"):
        _stochastic_kernel().path_moment((1, 0), (2, 1), ((1, 2), (1, 2)))
    with pytest.raises(ValueError, match="singular"):
        rational_basis_coordinates((1, 2), ((1, 1), (2, 2)))
