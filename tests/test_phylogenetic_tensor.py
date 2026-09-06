# =============================================================================
# MathKernel - Tests for branching general-Markov tensors and observation rank
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from fractions import Fraction
import random

import pytest

from mathkernel.phylogenetic_tensor import (
    FiniteMarkovTree,
    FiniteObservationChannel,
    edge_flattening_certificate,
    exact_matrix_rank,
    minimum_collectively_injective_channel_sets,
    observation_flattening_certificate,
    recover_latent_joint_law,
    singular_value_observation_bounds,
    tensor_flattening,
    transform_joint_law,
    branching_observation_contraction,
)


def _stochastic_matrix(size: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        weights = [rng.randint(1, 9) for _ in range(size)]
        total = sum(weights)
        rows.append(tuple(Fraction(weight, total) for weight in weights))
    matrix = tuple(rows)
    assert exact_matrix_rank(matrix) == size
    return matrix


def _gm4_quartet() -> FiniteMarkovTree:
    # Unrooted split 01|23, represented by the internal edge 4 -> 5.
    return FiniteMarkovTree.from_edges(
        (4, 4, 4, 4, 4, 4),
        root=4,
        root_distribution=(
            Fraction(1, 10),
            Fraction(2, 10),
            Fraction(3, 10),
            Fraction(4, 10),
        ),
        edges=(
            (4, 0, _stochastic_matrix(4, 1)),
            (4, 1, _stochastic_matrix(4, 2)),
            (4, 5, _stochastic_matrix(4, 3)),
            (5, 2, _stochastic_matrix(4, 4)),
            (5, 3, _stochastic_matrix(4, 5)),
        ),
        leaf_order=(0, 1, 2, 3),
        name="generic GM(4) quartet 01|23",
    )


def _hadamard4():
    return (
        (Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(-1), Fraction(1), Fraction(-1)),
        (Fraction(1), Fraction(1), Fraction(-1), Fraction(-1)),
        (Fraction(1), Fraction(-1), Fraction(-1), Fraction(1)),
    )


def test_branching_sum_product_matches_full_leaf_enumeration() -> None:
    tree = _gm4_quartet()
    functions = (
        (Fraction(-2), Fraction(1), Fraction(3), Fraction(5)),
        (Fraction(7), Fraction(-1), Fraction(2), Fraction(4)),
        (Fraction(0), Fraction(3), Fraction(-5), Fraction(2)),
        (Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5)),
    )
    variables = tuple(zip(tree.leaf_order, functions, strict=True))
    joint = tree.leaf_joint_law()
    assert tree.leaf_moment(variables) == joint.moment(functions)
    assert tree.leaf_cumulant(variables) == joint.cumulant(functions)


def test_stochastic_leaf_observation_has_three_exact_computation_paths() -> None:
    tree = _gm4_quartet()
    channels = tuple(
        FiniteObservationChannel.from_rows(
            _stochastic_matrix(4, 20 + index), name=f"channel {index}"
        )
        for index in range(4)
    )
    output_functions = (
        (Fraction(-1), Fraction(2), Fraction(0), Fraction(5)),
        (Fraction(3), Fraction(4), Fraction(-2), Fraction(1)),
        (Fraction(7, 3), Fraction(-1, 2), Fraction(8), Fraction(0)),
        (Fraction(5), Fraction(2), Fraction(1), Fraction(-4)),
    )
    variables = tuple(
        (leaf, channel, function)
        for leaf, channel, function in zip(
            tree.leaf_order, channels, output_functions, strict=True
        )
    )
    via_messages = tree.observed_leaf_cumulant(variables)
    observed_law = transform_joint_law(tree.leaf_joint_law(), channels)
    via_observed_law = observed_law.cumulant(output_functions)
    via_basis = branching_observation_contraction(
        tree, variables, (_hadamard4(),) * 4, connected=True
    )
    assert via_messages == via_observed_law == via_basis


def test_every_tree_edge_has_exact_boundary_factorization_and_rank_bound() -> None:
    tree = _gm4_quartet()
    for child in (0, 1, 5, 2, 3):
        certificate = edge_flattening_certificate(tree, child)
        assert certificate.three_factorization_exact
        assert certificate.factorization_exact
        assert certificate.rank_bound_holds
        assert certificate.flattening_rank <= certificate.sharp_rank_bound
        assert certificate.sharp_rank_bound <= certificate.parent_state_size
        assert certificate.sharp_rank_bound <= certificate.boundary_state_size
        assert certificate.sharp_rank_bound <= certificate.transition_rank
    internal = edge_flattening_certificate(tree, 5)
    assert internal.parent == 4
    assert internal.left_leaves == (0, 1)
    assert internal.right_leaves == (2, 3)
    assert internal.flattening_rank == 4


def test_edge_transition_rank_gives_sharper_flattening_bound() -> None:
    tree = _gm4_quartet()
    rank_one = tuple(
        (Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10))
        for _ in range(4)
    )
    transitions = list(tree.transitions)
    transitions[5] = rank_one
    reduced = FiniteMarkovTree(
        tree.state_sizes,
        tree.root,
        tree.root_distribution,
        tree.parent,
        tuple(transitions),
        tree.leaf_order,
        name="rank-one internal edge",
    )
    certificate = edge_flattening_certificate(reduced, 5)
    assert certificate.transition_rank == 1
    assert certificate.sharp_rank_bound == 1
    assert certificate.flattening_rank == 1
    assert certificate.three_factorization_exact


def test_generic_gm4_true_split_rank_four_false_splits_full_rank() -> None:
    law = _gm4_quartet().leaf_joint_law()
    assert tensor_flattening(law, (0, 1)).exact_rank == 4
    assert tensor_flattening(law, (0, 2)).exact_rank == 16
    assert tensor_flattening(law, (0, 3)).exact_rank == 16


def test_full_latent_rank_channels_preserve_every_flattening_rank_and_recover_law() -> None:
    law = _gm4_quartet().leaf_joint_law()
    channels = tuple(
        FiniteObservationChannel.from_rows(
            _stochastic_matrix(4, 40 + index), name=f"invertible {index}"
        )
        for index in range(4)
    )
    observed = transform_joint_law(law, channels)
    recovered = recover_latent_joint_law(observed, channels)
    assert recovered == law
    for split in ((0, 1), (0, 2), (0, 3)):
        certificate = observation_flattening_certificate(law, channels, split)
        assert certificate.identity_exact
        assert certificate.full_latent_rank_channels
        assert certificate.rank_nonincrease
        assert certificate.rank_preserved_when_injective
        assert certificate.observed_rank == certificate.latent_rank


def test_rectangular_overcomplete_channel_can_still_preserve_latent_rank() -> None:
    channel = FiniteObservationChannel.from_rows(
        (
            (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
            (Fraction(1, 5), Fraction(1, 5), Fraction(3, 5)),
        )
    )
    assert channel.output_states == 3
    assert channel.has_full_latent_rank
    inverse = channel.left_inverse_operator()
    assert exact_matrix_rank(inverse) == 2


def test_rank_deficient_channel_has_exact_distribution_collision() -> None:
    recoding = FiniteObservationChannel.deterministic(
        (0, 1, 0, 1), name="two-class recoding"
    )
    witness = recoding.collision_witness()
    assert witness["distinct"]
    assert witness["collision_exact"]
    assert witness["first"] != witness["second"]
    assert witness["observed_first"] == witness["observed_second"]


def test_single_binary_recodings_destroy_gm4_edge_rank_discrimination() -> None:
    law = _gm4_quartet().leaf_joint_law()
    recoding = FiniteObservationChannel.deterministic(
        (0, 1, 0, 1), name="M/K-like binary recoding"
    )
    for split in ((0, 1), (0, 2), (0, 3)):
        certificate = observation_flattening_certificate(
            law, (recoding,) * 4, split
        )
        assert certificate.identity_exact
        assert certificate.observed.shape == (4, 4)
        assert certificate.observed_rank == 4
        # The GM(4) edge bound is now the maximum possible matrix rank.
        assert certificate.observed.tail_frobenius(4) == 0.0


def test_two_complementary_binary_sensors_restore_collective_observability() -> None:
    channels = (
        FiniteObservationChannel.deterministic((0, 1, 0, 1), name="M/K"),
        FiniteObservationChannel.deterministic((0, 0, 1, 1), name="R/Y"),
        FiniteObservationChannel.deterministic((0, 1, 1, 0), name="W/S"),
    )
    assert [channel.rank for channel in channels] == [2, 2, 2]
    assert minimum_collectively_injective_channel_sets(channels) == (
        (0, 1),
        (0, 2),
        (1, 2),
    )
    joint = FiniteObservationChannel.jointly_conditionally_independent(
        channels[:2]
    )
    assert joint.output_states == 4
    assert joint.rank == 4
    assert joint.has_full_latent_rank

    law = _gm4_quartet().leaf_joint_law()
    certificate = observation_flattening_certificate(
        law, (joint,) * 4, (0, 2)
    )
    assert certificate.identity_exact
    assert certificate.latent_rank == certificate.observed_rank == 16


def test_singular_value_tail_transfer_bounds_hold_for_invertible_channels() -> None:
    law = _gm4_quartet().leaf_joint_law()
    channels = tuple(
        FiniteObservationChannel.from_rows(_stochastic_matrix(4, 60 + index))
        for index in range(4)
    )
    certificate = observation_flattening_certificate(law, channels, (0, 2))
    bounds = singular_value_observation_bounds(certificate, rank_bound=4)
    assert bounds["latent_tail_frobenius"] > 0
    assert bounds["observed_tail_frobenius"] > 0
    assert bounds["lower_bound_applicable"]
    assert bounds["bounds_hold"]


def test_heterogeneous_state_tree_supported() -> None:
    # Root has two states, internal boundary three, and leaves heterogeneous.
    tree = FiniteMarkovTree.from_edges(
        (2, 3, 2, 3, 2),
        root=0,
        root_distribution=(Fraction(2, 5), Fraction(3, 5)),
        edges=(
            (0, 1, ((Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
                    (Fraction(1, 5), Fraction(2, 5), Fraction(2, 5)))),
            (0, 2, ((Fraction(3, 4), Fraction(1, 4)),
                    (Fraction(1, 3), Fraction(2, 3)))),
            (1, 3, ((Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
                    (Fraction(1, 4), Fraction(1, 2), Fraction(1, 4)),
                    (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)))),
            (1, 4, ((Fraction(2, 3), Fraction(1, 3)),
                    (Fraction(1, 2), Fraction(1, 2)),
                    (Fraction(1, 4), Fraction(3, 4)))),
        ),
        leaf_order=(2, 3, 4),
    )
    law = tree.leaf_joint_law()
    assert law.state_sizes == (2, 3, 2)
    variables = (
        (2, (Fraction(-1), Fraction(2))),
        (3, (Fraction(0), Fraction(3), Fraction(5))),
        (4, (Fraction(7), Fraction(-2))),
    )
    assert tree.leaf_moment(variables) == law.moment(tuple(v for _, v in variables))
    certificate = edge_flattening_certificate(tree, 1)
    assert certificate.factorization_exact
    assert certificate.flattening_rank <= 3


def test_validation_guards() -> None:
    with pytest.raises(ValueError, match="probability vector"):
        FiniteObservationChannel.from_rows(((Fraction(1, 3), Fraction(1, 3)),))
    with pytest.raises(ValueError, match="not injective"):
        FiniteObservationChannel.deterministic((0, 0)).left_inverse_operator()
    with pytest.raises(ValueError, match="nontrivial"):
        tree = FiniteMarkovTree.from_edges(
            (2, 2),
            root=0,
            root_distribution=(Fraction(1, 2), Fraction(1, 2)),
            edges=((0, 1, ((1, 0), (0, 1))),),
            leaf_order=(1,),
        )
        edge_flattening_certificate(tree, 1)
