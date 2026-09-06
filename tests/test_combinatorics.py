# =============================================================================
# MathKernel - focused exact combinatorics / generating-function tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import json
from math import comb, factorial

import pytest

from mathkernel.combinatorics import (
    CombinatorialClass,
    CombinatorialKind,
    CombinatoricsEngine,
    GeneratingFunction,
    GfKind,
    LinearRecurrence,
    MAX_EXACT_ARGUMENT,
    RationalGeneratingFunction,
    bell_number,
    binomial,
    catalan_number,
    compositions_count,
    derangements_count,
    integer_partitions_count,
    multinomial,
    permutations_count,
    set_partitions_count,
    stirling_first,
    stirling_second,
)
from mathkernel.models import OperationStatus, TrustLevel

engine = CombinatoricsEngine()


# ---------------------------------------------------------------------------
# Exact counts
# ---------------------------------------------------------------------------

def test_permutation_and_combination_counts():
    assert permutations_count(5) == 120
    assert permutations_count(5, 2) == 20
    assert binomial(5, 2) == 10
    assert binomial(5, 7) == 0
    assert multinomial([2, 1]) == 3
    assert multinomial([3, 2, 1]) == 60


def test_stirling_bell_catalan_counts():
    assert stirling_second(5, 2) == 15
    assert stirling_first(4, 2) == 11
    assert stirling_first(4, 2, signed=True) == 11  # (-1)^(4-2) = +1
    assert stirling_first(4, 1, signed=True) == -6
    assert bell_number(5) == 52
    assert catalan_number(5) == 42
    assert catalan_number(0) == 1


def test_partition_composition_derangement_counts():
    assert integer_partitions_count(5) == 7
    assert integer_partitions_count(5, 2) == 2
    assert integer_partitions_count(200) == 3972999029388
    assert set_partitions_count(4) == 15
    assert set_partitions_count(4, 2) == 7
    assert compositions_count(4) == 8
    assert compositions_count(4, 2) == 3
    assert compositions_count(0) == 1
    assert derangements_count(4) == 9
    assert derangements_count(0) == 1
    assert derangements_count(1) == 0


def test_counts_reject_out_of_range_arguments_explicitly():
    with pytest.raises(ValueError, match="explicitly"):
        permutations_count(MAX_EXACT_ARGUMENT + 1)


def test_count_result_carries_exact_evidence():
    result = CombinatorialClass(
        kind=CombinatorialKind.COMBINATIONS, n=10, k=3).count()
    assert result.status == OperationStatus.AVAILABLE
    assert result.value == 120
    assert result.trust == TrustLevel.EXACT
    assert result.evidence.conservative_trust() == "exact"


# ---------------------------------------------------------------------------
# Lazy generation with explicit limits
# ---------------------------------------------------------------------------

def test_generation_is_capped_and_reports_truncation():
    result = CombinatorialClass(
        kind=CombinatorialKind.PERMUTATIONS, n=10, max_items=5).generate()
    assert result.returned == 5
    assert result.truncated is True
    assert result.total_count == factorial(10)
    assert result.status == OperationStatus.AVAILABLE
    assert "enumeration_matches_count" not in result.checks


def test_full_generation_is_verified_against_exact_count():
    result = CombinatorialClass(
        kind=CombinatorialKind.DERANGEMENTS, n=4).generate()
    assert result.status == OperationStatus.VERIFIED
    assert result.truncated is False
    assert result.returned == 9
    assert result.checks["enumeration_matches_count"] is True
    assert result.evidence.certificate[0].verified is True
    assert all(all(p[i] != i for i in range(4)) for p in result.items)


def test_combinations_and_multiset_generation():
    combos = CombinatorialClass(
        kind=CombinatorialKind.COMBINATIONS, n=5, k=2).generate()
    assert combos.returned == 10
    assert (0, 1) in combos.items and (3, 4) in combos.items

    multisets = CombinatorialClass(
        kind=CombinatorialKind.MULTISET_PERMUTATIONS,
        multiplicities=[2, 1]).generate()
    assert multisets.returned == 3
    assert sorted(multisets.items) == [(0, 0, 1), (0, 1, 0), (1, 0, 0)]


def test_integer_partition_set_partition_and_composition_generation():
    partitions = CombinatorialClass(
        kind=CombinatorialKind.INTEGER_PARTITIONS, n=5).generate()
    assert partitions.returned == 7
    assert all(sum(p) == 5 for p in partitions.items)
    assert all(list(p) == sorted(p, reverse=True) for p in partitions.items)

    set_partitions = CombinatorialClass(
        kind=CombinatorialKind.SET_PARTITIONS, n=3).generate()
    assert set_partitions.returned == 5  # Bell(3), restricted-growth strings

    into_two = CombinatorialClass(
        kind=CombinatorialKind.SET_PARTITIONS, n=4, k=2).generate()
    assert into_two.returned == 7  # S(4, 2)
    assert all(len(set(rgs)) == 2 for rgs in into_two.items)

    compositions = CombinatorialClass(
        kind=CombinatorialKind.COMPOSITIONS, n=4).generate()
    assert compositions.returned == 8
    assert all(sum(c) == 4 for c in compositions.items)


def test_generation_beyond_explicit_depth_is_unsupported():
    result = CombinatorialClass(
        kind=CombinatorialKind.DERANGEMENTS, n=100).generate()
    assert result.status == OperationStatus.UNSUPPORTED
    assert result.diagnostics


def test_invalid_class_parameters_are_rejected():
    with pytest.raises(ValueError):
        CombinatorialClass(kind=CombinatorialKind.COMBINATIONS, n=5)
    with pytest.raises(ValueError):
        CombinatorialClass(kind=CombinatorialKind.PERMUTATIONS, n=3, k=4)


# ---------------------------------------------------------------------------
# Generating functions
# ---------------------------------------------------------------------------

def test_recurrence_converts_to_rational_ogf_and_back():
    fibonacci = LinearRecurrence(coefficients=[1, 1], initial=[0, 1])

    converted = engine.ogf_from_recurrence(fibonacci)

    assert converted.status == OperationStatus.AVAILABLE
    assert converted.rational is not None
    # sum F_n x^n = x / (1 - x - x^2) since F_0 = 0.
    assert converted.rational.numerator == [0, 1]
    assert converted.rational.denominator == [1, -1, -1]

    restored = engine.recurrence_from_ogf(converted.rational)
    assert restored.status == OperationStatus.AVAILABLE
    assert restored.recurrence == fibonacci


def test_polynomial_ogf_recurrence_conversion_is_unsupported():
    result = engine.recurrence_from_ogf(
        RationalGeneratingFunction(numerator=[1, 2], denominator=[1]))
    assert result.status == OperationStatus.UNSUPPORTED


def test_egf_recurrence_conversion_is_explicitly_unsupported():
    result = engine.egf_from_recurrence(
        LinearRecurrence(coefficients=[1, 1], initial=[0, 1]))
    assert result.status == OperationStatus.UNSUPPORTED
    assert "ordinary" in result.diagnostics[0]


def test_coefficient_extraction_from_prefix_recurrence_and_rational():
    gf = GeneratingFunction(
        kind=GfKind.ORDINARY,
        coefficients=[0, 1],
        recurrence=LinearRecurrence(coefficients=[1, 1], initial=[0, 1]),
        source="fibonacci",
    )
    assert engine.coefficient(gf, 0).value == 0
    assert engine.coefficient(gf, 10).value == 55
    assert engine.coefficient(gf, 10).status == OperationStatus.AVAILABLE

    rational_only = GeneratingFunction(
        kind=GfKind.ORDINARY,
        rational=RationalGeneratingFunction(
            numerator=[0, 1], denominator=[1, -1, -1]),
    )
    assert engine.coefficient(rational_only, 7).value == 13


def test_egf_coefficient_reports_factorial_scaling():
    egf = GeneratingFunction(
        kind=GfKind.EXPONENTIAL, coefficients=[1, 1, 2, 6, 24])
    result = engine.coefficient(egf, 3)
    assert result.value == 6
    assert result.scaling == "factorial"


def test_coefficient_beyond_prefix_without_extension_is_unknown():
    gf = GeneratingFunction(kind=GfKind.ORDINARY, coefficients=[1, 2, 3])
    result = engine.coefficient(gf, 5)
    assert result.status == OperationStatus.UNKNOWN
    assert result.value is None


def test_catalan_ogf_matches_catalan_recurrence():
    terms = 12
    gf = engine.catalan_ogf(terms)
    assert gf.coefficients == [catalan_number(i) for i in range(terms)]
    # Catalan convolution: C_n = sum_i C_i C_{n-1-i}.
    for n in range(1, terms):
        assert gf.coefficients[n] == sum(
            gf.coefficients[i] * gf.coefficients[n - 1 - i] for i in range(n))


def test_binomial_ogf_coefficients_match_binomial_coefficients():
    gf = engine.binomial_ogf(6)
    assert gf.coefficients == [comb(6, k) for k in range(7)]
    for k in range(7):
        assert engine.coefficient(gf, k).value == comb(6, k)


def test_generating_function_verification_against_recurrence_and_rational():
    converted = engine.ogf_from_recurrence(
        LinearRecurrence(coefficients=[1, 1], initial=[0, 1]))
    gf = GeneratingFunction(
        kind=GfKind.ORDINARY,
        coefficients=[0, 1, 1, 2, 3, 5, 8, 13],
        recurrence=LinearRecurrence(coefficients=[1, 1], initial=[0, 1]),
        rational=converted.rational,
        source="fibonacci",
    )
    result = engine.verify_generating_function(gf)
    assert result.status == OperationStatus.VERIFIED
    assert result.verified is True
    assert result.checks == {
        "recurrence_extension": True, "rational_expansion": True}
    assert result.evidence.certificate[0].verified is True

    corrupted = gf.model_copy(update={"coefficients": [0, 1, 1, 2, 3, 5, 8, 14]})
    refuted = engine.verify_generating_function(corrupted)
    assert refuted.status == OperationStatus.REFUTED
    assert refuted.verified is False


def test_verification_without_recurrence_or_rational_is_unknown():
    gf = GeneratingFunction(kind=GfKind.ORDINARY, coefficients=[1, 2, 3])
    assert engine.verify_generating_function(gf).status == OperationStatus.UNKNOWN


def test_count_identities_verify_at_several_n():
    for n in (0, 1, 5, 12):
        result = engine.verify_count_identities(n)
        assert result.status == OperationStatus.VERIFIED
        assert result.verified is True
        assert all(result.checks.values()), result.checks
        assert result.evidence.conservative_trust() == "exact"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_results_serialize_to_json():
    generation = CombinatorialClass(
        kind=CombinatorialKind.COMBINATIONS, n=5, k=2).generate()
    payload = json.loads(json.dumps(generation.model_dump(mode="json")))
    assert payload["status"] == "verified"
    assert payload["total_count"] == 10
    assert payload["items"][0] == [0, 1]

    count = CombinatorialClass(
        kind=CombinatorialKind.DERANGEMENTS, n=20).count()
    payload = json.loads(json.dumps(count.model_dump(mode="json")))
    assert payload["value"] == derangements_count(20)
    assert payload["trust"] == "exact"

    verification = engine.verify_count_identities(6)
    json.dumps(verification.model_dump(mode="json"))
