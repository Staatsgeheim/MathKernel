# =============================================================================
# MathKernel - test finite algebra
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import json

import pytest

from mathkernel.finite_algebra import (
    AbelianGroupDecomposition,
    FieldElement,
    FiniteFieldSpec,
    FiniteRingSpec,
    LIMITS,
    ModulePresentation,
    RingElement,
    create_finite_field,
    create_finite_ring,
    decompose_abelian_group,
    field_add,
    field_inverse,
    field_mul,
    field_pow,
    hermite_normal_form,
    ring_add,
    ring_inverse,
    ring_mul,
    ring_pow,
    smith_normal_form,
    verify_hnf,
    verify_snf,
)
from mathkernel.models import OperationStatus, TrustLevel


def gf(prime, coeffs):
    result = create_finite_field(prime, coeffs)
    assert result.status == OperationStatus.VERIFIED, result.conditions
    return result.value


# ---------------------------------------------------------------------------
# Finite rings Z/nZ
# ---------------------------------------------------------------------------

def test_zmod_construction_and_primality():
    ring = create_finite_ring(7)
    assert ring.status == OperationStatus.VERIFIED
    assert ring.trust == TrustLevel.EXACT.value
    assert ring.value.is_field
    composite = create_finite_ring(15)
    assert composite.status == OperationStatus.VERIFIED
    assert not composite.value.is_field
    assert composite.value.order == 15


def test_zmod_rejects_bad_modulus():
    assert create_finite_ring(1).status == OperationStatus.ERROR
    huge = 1 << (LIMITS["max_modulus_bits"] + 1)
    assert create_finite_ring(huge).status == OperationStatus.UNSUPPORTED


def test_zmod_arithmetic_exact():
    ring = create_finite_ring(12).value
    assert ring_add(ring, 10, 5).value.value == 3
    assert ring_mul(ring, 5, 7).value.value == 11
    assert ring_pow(ring, 5, 3).value.value == 5
    assert ring_pow(ring, 5, -1).value.value == 5  # 5 is a unit, 5^2 = 1
    assert ring_pow(ring, 2, -1).status == OperationStatus.DOES_NOT_EXIST


def test_zmod_inverse_units_and_nonunits():
    ring = create_finite_ring(12).value
    inv = ring_inverse(ring, 5)
    assert inv.status == OperationStatus.VERIFIED
    assert inv.value.value == 5  # 5*5 = 25 = 1 mod 12
    assert inv.verification["a_times_inverse_is_one"] is True
    nonunit = ring_inverse(ring, 8)
    assert nonunit.status == OperationStatus.DOES_NOT_EXIST
    assert "gcd" in nonunit.conditions[0]


def test_ring_element_reduces_input():
    ring = create_finite_ring(7).value
    element = RingElement(ring=ring, value=19)
    assert element.value == 5


# ---------------------------------------------------------------------------
# Finite fields GF(p^m)
# ---------------------------------------------------------------------------

def test_gf_prime_field():
    field = gf(5, [0, 1])  # f = x over GF(5)
    assert field.degree == 1
    assert field.order == 5
    assert field.basis == ["1"]
    assert field.irreducibility_verified


def test_gf_quadratic_extension_arithmetic():
    # x^2 + 2 has no root in GF(5), hence irreducible.
    field = gf(5, [2, 0, 1])
    assert field.basis == ["1", "x"]
    assert field.order == 25
    x = [0, 1]
    x_squared = field_mul(field, x, x)
    assert x_squared.status == OperationStatus.VERIFIED
    assert x_squared.value.coeffs == [3, 0]  # x^2 = -2 = 3 mod 5
    total = field_add(field, [1, 2], [4, 4])
    assert total.value.coeffs == [0, 1]


def test_gf_inverse_and_pow_roundtrip():
    field = gf(5, [2, 0, 1])
    a = [3, 2]
    inv = field_inverse(field, a)
    assert inv.status == OperationStatus.VERIFIED
    assert inv.verification["a_times_inverse_is_one"] is True
    product = field_mul(field, a, inv.value.coeffs)
    assert product.value.coeffs == [1, 0]
    assert field_pow(field, a, 0).value.coeffs == [1, 0]
    # a^(q-1) = 1 for nonzero a in GF(q)
    assert field_pow(field, a, 24).value.coeffs == [1, 0]
    assert field_pow(field, a, -1).value.coeffs == inv.value.coeffs


def test_gf_zero_inverse_does_not_exist():
    field = gf(5, [2, 0, 1])
    result = field_inverse(field, [0, 0])
    assert result.status == OperationStatus.DOES_NOT_EXIST


def test_gf_aes_field():
    # AES polynomial x^8 + x^4 + x^3 + x + 1 is irreducible over GF(2).
    field = gf(2, [1, 1, 0, 1, 1, 0, 0, 0, 1])
    assert field.order == 256
    a = [1, 1, 0, 0, 0, 0, 0, 0]
    inv = field_inverse(field, a)
    assert inv.status == OperationStatus.VERIFIED
    assert field_mul(field, a, inv.value.coeffs).value.coeffs == [1] + [0] * 7


def test_gf_reducible_polynomial_refuted():
    # x^2 + 1 = (x + 2)(x + 3) over GF(5).
    result = create_finite_field(5, [1, 0, 1])
    assert result.status == OperationStatus.REFUTED
    assert result.verification["irreducible"] is False


def test_gf_non_prime_characteristic_refuted():
    assert create_finite_field(9, [1, 1]).status == OperationStatus.REFUTED


def test_gf_non_monic_rejected():
    assert create_finite_field(5, [1, 2]).status == OperationStatus.ERROR


def test_gf_degree_limit_unsupported():
    degree = LIMITS["max_field_degree"] + 1
    coeffs = [1] + [0] * (degree - 1) + [1]
    assert create_finite_field(2, coeffs).status == OperationStatus.UNSUPPORTED


def test_field_element_validates_shape():
    field = gf(5, [2, 0, 1])
    with pytest.raises(ValueError):
        FieldElement(field=field, coeffs=[1, 2, 3])
    assert field_add(field, [1], [1, 2]).status == OperationStatus.ERROR


def test_irreducibility_witness_recorded():
    field = gf(3, [1, 1, 2, 1])  # x^3 + 2x^2 + x + 1 over GF(3)
    witness = field.irreducibility_witness
    assert witness["prime_divisors_of_degree"] == [3]
    assert witness["x_pow_p^m_mod_f"] == [0, 1]


# ---------------------------------------------------------------------------
# Smith normal form
# ---------------------------------------------------------------------------

def test_snf_diagonal_and_certificate():
    A = [[2, 4], [6, 8]]
    result = smith_normal_form(A)
    assert result.status == OperationStatus.VERIFIED
    assert all(result.verification.values())
    D, U, V = result.value["D"], result.value["U"], result.value["V"]
    assert [D[0][0], D[1][1]] == [2, 4]
    assert verify_snf(A, D, U, V) == result.verification
    cert = result.claim_evidence[result.query].certificate[0]
    assert cert.certificate_type == "smith_normal_form"
    assert cert.verified


def test_snf_rectangular():
    A = [[2, 4, 4], [6, 8, 10]]
    result = smith_normal_form(A)
    assert result.status == OperationStatus.VERIFIED
    D = result.value["D"]
    assert [D[0][0], D[1][1]] == [2, 2]
    assert all(result.verification.values())


def test_snf_singular_and_zero_matrices():
    singular = smith_normal_form([[1, 2], [2, 4]])
    assert singular.status == OperationStatus.VERIFIED
    D = singular.value["D"]
    assert [D[0][0], D[1][1]] == [1, 0]
    zero = smith_normal_form([[0, 0], [0, 0]])
    assert zero.status == OperationStatus.VERIFIED
    assert zero.value["D"] == [[0, 0], [0, 0]]


def test_snf_limits():
    dim = LIMITS["max_matrix_dim"] + 1
    big = [[0] * dim for _ in range(dim)]
    assert smith_normal_form(big).status == OperationStatus.UNSUPPORTED
    assert smith_normal_form([]).status == OperationStatus.ERROR
    assert smith_normal_form([[1, 2], [3]]).status == OperationStatus.ERROR


def test_snf_random_small_matrices_verify():
    # Deterministic pseudo-random sweep; every certificate must verify.
    state = 12345

    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) % (1 << 31)
        return state % 11 - 5

    for rows in range(1, 5):
        for cols in range(1, 5):
            A = [[rnd() for _ in range(cols)] for _ in range(rows)]
            result = smith_normal_form(A)
            assert result.status == OperationStatus.VERIFIED
            assert all(result.verification.values())


# ---------------------------------------------------------------------------
# Hermite normal form
# ---------------------------------------------------------------------------

def test_hnf_properties_and_certificate():
    A = [[2, 4], [6, 8]]
    result = hermite_normal_form(A)
    assert result.status == OperationStatus.VERIFIED
    assert all(result.verification.values())
    H, U = result.value["H"], result.value["U"]
    assert verify_hnf(A, H, U) == result.verification
    cert = result.claim_evidence[result.query].certificate[0]
    assert cert.certificate_type == "hermite_normal_form_row"
    assert cert.verified


def test_hnf_known_value():
    # Row HNF of [[1, 2], [3, 4]] is [[1, 0], [0, 2]].
    result = hermite_normal_form([[1, 2], [3, 4]])
    assert result.value["H"] == [[1, 0], [0, 2]]


def test_hnf_rank_deficient_zero_rows_at_bottom():
    result = hermite_normal_form([[1, 2], [2, 4], [3, 6]])
    assert result.status == OperationStatus.VERIFIED
    H = result.value["H"]
    assert H[1] == [0, 0] and H[2] == [0, 0]
    assert all(result.verification.values())


def test_hnf_random_small_matrices_verify():
    state = 999

    def rnd():
        nonlocal state
        state = (1103515245 * state + 12345) % (1 << 31)
        return state % 13 - 6

    for rows in range(1, 5):
        for cols in range(1, 5):
            A = [[rnd() for _ in range(cols)] for _ in range(rows)]
            result = hermite_normal_form(A)
            assert result.status == OperationStatus.VERIFIED
            assert all(result.verification.values())


# ---------------------------------------------------------------------------
# Module presentations and abelian group decomposition
# ---------------------------------------------------------------------------

def test_module_presentation_validation():
    with pytest.raises(ValueError):
        ModulePresentation(base_ring="zmod", generators=["a"])
    with pytest.raises(ValueError):
        ModulePresentation(generators=["a"], relations=[[1, 2]])
    zmod = ModulePresentation(base_ring="zmod", modulus=6, generators=["a"])
    assert zmod.modulus == 6


def test_abelian_group_torsion():
    presentation = ModulePresentation(
        generators=["a", "b"], relations=[[2, 0], [0, 3]])
    result = decompose_abelian_group(presentation)
    assert result.status == OperationStatus.VERIFIED
    group = result.value
    assert isinstance(group, AbelianGroupDecomposition)
    assert group.free_rank == 0
    assert group.torsion_invariants == [6] or group.torsion_invariants == [2, 3]
    # Invariant factors must form a divisibility chain; 2|3 fails, so the
    # SNF merges them into the single invariant factor 6.
    assert group.structure == "Z/6Z"


def test_abelian_group_mixed_free_and_torsion():
    presentation = ModulePresentation(
        generators=["a", "b", "c"], relations=[[2, 4, 0], [6, 8, 0]])
    result = decompose_abelian_group(presentation)
    assert result.status == OperationStatus.VERIFIED
    group = result.value
    assert group.free_rank == 1
    assert group.torsion_invariants == [2, 4]
    assert group.structure == "Z x Z/2Z x Z/4Z"
    assert result.verification["rank_plus_torsion_covers_generators"] is True


def test_abelian_group_free_only():
    presentation = ModulePresentation(generators=["a", "b"], relations=[])
    result = decompose_abelian_group(presentation)
    assert result.status == OperationStatus.VERIFIED
    assert result.value.structure == "Z^2"


def test_abelian_group_zmod_unsupported():
    presentation = ModulePresentation(
        base_ring="zmod", modulus=6, generators=["a"], relations=[[2]])
    result = decompose_abelian_group(presentation)
    assert result.status == OperationStatus.UNSUPPORTED
    assert "Z only" in result.conditions[0]


# ---------------------------------------------------------------------------
# Evidence and serialization
# ---------------------------------------------------------------------------

def test_results_carry_exact_evidence():
    result = smith_normal_form([[2, 4], [6, 8]])
    bundle = result.claim_evidence[result.query]
    assert bundle.conservative_trust() == TrustLevel.EXACT.value
    assert bundle.computation[0].arithmetic == "exact"
    assert bundle.certificate[0].verified


def test_results_json_roundtrip():
    for result in (
        create_finite_ring(12),
        create_finite_field(5, [2, 0, 1]),
        field_mul(gf(5, [2, 0, 1]), [1, 2], [3, 4]),
        smith_normal_form([[2, 4], [6, 8]]),
        hermite_normal_form([[1, 2], [3, 4]]),
        decompose_abelian_group(ModulePresentation(
            generators=["a", "b"], relations=[[2, 0], [0, 3]])),
    ):
        payload = json.loads(result.model_dump_json())
        assert payload["status"] in {s.value for s in OperationStatus}
        assert payload["trust"] == TrustLevel.EXACT.value
        assert payload["limits"]["max_matrix_dim"] == LIMITS["max_matrix_dim"]
