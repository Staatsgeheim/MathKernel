# =============================================================================
# MathKernel - test finite groups
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import json

import pytest

from mathkernel.finite_groups import (
    LIMITS,
    MAX_CAYLEY_ORDER,
    FiniteAbelianGroup,
    FiniteGroup,
    GroupAction,
    GroupHomomorphism,
    PermutationGroup,
)
from mathkernel.models import OperationStatus, TrustLevel


# Helpers ---------------------------------------------------------------------
def compose(p, q):
    """Permutation composition (p o q)(x) = p[q[x]] in image-list form."""
    return [p[q[x]] for x in range(len(p))]


def perm_inverse(p):
    inv = [0] * len(p)
    for x, y in enumerate(p):
        inv[y] = x
    return inv


def group_from_permutations(perms):
    """FiniteGroup whose elements are the given permutations (closed set)."""
    n = len(perms)
    index = {tuple(p): i for i, p in enumerate(perms)}
    table = [[index[tuple(compose(perms[i], perms[j]))] for j in range(n)]
             for i in range(n)]
    identity = index[tuple(range(len(perms[0])))]
    return FiniteGroup(order=n, identity=identity, cayley_table=table)


def s3():
    """S3 with elements [e, r, r^2, s0, s1, s2] (r = 3-cycle, s* transpositions)."""
    e = [0, 1, 2]
    r = [1, 2, 0]
    r2 = [2, 0, 1]
    s0 = [1, 0, 2]
    s1 = [2, 1, 0]
    s2 = [0, 2, 1]
    return group_from_permutations([e, r, r2, s0, s1, s2])


E, R, R2, S0, S1, S2 = range(6)
A3 = [E, R, R2]


# FiniteGroup validation --------------------------------------------------------
def test_cyclic_group_constructs_and_validates():
    g = FiniteGroup.cyclic(6)
    assert g.order == 6
    assert g.inverses == [0, 5, 4, 3, 2, 1]
    assert g.is_abelian()


def test_invalid_tables_rejected():
    with pytest.raises(ValueError):
        FiniteGroup(order=2, identity=1, cayley_table=[[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        # Row 0 is not a permutation.
        FiniteGroup(order=2, identity=0, cayley_table=[[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        # Non-associative loop of order 5 (Latin, identity 0):
        # (1*1)*2 = 0*2 = 2 but 1*(1*2) = 1*3 = 4.
        FiniteGroup(order=5, identity=0, cayley_table=[
            [0, 1, 2, 3, 4],
            [1, 0, 3, 4, 2],
            [2, 4, 0, 1, 3],
            [3, 2, 4, 0, 1],
            [4, 3, 1, 2, 0],
        ])
    with pytest.raises(ValueError):
        FiniteGroup(order=3, identity=0,
                    cayley_table=[[0, 1], [1, 2], [2, 0]])
    with pytest.raises(ValueError):
        FiniteGroup(order=MAX_CAYLEY_ORDER + 1, identity=0, cayley_table=[])


def test_order_result_carries_exact_evidence():
    result = FiniteGroup.cyclic(5).order_result()
    assert result.status == OperationStatus.VERIFIED
    assert result.trust == TrustLevel.EXACT
    assert result.value == 5
    assert result.evidence.computation[0].trust == "exact"
    assert result.evidence.proof[0].verified


# Subgroup machinery ------------------------------------------------------------
def test_generated_subgroup_and_element_orders():
    g = FiniteGroup.cyclic(6)
    assert g.element_order(2) == 3
    assert g.generated_subgroup([2]).elements == [0, 2, 4]
    assert g.generated_subgroup([2, 3]).elements == list(range(6))


def test_all_subgroups_of_s3():
    result = s3().all_subgroups()
    assert result.status == OperationStatus.VERIFIED
    assert result.complete
    assert result.trust == TrustLevel.EXACT
    assert sorted(info.order for info in result.subgroups) == [1, 2, 2, 2, 3, 6]


def test_subgroup_enumeration_limit_is_explicit():
    g = FiniteGroup.cyclic(MAX_CAYLEY_ORDER)
    result = g.all_subgroups()
    assert result.status == OperationStatus.UNSUPPORTED
    assert not result.complete
    assert result.limits["max_subgroup_enum_order"] == LIMITS[
        "max_subgroup_enum_order"]


def test_cosets_partition_group():
    g = s3()
    left = g.cosets(A3, side="left")
    assert left.status == OperationStatus.VERIFIED
    assert left.cosets == [A3, [S0, S1, S2]]
    right = g.cosets(A3, side="right")
    assert right.cosets == left.cosets
    # Cosets of a non-normal subgroup differ by side but still partition.
    h = [E, S0]
    left_h = g.cosets(h, side="left").cosets
    assert sorted(map(sorted, left_h)) == [[E, S0], [R, S1], [R2, S2]]
    assert all(len(c) == 2 for c in left_h)


def test_normality_and_quotient():
    g = s3()
    normal = g.is_normal(A3)
    assert normal.is_normal
    assert normal.evidence.certificate[0].verified
    assert not g.is_normal([E, S0]).is_normal

    quotient = g.quotient(A3)
    assert quotient.status == OperationStatus.VERIFIED
    assert quotient.trust == TrustLevel.EXACT
    assert quotient.quotient is not None
    assert quotient.quotient.order == 2
    # The quotient table was re-validated by the FiniteGroup constructor.
    assert quotient.quotient.cayley_table == [[0, 1], [1, 0]]

    undefined = g.quotient([E, S0])
    assert undefined.status == OperationStatus.DOES_NOT_EXIST
    assert undefined.quotient is None


def test_center_centralizer_conjugacy_commutator():
    g = s3()
    assert g.center().elements == [E]
    assert g.centralizer([S0]).elements == [E, S0]
    assert g.centralizer([R]).elements == A3

    classes = g.conjugacy_classes()
    assert classes.status == OperationStatus.VERIFIED
    assert classes.classes == [[E], [R, R2], [S0, S1, S2]]
    assert sum(classes.class_sizes) == 6

    commutator = g.commutator_subgroup()
    assert commutator.elements == A3
    assert commutator.order == 3

    # Abelian group: center is everything, commutator trivial, classes singletons.
    c4 = FiniteGroup.cyclic(4)
    assert c4.center().order == 4
    assert c4.commutator_subgroup().elements == [0]
    assert c4.conjugacy_classes().class_sizes == [1, 1, 1, 1]


# Group actions ---------------------------------------------------------------
def test_group_action_orbits_and_stabilizers():
    c2 = FiniteGroup.cyclic(2)
    action = GroupAction(group=c2, n_points=3,
                         action=[[0, 1, 2], [1, 0, 2]])
    orbits = action.orbits()
    assert orbits.status == OperationStatus.VERIFIED
    assert orbits.orbits == [[0, 1], [2]]

    stab_fixed = action.stabilizer(2)
    assert stab_fixed.elements == [0, 1]
    assert stab_fixed.trust == TrustLevel.EXACT

    stab_moved = action.stabilizer(0)
    assert stab_moved.elements == [0]
    # Orbit-stabilizer verification metadata is present.
    proof = stab_moved.evidence.proof[0]
    assert proof.verified
    assert "orbit-stabilizer" in proof.proposition


def test_invalid_action_rejected():
    c2 = FiniteGroup.cyclic(2)
    with pytest.raises(ValueError):
        GroupAction(group=c2, n_points=2, action=[[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        # Rows are permutations but the action law fails.
        GroupAction(group=c2, n_points=2, action=[[1, 0], [1, 0]])


# Homomorphisms ---------------------------------------------------------------
def test_sign_homomorphism_kernel_image():
    g = s3()
    c2 = FiniteGroup.cyclic(2)
    sign = GroupHomomorphism(domain=g, codomain=c2,
                             images=[0, 0, 0, 1, 1, 1])
    kernel = sign.kernel()
    assert kernel.elements == A3
    assert kernel.evidence.certificate[0].verified
    assert sign.image().elements == [0, 1]
    assert not sign.is_injective()
    assert sign.is_surjective()
    assert not sign.is_isomorphism()


def test_isomorphism_detected():
    c4 = FiniteGroup.cyclic(4)
    iso = GroupHomomorphism(domain=c4, codomain=c4, images=[0, 3, 2, 1])
    assert iso.is_isomorphism()


def test_non_homomorphism_rejected():
    g = s3()
    c2 = FiniteGroup.cyclic(2)
    with pytest.raises(ValueError):
        GroupHomomorphism(domain=g, codomain=c2, images=[0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError):
        GroupHomomorphism(domain=g, codomain=c2, images=[0, 0])


# Permutation groups ------------------------------------------------------------
def test_permutation_group_order_and_chain_certificate():
    g = PermutationGroup.symmetric(4)
    chain = g.stabilizer_chain()
    assert chain.status == OperationStatus.VERIFIED
    assert chain.verified
    assert chain.order == 24
    product = 1
    for size in chain.transversal_sizes:
        product *= size
    assert product == 24
    cert = chain.evidence.certificate[0]
    assert cert.certificate_type == "stabilizer_chain"
    assert cert.verified
    assert cert.witness["transversal_product"] == 24

    order = g.order_result()
    assert order.value == 24
    assert order.trust == TrustLevel.EXACT


def test_permutation_group_membership():
    a3 = PermutationGroup(degree=3, generators=[[1, 2, 0]])
    assert a3.order_result().value == 3
    assert a3.contains([1, 2, 0]).contains
    assert not a3.contains([1, 0, 2]).contains
    with pytest.raises(ValueError):
        a3.contains([0, 0, 1])


def test_permutation_group_orbit_and_stabilizer():
    d4 = PermutationGroup.dihedral(4)
    assert d4.order_result().value == 8
    orbit = d4.orbit_of(0)
    assert orbit.orbits == [[0, 1, 2, 3]]
    stab = d4.stabilizer_of(0)
    assert stab.status == OperationStatus.VERIFIED
    assert stab.order == 2  # orbit-stabilizer: 4 * 2 = 8 re-verified


def test_permutation_group_validation_and_limits():
    with pytest.raises(ValueError):
        PermutationGroup(degree=3, generators=[[0, 0, 1]])
    with pytest.raises(ValueError):
        PermutationGroup(degree=3, generators=[[0, 1]])
    with pytest.raises(ValueError):
        PermutationGroup(degree=LIMITS["max_perm_degree"] + 1,
                         generators=[[]])


# Finite abelian groups ---------------------------------------------------------
def test_abelian_invariants_cyclic():
    result = FiniteAbelianGroup.from_finite_group(FiniteGroup.cyclic(6))
    assert result.status == OperationStatus.VERIFIED
    assert result.value is not None
    assert result.value.invariant_factors == [6]
    assert result.elementary_divisors == [2, 3]
    assert result.value.is_cyclic
    assert result.value.order == 6
    assert result.evidence.certificate[0].verified


def test_abelian_invariants_klein_four():
    # C2 x C2 on pairs (i, j) encoded as 2i + j.
    table = [[((i >> 1) ^ (j >> 1)) * 2 + ((i & 1) ^ (j & 1))
              for j in range(4)] for i in range(4)]
    g = FiniteGroup(order=4, identity=0, cayley_table=table)
    result = FiniteAbelianGroup.from_finite_group(g)
    assert result.status == OperationStatus.VERIFIED
    assert result.value.invariant_factors == [2, 2]
    assert result.elementary_divisors == [2, 2]
    assert not result.value.is_cyclic
    assert result.value.exponent == 2


def test_abelian_invariants_c2_times_c4():
    # C2 x C4 on pairs encoded as 4i + j.
    table = [[((i // 4) ^ (j // 4)) * 4 + ((i % 4) + (j % 4)) % 4
              for j in range(8)] for i in range(8)]
    g = FiniteGroup(order=8, identity=0, cayley_table=table)
    result = FiniteAbelianGroup.from_finite_group(g)
    assert result.status == OperationStatus.VERIFIED
    assert result.value.invariant_factors == [2, 4]
    assert result.elementary_divisors == [2, 4]


def test_abelian_invariants_reject_nonabelian():
    result = FiniteAbelianGroup.from_finite_group(s3())
    assert result.status == OperationStatus.DOES_NOT_EXIST
    assert result.value is None
    assert result.trust == TrustLevel.EXACT


def test_abelian_invariant_factor_validation():
    with pytest.raises(ValueError):
        FiniteAbelianGroup(invariant_factors=[4, 2])
    with pytest.raises(ValueError):
        FiniteAbelianGroup(invariant_factors=[1])
    trivial = FiniteAbelianGroup(invariant_factors=[])
    assert trivial.order == 1
    assert trivial.is_cyclic


# Serialization -----------------------------------------------------------------
def test_results_are_json_serializable():
    g = s3()
    payloads = [
        g.order_result(),
        g.all_subgroups(),
        g.cosets(A3),
        g.is_normal(A3),
        g.quotient(A3),
        g.center(),
        g.conjugacy_classes(),
        g.commutator_subgroup(),
        PermutationGroup.symmetric(3).stabilizer_chain(),
        FiniteAbelianGroup.from_finite_group(FiniteGroup.cyclic(6)),
    ]
    for result in payloads:
        round_tripped = json.loads(result.model_dump_json())
        assert round_tripped["status"] in {s.value for s in OperationStatus}
        assert round_tripped["trust"] in {t.value for t in TrustLevel}
        assert "evidence" in round_tripped
        assert "limits" in round_tripped
