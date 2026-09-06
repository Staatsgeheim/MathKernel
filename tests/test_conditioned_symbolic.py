# =============================================================================
# MathKernel - conditioned symbolic tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.conditioned_dynamics import (
    symbolic_affine_access, prove_factor_swap_conditioned_closure, fold_product
)
from mathkernel.kernel import MathKernel

def test_symbolic_xor_access_matches_closed_form():
    w=64; M=1<<w
    A=0x2d358dccaa6c78a5; C=0x8bb84b93962eacc9
    a=symbolic_affine_access(M,A,target_kind="xor",target_constant=C)
    inv=pow(A,-1,M)
    for x in (0,1,2,3,0x123456789abcdef0,(1<<64)-1):
        assert a.evaluate(x) == ((C-2*(x&C))*inv)%M

def test_factor_swap_discovery_current_wyrand():
    r=prove_factor_swap_conditioned_closure(
        64,0x2d358dccaa6c78a5,0x8bb84b93962eacc9)
    assert r["holds"] and r["symmetry"]["found"] and r["trust"]=="exact"

def test_factor_swap_discovery_old_wyrand():
    r=prove_factor_swap_conditioned_closure(
        64,0xa0761d6478bd642f,0xe7037ed1a0b428db)
    assert r["holds"]

def test_factor_swap_discovery_w1rand():
    c=0xd07ebc63274654c7
    r=prove_factor_swap_conditioned_closure(64,c,c)
    assert r["holds"]

def test_factor_swap_identity_numeric_sanity():
    w=16; c=0xacc9
    for x in range(1<<12):
        assert fold_product(x,x^c,w)==fold_product(x^c,x,w)

def test_kernel_discovery_is_exact():
    k=MathKernel()
    r=k.discover_factor_swap_conditioned_closure(
        64,0x2d358dccaa6c78a5,0x8bb84b93962eacc9)
    assert r.ok and str(r.trust).lower().endswith("exact")
    assert r.data["holds"]
