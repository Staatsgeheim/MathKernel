# =============================================================================
# MathKernel - structural discovery tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.structural_discovery import (
    Var,Const,Op,canonical,substitute,discover_symmetries,folded_mul_x_xor_c
)
from mathkernel.kernel import MathKernel

def expr(c):
    return {"op":"fold","args":[{"op":"mul","args":[
        {"op":"var","name":"x"},
        {"op":"xor","args":[{"op":"var","name":"x"},{"op":"const","value":c}]}
    ]}]}

def test_generic_discovery_current_wyrand():
    c=0x8bb84b93962eacc9
    r=discover_symmetries(folded_mul_x_xor_c(c),64,[c])
    assert [s["name"] for s in r["symmetries"]]==[f"xor({c})"]

def test_generic_discovery_w1rand():
    c=0xd07ebc63274654c7
    r=discover_symmetries(folded_mul_x_xor_c(c),64,[c])
    assert any(s["name"]==f"xor({c})" for s in r["symmetries"])

def test_no_false_symmetry_for_asymmetric_variant_shape():
    # WyRandA-like old-state/new-state expression:
    # fold(x * ((x+A) xor C)); factor swap no longer maps back to itself.
    x=Var(); A=0x2d358dccaa6c78a5; C=0x8bb84b93962eacc9
    e=Op("fold",Op("mul",x,Op("xor",Op("add",x,Const(A)),Const(C))))
    r=discover_symmetries(e,64,[A,C])
    assert not any(s["name"]==f"xor({C})" for s in r["symmetries"])

def test_kernel_ast_to_conditioned_closure_current_wyrand():
    A=0x2d358dccaa6c78a5; C=0x8bb84b93962eacc9
    r=MathKernel().discover_structural_conditioned_closure(expr(C),64,A,[A,C])
    assert r.ok
    assert any(c["symmetry"]==f"xor({C})" for c in r.data["conditioned_closures"])
    assert str(r.trust).lower().endswith("exact")

def test_kernel_ast_historical_and_w1rand():
    for A,C in [(0xa0761d6478bd642f,0xe7037ed1a0b428db),
                (0xd07ebc63274654c7,0xd07ebc63274654c7)]:
        r=MathKernel().discover_structural_conditioned_closure(expr(C),64,A,[A,C])
        assert r.ok and r.data["conditioned_closures"]
