# =============================================================================
# MathKernel - GF(2) conditioned orbit tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.gf2m import TransitionField
from mathkernel.gf2_conditioned import solve_orbit_access_bounded
from mathkernel.kernel import MathKernel

MASK=(1<<64)-1
def asr64(x,n):
    return ((x if x < (1<<63) else x-(1<<64)) >> n) & MASK
def dandelion_T(s):
    x=s&MASK; y=(s>>64)&MASK
    return (y ^ ((x<<7)&MASK)) | ((x ^ asr64(y,4))<<64)
DANDELION_COLS=[dandelion_T(1<<j) for j in range(128)]

def test_dandelion_transition_field_and_jump():
    tf=TransitionField(DANDELION_COLS)
    assert tf.m==128 and tf.field.verify_irreducible()
    s=0x123456789abcdef0|(0xfedcba9876543210<<64)
    for k in (1,7,487,4096):
        q=s
        for _ in range(k): q=dandelion_T(q)
        assert tf.apply_rows(tf.jump_rows(k),s)==q

def test_dandelion_bounded_access():
    tf=TransitionField(DANDELION_COLS)
    s=0x123456789abcdef0|(0xfedcba9876543210<<64)
    for k,limit in ((487,1000),(4096,5000),(100003,110000)):
        target=tf.apply_rows(tf.jump_rows(k),s)
        r=solve_orbit_access_bounded(DANDELION_COLS,s,target,limit)
        assert r is not None and r.lag==k and r.verified

def test_kernel_dandelion_access_exact():
    tf=TransitionField(DANDELION_COLS)
    s=0x123456789abcdef0|(0xfedcba9876543210<<64)
    target=tf.apply_rows(tf.jump_rows(487),s)
    r=MathKernel().gf2_conditioned_access(DANDELION_COLS,s,target,1000)
    assert r.ok and r.data["found"] and int(r.data["lag"])==487
    assert str(r.trust).lower().endswith("exact")
