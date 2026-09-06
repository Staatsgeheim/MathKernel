# =============================================================================
# MathKernel - predictive closure tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.gf2_conditioned import (
 gf2_predictive_closure,gf2_jump_polynomial_support,verify_sparse_predictive_closure)
from mathkernel.kernel import MathKernel

MASK=(1<<64)-1
def asr64(x,n): return ((x if x<(1<<63) else x-(1<<64))>>n)&MASK
def T(s):
 a=s&MASK;b=(s>>64)&MASK
 return b | (((((a<<2)&MASK)^asr64(a,19)^b)&MASK)<<64)
COLS=[T(1<<j) for j in range(128)]

def test_shioi_sparse_giant_jump_is_i_plus_t():
 support=gf2_jump_polynomial_support(COLS,1<<64)
 assert support==(0,1)
 assert verify_sparse_predictive_closure(COLS,1<<64,support)

def test_shioi_closure_matrix_maps_to_state_xor_next():
 from mathkernel.gf2m import TransitionField
 c=gf2_predictive_closure(COLS,1<<64)
 f=TransitionField(COLS)
 for j in range(128):
  basis=1<<j
  got=f.apply_rows(list(c.jump_columns),basis)
  assert got == (basis^COLS[j])

def test_kernel_predictive_closure_exact():
 r=MathKernel().gf2_predictive_closure(COLS,1<<64)
 assert r.ok
 assert r.trust.value=="exact"
 assert r.data["closure_kind"]=="predictive"
 assert r.data["polynomial_support"]==["0","1"]
 assert r.data["sparse_identity_verified"] is True

def test_non_special_lag_is_still_exact_predictive_relation():
 r=MathKernel().gf2_predictive_closure(COLS,487)
 assert r.ok and r.trust.value=="exact"
 assert r.data["lag"]=="487"
