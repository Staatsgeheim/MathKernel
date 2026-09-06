# =============================================================================
# MathKernel - vector structural synthesis tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.structural_discovery import Var,Const,Op,synthesize_vector_symmetries
from mathkernel.gf2m import TransitionField
from mathkernel.kernel import MathKernel
MASK=(1<<64)-1
def asr64(x,n): return ((x if x<(1<<63) else x-(1<<64))>>n)&MASK
def dT(s):
 x=s&MASK;y=(s>>64)&MASK
 return (y^((x<<7)&MASK)) | ((x^asr64(y,4))<<64)
COLS=[dT(1<<j) for j in range(128)]
def A(e):
 if e[0]=="var":return {"op":"var","name":e[1]}
 if e[0]=="const":return {"op":"const","value":e[1]}
 return {"op":e[0],"args":[A(a) for a in e[1:]]}

def test_vector_synthesis_finds_planted_swap_symmetry():
 x0,x1=Var("x0"),Var("x1")
 out=Op("xor",x0,x1)
 r=synthesize_vector_symmetries([out],6,[],probe_samples=256,proof_exhaustive_bits=12)
 assert any(s["name"]=="swap" and s["trust"]=="exact" for s in r["exact_symmetries"])

def test_vector_synthesis_finds_planted_xor_both():
 x0,x1=Var("x0"),Var("x1"); c=0x15
 # Difference is invariant under XORing both words by same C.
 out=Op("xor",x0,x1)
 r=synthesize_vector_symmetries([out],6,[c],probe_samples=256,proof_exhaustive_bits=12)
 assert any(s["name"]==f"xor_both({c})" for s in r["exact_symmetries"])

def test_dandelion_vector_search_runs_without_claiming_exact_from_sampling():
 x0,x1=Var("x0"),Var("x1")
 sq=Op("mul",x0,x0)
 out=Op("xor",Op("add",sq,x1),Op("fold",sq))
 r=synthesize_vector_symmetries([out],64,[],probe_samples=128,proof_exhaustive_bits=16)
 assert r["ranking_trust"]=="numeric"
 assert r["exact_symmetries"]==[]

def test_kernel_vector_to_gf2_bridge():
 # Use observable x0 xor x1; swap is exact structurally at reduced proof widths,
 # and test the full bridge on an actual Dandelion state with bounded search.
 x0,x1=Var("x0"),Var("x1")
 outputs=[A(Op("xor",x0,x1))]
 s=[0x123456789abcdef0,0xfedcba9876543210]
 r=MathKernel().synthesize_gf2_vector_conditioned_access(
     outputs,64,[],COLS,s,5000,probe_samples=128,top_k=16)
 assert r.ok and r.data["candidate_count"]>0
