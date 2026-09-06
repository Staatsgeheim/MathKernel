# =============================================================================
# MathKernel - conditioned dynamics tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.conditioned_dynamics import (
    AccessMap, apply_access, solve_access_to_target, observable_symmetry,
    conditioned_closure, affine_cyclic_access_formula,
)
from mathkernel.kernel import MathKernel

def test_access_solver_and_composition():
    T=[1,2,3,0]
    target=[2,3,0,1]
    a=solve_access_to_target(T,target)
    assert a.lags==(2,2,2,2)
    b=AccessMap((1,1,1,1))
    assert a.compose(b,T).lags==(3,3,3,3)

def test_symmetry_to_exact_conditioned_closure():
    # O(x)=O(x xor 1), transition is the 4-cycle.
    T=[1,2,3,0]; O=[0,0,1,1]; S=[1,0,3,2]
    assert observable_symmetry(O,S)
    k=solve_access_to_target(T,S)
    z=AccessMap((0,0,0,0))
    result=conditioned_closure(O,T,[z,k],[1,-1],2)
    assert result["holds"]

def test_affine_formula_arbitrary_integer():
    M=2**127-1; A=17; x=123456789; target=987654321
    k=affine_cyclic_access_formula(M,A,x,target)
    assert (x+k*A)%M==target

def test_reduced_wyrand_factor_swap_symmetry():
    # Exact analogue of F(x)=fold(x*(x xor K)) => F(x)=F(x xor K).
    w=8; M=1<<w; K=0xC9; A=0xA5
    def F(x):
        p=x*(x^K)
        return ((p&(M-1))^(p>>w))&(M-1)
    O=[F(x) for x in range(M)]
    S=[x^K for x in range(M)]
    T=[(x+A)%M for x in range(M)]
    assert observable_symmetry(O,S)
    access=solve_access_to_target(T,S)
    assert all(TARGET == S[x] for x,TARGET in enumerate(apply_access(T,access)))
    zero=AccessMap((0,)*M)
    assert conditioned_closure(O,T,[zero,access],[1,-1],M)["holds"]

def test_kernel_api_exact_trust():
    k=MathKernel()
    created=k.finite_system_create("uniform",[1,2,3,0],[0,0,1,1])
    sid=created.data["system_id"]
    r=k.conditioned_symmetry_access(sid,[1,0,3,2])
    assert r.ok and str(r.trust).lower().endswith("exact")
    c=k.conditioned_closure(sid,[[0,0,0,0],r.data["lags"]],[1,-1],2)
    assert c.ok and c.data["holds"]
