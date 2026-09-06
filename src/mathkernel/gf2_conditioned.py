# =============================================================================
# MathKernel - GF(2) state-conditioned orbit access
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from math import isqrt
from .gf2m import TransitionField

@dataclass(frozen=True)
class GF2OrbitAccess:
    lag: int
    state: int
    target: int
    degree: int
    verified: bool = True

def transition_columns_from_callable(degree:int, transition):
    mask=(1<<degree)-1
    return [int(transition(1<<j)) & mask for j in range(degree)]

def gf2_jump(columns:list[int], state:int, lag:int)->int:
    tf=TransitionField(columns)
    return tf.apply_rows(tf.jump_rows(int(lag)),int(state))

def _state_field_coords(tf:TransitionField,state:int)->int:
    # State coordinates in a cyclic *primal* basis can be recovered by using
    # jump images of a reference state. For orbit solving we avoid assuming
    # that tf.coords (which is dual-row coordinates) applies to state vectors.
    raise NotImplementedError

def solve_orbit_access_bounded(columns:list[int], state:int, target:int,
                               max_lag:int)->GF2OrbitAccess|None:
    """Exact bounded orbit solve T^k(state)=target using baby-step/giant-step on T.

    Works for arbitrary invertible GF(2)-linear T without a primal/dual basis
    convention. Complexity O(sqrt(max_lag)) stored states and jump applications.
    """
    if max_lag < 0: raise ValueError("max_lag must be nonnegative")
    tf=TransitionField(columns); m=tf.m; mask=(1<<m)-1
    state&=mask; target&=mask
    if state==target:return GF2OrbitAccess(0,state,target,m)
    q=isqrt(max_lag)+1
    # baby: T^j(state)
    baby={}
    cur=state
    one_rows=tf.jump_rows(1)
    for j in range(q):
        baby.setdefault(cur,j)
        cur=tf.apply_rows(one_rows,cur)
    # giant targets: T^{-iq}(target); order divides 2^m-1 for field construction.
    order=(1<<m)-1
    back_rows=tf.jump_rows((-q)%order)
    cur=target
    for i in range(q+1):
        j=baby.get(cur)
        if j is not None:
            k=i*q+j
            if k<=max_lag and tf.apply_rows(tf.jump_rows(k),state)==target:
                return GF2OrbitAccess(k,state,target,m)
        cur=tf.apply_rows(back_rows,cur)
    return None

def verify_orbit_access(columns:list[int],state:int,target:int,lag:int)->bool:
    tf=TransitionField(columns)
    return tf.apply_rows(tf.jump_rows(int(lag)),int(state))==int(target)


@dataclass(frozen=True)
class GF2PredictiveClosure:
    lag: int
    jump_columns: tuple[int, ...]
    relation: str
    exact: bool = True

def gf2_predictive_closure(columns: list[int], lag: int) -> GF2PredictiveClosure:
    """Return the exact linear state relation s[n+lag] = J_lag s[n]."""
    if lag < 0:
        raise ValueError("lag must be nonnegative")
    field = TransitionField([int(c) for c in columns])
    jump = tuple(field.jump_rows(int(lag)))
    return GF2PredictiveClosure(
        lag=int(lag), jump_columns=jump,
        relation="state[n+lag] = J_lag * state[n] over GF(2)", exact=True)

def gf2_jump_polynomial_support(columns: list[int], lag: int,
                                max_terms: int | None = None) -> tuple[int, ...]:
    """Express T^lag as a polynomial in T and return nonzero exponents.

    Uses the TransitionField reduction polynomial. This exposes sparse predictive
    closures such as T^K = I + T as support (0,1).
    """
    field = TransitionField([int(c) for c in columns])
    # In the field representation, x is the transition operator.
    from .gf2m import GF2mField
    x = 2
    ff = GF2mField(len(columns), field.red)
    v = ff.power(x, int(lag))
    support = tuple(i for i in range(len(columns)) if (v >> i) & 1)
    if max_terms is not None and len(support) > max_terms:
        return support
    return support

def verify_sparse_predictive_closure(columns: list[int], lag: int,
                                     support: list[int] | tuple[int,...]) -> bool:
    """Prove T^lag == XOR_{j in support} T^j by exact matrix equality."""
    n=len(columns)
    lhs=list(gf2_predictive_closure(columns,lag).jump_columns)
    rhs=[0]*n
    for j in support:
        jj=list(gf2_predictive_closure(columns,int(j)).jump_columns)
        rhs=[a^b for a,b in zip(rhs,jj)]
    return lhs==rhs
