# =============================================================================
# MathKernel - State-conditioned orbit access and closure machinery
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact state-conditioned orbit access for finite and affine cyclic dynamics.

An access map kappa:X->Z selects T^kappa(x)(x).  Constant kappa recovers
ordinary fixed-lag dynamics.  The routines here are deliberately exact and
enumerative; symbolic affine-cyclic solving is provided separately.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import gcd
from typing import Callable, Sequence

@dataclass(frozen=True)
class AccessMap:
    """Finite-state access map, represented by one nonnegative lag per state."""
    lags: tuple[int, ...]

    def __post_init__(self):
        if not self.lags:
            raise ValueError("access map must be nonempty")
        if any((not isinstance(k, int)) or k < 0 for k in self.lags):
            raise ValueError("access lags must be nonnegative integers")

    def compose(self, other: "AccessMap", transition: Sequence[int]) -> "AccessMap":
        """Return self star other: k(x)+lambda(T^k(x)x)."""
        if len(self.lags) != len(other.lags) or len(self.lags) != len(transition):
            raise ValueError("access maps and transition must have equal size")
        out=[]
        for x,k in enumerate(self.lags):
            y=iterate_transition(transition,x,k)
            out.append(k + other.lags[y])
        return AccessMap(tuple(out))

def iterate_transition(transition: Sequence[int], x: int, steps: int) -> int:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    # cycle-aware O(n) jump, so enormous conditioned lags are safe.
    seen: dict[int,int]={}
    path=[]
    cur=x
    i=0
    while i < steps:
        if cur in seen:
            start=seen[cur]; cycle=len(path)-start
            if cycle:
                rem=(steps-i)%cycle
                for _ in range(rem): cur=transition[cur]
                return cur
        seen[cur]=len(path); path.append(cur)
        cur=transition[cur]; i+=1
    return cur

def apply_access(transition: Sequence[int], access: AccessMap) -> list[int]:
    if len(transition) != len(access.lags):
        raise ValueError("transition and access map must have equal size")
    return [iterate_transition(transition,x,access.lags[x]) for x in range(len(transition))]

def solve_access_to_target(transition: Sequence[int], target: Sequence[int]) -> AccessMap:
    """Find the least k>=0 with T^k(x)=target[x], for every x."""
    n=len(transition)
    if len(target)!=n or any(t<0 or t>=n for t in target):
        raise ValueError("target must map every state to a valid state")
    lags=[]
    for x,t in enumerate(target):
        cur=x; seen=set(); k=0
        while cur not in seen:
            if cur==t:
                lags.append(k); break
            seen.add(cur); cur=transition[cur]; k+=1
        else:
            raise ValueError(f"target[{x}]={t} is not reachable from state {x}")
    return AccessMap(tuple(lags))

def observable_symmetry(observation: Sequence[int], transform: Sequence[int]) -> bool:
    n=len(observation)
    if len(transform)!=n or any(t<0 or t>=n for t in transform):
        raise ValueError("transform must map every state to a valid state")
    return all(observation[transform[x]]==observation[x] for x in range(n))

def conditioned_closure(observation: Sequence[int], transition: Sequence[int],
                        accesses: Sequence[AccessMap], coefficients: Sequence[int],
                        modulus: int, constant: int=0) -> dict:
    """Prove an additive conditioned closure by exhaustive exact enumeration."""
    if modulus < 2: raise ValueError("modulus must be >= 2")
    n=len(transition)
    if len(observation)!=n: raise ValueError("observation and transition size mismatch")
    if len(accesses)!=len(coefficients) or not accesses:
        raise ValueError("accesses and coefficients must be nonempty and equal length")
    advanced=[apply_access(transition,a) for a in accesses]
    residuals=[]
    for x in range(n):
        r=sum(int(h)*int(observation[advanced[j][x]]) for j,h in enumerate(coefficients))
        residuals.append((r-constant)%modulus)
    bad=[x for x,r in enumerate(residuals) if r]
    return {"holds":not bad,"checked_states":n,"counterexamples":bad[:32],
            "residuals":residuals if n<=256 else None}

def affine_cyclic_symmetry_access(modulus: int, increment: int,
                                  transform_values: Sequence[int]) -> AccessMap:
    """Solve x+k*A=S(x) (mod M) exactly for an enumerated symmetry S."""
    if modulus < 2 or len(transform_values)!=modulus:
        raise ValueError("transform_values must have length modulus")
    a=increment%modulus
    if gcd(a,modulus)!=1:
        raise ValueError("increment must be invertible modulo modulus")
    inv=pow(a,-1,modulus)
    return AccessMap(tuple((((int(transform_values[x])-x)%modulus)*inv)%modulus
                           for x in range(modulus)))

def affine_cyclic_access_formula(modulus: int, increment: int,
                                 state: int, target_state: int) -> int:
    """Single-state exact solution k=(target-state)*A^-1 mod M."""
    if modulus < 2: raise ValueError("modulus must be >= 2")
    a=increment%modulus
    if gcd(a,modulus)!=1:
        raise ValueError("increment must be invertible modulo modulus")
    return (((target_state-state)%modulus)*pow(a,-1,modulus))%modulus


@dataclass(frozen=True)
class SymbolicAccessMap:
    """Compact exact access map for affine cyclic dynamics."""
    variable: str
    modulus: int
    increment: int
    target_kind: str
    target_constant: int
    expression: str
    trust: str = "exact"

    def evaluate(self, state: int) -> int:
        m=self.modulus; x=state % m; c=self.target_constant % m
        if self.target_kind == "xor":
            target=x ^ c
        elif self.target_kind == "add":
            target=(x+c)%m
        else:
            raise ValueError(f"unsupported target kind {self.target_kind!r}")
        return affine_cyclic_access_formula(m,self.increment,x,target)

def symbolic_affine_access(modulus: int, increment: int, *,
                           target_kind: str, target_constant: int,
                           variable: str="x") -> SymbolicAccessMap:
    """Derive a compact exact kappa(x) for x+kA=S(x) mod M."""
    if modulus < 2 or gcd(increment % modulus, modulus) != 1:
        raise ValueError("increment must be invertible modulo modulus")
    c=target_constant % modulus
    inv=pow(increment % modulus,-1,modulus)
    if target_kind == "xor":
        # (x xor C)-x = C-2(x&C), exact over integers before reduction.
        expr=f"(({c} - 2*bitand({variable},{c}))*{inv}) mod {modulus}"
    elif target_kind == "add":
        expr=f"({c}*{inv}) mod {modulus}"
    else:
        raise ValueError("target_kind must be 'xor' or 'add'")
    return SymbolicAccessMap(variable,modulus,increment%modulus,target_kind,c,expr)

def fold_product(value_a: int, value_b: int, word_bits: int) -> int:
    """Fold a 2w-bit product by XORing its low/high w-bit halves."""
    if word_bits < 1: raise ValueError("word_bits must be positive")
    mask=(1<<word_bits)-1
    p=(value_a & mask)*(value_b & mask)
    return ((p & mask) ^ (p >> word_bits)) & mask

def discover_factor_swap_symmetry(word_bits: int, xor_constant: int) -> dict:
    """Exact structural discovery for F(x)=fold(x*(x xor C)).

    The proof is syntactic/algebraic: S(x)=x xor C swaps the two
    multiplicative factors; commutativity makes the folded product identical.
    """
    if word_bits < 1: raise ValueError("word_bits must be positive")
    m=1<<word_bits; c=xor_constant & (m-1)
    return {
        "found": True,
        "transform": {"kind":"xor","constant":str(c),"expression":f"x xor {c}"},
        "identity": "F(S(x)) = fold((x xor C)*x) = fold(x*(x xor C)) = F(x)",
        "proof_rule": "xor-involution + multiplication-commutativity + deterministic-fold",
        "trust": "exact",
    }

def prove_factor_swap_conditioned_closure(word_bits: int, increment: int,
                                           xor_constant: int) -> dict:
    """Discover the factor-swap symmetry and derive/prove its symbolic access."""
    m=1<<word_bits; c=xor_constant & (m-1)
    sym=discover_factor_swap_symmetry(word_bits,c)
    access=symbolic_affine_access(m,increment,target_kind="xor",
                                  target_constant=c)
    # Certificate obligations are exact algebraic identities, not samples.
    return {
        "holds": True,
        "word_bits": word_bits,
        "modulus": str(m),
        "symmetry": sym,
        "access_expression": access.expression,
        "access": access,
        "closure": "F(T^kappa(x)(x)) = F(x)",
        "obligations": [
            "(x xor C) xor C = x",
            "(x xor C)-x = C-2*(x&C)",
            "A*A^-1 = 1 (mod 2^w)",
            "fold((x xor C)*x) = fold(x*(x xor C))",
        ],
        "trust": "exact",
    }
