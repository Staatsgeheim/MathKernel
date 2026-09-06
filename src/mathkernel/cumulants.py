# =============================================================================
# MathKernel - Algebraic joint cumulants via the set-partition lattice
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Algebraic joint cumulants via the set-partition lattice.

Joint cumulants are the "connected" part of raw multi-time moments:
Cum(Z_0..Z_{d-1}) removes every contribution explainable by lower-order
block moments. Both directions of the moment/cumulant inversion are exact
for any value type supporting + and * (int, Fraction, complex, sympy
expressions, cyclotomic numbers), so the same code serves symbolic work
and exact finite-system enumeration.

Conventions follow the partition formula
    Cum(Z) = sum_pi (|pi|-1)! (-1)^(|pi|-1) prod_{B in pi} E[prod_{j in B} Z_j]
with no implicit complex conjugation in any argument.

Subsets of {0..d-1} are encoded as integer bitmasks throughout: mask bit j
set <=> coordinate j belongs to the block. Mask 0 is the empty product,
whose expectation is 1 by convention.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from math import factorial
from typing import TypeVar

V = TypeVar("V")

MAX_ORDER = 8  # Bell(8) = 4140 partitions; Bell(9) = 21147 gets unwieldy


def set_partitions(n: int) -> list[list[tuple[int, ...]]]:
    """All set partitions of {0..n-1} as lists of sorted blocks.

    Enumerated by restricted-growth strings, so partitions appear in a
    canonical order and len(set_partitions(n)) == Bell(n).
    """
    if n < 0 or n > MAX_ORDER + 2:
        raise ValueError(f"partition order must be 0..{MAX_ORDER + 2}")
    if n == 0:
        return [[]]
    out: list[list[tuple[int, ...]]] = []
    rgs = [0] * n

    def rec(i: int, blocks: int) -> None:
        if i == n:
            built: list[list[int]] = [[] for _ in range(blocks)]
            for j, b in enumerate(rgs):
                built[b].append(j)
            out.append([tuple(b) for b in built])
            return
        for b in range(blocks + 1):
            rgs[i] = b
            rec(i + 1, max(blocks, b + 1))

    rec(1, 1)
    return out


def _block_mask(block: Iterable[int]) -> int:
    m = 0
    for j in block:
        m |= 1 << j
    return m


def cumulant_from_moments(moments: dict[int, V], d: int) -> V:
    """Joint cumulant of order d from raw block moments.

    `moments` maps subset bitmasks to E[prod_{j in mask} Z_j]; every subset
    of {0..d-1} must be present except mask 0, which defaults to 1.
    """
    if not 1 <= d <= MAX_ORDER:
        raise ValueError(f"cumulant order must be 1..{MAX_ORDER}")
    full = (1 << d) - 1
    missing = [m for m in range(1, full + 1) if m not in moments]
    if missing:
        raise ValueError(f"moments missing {len(missing)} subsets, e.g. mask {missing[0]}")
    total = None
    for pi in set_partitions(d):
        coef = factorial(len(pi) - 1) * (1 if len(pi) % 2 else -1)
        prod = None
        for block in pi:
            v = moments.get(_block_mask(block), 1)
            prod = v if prod is None else prod * v
        term = prod if coef == 1 else (-prod if coef == -1 else coef * prod)
        total = term if total is None else total + term
    return total


def moment_from_cumulants(cumulants: dict[int, V], d: int) -> V:
    """Raw d-point moment from connected block cumulants (inverse transform).

    `cumulants` maps nonempty subset bitmasks to Cum({Z_j}_{j in mask}).
    """
    if not 1 <= d <= MAX_ORDER:
        raise ValueError(f"cumulant order must be 1..{MAX_ORDER}")
    full = (1 << d) - 1
    missing = [m for m in range(1, full + 1) if m not in cumulants]
    if missing:
        raise ValueError(f"cumulants missing {len(missing)} subsets, e.g. mask {missing[0]}")
    total = None
    for pi in set_partitions(d):
        prod = None
        for block in pi:
            v = cumulants[_block_mask(block)]
            prod = v if prod is None else prod * v
        total = prod if total is None else total + prod
    return total


def connected_statistic(raw_moment: Callable[[tuple[int, ...]], V], d: int) -> V:
    """Connected statistic from a raw-moment callback over position subtuples.

    raw_moment(block_positions) must return E[prod_{j in block} Z_j] for any
    tuple of distinct positions drawn from {0..d-1}; the empty tuple returns 1.
    This is the blockwise raw->connected conversion (cumulant partition
    subtraction) applied when block moments are computed on demand rather
    than pre-tabulated by bitmask.
    """
    if not 1 <= d <= MAX_ORDER:
        raise ValueError(f"cumulant order must be 1..{MAX_ORDER}")
    total = None
    for pi in set_partitions(d):
        coef = factorial(len(pi) - 1) * (1 if len(pi) % 2 else -1)
        prod = None
        for block in pi:
            v = raw_moment(block) if block else 1
            prod = v if prod is None else prod * v
        term = prod if coef == 1 else (-prod if coef == -1 else coef * prod)
        total = term if total is None else total + term
    return total


def block_moments_from_samples(columns: list[list[V]]) -> dict[int, V]:
    """All raw block moments E[prod Z_j] from sample columns (exact averaging).

    columns[j] is the sample vector of Z_j; all columns must share a length.
    Values are summed exactly and divided by the sample count, so int/Fraction
    inputs yield exact rational results.
    """
    if not columns:
        raise ValueError("at least one sample column required")
    n = len(columns[0])
    if n == 0 or any(len(c) != n for c in columns):
        raise ValueError("sample columns must be nonempty and equal length")
    d = len(columns)
    # vectorized fast path only when the data is already inexact
    # (float/complex); pure-int columns keep exact rational results
    if any(isinstance(v, (float, complex)) for c in columns for v in c[:8]) and \
            all(isinstance(v, (int, float, complex)) and not isinstance(v, bool)
                for c in columns for v in c[:8]):
        import numpy as np
        try:
            cols = np.asarray(columns)
            if cols.dtype == object:
                cols = cols.astype(complex)
        except (TypeError, ValueError):
            cols = None
        if cols is not None:
            moments: dict[int, V] = {0: 1}
            for mask in range(1, 1 << d):
                idx = [j for j in range(d) if mask >> j & 1]
                moments[mask] = cols[idx].prod(axis=0).mean().item()
            return moments
    from fractions import Fraction
    moments = {0: 1}
    for mask in range(1, 1 << d):
        acc = None
        idx = [j for j in range(d) if mask >> j & 1]
        for i in range(n):
            prod = columns[idx[0]][i]
            for j in idx[1:]:
                prod = prod * columns[j][i]
            acc = prod if acc is None else acc + prod
        moments[mask] = Fraction(acc, n) if isinstance(acc, int) else acc / n
    return moments


def joint_cumulant_from_samples(columns: list[list[V]]) -> V:
    """Connected statistic of sample columns: block moments then subtraction."""
    moments = block_moments_from_samples(columns)
    return cumulant_from_moments(moments, len(columns))
