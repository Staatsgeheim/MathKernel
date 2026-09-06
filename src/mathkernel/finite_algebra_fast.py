# =============================================================================
# MathKernel - Numba-compiled small-prime GF(p)[x] kernels
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Compiled exact polynomial multiplication modulo a defining polynomial.

The fast path is deliberately narrow: ``p < 2**24`` and degree ``m <= 64``,
so a schoolbook product coefficient sums at most ``64 * (2**24 - 1)**2`` and
therefore fits exactly in uint64 before reduction. Reduction by the monic
defining polynomial also keeps every intermediate below that bound. Anything
outside this fragment returns ``None`` and the caller uses the arbitrary-
precision Python reference implementation.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

MAX_FAST_PRIME = 1 << 24
MAX_FAST_DEGREE = 64

if HAVE_NUMBA:

    @njit(cache=True)
    def _poly_mulmod_u64(a, b, modulus, p):
        m = modulus.shape[0] - 1
        work = np.zeros(2 * m, dtype=np.uint64)
        for i in range(a.shape[0]):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(b.shape[0]):
                bj = b[j]
                if bj:
                    work[i + j] = (work[i + j] + ai * bj) % p
        for k in range(2 * m - 1, m - 1, -1):
            coeff = work[k] % p
            if coeff:
                base = k - m
                for j in range(m):
                    term = (coeff * modulus[j]) % p
                    work[base + j] = (work[base + j] + p - term) % p
                work[k] = 0
        out = np.zeros(m, dtype=np.uint64)
        for i in range(m):
            out[i] = work[i] % p
        return out


def poly_mulmod_fast(
    a: list[int], b: list[int], modulus: list[int], p: int,
) -> list[int] | None:
    """Exact ``a*b mod (modulus, p)`` for the checked small-prime fragment."""
    if not HAVE_NUMBA:
        return None
    m = len(modulus) - 1
    if m < 1 or m > MAX_FAST_DEGREE or p >= MAX_FAST_PRIME:
        return None
    if modulus[-1] != 1:
        return None
    if len(a) > m or len(b) > m:
        return None
    aa = np.zeros(m, dtype=np.uint64)
    bb = np.zeros(m, dtype=np.uint64)
    mod = np.asarray(modulus, dtype=np.uint64)
    aa[:len(a)] = np.asarray(a, dtype=np.uint64)
    bb[:len(b)] = np.asarray(b, dtype=np.uint64)
    out = _poly_mulmod_u64(aa, bb, mod, np.uint64(p))
    result = [int(v) for v in out]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result
