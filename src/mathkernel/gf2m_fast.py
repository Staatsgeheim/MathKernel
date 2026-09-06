# =============================================================================
# MathKernel - Numba-compiled GF(2^m) kernels, generic over the limb count (m <= 1024)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Numba-compiled GF(2^m) kernels, generic over the limb count (m <= 1024).

Elements are stored as little-endian uint64 limb arrays (limb 0 holds bits
0..63). Carry-less multiplication is a bit loop over the m significant bits
of the second operand with a multi-limb shift-and-reduce; reduction xors the
low m bits of the modulus (x^m + red) whenever a bit carries out of position
m-1. All arithmetic is exact — GF(2) arithmetic has no rounding — so the
compiled path returns bit-identical results to the pure-Python reference in
gf2m.py (differential-tested in tests/test_gf2m.py for m in {8, 128, 160,
1024}).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

M64 = (1 << 64) - 1
MAX_DEGREE = 1024

if HAVE_NUMBA:

    @njit(cache=True)
    def _mul_limbs(a, b, red, mask_top, m, out):
        nlimbs = len(a)
        work = np.empty(nlimbs, dtype=np.uint64)
        for l in range(nlimbs):
            out[l] = np.uint64(0)
            work[l] = a[l]
        one = np.uint64(1)
        top_limb = (m - 1) >> 6
        top_bit = (m - 1) & 63
        for i in range(m):
            if (b[i >> 6] >> np.uint64(i & 63)) & one:
                for l in range(nlimbs):
                    out[l] ^= work[l]
            carry = (work[top_limb] >> np.uint64(top_bit)) & one
            for l in range(nlimbs - 1, 0, -1):
                work[l] = ((work[l] << one) | (work[l - 1] >> np.uint64(63)))
            work[0] = work[0] << one
            work[nlimbs - 1] &= mask_top
            if carry:
                for l in range(nlimbs):
                    work[l] ^= red[l]
        return out

    @njit(cache=True)
    def _pow_limbs(a, e, red, mask_top, m, out):
        nlimbs = len(a)
        base = np.empty(nlimbs, dtype=np.uint64)
        acc = np.empty(nlimbs, dtype=np.uint64)
        for l in range(nlimbs):
            base[l] = a[l]
            acc[l] = np.uint64(0)
        acc[0] = np.uint64(1)
        one = np.uint64(1)
        for i in range(m):
            if (e[i >> 6] >> np.uint64(i & 63)) & one:
                _mul_limbs(acc, base, red, mask_top, m, out)
                for l in range(nlimbs):
                    acc[l] = out[l]
            _mul_limbs(base, base, red, mask_top, m, out)
            for l in range(nlimbs):
                base[l] = out[l]
        for l in range(nlimbs):
            out[l] = acc[l]
        return out


def _to_limbs(value: int, nlimbs: int) -> np.ndarray:
    return np.array([(value >> (64 * l)) & M64 for l in range(nlimbs)], dtype=np.uint64)


def _from_limbs(limbs: np.ndarray) -> int:
    out = 0
    for l in range(len(limbs) - 1, -1, -1):
        out = (out << 64) | int(limbs[l])
    return out


class FastOps:
    """Limb-array adapter around the njit kernels."""

    def __init__(self, m: int, red: int):
        self.m = m
        self.nlimbs = (m + 63) // 64
        self._red = _to_limbs(red, self.nlimbs)
        self._mask_top = np.uint64((1 << (m % 64)) - 1) if m % 64 else np.uint64(M64)
        self._out = np.empty(self.nlimbs, dtype=np.uint64)

    def mul(self, a: int, b: int) -> int:
        _mul_limbs(_to_limbs(a, self.nlimbs), _to_limbs(b, self.nlimbs),
                   self._red, self._mask_top, self.m, self._out)
        return _from_limbs(self._out)

    def pow(self, a: int, exponent: int) -> int:
        """a^exponent for 0 <= exponent < 2^m (callers reduce mod 2^m - 1)."""
        _pow_limbs(_to_limbs(a, self.nlimbs), _to_limbs(exponent, self.nlimbs),
                   self._red, self._mask_top, self.m, self._out)
        return _from_limbs(self._out)


def fast_ops(m: int, red: int) -> FastOps | None:
    """Compiled kernels for m <= MAX_DEGREE when numba is available, else None."""
    if not HAVE_NUMBA or m > MAX_DEGREE:
        return None
    return FastOps(m, red)
