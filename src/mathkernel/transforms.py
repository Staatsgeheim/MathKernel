# =============================================================================
# MathKernel - Exact integer transforms
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact integer transforms.

The Walsh-Hadamard transform (butterfly ordering, unnormalized) is the workhorse
of Boolean-function and LFSR correlation analysis: for a +/-1 accumulator indexed
by GF(2)^n, fwht(values)[mask] is the exact Walsh coefficient at that mask.
Exactness matters — correlations near the noise floor must not be perturbed by
floating-point rounding.

Fast path: when numba is available and sum(abs(v)) fits comfortably in int64,
the butterfly runs as a compiled int64 kernel. Every output coefficient is a
sum/difference of inputs, so |coefficient| <= sum(abs(v)) and the int64 path
is provably exact; larger inputs fall back to arbitrary-precision Python.
"""
from __future__ import annotations

try:
    import numpy as np
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

_INT64_SAFE_BOUND = 1 << 62  # headroom below int64 max

if HAVE_NUMBA:

    @njit(cache=True)
    def _fwht_i64(v):
        n = len(v)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = v[j]
                    y = v[j + h]
                    v[j] = x + y
                    v[j + h] = x - y
            h *= 2
        return v


def _fwht_python(values: list[int]) -> list[int]:
    out = list(values)
    n = len(out)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = out[j], out[j + h]
                out[j], out[j + h] = x + y, x - y
        h *= 2
    return out


def fwht(values: list[int]) -> list[int]:
    """Unnormalized fast Walsh-Hadamard transform.

    fwht(fwht(v)) == len(v) * v (unnormalized involution).
    """
    n = len(values)
    if n == 0 or n & (n - 1):
        raise ValueError("fwht length must be a power of two")
    if HAVE_NUMBA and sum(abs(v) for v in values) < _INT64_SAFE_BOUND:
        arr = np.array(values, dtype=np.int64)
        return [int(x) for x in _fwht_i64(arr)]
    return _fwht_python(values)
