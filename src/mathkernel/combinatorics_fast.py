# =============================================================================
# MathKernel - Numba-compiled exact recurrence kernels with bigint fallback
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Compiled exact recurrence extension for the safely bounded int64 fragment.

The kernel uses a conservative ``2**60`` intermediate bound. If any product or
partial sum would cross that bound, it reports overflow and the caller falls
back to Python's arbitrary-precision integers. The compiled path is therefore
exact: it never wraps and never returns an approximate or truncated value.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

_SAFE_BOUND = np.int64(1 << 60)

if HAVE_NUMBA:

    @njit(cache=True)
    def _extend_recurrence_i64(coefficients, initial, n):
        terms = np.empty(n + 1, dtype=np.int64)
        init_len = initial.shape[0]
        for i in range(min(init_len, n + 1)):
            terms[i] = initial[i]
        if n + 1 <= init_len:
            return terms, False
        k = coefficients.shape[0]
        for pos in range(init_len, n + 1):
            acc = np.int64(0)
            for i in range(k):
                c = coefficients[i]
                term = terms[pos - i - 1]
                if c == 0 or term == 0:
                    continue
                if term == -_SAFE_BOUND or c == -_SAFE_BOUND:
                    return terms, True
                limit = _SAFE_BOUND // (c if c >= 0 else -c)
                if term > limit or term < -limit:
                    return terms, True
                product = c * term
                if product > _SAFE_BOUND - acc or product < -_SAFE_BOUND - acc:
                    return terms, True
                acc += product
            terms[pos] = acc
        return terms, False


def extend_recurrence_fast(
    coefficients: list[int], initial: list[int], n: int,
) -> list[int] | None:
    """Exact recurrence terms through index ``n`` within the int64 fragment."""
    if not HAVE_NUMBA or n < 0:
        return None
    if not coefficients or not initial or len(initial) < len(coefficients):
        return None
    bound = 1 << 60
    values = [*coefficients, *initial]
    if any(not isinstance(v, int) or isinstance(v, bool) or abs(v) >= bound
           for v in values):
        return None
    terms, overflow = _extend_recurrence_i64(
        np.asarray(coefficients, dtype=np.int64),
        np.asarray(initial, dtype=np.int64),
        n,
    )
    if overflow:
        return None
    return [int(v) for v in terms]
