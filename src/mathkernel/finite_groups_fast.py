# =============================================================================
# MathKernel - Numba-compiled Cayley-table validation for finite groups
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Compiled exact validation of finite-group Cayley tables.

Group elements are indices ``0..n-1``, so every table entry fits int64 and the
compiled checks are exact. The validator returns structured failure codes; the
caller raises the same human-readable ``ValueError`` messages as the Python
reference path. If numba is unavailable the caller uses the Python path.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(cache=True)
    def _validate_cayley(table, identity):
        n = table.shape[0]
        inverses = np.empty(n, dtype=np.int64)
        for a in range(n):
            for b in range(n):
                value = table[a, b]
                if value < 0 or value >= n:
                    return 1, a, b, 0, inverses
        for g in range(n):
            if table[identity, g] != g or table[g, identity] != g:
                return 2, g, 0, 0, inverses
        seen = np.zeros(n, dtype=np.uint8)
        for a in range(n):
            for i in range(n):
                seen[i] = 0
            for b in range(n):
                value = table[a, b]
                if seen[value]:
                    return 3, a, 0, 0, inverses
                seen[value] = 1
        for b in range(n):
            for i in range(n):
                seen[i] = 0
            for a in range(n):
                value = table[a, b]
                if seen[value]:
                    return 4, b, 0, 0, inverses
                seen[value] = 1
        for a in range(n):
            for b in range(n):
                ab = table[a, b]
                for c in range(n):
                    if table[ab, c] != table[a, table[b, c]]:
                        return 5, a, b, c, inverses
        for g in range(n):
            found = -1
            for h in range(n):
                if table[g, h] == identity and table[h, g] == identity:
                    found = h
                    break
            if found < 0:
                return 6, g, 0, 0, inverses
            inverses[g] = found
        return 0, 0, 0, 0, inverses


def validate_cayley_fast(
    table: list[list[int]], identity: int,
) -> tuple[int, int, int, int, list[int]] | None:
    """Validate a square Cayley table; ``None`` when numba is unavailable."""
    if not HAVE_NUMBA:
        return None
    n = len(table)
    if n == 0 or any(len(row) != n for row in table):
        return None
    array = np.asarray(table, dtype=np.int64)
    code, a, b, c, inverses = _validate_cayley(array, identity)
    return int(code), int(a), int(b), int(c), [int(v) for v in inverses]
