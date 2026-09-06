# =============================================================================
# MathKernel - Compiled CPU kernel for the Collatz cycle-class sieve
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Compiled CPU kernel for the Collatz cycle-class sieve.

Numba nopython + prange: true multithreading (GIL released), no process spawn.
Exact 128-bit arithmetic via two uint64 limbs: C = sum_i 3^(n-1-i) 2^(s_i) fits
in 128 bits for n <= FAST_N_MAX (C < 3^(n-1) 2^(S+1) with S <= 2n), and
D = 2^S - 3^n < 2^62 so the bitwise 128-by-64 mod never overflows its remainder.

The compiled pass only counts leaves and hits (C % D == 0); hit patterns are
re-extracted by the pure-Python path (hits are vanishingly rare), which keeps
the compiled code free of any output-structure complexity.
"""
from __future__ import annotations

try:
    import numpy as np
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

FAST_N_MAX = 31


if HAVE_NUMBA:

    @njit(cache=True)
    def _shl128(v: np.uint64, s: np.int64) -> tuple[np.uint64, np.uint64]:
        if s == 0:
            return np.uint64(0), v
        if s < 64:
            return v >> np.uint64(64 - s), v << np.uint64(s)
        return v << np.uint64(s - 64), np.uint64(0)

    @njit(cache=True)
    def _mod128_64(c_hi: np.uint64, c_lo: np.uint64, d: np.uint64) -> np.uint64:
        # Bitwise long division; rem < d < 2^62 so rem << 1 never overflows.
        rem = np.uint64(0)
        for bit in range(127, -1, -1):
            if bit >= 64:
                b = (c_hi >> np.uint64(bit - 64)) & np.uint64(1)
            else:
                b = (c_lo >> np.uint64(bit)) & np.uint64(1)
            rem = (rem << np.uint64(1)) | b
            if rem >= d:
                rem -= d
        return rem

    @njit(cache=True)
    def _put(comp: np.ndarray, pos: np.int64, k: np.int64, total: np.int64,
             hi: np.int64) -> None:
        # Left-heavy distribution of `total` over comp[pos..k-1], parts in [1, hi].
        for i in range(pos, k):
            take = total - (k - 1 - i)
            if take > hi:
                take = hi
            comp[i] = take
            total -= take

    @njit(parallel=True, cache=True)
    def _scan_parallel(ns: np.ndarray, ss: np.ndarray, a0s: np.ndarray,
                       caps: np.ndarray, ds: np.ndarray, pow3: np.ndarray,
                       checked: np.ndarray, hits: np.ndarray) -> None:
        for e in prange(len(a0s)):
            n = ns[e]
            a0 = a0s[e]
            cap = caps[e]
            d = ds[e]
            k = n - 1
            r = ss[e] - a0
            count = np.int64(0)
            hit = np.int64(0)
            if k == 0:
                # n == 1: the single pattern (S,); C = 1.
                count = 1
                if d == 1:
                    hit = 1
            elif r >= k and r <= k * cap:
                comp = np.empty(k, np.int64)
                _put(comp, 0, k, r, cap)
                while True:
                    # C = sum_i 3^(n-1-i) 2^(s_i), s_i = a_0 + ... + a_{i-1}
                    c_hi = np.uint64(0)
                    c_lo = pow3[n - 1]
                    s = np.int64(a0)
                    for i in range(1, n):
                        t_hi, t_lo = _shl128(pow3[n - 1 - i], s)
                        lo = c_lo + t_lo
                        c_hi = c_hi + t_hi + (np.uint64(1) if lo < c_lo else np.uint64(0))
                        c_lo = lo
                        if i < n - 1:
                            s += comp[i - 1]  # a_i = comp[i-1] for i >= 1
                    count += 1
                    if _mod128_64(c_hi, c_lo, d) == 0:
                        hit += 1
                    # Odometer step: move one unit from the rightmost
                    # decrementable part into a left-heavy suffix reset.
                    j = k - 1
                    moved = False
                    while j >= 0:
                        if comp[j] > 1:
                            suffix = np.int64(0)
                            for t in range(j + 1, k):
                                suffix += comp[t]
                            if suffix + 1 <= (k - 1 - j) * cap:
                                comp[j] -= 1
                                _put(comp, j + 1, k, suffix + 1, cap)
                                moved = True
                                break
                        j -= 1
                    if not moved:
                        break
            checked[e] = count
            hits[e] = hit


def scan_fast(class_specs: list[tuple[int, int]], *, canonical: bool = True
              ) -> dict[tuple[int, int], dict]:
    """Scan (n, S) classes with the compiled kernel.

    Returns {(n, S): {"checked": int, "hit_a0s": [int, ...]}}. Hit patterns are
    not extracted here; callers rescan those a0 values with the Python path.
    """
    if not HAVE_NUMBA:
        raise RuntimeError("numba is not available")
    ns, ss, a0s, caps = [], [], [], []
    for n, s_total in class_specs:
        if n > FAST_N_MAX:
            raise ValueError(f"fast kernel supports n <= {FAST_N_MAX}")
        hi_a0 = s_total - (n - 1)
        lo_a0 = -(-s_total // n) if canonical and n > 1 else 1
        if n == 1:
            lo_a0 = hi_a0 = s_total
        for a0 in range(lo_a0, hi_a0 + 1):
            ns.append(n)
            ss.append(s_total)
            a0s.append(a0)
            caps.append(a0 if canonical else s_total)
    ns_a = np.array(ns, np.int64)
    ss_a = np.array(ss, np.int64)
    a0_a = np.array(a0s, np.int64)
    caps_a = np.array(caps, np.int64)
    ds_a = np.array([(1 << s) - 3 ** n for n, s in zip(ns, ss)], np.uint64)
    pow3 = np.array([3 ** i for i in range(FAST_N_MAX + 1)], np.uint64)
    checked = np.zeros(len(a0s), np.int64)
    hits = np.zeros(len(a0s), np.int64)
    _scan_parallel(ns_a, ss_a, a0_a, caps_a, ds_a, pow3, checked, hits)

    out: dict[tuple[int, int], dict] = {}
    for i in range(len(a0s)):
        key = (int(ns_a[i]), int(ss_a[i]))
        slot = out.setdefault(key, {"checked": 0, "hit_a0s": []})
        slot["checked"] += int(checked[i])
        if hits[i]:
            slot["hit_a0s"].append(int(a0_a[i]))
    return out
