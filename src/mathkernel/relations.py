# =============================================================================
# MathKernel - Closure-relation search: short exact relations selected by the dynamics
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Closure-relation search: short exact relations selected by the dynamics.

For affine dynamics on Z_M with K-step multiplier A_K, a frequency tuple
(k_0..k_{d-1}) survives uniform averaging iff sum_j k_j A_K^j = 0 (mod M);
for binary-linear dynamics T(s) = Ls, masks survive iff
xor_j (L^{jK})^T w_j = 0. These are the state-side selection rules: the
transition decides which relations can exist before any output is measured.

Searches are bounded (frequency complexity H, Hamming weight H) and use
meet-in-the-middle enumeration, so they find the SHORTEST relations first
without exhaustively scanning the dual space. Irreducibility filtering
removes tuples that are merely unions of lower-order closures.
"""
from __future__ import annotations

from itertools import combinations

from .gf2m import gf2_rows_power

try:
    import numpy as np
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

MAX_ENUMERATION = 4_000_000  # per-side MITM enumeration budget
MAX_RESULTS = 100_000

# The njit fast paths work on int64/uint64: cyclic requires d*H*m < 2^62
# (partial sums can't overflow), binary requires width <= 64. Larger
# instances fall back to the arbitrary-precision Python implementation.
_INT64_SAFE = 1 << 62


def _centered(v: int, m: int) -> int:
    v %= m
    return v - m if v > m // 2 else v


if HAVE_NUMBA:

    @njit(cache=True)
    def _enum_l1(d: int, budget: int, bound: int):
        """All tuples with sum|k_j| <= budget, |k_j| <= bound (two-pass odometer)."""
        count = 0
        ks = np.zeros(d, np.int64)
        used = np.zeros(d, np.int64)
        pos = 0
        ks[0] = -min(bound, budget)
        while pos >= 0:
            hi = min(bound, budget - (used[pos - 1] if pos else 0))
            if ks[pos] > hi:
                pos -= 1
                if pos >= 0:
                    ks[pos] += 1
                continue
            if pos == d - 1:
                count += 1
                ks[pos] += 1
            else:
                used[pos] = (used[pos - 1] if pos else 0) + abs(ks[pos])
                pos += 1
                ks[pos] = -min(bound, budget - used[pos - 1])
        out = np.empty((count, d), np.int64)
        i = 0
        pos = 0
        ks[0] = -min(bound, budget)
        while pos >= 0:
            hi = min(bound, budget - (used[pos - 1] if pos else 0))
            if ks[pos] > hi:
                pos -= 1
                if pos >= 0:
                    ks[pos] += 1
                continue
            if pos == d - 1:
                for j in range(d):
                    out[i, j] = ks[j]
                i += 1
                ks[pos] += 1
            else:
                used[pos] = (used[pos - 1] if pos else 0) + abs(ks[pos])
                pos += 1
                ks[pos] = -min(bound, budget - used[pos - 1])
        return out

    @njit(cache=True)
    def _cyclic_mitm_fast(mults: np.ndarray, m: int, budget: int, bound: int,
                          max_results: int):
        """Meet-in-the-middle short closure search, int64.

        Returns (tuples (n, d) int64, truncated flag). Tuples satisfy
        sum k_j * mults[j] == 0 (mod m), sum|k_j| <= budget, not all zero.
        """
        d = len(mults)
        d1 = d // 2
        left_t = _enum_l1(d1, budget, bound)
        right_t = _enum_l1(d - d1, budget, bound)
        nl, nr = len(left_t), len(right_t)
        ls = np.zeros(nl, np.int64)
        lw = np.zeros(nl, np.int64)
        for i in range(nl):
            s = 0
            w = 0
            for j in range(d1):
                s += left_t[i, j] * mults[j]
                w += abs(left_t[i, j])
            ls[i] = s % m
            lw[i] = w
        rs = np.zeros(nr, np.int64)
        rw = np.zeros(nr, np.int64)
        for i in range(nr):
            s = 0
            w = 0
            for j in range(d - d1):
                s += right_t[i, j] * mults[d1 + j]
                w += abs(right_t[i, j])
            rs[i] = s % m
            rw[i] = w
        order = np.argsort(ls)
        ls_sorted = ls[order]
        out = np.empty((max_results, d), np.int64)
        n_out = 0
        truncated = False
        for i in range(nr):
            comp = (m - rs[i]) % m
            lo = np.searchsorted(ls_sorted, comp)
            hi2 = np.searchsorted(ls_sorted, comp + 1)
            for p in range(lo, hi2):
                li = order[p]
                if lw[li] + rw[i] > budget:
                    continue
                allzero = True
                for j in range(d1):
                    v = left_t[li, j]
                    out[n_out, j] = v
                    if v != 0:
                        allzero = False
                for j in range(d - d1):
                    v = right_t[i, j]
                    out[n_out, d1 + j] = v
                    if v != 0:
                        allzero = False
                if allzero:
                    continue
                n_out += 1
                if n_out == max_results:
                    truncated = True
                    return out[:n_out], truncated
        return out[:n_out], truncated

    @njit(cache=True)
    def _masks_u64(width: int, max_weight: int):
        """All nonzero width-bit masks of Hamming weight <= max_weight (Gosper).

        width must be <= 63 (uint64 headroom for the Gosper step).
        """
        one = np.uint64(1)
        limit = (one << np.uint64(width))
        count = 0
        for w in range(1, max_weight + 1):
            mask = (one << np.uint64(w)) - one
            while mask < limit:
                count += 1
                c = mask & (np.uint64(0) - mask)
                r = mask + c
                mask = (((r ^ mask) >> np.uint64(2)) // c) | r
        out = np.empty(count, np.uint64)
        i = 0
        for w in range(1, max_weight + 1):
            mask = (one << np.uint64(w)) - one
            while mask < limit:
                out[i] = mask
                i += 1
                c = mask & (np.uint64(0) - mask)
                r = mask + c
                mask = (((r ^ mask) >> np.uint64(2)) // c) | r
        return out

    @njit(cache=True)
    def _binary_mitm_fast(powers: np.ndarray, width: int, max_weight: int,
                          max_results: int):
        """MITM binary closure: xor_j (L^{jK})^T w_j = 0, uint64 masks.

        powers[j] holds the rows of L^{jK} (row i = (L^{jK})^T e_i).
        Returns (tuples (n, d) uint64, truncated flag).
        """
        d = powers.shape[0]
        d1 = d // 2
        masks = _masks_u64(width, max_weight)
        nm = len(masks)
        # left side: all d1-tuples of masks -> transported xor
        nl = nm ** d1
        lt = np.empty((nl, d1), np.uint64)
        lv = np.zeros(nl, np.uint64)
        for i in range(nl):
            rem = i
            v = np.uint64(0)
            for j in range(d1 - 1, -1, -1):
                mi = rem % nm
                rem //= nm
                w = masks[mi]
                lt[i, j] = w
                # transport: XOR rows of powers[j] selected by w's bits
                t = np.uint64(0)
                mm = w
                while mm != 0:
                    bit = mm & (np.uint64(0) - mm)
                    b = 0
                    while bit > 1:
                        bit >>= np.uint64(1)
                        b += 1
                    t ^= powers[j, b]
                    mm ^= mm & (np.uint64(0) - mm)
                v ^= t
            lv[i] = v
        order = np.argsort(lv)
        lv_sorted = lv[order]
        d2 = d - d1
        nr = nm ** d2
        out = np.empty((max_results, d), np.uint64)
        n_out = 0
        truncated = False
        for i in range(nr):
            rem = i
            v = np.uint64(0)
            combo = np.empty(d2, np.uint64)
            for j in range(d2 - 1, -1, -1):
                mi = rem % nm
                rem //= nm
                w = masks[mi]
                combo[j] = w
                t = np.uint64(0)
                mm = w
                while mm != 0:
                    bit = mm & (np.uint64(0) - mm)
                    b = 0
                    while bit > 1:
                        bit >>= np.uint64(1)
                        b += 1
                    t ^= powers[d1 + j, b]
                    mm ^= mm & (np.uint64(0) - mm)
                v ^= t
            lo = np.searchsorted(lv_sorted, v)
            hi2 = np.searchsorted(lv_sorted, v + np.uint64(1))
            for p in range(lo, hi2):
                li = order[p]
                for j in range(d1):
                    out[n_out, j] = lt[li, j]
                for j in range(d2):
                    out[n_out, d1 + j] = combo[j]
                n_out += 1
                if n_out == max_results:
                    truncated = True
                    return out[:n_out], truncated
        return out[:n_out], truncated


def _bounded_tuples(d: int, budget: int, bound: int):
    """Integer tuples of length d with sum |k_j| <= budget, |k_j| <= bound."""
    if d == 0:
        yield ()
        return
    lo = max(-bound, -budget)
    hi = min(bound, budget)
    for k in range(lo, hi + 1):
        for rest in _bounded_tuples(d - 1, budget - abs(k), bound):
            yield (k,) + rest


def cyclic_closure(a_ks: list[int], m: int, weight_bound: int,
                   max_results: int = MAX_RESULTS) -> list[tuple[int, ...]]:
    """Short closure tuples sum_j k_j * a_ks[j] == 0 (mod m), sum |k_j| <= H.

    a_ks[j] is the lag-j multiplier (A_K^j for equal spacing, or A^{tau_j}
    for general lag geometry). Tuples are returned with centered (signed)
    representatives; the all-zero tuple is excluded. Meet-in-the-middle:
    enumeration cost is O((2H)^(d/2) / (d/2)!) per side.
    """
    d = len(a_ks)
    if d < 2:
        raise ValueError("closure search needs order d >= 2")
    if m < 2:
        raise ValueError("modulus must be >= 2")
    if weight_bound < 1:
        raise ValueError("weight bound must be >= 1")
    if any(not 0 <= a < m for a in a_ks):
        raise ValueError("multipliers must be reduced mod m")
    d1 = d // 2
    d2 = d - d1
    bound = min(weight_bound, m // 2)
    # upper bound on per-side enumeration: |{k in Z^d : sum|k_j| <= H}|
    # = sum_j 2^j C(d, j) C(H, j); the |k_j| <= bound cap only shrinks it.
    # Reject from this bound alone — never enumerate to discover the count.
    from math import comb
    def _l1_upper(dd: int) -> int:
        return sum(comb(dd, j) * comb(weight_bound, j) << j
                   for j in range(dd + 1))
    n1u, n2u = _l1_upper(d1), _l1_upper(d2)
    if n1u > MAX_ENUMERATION or n2u > MAX_ENUMERATION:
        raise ValueError(
            f"enumeration budget exceeded (up to {n1u}+{n2u} tuples; "
            f"lower the weight bound or order)")
    if HAVE_NUMBA and d * weight_bound * m < _INT64_SAFE:
        tuples, _ = _cyclic_mitm_fast(
            np.asarray(a_ks, dtype=np.int64), m, weight_bound, bound, max_results)
        return [tuple(int(v) for v in row) for row in tuples]

    left: dict[int, list[tuple[tuple[int, ...], int]]] = {}
    for t in _bounded_tuples(d1, weight_bound, bound):
        w = sum(abs(k) for k in t)
        s = sum(k * a for k, a in zip(t, a_ks[:d1])) % m
        left.setdefault(s, []).append((t, w))

    out: list[tuple[int, ...]] = []
    for t2 in _bounded_tuples(d2, weight_bound, bound):
        w2 = sum(abs(k) for k in t2)
        s2 = sum(k * a for k, a in zip(t2, a_ks[d1:])) % m
        for t1, w1 in left.get((-s2) % m, ()):
            if w1 + w2 > weight_bound:
                continue
            full = t1 + t2
            if any(full):
                out.append(full)
                if len(out) >= max_results:
                    return out
    return out


def is_irreducible_cyclic(k_tuple: tuple[int, ...], a_ks: list[int], m: int) -> bool:
    """No nonempty proper subtuple already satisfies the closure equation."""
    d = len(k_tuple)
    for size in range(1, d):
        for subset in combinations(range(d), size):
            if sum(k_tuple[j] * a_ks[j] for j in subset) % m == 0:
                return False
    return True


def cyclic_closure_order(a_k: int, m: int, weight_bound: int, d_min: int = 2,
                         d_max: int = 6) -> dict:
    """Smallest order d with an irreducible equal-spacing closure relation.

    Searches k_0 + k_1 A_K + ... + k_{d-1} A_K^{d-1} == 0 (mod m) with
    sum |k_j| <= H, returning the order and a witness tuple.
    """
    if d_min < 2 or d_max < d_min:
        raise ValueError("need 2 <= d_min <= d_max")
    powers = [1]
    for _ in range(1, d_max):
        powers.append(powers[-1] * a_k % m)
    for d in range(d_min, d_max + 1):
        for t in cyclic_closure(powers[:d], m, weight_bound, max_results=MAX_RESULTS):
            if is_irreducible_cyclic(t, powers[:d], m):
                return {"order": d, "witness": list(t), "found": True}
    return {"order": None, "witness": None, "found": False}


def _masks_by_weight(width: int, max_weight: int):
    for w in range(1, max_weight + 1):
        for bits in combinations(range(width), w):
            mask = 0
            for b in bits:
                mask |= 1 << b
            yield mask


def _transported_rows(rows: list[int], steps: int, width: int) -> list[int]:
    """Rows of L^steps, i.e. columns of (L^steps)^T: applying (L^steps)^T to a
    mask is the XOR of these rows selected by the mask's bits."""
    return gf2_rows_power(rows, steps, width)


def _apply_T(power_rows: list[int], mask: int) -> int:
    out = 0
    m = mask
    while m:
        bit = (m & -m).bit_length() - 1
        out ^= power_rows[bit]
        m &= m - 1
    return out


def binary_closure(rows: list[int], width: int, d: int, k_step: int,
                   max_weight: int, max_results: int = MAX_RESULTS) -> list[tuple[int, ...]]:
    """Bounded-weight mask tuples with xor_j (L^{jK})^T w_j = 0.

    rows are the bit-packed rows of the transition matrix L; masks are
    nonzero with Hamming weight <= max_weight. Meet-in-the-middle over the
    transported masks; enumeration cost is O(C(width, <=H)^(d/2)).
    """
    if d < 2:
        raise ValueError("closure search needs order d >= 2")
    if not 1 <= max_weight <= width:
        raise ValueError("max_weight must be 1..width")
    from math import comb
    n_masks = sum(comb(width, w) for w in range(1, max_weight + 1))
    d1 = d // 2
    d2 = d - d1
    if n_masks ** max(d1, d2) > MAX_ENUMERATION:
        raise ValueError(
            f"enumeration budget exceeded ({n_masks} masks per coordinate, "
            f"d={d}); lower max_weight or d")
    powers = [_transported_rows(rows, j * k_step, width) for j in range(d)]
    if HAVE_NUMBA and width <= 63:
        tuples, _ = _binary_mitm_fast(
            np.asarray(powers, dtype=np.uint64), width, max_weight, max_results)
        return [tuple(int(v) for v in row) for row in tuples]

    left: dict[int, list[tuple[int, ...]]] = {}
    for combo in _mask_tuples(d1, width, max_weight):
        v = 0
        for j, w in enumerate(combo):
            v ^= _apply_T(powers[j], w)
        left.setdefault(v, []).append(combo)

    out: list[tuple[int, ...]] = []
    for combo2 in _mask_tuples(d2, width, max_weight):
        v = 0
        for j, w in enumerate(combo2):
            v ^= _apply_T(powers[d1 + j], w)
        for combo1 in left.get(v, ()):
            out.append(combo1 + combo2)
            if len(out) >= max_results:
                return out
    return out


def _mask_tuples(d: int, width: int, max_weight: int):
    if d == 1:
        yield from ((m,) for m in _masks_by_weight(width, max_weight))
        return
    for first in _masks_by_weight(width, max_weight):
        for rest in _mask_tuples(d - 1, width, max_weight):
            yield (first,) + rest


def is_irreducible_binary(w_tuple: tuple[int, ...], rows: list[int], width: int,
                          k_step: int) -> bool:
    """No nonempty proper subtuple already satisfies the binary closure."""
    d = len(w_tuple)
    if any(w == 0 for w in w_tuple):
        return False
    powers = [_transported_rows(rows, j * k_step, width) for j in range(d)]
    for size in range(1, d):
        for subset in combinations(range(d), size):
            v = 0
            for j in subset:
                v ^= _apply_T(powers[j], w_tuple[j])
            if v == 0:
                return False
    return True


def binary_closure_order(rows: list[int], width: int, k_step: int, max_weight: int,
                         d_min: int = 2, d_max: int = 4) -> dict:
    """Smallest order d with an irreducible bounded-weight binary closure."""
    if d_min < 2 or d_max < d_min:
        raise ValueError("need 2 <= d_min <= d_max")
    for d in range(d_min, d_max + 1):
        for t in binary_closure(rows, width, d, k_step, max_weight):
            if is_irreducible_binary(t, rows, width, k_step):
                return {"order": d, "witness": [f"{w:x}" for w in t], "found": True}
    return {"order": None, "witness": None, "found": False}
