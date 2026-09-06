# =============================================================================
# MathKernel - Exact Pythagorean leg-pair sweep with a quadratic-residue prefilter
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact Pythagorean leg-pair sweep with a quadratic-residue prefilter.

For every 1 <= a < b <= bound, s = a^2 + b^2 must be a perfect square for
(a, b) to be a leg pair of an Euler brick. Every square is a quadratic
residue mod p, so a candidate failing a QR bitmap test for any battery prime
is rejected with no false negatives; the rare survivors are verified with an
exact integer square root. Residues are maintained incrementally (adds only,
no divisions) inside the scan loops.

Engines: "cuda" (one thread per a), "numba" (prange over a), and "python"
(process-parallel reference). All three return identical pair sets; the
filter is exact, so trust is preserved on every engine.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .parallel import process_map, resolve_workers

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:  # pragma: no cover
    HAVE_CUPY = False

# Odd primes <= 61: QR bitmaps fit in a single uint64 each.
BATTERY_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61)
MAX_BOUND = 2**31 - 1  # s = a^2 + b^2 < 2^63 stays exact in uint64
DEFAULT_MAX_HITS = 1 << 23


def qr_masks(primes: tuple[int, ...] = BATTERY_PRIMES) -> np.ndarray:
    """Bit r of mask[p] is set iff r is a quadratic residue mod p (0 included)."""
    masks = np.zeros(len(primes), np.uint64)
    for j, p in enumerate(primes):
        m = 0
        for x in range(p):
            m |= 1 << ((x * x) % p)
        masks[j] = m
    return masks


# --------------------------------------------------------------------------
# python engine (process-parallel reference)
# --------------------------------------------------------------------------

def scan_leg_pairs(a_lo: int, a_hi: int, bound: int) -> dict[int, tuple[int, ...]]:
    """All (a, b) with a in [a_lo, a_hi), a < b <= bound and a^2 + b^2 square."""
    pairs: dict[int, tuple[int, ...]] = {}
    isqrt = math.isqrt
    for a in range(a_lo, a_hi):
        a2 = a * a
        hits = []
        for b in range(a + 1, bound + 1):
            s = a2 + b * b
            d = isqrt(s)
            if d * d == s:
                hits.append(b)
        if hits:
            pairs[a] = tuple(hits)
    return pairs


def _scan_chunk(args: tuple[int, int, int]) -> dict[int, tuple[int, ...]]:
    return scan_leg_pairs(*args)


def _sweep_python(bound: int, workers: int | None) -> dict[int, tuple[int, ...]]:
    workers = resolve_workers(workers)
    chunks = max(workers * 4, 1)
    edges = [1 + round(i * (bound - 1) / chunks) for i in range(chunks)] + [bound + 1]
    args = [(edges[i], edges[i + 1], bound) for i in range(chunks)]
    pairs: dict[int, tuple[int, ...]] = {}
    for partial in process_map(_scan_chunk, args, workers=workers):
        pairs.update(partial)
    return pairs


# --------------------------------------------------------------------------
# numba engine
# --------------------------------------------------------------------------

if HAVE_NUMBA:

    # Two passes: count hits per a, then write at exact prefix offsets.
    # Each prange iteration owns row a exclusively, so no atomics are needed
    # (numba 0.66 has no CPU atomics) and memory is exactly total-hits sized.
    @njit(parallel=True, cache=True)
    def _sweep_numba(bound, primes, masks, offsets, hit_a, hit_b, hit_n, write):
        np_ = primes.shape[0]
        for a in prange(1, bound):
            a2 = np.uint64(a) * np.uint64(a)
            b0 = a + 1
            s = a2 + np.uint64(b0) * np.uint64(b0)
            r = np.empty(np_, np.uint64)
            db = np.empty(np_, np.uint64)
            for j in range(np_):
                p = primes[j]
                r[j] = s % p
                db[j] = np.uint64(2 * b0 + 1) % p
            cnt = 0
            base = offsets[a]
            for b in range(b0, bound + 1):
                ok = True
                for j in range(np_):
                    if (masks[j] >> r[j]) & 1 == 0:
                        ok = False
                        break
                if ok:
                    d = np.uint64(math.sqrt(np.float64(s)))
                    while (d + 1) * (d + 1) <= s:
                        d += 1
                    while d * d > s:
                        d -= 1
                    if d * d == s:
                        if write:
                            hit_a[base + cnt] = a
                            hit_b[base + cnt] = b
                        cnt += 1
                s += np.uint64(2 * b + 1)
                for j in range(np_):
                    p = primes[j]
                    v = r[j] + db[j]
                    if v >= p:
                        v -= p
                    r[j] = v
                    w = db[j] + 2
                    if w >= p:
                        w -= p
                    db[j] = w
            hit_n[a] = cnt


def _sweep_numba_host(bound: int, max_hits: int) -> tuple[dict[int, tuple[int, ...]], int]:
    primes = np.array(BATTERY_PRIMES, np.uint64)
    masks = qr_masks()
    hit_n = np.zeros(bound, np.int64)
    empty = np.empty(0, np.int64)
    _sweep_numba(bound, primes, masks, empty, empty, empty, hit_n, 0)
    total = int(hit_n.sum())
    if total > max_hits:
        raise RuntimeError(f"leg-pair buffer overflow: {total} hits > {max_hits}; "
                           "raise max_hits")
    offsets = np.zeros(bound, np.int64)
    offsets[1:] = np.cumsum(hit_n)[:-1]
    hit_a = np.empty(total, np.int64)
    hit_b = np.empty(total, np.int64)
    _sweep_numba(bound, primes, masks, offsets, hit_a, hit_b, hit_n, 1)
    pairs = {a: tuple(int(b) for b in hit_b[offsets[a]:offsets[a] + hit_n[a]])
             for a in range(1, bound) if hit_n[a]}
    return pairs, total


# --------------------------------------------------------------------------
# CUDA engine
# --------------------------------------------------------------------------

_KERNEL_SRC = r"""
#define MAXP 17
extern "C" __global__ void leg_sweep(
    const unsigned long long* __restrict__ primes,
    const unsigned long long* __restrict__ masks,
    int np, int bound,
    long long* __restrict__ hit_a, long long* __restrict__ hit_b,
    unsigned long long* __restrict__ counter, unsigned long long max_hits)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (a >= bound) return;
    unsigned long long r[MAXP], db[MAXP];
    unsigned long long a2 = (unsigned long long)a * (unsigned long long)a;
    int b0 = a + 1;
    unsigned long long s = a2 + (unsigned long long)b0 * (unsigned long long)b0;
    for (int j = 0; j < np; ++j) {
        unsigned long long p = primes[j];
        r[j] = s % p;
        db[j] = (2ULL * (unsigned long long)b0 + 1ULL) % p;
    }
    for (int b = b0; b <= bound; ++b) {
        bool ok = true;
        for (int j = 0; j < np; ++j) {
            if (((masks[j] >> r[j]) & 1ULL) == 0ULL) { ok = false; break; }
        }
        if (ok) {
            unsigned long long d = (unsigned long long)sqrt((double)s);
            while ((d + 1ULL) * (d + 1ULL) <= s) ++d;
            while (d * d > s) --d;
            if (d * d == s) {
                unsigned long long idx = atomicAdd(counter, 1ULL);
                if (idx < max_hits) { hit_a[idx] = a; hit_b[idx] = b; }
            }
        }
        s += 2ULL * (unsigned long long)b + 1ULL;
        for (int j = 0; j < np; ++j) {
            unsigned long long p = primes[j];
            unsigned long long v = r[j] + db[j];
            if (v >= p) v -= p;
            r[j] = v;
            unsigned long long w = db[j] + 2ULL;
            if (w >= p) w -= p;
            db[j] = w;
        }
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "leg_sweep")
    return _kernel


def _sweep_cuda(bound: int, max_hits: int) -> tuple[dict[int, tuple[int, ...]], int]:
    primes_dev = cp.asarray(np.array(BATTERY_PRIMES, np.uint64))
    masks_dev = cp.asarray(qr_masks())
    hit_a_dev = cp.full(max_hits, -1, cp.int64)
    hit_b_dev = cp.full(max_hits, -1, cp.int64)
    counter_dev = cp.zeros(1, cp.uint64)
    threads = 256
    blocks = -(-(bound - 1) // threads)
    _get_kernel()((blocks,), (threads,), (
        primes_dev, masks_dev, np.int32(len(BATTERY_PRIMES)), np.int32(bound),
        hit_a_dev, hit_b_dev, counter_dev, np.uint64(max_hits)))
    n = int(counter_dev.get()[0])
    if n > max_hits:
        raise RuntimeError(f"leg-pair buffer overflow: {n} hits > {max_hits}; "
                           "raise max_hits")
    return _group_hits(hit_a_dev.get(), hit_b_dev.get(), n), n


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def _group_hits(hit_a: np.ndarray, hit_b: np.ndarray, n: int) -> dict[int, tuple[int, ...]]:
    order = np.argsort(hit_a[:n], kind="stable")
    grouped: dict[int, list[int]] = {}
    for idx in order:
        grouped.setdefault(int(hit_a[idx]), []).append(int(hit_b[idx]))
    return {a: tuple(sorted(bs)) for a, bs in grouped.items()}


def sweep_leg_pairs(bound: int, *, engine: str = "auto", workers: int | None = None,
                    max_hits: int = DEFAULT_MAX_HITS) -> dict:
    """All Pythagorean leg pairs (a, b) with a < b <= bound, exactly.

    Returns {"pairs": {a: (b, ...)}, "engine", "pair_count",
             "pairs_examined", "seconds"}.
    """
    if engine not in ("auto", "cuda", "numba", "python"):
        raise ValueError("engine must be 'auto', 'cuda', 'numba' or 'python'")
    if bound > MAX_BOUND:
        raise ValueError(f"bound must be <= {MAX_BOUND} for exact uint64 arithmetic")
    if bound < 2:
        return {"pairs": {}, "engine": engine, "pair_count": 0,
                "pairs_examined": 0, "seconds": 0.0}
    if engine == "cuda" and not HAVE_CUPY:
        raise ValueError("cuda engine requested but cupy/CUDA is not available")
    if engine == "numba" and not HAVE_NUMBA:
        raise ValueError("numba engine requested but numba is not installed")
    if engine == "auto":
        engine = "cuda" if HAVE_CUPY else ("numba" if HAVE_NUMBA else "python")

    t0 = time.perf_counter()
    if engine == "cuda":
        pairs, n = _sweep_cuda(bound, max_hits)
    elif engine == "numba":
        pairs, n = _sweep_numba_host(bound, max_hits)
    else:
        pairs = _sweep_python(bound, workers)
        n = sum(len(v) for v in pairs.values())
    return {"pairs": pairs, "engine": engine, "pair_count": n,
            "pairs_examined": bound * (bound - 1) // 2,
            "seconds": time.perf_counter() - t0}
