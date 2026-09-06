# =============================================================================
# MathKernel - CUDA engine for the Collatz cycle-class sieve
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""CUDA engine for the Collatz cycle-class sieve.

Each class (n, S) is one kernel launch. The canonical leaf space (compositions
of S - a_0 into n-1 parts capped at a_0, a_0 maximal) is indexed by a single
uint64 via per-a_0 DP count tables; each thread unranks its first composition
from its global leaf index, then steps a lex-consistent odometer over its
contiguous range. Arithmetic is exact 128-bit via two uint64 limbs with a
bitwise 128-by-64 mod (D < 2^62 for n <= 31). Hits are vanishingly rare and
are written to a small buffer via atomicAdd; pattern extraction and full cycle
verification happen on the host with bigint (exact trust preserved).

Requires cupy (wheel bundles the CUDA runtime; NVRTC compiles the kernel at
first use). Falls back cleanly when no GPU/CUDA is present.
"""
from __future__ import annotations

try:
    import numpy as np
    import cupy as cp
    HAVE_CUPY = True
except ImportError:  # pragma: no cover
    HAVE_CUPY = False

from .collatz import cycle_constant, verify_cycle

GPU_N_MAX = 31
MAX_HITS = 1024

_KERNEL_SRC = r"""
extern "C" __global__ void collatz_scan(
    const long long* __restrict__ prefix,   // [na0+1] leaf-count prefix sums
    const long long* __restrict__ counts,   // [na0][k+1][S+1] DP tables
    const unsigned long long* __restrict__ pow3,  // [n] 3^i
    int n, int k, int S, int lo_a0, int na0, int canon,
    unsigned long long D,
    long long total, long long per_thread,
    int* hit_count, int* hit_patterns, int max_hits,
    unsigned long long* visited)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long g0 = t * per_thread;
    if (g0 >= total) return;
    long long g1 = g0 + per_thread < total ? g0 + per_thread : total;

    // Locate the a0 slot of g0: largest i with prefix[i] <= g0.
    int lo = 0, hi = na0 - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (prefix[mid] <= g0) lo = mid; else hi = mid - 1;
    }
    int ai = lo;
    long long local = g0 - prefix[ai];

    int comp[31];
    long long stride = (long long)(k + 1) * (S + 1);

    // Unrank `local` into comp[] for the current a0 (lex order, parts<=cap).
    // Done once per range and redone on a0 crossings (rare).
    {
        int a0 = lo_a0 + ai;
        int cap = canon ? a0 : S;
        const long long* tab = counts + (long long)ai * stride;
        long long idx = local;
        int rem = S - a0;
        for (int i = 0; i < k; ++i) {
            int parts_left = k - i - 1;
            int v = 1;
            for (; v <= cap; ++v) {
                long long c = (rem - v >= 0) ? tab[parts_left * (S + 1) + rem - v] : 0;
                if (idx < c) break;
                idx -= c;
            }
            comp[i] = v;
            rem -= v;
        }
    }

    unsigned long long done = 0ULL;
    for (long long g = g0; g < g1; ++g) {
        int a0 = lo_a0 + ai;
        int cap = canon ? a0 : S;
        done++;
        // C = sum_i 3^(n-1-i) 2^(s_i), exact 128-bit
        unsigned long long c_hi = 0ULL, c_lo = pow3[n - 1];
        long long s = a0;
        for (int i = 1; i < n; ++i) {
            unsigned long long v = pow3[n - 1 - i];
            unsigned long long t_hi, t_lo;
            if (s == 0) { t_hi = 0ULL; t_lo = v; }
            else if (s < 64) { t_hi = v >> (64 - s); t_lo = v << s; }
            else { t_hi = v << (s - 64); t_lo = 0ULL; }
            unsigned long long nlo = c_lo + t_lo;
            c_hi += t_hi + (nlo < c_lo ? 1ULL : 0ULL);
            c_lo = nlo;
            if (i < n - 1) s += comp[i - 1];
        }
        // 128-by-64 bitwise mod; rem < D < 2^62 so rem<<1 is safe
        unsigned long long rem = 0ULL;
        for (int bit = 127; bit >= 0; --bit) {
            unsigned long long b = (bit >= 64)
                ? ((c_hi >> (bit - 64)) & 1ULL) : ((c_lo >> bit) & 1ULL);
            rem = (rem << 1) | b;
            if (rem >= D) rem -= D;
        }
        if (rem == 0ULL) {
            int slot = atomicAdd(hit_count, 1);
            if (slot < max_hits) {
                int* out = hit_patterns + slot * n;
                out[0] = a0;
                for (int i = 1; i < n; ++i) out[i] = comp[i - 1];
            }
        }
        // Advance: next leaf
        local++;
        long long cnt = prefix[ai + 1] - prefix[ai];
        if (local >= cnt) {
            // Cross into the next a0: unrank index 0 (lex-first composition).
            ai++;
            if (ai >= na0) break;
            local = 0;
            int na0v = lo_a0 + ai;
            int cap2 = canon ? na0v : S;
            const long long* tab = counts + (long long)ai * stride;
            int rem2 = S - na0v;
            for (int i = 0; i < k; ++i) {
                int parts_left = k - i - 1;
                int v = 1;
                for (; v <= cap2; ++v) {
                    long long c = (rem2 - v >= 0) ? tab[parts_left * (S + 1) + rem2 - v] : 0;
                    if (c > 0) break;  // idx == 0: first v with a nonempty branch
                }
                comp[i] = v;
                rem2 -= v;
            }
        } else {
            // Lex odometer step: increment the rightmost incrementable part
            // (cap) whose suffix is above minimum, then right-heavy reset.
            int j = k - 2;
            bool moved = false;
            for (; j >= 0; --j) {
                if (comp[j] < cap) {
                    int suffix = 0;
                    for (int q = j + 1; q < k; ++q) suffix += comp[q];
                    if (suffix > (k - 1 - j)) {
                        comp[j]++;
                        // Right-heavy suffix reset, filled right-to-left so
                        // every part stays within [1, cap].
                        int rem3 = suffix - 1;
                        for (int q = k - 1; q > j; --q) {
                            int take = rem3 - (q - j - 1);
                            if (take > cap) take = cap;
                            comp[q] = take;
                            rem3 -= take;
                        }
                        moved = true;
                        break;
                    }
                }
            }
            if (!moved) break;  // unreachable within a valid range
        }
    }
    atomicAdd(visited, done);
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "collatz_scan")
    return _kernel


def _dp_counts(cap: int, k: int, s_max: int) -> np.ndarray:
    """counts[p][r] = # compositions of r into p parts each in [1, cap]."""
    counts = np.zeros((k + 1, s_max + 1), np.int64)
    counts[0, 0] = 1
    for p in range(1, k + 1):
        cum = np.cumsum(counts[p - 1])
        # counts[p][r] = sum_{v=1..cap} counts[p-1][r-v] = cum[r-1] - cum[r-cap-1]
        r = np.arange(s_max + 1)
        hi = np.clip(r - 1, -1, s_max)
        lo = np.clip(r - cap - 1, -1, s_max)
        counts[p] = np.where(r >= 1, cum[np.clip(hi, 0, s_max)] * (hi >= 0), 0) \
            - np.where(lo >= 0, cum[np.clip(lo, 0, s_max)], 0)
    return counts


def scan_class_gpu(n: int, s_total: int, *, x_min: int = 1,
                   canonical: bool = True) -> dict:
    """Scan one (n, S) class on the GPU. Returns {"checked", "candidates"}."""
    if not HAVE_CUPY:
        raise RuntimeError("cupy is not available")
    if n > GPU_N_MAX:
        raise ValueError(f"GPU engine supports n <= {GPU_N_MAX}")
    d = (1 << s_total) - 3 ** n
    if n == 1:
        candidates = []
        if d == 1:
            x0 = cycle_constant((s_total,)) // d
            if verify_cycle(x0, (s_total,), x_min):
                candidates.append({"x0": str(x0), "pattern": [s_total],
                                   "trivial": x0 == 1})
        return {"checked": 1, "candidates": candidates}

    k = n - 1
    hi_a0 = s_total - k
    lo_a0 = -(-s_total // n) if canonical else 1
    a0_values = list(range(lo_a0, hi_a0 + 1))
    na0 = len(a0_values)

    tables = []
    prefix = [0]
    for a0 in a0_values:
        cap = a0 if canonical else s_total
        tab = _dp_counts(cap, k, s_total)
        tables.append(tab)
        prefix.append(prefix[-1] + int(tab[k, s_total - a0]))
    total = prefix[-1]
    if total == 0:
        return {"checked": 0, "candidates": []}

    counts_dev = cp.asarray(np.stack(tables))
    prefix_dev = cp.asarray(np.array(prefix, np.int64))
    pow3_dev = cp.asarray(np.array([3 ** i for i in range(n)], np.uint64))
    hit_count_dev = cp.zeros(1, np.int32)
    hit_patterns_dev = cp.full(MAX_HITS * n, -1, np.int32)
    visited_dev = cp.zeros(1, np.uint64)

    threads = 256
    per_thread = max(1, -(-total // (threads * 2048)))
    blocks = -(-total // (per_thread * threads))
    _get_kernel()((blocks,), (threads,), (
        prefix_dev, counts_dev, pow3_dev,
        np.int32(n), np.int32(k), np.int32(s_total), np.int32(lo_a0),
        np.int32(na0), np.int32(1 if canonical else 0),
        np.uint64(d), np.int64(total), np.int64(per_thread),
        hit_count_dev, hit_patterns_dev, np.int32(MAX_HITS), visited_dev))

    visited = int(visited_dev.get()[0])
    if visited != total:
        raise RuntimeError(
            f"GPU scan integrity failure: visited {visited} of {total} leaves")
    hits = min(int(hit_count_dev.get()[0]), MAX_HITS)
    candidates = []
    if hits:
        pats = hit_patterns_dev.get().reshape(MAX_HITS, n)[:hits]
        for row in pats:
            pattern = tuple(int(v) for v in row)
            c = cycle_constant(pattern)
            if c % d == 0:
                x0 = c // d
                if verify_cycle(x0, pattern, x_min):
                    candidates.append({"x0": str(x0), "pattern": list(pattern),
                                       "trivial": x0 == 1})
    return {"checked": int(total), "candidates": candidates}
