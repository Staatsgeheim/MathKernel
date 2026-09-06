# =============================================================================
# MathKernel - Collatz cycle-class sieve via the cycle closure equation
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Collatz cycle-class sieve via the cycle closure equation.

Inspired by spectral-selection: instead of simulating trajectories, derive the
closure condition that any cycle must satisfy and enumerate only its sparse
candidate set.

A cycle with n odd elements and halving counts a = (a_0, ..., a_{n-1}) (each
a_i >= 1, S = sum a_i) obeys

    (2^S - 3^n) * x_0 = C(a),   C(a) = sum_i 3^(n-1-i) * 2^(a_0 + ... + a_{i-1})

Admissible S are pinned exactly: 2^S > 3^n (positivity around the cycle) and
2^S <= 3^n (1 + 1/(3 x_min))^n (every odd element is >= x_min). Both bounds are
evaluated with exact bigint comparisons. Candidates passing C % D == 0 are then
verified by exact simulation of the claimed parity pattern, so any reported
cycle is a genuine Collatz cycle (trust: exact).

With x_min = 1 the sieve is self-contained: a refuted (n, S) class means no
positive-integer cycle with exactly n odd elements exists. With x_min > 1 the
statement is conditional on a published verification floor (e.g. Barina's
2^68), which is recorded as a side condition.
"""
from __future__ import annotations

from .collatz_fast import FAST_N_MAX, HAVE_NUMBA, scan_fast
from .parallel import process_map

HARD_N_MAX = 64
MAX_LEAVES_PER_CLASS = 200_000_000
MIN_LEAVES_PER_JOB = 200_000


def admissible_s_values(n: int, x_min: int) -> list[int]:
    """Exact admissible total-halving counts for an n-odd-element cycle."""
    if n < 1 or x_min < 1:
        raise ValueError("n >= 1 and x_min >= 1 are required")
    p3 = 3 ** n
    lo = p3.bit_length()  # smallest S with 2^S > 3^n (3^n is never a power of 2)
    lhs_factor = (3 * x_min) ** n
    rhs = p3 * (3 * x_min + 1) ** n
    out = []
    s = lo
    while (1 << s) * lhs_factor <= rhs:
        out.append(s)
        s += 1
    return out


def cycle_constant(pattern: tuple[int, ...]) -> int:
    """C(a) = sum_i 3^(n-1-i) 2^(a_0+...+a_{i-1})."""
    n = len(pattern)
    c = 0
    s = 0
    for i in range(n):
        if i:
            s += pattern[i - 1]
        c += 3 ** (n - 1 - i) << s
    return c


def verify_cycle(x0: int, pattern: tuple[int, ...], x_min: int) -> bool:
    """Exact simulation of the claimed parity pattern from candidate x0."""
    if x0 < x_min or x0 % 2 == 0:
        return False
    x = x0
    for a in pattern:
        x = 3 * x + 1
        if x & ((1 << a) - 1):
            return False
        x >>= a
        if x % 2 == 0 or x < x_min:
            return False
    return x == x0


def _compositions(remaining: int, parts: int):
    if parts < 1 or remaining < parts:
        return
    if parts == 1:
        yield (remaining,)
        return
    for first in range(1, remaining - parts + 2):
        for rest in _compositions(remaining - first, parts - 1):
            yield (first,) + rest


def _compositions_capped(remaining: int, parts: int, hi: int):
    """Compositions with every part in [1, hi] (rotation-canonical pruning)."""
    if parts < 1 or remaining < parts or remaining > parts * hi:
        return
    if parts == 1:
        yield (remaining,)
        return
    lo_first = max(1, remaining - (parts - 1) * hi)
    hi_first = min(hi, remaining - (parts - 1))
    for first in range(lo_first, hi_first + 1):
        for rest in _compositions_capped(remaining - first, parts - 1, hi):
            yield (first,) + rest


def _scan_chunk(args: tuple[int, int, int, int, int, bool]) -> dict:
    """Scan compositions of S into n parts with a_0 in [a0_lo, a0_hi). Picklable.

    In canonical mode only patterns with a_0 = max(a_i) are enumerated: every
    cyclic rotation class has such a representative, so cycle-existence
    completeness is preserved while the leaf count drops by a factor ~n.
    """
    n, s_total, a0_lo, a0_hi, x_min, canonical = args
    d = (1 << s_total) - 3 ** n
    if d <= 0:
        return {"checked": 0, "candidates": []}
    checked = 0
    candidates = []

    def consider(pattern: tuple[int, ...]) -> None:
        nonlocal checked
        checked += 1
        c = cycle_constant(pattern)
        if c % d == 0:
            x0 = c // d
            if verify_cycle(x0, pattern, x_min):
                candidates.append({"x0": str(x0), "pattern": list(pattern),
                                   "trivial": x0 == 1})

    if n == 1:
        # The only composition of S into one part is (S,) itself.
        if a0_lo <= s_total < a0_hi:
            consider((s_total,))
        return {"checked": checked, "candidates": candidates}

    for a0 in range(a0_lo, a0_hi):
        remaining = s_total - a0
        if remaining < n - 1:
            break
        rests = (_compositions_capped(remaining, n - 1, a0) if canonical
                 else _compositions(remaining, n - 1))
        for rest in rests:
            consider((a0,) + rest)
    return {"checked": checked, "candidates": candidates}


def _composition_count(s_total: int, n: int) -> int:
    from math import comb
    return comb(s_total - 1, n - 1)


def sieve_cycle_classes(n_max: int, *, x_min: int = 1, workers: int = 1,
                        n_min: int = 1, canonical: bool = True,
                        engine: str = "auto") -> dict:
    """Exhaustively check all cycle classes with n_min <= n <= n_max odd elements.

    With canonical=True (default) only rotation-canonical patterns (a_0 maximal)
    are enumerated; every cycle has such a rotation, so refutation remains
    complete for cycle existence. engine="auto" uses the compiled multithreaded
    kernel when numba is available (n <= 31), else the pure-Python process pool.
    Returns a per-class report; a class with no non-trivial verified candidate is
    refuted exactly.
    """
    if not 1 <= n_min <= n_max <= HARD_N_MAX:
        raise ValueError(f"require 1 <= n_min <= n_max <= {HARD_N_MAX}")
    if x_min < 1:
        raise ValueError("x_min must be >= 1")
    if engine not in ("auto", "numba", "python", "cuda"):
        raise ValueError("engine must be 'auto', 'numba', 'python' or 'cuda'")
    if engine == "numba" and not HAVE_NUMBA:
        raise ValueError("numba engine requested but numba is not installed")

    from .collatz_gpu import GPU_N_MAX, HAVE_CUPY, scan_class_gpu
    if engine == "cuda" and not HAVE_CUPY:
        raise ValueError("cuda engine requested but cupy/CUDA is not available")

    classes = []
    for n in range(n_min, n_max + 1):
        for s_total in admissible_s_values(n, x_min):
            leaves = _composition_count(s_total, n)
            est = leaves // n if canonical and n > 1 else leaves
            if est > MAX_LEAVES_PER_CLASS:
                raise ValueError(
                    f"class (n={n}, S={s_total}) has ~{est} canonical patterns; "
                    f"prototype limit is {MAX_LEAVES_PER_CLASS}")
            classes.append({"n": n, "S": s_total, "patterns": leaves})

    use_cuda = (engine == "cuda"
                or (engine == "auto" and HAVE_CUPY and n_max <= GPU_N_MAX))
    use_fast = (engine == "numba"
                or (engine == "auto" and not use_cuda
                    and HAVE_NUMBA and n_max <= FAST_N_MAX))
    per_class: dict[tuple[int, int], dict] = {}
    used_engine = "python"

    if use_cuda and classes:
        used_engine = "cuda"
        for cls in classes:
            per_class[(cls["n"], cls["S"])] = scan_class_gpu(
                cls["n"], cls["S"], x_min=x_min, canonical=canonical)
    elif use_fast and classes:
        used_engine = "numba"
        if workers > 0:
            from numba import set_num_threads
            set_num_threads(max(1, workers))
        fast = scan_fast([(c["n"], c["S"]) for c in classes], canonical=canonical)
        for cls in classes:
            key = (cls["n"], cls["S"])
            res = fast.get(key, {"checked": 0, "hit_a0s": []})
            candidates = []
            # Hits are vanishingly rare; re-extract their patterns with the
            # exact Python path (including full cycle verification).
            for a0 in res["hit_a0s"]:
                candidates.extend(
                    _scan_chunk((cls["n"], cls["S"], a0, a0 + 1, x_min,
                                 canonical))["candidates"])
            per_class[key] = {"checked": res["checked"], "candidates": candidates}
    else:
        jobs = []
        for cls in classes:
            n, s_total = cls["n"], cls["S"]
            hi_a0 = s_total - (n - 1)
            lo_a0 = -(-s_total // n) if canonical and n > 1 else 1
            if n == 1:
                lo_a0 = hi_a0 = s_total
            if lo_a0 > hi_a0:
                per_class[(n, s_total)] = {"checked": 0, "candidates": []}
                continue
            span = hi_a0 - lo_a0 + 1
            n_chunks = max(1, min(span, workers * 4,
                                  max(1, cls["patterns"] // MIN_LEAVES_PER_JOB)))
            bounds = [lo_a0 + round(i * span / n_chunks) for i in range(n_chunks)] + [hi_a0 + 1]
            for i in range(n_chunks):
                jobs.append((n, s_total, bounds[i], bounds[i + 1], x_min, canonical))

        results = process_map(_scan_chunk, jobs, workers=workers)
        for (n, s_total, *_), res in zip(jobs, results):
            slot = per_class.setdefault((n, s_total), {"checked": 0, "candidates": []})
            slot["checked"] += res["checked"]
            slot["candidates"].extend(res["candidates"])

    class_reports = []
    for cls in classes:
        key = (cls["n"], cls["S"])
        agg = per_class.get(key, {"checked": 0, "candidates": []})
        # Rotations of one cycle share its odd elements; dedupe within the class
        # by the starting element. The trivial cycle (x0=1, all a_i=2) is kept
        # visible per class as a built-in soundness check of the machinery.
        seen_x0 = set()
        cycles = []
        for cand in agg["candidates"]:
            if cand["x0"] in seen_x0:
                continue
            seen_x0.add(cand["x0"])
            cycles.append(cand)
        nontrivial = [c for c in cycles if not c["trivial"]]
        complete = True if canonical else agg["checked"] == cls["patterns"]
        class_reports.append({
            "n": cls["n"], "S": cls["S"],
            "patterns_checked": agg["checked"],
            "patterns_expected": cls["patterns"],
            "rotation_pruned": canonical,
            "complete": complete,
            "cycles": cycles,
            "refuted": not nontrivial and complete,
        })

    return {
        "n_min": n_min, "n_max": n_max, "x_min": str(x_min),
        "self_contained": x_min == 1,
        "workers": workers,
        "engine": used_engine,
        "rotation_pruned": canonical,
        "classes": class_reports,
        "total_patterns_checked": sum(c["patterns_checked"] for c in class_reports),
        "nontrivial_cycles": [c for r in class_reports for c in r["cycles"] if not c["trivial"]],
        "all_classes_refuted": all(c["refuted"] for c in class_reports),
    }
