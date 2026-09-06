# =============================================================================
# MathKernel Viz - generic, domain-agnostic data transforms
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Deterministic data-reshaping utilities for composing visualizations.

These functions reshape already-computed data (delay embeddings, binning,
decimation, normalisation).  They are presentation-layer plumbing: they never
claim mathematical evidence, carry the input's trust through unchanged, and
are deliberately free of any application-domain knowledge.
"""
from __future__ import annotations

import math
from typing import Sequence


def lag_embedding(sequence: Sequence[float], lags: Sequence[int] = (0, 1, 2),
                  *, stride: int = 1, offset: int = 0,
                  differences: bool = False) -> list[list[float]]:
    """Delay-coordinate embedding of a 1D sequence.

    Returns rows ``(u[n+l0], u[n+l1], ...)`` for ``n = offset, offset+stride,
    ...``.  With ``differences=True`` each coordinate is the first difference
    ``u[n+lag+1] - u[n+lag]``.  Generic dynamical-systems reshaping; works on
    any numeric sequence.
    """
    vals = [float(v) for v in sequence]
    lags = [int(k) for k in lags]
    if not lags:
        raise ValueError("lags must be non-empty")
    if any(k < 0 for k in lags):
        raise ValueError("lags must be >= 0")
    stride = max(1, int(stride))
    extra = 1 if differences else 0
    count = len(vals) - max(lags) - extra
    if count <= 0:
        return []
    out = []
    for n in range(int(offset), count, stride):
        if differences:
            out.append([vals[n + k + 1] - vals[n + k] for k in lags])
        else:
            out.append([vals[n + k] for k in lags])
    return out


def normalize(values: Sequence[float]) -> list[float]:
    """Affine map to [-0.5, 0.5] around the data midpoint (span-safe)."""
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-300)
    center = (lo + hi) / 2
    return [(v - center) / span for v in vals]


def downsample(values: Sequence, limit: int = 22000) -> list:
    """Deterministic strided decimation to at most ``limit`` points."""
    vals = list(values)
    if len(vals) <= limit:
        return vals
    step = max(1, math.floor(len(vals) / max(1, int(limit))))
    return vals[::step]


def histogram_counts(values: Sequence[float], bins: int = 64,
                     value_range: tuple[float, float] | None = None
                     ) -> tuple[list[float], list[int]]:
    """Bin values into ``bins`` buckets; returns (left_edges, counts)."""
    vals = [float(v) for v in values]
    bins = max(1, int(bins))
    if not vals:
        return [], []
    lo, hi = (value_range if value_range else (min(vals), max(vals)))
    lo, hi = float(lo), float(hi)
    if lo == hi:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        if v < lo or v > hi:
            continue
        i = min(bins - 1, int((v - lo) / width))
        counts[i] += 1
    edges = [lo + i * width for i in range(bins)]
    return edges, counts


def linear_form_mod1(columns: Sequence[Sequence[float]],
                     weights: Sequence[float]) -> list[float]:
    """``(w0*c0[n] + w1*c1[n] + ...) mod 1`` row-wise, mapped into [0, 1).

    Generic weighted-phase reshaping of parallel columns; no domain semantics.
    """
    cols = [[float(v) for v in c] for c in columns]
    w = [float(x) for x in weights]
    if len(cols) != len(w):
        raise ValueError("columns and weights must have the same length")
    if not cols:
        return []
    n = min(len(c) for c in cols)
    out = []
    for i in range(n):
        phase = sum(wj * cj[i] for wj, cj in zip(w, cols)) % 1.0
        out.append(phase)
    return out


def sequence_stats(values: Sequence[float]) -> dict:
    """Basic descriptive statistics of a 1D sequence (count/mean/std/min/max)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"count": 0}
    mean = math.fsum(vals) / n
    var = max(0.0, math.fsum((v - mean) ** 2 for v in vals) / n)
    return {"count": n, "mean": mean, "std": math.sqrt(var),
            "min": min(vals), "max": max(vals),
            "distinct": len(set(vals))}
