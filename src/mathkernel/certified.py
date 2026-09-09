# =============================================================================
# MathKernel - Certified ball arithmetic via python-flint/Arb (optional `certified` extra)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Certified ball arithmetic via python-flint/Arb (optional `certified` extra).

When python-flint is installed this provides an INTERVAL_CERTIFIED engine
alongside mpmath.iv, with arbitrary-precision midpoint-radius balls. Every
function degrades to None when flint is unavailable so callers can fall back
to mpmath.iv.
"""

from __future__ import annotations


def arb_available() -> bool:
    try:
        import flint  # noqa: F401
        return True
    except ImportError:
        return False


def interval_enclosure(fn, lo: str, hi: str, prec: int = 128) -> dict | None:
    """Enclose fn over [lo, hi] using Arb balls via subdivision.

    fn: callable accepting an arb-compatible number (mpmath semantics).
    Returns {"engine": "arb", "lower": str, "upper": str} or None if flint
    is not installed. Subdivision keeps over-estimation small for smooth fn.
    """
    try:
        from flint import arb, arb_series  # noqa: F401
    except ImportError:
        return None
    import mpmath as mp
    lo_a, hi_a = arb(lo), arb(hi)
    best_lo, best_hi = None, None
    parts = 16
    step = (hi_a - lo_a) / parts
    for i in range(parts):
        a, b = lo_a + i * step, lo_a + (i + 1) * step
        ball = arb(0).set_mid_rad((a + b) / 2, (b - a) / 2)
        out = fn(ball)
        l, u = out.lower(), out.upper()
        best_lo = l if best_lo is None or l < best_lo else best_lo
        best_hi = u if best_hi is None or u > best_hi else best_hi
    return {"engine": "arb", "lower": best_lo.str(radius=False),
            "upper": best_hi.str(radius=False)}


def certified_root_check(fn, center: str, width: str = "1e-30") -> dict | None:
    """Rigorous sign-change / containment check of fn near `center`.
    Returns None when flint is unavailable."""
    try:
        from flint import arb
    except ImportError:
        return None
    c, w = arb(center), arb(width)
    ball = arb(0).set_mid_rad(c, w)
    out = fn(ball)
    return {"engine": "arb", "enclosure": out.str(),
            "contains_zero": bool(out.contains(arb(0)))}


def mathir_interval_enclosure(node, variable: str, lower_node, upper_node, *, dps: int = 50) -> dict:
    """Enclose MathIR without point-rounding constants, endpoints or coefficients.

    A private interval context avoids shared precision mutation between jobs.
    Unsupported expressions, unordered/overlapping endpoint enclosures and
    non-finite results fail closed. This is an enclosure, not a PDE theorem.
    """
    from mpmath.ctx_iv import MPIntervalContext
    from .intervals import IntervalEngine
    if not 15 <= dps <= 500:
        raise ValueError("dps must be between 15 and 500")
    iv = MPIntervalContext()
    iv.dps = dps
    engine = IntervalEngine()
    constants = {"pi": iv.pi, "e": iv.exp(iv.mpf(1))}
    low = engine._convert(lower_node, constants, iv)
    high = engine._convert(upper_node, constants, iv)
    def finite_real(value):
        return hasattr(value, "_mpi_") and bool(value.a > iv.ninf) and bool(value.b < iv.inf)
    if not finite_real(low) or not finite_real(high):
        raise ValueError("Certified bounds must be finite")
    # Exact equal syntax may still enclose an irrational number with width.
    # Otherwise require separated endpoint enclosures to establish order.
    equal = lower_node == upper_node
    if not equal and not bool(low.b <= high.a):
        raise ValueError("Bounds are reversed or their order is unresolved")
    domain = iv.mpf([low.a, high.b])
    enclosure = engine._convert(node, {**constants, variable: domain}, iv)
    if not finite_real(enclosure):
        raise ValueError("Non-finite enclosure: domain singularity or unsupported bound")
    return {"engine": "mpmath.iv", "enclosure": str(enclosure),
            "lower": str(enclosure.a), "upper": str(enclosure.b), "dps": dps,
            "domain_enclosure": str(domain), "endpoint_evaluation": "outward_interval_MathIR",
            "strictly_positive": bool(enclosure.a > 0),
            "strictly_negative": bool(enclosure.b < 0), "contains_zero": bool(0 in enclosure)}
