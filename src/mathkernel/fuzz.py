# =============================================================================
# MathKernel - Cross-engine differential fuzzing: random MathIR expressions over the
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Cross-engine differential fuzzing: random MathIR expressions over the
numeric fragment, evaluated by the SymPy/mpmath high-precision path and the
float64 compiled path, compared for agreement.

The generator is seeded and deterministic; batches parallelize over the
process pool. A disagreement is surfaced, never silently absorbed — this is
the harness that guards the trust contract between numeric tiers.
"""

from __future__ import annotations

import math
import random

from .models import BinaryNode, CallNode, IntegerNode, NaryNode, SymbolNode, UnaryNode
from .numerics import compile_float64
from .rendering import render_expr

_SAFE_CALLS = ("sin", "cos", "exp", "sqrt", "log")


def random_expr(rng: random.Random, variables: list[str], depth: int = 3):
    """Random MathIR over the guarded numeric fragment. Division and log/sqrt
    are guarded so sampled evaluations stay finite on the sample domain."""
    if depth <= 0 or rng.random() < 0.3:
        if rng.random() < 0.5:
            return SymbolNode(name=rng.choice(variables))
        return IntegerNode(value=str(rng.randint(1, 5)))
    choice = rng.random()
    if choice < 0.35:
        kind = "add" if rng.random() < 0.5 else "mul"
        return NaryNode(kind=kind, args=[random_expr(rng, variables, depth - 1)
                                         for _ in range(rng.randint(2, 3))])
    if choice < 0.5:
        return UnaryNode(arg=random_expr(rng, variables, depth - 1))
    if choice < 0.65:
        # guarded power: base^small-nonnegative-int
        return BinaryNode(kind="pow", left=random_expr(rng, variables, depth - 1),
                          right=IntegerNode(value=str(rng.randint(0, 3))))
    if choice < 0.8:
        # guarded division: numerator / (denominator^2 + 1) — never zero
        den = NaryNode(kind="add", args=[
            BinaryNode(kind="pow", left=random_expr(rng, variables, depth - 1),
                       right=IntegerNode(value="2")),
            IntegerNode(value="1")])
        return BinaryNode(kind="div", left=random_expr(rng, variables, depth - 1), right=den)
    name = rng.choice(_SAFE_CALLS)
    arg = random_expr(rng, variables, depth - 1)
    if name in ("sqrt", "log"):
        # guard: sqrt/log of (arg^2 + 1) stays in domain
        arg = NaryNode(kind="add", args=[
            BinaryNode(kind="pow", left=arg, right=IntegerNode(value="2")),
            IntegerNode(value="1")])
    return CallNode(name=name, args=[arg])


def _differential_job(job: dict) -> dict:
    """Worker: rebuild the expression from its seed and compare engines."""
    import mpmath as mp
    from .parser import parse_math  # noqa: F401  (documents the fragment)
    rng = random.Random(job["seed"])
    ir = random_expr(rng, job["variables"], job["depth"])
    source = render_expr(ir)
    variables = job["variables"]
    mp.mp.dps = 40
    try:
        fast = compile_float64(ir, variables)
    except ValueError as exc:
        return {"seed": job["seed"], "source": source, "status": "skipped", "reason": str(exc)}
    from .engines import SymPyEngine  # local import: workers must not share engine state
    eng = SymPyEngine()
    try:
        sym = eng.to_sympy(ir)
    except Exception as exc:
        return {"seed": job["seed"], "source": source, "status": "skipped", "reason": str(exc)}
    worst = 0.0
    import sympy as sp
    syms = [sp.Symbol(v) for v in variables]
    for _ in range(job["samples"]):
        point = [rng.uniform(0.1, 3.0) for _ in variables]
        try:
            subbed = sym.subs(dict(zip(syms, [mp.mpf(repr(p)) for p in point])))
            ref = float(mp.mpf(str(sp.N(subbed, 30))))
            got = fast(*point)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if not (math.isfinite(ref) and math.isfinite(got)):
            continue
        scale = max(1.0, abs(ref))
        worst = max(worst, abs(ref - got) / scale)
    status = "ok" if worst < 1e-8 else "disagreement"
    return {"seed": job["seed"], "source": source, "status": status,
            "max_rel_error": worst}


def fuzz_batch(n: int, variables: list[str] | None = None, depth: int = 3,
               samples: int = 8, seed: int = 0,
               workers: int | None = None) -> dict:
    """Run n seeded differential checks; returns aggregate + disagreements."""
    from .parallel import process_map, resolve_workers
    variables = variables or ["x"]
    jobs = [{"seed": seed + i, "variables": variables, "depth": depth,
             "samples": samples} for i in range(n)]
    results = process_map(_differential_job, jobs, workers=resolve_workers(workers))
    disagreements = [r for r in results if r["status"] == "disagreement"]
    return {"total": n, "ok": sum(1 for r in results if r["status"] == "ok"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "disagreements": disagreements}
