# =============================================================================
# MathKernel Sonify - restricted declarative mapping evaluator
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Restricted declarative mapping evaluator. No eval/code execution."""
from __future__ import annotations
import math
from typing import Any

def apply_transform(value: float, spec: dict[str, Any], *, context: dict[str, Any] | None=None) -> float:
    op = spec.get("type", "identity")
    x = float(value); c = context or {}
    if op == "identity": return x
    if op == "linear": return float(spec.get("offset",0.0)) + float(spec.get("scale",1.0))*x
    if op == "abs": return abs(x)
    if op == "log": return math.log(max(x, float(spec.get("floor",1e-300))), float(spec.get("base", math.e)))
    if op == "clamp": return min(float(spec["max"]), max(float(spec["min"]), x))
    if op == "normalize":
        lo=float(spec.get("source_min",c.get("min",0.0))); hi=float(spec.get("source_max",c.get("max",1.0)))
        a=float(spec.get("target_min",0.0)); b=float(spec.get("target_max",1.0))
        return (a+b)/2 if hi == lo else a+(x-lo)*(b-a)/(hi-lo)
    if op == "log_map":
        lo=float(spec.get("source_min",c.get("min",0.0))); hi=float(spec.get("source_max",c.get("max",1.0)))
        a=float(spec["target_min"]); b=float(spec["target_max"])
        u=.5 if hi == lo else min(1,max(0,(x-lo)/(hi-lo)))
        return a*((b/a)**u)
    if op == "harmonic": return float(spec.get("fundamental",110.0))*x
    raise ValueError(f"unsupported sonification transform {op!r}")
