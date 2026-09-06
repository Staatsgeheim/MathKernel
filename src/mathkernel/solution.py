# =============================================================================
# MathKernel - solution
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import sympy as sp
from .models import SolutionSet


def _bound(v) -> str:
    return sp.sstr(v)


def serialize_solution_set(sol, domain: str) -> SolutionSet:
    if sol == sp.S.EmptySet:
        return SolutionSet(kind="empty", domain=domain)
    universal = {"real": sp.S.Reals, "integer": sp.S.Integers, "complex": sp.S.Complexes}.get(domain)
    if universal is not None and sol == universal:
        return SolutionSet(kind="universal", domain=domain, expression=sp.sstr(sol))
    if isinstance(sol, sp.FiniteSet):
        return SolutionSet(kind="finite", domain=domain, values=[sp.sstr(x) for x in sorted(sol, key=sp.default_sort_key)])
    if isinstance(sol, sp.Interval):
        return SolutionSet(kind="interval", domain=domain, intervals=[{
            "lower": _bound(sol.start), "upper": _bound(sol.end),
            "lower_closed": not sol.left_open, "upper_closed": not sol.right_open,
        }])
    if isinstance(sol, sp.Union):
        return SolutionSet(kind="union", domain=domain,
            parts=[serialize_solution_set(part, domain) for part in sol.args])
    if isinstance(sol, sp.ConditionSet):
        return SolutionSet(kind="conditional", domain=domain, parameter=sp.sstr(sol.sym),
            condition=sp.sstr(sol.condition), expression=sp.sstr(sol.base_set))
    if isinstance(sol, sp.ImageSet):
        return SolutionSet(kind="parametric", domain=domain, expression=sp.sstr(sol))
    if isinstance(sol, sp.Set):
        return SolutionSet(kind="symbolic", domain=domain, expression=sp.sstr(sol))
    return SolutionSet(kind="unknown", domain=domain, expression=sp.sstr(sol))
