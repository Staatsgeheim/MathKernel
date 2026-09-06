# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Apply declared domains/relations to solution sets without erasing them."""
from __future__ import annotations
import sympy as sp

_DOMAINS = {
    "real": ("real", sp.S.Reals), "reals": ("real", sp.S.Reals),
    "integer": ("integer", sp.S.Integers), "integers": ("integer", sp.S.Integers),
    "int": ("integer", sp.S.Integers),
    "complex": ("complex", sp.S.Complexes), "complexes": ("complex", sp.S.Complexes),
    "natural": ("integer", sp.S.Naturals0), "naturals": ("integer", sp.S.Naturals0),
    "positive_integer": ("integer", sp.S.Naturals),
    "rational": ("real", sp.S.Rationals), "rationals": ("real", sp.S.Rationals),
    "positive": ("real", sp.Interval.open(0, sp.oo)),
    "nonnegative": ("real", sp.Interval(0, sp.oo)),
    "negative": ("real", sp.Interval.open(-sp.oo, 0)),
    "nonpositive": ("real", sp.Interval(-sp.oo, 0)),
    "nonzero": ("complex", sp.Complement(sp.S.Complexes, sp.FiniteSet(0))),
}


def domain_set(domain: str):
    if domain.lower() not in _DOMAINS:
        raise ValueError(f"Unsupported solution domain: {domain}; no fallback to complex is permitted")
    return _DOMAINS[domain.lower()]


def context_condition(engine, ctx, env):
    if ctx is None:
        return sp.S.true
    conditions = [engine.to_sympy(a.expression, env) for a in ctx.assumptions]
    for name, properties in ctx.symbol_properties.items():
        x = env.get(name, sp.Symbol(name))
        for prop in properties:
            factories = {"positive": lambda: x > 0, "negative": lambda: x < 0,
                         "nonnegative": lambda: x >= 0, "nonpositive": lambda: x <= 0,
                         "nonzero": lambda: sp.Ne(x, 0)}
            if prop in factories:
                conditions.append(factories[prop]())
    return sp.And(*conditions)


def constrained_solve(engine, ir, variable: str, domain: str, ctx, env):
    """Return an explicitly constrained set; undecidable cases stay conditional."""
    base_name, restricted_domain = domain_set(domain)
    x = sp.Symbol(variable)
    neutral = dict(env)
    neutral[variable] = x
    # Do not let x's own positivity collapse sqrt(x**2) before solving.
    result = engine.solve(ir, variable, base_name, neutral)
    result = sp.Intersection(result, restricted_domain)
    condition = context_condition(engine, ctx, neutral)
    if condition is sp.S.true:
        return result
    if condition is sp.S.false:
        return sp.S.EmptySet
    try:
        if condition.free_symbols <= {x}:
            return sp.Intersection(result, condition.as_set())
    except (NotImplementedError, ValueError):
        pass
    return sp.ConditionSet(x, condition, result)


def constrain_system_solutions(engine, solutions, variables, ctx, env):
    """Remove refuted candidates and attach unproved predicates to survivors."""
    condition = context_condition(engine, ctx, env)
    out, conditions = [], []
    for solution in solutions:
        predicates = [condition.subs(solution)]
        for variable in variables:
            symbol = env.get(variable, sp.Symbol(variable))
            if symbol in solution:
                domain = (ctx.domains.get(variable, "complex") if ctx else "complex")
                predicates.append(domain_set(domain)[1].contains(solution[symbol]))
        predicate = sp.simplify(sp.And(*predicates))
        if predicate is sp.S.false:
            continue
        out.append(solution)
        conditions.append([] if predicate is sp.S.true else [sp.sstr(predicate)])
    return out, conditions
