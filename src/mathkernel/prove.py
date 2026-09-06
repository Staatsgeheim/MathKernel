# =============================================================================
# MathKernel - General theorem proving: fragment classification, SMT portfolio
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""General theorem-proving interface beyond the Lean arithmetic fragment.

Pipeline: classify the MathIR fragment -> race a portfolio of Z3 encodings
(thread pool; Z3 releases the GIL) -> optionally emit a Lean certificate for
the supported fragment. Trust discipline: FORMAL only with a checked Lean
certificate, EXACT for a decisive SMT result, UNKNOWN otherwise.
"""

from __future__ import annotations

from .models import (BinaryNode, BoolNode, CallNode, Expr, IntegerNode,
                     NaryNode, NumberNode, QuantifierNode, RationalNode,
                     RealNode, RelationNode, SetNode, SymbolNode, UnaryNode)
from .parallel import resolve_workers, thread_map


def classify_fragment(ir: Expr, domains: dict[str, str] | None = None) -> dict:
    """Classify a statement for engine selection. Returned in results so the
    caller can see exactly why an engine was or was not attempted."""
    domains = domains or {}
    quantified = False
    has_functions = False
    has_reals = False
    symbols: set[str] = set()

    def degree(node: Expr, bound: set[str]) -> int:
        nonlocal quantified, has_functions, has_reals
        if isinstance(node, (IntegerNode,)):
            return 0
        if isinstance(node, (RationalNode, RealNode, NumberNode)):
            has_reals = has_reals or isinstance(node, (RealNode, NumberNode)) \
                or (isinstance(node, NumberNode) and "." in node.value)
            return 0
        if isinstance(node, SymbolNode):
            if node.name not in bound:
                symbols.add(node.name)
            return 1
        if isinstance(node, UnaryNode):
            return degree(node.arg, bound)
        if isinstance(node, NaryNode):
            ds = [degree(a, bound) for a in node.args]
            return max(ds, default=0) if node.kind == "add" else sum(ds)
        if isinstance(node, BinaryNode):
            if node.kind == "div":
                # division by a nonzero literal stays in the arithmetic fragment
                if node.right.kind in ("integer", "rational"):
                    return degree(node.left, bound)
                return 99
            if node.right.kind in ("integer", "number"):
                return degree(node.left, bound) * max(1, int(node.right.value))
            return 99
        if isinstance(node, RelationNode):
            return max(degree(node.left, bound), degree(node.right, bound))
        if isinstance(node, BoolNode):
            return max((degree(a, bound) for a in node.args), default=0)
        if isinstance(node, QuantifierNode):
            quantified = True
            return degree(node.body, bound | {node.variable})
        if isinstance(node, SetNode):
            return 0
        if isinstance(node, CallNode):
            has_functions = True
            return 99
        return 99

    d = degree(ir, set())
    sorts = {domains.get(s, "real").lower() for s in symbols} or {"real"}
    sort = "int" if sorts == {"int"} or sorts == {"integer"} else \
        "rational" if not has_reals else "real"
    return {
        "logic": "quantified" if quantified else "qf",
        "arithmetic": "none" if d == 0 else "linear" if d <= 1 else
                      "nonlinear" if d < 99 else "outside",
        "sort": sort,
        "has_functions": has_functions,
        "symbols": sorted(symbols),
    }


def _smt_attempt(prepared: tuple) -> dict:
    """One portfolio encoding on its own Z3 context. Z3 contexts are not
    thread-safe, so constraints are translated into a per-attempt context on
    the main thread; only solving runs concurrently. `valid_on` is the solver
    answer that proves the goal ("unsat" for refutation-style encodings,
    "sat" for witness-style ones)."""
    kind, z3, ctx, constraints, valid_on, timeout_ms = prepared
    try:
        if kind in ("direct", "skolem"):
            solver = z3.Solver(ctx=ctx)
        elif kind == "qe-light":
            solver = z3.Then("simplify", "solve-eqs", "qe-light", "smt", ctx=ctx).solver()
        elif kind == "lia":
            solver = z3.SolverFor("LIA", ctx=ctx)
        else:
            return {"encoding": kind, "status": "error", "error": "unknown encoding"}
        if timeout_ms:
            solver.set(timeout=timeout_ms)
        for c in constraints:
            solver.add(c)
        answer = solver.check()
        if answer == z3.unsat:
            decisive = "unsat"
        elif answer == z3.sat:
            decisive = "sat"
        else:
            return {"encoding": kind, "status": "unknown"}
        if decisive == valid_on:
            out = {"encoding": kind, "status": "valid"}
            if valid_on == "sat":  # witness-style: surface the model
                model = solver.model()
                out["witness"] = {str(d): str(model[d]) for d in model.decls()} if model else {}
            return out
        out = {"encoding": kind, "status": "not_valid"}
        if answer == z3.sat:
            model = solver.model()
            out["countermodel"] = {str(d): str(model[d]) for d in model.decls()} if model else {}
        return out
    except Exception as exc:
        return {"encoding": kind, "status": "error", "error": str(exc)[:500]}


def portfolio_encodings(fragment: dict, size: int) -> list[str]:
    """Pick the portfolio for a fragment, capped at `size`."""
    enc = []
    if fragment.get("logic") == "quantified":
        enc.append("skolem")  # materializes witnesses/countermodels
    enc += ["direct", "qe-light"]
    if fragment["sort"] == "int" and fragment["arithmetic"] in ("none", "linear"):
        enc.append("lia")
    return enc[:max(1, size)]


def _skolem_constraints(z3eng, ir: Expr, domains: dict[str, str]):
    """Strip the outermost quantifier so witnesses materialize in the model.
    forall(x, D, B): constraints [x ∈ D, ¬B], valid_on=unsat.
    exists(x, D, B): constraints [x ∈ D, B], valid_on=sat."""
    if not isinstance(ir, QuantifierNode):
        return None
    z3 = z3eng.z3
    sort_int = isinstance(ir.domain, SetNode) and ir.domain.name in ("integers", "naturals")
    if domains.get(ir.variable, "").lower() in ("int", "integer"):
        sort_int = True
    bound = z3.Int(ir.variable) if sort_int else z3.Real(ir.variable)
    env = {**z3eng._env([ir], domains), ir.variable: bound}
    constraints = []
    if ir.domain is not None:
        constraints.append(z3eng._membership(ir.domain, bound, env))
    body = z3eng.to_z3(ir.body, env)
    if ir.quantifier == "forall":
        return constraints + [z3.Not(body)], "unsat"
    return constraints + [body], "sat"


def prove_smt(z3eng, ir: Expr, domains: dict[str, str], fragment: dict,
              assumptions: list[Expr] | None = None, portfolio_size: int = 3,
              workers: int | None = None) -> dict:
    """Race the encoding portfolio; first decisive answer wins. Assumptions
    are part of the goal: prove And(assumptions) => ir."""
    if not z3eng.available:
        return {"status": "unavailable", "reason": "z3-solver is not installed"}
    z3 = z3eng.z3
    assumptions = assumptions or []
    encodings = portfolio_encodings(fragment, portfolio_size)
    # Build all constraints on this thread (Z3 contexts are not thread-safe),
    # then translate into a fresh context per encoding; only solving is
    # concurrent.
    env = z3eng._env([ir, *assumptions], domains)
    goal = z3eng.to_z3(ir, env)
    if assumptions:
        goal = z3.Implies(z3.And(*[z3eng.to_z3(a, env) for a in assumptions]), goal)
    prepared = []
    for kind in encodings:
        if kind == "skolem":
            if assumptions:
                continue  # skolemization with assumptions is handled by direct
            built = _skolem_constraints(z3eng, ir, domains)
            if built is None:
                continue
            constraints, valid_on = built
        else:
            constraints, valid_on = [z3.Not(goal)], "unsat"
        ctx = z3.Context()
        prepared.append((kind, z3, ctx, [c.translate(ctx) for c in constraints],
                         valid_on, z3eng.timeout_ms))
    results = thread_map(_smt_attempt, prepared,
                         workers=resolve_workers(workers), min_parallel=2)
    for r in results:  # decisive answers first, in portfolio order
        if r["status"] == "valid":
            return {"status": "valid", "encoding": r["encoding"],
                    "witness": r.get("witness"), "attempts": results}
    for r in results:
        if r["status"] == "not_valid":
            return {"status": "not_valid", "encoding": r["encoding"],
                    "countermodel": r.get("countermodel", {}), "attempts": results}
    return {"status": "unknown", "attempts": results}


def _prove_job(job: dict) -> dict:
    """Process-pool worker: re-parse from source and run the SMT tier with a
    fresh engine (no engine state crosses process boundaries)."""
    from .engines import Z3Engine
    from .parser import parse_math
    ir = parse_math(job["source"])
    eng = Z3Engine()
    fragment = classify_fragment(ir, job["domains"])
    out = prove_smt(eng, ir, job["domains"], fragment)
    return {"expr_id": job["expr_id"], "status": out["status"],
            "countermodel": out.get("countermodel"), "fragment": fragment}
