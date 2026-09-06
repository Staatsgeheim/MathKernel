# =============================================================================
# MathKernel - Quantifier elimination and nested-quantifier validity (v0.22)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Richer quantifier handling on top of the v0.20 fragment classifier.

- `negate` / `nnf`: negation normal form over MathIR (De Morgan + relation
  negation + quantifier flipping).
- `strip_existentials`: sat-preserving Skolemization loop for arbitrarily
  nested leading existentials (after negation), with per-level witnesses.
- `quantifier_eliminate`: true QE via Z3's `qe` tactic (LRA/LIA fragments),
  mapped back to MathIR via `from_z3` so results render and re-parse.
- Batch QE over the process pool (QE is single-threaded inside Z3).

Trust: EXACT only when the tactic produces a quantifier-free equivalent on a
decidable fragment; structured UNKNOWN (with fragment classification) else.
"""

from __future__ import annotations

from .models import (BinaryNode, BoolNode, Expr, IntegerNode, NaryNode,
                     QuantifierNode, RationalNode, RelationNode, SetNode,
                     SymbolNode, UnaryNode)
from .parallel import process_map, resolve_workers
from .prove import classify_fragment

_NEG_REL = {"eq": "ne", "ne": "eq", "lt": "ge", "ge": "lt",
            "gt": "le", "le": "gt"}


def negate(ir: Expr) -> Expr:
    """Negation pushed to NNF: relations negate directly, booleans De Morgan,
    quantifiers flip."""
    if isinstance(ir, RelationNode):
        return RelationNode(kind=_NEG_REL[ir.kind], left=ir.left, right=ir.right)
    if isinstance(ir, BoolNode):
        if ir.op == "not":
            return ir.args[0]
        flipped = "or" if ir.op == "and" else "and"
        return BoolNode(op=flipped, args=[negate(a) for a in ir.args])
    if isinstance(ir, QuantifierNode):
        return QuantifierNode(
            quantifier="exists" if ir.quantifier == "forall" else "forall",
            variable=ir.variable, domain=ir.domain, body=negate(ir.body))
    raise ValueError(f"cannot negate {ir.kind} in the logic fragment")


def strip_existentials(z3eng, ir: Expr, domains: dict[str, str]):
    """Sat-preservingly strip leading existential quantifiers, introducing a
    fresh constant per level.     Returns (constraints, remaining_ir, witnesses, env) where witnesses names
    the per-level Skolem constants (outermost first) and env binds them."""
    z3 = z3eng.z3
    constraints: list = []
    witnesses: list[str] = []
    node = ir
    env: dict = {}
    while isinstance(node, QuantifierNode) and node.quantifier == "exists":
        sort_int = isinstance(node.domain, SetNode) and node.domain.name in (
            "naturals", "integers")
        if domains.get(node.variable, "").lower() in ("int", "integer"):
            sort_int = True
        bound = z3.Int(node.variable) if sort_int else z3.Real(node.variable)
        env = {**z3eng._env([node], domains), **env, node.variable: bound}
        if node.domain is not None:
            constraints.append(z3eng._membership(node.domain, bound, env))
        witnesses.append(node.variable)
        node = node.body
    return constraints, node, witnesses, env


_FLIP_REL = {"le": "ge", "ge": "le", "lt": "gt", "gt": "lt", "eq": "eq"}


def _relation(kind: str, left: Expr, right: Expr) -> RelationNode:
    """Normalize Z3's constant-first orientation: keep literals on the right."""
    numeric = (IntegerNode, RationalNode)
    if isinstance(left, numeric) and not isinstance(right, numeric):
        return RelationNode(kind=_FLIP_REL[kind], left=right, right=left)
    return RelationNode(kind=kind, left=left, right=right)


def from_z3(z3mod, ast) -> Expr:
    """Map a quantifier-free Z3 formula back to MathIR (And/Or/Not/relations
    over arithmetic) so results render and re-parse."""
    z3 = z3mod
    if z3.is_and(ast) or z3.is_or(ast):
        op = "and" if z3.is_and(ast) else "or"
        args = [from_z3(z3, a) for a in ast.children()]
        if len(args) == 1:
            return args[0]  # MathIR and/or require at least two operands
        return BoolNode(op=op, args=args)
    if z3.is_not(ast):
        return BoolNode(op="not", args=[from_z3(z3, ast.arg(0))])
    if z3.is_true(ast):
        return RelationNode(kind="eq", left=IntegerNode(value="1"),
                            right=IntegerNode(value="1"))
    if z3.is_false(ast):
        return RelationNode(kind="ne", left=IntegerNode(value="1"),
                            right=IntegerNode(value="1"))
    if z3.is_eq(ast):
        return _relation("eq", from_z3(z3, ast.arg(0)), from_z3(z3, ast.arg(1)))
    if z3.is_le(ast):
        return _relation("le", from_z3(z3, ast.arg(0)), from_z3(z3, ast.arg(1)))
    if z3.is_lt(ast):
        return _relation("lt", from_z3(z3, ast.arg(0)), from_z3(z3, ast.arg(1)))
    if z3.is_ge(ast):
        return _relation("ge", from_z3(z3, ast.arg(0)), from_z3(z3, ast.arg(1)))
    if z3.is_gt(ast):
        return _relation("gt", from_z3(z3, ast.arg(0)), from_z3(z3, ast.arg(1)))
    if z3.is_add(ast):
        return NaryNode(kind="add", args=[from_z3(z3, a) for a in ast.children()])
    if z3.is_mul(ast):
        return NaryNode(kind="mul", args=[from_z3(z3, a) for a in ast.children()])
    if z3.is_sub(ast):
        return NaryNode(kind="add", args=[from_z3(z3, ast.arg(0)),
                                          UnaryNode(arg=from_z3(z3, ast.arg(1)))])
    if z3.is_div(ast):
        return BinaryNode(kind="div", left=from_z3(z3, ast.arg(0)),
                          right=from_z3(z3, ast.arg(1)))
    if ast.decl().kind() == z3.Z3_OP_UMINUS:
        return UnaryNode(arg=from_z3(z3, ast.arg(0)))
    if z3.is_int_value(ast):
        return IntegerNode(value=str(ast.as_long()))
    if z3.is_rational_value(ast):
        num, den = ast.numerator_as_long(), ast.denominator_as_long()
        if den == 1:
            return IntegerNode(value=str(num))
        return RationalNode(numerator=str(num), denominator=str(den))
    if z3.is_const(ast):
        return SymbolNode(name=ast.decl().name())
    raise ValueError(f"from_z3: unsupported Z3 node: {ast.sexpr()[:200]}")


def _count_variables(ir: Expr, fragment: dict) -> int:
    """Free symbols (from the classifier) plus bound quantifier variables.
    Safe on fragments Z3 cannot map (functions etc.)."""
    bound = 0

    def walk(n: Expr) -> None:
        nonlocal bound
        if isinstance(n, QuantifierNode):
            bound += 1
            if n.domain is not None:
                walk(n.domain)
            walk(n.body)
        elif isinstance(n, BoolNode):
            for a in n.args:
                walk(a)

    walk(ir)
    return len(fragment["symbols"]) + bound


def quantifier_eliminate(z3eng, ir: Expr, domains: dict[str, str],
                         max_variables: int = 16) -> dict:
    """True QE via Z3's qe tactic. Returns an equivalent quantifier-free
    MathIR formula (EXACT) or a structured UNKNOWN with the fragment
    classification."""
    if not z3eng.available:
        return {"status": "unknown", "reason": "z3-solver is not installed",
                "fragment": classify_fragment(ir, domains)}
    fragment = classify_fragment(ir, domains)
    nvars = _count_variables(ir, fragment)
    if nvars > max_variables:
        return {"status": "unknown", "fragment": fragment,
                "reason": f"{nvars} variables exceeds max_qe_variables={max_variables}"}
    z3 = z3eng.z3
    try:
        env = z3eng._env([ir], domains)
        goal = z3eng.to_z3(ir, env)
        tactic = z3.Tactic("qe")
        if z3eng.timeout_ms:
            tactic = z3.TryFor(tactic, z3eng.timeout_ms)
        goal_obj = z3.Goal()
        goal_obj.add(goal)
        result = tactic(goal_obj)
    except Exception as exc:
        return {"status": "unknown", "fragment": fragment,
                "reason": f"qe tactic failed: {str(exc)[:300]}"}
    if len(result) != 1:
        return {"status": "unknown", "fragment": fragment,
                "reason": f"qe tactic split into {len(result)} subgoals"}
    sub = result[0]
    try:
        qf = z3.And(*[a for a in sub]) if len(sub) else z3.BoolVal(True)
        if z3.is_quantifier(qf):
            return {"status": "unknown", "fragment": fragment,
                    "reason": "qe result still quantified (undecidable fragment)"}
        formula = from_z3(z3, qf)
    except (ValueError, TypeError) as exc:
        return {"status": "unknown", "fragment": fragment,
                "reason": f"qe result outside the MathIR mapping: {str(exc)[:300]}"}
    return {"status": "eliminated", "fragment": fragment, "formula": formula}


def _qe_job(job: dict) -> dict:
    """Process-pool worker: fresh engine per process (Z3 is single-threaded;
    parallelism is across problems)."""
    from .engines import Z3Engine
    from .parser import parse_math
    from .rendering import render_expr
    ir = parse_math(job["source"])
    eng = Z3Engine(timeout_ms=job.get("timeout_ms"))
    out = quantifier_eliminate(eng, ir, job["domains"],
                               job.get("max_variables", 16))
    if out["status"] == "eliminated":
        out = {**out, "formula": render_expr(out["formula"])}
    else:
        out.pop("formula", None)
    return {"expr_id": job["expr_id"], **out}


def quantifier_eliminate_batch(jobs: list[dict],
                               workers: int | None = None) -> list[dict]:
    return process_map(_qe_job, jobs, workers=resolve_workers(workers))
