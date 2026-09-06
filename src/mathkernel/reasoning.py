# =============================================================================
# MathKernel - reasoning
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import uuid
from .models import (AlgebraicNumberNode, BinaryNode, BinderNode, BoolNode, CallNode, ComplexNode, Expr, IntegerNode,
                     IntervalNode, MathContext, MembershipNode, NaryNode, Obligation, ProblemPlan,
                     QuantifierNode, RelationNode, SetNode, SetOpNode, SymbolNode, UnaryNode)
from .rendering import render_expr


def _walk(node: Expr, symbols: set[str], calls: set[str], kinds: set[str]) -> None:
    kinds.add(node.kind)
    if isinstance(node, SymbolNode):
        symbols.add(node.name)
    elif isinstance(node, AlgebraicNumberNode):
        _walk(node.minimal_polynomial, symbols, calls, kinds)
    elif isinstance(node, ComplexNode):
        _walk(node.real, symbols, calls, kinds); _walk(node.imag, symbols, calls, kinds)
    elif isinstance(node, IntervalNode):
        _walk(node.lower, symbols, calls, kinds); _walk(node.upper, symbols, calls, kinds)
    elif isinstance(node, UnaryNode):
        _walk(node.arg, symbols, calls, kinds)
    elif isinstance(node, NaryNode):
        for a in node.args: _walk(a, symbols, calls, kinds)
    elif isinstance(node, BinaryNode):
        _walk(node.left, symbols, calls, kinds); _walk(node.right, symbols, calls, kinds)
    elif isinstance(node, CallNode):
        calls.add(node.name)
        for a in node.args: _walk(a, symbols, calls, kinds)
    elif isinstance(node, RelationNode):
        _walk(node.left, symbols, calls, kinds); _walk(node.right, symbols, calls, kinds)
    elif isinstance(node, SetNode):
        for e in node.elements or []: _walk(e, symbols, calls, kinds)
    elif isinstance(node, MembershipNode):
        _walk(node.element, symbols, calls, kinds); _walk(node.set, symbols, calls, kinds)
    elif isinstance(node, SetOpNode):
        for a in node.args: _walk(a, symbols, calls, kinds)
    elif isinstance(node, BoolNode):
        for a in node.args: _walk(a, symbols, calls, kinds)
    elif isinstance(node, QuantifierNode):
        if node.domain is not None: _walk(node.domain, symbols, calls, kinds)
        _walk(node.body, symbols, calls, kinds)
        symbols.discard(node.variable)  # bound, not free
    elif isinstance(node, BinderNode):
        _walk(node.domain, symbols, calls, kinds); _walk(node.body, symbols, calls, kinds)
        symbols.discard(node.variable)


def _oid() -> str:
    return "obl_" + uuid.uuid4().hex[:10]


def plan_problem(
    expr: Expr,
    context: MathContext | None,
    engine_manifest: list[dict],
    solve_for: str | None = None,
    capability_manifest: list[dict] | None = None,
) -> ProblemPlan:
    """Build an executable semantic plan.

    The planner deliberately separates *solving a relation* from *proving a theorem*.
    A bare equation such as ``x^2 - 2 = 0`` is treated as a constraint/solve problem,
    so formal/SMT obligations verify generated candidates rather than incorrectly
    attempting to prove the equation universally.
    """
    symbols: set[str] = set(); calls: set[str] = set(); kinds: set[str] = set()
    _walk(expr, symbols, calls, kinds)
    variables = sorted(symbols)
    domains = {s: (context.domains.get(s) if context else None) for s in variables}
    missing = [f"Domain for {s} is unspecified" for s, d in domains.items() if d is None]
    if solve_for and solve_for not in symbols:
        missing.append(f"Solve variable {solve_for} does not occur in the expression")
    solve_variables = [solve_for] if solve_for else variables
    if isinstance(expr, RelationNode) and expr.kind == "eq" and not solve_for and len(variables) > 1:
        missing.append("Solve variable is unspecified; pass solve_for to math_plan/math_reason")

    if not symbols and kinds <= {"integer", "rational", "real", "add", "mul", "neg", "pow", "div"}:
        classification = "exact_arithmetic"
        required = ["evaluate"]
    elif isinstance(expr, RelationNode):
        classification = "transcendental_equation" if calls and expr.kind == "eq" else (
            "algebraic_relation" if not calls else "transcendental_relation"
        )
        required = ["solve"] if expr.kind == "eq" else ["smt", "real_arithmetic"]
    elif calls:
        classification = "transcendental_expression"
        required = ["simplify", "evaluate"]
    else:
        classification = "algebraic_expression"
        required = ["simplify"]

    if symbols and all((domains[s] or "").lower() in {"integer", "int", "integers"} for s in symbols):
        required.append("integer_arithmetic")

    strategy = ["Normalize and preserve the typed MathIR and context."]
    obligations: list[Obligation] = []

    domain_id = _oid()
    if variables:
        obligations.append(Obligation(
            obligation_id=domain_id,
            kind="domain",
            action="validate_context",
            statement="Validate declared domains, assumptions, and context consistency.",
            required_capabilities=[],
            preferred_engines=["mathir", "z3"],
            produces=["validated_context", "assumptions", "side_conditions"],
        ))

    if isinstance(expr, RelationNode) and expr.kind == "eq":
        solve_id = _oid()
        verify_id = _oid()
        smt_id = _oid()
        formal_id = _oid()
        deps = [domain_id] if variables else []
        obligations.append(Obligation(
            obligation_id=solve_id,
            kind="symbolic",
            action="solve_relation",
            statement=f"Solve the equation: {render_expr(expr)}",
            required_capabilities=["solve"],
            preferred_engines=["sympy"],
            depends_on=deps,
            parameters={"variables": solve_variables},
            produces=["candidate_solution_set", "solve_side_conditions"],
        ))
        obligations.append(Obligation(
            obligation_id=verify_id,
            kind="verification",
            action="verify_solution_candidates",
            statement="Substitute every explicit candidate back into the original relation.",
            required_capabilities=["symbolic_equivalence"],
            preferred_engines=["sympy"],
            depends_on=[solve_id],
            parameters={"variables": variables},
            produces=["candidate_soundness"],
        ))
        obligations.append(Obligation(
            obligation_id=smt_id,
            kind="smt",
            action="check_solution_completeness",
            statement="When supported, search for a model satisfying the equation outside the candidate set.",
            required_capabilities=["smt"],
            preferred_engines=["z3"],
            depends_on=[solve_id, verify_id],
            parameters={"variables": variables},
            produces=["completeness_evidence", "counterexample"],
        ))
        obligations.append(Obligation(
            obligation_id=formal_id,
            kind="formal",
            action="formalize_solution_soundness",
            statement="Attempt a Lean certificate for candidate soundness in the supported arithmetic fragment.",
            required_capabilities=["formal_certificate"],
            preferred_engines=["lean"],
            depends_on=[solve_id, verify_id],
            parameters={"variables": variables},
            produces=["formal_certificate"],
        ))
        strategy += [
            "Obtain a symbolic candidate solution set.",
            "Verify every explicit candidate by exact substitution.",
            "Use SMT, when applicable, to look for missed solutions.",
            "Attempt formal certification of candidate soundness when representable.",
        ]
    elif isinstance(expr, RelationNode):
        smt_id = _oid()
        obligations.append(Obligation(
            obligation_id=smt_id,
            kind="smt",
            action="check_relation_satisfiability",
            statement=f"Check satisfiability of the relation: {render_expr(expr)}",
            required_capabilities=["smt"],
            preferred_engines=["z3"],
            depends_on=([domain_id] if variables else []),
            produces=["satisfiability", "model"],
        ))
        strategy += ["Check the relation against the explicit context without assuming it is a theorem."]
    else:
        simplify_id = _oid()
        obligations.append(Obligation(
            obligation_id=simplify_id,
            kind="symbolic",
            action="normalize_expression",
            statement=f"Normalize/evaluate: {render_expr(expr)}",
            required_capabilities=required,
            preferred_engines=["sympy"],
            depends_on=([domain_id] if variables else []),
            produces=["normalized_expression"],
        ))
        strategy.append("Evaluate with an exact symbolic backend before using numerical approximations.")

    semantic_capabilities = capability_manifest or []
    declared_caps = {
        cap for e in engine_manifest for cap in e.get("capabilities", [])
    }
    declared_caps.update(
        str(item.get("operation") or item.get("name"))
        for item in semantic_capabilities
    )
    capability_routes: dict[str, list[str]] = {}
    for obligation in obligations:
        matches = [
            item
            for capability in obligation.required_capabilities
            for item in semantic_capabilities
            if item.get("domain") == "engine"
            and capability in {item.get("name"), item.get("operation")}
        ]
        providers = sorted({
            str(engine)
            for item in matches
            for engine in item.get("engines", [])
        })
        if matches:
            obligation.capability_ref = str(matches[0].get("name"))
            obligation.resolved_handler = matches[0].get("handler")
        if providers:
            preferred = [
                engine for engine in obligation.preferred_engines
                if engine in providers
            ]
            obligation.preferred_engines = [
                *preferred,
                *[engine for engine in providers if engine not in preferred],
            ]
        capability_routes[obligation.obligation_id] = list(
            obligation.preferred_engines)
    formal_possible = "formal_certificate" in declared_caps and not calls and isinstance(expr, RelationNode) and expr.kind == "eq"
    return ProblemPlan(
        classification=classification,
        variables=variables,
        domains=domains,
        missing_assumptions=missing,
        required_capabilities=sorted(set(required)),
        strategy=strategy,
        obligations=obligations,
        capability_routes=capability_routes,
        formal_verification_possible=formal_possible,
    )
