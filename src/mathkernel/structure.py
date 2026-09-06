# =============================================================================
# MathKernel - structure
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from .bigint import decimal_to_int
from .models import (
    AlgebraicNumberNode, BinaryNode, CallNode, ComplexNode, Expr, IntegerNode,
    IntervalNode, NaryNode, NumberNode, RationalNode, RealNode, RelationNode,
    SymbolNode, UnaryNode,
)
from .rendering import render_expr

# Operation-to-capability mapping follows the design's capability model.
TRIG_FUNCTIONS = {"sin", "cos", "tan"}


def _is_integer_literal(node: Expr) -> bool:
    return isinstance(node, (IntegerNode, NumberNode)) and "." not in node.value


def infer_capabilities(node: Expr) -> set[str]:
    """Collect the algebraic capabilities an expression requires."""
    caps: set[str] = set()

    def walk(n: Expr) -> None:
        if isinstance(n, (IntegerNode, RationalNode, RealNode, NumberNode)):
            if isinstance(n, RationalNode):
                caps.add("Inv")
            return
        if isinstance(n, AlgebraicNumberNode):
            caps.add("Sqrt")  # algebraic roots need radical/RootOf support
            walk(n.minimal_polynomial)
            return
        if isinstance(n, ComplexNode):
            caps.add("Complex")
            walk(n.real); walk(n.imag)
            return
        if isinstance(n, IntervalNode):
            caps.add("Order")
            walk(n.lower); walk(n.upper)
            return
        if isinstance(n, SymbolNode):
            return
        if isinstance(n, UnaryNode):
            caps.add("Neg")
            walk(n.arg)
            return
        if isinstance(n, NaryNode):
            caps.add("Add" if n.kind == "add" else "Mul")
            for a in n.args:
                walk(a)
            return
        if isinstance(n, BinaryNode):
            if n.kind == "div":
                caps.update({"Div", "Inv"})
            elif _is_integer_literal(n.right):
                caps.add("PowInteger")
                try:
                    if decimal_to_int(n.right.value) < 0:  # type: ignore[union-attr]
                        caps.add("Inv")
                except ValueError:
                    caps.add("PowGeneral")
            else:
                caps.add("PowGeneral")
            walk(n.left); walk(n.right)
            return
        if isinstance(n, CallNode):
            if n.name == "sqrt":
                caps.add("Sqrt")
            elif n.name == "exp":
                caps.add("Exp")
            elif n.name == "log":
                caps.add("Log")
            elif n.name in TRIG_FUNCTIONS:
                caps.add("Trig")
            elif n.name == "abs":
                caps.add("Abs")
            for a in n.args:
                walk(a)
            return
        if isinstance(n, RelationNode):
            caps.add("Order" if n.kind in {"lt", "le", "gt", "ge"} else "Eq")
            walk(n.left); walk(n.right)
            return
        raise TypeError(type(n))

    walk(node)
    # Normalization: in this kernel subtraction is add+neg, and division is
    # multiplication by an inverse, so the weaker capabilities imply the
    # stronger structural ones.
    if "Add" in caps:
        caps.add("Neg")
    if {"Div", "Inv"} & caps:
        caps.add("Mul")
    return caps


def suggest_structure(caps: set[str]) -> str:
    """Map a capability set to the weakest algebraic structure that supports it."""
    ring = {"Add", "Mul", "Neg"} <= caps
    field = ring and ({"Inv", "Div"} & caps != set())
    if {"Complex"} & caps and field and "Sqrt" in caps:
        return "complex_field"
    if field and "Sqrt" in caps:
        return "field_with_sqrt"
    if field and "Order" in caps:
        return "ordered_field"
    if field:
        return "field"
    if ring:
        return "commutative_ring"
    if {"Add", "Mul"} <= caps:
        return "semiring"
    if "Add" in caps:
        return "additive_monoid"
    if "Mul" in caps:
        return "multiplicative_monoid"
    return "discrete_set"


def extract_constraints(node: Expr) -> list[str]:
    """Extract side conditions required for the expression to be well-defined.

    Division by a non-constant denominator requires the denominator to be
    nonzero; even roots and logs require non-negative/positive radicands and
    arguments over the reals.
    """
    constraints: list[str] = []

    def add(c: str) -> None:
        if c not in constraints:
            constraints.append(c)

    def is_constant(n: Expr) -> bool:
        return isinstance(n, (IntegerNode, RationalNode, RealNode, NumberNode))

    def walk(n: Expr) -> None:
        if isinstance(n, BinaryNode):
            if n.kind == "div" and not is_constant(n.right):
                add(f"{render_expr(n.right)} != 0")
            if n.kind == "pow":
                if not is_constant(n.right):
                    add(f"{render_expr(n.left)} > 0  -- required for general real exponentiation")
                elif (not is_constant(n.left)
                      and isinstance(n.right, (IntegerNode, NumberNode)) and "." not in n.right.value):
                    try:
                        if decimal_to_int(n.right.value) < 0:
                            add(f"{render_expr(n.left)} != 0")
                    except ValueError:
                        pass
            walk(n.left); walk(n.right)
            return
        if isinstance(n, CallNode):
            if n.name == "sqrt" and not is_constant(n.args[0]):
                add(f"{render_expr(n.args[0])} >= 0  -- required for real sqrt")
            if n.name == "log" and not is_constant(n.args[0]):
                add(f"{render_expr(n.args[0])} > 0  -- required for real log")
            for a in n.args:
                walk(a)
            return
        if isinstance(n, UnaryNode):
            walk(n.arg)
            return
        if isinstance(n, NaryNode):
            for a in n.args:
                walk(a)
            return
        if isinstance(n, RelationNode):
            walk(n.left); walk(n.right)
            return
        if isinstance(n, AlgebraicNumberNode):
            walk(n.minimal_polynomial)
            return
        if isinstance(n, ComplexNode):
            walk(n.real); walk(n.imag)
            return
        if isinstance(n, IntervalNode):
            walk(n.lower); walk(n.upper)
            return

    walk(node)
    return constraints
