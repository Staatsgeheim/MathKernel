# =============================================================================
# MathKernel - rendering
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from .models import (AlgebraicNumberNode, BinaryNode, BinderNode, BoolNode, CallNode, ComplexNode, Expr, IntegerNode,
                     IntervalNode, MembershipNode, NaryNode, NumberNode, QuantifierNode, RationalNode,
                     RealNode, RelationNode, SetNode, SetOpNode, SymbolNode, UnaryNode)


def render_expr(node: Expr) -> str:
    if isinstance(node, IntegerNode): return node.value
    if isinstance(node, RationalNode): return f"{node.numerator}/{node.denominator}"
    if isinstance(node, RealNode): return node.value
    if isinstance(node, NumberNode): return node.value
    if isinstance(node, AlgebraicNumberNode): return f"rootof({render_expr(node.minimal_polynomial)}, {node.root_index})"
    if isinstance(node, ComplexNode): return f"complex({render_expr(node.real)}, {render_expr(node.imag)})"
    if isinstance(node, IntervalNode):
        l = "[" if node.lower_closed else "("
        r = "]" if node.upper_closed else ")"
        return f"{l}{render_expr(node.lower)}, {render_expr(node.upper)}{r}"
    if isinstance(node, SymbolNode): return node.name
    if isinstance(node, UnaryNode): return f"-({render_expr(node.arg)})"
    if isinstance(node, NaryNode):
        op = " + " if node.kind == "add" else " * "
        return "(" + op.join(render_expr(a) for a in node.args) + ")"
    if isinstance(node, BinaryNode):
        op = "^" if node.kind == "pow" else "/"
        left, right = render_expr(node.left), render_expr(node.right)
        # Literal rationals contain an operator even though they are single IR
        # nodes. Likewise a negative literal/unary base is not a power atom.
        # Without these parentheses, (-2)**x becomes -(2**x), x**(1/3)
        # becomes (x**1)/3, and x/(1/2) becomes (x/1)/2 on reparsing.
        if node.kind == "pow" and (
            isinstance(node.left, (RationalNode, UnaryNode))
            or isinstance(node.left, (IntegerNode, RealNode, NumberNode)) and left.startswith(("-", "+"))
        ):
            left = f"({left})"
        if isinstance(node.right, RationalNode):
            right = f"({right})"
        return f"({left} {op} {right})"
    if isinstance(node, CallNode): return f"{node.name}({', '.join(render_expr(a) for a in node.args)})"
    if isinstance(node, RelationNode):
        op = {"eq":"=", "ne":"!=", "lt":"<", "le":"<=", "gt":">", "ge":">="}[node.kind]
        return f"{render_expr(node.left)} {op} {render_expr(node.right)}"
    if isinstance(node, SetNode):
        if node.name is not None:
            return {"naturals": "Naturals", "integers": "Integers", "rationals": "Rationals",
                    "reals": "Reals", "complexes": "Complexes", "empty": "EmptySet"}[node.name] + "()"
        return "{" + ", ".join(render_expr(e) for e in node.elements or []) + "}"
    if isinstance(node, MembershipNode):
        return f"in({render_expr(node.element)}, {render_expr(node.set)})"
    if isinstance(node, SetOpNode):
        return f"{node.op}({', '.join(render_expr(a) for a in node.args)})"
    if isinstance(node, BoolNode):
        return f"{node.op}({', '.join(render_expr(a) for a in node.args)})"
    if isinstance(node, QuantifierNode):
        domain = f", {render_expr(node.domain)}" if node.domain is not None else ""
        return f"{node.quantifier}({node.variable}{domain}, {render_expr(node.body)})"
    if isinstance(node, BinderNode):
        fn = "sumover" if node.op == "sum" else "productover"
        return f"{fn}({node.variable}, {render_expr(node.domain)}, {render_expr(node.body)})"
    raise TypeError(type(node))
