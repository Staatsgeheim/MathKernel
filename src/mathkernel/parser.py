# =============================================================================
# MathKernel - parser
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import math
import re
from lark import Lark, Transformer, UnexpectedInput
from lark.exceptions import VisitError
from .bigint import decimal_to_int
from .models import (AlgebraicNumberNode, BinaryNode, BinderNode, BoolNode, CallNode, ComplexNode, Expr, IntegerNode,
                     IntervalNode, MembershipNode, NaryNode, ParseAmbiguity, QuantifierNode, RationalNode,
                     RealNode, RelationNode, SetNode, SetOpNode, SymbolNode, UnaryNode)

_GRAMMAR = r"""
?start: relation
?relation: sum REL sum -> relation
         | sum
?sum: product
    | sum "+" product -> add
    | sum "-" product -> sub
?product: unary
        | product "*" unary -> mul
        | product "/" unary -> div
?unary: "-" unary -> neg
      | "+" unary
      | power
?power: atom
      | atom "^" unary -> pow
      | atom "**" unary -> pow
?atom: DECIMAL -> real
     | INTEGER -> integer
     | NAME "(" [args] ")" -> call
     | NAME -> symbol
     | "{" [args] "}" -> set_literal
     | "(" relation ")"
?args: relation ("," relation)*
REL: "=" | "!=" | "<=" | ">=" | "<" | ">"
NAME: /[A-Za-z_][A-Za-z0-9_]*/
DECIMAL: /(?:[0-9]+\.[0-9]+|\.[0-9]+)/
INTEGER: /[0-9]+/
%import common.WS_INLINE
%ignore WS_INLINE
"""

_parser = Lark(_GRAMMAR, parser="lalr", start="start")
_ALLOWED_FUNCTIONS = {"sqrt", "sin", "cos", "tan", "exp", "log", "abs", "complex", "interval", "rootof",
                      "factorial", "gamma", "binomial", "pi",
                      "forall", "exists", "in", "union", "intersect", "difference", "complement",
                      "and", "or", "not",
                      "sumover", "productover",
                      "Naturals", "Integers", "Rationals", "Reals", "Complexes", "EmptySet"}

_NAMED_SETS = {"Naturals": "naturals", "Integers": "integers", "Rationals": "rationals",
               "Reals": "reals", "Complexes": "complexes", "EmptySet": "empty"}


def _normalize_rational(a: str, b: str) -> tuple[str, str]:
    if set(b) == {"0"}:
        raise ValueError("Division by zero")
    # Avoid Python's decimal-string conversion limit for gigantic literals.
    # gcd on large decimal strings is intentionally deferred; small values are normalized.
    if len(a) <= 1000 and len(b) <= 1000:
        ai, bi = int(a), int(b)
        g = math.gcd(ai, bi)
        return str(ai // g), str(bi // g)
    return a.lstrip("0") or "0", b.lstrip("0") or "0"


class _ToIR(Transformer):
    def integer(self, xs): return IntegerNode(value=str(xs[0]))
    def real(self, xs):
        value = str(xs[0])
        digits = len(value.replace(".", "").lstrip("0")) or 1
        return RealNode(value=value, precision=digits)
    def symbol(self, xs):
        name = str(xs[0])
        if name in _NAMED_SETS:
            return SetNode(name=_NAMED_SETS[name])
        return SymbolNode(name=name)
    def neg(self, xs): return UnaryNode(arg=xs[0])
    def add(self, xs): return self._nary("add", xs[0], xs[1])
    def sub(self, xs): return self._nary("add", xs[0], UnaryNode(arg=xs[1]))
    def mul(self, xs): return self._nary("mul", xs[0], xs[1])
    def div(self, xs):
        if isinstance(xs[0], IntegerNode) and isinstance(xs[1], IntegerNode):
            p, q = _normalize_rational(xs[0].value, xs[1].value)
            return RationalNode(numerator=p, denominator=q)
        return BinaryNode(kind="div", left=xs[0], right=xs[1])
    def pow(self, xs): return BinaryNode(kind="pow", left=xs[0], right=xs[1])
    def args(self, xs): return list(xs)
    def set_literal(self, xs):
        elements = [] if len(xs) == 0 else (xs[0] if isinstance(xs[0], list) else [xs[0]])
        return SetNode(elements=[e for e in elements if e is not None])
    def call(self, xs):
        name = str(xs[0])
        if name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function: {name}")
        args = [] if len(xs) == 1 else (xs[1] if isinstance(xs[1], list) else [xs[1]])
        args = [a for a in args if a is not None]
        if name in _NAMED_SETS:
            if args: raise ValueError(f"{name}() takes no arguments")
            return SetNode(name=_NAMED_SETS[name])
        if name == "in":
            if len(args) != 2: raise ValueError("in(element, set) expects exactly two arguments")
            return MembershipNode(element=args[0], set=args[1])
        if name in ("union", "intersect"):
            if len(args) < 2: raise ValueError(f"{name} expects at least two sets")
            return SetOpNode(op=name, args=args)
        if name == "difference":
            if len(args) != 2: raise ValueError("difference(A, B) expects exactly two sets")
            return SetOpNode(op="difference", args=args)
        if name == "complement":
            if not 1 <= len(args) <= 2: raise ValueError("complement(A) or complement(A, universe)")
            return SetOpNode(op="complement", args=args)
        if name in ("and", "or"):
            if len(args) < 2: raise ValueError(f"{name}(...) expects at least two propositions")
            return BoolNode(op=name, args=args)
        if name == "not":
            if len(args) != 1: raise ValueError("not(p) expects exactly one proposition")
            return BoolNode(op="not", args=args)
        if name in ("forall", "exists"):
            if len(args) not in (2, 3) or not isinstance(args[0], SymbolNode):
                raise ValueError(f"{name}(x, body) or {name}(x, domain, body); the bound variable must be a symbol")
            domain = args[1] if len(args) == 3 else None
            body = args[-1]
            return QuantifierNode(quantifier=name, variable=args[0].name, domain=domain, body=body)
        if name in ("sumover", "productover"):
            if len(args) != 3 or not isinstance(args[0], SymbolNode):
                raise ValueError(f"{name}(x, set, body) expects a symbol, a finite set, and a body")
            return BinderNode(op="sum" if name == "sumover" else "product",
                              variable=args[0].name, domain=args[1], body=args[2])
        if name == "complex":
            if len(args) != 2: raise ValueError("complex(real, imag) expects exactly two arguments")
            return ComplexNode(real=args[0], imag=args[1])
        if name == "interval":
            if len(args) != 2: raise ValueError("interval(lower, upper) expects exactly two arguments")
            return IntervalNode(lower=args[0], upper=args[1])
        if name == "rootof":
            if len(args) != 2 or not isinstance(args[1], IntegerNode):
                raise ValueError("rootof(polynomial, root_index) expects an integer root index")
            root_index = decimal_to_int(args[1].value)
            if root_index < 0 or root_index > 1_000_000:
                raise ValueError("rootof root_index must be between 0 and 1,000,000")
            return AlgebraicNumberNode(minimal_polynomial=args[0], root_index=root_index)
        return CallNode(name=name, args=args)
    def relation(self, xs):
        table = {"=":"eq", "!=":"ne", "<":"lt", "<=":"le", ">":"gt", ">=":"ge"}
        return RelationNode(kind=table[str(xs[1])], left=xs[0], right=xs[2])
    @staticmethod
    def _nary(kind, a, b):
        args = []
        for x in (a, b):
            if isinstance(x, NaryNode) and x.kind == kind: args.extend(x.args)
            else: args.append(x)
        return NaryNode(kind=kind, args=args)


def ambiguity_diagnostics(text: str) -> list[ParseAmbiguity]:
    out: list[ParseAmbiguity] = []
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*([A-Za-z_][A-Za-z0-9_]*)\b", text):
        a, b, x = m.groups()
        out.append(ParseAmbiguity(span=m.group(0), reason="division followed by implicit multiplication",
            candidates=[f"({a}/{b})*{x}", f"{a}/({b}*{x})"], severity="error"))
    for m in re.finditer(r"\b(sin|cos|tan|log|exp|sqrt|forall|exists|sumover|productover)\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*\^\s*\d+)?)", text):
        fn, arg = m.groups()
        out.append(ParseAmbiguity(span=m.group(0), reason="function application without parentheses",
            candidates=[f"{fn}({arg.replace(' ', '')})"], severity="error"))
    for m in re.finditer(r"\b([A-Za-z])([A-Za-z])\b", text):
        token = m.group(0)
        if token not in _ALLOWED_FUNCTIONS:
            out.append(ParseAmbiguity(span=token, reason="adjacent letters can denote one symbol or a product",
                candidates=[token, f"{token[0]}*{token[1]}"], severity="warning"))
    return out


def parse_math(text: str) -> Expr:
    if len(text) > 100_000:
        raise ValueError("Input too long (maximum 100,000 characters)")
    try:
        tree = _parser.parse(text)
        return _ToIR().transform(tree)
    except UnexpectedInput as exc:
        raise ValueError(f"Parse error near position {exc.pos_in_stream}") from exc
    except VisitError as exc:
        if isinstance(exc.orig_exc, ValueError):
            raise exc.orig_exc from exc
        raise ValueError(f"Invalid mathematical expression: {exc.orig_exc}") from exc
