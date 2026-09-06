# =============================================================================
# MathKernel - codegen
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
from .structure import extract_constraints, infer_capabilities, suggest_structure


class CodegenError(ValueError):
    """Raised when an expression is outside the supported codegen fragment."""


MAX_TS_SAFE_INT = 9_007_199_254_740_991  # 2**53 - 1
MAX_RUST_I64 = 9_223_372_036_854_775_807

_STRUCTURE_NAMES = {
    "complex_field": "ComplexField",
    "field_with_sqrt": "FieldWithSqrt",
    "ordered_field": "OrderedField",
    "field": "Field",
    "commutative_ring": "CommutativeRing",
    "semiring": "Semiring",
    "additive_monoid": "AdditiveMonoid",
    "multiplicative_monoid": "MultiplicativeMonoid",
    "discrete_set": "RequiredOps",
}

_REL_METHODS = {
    "eq": "equals", "ne": "not_equals", "lt": "less_than",
    "le": "less_eq", "gt": "greater_than", "ge": "greater_eq",
}


def _int_value(node: Expr) -> int:
    if isinstance(node, (IntegerNode, NumberNode)) and "." not in node.value:
        return decimal_to_int(node.value)
    raise CodegenError("Expected an integer literal")


def _check_literal_range(n: int, language: str) -> None:
    if language == "typescript" and abs(n) > MAX_TS_SAFE_INT:
        raise CodegenError(
            f"Integer literal {n} exceeds the safe range for TypeScript numbers; "
            "use Python or Rust codegen for exact big-integer literals."
        )
    if language == "rust" and abs(n) > MAX_RUST_I64:
        raise CodegenError(f"Integer literal {n} exceeds the Rust i64 range used by from_number.")


def collect_capabilities(bodies: list[Expr]) -> set[str]:
    caps: set[str] = set()
    for body in bodies:
        caps |= infer_capabilities(body)
    if any(_has_literal(b) for b in bodies):
        caps.add("FromNumber")
    return caps


def _has_literal(node: Expr) -> bool:
    found = False

    def walk(n: Expr) -> None:
        nonlocal found
        if isinstance(n, (IntegerNode, RationalNode, RealNode, NumberNode)):
            found = True
            return
        for child in _children(n):
            walk(child)

    walk(node)
    return found


def _children(n: Expr) -> list[Expr]:
    if isinstance(n, UnaryNode):
        return [n.arg]
    if isinstance(n, NaryNode):
        return list(n.args)
    if isinstance(n, BinaryNode):
        return [n.left, n.right]
    if isinstance(n, CallNode):
        return list(n.args)
    if isinstance(n, RelationNode):
        return [n.left, n.right]
    if isinstance(n, AlgebraicNumberNode):
        return [n.minimal_polynomial]
    if isinstance(n, ComplexNode):
        return [n.real, n.imag]
    if isinstance(n, IntervalNode):
        return [n.lower, n.upper]
    return []


_CODEGEN_CALLS = {"sqrt", "sin", "cos", "tan", "exp", "log", "abs"}


def _reject_unsupported(node: Expr) -> None:
    if isinstance(node, RealNode) or (isinstance(node, NumberNode) and "." in node.value):
        raise CodegenError(
            "Decimal literals cannot be emitted in generic code; "
            "rewrite them as exact rationals (e.g. 125/100)."
        )
    if isinstance(node, (ComplexNode, IntervalNode, AlgebraicNumberNode)):
        raise CodegenError(f"Codegen does not yet support {node.kind} nodes.")
    if isinstance(node, CallNode) and node.name not in _CODEGEN_CALLS:
        raise CodegenError(
            f"Generic codegen does not support '{node.name}'; supported functions: "
            f"{sorted(_CODEGEN_CALLS)}. factorial/gamma/binomial are not field operations "
            "and pi is not definable in a generic algebraic structure."
        )
    for child in _children(node):
        _reject_unsupported(child)


# ---------------------------------------------------------------------------
# TypeScript emitter
# ---------------------------------------------------------------------------

_TS_INTERFACES = {
    "Add": "export interface Add<T> {\n  add(a: T, b: T): T;\n}",
    "Mul": "export interface Mul<T> {\n  mul(a: T, b: T): T;\n}",
    "Neg": "export interface Neg<T> {\n  neg(a: T): T;\n}",
    "Inv": "export interface Inv<T> {\n  inv(a: T): T;\n}",
    "PowInteger": "export interface PowInteger<T> {\n  powInt(a: T, n: number): T;\n}",
    "PowGeneral": "export interface PowGeneral<T> {\n  pow(a: T, b: T): T;\n}",
    "Sqrt": "export interface Sqrt<T> {\n  sqrt(a: T): T;\n}",
    "Exp": "export interface Exp<T> {\n  exp(a: T): T;\n}",
    "Log": "export interface Log<T> {\n  log(a: T): T;\n}",
    "Trig": "export interface Trig<T> {\n  sin(a: T): T;\n  cos(a: T): T;\n  tan(a: T): T;\n}",
    "Abs": "export interface Abs<T> {\n  abs(a: T): T;\n}",
    "Eq": "export interface Eq<T> {\n  equals(a: T, b: T): boolean;\n  not_equals(a: T, b: T): boolean;\n}",
    "Order": ("export interface Order<T> {\n  less_than(a: T, b: T): boolean;\n"
              "  less_eq(a: T, b: T): boolean;\n  greater_than(a: T, b: T): boolean;\n"
              "  greater_eq(a: T, b: T): boolean;\n}"),
    "FromNumber": "export interface FromNumber<T> {\n  fromNumber(n: number): T;\n}",
}


class TypeScriptEmitter:
    language = "typescript"
    extension = "ts"

    def emit_expr(self, node: Expr) -> str:
        _reject_unsupported(node)
        return self._e(node)

    def _e(self, n: Expr) -> str:
        if isinstance(n, IntegerNode) or isinstance(n, NumberNode):
            value = _int_value(n)
            _check_literal_range(value, self.language)
            return f"F.fromNumber({value})"
        if isinstance(n, RationalNode):
            p, q = decimal_to_int(n.numerator), decimal_to_int(n.denominator)
            _check_literal_range(p, self.language)
            _check_literal_range(q, self.language)
            return f"F.mul(F.fromNumber({p}), F.inv(F.fromNumber({q})))"
        if isinstance(n, SymbolNode):
            return n.name
        if isinstance(n, UnaryNode):
            return f"F.neg({self._e(n.arg)})"
        if isinstance(n, NaryNode):
            op = "add" if n.kind == "add" else "mul"
            out = self._e(n.args[0])
            for a in n.args[1:]:
                out = f"F.{op}({out}, {self._e(a)})"
            return out
        if isinstance(n, BinaryNode):
            if n.kind == "div":
                return f"F.mul({self._e(n.left)}, F.inv({self._e(n.right)}))"
            try:
                exponent = _int_value(n.right)
            except CodegenError:
                return f"F.pow({self._e(n.left)}, {self._e(n.right)})"
            if exponent == 0:
                return "F.fromNumber(1)"
            if exponent == 1:
                return self._e(n.left)
            if exponent == -1:
                return f"F.inv({self._e(n.left)})"
            if exponent >= 0:
                return f"F.powInt({self._e(n.left)}, {exponent})"
            return f"F.inv(F.powInt({self._e(n.left)}, {-exponent}))"
        if isinstance(n, CallNode):
            args = ", ".join(self._e(a) for a in n.args)
            return f"F.{n.name}({args})"
        if isinstance(n, RelationNode):
            return f"F.{_REL_METHODS[n.kind]}({self._e(n.left)}, {self._e(n.right)})"
        raise CodegenError(f"Unsupported node for TypeScript codegen: {n.kind}")

    def emit_file(self, *, function_name: str, params: list[str], bodies: list[Expr],
                  caps: set[str], structure: str, source: str, assumptions: list[str],
                  returns_relation: bool) -> str:
        ordered = [c for c in _TS_INTERFACES if c in caps]
        parts = [_TS_INTERFACES[c] for c in ordered]
        combined = _STRUCTURE_NAMES.get(structure, "RequiredOps")
        extends = ", ".join(f"{c}<T>" for c in ordered) or "object"
        parts.append(f"export interface {combined}<T> extends {extends} {{}}")
        header = (f" * Generated from:\n *   {source}\n *\n"
                  + (f" * Assumptions:\n" + "".join(f" *   {a}\n" for a in assumptions) if assumptions else ""))
        ret = "boolean" if returns_relation else ("T" if len(bodies) == 1 else "T[]")
        sig_params = "".join(f"  {p}: T,\n" for p in params)
        body_lines = []
        if len(bodies) == 1:
            body_lines.append(f"  return {self.emit_expr(bodies[0])};")
        else:
            body_lines.append("  return [")
            body_lines += [f"    {self.emit_expr(b)}," for b in bodies]
            body_lines.append("  ];")
        parts.append(
            f"/**\n{header} */\n"
            f"export function {function_name}<T>(\n{sig_params}  F: {combined}<T>\n): {ret} {{\n"
            + "\n".join(body_lines) + "\n}"
        )
        return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Python emitter
# ---------------------------------------------------------------------------

_PY_PROTOCOLS = {
    "Add": "class Add(Protocol[T]):\n    def add(self, a: T, b: T) -> T: ...",
    "Mul": "class Mul(Protocol[T]):\n    def mul(self, a: T, b: T) -> T: ...",
    "Neg": "class Neg(Protocol[T]):\n    def neg(self, a: T) -> T: ...",
    "Inv": "class Inv(Protocol[T]):\n    def inv(self, a: T) -> T: ...",
    "PowInteger": "class PowInteger(Protocol[T]):\n    def pow_int(self, a: T, n: int) -> T: ...",
    "PowGeneral": "class PowGeneral(Protocol[T]):\n    def pow(self, a: T, b: T) -> T: ...",
    "Sqrt": "class Sqrt(Protocol[T]):\n    def sqrt(self, a: T) -> T: ...",
    "Exp": "class Exp(Protocol[T]):\n    def exp(self, a: T) -> T: ...",
    "Log": "class Log(Protocol[T]):\n    def log(self, a: T) -> T: ...",
    "Trig": ("class Trig(Protocol[T]):\n    def sin(self, a: T) -> T: ...\n"
             "    def cos(self, a: T) -> T: ...\n    def tan(self, a: T) -> T: ..."),
    "Abs": "class Abs(Protocol[T]):\n    def abs(self, a: T) -> T: ...",
    "Eq": ("class Eq(Protocol[T]):\n    def equals(self, a: T, b: T) -> bool: ...\n"
           "    def not_equals(self, a: T, b: T) -> bool: ..."),
    "Order": ("class Order(Protocol[T]):\n    def less_than(self, a: T, b: T) -> bool: ...\n"
              "    def less_eq(self, a: T, b: T) -> bool: ...\n"
              "    def greater_than(self, a: T, b: T) -> bool: ...\n"
              "    def greater_eq(self, a: T, b: T) -> bool: ..."),
    "FromNumber": "class FromInt(Protocol[T]):\n    def from_int(self, n: int) -> T: ...",
}


class PythonEmitter:
    language = "python"
    extension = "py"

    def emit_expr(self, node: Expr) -> str:
        _reject_unsupported(node)
        return self._e(node)

    def _e(self, n: Expr) -> str:
        if isinstance(n, IntegerNode) or isinstance(n, NumberNode):
            return f"F.from_int({_int_value(n)})"
        if isinstance(n, RationalNode):
            p, q = decimal_to_int(n.numerator), decimal_to_int(n.denominator)
            return f"F.mul(F.from_int({p}), F.inv(F.from_int({q})))"
        if isinstance(n, SymbolNode):
            return n.name
        if isinstance(n, UnaryNode):
            return f"F.neg({self._e(n.arg)})"
        if isinstance(n, NaryNode):
            op = "add" if n.kind == "add" else "mul"
            out = self._e(n.args[0])
            for a in n.args[1:]:
                out = f"F.{op}({out}, {self._e(a)})"
            return out
        if isinstance(n, BinaryNode):
            if n.kind == "div":
                return f"F.mul({self._e(n.left)}, F.inv({self._e(n.right)}))"
            try:
                exponent = _int_value(n.right)
            except CodegenError:
                return f"F.pow({self._e(n.left)}, {self._e(n.right)})"
            if exponent == 0:
                return "F.from_int(1)"
            if exponent == 1:
                return self._e(n.left)
            if exponent == -1:
                return f"F.inv({self._e(n.left)})"
            if exponent >= 0:
                return f"F.pow_int({self._e(n.left)}, {exponent})"
            return f"F.inv(F.pow_int({self._e(n.left)}, {-exponent}))"
        if isinstance(n, CallNode):
            args = ", ".join(self._e(a) for a in n.args)
            return f"F.{n.name}({args})"
        if isinstance(n, RelationNode):
            return f"F.{_REL_METHODS[n.kind]}({self._e(n.left)}, {self._e(n.right)})"
        raise CodegenError(f"Unsupported node for Python codegen: {n.kind}")

    def emit_file(self, *, function_name: str, params: list[str], bodies: list[Expr],
                  caps: set[str], structure: str, source: str, assumptions: list[str],
                  returns_relation: bool) -> str:
        ordered = [c for c in _PY_PROTOCOLS if c in caps]
        parts = ["from typing import Protocol, TypeVar", "", 'T = TypeVar("T")', ""]
        parts += [_PY_PROTOCOLS[c] for c in ordered]
        combined = _STRUCTURE_NAMES.get(structure, "RequiredOps")
        bases = ", ".join(f"{c if c != 'FromNumber' else 'FromInt'}[T]" for c in ordered)
        parts.append(f"class {combined}({bases + ', ' if bases else ''}Protocol[T]):\n    pass")
        doc = f'    """Generated from: {source}' + \
              ("".join(f"\n    Assumption: {a}" for a in assumptions) if assumptions else "") + '\n    """'
        sig = ", ".join([*params, f"F: {combined}[T]"])
        ret = "bool" if returns_relation else ("T" if len(bodies) == 1 else "tuple[T, ...]")
        if len(bodies) == 1:
            body = f"    return {self.emit_expr(bodies[0])}"
        else:
            inner = ",\n        ".join(self.emit_expr(b) for b in bodies)
            body = f"    return (\n        {inner},\n    )"
        parts.append(f"def {function_name}({sig}) -> {ret}:\n{doc}\n{body}")
        return "\n\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Rust emitter
# ---------------------------------------------------------------------------

_RS_TRAITS = {
    "Add": "pub trait Add {\n    fn add(self, rhs: Self) -> Self;\n}",
    "Mul": "pub trait Mul {\n    fn mul(self, rhs: Self) -> Self;\n}",
    "Neg": "pub trait Neg {\n    fn neg(self) -> Self;\n}",
    "Inv": "pub trait Inv {\n    fn inv(self) -> Self;\n}",
    "PowInteger": "pub trait PowInteger {\n    fn pow_int(self, n: u32) -> Self;\n}",
    "PowGeneral": "pub trait PowGeneral {\n    fn pow(self, rhs: Self) -> Self;\n}",
    "Sqrt": "pub trait Sqrt {\n    fn sqrt(self) -> Self;\n}",
    "Exp": "pub trait Exp {\n    fn exp(self) -> Self;\n}",
    "Log": "pub trait Log {\n    fn log(self) -> Self;\n}",
    "Trig": ("pub trait Trig {\n    fn sin(self) -> Self;\n    fn cos(self) -> Self;\n"
             "    fn tan(self) -> Self;\n}"),
    "Abs": "pub trait Abs {\n    fn abs(self) -> Self;\n}",
    "Eq": ("pub trait EqOp {\n    fn equals(self, rhs: Self) -> bool;\n"
           "    fn not_equals(self, rhs: Self) -> bool;\n}"),
    "Order": ("pub trait OrderOp {\n    fn less_than(self, rhs: Self) -> bool;\n"
              "    fn less_eq(self, rhs: Self) -> bool;\n    fn greater_than(self, rhs: Self) -> bool;\n"
              "    fn greater_eq(self, rhs: Self) -> bool;\n}"),
    "FromNumber": "pub trait FromNumber {\n    fn from_number(n: i64) -> Self;\n}",
}

_RS_REL = {
    "eq": "equals", "ne": "not_equals", "lt": "less_than",
    "le": "less_eq", "gt": "greater_than", "ge": "greater_eq",
}


class RustEmitter:
    language = "rust"
    extension = "rs"

    def emit_expr(self, node: Expr) -> str:
        _reject_unsupported(node)
        return self._e(node)

    def _e(self, n: Expr) -> str:
        if isinstance(n, IntegerNode) or isinstance(n, NumberNode):
            value = _int_value(n)
            _check_literal_range(value, self.language)
            return f"T::from_number({value})"
        if isinstance(n, RationalNode):
            p, q = decimal_to_int(n.numerator), decimal_to_int(n.denominator)
            _check_literal_range(p, self.language)
            _check_literal_range(q, self.language)
            return f"T::from_number({p}).mul(T::from_number({q}).inv())"
        if isinstance(n, SymbolNode):
            return f"{n.name}.clone()"
        if isinstance(n, UnaryNode):
            return f"{self._e(n.arg)}.neg()"
        if isinstance(n, NaryNode):
            op = "add" if n.kind == "add" else "mul"
            out = self._e(n.args[0])
            for a in n.args[1:]:
                out = f"{out}.{op}({self._e(a)})"
            return out
        if isinstance(n, BinaryNode):
            if n.kind == "div":
                return f"{self._e(n.left)}.mul({self._e(n.right)}.inv())"
            try:
                exponent = _int_value(n.right)
            except CodegenError:
                return f"{self._e(n.left)}.pow({self._e(n.right)})"
            if exponent == 0:
                return "T::from_number(1)"
            if exponent == 1:
                return self._e(n.left)
            if exponent == -1:
                return f"{self._e(n.left)}.inv()"
            if exponent < 0:
                return f"{self._e(n.left)}.pow_int({-exponent}).inv()"
            if exponent > 4_294_967_295:
                raise CodegenError("Exponent exceeds the u32 range used by pow_int.")
            return f"{self._e(n.left)}.pow_int({exponent})"
        if isinstance(n, CallNode):
            if len(n.args) != 1:
                raise CodegenError(
                    f"Rust codegen supports only single-argument calls, not {n.name}/{len(n.args)}")
            out = self._e(n.args[0])
            return f"{out}.{n.name}()"
        if isinstance(n, RelationNode):
            return f"{self._e(n.left)}.{_RS_REL[n.kind]}({self._e(n.right)})"
        raise CodegenError(f"Unsupported node for Rust codegen: {n.kind}")

    def emit_file(self, *, function_name: str, params: list[str], bodies: list[Expr],
                  caps: set[str], structure: str, source: str, assumptions: list[str],
                  returns_relation: bool) -> str:
        ordered = [c for c in _RS_TRAITS if c in caps]
        parts = [_RS_TRAITS[c] for c in ordered]
        combined = _STRUCTURE_NAMES.get(structure, "RequiredOps")
        bounds = " + ".join([*ordered, "Clone"])
        parts.append(f"pub trait {combined}: {bounds} {{}}\n")
        parts.append(f"impl<T> {combined} for T where T: {bounds} {{}}")
        comment = f"// Generated from: {source}\n" + "".join(f"// Assumption: {a}\n" for a in assumptions)
        sig_params = ", ".join(f"{p}: T" for p in params)
        ret = "bool" if returns_relation else ("T" if len(bodies) == 1 else "Vec<T>")
        if len(bodies) == 1:
            body = f"    {self.emit_expr(bodies[0])}"
        else:
            inner = ",\n        ".join(self.emit_expr(b) for b in bodies)
            body = f"    vec![\n        {inner},\n    ]"
        parts.append(
            f"{comment}pub fn {function_name}<T: {combined}>({sig_params}) -> {ret} {{\n{body}\n}}"
        )
        return "\n\n".join(parts) + "\n"


EMITTERS = {
    "typescript": TypeScriptEmitter,
    "python": PythonEmitter,
    "rust": RustEmitter,
}


def build_artifact(*, emitter, function_name: str, params: list[str], bodies: list[Expr],
                   source: str, assumptions: list[str], target: str,
                   returns_relation: bool, extra_constraints: list[str] | None = None) -> dict:
    caps = collect_capabilities(bodies)
    structure = suggest_structure(caps)
    constraints: list[str] = []
    for b in bodies:
        for c in extract_constraints(b):
            if c not in constraints:
                constraints.append(c)
    for c in extra_constraints or []:
        if c not in constraints:
            constraints.append(c)
    content = emitter.emit_file(
        function_name=function_name, params=params, bodies=bodies, caps=caps,
        structure=structure, source=source, assumptions=assumptions,
        returns_relation=returns_relation,
    )
    return {
        "language": emitter.language,
        "function_name": function_name,
        "params": list(params),
        "files": [{"path": f"{function_name}.{emitter.extension}", "content": content}],
        "required_capabilities": sorted(caps),
        "suggested_structure": structure,
        "constraints": constraints,
        "assumptions": list(assumptions),
        "target": target,
    }
