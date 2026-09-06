# =============================================================================
# MathKernel - Dimensional analysis: exact 7-vector SI dimensions, unit registry,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Dimensional analysis: exact 7-vector SI dimensions, unit registry,
dimensional checking of MathIR expressions, exact rational unit conversion.

Dimensions are exact Fraction exponent vectors over the SI base dimensions
(L, M, T, I, Theta, N, J). Conversions are exact rational arithmetic.
Dimensional inconsistencies are hard errors. sympy.units is intentionally
not used at runtime; the registry below is self-contained and exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .models import (BinaryNode, CallNode, Expr, IntegerNode, NaryNode,
                     NumberNode, RationalNode, RealNode, SymbolNode, UnaryNode)

_NUMERIC_KINDS = ("integer", "rational", "real", "number")

_BASE_NAMES = ("L", "M", "T", "I", "Th", "N", "J")


@dataclass(frozen=True)
class Dimension:
    """Exact SI dimension vector: (length, mass, time, current, temperature,
    amount of substance, luminous intensity)."""

    exponents: tuple[Fraction, ...]

    def __post_init__(self):
        if len(self.exponents) != 7 or not all(isinstance(e, Fraction) for e in self.exponents):
            raise ValueError("dimension must be a 7-tuple of Fractions")

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, n: Fraction) -> "Dimension":
        return Dimension(tuple(a * n for a in self.exponents))

    def is_dimensionless(self) -> bool:
        return all(e == 0 for e in self.exponents)

    def __str__(self) -> str:
        if self.is_dimensionless():
            return "1"
        parts = []
        for name, e in zip(_BASE_NAMES, self.exponents):
            if e == 0:
                continue
            parts.append(name if e == 1 else f"{name}^{e}")
        return "*".join(parts)


DIMENSIONLESS = Dimension(tuple(Fraction(0) for _ in range(7)))


def _dim(**kw: int) -> Dimension:
    v = {n: Fraction(0) for n in _BASE_NAMES}
    for k, val in kw.items():
        v[k] = Fraction(val)
    return Dimension(tuple(v[n] for n in _BASE_NAMES))


@dataclass(frozen=True)
class Unit:
    name: str
    dimension: Dimension
    scale: Fraction  # exact factor to the SI base-unit product


def _u(name: str, dim: Dimension, scale) -> tuple[str, Unit]:
    return name, Unit(name, dim, Fraction(scale))


# SI base + common derived/prefixed units. Scales are exact by definition
# (2019 SI): minute/hour/day, litre, and metric prefixes are exact rationals.
REGISTRY: dict[str, Unit] = dict([
    _u("m", _dim(L=1), 1), _u("kg", _dim(M=1), 1), _u("s", _dim(T=1), 1),
    _u("A", _dim(I=1), 1), _u("K", _dim(Th=1), 1), _u("mol", _dim(N=1), 1),
    _u("cd", _dim(J=1), 1),
    _u("g", _dim(M=1), Fraction(1, 1000)),
    _u("km", _dim(L=1), 1000), _u("cm", _dim(L=1), Fraction(1, 100)),
    _u("mm", _dim(L=1), Fraction(1, 1000)),
    _u("min", _dim(T=1), 60), _u("h", _dim(T=1), 3600), _u("day", _dim(T=1), 86400),
    _u("L", _dim(L=3), Fraction(1, 1000)), _u("mL", _dim(L=3), Fraction(1, 10**6)),
    _u("Hz", _dim(T=-1), 1),
    _u("N", _dim(M=1, L=1, T=-2), 1),
    _u("Pa", _dim(M=1, L=-1, T=-2), 1),
    _u("J", _dim(M=1, L=2, T=-2), 1),
    _u("W", _dim(M=1, L=2, T=-3), 1),
    _u("C", _dim(I=1, T=1), 1),
    _u("V", _dim(M=1, L=2, T=-3, I=-1), 1),
    _u("ohm", _dim(M=1, L=2, T=-3, I=-2), 1),
    _u("eV", _dim(M=1, L=2, T=-2), Fraction(1602176634, 10**28)),
    _u("cal", _dim(M=1, L=2, T=-2), Fraction(4184, 1000)),  # thermochemical, exact
    _u("bar", _dim(M=1, L=-1, T=-2), 100000),
    _u("mph", _dim(L=1, T=-1), Fraction(44704, 100000)),  # 0.44704 m/s exact
])


def parse_unit(text: str) -> Unit:
    """Parse a unit expression like 'kg*m/s^2' or 'km/h' into a composite
    unit (dimension + exact scale). No parentheses; ^ takes integer powers."""
    dim, scale = DIMENSIONLESS, Fraction(1)
    for i, segment in enumerate(text.split("/")):
        for token in segment.split("*"):
            token = token.strip()
            if not token:
                continue
            if "^" in token:
                name, _, exp = token.partition("^")
                power = Fraction(int(exp))
            else:
                name, power = token, Fraction(1)
            unit = REGISTRY.get(name)
            if unit is None:
                raise ValueError(f"unknown unit '{name}'")
            if i > 0:
                power = -power
            dim = dim * (unit.dimension ** power)
            scale *= unit.scale ** power
    return Unit(text, dim, scale)


def convert(value: str, from_unit: str, to_unit: str) -> dict:
    """Exact rational conversion of a numeric value between units."""
    src, dst = parse_unit(from_unit), parse_unit(to_unit)
    if src.dimension != dst.dimension:
        raise ValueError(
            f"dimensional mismatch: '{from_unit}' is [{src.dimension}] but "
            f"'{to_unit}' is [{dst.dimension}]")
    result = Fraction(value) * src.scale / dst.scale
    return {"value": str(result), "dimension": str(src.dimension)}


def simplify_unit(text: str) -> dict:
    """Reduce a unit expression to its SI dimension and exact scale."""
    unit = parse_unit(text)
    return {"dimension": str(unit.dimension), "si_scale": str(unit.scale)}


def dimension_of(ir: Expr, units: dict[str, str]) -> Dimension:
    """Dimensional check of a MathIR expression. `units` maps free symbol
    names to unit expressions. Raises ValueError on any inconsistency."""
    dims = {name: parse_unit(u).dimension for name, u in units.items()}

    def walk(node: Expr) -> Dimension:
        if isinstance(node, (IntegerNode, RationalNode, RealNode, NumberNode)):
            return DIMENSIONLESS
        if isinstance(node, SymbolNode):
            return dims.get(node.name, DIMENSIONLESS)
        if isinstance(node, UnaryNode):  # neg
            return walk(node.arg)
        if isinstance(node, NaryNode):
            ds = [walk(a) for a in node.args]
            if node.kind == "add":
                if len(set(ds)) > 1:
                    raise ValueError(
                        f"cannot add/subtract quantities of different dimensions: "
                        f"{[str(d) for d in ds]}")
                return ds[0]
            out = DIMENSIONLESS  # mul
            for d in ds:
                out = out * d
            return out
        if isinstance(node, BinaryNode):
            dl, dr = walk(node.left), walk(node.right)
            if node.kind == "div":
                return dl / dr
            if not dr.is_dimensionless():  # pow
                raise ValueError("exponent must be dimensionless")
            if node.right.kind == "integer":
                return dl ** Fraction(int(node.right.value))
            if node.right.kind == "rational":
                return dl ** Fraction(node.right.value)
            if not dl.is_dimensionless():
                raise ValueError(
                    "only integer/rational powers of dimensioned quantities "
                    "are supported")
            return DIMENSIONLESS
        if isinstance(node, CallNode):
            ds = [walk(a) for a in node.args]
            if node.name in ("sqrt",):
                return ds[0] ** Fraction(1, 2)
            if node.name in ("sin", "cos", "tan", "exp", "log", "ln"):
                if not ds[0].is_dimensionless():
                    raise ValueError(f"{node.name} requires a dimensionless argument")
                return DIMENSIONLESS
            if node.name in ("abs", "floor", "ceil"):
                return ds[0]
            raise ValueError(f"unsupported function '{node.name}' for dimensional analysis")
        raise ValueError(f"node kind '{node.kind}' has no dimensional semantics")

    return walk(ir)
