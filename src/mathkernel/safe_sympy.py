# =============================================================================
# MathKernel - safe SymPy srepr decoding for persistence
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Decode MathKernel-owned SymPy ``srepr`` data without eval/sympify.

Persistence is an input boundary.  Stored symbolic values are therefore
reconstructed from a restricted Python AST and an explicit constructor
whitelist.  Attribute access, imports, comprehensions, lambdas and arbitrary
function calls are never accepted.
"""
from __future__ import annotations

import ast
from typing import Any

import sympy as sp
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.geometry import Point2D, Point3D, Segment2D, Segment3D, Line2D, Line3D, Ray2D, Ray3D

_EXPLICIT = {
    "ExprCondPair": ExprCondPair,
    "Point2D": Point2D, "Point3D": Point3D,
    "Segment2D": Segment2D, "Segment3D": Segment3D,
    "Line2D": Line2D, "Line3D": Line3D,
    "Ray2D": Ray2D, "Ray3D": Ray3D,
}


# Broad enough for MathKernel's typed-domain objects while still explicit.
_ALLOWED_NAMES = {
    # atoms / core
    "Symbol", "Dummy", "Integer", "Rational", "Float", "Add", "Mul", "Pow",
    "Mod", "Tuple", "Function", "Lambda",
    # constants / singleton classes as emitted by srepr
    "true", "false", "oo", "zoo", "nan", "I", "E", "pi", "EulerGamma",
    "Infinity", "NegativeInfinity", "ComplexInfinity", "NaN", "ImaginaryUnit",
    "Exp1", "Pi", "EulerGamma",
    # relations / logic
    "Equality", "Unequality", "StrictGreaterThan", "StrictLessThan", "GreaterThan",
    "LessThan", "And", "Or", "Not", "Xor", "Implies", "Equivalent", "Contains",
    # elementary / special functions commonly persisted by typed domains
    "Abs", "sign", "floor", "ceiling", "exp", "log", "sin", "cos", "tan",
    "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "asinh", "acosh", "atanh", "factorial",
    "gamma", "loggamma", "polygamma", "digamma", "beta", "erf", "erfc",
    "LambertW", "Heaviside", "DiracDelta", "re", "im", "conjugate", "arg",
    "Min", "Max",
    # calculus / piecewise
    "Piecewise", "ExprCondPair", "Derivative", "Integral", "Sum", "Product",
    "Limit", "Order",
    # sets / geometry
    "Interval", "FiniteSet", "Union", "Intersection", "Complement", "ProductSet",
    "EmptySet", "UniversalSet", "Naturals", "Naturals0", "Integers", "Rationals",
    "Reals", "Complexes", "Range", "ImageSet", "ConditionSet",
    "Point2D", "Point3D", "Segment2D", "Segment3D", "Line2D", "Line3D",
    "Ray2D", "Ray3D", "Circle", "Polygon", "Triangle",
    # matrices / polynomials occasionally present in certificates
    "Matrix", "ImmutableDenseMatrix", "ImmutableSparseMatrix", "Poly", "ComplexRootOf",
}


def _sympy_constructor(name: str) -> Any:
    if name not in _ALLOWED_NAMES:
        raise ValueError(f"stored SymPy constructor is not allowed: {name}")
    # srepr emits singleton names such as true/oo directly rather than calls.
    if name in _EXPLICIT:
        return _EXPLICIT[name]
    if hasattr(sp, name):
        return getattr(sp, name)
    # Some geometry classes are not re-exported uniformly across SymPy versions.
    for module in (sp.geometry, sp.sets, sp.matrices, sp.polys):
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"stored SymPy constructor is unavailable: {name}")


def _decode(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _decode(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool, type(None))):
            return node.value
        raise ValueError("unsupported constant in stored SymPy expression")
    if isinstance(node, ast.Name):
        value = _sympy_constructor(node.id)
        # Singleton objects (oo, pi, true, Reals, ...) may be names in srepr.
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _decode(node.operand)
        if isinstance(value, (int, float)):
            return -value
        raise ValueError("unsupported unary value in stored SymPy expression")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_decode(item) for item in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("attribute/dynamic calls are forbidden in stored SymPy data")
        constructor = _sympy_constructor(node.func.id)
        if not callable(constructor):
            raise ValueError(f"stored SymPy name is not callable: {node.func.id}")
        args = [_decode(arg) for arg in node.args]
        kwargs = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("**kwargs are forbidden in stored SymPy data")
            kwargs[keyword.arg] = _decode(keyword.value)
        try:
            return constructor(*args, **kwargs)
        except Exception as exc:  # normalize parser/constructor errors
            raise ValueError(
                f"invalid stored SymPy constructor {node.func.id}: {exc}") from exc
    raise ValueError(
        f"forbidden syntax in stored SymPy expression: {type(node).__name__}")


def decode_srepr(text: str) -> sp.Basic:
    """Safely reconstruct one SymPy object from trusted-format ``srepr`` text."""
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError("stored SymPy expression is invalid or too large")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError("stored SymPy expression is not valid srepr syntax") from exc
    value = _decode(tree)
    if not isinstance(value, sp.Basic):
        raise ValueError("stored SymPy value did not decode to a SymPy object")
    return value
