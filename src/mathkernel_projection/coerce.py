# =============================================================================
# MathKernel Projection - numeric/symbolic coercion helpers for adapters
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Shared coercion helpers for result adapters.

Adapters receive JSON-like dicts whose numbers may be plain floats, decimal
strings, Fraction encodings, or SymPy serialisations.  These helpers convert
them defensively; anything non-numeric raises a typed ``ValueError`` instead
of being silently dropped.
"""
from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float:
    """Coerce an int/float/decimal string/Fraction/SymPy literal to float."""
    if isinstance(value, bool):
        raise ValueError(f"not a numeric value: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        # SymPy string serialisation (e.g. "sqrt(2)", "2*pi/3").
        import sympy as sp
        try:
            expr = sp.sympify(text, rational=True)
        except Exception as exc:
            raise ValueError(f"not a numeric value: {value!r}") from exc
        if not expr.is_number:
            raise ValueError(f"not a closed numeric value: {value!r}")
        return float(expr.evalf())
    if isinstance(value, dict):
        if "__fraction__" in value:
            from fractions import Fraction
            return float(Fraction(value["__fraction__"]))
        if "__sympy_srepr__" in value:
            import sympy as sp
            from mathkernel.safe_sympy import decode_srepr
            expr = decode_srepr(value["__sympy_srepr__"])
            if not isinstance(expr, sp.Basic) or not expr.is_number:
                raise ValueError(f"not a closed numeric value: {value!r}")
            return float(expr.evalf())
    if hasattr(value, "evalf"):
        return float(value.evalf())
    raise ValueError(f"not a numeric value: {value!r}")


def to_complex(value: Any) -> complex:
    """Coerce a complex number serialisation ("1 + 2*I", [re, im]) to complex."""
    if isinstance(value, complex):
        return value
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(to_float(value[0]), to_float(value[1]))
    if isinstance(value, dict) and "real" in value and "imag" in value:
        return complex(to_float(value["real"]), to_float(value["imag"]))
    import sympy as sp
    try:
        expr = sp.sympify(str(value), rational=True)
    except Exception as exc:
        raise ValueError(f"not a complex value: {value!r}") from exc
    try:
        return complex(expr.evalf())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"not a complex value: {value!r}") from exc


def to_float_list(values: Any, *, what: str = "values") -> list[float]:
    """Coerce a sequence to floats with a typed error naming the context."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{what} must be a sequence, got {type(values).__name__}")
    return [to_float(v) for v in values]


def sample_symbolic(expression: Any, variable: Any, lo: float, hi: float,
                    samples: int) -> tuple[list[float], list[float]]:
    """Sample a SymPy-serialised scalar expression on a uniform grid.

    Returns (xs, ys).  This is a presentation-layer sampling: callers must
    declare ``information_loss=['sampling']`` and record the grid in the
    projection parameters.
    """
    import sympy as sp
    expr = expression if isinstance(expression, sp.Basic) else sp.sympify(
        str(expression), rational=True)
    var = variable if isinstance(variable, sp.Symbol) else sp.Symbol(str(variable))
    if samples < 2:
        raise ValueError("samples must be at least 2")
    try:
        fn = sp.lambdify(var, expr, modules="math")
    except Exception:
        fn = None
    step = (hi - lo) / (samples - 1)
    xs: list[float] = []
    ys: list[float] = []
    for i in range(samples):
        x = lo + i * step
        try:
            y = float(fn(x)) if fn is not None else float(expr.subs(var, x).evalf())
        except Exception as exc:
            raise ValueError(
                f"expression is not numerically evaluable at {variable}={x}: "
                f"{exc}") from exc
        xs.append(x)
        ys.append(y)
    return xs, ys
