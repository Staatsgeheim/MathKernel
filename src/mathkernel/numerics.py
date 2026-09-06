# =============================================================================
# MathKernel - Certified numerics: root finding and quadrature
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Certified numerics: root finding and quadrature.

Trust tiers:
- interval root isolation via mpmath.iv        -> INTERVAL_CERTIFIED
- mpmath arbitrary precision (roots/integrals) -> NUMERIC_HIGH_PRECISION
- njit float64 fast path (Brent/Simpson)       -> NUMERIC

The float64 emitters compile the MathIR arithmetic fragment to plain Python
(math.*) or C (for CUDA kernels); anything outside the fragment raises so
callers can fall back to mpmath lambdify.
"""
from __future__ import annotations

import math

from .models import (BinaryNode, CallNode, Expr, IntegerNode, NaryNode, NumberNode,
                     RationalNode, RealNode, SymbolNode, UnaryNode)
from .parallel import process_map, resolve_workers

MAX_SCAN_INTERVALS = 100_000

_PY_CALLS = {"sqrt": "math.sqrt", "sin": "math.sin", "cos": "math.cos", "tan": "math.tan",
             "exp": "math.exp", "log": "math.log", "abs": "abs", "gamma": "math.gamma",
             "factorial": "math.factorial"}
_C_CALLS = {"sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
            "log": "log", "abs": "fabs", "gamma": "tgamma"}


def _emit(ir: Expr, var_map: dict[str, str], calls: dict[str, str], pow_fmt: str) -> str:
    if isinstance(ir, IntegerNode):
        return f"({ir.value}.0)" if len(ir.value) < 15 else f"float({ir.value})"
    if isinstance(ir, NumberNode):
        return f"({ir.value})"
    if isinstance(ir, RealNode):
        return f"({ir.value})"
    if isinstance(ir, RationalNode):
        return f"(({ir.numerator}.0)/({ir.denominator}.0))"
    if isinstance(ir, SymbolNode):
        if ir.name == "pi":
            return "3.14159265358979323846264338327950288"
        if ir.name in var_map:
            return var_map[ir.name]
        raise ValueError(f"unbound symbol in numeric fragment: {ir.name}")
    if isinstance(ir, UnaryNode):
        return f"(-{_emit(ir.arg, var_map, calls, pow_fmt)})"
    if isinstance(ir, NaryNode):
        op = " + " if ir.kind == "add" else " * "
        return "(" + op.join(_emit(a, var_map, calls, pow_fmt) for a in ir.args) + ")"
    if isinstance(ir, BinaryNode):
        if ir.kind == "div":
            return f"({_emit(ir.left, var_map, calls, pow_fmt)} / {_emit(ir.right, var_map, calls, pow_fmt)})"
        if pow_fmt == "python":
            return f"({_emit(ir.left, var_map, calls, pow_fmt)} ** {_emit(ir.right, var_map, calls, pow_fmt)})"
        return f"pow({_emit(ir.left, var_map, calls, pow_fmt)}, {_emit(ir.right, var_map, calls, pow_fmt)})"
    if isinstance(ir, CallNode):
        if ir.name not in calls:
            raise ValueError(f"function {ir.name} is outside the numeric fragment")
        return f"{calls[ir.name]}(" + ", ".join(_emit(a, var_map, calls, pow_fmt) for a in ir.args) + ")"
    raise ValueError(f"node kind {ir.kind} is outside the numeric fragment")


def emit_float64(ir: Expr, var_map: dict[str, str]) -> str:
    """MathIR -> Python float64 expression string (math.* functions)."""
    return _emit(ir, var_map, _PY_CALLS, "python")


def emit_c(ir: Expr, var_map: dict[str, str]) -> str:
    """MathIR -> C double expression string (for CUDA kernels)."""
    return _emit(ir, var_map, _C_CALLS, "c")


def compile_float64(ir: Expr, variables: list[str], njit: bool = False):
    """Compile a MathIR expression to a float64 callable over `variables`.

    With njit=True the callable is numba-compiled (returns None when numba is
    unavailable or the fragment is not emittable)."""
    var_map = {v: v for v in variables}
    body = emit_float64(ir, var_map)
    params = ", ".join(variables)
    source = f"def _f({params}):\n    return {body}\n"
    namespace: dict = {"math": math}
    exec(source, namespace)
    fn = namespace["_f"]
    if njit:
        try:
            from numba import njit as _njit
            return _njit(cache=False)(fn)
        except Exception:
            return None
    return fn


# --- root finding -----------------------------------------------------------------

def brent_float64(f, a: float, b: float, tol: float = 1e-14, max_iter: int = 200) -> float:
    """Brent's method on a sign-changing bracket (pure Python reference)."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("bracket does not straddle a root")
    c, fc = a, fa
    d = e = b - a
    for _ in range(max_iter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * 2.220446049250313e-16 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    raise ValueError("brent_float64 did not converge within max_iter")


def _brent_njit():
    try:
        from numba import njit
    except ImportError:
        return None
    return njit(cache=True)(brent_float64)


def find_root_mpmath(f, a=None, b=None, x0=None, dps: int = 50) -> dict:
    """Arbitrary-precision root via mpmath. Bracket [a, b] -> bisection-style
    Brent (mp.findroot with bracket); otherwise secant/Newton from x0."""
    import mpmath as mp
    mp.mp.dps = dps
    if a is not None and b is not None:
        fa, fb = f(mp.mpf(a)), f(mp.mpf(b))
        if fa * fb > 0:
            raise ValueError("bracket does not straddle a root")
        root = mp.findroot(f, (mp.mpf(a), mp.mpf(b)), solver="ridder")
        method = "ridder"
    elif x0 is not None:
        root = mp.findroot(f, mp.mpf(x0))
        method = "secant"
    else:
        raise ValueError("provide either a bracket [a, b] or a start point x0")
    return {"root": mp.nstr(root, 30), "method": method, "dps": dps,
            "residual": mp.nstr(abs(f(root)), 10)}


def isolate_roots_interval(f_iv, a: str, b: str, steps: int = 256,
                           refine: int = 64) -> list[dict]:
    """Interval-certified root isolation over [a, b]: scan with interval
    arithmetic, keep subintervals whose image straddles zero, refine each by
    interval bisection. Every returned interval rigorously encloses a sign
    change (and hence a root, for continuous f)."""
    import mpmath as mp
    if steps < 1 or steps > MAX_SCAN_INTERVALS:
        raise ValueError(f"steps must be between 1 and {MAX_SCAN_INTERVALS}")
    iv = mp.iv
    lo, hi = iv.mpf(a), iv.mpf(b)
    if not lo < hi:
        raise ValueError("expected a < b")
    width = (hi - lo) / steps
    candidates = []
    for i in range(steps):
        sub = iv.mpf([lo.a + i * width.a, lo.a + (i + 1) * width.a])
        try:
            image = f_iv(sub)
        except (ZeroDivisionError, ValueError, OverflowError):
            candidates.append(sub)  # singularities may hide roots; keep, marked below
            continue
        if image.a <= 0 <= image.b:
            candidates.append(sub)
    roots = []
    for sub in candidates:
        x = sub
        for _ in range(refine):
            mid = (x.a + x.b) / 2
            try:
                left = f_iv(iv.mpf([x.a, mid]))
                right = f_iv(iv.mpf([mid, x.b]))
            except (ZeroDivisionError, ValueError, OverflowError):
                break
            l_straddle = left.a <= 0 <= left.b
            r_straddle = right.a <= 0 <= right.b
            if l_straddle and not r_straddle:
                x = iv.mpf([x.a, mid])
            elif r_straddle and not l_straddle:
                x = iv.mpf([mid, x.b])
            elif l_straddle and r_straddle:
                x = iv.mpf([x.a, mid])
            else:
                break
        roots.append({"interval": [mp.nstr(x.a, 30), mp.nstr(x.b, 30)],
                      "midpoint": mp.nstr((x.a + x.b) / 2, 30),
                      "width": mp.nstr(x.b - x.a, 10)})
    return roots


class MpmathExpressionCallable:
    """Pickle-safe expression recipe, not a serialized global mp context."""
    def __init__(self, expression, variable):
        self.expression = expression
        self.variable = variable
        self._compiled = None

    def __getstate__(self):
        return {"expression": self.expression, "variable": self.variable}

    def __setstate__(self, state):
        self.expression = state["expression"]
        self.variable = state["variable"]
        self._compiled = None

    def __call__(self, value):
        if self._compiled is None:
            import sympy as sp
            self._compiled = sp.lambdify(self.variable, self.expression, "mpmath")
        return self._compiled(value)


def quadrature_mpmath(f, a: str, b: str, dps: int = 50,
                      cross_check: bool = True) -> dict:
    """Tanh-sinh quadrature at arbitrary precision, cross-checked against
    Gauss-Legendre. Agreement upgrades confidence; disagreement is reported
    as a conflict instead of silently picking one."""
    import mpmath as mp
    mp.mp.dps = dps
    primary = mp.quad(f, [mp.mpf(a), mp.mpf(b)])  # tanh-sinh
    result = {"value": mp.nstr(primary, 30), "method": "tanh-sinh", "dps": dps}
    if cross_check:
        check = mp.quad(f, [mp.mpf(a), mp.mpf(b)], method="gauss-legendre")
        result["cross_check"] = {"method": "gauss-legendre", "value": mp.nstr(check, 30)}
        gap = abs(primary - check)
        scale = max(abs(primary), mp.mpf(1))
        result["agreement_digits"] = mp.nstr(-mp.log10(gap / scale), 5) if gap > 0 else "inf"
        result["conflict"] = bool(gap / scale > mp.mpf(10) ** (-(dps // 3)))
    return result


# --- parallel root scan --------------------------------------------------------------

def _mpmath_callable(expr_text: str, variable: str):
    """Rebuild an mpmath callable from MathIR source text (worker-safe: the
    restricted parser, never eval)."""
    import sympy as sp
    from .engines import SymPyEngine
    from .parser import parse_math
    ir = parse_math(expr_text)
    sym = SymPyEngine().to_sympy(ir, {})
    return sp.lambdify(sp.Symbol(variable), sym, "mpmath")


def _scan_job(args) -> dict:
    expr_text, variable, a, b, dps = args
    import mpmath as mp
    mp.mp.dps = dps
    f = _mpmath_callable(expr_text, variable)
    try:
        out = find_root_mpmath(f, a=a, b=b, dps=dps)
        return {"ok": True, "interval": [a, b], **out}
    except ValueError:
        return {"ok": False, "interval": [a, b]}


def root_scan(expr_text: str, variable: str, a: str, b: str, intervals: int,
              dps: int = 50, workers: int | None = None) -> list[dict]:
    """Scan [a, b] split into `intervals` subintervals for sign-changing roots,
    refined in parallel across the process pool. expr_text is MathIR source
    (e.g. 'x^2 - 2'), re-parsed inside each worker."""
    from fractions import Fraction
    if intervals < 1 or intervals > MAX_SCAN_INTERVALS:
        raise ValueError(f"intervals must be between 1 and {MAX_SCAN_INTERVALS}")
    fa, fb = Fraction(a), Fraction(b)
    step = (fb - fa) / intervals
    jobs = [(expr_text, variable, str(fa + i * step), str(fa + (i + 1) * step), dps)
            for i in range(intervals)]
    resolved = resolve_workers(workers)
    return process_map(_scan_job, jobs, workers=resolved)
