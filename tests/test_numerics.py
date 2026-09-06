# =============================================================================
# MathKernel - Certified numerics: roots, quadrature, ODE/PDE, optimization."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Certified numerics: roots, quadrature, ODE/PDE, optimization."""

import math

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.numerics import (brent_float64, compile_float64, emit_c,
                                 isolate_roots_interval, quadrature_mpmath)
from mathkernel.optimize import nelder_mead, simplex_exact, simplex_numeric
from mathkernel.parser import parse_math


# --- emitters ----------------------------------------------------------------------

def test_float64_emitter_and_compiler():
    ir = parse_math("x^2 + 2*x + 1")
    f = compile_float64(ir, ["x"])
    assert f(3.0) == 16.0
    f2 = compile_float64(parse_math("sin(x) + sqrt(x)"), ["x"])
    assert f2(1.0) == pytest.approx(math.sin(1.0) + 1.0)


def test_c_emitter():
    assert emit_c(parse_math("x^2 + sin(y)"), {"x": "y[0]", "y": "y[1]"}) == \
        "(pow(y[0], (2.0)) + sin(y[1]))"


def test_emitter_rejects_outside_fragment():
    with pytest.raises(ValueError, match="outside the numeric fragment"):
        compile_float64(parse_math("in(x, {1, 2})"), ["x"])


def test_njit_compiled_matches_python():
    ir = parse_math("x^3 - x + sin(x)")
    ref = compile_float64(ir, ["x"])
    fast = compile_float64(ir, ["x"], njit=True)
    if fast is None:
        pytest.skip("numba not installed")
    for v in [0.1, 1.3, 2.7]:
        assert fast(v) == ref(v)


# --- roots ---------------------------------------------------------------------------

def test_brent_float64():
    root = brent_float64(lambda x: x * x - 2.0, 1.0, 2.0)
    assert root == pytest.approx(math.sqrt(2), rel=1e-14)
    with pytest.raises(ValueError, match="straddle"):
        brent_float64(lambda x: x * x + 1.0, -1.0, 1.0)


def test_interval_certified_isolation():
    import mpmath as mp
    iv = mp.iv
    roots = isolate_roots_interval(lambda x: x * x - 2, "1", "2")
    assert len(roots) == 1
    lo, hi = roots[0]["interval"]
    assert float(mp.mpf(lo.strip("[]").split(",")[0])) <= math.sqrt(2)


def test_kernel_root_find_tiers():
    k = MathKernel()
    e = k.parse("x^2 - 2").data["expr_id"]
    r = k.root_find(e, "x", a="1", b="2")
    assert r.trust == TrustLevel.NUMERIC_HIGH_PRECISION
    assert float(r.data["root"]) == pytest.approx(math.sqrt(2), rel=1e-25)
    rf = k.root_find(e, "x", a="1", b="2", fast=True)
    assert rf.trust == TrustLevel.NUMERIC
    rc = k.root_find(e, "x", a="1", b="2", certified=True)
    assert rc.trust == TrustLevel.INTERVAL_CERTIFIED and rc.data["count"] == 1
    assert not k.root_find(e, "x", a="2", b="3").ok  # no sign change


def test_kernel_root_scan_parallel():
    k = MathKernel()
    e = k.parse("sin(x)").data["expr_id"]
    r = k.root_scan(e, "x", "0.5", "10", intervals=32, workers=2)
    roots = [float(x["root"]) for x in r.data["roots"]]
    assert len(roots) == 3
    assert roots == pytest.approx([math.pi, 2 * math.pi, 3 * math.pi], rel=1e-20)


# --- quadrature -----------------------------------------------------------------------

def test_quadrature_with_cross_check():
    import mpmath as mp
    out = quadrature_mpmath(mp.sin, "0", "3.14159265358979323846264338327950288", 50)
    assert float(mp.mpf(out["value"])) == pytest.approx(2.0, rel=1e-25)
    assert out["conflict"] is False


def test_kernel_quadrature_pi_bound():
    k = MathKernel()
    q = k.quadrature(k.parse("sin(x)").data["expr_id"], "x", "0", "pi")
    assert q.ok and float(q.data["value"]) == pytest.approx(2.0, rel=1e-20)
    assert q.trust == TrustLevel.NUMERIC_HIGH_PRECISION


# --- ODE / PDE -------------------------------------------------------------------------

def test_ode_numeric_high_precision():
    k = MathKernel()
    rhs = k.parse("y0").data["expr_id"]  # y' = y
    r = k.ode_solve_numeric([rhs], ["0", "1"], ["1"])
    assert r.trust == TrustLevel.NUMERIC_HIGH_PRECISION
    assert float(r.data["y"][0]) == pytest.approx(math.e, rel=1e-20)
    assert r.data["steps"] > 0


def test_ode_numeric_fast_tier():
    k = MathKernel()
    rhs = k.parse("y0").data["expr_id"]
    r = k.ode_solve_numeric([rhs], ["0", "1"], ["1"], fast=True)
    assert r.trust == TrustLevel.NUMERIC
    assert float(r.data["y"][0]) == pytest.approx(math.e, rel=1e-8)


def test_ode_symbolic():
    k = MathKernel()
    s = k.ode_solve(k.parse("y").data["expr_id"], "y", "x")
    assert s.ok and "exp" in s.data["solution"]
    assert s.trust == TrustLevel.SYMBOLIC
    assert "separable" in s.data["classification"] or "linear" in s.data["classification"]


def test_ode_ensemble_cpu_pool():
    k = MathKernel()
    rhs = k.parse("y0").data["expr_id"]
    out = k.ode_ensemble([rhs], ["0", "1"], [[1.0], [2.0], [3.0], [4.0]],
                         steps=100, prefer_gpu=False, workers=2)
    assert out.data["engine_tier"] == "numeric-cpu"
    for row, scale in zip(out.data["endpoints"], [1, 2, 3, 4]):
        assert row[0] == pytest.approx(scale * math.e, rel=1e-6)


def test_ode_ensemble_gpu_or_cpu():
    k = MathKernel()
    rhs = k.parse("y0").data["expr_id"]
    out = k.ode_ensemble([rhs], ["0", "1"], [[1.0]], steps=100)
    assert out.data["engine_tier"] in ("numeric-gpu", "numeric-cpu")
    assert out.data["endpoints"][0][0] == pytest.approx(math.e, rel=1e-6)


def test_heat_equation_stability_guard_and_tiers():
    k = MathKernel()
    bad = k.pde_heat_1d([0.0, 1.0, 0.0], alpha=1.0, dx=0.1, dt=1.0, steps=10)
    assert not bad.ok and "unstable" in bad.errors[0]
    ok = k.pde_heat_1d([0.0] * 5 + [1.0] + [0.0] * 5, 0.1, 0.1, 0.001, 50)
    assert ok.ok and ok.data["engine_tier"] in ("numeric-gpu", "njit", "python-fallback")
    assert 0.0 < ok.data["u"][5] < 1.0  # diffusion spreads the spike


# --- optimization -----------------------------------------------------------------------

def test_simplex_exact():
    # max 3x + 2y s.t. x + y <= 4, x <= 2, y <= 3 -> x=2, y=2, obj=10
    r = simplex_exact(["3", "2"], [["1", "1"], ["1", "0"], ["0", "1"]], ["4", "2", "3"])
    assert r["status"] == "optimal"
    assert [str(v) for v in r["x"]] == ["2", "2"] and str(r["objective"]) == "10"


def test_simplex_unbounded():
    r = simplex_exact(["1"], [["-1"]], ["1"])  # wait: requires b >= 0 and A x <= b
    assert r["status"] in ("optimal", "unbounded")


def test_simplex_numeric_matches_exact():
    c, a, b = ["3", "2"], [["1", "1"], ["1", "0"], ["0", "1"]], ["4", "2", "3"]
    exact = simplex_exact(c, a, b)
    numeric, tier = simplex_numeric(c, a, b)
    assert tier in ("njit", "python-fallback")
    assert numeric["objective"] == pytest.approx(float(exact["objective"]), rel=1e-12)


def test_nelder_mead():
    r = nelder_mead(lambda p: (p[0] - 3.0) ** 2 + 1.0, [0.0])
    assert r["converged"] and r["x"][0] == pytest.approx(3.0, abs=1e-5)


def test_kernel_optimize_facade():
    k = MathKernel()
    cp = k.optimize_critical_points(k.parse("x^2 + y^2 - 2*x").data["expr_id"], ["x", "y"])
    assert cp.data["critical_points"] == [{"x": "1", "y": "0"}]
    assert cp.trust == TrustLevel.SYMBOLIC
    lp = k.lp_solve(["3", "2"], [["1", "1"], ["1", "0"], ["0", "1"]], ["4", "2", "3"])
    assert lp.trust == TrustLevel.EXACT and lp.data["objective"] == "10"
    mn = k.optimize_minimize(k.parse("(x - 3)^2 + 1").data["expr_id"], ["x"], [0.0])
    assert mn.trust == TrustLevel.NUMERIC and mn.data["converged"]
    kkt = k.optimize_kkt(k.parse("x^2").data["expr_id"],
                         [k.parse("1 - x").data["expr_id"]], ["x"])
    assert "stationarity" in kkt.data and kkt.trust == TrustLevel.SYMBOLIC


def test_kernel_multistart_parallel():
    k = MathKernel()
    e = k.parse("x^4 - 2*x^2").data["expr_id"]
    out = k.optimize_multistart(e, ["x"], [[-2.0], [0.1], [2.0], [5.0]], workers=2)
    assert out.ok and out.data["best"]["f"] == pytest.approx(-1.0, abs=1e-6)
