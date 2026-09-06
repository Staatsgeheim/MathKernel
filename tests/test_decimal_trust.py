# =============================================================================
# MathKernel - decimal trust hardening tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Decimal literals must not acquire exact/symbolic-only trust."""
from mathkernel.kernel import MathKernel
from mathkernel.models import TrustLevel

def test_parse_decimal_is_numeric_not_exact():
    k=MathKernel()
    r=k.parse("0.7*x + 1")
    assert r.trust == TrustLevel.NUMERIC

def test_parse_exact_rational_remains_symbolic_or_exact_not_numeric():
    k=MathKernel()
    r=k.parse("(7/10)*x + 1")
    assert r.trust != TrustLevel.NUMERIC

def test_simplify_decimal_inherits_numeric_trust():
    k=MathKernel()
    p=k.parse("0.7*x + 1")
    r=k.simplify(p.data["expr_id"])
    assert r.trust == TrustLevel.NUMERIC

def test_differentiate_decimal_inherits_numeric_trust():
    k=MathKernel()
    p=k.parse("0.7*x*x")
    r=k.differentiate(p.data["expr_id"],"x")
    assert r.trust == TrustLevel.NUMERIC

def test_integrate_decimal_inherits_numeric_trust():
    k=MathKernel()
    p=k.parse("0.7*x")
    r=k.integrate(p.data["expr_id"],"x")
    assert r.trust == TrustLevel.NUMERIC

def test_solve_decimal_inherits_numeric_trust():
    k=MathKernel()
    p=k.parse("0.7*x - 1.4")
    r=k.solve(p.data["expr_id"],"x")
    assert r.trust == TrustLevel.NUMERIC

def test_exact_rational_simplify_stays_symbolic():
    k=MathKernel()
    p=k.parse("(7/10)*x + 1")
    r=k.simplify(p.data["expr_id"])
    assert r.trust == TrustLevel.SYMBOLIC
