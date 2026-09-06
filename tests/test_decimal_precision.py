# =============================================================================
# MathKernel - decimal precision tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Regression tests for short decimal literals in the symbolic engine."""
from mathkernel.kernel import MathKernel
from mathkernel.engines import SymPyEngine
from mathkernel.parser import parse_math


def test_short_decimal_literal_keeps_value_in_sympy():
    eng = SymPyEngine()
    value = eng.to_sympy(parse_math("0.7"))
    assert abs(float(value) - 0.7) < 1e-14


def test_short_decimal_does_not_round_kuramoto_expression_to_one():
    k = MathKernel()
    parsed = k.parse("2/(pi*(1/(pi*0.7)))")
    result = k.simplify(parsed.data["expr_id"])
    assert abs(float(result.data["result"]) - 1.4) < 1e-14


def test_decimal_and_exact_rational_agree_numerically():
    k = MathKernel()
    pd = k.parse("2/(pi*(1/(pi*0.7)))")
    pr = k.parse("2/(pi*(1/(pi*(7/10))))")
    rd = k.simplify(pd.data["expr_id"])
    rr = k.simplify(pr.data["expr_id"])
    assert abs(float(rd.data["result"]) - float(7/5)) < 1e-14
    assert rr.data["result"] == "7/5"


def test_other_short_decimals_are_not_degraded():
    eng = SymPyEngine()
    for text in ("0.1", "0.2", "0.9", "1.5", "12.3"):
        got = float(eng.to_sympy(parse_math(text)))
        assert abs(got - float(text)) < 1e-14
