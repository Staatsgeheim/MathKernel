# =============================================================================
# MathKernel - test latex
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import sympy as sp
import pytest

from mathkernel import MathKernel

pytest.importorskip("antlr4", reason="antlr4-python3-runtime not installed")


@pytest.fixture
def kernel():
    return MathKernel()


def _parse_latex(kernel, text):
    result = kernel.parse_latex(text)
    assert result.ok, result.errors
    return result


def test_fraction_and_power(kernel):
    result = _parse_latex(kernel, r"\frac{x^2}{2}")
    expr = kernel.sympy.to_sympy(kernel.expressions[result.data["expr_id"]])
    assert sp.simplify(expr - sp.Symbol("x")**2 / 2) == 0


def test_sqrt_and_pi(kernel):
    result = _parse_latex(kernel, r"\sqrt{y} + \pi")
    expr = kernel.sympy.to_sympy(kernel.expressions[result.data["expr_id"]])
    assert sp.simplify(expr - (sp.sqrt(sp.Symbol("y")) + sp.pi)) == 0


def test_binomial(kernel):
    result = _parse_latex(kernel, r"\binom{n}{k}")
    expr = kernel.sympy.to_sympy(kernel.expressions[result.data["expr_id"]])
    n, k = sp.symbols("n k")
    assert sp.simplify(expr - sp.binomial(n, k)) == 0


def test_euler_identity_relation(kernel):
    result = _parse_latex(kernel, r"e^{i\pi} + 1 = 0")
    ir = kernel.expressions[result.data["expr_id"]]
    assert ir.kind == "eq"


def test_latex_result_is_chainable(kernel):
    parsed = _parse_latex(kernel, r"x^3")
    result = kernel.differentiate(parsed.data["expr_id"], "x")
    assert result.ok
    out = kernel.sympy.to_sympy(kernel.expressions[result.data["result_expr_id"]])
    assert sp.simplify(out - 3*sp.Symbol("x")**2) == 0


def test_latex_source_preserved(kernel):
    result = _parse_latex(kernel, r"\frac{1}{x}")
    eid = result.data["expr_id"]
    assert result.data["source"] == r"\frac{1}{x}"
    assert kernel.expression_sources[eid] == r"\frac{1}{x}"
    assert result.data["display"]


def test_latex_parse_error(kernel):
    result = kernel.parse_latex(r"\frac{1")
    assert not result.ok
    assert "LaTeX parse error" in result.errors[0]


def test_numeric_evaluate_plain(kernel):
    parsed = kernel.parse("pi()")
    eid = parsed.data["expr_id"]
    result = kernel.numeric_evaluate(eid, dps=30)
    assert result.ok
    assert result.data["value"].startswith("3.1415926535897932384626433832")
    assert result.trust.value == "numeric"


def test_numeric_evaluate_with_values(kernel):
    parsed = kernel.parse("x^2 + y")
    eid = parsed.data["expr_id"]
    result = kernel.numeric_evaluate(eid, values={"x": "3", "y": "1/3"}, dps=20)
    assert result.ok
    assert result.data["value"].startswith("9.3333333333333333333")


def test_numeric_evaluate_unbound_symbol(kernel):
    parsed = kernel.parse("x^2")
    result = kernel.numeric_evaluate(parsed.data["expr_id"])
    assert not result.ok
    assert "Unbound symbols" in result.errors[0]


def test_numeric_evaluate_dps_validation(kernel):
    parsed = kernel.parse("2")
    assert not kernel.numeric_evaluate(parsed.data["expr_id"], dps=1).ok
    assert not kernel.numeric_evaluate(parsed.data["expr_id"], dps=501).ok
