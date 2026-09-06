# =============================================================================
# MathKernel - test calculus
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import sympy as sp
import pytest

from mathkernel import MathKernel


@pytest.fixture
def kernel():
    return MathKernel()


def _parse(kernel, text):
    result = kernel.parse(text)
    assert result.ok, result.errors
    return result.data["expr_id"]


def _result_sympy(kernel, result):
    eid = result.data["result_expr_id"]
    assert eid is not None, result.warnings
    return kernel.sympy.to_sympy(kernel.expressions[eid])


def test_differentiate_polynomial(kernel):
    eid = _parse(kernel, "x^3 + 2*x")
    result = kernel.differentiate(eid, "x")
    assert result.ok
    assert sp.simplify(_result_sympy(kernel, result) - (3*sp.Symbol("x")**2 + 2)) == 0


def test_differentiate_higher_order(kernel):
    eid = _parse(kernel, "x^4")
    result = kernel.differentiate(eid, "x", order=2)
    assert result.ok
    assert sp.simplify(_result_sympy(kernel, result) - 12*sp.Symbol("x")**2) == 0


def test_differentiate_trig_chain(kernel):
    eid = _parse(kernel, "sin(x^2)")
    result = kernel.differentiate(eid, "x")
    assert result.ok
    x = sp.Symbol("x")
    assert sp.simplify(_result_sympy(kernel, result) - 2*x*sp.cos(x**2)) == 0


def test_integrate_indefinite_roundtrip(kernel):
    eid = _parse(kernel, "2*x + cos(x)")
    result = kernel.integrate(eid, "x")
    assert result.ok
    assert "additive constant" in result.side_conditions[0]
    x = sp.Symbol("x")
    assert sp.simplify(sp.diff(_result_sympy(kernel, result), x) - (2*x + sp.cos(x))) == 0


def test_integrate_definite(kernel):
    eid = _parse(kernel, "x")
    result = kernel.integrate(eid, "x", lower="0", upper="1")
    assert result.ok
    assert not result.side_conditions
    assert _result_sympy(kernel, result) == sp.Rational(1, 2)


def test_integrate_definite_with_infinity(kernel):
    eid = _parse(kernel, "exp(-x)")
    result = kernel.integrate(eid, "x", lower="0", upper="oo")
    assert result.ok
    assert _result_sympy(kernel, result) == sp.Integer(1)


def test_integrate_requires_both_bounds(kernel):
    eid = _parse(kernel, "x")
    result = kernel.integrate(eid, "x", lower="0")
    assert not result.ok


def test_limit_sin_over_x(kernel):
    eid = _parse(kernel, "sin(x)/x")
    result = kernel.limit(eid, "x", "0")
    assert result.ok
    assert _result_sympy(kernel, result) == sp.Integer(1)


def test_limit_at_infinity(kernel):
    eid = _parse(kernel, "(1 + 1/x)^x")
    result = kernel.limit(eid, "x", "oo")
    assert result.ok
    assert sp.simplify(_result_sympy(kernel, result) - sp.E) == 0


def test_limit_direction_validation(kernel):
    eid = _parse(kernel, "1/x")
    assert not kernel.limit(eid, "x", "0", direction="sideways").ok


def test_series_exp(kernel):
    eid = _parse(kernel, "exp(x)")
    result = kernel.series(eid, "x", point="0", order=4)
    assert result.ok
    assert result.side_conditions and "truncated" in result.side_conditions[0]
    x = sp.Symbol("x")
    expected = 1 + x + x**2/2 + x**3/6
    assert sp.expand(_result_sympy(kernel, result) - expected) == 0


def test_summation_finite_symbolic(kernel):
    eid = _parse(kernel, "k")
    result = kernel.summation(eid, "k", "1", "n")
    assert result.ok
    n = sp.Symbol("n")
    assert sp.simplify(_result_sympy(kernel, result) - n*(n + 1)/2) == 0


def test_summation_infinite_geometric(kernel):
    eid = _parse(kernel, "(1/2)^k")
    result = kernel.summation(eid, "k", "0", "oo")
    assert result.ok
    assert _result_sympy(kernel, result) == sp.Integer(2)


def test_product_factorial(kernel):
    eid = _parse(kernel, "k")
    result = kernel.product(eid, "k", "1", "n")
    assert result.ok
    n = sp.Symbol("n")
    assert sp.simplify(_result_sympy(kernel, result) - sp.factorial(n)) == 0


def test_solve_system_linear(kernel):
    e1 = _parse(kernel, "x + y = 3")
    e2 = _parse(kernel, "x - y = 1")
    result = kernel.solve_system([e1, e2], ["x", "y"])
    assert result.ok
    assert result.data["solution_count"] == 1
    sol = result.data["solutions"][0]
    assert sol["x"]["display"] == "2"
    assert sol["y"]["display"] == "1"


def test_solve_system_quadratic_two_solutions(kernel):
    e1 = _parse(kernel, "x^2 + y^2 = 25")
    e2 = _parse(kernel, "x - y = 1")
    result = kernel.solve_system([e1, e2], ["x", "y"])
    assert result.ok
    assert result.data["solution_count"] == 2
    pairs = {(s["x"]["display"], s["y"]["display"]) for s in result.data["solutions"]}
    assert pairs == {("4", "3"), ("-3", "-4")}


def test_solve_system_rejects_empty(kernel):
    assert not kernel.solve_system([], ["x"]).ok
    eid = _parse(kernel, "x = 1")
    assert not kernel.solve_system([eid], []).ok
    assert not kernel.solve_system(["expr_nope"], ["x"]).ok


def test_assumption_aware_simplify_positive(kernel):
    eid = _parse(kernel, "sqrt(x^2)")
    plain = kernel.simplify(eid)
    assert plain.ok
    ctx = kernel.create_context(domains={"x": "positive"})
    assumed = kernel.simplify(eid, context_id=ctx.context_id)
    assert assumed.ok
    assert assumed.data["result"] == "x"
    # Without the assumption the result must not collapse to x.
    assert plain.data["result"] != "x"


def test_assumption_aware_solve_integer_domain(kernel):
    eid = _parse(kernel, "x^2 = 4")
    ctx = kernel.create_context(domains={"x": "integer"})
    result = kernel.solve(eid, "x", context_id=ctx.context_id)
    assert result.ok
    assert result.data["solution_set"]["kind"] == "finite"
    assert set(result.data["solution_set"]["values"]) == {"-2", "2"}


def test_assumption_aware_equivalence(kernel):
    ctx = kernel.create_context(domains={"x": "positive"})
    result = kernel.prove_equivalence("sqrt(x^2)", "x", context_id=ctx.context_id, formal=False)
    assert result.ok
    assert result.status == "verified"


def test_special_function_roundtrip(kernel):
    eid = _parse(kernel, "factorial(n) + gamma(x) + binomial(n, k)")
    expr = kernel.sympy.to_sympy(kernel.expressions[eid])
    back = kernel.sympy.from_sympy(expr)
    assert back.kind == "add"
    names = {a.name for a in back.args}
    assert names == {"factorial", "gamma", "binomial"}


def test_derivation_recorded(kernel):
    eid = _parse(kernel, "x^2")
    result = kernel.differentiate(eid, "x")
    assert result.ok
    step = result.derivation[0]
    assert step.operation == "differentiate"
    assert step.output_expr_id == result.data["result_expr_id"]
    graph = kernel.derivation_trace(step.step_id)
    assert graph.ok
