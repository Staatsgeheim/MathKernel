# =============================================================================
# MathKernel - MathIR v2: sets, membership, quantifiers, binders."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""MathIR v2: sets, membership, quantifiers, binders."""

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.engines import SymPyEngine, Z3Engine
from mathkernel.parser import parse_math
from mathkernel.rendering import render_expr


# --- parser / rendering --------------------------------------------------------

def test_parse_set_literal():
    ir = parse_math("{1, 2, 3}")
    assert ir.kind == "set" and len(ir.elements) == 3
    assert render_expr(ir) == "{1, 2, 3}"


def test_parse_empty_set_literal():
    ir = parse_math("{}")
    assert ir.kind == "set" and ir.elements == []


def test_parse_named_sets():
    assert parse_math("Integers()").name == "integers"
    assert parse_math("Reals()").name == "reals"
    assert parse_math("EmptySet()").name == "empty"


def test_parse_membership_and_ops():
    assert parse_math("in(x, Integers())").kind == "in"
    assert parse_math("union({1}, {2})").op == "union"
    assert parse_math("intersect({1}, {2})").op == "intersect"
    assert parse_math("difference({1}, {2})").op == "difference"
    assert parse_math("complement({1}, {1, 2})").op == "complement"


def test_parse_quantifiers_and_binders():
    q = parse_math("forall(x, Integers(), x + 0 = x)")
    assert (q.quantifier, q.variable) == ("forall", "x")
    e = parse_math("exists(y, {1, 2}, y > 1)")
    assert (e.quantifier, e.variable) == ("exists", "y")
    b = parse_math("sumover(k, {1, 2, 3}, k^2)")
    assert (b.op, b.variable) == ("sum", "k")
    p = parse_math("productover(k, {1, 2}, k)")
    assert p.op == "product"


def test_quantifier_requires_symbol_variable():
    with pytest.raises(ValueError):
        parse_math("forall(1, Integers(), 1 = 1)")


def test_render_roundtrip():
    for text in ["forall(x, Integers(), x + 0 = x)", "in(x, {1, 2})",
                 "sumover(k, {1, 2}, k)"]:
        assert render_expr(parse_math(text)) == render_expr(parse_math(render_expr(parse_math(text))))


# --- engine mappings ------------------------------------------------------------

def test_sympy_set_mappings():
    eng = SymPyEngine()
    assert eng.to_sympy(parse_math("union({1, 2}, {2, 3})")) == __import__("sympy").FiniteSet(1, 2, 3)
    assert eng.to_sympy(parse_math("sumover(k, {1, 2, 3}, k^2)")) == 14
    assert eng.to_sympy(parse_math("productover(k, {2, 3}, k)")) == 6
    assert bool(eng.to_sympy(parse_math("in(2, {1, 2, 3})"))) is True


def test_sympy_set_roundtrip():
    eng = SymPyEngine()
    ir = eng.from_sympy(eng.to_sympy(parse_math("union({1}, {2})")))
    assert ir.kind == "set" and len(ir.elements) == 2


def test_sympy_rejects_quantifier():
    eng = SymPyEngine()
    with pytest.raises(ValueError, match="quantifier"):
        eng.to_sympy(parse_math("forall(x, Integers(), x = x)"))


def test_z3_quantifier_validity():
    z3 = Z3Engine()
    if not z3.available:
        pytest.skip("z3-solver not installed")
    q = parse_math("forall(x, Integers(), x + 0 = x)")
    env = z3._env([q], {})
    solver = z3._new_solver()
    solver.add(z3.z3.Not(z3.to_z3(q, env)))
    assert solver.check() == z3.z3.unsat


def test_substitution_respects_bound_variables():
    from mathkernel.kernel import _substitute_ir
    from mathkernel.models import IntegerNode, SymbolNode
    q = parse_math("forall(x, Integers(), x + y = y)")
    out = _substitute_ir(q, {"x": IntegerNode(value="5"), "y": SymbolNode(name="z")})
    # bound x must not be substituted; free y must be
    assert render_expr(out) == render_expr(parse_math("forall(x, Integers(), x + z = z)"))


# --- kernel facade ----------------------------------------------------------------

def test_set_create_and_membership():
    k = MathKernel()
    s = k.set_create(elements=["1", "2", "3"])
    assert s.ok and s.trust == TrustLevel.EXACT
    assert k.set_membership("2", s.data["expr_id"]).data["member"] is True
    assert k.set_membership("7", s.data["expr_id"]).data["member"] is False


def test_set_create_validation():
    k = MathKernel()
    assert not k.set_create().ok
    assert not k.set_create(elements=["1"], name="reals").ok
    assert not k.set_create(name="ordinals").ok


def test_set_ops():
    k = MathKernel()
    a = k.set_create(elements=["1", "2"]).data["expr_id"]
    b = k.set_create(elements=["2", "3"]).data["expr_id"]
    u = k.set_op("union", [a, b])
    assert u.ok and u.data["display"] == "{1, 2, 3}"
    i = k.set_op("intersect", [a, b])
    assert i.ok and i.data["display"] == "{2}"
    d = k.set_op("difference", [a, b])
    assert d.ok and d.data["display"] == "{1}"


def test_set_op_numeric_elements_downgrade_trust():
    k = MathKernel()
    s = k.set_create(elements=["0.5", "1"])
    assert s.trust == TrustLevel.NUMERIC


def test_quantifier_check_valid():
    k = MathKernel()
    q = k.parse("forall(x, Integers(), x + 0 = x)").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "valid" and r.trust == TrustLevel.EXACT


def test_quantifier_check_invalid_with_countermodel():
    k = MathKernel()
    q = k.parse("forall(x, Integers(), x > 0)").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "invalid"
    assert "x" in r.data["countermodel"]


def test_quantifier_check_exists_witness():
    k = MathKernel()
    q = k.parse("exists(x, Integers(), x^2 = 4)").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "valid" and "x" in r.data["witness"]


def test_quantifier_check_rejects_non_quantifier():
    k = MathKernel()
    eid = k.parse("x + 1").data["expr_id"]
    assert not k.quantifier_check(eid).ok
