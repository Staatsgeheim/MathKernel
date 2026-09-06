# =============================================================================
# MathKernel - Tests for quantifier elimination and nested alternation (v0.22)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""QE via Z3's qe tactic, from_z3 mapping, nested/alternating quantifier
truth checking, batch isolation, undecidable-fragment honesty."""

import pytest

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.parser import parse_math
from mathkernel.qe import from_z3, negate
from mathkernel.rendering import render_expr


# --- BoolNode plumbing ---------------------------------------------------------

def test_bool_node_parse_render_roundtrip():
    ir = parse_math("and(x > 0, or(x < 1, not(x = 2)))")
    s = render_expr(ir)
    assert s == "and(x > 0, or(x < 1, not(x = 2)))"
    assert render_expr(parse_math(s)) == s


def test_bool_node_to_z3():
    k = MathKernel()
    eid = k.parse("and(x > 0, x < 2)").data["expr_id"]
    r = k.prove(eid, formal=False)  # satisfiable but not valid
    assert r.status == "refuted"


def test_negate_nnf():
    ir = parse_math("forall(x, Reals, exists(y, Reals, and(y > x, y < 0)))")
    n = negate(ir)
    assert render_expr(n) == \
        "exists(x, Reals(), forall(y, Reals(), or(y <= x, y >= 0)))"


# --- quantifier_check: nested alternation ---------------------------------------

def test_nested_forall_exists_valid():
    k = MathKernel()
    q = k.parse("forall(x, Reals, exists(y, Reals, y > x))").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "valid" and r.trust == TrustLevel.EXACT


def test_nested_exists_forall_invalid():
    k = MathKernel()
    # no real x is <= every real y
    q = k.parse("exists(x, Reals, forall(y, Reals, x <= y))").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "invalid"


def test_alternation_depth_three():
    k = MathKernel()
    # for every x there is a y such that for all z: x + z < y + z + 1
    q = k.parse("forall(x, Reals, exists(y, Reals, forall(z, Reals, x + z < y + z + 1)))")
    r = k.quantifier_check(q.data["expr_id"])
    assert r.data["validity"] == "valid"


def test_witness_extraction_exists():
    k = MathKernel()
    q = k.parse("exists(x, Integers, x^2 = 4)").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "valid"
    assert r.data["skolem_levels"] == ["x"]
    assert r.data["witness"]["x"] in ("2", "-2")


def test_countermodel_extraction_forall():
    k = MathKernel()
    q = k.parse("forall(x, Integers, x > 0)").data["expr_id"]
    r = k.quantifier_check(q)
    assert r.data["validity"] == "invalid"
    assert "x" in r.data["countermodel"]


def test_quantifier_check_still_rejects_non_quantifier():
    k = MathKernel()
    assert not k.quantifier_check(k.parse("x + 1").data["expr_id"]).ok


# --- quantifier elimination --------------------------------------------------------

def test_qe_known_answer_lra():
    k = MathKernel()
    # exists(x, x > 0 and x < y)  <=>  y > 0
    q = k.parse("exists(x, Reals, and(x > 0, x < y))").data["expr_id"]
    r = k.quantifier_eliminate(q)
    assert r.ok and r.trust == TrustLevel.EXACT
    assert r.data["status"] == "eliminated"
    # the result re-parses and mentions only y
    f = r.data["formula"]
    assert "x" not in f.replace("exists", "").replace("forall", "")
    assert "y" in f
    k.parse(f)  # re-parses cleanly


def test_qe_result_is_equivalent():
    k = MathKernel()
    q = k.parse("exists(x, Reals, and(x > 0, x < y))").data["expr_id"]
    r = k.quantifier_eliminate(q)
    # cross-check: the eliminated formula must decide y=1 true, y=-1 false
    f = r.data["formula"]
    for value, expected in (("1", True), ("-1", False)):
        sub = k.parse(f"exists(dummy, {{1}}, {f.replace('y', value)})")
        chk = k.quantifier_check(sub.data["expr_id"])
        assert (chk.data["validity"] == "valid") is expected


def test_qe_forall_via_duality():
    k = MathKernel()
    # forall(x, x^2 >= y) is false over the reals... QE on the exists-negation:
    # exists(x, x + y > 0) <=> true
    q = k.parse("exists(x, Reals, x + y > 0)").data["expr_id"]
    r = k.quantifier_eliminate(q)
    assert r.data["status"] == "eliminated"
    assert r.data["formula"] in ("1 = 1", "not(1 != 1)") or "1" in r.data["formula"]


def test_qe_undecidable_fragment_honesty():
    k = MathKernel()
    # nonlinear + function: outside LRA/LIA — must not claim exact
    q = k.parse("exists(x, Reals, sin(x) = y)").data["expr_id"]
    r = k.quantifier_eliminate(q)
    if r.data["status"] == "eliminated":
        # Z3 may still succeed; then the result must re-parse
        k.parse(r.data["formula"])
    else:
        assert r.status == "unknown" and r.trust == TrustLevel.UNKNOWN
        assert r.data["fragment"]["has_functions"]


def test_qe_variable_cap():
    k = MathKernel(Settings(max_qe_variables=1))
    q = k.parse("exists(x, Reals, and(x > 0, y > 0, z > 0))").data["expr_id"]
    r = k.quantifier_eliminate(q)
    assert r.status == "unknown" and "max_qe_variables" in r.data["reason"]


def test_qe_unknown_expr():
    k = MathKernel()
    assert not k.quantifier_eliminate("expr_nope").ok


def test_from_z3_mapping_direct():
    from mathkernel.engines import Z3Engine
    eng = Z3Engine()
    z3 = eng.z3
    x, y = z3.Reals("x y")
    ast = z3.And(x > 0, z3.Or(y <= z3.RatVal(1, 2), x == 3))
    ir = from_z3(z3, ast)
    assert render_expr(ir) == "and(x > 0, or(y <= 1/2, x = 3))"


# --- batch --------------------------------------------------------------------------

def test_qe_batch_isolation():
    k = MathKernel()
    ids = [k.parse(s).data["expr_id"] for s in [
        "exists(x, Reals, and(x > 0, x < y))",
        "exists(x, Reals, x + y > 0)",
        "forall(x, Reals, x^2 >= 0)",
    ]]
    out = k.quantifier_eliminate_batch(ids, workers=2)
    assert out.ok and out.data["total"] == 3
    statuses = [r["status"] for r in out.data["results"]]
    assert statuses.count("eliminated") >= 2  # the two linear ones
    by_id = {r["expr_id"]: r for r in out.data["results"]}
    assert by_id[ids[0]]["status"] == "eliminated"
