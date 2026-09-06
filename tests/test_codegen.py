# =============================================================================
# MathKernel - test codegen
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

from mathkernel import MathKernel, Settings


def test_get_expression_roundtrip():
    k = MathKernel()
    p = k.parse("x^2 + 1")
    eid = p.data["expr_id"]
    g = k.get_expression(eid)
    assert g.ok and g.data["ir"]["kind"] == "add"
    assert g.data["source"] == "x^2 + 1"
    assert not k.get_expression("expr_nope").ok


def test_substitute_produces_new_expression():
    k = MathKernel()
    p = k.parse("x^2 + y")
    s = k.substitute(p.data["expr_id"], {"x": "2"})
    assert s.ok and s.data["expr_id"] != p.data["expr_id"]
    simplified = k.simplify(s.data["expr_id"])
    assert "4" in simplified.data["result"] and "y" in simplified.data["result"]
    # original expression is untouched
    assert k.get_expression(p.data["expr_id"]).data["source"] == "x^2 + y"


def test_substitute_rejects_bad_replacement():
    k = MathKernel()
    p = k.parse("x + 1")
    r = k.substitute(p.data["expr_id"], {"x": "foo("})
    assert not r.ok


def test_infer_structure_quadratic_equation():
    k = MathKernel()
    p = k.parse("a*x^2 + b*x + c = 0")
    r = k.infer_structure(p.data["expr_id"])
    assert r.ok
    caps = set(r.data["required_capabilities"])
    assert {"Add", "Mul", "PowInteger", "Eq"} <= caps
    assert r.data["suggested_structure"] in {"semiring", "commutative_ring"}


def test_infer_structure_division_constraint():
    k = MathKernel()
    p = k.parse("1/x + sqrt(y)")
    r = k.infer_structure(p.data["expr_id"])
    assert r.data["suggested_structure"] == "field_with_sqrt"
    assert any(c.startswith("x != 0") for c in r.data["constraints"])
    assert any(c.startswith("y >= 0") for c in r.data["constraints"])


def test_codegen_typescript_evaluate():
    k = MathKernel()
    p = k.parse("a*x^2 + b*x + c")
    r = k.codegen(p.data["expr_id"], language="typescript", target="evaluate")
    assert r.ok
    content = r.data["files"][0]["content"]
    assert "export function evaluate_expression<T>" in content
    assert "F.powInt(x, 2)" in content
    assert "fromNumber" in content
    assert r.data["suggested_structure"] == "commutative_ring"
    assert {"Add", "Mul", "Neg", "PowInteger", "FromNumber"} <= set(r.data["required_capabilities"])


def test_codegen_python_solve_quadratic_matches_design_shape():
    k = MathKernel()
    c = k.create_context({"a": "real", "b": "real", "c": "real"}, ["a != 0"])
    p = k.parse("a*x^2 + b*x + c = 0")
    r = k.codegen(p.data["expr_id"], language="python", target="solve", variable="x",
                  context_id=c.context_id)
    assert r.ok
    assert r.data["function_name"] == "solve_for_x"
    assert r.data["params"] == ["a", "b", "c"]
    assert r.data["suggested_structure"] == "field_with_sqrt"
    assert any(c.startswith("a != 0") for c in r.data["constraints"])
    content = r.data["files"][0]["content"]
    assert "class FieldWithSqrt" in content
    assert "def solve_for_x(a, b, c, F: FieldWithSqrt[T])" in content
    assert "F.sqrt" in content


def test_codegen_rust_evaluate():
    k = MathKernel()
    p = k.parse("x^2 + 2*x + 1")
    r = k.codegen(p.data["expr_id"], language="rust", target="evaluate")
    assert r.ok
    content = r.data["files"][0]["content"]
    assert "pub fn evaluate_expression<T:" in content
    assert "T::from_number(2)" in content
    assert ".pow_int(2)" in content


def test_codegen_constraint_target_python():
    k = MathKernel()
    p = k.parse("x^2 - 2 >= 0")
    r = k.codegen(p.data["expr_id"], language="python", target="constraint")
    assert r.ok
    assert "def check_constraint(x, F:" in r.data["files"][0]["content"]
    assert "greater_eq" in r.data["files"][0]["content"]


def test_codegen_rejects_decimal_literals_in_generic_mode():
    k = MathKernel()
    p = k.parse("1.5*x")
    r = k.codegen(p.data["expr_id"], language="python", target="evaluate")
    assert not r.ok and "rational" in r.errors[0].lower()


def test_codegen_rejects_unknown_language_and_bad_target():
    k = MathKernel()
    p = k.parse("x + 1")
    assert not k.codegen(p.data["expr_id"], language="cobol").ok
    assert not k.codegen(p.data["expr_id"], language="python", target="frobnicate").ok
    assert not k.codegen(p.data["expr_id"], language="python", target="solve").ok  # needs equality + variable


def test_verify_code_typecheck_and_symbolic_roundtrip():
    k = MathKernel()
    p = k.parse("a*x^2 + b*x + c = 0")
    r = k.codegen(p.data["expr_id"], language="python", target="solve", variable="x")
    v = k.verify_code(r.data["artifact_id"], ["typecheck", "symbolic_roundtrip"])
    assert v.ok
    checks = {c["check"]: c for c in v.data["checks"]}
    assert checks["typecheck"]["status"] == "passed"
    assert checks["symbolic_roundtrip"]["status"] == "passed", checks["symbolic_roundtrip"]["detail"]
    assert v.data["all_passed"] is True


def test_verify_code_roundtrip_evaluate():
    k = MathKernel()
    p = k.parse("(x + y)^2")
    r = k.codegen(p.data["expr_id"], language="python", target="evaluate")
    v = k.verify_code(r.data["artifact_id"], ["symbolic_roundtrip"])
    assert v.ok and v.data["checks"][0]["status"] == "passed"


def test_verify_code_unknown_artifact():
    k = MathKernel()
    assert not k.verify_code("artifact_nope").ok


def test_execute_code_disabled_by_default():
    k = MathKernel()
    p = k.parse("x + 1")
    r = k.codegen(p.data["expr_id"], language="python", target="evaluate")
    e = k.execute_code(r.data["artifact_id"], {"x": 1})
    assert not e.ok and "disabled" in e.errors[0].lower()


def test_execute_code_enabled_runs_sandboxed():
    k = MathKernel(Settings(enable_execution=True))
    p = k.parse("a*x^2 + b*x + c = 0")
    r = k.codegen(p.data["expr_id"], language="python", target="solve", variable="x")
    e = k.execute_code(r.data["artifact_id"], {"a": 1, "b": -3, "c": 2})
    assert e.ok, e.errors or e.data.get("stderr")
    roots = sorted(e.data["result"])
    assert roots == pytest.approx([1.0, 2.0])
    assert e.trust.value == "numeric"


def test_execute_code_validates_inputs():
    k = MathKernel(Settings(enable_execution=True))
    p = k.parse("x + 1")
    r = k.codegen(p.data["expr_id"], language="python", target="evaluate")
    aid = r.data["artifact_id"]
    assert not k.execute_code(aid, {}).ok  # missing x
    assert not k.execute_code(aid, {"x": 1, "evil": 2}).ok  # unknown input
    assert not k.execute_code(aid, {"x": "1"}).ok  # non-numeric input


def test_simplify_extra_modes():
    k = MathKernel()
    p = k.parse("(x^2 - 1)/(x - 1)")
    r = k.simplify(p.data["expr_id"], "cancel")
    assert r.ok and r.data["result"].strip("()") in {"x + 1", "1 + x"}
    p2 = k.parse("sin(x)^2 + cos(x)^2")
    r2 = k.simplify(p2.data["expr_id"], "trig")
    assert r2.ok and r2.data["result"] == "1"
    bad = k.simplify(p.data["expr_id"], "frobnicate")
    assert not bad.ok


def test_capabilities_include_codegen_and_limits():
    k = MathKernel()
    caps = k.capabilities()
    assert "codegen" in caps["operations"] and "substitute" in caps["operations"]
    assert caps["codegen"]["languages"] == ["python", "rust", "typescript"]
    assert caps["limits"]["enable_execution"] is False
    assert caps["security"]["code_execution_enabled"] is False


def test_parse_enforces_input_length_limit():
    k = MathKernel(Settings(max_input_length=10))
    r = k.parse("x^2 + 2*x + 1")
    assert not r.ok and "too long" in r.errors[0].lower()


def test_reason_with_solve_for_parametric_quadratic():
    k = MathKernel()
    c = k.create_context({"x": "real", "a": "real", "b": "real", "c": "real"}, ["a != 0"])
    p = k.parse("a*x^2 + b*x + c = 0")
    r = k.reason(p.data["expr_id"], c.context_id, formal=False, solve_for="x")
    ex = r.data["execution"]
    solve = next(o for o in ex["obligations"] if o["action"] == "solve_relation")
    assert solve["state"] == "succeeded"
    assert solve["result"]["solution_set"]["kind"] == "finite"
    assert len(solve["result"]["solution_set"]["values"]) == 2
    # complex fallback must be disclosed as a side condition
    assert any("complex field" in c for c in r.side_conditions)


def test_plan_flags_unspecified_solve_variable():
    k = MathKernel()
    p = k.parse("a*x + b = 0")
    r = k.plan(p.data["expr_id"])
    assert any("Solve variable is unspecified" in m for m in r.data["plan"]["missing_assumptions"])
    r2 = k.plan(p.data["expr_id"], solve_for="x")
    solve = next(o for o in r2.data["plan"]["obligations"] if o["action"] == "solve_relation")
    assert solve["parameters"]["variables"] == ["x"]


def test_server_registers_new_tools():
    from mathkernel_mcp.server import mcp
    assert mcp is not None
