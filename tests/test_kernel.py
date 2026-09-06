# =============================================================================
# MathKernel - test kernel
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest
from mathkernel import MathKernel, ambiguity_diagnostics, parse_math
from mathkernel.engines import LeanEngine


def test_safe_parser_builds_mathir():
    ir=parse_math("(x+1)^2 = x^2 + 2*x + 1")
    assert ir.kind == "eq"


def test_unknown_function_rejected():
    with pytest.raises(ValueError):
        parse_math("__import__(x)")


def test_ambiguity_diagnostic_refuses_division_implicit_mul():
    d=ambiguity_diagnostics("1/2x")
    assert d and d[0].severity == "error"
    k=MathKernel(); r=k.parse("1/2x")
    assert not r.ok and r.data["ambiguities"]


def test_parse_analyze_simplify_and_semantics():
    k=MathKernel(); p=k.parse("(x+1)^2"); eid=p.data["expr_id"]
    a=k.analyze(eid)
    assert "x" in a.data["symbols"]
    assert {e["name"] for e in a.data["engine_manifest"]} == {"sympy", "z3", "lean", "mpmath_interval", "integer_exact"}
    s=k.simplify(eid, "expand")
    assert "x ^ 2" in s.data["result"]
    assert s.derivation[0].trust.value == "symbolic"
    assert s.data["result_expr_id"] is not None


def test_structured_quadratic_solution_set_respects_context_domain():
    k=MathKernel(); c=k.create_context({"x":"real"}, [])
    p=k.parse("x^2 - 3*x + 2 = 0")
    r=k.solve(p.data["expr_id"], "x", c.context_id)
    ss=r.data["solution_set"]
    assert r.ok and ss["domain"] == "real" and ss["kind"] == "finite"
    assert ss["values"] == ["1", "2"]


def test_context_inference_from_sign_assumption():
    k=MathKernel(); c=k.create_context({}, ["x > 0"])
    assert c.domains["x"] == "real"
    assert "positive" in c.symbol_properties["x"]
    assert "nonzero" in c.symbol_properties["x"]
    a=k.parse("x+1")
    info=k.analyze(a.data["expr_id"], c.context_id)
    assert info.data["semantic_symbols"]["x"]["domain"] == "real"


def test_equivalence_verified_and_evidence_is_visible():
    k=MathKernel(); r=k.prove_equivalence("(x+y)^2", "x^2 + 2*x*y + y^2")
    assert r.status == "verified"
    assert any(e.engine == "sympy" for e in r.evidence)
    assert "lean_certificate" in r.data and r.data["lean_tactic"] == "ring"


def test_lean_tactic_selection_ring_norm_num_and_linarith():
    e=LeanEngine()
    l=parse_math("(x+y)^2"); r=parse_math("x^2 + 2*x*y + y^2")
    script,t=e.build_certificate(l,r,[])
    assert t == "ring" and "  ring" in script
    _,t2=e.build_certificate(parse_math("2+2"),parse_math("4"),[])
    assert t2 == "norm_num"
    a=parse_math("x > 0")
    _,t3=e.build_certificate(parse_math("x+1"),parse_math("1+x"),[a])
    assert t3 == "linarith"


def test_derivation_is_a_graph_with_parent_edges():
    k=MathKernel(); p=k.parse("(x+1)^2"); s=k.simplify(p.data["expr_id"], "expand")
    sid=s.derivation[0].step_id
    trace=k.derivation_trace(sid)
    g=trace.data["graph"]
    assert len(g["nodes"]) == 2
    assert len(g["edges"]) == 1
    assert g["edges"][0]["to"] == sid
    assert trace.evidence_bundle.computation[0].engine == "sympy"
    assert trace.claim_evidence["result"].computation


def test_interval_evaluation_returns_enclosure_and_trust():
    k=MathKernel(); p=k.parse("x^2 + 1")
    r=k.interval_evaluate(p.data["expr_id"], {"x":[2,3]}, 30)
    assert r.ok and r.trust.value == "interval_certified"
    assert "enclosure" in r.data


def test_context_assumptions_are_structured():
    k=MathKernel(); c=k.create_context({"x":"real"}, ["x > 0"])
    assert c.domains["x"] == "real"
    assert c.assumptions[0].expression.kind == "gt"


def test_capability_router_reports_optional_backends():
    k=MathKernel(); manifest={e["name"]:e for e in k.capabilities()["engines"]}
    assert manifest["sympy"]["available"] is True
    assert "counterexample" in manifest["z3"]["capabilities"]
    assert "formal_linarith" in manifest["lean"]["capabilities"]
    assert "certified_enclosure" in manifest["mpmath_interval"]["capabilities"]


def test_counterexample_behaviour_when_z3_missing_or_present():
    k=MathKernel(); r=k.counterexample("x", "x+1")
    if k.z3.available:
        assert r.status == "refuted" and r.data["counterexample"] is not None
    else:
        assert r.status == "error" and "z3-solver" in r.errors[0]



def test_numeric_ir_distinguishes_integer_rational_and_real():
    assert parse_math("42").kind == "integer"
    r = parse_math("6/8")
    assert r.kind == "rational" and r.numerator == "3" and r.denominator == "4"
    assert parse_math("1.25").kind == "real"


def test_big_integer_parser_and_exact_gcd_beyond_python_decimal_limit():
    k = MathKernel()
    a = "9" * 5000
    b = "3" * 5000
    parsed = k.parse(a)
    assert parsed.ok and parsed.data["ir"]["kind"] == "integer"
    result = k.integer_compute("gcd", [a, b], max_output_digits=6000)
    assert result.ok and result.trust.value == "exact"
    assert result.data["result"]["decimal_digits"] >= 4999


def test_big_integer_symbolic_simplification_does_not_hit_python_digit_limit():
    k = MathKernel()
    huge = "9" * 5000
    p = k.parse(huge + "+1")
    r = k.simplify(p.data["expr_id"])
    assert r.ok
    assert len(r.data["result"]) == 5001
    assert r.data["result"].startswith("1")


def test_integer_number_theory_operations():
    k = MathKernel()
    assert k.integer_compute("pow_mod", ["7", "560"], modulus="561").data["result"]["value"] == "1"
    assert k.integer_compute("mod_inverse", ["3"], modulus="11").data["result"]["value"] == "4"
    f = k.integer_compute("factor", ["360"]).data
    assert f["factors"] == {"2": 3, "3": 2, "5": 1} and f["complete"] is True


def test_problem_planner_decomposes_relation_into_obligations():
    k = MathKernel()
    c = k.create_context({"x": "real"}, [])
    p = k.parse("x^2 - 2 = 0")
    r = k.plan(p.data["expr_id"], c.context_id)
    plan = r.data["plan"]
    assert plan["classification"] == "algebraic_relation"
    assert {o["kind"] for o in plan["obligations"]} >= {"domain", "symbolic", "smt", "formal"}
    assert plan["formal_verification_possible"] is True



def test_complex_interval_and_algebraic_numeric_ir_constructors():
    c = parse_math("complex(2,3)")
    assert c.kind == "complex" and c.real.kind == "integer" and c.imag.kind == "integer"
    i = parse_math("interval(1,2)")
    assert i.kind == "interval" and i.lower.kind == "integer"
    a = parse_math("rootof(x^2-2,0)")
    assert a.kind == "algebraic" and a.root_index == 0


def test_v05_plan_is_executable_dependency_dag():
    k=MathKernel(); c=k.create_context({'x':'real'},[]); p=k.parse('x^2-3*x+2=0')
    r=k.plan(p.data['expr_id'],c.context_id)
    assert r.ok and r.data['plan_id'].startswith('plan_')
    obligations=r.data['plan']['obligations']
    by_action={o['action']:o for o in obligations}
    assert by_action['solve_relation']['depends_on']==[by_action['validate_context']['obligation_id']]
    assert by_action['verify_solution_candidates']['depends_on']==[by_action['solve_relation']['obligation_id']]
    assert set(by_action['check_solution_completeness']['depends_on'])=={
        by_action['solve_relation']['obligation_id'],by_action['verify_solution_candidates']['obligation_id']}
    routes = r.data['plan']['capability_routes']
    assert routes[by_action['solve_relation']['obligation_id']][0] == 'sympy'
    assert 'z3' in routes[by_action['check_solution_completeness']['obligation_id']]
    assert by_action['solve_relation']['capability_ref'] == 'solve'
    assert by_action['solve_relation']['resolved_handler'] == 'router:sympy'


def test_v05_execute_plan_verifies_candidates_and_builds_unified_graph():
    k=MathKernel(); c=k.create_context({'x':'real'},[]); p=k.parse('x^2-3*x+2=0')
    plan=k.plan(p.data['expr_id'],c.context_id)
    r=k.execute_plan(plan.data['plan_id'],formal=False)
    ex=r.data['execution']; by_action={o['action']:o for o in ex['obligations']}
    assert r.status=='verified' and r.trust.value in {'symbolic','exact'}
    assert by_action['solve_relation']['result']['solution_set']['values']==['1','2']
    assert by_action['verify_solution_candidates']['state']=='verified'
    assert by_action['verify_solution_candidates']['result']['all_candidates_valid'] is True
    assert any(e['type']=='depends_on' for e in ex['graph']['edges'])
    assert any(n['type']=='derivation_step' for n in ex['graph']['nodes'])
    verify_id = by_action['verify_solution_candidates']['obligation_id']
    assert ex['claim_evidence'][verify_id]['proof'][0]['verified'] is True
    assert r.claim_evidence[verify_id].proof[0].verified is True
    assert all(step.evidence_bundle.computation for step in r.derivation)


def test_v05_context_assumptions_propagate_and_filter_candidates():
    k=MathKernel(); c=k.create_context({'x':'real'},['x > 0']); p=k.parse('x^2=1')
    r=k.reason(p.data['expr_id'],c.context_id,formal=False)
    ex=r.data['execution']; solve=next(o for o in ex['obligations'] if o['action']=='solve_relation')
    assert r.assumptions_used==['x > 0']
    assert solve['result']['solution_set']['values']==['1']
    assert solve['result']['excluded_candidates'][0]['candidate']=='-1'


def test_v05_missing_domains_survive_as_explicit_side_conditions():
    k=MathKernel(); p=k.parse('x^2=1')
    r=k.reason(p.data['expr_id'],formal=False)
    assert 'Domain for x is unspecified' in r.side_conditions
    assert any('unresolved domain' in w.lower() for w in r.warnings)


def test_v05_plan_and_execution_are_retrievable():
    k=MathKernel(); p=k.parse('2+2'); plan=k.plan(p.data['expr_id'])
    assert k.plan_get(plan.data['plan_id']).data['expr_id']==p.data['expr_id']
    r=k.execute_plan(plan.data['plan_id'],formal=False)
    execution_id=r.data['execution']['execution_id']
    fetched=k.execution_get(execution_id)
    assert fetched.ok and fetched.data['execution']['execution_id']==execution_id
    assert fetched.data['execution']['summary']['normalized_expression']=='4'


def test_v05_independent_verification_detects_bad_symbolic_candidate(monkeypatch):
    import sympy as sp
    k=MathKernel(); c=k.create_context({'x':'real'},[]); p=k.parse('x^2=1')
    monkeypatch.setattr(k.sympy,'solve',lambda node,variable,domain: sp.FiniteSet(999))
    r=k.reason(p.data['expr_id'],c.context_id,formal=False)
    ex=r.data['execution']; verify=next(o for o in ex['obligations'] if o['action']=='verify_solution_candidates')
    assert r.status=='refuted'
    assert verify['state']=='refuted'
    assert ex['conflicts'][0]['type']=='invalid_candidate'


def test_v05_bare_equation_is_not_planned_as_universal_theorem():
    k=MathKernel(); p=k.parse('x^2-2=0'); plan=k.plan(p.data['expr_id']).data['plan']
    actions={o['action'] for o in plan['obligations']}
    assert 'solve_relation' in actions
    assert 'formalize_solution_soundness' in actions
    assert all('prove the equation' not in o['statement'].lower() for o in plan['obligations'])
