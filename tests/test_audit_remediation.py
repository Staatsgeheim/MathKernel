# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Regression contracts reproduced from the dev27 audit on the dev28 baseline.

Test transitions as well as individual operations: persistence must not alter
mathematical meaning, and display/retrieval must not invent stronger evidence.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time

import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, Settings, TrustLevel, complete_result
from mathkernel.engines import run_with_timeout, SolverTimeoutError
from mathkernel.output_policy import wire_bytes
from mathkernel.composite_relation_inference import composite_detection_bounds
from mathkernel.information_geometry import score_subspace_geometry, subspace_minimax_bounds


@pytest.mark.parametrize("text,expected", [
    ("-2^2", -4), ("-2**2", -4), ("(-2)^2", 4), ("-(2^2)", -4),
    ("2^3^2", 512), ("2**3**2", 512), ("2^-2", sp.Rational(1,4)),
    ("2^-2^2", sp.Rational(1,16)), ("-2^-2", sp.Rational(-1,4)),
    ("2*-3^2", -18), ("2/-2^2", sp.Rational(-1,2)), ("--2^2", 4),
])
def test_conventional_power_and_unary_precedence(text, expected):
    kernel = MathKernel()
    handle = kernel.expression(text).simplify()
    ir, error = kernel._get_expr(handle.expression_id, "test")
    assert error is None
    assert kernel.sympy.to_sympy(ir) == expected


def test_symbolic_minus_preserved():
    kernel = MathKernel()
    handle = kernel.expression("-x^2").simplify()
    ir, _ = kernel._get_expr(handle.expression_id, "test")
    assert sp.expand(kernel.sympy.to_sympy(ir) + sp.Symbol("x")**2) == 0


def test_decimal_substitution_is_numeric_even_after_cancellation():
    kernel = MathKernel()
    original = kernel.expression("x-x")
    replaced = original.substitute({"x": "0.1"}).simplify().simplify()
    for result in (replaced.last_result, replaced.result):
        assert result.trust == TrustLevel.NUMERIC
        assert result.semantic_status.value != "verified_exact"
        assert result.evidence_bundle.conservative_trust() == "numeric"
    assert "x" in original.display


def test_ancestry_survives_restart_and_trace(tmp_path):
    settings = Settings(store_path=str(tmp_path / "ancestry.sqlite"))
    kernel = MathKernel(settings)
    source = kernel.expression("0.5*x-0.5*x")
    first = source.simplify()
    second = first.simplify()
    before = second.last_result
    restarted = MathKernel(settings)
    for eid in (source.expression_id, first.expression_id, second.expression_id):
        retrieved = restarted.get_expression(eid)
        assert retrieved.ok and retrieved.trust == TrustLevel.NUMERIC
    again = restarted.simplify(second.expression_id)
    assert again.ok and again.trust == TrustLevel.NUMERIC
    assert restarted.expression_producers[first.expression_id]
    assert source.expression_id in restarted.expression_provenance[first.expression_id]["input_ids"]
    assert before.derivation[-1].parents


def test_assumptions_filter_solutions_and_remain_readable(tmp_path):
    settings = Settings(store_path=str(tmp_path / "contexts.sqlite"))
    kernel = MathKernel(settings)
    ctx = kernel.create_context({"x": "real"}, ["x > 0"]).context_id
    solution = kernel.expression("x^2-1=0").solve("x", ctx)
    assert solution.ok and solution.data["display"] == "{1}"
    assert solution.assumptions_used == ["x > 0"]
    source = kernel.expression("sqrt(x^2)")
    simplified = source.simplify(context_id=ctx)
    assert simplified.display == "x"
    assert "x > 0" in simplified.last_result.assumptions_used
    restarted = MathKernel(settings)
    fetched = restarted.get_expression(simplified.expression_id)
    assert fetched.assumptions_used == ["x > 0"]
    assert fetched.data["required_domains"] == {"x": "real"}
    derivative = restarted.differentiate(simplified.expression_id, "x")
    assert derivative.ok and derivative.assumptions_used == ["x > 0"]


def test_real_domain_does_not_turn_complex_after_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "real.sqlite"))
    kernel = MathKernel(settings)
    ctx = kernel.create_context({"x": "real"}).context_id
    eid = kernel.expression("x^2+1=0").expression_id
    first = kernel.solve(eid, "x", ctx)
    second = MathKernel(settings).solve(eid, "x", ctx)
    assert first.ok and second.ok
    assert first.data["display"] == second.data["display"] == "EmptySet"
    assert first.data["domain"] == second.data["domain"] == "real"


@pytest.mark.parametrize("operation", ["solve", "simplify", "differentiate", "infer_structure"])
def test_unknown_context_is_not_ignored(operation):
    kernel = MathKernel(); eid = kernel.expression("x^2-1").expression_id
    kwargs = {"context_id": "ctx_missing"}
    if operation in {"solve", "differentiate"}:
        kwargs["variable"] = "x"
    result = getattr(kernel, operation)(eid, **kwargs)
    assert not result.ok and "context" in str(result.errors).lower()


def test_nested_unknown_context_is_rejected():
    result = MathKernel().object_create("Signal", {"samples": [1, 2], "context_id": "ctx_missing"})
    assert not result.ok and "context" in str(result.errors).lower()


def test_derived_context_cannot_be_silently_reinterpreted():
    kernel = MathKernel()
    positive = kernel.create_context({"x": "real"}, ["x > 0"]).context_id
    negative = kernel.create_context({"x": "real"}, ["x < 0"]).context_id
    derived = kernel.expression("sqrt(x^2)").simplify(context_id=positive)
    result = kernel.simplify(derived.expression_id, context_id=negative)
    assert not result.ok and "Incompatible contexts" in str(result.errors)


@pytest.mark.parametrize("domain,expected", [("integer", "EmptySet"), ("real", "{-sqrt(2), sqrt(2)}"), ("positive", "{sqrt(2)}")])
def test_explicit_solution_domains(domain, expected):
    kernel = MathKernel()
    ctx = kernel.create_context({"x": domain}).context_id
    result = kernel.expression("x^2-2=0").solve("x", ctx)
    assert result.ok and result.data["display"] == expected


def test_system_filters_and_persists_each_solution_handle(tmp_path):
    settings = Settings(store_path=str(tmp_path / "system.sqlite"))
    kernel = MathKernel(settings)
    ctx = kernel.create_context({"x": "real", "y": "real"}, ["x > 0"]).context_id
    inputs = [kernel.expression(s).expression_id for s in ("x^2-1=0", "y-x=0")]
    result = kernel.solve_system(inputs, ["x", "y"], ctx)
    assert result.ok and result.data["solution_count"] == 1
    restarted = MathKernel(settings)
    for variable in ("x", "y"):
        value = result.data["solutions"][0][variable]
        assert value["display"] == "1"
        retrieved = restarted.get_expression(value["expr_id"])
        assert retrieved.ok and "x > 0" in retrieved.assumptions_used


def test_decimal_bounds_cannot_disappear_into_exact_zero():
    kernel = MathKernel()
    result = kernel.integrate(kernel.expression("1").expression_id, "x", "0.1", "0.1")
    assert result.ok and result.trust == TrustLevel.NUMERIC
    assert kernel.get_expression(result.data["result_expr_id"]).trust == TrustLevel.NUMERIC


def geometry(delta):
    return score_subspace_geometry([0.5, 0.5], [[0.5+delta, 0.5-delta], [0.5-delta, 0.5+delta]], [[-1], [1]])


def test_small_positive_information_is_not_exact_blindness():
    bounds = composite_detection_bounds(geometry(1e-6), 0.1)
    assert not bounds.exact_blind_direction
    assert bounds.weakest_information_retention > 0
    assert bounds.resolution_status == "resolved"
    assert bounds.asymptotic_search_status == "target_attained"
    assert math.isfinite(bounds.asymptotic_composite_samples)


def test_true_rank_loss_has_explicit_exact_finite_model_scope():
    bounds = composite_detection_bounds(geometry(0), 0.1)
    assert bounds.exact_blind_direction
    assert bounds.resolution_status == "exact_rank_loss"
    assert math.isinf(bounds.asymptotic_composite_samples)
    assert "declared binary-float" in bounds.rank_certificate_scope


def test_tolerance_loss_is_reported_as_numerically_unresolved():
    bounds = composite_detection_bounds(geometry(1e-8), 0.1)
    assert not bounds.exact_blind_direction
    assert bounds.asymptotic_composite_samples is None
    assert bounds.resolution_status == "numerically_unresolved"


def test_capped_search_does_not_claim_target_power():
    bounds = composite_detection_bounds(geometry(0.001), 0.001, max_samples=2_000_000_000)
    assert bounds.asymptotic_composite_samples is None
    assert bounds.asymptotic_search_status == "search_limit_reached"
    assert bounds.power_at_search_bound < bounds.requested_power
    assert bounds.sample_search_limit == 2_000_000_000


def test_unbounded_legacy_cap_case_actually_reaches_target():
    from scipy.stats import chi2, ncx2
    bounds = composite_detection_bounds(geometry(0.001), 0.001)
    n = bounds.asymptotic_composite_samples
    assert n > 2_000_000_000
    threshold = chi2.ppf(0.95, bounds.relation_dimension)
    assert ncx2.sf(threshold, bounds.relation_dimension, n*bounds.per_sample_worst_direction_noncentrality) >= 0.9
    assert ncx2.sf(threshold, bounds.relation_dimension, (n-1)*bounds.per_sample_worst_direction_noncentrality) < 0.9


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_search_caps_rejected(cap):
    with pytest.raises(ValueError):
        composite_detection_bounds(geometry(0.01), 0.1, max_samples=cap)


def test_complete_envelope_budget_and_durable_resource(tmp_path):
    settings = Settings(max_output_size_bytes=1000, store_path=str(tmp_path / "resources.sqlite"))
    kernel = MathKernel(settings)
    receipt = kernel.parse("0.5*x-0.5*x")
    assert len(wire_bytes(receipt)) <= 1000
    assert receipt.data["truncated"]
    assert receipt.trust == TrustLevel.UNKNOWN
    restarted = MathKernel(settings)
    full = complete_result(restarted, receipt)
    assert full.ok and full.trust == TrustLevel.NUMERIC
    assert full.derivation
    page = restarted.result_resource_get(receipt.data["resource_id"], 0, 1_000_000)
    assert len(wire_bytes(page)) <= 1000
    assert page["next_offset"] > 0


def test_raw_discovery_is_also_budgeted_and_pages_reconstruct():
    kernel = MathKernel(Settings(max_output_size_bytes=1000))
    receipt = kernel.capabilities()
    assert receipt["truncated"] and len(wire_bytes(receipt)) <= 1000
    offset, chunks = 0, []
    while True:
        page = kernel.result_resource_get(receipt["resource_id"], offset, 100_000)
        assert len(wire_bytes(page)) <= 1000
        chunks.append(page["content"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    payload = "".join(chunks).encode("ascii")
    assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
    assert json.loads(payload)["version"] == kernel.capabilities(detail="summary").get("version", __import__("mathkernel").__version__)


def test_compact_discovery_and_stable_pagination():
    kernel = MathKernel()
    compact = kernel.capabilities(detail="summary")
    assert len(wire_bytes(compact)) < 5000
    pages, offset = [], 0
    while True:
        page = kernel.capability_query(offset=offset, limit=17, include_schema=True)
        assert page["count"] <= 17
        pages.extend(page["capabilities"])
        if page["next_offset"] is None: break
        offset = page["next_offset"]
    assert len(pages) == compact["capability_count"]
    expected = kernel.router.registry.manifest()
    assert [item["name"] for item in pages] == [item["name"] for item in expected]
    assert all("parameter_json_schema" in item for item in pages)
    assert kernel.capability_query()["count"] == 25


def _sleeper():
    time.sleep(2)


def test_timeout_kills_workers_and_followup_is_not_starved():
    def timed_call(_):
        with pytest.raises(SolverTimeoutError):
            run_with_timeout(_sleeper, 0.03)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(timed_call, range(4)))
    assert run_with_timeout(abs, 0.2, -1) == 1


@pytest.mark.parametrize("timeout", [True, False, -1, float("nan"), float("inf")])
def test_invalid_timeouts_fail_closed(timeout):
    with pytest.raises(ValueError):
        run_with_timeout(abs, timeout, -1)


def test_python_facade_uses_same_provenance_even_with_small_output_budget():
    kernel = MathKernel(Settings(max_output_size_bytes=1000))
    original = kernel.expression("x+1")
    output = original.substitute({"x": "0.1"}).simplify()
    assert output.result.trust == TrustLevel.NUMERIC
    assert original.display == "(x + 1)"


