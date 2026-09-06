# =============================================================================
# MathKernel - Regression tests for the trust/resource-invariant hardening round
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Regression tests for the trust/resource-invariant hardening round.

Covers: closure enumeration budget guard, central output-size budget,
engine timeout wiring, matrix trust propagation, verify_code trust
semantics, and the full advertised integer operation set.
"""

import time

import pytest

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.relations import cyclic_closure


# --- closure enumeration budget guard ---------------------------------------

def test_closure_budget_guard_rejects_from_bound_without_enumerating():
    # The combinatorial upper bound for this search is ~85M tuples per side,
    # far over MAX_ENUMERATION. Rejection must be immediate: the guard must
    # never enumerate to discover the count.
    start = time.perf_counter()
    with pytest.raises(ValueError, match="budget"):
        cyclic_closure([1, 2, 3, 4, 5, 6], 2**31 - 1, 400)
    assert time.perf_counter() - start < 5.0


def test_closure_within_budget_still_works():
    relations = cyclic_closure([1, 5], 97, 10)
    assert (5, -1) in relations and (-5, 1) in relations


# --- central output-size budget ----------------------------------------------

def test_output_budget_truncates_oversized_payload():
    kernel = MathKernel(Settings(max_output_size_bytes=50_000))
    result = kernel.fwht(["0"] * 262144)
    assert result.ok
    assert result.data["truncated"] is True
    assert result.data["size_bytes"] > 50_000
    assert result.data["limit_bytes"] == 50_000
    assert "summary" in result.data
    assert any("budget" in w for w in result.warnings)


def test_output_budget_leaves_small_results_untouched():
    kernel = MathKernel()
    result = kernel.fwht(["1", "-1", "1", "-1"])
    assert result.ok
    assert result.data["values"] == ["0", "4", "0", "0"]
    assert "truncated" not in result.data


def test_output_budget_covers_methods_without_return_annotation():
    # regression: the budget wrapper used to key on the return type
    # annotation, so annotation-less public methods (parse, analyze, solve,
    # ...) bypassed it. It now applies to any MathResult at runtime.
    kernel = MathKernel(Settings(max_output_size_bytes=1000))
    setup = MathKernel()  # untruncated ids for the follow-up calls
    eid = setup.parse("x^2 - 1").data["expr_id"]
    kernel.expressions[eid] = setup.expressions[eid]
    kernel.expression_sources[eid] = setup.expression_sources[eid]
    for call in (lambda: kernel.parse("x^2 + 2*x + 1"),
                 lambda: kernel.analyze(eid),
                 lambda: kernel.solve(eid, "x")):
        result = call()
        assert result.ok
        assert result.data["truncated"] is True
        assert result.data["limit_bytes"] == 1000


# --- engine timeout wiring ----------------------------------------------------

def test_sympy_timeout_wrapper_fires():
    # deterministic: a 5s sleep under a 50ms budget always times out,
    # regardless of thread scheduling or machine load
    from mathkernel.engines import SymPyEngine, SolverTimeoutError, _with_timeout

    engine = SymPyEngine(timeout_seconds=0.05)

    @_with_timeout
    def slow(self):
        time.sleep(5)

    with pytest.raises(SolverTimeoutError, match="solver_timeout_seconds"):
        slow(engine)


def test_solver_timeout_surfaces_as_structured_error(monkeypatch):
    from mathkernel.engines import SolverTimeoutError

    kernel = MathKernel()
    expr_id = kernel.parse("x + x").data["expr_id"]

    def boom(*args, **kwargs):
        raise SolverTimeoutError("sympy simplify exceeded solver_timeout_seconds=30.0")

    monkeypatch.setattr(type(kernel.sympy), "simplify", boom)
    result = kernel.simplify(expr_id)
    assert not result.ok
    assert "solver_timeout_seconds" in result.errors[0]


def test_sympy_default_timeout_allows_normal_work():
    kernel = MathKernel()
    expr_id = kernel.parse("x + x").data["expr_id"]
    result = kernel.simplify(expr_id)
    assert result.ok and result.data["result"] == "(2 * x)"


def test_z3_and_lean_timeouts_are_configured():
    kernel = MathKernel(Settings(z3_timeout_ms=1234, lean_timeout_seconds=9.0))
    assert kernel.z3.timeout_ms == 1234
    assert kernel.lean.timeout == 9.0


# --- matrix trust propagation --------------------------------------------------

def test_numeric_matrix_rank_is_not_exact():
    kernel = MathKernel()
    mid = kernel.matrix_create([["0.1", "0.2"], ["0.2", "0.4"]]).data["matrix_id"]
    result = kernel.matrix_rank(mid)
    assert result.data["rank"] == 1
    assert result.trust == TrustLevel.NUMERIC


def test_matrix_solve_trust_includes_rhs():
    kernel = MathKernel()
    a = kernel.matrix_create([["1", "0"], ["0", "1"]]).data["matrix_id"]
    b = kernel.matrix_create([["0.1"], ["0.2"]]).data["matrix_id"]
    result = kernel.matrix_solve(a, b)
    assert result.ok
    assert result.trust == TrustLevel.NUMERIC


def test_matrix_multiply_trust_is_weakest_input():
    kernel = MathKernel()
    exact = kernel.matrix_create([["1", "0"], ["0", "1"]]).data["matrix_id"]
    numeric = kernel.matrix_create([["0.5", "1"], ["2", "3"]]).data["matrix_id"]
    assert kernel.matrix_multiply(exact, numeric).trust == TrustLevel.NUMERIC
    exact2 = kernel.matrix_create([["1", "2"], ["3", "4"]]).data["matrix_id"]
    assert kernel.matrix_multiply(exact, exact2).trust == TrustLevel.EXACT


def test_exact_matrix_rank_stays_exact():
    kernel = MathKernel()
    mid = kernel.matrix_create([["1", "2"], ["3", "4"]]).data["matrix_id"]
    assert kernel.matrix_rank(mid).trust == TrustLevel.EXACT


# --- verify_code trust semantics ------------------------------------------------

def test_verify_code_unavailable_checks_never_exact():
    kernel = MathKernel()
    expr_id = kernel.parse("x^2 + 2*x + 1").data["expr_id"]
    for language in ("python", "typescript", "rust"):
        artifact = kernel.codegen(expr_id, language=language)
        assert artifact.ok, language
        result = kernel.verify_code(artifact.data["artifact_id"])
        assert result.ok, language
        statuses = {c["check"]: c["status"] for c in result.data["checks"]}
        if "unavailable" in statuses.values():
            assert result.trust != TrustLevel.EXACT, language
            assert result.data["all_passed"] is False
        else:
            assert result.trust == TrustLevel.EXACT, language
            assert result.data["all_passed"] is True


# --- integer operation set -------------------------------------------------------

def test_advertised_integer_operations_all_exist():
    kernel = MathKernel()
    cases = [
        ("add", ["2", "3"], {}, "5"),
        ("sub", ["10", "4", "1"], {}, "5"),
        ("mul", ["6", "7"], {}, "42"),
        ("div", ["20", "5"], {}, "4"),
        ("mod", ["17"], {"modulus": "5"}, "2"),
        ("pow", ["2", "64"], {}, "18446744073709551616"),
        ("pow_mod", ["4", "13"], {"modulus": "497"}, "445"),
        ("gcd", ["48", "36"], {}, "12"),
        ("lcm", ["4", "6"], {}, "12"),
        ("next_prime", ["1000"], {}, "1009"),
        ("prev_prime", ["1000"], {}, "997"),
        ("mod_inverse", ["3"], {"modulus": "11"}, "4"),
        ("factorial", ["10"], {}, "3628800"),
        ("binomial", ["10", "3"], {}, "120"),
        ("fibonacci", ["50"], {}, "12586269025"),
    ]
    for operation, values, kwargs, expected in cases:
        result = kernel.integer_compute(operation, values, **kwargs)
        assert result.ok, f"{operation}: {result.errors}"
        raw = result.data["result"]
        value = raw["value"] if isinstance(raw, dict) else raw
        assert value == expected, operation
        assert result.trust == TrustLevel.EXACT, operation


def test_integer_div_is_exact_rational_when_not_divisible():
    kernel = MathKernel()
    result = kernel.integer_compute("div", ["22", "7"])
    assert result.ok
    assert result.data["result"] == "22/7"
    assert result.data["exact_rational"] is True


def test_integer_is_prime_factor_crt_affine_jump():
    kernel = MathKernel()
    assert kernel.integer_compute("is_prime", ["97"]).data["is_prime"] is True
    factors = kernel.integer_compute("factor", ["360"]).data["factors"]
    assert factors == {"2": 3, "3": 2, "5": 1}
    crt = kernel.integer_compute("crt", ["2", "3"], moduli=["3", "5"])
    assert crt.data["consistent"] and crt.data["result"]["value"] == "8"
    jump = kernel.integer_compute("affine_jump", ["5", "1", "10"], modulus="16")
    assert jump.ok


def test_integer_pow_resource_guard():
    kernel = MathKernel()
    result = kernel.integer_compute("pow", ["10", "1000000000"])
    assert not result.ok
    assert "max_output_digits" in result.errors[0]


# --- job trust passthrough -------------------------------------------------------

def test_job_result_preserves_underlying_trust():
    kernel = MathKernel()
    job_id = kernel.job_submit("collatz_sieve", {"n_max": 8}).data["job_id"]
    for _ in range(200):
        status = kernel.job_status(job_id).data["status"]
        if status in {"done", "failed"}:
            break
        time.sleep(0.05)
    result = kernel.job_result(job_id)
    assert result.ok
    assert result.trust == TrustLevel(result.data["trust"])
