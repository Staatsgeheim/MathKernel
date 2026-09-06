# =============================================================================
# MathKernel - test polish
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import time

import pytest

from mathkernel import MathKernel, Settings
from mathkernel.codegen import CodegenError, RustEmitter
from mathkernel.models import CallNode


@pytest.fixture
def kernel():
    return MathKernel()


def _parse(kernel, text):
    result = kernel.parse(text)
    assert result.ok, result.errors
    return result.data["expr_id"]


# --- codegen guards ---------------------------------------------------------

@pytest.mark.parametrize("language", ["typescript", "python", "rust"])
@pytest.mark.parametrize("expr", ["factorial(n)", "gamma(x)", "binomial(n, k)", "pi()"])
def test_codegen_rejects_non_field_calls(kernel, language, expr):
    eid = _parse(kernel, expr)
    result = kernel.codegen(eid, language=language)
    assert not result.ok
    assert "does not support" in result.errors[0]


def test_rust_emitter_rejects_zero_arg_call():
    with pytest.raises(CodegenError, match="single-argument"):
        RustEmitter().emit_expr(CallNode(name="sin", args=[]))


def test_codegen_still_accepts_algebraic_calls(kernel):
    eid = _parse(kernel, "sqrt(x) + sin(x)")
    for language in ("typescript", "python", "rust"):
        result = kernel.codegen(eid, language=language)
        assert result.ok, (language, result.errors)


# --- uniform unknown-ID errors ----------------------------------------------

def test_unknown_expr_id_returns_clean_errors(kernel):
    assert not kernel.analyze("expr_nope").ok
    assert not kernel.plan("expr_nope").ok
    assert not kernel.simplify("expr_nope").ok
    assert not kernel.solve("expr_nope", "x").ok
    assert not kernel.interval_evaluate("expr_nope", {}).ok
    assert not kernel.differentiate("expr_nope", "x").ok
    assert not kernel.integrate("expr_nope", "x").ok
    assert not kernel.limit("expr_nope", "x", "0").ok
    assert not kernel.series("expr_nope", "x").ok
    assert not kernel.summation("expr_nope", "k", "1", "n").ok
    assert not kernel.numeric_evaluate("expr_nope").ok
    assert not kernel.codegen("expr_nope").ok
    assert not kernel.get_expression("expr_nope").ok
    assert not kernel.substitute("expr_nope", {}).ok
    assert not kernel.infer_structure("expr_nope").ok


def test_unknown_step_and_context_ids(kernel):
    assert not kernel.derivation_get("step_nope").ok
    assert not kernel.derivation_trace("step_nope").ok
    assert not kernel.check_context("ctx_nope").ok
    assert not kernel.infer_context("ctx_nope").ok


# --- settings-driven limits ---------------------------------------------------

def test_matrix_dim_limit_from_settings():
    kernel = MathKernel(Settings(max_matrix_dim=2))
    assert kernel.matrix_create([["1", "2"], ["3", "4"]]).ok
    assert not kernel.matrix_create([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]).ok


def _wait_done(kernel, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = kernel.job_status(job_id)
        if status.ok and status.data["status"] in {"done", "failed"}:
            return
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_retention_cap():
    kernel = MathKernel(Settings(max_jobs_retained=2))
    ids = []
    for _ in range(3):
        submitted = kernel.job_submit("collatz_sieve", {"n_max": 2})
        assert submitted.ok
        ids.append(submitted.data["job_id"])
        _wait_done(kernel, ids[-1])
    # Submit one more to trigger eviction of the oldest finished job.
    submitted = kernel.job_submit("collatz_sieve", {"n_max": 2})
    ids.append(submitted.data["job_id"])
    _wait_done(kernel, ids[-1])
    listed = kernel.job_list()
    assert listed.ok
    assert listed.data["count"] <= 2
    assert not kernel.job_status(ids[0]).ok  # oldest evicted


def test_job_list_filter(kernel):
    submitted = kernel.job_submit("collatz_sieve", {"n_max": 2})
    _wait_done(kernel, submitted.data["job_id"])
    done = kernel.job_list(status="done")
    assert done.ok and done.data["count"] >= 1
    assert all(j["status"] == "done" for j in done.data["jobs"])
    assert kernel.job_list(status="running").data["count"] == 0


# --- capabilities surface -----------------------------------------------------

def test_capabilities_advertise_new_surfaces(kernel):
    caps = kernel.capabilities()
    assert caps["version"] == __import__("mathkernel").__version__
    for op in ("differentiate", "integrate", "limit", "series", "summation", "product",
               "solve_system", "numeric_evaluate", "parse_latex", "matrix_create",
               "cuboid_sweep", "job_submit", "job_list"):
        assert op in caps["operations"], op
    assert caps["jobs"]["kinds"] == ["collatz_sieve", "cuboid_sweep"]
    assert caps["matrices"]["max_dim"] == 128
    assert "det" in caps["matrices"]["operations"]
    assert caps["latex"]["engine"] == "sympy.parsing.latex"
    assert caps["limits"]["max_matrix_dim"] == 128
    assert caps["limits"]["max_jobs_retained"] == 100
