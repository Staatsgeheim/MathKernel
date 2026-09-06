# =============================================================================
# MathKernel - test jobs
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import time

import pytest

from mathkernel import MathKernel


@pytest.fixture
def kernel():
    return MathKernel()


def _wait_done(kernel, job_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = kernel.job_status(job_id)
        assert status.ok
        if status.data["status"] in {"done", "failed"}:
            return status.data
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_submit_unknown_kind(kernel):
    result = kernel.job_submit("nope")
    assert not result.ok
    assert "collatz_sieve" in result.errors[0]


def test_submit_rejects_unknown_params(kernel):
    result = kernel.job_submit("collatz_sieve", {"n_max": 2, "evil": True})
    assert not result.ok
    assert "evil" in result.errors[0]


def test_collatz_job_lifecycle(kernel):
    submitted = kernel.job_submit("collatz_sieve", {"n_max": 3})
    assert submitted.ok
    job_id = submitted.data["job_id"]
    final = _wait_done(kernel, job_id)
    assert final["status"] == "done"
    assert final["elapsed_seconds"] >= 0
    result = kernel.job_result(job_id)
    assert result.ok
    report = result.data["data"]
    assert report["total_patterns_checked"] > 0
    assert report["nontrivial_cycles"] == []
    assert result.evidence_bundle.computation
    assert result.claim_evidence["result"].computation


def test_cuboid_job_lifecycle(kernel):
    submitted = kernel.job_submit("cuboid_sweep", {"bound": 50, "engine": "numba"})
    assert submitted.ok
    job_id = submitted.data["job_id"]
    final = _wait_done(kernel, job_id)
    assert final["status"] == "done"
    result = kernel.job_result(job_id)
    assert result.ok
    report = result.data["data"]
    assert report["pair_count"] > 0
    assert "3" in report["pairs"]  # (3, 4, 5)


def test_failed_job_reports_error(kernel):
    submitted = kernel.job_submit("collatz_sieve", {"n_max": 0})
    job_id = submitted.data["job_id"]
    final = _wait_done(kernel, job_id)
    assert final["status"] == "failed"
    assert final["error"]


def test_job_result_while_pending_or_unknown(kernel):
    assert not kernel.job_result("job_nope").ok
    assert not kernel.job_status("job_nope").ok


def test_cuboid_sweep_direct(kernel):
    result = kernel.cuboid_sweep(25)
    assert result.ok
    assert result.data["pair_count"] > 0
    assert result.trust.value == "exact"
