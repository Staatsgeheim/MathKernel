# =============================================================================
# MathKernel - test parallel
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from mathkernel import MathKernel, Settings
from mathkernel.parallel import process_map, resolve_workers, thread_map

from mathkernel.cuboid import scan_leg_pairs

from cuboid_attack import find_bricks, is_square


def test_resolve_workers_defaults_to_cpu_count():
    assert resolve_workers() == (os.cpu_count() or 4)
    assert resolve_workers(3) == 3
    assert resolve_workers(99, cap=8) == 8
    assert resolve_workers(0) == (os.cpu_count() or 4)


def test_thread_map_matches_serial_order():
    items = list(range(50))
    assert thread_map(lambda x: x * x, items, workers=4) == [x * x for x in items]
    assert thread_map(lambda x: x * x, items, workers=1) == [x * x for x in items]


def test_process_map_matches_serial_order():
    from mathkernel.integers import _integer_job

    jobs = [{"operation": "is_prime", "values": [str(n)]} for n in (2, 4, 97, 221, 7919)]
    expected = [_integer_job(j) for j in jobs]
    assert process_map(_integer_job, jobs, workers=2) == expected
    assert [r["data"]["is_prime"] for r in expected] == [True, False, True, False, True]


def test_integer_batch_via_kernel():
    kernel = MathKernel(Settings(enable_parallel=True, max_workers=4))
    jobs = [
        {"operation": "gcd", "values": ["48", "36"]},
        {"operation": "is_prime", "values": ["97"]},
        {"operation": "pow_mod", "values": ["7", "128"], "modulus": "13"},
        {"operation": "factor", "values": ["360"]},
    ]
    result = kernel.integer_batch(jobs, workers=4)
    assert result.ok
    assert result.data["job_count"] == 4
    assert result.data["failed"] == 0
    results = result.data["results"]
    assert results[0]["data"]["result"]["value"] == "12"
    assert results[1]["data"]["is_prime"] is True
    assert results[2]["data"]["result"]["value"] == str(pow(7, 128, 13))
    assert results[3]["data"]["factors"] == {"2": 3, "3": 2, "5": 1}


def test_integer_batch_per_job_errors_do_not_abort():
    kernel = MathKernel(Settings(enable_parallel=False))
    jobs = [
        {"operation": "is_prime", "values": ["97"]},
        {"operation": "mod", "values": ["5"], "modulus": "0"},
    ]
    result = kernel.integer_batch(jobs)
    assert result.data["failed"] == 1
    assert result.data["results"][0]["ok"] is True
    assert result.data["results"][1]["ok"] is False
    assert "zero" in result.data["results"][1]["error"]


def test_integer_batch_limits():
    kernel = MathKernel(Settings(enable_parallel=False))
    assert not kernel.integer_batch([]).ok
    too_many = kernel.integer_batch([{"operation": "is_prime", "values": ["3"]}] * 10_001)
    assert not too_many.ok
    assert "limit" in too_many.errors[0]


def test_parallel_wave_execution_matches_serial():
    serial = MathKernel(Settings(enable_parallel=False))
    parallel = MathKernel(Settings(enable_parallel=True, max_workers=8))
    results = []
    for kernel in (serial, parallel):
        expr = kernel.parse("x^2 - 5*x + 6 = 0").data["expr_id"]
        results.append(kernel.reason(expr, formal=False))
    s, p = results
    assert s.status == p.status == "verified"
    assert s.trust == p.trust
    s_states = [(o["action"], o["state"]) for o in s.data["execution"]["obligations"]]
    p_states = [(o["action"], o["state"]) for o in p.data["execution"]["obligations"]]
    assert s_states == p_states
    s_solutions = s.data["execution"]["summary"]["solution_set"]
    p_solutions = p.data["execution"]["summary"]["solution_set"]
    assert s_solutions == p_solutions


def test_scan_leg_pairs_small_bound():
    pairs = scan_leg_pairs(1, 5, 12)
    assert pairs[3] == (4,)  # 3-4-5


def test_find_bricks_recovers_smallest_euler_brick():
    pairs = scan_leg_pairs(1, 241, 240)
    bricks = find_bricks(pairs)
    assert (44, 117, 240) in bricks
    assert not any(is_square(a * a + b * b + c * c) for a, b, c in bricks)


def test_capabilities_advertise_parallelism():
    kernel = MathKernel()
    caps = kernel.capabilities()
    assert "integer_batch" in caps["operations"]
    assert caps["parallel"]["obligation_waves"] == "threads"
    assert caps["parallel"]["integer_batch"] == "processes"
    assert caps["limits"]["max_workers"] >= 1
