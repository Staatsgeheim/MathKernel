# =============================================================================
# MathKernel - test collatz
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import pytest

from mathkernel import MathKernel, Settings
from mathkernel.collatz import (
    admissible_s_values, cycle_constant, sieve_cycle_classes, verify_cycle,
)
from mathkernel.collatz_fast import HAVE_NUMBA


def test_admissible_s_self_contained():
    # x_min=1 gives 2^S <= 3^n (4/3)^n = 4^n, i.e. S <= 2n, plus 2^S > 3^n.
    assert admissible_s_values(1, 1) == [2]
    assert admissible_s_values(2, 1) == [4]
    assert admissible_s_values(3, 1) == [5, 6]
    assert admissible_s_values(4, 1) == [7, 8]


def test_admissible_s_with_high_floor():
    # With a high verification floor the admissible range collapses entirely for
    # small n: 2^S <= 3^n (1 + 1/(3 x_min))^n < 2^ceil(n log2 3). The class is
    # then vacuously excluded with zero enumeration (the Eliahou-style bound).
    floor = 2**68
    for n in (1, 2, 3, 5, 8):
        assert admissible_s_values(n, floor) == []
    # A moderate floor still leaves work: n=1 needs 2^S <= 3(1 + 1/9) = 10/3.
    assert admissible_s_values(1, 3) == []


def test_cycle_constant_formula():
    # C(a) = sum_i 3^(n-1-i) 2^(a_0+...+a_{i-1})
    assert cycle_constant((2,)) == 1
    assert cycle_constant((2, 2)) == 3 + 4
    assert cycle_constant((1, 3)) == 3 + 2
    assert cycle_constant((2, 2, 2)) == 9 + 3 * 4 + 16


def test_verify_cycle_trivial():
    assert verify_cycle(1, (2,), 1)
    assert verify_cycle(1, (2, 2), 1)
    assert not verify_cycle(1, (1,), 1)       # (3*1+1)/2 = 2 is even -> not odd
    assert not verify_cycle(2, (2,), 1)       # x0 must be odd
    assert not verify_cycle(1, (2,), 5)       # below x_min floor


def test_sieve_recovers_trivial_cycle_and_refutes_small_classes():
    report = sieve_cycle_classes(4, x_min=1, workers=1)
    assert report["all_classes_refuted"]
    assert report["nontrivial_cycles"] == []
    # The trivial cycle (x0=1, all a_i=2) lives exactly in classes with
    # S = 2n; it must be recovered there as a soundness check.
    for cls in report["classes"]:
        assert cls["complete"]
        has_trivial = any(c["trivial"] for c in cls["cycles"])
        assert has_trivial == (cls["S"] == 2 * cls["n"]), \
            f"trivial cycle mismatch at {cls}"
    # Steiner's theorem (1977), recomputed: no 1-cycle besides the trivial one.
    n1 = [c for c in report["classes"] if c["n"] == 1]
    assert all(c["refuted"] for c in n1)


def test_sieve_parallel_matches_serial():
    serial = sieve_cycle_classes(6, x_min=1, workers=1)
    parallel = sieve_cycle_classes(6, x_min=1, workers=4)
    assert serial["total_patterns_checked"] == parallel["total_patterns_checked"]
    assert serial["all_classes_refuted"] == parallel["all_classes_refuted"]
    s_cycles = sorted(c["x0"] for r in serial["classes"] for c in r["cycles"])
    p_cycles = sorted(c["x0"] for r in parallel["classes"] for c in r["cycles"])
    assert s_cycles == p_cycles


def test_canonical_matches_full_enumeration():
    # Rotation pruning must not lose cycles: every cycle has a rotation whose
    # first halving count is maximal, so the cycle sets must be identical.
    full = sieve_cycle_classes(6, x_min=1, workers=1, canonical=False)
    canon = sieve_cycle_classes(6, x_min=1, workers=1, canonical=True)
    f_cycles = sorted(c["x0"] for r in full["classes"] for c in r["cycles"])
    c_cycles = sorted(c["x0"] for r in canon["classes"] for c in r["cycles"])
    assert f_cycles == c_cycles
    assert full["all_classes_refuted"] and canon["all_classes_refuted"]
    # Pruning actually prunes, and the trivial cycle (all a_i = 2) survives it.
    assert canon["total_patterns_checked"] < full["total_patterns_checked"]
    for cls in canon["classes"]:
        assert cls["complete"]
        has_trivial = any(c["trivial"] for c in cls["cycles"])
        assert has_trivial == (cls["S"] == 2 * cls["n"])


def test_capped_compositions():
    from math import comb
    from mathkernel.collatz import _compositions_capped
    # Compositions of 6 into 3 parts capped at 2: only (2,2,2).
    assert list(_compositions_capped(6, 3, 2)) == [(2, 2, 2)]
    # Capped at hi >= remaining coincides with uncapped count.
    assert len(list(_compositions_capped(7, 3, 7))) == comb(6, 2)
    # Infeasible: remaining too large for the cap.
    assert list(_compositions_capped(10, 3, 2)) == []


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("canonical", [True, False])
def test_numba_engine_matches_python(canonical):
    py = sieve_cycle_classes(8, x_min=1, workers=1, engine="python",
                             canonical=canonical)
    nb = sieve_cycle_classes(8, x_min=1, workers=4, engine="numba",
                             canonical=canonical)
    assert nb["engine"] == "numba"
    assert py["total_patterns_checked"] == nb["total_patterns_checked"]
    assert py["all_classes_refuted"] == nb["all_classes_refuted"]
    for cls_py, cls_nb in zip(py["classes"], nb["classes"]):
        assert cls_py["patterns_checked"] == cls_nb["patterns_checked"]
        assert sorted(c["x0"] for c in cls_py["cycles"]) == \
               sorted(c["x0"] for c in cls_nb["cycles"])


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_engine_recovers_trivial_cycle():
    report = sieve_cycle_classes(4, x_min=1, workers=2, engine="numba")
    for cls in report["classes"]:
        has_trivial = any(c["trivial"] for c in cls["cycles"])
        assert has_trivial == (cls["S"] == 2 * cls["n"])


def _have_cuda():
    from mathkernel.collatz_gpu import HAVE_CUPY
    if not HAVE_CUPY:
        return False
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_cuda(), reason="CUDA GPU not available")
@pytest.mark.parametrize("canonical", [True, False])
def test_cuda_engine_matches_python(canonical):
    py = sieve_cycle_classes(7, x_min=1, workers=1, engine="python",
                             canonical=canonical)
    cu = sieve_cycle_classes(7, x_min=1, workers=1, engine="cuda",
                             canonical=canonical)
    assert cu["engine"] == "cuda"
    assert py["total_patterns_checked"] == cu["total_patterns_checked"]
    for cls_py, cls_cu in zip(py["classes"], cu["classes"]):
        assert cls_py["patterns_checked"] == cls_cu["patterns_checked"]
        assert sorted(c["x0"] for c in cls_py["cycles"]) == \
               sorted(c["x0"] for c in cls_cu["cycles"])


def test_kernel_collatz_sieve_result_shape():
    kernel = MathKernel(Settings(enable_parallel=True, max_workers=4))
    result = kernel.collatz_sieve(4, x_min="1", workers=4)
    assert result.ok
    assert result.status == "verified"
    assert result.trust.value == "exact"
    assert result.data["all_classes_refuted"]
    assert result.side_conditions == []


def test_kernel_collatz_sieve_external_floor_side_condition():
    kernel = MathKernel(Settings(enable_parallel=False))
    result = kernel.collatz_sieve(3, x_min="1000")
    assert result.ok
    assert result.side_conditions
    assert "external verification floor" in result.side_conditions[0]


def test_kernel_collatz_sieve_validation():
    kernel = MathKernel()
    assert not kernel.collatz_sieve(0).ok
    assert not kernel.collatz_sieve(5, x_min="0").ok
    assert not kernel.collatz_sieve(5, x_min="abc").ok
