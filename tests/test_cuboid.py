# =============================================================================
# MathKernel - Tests for the QR-prefiltered leg-pair sweep: all engines must agree exactly."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Tests for the QR-prefiltered leg-pair sweep: all engines must agree exactly."""
from __future__ import annotations

import pytest

from mathkernel.cuboid import (BATTERY_PRIMES, HAVE_CUPY, HAVE_NUMBA,
                                   qr_masks, sweep_leg_pairs)

# Leg pairs of the smallest Euler brick (44, 117, 240), plus small triples.
KNOWN_PAIRS = [(3, 4), (5, 12), (8, 15), (7, 24), (20, 21),
               (44, 117), (44, 240), (117, 240)]


def test_qr_masks():
    masks = qr_masks()
    for j, p in enumerate(BATTERY_PRIMES):
        residues = {(x * x) % p for x in range(p)}
        for r in range(p):
            assert bool((int(masks[j]) >> r) & 1) == (r in residues)


def test_python_engine_finds_known_pairs():
    pairs = sweep_leg_pairs(300, engine="python", workers=1)["pairs"]
    for a, b in KNOWN_PAIRS:
        assert b in pairs.get(a, ()), f"pair ({a}, {b}) missing"


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_python():
    ref = sweep_leg_pairs(2000, engine="python", workers=1)["pairs"]
    got = sweep_leg_pairs(2000, engine="numba")["pairs"]
    assert got == ref


@pytest.mark.skipif(not HAVE_CUPY, reason="cupy/CUDA not available")
def test_cuda_matches_python():
    ref = sweep_leg_pairs(2000, engine="python", workers=1)["pairs"]
    got = sweep_leg_pairs(2000, engine="cuda")["pairs"]
    assert got == ref


def test_sweep_metadata():
    out = sweep_leg_pairs(100, engine="python", workers=1)
    assert out["pairs_examined"] == 100 * 99 // 2
    assert out["pair_count"] == sum(len(v) for v in out["pairs"].values())


def test_invalid_engine_and_bound():
    with pytest.raises(ValueError):
        sweep_leg_pairs(100, engine="quantum")
    with pytest.raises(ValueError):
        sweep_leg_pairs(2**31, engine="python", workers=1)
