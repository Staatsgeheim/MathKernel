# =============================================================================
# MathKernel - Tests for the array-level numba kernels in integers_fast."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Tests for the array-level numba kernels in integers_fast."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
import sympy as sp

from mathkernel.integers_fast import (HAVE_NUMBA, gcd_array, is_prime_array,
                                          mod_array, modinv_array, powmod_array)

pytestmark = pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")

RNG = random.Random(20260901)


def test_powmod_array_matches_python():
    n = 500
    bases = np.array([RNG.randrange(0, 1 << 32) for _ in range(n)], np.uint64)
    exps = np.array([RNG.randrange(0, 1 << 40) for _ in range(n)], np.uint64)
    mods = np.array([RNG.randrange(1, 1 << 32) for _ in range(n)], np.uint64)
    got = powmod_array(bases, exps, mods)
    for i in range(n):
        assert int(got[i]) == pow(int(bases[i]), int(exps[i]), int(mods[i]))


def test_powmod_array_edges():
    got = powmod_array(np.array([0, 5, 123], np.uint64),
                       np.array([0, 0, 456], np.uint64),
                       np.array([7, 1, 1], np.uint64))
    assert list(got) == [pow(0, 0, 7), pow(5, 0, 1), pow(123, 456, 1)]


def test_is_prime_array_matches_sympy():
    specials = [0, 1, 2, 3, 4, 2**31 - 1, 2**32 - 5, 3215031751,
                561, 1105, 1729, 2047, 3277]  # Carmichael + base-2 pseudoprimes
    ns = specials + [RNG.randrange(0, 1 << 32) for _ in range(1000)]
    got = is_prime_array(np.array(ns, np.int64))
    for v, g in zip(ns, got):
        assert bool(g) == bool(sp.isprime(v)), f"is_prime mismatch at {v}"


def test_modinv_array_matches_python():
    n = 500
    vals = np.array([RNG.randrange(0, 1 << 32) for _ in range(n)], np.int64)
    mods = np.array([RNG.randrange(1, 1 << 32) for _ in range(n)], np.int64)
    got = modinv_array(vals, mods)
    for i in range(n):
        a, m = int(vals[i]), int(mods[i])
        if m == 1:
            assert int(got[i]) == 0
        elif math.gcd(a, m) != 1:
            assert int(got[i]) == -1
        else:
            assert int(got[i]) == pow(a, -1, m)


def test_mod_array_matches_python():
    n = 500
    vals = np.array([RNG.randrange(0, 1 << 63) for _ in range(n)], np.int64)
    mods = np.array([RNG.randrange(1, 1 << 63) for _ in range(n)], np.int64)
    got = mod_array(vals, mods)
    for i in range(n):
        assert int(got[i]) == int(vals[i]) % int(mods[i])


def test_gcd_array_matches_python():
    rows = [[RNG.randrange(-(1 << 62), 1 << 62) for _ in range(RNG.randrange(1, 6))]
            for _ in range(200)]
    width = max(len(r) for r in rows)
    vals = np.zeros((len(rows), width), np.int64)
    for i, r in enumerate(rows):
        vals[i, :len(r)] = r
    got = gcd_array(vals)
    for row, g in zip(rows, got):
        assert int(g) == math.gcd(*row)
