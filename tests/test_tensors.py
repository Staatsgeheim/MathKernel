# =============================================================================
# MathKernel - Sparse tensors: exact contraction, exact sparse solve, numeric tiers."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Sparse tensors: exact contraction, exact sparse solve, numeric tiers."""

from fractions import Fraction

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.tensors import (SparseTensor, _spgemm_njit, _spgemm_python, _to_csr,
                                contract, matmul_numeric, sparse_solve_exact)


def _demo_matrices():
    a = SparseTensor((2, 3), {(0, 0): Fraction(1), (0, 2): Fraction(2), (1, 1): Fraction(3)})
    b = SparseTensor((3, 2), {(0, 0): Fraction(1), (1, 1): Fraction(1),
                              (2, 0): Fraction(1), (2, 1): Fraction(1)})
    return a, b


def test_exact_contraction_matmul():
    a, b = _demo_matrices()
    c = contract("ij,jk->ik", a, b)
    assert c.entries == {(0, 0): 3, (0, 1): 2, (1, 1): 3}
    assert c.shape == (2, 2)


def test_exact_contraction_trace_and_sum():
    t = SparseTensor((2, 2), {(0, 0): Fraction(5), (1, 1): Fraction(7), (0, 1): Fraction(1)})
    assert contract("ii->", t).entries == {(): 12}
    assert contract("ij->i", t).entries == {(0,): 6, (1,): 7}
    assert contract("ij->", t).entries == {(): 13}


def test_exact_contraction_three_way():
    x = SparseTensor((2, 2), {(0, 0): Fraction(1), (1, 1): Fraction(1)})
    y = SparseTensor((2, 2), {(0, 1): Fraction(2), (1, 0): Fraction(3)})
    z = SparseTensor((2, 2), {(0, 0): Fraction(1), (1, 1): Fraction(1)})
    w = contract("ij,jk,kl->il", x, y, z)
    assert w.entries == {(0, 1): 2, (1, 0): 3}


def test_contraction_dimension_mismatch():
    a, _ = _demo_matrices()
    bad = SparseTensor((4, 4), {(0, 0): Fraction(1)})
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        contract("ij,jk->ik", a, bad)


def test_sparse_solve_exact():
    m = SparseTensor((2, 2), {(0, 0): Fraction(2), (0, 1): Fraction(1),
                              (1, 0): Fraction(1), (1, 1): Fraction(3)})
    b = SparseTensor((2,), {(0,): Fraction(5), (1,): Fraction(5)})
    x = sparse_solve_exact(m, b)
    assert x.entries == {(0,): 2, (1,): 1}
    with pytest.raises(ValueError, match="singular"):
        sparse_solve_exact(SparseTensor((2, 2), {(0, 0): Fraction(1), (0, 1): Fraction(2),
                                                 (1, 0): Fraction(2), (1, 1): Fraction(4)}), b)


def test_sparse_solve_exact_matrix_rhs():
    m = SparseTensor((2, 2), {(0, 0): Fraction(2), (1, 1): Fraction(4)})
    b = SparseTensor((2, 2), {(0, 0): Fraction(2), (1, 1): Fraction(4)})
    x = sparse_solve_exact(m, b)
    assert x.entries == {(0, 0): 1, (1, 1): 1}


def test_njit_matches_python_reference():
    """Cross-tier bit-check: njit Gustavson vs pure-Python float64 reference."""
    kernel = _spgemm_njit()
    if kernel is None:
        pytest.skip("numba not installed")
    import random
    rng = random.Random(0)
    n = 30
    entries_a, entries_b = {}, {}
    for _ in range(120):
        entries_a[(rng.randrange(n), rng.randrange(n))] = rng.random()
        entries_b[(rng.randrange(n), rng.randrange(n))] = rng.random()
    a = SparseTensor((n, n), entries_a)
    b = SparseTensor((n, n), entries_b)
    ref = _spgemm_python(*_to_csr(a), *_to_csr(b), n)
    rows, cols, vals = kernel(*_to_csr(a), *_to_csr(b), n, n)
    got = {(int(i), int(j)): float(v) for i, j, v in zip(rows, cols, vals)}
    assert got == ref


def test_matmul_numeric_tiers_consistent():
    """All available numeric tiers agree with the exact result."""
    a, b = _demo_matrices()
    an = SparseTensor(a.shape, {k: float(v) for k, v in a.entries.items()})
    bn = SparseTensor(b.shape, {k: float(v) for k, v in b.entries.items()})
    result, tier = matmul_numeric(an, bn)
    assert tier in ("numeric-gpu", "njit", "python-fallback")
    exact = contract("ij,jk->ik", a, b)
    assert result.entries == {k: float(v) for k, v in exact.entries.items()}


def test_kernel_tensor_facade():
    k = MathKernel()
    t = k.tensor_create([2, 2], {"0,0": "2", "0,1": "1", "1,0": "1", "1,1": "3"})
    assert t.ok and t.trust == TrustLevel.EXACT
    tb = k.tensor_create([2], {"0": "5", "1": "5"})
    sol = k.tensor_solve(t.data["tensor_id"], tb.data["tensor_id"])
    assert sol.ok and sol.trust == TrustLevel.EXACT
    got = k.tensor_get(sol.data["tensor_id"])
    assert got.data["entries"] == {"0": "2", "1": "1"}
    c = k.tensor_contract("ij,jk->ik", [t.data["tensor_id"], t.data["tensor_id"]])
    assert c.trust == TrustLevel.EXACT and c.data["engine_tier"] == "exact"
    assert k.tensor_get(c.data["tensor_id"]).data["entries"] == \
        {"0,0": "5", "0,1": "5", "1,0": "5", "1,1": "10"}


def test_kernel_tensor_numeric_trust():
    k = MathKernel()
    t = k.tensor_create([2, 2], {"0,0": "0.5", "1,1": "1.5"})
    assert t.trust == TrustLevel.NUMERIC
    c = k.tensor_contract("ij,jk->ik", [t.data["tensor_id"], t.data["tensor_id"]],
                          exact=False)
    assert c.trust == TrustLevel.NUMERIC
    assert c.data["engine_tier"] in ("numeric-gpu", "njit", "python-fallback")
    got = k.tensor_get(c.data["tensor_id"]).data["entries"]
    assert float(got["0,0"]) == pytest.approx(0.25)
    assert float(got["1,1"]) == pytest.approx(2.25)


def test_kernel_tensor_validation():
    k = MathKernel()
    assert not k.tensor_create([2, 2], {"5,0": "1"}).ok  # out of bounds
    assert not k.tensor_solve("tensor_nope", "tensor_nope").ok
    t = k.tensor_create([2, 2], {"0,0": "1", "1,1": "1"})
    assert not k.tensor_contract("ij,jk->ik", [t.data["tensor_id"]]).ok
