# =============================================================================
# MathKernel - test linalg
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import sympy as sp
import pytest

from mathkernel import MathKernel
from mathkernel.models import TrustLevel


@pytest.fixture
def kernel():
    return MathKernel()


def _create(kernel, rows):
    result = kernel.matrix_create(rows)
    assert result.ok, result.errors
    return result.data["matrix_id"]


def _cells(kernel, matrix_id):
    return kernel.matrix_get(matrix_id).data["cells"]


def test_create_and_get(kernel):
    mid = _create(kernel, [["1", "2"], ["3", "4"]])
    got = kernel.matrix_get(mid)
    assert got.ok
    assert got.data["rows"] == 2 and got.data["cols"] == 2
    assert got.data["cells"] == [["1", "2"], ["3", "4"]]
    assert got.trust == TrustLevel.EXACT


def test_create_rejects_ragged_and_empty(kernel):
    assert not kernel.matrix_create([]).ok
    assert not kernel.matrix_create([[]]).ok
    assert not kernel.matrix_create([["1"], ["2", "3"]]).ok
    assert not kernel.matrix_create([["1", "2"], ["oops("]]).ok


def test_det_exact(kernel):
    mid = _create(kernel, [["1", "2"], ["3", "4"]])
    result = kernel.matrix_det(mid)
    assert result.ok
    assert result.data["result"] == "-2"
    assert result.trust == TrustLevel.EXACT


def test_det_symbolic(kernel):
    mid = _create(kernel, [["a", "b"], ["c", "d"]])
    result = kernel.matrix_det(mid)
    assert result.ok
    assert result.trust == TrustLevel.SYMBOLIC
    out = kernel.sympy.to_sympy(kernel.expressions[result.data["result_expr_id"]])
    a, b, c, d = sp.symbols("a b c d")
    assert sp.simplify(out - (a*d - b*c)) == 0


def test_inverse_times_original_is_identity(kernel):
    mid = _create(kernel, [["1", "2"], ["3", "4"]])
    inv = kernel.matrix_inverse(mid)
    assert inv.ok
    product = kernel.matrix_multiply(mid, inv.data["matrix_id"])
    assert product.ok
    assert _cells(kernel, product.data["matrix_id"]) == [["1", "0"], ["0", "1"]]


def test_inverse_singular(kernel):
    mid = _create(kernel, [["1", "2"], ["2", "4"]])
    result = kernel.matrix_inverse(mid)
    assert not result.ok
    assert "singular" in result.errors[0]


def test_transpose(kernel):
    mid = _create(kernel, [["1", "2", "3"], ["4", "5", "6"]])
    result = kernel.matrix_transpose(mid)
    assert result.ok
    assert _cells(kernel, result.data["matrix_id"]) == [["1", "4"], ["2", "5"], ["3", "6"]]


def test_multiply_dimension_mismatch(kernel):
    a = _create(kernel, [["1", "2", "3"]])
    b = _create(kernel, [["1", "2"]])
    assert not kernel.matrix_multiply(a, b).ok


def test_rank(kernel):
    assert kernel.matrix_rank(_create(kernel, [["1", "2"], ["2", "4"]])).data["rank"] == 1
    assert kernel.matrix_rank(_create(kernel, [["1", "2"], ["3", "4"]])).data["rank"] == 2


def test_rref(kernel):
    mid = _create(kernel, [["1", "2"], ["2", "4"]])
    result = kernel.matrix_rref(mid)
    assert result.ok
    assert result.data["pivot_columns"] == [0]
    assert _cells(kernel, result.data["matrix_id"]) == [["1", "2"], ["0", "0"]]


def test_eigenvalues_diagonal(kernel):
    mid = _create(kernel, [["2", "0"], ["0", "3"]])
    result = kernel.matrix_eigenvalues(mid)
    assert result.ok
    pairs = {(e["display"], e["multiplicity"]) for e in result.data["eigenvalues"]}
    assert pairs == {("2", 1), ("3", 1)}


def test_eigenvalues_nontrivial(kernel):
    mid = _create(kernel, [["1", "2"], ["2", "1"]])
    result = kernel.matrix_eigenvalues(mid)
    assert result.ok
    values = {e["display"] for e in result.data["eigenvalues"]}
    assert values == {"3", "-1"}


def test_matrix_solve(kernel):
    a = _create(kernel, [["2", "0"], ["0", "2"]])
    b = _create(kernel, [["4"], ["6"]])
    result = kernel.matrix_solve(a, b)
    assert result.ok
    assert _cells(kernel, result.data["matrix_id"]) == [["2"], ["3"]]


def test_matrix_solve_symbolic(kernel):
    a = _create(kernel, [["a", "0"], ["0", "a"]])
    b = _create(kernel, [["a^2"], ["1"]])
    result = kernel.matrix_solve(a, b)
    assert result.ok
    assert _cells(kernel, result.data["matrix_id"]) == [["a"], ["(a ^ -1)"]]


def test_matrix_solve_singular(kernel):
    a = _create(kernel, [["1", "2"], ["2", "4"]])
    b = _create(kernel, [["1"], ["2"]])
    assert not kernel.matrix_solve(a, b).ok


def test_unknown_matrix_id(kernel):
    assert not kernel.matrix_get("mat_nope").ok
    assert not kernel.matrix_det("mat_nope").ok
    assert not kernel.matrix_inverse("mat_nope").ok
    assert not kernel.matrix_transpose("mat_nope").ok
    assert not kernel.matrix_rank("mat_nope").ok
    assert not kernel.matrix_rref("mat_nope").ok
    assert not kernel.matrix_eigenvalues("mat_nope").ok
    assert not kernel.matrix_solve("mat_nope", "mat_nope").ok


def test_matrix_derivation_recorded(kernel):
    mid = _create(kernel, [["3"]])
    result = kernel.matrix_det(mid)
    assert result.ok
    assert result.derivation[0].operation == "matrix_det"
    assert kernel.derivation_trace(result.derivation[0].step_id).ok
