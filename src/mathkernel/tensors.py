# =============================================================================
# MathKernel - Sparse exact tensors with Einstein-style contraction and sparse exact
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Sparse exact tensors with Einstein-style contraction and sparse exact
linear solve.

Storage is dict-of-keys: {index_tuple: value} with exact Fraction or SymPy
entries. Three compute tiers, bit-checked in tests:

1. exact reference (Fraction/sympy dict-of-keys) — EXACT trust
2. njit float64 CSR Gustavson matmul (numba, `perf` extra) — NUMERIC
3. CuPy sparse CSR matmul (`cuda` extra, runtime-probed) — NUMERIC

General (non-matmul) numeric contractions fall back to dense np.einsum.
Large tensors are returned as shape/nnz summaries by design — the kernel's
output budget would truncate full serializations anyway.
"""
from __future__ import annotations

from fractions import Fraction

MAX_DIM_TOTAL = 10_000_000  # product of shape entries
MAX_ENTRIES = 5_000_000


class SparseTensor:
    def __init__(self, shape: tuple[int, ...], entries: dict[tuple[int, ...], object] | None = None):
        if any(d <= 0 for d in shape):
            raise ValueError("shape dimensions must be positive")
        if len(shape) > 8:
            raise ValueError("at most 8 tensor axes (rank 0 is a scalar)")
        total = 1
        for d in shape:
            total *= d
        if total > MAX_DIM_TOTAL:
            raise ValueError(f"tensor size {total} exceeds limit {MAX_DIM_TOTAL}")
        self.shape = tuple(shape)
        self.entries: dict[tuple[int, ...], object] = {}
        for key, value in (entries or {}).items():
            self.set(key, value)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def _check_key(self, key: tuple[int, ...]) -> None:
        if len(key) != self.ndim:
            raise ValueError(f"index {key} has rank {len(key)}; tensor rank is {self.ndim}")
        if any(i < 0 or i >= d for i, d in zip(key, self.shape)):
            raise ValueError(f"index {key} out of bounds for shape {self.shape}")

    def set(self, key: tuple[int, ...], value) -> None:
        self._check_key(key)
        if value == 0:
            self.entries.pop(tuple(key), None)
            return
        if len(self.entries) >= MAX_ENTRIES and tuple(key) not in self.entries:
            raise ValueError(f"entry count exceeds limit {MAX_ENTRIES}")
        self.entries[tuple(key)] = value

    def get(self, key: tuple[int, ...]):
        self._check_key(key)
        return self.entries.get(tuple(key), 0)

    def summary(self) -> dict:
        return {"shape": list(self.shape), "nnz": self.nnz, "rank": self.ndim}


def _parse_spec(spec: str, n_tensors: int):
    if "->" not in spec:
        raise ValueError("contraction spec must look like 'ij,jk->ik'")
    lhs, rhs = spec.split("->")
    inputs = [part.strip() for part in lhs.split(",")]
    output = rhs.strip()
    if len(inputs) != n_tensors:
        raise ValueError(f"spec has {len(inputs)} operands but {n_tensors} tensors were given")
    if any(not part.isalpha() for part in inputs) or (output and not output.isalpha()):
        raise ValueError("index labels must be single letters")
    if len(set(output)) != len(output):
        raise ValueError("output labels must be distinct")
    return inputs, output


def contract(spec: str, *tensors: SparseTensor) -> SparseTensor:
    """Exact Einstein-style contraction over dict-of-keys tensors."""
    inputs, output = _parse_spec(spec, len(tensors))
    for part, t in zip(inputs, tensors):
        if len(part) != t.ndim:
            raise ValueError(f"spec operand '{part}' has rank {len(part)}; tensor rank is {t.ndim}")
        for label, dim in zip(part, t.shape):
            for other_part, other_t in zip(inputs, tensors):
                if label in other_part:
                    od = other_t.shape[other_part.index(label)]
                    if od != dim:
                        raise ValueError(f"label '{label}' has inconsistent dimensions {dim} vs {od}")
    dims: dict[str, int] = {}
    for part, t in zip(inputs, tensors):
        for label, dim in zip(part, t.shape):
            dims[label] = dim
    summed = [label for label in dims if label not in output]
    out_shape = tuple(dims[label] for label in output)
    result: dict[tuple[int, ...], object] = {}

    def recurse(t_idx: int, assignment: dict[str, int], value) -> None:
        if t_idx == len(tensors):
            out_key = tuple(assignment[label] for label in output)
            free_summed = [l for l in summed if l not in assignment]

            def sum_rec(s_idx: int, acc_value) -> None:
                if s_idx == len(free_summed):
                    result[out_key] = result.get(out_key, 0) + acc_value
                    return
                label = free_summed[s_idx]
                for i in range(dims[label]):
                    assignment[label] = i
                    sum_rec(s_idx + 1, acc_value)
                del assignment[label]
            sum_rec(0, value)
            return
        part = inputs[t_idx]
        t = tensors[t_idx]
        # enumerate entries consistent with the partial assignment
        for key, v in t.entries.items():
            ok = True
            added = []
            for label, i in zip(part, key):
                if label in assignment:
                    if assignment[label] != i:
                        ok = False
                        break
                else:
                    assignment[label] = i
                    added.append(label)
            if ok:
                recurse(t_idx + 1, assignment, v if value is None else value * v)
            for label in added:
                del assignment[label]

    if not tensors:
        raise ValueError("contract needs at least one tensor")
    if len(tensors) == 1:
        # single operand: repeated labels extract the diagonal; labels absent
        # from the output are summed out
        part = inputs[0]
        t = tensors[0]
        for key, v in t.entries.items():
            if any(len({key[i] for i, x in enumerate(part) if x == label}) > 1
                   for label in set(part)):
                continue
            assignment = dict(zip(part, key))
            out_key = tuple(assignment[label] for label in output)
            result[out_key] = result.get(out_key, 0) + v
    else:
        recurse(0, {}, None)
    cleaned = {k: v for k, v in result.items() if v != 0}
    return SparseTensor(out_shape, cleaned)


# --- exact sparse linear solve (2D tensors) --------------------------------------

def sparse_solve_exact(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """Solve A x = b exactly over the rationals. A: (n, n), b: (n,) or (n, k).

    Sparse-row Gaussian elimination: rows stored as dicts, pivot rows chosen
    by fewest nonzeros among rows with a nonzero in the pivot column
    (Markowitz-lite) to limit fill-in. Exact Fraction arithmetic."""
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("coefficient tensor must be square rank-2")
    n = a.shape[0]
    if b.ndim == 1:
        if b.shape[0] != n:
            raise ValueError("rhs length must match A")
        b_cols, b_shape = 1, (n,)
    elif b.ndim == 2 and b.shape[0] == n:
        b_cols, b_shape = b.shape[1], b.shape
    else:
        raise ValueError("rhs must have shape (n,) or (n, k)")
    rows: list[dict[int, Fraction]] = [dict() for _ in range(n)]
    for (i, j), v in a.entries.items():
        rows[i][j] = Fraction(v)
    rhs: list[dict[int, Fraction]] = [dict() for _ in range(n)]
    for key, v in b.entries.items():
        rhs[key[0]][key[1] if b.ndim == 2 else 0] = Fraction(v)
    # Full Gauss-Jordan: at each column, pivot the sparsest candidate row and
    # eliminate the column from every other row. Afterwards each pivot row
    # holds only its pivot column, so x[pivot_col] = rhs[pivot_row] directly.
    used = [False] * n
    pivot_col_of: dict[int, int] = {}
    for col in range(n):
        candidates = [r for r in range(n) if not used[r] and col in rows[r]]
        if not candidates:
            raise ValueError("coefficient matrix is singular")
        pr = min(candidates, key=lambda r: len(rows[r]))
        used[pr] = True
        pivot_col_of[pr] = col
        inv = 1 / rows[pr][col]
        rows[pr] = {j: v * inv for j, v in rows[pr].items()}
        rhs[pr] = {j: v * inv for j, v in rhs[pr].items()}
        prow, prhs = rows[pr], rhs[pr]
        for r in range(n):
            if r == pr or col not in rows[r]:
                continue
            factor = rows[r][col]
            row = rows[r]
            for j, v in prow.items():
                nv = row.get(j, Fraction(0)) - factor * v
                if nv == 0:
                    row.pop(j, None)
                else:
                    row[j] = nv
            rrow = rhs[r]
            for j, v in prhs.items():
                nv = rrow.get(j, Fraction(0)) - factor * v
                if nv == 0:
                    rrow.pop(j, None)
                else:
                    rrow[j] = nv
    entries = {}
    for r, col in pivot_col_of.items():
        for c, v in rhs[r].items():
            if v != 0:
                key = (col,) if b.ndim == 1 else (col, c)
                entries[key] = v
    return SparseTensor(b_shape, entries)


# --- numeric tiers: njit CSR Gustavson + CuPy sparse -------------------------------

def _to_csr(t: SparseTensor):
    """Rank-2 tensor -> sorted CSR float64 arrays (indptr, indices, data)."""
    import numpy as np
    if t.ndim != 2:
        raise ValueError("CSR conversion requires a rank-2 tensor")
    n_rows = t.shape[0]
    items = sorted(t.entries.items())
    rows = np.array([k[0][0] for k in items], dtype=np.int64)
    cols = np.array([k[0][1] for k in items], dtype=np.int64)
    data = np.array([float(v) for _, v in items], dtype=np.float64)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, cols, data


def _spgemm_python(a_ip, a_ix, a_d, b_ip, b_ix, b_d, n_cols):
    """Pure-Python Gustavson reference (exact tier uses Fraction dicts; this is
    the float64 reference the njit kernel is bit-checked against)."""
    out: dict[tuple[int, int], float] = {}
    for i in range(len(a_ip) - 1):
        acc: dict[int, float] = {}
        for pa in range(a_ip[i], a_ip[i + 1]):
            k = a_ix[pa]
            va = a_d[pa]
            for pb in range(b_ip[k], b_ip[k + 1]):
                j = b_ix[pb]
                acc[j] = acc.get(j, 0.0) + va * b_d[pb]
        for j, v in acc.items():
            if v != 0.0:
                out[(i, j)] = v
    return out


def _spgemm_njit():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def spgemm(a_ip, a_ix, a_d, b_ip, b_ix, b_d, n_rows, n_cols):
        counts = [0]
        buffer = [0.0] * n_cols
        seen = [-1] * n_cols
        rows_out = []
        cols_out = []
        vals_out = []
        for i in range(n_rows):
            used = []
            for pa in range(a_ip[i], a_ip[i + 1]):
                k = a_ix[pa]
                va = a_d[pa]
                for pb in range(b_ip[k], b_ip[k + 1]):
                    j = b_ix[pb]
                    if seen[j] != i:
                        seen[j] = i
                        buffer[j] = 0.0
                        used.append(j)
                    buffer[j] += va * b_d[pb]
            for j in used:
                if buffer[j] != 0.0:
                    rows_out.append(i)
                    cols_out.append(j)
                    vals_out.append(buffer[j])
            counts.append(len(rows_out))
        return rows_out, cols_out, vals_out

    return spgemm


def matmul_numeric(a: SparseTensor, b: SparseTensor, prefer_gpu: bool = True) -> tuple[SparseTensor, str]:
    """Numeric A @ B for rank-2 tensors. Returns (result, engine_tier).
    Tiers: numeric-gpu (CuPy CSR) > njit > python float64 reference."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul requires rank-2 tensors with matching inner dimension")
    a_csr = _to_csr(a)
    b_csr = _to_csr(b)
    if prefer_gpu:
        from .koopman import _xp
        xp = _xp(prefer_gpu=True)
        if xp.__name__ == "cupy":
            try:
                from cupyx.scipy.sparse import csr_matrix
                ag = csr_matrix((xp.asarray(a_csr[2]), xp.asarray(a_csr[1]),
                                 xp.asarray(a_csr[0])), shape=a.shape)
                bg = csr_matrix((xp.asarray(b_csr[2]), xp.asarray(b_csr[1]),
                                 xp.asarray(b_csr[0])), shape=b.shape)
                cg = (ag @ bg).tocoo()
                entries = {(int(i), int(j)): float(v)
                           for i, j, v in zip(cg.row.get(), cg.col.get(), cg.data.get())
                           if float(v) != 0.0}
                return SparseTensor((a.shape[0], b.shape[1]), entries), "numeric-gpu"
            except Exception:
                pass  # fall through to CPU tiers
    kernel = _spgemm_njit()
    if kernel is not None:
        rows, cols, vals = kernel(*a_csr, *b_csr, a.shape[0], b.shape[1])
        entries = {(int(i), int(j)): float(v) for i, j, v in zip(rows, cols, vals)}
        return SparseTensor((a.shape[0], b.shape[1]), entries), "njit"
    out = _spgemm_python(*a_csr, *b_csr, b.shape[1])
    return SparseTensor((a.shape[0], b.shape[1]), out), "python-fallback"


def contract_numeric(spec: str, *tensors: SparseTensor, prefer_gpu: bool = True) -> tuple[SparseTensor, str]:
    """Numeric contraction: CSR matmul tiers for 'ij,jk->ik'-style specs,
    dense np.einsum otherwise."""
    inputs, output = _parse_spec(spec, len(tensors))
    if (len(tensors) == 2 and all(t.ndim == 2 for t in tensors)
            and inputs[0][1] == inputs[1][0]
            and output == inputs[0][0] + inputs[1][1]
            and len(set(inputs[0] + inputs[1])) == 3):
        return matmul_numeric(tensors[0], tensors[1], prefer_gpu=prefer_gpu)
    import numpy as np
    dense = [np.zeros(t.shape) for t in tensors]
    for t, arr in zip(tensors, dense):
        for key, v in t.entries.items():
            arr[key] = float(v)
    out = np.einsum(spec, *dense)
    entries = {tuple(int(i) for i in idx): float(v)
               for idx, v in np.ndenumerate(out) if v != 0.0}
    return SparseTensor(tuple(out.shape), entries), "numeric-cpu"
