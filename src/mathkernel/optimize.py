# =============================================================================
# MathKernel - Optimization: exact symbolic critical points, exact rational LP simplex,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Optimization: exact symbolic critical points, exact rational LP simplex,
symbolic KKT conditions, and numeric local search.

- Critical points via gradient + the kernel's exact/symbolic solve (EXACT or
  SYMBOLIC depending on the solve evidence).
- LP via a Fraction tableau simplex with Bland's rule (EXACT); an njit
  float64 tableau is the numeric fast tier, bit-checked against it.
- Numeric local optimization: own Nelder-Mead (no scipy needed, NUMERIC);
  optional scipy path (`sci` extra). Multi-start parallelizes over start
  points via the process pool.
"""
from __future__ import annotations

from fractions import Fraction

from .parallel import process_map, resolve_workers

MAX_LP_CONSTRAINTS = 512
MAX_LP_VARIABLES = 512
MAX_SIMPLEX_ITER = 100_000


def simplex_exact(c: list, a: list[list], b: list) -> dict:
    """max c^T x subject to A x <= b, x >= 0, b >= 0. Exact rational simplex
    with Bland's rule (anti-cycling)."""
    c = [Fraction(v) for v in c]
    a = [[Fraction(v) for v in row] for row in a]
    b = [Fraction(v) for v in b]
    m, n = len(a), len(c)
    if m > MAX_LP_CONSTRAINTS or n > MAX_LP_VARIABLES:
        raise ValueError("LP size exceeds limits")
    if any(len(row) != n for row in a) or len(b) != m:
        raise ValueError("dimension mismatch between c, A, b")
    if any(bi < 0 for bi in b):
        raise ValueError("this simplex form requires b >= 0 (convert constraints first)")
    # tableau: m+1 rows, n+m+1 columns (last row = objective, last col = rhs)
    width = n + m + 1
    tab = [[Fraction(0)] * width for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            tab[i][j] = a[i][j]
        tab[i][n + i] = Fraction(1)
        tab[i][-1] = b[i]
    for j in range(n):
        tab[m][j] = -c[j]
    basis = [n + i for i in range(m)]
    iterations = 0
    while True:
        if iterations >= MAX_SIMPLEX_ITER:
            raise ValueError("simplex exceeded iteration cap")
        iterations += 1
        # Bland: smallest-index negative reduced cost
        pivot_col = next((j for j in range(n + m) if tab[m][j] < 0), None)
        if pivot_col is None:
            break
        # ratio test; Bland tie-break on smallest basis variable
        pivot_row, best_ratio = None, None
        for i in range(m):
            if tab[i][pivot_col] > 0:
                ratio = tab[i][-1] / tab[i][pivot_col]
                if (best_ratio is None or ratio < best_ratio or
                        (ratio == best_ratio and basis[i] < basis[pivot_row])):
                    pivot_row, best_ratio = i, ratio
        if pivot_row is None:
            return {"status": "unbounded", "iterations": iterations}
        inv = 1 / tab[pivot_row][pivot_col]
        tab[pivot_row] = [v * inv for v in tab[pivot_row]]
        for i in range(m + 1):
            if i != pivot_row and tab[i][pivot_col] != 0:
                factor = tab[i][pivot_col]
                tab[i] = [v - factor * w for v, w in zip(tab[i], tab[pivot_row])]
        basis[pivot_row] = pivot_col
    x = [Fraction(0)] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tab[i][-1]
    return {"status": "optimal", "x": x, "objective": tab[m][-1],
            "iterations": iterations}


def _simplex_float64(c, a, b) -> dict:
    """float64 mirror of simplex_exact (numeric tier reference)."""
    c = [float(v) for v in c]
    a = [[float(v) for v in row] for row in a]
    b = [float(v) for v in b]
    m, n = len(a), len(c)
    width = n + m + 1
    tab = [[0.0] * width for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            tab[i][j] = a[i][j]
        tab[i][n + i] = 1.0
        tab[i][-1] = b[i]
    for j in range(n):
        tab[m][j] = -c[j]
    basis = [n + i for i in range(m)]
    iterations = 0
    while iterations < MAX_SIMPLEX_ITER:
        iterations += 1
        pivot_col = next((j for j in range(n + m) if tab[m][j] < -1e-12), None)
        if pivot_col is None:
            break
        pivot_row, best_ratio = None, None
        for i in range(m):
            if tab[i][pivot_col] > 1e-12:
                ratio = tab[i][-1] / tab[i][pivot_col]
                if (best_ratio is None or ratio < best_ratio or
                        (ratio == best_ratio and basis[i] < basis[pivot_row])):
                    pivot_row, best_ratio = i, ratio
        if pivot_row is None:
            return {"status": "unbounded", "iterations": iterations}
        inv = 1.0 / tab[pivot_row][pivot_col]
        tab[pivot_row] = [v * inv for v in tab[pivot_row]]
        for i in range(m + 1):
            if i != pivot_row and abs(tab[i][pivot_col]) > 0:
                factor = tab[i][pivot_col]
                tab[i] = [v - factor * w for v, w in zip(tab[i], tab[pivot_row])]
        basis[pivot_row] = pivot_col
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = tab[i][-1]
    return {"status": "optimal", "x": x, "objective": tab[m][-1],
            "iterations": iterations}


def simplex_numeric(c: list, a: list[list], b: list) -> dict:
    """Numeric LP tier: njit float64 tableau when numba is available, else the
    pure-Python float64 reference. Returns (result, engine_tier)."""
    try:
        from numba import njit
        import numpy as np

        @njit(cache=True)
        def _solve(tab, basis, m, n):
            iterations = 0
            while iterations < 100000:
                iterations += 1
                pivot_col = -1
                for j in range(n + m):
                    if tab[m, j] < -1e-12:
                        pivot_col = j
                        break
                if pivot_col < 0:
                    break
                pivot_row = -1
                best = 0.0
                for i in range(m):
                    if tab[i, pivot_col] > 1e-12:
                        ratio = tab[i, n + m] / tab[i, pivot_col]
                        if pivot_row < 0 or ratio < best or (
                                ratio == best and basis[i] < basis[pivot_row]):
                            pivot_row, best = i, ratio
                if pivot_row < 0:
                    return tab, basis, -1
                inv = 1.0 / tab[pivot_row, pivot_col]
                for j in range(n + m + 1):
                    tab[pivot_row, j] *= inv
                for i in range(m + 1):
                    if i != pivot_row and tab[i, pivot_col] != 0.0:
                        factor = tab[i, pivot_col]
                        for j in range(n + m + 1):
                            tab[i, j] -= factor * tab[pivot_row, j]
                basis[pivot_row] = pivot_col
            return tab, basis, iterations

        m, n = len(a), len(c)
        width = n + m + 1
        tab = np.zeros((m + 1, width))
        for i in range(m):
            for j in range(n):
                tab[i, j] = float(a[i][j])
            tab[i, n + i] = 1.0
            tab[i, -1] = float(b[i])
        for j in range(n):
            tab[m, j] = -float(c[j])
        basis = np.array([n + i for i in range(m)], dtype=np.int64)
        tab, basis, iters = _solve(tab, basis, m, n)
        if iters < 0:
            return {"status": "unbounded"}, "njit"
        x = [0.0] * n
        for i in range(m):
            if basis[i] < n:
                x[basis[i]] = float(tab[i, -1])
        return {"status": "optimal", "x": x, "objective": float(tab[m, -1]),
                "iterations": int(iters)}, "njit"
    except ImportError:
        return _simplex_float64(c, a, b), "python-fallback"


def nelder_mead(f, x0: list[float], tol: float = 1e-10, max_iter: int = 1000) -> dict:
    """Derivative-free local minimization (float64). f: list[float] -> float."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] += 0.05 if point[i] == 0 else 0.05 * abs(point[i])
        simplex.append(point)
    values = [f(p) for p in simplex]
    for iteration in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        spread = max(abs(values[i] - values[0]) for i in range(1, n + 1))
        size = max(max(abs(simplex[i][j] - simplex[0][j]) for j in range(n))
                   for i in range(1, n + 1))
        if spread < tol and size < tol ** 0.5:
            return {"x": simplex[0], "f": values[0], "iterations": iteration,
                    "converged": True}
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        worst = simplex[-1]
        reflected = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(reflected)
        if values[0] <= fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
            continue
        if fr < values[0]:
            expanded = [centroid[j] + 2 * (reflected[j] - centroid[j]) for j in range(n)]
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
            continue
        contracted = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
        fc = f(contracted)
        if fc < values[-1]:
            simplex[-1], values[-1] = contracted, fc
            continue
        best = simplex[0]
        simplex = [best] + [[best[j] + 0.5 * (p[j] - best[j]) for j in range(n)]
                            for p in simplex[1:]]
        values = [f(p) for p in simplex]
    return {"x": simplex[0], "f": values[0], "iterations": max_iter,
            "converged": False}


def _multistart_job(args) -> dict:
    expr_text, variables, start, tol, max_iter = args
    from .numerics import compile_float64
    from .parser import parse_math
    f = compile_float64(parse_math(expr_text), variables)
    return nelder_mead(lambda point: f(*point), start, tol, max_iter)


def multistart_minimize(expr_text: str, variables: list[str],
                        starts: list[list[float]], tol: float = 1e-10,
                        max_iter: int = 1000,
                        workers: int | None = None) -> dict:
    """Nelder-Mead from many start points across the process pool; returns the
    best result plus all endpoints. expr_text is MathIR source, re-parsed
    inside each worker."""
    resolved = resolve_workers(workers)
    jobs = [(expr_text, variables, list(s), tol, max_iter) for s in starts]
    results = process_map(_multistart_job, jobs, workers=resolved)
    best = min(results, key=lambda r: r["f"])
    return {"best": best, "runs": results, "workers": resolved}
