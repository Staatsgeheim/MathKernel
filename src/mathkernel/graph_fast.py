# =============================================================================
# MathKernel - Numba-compiled CSR kernels for exact graph traversals
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Compiled exact graph kernels over compressed sparse row adjacency.

These kernels accelerate only index-based traversal over an already validated
``_View``: neighbour order is the deterministic sorted order built by the
caller, and every compiled result is still re-verified by the pure-Python
certificate checkers in ``graph_theory.py``. If numba is unavailable, or a
compiled result fails verification, callers fall back to the reference Python
implementation. No floating-point arithmetic is used.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


def _to_csr(adjacency: list[list[tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``[(neighbour, edge_id), ...]`` adjacency to sorted CSR."""
    n = len(adjacency)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    if n:
        np.cumsum(np.fromiter(
            (len(neighbours) for neighbours in adjacency),
            dtype=np.int64, count=n,
        ), out=indptr[1:])
    total = int(indptr[-1])
    indices = np.fromiter(
        (v for neighbours in adjacency for v, _ in neighbours),
        dtype=np.int64, count=total,
    )
    return indptr, indices


if HAVE_NUMBA:

    @njit(cache=True)
    def _bfs_kernel(indptr, indices, source):
        n = indptr.shape[0] - 1
        depth = np.full(n, -1, dtype=np.int64)
        parent = np.full(n, -1, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        queue = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        count = 0
        depth[source] = 0
        queue[tail] = source
        tail += 1
        while head < tail:
            u = queue[head]
            head += 1
            order[count] = u
            count += 1
            for p in range(indptr[u], indptr[u + 1]):
                v = indices[p]
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue[tail] = v
                    tail += 1
        return order, depth, parent, count

    @njit(cache=True)
    def _components_kernel(indptr, indices):
        n = indptr.shape[0] - 1
        comp = np.full(n, -1, dtype=np.int64)
        parent = np.full(n, -1, dtype=np.int64)
        queue = np.empty(n, dtype=np.int64)
        ncomp = 0
        for s in range(n):
            if comp[s] >= 0:
                continue
            head = 0
            tail = 0
            comp[s] = ncomp
            queue[tail] = s
            tail += 1
            while head < tail:
                u = queue[head]
                head += 1
                for p in range(indptr[u], indptr[u + 1]):
                    v = indices[p]
                    if comp[v] < 0:
                        comp[v] = ncomp
                        parent[v] = u
                        queue[tail] = v
                        tail += 1
            ncomp += 1
        return comp, parent, ncomp


def bfs_csr(
    adjacency: list[list[tuple[int, int]]], source: int,
) -> tuple[list[int], list[int], list[int]] | None:
    """BFS ``(order, depth, parent)`` on sorted CSR adjacency, if compiled."""
    if not HAVE_NUMBA:
        return None
    n = len(adjacency)
    if not 0 <= source < n:
        raise ValueError(f"unknown vertex index: {source}")
    indptr, indices = _to_csr(adjacency)
    order, depth, parent, count = _bfs_kernel(indptr, indices, source)
    return (
        order[:count].tolist(),
        depth.tolist(),
        parent.tolist(),
    )


def components_csr(
    adjacency: list[list[tuple[int, int]]],
) -> tuple[list[int], list[int], int] | None:
    """Connected components ``(component_id, parent, count)``, if compiled."""
    if not HAVE_NUMBA:
        return None
    indptr, indices = _to_csr(adjacency)
    comp, parent, ncomp = _components_kernel(indptr, indices)
    return comp.tolist(), parent.tolist(), int(ncomp)
