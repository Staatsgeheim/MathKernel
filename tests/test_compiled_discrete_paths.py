# =============================================================================
# MathKernel - optimized-path differential tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Differential tests for compiled paths against Python fallbacks."""
from __future__ import annotations

import pytest

from mathkernel import combinatorics, finite_algebra, finite_groups, graph_theory
from mathkernel.combinatorics import LinearRecurrence
from mathkernel.finite_algebra import create_finite_field, field_mul
from mathkernel.finite_groups import FiniteGroup
from mathkernel.graph_theory import Graph, breadth_first_search, connected_components
from mathkernel.models import OperationStatus


def test_graph_traversal_fast_path_matches_python_fallback(monkeypatch):
    graph = Graph(vertices=["a", "b", "c", "d"], edges=[
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
        {"source": "c", "target": "a"},
        {"source": "c", "target": "d"},
    ])
    fast_bfs = breadth_first_search(graph, "a")
    fast_components = connected_components(graph)
    monkeypatch.setattr(graph_theory, "_HAVE_GRAPH_NUMBA", False)
    slow_bfs = breadth_first_search(graph, "a")
    slow_components = connected_components(graph)

    assert fast_bfs.status == OperationStatus.VERIFIED
    assert slow_bfs.status == OperationStatus.VERIFIED
    assert fast_bfs.order == slow_bfs.order == ["a", "b", "c", "d"]
    assert fast_bfs.depth == slow_bfs.depth
    assert fast_components.count == slow_components.count == 1
    assert fast_components.components == slow_components.components
    backend = fast_bfs.evidence.computation[0].metadata.get("backend")
    assert backend in {"numba-csr", "python"}


def test_finite_field_multiplication_fast_path_matches_fallback(monkeypatch):
    field = create_finite_field(2, [1, 1, 0, 1, 1, 0, 0, 0, 1]).value
    a = [1, 0, 1, 1, 0, 0, 1, 1]
    b = [1, 1, 0, 0, 1, 0, 1, 0]
    fast = field_mul(field, a, b)
    monkeypatch.setattr(finite_algebra, "_poly_mulmod_fast", None)
    slow = field_mul(field, a, b)

    assert fast.status == OperationStatus.VERIFIED
    assert slow.status == OperationStatus.VERIFIED
    assert fast.value == slow.value
    assert fast.value.coeffs == slow.value.coeffs


def test_recurrence_extension_fast_path_matches_bigint_fallback(monkeypatch):
    recurrence = LinearRecurrence(coefficients=[1, 1], initial=[0, 1])
    fast = combinatorics._extend_from_recurrence(recurrence, 80)
    monkeypatch.setattr(combinatorics, "_extend_fast", None)
    slow = combinatorics._extend_from_recurrence(recurrence, 80)
    assert fast == slow
    assert fast[-1] == 23416728348467685

    # Beyond the conservative int64 fragment the fallback must stay exact.
    assert combinatorics._extend_from_recurrence(recurrence, 200)[200] == \
        280571172992510140037611932413038677189525


def test_cayley_validation_fast_path_matches_fallback(monkeypatch):
    group = FiniteGroup.cyclic(12)
    fast_inverses = list(group.inverses)
    monkeypatch.setattr(finite_groups, "_validate_cayley_fast", None)
    slow = FiniteGroup.cyclic(12)
    assert slow.inverses == fast_inverses

    table = [[(i + j) % 3 for j in range(3)] for i in range(3)]
    table[1][2] = 1  # row 1 is no longer a permutation
    with pytest.raises(ValueError, match="row 1 is not a permutation"):
        FiniteGroup(order=3, identity=0, cayley_table=table)
