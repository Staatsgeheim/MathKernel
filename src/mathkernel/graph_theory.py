# =============================================================================
# MathKernel - exact graph theory core (Stage C)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact, deterministic graph algorithms with machine-checkable certificates.

Conventions
-----------
* Vertex labels are exact and JSON-safe: ``str``, ``int`` or ``bool``.  They
  compare type-aware, so ``1`` and ``True`` are distinct vertices.
* Weights and capacities are exact ``Fraction`` values.  Floats and booleans
  are rejected at the model boundary because they are not exact rationals.
* Every algorithm is deterministic: vertices and neighbours are processed in
  sorted label order and ties break toward the smaller label.
* Results carry an ``EvidenceBundle``.  A result reports
  ``OperationStatus.VERIFIED`` only when its certificate re-verified exactly;
  conservative reconciliation caps trust at the weakest evidence, so a
  fabricated or tampered certificate cannot launder trust.
* NP-hard objectives use ``OptimalityStatus``: ``OPTIMUM`` only with an exact
  proof, ``CANDIDATE`` for feasible-but-unproven bounds, ``IMPOSSIBLE`` only
  for exact infeasibility proofs, ``UNKNOWN`` when a search budget was
  exhausted.  A heuristic that finds nothing never implies impossibility.
"""
from __future__ import annotations

import heapq
from collections import deque
from enum import Enum
from fractions import Fraction
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from mathkernel_artifacts import (
    TRUST_RANK,
    CertificateEvidence,
    ComputationEvidence,
    EvidenceBundle,
    ProofEvidence,
)

from .models import OperationStatus, TrustLevel

try:
    from .graph_fast import HAVE_NUMBA as _HAVE_GRAPH_NUMBA
    from .graph_fast import bfs_csr as _bfs_csr
    from .graph_fast import components_csr as _components_csr
except ImportError:  # pragma: no cover
    _HAVE_GRAPH_NUMBA = False
    _bfs_csr = None
    _components_csr = None

Label = str | int | bool

__all__ = [
    "Label", "GraphEdge", "Graph", "DirectedGraph", "WeightedGraph", "MultiGraph",
    "SearchLimits", "OptimalityStatus", "OperationStatus", "TrustLevel",
    "GraphResult", "TraversalResult", "ComponentsResult", "ShortestPathResult",
    "SpanningTreeResult", "EdgeFlow", "FlowVerification", "MaxFlowResult",
    "MatchingResult", "EulerResult", "ColoringResult", "TopologicalSortResult",
    "CycleResult", "CentralityResult", "IsomorphismResult",
    "verify_graph", "breadth_first_search", "depth_first_search",
    "connected_components",
    "strongly_connected_components", "shortest_paths", "minimum_spanning_tree",
    "maximum_flow", "verify_maximum_flow", "bipartite_matching",
    "verify_matching", "euler_trail", "graph_coloring", "verify_coloring",
    "topological_sort", "find_cycle", "degree_centrality",
    "graph_isomorphism", "verify_isomorphism",
]


# ---------------------------------------------------------------------------
# Exact JSON-safe values
# ---------------------------------------------------------------------------

def _sort_key(label: Label) -> tuple:
    """Total, type-aware order on labels: bool < int < str."""
    if isinstance(label, bool):
        return (0, int(label))
    if isinstance(label, int):
        return (1, label)
    if isinstance(label, str):
        return (2, label)
    raise TypeError(
        f"graph labels must be JSON-safe (str, int, bool), got {type(label).__name__}")


def _json_exact(value: Any) -> Any:
    """Serialize exact values without executing or importing anything."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        items = sorted(
            value.items(),
            key=lambda kv: _sort_key(kv[0])
            if isinstance(kv[0], (str, int, bool)) else (9, str(kv[0])),
        )
        return {str(key): _json_exact(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_json_exact(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_exact(item) for item in sorted(value, key=repr)]
    return value


def _exact_rational(cls: type, value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(
            "edge weights must be exact rationals (int, str, or Fraction); "
            "bool and float are rejected as inexact")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"edge weights must be exact rationals, got {type(value).__name__}")


def _json_safe_label(cls: type, value: Any) -> Label:
    if not isinstance(value, (str, int, bool)):
        raise TypeError(
            f"graph labels must be JSON-safe (str, int, bool), got {type(value).__name__}")
    return value


class GraphModel(BaseModel):
    """Base model with exact JSON serialization (Fractions render as strings)."""

    @model_serializer(mode="plain", when_used="json")
    def serialize_exact_model(self) -> dict[str, Any]:
        return {
            key: _json_exact(value)
            for key, value in self.__dict__.items()
        }


# ---------------------------------------------------------------------------
# Graph input models
# ---------------------------------------------------------------------------

class GraphEdge(GraphModel):
    source: Label
    target: Label
    weight: Fraction = Fraction(1)

    _label = field_validator("source", "target", mode="before")(_json_safe_label)
    _weight = field_validator("weight", mode="before")(_exact_rational)


def _check_endpoints(vertices: list[Label], edges: list[GraphEdge]) -> None:
    known = {_sort_key(v) for v in vertices}
    for edge in edges:
        if _sort_key(edge.source) not in known or _sort_key(edge.target) not in known:
            raise ValueError(
                f"edge ({edge.source!r}, {edge.target!r}) references an unknown vertex")


def _check_vertices(cls: type, vertices: list[Label]) -> list[Label]:
    keys = [_sort_key(v) for v in vertices]
    if len(set(keys)) != len(keys):
        raise ValueError(
            "vertices must be distinct (labels compare type-aware: 1 and True differ)")
    return vertices


class Graph(GraphModel):
    """Undirected simple graph: no self-loops, no parallel edges, unweighted."""

    vertices: list[Label]
    edges: list[GraphEdge] = Field(default_factory=list)

    _vertices = field_validator("vertices")(_check_vertices)

    @model_validator(mode="after")
    def _simple_undirected(self) -> "Graph":
        _check_endpoints(self.vertices, self.edges)
        seen: set[tuple] = set()
        for edge in self.edges:
            a, b = _sort_key(edge.source), _sort_key(edge.target)
            if a == b:
                raise ValueError("self-loops are not allowed in a simple Graph")
            key = (a, b) if a < b else (b, a)
            if key in seen:
                raise ValueError("parallel edges are not allowed in a simple Graph")
            seen.add(key)
            if edge.weight != 1:
                raise ValueError("Graph is unweighted; use WeightedGraph for weights")
        return self


class DirectedGraph(GraphModel):
    """Directed simple graph: no self-loops, no parallel arcs (anti-parallel ok)."""

    vertices: list[Label]
    edges: list[GraphEdge] = Field(default_factory=list)

    _vertices = field_validator("vertices")(_check_vertices)

    @model_validator(mode="after")
    def _simple_directed(self) -> "DirectedGraph":
        _check_endpoints(self.vertices, self.edges)
        seen: set[tuple] = set()
        for edge in self.edges:
            a, b = _sort_key(edge.source), _sort_key(edge.target)
            if a == b:
                raise ValueError("self-loops are not allowed in a simple DirectedGraph")
            if (a, b) in seen:
                raise ValueError("parallel arcs are not allowed in a simple DirectedGraph")
            seen.add((a, b))
            if edge.weight != 1:
                raise ValueError("DirectedGraph is unweighted; use WeightedGraph")
        return self


class WeightedGraph(GraphModel):
    """Simple graph with exact Fraction weights; directed iff ``directed``."""

    vertices: list[Label]
    edges: list[GraphEdge] = Field(default_factory=list)
    directed: bool = False

    _vertices = field_validator("vertices")(_check_vertices)

    @model_validator(mode="after")
    def _simple_weighted(self) -> "WeightedGraph":
        _check_endpoints(self.vertices, self.edges)
        seen: set[tuple] = set()
        for edge in self.edges:
            a, b = _sort_key(edge.source), _sort_key(edge.target)
            if a == b:
                raise ValueError("self-loops are not allowed in a simple WeightedGraph")
            key = (a, b) if self.directed or a < b else (b, a)
            if key in seen:
                raise ValueError("parallel edges are not allowed in a simple WeightedGraph")
            seen.add(key)
        return self


class MultiGraph(GraphModel):
    """Undirected multigraph: parallel edges and self-loops are allowed."""

    vertices: list[Label]
    edges: list[GraphEdge] = Field(default_factory=list)

    _vertices = field_validator("vertices")(_check_vertices)

    @model_validator(mode="after")
    def _multigraph(self) -> "MultiGraph":
        _check_endpoints(self.vertices, self.edges)
        for edge in self.edges:
            if edge.weight != 1:
                raise ValueError("MultiGraph is unweighted; use WeightedGraph")
        return self


class SearchLimits(GraphModel):
    """Resource budget for the exponential-time exact searches."""

    max_vertices: int = Field(default=256, ge=1)
    max_operations: int = Field(default=100_000, ge=1)
    exact_coloring_max_vertices: int = Field(default=24, ge=0)


class OptimalityStatus(str, Enum):
    """Optimality semantics for objectives whose exact solution is NP-hard."""

    CANDIDATE = "candidate"      # feasible solution; optimality not established
    VERIFIED = "verified"        # feasibility verified exactly; optimality open
    OPTIMUM = "optimum"          # feasible and proven optimal exactly
    IMPOSSIBLE = "impossible"    # infeasibility proven exactly
    UNKNOWN = "unknown"          # budget exhausted; no conclusion either way


class _BudgetExceeded(Exception):
    """Internal control flow: search budget exhausted.  Never a verdict."""


class _Budget:
    def __init__(self, operations: int):
        self.remaining = operations

    def spend(self, count: int = 1) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise _BudgetExceeded


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class GraphResult(GraphModel):
    """Base result: status, conservative trust, evidence, diagnostics."""

    status: OperationStatus
    trust: TrustLevel = TrustLevel.UNKNOWN
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _conservative_reconciliation(self) -> "GraphResult":
        if self.evidence.is_empty():
            if TRUST_RANK[self.trust.value] > TRUST_RANK[TrustLevel.UNKNOWN.value]:
                self.trust = TrustLevel.UNKNOWN
                self.diagnostics.append("trust downgraded: no evidence attached")
        else:
            supported = self.evidence.conservative_trust()
            if TRUST_RANK[supported] < TRUST_RANK[self.trust.value]:
                self.trust = TrustLevel(supported)
                self.diagnostics.append("trust downgraded to weakest attached evidence")
        if (
            self.status == OperationStatus.VERIFIED
            and TRUST_RANK[self.trust.value] < TRUST_RANK[TrustLevel.EXACT.value]
        ):
            self.status = OperationStatus.CANDIDATE
            self.diagnostics.append("status downgraded: VERIFIED requires exact trust")
        return self


class TraversalResult(GraphResult):
    source: Label
    order: list[Label] = Field(default_factory=list)
    parent: dict[Label, Label | None] = Field(default_factory=dict)
    depth: dict[Label, int] = Field(default_factory=dict)


class ComponentsResult(GraphResult):
    components: list[list[Label]] = Field(default_factory=list)
    count: int = 0
    component_index: dict[Label, int] = Field(default_factory=dict)


class ShortestPathResult(GraphResult):
    source: Label
    target: Label | None = None
    algorithm: str = ""
    distances: dict[Label, Fraction] = Field(default_factory=dict)
    predecessors: dict[Label, Label | None] = Field(default_factory=dict)
    path: list[Label] | None = None
    reachable: bool | None = None
    negative_cycle: list[Label] | None = None


class SpanningTreeResult(GraphResult):
    edges: list[GraphEdge] = Field(default_factory=list)
    total_weight: Fraction | None = None
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN


class EdgeFlow(GraphModel):
    source: Label
    target: Label
    flow: Fraction

    _label = field_validator("source", "target", mode="before")(_json_safe_label)
    _flow = field_validator("flow", mode="before")(_exact_rational)


class FlowVerification(GraphModel):
    """Independent exact re-check of a candidate maximum flow."""

    capacity_respected: bool = False
    conservation_holds: bool = False
    cut_valid: bool = False
    value_matches_cut: bool = False
    flow_value: Fraction | None = None
    cut_capacity: Fraction | None = None
    accepted: bool = False
    diagnostics: list[str] = Field(default_factory=list)


class MaxFlowResult(GraphResult):
    flow_value: Fraction | None = None
    flows: list[EdgeFlow] = Field(default_factory=list)
    cut: tuple[list[Label], list[Label]] | None = None
    verification: FlowVerification | None = None
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN

    @model_validator(mode="after")
    def _optimality_requires_accepted_verification(self) -> "MaxFlowResult":
        if self.optimality == OptimalityStatus.OPTIMUM:
            proven = (
                self.verification is not None
                and self.verification.accepted
                and any(
                    item.verified and item.certificate_type == "max_flow_min_cut"
                    for item in self.evidence.certificate
                )
            )
            if not proven:
                self.optimality = OptimalityStatus.UNKNOWN
                self.diagnostics.append(
                    "optimality downgraded: no accepted max-flow/min-cut verification")
        return self


class MatchingResult(GraphResult):
    bipartition: tuple[list[Label], list[Label]] | None = None
    matching: list[tuple[Label, Label]] = Field(default_factory=list)
    size: int = 0
    vertex_cover: list[Label] | None = None
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN
    odd_cycle: list[Label] | None = None

    @model_validator(mode="after")
    def _optimality_requires_konig_certificate(self) -> "MatchingResult":
        if self.optimality == OptimalityStatus.OPTIMUM:
            proven = any(
                item.verified and item.certificate_type == "konig_cover"
                for item in self.evidence.certificate
            )
            if not proven:
                self.optimality = OptimalityStatus.UNKNOWN
                self.diagnostics.append(
                    "optimality downgraded: no verified Kőnig cover certificate")
        return self


class EulerResult(GraphResult):
    kind: Literal["cycle", "path"] | None = None
    trail: list[Label] = Field(default_factory=list)
    edge_ids: list[int] = Field(default_factory=list)


class ColoringResult(GraphResult):
    coloring: dict[Label, int] | None = None
    num_colors: int | None = None
    chromatic_number: int | None = None
    lower_bound: int = 0
    upper_bound: int | None = None
    optimality: OptimalityStatus = OptimalityStatus.UNKNOWN

    @model_validator(mode="after")
    def _optimality_requires_exact_proof(self) -> "ColoringResult":
        if self.optimality == OptimalityStatus.OPTIMUM:
            proven = any(
                item.verified and item.method in (
                    "exact_k_colorability", "clique_bound_meets_greedy",
                    "trivial_coloring")
                for item in self.evidence.proof
            )
            if not proven:
                self.optimality = OptimalityStatus.UNKNOWN
                self.chromatic_number = None
                self.diagnostics.append(
                    "optimality downgraded: no verified exact coloring proof")
        return self


class TopologicalSortResult(GraphResult):
    order: list[Label] | None = None
    cycle: list[Label] | None = None


class CycleResult(GraphResult):
    has_cycle: bool | None = None
    cycle: list[Label] | None = None


class CentralityResult(GraphResult):
    centrality: dict[Label, Fraction] = Field(default_factory=dict)
    in_centrality: dict[Label, Fraction] | None = None
    out_centrality: dict[Label, Fraction] | None = None


class IsomorphismResult(GraphResult):
    isomorphic: bool | None = None
    permutation: dict[Label, Label] | None = None
    search_nodes: int = 0

    @model_validator(mode="after")
    def _isomorphism_requires_permutation_certificate(self) -> "IsomorphismResult":
        if self.isomorphic is True:
            proven = any(
                item.verified and item.certificate_type == "isomorphism_permutation"
                for item in self.evidence.certificate
            )
            if not proven:
                self.isomorphic = None
                self.diagnostics.append(
                    "isomorphism downgraded: no verified permutation certificate")
        return self


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _computation(method: str, **metadata: Any) -> ComputationEvidence:
    return ComputationEvidence(
        engine="mathkernel",
        method=method,
        arithmetic="exact_rational",
        deterministic=True,
        trust="exact",
        metadata=metadata,
    )


def _certificate(certificate_type: str, claim: str, witness: Any,
                 verified: bool) -> CertificateEvidence:
    return CertificateEvidence(
        certificate_type=certificate_type,
        claim=claim,
        witness=_json_exact(witness),
        verifier="mathkernel.graph_theory.verify",
        verified=verified,
        trust="exact" if verified else "unknown",
    )


def _proof(proposition: str, method: str, verified: bool = True) -> ProofEvidence:
    return ProofEvidence(
        proposition=proposition,
        method=method,
        engine="mathkernel",
        verified=verified,
        trust="exact" if verified else "unknown",
    )


def verify_graph(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
) -> GraphResult:
    """Re-validate a graph value and attach a model-invariant certificate."""
    view = _View(graph)
    evidence = EvidenceBundle(
        computation=[_computation(
            "graph_model_validation", vertices=len(view), edges=len(view.edges))],
        certificate=[_certificate(
            "graph_model_invariants",
            "vertices are distinct, endpoints exist, and class invariants hold",
            {"graph_type": type(graph).__name__}, True)],
    )
    return GraphResult(
        status=OperationStatus.VERIFIED,
        trust=TrustLevel.EXACT,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Internal normalized view
# ---------------------------------------------------------------------------

class _View:
    """Index-based adjacency with deterministic (sorted) neighbour order."""

    __slots__ = ("labels", "index", "adj", "adjset", "edges", "directed")

    def __init__(self, graph: Graph | DirectedGraph | WeightedGraph | MultiGraph):
        self.directed = isinstance(graph, DirectedGraph) or (
            isinstance(graph, WeightedGraph) and graph.directed)
        self.labels = sorted(graph.vertices, key=_sort_key)
        self.index = {_sort_key(label): i for i, label in enumerate(self.labels)}
        self.edges: list[tuple[int, int, Fraction]] = []
        self.adj: list[list[tuple[int, int]]] = [[] for _ in self.labels]
        for eid, edge in enumerate(graph.edges):
            u = self.index[_sort_key(edge.source)]
            v = self.index[_sort_key(edge.target)]
            self.edges.append((u, v, edge.weight))
            self.adj[u].append((v, eid))
            if not self.directed:
                self.adj[v].append((u, eid))
        for neighbours in self.adj:
            neighbours.sort()
        self.adjset = [set() for _ in self.labels]
        for u in range(len(self.labels)):
            for v, _ in self.adj[u]:
                self.adjset[u].add(v)

    def __len__(self) -> int:
        return len(self.labels)


def _require_vertex(view: _View, label: Label) -> int:
    key = _sort_key(label)
    if key not in view.index:
        raise ValueError(f"unknown vertex label: {label!r}")
    return view.index[key]


def _require_undirected(view: _View, operation: str) -> None:
    if view.directed:
        raise ValueError(f"{operation} requires an undirected graph")


def _require_directed(view: _View, operation: str) -> None:
    if not view.directed:
        raise ValueError(f"{operation} requires a directed graph")


def _verify_cycle(view: _View, cycle: list[int]) -> bool:
    if len(cycle) < 2 or cycle[0] != cycle[-1]:
        return False
    for a, b in zip(cycle, cycle[1:]):
        if b not in view.adjset[a]:
            return False
    return True


# ---------------------------------------------------------------------------
# Traversals
# ---------------------------------------------------------------------------

def _verify_bfs_tree(view: _View, source: int, order: list[int],
                     depth: list[int], parent: list[int]) -> bool:
    """Verify a BFS certificate in linear time.

    Strictly increasing parent depths prove that the parent relation is an
    acyclic rooted tree, so walking every parent chain separately is both
    unnecessary and quadratic on path graphs.
    """
    n = len(view)
    if (
        not 0 <= source < n
        or len(depth) != n
        or len(parent) != n
        or not order
        or order[0] != source
        or len(set(order)) != len(order)
        or any(v < 0 or v >= n for v in order)
        or depth[source] != 0
        or parent[source] != -1
    ):
        return False
    reachable = {v for v, value in enumerate(depth) if value >= 0}
    if set(order) != reachable:
        return False
    if any(depth[left] > depth[right] for left, right in zip(order, order[1:])):
        return False
    for v in range(n):
        if depth[v] < 0:
            if parent[v] != -1:
                return False
            continue
        if v == source:
            continue
        p = parent[v]
        if p < 0 or p >= n or v not in view.adjset[p]:
            return False
        if depth[v] != depth[p] + 1:
            return False
    for u in range(len(view)):
        if depth[u] < 0:
            continue
        for v, _ in view.adj[u]:
            if depth[v] < 0 or depth[v] > depth[u] + 1:
                return False
    return True


def breadth_first_search(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
    source: Label,
) -> TraversalResult:
    """BFS from ``source``; depths are exact unweighted shortest distances."""
    view = _View(graph)
    s = _require_vertex(view, source)
    n = len(view)
    backend = "python"
    fast = _bfs_csr(view.adj, s) if _HAVE_GRAPH_NUMBA else None
    if fast is not None:
        order, depth, parent = fast
        verified = _verify_bfs_tree(view, s, order, depth, parent)
        backend = "numba-csr"
    else:
        verified = False
    if not verified:
        depth = [-1] * n
        parent = [-1] * n
        order = []
        depth[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            u = queue.popleft()
            order.append(u)
            for v, _ in view.adj[u]:
                if depth[v] < 0:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
        verified = _verify_bfs_tree(view, s, order, depth, parent)
        backend = "python"
    parent_map = {
        view.labels[v]: (view.labels[parent[v]] if parent[v] >= 0 else None)
        for v in range(n) if depth[v] >= 0
    }
    evidence = EvidenceBundle(
        computation=[_computation(
            "breadth_first_search", vertices=n, backend=backend)],
        certificate=[_certificate(
            "bfs_tree",
            "depths are exact shortest-path distances from the source",
            {"parent": parent_map}, verified)],
    )
    return TraversalResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        source=source,
        order=[view.labels[v] for v in order],
        parent=parent_map,
        depth={view.labels[v]: depth[v] for v in range(n) if depth[v] >= 0},
    )


def _verify_dfs_tree(view: _View, source: int, order: list[int],
                     parent: list[int]) -> bool:
    if not order or order[0] != source or len(set(order)) != len(order):
        return False
    position = {v: i for i, v in enumerate(order)}
    for v in order[1:]:
        p = parent[v]
        if p < 0 or p not in position or position[p] > position[v]:
            return False
        if v not in view.adjset[p]:
            return False
    return True


def depth_first_search(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
    source: Label,
) -> TraversalResult:
    """Iterative DFS from ``source`` in deterministic sorted-neighbour order."""
    view = _View(graph)
    s = _require_vertex(view, source)
    visited = [False] * len(view)
    parent = [-1] * len(view)
    order = [s]
    visited[s] = True
    stack = [(s, 0)]
    while stack:
        u, i = stack[-1]
        if i < len(view.adj[u]):
            v, _ = view.adj[u][i]
            stack[-1] = (u, i + 1)
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                order.append(v)
                stack.append((v, 0))
        else:
            stack.pop()
    verified = _verify_dfs_tree(view, s, order, parent)
    evidence = EvidenceBundle(
        computation=[_computation("depth_first_search", vertices=len(view))],
        certificate=[_certificate(
            "dfs_tree",
            "parents form a valid rooted search tree consistent with the order",
            {"order": [view.labels[v] for v in order]}, verified)],
    )
    return TraversalResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        source=source,
        order=[view.labels[v] for v in order],
        parent={
            view.labels[v]: (view.labels[parent[v]] if parent[v] >= 0 else None)
            for v in order
        },
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def _verify_components(view: _View, comp: list[int], parent: list[int],
                       ncomp: int) -> bool:
    """Partition + spanning forest + no crossing edges = exact components."""
    n = len(view)
    if len(comp) != n or len(parent) != n or ncomp < 0:
        return False
    if n == 0:
        return ncomp == 0
    if set(comp) != set(range(ncomp)):
        return False

    # Resolve each parent forest once.  The three-state walk rejects cycles
    # and records a root for every vertex in O(V), even for long path trees.
    state = [0] * n
    root = [-1] * n
    for start in range(n):
        if state[start] == 2:
            continue
        path: list[int] = []
        current = start
        while current != -1 and state[current] == 0:
            state[current] = 1
            path.append(current)
            p = parent[current]
            if p < -1 or p >= n:
                return False
            if p != -1:
                if current not in view.adjset[p] or comp[current] != comp[p]:
                    return False
            current = p
        if current != -1 and state[current] == 1:
            return False
        resolved_root = path[-1] if current == -1 else root[current]
        if resolved_root < 0:
            return False
        for vertex in reversed(path):
            root[vertex] = resolved_root
            state[vertex] = 2

    component_roots: dict[int, int] = {}
    for v in range(n):
        r = root[v]
        if parent[r] != -1 or comp[r] != comp[v]:
            return False
        existing = component_roots.setdefault(comp[v], r)
        if existing != r:
            return False
    for u in range(n):
        for v, _ in view.adj[u]:
            if comp[u] != comp[v]:
                return False
    return True


def connected_components(
    graph: Graph | WeightedGraph | MultiGraph,
) -> ComponentsResult:
    """Connected components of an undirected graph, exactly certified."""
    view = _View(graph)
    _require_undirected(view, "connected_components")
    n = len(view)
    backend = "python"
    fast = _components_csr(view.adj) if _HAVE_GRAPH_NUMBA else None
    if fast is not None:
        comp, parent, ncomp = fast
        verified = _verify_components(view, comp, parent, ncomp)
        backend = "numba-csr"
    else:
        verified = False
    if not verified:
        comp = [-1] * n
        parent = [-1] * n
        ncomp = 0
        for s in range(n):
            if comp[s] >= 0:
                continue
            comp[s] = ncomp
            queue: deque[int] = deque([s])
            while queue:
                u = queue.popleft()
                for v, _ in view.adj[u]:
                    if comp[v] < 0:
                        comp[v] = ncomp
                        parent[v] = u
                        queue.append(v)
            ncomp += 1
        verified = _verify_components(view, comp, parent, ncomp)
        backend = "python"
    components = [
        [view.labels[i] for i in range(n) if comp[i] == c] for c in range(ncomp)
    ]
    evidence = EvidenceBundle(
        computation=[_computation(
            "connected_components", vertices=n, backend=backend)],
        certificate=[_certificate(
            "component_forest",
            "components partition the vertices, are internally connected, "
            "and no edge crosses components",
            {"components": components}, verified)],
    )
    return ComponentsResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        components=components,
        count=ncomp,
        component_index={view.labels[i]: comp[i] for i in range(n)},
    )


def _verify_scc(view: _View, radj: list[list[tuple[int, int]]],
                comp: list[int], ncomp: int) -> bool:
    """Strong connectivity per class + acyclic condensation = exact SCCs."""
    n = len(view)
    if any(c < 0 or c >= ncomp for c in comp):
        return False
    members: list[list[int]] = [[] for _ in range(ncomp)]
    for v, c in enumerate(comp):
        members[c].append(v)
    for c in range(ncomp):
        root = members[c][0]
        for neighbours in (view.adj, radj):
            seen = {root}
            stack = [root]
            while stack:
                u = stack.pop()
                for v, _ in neighbours[u]:
                    if comp[v] == c and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if len(seen) != len(members[c]):
                return False
    for u in range(n):
        for v, _ in view.adj[u]:
            if comp[u] != comp[v] and not comp[u] < comp[v]:
                return False
    return True


def strongly_connected_components(digraph: DirectedGraph | WeightedGraph) -> ComponentsResult:
    """SCCs via iterative Kosaraju; ids are a condensation topological order."""
    view = _View(digraph)
    _require_directed(view, "strongly_connected_components")
    n = len(view)
    visited = [False] * n
    finish: list[int] = []
    for s in range(n):
        if visited[s]:
            continue
        visited[s] = True
        stack = [(s, 0)]
        while stack:
            u, i = stack[-1]
            if i < len(view.adj[u]):
                v, _ = view.adj[u][i]
                stack[-1] = (u, i + 1)
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, 0))
            else:
                finish.append(u)
                stack.pop()
    radj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u in range(n):
        for v, eid in view.adj[u]:
            radj[v].append((u, eid))
    for neighbours in radj:
        neighbours.sort()
    comp = [-1] * n
    ncomp = 0
    for s in reversed(finish):
        if comp[s] >= 0:
            continue
        comp[s] = ncomp
        stack = [s]
        while stack:
            u = stack.pop()
            for v, _ in radj[u]:
                if comp[v] < 0:
                    comp[v] = ncomp
                    stack.append(v)
        ncomp += 1
    verified = _verify_scc(view, radj, comp, ncomp)
    components = [
        [view.labels[i] for i in range(n) if comp[i] == c] for c in range(ncomp)
    ]
    evidence = EvidenceBundle(
        computation=[_computation("strongly_connected_components", vertices=n)],
        certificate=[_certificate(
            "scc_reachability",
            "each class is mutually reachable and the condensation order is acyclic",
            {"components": components}, verified)],
    )
    return ComponentsResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        components=components,
        count=ncomp,
        component_index={view.labels[i]: comp[i] for i in range(n)},
    )


# ---------------------------------------------------------------------------
# Shortest paths
# ---------------------------------------------------------------------------

def _arcs(view: _View) -> list[tuple[int, int, Fraction]]:
    arcs = [(u, v, w) for u, v, w in view.edges]
    if not view.directed:
        arcs += [(v, u, w) for u, v, w in view.edges]
    return arcs


def _verify_shortest(view: _View, source: int, dist: list[Fraction | None],
                     pred: list[int], arcs: list[tuple[int, int, Fraction]]) -> bool:
    """No improving arc + achieving predecessors = exact shortest distances."""
    if dist[source] != 0:
        return False
    for u, v, w in arcs:
        if dist[u] is not None and (dist[v] is None or dist[v] > dist[u] + w):
            return False
    for v in range(len(view)):
        if v == source or dist[v] is None:
            continue
        p = pred[v]
        if p < 0 or dist[p] is None:
            return False
        if not any(a == p and b == v and dist[p] + w == dist[v] for a, b, w in arcs):
            return False
    return True


def _verify_negative_cycle(view: _View, cycle: list[int]) -> bool:
    weight: dict[tuple[int, int], Fraction] = {}
    for u, v, w in _arcs(view):
        weight[(u, v)] = w
    total = Fraction(0)
    for a, b in zip(cycle, cycle[1:]):
        if (a, b) not in weight:
            return False
        total += weight[(a, b)]
    return total < 0


def shortest_paths(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
    source: Label,
    target: Label | None = None,
) -> ShortestPathResult:
    """Exact single-source shortest paths.

    Unweighted graphs use BFS; nonnegative weights use exact Dijkstra over
    Fractions; negative weights use Bellman-Ford with exact negative-cycle
    detection (a reachable negative cycle means no shortest path exists).
    """
    view = _View(graph)
    s = _require_vertex(view, source)
    t = _require_vertex(view, target) if target is not None else None
    n = len(view)
    weighted = isinstance(graph, WeightedGraph)
    arcs = _arcs(view)
    negative = weighted and any(w < 0 for _, _, w in arcs)

    dist: list[Fraction | None] = [None] * n
    pred = [-1] * n
    dist[s] = Fraction(0)
    verification_arcs = arcs

    if not weighted:
        algorithm = "bfs"
        queue: deque[int] = deque([s])
        while queue:
            u = queue.popleft()
            for v, _ in view.adj[u]:
                if dist[v] is None:
                    dist[v] = dist[u] + 1
                    pred[v] = u
                    queue.append(v)
    elif not negative:
        algorithm = "dijkstra_exact"
        heap: list[tuple[Fraction, int]] = [(Fraction(0), s)]
        done = [False] * n
        while heap:
            d, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            for v, eid in view.adj[u]:
                w = view.edges[eid][2]
                candidate = d + w
                if dist[v] is None or candidate < dist[v]:
                    dist[v] = candidate
                    pred[v] = u
                    heapq.heappush(heap, (candidate, v))
    else:
        algorithm = "bellman_ford_exact"
        problem_arcs = arcs
        if t is not None:
            # Restrict Bellman-Ford to vertices that can reach the requested
            # target. A negative cycle outside this reverse-reachable region
            # cannot improve any s->t path and must not invalidate it.
            reverse: list[list[int]] = [[] for _ in range(n)]
            for u, v, _ in arcs:
                reverse[v].append(u)
            can_reach_target = {t}
            queue = deque([t])
            while queue:
                v = queue.popleft()
                for u in reverse[v]:
                    if u not in can_reach_target:
                        can_reach_target.add(u)
                        queue.append(u)
            problem_arcs = [
                (u, v, w) for u, v, w in arcs
                if u in can_reach_target and v in can_reach_target
            ]
        verification_arcs = problem_arcs
        for _ in range(n - 1):
            changed = False
            for u, v, w in problem_arcs:
                if dist[u] is not None and (dist[v] is None or dist[u] + w < dist[v]):
                    dist[v] = dist[u] + w
                    pred[v] = u
                    changed = True
            if not changed:
                break
        for u, v, w in problem_arcs:
            if dist[u] is not None and (dist[v] is None or dist[u] + w < dist[v]):
                x = v
                for _ in range(n):
                    x = pred[x]
                    if x < 0:
                        raise RuntimeError("corrupted predecessor chain")
                cycle = [x]
                y = pred[x]
                while y != x:
                    cycle.append(y)
                    y = pred[y]
                cycle.append(x)
                cycle.reverse()
                verified = _verify_negative_cycle(view, cycle)
                cycle_labels = [view.labels[i] for i in cycle]
                evidence = EvidenceBundle(
                    computation=[_computation(algorithm, vertices=n)],
                    certificate=[_certificate(
                        "negative_cycle",
                        "a reachable cycle of negative total weight; "
                        "no shortest path exists",
                        {"cycle": cycle_labels}, verified)],
                )
                return ShortestPathResult(
                    status=(OperationStatus.DOES_NOT_EXIST if verified
                            else OperationStatus.ERROR),
                    trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
                    evidence=evidence,
                    diagnostics=["reachable negative-weight cycle detected"],
                    source=source,
                    target=target,
                    algorithm=algorithm,
                    negative_cycle=cycle_labels,
                )

    verified = _verify_shortest(view, s, dist, pred, verification_arcs)
    path: list[Label] | None = None
    reachable: bool | None = None
    status = OperationStatus.VERIFIED if verified else OperationStatus.ERROR
    if t is not None:
        reachable = dist[t] is not None
        if reachable:
            walk = [t]
            while walk[-1] != s:
                walk.append(pred[walk[-1]])
            walk.reverse()
            path = [view.labels[i] for i in walk]
        elif verified:
            status = OperationStatus.DOES_NOT_EXIST
    evidence = EvidenceBundle(
        computation=[_computation(algorithm, vertices=n)],
        certificate=[_certificate(
            "shortest_path_optimality",
            "no arc improves a distance and every distance is achieved "
            "by a predecessor arc",
            {"source": source},
            verified)],
    )
    if t is not None and not reachable and verified:
        evidence.proof.append(_proof(
            f"no path from {source!r} to {target!r} exists",
            "exhaustive_relaxation"))
    return ShortestPathResult(
        status=status,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        source=source,
        target=target,
        algorithm=algorithm,
        distances={view.labels[i]: dist[i] for i in range(n) if dist[i] is not None},
        predecessors={
            view.labels[i]: (view.labels[pred[i]] if pred[i] >= 0 else None)
            for i in range(n) if dist[i] is not None
        },
        path=path,
        reachable=reachable,
    )


# ---------------------------------------------------------------------------
# Minimum spanning tree
# ---------------------------------------------------------------------------

class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _tree_path_max(tadj: list[list[tuple[int, Fraction]]], u: int, v: int):
    prev: dict[int, tuple[int | None, Fraction | None]] = {u: (None, None)}
    queue: deque[int] = deque([u])
    while queue:
        x = queue.popleft()
        if x == v:
            break
        for y, w in tadj[x]:
            if y not in prev:
                prev[y] = (x, w)
                queue.append(y)
    if v not in prev:
        return None
    best: Fraction | None = None
    x = v
    while prev[x][0] is not None:
        p, w = prev[x]
        if best is None or w > best:
            best = w
        x = p
    return best


def _verify_mst(view: _View, selected: list[int]) -> bool:
    """Spanning tree + cycle property (every non-tree edge is a maximum on
    its fundamental cycle) = exact minimum spanning tree."""
    n = len(view)
    if n == 0:
        return not selected
    if len(selected) != n - 1:
        return False
    dsu = _DSU(n)
    tadj: list[list[tuple[int, Fraction]]] = [[] for _ in range(n)]
    chosen = set(selected)
    for eid in selected:
        u, v, w = view.edges[eid]
        if not dsu.union(u, v):
            return False
        tadj[u].append((v, w))
        tadj[v].append((u, w))
    if len({dsu.find(v) for v in range(n)}) != 1:
        return False
    for eid, (u, v, w) in enumerate(view.edges):
        if eid in chosen:
            continue
        path_max = _tree_path_max(tadj, u, v)
        if path_max is None or path_max > w:
            return False
    return True


def minimum_spanning_tree(graph: WeightedGraph) -> SpanningTreeResult:
    """Kruskal MST with exact Fraction weights and a cycle-property certificate."""
    if not isinstance(graph, WeightedGraph):
        raise ValueError("minimum_spanning_tree requires a WeightedGraph")
    view = _View(graph)
    _require_undirected(view, "minimum_spanning_tree")
    n = len(view)
    dsu = _DSU(n)
    order = sorted(
        range(len(view.edges)),
        key=lambda i: (view.edges[i][2], view.edges[i][0], view.edges[i][1]),
    )
    selected: list[int] = []
    for eid in order:
        u, v, _ = view.edges[eid]
        if dsu.union(u, v):
            selected.append(eid)
    if n > 0 and len(selected) != n - 1:
        comp = connected_components(graph)
        evidence = EvidenceBundle(
            computation=[_computation("kruskal_mst", vertices=n)],
            certificate=comp.evidence.certificate,
            proof=[_proof(
                "no spanning tree exists: the graph is disconnected",
                "component_count")],
        )
        return SpanningTreeResult(
            status=OperationStatus.DOES_NOT_EXIST,
            trust=TrustLevel.EXACT,
            evidence=evidence,
            diagnostics=["graph is disconnected; no spanning tree exists"],
            optimality=OptimalityStatus.IMPOSSIBLE,
        )
    verified = _verify_mst(view, selected)
    tree_edges = [
        GraphEdge(source=view.labels[u], target=view.labels[v], weight=w)
        for eid in selected for (u, v, w) in [view.edges[eid]]
    ]
    total = sum((view.edges[eid][2] for eid in selected), Fraction(0))
    evidence = EvidenceBundle(
        computation=[_computation("kruskal_mst", vertices=n)],
        certificate=[_certificate(
            "mst_cycle_property",
            "spanning tree whose every non-tree edge is a maximum-weight "
            "edge on its fundamental cycle",
            {"total_weight": total}, verified)],
    )
    return SpanningTreeResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        edges=tree_edges,
        total_weight=total,
        optimality=OptimalityStatus.OPTIMUM if verified else OptimalityStatus.UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Maximum flow
# ---------------------------------------------------------------------------

def _edmonds_karp(n: int, arcs: list[tuple[int, int, Fraction]],
                  s: int, t: int) -> tuple[list[Fraction], list[bool]]:
    """Exact Edmonds-Karp over Fraction capacities; returns flows and the
    source side of the minimum cut (residual-reachable set)."""
    flow = [Fraction(0)] * len(arcs)
    radj: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    for i, (u, v, _) in enumerate(arcs):
        radj[u].append((v, i, 1))
        radj[v].append((u, i, -1))
    for neighbours in radj:
        neighbours.sort()
    while True:
        prev = [(-1, -1, 0)] * n
        prev[s] = (s, -1, 0)
        queue: deque[int] = deque([s])
        while queue and prev[t][0] == -1:
            u = queue.popleft()
            for v, i, direction in radj[u]:
                if prev[v][0] != -1:
                    continue
                residual = arcs[i][2] - flow[i] if direction == 1 else flow[i]
                if residual > 0:
                    prev[v] = (u, i, direction)
                    queue.append(v)
        if prev[t][0] == -1:
            break
        bottleneck: Fraction | None = None
        v = t
        while v != s:
            u, i, direction = prev[v]
            residual = arcs[i][2] - flow[i] if direction == 1 else flow[i]
            bottleneck = residual if bottleneck is None else min(bottleneck, residual)
            v = u
        v = t
        while v != s:
            u, i, direction = prev[v]
            flow[i] += bottleneck if direction == 1 else -bottleneck
            v = u
    reachable = [False] * n
    reachable[s] = True
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v, i, direction in radj[u]:
            residual = arcs[i][2] - flow[i] if direction == 1 else flow[i]
            if residual > 0 and not reachable[v]:
                reachable[v] = True
                queue.append(v)
    return flow, reachable


def verify_maximum_flow(
    graph: WeightedGraph,
    source: Label,
    sink: Label,
    flows: list[EdgeFlow],
    cut: tuple[list[Label], list[Label]] | None,
) -> FlowVerification:
    """Independently re-check a candidate maximum flow, exactly.

    Acceptance requires all of: every edge carries an explicit flow with
    ``0 <= f <= c``; conservation at every non-terminal vertex; a valid
    s-t cut; and flow value equal to the cut capacity (max-flow min-cut).
    """
    diagnostics: list[str] = []
    try:
        view = _View(graph)
        s = _require_vertex(view, source)
        t = _require_vertex(view, sink)
    except (TypeError, ValueError) as exc:
        return FlowVerification(diagnostics=[f"invalid input: {exc}"])
    if not view.directed:
        return FlowVerification(diagnostics=["flow verification requires a directed graph"])
    if s == t:
        return FlowVerification(diagnostics=["source and sink must differ"])

    m = len(view.edges)
    edge_index = {(u, v): i for i, (u, v, _) in enumerate(view.edges)}
    assigned: list[Fraction | None] = [None] * m
    complete = True
    for entry in flows:
        try:
            u = _require_vertex(view, entry.source)
            v = _require_vertex(view, entry.target)
        except (TypeError, ValueError):
            complete = False
            diagnostics.append(
                f"flow entry references unknown vertex: {entry.source!r}->{entry.target!r}")
            continue
        if (u, v) not in edge_index:
            complete = False
            diagnostics.append(f"flow on non-edge {entry.source!r}->{entry.target!r}")
            continue
        i = edge_index[(u, v)]
        if assigned[i] is not None:
            complete = False
            diagnostics.append(f"duplicate flow entry for edge {entry.source!r}->{entry.target!r}")
            continue
        assigned[i] = entry.flow
    if any(f is None for f in assigned):
        complete = False
        diagnostics.append("every edge must carry an explicit flow value")

    capacity_respected = complete and all(
        Fraction(0) <= assigned[i] <= view.edges[i][2] for i in range(m))
    if not capacity_respected:
        diagnostics.append("capacity constraint violated (need 0 <= flow <= capacity)")

    conservation_holds = complete
    if complete:
        balance = [Fraction(0)] * len(view)
        for i, (u, v, _) in enumerate(view.edges):
            balance[u] -= assigned[i]
            balance[v] += assigned[i]
        for x in range(len(view)):
            if x not in (s, t) and balance[x] != 0:
                conservation_holds = False
                diagnostics.append(f"conservation violated at {view.labels[x]!r}")
                break

    flow_value: Fraction | None = None
    if complete:
        flow_value = -balance[s]

    cut_valid = False
    cut_capacity: Fraction | None = None
    if cut is not None:
        s_keys = {_sort_key(label) for label in cut[0]}
        t_keys = {_sort_key(label) for label in cut[1]}
        cut_valid = (
            not (s_keys & t_keys)
            and s_keys | t_keys == set(view.index)
            and _sort_key(source) in s_keys
            and _sort_key(sink) in t_keys
        )
        if cut_valid:
            s_idx = {view.index[key] for key in s_keys}
            cut_capacity = sum(
                (c for (u, v, c) in view.edges if u in s_idx and v not in s_idx),
                Fraction(0))
        else:
            diagnostics.append("cut is not a valid s-t partition of the vertices")

    value_matches_cut = (
        flow_value is not None
        and cut_capacity is not None
        and flow_value == cut_capacity
    )
    if cut_valid and not value_matches_cut:
        diagnostics.append("flow value does not equal the cut capacity")

    accepted = (
        capacity_respected and conservation_holds and cut_valid and value_matches_cut)
    return FlowVerification(
        capacity_respected=capacity_respected,
        conservation_holds=conservation_holds,
        cut_valid=cut_valid,
        value_matches_cut=value_matches_cut,
        flow_value=flow_value,
        cut_capacity=cut_capacity,
        accepted=accepted,
        diagnostics=diagnostics,
    )


def maximum_flow(graph: WeightedGraph, source: Label, sink: Label) -> MaxFlowResult:
    """Exact maximum s-t flow with a minimum-cut certificate.

    The result reports ``VERIFIED``/``OPTIMUM`` only when the independent
    re-check (capacity, conservation, value == cut capacity) accepts.
    """
    if not isinstance(graph, WeightedGraph) or not graph.directed:
        raise ValueError("maximum_flow requires a directed WeightedGraph")
    view = _View(graph)
    s = _require_vertex(view, source)
    t = _require_vertex(view, sink)
    if s == t:
        raise ValueError("source and sink must differ")
    for _, _, capacity in view.edges:
        if capacity < 0:
            raise ValueError("capacities must be nonnegative exact rationals")
    arcs = [(u, v, c) for u, v, c in view.edges]
    flow, reachable = _edmonds_karp(len(view), arcs, s, t)
    value = sum(
        (flow[i] for i, (u, _, _) in enumerate(view.edges) if u == s), Fraction(0))
    value -= sum(
        (flow[i] for i, (_, v, _) in enumerate(view.edges) if v == s), Fraction(0))
    cut_s = [view.labels[i] for i in range(len(view)) if reachable[i]]
    cut_t = [view.labels[i] for i in range(len(view)) if not reachable[i]]
    flows = [
        EdgeFlow(source=view.labels[u], target=view.labels[v], flow=flow[i])
        for i, (u, v, _) in enumerate(view.edges)
    ]
    verification = verify_maximum_flow(graph, source, sink, flows, (cut_s, cut_t))
    evidence = EvidenceBundle(
        computation=[_computation("edmonds_karp", vertices=len(view),
                                  edges=len(view.edges))],
        certificate=[_certificate(
            "max_flow_min_cut",
            "flow is feasible and its value equals the cut capacity",
            {"flow_value": value,
             "cut_capacity": verification.cut_capacity,
             "cut": (cut_s, cut_t)},
            verification.accepted)],
    )
    return MaxFlowResult(
        status=(OperationStatus.VERIFIED if verification.accepted
                else OperationStatus.ERROR),
        trust=TrustLevel.EXACT if verification.accepted else TrustLevel.UNKNOWN,
        evidence=evidence,
        diagnostics=list(verification.diagnostics),
        flow_value=value,
        flows=flows,
        cut=(cut_s, cut_t),
        verification=verification,
        optimality=(OptimalityStatus.OPTIMUM if verification.accepted
                    else OptimalityStatus.UNKNOWN),
    )


# ---------------------------------------------------------------------------
# Bipartite matching
# ---------------------------------------------------------------------------

def _odd_cycle(parent: list[int], depth: list[int], u: int, v: int) -> list[int]:
    """Odd cycle through same-colour edge (u, v) of a BFS tree."""
    path_u: list[int] = []
    path_v: list[int] = []
    a, b = u, v
    while depth[a] > depth[b]:
        path_u.append(a)
        a = parent[a]
    while depth[b] > depth[a]:
        path_v.append(b)
        b = parent[b]
    while a != b:
        path_u.append(a)
        path_v.append(b)
        a = parent[a]
        b = parent[b]
    cycle = path_u + [a] + path_v[::-1]
    cycle.append(cycle[0])
    return cycle


def verify_matching(
    graph: Graph | WeightedGraph | MultiGraph,
    matching: list[tuple[Label, Label]],
) -> bool:
    """Matching validity: every pair is an edge and vertices are disjoint."""
    view = _View(graph)
    _require_undirected(view, "verify_matching")
    used: set[int] = set()
    for a, b in matching:
        try:
            u = _require_vertex(view, a)
            v = _require_vertex(view, b)
        except (TypeError, ValueError):
            return False
        if v not in view.adjset[u] or u in used or v in used or u == v:
            return False
        used.add(u)
        used.add(v)
    return True


def _verify_cover(view: _View, cover: set[int], matching_size: int) -> bool:
    """Kőnig certificate: cover touches every edge and matches the size."""
    if len(cover) != matching_size:
        return False
    return all(u in cover or v in cover for u, v, _ in view.edges)


def bipartite_matching(graph: Graph | WeightedGraph | MultiGraph) -> MatchingResult:
    """Maximum bipartite matching with a Kőnig vertex-cover certificate.

    Non-bipartite input returns ``DOES_NOT_EXIST`` (for the bipartition) with
    an exact odd-cycle certificate; it is never a heuristic failure.
    """
    view = _View(graph)
    _require_undirected(view, "bipartite_matching")
    n = len(view)
    color = [-1] * n
    parent = [-1] * n
    depth = [0] * n
    for s in range(n):
        if color[s] >= 0:
            continue
        color[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            u = queue.popleft()
            for v, _ in view.adj[u]:
                if color[v] < 0:
                    color[v] = 1 - color[u]
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    queue.append(v)
                elif color[v] == color[u]:
                    cycle = _odd_cycle(parent, depth, u, v)
                    verified = _verify_cycle(view, cycle) and len(cycle) % 2 == 0
                    cycle_labels = [view.labels[i] for i in cycle]
                    evidence = EvidenceBundle(
                        computation=[_computation("bipartition_check", vertices=n)],
                        certificate=[_certificate(
                            "odd_cycle",
                            "an odd-length cycle; no bipartition exists",
                            {"cycle": cycle_labels}, verified)],
                    )
                    return MatchingResult(
                        status=(OperationStatus.DOES_NOT_EXIST if verified
                                else OperationStatus.ERROR),
                        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
                        evidence=evidence,
                        diagnostics=["graph is not bipartite (odd cycle certificate)"],
                        odd_cycle=cycle_labels,
                        optimality=(OptimalityStatus.IMPOSSIBLE if verified
                                    else OptimalityStatus.UNKNOWN),
                    )

    left = [i for i in range(n) if color[i] == 0]
    right = [i for i in range(n) if color[i] == 1]
    right_of = {v: k for k, v in enumerate(right)}
    nl, nr = len(left), len(right)
    s_node, t_node = nl + nr, nl + nr + 1
    arcs: list[tuple[int, int, Fraction]] = []
    for k in range(nl):
        arcs.append((s_node, k, Fraction(1)))
    cross_start = len(arcs)
    for k, u in enumerate(left):
        for v, _ in view.adj[u]:
            arcs.append((k, nl + right_of[v], Fraction(1)))
    cross_end = len(arcs)
    for j in range(nr):
        arcs.append((nl + j, t_node, Fraction(1)))
    flow, reachable = _edmonds_karp(t_node + 1, arcs, s_node, t_node)

    pairs = [
        (left[arcs[i][0]], right[arcs[i][1] - nl])
        for i in range(cross_start, cross_end) if flow[i] == 1
    ]
    cover = (
        [left[k] for k in range(nl) if not reachable[k]]
        + [right[j] for j in range(nr) if reachable[nl + j]]
    )
    matching_ok = verify_matching(graph, [(view.labels[a], view.labels[b])
                                          for a, b in pairs])
    cover_ok = _verify_cover(view, set(cover), len(pairs))
    verified = matching_ok and cover_ok
    witness = {
        "matching": [(view.labels[a], view.labels[b]) for a, b in pairs],
        "vertex_cover": [view.labels[i] for i in cover],
    }
    evidence = EvidenceBundle(
        computation=[_computation("bipartite_matching_maxflow", vertices=n)],
        certificate=[_certificate(
            "konig_cover",
            "matching is valid and a vertex cover of equal size covers "
            "every edge (Kőnig optimality)",
            witness, verified)],
    )
    return MatchingResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        bipartition=([view.labels[i] for i in left],
                     [view.labels[i] for i in right]),
        matching=[(view.labels[a], view.labels[b]) for a, b in pairs],
        size=len(pairs),
        vertex_cover=[view.labels[i] for i in cover],
        optimality=OptimalityStatus.OPTIMUM if verified else OptimalityStatus.UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Euler trails
# ---------------------------------------------------------------------------

def _verify_euler(view: _View, trail: list[int], edge_ids: list[int],
                  kind: str) -> bool:
    m = len(view.edges)
    if len(edge_ids) != m or sorted(edge_ids) != list(range(m)):
        return False
    if len(trail) != m + 1:
        return False
    for k, eid in enumerate(edge_ids):
        u, v, _ = view.edges[eid]
        a, b = trail[k], trail[k + 1]
        if not ((a == u and b == v) or (a == v and b == u)):
            return False
    return (trail[0] == trail[-1]) == (kind == "cycle")


def euler_trail(graph: Graph | WeightedGraph | MultiGraph) -> EulerResult:
    """Euler path/cycle in an undirected (multi)graph via Hierholzer.

    Non-existence is exact: an odd-degree count other than 0/2 or a
    disconnected edge set proves no Euler trail exists.
    """
    view = _View(graph)
    _require_undirected(view, "euler_trail")
    n, m = len(view), len(view.edges)
    degrees = [len(neighbours) for neighbours in view.adj]
    odd = [i for i in range(n) if degrees[i] % 2 == 1]

    def _impossible(reason: str, witness: Any, verified: bool) -> EulerResult:
        evidence = EvidenceBundle(
            computation=[_computation("euler_trail", vertices=n, edges=m)],
            certificate=[_certificate("euler_impossibility", reason, witness, verified)],
        )
        return EulerResult(
            status=OperationStatus.DOES_NOT_EXIST if verified else OperationStatus.ERROR,
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            evidence=evidence,
            diagnostics=[reason],
        )

    if m == 0:
        trail = [view.labels[0]] if n else []
        return EulerResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            evidence=EvidenceBundle(
                computation=[_computation("euler_trail", vertices=n, edges=m)],
                proof=[_proof("an edgeless graph has a trivial Euler circuit",
                              "trivial_euler")]),
            kind="cycle",
            trail=trail,
        )
    non_isolated = [i for i in range(n) if degrees[i] > 0]
    seen = {non_isolated[0]}
    queue: deque[int] = deque(non_isolated[:1])
    while queue:
        u = queue.popleft()
        for v, _ in view.adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    if len(seen) != len(non_isolated):
        witness = sorted(view.labels[i] for i in non_isolated if i not in seen)
        return _impossible(
            "edges span more than one component; no Euler trail exists",
            {"unreached_vertices": witness}, True)
    if len(odd) not in (0, 2):
        return _impossible(
            f"{len(odd)} odd-degree vertices; no Euler trail exists",
            {"odd_vertices": [view.labels[i] for i in odd]}, True)

    start = odd[0] if odd else non_isolated[0]
    used = [False] * m
    pointer = [0] * n
    stack = [start]
    edge_stack: list[int] = []
    trail: list[int] = []
    trail_edges: list[int] = []
    while stack:
        u = stack[-1]
        while pointer[u] < len(view.adj[u]) and used[view.adj[u][pointer[u]][1]]:
            pointer[u] += 1
        if pointer[u] == len(view.adj[u]):
            trail.append(u)
            if edge_stack:
                trail_edges.append(edge_stack.pop())
            stack.pop()
        else:
            v, eid = view.adj[u][pointer[u]]
            used[eid] = True
            pointer[u] += 1
            edge_stack.append(eid)
            stack.append(v)
    trail.reverse()
    trail_edges.reverse()
    kind = "cycle" if not odd else "path"
    verified = _verify_euler(view, trail, trail_edges, kind)
    evidence = EvidenceBundle(
        computation=[_computation("hierholzer_euler", vertices=n, edges=m)],
        certificate=[_certificate(
            "euler_trail",
            "trail uses every edge exactly once with consistent endpoints",
            {"edge_ids": trail_edges}, verified)],
    )
    return EulerResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        kind=kind,
        trail=[view.labels[i] for i in trail],
        edge_ids=trail_edges,
    )


# ---------------------------------------------------------------------------
# Coloring
# ---------------------------------------------------------------------------

def verify_coloring(graph: Graph | WeightedGraph | MultiGraph,
                    coloring: dict[Label, int]) -> bool:
    """Proper-coloring check: every vertex colored, every edge bichromatic."""
    view = _View(graph)
    _require_undirected(view, "verify_coloring")
    if len(coloring) != len(view):
        return False
    try:
        color = [_require_vertex(view, label) for label in coloring]
    except (TypeError, ValueError):
        return False
    values = {view.index[_sort_key(label)]: c for label, c in coloring.items()}
    if sorted(color) != list(range(len(view))) or len(values) != len(view):
        return False
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0
           for c in values.values()):
        return False
    return all(values[u] != values[v] for u, v, _ in view.edges)


def _k_colorable(view: _View, k: int, budget: _Budget) -> bool:
    """Exact backtracking k-colorability; raises _BudgetExceeded, never guesses."""
    n = len(view)
    if n == 0:
        return True
    if k <= 0:
        return False
    order = sorted(range(n), key=lambda i: (-len(view.adj[i]), i))
    color = [-1] * n

    def search(position: int) -> bool:
        budget.spend()
        if position == n:
            return True
        v = order[position]
        used = {color[u] for u in view.adjset[v] if color[u] >= 0}
        for c in range(k):
            if c not in used:
                color[v] = c
                if search(position + 1):
                    return True
                color[v] = -1
        return False

    return search(0)


def graph_coloring(
    graph: Graph | WeightedGraph | MultiGraph,
    limits: SearchLimits | None = None,
) -> ColoringResult:
    """Greedy proper coloring plus an exact chromatic number for small graphs.

    The greedy coloring is always verified edge-by-edge.  The chromatic
    number is ``OPTIMUM`` only when an exact budgeted search (or a matching
    clique lower bound) proves it; a budget overrun yields ``UNKNOWN`` with
    certified bounds — never a heuristic claim of impossibility.
    """
    limits = limits or SearchLimits()
    view = _View(graph)
    _require_undirected(view, "graph_coloring")
    n, m = len(view), len(view.edges)

    loop = next((i for i, (u, v, _) in enumerate(view.edges) if u == v), None)
    if loop is not None:
        u = view.edges[loop][0]
        evidence = EvidenceBundle(
            computation=[_computation("graph_coloring", vertices=n)],
            proof=[_proof(
                f"vertex {view.labels[u]!r} has a self-loop; "
                "no proper coloring exists", "self_loop")],
        )
        return ColoringResult(
            status=OperationStatus.DOES_NOT_EXIST,
            trust=TrustLevel.EXACT,
            evidence=evidence,
            diagnostics=["self-loop admits no proper coloring"],
            optimality=OptimalityStatus.IMPOSSIBLE,
        )

    order = sorted(range(n), key=lambda i: (-len(view.adj[i]), i))
    color = [-1] * n
    for v in order:
        used = {color[u] for u in view.adjset[v] if color[u] >= 0}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    num_colors = max(color) + 1 if n else 0
    coloring = {view.labels[i]: color[i] for i in range(n)}
    proper = verify_coloring(graph, coloring)

    clique: list[int] = []
    for v in order:
        if all(u in view.adjset[v] for u in clique):
            clique.append(v)
    lower = len(clique)

    chromatic: int | None = None
    optimality = OptimalityStatus.CANDIDATE
    proof_method: str | None = None
    diagnostics: list[str] = []
    if n == 0 or m == 0:
        chromatic = 0 if n == 0 else 1
        optimality = OptimalityStatus.OPTIMUM
        proof_method = "trivial_coloring"
    elif lower == num_colors:
        chromatic = num_colors
        optimality = OptimalityStatus.OPTIMUM
        proof_method = "clique_bound_meets_greedy"
    elif n <= limits.exact_coloring_max_vertices:
        budget = _Budget(limits.max_operations)
        try:
            for k in range(max(lower, 1), num_colors):
                if _k_colorable(view, k, budget):
                    chromatic = k
                    break
            if chromatic is None:
                chromatic = num_colors
            optimality = OptimalityStatus.OPTIMUM
            proof_method = "exact_k_colorability"
        except _BudgetExceeded:
            optimality = OptimalityStatus.UNKNOWN
            diagnostics.append(
                "exact coloring search exceeded the operation budget; "
                "chromatic number is unknown within certified bounds")
    else:
        diagnostics.append(
            f"n={n} exceeds exact_coloring_max_vertices="
            f"{limits.exact_coloring_max_vertices}; reporting certified bounds only")

    evidence = EvidenceBundle(
        computation=[_computation("greedy_coloring", vertices=n, edges=m)],
        certificate=[_certificate(
            "proper_coloring",
            "every edge is bichromatic under the reported coloring",
            {"num_colors": num_colors}, proper)],
    )
    if optimality == OptimalityStatus.OPTIMUM:
        evidence.proof.append(_proof(
            f"chromatic number is {chromatic}", proof_method or "exact_k_colorability"))
    if lower >= 2:
        evidence.certificate.append(_certificate(
            "clique_lower_bound",
            f"a clique of size {lower} requires at least {lower} colors",
            {"clique": [view.labels[i] for i in clique]}, True))
    return ColoringResult(
        status=OperationStatus.VERIFIED if proper else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if proper else TrustLevel.UNKNOWN,
        evidence=evidence,
        diagnostics=diagnostics,
        coloring=coloring,
        num_colors=num_colors,
        chromatic_number=chromatic,
        lower_bound=lower if n else 0,
        upper_bound=num_colors,
        optimality=optimality,
    )


# ---------------------------------------------------------------------------
# Topological sort and cycle detection
# ---------------------------------------------------------------------------

def topological_sort(digraph: DirectedGraph | WeightedGraph) -> TopologicalSortResult:
    """Kahn's algorithm; failure returns an exact cycle certificate."""
    view = _View(digraph)
    _require_directed(view, "topological_sort")
    n = len(view)
    indeg = [0] * n
    for _, v, _ in view.edges:
        indeg[v] += 1
    heap = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v, _ in view.adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)
    if len(order) == n:
        position = {v: i for i, v in enumerate(order)}
        verified = all(position[u] < position[v] for u, v, _ in view.edges)
        evidence = EvidenceBundle(
            computation=[_computation("kahn_topological_sort", vertices=n)],
            certificate=[_certificate(
                "topological_order",
                "every arc points forward in the reported order",
                {"order": [view.labels[i] for i in order]}, verified)],
        )
        return TopologicalSortResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            evidence=evidence,
            order=[view.labels[i] for i in order],
        )
    remaining = sorted(i for i in range(n) if indeg[i] > 0)
    remaining_set = set(remaining)
    preds: dict[int, list[int]] = {
        v: sorted(u for u, w, _ in view.edges if w == v and u in remaining_set)
        for v in remaining
    }
    seq = [remaining[0]]
    while True:
        nxt = preds[seq[-1]][0]
        if nxt in seq:
            idx = seq.index(nxt)
            cycle = [nxt] + list(reversed(seq[idx + 1:])) + [nxt]
            break
        seq.append(nxt)
    verified = _verify_cycle(view, cycle)
    cycle_labels = [view.labels[i] for i in cycle]
    evidence = EvidenceBundle(
        computation=[_computation("kahn_topological_sort", vertices=n)],
        certificate=[_certificate(
            "directed_cycle",
            "a directed cycle; no topological order exists",
            {"cycle": cycle_labels}, verified)],
    )
    return TopologicalSortResult(
        status=OperationStatus.DOES_NOT_EXIST if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        diagnostics=["digraph contains a cycle; no topological order exists"],
        cycle=cycle_labels,
    )


def _verify_forest(view: _View, parent: list[int], ncomp: int) -> bool:
    """Undirected acyclicity: a valid spanning forest with m == n - components."""
    n = len(view)
    for v in range(n):
        p, hops = v, 0
        while parent[p] != -1:
            p = parent[p]
            hops += 1
            if hops > n:
                return False
        if parent[v] != -1 and v not in view.adjset[parent[v]]:
            return False
    roots = sum(1 for v in range(n) if parent[v] == -1)
    return roots == ncomp and len(view.edges) == n - ncomp


def find_cycle(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
) -> CycleResult:
    """Exact cycle detection; both outcomes carry a verifiable certificate."""
    view = _View(graph)
    n = len(view)
    color = [0] * n
    parent = [-1] * n
    parent_eid = [-1] * n
    cycle: list[int] | None = None
    ncomp = 0
    for s in range(n):
        if color[s] != 0:
            continue
        ncomp += 1
        color[s] = 1
        stack = [(s, 0)]
        while stack and cycle is None:
            u, i = stack[-1]
            if i < len(view.adj[u]):
                v, eid = view.adj[u][i]
                stack[-1] = (u, i + 1)
                if color[v] == 0:
                    color[v] = 1
                    parent[v] = u
                    parent_eid[v] = eid
                    stack.append((v, 0))
                elif color[v] == 1:
                    if not view.directed and v == parent[u] and eid == parent_eid[u]:
                        continue
                    if v == u:
                        cycle = [u, u]
                    else:
                        seq = [u]
                        while seq[-1] != v:
                            seq.append(parent[seq[-1]])
                        cycle = (list(reversed(seq)) + [v]) if view.directed \
                            else (seq + [u])
            else:
                color[u] = 2
                stack.pop()
        if cycle is not None:
            break

    if cycle is not None:
        verified = _verify_cycle(view, cycle)
        cycle_labels = [view.labels[i] for i in cycle]
        evidence = EvidenceBundle(
            computation=[_computation("find_cycle", vertices=n)],
            certificate=[_certificate(
                "cycle",
                "consecutive vertices are adjacent and the walk closes",
                {"cycle": cycle_labels}, verified)],
        )
        return CycleResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            evidence=evidence,
            has_cycle=True,
            cycle=cycle_labels,
        )

    if view.directed:
        topo = topological_sort(graph)
        verified = topo.status == OperationStatus.VERIFIED
        evidence = EvidenceBundle(
            computation=[_computation("find_cycle", vertices=n)],
            certificate=[_certificate(
                "acyclic_topological_order",
                "a topological order exists, hence the digraph is acyclic",
                {"order": topo.order}, verified)],
        )
    else:
        verified = _verify_forest(view, parent, ncomp)
        evidence = EvidenceBundle(
            computation=[_computation("find_cycle", vertices=n)],
            certificate=[_certificate(
                "spanning_forest",
                "edges == vertices - components over a valid spanning forest, "
                "hence acyclic",
                {"components": ncomp}, verified)],
        )
    return CycleResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        has_cycle=False,
    )


# ---------------------------------------------------------------------------
# Centrality
# ---------------------------------------------------------------------------

def degree_centrality(
    graph: Graph | DirectedGraph | WeightedGraph | MultiGraph,
) -> CentralityResult:
    """Exact degree centrality as Fractions (degree / (n - 1))."""
    view = _View(graph)
    n = len(view)
    base = max(n - 1, 1)
    if view.directed:
        indeg = [0] * n
        outdeg = [0] * n
        for u, v, _ in view.edges:
            outdeg[u] += 1
            indeg[v] += 1
        in_cent = {view.labels[i]: Fraction(indeg[i], base) for i in range(n)}
        out_cent = {view.labels[i]: Fraction(outdeg[i], base) for i in range(n)}
        check_in = [0] * n
        check_out = [0] * n
        for u, v, _ in view.edges:
            check_out[u] += 1
            check_in[v] += 1
        verified = check_in == indeg and check_out == outdeg
        evidence = EvidenceBundle(
            computation=[_computation("degree_centrality", vertices=n)],
            certificate=[_certificate(
                "degree_recomputation",
                "in/out degrees recomputed independently from the edge list",
                {}, verified)],
        )
        return CentralityResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            evidence=evidence,
            in_centrality=in_cent,
            out_centrality=out_cent,
        )
    degrees = [len(neighbours) for neighbours in view.adj]
    check = [0] * n
    for u, v, _ in view.edges:
        check[u] += 1
        check[v] += 1
    verified = check == degrees
    diagnostics = []
    if any(u == v for u, v, _ in view.edges) or len(view.edges) != len({
            (min(u, v), max(u, v)) for u, v, _ in view.edges}):
        diagnostics.append(
            "multigraph: parallel edges and self-loops count toward degree; "
            "centrality may exceed 1")
    evidence = EvidenceBundle(
        computation=[_computation("degree_centrality", vertices=n)],
        certificate=[_certificate(
            "degree_recomputation",
            "degrees recomputed independently from the edge list",
            {}, verified)],
    )
    return CentralityResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        diagnostics=diagnostics,
        centrality={view.labels[i]: Fraction(degrees[i], base) for i in range(n)},
    )


# ---------------------------------------------------------------------------
# Isomorphism
# ---------------------------------------------------------------------------

def verify_isomorphism(g1: Graph, g2: Graph,
                       permutation: dict[Label, Label]) -> bool:
    """Check that ``permutation`` is a bijection preserving edges exactly."""
    if not isinstance(g1, Graph) or not isinstance(g2, Graph):
        raise ValueError("verify_isomorphism supports simple undirected Graph")
    v1, v2 = _View(g1), _View(g2)
    if len(permutation) != len(v1) or len(v1) != len(v2):
        return False
    try:
        mapping = {_require_vertex(v1, a): _require_vertex(v2, b)
                   for a, b in permutation.items()}
    except (TypeError, ValueError):
        return False
    if sorted(mapping) != list(range(len(v1))):
        return False
    if sorted(mapping.values()) != list(range(len(v2))):
        return False
    if len(v1.edges) != len(v2.edges):
        return False
    for u, v, _ in v1.edges:
        if mapping[v] not in v2.adjset[mapping[u]]:
            return False
    return True


def graph_isomorphism(
    g1: Graph,
    g2: Graph,
    limits: SearchLimits | None = None,
) -> IsomorphismResult:
    """Exact isomorphism by invariant checks plus budgeted backtracking.

    Outcomes: ``VERIFIED`` with a re-checked permutation certificate;
    ``REFUTED`` only when an invariant differs or the search space was
    completely exhausted; ``UNKNOWN`` when the budget ran out — an
    inconclusive search is never reported as non-isomorphism.
    """
    if not isinstance(g1, Graph) or not isinstance(g2, Graph):
        raise ValueError("graph_isomorphism supports simple undirected Graph")
    limits = limits or SearchLimits()
    v1, v2 = _View(g1), _View(g2)
    n1, n2 = len(v1), len(v2)

    def _refuted(reason: str, method: str, nodes: int) -> IsomorphismResult:
        evidence = EvidenceBundle(
            computation=[_computation("graph_isomorphism", search_nodes=nodes)],
            proof=[_proof(f"graphs are not isomorphic: {reason}", method)],
        )
        return IsomorphismResult(
            status=OperationStatus.REFUTED,
            trust=TrustLevel.EXACT,
            evidence=evidence,
            isomorphic=False,
            search_nodes=nodes,
        )

    if n1 != n2:
        return _refuted("vertex counts differ", "invariant_check", 0)
    if len(v1.edges) != len(v2.edges):
        return _refuted("edge counts differ", "invariant_check", 0)
    if n1 > limits.max_vertices:
        return IsomorphismResult(
            status=OperationStatus.UNKNOWN,
            evidence=EvidenceBundle(computation=[
                _computation("graph_isomorphism", vertices=n1)]),
            diagnostics=[
                f"n={n1} exceeds max_vertices={limits.max_vertices}; "
                "no conclusion drawn"],
            isomorphic=None,
        )
    deg1 = sorted(len(a) for a in v1.adj)
    deg2 = sorted(len(a) for a in v2.adj)
    if deg1 != deg2:
        return _refuted("degree sequences differ", "invariant_check", 0)

    order = sorted(range(n1), key=lambda i: (-len(v1.adj[i]), i))
    candidates = {
        v: [u for u in range(n2) if len(v2.adj[u]) == len(v1.adj[v])]
        for v in range(n1)
    }
    mapping = [-1] * n1
    used = [False] * n2
    budget = _Budget(limits.max_operations)
    nodes = [0]

    def consistent(v: int, u: int) -> bool:
        for pv in range(n1):
            pu = mapping[pv]
            if pu >= 0 and ((pv in v1.adjset[v]) != (pu in v2.adjset[u])):
                return False
        return True

    def search(position: int) -> bool:
        budget.spend()
        nodes[0] += 1
        if position == n1:
            return True
        v = order[position]
        for u in candidates[v]:
            if not used[u] and consistent(v, u):
                mapping[v] = u
                used[u] = True
                if search(position + 1):
                    return True
                mapping[v] = -1
                used[u] = False
        return False

    try:
        found = search(0) if n1 else True
    except _BudgetExceeded:
        return IsomorphismResult(
            status=OperationStatus.UNKNOWN,
            evidence=EvidenceBundle(computation=[
                _computation("graph_isomorphism", search_nodes=nodes[0])]),
            diagnostics=[
                "isomorphism search exceeded the operation budget; "
                "this is not evidence of non-isomorphism"],
            isomorphic=None,
            search_nodes=nodes[0],
        )
    if not found:
        return _refuted(
            "the degree-compatible search space was exhausted",
            "exhaustive_backtracking", nodes[0])
    permutation = {v1.labels[v]: v2.labels[mapping[v]] for v in range(n1)}
    verified = verify_isomorphism(g1, g2, permutation)
    evidence = EvidenceBundle(
        computation=[_computation("graph_isomorphism", search_nodes=nodes[0])],
        certificate=[_certificate(
            "isomorphism_permutation",
            "a bijection preserving adjacency exactly",
            {"permutation": permutation}, verified)],
    )
    return IsomorphismResult(
        status=OperationStatus.VERIFIED if verified else OperationStatus.ERROR,
        trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
        evidence=evidence,
        isomorphic=True if verified else None,
        permutation=permutation,
        search_nodes=nodes[0],
    )
