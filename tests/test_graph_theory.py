# =============================================================================
# MathKernel - certified exact graph theory tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from fractions import Fraction

import pytest

from mathkernel.graph_theory import (
    DirectedGraph, Graph, GraphEdge, GraphResult, MultiGraph, OptimalityStatus,
    SearchLimits, WeightedGraph, bipartite_matching, breadth_first_search,
    connected_components, degree_centrality, depth_first_search, euler_trail,
    find_cycle, graph_coloring, graph_isomorphism, maximum_flow,
    minimum_spanning_tree, shortest_paths, strongly_connected_components,
    topological_sort, verify_coloring, verify_graph,
)
from mathkernel.graph_theory import _View, _verify_bfs_tree, _verify_components
from mathkernel.models import OperationStatus, TrustLevel
from mathkernel_artifacts import CertificateEvidence, EvidenceBundle


def edge(source, target, weight=1):
    return GraphEdge(source=source, target=target, weight=Fraction(weight))


def test_model_validation_and_verify_certificate():
    graph = Graph(vertices=["a", "b"], edges=[edge("a", "b")])
    result = verify_graph(graph)
    assert result.status == OperationStatus.VERIFIED
    assert result.trust == TrustLevel.EXACT
    assert result.evidence.certificate[0].verified

    with pytest.raises(ValueError, match="parallel edges"):
        Graph(vertices=[0, 1], edges=[edge(0, 1), edge(1, 0)])
    with pytest.raises(ValueError, match="unknown vertex"):
        Graph(vertices=[0], edges=[edge(0, 1)])
    with pytest.raises(TypeError, match="exact rationals"):
        GraphEdge(source=0, target=1, weight=0.5)


def test_bfs_dfs_and_components_are_exact():
    graph = Graph(vertices=["a", "b", "c", "d"], edges=[
        edge("a", "b"), edge("b", "c"),
    ])
    bfs = breadth_first_search(graph, "a")
    assert bfs.status == OperationStatus.VERIFIED
    assert bfs.order == ["a", "b", "c"]
    assert bfs.depth == {"a": 0, "b": 1, "c": 2}
    assert bfs.evidence.certificate[0].certificate_type == "bfs_tree"

    dfs = depth_first_search(graph, "a")
    assert dfs.status == OperationStatus.VERIFIED
    assert dfs.order == ["a", "b", "c"]

    components = connected_components(graph)
    assert components.status == OperationStatus.VERIFIED
    assert components.components == [["a", "b", "c"], ["d"]]
    assert components.count == 2


def test_linear_graph_certificate_verifiers_reject_adversarial_forests():
    graph = Graph(vertices=list(range(6)), edges=[
        edge(0, 1), edge(1, 2), edge(3, 4), edge(4, 5),
    ])
    view = _View(graph)

    assert _verify_bfs_tree(
        view, 0, [0, 1, 2], [0, 1, 2, -1, -1, -1],
        [-1, 0, 1, -1, -1, -1],
    )
    assert not _verify_bfs_tree(
        view, 0, [0, 2, 1], [0, 1, 2, -1, -1, -1],
        [-1, 0, 1, -1, -1, -1],
    )
    assert not _verify_bfs_tree(
        view, 0, [0, 1, 2], [0, 1, 2, -1, -1, -1],
        [-1, 2, 1, -1, -1, -1],
    )

    assert _verify_components(
        view, [0, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 3, 4], 2,
    )
    assert not _verify_components(
        view, [0, 0, 0, 1, 1, 1], [-1, 2, 1, -1, 3, 4], 2,
    )
    assert not _verify_components(
        view, [0, 0, 0, 0, 0, 0], [-1, 0, 1, -1, 3, 4], 1,
    )


def test_graph_certificate_verifiers_handle_long_path_without_recursion():
    size = 5_000
    graph = Graph(
        vertices=list(range(size)),
        edges=[edge(v, v + 1) for v in range(size - 1)],
    )
    view = _View(graph)
    order = list(range(size))
    parent = [-1, *range(size - 1)]
    assert _verify_bfs_tree(view, 0, order, order, parent)
    assert _verify_components(view, [0] * size, parent, 1)


def test_scc_topological_sort_and_cycle_certificates():
    dag = DirectedGraph(vertices=[0, 1, 2], edges=[edge(0, 1), edge(1, 2)])
    topo = topological_sort(dag)
    assert topo.status == OperationStatus.VERIFIED
    assert topo.order == [0, 1, 2]
    assert find_cycle(dag).has_cycle is False

    cyclic = DirectedGraph(vertices=[0, 1], edges=[edge(0, 1), edge(1, 0)])
    scc = strongly_connected_components(cyclic)
    assert scc.status == OperationStatus.VERIFIED
    assert scc.components == [[0, 1]]
    impossible = topological_sort(cyclic)
    assert impossible.status == OperationStatus.DOES_NOT_EXIST
    assert impossible.evidence.certificate[0].certificate_type == "directed_cycle"
    cycle = find_cycle(cyclic)
    assert cycle.has_cycle is True
    assert cycle.cycle == [0, 1, 0]


def test_shortest_path_has_optimality_certificate():
    graph = WeightedGraph(vertices=["s", "a", "b", "t"], edges=[
        edge("s", "a", 2), edge("s", "b", 5), edge("a", "b", 1),
        edge("b", "t", 2), edge("a", "t", 7),
    ])
    result = shortest_paths(graph, "s", "t")
    assert result.status == OperationStatus.VERIFIED
    assert result.path == ["s", "a", "b", "t"]
    assert result.distances["t"] == 5
    assert result.evidence.certificate[0].certificate_type == (
        "shortest_path_optimality")


def test_unreachable_and_negative_cycle_are_exact_nonexistence():
    disconnected = Graph(vertices=[0, 1], edges=[])
    missing = shortest_paths(disconnected, 0, 1)
    assert missing.status == OperationStatus.DOES_NOT_EXIST
    assert missing.trust == TrustLevel.EXACT

    negative = WeightedGraph(vertices=[0, 1], directed=True, edges=[
        edge(0, 1, -2), edge(1, 0, 1),
    ])
    result = shortest_paths(negative, 0, 1)
    assert result.status == OperationStatus.DOES_NOT_EXIST
    assert result.negative_cycle is not None
    assert result.evidence.certificate[0].certificate_type == "negative_cycle"


def test_minimum_spanning_tree_cycle_property_certificate():
    graph = WeightedGraph(vertices=[0, 1, 2, 3], edges=[
        edge(0, 1, 1), edge(1, 2, 2), edge(2, 3, 1),
        edge(0, 3, 10), edge(1, 3, 4),
    ])
    result = minimum_spanning_tree(graph)
    assert result.status == OperationStatus.VERIFIED
    assert result.total_weight == 4
    assert result.optimality == OptimalityStatus.OPTIMUM
    assert result.evidence.certificate[0].certificate_type == (
        "mst_cycle_property")


def test_maximum_flow_min_cut_acceptance_semantics():
    graph = WeightedGraph(vertices=["s", "a", "b", "t"], directed=True, edges=[
        edge("s", "a", 3), edge("s", "b", 2), edge("a", "b", 1),
        edge("a", "t", 2), edge("b", "t", 3),
    ])
    result = maximum_flow(graph, "s", "t")
    assert result.status == OperationStatus.VERIFIED
    assert result.flow_value == 5
    assert result.optimality == OptimalityStatus.OPTIMUM
    assert result.verification is not None
    assert result.verification.accepted
    assert result.verification.capacity_respected
    assert result.verification.conservation_holds
    assert result.verification.value_matches_cut
    assert result.evidence.certificate[0].certificate_type == "max_flow_min_cut"


def test_bipartite_matching_has_konig_certificate_and_odd_cycle_refutation():
    graph = Graph(vertices=["u1", "u2", "v1", "v2"], edges=[
        edge("u1", "v1"), edge("u1", "v2"), edge("u2", "v1"),
    ])
    result = bipartite_matching(graph)
    assert result.status == OperationStatus.VERIFIED
    assert result.size == 2
    assert result.optimality == OptimalityStatus.OPTIMUM
    assert result.evidence.certificate[0].certificate_type == "konig_cover"

    triangle = Graph(vertices=[0, 1, 2], edges=[
        edge(0, 1), edge(1, 2), edge(2, 0),
    ])
    impossible = bipartite_matching(triangle)
    assert impossible.status == OperationStatus.DOES_NOT_EXIST
    assert impossible.optimality == OptimalityStatus.IMPOSSIBLE
    assert impossible.odd_cycle is not None


def test_euler_trail_uses_multigraph_edge_identity():
    graph = MultiGraph(vertices=[0, 1, 2], edges=[
        edge(0, 1), edge(1, 2), edge(2, 0),
    ])
    result = euler_trail(graph)
    assert result.status == OperationStatus.VERIFIED
    assert result.kind == "cycle"
    assert sorted(result.edge_ids) == [0, 1, 2]

    path_graph = Graph(vertices=[0, 1, 2], edges=[edge(0, 1), edge(1, 2)])
    path = euler_trail(path_graph)
    assert path.status == OperationStatus.VERIFIED
    assert path.kind == "path"


def test_coloring_distinguishes_candidate_optimum_and_budget_unknown():
    triangle = Graph(vertices=[0, 1, 2], edges=[
        edge(0, 1), edge(1, 2), edge(2, 0),
    ])
    exact = graph_coloring(triangle)
    assert exact.status == OperationStatus.VERIFIED
    assert exact.chromatic_number == 3
    assert exact.optimality == OptimalityStatus.OPTIMUM
    assert verify_coloring(triangle, exact.coloring)

    five_cycle = Graph(vertices=list(range(5)), edges=[
        edge(0, 1), edge(1, 2), edge(2, 3), edge(3, 4), edge(4, 0),
    ])
    bounded = graph_coloring(five_cycle, SearchLimits(max_operations=1))
    assert bounded.status == OperationStatus.VERIFIED
    assert bounded.optimality == OptimalityStatus.UNKNOWN
    assert bounded.chromatic_number is None
    assert "budget" in bounded.diagnostics[0]

    looped = MultiGraph(vertices=[0], edges=[edge(0, 0)])
    impossible = graph_coloring(looped)
    assert impossible.status == OperationStatus.DOES_NOT_EXIST
    assert impossible.optimality == OptimalityStatus.IMPOSSIBLE


def test_isomorphism_permutation_certificate_and_exact_refutation():
    left = Graph(vertices=["a", "b", "c"], edges=[edge("a", "b"), edge("b", "c")])
    right = Graph(vertices=[0, 1, 2], edges=[edge(0, 1), edge(1, 2)])
    result = graph_isomorphism(left, right)
    assert result.status == OperationStatus.VERIFIED
    assert result.isomorphic is True
    assert result.permutation is not None
    assert result.evidence.certificate[0].certificate_type == (
        "isomorphism_permutation")

    different = Graph(vertices=[0, 1, 2], edges=[edge(0, 1)])
    refuted = graph_isomorphism(left, different)
    assert refuted.status == OperationStatus.REFUTED
    assert refuted.trust == TrustLevel.EXACT

    unknown = graph_isomorphism(left, right, SearchLimits(max_operations=1))
    assert unknown.status == OperationStatus.UNKNOWN
    assert unknown.isomorphic is None


def test_unverified_certificate_cannot_launder_exact_trust():
    result = GraphResult(
        status=OperationStatus.VERIFIED,
        trust=TrustLevel.EXACT,
        evidence=EvidenceBundle(certificate=[CertificateEvidence(
            certificate_type="max_flow_min_cut",
            claim="fabricated optimality claim",
            witness={},
            verifier="external",
            verified=False,
            trust="unknown",
        )]),
    )
    assert result.status == OperationStatus.CANDIDATE
    assert result.trust == TrustLevel.UNKNOWN
    assert any("downgraded" in item for item in result.diagnostics)


def test_degree_centrality_is_exact_rational():
    graph = Graph(vertices=[0, 1, 2], edges=[edge(0, 1), edge(0, 2)])
    result = degree_centrality(graph)
    assert result.status == OperationStatus.VERIFIED
    assert result.centrality[0] == 1
    assert result.centrality[1] == Fraction(1, 2)
    assert result.trust == TrustLevel.EXACT
