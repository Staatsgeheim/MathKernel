# =============================================================================
# MathKernel - exact-discrete integration tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel import MathKernel, Settings
from mathkernel.models import ResultStatus, TrustLevel


def test_graph_maximum_flow_has_min_cut_certificate():
    kernel = MathKernel()
    created = kernel.object_create("weighted_graph", {
        "nodes": ["s", "a", "b", "t"],
        "directed": True,
        "edges": [
            {"source": "s", "target": "a", "weight": "3"},
            {"source": "s", "target": "b", "weight": "2"},
            {"source": "a", "target": "b", "weight": "1"},
            {"source": "a", "target": "t", "weight": "2"},
            {"source": "b", "target": "t", "weight": "3"},
        ],
    })
    assert created.ok
    result = kernel.apply(
        created.data["object_id"], "maximum_flow",
        {"source": "s", "sink": "t"},
    )
    assert result.status == "verified"
    assert result.trust == TrustLevel.EXACT
    assert result.data["flow_value"] == "5"
    assert result.data["optimality"] == "optimum"
    assert result.data["verification"]["accepted"] is True
    assert result.data["verification"]["capacity_respected"] is True
    assert result.data["verification"]["conservation_holds"] is True
    assert result.data["verification"]["value_matches_cut"] is True
    assert result.evidence_bundle.certificate[0].certificate_type == (
        "max_flow_min_cut")


def test_graph_np_hard_search_exhaustion_is_unknown_not_impossible():
    kernel = MathKernel()
    triangle = kernel.object_create("graph", {
        "nodes": [0, 1, 2],
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 0},
        ],
    })
    result = kernel.apply(
        triangle.data["object_id"], "coloring",
        {"max_operations": 1},
    )
    assert result.status == "verified"
    assert result.trust == TrustLevel.EXACT
    assert result.data["optimality"] == "optimum"
    assert result.data["chromatic_number"] == 3

    five_cycle = kernel.object_create("graph", {
        "vertices": list(range(5)),
        "edges": [
            {"source": 0, "target": 1},
            {"source": 1, "target": 2},
            {"source": 2, "target": 3},
            {"source": 3, "target": 4},
            {"source": 4, "target": 0},
        ],
    })
    bounded = kernel.apply(
        five_cycle.data["object_id"], "coloring", {"max_operations": 1})
    assert bounded.status == "verified"
    assert bounded.trust == TrustLevel.EXACT
    assert bounded.data["optimality"] == "unknown"
    assert bounded.data["chromatic_number"] is None
    assert "budget" in bounded.warnings[0]


def test_combinatorial_class_count_and_lazy_generation_are_exact():
    kernel = MathKernel()
    created = kernel.object_create("combinatorial_class", {
        "kind": "combinations", "n": 6, "k": 2, "max_items": 4,
    })
    assert created.ok
    object_id = created.data["object_id"]

    counted = kernel.apply(object_id, "count")
    assert counted.status == "ok"
    assert counted.trust == TrustLevel.EXACT
    assert counted.data["value"] == 15
    assert any(
        item.arithmetic == "exact_integer"
        for item in counted.evidence_bundle.computation
    )

    generated = kernel.apply(object_id, "generate", {"limit": 4})
    assert generated.data["returned"] == 4
    assert generated.data["truncated"] is True
    assert generated.data["total_count"] == 15


def test_generating_function_coefficient_and_recurrence_are_compositional():
    kernel = MathKernel()
    created = kernel.object_create("generating_function", {
        "kind": "ordinary",
        "coefficients": [0, 1],
        "recurrence": {"coefficients": [1, 1], "initial": [0, 1]},
        "source": "fibonacci",
    })
    assert created.ok
    object_id = created.data["object_id"]

    coefficient = kernel.apply(object_id, "coefficient", {"n": 10})
    assert coefficient.trust == TrustLevel.EXACT
    assert coefficient.data["value"] == 55

    recurrence = kernel.apply(object_id, "recurrence")
    assert recurrence.data["recurrence"] == {
        "coefficients": [1, 1], "initial": [0, 1],
    }
    verified = kernel.apply(object_id, "verify", {"terms": 2})
    assert verified.status == "verified"
    assert verified.semantic_status == ResultStatus.VERIFIED_EXACT


def test_finite_group_operations_carry_exact_evidence():
    kernel = MathKernel()
    created = kernel.object_create("finite_group", {"cyclic": 6})
    assert created.ok
    object_id = created.data["object_id"]

    order = kernel.apply(object_id, "order")
    assert order.status == "verified"
    assert order.trust == TrustLevel.EXACT
    assert order.data["value"] == 6

    center = kernel.apply(object_id, "center")
    assert center.status == "verified"
    assert center.data["elements"] == list(range(6))

    generated = kernel.apply(object_id, "generated_subgroup", {"generators": [2]})
    assert generated.data["elements"] == [0, 2, 4]
    assert generated.evidence_bundle.certificate


def test_finite_field_and_ring_operations_are_exact_and_conditioned():
    kernel = MathKernel()
    ring = kernel.object_create("finite_ring", {"modulus": 12})
    assert ring.ok
    inverse = kernel.apply(ring.data["object_id"], "inverse", {"element": 5})
    assert inverse.status == "verified"
    assert inverse.data["value"]["value"] == 5
    nonunit = kernel.apply(ring.data["object_id"], "inverse", {"element": 8})
    assert nonunit.status == "unknown"
    assert nonunit.semantic_status == ResultStatus.DOES_NOT_EXIST

    field = kernel.object_create("finite_field", {
        "prime": 5, "modulus_coeffs": [2, 0, 1],
    })
    assert field.ok
    product = kernel.apply(
        field.data["object_id"], "multiply",
        {"left": [0, 1], "right": [0, 1]},
    )
    assert product.status == "verified"
    assert product.data["value"]["coeffs"] == [3, 0]
    assert product.arithmetic_transition["computation"] == "exact"


def test_reducible_field_construction_is_refuted_not_silent():
    kernel = MathKernel()
    result = kernel.object_create("finite_field", {
        "prime": 5, "modulus_coeffs": [1, 0, 1],
    })
    assert result.ok
    assert result.status == "refuted"
    assert result.semantic_status == ResultStatus.REFUTED
    assert result.data["verification"]["irreducible"] is False


def test_module_normal_forms_and_abelian_decomposition_are_certified():
    kernel = MathKernel()
    module = kernel.object_create("module", {
        "generators": ["a", "b"], "relations": [[2, 0], [0, 4]],
    })
    assert module.ok
    object_id = module.data["object_id"]

    snf = kernel.apply(object_id, "smith_normal_form")
    assert snf.status == "verified"
    assert all(snf.data["verification"].values())
    assert snf.evidence_bundle.certificate

    decomposition = kernel.apply(object_id, "abelian_group")
    assert decomposition.status == "verified"
    assert decomposition.data["value"]["structure"] == "Z/2Z x Z/4Z"


def test_exact_discrete_capabilities_are_discoverable():
    kernel = MathKernel()
    graph = kernel.capability_query(domain="graph", operation="maximum_flow")
    assert graph["count"] >= 1
    assert graph["capabilities"][0]["handler"] == "module:graph_theory"

    algebra = kernel.capability_query(
        domain="finite_algebra", input_type="FiniteField", operation="multiply")
    assert algebra["capabilities"][0]["trust_levels"] == ["exact", "unknown"]
    assert "field_degree" in algebra["capabilities"][0]["cost_dimensions"]


def test_exact_discrete_typed_objects_and_evidence_survive_store_restart(tmp_path):
    store = tmp_path / "kernel.sqlite"
    kernel = MathKernel(settings=Settings(store_path=str(store)))
    created = kernel.object_create("finite_field", {
        "prime": 5, "modulus_coeffs": [2, 0, 1],
    })
    object_id = created.data["object_id"]
    applied = kernel.apply(object_id, "multiply", {
        "left": [0, 1], "right": [0, 1],
    })
    assert applied.status == "verified"

    restarted = MathKernel(settings=Settings(store_path=str(store)))
    restored = restarted.object_get(object_id)
    assert restored.ok
    assert restored.data["object_type"] == "FiniteField"
    # Applying an operation must not rewrite the source object's construction
    # semantics.  The operation result carries its own exact verification.
    assert restored.data["semantic_status"] == "candidate"
    assert restored.evidence_bundle.computation
    again = restarted.apply(object_id, "inverse", {"element": [3, 2]})
    assert again.status == "verified"
    assert again.trust == TrustLevel.EXACT

    graph = kernel.object_create("graph", {
        "vertices": ["a", "b", "c"],
        "edges": [{"source": "a", "target": "b"}],
    })
    graph_id = graph.data["object_id"]
    assert kernel.apply(graph_id, "centrality").status == "verified"
    restored_graph = restarted.object_get(graph_id)
    assert restored_graph.ok
    assert restored_graph.data["object_type"] == "Graph"
    components = restarted.apply(graph_id, "connected_components")
    assert components.status == "verified"
    assert components.data["components"] == [["a", "b"], ["c"]]
    assert components.evidence_bundle.certificate[0].verified
