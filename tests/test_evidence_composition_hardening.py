# =============================================================================
# MathKernel - evidence/composition hardening regressions
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.finite_algebra import create_finite_ring
from mathkernel.graph_theory import WeightedGraph, shortest_paths
from mathkernel.kernel import _decode_typed_value
from mathkernel.models import ResultStatus
from mathkernel_artifacts import (
    ComputationEvidence,
    EvidenceBundle,
    ProofEvidence,
    merge_evidence_bundles,
)


def test_optional_failed_cross_check_does_not_downgrade_required_path():
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="sympy", method="symbolic_solution", arithmetic="symbolic",
            trust="symbolic",
        )],
        proof=[ProofEvidence(
            proposition="candidate set is complete", method="smt_cross_check",
            engine="z3", verified=False, trust="formal", role="cross_check",
        )],
    )
    assert bundle.conservative_trust() == "symbolic"


def test_alternative_support_path_can_establish_claim_when_other_path_fails():
    bundle = EvidenceBundle(proof=[
        ProofEvidence(
            proposition="P", method="unavailable_prover", engine="a",
            verified=False, trust="formal", support_path="path_a",
        ),
        ProofEvidence(
            proposition="P", method="exact_certificate", engine="b",
            verified=True, trust="exact", support_path="path_b",
        ),
    ])
    assert bundle.conservative_trust() == "exact"


def test_merge_preserves_global_ancestry_ceiling_across_alternative_paths():
    numeric_input = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="mathir", method="required_input", arithmetic="inherited",
            trust="numeric",
        )],
        justified_trust="numeric",
    )
    formal_alternative = EvidenceBundle(proof=[ProofEvidence(
        proposition="P", method="formal_proof", engine="lean",
        verified=True, trust="formal", support_path="formal_path",
    )])
    merged = merge_evidence_bundles((numeric_input, formal_alternative))
    assert merged.justified_trust == "numeric"
    assert merged.conservative_trust() == "numeric"


def test_querying_typed_object_does_not_mutate_source_scientific_node():
    kernel = MathKernel()
    created = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"], "variable": "x",
    })
    object_id = created.data["object_id"]
    before = kernel.object_get(object_id)
    queried = kernel.apply(object_id, "pdf", {"point": "0"})
    after = kernel.object_get(object_id)

    assert queried.ok
    assert set(before.claim_evidence) == {"construction"}
    assert set(after.claim_evidence) == {"construction"}
    assert before.data["object"] == after.data["object"]
    assert before.data["semantic_status"] == after.data["semantic_status"]


@pytest.mark.parametrize("payload", [
    "__import__('os').getcwd()",
    "eval('1+1')",
    "Symbol.__mro__",
    "(lambda: 1)()",
])
def test_persisted_sympy_decoder_rejects_executable_or_dynamic_syntax(payload):
    with pytest.raises(ValueError):
        _decode_typed_value({"__sympy_srepr__": payload})


def test_bellman_ford_ignores_negative_cycle_that_cannot_reach_target():
    graph = WeightedGraph(
        vertices=["s", "t", "a", "b"],
        edges=[
            {"source": "s", "target": "t", "weight": 5},
            {"source": "s", "target": "a", "weight": 0},
            {"source": "a", "target": "b", "weight": -2},
            {"source": "b", "target": "a", "weight": 1},
        ],
        directed=True,
    )
    result = shortest_paths(graph, "s", "t")
    assert result.status.value == "verified"
    assert result.trust == TrustLevel.EXACT
    assert result.distances["t"] == 5
    assert result.path == ["s", "t"]


def test_numeric_contour_cannot_be_laundered_into_exact_winding_number():
    kernel = MathKernel()
    contour = kernel.object_create("Contour", {
        "vertices": [
            "complex(-1.1,-1.1)", "complex(1.1,-1.1)",
            "complex(1.1,1.1)", "complex(-1.1,1.1)",
            "complex(-1.1,-1.1)",
        ],
        "orientation": "ccw",
    })
    result = kernel.apply(contour.data["object_id"], "winding_number", {
        "point": "0",
    })
    assert result.trust == TrustLevel.NUMERIC
    assert result.semantic_status == ResultStatus.VERIFIED_NUMERIC
    assert result.data["provenance"]["arithmetic_transition"][
        "required_input_trust"] == "numeric"


def test_numeric_secondary_object_caps_cross_object_complex_claim():
    kernel = MathKernel()
    expression = kernel.parse("1/z")
    function = kernel.object_create("ComplexFunction", {
        "expression_id": expression.data["expr_id"], "variable": "z",
    })
    contour = kernel.object_create("Contour", {
        "vertices": [
            "complex(-1.1,-1.1)", "complex(1.1,-1.1)",
            "complex(1.1,1.1)", "complex(-1.1,1.1)",
            "complex(-1.1,-1.1)",
        ],
        "orientation": "ccw",
    })
    result = kernel.apply(function.data["object_id"], "argument_principle", {
        "contour_id": contour.data["object_id"],
    })
    assert result.trust == TrustLevel.NUMERIC
    dependencies = result.data["provenance"]["required_object_inputs"]
    assert {item["object_type"] for item in dependencies} == {
        "ComplexFunction", "Contour",
    }


def test_numeric_mathir_parameter_caps_exact_contour_operation():
    kernel = MathKernel()
    contour = kernel.object_create("Contour", {
        "vertices": [
            "complex(-1,-1)", "complex(1,-1)", "complex(1,1)",
            "complex(-1,1)", "complex(-1,-1)",
        ],
        "orientation": "ccw",
    })
    result = kernel.apply(contour.data["object_id"], "winding_number", {
        "point": "0.1",
    })
    assert result.trust == TrustLevel.NUMERIC
    assert result.data["provenance"]["arithmetic_transition"][
        "required_input_trust"] == "numeric"


def test_large_ring_construction_exact_but_primality_not_claimed_exact():
    # 2^127-1 is prime, but the built-in deterministic Miller-Rabin guarantee
    # intentionally ends below 2^64.  Passing the fixed bases is therefore only
    # probable-prime evidence unless a separate certificate is supplied.
    result = create_finite_ring((1 << 127) - 1)
    ring = result.value
    assert result.trust == TrustLevel.EXACT.value
    assert ring.primality_status == "probable_prime"
    assert ring.primality_exact is False
    assert ring.modulus_is_prime is None
    assert result.claim_evidence["modulus_primality"].conservative_trust() == "heuristic"


def test_roc_check_name_states_only_what_is_actually_verified():
    kernel = MathKernel()
    expression = kernel.parse("exp(-t)")
    problem = kernel.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
    })
    result = kernel.apply(problem.data["object_id"], "apply")
    names = {item["name"] for item in result.data["verified_checks"]}
    assert "roc_nonempty_consistency" in names
    assert "roc_consistency" not in names


def test_relational_context_discharges_distribution_parameter_obligation():
    kernel = MathKernel()
    context = kernel.create_context(
        {"a": "real", "b": "real"}, ["b > a"])
    created = kernel.object_create("Distribution", {
        "family": "uniform", "parameters": ["a", "b"], "variable": "x",
        "context_id": context.context_id,
    })
    assert created.ok
    assert created.data["object"]["conditions"] == []
    assert created.data["object"]["assumptions"] == ["b > a"]

    result = kernel.apply(created.data["object_id"], "mean")
    assert result.side_conditions == []
    assert result.assumptions_used == ["b > a"]


def test_relational_context_controls_parameter_dependent_moment_existence():
    kernel = MathKernel()
    context = kernel.create_context({"nu": "real"}, ["nu > 2"])
    created = kernel.object_create("Distribution", {
        "family": "student_t", "parameters": ["nu"], "variable": "x",
        "context_id": context.context_id,
    })
    result = kernel.apply(created.data["object_id"], "variance")
    assert result.ok
    assert result.data["value"] == "nu/(nu - 2)"
    assert result.assumptions_used == ["nu > 2"]


def test_persistence_rejects_unregistered_pydantic_model_even_in_allowed_module():
    payload = {
        "__pydantic_model__": "mathkernel.complex_analysis:SymbolicModel",
        "fields": {},
    }
    with pytest.raises(ValueError, match="stored model type is not allowed"):
        _decode_typed_value(payload)
