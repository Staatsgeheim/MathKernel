# =============================================================================
# MathKernel - evidence foundation tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest
import sympy as sp
from pydantic import ValidationError

from mathkernel import (
    CertificateEvidence,
    ComputationEvidence,
    EmpiricalEvidence,
    EvidenceBundle,
    MathResult,
    ProofEvidence,
    ResultStatus,
    TrustLevel,
    extract_evidence,
    reconcile_claim_evidence,
)
from mathkernel.capabilities import Capability, CapabilityRegistry
from mathkernel_artifacts import MathKernelArtifact


def test_claim_specific_reconciliation_uses_only_required_claims():
    claims = {
        "coefficients": EvidenceBundle(
            computation=[ComputationEvidence(
                engine="exact", method="rational_elimination",
                arithmetic="exact_rational", trust="exact")]),
        "model_validity": EvidenceBundle(
            empirical=[EmpiricalEvidence(sample_size=20, trust="empirical")]),
    }
    assert reconcile_claim_evidence(claims, ["coefficients"]) == "exact"
    assert reconcile_claim_evidence(
        claims, ["coefficients", "model_validity"]) == "empirical"
    assert reconcile_claim_evidence(claims, ["missing"]) == "unknown"


def test_bundle_does_not_launder_numeric_ancestry():
    bundle = EvidenceBundle(computation=[
        ComputationEvidence(engine="input", method="decimal_parse",
                            arithmetic="floating_point", trust="numeric"),
        ComputationEvidence(engine="sympy", method="simplify",
                            arithmetic="symbolic", trust="symbolic"),
    ])
    assert bundle.conservative_trust() == "numeric"


def test_justified_trust_is_a_ceiling_not_an_override():
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="input", method="decimal_parse",
            arithmetic="floating_point", trust="numeric")],
        justified_trust="exact",
    )
    assert bundle.conservative_trust() == "numeric"
    assert EvidenceBundle(justified_trust="exact").conservative_trust() == "unknown"


def test_unverified_proof_and_certificate_do_not_support_a_claim():
    bundle = EvidenceBundle(
        proof=[ProofEvidence(
            proposition="x", method="candidate", engine="solver",
            verified=False, trust="formal")],
        certificate=[CertificateEvidence(
            certificate_type="candidate", claim="x", witness={},
            verifier="unchecked", verified=False, trust="exact")],
    )
    assert bundle.conservative_trust() == "unknown"


def test_unknown_evidence_trust_labels_are_rejected():
    with pytest.raises(ValidationError):
        ComputationEvidence(
            engine="bad", method="bad", arithmetic="bad",
            trust="super_exact")


def test_engine_native_witnesses_serialize_without_execution():
    x = sp.Symbol("x")
    bundle = EvidenceBundle(certificate=[CertificateEvidence(
        certificate_type="symbolic_identity",
        claim="identity",
        witness={"residual": sp.expand((x + 1) ** 2)},
        verifier="independent_checker",
        verified=True,
        trust="symbolic",
    )])
    encoded = bundle.model_dump(mode="json")
    assert encoded["certificate"][0]["witness"]["residual"] == "x**2 + 2*x + 1"


def test_math_result_adds_evidence_without_removing_legacy_fields():
    result = MathResult(ok=True, data={"value": "2"}, trust=TrustLevel.EXACT,
                        engine="integer_exact")
    encoded = result.model_dump(mode="json")
    assert encoded["trust"] == "exact"
    assert encoded["semantic_status"] == "verified_exact"
    assert encoded["evidence_bundle"]["computation"][0]["engine"] == "integer_exact"
    assert "result" in encoded["claim_evidence"]
    assert result.reconciled_trust(["result"]) == TrustLevel.EXACT


def test_math_result_caps_legacy_summary_and_semantic_status_to_evidence():
    bundle = EvidenceBundle(computation=[ComputationEvidence(
        engine="numeric", method="quadrature",
        arithmetic="floating_point", trust="numeric")])
    result = MathResult(
        ok=True,
        trust=TrustLevel.EXACT,
        semantic_status=ResultStatus.PROVED,
        evidence_bundle=bundle,
    )
    assert result.trust == TrustLevel.NUMERIC
    assert result.semantic_status == ResultStatus.NUMERIC
    assert result.claim_evidence["result"].conservative_trust() == "numeric"


def test_math_result_caps_summary_to_all_claims_without_primary_claim():
    result = MathResult(
        ok=True,
        trust=TrustLevel.EXACT,
        claim_evidence={
            "coefficients": EvidenceBundle(computation=[ComputationEvidence(
                engine="exact", method="rational",
                arithmetic="exact_rational", trust="exact")]),
            "model": EvidenceBundle(empirical=[
                EmpiricalEvidence(sample_size=20, trust="empirical")]),
        },
    )
    assert result.trust == TrustLevel.EMPIRICAL
    assert result.reconciled_trust() == TrustLevel.EMPIRICAL
    assert result.evidence_bundle.is_empty()


def test_evidence_extraction_deep_copies_claims_and_merges_support():
    original = EvidenceBundle(computation=[ComputationEvidence(
        engine="exact", method="integer",
        arithmetic="exact_integer", trust="exact")])
    bundle, claims = extract_evidence({"claim_evidence": {"value": original}})
    bundle.computation[0].trust = "numeric"
    assert claims["value"].conservative_trust() == "exact"
    assert original.conservative_trust() == "exact"


def test_math_result_evidence_channels_do_not_alias():
    bundle = EvidenceBundle(computation=[ComputationEvidence(
        engine="exact", method="integer",
        arithmetic="exact_integer", trust="exact")])
    result = MathResult(
        ok=True,
        trust=TrustLevel.EXACT,
        evidence_bundle=bundle,
        claim_evidence={"result": bundle},
    )
    result.evidence_bundle.computation[0].trust = "numeric"
    assert result.claim_evidence["result"].conservative_trust() == "exact"
    assert bundle.conservative_trust() == "exact"


def test_claim_only_result_populates_top_level_bundle():
    result = MathResult(
        ok=True,
        trust=TrustLevel.EXACT,
        claim_evidence={"result": EvidenceBundle(computation=[
            ComputationEvidence(
                engine="exact", method="integer",
                arithmetic="exact_integer", trust="exact")
        ])},
    )
    assert result.evidence_bundle.conservative_trust() == "exact"
    assert result.reconciled_trust() == TrustLevel.EXACT


def test_legacy_trust_without_engine_remains_backward_compatible():
    result = MathResult(ok=True, trust=TrustLevel.EXACT)
    assert result.trust == TrustLevel.EXACT
    assert result.evidence_bundle.computation[0].engine == "unknown"


def test_semantic_failure_status_distinguishes_nonexistence():
    result = MathResult(
        ok=True, status="unknown", data={},
        semantic_status=ResultStatus.DOES_NOT_EXIST)
    assert result.ok
    assert result.semantic_status == ResultStatus.DOES_NOT_EXIST


def test_artifact_roundtrip_preserves_claim_evidence():
    artifact = MathKernelArtifact(
        artifact_id="a",
        claim_evidence={
            "value": EvidenceBundle(computation=[
                ComputationEvidence(engine="integer_exact", method="gcd",
                                    arithmetic="exact_integer", trust="exact")
            ])
        },
    )
    restored = MathKernelArtifact.model_validate(artifact.model_dump(mode="json"))
    assert restored.claim_evidence["value"].conservative_trust() == "exact"


def test_capability_registry_filters_and_merges_engines():
    registry = CapabilityRegistry()
    for engine in ("dijkstra", "bellman_ford"):
        registry.register(Capability(
            name="graph.shortest_path",
            domain="graph",
            operation="shortest_path",
            input_types=("WeightedGraph",),
            output_types=("Path",),
            evidence=("exact", "certificate"),
            engines=(engine,),
        cost_dimensions=("vertices", "edges"),
        handler="module:graphs",
        ))
    found = registry.query(domain="graph", object_type="WeightedGraph",
                           operation="shortest_path", evidence="certificate")
    assert len(found) == 1
    assert found[0].engines == ("bellman_ford", "dijkstra")
    assert found[0].trust_levels == ("exact",)
    assert found[0].verification_methods == ("certificate",)
    assert found[0].cost_dimensions == ("edges", "vertices")
    assert found[0].handler == "module:graphs"
    assert registry.query(
        input_type="WeightedGraph", trust="exact",
        verification_method="certificate") == found
    assert registry.query(input_type="Path") == []
    assert registry.resolve(
        input_type="WeightedGraph", operation="shortest_path") == found[0]


def test_capability_registry_rejects_unknown_trust_claims():
    with pytest.raises(ValueError, match="unknown capability trust"):
        Capability(
            name="invalid", domain="test",
            trust_levels=("super_exact",))
