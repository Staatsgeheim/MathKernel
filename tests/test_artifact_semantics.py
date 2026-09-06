# =============================================================================
# MathKernel - shared artifact semantics tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel_artifacts import (
    ComputationEvidence, EvidenceBundle, EvidenceItem, MathKernelArtifact,
    ScientificAnnotation, SourceRef, SynchronizationLink, Transformation,
)
import mathkernel_viz as viz


def test_shared_artifact_is_multimodal_and_evidence_carrying():
    a=MathKernelArtifact(
        artifact_id="a1", title="result", trust="numeric",
        sources={"s": SourceRef(source_id="s", kind="observation", trust="numeric")},
        evidence={"e": EvidenceItem(evidence_id="e", kind="numeric_validation", trust="numeric")},
        transformations=[Transformation(transformation_id="t", operation="project", inputs=["s"], outputs=["v"])],
        annotations=[ScientificAnnotation(annotation_id="n", text="heard tone", kind="perceptual", claim_status="candidate")],
        visualizations=[{"schema":"viz"}], sonifications=[{"schema":"sonify"}],
        synchronization=[SynchronizationLink(link_id="l", source_ref="s", visual_ref="v", sonification_ref="audio")])
    assert a.visualizations and a.sonifications
    assert a.annotations[0].claim_status == "candidate"


def test_visualization_keeps_structured_source_lineage():
    result={"result_id":"r42","trust":"numeric","engine":"test","data":{"points":[[0,1],[1,2]]}}
    doc=viz.visualize(result)
    refs=[s.source_ref for s in doc.series.values()] + [d.source_ref for d in doc.datasets.values()]
    assert refs and all(r is not None for r in refs)
    assert all(r.result_id == "r42" for r in refs)


def test_visualization_to_shared_artifact_preserves_derivation_and_transformations():
    result={"result_id":"r1","trust":"exact","engine":"sympy","data":{"points":[[0,0],[1,1]]},
            "derivation":[{"step_id":"s1","operation":"derive","inputs":[],"parents":[],"output":"x","engine":"sympy","trust":"exact","conditions":[]}]}
    doc=viz.visualize(result)
    doc.transformations.append(Transformation(transformation_id="tx", operation="linear_map", purpose="presentation"))
    art=viz.to_artifact(doc,result=result,artifact_id="artifact-1",mathkernel_version="1.2.0")
    assert art.artifact_schema == "mathkernel-artifact/1.0"
    assert art.sources
    assert art.transformations[0].transformation_id == "tx"
    assert art.reproducibility.result_hash


def test_visualization_automatically_carries_and_caps_linked_evidence():
    result = {
        "result_id": "r2",
        "trust": "exact",
        "engine": "numeric",
        "data": {"points": [[0, 0], [1, 1]]},
        "evidence_bundle": {
            "computation": [{
                "engine": "numeric",
                "method": "sample",
                "arithmetic": "floating_point",
                "trust": "numeric",
            }],
        },
    }
    result["data"]["provenance"] = {
        "engine_versions": {
            "mathkernel": "1.2.0", "sympy": "test-version",
        },
    }
    doc = viz.visualize(result)
    art = viz.to_artifact(doc)
    assert doc.linked_result["result_id"] == "r2"
    assert art.evidence_bundle.conservative_trust() == "numeric"
    assert art.claim_evidence["result"].conservative_trust() == "numeric"
    assert art.trust == "numeric"
    assert art.reproducibility.engine_versions["sympy"] == "test-version"


def test_artifact_trust_cannot_exceed_claim_evidence():
    bundle = EvidenceBundle(computation=[ComputationEvidence(
        engine="numeric", method="sample",
        arithmetic="floating_point", trust="numeric")])
    artifact = MathKernelArtifact(
        artifact_id="capped",
        trust="exact",
        claim_evidence={"result": bundle},
    )
    assert artifact.trust == "numeric"


def test_typed_symbolic_result_gets_domain_condition_blocks():
    result = {
        "trust": "symbolic",
        "semantic_status": "candidate",
        "engine": "integral_transforms",
        "data": {
            "value": "1/(s + 1)",
            "roc": "re(s) > -1",
            "verified_checks": [{"name": "roc_consistency",
                                 "status": "verified"}],
        },
    }
    document = viz.visualize(result)
    assert [block.kind for block in document.blocks] == [
        "metric_grid", "data_table"]
    assert document.linked_result["data"]["roc"] == "re(s) > -1"


def test_annotation_semantics_survive_bridge():
    doc=viz.dashboard("x")
    doc.annotations.append(viz.Annotation(text="possible anomaly", kind="candidate",
                                          claim_status="candidate", author_type="human"))
    art=viz.to_artifact(doc)
    assert art.annotations[0].kind == "candidate"
    assert art.annotations[0].claim_status == "candidate"
