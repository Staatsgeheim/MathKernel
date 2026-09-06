# =============================================================================
# MathKernel Viz - visualization to shared artifact migration bridge
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Bridge between visualization IR and the shared MathKernel artifact model."""
from __future__ import annotations
import hashlib, json
from typing import Any
from mathkernel_artifacts import (
    EvidenceItem, MathKernelArtifact, Reproducibility, ScientificAnnotation,
    SourceRef, extract_engine_versions, extract_evidence,
)
from .document import VisualizationDocument

def _hash(obj: Any) -> str:
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True,default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

def to_artifact(doc: VisualizationDocument, *, result: dict | None = None,
                artifact_id: str = "", mathkernel_version: str | None = None) -> MathKernelArtifact:
    linked_result = result if result is not None else doc.linked_result
    sources: dict[str, SourceRef] = {}
    for d in doc.datasets.values():
        if d.source_ref:
            sources[d.source_ref.source_id]=d.source_ref
    for s in doc.series.values():
        if s.source_ref:
            sources[s.source_ref.source_id]=s.source_ref
    evidence={}
    for step in doc.provenance.steps:
        eid=f"derivation:{step.step_id}"
        evidence[eid]=EvidenceItem(evidence_id=eid, kind="derivation",
            claim=step.output or step.operation, trust=step.trust, engine=step.engine,
            conditions=step.conditions, metadata={"operation":step.operation,
                                                   "parents":step.parents,
                                                   "inputs":step.inputs})
    # Preserve richer caller-supplied evidence without forcing a schema migration.
    for key, value in doc.evidence.items():
        if isinstance(value, EvidenceItem): evidence[key]=value
        elif isinstance(value, dict): evidence[key]=EvidenceItem(evidence_id=key, **value)
    anns=[]
    for i,a in enumerate(doc.annotations):
        anns.append(ScientificAnnotation(annotation_id=f"viz-ann-{i+1}", text=a.text,
            kind=a.kind if a.kind in {"mathematical","perceptual","candidate","validated_claim","caption","warning","note"} else "note",
            claim_status=a.claim_status if a.claim_status in {"none","candidate","validated","rejected"} else "none",
            author_type=a.author_type if a.author_type in {"human","llm","system","unknown"} else "unknown",
            source_refs=a.source_refs,evidence_refs=a.evidence_refs,position=a.position,
            trust=a.trust,metadata=a.metadata))
    datasets={k:v.sha256 for k,v in doc.datasets.items() if v.sha256}
    body=doc.model_dump(mode="json")
    repro=Reproducibility(mathkernel_version=mathkernel_version,
        result_hash=_hash(linked_result) if linked_result is not None else None,
        dataset_hashes=datasets,
        engine_versions=extract_engine_versions(linked_result),
        deterministic=True)
    bundle, claim_evidence = extract_evidence(linked_result)
    return MathKernelArtifact(artifact_id=artifact_id,title=doc.title,trust=doc.trust,
        result=linked_result,sources=sources,evidence=evidence,evidence_bundle=bundle,
        claim_evidence=claim_evidence,transformations=doc.transformations,
        annotations=anns,assumptions=doc.assumptions,visualizations=[body],
        reproducibility=repro,metadata={"visualization_schema":doc.artifact_schema})

def attach_result_lineage(doc: VisualizationDocument, result: Any) -> SourceRef:
    """Attach one structured source reference to unbound visual data.

    A content hash is used when MathResult has no stable externally assigned id,
    so lineage remains deterministic rather than relying on process-local ids.
    """
    raw = result if isinstance(result, dict) else result.model_dump(mode="json")
    rid = raw.get("result_id") or raw.get("id")
    digest = _hash(raw)
    sid = f"mathresult:{rid}" if rid else f"mathresult:sha256:{digest}"
    ref = SourceRef(source_id=sid, kind="math_result", result_id=rid,
                    sha256=digest, trust=str(raw.get("trust", "unknown")),
                    metadata={"engine": raw.get("engine")})
    for d in doc.datasets.values():
        if d.source_ref is None:
            d.source_ref = ref.model_copy(deep=True)
    for s in doc.series.values():
        if s.source_ref is None:
            s.source_ref = ref.model_copy(deep=True)
    return ref
