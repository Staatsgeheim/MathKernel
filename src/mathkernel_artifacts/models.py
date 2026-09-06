# =============================================================================
# MathKernel Artifacts - shared scientific artifact model
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Renderer-neutral scientific artifact semantics shared by sensory frontends.

This module deliberately contains no visualization, audio, browser, solver or
MathKernel execution code.  It records what a result *is*, where it came from,
how it was transformed for presentation, and what evidence supports it.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator
from .evidence import (
    TRUST_RANK,
    EvidenceBundle,
    merge_evidence_bundles,
    reconcile_claim_evidence,
)

ARTIFACT_SCHEMA = "mathkernel-artifact/1.0"

class SourceRef(BaseModel):
    """Stable lineage pointer to mathematical or external source data."""
    source_id: str
    kind: Literal["math_result", "dataset", "expression", "derivation_step",
                  "external_dataset", "observation", "prediction", "other"] = "other"
    result_id: str | None = None
    dataset_id: str | None = None
    expression_id: str | None = None
    step_id: str | None = None
    path: str | None = None
    uri: str | None = None
    label: str = ""
    sha256: str | None = None
    trust: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

class EvidenceItem(BaseModel):
    evidence_id: str
    kind: Literal["derivation", "formal_certificate", "smt_certificate",
                  "interval_certificate", "exact_check", "numeric_validation",
                  "empirical_validation", "reference", "other"] = "other"
    claim: str = ""
    trust: str = "unknown"
    engine: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    certificate: Any | None = None
    certificate_sha256: str | None = None
    conditions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class Transformation(BaseModel):
    """Declarative presentation/data transformation; never executable code."""
    transformation_id: str
    operation: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    purpose: Literal["presentation", "projection", "normalization", "sampling",
                     "aggregation", "comparison", "encoding", "other"] = "presentation"
    deterministic: bool = True
    seed: int | None = None
    trust: str = "unknown"
    notes: str = ""

class ScientificAnnotation(BaseModel):
    annotation_id: str
    text: str
    kind: Literal["mathematical", "perceptual", "candidate", "validated_claim",
                  "caption", "warning", "note"] = "note"
    claim_status: Literal["none", "candidate", "validated", "rejected"] = "none"
    author_type: Literal["human", "llm", "system", "unknown"] = "unknown"
    source_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    position: list[float] | None = None
    time_range: list[float] | None = None
    trust: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

class Reproducibility(BaseModel):
    mathkernel_version: str | None = None
    artifact_schema: str = ARTIFACT_SCHEMA
    created_utc: str | None = None
    computation_id: str | None = None
    input_hash: str | None = None
    result_hash: str | None = None
    dataset_hashes: dict[str, str] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=dict)
    deterministic: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

class SynchronizationLink(BaseModel):
    """Links coordinates/events across visual, auditory and source domains."""
    link_id: str
    source_ref: str
    visual_ref: str | None = None
    sonification_ref: str | None = None
    source_range: list[float] | None = None
    time_range: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class MathKernelArtifact(BaseModel):
    """Top-level evidence-carrying, multimodal scientific result."""
    schema_version: str = "1.0"
    artifact_schema: str = ARTIFACT_SCHEMA
    artifact_id: str = ""
    title: str = "MathKernel artifact"
    summary: str = ""
    trust: str = "unknown"
    result: dict[str, Any] | None = None
    sources: dict[str, SourceRef] = Field(default_factory=dict)
    evidence: dict[str, EvidenceItem] = Field(default_factory=dict)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)
    transformations: list[Transformation] = Field(default_factory=list)
    annotations: list[ScientificAnnotation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    visualizations: list[Any] = Field(default_factory=list)
    sonifications: list[Any] = Field(default_factory=list)
    synchronization: list[SynchronizationLink] = Field(default_factory=list)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)
    integrity: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reconcile_artifact_evidence(self) -> "MathKernelArtifact":
        if self.evidence_bundle.is_empty() and self.claim_evidence:
            self.evidence_bundle = (
                self.claim_evidence["result"].model_copy(deep=True)
                if "result" in self.claim_evidence
                else merge_evidence_bundles(
                    tuple(self.claim_evidence.values()))
            )
        if not self.claim_evidence and not self.evidence_bundle.is_empty():
            self.claim_evidence["result"] = self.evidence_bundle.model_copy(
                deep=True)
        else:
            self.evidence_bundle = self.evidence_bundle.model_copy(deep=True)
            self.claim_evidence = {
                name: bundle.model_copy(deep=True)
                for name, bundle in self.claim_evidence.items()
            }

        support: list[str] = [
            item.trust for item in self.evidence.values()
            if item.trust in TRUST_RANK
        ]
        if self.claim_evidence:
            support.append(reconcile_claim_evidence(self.claim_evidence))
        elif not self.evidence_bundle.is_empty():
            support.append(self.evidence_bundle.conservative_trust())
        support = [level for level in support if level in TRUST_RANK]
        if support and self.trust in TRUST_RANK:
            evidence_ceiling = min(support, key=TRUST_RANK.__getitem__)
            if TRUST_RANK[evidence_ceiling] < TRUST_RANK[self.trust]:
                self.trust = evidence_ceiling
        return self
