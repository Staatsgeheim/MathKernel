# =============================================================================
# MathKernel Artifacts - shared scientific artifact schema
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Shared evidence/provenance schema for MathKernel research artifacts."""
from .evidence import (
    CertificateEvidence, ComputationEvidence, EmpiricalEvidence, EvidenceBundle,
    ModelEvidence, NumericalEvidence, ProofEvidence, extract_engine_versions,
    extract_evidence,
    merge_evidence_bundles, reconcile_claim_evidence, tag_evidence_bundle, TRUST_RANK, EvidenceRole,
)
from .models import (ARTIFACT_SCHEMA, EvidenceItem, MathKernelArtifact,
    Reproducibility, ScientificAnnotation, SourceRef, SynchronizationLink,
    Transformation)
__all__ = ["ARTIFACT_SCHEMA", "MathKernelArtifact", "SourceRef", "EvidenceItem",
           "Transformation", "ScientificAnnotation", "Reproducibility",
           "SynchronizationLink", "EvidenceBundle", "ComputationEvidence",
           "ProofEvidence", "CertificateEvidence", "NumericalEvidence",
           "ModelEvidence", "EmpiricalEvidence", "extract_evidence",
           "extract_engine_versions",
           "merge_evidence_bundles", "reconcile_claim_evidence", "tag_evidence_bundle", "TRUST_RANK", "EvidenceRole"]
