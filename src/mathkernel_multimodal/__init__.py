# =============================================================================
# MathKernel Multimodal - unified evidence-carrying research artifacts
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""mathkernel_multimodal: unified portable research artifacts.

Assembles one ``MathKernelArtifact`` from visualization documents
(``mathkernel_viz``) and sonification documents (``mathkernel_sonify``),
derives cross-modal synchronization links from shared source lineage, and
exports a single self-contained interactive HTML file with unified Result /
Evidence / Provenance / Data / Reproduction / Visual Mapping / Audio Mapping /
Sync / Annotations inspectors.

    import mathkernel_multimodal as mkm

    artifact = mkm.build_artifact(title="Result",
                                  visualizations=[viz_doc],
                                  sonifications=[son_doc])
    mkm.export_html(artifact, "result.html")

This package owns no mathematics, no visual rendering and no audio synthesis:
it packages the shared artifact boundary defined in ``mathkernel_artifacts``.
"""
from __future__ import annotations

from mathkernel_artifacts import (MathKernelArtifact, SourceRef, EvidenceItem,
                                  Transformation, ScientificAnnotation,
                                  Reproducibility, SynchronizationLink)
from .assemble import build_artifact, derive_synchronization, weakest_trust
from .exporter import render_html, export_html

__all__ = [
    "MathKernelArtifact", "SourceRef", "EvidenceItem", "Transformation",
    "ScientificAnnotation", "Reproducibility", "SynchronizationLink",
    "build_artifact", "derive_synchronization", "weakest_trust",
    "render_html", "export_html",
]
