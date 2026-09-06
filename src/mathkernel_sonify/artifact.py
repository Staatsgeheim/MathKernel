# =============================================================================
# MathKernel Sonify - sonification to shared artifact bridge
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Bridge sonification IR into shared MathKernelArtifact."""
from __future__ import annotations
from mathkernel_artifacts import MathKernelArtifact, SynchronizationLink
from .models import SonificationDocument

def attach_to_artifact(artifact: MathKernelArtifact, doc: SonificationDocument) -> MathKernelArtifact:
    out=artifact.model_copy(deep=True)
    out.sources.update(doc.sources); out.transformations.extend(doc.transformations); out.annotations.extend(doc.annotations)
    out.sonifications.append(doc.model_dump(mode="json"))
    return out

def synchronize(source_ref: str, sonification_ref: str, *, visual_ref: str|None=None, source_range=None, time_range=None, link_id="sync-1") -> SynchronizationLink:
    return SynchronizationLink(link_id=link_id,source_ref=source_ref,visual_ref=visual_ref,sonification_ref=sonification_ref,source_range=source_range,time_range=time_range)
