# =============================================================================
# MathKernel Multimodal - assemble MathKernelArtifact from sensory documents
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Assemble a shared ``MathKernelArtifact`` from visualization and
sonification documents.

The assembler merges source lineage, evidence, transformations, annotations
and trust from both sensory frontends and derives cross-modal
``SynchronizationLink`` records where a sonification event and a visual block
resolve to the same mathematical source.  It never recomputes mathematics and
never upgrades trust: the artifact trust is the weakest evidence in it.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from mathkernel_artifacts import (
    TRUST_RANK,
    MathKernelArtifact,
    Reproducibility,
    SynchronizationLink,
    extract_engine_versions,
    extract_evidence,
    merge_evidence_bundles,
)
from mathkernel_viz.document import VisualizationDocument
from mathkernel_viz.artifact import to_artifact as _viz_to_artifact
from mathkernel_sonify.models import SonificationDocument


def _hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def weakest_trust(levels: Iterable[str]) -> str:
    known = [level for level in levels if level in TRUST_RANK]
    return min(known, key=TRUST_RANK.__getitem__) if known else "unknown"


def _as_viz(doc: VisualizationDocument | dict) -> VisualizationDocument:
    return doc if isinstance(doc, VisualizationDocument) \
        else VisualizationDocument.model_validate(doc)


def _as_sonify(doc: SonificationDocument | dict) -> SonificationDocument:
    return doc if isinstance(doc, SonificationDocument) \
        else SonificationDocument.model_validate(doc)


def _viz_block_source_ids(doc: VisualizationDocument, block_id: str) -> set[str]:
    refs: set[str] = set()
    block = next((b for b in doc.blocks if b.block_id == block_id), None)
    if block is None:
        return refs
    for sid in block.series:
        s = doc.series.get(sid)
        if s is not None and s.source_ref is not None:
            refs.add(s.source_ref.source_id)
    for did in block.datasets:
        d = doc.datasets.get(did)
        if d is not None and d.source_ref is not None:
            refs.add(d.source_ref.source_id)
    return refs


def derive_synchronization(viz_docs: list[VisualizationDocument],
                           son_docs: list[SonificationDocument],
                           *, start: int = 0) -> list[SynchronizationLink]:
    """Link every sonification event to visual blocks sharing its source.

    ``visual_ref`` is ``"v<i>:<block_id>"`` (visualization index + block) and
    ``sonification_ref`` is ``"s<j>"`` (sonification index); the event's
    playback interval becomes the link ``time_range`` so a viewer can resolve
    audio time to visual objects and visual selections back to audio time.
    """
    links: list[SynchronizationLink] = []
    n = start
    for j, sd in enumerate(son_docs):
        for track in sd.tracks:
            for ev in track.events:
                if not ev.source_ref:
                    continue
                for i, vd in enumerate(viz_docs):
                    for block in vd.blocks:
                        if ev.source_ref not in _viz_block_source_ids(
                                vd, block.block_id):
                            continue
                        n += 1
                        links.append(SynchronizationLink(
                            link_id=f"sync-{n}",
                            source_ref=ev.source_ref,
                            visual_ref=f"v{i}:{block.block_id}",
                            sonification_ref=f"s{j}",
                            time_range=[ev.time, ev.time + ev.duration],
                            metadata={"event_id": ev.event_id,
                                      "track_id": track.track_id}))
    return links


def build_artifact(*, title: str = "MathKernel research artifact",
                   visualizations: Iterable[VisualizationDocument | dict] = (),
                   sonifications: Iterable[SonificationDocument | dict] = (),
                   result: dict | None = None,
                   artifact_id: str = "",
                   mathkernel_version: str | None = None,
                   synchronization: Iterable[SynchronizationLink | dict] = (),
                   auto_synchronize: bool = True,
                   assumptions: Iterable[str] = (),
                   metadata: dict | None = None,
                   deterministic: bool = True,
                   created_utc: str | None = None) -> MathKernelArtifact:
    """Combine visualization/sonification documents into one shared artifact."""
    viz_docs = [_as_viz(d) for d in visualizations]
    son_docs = [_as_sonify(d) for d in sonifications]

    art = MathKernelArtifact(artifact_id=artifact_id, title=title)
    evidence_bundles = []
    claim_evidence = {}
    engine_versions: dict[str, str] = {}

    for index, vd in enumerate(viz_docs):
        sub = _viz_to_artifact(vd)          # tested viz -> artifact bridge
        art.sources.update(sub.sources)
        art.evidence.update(sub.evidence)
        engine_versions.update(sub.reproducibility.engine_versions)
        if not sub.evidence_bundle.is_empty():
            evidence_bundles.append(sub.evidence_bundle)
        for name, bundle in sub.claim_evidence.items():
            claim_evidence[f"visualization:{index}:{name}"] = (
                bundle.model_copy(deep=True))
        art.transformations.extend(sub.transformations)
        art.annotations.extend(sub.annotations)
        for a in vd.assumptions:
            if a not in art.assumptions:
                art.assumptions.append(a)
        art.visualizations.append(vd.model_dump(mode="json"))

    for sd in son_docs:
        art.sources.update(sd.sources)
        art.transformations.extend(sd.transformations)
        art.annotations.extend(sd.annotations)
        for a in sd.assumptions:
            if a not in art.assumptions:
                art.assumptions.append(a)
        art.sonifications.append(sd.model_dump(mode="json"))

    for a in assumptions:
        if a not in art.assumptions:
            art.assumptions.append(a)

    art.result = result
    if isinstance(result, dict):
        engine_versions.update(extract_engine_versions(result))
        result_bundle, result_claims = extract_evidence(result)
        if not result_bundle.is_empty():
            evidence_bundles.append(result_bundle)
        claim_evidence.update({
            key: bundle.model_copy(deep=True)
            for key, bundle in result_claims.items()
        })
    art.evidence_bundle = merge_evidence_bundles(tuple(evidence_bundles))
    art.claim_evidence = claim_evidence
    if not art.evidence_bundle.is_empty():
        art.claim_evidence["result"] = art.evidence_bundle.model_copy(deep=True)
    trust_inputs = [d.trust for d in viz_docs] + [d.trust for d in son_docs]
    if isinstance(result, dict) and result.get("trust"):
        trust_inputs.append(str(result["trust"]))
    if not art.evidence_bundle.is_empty():
        trust_inputs.append(art.evidence_bundle.conservative_trust())
    art.trust = weakest_trust(trust_inputs)

    links = [l if isinstance(l, SynchronizationLink)
             else SynchronizationLink.model_validate(l)
             for l in synchronization]
    if auto_synchronize:
        links.extend(derive_synchronization(viz_docs, son_docs,
                                            start=len(links)))
    art.synchronization = links

    dataset_hashes: dict[str, str] = {}
    for vd in viz_docs:
        for k, d in vd.datasets.items():
            if d.sha256:
                dataset_hashes[k] = d.sha256
    for sd in son_docs:
        for k, s in sd.sources.items():
            if s.sha256:
                dataset_hashes[k] = s.sha256
    art.reproducibility = Reproducibility(
        mathkernel_version=mathkernel_version,
        created_utc=created_utc,
        result_hash=_hash(result) if result is not None else None,
        dataset_hashes=dataset_hashes,
        engine_versions=engine_versions,
        deterministic=deterministic)

    integrity: dict[str, str] = {}
    for i, vd in enumerate(viz_docs):
        integrity[f"visualization:{i}"] = _hash(art.visualizations[i])
    for j, sd in enumerate(son_docs):
        integrity[f"sonification:{j}"] = _hash(art.sonifications[j])
    art.integrity = integrity

    art.metadata = dict(metadata or {})
    art.metadata.setdefault("visualization_count", len(viz_docs))
    art.metadata.setdefault("sonification_count", len(son_docs))
    return art
