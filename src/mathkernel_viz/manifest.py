# =============================================================================
# MathKernel Viz - reproducibility manifest and integrity hashing
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Reproducibility manifests and artifact integrity.

Deterministic mode: given identical input data, configuration, renderer
version and schema, the generated artifact is byte-for-byte reproducible.
Timestamps are excluded unless explicitly supplied by the caller.

Manifests never include environment paths, usernames, secrets, or unrelated
machine metadata.
"""
from __future__ import annotations

import hashlib
import json

from .document import ARTIFACT_SCHEMA, VisualizationDocument

VIZ_VERSION = "1.2.0"


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(doc: VisualizationDocument, *, renderer: str,
                   mathkernel_version: str, deterministic: bool = True,
                   created_utc: str | None = None,
                   computation_id: str | None = None,
                   seed: int | None = None) -> dict:
    dataset_hashes = {k: d.sha256 for k, d in doc.datasets.items() if d.sha256}
    body = doc.model_dump(mode="json")
    body.pop("manifest", None)
    result_hash = sha256_text(canonical_json(body))
    manifest = {
        "mathkernel_version": mathkernel_version,
        "mathkernel_viz_version": VIZ_VERSION,
        "artifact_schema": ARTIFACT_SCHEMA,
        "computation_id": computation_id,
        "result_hash": result_hash,
        "dataset_hashes": dataset_hashes,
        "renderer": renderer,
        "deterministic": deterministic,
    }
    if seed is not None:
        manifest["seed"] = seed
    if not deterministic:
        manifest["created_utc"] = created_utc
    elif created_utc is not None:
        manifest["created_utc"] = created_utc
    return manifest
