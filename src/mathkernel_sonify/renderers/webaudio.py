# =============================================================================
# MathKernel Sonify - portable WebAudio payload
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Portable WebAudio payload; browser artifact layers can consume this directly."""
from __future__ import annotations
from ..models import SonificationDocument

def webaudio_payload(doc: SonificationDocument) -> dict:
    return {"schema":doc.artifact_schema,"title":doc.title,"render":doc.render.model_dump(mode='json'),
            "mappings":{k:v.model_dump(mode='json') for k,v in doc.mappings.items()},
            "tracks":[t.model_dump(mode='json') for t in doc.tracks],"duration":doc.duration(),
            "trust":doc.trust,"metadata":doc.metadata}
