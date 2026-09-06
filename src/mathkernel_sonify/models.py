# =============================================================================
# MathKernel Sonify - renderer-neutral sonification IR
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Renderer-neutral scientific sonification IR."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
from mathkernel_artifacts import SourceRef, Transformation, ScientificAnnotation

SONIFY_SCHEMA = "mathkernel-sonify/1.0"

class Mapping(BaseModel):
    mapping_id: str
    source: str
    target: Literal["frequency","gain","phase","pan","time","duration","timbre"]
    transform: dict[str, Any] = Field(default_factory=dict)
    units_in: str | None = None
    units_out: str | None = None
    notes: str = ""

class SonificationEvent(BaseModel):
    event_id: str
    time: float
    duration: float
    source_ref: str | None = None
    label: str = ""
    values: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

class AudioTrack(BaseModel):
    track_id: str
    label: str = ""
    role: Literal["data","prediction","observation","residual","control"] = "data"
    source_ref: SourceRef | None = None
    mapping_refs: list[str] = Field(default_factory=list)
    events: list[SonificationEvent] = Field(default_factory=list)
    gain: float = 1.0
    pan: float = 0.0
    trust: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

class RenderConfig(BaseModel):
    sample_rate: int = 48000
    channels: Literal[1,2] = 2
    bit_depth: Literal[16,24,32] = 24
    peak_limit: float = 0.90
    normalization: Literal["none","per_track","per_group","global","reference_relative"] = "global"
    deterministic: bool = True

class SonificationDocument(BaseModel):
    schema_version: str = "1.0"
    artifact_schema: str = SONIFY_SCHEMA
    title: str = "MathKernel sonification"
    sources: dict[str, SourceRef] = Field(default_factory=dict)
    mappings: dict[str, Mapping] = Field(default_factory=dict)
    tracks: list[AudioTrack] = Field(default_factory=list)
    transformations: list[Transformation] = Field(default_factory=list)
    annotations: list[ScientificAnnotation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    render: RenderConfig = Field(default_factory=RenderConfig)
    trust: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def duration(self) -> float:
        return max((e.time + e.duration for t in self.tracks for e in t.events), default=0.0)
