# =============================================================================
# MathKernel Viz - renderer-neutral visualization document (artifact schema 2.0)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""VisualizationDocument: the stable boundary between mathematics and rendering.

Schema 2.0 models every artifact as a **composition of small, domain-agnostic
building blocks** arranged on a grid layout:

* blocks (point clouds, plots, histograms, tables, metric grids, text,
  selectors, ...) are self-contained panel specifications;
* blocks reference shared, content-addressed datasets and series by id;
* controls (``select`` blocks) publish parameters that other blocks bind to,
  giving linked, interactive dashboards without any domain-specific wiring.

Renderers consume this IR; adapters produce it from MathKernel results.  The
artifact layer never recomputes mathematics — it packages already-produced
MathResult structures together with their evidence.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from mathkernel_artifacts import (
    TRUST_RANK,
    ScientificAnnotation,
    SourceRef,
    Transformation,
)

ARTIFACT_SCHEMA = "mathkernel-viz/2.0"

TRUST_LEVELS = tuple(sorted(
    TRUST_RANK,
    key=TRUST_RANK.__getitem__,
    reverse=True,
))

BLOCK_KINDS = (
    # 3D (WebGL / Three.js in the browser viewer)
    "point_cloud_3d", "trajectory_3d", "surface_3d", "vector_field_3d",
    # 2D canvas
    "plot2d", "histogram", "heatmap", "dag",
    # content
    "metric_grid", "data_table", "text",
    # control
    "select",
)

BLOCK_3D_KINDS = ("point_cloud_3d", "trajectory_3d", "surface_3d",
                  "vector_field_3d")


class Dataset(BaseModel):
    """A typed numeric or structured payload embedded in the artifact."""
    dataset_id: str
    label: str = ""
    kind: Literal["json", "f64", "i64"] = "json"
    shape: list[int] = Field(default_factory=list)
    encoding: Literal["inline", "zlib+base64", "zlib+base64+chunked"] = "inline"
    data: Any = None                      # inline JSON value, or base64 string(s)
    byte_length: int = 0
    sha256: str = ""
    trust: str = "unknown"
    role: Literal["data", "prediction", "observation"] = "data"
    source: str = ""                  # legacy human-readable source
    source_ref: SourceRef | None = None    # structured lineage
    transformation_refs: list[str] = Field(default_factory=list)


class Series(BaseModel):
    """One displayable series bound to datasets or inline points."""
    series_id: str
    label: str = ""
    role: Literal["data", "prediction", "observation"] = "data"
    source: str = ""                  # legacy human-readable source
    source_ref: SourceRef | None = None    # structured lineage
    transformation_refs: list[str] = Field(default_factory=list)
    trust: str = "unknown"
    x: str | None = None                  # dataset ids
    y: str | None = None
    z: str | None = None
    points: list | None = None            # inline [[x, y], ...] for tiny series
    style: dict = Field(default_factory=dict)
    # For interval-certified values: clickable enclosure shown by the viewer.
    enclosure: dict | None = None         # {"lower": str, "upper": str}


class Block(BaseModel):
    """One composable panel in the artifact layout.

    ``bindings`` maps a config key (dotted path, or "*" to merge an option
    object into ``config``) to the parameter published by a ``select`` block,
    giving linked interactions without domain-specific wiring.
    """
    block_id: str
    kind: str                             # one of BLOCK_KINDS
    title: str = ""
    config: dict = Field(default_factory=dict)
    series: list[str] = Field(default_factory=list)    # series ids shown
    datasets: list[str] = Field(default_factory=list)  # dataset ids used
    bindings: dict[str, str] = Field(default_factory=dict)
    span: int = 1                         # grid column span
    trust: str = "unknown"


class Layout(BaseModel):
    kind: Literal["grid"] = "grid"
    cols: int = 2


class ProvenanceStep(BaseModel):
    step_id: str
    operation: str
    inputs: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    output: str | None = None
    engine: str | None = None
    trust: str = "unknown"
    conditions: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    steps: list[ProvenanceStep] = Field(default_factory=list)


class Annotation(BaseModel):
    """Backwards-compatible visual annotation with scientific semantics."""
    text: str
    position: list[float] | None = None
    trust: str = "unknown"
    kind: str = "note"
    claim_status: str = "none"
    author_type: str = "unknown"
    source_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class VisualizationDocument(BaseModel):
    """An evidence-carrying composition of visualization building blocks."""
    schema_version: str = "2.0"
    artifact_schema: str = ARTIFACT_SCHEMA
    title: str = "MathKernel visualization"
    layout: Layout = Field(default_factory=Layout)
    blocks: list[Block] = Field(default_factory=list)
    series: dict[str, Series] = Field(default_factory=dict)
    datasets: dict[str, Dataset] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    trust: str = "unknown"                # overall: weakest evidence in the doc
    engine: str | None = None
    annotations: list[Annotation] = Field(default_factory=list)
    transformations: list[Transformation] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)
    linked_result: dict | None = None
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    manifest: dict = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # composition helpers (the building-block API)
    # ------------------------------------------------------------------

    def add_dataset(self, dataset: Dataset) -> Dataset:
        self.datasets[dataset.dataset_id] = dataset
        return dataset

    def add_series(self, series: Series) -> Series:
        self.series[series.series_id] = series
        return series

    def add_block(self, kind: str, *, title: str = "", config: dict | None = None,
                  series: list[str] | None = None, datasets: list[str] | None = None,
                  bindings: dict[str, str] | None = None, span: int = 1,
                  block_id: str | None = None) -> Block:
        if kind not in BLOCK_KINDS:
            raise ValueError(f"unknown block kind {kind!r}; expected one of "
                             f"{', '.join(BLOCK_KINDS)}")
        if block_id is None:
            # per-document counter: identical composition order gives identical
            # ids, keeping artifacts byte-for-byte deterministic
            existing = {b.block_id for b in self.blocks}
            n = len(self.blocks) + 1
            while f"b{n}" in existing:
                n += 1
            block_id = f"b{n}"
        bid = block_id
        block = Block(block_id=bid, kind=kind, title=title,
                      config=dict(config or {}), series=list(series or []),
                      datasets=list(datasets or []),
                      bindings=dict(bindings or {}), span=max(1, int(span)))
        block.trust = self._block_trust(block)
        self.blocks.append(block)
        self.trust = self.weakest_trust()
        return block

    def _block_trust(self, block: Block) -> str:
        order = {t: i for i, t in enumerate(reversed(TRUST_LEVELS))}
        levels = [self.series[s].trust for s in block.series if s in self.series]
        levels += [self.datasets[d].trust for d in block.datasets
                   if d in self.datasets]
        if block.kind == "dag":
            levels += [p.trust for p in self.provenance.steps]
        levels = [l for l in levels if l in order]
        return min(levels, key=lambda l: order[l]) if levels else "unknown"

    def uses_3d(self) -> bool:
        return any(b.kind in BLOCK_3D_KINDS for b in self.blocks)

    def weakest_trust(self) -> str:
        """Weakest evidence among blocks, series, datasets and provenance.

        The document's own `trust` field is deliberately excluded — it is the
        output of this meet, not an input.
        """
        levels = [s.trust for s in self.series.values()]
        levels += [d.trust for d in self.datasets.values()]
        levels += [p.trust for p in self.provenance.steps]
        levels = [level for level in levels if level in TRUST_RANK]
        return min(levels, key=TRUST_RANK.__getitem__) if levels else "unknown"
