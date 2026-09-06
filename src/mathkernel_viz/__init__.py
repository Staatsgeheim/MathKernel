# =============================================================================
# MathKernel Viz - evidence-carrying interactive artifacts
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""mathkernel_viz: visualization and portable HTML artifacts for MathKernel.

Artifacts are composed from small, domain-agnostic building blocks (point
clouds, plots, histograms, heatmaps, DAGs, metric grids, tables, text and
select controls) arranged on a grid layout.  The artifact layer consumes
already-produced MathResults and packages them with their data, provenance,
trust and integrity hashes.  It never recomputes mathematics; the browser
artifact is a viewer, not an execution environment.

Compose a dashboard from building blocks:

    import mathkernel_viz as viz

    doc = viz.dashboard("My result", cols=2)
    viz.add_point_cloud(doc, points, trust="numeric")
    viz.add_histogram(doc, values, bins=128)
    viz.add_select(doc, "k", [{"label": "k=4", "value": {"lags": [0, 4, 8]}}])
    viz.export_html(doc, "result.html")
"""
from __future__ import annotations

from pathlib import Path

from . import blocks, transforms
from .adapters import (compare_document, dag_document, from_mathresult,
                       heatmap_document, interval_plot_document, plot_document,
                       points3d_document, surface_document,
                       trajectory3d_document, vector_field3d_document, from_projection)
from .auto import select_renderer, select_view
from .artifact import attach_result_lineage, to_artifact
from mathkernel_artifacts import SourceRef, EvidenceItem, Transformation, MathKernelArtifact
from .blocks import (add_data_table, add_dag, add_heatmap, add_histogram,
                     add_metric_grid, add_plot, add_point_cloud, add_select,
                     add_surface_3d, add_text, add_trajectory_3d,
                     add_vector_field_3d, dashboard)
from .datasets import decode_dataset, encode_dataset, verify_dataset
from .document import (ARTIFACT_SCHEMA, BLOCK_KINDS, Annotation, Block,
                       Dataset, Layout, Provenance, Series,
                       VisualizationDocument)
from .manifest import VIZ_VERSION, build_manifest
from .renderers.html import export_html, render_html
from .renderers.svg import render_svg

__all__ = [
    "ARTIFACT_SCHEMA", "VIZ_VERSION", "BLOCK_KINDS",
    "VisualizationDocument", "Layout", "Block", "Series", "Dataset",
    "Provenance", "Annotation",
    "blocks", "transforms",
    "dashboard", "add_point_cloud", "add_trajectory_3d", "add_surface_3d",
    "add_vector_field_3d", "add_plot", "add_histogram", "add_heatmap",
    "add_dag", "add_metric_grid", "add_data_table", "add_text", "add_select",
    "visualize", "export_html", "render_html", "export_svg", "export_png",
    "render_svg", "from_mathresult", "plot_document", "dag_document",
    "heatmap_document", "interval_plot_document", "points3d_document",
    "trajectory3d_document", "surface_document", "vector_field3d_document",
    "compare_document", "from_projection", "encode_dataset", "decode_dataset", "verify_dataset",
    "build_manifest", "select_view", "select_renderer", "to_artifact",
    "SourceRef", "EvidenceItem", "Transformation", "MathKernelArtifact", "attach_result_lineage",
]


def visualize(result, *, view: str = "auto", renderer: str = "auto",
              title: str | None = None) -> VisualizationDocument:
    """Build a VisualizationDocument from a MathResult (or result dict)."""
    if view == "auto":
        view = select_view(result)
    doc = from_mathresult(result, view=view, title=title)
    attach_result_lineage(doc, result)
    doc.metadata["view"] = view
    doc.metadata["renderer"] = (select_renderer(view) if renderer == "auto"
                                else renderer)
    return doc


def export_svg(doc: VisualizationDocument, path: str | Path) -> dict:
    from .manifest import sha256_text
    svg = render_svg(doc)
    p = Path(path)
    p.write_text(svg, encoding="utf-8", newline="\n")
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_text(svg)}


def export_png(doc: VisualizationDocument, path: str | Path) -> dict:
    from .manifest import sha256_text
    from .renderers.matplotlib import render_matplotlib
    data = render_matplotlib(doc, "png")
    p = Path(path)
    p.write_bytes(data)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": sha256_text(data.decode("latin1"))}
