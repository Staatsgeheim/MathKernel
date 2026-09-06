# =============================================================================
# MathKernel Viz - optional matplotlib backend (PNG/PDF)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Optional matplotlib renderer. Requires the ``viz`` extra.

Renders the first SVG-capable data block of the document (plot2d, heatmap,
dag); the full multi-block layout is available through the SVG and HTML
renderers.
"""
from __future__ import annotations

import io

from ..datasets import decode_dataset
from ..document import VisualizationDocument


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for PNG/PDF export; install the 'viz' extra"
        ) from exc


def _pick_block(doc: VisualizationDocument):
    for kind in ("plot2d", "heatmap", "dag"):
        for b in doc.blocks:
            if b.kind == kind:
                return b
    return doc.blocks[0] if doc.blocks else None


def render_matplotlib(doc: VisualizationDocument, fmt: str = "png") -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    block = _pick_block(doc)
    if block is not None and block.kind == "heatmap":
        ds = doc.datasets.get(block.config.get("dataset", ""))
        if ds is not None:
            n, m = (ds.shape + [1, 1])[:2]
            vals = decode_dataset(ds)
            grid = [vals[i * m:(i + 1) * m] for i in range(n)]
            ax.imshow(grid, cmap="viridis", aspect="auto")
    elif block is not None and block.kind == "dag":
        steps = doc.provenance.steps
        depth = {}
        for s in steps:
            depth[s.step_id] = 1 + max((depth.get(p, -1) for p in s.parents), default=-1)
        layers: dict[int, list] = {}
        for s in steps:
            layers.setdefault(depth[s.step_id], []).append(s)
        pos = {}
        for d, group in sorted(layers.items()):
            for i, s in enumerate(group):
                pos[s.step_id] = (d, -(i + 1) / (len(group) + 1))
        for s in steps:
            x2, y2 = pos[s.step_id]
            for p in s.parents:
                if p in pos:
                    x1, y1 = pos[p]
                    ax.plot([x1, x2], [y1, y2], color="#9ca3af", lw=1)
            ax.annotate(s.operation[:18], (x2, y2), ha="center", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#374151"))
        ax.axis("off")
    else:
        series_ids = block.series if block is not None else list(doc.series)
        for sid in series_ids:
            s = doc.series.get(sid)
            if s is None:
                continue
            if s.x and s.y and s.x in doc.datasets:
                xs = decode_dataset(doc.datasets[s.x])
                ys = decode_dataset(doc.datasets[s.y])
            elif s.points:
                xs = [p[0] for p in s.points]
                ys = [p[1] for p in s.points]
            else:
                continue
            style = "--" if s.role == "prediction" else "-"
            ax.plot(xs, ys, style, label=f"{s.label} [{s.trust}]", ms=3)
        if series_ids:
            ax.legend(fontsize=8)
        cfg = block.config if block is not None else {}
        ax.set_xlabel(cfg.get("x_label", "x"))
        ax.set_ylabel(cfg.get("y_label", "y"))
        ax.grid(True, alpha=0.3)
    ax.set_title(f"{doc.title}  (trust: {doc.trust})", fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
