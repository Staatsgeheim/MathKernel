# =============================================================================
# MathKernel Viz - composable, domain-agnostic building blocks
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Building-block constructors.

Every function here wires the three layers together on a document:

    data  -> Dataset (typed, content-hashed, trust-carrying)
    view  -> Series (a named, styled view onto datasets)
    panel -> Block (a self-contained panel in the layout grid)

Nothing in this module knows anything about any application domain; blocks
are generic panels that can be composed into arbitrary dashboards.  The
browser viewer renders each block kind; ``select`` blocks publish parameters
that other blocks bind to via ``bindings``.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from .datasets import encode_dataset
from .document import (Block, Dataset, Layout, Series, VisualizationDocument)


def dashboard(title: str = "Dashboard", *, cols: int = 2,
              trust: str = "unknown", engine: str | None = None,
              metadata: dict | None = None) -> VisualizationDocument:
    """An empty composition canvas; add blocks with the helpers below."""
    return VisualizationDocument(title=title, layout=Layout(cols=max(1, cols)),
                                 trust=trust, engine=engine,
                                 metadata=dict(metadata or {}))


def _num(value: Any) -> float:
    """Exact-preserving coercion of numeric literals (int/Fraction/'p/q')."""
    from fractions import Fraction
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    s = str(value)
    if "/" in s:
        num, den = s.split("/", 1)
        return float(Fraction(int(num.strip()), int(den.strip())))
    return float(s)


def _flat3(points: Sequence[Sequence]) -> list[float]:
    return [_num(c) for p in points for c in list(p)[:3]]


# ----------------------------------------------------------------------
# 3D blocks
# ----------------------------------------------------------------------

def add_point_cloud(doc: VisualizationDocument,
                    points: Sequence[Sequence] | None = None, *,
                    series_id: str | None = None, label: str = "",
                    trust: str = "numeric", role: str = "data", source: str = "",
                    color: str | None = None, point_size: float | None = None,
                    opacity: float | None = None,
                    embed: dict | None = None, dataset_id: str | None = None,
                    title: str = "", span: int = 1, config: dict | None = None,
                    bindings: dict[str, str] | None = None,
                    block_id: str | None = None) -> Block:
    """Interactive 3D point cloud (orbit/pan/zoom in the browser viewer).

    Either pass explicit ``points`` ([[x, y, z], ...]) or an ``embed`` spec
    mapping a 1D dataset to delayed coordinates client-side, e.g.
    ``embed={"dataset": "seq", "lags": [0, 4, 8], "differences": False}``.
    """
    cfg = dict(config or {})
    if embed:
        cfg["embed"] = dict(embed)
    if point_size is not None:
        cfg["point_size"] = float(point_size)
    if opacity is not None:
        cfg["opacity"] = float(opacity)
    sids: list[str] = []
    dids: list[str] = []
    if points is not None:
        sid = series_id or f"s{len(doc.series) + 1}"
        did = dataset_id or f"{sid}_pts"
        flat = _flat3(points)
        doc.add_dataset(encode_dataset(did, flat, label=label or "points",
                                       trust=trust, role=role, source=source,
                                       shape=[len(flat) // 3, 3]))
        style: dict = {}
        if color:
            style["color"] = color
        doc.add_series(Series(series_id=sid, label=label, role=role,
                              source=source, trust=trust, x=did, style=style))
        sids.append(sid)
        dids.append(did)
    if embed and embed.get("dataset"):
        dids.append(embed["dataset"])
    return doc.add_block("point_cloud_3d", title=title, config=cfg,
                         series=sids, datasets=dids, bindings=bindings,
                         span=span, block_id=block_id)


def add_trajectory_3d(doc: VisualizationDocument, states: Sequence[Sequence], *,
                      label: str = "", trust: str = "numeric", role: str = "data",
                      source: str = "", color: str | None = None,
                      show_points: bool = False, title: str = "", span: int = 1,
                      config: dict | None = None,
                      block_id: str | None = None) -> Block:
    """A 3D trajectory (polyline through state space)."""
    sid = f"s{len(doc.series) + 1}"
    did = f"{sid}_traj"
    flat = _flat3(states)
    doc.add_dataset(encode_dataset(did, flat, label=label or "trajectory",
                                   trust=trust, role=role, source=source,
                                   shape=[len(flat) // 3, 3]))
    style = {"color": color} if color else {}
    doc.add_series(Series(series_id=sid, label=label, role=role, source=source,
                          trust=trust, x=did, style=style))
    cfg = dict(config or {})
    cfg["show_points"] = bool(show_points)
    if block_id is not None:
        for b in doc.blocks:
            if b.block_id == block_id:
                b.series.append(sid)
                if did not in b.datasets:
                    b.datasets.append(did)
                b.trust = doc._block_trust(b)
                doc.trust = doc.weakest_trust()
                return b
    return doc.add_block("trajectory_3d", title=title, config=cfg,
                         series=[sid], datasets=[did], span=span,
                         block_id=block_id)


def add_surface_3d(doc: VisualizationDocument, grid: Sequence[Sequence], *,
                   x_range: tuple[float, float] | None = None,
                   y_range: tuple[float, float] | None = None,
                   label: str = "", trust: str = "numeric", role: str = "data",
                   source: str = "", title: str = "", span: int = 1,
                   config: dict | None = None,
                   block_id: str | None = None) -> Block:
    """A 3D surface from a 2D grid of heights."""
    rows = [[_num(v) for v in row] for row in grid]
    n, m = len(rows), (len(rows[0]) if rows else 0)
    did = f"surf{len(doc.datasets) + 1}"
    doc.add_dataset(encode_dataset(did, [v for row in rows for v in row],
                                   label=label or "surface", trust=trust,
                                   role=role, source=source, shape=[n, m]))
    cfg = dict(config or {})
    cfg.update({"rows": n, "cols": m,
                "x_range": list(x_range) if x_range else None,
                "y_range": list(y_range) if y_range else None})
    return doc.add_block("surface_3d", title=title, config=cfg,
                         datasets=[did], span=span, block_id=block_id)


def add_vector_field_3d(doc: VisualizationDocument, origins: Sequence[Sequence],
                        vectors: Sequence[Sequence], *, label: str = "",
                        trust: str = "numeric", role: str = "data",
                        source: str = "", title: str = "", span: int = 1,
                        config: dict | None = None,
                        block_id: str | None = None) -> Block:
    """A 3D vector field (arrows at origins)."""
    o, v = _flat3(origins), _flat3(vectors)
    base = len(doc.datasets) + 1
    do, dv = f"vf{base}_o", f"vf{base}_v"
    doc.add_dataset(encode_dataset(do, o, label=label or "origins",
                                   trust=trust, role=role, source=source,
                                   shape=[len(o) // 3, 3]))
    doc.add_dataset(encode_dataset(dv, v, label=label or "vectors",
                                   trust=trust, role=role, source=source,
                                   shape=[len(v) // 3, 3]))
    return doc.add_block("vector_field_3d", title=title,
                         config=dict(config or {}), datasets=[do, dv],
                         span=span, block_id=block_id)


# ----------------------------------------------------------------------
# 2D blocks
# ----------------------------------------------------------------------

def add_plot(doc: VisualizationDocument, xs: Sequence, ys: Sequence, *,
             label: str = "", x_label: str = "x", y_label: str = "y",
             trust: str = "numeric", role: str = "data", source: str = "",
             marker: str | None = None, color: str | None = None,
             size: float | None = None, stack: str | None = None,
             log_y: bool = False, log_x: bool = False,
             enclosure: dict | None = None,
             title: str = "", span: int = 1, config: dict | None = None,
             bindings: dict[str, str] | None = None,
             block_id: str | None = None) -> Block:
    """A 2D line/scatter/bar plot.  Call repeatedly with the same ``block_id``
    to overlay multiple series on one panel."""
    cfg = dict(config or {})
    cfg.setdefault("x_label", x_label)
    cfg.setdefault("y_label", y_label)
    if log_y:
        cfg["log_y"] = True
    if log_x:
        cfg["log_x"] = True
    sid = f"s{len(doc.series) + 1}"
    dx, dy = f"{sid}_x", f"{sid}_y"
    doc.add_dataset(encode_dataset(dx, [_num(v) for v in xs], label=x_label,
                                   trust=trust, role=role, source=source))
    doc.add_dataset(encode_dataset(dy, [_num(v) for v in ys], label=y_label,
                                   trust=trust, role=role, source=source))
    style: dict = {}
    if marker:
        style["marker"] = marker
    if color:
        style["color"] = color
    if size is not None:
        style["size"] = float(size)
    if stack:
        style["stack"] = stack
    doc.add_series(Series(series_id=sid, label=label or y_label, role=role,
                          source=source, trust=trust, x=dx, y=dy, style=style,
                          enclosure=enclosure))
    if block_id is not None:
        for b in doc.blocks:
            if b.block_id == block_id:
                b.series.append(sid)
                if dx not in b.datasets:
                    b.datasets.extend([dx, dy])
                b.trust = doc._block_trust(b)
                doc.trust = doc.weakest_trust()
                return b
        # named panel does not exist yet: create it under that name
    return doc.add_block("plot2d", title=title, config=cfg, series=[sid],
                         datasets=[dx, dy], bindings=bindings, span=span,
                         block_id=block_id)


def add_histogram(doc: VisualizationDocument, values: Sequence | str, *,
                  bins: int = 64, value_range: tuple[float, float] | None = None,
                  label: str = "", trust: str = "numeric", role: str = "data",
                  source: str = "", log_y: bool = False, color: str | None = None,
                  title: str = "", span: int = 1, config: dict | None = None,
                  bindings: dict[str, str] | None = None,
                  block_id: str | None = None) -> Block:
    """A histogram; the viewer bins the raw dataset client-side so ``bins``
    stays interactive (and can be driven by a bound ``select`` control)."""
    if isinstance(values, str):
        did = values
        if did not in doc.datasets:
            raise ValueError(f"unknown dataset {did!r}")
    else:
        did = f"hist{len(doc.datasets) + 1}"
        doc.add_dataset(encode_dataset(did, [_num(v) for v in values],
                                       label=label or "values", trust=trust,
                                       role=role, source=source))
    cfg = dict(config or {})
    cfg.update({"dataset": did, "bins": int(bins)})
    if value_range:
        cfg["range"] = [float(value_range[0]), float(value_range[1])]
    if log_y:
        cfg["log_y"] = True
    if color:
        cfg["color"] = color
    return doc.add_block("histogram", title=title, config=cfg, datasets=[did],
                         bindings=bindings, span=span, block_id=block_id)


def add_heatmap(doc: VisualizationDocument, matrix: Sequence[Sequence], *,
                label: str = "", trust: str = "exact", role: str = "data",
                source: str = "", title: str = "", span: int = 1,
                config: dict | None = None,
                block_id: str | None = None) -> Block:
    """A matrix heatmap. A flat sequence is rendered as a single row."""
    matrix = list(matrix)
    if matrix and not isinstance(matrix[0], (list, tuple)):
        matrix = [matrix]
    rows = [[_num(v) for v in row] for row in matrix]
    n, m = len(rows), (len(rows[0]) if rows else 0)
    did = f"hm{len(doc.datasets) + 1}"
    doc.add_dataset(encode_dataset(did, [v for row in rows for v in row],
                                   label=label or "matrix", trust=trust,
                                   role=role, source=source, shape=[n, m]))
    cfg = dict(config or {})
    cfg.update({"dataset": did, "rows": n, "cols": m})
    return doc.add_block("heatmap", title=title, config=cfg, datasets=[did],
                         span=span, block_id=block_id)


def add_dag(doc: VisualizationDocument, *,
            nodes: list[dict] | None = None, title: str = "",
            span: int = 1, config: dict | None = None,
            block_id: str | None = None) -> Block:
    """A provenance/dependency DAG.

    With ``nodes=None`` the block renders the document's provenance graph;
    pass explicit ``nodes`` ([{"id", "label", "parents", "trust"}]) for any
    other directed graph.
    """
    cfg = dict(config or {})
    if nodes is not None:
        cfg["nodes"] = [dict(n) for n in nodes]
    return doc.add_block("dag", title=title, config=cfg, span=span,
                         block_id=block_id)


# ----------------------------------------------------------------------
# content blocks
# ----------------------------------------------------------------------

def add_metric_grid(doc: VisualizationDocument,
                    entries: list[dict] | None = None, *,
                    dataset: str | None = None,
                    stats: Sequence[str] | None = None,
                    title: str = "", span: int = 1, config: dict | None = None,
                    bindings: dict[str, str] | None = None,
                    block_id: str | None = None) -> Block:
    """A grid of metric cards.

    Either explicit ``entries`` ([{"label", "value", "hint"?}]) or a
    ``dataset`` + ``stats`` list ("count", "mean", "std", "min", "max",
    "distinct") computed by the viewer from the raw data.
    """
    cfg = dict(config or {})
    dids: list[str] = []
    if entries is not None:
        cfg["entries"] = [dict(e) for e in entries]
    if dataset is not None:
        if dataset not in doc.datasets:
            raise ValueError(f"unknown dataset {dataset!r}")
        cfg["dataset"] = dataset
        cfg["stats"] = list(stats or ["count", "mean", "std", "min", "max"])
        dids.append(dataset)
    return doc.add_block("metric_grid", title=title, config=cfg,
                         datasets=dids, bindings=bindings, span=span,
                         block_id=block_id)


def add_data_table(doc: VisualizationDocument,
                   columns: list[dict] | None = None, *,
                   rows: list[list] | None = None,
                   max_rows: int = 100, title: str = "", span: int = 1,
                   config: dict | None = None,
                   bindings: dict[str, str] | None = None,
                   block_id: str | None = None) -> Block:
    """A scrollable data table.

    ``columns``: [{"label", "dataset"}] referencing 1D datasets, or
    [{"label", "values": [...]}] inline.  Alternatively pass raw ``rows``.
    """
    cfg = dict(config or {})
    dids: list[str] = []
    if columns is not None:
        cols = []
        for c in columns:
            c = dict(c)
            if "dataset" in c:
                if c["dataset"] not in doc.datasets:
                    raise ValueError(f"unknown dataset {c['dataset']!r}")
                dids.append(c["dataset"])
            cols.append(c)
        cfg["columns"] = cols
    if rows is not None:
        cfg["rows"] = [[str(v) for v in row] for row in rows]
    cfg["max_rows"] = int(max_rows)
    return doc.add_block("data_table", title=title, config=cfg, datasets=dids,
                         bindings=bindings, span=span, block_id=block_id)


def add_text(doc: VisualizationDocument, text: str, *, title: str = "",
             span: int = 1, config: dict | None = None,
             block_id: str | None = None) -> Block:
    """A plain-text/markdown-lite note block (rendered escaped, never HTML)."""
    cfg = dict(config or {})
    cfg["text"] = str(text)
    return doc.add_block("text", title=title, config=cfg, span=span,
                         block_id=block_id)


# ----------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------

def add_select(doc: VisualizationDocument, param: str, options: list[dict], *,
               label: str = "", default: int = 0, title: str = "",
               span: int = 1, block_id: str | None = None) -> Block:
    """A dropdown control publishing ``param``.

    ``options``: [{"label": str, "value": any}].  Other blocks bind with
    ``bindings={"*": param}`` (merge an object value into their config) or
    ``bindings={"config.key": param}`` (set one key).  Changing the control
    re-renders every bound block — the generic linked-dashboard mechanism.
    """
    if not options:
        raise ValueError("select control requires at least one option")
    cfg = {"param": str(param), "label": label or str(param),
           "default": int(default),
           "options": [{"label": str(o.get("label", i)),
                        "value": o.get("value")} for i, o in enumerate(options)]}
    return doc.add_block("select", title=title, config=cfg, span=span,
                         block_id=block_id)
