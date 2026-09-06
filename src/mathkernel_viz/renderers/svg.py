# =============================================================================
# MathKernel Viz - dependency-free SVG renderer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Pure-Python SVG renderer for VisualizationDocuments.

Renders each SVG-capable block (plot2d, histogram, heatmap, dag, text,
metric_grid) into a grid matching the document layout; 3D blocks get a
placeholder pointing at the interactive HTML artifact.

Deterministic: identical documents produce byte-identical SVG.  All text is
HTML-escaped; MathIR is display data, never markup.
"""
from __future__ import annotations

import html
import math

from ..datasets import decode_dataset
from ..document import Block, VisualizationDocument

_W, _H = 900, 560
_PAD = 70

SVG_KINDS = ("plot2d", "histogram", "heatmap", "dag", "text", "metric_grid")

_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
            "#0891b2", "#be185d", "#65a30d"]

_TRUST_COLOR = {
    "formal": "#7c3aed", "exact": "#16a34a", "symbolic": "#2563eb",
    "interval_certified": "#0891b2", "numeric_high_precision": "#ca8a04",
    "numeric": "#ea580c", "empirical": "#dc2626", "heuristic": "#be185d",
    "unknown": "#6b7280",
}


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 1e6 or av < 1e-4:
        return f"{v:.3e}"
    return f"{v:.6g}"


def _nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return [lo]
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 5, 10):
        if step * m * n >= span:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-12:
        if lo - 1e-12 <= t:
            ticks.append(round(t, 12))
        t += step
    return ticks


def _block_series(doc: VisualizationDocument, block: Block):
    for sid in block.series:
        s = doc.series.get(sid)
        if s is None:
            continue
        if s.x and s.y and s.x in doc.datasets and s.y in doc.datasets:
            yield s, decode_dataset(doc.datasets[s.x]), decode_dataset(doc.datasets[s.y])
        elif s.points:
            yield s, [p[0] for p in s.points], [p[1] for p in s.points]


def _render_plot2d(doc: VisualizationDocument, block: Block) -> str:
    cfg = block.config
    series = list(_block_series(doc, block))
    log_y = bool(cfg.get("log_y"))
    if log_y:
        series = [(s, xs, [math.log10(max(v, 1e-300)) for v in ys])
                  for s, xs, ys in series]
    all_x = [v for _, xs, _ in series for v in xs]
    all_y = [v for _, _, ys in series for v in ys]
    if not all_x or not all_y:
        return '<text x="450" y="280" text-anchor="middle" class="mk-empty">no plottable series</text>'
    x0, x1 = min(all_x), max(all_x)
    y0, y1 = min(all_y), max(all_y)
    if x0 == x1:
        x0, x1 = x0 - 1, x1 + 1
    if y0 == y1:
        y0, y1 = y0 - 1, y1 + 1
    has_bar = any((s.style or {}).get("marker") == "bar" for s, _, _ in series)
    mx = (x1 - x0) * 0.05
    my = 0.0 if has_bar else (y1 - y0) * 0.08
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
    if has_bar:
        y0 = min(y0, 0.0)

    def X(x):
        return _PAD + (x - x0) / (x1 - x0) * (_W - 2 * _PAD)

    def Y(y):
        return _H - _PAD - (y - y0) / (y1 - y0) * (_H - 2 * _PAD)

    parts = []
    for t in _nice_ticks(x0, x1):
        parts.append(f'<line x1="{X(t):.2f}" y1="{_PAD}" x2="{X(t):.2f}" '
                     f'y2="{_H - _PAD}" class="mk-grid"/>'
                     f'<text x="{X(t):.2f}" y="{_H - _PAD + 18}" text-anchor="middle" '
                     f'class="mk-tick">{_esc(_fmt(t))}</text>')
    for t in _nice_ticks(y0, y1):
        label = f"1e{_fmt(t)}" if log_y else _fmt(t)
        parts.append(f'<line x1="{_PAD}" y1="{Y(t):.2f}" x2="{_W - _PAD}" '
                     f'y2="{Y(t):.2f}" class="mk-grid"/>'
                     f'<text x="{_PAD - 8}" y="{Y(t) + 4:.2f}" text-anchor="end" '
                     f'class="mk-tick">{_esc(label)}</text>')
    parts.append(f'<rect x="{_PAD}" y="{_PAD}" width="{_W - 2 * _PAD}" '
                 f'height="{_H - 2 * _PAD}" class="mk-frame"/>')

    stack_offsets: dict = {}
    max_n = max((len(xs) for _, xs, _ in series), default=1)
    for i, (s, xs, ys) in enumerate(series):
        st = s.style or {}
        color = st.get("color") or _TRUST_COLOR.get(s.trust, _PALETTE[i % len(_PALETTE)])
        dash = ' stroke-dasharray="6,4"' if s.role == "prediction" else ""
        marker = st.get("marker") or ("line" if len(xs) > 64 else "points")
        if marker == "bar":
            bw = max(1.0, (_W - 2 * _PAD) / max(max_n, 1) * 0.7)
            stack = st.get("stack") if not log_y else None
            for x, y in zip(xs, ys):
                base = stack_offsets.get((stack, x), 0.0) if stack else 0.0
                if stack:
                    stack_offsets[(stack, x)] = base + y
                parts.append(f'<rect x="{X(x) - bw / 2:.2f}" y="{Y(base + y):.2f}" '
                             f'width="{bw:.2f}" height="{Y(base) - Y(base + y):.2f}" '
                             f'fill="{color}" fill-opacity="0.85">'
                             f'<title>{_esc(s.label)}: {_esc(_fmt(x))} = '
                             f'{_esc(_fmt(y))}</title></rect>')
            continue
        if marker == "points":
            r = st.get("size", 1.4 if len(xs) > 2000 else 3.5)
            for x, y in zip(xs, ys):
                parts.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="{r}" '
                             f'fill="{color}"><title>{_esc(s.label)} '
                             f'(trust: {_esc(s.trust)})</title></circle>')
        else:
            pts = " ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in zip(xs, ys))
            parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                         f'stroke-width="1.8"{dash}>'
                         f'<title>{_esc(s.label)} (trust: {_esc(s.trust)})</title></polyline>')
        if len(series) > 1:
            parts.append(f'<rect x="{_PAD + 8 + i * 200}" y="{_PAD - 36}" width="11" '
                         f'height="11" fill="{color}"/>'
                         f'<text x="{_PAD + 24 + i * 200}" y="{_PAD - 26}" '
                         f'class="mk-tick">{_esc((s.label or s.series_id)[:24])} '
                         f'({len(xs)})</text>')
        if s.enclosure:
            parts.append(f'<text x="{_W - _PAD + 6}" y="{_PAD + 16 * i}" '
                         f'class="mk-tick">[{_esc(s.enclosure["lower"][:18])}, '
                         f'{_esc(s.enclosure["upper"][:18])}]</text>')

    parts.append(f'<text x="{_W / 2}" y="{_H - 18}" text-anchor="middle" '
                 f'class="mk-axis">{_esc(cfg.get("x_label", "x"))}</text>')
    parts.append(f'<text x="18" y="{_H / 2}" text-anchor="middle" '
                 f'transform="rotate(-90 18 {_H / 2})" class="mk-axis">'
                 f'{_esc(cfg.get("y_label", "y"))}</text>')
    return "".join(parts)


def _render_histogram(doc: VisualizationDocument, block: Block) -> str:
    cfg = block.config
    ds = doc.datasets.get(cfg.get("dataset", ""))
    if ds is None:
        return '<text x="450" y="280" text-anchor="middle" class="mk-empty">dataset unavailable</text>'
    vals = decode_dataset(ds)
    if not vals:
        return '<text x="450" y="280" text-anchor="middle" class="mk-empty">empty dataset</text>'
    bins = max(1, min(2048, int(cfg.get("bins", 64))))
    lo, hi = (cfg.get("range") or (min(vals), max(vals)))
    if lo == hi:
        hi = lo + 1
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        if lo <= v <= hi:
            counts[min(bins - 1, int((v - lo) / width))] += 1
    max_c = max(counts) or 1
    log_y = bool(cfg.get("log_y"))
    top = math.log10(max_c) if log_y else max_c
    color = cfg.get("color", "#2563eb")
    bw = (_W - 2 * _PAD) / bins
    parts = []
    for b, c in enumerate(counts):
        cv = math.log10(c) if (log_y and c) else (0 if log_y else c)
        hgt = (cv / top) * (_H - 2 * _PAD) if top else 0
        parts.append(f'<rect x="{_PAD + b * bw:.2f}" y="{_H - _PAD - hgt:.2f}" '
                     f'width="{max(1.0, bw - 0.5):.2f}" height="{hgt:.2f}" '
                     f'fill="{color}" fill-opacity="0.9">'
                     f'<title>[{_esc(_fmt(lo + b * width))}, '
                     f'{_esc(_fmt(lo + (b + 1) * width))}): {c}</title></rect>')
    parts.append(f'<rect x="{_PAD}" y="{_PAD}" width="{_W - 2 * _PAD}" '
                 f'height="{_H - 2 * _PAD}" class="mk-frame"/>')
    parts.append(f'<text x="{_PAD}" y="{_H - _PAD + 18}" class="mk-tick">'
                 f'{_esc(_fmt(lo))}</text>')
    parts.append(f'<text x="{_W - _PAD}" y="{_H - _PAD + 18}" text-anchor="end" '
                 f'class="mk-tick">{_esc(_fmt(hi))}</text>')
    return "".join(parts)


def _render_heatmap(doc: VisualizationDocument, block: Block) -> str:
    cfg = block.config
    ds = doc.datasets.get(cfg.get("dataset", ""))
    if ds is None:
        return '<text x="450" y="280" text-anchor="middle" class="mk-empty">no matrix</text>'
    vals = decode_dataset(ds)
    n, m = int(cfg.get("rows", 1)), int(cfg.get("cols", 1))
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = hi - lo or 1.0
    cw = (_W - 2 * _PAD) / max(m, 1)
    ch = (_H - 2 * _PAD) / max(n, 1)
    parts = []
    for i in range(n):
        for j in range(m):
            v = vals[i * m + j]
            t = (v - lo) / span
            r = int(37 + t * (220 - 37))
            b = int(99 + t * (235 - 99))
            g = int(235 - t * (235 - 180))
            parts.append(f'<rect x="{_PAD + j * cw:.2f}" y="{_PAD + i * ch:.2f}" '
                         f'width="{cw + 0.5:.2f}" height="{ch + 0.5:.2f}" '
                         f'fill="rgb({r},{g},{b})"><title>[{i},{j}] = '
                         f'{_esc(_fmt(v))}</title></rect>')
    parts.append(f'<text x="{_W / 2}" y="{_H - 18}" text-anchor="middle" class="mk-axis">'
                 f'{_esc(n)} x {_esc(m)} — min {_esc(_fmt(lo))}, max {_esc(_fmt(hi))}</text>')
    return "".join(parts)


def _render_dag(doc: VisualizationDocument, block: Block) -> str:
    nodes = block.config.get("nodes")
    if nodes is None:
        steps = doc.provenance.steps
        nodes = [{"id": s.step_id, "label": s.operation, "parents": s.parents,
                  "trust": s.trust, "engine": s.engine} for s in steps]
    if not nodes:
        return '<text x="450" y="280" text-anchor="middle" class="mk-empty">no graph</text>'
    depth: dict[str, int] = {}
    for s in nodes:
        depth[s["id"]] = 1 + max((depth.get(p, -1) for p in s.get("parents", [])),
                                 default=-1)
    layers: dict[int, list] = {}
    for s in nodes:
        layers.setdefault(depth[s["id"]], []).append(s)
    max_layer = max(layers)
    pos: dict[str, tuple[float, float]] = {}
    for d, group in sorted(layers.items()):
        for i, s in enumerate(group):
            x = _PAD + (max_layer and d / max_layer) * (_W - 2 * _PAD - 120)
            y = _PAD + (i + 1) / (len(group) + 1) * (_H - 2 * _PAD)
            pos[s["id"]] = (x, y)
    parts = []
    for s in nodes:
        x2, y2 = pos[s["id"]]
        for p in s.get("parents", []):
            if p in pos:
                x1, y1 = pos[p]
                parts.append(f'<line x1="{x1 + 110:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
                             f'y2="{y2:.2f}" class="mk-edge"/>')
    for s in nodes:
        x, y = pos[s["id"]]
        color = _TRUST_COLOR.get(s.get("trust", "unknown"), "#6b7280")
        label = str(s.get("label", s["id"]))[:22]
        parts.append(f'<rect x="{x:.2f}" y="{y - 16:.2f}" width="110" height="32" rx="6" '
                     f'fill="none" stroke="{color}" stroke-width="2"/>'
                     f'<text x="{x + 55:.2f}" y="{y + 4:.2f}" text-anchor="middle" '
                     f'class="mk-node">{_esc(label)}</text>'
                     f'<title>{_esc(s["id"])} | {_esc(label)} | '
                     f'trust={_esc(s.get("trust", "unknown"))}</title>')
    return "".join(parts)


def _render_text(doc: VisualizationDocument, block: Block) -> str:
    lines = str(block.config.get("text", "")).split("\n")[:18]
    parts = []
    for i, line in enumerate(lines):
        parts.append(f'<text x="{_PAD}" y="{_PAD + 24 + i * 22}" class="mk-node">'
                     f'{_esc(line[:110])}</text>')
    return "".join(parts)


def _render_metric_grid(doc: VisualizationDocument, block: Block) -> str:
    entries = block.config.get("entries") or []
    if block.config.get("dataset"):
        from ..transforms import sequence_stats
        ds = doc.datasets.get(block.config["dataset"])
        if ds is not None:
            stats = sequence_stats(decode_dataset(ds))
            for k in block.config.get("stats", ["count", "mean", "std", "min", "max"]):
                if k in stats:
                    entries = entries + [{"label": k, "value": _fmt(stats[k])
                                          if isinstance(stats[k], float) else stats[k]}]
    parts = []
    for i, e in enumerate(entries[:12]):
        x = _PAD + (i % 3) * 250
        y = _PAD + (i // 3) * 70
        parts.append(f'<rect x="{x}" y="{y}" width="230" height="56" rx="6" '
                     f'fill="none" stroke="#d1d5db"/>'
                     f'<text x="{x + 12}" y="{y + 22}" class="mk-tick">'
                     f'{_esc(e.get("label", ""))}</text>'
                     f'<text x="{x + 12}" y="{y + 44}" class="mk-node">'
                     f'{_esc(str(e.get("value", ""))[:26])}</text>')
    if not entries:
        parts.append('<text x="450" y="280" text-anchor="middle" class="mk-empty">'
                     'no metrics</text>')
    return "".join(parts)


_BLOCK_RENDERERS = {
    "plot2d": _render_plot2d,
    "histogram": _render_histogram,
    "heatmap": _render_heatmap,
    "dag": _render_dag,
    "text": _render_text,
    "metric_grid": _render_metric_grid,
}


def _frame(doc: VisualizationDocument, body: str) -> str:
    trust_color = _TRUST_COLOR.get(doc.trust, "#6b7280")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" font-family="ui-monospace, monospace">'
        f'<style>'
        f'.mk-grid{{stroke:#e5e7eb;stroke-width:1}}'
        f'.mk-frame{{fill:none;stroke:#9ca3af;stroke-width:1.2}}'
        f'.mk-tick{{font-size:11px;fill:#6b7280}}'
        f'.mk-axis{{font-size:13px;fill:#374151}}'
        f'.mk-title{{font-size:16px;fill:#111827}}'
        f'.mk-trust{{font-size:12px;font-weight:bold}}'
        f'.mk-edge{{stroke:#9ca3af;stroke-width:1.2}}'
        f'.mk-node{{font-size:11px;fill:#111827}}'
        f'.mk-empty{{font-size:14px;fill:#6b7280}}'
        f'</style>'
        f'<rect width="{_W}" height="{_H}" fill="white"/>'
        f'<text x="{_PAD}" y="34" class="mk-title">{_esc(doc.title)}</text>'
        f'<text x="{_W - _PAD}" y="34" text-anchor="end" class="mk-trust" '
        f'fill="{trust_color}">trust: {_esc(doc.trust)}</text>'
        f'{body}</svg>')


def render_svg(doc: VisualizationDocument) -> str:
    """Render the document's blocks into one static SVG grid."""
    if not doc.blocks:
        return _frame(doc, '<text x="450" y="280" text-anchor="middle" '
                           'class="mk-empty">empty document</text>')
    if len(doc.blocks) == 1 and doc.blocks[0].kind in _BLOCK_RENDERERS:
        return _frame(doc, _BLOCK_RENDERERS[doc.blocks[0].kind](doc, doc.blocks[0]))

    cols = max(1, doc.layout.cols)
    total_span = sum(min(b.span, cols) for b in doc.blocks)
    rows = (total_span + cols - 1) // cols
    pw, ph = _W / cols, _H / max(rows, 1)
    parts = []
    cell = 0
    for b in doc.blocks:
        span = min(b.span, cols)
        ox, oy = (cell % cols) * pw, (cell // cols) * ph
        cell += span
        renderer = _BLOCK_RENDERERS.get(b.kind)
        if renderer is not None:
            body = renderer(doc, b)
        else:
            body = ('<text x="450" y="280" text-anchor="middle" class="mk-empty">'
                    + _esc(b.kind) + ' — open the HTML artifact for the '
                    'interactive view</text>')
        title = (f'<text x="{_PAD}" y="44" class="mk-axis">{_esc(b.title)}</text>'
                 if b.title else "")
        parts.append(f'<svg x="{ox:.0f}" y="{oy:.0f}" width="{pw * span:.0f}" '
                     f'height="{ph:.0f}" viewBox="0 0 {_W} {_H}">{title}{body}</svg>'
                     f'<rect x="{ox:.0f}" y="{oy:.0f}" width="{pw * span:.0f}" '
                     f'height="{ph:.0f}" fill="none" stroke="#e5e7eb"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
            f'viewBox="0 0 {_W} {_H}" font-family="ui-monospace, monospace">'
            f'<rect width="{_W}" height="{_H}" fill="white"/>'
            + "".join(parts) + "</svg>")
