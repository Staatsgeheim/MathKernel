# =============================================================================
# MathKernel Viz - MathKernel -> VisualizationDocument adapters
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Adapters turn already-produced MathKernel results into the visualization IR.

They never recompute mathematics; they repackage result structures, carrying
trust, engine, assumptions and provenance into the document.  Every adapter
is a thin composition of the generic building blocks in ``blocks.py`` — the
same primitives available to users and MCP clients.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import blocks as B
from .blocks import _num
from .document import (Annotation, Provenance, ProvenanceStep,
                       VisualizationDocument)


def _trust(value: Any) -> str:
    v = getattr(value, "value", value)
    return str(v) if v else "unknown"


def provenance_from_steps(steps: Iterable[Any]) -> Provenance:
    out = []
    for s in steps:
        d = s if isinstance(s, dict) else s.model_dump(mode="json")
        out.append(ProvenanceStep(
            step_id=d.get("step_id", ""),
            operation=d.get("operation", ""),
            inputs=list(d.get("inputs", [])),
            parents=list(d.get("parents", [])),
            output=d.get("output"),
            engine=d.get("engine"),
            trust=_trust(d.get("trust", "unknown")),
            conditions=list(d.get("conditions", []))))
    return Provenance(steps=out)


def _attach_result(doc: VisualizationDocument, result: Any) -> VisualizationDocument:
    d = result if isinstance(result, dict) else result.model_dump(mode="json")
    doc.assumptions = list(d.get("assumptions_used", []))
    doc.provenance = provenance_from_steps(d.get("derivation", []))
    doc.metadata["status"] = d.get("status", "ok")
    if d.get("warnings"):
        doc.metadata["warnings"] = list(d["warnings"])
    doc.trust = doc.weakest_trust()
    return doc


def dag_document(steps: Iterable[Any], *, title: str = "Derivation DAG",
                 trust: str = "unknown", engine: str | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, trust=trust, engine=engine)
    doc.provenance = provenance_from_steps(steps)
    B.add_dag(doc)
    doc.trust = doc.weakest_trust()
    return doc


def plot_document(xs: Sequence, ys: Sequence, *, title: str = "Plot",
                  x_label: str = "x", y_label: str = "y",
                  trust: str = "numeric", role: str = "data",
                  source: str = "", engine: str | None = None,
                  series_label: str = "") -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_plot(doc, xs, ys, label=series_label or y_label, x_label=x_label,
               y_label=y_label, trust=trust, role=role, source=source)
    return doc


def _interval_bounds(p: dict) -> tuple[Any, Any]:
    """Accept both catalog-style lo/hi and lower/upper interval keys."""
    lo = p.get("lower", p.get("lo"))
    hi = p.get("upper", p.get("hi"))
    if lo is None or hi is None:
        raise ValueError(
            "interval entries require 'lower'/'upper' (or 'lo'/'hi') keys; "
            f"got {sorted(p)}")
    return lo, hi


def interval_plot_document(points: Sequence[dict], *, title: str = "Certified enclosures",
                           engine: str | None = None,
                           trust: str = "interval_certified") -> VisualizationDocument:
    """points: [{"x": ..., "lower"/"lo": str, "upper"/"hi": str, "label": str}]."""
    doc = B.dashboard(title, cols=1, engine=engine)
    block_id = "intervals"
    for i, p in enumerate(points):
        lo_raw, hi_raw = _interval_bounds(p)
        lo, hi = _num(lo_raw), _num(hi_raw)
        B.add_plot(doc, [float(p.get("x", i))], [(lo + hi) / 2],
                   label=p.get("label", f"root {i}"), x_label="index",
                   y_label="value", trust=trust, marker="points",
                   enclosure={"lower": str(lo_raw), "upper": str(hi_raw)},
                   block_id=block_id)
    return doc


def heatmap_document(matrix: Sequence[Sequence], *, title: str = "Matrix",
                     trust: str = "exact", engine: str | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_heatmap(doc, matrix, label=title, trust=trust)
    return doc


def points3d_document(points: Sequence[Sequence], *, title: str = "Point cloud",
                      trust: str = "numeric", role: str = "data",
                      source: str = "", engine: str | None = None,
                      labels: Sequence[str] | None = None,
                      meta: Sequence[dict] | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_point_cloud(doc, points, label=title, trust=trust, role=role,
                      source=source)
    return doc


def trajectory3d_document(trajectory: Sequence[Sequence], *, title: str = "Trajectory",
                          trust: str = "numeric", engine: str | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_trajectory_3d(doc, trajectory, label=title, trust=trust)
    return doc


def surface_document(grid: Sequence[Sequence], *, title: str = "Surface",
                     x_range: tuple[float, float] | None = None,
                     y_range: tuple[float, float] | None = None,
                     trust: str = "numeric", engine: str | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_surface_3d(doc, grid, x_range=x_range, y_range=y_range, label=title,
                     trust=trust)
    return doc


def vector_field3d_document(origins: Sequence[Sequence], vectors: Sequence[Sequence],
                            *, title: str = "Vector field", trust: str = "numeric",
                            engine: str | None = None) -> VisualizationDocument:
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_vector_field_3d(doc, origins, vectors, label=title, trust=trust)
    return doc


def compare_document(prediction: tuple[Sequence, Sequence],
                     observation: tuple[Sequence, Sequence], *,
                     title: str = "Prediction vs observation",
                     prediction_trust: str = "exact",
                     observation_trust: str = "numeric",
                     observation_source: str = "",
                     engine: str | None = None) -> VisualizationDocument:
    """Prediction-vs-observation overlay preserving the provenance boundary."""
    doc = B.dashboard(title, cols=1, engine=engine)
    B.add_plot(doc, prediction[0], prediction[1], label="prediction",
               role="prediction", source="MathKernel", trust=prediction_trust,
               block_id="compare")
    B.add_plot(doc, observation[0], observation[1], label="observation",
               role="observation", source=observation_source,
               trust=observation_trust, marker="points", block_id="compare")
    doc.trust = doc.weakest_trust()
    return doc


def from_mathresult(result: Any, *, view: str = "auto",
                    title: str | None = None) -> VisualizationDocument:
    """Best-effort adapter dispatch on a MathResult's data shape."""
    d = result if isinstance(result, dict) else result.model_dump(mode="json")
    data = d.get("data", {})
    trust = _trust(d.get("trust", "unknown"))
    engine = d.get("engine")
    t = title or f"MathKernel result ({engine or 'unknown engine'})"

    def linked(doc: VisualizationDocument) -> VisualizationDocument:
        doc.linked_result = d
        return doc

    if view == "dag" and d.get("derivation"):
        return linked(dag_document(
            d["derivation"], title=t, trust=trust, engine=engine))

    if "matrix" in data and isinstance(data["matrix"], list):
        return linked(heatmap_document(
            data["matrix"], title=t, trust=trust, engine=engine))
    if "roots" in data and isinstance(data["roots"], list) and data["roots"] \
            and isinstance(data["roots"][0], dict) and "lower" in data["roots"][0]:
        return linked(interval_plot_document(
            data["roots"], title=t, engine=engine))
    if "points" in data and isinstance(data["points"], list):
        pts = data["points"]
        if pts and len(pts[0]) >= 3:
            return linked(points3d_document(
                pts, title=t, trust=trust, engine=engine))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return linked(plot_document(
            xs, ys, title=t, trust=trust, engine=engine))
    if "trajectory" in data:
        return linked(trajectory3d_document(
            data["trajectory"], title=t, trust=trust, engine=engine))
    if "grid" in data:
        return linked(surface_document(
            data["grid"], title=t, trust=trust, engine=engine))
    if any(key in data for key in (
        "value", "roc", "support", "query", "verification",
    )):
        doc = B.dashboard(t, cols=2, trust=trust, engine=engine)
        entries = [
            {"label": "status", "value": str(
                data.get("status", d.get("semantic_status", "unknown")))},
            {"label": "trust", "value": trust},
        ]
        if "query" in data:
            entries.append({"label": "query", "value": str(data["query"])})
        if "value" in data:
            entries.append({"label": "value", "value": str(data["value"])})
        B.add_metric_grid(doc, entries, title="Typed result")
        domain_rows = [
            [key, str(data[key])]
            for key in (
                "roc", "support", "conditions", "side_conditions",
                "branch_information", "verification", "verified_checks",
            )
            if data.get(key) not in (None, [], {})
        ]
        if domain_rows:
            B.add_data_table(
                doc, rows=domain_rows, title="Domain conditions")
        if d.get("derivation"):
            doc.provenance = provenance_from_steps(d["derivation"])
            doc.trust = doc.weakest_trust()
        return linked(doc)

    if d.get("derivation"):
        return linked(dag_document(
            d["derivation"], title=t, trust=trust, engine=engine))

    doc = B.dashboard(t, cols=1, trust=trust, engine=engine)
    B.add_text(doc, "Result shape has no dedicated adapter; provenance and "
                    "metadata only.")
    doc.annotations.append(Annotation(
        text="Result shape has no dedicated adapter; provenance and metadata only.",
        trust=trust))
    return linked(doc)

# ----------------------------------------------------------------------
# Shared multimodal projection adapter
# ----------------------------------------------------------------------

def _projection_dict(projection: Any) -> dict:
    return projection if isinstance(projection, dict) else projection.model_dump(mode="json")


def _link_projection(doc: VisualizationDocument, projection: Any) -> VisualizationDocument:
    """Attach canonical projection lineage to every generated visual payload."""
    p = _projection_dict(projection)
    from mathkernel_artifacts import SourceRef, Transformation
    src = SourceRef.model_validate(p["source_ref"])
    for ds in doc.datasets.values():
        ds.source_ref = src.model_copy(deep=True)
        ds.transformation_refs.append(f"{p['projection_id']}:construct")
    for series in doc.series.values():
        series.source_ref = src.model_copy(deep=True)
        series.transformation_refs.append(f"{p['projection_id']}:construct")
    doc.transformations.extend(Transformation.model_validate(t)
                               for t in p.get("transformations", []))
    doc.assumptions = list(p.get("assumptions", []))
    doc.metadata["projection"] = {
        "projection_id": p.get("projection_id"),
        "kind": p.get("kind"),
        "parameters": p.get("parameters", {}),
        "information_loss": p.get("information_loss", []),
        "information_loss_notes": p.get("information_loss_notes", []),
        "evidence_refs": p.get("evidence_refs", []),
    }
    # The whole document derives from the projection source, so the projection
    # trust is legitimate ancestry for every block — including blocks (DAG,
    # metric grids) that carry no trust-bearing series/datasets.  Include it in
    # the meet; this can weaken but never strengthen component trust.
    from mathkernel_artifacts import TRUST_RANK
    ptrust = str(p.get("trust", "unknown"))
    levels = [s.trust for s in doc.series.values()]
    levels += [d.trust for d in doc.datasets.values()]
    levels += [ps.trust for ps in doc.provenance.steps]
    levels = [l for l in levels if l in TRUST_RANK]
    if ptrust in TRUST_RANK:
        levels.append(ptrust)
    doc.trust = min(levels, key=TRUST_RANK.__getitem__) if levels else "unknown"
    return doc


def _graph_nodes(payload: dict) -> list[dict]:
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    out: dict[str, dict] = {}
    for i, n in enumerate(nodes):
        if isinstance(n, dict):
            nid = str(n.get("id", i))
            out[nid] = {"id": nid, "label": str(n.get("label", nid)),
                        "parents": list(map(str, n.get("parents", []))),
                        "trust": n.get("trust", "unknown")}
        else:
            nid = str(n)
            out[nid] = {"id": nid, "label": nid, "parents": [], "trust": "unknown"}
    for e in edges:
        if isinstance(e, dict):
            a, b = e.get("source"), e.get("target")
        else:
            a, b = e[0], e[1]
        a, b = str(a), str(b)
        out.setdefault(a, {"id": a, "label": a, "parents": [], "trust": "unknown"})
        out.setdefault(b, {"id": b, "label": b, "parents": [], "trust": "unknown"})
        if a not in out[b]["parents"]:
            out[b]["parents"].append(a)
    return list(out.values())


def _mesh_edges(vertices: list, cells: list) -> list[tuple[list, list]]:
    edges: set[tuple[int, int]] = set()
    for cell in cells:
        ids = list(cell.get("vertices", [])) if isinstance(cell, dict) else list(cell)
        if len(ids) == 2:
            pairs = [(ids[0], ids[1])]
        else:
            pairs = [(ids[i], ids[(i + 1) % len(ids)]) for i in range(len(ids))]
        for a, b in pairs:
            edges.add(tuple(sorted((int(a), int(b)))))
    return [(vertices[a], vertices[b]) for a, b in sorted(edges)
            if 0 <= a < len(vertices) and 0 <= b < len(vertices)]


def from_projection(projection: Any, *, title: str | None = None) -> VisualizationDocument:
    """Visualize any canonical ``MultimodalProjection`` without recomputation.

    The adapter uses a small set of stable rendering primitives.  Domain
    semantics stay in the projection object, so adding a new mathematical
    domain does not require a new renderer merely to preserve lineage.
    """
    p = _projection_dict(projection)
    kind = p["kind"]
    q = p.get("payload", {})
    trust = p.get("trust", "unknown")
    t = title or p.get("title") or kind.replace("_", " ").title()

    if kind in {"matrix", "finite_field"} and "matrix" in q:
        return _link_projection(heatmap_document(q["matrix"], title=t, trust=trust), p)
    if kind == "finite_field" and "values" in q:
        return _link_projection(plot_document(list(range(len(q["values"]))), q["values"],
                                              title=t, trust=trust), p)

    if kind == "tensor":
        matrix = q.get("slice")
        if matrix is None:
            shape = list(q.get("shape", []))
            values = list(q.get("values", []))
            if len(shape) == 2 and shape[0] * shape[1] == len(values):
                matrix = [values[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]
            else:
                doc = B.dashboard(t, cols=1, trust=trust)
                B.add_data_table(doc, rows=[["shape", shape], ["values", len(values)]],
                                 title="Tensor projection")
                return _link_projection(doc, p)
        return _link_projection(heatmap_document(matrix, title=t, trust=trust), p)

    if kind in {"scalar_field", "surface", "pde_solution"}:
        if "grid" in q:
            return _link_projection(surface_document(q["grid"], title=t, trust=trust), p)
        pts, vals = q.get("points", []), q.get("values", [])
        if pts and len(pts[0]) >= 2:
            doc = B.dashboard(t, cols=1, trust=trust)
            # Scatter encoded as x/y; scalar value retained in a table and in
            # projection metadata rather than silently mapped to arbitrary color.
            B.add_plot(doc, [x[0] for x in pts], [x[1] for x in pts], marker="points",
                       label="sample locations", trust=trust)
            if vals:
                B.add_data_table(doc, rows=[[i, *pts[i], vals[i]] for i in range(min(len(pts), len(vals)))],
                                 title="Field samples")
            return _link_projection(doc, p)

    if kind == "vector_field":
        origins, vectors = q["origins"], q["vectors"]
        if origins and len(origins[0]) >= 3:
            return _link_projection(vector_field3d_document(origins, vectors, title=t, trust=trust), p)
        if origins and len(origins[0]) == 2:
            # Embed the 2D field in the z=0 plane so the arrow renderer
            # applies; an embedding is not information loss.
            o3 = [[o[0], o[1], 0.0] for o in origins]
            v3 = [[v[0], v[1], 0.0] for v in vectors]
            doc = vector_field3d_document(o3, v3, title=t, trust=trust)
            doc.metadata["embedding"] = "2D vector field embedded in the z=0 plane"
            return _link_projection(doc, p)
        doc = B.dashboard(t, cols=1, trust=trust)
        block = None
        for o, v in zip(origins, vectors):
            end = [float(o[0]) + float(v[0]), float(o[1]) + float(v[1])]
            block = B.add_plot(doc, [o[0], end[0]], [o[1], end[1]], trust=trust,
                               label="vector" if block is None else "",
                               block_id=(block.block_id if block else None))
        return _link_projection(doc, p)

    if kind in {"point_set", "point_cloud", "high_dimensional"}:
        pts = q["points"]
        if pts and len(pts[0]) >= 3:
            return _link_projection(points3d_document(pts, title=t, trust=trust), p)
        if pts and len(pts[0]) >= 2:
            return _link_projection(plot_document([x[0] for x in pts], [x[1] for x in pts],
                                                  title=t, trust=trust), p)
        vals = [x[0] if isinstance(x, (list, tuple)) else x for x in pts]
        return _link_projection(plot_document(list(range(len(vals))), vals, title=t, trust=trust), p)

    if kind == "intervals":
        return _link_projection(interval_plot_document(q["intervals"], title=t, trust=trust), p)

    if kind in {"curve", "function", "sequence", "spectrum", "ode_solution", "trajectory", "prng_analysis"}:
        if kind == "curve":
            if "points" in q:
                xs, ys = [x[0] for x in q["points"]], [x[1] for x in q["points"]]
            else:
                xs, ys = q["x"], q["y"]
        elif kind == "function":
            ys = q.get("y", q.get("values", []))
            xs = q.get("x", list(range(len(ys))))
        elif kind == "trajectory":
            tr = q.get("states", q.get("points", []))
            if tr and isinstance(tr[0], (list, tuple)) and len(tr[0]) >= 3:
                return _link_projection(trajectory3d_document(tr, title=t, trust=trust), p)
            if tr and isinstance(tr[0], (list, tuple)) and len(tr[0]) >= 2:
                xs, ys = [x[0] for x in tr], [x[1] for x in tr]
            else:
                ys = [x[0] if isinstance(x, (list, tuple)) else x for x in tr]
                xs = list(range(len(ys)))
        elif kind == "ode_solution":
            if "t" in q and "states" in q:
                states = q["states"]
                if states and isinstance(states[0], (list, tuple)):
                    doc = B.dashboard(t, cols=1, trust=trust)
                    block = None
                    for j in range(len(states[0])):
                        block = B.add_plot(doc, q["t"], [row[j] for row in states],
                                           label=f"state {j}", trust=trust,
                                           block_id=(block.block_id if block else None))
                    return _link_projection(doc, p)
                xs, ys = q["t"], states
            else:
                tr = q["trajectory"]
                if tr and len(tr[0]) >= 3:
                    return _link_projection(trajectory3d_document(tr, title=t, trust=trust), p)
                xs, ys = list(range(len(tr))), [x[-1] if isinstance(x, (list, tuple)) else x for x in tr]
        elif kind == "spectrum":
            ys = q.get("amplitude", q.get("values", []))
            xs = q.get("frequency", list(range(len(ys))))
        elif kind == "prng_analysis":
            if "matrix" in q:
                return _link_projection(heatmap_document(q["matrix"], title=t, trust=trust), p)
            ys = q.get("spectrum", q.get("values", []))
            xs = q.get("lags", list(range(len(ys))))
        else:
            ys = q["values"]
            xs = q.get("x", q.get("t", list(range(len(ys)))))
        return _link_projection(plot_document(xs, ys, title=t, trust=trust), p)

    if kind == "distribution":
        doc = B.dashboard(t, cols=1, trust=trust)
        if "pdf" in q:
            B.add_plot(doc, q["x"], q["pdf"], label="PDF", trust=trust)
        elif "cdf" in q:
            B.add_plot(doc, q["x"], q["cdf"], label="CDF", trust=trust)
        else:
            B.add_histogram(doc, q["values"], trust=trust)
        return _link_projection(doc, p)

    if kind in {"graph", "evidence_graph", "expression_tree", "certificate_tree", "dynamical_system", "geometric_complex", "finite_field"} and "nodes" in q:
        doc = B.dashboard(t, cols=1, trust=trust)
        B.add_dag(doc, nodes=_graph_nodes(q), title=t)
        return _link_projection(doc, p)

    if kind in {"mesh", "geometric_complex"} and "vertices" in q:
        vertices, cells = q["vertices"], q["cells"]
        if vertices and len(vertices[0]) >= 3:
            doc = B.dashboard(t, cols=1, trust=trust)
            B.add_point_cloud(doc, vertices, trust=trust, title="Mesh vertices")
            block = None
            for a, b in _mesh_edges(vertices, cells):
                block = B.add_trajectory_3d(
                    doc, [a, b], trust=trust,
                    label="edge" if block is None else "",
                    title="Mesh edges" if block is None else "",
                    block_id=(block.block_id if block else None))
            B.add_data_table(doc, rows=[[i, cell] for i, cell in enumerate(cells)], title="Cells")
            return _link_projection(doc, p)
        doc = B.dashboard(t, cols=1, trust=trust)
        block = None
        for a, b in _mesh_edges(vertices, cells):
            block = B.add_plot(doc, [a[0], b[0]], [a[1], b[1]], trust=trust,
                               block_id=(block.block_id if block else None))
        return _link_projection(doc, p)

    if kind == "complex_field":
        doc = B.dashboard(t, cols=2, trust=trust)
        if "magnitude" in q and "phase" in q:
            B.add_heatmap(doc, q["magnitude"], trust=trust, title="Magnitude")
            B.add_heatmap(doc, q["phase"], trust=trust, title="Phase")
        elif "real" in q and "imag" in q:
            B.add_heatmap(doc, q["real"], trust=trust, title="Real")
            B.add_heatmap(doc, q["imag"], trust=trust, title="Imaginary")
        else:
            pts = q["points"]
            B.add_plot(doc, [z[0] for z in pts], [z[1] for z in pts], marker="points",
                       x_label="Re", y_label="Im", trust=trust)
        return _link_projection(doc, p)

    if kind in {"region", "set", "implicit_set"}:
        doc = B.dashboard(t, cols=1, trust=trust)
        boundary = q.get("boundary")
        if boundary:
            B.add_plot(doc, [x[0] for x in boundary], [x[1] for x in boundary], trust=trust)
        elif "polygons" in q:
            block = None
            for poly in q["polygons"]:
                pts = list(poly) + ([poly[0]] if poly and poly[0] != poly[-1] else [])
                block = B.add_plot(doc, [x[0] for x in pts], [x[1] for x in pts], trust=trust,
                                   block_id=(block.block_id if block else None))
        elif "points" in q:
            B.add_plot(doc, [x[0] for x in q["points"]], [x[1] for x in q["points"]],
                       marker="points", trust=trust)
        elif "grid" in q:
            B.add_heatmap(doc, q["grid"], trust=trust)
        else:
            B.add_data_table(doc, rows=[["values", q.get("values", [])]], title="Set values")
        return _link_projection(doc, p)

    if kind == "partition":
        doc = B.dashboard(t, cols=1, trust=trust)
        rows = [[i, part] for i, part in enumerate(q["parts"])]
        B.add_data_table(doc, rows=rows, title="Partition parts")
        return _link_projection(doc, p)

    if kind == "piecewise":
        doc = B.dashboard(t, cols=1, trust=trust)
        block = None
        for i, piece in enumerate(q["pieces"]):
            xs = piece.get("x", []) if isinstance(piece, dict) else []
            ys = piece.get("y", []) if isinstance(piece, dict) else []
            if xs and ys:
                block = B.add_plot(doc, xs, ys, label=str(piece.get("domain", f"piece {i}")),
                                   trust=trust, block_id=(block.block_id if block else None))
        if block is None:
            B.add_data_table(doc, rows=[[i, piece] for i, piece in enumerate(q["pieces"])])
        return _link_projection(doc, p)

    if kind == "quantity":
        doc = B.dashboard(t, cols=1, trust=trust)
        entries = [{"label": "value", "value": q["value"]}]
        for key in ("unit", "lower", "upper", "uncertainty"):
            if key in q:
                entries.append({"label": key, "value": q[key]})
        B.add_metric_grid(doc, entries, title=t)
        return _link_projection(doc, p)

    if kind in {"optimization", "statistical_inference", "relation_geometry"}:
        doc = B.dashboard(t, cols=2, trust=trust)
        if "trace" in q:
            vals = [x.get("objective", x) if isinstance(x, dict) else x for x in q["trace"]]
            B.add_plot(doc, list(range(len(vals))), vals, title="Optimization trace", trust=trust)
        if "samples" in q:
            B.add_histogram(doc, q["samples"], title="Sampling distribution", trust=trust)
        if "values" in q:
            B.add_histogram(doc, q["values"], title="Distribution", trust=trust)
        if "eigenvalues" in q:
            B.add_plot(doc, list(range(len(q["eigenvalues"]))), q["eigenvalues"],
                       title="Eigenvalues", trust=trust)
        if "matrix" in q:
            B.add_heatmap(doc, q["matrix"], title="Matrix", trust=trust)
        if "points" in q and q["points"]:
            pts = q["points"]
            if len(pts[0]) >= 3:
                B.add_point_cloud(doc, pts, title="Geometry", trust=trust)
            else:
                B.add_plot(doc, [x[0] for x in pts], [x[1] for x in pts], marker="points",
                           title="Geometry", trust=trust)
        if "observed" in q:
            B.add_metric_grid(doc, [{"label": "observed", "value": q["observed"]}],
                              title="Observed statistic")
        return _link_projection(doc, p)

    if kind == "ensemble":
        doc = B.dashboard(t, cols=1, trust=trust)
        members = q.get("members", [])
        if members and isinstance(members[0], (list, tuple)):
            block = None
            for i, member in enumerate(members):
                block = B.add_plot(doc, list(range(len(member))), member, label=f"member {i}",
                                   trust=trust, block_id=(block.block_id if block else None))
        else:
            B.add_histogram(doc, q.get("values", members), trust=trust)
        return _link_projection(doc, p)

    # Every supported projection has a lossless descriptive fallback.  This is
    # deliberately explicit rather than pretending an arbitrary visual mapping
    # has canonical mathematical meaning.
    doc = B.dashboard(t, cols=1, trust=trust)
    B.add_data_table(doc, rows=[[k, v] for k, v in q.items()], title="Projection data")
    return _link_projection(doc, p)
