# =============================================================================
# MathKernel Sonify - adapters for spectra, sequences and comparisons
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Scientific adapters: spectra, sequences, comparisons and MathResult-like data."""
from __future__ import annotations
import hashlib, json, math
from typing import Any
from mathkernel_artifacts import SourceRef, Transformation
from .models import SonificationDocument, Mapping, AudioTrack, SonificationEvent, RenderConfig

def _hash(x: Any)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def harmonic_sonification(amplitudes, phases=None, *, fundamental=110.0, duration=2.0, title="Harmonic sonification", trust="numeric", sample_rate=48000):
    amps=[float(x) for x in amplitudes]; phases=[0.0]*len(amps) if phases is None else [float(x) for x in phases]
    if not amps: raise ValueError("amplitudes must contain at least one value")
    if len(phases)!=len(amps): raise ValueError("phases and amplitudes must have equal length")
    source=SourceRef(source_id=f"spectrum:sha256:{_hash([amps,phases])}",kind="dataset",sha256=_hash([amps,phases]),trust=trust,label="harmonic spectrum")
    doc=SonificationDocument(title=title,trust=trust,render=RenderConfig(sample_rate=sample_rate)); doc.sources[source.source_id]=source
    doc.mappings["harmonic-frequency"]=Mapping(mapping_id="harmonic-frequency",source="harmonic",target="frequency",transform={"type":"harmonic","fundamental":fundamental},units_out="Hz")
    doc.mappings["amplitude-gain"]=Mapping(mapping_id="amplitude-gain",source="amplitude",target="gain",transform={"type":"identity"})
    doc.mappings["complex-phase"]=Mapping(mapping_id="complex-phase",source="coefficient_phase",target="phase",transform={"type":"identity"},units_out="rad")
    events=[]
    for i,(a,p) in enumerate(zip(amps,phases),1):
        events.append(SonificationEvent(event_id=f"h{i}",time=0,duration=duration,source_ref=source.source_id,label=f"h{i}",values={"harmonic":i,"amplitude":a,"coefficient_phase":p}))
    doc.tracks=[AudioTrack(track_id="spectrum",label="spectrum",source_ref=source,mapping_refs=list(doc.mappings),events=events,trust=trust)]
    doc.transformations.append(Transformation(transformation_id="sonify:harmonic",operation="harmonic_additive_synthesis",inputs=[source.source_id],outputs=["track:spectrum"],parameters={"fundamental_hz":fundamental,"duration_s":duration,"amplitude":"identity","phase":"identity"},purpose="encoding",trust=trust))
    return doc

def scan_sonification(values, *, fmin=110.0, fmax=1760.0, seconds_per_item=.1, title="Sequence scan", trust="numeric", sample_rate=48000):
    vals=[float(v) for v in values]
    if not vals: raise ValueError("values must contain at least one item to sonify")
    lo=min(vals); hi=max(vals)
    src=SourceRef(source_id=f"sequence:sha256:{_hash(vals)}",kind="dataset",sha256=_hash(vals),trust=trust,label="sequence")
    doc=SonificationDocument(title=title,trust=trust,render=RenderConfig(sample_rate=sample_rate)); doc.sources[src.source_id]=src
    doc.mappings["value-frequency"]=Mapping(mapping_id="value-frequency",source="value",target="frequency",transform={"type":"log_map","source_min":lo,"source_max":hi,"target_min":fmin,"target_max":fmax},units_out="Hz")
    ev=[SonificationEvent(event_id=f"item-{i}",time=i*seconds_per_item,duration=seconds_per_item,source_ref=src.source_id,label=str(i),values={"value":v,"gain":.7}) for i,v in enumerate(vals)]
    doc.tracks=[AudioTrack(track_id="scan",source_ref=src,mapping_refs=["value-frequency"],events=ev,trust=trust)]
    doc.transformations.append(Transformation(transformation_id="sonify:scan",operation="sequence_to_log_frequency",inputs=[src.source_id],outputs=["track:scan"],parameters={"source_min":lo,"source_max":hi,"fmin":fmin,"fmax":fmax,"seconds_per_item":seconds_per_item},purpose="encoding",trust=trust))
    return doc

def compare_sonification(prediction, observation, *, mode="stereo", **kwargs):
    p=[float(x) for x in prediction]; o=[float(x) for x in observation]
    if len(p)!=len(o): raise ValueError("prediction and observation lengths differ")
    if mode=="residual": return scan_sonification([b-a for a,b in zip(p,o)],title=kwargs.pop("title","Prediction residual"),**kwargs)
    dp=scan_sonification(p,title=kwargs.get("title","Prediction vs observation"),**{k:v for k,v in kwargs.items() if k!='title'})
    do=scan_sonification(o,title=dp.title,**{k:v for k,v in kwargs.items() if k!='title'})
    # Keep one common scale to avoid independent-normalization deception.
    vals=p+o; lo=min(vals,default=0); hi=max(vals,default=1)
    for d in (dp,do): d.mappings["value-frequency"].transform.update(source_min=lo,source_max=hi)
    pt=dp.tracks[0]; ot=do.tracks[0]; pt.role="prediction"; ot.role="observation"; pt.pan=-1 if mode=="stereo" else 0; ot.pan=1 if mode=="stereo" else 0
    ot.track_id="observation"; pt.track_id="prediction"
    dp.tracks=[pt,ot]; dp.sources.update(do.sources); dp.metadata["comparison_mode"]=mode
    return dp

# ----------------------------------------------------------------------
# Shared multimodal projection sonification
# ----------------------------------------------------------------------

def _projection_dict(projection: Any) -> dict:
    return projection if isinstance(projection, dict) else projection.model_dump(mode="json")


def _retarget_projection(doc: SonificationDocument, projection: Any, *,
                         extraction: str, extraction_parameters: dict | None = None) -> SonificationDocument:
    """Bind a sonification to the same source lineage as its visual projection."""
    from mathkernel_artifacts import SourceRef, Transformation
    p = _projection_dict(projection)
    src = SourceRef.model_validate(p["source_ref"])
    old_sources = set(doc.sources)
    doc.sources = {src.source_id: src}
    for track in doc.tracks:
        track.source_ref = src.model_copy(deep=True)
        track.trust = p.get("trust", "unknown")
        for ev in track.events:
            ev.source_ref = src.source_id
    # Preserve the canonical projection transform, then explicitly document the
    # domain-specific extraction used to turn structure into acoustic events.
    base = [Transformation.model_validate(t) for t in p.get("transformations", [])]
    extract_id = f"{p['projection_id']}:sonify-extract"
    base.append(Transformation(
        transformation_id=extract_id, operation=extraction,
        inputs=[p["projection_id"]], outputs=[f"{p['projection_id']}:audio-sequence"],
        parameters=dict(extraction_parameters or {}), purpose="encoding",
        trust=p.get("trust", "unknown"),
        notes="Explicit acoustic projection; audible structure is candidate observation, not proof."))
    for tr in doc.transformations:
        if any(x in old_sources for x in tr.inputs):
            tr.inputs = [f"{p['projection_id']}:audio-sequence"]
    doc.transformations = base + doc.transformations
    doc.assumptions = list(p.get("assumptions", []))
    doc.trust = p.get("trust", "unknown")
    doc.metadata["projection"] = {
        "projection_id": p.get("projection_id"), "kind": p.get("kind"),
        "parameters": p.get("parameters", {}),
        "information_loss": p.get("information_loss", []),
        "information_loss_notes": p.get("information_loss_notes", []),
        "evidence_refs": p.get("evidence_refs", []),
        "acoustic_extraction": extraction,
        "acoustic_extraction_parameters": dict(extraction_parameters or {}),
    }
    return doc


def _flatten_matrix(matrix) -> list[float]:
    return [float(x) for row in matrix for x in row]


def _norms(rows) -> list[float]:
    return [math.sqrt(sum(float(x) ** 2 for x in row)) for row in rows]


def _degrees(payload: dict) -> list[float]:
    nodes = payload.get("nodes", [])
    ids = [str(n.get("id", i)) if isinstance(n, dict) else str(n) for i, n in enumerate(nodes)]
    deg = {n: 0 for n in ids}
    for e in payload.get("edges", []):
        if isinstance(e, dict):
            a, b = str(e.get("source")), str(e.get("target"))
        else:
            a, b = str(e[0]), str(e[1])
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return [float(deg[n]) for n in ids] or [0.0]


def _interval_bounds(p: dict) -> tuple[Any, Any]:
    """Accept both catalog-style lo/hi and lower/upper interval keys."""
    lo = p.get("lower", p.get("lo"))
    hi = p.get("upper", p.get("hi"))
    if lo is None or hi is None:
        raise ValueError(
            "interval entries require 'lower'/'upper' (or 'lo'/'hi') keys; "
            f"got {sorted(p)}")
    return lo, hi


def _boundary_lengths(points) -> list[float]:
    pts = list(points)
    if len(pts) < 2:
        return [0.0]
    return [math.sqrt(sum((float(b) - float(a)) ** 2 for a, b in zip(pts[i], pts[i + 1])))
            for i in range(len(pts) - 1)]


def projection_sonification(projection: Any, *, mode: str = "auto",
                             options: dict | None = None) -> SonificationDocument:
    """Sonify every canonical multimodal projection with an explicit mapping.

    Structured objects are never silently flattened.  Any ordering, reduction,
    slice or summary used to obtain an audible scalar sequence is named in the
    transformation provenance and exposed in document metadata.
    """
    p = _projection_dict(projection)
    kind, q = p["kind"], p.get("payload", {})
    trust, title = p.get("trust", "unknown"), p.get("title", "Projection sonification")
    opts = dict(options or {})
    opts.setdefault("title", title)
    opts.setdefault("trust", trust)

    if kind == "spectrum":
        amps = q.get("amplitude", q.get("values", []))
        phases = q.get("phase")
        harmonic_opts = {k: v for k, v in opts.items()
                         if k in {"fundamental", "duration", "title", "trust", "sample_rate"}}
        doc = harmonic_sonification(amps, phases=phases, **harmonic_opts)
        return _retarget_projection(doc, p, extraction="spectrum_coefficients",
                                    extraction_parameters={"ordering": "input order"})

    if kind == "intervals":
        iv = q["intervals"]
        bounds = [_interval_bounds(x) for x in iv]
        lowers = [float(lo) for lo, _ in bounds]
        uppers = [float(hi) for _, hi in bounds]
        doc = compare_sonification(lowers, uppers, mode="stereo",
                                   **{k: v for k, v in opts.items()
                                      if k in {"fmin", "fmax", "seconds_per_item", "title", "trust", "sample_rate"}})
        return _retarget_projection(doc, p, extraction="interval_lower_upper_stereo",
                                    extraction_parameters={"left_channel": "lower", "right_channel": "upper", "ordering": "interval order"})

    if kind == "complex_field":
        if "magnitude" in q and "phase" in q:
            mags = _flatten_matrix(q["magnitude"]) if q["magnitude"] and isinstance(q["magnitude"][0], (list, tuple)) else list(q["magnitude"])
            phases = _flatten_matrix(q["phase"]) if q["phase"] and isinstance(q["phase"][0], (list, tuple)) else list(q["phase"])
            extraction = "complex_magnitude_phase_row_major"
        elif "real" in q and "imag" in q:
            real = _flatten_matrix(q["real"]) if q["real"] and isinstance(q["real"][0], (list, tuple)) else list(q["real"])
            imag = _flatten_matrix(q["imag"]) if q["imag"] and isinstance(q["imag"][0], (list, tuple)) else list(q["imag"])
            if len(real) != len(imag):
                raise ValueError("complex real/imaginary components differ in length")
            mags = [math.hypot(float(a), float(b)) for a, b in zip(real, imag)]
            phases = [math.atan2(float(b), float(a)) for a, b in zip(real, imag)]
            extraction = "complex_real_imag_to_magnitude_phase_row_major"
        else:
            pts = q["points"]
            mags = [math.hypot(float(z[0]), float(z[1])) for z in pts]
            phases = [math.atan2(float(z[1]), float(z[0])) for z in pts]
            extraction = "complex_points_to_magnitude_phase"
        doc = harmonic_sonification(mags, phases=phases,
                                    **{k: v for k, v in opts.items()
                                       if k in {"fundamental", "duration", "title", "trust", "sample_rate"}})
        return _retarget_projection(doc, p, extraction=extraction,
                                    extraction_parameters={"magnitude_to": "gain", "phase_to": "oscillator_phase"})

    extraction = "identity_sequence"
    params: dict[str, Any] = {}
    values: list[float]

    if kind == "sequence":
        values = [float(x) for x in q["values"]]
    elif kind == "function":
        values = [float(x) for x in q.get("y", q.get("values", []))]
        extraction, params = "function_value_scan", {"ordering": "x/sample order"}
    elif kind == "trajectory":
        tr = q.get("states", q.get("points", []))
        values = _norms(tr) if tr and isinstance(tr[0], (list, tuple)) else [float(x) for x in tr]
        extraction, params = "trajectory_state_norm_scan", {"reduction": "euclidean_norm when vector-valued", "ordering": "trajectory order"}
    elif kind == "curve":
        values = [float(x) for x in (q.get("y") or [pt[1] for pt in q.get("points", [])])]
        extraction, params = "curve_y_scan", {"ordering": "parameter/sample order"}
    elif kind == "distribution":
        values = [float(x) for x in q.get("pdf", q.get("cdf", q.get("values", [])))]
        extraction, params = "distribution_scan", {"representation": "pdf" if "pdf" in q else "cdf" if "cdf" in q else "empirical_values"}
    elif kind in {"matrix", "finite_field"} and "matrix" in q:
        values = _flatten_matrix(q["matrix"])
        extraction, params = "matrix_row_major_scan", {"ordering": "row-major", "shape": [len(q["matrix"]), len(q["matrix"][0]) if q["matrix"] else 0]}
    elif kind == "tensor":
        if "slice" in q:
            values = _flatten_matrix(q["slice"])
            extraction, params = "tensor_slice_row_major_scan", {"slice": p.get("parameters", {}).get("slice"), "ordering": "row-major"}
        else:
            values = [float(x) for x in q["values"]]
            extraction, params = "tensor_storage_order_scan", {"shape": q.get("shape"), "ordering": p.get("parameters", {}).get("ordering", "declared storage order")}
    elif kind in {"scalar_field", "surface", "pde_solution"}:
        if "grid" in q:
            values = _flatten_matrix(q["grid"])
            extraction, params = "field_grid_row_major_scan", {"ordering": "row-major"}
        elif "values" in q:
            values = [float(x) for x in q.get("values", [])]
            extraction, params = "field_sample_scan", {"ordering": "point order"}
        else:
            pts = q.get("points", [])
            values = [float(x[-1]) for x in pts]
            extraction, params = "surface_height_scan", {"ordering": "point order", "value": "last coordinate"}
    elif kind == "vector_field":
        values = _norms(q["vectors"])
        extraction, params = "vector_magnitude_scan", {"reduction": "euclidean_norm", "ordering": "origin order"}
    elif kind in {"point_set", "point_cloud", "high_dimensional"}:
        values = _norms(q["points"])
        extraction, params = "point_radius_scan", {"reduction": "euclidean_norm", "ordering": "point order"}
    elif kind in {"graph", "evidence_graph", "expression_tree", "certificate_tree", "geometric_complex"} and "nodes" in q:
        values = _degrees(q)
        extraction, params = "graph_degree_scan", {"ordering": "node order", "value": "undirected incident degree"}
    elif kind == "finite_field":
        if "values" in q:
            values = [float(x) for x in q["values"]]
            extraction, params = "finite_field_value_scan", {"ordering": "declared value order"}
        elif "nodes" in q:
            values = _degrees(q)
            extraction, params = "finite_field_state_graph_degree_scan", {"ordering": "node order"}
        else:
            values = _flatten_matrix(q["matrix"])
            extraction, params = "finite_field_matrix_row_major_scan", {"ordering": "row-major"}
    elif kind in {"mesh", "geometric_complex"} and "vertices" in q:
        verts = q["vertices"]
        vals = []
        for cell in q["cells"]:
            ids = list(cell.get("vertices", [])) if isinstance(cell, dict) else list(cell)
            if len(ids) >= 2:
                for i in range(len(ids)):
                    a, b = verts[int(ids[i])], verts[int(ids[(i + 1) % len(ids)])]
                    vals.append(math.sqrt(sum((float(y) - float(x)) ** 2 for x, y in zip(a, b))))
        values = vals or [0.0]
        extraction, params = "mesh_edge_length_scan", {"ordering": "cell order then boundary edge order"}
    elif kind in {"region", "set", "implicit_set"}:
        if "boundary" in q:
            values = _boundary_lengths(q["boundary"])
            extraction, params = "boundary_segment_length_scan", {"ordering": "boundary order"}
        elif "points" in q:
            values = _norms(q["points"])
            extraction, params = "set_point_radius_scan", {"ordering": "point order"}
        elif "grid" in q:
            values = _flatten_matrix(q["grid"])
            extraction, params = "occupancy_grid_row_major_scan", {"ordering": "row-major"}
        else:
            values = [float(len(q.get("values", [])))]
            extraction, params = "set_cardinality", {"summary": "cardinality"}
    elif kind == "partition":
        values = [float(len(part)) if hasattr(part, "__len__") else 1.0 for part in q["parts"]]
        extraction, params = "partition_part_size_scan", {"ordering": "part order"}
    elif kind == "piecewise":
        values = []
        for piece in q["pieces"]:
            if isinstance(piece, dict) and "y" in piece:
                values.extend(float(x) for x in piece["y"])
        if not values:
            values = [float(len(q["pieces"]))]
        extraction, params = "piecewise_branch_scan", {"ordering": "piece order then sample order"}
    elif kind == "quantity":
        values = [float(q["value"])]
        extraction, params = "scalar_quantity", {"unit": q.get("unit")}
    elif kind == "optimization":
        trace = q.get("trace", [])
        if trace:
            values = [float(x.get("objective", x)) if isinstance(x, dict) else float(x) for x in trace]
            extraction, params = "optimization_objective_trace", {"ordering": "iteration order"}
        elif "objective" in q:
            values = [float(x) for x in q["objective"]]
            extraction, params = "objective_sample_scan", {"ordering": "point order"}
        else:
            values = _boundary_lengths(q.get("feasible_boundary", []))
            extraction, params = "feasible_boundary_segment_scan", {"ordering": "boundary order"}
    elif kind == "statistical_inference":
        values = [float(x) for x in q.get("samples", q.get("values", []))]
        if not values and "observed" in q:
            values = [float(q["observed"])]
        extraction, params = "inference_distribution_scan", {"representation": "samples/null/bootstrap distribution"}
    elif kind == "dynamical_system":
        if "trajectory" in q:
            tr = q["trajectory"]
            values = _norms(tr) if tr and isinstance(tr[0], (list, tuple)) else [float(x) for x in tr]
            extraction, params = "dynamical_trajectory_scan", {"vector_reduction": "euclidean_norm when vector-valued"}
        elif "values" in q:
            values = [float(x) for x in q["values"]]
            extraction = "dynamical_observation_scan"
        else:
            values = _degrees(q)
            extraction = "state_graph_degree_scan"
    elif kind == "ode_solution":
        states = q.get("states", q.get("trajectory", []))
        values = _norms(states) if states and isinstance(states[0], (list, tuple)) else [float(x) for x in states]
        extraction, params = "ode_state_norm_scan", {"reduction": "euclidean_norm when vector-valued", "ordering": "time order"}
    elif kind == "prng_analysis":
        if "spectrum" in q:
            values = [float(x) for x in q["spectrum"]]
            extraction, params = "prng_spectrum_scan", {"ordering": "lag/harmonic order"}
        elif "matrix" in q:
            values = _flatten_matrix(q["matrix"])
            extraction, params = "prng_matrix_row_major_scan", {"ordering": "row-major"}
        else:
            values = [float(x) for x in q["values"]]
            extraction, params = "prng_value_scan", {"ordering": "sample/lag order"}
    elif kind == "relation_geometry":
        if "eigenvalues" in q:
            values = [float(x) for x in q["eigenvalues"]]
            extraction = "relation_eigenvalue_scan"
        elif "matrix" in q:
            values = _flatten_matrix(q["matrix"])
            extraction, params = "relation_matrix_row_major_scan", {"ordering": "row-major"}
        else:
            values = _norms(q["points"])
            extraction = "relation_geometry_radius_scan"
    elif kind == "ensemble":
        members = q.get("members", [])
        if members and isinstance(members[0], (list, tuple)):
            values = [float(x) for member in members for x in member]
            extraction, params = "ensemble_member_concatenation", {"ordering": "member order then sample order", "member_lengths": [len(x) for x in members]}
        else:
            values = [float(x) for x in q.get("values", members)]
            extraction = "ensemble_value_scan"
    else:
        raise ValueError(f"projection kind {kind!r} has no sonification adapter")

    if not values:
        raise ValueError(f"projection kind {kind!r} contains no values to sonify")
    scan_opts = {k: v for k, v in opts.items()
                 if k in {"fmin", "fmax", "seconds_per_item", "title", "trust", "sample_rate"}}
    doc = scan_sonification(values, **scan_opts)
    return _retarget_projection(doc, p, extraction=extraction, extraction_parameters=params)
