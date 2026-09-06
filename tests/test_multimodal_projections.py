from __future__ import annotations

import mathkernel_projection as mkp
import mathkernel_viz as viz
import mathkernel_sonify as son
from mathkernel import MathKernel


PAYLOADS = {
    "scalar_field": {"grid": [[0, 1], [2, 3]]},
    "vector_field": {"origins": [[0, 0], [1, 0]], "vectors": [[1, 0], [0, 1]]},
    "point_set": {"points": [[0, 0], [1, 1]]},
    "point_cloud": {"points": [[0, 0, 0], [1, 1, 1]]},
    "curve": {"points": [[0, 0], [1, 1], [2, 0]]},
    "function": {"x": [0, 1, 2], "y": [0, 1, 4]},
    "trajectory": {"states": [[0, 0], [1, 0], [1, 1]]},
    "intervals": {"intervals": [{"x": 0, "lower": "0.9", "upper": "1.1"}, {"x": 1, "lower": "1.8", "upper": "2.2"}]},
    "surface": {"grid": [[0, 1], [1, 0]]},
    "sequence": {"values": [1, 2, 3]},
    "distribution": {"x": [0, 1, 2], "pdf": [0.2, 0.6, 0.2]},
    "matrix": {"matrix": [[1, 2], [3, 4]]},
    "tensor": {"values": [1, 2, 3, 4], "shape": [2, 2]},
    "graph": {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]},
    "spectrum": {"values": [0.4, 0.2, 0.1]},
    "region": {"boundary": [[0, 0], [1, 0], [1, 1], [0, 0]]},
    "implicit_set": {"grid": [[0, 1], [1, 0]]},
    "evidence_graph": {"nodes": [{"id": "claim", "label": "claim"}, {"id": "e", "label": "evidence", "parents": ["claim"]}]},
    "mesh": {"vertices": [[0, 0], [1, 0], [0, 1]], "cells": [[0, 1, 2]]},
    "complex_field": {"real": [[1, 0], [0, -1]], "imag": [[0, 1], [-1, 0]]},
    "optimization": {"trace": [{"objective": 4}, {"objective": 2}, {"objective": 1}]},
    "statistical_inference": {"samples": [-1, 0, 0.5, 1], "observed": 0.7},
    "dynamical_system": {"trajectory": [[0, 0], [1, 0], [1, 1]]},
    "pde_solution": {"grid": [[0, 1], [1, 0]]},
    "ode_solution": {"t": [0, 1, 2], "states": [[0, 0], [1, 0], [1, 1]]},
    "finite_field": {"matrix": [[1, 0], [1, 1]]},
    "relation_geometry": {"eigenvalues": [2, 0.5, 0.1]},
    "prng_analysis": {"spectrum": [0.1, 1.2, 0.2], "lags": [1, 2, 3]},
    "set": {"points": [[0, 0], [1, 1]]},
    "partition": {"parts": [[1, 2], [3], [4, 5, 6]]},
    "piecewise": {"pieces": [{"domain": "x<0", "x": [-2, -1], "y": [4, 1]}, {"domain": "x>=0", "x": [0, 1], "y": [0, 1]}]},
    "expression_tree": {"nodes": [{"id": "x", "label": "x"}, {"id": "sq", "label": "pow", "parents": ["x"]}]},
    "certificate_tree": {"nodes": [{"id": "goal", "label": "goal"}, {"id": "check", "label": "certificate", "parents": ["goal"]}]},
    "quantity": {"value": 9.81, "unit": "m/s^2"},
    "geometric_complex": {"vertices": [[0, 0], [1, 0], [0, 1]], "cells": [[0, 1, 2]]},
    "high_dimensional": {"points": [[0, 0], [1, 1], [2, 0]]},
    "ensemble": {"members": [[0, 1, 2], [0, 1.1, 1.9]]},
}


def make(kind: str):
    kwargs = {}
    if kind == "high_dimensional":
        kwargs = {
            "parameters": {"input_dimension": 7, "output_dimension": 2, "method": "PCA", "basis": [[1, 0], [0, 1]]},
            "information_loss": ["projection"],
            "information_loss_notes": ["7D coordinates projected into a declared 2D PCA basis."],
        }
    return mkp.create_projection(kind, PAYLOADS[kind], trust="numeric", **kwargs)


def test_catalog_covers_all_declared_projection_kinds():
    catalog = mkp.projection_catalog()
    assert set(PAYLOADS) == {row["kind"] for row in catalog}


def test_high_dimensional_projection_must_be_explicit():
    try:
        mkp.create_projection("high_dimensional", PAYLOADS["high_dimensional"],
                              parameters={"input_dimension": 7, "output_dimension": 2},
                              information_loss=["projection"])
    except ValueError as exc:
        assert "explicit method" in str(exc)
    else:
        raise AssertionError("implicit high-dimensional projection was accepted")


def test_every_projection_has_a_visualization_adapter():
    for kind in PAYLOADS:
        doc = viz.from_projection(make(kind))
        assert doc.blocks, kind
        assert doc.metadata["projection"]["kind"] == kind
        assert doc.trust == "numeric" or doc.trust == "unknown"


def test_every_projection_has_a_sonification_adapter():
    for kind in PAYLOADS:
        doc = son.projection_sonification(make(kind), options={"seconds_per_item": 0.001})
        assert doc.tracks, kind
        assert doc.metadata["projection"]["kind"] == kind
        assert doc.metadata["projection"]["acoustic_extraction"], kind
        assert doc.trust == "numeric"


def test_matrix_audio_records_flattening_order():
    doc = son.projection_sonification(make("matrix"))
    meta = doc.metadata["projection"]
    assert meta["acoustic_extraction"] == "matrix_row_major_scan"
    assert meta["acoustic_extraction_parameters"]["ordering"] == "row-major"


def test_projection_source_is_shared_between_visual_and_audio():
    proj = make("sequence")
    v = viz.from_projection(proj)
    a = son.projection_sonification(proj)
    source_id = proj.source_ref.source_id
    assert all(ds.source_ref.source_id == source_id for ds in v.datasets.values())
    assert all(track.source_ref.source_id == source_id for track in a.tracks)


def test_kernel_projection_facade_roundtrip():
    kernel = MathKernel()
    created = kernel.projection_create("matrix", {"matrix": [[1, 2], [3, 4]]}, trust="exact")
    assert created.ok
    pid = created.data["projection_id"]
    described = kernel.projection_describe(pid)
    assert described.ok and described.data["kind"] == "matrix"
    vr = kernel.viz_projection(pid)
    ar = kernel.sonify_projection(pid, options={"seconds_per_item": 0.001})
    assert vr.ok and ar.ok
    assert vr.data["projection_id"] == pid
    assert ar.data["projection_id"] == pid
    assert ar.data["acoustic_extraction"] == "matrix_row_major_scan"


def test_projection_cannot_upgrade_source_trust():
    from mathkernel_artifacts import SourceRef
    p = mkp.create_projection(
        "sequence", {"values": [1, 2]}, trust="exact",
        source_ref=SourceRef(source_id="src", kind="dataset", trust="numeric"),
    )
    assert p.trust == "numeric"
