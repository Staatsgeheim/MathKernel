# =============================================================================
# MathKernel - regression tests for viz/sonify/projection defect fixes
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

from mathkernel import MathKernel


@pytest.fixture
def kernel():
    return MathKernel()


# -- interval key aliases (lo/hi vs lower/upper) ------------------------------

def test_interval_projection_lo_hi_viz(kernel):
    p = kernel.projection_create("intervals", {
        "intervals": [{"lo": "1.41", "hi": "1.42"}]}, trust="interval_certified")
    assert p.ok
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok and v.data["trust"] == "interval_certified"


def test_interval_projection_lo_hi_sonify(kernel):
    p = kernel.projection_create("intervals", {
        "intervals": [{"lo": "1.41", "hi": "1.42"},
                      {"lo": "1.73", "hi": "1.74"}]}, trust="interval_certified")
    s = kernel.sonify_projection(p.data["projection_id"])
    assert s.ok and s.data["tracks"] == 2


def test_interval_missing_bounds_typed_error(kernel):
    p = kernel.projection_create("intervals", {
        "intervals": [{"value": "1.41"}]})
    assert p.ok  # payload key 'intervals' is present; element keys fail later
    v = kernel.viz_projection(p.data["projection_id"])
    assert not v.ok
    assert "lower" in v.errors[0] and "upper" in v.errors[0]


# -- complex_field flat payloads ----------------------------------------------

def test_complex_field_flat_magnitude_phase_viz(kernel):
    p = kernel.projection_create("complex_field", {
        "magnitude": [1.0, 2.0, 3.0, 4.0],
        "phase": [0.0, 0.5, 1.0, 1.5]})
    assert p.ok
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok and "heatmap" in v.data["block_kinds"]


def test_complex_field_flat_real_imag_viz(kernel):
    p = kernel.projection_create("complex_field", {
        "real": [1.0, 0.0], "imag": [0.0, 1.0]})
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok and v.data["block_kinds"] == ["heatmap", "heatmap"]


# -- high_dimensional contract -------------------------------------------------

def test_high_dimensional_infers_input_dimension(kernel):
    r = kernel.projection_create("high_dimensional", {
        "points": [[1, 2, 3, 4], [5, 6, 7, 8]]})
    assert not r.ok
    assert "output_dimension" in r.errors[0] or "method" in r.errors[0]


def test_high_dimensional_complete_contract_accepted(kernel):
    r = kernel.projection_create("high_dimensional", {
        "points": [[1, 2, 3], [4, 5, 6]]},
        parameters={"input_dimension": 4, "output_dimension": 2,
                    "method": "pca"},
        information_loss=["projection"])
    assert r.ok


# -- trust preservation --------------------------------------------------------

def test_graph_projection_viz_preserves_exact_trust(kernel):
    p = kernel.projection_create("graph", {
        "nodes": ["a", "b"], "edges": [["a", "b"]]}, trust="exact")
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok and v.data["trust"] == "exact"


def test_viz_export_does_not_upgrade_trust(kernel, tmp_path):
    v = kernel.viz_create(data={"xs": [0, 1], "ys": [0.5, 1.5]})
    assert v.ok and v.data["trust"] == "numeric"
    out = kernel.viz_export(v.data["viz_id"], str(tmp_path / "a.html"))
    assert out.ok and out.trust.value == "numeric"


def test_research_artifact_export_does_not_upgrade_trust(kernel, tmp_path):
    v = kernel.viz_create(data={"xs": [0, 1], "ys": [0.5, 1.5]})
    a = kernel.research_artifact_create(viz_ids=[v.data["viz_id"]])
    assert a.ok and a.data["trust"] == "numeric"
    out = kernel.research_artifact_export(a.data["artifact_id"],
                                          str(tmp_path / "a.html"))
    assert out.ok and out.trust.value == "numeric"


# -- block-kind dispatch --------------------------------------------------------

def test_vector_field_2d_uses_arrow_renderer(kernel):
    p = kernel.projection_create("vector_field", {
        "origins": [[0, 0], [1, 1]], "vectors": [[1, 0], [0, 1]]})
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok and v.data["block_kinds"] == ["vector_field_3d"]


def test_mesh_3d_renders_edges(kernel):
    p = kernel.projection_create("mesh", {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "cells": [[0, 1, 2]]})
    v = kernel.viz_projection(p.data["projection_id"])
    assert v.ok
    kinds = v.data["block_kinds"]
    assert "point_cloud_3d" in kinds and "trajectory_3d" in kinds
    doc = kernel.viz_documents[v.data["viz_id"]]
    edge_block = next(b for b in doc.blocks if b.kind == "trajectory_3d")
    assert len(edge_block.series) == 3  # triangle has 3 edges in one block


# -- empty sonify rejection ------------------------------------------------------

def test_sonify_empty_data_rejected(kernel):
    r = kernel.sonify_create([])
    assert not r.ok
    assert "at least one" in r.errors[0]


def test_sonify_compare_empty_rejected(kernel):
    r = kernel.sonify_compare([], [])
    assert not r.ok


# -- capability discovery ---------------------------------------------------------

@pytest.mark.parametrize("domain", ["visualization", "sonification", "projection"])
def test_capability_query_presentation_domains(kernel, domain):
    r = kernel.capability_query(domain=domain)
    assert r["total_count"] > 0
    assert all(c["domain"] == domain for c in r["capabilities"])


# -- dashboard text block contract ------------------------------------------------

def test_compose_text_block_accepts_plain_string(kernel):
    r = kernel.viz_create(data={"blocks": [
        {"kind": "text", "data": "plain caption"}]})
    assert r.ok and r.data["block_kinds"] == ["text"]


def test_compose_non_dict_data_typed_error(kernel):
    r = kernel.viz_create(data={"blocks": [
        {"kind": "plot2d", "data": "not-a-dict"}]})
    assert not r.ok
    assert "requires 'data' as an object" in r.errors[0]
