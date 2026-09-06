# =============================================================================
# MathKernel - viz renderer tests (SVG, HTML artifact, auto-selection)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import json
import re

import pytest

import mathkernel_viz as viz


def _plot():
    return viz.plot_document([0, 1, 2, 3], [0, 1, 4, 9], title="squares",
                             trust="exact")


def test_svg_plot_smoke():
    svg = viz.render_svg(_plot())
    assert svg.startswith("<svg")
    assert "trust: exact" in svg
    assert "squares" in svg


def test_svg_heatmap_and_dag():
    hm = viz.render_svg(viz.heatmap_document([[1, 2], [3, 4]], title="M"))
    assert "2 x 2" in hm
    dag = viz.render_svg(viz.dag_document(
        [{"step_id": "a", "operation": "parse", "trust": "exact"},
         {"step_id": "b", "operation": "solve", "parents": ["a"],
          "trust": "symbolic"}]))
    assert "parse" in dag and "solve" in dag


def test_svg_histogram_block():
    doc = viz.dashboard("h", cols=1)
    viz.add_histogram(doc, [0.1, 0.2, 0.2, 0.9], bins=4)
    svg = viz.render_svg(doc)
    assert "<rect" in svg


def test_svg_escapes_markup():
    doc = viz.plot_document([1], [2], title='<script>alert(1)</script>')
    svg = viz.render_svg(doc)
    assert "<script>alert" not in svg
    assert "&lt;script&gt;" in svg


def test_html_deterministic():
    a = viz.render_html(_plot(), mathkernel_version="1.2.0")
    b = viz.render_html(_plot(), mathkernel_version="1.2.0")
    assert a == b


def test_html_structure_and_security():
    h = viz.render_html(_plot(), mathkernel_version="1.2.0")
    assert 'id="mathkernel-data"' in h
    assert "Content-Security-Policy" in h
    assert "connect-src 'none'" in h
    assert "eval(" not in h
    # no network references (the SVG namespace is a spec identifier, not a fetch)
    refs = set(re.findall(r"https?://[^\s\"'<>]+", h))
    assert refs <= {"http://www.w3.org/2000/svg"}
    # payload hash attribute matches the exact embedded payload text
    m = re.search(r'data-sha256="([0-9a-f]{64})"', h)
    assert m
    payload = h.split('data-sha256="' + m.group(1) + '">', 1)[1].split("</script>", 1)[0]
    from mathkernel_viz.manifest import sha256_text
    assert sha256_text(payload) == m.group(1)
    doc = json.loads(payload.replace("<\\/", "</"))
    assert doc["artifact_schema"] == "mathkernel-viz/2.0"
    assert doc["manifest"]["deterministic"] is True
    assert doc["blocks"][0]["kind"] == "plot2d"


def test_html_xss_labels_inert():
    doc = viz.plot_document([1], [2], title='<script>alert(1)</script>',
                            series_label='<img src=x onerror=alert(1)>')
    h = viz.render_html(doc)
    head = h.split('<script id="mathkernel-data"', 1)[0]
    assert "<script>alert" not in head
    assert "<img src=x" not in head
    # inside the JSON payload the markup is inert data, escaped against
    # early tag termination
    assert "<\\/script>" in h or "alert" not in head


def test_html_portable_inlines_threejs_for_3d():
    doc = viz.points3d_document([[0, 0, 0], [1, 1, 1]], title="cloud")
    h = viz.render_html(doc)
    assert "window.THREE" in h
    assert "src=" not in h.split('<script id="mathkernel-data"')[1].split(">", 1)[0]
    h2d = viz.render_html(_plot())
    assert "window.THREE" not in h2d


def test_html_web_mode_references_assets():
    doc = viz.points3d_document([[0, 0, 0]], title="c")
    h = viz.render_html(doc, mode="web", embed_runtime=False)
    assert 'src="assets/viewer.js"' in h
    assert 'src="assets/three.global.js"' in h


def test_html_rejects_bad_mode():
    with pytest.raises(ValueError):
        viz.render_html(_plot(), mode="weird")


def test_auto_selection():
    assert viz.select_view({"data": {}, "derivation": [{"x": 1}]}) == "dag"
    big = [[0.0] * 8 for _ in range(8)]
    assert viz.select_view({"data": {"matrix": big}, "derivation": []}) == "heatmap"
    assert viz.select_view({"data": {"points": [[1, 2]]}, "derivation": []}) == "plot2d"
    assert viz.select_view({"data": {"points": [[1, 2, 3]]},
                            "derivation": []}) == "point_cloud_3d"
    assert viz.select_view({"data": {"grid": [[1]]}, "derivation": []}) == "surface_3d"
    assert viz.select_renderer("point_cloud_3d") == "threejs"
    assert viz.select_renderer("dag") == "svg"


def test_export_html_writes_file(tmp_path):
    info = viz.export_html(_plot(), tmp_path / "a.html", mathkernel_version="1.2.0")
    assert info["bytes"] > 1000
    assert info["artifact_schema"] == "mathkernel-viz/2.0"
    text = (tmp_path / "a.html").read_text(encoding="utf-8")
    assert "mathkernel-data" in text


def test_export_svg_writes_file(tmp_path):
    info = viz.export_svg(_plot(), tmp_path / "a.svg")
    assert info["bytes"] > 500


def test_scene3d_svg_fallback_message():
    doc = viz.points3d_document([[0, 0, 0]], title="c")
    svg = viz.render_svg(doc)
    assert "interactive" in svg


def _dashboard():
    doc = viz.dashboard("dash", cols=2, trust="numeric")
    viz.add_point_cloud(doc, [[0, 0, 0], [1, 1, 1]], label="cloud",
                        trust="numeric", color="#e74c3c", title="3D")
    viz.add_plot(doc, [0, 1, 2], [10, 100, 1000], label="bars",
                 x_label="order", y_label="count", marker="bar", log_y=True,
                 trust="numeric", color="#3498db", title="Harmonic Content")
    return doc


def test_dashboard_svg_smoke():
    svg = viz.render_svg(_dashboard())
    assert "Harmonic Content" in svg
    assert "interactive" in svg       # static fallback for the WebGL block
    assert svg.count("<svg") >= 3     # frame + one nested svg per block


def test_dashboard_html_bundles_threejs():
    h = viz.render_html(_dashboard())
    assert "window.THREE" in h        # 3D block forces the runtime into the artifact


def test_bar_log_scale_svg():
    doc = viz.dashboard("bars", cols=1)
    viz.add_plot(doc, [0, 1, 2], [1, 10, 100], marker="bar", log_y=True)
    svg = viz.render_svg(doc)
    assert "<rect" in svg and "1e" in svg


def test_series_style_color_honored():
    doc = viz.dashboard("c", cols=1)
    viz.add_plot(doc, [0, 1], [0, 1], color="#e74c3c", marker="points")
    svg = viz.render_svg(doc)
    assert "#e74c3c" in svg


def test_static_svg_embedded_only_when_all_blocks_svg_capable():
    h2d = viz.render_html(_plot())
    assert "mathkernel-static-svg" in h2d
    h3d = viz.render_html(_dashboard())
    assert 'id="mathkernel-static-svg"' not in h3d
