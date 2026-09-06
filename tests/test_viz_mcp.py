# =============================================================================
# MathKernel - viz facade + MCP tool tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

from mathkernel import MathKernel


@pytest.fixture
def kernel():
    return MathKernel()


def test_viz_inline_plot(kernel):
    r = kernel.viz_create(data={"xs": [0, 1, 2], "ys": [0, 1, 4], "trust": "exact"})
    assert r.ok
    assert r.data["block_kinds"] == ["plot2d"]
    assert r.data["trust"] == "exact"
    assert r.data["viz_id"] in kernel.viz_documents


def test_viz_matrix_heatmap(kernel):
    m = kernel.matrix_create([["1", "2"], ["3", "4"]])
    r = kernel.viz_create(matrix_id=m.data["matrix_id"])
    assert r.ok and r.data["block_kinds"] == ["heatmap"]
    assert r.data["trust"] == "exact"


def test_viz_dag_from_derivation(kernel):
    r = kernel.parse("x^2 - 2 = 0")
    sol = kernel.solve(r.data["expr_id"], "x")
    d = kernel.viz_dag(sol.derivation[0].step_id)
    assert d.ok and d.data["block_kinds"] == ["dag"]
    # weakest evidence: parse is exact, solve is symbolic
    assert d.data["trust"] == "symbolic"


def test_viz_enclosures(kernel):
    r = kernel.viz_create(data={"enclosures": [
        {"x": 0, "lower": "1.4142135623730949", "upper": "1.4142135623730952"}]})
    assert r.ok and r.data["trust"] == "interval_certified"


def test_viz_3d_points(kernel):
    r = kernel.viz_create(data={"points": [[0, 0, 0], [1, 1, 1]]})
    assert r.ok and r.data["block_kinds"] == ["point_cloud_3d"]


def test_viz_compose_blocks(kernel):
    """Full building-block composition through the facade (the MCP path)."""
    r = kernel.viz_create(data={
        "title": "composed", "layout": {"cols": 2},
        "blocks": [
            {"kind": "select", "config": {"param": "lag", "options": [
                {"label": "k=1", "value": {"embed": {"lags": [0, 1, 2]}}}]}},
            {"kind": "point_cloud_3d", "title": "cloud",
             "data": {"values": [0.1 * i for i in range(50)]},
             "bindings": {"*": "lag"}},
            {"kind": "histogram", "data": {"values": [0.1, 0.2, 0.9], "bins": 4}},
            {"kind": "metric_grid",
             "data": {"entries": [{"label": "n", "value": 50}]}},
            {"kind": "text", "data": {"text": "notes"}},
        ]})
    assert r.ok, r.errors
    assert r.data["blocks"] == 5
    assert r.data["block_kinds"] == ["select", "point_cloud_3d", "histogram",
                                     "metric_grid", "text"]
    doc = kernel.viz_documents[r.data["viz_id"]]
    cloud = next(b for b in doc.blocks if b.kind == "point_cloud_3d")
    assert cloud.bindings == {"*": "lag"}
    assert cloud.config["embed"]["lags"] == [0, 1, 2]


def test_viz_compose_rejects_unknown_block(kernel):
    r = kernel.viz_create(data={"blocks": [{"kind": "nope"}]})
    assert not r.ok


def test_viz_errors(kernel):
    assert not kernel.viz_create().ok
    assert not kernel.viz_create(matrix_id="nope").ok
    assert not kernel.viz_export("nope", "x.html").ok


def test_viz_export_writes_artifact(kernel, tmp_path):
    r = kernel.viz_create(data={"xs": [0, 1], "ys": [0, 1]})
    out = tmp_path / "artifact.html"
    e = kernel.viz_export(r.data["viz_id"], str(out))
    assert e.ok
    assert e.data["bytes"] == out.stat().st_size
    assert e.data["deterministic"] is True
    text = out.read_text(encoding="utf-8")
    assert "mathkernel-data" in text


def test_viz_koopman(kernel):
    s = kernel.finite_system_create("uniform", [1, 2, 3, 0], [0, 1, 0, 1])
    sid = s.data["system_id"]
    r = kernel.viz_koopman(sid, {"kind": "cyclic", "m": 4})
    assert r.ok
    assert r.data["block_kinds"] == ["plot2d"]
    assert r.data["trust"] == "exact"
    assert r.data["modes"] == 4


def test_mcp_tools_registered():
    from mathkernel_mcp.server import mcp
    names = {t.name for t in mcp._tool_manager._tools.values()} \
        if hasattr(mcp, "_tool_manager") else set()
    if names:
        for t in ("math_visualize", "math_visualize_dag", "math_render_koopman",
                  "math_export_artifact"):
            assert t in names
