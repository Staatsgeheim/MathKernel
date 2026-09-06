# =============================================================================
# MathKernel - viz building-block composition tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

import mathkernel_viz as viz
from mathkernel_viz.document import BLOCK_KINDS


def test_all_block_kinds_compose():
    doc = viz.dashboard("everything", cols=3)
    viz.add_point_cloud(doc, [[0, 0, 0], [1, 1, 1]], trust="numeric")
    viz.add_trajectory_3d(doc, [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    viz.add_surface_3d(doc, [[0, 1], [1, 0]])
    viz.add_vector_field_3d(doc, [[0, 0, 0]], [[1, 0, 0]])
    viz.add_plot(doc, [0, 1], [0, 1])
    viz.add_histogram(doc, [0.1, 0.2, 0.2, 0.9], bins=4)
    viz.add_heatmap(doc, [[1, 2], [3, 4]])
    viz.add_dag(doc, nodes=[{"id": "a", "label": "root"}])
    viz.add_metric_grid(doc, entries=[{"label": "n", "value": 4}])
    viz.add_data_table(doc, rows=[["a", "1"], ["b", "2"]])
    viz.add_text(doc, "hello")
    viz.add_select(doc, "p", [{"label": "x", "value": {"bins": 8}}])
    kinds = [b.kind for b in doc.blocks]
    assert set(kinds) == set(BLOCK_KINDS)
    assert doc.uses_3d()


def test_unknown_block_kind_rejected():
    doc = viz.dashboard("x")
    with pytest.raises(ValueError):
        doc.add_block("pie_chart")


def test_block_trust_is_weakest_of_its_data():
    doc = viz.dashboard("t")
    b = viz.add_plot(doc, [0, 1], [0, 1], trust="exact")
    assert b.trust == "exact"
    b2 = viz.add_plot(doc, [0, 1], [0, 0.5], trust="numeric")
    assert b2.trust == "numeric"
    assert doc.trust == "numeric"


def test_overlay_series_on_shared_block():
    doc = viz.dashboard("overlay")
    b = viz.add_plot(doc, [0, 1], [0, 1], label="a", trust="exact")
    viz.add_plot(doc, [0, 1], [1, 0], label="b", trust="numeric",
                 block_id=b.block_id)
    assert len(doc.blocks) == 1
    assert len(b.series) == 2
    assert b.trust == "numeric"
    # an unknown name creates a new named panel rather than failing
    viz.add_plot(doc, [0], [0], block_id="second-panel")
    assert len(doc.blocks) == 2
    assert doc.blocks[1].block_id == "second-panel"


def test_select_and_bindings_in_payload():
    doc = viz.dashboard("linked")
    doc.add_dataset(viz.encode_dataset("seq", [0.1 * i for i in range(100)]))
    viz.add_select(doc, "lag", [
        {"label": "k=1", "value": {"embed": {"lags": [0, 1, 2]}}},
        {"label": "k=5", "value": {"embed": {"lags": [0, 5, 10]}}},
    ], default=1)
    doc.add_block("point_cloud_3d",
                  config={"embed": {"dataset": "seq", "lags": [0, 5, 10]}},
                  datasets=["seq"], bindings={"*": "lag"})
    payload = doc.model_dump(mode="json")
    sel = next(b for b in payload["blocks"] if b["kind"] == "select")
    assert sel["config"]["param"] == "lag"
    assert sel["config"]["options"][1]["value"]["embed"]["lags"] == [0, 5, 10]
    cloud = next(b for b in payload["blocks"] if b["kind"] == "point_cloud_3d")
    assert cloud["bindings"] == {"*": "lag"}
    assert cloud["config"]["embed"]["dataset"] == "seq"


def test_histogram_reuses_existing_dataset():
    doc = viz.dashboard("h")
    doc.add_dataset(viz.encode_dataset("v", [1.0, 2.0, 3.0]))
    b = viz.add_histogram(doc, "v", bins=8)
    assert b.config["dataset"] == "v"
    assert b.datasets == ["v"]
    with pytest.raises(ValueError):
        viz.add_histogram(doc, "missing")


def test_metric_grid_dataset_stats():
    doc = viz.dashboard("m")
    doc.add_dataset(viz.encode_dataset("v", [1.0, 2.0, 3.0]))
    b = viz.add_metric_grid(doc, dataset="v", stats=["count", "mean"])
    assert b.config["stats"] == ["count", "mean"]
    assert b.datasets == ["v"]


def test_data_table_column_dataset_reference():
    doc = viz.dashboard("t")
    doc.add_dataset(viz.encode_dataset("v", [1.0, 2.0]))
    b = viz.add_data_table(doc, columns=[{"label": "v", "dataset": "v"}])
    assert b.datasets == ["v"]
    with pytest.raises(ValueError):
        viz.add_data_table(doc, columns=[{"label": "x", "dataset": "nope"}])


def test_span_is_clamped_positive():
    doc = viz.dashboard("s")
    b = viz.add_text(doc, "x", span=0)
    assert b.span == 1
