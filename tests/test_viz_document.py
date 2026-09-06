# =============================================================================
# MathKernel - viz document/dataset/manifest tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

import mathkernel_viz as viz
from mathkernel_viz.document import (ARTIFACT_SCHEMA, VisualizationDocument)


def test_document_roundtrip():
    doc = viz.plot_document([0, 1, 2], [0, 1, 4], title="t", trust="exact")
    text = doc.model_dump_json()
    back = VisualizationDocument.model_validate_json(text)
    assert back == doc
    assert back.artifact_schema == ARTIFACT_SCHEMA
    assert back.schema_version == "2.0"
    assert ARTIFACT_SCHEMA == "mathkernel-viz/2.0"


def test_document_defaults_to_empty_composition():
    doc = VisualizationDocument()
    assert doc.blocks == []
    assert doc.layout.cols == 2
    assert doc.trust == "unknown"


def test_weakest_trust_meet():
    doc = viz.compare_document(([0, 1], [0, 1]), ([0, 1], [0, 0.9]),
                               prediction_trust="exact",
                               observation_trust="numeric")
    assert doc.trust == "numeric"
    roles = {s.role for s in doc.series.values()}
    assert roles == {"prediction", "observation"}
    # both series share one plot block
    assert len(doc.blocks) == 1
    assert doc.blocks[0].kind == "plot2d"
    assert len(doc.blocks[0].series) == 2


def test_weakest_trust_excludes_self():
    doc = viz.dag_document([{"step_id": "a", "operation": "op", "trust": "exact"}])
    assert doc.trust == "exact"


def test_dataset_tiers():
    small = viz.encode_dataset("s", [1, 2, 3])
    mid = viz.encode_dataset("m", [float(i) for i in range(5000)])
    big = viz.encode_dataset("b", list(range(200000)))
    assert small.encoding == "inline"
    assert mid.encoding == "zlib+base64"
    assert big.encoding == "zlib+base64+chunked"
    assert mid.kind == "f64"
    assert big.kind == "i64"


def test_dataset_roundtrip_and_integrity():
    for vals in ([1, 2, 3], [0.5 * i for i in range(5000)], list(range(200000))):
        ds = viz.encode_dataset("d", vals)
        out = viz.decode_dataset(ds)
        assert len(out) == len(vals)
        assert out[0] == vals[0] and out[-1] == vals[-1]
        assert viz.verify_dataset(ds)
        ds_corrupt = ds.model_copy(update={"sha256": "0" * 64})
        assert not viz.verify_dataset(ds_corrupt)


def test_dataset_rejects_non_numeric():
    with pytest.raises(TypeError):
        viz.encode_dataset("s", ["a", "b"])


def test_dataset_size_limit():
    from mathkernel_viz import datasets
    with pytest.raises(ValueError):
        viz.encode_dataset("huge", [0.0] * (datasets.MAX_DATASET_BYTES // 8 + 1))


def test_manifest_deterministic_excludes_timestamp():
    doc = viz.plot_document([0, 1], [0, 1])
    m = viz.build_manifest(doc, renderer="svg", mathkernel_version="1.2.0")
    assert m["deterministic"] is True
    assert "created_utc" not in m
    m2 = viz.build_manifest(doc, renderer="svg", mathkernel_version="1.2.0",
                            deterministic=False, created_utc="2026-09-02T00:00:00Z")
    assert m2["created_utc"] == "2026-09-02T00:00:00Z"


def test_manifest_hashes_stable():
    doc = viz.plot_document([0, 1], [0, 1])
    a = viz.build_manifest(doc, renderer="svg", mathkernel_version="1.2.0")
    b = viz.build_manifest(doc, renderer="svg", mathkernel_version="1.2.0")
    assert a == b
    assert len(a["dataset_hashes"]) == 2


def test_interval_enclosure_preserved():
    doc = viz.interval_plot_document(
        [{"x": 0, "lower": "1.4142135623730949", "upper": "1.4142135623730952",
          "label": "sqrt(2)"}])
    s = next(iter(doc.series.values()))
    assert s.trust == "interval_certified"
    assert s.enclosure["lower"] == "1.4142135623730949"
    assert doc.trust == "interval_certified"
