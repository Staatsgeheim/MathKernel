# =============================================================================
# MathKernel - multimodal research artifact tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Tests for mathkernel_multimodal: artifact assembly, cross-modal
synchronization derivation, the unified portable HTML exporter, and the
kernel/MCP facade."""
import json
import re

import pytest

import mathkernel_multimodal as mkm
import mathkernel_sonify as son
import mathkernel_viz as viz
from mathkernel_artifacts import SourceRef, SynchronizationLink


def _shared_ref(source_id="mathresult:sha256:deadbeef", trust="numeric"):
    return SourceRef(source_id=source_id, kind="math_result", trust=trust)


def _viz_doc(ref=None, title="viz"):
    doc = viz.dashboard(title, cols=2)
    viz.add_plot(doc, [0, 1, 2], [0, 1, 0], label="obs", trust="numeric")
    if ref is not None:
        for s in doc.series.values():
            s.source_ref = ref
    return doc


def _son_doc(ref_id=None, values=(1.0, 2.0, 4.0)):
    doc = son.scan_sonification(list(values), seconds_per_item=0.25,
                                sample_rate=8000)
    if ref_id is not None:
        for tr in doc.tracks:
            for ev in tr.events:
                ev.source_ref = ref_id
    return doc


# ---------------------------------------------------------------- assembly

def test_build_artifact_merges_sources_and_weakest_trust():
    ref = _shared_ref()
    art = mkm.build_artifact(title="t", visualizations=[_viz_doc(ref)],
                             sonifications=[_son_doc(ref.source_id)],
                             mathkernel_version="1.2.0")
    assert art.artifact_schema == "mathkernel-artifact/1.0"
    assert ref.source_id in art.sources
    assert art.trust == "numeric"
    assert len(art.visualizations) == 1
    assert len(art.sonifications) == 1
    assert art.reproducibility.mathkernel_version == "1.2.0"
    assert art.reproducibility.deterministic is True
    assert art.reproducibility.created_utc is None


def test_build_artifact_merges_linked_visualization_evidence():
    result = {
        "trust": "exact",
        "engine": "numeric",
        "data": {"points": [[0, 0], [1, 1]]},
        "claim_evidence": {
            "value": {
                "computation": [{
                    "engine": "numeric",
                    "method": "sample",
                    "arithmetic": "floating_point",
                    "trust": "numeric",
                }],
            },
        },
    }
    doc = viz.visualize(result)
    art = mkm.build_artifact(visualizations=[doc])
    assert art.trust == "numeric"
    assert art.evidence_bundle.conservative_trust() == "numeric"
    assert "visualization:0:value" in art.claim_evidence


def test_build_artifact_weakest_trust_drops_with_heuristic_member():
    doc = _viz_doc()
    for s in doc.series.values():
        s.trust = "heuristic"
    doc.trust = doc.weakest_trust()
    art = mkm.build_artifact(visualizations=[doc], sonifications=[_son_doc()])
    assert art.trust == "heuristic"


def test_derive_synchronization_links_shared_sources():
    ref = _shared_ref()
    viz_docs = [_viz_doc(ref)]
    son_docs = [_son_doc(ref.source_id)]
    links = mkm.derive_synchronization(viz_docs, son_docs)
    assert len(links) == 3                      # one per scan event
    for link, (lo, hi) in zip(links, [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75)]):
        assert link.source_ref == ref.source_id
        assert link.visual_ref == "v0:b1"
        assert link.sonification_ref == "s0"
        assert link.time_range == [pytest.approx(lo), pytest.approx(hi)]


def test_derive_synchronization_ignores_unshared_sources():
    art = mkm.build_artifact(visualizations=[_viz_doc(_shared_ref("a"))],
                             sonifications=[_son_doc("b")])
    assert art.synchronization == []


def test_explicit_links_preserved_and_numbered_before_derived():
    ref = _shared_ref()
    explicit = SynchronizationLink(link_id="sync-1", source_ref="manual",
                                   visual_ref="v0:b1", sonification_ref="s0")
    art = mkm.build_artifact(visualizations=[_viz_doc(ref)],
                             sonifications=[_son_doc(ref.source_id)],
                             synchronization=[explicit])
    assert art.synchronization[0].link_id == "sync-1"
    assert art.synchronization[0].source_ref == "manual"
    assert {l.link_id for l in art.synchronization[1:]} == {
        "sync-2", "sync-3", "sync-4"}


def test_build_artifact_is_deterministic():
    ref = _shared_ref()
    kw = dict(title="t", visualizations=[_viz_doc(ref)],
              sonifications=[_son_doc(ref.source_id)])
    a = mkm.build_artifact(**kw)
    b = mkm.build_artifact(**kw)
    assert a.model_dump_json() == b.model_dump_json()
    assert a.integrity == b.integrity


def test_build_artifact_accepts_dict_documents():
    ref = _shared_ref()
    art = mkm.build_artifact(
        visualizations=[_viz_doc(ref).model_dump(mode="json")],
        sonifications=[_son_doc(ref.source_id).model_dump(mode="json")])
    assert len(art.visualizations) == 1
    assert len(art.synchronization) == 3


# ---------------------------------------------------------------- exporter

def _artifact(with_3d=False):
    ref = _shared_ref()
    doc = viz.dashboard("viz", cols=2)
    if with_3d:
        viz.add_point_cloud(doc, [[0, 0, 0], [1, 1, 1]], trust="numeric")
    else:
        viz.add_plot(doc, [0, 1], [0, 1], trust="numeric")
    for s in doc.series.values():
        s.source_ref = ref
    return mkm.build_artifact(title="Research", visualizations=[doc],
                              sonifications=[_son_doc(ref.source_id)])


def test_export_html_single_file_portable(tmp_path):
    info = mkm.export_html(_artifact(), tmp_path / "a.html")
    text = (tmp_path / "a.html").read_text(encoding="utf-8")
    assert info["artifact_schema"] == "mathkernel-artifact/1.0"
    assert info["visualizations"] == 1 and info["sonifications"] == 1
    assert info["synchronization_links"] == 3
    assert info["sha256"]
    # payload + integrity attribute
    m = re.search(r'<script id="mathkernel-artifact" type="application/json" '
                  r'data-sha256="([0-9a-f]{64})">', text)
    assert m
    # CSP + offline: no external references of any kind
    assert "Content-Security-Policy" in text
    assert "https://" not in text and "http://" not in text
    assert 'src="' not in text  # no external scripts; everything inline
    # runtimes embedded
    assert "MathKernelViz" in text          # mountable viz viewer
    assert "applyTransform" in text         # multimodal audio/sync runtime
    # no 3D content -> no vendored three.js (viewer.js references the global
    # by name; only the vendored runtime assigns it)
    assert "window.THREE=" not in text


def test_export_html_embeds_threejs_only_for_3d(tmp_path):
    info = mkm.export_html(_artifact(with_3d=True), tmp_path / "b.html")
    text = (tmp_path / "b.html").read_text(encoding="utf-8")
    assert "window.THREE=" in text          # vendored runtime embedded
    assert info["bytes"] > 600_000          # vendored runtime included


def test_export_html_deterministic(tmp_path):
    a = mkm.render_html(_artifact())
    b = mkm.render_html(_artifact())
    assert a == b


def test_export_html_escapes_script_terminator(tmp_path):
    art = _artifact()
    art.summary = "x </script><script>alert(1)</script>"
    text = mkm.render_html(art)
    assert "</script><script>alert(1)" not in text
    assert "<\\/script>" in text


def test_export_html_rejects_non_portable_mode():
    with pytest.raises(ValueError):
        mkm.render_html(_artifact(), mode="web")


def test_export_html_created_utc_only_when_supplied():
    assert '"created_utc":null' in mkm.render_html(_artifact())
    assert "2026-09-02" in mkm.render_html(_artifact(),
                                           created_utc="2026-09-02T00:00:00Z")


# ---------------------------------------------------------------- kernel facade

def test_kernel_research_artifact_roundtrip(tmp_path):
    from mathkernel import MathKernel
    k = MathKernel()
    v = k.viz_create(data={"xs": [0, 1, 2], "ys": [0, 1, 0], "trust": "numeric"})
    assert v.ok, v.errors
    s = k.sonify_create([1.0, 2.0, 4.0], mode="scan",
                        options={"sample_rate": 8000})
    assert s.ok, s.errors
    r = k.research_artifact_create(
        title="rt", viz_ids=[v.data["viz_id"]],
        sonification_ids=[s.data["sonification_id"]])
    assert r.ok, r.errors
    assert r.data["visualizations"] == 1
    assert r.data["sonifications"] == 1
    assert r.data["artifact_schema"] == "mathkernel-artifact/1.0"
    out = k.research_artifact_export(r.data["artifact_id"],
                                     str(tmp_path / "rt.html"))
    assert out.ok, out.errors
    text = (tmp_path / "rt.html").read_text(encoding="utf-8")
    assert "mathkernel-artifact" in text


def test_kernel_research_artifact_validates_ids(tmp_path):
    from mathkernel import MathKernel
    k = MathKernel()
    assert not k.research_artifact_create().ok               # nothing referenced
    assert not k.research_artifact_create(viz_ids=["nope"]).ok
    assert not k.research_artifact_create(sonification_ids=["nope"]).ok
    assert not k.research_artifact_export("nope",
                                          str(tmp_path / "x.html")).ok


def test_mcp_tools_registered():
    from mathkernel_mcp.server import mcp
    names = {t.name for t in mcp._tool_manager._tools.values()} \
        if hasattr(mcp, "_tool_manager") else set()
    if names:
        for t in ("math_research_artifact_create",
                  "math_export_research_artifact"):
            assert t in names
