# =============================================================================
# MathKernel Multimodal - unified portable research artifact HTML exporter
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Export a ``MathKernelArtifact`` as one self-contained interactive HTML file.

The artifact embeds the full evidence-carrying payload (result, sources,
evidence, transformations, annotations, synchronization, reproducibility,
integrity) plus every visualization and sonification document.  Portable mode
inlines the vendored Three.js runtime (only when a 3D block is present), the
``mathkernel_viz`` block viewer and the multimodal audio/sync runtime, so the
file needs no server, network, Python or MathKernel install and works from
``file://``.

Security: the payload travels in a non-executable JSON script tag with a
SHA-256 tag attribute; the runtime uses textContent only; no eval; strict CSP;
no network requests.  MathIR and labels are data, never code.
"""
from __future__ import annotations

import hashlib
import html as _html
import json
from importlib import resources
from pathlib import Path

from mathkernel_artifacts import MathKernelArtifact
from mathkernel_viz.document import BLOCK_3D_KINDS

MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "connect-src 'none'; font-src 'none'; img-src 'none'; media-src 'none'")

_CSS = """
:root{color-scheme:dark}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b0e11;color:#dbe8f3}
.mkm-head{padding:14px 20px;background:#10161b;border-bottom:1px solid #243039;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.mkm-head h1{font-size:17px;margin:0;font-weight:620}
.mkm-badge{font:11px ui-monospace,monospace;padding:3px 8px;border:1px solid #23415a;border-radius:999px;color:#66bfff;background:#10202c}
.mkm-main{display:grid;grid-template-columns:minmax(0,1fr) 360px;min-height:calc(100vh - 52px)}
.mkm-content{padding:14px;min-width:0}
.mkm-viz,.mkm-audio{background:#11171c;border:1px solid #1d2831;border-radius:8px;margin-bottom:14px;padding-bottom:10px}
.mkm-sec-head{display:flex;justify-content:space-between;align-items:baseline;padding:10px 14px;border-bottom:1px solid #1d2831}
.mkm-sec-head h2{margin:0;font-size:13px;font-weight:650;color:#a9c0cd;text-transform:uppercase;letter-spacing:.07em}
.mkm-viz .mkv-view{padding:10px}
.mkm-audio{padding:0 0 10px}
.mkm-warn{margin:8px 14px 0;color:#efb34c;font-size:11px}
.mkm-transport{display:flex;gap:8px;align-items:center;padding:10px 14px;flex-wrap:wrap}
.mkm-transport button,.mkm-side button{padding:5px 12px;font-size:11px;border:1px solid #293943;border-radius:5px;background:#172027;color:#c6d5df;cursor:pointer}
.mkm-transport button:hover,.mkm-side button:hover{background:#1b252c}
.mkm-seek{flex:1;min-width:120px;accent-color:#35a9ef}
.mkm-time{font:11px ui-monospace,monospace;color:#8ca1ae}
.mkm-note{flex:1;min-width:140px;padding:5px 8px;font:11px ui-monospace,monospace;color:#e2edf4;border:1px solid #293943;border-radius:5px;background:#0c1216}
.mkm-tracks{display:flex;flex-wrap:wrap;gap:10px;padding:0 14px}
.mkm-track{display:flex;gap:5px;align-items:center;font:11px ui-monospace,monospace;color:#8ca1ae}
.mkm-track-name{color:#c6d5df}
.mkm-side{border-left:1px solid #1d2831;background:#11171c;padding:12px;overflow:auto;font-size:12px}
.mkm-tabs button{border:1px solid #293943;background:#172027;color:#9cb0bd;padding:4px 9px;font-size:11px;cursor:pointer;border-radius:5px;margin:0 3px 6px 0}
.mkm-tabs button.mkm-active{background:#176795;color:#eaf7ff;border-color:#296a94}
.mkm-pane h3{font-size:13px;margin:4px 0}
.mkm-pane h4{font-size:11px;margin:12px 0 4px;color:#8ca1ae;text-transform:uppercase;letter-spacing:.07em}
.mkm-pane p{margin:4px 0;color:#b8c8d3;word-break:break-word}
.mkm-pane strong{color:#8ca1ae;font-weight:600}
.mkm-pre{white-space:pre-wrap;background:#0c1216;border:1px solid #1d2831;border-radius:5px;padding:8px;font:10px ui-monospace,monospace;max-height:300px;overflow:auto;color:#a9c0cd}
.mkm-hash{font:10px ui-monospace,monospace;word-break:break-all}
.mkm-live{font:11px ui-monospace,monospace;color:#5ad4d4}
.mkm-candidate{color:#efb34c}
.mkm-error{padding:18px;background:#2a1414;border:1px solid #5c2626;border-radius:6px;color:#e8a1a1;font-size:13px;margin:8px}
.mkv-sync-hl{outline:2px solid #5ad4d4;outline-offset:2px;box-shadow:0 0 18px #5ad4d455}
.mkv-syncable .mkv-block-head{cursor:pointer}
.mkv-syncable .mkv-block-head:hover{background:#172027}
/* viz block chrome (the block viewer styles its own internals) */
.mkv-grid{display:grid;gap:12px}
.mkv-block{background:#0c1216;border:1px solid #1d2831;border-radius:8px;min-width:0;overflow:hidden}
.mkv-block-head{display:flex;justify-content:space-between;align-items:baseline;padding:9px 12px;border-bottom:1px solid #1d2831}
.mkv-block-head h3{margin:0;font-size:12px;font-weight:650;color:#a9c0cd;text-transform:uppercase;letter-spacing:.07em}
.mkv-block-trust{font:10px ui-monospace,monospace}
.mkv-block-body{padding:8px}
.mkv-block-canvas{width:100%;display:block;background:#0b1014;border:1px solid #1d2831;border-radius:5px}
.mkv-3d{position:relative;width:100%}
.mkv-3d canvas{display:block;width:100%;height:100%}
.mkv-hint{position:absolute;left:12px;top:10px;color:#78909e;font-size:11px;pointer-events:none;text-shadow:0 2px 8px #000}
.mkv-metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;padding:4px}
.mkv-metric{padding:9px;background:#0c1216;border:1px solid #26353e;border-radius:5px}
.mkv-metric span{display:block;color:#76909e;font-size:10px;text-transform:uppercase;letter-spacing:.08em}
.mkv-metric strong{font:13px ui-monospace,monospace;color:#d7e9f4}
.mkv-metric em{display:block;color:#5c707c;font-size:10px;font-style:normal;margin-top:2px}
.mkv-table{width:100%;border-collapse:collapse;font:11px ui-monospace,monospace}
.mkv-table th,.mkv-table td{text-align:right;padding:5px 7px;border-bottom:1px solid #1d2831;color:#c6d5df}
.mkv-table th{color:#8ca1ae;text-transform:uppercase;font-size:10px;letter-spacing:.07em}
.mkv-text{padding:6px 10px;font-size:12px;line-height:1.55;color:#b8c8d3}
.mkv-text p{margin:0 0 8px}
.mkv-field{display:grid;gap:6px;padding:6px 8px}
.mkv-field label{font-size:10px;color:#8ca1ae;text-transform:uppercase;letter-spacing:.08em;font-weight:650}
.mkv-field select{width:100%;padding:8px 10px;color:#e2edf4;border:1px solid #293943;border-radius:6px;background:#172027;font:inherit;font-size:12px}
.mkv-tip{display:none;position:fixed;z-index:50;background:#0d1317;border:1px solid #2b3d49;color:#dbe8f3;font:11px ui-monospace,monospace;padding:8px 10px;border-radius:6px;pointer-events:none;white-space:pre;max-width:420px;box-shadow:0 6px 24px #000a}
.mkv-error{padding:18px;background:#2a1414;border:1px solid #5c2626;border-radius:6px;color:#e8a1a1;font-size:13px;margin:8px}
@media(max-width:900px){.mkm-main{grid-template-columns:1fr}.mkm-side{border-left:0;border-top:1px solid #1d2831}}
"""

_TABS = [
    ("mkm-tab-result", "Result"),
    ("mkm-tab-evidence", "Evidence"),
    ("mkm-tab-provenance", "Provenance"),
    ("mkm-tab-data", "Data"),
    ("mkm-tab-repro", "Reproduction"),
    ("mkm-tab-visual", "Visual Mapping"),
    ("mkm-tab-audio", "Audio Mapping"),
    ("mkm-tab-sync", "Sync"),
    ("mkm-tab-annotations", "Annotations"),
]


def _asset(package: str, name: str) -> str:
    return resources.files(package).joinpath(name).read_text(encoding="utf-8")


def _esc(s: object) -> str:
    return _html.escape(str(s), quote=True)


def _uses_3d(artifact: MathKernelArtifact) -> bool:
    for viz in artifact.visualizations:
        for block in (viz or {}).get("blocks", []):
            if block.get("kind") in BLOCK_3D_KINDS:
                return True
    return False


def render_html(artifact: MathKernelArtifact, *, mode: str = "portable",
                deterministic: bool = True,
                created_utc: str | None = None) -> str:
    if mode != "portable":
        raise ValueError("only mode='portable' is supported for multimodal "
                         "research artifacts")
    if created_utc is not None:
        artifact = artifact.model_copy(deep=True)
        artifact.reproducibility.created_utc = created_utc
    artifact.reproducibility.deterministic = deterministic

    payload = json.dumps(artifact.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    # A literal "</script>" inside the payload would terminate the tag early.
    payload = payload.replace("</", "<\\/")
    if len(payload.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("artifact payload exceeds the safety limit")
    payload_hash = hashlib.sha256(payload.encode()).hexdigest()

    three_tag = ""
    if _uses_3d(artifact):
        three_tag = "<script>\n" + _asset("mathkernel_viz.assets",
                                          "three.global.js") + "\n</script>"
    viewer_tag = "<script>\n" + _asset("mathkernel_viz.assets",
                                       "viewer.js") + "\n</script>"
    runtime_tag = "<script>\n" + _asset("mathkernel_multimodal.assets",
                                        "multimodal.js") + "\n</script>"

    tabs = "".join(
        f'<button data-tab="{tid}"{" class=\"mkm-active\"" if i == 0 else ""}>'
        f"{label}</button>"
        for i, (tid, label) in enumerate(_TABS))
    panes = "".join(
        f'<div id="{tid}" class="mkm-pane"{"" if i == 0 else " style=\"display:none\""}></div>'
        for i, (tid, _) in enumerate(_TABS))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{_CSP}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(artifact.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="mkm-head">
  <h1>{_esc(artifact.title)}</h1>
  <span class="mkm-badge">trust: {_esc(artifact.trust)}</span>
  <span class="mkm-badge">{_esc(artifact.artifact_schema)}</span>
  <span class="mkm-badge">{_esc(artifact.artifact_id or "research artifact")}</span>
</div>
<div class="mkm-main">
  <div class="mkm-content" id="mkm-content"></div>
  <div class="mkm-side">
    <div class="mkm-tabs">{tabs}</div>
    {panes}
  </div>
</div>
<script id="mathkernel-artifact" type="application/json" data-sha256="{payload_hash}">{payload}</script>
{three_tag}
{viewer_tag}
{runtime_tag}
<script>
document.querySelectorAll(".mkm-tabs button").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll(".mkm-tabs button").forEach(function (x) {{
      x.classList.remove("mkm-active");
    }});
    b.classList.add("mkm-active");
    document.querySelectorAll(".mkm-pane").forEach(function (p) {{
      p.style.display = "none";
    }});
    document.getElementById(b.getAttribute("data-tab")).style.display = "block";
  }});
}});
</script>
</body>
</html>
"""


def export_html(artifact: MathKernelArtifact, path: str | Path, **kwargs) -> dict:
    text = render_html(artifact, **kwargs)
    p = Path(path)
    p.write_text(text, encoding="utf-8", newline="\n")
    raw = text.encode()
    return {"path": str(p), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_schema": artifact.artifact_schema,
            "visualizations": len(artifact.visualizations),
            "sonifications": len(artifact.sonifications),
            "synchronization_links": len(artifact.synchronization),
            "deterministic": artifact.reproducibility.deterministic}
