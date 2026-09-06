# =============================================================================
# MathKernel Viz - self-contained interactive HTML artifact renderer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""HTMLArtifactRenderer: packages a VisualizationDocument into a single
portable .html file (or a slim web-mode file referencing pinned assets).

Security: all embedded strings are JSON-escaped inside a non-executable
script tag; the viewer uses textContent only; no eval; CSP meta; no network
requests in portable mode.  MathIR is data, never code.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from ..document import VisualizationDocument
from ..manifest import build_manifest, sha256_text
from .svg import SVG_KINDS, render_svg

MAX_PAYLOAD_BYTES = 256 * 1024 * 1024

_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'none'; font-src 'none'")

_CSS = """
:root{color-scheme:dark}
body{margin:0;font-family:Inter,ui-sans-serif,system-ui,sans-serif;background:#0b0e11;color:#dbe8f3}
.mkv-head{padding:14px 20px;background:#10161b;border-bottom:1px solid #243039;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.mkv-head h1{font-size:17px;margin:0;font-weight:620}
.mkv-badge{font:11px ui-monospace,monospace;padding:3px 8px;border:1px solid #23415a;border-radius:999px;color:#66bfff;background:#10202c}
.mkv-main{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:calc(100vh - 52px)}
.mkv-view{padding:14px;min-width:0}
.mkv-grid{display:grid;gap:12px}
.mkv-block{background:#11171c;border:1px solid #1d2831;border-radius:8px;min-width:0;overflow:hidden}
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
.mkv-side{border-left:1px solid #1d2831;background:#11171c;padding:12px;overflow:auto;font-size:12px}
.mkv-tabs button{border:1px solid #293943;background:#172027;color:#9cb0bd;padding:4px 9px;font-size:11px;cursor:pointer;border-radius:5px;margin:0 3px 6px 0}
.mkv-tabs button.mkv-active{background:#176795;color:#eaf7ff;border-color:#296a94}
.mkv-pane h3{font-size:13px;margin:4px 0}
.mkv-pane h4{font-size:11px;margin:12px 0 4px;color:#8ca1ae;text-transform:uppercase;letter-spacing:.07em}
.mkv-pane p{margin:4px 0;color:#b8c8d3}
.mkv-step{border-left:3px solid #293943;padding:2px 8px;margin:6px 0}
.mkv-dim{color:#5c707c;font-size:11px}
.mkv-hash{word-break:break-all;font-family:ui-monospace,monospace}
.mkv-trust{font-weight:700}
.mkv-error{padding:18px;background:#2a1414;border:1px solid #5c2626;border-radius:6px;color:#e8a1a1;font-size:13px;margin:8px}
.mkv-tip{display:none;position:fixed;z-index:50;background:#0d1317;border:1px solid #2b3d49;color:#dbe8f3;font:11px ui-monospace,monospace;padding:8px 10px;border-radius:6px;pointer-events:none;white-space:pre;max-width:420px;box-shadow:0 6px 24px #000a}
.mkv-exp{margin-top:14px;border-top:1px solid #1d2831;padding-top:10px}
.mkv-exp h4{margin:0 0 6px;font-size:11px;color:#8ca1ae;text-transform:uppercase;letter-spacing:.07em}
.mkv-exp button{margin:2px 4px 2px 0;padding:5px 10px;font-size:11px;border:1px solid #293943;border-radius:5px;background:#172027;color:#c6d5df;cursor:pointer}
.mkv-exp button:hover{background:#1b252c}
#mathkernel-static-svg{display:none}
@media(max-width:900px){.mkv-main{grid-template-columns:1fr}.mkv-side{border-left:0;border-top:1px solid #1d2831}}
"""


def _asset(name: str) -> str:
    return resources.files("mathkernel_viz.assets").joinpath(name).read_text(encoding="utf-8")


def _payload_json(doc: VisualizationDocument) -> str:
    text = json.dumps(doc.model_dump(mode="json"), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True)
    # A literal "</script>" inside the payload would terminate the tag early.
    return text.replace("</", "<\\/")


def render_html(doc: VisualizationDocument, *, mode: str = "portable",
                embed_data: bool = True, embed_runtime: bool = True,
                include_provenance: bool = True,
                include_reproducibility: bool = True,
                mathkernel_version: str = "unknown",
                deterministic: bool = True,
                created_utc: str | None = None,
                seed: int | None = None) -> str:
    if mode not in ("portable", "web"):
        raise ValueError("mode must be 'portable' or 'web'")
    if not include_provenance:
        doc = doc.model_copy(update={"provenance": {"steps": []}})

    renderer_name = "threejs+canvas" if doc.uses_3d() else "canvas+svg"
    if include_reproducibility:
        doc.manifest = build_manifest(
            doc, renderer=renderer_name, mathkernel_version=mathkernel_version,
            deterministic=deterministic, created_utc=created_utc, seed=seed)

    payload = _payload_json(doc)
    if len(payload.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("artifact payload exceeds the safety limit; "
                         "use chunked datasets or web mode")
    # The payload hash travels as a script-tag attribute so the viewer can
    # verify the exact embedded text (a hash inside the payload could never
    # cover itself).
    payload_hash = sha256_text(payload)

    if not embed_data:
        raise ValueError("embed_data=False is only meaningful for multi-file "
                         "web export, which is not part of the single-file artifact")

    static_svg = ""
    if doc.blocks and all(b.kind in SVG_KINDS for b in doc.blocks):
        static_svg = '<div id="mathkernel-static-svg">' + render_svg(doc) + "</div>"

    needs_three = doc.uses_3d()
    if mode == "portable" or embed_runtime:
        viewer_tag = "<script>\n" + _asset("viewer.js") + "\n</script>"
        three_tag = ""
        if needs_three:
            three_tag = "<script>\n" + _asset("three.global.js") + "\n</script>"
    else:
        viewer_tag = '<script src="assets/viewer.js"></script>'
        three_tag = '<script src="assets/three.global.js"></script>' if needs_three else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{_CSP}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc_title(doc.title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="mkv-head">
  <h1>{_esc_title(doc.title)}</h1>
  <span class="mkv-badge">trust: {_esc_title(doc.trust)}</span>
  <span class="mkv-badge">engine: {_esc_title(doc.engine or "unknown")}</span>
  <span class="mkv-badge">{_esc_title(doc.artifact_schema)}</span>
</div>
<div class="mkv-main">
  <div class="mkv-view" id="mkv-view"></div>
  <div class="mkv-side">
    <div class="mkv-tabs">
      <button data-tab="mkv-tab-result" class="mkv-active">Result</button>
      <button data-tab="mkv-tab-evidence">Evidence</button>
      <button data-tab="mkv-tab-provenance">Provenance</button>
      <button data-tab="mkv-tab-data">Data</button>
      <button data-tab="mkv-tab-repro">Reproduction</button>
    </div>
    <div id="mkv-tab-result" class="mkv-pane"></div>
    <div id="mkv-tab-evidence" class="mkv-pane" style="display:none"></div>
    <div id="mkv-tab-provenance" class="mkv-pane" style="display:none"></div>
    <div id="mkv-tab-data" class="mkv-pane" style="display:none"></div>
    <div id="mkv-tab-repro" class="mkv-pane" style="display:none"></div>
    <div class="mkv-exp">
      <h4>Export</h4>
      <button id="mkv-exp-png">Screenshot (PNG)</button>
      <button id="mkv-exp-svg">SVG</button>
      <button id="mkv-exp-csv">CSV</button>
      <button id="mkv-exp-json">JSON</button>
    </div>
  </div>
</div>
<div class="mkv-tip" id="mkv-tip"></div>
{static_svg}
<script id="mathkernel-data" type="application/json" data-sha256="{payload_hash}">{payload}</script>
{three_tag}
{viewer_tag}
<script>
document.querySelectorAll(".mkv-tabs button").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll(".mkv-tabs button").forEach(function (x) {{
      x.classList.remove("mkv-active");
    }});
    b.classList.add("mkv-active");
    document.querySelectorAll(".mkv-pane").forEach(function (p) {{
      p.style.display = "none";
    }});
    document.getElementById(b.getAttribute("data-tab")).style.display = "block";
  }});
}});
</script>
</body>
</html>
"""


def _esc_title(s) -> str:
    import html as _html
    return _html.escape(str(s), quote=True)


def export_html(doc: VisualizationDocument, path: str | Path, **kwargs) -> dict:
    text = render_html(doc, **kwargs)
    p = Path(path)
    p.write_text(text, encoding="utf-8", newline="\n")
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": sha256_text(text),
            "artifact_schema": doc.artifact_schema,
            "renderer": doc.manifest.get("renderer"),
            "deterministic": doc.manifest.get("deterministic", True)}
