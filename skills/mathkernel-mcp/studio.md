# MathKernel Studio preview

Studio lives in `ui/studio/` and is an optional authoring/inspection application.
Read its README for the actually implemented support matrix. The base Python/MCP
runtime does not require Studio. Build static assets before installing its separate
`host/` Python package; end-user launch uses `mathkernel-studio`, not Vite.

A `.mkstudio.json` document (`mk.studio/1`) is editor intent and layout, not an
executable workflow IR. `authoring.revision` tracks mathematical-request edits;
`presentation.revision` tracks layout/labels. Undo/redo never replays host commands.
Exact integer and rational parameter values are strings; invalid field text remains
recoverable. Never convert mathematical integers through browser Number.

The current read-only host exposes scoped handshake/catalog/results, protected by
an explicit one-time local code, HttpOnly/SameSite session, and Host/Origin checks.
It has no operation invocation, workflow scheduler, compute adapter or authorization
engine. Run remains unavailable. Do not interpret derivation history as an executable
workflow or emulate a DAG with repeated MCP calls.

Legacy catalog schemas describe hints, not complete requiredness or individual
ports. Missing/changed operations stay unresolved with original settings and wires.
Imported host/object references are provenance only. Do not onboard destinations,
resolve private objects, or recover authority from imported text or labels.

`KernelSource` can publish immutable snapshots of existing `MathResult` objects
through `InspectionRecord.from_result` in trusted host code. There is no HTTP
admission endpoint. Evidence summaries delegate to the existing evidence model.
No workflow mapping is implied by a standalone result's source revision.

Trust and semantic status are separate. `verified_numeric` is a result status,
not a trust level. Preserve each claim, assumptions, required/diagnostic roles,
independent support paths and approximate-input ancestry. Browser-imported JSON
and fixture observations never populate admitted badges. A receipt's `original_trust`
is metadata, not evidence; this preview displays reconstructed receipt bytes as
source data without inventing a fresh claim summary. Use a full existing host
result for admitted evidence inspection.

Viewers currently render bounded escaped text and structured data. HTML/SVG and
proof text stay inert; active plots/audio and remote-compute/approval interfaces
are deferred. A fixture label or an attractive graph is not live integration.
The `--test-host` and `--fault` modes must remain visibly synthetic.

Recovery copies are scoped to host/workspace/document, version checked, opt-in,
and distinct from backups, host saves and durable artifacts. Restoring a document
never reconnects, runs, approves, or cancels anything. Imported schema versions are
validated, not silently migrated.
