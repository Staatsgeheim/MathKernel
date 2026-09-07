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

The default local host exposes scoped handshake/catalog/results plus the separate
`mathkernel_workflow` backend, protected by a one-time local code,
HttpOnly/SameSite session and Host/Origin checks. It compiles only complete explicit
`workflow.*` adapters into frozen plans. Kernel workers run under host-owned
wall-time/concurrency policy; no browser or MCP-call loop schedules them.
`--read-only` disables this runtime. `--deny-execution` cannot be overridden by
approval. Persistent state binds host/workspace; restart marks active work
interrupted without replay. Retained requests/results support reconnect.

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

Viewers render bounded inline plots/point clouds/trajectories and an isolated
resolved sine-event audio audition with explicit playback controls. HTML/SVG and
proof text stay inert. Audio audition is not the exported PCM waveform; it performs
no mappings or mathematical analysis. The default host supports local CPU policy only; no remote/GPU fallback is configured. A fixture label or an attractive graph is not live integration.
The `--test-host` and `--fault` modes must remain visibly synthetic.

Recovery copies are scoped to host/workspace/document, version checked, opt-in,
and distinct from backups, host saves and durable artifacts. Restoring a document
never reconnects, runs, approves, or cancels anything. Imported schema versions are
validated, not silently migrated.


Advanced workflow UI is gated by `studio-workflow/1`. Unknown outcomes retain
request/plan identity only; use read-only reconciliation before another submission.
Never clear recovery state just to enable Run. Approval is host-issued, expires,
and binds the immutable plan; unknown prices and resource cleanup remain separate
from result readiness. `--test-workflow` is synthetic, not execution evidence.
Subworkflow boundary inspection never rebinds or inlines a draft automatically.
Saved self-contained subworkflows pin content, kernel version and revision. Publication
is not execution; unsupported external input/control constructs fail validation.
Consult `ui/studio/notes/local-execution-report.md` for the current tested support boundary.
