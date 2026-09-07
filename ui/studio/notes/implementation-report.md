# Studio implementation notes

Frontend package 0.2.0-alpha.1. Optional Python package 0.2.0a1. U0–U3 authoring and inspection preview.

Current source is the authority for capability discovery, `MathResult`, evidence support paths/trust ceilings, output-budget receipts and resource paging. There is still no universal workflow service. Studio does not emulate one by issuing operations sequentially.

## U0–U3 completion

| Milestone | Delivered state |
| --- | --- |
| U0 | Current-source map, explicit capability gaps, versioned DTOs, protected loopback facade, synthetic fault host, optional workflow-service boundary and threat controls. |
| U1 | Offline packaged shell, one-time session bootstrap/revocation, scoped capability negotiation, paged/searchable catalog, drift/cache review, test-host disclosure, help/preferences and honest unavailable states. |
| U2 | Independent exact document model, canvas and equivalent outline, atomic connect/reconnect, duplication/deletion, monotonic semantic/layout revisions, bounded undo/redo, background diagnostics, typed exact/numerical/matrix input review, object/subworkflow references, groups, authority-free fragments, revision comparison, bounded inert import/export and scoped multi-copy recovery browser. |
| U3 | Existing-result and candidate adapters, immutable source bindings, trust/status separation, per-claim support paths, candidate admission rules, receipt integrity/paging/admission refresh, exact source tables, and a fixed isolated plot/point/trajectory renderer with a coordinate-table alternative. |

U0–U3 are code-complete for the capabilities available in this repository. The original acceptance catalog intentionally remains a design artifact with `planned_not_executed` labels; executable tests and the browser run are the implementation evidence.

U4–U7 are not falsely claimed complete. The recovered code contains capability-gated client/presentation support for authoritative validation, planning, approval, submission, reconciliation and run observation, plus composition primitives useful to later milestones. Actual workflow execution still requires a separately supplied host implementation. Remote compute/provider provisioning, durable run event streams, localization, the complete multi-browser/screen-reader matrix and release hardening remain outside U0–U3.

## Safety and authority boundaries

- Exact mathematics stays textual. JSON numeric literals outside safe bounds are rejected rather than rounded.
- Imports are previewed in a bounded worker and cannot reconnect, execute, approve, provision or admit evidence.
- Unknown operations, schemas, ports and compatibility remain unresolved; Studio never invents requiredness or positional remaps.
- Candidate labels never become host-admitted trust. A receipt never creates evidence. Admission requires the authenticated original result identity.
- Every successful host response is protocol-, host-, workspace-, request- and observation-bound. Late generations and scope changes are rejected.
- The default host remains read-only and advertises no workflow execution. Optional services require an explicit typed contract; there is no generic dispatch route.
- The host binds to `127.0.0.1`, validates exact Host/Origin, uses a one-time code and HttpOnly/SameSite session cookie, rate-limits reads and supports explicit session revocation.
- The main application has no CDN, telemetry, provider discovery or remote assets. Validators run without JIT/eval.
- Active scientific presentation accepts only bounded inline numeric projections. The iframe remains an opaque origin (`sandbox="allow-scripts"`, without `allow-same-origin`), uses a self-contained document with a SHA-256-authorized script, has `connect-src 'none'`, and communicates through a narrow typed `MessageChannel`. Source data and exact tables remain available separately.

## Verification executed

| Verification | Result | Scope |
| --- | --- | --- |
| Vitest | 84 passed | Exact codecs and graph commands, import/security caps, client envelopes, recovery, composition, matrix review, result/viewer contracts and static rendering. |
| Python unittest | 35 passed | Session/origin/rate controls, fault host, packaging/manifest/CSP integrity, optional service behavior and four live full-kernel integration cases. |
| TypeScript + production builds | passed | Main shell, import/diagnostic workers, isolated viewer, dependency inventory, license notices and exact asset manifest. |
| Formatting and diff checks | passed | Frontend format and repository whitespace checks. |
| Real browser | Chrome passed | Offline shell; exact large integer; editor checks; recovery conflict; fixture disclosure; catalog; click-only typed connection; result identity/evidence; isolated 3-point renderer and table alternative. |

The live integration constructed `MathKernel`, queried its actual registry, inspected an exact value beyond JavaScript safe integer precision, froze result snapshots, resolved registered receipt pages and verified that admission requires the matching original result.

The browser produced no Studio-origin console error. Messages from the cloud browser extension itself were excluded from application evidence. The isolated renderer initially exposed an unfinished external-resource/CSP interaction; it was replaced with a single-file hash-authorized viewer and re-tested successfully without adding `allow-same-origin`.

## Package and continuation boundary

All implementation changes are under `ui/studio/` plus its scoped CI workflow and small additive documentation/skill routes. Core mathematics, evidence reconciliation, compute policy and the default package version are unchanged. Node is required only to build assets; the wheel runs without Node.

Remaining work begins at U4 and depends on an authoritative workflow host: validate a frozen revision, create an immutable plan/digest, obtain policy decisions and any challenge-bound approval, submit idempotently, observe durable lifecycle facts, and reconcile unknown acknowledgements. Studio must continue to keep cleanup, cost and mathematical status as independent axes.

Remaining work stays on this feature branch until it is ready to publish.
