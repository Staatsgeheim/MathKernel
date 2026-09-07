# Studio implementation handover

Date: 7 September 2026. Package: 0.1.0-alpha.1 / optional Python 0.1.0a1.
Branch: `mathkernel-studio`. Default branch `master` was not modified or merged.

## Grounding

- Live repository: `Staatsgeheim/MathKernel`.
- Base: `2146c6135f056e144cb9d1825764fa127f4fc3c2`, “Add trusted-publishing workflow for PyPI.”
- Live core version: **1.3.0**. The supplied design's older 1.3.0.dev30 archive was used only as design context; no older runtime source replaced current code.
- Input package: `MathKernel_Studio_UI_Design_Package(1).zip`.
- Input SHA-256: `1ee51fe7bc6672746c8dabb4f8e5f7faf6e5e6db8829143afcd322c44f3f33c2`.
- Normative design: MK-STUDIO-UI-001 revision 1.0. Its Markdown, 144 planned acceptance cases, example files and document schema are included under `ui/studio/design` and `ui/studio/schemas`.

Current APIs inspected: `Capability`/registry/query, `parameter_json_schema`, `MathResult`, `TrustLevel`, `ResultStatus`, `EvidenceBundle` support paths and trust ceilings, output-budget receipts and resource paging, narrow local jobs and existing visualization/sonification boundaries. There is no universal executable workflow host or browser-authorized compute service in this base.

## What exists

| Design milestone | Implemented foundation | Remaining exit criteria |
| --- | --- | --- |
| U0 | Live source map, explicit scope/capability gaps, scoped protocol DTOs, read-only facade, fault host, security boundary | Full live-kernel integration execution in a dependency-complete environment |
| U1 | Optional static React/TypeScript shell, loopback/session handshake, conservative negotiation, paged catalog, unknown/partial states, locally bundled assets | Actual browser/offline UI journeys, cached catalog UX, complete navigation and localization |
| U2 | Independent `mk.studio/1` model, canvas/outline, add/move/delete/duplicate/connect/reconnect, exact/numerical drafts, monotonic revision separation, bounded/coalesced undo, worker import preview, scoped/versioned IndexedDB recovery | Browser/assistive-technology verification, typed matrix/object pickers, broader widgets, worker-based graph diagnostics, advanced grouping, full recovery selection UX |
| U3 | Existing-result adapter, source bindings, trust/status separation, per-claim support paths, candidate separation, explicit receipt paging/integrity, escaped structured text viewer | Live full-kernel roundtrip, rich basic artifact adapters, paged admitted-claim refresh, complete viewer isolation |
| U4–U7 | Disabled capability controls, documented dependencies, initial packaging/security tests and CI | Workflow runtime integration, durable observations, fragments/subworkflows, remote compute/approval UI, active viewers, complete release hardening |

This is a **catalog/authoring/inspection development preview**. U1–U3 are not claimed to have met every specification exit criterion. All original 144 acceptance cases retain `planned_not_executed`; the new executable tests provide evidence for specific invariants, not blanket completion of that catalog.

## Host contract and boundaries

The optional `mathkernel-studio` distribution does not alter the core install or mathematical source. A stdlib loopback server serves static assets and fixed read-only endpoints. The only POST accepts the explicit one-time connection code; no generic RPC, filesystem export, operation invocation, workflow, provider or approval endpoint is exposed.

`KernelSource` pages the real `capability_query`. Descriptors retain real operation names, types, engines and declared schema hints. No ports or requiredness are invented; availability is conservatively unknown. The catalog digest binds references to metadata snapshots. Browser editing preserves unresolved operations across missing or changed schemas.

`InspectionRecord.from_result` accepts existing MathResult objects only from trusted Python host code, freezes serialization, and delegates claim summaries to the existing evidence model. There is no file/HTTP admission path. Results have source identity and revision, plus optional explicit workflow mapping. A standalone source revision never pretends to be a frozen workflow revision.

Protocol `studio-host/1` is this preview's new adapter contract, not an API claimed to exist in the baseline. Every successful response carries protocol, host/workspace, correlation and observation time. The client validates envelopes/payloads, byte limits and scope, rejects late generations, and does not retry mutations. No network request occurs on mounting the application, opening a document or mounting a viewer. Connection/catalog/result requests are explicit user actions.

Consumed features: `catalog_read`, local `authoring_draft`, and optional `result_inspect` when existing results are published. `operation_invoke`, `workflow_validate`, `workflow_execute`, `run_observe`, `artifact_view` (active/rich viewers), `compute_review` and `approval_interact` stay false in the shipped host. Workflow buttons remain disabled even if an unimplemented future feature is advertised.

## Source changes

New application code, contracts and tests are entirely under `ui/studio/`. Main groups:

- `src/app`: shell, reusable UI primitives, responsive system-theme styles.
- `src/editor`: schema, commands, advisory diagnostics, React Flow adapter, connection dialog, import and recovery.
- `src/inspectors`: exact/numerical codecs and parameter inspector.
- `src/host`, `src/evidence`, `src/security`, `src/workers`: scoped client, claim/receipt views, bounded parsing, no-JIT validation and cancellable import.
- `host/src/mathkernel_studio`: static/session host, real discovery projection and synthetic faults.
- `tests`, `host/tests`: 91 tests including three dependency-gated integration tests.
- `scripts/manifest.mjs`, `host/setup.py`: asset/dependency/license packaging and stale-asset rejection.
- `.github/workflows/studio.yml`: scoped build/test/wheel CI. It has not run remotely because push is blocked.
- `README.md`: 21 additive lines; existing functional coverage retained.
- Existing Python and MCP skills: a short Studio route and equivalent `studio.md` guidance.

No MathKernel core mathematics, evidence reconciliation, package version, compute policy or default-branch implementation was changed. `mk.studio/1` is unchanged from the supplied schema; there is no migration. Runtime validation adds referential integrity, duplicate-key rejection and combined byte budgets.

## Verification actually executed

| Verification | Pass | Fail | Skip | Scope |
| --- | ---: | ---: | ---: | --- |
| Vitest | 66 | 0 | 0 | Codecs/reducer 45; client/receipt contracts 12; recovery 3; static component rendering/CSP-compatible validation 6 |
| Python unittest | 22 | 0 | 3 | Session/origin/fault host, real legacy registry classes, generated asset/HTTP packaging; full-kernel integration skipped |
| Strict TypeScript / production Vite build | passed | — | — | Locked dependencies, independent import-worker bundle |
| Existing skill validators | 2 | 0 | 0 | Python and MCP skill frontmatter/reference updates |
| Whitespace/diff checks | passed | — | — | No core source edits |

**88 executable tests passed; 3 were explicitly skipped.** No mathematical execution happened in the browser or fixture host. Actual current `capabilities.py` and `discovery.py` were exercised through isolated imports. Whole `MathKernel` initialization/result integration tests were skipped because sympy/z3 and other core dependencies were unavailable. Attempted dependency installation was stopped by environment network approval; no live-kernel success is claimed.

Measured environment: Linux x86-64 (kernel 6.18.35), Python 3.12.13, Node 24.19.0, npm 11.9.0, TypeScript 5.9.2, Vite 7.1.5, Vitest 3.2.4, React 19.1.1, React Flow 12.8.5. The final frontend test run took 0.784 seconds; Python tests took 6.763 seconds; production bundling took 4.92 seconds. These are container test/build measurements, not interactive browser benchmarks.

Current bundles: main JS about 480.70 kB (150.46 kB gzip), import worker 56.60 kB, CSS 28.36 kB (5.70 kB gzip). Eight manifest-covered local files plus the manifest; dependency inventory covers 180 lockfile records. The dependency inventory is not a standards-format SBOM or an online vulnerability assessment.

**Browsers and assistive technologies exercised: none.** No Chrome/Edge/Firefox/Safari, NVDA/JAWS/VoiceOver, manual drag or keyboard journey, 200% zoom or actual CSP browser enforcement claim is made. Static component rendering and schema tests with dynamic Function disabled are narrower checks. No UI latency/frame-rate/memory benchmark has been run; the 200-node canvas cutoff is conservative, not a measured supported envelope.

## Security and incomplete integration

- Exact math uses strings. JSON rejects unsafe large numeric literals rather than rounding them; extremely large approximate JSON numbers may consequently require host string encoding too.
- Import requires preview/explicit open and runs in a bounded worker. Unknown top-level authority/trust payloads are rejected; user-authored text remains inert.
- Parameters persist as invalid draft text while editing. Labels and layout fields commit on blur. No silent conversion or dynamic remote schema fetching occurs.
- Recovery is opt-in, max ten documents, keyed by host/workspace/document and version checked. It is not an authoritative save or backup. The recovery dialog currently offers the most recent copy in the scope rather than a complete recovery browser.
- Host session bootstrap checks exact Host and Origin; API reads require an HttpOnly/SameSite cookie. Session revocation occurs on host stop/expiry; UI disconnect clears browser observations but does not revoke the cookie. Remote HTTPS/proxy support is outside this loopback preview.
- The protected host is a small single-process read-only preview, not a general Internet service. Host request-rate hardening and operational multiuser authentication remain future work.
- Script policy does not allow inline script or eval. Zod runs with JIT disabled. Styles permit dynamic attributes for React Flow. Real-browser CSP verification is still required.
- Structured text only: no arbitrary HTML, SVG, remote images, active iframes, audio or 3D. The complete isolated active-viewer contract is deferred; inert fallback is deliberate.
- Receipt bytes are integrity checked and capped at 2 MiB, but remain source data; admitted claim summaries need a complete host observation. This prevents receipt original-trust metadata from being presented as proof.
- Read permission loss clears private result observations; host scope and generation checks reject stale responses. Durable event streams, unknown workflow submission reconciliation and cost/cleanup axes remain unimplemented because their host APIs are absent.

## External blockers and continuation

1. **GitHub push:** The connector reported repository permission flags but actual branch creation returned HTTP 403 `Resource not accessible by integration`. The local Git client has no authenticated push credential. The `mathkernel-studio` branch exists locally; no remote branch or PR is claimed. Do not merge into `master` unless Maarten asks.
2. **Full MathKernel integration tests:** Install the core dependencies in an approved environment; run the three explicit integration tests before advertising live end-to-end inspection.
3. **Browser verification and accessibility:** Exercise the packaged host's canvas, outline, import worker, recovery, exact inputs, candidate/receipt views and CSP in real browsers. Current tests are not a substitute.
4. **Workflow host:** U4 needs separately supplied authoritative validate/plan/submit/observe contracts. Do not implement a scheduler in Studio to remove this dependency.
5. **U2/U3 completion:** Expand safe typed widgets/object pickers, diagnostic focus, catalog drift review, receipt admission refresh and rich viewer adapters before broader authoring/inspection release claims.
6. **U5–U7:** Follow the supplied design for fragments, existing host subworkflows, compute/approval presentation, independent lifecycle facts, real active-viewer isolation, localization, accessibility and measured performance.

The delivery archive includes the local branch as a Git bundle, source snapshot, optional built wheel and reproduction instructions. It can be fetched into the user's existing clone without touching the default branch. The implementation source remains on the dedicated development branch.
