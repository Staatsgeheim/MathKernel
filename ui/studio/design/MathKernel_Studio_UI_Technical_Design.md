# MathKernel Studio — Visual Workflow User Interface

## Technical design and implementation contract

**Document ID:** MK-STUDIO-UI-001\
**Revision:** 1.0 — proposed UI design, not an implemented feature\
**Prepared for:** Maarten Boone\
**Date:** 7 September 2026\
**Baseline:** supplied `MathKernel(6).zip`, package version `1.3.0.dev30`\
**Companion reference:** MK-RHC-001 revision 1.0, the separate MathKernel remote and heterogeneous compute design, provisionally associated with release 1.4.\
**Release placement:** an independently deliverable optional UI. No MathKernel package version is assigned by this document.

**Baseline archive SHA-256**

```text
6c5975c3ddbc6e467737ebf4ed67b1843d62b97fe07b9b0b4dda3b8a41601f1c
```

### Purpose

Specify a local-first, Node-RED-inspired visual interface for discovering, configuring, composing, running where supported, and inspecting MathKernel operations. The interface exposes mathematical types, assumptions, evidence, execution provenance, visualizations, and sonification without becoming another mathematics engine or workflow scheduler.

This document covers the UI, its presentation and authoring models, browser-side behavior, and the observable host interfaces it requires. It does **not** define a new execution IR, graph compiler, scheduler, compute provider, mathematical verifier, artifact format, billing ledger, or authorization implementation. An interface requirement is not a claim that the baseline supplies that interface.

The deliverable is deliberately separate from the 1.4 compute design. It does not modify or supersede MK-RHC-001. All proposed `Studio*` types, application routes, host-facade methods, feature identifiers, and filenames are new design names. Numeric limits and performance budgets are proposed targets, not measurements. The accompanying fixtures and acceptance catalog describe future tests, not passing implementation tests.

### Central architectural rule

> Studio authors requests and displays facts. MathKernel establishes mathematical meaning and evidence. The workflow host owns execution. The compute service owns placement and lifecycle. The authorization service owns permission. A browser drawing, badge, or button cannot substitute for any of them.

### The essential dependency qualification

The inspected baseline has typed mathematical objects, capabilities, result/evidence structures, a derivation history, visualization and sonification documents, and a small local-job facility. It does **not** establish that a general executable multi-node workflow service already exists. MK-RHC-001 describes registered compute requests and bounded batches, not a universal visual-DAG runtime. A historical derivation DAG is not automatically an executable workflow. [B1–B9; P1 §§2, 5–7, 22]

Studio can ship useful catalog, authoring, result-inspection, and supported single-operation experiences without that service. Advertising end-to-end visual workflow execution requires a separately supplied host capability and integration tests. Do not hide this dependency by executing nodes sequentially in JavaScript.

## Contents

1. [Product scope and ownership boundaries](#1-product-scope-and-ownership-boundaries)
2. [Baseline integration map and capability gaps](#2-baseline-integration-map-and-capability-gaps)
3. [UI invariants and acceptance principles](#3-ui-invariants-and-acceptance-principles)
4. [Users, interaction modes, and primary journeys](#4-users-interaction-modes-and-primary-journeys)
5. [Information architecture and application shell](#5-information-architecture-and-application-shell)
6. [Visual language and component anatomy](#6-visual-language-and-component-anatomy)
7. [Frontend architecture and dependency policy](#7-frontend-architecture-and-dependency-policy)
8. [Operation catalog and presentation metadata](#8-operation-catalog-and-presentation-metadata)
9. [Studio authoring document and identity model](#9-studio-authoring-document-and-identity-model)
10. [Canvas editing and command semantics](#10-canvas-editing-and-command-semantics)
11. [Typed ports, contexts, and compatibility feedback](#11-typed-ports-contexts-and-compatibility-feedback)
12. [Node inspector and mathematical input editing](#12-node-inspector-and-mathematical-input-editing)
13. [Groups, reusable fragments, and subworkflow presentation](#13-groups-reusable-fragments-and-subworkflow-presentation)
14. [Validation, diagnostics, and repair interactions](#14-validation-diagnostics-and-repair-interactions)
15. [Run controls, preview behavior, and revision freezing](#15-run-controls-preview-behavior-and-revision-freezing)
16. [Run monitoring, event handling, and reconnect behavior](#16-run-monitoring-event-handling-and-reconnect-behavior)
17. [Evidence, assumptions, and provenance inspection](#17-evidence-assumptions-and-provenance-inspection)
18. [Results, objects, and safe artifact viewers](#18-results-objects-and-safe-artifact-viewers)
19. [Visualization, sonification, and presentation controls](#19-visualization-sonification-and-presentation-controls)
20. [Compute-plan review and authorization UI](#20-compute-plan-review-and-authorization-ui)
21. [Saving, recovery, history, comparison, and export](#21-saving-recovery-history-comparison-and-export)
22. [Connection onboarding, preferences, and compatibility](#22-connection-onboarding-preferences-and-compatibility)
23. [Required host interfaces and dependency contracts](#23-required-host-interfaces-and-dependency-contracts)
24. [Browser security and artifact isolation](#24-browser-security-and-artifact-isolation)
25. [Accessibility, localization, and inclusive interaction](#25-accessibility-localization-and-inclusive-interaction)
26. [Performance, resource budgets, and large-workflow behavior](#26-performance-resource-budgets-and-large-workflow-behavior)
27. [Test strategy and acceptance catalog](#27-test-strategy-and-acceptance-catalog)
28. [Packaging, distribution, and documentation](#28-packaging-distribution-and-documentation)
29. [Delivery sequence and agent handover](#29-delivery-sequence-and-agent-handover)
30. [Architecture decisions, risks, and deferred features](#30-architecture-decisions-risks-and-deferred-features)
31. [Worked end-to-end UI scenarios](#31-worked-end-to-end-ui-scenarios)
32. [Source register and reference boundaries](#32-source-register-and-reference-boundaries)

# 1. Product scope and ownership boundaries

## 1.1 Goals

Studio shall make MathKernel's existing breadth discoverable through a searchable operation palette and a visual composition surface. A user should be able to inspect a component's supported mathematical inputs, parameters, outputs, assumptions, engines, evidence behavior, and availability before attempting a calculation. An operation's description must remain accessible even when a required optional backend is not installed.

The product shall support progressive disclosure. A new user can connect a few components and inspect an output without reading every engine setting. An expert can inspect exact types, arithmetic transitions, context bindings, per-claim evidence, input identities, runtime versions, and failure details without leaving the same workspace. Neither mode may hide a consequential assumption, lossy conversion, permission requirement, or unresolved cleanup warning.

Studio is optional. Existing Python and MCP users must not need a browser, JavaScript runtime, UI package, cloud account, or Studio workspace. The installed UI must work with locally delivered static assets and user-owned infrastructure. No mandatory hosted login, telemetry endpoint, CDN, license check, or MathKernel relay is introduced.

## 1.2 Included work

Included work consists of the application shell; operation catalog; node and connection editing; parameter inspectors; input/object pickers; draft validation display; reusable visual fragments; host-provided subworkflow presentation; execution controls; run monitoring; result, evidence, and artifact inspection; compute-plan review; host-mediated approval presentation; local draft recovery; import/export; accessibility; performance; browser security; packaging; and UI conformance tests.

A thin presentation facade may adapt existing approved host interfaces into browser DTOs. It may scope references, page results, translate transport envelopes, and serve static assets. It must not make new mathematical claims, silently run dependent operations, bypass host limits, mint its own spending grants, or create cloud resources. Its observable contract is specified here; its service internals are not.

## 1.3 Excluded work

The following are explicitly outside this plan: new solvers or mathematical algorithms; a workflow compiler or execution interpreter; automatic parallelization; distributed scheduling; provider SDK integration; worker management; verification algorithms; secret-vault implementation; account-wide billing; artifact computation engines; arbitrary Python or shell nodes; remote package installation; a plugin marketplace; collaborative multi-user editing; a hosted SaaS service; and an embedded AI assistant.

A graph node named “Map,” “Branch,” “Sweep,” or “Subworkflow” is available only when the host advertises the corresponding semantics. Drawing it is not implementing it. A compute target is a node property or explicit policy annotation, never a mathematical transform inserted between operations.

## 1.4 Ownership matrix

| Concern | Studio owns | External owner |
|---|---|---|
| Composition | Editing intent, connections, labels, layout, navigation | Host validates and resolves executable meaning |
| Types | Display and bounded advisory compatibility feedback | MathKernel/host decides admissibility and conversions |
| Execution | Plan/run/cancel controls and status presentation | Workflow host and compute service |
| Evidence | Faithful rendering, source links, claim navigation | MathKernel and its admitted verification records |
| Money and export | Explicit review, warnings, confirmation interaction | Host authorization and compute accounting |
| Artifacts | Safe, bounded viewing of supported documents | Existing artifact/projection/sonification producers |
| Persistence | UI draft recovery, layout, preferences | Host stores authoritative workflows, jobs, and results |
| Credentials | Connection status and credential-reference labels | Operator-controlled host credential mechanisms |

The boundary remains valid when all processes run on one laptop. Co-location does not make a browser trustworthy enough to replace server-side validation.

# 2. Baseline integration map and capability gaps

## 2.1 Inspected integration points

The source references below identify this exact supplied archive. The companion baseline manifest records hashes. Re-ground on the newest supplied source before implementation; none of these observations licenses overwriting a newer API.

| Existing component | Observed responsibility | UI implication |
|---|---|---|
| `MathKernel.capabilities`, `capability_query` | Semantic discovery through a capability registry | Seed catalog entries; do not infer complete visual contracts |
| `capabilities.py::Capability` | Domain, operation, type names, engines, evidence, optional parameter schema | Reuse IDs and semantics; add presentation metadata separately |
| `discovery.py::parameter_json_schema` | Declared primitive hints; explicitly incomplete requiredness | Show schema coverage; never invent required inputs |
| `models.py::MathResult` | Trust, semantic status, per-claim evidence, assumptions, derivations | Render existing axes independently |
| MathIR numeric nodes | Integers and rationals encoded with strings | Preserve exact values in browser editors and transport |
| `object_create`, `object_get`, typed-operation paths | Construct and inspect typed mathematical objects | Use scoped object references; no browser object-store clone |
| `output_policy.py` | Bounded result receipts and paged retrieval | Recognize receipts before rendering a mathematical result |
| `mathkernel_projection` | Shared evidence-preserving representation | Reuse source lineage and information-loss metadata |
| `mathkernel_viz/document.py` | Renderer-neutral block-based visual documents | Embed supported viewers, not a second plot specification |
| `mathkernel_sonify/models.py` | Sonification mappings, events, tracks | Inspect mappings and play supported representations safely |
| `viz_dag` and derivations | Existing recorded provenance views | Keep provenance separate from editable authoring graph |
| `job_submit/status/result/list` | Narrow local job operations | Expose their real limits; do not assume general DAG execution |
| `mathkernel_mcp/server.py` | Existing tool/resource registration | Host may reuse approved calls; browser must not reflect arbitrary methods |

These are integration opportunities, not a claim that a complete browser API, typed node registry, authorization UI, or durable workflow editor exists. [B1–B10]

## 2.2 Catalog completeness is explicit

`Capability.input_types` and `output_types` are useful semantic labels, but they do not necessarily define stable individual port IDs, cardinality, dependent shapes, context compatibility, requiredness, or branch behavior. The generated legacy parameter schema deliberately states that mathematical invariants remain adapter-validated. Studio must preserve this qualification. [B1–B2]

Introduce a host-delivered presentation descriptor associated with a real capability ID. It may declare a complete node contract, a partial inspector, or a documentation-only entry. Missing information produces “Requires host validation” or “Not visually composable yet,” not a guessed port. A local presentation overlay can choose a matrix widget, icon, or description; it cannot manufacture execution eligibility.

## 2.3 Feature tiers

`catalog_read` permits discovery and documentation. `authoring_draft` permits local graph editing and export. `operation_invoke` permits explicitly supported standalone operations. `workflow_validate` permits authoritative graph diagnostics. `workflow_execute` permits host-submitted workflows. `run_observe`, `result_inspect`, `artifact_view`, and `compute_review` enable their respective surfaces. `approval_interact` is a separately authenticated host capability.

A deployment may support any consistent subset. The UI renders a clear capability summary and explains disabled actions. Tests must run against intentionally reduced hosts, including a disconnected authoring session, rather than assuming the full future stack is always present.

## 2.4 No disguised runtime in the frontend

React Flow offers graph interaction and examples of data-driven node UIs; it does not establish MathKernel's mathematical or security semantics. Studio must not interpret a completed predecessor as permission to call the next backend tool. It must not implement its own retry policy, batch partitioner, result cache, exactness reconciliation, or topological job launcher. Graph validation and run submission go through explicit host capabilities. [S1; P1]

# 3. UI invariants and acceptance principles

“MUST” denotes a release requirement. “SHOULD” requires a documented deviation. An attractive screenshot does not waive these requirements.

| ID | Invariant |
|---|---|
| UI-INV-01 | Opening, importing, arranging, validating locally, or viewing a graph performs no mathematical execution or remote export |
| UI-INV-02 | No browser code schedules graph nodes, allocates compute, or implements mathematical verification |
| UI-INV-03 | Runtime facts and evidence come from the authenticated host, never imported labels or worker-supplied rich text |
| UI-INV-04 | Exact numeric input survives browser editing and serialization without floating-point coercion |
| UI-INV-05 | Every displayed result is bound to a specific host, workspace, run, revision, node mapping, and result reference |
| UI-INV-06 | Draft edits cannot mutate an active run or silently reuse an obsolete plan or approval |
| UI-INV-07 | Execution, verification, artifact availability, cleanup, and cost are separate visible facts |
| UI-INV-08 | Connection compatibility and a connected verifier are not proof that a mathematical claim has been established |
| UI-INV-09 | Imported documents, schemas, labels, plots, logs, and proof text are untrusted presentation inputs |
| UI-INV-10 | No provider secret, reusable grant token, or authenticated signed artifact URL is persisted in an editor document |
| UI-INV-11 | Unknown capability, missing metadata, or stale state is displayed as unknown, not converted to success |
| UI-INV-12 | A non-dragging, keyboard-accessible path exists for core authoring, inspection, and approval review |
| UI-INV-13 | Draft saving, server saving, result retention, and backup are distinct concepts |
| UI-INV-14 | Optional viewers and compute features cannot make the base library or UI catalog require cloud access |
| UI-INV-15 | Changing an engine, precision, assumptions, input binding, or target preference is a semantic request change |
| UI-INV-16 | A cancelled HTTP request or closed tab is not reported as cancelled backend computation |
| UI-INV-17 | Presentation plugins cannot introduce arbitrary scripts or expand host authority |
| UI-INV-18 | UI-only artifacts never claim to be executable MathKernel workflows or mathematical certificates |
| UI-INV-19 | All externally meaningful actions are explicit commands, never consequences of component mounting or hover |
| UI-INV-20 | Unsupported functionality remains visibly unavailable; mocks and planned nodes are never labelled production support |

The authority boundary is also an accessibility boundary. A user operating through a connection dialog or list view must receive the same warnings and authorization checks as a user dragging on the canvas. Alternate input methods must not take “simpler” execution paths with fewer checks.

# 4. Users, interaction modes, and primary journeys

## 4.1 Intended users

The first audience is an individual researcher or engineer using a local MathKernel installation, including a laptop with limited GPU memory and optional remote resources. The second is a mathematically curious user who can select concepts but does not know every Python signature. The third is an expert auditing a run's exact assumptions, evidence, or reproducibility.

These are interaction needs, not security roles. Access control is host-provided. An “Expert” toggle reveals detail but grants no extra privilege and never disables protective limits. A “Simple” view can hide advanced tuning fields while still showing the effective values and significant consequences in the review screen.

## 4.2 Four user-visible modes

**Author** edits a draft. **Inspect run** shows an immutable recorded run. **Compare** places two explicit revisions or runs side by side. **Inspect artifact** focuses on a result's presentation. The mode appears in the main header, with a stable breadcrumb back to the originating document.

Authoring and monitoring may be visible simultaneously, but the canvas must say which revision its badges describe. When a user edits a running graph, the run overlay remains tied to the frozen revision and becomes a separately labelled overlay or read-only tab. Never repaint a modified node with an older run's green success badge without a visible mismatch marker.

## 4.3 Minimum successful journeys

A first-use journey connects to the local host, discovers an available capability, adds a node without dragging, enters exact input, inspects advisory and authoritative validation, explicitly invokes a supported operation, and opens its result with evidence and assumptions. It must work without remote compute.

A composition journey builds a small multi-node draft, repairs a dimensional mismatch through explicit editing, saves it, asks a workflow-capable host to validate and plan, reviews the resolved scope, submits, and reconnects after closing the browser. This journey is blocked rather than simulated when `workflow_execute` is unavailable.

An audit journey starts from a run ID, identifies a candidate-only result, distinguishes verification failure from an inconclusive timeout, follows the supported claim path, opens an artifact with its information-loss label, and exports an inspection document without rerunning the mathematics.

## 4.4 Empty, incomplete, and failure journeys

An empty palette can mean an unavailable host, a permission-limited catalog, or filters excluding all entries. Show those causes separately. A missing engine should offer documentation or an operator-mediated setup route, not install packages. A broken graph import opens a bounded recovery report; valid parts are not silently executed or substituted.

Include deliberate “nothing happened” tests: opening a template, refreshing capabilities, selecting a GPU target, expanding a proof panel, and viewing a cached artifact must not trigger computation. Small UI actions are precisely where accidental background work is easy to overlook.

# 5. Information architecture and application shell

## 5.1 Primary layout

Use a desktop workbench with a top application bar, left catalog/navigation rail, central canvas or inspector, right properties panel, and collapsible bottom diagnostics/run drawer. The layout is familiar without copying Node-RED's runtime model. Node-RED's documented workspace uses a palette, wired nodes, and tabs; Studio adopts those interaction ideas rather than its message execution semantics. [S2]

The top bar contains workspace/connection identity, document name, draft/save state, current mode, Validate, Plan, Run, and the command menu. Run has a split menu only for genuinely supported scopes such as selected node, selected subgraph, or workflow. An ambiguous triangle icon with no scope label is insufficient.

The left rail switches between Operations, Objects, Fragments, and Runs. The right panel has Properties, Inputs/Outputs, Evidence, and Documentation tabs according to selection. The bottom drawer presents Problems, Activity, and resource/cost warnings. It must not become a second independent result browser with inconsistent selection state.

## 5.2 Navigation model

Proposed browser routes are `/studio`, `/studio/workspaces/:workspaceId/documents/:documentId`, `/studio/workspaces/:workspaceId/runs/:runId`, and `/studio/workspaces/:workspaceId/artifacts/:artifactId`. These are client navigation routes, not promised backend endpoints. Node and claim focus can use non-secret route parameters or fragments, with length bounds.

A deep link selects an entity only after host authorization. It never grants access, performs a run, expands remote data export, or changes a workflow. Unknown links show a not-found-or-not-authorized message without revealing whether a private entity exists.

Browser Back restores route and focus context without submitting commands. Reload reconstructs the authenticated host identity and document/run reference before restoring cached display state. A workspace change clears incompatible selections, pending requests, and private previews.

## 5.3 Panels, tabs, and responsive behavior

Panel widths and collapsed state are per-user presentation preferences. Drag-resizing also has keyboard buttons and numeric width controls. The central workspace must remain usable at 1280×720 CSS pixels with nonessential panels collapsed. A narrow screen defaults to a list-oriented graph and full-screen inspector instead of shrinking every control until it is unreadable.

Open tabs identify drafts versus runs using words and icons, not color alone. Closing a run tab does not cancel the run. Closing a dirty draft asks whether to keep its recovery copy or discard it; the browser's generic unload prompt is a last-resort supplement, not the primary saving interface.

## 5.4 Global status and notifications

Persistent warnings cover disconnected/stale host state, unsaved recovery, unresolved submission outcome, and outstanding resource exposure. Transient success notifications may disappear; unresolved safety or data-loss conditions must not. Notifications include a direct route to the affected run or document and do not steal keyboard focus.

A host-wide cleanup warning must remain discoverable after the user closes the originating graph. Its display does not implement cleanup: the associated action invokes the already authorized host operation, and the UI reports that operation's observed outcome.

# 6. Visual language and component anatomy

## 6.1 Design system

Use semantic tokens for surfaces, text, borders, focus, selection, warnings, failures, and candidate evidence. Provide light, dark, and high-contrast-aware presentations. The default can follow the operating system. Domain colors help navigation but must not imply stronger mathematical evidence, paid authorization, or execution success.

Use ordinary readable UI fonts from bundled or system resources. Mathematical display may use bundled typesetting assets with proper licenses; no remote font requests. The release package includes required licensed UI assets, not runtime downloads. Maintain spacing, typography, icon, badge, and empty-state components in a small shared library with visual regression fixtures.

## 6.2 Node anatomy

A normal node has a stable title, operation subtitle, input and output ports, a compact parameter summary, availability marker, and optional run/evidence strip. The drag handle is distinct from form controls and selectable mathematical text. Titles can be edited as labels, but the canonical operation ID remains visible in the inspector.

Do not embed a large editable form, full matrix, or live 3D canvas into every node. The node summarizes; the inspector edits; the viewer renders. A compact preview is opt-in and bounded. At distant zoom, hide detail rather than changing the meaning of status indicators.

Special node shells are limited to input/object bindings, mathematical operations, verifier operations, supported host control constructs, artifact outputs, and unresolved placeholders. Shared port and badge components preserve consistent behavior across all categories.

## 6.3 Edge anatomy

Solid directed edges indicate authored data dependencies. A distinct labelled style indicates context/assumption binding when the host declares such ports. Selection highlights, proposed rewiring, incompatible connections, and stale run overlays have separate presentation states. Evidence relationships appear in the evidence viewer, not as extra unlabelled mathematical data wires.

An edge label can show a type, shape, or source output name. Actual data content is revealed only through inspection, not automatically along every wire. Animated dashes are optional run decorations, never the only signal that work is happening, and are disabled with reduced-motion preferences.

## 6.4 Status vocabulary

Keep at least four compact concepts: configuration state, execution state, evidence summary, and result freshness. A node may simultaneously be configured, execution-complete, numerically checked, and stale relative to the current draft. A single colored dot cannot encode that combination honestly.

Use labels such as “Not run,” “Host validation pending,” “Candidate available,” “Numeric result,” “Check inconclusive,” and “Result from revision 12.” Reserve proof/certification labels for host-admitted evidence of the precise claim. Informational tooltips are available on focus as well as hover, with essential warnings repeated in persistent text.

# 7. Frontend architecture and dependency policy

## 7.1 Chosen application shape

Use a React and TypeScript single-page application built into static assets, with React Flow (`@xyflow/react`) as the graph-interaction library. Vite is the proposed build tool, not a production server or execution framework. TypeScript strict checking is required. Choose and lock compatible dependency versions during implementation rather than installing floating latest versions in an end-user launch command. [S1, S3–S4]

React Flow is MIT-licensed; paid examples or templates are not required for the design. Implement product-specific undo, persistence, schema interpretation, and evidence presentation against public library APIs. Do not copy licensed premium example code into a public MathKernel release without appropriate permission. [S1]

A server-rendered Next.js application is not required for this local workbench. A desktop wrapper is also unnecessary initially. A browser UI served by the user's existing host avoids introducing an Electron/Tauri lifecycle before the interface itself is stable. Packaging can evolve later without moving mathematical execution into the renderer process.

## 7.2 State partitions

Maintain separate stores for `ConnectionState`, `CatalogState`, `DraftState`, `PresentationState`, `RunObservationState`, and `ViewerState`. Treat host responses as scoped, revisioned observations. Do not merge live results into the mutable authoring document. Keep transient pointer/viewport activity out of the run journal projection.

A command reducer owns draft edits. Network commands are issued by an explicit command service invoked from user actions; React render functions, effects responding to selection, and graph layout code must not cause runs. Subscription effects may fetch permitted read-only metadata, but their side effects must be auditable and bounded.

A selector-based store avoids repainting the full graph whenever one progress value changes. Use React Flow through an adapter component; its internal node objects are not the persisted document schema. This prevents UI-library upgrades from redefining the portable authoring document.

## 7.3 Worker usage

Web Workers may parse bounded editor documents, build local search indexes, perform advisory graph-structure checks, and calculate canvas layouts. They do not execute MathIR, validate proof objects, run FFTs, reproduce certificates, or schedule jobs. Their inputs and outputs are typed messages with cancellation and revision IDs.

For layout, start with an explicitly chosen deterministic arrangement and add a vetted layout library only if required. React Flow documents integration with several layout libraries rather than supplying one universal mathematical layout. Layout is a reversible presentation operation; it must not reorder semantic lists or insert connections. [S5]

## 7.4 Dependency minimization

Use an existing accessible component primitive set or a small audited internal set, not several overlapping UI frameworks. A schema validator, state library, and test runner may be selected after compatibility testing. Their responsibilities must remain explicit. Dynamic remote schema fetching and arbitrary UI code from operation metadata are prohibited.

Start expression editing with a capable textarea plus safe typeset preview. A large code editor is optional and lazy-loaded only if expert editing needs justify it. Do not add a JavaScript execution console. Bundle assets locally, document licenses, generate an SBOM at release, and test with external networking disabled.

# 8. Operation catalog and presentation metadata

## 8.1 One semantic source, bounded presentation overlays

The host catalog remains authoritative for capability identity, supported inputs/outputs, parameter validation, operation availability, and execution support. Studio maintains presentation hints keyed to that identity and schema digest: title, category, icon, preferred widget, help anchor, and compact summary layout. These hints cannot change requiredness, accepted values, arithmetic semantics, or evidence promises.

A catalog response includes host identity, workspace scope, catalog revision, observation time, entries, and paging/capability information. Cache it for offline discovery, but mark the cached snapshot and disable actions requiring fresh authority. An absent operation may be unavailable, permission-filtered, or removed; the host supplies a public-safe reason where appropriate.

## 8.2 Proposed descriptor

The following is interface notation, not a claim that the baseline exports these fields:

```typescript
interface StudioOperationDescriptor {
  operationRef: string;
  operationVersion: string;
  catalogRevision: string;
  schemaDigest: string;
  title: string;
  domain: string;
  descriptionText: string;
  availability: 'available' | 'unavailable' | 'experimental' | 'unknown';
  composition: 'complete' | 'partial' | 'documentation_only';
  parameterSchemaRef: string;
  inputPorts: readonly StudioPortDescriptor[];
  outputPorts: readonly StudioPortDescriptor[];
  engineChoices: readonly HostChoice[];
  requiredFeatureIds: readonly string[];
  evidenceDescription: readonly HostClaimDescription[];
  presentation: SafePresentationHints;
}
```

Referenced interfaces are UI-facing data contracts completed during host integration. They must be runtime-validated, not accepted because TypeScript compiled. A `descriptionText` is plain text or a restricted documented markdown subset. An icon is an allowlisted built-in identifier, never an arbitrary SVG or URL. Schemas resolve through the configured host/catalog, never through an untrusted `$ref` network fetch.

## 8.3 Search and discovery

Search titles, operation IDs, domain names, aliases, and descriptions locally over the authorized catalog. Support keyboard selection, recent operations, favorites, compatible-input filtering, and explicit unavailable/experimental filters. Favorites record operation identity, not a copy of its signature.

A compatibility-filtered palette must distinguish “Known compatible” from “Compatibility not determined.” Unknown entries may remain available under an explicit section; they must not vanish so completely that an incomplete catalog looks like a mathematical impossibility. Search ranking never upgrades availability or suggests unsupported GPU acceleration.

A catalog result opens a detail sheet showing input/output contracts, parameter coverage, engines, examples, evidence caveats, and required optional components. Example values are inert. Adding an example graph requires an explicit authoring action and does not fetch datasets or initialize an engine.

## 8.4 Capability drift

On a catalog revision change, compare each referenced operation's version and schema digest. Cosmetic metadata can refresh without modifying the draft. Semantic differences mark affected nodes for review and invalidate relevant host validation/plan bindings. Never silently rename ports and reconnect them by position.

Removed operations remain unresolved placeholders retaining their original IDs, settings, and connections. The user can replace them through a diff-and-remap dialog. A suggested replacement is a new user decision, not a guarantee of equivalent mathematics. Export preserves the unresolved original data for recovery.

# 9. Studio authoring document and identity model

## 9.1 A UI document, not an execution IR

Use `mk.studio/1` as the proposed editor-document schema, with the extension `.mkstudio.json`. This avoids preemptively assigning `.mkflow` to a future host-owned executable workflow format. The document records human authoring intent and presentation. It cannot be executed without host resolution, validation, planning, and authorization.

The document has `identity`, `authoring`, `presentation`, and nullable `host_binding` sections. `authoring` contains draft nodes, operation references, parameter drafts, scoped input-reference intents, connections, and selected desired outputs. For serialization convenience, each node also stores its cosmetic `label`; changing that label increments only the presentation revision and is excluded from the host's semantic projection. `presentation` contains positions, group frames, and viewport. Panel preferences remain in per-user UI settings rather than this initial portable schema. `host_binding` is a non-authoritative reference to a host-saved definition/revision, never an authorization credential.

A document may be incomplete or invalid while being edited. A parameter draft can preserve text that has not yet parsed. Submission uses only host-resolved valid values, never the UI's incomplete field buffers. Saving an invalid draft is useful; calling it a runnable workflow is not.

## 9.2 Identity and revision rules

Assign stable opaque IDs to documents, nodes, edges, and visual fragments. Labels are not identifiers. Renaming a node must not break its connections or reassign result history. Copying nodes generates fresh IDs and remaps internal connections in one transaction; links to outside nodes require explicit retention or rebinding.

Track a monotonic local draft revision and a separate layout revision. Layout changes do not by themselves invalidate mathematical validation. Operation, connection, input, parameter, assumption, precision, or target-preference changes do. The host computes the authoritative semantic identity; the browser's revision is only correlation information.

A result mapping binds a host run's node identity to the authoring node ID and frozen revision. It may be one-to-many when the host expands a supported construct. The UI displays the supplied mapping and never guesses a correspondence from titles or graph positions.

## 9.3 Minimal document shape

```json
{
  "schema": "mk.studio/1",
  "identity": {"document_id": "example-document", "title": "Untitled experiment"},
  "authoring": {
    "revision": 1,
    "nodes": [],
    "edges": [],
    "desired_outputs": []
  },
  "presentation": {
    "revision": 1,
    "node_positions": {},
    "groups": [],
    "viewport": {"x": 0, "y": 0, "zoom": 1}
  },
  "host_binding": null
}
```

The companion schema and fixtures further specify the proposed editor shape. They are not MathKernel execution schemas. General JSON schema validation cannot establish operation-specific mathematical correctness.

## 9.4 Portable and nonportable references

An object reference identifies a host and workspace as well as an object ID. Cross-host import produces an unresolved binding unless the host explicitly imports an authorized object bundle. Matching IDs or content digests alone do not confer access. Local filesystem paths, signed URLs, and credentials are not portable bindings.

Run links may be exported as opaque historical references with origin information, but no live result status is trusted on import. An exported screenshot or cached summary is labelled historical presentation until the authenticated host re-resolves it. A user-created title containing “FORMALLY PROVED” remains a title, never a badge.

# 10. Canvas editing and command semantics

## 10.1 Core authoring actions

Support adding from the palette, click-to-add, keyboard insertion, selection, multi-selection, move, duplicate, delete, connect, reconnect, disconnect, align, distribute, fit view, and search-to-focus. Every action uses the same command reducer regardless of input method. Mouse-only shortcuts are conveniences, not exclusive routes.

Node movement commits one undoable command per gesture, not hundreds of intermediate positions. A multi-node delete previews incident connections and any output selections affected; deletion is one transaction. It never deletes host objects, cancels runs, revokes evidence, or destroys remote artifacts. Those actions require separate explicit controls outside canvas editing.

## 10.2 Undo and redo

Undo/redo applies to authoring and presentation commands only. It does not undo execution, approval, export, publication, or provider allocation. After a run is submitted, undoing a parameter edit cannot reverse that submitted job. Show that distinction where users might otherwise expect rollback.

Group rapid text editing into field-level transactions with explicit commit/blur semantics, but preserve recoverable text while the field is invalid. Undo must restore IDs, edge endpoints, input-binding modes, and parameter values exactly. Redo does not replay network side effects. Host-save acknowledgements update save state rather than becoming undo entries.

## 10.3 Wiring interaction

Users can drag from an output port, click a source then a destination, or open “Connect output” and choose a destination from a searchable list. Candidate ports show direction, name, mathematical type, shape, and compatibility state. On rejection, retain focus and provide a reason tied to both endpoints.

Reconnecting an edge is atomic: the original remains until the replacement is confirmed. A failed or cancelled gesture does not leave an accidental disconnected graph. Connecting to an occupied single-input port requires explicit Replace, while collection ports use declared cardinality and ordering. Wire order is never inferred from screen position.

An explicit conversion operation can be suggested when the host catalog advertises it. The user sees the conversion's effect and accepts insertion as one command. Never silently cast exact rationals to floating point merely to make a wire turn green.

## 10.4 Layout and graph navigation

Auto-layout previews or applies reversible positions only. It respects pinned nodes, groups, and text direction, and can be cancelled. Repeated layout on an unchanged graph should be stable for the same layout version/settings. Large layout tasks run in a worker and cannot block Run status updates or safety warnings.

Provide an outline/list view, upstream/downstream navigation, breadcrumb navigation into supported subworkflows, and “Reveal source” from a diagnostic or result. The minimap is optional. Finding an offscreen invalid node must not require manually panning through a large diagram.

## 10.5 Interaction conflicts

Wheel and trackpad behavior is configurable and documented. Browser page zoom must remain available. Text selection, field editing, sliders, and audio controls must not accidentally drag nodes. Escape cancels the current gesture before dismissing panels. Deletion shortcuts are inactive inside editable fields. Shortcut dispatch respects composition/IME events and operating-system conventions.

# 11. Typed ports, contexts, and compatibility feedback

## 11.1 Port display contract

Each port descriptor has a stable ID, direction, display name, host type reference, cardinality, optionality, and connection constraints where supplied. A type presentation may include container kind, scalar domain, symbolic shape, units, field/ring identity, basis, context, and arithmetic mode. The host type reference remains authoritative when the UI cannot fully render it.

For example, `Matrix[Q, m×n]` and `Matrix[R64, m×n]` must not be presented as interchangeable exact values. `GF(p)` and `GF(p^k)` need more than the same display color. Two finite-field representations may require an explicit isomorphism or basis conversion; Studio does not infer that conversion.

## 11.2 Four-state compatibility

Use `compatible`, `incompatible`, `requires_check`, and `unavailable`. `compatible` means the declared port contracts match at the displayed validation scope, not that execution will succeed or that the result is proved. `requires_check` covers symbolic dimensions, incomplete metadata, context obligations, and data-dependent conditions. `unavailable` covers inaccessible or missing capabilities.

Client checks may compare exact known dimensions, direction, cardinality, and explicit host type identifiers. More complex mathematical reasoning is delegated. A UI worker is not a symbolic unification engine. The connection may be saved as unresolved, but the host must reject invalid submissions.

## 11.3 Shapes and collections

A dot-product node with known lengths 200 and 256 should expose an incompatibility before execution. If one length is symbolic `n`, show the unresolved equality rather than inventing a value. Batched tensors display batch axes separately where the host contract supplies them. Broadcasting is never assumed from generic array shapes.

Collection ports state order, duplicate behavior, and empty-collection semantics. Adding another wire to a list input must not convert an unordered set into an ordered vector silently. Variadic inputs use named slots or explicit host-defined collections; rearranging slots is a semantic edit.

## 11.4 Context and assumptions

The inspector shows the effective context, explicitly selected assumptions, inherited assumptions reported by the host, and outstanding obligations. A graph-level visual annotation does not automatically impose assumptions on every operation. Actual context binding must be represented by host-supported inputs or settings.

Switching from a real to a complex domain, changing a branch convention, or importing a measurement with different units invalidates affected validation and planning. The UI displays the returned scope. It cannot remove an assumption from a result by hiding the corresponding node or collapsing a group.

## 11.5 Arithmetic transitions

A requested exact-to-numeric conversion gets a visible transition marker and precision summary. Numeric-to-exact rationalization must state what new numbers are being defined; it cannot retroactively make measured input exact. A host-supplied interval claim is displayed with its interval semantics and source, not inferred from square brackets around two numbers.

Edge previews must preserve provenance and information loss when data was sampled, rounded, projected, filtered, or aggregated. A selected display precision changes formatting only; an operation precision parameter changes the mathematical request. These controls must be labelled differently.

# 12. Node inspector and mathematical input editing

## 12.1 Inspector structure

The inspector contains Overview, Parameters, Bindings, Execution preferences, Outputs, Evidence, and Help sections as applicable. Required fields appear first, advanced fields are collapsible, and effective defaults remain inspectable. A mixed multi-selection shows shared editable presentation properties; it does not blindly apply mathematical parameters to unrelated operation schemas.

Validation messages attach to stable field paths. The summary identifies the first blocking issue and can focus it. Changing a field preserves unrelated settings and reports any dependent fields invalidated by the change. The UI must not auto-submit on blur or on pressing Enter inside a mathematical expression editor.

## 12.2 Exact number editor

Integer and rational editors retain decimal text and use MathKernel-compatible string representations. Never round through JavaScript `Number`, HTML numeric-input coercion, JSON numeric literals, or a generic spreadsheet grid's floating storage. BigInt may support local bounds checking but must be serialized through the declared string codec, not generic JSON conversion. [B3]

Support signed integers, numerator/denominator entry, explicit zero-denominator errors, and copy of full exact values. Locale display formatting is separate from canonical input. A Dutch decimal comma is either supported through an explicit locale-aware numeric field or rejected with an explanation; it is not silently interpreted as a list separator.

For approximate values, show declared precision, numeric representation, and special-value support. Nonfinite values require a host-supported tagged encoding. A blank field, missing value, null, zero, and the literal text “NaN” are distinct. Tests include integers beyond 2^53 and very large numerator/denominator strings.

## 12.3 Expressions, matrices, and typed objects

Expression editing accepts the host's supported notation and requests host parsing/normalization when needed. The browser may typeset a preview but does not execute the expression or replace it with an independently parsed CAS tree. Raw entered text and normalized host interpretation can be compared before submission.

Matrix entry supports bounded cell editing, pasted tables, row/column counts, exact/numeric mode, and explicit shape errors. Larger inputs use an authorized object picker or upload workflow. A paste preview shows delimiter, decimal convention, inferred dimensions, and rejected cells before creating a binding. No CSV formula evaluation occurs.

Object pickers show type, size/shape summary, origin, current accessibility, and source evidence where available. They do not enumerate the user's whole filesystem. File selection authorizes only the selected upload action; exporting the uploaded data to remote compute is a separate host policy decision.

## 12.4 Schema-driven controls

Use safe widgets for enums, booleans, strings, bounded numeric text, arrays, objects, unions, and host-defined typed references. Requiredness and defaults come from the actual schema. Defaults are presented as defaults, not automatically inserted as user intent when omission has different semantics. JSON Schema annotations and validation constraints are distinct. [S6]

Unknown schema constructs fall back to a bounded structured-text editor with “Host validation required,” or documentation-only mode when safe submission is unavailable. Schema metadata cannot supply React component names, scripts, dynamic imports, CSS, event handlers, or arbitrary network resources.

## 12.5 Typeset preview safety

Use KaTeX or an equivalently reviewed local renderer with untrusted commands disabled, finite expansion/size limits, escaped errors, and isolated macro state. KaTeX documents `trust`, `maxExpand`, and `maxSize` controls and warns about persistent macro trust boundaries. These are presentation safeguards, not mathematical validation. [S7]

A malformed expression must fail inside its preview boundary without crashing the graph or showing raw HTML. Long expressions receive truncation with an explicit full-text view. Copy actions distinguish source text, normalized MathIR text, and typeset display so a screenshot is not mistaken for machine-readable input.

# 13. Groups, reusable fragments, and subworkflow presentation

## 13.1 Three concepts that must remain distinct

A **visual group** is a frame with a title and optional annotation. It organizes nodes but has no execution semantics. A **visual fragment** is a reusable authoring template that inserts a reviewed set of draft nodes and connections with fresh IDs. A **subworkflow reference** points to a host-defined compositional unit with an actual callable interface. These are different objects even when each can appear as a collapsed box.

The palette uses separate icons and descriptions for fragments and host subworkflows. A collapsed group is not a reusable algorithm; a fragment is not automatically versioned executable code. Node-RED's subflow interaction is a useful reference for opening and navigating nested components, not a license to inherit its runtime semantics. [S2]

## 13.2 Visual groups

Creating, moving, collapsing, resizing, and deleting a group changes presentation only. Deleting the frame does not delete its members unless a separate command explicitly requests that. Crossing edges remain visible through boundary summaries; collapsing a group does not hide an invalid input, failed verifier, candidate result, or unresolved cost warning.

A group-level execution preference is allowed only as an explicit bulk-edit action or a host-supported policy binding. Moving a node into a visually marked “GPU” frame cannot change its target. The inspector shows the effective property on each member and reports overrides or unsupported nodes. Group labels alone never determine execution.

## 13.3 Fragment lifecycle

Save a selection as a fragment after identifying external inputs and intended outputs. The fragment includes operation references and schema identities, bounded parameter drafts, internal wiring, documentation, and layout. It excludes authority, result payloads, private resolved object bindings by default, active run state, and provider credentials.

Insertion presents missing operations and external bindings before editing the current draft. Internal IDs are remapped; external references remain explicit unresolved inputs unless the user chooses a permitted binding. An inserted fragment is a copy, not a hidden live dependency that changes when its original is edited.

Updating a saved fragment does not mutate previously inserted graphs. Offer a compare-and-replace operation with a full field/connection diff. A renamed port is not matched solely by display name. Recursive fragment expansion and excessive nesting are rejected under document limits.

## 13.4 Host-owned subworkflows

When the host advertises subworkflows, Studio displays their exported port contracts, version, digest, capability requirements, and read-only or editable status. Navigation uses breadcrumbs and a boundary view showing external-to-internal mappings supplied by the host. Editing a definition creates a new draft/revision rather than changing a running instance.

Recursion, loops, map/reduce, branching, and stateful nodes remain host-defined capabilities. If unsupported, the UI shows their imported references as unresolved. Do not implement a JavaScript loop to make a “Repeat” node appear functional. A host-expanded batch is displayed through aggregate views and paged item inspection, not millions of browser nodes.

# 14. Validation, diagnostics, and repair interactions

## 14.1 Three validation layers

**Editor integrity** checks the document schema, ID uniqueness, dangling connections, bounded sizes, and valid presentation structures. **Advisory compatibility** checks only explicitly known port metadata, required bindings, and schema constraints. **Host validation** establishes operation admissibility, mathematical constraints, context consistency, executable workflow support, and relevant policy conditions.

The Problems panel identifies the layer and snapshot revision for every diagnostic. “Editor checks passed” must not be abbreviated to “Verified.” A host result for revision 17 cannot clear revision 18's problems after the user changes a field. Late responses are stored or discarded by correlation rules, never applied indiscriminately.

## 14.2 Diagnostic contract

Each diagnostic has a stable code, severity, plain-language message, source layer, node/edge/field path, related entities, applicable revision, and optional host-provided repair choices. Severity distinguishes blocking error, warning, information, and unresolved check. An unresolved check is not a successful validation with an insignificant footnote.

Messages provide concrete comparisons: “Input requires a square matrix; supplied shape is 3×4,” or “These inputs have different field representations; no conversion is selected.” When metadata is insufficient, say so. Do not fabricate a precise explanation from an unknown host exception.

Repair choices are safe authoring commands, such as binding a missing object, selecting an available engine, inserting a registered conversion, or editing a known dimension. They are never shell snippets, arbitrary JavaScript, automatic downloads, or remote resource provisioning. A repair that changes mathematics displays its consequences before application.

## 14.3 Evidence requirements during authoring

A user may select desired claim requirements only from the host's supported policy vocabulary. The UI displays host-reported expected evidence or missing support paths. A connected interval operation does not establish that the final output will be interval-certified. A proof node is not proof until the host accepts evidence for the bound proposition.

Use “Planned check,” “Host predicts support for this claim,” or “Requirement not yet established.” Suggestions such as “Add an enclosure verifier” are permissible only when the host identifies an applicable registered operation and its prerequisites. The browser must not promise automatic Lean translation or certification of arbitrary expressions.

## 14.4 Validation timing

Run cheap editor checks incrementally in a bounded worker. Host validation is debounced, cancellable as a request, and scoped to an explicit draft snapshot. A host that treats parsing/validation as resource-consuming must declare limits; Studio must not run unbounded validation on every keystroke. Provide manual validation when the automatic path is unsupported or expensive.

Validation requests do not upload large source datasets or start mathematical jobs without an explicit host-defined action. If validating data requires an upload or costly analysis, expose it as a separate reviewable command. An advisory “unknown” is preferable to hidden computation.

## 14.5 Cycles and disconnected components

Editor integrity can detect a graph cycle structurally, but whether a cycle is permitted is host-defined. The initial UI supports ordinary acyclic operation compositions when the host does. Imported unsupported cycles remain inspectable with a blocking diagnostic; the editor does not silently remove an edge to force a DAG.

Disconnected components may be valid drafts. The Plan review shows the actual requested output roots and execution scope returned by the host. Do not imply every node on the canvas runs when the user requests one output, nor prune components without telling the user. Side-effecting operations require especially explicit inclusion and authorization.

# 15. Run controls, preview behavior, and revision freezing

## 15.1 Explicit action sequence

The normal workflow is Edit → Validate → Plan → Review → Authorize where required → Submit → Observe → Inspect. The UI may streamline repeated local actions only when the host exposes a bounded preauthorization policy; it must not replace these semantic boundaries with a generic “always allow” preference.

Plan returns a host-owned immutable reference, digest, resolved operation versions, input bindings, selected outputs, execution scope, assumptions, precision, relevant target choices, evidence requirements, and authorization needs. The browser displays that snapshot. Run submits the host plan reference and applicable authority reference; it does not resubmit a mutable draft object as if it were the same plan.

## 15.2 Scope controls

Supported scopes can include a standalone operation, selected subgraph, or declared workflow outputs. The host decides whether upstream dependencies are included, reusable results are eligible, and a selected node is independently executable. Studio displays the resolved scope before submission.

A standalone operation requires explicit existing inputs; it is not a loophole for the browser to successively compute all predecessors. “Run to here” is disabled unless the workflow host supports it. “Run all” is not provided merely because the UI can enumerate nodes.

## 15.3 Preview modes

Distinguish three actions. A **view preview** formats already available data and is noncomputational. A **host preview** invokes a supported bounded operation and requires the appropriate local/export/cost policy. **Live recomputation** is optional, off by default, and available only through a host-managed preview session with strict declared limits.

Moving a slider always updates its draft value. It does not automatically launch a GPU job. With an explicitly enabled preview session, changes can request debounced host previews; stale preview results remain visibly tied to their input revision. Heavy or remote operations require explicit Apply/Run unless the host has independently authorized a bounded policy.

Closing a preview panel stops subscription and requests preview-session termination where supported. It does not claim computation stopped until the host confirms. The preview label must remain visible on artifacts and must not be confused with the saved reproducible run.

## 15.4 Submission ambiguity

Disable repeated submission while a command is in flight, but do not rely on a disabled button as idempotency. Pass the host-supported client request ID, retain it for correlation, and query the host after a lost acknowledgement. Studio cannot guarantee exactly-once execution and must not retry mutations automatically through a query library or network interceptor.

An unknown outcome shows “Submission outcome unknown; checking existing request,” with no green success or immediate resubmit offer. If the host cannot resolve it, expose the host's reconciliation route and residual risk. A user can leave the page without losing the visible request correlation, but the host remains responsible for durable intent.

## 15.5 Editing during execution

Submit freezes the host revision. Subsequent edits create a different draft. The UI offers “Inspect running revision” and “Continue editing draft,” with an optional comparison. Moving nodes is allowed as a presentation overlay without changing the run binding; mathematical edits cannot rewrite historical labels, values, or assumption summaries.

Cancel, retry, reverify, resume, and reexecute are separate host commands, each with scope and any required approval. Resume is unavailable without a host-compatible checkpoint. Retry does not mean repeat identical billing authorization unless the host policy actually permits it.

# 16. Run monitoring, event handling, and reconnect behavior

## 16.1 Run view

The run view starts with run identity, host/workspace, frozen revision, requested outputs, creation time, and provenance of its plan. It contains a graph projection, node/attempt list, timeline, results, and lifecycle summary. A host may execute several attempts for one node; show the attempt history rather than overwrite it with the last result.

The graph projection is a visualization of host observations, not an execution controller. A node's progress can be indeterminate, estimated, or measured. Estimated progress and ETA identify their source and observation time. The UI must not invent percentages from elapsed time or count every batch shard as equal work without host support.

## 16.2 State axes

| Axis | Examples | Presentation rule |
|---|---|---|
| Execution | queued, running, output ready, failed, submission unknown | Shows computation/dispatch state only |
| Verification | pending, passed, failed, inconclusive, unsupported | Bound to a specific checked claim |
| Artifacts | available, partial, fetching, expired, unavailable | A result reference is not proof of retained bytes |
| Resources | not owned, active, release requested, confirmed, unknown | Never equate run completion with cleanup |
| Cost | reserved, accruing, estimated final, reported final, unknown | Never display missing usage as zero |
| Freshness | current, stale, reconnecting, permission lost | Orthogonal to every other axis |

These axes are rendered from the host contract and the compute design where available. Legacy local jobs expose less detail; mark fields not reported rather than manufacture remote-style guarantees. [P1 §10]

## 16.3 Snapshot and stream contract

Prefer ordinary authenticated HTTP for commands and one scoped server-sent event stream for observations. SSE has standardized event IDs and reconnect behavior, but application replay, authorization, deduplication, and gap handling remain explicit host/UI requirements. A WebSocket is optional only where an existing host already requires bidirectional interactive transport. [S8]

Obtain a snapshot with an event cursor/watermark that permits a gap-free continuation. The host must either replay after that watermark or supply a full resynchronization instruction. UI events carry host/workspace/run scope, event ID, entity revision, observation timestamp, and validated payload. Unknown event kinds trigger safe ignore plus compatibility diagnostics, not execution.

After reconnect, deduplicate events, reject scope mismatches, and prevent lower entity revisions from overwriting newer observations. Preserve out-of-order events for diagnostic display only when bounded. A cursor outside retention triggers a fresh snapshot; the browser cannot reconstruct missing truth by replaying its cached animations.

## 16.4 Connection and authentication changes

When connectivity fails, freeze the last observed facts and display their age. Do not mark running jobs failed because the browser is offline. When authentication expires, stop streams and private fetches, clear sensitive previews according to policy, and present reauthentication. Do not repeatedly refresh tokens or relaunch commands from component remounts.

Switching host or workspace cancels outstanding UI requests and clears scoped query caches. Late responses from the previous connection are rejected. Restoring a tab must verify the current host identity before applying any stored cursor or run link.

## 16.5 Event-volume control

Coalesce high-frequency progress updates per entity for rendering while preserving authoritative state transitions and warnings. Do not accumulate unlimited log messages in React state. Use bounded log pages with explicit gaps and a pause-scroll option that does not pause the backend.

The UI may use a snapshot/polling fallback with a disclosed refresh interval when streaming is unavailable. Polling failure does not cause run retry. Browser background throttling means observation can pause; the host continues independently. Returning to the tab requests a current snapshot before presenting controls based on stale status.

# 17. Evidence, assumptions, and provenance inspection

## 17.1 Preserve the actual evidence model

The baseline `TrustLevel` values are `formal`, `exact`, `symbolic`, `interval_certified`, `numeric_high_precision`, `numeric`, `empirical`, `heuristic`, and `unknown`. `ResultStatus` is a different enum and includes statuses such as `verified_numeric`, `candidate`, and `unsupported`. Studio must not invent a new trust level named `VERIFIED_NUMERIC` or conflate semantic status with evidence grade. [B3]

Show host-admitted trust, semantic status, engine, arithmetic transitions, assumptions, side conditions, and per-claim evidence. Imported files and raw candidate envelopes can display claimed labels only under “Untrusted source metadata.” They cannot populate the normal certification badge components.

## 17.2 Claim-first presentation

A result may establish one property but not another. The evidence panel lists each requested or supplied claim, its exact statement or structured description, outcome, required assumptions, verification method, and supporting records. It distinguishes feasibility from optimality, individual roots from completeness, residual checks from rigorous error bounds, and a rendered interval from a certified enclosure.

A compact summary says “2 of 3 requested claims established; completeness unresolved” when the host supplies that information. It does not calculate a global percentage of mathematical truth or label the whole graph proved because one subclaim has a formal certificate.

## 17.3 Evidence graph versus authoring graph

Open an explicit Evidence view separate from the editable workflow. It displays required support paths, alternative support paths, diagnostic observations, and input ancestry as supplied by MathKernel. A selected claim highlights its dependencies. Read-only navigation can return to the producer node or source object without turning an evidence edge into an execution connection.

Do not flatten the model into the minimum or maximum badge on all ancestors. A valid independent local certificate can establish a claim without depending on an untrusted remote candidate's authority; conversely, approximate input ancestry cannot disappear just because later arithmetic was exact. The host reconciles those roles. [B4; P1 §21]

## 17.4 Formal and numerical detail

Formal evidence inspection shows the bound proposition, assumptions/allowed axioms when reported, checker/toolchain identity, outcome, and relevant artifact references. Proof source is displayed as inert text. Opening it does not invoke an elaborator, execute tactics, install Lean, or compile remote objects.

Numerical evidence shows the checked input version, norm, residual, scale, tolerances, precision, and conditioning information when supplied. Missing diagnostics remain missing. Display precision must not make a nonzero residual appear to be exactly zero; use scientific notation or a “rounded display” indicator with full-value access.

Interval evidence identifies the checked enclosure, engine/verifier, interval endpoints and openness, and assumptions. A host compatibility warning or unresolved certification issue is not hidden because the UI prefers a clean badge. Studio cannot repair an upstream evidence defect by relabelling it.

## 17.5 Evidence persistence and export

A pinned result references the host result identity and retains its displayed revision. A static evidence export states whether it includes full evidence, bounded summaries, or external references. It is an inspection record, not a newly issued certificate. Reverification is a separate host action with its own limits and result.

When evidence records are unavailable or access is revoked, keep the historical reference with an unavailable marker. Do not preserve a formerly green badge without indicating that current inspection cannot resolve its supporting material.

# 18. Results, objects, and safe artifact viewers

## 18.1 Result receipt handling

The baseline output policy can return a receipt containing a resource ID, size, hash, and original-trust metadata rather than the full mathematical result. Studio must recognize that shape and fetch authorized pages before presenting complete evidence. `original_trust` in a receipt is not itself a new accepted result. [B5]

Use bounded host result summaries for the run overview. Full payloads load on demand under byte, depth, and pagination limits. A partial fetch shows “Partial result loaded” with the missing range and does not imply all assumptions or errors have been examined. An expired resource is not a mathematical failure and must not trigger automatic recomputation.

## 18.2 Viewer routing

Route by an allowlisted artifact schema/media type and the host's validated descriptor, not by filename extension alone. Provide safe viewers for scalar/expression summaries, exact matrices, paged tables, text, structured evidence, supported visualization documents, images, audio, and read-only graph/projection artifacts. Unknown types offer metadata and an explicit download/open-outside-app route with appropriate warning.

A viewer header always identifies the source result, run/revision, admitted-versus-candidate status, completeness, and information loss. The mathematical content remains useful when a visual renderer fails: show bounded metadata, accessible text, or a table fallback instead of a blank panel.

## 18.3 Large data

Matrices and tables use paging or virtualization, fixed display budgets, explicit sample indicators, and exact-value copy. Sorting/filtering a loaded page must say “Loaded page only” unless the host performs a full-dataset operation. A preview of 100 rows is never presented as the whole dataset.

Browser-side display downsampling is allowed only as a clearly labelled presentation transformation over already available data; it cannot create a new mathematical result or feed a subsequent solver. Prefer host-provided preview representations with recorded information loss. New analysis, filtering used as mathematical input, or export of computed summaries uses an actual registered operation.

## 18.4 Candidate and trusted boundaries

Candidate artifacts are inspectable, but carry a persistent “Unverified candidate” treatment independent of the artifact's own text. A malicious document can contain the words “exact” or “formal”; those words are not parsed into badges. Candidate-to-accepted transitions occur only after a new authenticated host observation references an admitted result.

An artifact's metadata, hash, and publisher signature can describe identity and integrity, not establish its mathematics. Trust presentation must follow the linked host claim records. Opening a verified plot does not prove that every visual inference a user makes from it is supported.

## 18.5 Safe display technology

Prefer structured artifact documents rendered by bundled trusted components over arbitrary returned HTML. Inert images use verified media handling and bounded dimensions. SVG is not inserted directly into the parent DOM from an untrusted artifact. Text and logs use escaping, including control characters and terminal escape sequences.

Existing HTML/Three.js artifacts require a dedicated isolated viewer path. The safe initial fallback is inert source/metadata or a static rendering; active embedding is enabled only after the security contract in section 24 is implemented. Never make “Open visualization” equivalent to executing an arbitrary HTML file on the Studio application origin.

# 19. Visualization, sonification, and presentation controls

## 19.1 Reuse the existing artifact boundary

MathKernel already has a renderer-neutral visualization document composed of blocks, shared datasets, series, controls, and provenance. Its projection layer records source lineage and information loss; sonification has a separate renderer-neutral mapping/event model. Studio should consume these structures through versioned viewer adapters instead of inventing another plot or audio specification. [B6–B8]

Supported artifact viewers are registered by schema/version. A new viewer may expose a different arrangement or interaction while preserving the source document. Unsupported schema versions remain inspectable as metadata and inert structured text; they are not guessed into a superficially similar chart.

## 19.2 Presentation controls versus mathematical parameters

Panning a chart, changing a camera, selecting already included series, adjusting playback volume, or choosing display precision is presentation. Changing a sampling range, requesting another projection, recalculating a histogram, or changing a sonification mapping that requires regeneration is a mathematical/artifact-production request when the host so defines it.

The viewer labels controls as “View only” or “Recompute required” where the distinction is not obvious. A display filter must not silently become a new dataset used by downstream nodes. “Use selection as input” creates an explicit draft binding or registered transformation and requires host validation.

A projection or aggregation banner remains visible when the data shown is not the full original object. Logarithmic axes, normalization, clipping, sampling, and high-dimensional slices have inspectable parameters. The viewer must not make exact data look approximate without marking display rounding, or approximate data look exact by printing extra zeros.

## 19.3 Linked selections

Synchronize table rows, plot points, graph vertices, and audio events through existing source references or explicit synchronization mappings supplied in the artifact. A linked selection is an interaction message, not a new derivation. When mappings are incomplete, highlight only known matches and disclose missing linkage.

Use scoped viewer channels containing artifact ID, schema version, source reference, bounded selection ranges, and a sequence number. A viewer message cannot invoke Run, alter a grant, change a target, read arbitrary objects, or navigate the parent to an external page. The parent accepts only the exact allowlisted presentation message types.

## 19.4 Sonification behavior

Audio playback requires an explicit user action and starts at a conservative, user-adjustable level. Provide Play, Pause, Stop, seek, loop, mute, and a clearly labelled master stop. Changing tabs does not unexpectedly start playback. Unmounting a viewer disposes its audio graph and buffers; stopping UI playback does not claim to cancel host audio generation.

Show mapping from source quantity to pitch, gain, time, pan, or timbre, including scaling and clipping. A pleasant or audible pattern is not evidence of significance. Display the linked mathematical result and statistical/evidence caveats alongside the player.

No microphone permission, speech recognition, assistant trigger handling, or automatic audio transcription is part of this plan. Audio remains data. Provide text descriptions, source-event tables, and visual timing alternatives so core interpretation is not audio-only.

## 19.5 Viewers as bounded resources

Allow one active heavyweight 3D viewer and one active audio playback context by default, with explicit user expansion where supported. Dispose GPU buffers, canvas contexts, object URLs, event listeners, and decoded audio on closure. Detect WebGL loss and show a recoverable fallback without recreating an infinite failure loop.

Thumbnail generation is not silently delegated to paid compute. Existing thumbnails can be displayed; new rendering follows a supported host or bounded local presentation path. Export controls distinguish a screenshot, an existing artifact download, and an authoritative host artifact/replay export.

# 20. Compute-plan review and authorization UI

## 20.1 UI scope

This section specifies how Studio presents the contracts in MK-RHC-001. It does not implement resource estimation, cost accounting, provider integration, grant creation rules, provisioning, or cleanup. Those capabilities remain in the compute/authorization services. A host without compute-plan support omits the corresponding controls rather than fabricating estimates. [P1 §§7, 10, 12–13, 24]

A node can express a target preference such as local, configured target, or host-planned automatic selection. The effective engine and actual compute target are displayed separately. Target configuration and credential administration are operator functions; a graph import cannot add a server, change a provider account, or attach a secret.

## 20.2 Plan review screen

The review screen identifies plan ID/digest, expiry, bound draft/workflow revision, selected outputs, operation/engine versions, input scope, arithmetic settings, requested evidence, and concrete execution alternatives. Show rejected alternatives with the host's reasons, including insufficient per-device memory, unsupported operation, unavailable verifier, prohibited export, or unknown capabilities.

Resource summaries distinguish CPU units, host RAM, per-device GPU memory, device count, temporary storage, transfer size, queue/runtime limits, and evidence-check cost where reported. A four-GPU target is not displayed as one pooled-memory device unless the host's operation contract explicitly supports that topology.

Timing and pricing fields include their source, observation time, uncertainty, currency, and exclusions. Unknown is a first-class value. A local estimate is not a guarantee. Do not display a converted euro budget as enforceable when the host enforces a different native currency and has not provided an approved conversion basis.

## 20.3 Export disclosure

The export panel lists the actual host-resolved input closure, classifications, destinations, and byte totals. A collapsible detailed manifest allows inspection of ancestors, not just the selected root object. Sensitive names can be redacted in public display while preserving an authorized local detail view.

Approval wording separates computational execution, data export, provisioning, retries, and residual exposure. An institutional zero-price job can still consume allocation credits or export private data. “Free” must not be a shortcut around those permissions. When policy denies export, no UI confirmation button can override it.

## 20.4 Trusted confirmation interaction

The host presents an approval challenge bound to the plan digest and authenticated user/session. Studio renders it in first-party application chrome, never inside an artifact iframe or node-authored HTML. The UI submits the user's decision through the host's designated confirmation interface. A browser field named `approved=true` is not authority by itself.

The host enforces permission, expiry, revision, scope, and replay protection. A changed plan invalidates the challenge. Keyboard confirmation requires deliberate focus and action; the Enter key used to finish a mathematical input must not also approve a newly opened dialog. Imported workflows, agents, and logs cannot synthesize a trusted confirmation.

A host may support administrative policy grants, but their creation interface is outside the graph document. The first UI should prefer per-plan confirmation or selecting an existing permitted grant. It does not expose a generic “disable all checks” option.

## 20.5 Lifecycle warnings

After submission, show calculation, verification, resource release, and cost settlement separately. “Result ready; resource cleanup unknown” is a legitimate persistent state. A cancelled calculation may still have pending release or charges. Studio must preserve the compute service's distinction rather than replacing it with one success toast.

Actions such as “Request cancellation” and “Open resource details” invoke scoped host operations. Do not expose broad account deletion or provider-console API keys. A link to an operator's provider console is ordinary navigation, not proof that the correct resource has been terminated.

# 21. Saving, recovery, history, comparison, and export

## 21.1 Three persistence layers

Separate a local editor recovery copy, an authoritative host-saved workflow/document revision, and durable mathematical results/artifacts. Each has a different lifetime and owner. “Saved locally in this browser” is not “Backed up,” and “Run completed” is not “Artifact retained indefinitely.”

Use IndexedDB for bounded draft recovery when available. Browser storage can have quotas and be evicted; persistent-storage requests do not justify promising universal durability. Provide explicit file export and clear recovery status. Private browsing, policy restrictions, or quota errors must produce a visible fallback instead of silently losing edits. [S9]

Default recovery stores authoring data and presentation, not full datasets, secrets, result payloads, or transient signed URLs. A workspace may disable browser recovery of sensitive expressions; in that mode, use memory-only editing plus explicit authorized saving/export. Inform the user before they assume recovery exists.

## 21.2 Autosave and conflicts

Coalesce draft writes after edits and record the last successful local revision. A host save uses expected revision/ETag semantics where available. A conflict preserves both versions and opens a comparison; it never overwrites the remote revision just because the current tab is active.

Multiple tabs can share awareness through a browser-local channel, but this is a convenience, not a distributed lock. The host remains authoritative for revision conflicts. Browser recovery entries include host/workspace/document scope so changing ports or connecting to another installation does not overwrite an unrelated document.

Recovering a draft asks whether to restore, compare, or discard. Restoration never reconnects to unknown hosts, replays pending mutations, restarts previews, or consumes stored authority. Pending run references are resolved read-only against the host.

## 21.3 History and diff

Provide a structural diff for operation/version changes, parameters, input bindings, connections, assumptions, arithmetic, target preferences, and desired outputs. Presentation-only changes appear in a separate category. A moved node must not obscure a one-character change in a polynomial coefficient.

A run comparison identifies both frozen revisions and input identities before showing value differences. Numerical difference analysis beyond safe formatting is a host operation, not an improvised browser calculation presented as verification. Displaying side-by-side residuals already reported by the host is permissible; inventing a new statistical test in the UI is not.

## 21.4 Import pipeline

Import is inert: select file → check type/size → bounded parse → validate editor schema/version → inspect dependency and reference summary → preview changes → explicitly merge or open separately. Reject duplicate JSON keys, prototype-related property attacks, oversized nesting, unknown critical schema versions, and invalid IDs under the defined parser contract.

An editor document does not contain arbitrary executable code. Unknown operation references are retained as unresolved; unknown presentation fields follow an explicit extension policy rather than being evaluated. YAML may be supported later through a safe, constrained conversion, but the initial canonical editor format is JSON.

The importer strips or quarantines prohibited embedded credentials, approval assertions, live status claims, and executable viewer payloads. It reports exactly what was rejected or omitted. Never quietly convert a malicious `trust` string into a legitimate evidence record.

## 21.5 Export distinctions

“Export Studio document” produces the editable authoring/presentation JSON. “Export view” produces a visual snapshot labelled with draft/run identity and visible limitations. “Export result/artifact” retrieves an existing authorized host representation. “Export reproducibility bundle” delegates to the host and discloses missing external artifacts or unsupported capability.

No export silently reruns a job. Browser downloads use safe filenames and explicit user action. Host filesystem exports require an authorized scoped path chooser; do not pass arbitrary browser-supplied paths to legacy export methods. A copied link is an address, not an access capability.

## 21.6 Deletion and retention

Delete draft, remove local recovery, delete host document, remove a pinned link, cancel a run, and delete a retained artifact are separate actions. The first release may expose only the first three where host support exists. The UI must state which data is affected and which historical references will remain.

Removing a node from a draft does not destroy its past evidence. Clearing a browser cache does not cancel a remote resource. An exported document can refer to deleted host objects; on reimport those references become unresolved rather than being replaced automatically.

# 22. Connection onboarding, preferences, and compatibility

## 22.1 Local-first startup

The first launch connects only to the explicitly configured local or user-selected host. It does not scan the LAN, enumerate SSH hosts, probe cloud providers, or contact a MathKernel registration service. Serve the built UI from the user-owned host with a clearly shown host/workspace identity.

An unauthenticated local web page is not enough to protect a privileged kernel. The host must implement an appropriate bootstrap/session mechanism, origin checks, and scoped permissions before Studio can issue commands. A one-time operator-mediated connection step may be used; reusable credentials must not appear in URL query strings or browser storage.

## 22.2 Handshake and compatibility

The handshake returns host instance identity, workspace identity, MathKernel version, UI protocol range, supported feature IDs, catalog revision, schema/viewer versions, limit profile, authentication status, and available read/write actions. Negotiate conservatively. Unknown major protocol versions allow a safe explanatory screen, not best-effort mutation.

UI package version, editor schema version, host protocol version, mathematical operation version, and artifact schema version are separate. A user should be able to inspect all of them in About/Diagnostics. Version mismatches explain the affected feature; they do not produce generic “install latest everything” instructions.

## 22.3 User preferences

Preferences include theme, text size, canvas interaction style, default panel arrangement, display precision, locale, date/time presentation, reduced animation, audio volume, recovery policy, and notification behavior. They do not include provider API secrets or bypasses for mathematical/security limits.

A preference for a target or engine is merely a default request value. It cannot authorize paid automatic execution. A preference for numeric display precision cannot alter operation precision. Workspace-wide computation policies are host-owned and shown separately from personal UI preferences.

## 22.4 Offline and degraded modes

Distinguish “No Internet, local host available” from “Host disconnected.” A local host can still perform supported local mathematics without Internet; the browser cannot assume that all actions are unavailable. Conversely, a cached catalog with no host supports authoring and inspection of explicitly retained presentation data, not computation.

When a feature is unavailable, retain the user's work and show an exact reason: missing workflow service, unsupported artifact schema, unavailable engine, insufficient permission, missing compute adapter, or incompatible catalog revision. Avoid giant blocking dialogs when only one optional viewer is affected.

## 22.5 Operator-only settings

Credential references, trusted origins, provider targets, allowed artifact stores, authentication, and exposure policies remain operator settings. Studio can show their sanitized status and link to existing administrative documentation. A future administration UI is a separate scope and threat review, not a hidden tab populated from arbitrary backend configuration.

Do not expose MathKernel's general settings endpoint wholesale. A legacy permissive mode cannot be surfaced as “make graph work” if it bypasses bounds. The safe browser facade must define the exact allowlist of settings it permits.

# 23. Required host interfaces and dependency contracts

## 23.1 Nature of this specification

This section describes what the UI needs to observe and request. It does not prescribe a database, scheduler, execution algorithm, new workflow IR, or cloud service. Method names are proposed logical facade names; existing HTTP/MCP/embedded transports may implement equivalent behavior after a security and compatibility review.

The browser does not connect directly to a stdio MCP process. A user-owned host can adapt approved existing tools to scoped web endpoints, or expose an existing compatible transport. Tool descriptions are not a substitute for strong request validation and host-side authorization.

## 23.2 Facade surface

| Logical operation | UI input/output | External responsibility |
|---|---|---|
| `handshake()` | Capabilities, versions, limits, identity | Host authenticates and negotiates |
| `catalog.list/get()` | Scoped descriptors and schemas | Catalog publishes real contracts |
| `objects.list/get()` | Paged authorized object summaries | Kernel/store enforce identity and access |
| `drafts.load/save()` | Editor data and expected revision | Optional host document persistence |
| `workflow.validate()` | Frozen authoring snapshot → diagnostics/binding | External workflow service resolves semantics |
| `workflow.plan()` | Validated binding → immutable plan | External workflow/compute services plan |
| `approval.challenge/confirm()` | Host challenge → decision receipt | Authorization service enforces scope/authority |
| `workflow.submit()` | Plan reference + host authority reference | External runtime owns durable submission |
| `operation.invoke()` | Explicit supported standalone request | Existing operation host executes, with policy |
| `runs.list/snapshot/events()` | Scoped run state and cursor | Runtime publishes durable observations |
| `runs.cancel/retry/reverify()` | Explicit command reference | Runtime/compute host enforce action semantics |
| `results.summary/page()` | Accepted result or candidate receipt | Result service validates scope and pagination |
| `artifacts.describe/read()` | Descriptor, bounded data or opaque download | Artifact service controls access and type |
| `exports.create/read()` | Authorized existing representation | Host packages data without implicit rerun |

`workflow.*` is an external dependency absent until implemented and advertised. `operation.invoke` cannot be used internally as a substitute graph scheduler. Host adaptation must also prevent browser-originated arbitrary filesystem exports through existing path-taking APIs.

## 23.3 Common envelope

Every response carries protocol version, host/workspace scope, request correlation, server observation time, and a typed payload or structured error. Entity responses carry revision/version information. Money uses currency plus decimal string; mathematical exact values use their existing string codecs. Sizes and counts exceeding safe JSON integer precision use declared string forms.

A host-provided opaque reference is not placed into HTML unescaped or treated as a URL. Links are constructed by trusted application routing. External resource URLs, when unavoidable, come through a scoped host delivery mechanism and are never accepted from a node's text fields.

## 23.4 Runtime validation and schema delivery

Generate or validate TypeScript DTOs against versioned host schemas, but treat network data as unknown until checked. Compile known validators at build time or use an audited no-eval interpreter/subset. Do not enable `unsafe-eval` simply because a runtime schema library generates functions. Complex dynamic validation can remain host-side with explicit UI coverage labels.

Schema references resolve only within the authorized catalog snapshot. Bound recursion, array sizes, string lengths, and reference count. Operation-provided regular expressions or format plugins cannot execute unchecked in the browser. Unsupported validation keywords are reported, not ignored while claiming full validation.

## 23.5 Contract failure behavior

Permission failures clear inaccessible private data but preserve safe draft intent. Version errors disable mutations. Missing entities show unresolved references. Rate limits back off read-only requests. A command timeout becomes an unknown outcome when the host may have accepted it. A host-provided error is displayed as escaped text with a stable category, not a serialized exception to execute.

A contract fixture server is required for UI development. It must simulate incomplete catalogs, denied permissions, stale plans, duplicate/out-of-order events, unknown submissions, candidate results, expired artifacts, and cleanup uncertainty. It is visibly marked Test host and cannot be advertised as live MathKernel functionality.

## 23.6 Dependency acceptance

For workflow execution release, the host must prove the plan binds a frozen revision, outputs map to that revision, cancellation does not require an open tab, and result/evidence admission is authoritative. These are black-box integration gates, not an instruction for the UI team to implement the missing runtime.

When a dependency is missing, the milestone report lists it under Blocked integration. The valid fallback is an authoring/inspection release with Run unavailable for unsupported workflows. A polished mock graph is never evidence that this gate passed.

# 24. Browser security and artifact isolation

## 24.1 Threat boundary

Potentially hostile inputs include imported graphs, shared fragments, catalog prose, field labels, mathematical text, schema extensions, remote candidate metadata, logs, images, SVG, HTML, proof source, and artifact messages. The browser application also interacts with a privileged local host, so cross-origin web pages and installed extensions are relevant risks. A loopback address is not an authentication mechanism.

Use layered defenses: scoped sessions, strict host/origin validation, CSRF protection on state-changing requests, restrictive response policies, no arbitrary script execution, bounded parsing, safe rendering, and independent host authorization. OWASP's CSRF, HTML5, XSS, and CSP guidance informs these boundaries. [S10–S13]

## 24.2 Session and local-host protection

Keep provider credentials and host signing keys entirely outside the browser. Prefer host-managed short-lived sessions with appropriately protected cookies or an equivalently reviewed mechanism. Do not store reusable authentication tokens, grant secrets, or private keys in localStorage, IndexedDB, exported documents, or URLs. Browser storage is not a secret vault. [S10]

The host validates expected Host/Origin values, rejects unauthorized cross-origin mutations, applies CSRF defenses, and uses narrowly configured CORS only when required. Treat alternate loopback names, rebinding scenarios, and reverse proxies explicitly. SameSite cookies and CORS are not the entire authorization model. Production remote access requires an operator-secured HTTPS deployment; the UI must not recommend exposing an unauthenticated development server.

Connection bootstrapping must not trust a host URL supplied inside an imported graph. The operator chooses the host through trusted application configuration. Imported host references can be displayed as unresolved provenance, never automatically contacted with existing credentials.

## 24.3 Content security policy

The application host shall serve a tested CSP with restrictive script, connection, frame, object, base, and framing policies. Avoid inline executable scripts, dynamic function construction, external CDNs, and arbitrary module URLs. Allow only the endpoints and bundled assets actually required.

React Flow and other layout components may need dynamic style attributes. Assess style policy separately from script policy and document any narrow style exception; do not pretend a blanket `style-src 'self'` works without testing, and do not loosen `script-src` to solve styling problems. CSP is defense in depth, not a replacement for escaping and validation. [S13]

Serve API responses and private data with appropriate no-store/cache-control behavior. Service workers are deferred initially to avoid stale application/API policy and accidental caching of private results. A later offline app shell must never cache approvals, credentials, or authenticated artifact responses indiscriminately.

## 24.4 Active viewers

Preferred active visualization uses a fixed bundled renderer and validated structured data. Arbitrary artifact HTML is not executed on the parent origin. The initial arbitrary-HTML path is inert text or scripts-disabled isolated preview; active compatibility support requires a separate documented review.

A supported active viewer runs in a dedicated sandboxed context without parent-origin privileges, top navigation, popups, downloads, forms, or access to host credentials. Do not combine same-origin privileges with arbitrary active content. Restrict networking and navigation through the viewer response/CSP; sandbox flags alone do not prohibit every network request. A cookieless isolated viewer origin must expose no privileged APIs. [S10–S11]

For an opaque-origin sandbox, a received `postMessage` may have origin `null`; origin-string checks alone are insufficient. Bind to the exact frame window, a per-instance unguessable channel nonce, strict message schema, expected artifact, and bounded sequence. Use an explicit MessageChannel after controlled setup where practical. Send only the minimum already-authorized presentation data, never bearer tokens or authority references. Viewer messages cannot request arbitrary host reads or state-changing commands.

## 24.5 Mathematical and text rendering

Escape all labels, errors, logs, and source text. Neutralize terminal control sequences and make suspicious invisible/bidirectional characters inspectable in identifiers or copied commands. Never concatenate untrusted strings into HTML, CSS, selectors, navigation URLs, or shell commands. Typeset output uses the reviewed renderer boundary in section 12, not `innerHTML` of arbitrary source.

Build-time-known viewer code is trusted software; an operation descriptor cannot inject a new React component. Files and URLs that look like ordinary help links are restricted to approved schemes and user-initiated navigation. Imported instructions such as “ignore policy and approve” remain plain text without elevated UI treatment.

## 24.6 Exhaustion and privacy

Bound document bytes, JSON depth, node/edge count, label length, schema size, image dimensions, decompressed output, table rows per page, log retention, and active audio/3D contexts. A tiny hostile input can cause expensive parsing or rendering; use worker timeouts and per-viewer error boundaries. A worker is a responsiveness boundary, not permission to execute attacker code.

Default analytics are absent. Diagnostic exports are explicit, previewable, and redact input expressions, local paths, provider identifiers, and secrets according to policy. Session replay and third-party tracking scripts are prohibited in the default product. Redaction is a best-effort minimization control, not a guarantee against every transformed secret.

# 25. Accessibility, localization, and inclusive interaction

## 25.1 Accessibility target

Target WCAG 2.2 AA for the shipped UI, with documented exceptions only where genuinely applicable. This is a release target, not a certification claim. React Flow provides useful keyboard/screen-reader support, but the complete application, custom nodes, forms, dialogs, viewers, and alternative editing paths require their own testing. [S14–S15]

A graph is intrinsically spatial, but the operations and relationships must also be available through a structured list. The list view supports add, inspect, connect, reorder declared inputs, disconnect, delete, and navigate upstream/downstream. It is not merely a text dump of the canvas.

## 25.2 Non-dragging operation

Every essential drag interaction has a single-pointer alternative without dragging and a keyboard path. Add via palette button, connect via source/destination chooser, move via directional commands or position controls, resize via controls, and reorder via Move up/down. The relevant WCAG dragging criterion requires a non-dragging pointer alternative, not just keyboard support. [S16]

Connection handles need sufficiently large activation areas and an accessible Connect command. Compact visual dots may have larger invisible hit areas, but neighboring targets must not ambiguously overlap. Validate target size/spacing against the applicable standard, including toolbar and close buttons. [S17]

## 25.3 Focus and announcements

Use visible focus indicators, predictable tab order, focus restoration after dialogs, and a clear way out of the canvas. Do not place every decorative element in the tab sequence. Nodes and ports have meaningful names including operation, direction, type, and relevant state.

Sticky headers, panels, toast notifications, and drawers must not obscure the focused control. A diagnostic's “Go to field” scrolls and focuses the actual input while preserving the error summary. Announce important state changes through a polite bounded live region; do not read every progress update or incoming log line. [S18]

Approval dialogs identify their subject and consequence, trap focus only while open, and restore focus to the initiating control. Expiration or plan invalidation is announced and disables confirmation without unexpectedly moving focus to another action.

## 25.4 Mathematical accessibility

Provide accessible typeset output with a source-text fallback and inspectable exact values. A matrix preview needs row/column navigation and explicit headers. Charts need summaries, underlying tables or data descriptions, and visible labels independent of color. Audio has text/event alternatives, and animated plots can be paused.

Do not claim that screen-reader support is solved by adding an `aria-label` to a graph canvas. Test at least one real screen reader on Windows and one on macOS for the documented support matrix. A user must be able to discover a type mismatch, inspect an assumption, and distinguish candidate from accepted evidence without vision or a pointer.

## 25.5 Localization and time

Externalize UI strings, including errors and status explanations. Stable operation IDs, schema codes, and canonical mathematical values remain unchanged by translation. Keep localized labels separate from backend identifiers. Long translated strings must not hide action consequences or truncate the only evidence warning.

Display timestamps in the user's selected timezone with exact UTC available. Durations, event ordering, and expiry decisions come from host data, not the browser's wall clock alone. A live countdown is approximate when clock skew is uncertain; actual approval validity is host-enforced.

Locale-specific number entry is explicit and tested. Copying canonical input uses the machine representation; copying a formatted display says that it is formatted. Right-to-left layout support must preserve mathematical expression direction and port semantics rather than mirror them blindly.

# 26. Performance, resource budgets, and large-workflow behavior

## 26.1 Proposed acceptance targets

Performance targets below are starting engineering budgets, not measured claims. Record the actual hardware, browser, device-pixel ratio, build, graph fixture, visible-node count, and enabled viewers for every benchmark. Use a modest laptop reference in addition to a developer workstation; document the final supported envelope from measurements.

| Scenario | Proposed target | Qualification |
|---|---|---|
| Search in a 2,000-entry local catalog | p95 feedback within 150 ms after debounce | Excludes first network fetch |
| Inspector selection in a 200-node graph | p95 visible response within 100 ms | With preloaded metadata |
| Pan/drag in a 200-node, 400-edge graph | Normally within a 16.7-ms frame budget; investigate repeated frames over 33 ms | No heavyweight embedded viewers |
| Open a 1,000-node, 2,000-edge draft | Interactive outline within 2 seconds; canvas progressively available | Reference hardware must be recorded |
| Live status rendering | No unbounded memory growth under a 10-minute event-burst fixture | Rendering can coalesce progress |
| Large supported import | UI remains cancellable; work is chunked/worker-based | Hitting a configured cap is a clean refusal |

These targets may be revised with measured justification before release. Do not market arbitrary giant graphs as smooth based on a blank-node demo. Correctness, authority separation, and warning visibility are hard requirements even when performance budgets are missed.

## 26.2 Initial defensive limits

Proposed defaults are a 10-MiB editor document, 5,000 nodes, 10,000 edges, nesting depth 32, 8-KiB labels/descriptions per field, and 1-MiB inline parameter text per node, subject to a total document limit. The first supported interactive performance envelope is much smaller than these hard refusal limits. Large datasets are references, not embedded arrays.

Use a 2-MiB control-response budget unless the host's smaller policy applies, paged logs, and bounded retained event summaries. A 100-row table preview is a default display size, not a dataset limit. Thresholds are configurable by the operator but cannot exceed host security limits through a graph document.

## 26.3 Rendering strategy

Memoize node/edge components, use stable callbacks, keep selected-node state separate from global graph arrays, and avoid expensive effects on every position update. React Flow's performance guidance highlights avoidable rerenders and unnecessary subscriptions; benchmark the actual custom nodes rather than assuming the library alone supplies large-graph performance. [S19]

Cull or simplify offscreen/distant content where accessible alternatives remain available. Lazy-load 3D, audio, rich text, and specialized domain viewers. Do not render a full equation typesetter or matrix grid in every collapsed node. A failed optional viewer chunk must not take down the authoring shell.

## 26.4 Data movement and memory

Keep large binary artifacts outside the central state store. Stream or fetch bounded slices through approved host APIs. Transfer worker buffers where appropriate rather than cloning repeatedly. Release object URLs, worker instances, listeners, audio contexts, and image/3D buffers deterministically when views close.

Instrument local performance without capturing mathematical payloads. Track render latency, memory trend, request cancellation, event lag, and parsing duration. Do not log a user's polynomial, matrix, or dataset to understand a slow frame. Diagnostics can include dimensions and anonymized fixture identities with explicit user approval.

## 26.5 Graceful degradation

Above the interactive envelope, default to collapsed groups, paged run lists, and the outline view. Show a reason and offer focused subgraph inspection. Never drop nodes from the saved document merely to improve rendering speed. A million-shard batch is a paged aggregate host view, not a million-node canvas.

When a worker times out, keep the previous good layout or draft, expose cancellation/retry of the UI operation, and preserve data. Retrying a layout is harmless presentation work; retrying a failed mathematical run is a separate host action and must not share the same button behavior.

# 27. Test strategy and acceptance catalog

## 27.1 Test layers

Use reducer/unit tests for commands and serialization, property-based tests for edit roundtrips and ID remapping, component tests for inspectors and state badges, contract tests against a fault-capable mock host, browser end-to-end tests, visual regression tests, accessibility tests, packaged-install tests, and real MathKernel integration tests for every advertised executable journey.

The companion `UI_Acceptance_Test_Catalog.json` contains planned cases with stable IDs, expected outcomes, invariant references, and milestone ownership. It is not executable test code and not a pass report. Convert cases into tests appropriate to their layer; do not replace them with one screenshot assertion each.

## 27.2 Required test families

| Family | Representative cases |
|---|---|
| Exact-input fidelity | Integers beyond 2^53, rationals, decimal locale, blank/null/zero, display precision |
| Graph commands | Atomic reconnect, duplicate ID remap, group deletion, undo/redo, semantic versus layout revision |
| Catalog truthfulness | Partial metadata, missing operations, unsupported controls, schema drift, absent engines |
| Revision safety | Edit during run, stale validation, expired plan, host switch, late response |
| Execution boundary | No run on mount/import/hover; no browser scheduling; no automatic mutation retries |
| Evidence | Candidate spoofing, status/trust distinction, per-claim scope, independent support paths, approximate ancestry |
| Stream behavior | Snapshot watermark, duplicates, out-of-order events, gaps, auth expiry, offline recovery |
| Artifact safety | Hostile HTML/SVG, KaTeX errors, iframe messages, oversized tables/images, expired results |
| Compute review | Unknown price, unsupported hard cap, export denial, invalid approval challenge, cleanup uncertainty |
| Accessibility | Keyboard and non-drag paths, screen reader, focus restoration, zoom, target sizes, reduced motion |
| Persistence | Quota failure, private mode, conflicting tabs, cross-host recovery, import limits, credential omission |
| Packaging | Offline assets, base-path routing, no runtime Node requirement, CSP, optional UI absent |

## 27.3 Mathematical truthfulness gate

The same accepted MathKernel result displayed through Studio and the existing API must preserve trust, semantic status, claim scope, assumptions, source identities, and information loss. A receipt, raw candidate, chart thumbnail, or worker-authored label cannot become accepted evidence. Tests must include both forged stronger claims and valid independent evidence so the UI neither launders nor incorrectly suppresses the host's result.

A verification timeout must not appear as a disproved theorem. A failed search must not appear as nonexistence. A plot's rendering success must not appear as computation success. These are presentation correctness tests, not new mathematical verifier implementations.

## 27.4 Browser and assistive-technology matrix

Record exact tested versions for Chromium-based Chrome/Edge on Windows, Firefox on a supported desktop OS, and Safari on macOS when advertised. Automated WebKit testing is useful but is not a substitute for testing actual Safari behavior before claiming Safari support. Test a non-CUDA client and a machine without installed optional engines.

Automated accessibility checks identify only some issues; combine them with manual keyboard, screen-reader, zoom, and non-dragging journey tests. Playwright's own documentation makes that limitation explicit. [S20]

## 27.5 Security and integration gate

Use a fake host that can return hostile labels, enormous schemas, malicious artifact URLs, forged candidate badges, duplicate events, and stale approval challenges. Assert that no privileged request occurs while importing or viewing those fixtures. Test real host origin/session controls as part of the full deployment, not only unit-tested client conditionals.

For live integrations, use ordinary local MathKernel operations first. Remote compute tests belong to the existing compute service's authorized test environment; UI tests must not provision resources automatically on pull requests. Record which backend capability and operation combinations were actually exercised, and label all other cases mocked, skipped, or blocked.

# 28. Packaging, distribution, and documentation

## 28.1 Optional distribution

Deliver Studio as optional static assets plus the minimum approved user-owned host integration. The base MathKernel install/import remains unchanged. End users should not need npm or Node.js to open a packaged release; those are build-time tools. Do not launch the Vite development server as the production application.

A proposed layout is:

```text
ui/studio/
  src/app/                 # shell, routing, connection boundaries
  src/catalog/             # palette and descriptor presentation
  src/editor/              # document model, commands, canvas adapter
  src/inspectors/          # bounded schema-driven field editors
  src/runs/                # host observation projections
  src/evidence/            # read-only claim and support-path views
  src/viewers/             # versioned safe artifact viewers
  src/host/                # typed facade client, no scheduler
  src/security/            # safe rendering and message guards
  src/workers/             # parsing, search, layout only
  src/testing/             # fixtures and mock-host controls
  schemas/                 # editor/presentation contracts
  tests/
  package.json
  lockfile
  vite.config.ts
```

A minimal optional Python asset-serving package may be added only after the actual host integration is selected. It is not permission to implement an execution service under a UI package name. Keep mathematical changes out of this workstream.

## 28.2 Release artifacts

Build from locked dependencies in a clean environment. Include compiled assets, required fonts/viewer assets under their licenses, schema files, license notices, an asset manifest, SBOM, and source/build instructions. No credentials, local IndexedDB dumps, private test data, provider IDs, or production session captures belong in the distribution.

Test direct navigation to nested routes, a non-root base path, offline asset resolution, strict CSP, cached-old-assets after upgrade, and a read-only installation directory. A missing hashed bundle produces a recovery screen rather than silently mixing incompatible UI versions.

## 28.3 Documentation and skills

Add a Studio user guide covering the first local journey, graph versus run distinction, input exactness, connection types, evidence interpretation, saving/recovery, optional compute review, accessibility paths, and limitations. Integrate a concise Studio section into the README's product description and installation/functionality areas. Do not replace the README or append a phase-by-phase engineering diary.

Update relevant MathKernel skills to explain how saved UI drafts, host workflows, runs, results, and artifacts differ. A proposed `studio.md` skill can document browsing a graph and interpreting screenshots without assuming execution occurred. Existing compute skills remain authoritative for planning, grants, cleanup, and billing; do not duplicate them with conflicting UI shortcuts.

## 28.4 Release labels

A catalog/authoring/inspection release may be called that explicitly. A visual workflow execution release must have a tested workflow-capable host. Experimental viewers and partial-schema nodes carry visible support labels. No screenshot, fixture, or disabled button counts as completed functionality.

Publish a support matrix by host protocol, feature tier, browser, artifact schema, and operation composition coverage. Generate catalog coverage reports from real registrations rather than claiming a fixed number of fully composable nodes because the MCP server has that many tools.

# 29. Delivery sequence and agent handover

## 29.1 UI milestones

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| U0 — Grounding and contracts | Baseline map, scope boundary, catalog/host gap report, threat model, fixtures | Missing runtime capabilities explicitly recorded; no backend work smuggled into UI scope |
| U1 — Shell and catalog | Local application shell, handshake, palette, docs, unavailable states | Packaged assets run offline; incomplete catalogs displayed honestly |
| U2 — Authoring and inspectors | Document schema, canvas/list editing, exact inputs, advisory diagnostics, undo | Exact roundtrips, atomic commands, keyboard/non-drag paths, inert import pass |
| U3 — Results and evidence | Existing result receipts, per-claim inspector, safe basic viewers | No evidence laundering; paging/expiry and hostile text tests pass |
| U4 — Host execution integration | Validation/plan/run UI against a supplied workflow host | Real frozen-revision execution, no browser scheduler, ambiguity/reconnect tests pass |
| U5 — Advanced composition UI | Fragments, supported subworkflow navigation, history/diff | Scope boundaries preserved; unsupported control constructs remain unavailable |
| U6 — Compute review and richer viewers | Optional 1.4 plan/approval displays, supported visual/audio interactions | Real host policy integration; lifecycle axes and isolated-viewer tests pass |
| U7 — Release hardening | Accessibility, performance, browser matrix, packaging, docs/skills | All advertised tiers have evidence of tests; blocked or mocked integrations labelled |

U3 can precede U4 because inspecting existing MathKernel results does not require a general workflow runtime. U4 is blocked until the external workflow capability is real. U6 depends only on the advertised compute contracts it uses, not on every provider in the remote-compute roadmap.

## 29.2 Implementation discipline

Start with data contracts and reducers rather than visual polish alone. Implement one exact-input operation inspector, one numerical-result inspector, one candidate/evidence comparison, and one supported artifact viewer before expanding across domains. Use the same generic shells and descriptor pipeline to expose subsequent capabilities.

Build the non-dragging/list interaction path alongside the canvas, not after the application assumes all actions are pointer gestures. Implement stale-revision handling before automatic validation or live previews. Implement the artifact isolation boundary before embedding existing active HTML.

A working demo against fixtures is a useful milestone artifact, but must contain a permanent Test host indicator and no real-execution claim. Replace fixtures with real host contracts incrementally, documenting every behavior difference.

## 29.3 Required handover report

Each milestone reports the exact input source/hash, changed files, supported UI tiers, host capabilities used, runtime dependencies still external, document/schema changes, tests actually run, pass/fail/skip counts, browser versions, accessibility checks, measured performance, known security limitations, and resulting artifact paths.

Keep progress notes in a dedicated engineering handover. Preserve the existing README's coherent coverage. Do not repair missing UI integration by changing evidence semantics, adding automatic proof labels, or bypassing authorization. A missing backend dependency must remain a visible integration blocker until separately supplied.

## 29.4 Definition of done

For an advertised feature tier, a clean installation lets the user perform its documented journeys through mouse, keyboard, and non-dragging controls; preserves exact input and result/evidence scope; survives stale/reordered observations; handles missing capabilities honestly; keeps imported content inert; and never creates unapproved execution or export.

For visual workflow execution specifically, the browser can be closed after submission and the host continues owning the run. On reopening, Studio reconstructs facts from the host rather than replaying browser actions. That is a release gate, not a request to implement a new scheduler here.

# 30. Architecture decisions, risks, and deferred features

## 30.1 Decisions

| Decision | Reason | Rejected shortcut |
|---|---|---|
| Separate Studio from compute design | Keep UI scope deliverable and independently testable | Expand 1.4 into a whole new platform |
| React Flow as interaction library | Reuse graph editing without adopting another runtime | Hand-build every canvas behavior |
| Host catalog plus safe presentation metadata | Avoid duplicating mathematical signatures | Hand-maintained second operation registry |
| Editor document distinct from executable workflow | Preserve draft flexibility without inventing semantics | Serialize React Flow and call it a runnable IR |
| Host-owned execution | Maintain durability, evidence, policy | JavaScript topological tool-call loops |
| Claim-specific evidence presentation | Preserve MathKernel's differentiator | One green “verified” flag for everything |
| Separate draft and run identity | Prevent stale results from looking current | Mutate active run graphs in place |
| Structured, isolated viewers | Preserve browser authority boundary | Execute returned HTML on the app origin |
| Capability-gated feature tiers | Useful UI before all backends exist | Mock missing functionality as success |
| No default AI assistant | Avoid a second product/security scope | Add chat because the interface uses MCP |

## 30.2 Main risks

The largest risk is scope inflation: a missing workflow host tempts the UI implementation to become a scheduler. The second is evidence distortion: visually simplifying multiple statuses into a success badge. The third is security: locally hosted applications are easily treated as harmless even when they control privileged mathematical and compute services.

Other risks are incomplete catalog schemas, silent numeric precision loss, stale state after revisions, overambitious live previews, expensive graph rerenders, unbounded viewer data, inaccessible wiring, misleading save guarantees, and dependency/license drift. Each maps to a concrete acceptance case or capability restriction rather than a general promise of quality.

## 30.3 Deferred work

Defer multi-user real-time collaboration, CRDTs, shared hosted workspaces, marketplace nodes, arbitrary code blocks, provider administration, new visualization algorithms, a notebook engine, WebGPU numerical computation, natural-language graph generation, agent execution, mobile-first authoring, native desktop wrappers, and publishing executable workflows to third parties.

An eventual assistant can propose edits as an untrusted draft diff; it must not be able to approve spending or bypass the host. An eventual collaboration design needs separate identity, conflict, authorization, and privacy work. Neither is implicitly authorized by this UI plan.

## 30.4 Implementation-time decisions

Confirm exact frontend dependency versions, component primitives, optional editor/layout libraries, supported host transport, session bootstrap, packaging integration, artifact-viewer isolation mechanism, supported browsers, and calibrated performance limits at U0/U1. These are version-sensitive selections, not permission to weaken invariants.

Where host APIs differ from proposed facade names, adapt the UI contract with explicit migration notes. Do not rename or redesign core MathKernel semantics merely to fit a frontend library's defaults.

# 31. Worked end-to-end UI scenarios

## 31.1 Exact matrix entry and incompatible wiring

A user adds a registered exact-matrix input and a supported operation requiring a square matrix. They paste a 3×4 rational matrix including an integer larger than 2^53. The inspector preserves every integer/rational string, previews the dimensions, and highlights the shape mismatch through declared port metadata. The connection is retained as a draft problem or rejected by the selected editing action, not silently reshaped.

After the user corrects the input to 4×4, editor feedback clears the known shape mismatch. Host validation still decides operation admissibility. A successful check is labelled “Host validation passed for this revision,” not a proof of the future result. Moving the input node afterward changes layout only.

## 31.2 Numerical transform and evidence

A user builds a supported signal-transform pipeline and selects a residual or identity check exposed by the host. The check is represented as a planned verification operation. When the host returns numerical output and an accepted numerical check, the node shows the actual trust/semantic-status pair and the bound residual details.

The chart displays a sampled or transformed representation with its source identity and information-loss label. Passing an identity check does not make the transform exact, and the presence of a verifier wire does not certify all downstream analysis. Copying the chart exports a view, not a mathematical certificate.

## 31.3 Graph authoring without a workflow service

A local host supplies catalog entries, object inspection, and standalone operation invocation but no `workflow_execute`. The user can compose, save, and inspect a graph, and can explicitly invoke a separately supported operation with already available inputs. Run workflow is disabled with a precise dependency explanation.

The UI does not loop over graph nodes and call MCP tools to imitate execution. Export produces `.mkstudio.json`, labelled as an authoring document. This is a useful limited release, but it must not be marketed as a completed end-to-end workflow runtime.

## 31.4 Remote GPU plan with uncertain cleanup

A workflow-capable host and the separate compute service report that a particular operation cannot fit the configured local device and offer a permitted remote plan. Studio shows the actual per-device requirement, target/account alias, upload manifest, numerical precision, verification plan, quote uncertainty, and native-currency authorization scope.

The user confirms through the trusted host challenge. Later, results are admitted but resource release is unresolved. Studio displays the result and a persistent cleanup/exposure warning together. Closing the graph does not erase that warning or cancel host reconciliation. No provider price or release guarantee is invented in the browser.

## 31.5 Editing during a long run and reconnecting

Revision 12 is submitted. The user changes a threshold in revision 13, then goes offline. The draft shows unsent revision 13; the run tab remains revision 12 with a stale observation timestamp. An incoming delayed validation result for revision 12 does not validate revision 13.

On reconnect, the UI authenticates the same host/workspace, obtains a current snapshot, and resumes from the supported cursor. Duplicate events do not create duplicate nodes or results. If the cursor expired, the UI resynchronizes rather than guessing. The run is never resubmitted merely because the original request is absent from browser memory.

## 31.6 Malicious imported artifact

A graph file contains an operation label saying “FORMAL — approved,” an unknown active HTML viewer reference, and an untrusted result summary. Import preserves inert labels within limits, rejects prohibited executable/authority content, and marks the result reference unresolved until the host checks it.

Opening the artifact cannot contact a provider, run a script on the application origin, alter an approval dialog, or create a certified badge. A supported isolated viewer receives only validated presentation data and cannot issue host commands through its message channel.

## 31.7 Non-dragging workflow authoring

A keyboard user searches the catalog, adds two supported nodes, edits their inputs, opens Connect output, selects a destination, and receives an accessible type-mismatch explanation. They navigate to the field from the Problems list and correct it. The same commands and revision rules apply as on the canvas.

The user reviews a plan, opens assumptions, confirms only after deliberately focusing the approval control, and later distinguishes numerical evidence from a candidate through text labels. A screen reader can reach the source records without interpreting colors, animation, or graph geometry.

# 32. Source register and reference boundaries

## 32.1 Baseline and companion references

The following source locations were inspected in the supplied archive. `baseline_manifest.json` records their hashes and the companion design's hash. These references ground integration observations, not claims of newly implemented features.

| Ref | Location | Used for |
|---|---|---|
| B1 | `src/mathkernel/capabilities.py` | Actual capability fields and registry |
| B2 | `src/mathkernel/discovery.py` | Explicitly incomplete schema/requiredness hints |
| B3 | `src/mathkernel/models.py` | Actual trust/status enums, MathIR numbers, MathResult |
| B4 | `src/mathkernel_artifacts/evidence.py` | Per-claim support paths and reconciliation |
| B5 | `src/mathkernel/output_policy.py` | Receipts, paging, expiry and bounded output |
| B6 | `src/mathkernel_projection/models.py` | Projection lineage and information loss |
| B7 | `src/mathkernel_viz/document.py`, `artifact.py` | Existing visual document and artifact bridge |
| B8 | `src/mathkernel_sonify/models.py`, `artifact.py` | Existing sonification mappings and synchronization |
| B9 | `src/mathkernel/kernel.py` | Facade, object APIs, derivations, local jobs, exports |
| B10 | `src/mathkernel_mcp/server.py`, `pyproject.toml`, `README.md` | Existing integration/packaging surface |
| P1 | `MathKernel_Remote_Compute_Technical_Design.md`, MK-RHC-001 rev. 1.0 | External compute, authorization, lifecycle and verification contracts |

P1 remains a proposed design. This UI specification must not promote its APIs to implemented status. Recheck the newest source before starting U0 and record which external capabilities actually exist then.

## 32.2 External primary references

Official documentation and primary specifications below were checked on 7 September 2026. They support external-library and platform observations. The original UI requirements, limits, workflows, and milestones in this document are design decisions. Exact dependency versions and browser behavior require implementation-time testing.

**S1 — React Flow overview, installation, licensing, and API.** Graph-interaction library and MIT licensing; not MathKernel execution semantics. `https://reactflow.dev/`; `https://reactflow.dev/learn`; `https://reactflow.dev/api-reference/react-flow`

**S2 — Node-RED workspace and subflows.** Interaction reference for palette, wired components, tabs, and nested navigation. `https://nodered.org/docs/user-guide/editor/workspace/`; `https://nodered.org/docs/user-guide/editor/workspace/subflows`

**S3 — Vite getting started.** Development/build tooling and static production assets. `https://vite.dev/guide/`

**S4 — TypeScript strict option.** Strict compiler checking. `https://www.typescriptlang.org/tsconfig/strict.html`

**S5 — React Flow layouting overview.** External layout approaches and integration. `https://reactflow.dev/learn/layouting/layouting`

**S6 — JSON Schema reference.** Validation keywords, annotations, composition, and declared dialects. `https://json-schema.org/understanding-json-schema/reference`

**S7 — KaTeX security, options, and API.** Untrusted rendering controls and persistent macro boundaries. `https://katex.org/docs/security`; `https://katex.org/docs/options`; `https://katex.org/docs/api`

**S8 — WHATWG HTML, server-sent events.** Event IDs and reconnect protocol. `https://html.spec.whatwg.org/multipage/server-sent-events.html`

**S9 — MDN storage quotas and eviction.** Browser storage lifetime/availability limitations. `https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria`

**S10 — OWASP HTML5 Security Cheat Sheet.** Browser storage, messaging, and sandbox boundaries. `https://cheatsheetseries.owasp.org/cheatsheets/HTML5_Security_Cheat_Sheet.html`

**S11 — OWASP Third Party JavaScript Management.** Isolated frame and script trust considerations. `https://cheatsheetseries.owasp.org/cheatsheets/Third_Party_Javascript_Management_Cheat_Sheet.html`

**S12 — OWASP CSRF and XSS prevention.** Independent request protection and safe rendering. `https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html`; `https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html`

**S13 — OWASP Content Security Policy.** CSP as a defense-in-depth browser policy. `https://cheatsheetseries.owasp.org/cheatsheets/Content_Security_Policy_Cheat_Sheet.html`

**S14 — W3C WCAG 2.2.** Accessibility conformance target. `https://www.w3.org/TR/WCAG22/`

**S15 — React Flow accessibility.** Library keyboard/screen-reader configuration, not whole-product certification. `https://reactflow.dev/learn/advanced-use/accessibility`

**S16 — W3C Understanding 2.5.7, Dragging Movements.** Non-dragging pointer alternatives. `https://www.w3.org/WAI/WCAG22/Understanding/dragging-movements.html`

**S17 — W3C Understanding 2.5.8, Target Size.** Activation size/spacing requirements. `https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html`

**S18 — W3C Understanding 2.4.11, Focus Not Obscured.** Visibility of focused components. `https://www.w3.org/WAI/WCAG22/Understanding/focus-not-obscured-minimum.html`

**S19 — React Flow performance.** Rendering and state-subscription considerations. `https://reactflow.dev/learn/advanced-use/performance`

**S20 — Playwright accessibility testing.** Automated testing capabilities and need for manual assessment. `https://playwright.dev/docs/accessibility-testing`

## 32.3 Document status

No UI application, host facade, workflow service, provider integration, or mathematical feature was implemented while preparing this package. No live backend or cloud account was exercised. Document/schema/fixture consistency checks are reported separately and must not be mistaken for runtime, security, accessibility, or browser compatibility certification.
