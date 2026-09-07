# Studio 0.4 local execution alpha

Grounded on live `Staatsgeheim/MathKernel`, branch `mathkernel-studio`,
`9a9ee767fcaa97a825279ffcc24834515f5890cc`, 7 September 2026.
This implementation extends that branch; no default-branch merge is included.

The workflow dependency is now implemented separately in `src/mathkernel_workflow`.
Studio remains the authoring/inspection and explicit command interface. Kernel
processes, plans, policy, authority, requests and run state belong to the backend.

## Milestone evidence

| Milestone | Implemented tier | Qualification limit |
| --- | --- | --- |
| U4 | Typed host compilation; frozen output/subgraph plans; real kernel execution; session-bound approvals; SQLite submission identity and reconciliation; browser-independent process ownership; cancellation, timeout and durable results | Bounded local DAGs, 15 explicit adapters; manual snapshots, no event stream |
| U5 | Existing groups, fragments and revision comparison; immutable saved self-contained single-output subworkflows; explicit pinned catalog/boundary inspection; host expansion within graph budget | External subworkflow inputs and general control constructs unavailable |
| U6 | Real local operator policy, concurrency/time limits, optional POSIX memory limit, no-provider cost disclosures, no remote fallback, independent execution/evidence/artifact/cleanup observations; existing isolated plot and audio viewers | No cloud/GPU adapter or metered local electricity/hardware cost; bounded preview audio, not PCM fidelity |
| U7 | Versioned core and Studio wheels, exact static-asset manifest, dependency/license inventory, updated docs/skills, regression tests and Linux/Windows CI definition | Full browser, screen-reader, zoom, localization and reference-hardware performance qualification outstanding; CI definition is not a platform test result |

## Contract and persistence

- Explicit adapters: parse, simplify, differentiate, integrate (indefinite), solve,
  analyze, matrix_create, matrix_det, matrix_inverse, matrix_transpose,
  matrix_multiply, matrix_rank, matrix_rref, matrix_eigenvalues and matrix_solve.
  These call existing MathKernel methods; no mathematical engine is reimplemented.
- Drafts and bindings are revalidated on the host. Canonical SHA-256 bindings
  match Python and TypeScript, including host provenance. Labels/layout do not
  change mathematical intent. Unknown descriptors, bad wiring, stale schema
  digests, unsupported inputs and cycles fail closed.
- Frozen plans bind operations, scope, outputs, kernel/adapter versions and policy.
  Expired or changed plans cannot issue usable authority. A one-time authority is
  bound to the current session and immutable plan. Confirmation cannot override
  operator denial. No approval is recovered from imported documents.
- Accepted and rejected submission outcomes are retained by exact request intent.
  A retry of the same request returns the same result without another worker;
  reusing its ID for changed intent is refused. Reconciliation works in a fresh
  session in the same host/workspace. State quotas preserve existing request reads.
- One owner locks each SQLite state directory. WAL/FULL commits precede launch.
  Completed result snapshots survive restart. Unfinished runs/attempts become
  interrupted, with no automatic execution replay. Worker parent monitoring and
  explicit supervisor cleanup are limited to owned processes.
- Browser disconnect is not cancellation. Cancellation acknowledgement precedes
  cleanup confirmation. Timeout/worker failure is an execution fact, not a claim
  that the mathematics is false. Original MathResult values, trust, semantic
  status, assumptions and per-claim evidence are preserved without aggregate
  trust promotion.
- Workers use explicit in-memory kernel settings, preventing inheritance of an
  operator's persisted kernel store. The local process is a resource/ownership
  boundary for allowlisted operations, not an arbitrary-code security sandbox.

## Verification

Delivery checks are recorded in `verification.txt` in the downloadable handoff:
frontend/component/binding tests, real-kernel backend tests, authenticated host
integration tests, targeted core parser regressions, production bundles, wheel
contents and execution from the installed wheel payloads.

Every advertised operation is exercised against a real kernel. The HTTP test
connects with a one-time code, validates, plans, approves and submits, disconnects
while running, reconnects and reads/adopts the original result. It compares values,
trust, semantic status, assumptions and side conditions to direct kernel output.
Other tests cover stale authority, scope isolation, tampered plans, cancellation,
process timeout, restart recovery, durable rejection and saved subworkflow execution.

Chrome automation could not establish its tab connection after the documented
recovery attempt. No new real-browser success is claimed. Existing DOM tests are
not a substitute for screen-reader/browser qualification. A broad optional core
suite stalled in an unrelated formal-engine path; only the completed targeted
core checks are counted. The full 144-case design acceptance catalog has not been
formally executed. This remains a local execution alpha, not a completed U7 release.

## Installation

Install **both** supplied wheels together (core `1.3.1.dev1`, Studio `0.4.0a1`),
then launch `mathkernel-studio`. Python dependencies must already be installed for
an offline installation. End users do not need Node/npm. Use the printed loopback
address and connection code, then the empty-draft **Create exact example** action.
See the Studio README for the explicit validate/plan/approve/submit journey and
policy/state-directory options. The source and Git bundle preserve the complete
branch for further platform qualification.
