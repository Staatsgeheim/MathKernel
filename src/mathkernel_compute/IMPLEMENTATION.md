# Remote-compute implementation checkpoint

## Grounding and branch

- Repository: `Staatsgeheim/MathKernel`; source branch `master`.
- Exact base: `bf71fc88333ee6cf455cb12538e64fc385512a8a`, merged Studio PR #1.
- Implementation branch: `mathkernel-remote`; no merge into `master`.
- Previous local R0–R2 checkpoint: `3b71464ae005c25ad9f48c550ca5726d9528c550`.
- Design: supplied `MathKernel_Remote_Compute_Design_Package.zip`, MK-RHC-001,
  revision 1.0, 7 September 2026. Current implementation date: 9 September 2026.
- Package development version: `1.4.0.dev3`; optional extras: `compute`, `compute-modal`, `compute-runpod`, `compute-runpod-worker`.

The independent `mathkernel_compute` subsystem does not replace the mathematical
facade, legacy `job_*` APIs, `mathkernel_workflow`, or the Studio host. Remote
candidates remain independent DTOs, not serialized kernel results. Existing formal
and interval engines are not claimed as qualified remote verifiers.

## Implemented boundary

This checkpoint extends the native R3/R4 implementation at
`2ee9d04a9864c8ffb9a48064069d668b62b79c24` with **experimental R5 Modal Sandbox and
R6 existing Runpod Serverless adapters**, paid reservation accounting, explicit CUDA
eligibility and bounded independent batches. It does not claim live cloud/GPU
qualification, complete R3–R6 release gates or execution of all 114 design cases.

| Area | Delivered behavior | Remaining boundary |
| --- | --- | --- |
| R0–R2 contracts/lifecycle | Frozen strict contracts, RFC 8785 identities, exact input strings, explicit operation registry, scoped grants, independent SQLite journal, durable outbox, local native supervisor, quarantine and independent verification | Generic object/context export, array codecs, distributed coordinator and general verifier registries |
| R3 planning/authority | Explicit operator target aliases; host/port/account and allocation review; pinned known-hosts content, worker configuration and remote/local runtime profiles; exact-scope single-attempt export approval | Remote target auto-selection and provider-native price feeds |
| R3 SSH | Real OpenSSH client; restricted config sources, strict host keys, no forwarding/prompts; fixed gateway command; bounded data-only stdin; private confined spool; durable acceptance and duplicate/conflict handling | External OpenSSH daemon/host qualification, Windows/macOS controller qualification |
| R3 supervision/recovery | Detached native Linux supervisor, start/runtime deadlines, process incarnation checks, cancellation intent, bounded retained bytes, host disconnect and controller exit recovery, uncertainty on unavailable ownership | Independent watchdog for host/supervisor failure; automatic retention/garbage collection |
| R4 native Slurm | Reviewed partition/account/QOS/constraint and resource shape; fixed sbatch script, one node/task/CPU; no login-node mathematics; explicit queue/accounting fields; lost-ack reconciliation; UUID/digest/UID/submit-incarnation checks; no requeue; filtered cancellation | Live cluster/site qualification, Apptainer/SIF profile, arrays/MPI, GPU, federation |
| R4 outcomes/accounting | Separate OOM/timeout/cancel/preempt/node-failure/expiry/exit states; no completion inference from absent squeue; no cleanup inference from scancel ack; bounded CPU/memory reservation and retained sacct usage | Institutional allocation-credit prices and currency conversion; exact queue expiry cancellation while controller is offline |
| Admission | Raw candidate bytes preserved into quarantine; exact attempt/input/execution binding; independent Euclid/witness or Decimal numerical checks; local verifier runtime distinct from worker runtime; only new local evidence admitted | Arbitrary remote evidence/Lean import, general exact/numeric ancestry graphs and typed object re-import |
| Interfaces | Python client, operator profiles, explicit probe/workspace/approval CLI, inert replay, setup guide | Compute MCP tools and Studio compute integration |

### Managed-provider implementation

| Area | Delivered behavior | Qualification boundary |
| --- | --- | --- |
| R5 Modal | Pinned SDK 1.5.5; existing app/image/Volume v2; fixed Sandbox process; per-attempt volume submount; explicit CPU/memory limits, blocked network and native lifetime; v1 ID/tag ownership checks; durable candidate then terminal `sync` | No live Sandbox, volume lifetime, GPU, network/resource or billing qualification |
| R5 SDK boundary | Byte/protobuf control paths only; SDK calls in Linux memory/CPU/wall/output-limited subprocess; real SDK poll/reattach/terminate test with deserialization forbidden | Linux controller only; SDK transitive behavior and provider deployment must be requalified on upgrades |
| R6 Runpod | Bounded REST JSON; existing queue endpoint deployment digest check; immutable image, worker/idle/placement scope; explicit execution timeout and total TTL; scoped cancellation only | No live endpoint; provider acknowledgement required for authoritative job handle; worker receipt cannot authorize a cancel |
| R6 durable broker | Existing AWS S3 store; explicit credentials; bounded streaming reads and conditional immutable writes; durable acceptance claim before math; real native worker; candidate before terminal descriptor | AWS credentials/IAM and live retention/connectivity require operator qualification; no arbitrary S3-compatible endpoint |
| Monetary admission | Separate paid/export/storage grant; USD quote source/scope/expiry; one-journal aggregate reservations and retained uncertain exposure; immutable account/budget identity; reviewed cost/storage settlement | Soft managed exposure only; no account-global cap, native billing import or guaranteed final price |
| R6 batches | 1–16 independent registered requests, stable shard IDs, atomic complete-set grants/reservations/outbox, bounded dispatch concurrency, parent-bounded start authority, duplicate/swap rejection, per-shard local verification | No mathematical reducer, stochastic stream, checkpoint/resume, distributed graph, MPI or contribution cache |
| GPU eligibility | Existing integer CUDA sweep only, explicit one-GPU target/engine/resources, pinned CuPy/CUDA runtime and driver; independent CPU verifier | GPU execution/performance not exercised; no automatic GPU selection or cross-architecture bitwise guarantee |

The operator guide is [MANAGED.md](MANAGED.md). Setup failures known to precede an
invocation release invocation ownership while retaining possible storage charges.
A timeout during the invocation call remains ambiguous and never triggers a new
submission. A finished job does not close its monetary reservation. A provider 404
cannot erase confirmed execution/cleanup facts or establish a zero bill.

`auto` remains local CPU. A remote request cannot synthesize its connection profile
or issue its own grant. Existing-host costs and Slurm allocation credits remain
unknown; the plan does not present a zero-price estimate for remote work. No paid
host was provisioned or provider account accessed during implementation.

## Protocol and retained records

Worker candidates use protocol `1.0`, bundles `mk.bundle/1`, and the controller
journal upgrades schema 1 to schema 2 transactionally. The fixed gateway adds its own `mk.gateway/1` protocol.
New remote/managed bindings, accelerator metadata, batch bindings and attempt start
deadlines are omitted from canonical records when absent, preserving existing
R0–R4 plan/spec/attempt identities. Schema 2 adds provider handle/observation,
settlement and batch tables without rewriting existing payloads. Unknown
schema versions and unknown fields remain rejected. New side effects still require
current policy/runtime identities; upgrading source is not implicit reauthorization.

The remote endpoint locks a private spool across independent SSH sessions. A durable
intent precedes native launch or sbatch. If dispatch becomes ambiguous, observation
never calls submit again. Status/cancel/fetch carry only references, not another
copy of mathematical inputs. Gateway replies preserve candidate bytes through a
bounded base64 wrapper, including malformed/untrusted candidate documents, for local
quarantine before mathematical parsing.

The controller separately records transport availability. Unreachability preserves
last confirmed execution observations and exposes unresolved cleanup. Local startup
reconciliation is optional; CLI discovery/plan/status open with recovery disabled,
so a read-only command cannot dispatch an old pending intent. Explicit reconcile
and submit still process already authorized intents.

## Qualification evidence

Verified on Linux/Python 3.12: **96 compute tests passed** (plus 20 subtest
assertions), **56 selected legacy compatibility tests passed** (one CUDA skip),
and **15 workflow backend tests passed**: **167 selected tests passed, one skipped**.
Four installed-wheel journeys also passed; these re-exercise existing cases and
are not added to that count. The installed modules were checked to come from the
wheel target. SDK control tests use Modal 1.5.5 and boto3 1.43.90.

The native checkpoint had 63 compute tests. This continuation adds 33 managed
lifecycle, budget, S3/Modal artifact, real Runpod broker subprocess, SDK, batch and
migration tests. Skill metadata validation, wheel/sdist building, exact compute
source/guide package comparison and provider-extra metadata checks passed. The
handoff's `verification.txt` and logs contain the exact commands and limits.
The complete core suite, a clean dependency-resolution environment and remote CI
matrix are not claimed.

The compute suite includes both R0–R2 regression coverage and new cases in
`test_remote.py`, `test_slurm.py`, `test_managed.py` and `test_modal.py`.

| Design case | Evidence in this checkpoint | Qualification limit |
| --- | --- | --- |
| SSH-01 | Actual OpenSSH client rejects empty and mismatched known-host files before sending a gateway command | Loopback Paramiko server; external OpenSSH daemon pending |
| SSH-02 | Profile/path/schema injection rejection; fixed remote command; truncated input has no staging effect | Two registered self-contained DTOs only |
| SSH-03 | Actual encrypted transport, reconnect, retained output, controller subprocess exit, duplicate acceptance | One Linux host simulates both sides; distributed failure modes pending |
| SSH-04 | PID incarnation mismatch retains LOST; cancellation cannot kill an unrelated process | Native Linux ownership; Windows remote worker unsupported |
| SSH-05 | Cancellation/deadline cleanup and unrelated-process survival; no host destruction API | Automatic deletion/retention not implemented |
| SLURM-01 | No worker before allocation; simulator runs actual generated batch script and candidate passes local verification | Scheduler simulator, not a real Slurm daemon/cluster |
| SLURM-02 | Missing queue/delayed accounting retain unknown allocation; no fabricated result | Site accounting configuration must be qualified |
| SLURM-03 | scancel acknowledgement leaves ACTIVE until terminal accounting; scheduler UID/name filters inspected | Cluster cancellation races need live qualification |
| SLURM-04 | Changed submit incarnation, duplicate rows and restart counts block ownership/admission; actual requeue entrypoint rejects execution | Single-job profile only; site-enforced no-requeue still required |

Other new checks cover exact export approval scope, remote capacity/scope/path
boundaries, expired start authority, bounded transport output/time, raw hostile
candidate preservation, prior local canonical bytes and read-only startup.
The previous local controller exit, process-tree cleanup, mathematical admission,
concurrency, rollback and journal-integrity tests remain in the suite.

The initial loopback SSH fixture closed its transport too early after sending the
channel result. Waiting for the client to consume channel closure fixed the
intermittent transfer failure. The server remains a test-only dependency; Paramiko
is not installed by the compute extra or imported by the product. Required runtime
native SSH uses the system OpenSSH client.

No external compute account/host, live Slurm cluster, GPU, Windows/macOS controller,
Apptainer image or full 114-case run was used. The CI Python matrix is defined but
was not remotely executed in this session. Existing Studio UI/browser code did not
change; no new browser qualification is claimed.

## Managed design-case evidence

| Design case | Local evidence | Remaining gate |
| --- | --- | --- |
| MODAL-01 | Real Modal 1.5.5 protobuf reattach/poll/tags/terminate with deserialization disabled; fixed adapter calls, bounded bridge and raw candidate tests | Live SDK/provider conformance and independent security review |
| MODAL-02 | Explicit resource units/limits, network block, no secrets/ports; beta ID rejection | Live enforcement on the pinned runtime |
| MODAL-03 | Candidate/terminal durability order, volume retrieval after simulated Sandbox exit | Live Volume v2 sync, controller loss and retention |
| MODAL-04 | Physical-core conversion, equal request/hard limits; USD reservation separate from billing | Provider pricing/usage integration |
| RUNPOD-01/02 | 404 retains unknown cleanup; durable S3 output is still locally verified; no replacement | Live queue/retention expiry and S3 recovery |
| RUNPOD-03 | Millisecond execution/TTL policies; independent native runtime and start deadlines | Live queue timing and provider cancellation lag |
| RUNPOD-04 | Original provider handle only; forged worker receipt cannot redirect cancellation; no endpoint mutation API | Live shared-endpoint qualification |
| BATCH-01/03 | Duplicate delivery does not execute twice; unique shard identity and exact manifest checks; missing/duplicate/swap coverage stays explicit | No reducer or stochastic contribution engine advertised |
| BATCH-02/04/05/06/07/08 | Unsupported replicate/checkpoint/cache fields are rejected; no bitwise/cross-workspace cache claim | Those subsystems remain unimplemented; these cases are not marked passed |

The batch review found and fixed an authority bug: child plans created a few
milliseconds apart had later expiry times than the batch's earliest deadline.
Child grants now inherit the parent's deadline, and dispatch checks the grant
again before any provider call. Expired pending shards are rejected even when an
active sibling occupies the concurrency slot, releasing their unexported reservation.

No live Modal, Runpod, AWS S3, paid GPU, external SSH host or Slurm cluster was used.
Cloud service fixtures test protocol behavior; real subprocesses test mathematical
execution and admission. SDK method tests use the actual pinned SDKs without
provider credentials. None of these are a substitute for the deployment gates above.

## Next work

R7: explicit Lambda instance provisioning and generation-bound provider termination,
with independently qualified cleanup/watchdog policy. Existing native SSH attachment
is already separate from provisioning, but a native GPU profile is not yet supplied.
R8: broader registered operations/artifact codecs, cache/checkpoint policies where
applicable, Studio/MCP integration, complete provider conformance and release hardening.
Continue external SSH/Slurm and Apptainer qualification in parallel with those gates
when authorized hosts and credentials are available. No stable managed-provider or
full-design completion claim is made by this checkpoint.
