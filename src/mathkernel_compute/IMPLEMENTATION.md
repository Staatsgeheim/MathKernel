# Remote-compute implementation checkpoint

## Grounding and branch

- Repository: `Staatsgeheim/MathKernel`; source branch `master`.
- Exact base: `bf71fc88333ee6cf455cb12538e64fc385512a8a`, merged Studio PR #1.
- Implementation branch: `mathkernel-remote`; no merge into `master`.
- Previous local R0–R2 checkpoint: `3b71464ae005c25ad9f48c550ca5726d9528c550`.
- Design: supplied `MathKernel_Remote_Compute_Design_Package.zip`, MK-RHC-001,
  revision 1.0, 7 September 2026. Implementation date: 8 September 2026.
- Package development version: `1.4.0.dev2`; optional extra: `compute`.

The independent `mathkernel_compute` subsystem does not replace the mathematical
facade, legacy `job_*` APIs, `mathkernel_workflow`, or the Studio host. Remote
candidates remain independent DTOs, not serialized kernel results. Existing formal
and interval engines are not claimed as qualified remote verifiers.

## Implemented boundary

This checkpoint extends the R0–R2 local pilot with **native R3 SSH and R4 single-job
Slurm adapters**. It does not complete all requirements or acceptance gates of R3/R4,
all providers, or the design's 114-case acceptance catalog.

| Area | Delivered behavior | Remaining boundary |
| --- | --- | --- |
| R0–R2 contracts/lifecycle | Frozen strict contracts, RFC 8785 identities, exact input strings, explicit operation registry, scoped grants, independent SQLite journal, durable outbox, local native supervisor, quarantine and independent verification | Generic object/context export, array codecs, distributed coordinator and general verifier registries |
| R3 planning/authority | Explicit operator target aliases; host/port/account and allocation review; pinned known-hosts content, worker configuration and remote/local runtime profiles; exact-scope single-attempt export approval | Remote target auto-selection, paid authority and provider pricing |
| R3 SSH | Real OpenSSH client; restricted config sources, strict host keys, no forwarding/prompts; fixed gateway command; bounded data-only stdin; private confined spool; durable acceptance and duplicate/conflict handling | External OpenSSH daemon/host qualification, Windows/macOS controller qualification |
| R3 supervision/recovery | Detached native Linux supervisor, start/runtime deadlines, process incarnation checks, cancellation intent, bounded retained bytes, host disconnect and controller exit recovery, uncertainty on unavailable ownership | Independent watchdog for host/supervisor failure; automatic retention/garbage collection |
| R4 native Slurm | Reviewed partition/account/QOS/constraint and resource shape; fixed sbatch script, one node/task/CPU; no login-node mathematics; explicit queue/accounting fields; lost-ack reconciliation; UUID/digest/UID/submit-incarnation checks; no requeue; filtered cancellation | Live cluster/site qualification, Apptainer/SIF profile, arrays/MPI, GPU, federation |
| R4 outcomes/accounting | Separate OOM/timeout/cancel/preempt/node-failure/expiry/exit states; no completion inference from absent squeue; no cleanup inference from scancel ack; bounded CPU/memory reservation and retained sacct usage | Institutional allocation-credit prices and currency conversion; exact queue expiry cancellation while controller is offline |
| Admission | Raw candidate bytes preserved into quarantine; exact attempt/input/execution binding; independent Euclid/witness or Decimal numerical checks; local verifier runtime distinct from worker runtime; only new local evidence admitted | Arbitrary remote evidence/Lean import, general exact/numeric ancestry graphs and typed object re-import |
| Interfaces | Python client, operator profiles, explicit probe/workspace/approval CLI, inert replay, setup guide | Compute MCP tools, Studio compute integration, managed cloud providers |

`auto` remains local CPU. A remote request cannot synthesize its connection profile
or issue its own grant. Existing-host costs and Slurm allocation credits remain
unknown; the plan does not present a zero-price estimate for remote work. No paid
host was provisioned or provider account accessed during implementation.

## Protocol and retained records

Worker candidates use protocol `1.0`, bundles `mk.bundle/1`, and the controller
journal remains schema 1. The fixed gateway adds its own `mk.gateway/1` protocol.
New remote bindings and attempt start deadlines are omitted from canonical local
records when absent, preserving existing R0–R2 plan/spec/attempt identities. Unknown
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

Verified on Linux/Python 3.12: **63 compute tests passed** (plus 20 subtest
assertions), **56 selected legacy compatibility tests passed** (1 CUDA skip),
and **15 workflow backend tests passed**: 134 passing selected tests, 1 skip.
Skill metadata validation, wheel/sdist creation, wheel contents and entrypoint
validation passed. The handoff's `verification.txt` records the installed-wheel
journeys and exact commands.

The compute suite includes both R0–R2 regression coverage and new cases in
`test_remote.py` and `test_slurm.py`.

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

## Next work

Qualify the native profiles on an explicitly authorized OpenSSH host and Slurm
site, including actual accounting fields, allocation cleanup, transport loss and
site clock behavior. Add pinned Apptainer/SIF support and unattended queue-retention
policy before claiming the full R4 release gate. Continue with R5 managed-provider
adapters using these same grant/outbox/receipt/admission boundaries; do not weaken
candidate admission or substitute simulated provider success for a real backend.
