# Remote-compute implementation checkpoint

## Grounding

- Repository: `Staatsgeheim/MathKernel`, branch `mathkernel-remote`.
- Unchanged master/base: `bf71fc88333ee6cf455cb12538e64fc385512a8a` (merged Studio PR #1).
- Previous checkpoint: `5444d76fb34e83d7428342759c2161a2d551db11` (R5–R6).
- Design: supplied MathKernel Remote Compute Design Package, MK-RHC-001 revision 1.0,
  7 September 2026. Implementation/qualification date: 9 September 2026.
- Development package version: `1.4.0.dev4`.

This continuation implements the **R7 owned-VM lifecycle and independent-watchdog
strategy**, and the **R8 local release-hardening scope**. Remote deployments remain
experimental. It does not claim every operation/profile or all 114 planned cases
are complete. No changes route legacy mathematics, `job_*`, Studio workflows or MCP
calls into paid remote execution. No cloud resource was launched during development.

## Delivered scope

| Area | Behavior | Boundary |
| --- | --- | --- |
| R0–R2 core | Strict immutable data contracts, exact inputs, RFC 8785 identities, explicit eligibility, host grants, durable journal/outbox, native worker supervision, quarantine and independent verification | Two self-contained operations; no arbitrary code/object graph export |
| R3 SSH | Pinned host keys/configuration/runtime, bounded OpenSSH transport, private gateway spool, reconnect/cancel/deadline ownership | Native Linux CPU profile; external host and other controller OS qualification pending |
| R4 Slurm | Native single-job allocation, owned scheduler incarnation, no requeue, separate accounting/cleanup observations | Simulator-qualified; no live cluster, GPU, arrays, MPI or Apptainer |
| R5 Modal | Pinned SDK, existing app/image/Volume v2, bounded Sandbox, data/protobuf paths, durable artifacts, scoped cancellation | Experimental, Linux bridge; live GPU/network/durability/billing qualification pending |
| R6 Runpod/batches | Existing queue endpoint, durable S3 broker, bounded CPU/explicit integer CUDA, 1–16 immutable independent shards, aggregate paid reservations | No live deployment or GPU measurement; no reducer/checkpoint/cache/stochastic stream engine |
| R7 Lambda | Separate VM plan/grant/lease, pinned image/region/type/key/firewall, one-shot launch, ownership-tag lookup, exact provider termination, shared USD exposure ledger | Experimental managed exposure; no hard cap or qualified native expiry; native CPU-only SSH delegation |
| R7 watchdog | Prearmable inert ticket, separate host approval/journal/credentials, exact owned-instance expiry cleanup, lost-ack/termination reconciliation | Operator-deployed independent failure domain; no machine attestation or guarantee during outages |
| R8 contracts/replay | 19 runtime-derived schemas, bounded inert v1/v2 replay inspection, explicit external artifact requirements, no imported trust or authority | Replay is data, not an automatic rerun or self-contained artifact archive |
| R8 packaging | Wheel/sdist hygiene, clean base and each provider-extra installation, sdist rebuild/install, installed examples, dependency freezes | Linux/Python 3.12 qualified here; CI matrix declared but not run remotely |
| R8 measurements | Local exact/numerical submit/restart/retrieve/verify journeys with measured startup/handler/retrieval/verification/cleanup | No paid calibration, remote queue/provisioning, GPU speedup or statistical calibration claim |

Existing native/managed setup remains in [REMOTE.md](REMOTE.md) and [MANAGED.md](MANAGED.md).
New operator and qualification guides are [LAMBDA.md](LAMBDA.md) and [RELEASE.md](RELEASE.md).
README/skills describe current behavior; progress is kept here.

## Owned-VM authority and recovery

A plan is not provisioning authority. `vm-approve` issues a distinct single-use
host grant, bound to the exact profile, budget, quote, workspace and launch deadline.
It explicitly acknowledges provider privileges, network egress, residual exposure
and possible output loss at expiry. Strict mode refuses provisioning: a local thread
or an unqualified watchdog cannot establish disconnected cleanup guarantees.
Ordinary local/export/managed grants cannot authorize VM creation.

Launch intent, reservation and consumed grant are committed before the provider
call. The bridge rechecks start authority after rate limiting and before networking.
A definite rejection voids the reservation; ambiguity keeps it. Repeated launch
with the same request identity never creates a replacement. A unique authenticated
ownership-tag match can recover the UUID; missing/duplicate/changed ownership stays
unknown. Broad name-prefix deletion and guest shutdown do not exist in the adapter.

SSH attachment requires the provider-observed IP, an independently authenticated
host key, the approved runtime and pinned gateway configuration. A single job is
bound transactionally to the VM. Its export requires a separate SSH approval.
Changing the profile cannot silently detach it from VM ownership. Normal completion
retains and integrity-checks candidate bytes in durable local quarantine before
termination. Temporarily missing output keeps the VM. Deadline/cancel cleanup may
discard output, as explicitly acknowledged by the host grant.

Termination intent precedes the exact-ID provider operation. A successful reply
that says `terminating`, a timeout, a missing instance or a process exit cannot
establish release or settle billing. Confirmed release remains distinct from the
reviewed actual cost, including residual storage/transfer. VM and managed-job
reservations share the coordinator ledger and concurrency policy; queued batches
cannot dispatch around an occupied VM slot.

The independent watchdog can be armed before launch on another host. It uses its
own local journal, reviewed credentials and terminal approval, not the laptop or
guest state. It never launches resources or admits mathematics. Its physical
independence remains operator-acknowledged, so strict mode remains unsupported.

## Current provider reference

The rendered Lambda API reference was read on 9 September 2026, OpenAPI 1.10.0.
The implementation uses only Bearer-authenticated launch/list/get/terminate,
explicit image IDs, ownership tags and existing firewall references. No obsolete
quantity field, automatic retries, guest bootstrap commands or filesystem creation
are sent. Jupyter tokens/URLs returned by the API are discarded before retention.
Requests disable proxies/redirects, bound replies and run inside a 30-second parent
wall deadline. Account API limits are spaced and reconciliation failures back off.
Official references and remaining live gates are linked in LAMBDA.md.

## Retained-data compatibility

The controller transactionally upgrades journal versions 0–2 to **schema 3**, adding
VM plans/grants/leases/events/attachments/job bindings and watchdog records. Existing
payload bytes and integrity hashes are preserved. Unknown versions are rejected.
Legacy local/native/managed execution bindings retain their canonical form.

New replay exports use `mk.compute-replay/2`, with declared external artifacts;
inspection also accepts v1. Neither grants nor provider credentials are included.
Inspection creates no journal or side effect, and historical status never establishes
trust. A fresh execution still requires current runtime/policy identities and host
authority. Runtime source upgrades do not silently reauthorize old plans.

Worker timing metadata is an untrusted diagnostic observation, separate from claims
and evidence. Local supervisor telemetry is retained in its private spool; there is
no new billing inference or admission path from those timestamps.

## Verification

The complete compute suite passed against the installed wheel with fresh resolved
dependencies: **128 tests plus 20 subtest assertions**. Selected legacy compatibility
tests passed **56**, with **one CUDA skip**; workflow backend tests passed **15**:
**199 selected tests passed, one skipped**. The complete core suite and all 114
design cases are not claimed. Exact commands/logs accompany the ZIP.

Six clean environments passed dependency resolution, `pip check`, base mathematics,
SDK version and installed-contract checks: base wheel, compute wheel, Modal extra,
Runpod controller extra, Runpod worker extra, and rebuilt source distribution with
compute. The environments had no checkout PYTHONPATH, inherited provider credentials
or user site packages. Distribution scans rejected bytecode/private keys/runtime
state. Installed runtime Python files matched current source byte for byte.

Two packaged local measurement journeys also passed, including client close/reopen,
retained output, independent verification and evidence-preserving result retrieval.
These are separate qualification journeys, not added to the unit-test count.

| Local operation | Handler including lazy imports | Local verification/admission | End-to-end |
| --- | ---: | ---: | ---: |
| Cuboid sweep, bound 50, full exact coverage | 0.656 ms | 160.628 ms | 1,729.143 ms |
| Real convolution, supplied two-vector example | 739.110 ms | 212.225 ms | 2,355.411 ms |

These are single observations on this runtime, not benchmarks across machines or
calibrated percentiles. Interpreter startup dominates the small exact example.
Queue/provisioning/network transfer are inapplicable to local direct dispatch.
Full per-stage records, dependency freezes and limits are in qualification.json.

| Design case | Local evidence | Remaining gate |
| --- | --- | --- |
| LAMBDA-01 | Lost launch acknowledgement, journal restart, unique ownership recovery and no duplicate launch; duplicate/missing ownership retains budget | Live account tag visibility/consistency and disconnected failure tests |
| LAMBDA-02 | Unhealthy guest and terminating response retain unresolved resource/billing; only provider terminated state confirms release | Live termination latency and billing |
| LAMBDA-03 | Strict refusal; prearmed watchdog on separate journal recovers after main client closes | Independently deployed host availability/failure-domain qualification |
| LAMBDA-04 | Control key only enters dedicated API bridge; sanitized provider records and worker environment | Deployment credential isolation review |
| LAMBDA-05 | Termination timeout retains exact UUID, restart observes pending termination without relaunch | Live provider cancellation/reconciliation behavior |
| RELEASE-01/03 | Clean wheel/sdist installs, runtime schema/entrypoints/assets and artifact hygiene | Other OS/Python combinations and remote CI |
| RELEASE-02 | Pinned SDK tests; remote profiles remain explicitly experimental | Live SDK/image/provider conformance |
| RELEASE-04/05 | README/skill updates, strict schema catalog and replay negative/secret-sentinel tests | Broader user deployment review |
| RELEASE-06 | Actual installed local per-stage measurements; no speedup or price claims | Authorized remote/GPU calibration |

No live SSH server outside this workspace, Slurm daemon, Lambda/Modal/Runpod/S3
account, GPU or bill was used. The encrypted loopback SSH journey runs real gateway,
worker and verifier subprocesses against a test-only server. Provider services and
failure modes are deterministic fixtures. No Studio UI/browser changes are included.

## Remaining work

Qualify actual remote deployments before promoting their profiles to stable. Native
SSH GPU support, Apptainer, wider operation/artifact eligibility, Studio compute/MCP
integration, caches/checkpoints, distributed algorithms and provider billing feeds
remain explicit extensions. R8 hardening is not a claim that those extensions have
been implemented. The core and local release can remain useful while remote
configurations are experimental.
