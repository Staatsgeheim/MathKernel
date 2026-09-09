# Managed compute and independent batches

The optional Modal Sandbox and Runpod Serverless adapters use the same durable
controller, scoped grants, quarantined candidates and local verifiers as native
compute. They are **experimental**: their API and failure paths have local tests,
but live GPU execution, provider billing and disconnected cloud recovery have not
been qualified. No provider account is contacted by installation, import, target
discovery or planning. `auto` remains local CPU.

## Installation and authority

Install only the controller adapter you need:

```sh
python -m pip install '.[compute-modal]'
python -m pip install '.[compute-runpod]'
```

The controller pins Modal **1.5.5** and boto3 **1.43.90**. The Runpod worker image
uses `.[compute-runpod-worker]`, which also pins Runpod **1.12.0**. A CUDA worker
additionally needs a separately locked CuPy/CUDA environment. SDK version changes
require review; the adapters fail closed on an unqualified version.

Provider profiles and budget limits are installed by the host/operator, never
supplied inside mathematical requests. Profiles contain credential **environment
variable names**, not credential values. Keep configuration and the private
controller journal under the same trusted host account. One coordinator owns one
local state directory; a shared/network SQLite directory is unsupported.

Each profile pins a target alias, account scope, budget ID, native worker runtime,
immutable source OCI image, resource shape and a reviewed USD reservation quote.
The quote must include its source, scope and expiry; it is an exposure allowance,
not invented provider pricing or a guaranteed maximum bill. Changing the profile,
quote, limit, policy, input or verifier runtime invalidates unsubmitted approvals.
Retain old profiles while their attempts need reconciliation. Use new aliases for
new deployment profiles instead of replacing an active attempt's identity.

`RuntimeProfile` is obtained from the installed worker environment with
`mathkernel_compute.registry.runtime_profile()`. For an explicitly authorized GPU
worker, use `runtime_profile(gpu=True)` to include CuPy and CUDA runtime/driver
versions. Pin the resulting profile in controller configuration. Image building,
account setup and any GPU allocation needed for onboarding are separate operator
actions; a read-only target probe does not launch a container to discover them.

## Modal Sandbox

Install a prebuilt image containing the exact MathKernel wheel and its locked
dependencies. `/opt/venv/bin/python` must run that environment and `/usr/bin/sync`
must be present. Record both its immutable source OCI reference and the Modal
image ID produced from it. Create an app and a **Volume v2** through your normal
administrative process, then record their names and immutable IDs. The adapter
never creates an app, volume, image build or remote Python function.

`ModalProfile` adds these fields to the common profile:

| Field | Meaning |
| --- | --- |
| `app_name`, `app_id`, `environment` | Existing app in an explicit Modal environment |
| `image_id`, `source_image` | Existing Modal image and its source OCI digest |
| `volume_name`, `volume_id`, `volume_version: 2` | Existing durable user-owned volume |
| `region` | Explicit allocation region |
| `credential_env_prefix` | For example `MK_MODAL`; host provides `MK_MODAL_TOKEN_ID` and `MK_MODAL_TOKEN_SECRET` |
| `resources.network` | Must be `blocked` |

Submission stages only the canonical attempt to its workspace/attempt prefix.
The Sandbox mounts that subdirectory at `/mk-data`, receives no account secrets,
blocks network access, opens no ports and runs the fixed broker command. The broker
runs the registered scientific operation in a clean child process. It persists
the candidate before its terminal manifest using Volume v2 `sync`, then exits.
CPU is specified in physical-core units; request and hard limit are equal, as are
the memory request and limit. The maximum Sandbox lifetime is explicit.

The implementation uses the stable Sandbox backend. It rejects beta-shaped
Sandbox IDs before attaching or cancelling. Reconciliation may recover a running
Sandbox by its stable name, then checks workspace/attempt/execution/profile tags.
If the create acknowledgement was lost and the Sandbox has already disappeared
from name lookup, output may still be retrieved from the volume, but ownership
and billing remain unknown. It never issues another create to resolve ambiguity.

Only SDK byte/protobuf control paths are used: app/volume/image lookup, volume
upload/read, Sandbox create/name/ID lookup, tags, poll and terminate. There is no
Functions `.remote()`, Python return-value, pickle or exception-deserialization
path for mathematical output. Modal 1.5.5's volume reader can prefetch whole
blocks; therefore SDK calls run in a separate Linux process with a 1 GiB address
space limit, 30 CPU-second limit, 45-second wall deadline and bounded framed
output. A size check also precedes reading. Failure of any limit retains unknown
state rather than weakening the output boundary.

These choices follow the [Sandbox API](https://modal.com/docs/sdk/py/latest/Sandbox)
and [Sandbox filesystem/Volume v2 sync contract](https://modal.com/docs/guide/sandbox-files).
Live volume lifetime, sandbox cancellation and resource-limit tests are still
required before calling a deployment qualified.

## Existing Runpod queue endpoint

Deploy the pinned worker image to an existing **QUEUE** endpoint. Its fixed
entrypoint is:

```sh
/opt/venv/bin/python -m mathkernel_compute.managed_worker runpod
```

Install `/etc/mathkernel/runpod-worker.json` as a `RunpodWorkerConfig` with the
target/account/endpoint identity, source image, runtime, resources, S3 storage
configuration and allowed workspace IDs. The `workspace` CLI command returns the
controller's persistent workspace ID. The handler accepts only the registered
attempt reference and request digest; it cannot select code, an image, a command,
an endpoint or a download URL. Realtime/API worker mode is refused.

The existing S3 bucket is a commercial AWS S3 bucket in an explicit region. `S3Storage`
contains `bucket`, `region`, an identifier-only `prefix` and a credential variable
prefix. For prefix `MK_ARTIFACT`, provision `MK_ARTIFACT_ACCESS_KEY_ID`,
`MK_ARTIFACT_SECRET_ACCESS_KEY`, and optionally `MK_ARTIFACT_SESSION_TOKEN` in the
controller and broker environment. Restrict their IAM access to the required
workspace prefixes and object operations; do not grant endpoint/account control
credentials to the scientific child. No arbitrary S3 endpoint, caller-selected
URL, signed URL or provider `s3Config` credential object is sent in a request.
Ambient SDK endpoint overrides are disabled, and each outgoing S3 request is
checked against the pinned bucket's regional HTTPS hostname before transmission;
an SDK region redirect is refused. Other AWS partitions/custom endpoints are unsupported.

`RunpodProfile` additionally records:

| Field | Meaning |
| --- | --- |
| `endpoint_id`, `source_image` | Existing queue endpoint and immutable OCI image |
| `deployment_digest` | Digest returned by an explicit, read-only endpoint probe |
| `workers_min`, `workers_max`, `idle_timeout_seconds` | Reviewed warm-worker cost scope |
| `storage` | The existing S3 store described above |
| `credential_env_prefix` | For example `MK_RUNPOD`; controller provides `MK_RUNPOD_API_KEY` |
| `resources.network` | Must be `broker-only`; strict scientific network isolation is unsupported |

A probe returns a sanitized deployment snapshot and digest without returning the
endpoint's secret environment values. Before export, submission checks that the
image, worker counts, idle timeout, placement, compute selection and other retained
deployment settings still match. A GPU profile requires exactly one pinned GPU
pool and one GPU. The adapter never deploys, mutates, purges or deletes an endpoint.

`/run` receives explicit millisecond `executionTimeout` and total `ttl` policies.
The operation's own Linux supervisor has an additional, shorter execution deadline;
start authority expires independently of queue/runtime limits. The synchronous
provider result API is not used. The broker records receipt and an atomic S3
`If-None-Match: *` claim before executing mathematics. Duplicate delivery does not
rerun the operation, and an ambiguous claim fails closed. Outputs and the terminal
manifest are immutable; the terminal manifest is uploaded only after the candidate.

Durable candidates remain available when provider status/result retention expires.
A 404 cannot establish non-execution, release, failure or zero spend. The adapter
has no qualified provider lookup by our attempt key. A worker-written job ID is
advisory and **cannot authorize cancellation**. If the submission acknowledgement
is lost, durable retrieval may succeed while resource/billing exposure remains
unresolved. No replacement request is sent. Cancellation writes a scoped durable
control marker and addresses only the original provider-confirmed job ID, if known.
An acknowledgement does not confirm that the scientific process stopped.

Runpod's finite result retention and queue operations are documented in its
[operation reference](https://docs.runpod.io/serverless/endpoints/operation-reference);
the deployment probe uses the
[v2 endpoint API](https://docs.runpod.io/api-reference-v2/serverless/get-a-serverless-endpoint).
Stopping a job never stops existing warm workers or S3 storage charges.

## Paid approval, ledger and CLI

The host budget policy is a `BudgetLimit`, for example:

```json
{"budget_id":"research-pilot","account_scope":"my-account","limit":{"currency":"USD","amount":"5.00"}}
```

This is a user-selected budget, not a price estimate. The controller aggregates
open reservations, uncertain exposure and recorded spend for that budget. Unknown
outcomes and successful cleanup both retain the reservation until billing/storage
accounting is reviewed. New grants cannot reset it. Budget/account identity is
retained across restarts; rebinding a budget to hide old exposure is refused.
The ledger covers **this coordinator only**, not every controller or other spending
in the provider account, and it is not a provider-enforced hard monetary cap.

Use the same profile, budget and state arguments for each command:

```sh
mathkernel-compute --state-dir ./state --managed-profile ./target.json --budget-limit ./budget.json targets
mathkernel-compute --state-dir ./state --managed-profile ./target.json --budget-limit ./budget.json probe TARGET
mathkernel-compute --state-dir ./state --managed-profile ./target.json --budget-limit ./budget.json plan request.json
mathkernel-compute --state-dir ./state --managed-profile ./target.json --budget-limit ./budget.json approve-managed PLAN_ID --acknowledge-exposure --allow-storage
mathkernel-compute --state-dir ./state --managed-profile ./target.json --budget-limit ./budget.json submit PLAN_ID --grant GRANT_ID --request-id experiment-1
```

Approval prints the complete plan and requires its digest at a human terminal.
The private `_authorize_managed` Python method is a trusted host integration point,
not model authority. A local or SSH export grant cannot authorize paid execution.
`reconcile`, `fetch`, `verify`, `result`, `cancel` and `budget BUDGET_ID` expose the
separate lifecycle and evidence states. Read-only CLI commands open with automatic
reconciliation disabled; `reconcile` can dispatch already authorized pending work.

After confirmed release and review of actual costs **including retained storage**,
the host can use `reconcile-cost JOB_ID --usd AMOUNT --source SOURCE
--storage-accounted` with the same global configuration arguments. This requires
terminal confirmation and records an immutable **USER_RECONCILED** attestation,
not provider-verified billing. Actual overspend is recorded and blocks further
admission; it is not clamped to the quote. There is no automatic artifact deletion,
invoice ingestion, currency conversion or budget reset.

## Independent batches and GPU eligibility

`BatchRequest` contains 1–16 uniquely named shards. Each shard is a complete,
independent `ComputeRequest`; the initial batch targets are local CPU and an
existing Runpod endpoint. A batch has one target and a finite manifest. Example:

```json
{"shards":[{"shard_id":"bound-40","request":{"operation":"cuboid_sweep","parameters":{"bound":"40"},"target":"my-runpod"}},{"shard_id":"bound-50","request":{"operation":"cuboid_sweep","parameters":{"bound":"50"},"target":"my-runpod"}}]}
```

Use `plan-batch FILE`, `approve-batch PLAN_ID --allow-export
--acknowledge-exposure --allow-storage`, then `submit-batch PLAN_ID --grant GRANT_ID
--request-id REQUEST_ID`. As above, supply the global state/profile/budget arguments.
Local-only batch approval needs no export/exposure/storage flags.

All shard jobs, bindings, grants and monetary reservations are committed in one
transaction before any dispatch. A failed aggregate budget check rolls back the
whole submission. Pending shards respect the existing concurrency limit; cancelled
or expired undispatched shards release their reservation without export. Stable
batch/shard/attempt bindings reject swapped or duplicated contributions.
`batch-status`, `reconcile-batch` and `cancel-batch` operate on the retained manifest.
Call `verify` for each received shard to obtain its own locally admitted result.

The batch result is a coverage/receipt view, not an aggregate `MathResult`. It
explicitly reports expected and admitted shards. There is no distributed graph,
MPI, checkpoint/resume, random-stream generator, statistical independence claim,
mathematical reducer or cross-workspace cache. These unsupported fields are
rejected rather than inferred from repeated outputs.

Only `cuboid_sweep` has a registered GPU path: select `parameters.engine: "cuda"`,
`resources.gpu_count: 1`, and an explicit GPU target with a pinned accelerator
runtime. It calls the existing integer CUDA implementation with bounded hit storage;
exact claims still require the independent local CPU verifier. Python/Scipy requests
require zero GPUs. Neither `auto` nor `signal_convolve` silently allocates a GPU or
changes arithmetic. GPU execution and performance remain live qualification gates.

## Upgrade and qualification

Journal schema 2 adds managed handle/observation, cost settlement and batch tables
transactionally. Existing schema 1 payloads and absent-field canonical identities
are preserved. Back up the private state directory before upgrading; older clients
refuse schema 2. A source/runtime upgrade never silently reauthorizes pending plans.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the executed local checks and remaining
design gates. Live provider setup must still qualify data retention, GPU runtime,
actual resource limits, controller loss, cancellation/cleanup and billing lag.
Windows/macOS managed controllers, Lambda provisioning, generic array artifacts,
distributed execution and Studio/MCP compute integration are outside this checkpoint.
