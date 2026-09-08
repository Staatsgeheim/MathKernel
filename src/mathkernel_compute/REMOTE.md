# Existing-host SSH and native Slurm setup

These adapters execute the two registered CPU operations on explicitly installed
Linux workers. The controller retains the journal and performs mathematical
verification locally. They create no hosts, install no services, discover no
credentials and provision no paid resources. `auto` continues to select local CPU.

An SSH profile is trusted operator configuration, not a compute-request field.
Possession of a profile alone does not authorize input export. Every submission
also requires a host-issued, single-attempt grant for the exact plan, bundle and
target-profile digest. Grant expiry bounds **when mathematics may start**; the
operation has its own execution deadline. Keep controller/worker clocks synchronized.

## Install and onboard

Install the same reviewed MathKernel source/wheel with its `compute` extra on the
controller and worker. Dependencies and Python versions may differ across hosts:
the plan pins the worker profile and the local verifier profile separately. Use a
stable virtual environment on the worker; source/dependency changes require a new
profile and new plans. Never copy a controller kernel object or pickle to a worker.

First obtain the controller workspace ID without contacting a target:

```sh
mathkernel-compute --state-dir ./compute-state workspace
```

On the worker, as the account that will own the jobs, create a dedicated **private,
non-symlink** spool and an operator-owned configuration file. This explicit setup
example creates a native SSH target; replace the paths and workspace ID:

```python
from pathlib import Path
from mathkernel_compute.remote import WorkerConfig
from mathkernel_compute.registry import runtime_profile
from mathkernel_compute.protocol import canonical, digest

spool = Path('/home/compute/mathkernel-spool')
spool.mkdir(mode=0o700)  # Dedicated directory; do not reuse a shared working directory.
config = WorkerConfig(
    target_id='research-cpu', adapter='ssh', spool_root=str(spool),
    allowed_workspaces=('WORKSPACE_ID_FROM_CONTROLLER',),
)
Path('/home/compute/worker.json').write_bytes(canonical(config))
print(canonical({'worker_config_digest': digest(config),
                 'runtime': runtime_profile().model_dump(mode='json')}).decode())
```

Obtain the SSH host public key through an authenticated administrator/console
channel and pin it in a dedicated known-hosts file. `ssh-keyscan` output by itself
is not an authenticated onboarding decision. Install a dedicated authentication
key or use an existing operator-approved identity file. This profile supports
noninteractive key authentication; agent forwarding and interactive password
prompts are disabled. Protect the private key with normal operating-system access
controls. A dedicated account/forced command can further restrict SSH access;
configure that through the site's normal administrative process.

Create the controller's `research-cpu.json` using `SSHProfile`. All connection
settings come from the operator. `runtime` and `worker_config_digest` are the
reviewed values from the installed worker, not automatically trusted probe output:

```python
from pathlib import Path
from mathkernel_compute.remote import SSHProfile
from mathkernel_compute.models import RuntimeProfile
from mathkernel_compute.protocol import canonical

# Populate these with the authenticated onboarding record above.
runtime_record = ...
config_digest = ...
profile = SSHProfile(
    target_id='research-cpu', host='compute.example.org', account='compute', port=22,
    known_hosts='/home/controller/compute_known_hosts',
    identity_file='/home/controller/compute_identity',
    worker_executable='/home/compute/venv/bin/mathkernel-compute-gateway',
    worker_config='/home/compute/worker.json', spool_root='/home/compute/mathkernel-spool',
    runtime=RuntimeProfile.model_validate_json(canonical(runtime_record)), worker_config_digest=config_digest,
)
Path('research-cpu.json').write_bytes(canonical(profile))
```

Remote executable/config/spool paths must be absolute POSIX paths containing only
letters, digits, `_`, `-`, `.`, and `/`, without `..` components. Local identity and
known-hosts paths must be absolute without whitespace or SSH `%` expansion tokens.
The current profile deliberately does not expose arbitrary SSH options, shell
fragments, aliases, proxy commands, environment exports, passwords or URLs.

`ssh -F none` excludes ambient configuration. The adapter sets strict host-key
checking, batch mode, explicit identity and known-hosts sources, disables forwarding
and multiplexing, and invokes only the fixed installed gateway/config command.
Requests use bounded length-prefixed JSON on stdin. Consult the official
[OpenSSH command/configuration semantics](https://man.openbsd.org/ssh.1).

## Probe, plan and approve export

A probe authenticates the host and compares its installed configuration/runtime
against the pinned profile. It does not submit a job, stage mathematical inputs or
run a smoke calculation. For Slurm, it also checks required CLI help/version profiles.
Unsupported flags or changed identities fail closed.

```sh
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json probe research-cpu
```

Set `"target":"research-cpu"` in an otherwise normal compute request. Then:

```sh
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json plan request.json
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json approve-remote PLAN_ID
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json submit PLAN_ID --grant GRANT_ID --request-id stable-request-1
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json reconcile JOB_ID
mathkernel-compute --state-dir ./compute-state --target-profile research-cpu.json verify JOB_ID
mathkernel-compute --state-dir ./compute-state result JOB_ID
```

The interactive approval prints the full plan, including inputs, target digest,
retention and cost limitations, and requires the plan digest to be typed at a
terminal. It is a host administration action, not a capability exposed to an agent.
Host applications can call `_authorize_remote()` only after their own approval
workflow has authorized the exact plan, bundle and target-profile digests. No grant
secret or SSH private key is sent to the worker. SSH authenticates the configured
controller account; the worker additionally enforces its workspace allowlist,
finite spool/active-attempt limits, runtime identity and start deadline.

The worker's gateway journal commits an attempt intent before any launch. A repeat
of the same intent returns its retained identity. Conflicting bytes are rejected.
A crash between intent and launch can remain unknown; neither controller nor worker
creates a replacement attempt to resolve uncertainty. Status/cancel/fetch requests
carry only an attempt reference, never a fresh copy of the mathematical inputs.

## Slurm site profile

Use `adapter='slurm'` in both configurations and add a `SlurmConfig` to the worker
configuration. Set the controller SSH profile's `allocation` to
`config.slurm.allocation()` from that reviewed site profile; probe compares the
full allocation shape as well as the configuration digest. The spool and reviewed native virtual environment must be available
at the same absolute paths on the login node and allocated compute node. This is a
single local-cluster profile; federated submissions, arrays and MPI are excluded.

```python
from mathkernel_compute.remote import WorkerConfig, SlurmConfig

config = WorkerConfig(
    target_id='institution-cpu', adapter='slurm',
    spool_root='/home/compute/mathkernel-spool',
    allowed_workspaces=('WORKSPACE_ID_FROM_CONTROLLER',),
    slurm=SlurmConfig(partition='cpu', account='research', qos='short',
                      memory_mib=1024, walltime_seconds=180),
)
```

The plan displays the export host/port, SSH account and effective allocation shape.
Each configured target supplies one approved partition/account/QOS/constraint and
resource shape; create distinct reviewed target aliases for other shapes. Requests
cannot supply directives. The profile uses one node, one task, one CPU and no GPU.
Memory is explicitly MiB. Slurm walltime includes worker startup and cleanup in
addition to the operation's at-most-60-second deadline. Slurm's configured memory
limit is not advertised as a verified hard RAM isolation guarantee.

The fixed batch script is passed to `sbatch --parsable`, with `--no-requeue` and
`--export=NIL`. Mathematical work runs only after Slurm launches that script.
`NIL` avoids implicit login-environment loading; sites must support that export
mode. The script starts a fresh allowlisted worker environment and preserves only
the scheduler job ID/restart count needed for incarnation checks. See the official
[`sbatch` contract](https://slurm.schedmd.com/sbatch.html).

Active state comes from explicit `squeue` fields. Terminal allocation facts and
usage come from explicit pipe-delimited `sacct` fields, including `Restarts`.
**This profile requires retained accounting with
`AccountingStoreFlags=job_comment`**, so the full attempt digest remains available
for ownership checks. A site without these fields remains unqualified and reports
unknown state; it does not fall back to parsing a human table. See
[`squeue`](https://slurm.schedmd.com/squeue.html) and
[`sacct`](https://slurm.schedmd.com/sacct.html).

Cancellation checks the UUID job name, full attempt digest, account UID, scheduler
job ID and recorded submit incarnation, then also applies scheduler-side UID/name
filters to `scancel`. Its acknowledgement is only a cancellation request. Missing
accounting, duplicate incarnations, changed submit times or requeue counts retain
uncertainty. Forced requeues cannot launch mathematics again: the batch incarnation
check and exclusive accepted marker reject them. See
[`scancel`](https://slurm.schedmd.com/scancel.html).

The worker refuses mathematics if the allocation starts after authorization
expiry. This is a start gate, not a scheduler promise to remove a queued job at that
instant while the controller is offline. Scheduler allocation overhead can still
occur and is retained as usage. Sites requiring stronger unattended queue cleanup
need their own qualified scheduler policy/watchdog before enabling that mode.

OOM, timeout, cancellation, preemption, node failure, expiry and nonzero exit remain
distinct. A completed allocation without a complete worker result does not become
a mathematical success. The journal reserves the approved maximum CPU seconds and memory shape separately
from unknown allocation credits. CPU seconds and allocated TRES are retained when supplied
by accounting; institutional allocation units and currency pricing remain unknown.
No provider-budget or zero-cost guarantee is inferred from an existing allocation.

## Recovery, retention and qualification

Keep the original state directory and target profile. A missing/changed profile
on restart permits reading retained facts but blocks new remote effects. An
unreachable host exposes `transport=UNAVAILABLE` and unresolved resources as
`CLEANUP_UNKNOWN`, without rewriting the last confirmed execution observation.
Restoring the pinned profile/connectivity allows reconciliation of the same attempt.
Do not clear a journal or regenerate an attempt to bypass retained uncertainty.

Inputs and candidates remain in the private remote spool until an operator removes
an **already released, retained** attempt through the site's retention process.
This version has bounded admission quotas and no automatic deletion/GC API. Never
remove live/unknown attempts, shared caches, or a host working directory. After
local fetch, independent verification can continue without remote connectivity.
Candidate bytes are base64-wrapped for transport, bounded, and decoded into local
quarantine before any mathematical model is invoked. Candidate DTO/output limit is
1 MiB; its transport wrapper is bounded separately at 2 MiB. Evidence is constructed
only by the controller's verifier.

Linux/Python 3.12 tests exercise the actual OpenSSH client against a local Paramiko
SSH server and real native worker subprocesses. Slurm tests use a scheduler
simulator and run the real generated batch script. These establish implementation
and protocol behavior, **not qualification of an external OpenSSH host or Slurm
cluster**. Windows/macOS controller code paths are present but not qualified here.
Apptainer/SIF profiles, GPU profiles, job arrays/MPI, paid cloud providers, independent
host-failure watchdogs, automated retention and Studio/MCP compute integration remain
separate implementation/qualification work. No container support is advertised by
these native profiles.
