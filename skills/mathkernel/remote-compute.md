# Optional compute service

The implemented pilot is `mathkernel_compute`, separate from the mathematical
facade and Studio workflow service. Read `src/mathkernel_compute/README.md` for
actual support, `src/mathkernel_compute/REMOTE.md` for native operator setup and
`src/mathkernel_compute/MANAGED.md` for experimental managed targets. Linux
native local CPU, SSH and single-job native Slurm adapters support `cuboid_sweep`
and numerical `signal_convolve`. External hosts/clusters and Windows/macOS
controllers remain unqualified. Experimental Modal Sandbox and existing Runpod
queue adapters support explicitly approved paid execution and durable output;
Runpod/local CPU also support bounded independent batches. Only the integer sweep
has explicit managed CUDA eligibility; live GPU/billing qualification is pending.
Apptainer, VM provisioning and compute MCP tools are absent. Existing `job_*` calls
remain local and unchanged.

Plan first. A plan is read-only with respect to execution, staging and export.
Use an existing host-issued grant reference when submitting; model-generated
`approved`, `trust_remote`, target strings or environment changes cannot authorize
anything. `approve-local`, `approve-remote`, `approve-managed` and `approve-batch`
are interactive administrative CLI actions, not tools
for an agent to mint its own authority. Do not call the private host authorization
entrypoint to bypass the user's host workflow. A local grant never permits export.
Remote approval binds the full plan, input bundle and operator target-profile
digests. Do not mint or alter SSH/Slurm profiles from model request fields. Read-only
discovery must use `reconcile_on_open=False`; recovery is explicit in the CLI.

Keep the same client request ID when reconciling an uncertain submission.
`SUBMISSION_UNKNOWN` is not a failed job to replace. A wait timeout does not cancel
work. `status` reads retained facts; `reconcile` observes the supervisor and collects
bounded output. Poll with a finite caller deadline. `cancel` records intent; check
resource cleanup independently of execution and verification.

`candidate()` returns unverified data. Only `verify()` followed by
`accepted_result()` supplies local evidence. A worker's formal, verified, image,
role or support-path metadata has no admission authority. Exact witness checks do
not establish full search coverage; numerical residuals do not establish exactness
or interval certification. A verifier timeout is `INCONCLUSIVE`.

Client close preserves host-owned runs. A lost supervisor retains cleanup
uncertainty and its concurrency reservation. Do not delete journal/spool data to
clear that state. Managed Sandbox/invocation creation requires a distinct paid grant
bound to an operator quote, image, account, resource shape and durable store. An
SSH export grant cannot authorize spending. Aggregate reservations, uncertain
exposure and reviewed spend remain in the coordinator's native USD budget ledger.
This is not an account-wide provider hard cap. Cleanup never silently sets the
bill to zero; `USER_RECONCILED` billing requires host-reviewed cost/storage records.
Never use that administrative entrypoint to fabricate a settlement or unblock a
budget. Existing-host
cost and institutional allocation units/pricing remain unknown. Slurm completion
requires terminal accounting; a `scancel` acknowledgement does not prove release.
Do not replace missing accounting, a changed job incarnation, or an unreachable
host with a successful/zero-cost claim. Saved replay data contains no grant and performs no action.

Runpod's worker receipt may name a job, but cannot grant cancellation authority;
only the provider-confirmed submission handle can do that. A missing status or
lost acknowledgement can coexist with a durable candidate and unresolved cleanup.
Never resubmit, purge a queue, mutate warm workers or delete an endpoint to clear
the uncertainty. Modal recovery checks stable-backend IDs and exact ownership tags.

An independent batch has 1–16 fixed, uniquely identified complete requests on one
target. Its full reservations and shard attempts commit atomically before dispatch.
Verify each shard separately; the batch receipt is not an aggregate mathematical
result. Duplicate delivery, repeated identical outputs and caches do not establish
statistical independence or search coverage. Random streams, reducers and checkpoint
resume are not implemented and must not be synthesized by request fields.
