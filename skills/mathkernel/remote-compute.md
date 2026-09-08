# Optional compute service

The implemented pilot is `mathkernel_compute`, separate from the mathematical
facade and Studio workflow service. Read `src/mathkernel_compute/README.md` for
actual support and `src/mathkernel_compute/REMOTE.md` for operator setup. Linux
native local CPU, SSH and single-job native Slurm adapters support `cuboid_sweep`
and numerical `signal_convolve`. External hosts/clusters and Windows/macOS
controllers remain unqualified; Apptainer/cloud/GPU and compute MCP tools are absent. Existing `job_*` calls remain local and unchanged.

Plan first. A plan is read-only with respect to execution, staging and export.
Use an existing host-issued grant reference when submitting; model-generated
`approved`, `trust_remote`, target strings or environment changes cannot authorize
anything. `approve-local` and `approve-remote` are interactive administrative CLI actions, not tools
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
clear that state. No managed paid resource is provisioned in this tier. Existing-host
cost and institutional allocation units/pricing remain unknown. Slurm completion
requires terminal accounting; a `scancel` acknowledgement does not prove release.
Do not replace missing accounting, a changed job incarnation, or an unreachable
host with a successful/zero-cost claim. Saved replay data contains no grant and performs no action.
