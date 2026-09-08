# Optional compute service

The implemented pilot is `mathkernel_compute`, separate from the mathematical
facade and Studio workflow service. Read `src/mathkernel_compute/README.md` for
actual support. Only Linux native local CPU, `cuboid_sweep` and numerical
`signal_convolve` are registered. No SSH/cloud/GPU executor or compute MCP tool is
implemented yet. Existing `job_*` calls remain local and unchanged.

Plan first. A plan is read-only with respect to execution, staging and export.
Use an existing host-issued grant reference when submitting; model-generated
`approved`, `trust_remote`, target strings or environment changes cannot authorize
anything. `approve-local` is an interactive administrative CLI action, not a tool
for an agent to mint its own authority. Do not call the private host authorization
entrypoint to bypass the user's host workflow.

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
clear that state. No paid resource exists in this tier; local electricity/hardware
cost remains unknown. Saved replay data contains no grant and performs no action.
