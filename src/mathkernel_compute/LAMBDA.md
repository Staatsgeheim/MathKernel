# Owned Lambda instances (experimental)

`ComputeClient.vm` manages one job-owned VM at a time. It shares the existing
coordinator's USD reservation ledger, but VM provisioning has its own immutable
plan, grant and lease. Ordinary execution/export grants cannot create a VM.
Existing Lambda machines can still use the ordinary SSH adapter without granting
MathKernel authority to terminate them.

This implementation delegates mathematical execution to the native **CPU-only SSH
profile**: integer cuboid sweeps and numerical convolution. Renting a GPU machine
does not enable CUDA through this adapter. Modal/Runpod have separate experimental
CUDA profiles. No live Lambda account or paid VM was used to qualify this release.

## Operator prerequisites

Install `mathkernel[compute]` on the controller and worker. Use an isolated Lambda account/workspace,
an existing SSH key, an existing restrictive firewall ruleset, and an explicit
image ID. The runtime object comes from the trusted worker installation as
described in [REMOTE.md](REMOTE.md). It is checked again by the SSH gateway.

The profile is trusted host configuration. Its schema is available under
`LambdaProfile` in `mathkernel-compute schemas`. Required fields are:

| Field | Meaning |
| --- | --- |
| `target_id` | Local alias; later SSH profile uses this same alias |
| `account_scope`, `budget_id` | Reviewed account and durable local USD ledger |
| `region_name`, `instance_type_name` | Exact deployment choice |
| `image_id` | Provider image UUID, never a floating family/default |
| `ssh_key_name`, `firewall_ruleset_id` | Existing provider objects; no key/firewall creation |
| `runtime` | Pinned native worker runtime, without accelerator metadata |
| `quote` | Reviewed USD reservation, scope/source and expiry; include startup, idle and transfer exposure |
| `lease_ms` | 120,000–3,600,000 milliseconds after the latest authorized launch time |
| `credential_env` | Default `MK_LAMBDA_API_KEY`; credential value stays out of documents |
| `cleanup_policy` | Default `strict`; opt-in `managed-exposure` for this experimental implementation |

No `user_data`, arbitrary commands, filesystem mounts, URLs, key material or
worker secrets are accepted. An image ID alone does not authenticate an SSH host.
Install the gateway through your trusted administration path and obtain its host
key independently. `vm-attach` requires strict known-hosts verification, the
provider-observed IP, the approved runtime and the installed gateway configuration.
There is no automatic SSH key scanning or trust-on-first-use shortcut.

## Lifecycle

Use the same `--state-dir`, `--lambda-profile lambda.json`, and
`--budget-limit budget.json` for controller commands. Add
`--target-profile ssh.json` once the gateway is installed and authenticated.

1. `vm-plan lambda-pilot` creates an offline reviewable plan and lease ID.
2. Export `vm-watchdog-ticket LEASE_ID` and arm it on the independent watchdog
   host **before launch**, using the procedure below.
3. `vm-approve LEASE_ID --acknowledge-exposure --acknowledge-broad-key
   --acknowledge-egress --acknowledge-output-loss` requires typing the complete
   plan digest. It grants one VM launch and its eventual termination, not input export.
4. `vm-launch LEASE_ID --grant GRANT_ID --request-id UNIQUE_ID` commits authority,
   reservation and intent before its one provider launch call. Repeating it returns
   the original lease. `vm-observe LEASE_ID` polls readiness without another launch.
5. Complete trusted gateway installation, then `vm-attach LEASE_ID lambda-pilot`.
   Use the ordinary `plan`, `approve-remote`, `submit` flow for a request with that
   target. There is exactly one mathematical job per lease.
6. Repeated `vm-reconcile LEASE_ID` observes that job, retrieves and integrity-checks
   its candidate in the local durable quarantine, then requests provider termination.
   Verification/admission remains the separate ordinary `verify JOB_ID` operation.
7. Keep reconciling until provider status confirms termination. Use
   `vm-reconcile-cost LEASE_ID --usd AMOUNT --source DESCRIPTION --storage-accounted`
   only after reviewing actual charges. Termination never silently closes the bill.

`vm-status` reads retained facts. `vm-cancel` records termination intent and may
discard unretrieved output. Lease expiry similarly prioritizes cleanup. A completed
job without a retrievable candidate keeps the VM until retrieval succeeds or its
deadline/cancellation requires cleanup. `ComputeClient.close()` does not terminate
anything; a scheduler must keep reconciling. There is no automatic paid replacement.

## Independent watchdog

On a separately installed, user-owned control host in a different failure domain,
install the same package and a reviewed Lambda profile. Keep the provider key on
that control host, never inside the rented guest. Transfer the **data-only ticket**
through your own trusted channel. The ticket contains no API key or mathematical
inputs and does not authorize itself.

```sh
mathkernel-compute --state-dir watchdog-state --lambda-profile lambda.json watchdog-arm ticket.json --acknowledge-independent-host
mathkernel-compute --state-dir watchdog-state --lambda-profile lambda.json watchdog-tick
```

Arming requires a separate terminal digest confirmation on that host. Install the
one-shot `watchdog-tick` command in your own periodic scheduler (for example every
minute); use absolute paths, protected credentials, monitoring and synchronized
clocks. It needs no access to the controller journal, guest, worker secrets or
candidate store. Repeated ticks persist termination intent and reconcile the exact
instance UUID plus ownership tags, region, image, SSH key and filesystem scope.
A unique authenticated tag match can recover a lost launch acknowledgement;
missing, duplicate or changed ownership remains unresolved. It never deletes by
name prefix or creates a replacement.

This deployment is **operator-acknowledged, not machine-attested**. The library
cannot establish failure-domain independence or guarantee tag lookup/cleanup during
an outage. The watchdog might itself fail, termination can lag, and a lost launch
may remain undiscoverable. Therefore strict provisioning remains rejected even if
a ticket exists. Managed exposure explicitly acknowledges those limits. No native
expiry or provider monetary ceiling is claimed.

## Provider contract and qualification

The interactive [Lambda Cloud API reference](https://docs-api.lambda.ai/api/cloud)
was checked on 9 September 2026 (OpenAPI version 1.10.0). This adapter uses its
Bearer-authenticated launch/list/get/terminate subset, image IDs, ownership tags
and existing firewall references. It sends one launch request; there is no
undocumented `quantity` field or automatic retry. Calls have bounded output and a
30-second parent deadline, disable proxies/redirects, and retain no Jupyter tokens
or URLs. One coordinator spaces API requests and applies delayed reconciliation;
separate controllers still share the provider's account rate limits.

Lambda's [instance guide](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/)
warns that guest shutdown/poweroff does not stop billing. The controller uses only
provider termination; an acknowledgement or 404 is not confirmed release. Its
[access guide](https://docs.lambda.ai/public-cloud/access-security/)
describes broadly privileged API keys. Those privileges must be reviewed and kept
on isolated control hosts. These provider facts do not qualify a live deployment:
account capabilities, SSH onboarding, tag consistency, cancellation latency,
firewall behavior, independent watchdog availability and final billing remain
operator qualification gates.
