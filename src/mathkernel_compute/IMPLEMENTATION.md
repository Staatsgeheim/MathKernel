# Remote-compute implementation checkpoint

## Grounding

- Repository: `Staatsgeheim/MathKernel`; input branch `master`.
- Exact base: `bf71fc88333ee6cf455cb12538e64fc385512a8a`, merged Studio PR #1.
- Implementation branch: `mathkernel-remote`, created locally from that commit.
- Design: supplied `MathKernel_Remote_Compute_Design_Package.zip`, MK-RHC-001,
  revision 1.0, 7 September 2026. Implementation checkpoint: 8 September 2026.
- GitHub API branch creation returned 403, `Resource not accessible by integration`.
  No default-branch changes or remote publication are included in this checkpoint.

The design baseline was 1.3.0.dev30. Current master is 1.3.1.dev1 and already
contains the independent `mathkernel_workflow` service and Studio adapter. That
service returns host-owned local MathResult snapshots; it is not reused as a
remote-worker admission boundary. The new subsystem is `mathkernel_compute`, with
its own DTOs, journal, supervisor and local verifier. Development version becomes
`1.4.0.dev1`; protocol remains `1.0`, bundle `mk.bundle/1`, journal schema 1.

The live kernel still exposes `_JOB_RUNNERS` for Collatz and cuboid local jobs.
Their API and evidence behavior are unchanged. The cuboid operation's Python
engine and the typed signal module's numeric convolution are the two registered
pilot handlers. Provider SDKs are not imported, installed or initialized by the
base package. Existing formal/interval engines are not used as remote verifiers;
no claim about qualifying those paths is implied.

## Delivered boundary

This is an **R0–R2 local pilot**, not an implementation of all remote providers or
all 114 design acceptance cases. It establishes and tests the common lifecycle
and untrusted-output boundary on two self-contained operation/profile combinations.

| Area | Implemented | Deliberately not advertised yet |
| --- | --- | --- |
| Contracts | Strict frozen request/parameter/plan/spec/grant/attempt/job/lease/candidate/report/receipt models; RFC 8785 bytes; independent schema versions; full SHA-256 identities | Generic context/object dependency export, array codecs and arbitrary registry reflection |
| Planning/authority | Local static target, pinned runtime code/dependency identities, finite limits, immutable plans, host-only single-attempt grants, policy/expiry rechecks, atomic duplicate/concurrency protection | Paid grants, provider prices, currency conversion, provider spend caps, export/provisioning authority |
| Journal | Separate versioned SQLite journal, exclusive coordinator lock, integrity hashes, request and attempt uniqueness, outbox intent, events, manifests/references, reports, leases and local-zero reservations/usage | Distributed coordinator, network-filesystem journal, authenticated multi-client server |
| Worker | Fixed bounded JSON framing, allowlisted handlers/environment, detached Linux supervisor, private retained output, native worker deadline, process-group cleanup with subreaper adoption | Windows Job Objects, macOS qualification, network/RAM isolation, GPU quotas, independent external watchdog |
| Recovery | Controller-process exit, pending/ambiguous intent, idempotent request lookup, visibility delays, immutable candidates, monotonic observations, cleanup uncertainty, stale verifier state | Universal exactly-once promise, automatic replacement of unresolvable launches, provider accounting |
| Admission | Quarantine before mathematical models, exact problem/attempt binding, integer witness checking, independent complete Euclid enumeration, Decimal numerical reference, locally built evidence | Generic proof/Lean import, interval certification, arbitrary approximate ancestry graphs, typed kernel object re-import |
| Interfaces | Python compute client and explicit CLI; interactive local approval command; inert replay data; operation descriptor inventory | New MCP tools or Studio compute adapter; these existing interfaces remain unchanged |

A lost supervisor retains `CLEANUP_UNKNOWN`. There is no claim of unattended
recovery from host destruction or malicious local operator code. Only the detached
supervisor continues deadlines when the controller disappears. No cloud resource
was created, no provider account accessed, and no provider charge incurred.
Electricity/hardware cost was not measured. No test resource was intentionally
left running; fake-provider cleanup uncertainty exists only in test fixtures.

## Verification record

Completed on the supplied Linux/Python 3.12 runtime:

- 27 compute tests passed, including two real operation journeys, controller
  process exit, cancellation, deadline, verifier timeout, nested process cleanup,
  unrelated-process survival, concurrent duplicate submissions, journal rollback,
  lost acknowledgements, delayed visibility, missing output, conflicting bytes,
  stale observations, scope isolation and hostile candidate evidence.
- 56 selected legacy jobs/cuboid/signal/evidence compatibility tests passed;
  1 CUDA test skipped because a CUDA runtime was unavailable.
- 15 existing Studio workflow backend tests passed.
- Existing skill metadata validation passed.
- Wheel/sdist and installed-wheel execution evidence is recorded in the handoff's
  `verification.txt` alongside exact commands and artifacts.

The first compatibility attempt was 54 passed, 2 skipped, 1 failed: a legacy test
unconditionally requested Numba while the optional dependency was absent. Installing
the optional Numba dependency resolved it; no kernel algorithm was changed.
The full MathKernel suite, all Python versions in CI, Windows/macOS, live providers,
GPU execution and the complete design acceptance catalog were not run.

The automated cases cover portions of BASE, SCHEMA, ARTIFACT, AUTH, RECOVERY,
WORKER and VERIFY. They do not certify the full families: for example exact
string preservation is tested but typed large-integer object export is not;
private-file confinement is tested but general array/archive decoding is absent;
local concurrency reservations are tested but monetary provider budgets are not.

## Next implementation slice

R3 should add explicitly onboarded existing Linux SSH hosts using this data-only
contract, strict host-key verification, fixed commands, bounded staging and durable
supervision. First reconcile the native runtime identity across the actual target
installation, and qualify disconnect, cancellation and output retention on a real
host. Do not replace the current candidate DTO with MathResult deserialization or
add fake successful provider modules. Slurm and managed providers follow their
own design gates. Studio/MCP integration must use a host-owned coordinator rather
than letting a browser or model issue its own authorization.
