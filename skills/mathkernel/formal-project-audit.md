# External Lean project auditing

MathKernel has two intentionally separate paths: **source inspection** and
**operator-authorized Comparator/nanoda replay**. An `inspect` result is always
`UNKNOWN`. No missing tool, timeout, grep result, exit-code-only build, or
self-reported metadata is promoted to a proof.

## What is implemented and what is qualified

The Python library, MCP read-only adapters, CLI, policy validation, bounded
subprocess runner, evidence projection and test suite are implemented. The
Comparator service command is constructed and tested with fixtures; **a live
Lean/Comparator/Landrun/nanoda run has not been qualified in the development
container**, which has none of those binaries and runs as root. Successful
mocked receipts test the integration contract, not any mathematics. Treat
production replay as experimental until it has passed an independently prepared
positive/negative challenge suite on the target machine.

A real successful replay can produce claim-specific `FORMAL` evidence for the
listed Lean statements **relative to an operator-trusted reference and explicit
axiom policy**. It never asserts that those statements match a paper, model,
physical system, or Millennium problem. Report JSON is an execution record, not
a signed artifact or a self-authenticating certificate. Parsing supplied report
JSON is not a replay operation.

## Read-only inspection

```python
from mathkernel import MathKernel
from mathkernel.formal_audit import FormalProjectSpec, FormalTarget, audit_lean_project

spec = FormalProjectSpec(
    targets=(FormalTarget(module="Solution", declaration="Demo.target"),),
    expected_toolchain="leanprover/lean4:v4.34.0-rc2",
)
report = audit_lean_project("/path/to/frozen/project", spec)
assert report.trust == "unknown"

kernel = MathKernel()
result = kernel.formal_project_audit("/path/to/frozen/project", spec.model_dump(mode="json"))
assert result.trust.value == "unknown"
```

The scan inventories and hashes regular files, binds concrete toolchain and
Git dependency pins, extracts a **limited lexical** import graph and declaration
index, and identifies placeholders, source axioms, native/evaluator constructs,
kernel-check overrides, interpolated syntax and executable elaborator extensions.
Nested comments and string literals are masked. This is not a complete Lean
parser. Unicode/quoted declaration names, custom syntax, macros and imported
extensions need elaboration by the trusted verifier. Missing imports remain
explicitly unresolved. Module reachability is **not** theorem dependency
reachability: a reachable file may contain an unused placeholder.

Source inventories exclude `.git`, `.lake`, virtual environments, `node_modules`
and `__pycache__`. The SHA-256 is over a canonical path/size/content-hash manifest,
not a Git commit or tree hash. `--include-dependencies` fingerprints a reviewed
reference including `.lake`, but still excludes `.git`. Default limits are
20,000 files, 16 MiB per file, 512 MiB total and 2,000 findings. Exhausted file
budgets abort; exhausted finding budgets mark the report incomplete. Inspection
has no hard CPU deadline and is not a filesystem security sandbox: operate on
immutable, operator-owned snapshots, not directories concurrently writable by an
adversary. Symlink and nonregular entries are rejected.

```sh
mathkernel-formal-audit inspect /path/to/project --spec spec.json --output /outside/project/audit.json
mathkernel-formal-audit fingerprint /path/to/reviewed-reference --include-dependencies
mathkernel-formal-audit probe spec.json --output /outside/project/diagnostic.lean
```

Output paths must be new files. A diagnostic contains `#check`/`#print axioms`
commands, but is **not executed** and is never labelled a certificate. Do not
execute it on an untrusted project before establishing the verification boundary.

## MCP access boundary

Only `math_formal_project_audit` and `math_formal_project_probe` are exposed.
They are discoverable through `math_capability_query(domain="formal_project")`.
The audit tool requires an operator-owned read allowlist captured when the
server starts:

```sh
export MATHKERNEL_FORMAL_PROJECT_ROOTS=/srv/formal-candidates
mathkernel-mcp
```

Use the operating system's path separator for multiple roots (`:` on Linux,
`;` on Windows). The client cannot expand this list, select replay binaries,
install tools, or authorize execution through these tools. Existing result
pagination preserves oversized reports. Direct Python calls are trusted library
use and may inspect an explicitly supplied local path.

## Comparator replay: reference first, never pre-build the submission

Comparator's guarantee assumes an independently trusted challenge **and its
transitive imports**, a trusted Lake configuration, and a checking environment
uncompromised by earlier submission builds. Running `lake build` on untrusted
sources before establishing this boundary defeats that assumption. The project
under review is not permitted to choose its own independent reference, checker,
axiom whitelist or external-kernel command.

The runner therefore requires:

1. A separate, reviewed **reference workspace** containing the challenge, trusted
   `lakefile.toml` or `lakefile.lean`, locked dependencies and preinstalled trusted
   dependency caches. It must not contain the solution module or a top-level
   `.lake/build`. Review the complete challenge import closure. A fingerprint
   records your trust decision; it does not make arbitrary input trustworthy.
2. Explicit operator-provided SHA-256 pins for that entire workspace and nine
   executables: `lean`, `lake`, `comparator`, `landrun`, `lean4export`, `nanoda`,
   `systemd_run`, `systemctl`, and `env`. Tool executables must be outside both
   source trees. Lean/Lake must be direct binaries from one installed toolchain,
   not auto-installing Elan proxies. Shared libraries, the Lean installation,
   OS and hardware remain part of the operator's trusted computing base; binary
   hashes alone do not inventory every dynamically loaded dependency.
3. Linux, a **dedicated unprivileged account without sensitive files**, a working
   systemd user service manager and functional sandbox tools. The runner requires
   an AF_UNIX restriction, private network, no-new-privileges, service memory/time
   limits and cgroup cleanup. Unsupported sandbox settings block the run; there
   is no local/fake-Landrun fallback. Landrun's read scope is not a confidentiality
   boundary, hence the dedicated disposable account requirement.

Each run creates a fresh staging workspace. Reference-controlled files are
copied first. Only submission `.lean` source modules are admitted; submission
`lakefile.lean`, Lake TOML/JSON, binaries, caches and scripts are not admitted.
The challenge is retained from the reference. Differing collisions with trusted
modules are rejected. Consequently, projects requiring additional native assets
or unusual build-generation paths need an explicitly reviewed reference setup;
this runner does not silently execute their build hooks.

The fixed invocation is a pinned `systemd-run` service running pinned `env -i`,
then pinned `lake env <comparator> <trusted-config>`. There is **no submission
build or diagnostic execution before Comparator**. The reference's config must
request exactly the specified target declarations and axiom whitelist, with
`enable_nanoda: true`. Unknown keys, arbitrary external-kernel commands and
open definition holes are refused. Changing the staged source/configuration or
pinned executables during verification prevents acceptance. Output and wall-time
are bounded; failures trigger a service stop as well as the service watchdog.

### Constructing the operator request

Use typed models or obtain the full JSON Schema; no valid fingerprints are
invented by a template:

```python
import json
from mathkernel.formal_audit import ComparatorRequest
print(json.dumps(ComparatorRequest.model_json_schema(), indent=2))
```

Top-level fields are `submission_root`, `reference_root`,
`reference_tree_sha256`, `config` (relative path inside the reference), `tools`
(the nine `{path, sha256}` records), `spec`, and optional `timeout_seconds`,
`memory_bytes`, `max_output_bytes`. The inspected submission's source manifest
may additionally be pinned using `spec.expected_source_sha256`.

```sh
mathkernel-formal-audit replay operator-request.json --authorize-execution --output replay.json
```

Alternatively call `verify_with_comparator(request, authorize_execution=True)`
or `kernel.formal_project_verify(request_dict, authorize_execution=True)`.
Replay is not registered as an MCP tool, job kind or planner capability.
`accepted` is conditional formal evidence for the declared reference statement.
`not_accepted`, `blocked`, `timeout`, `output_limit` and `error` retain `UNKNOWN`.
None means that the mathematical theorem has been disproved.

## Numerical audit fixes

Exercising the Navier-Stokes scaling formulas uncovered a reproducible bug in
`certified_enclose`: the prior fallback mixed point `mpmath.mpf` coefficients
with nonzero-width interval arguments. `1/2 - 3*h` raised a `ValueError`.
Endpoints also passed through point-rounded conversion before enclosure.

`certified_enclose` now evaluates MathIR coefficients and endpoint expressions
entirely in a private `mpmath.iv` context. It rejects unresolved endpoint order,
unsupported expressions, singular/unbounded results and complex-domain failures;
it returns structured errors rather than an unchecked enclosure. Decimal input
or bound ancestry remains `NUMERIC`. This entry point consistently uses the
MathIR interval path rather than an unqualified SymPy-to-Arb callable. The
separate optional Arb utilities remain available but are not used by this audit.

Closed, unequal exact rational expressions can now produce an exact refutation
without Z3. This shortcut is not used for variable-bearing, conditional or
approximate goals. It lets the audit's deliberately corrupted rational control
cases be rejected rather than merely returning `UNKNOWN` when Z3 is absent.

## Primary references

- Comparator trust assumptions and execution contract: https://github.com/leanprover/comparator
- OpenAI artifact pinned for this case: https://github.com/openai/NavierStokesAndEuler/tree/8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538
- Independent statement source: https://github.com/google-deepmind/formal-conjectures/blob/8bf45ed70d48b2b2a501de9c00b26bfa38c573ee/FormalConjectures/Millenium/NavierStokes.lean

The upstream Comparator README used during design had blob
`e1d67f66231d4d18feae57797ee850ab5e00f2c0`. The implementation is deliberately
conservative but should still undergo live checker qualification and a separate
security review before being used on hostile repositories.
