# External Lean project inspection over MCP

`math_formal_project_audit(root, spec=None)` performs bounded read-only source and
configuration inspection. It does not compile sources or run Lean, Lake, a shell,
Comparator or a network client. `root` must be inside an operator-set
`MATHKERNEL_FORMAL_PROJECT_ROOTS` directory. The allowlist is captured at server
startup and cannot be changed by tool arguments or YOLO settings.

`spec` accepts `targets` with `module`/`declaration`, `expected_toolchain`, optional
`expected_source_sha256`, `comparator_configs`, and `permitted_axioms`. A source
fingerprint is the inspector's canonical SHA-256 manifest, not a Git commit SHA.
Use `math_capability_query(domain="formal_project")` for discovery.

Results remain UNKNOWN even if every source check passes. A `PLACEHOLDER` in a
file is not necessarily a transitive dependency of the target theorem. Missing
imports, limitations and source-only findings remain explicit. Retrieve oversized
reports using the standard result-resource paging tools.

`math_formal_project_probe(spec)` returns `script` and `checked: false`. This is a
candidate diagnostic, **not** a proof or certificate. Do not execute it on an
untrusted submission before establishing a clean Comparator boundary.

There is deliberately **no** `math_formal_project_verify` MCP tool, arbitrary
binary selector, implicit installation, or execution-authorization tool. Actual
replay is a trusted-operator CLI/Python operation with a reviewed independent
reference. See the [complete guide](../mathkernel/formal-project-audit.md).
