# Evidence and typed-object semantics over MCP

Use these rules whenever `math_object_create`, `math_apply`, proof tools, or
multi-object operations are involved.

## Read the requested claim, not an attractive evidence item

Every result may carry several evidence records. `claim_evidence` is the
relevant source for the requested conclusion. Evidence has support semantics:

- `required`: must succeed on its support path;
- `cross_check`: corroborates but does not invalidate an otherwise established
  result if unavailable;
- `diagnostic`: reports quality/health information only;
- separate `support_path` values may represent alternative derivations.

The top-level `trust` is conservative and subject to the full required input
ancestry. Never report a stronger claim just because a formal/symbolic item is
present somewhere in the response.

## Typed objects stay immutable

`math_object_create` creates a scientific source node. `math_apply` asks a
question about it; it does not change what that source object represents. Use
returned result/derived-object IDs for subsequent derived mathematics rather
than expecting the original object's evidence to mutate.

## Multi-object trust

When an operation references other MathKernel objects or MathIR parameters,
all required inputs are dependencies. Examples:

```text
symbolic ComplexFunction + numeric Contour -> numeric ceiling
exact Contour + decimal point             -> numeric ceiling
```

If a result appears stronger than a required input, inspect
`required_input_trust`, `required_object_inputs`, the derivation trace, and
`claim_evidence` before communicating it.

## Context and conditional objects

Pass `context_id` when symbolic parameter relations are known. A condition such
as `b > a` can discharge the corresponding `Uniform(a,b)` well-formedness
obligation. Otherwise the condition should remain explicit rather than being
silently assumed or causing an unnecessary hard rejection.

## Security/replay

Do not ask an agent to construct arbitrary persisted Pydantic/SymPy payloads.
The store intentionally accepts only restricted mathematical AST data and an
explicit allowlist of MathKernel model types. Replay is not an escape hatch to
arbitrary Python expression evaluation.

## Communication rule

Always preserve this invariant in prose:

```text
strength of claim <= strength of required evidence ancestry
```

Optional failed verifiers should be reported as unavailable cross-checks, not
as proof that a valid result is unknown. Conversely, exact symbolic operations
must not wash approximate/empirical ancestry into exactness.
