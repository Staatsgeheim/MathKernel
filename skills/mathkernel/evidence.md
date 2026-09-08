# Evidence, typed objects, and trust ancestry

Use this guidance whenever a MathKernel workflow creates typed mathematical
objects or combines results from more than one source.

## Core invariant

MathKernel's non-negotiable invariant is:

```text
strength of claim <= strength of the required evidence ancestry
```

A later symbolic-looking or formally checked operation does not erase weaker
required inputs. Decimal/empirical/numeric ancestry remains visible unless a
separate rigorous enclosure or proof establishes the requested claim from a
stronger mathematical statement.

## Evidence is a support graph, not a flat list

`EvidenceBundle` records evidence items with a `role` and `support_path`.

- `required` — part of a derivation path that establishes the claim.
- `cross_check` — independent corroboration. Failure/unavailability is visible
  but does not invalidate an already complete required proof path.
- `diagnostic` — useful metadata/health checking, never claim support by itself.
- Evidence sharing one `support_path` is conjunctive: the weakest required item
  limits that path.
- Distinct complete support paths are alternatives: one valid path may establish
  the claim even if another path is unavailable.
- `justified_trust` is a global ancestry ceiling and MUST survive bundle merges.

Never determine trust by choosing the strongest evidence item or by taking the
weakest item across optional diagnostics.

## Typed mathematical objects are immutable scientific source nodes

`MathKernel.object_create(...)` creates a source object and construction
record. Calling `MathKernel.apply(...)` MUST NOT rewrite the source object's
construction evidence, assumptions, or scientific status.

An operation produces an operation result and may produce a derived typed
object. Provenance should be read as:

```text
source object(s) -> operation -> result / derived object
```

Querying `Normal(0,1).pdf(0)` does not turn the stored Normal distribution into
"a verified PDF calculation".

## All required operands participate in ancestry

The typed executor centrally resolves object-reference parameters (`*_id`,
`*_ids`, and declared typed references) and declared MathIR parameters. Trust
from every required operand limits the requested conclusion.

Examples:

- symbolic function + numeric contour -> at most numeric unless separately
  certified;
- exact contour + decimal point -> at most numeric;
- exact formula + empirical dataset -> the scientific conclusion remains
  empirical/model-conditional;
- exact PDE + numeric mesh must never become an exact continuous solution.

Do not implement domain-local trust fixes when generic dependency resolution
can express the relation.

## Assumptions and conditions

Symbolic objects may be constructed conditionally. Parameter requirements such
as `sigma > 0`, `b > a`, or `nu > 2` should become obligations when they are
not already proved.

A `MathContext` can discharge those conditions. The assumptions used must then
remain in result provenance. Do not reject useful symbolic objects merely
because a condition is not locally decidable, and do not silently forget a
condition after a later operation.

## Mathematical outcome vs trust

Keep semantic outcome separate from evidence strength. Important outcomes
include `verified_*`, `candidate`, `unknown`, `does_not_exist`, `undefined`,
`infeasible`, and `unsupported`. A failure to compute is not a proof of
nonexistence.

## Persistence boundary

Persisted typed values are untrusted input. MathKernel decodes SymPy `srepr`
through a restricted AST constructor whitelist and instantiates only explicit
allowlisted MathKernel Pydantic model classes. Never reintroduce `sympify`,
`eval`, dynamic attribute calls, or arbitrary class restoration in persistence.

## Review checklist for new domains

Before accepting a domain operation, verify that:

1. every required mathematical input appears in ancestry;
2. optional cross-checks cannot downgrade a complete required path;
3. alternative proof paths cannot bypass the global input-trust ceiling;
4. conditions/assumptions survive composition;
5. source typed objects are not mutated by queries;
6. `does_not_exist`/`infeasible` claims have proof-grade semantics appropriate
   to the domain;
7. serialization round-trips without changing evidence roles or source lineage.

## Compute candidates

Worker bytes enter `RemoteResultEnvelope`, never a legacy `MathResult` validator.
Only local checking constructs accepted evidence, bound to candidate, bundle,
execution and claim. A worker's trust/verified/image labels cannot affect it.
The cuboid pilot separates witness validity from complete-search coverage; the
convolution pilot stays numeric with fixed binary64 input ancestry. Verifier
timeouts are inconclusive. See [remote-compute.md](remote-compute.md).
