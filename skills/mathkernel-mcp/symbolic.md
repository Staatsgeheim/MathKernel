# Sub-skill: Symbolic algebra, calculus, and matrices

Part of the `mathkernel-mcp` skill. Load when the task is expression
manipulation, equation solving, calculus, or linear algebra via `math_*`
tools.

For continuous transforms and complex analysis, use the typed compositional
API rather than the flat expression tools:

- Create a `TransformProblem` with an explicit Laplace/Fourier/Mellin/Z
  convention, source/target variables and domain, then `math_apply` with
  `apply`, `solve` or `verify`. Inspect ROC and each verification obligation.
- Create `ComplexDomain`, `ComplexFunction` and `Contour` objects for
  residues, Laurent series, contour integration, argument-principle
  accounting, analytic continuation and conformal maps. Branch-sensitive
  expressions require explicit branch metadata.

Same-engine symbolic identities remain `symbolic`; unknown ROC, branch or
contour obligations must not be described as verified.

Inverse Laplace results retain an inferred convergence half-plane when
isolated singularities make it decidable. Fourier results retain their
integrability condition; bilateral Z results remain candidates below
`symbolic` when no ROC can be established. Unsupported kernels and unresolved
ROCs are explicit outcomes.

Complex continuation is currently conservative overlap continuation.
Argument-principle and contour claims require explicit contours and complete
singularity accounting; branch-sensitive or unsupported non-meromorphic cases
must remain `unsupported` or `unknown`. Contour vertex count, Laurent order,
expression size, solver time, and plan steps are bounded by kernel settings.

## Expression lifecycle

1. `math_parse("x^2 + 2*x + 1 = 0")` → `expr_id` (or `math_parse_latex` for
   LaTeX source; needs the `latex` extra).
2. `math_get(expr_id)` inspects the MathIR; `math_substitute(expr_id, {"x":
   "a + 1"})` returns a new expr_id (original preserved).
3. Operate: every tool below takes `expr_id` and optional `context_id`.

## Solving and simplification

- `math_solve(expr_id, variable)` — closed-form solutions with domain
  side conditions.
- `math_solve_system(expr_ids, variables)` — simultaneous equations.
- `math_simplify(expr_id, mode)` — mode: `simplify` | `expand` | `factor`.
- `math_infer_structure(expr_id)` — required algebraic capabilities and
  weakest structure (group/ring/field) before generating code.

## Calculus

- `math_differentiate(expr_id, variable, order)`
- `math_integrate(expr_id, variable, lower, upper)` — omit bounds for
  antiderivative.
- `math_limit(expr_id, variable, point, direction)` — direction `+-`, `+`,
  `-`; point may be `oo`.
- `math_series(expr_id, variable, point, order)` — Taylor/Laurent expansion.
- `math_summation` / `math_product(expr_id, variable, lower, upper)` —
  symbolic sums/products, bounds may be symbolic.

## Contexts (assumptions)

`math_context_create(domains={"x": "positive", "n": "integer"},
assumptions=["x > y"])` → `context_id`. Pass it to solve/simplify/calculus
tools so results respect the assumptions. `math_context_infer` extracts
domains from expressions; `math_context_check` validates consistency.

## Matrices

`math_matrix_create(rows)` with exact rational string cells (`"2/3"`) →
`matrix_id`, then: `math_matrix_det`, `math_matrix_inverse`,
`math_matrix_transpose`, `math_matrix_multiply`, `math_matrix_rank`,
`math_matrix_rref`, `math_matrix_eigenvalues`, `math_matrix_solve(A, b)`.
Exact arithmetic throughout; dimension limit in `math_capabilities`.

## Numeric evaluation

- `math_numeric_evaluate(expr_id, values, dps)` — arbitrary precision,
  trust `numeric`.
- `math_interval_evaluate(expr_id, bounds, dps)` — certified interval
  enclosure, trust `interval_certified`. Prefer this when a numeric claim
  must be rigorous.

## Constants

`E` is written `exp(1)`, pi as `pi()`. Both survive the exact/symbolic
round-trip.
