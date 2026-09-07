---
name: mathkernel-mcp
description: Use the MathKernel MCP server tools for trustworthy mathematics — symbolic algebra, continuous transforms, complex analysis, continuous probability, certified exact graphs/combinatorics/finite algebra, matrices, formal proof (Z3/Lean), certified numerics, integer/PRNG analysis, finite dynamics, shared multimodal projections, visualization, and scientific sonification. Use when the user asks to solve, simplify, prove, verify, evaluate, or sweep mathematical problems through the math_* MCP tools.
---

# MathKernel MCP

MathKernel MCP exposes a multi-engine mathematics kernel as `math_*` tools.
Mathematical operation results are `MathResult` objects with `ok`, `data`,
`trust`, `semantic_status`, `engine`, a `derivation` trail, `evidence_bundle`,
and `claim_evidence`. Discovery endpoints also return plain dictionaries;
oversized responses return resource receipts rather than truncated evidence.
The trust level is a conservative summary: `formal` is a
mechanically checked proof certificate, `exact` is exact computation or an
independently established fact, `symbolic` is symbolic-engine output (strong
evidence, not independent proof), `interval_certified` is a rigorous
enclosure, and `numeric` is numerical evidence.

## Dev30 discovery and large results

`math_capabilities` now returns a compact summary by default; request `detail="full"`
only when needed. `math_capability_query` defaults to 25 records. Follow `next_offset`
and use a `limit` from 1 to 100. Optional parameter JSON Schemas describe declared
primitive hints, not complete requiredness or mathematical invariants; unsupported
hints stay explicitly unconstrained rather than being guessed.

A response exceeding the complete serialized payload budget is preserved as a
SHA-256-addressed resource. Call `math_result_resource_get`, concatenate `content`
in byte-offset order until `next_offset` is null, verify `sha256`, then decode the
JSON. A receipt's `unknown` trust is a delivery status, not a replacement for the
original result's trust. The minimum configurable budget is 1,000 bytes; outer MCP
framing is excluded. Stored contexts and derived expression ancestry survive restart.
Unknown contexts or incompatible context reuse must be resolved, not silently ignored.

## Studio authoring and inspection

When working with `.mkstudio.json` documents, Studio screenshots, or the optional
Studio host, read [studio.md](studio.md). These are authoring/presentation objects;
opening or connecting a graph does not establish execution or mathematical evidence.

## Shared multimodal projection contract

Use `math_projection_catalog` and `math_projection_create` when a mathematical object/result should feed visualization or audio. `math_visualize_projection` and `math_sonify_projection` consume the same evidence-carrying projection. High-dimensional reductions must explicitly name their method/dimensions and declare projection loss; structured audio reductions are returned as transformation provenance rather than hidden flattening.

## Golden workflow

1. **Discover** — call `math_capabilities` once per session. Use
   `math_capability_query` to filter by domain, input/output type, operation,
   trust level, verification method, or engine before choosing an operation.
2. **Parse** — `math_parse` (ASCII) or `math_parse_latex` returns an
   `expr_id`; chain it into all expression tools. Integers are decimal
   strings (arbitrary precision). Implicit multiplication (`1/2x`) is
   rejected with candidates — rewrite explicitly.
3. **Context** — `math_context_create` attaches domains/assumptions
   (`{"x": "positive", "n": "integer"}`); pass the `context_id` to downstream
   calls so SymPy/Z3 respect them.
4. **Compose typed mathematics** — create `TransformProblem`,
   `ComplexDomain`/`ComplexFunction`/`Contour`,
   `Distribution`/`RandomVariable`/`JointDistribution`/
   `ConditionalDistribution`, exact graphs, combinatorial classes,
   generating functions, finite groups/rings/fields, modules, or
   `Manifold`/`Chart`/`Metric`/`CoordinateMap`/`TensorField`/
   `DifferentialForm` geometry objects with
   `math_object_create`; retain its `object_id` and call `math_apply`.
   Transform conventions, probability supports, complex domains, branch
   metadata, graph direction/weights, and algebraic presentations must be
   explicit. For geometry also preserve chart order/domain, variance,
   orientation, metric signature and map direction.
   Computational geometry also uses `Point`, `PointSet`, `Polygon`, `Polytope`
   and `Triangulation`; exact topology requires exact coordinates, while an
   `ambiguous` filtered predicate must remain unresolved.
   Algebraic topology uses exact `SimplicialComplex`, `CubicalComplex`, and
   integral `ChainComplex` sources; homology claims require verified
   `boundary_squared_zero` evidence. An exact verified `Triangulation` may use
   `to_simplicial_complex`; numeric or refuted geometry cannot cross that
   trust boundary.
   Phase F statistics uses `StatisticalSample`; exact results describe only
   the stored observations, while empirical/model evidence and population
   non-inference remain explicit. `GeneralizedLinearModel` adds canonical
   Gaussian/logistic/Poisson fits and derived `GLMFit` verification,
   diagnostics and prediction; these do not establish model validity or
   causality. The same sample exposes six named non-parametric tests plus
   exact/seeded permutation and bootstrap operations; replay derived results
   with their `verify` operation. Survival work uses `SurvivalDataset` for
   right censoring/delayed entry/strata and derived `KaplanMeierEstimate`
   curves; unstratified `CoxProportionalHazardsModel` fits require explicit tie
   handling and retain proportional-hazards/censoring nonclaims.
   Ordered time-series work uses `TimeSeriesDataset` for exact ACF/PACF and
   numerical ADF diagnostics, plus `TimeSeriesModel` for explicit
   AR/MA/ARMA/ARIMA/GARCH fits and replayable diagnostics/forecasts. Preserve
   row order and never promote a diagnostic into stationarity/model validity.
5. **Verify** — prefer proof-grade tools (`math_prove_equivalence`,
   `math_reason`, `math_interval_evaluate`) over numeric evaluation when the
   claim matters. Inspect the requested claim in `claim_evidence`, not merely
   the strongest evidence item or symbolic-looking output. Use
   `math_derivation_trace` for provenance.

## Sub-skills

Load the file matching the task domain:

- [symbolic.md](symbolic.md) — parse/substitute/simplify/solve, calculus,
  sums/products, matrices, contexts
- [engineering.md](engineering.md) — sampled signals, streaming filters, MIMO control, LQR/Kalman/LQG, finite-horizon policies, constrained MPC and immutable estimator states, LP/QP/MILP, SOCP/SDP and quadratic-constraint proof workflows
- [geometry.md](geometry.md) — differential/computational geometry, exact
  triangulation conversion, and finite topology through generic typed tools
- [evidence.md](evidence.md) — evidence support paths, typed-object immutability, multi-object ancestry, context obligations, and trust-safe communication
- [logic.md](logic.md) — sets, membership, quantifier validity (Z3),
  Gröbner bases, polynomial division/resultant/factor, ideal membership
- [probability-stats.md](probability-stats.md) — exact rational probability,
  Markov chains, typed statistical samples/GLMs, rank/resampling inference,
  right-censored Kaplan–Meier/Cox workflows, and ordered time-series models
- [exact-discrete.md](exact-discrete.md) — certified graph algorithms,
  combinatorics/generating functions, finite groups, GF(p^m), modules
- [tensors.md](tensors.md) — sparse tensors, Einstein contraction, GPU tiers
- [numerics.md](numerics.md) — certified roots, quadrature, ODE/PDE,
  optimization
- [units-persistence.md](units-persistence.md) — units, store/replay,
  certified enclosure, fuzzing
- [proving.md](proving.md) — theorem proving: SMT portfolio, Lean
  certificates, replay
- [reasoning.md](reasoning.md) — planner (`math_reason`/plans), equivalence
  proofs, counterexamples, verified codegen, derivation audit
- [sweeps.md](sweeps.md) — Collatz sieve, cuboid sweep, integer batch,
  async job API
- [dynamics.md](dynamics.md) — finite systems, Koopman/visibility, finite
  Fourier, closure search, cumulants, GF(2^m)/GF(2)/FWHT for PRNG analysis
- [conditioned-dynamics.md](conditioned-dynamics.md) — state-conditioned
  orbit access, conditioned closures, GF(2) predictive closures, symmetry
  discovery/synthesis
- [viz.md](viz.md) — `math_projection_*` + projection-driven visualization and portable HTML artifacts
- [sonification.md](sonification.md) — `math_sonify_projection` plus legacy scalar sonification, explicit acoustic extraction provenance and WAV export
- [multimodal.md](multimodal.md) — shared projection lineage with `math_research_artifact_create` /
  `math_export_research_artifact`: one portable HTML with synchronized
  visual+audio browsing and unified inspectors

## Rules of thumb

- Never present `trust: numeric` output as proof; upgrade with
  `math_interval_evaluate` or an exact tool.
- A producer's `justified_trust` cannot override weaker evidence ancestry.
  Unverified proof/certificate entries support only `unknown`. Preserve
  `does_not_exist`, `undefined`, `infeasible`, `unbounded`, and `unsupported` as distinct
  mathematical outcomes.
- Decimal literals are approximate observations: any expression containing
  one is capped at `trust: numeric` from `math_parse` onward, and
  `math_prove_equivalence`/`math_counterexample` will not issue formal
  certificates or exact SMT counterexamples for approximate inputs (the
  backends would encode the decimals as exact rationals — a different
  statement). Rewrite as exact rationals (`1/10` instead of `0.1`) when
  proof-grade evidence is needed.
- Large sweeps: use `math_job_submit` + `math_job_status`/`math_job_result`
  instead of blocking calls.
- Koopman/finite-dynamics tools default to `exact=true` (proof-grade);
  `exact=false` is the GPU-accelerated numeric path with downgraded trust.
- If a tool errors with limits (max states, max order, enumeration budget),
  lower the parameters — do not retry identical calls. `math_yolo_settings`
  can raise live `MATHKERNEL_*` caps, but only when the server was started
  with `MATHKERNEL_YOLO_MODE=true` (default off). Values are type-checked.
  This is a process safety gate, not mathematical evidence.
