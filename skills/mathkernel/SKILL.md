---
name: mathkernel
description: Use the MathKernel Python library (package `mathkernel`) for programmatic trustworthy mathematics — MathKernel facade, MathIR expressions, exact integer/GF(2^m) arithmetic, certified graph/combinatorics/finite-algebra objects, Koopman finite-dynamics, closure search, state-conditioned orbit access, symmetry discovery, cumulants, shared multimodal projections, visualization/sonification, and numba/CUDA-accelerated engines. Use when writing or modifying Python code that imports mathkernel, or when the MCP server is unavailable and direct library use is preferred.
---

# MathKernel (Python library)

`mathkernel` is the core library; `mathkernel_mcp` is a thin FastMCP layer
over it. Use the library directly for in-process work, notebooks, scripts,
and pipelines — the MCP server adds nothing but transport.

```python
from mathkernel import MathKernel

kernel = MathKernel()
r = kernel.parse("x^2 - 2 = 0")
sol = kernel.solve(r.data["expr_id"], "x")
assert sol.ok and sol.trust.value == "symbolic"
```

## Dev30 execution and retrieval contract

Expression assumptions and evidence ancestry survive substitution, simplification,
retrieval, and SQLite restart. Unknown context handles are errors, never permission
to change the mathematical domain. A legacy expression without sufficient stored
provenance remains uncertified; rederive it from its original inputs rather than
upgrading it by inspection of its current syntax.

For direct Python use, `kernel.expression("x^2 + 1")` returns an immutable
`ExpressionHandle` with convenience methods using the same kernel contract.
Use `mathkernel.python_api.complete_result(kernel, result)` to reconstruct and
integrity-check a paged result before inspecting its mathematical evidence.
The whole serialized response has a minimum configurable budget of 1,000 bytes;
a delivery receipt is not the original mathematical result. Its `unknown` trust
does not replace the trust of the preserved result.

## Studio authoring and inspection

When working with `.mkstudio.json` documents, Studio screenshots, or the optional
Studio host, read [studio.md](studio.md). These are authoring/presentation objects;
opening or connecting a graph does not establish execution or mathematical evidence.

## Optional compute planning and execution

For `mathkernel_compute` or `mathkernel-compute`, read [remote-compute.md](remote-compute.md).
The current pilot has separate native local, SSH and Slurm execution paths with
explicit grants and local candidate verification. SSH/Slurm export requires an
exact-scope host approval and an operator-pinned target profile. Do not route existing jobs to it automatically.

## Shared multimodal projection contract

Visualization and scientific audio now share `mathkernel_projection.MultimodalProjection`. Use it for domain objects rather than independently flattening data in each frontend. The live catalog covers fields, matrices/tensors, graphs/evidence trees, distributions, spectra/complex data, meshes/complexes, ODE/PDE solutions, optimization/inference, finite dynamics/finite fields, relation geometry, sets/partitions/piecewise objects, quantities, ensembles and explicit higher-dimensional projections. Projection parameters and information loss are provenance; they never create mathematical evidence.

## Architecture

- `mathkernel.kernel.MathKernel` — single facade over all engines; holds
  state (expressions, matrices, fields, finite systems, jobs).
- `mathkernel.contracts` — common expression/context resolution, conservative
  ancestry, and evidence-preserving persistence boundary.
- `mathkernel.output_policy` — complete-response budgets and integrity-checked
  result resources.
- `mathkernel.models` — `MathResult` (`ok`, `data`, `trust`,
  `semantic_status`, `engine`, `derivation`, `warnings`, `side_conditions`,
  `evidence_bundle`, `claim_evidence`), `TrustLevel`, `MathContext`.
- `mathkernel.settings.Settings` — env-driven limits and toggles
  (`MATHKERNEL_*`); `Settings().limits()` introspects them.
- Standalone modules usable without the facade: `gf2m`, `gf2m_fast`,
  `transforms` (FWHT), `cumulants`, `finite_fourier`, `koopman`,
  `relations`, `integral_transforms`, `complex_analysis`,
  `continuous_probability`, `graph_theory`, `combinatorics`,
  `finite_groups`, `finite_algebra`, `differential_geometry`,
  `computational_geometry`, `algebraic_topology`, `statistical_inference`,
  `nonparametric`, `survival_analysis`, `time_series`,
  `integers`, `collatz`, `cuboid`,
  `parallel`.

## Typed compositional domains

Use `MathKernel.object_create(...)`, retain the returned `object_id`, then call
`MathKernel.apply(object_id, operation, parameters)`. Query the capability
registry first when selecting an operation or evidence requirement.

- `TransformProblem`: Laplace/Fourier/Mellin/Z transforms with explicit
  convention, domain and ROC obligations.
- `ComplexDomain`, `ComplexFunction`, `Contour`: residues, contour accounting,
  argument principle, analytic continuation and conformal maps with explicit
  branch metadata.
- `Distribution`, `RandomVariable`, `JointDistribution`,
  `ConditionalDistribution`: univariate and multivariate probability,
  conditioning/Bayes, covariance/correlation and order statistics.
- `Graph`, `DirectedGraph`, `WeightedGraph`, `MultiGraph`: certified exact
  graph algorithms with explicit witnesses and budgeted NP-hard search
  semantics.
- `CombinatorialClass`, `GeneratingFunction`: exact counts, lazy generation,
  coefficient extraction and recurrence conversion.
- `FiniteGroup`, `PermutationGroup`, `FiniteAbelianGroup`,
  `GroupHomomorphism`, `FiniteRing`, `FiniteField`, `Module`: exact finite
  algebra with axiom/homomorphism/normal-form certificates.
- `Manifold`, `Chart`, `Metric`, `CoordinateMap`, `TensorField`,
  `DifferentialForm`: local differential geometry, tensor calculus and exterior
  calculus with explicit coordinates, variance, Jacobians and symbolic identity
  checks. Read [geometry.md](geometry.md) for this workflow.
- `Point`, `PointSet`, `Polygon`, `Polytope`, `Triangulation`: exact and
  ambiguity-aware computational geometry, also covered by
  [geometry.md](geometry.md). Exact verified triangulations can cross into the
  finite-topology workflow without trust promotion.
- `SimplicialComplex`, `CubicalComplex`, `ChainComplex`: exact finite
  complexes, boundary matrices and homology over Z/Q/GF(p), covered by
  [geometry.md](geometry.md).
- `StatisticalSample`: immutable typed observations with sample-scoped
  summaries, covariance, empirical distributions, and explicit separation of
  computation from empirical/model evidence. Read
  [probability-stats.md](probability-stats.md).
- `GeneralizedLinearModel`, `GLMFit`: canonical Gaussian/identity,
  binomial/logit and Poisson/log models with checked rank, convergence, score,
  covariance, deviance, diagnostics and conditional-mean prediction. Fits are
  conditional computations, never automatic model-validity or causal claims.
- `NonparametricTestResult`, `ResamplingResult`: derived-only rank-test,
  exact/asymptotic inference and seeded permutation/bootstrap records with
  source-linked verification replay.
- `SurvivalDataset`, `KaplanMeierEstimate`,
  `CoxProportionalHazardsModel`, `CoxPHFit`: right-censored data, exact
  product-limit curves with numerical Greenwood/log-log uncertainty, and
  checked conditional Cox fits with explicit assumptions and diagnostics.
- `TimeSeriesDataset`, `TimeSeriesAnalysis`, `TimeSeriesModel`,
  `TimeSeriesFit`, `TimeSeriesForecast`: ordered univariate ACF/PACF/ADF,
  conditional AR/MA/ARMA/ARIMA/GARCH fits, residual diagnostics, and analytic
  forecasts with replay and explicit stationarity/coverage nonclaims.

Each result embeds its typed execution DAG in `data.plan`; unresolved
verification obligations remain visible and cap the semantic status.

## Conventions

- **Trust**: `FORMAL` is a mechanically checked proof; `EXACT` is exact
  computation; `SYMBOLIC` is symbolic-engine evidence (not independent
  proof); `NUMERIC` is float evidence. Derived results carry the weakest
  trust of their inputs. Never upgrade trust silently.
- **Evidence**: inspect `claim_evidence` for the requested conclusion before
  communicating it. Computation, proof, certificate, numerical, model and
  empirical support remain separate. `trust` is only their conservative
  summary; `justified_trust` cannot override weaker ancestry, and unverified
  proof/certificate records support only `unknown`.
- **Mathematical outcomes**: use `semantic_status` to distinguish a computed
  failure from `does_not_exist`, `undefined`, `infeasible`, `unbounded`, `unsupported` and
  `unknown`.
- **Decimal literals are approximate**: a `RealNode` anywhere in a MathIR
  tree caps the expression's trust at `NUMERIC` — `parse("0.1 + x")` is
  `numeric`, `parse("1/2 + x")` is `symbolic`. Formal certificates (Lean)
  and exact SMT counterexamples are refused for approximate inputs, because
  backends encode decimal syntax as exact rationals — a different statement.
  Use exact rationals or interval certification for stronger evidence.
- **Big integers**: API boundaries use decimal/hex strings; internally
  Python ints.
- **Errors**: methods return `MathResult(ok=False, errors=[...])` rather
  than raising for expected failures (invalid input, limits); programming
  errors still raise.
- **Derivations**: every facade call records `DerivationStep`s;
  `kernel.derivation_trace(step_id)` reconstructs provenance.
- **Optimization tiers**: new domains must ship exact semantics and performance
  tiers together. Use Numba/process/GPU paths only inside a narrow exactness
  fragment, fall back to the Python reference automatically, re-verify or
  differential-test fast results, and record the selected backend in evidence
  metadata. A fast path never raises trust beyond its certificate.

## Sub-skills

Load the file matching the task domain:

- [library-api.md](library-api.md) — symbolic/calculus/matrix/integer
  facade methods, contexts, planner, codegen
- [engineering.md](engineering.md) — sampled signals, streaming filters, MIMO control, LQR/Kalman/LQG, finite-horizon policies, constrained MPC and immutable estimator states, LP/QP/MILP, SOCP/SDP and quadratic-constraint proof workflows
- [geometry.md](geometry.md) — manifolds, coordinate tensor/form calculus,
  robust computational geometry, the exact triangulation bridge, and finite
  complexes/homology
- [evidence.md](evidence.md) — evidence support paths, immutable typed objects, generic input ancestry, conditional assumptions, and persistence trust boundaries
- [logic-sets.md](logic-sets.md) — MathIR v2 sets, membership,
  quantifiers (Z3), binders
- [polynomial.md](polynomial.md) — Gröbner bases, division, resultants,
  factorization, ideal membership, batch
- [probability-stats.md](probability-stats.md) — exact rational probability,
  Markov chains, typed statistical samples/GLMs, rank/resampling inference,
  right-censored Kaplan–Meier/Cox workflows, and ordered time-series models
- [exact-discrete.md](exact-discrete.md) — certified graphs, combinatorics,
  generating functions, finite groups/rings/fields, modules and normal forms
- [tensors.md](tensors.md) — sparse exact tensors, Einstein contraction,
  sparse solve, GPU/njit tiers
- [numerics.md](numerics.md) — certified roots, quadrature, ODE/PDE,
  optimization (simplex, Nelder-Mead, ensembles)
- [units.md](units.md) — dimensional analysis, exact unit conversion
- [proving.md](proving.md) — general theorem proving: SMT portfolio, Lean
  certificates, batch, replay
- [persistence.md](persistence.md) — SQLite store, replay, Arb, fuzzing
- [finite-dynamics.md](finite-dynamics.md) — gf2m/gf2/transforms,
  cumulants, finite_fourier, koopman, relations module APIs
- [conditioned-dynamics.md](conditioned-dynamics.md) — state-conditioned
  orbit access, conditioned closures, GF(2) orbit solving, symmetry
  discovery/synthesis
- [viz.md](viz.md) — shared projection-driven visualization, evidence DAGs, high-dimensional projection discipline, SVG/PNG and portable HTML
- [sonification.md](sonification.md) — projection-driven scientific audio, explicit structured-object reductions, deterministic WAV/WebAudio and candidate-observation discipline
- [multimodal.md](multimodal.md) — shared `MultimodalProjection` lineage plus `mathkernel_multimodal`: assemble viz +
  sonify documents into one portable research artifact with auto-derived
  cross-modal sync links and unified inspectors
- [performance.md](performance.md) — numba/CUDA/parallel engines,
  settings, GPU troubleshooting

## Testing

`python -m pytest tests/ -q` — full suite (~2 min). GPU smoke test:
`python scripts/gpu_smoke.py` (verifies cuBLAS, NVRTC, sieve, koopman
numeric path).
