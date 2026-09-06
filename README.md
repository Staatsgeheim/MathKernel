# MathKernel

**An evidence-aware multi-engine mathematics kernel — usable both as a Python library (`mathkernel`) and as an MCP server (`mathkernel-mcp`) — so applications and LLMs can do advanced mathematics while preserving assumptions, provenance, and claim-specific evidence.**

> The LLM interprets intent; the MathKernel establishes mathematical evidence.

Mathematical results carry an explicit **trust level**, an **engine** tag, and a **derivation trail**. Exact computation, checked certificates, symbolic results, certified enclosures, empirical evidence, and formal proofs are distinct claims. Exact arithmetic alone is not a formal proof; approximate-input ancestry must not silently disappear.

[![version](https://img.shields.io/badge/version-1.3.0-blue)]()
[![python](https://img.shields.io/badge/python-%3E%3D3.11-blue)]()
[![engines](https://img.shields.io/badge/engines-sympy%20%C2%B7%20z3%20%C2%B7%20lean%20%C2%B7%20numba%20%C2%B7%20cuda-orange)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Table of contents

- [Why](#why)
- [Architecture](#architecture)
- [Feature matrix](#feature-matrix)
- [Installation](#installation)
- [Quickstart — MCP server](#quickstart--mcp-server)
- [Quickstart — Python library](#quickstart--python-library)
- [Trust model](#trust-model)
- [Continuous symbolic mathematics](#continuous-symbolic-mathematics)
- [Finite dynamics & PRNG analysis](#finite-dynamics--prng-analysis)
- [Engineering mathematics](#engineering-mathematics)
- [Geometry and topology](#geometry-and-topology)
- [Statistics and stochastic modeling](#statistics-and-stochastic-modeling)
- [PDEs and adaptive finite elements](#pdes-and-adaptive-finite-elements)
- [Relation and information-geometry inference](#relation-and-information-geometry-inference)
- [Performance: numba · CUDA · parallelism](#performance-numba--cuda--parallelism)
- [Visualization & portable artifacts](#visualization--portable-artifacts)
- [Shared multimodal projections](#shared-multimodal-projections)
- [Scientific sonification](#scientific-sonification-mathkernel-sonify)
- [Unified multimodal artifacts](#unified-multimodal-artifacts-mathkernel-multimodal)
- [MCP tool surface](#mcp-tool-surface)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Skill packages](#skill-packages)
- [Testing](#testing)
- [Safety boundaries](#safety-boundaries)
- [License](#license)

## Why

LLMs are good at mathematical *intent* and bad at mathematical *arithmetic*. MathKernel inverts the division of labor: the model parses, plans, and interprets; the kernel computes and records claim-specific evidence. Some claims use independent certificates or cross-checks; others are exact computations in one engine. Engine agreement alone is not a proof, and a single trust label does not replace the evidence bundle.

![alt text](WHY.png "Why MathKernel")

## Architecture

MathKernel is a typed orchestration layer rather than a single solver. The public
facade owns parsing, contexts, object identity, persistence, evidence composition,
resource policy and derivation tracking; domain adapters own the actual mathematics.
Presentation layers sit downstream and cannot silently change the claim being made.

```text
Python / MCP
    |
    v
MathKernel facade
    |-- parser + contexts + typed objects
    |-- execution/evidence contract
    |-- persistence + derivation graph
    |
    +--> symbolic / exact / certified / formal / numerical engines
    |
    +--> MathResult and derived mathematical objects
             |
             +--> MultimodalProjection
                     |--> mathkernel-viz
                     |--> mathkernel-sonify
                     +--> unified portable artifacts
```

This separation is deliberate: a renderer may present evidence, but it does not create
stronger mathematical evidence merely by producing a polished plot or audio artifact.

## Feature matrix

| Domain | Compute surface | Engines | Verification / evidence ceiling |
|---|---|---|---|
| Symbolic algebra | parse, substitute, simplify/expand/factor, solve, systems | SymPy | SYMBOLIC; input ancestry may lower it |
| Calculus | differentiation, integration, limits, series, sums, products | SymPy | SYMBOLIC + conditions |
| Integral transforms | Laplace/Fourier/Mellin/bilateral Z, inverses, ROC and property obligations | typed transform adapter + SymPy | SYMBOLIC; NUMERIC for approximate ancestry |
| Complex analysis | branches/domains, zeros/singularities, residues, Laurent series, contours, argument principle, continuation, conformal maps | typed complex adapter + SymPy | SYMBOLIC defining identities; EXACT winding certificates only for exact geometry, ancestry-capped otherwise |
| Continuous probability | typed univariate/joint/conditional distributions, transformations, marginals, Bayes, covariance, divergence, order statistics | typed probability adapter + SymPy | SYMBOLIC normalization/identity evidence; mathematical nonexistence retained |
| Exact graphs | typed simple/directed/weighted/multi graphs, traversal, components, shortest paths, MST, max-flow/min-cut, bipartite matching, Euler trails, coloring, topological sort, cycles, centrality, isomorphism | deterministic exact graph algorithms over `Fraction` + njit CSR traversal kernels | EXACT witness certificates; NP-hard optimality is OPTIMUM/CANDIDATE/IMPOSSIBLE/UNKNOWN, never heuristic nonexistence |
| Exact combinatorics | combinatorial classes, exact counts, lazy generation, ordinary/exponential generating functions, recurrences | exact integer/`Fraction` enumeration + SymPy + checked njit recurrence kernels | EXACT counts and recurrence/coefficient checks |
| Finite algebra | finite groups, permutation groups, abelian groups, homomorphisms, Z/nZ, GF(p^m), modules, Smith/Hermite normal forms | exact algebra + SymPy combinatorics + njit Cayley/GF(p)[x] kernels | EXACT axiom, homomorphism, irreducibility, and normal-form certificates |
| Linear algebra | determinant, inverse, multiply, rank, RREF, eigenvalues, exact solves | SymPy | EXACT for exact arithmetic; otherwise ancestry-capped |
| Reasoning | obligation-DAG planning, equivalence, counterexamples | SymPy + Z3 + Lean | SYMBOLIC / EXACT / FORMAL by verifier |
| Certified numerics | arbitrary-precision evaluation and interval enclosures | mpmath + mpmath.iv | CERTIFIED NUMERIC or NUMERIC |
| Integers | arbitrary precision, gcd/lcm, primality, factorization, CRT, modular arithmetic | exact + numba batch | EXACT |
| Code generation | TypeScript/Python/Rust emission, typecheck, symbolic round-trip, sandbox | compilers + SymPy | SYMBOLIC verification; never stronger than source |
| Binary fields | GF(2^m) arithmetic/construction and Rabin irreducibility | njit n-limb kernels | EXACT certificates |
| GF(2) linear algebra | rank, nullspace, powers, Berlekamp–Massey, carry-free columns | bit-packed integers | EXACT |
| Discrete transforms | exact FWHT with bigint fallback | numba | EXACT |
| Finite dynamics | Koopman/observation transfer, visibility, lagged tensors, diagnostics | exact + NumPy/CuPy | EXACT or NUMERIC, selected explicitly |
| Branching Markov tensors | arbitrary finite rooted Markov trees, exact leaf laws/cumulants, true-edge flattening certificates, stochastic leaf observations, channel-rank transfer, exact recovery and collective sensor fusion | exact `Fraction` sum-product/enumeration + NumPy SVD diagnostics | EXACT algebraic identities/ranks/recovery; NUMERIC singular-value and conditioning evidence kept separate |
| Connected-relation detectability | pure connected-interaction laws, stochastic mode visibility, conditional-expectation spectra, exact chi-square/Fisher retention, invisibility certificates, finite sample bounds and sensor fusion | exact `Fraction` laws + weighted NumPy SVD + exact binomial likelihood-ratio validation | EXACT transfer/information identities and lower/upper bounds; EMPIRICAL Monte Carlo checks remain separately labelled |
| Relation-subspace visibility | multi-relation Fisher Gram transfer, generalized visibility spectra, blind-combination collision certificates, cost-constrained sensor design, empirical partitions and long-run-covariance correction | finite probability algebra + weighted NumPy generalized eigensystems + exact finite sensor enumeration | EXACT local transfer/data-processing/collision identities; NUMERIC spectra and EMPIRICAL dependence/SkewDB checks retain explicit scope |
| Intrinsic observation information geometry | finite-simplex Fisher tangents, coordinate-invariant retained-information spectra, exact local chi-square transfer, worst-direction testing lower bounds, finite Bhattacharyya upper bounds, iid/block/cluster spectrum bootstrap, local-resolution SkewDB adapter | finite probability algebra + weighted generalized eigensystems + SciPy exact-binomial validation + seeded resampling | EXACT finite tangent/data-processing/divergence identities and finite simple-testing bounds; NUMERIC eigensystems and EMPIRICAL uncertainty checks remain separately labelled |
| Composite relation inference | one direction-agnostic relation-subspace test, dimension-aware finite bound, nuisance-efficient Fisher geometry, eigenspace regions, studentized/block bootstrap, HAC and misspecification diagnostics | finite Fisher algebra + NumPy eigensystems + optional SciPy chi-square calibration + seeded resampling | EXACT nuisance/data-processing identities and conservative bounded-score guarantee; ASYMPTOTIC composite calibration and EMPIRICAL bootstrap/dependence checks are labelled |
| Finite Fourier | cyclotomic DFT/transfer/coefficient/orbit calculations | exact + NumPy FFT | EXACT or NUMERIC cross-check |
| Closure search | cyclic/XOR irreducible closure relations | njit meet-in-the-middle | EXACT witness/exhaustive evidence |
| Conditioned dynamics | orbit access, cocycles, closures and symmetry synthesis | exact enumeration + canonical rewrite | EXACT witnesses |
| Cumulants | moments/cumulants and connected sample statistics | exact + NumPy | EXACT algebra or EMPIRICAL samples |
| Sets & logic | set algebra, membership, quantified truth and elimination | SymPy sets + Z3 | EXACT SMT witnesses where established |
| Polynomial algebra | Gröbner bases, division, resultants, factorization, ideal membership | exact SymPy polynomial algorithms | EXACT algebraic certificates |
| Discrete probability | rational RVs, Bayes, Markov quantities, seeded sampling | Fraction + NumPy | EXACT distributions; EMPIRICAL sampling |
| Statistics and stochastic systems | typed samples, GLMs, rank/resampling inference, survival/time-series analysis; Poisson/Wiener/GP/CTMC laws; typed Itô SDEs, Euler–Maruyama/scalar Milstein paths and coupled convergence studies | typed statistical/survival/time-series/stochastic/SDE adapters + SymPy + NumPy/SciPy/mpmath | EXACT identities remain separate from labelled NUMERIC fits/conditioning/exponentials and seeded EMPIRICAL resampling/simulation; no implied process/model validity, convergence theorem, population inference or causality |
| Tensors | sparse tensors, contraction and sparse solves | exact + njit + CuPy | EXACT or NUMERIC by arithmetic path |
| ODEs / PDE | symbolic ODE classification/dsolve; numerical IVP/named PDE solvers; typed PDE systems, weak forms, oriented simplex meshes, P1 spaces, sparse assembly, checked algebraic solves, residual–jump indicators, marking, conforming refinement, nodal transfer and observed estimator rates | typed PDE/FEM/adaptivity adapters + SymPy + SciPy sparse + mpmath + njit + CUDA/CuPy | estimators and empirical rates retain ancestry and never become rigorous continuum bounds or convergence theorems |
| Optimization | critical points, KKT, exact LP, numerical nonlinear/multistart | Fraction + njit + process pool | EXACT LP certificates or NUMERIC candidates |
| Units | SI dimensions, rational conversions and semantic-unit propagation | exact Fraction | EXACT |
| Assurance | interval obligations, Lean replay, Arb balls, persistence and fuzzing | mpmath.iv + flint + Lean | CERTIFIED NUMERIC / FORMAL / differential evidence |
| Theorem proving | SMT portfolio and Lean certificates | Z3 + Lean | EXACT SMT witness or FORMAL kernel-checked proof |
| Exhaustive sweeps | Collatz and cuboid searches | numba + CUDA + process pools | EXACT only when coverage is exhaustive |
| Async jobs | submit/status/result/list with evidence-preserving retrieval | job pool | Preserves underlying evidence |
| Visualization | renderer-neutral interactive/static mathematical artifacts | Python SVG + vendored three.js | No new evidence; preserves source trust |
| Sonification | declarative scientific audio mappings and deterministic WAV | Python PCM + WebAudio | Candidate observation only |
| Multimodal artifacts | synchronized visual/audio artifact assembly | shared artifact schema | Weakest included claim/evidence |
| Differential geometry | manifolds, oriented charts, metrics, coordinate maps, tensor fields, forms, curvature, covariant/Lie/exterior derivatives, wedge/interior/pullback/Hodge operations | typed geometry adapter + SymPy | SYMBOLIC identities with explicit domains, Jacobians, signature and ancestry; numeric input stays NUMERIC |
| Computational geometry | concrete points/sets, polygons, half-space polytopes, triangulations, hull, containment, intersection, nearest neighbor, Delaunay and Voronoi | exact SymPy determinants + adaptive float filters | EXACT topology for exact coordinates; NUMERIC only when filters decide; otherwise explicit AMBIGUOUS outcome |
| Algebraic topology | finite simplicial/cubical/integral chain complexes, exact triangulation conversion, oriented boundaries, Euler characteristic, homology over Z/Q/GF(p) | exact integer matrices + certified Smith normal form + rational/modular elimination | EXACT face-closure, boundary², rank-nullity, quotient, torsion and Euler–Poincaré certificates |

### Typed functionality surface

The generic MCP tools `math_object_create`, `math_object_get`, and `math_apply`
expose the following compositional operations. This is the full typed-operation
inventory; `math_capability_query` is the live source of parameter schemas,
output types, limits, engines and verification methods.

| Domain | Object | Operations |
|---|---|---|
| Integral transforms | `TransformProblem` | `apply`, `solve`, `verify` |
| Complex analysis | `ComplexFunction` | `analytic_continuation`, `analyticity`, `argument_principle`, `classify_singularity`, `conformal_at`, `conformal_map`, `contour_integral`, `derivative`, `laurent_series`, `residue`, `singularities`, `zeros` |
| Complex analysis | `Contour` | `winding_number` |
| Continuous probability | `Distribution` | `cdf`, `characteristic_function`, `convolve`, `cross_entropy`, `entropy`, `expectation`, `kl_divergence`, `mean`, `mgf`, `mixture`, `moment`, `order_statistic`, `pdf`, `quantile`, `query`, `survival`, `truncate`, `variance`, `verify` |
| Continuous probability | `JointDistribution` | `bayes`, `condition`, `correlation`, `covariance`, `marginal`, `order_statistic`, `verify` |
| Continuous probability | `ConditionalDistribution`, `RandomVariable` | conditional `cdf`/`mean`/`pdf`/`variance`/`verify`; random-variable `transform` |
| Exact graphs | `Graph`, `MultiGraph` | `bfs`, `centrality`, `coloring`, `connected_components`, `cycle_detection`, `dfs`, `euler_path`, `matching`, `shortest_path`, `verify`; `Graph` also has `isomorphic_to` |
| Exact graphs | `DirectedGraph` | `bfs`, `centrality`, `cycle_detection`, `dfs`, `shortest_path`, `strongly_connected_components`, `topological_sort`, `verify` |
| Exact graphs | `WeightedGraph` | `bfs`, `centrality`, `coloring`, `connected_components`, `cycle_detection`, `dfs`, `euler_path`, `matching`, `maximum_flow`, `minimum_cut`, `minimum_spanning_tree`, `shortest_path`, `strongly_connected_components`, `topological_sort`, `verify` |
| Combinatorics | `CombinatorialClass`, `GeneratingFunction` | class `count`/`generate`/`verify`; generating-function `coefficient`/`recurrence`/`verify` |
| Finite groups | `FiniteGroup` | `center`, `centralizer`, `closure`, `commutator_subgroup`, `conjugacy_classes`, `cosets`, `generated_subgroup`, `normality`, `orbits`, `order`, `quotient`, `stabilizers`, `subgroups`, `verify` |
| Finite groups | `PermutationGroup` | `contains`, `orbits`, `order`, `stabilizer_chain`, `stabilizers`, `verify` |
| Finite groups | `FiniteAbelianGroup`, `GroupHomomorphism` | abelian `order`/`verify`; homomorphism `image`/`kernel`/`verify` |
| Finite algebra | `FiniteRing`, `FiniteField` | `add`, `inverse`, `multiply`, `verify` |
| Finite algebra | `Module` | `abelian_group`, `hermite_normal_form`, `smith_normal_form`, `verify` |
| Signals | `ContinuousSignal`, `DiscreteSignal` | continuous `sample`; discrete `autocorrelation`, `convolution`, `correlation`, `cross_spectrum`, `dft`, `resample`, `stft`, `window` |
| Signals | `Spectrum`, `Filter`, `FilterDesign`, `FilterState` | spectrum `idft`; filter `apply_signal`/`initial_state`/`to_transfer_function`; design `design`; state `process` |
| Control | `TransferFunction` | `bode`, `feedback`, `frequency_response`, `impulse_response`, `nyquist`, `poles`, `root_locus`, `series`, `stability`, `step_response`, `to_filter`, `to_state_space`, `to_zero_pole_gain`, `zeros` |
| Control | `StateSpaceSystem` | `bode`, `coefficient_units`, `controllability`, `discretize`, `finite_lqr`, `frequency_response`, `kalman`, `kalman_state`, `lqg`, `lqr`, `mpc`, `nyquist`, `observability`, `observer`, `place_poles`, `poles`, `stability`, `state_feedback`, `to_discrete_control`, `to_transfer_function`, `zeros` |
| Control | `DiscreteControlSystem` | `bode`, `controllability`, `frequency_response`, `nyquist`, `observability`, `poles`, `stability`, `to_state_space`, `to_transfer_function`, `zeros` |
| Control | `ZeroPoleGain`, `TransferMatrix` | ZPK `bode`/`nyquist`/`poles`/`to_transfer_function`/`zeros`; matrix `entry` |
| Sequential control | `FiniteHorizonLQR`, `KalmanState`, `MPCPlan` | LQR `control`/`rollout`/`verify`; Kalman `predict`/`update`; MPC `first_control`/`verify` |
| Optimization | `OptimizationProblem` | `certify_milp`, `solve`, `to_conic`, `verify_certificate`, `verify_milp_certificate` |
| Optimization | `ConicProblem`, `QuadraticallyConstrainedProblem` | `solve`, `verify_certificate` |
| Differential geometry | `Metric` | `inverse_metric`, `christoffel`, `riemann`, `ricci`, `scalar_curvature`, `einstein`, `geodesic_equations` |
| Differential geometry | `CoordinateMap` | `jacobian`, `verify` |
| Differential geometry | `TensorField` | `covariant_derivative`, `lie_derivative` |
| Differential geometry | `DifferentialForm` | `wedge`, `exterior_derivative`, `interior_product`, `pullback`, `hodge_star` |
| Computational geometry | `Point` | `distance_to` |
| Computational geometry | `PointSet` | `orientation`, `incircle`, `segment_intersection`, `convex_hull`, `nearest_neighbor`, `delaunay`, `voronoi` |
| Computational geometry | `Polygon` | `verify`, `contains`, `intersection`, `triangulate` |
| Computational geometry | `Polytope` | `verify`, `contains` |
| Computational geometry | `Triangulation` | `verify`, `to_simplicial_complex` |
| Algebraic topology | `SimplicialComplex`, `CubicalComplex` | `verify`, `chain_complex`, `boundary_matrix`, `homology` |
| Algebraic topology | `ChainComplex` | `verify`, `boundary_matrix`, `homology`, `euler_characteristic` |
| Statistical evidence and inference | `StatisticalSample` | `describe`, `covariance`, `empirical_distribution`, `evidence_profile`, `mann_whitney`, `wilcoxon`, `kruskal_wallis`, `ks_2samp`, `spearman`, `kendall`, `permutation_test`, `bootstrap` |
| Survival analysis | `SurvivalDataset` | `verify`, `kaplan_meier` |
| Survival analysis | `KaplanMeierEstimate` | `verify`, `survival_at` |
| Survival analysis | `CoxProportionalHazardsModel` | `verify`, `fit` |
| Survival analysis | `CoxPHFit` | `verify`, `diagnostics`, `predict_partial_hazard` |
| Time series | `TimeSeriesDataset` | `verify`, `acf`, `pacf`, `stationarity_test` |
| Time series | `TimeSeriesAnalysis` | `verify` |
| Time series | `TimeSeriesModel` | `verify`, `fit` |
| Time series | `TimeSeriesFit` | `verify`, `diagnostics`, `forecast` |
| Time series | `TimeSeriesForecast` | `verify` |
| Stochastic processes | `PoissonProcess` | `verify`, `pmf`, `moments`, `increment_distribution` |
| Stochastic processes | `WienerProcess` | `verify`, `finite_dimensional`, `increment_distribution` |
| Stochastic processes | `GaussianProcess` | `verify`, `finite_dimensional`, `condition` |
| Stochastic processes | `ContinuousTimeMarkovChain` | `verify`, `transition_matrix`, `distribution`, `stationary_distribution` |
| Stochastic process results | `FiniteDimensionalDistribution`, `GaussianProcessPosterior`, `CTMCTransition` | `verify` |
| Stochastic differential equations | `StochasticDifferentialEquation` | `verify`, `simulate`, `convergence_study` |
| SDE simulations | `SDESimulation` | `verify`, `path`, `terminal_values` |
| SDE convergence | `SDEConvergenceStudy` | `verify` |
| Generalized linear models | `GeneralizedLinearModel` | `verify`, `fit` |
| Generalized linear models | `GLMFit` | `verify`, `diagnostics`, `predict` |
| Non-parametric results | `NonparametricTestResult`, `ResamplingResult` | `verify` |
| Partial differential equations | `PDEProblem` | `verify`, `classify`, `boundary_compatibility`, `derive_weak_form` |
| PDE results | `PDEClassification`, `PDECompatibilityReport` | `verify` |
| Weak formulations | `WeakForm` | `verify` |
| Finite-element mesh | `FEMMesh` | `verify`, `reference_element`, `finite_element_space` |
| Reference element | `ReferenceElement` | `verify`, `basis`, `quadrature` |
| Finite-element results | `BasisFunctionSet`, `QuadratureRule`, `FiniteElementSpace` | `verify` |
| FEM algebra | `AssembledSystem` | `verify`, `solve` |
| FEM solution | `FEMSolution` | `verify`, `estimate_error` |
| FEM error estimate | `FEMErrorEstimate` | `verify`, `mark`, `compare` |
| Refinement | `RefinementMarking` | `verify`, `refine` |
| Refined mesh | `RefinedMesh` | `verify`, `reference_element`, `finite_element_space` |
| Mesh transfer / convergence | `MeshTransfer`, `FEMConvergenceObservation` | `verify` |

Source objects use the same boundary: transform/complex/probability objects,
graphs and combinatorial structures, finite groups/rings/fields/modules,
signals/filters/control systems, optimization problems, and `Manifold` →
`Chart` → `Metric`/`CoordinateMap`/`TensorField`/`DifferentialForm`, plus
`Point`/`PointSet`/`Polygon`/`Polytope`/`Triangulation`, and finite
`SimplicialComplex`/`CubicalComplex`/integral `ChainComplex`, and typed
`StatisticalSample` observations, `GeneralizedLinearModel` specifications, and
`SurvivalDataset`/`CoxProportionalHazardsModel` survival sources, plus
`TimeSeriesDataset`/`TimeSeriesModel` ordered-time sources, and
`PoissonProcess`/`WienerProcess`/`GaussianProcess`/`ContinuousTimeMarkovChain`
process-law sources, `StochasticDifferentialEquation` Itô models, and structured
`PDEProblem` equations/domains/conditions.
`NonparametricTestResult`, `ResamplingResult`, `KaplanMeierEstimate`, `GLMFit`,
and `CoxPHFit` are derived-only, source-linked records with deterministic exact,
numerical, or seeded-stream replay. `TimeSeriesAnalysis`, `TimeSeriesFit`, and
`TimeSeriesForecast`, `FiniteDimensionalDistribution`,
`GaussianProcessPosterior`, and `CTMCTransition` follow the same output-only
replay boundary. `PDEClassification`, `PDECompatibilityReport`, and `WeakForm`
replay their principal-part, represented-trace, or complete weak-identity result
from the source problem. `FEMMesh` links that weak form and an optional verified
triangulation. `ReferenceElement`, `BasisFunctionSet`, `QuadratureRule`, and
`FiniteElementSpace` are output-only with replayable single- or multi-source
ancestry. `AssembledSystem` retains local and sparse global contributions plus
its space/quadrature sources; output-only `FEMSolution` retains the exact
assembled-system source and replayable solver diagnostics. G.5 output-only
`FEMErrorEstimate`, `RefinementMarking`, `RefinedMesh`, `MeshTransfer`, and
`FEMConvergenceObservation` records retain the complete solution-to-child-mesh
chain, marking policy, parent/child cells, interpolation weights and empirical
rate inputs.
`SDESimulation` and `SDEConvergenceStudy` additionally replay
their PCG64 streams and discretizations. Derived-only types cannot be forged through
public input.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .           # Python mathematical core
pip install -e '.[mcp]'  # add the optional MCP transport
```

Optional extras:

```bash
pip install -e '.[perf]'    # numba — JIT kernels (sieves, GF(2^m), FWHT, closure search)
pip install -e '.[cuda]'    # CuPy + all nvidia-*-cu12 runtime libraries (RTX-class GPU)
pip install -e '.[latex]'   # antlr4 runtime for math_parse_latex
pip install -e '.[dev]'     # pytest
```

Lean 4 + Mathlib is installed by default on first `mathkernel-mcp` start and
via `mathkernel-lean-setup` (`elan` + a pinned lake workspace). Skip with
`MATHKERNEL_SKIP_LEAN_INSTALL=1` (CI/wheel smoke).

> **GPU note:** CuPy wheels ship no CUDA libraries. The `cuda` extra installs the
> matching `nvidia-*-cu12` pip packages — without them, cuBLAS/NVRTC DLL loads fail
> even though `import cupy` succeeds. GPU availability is probed at runtime with a
> real matmul, so a broken stack degrades gracefully to CPU. Verify your stack with
> `python scripts/gpu_smoke.py`.

## Quickstart — MCP server

```bash
mathkernel-mcp
```

The server speaks MCP over stdio (FastMCP 3) and ships **core instructions** to the
client at initialize time: discover → parse → context → trust discipline → async jobs
→ provenance. 162 tools, all prefixed `math_`.

Typical agent session:

```text
math_capabilities                                   # discover surface, limits, engines
math_parse("x^2 - 3*x + 2 = 0")                     # -> expr_id
math_context_create(domains={"x": "real"})          # -> context_id
math_reason(expr_id, context_id, formal=true)       # solve + independently verify
math_derivation_trace(step_id)                      # full provenance on demand
```

Long-running sweeps are async:

```text
math_job_submit("collatz", {"n_max": 14})  ->  math_job_status(job_id)  ->  math_job_result(job_id)
```

## Quickstart — Python library

The MCP server is a thin transport layer; everything is available in-process:

```python
from mathkernel import MathKernel

kernel = MathKernel()

# symbolic
r = kernel.parse("x^2 - 2 = 0")
sol = kernel.solve(r.data["expr_id"], "x")
assert sol.ok and sol.trust.value == "symbolic"

# exact GF(2^m) field arithmetic
f = kernel.gf2m_create(8, "1b")  # AES polynomial x^8 + x^4 + x^3 + x + 1 (hex reduction part)
kernel.gf2m_compute(f.data["field_id"], "mul", ["53", "ca"])

# finite dynamics: an explicit eight-state cyclic permutation
transition = [1, 2, 3, 4, 5, 6, 7, 0]
fs = kernel.finite_system_create("uniform", transition)
km = kernel.koopman_matrix(fs.data["system_id"], {"kind": "walsh", "r": 3})
vis = kernel.koopman_visibility(fs.data["system_id"], {"kind": "walsh", "r": 3})
# Exact zeros certify the requested modes in this declared finite model.

# closure relations (njit meet-in-the-middle)
kernel.closure_search("cyclic", m="97", weight_bound=10, multipliers=["1", "5"])
```

Standalone modules (`mathkernel.gf2m`, `mathkernel.koopman`, `mathkernel.relations`,
`mathkernel.cumulants`, `mathkernel.finite_fourier`, `mathkernel.transforms`,
`mathkernel.integral_transforms`, `mathkernel.complex_analysis`,
`mathkernel.continuous_probability`, `mathkernel.integers`,
`mathkernel.computational_geometry`, `mathkernel.algebraic_topology`,
`mathkernel.collatz`, `mathkernel.cuboid`) are usable without the facade when
you don't need derivation tracking.

## Trust model

```text
formal                  Lean certificate accepted by the Lean kernel
exact                   exact computation / checked claim-specific certificate
symbolic                symbolic engine agreement (e.g. SymPy residual checks)
interval_certified      rigorous enclosure (mpmath interval)
numeric_high_precision  arbitrary-precision numeric
numeric                 float evidence (incl. GPU fast paths)
empirical / heuristic / unknown
```

**Overall trust is limited by the weakest evidence required to establish the claimed
result** — never the maximum trust emitted by any single node. Independent backend
disagreement is preserved as an explicit conflict, not averaged away.

Every `MathResult` also carries an `evidence_bundle` with separate computation,
proof, certificate, numerical, model and empirical evidence. `claim_evidence`
retains those bundles per conclusion instead of flattening unlike claims into one
score. The legacy `trust` field remains a conservative summary and is automatically
capped by the evidence required for the result. A producer-supplied
`justified_trust` is a ceiling, never an override; an unverified proof or certificate
supports only `unknown`.

Semantic statuses distinguish proof or certification strength from mathematical
outcomes such as `does_not_exist`, `undefined`, `infeasible` and `unsupported`.
These distinctions survive MCP serialization, asynchronous job retrieval,
derivation replay, visualization and multimodal artifact assembly.

The capability registry separates advertised trust levels from verification
methods. Query it by domain, input/output type, operation, trust level,
verification method or engine; capability records also identify their execution
handler and meaningful cost dimensions. Expression plans record the resolved
capability route before the existing obligation executor runs it.

Exact and numeric paths are strictly separated: koopman/finite-dynamics tools default
to `exact=true` (proof-grade rational/cyclotomic values); `exact=false` selects the
vectorized numeric path (CuPy GPU when usable) and downgrades trust to `numeric`.

**Decimal literals are approximate observations.** A decimal (`RealNode`) anywhere in
an expression caps its trust at `numeric` from `parse` onward — `0.1 + x` parses as
`numeric`, `1/2 + x` as `symbolic`. Formal certificates (Lean) and exact SMT
counterexamples are refused for approximate inputs, because the backends would encode
decimal syntax as exact rationals — silently proving a different statement. Use exact
rationals or interval certification when proof-grade evidence is needed.

## Continuous symbolic mathematics

Continuous domains use typed objects and the compositional
`object_create` → `apply` model rather than exposing a flat CAS surface.
Every operation records a four-obligation DAG: typed-input validation,
candidate computation, domain-invariant verification and conservative evidence
reconciliation.

- **Integral transforms** — Laplace, Fourier, Mellin and bilateral Z transforms
  with explicit conventions, assumptions and regions of convergence. Inverse Z
  uses annulus-aware Laurent/residue extraction when justified. Verification
  records round-trip, linearity, convolution, differentiation, value-theorem and
  ROC obligations separately; unresolved obligations remain `unknown`.
- **Complex analysis** — derivatives, analyticity candidates, zeros,
  singularities, Laurent series, residues, contour integration, winding numbers,
  argument-principle accounting, conservative identity continuation and
  domain-aware conformal maps. Branch conventions, cuts, excluded points,
  contour orientation and boundary incidents remain explicit.
- **Continuous probability** — typed univariate, random-variable, joint and
  conditional distributions; PDF/CDF/survival/quantile, moments, transforms,
  entropy, truncation, convolution, mixtures, divergence, marginals,
  conditioning/Bayes, covariance/correlation and order statistics. Support,
  parameter constraints, Jacobians and inverse branches are retained.

Symbolic availability is candidate evidence, not independent proof. Same-engine
identities are capped at `symbolic`; decimal ancestry remains capped at
`numeric`. `does_not_exist` (for example, a Cauchy mean) is distinct from an
unsupported method or an unresolved convergence question.

Conventions and assumptions are part of the object. Fourier sign and
normalization, transform source/target variables, complex branches/cuts,
probability supports and parameter constraints are never selected silently.
Contour orientation and singularity accounting are mandatory where the theorem
depends on them.

Verification is operation-specific. Transforms retain every checked or unresolved
identity and ROC obligation. Residues are compared with defining limit/derivative
or Laurent-coefficient formulas; contour claims retain enclosed singularities,
cuts and winding numbers. Probability verifies normalization, support-aware
nonnegativity, CDF boundaries/derivative/monotonicity when decidable, and
Jacobian branches. These are symbolic checks unless an exact certificate or
separate numerical record says otherwise.

Failures use semantic statuses: `candidate`, `unknown`, `unsupported`,
`does_not_exist`, and `error` are distinct. Known limitations include
non-product joint supports, continuation without an explicit overlapping source
domain, branch-sensitive argument-principle inputs, transforms whose ROC SymPy
cannot establish, and general multivariate changes of variables without supplied
inverse branches/Jacobians.

Continuous symbolic work is bounded by the global AST/output/solver-time limits
and dedicated contour, joint-dimension, mixture-component, series-order,
order-statistic and inverse-branch limits. Raise the corresponding
`MATHKERNEL_MAX_*` value explicitly when a larger request is intentional.

```python
# PDF → Laplace transform, preserving support and evidence ancestry
d = kernel.object_create("Distribution", {
    "family": "exponential", "parameters": ["2"], "variable": "x",
})
r = kernel.apply(d.data["object_id"], "integral_transform", {
    "transform": "laplace",
    "transform_variable": "s",
    "convention": "laplace_standard",
})
assert r.data["value"] == "2/(s + 2)"
```

## Finite dynamics & PRNG analysis

A distinctive capability: exact spectral analysis of finite dynamical systems
`(X, μ, T, O)` — built for (and validated on) PRNG structure analysis.

- **Koopman suite** — transport matrix Q, observation-transfer C, mode visibility
  ρ_O, lagged state tensors (raw/connected), observed statistics, IPR/entropy
  diagnostics. Walsh bases for GF(2)^r, character bases for Z_M.
- **Stochastic observation transfer (library API)** — exact
  `FiniteJointLaw` contractions for arbitrary finite latent joint laws; ordered
  Markov path moments/cumulants with the required multiplication operators;
  statewise multiplicativity-defect certificates; and exact finite-noise
  deterministic dilations for rational Markov kernels. The accompanying
  published primate quartet pilot deliberately records that the earlier K3ST
  split-zero diagnostic does not survive outside its group-based assumptions.
- **Branching General Markov tensors (library API)** — exact
  `FiniteMarkovTree` sum-product laws and cumulants on heterogeneous rooted
  trees; exact `L M R` edge-flattening certificates with the sharp transition-
  rank bound; local stochastic observation channels as Kronecker transforms;
  exact left-inverse recovery, collision witnesses, collective sensor fusion,
  and channel-conditioned singular-value bounds. The published primate pilot
  distinguishes algebraic identifiability from finite-sample stability.
- **Statistical phylogenetic inference (library API)** —
  probability-simplex projection; known-channel EM and constrained ridge
  recovery; held-out regularization selection; multinomial covariance and
  tangent-space Fisher information; nonnegative-rank multinomial likelihood;
  covariance-Wald rank diagnostics; and tie-safe quartet scoring. Controlled
  GM(4) experiments quantify the shared singular-value origin of visibility
  loss and inverse instability. Two fixed published-data pilots add site and
  moving-block bootstrap checks without claiming broad competitive accuracy.
- **Frozen phylogenetic benchmarking (library API)** —
  FASTA, relaxed PHYLIP, practical NEXUS and Newick ingestion; portable source
  SHA-256 manifests; canonical protocol and corpus locks; result-blind quartet
  sampling from reference-tree splits; complete-case site provenance; site,
  circular-block, partition-stratified and whole-partition resampling; rank-tail,
  p-distance and normalized log-det baselines; and tie-safe corpus summaries.
  The bundled execution evaluates 22 predeclared correlated units from two
  published source alignments and a 1,920-alignment known-truth stress grid. A
  separate lock fixes the first 20 eligible BenchmarkAlignments datasets before
  acquisition; that external corpus is explicitly pending rather than silently
  replaced.
- **Observable connected-relation detection (library API)** -
  exact and numerical pure-interaction laws; weighted conditional-expectation
  singular spectra; mode-specific stochastic visibility; exact local-channel
  transfer of connected amplitude; chi-square and null-Fisher information
  retention; exact invisibility certificates; finite necessary and constructive
  sufficient sample bounds; binary-parity scaling; and complementary sensor
  fusion. The controlled theorem shows that local visibility losses multiply
  in amplitude and square in information, yielding an `s^(-2d)` detection-cost
  law in the homogeneous binary specialization.
- **Relation-subspace visibility and sensor design (library API)** -
  finite multi-parameter local relation laws; latent and observed Fisher Gram
  matrices; generalized retained-information eigenvalues and principal
  visibility directions; exact observation-blind collision certificates;
  direction-level information and sample multipliers; rank, E-optimal, trace,
  D-optimal and pseudo-logdet sensor-subset selection; efficient empirical
  partition transfer; and score-mean long-run-covariance correction. A frozen
  SkewDB adapter adds source/schema auditing, discovery/validation/challenge
  splits by held-out taxonomy, discovery-only preprocessing, source hashing and
  a fail-closed raw-data runner. The bundled SkewDB fixture is explicitly
  synthetic because the current full payload was not acquired in this
  environment.
- **Coordinate-invariant relation geometry (library API)** -
  finite-simplex tangent vectors with the intrinsic Fisher metric; stochastic
  tangent pushforward; coordinate-invariant generalized retained-information
  eigenvalues; exact score/tangent equivalence; exact local chi-square transfer;
  worst-direction minimax necessary sample bounds; finite Bhattacharyya and
  retention-based pointwise sufficient counts; and iid, moving-block and
  cluster bootstrap intervals for ordered relation spectra. A SHA-256-locked
  local-resolution SkewDB adapter converts documented cumulative `*_fit.csv`
  tracks to window increments and explicitly separates genuine inputs from the
  bundled source-parameterized generated fixture.
- **Finite Fourier** — exact arithmetic in ℚ(ζ_L) via cyclotomic polynomials:
  DFT over Z_M, output-transfer transforms, two-point difference coefficients,
  measure Fourier transforms, orbit corrections.
- **Closure search** — short irreducible relations selected by the dynamics:
  cyclic (`Σ k_j·a^j ≡ 0 mod m`) and binary (`⊕ (L^{jK})ᵀ w_j = 0`),
  meet-in-the-middle with L1/Hamming weight bounds.
- **GF(2^m) from transitions** — reconstruct the field (dual-orbit cyclic basis,
  minimal/reduction polynomial, Rabin-verified) purely from a generator's
  GF(2)-linear transition columns.
- **State-conditioned dynamics** — exact per-state orbit access `T^κ(x)(x)`:
  least-lag solving, symmetry-to-access conversion, cocycle composition,
  exhaustive additive closure proofs, symbolic affine access maps, GF(2)
  baby-step/giant-step orbit solving, sparse giant-lag predictive closures,
  and constrained symmetry discovery where numeric probing only ranks
  candidates — canonical-rewrite or exhaustive proofs decide.

The `scripts/` tree contains uniform, end-to-end reproductions for 25+ generators
(xorshift/xoroshiro/xorwow families, MT19937, Melg19937, WELL19937a, MRG32k3a,
PCG32/64(+fast), LXM, SplitMix64, SFC64, JSF64, Romu, Philox, Threefry, RXS-M-XS),
each runnable from scratch with `scripts/families/run_all.py` and
`scripts/companion/run_all.py`. Reference data ships in `scripts/data/` — no
external fixtures required.

## Engineering mathematics

MathKernel provides typed engineering mathematics for signals, control systems and constrained optimization while preserving the same evidence and persistence contracts as the symbolic core.

### Signals and spectra

Continuous and sampled signals carry explicit domains, sample grids and units. Spectral representations are typed rather than treated as anonymous arrays. FIR/IIR filters and filter designs retain coefficients, conventions and source signals, while immutable streaming state makes block-by-block processing replayable. Frequency-response and time-response operations record whether they used exact symbolic algebra or numerical evaluation.

### Control systems

Typed SISO and MIMO models support state-space and transfer-function representations, continuous/discrete conversion, poles and zeros, stability checks, discretization, controller construction and observer construction. LQR, finite-horizon LQR, steady-state Kalman filtering, LQG composition and immutable Kalman prediction/update states retain plant/model ancestry and separate algebraic checks from modeling assumptions.

Constrained finite-horizon MPC keeps feasibility, optimality, terminal invariance, recursive-feasibility and stability claims separate. Frequency-domain analysis includes Bode, Nyquist and root-locus representations together with checked time responses.

### Optimization and certificates

Linear and quadratic programs can return exact/checkable optimality witnesses where the supported fragment permits it. Infeasible LPs can expose Farkas certificates and unbounded problems can expose recession rays. MILP search results carry replayable proof trees rather than only an incumbent value. Conic and quadratic-constraint workflows support bounded SOCP/SDP product cones and Lagrangian-style certificates in their declared fragments.

External native candidate solvers are isolated in fresh processes with bounded requests and hard timeout termination. Candidate generation and certificate verification are distinct steps: a solver finding a point does not by itself establish a stronger claim than the verifier can check.

## Geometry and topology

### Differential geometry and tensor calculus

Immutable `Manifold`, `Chart`, and `Metric` objects feed typed `GeometryTensor`, `Connection`, and `GeodesicSystem` outputs. Metric operations compute inverse metrics, Christoffel symbols, Riemann/Ricci/scalar/Einstein curvature and affine geodesic equations. Exact symbolic checks cover inverse identities, torsion freedom, metric compatibility, Riemann symmetries, the first Bianchi identity and the contracted Bianchi identity. Chart domains and metric nondegeneracy conditions remain explicit.

Directional `CoordinateMap` objects carry explicit Jacobians and inverse-composition checks. Dense variance-aware `TensorField` objects and canonical sparse `DifferentialForm` objects support covariant and Lie derivatives, wedge products, exterior derivatives, interior products, pullbacks and Hodge stars. Checks include graded commutativity, `d²=0`, pullback commutation with `d`, metric compatibility, coordinate-map composition and the Hodge double-star sign when metric signature is supplied. Orientation and signature are never guessed.

### Computational geometry

`Point`, `PointSet`, `Polygon`, half-space `Polytope`, `Triangulation`, and derived `VoronoiDiagram` objects provide exact orientation, incircle and segment-intersection predicates, monotone-chain convex hulls, winding containment, exact squared-distance nearest neighbors, certified ear clipping, convex polygon clipping, empty-circumcircle Delaunay triangulation and finite Voronoi duals with explicit unbounded rays. Decimal predicates use conservative floating-point error filters; when topology cannot be established, the result is explicitly ambiguous rather than promoted to an exact classification.

### Algebraic topology

Exact finite `SimplicialComplex`, `CubicalComplex`, and integral `ChainComplex` objects expand cells to canonical face closures and derive oriented boundary matrices. Complexes verify `boundary[k-1] * boundary[k] = 0` before homology is attempted. `homology` computes free ranks and integer torsion over Z through certified Smith-kernel/quotient reductions, and exact Betti numbers plus representative cycles over Q or GF(p). `boundary_matrix`, `chain_complex`, and `euler_characteristic` expose ordered bases and the Euler–Poincaré cross-check.

Verified exact triangulations can be converted into canonical simplicial complexes and composed directly with homology operations; numeric or refuted triangulations cannot cross that exactness boundary. Closure expansion is bounded before combinatorial growth can exceed configured topology limits. Persistent homology, cohomology products and infinite/CW-complex inference are not claimed.

## Statistics and stochastic modeling

### Samples and descriptive statistics

`StatisticalSample` stores a rectangular nonempty matrix of finite concrete real observations, unique variable labels, optional unique observation IDs and explicit asserted sampling/population/design metadata. `describe` derives exact or ancestry-capped numeric moments and type-7 order statistics; `covariance` derives centered cross-products with sample or population normalization; `empirical_distribution` preserves exact frequency counts and rational probabilities; and `evidence_profile` audits the evidence boundary itself.

The required evidence establishes only calculations on the stored observations. Sampling metadata, empirical support and model assumptions stay in separate diagnostic evidence records, while population generalization and model validity remain explicitly unestablished. Missing values, unresolved symbolic observations and silent imputation are refused. Decimal input cannot upgrade, resource limits are checked before expensive work, and every derived object retains its source across persistence and restart.

### Generalized linear models

Immutable `GeneralizedLinearModel` objects link to stored samples and produce derived-only `GLMFit` objects. Supported canonical pairs are Gaussian/identity, binomial/logit and Poisson/log. `verify` checks response domain, design rank and residual degrees of freedom; `fit` reports ordered coefficients, covariance/standard errors, fitted conditional means, deviance, null deviance, dispersion, convergence, score residual and conditioning. Fits independently support `verify`, `diagnostics`, and `predict`.

Exact-input Gaussian models use sufficient cross-products and exact normal equations. Numeric Gaussian fits use checked float64 least squares; logistic and Poisson fits use deterministic float64 IRLS. Rank deficiency, invalid or degenerate response domains, non-convergence, singular/ill-conditioned information and detected complete/quasi separation fail closed without a fit object. No ridge term, row deletion, imputation or family/link substitution is silent. Coefficient, covariance, deviance and prediction claims remain conditional on the stored sample/design; model validity, population generalization and causal effects are not inferred.

### Nonparametric tests, permutation tests and bootstrap

Stored samples support `mann_whitney`, `wilcoxon`, `kruskal_wallis`, `ks_2samp`, `spearman`, and `kendall`, with explicit average ranks and tie corrections. `method="auto"` performs complete exact sign/label/permutation enumeration only when both state and work estimates fit configured bounds; otherwise the result names its normal, chi-square, Kolmogorov or Student-t approximation. Thus an exact p-value is an exact conditional null calculation for the stored observations, while an asymptotic p-value remains numerical evidence without a finite-sample error theorem.

`permutation_test` supports mean/median differences using exact enumeration or explicitly seeded PCG64 Monte Carlo with an add-one p-value. `bootstrap` supports mean/median percentile intervals with a mandatory uint64 seed, bounded draws and memory-bounded batches. Simulated results record random algorithm, seed, draw count and replay configuration. Exchangeability, sampling design, asymptotic validity, population coverage and causal interpretation remain separate assumptions or unestablished claims.

### Survival analysis

`SurvivalDataset` stores durations, exact binary event indicators, optional delayed-entry times and optional strata inside an immutable statistical sample. `kaplan_meier` constructs exact risk sets and product-limit values together with numerical Greenwood standard errors and two-sided log-log intervals. Multi-stratum inputs require an explicit stratum, and `survival_at` queries the right-continuous step curve.

`CoxProportionalHazardsModel` provides an unstratified Cox surface with explicit Efron or Breslow ties. Its deterministic float64 Newton fit uses monotone line search and refuses rank-deficient, event-sparse, non-convergent, singular, over-conditioned or separation-like cases. `CoxPHFit` records coefficients/hazard ratios, covariance/standard errors, partial likelihood, score residual, baseline hazard, concordance and Schoenfeld time correlations, with replay verification, diagnostics and bounded partial-hazard prediction. Independent censoring, proportional hazards, population generalization and causality remain assumptions or unestablished.

### Time-series models and forecasting

`TimeSeriesDataset` preserves row order, distinct time/value columns, strict timestamps, reject-missing policy and detected regular spacing. Exact-source `acf` uses a common lag-zero centered denominator and `pacf` uses Durbin–Levinson recursion. `stationarity_test` provides a numerical constant-case ADF regression with named asymptotic critical values rather than inventing an exact p-value or claiming stationarity is proved.

`TimeSeriesModel` covers AR, MA, ARMA, ARIMA and GARCH orders, constant choice, Gaussian innovations and initialization. ARMA-family fits use bounded conditional-sum-of-squares optimization; GARCH uses constrained Gaussian likelihood with positive variance and persistence below one. Derived fits record coefficients, residual/fitted series, conditional variance, roots, likelihood, AIC/BIC and convergence, with Ljung–Box/Jarque–Bera diagnostics. Forecasts derive regular future times, recursive means and Gaussian intervals using ARIMA impulse responses or GARCH variance recursion. Irregular spacing may be analyzed but not fitted.

### Stochastic processes

Immutable `PoissonProcess`, `WienerProcess`, `GaussianProcess`, and `ContinuousTimeMarkovChain` objects expose finite-dimensional laws and checked derived artifacts. Poisson count masses/moments and Wiener means/covariances are symbolic or exact. Gaussian-process finite laws support RBF, Matérn-3/2, linear and Brownian kernels with numerical PSD checks; conditioning uses bounded float64 Cholesky solves, explicit observation-noise variance and optional stored jitter without silently fitting hyperparameters. CTMC verification checks generator and initial-law axioms exactly; transitions use a checked matrix exponential, while stationary laws use an exact left-nullspace system and preserve nonuniqueness.

Independent/stationary increments, continuity, Gaussianity, kernel suitability and time homogeneity remain declared model assumptions rather than facts established by calculation.

### Stochastic differential equations

`StochasticDifferentialEquation` supports vector Itô systems with declared symbol scope, drift vector, full state-by-noise diffusion matrix, concrete initial state and finite interval. Euler–Maruyama supports vector states and full diffusion. Milstein is restricted to scalar state/scalar noise and uses the symbolic diffusion derivative; unsupported multidimensional cases are refused rather than silently substituting another scheme.

Simulation records the exact step grid when possible, float64 paths, PCG64 algorithm/seed/stream, terminal sample moments and nominal strong/weak orders. Large outputs expose compact metadata plus bounded path/terminal queries. Coupled convergence studies reuse a finest Brownian stream across multiple step sizes and report observed terminal RMS convergence when defined. Simulation and convergence remain numerical/empirical; nominal orders, existence, uniqueness and regularity are assumptions, not proofs.

### Statistical evidence and persistence

Across all statistical/stochastic objects, exact, symbolic, asymptotic, numerical, empirical and model evidence remain distinct. Derived types are output-only, replay operates under current limits, decimal ancestry cannot upgrade, persisted JSON is integrity checked before decoding, and stored type/class/source fields are reconciled to prevent cross-type source substitution.

## PDEs and adaptive finite elements

### PDE representation and classification

Typed PDE problems support scalar and coupled systems, declared independent/dependent variables, derivative multi-indices, coefficients/parameters and explicit initial/boundary conditions. Principal-part analysis classifies the represented system only within the declared symbolic fragment, and trace compatibility checks distinguish represented boundary information from stronger claims such as existence, uniqueness, regularity or well-posedness.

### Weak forms

`PDEFunctionSpace`, `PDEMeasure`, `WeakIntegralTerm`, `IntegrationByPartsStep`, and output-only `WeakForm` artifacts represent weak formulations explicitly. `derive_weak_form` requires integration variables, ordered trial spaces, test spaces, boundary-trace indices and selected term/coordinate transfers; it does not guess analytic spaces or silently integrate terms.

Variable-coefficient integration by parts retains the complete product rule, storing differentiated-test and coefficient-derivative volume terms separately. Every transfer emits oriented boundary faces. Boundary terms that vanish under declared zero test traces remain represented and are marked as such. Dirichlet, Neumann/Robin and periodic indices are recorded as essential, natural and periodic partitions. `WeakForm.verify` reconstructs spaces, measures, volume/boundary terms, signs, product-rule derivatives, partitions and derivation steps from the source PDE. The verified claim is the represented integral identity under declared assumptions—not a theorem of solvability or regularity.

### Meshes, reference elements and finite-element spaces

`FEMMesh` supports interval, triangle and tetrahedron simplices. Construction checks bounded connectivity, nondegeneracy, canonical positive orientation, boundary/interior facet incidence, induced boundary ownership and cell connected components. A compatible stored `Triangulation` can provide triangle connectivity while preserving geometry and weak-form ancestry. Combinatorial replay does not infer geometric non-overlap or approximation quality.

`reference_element` provides canonical unit simplices. `basis` derives symbolic nodal P1 Lagrange functions and gradients and checks the Kronecker property, partition of unity and gradient sum. `quadrature` supplies bounded exact-moment rules for the supported simplex degrees. `finite_element_space` builds P1 vertex-DOF C0 spaces with explicit local-to-global connectivity and essential boundary DOFs. Derived objects are replayable and output-only.

### Assembly and algebraic solves

`AssembledSystem` and `FEMSolution` support scalar linear stationary weak forms on affine P1 simplices. Assembly stores dense local matrices/vectors and Jacobian determinants, coalesces the global matrix into ordered sparse entries, integrates supported Neumann/Robin facet terms and performs documented symmetric elimination for Dirichlet DOFs while retaining raw and transformed systems. Concrete substitutions resolve remaining PDE parameters through restricted MathIR.

Assembly distinguishes exact integration from an exact finite quadrature sum. Insufficient-order or non-polynomial quadrature may still define a replayable algebraic system, but `quadrature_exact=false` records the limitation. Unsupported strong second derivatives, time derivatives, coupled/nonlinear fields, periodic constraints, unresolved parameters and missing boundary fluxes fail closed.

Solves select exact rank/augmented-rank analysis or an explicit SciPy sparse numeric path. `FEMSolution` records `unique`, `ill_conditioned`, `singular_inconsistent`, `singular_underdetermined`, or `singular_least_squares`, together with residual and conditioning diagnostics. Verification establishes the transformed finite-dimensional system and solver outcome only, never a continuous PDE solution theorem or continuum error bound.

### Error estimation and adaptivity

`FEMSolution.estimate_error` provides residual–jump indicators for complete unique or ill-conditioned P1 solutions in its supported scalar stationary diffusion fragment. Each `CellErrorIndicator` retains diameter-weighted strong residual, interior conormal-jump contribution, natural-boundary contribution and total. `FEMErrorEstimate` stores local/global estimator values, quadrature-exactness and algebraic residual separately, and always records `rigorous_error_bound=false`; reliability and efficiency constants are not inferred.

`FEMErrorEstimate.mark` implements deterministic Dörfler and maximum policies. `RefinementMarking.refine` applies triangle red refinement and propagates conforming closure through shared edges. `RefinedMesh` records requested/closure cells and child-to-parent mappings; `MeshTransfer` records refined P1 nodal values as explicit affine combinations of parent DOFs. Refined meshes can re-enter the basis, quadrature, space, assembly, solve and estimation chain.

`FEMErrorEstimate.compare` accepts direct parent/child refinement pairs and reports estimator ratios and observed two-mesh rates. `FEMConvergenceObservation` is explicitly empirical evidence about an estimator sequence, not a convergence theorem or continuum error bound.

## Relation and information-geometry inference

### Composite relation inference

`mathkernel.composite_relation_inference` provides a quadratic score test for an entire visible relation subspace. Generalized observed scores are whitened under the nominal law and the statistic is the squared norm of their sample mean. A finite bounded-score argument supplies a conservative guarantee with explicit dependence on relation dimension, weakest retained-information eigenvalue, perturbation radius and score bound.

The same module computes nuisance-adjusted target information through latent and observed Fisher Schur complements. It reports exact post-observation confounding when a target direction can be reproduced by nuisance variation. For repeated or nearly repeated information eigenvalues, bootstrap uncertainty is attached to invariant eigenspaces through principal angles rather than arbitrary individual eigenvectors. Studentized ordered-spectrum intervals, dependence-informed circular-block heuristics, nominal/empirical/HAC covariance modes and norm-bounded misspecification guarantees are available with their assumptions recorded.

### Robust relation inference

`mathkernel.robust_relation_inference` provides model-scoped quadratic inference, learned nuisance projections, orthogonal residual relations and VAR-prewhitened long-run covariance estimation.

| Python API | Function and evidence boundary |
| --- | --- |
| `quadratic_minimax_bounds` | Gaussian-sequence lower/upper rates using the inverse information spectrum; separate finite iid U-statistic bound under a justified covariance envelope |
| `gaussian_quadratic_test` | Weighted-square test with finite Gaussian Chernoff threshold |
| `quadratic_u_test` | O(Nr) unbiased pair statistic; finite Cantelli calibration for iid known-null scores |
| `prewhitened_long_run_covariance` | VAR(1), automatic Bartlett bandwidth, recoloring and persistence diagnostics; consistency assumptions remain necessary |
| `quadratic_moment_test` | Full-rank asymptotic Wald test with empirical or supplied covariance; singular covariance is rejected |
| `relation_folds` | Reproducible iid, group-preserving or contiguous folds |
| `crossfit_nuisance_projection` | Out-of-fold nuisance-projection estimation in a declared candidate span |
| `crossfit_residual_relations` | Orthogonal residual cross-moments with learned conditional means, custom learners and exclusion gaps |

These research APIs remain numerical/model-scoped unless a stronger finite guarantee is explicitly returned. They do not acquire formal-proof or interval-certification labels merely because they are composed with other MathKernel objects.

### Relation visibility, sensor design and information geometry

The relation-analysis stack also includes exact observable-relation visibility, information-retention calculations, sample-cost diagnostics, multi-relation Fisher geometry, sensor-design objectives, coordinate-invariant tangent representations, local testing bounds and uncertainty for information spectra. Numerical near-null directions are kept distinct from mathematically exact blind directions.

## Performance: numba · CUDA · parallelism

| Workload | CPU fast path | GPU path | Parallel |
|---|---|---|---|
| Collatz sieve | njit (n ≤ 31) | CUDA RawKernel | persistent process pool |
| Cuboid sweep | njit leg-pair scan + QR prefilter | CUDA RawKernel | process pool |
| GF(2^m) ≤ 1024 | njit n-limb (uint64×N) kernels | — | — |
| Integer batch | njit array kernels | — | persistent process pool, adaptive chunksize |
| Graph BFS/components | njit CSR traversal, certificate re-verified | — | — |
| GF(p^m), p < 2^24, m ≤ 64 | njit uint64 polynomial mul/mod | — | — |
| Cayley-table validation | njit axiom scan | — | — |
| Recurrence extension | checked int64 njit, bigint fallback | — | — |
| FWHT | int64 njit butterfly | — | — |
| Closure search | njit MITM (int64/uint64) | — | — |
| Koopman / finite dynamics | numpy complex128 | CuPy matmul | — |
| Obligation DAG | — | — | thread waves |
| Long sweeps | — | — | async job pool |

Exact symbolic types (`Fraction`, `CyclotomicNumber`) are deliberately pure Python —
a visibility zero or closure cancellation must remain a *proof*. Numeric twins exist
where scale demands it and always carry `trust: numeric`.

**Expansion contract.** New domains must design verification and
performance tiers together from the start: exact typed semantics and limits,
an independently checkable certificate for every `VERIFIED` claim, and — where the
workload is regular enough — a Numba/process/GPU fast path behind a narrow exactness
fragment with automatic Python fallback. Fast paths must be re-verified or
differential-tested against the reference implementation and must record the selected
backend in evidence metadata; they may never raise trust beyond the underlying proof.
GPU offload is mandatory only for regular device-exact workloads; irregular
arbitrary-precision algorithms document the considered tiers instead.

### Correctness-preserving optimization

MathKernel optimizes only where the mathematical contract survives the optimization. Regular bounded integer/array workloads use Numba, process or GPU paths with differential checks and guarded fallbacks. Exact symbolic workloads stay on exact representations when converting them to floating point would weaken the claim. Profiling is used to remove repeated symbolic work, hoist invariant computations, cache replayable certificates and replace avoidable superlinear verification passes without changing stored mathematical evidence. Backend selection is recorded in evidence metadata and never raises trust above the underlying computation or certificate.

## Visualization & portable artifacts

`mathkernel_viz` turns MathKernel objects and results into evidence-carrying interactive
artifacts. Visualization is downstream of mathematics: it consumes typed source data or
a `MultimodalProjection`, records presentation transformations, and never upgrades the
source evidence merely because a particular graphical form is used.

```python
import mathkernel_projection as mkp
import mathkernel_viz as viz

projection = mkp.create_projection(
    "matrix",
    {"matrix": [[1, 2], [3, 4]]},
    trust="exact",
)
doc = viz.from_projection(projection)
viz.export_html(doc, "matrix.html", mode="portable")
```

The lower-level dashboard API remains available for direct composition:

```python
import mathkernel_viz as viz

doc = viz.dashboard("My result", cols=2)
viz.add_point_cloud(doc, points, trust="numeric")
viz.add_histogram(doc, values, bins=128)
viz.add_select(doc, "lag", [
    {"label": "k=4", "value": {"embed": {"lags": [0, 4, 8]}}}
])
viz.export_html(doc, "out.html", mode="portable")
```

- **Building blocks, not monoliths** — artifacts compose reusable panels such as
  `point_cloud_3d`, `trajectory_3d`, `surface_3d`, `vector_field_3d`, `plot2d`,
  `histogram`, `heatmap`, `dag`, `metric_grid`, `data_table`, `text` and `select`.
- **Renderer-neutral IR** — the versioned `VisualizationDocument` is consumed by
  pure-Python SVG, optional matplotlib PNG/PDF, and the HTML+Three.js renderer.
- **Interactive 3D** — orbit/pan/zoom and hover inspection of identity and trust.
- **Portable HTML** — one self-contained `.html` with embedded datasets, provenance,
  reproducibility metadata and viewer runtime; no server or CDN is required.
- **Evidence-preserving** — block/series/dataset trust is inherited conservatively;
  interval-certified display is only used when the source itself carries that support.
- **Integrity & determinism** — payload and per-dataset SHA-256 are exposed, and
  identical inputs produce deterministic artifacts.
- **Secure presentation boundary** — CSP, escaped labels, no `eval`, dataset limits,
  and MathIR treated as data rather than executable code.

## Shared multimodal projections

The shared `mathkernel_projection` layer defines canonical mathematical projection
families that can feed visualization, sonification, or a combined research artifact.
This prevents each renderer from inventing its own interpretation of a matrix, mesh,
graph, field, distribution or high-dimensional object.

A `MultimodalProjection` records:

- source lineage (`SourceRef`);
- projection family and structured payload;
- coordinates, units and labels;
- assumptions and evidence references;
- deterministic transformation provenance;
- explicit basis, slice, traversal or ordering parameters;
- output dimensionality and declared information loss.

The canonical families cover scalar/vector fields; point sets/clouds; curves, surfaces
and trajectories; sequences and distributions; matrices and tensors; graphs, evidence
graphs, expression trees and certificate trees; spectra and complex-valued fields;
regions and implicit sets; meshes and geometric complexes; ODE/PDE solutions and
dynamical systems; optimization and statistical-inference objects; finite-field/GF(2)
structures; relation/information geometry; sets, partitions and piecewise objects;
quantities with units; ensembles; and explicit higher-dimensional projections.

For source dimension greater than three, a projection method and output dimensionality
must be explicit. Coordinate selection, a declared basis, PCA-like reduction or a
domain-specific spectral projection are transformations that must be recorded; a
renderer cannot silently decide which view is canonical.

A registry of result adapters (`mathkernel_projection.result_adapters`) maps stored
typed objects and flat result payloads onto these families automatically. Adapters
are pure extraction functions: they never recompute mathematics, never upgrade
trust, and declare any presentation choice (sampling grids, magnitude-only spectra,
channel selection, covariance-to-band reduction) in `parameters` and
`information_loss`. `math_visualize(object_id=...)` and
`math_projection_create(source_object_id=...)` use the registry to choose the
canonical projection for signals, spectra, filters, pole-zero maps, frequency
responses, root loci, time responses, distributions (symbolic densities are sampled
on a declared window), empirical/discrete distributions, statistical samples,
GLM fits, Kaplan-Meier estimates, Cox baseline hazards, ACF/PACF diagnostics,
time-series fits, graphs and traversal trees, optimization results, ODE/SDE
ensembles, FEM meshes/solutions/error indicators/convergence observations,
assembled-system sparsity patterns, PDE grids, point sets, polygons, triangulations,
Voronoi diagrams, generating functions, Cayley tables, contours, singularity maps,
subgroup/coset/orbit partitions, combinatorial counts, and unit quantities.
Unregistered object types fail with a typed error rather than an invented view.

Evidence graphs are first-class: claim -> evidence -> assumption/source relationships
can be visualized directly, making MathKernel's verification structure inspectable rather
than hiding it in metadata. Complex-valued projections retain magnitude/phase structure,
and mesh/field projections preserve the geometric entity to which each value belongs.

### Artifact lineage and scientific presentation

`mathkernel_viz`, `mathkernel_sonify` and `mathkernel_multimodal` share the `mathkernel_artifacts` semantic layer. `MathKernelArtifact` carries typed source lineage, evidence/certificates, presentation transformations, scientific/perceptual annotations, reproducibility metadata and visual/audio synchronization. `mathkernel_viz.visualize(result)` attaches deterministic structured lineage to visual datasets and series, while `mathkernel_viz.to_artifact(doc, result=...)` promotes a visual document into the same evidence-carrying artifact model used by multimodal exports. Presentation remains downstream of mathematics and cannot upgrade source trust.

## Scientific sonification (`mathkernel-sonify`)

`mathkernel_sonify` is the auditory sibling of `mathkernel_viz`. It consumes the same
source lineage and `MultimodalProjection` contract, while `SonificationDocument` owns the
auditory mapping itself. The mathematical result remains untouched.

```python
import mathkernel_projection as mkp
import mathkernel_sonify as son

projection = mkp.create_projection(
    "spectrum",
    {"amplitudes": [1.0, 0.42, 0.17], "phases": [0.0, 0.3, -0.2]},
    trust="numeric",
)
audio = son.projection_sonification(projection)
son.write_wav(audio, "spectrum.wav")
son.export_html(audio, "spectrum.html")
```

The IR records every value-to-audio mapping as declarative provenance. Structured
objects are never silently flattened: matrix scans record row/column ordering; tensor
sonification records the selected slice/order; graphs record traversal or degree
reduction; meshes record the geometric reduction; complex objects preserve magnitude
and phase mapping; optimization traces, bootstrap/null distributions, relation spectra
and ensemble orderings are likewise explicit.

Built-in adapters cover harmonic/Fourier additive synthesis, sequential scans,
prediction-vs-observation stereo comparison, residual sonification and projection-aware
structured mappings. Offline PCM/WAV rendering is deterministic, rejects silent Nyquist
aliasing, and applies explicit normalization/peak limits. The WebAudio exporter is a
single offline HTML file with no network dependency.

**Scientific rule:** an audible pattern is a perceptual candidate, not mathematical
evidence. Any pattern discovered by listening must be validated quantitatively, exactly,
formally or empirically through MathKernel.

## Unified multimodal artifacts (`mathkernel-multimodal`)

`mathkernel_multimodal` combines visualization and sonification derived from the same
source/projection into one portable `MathKernelArtifact`. Shared `SourceRef` ancestry
allows automatic cross-modal synchronization without weakening the mathematical trust
model.

```python
import mathkernel_multimodal as mkm

artifact = mkm.build_artifact(
    title="Result",
    visualizations=[viz_doc],
    sonifications=[son_doc],
    mathkernel_version="current",
)
mkm.export_html(artifact, "result.html")
```

- visual blocks can highlight during linked audio playback and linked audio can seek
  from a visual block;
- one inspector surface exposes Result, Evidence, Provenance, Data, Reproduction,
  Visual Mapping, Audio Mapping, Sync and Annotations;
- payload verification and document integrity hashes remain available in the exported
  artifact;
- portable output works from `file://`, with no running MathKernel server required;
- artifact trust remains the weakest justified source/member trust.

Via MCP, research artifacts can be assembled from stored visualization and sonification
objects and exported as a single self-contained file.

## MCP tool surface

<details>
<summary><b>All 167 tools</b> (click to expand)</summary>

| Group | Tools |
|---|---|
| Discovery | `math_capabilities`, `math_capability_query`, `math_result_resource_get` |
| Typed mathematics | `math_object_create`, `math_object_get`, `math_apply` — complete compositional surface tabulated above, including geometry, signals/control, certified optimization, statistics/stochastic systems and general PDE representation |
| Parsing | `math_parse`, `math_parse_latex`, `math_get`, `math_substitute`, `math_infer_structure` |
| Algebra | `math_simplify`, `math_solve`, `math_solve_system` |
| Calculus | `math_differentiate`, `math_integrate`, `math_limit`, `math_series`, `math_summation`, `math_product` |
| Numeric | `math_numeric_evaluate`, `math_interval_evaluate` |
| Matrices | `math_matrix_create`, `math_matrix_get`, `math_matrix_det`, `math_matrix_inverse`, `math_matrix_transpose`, `math_matrix_multiply`, `math_matrix_rank`, `math_matrix_rref`, `math_matrix_eigenvalues`, `math_matrix_solve` |
| Context | `math_context_create`, `math_context_infer`, `math_context_check` |
| Reasoning | `math_analyze`, `math_plan`, `math_plan_get`, `math_execute_plan`, `math_reason`, `math_execution_get`, `math_prove_equivalence`, `math_counterexample` |
| Codegen | `math_codegen`, `math_verify_code`, `math_execute_code` |
| Integers | `math_integer_analyze`, `math_integer_compute`, `math_integer_batch` |
| Sweeps | `math_collatz_sieve`, `math_cuboid_sweep` |
| Jobs | `math_job_submit`, `math_job_status`, `math_job_result`, `math_job_list` |
| GF(2^m) | `math_gf2m_create`, `math_gf2m_from_transition`, `math_gf2m_compute`, `math_gf2m_coords`, `math_gf2m_root_jump_rows`, `math_gf2m_closure_roots`, `math_gf2m_jump_rows` |
| GF(2) | `math_gf2_rank`, `math_gf2_nullspace`, `math_gf2_carryfree_cols`, `math_gf2_minpoly` |
| Transforms | `math_fwht` |
| Finite dynamics | `math_finite_system_create`, `math_koopman_matrix`, `math_koopman_transfer`, `math_koopman_visibility`, `math_koopman_lagged`, `math_koopman_observed`, `math_koopman_diagnostics`, `math_finite_fourier_compute`, `math_closure_search`, `math_cumulant_compute` |
| Conditioned dynamics | `math_conditioned_access_solve`, `math_conditioned_symmetry_access`, `math_conditioned_access_compose`, `math_conditioned_closure`, `math_symbolic_conditioned_access`, `math_affine_conditioned_access`, `math_gf2_conditioned_access`, `math_gf2_predictive_closure`, `math_synthesize_conditioned_closures`, `math_synthesize_gf2_vector_conditioned_access`, `math_discover_structural_conditioned_closure`, `math_discover_factor_swap_conditioned_closure` |
| Multimodal projections | `math_projection_catalog`, `math_projection_create`, `math_projection_describe` |
| Visualization | `math_visualize`, `math_visualize_dag`, `math_render_koopman`, `math_visualize_projection`, `math_export_artifact` |
| Sonification | `math_sonify`, `math_sonify_compare`, `math_sonification_describe`, `math_sonify_projection`, `math_export_audio` |
| Multimodal artifacts | `math_research_artifact_create`, `math_export_research_artifact` |
| Sets & logic | `math_set_create`, `math_set_op`, `math_set_membership`, `math_quantifier_check`, `math_quantifier_eliminate`, `math_quantifier_eliminate_batch` |
| Polynomials | `math_poly_groebner`, `math_poly_divide`, `math_poly_resultant`, `math_poly_discriminant`, `math_poly_factor`, `math_ideal_membership`, `math_poly_groebner_batch` |
| Probability | `math_prob_rv_create`, `math_prob_expectation`, `math_prob_variance`, `math_prob_covariance`, `math_prob_bayes`, `math_prob_markov_stationary`, `math_prob_markov_hitting_time`, `math_prob_sample`, `math_prob_distribution` |
| Statistics | `math_stats_moments`, `math_stats_order`, `math_stats_regression`, `math_stats_correlation`, `math_stats_ttest`, `math_stats_chi2`, `math_stats_confidence_interval`, `math_stats_batch_moments` |
| Tensors | `math_tensor_create`, `math_tensor_get`, `math_tensor_contract`, `math_tensor_solve` |
| Numerics | `math_root_find`, `math_root_scan`, `math_quadrature` |
| ODE/PDE | `math_ode_solve`, `math_ode_solve_numeric`, `math_ode_ensemble`, `math_pde_heat_1d`, `math_pde_heat_2d`, `math_pde_wave_1d`, `math_pde_advect_1d`, `math_pde_ensemble`, `math_pde_mol_heat` |
| Optimization | `math_optimize_critical_points`, `math_optimize_kkt`, `math_lp_solve`, `math_optimize_minimize`, `math_optimize_multistart` |
| Units | `math_unit_check`, `math_unit_convert`, `math_unit_simplify` |
| Assurance | `math_store_status`, `math_replay`, `math_fuzz_differential`, `math_certified_enclose` |
| Proving | `math_prove`, `math_prove_batch`, `math_prove_replay` |
| Provenance | `math_derivation_get`, `math_derivation_trace` |

</details>

Every tool docstring is written LLM-facing: parameter formats, exact-vs-numeric
semantics, limits, and follow-up hints are documented in-place.

## Configuration

All settings are environment-driven with the `MATHKERNEL_` prefix
(`Settings.from_env()`), introspectable via `math_capabilities`:

| Variable | Default | Purpose |
|---|---|---|
| `MATHKERNEL_MAX_INPUT_LENGTH` | 100000 | parser input cap |
| `MATHKERNEL_MAX_OUTPUT_SIZE_BYTES` | 256000000 | whole-response byte budget; oversized payloads are preserved as integrity-checked resources and returned by receipt |
| `MATHKERNEL_SOLVER_TIMEOUT_SECONDS` | 30 | symbolic operation budget using bounded cancellable subprocess workers |
| `MATHKERNEL_ENABLE_EXECUTION` | **false** | sandboxed codegen execution (opt-in) |
| `MATHKERNEL_YOLO_MODE` | **false** | unlocks `math_yolo_settings` to mutate live `MATHKERNEL_*` settings (typed coerce; default off) |
| `MATHKERNEL_Z3_TIMEOUT_MS` | 10000 | SMT budget (set on every Z3 solver instance) |
| `MATHKERNEL_LEAN_BINARY` / `MATHKERNEL_LEAN_TIMEOUT_SECONDS` | `lean` / 90 | Lean adapter (timeout passed to every `lake env lean` check) |
| `MATHKERNEL_SKIP_LEAN_INSTALL` | unset | skip the default Lean 4 + Mathlib download |
| `MATHKERNEL_LEAN_CACHE` | platform cache | elan + lake workspace root |
| `MATHKERNEL_ENABLE_PARALLEL` / `MATHKERNEL_MAX_WORKERS` | true / cpu_count | process & thread pools |
| `MATHKERNEL_MAX_ITERATIONS` | 10000 | iteration cap for simplex / Nelder-Mead |
| `MATHKERNEL_TOLERANCE` | 1e-12 | numeric convergence tolerance |
| `MATHKERNEL_MAX_ODE_STEPS` | 100000 | RK45 integration step cap |
| `MATHKERNEL_STORE_PATH` | unset | opt-in SQLite persistence for expressions/derivations + `math_replay` |
| `MATHKERNEL_PROVE_PORTFOLIO_SIZE` | 3 | SMT encodings raced per `math_prove` call |
| `MATHKERNEL_MAX_PDE_GRID` | 1000000 | PDE solver grid-cell cap |
| `MATHKERNEL_MAX_PDE_FIELDS` / `MATHKERNEL_MAX_PDE_DIMENSIONS` | 16 / 8 | typed PDE field and independent-variable caps |
| `MATHKERNEL_MAX_PDE_EQUATIONS` / `MATHKERNEL_MAX_PDE_TERMS` | 32 / 1024 | typed PDE system and total-term caps |
| `MATHKERNEL_MAX_PDE_CONDITIONS` | 1024 | total typed boundary/initial-condition cap |
| `MATHKERNEL_MAX_PDE_DERIVATIVE_ORDER` / `MATHKERNEL_MAX_PDE_NONLINEAR_POWER` | 4 / 8 | derivative and represented-power caps |
| `MATHKERNEL_MAX_PDE_WORK` | 2000000 | typed PDE construction/replay work cap |
| `MATHKERNEL_MAX_PDE_SPACES` / `MATHKERNEL_MAX_PDE_SPACE_ORDER` | 64 / 8 | weak-form space-count and regularity-order caps |
| `MATHKERNEL_MAX_PDE_WEAK_TERMS` / `MATHKERNEL_MAX_PDE_IBP_STEPS` | 4096 / 256 | derived integral-term and integration-by-parts caps |
| `MATHKERNEL_MAX_PDE_WEAK_WORK` | 5000000 | weak-form derivation/replay work cap |
| `MATHKERNEL_MAX_FEM_POINTS` / `MATHKERNEL_MAX_FEM_CELLS` | 100000 / 200000 | simplex mesh vertex/cell caps |
| `MATHKERNEL_MAX_FEM_DOFS` | 200000 | finite-element-space DOF cap |
| `MATHKERNEL_MAX_FEM_WORK` | 20000000 | finite-element construction/replay work cap |
| `MATHKERNEL_MAX_FEM_ASSEMBLY_NNZ` / `MATHKERNEL_MAX_FEM_ASSEMBLY_WORK` | 2000000 / 50000000 | sparse-entry and assembly-work caps |
| `MATHKERNEL_MAX_FEM_EXACT_SOLVE_DOFS` / `MATHKERNEL_MAX_FEM_NUMERIC_SOLVE_DOFS` | 256 / 100000 | exact dense-diagnostic and numeric sparse-solve caps |
| `MATHKERNEL_MAX_FEM_ESTIMATOR_WORK` / `MATHKERNEL_MAX_FEM_REFINED_CELLS` | 50000000 / 500000 | residual-indicator replay work and refined-output cell caps |
| `MATHKERNEL_MAX_QE_VARIABLES` | 16 | quantifier-elimination variable cap |
| `MATHKERNEL_MAX_BATCH_JOBS` | 10000 | integer batch cap |
| `MATHKERNEL_MAX_MATRIX_DIM` | 128 | matrix engine cap |
| `MATHKERNEL_MAX_JOBS_RETAINED` | 100 | async job retention |
| `MATHKERNEL_MAX_MATH_OBJECTS` | 10000 | retained typed-object cap |
| `MATHKERNEL_MAX_CONTOUR_VERTICES` | 4096 | contour complexity cap |
| `MATHKERNEL_MAX_JOINT_DIMENSIONS` | 8 | joint-distribution dimension cap |
| `MATHKERNEL_MAX_DISTRIBUTION_COMPONENTS` | 256 | mixture component cap |
| `MATHKERNEL_MAX_SYMBOLIC_SERIES_ORDER` | 128 | Laurent/classification order cap |
| `MATHKERNEL_MAX_ORDER_STATISTIC_SAMPLE_SIZE` | 1024 | symbolic order-statistic sample cap |
| `MATHKERNEL_MAX_GRAPH_VERTICES` / `MATHKERNEL_MAX_GRAPH_EDGES` | 4096 / 65536 | typed graph size caps |
| `MATHKERNEL_MAX_COMBINATORIAL_ITEMS` | 10000 | lazy combinatorial generation cap |
| `MATHKERNEL_MAX_GROUP_ELEMENTS` | 4096 | finite-group enumeration cap |
| `MATHKERNEL_MAX_FIELD_DEGREE` | 64 | GF(p^m) extension-degree cap |
| `MATHKERNEL_MAX_NORMAL_FORM_DIM` | 128 | Smith/Hermite matrix dimension cap |
| `MATHKERNEL_MAX_INVERSE_BRANCHES` | 256 | change-of-variable branch/Jacobian cap |
| `MATHKERNEL_MAX_OBLIGATION_STEPS` | 128 | maximum executable plan obligations |
| `MATHKERNEL_MAX_FWHT_SIZE` | 2²⁰ | FWHT length cap |
| `MATHKERNEL_MAX_FINITE_STATES` | 4096 | finite-system enumeration cap |
| `MATHKERNEL_MAX_CUMULANT_ORDER` | 8 | cumulant/connected-tensor order cap |
| `MATHKERNEL_MAX_CLOSURE_RESULTS` | 10000 | closure-search result cap |
| `MATHKERNEL_MAX_GEOMETRY_DIMENSION` | 8 | manifold/chart dimension cap |
| `MATHKERNEL_MAX_GEOMETRY_RANK` | 6 | dense tensor-field rank cap |
| `MATHKERNEL_MAX_GEOMETRY_POINTS` | 10000 | point/vertex count cap |
| `MATHKERNEL_MAX_GEOMETRY_SIMPLICES` | 100000 | halfspace/triangle count cap |
| `MATHKERNEL_MAX_GEOMETRY_WORK` | 1000000 | preflight symbolic geometry work cap |
| `MATHKERNEL_MAX_TOPOLOGY_DIMENSION` | 16 | maximum finite-complex degree/ambient dimension |
| `MATHKERNEL_MAX_TOPOLOGY_CELLS` | 10000 | total simplicial/cubical/chain-basis cell cap |
| `MATHKERNEL_MAX_TOPOLOGY_MATRIX_ENTRIES` | 1000000 | stored boundary-matrix entry cap |
| `MATHKERNEL_MAX_TOPOLOGY_ENTRY_BITS` | 4096 | integer boundary-entry bit-length cap |
| `MATHKERNEL_MAX_TOPOLOGY_WORK` | 2000000 | exact topology preflight work cap |
| `MATHKERNEL_MAX_STATISTICAL_VARIABLES` | 256 | typed sample column cap |
| `MATHKERNEL_MAX_STATISTICAL_OBSERVATIONS` | 100000 | typed sample row cap |
| `MATHKERNEL_MAX_STATISTICAL_CELLS` | 1000000 | typed sample rectangular cell cap |
| `MATHKERNEL_MAX_STATISTICAL_WORK` | 2000000 | descriptive/covariance preflight work cap |
| `MATHKERNEL_MAX_GLM_PARAMETERS` | 64 | fitted coefficient cap, including the intercept |
| `MATHKERNEL_MAX_GLM_ITERATIONS` | 200 | requested IRLS iteration cap |
| `MATHKERNEL_MAX_GLM_PREDICTION_ROWS` | 100000 | conditional-mean rows per prediction request |
| `MATHKERNEL_MAX_GLM_WORK` | 20000000 | GLM rank/matrix/iteration preflight work cap |
| `MATHKERNEL_MAX_NONPARAMETRIC_GROUPS` | 64 | selected Kruskal–Wallis group cap |
| `MATHKERNEL_MAX_EXACT_RESAMPLING_STATES` | 100000 | complete sign/label/permutation state cap |
| `MATHKERNEL_MAX_RESAMPLES` | 1000000 | Monte Carlo permutation/bootstrap draw cap |
| `MATHKERNEL_MAX_RESAMPLING_BATCH_CELLS` | 1000000 | generated cells per bootstrap batch |
| `MATHKERNEL_MAX_RESAMPLING_WORK` | 20000000 | rank/enumeration/resampling preflight work cap |
| `MATHKERNEL_MAX_SURVIVAL_STRATA` | 64 | distinct survival-stratum cap |
| `MATHKERNEL_MAX_SURVIVAL_TIMELINE_POINTS` | 100000 | selected Kaplan–Meier timeline cap |
| `MATHKERNEL_MAX_COX_PARAMETERS` | 64 | Cox predictor cap |
| `MATHKERNEL_MAX_COX_ITERATIONS` | 200 | requested Cox Newton-iteration cap |
| `MATHKERNEL_MAX_COX_PREDICTION_ROWS` | 100000 | partial-hazard prediction-row cap |
| `MATHKERNEL_MAX_COX_INFORMATION_CONDITION` | 1000000000000 | observed-information condition ceiling |
| `MATHKERNEL_MAX_SURVIVAL_WORK` | 20000000 | survival risk-set/matrix/iteration work cap |
| `MATHKERNEL_MAX_TIME_SERIES_LAG` | 1000 | ACF/PACF/diagnostic lag cap |
| `MATHKERNEL_MAX_TIME_SERIES_DIFFERENCE` | 2 | ARIMA differencing-order cap |
| `MATHKERNEL_MAX_TIME_SERIES_PARAMETERS` | 32 | AR/MA/GARCH dynamic-parameter cap |
| `MATHKERNEL_MAX_TIME_SERIES_ITERATIONS` | 500 | fit-optimizer iteration cap |
| `MATHKERNEL_MAX_TIME_SERIES_FORECAST_STEPS` | 10000 | forecast-horizon cap |
| `MATHKERNEL_MAX_TIME_SERIES_WORK` | 50000000 | analysis/fit/forecast work cap |
| `MATHKERNEL_MAX_STOCHASTIC_STATES` | 256 | CTMC state cap |
| `MATHKERNEL_MAX_STOCHASTIC_TIME_POINTS` | 10000 | finite-dimensional/prediction time cap |
| `MATHKERNEL_MAX_GP_CONDITIONING_POINTS` | 2000 | GP observation cap |
| `MATHKERNEL_MAX_STOCHASTIC_MATRIX_ENTRIES` | 1000000 | covariance/generator workspace cap |
| `MATHKERNEL_MAX_GP_CONDITION_NUMBER` | 1000000000000 | GP conditioning ceiling |
| `MATHKERNEL_MAX_STOCHASTIC_WORK` | 50000000 | factorization/exponential work cap |
| `MATHKERNEL_MAX_SDE_STATE_DIMENSION` | 32 | SDE state dimension cap |
| `MATHKERNEL_MAX_SDE_NOISE_DIMENSION` | 32 | Brownian driver dimension cap |
| `MATHKERNEL_MAX_SDE_STEPS` | 1000000 | simulation/convergence step cap |
| `MATHKERNEL_MAX_SDE_PATHS` | 100000 | simulation path cap |
| `MATHKERNEL_MAX_SDE_SIMULATION_CELLS` | 5000000 | stored-path/random-increment cell cap |
| `MATHKERNEL_MAX_SDE_WORK` | 50000000 | SDE update-work cap |
| `MATHKERNEL_MAX_SDE_QUERY_VALUES` | 20000 | path/terminal values returned per query |

## Repository layout

```text
src/mathkernel/            core library, typed mathematics and kernel facade
src/mathkernel_mcp/        FastMCP server layer and public math_* tools
src/mathkernel_projection/ shared typed multimodal projection layer
src/mathkernel_viz/        visualization IR, viewers and portable renderers
src/mathkernel_sonify/     scientific sonification IR, PCM/WAV and WebAudio
src/mathkernel_artifacts/  shared evidence, lineage and synchronization schema
src/mathkernel_multimodal/ unified visual/audio research-artifact exporter
scripts/                   reproducibility, GPU checks and demonstrations
experiments/               research validation programs and datasets
skills/                    synchronized Python and MCP agent skills
tests/                     core, regression, multimodal and domain test suites
benchmarks/                correctness-gated performance measurements
```

## Skill packages

MathKernel ships two synchronized agent-skill packages: one for direct Python use and
one for MCP clients. They document the same evidence contract, object lifecycle and
mathematical semantics, while adapting examples to their respective interfaces.

The skills cover symbolic/exact work, reasoning and proving, persistence, finite
dynamics, probability/statistics, numerics, tensors/units, performance, visualization,
scientific sonification and the shared multimodal projection workflow. The viz/audio
skills now require projection-first provenance for structured objects and explicit
high-dimensional reduction or acoustic extraction rather than hidden flattening.

## Testing

Run the complete source-tree suite with the optional dependencies required by the domains you want to validate:

```bash
PYTHONPATH=src:. python -m pytest -q
python scripts/gpu_smoke.py
```

The repository degrades unavailable optional engines to `unknown` or `unavailable` rather than fabricating success. FastMCP is required for MCP registration tests, `z3-solver` for SMT/proving/quantifier-elimination tests, and the compatible ANTLR runtime for SymPy LaTeX parsing. Domain-specific test modules and experiment runners can be executed independently when validating a particular mathematical surface.

Coverage includes parser and ambiguity handling, symbolic algebra and calculus, exact integer and finite-field arithmetic, graph algorithms, linear algebra, Numba/CUDA differential paths, asynchronous jobs, code generation and checking, GF(2) and finite Fourier methods, Koopman/finite dynamics, PRNG analysis, typed engineering mathematics, geometry/topology, statistics and stochastic systems, PDE/FEM/adaptivity, evidence propagation, persistence integrity, visualization, sonification, multimodal artifacts and the MCP tool surface.

CI targets supported Python versions with native thread fan-out bounded per worker. Distribution checks build the sdist and wheel, verify metadata, install the wheel in a clean environment, confirm the runtime version and check that vendored offline visualization/multimodal assets are present. Portable exports therefore do not require a CDN after installation.

## Safety boundaries

- No raw user expression ever reaches `sympify()`/`parse_expr()`; restricted grammar,
  unknown functions rejected, ambiguous notation refused with candidates.
- Chunked arbitrary-length integer conversion; big-result output guards; bounded
  automatic number-theory work; obligation step ceilings; dependency/cycle validation.
- Sandboxed code execution is **opt-in** (`MATHKERNEL_ENABLE_EXECUTION=1`), runs in an
  isolated subprocess with a timeout, and is always labeled numeric evidence.
- Lean subprocess invocation uses `shell=False`; optional engines report
  `unknown`/`unavailable` rather than fabricating success.
- External native LP/QP/MILP, conic/QCQP, Riccati/LQG and numerical pole-placement
  candidate searches run in fresh interpreters whose process groups are killed
  on timeout. Requests/results are bounded and BLAS/OpenMP fan-out is capped.
- SQLite persistence checks every JSON payload with SHA-256 before decoding.
  canonical typed records additionally reconcile their declared object type, decoded
  model class, and source-link field before retrieval or execution. Corrupt or
  substituted records fail closed without producing derived objects.

This termination boundary is not a hostile-code sandbox and does not impose an
OS memory quota. Multi-tenant isolation still belongs in an external worker or
sandbox layer.

## License

Copyright © 2026 Maarten Boone.

Released under the [MIT License](LICENSE).
