# Probability and statistics

Exact-first: rational pmfs and samples compute with `Fraction` — results are
`EXACT`. Inference (t/chi-square/CIs) is mpmath-based and reports
`NUMERIC_HIGH_PRECISION`, never `EXACT`. Sampling is `NUMERIC`.

## Probability

```python
rv = kernel.prob_rv_create(["0", "1", "2"], ["1/4", "1/2", "1/4"])
kernel.prob_expectation(rv.data["rv_id"])        # "1"
kernel.prob_expectation(rv.data["rv_id"], 2)     # raw moment E[X^2]
kernel.prob_variance(rv.data["rv_id"])           # "1/2"
kernel.prob_covariance(x_values, y_values, joint_probabilities)
kernel.prob_bayes(["1/2", "1/2"], ["9/10", "1/5"])   # -> ["9/11", "2/11"]
kernel.prob_sample(rv_id, n=1000, seed=42)       # numeric, reproducible
```

Markov chains reuse exact kernel matrices (integer/rational entries only):

```python
m = kernel.matrix_create([["1/2", "1/2"], ["1/4", "3/4"]])
kernel.prob_markov_stationary(m.data["matrix_id"])      # ["1/3", "2/3"]
kernel.prob_markov_hitting_time(m.data["matrix_id"], [0])  # ["0", "4"]
```

Continuous distributions use typed objects so support, constraints, evidence,
and mathematical nonexistence remain attached to each claim:

```python
normal = kernel.object_create("Distribution", {
    "family": "normal", "parameters": ["0", "1"], "variable": "x",
})
oid = normal.data["object_id"]
kernel.apply(oid, "verify")
kernel.apply(oid, "variance")
kernel.apply(oid, "cdf", {"point": "1"})
```

Inspect `status`, `semantic_status`, `side_conditions`, `claim_evidence`, and
the four-node executable plan before describing a result as verified. A
symbolically verified claim remains `VERIFIED_SYMBOLIC`; decimal ancestry
caps trust at `NUMERIC`. `DOES_NOT_EXIST`, `UNSUPPORTED`, and unresolved
`UNKNOWN` outcomes are distinct.

Joint and conditional distributions support marginals, conditioning/Bayes,
covariance, correlation, and order statistics. Mixture size, joint dimension,
order-statistic sample size, inverse branches, symbolic integration time, and
expression size are bounded by kernel settings.

`prob_distribution` remains available only as a legacy compatibility bridge
and does not provide the typed claim-specific workflow.

## Statistics

Prefer the Phase F typed boundary for compositional sample and model work:

```python
sample = kernel.object_create("StatisticalSample", {
    "variables": ["x", "y"],
    "observations": [[1, 2], [2, 4], [3, 8]],
    "sampling_method": "unspecified",
})
sid = sample.data["object_id"]
kernel.apply(sid, "describe")
kernel.apply(sid, "covariance", {"normalization": "sample"})
kernel.apply(sid, "empirical_distribution", {"variable": "x"})
kernel.apply(sid, "evidence_profile")

glm = kernel.object_create("GeneralizedLinearModel", {
    "sample_id": sid, "response": "y", "predictors": ["x"],
    "family": "gaussian", "link": "identity",
})
gid = glm.data["object_id"]
kernel.apply(gid, "verify")
fitted = kernel.apply(gid, "fit", {"max_iterations": 100, "tolerance": 1e-9})
fid = fitted.data["object_id"]
kernel.apply(fid, "diagnostics")
kernel.apply(fid, "verify")
kernel.apply(fid, "predict", {"rows": [[4], [5]]})

ranked = kernel.object_create("StatisticalSample", {
    "variables": ["group", "value"],
    "observations": [[0, 1], [0, 2], [0, 3], [1, 6], [1, 7], [1, 8]],
})
rid = ranked.data["object_id"]
test = kernel.apply(rid, "mann_whitney", {
    "value": "value", "group": "group", "group_a": 0, "group_b": 1,
    "alternative": "two_sided", "method": "auto",
})
kernel.apply(test.data["object_id"], "verify")
boot = kernel.apply(rid, "bootstrap", {
    "variable": "value", "statistic": "mean",
    "confidence_level": "0.95", "resamples": 10000, "seed": 42,
})
kernel.apply(boot.data["object_id"], "verify")

survival_sample = kernel.object_create("StatisticalSample", {
    "variables": ["time", "event"],
    "observations": [[1, 1], [2, 1], [2, 0], [3, 1]],
})
survival = kernel.object_create("SurvivalDataset", {
    "sample_id": survival_sample.data["object_id"],
    "duration": "time", "event": "event",
})
curve = kernel.apply(survival.data["object_id"], "kaplan_meier",
                     {"confidence_level": "19/20"})
kernel.apply(curve.data["object_id"], "survival_at", {"time": 2})
kernel.apply(curve.data["object_id"], "verify")
```

An exact typed result proves only the statistic of the stored observations.
Sampling-method/population/design metadata is asserted, not verified.
`claim_evidence` keeps required computation separate from diagnostic empirical
and model records, and results explicitly mark population generalization and
model validity `not_established`. Decimal observations remain numeric; missing
or symbolic values are refused. Respect the statistical variable, observation,
cell, and work limits rather than silently filtering or retrying oversized data.

GLMs accept only Gaussian/identity, binomial/logit, and Poisson/log in F.2.
Exact Gaussian fits may be exact computations on the stored design; logistic,
Poisson, and decimal-ancestry fits are checked float64 numerical results.
Reject rank deficiency, invalid/degenerate responses, non-convergence,
ill-conditioned information, or detected separation. Do not add implicit
regularization. Coefficients, covariance, deviance, convergence and predictions
remain conditional on the asserted model; they do not prove validity,
population generalization, or causality.

F.3 sample operations are `mann_whitney`, `wilcoxon`, `kruskal_wallis`,
`ks_2samp`, `spearman`, `kendall`, `permutation_test`, and `bootstrap`.
Named tests accept `method=auto|exact|asymptotic`; `auto` enumerates the
complete conditional null only within exact-state and work budgets, otherwise
it records the named approximation. Generic permutation supports mean/median
differences and exact or seeded PCG64 Monte Carlo. Bootstrap supports seeded
mean/median percentile intervals. Never invent a seed or call an asymptotic or
simulated p-value exact. Inspect tie corrections, enumeration/draw counts,
method, seed/algorithm, and `claim_evidence`; exact verification remains
conditional on the stored observations and does not validate exchangeability,
coverage, population inference, or causality. Result `verify` replays the
exact enumeration or recorded PCG64 stream and is resource-bounded.

F.4 survival data require exact binary event indicators and
`0 <= entry <= duration`; right censoring, optional delayed entry, and strata
are first-class. Kaplan–Meier point values may be exact on exact data, while
Greenwood/log-log uncertainty remains numerical and conditional on
non-informative censoring. Select an explicit stratum rather than pooling a
multi-stratum dataset. Cox fitting is initially unstratified and float64, with
explicit `efron` or `breslow` ties. Refuse rank/event sparsity, convergence,
conditioning, or separation failures. Treat Schoenfeld correlations and
concordance as diagnostics, not proof of proportional hazards; partial-hazard
prediction does not apply an absolute baseline risk. Independent censoring,
proportional hazards, population validity, and causality remain unestablished.

F.5 time-series work starts with a `TimeSeriesDataset` linked to a sample's
distinct time/value columns. Stored row order is temporal order; timestamps
must increase strictly and are never silently sorted. Use `acf`, `pacf`, and
`stationarity_test(method="adf")` for analysis. Create `TimeSeriesModel` with
an explicit `family` (`ar`, `ma`, `arma`, `arima`, or `garch`) and `p,d,q`,
then `verify` before `fit`; derived fits support `verify`, `diagnostics`, and
`forecast`. Irregular series remain analyzable but not fit-able. Treat ADF,
Ljung–Box/Jarque–Bera, root and variance checks as diagnostics/conditional
checks—not proof of stationarity, innovation assumptions, specification,
coverage, population validity, or causality. Never silently sort, impute,
difference, regularize, change orders, or substitute a model family.

F.6 process-law sources are `PoissonProcess`, `WienerProcess`,
`GaussianProcess`, and `ContinuousTimeMarkovChain`. Use Poisson `pmf`/
`moments`/`increment_distribution`; Wiener `finite_dimensional`/
`increment_distribution`; GP `finite_dimensional`/`condition`; and CTMC
`transition_matrix`/`distribution`/`stationary_distribution`. Always run
`verify` when generator or parameter validity matters. Derived finite laws, GP
posteriors, and CTMC transitions are output-only and replayable. GP conditioning
and CTMC exponentials are numerical; Poisson/Wiener formulas and CTMC
generator/stationary identities can be exact. Treat continuity, Gaussianity,
independent increments, kernel suitability and time homogeneity as asserted
model assumptions—not evidence that data follow the process. Do not silently
fit GP hyperparameters, add jitter, repair a generator, or select one stationary
law when it is nonunique; respect the stochastic state/time/matrix/work and GP
conditioning limits.

F.7 SDE work uses `StochasticDifferentialEquation` with declared state/time/
parameter symbols, drift vector, full state-by-noise diffusion matrix, concrete
initial state and interval, and explicit Itô interpretation. Use
`euler_maruyama` for vector/full-diffusion systems; `milstein` is supported only
for scalar state/scalar noise and must fail rather than downgrade elsewhere.
Every `simulate` call requires steps, paths and a uint64 seed. Treat PCG64 paths,
terminal moments and coupled `convergence_study` results as empirical/numerical,
not proof of existence, uniqueness, nominal order, model validity or population
scope. Derived `SDESimulation` and `SDEConvergenceStudy` replay their streams.
Use paginated `path` and `terminal_values` instead of expecting simulation
creation to echo large arrays, and respect SDE dimension/step/path/cell/work/
query limits.

F.8 closes this surface. Persisted payloads are integrity-checked before decode,
and Phase F records reconcile their stored type, decoded class, and internal
source ID before use. Treat an integrity/source failure as a hard stop; do not
recreate, relabel, or bypass the record. `object_get` is also compact for
`SDESimulation`, so use the bounded path queries after retrieval or restart.

The direct methods below remain compatibility APIs for raw arrays:

```python
kernel.stats_moments(values, max_order=4)   # exact mean/var/central moments
kernel.stats_order(values)                  # sorted, median, quartiles (exact)
kernel.stats_regression(xs, ys)             # exact slope/intercept/R^2
kernel.stats_correlation(xs, ys)            # exact r^2; r numeric -> NHP trust
kernel.stats_ttest(values, mu0="3")         # numeric_high_precision
kernel.stats_chi2(observed, expected=None)  # numeric_high_precision
kernel.stats_confidence_interval(values, "0.95")
kernel.stats_batch_moments(columns, workers=4, numeric=False)  # process pool
```

`numeric=True` in batch moments takes the vectorized float64 NumPy path
(`NUMERIC`); the default is exact `Fraction` moments per column.
