# Probability and statistics (MCP)

Trust discipline: exact rational cores report `exact`; inference reports
`numeric_high_precision` (never exact); sampling reports `numeric`.

## Probability

```
math_prob_rv_create(values=["0","1","2"], probabilities=["1/4","1/2","1/4"])
math_prob_expectation(rv_id, power=1)      # exact E[X^power]
math_prob_variance(rv_id)                  # exact
math_prob_covariance(x_values, y_values, joint_probabilities)
math_prob_bayes(prior=["1/2","1/2"], likelihood=["9/10","1/5"])
math_prob_sample(rv_id, n=1000, seed=42)   # numeric, reproducible
```

Markov chains: build the transition matrix with `math_matrix_create`
(integer/rational entries only), then:

```
math_prob_markov_stationary(matrix_id)          # exact stationary distribution
math_prob_markov_hitting_time(matrix_id, targets=[0])  # null = never reaches
```

Continuous symbolic distributions use typed objects (`symbolic` trust):

```
normal = math_object_create("Distribution", {
  "family": "normal", "parameters": ["0", "1"], "variable": "x"
})
math_apply(normal.data.object_id, "variance", {})
math_apply(normal.data.object_id, "cdf", {"point": "1"})
math_apply(normal.data.object_id, "verify", {})
```

Families include uniform, normal, lognormal, exponential, gamma, beta,
Cauchy, Student-t, chi-squared, F, Weibull, Pareto, Laplace and logistic.
Operations include PDF/CDF/survival/quantile, moments, MGF/characteristic
function, entropy, truncation, convolution, mixture, divergence and order
statistics.

For multivariate densities, parse the density first, then create a
`JointDistribution` with explicit variables and one support per variable.
Use `marginal`, `condition`/`bayes`, `covariance`, `correlation`, and
`order_statistic`. Conditioning returns a `ConditionalDistribution`.
Check `semantic_status`: a nonexistent moment is not the same as unsupported
or unresolved symbolic convergence.

For every result, inspect `status`, `semantic_status`, `side_conditions`,
`claim_evidence`, and `data.plan.execution.obligations`. Only a result whose
domain-invariant obligation is `verified` may be described as symbolically
verified; same-engine checks do not raise trust above `symbolic`, and decimal
ancestry caps it at `numeric`.

Mixture components, joint dimensions, symbolic order-statistic sample size,
inverse branches, expression size, and solver time are bounded. Non-product
joint supports and undecidable symbolic integrals return explicit
`unsupported` or `unknown` outcomes instead of inferred formulas.

`math_prob_distribution` is retained as a legacy compatibility bridge. Use
`math_object_create` and `math_apply` for continuous-distribution reasoning.

## Statistics

For compositional Phase F work, use the generic typed tools:

```
sample = math_object_create("StatisticalSample", {
  "variables": ["x", "y"],
  "observations": [[1, 2], [2, 4], [3, 8]],
  "sampling_method": "unspecified"
})
math_apply(sample.data.object_id, "describe", {})
math_apply(sample.data.object_id, "covariance", {"normalization": "sample"})
math_apply(sample.data.object_id, "empirical_distribution", {"variable": "x"})
math_apply(sample.data.object_id, "evidence_profile", {})

glm = math_object_create("GeneralizedLinearModel", {
  "sample_id": sample.data.object_id,
  "response": "y", "predictors": ["x"],
  "family": "gaussian", "link": "identity"
})
math_apply(glm.data.object_id, "verify", {})
fitted = math_apply(glm.data.object_id, "fit", {
  "max_iterations": 100, "tolerance": 1e-9
})
math_apply(fitted.data.object_id, "diagnostics", {})
math_apply(fitted.data.object_id, "verify", {})
math_apply(fitted.data.object_id, "predict", {"rows": [[4], [5]]})

ranked = math_object_create("StatisticalSample", {
  "variables": ["group", "value"],
  "observations": [[0, 1], [0, 2], [0, 3], [1, 6], [1, 7], [1, 8]]
})
test = math_apply(ranked.data.object_id, "mann_whitney", {
  "value": "value", "group": "group", "group_a": 0, "group_b": 1,
  "alternative": "two_sided", "method": "auto"
})
math_apply(test.data.object_id, "verify", {})
boot = math_apply(ranked.data.object_id, "bootstrap", {
  "variable": "value", "statistic": "mean",
  "confidence_level": "0.95", "resamples": 10000, "seed": 42
})
math_apply(boot.data.object_id, "verify", {})

survival_sample = math_object_create("StatisticalSample", {
  "variables": ["time", "event"],
  "observations": [[1, 1], [2, 1], [2, 0], [3, 1]]
})
survival = math_object_create("SurvivalDataset", {
  "sample_id": survival_sample.data.object_id,
  "duration": "time", "event": "event"
})
curve = math_apply(survival.data.object_id, "kaplan_meier", {
  "confidence_level": "19/20"
})
math_apply(curve.data.object_id, "survival_at", {"time": 2})
math_apply(curve.data.object_id, "verify", {})
```

Interpret `trust: exact` only as exact arithmetic on the stored sample. It does
not establish representativeness, iid sampling, model validity, causality, or a
population result. Sampling/design metadata is asserted. Inspect the required
computation and diagnostic empirical/model groups in `claim_evidence`; the
population and model scopes must remain `not_established`. Do not silently
drop missing/symbolic values or bypass statistical resource limits.

F.2 GLMs accept only Gaussian/identity, binomial/logit, and Poisson/log.
Exact Gaussian fits can retain exact stored-design arithmetic; IRLS and
decimal-ancestry fits are numeric. A rank, response-domain, convergence,
conditioning, or separation failure must remain a failure without a derived
fit or silent regularization. Treat coefficients, covariance, deviance,
diagnostics, and predictions as conditional on the asserted model—not proof of
model validity, population generalization, or causality.

F.3 operations are `mann_whitney`, `wilcoxon`, `kruskal_wallis`,
`ks_2samp`, `spearman`, `kendall`, `permutation_test`, and `bootstrap`.
Named tests choose bounded complete enumeration or explicitly labelled
asymptotics. Generic permutation supports exact or seeded PCG64 Monte Carlo;
bootstrap produces seeded mean/median percentile intervals. Never invent a
seed, promote an approximation/simulation to exact, or infer exchangeability,
coverage, population validity, or causality. Inspect method, tie correction,
enumeration/draw count, seed/algorithm and evidence. Derived-result `verify`
replays the exact enumeration or random stream under current resource limits.

F.4 `SurvivalDataset` rows require an exact binary event indicator and
`0 <= entry <= duration`; optional delayed entry and strata stay explicit.
Kaplan–Meier point estimates may be exact, but Greenwood/log-log uncertainty is
numerical and conditional on non-informative censoring. Multi-stratum curves
require a selected stratum. `CoxProportionalHazardsModel` fitting is initially
unstratified float64 with explicit `efron` or `breslow` ties; rank/event
sparsity, convergence, conditioning, and separation failures remain failures.
Use derived `CoxPHFit` `verify`, `diagnostics`, and
`predict_partial_hazard`, but do not treat concordance or Schoenfeld
correlations as proof of proportional hazards, or partial hazard as absolute
risk. Censoring/model validity, population inference, and causality remain
unestablished.

For F.5 create `TimeSeriesDataset` with distinct `time`/`value` columns. Row
order is preserved and timestamps must be strictly increasing. Use `acf`,
`pacf`, or `stationarity_test(method="adf")`; create `TimeSeriesModel` with
explicit `family` (`ar`, `ma`, `arma`, `arima`, `garch`) and `p,d,q` for
`verify`/`fit`. Derived `TimeSeriesFit` supports replay, Ljung–Box/Jarque–Bera
`diagnostics`, and analytic Gaussian `forecast`; `TimeSeriesForecast` also
replays. Do not silently sort, impute, difference, regularize, alter orders, or
fit irregular spacing. ADF, residual diagnostics, roots, convergence, and
GARCH constraints do not establish stationarity, correct specification,
forecast coverage, population validity, or causality.

For F.6 create `PoissonProcess`, `WienerProcess`, `GaussianProcess`, or
`ContinuousTimeMarkovChain` through `math_object_create`, then use
`math_apply`. Poisson supports `pmf`, `moments`, and
`increment_distribution`; Wiener supports `finite_dimensional` and
`increment_distribution`; GP supports `finite_dimensional` and `condition`;
CTMC supports `transition_matrix`, `distribution`, and
`stationary_distribution`. Use `verify` for every source and replayable derived
result. GP conditioning and CTMC exponentials are numerical, while configured
Poisson/Wiener formulas and CTMC generator/stationary identities can be exact.
Process laws remain asserted model assumptions. Never silently infer validity,
fit hyperparameters, add jitter, repair a generator, or choose among nonunique
stationary laws. Respect the advertised stochastic resource limits.

For F.7 create a `StochasticDifferentialEquation` with declared state, time and
parameter symbols, drift, full diffusion matrix, concrete initial state/time
interval, and explicit Itô interpretation. `simulate` requires
`scheme=euler_maruyama|milstein`, steps, paths and a uint64 seed. Euler supports
vector/full diffusion; Milstein is scalar-state/scalar-noise only and must not
be silently substituted. Simulation creation returns compact metadata and an
`SDESimulation` ID; use bounded `path` and `terminal_values` queries, then
`verify` for PCG64 replay. `convergence_study` couples `h`, `h/2`, and `h/4`
Brownian paths and returns replayable `SDEConvergenceStudy` diagnostics. All
paths/orders/errors are empirical/numerical, not proof of regularity, model
validity, population scope or causality. Respect SDE resource limits.

F.8 closes this surface. Stored payload integrity and Phase F type/source
ancestry are checked before retrieval or execution. Stop on either failure;
never relabel or reconstruct a rejected result. `math_object_get` returns
compact `SDESimulation` metadata too, so full paths remain accessible only via
bounded `path` and `terminal_values` calls after persistence restart.

The dedicated raw-array tools remain compatibility APIs:

```
math_stats_moments(values, max_order=4)            # exact
math_stats_order(values)                           # exact median/quartiles
math_stats_regression(x_values, y_values)          # exact slope/intercept/R^2
math_stats_correlation(x_values, y_values)         # exact r^2, numeric r
math_stats_ttest(values, mu0="3")                  # numeric_high_precision
math_stats_chi2(observed, expected)                # numeric_high_precision
math_stats_confidence_interval(values, "0.95")     # numeric_high_precision
math_stats_batch_moments(columns, workers=4, numeric=false)  # process pool
```
