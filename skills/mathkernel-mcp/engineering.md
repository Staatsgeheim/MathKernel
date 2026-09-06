# Engineering MCP workflows

Discover `signal`, `control`, or `optimization` using `math_capability_query`.
Create objects with `math_object_create`; execute with `math_apply`. Do not invent
specialized MCP tools for these domains. The repository's `ENGINEERING_MATH.md`
documents complete schemas and remaining scope.

| Goal | Object and operation |
|---|---|
| Fourier analysis | `DiscreteSignal.dft`, then `Spectrum.idft` |
| Filter a signal | `Filter.apply_signal` with `signal_id` |
| Design a filter | `FilterDesign.design`, explicit numeric mode |
| Control conversion | `TransferFunction.to_state_space` / `StateSpaceSystem.to_transfer_function` |
| Stability | `stability`, preserving internal versus BIBO scope |
| Optimize | `OptimizationProblem.solve`, inspect `data.details.conclusion` |
| Check a witness | `OptimizationProblem.verify_certificate`, with `certificate_id` or inline `certificate` |

Use sample rates in Hz, start/sample time in seconds, control frequencies in
rad/s. Filter arrays use lag coefficients; transfer-function arrays use descending
powers. Preserve this distinction when composing objects.

Exact arithmetic is the default. Numeric FFT/STFT, resampling, filter design,
frequency response and numerical optimization require `mode="numeric"`.
High-precision DFT uses `precision` in bits; no new GPU backend is advertised.
A successful numeric method does not yield an exact result unless a separate
certificate proves the original exact problem. A decimal problem or decimal
witness cannot be upgraded by rationalizing its display.

`certified_global_optimum` means the original rational LP/convex QP passed exact
KKT, primal/dual gap and PSD checks. An MILP solver bound is explicitly unverified.
Infeasible/unbounded solver reports without certificates remain UNKNOWN. A small
residual is not a rigorous error bound. Filter-design specifications remain
candidates until separately certified. A stable reduced transfer function need
not describe an internally stable state-space realization.

Sources remain immutable. Use each derived `data.object_id` for subsequent
operations and keep every required operand's ancestry. Return `side_conditions`,
uncertainty and claim scope when communicating results; sensory output cannot
strengthen evidence.

## Phase D.2 continuations

- Streaming: call `Filter.initial_state(start, unit)` once. Feed a contiguous
  `DiscreteSignal` to `FilterState.process(signal_id)`. Retain both
  `data.object_ids["output"]` and `data.object_ids["state"]`; the latter is the
  continuation for the next chunk. Rates and amplitude units must match. Use
  rational seconds for exact chunk boundaries. SOS states require numeric mode.
- MIMO: `StateSpaceSystem` allows n×m B and p×n C. Conversion returns a
  `TransferMatrix` for multiple channels. Select `entry(output, input)` with
  zero-based indices. Numerical frequency response returns frequency×output×input.
- Declare `state_units`, `input_units`, `output_units` for coefficient-unit
  inference. `coefficient_units` derives required dimensions/scales; it does
  not validate supplied coefficient annotations or convert coordinate magnitudes.
- `discretize(sample_time, method)` accepts `zoh` and `bilinear`. ZOH assumes
  piecewise-constant inputs; bilinear is a transfer substitution. Numeric residual
  checks and symbolic exponential identities are not interval trajectory bounds.
- `state_feedback(gain)` uses u=r−Kx and preserves C−DK feedthrough. `place_poles`
  and `observer` independently check the requested characteristic polynomial.
  Exact mode handles one control input / one measured output; numeric mode
  supports MIMO. The observer returns `output` and `error` IDs. Query stability
  separately; a successful placement operation does not assert stability.
- Infeasibility certificates use `kind="infeasible"` with canonical inequality
  and equality multipliers (Farkas). Unboundedness certificates use
  `kind="unbounded"`, feasible `primal`, and `ray`. Exact original-data checks
  are required, including integer anchor/step coordinates for integer problems.
- `OptimizationProblem.certify_milp(max_nodes)` constructs a bounded integer
  partition tree. Replay with `verify_milp_certificate(certificate_id)` without
  invoking a solver. Every branch must close with a checked infeasibility witness
  or a checked relaxation bound against a feasible incumbent. An open/budget-limited
  tree is a candidate, even if it contains a good incumbent. `solve(mode="numeric")`
  still exposes a numerical MILP candidate and unverified solver bound.

Do not construct FilterState or TransferMatrix via `object_create`; they are
validated derived objects. All outputs preserve required ancestry through
SQLite restart. See `HANDOVER_PHASE_D3.md` for the current tested slice and next work.

## Phase D.3 conic and quadratic optimization

Read `CONIC_QUADRATIC_OPTIMIZATION.md` in the repository for layouts/examples.

- `ConicProblem` means b−Ax in a product of `nonnegative`, `second_order` and
  `psd` cones, plus optional equalities. Variables are free. Do not assume the
  default nonnegative bounds used by `OptimizationProblem`; use `to_conic` on
  that object when converting its linear constraints and bounds.
- PSD blocks contain dimension² full row-major entries. Constant and every
  coefficient matrix must be symmetric. Duals use the same full trace-inner-
  product layout, not the solver's packed triangle or sqrt(2) coordinates.
- `ConicCertificate.kind` selects `optimal`, `bound`, `infeasible`, or `unbounded`.
  A certified bound need not be attained. An unboundedness witness needs a
  feasible anchor as well as an improving ray. Weak SDP infeasibility may lack
  a strict Farkas witness; no accepted witness means no infeasibility claim.
- `QuadraticallyConstrainedProblem` / `QCQP` adds constraints
  xᵀQi x/2 + aiᵀx + ri ≤ 0. `QuadraticCertificate` uses primal, canonical linear
  duals, equality duals and one multiplier per quadratic. Exact PSD of the full
  Lagrangian Hessian is essential; stationarity alone never proves a minimum.
  The sufficient global test can certify some nonconvex problems but is not a
  general nonconvex optimizer.
- New `solve` operations require explicit numeric mode. Conic search needs the
  `conic` extra; QCQP needs `sci`. Exact verification needs neither solver.
  Returned `data.object_id` references a stored certificate. Replay against the
  original problem with `verify_certificate(certificate_id)`.
- Inspect `accepted`, `conclusion`, and claim evidence. Irrational optima and
  unsuccessful rational reconstruction remain candidates. Decimal ancestry
  survives even when a witness expression simplifies to an integer or zero;
  it must not set the exact certificate acceptance flag.
- Discovery using `input_type` finds operations an object accepts; `object_type`
  also includes operations producing it. Certificate enum labels and tree
  indices are metadata; mathematical witness values remain parsed MathIR.

## Phases D.4–D.8 control synthesis, analysis, and hardening

Read `CONTROL_SYNTHESIS.md` for equations, timing, and claim scope.

- `StateSpaceSystem.lqr(Q, R)` and `kalman(W, V)` require numeric mode for
  automatic CARE/DARE search. For exact replay, supply `certificate_id` or
  `certificate={"P": ...}`. `RiccatiCertificate` is a stored immutable witness.
- Use `data.object_ids["certificate"]` for replay. `output` is the LQR closed
  loop or estimator; Kalman also returns `error`. A failed exact witness does
  not produce a controller. Source and certificate ancestry remain attached.
- Exact acceptance requires PSD weights/P, PD input/measurement weights,
  Riccati and gain equations, and independent strict internal stability.
  Numerical residuals and spectral margins never establish exact acceptance.
- LQR optimality is scoped to controls with terminal x.T*P*x tending to zero.
  Do not infer unrestricted finite-cost optimality when Q is singular.
- Discrete Kalman P is prior prediction covariance; L=A*P*C.T*(V+C*P*C.T)^-1.
  Preserve the A factor and B-LD correction. Continuous noise matrices are
  spectral intensities; discrete matrices are per-sample covariances.
- `StateSpaceSystem.lqg` verifies separation/feedthrough identities; exact mode
  requires both `lqr_certificate_id` and `kalman_certificate_id`.
- Discrete `finite_lqr` creates an immutable policy exposing `verify`, `control`
  and `rollout`. Discrete `kalman_state` creates an immutable prior exposing the
  enforced sequence `update(measurement,control)` then `predict()`.
- Covariance checks do not validate noise assumptions, Gaussianity or empirical
  performance. The fixed Phase D endpoint is complete through D.8.

### D.6 constrained MPC

- `StateSpaceSystem.mpc` is discrete and requires Q/R/terminal, initial state,
  positive horizon and at least one finite box constraint. Numeric mode searches;
  exact mode replays an `OptimizationCertificate` ID.
- `output` is an immutable `MPCPlan`; `problem` is the generated original QP.
  Inspect feasibility, optimality, terminal invariance, recursive feasibility
  and terminal-feedback stability as separate claims.
- `MPCPlan.verify` and `first_control` are solver-free. First control is only the
  first move of this plan; re-solve from the next measured state.
- An exact invariant/admissible terminal box may certify recursive feasibility.
  Whole-loop stability and robustness are not claimed at the D.8 endpoint.

### D.7 representations and analysis

- `ContinuousSignal.sample` evaluates a bounded uniform grid and makes no
  bandlimit or reconstruction claim.
- `ZeroPoleGain` preserves multiplicity and does not silently cancel factors.
  `DiscreteControlSystem` explicitly represents sampled state-space dynamics.
- Poles/zeros have internal-versus-transfer scope. Bode/Nyquist/root-locus are
  explicit numeric operations with typed outputs. Nyquist has no encirclement or
  stability claim; root-locus branch ordering is heuristic.
- Typed time responses record causal/zero-state and unit conventions.
  `fir_window` adds bounded windowed linear-phase FIR design; specifications
  remain candidates. Read `CONTROL_ANALYSIS.md` for their semantics.

### D.8 hardening and audit

- Numeric native candidate searches run in fresh interpreter process groups.
  Timeouts hard-kill the group; input/output transport and native thread fan-out
  are bounded. Surface timeout or dependency failures as structured MCP results.
- Do not describe this as a hostile-code sandbox or memory quota. It is a
  termination and failure-containment boundary for trusted MathKernel workers.
- Exact replay never requires the candidate subprocess. Solver status and small
  residuals remain nonrigorous until an independent original-data checker accepts
  a certificate.
- `PHASE_D_AUDIT.md` records the completed capability, resource, persistence,
  ancestry and claim audit. Later optional engineering families are not implied.
