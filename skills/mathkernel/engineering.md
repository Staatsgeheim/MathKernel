# Engineering workflows

Use typed `object_create` / `apply` with capability discovery for the domains
`signal`, `control`, and `optimization`. Read `ENGINEERING_MATH.md` in the
repository for complete schemas and unsupported cases.

- Signals: `DiscreteSignal` -> `dft` -> `Spectrum` -> `idft`. Retain sample rate
  in Hz and time origin in seconds. `mode="numeric"` explicitly selects FFT;
  exact input alone does not make a numeric FFT exact. `precision` above 53 bits
  selects bounded mpmath DFT. Do not promise GPU execution.
- `convolution`/`correlation` use `other_id`; `Filter.apply_signal` uses
  `signal_id`. Both source objects are required ancestry. Sample rates must match.
- `FilterDesign.design` requires numeric mode and returns an SOS-backed Filter.
  Design specifications and stability remain numerical candidates, even when
  the difference-equation residual is small.
- Filters use coefficients of z^-j; transfer functions use descending powers
  of s or z. Convert with `to_transfer_function` / `to_filter` rather than
  reinterpreting the arrays manually.
- `StateSpaceSystem.stability` tests internal modes. `TransferFunction.stability`
  tests reduced zero-state BIBO behavior. Hidden unstable states can cancel in
  the transfer function; do not infer internal stability from that conversion.
- `OptimizationProblem.solve` supports exact LP search and explicit numeric
  LP/QP/MILP search. A solver's optimal status is not a proof. Inspect
  `data.details.conclusion`, `accepted`, the certificate, and `claim_evidence`.
- Exact rational reconstruction is only a proposed witness. Exact optimality
  requires original-data feasibility, dual feasibility, stationarity,
  complementarity, zero gap, and PSD checks. Decimal originals/witnesses remain
  numerical. Do not turn an unverified MILP bound into an exact optimum.
- Infeasible/unbounded solver reports without checked witnesses are UNKNOWN.
  Failed simplification is not mathematical refutation. A timeout is not
  nonexistence. Optional SciPy absence is UNSUPPORTED.

Sources are immutable. Always retain the derived `data.object_id` for the next
operation. Evidence/provenance and source IDs survive SQLite restart. Use
`mode="numeric"` only where approximate computation meets the actual task.

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
- `lqg(Q,R,W,V)` composes both checked designs and verifies its separation and
  feedthrough-cancellation identities. Exact mode needs both certificate IDs.
  Inspect `output`, `closed_loop`, `lqr_certificate`, and `kalman_certificate`.
- Discrete `finite_lqr(Q,R,terminal,horizon)` returns a derived-only immutable
  policy. Use its `verify`, `control(state,step)`, and `rollout(initial)` methods.
  Exact replay checks stagewise gain, Bellman and positivity identities without
  rerunning synthesis. It makes no asymptotic-stability claim.
- Discrete `kalman_state(mean,covariance,W,V)` returns a prior. Call
  `update(measurement,control)` then `predict()`; the phase order is enforced and
  prediction reuses the control stored by update. States are immutable and can
  branch safely.
- Covariance-equation certification does not validate noise assumptions,
  Gaussianity, or empirical/MMSE performance. MPC has its separate D.6 contract.
- `PHASE_D_COMPLETION.md` fixes the baseline endpoint; it is complete through D.8.

### D.6 constrained MPC

Read `CONSTRAINED_MPC.md` before making MPC claims.

- Discrete `StateSpaceSystem.mpc(Q,R,terminal,initial,horizon,...)` transcribes
  the explicit trajectory QP. Numeric mode searches; exact mode requires an
  `OptimizationCertificate` ID and never invokes a candidate solver.
- Inspect named outputs: `output` is an immutable `MPCPlan`, `problem` is the
  exact generated QP, `certificate` is its witness, and `terminal_controller`
  exists when `terminal_gain` is supplied. Invalid exact witnesses create no
  plan. Certified Farkas infeasibility returns no plan.
- Read `feasibility_certified`, `optimality_certified`, and `terminal_analysis`
  independently. Numeric feasibility checks remain candidates. Solver status
  never proves feasibility, optimality, or infeasibility.
- `MPCPlan.verify()` is solver-free. `first_control()` rechecks the stored plan
  and explicitly requires reoptimization from the next measured state.
- Exact terminal box invariance plus terminal-control admissibility can certify
  recursive feasibility. Terminal-feedback stability is a separate Schur claim;
  D.6 does not claim whole receding-horizon closed-loop stability or robustness.
- Whole-loop MPC stability and robustness remain unclaimed at the completed D.8 endpoint.

### D.7 representations and analysis

- `ContinuousSignal.sample` is bounded uniform evaluation. It records its grid
  and never implies bandlimiting or reconstruction accuracy.
- `ZeroPoleGain` conversion preserves factor multiplicity without silent
  cancellation. State-space poles are internal modes; exact state-space zeros
  are currently SISO transfer zeros.
- `frequency_response`, `bode`, `nyquist`, and `root_locus` require explicit
  numeric mode. Nyquist returns a curve without an encirclement/stability claim;
  root-locus branch ordering is heuristic even though every root residual is checked.
- Typed `FrequencyResponse`, `RootLocus`, and `TimeResponse` outputs preserve
  grids, units, conventions and timing. `DiscreteControlSystem` is an explicit
  sampled state-space representation.
- `fir_window` adds linear-phase windowed FIR design. FIR symmetry and IIR
  SOS/direct-form agreement are cross-checks, not certified bandwise specs.
- Read `CONTROL_ANALYSIS.md` for analysis semantics.

### D.8 hardening and audit

- External native LP/QP/MILP, conic/QCQP, Riccati/LQG and numerical pole-placement
  candidate searches run in a fresh interpreter process group. A timeout kills
  that group; the request and result transports are bounded and native thread
  fan-out is capped.
- This boundary improves termination and failure containment; it is not a
  hostile-code sandbox or an OS memory quota. Keep multi-tenant isolation outside
  MathKernel.
- Exact certificate replay is solver-free. Native status, residuals and simulated
  behavior remain numeric evidence unless an independent original-data checker
  accepts a witness.
- Capability output types, units, sampling metadata, immutable ancestry and
  SQLite restart behavior are covered by the phase-wide audit. Read
  `PHASE_D_AUDIT.md` before making Phase D coverage claims.
