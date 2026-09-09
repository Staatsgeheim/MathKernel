# Theorem proving

`kernel.prove(expr_id, context_id=None, formal=True)` is the general
proving interface, beyond the Lean arithmetic fragment.

## Pipeline

1. **Fragment classification** — every result includes
   `data.fragment = {logic: qf|quantified, arithmetic: none|linear|nonlinear|outside,
   sort: int|rational|real, has_functions: bool}`. Read it to understand why
   an engine was or was not attempted.
2. **SMT portfolio** — Z3 encodings race on separate contexts (Z3 contexts
   are not thread-safe; constraints are translated per encoding, only
   `check()` runs concurrently):
   - `skolem` — strips the outer quantifier so witnesses/countermodels
     materialize in the model
   - `direct` — plain solver on the negated goal
   - `qe-light` — quantifier-elimination-light tactic pipeline
   - `lia` — `SolverFor("LIA")` for pure integer linear goals
3. **Lean certificate** — for qf relation goals in the arithmetic fragment
   when SMT says valid. Tactics: `norm_num` (closed), `ring` (polynomial
   identities), `linarith`/`nlinarith` (rational, with assumptions),
   `omega` (linear integer arithmetic, a decision procedure). Division by nonzero
   exact literals is supported. Approximate decimal inputs, assumptions and
   inherited approximate values are refused by the exact/formal proof routes;
   use exact rationals or interval certification.

Lean/Mathlib setup is explicit: `mathkernel-lean-setup` installs the pinned local
toolchain, `--check` probes its health without downloads, and `--repair` stages
a replacement before activation. Budget several GiB. Proof calls, discovery and
MCP startup never install or repair anything. Missing or unhealthy Lean reports
`unavailable` while the other verifiers remain usable.

`prove_equivalence` publishes `lean_certificate` only with a matching verified
Lean proof record. An unchecked `lean_candidate` has `checked: false` and is not
proof evidence. Refuted results contain neither artifact. Declined independent
verifiers are diagnostic; unknown required dependencies still limit trust.

## Trust

- `formal` — Lean certificate checked (status `verified`)
- `exact` — decisive SMT (status `verified` or `refuted`, with countermodel)
- `unknown` — timeout / unsupported fragment / engines unavailable

## Batch, persistence, replay

```python
kernel.prove_batch(expr_ids, workers=None)   # process pool, SMT tier
kernel.prove_replay(cert_id)                 # re-check a stored certificate
```

With `MATHKERNEL_STORE_PATH` set, certificates persist and `prove` returns
`data.certificate_id`; otherwise the script is inlined as `data.certificate`.
`MATHKERNEL_PROVE_PORTFOLIO_SIZE` (default 3) caps the encoding portfolio.

## External projects

`formal_project_audit` and `formal_project_probe` inspect external Lean projects
without running them; neither is a theorem proof. Full external Comparator replay
is operator-only and uses a separate trusted reference. See
[formal-project-audit.md](formal-project-audit.md). Closed exact rational
inequalities also have a solver-free refutation path; variable, conditional and
approximate claims are not silently reduced to that fragment.
