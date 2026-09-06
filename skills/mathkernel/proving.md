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
   `omega` (linear integer arithmetic, a decision procedure). Finite decimal
   literals are exact rationals; division by nonzero literals is supported.

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
