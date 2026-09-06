# Theorem proving (MCP)

```
math_prove(expr_id, context_id=null, formal=true)
math_prove_batch(expr_ids, context_id=null, workers=null)
math_prove_replay(cert_id)
```

`math_prove` classifies the fragment, races a portfolio of Z3 encodings
(`skolem` / `direct` / `qe-light` / `lia`), and emits a Lean certificate for
supported relation goals. Lean 4 + Mathlib is installed by default on
first MCP start, first formal check, or `mathkernel-lean-setup` using
the packaged lake workspace. Certificates are checked with
`lake env lean`, not a bare `lean` on PATH. Set
`MATHKERNEL_SKIP_LEAN_INSTALL=1` only when the toolchain must stay absent.

Reading results:

- `status: verified` + `trust: formal` — Lean certificate checked. Strongest.
- `status: verified` + `trust: exact` — SMT proved validity.
- `status: refuted` + `trust: exact` — SMT found a countermodel
  (`data.smt.countermodel`).
- `status: unknown` — timeout or outside the decidable fragment. Check
  `data.fragment` to see why; do not treat as evidence either way.

With `MATHKERNEL_STORE_PATH` set, proved certificates return
`data.certificate_id`; re-check later with `math_prove_replay` (formal trust
on successful replay). `math_prove_batch` runs the SMT tier across the
process pool.
