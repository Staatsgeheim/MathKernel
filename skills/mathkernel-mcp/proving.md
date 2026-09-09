# Theorem proving (MCP)

```
math_prove(expr_id, context_id=null, formal=true)
math_prove_batch(expr_ids, context_id=null, workers=null)
math_prove_replay(cert_id)
```

`math_prove` classifies the fragment, races a portfolio of Z3 encodings
(`skolem` / `direct` / `qe-light` / `lia`), and emits a Lean certificate for
supported relation goals only after a successful Lean check. Lean 4 + Mathlib
requires explicit operator setup with `mathkernel-lean-setup`; MCP startup,
discovery and proof calls never download it. Use `mathkernel-lean-setup --check`
for local health and `--repair` for a staged replacement of a broken managed
installation. Setup requires several GiB and preserves the active generation
until its replacement passes a proof check. Certificates are checked with the
pinned local `lake env lean`. `MATHKERNEL_SKIP_LEAN_INSTALL=1` also blocks explicit
setup unless forced; `0` does not enable automatic installation.

Approximate inputs, assumptions and inherited approximate values are refused by
the exact/formal proof routes. Use exact rationals or interval certification.
For `math_prove_equivalence`, only checked scripts appear in `lean_certificate`;
`lean_candidate` with `checked: false` is an unverified attempt. Refuted results
contain neither. Declined independent verifiers remain diagnostic evidence.

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
