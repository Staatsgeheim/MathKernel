# Persistence, replay, certified Arb, fuzzing

## SQLite store (opt-in)

Set `MATHKERNEL_STORE_PATH=/path/to/mk.db`. With `MATHKERNEL_YOLO_MODE=true`,
`kernel.yolo_settings({"store_path": ...})` / `math_yolo_settings` can change
that path on a live kernel (null/empty closes the store). Expressions and derivation steps
are persisted append-only (idempotent upserts). `kernel.store_status()`
reports state; `kernel.replay(step_id)` reconstructs the provenance chain in
topological order and validates DAG integrity — missing parents or cycles are
hard errors, because a broken store is not a provenance record.

## Dynamic obligations

Executors may discover follow-up obligations mid-DAG
(`ObligationExecution.followups`). Concretely: when a solution candidate
verifies only numerically (symbolic residual inconclusive, |residual| < 1e-20),
the executor spawns an `interval_enclose` obligation that rigorously checks
the candidate with `mpmath.iv` — containment upgrades the claim to
`interval_certified`, exclusion rigorously refutes it. Dynamic obligations are
capped by `max_steps` and scheduled after static ones in discovery order.

## Certified ball arithmetic

`kernel.certified_enclose(expr_id, "x", "0", "1")` uses Arb balls when
`python-flint` is installed (the `certified` extra), falling back to
`mpmath.iv`. Both are `interval_certified`. `capabilities()["certified"]`
reports which engine is active.

## Lean coverage and replay

`kernel.lean.tactic_coverage()` maps each tactic (norm_num / ring / linarith /
nlinarith) to the MathIR fragment it covers. `kernel.lean.replay_certificate(script)`
deterministically re-checks a stored certificate.

## Differential fuzzing

`kernel.fuzz_differential(n, variables=["x"], seed=0, workers=None)` generates
seeded random MathIR over the guarded numeric fragment and compares the
mpmath 40-dps path against the float64 compiled path. Disagreements return
`status="conflict"` with the offending expressions — they are surfaced, never
absorbed. Parallel over the process pool; wired into the test suite.

## Compute journal

The compute pilot uses its own `compute.sqlite3` and private attempt/quarantine
spool. It does not migrate KernelStore. One coordinator owns the directory.
Submission IDs and outbox intent survive restart; an unknown launch is reconciled
without replacement. Closing a client does not cancel its supervisor. Archive the
state directory only after owned work and cleanup are resolved. See
[remote-compute.md](remote-compute.md).
