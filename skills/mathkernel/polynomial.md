# Polynomial algebra

Exact multivariate polynomial machinery (`mathkernel.polynomial`), all
`EXACT` trust — there is deliberately no numeric path.

## Kernel facade

```python
a = kernel.parse("x^2 - y^2").data["expr_id"]
b = kernel.parse("x - y").data["expr_id"]

kernel.poly_groebner([a, b], ["x", "y"], order="lex")
# -> basis (display strings + stored expr_ids)

kernel.poly_divide(a, [b], ["x", "y"])
# -> quotients, remainder (dividend = sum(q_i*d_i) + remainder)

kernel.ideal_membership(a, [b, c], ["x", "y"])
# -> member: true/false via the Groebner remainder test

kernel.poly_resultant(a_id, b_id, "x")
kernel.poly_discriminant(a_id, "x")
kernel.poly_factor(a_id)                       # over ZZ/QQ
kernel.poly_factor(a_id, extension="sqrt(2)")  # algebraic extension

kernel.poly_groebner_batch(jobs, workers=4)    # process pool
# jobs: [{"polys": ["x**2 - y**2", "x - y"], "variables": ["x", "y"]}, ...]
```

## Notes

- Monomial orders: `lex`, `grlex`, `grevlex`, `ilex`, `igrlex`, `igrevlex`.
- Guards: at most 64 polynomials and 16 variables per call; batch size is
  limited by `MATHKERNEL_MAX_BATCH_JOBS`.
- All single-call operations run under `MATHKERNEL_SOLVER_TIMEOUT_SECONDS`
  (abandoned-worker semantics, like the SymPy engine).
- Batches use the persistent process pool (`parallel.process_map`), so
  per-call SymPy work parallelizes across cores; per-job failures are
  isolated and reported in-place (`{"ok": false, "error": ...}`).
- Module-level API (`mathkernel.polynomial.groebner_basis`, ...) takes
  SymPy objects directly and is usable without the facade.
