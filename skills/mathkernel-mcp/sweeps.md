# Sub-skill: Sweeps, batch integer work, and async jobs

Part of the `mathkernel-mcp` skill. Load when the task is exhaustive
search (Collatz, cuboid), bulk integer arithmetic, or any long-running
computation.

## Collatz sieve

`math_collatz_sieve(n_max, x_min="1", workers, engine="auto")` —
exhaustively refutes/verifies all cycle classes with `n_min <= n <= n_max`
odd elements. `n_max` is a **cycle-length bound** (small: tens), not a
numeric range. Engines: `auto` | `numba` | `cuda` | `python`.
`x_min > 1` adds a side condition: refutation then relies on an external
verification floor, reported in `side_conditions`.

## Cuboid sweep

`math_cuboid_sweep(bound, engine="auto", workers)` — Pythagorean leg-pair
sweep for Euler bricks / perfect cuboid up to `bound`. Engines:
`auto` | `numba` | `cuda` | `python`. QR prefilter included.

## Integer engine

- `math_integer_analyze(value, factor_limit)` — primality, factorization
  (bounded), digit/structure summary for one big integer.
- `math_integer_compute(operation, values, modulus)` — exact bigint ops
  (add/mul/pow/gcd/modinv/...) and number theory (is_prime, factor,
  affine_jump for LCG-style recurrences `s' = a*s + c`).
- `math_integer_batch(jobs, workers)` — many independent jobs in one call;
  runs on a persistent process pool. Always prefer batch over loops of
  single computes.

## Async job API

For anything that may take minutes:

1. `math_job_submit(kind, params)` → `job_id`. Kinds are listed under
   `jobs.kinds` in `math_capabilities` (includes collatz and cuboid).
2. `math_job_status(job_id)` — poll; `math_job_list(status)` enumerates.
3. `math_job_result(job_id)` — fetch the final MathResult.

Do not block on large direct calls; submit a job and poll with
exponential backoff. Retained job count is limited
(`jobs.max_retained`).

## Parallelism notes

- `workers=null` resolves to a sane default capped by
  `max_workers`; pass an explicit value to control it.
- `engine="auto"` picks CUDA when a working GPU stack is present, else
  numba, else Python — the chosen engine is reported in the result.
