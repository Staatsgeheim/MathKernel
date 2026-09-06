# Sub-skill: Performance — numba, CUDA, parallelism

Part of the `mathkernel` skill. Load when optimizing, benchmarking, or
troubleshooting engine selection and GPU/parallel behavior.

## Engine map

| Workload | Module | CPU fast path | GPU path | Parallel |
|---|---|---|---|---|
| Collatz sieve | `collatz` | `collatz_fast` (njit, n<=31) | `collatz_gpu` (RawKernel) | process pool |
| Cuboid sweep | `cuboid` | njit leg-pair scan | CuPy RawKernel | process pool |
| GF(2^m) arithmetic | `gf2m` | `gf2m_fast` n-limb njit (m<=1024) | — | — |
| Integer batch | `integers` | `integers_fast` (njit) | — | persistent process pool |
| Graph BFS/components | `graph_theory` | `graph_fast` CSR njit, certificate re-verified | — | — |
| GF(p^m) multiplication | `finite_algebra` | `finite_algebra_fast` uint64 njit (p<2^24, m<=64) | — | — |
| Cayley validation | `finite_groups` | `finite_groups_fast` njit axiom scan | — | — |
| Recurrence extension | `combinatorics` | `combinatorics_fast` checked int64 njit + bigint fallback | — | — |
| FWHT | `transforms` | int64 njit + bigint fallback | — | — |
| Closure search | `relations` | njit MITM (int64/uint64) | — | — |
| Koopman/finite dyn. | `koopman` | numpy complex128 | CuPy (matmul) | — |
| Obligation DAG | `execution` | — | — | thread waves |
| Long sweeps | `jobs` | — | — | async job pool |

Exact symbolic types (`Fraction`, `CyclotomicNumber`) cannot be JIT-compiled
by design — exactness is the product. Numeric twins exist where scale
demands it and always downgrade `trust` to `numeric`.

For new domains, add the fast tier in the same change as the mathematical
semantics: define the exactness fragment, provide automatic Python fallback,
re-verify or differential-test compiled output, and record the backend in
evidence metadata. GPU is required only for regular device-exact workloads;
otherwise document the considered Numba/process tiers.

## Engine selection

Facade sweep methods take `engine="auto" | "numba" | "cuda" | "python"`.
`auto` prefers CUDA when the probe passes, then numba, then Python; the
chosen engine is reported in the result. Koopman methods take
`exact=True|False` instead; `exact=False` uses CuPy when usable, else
numpy.

## Settings (env-driven, `MATHKERNEL_*`)

- `enable_parallel` (default on), `max_workers` — process/thread caps.
- `max_finite_states`, `max_cumulant_order`, `max_closure_results` —
  finite-dynamics limits.
- `max_fwht_size`, `max_matrix_dim`, `max_jobs_retained`.
- Inspect at runtime: `kernel.capabilities()` / `Settings().limits()`.

## GPU requirements and troubleshooting

CuPy wheels ship **no CUDA libraries**. A working GPU stack needs
`cupy-cuda12x` **plus** the `nvidia-*-cu12` pip packages (runtime, cublas,
cufft, curand, cusolver, cusparse, nvrtc, nvjitlink) — all installed by
`pip install 'mathkernel-mcp[cuda]'`. A system CUDA toolkit only helps if
its major version matches (CUDA 12.x for cupy-cuda12x) and it actually
contains `bin/` DLLs.

Symptoms and fixes:
- `DLL load failed while importing cublas` → missing `nvidia-*-cu12`
  packages; install the `cuda` extra.
- `cuda engine requested but cupy/CUDA is not available` → cupy not
  installed or import failing.
- Harmless warning `CUDA path could not be detected` → cosmetic when the
  pip `nvidia-*` packages are present; do not set `CUDA_PATH` to a
  mismatched toolkit.

GPU detection is a cached runtime probe (a real matmul), not just
`import cupy` — the package can import while its backend DLLs are broken.
Verify the whole stack with `python scripts/gpu_smoke.py` (cuBLAS matmul,
NVRTC RawKernel, collatz CUDA sieve, koopman GPU matmul).

## Benchmarking guidance

- Warm the JIT before timing (first njit call compiles; `cache=True`
  persists across processes).
- MITM searches: measure near the enumeration budget, not on tiny cases —
  per-side enumeration, not result count, drives cost.
- GPU pays off for large dense work (1024+ mode koopman matrices, big
  sieve classes); small cases are CPU-faster due to transfer overhead.
