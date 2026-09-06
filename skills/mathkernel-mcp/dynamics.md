# Sub-skill: Finite dynamics, spectral analysis, and PRNG tooling

Part of the `mathkernel-mcp` skill. Load when the task involves finite
dynamical systems, Koopman/spectral analysis, closure relations, Walsh/
Fourier transforms, GF(2) linear algebra, or PRNG structure.

## Finite systems + Koopman suite

1. `math_finite_system_create(mu, transition, observation)` → `system_id`.
   States are `0..n-1`; `transition[x]` is the successor; `observation[x]`
   the output index (default identity). `mu` is `"uniform"` or a list of
   rational strings. Response reports stationarity.
2. Basis spec for the koopman tools: `{"kind": "walsh", "r": r}` for
   GF(2)^r (n = 2^r states) or `{"kind": "cyclic", "m": m}` for Z_M
   characters.
3. Tools (all take `system_id`, `basis`, and `exact=true|false`):
   - `math_koopman_matrix` — transport matrix Q; unitary under invariant
     measure.
   - `math_koopman_transfer` — observation-transfer C (depends on
     observation + output dictionary only).
   - `math_koopman_visibility` — rho_O per mode; lists invisible and fully
     visible modes. Exact zeros are proofs of invisibility.
   - `math_koopman_lagged` — lagged state tensor entry J_tau(alpha).
   - `math_koopman_observed` — observed statistic K_h(tau) by direct
     enumeration (ground truth for the basis contraction).
   - `math_koopman_diagnostics` — per-column IPR (exact; 1 = monomial
     transport) and transport entropy (numeric).

`exact=true` (default): rational/cyclotomic values, trust `exact`.
`exact=false`: vectorized complex128 path (GPU via CuPy when usable),
trust `numeric` — use for large systems, then confirm key zeros exactly.

## Finite Fourier (exact cyclotomic)

`math_finite_fourier_compute(op, params)`:
- `transfer` — T_F(h,k), state-character coefficients of chi_h ∘ F.
  Params: `{F: [int], N, h}`.
- `two_point` — B_m(K) for affine K-step maps. Params: `{F, N, m, a_k,
  b_k}`.
- `measure` — Fourier transform of a measure on Z_M. Params: `{mu: [...]}`.
- `dft` — exact conjugated DFT. Params: `{values: [...]}`.
- `orbit_correction` — nonzero-orbit average from closure sums. Params:
  `{closed_sum, total_sum, orbit_size}`.

## Closure relations

`math_closure_search(kind, params)`:
- `cyclic` — tuples k with sum k_j*mult_j == 0 (mod m), sum|k_j| <= H.
  Params: `{m, weight_bound, multipliers: [...]}`.
- `cyclic_order` — smallest order d with an irreducible equal-spacing
  closure. Params: `{a_k, m, weight_bound, d_min, d_max}`.
- `binary` / `binary_order` — XOR analog over GF(2)^width. Params:
  `{rows: [hex], width, k_step, max_weight, d, d_min, d_max}`.

Searches run on njit meet-in-the-middle kernels with an enumeration
budget; on budget errors, lower the weight/order.

## Cumulants

`math_cumulant_compute(op, order, values, columns)` — exact joint
cumulants via the set-partition lattice. op: `cumulant` (from block
moments), `moment` (inverse), `samples` (from sample columns). Subset
keys list coordinates: `"0,1,3"` → E[Z0*Z1*Z3].

## Binary fields and GF(2) linear algebra

- `math_gf2m_create(degree, reduction)` / `math_gf2m_from_transition
  (columns)` → `field_id`; then `math_gf2m_compute` (add/mul/pow/inv/
  sqrt/trace/quadratic_roots), `math_gf2m_coords`, `math_gf2m_jump_rows`,
  `math_gf2m_root_jump_rows`, `math_gf2m_closure_roots`. Elements are
  hex strings; degrees up to 1024 (njit n-limb kernels).
- `math_gf2_rank`, `math_gf2_nullspace(rows, width, side)`,
  `math_gf2_carryfree_cols` — bit-packed GF(2) matrix algebra.
- `math_gf2_minpoly(bits_hex, nbits)` — Berlekamp–Massey minimal
  polynomial of a bit sequence.
- `math_fwht(values)` — exact unnormalized Walsh–Hadamard transform
  (int64 njit path with arbitrary-precision fallback).
