# Sub-skill: State-conditioned dynamics and structural discovery

Part of the `mathkernel-mcp` skill. Load when the task involves
state-conditioned orbit access (`T^kappa(x)(x)` — per-state lags instead of
one fixed lag), conditioned closure proofs, GF(2) orbit solving, or
symmetry discovery over word expressions (PRNG output functions).

## Exact access maps (finite systems)

These operate on a `system_id` from `math_finite_system_create`:

- `math_conditioned_access_solve(system_id, target)` — least nonnegative
  per-state lags `k(x)` with `T^k(x) = target[x]`. Error if any target is
  unreachable.
- `math_conditioned_symmetry_access(system_id, transform)` — first proves
  `O(S(x)) = O(x)` (observation symmetry), then solves `T^k(x) = S(x)`
  exactly.
- `math_conditioned_access_compose(system_id, first, second)` — compose two
  access maps with the exact cocycle law `k(x) + lambda(T^k(x) x)`.
- `math_conditioned_closure(system_id, accesses, coefficients, modulus,
  constant=0)` — proves `sum h_j * O(T^kappa_j(x) x) = constant (mod N)` by
  exhaustive exact enumeration over all states. `holds: true` is a proof;
  otherwise counterexample states are returned.

## Affine cyclic dynamics (`T(x) = x + A mod M`, `gcd(A, M) = 1`)

- `math_affine_conditioned_access(modulus, increment, state, target_state)`
  — exact `k = (target - state) * A^-1 mod M`; arbitrary-size integers.
- `math_symbolic_conditioned_access(modulus, increment, target_kind,
  target_constant, variable="x")` — compact exact symbolic access map
  `kappa(x)` for `target_kind` `"xor"` or `"add"` targets.

## GF(2)-linear dynamics

`columns` are bit-packed GF(2)-linear transition columns (same format as
`math_gf2m_from_transition`).

- `math_gf2_conditioned_access(columns, state, target, max_lag)` — exact
  bounded orbit solve via baby-step/giant-step, `O(sqrt(max_lag))`.
  `found: false` within the bound is itself an exact negative result.
- `math_gf2_predictive_closure(columns, lag, sparse_term_limit=16)` — exact
  relation `state[n+lag] = J_lag * state[n]`; when `T^lag` is sparse as a
  polynomial in `T` (e.g. `T^K = I + T`), the sparse identity is verified by
  exact matrix equality (`sparse_identity_verified: true`).

## Symmetry discovery and synthesis

Expression schema: `{"op": "var"|"const"|"xor"|"add"|"mul"|"fold"|"rotl"|...}`,
with `{"op": "var", "name": "x"}`, `{"op": "const", "value": n}`, and
`args` for compound nodes.

- `math_discover_structural_conditioned_closure(expression, word_bits,
  increment, constants, variable="x")` — exact only: canonical-rewrite
  proofs over a bounded candidate family; derives conditioned closures for
  proved symmetries. Overall `trust: exact`.
- `math_discover_factor_swap_conditioned_closure(word_bits, increment,
  xor_constant)` — proves the `fold(x * (x xor C))` factor-swap symmetry
  (xor-involution + multiplication commutativity) and derives its exact
  conditioned lag, with explicit certificate obligations.
- `math_synthesize_conditioned_closures(expression, word_bits, increment,
  constants, ...)` — enumerates a small transformation grammar, ranks
  candidates by deterministic numeric probing, then exactly proves the
  top-k survivors by canonical rewrite. **Overall `trust: numeric`** because
  ranking is sampled; each entry in `conditioned_closures` carries its own
  `"trust": "exact"`.
- `math_synthesize_gf2_vector_conditioned_access(outputs, word_bits,
  constants, columns, state_words, max_lag, ...)` — two-word state variant;
  exact symmetry proofs by exhaustive enumeration when
  `2 * word_bits <= 16`, plus exact GF(2) orbit-access lags.

## Trust discipline

- Exhaustive/enumerative results (`holds`, `found: false`, canonical-rewrite
  symmetries, verified sparse closures) are `exact` — they are proofs over a
  finite state space or a certified rewrite system.
- Synthesis *rankings* are `numeric` and never presented as proof; only the
  per-closure records marked `"trust": "exact"` are proved.
