# Sub-skill: State-conditioned dynamics and structural discovery

Part of the `mathkernel` skill. Load when working with state-conditioned
orbit access (`T^kappa(x)(x)`), conditioned closure proofs, GF(2) orbit
solving, or word-expression symmetry discovery: modules
`conditioned_dynamics`, `gf2_conditioned`, `structural_discovery`.

## conditioned_dynamics — exact access maps

An access map `kappa: X -> Z>=0` selects a per-state lag; constant kappa
recovers ordinary fixed-lag dynamics.

```python
from mathkernel.conditioned_dynamics import (
    AccessMap, iterate_transition, apply_access, solve_access_to_target,
    observable_symmetry, conditioned_closure,
    affine_cyclic_symmetry_access, affine_cyclic_access_formula,
    symbolic_affine_access, fold_product,
    discover_factor_swap_symmetry, prove_factor_swap_conditioned_closure)

acc = solve_access_to_target(transition, target)   # least k: T^k(x)=target[x]
comp = acc.compose(other, transition)              # cocycle law k + lambda(T^k x)
out  = conditioned_closure(observation, transition, [acc1, acc2],
                           coefficients=[1, -1], modulus=97, constant=0)
# -> {"holds": bool, "checked_states": n, "counterexamples": [...]}
```

`iterate_transition` is cycle-aware, so enormous conditioned lags are safe.
`conditioned_closure` proves `sum h_j * O(T^kappa_j(x) x) = constant (mod N)`
by exhaustive exact enumeration — `holds=True` is a proof, not evidence.

Affine cyclic dynamics (`T(x) = x + A mod M`, `gcd(A, M) = 1`):

```python
k = affine_cyclic_access_formula(M, A, state, target)   # arbitrary-size ints
sym = symbolic_affine_access(M, A, target_kind="xor", target_constant=C)
sym.expression                                       # compact exact kappa(x)
sym.evaluate(x)                                      # lag for one state
```

## gf2_conditioned — GF(2)-linear orbit access

```python
from mathkernel.gf2_conditioned import (
    solve_orbit_access_bounded, verify_orbit_access, gf2_jump,
    gf2_predictive_closure, gf2_jump_polynomial_support,
    verify_sparse_predictive_closure)

r = solve_orbit_access_bounded(columns, state, target, max_lag)
# baby-step/giant-step, O(sqrt(max_lag)); None if unreachable within bound
ok = verify_orbit_access(columns, state, target, r.lag)

c = gf2_predictive_closure(columns, lag)        # s[n+lag] = J_lag s[n]
sup = gf2_jump_polynomial_support(columns, lag) # T^lag as poly in T
verified = verify_sparse_predictive_closure(columns, lag, sup)  # exact matrix equality
```

`columns` are the bit-packed GF(2)-linear transition columns (same format as
`gf2m_from_transition`). A sparse support (e.g. `T^K = I + T` -> `(0, 1)`)
is a giant-lag predictive closure; verification is exact, never sampled.

## structural_discovery — proof-producing symmetry search

Word-expression ASTs: `Var("x")`, `Const(c)`,
`Op("xor"|"add"|"mul"|"fold"|"rotl"|"low"|"high", ...)`. Discovery is
constrained and auditable: candidates are substituted structurally and
canonicalized under certified rewrite rules; a symmetry is reported only
when both sides reduce to the same canonical form.

```python
from mathkernel.structural_discovery import (
    Var, Const, Op, discover_symmetries, synthesize_symmetries,
    synthesize_vector_symmetries, folded_mul_x_xor_c)

expr = folded_mul_x_xor_c(0x9E37)                    # fold(x * (x xor C))
d = discover_symmetries(expr, word_bits=64, constants=[0x9E37])
# exact only: canonical-rewrite proofs

s = synthesize_symmetries(expr, 64, [0x9E37], max_depth=2,
                          probe_samples=1024, proof_top_k=64)
# numeric ranking (trust "numeric") + exact canonical-rewrite proofs
# for the top-k survivors

v = synthesize_vector_symmetries(outputs, word_bits, constants,
                                 probe_samples=2048, top_k=64)
# two-word states (x0, x1); exact proof by exhaustive finite-state
# enumeration when 2*word_bits <= 16
```

Numeric probing only ever *ranks* candidates — it never upgrades trust to
EXACT. Overall synthesis results are `trust: numeric`; individual proved
closures carry explicit `"trust": "exact"` records.

## Facade equivalents

All of the above are reachable through `MathKernel` methods (which add
validation, derivation tracking, and trust labels):
`conditioned_access_solve`, `conditioned_symmetry_access`,
`conditioned_access_compose`, `conditioned_closure`,
`symbolic_conditioned_access`, `affine_conditioned_access`,
`gf2_conditioned_access`, `gf2_predictive_closure`,
`synthesize_conditioned_closures`, `synthesize_gf2_vector_conditioned_access`,
`discover_structural_conditioned_closure`,
`discover_factor_swap_conditioned_closure`.
