# Logic, sets, and quantifiers

MathIR v2 node kinds: `set`, `in`, `setop`, `bool`, `quantifier`, `binder`.

## Parsing surface

| Syntax | Node |
|---|---|
| `{1, 2, 3}` / `{}` | finite `SetNode` |
| `Naturals()` `Integers()` `Rationals()` `Reals()` `Complexes()` `EmptySet()` | named `SetNode` |
| `in(x, S)` | `MembershipNode` |
| `union(A, B, ...)` `intersect(A, B, ...)` `difference(A, B)` `complement(A, universe)` | `SetOpNode` |
| `and(p, q, ...)` `or(p, q, ...)` `not(p)` | `BoolNode` |
| `forall(x, body)` / `forall(x, domain, body)` / `exists(...)` | `QuantifierNode` |
| `sumover(k, {1,2,3}, k^2)` / `productover(...)` | `BinderNode` (finite domains) |

Bound variables are scoped: substitution never replaces a variable bound by
an enclosing quantifier/binder, and free-symbol analysis excludes them.

## Kernel facade

```python
s = kernel.set_create(elements=["1", "2", "3"])     # or name="integers"
u = kernel.set_op("union", [s.data["expr_id"], other_id])
kernel.set_membership("2", u.data["expr_id"])       # -> member: true, exact

q = kernel.parse("forall(x, Integers(), x + 0 = x)").data["expr_id"]
r = kernel.quantifier_check(q)
# r.data["validity"] in {"valid", "invalid", "unknown"}
# invalid -> r.data["countermodel"]; valid exists -> r.data["witness"]
```

`quantifier_check` decides the truth of the closed statement with Z3's
native quantifier support — arbitrary alternation depth
(`forall(x, exists(y, forall(z, ...)))`). Witnesses/countermodels are
extracted per level by sat-preservingly stripping the leading quantifiers
of the statement (or its negation-normal-form negation) into Skolem
constants. `data.skolem_levels` names the levels that materialized.

Trust is `EXACT` when Z3 decides, `UNKNOWN` on timeout/unsupported fragment
(complex domains, non-arithmetic functions). Z3 honors
`MATHKERNEL_Z3_TIMEOUT_MS`.

## Quantifier elimination

```python
q = kernel.parse("exists(x, Reals, and(x > 0, x < y))").data["expr_id"]
r = kernel.quantifier_eliminate(q)
# r.data["formula"] == "not(y <= 0)"-style quantifier-free equivalent
# r.data["expr_id"] holds the result as a new expression (re-parses)
kernel.quantifier_eliminate_batch([q1, q2, ...], workers=4)  # process pool
```

True QE via Z3's `qe` tactic (LRA/LIA fragments), mapped back to MathIR
through `from_z3` (And/Or/Not/relations over linear arithmetic). EXACT when
the tactic returns a quantifier-free equivalent; structured UNKNOWN with the
v0.20 fragment classification on undecidable fragments — never claims exact
on partial evidence. `MATHKERNEL_MAX_QE_VARIABLES` (default 16) caps the
variable count.

## Engine mappings

- SymPy: sets → `sympy.sets` (FiniteSet/Union/Intersection/Complement/Contains);
  binders over finite sets evaluate exactly; booleans → `And`/`Or`/`Not`.
  Quantifiers raise (no SymPy support).
- Z3: quantifiers → `ForAll`/`Exists` with domain guards; membership over
  finite sets → disjunction of equalities; naturals → `Int` sort + `x >= 0`;
  rationals/reals → `Real` sort; booleans → `And`/`Or`/`Not`.
