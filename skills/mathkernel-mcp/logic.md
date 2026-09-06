# Logic, sets, quantifiers, and polynomial algebra (MCP)

## Sets

```
math_set_create(elements=["1", "2", "3"])        # or name="integers"
math_set_op(op="union", set_ids=[a, b])          # union/intersect/difference/complement
math_set_membership(element="2", set_id=s)       # -> member: true, exact
```

Set syntax inside `math_parse`: `{1, 2, 3}`, `Integers()`, `Reals()`,
`in(x, S)`, `union(A, B)`, `intersect(A, B)`, `difference(A, B)`,
`complement(A, universe)`, and binders `sumover(k, {1,2,3}, k^2)` /
`productover(...)`.

## Quantifier validity

```
e = math_parse("forall(x, Integers(), x + 0 = x)")
math_quantifier_check(e.expr_id)
# -> validity: valid | invalid | unknown, plus witness/countermodel when decidable
```

Arbitrary alternation depth is supported
(`forall(x, Reals, exists(y, Reals, forall(z, Reals, ...)))`); Z3 decides
the closed statement natively, and witnesses/countermodels are extracted per
level via Skolem constants (`data.skolem_levels`). Boolean connectives
`and(p, q, ...)`, `or(p, q, ...)`, `not(p)` are available inside
`math_parse`. Trust is `exact` when Z3 decides, `unknown` on timeout
(`MATHKERNEL_Z3_TIMEOUT_MS`) or outside the arithmetic fragment (complex
domains, transcendental functions). Never report `unknown` as either true
or false.

## Quantifier elimination

```
e = math_parse("exists(x, Reals, and(x > 0, x < y))")
r = math_quantifier_eliminate(e.expr_id)
# -> status: eliminated, formula: "not(y <= 0)", expr_id: <new id>
math_quantifier_eliminate_batch([e1, e2, ...], workers=4)   # process pool
```

True QE via Z3's `qe` tactic (LRA/LIA). The result is a quantifier-free
MathIR formula that renders and re-parses. `exact` on success; structured
`unknown` with fragment classification on undecidable fragments.
`MATHKERNEL_MAX_QE_VARIABLES` (default 16) caps variables.

## Polynomial algebra (all exact)

```
math_poly_groebner(expr_ids=[a, b], variables=["x", "y"], order="lex")
math_poly_divide(dividend_id=a, divisor_ids=[b], variables=["x", "y"])
math_poly_resultant(a_id=a, b_id=b, variable="x")
math_poly_discriminant(expr_id=a, variable="x")
math_poly_factor(expr_id=a)                    # over ZZ/QQ
math_poly_factor(expr_id=a, extension="sqrt(2)")
math_ideal_membership(expr_id=a, generator_ids=[b, c], variables=["x", "y"])
math_poly_groebner_batch(jobs=[...], workers=4)  # process pool
```

Monomial orders: `lex`, `grlex`, `grevlex` (+ `i*` variants). Limits: 64
polynomials / 16 variables per call; single calls honor
`MATHKERNEL_SOLVER_TIMEOUT_SECONDS`. Ideal membership uses the Gröbner
remainder test — `member: true` is an exact proof of membership.
