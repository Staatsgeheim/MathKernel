# Sub-skill: Library facade API (symbolic, calculus, matrices, integers)

Part of the `mathkernel` skill. Load when writing Python code against the
`MathKernel` facade for symbolic math, calculus, linear algebra, integer
work, planning, or codegen.

## Expressions

```python
r = kernel.parse("x^2 + 2*x + 1 = 0")        # or kernel.parse_latex(...)
eid = r.data["expr_id"]
kernel.get(eid)                               # MathIR + display
r2 = kernel.substitute(eid, {"x": "a + 1"})   # new expr_id
```

Implicit multiplication is rejected with candidates; `E` = `exp(1)`,
pi = `pi()`.

## Contexts

```python
ctx = kernel.context_create(domains={"x": "positive", "n": "integer"})
cid = ctx.data["context_id"]
kernel.solve(eid, "x", context_id=cid)        # assumptions threaded through
```

## Solve / simplify / calculus

```python
kernel.solve(eid, "x")
kernel.solve_system([eid1, eid2], ["x", "y"])
kernel.simplify(eid, mode="simplify")         # simplify|expand|factor
kernel.differentiate(eid, "x", order=2)
kernel.integrate(eid, "x")                    # definite: lower=, upper=
kernel.limit(eid, "x", "oo", direction="+-")
kernel.series(eid, "x", point="0", order=6)
kernel.summation(eid, "k", "1", "n")
kernel.product(eid, "k", "1", "n")
```

## Numeric

```python
kernel.numeric_evaluate(eid, values={"x": "1.5"}, dps=50)   # trust numeric
kernel.interval_evaluate(eid, bounds={"x": [1, 2]}, dps=50) # certified
```

## Matrices

```python
m = kernel.matrix_create([["1", "2"], ["3", "4"]])
mid = m.data["matrix_id"]
kernel.matrix_det(mid); kernel.matrix_inverse(mid); kernel.matrix_rref(mid)
kernel.matrix_eigenvalues(mid); kernel.matrix_solve(mid, rhs_id)
```

## Integers

```python
kernel.integer_analyze("170141183460469231731687303715884105727")
kernel.integer_compute("modpow", ["3", "1000", "97"])
kernel.integer_compute("affine_jump", ["a", "c", "K"], modulus="2147483648")
kernel.integer_batch([{"operation": "is_prime", "values": ["101"]}, ...])
```

## Reasoning / proof

```python
kernel.reason(eid, formal=True)               # one-shot planner + verify
plan = kernel.plan(eid); kernel.execute_plan(plan.data["plan_id"])
kernel.prove_equivalence("x^2 - 1", "(x-1)*(x+1)")
kernel.counterexample("x^2 > 0", "x > 0")
```

## Codegen

```python
art = kernel.codegen(eid, language="python", target="evaluate")
kernel.verify_code(art.data["artifact_id"])   # always verify before use
kernel.execute_code(art.data["artifact_id"], inputs={"x": 1.0})
```

## Jobs (async)

```python
job = kernel.job_submit("collatz", {"n_max": 12})
kernel.job_status(job.data["job_id"])
kernel.job_result(job.data["job_id"])
```
