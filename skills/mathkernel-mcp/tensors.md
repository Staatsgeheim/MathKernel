# Sparse tensors (MCP)

```
math_tensor_create(shape=[2, 3], entries={"0,0": "1", "0,2": "2", "1,1": "3"})
math_tensor_contract(spec="ij,jk->ik", tensor_ids=[a, b])           # exact
math_tensor_contract(spec="ij,jk->ik", tensor_ids=[a, b], exact=false)  # GPU/njit
math_tensor_solve(a_id, b_id)        # exact sparse A x = b over rationals
math_tensor_get(tensor_id)           # entries when small, summary when large
```

- Exact rational entries (`"2"`, `"1/3"`) keep `exact` trust; float literals
  (`"0.5"`) mark the tensor `numeric` and bar it from `math_tensor_solve`.
- Specs: matmul `"ij,jk->ik"`, trace `"ii->"`, reductions `"ij->i"`,
  multi-operand chains. Labels are single letters.
- `exact=false` dispatches: CuPy CSR (`numeric-gpu`) → njit Gustavson
  (`njit`) → float64 reference (`python-fallback`); the active tier is in
  `data.engine_tier` and `math_capabilities` under `tensors.engines`.
- Guards: rank <= 8, total size <= 10M, nnz <= 5M. Results above 256 nonzeros
  come back as shape/nnz summaries — contract further or slice instead of
  pulling full serializations.
