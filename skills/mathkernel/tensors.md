# Sparse tensors

Dict-of-keys sparse tensors with exact rational entries, Einstein-style
contraction, and exact sparse linear solve (`mathkernel.tensors`).

## Facade

```python
a = kernel.tensor_create([2, 3], {"0,0": "1", "0,2": "2", "1,1": "3"})
b = kernel.tensor_create([3, 2], {"0,0": "1", "1,1": "1", "2,0": "1", "2,1": "1"})
c = kernel.tensor_contract("ij,jk->ik", [a.data["tensor_id"], b.data["tensor_id"]])
kernel.tensor_get(c.data["tensor_id"])      # entries when nnz <= 256, else summary
x = kernel.tensor_solve(a2_id, b_id)        # exact sparse Gauss-Jordan, A x = b
```

- Values `"2"` / `"1/3"` are exact rationals (`EXACT`); float literals like
  `"0.5"` mark the tensor numeric (`NUMERIC`) and bar it from exact solve.
- Contraction specs: `"ij,jk->ik"`, traces `"ii->"`, reductions `"ij->i"`,
  multi-operand `"ij,jk,kl->il"`. Labels are single letters; repeated labels
  on one operand extract the diagonal.
- `exact=False` (or numeric inputs) dispatches numeric tiers:
  CuPy CSR matmul (`numeric-gpu`, runtime-probed) → njit Gustavson CSR
  (`njit`) → pure-Python float64 reference (`python-fallback`). The active
  tier is reported in `data["engine_tier"]` and `capabilities()["tensors"]`.
- Guards: rank <= 8, total size <= 10M, nnz <= 5M. Large tensors return
  shape/nnz summaries by design.

## Module API

`SparseTensor(shape, entries)`, `contract(spec, *tensors)`,
`sparse_solve_exact(a, b)`, `matmul_numeric(a, b)`,
`contract_numeric(spec, *tensors)` — usable without the facade.
