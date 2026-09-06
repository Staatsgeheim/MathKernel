# Exact discrete mathematics

Use Stage C typed objects through `MathKernel.object_create` and
`MathKernel.apply`, or import the standalone modules for direct Python work.
All graph weights/capacities are exact `Fraction` values; combinatorial counts
are exact integers; finite algebra validates its axioms before returning an
object.

## Graphs

```python
from mathkernel import MathKernel

kernel = MathKernel()
graph = kernel.object_create("weighted_graph", {
    "vertices": ["s", "a", "t"],
    "directed": True,
    "edges": [
        {"source": "s", "target": "a", "weight": "3"},
        {"source": "a", "target": "t", "weight": "2"},
        {"source": "s", "target": "t", "weight": "4"},
    ],
})
flow = kernel.apply(graph.data["object_id"], "maximum_flow",
                    {"source": "s", "sink": "t"})
assert flow.data["flow_value"] == "5"
assert flow.data["verification"]["accepted"]
```

Supported kinds are `graph`, `directed_graph`, `weighted_graph`, and
`multi_graph`. `nodes` is accepted as an alias for `vertices` at the facade.
Operations include `bfs`, `dfs`, `connected_components`,
`strongly_connected_components`, `shortest_path`, `minimum_spanning_tree`,
`maximum_flow`, `minimum_cut`, `matching`, `euler_path`, `coloring`,
`topological_sort`, `cycle_detection`, `centrality`, and `isomorphic_to`.

Graph results distinguish feasibility from optimality. A greedy coloring can be
`verified` while `data["optimality"]` remains `candidate` or `unknown`; budget
exhaustion is never reported as impossibility. Maximum flow carries a
`max_flow_min_cut` certificate, matching carries a Kőnig cover, and isomorphism
carries a checked permutation.

BFS and connected components use a Numba CSR kernel when available; the compiled
order/parent arrays are still checked by the Python certificate verifier, and the
computation evidence records `backend="numba-csr"` or `"python"`.

## Combinatorics and generating functions

```python
cls = kernel.object_create("combinatorial_class",
                           {"kind": "combinations", "n": 6, "k": 2})
assert kernel.apply(cls.data["object_id"], "count").data["value"] == 15

gf = kernel.object_create("generating_function", {
    "kind": "ordinary",
    "coefficients": [0, 1],
    "recurrence": {"coefficients": [1, 1], "initial": [0, 1]},
})
assert kernel.apply(gf.data["object_id"], "coefficient", {"n": 10}).data["value"] == 55
```

Generation is lazy and bounded by `max_items` /
`MATHKERNEL_MAX_COMBINATORIAL_ITEMS`; truncated output reports the exact total
count when known. Linear-recurrence coefficient extension uses a checked int64
Numba kernel and falls back to arbitrary-precision integers before any overflow
is possible.

## Finite groups, rings, fields, and modules

```python
group = kernel.object_create("finite_group", {"cyclic": 6})
assert kernel.apply(group.data["object_id"], "order").data["value"] == 6

field = kernel.object_create("finite_field",
                             {"prime": 5, "modulus_coeffs": [2, 0, 1]})
product = kernel.apply(field.data["object_id"], "multiply",
                       {"left": [0, 1], "right": [0, 1]})
assert product.data["value"]["coeffs"] == [3, 0]

module = kernel.object_create("module", {
    "generators": ["a", "b"], "relations": [[2, 0], [0, 4]],
})
assert kernel.apply(module.data["object_id"], "abelian_group").data[
    "value"]["structure"] == "Z/2Z x Z/4Z"
```

Invalid constructions are mathematical verdicts, not silent objects: a
reducible field modulus returns `status="refuted"`, and a non-unit ring inverse
returns `semantic_status="does_not_exist"`. Cayley-table validation and
small-prime GF(p)[x] multiplication have exact Numba tiers (`p < 2**24`,
degree `<= 64` for field multiplication) with Python fallback outside the
fragment.

## Resource limits

Stage C limits are environment-driven: `MATHKERNEL_MAX_GRAPH_VERTICES`,
`MATHKERNEL_MAX_GRAPH_EDGES`, `MATHKERNEL_MAX_COMBINATORIAL_ITEMS`,
`MATHKERNEL_MAX_GROUP_ELEMENTS`, `MATHKERNEL_MAX_FIELD_DEGREE`, and
`MATHKERNEL_MAX_NORMAL_FORM_DIM`.
