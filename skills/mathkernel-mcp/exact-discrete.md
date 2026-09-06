# Exact discrete mathematics over MCP

Use the typed-object tools for Stage C domains:

1. `math_capabilities` or `math_capability_query(domain="graph" |
   "combinatorics" | "finite_group" | "finite_algebra")`
2. `math_object_create(kind, definition)` → retain `data.object_id`
3. `math_apply(object_id, operation, parameters)`
4. Inspect `semantic_status`, `evidence_bundle`, and `claim_evidence` before
   communicating a result.

## Graphs

Kinds: `graph`, `directed_graph`, `weighted_graph`, `multi_graph`. The facade
accepts `nodes` as an alias for `vertices`; edge weights are exact rational
strings or integers, never floats.

```json
{
  "kind": "weighted_graph",
  "definition": {
    "vertices": ["s", "a", "t"],
    "directed": true,
    "edges": [
      {"source": "s", "target": "a", "weight": "3"},
      {"source": "a", "target": "t", "weight": "2"},
      {"source": "s", "target": "t", "weight": "4"}
    ]
  }
}
```

Then call `math_apply` with `operation="maximum_flow"` and
`parameters={"source": "s", "sink": "t"}`. A verified result includes a
`max_flow_min_cut` certificate and `data.verification.accepted=true`.

Other operations: `bfs`, `dfs`, `connected_components`,
`strongly_connected_components`, `shortest_path`, `minimum_spanning_tree`,
`minimum_cut`, `matching`, `euler_path`, `coloring`, `topological_sort`,
`cycle_detection`, `centrality`, `isomorphic_to`.

For NP-hard objectives, distinguish `data.optimality`: `optimum` requires an
exact proof/certificate; `candidate` is feasible but unproven; `impossible` is
exact infeasibility; `unknown` means a search budget was exhausted. Never
describe `unknown` as nonexistence.

Traversal operations may use a Numba CSR backend; check
`evidence_bundle.computation[*].metadata.backend`. Compiled results are still
certificate-checked and fall back to Python automatically.

## Combinatorics and generating functions

- `combinatorial_class`: `count`, `generate`, `verify`
- `generating_function`: `coefficient`, `recurrence`, `verify`

`generate` is lazy and bounded; truncated output includes `total_count` when
known. Recurrence and rational generating-function conversions are exact.

## Finite algebra

- `finite_group`, `permutation_group`, `finite_abelian_group`,
  `group_homomorphism`
- `finite_ring`, `finite_field`
- `module`

Constructions validate axioms. A reducible GF(p^m) modulus returns
`status="refuted"`; a non-unit inverse returns
`semantic_status="does_not_exist"`; module normal forms include independently
checked reconstruction certificates.

Relevant limits are reported by `math_capabilities`:
`max_graph_vertices`, `max_graph_edges`, `max_combinatorial_items`,
`max_group_elements`, `max_field_degree`, and `max_normal_form_dim`.
