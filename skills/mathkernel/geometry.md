# Geometry and finite topology

Read `DIFFERENTIAL_GEOMETRY.md` for E.1–E.2 and
`COMPUTATIONAL_GEOMETRY.md` for E.3. Read
`ALGEBRAIC_TOPOLOGY.md` for E.4–E.5 and `PHASE_E_COMPLETION.md` plus
`PHASE_E_AUDIT.md` for the closed Phase E scope.

- Create `Manifold(name, dimension)`, then `Chart` objects with unique ordered
  coordinates, explicit domains, and optional orientation. Create a symmetric
  `Metric` with optional `(positive, negative)` signature, a directional
  `CoordinateMap`, dense variance-aware `TensorField`, or canonical sparse
  `DifferentialForm`. Public mathematical entries are restricted MathIR.
- Chart coordinates are real and ordered. Preserve the order, chart ID, domain,
  tensor variance and curvature convention in downstream use.
- Metric operations are `inverse_metric`, `christoffel`, `riemann`, `ricci`,
  `scalar_curvature`, `einstein`, and `geodesic_equations`. Most produce a typed
  derived object in `data.object_id`; scalar curvature is a scalar result.
- Coordinate-map operations are `jacobian` and `verify`. Tensor-field operations
  are `covariant_derivative(metric_id=...)` and
  `lie_derivative(vector_field_id=...)`. Form operations are
  `wedge(other_id=...)`, `exterior_derivative`,
  `interior_product(vector_field_id=...)`, `pullback(map_id=...)`, and
  `hodge_star(metric_id=..., orientation=...)`.
- Inspect `side_conditions`, especially chart-domain conditions and
  `det(g) != 0`. A symbolic quotient is only valid where those conditions hold.
- Inspect `data.details.identity_checks`. Checks include inverse identity, torsion
  freedom, metric compatibility, Riemann symmetries, first Bianchi, Ricci
  symmetry, contracted Bianchi, map composition, graded commutativity, `d²=0`,
  pullback commutation with `d`, and the Hodge double-star sign when signature
  is supplied. An undecided identity is not a refutation or a proof.
- Decimal components cap trust at numeric even if the displayed curvature is an
  integer. Coordinate-local symbolic computation does not prove global manifold
  properties, chart coverage, completeness or topology.
- `max_geometry_dimension`, `max_geometry_rank`, and `max_geometry_work` reject oversized symbolic
  tensors before construction. Repeated curvature operations reuse immutable
  exact derivative/contraction caches.
- For computational geometry, create concrete finite `Point`/`PointSet`,
  `Polygon`, half-space `Polytope`, or `Triangulation` objects. Use
  `orientation`, `incircle`, `segment_intersection`, `convex_hull`,
  `nearest_neighbor`, `delaunay`, `voronoi`, `contains`, `intersection`,
  `triangulate`, and `verify` only where capability discovery advertises them.
- Exact coordinates can establish exact topology. Decimal inputs are numeric;
  if a filtered predicate returns `classification: ambiguous`, do not infer an
  orientation or topology. Exact cocircular Delaunay input is non-unique and is
  intentionally ambiguous. Polygon intersection is convex-only; general exact
  high-dimensional hull/facet enumeration is not part of E.3. Triangulation
  verification checks face orientation, edge incidence, crossing/shared-edge
  consistency, and nested interiors; a refuted result is not a valid mesh.
- Computational limits are `max_geometry_points`, `max_geometry_simplices`, and
  `max_geometry_work`. Exact Delaunay/Voronoi are deliberately bounded rather
  than delegated to an unverifiable approximate topology engine.
- In E.5, an exact `Triangulation` supports `to_simplicial_complex`. The bridge
  re-verifies orientation, incidence, and nonoverlap, checks the derived
  `boundary²=0`, and retains source ancestry. Never invoke it for a numeric,
  ambiguous, or refuted triangulation; those inputs cannot acquire exact
  topology through conversion.
- For E.4, define a `SimplicialComplex` from unique vertex labels and maximal
  vertex-index simplices, a `CubicalComplex` from elementary integer intervals,
  or an integral `ChainComplex` from ranks and boundary matrices. Use `verify`,
  `chain_complex`, `boundary_matrix`, `homology`, and `euler_characteristic`
  only where advertised.
- Never report homology unless every integral boundary composition is zero.
  `homology` defaults to Z; use `coefficient="Q"` or
  `coefficient="GF(p)", prime=p` for exact base change. Over Z report both the
  free rank and invariant-factor torsion. Over fields, torsion coefficients are
  not defined; report Betti dimension and representative cycles.
- Preserve stored basis order and orientation conventions when interpreting
  representatives. E.4 does not provide persistent homology, cup products,
  homotopy groups, or topology inferred from approximate point clouds.
- Topology limits are `max_topology_dimension`, `max_topology_cells`,
  `max_topology_matrix_entries`, `max_topology_entry_bits`, and
  `max_topology_work`; integer homology also respects `max_normal_form_dim`.
  Face closures enforce the cell cap while expanding, so a limit error is a
  stopping condition rather than a reason to retry an oversized definition.
