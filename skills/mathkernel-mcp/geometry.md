# Geometry MCP workflows

Use only the generic typed tools: `math_object_create`, `math_apply`,
`math_object_get`, and `math_capability_query(domain="geometry")`.

1. Create `Manifold` with `name` and bounded positive `dimension`.
2. Create `Chart` with that `manifold_id`, a name, ordered coordinate strings,
   optional MathIR domain relations, and optional `orientation` (`1` or `-1`).
3. Create `Metric` with the `chart_id`, symmetric square `components`, and
   optional `(positive, negative)` `signature`.
4. Apply `inverse_metric`, `christoffel`, `riemann`, `ricci`,
   `scalar_curvature`, `einstein`, or `geodesic_equations` to the metric ID.
5. For E.2, create a directional `CoordinateMap` with source/target chart IDs,
   forward expressions and optional inverse; a dense `TensorField` with an
   `up`/`down` variance sequence; or a sparse `DifferentialForm` with increasing
   index tuples. Apply `jacobian`/`verify`, covariant/Lie derivatives, or
   `wedge`/`exterior_derivative`/`interior_product`/`pullback`/`hodge_star`.

Retain derived object IDs. Report chart/domain conditions, coordinate ordering,
variance and the stored `R^rho_(sigma mu nu)` convention. Check
`data.details.identity_checks` before making a strong claim. Symbolic output is
not formal proof; numeric/decimal ancestry cannot be upgraded because the final
formula looks exact. Check `d_squared_zero`, `pullback_commutes_with_d`, graded
commutativity, coordinate inverse composition, and signature-aware double-star
results when applicable. E.1–E.2 remain coordinate-local and do not claim global
chart coverage, geodesic completeness, topology or physical validity.

For E.3, create concrete `Point`, `PointSet`, `Polygon`, half-space `Polytope`,
or `Triangulation` objects. Capability discovery exposes distance, orientation,
incircle, segment intersection, hull, containment, convex intersection,
triangulation, nearest-neighbor, Delaunay and Voronoi operations. Exact input may
establish exact topology; decimal input is ancestry-capped at numeric, and
`classification: ambiguous` must remain unresolved. Cocircular Delaunay input
is deliberately non-unique. Do not imply general concave polygon Boolean
operations or high-dimensional hull enumeration. `Triangulation.verify` also
checks nonoverlapping interiors, not only face orientation and edge incidence.
An exact verified triangulation supports `to_simplicial_complex`; the bridge
replays those checks, verifies `boundary_squared_zero`, and returns a derived
complex whose source is the triangulation. Numeric, ambiguous, or refuted
geometry cannot use this operation to gain exact trust.

For E.4, create a `SimplicialComplex` with vertex labels and
`maximal_simplices`, a `CubicalComplex` with elementary integer
`maximal_cells`, or a `ChainComplex` with chain `ranks`, integral `boundaries`,
and optional `basis_labels`. Apply `verify` before communicating topology.
`chain_complex` exposes oriented cellular boundaries; `boundary_matrix` selects
one degree; `homology` accepts `Z` (default), `Q`, or `GF(p)` with a verified
prime; a chain complex also supports `euler_characteristic`.

Inspect all degree-specific checks plus `euler_poincare`. Over Z communicate
both `betti_number` and `torsion_coefficients`; over a field communicate vector-
space dimension and cycle representatives, never integer torsion. Preserve the
stored basis order. Do not imply persistent homology, cohomology products,
homotopy groups, or topology from approximate geometry.

Treat topology resource-limit failures as final for that input size. Canonical
simplex/cube closure is capped during expansion, before an oversized face set
or boundary matrix is materialized.
