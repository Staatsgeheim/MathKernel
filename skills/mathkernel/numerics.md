# Certified numerics, ODEs, optimization

Trust tiers: `interval_certified` (rigorous mpmath.iv enclosures) >
`numeric_high_precision` (mpmath arbitrary precision) > `numeric` (float64
fast paths, njit/CUDA). Numeric results are evidence, never proof.

## Roots and quadrature

```python
kernel.root_find(eid, "x", a="1", b="2")                  # ridder, 50 dps
kernel.root_find(eid, "x", a="1", b="2", certified=True)  # interval_certified
kernel.root_find(eid, "x", a="1", b="2", fast=True)       # float64 Brent
kernel.root_scan(eid, "x", "0.5", "10", intervals=64)     # process pool
kernel.quadrature(eid, "x", "0", "pi")                    # tanh-sinh + GL check
```

Bounds accept MathIR (`"pi"`, `"1/3"`, `"sqrt(2)"`). Quadrature cross-checks
tanh-sinh against Gauss-Legendre; disagreement returns `status="conflict"`.

## ODEs / PDE

For a general PDE, create a typed `PDEProblem` before discussing a method or
solution. Its equation terms use derivative multi-indices aligned with
`independent_variables`; coefficients, sources, domain bounds and condition
values are restricted MathIR scalars. Use:

```python
pde_id = kernel.object_create("PDEProblem", definition).data["object_id"]
kernel.apply(pde_id, "verify")
kernel.apply(pde_id, "classify", {"equation_index": 0})
kernel.apply(pde_id, "boundary_compatibility")
weak = kernel.apply(pde_id, "derive_weak_form", {
    "integration_variables": ["x", "y"],
    "trial_spaces": [{"name": "U", "field": "u", "family": "H1",
                      "regularity_order": 1,
                      "trace_boundary_indices": [0, 1]}],
    "test_space": {"name": "v", "field": "u", "family": "H1_D",
                   "regularity_order": 1,
                   "trace_boundary_indices": [0, 1]},
    "integration_by_parts": [{"term_index": 0, "coordinate": "x"}],
})
```

`classify` establishes only the represented scalar two-variable linear
second-order principal-part type, returning explicit sign conditions when it
cannot decide the discriminant. `boundary_compatibility` compares represented
Dirichlet corner and initial-boundary traces; it does not establish condition
completeness. For a weak form, declare integration variables, ordered trial
spaces, the test space, every essential trace index, and each requested
term/coordinate integration-by-parts transfer. Check the returned volume terms,
both outward-oriented boundary faces, and any coefficient-derivative term.
Terms marked `vanishes_by_trace` remain part of the identity. Never infer space
membership, existence, uniqueness, well-posedness, discretization or a solution
from a verified `WeakForm`.

For the G.3 finite-element representation, create an `FEMMesh` linked to the
weak-form ID with `cell_type` (`interval`, `triangle`, or `tetrahedron`), points
and simplex cells. A verified `Triangulation` ID may replace direct triangle
data. Then derive rather than publicly construct the remaining objects:

```python
mesh_id = kernel.object_create("FEMMesh", mesh_definition).data["object_id"]
reference = kernel.apply(mesh_id, "reference_element")
reference_id = reference.data["object_id"]
basis = kernel.apply(reference_id, "basis")
quadrature = kernel.apply(reference_id, "quadrature", {"degree_exact": 2})
space = kernel.apply(mesh_id, "finite_element_space", {
    "reference_element_id": reference_id,
    "basis_id": basis.data["object_id"], "field": "u"})
```

Replay each derived object with `verify`. Treat mesh coverage and geometric
non-overlap as unknown even when determinant, orientation, incidence, basis,
quadrature and DOF checks pass. Do not treat the space as an assembled system or
solution. Near-degenerate numeric orientation is intentionally unable to
advance. G.3 supports only P1 Lagrange bases and the advertised bounded
reference-simplex quadrature degrees.

For G.4, assemble only after all G.3 sources replay and supply concrete values
for unresolved PDE parameters:

```python
system = kernel.apply(space.data["object_id"], "assemble", {
    "quadrature_id": quadrature.data["object_id"],
    "substitutions": {"f": 1},
})
solution = kernel.apply(system.data["object_id"], "solve", {
    "method": "auto", "tolerance": 1e-10,
    "condition_limit": 1e12,
})
```

Inspect `quadrature_exact`, the raw and transformed sparse systems, natural
facet contributions and essential constraints before interpreting the result.
`auto` selects exact arithmetic unless floating assembled values require the
numeric sparse path. Always report `solve_status`, residual, ranks and condition
diagnostic. A verified singular or inconsistent status verifies that outcome;
it is not a solution. A verified unique `FEMSolution` solves only the assembled
finite-dimensional system. Never promote it to a continuous PDE solution or
continuum error bound. G.4 refuses coupled/nonlinear/time-dependent forms,
periodic constraints, unresolved fluxes and weak factors above first order.

For G.5 adaptivity, call `estimate_error` only on a complete unique or
ill-conditioned solution in the advertised constant-diagonal triangle-P1,
essential-boundary scope. Then derive marking and refinement artifacts:

```python
estimate = kernel.apply(solution.data["object_id"], "estimate_error")
marking = kernel.apply(estimate.data["object_id"], "mark", {
    "strategy": "dorfler", "theta": 0.5})
refinement = kernel.apply(marking.data["object_id"], "refine")
refined_mesh_id = refinement.data["object_ids"]["mesh"]
transfer_id = refinement.data["object_ids"]["transfer"]
```

Inspect every local volume/jump component, `quadrature_exact`, the requested
and closure-refined cells, child-to-parent indices, and exact interpolation
rows. Re-enter the normal reference/basis/quadrature/space/assemble/solve chain
from `refined_mesh_id`. Compare a fine estimate only with its direct parent via
`compare`. Report that result as an empirical observed estimator rate. Never
describe the estimator as a rigorous error bound or the observed rate as a
convergence theorem; G.5 stores all three flags explicitly.

```python
kernel.ode_solve(rhs_id, "y", "x")            # sympy.dsolve + classification
kernel.ode_solve_numeric([rhs_id], ["0", "1"], ["1"])          # RK45 mpmath
kernel.ode_solve_numeric([rhs_id], ["0", "1"], ["1"], fast=True)  # float64
kernel.ode_ensemble([rhs_id], ["0", "1"], y0s, steps=1000)      # GPU/CPU batch
kernel.pde_heat_1d(u0, alpha, dx, dt, steps)  # FTCS, r <= 1/2 enforced
kernel.pde_heat_2d(u0, alpha, dx, dt, steps)  # 2D FTCS, r <= 1/4 enforced
kernel.pde_wave_1d(u0, v0, c, dx, dt, steps)   # leapfrog, CFL <= 1 enforced
kernel.pde_advect_1d(u0, c, dx, dt, steps)    # upwind, CFL <= 1, diffusive
kernel.pde_mol_heat(u0, alpha, dx, t1, steps) # method of lines + RK4
kernel.pde_ensemble(u0, alphas, dx, dt, steps)  # batched 2D heat sweep
```

The raw finite-difference results below are NUMERIC trust; stability violations are hard errors;
boundaries are Dirichlet-zero. Tiers: Python reference -> njit -> CuPy
(2D thread-per-cell / batched ensemble stencils), bit-checked in tests.
`MATHKERNEL_MAX_PDE_GRID` caps grid cells.

Systems use state names `y0, y1, ...` in the rhs expressions; in symbolic
`ode_solve` the bare `y` symbol means `y(x)`. Ensembles dispatch to a CUDA
RawKernel (one thread per trajectory, RHS compiled from MathIR to C) or a
process pool of float64 RK4. `MATHKERNEL_MAX_ODE_STEPS` caps the RK45 loop.

## Optimization

```python
kernel.optimize_critical_points(eid, ["x", "y"])   # grad f = 0 (symbolic)
kernel.optimize_kkt(obj_id, [g_id], ["x"])         # KKT conditions (symbolic)
kernel.lp_solve(c, A, b)                           # exact rational simplex
kernel.lp_solve(c, A, b, exact=False)              # njit float64 tableau
kernel.optimize_minimize(eid, ["x"], start=[0.0])  # Nelder-Mead
kernel.optimize_multistart(eid, ["x"], starts)     # process pool
```

LP form: max cᵀx s.t. Ax ≤ b, x ≥ 0, b ≥ 0. Bland's rule prevents cycling.
`MATHKERNEL_MAX_ITERATIONS` / `MATHKERNEL_TOLERANCE` bound the numeric loops.
