# Certified numerics, ODEs, optimization (MCP)

Trust: `interval_certified` = rigorous enclosure; `numeric_high_precision` =
arbitrary-precision evidence; `numeric` = float64 evidence. None of these is
a proof — upgrade with exact/symbolic tools when the claim matters.

## Roots and integrals

```
math_root_find(expr_id, "x", a="1", b="2")                    # high precision
math_root_find(expr_id, "x", a="1", b="2", certified=true)    # interval proof of existence
math_root_find(expr_id, "x", a="1", b="2", fast=true)         # float64 Brent
math_root_scan(expr_id, "x", "0.5", "10", intervals=64)       # parallel scan
math_quadrature(expr_id, "x", "0", "pi")                      # tanh-sinh + cross-check
```

Bounds accept expressions (`"pi"`, `"sqrt(2)"`). Quadrature disagreement
between tanh-sinh and Gauss-Legendre returns `status: conflict` — do not
quote the value as reliable.

## ODEs / PDE

Represent a general PDE through the generic typed tools before selecting any
numerical method:

```
math_object_create(object_type="PDEProblem", definition={...})
math_apply(object_id=pde_id, operation="verify")
math_apply(object_id=pde_id, operation="classify", parameters={"equation_index": 0})
math_apply(object_id=pde_id, operation="boundary_compatibility")
math_apply(object_id=pde_id, operation="derive_weak_form", parameters={
  "integration_variables": ["x", "y"],
  "trial_spaces": [{"name": "U", "field": "u", "family": "H1",
                    "regularity_order": 1, "trace_boundary_indices": [0, 1]}],
  "test_space": {"name": "v", "field": "u", "family": "H1_D",
                 "regularity_order": 1, "trace_boundary_indices": [0, 1]},
  "integration_by_parts": [{"term_index": 0, "coordinate": "x"}]
})
```

Equation derivatives are bounded multi-indices aligned to the declared
independent variables. Classification covers only the represented scalar,
linear, two-variable second-order principal part and returns conditional sign
cases when needed. Compatibility checks only represented Dirichlet corner and
initial-boundary traces. Weak-form derivation requires explicit spaces, trace
indices, integration variables and transfers. Inspect the complete product-rule
volume terms and both outward-oriented boundary faces; a zero-trace marker does
not erase a term. Verification establishes only the represented conditional
integral identity. Do not claim space membership, condition completeness,
existence, uniqueness, well-posedness, FEM discretization or a PDE solution.

G.3 represents the finite-element layer through the same generic tools:

```
math_object_create(object_type="FEMMesh", definition={
  "weak_form_id": weak_id, "cell_type": "triangle",
  "points": [[0,0],[1,0],[1,1],[0,1]],
  "cells": [[0,1,2],[0,2,3]]
})
math_apply(object_id=mesh_id, operation="reference_element", parameters={})
math_apply(object_id=reference_id, operation="basis", parameters={})
math_apply(object_id=reference_id, operation="quadrature",
           parameters={"degree_exact": 2})
math_apply(object_id=mesh_id, operation="finite_element_space", parameters={
  "reference_element_id": reference_id, "basis_id": basis_id, "field": "u"
})
```

Use the returned IDs and replay every derived object with `verify`. Only P1
Lagrange bases are represented. A mesh may verify its combinatorial topology
while the overall status remains unknown because domain coverage/geometric
non-overlap are not established. Numeric near-degeneracy cannot advance. Never
report these artifacts as assembly, an algebraic solve or a PDE solution.

For a supported scalar linear stationary P1 weak form, G.4 adds:

```
math_apply(object_id=space_id, operation="assemble", parameters={
  "quadrature_id": quadrature_id, "substitutions": {"f": 1}
})
math_apply(object_id=system_id, operation="solve", parameters={
  "method": "auto", "tolerance": 1e-10, "condition_limit": 1e12
})
```

Replay `AssembledSystem` and `FEMSolution` with `verify`. Inspect
`quadrature_exact`, local contributions, sparse raw/transformed matrices,
natural contributions, essential constraints, `solve_status`, residual, ranks
and conditioning. A verified singular/inconsistent result is a verified solver
outcome, not a solution. A unique result solves only the assembled algebraic
system; it does not prove a continuous PDE solution, convergence or a continuum
error bound. Unsupported coupled, nonlinear, time-dependent, periodic or
unresolved-boundary problems must remain refused.

G.5 uses the same generic MCP operation:

```
math_apply(object_id=solution_id, operation="estimate_error", parameters={})
math_apply(object_id=estimate_id, operation="mark",
           parameters={"strategy": "dorfler", "theta": 0.5})
math_apply(object_id=marking_id, operation="refine", parameters={})
math_apply(object_id=fine_estimate_id, operation="compare",
           parameters={"coarse_estimate_id": coarse_estimate_id})
```

`refine` returns both `object_ids.mesh` (`RefinedMesh`) and
`object_ids.transfer` (`MeshTransfer`). Replay both. The estimator currently
supports constant diagonal scalar diffusion on essential-boundary triangle-P1
meshes; other estimator conventions fail closed. Keep its local residual/jump
components separate from the algebraic residual. A residual estimator is not a
rigorous continuum bound, and a direct two-mesh observed estimator rate is
empirical rather than a convergence theorem.

```
math_ode_solve(rhs_id, y_var="y", x_var="x")                 # symbolic
math_ode_solve_numeric([rhs_id], ["0","1"], ["1"])           # RK45 mpmath
math_ode_ensemble([rhs_id], ["0","1"], [[1.0],[2.0]], steps=1000)  # GPU batch
math_pde_heat_1d(u0, alpha=0.1, dx=0.1, dt=0.001, steps=100)
math_pde_heat_2d(u0, alpha, dx, dt, steps)    # 2D FTCS, r <= 1/4 enforced
math_pde_wave_1d(u0, v0, c, dx, dt, steps)     # leapfrog, CFL <= 1
math_pde_advect_1d(u0, c, dx, dt, steps)      # upwind, CFL <= 1, diffusive
math_pde_mol_heat(u0, alpha, dx, t1, steps)   # method of lines + RK4
math_pde_ensemble(u0, alphas, dx, dt, steps)  # batched 2D heat sweep
```

The raw finite-difference results below are numeric trust; stability violations are hard errors.

Numeric systems name state variables `y0, y1, ...` in the rhs. The ensemble
tool runs one CUDA thread per trajectory when the GPU works, else a process
pool. FTCS enforces the stability condition and errors instead of returning
garbage.

## Optimization

```
math_optimize_critical_points(expr_id, ["x","y"])   # grad f = 0 (symbolic)
math_optimize_kkt(objective_id, [g_id], ["x"])      # KKT conditions
math_lp_solve(c, A, b)                              # exact rational simplex
math_optimize_minimize(expr_id, ["x"], [0.0])       # Nelder-Mead
math_optimize_multistart(expr_id, ["x"], starts)    # parallel multi-start
```

LP form: max cᵀx s.t. Ax ≤ b, x ≥ 0, b ≥ 0. Limits:
`MATHKERNEL_MAX_ITERATIONS`, `MATHKERNEL_TOLERANCE`, `MATHKERNEL_MAX_ODE_STEPS`.
