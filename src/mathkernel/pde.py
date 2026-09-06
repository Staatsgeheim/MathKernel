# =============================================================================
# MathKernel - PDE solvers beyond 1D heat: 2D heat, wave, advection, ensembles
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Explicit finite-difference PDE solvers with three-tier dispatch:
pure-Python reference -> njit stencil loops -> CuPy/CUDA stencils.

All results are NUMERIC evidence. Stability conditions (FTCS r <= 1/4 in 2D,
CFL c*dt/dx <= 1) are hard errors — never silent garbage. Boundaries are
Dirichlet-zero. Upwind advection is first-order and diffusive; that is
documented, not hidden.
"""

from __future__ import annotations


def _njit(fn):
    try:
        from numba import njit
        return njit(cache=False)(fn)
    except Exception:
        return None


# --- reference stencils (proof-grade Python) -----------------------------------

def _heat2d_py(u, nx, ny, r, steps):
    u = [row[:] for row in u]
    out = [row[:] for row in u]
    for _ in range(steps):
        for i in range(1, nx - 1):
            ui, um, up = u[i], u[i - 1], u[i + 1]
            oi = out[i]
            for j in range(1, ny - 1):
                oi[j] = ui[j] + r * (um[j] + up[j] + ui[j - 1] + ui[j + 1] - 4.0 * ui[j])
        u, out = out, u
    return u


def _wave_py(u, v, n, c2, steps):
    # leapfrog: u^{n+1} = 2u^n - u^{n-1} + C^2 * laplacian(u^n);
    # first step incorporates the (dt-scaled) initial velocity
    prev = u[:]
    cur = [u[i] + v[i] + 0.5 * c2 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
           if 0 < i < n - 1 else 0.0 for i in range(n)]
    for _ in range(1, steps):
        nxt = [0.0] * n
        for i in range(1, n - 1):
            nxt[i] = 2.0 * cur[i] - prev[i] + c2 * (cur[i - 1] - 2.0 * cur[i] + cur[i + 1])
        prev, cur = cur, nxt
    return cur


def _advect_py(u, n, courant, steps):
    # first-order upwind for c > 0 (diffusive by construction)
    for _ in range(steps):
        prev = u[:]
        for i in range(1, n):
            u[i] = prev[i] - courant * (prev[i] - prev[i - 1])
        u[0] = 0.0
    return u


def _heat2d_fast(u, nx, ny, r, steps):
    # array-native stencil for numba
    out = u.copy()
    for _ in range(steps):
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                out[i, j] = u[i, j] + r * (u[i - 1, j] + u[i + 1, j]
                                           + u[i, j - 1] + u[i, j + 1]
                                           - 4.0 * u[i, j])
        u, out = out, u
    return u


def _wave_fast(u, v, n, c2, steps):
    prev = u.copy()
    cur = u.copy()
    for i in range(1, n - 1):
        cur[i] = u[i] + v[i] + 0.5 * c2 * (u[i - 1] - 2.0 * u[i] + u[i + 1])
    cur[0] = 0.0; cur[n - 1] = 0.0
    nxt = u.copy()
    for _ in range(1, steps):
        for i in range(1, n - 1):
            nxt[i] = 2.0 * cur[i] - prev[i] + c2 * (cur[i - 1] - 2.0 * cur[i] + cur[i + 1])
        nxt[0] = 0.0; nxt[n - 1] = 0.0
        prev, cur, nxt = cur, nxt, prev
    return cur


def _advect_fast(u, n, courant, steps):
    prev = u.copy()
    for _ in range(steps):
        prev[:] = u
        for i in range(1, n):
            u[i] = prev[i] - courant * (prev[i] - prev[i - 1])
        u[0] = 0.0
    return u


_heat2d_njit = _njit(_heat2d_fast)
_wave_njit = _njit(_wave_fast)
_advect_njit = _njit(_advect_fast)


def _xp(prefer_gpu: bool):
    from .koopman import _xp as probe
    return probe(prefer_gpu=prefer_gpu)


# --- public solvers --------------------------------------------------------------

def heat_2d(u0: list[list[float]], alpha: float, dx: float, dt: float,
            steps: int, prefer_gpu: bool = True) -> dict:
    """u_t = alpha * (u_xx + u_yy) on a rectangular grid, Dirichlet-zero
    boundaries. Stability: r = alpha*dt/dx^2 <= 1/4 (hard error)."""
    nx, ny = len(u0), len(u0[0])
    r = alpha * dt / (dx * dx)
    if r > 0.25:
        raise ValueError(f"FTCS 2D unstable: r = {r:.4f} > 1/4; reduce dt or alpha")
    xp = _xp(prefer_gpu)
    if xp.__name__ == "cupy":
        u = xp.asarray(u0, dtype=xp.float64)
        for _ in range(steps):
            v = u.copy()
            v[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
                u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
                - 4.0 * u[1:-1, 1:-1])
            u = v
        return {"u": u.get().tolist(), "engine_tier": "numeric-gpu"}
    if _heat2d_njit is not None:
        import numpy as np
        u = _heat2d_njit(np.asarray(u0, dtype=np.float64), nx, ny, r, steps)
        return {"u": [list(map(float, row)) for row in u], "engine_tier": "njit"}
    return {"u": _heat2d_py([list(map(float, row)) for row in u0], nx, ny, r, steps),
            "engine_tier": "python-fallback"}


def wave_1d(u0: list[float], v0: list[float], c: float, dx: float, dt: float,
            steps: int, prefer_gpu: bool = True) -> dict:
    """u_tt = c^2 u_xx via leapfrog central differences. CFL: c*dt/dx <= 1.
    v0 is the initial velocity; it is scaled by dt internally."""
    n = len(u0)
    courant = c * dt / dx
    if courant > 1.0:
        raise ValueError(f"wave CFL violated: c*dt/dx = {courant:.4f} > 1")
    c2 = courant * courant
    v_scaled = [dt * v for v in v0]
    xp = _xp(prefer_gpu)
    if xp.__name__ == "cupy":
        prev = xp.asarray(u0, dtype=xp.float64)
        cur = prev.copy()
        lap = xp.zeros(n, dtype=xp.float64)
        lap[1:-1] = prev[:-2] - 2.0 * prev[1:-1] + prev[2:]
        cur[1:-1] = prev[1:-1] + xp.asarray(v_scaled[1:-1]) + 0.5 * c2 * lap[1:-1]
        cur[0] = 0.0; cur[-1] = 0.0
        for _ in range(1, steps):
            nxt = xp.zeros(n, dtype=xp.float64)
            nxt[1:-1] = 2.0 * cur[1:-1] - prev[1:-1] + c2 * (
                cur[:-2] - 2.0 * cur[1:-1] + cur[2:])
            prev, cur = cur, nxt
        return {"u": cur.get().tolist(), "engine_tier": "numeric-gpu"}
    if _wave_njit is not None:
        import numpy as np
        u = _wave_njit(np.asarray(u0, dtype=np.float64),
                       np.asarray(v_scaled, dtype=np.float64), n, c2, steps)
        return {"u": [float(x) for x in u], "engine_tier": "njit"}
    return {"u": _wave_py(list(map(float, u0)), v_scaled, n, c2, steps),
            "engine_tier": "python-fallback"}


def advect_1d(u0: list[float], c: float, dx: float, dt: float, steps: int,
              prefer_gpu: bool = True) -> dict:
    """u_t + c u_x = 0, first-order upwind (c >= 0). CFL: c*dt/dx <= 1.
    First-order upwind is numerically diffusive — treat sharp features with
    suspicion."""
    if c < 0:
        raise ValueError("upwind advection requires c >= 0")
    n = len(u0)
    courant = c * dt / dx
    if courant > 1.0:
        raise ValueError(f"advection CFL violated: c*dt/dx = {courant:.4f} > 1")
    xp = _xp(prefer_gpu)
    if xp.__name__ == "cupy":
        u = xp.asarray(u0, dtype=xp.float64)
        for _ in range(steps):
            v = u.copy()
            v[1:] = u[1:] - courant * (u[1:] - u[:-1])
            v[0] = 0.0
            u = v
        return {"u": u.get().tolist(), "engine_tier": "numeric-gpu"}
    if _advect_njit is not None:
        import numpy as np
        u = _advect_njit(np.asarray(u0, dtype=np.float64), n, courant, steps)
        return {"u": [float(x) for x in u], "engine_tier": "njit"}
    return {"u": _advect_py(list(map(float, u0)), n, courant, steps),
            "engine_tier": "python-fallback"}


def heat_2d_ensemble(u0: list[list[float]], alphas: list[float], dx: float,
                     dt: float, steps: int, prefer_gpu: bool = True,
                     workers: int | None = None) -> dict:
    """Parameter sweep of heat_2d over diffusivities. GPU: one batched
    (B, nx, ny) CuPy stencil — all trajectories advance per kernel launch.
    CPU: process pool over alphas."""
    from .parallel import process_map, resolve_workers
    for alpha in alphas:
        if alpha * dt / (dx * dx) > 0.25:
            raise ValueError(f"FTCS 2D unstable for alpha={alpha}: r > 1/4")
    xp = _xp(prefer_gpu)
    if xp.__name__ == "cupy":
        nx, ny = len(u0), len(u0[0])
        base = xp.asarray(u0, dtype=xp.float64)
        u = xp.broadcast_to(base, (len(alphas), nx, ny)).copy()
        rs = xp.asarray([a * dt / (dx * dx) for a in alphas], dtype=xp.float64)
        rs = rs[:, None, None]
        for _ in range(steps):
            v = u.copy()
            v[:, 1:-1, 1:-1] = u[:, 1:-1, 1:-1] + rs * (
                u[:, :-2, 1:-1] + u[:, 2:, 1:-1] + u[:, 1:-1, 2:] + u[:, 1:-1, :-2]
                - 4.0 * u[:, 1:-1, 1:-1])
            u = v
        return {"solutions": u.get().tolist(), "engine_tier": "numeric-gpu"}
    jobs = [{"u0": u0, "alpha": a, "dx": dx, "dt": dt, "steps": steps}
            for a in alphas]
    results = process_map(_heat2d_job, jobs,
                          workers=resolve_workers(workers))
    return {"solutions": [r["u"] for r in results],
            "engine_tier": results[0]["engine_tier"] if results else "python-fallback"}


def _heat2d_job(job: dict) -> dict:
    return heat_2d(job["u0"], job["alpha"], job["dx"], job["dt"], job["steps"],
                   prefer_gpu=False)


def mol_heat_1d(u0: list[float], alpha: float, dx: float, t1: float,
                steps: int = 1000) -> dict:
    """Method of lines: semidiscretize u_t = alpha*u_xx into the ODE system
    du_i/dt = alpha/dx^2 (u_{i-1} - 2u_i + u_{i+1}) and integrate with the
    existing fixed-step RK4 — no duplicated integrator logic."""
    from .ode import rk4_float64
    n = len(u0)
    lam = alpha / (dx * dx)

    def rhs(t, y, out):
        out[0] = 0.0
        out[n - 1] = 0.0
        for i in range(1, n - 1):
            out[i] = lam * (y[i - 1] - 2.0 * y[i] + y[i + 1])

    u = rk4_float64(rhs, 0.0, list(map(float, u0)), t1, steps)
    return {"u": [float(x) for x in u], "engine_tier": "numeric",
            "integrator": "rk4-float64", "semidiscretization": "centered-2nd-order"}
