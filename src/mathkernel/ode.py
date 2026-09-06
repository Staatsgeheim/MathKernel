# =============================================================================
# MathKernel - ODEs and a scoped PDE solver
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""ODEs and a scoped PDE solver.

- Symbolic ODEs via sympy.dsolve with classification (SYMBOLIC).
- Numeric IVP: own adaptive Dormand-Prince RK45 on mpmath
  (NUMERIC_HIGH_PRECISION) with per-step error reporting; njit float64 fast
  path (NUMERIC); optional scipy solve_ivp cross-check (`sci` extra).
- IVP ensembles (parameter sweeps / many initial conditions) batch onto the
  GPU via a CuPy RawKernel — one thread per trajectory, fixed-step RK4 with
  the RHS compiled from MathIR to C. CPU fallback: process pool of njit RK4.
- PDE scope is deliberately honest: 1D heat equation via explicit FTCS
  finite differences (NUMERIC evidence only), njit stencil with a CuPy path.
"""
from __future__ import annotations

import math

from .models import Expr
from .numerics import compile_float64, emit_c, emit_float64
from .parallel import process_map, resolve_workers

# Dormand-Prince coefficients
_DP_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0]
_DP_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]]
_DP_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
_DP_B4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
          187 / 2100, 1 / 40]


def dsolve_symbolic(rhs, y_var: str, x_var: str, ics: dict | None = None):
    """Solve dy/dx = rhs(x, y) symbolically. Returns (solution, classification)."""
    import sympy as sp
    x = sp.Symbol(x_var)
    f = sp.Function(y_var)
    # in the rhs, the bare dependent-variable symbol means y(x)
    rhs = rhs.subs(sp.Symbol(y_var), f(x))
    equation = sp.Eq(f(x).diff(x), rhs)
    classification = list(sp.classify_ode(equation, f(x)))
    solution = sp.dsolve(equation, f(x), ics=ics or None)
    return solution, classification


def rk45_mpmath(f, t0: str, y0: list[str], t1: str, tol: float = 1e-10,
                max_steps: int = 100_000, dps: int = 50) -> dict:
    """Adaptive RK45 at arbitrary precision. f(t, y) -> list of derivatives
    (mpmath objects). Returns the endpoint, step count, and error stats."""
    import mpmath as mp
    mp.mp.dps = dps
    t = mp.mpf(t0)
    y = [mp.mpf(v) for v in y0]
    target = mp.mpf(t1)
    direction = 1 if target >= t else -1
    h = direction * min(abs(target - t) / 100, mp.mpf("0.1")) or mp.mpf("0.01") * direction
    steps = 0
    rejected = 0
    max_err = mp.mpf(0)
    n = len(y)
    while (t - target) * direction < 0:
        if steps >= max_steps:
            raise ValueError(f"rk45 exceeded max_steps={max_steps}")
        if (t + h - target) * direction > 0:
            h = target - t
        k = []
        k.append(f(t, y))
        for stage in range(1, 6):
            ys = [y[i] + h * sum(_DP_A[stage][j] * k[j][i] for j in range(stage))
                  for i in range(n)]
            k.append(f(t + _DP_C[stage] * h, ys))
        y5 = [y[i] + h * sum(_DP_B[j] * k[j][i] for j in range(6)) for i in range(n)]
        ks = k + [f(t + h, y5)]
        y4 = [y[i] + h * sum(_DP_B4[j] * ks[j][i] for j in range(7)) for i in range(n)]
        err = max(abs(y5[i] - y4[i]) for i in range(n))
        max_err = max(max_err, err)
        if err <= tol:
            t += h
            y = y5
            steps += 1
        else:
            rejected += 1
        factor = 0.9 * (mp.mpf(tol) / (err + mp.mpf("1e-300"))) ** mp.mpf("0.2")
        h *= min(mp.mpf(5), max(mp.mpf("0.2"), factor))
    return {"t": mp.nstr(t, 30), "y": [mp.nstr(v, 30) for v in y],
            "steps": steps, "rejected_steps": rejected,
            "max_local_error": mp.nstr(max_err, 5), "dps": dps}


def compile_rhs_float64(rhs_irs: list[Expr], y_vars: list[str], t_var: str = "t",
                        njit: bool = False):
    """Compile a first-order system y' = F(t, y) to f(t, y, out)."""
    parts = []
    for i, ir in enumerate(rhs_irs):
        var_map = {t_var: "t"}
        var_map.update({v: f"y[{j}]" for j, v in enumerate(y_vars)})
        parts.append(f"    out[{i}] = {emit_float64(ir, var_map)}")
    source = "def _rhs(t, y, out):\n" + "\n".join(parts) + "\n"
    namespace: dict = {"math": math}
    exec(source, namespace)
    fn = namespace["_rhs"]
    if njit:
        try:
            from numba import njit as _njit
            return _njit(cache=False)(fn)
        except Exception:
            return None
    return fn


def rk4_float64(f, t0: float, y0: list[float], t1: float, steps: int) -> list[float]:
    """Fixed-step RK4 (float64). f(t, y, out) writes derivatives into out."""
    import numpy as np
    y = np.asarray(y0, dtype=np.float64)
    n = y.size
    h = (t1 - t0) / steps
    out = np.zeros(n)
    k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
    yt = np.zeros(n)
    t = t0
    for s in range(steps):
        f(t, y, k1)
        yt[:] = y + 0.5 * h * k1
        f(t + 0.5 * h, yt, k2)
        yt[:] = y + 0.5 * h * k2
        f(t + 0.5 * h, yt, k3)
        yt[:] = y + h * k3
        f(t + h, yt, k4)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t0 + (s + 1) * h
    return y.tolist()


def _rk4_njit():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def rk4(f, t0, y0, t1, steps):
        import numpy as np
        y = y0.copy()
        n = y.shape[0]
        h = (t1 - t0) / steps
        k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
        yt = np.zeros(n)
        t = t0
        for s in range(steps):
            f(t, y, k1)
            for i in range(n):
                yt[i] = y[i] + 0.5 * h * k1[i]
            f(t + 0.5 * h, yt, k2)
            for i in range(n):
                yt[i] = y[i] + 0.5 * h * k2[i]
            f(t + 0.5 * h, yt, k3)
            for i in range(n):
                yt[i] = y[i] + h * k3[i]
            f(t + h, yt, k4)
            for i in range(n):
                y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            t = t0 + (s + 1) * h
        return y

    return rk4


# --- GPU ensemble (CuPy RawKernel, one thread per trajectory) -----------------------

def ensemble_rk4_gpu(rhs_irs: list[Expr], y_vars: list[str], t0: float, t1: float,
                     y0s: list[list[float]], steps: int, t_var: str = "t") -> dict:
    """Integrate many trajectories on the GPU. The RHS is compiled from MathIR
    to C and inlined into an NVRTC kernel. Returns endpoints per trajectory."""
    from .koopman import _xp
    xp = _xp(prefer_gpu=True)
    if xp.__name__ != "cupy":
        raise ValueError("GPU ensemble requires a working CuPy/CUDA installation")
    dim = len(rhs_irs)

    def stage(k_name: str, t_expr: str) -> str:
        lines = []
        for i, ir in enumerate(rhs_irs):
            var_map = {t_var: "t_"}
            var_map.update({v: f"y[{j}]" for j, v in enumerate(y_vars)})
            lines.append(f"        {k_name}[{i}] = {emit_c(ir, var_map)};")
        return "\n".join(lines)

    source = f"""
extern "C" __global__
void rk4_ensemble(double* yall, int n_traj, int dim, double t0, double t1, int steps) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_traj) return;
    double* yi = yall + idx * dim;
    double h = (t1 - t0) / steps;
    double t = t0;
    double k1[16], k2[16], k3[16], k4[16], yt[16];
    for (int s = 0; s < steps; s++) {{
        {{ double* y = yi; double t_ = t;
{stage("k1", "t")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + 0.5*h*k1[i];
        {{ double* y = yt; double t_ = t + 0.5*h;
{stage("k2", "t+h/2")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + 0.5*h*k2[i];
        {{ double* y = yt; double t_ = t + 0.5*h;
{stage("k3", "t+h/2")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + h*k3[i];
        {{ double* y = yt; double t_ = t + h;
{stage("k4", "t+h")} }}
        for (int i = 0; i < dim; i++)
            yi[i] += (h/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        t += h;
    }}
}}
"""
    if dim > 16:
        raise ValueError("GPU ensemble supports at most 16 state variables")
    kernel = xp.RawKernel(source, "rk4_ensemble")
    n_traj = len(y0s)
    if any(len(row) != dim for row in y0s):
        raise ValueError("every initial condition must match the system dimension")
    data = xp.asarray(y0s, dtype=xp.float64).reshape(-1)
    block = 128
    grid = (n_traj + block - 1) // block
    kernel((grid,), (block,), (data, n_traj, dim, float(t0), float(t1), int(steps)))
    out = data.get().reshape(n_traj, dim)
    return {"endpoints": out.tolist(), "trajectories": n_traj, "steps": steps,
            "engine_tier": "numeric-gpu"}


def _ensemble_cpu_job(args) -> list[float]:
    rhs_srcs, y_vars, t0, t1, y0, steps = args
    from .parser import parse_math
    irs = [parse_math(s) for s in rhs_srcs]
    f = compile_rhs_float64(irs, y_vars, njit=False)
    return rk4_float64(f, t0, y0, t1, steps)


def ensemble_rk4_cpu(rhs_sources: list[str], y_vars: list[str], t0: float, t1: float,
                     y0s: list[list[float]], steps: int,
                     workers: int | None = None) -> dict:
    """CPU ensemble fallback: process pool of float64 RK4 trajectories."""
    resolved = resolve_workers(workers)
    jobs = [(rhs_sources, y_vars, t0, t1, y0, steps) for y0 in y0s]
    results = process_map(_ensemble_cpu_job, jobs, workers=resolved)
    return {"endpoints": results, "trajectories": len(y0s), "steps": steps,
            "engine_tier": "numeric-cpu", "workers": resolved}


# --- 1D heat equation (explicit FTCS) ------------------------------------------------

def heat_ftcs_float64(u0: list[float], alpha: float, dx: float, dt: float,
                      steps: int) -> list[float]:
    """u_t = alpha * u_xx on a fixed grid, Dirichlet-zero boundaries.
    Stability requires r = alpha*dt/dx^2 <= 1/2 (checked)."""
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = {r:.4f} > 1/2; reduce dt or alpha")
    u = list(u0)
    n = len(u)
    for _ in range(steps):
        new = [0.0] * n
        for i in range(1, n - 1):
            new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
        u = new
    return u


def _heat_njit():
    try:
        from numba import njit
    except ImportError:
        return None

    import numpy as np

    @njit(cache=True)
    def heat(u, r, steps):
        n = u.shape[0]
        out = np.zeros(n)
        for _ in range(steps):
            for i in range(1, n - 1):
                out[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
            u, out = out, u
            out[0] = 0.0
            out[n - 1] = 0.0
        return u

    return heat


def heat_ftcs(u0: list[float], alpha: float, dx: float, dt: float, steps: int,
              prefer_gpu: bool = True) -> dict:
    """1D heat equation with tier dispatch: CuPy stencil > njit > Python."""
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = {r:.4f} > 1/2; reduce dt or alpha")
    if prefer_gpu:
        from .koopman import _xp
        xp = _xp(prefer_gpu=True)
        if xp.__name__ == "cupy":
            u = xp.asarray(u0, dtype=xp.float64)
            for _ in range(steps):
                new = xp.zeros_like(u)
                new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
                u = new
            return {"u": u.get().tolist(), "engine_tier": "numeric-gpu",
                    "stability_r": r}
    kernel = _heat_njit()
    if kernel is not None:
        import numpy as np
        u = kernel(np.asarray(u0, dtype=np.float64), r, steps)
        return {"u": u.tolist(), "engine_tier": "njit", "stability_r": r}
    return {"u": heat_ftcs_float64(u0, alpha, dx, dt, steps),
            "engine_tier": "python-fallback", "stability_r": r}
