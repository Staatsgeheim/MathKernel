# =============================================================================
# MathKernel - Tests for PDE solvers beyond 1D heat (v0.21)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""PDEs: 2D heat, wave, advection, method-of-lines, ensembles; tier
cross-checks and stability enforcement."""

import math

import pytest

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel import pde


def _pulse(n=21):
    return [math.sin(math.pi * i / (n - 1)) for i in range(n)]


def _pulse2d(n=11):
    return [[math.sin(math.pi * i / (n - 1)) * math.sin(math.pi * j / (n - 1))
             for j in range(n)] for i in range(n)]


# --- 2D heat -----------------------------------------------------------------

def test_heat_2d_decays_and_matches_reference():
    u0 = _pulse2d()
    ref = pde._heat2d_py([row[:] for row in u0], 11, 11, 0.2, 50)
    out = pde.heat_2d(u0, 1.0, 0.1, 0.002, 50, prefer_gpu=False)
    assert out["engine_tier"] in ("njit", "python-fallback")
    for a_row, b_row in zip(ref, out["u"]):
        for a, b in zip(a_row, b_row):
            assert abs(a - b) < 1e-12
    # diffusion decays the peak
    assert out["u"][5][5] < u0[5][5]


def test_heat_2d_stability_hard_error():
    with pytest.raises(ValueError, match="unstable"):
        pde.heat_2d(_pulse2d(), 1.0, 0.1, 0.01, 10)  # r = 1.0 > 1/4


def test_heat_2d_boundaries_stay_zero():
    out = pde.heat_2d(_pulse2d(), 1.0, 0.1, 0.002, 20, prefer_gpu=False)
    # sin(pi) is ~1.2e-16 in float64, so "zero" means machine-epsilon small
    assert all(abs(v) < 1e-15 for v in out["u"][0])
    assert all(abs(row[0]) < 1e-15 and abs(row[-1]) < 1e-15 for row in out["u"])


# --- wave --------------------------------------------------------------------

def test_wave_1d_conserves_shape_approximately():
    n = 41
    u0 = _pulse(n)
    v0 = [0.0] * n
    out = pde.wave_1d(u0, v0, 1.0, 1.0 / (n - 1), 0.4 / (n - 1), 20,
                      prefer_gpu=False)
    ref = pde._wave_py(u0, v0, n, 0.16, 20)
    for a, b in zip(ref, out["u"]):
        assert abs(a - b) < 1e-12
    # boundaries pinned
    assert out["u"][0] == 0.0 and out["u"][-1] == 0.0


def test_wave_1d_cfl_hard_error():
    with pytest.raises(ValueError, match="CFL"):
        pde.wave_1d(_pulse(), [0.0] * 21, 1.0, 0.05, 0.06, 10)


def test_wave_zero_velocity_starts_symmetric():
    n = 21
    out = pde.wave_1d(_pulse(n), [0.0] * n, 1.0, 0.05, 0.02, 5,
                      prefer_gpu=False)
    mid = n // 2
    for i in range(n):
        assert abs(out["u"][i] - out["u"][n - 1 - i]) < 1e-12


# --- advection -----------------------------------------------------------------

def test_advect_1d_shifts_profile():
    n = 31
    u0 = [1.0 if 5 <= i <= 10 else 0.0 for i in range(n)]
    out = pde.advect_1d(u0, 1.0, 1.0, 0.5, 4, prefer_gpu=False)
    ref = pde._advect_py(u0[:], n, 0.5, 4)
    assert out["u"] == ref
    # the block moved right
    assert out["u"][7] > 0.0 and out["u"][13] > 0.0


def test_advect_cfl_and_sign_errors():
    with pytest.raises(ValueError, match="CFL"):
        pde.advect_1d(_pulse(), 1.0, 0.05, 0.06, 10)
    with pytest.raises(ValueError, match="c >= 0"):
        pde.advect_1d(_pulse(), -1.0, 0.05, 0.01, 10)


# --- method of lines -----------------------------------------------------------

def test_mol_heat_matches_ftcs():
    u0 = _pulse(21)
    mol = pde.mol_heat_1d(u0, 1.0, 0.05, 0.002, steps=2000)
    # FTCS with r=0.4, dt = r*dx^2/alpha = 0.001, 2 steps -> t=0.002
    from mathkernel.ode import heat_ftcs
    ftcs = heat_ftcs(u0, 1.0, 0.05, 0.001, 2, prefer_gpu=False)
    for a, b in zip(mol["u"], ftcs["u"]):
        assert abs(a - b) < 1e-4  # different time integrators, same physics


# --- ensemble -------------------------------------------------------------------

def test_heat_2d_ensemble_matches_single():
    u0 = _pulse2d(9)
    out = pde.heat_2d_ensemble(u0, [0.5, 1.0], 0.1, 0.002, 10,
                               prefer_gpu=False, workers=2)
    assert len(out["solutions"]) == 2
    single = pde.heat_2d(u0, 1.0, 0.1, 0.002, 10, prefer_gpu=False)
    for a_row, b_row in zip(out["solutions"][1], single["u"]):
        for a, b in zip(a_row, b_row):
            assert abs(a - b) < 1e-12


def test_heat_2d_ensemble_stability_checked_per_alpha():
    with pytest.raises(ValueError, match="unstable"):
        pde.heat_2d_ensemble(_pulse2d(9), [0.5, 5.0], 0.1, 0.002, 10)


# --- kernel facade -----------------------------------------------------------------

def test_kernel_pde_facades():
    k = MathKernel()
    r = k.pde_heat_2d(_pulse2d(9), 1.0, 0.1, 0.002, 10, prefer_gpu=False)
    assert r.ok and r.trust == TrustLevel.NUMERIC
    r = k.pde_wave_1d(_pulse(21), [0.0] * 21, 1.0, 0.05, 0.02, 10,
                     prefer_gpu=False)
    assert r.ok and r.trust == TrustLevel.NUMERIC
    r = k.pde_advect_1d(_pulse(21), 1.0, 0.05, 0.02, 10, prefer_gpu=False)
    assert r.ok and r.trust == TrustLevel.NUMERIC
    r = k.pde_mol_heat(_pulse(21), 1.0, 0.05, 0.001, steps=500)
    assert r.ok and r.trust == TrustLevel.NUMERIC


def test_kernel_pde_stability_error_paths():
    k = MathKernel()
    assert not k.pde_heat_2d(_pulse2d(9), 1.0, 0.1, 0.01, 10).ok
    assert not k.pde_wave_1d(_pulse(21), [0.0] * 21, 1.0, 0.05, 0.06, 10).ok
    assert not k.pde_advect_1d(_pulse(21), 1.0, 0.05, 0.06, 10).ok


def test_kernel_pde_grid_cap():
    k = MathKernel(Settings(max_pde_grid=10))
    assert not k.pde_heat_2d(_pulse2d(9), 1.0, 0.1, 0.002, 10).ok  # 81 > 10


def test_capabilities_pde():
    caps = MathKernel().capabilities()
    assert "pde_heat_2d" in caps["pde"]["operations"]
    assert caps["pde"]["max_pde_grid"] > 0
