# =============================================================================
# MathKernel - Tests for the finite-dynamics modules: cumulants, finite_fourier, koopman,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Tests for the finite-dynamics modules: cumulants, finite_fourier, koopman,
relations — including the two exact algebraic sanity checks from
generalized_spectral_selection_math_round2_revised.pdf (section 19).
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from mathkernel.cumulants import (
    block_moments_from_samples,
    cumulant_from_moments,
    joint_cumulant_from_samples,
    moment_from_cumulants,
    set_partitions,
)
from mathkernel.finite_fourier import (
    CyclotomicNumber,
    dft_zm,
    measure_fourier,
    orbit_correction,
    transfer_transform,
    two_point_difference,
)
from mathkernel.gf2m import gf2_apply, gf2_apply_transpose, gf2_rows_mul, gf2_rows_power
from mathkernel.kernel import MathKernel
from mathkernel.koopman import (
    FiniteSystem,
    contraction_value,
    koopman_matrix,
    lagged_tensor_value,
    mode_visibility,
    observed_statistic,
    path_tensor_value,
    static_cumulant_tensor,
    transfer_matrix,
    transport_diagnostics,
    walsh_basis,
)
from mathkernel.relations import (
    binary_closure,
    binary_closure_order,
    cyclic_closure,
    cyclic_closure_order,
    is_irreducible_binary,
    is_irreducible_cyclic,
)


@pytest.fixture(scope="module")
def kernel() -> MathKernel:
    return MathKernel()


# ---------------------------------------------------------------------------
# cumulants
# ---------------------------------------------------------------------------

def test_set_partitions_bell_numbers() -> None:
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140]
    for n, expected in enumerate(bell):
        assert len(set_partitions(n)) == expected


def test_cumulant_order2_is_covariance() -> None:
    moments = {1: Fraction(1, 2), 2: Fraction(1, 3), 3: Fraction(1, 4)}
    assert cumulant_from_moments(moments, 2) == Fraction(1, 4) - Fraction(1, 2) * Fraction(1, 3)


def test_cumulant_order3_matches_explicit_formula() -> None:
    m = {1: Fraction(1), 2: Fraction(2), 4: Fraction(3),
         3: Fraction(4), 5: Fraction(5), 6: Fraction(6), 7: Fraction(7)}
    got = cumulant_from_moments(m, 3)
    want = m[7] - m[3] * m[4] - m[5] * m[2] - m[6] * m[1] + 2 * m[1] * m[2] * m[4]
    assert got == want


def test_moment_cumulant_roundtrip() -> None:
    def reindex(sub: dict[int, Fraction], m: int) -> dict[int, Fraction]:
        # compact the set bits of m onto positions 0..bit_count-1
        pos = [j for j in range(4) if m >> j & 1]
        return {sum(1 << i for i, j in enumerate(pos) if s >> j & 1): v
                for s, v in sub.items()}

    moments = {m: Fraction(7 + m * m, 11) for m in range(1, 16)}
    cums = {}
    for m in range(1, 16):
        sub = {s: v for s, v in moments.items() if s & ~m == 0}
        cums[m] = cumulant_from_moments(reindex(sub, m), m.bit_count())
    for m in range(1, 16):
        sub = {s: cums[s] for s in range(1, 16) if s & ~m == 0}
        rebuilt = moment_from_cumulants(reindex(sub, m), m.bit_count())
        assert rebuilt == moments[m]


def test_cumulant_vanishes_for_independent_coordinates() -> None:
    # Z0 = Z1 identical fair bits, Z2 independent fair bit
    cols = [[Fraction(b) for b in (0, 1, 0, 1, 0, 1, 0, 1)],
            [Fraction(b) for b in (0, 1, 0, 1, 0, 1, 0, 1)],
            [Fraction(b) for b in (0, 0, 1, 1, 0, 0, 1, 1)]]
    assert joint_cumulant_from_samples(cols) == 0
    assert joint_cumulant_from_samples(cols[:2]) == Fraction(1, 4)


def test_block_moments_exact() -> None:
    cols = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]
    moments = block_moments_from_samples(cols)
    assert moments[3] == Fraction(1 * 3 + 2 * 4, 2)


# ---------------------------------------------------------------------------
# finite_fourier
# ---------------------------------------------------------------------------

def test_cyclotomic_arithmetic_exact() -> None:
    z4 = CyclotomicNumber.zeta(4)  # i
    assert (z4 * z4).to_rational() == Fraction(-1)
    w = CyclotomicNumber.one(4) + z4
    assert (w * w.conjugate()).to_rational() == 2
    # 1 + z5 + ... + z5^4 = 0 (cyclotomic reduction for prime order)
    s = sum((CyclotomicNumber.zeta(5, e) for e in range(5)), CyclotomicNumber.zero(5))
    assert s.is_zero()


def test_dft_of_delta_is_flat() -> None:
    out = dft_zm([1, 0, 0, 0])
    assert all(v.to_rational() == 1 for v in out)


def test_transfer_transform_identity_observation() -> None:
    # Paper 19.1: F(x) = x on Z_M -> T_F(h, k) = delta_{h,k}
    m = 7
    tf = transfer_transform(list(range(m)), m, 3)
    for k in range(m):
        expect = Fraction(int(k == 3))
        assert tf[k].to_rational() == expect


def test_transfer_conjugation_identity() -> None:
    # Lemma 10.3: T_F(-m, -l) = conj(T_F(m, l)) on a nonlinear F
    m = 8
    F = [(x * x + x) % m for x in range(m)]
    tf = transfer_transform(F, m, 3)
    tf_neg = transfer_transform(F, m, (-3) % m)
    for ell in range(m):
        assert tf_neg[(-ell) % m] == tf[ell].conjugate()


def test_transfer_parseval() -> None:
    m = 6
    F = [(x * x) % m for x in range(m)]
    tf = transfer_transform(F, m, 1)
    total = sum((tf[k].norm_sq() for k in range(m)), CyclotomicNumber.zero(m))
    # |T_F|^2 sums to 1 because chi_h ∘ F has unit norm under uniform measure
    assert total.to_rational() == 1


def test_two_point_translation_paper_check() -> None:
    # Paper 19.1: T(x) = x + b, identity observation -> B_m(K) = zeta_M^{-m b K}
    m = 5
    F = list(range(m))
    b = two_point_difference(F, m, 1, A_K=1, B_K=2)
    assert b == CyclotomicNumber(5, {(-2) % 5: 1})


def test_measure_fourier_uniform_is_indicator() -> None:
    m = 6
    mh = measure_fourier([Fraction(1, m)] * m)
    assert mh[0].to_rational() == 1
    assert all(v.is_zero() for v in mh[1:])


def test_orbit_correction() -> None:
    # orbit of size 2^r - 1 = 7: all weight on the closure -> unchanged;
    # all weight off the closure -> -1/7 (eq. 92)
    assert orbit_correction(Fraction(1), Fraction(1), 7) == Fraction(1)
    assert orbit_correction(Fraction(0), Fraction(1), 7) == Fraction(-1, 7)


# ---------------------------------------------------------------------------
# koopman
# ---------------------------------------------------------------------------

def _lfsr_system(r: int, taps: int) -> FiniteSystem:
    n = 1 << r

    def L(s: int) -> int:
        return (s >> 1) | (((s & taps).bit_count() & 1) << (r - 1))

    return FiniteSystem.uniform(n, [L(s) for s in range(n)])


def test_walsh_koopman_monomial_for_linear_map() -> None:
    # Corollary 11.2: binary-linear dynamics -> permutation Q, IPR=1, H=0
    sys = _lfsr_system(3, 0b011)
    assert sys.is_stationary()
    basis, nsq = walsh_basis(3)
    diag = transport_diagnostics(koopman_matrix(sys, basis, nsq))
    assert all(v == 1 for v in diag["ipr"])
    assert all(h == 0 for h in diag["entropy"])


def test_visibility_extremes() -> None:
    sys = _lfsr_system(3, 0b011)
    basis, nsq = walsh_basis(3)
    # identity observation sees everything
    assert all(v == 1 for v in mode_visibility(sys, basis, nsq))
    # msb observation sees exactly the constant and the msb modes
    sys_msb = FiniteSystem(sys.mu, sys.transition, [(s >> 2) & 1 for s in range(8)])
    vis = mode_visibility(sys_msb, basis, nsq)
    assert vis[0] == 1 and vis[4] == 1
    assert all(vis[a] == 0 for a in (1, 2, 3, 5, 6, 7))


def test_path_representation_matches_direct() -> None:
    # Theorem 8.4: Jc via Q powers and static cumulant tensor == direct
    sys = _lfsr_system(3, 0b011)
    basis, nsq = walsh_basis(3)
    Q = koopman_matrix(sys, basis, nsq)
    kappa = static_cumulant_tensor(sys, basis, 2)
    for a0 in range(8):
        for a1 in range(8):
            assert lagged_tensor_value(sys, basis, [0, 3], [a0, a1]) == \
                path_tensor_value(Q, kappa, [0, 3], [a0, a1])


def test_contraction_identity() -> None:
    # Proposition 7.10: observed connected statistic == C-wedge-Jc contraction
    sys = _lfsr_system(3, 0b011)
    basis, nsq = walsh_basis(3)
    phi = [[Fraction(1), Fraction(2)]]  # non-character output on {0,1}
    sys_msb = FiniteSystem(sys.mu, sys.transition, [(s >> 2) & 1 for s in range(8)])
    C = transfer_matrix(sys_msb, phi, basis, nsq)
    direct = observed_statistic(sys_msb, phi, [0, 0], [0, 2])
    via = contraction_value(
        C, lambda ab: lagged_tensor_value(sys_msb, basis, [0, 2], list(ab)), [0, 0], range(8))
    assert direct == via


def test_binary_linear_identity_observation_paper_check() -> None:
    # Paper 19.2: output = single Walsh character -> M = 1{w0 = (L^1)^T w1}
    sys = _lfsr_system(3, 0b011)
    basis, _ = walsh_basis(3)
    rows = [sum(((1 if (sys.transition[1 << i] >> j) & 1 else 0) << i) for i in range(3))
            for j in range(3)]
    w1 = 5
    w0 = gf2_apply_transpose(rows, w1, 3)
    assert lagged_tensor_value(sys, basis, [0, 1], [w0, w1], connected=False) == 1
    assert lagged_tensor_value(sys, basis, [0, 1], [w0 ^ 1, w1], connected=False) == 0


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------

def test_gf2_adjoint_property() -> None:
    import random
    rng = random.Random(42)
    for _ in range(100):
        rows = [rng.getrandbits(8) for _ in range(8)]
        v, w = rng.getrandbits(8), rng.getrandbits(8)
        assert ((gf2_apply(rows, v, 8) & w).bit_count() & 1) == \
            ((v & gf2_apply_transpose(rows, w, 8)).bit_count() & 1)


def test_gf2_rows_power() -> None:
    sys = _lfsr_system(4, 0b0011)
    rows = [sum(((1 if (sys.transition[1 << i] >> j) & 1 else 0) << i) for i in range(4))
            for j in range(4)]
    assert gf2_rows_mul(gf2_rows_power(rows, 2, 4), gf2_rows_power(rows, 3, 4), 4) == \
        gf2_rows_power(rows, 5, 4)
    assert gf2_rows_power(rows, 15, 4) == [1 << i for i in range(4)]  # period 15


def test_cyclic_closure_finds_short_relations() -> None:
    rels = cyclic_closure([1, 5], 97, 10)
    assert (5, -1) in rels and (-5, 1) in rels
    assert all((k0 + 5 * k1) % 97 == 0 for k0, k1 in rels)
    assert all(is_irreducible_cyclic(t, [1, 5], 97) for t in rels)


def test_cyclic_closure_order() -> None:
    # 2^5 = 1 mod 31: equal-spacing relation 1+2+4+8+16 = 31 at d=5, but
    # 2 - 2*1 = 0... A=2: k0 + 2 k1 = 0 has (2,-1) at d=2
    res = cyclic_closure_order(2, 31, 5, d_min=2, d_max=5)
    assert res["found"] and res["order"] == 2
    # A=6 mod 31: order of 6 is ... smallest irreducible relation found within H
    res2 = cyclic_closure_order(6, 31, 3, d_min=3, d_max=4)
    assert res2["found"]
    t = res2["witness"]
    d = res2["order"]
    assert sum(t[j] * pow(6, j, 31) for j in range(d)) % 31 == 0


def test_binary_closure_and_irreducibility() -> None:
    sys = _lfsr_system(4, 0b0011)
    rows = [sum(((1 if (sys.transition[1 << i] >> j) & 1 else 0) << i) for i in range(4))
            for j in range(4)]
    rels = binary_closure(rows, 4, 2, 1, 2)
    assert rels
    assert all(a == gf2_apply_transpose(rows, b, 4) for a, b in rels)
    order = binary_closure_order(rows, 4, 1, 2, d_min=2, d_max=3)
    assert order["found"] and order["order"] == 2
    witness = tuple(int(x, 16) for x in order["witness"])
    assert is_irreducible_binary(witness, rows, 4, 1)


def test_closure_enumeration_budget_guard() -> None:
    with pytest.raises(ValueError, match="budget"):
        cyclic_closure([1, 2, 3, 4, 5, 6], 2**31 - 1, 400)


# ---------------------------------------------------------------------------
# kernel + server surface
# ---------------------------------------------------------------------------

def test_kernel_cumulant_roundtrip(kernel: MathKernel) -> None:
    r = kernel.cumulant_compute("samples", 2,
                                columns=[["0", "1", "0", "1"], ["0", "1", "0", "1"]])
    assert r.ok and r.data["result"] == "1/4"
    r2 = kernel.cumulant_compute("cumulant", 2,
                                 values={"0": "1/2", "1": "1/3", "0,1": "1/4"})
    assert r2.ok and r2.data["result"] == "1/12"
    bad = kernel.cumulant_compute("cumulant", 2, values={"0": "1/2"})
    assert not bad.ok


def test_kernel_koopman_surface(kernel: MathKernel) -> None:
    fs = kernel.finite_system_create("uniform", [1, 2, 3, 4, 5, 6, 7, 0])
    assert fs.ok and fs.data["stationary"]
    sid = fs.data["system_id"]
    km = kernel.koopman_matrix(sid, {"kind": "cyclic", "m": 8})
    assert km.ok and km.data["matrix"][1][1] == {"cyclotomic": {"order": 8, "terms": {"1": "1/1"}}}
    vis = kernel.koopman_visibility(sid, {"kind": "cyclic", "m": 8})
    assert vis.ok and vis.data["invisible_modes"] == []
    diag = kernel.koopman_diagnostics(sid, {"kind": "cyclic", "m": 8})
    assert diag.ok and all(v == "1" for v in diag.data["ipr"])
    lag = kernel.koopman_lagged(sid, {"kind": "cyclic", "m": 8}, [0, 1], [1, 7])
    assert lag.ok  # closure k0 + k1 = 0 mod 8 satisfied: raw moment is a root of unity
    obs = kernel.koopman_observed(sid, [["0", "1", "0", "1", "0", "1", "0", "1"]], [0, 0], [0, 1])
    assert obs.ok


def test_kernel_finite_fourier_and_closure(kernel: MathKernel) -> None:
    tp = kernel.finite_fourier_compute("two_point", F=[0, 1, 2, 3, 4], N=5, m=1, a_k=1, b_k=2)
    assert tp.ok and tp.data["value"] == {"cyclotomic": {"order": 5, "terms": {"3": "1/1"}}}
    cs = kernel.closure_search("cyclic", m="97", weight_bound=10, multipliers=["1", "5"])
    assert cs.ok and [5, -1] in cs.data["tuples"]
    co = kernel.closure_search("cyclic_order", a_k="2", m="31", weight_bound=5, d_max=5)
    assert co.ok and co.data["order"] == 2


def test_capabilities_advertise_finite_dynamics(kernel: MathKernel) -> None:
    caps = kernel.capabilities()
    for op in ("cumulant_compute", "finite_system_create", "koopman_matrix",
               "koopman_transfer", "koopman_visibility", "koopman_lagged",
               "koopman_observed", "koopman_diagnostics", "finite_fourier_compute",
               "closure_search"):
        assert op in caps["operations"], op
    assert caps["finite_dynamics"]["bases"] == ["walsh", "cyclic"]
    engines = caps["finite_dynamics"]["engines"]
    assert engines["koopman"][0] == "exact"
    assert engines["koopman"][1].startswith("numeric-")
    assert engines["closure_search"][0] in ("njit", "python")


# ---------------------------------------------------------------------------
# fast paths: njit relations, numeric koopman, vectorized cumulants
# ---------------------------------------------------------------------------

def _python_reference(fn, *args):
    import mathkernel.relations as R
    saved = R.HAVE_NUMBA
    R.HAVE_NUMBA = False
    try:
        return fn(*args)
    finally:
        R.HAVE_NUMBA = saved


def test_cyclic_closure_njit_matches_python() -> None:
    for a_ks, m, h in [([1, 5], 97, 10), ([3, 7, 11, 13], 997, 8),
                       ([1, 2, 4, 8, 16], 31, 10), ([1, 7, 49], 2**31 - 1, 12)]:
        assert set(cyclic_closure(a_ks, m, h)) == \
            set(_python_reference(cyclic_closure, a_ks, m, h))


def test_binary_closure_njit_matches_python() -> None:
    sys = _lfsr_system(4, 0b0011)
    rows = [sum(((1 if (sys.transition[1 << i] >> j) & 1 else 0) << i)
                for i in range(4)) for j in range(4)]
    for d, k, h in [(2, 1, 2), (3, 1, 2), (2, 3, 3), (4, 1, 2)]:
        assert set(binary_closure(rows, 4, d, k, h)) == \
            set(_python_reference(binary_closure, rows, 4, d, k, h))


def test_koopman_numeric_matches_exact() -> None:
    from mathkernel.koopman import (
        koopman_matrix_numeric, lagged_tensor_numeric,
        mode_visibility_numeric, observed_statistic_numeric,
        transfer_matrix_numeric, transport_diagnostics_numeric,
    )
    sys = _lfsr_system(4, 0b0011)
    basis, nsq = walsh_basis(4)
    Qe = koopman_matrix(sys, basis, nsq)
    Qn = koopman_matrix_numeric(sys, basis, nsq, prefer_gpu=False)
    for a in range(16):
        for b in range(16):
            assert abs(complex(Qn[b][a]) - Qe[b][a]) < 1e-12
    ve = mode_visibility(sys, basis, nsq)
    vn = mode_visibility_numeric(sys, basis, nsq, prefer_gpu=False)
    assert all(abs(a - b) < 1e-12 for a, b in zip(ve, vn))
    funcs = [[Fraction(1 if y & 1 else -1) for y in range(16)]]
    Ce = transfer_matrix(sys, funcs, basis, nsq)
    Cn = transfer_matrix_numeric(sys, funcs, basis, nsq, prefer_gpu=False)
    assert all(abs(complex(Cn[0][a]) - Ce[0][a]) < 1e-12 for a in range(16))
    le = lagged_tensor_value(sys, basis, [0, 1], [3, 5], connected=True)
    ln = lagged_tensor_numeric(sys, basis, [0, 1], [3, 5], connected=True,
                               prefer_gpu=False)
    assert abs(ln - le) < 1e-12
    oe = observed_statistic(sys, funcs, [0, 0], [0, 2], connected=True)
    on = observed_statistic_numeric(sys, funcs, [0, 0], [0, 2], connected=True,
                                    prefer_gpu=False)
    assert abs(on - oe) < 1e-12
    de = transport_diagnostics(Qe)
    dn = transport_diagnostics_numeric(Qn, prefer_gpu=False)
    assert all(abs(a - b) < 1e-12 for a, b in zip(de["ipr"], dn["ipr"]))
    assert all(abs(a - b) < 1e-9 for a, b in zip(de["entropy"], dn["entropy"]))


def test_kernel_koopman_numeric_trust(kernel: MathKernel) -> None:
    fs = kernel.finite_system_create("uniform", [1, 2, 3, 4, 5, 6, 7, 0])
    sid = fs.data["system_id"]
    km = kernel.koopman_matrix(sid, {"kind": "cyclic", "m": 8}, exact=False)
    assert km.ok and km.trust.value == "numeric" and not km.data["exact"]
    assert km.engine.startswith("koopman-numeric-")
    entry = km.data["matrix"][1][1]
    z = complex(float(entry["re"]), float(entry["im"]))
    import cmath
    assert abs(z - cmath.exp(2j * cmath.pi / 8)) < 1e-12
    vis = kernel.koopman_visibility(sid, {"kind": "cyclic", "m": 8}, exact=False)
    assert vis.ok and vis.data["invisible_modes"] == []
    diag = kernel.koopman_diagnostics(sid, {"kind": "cyclic", "m": 8}, exact=False)
    assert diag.ok and all(abs(float(v) - 1) < 1e-9 for v in diag.data["ipr"])


def test_cumulants_numeric_vectorized() -> None:
    # float columns take the numpy path and match exact Fraction arithmetic
    cols_f = [[0.5, -1.5, 2.0, 0.25], [1.0, 0.5, -0.5, 2.0]]
    cols_q = [[Fraction(str(v)) for v in col] for col in cols_f]
    mf = block_moments_from_samples(cols_f)
    mq = block_moments_from_samples(cols_q)
    assert set(mf) == set(mq)
    for mask in mq:
        assert abs(mf[mask] - float(mq[mask])) < 1e-15
    # pure-int columns stay exact (Fraction results, no numpy path)
    mi = block_moments_from_samples([[0, 1, 0, 1], [1, 1, 0, 0]])
    assert all(isinstance(v, Fraction) for k, v in mi.items() if k)
    assert joint_cumulant_from_samples([[0, 1, 0, 1], [0, 1, 0, 1]]) == Fraction(1, 4)
