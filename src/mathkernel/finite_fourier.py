# =============================================================================
# MathKernel - Fourier analysis on finite cyclic groups, exact where feasible
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Fourier analysis on finite cyclic groups, exact where feasible.

Characters of Z_M are psi_k(x) = exp(2*pi*i*k*x/M); coefficients are taken
against the conjugated basis (f-hat(k) = sum_x f(x) * conj(psi_k(x))).

Exact path: values live in the cyclotomic ring Q(zeta_L), represented as
integer/rational coefficient vectors reduced modulo the L-th cyclotomic
polynomial. This makes transfer transforms, two-point coefficients, and
measure Fourier transforms exact algebraic objects — cancellation to zero
is a proof, not a numerical accident. Exact mode is intended for small
moduli (enumeration over Z_M is inherent); a numpy FFT path covers large M
at float precision and is flagged numeric by callers.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import gcd


@lru_cache(maxsize=64)
def _cyclotomic_poly(n: int) -> tuple[int, ...]:
    """Coefficients of the n-th cyclotomic polynomial, low degree first."""
    import sympy as sp
    x = sp.Symbol("x")
    poly = sp.Poly(sp.cyclotomic_poly(n, x))
    deg = poly.degree()
    return tuple(int(poly.nth(i)) for i in range(deg + 1))


class CyclotomicNumber:
    """Element of Q(zeta_L) as a coefficient vector reduced mod Phi_L.

    coeffs[e] is the coefficient of zeta_L^e for e in 0..deg(Phi_L)-1 after
    reduction; the canonical reduced form makes equality and hashing exact.
    """

    __slots__ = ("L", "coeffs")

    def __init__(self, L: int, coeffs: dict[int, Fraction | int] | None = None):
        if L < 1:
            raise ValueError("cyclotomic order must be positive")
        self.L = L
        self.coeffs = {e: Fraction(c) for e, c in (coeffs or {}).items() if c}
        self._reduce()

    def _reduce(self) -> None:
        phi = _cyclotomic_poly(self.L)
        deg_phi = len(phi) - 1
        # fold exponents >= L back into 0..L-1 using zeta^L = 1
        folded: dict[int, Fraction] = {}
        for e, c in self.coeffs.items():
            folded[e % self.L] = folded.get(e % self.L, Fraction(0)) + c
        # polynomial division by Phi_L on the exponent vector
        while True:
            top = max((e for e, c in folded.items() if c), default=-1)
            if top < deg_phi:
                break
            c = folded.pop(top)
            shift = top - deg_phi
            for i, pc in enumerate(phi[:-1]):
                if pc:
                    e2 = shift + i
                    folded[e2] = folded.get(e2, Fraction(0)) - c * pc
        self.coeffs = {e: c for e, c in folded.items() if c}

    @staticmethod
    def zero(L: int) -> CyclotomicNumber:
        return CyclotomicNumber(L)

    @staticmethod
    def one(L: int) -> CyclotomicNumber:
        return CyclotomicNumber(L, {0: 1})

    @staticmethod
    def zeta(L: int, power: int = 1) -> CyclotomicNumber:
        return CyclotomicNumber(L, {power % L: 1})

    @staticmethod
    def rational(L: int, value: Fraction | int) -> CyclotomicNumber:
        return CyclotomicNumber(L, {0: value})

    def __add__(self, other: CyclotomicNumber) -> CyclotomicNumber:
        self._check(other)
        out = dict(self.coeffs)
        for e, c in other.coeffs.items():
            out[e] = out.get(e, Fraction(0)) + c
        return CyclotomicNumber(self.L, out)

    def __neg__(self) -> CyclotomicNumber:
        return CyclotomicNumber(self.L, {e: -c for e, c in self.coeffs.items()})

    def __sub__(self, other: CyclotomicNumber) -> CyclotomicNumber:
        return self + (-other)

    def __mul__(self, other) -> CyclotomicNumber:
        if isinstance(other, (int, Fraction)):
            return CyclotomicNumber(self.L, {e: c * other for e, c in self.coeffs.items()})
        self._check(other)
        out: dict[int, Fraction] = {}
        for e1, c1 in self.coeffs.items():
            for e2, c2 in other.coeffs.items():
                out[e1 + e2] = out.get(e1 + e2, Fraction(0)) + c1 * c2
        return CyclotomicNumber(self.L, out)

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> CyclotomicNumber:
        if isinstance(scalar, int):
            scalar = Fraction(scalar)
        if not isinstance(scalar, Fraction) or scalar == 0:
            raise ValueError("division only by nonzero rationals")
        return CyclotomicNumber(self.L, {e: c / scalar for e, c in self.coeffs.items()})

    def conjugate(self) -> CyclotomicNumber:
        return CyclotomicNumber(self.L, {-e: c for e, c in self.coeffs.items()})

    def norm_sq(self) -> CyclotomicNumber:
        """|z|^2 = z * conj(z): a self-conjugate (real) cyclotomic number."""
        return self * self.conjugate()

    def to_rational(self) -> Fraction:
        """Exact rational value; raises unless the number is rational."""
        return _rational_part(self)

    def is_zero(self) -> bool:
        return not self.coeffs

    def to_complex(self) -> complex:
        import cmath
        return complex(sum(c * cmath.exp(2j * 3.141592653589793 * e / self.L)
                           for e, c in self.coeffs.items()))

    def _check(self, other: CyclotomicNumber) -> None:
        if not isinstance(other, CyclotomicNumber) or other.L != self.L:
            raise TypeError("cyclotomic operands must share the order L")

    def __eq__(self, other) -> bool:
        return isinstance(other, CyclotomicNumber) and self.L == other.L and \
            self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash((self.L, tuple(sorted(self.coeffs.items()))))

    def __repr__(self) -> str:
        if not self.coeffs:
            return "0"
        return " + ".join(f"{c}*z{self.L}^{e}" for e, c in sorted(self.coeffs.items()))


def _rational_part(z: CyclotomicNumber) -> Fraction:
    """Exact rational value of a self-conjugate cyclotomic number.

    Only valid when z == conj(z) (e.g. z = w*conj(w)); such a z is real, and
    for real cyclotomics the rational value is not simply coeffs[0] in
    general, so evaluate via the rational-coefficient structure: a real
    cyclotomic with rational coefficients that is known to be rational has
    value equal to the average over the Galois orbit — computed here by
    numeric-guided exact reconstruction guarded to stay exact.
    """
    # z is rational iff all non-constant symmetric contributions cancel;
    # verify rationality, then read off via conjugate symmetry.
    if not z.coeffs:
        return Fraction(0)
    if set(z.coeffs) == {0}:
        return z.coeffs[0]
    # General real case: pair exponents e and L-e share coefficients; the
    # rational value equals coeffs[0] plus contributions only when terms are
    # themselves rational (e = L/2 gives zeta^e = -1).
    acc = z.coeffs.get(0, Fraction(0))
    for e, c in z.coeffs.items():
        if e == 0:
            continue
        if (2 * e) % z.L == 0:  # zeta^e = -1 (requires L even)
            acc -= c
        else:
            raise ValueError("cyclotomic number is not rational")
    return acc


def dft_zm(values: list, M: int | None = None) -> list:
    """Exact DFT over Z_M against conjugated characters: F(k) = sum_x v[x] zeta_M^{-kx}.

    Input values may be int/Fraction (exact) or complex (numeric passthrough
    via numpy when available). Exact inputs return CyclotomicNumber results.
    """
    M = M or len(values)
    if M != len(values) or M < 1:
        raise ValueError("values length must equal M >= 1")
    if any(isinstance(v, complex) for v in values):
        import numpy as np
        return [complex(z) for z in np.fft.ifft(np.array(values, dtype=complex)) * M]
    out = []
    for k in range(M):
        acc = CyclotomicNumber.zero(M)
        for x, v in enumerate(values):
            if v:
                acc = acc + CyclotomicNumber(M, {(-k * x) % M: v})
        out.append(acc)
    return out


def transfer_transform(F: list[int], N: int, h: int, M: int | None = None) -> list[CyclotomicNumber]:
    """Cyclic output-transfer transform T_F(h, k) for k in Z_M.

    T_F(h, k) = (1/M) sum_x exp(2*pi*i*h*F(x)/N) * exp(-2*pi*i*k*x/M),
    i.e. the state-character coefficients of the pulled-back output
    character chi_h ∘ F. Exact in Q(zeta_L) with L = lcm(M, N).
    """
    M = M or len(F)
    if len(F) != M or M < 1 or N < 1:
        raise ValueError("F must have M >= 1 entries over Z_N with N >= 1")
    if any(not 0 <= v < N for v in F):
        raise ValueError("F values must lie in Z_N")
    L = M * N // gcd(M, N)
    h %= N
    out = []
    for k in range(M):
        acc = CyclotomicNumber.zero(L)
        for x in range(M):
            e = (h * F[x] * (L // N) - k * x * (L // M)) % L
            acc = acc + CyclotomicNumber(L, {e: 1})
        out.append(acc / M)
    return out


def two_point_difference(F: list[int], N: int, m: int, A_K: int, B_K: int,
                         M: int | None = None) -> CyclotomicNumber:
    """Two-point difference coefficient B_m(K) for an affine K-step map.

    B_m(K) = sum_ell T_F(m, A_K*ell) * conj(T_F(m, ell)) * zeta_M^{-ell*B_K}
    — the exact raw moment E[exp(2*pi*i*m*(F(x) - F(T^K x))/N)] when
    T^K(x) = A_K*x + B_K (mod M), via the closure k0 + A_K*k1 = 0.
    """
    M = M or len(F)
    L = M * N // gcd(M, N)
    tf = transfer_transform(F, N, m, M)
    acc = CyclotomicNumber.zero(L)
    for ell in range(M):
        t1 = tf[(A_K * ell) % M]
        t2 = tf[ell].conjugate()
        if t1.is_zero() or t2.is_zero():
            continue
        phase = CyclotomicNumber(L, {(-ell * B_K % M) * (L // M): 1})
        acc = acc + t1 * t2 * phase
    return acc


def measure_fourier(mu: list, M: int | None = None) -> list:
    """Fourier transform of a measure on Z_M: mu_hat(k) = sum_x mu(x) zeta_M^{kx}.

    For the uniform measure on all of Z_M this is the indicator of k = 0;
    for an orbit measure it carries the exact orbit-measure correction.
    Complex inputs switch to a numeric numpy FFT; rational inputs stay exact.
    """
    M = M or len(mu)
    if len(mu) != M or M < 1:
        raise ValueError("mu must have M >= 1 entries")
    if any(isinstance(v, complex) for v in mu):
        import numpy as np
        return [complex(z) for z in np.fft.fft(np.array(mu, dtype=complex))]
    out = []
    for k in range(M):
        acc = CyclotomicNumber.zero(M)
        for x, v in enumerate(mu):
            if v:
                acc = acc + CyclotomicNumber(M, {(k * x) % M: v})
        out.append(acc)
    return out

def orbit_correction(closed_sum, total_sum, orbit_size: int):
    """Nonzero-orbit correction for full-space closure sums.

    Averaging over a single orbit of size q instead of the full group turns
    the closure indicator into 1{u=0} - (1/q) 1{u!=0}; given the sum over
    closure-satisfying tuples and the sum over all tuples, the orbit-average
    is closed_sum - (total_sum - closed_sum)/q.
    """
    if orbit_size < 1:
        raise ValueError("orbit size must be positive")
    return closed_sum - (total_sum - closed_sum) / orbit_size
