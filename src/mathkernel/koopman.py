# =============================================================================
# MathKernel - Exact finite-dimensional Koopman / observation-transfer machinery
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact finite-dimensional Koopman / observation-transfer machinery.

A finite partially observed system (X, mu, T, O) is enumerated explicitly:
states are indices 0..n-1, mu is an exact probability vector, T and O are
successor/output index maps. All linear algebra is done in the exact value
type of the chosen basis (Fraction for Walsh, CyclotomicNumber for cyclic
characters), so visibility zeros and closure cancellations are exact.

Bases are unnormalized: each mode carries norm_sq[alpha] = <psi_a, psi_a>,
and coefficient extraction divides by it. This keeps the Walsh basis
rational (entries +/-1, norm 2^r) and character bases cyclotomic.

Conventions: inner product linear in the first argument,
<f, g> = sum_x mu(x) f(x) conj(g(x)); Koopman U f = f ∘ T;
Q[b][a] = <U psi_a, psi_b> / norm_sq[b].
"""
from __future__ import annotations

from fractions import Fraction
from math import log

from .cumulants import connected_statistic
from .finite_fourier import CyclotomicNumber


def _conj(v):
    return v.conjugate() if hasattr(v, "conjugate") else v


_GPU_OK: bool | None = None


def _xp(prefer_gpu: bool = True):
    """Array namespace: CuPy on GPU when usable, else NumPy.

    CuPy is probed with a real matmul — the package can import fine while
    its CUDA backend DLLs (cublas) are missing, so import alone is not
    sufficient evidence. The probe result is cached process-wide.
    """
    global _GPU_OK
    if prefer_gpu:
        if _GPU_OK is None:
            try:
                import cupy as cp
                a = cp.ones((2, 2)) @ cp.ones((2, 2))
                _GPU_OK = float(a.sum()) == 8.0
            except Exception:
                _GPU_OK = False
        if _GPU_OK:
            import cupy as cp
            return cp
    import numpy as np
    return np


def _to_host(a, xp):
    if xp.__name__ == "cupy":
        return a.get()
    return a


def _complex(v):
    if isinstance(v, CyclotomicNumber):
        return v.to_complex()
    if isinstance(v, Fraction):
        return float(v)
    return complex(v)


def _basis_array(basis, xp):
    return xp.asarray([[_complex(v) for v in row] for row in basis],
                      dtype=complex)


def _mu_array(sys: FiniteSystem, xp):
    return xp.asarray([float(m) for m in sys.mu], dtype=float)


def koopman_matrix_numeric(sys: FiniteSystem, basis: list[list],
                           norm_sq: list[Fraction], prefer_gpu: bool = True):
    """Numeric Q via one matmul: Q = (B_T @ diag(mu) @ B^H)^T / norm."""
    xp = _xp(prefer_gpu)
    B = _basis_array(basis, xp)
    BT = B[:, xp.asarray(sys.transition)]
    M = B.conj() * _mu_array(sys, xp)[None, :]
    Q = (BT @ M.T).T / xp.asarray([float(v) for v in norm_sq])[:, None]
    return _to_host(Q, xp).tolist()


def transfer_matrix_numeric(sys: FiniteSystem, output_functions: list[list],
                            basis: list[list], norm_sq: list[Fraction],
                            prefer_gpu: bool = True):
    """Numeric observation-transfer C[h][a] = <phi_h ∘ O, psi_a> / norm_a."""
    xp = _xp(prefer_gpu)
    n_out = sys.n_outputs
    for phi in output_functions:
        if len(phi) != n_out:
            raise ValueError(
                f"output functions must be defined on all {n_out} outputs")
    F = xp.asarray([[_complex(phi[y]) for y in sys.observation]
                    for phi in output_functions], dtype=complex)
    B = _basis_array(basis, xp)
    M = B.conj() * _mu_array(sys, xp)[None, :]
    C = (F @ M.T) / xp.asarray([float(v) for v in norm_sq])[None, :]
    return _to_host(C, xp).tolist()


def mode_visibility_numeric(sys: FiniteSystem, basis: list[list],
                            norm_sq: list[Fraction],
                            prefer_gpu: bool = True) -> list[float]:
    """Numeric rho_O(a) = ||P_O psi_a||^2 / ||psi_a||^2."""
    xp = _xp(prefer_gpu)
    B = _basis_array(basis, xp)
    mu = _mu_array(sys, xp)
    obs = xp.asarray(sys.observation)
    n_out = sys.n_outputs
    W = xp.zeros((sys.n, n_out))
    W[xp.arange(sys.n), obs] = 1.0
    G = (B * mu[None, :]) @ W                     # G[a, y] = sum_{O(x)=y} mu B
    nu = mu @ W                                   # output masses
    proj = G[:, obs] / nu[obs][None, :]           # P_O psi_a at each x
    num = (xp.abs(proj) ** 2) @ mu
    rho = num / xp.asarray([float(v) for v in norm_sq])
    return [float(v) for v in _to_host(rho, xp)]


def _advanced_indices(sys: FiniteSystem, tau: list[int], xp):
    """Index arrays: adv[j][i] = T^{tau_j}(support[i])."""
    sup = xp.asarray(sys.support)
    out = []
    for t in tau:
        idx = sup
        for _ in range(t):
            idx = xp.asarray(sys.transition)[idx]
        out.append(idx)
    return out


def lagged_tensor_numeric(sys: FiniteSystem, basis: list[list], tau: list[int],
                          alphas: list[int], connected: bool = True,
                          prefer_gpu: bool = True):
    """Numeric lagged state tensor entry (raw or connected)."""
    xp = _xp(prefer_gpu)
    d = len(tau)
    if len(alphas) != d or d < 1:
        raise ValueError("tau and alphas must be nonempty and equal length")
    B = _basis_array(basis, xp)
    adv = _advanced_indices(sys, tau, xp)
    Z = [B[alphas[j], adv[j]] for j in range(d)]
    mu_sup = _mu_array(sys, xp)[xp.asarray(sys.support)]

    def raw(block: tuple[int, ...]):
        prod = Z[block[0]]
        for j in block[1:]:
            prod = prod * Z[j]
        return complex((prod * mu_sup).sum())

    if not connected or d == 1:
        return raw(tuple(range(d)))
    return complex(connected_statistic(raw, d))


def observed_statistic_numeric(sys: FiniteSystem, output_functions: list[list],
                               h_tuple: list[int], tau: list[int],
                               connected: bool = True,
                               prefer_gpu: bool = True):
    """Numeric connected observed statistic K_h^(d)(tau)."""
    xp = _xp(prefer_gpu)
    d = len(tau)
    if len(h_tuple) != d or d < 1:
        raise ValueError("tau and h_tuple must be nonempty and equal length")
    adv = _advanced_indices(sys, tau, xp)
    obs = xp.asarray(sys.observation)
    cols = [xp.asarray([_complex(v) for v in output_functions[h_tuple[j]]],
                       dtype=complex)[obs[adv[j]]] for j in range(d)]
    mu_sup = _mu_array(sys, xp)[xp.asarray(sys.support)]

    def raw(block: tuple[int, ...]):
        prod = cols[block[0]]
        for j in block[1:]:
            prod = prod * cols[j]
        return complex((prod * mu_sup).sum())

    if not connected or d == 1:
        return raw(tuple(range(d)))
    return complex(connected_statistic(raw, d))


def transport_diagnostics_numeric(Q, prefer_gpu: bool = True) -> dict:
    """Numeric IPR / transport entropy from a Q matrix (list or array)."""
    xp = _xp(prefer_gpu)
    A = xp.asarray(Q, dtype=complex)
    p = xp.abs(A) ** 2
    ipr = (p ** 2).sum(axis=0)
    with np_errstate():
        plogp = xp.where(p > 0, p * xp.log(xp.where(p > 0, p, 1.0)), 0.0)
    entropy = -plogp.sum(axis=0)
    return {"ipr": [float(v) for v in _to_host(ipr, xp)],
            "entropy": [float(v) for v in _to_host(entropy, xp)],
            "modes": int(A.shape[0])}


def np_errstate():
    import numpy as np
    return np.errstate(divide="ignore", invalid="ignore")


class FiniteSystem:
    """Finite partially observed deterministic system, exactly enumerated."""

    def __init__(self, mu: list[Fraction], transition: list[int],
                 observation: list[int] | None = None):
        n = len(mu)
        if n == 0 or len(transition) != n:
            raise ValueError("mu and transition must be nonempty and equal length")
        if any(not 0 <= t < n for t in transition):
            raise ValueError("transition targets must be state indices")
        total = sum(mu, Fraction(0))
        if total != 1 or any(m < 0 for m in mu):
            raise ValueError("mu must be a probability vector (nonnegative, sums to 1)")
        self.n = n
        self.mu = list(mu)
        self.transition = list(transition)
        self.observation = list(observation) if observation is not None else list(range(n))
        if len(self.observation) != n:
            raise ValueError("observation must map every state")
        self.n_outputs = max(self.observation) + 1
        self.support = [x for x in range(n) if mu[x] > 0]

    @staticmethod
    def uniform(n: int, transition: list[int], observation: list[int] | None = None,
                support: list[int] | None = None) -> FiniteSystem:
        """Uniform measure on `support` (default: all n states)."""
        if n < 1:
            raise ValueError("n must be positive")
        if support is None:
            mu = [Fraction(1, n)] * n
        else:
            if not support:
                raise ValueError("support must be nonempty")
            mu = [Fraction(0)] * n
            for x in support:
                mu[x] = Fraction(1, len(support))
        return FiniteSystem(mu, transition, observation)

    def is_stationary(self) -> bool:
        """mu(T^{-1} A) = mu(A) for all A: checked exactly on singletons."""
        pre = [Fraction(0)] * self.n
        for x in self.support:
            pre[self.transition[x]] += self.mu[x]
        return all(pre[y] == self.mu[y] for y in range(self.n))

    def iterate(self, x: int, steps: int) -> int:
        for _ in range(steps):
            x = self.transition[x]
        return x


def walsh_basis(r: int) -> tuple[list[list[int]], list[Fraction]]:
    """Unnormalized Walsh characters on GF(2)^r: psi_w(S) = (-1)^{w·S}.

    States are integers 0..2^r-1; modes indexed by mask w. The returned
    norm_sq is with respect to the UNIFORM probability inner product
    (<psi, psi> = 1); callers using a non-uniform measure must recompute
    norms and should note the basis is orthogonal only under uniformity.
    """
    if not 1 <= r <= 16:
        raise ValueError("walsh basis dimension must be 1..16")
    n = 1 << r
    basis = [[1 - 2 * ((w & s).bit_count() & 1) for s in range(n)] for w in range(n)]
    return basis, [Fraction(1)] * n


def cyclic_basis(m: int) -> tuple[list[list[CyclotomicNumber]], list[Fraction]]:
    """Character basis of Z_M: psi_k(x) = zeta_M^{kx}; orthonormal under uniform."""
    if not 2 <= m <= 512:
        raise ValueError("cyclic basis order must be 2..512")
    basis = [[CyclotomicNumber(m, {(k * x) % m: 1}) for x in range(m)] for k in range(m)]
    return basis, [Fraction(1)] * m


def inner_product(sys: FiniteSystem, f: list, g: list):
    """<f, g>_mu = sum_x mu(x) f(x) conj(g(x)) over the support."""
    acc = None
    for x in sys.support:
        term = sys.mu[x] * f[x] * _conj(g[x])
        acc = term if acc is None else acc + term
    return acc if acc is not None else Fraction(0)


def transfer_matrix(sys: FiniteSystem, output_functions: list[list],
                    basis: list[list], norm_sq: list[Fraction]) -> list[list]:
    """Observation-transfer coefficients C[h][a] = <phi_h ∘ O, psi_a> / norm_sq[a].

    output_functions[h][y] gives phi_h on the output set; rows of C are the
    state-basis expansions of the pulled-back observables.
    """
    n_out = sys.n_outputs
    for phi in output_functions:
        if len(phi) != n_out:
            raise ValueError(f"output functions must be defined on all {n_out} outputs")
    pulled = [[phi[sys.observation[x]] for x in range(sys.n)] for phi in output_functions]
    return [[inner_product(sys, f, basis[a]) / norm_sq[a] for a in range(len(basis))]
            for f in pulled]


def koopman_matrix(sys: FiniteSystem, basis: list[list],
                   norm_sq: list[Fraction]) -> list[list]:
    """Koopman transport matrix Q[b][a] = <U psi_a, psi_b> / norm_sq[b]."""
    transported = [[basis[a][sys.transition[x]] for x in range(sys.n)]
                   for a in range(len(basis))]
    return [[inner_product(sys, transported[a], basis[b]) / norm_sq[b]
             for a in range(len(basis))] for b in range(len(basis))]


def observation_projection(sys: FiniteSystem, g: list) -> list:
    """Conditional-average projection P_O onto the observation subspace.

    (P_O g)(x) = (1/nu(O(x))) sum_{x': O(x')=O(x)} mu(x') g(x').
    """
    nu = [Fraction(0)] * sys.n_outputs
    acc = [None] * sys.n_outputs
    for x in range(sys.n):
        y = sys.observation[x]
        nu[y] += sys.mu[x]
        v = sys.mu[x] * g[x]
        acc[y] = v if acc[y] is None else acc[y] + v
    return [acc[sys.observation[x]] / nu[sys.observation[x]] for x in range(sys.n)]


def mode_visibility(sys: FiniteSystem, basis: list[list],
                    norm_sq: list[Fraction]) -> list[Fraction]:
    """rho_O(a) = ||P_O psi_a||^2 / ||psi_a||^2 in [0, 1], exact.

    rho = 0 iff the mode is orthogonal to everything the observation can
    express; rho = 1 iff the mode is fully determined by the observation.
    """
    out = []
    for a in range(len(basis)):
        proj = observation_projection(sys, basis[a])
        num = inner_product(sys, proj, proj)
        val = num / norm_sq[a]
        out.append(val.to_rational() if isinstance(val, CyclotomicNumber) else Fraction(val))
    return out


def lagged_tensor_value(sys: FiniteSystem, basis: list[list], tau: list[int],
                        alphas: list[int], connected: bool = True):
    """Lagged state tensor entry J_tau(alpha) (raw or connected).

    J = E[prod_j psi_{a_j}(T^{tau_j} S)]; the connected form applies exact
    cumulant partition subtraction over position subtuples.
    """
    d = len(tau)
    if len(alphas) != d or d < 1:
        raise ValueError("tau and alphas must be nonempty and equal length")
    advanced = [[sys.iterate(x, t) for x in sys.support] for t in tau]

    def raw(block: tuple[int, ...]):
        acc = None
        for i, x0 in enumerate(sys.support):
            prod = basis[alphas[block[0]]][advanced[block[0]][i]]
            for j in block[1:]:
                prod = prod * basis[alphas[j]][advanced[j][i]]
            term = sys.mu[x0] * prod
            acc = term if acc is None else acc + term
        return acc if acc is not None else Fraction(0)

    if not connected or d == 1:
        return raw(tuple(range(d)))
    return connected_statistic(raw, d)


def observed_statistic(sys: FiniteSystem, output_functions: list[list],
                       h_tuple: list[int], tau: list[int], connected: bool = True):
    """Connected observed statistic K_h^(d)(tau) by direct enumeration.

    Z_j = phi_{h_j}(O(T^{tau_j} S)); exact cumulant over the support.
    """
    d = len(tau)
    if len(h_tuple) != d or d < 1:
        raise ValueError("tau and h_tuple must be nonempty and equal length")
    advanced = [[sys.iterate(x, t) for x in sys.support] for t in tau]
    cols = [[output_functions[h_tuple[j]][sys.observation[advanced[j][i]]]
             for i in range(len(sys.support))] for j in range(d)]

    def raw(block: tuple[int, ...]):
        acc = None
        for i in range(len(sys.support)):
            prod = cols[block[0]][i]
            for j in block[1:]:
                prod = prod * cols[j][i]
            term = sys.mu[sys.support[i]] * prod
            acc = term if acc is None else acc + term
        return acc if acc is not None else Fraction(0)

    if not connected or d == 1:
        return raw(tuple(range(d)))
    return connected_statistic(raw, d)


def contraction_value(C: list[list], Jc, h_tuple: list[int], alphas_range: range):
    """Observed statistic via the basis contraction sum_h C * Jc (Prop. 7.10).

    Jc is a callback alpha_tuple -> connected state tensor entry; the sum
    runs over all alpha in alphas_range^d. Exact whenever C and Jc are.
    """
    d = len(h_tuple)
    total = None
    for alpha in _tuples(alphas_range, d):
        prod = C[h_tuple[0]][alpha[0]]
        for j in range(1, d):
            prod = prod * C[h_tuple[j]][alpha[j]]
        if prod:
            term = prod * Jc(alpha)
            total = term if total is None else total + term
    return total if total is not None else Fraction(0)


def _tuples(r: range, d: int):
    if d == 1:
        yield from ((a,) for a in r)
        return
    for rest in _tuples(r, d - 1):
        for a in r:
            yield (a,) + rest


def static_cumulant_tensor(sys: FiniteSystem, basis: list[list], d: int) -> dict:
    """Equal-time cumulant tensor kappa(beta) = Cum(psi_{b_0}, ..., psi_{b_{d-1}}).

    Returned sparse: only nonzero entries, keyed by beta tuples.
    """
    out = {}
    for beta in _tuples(range(len(basis)), d):
        v = lagged_tensor_value(sys, basis, [0] * d, list(beta), connected=True)
        if v:
            out[beta] = v
    return out


def path_tensor_value(Q: list[list], kappa: dict, tau: list[int], alphas: list[int]):
    """Connected state tensor via the Koopman path representation.

    Jc_tau(alpha) = sum_beta prod_j (Q^{tau_j})[b_j][a_j] * kappa(beta).
    """
    d = len(tau)
    powers = [_matrix_power(Q, t) for t in tau]
    total = None
    for beta, kv in kappa.items():
        prod = powers[0][beta[0]][alphas[0]]
        for j in range(1, d):
            prod = prod * powers[j][beta[j]][alphas[j]]
        if prod:
            term = prod * kv
            total = term if total is None else total + term
    return total if total is not None else Fraction(0)


def _matrix_power(Q: list[list], k: int):
    n = len(Q)
    result = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    base = Q
    while k:
        if k & 1:
            result = _matmul(result, base)
        base = _matmul(base, base)
        k >>= 1
    return result


def _matmul(A: list[list], B: list[list]) -> list[list]:
    n = len(A)
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = None
            for k in range(n):
                if A[i][k] and B[k][j]:
                    term = A[i][k] * B[k][j]
                    acc = term if acc is None else acc + term
            out[i][j] = acc if acc is not None else Fraction(0)
    return out


def transport_diagnostics(Q: list[list]) -> dict:
    """Per-column spectral spreading: exact IPR = sum_b |Q[b][a]|^4, entropy.

    IPR is exact in the value type (rational for Walsh bases); the transport
    entropy involves logarithms and is therefore numeric.
    """
    n = len(Q)
    ipr, entropy = [], []
    for a in range(n):
        col = [Q[b][a] for b in range(n)]
        p = []
        for v in col:
            q = v.norm_sq() if isinstance(v, CyclotomicNumber) else v * v
            p.append(q)
        ipr_val = None
        for q in p:
            term = q * q
            ipr_val = term if ipr_val is None else ipr_val + term
        if isinstance(ipr_val, CyclotomicNumber):
            ipr_val = ipr_val.to_rational()
        ipr.append(ipr_val)
        h = 0.0
        for q in p:
            f = float(q.to_rational()) if isinstance(q, CyclotomicNumber) else float(q)
            if f > 0:
                h -= f * log(f)
        entropy.append(h)
    return {"ipr": ipr, "entropy": entropy, "modes": n}
