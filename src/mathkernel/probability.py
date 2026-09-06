# =============================================================================
# MathKernel - Exact-first probability: discrete random variables with rational pmfs,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact-first probability: discrete random variables with rational pmfs,
Bayes, and finite Markov chains with exact rational stationary distributions
and hitting times.

Exact core uses Fraction throughout — expectation, variance, covariance,
Bayes posteriors, stationary distributions, and hitting times are exact
rational numbers. A sympy.stats bridge covers continuous/symbolic
distributions at SYMBOLIC trust; sampling is NUMERIC evidence.
"""
from __future__ import annotations

import random
from fractions import Fraction

MAX_SUPPORT = 100_000
MAX_MARKOV_STATES = 512
MAX_SAMPLES = 1_000_000


def _frac(value) -> Fraction:
    f = Fraction(value)
    return f


class DiscreteRV:
    """A discrete random variable with exact rational probabilities."""

    def __init__(self, values: list, probabilities: list):
        if len(values) != len(probabilities):
            raise ValueError("values and probabilities must have equal length")
        if not values:
            raise ValueError("a random variable needs at least one outcome")
        if len(values) > MAX_SUPPORT:
            raise ValueError(f"support size {len(values)} exceeds limit {MAX_SUPPORT}")
        probs = [_frac(p) for p in probabilities]
        if any(p < 0 for p in probs):
            raise ValueError("probabilities must be nonnegative")
        total = sum(probs, Fraction(0))
        if total != 1:
            raise ValueError(f"probabilities must sum to 1, got {total}")
        self.values = [_frac(v) for v in values]
        self.probs = probs

    def expectation(self, power: int = 1) -> Fraction:
        if power < 1 or power > 16:
            raise ValueError("power must be between 1 and 16")
        return sum(p * v**power for v, p in zip(self.values, self.probs))

    def variance(self) -> Fraction:
        mu = self.expectation()
        return sum(p * (v - mu) ** 2 for v, p in zip(self.values, self.probs))

    def sample(self, n: int, seed: int | None = None) -> list[Fraction]:
        if n < 1 or n > MAX_SAMPLES:
            raise ValueError(f"sample count must be between 1 and {MAX_SAMPLES}")
        rng = random.Random(seed)
        cumulative = []
        acc = Fraction(0)
        for p in self.probs:
            acc += p
            cumulative.append(acc)
        out = []
        for _ in range(n):
            r = Fraction(rng.random()).limit_denominator(10**12)
            for v, c in zip(self.values, cumulative):
                if r < c:
                    out.append(v)
                    break
            else:
                out.append(self.values[-1])
        return out


def covariance(x_values: list, y_values: list, joint_probabilities: list) -> Fraction:
    """Exact covariance from a joint pmf over paired outcomes."""
    if not (len(x_values) == len(y_values) == len(joint_probabilities)):
        raise ValueError("x values, y values, and probabilities must have equal length")
    probs = [_frac(p) for p in joint_probabilities]
    if sum(probs, Fraction(0)) != 1:
        raise ValueError("joint probabilities must sum to 1")
    xs, ys = [_frac(v) for v in x_values], [_frac(v) for v in y_values]
    ex = sum(p * v for v, p in zip(xs, probs))
    ey = sum(p * v for v, p in zip(ys, probs))
    return sum(p * (a - ex) * (b - ey) for a, b, p in zip(xs, ys, probs))


def bayes(prior: list, likelihood: list) -> list[Fraction]:
    """Exact Bayes: posterior_i ∝ likelihood_i * prior_i."""
    if len(prior) != len(likelihood):
        raise ValueError("prior and likelihood must have equal length")
    priors, likes = [_frac(p) for p in prior], [_frac(l) for l in likelihood]
    if any(p < 0 for p in priors) or sum(priors, Fraction(0)) != 1:
        raise ValueError("prior must be a probability distribution")
    if any(l < 0 for l in likes):
        raise ValueError("likelihoods must be nonnegative")
    unnorm = [l * p for l, p in zip(likes, priors)]
    total = sum(unnorm, Fraction(0))
    if total == 0:
        raise ValueError("evidence has zero probability under every hypothesis")
    return [u / total for u in unnorm]


# --- finite Markov chains (exact rational linear algebra) ----------------------

def _to_fraction_matrix(matrix: list[list]) -> list[list[Fraction]]:
    n = len(matrix)
    if n == 0 or n > MAX_MARKOV_STATES:
        raise ValueError(f"state count must be between 1 and {MAX_MARKOV_STATES}")
    if any(len(row) != n for row in matrix):
        raise ValueError("transition matrix must be square")
    return [[_frac(x) for x in row] for row in matrix]


def _exact_solve(a: list[list[Fraction]], b: list[Fraction]) -> list[Fraction]:
    """Fraction-free Gaussian elimination with partial pivoting. Exact."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if m[pivot][col] == 0:
            raise ValueError("system is singular")
        m[col], m[pivot] = m[pivot], m[col]
        inv = 1 / m[col][col]
        m[col] = [x * inv for x in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                factor = m[r][col]
                m[r] = [x - factor * y for x, y in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def stationary_distribution(matrix: list[list]) -> list[Fraction]:
    """Exact stationary distribution π with πP = π, sum π = 1, for a rational
    row-stochastic transition matrix. Requires a unique stationary distribution."""
    p = _to_fraction_matrix(matrix)
    n = len(p)
    for i, row in enumerate(p):
        if any(x < 0 for x in row) or sum(row, Fraction(0)) != 1:
            raise ValueError(f"row {i} is not a probability distribution")
    # Solve (P^T - I)π = 0 with the last equation replaced by sum π = 1.
    a = [[p[j][i] - (1 if i == j else 0) for j in range(n)] for i in range(n)]
    b = [Fraction(0)] * n
    a[n - 1] = [Fraction(1)] * n
    b[n - 1] = Fraction(1)
    return _exact_solve(a, b)


def hitting_times(matrix: list[list], targets: list[int]) -> list[Fraction | None]:
    """Exact expected hitting times to the target set. h_t = 0 on targets;
    h_i = 1 + Σ_j P_ij h_j elsewhere. None marks states that never reach the
    target set (detected as a singular/negative system)."""
    p = _to_fraction_matrix(matrix)
    n = len(p)
    target_set = set(targets)
    if not target_set or any(t < 0 or t >= n for t in target_set):
        raise ValueError("targets must be valid state indices")
    free = [i for i in range(n) if i not in target_set]
    index = {s: k for k, s in enumerate(free)}
    a = [[(Fraction(1) if i == j else Fraction(0)) - p[s][t]
          for j, t in enumerate(free)] for i, s in enumerate(free)]
    b = [Fraction(1)] * len(free)
    solved: dict[int, Fraction] = {}
    if free:
        try:
            sol = _exact_solve(a, b)
        except ValueError:
            sol = None
        if sol is not None and all(x >= 0 for x in sol):
            solved = dict(zip(free, sol))
    return [Fraction(0) if i in target_set else solved.get(i) for i in range(n)]


# --- sympy.stats bridge (SYMBOLIC trust) ----------------------------------------

_DISTRIBUTIONS = {
    "normal": "Normal", "exponential": "Exponential", "uniform": "Uniform",
    "binomial": "Binomial", "poisson": "Poisson", "bernoulli": "Bernoulli",
    "geometric": "Geometric", "beta": "Beta", "gamma": "Gamma",
    "cauchy": "Cauchy", "exponentialpower": None,
}


def sympy_stats_query(distribution: str, parameters: list[str], query: str,
                      point: str | None = None):
    """Bridge to sympy.stats for continuous/symbolic distributions.

    query: expectation | variance | std | density (needs point) | cdf (needs point).
    Returns a SymPy object. SYMBOLIC trust at the facade."""
    from sympy import stats as st
    from .engines import SymPyEngine
    from .parser import parse_math

    name = _DISTRIBUTIONS.get(distribution.lower())
    if not name:
        raise ValueError(f"unsupported distribution: {distribution}; "
                         f"choose from {sorted(k for k, v in _DISTRIBUTIONS.items() if v)}")
    converter = SymPyEngine()
    params = [converter.to_sympy(parse_math(p)) for p in parameters]
    ctor = getattr(st, name)
    rv = ctor("X", *params)
    if query == "expectation":
        return st.E(rv)
    if query == "variance":
        return st.variance(rv)
    if query == "std":
        return st.std(rv)
    if query == "density":
        if point is None:
            raise ValueError("density query requires a point")
        return st.density(rv)(converter.to_sympy(parse_math(point)))
    if query == "cdf":
        if point is None:
            raise ValueError("cdf query requires a point")
        return st.cdf(rv)(converter.to_sympy(parse_math(point)))
    raise ValueError(f"unsupported query: {query}; choose expectation/variance/std/density/cdf")
