# =============================================================================
# MathKernel - Statistics: exact rational sample moments, order statistics, exact linear
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Statistics: exact rational sample moments, order statistics, exact linear
regression and correlation; numeric inference (t/chi-square tests, confidence
intervals) via mpmath at NUMERIC_HIGH_PRECISION trust — never EXACT.

Numeric sample columns take the vectorized NumPy path; batch statistics over
many columns parallelize across the persistent process pool.
"""
from __future__ import annotations

from fractions import Fraction

from .parallel import process_map, resolve_workers

MAX_SAMPLE = 10_000_000


def _fracs(values: list) -> list[Fraction]:
    if not values:
        raise ValueError("sample must not be empty")
    if len(values) > MAX_SAMPLE:
        raise ValueError(f"sample size {len(values)} exceeds limit {MAX_SAMPLE}")
    return [Fraction(v) for v in values]


def sample_moments(values: list, max_order: int = 4) -> dict:
    """Exact central/raw moments via Fraction. Returns mean, population and
    sample variance, skewness (m3/m2^1.5 as exact numerator/denominator pair
    m2, m3), and central moments up to max_order."""
    xs = _fracs(values)
    if max_order < 2 or max_order > 8:
        raise ValueError("max_order must be between 2 and 8")
    n = len(xs)
    mean = sum(xs, Fraction(0)) / n
    central = [sum((x - mean) ** k for x in xs) / n for k in range(2, max_order + 1)]
    m2 = central[0]
    sample_var = m2 * n / (n - 1) if n > 1 else None
    return {"n": n, "mean": mean, "variance_population": m2,
            "variance_sample": sample_var,
            "central_moments": {k + 2: central[k] for k in range(len(central))}}


def order_statistics(values: list) -> dict:
    """Exact sorted sample, median, and quartiles (linear interpolation kept
    rational)."""
    xs = sorted(_fracs(values))
    n = len(xs)

    def quantile(q: Fraction) -> Fraction:
        pos = q * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    return {"sorted": xs, "min": xs[0], "max": xs[-1],
            "median": quantile(Fraction(1, 2)),
            "q1": quantile(Fraction(1, 4)), "q3": quantile(Fraction(3, 4))}


def linear_regression(x_values: list, y_values: list) -> dict:
    """Exact rational least squares via the normal equations. Returns slope,
    intercept, and exact R² (coefficient of determination)."""
    xs, ys = _fracs(x_values), _fracs(y_values)
    if len(xs) != len(ys):
        raise ValueError("x and y samples must have equal length")
    n = len(xs)
    if n < 2:
        raise ValueError("regression needs at least two points")
    mx = sum(xs, Fraction(0)) / n
    my = sum(ys, Fraction(0)) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("all x values are identical; slope is undefined")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    r_squared = (sxy * sxy) / (sxx * syy) if syy != 0 else None
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared, "n": n}


def correlation(x_values: list, y_values: list) -> dict:
    """Exact Pearson r² and covariance; r itself involves a square root and is
    reported as a high-precision numeric value."""
    xs, ys = _fracs(x_values), _fracs(y_values)
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation needs two equal-length samples of size >= 2")
    n = len(xs)
    mx = sum(xs, Fraction(0)) / n
    my = sum(ys, Fraction(0)) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        raise ValueError("correlation is undefined for a constant sample")
    r_squared = (sxy * sxy) / (sxx * syy)
    import mpmath as mp
    sign = -1 if sxy < 0 else 1
    r_numeric = mp.nstr(sign * mp.sqrt(mp.mpf(r_squared.numerator) / r_squared.denominator), 30)
    return {"r_squared": r_squared, "r": r_numeric,
            "covariance_population": sxy / n}


# --- numeric inference (mpmath; NUMERIC_HIGH_PRECISION) -------------------------

def _t_cdf(t, df):
    """Student-t CDF via the regularized incomplete beta function."""
    import mpmath as mp
    x = df / (df + t * t)
    ib = mp.betainc(df / 2, mp.mpf(1) / 2, 0, x, regularized=True)
    return 1 - ib / 2 if t >= 0 else ib / 2


def t_test_1samp(values: list, mu0, dps: int = 50) -> dict:
    """One-sample t-test against mu0. Numeric evidence only."""
    import mpmath as mp
    mp.mp.dps = dps
    xs = _fracs(values)
    n = len(xs)
    if n < 2:
        raise ValueError("t-test needs at least two observations")
    moments = sample_moments(xs)
    mean = mp.mpf(moments["mean"].numerator) / moments["mean"].denominator
    sv = moments["variance_sample"]
    sd = mp.sqrt(mp.mpf(sv.numerator) / sv.denominator)
    t = (mean - mp.mpf(str(mu0))) / (sd / mp.sqrt(n))
    df = n - 1
    cdf = _t_cdf(t, df)
    p_two_sided = 2 * min(cdf, 1 - cdf)
    return {"t": mp.nstr(t, 30), "df": df, "p_two_sided": mp.nstr(p_two_sided, 30),
            "mean": mp.nstr(mean, 30)}


def chi_square_test(observed: list, expected: list | None = None, dps: int = 50) -> dict:
    """Pearson chi-square goodness-of-fit. Numeric evidence only."""
    import mpmath as mp
    mp.mp.dps = dps
    obs = _fracs(observed)
    k = len(obs)
    if k < 2:
        raise ValueError("chi-square needs at least two categories")
    total = sum(obs, Fraction(0))
    exp = _fracs(expected) if expected is not None else [total / k] * k
    if len(exp) != k or any(e <= 0 for e in exp):
        raise ValueError("expected counts must be positive and match observed length")
    stat = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    stat_mp = mp.mpf(stat.numerator) / stat.denominator
    df = k - 1
    p = mp.gammainc(df / 2, a=stat_mp / 2, regularized=True)  # upper tail
    return {"chi2": mp.nstr(stat_mp, 30), "df": df, "p_value": mp.nstr(p, 30)}


def mean_confidence_interval(values: list, confidence: str = "0.95", dps: int = 50) -> dict:
    """t-based confidence interval for the mean. Numeric evidence only."""
    import mpmath as mp
    mp.mp.dps = dps
    conf = Fraction(confidence)
    if not 0 < conf < 1:
        raise ValueError("confidence must be in (0, 1)")
    xs = _fracs(values)
    n = len(xs)
    if n < 2:
        raise ValueError("confidence interval needs at least two observations")
    moments = sample_moments(xs)
    mean = mp.mpf(moments["mean"].numerator) / moments["mean"].denominator
    sv = moments["variance_sample"]
    se = mp.sqrt(mp.mpf(sv.numerator) / sv.denominator) / mp.sqrt(n)
    alpha = 1 - mp.mpf(conf.numerator) / conf.denominator
    # invert the t CDF by bisection on |t|
    lo, hi = mp.mpf(0), mp.mpf(1000)
    for _ in range(200):
        mid = (lo + hi) / 2
        if 2 * (1 - _t_cdf(mid, n - 1)) > alpha:
            lo = mid
        else:
            hi = mid
    crit = (lo + hi) / 2
    return {"confidence": str(conf), "mean": mp.nstr(mean, 30),
            "lower": mp.nstr(mean - crit * se, 30), "upper": mp.nstr(mean + crit * se, 30),
            "standard_error": mp.nstr(se, 30)}


# --- vectorized/batch paths --------------------------------------------------------

def moments_numpy(values) -> dict:
    """Vectorized float64 moments for large numeric samples (NUMERIC trust)."""
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise ValueError("sample must not be empty")
    mean = float(a.mean())
    centered = a - mean
    m2 = float((centered ** 2).mean())
    m3 = float((centered ** 3).mean())
    m4 = float((centered ** 4).mean())
    return {"n": int(a.size), "mean": mean, "variance_population": m2,
            "variance_sample": float(a.var(ddof=1)) if a.size > 1 else None,
            "skewness": m3 / m2 ** 1.5 if m2 > 0 else 0.0,
            "kurtosis_excess": m4 / m2 ** 2 - 3.0 if m2 > 0 else 0.0}


def _moments_column_job(args) -> dict:
    column, as_float = args
    if as_float:
        return moments_numpy(column)
    m = sample_moments(column)
    return {"n": m["n"], "mean": str(m["mean"]),
            "variance_population": str(m["variance_population"]),
            "variance_sample": str(m["variance_sample"]) if m["variance_sample"] is not None else None}


def batch_moments(columns: list[list], workers: int | None = None,
                  numeric: bool = False) -> list[dict]:
    """Moments for many sample columns across the process pool. numeric=True
    takes the vectorized float64 path (NUMERIC); otherwise exact Fractions."""
    resolved = resolve_workers(workers)
    return process_map(_moments_column_job, [(c, numeric) for c in columns],
                       workers=resolved)
