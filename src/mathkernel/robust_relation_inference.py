# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Phase 10: finite quadratic bounds, learned nuisances, and long-run covariance.

Numerical research APIs. Finite bounds have explicit model assumptions; estimated
covariances and nuisance spaces do not acquire finite-sample guarantees merely
by being passed to a quadratic test. See ROBUST_RELATION_INFERENCE_PHASE10.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, log, sqrt
from typing import Callable, Sequence

import numpy as np


def _data(value, name="scores", minimum_rows=2):
    x = np.asarray(value, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < minimum_rows or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite nonempty matrix with >= {minimum_rows} rows")
    return x


def _spectrum(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("information eigenvalues must be a finite nonnegative vector")
    return a


def _errors(alpha, beta):
    if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
        raise ValueError("require alpha,beta > 0 and alpha + beta < 1")


def _first_integer(predicate, minimum=2):
    high = minimum
    while not predicate(high):
        high *= 2
        if high > 2**60:
            raise OverflowError("sample bound exceeds the supported numerical search range; not an impossibility certificate")
    low = minimum - 1
    while high - low > 1:
        middle = (low + high) // 2
        if predicate(middle):
            high = middle
        else:
            low = middle
    return high


@dataclass(frozen=True, slots=True)
class QuadraticMinimaxBounds:
    relation_dimension: int
    epsilon: float
    alpha: float
    beta: float
    inverse_spectrum_l2: float
    weakest_information: float
    gaussian_necessary_samples: int | float
    gaussian_sufficient_samples: int | float
    iid_u_sufficient_samples: int | float
    alternative_covariance_envelope: float
    blind_direction: bool
    scope: str = "finite Gaussian sequence lower/upper; separate iid covariance-envelope U-statistic upper"


def quadratic_minimax_bounds(information_eigenvalues, epsilon, *, alpha=.05, beta=.1,
                             alternative_covariance_envelope=1.0):
    """Matching spectrum-dependent rates in the Gaussian sequence experiment.

    Observe Y_j ~ N(sqrt(N*lambda_j)*theta_j, 1) independently and test
    theta=0 against ||theta|| >= epsilon. Necessary and sufficient N scale as
    sqrt(sum(lambda_j**-2))/epsilon**2 for fixed errors. The iid U-statistic
    upper instead assumes E0[Z]=0, Cov0[Z]=I, Etheta[Z]=sqrt(lambda)*theta,
    and Covtheta[Z] <= kappa*I throughout the alternatives. A plug-in estimate
    of kappa is NOT a certified envelope. Zero eigenvalues give impossibility.
    """
    a = _spectrum(information_eigenvalues)
    _errors(alpha, beta)
    kappa = float(alternative_covariance_envelope)
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("epsilon and covariance envelope must be finite and positive")
    if np.any(a == 0):
        return QuadraticMinimaxBounds(len(a), epsilon, alpha, beta, inf, 0., inf, inf, inf, kappa, True)
    w = 1. / a
    s = float(w @ w)
    largest = float(w.max())
    if not np.isfinite(s):
        raise ValueError("inverse information spectrum overflows float64")
    e2 = epsilon**2
    lower = sqrt(2*s*log(1+4*(1-alpha-beta)**2))/e2
    x, y = log(1/alpha), log(1/beta)
    c0 = 2*sqrt(s*x) + 2*largest*x
    gauss = _first_integer(lambda n: n*e2 >= c0 + 2*sqrt((s+2*n*e2*largest)*y), 1)

    def iid_ok(n):
        threshold = sqrt(2*s*(1-alpha)/(n*(n-1)*alpha))
        variance = 2*kappa*kappa*s/(n*(n-1)) + 4*kappa*e2*largest/n
        return e2 > threshold and (e2-threshold)**2 >= (1-beta)/beta*variance

    return QuadraticMinimaxBounds(len(a), epsilon, alpha, beta, sqrt(s), float(a.min()),
                                 ceil(lower), gauss, _first_integer(iid_ok), kappa, False)


def gaussian_quadratic_test(sequence, information_eigenvalues, sample_size, *, alpha=.05):
    """Finite Gaussian weighted-square test, calibrated by a Chernoff bound.

    ``sequence`` is sqrt(N) times a nominally whitened sample mean ONLY when
    that quantity is exactly Gaussian. For other models calibration is local
    asymptotic, not finite. Returns statistic in latent squared-norm units.
    """
    a = _spectrum(information_eigenvalues)
    z = np.asarray(sequence, dtype=float)
    if z.shape != a.shape or not np.all(np.isfinite(z)) or np.any(a <= 0):
        raise ValueError("sequence must match a strictly positive information spectrum")
    if isinstance(sample_size, bool) or int(sample_size) != sample_size or sample_size < 1 or not 0 < alpha < 1:
        raise ValueError("positive integer sample size and 0 < alpha < 1 required")
    w = 1/a
    value = float(w @ (z*z-1)/sample_size)
    x = log(1/alpha)
    threshold = float((2*sqrt(float(w@w)*x)+2*w.max()*x)/sample_size)
    return {"statistic": value, "threshold": threshold, "reject": value > threshold,
            "calibration": "finite Gaussian Chernoff bound", "alpha": alpha}


def quadratic_u_test(scores, information_eigenvalues, *, alpha=.05):
    """Unbiased pairwise quadratic statistic, O(Nr) memory/time rather than N^2.

    Calibration requires iid scores with KNOWN null mean zero and covariance I;
    scores fitted on these rows, or dependent rows, do not satisfy this contract.
    No fourth-moment assumption is needed: the kernel uses distinct pairs.
    """
    z = _data(scores)
    a = _spectrum(information_eigenvalues)
    if len(a) != z.shape[1] or np.any(a <= 0) or not 0 < alpha < 1:
        raise ValueError("positive spectrum matching score dimension and 0 < alpha < 1 required")
    n = len(z)
    w = 1/a
    total = z.sum(axis=0)
    u = float(w @ (total*total - np.sum(z*z, axis=0))/(n*(n-1)))
    variance = float(2*(w@w)/(n*(n-1)))
    threshold = sqrt(variance*(1-alpha)/alpha)
    p_bound = 1. if u <= 0 else variance/(variance+u*u)
    return {"statistic": u, "threshold": threshold, "reject": u > threshold,
            "p_value_upper_bound": p_bound, "null_variance": variance,
            "sample_size": n, "calibration": "finite iid known-covariance Cantelli"}


@dataclass(frozen=True, slots=True)
class LongRunCovarianceEstimate:
    covariance: np.ndarray
    bandwidth: int
    prewhitened: bool
    ar_matrix: np.ndarray
    spectral_radius: float
    recoloring_condition: float
    bandwidth_was_capped: bool
    pilot_ar1: np.ndarray
    sample_size: int
    warnings: tuple[str, ...]


def _bartlett(centered, bandwidth):
    n = len(centered)
    s = centered.T @ centered/n
    for h in range(1, bandwidth+1):
        g = centered[h:].T @ centered[:-h]/n
        s += (1-h/(bandwidth+1))*(g+g.T)
    return (s+s.T)/2


def prewhitened_long_run_covariance(scores, *, bandwidth=None, prewhiten=True,
                                   maximum_bandwidth=None):
    """VAR(1) prewhitening, automatic Bartlett bandwidth, and recoloring.

    Automatic bandwidth uses the maximum marginal AR(1) persistence functional
    4*rho^2/(1-rho^2)^2 and ceil(1.1447*(n*functional)^(1/3)), with minimum 1.
    This maximum-over-marginals variant is an explicit implementation choice,
    not a claim of exact Andrews multivariate optimal bandwidth. Unstable VARs
    and singular pilot designs fail closed. Warnings expose near-unit roots,
    bandwidth capping, and weak effective sample sizes. Requires stationary
    weak dependence for consistency, not arbitrary genomic nonstationarity.
    """
    z = _data(scores, minimum_rows=12)
    n, r = z.shape
    if n <= 3*r+3:
        raise ValueError("too few observations relative to covariance dimension")
    z = z-z.mean(axis=0)
    a = np.zeros((r,r))
    warnings = []
    if prewhiten:
        coefficients, _, rank, _ = np.linalg.lstsq(z[:-1], z[1:], rcond=None)
        if rank < r:
            raise ValueError("rank-deficient VAR pilot; reduce to a declared full-rank score space")
        a = coefficients.T
        radius = float(np.max(np.abs(np.linalg.eigvals(a))))
        condition = float(np.linalg.cond(np.eye(r)-a))
        if radius >= 1-1e-8 or condition > 1e8:
            raise ValueError("unstable or ill-conditioned VAR recoloring")
        residuals = z[1:] - z[:-1]@coefficients
        if radius > .95:
            warnings.append("near-unit persistence: small-sample calibration may be poor")
    else:
        radius, condition, residuals = 0., 1., z
    residuals = residuals-residuals.mean(axis=0)
    m = len(residuals)
    denominator = np.sum(residuals[:-1]**2, axis=0)
    if np.any(denominator <= np.finfo(float).tiny):
        raise ValueError("zero-variance residual direction")
    rho = np.sum(residuals[1:]*residuals[:-1], axis=0)/denominator
    clipped = np.clip(rho, -.97, .97)
    if not np.array_equal(clipped, rho):
        warnings.append("pilot AR coefficient clipped for bandwidth only")
    cap = min(m-1, max(1, m//4)) if maximum_bandwidth is None else maximum_bandwidth
    if isinstance(cap, bool) or int(cap) != cap or not 0 <= cap < m:
        raise ValueError("maximum bandwidth must be an integer between 0 and residual count - 1")
    if bandwidth is None:
        functional = float(np.max(4*clipped**2/(1-clipped**2)**2))
        proposed = max(1, ceil(1.1447*(m*functional)**(1/3)))
        b = min(proposed, int(cap))
        capped = b < proposed
    else:
        if isinstance(bandwidth, bool) or int(bandwidth) != bandwidth or not 0 <= bandwidth < m:
            raise ValueError("bandwidth must be an integer between 0 and residual count - 1")
        b, capped = int(bandwidth), False
    if capped:
        warnings.append("automatic bandwidth capped; inspect sensitivity")
    s = _bartlett(residuals, b)
    if prewhiten:
        recolor = np.linalg.solve(np.eye(r)-a, np.eye(r))
        s = recolor@s@recolor.T
    s = (s+s.T)/2
    if np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError("estimated covariance is singular or indefinite; no silent pseudoinverse")
    if n*(1-radius)/(1+radius) < 40:
        warnings.append("persistence implies fewer than 40 AR-equivalent observations")
    return LongRunCovarianceEstimate(s, b, prewhiten, a, radius, condition, capped, rho, n, tuple(warnings))


def quadratic_moment_test(scores, *, covariance=None, alpha=.05):
    """Asymptotic full-rank Wald test with empirical or supplied long-run covariance.

    Singular covariance is rejected, so a score-mean component cannot disappear
    silently through a pseudoinverse. Covariance is for sqrt(N)*sample_mean.
    """
    z = _data(scores)
    if not 0 < alpha < 1:
        raise ValueError("require 0 < alpha < 1")
    s = np.atleast_2d(np.cov(z, rowvar=False, ddof=1)) if covariance is None else np.asarray(covariance, dtype=float)
    r = z.shape[1]
    if s.shape != (r,r) or not np.all(np.isfinite(s)) or not np.allclose(s,s.T,atol=1e-12,rtol=1e-10):
        raise ValueError("covariance must be a finite symmetric matrix matching score dimension")
    if np.linalg.eigvalsh(s).min() <= 0:
        raise ValueError("full-rank positive covariance required")
    mean = z.mean(axis=0)
    value = float(len(z)*mean@np.linalg.solve(s,mean))
    from scipy.stats import chi2
    p = float(chi2.sf(value,r))
    return {"statistic":value,"p_value":p,"reject":p<alpha,"degrees_of_freedom":r,
            "sample_size":len(z),"calibration":"asymptotic chi-square; covariance consistency required"}


def relation_folds(sample_size, *, n_splits=5, seed=20260906, groups=None, contiguous=False):
    """Make out-of-fold labels; groups are never split across training/evaluation.

    Contiguous folds alone do not remove dependence. Pair them with ``gap`` in
    cross-fitting and justify an appropriate mixing or independent-cluster CLT.
    """
    if any(isinstance(v,bool) or int(v)!=v for v in (sample_size,n_splits)) or not 2 <= n_splits <= sample_size:
        raise ValueError("integer sample size and 2 <= n_splits <= sample size required")
    if groups is not None and contiguous:
        raise ValueError("choose either group folds or contiguous folds")
    rng = np.random.default_rng(seed)
    if groups is not None:
        g = np.asarray(groups)
        if g.shape != (sample_size,):
            raise ValueError("one group label per observation required")
        unique, inverse = np.unique(g, return_inverse=True)
        if len(unique) < n_splits:
            raise ValueError("fewer groups than folds")
        allocation = np.empty(len(unique), dtype=int)
        allocation[rng.permutation(len(unique))] = np.arange(len(unique)) % n_splits
        return allocation[inverse]
    indices = np.arange(sample_size) if contiguous else rng.permutation(sample_size)
    folds = np.empty(sample_size,dtype=int)
    for k, indices_k in enumerate(np.array_split(indices,n_splits)):
        folds[indices_k] = k
    return folds


def _folds(fold_ids, n, gap=0):
    f = np.asarray(fold_ids)
    if f.shape != (n,) or not np.issubdtype(f.dtype,np.integer) or len(np.unique(f)) < 2:
        raise ValueError("at least two integer fold labels, one per observation, required")
    if isinstance(gap,bool) or int(gap)!=gap or gap < 0:
        raise ValueError("gap must be a nonnegative integer")
    for k in np.unique(f):
        test = np.flatnonzero(f==k)
        train_mask = f!=k
        if gap:
            # Range updates avoid O(N * test-size) distance matrices.
            excluded = np.zeros(n+1,dtype=np.int64)
            np.add.at(excluded,np.maximum(0,test-gap),1)
            np.add.at(excluded,np.minimum(n,test+gap+1),-1)
            train_mask &= np.cumsum(excluded[:-1])==0
        train = np.flatnonzero(train_mask)
        if len(train) < 4:
            raise ValueError("fold/gap leaves too few training observations")
        yield int(k), train, test


@dataclass(frozen=True, slots=True)
class CrossFittedRelations:
    scores: np.ndarray
    fold_ids: np.ndarray
    diagnostics: tuple[dict, ...]
    method: str
    assumptions: str


def crossfit_nuisance_projection(target_scores, nuisance_scores, fold_ids, *, gap=0):
    """Learn target/nuisance covariance projections on other folds only.

    Training covariances are centered; evaluation scores are target - nuisance B
    WITHOUT subtracting the training target mean (which would erase the signal).
    Candidate nuisance scores must have known zero baseline means. Learning the
    span's coefficients is supported; learning arbitrary unknown score functions
    requires a separate model. This operation alone is not a DML validity theorem.
    """
    t, u = _data(target_scores), _data(nuisance_scores,"nuisance_scores")
    if len(t)!=len(u):
        raise ValueError("target and nuisance row counts must match")
    output, diagnostics = np.empty_like(t), []
    for k, train, test in _folds(fold_ids,len(t),gap):
        tc, uc = t[train]-t[train].mean(0), u[train]-u[train].mean(0)
        coefficient, _, rank, _ = np.linalg.lstsq(uc,tc,rcond=None)
        output[test] = t[test]-u[test]@coefficient
        diagnostics.append({"fold":k,"train_rows":len(train),"test_rows":len(test),
                            "nuisance_rank":int(rank),"coefficients":coefficient.tolist()})
    return CrossFittedRelations(output,np.asarray(fold_ids).copy(),tuple(diagnostics),
        "out-of-fold empirical nuisance projection",
        "known-zero-mean nuisance candidates; consistent projection; suitable sampling CLT")


def _ols_predict(train_x, train_y, test_x):
    design = np.column_stack((np.ones(len(train_x)),train_x))
    beta = np.linalg.lstsq(design,train_y,rcond=None)[0]
    return np.column_stack((np.ones(len(test_x)),test_x))@beta


def crossfit_residual_relations(first, second, controls, fold_ids, *,
                               learner: Callable | None=None, gap=0):
    """Cross-fitted Neyman-orthogonal residual-product relation moments.

    psi=(A-m_A(X)) tensor (B-m_B(X)); target is the unconditional residual
    cross-moment, not conditional independence or causality. With true conditional
    means, both nuisance Gateaux derivatives have expectation zero. Standard
    root-N inference additionally needs L2 consistency and product error o(N^-1/2),
    moment/nondegeneracy conditions and an appropriate sampling CLT. Default
    learner is unpenalized OLS on the PROVIDED features plus an intercept.
    """
    a,b,x = _data(first,"first"),_data(second,"second"),_data(controls,"controls")
    if len(a)!=len(b) or len(a)!=len(x):
        raise ValueError("all input row counts must match")
    predict = _ols_predict if learner is None else learner
    output = np.empty((len(a),a.shape[1]*b.shape[1]))
    diagnostics = []
    for k, train, test in _folds(fold_ids,len(a),gap):
        pa = np.asarray(predict(x[train],a[train],x[test]),dtype=float)
        pb = np.asarray(predict(x[train],b[train],x[test]),dtype=float)
        if pa.shape!=a[test].shape or pb.shape!=b[test].shape or not np.all(np.isfinite(pa)) or not np.all(np.isfinite(pb)):
            raise ValueError("nuisance learner returned wrong-shaped or nonfinite predictions")
        ra, rb = a[test]-pa,b[test]-pb
        output[test] = np.einsum('ni,nj->nij',ra,rb).reshape(len(test),-1)
        diagnostics.append({"fold":k,"train_rows":len(train),"test_rows":len(test),
                            "first_residual_mse":float(np.mean(ra*ra)),
                            "second_residual_mse":float(np.mean(rb*rb))})
    return CrossFittedRelations(output,np.asarray(fold_ids).copy(),tuple(diagnostics),
        "cross-fitted orthogonal residual-product moments",
        "consistent conditional means; nuisance error product o(N^-1/2); suitable sampling CLT")
