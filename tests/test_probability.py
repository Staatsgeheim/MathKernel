# =============================================================================
# MathKernel - Exact probability: discrete RVs, Bayes, Markov chains, sympy.stats bridge."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact probability: discrete RVs, Bayes, Markov chains, sympy.stats bridge."""

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.probability import DiscreteRV, bayes, covariance, hitting_times, stationary_distribution


def test_rv_exact_moments():
    rv = DiscreteRV(["0", "1", "2"], ["1/4", "1/2", "1/4"])
    assert str(rv.expectation()) == "1"
    assert str(rv.expectation(2)) == "3/2"
    assert str(rv.variance()) == "1/2"


def test_rv_rejects_bad_pmf():
    with pytest.raises(ValueError, match="sum to 1"):
        DiscreteRV(["0", "1"], ["1/2", "1/3"])
    with pytest.raises(ValueError, match="nonnegative"):
        DiscreteRV(["0", "1"], ["3/2", "-1/2"])


def test_bayes_exact():
    posterior = bayes(["1/2", "1/2"], ["9/10", "1/5"])
    assert [str(p) for p in posterior] == ["9/11", "2/11"]


def test_covariance_exact():
    # fair coin pairs: X, Y independent -> cov 0
    cov = covariance(["0", "0", "1", "1"], ["0", "1", "0", "1"],
                     ["1/4", "1/4", "1/4", "1/4"])
    assert cov == 0
    cov2 = covariance(["0", "1"], ["0", "1"], ["1/2", "1/2"])
    assert str(cov2) == "1/4"


def test_stationary_distribution_exact():
    pi = stationary_distribution([["1/2", "1/2"], ["1/4", "3/4"]])
    assert [str(p) for p in pi] == ["1/3", "2/3"]


def test_stationary_rejects_non_stochastic():
    with pytest.raises(ValueError, match="probability distribution"):
        stationary_distribution([["1/2", "1/2"], ["1/4", "1/2"]])


def test_hitting_times_exact():
    h = hitting_times([["1/2", "1/2"], ["1/4", "3/4"]], [0])
    assert h[0] == 0 and str(h[1]) == "4"


def test_kernel_facade_probability():
    k = MathKernel()
    rv = k.prob_rv_create(["0", "1", "2"], ["1/4", "1/2", "1/4"])
    assert rv.ok and rv.trust == TrustLevel.EXACT
    rid = rv.data["rv_id"]
    assert k.prob_expectation(rid).data["value"] == "1"
    assert k.prob_variance(rid).data["value"] == "1/2"
    assert k.prob_expectation(rid, 2).data["value"] == "3/2"
    assert not k.prob_expectation("rv_nope").ok


def test_kernel_markov_via_matrix():
    k = MathKernel()
    m = k.matrix_create([["1/2", "1/2"], ["1/4", "3/4"]])
    mid = m.data["matrix_id"]
    s = k.prob_markov_stationary(mid)
    assert s.data["stationary"] == ["1/3", "2/3"] and s.trust == TrustLevel.EXACT
    h = k.prob_markov_hitting_time(mid, [0])
    assert h.data["hitting_times"] == ["0", "4"]


def test_kernel_markov_rejects_numeric_matrix():
    k = MathKernel()
    m = k.matrix_create([["0.5", "0.5"], ["0.25", "0.75"]])
    r = k.prob_markov_stationary(m.data["matrix_id"])
    assert not r.ok and "exact" in r.errors[0]


def test_kernel_sample_numeric_trust_seeded():
    k = MathKernel()
    rid = k.prob_rv_create(["0", "1"], ["1/2", "1/2"]).data["rv_id"]
    a = k.prob_sample(rid, 8, seed=7)
    b = k.prob_sample(rid, 8, seed=7)
    assert a.trust == TrustLevel.NUMERIC
    assert a.data["samples"] == b.data["samples"]  # deterministic under a seed


def test_kernel_sympy_stats_bridge():
    k = MathKernel()
    r = k.prob_distribution("normal", ["0", "1"], "variance")
    assert r.ok and r.data["result"] == "1" and r.trust == TrustLevel.SYMBOLIC
    r2 = k.prob_distribution("poisson", ["3"], "expectation")
    assert r2.data["result"] == "3"
    bad = k.prob_distribution("cauchy-not-real", ["0"], "expectation")
    assert not bad.ok
