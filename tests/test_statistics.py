# =============================================================================
# MathKernel - Statistics: exact moments/regression/correlation, numeric inference, batch."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Statistics: exact moments/regression/correlation, numeric inference, batch."""

import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.statistics import (batch_moments, linear_regression, moments_numpy,
                                   order_statistics, sample_moments)


def test_exact_moments():
    m = sample_moments(["1", "2", "3", "4"])
    assert str(m["mean"]) == "5/2"
    assert str(m["variance_population"]) == "5/4"
    assert str(m["variance_sample"]) == "5/3"
    assert str(m["central_moments"][4]) == "41/16"


def test_order_statistics_exact():
    o = order_statistics(["3", "1", "4", "1", "5"])
    assert [str(x) for x in o["sorted"]] == ["1", "1", "3", "4", "5"]
    assert str(o["median"]) == "3"
    assert str(o["q1"]) == "1" and str(o["q3"]) == "4"


def test_regression_exact():
    r = linear_regression(["1", "2", "3"], ["2", "4", "6"])
    assert str(r["slope"]) == "2" and str(r["intercept"]) == "0"
    assert str(r["r_squared"]) == "1"
    with pytest.raises(ValueError, match="identical"):
        linear_regression(["2", "2"], ["1", "2"])


def test_kernel_stats_trust_levels():
    k = MathKernel()
    assert k.stats_moments(["1", "2", "3"]).trust == TrustLevel.EXACT
    assert k.stats_order(["1", "2"]).trust == TrustLevel.EXACT
    assert k.stats_regression(["1", "2"], ["1", "2"]).trust == TrustLevel.EXACT
    corr = k.stats_correlation(["1", "2", "3"], ["2", "4", "5"])
    assert corr.data["r_squared"] == "27/28"
    assert corr.trust == TrustLevel.NUMERIC_HIGH_PRECISION  # r involves a sqrt
    assert k.stats_ttest(["1", "2", "3", "4"], "2").trust == TrustLevel.NUMERIC_HIGH_PRECISION
    assert k.stats_chi2(["10", "10"]).trust == TrustLevel.NUMERIC_HIGH_PRECISION
    assert k.stats_confidence_interval(["1", "2", "3"]).trust == TrustLevel.NUMERIC_HIGH_PRECISION


def test_ttest_known_value():
    k = MathKernel()
    # sample mean exactly mu0 -> t = 0, p = 1
    r = k.stats_ttest(["1", "2", "3", "4", "5"], "3")
    assert r.data["t"] == "0.0" and r.data["p_two_sided"] == "1.0"


def test_chi2_known_value():
    k = MathKernel()
    r = k.stats_chi2(["10", "12", "8"], ["10", "10", "10"])
    assert r.data["chi2"] == "0.8" and r.data["df"] == 2
    assert float(r.data["p_value"]) == pytest.approx(0.6703200460, rel=1e-9)


def test_moments_numpy_matches_exact():
    values = [str(i) for i in range(1, 101)]
    exact = sample_moments(values)
    numeric = moments_numpy([float(i) for i in range(1, 101)])
    assert numeric["mean"] == pytest.approx(float(exact["mean"]), rel=1e-12)
    assert numeric["variance_population"] == pytest.approx(
        float(exact["variance_population"]), rel=1e-12)


def test_batch_moments_parallel():
    k = MathKernel()
    columns = [["1", "2", "3"], ["4", "5"], ["10", "20", "30"], ["7"], ["2", "4"]]
    r = k.stats_batch_moments(columns, workers=2)
    assert r.ok and r.trust == TrustLevel.EXACT
    means = [c["mean"] for c in r.data["results"]]
    assert means == ["2", "9/2", "20", "7", "3"]
    rn = k.stats_batch_moments(columns, workers=2, numeric=True)
    assert rn.trust == TrustLevel.NUMERIC
    assert rn.data["results"][0]["mean"] == pytest.approx(2.0)
