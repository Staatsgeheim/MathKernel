# =============================================================================
# MathKernel - viz generic transform tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest

from mathkernel_viz import transforms as T


def test_lag_embedding_basic():
    rows = T.lag_embedding([0, 1, 2, 3, 4], lags=(0, 1, 2))
    assert rows == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]


def test_lag_embedding_differences():
    rows = T.lag_embedding([0, 1, 3, 6], lags=(0, 1), differences=True)
    assert rows == [[1.0, 2.0], [2.0, 3.0]]


def test_lag_embedding_stride_offset():
    rows = T.lag_embedding(list(range(10)), lags=(0, 2), stride=3, offset=1)
    assert rows[0] == [1.0, 3.0]
    assert all((r[0] - 1) % 3 == 0 for r in rows)


def test_lag_embedding_validation():
    with pytest.raises(ValueError):
        T.lag_embedding([1, 2], lags=())
    with pytest.raises(ValueError):
        T.lag_embedding([1, 2], lags=(-1,))
    assert T.lag_embedding([1, 2], lags=(0, 5)) == []


def test_normalize_centered_unit_span():
    out = T.normalize([10, 20, 30])
    assert min(out) == -0.5 and max(out) == 0.5
    assert out[1] == 0.0
    assert T.normalize([]) == []


def test_downsample_deterministic():
    vals = list(range(1000))
    out = T.downsample(vals, limit=100)
    assert len(out) <= 100
    assert out == T.downsample(vals, limit=100)
    assert T.downsample([1, 2], limit=10) == [1, 2]


def test_histogram_counts():
    edges, counts = T.histogram_counts([0.0, 0.5, 0.5, 1.0], bins=2,
                                       value_range=(0.0, 1.0))
    assert counts == [1, 3]
    assert edges == [0.0, 0.5]
    assert T.histogram_counts([], bins=4) == ([], [])


def test_linear_form_mod1():
    cols = [[0.0, 0.5], [0.25, 0.25]]
    out = T.linear_form_mod1(cols, [1.0, 2.0])
    assert out == [0.5, 0.0]
    with pytest.raises(ValueError):
        T.linear_form_mod1(cols, [1.0])


def test_sequence_stats():
    s = T.sequence_stats([1.0, 2.0, 3.0])
    assert s["count"] == 3
    assert s["mean"] == 2.0
    assert s["min"] == 1.0 and s["max"] == 3.0
    assert s["distinct"] == 3
    assert T.sequence_stats([]) == {"count": 0}
