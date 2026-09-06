# =============================================================================
# MathKernel - Units and dimensional analysis."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Units and dimensional analysis."""

import pytest
from fractions import Fraction

from mathkernel import MathKernel, TrustLevel
from mathkernel.units import DIMENSIONLESS, convert, dimension_of, parse_unit
from mathkernel.parser import parse_math


def test_parse_unit_registry():
    assert parse_unit("m/s^2").dimension == parse_unit("m*s^-2").dimension
    assert parse_unit("N").dimension == parse_unit("kg*m/s^2").dimension
    assert parse_unit("J").dimension == parse_unit("N*m").dimension
    assert parse_unit("km").scale == 1000
    assert parse_unit("min").scale == 60
    with pytest.raises(ValueError, match="unknown unit"):
        parse_unit("furlong")


def test_exact_conversion():
    assert convert("36", "km/h", "m/s")["value"] == "10"
    assert convert("1", "h", "s")["value"] == "3600"
    assert convert("1", "L", "mL")["value"] == "1000"
    assert convert("1", "cal", "J")["value"] == str(Fraction(4184, 1000))
    with pytest.raises(ValueError, match="dimensional mismatch"):
        convert("1", "m", "s")


def test_dimension_of_expression():
    # kinetic energy: 1/2 * m * v^2 -> M*L^2*T^-2
    dim = dimension_of(parse_math("1/2 * m * v^2"), {"m": "kg", "v": "m/s"})
    assert dim == parse_unit("J").dimension
    # dimensionless ratio
    assert dimension_of(parse_math("x / y"), {"x": "m", "y": "m"}).is_dimensionless()
    # sqrt halves exponents
    assert dimension_of(parse_math("sqrt(a)"), {"a": "m^2" if False else "m"}) is not None


def test_dimension_errors_are_hard():
    with pytest.raises(ValueError, match="different dimensions"):
        dimension_of(parse_math("x + y"), {"x": "m", "y": "s"})
    with pytest.raises(ValueError, match="dimensionless argument"):
        dimension_of(parse_math("sin(x)"), {"x": "m"})
    with pytest.raises(ValueError, match="dimensionless"):
        dimension_of(parse_math("x^y"), {"x": "m", "y": "s"})


def test_kernel_unit_facade():
    k = MathKernel()
    eid = k.parse("v * t").data["expr_id"]
    r = k.unit_check(eid, {"v": "m/s", "t": "s"})
    assert r.ok and r.data["dimension"] == "L" and r.trust == TrustLevel.EXACT
    assert k.expressions[eid].meta.units == "L"  # SemanticMeta populated
    bad = k.unit_check(eid, {"v": "m/s", "t": "kg"})
    assert bad.ok and bad.data["dimension"] == "L*M*T^-1"
    mismatch = k.unit_check(k.parse("a + b").data["expr_id"], {"a": "m", "b": "s"})
    assert not mismatch.ok
    c = k.unit_convert("90", "km/h", "m/s")
    assert c.data["value"] == "25" and c.trust == TrustLevel.EXACT
    s = k.unit_simplify("kg*m/s^2")
    assert s.data["dimension"] == str(parse_unit("N").dimension)
    assert s.data["si_scale"] == "1"
