# =============================================================================
# MathKernel - Exact polynomial algebra: Groebner bases, division, resultants, factorization,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact polynomial algebra: Groebner bases, division, resultants, factorization,
ideal membership, and batch parallelism."""

import sympy as sp

from mathkernel import MathKernel, TrustLevel
from mathkernel.polynomial import (discriminant_poly, factor_poly, groebner_basis,
                                   ideal_membership, poly_divide, resultant_poly)

x, y = sp.symbols("x y")


# --- module level -----------------------------------------------------------------

def test_groebner_basis_module():
    basis = groebner_basis([x**2 - y**2, x - y], [x, y])
    assert len(basis) == 1 and sp.expand(basis[0] - (x - y)) == 0


def test_poly_divide_module():
    quotients, remainder = poly_divide(x**2 - y**2, [x - y], [x, y])
    assert sp.expand(quotients[0] - (x + y)) == 0 and remainder == 0


def test_ideal_membership_module():
    member, remainder = ideal_membership(x**2 - y**2, [x - y, x + y], [x, y])
    assert member and remainder == 0
    member2, _ = ideal_membership(x**2 + y**2, [x - y], [x, y])
    assert not member2


def test_resultant_discriminant_module():
    assert resultant_poly(x**2 - 2, x - 3, x) == 7
    assert discriminant_poly(x**2 + 2 * x + 1, x) == 0
    assert discriminant_poly(x**2 + 1, x) == -4


def test_factor_module():
    constant, factors = factor_poly(x**2 - 2, [x])
    assert constant == 1 and len(factors) == 1  # irreducible over QQ
    _, factors_ext = factor_poly(x**2 - 2, [x], extension="sqrt(2)")
    assert len(factors_ext) == 2


def test_size_guards():
    import pytest
    with pytest.raises(ValueError, match="variable"):
        groebner_basis([x], [sp.Symbol(f"v{i}") for i in range(17)])


# --- kernel facade ------------------------------------------------------------------

def _kernel():
    return MathKernel()


def test_kernel_groebner_exact_trust():
    k = _kernel()
    a = k.parse("x^2 - y^2").data["expr_id"]
    b = k.parse("x - y").data["expr_id"]
    r = k.poly_groebner([a, b], ["x", "y"])
    assert r.ok and r.trust == TrustLevel.EXACT
    assert len(r.data["basis"]) == 1
    assert all(eid for eid in r.data["basis_expr_ids"])


def test_kernel_divide_and_membership():
    k = _kernel()
    a = k.parse("x^2 - y^2").data["expr_id"]
    b = k.parse("x - y").data["expr_id"]
    c = k.parse("x + y").data["expr_id"]
    d = k.poly_divide(a, [b], ["x", "y"])
    assert d.ok and d.data["remainder"] == "0"
    m = k.ideal_membership(a, [b, c], ["x", "y"])
    assert m.status == "verified" and m.data["member"] is True
    m2 = k.ideal_membership(k.parse("x^2 + y^2").data["expr_id"], [b], ["x", "y"])
    assert m2.status == "refuted" and m2.data["member"] is False


def test_kernel_resultant_discriminant_factor():
    k = _kernel()
    r = k.poly_resultant(k.parse("x^2 - 2").data["expr_id"],
                         k.parse("x - 3").data["expr_id"], "x")
    assert r.data["result"] == "7" and r.trust == TrustLevel.EXACT
    disc = k.poly_discriminant(k.parse("x^2 + 1").data["expr_id"], "x")
    assert disc.data["result"] == "-4"
    f = k.poly_factor(k.parse("x^3 - x").data["expr_id"])
    assert {f_["factor"] for f_ in f.data["factors"]} == {"x", "x - 1", "x + 1"}


def test_kernel_unknown_expr_ids():
    k = _kernel()
    assert not k.poly_groebner(["expr_nope"], ["x"]).ok
    assert not k.poly_divide("expr_nope", [], ["x"]).ok
    assert not k.ideal_membership("expr_nope", [], ["x"]).ok


def test_kernel_bad_order_rejected():
    k = _kernel()
    a = k.parse("x").data["expr_id"]
    r = k.poly_groebner([a], ["x"], order="graduated-cylinder")
    assert not r.ok and "monomial order" in r.errors[0]


def test_groebner_batch_parallel():
    k = _kernel()
    jobs = [{"polys": ["x**2 - y**2", "x - y"], "variables": ["x", "y"]},
            {"polys": ["x**2 + 2*x + 1"], "variables": ["x"]},
            {"polys": ["a**2 - b**2", "a - b"], "variables": ["a", "b"], "order": "grevlex"},
            {"polys": ["x +"], "variables": ["x"]}]  # deliberate failure
    r = k.poly_groebner_batch(jobs, workers=2)
    assert r.data["job_count"] == 4 and r.data["failed"] == 1
    assert r.data["results"][0]["ok"] and r.data["results"][0]["basis"] == ["x - y"]
    assert not r.data["results"][3]["ok"]
    assert r.trust == TrustLevel.EXACT
