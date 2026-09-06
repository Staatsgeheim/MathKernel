# =============================================================================
# MathKernel - Tests for the general theorem-proving interface (v0.20)
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""General theorem proving: fragment classifier, SMT portfolio, Lean tactics,
certificate store/replay, batch proving."""

import pytest

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.parser import parse_math
from mathkernel.prove import classify_fragment, portfolio_encodings


# --- fragment classifier -----------------------------------------------------

def test_classify_qf_linear_rational():
    f = classify_fragment(parse_math("2*x + 3 <= 7"))
    assert f["logic"] == "qf" and f["arithmetic"] == "linear"
    assert f["sort"] == "rational" and not f["has_functions"]


def test_classify_quantified_nonlinear():
    f = classify_fragment(parse_math("forall(x, Reals, x^2 >= 0)"))
    assert f["logic"] == "quantified" and f["arithmetic"] == "nonlinear"


def test_classify_int_sort_and_functions():
    f = classify_fragment(parse_math("x + y > 0"), {"x": "int", "y": "int"})
    assert f["sort"] == "int"
    g = classify_fragment(parse_math("sin(x) > 0"))
    assert g["has_functions"] and g["arithmetic"] == "outside"


def test_portfolio_selection():
    assert "lia" in portfolio_encodings(
        {"sort": "int", "arithmetic": "linear"}, 3)
    assert portfolio_encodings({"sort": "real", "arithmetic": "nonlinear"}, 3) == \
        ["direct", "qe-light"]
    assert len(portfolio_encodings({"sort": "int", "arithmetic": "linear"}, 1)) == 1


# --- SMT portfolio via kernel.prove --------------------------------------------

def test_prove_valid_universal():
    k = MathKernel()
    eid = k.parse("forall(x, Reals, x^2 >= 0)").data["expr_id"]
    r = k.prove(eid, formal=False)
    assert r.status == "verified" and r.trust == TrustLevel.EXACT
    assert r.data["fragment"]["logic"] == "quantified"


def test_prove_not_valid_with_countermodel():
    k = MathKernel()
    eid = k.parse("forall(x, Reals, x^2 > 0)").data["expr_id"]
    r = k.prove(eid, formal=False)
    assert r.status == "refuted" and r.trust == TrustLevel.EXACT
    assert "x" in r.data["smt"]["countermodel"]


def test_prove_relation_with_context():
    k = MathKernel()
    ctx = k.create_context(domains={"x": "real", "y": "real"},
                           assumptions=["x > 0", "y > 0"])
    eid = k.parse("x + y > 0").data["expr_id"]
    r = k.prove(eid, ctx.context_id)
    assert r.status == "verified" and r.trust in (TrustLevel.EXACT, TrustLevel.FORMAL)


def test_prove_unknown_expr():
    k = MathKernel()
    assert not k.prove("expr_nope").ok


# --- Lean tier ------------------------------------------------------------------

def test_lean_omega_selection():
    k = MathKernel()
    rel = parse_math("x + 1 > x")
    script, tactic = k.lean.build_relation_certificate(rel, integer=True)
    assert tactic == "omega" and "(x : ℤ)" in script


def test_lean_decimal_and_division_fragment():
    k = MathKernel()
    # finite decimals become exact rationals; division by nonzero literals ok
    assert k.lean.to_lean(parse_math("0.5")) == "(1 / 2)"
    assert k.lean.to_lean(parse_math("x / 2")) == "(x / 2)"
    with pytest.raises(ValueError, match="nonzero literals"):
        k.lean.to_lean(parse_math("x / y"))


def test_lean_relation_certificate_linarith():
    k = MathKernel()
    rel = parse_math("x + y > 0")
    script, tactic = k.lean.build_relation_certificate(
        rel, assumptions=[parse_math("x > 0"), parse_math("y > 0")])
    assert tactic == "linarith" and "h0" in script


def test_prove_formal_upgrade_when_lean_available():
    k = MathKernel()
    eid = k.parse("x + x = 2*x").data["expr_id"]
    r = k.prove(eid)
    assert r.status == "verified"
    if k.lean.available:
        assert r.trust == TrustLevel.FORMAL
        assert r.data["lean"]["tactic"] == "ring"
    else:
        assert r.trust == TrustLevel.EXACT
        assert r.data["lean"]["status"] in ("unavailable", "error", "unsupported")


# --- certificate store + replay ---------------------------------------------------

def test_certificate_store_and_replay(tmp_path):
    k = MathKernel(Settings(store_path=str(tmp_path / "mk.db")))
    eid = k.parse("x + x = 2*x").data["expr_id"]
    r = k.prove(eid)
    if not k.lean.available:
        pytest.skip("Lean not installed")
    cert_id = r.data.get("certificate_id")
    assert cert_id is not None
    rep = k.prove_replay(cert_id)
    assert rep.status == "verified" and rep.trust == TrustLevel.FORMAL
    k._store.close()


def test_prove_replay_requires_store():
    k = MathKernel()
    assert not k.prove_replay("cert_nope").ok


# --- batch proving -----------------------------------------------------------------

def test_prove_batch_parallel():
    k = MathKernel()
    ids = [k.parse(s).data["expr_id"] for s in
           ["forall(x, Reals, x^2 >= 0)", "1 + 1 = 2",
            "forall(x, Reals, x > 1)", "2 * 3 = 6"]]
    out = k.prove_batch(ids, workers=2)
    assert out.ok and out.data["total"] == 4
    by_id = {r["expr_id"]: r for r in out.data["results"]}
    assert by_id[ids[0]]["status"] == "valid"
    assert by_id[ids[2]]["status"] == "not_valid"
    assert out.data["valid"] == 3
    assert out.trust == TrustLevel.EXACT  # all children decisive


def test_prove_batch_trust_is_weakest_evidence(monkeypatch):
    # regression: an unavailable/unknown child must cap the batch trust at
    # UNKNOWN — the outer result may never be stronger than its evidence
    import mathkernel.parallel as par
    import mathkernel.prove as prove_mod

    def fake_job(job):
        return {"expr_id": job["expr_id"], "status": "unavailable",
                "reason": "z3-solver is not installed"}

    monkeypatch.setattr(prove_mod, "_prove_job", fake_job)
    monkeypatch.setattr(par, "process_map",
                        lambda fn, jobs, workers=None: [fn(j) for j in jobs])
    k = MathKernel()
    ids = [k.parse("1 + 1 = 2").data["expr_id"]]
    out = k.prove_batch(ids, workers=1)
    assert out.ok and out.trust == TrustLevel.UNKNOWN
    assert out.data["decisive"] == 0
