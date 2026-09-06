# =============================================================================
# MathKernel - decimal trust adversarial tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from mathkernel.kernel import MathKernel
from mathkernel.models import TrustLevel

def test_cancellation_keeps_numeric_provenance():
    k=MathKernel(); p=k.parse('0.7-0.7'); r=k.simplify(p.data['expr_id'])
    assert r.data['result']=='0' and r.trust==TrustLevel.NUMERIC

def test_deep_real_keeps_numeric_provenance():
    k=MathKernel(); p=k.parse('(x+0.001)/(1+x*x)'); r=k.simplify(p.data['expr_id'])
    assert p.trust==r.trust==TrustLevel.NUMERIC

def test_get_expression_does_not_upgrade_decimal():
    k=MathKernel(); p=k.parse('0.7*x'); assert k.get_expression(p.data['expr_id']).trust==TrustLevel.NUMERIC

def test_equivalence_decimal_never_formal_or_exact():
    k=MathKernel(); r=k.prove_equivalence('0.7+0.3','1',formal=True)
    assert r.trust==TrustLevel.NUMERIC
    assert 'lean_certificate' not in r.data
    assert any('Approximate decimal input' in w for w in r.warnings)

def test_counterexample_decimal_refuses_exact_smt_semantics():
    k=MathKernel(); r=k.counterexample('0.7*x','x')
    assert r.status=='unknown' and r.trust==TrustLevel.UNKNOWN
