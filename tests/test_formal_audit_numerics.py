"""Regressions uncovered by exercising the real Navier-Stokes audit formulas."""
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import importlib.util
from pathlib import Path
import mpmath as mp
import pytest
from mathkernel import MathKernel, TrustLevel
from mathkernel.certified import mathir_interval_enclosure
from mathkernel.parser import parse_math


def test_affine_rational_interval_no_point_float_mixing():
    kernel=MathKernel()
    eid=kernel.parse('1/2-3*h').data['expr_id']
    result=kernel.certified_enclose(eid,'h','0','1/100')
    assert result.ok and result.trust==TrustLevel.INTERVAL_CERTIFIED
    assert result.data['strictly_positive'] and not result.data['contains_zero']


def test_rational_endpoint_is_enclosed_not_point_rounded():
    r=mathir_interval_enclosure(parse_math('x'),'x',parse_math('1/10'),parse_math('1/10'))
    lo, hi = [Fraction(x.strip()) for x in r['enclosure'].strip('[]').split(',')]
    assert lo < Fraction(1,10) < hi


def test_decimal_endpoint_ancestry_is_not_certified_exact():
    k=MathKernel();e=k.parse('x').data['expr_id']
    r=k.certified_enclose(e,'x','0.1','0.2')
    assert r.ok and r.trust==TrustLevel.NUMERIC
    assert r.reconciled_trust()==TrustLevel.NUMERIC


@pytest.mark.parametrize('expr,lo,hi', [('1/x','-1','1'),('sqrt(x)','-1','1'),
    ('log(x)','-1','1'),('x','2','1'),('x+y','0','1'),('x','a','1'),
    ('x','1/0','1'),('gamma(x)','1','2')])
def test_invalid_interval_queries_return_structured_errors(expr,lo,hi):
    k=MathKernel();e=k.parse(expr).data['expr_id']
    r=k.certified_enclose(e,'x',lo,hi)
    assert not r.ok and r.trust==TrustLevel.UNKNOWN and r.errors


@pytest.mark.parametrize('expr,lo,hi', [('sin(x)','0','pi'),('exp(x)','0','1'),
    ('sqrt(x)','1','2'),('log(x)','1','2'),('x','pi','pi'),('x','e','e'),('x^2','-1','1')])
def test_supported_interval_functions(expr,lo,hi):
    k=MathKernel();e=k.parse(expr).data['expr_id']
    assert k.certified_enclose(e,'x',lo,hi).ok


def test_interval_contexts_do_not_change_global_precision():
    old=mp.iv.dps
    def run(dps):
        return mathir_interval_enclosure(parse_math('x/3'),'x',parse_math('1'),parse_math('2'),dps=dps)['dps']
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run,[25,50,75,100]*3)) == [25,50,75,100]*3
    assert mp.iv.dps==old


def test_closed_rational_false_identity_needs_no_z3_or_lean(monkeypatch):
    k=MathKernel()
    monkeypatch.setattr(type(k.z3),'available',property(lambda self:False))
    monkeypatch.setattr(k.lean,'prove_equivalence',lambda *args:pytest.fail('Lean ran despite exact refutation'))
    r=k.prove_equivalence('1/200 + 1/2','2/200 + 1/2',formal=True)
    assert r.status=='refuted' and r.trust==TrustLevel.EXACT
    assert r.data['counterexample']=={'left_value':'101/200','right_value':'51/100'}
    assert not r.evidence_bundle.proof


def test_approximate_false_identity_cannot_use_rational_refutation(monkeypatch):
    k=MathKernel()
    monkeypatch.setattr(type(k.z3),'available',property(lambda self:False))
    r=k.prove_equivalence('0.1','0.2',formal=False)
    assert r.trust != TrustLevel.EXACT
    assert not any(e.engine=='rational' for e in r.evidence)


def test_real_local_audit_cases_and_negative_controls():
    path=Path(__file__).parents[1]/'examples/audits/openai_navier_stokes/local_checks.py'
    spec=importlib.util.spec_from_file_location('navier_local_checks',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.run_checks()
    assert report['theorem_verification']=='not_run'
    assert all(c['passed'] for c in report['identities']+report['certified_margins'])
    assert all(c['rejected'] for c in report['negative_controls'])
    assert len(report['identities'])==13
