import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, TrustLevel
from test_engineering_signal import create


@pytest.mark.parametrize('numerator,denominator', [([1],[1,3,2]),([1,2],[2,4,6]),([2,3,4],[1,4,5]),([0],[1,1])])
def test_state_transfer_roundtrip(numerator,denominator):
    k=MathKernel()
    source=create(k,'TransferFunction',numerator=numerator,denominator=denominator)
    result=k.apply(source,'to_state_space')
    assert result.ok,result.errors
    assert all(result.data['verification'].values())
    back=k.apply(result.data['object_id'],'to_transfer_function')
    assert back.ok,back.errors
    a=k.math_objects[source]['value'].expression()
    b=k.math_objects[back.data['object_id']]['value'].expression()
    assert sp.cancel(a-b)==0


@pytest.mark.parametrize('coefficients,stable', [([1,3,2],True),([1,-3,2],False),([1,0,1],False),([1,1,1,1],False),([1,6,11,6],True)])
def test_hurwitz_criteria_match_independent_roots(coefficients,stable):
    k=MathKernel();src=create(k,'TransferFunction',numerator=[1],denominator=coefficients)
    r=k.apply(src,'stability')
    assert r.ok,r.errors
    assert r.data['value'] is stable
    assert r.trust==TrustLevel.EXACT
    if stable:
        assert all(np.real(np.roots(coefficients)) < -1e-10)
    else:
        assert max(np.real(np.roots(coefficients))) >= -1e-10


@pytest.mark.parametrize('coefficients,stable', [([1,'-1/2'],True),([1,0,'1/4'],True),([1,0,1],False),([1,-2],False),([1,'-3/2','1/2'],False)])
def test_schur_criteria_and_unit_circle_boundary(coefficients,stable):
    k=MathKernel();src=create(k,'TransferFunction',numerator=[1],denominator=coefficients,time_domain='discrete',sample_time='1/10')
    r=k.apply(src,'stability')
    assert r.ok,r.errors
    assert r.data['value'] is stable
    assert r.data['details']['criterion']=='schur_cohn_real_recursion'


def test_hidden_unstable_modes_are_not_erased_from_internal_stability():
    k=MathKernel()
    state=create(k,'StateSpaceSystem',A=[[-1,0],[0,2]],B=[[1],[0]],C=[[1,0]],D=[[0]])
    internal=k.apply(state,'stability')
    assert internal.data['value'] is False
    tf=k.apply(state,'to_transfer_function')
    bibo=k.apply(tf.data['object_id'],'stability')
    assert bibo.data['value'] is True
    assert internal.data['details']['scope']=='internal_asymptotic_stability'
    assert bibo.data['details']['scope']=='bibo_stability'
    assert k.apply(state,'controllability').data['value']['rank']==1
    assert k.apply(state,'observability').data['value']['rank']==1


def test_feedback_multisource_trust_and_units():
    k=MathKernel()
    g=create(k,'TransferFunction',numerator=[1],denominator=[1,1],input_unit='V',output_unit='A')
    h=create(k,'TransferFunction',numerator=['0.5'],denominator=[1],input_unit='A',output_unit='V')
    r=k.apply(g,'feedback',{'other_id':h})
    assert r.ok,r.errors
    assert r.trust==TrustLevel.NUMERIC
    assert set(k.object_get(r.data['object_id']).data['sources'])=={g,h}
    incompatible=create(k,'TransferFunction',numerator=[1],denominator=[1],input_unit='m')
    assert not k.apply(g,'series',{'other_id':incompatible}).ok


def test_continuous_responses_and_frequency_data():
    k=MathKernel();src=create(k,'TransferFunction',numerator=[1],denominator=[1,1])
    for operation in ['step_response','impulse_response']:
        r=k.apply(src,operation)
        assert r.ok,r.errors
        assert r.trust==TrustLevel.SYMBOLIC
        assert r.data['verification']['laplace_identity']
    f=k.apply(src,'frequency_response',{'frequencies':[0,1,2],'mode':'numeric'})
    assert f.ok,f.errors
    np.testing.assert_allclose(f.data['details']['magnitude'],[1,1/np.sqrt(2),1/np.sqrt(5)])


def test_decimal_stability_cannot_be_exact_even_with_boolean_conclusion():
    k=MathKernel();src=create(k,'TransferFunction',numerator=[1],denominator=[1,'0.1'])
    result=k.apply(src,'stability')
    assert result.data['value'] is True
    assert result.trust==TrustLevel.NUMERIC


def test_unknown_symbolic_stability_retains_conditions():
    k=MathKernel();ctx=k.create_context(domains={'a':'real'})
    src=create(k,'TransferFunction',numerator=[1],denominator=[1,'a'],context_id=ctx.context_id)
    result=k.apply(src,'stability')
    assert result.ok,result.errors
    assert result.data['value'] is None
    assert result.side_conditions==['a > 0']
    assert result.status=='candidate'


def test_discrete_and_singular_control_rejections():
    k=MathKernel()
    assert not k.object_create('TransferFunction',dict(numerator=[1],denominator=[0,1])).ok
    assert not k.object_create('TransferFunction',dict(numerator=[1],denominator=[1],time_domain='discrete')).ok
    a=create(k,'TransferFunction',numerator=[1],denominator=[1])
    b=create(k,'TransferFunction',numerator=[-1],denominator=[1])
    assert not k.apply(a,'feedback',{'other_id':b}).ok
