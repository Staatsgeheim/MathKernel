import copy
import numpy as np
import pytest
from mathkernel import MathKernel, Settings, TrustLevel, ResultStatus
from test_engineering_signal import create


def ball(c=-1, radius_squared=1):
    return dict(variables=['x'],c=[c],lower=[None],quadratics=[dict(Q=[[2]],a=[0],r=-radius_squared)])


def test_numeric_convex_constraint_search_and_replay():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'QCQP',**ball())
    result=k.apply(p,'solve',{'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.EXACT
    assert result.data['value']=='-1' and result.data['details']['accepted']
    replay=k.apply(p,'verify_certificate',{'certificate_id':result.data['object_id']})
    assert replay.ok and replay.data['details']['accepted']
    assert replay.data['details']['lagrangian_hessian']==[['1']]


@pytest.mark.parametrize('definition,certificate,expected',[
    ({**ball(0),'Q':[[-2]]},{'primal':[1],'quadratic_dual':[1]},'-1'),
    ({**ball(0),'Q':[[2]],'quadratics':[{'Q':[[-2]],'a':[0],'r':1}]},{'primal':[1],'quadratic_dual':[1]},'1'),
    (dict(variables=['x','y'],c=[0,0],Q=[[-2,0],[0,2]],lower=[None,None],quadratics=[dict(Q=[[2,0],[0,2]],a=[0,0],r=-1)]),
     {'primal':[1,0],'quadratic_dual':[1]},'-1'),
    (dict(variables=['x','y'],c=[1,1],sense='max',A_eq=[[0,1]],b_eq=[2],quadratics=[dict(Q=[[2,0],[0,0]],a=[0,0],r=-1)]),
     {'primal':[1,2],'inequality_dual':[0,0],'equality_dual':[1],'quadratic_dual':['1/2']},'3'),
    ({**ball(),'upper':['1/2']},{'primal':['1/2'],'inequality_dual':[1],'quadratic_dual':[0]},'-1/2'),
])
def test_global_lagrangian_witnesses_include_nonconvex_and_singular_cases(definition,certificate,expected):
    k=MathKernel();p=create(k,'QCQP',**definition)
    result=k.apply(p,'verify_certificate',{'certificate':certificate})
    assert result.ok,result.errors
    assert result.data['details']['accepted'],result.data
    assert result.data['value']==expected and result.trust==TrustLevel.EXACT


def test_nonconvex_stationary_point_never_certifies_as_minimum():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'QCQP',**{**ball(0),'Q':[[-2]]})
    result=k.apply(p,'verify_certificate',{'certificate':{'primal':[0],'quadratic_dual':[0]}})
    assert result.ok and not result.data['details']['accepted']
    assert result.data['details']['certificate_checks']['stationarity']
    assert not result.data['details']['certificate_checks']['lagrangian_psd']
    numerical=k.apply(p,'solve',{'mode':'numeric'})
    assert numerical.ok and not numerical.data['details']['accepted']
    assert numerical.trust==TrustLevel.NUMERIC
    global_result=k.apply(p,'solve',{'mode':'numeric','initial':[1]})
    assert global_result.ok and global_result.data['details']['accepted']
    assert global_result.data['value']=='-1'


@pytest.mark.parametrize('mutation',['negative_multiplier','bad_primal','bad_stationarity','bad_dimension','decimal'])
def test_tampered_quadratic_witness(mutation):
    k=MathKernel();p=create(k,'QCQP',**ball())
    cert={'primal':[1],'quadratic_dual':['1/2']}
    if mutation=='negative_multiplier':cert['quadratic_dual']=['-1/2']
    elif mutation=='bad_primal':cert['primal']=[2]
    elif mutation=='bad_stationarity':cert['quadratic_dual']=[1]
    elif mutation=='bad_dimension':cert['quadratic_dual']=[]
    else:cert['primal']=['1.0']
    result=k.apply(p,'verify_certificate',{'certificate':cert})
    assert not result.ok or not result.data['details']['accepted']
    if mutation=='decimal':assert result.trust==TrustLevel.NUMERIC


def test_irrational_optimum_is_numerical_and_decimal_problem_cannot_be_promoted():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'QCQP',**ball(radius_squared=2))
    r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.ok and not r.data['details']['accepted']
    assert r.trust==TrustLevel.NUMERIC and float(r.data['value'])==pytest.approx(-2**.5,abs=1e-7)
    definition=ball();definition['quadratics'][0]['r']='-1.0'
    p=create(k,'QCQP',**definition);r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.ok and r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']


def test_random_one_dimensional_balls_match_analytic_endpoint_optima():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    rng=np.random.default_rng(712)
    for _ in range(8):
        coefficient=int(rng.choice([-4,-3,-2,-1,1,2,3,4]));radius=int(rng.integers(1,5))
        k=MathKernel();p=create(k,'QCQP',**ball(coefficient,radius**2))
        r=k.apply(p,'solve',{'mode':'numeric'})
        assert r.ok and r.data['details']['accepted'],r.model_dump()
        assert r.data['value']==str(-abs(coefficient)*radius)


def test_qcqp_input_limits_and_parser():
    k=MathKernel(Settings(max_quadratic_constraints=1,max_engineering_work=2))
    assert not k.object_create('QCQP',ball()).ok
    for update in [{'integrality':[1]},{'quadratics':[]},
                   {'quadratics':[{'Q':[[2,1],[0,2]],'a':[0,0]}]},
                   {'quadratics':[{'Q':[[2]],'a':[0],'r':'__import__("os")'}]}]:
        assert not MathKernel().object_create('QCQP',{**ball(),**update}).ok
    k=MathKernel();p=create(k,'QCQP',**ball())
    assert not k.apply(p,'solve').ok
    assert not k.apply(p,'solve',{'mode':'numeric','initial':[0,1]}).ok
