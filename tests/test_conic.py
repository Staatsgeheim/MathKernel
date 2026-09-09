import copy
import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, Settings, TrustLevel, ResultStatus
from mathkernel.conic import ConeBlock, cone_membership
from test_engineering_signal import create


def soc_definition(a=3, b=4):
    return dict(variables=['t'], c=[1], A=[[-1], [0], [0]], b=[0, a, b], cones=[dict(kind='second_order', dimension=3)])


def psd_definition(offdiagonal=False):
    return dict(variables=['x'], c=[1], sense='max' if offdiagonal else 'min',
        A=[[0], [-1], [-1], [0]] if offdiagonal else [[-1], [0], [0], [-1]],
        b=[1, 0, 0, 1] if offdiagonal else [0, 1, 1, 0], cones=[dict(kind='psd', dimension=2)])


@pytest.mark.parametrize('definition,expected', [(soc_definition(), '5'), (psd_definition(), '1'), (psd_definition(True), '1')])
def test_cone_search_checked_exact_and_solver_free_replay(definition, expected, monkeypatch, isolated_patch):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    import clarabel
    k=MathKernel();p=create(k, 'ConicProblem', **definition)
    before=k.object_get(p).model_dump(mode='json')
    result=k.apply(p, 'solve', {'mode':'numeric'})
    assert result.ok, result.errors
    assert result.trust==TrustLevel.EXACT and result.data['value']==expected
    assert result.data['details']['accepted']
    assert k.object_get(p).model_dump(mode='json')==before
    def forbidden(*args, **kwargs): raise AssertionError('solver invoked during replay')
    monkeypatch.setattr(clarabel, 'DefaultSolver', forbidden)
    isolated_patch('clarabel', 'DefaultSolver', forbidden)
    replay=k.apply(p, 'verify_certificate', {'certificate_id':result.data['object_id']})
    assert replay.ok and replay.data['details']['accepted']
    assert replay.data['value']==expected


def test_product_cones_and_equality_multipliers():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k, 'ConicProblem', variables=['t','x'], c=[1,0],
        A=[[-1,0],[0,-1],[0,0],[1,0]],b=[0,0,4,10],A_eq=[[0,1]],b_eq=[3],
        cones=[{'kind':'second_order','dimension':3},{'kind':'nonnegative','dimension':1}])
    cert={'primal':[5,3],'cone_dual':[1,'-3/5','-4/5',0],'equality_dual':['-3/5']}
    r=k.apply(p,'verify_certificate',{'certificate':cert})
    assert r.ok and r.data['details']['accepted']
    solved=k.apply(p,'solve',{'mode':'numeric'})
    assert solved.ok and solved.data['details']['accepted']
    assert solved.data['value']=='5'


@pytest.mark.parametrize('cone,values,expected',[
    (('second_order',3),[-5,3,4],False), (('second_order',3),[5,3,4],True),
    (('second_order',3),[0,0,0],True), (('second_order',2),[1,'1000000000001/1000000000000'],False),
    (('psd',2),[0,1,1,0],False), (('psd',2),[1,1,1,1],True),
    (('psd',2),[0,0,0,0],True), (('psd',2),[1,0,1,1],False),
    (('psd',2),[1,1,1,'999999999999/1000000000000'],False),
])
def test_exact_cone_membership_edges(cone,values,expected):
    block=ConeBlock(kind=cone[0],dimension=cone[1])
    result,_=cone_membership(tuple(sp.Rational(v) for v in values),(block,))
    assert result is expected


@pytest.mark.parametrize('kind',['optimal','bound','infeasible','unbounded'])
def test_bad_witnesses_never_establish_outcomes(kind):
    k=MathKernel();p=create(k,'ConicProblem',**soc_definition())
    cert={'kind':kind}
    if kind in {'optimal','unbounded'}:cert['primal']=[5]
    if kind=='unbounded':cert['ray']=[-1]
    else:cert['cone_dual']=[-1,0,0]
    result=k.apply(p,'verify_certificate',{'certificate':cert})
    assert result.ok and not result.data['details']['accepted']
    assert result.semantic_status==ResultStatus.CANDIDATE


def test_rational_dual_bound_for_irrational_optimum():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'ConicProblem',**soc_definition(1,1))
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'bound','cone_dual':[1,'-7/10','-7/10']}})
    assert r.ok and r.trust==TrustLevel.EXACT
    assert r.data['value']=='7/5' and r.data['details']['conclusion']=='certified_lower_bound'
    numeric=k.apply(p,'solve',{'mode':'numeric'})
    assert numeric.ok and not numeric.data['details']['accepted']
    assert numeric.trust==TrustLevel.NUMERIC
    assert float(numeric.data['value'])==pytest.approx(2**.5,abs=1e-7)


@pytest.mark.parametrize('definition,certificate,outcome',[
    (dict(variables=['x'],c=[0],A=[[0],[0]],b=[-1,0],cones=[dict(kind='second_order',dimension=2)]),
     dict(kind='infeasible',cone_dual=[1,0]),'infeasible'),
    (dict(variables=['x'],c=[0],A=[[0],[0],[0],[0]],b=[-1,0,0,1],cones=[dict(kind='psd',dimension=2)]),
     dict(kind='infeasible',cone_dual=[1,0,0,0]),'infeasible'),
    (dict(variables=['x'],c=[-1],A=[[-1],[0]],b=[0,0],cones=[dict(kind='second_order',dimension=2)]),
     dict(kind='unbounded',primal=[0],ray=[1]),'unbounded'),
    (dict(variables=['x'],c=[1]),dict(kind='unbounded',primal=[0],ray=[-1]),'unbounded'),
])
def test_outcome_witnesses_and_search(definition,certificate,outcome):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'ConicProblem',**definition)
    r=k.apply(p,'verify_certificate',{'certificate':certificate})
    assert r.ok and r.data['details']['accepted']
    assert r.semantic_status.value==outcome
    numeric=k.apply(p,'solve',{'mode':'numeric'})
    assert numeric.ok,numeric.errors
    assert numeric.semantic_status.value==outcome,numeric.data['details']
    assert numeric.trust==TrustLevel.EXACT


def test_decimal_inputs_and_witnesses_never_become_exact():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();definition=soc_definition();definition['b'][1]='3.0'
    p=create(k,'ConicProblem',**definition)
    r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.ok and r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']
    p=create(k,'ConicProblem',**soc_definition())
    r=k.apply(p,'verify_certificate',{'certificate':{'primal':['5.0'],'cone_dual':[1,'-3/5','-4/5']}})
    assert r.ok and r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']


def test_lp_conversion_preserves_free_variables_bounds_and_maximization():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x','y'],c=[2,1],A_ub=[[1,1]],b_ub=[3],
                          lower=[None,-1],upper=[2,None],sense='max')
    converted=k.apply(p,'to_conic')
    assert converted.ok,converted.errors
    before=k.apply(p,'solve')
    after=k.apply(converted.data['object_id'],'solve',{'mode':'numeric'})
    assert before.data['value']==after.data['value']=='5'
    assert after.data['details']['accepted']
    integer=create(k,'OptimizationProblem',variables=['x'],c=[1],integrality=[1])
    assert not k.apply(integer,'to_conic').ok


def test_psd_packing_preserves_trace_pairing_and_stationarity():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    pytest.importorskip("clarabel", reason="Install mathkernel[test] for optional backend coverage")
    from mathkernel.conic_search import pack_problem,unpack_dual
    from mathkernel.conic import ConicProblem
    rng=np.random.default_rng(92);d=4;n=3
    coeff=rng.integers(-3,4,(n+1,d,d));coeff=coeff+coeff.transpose(0,2,1)
    problem=ConicProblem(variables=('x','y','z'),c=tuple(map(sp.Integer,[1,2,3])),
        A=tuple(tuple(sp.Integer(coeff[k+1,i,j]) for k in range(n)) for i in range(d) for j in range(d)),
        b=tuple(map(sp.Integer,coeff[0].ravel())),cones=(ConeBlock(kind='psd',dimension=d),))
    packed,rhs,_,mapping,original,_,_=pack_problem(problem)
    dual=rng.normal(size=len(mapping));unpacked=unpack_dual(dual,mapping,d*d)
    np.testing.assert_allclose(packed.T@dual,original.T@unpacked,atol=1e-12)
    np.testing.assert_allclose(rhs@dual,np.asarray(problem.b,dtype=float)@unpacked,atol=1e-12)
    assert packed.shape[0]==d*(d+1)//2


def test_cone_validation_and_resource_limits():
    k=MathKernel(Settings(max_psd_cone_order=1))
    assert not k.object_create('ConicProblem',psd_definition()).ok
    for change in [{'cones':[{'kind':'second_order','dimension':True}]},{'b':[0,3]},
                   {'A':[[0],[1],[0]],'b':[0,3,4],'cones':[{'kind':'psd','dimension':2}]}]:
        definition={**soc_definition(),**change}
        assert not MathKernel().object_create('ConicProblem',definition).ok
    p=create(k,'ConicProblem',**soc_definition())
    assert not k.apply(p,'solve').ok
    for params in [{'mode':'numeric','tolerance':float('inf')},{'mode':'numeric','max_iterations':True}]:
        assert not k.apply(p,'solve',params).ok


def test_weakly_infeasible_sdp_has_no_strict_farkas_claim_from_zero_pairing():
    # [[x,1],[1,0]] cannot be PSD, but its affine slice approaches the cone.
    # The PSD dual [[0,0],[0,1]] has zero pairing, not a strict contradiction.
    k=MathKernel();p=create(k,'ConicProblem',variables=['x'],c=[0],
        A=[[-1],[0],[0],[0]],b=[0,1,1,0],cones=[dict(kind='psd',dimension=2)])
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'infeasible','cone_dual':[0,0,0,1]}})
    assert r.ok and not r.data['details']['accepted']
    assert r.data['details']['certificate_checks']['dual_cone']
    assert not r.data['details']['certificate_checks']['strict_contradiction']


def test_equality_only_infeasibility_and_maximization_bound_sign():
    k=MathKernel();p=create(k,'ConicProblem',variables=['x'],c=[0],A_eq=[[1],[1]],b_eq=[0,1])
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'infeasible','equality_dual':[1,-1]}})
    assert r.ok and r.trust==TrustLevel.EXACT and r.semantic_status==ResultStatus.INFEASIBLE
    p=create(k,'ConicProblem',variables=['x'],c=[1],sense='max',A=[[1]],b=[2],cones=[dict(kind='nonnegative',dimension=1)])
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'bound','cone_dual':[1]}})
    assert r.ok and r.data['details']['conclusion']=='certified_upper_bound' and r.data['value']=='2'
