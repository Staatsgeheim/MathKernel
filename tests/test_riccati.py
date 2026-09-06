import asyncio
import copy
import numpy as np
import pytest
from mathkernel import MathKernel,Settings,TrustLevel,ResultStatus
from test_engineering_signal import create


def system(k,discrete=False,A=0,D=0):
    data=dict(A=[[A]],B=[[1]],C=[[1]],D=[[D]])
    if discrete:data.update(time_domain='discrete',sample_time='1/10')
    return create(k,'StateSpaceSystem',**data)


def weights(op,q=1):return {'Q':[[q]],'R':[[1]]} if op=='lqr' else {'W':[[q]],'V':[[1]]}


@pytest.mark.parametrize('op',['lqr','kalman'])
@pytest.mark.parametrize('discrete',[False,True])
def test_exact_search_and_solver_free_replay(op,discrete,monkeypatch,isolated_patch):
    from scipy import linalg
    k=MathKernel();src=system(k,discrete,A=int(discrete),D=2)
    before=k.object_get(src).model_dump(mode='json');args=weights(op,'1/2' if discrete else 1)
    result=k.apply(src,op,{**args,'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.EXACT and result.data['details']['accepted']
    assert result.data['value']==[['1/2' if discrete else '1']]
    assert k.object_get(src).model_dump(mode='json')==before
    def forbidden(*args,**kwargs):raise AssertionError('replay called a solver')
    monkeypatch.setattr(linalg,'solve_continuous_are',forbidden);monkeypatch.setattr(linalg,'solve_discrete_are',forbidden)
    isolated_patch('scipy.linalg', 'solve_continuous_are', forbidden)
    isolated_patch('scipy.linalg', 'solve_discrete_are', forbidden)
    replay=k.apply(src,op,{**args,'certificate_id':result.data['object_ids']['certificate']})
    assert replay.ok and replay.data['details']['accepted']
    child=k.math_objects[replay.data['object_ids']['output']]['value']
    if op=='lqr':assert str(child.C[0][0])==('0' if discrete else '-1') # C-DK
    else:
        assert str(child.B[0][0])==('0' if discrete else '-1') # B-LD
        assert replay.data['details']['model_validated'] is False
        assert k.apply(replay.data['object_ids']['error'],'stability').data['value'] is True


@pytest.mark.parametrize('op',['lqr','kalman'])
@pytest.mark.parametrize('P',[0,-1,2])
def test_bad_riccati_witness_does_not_create_a_controller(op,P):
    k=MathKernel();src=system(k)
    result=k.apply(src,op,{**weights(op),'certificate':{'P':[[P]]}})
    assert result.ok,result.errors
    assert not result.data['details']['accepted']
    assert result.semantic_status==ResultStatus.CANDIDATE
    assert 'output' not in result.data['object_ids']


def test_zero_cost_unstable_root_rejected_and_stabilizing_scope_explicit():
    k=MathKernel();src=system(k,A=1)
    bad=k.apply(src,'lqr',{'Q':[[0]],'R':[[1]],'certificate':{'P':[[0]]}})
    assert bad.data['details']['certificate_checks']['riccati_identity']
    assert not bad.data['details']['accepted']
    good=k.apply(src,'lqr',{'Q':[[0]],'R':[[1]],'certificate':{'P':[[2]]}})
    assert good.data['details']['accepted']
    assert good.data['details']['unrestricted_finite_cost_optimality_claimed'] is False
    # u=0 costs zero but grows forever; it is excluded by the stated terminal condition.
    assert 'terminal' in good.data['details']['claim_scope']


@pytest.mark.parametrize('op',['lqr','kalman'])
@pytest.mark.parametrize('stored',[False,True])
def test_decimal_certificate_cancellation_cannot_promote(op,stored):
    k=MathKernel();src=system(k)
    cert={'P':[['1+(1.0-1.0)']]}
    params={'certificate_id':create(k,'RiccatiCertificate',**cert)} if stored else {'certificate':cert}
    result=k.apply(src,op,{**weights(op),**params})
    assert result.ok and result.trust==TrustLevel.NUMERIC and not result.data['details']['accepted']
    result=k.apply(src,op,{**weights(op,'1+(1.0-1.0)'),'mode':'numeric'})
    assert result.ok and result.trust==TrustLevel.NUMERIC and not result.data['details']['accepted']


@pytest.mark.parametrize('op',['lqr','kalman'])
@pytest.mark.parametrize('discrete',[False,True])
def test_irrational_solution_numerical_and_lyapunov_covariance_identity(op,discrete):
    k=MathKernel();src=system(k,discrete,A=1)
    result=k.apply(src,op,{**weights(op),'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.NUMERIC and not result.data['details']['accepted']
    assert all(result.data['details']['numeric_checks'].values())
    P=float(result.data['details']['P'][0][0]);gain=float(result.data['details']['gain'][0][0]);closed=1-gain
    # Independent completion-of-squares / noise covariance equation.
    lhs=(closed**2*P-P+1+gain**2) if discrete else (2*closed*P+1+gain**2)
    assert abs(lhs)<1e-10


def test_mimo_care_and_dare_against_finite_horizon_iteration():
    from scipy.linalg import expm,solve_continuous_lyapunov
    for discrete in (False,True):
        k=MathKernel();A=np.array([[.3,.1],[0,.2]]) if discrete else np.array([[-1.,.2],[0,-2.]])
        B=np.array([[1.,.3],[0.,1.]])
        params=dict(A=A.tolist(),B=B.tolist(),C=np.eye(2).tolist(),D=np.zeros((2,2)).tolist())
        if discrete:params.update(time_domain='discrete',sample_time=1)
        src=create(k,'StateSpaceSystem',**params)
        r=k.apply(src,'lqr',{'Q':np.eye(2).tolist(),'R':np.eye(2).tolist(),'mode':'numeric'})
        assert r.ok,r.errors
        P=np.array(r.data['details']['P'],float);K=np.array(r.data['details']['gain'],float)
        if discrete:
            ref=np.zeros((2,2))
            for _ in range(100):ref=np.eye(2)+A.T@ref@A-A.T@ref@B@np.linalg.solve(np.eye(2)+B.T@ref@B,B.T@ref@A)
        else:
            closed=A-B@K;ref=solve_continuous_lyapunov(closed.T,-(np.eye(2)+K.T@K))
        np.testing.assert_allclose(P,ref,atol=1e-10)


def test_resource_and_invalid_dimensions_and_noise():
    k=MathKernel();src=system(k)
    for params in [weights('lqr'),{'Q':[[1]],'R':[[0]],'mode':'numeric'},
                   {'Q':[[1,0],[0,1]],'R':[[1]],'mode':'numeric'},
                   {'Q':[[1]],'R':[[1]],'mode':'numeric','tolerance':True}]:
        assert not k.apply(src,'lqr',params).ok
    assert not k.object_create('RiccatiCertificate',{'P':[[1,2],[0,1]]}).ok
    k=MathKernel(Settings(max_exact_control_order=1));src=create(k,'StateSpaceSystem',A=[[-1,0],[0,-1]],B=[[1],[0]],C=[[1,0]],D=[[0]])
    r=k.apply(src,'lqr',{'Q':[[1,0],[0,1]],'R':[[1]],'certificate':{'P':[[1,0],[0,1]]}})
    assert not r.ok


@pytest.mark.parametrize('op',['lqr','kalman'])
def test_synthesis_persistence_sources_and_live_mcp(tmp_path,op):
    settings=Settings(store_path=str(tmp_path/'riccati.sqlite'));k=MathKernel(settings);src=system(k)
    r=k.apply(src,op,{**weights(op),'mode':'numeric'});restart=MathKernel(settings)
    cert=r.data['object_ids']['certificate']
    replay=restart.apply(src,op,{**weights(op),'certificate_id':cert})
    assert replay.ok and replay.data['details']['accepted']
    for child in replay.data['object_ids'].values():
        obj=restart.object_get(child);assert set(obj.data['sources'])=={src,cert}
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def go():
        async with Client(mcp) as client:
            obj=(await client.call_tool('math_object_create',{'object_type':'StateSpaceSystem','definition':{'A':[[0]],'B':[[1]],'C':[[1]],'D':[[0]]}})).data
            result=(await client.call_tool('math_apply',{'object_id':obj['data']['object_id'],'operation':op,'parameters':{**weights(op),'mode':'numeric'}})).data
            assert result['ok'] and result['data']['details']['accepted']
    asyncio.run(go())
