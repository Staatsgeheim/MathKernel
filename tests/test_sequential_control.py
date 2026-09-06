import asyncio
import copy
import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, Settings, TrustLevel, FiniteHorizonLQR, KalmanState
from test_engineering_signal import create
from test_riccati import system


def run(k, src, op, **params):
    r = k.apply(src, op, params)
    assert r.ok, r.errors
    return r


def child(r):
    return r.data['object_ids']['output']


def finite(k, src, **overrides):
    return run(k, src, 'finite_lqr', **{**dict(Q=[[1]], R=[[1]], terminal=[[1]], horizon=2), **overrides})


def prior(k, src, **overrides):
    return run(k, src, 'kalman_state', **{**dict(mean=[0], covariance=[[1]], W=[[1]], V=[[1]]), **overrides})


def test_exact_finite_horizon_known_solution_and_solver_free_check(monkeypatch, isolated_patch):
    k = MathKernel(); src = system(k, True, A=1)
    snapshot = k.object_get(src).model_dump(mode='json')
    r = finite(k, src); policy = k.math_objects[child(r)]['value']
    assert isinstance(policy, FiniteHorizonLQR) and r.data['details']['accepted']
    assert policy.gains == (((sp.Rational(3,5),),), ((sp.Rational(1,2),),))
    assert policy.values[0] == ((sp.Rational(8,5),),)
    def forbidden(*a, **kw): raise AssertionError('verification used a solve')
    monkeypatch.setattr(sp.MatrixBase, 'solve', forbidden)
    isolated_patch('sympy', 'MatrixBase.solve', forbidden)
    assert run(k, child(r), 'verify').data['details']['accepted']
    assert run(k, child(r), 'control', state=[2], step=0).data['value'] == ['-6/5']
    rollout = run(k, child(r), 'rollout', initial=[2])
    assert rollout.data['details']['accepted']
    assert rollout.data['details']['total_cost'] == '32/5'
    assert k.object_get(src).model_dump(mode='json') == snapshot
    with pytest.raises(Exception): policy.backend = 'numeric'


def test_policy_tampering_rejected_by_independent_bellman_checker():
    from mathkernel.control_sequential import verify_policy
    k=MathKernel(); src=system(k,True,A=1); result=finite(k,src)
    policy=k.math_objects[child(result)]['value']; plant=k.math_objects[src]['value']
    for field, replacement in [('gains', (((sp.S.Zero,),),)*2),
                                ('values', (((sp.S.One,),),)*3)]:
        bad=FiniteHorizonLQR(**{**policy.model_dump(mode='python'),field:replacement})
        check=verify_policy(bad,plant)
        assert not check.details['accepted']
        assert not all(check.details['checks'].values())


@pytest.mark.parametrize('horizon',[0,1,5])
def test_mimo_finite_policy_matches_independently_condensed_qp(horizon):
    # Stack x[0]..x[N], eliminate dynamics once, then minimize the dense quadratic.
    k=MathKernel(); A=np.array([[1.,.1],[0.,1.]])
    B=np.array([[.1],[1.]])
    src=create(k,'StateSpaceSystem',A=A.tolist(),B=B.tolist(),C=[[1,0]],D=[[0]],time_domain='discrete',sample_time='1/10')
    Q=np.diag([2.,1.]); R=np.array([[3.]]); F=np.diag([4.,2.]); x0=np.array([2.,-1.])
    r=run(k,src,'finite_lqr',Q=Q.tolist(),R=R.tolist(),terminal=F.tolist(),horizon=horizon,mode='numeric')
    assert r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']
    T=np.vstack([np.linalg.matrix_power(A,t) for t in range(horizon+1)])
    U=np.zeros((2*(horizon+1),horizon))
    for t in range(1,horizon+1):
        for j in range(t): U[2*t:2*t+2,j:j+1]=np.linalg.matrix_power(A,t-j-1)@B
    from scipy.linalg import block_diag
    weights=block_diag(*([Q]*horizon+[F])); base=T@x0
    inputs=np.linalg.solve(U.T@weights@U+3*np.eye(horizon),-U.T@weights@base)
    states=base+U@inputs
    roll=run(k,child(r),'rollout',initial=x0.tolist())
    np.testing.assert_allclose(np.array(roll.data['value'],float).reshape(-1),states,atol=1e-12)
    np.testing.assert_allclose(np.array(roll.data['details']['controls'],float).reshape(-1),inputs,atol=1e-12)
    np.testing.assert_allclose(float(roll.data['details']['total_cost']),states@weights@states+3*inputs@inputs,rtol=1e-12)


def test_kalman_feedthrough_timing_exact_and_immutable_branches():
    k=MathKernel(); src=system(k,True,A=1,D=2); r=prior(k,src)
    before=k.object_get(child(r)).model_dump(mode='json')
    post=run(k,child(r),'update',measurement=[3],control=[1])
    s=k.math_objects[child(post)]['value']
    assert isinstance(s,KalmanState) and s.mean==(sp.Rational(1,2),) and s.covariance==((sp.Rational(1,2),),)
    assert post.data['details']['innovation']==[['1']]
    predicted=run(k,child(post),'predict'); s=k.math_objects[child(predicted)]['value']
    assert s.mean==(sp.Rational(3,2),) and s.covariance==((sp.Rational(3,2),),)
    assert s.phase=='prior' and s.index==1 and s.pending_control==()
    other=run(k,child(r),'update',measurement=[2],control=[1])
    assert k.math_objects[child(other)]['value'].mean==(sp.S.Zero,)
    assert k.object_get(child(r)).model_dump(mode='json')==before
    assert not k.apply(child(r),'predict').ok
    assert not k.apply(child(post),'update',{'measurement':[3],'control':[1]}).ok
    assert not k.apply(child(post),'predict',{'control':[2]}).ok


def test_kalman_matches_joint_gaussian_conditioning_and_joseph_psd():
    k=MathKernel(); A=np.array([[1.,.1],[0.,.8]]); B=np.array([[0.],[1.]])
    C=np.array([[1.,.5],[0.,1.]]); D=np.array([[2.],[.3]])
    P=np.array([[2.,.3],[.3,1.]]); V=np.array([[.4,.1],[.1,.7]]); W=.01*np.eye(2)
    mean=np.array([1.,-2.]); y=np.array([.3,.7]); u=np.array([.5])
    src=create(k,'StateSpaceSystem',A=A.tolist(),B=B.tolist(),C=C.tolist(),D=D.tolist(),time_domain='discrete',sample_time=1)
    r=prior(k,src,mean=mean.tolist(),covariance=P.tolist(),W=W.tolist(),V=V.tolist(),mode='numeric')
    # Independent information-form posterior, not the implementation's gain/Joseph expression.
    precision=np.linalg.inv(P)+C.T@np.linalg.solve(V,C)
    reference_P=np.linalg.inv(precision)
    reference_mean=np.linalg.solve(precision,np.linalg.solve(P,mean)+C.T@np.linalg.solve(V,y-D@u))
    post=run(k,child(r),'update',measurement=y.tolist(),control=u.tolist())
    state=k.math_objects[child(post)]['value']
    np.testing.assert_allclose(np.array(state.mean,float),reference_mean,atol=1e-12)
    np.testing.assert_allclose(np.array(state.covariance,float),reference_P,atol=1e-12)
    assert not post.data['details']['accepted'] and all(post.data['details']['checks'].values())
    for _ in range(30):
        predicted=run(k,child(post),'predict')
        post=run(k,child(predicted),'update',measurement=y.tolist(),control=u.tolist())
    state=k.math_objects[child(post)]['value']
    assert np.linalg.eigvalsh(np.array(state.covariance,float)).min()>0


@pytest.mark.parametrize('discrete',[False,True])
def test_lqg_separation_feedthrough_certificates_and_stability(discrete):
    k=MathKernel(); src=system(k,discrete,A=int(discrete),D=3)
    cert=create(k,'RiccatiCertificate',P=[[1]])
    args=dict(Q=[['1/2' if discrete else 1]],R=[[1]],W=[['1/2' if discrete else 1]],V=[[1]],
              lqr_certificate_id=cert,kalman_certificate_id=cert)
    result=run(k,src,'lqg',**args)
    assert result.data['details']['accepted']
    gain=sp.Rational(1,2) if discrete else sp.S.One
    plant=k.math_objects[src]['value']; controller=k.math_objects[child(result)]['value']
    assert sp.Matrix(controller.A)==sp.Matrix([[int(discrete)-2*gain+3*gain**2]])
    # Direct plant/controller interconnection, including u=Cc*xhat and plant D.
    A,B,C,D=map(sp.Matrix,(plant.A,plant.B,plant.C,plant.D))
    Ac,Bc,Cc,Dc=map(sp.Matrix,(controller.A,controller.B,controller.C,controller.D))
    direct=A.row_join(B*Cc).col_join((Bc*C).row_join(Ac+Bc*D*Cc))
    closed=result.data['object_ids']['closed_loop']; closed_obj=k.math_objects[closed]['value']
    assert direct==sp.Matrix(closed_obj.A)
    assert run(k,closed,'stability').data['value'] is True
    if not discrete: assert run(k,child(result),'stability').data['value'] is False
    assert not result.data['details']['stochastic_optimality_claimed']
    for objid in result.data['object_ids'].values():
        assert set(k.object_get(objid).data['sources'])=={src,cert}


def test_lqg_bad_witness_and_numeric_search():
    k=MathKernel(); src=system(k,True,A=1)
    args=dict(Q=[[1]],R=[[1]],W=[[1]],V=[[1]])
    r=run(k,src,'lqg',**args,mode='numeric')
    assert r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']
    assert all(r.data['details']['checks'].values()) and 'output' in r.data['object_ids']
    bad=create(k,'RiccatiCertificate',P=[[0]])
    r=run(k,src,'lqg',**args,lqr_certificate_id=bad,kalman_certificate_id=bad)
    assert not r.data['details']['accepted'] and 'output' not in r.data['object_ids']
    assert not k.apply(src,'lqg',{**args,'lqr_certificate_id':bad}).ok


def test_mimo_lqg_shapes_units_and_exact_block_identity():
    k=MathKernel()
    src=create(k,'StateSpaceSystem',A=[[0,0],[0,0]],B=[[1,0],[0,1]],
        C=[[1,0],[0,1]],D=[[1,2],[3,4]],state_units=['m','kg'],
        input_units=['m/s','kg/s'],output_units=['m','kg'])
    cert=create(k,'RiccatiCertificate',P=[[1,0],[0,1]])
    result=run(k,src,'lqg',Q=[[1,0],[0,1]],R=[[1,0],[0,1]],
        W=[[1,0],[0,1]],V=[[1,0],[0,1]],lqr_certificate_id=cert,
        kalman_certificate_id=cert)
    assert result.data['details']['accepted']
    controller=k.math_objects[result.data['object_ids']['output']]['value']
    closed=k.math_objects[result.data['object_ids']['closed_loop']]['value']
    assert controller.input_units==('m','kg')
    assert controller.output_units==('m/s','kg/s')
    assert closed.state_units==('m','kg','m','kg')
    assert closed.input_units==('m/s','kg/s','m','kg')
    assert sp.Matrix(result.data['details']['separated_A']) == sp.Matrix([
        [-1,0,1,0],[0,-1,0,1],[0,0,-1,0],[0,0,0,-1]])


def test_continuous_lqg_process_noise_rate_units():
    k=MathKernel(); src=create(k,'StateSpaceSystem',A=[[0]],B=[[1]],C=[[1]],D=[[0]],
        state_units=['m'],input_units=['m/s'],output_units=['m'])
    cert=create(k,'RiccatiCertificate',P=[[1]])
    result=run(k,src,'lqg',Q=[[1]],R=[[1]],W=[[1]],V=[[1]],
        lqr_certificate_id=cert,kalman_certificate_id=cert)
    closed=k.math_objects[result.data['object_ids']['closed_loop']]['value']
    assert closed.input_units==('m/s','m')


@pytest.mark.parametrize('kind',['weight','measurement','certificate','state'])
def test_decimal_cancellation_ancestry_survives(kind):
    k=MathKernel(); src=system(k,True,A=1)
    decimal='1+(1.0-1.0)'
    if kind=='weight':
        r=finite(k,src,Q=[[decimal]],mode="numeric")
        replay=run(k,child(r),'verify')
    elif kind=='state':
        r=finite(k,src)
        replay=run(k,child(r),'control',state=[decimal],step=0)
    elif kind=='measurement':
        r=prior(k,src)
        post=run(k,child(r),'update',measurement=[decimal],control=[0])
        replay=run(k,child(post),'predict')
    else:
        cert=create(k,'RiccatiCertificate',P=[[decimal]])
        replay=run(k,src,'lqg',Q=[['1/2']],R=[[1]],W=[['1/2']],V=[[1]],
                   lqr_certificate_id=cert,kalman_certificate_id=cert)
        assert 'output' not in replay.data['object_ids']
    assert replay.trust==TrustLevel.NUMERIC and not replay.data['details']['accepted']


@pytest.mark.parametrize('field,value',[('horizon',True),('horizon',-1),('horizon',33),
    ('terminal',[[-1]]),('R',[[0]]),('Q',[[1,0],[0,1]]),('mode','fast'),('terminal',[[True]])])
def test_finite_invalid_parameters(field,value):
    k=MathKernel(); src=system(k,True,A=1)
    params=dict(Q=[[1]],R=[[1]],terminal=[[1]],horizon=2);params[field]=value
    assert not k.apply(src,'finite_lqr',params).ok


@pytest.mark.parametrize('field,value',[('index',True),('index',-1),('mean',[1,2]),
    ('covariance',[[-1]]),('V',[[0]]),('W',[['x']])])
def test_kalman_invalid_parameters(field,value):
    k=MathKernel(); src=system(k,True,A=1)
    params=dict(mean=[0],covariance=[[1]],W=[[1]],V=[[1]]);params[field]=value
    assert not k.apply(src,'kalman_state',params).ok


def test_resource_and_time_domain_limits():
    k=MathKernel(Settings(max_engineering_work=1));src=system(k,True,A=1)
    assert not k.apply(src,'finite_lqr',dict(Q=[[1]],R=[[1]],terminal=[[1]],horizon=2)).ok
    k=MathKernel(Settings(max_control_order=1));src=system(k)
    assert not k.apply(src,'lqg',dict(Q=[[1]],R=[[1]],W=[[1]],V=[[1]],mode='numeric')).ok
    k=MathKernel();src=system(k)
    assert not k.apply(src,'finite_lqr',dict(Q=[[1]],R=[[1]],terminal=[[1]],horizon=2)).ok
    assert not k.apply(src,'kalman_state',dict(mean=[0],covariance=[[1]],W=[[1]],V=[[1]])).ok
    src=system(k,True,A=1);r=finite(k,src,horizon=0)
    assert not k.apply(child(r),'control',dict(state=[1],step=0)).ok
    assert run(k,child(r),'rollout',initial=[2]).data['details']['total_cost']=='4'


def test_sqlite_policy_kalman_restart_and_discovery(tmp_path):
    settings=Settings(store_path=str(tmp_path/'d5.sqlite'));k=MathKernel(settings)
    src=system(k,True,A=1,D=2);policy=finite(k,src);state=prior(k,src)
    post=run(k,child(state),'update',measurement=[3],control=[1])
    restarted=MathKernel(settings)
    assert run(restarted,child(policy),'verify').data['details']['accepted']
    nxt=run(restarted,child(post),'predict')
    assert nxt.data['details']['accepted']
    assert restarted.math_objects[child(nxt)]['value'].mean==(sp.Rational(3,2),)
    assert set(restarted.object_get(child(nxt)).data['sources'])=={child(post)}
    assert set(restarted.object_get(child(post)).data['sources'])=={child(state)}
    assert set(restarted.object_get(child(state)).data['sources'])=={src}
    assert not k.object_create('KalmanState',{}).ok  # derived-only objects
    assert not k.object_create('FiniteHorizonLQR',{}).ok
    caps=k.capability_query(domain='control')
    assert 'finite_lqr' in str(caps) and 'KalmanState' in str(caps)


def test_live_mcp_policy_and_filter():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def go():
        async with Client(mcp) as client:
            async def call(name,**args):
                result=(await client.call_tool(name,args)).data
                assert result['ok'],result
                return result['data']
            obj=await call('math_object_create',object_type='StateSpaceSystem',definition={
                'A':[[1]],'B':[[1]],'C':[[1]],'D':[[0]],'time_domain':'discrete','sample_time':1})
            policy=await call('math_apply',object_id=obj['object_id'],operation='finite_lqr',parameters={
                'Q':[[1]],'R':[[1]],'terminal':[[1]],'horizon':2})
            assert policy['details']['accepted']
            check=await call('math_apply',object_id=policy['object_ids']['output'],operation='rollout',parameters={'initial':[2]})
            assert check['details']['total_cost']=='32/5'
            state=await call('math_apply',object_id=obj['object_id'],operation='kalman_state',parameters={
                'mean':[0],'covariance':[[1]],'W':[[1]],'V':[[1]]})
            post=await call('math_apply',object_id=state['object_ids']['output'],operation='update',parameters={'measurement':[1],'control':[0]})
            predicted=await call('math_apply',object_id=post['object_ids']['output'],operation='predict',parameters={})
            assert predicted['details']['index']==1
    asyncio.run(go())
