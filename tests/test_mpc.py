import asyncio
import copy

import numpy as np
import pytest
import sympy as sp

from mathkernel import MPCPlan, MathKernel, ResultStatus, Settings, TrustLevel
from mathkernel.mpc import verify_plan
from test_engineering_signal import create


def system(k, *, A=1, B=1, n=1):
    if n == 1:
        matrices=dict(A=[[A]],B=[[B]],C=[[1]],D=[[0]])
    else:
        matrices=dict(A=[[1,1],[0,1]],B=[[0],[1]],C=[[1,0]],D=[[0]])
    return create(k,'StateSpaceSystem',**matrices,time_domain='discrete',sample_time=1)


def params(**updates):
    base=dict(Q=[[1]],R=[[1]],terminal=[[1]],initial=[2],horizon=1,
        state_lower=[-10],state_upper=[10],input_lower=['-1/2'],input_upper=['1/2'],
        terminal_lower=[-2],terminal_upper=[2],terminal_gain=[['1/4']])
    return {**base,**updates}


def run(k,obj,op,arguments=None,**kwargs):
    r=k.apply(obj,op,arguments or kwargs)
    assert r.ok,r.errors
    return r


def test_numeric_search_exact_promotion_and_solver_free_replay(monkeypatch, isolated_patch):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k);before=k.object_get(src).model_dump(mode='json')
    solved=run(k,src,'mpc',params(mode='numeric'))
    assert solved.trust==TrustLevel.EXACT and solved.data['details']['feasibility_certified']
    assert solved.data['details']['optimality_certified']
    assert solved.data['value']=='13/2'
    assert solved.data['details']['terminal_analysis']['recursive_feasibility_certified']
    ids=solved.data['object_ids'];plan=k.math_objects[ids['output']]['value']
    assert isinstance(plan,MPCPlan) and plan.controls==((sp.Rational(-1,2),),)
    from scipy import optimize
    forbidden = lambda *a,**kw: (_ for _ in ()).throw(AssertionError('solver called'))
    monkeypatch.setattr(optimize, 'minimize', forbidden)
    isolated_patch('scipy.optimize', 'minimize', forbidden)
    replay=run(k,src,'mpc',params(certificate_id=ids['certificate']))
    assert replay.data['details']['optimality_certified']
    check=run(k,replay.data['object_ids']['output'],'verify')
    assert check.data['details']['feasibility_certified']
    assert check.data['details']['optimality_certified']
    first=run(k,replay.data['object_ids']['output'],'first_control')
    assert first.data['value']==['-1/2'] and first.data['details']['requires_reoptimization']
    assert k.object_get(src).model_dump(mode='json')==before


def test_separate_terminal_and_stability_claims():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k)
    result=run(k,src,'mpc',params(mode='numeric'))
    terminal=result.data['details']['terminal_analysis']
    assert terminal['terminal_set_invariant'] and terminal['terminal_control_admissible']
    assert terminal['terminal_feedback_stability_certified']
    assert terminal['recursive_feasibility_certified']
    assert not terminal['closed_loop_mpc_stability_claimed']
    controller=k.math_objects[result.data['object_ids']['terminal_controller']]['value']
    assert controller.A==((sp.Rational(3,4),),)
    bad=run(k,src,'mpc',params(mode='numeric',terminal_gain=[[3]]))
    terminal=bad.data['details']['terminal_analysis']
    assert not terminal['terminal_set_invariant']
    assert not terminal['terminal_control_admissible']
    assert not terminal['recursive_feasibility_certified']
    # A stable terminal gain is not silently interpreted as MPC-loop stability.
    assert not terminal['closed_loop_mpc_stability_claimed']


def test_plan_tampering_breaks_feasibility_and_plan_optimality():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k);solved=run(k,src,'mpc',params(mode='numeric'))
    plan=k.math_objects[solved.data['object_ids']['output']]['value']
    bad=MPCPlan(**{**plan.model_dump(mode='python'),'states':((sp.Integer(2),),(sp.Integer(9),))})
    report,_=verify_plan(bad,k.math_objects[src]['value'])
    assert not report.details['feasibility_certified']
    assert report.details['problem_optimum_certified']
    assert not report.details['plan_matches_certificate']
    assert not report.details['optimality_certified']


def test_certified_infeasible_mpc_preserves_outcome():
    from mathkernel.mpc import build_problem
    from mathkernel.optimization_proofs import find_outcome
    k=MathKernel();src=system(k);plant=k.math_objects[src]['value']
    problem,_=build_problem(plant,((sp.S.One,),),((sp.S.One,),),((sp.S.One,),),(sp.Integer(2),),1,
        terminal_lower=(sp.S.Zero,),terminal_upper=(sp.S.Zero,),input_lower=(sp.S.Zero,),input_upper=(sp.S.Zero,))
    proof=find_outcome(problem,'infeasible');assert proof and proof.details['accepted']
    certificate=create(k,'OptimizationCertificate',**proof.details['certificate'])
    result=run(k,src,'mpc',params(initial=[2],input_lower=[0],input_upper=[0],
        terminal_lower=[0],terminal_upper=[0],terminal_gain=None,
        state_lower=[None],state_upper=[None],certificate_id=certificate))
    assert result.semantic_status==ResultStatus.INFEASIBLE
    assert result.data['details']['accepted']
    assert not result.data['details']['plan_created']
    assert 'output' not in result.data['object_ids']


def test_invalid_certificate_creates_no_plan():
    k=MathKernel();src=system(k)
    bad=create(k,'OptimizationCertificate',primal=[2,2,0],inequality_dual=[0]*6,equality_dual=[0,0])
    result=run(k,src,'mpc',params(certificate_id=bad))
    assert not result.data['details']['accepted']
    assert 'output' not in result.data['object_ids']


def test_numeric_irrational_data_stays_candidate_but_feasibility_is_checked():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k,A='sqrt(2)')
    result=run(k,src,'mpc',params(mode='numeric',input_lower=[-10],input_upper=[10],
        state_lower=[-100],state_upper=[100],terminal_lower=[-100],terminal_upper=[100],terminal_gain=None))
    assert result.trust==TrustLevel.NUMERIC
    assert not result.data['details']['feasibility_certified']
    assert not result.data['details']['optimality_certified']
    assert all(result.data['details']['feasibility_checks'].values())
    assert result.data['details']['plan_created'] and 'output' in result.data['object_ids']
    check=run(k,result.data['object_ids']['output'],'verify')
    assert check.semantic_status==ResultStatus.CANDIDATE
    assert all(check.data['details']['feasibility_checks'].values())


def test_mimo_horizon_matches_independent_condensed_solution():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k,n=2)
    Q=np.diag([2.,1.]);R=np.array([[3.]]);F=np.diag([4.,2.]);x0=np.array([.3,-.2]);N=3
    result=run(k,src,'mpc',dict(Q=Q.tolist(),R=R.tolist(),terminal=F.tolist(),initial=x0.tolist(),
        horizon=N,state_lower=[-100,-100],state_upper=[100,100],input_lower=[-100],input_upper=[100],mode='numeric'))
    plan=k.math_objects[result.data['object_ids']['output']]['value']
    A=np.array([[1.,1.],[0.,1.]]);B=np.array([[0.],[1.]])
    T=np.vstack([np.linalg.matrix_power(A,t) for t in range(N+1)])
    U=np.zeros((2*(N+1),N))
    for t in range(1,N+1):
        for j in range(t):U[2*t:2*t+2,j:j+1]=np.linalg.matrix_power(A,t-j-1)@B
    from scipy.linalg import block_diag
    weights=block_diag(*([Q]*N+[F]));controls=np.linalg.solve(U.T@weights@U+3*np.eye(N),-U.T@weights@T@x0)
    np.testing.assert_allclose(np.asarray(plan.controls,float).ravel(),controls,atol=2e-7)
    np.testing.assert_allclose(np.asarray(plan.states,float).ravel(),T@x0+U@controls,atol=2e-7)


@pytest.mark.parametrize('mutation',[
    {'horizon':True},{'horizon':0},{'horizon':33},{'Q':[[-1]]},{'R':[[0]]},
    {'terminal':[[-1]]},{'initial':[1,2]},{'state_lower':[2],'state_upper':[1]},
    {'terminal_lower':[-11]},{'input_lower':[0,1]},{'terminal_gain':[[1,2]]},
    {'state_lower':[None],'state_upper':[None],'input_lower':[None],'input_upper':[None],
     'terminal_lower':[None],'terminal_upper':[None]}, {'mode':'fast'}])
def test_invalid_parameters(mutation):
    k=MathKernel();src=system(k);arguments=params(**mutation)
    if 'mode' not in mutation:arguments['mode']='numeric'
    assert not k.apply(src,'mpc',arguments).ok


def test_exact_requires_certificate_and_continuous_is_rejected():
    k=MathKernel();src=system(k)
    assert not k.apply(src,'mpc',params()).ok
    continuous=create(k,'StateSpaceSystem',A=[[1]],B=[[1]],C=[[1]],D=[[0]])
    assert not k.apply(continuous,'mpc',params(mode='numeric')).ok
    assert not k.object_create('MPCPlan',{}).ok


def test_resource_limits_and_false_solver_feasibility(monkeypatch):
    k=MathKernel(Settings(max_mpc_horizon=1));src=system(k)
    assert not k.apply(src,'mpc',params(horizon=2,mode='numeric')).ok
    k=MathKernel(Settings(max_optimization_variables=2));src=system(k)
    assert not k.apply(src,'mpc',params(mode='numeric')).ok
    k=MathKernel(Settings(max_engineering_work=20));src=system(k)
    assert not k.apply(src,'mpc',params(horizon=2,mode='numeric')).ok
    k=MathKernel(Settings(max_optimization_constraints=5));src=system(k)
    assert not k.apply(src,'mpc',params(mode='numeric')).ok
    from mathkernel import engines
    k=MathKernel();src=system(k)
    fake={'x':np.array([2.,99.,0.]),'status':0,'message':'pretend optimal'}
    monkeypatch.setattr(engines,'run_in_subprocess',lambda *a,**kw:fake)
    result=run(k,src,'mpc',params(mode='numeric'))
    assert not result.data['details']['plan_created']
    assert 'output' not in result.data['object_ids']


@pytest.mark.parametrize('field,value',[('Q',[['1+(1.0-1.0)']]),
    ('input_lower',["-1/2+(1.0-1.0)"]),('initial',["2+(1.0-1.0)"])])
def test_decimal_cancellation_cannot_promote(field,value):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k);result=run(k,src,'mpc',params(**{field:value,'mode':'numeric'}))
    assert result.trust==TrustLevel.NUMERIC
    assert not result.data['details']['feasibility_certified']
    assert not result.data['details']['optimality_certified']


def test_decimal_certificate_cannot_promote_or_create_plan():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k=MathKernel();src=system(k);solved=run(k,src,'mpc',params(mode='numeric'))
    witness=copy.deepcopy(solved.data['details']['search']['certificate'])
    witness['primal'][0]='2+(1.0-1.0)'
    certificate=create(k,'OptimizationCertificate',**witness)
    replay=run(k,src,'mpc',params(certificate_id=certificate))
    assert replay.trust==TrustLevel.NUMERIC
    assert not replay.data['details']['accepted']
    assert 'output' not in replay.data['object_ids']


def test_outputs_sources_persistence_and_live_mcp(tmp_path):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    settings=Settings(store_path=str(tmp_path/'mpc.sqlite'));k=MathKernel(settings);src=system(k)
    result=run(k,src,'mpc',params(mode='numeric'));ids=result.data['object_ids']
    for object_id in ids.values():assert src in k.object_get(object_id).data['sources']
    restarted=MathKernel(settings);check=run(restarted,ids['output'],'verify')
    assert check.data['details']['optimality_certified']
    assert run(restarted,ids['output'],'first_control').data['value']==['-1/2']
    caps=k.capability_query(domain='control')
    assert 'MPCPlan' in str(caps) and 'terminal_box_invariance' in str(caps)
    from fastmcp import Client
    from mathkernel_mcp.server import mcp as server
    async def go():
        async with Client(server) as client:
            made=(await client.call_tool('math_object_create',{'object_type':'StateSpaceSystem','definition':{
                'A':[[1]],'B':[[1]],'C':[[1]],'D':[[0]],'time_domain':'discrete','sample_time':1}})).data
            answer=(await client.call_tool('math_apply',{'object_id':made['data']['object_id'],'operation':'mpc',
                'parameters':params(mode='numeric')})).data
            assert answer['ok'] and answer['data']['details']['optimality_certified']
            first=(await client.call_tool('math_apply',{'object_id':answer['data']['object_ids']['output'],
                'operation':'first_control','parameters':{}})).data
            assert first['ok'] and first['data']['value']==['-1/2']
    asyncio.run(go())
