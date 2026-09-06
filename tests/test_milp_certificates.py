import copy
import itertools
import numpy as np
import pytest
from mathkernel import MathKernel,Settings,TrustLevel,ResultStatus
from test_engineering_signal import create


@pytest.mark.parametrize('mode',['exact','numeric'])
@pytest.mark.parametrize('definition,outcome',[
    ({'variables':['x'],'c':[1],'lower':[2],'upper':[1]},'infeasible'),
    ({'variables':['x'],'c':[0],'A_eq':[[1],[1]],'b_eq':[0,1],'lower':[None]},'infeasible'),
    ({'variables':['x'],'c':[-1]},'unbounded'),
    ({'variables':['x'],'c':[1],'lower':[None],'sense':'max'},'unbounded'),
])
def test_original_data_outcomes_are_certified_and_replayable(mode,definition,outcome):
    k=MathKernel();src=create(k,'OptimizationProblem',**definition)
    result=k.apply(src,'solve',{'mode':mode})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.EXACT
    assert result.semantic_status.value==outcome
    assert result.data['details']['conclusion']=='certified_'+outcome
    certificate=create(k,'OptimizationCertificate',**result.data['details']['certificate'])
    replay=k.apply(src,'verify_certificate',{'certificate_id':certificate})
    assert replay.semantic_status.value==outcome
    assert replay.data['details']['accepted']


def test_forged_farkas_and_ray_are_rejected():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[1],upper=[3])
    for cert in [dict(kind='infeasible',inequality_dual=[0,1]),
                 dict(kind='unbounded',primal=[0],ray=[1]),
                 dict(kind='unbounded',primal=[100],ray=[-1])]:
        r=k.apply(p,'verify_certificate',{'certificate':cert})
        assert r.ok and not r.data['details']['accepted']
        assert r.semantic_status==ResultStatus.CANDIDATE


def test_decimal_witness_never_certifies_even_exact_looking_contradiction():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[1],lower=[2],upper=[1])
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'infeasible','inequality_dual':['1.0',1]}})
    assert r.trust==TrustLevel.NUMERIC
    assert not r.data['details']['accepted']
    assert r.semantic_status==ResultStatus.CANDIDATE
    p=create(k,'OptimizationProblem',variables=['x'],c=[1],lower=['2.0'],upper=[1])
    r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.semantic_status==ResultStatus.UNKNOWN


def test_solver_flag_alone_does_not_prove_infeasible(monkeypatch):
    from mathkernel import engines
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[1],upper=[2])
    monkeypatch.setattr(engines,'run_in_subprocess',lambda *args,**kwargs:{'x':None,'status':2,
        'message':'pretend infeasible','inequality_dual':None,'equality_dual':None})
    r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.semantic_status==ResultStatus.UNKNOWN
    assert r.data['details']['conclusion']=='unverified_solver_report'


@pytest.mark.parametrize('mode',['exact','numeric'])
def test_milp_tree_proves_integer_optimum_and_survives_transport(mode):
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    r=k.apply(p,'certify_milp',{'mode':mode})
    assert r.ok,r.errors
    assert r.data['details']['accepted']
    assert r.data['value']=='-2'
    assert r.data['details']['nodes']==3
    assert r.trust==TrustLevel.EXACT
    replay=k.apply(p,'verify_milp_certificate',{'certificate_id':r.data['object_id']})
    assert replay.data['details']['accepted']
    inline=k.apply(p,'verify_milp_certificate',{'certificate':r.data['details']['certificate']})
    assert inline.data['details']['accepted']


def test_integer_infeasibility_requires_complete_partition():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[0],A_eq=[[1]],b_eq=['1/2'],integrality=[1])
    r=k.apply(p,'certify_milp')
    assert r.ok,r.errors
    assert r.semantic_status==ResultStatus.INFEASIBLE
    assert r.data['details']['closed_leaves']==2
    partial=k.apply(p,'certify_milp',{'max_nodes':1})
    assert partial.semantic_status==ResultStatus.CANDIDATE
    assert not partial.data['details']['accepted']
    assert partial.data['details']['termination']=='node budget'


@pytest.mark.parametrize('mutation',['cycle','unreachable','missing_branch','wrong_bound','nonintegral_split'])
def test_milp_tamper_rejection(mutation):
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    solved=k.apply(p,'certify_milp');cert=copy.deepcopy(solved.data['details']['certificate'])
    if mutation=='cycle':cert['nodes'][0]['left']=0
    elif mutation=='unreachable':cert['nodes'].append({'kind':'open'})
    elif mutation=='missing_branch':cert['nodes'][0]['right']=cert['nodes'][0]['left']
    elif mutation=='wrong_bound':cert['nodes'][1]['certificate']['primal']=['3']
    else:cert['nodes'][0]['split']='3/2'
    r=k.apply(p,'verify_milp_certificate',{'certificate':cert})
    assert not r.ok or not r.data['details']['accepted']
    assert r.semantic_status not in {ResultStatus.INFEASIBLE,ResultStatus.VERIFIED_EXACT}


def test_mixed_integer_continuous_problem_and_maximize():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x','y'],c=[2,1],A_ub=[[1,1]],b_ub=['7/2'],upper=[3,3],integrality=[1,0],sense='max')
    r=k.apply(p,'certify_milp',{'mode':'numeric'})
    assert r.ok,r.errors
    assert r.data['details']['accepted']
    assert r.data['value']=='13/2'


def test_small_random_milps_against_exhaustive_grid():
    rng=np.random.default_rng(203)
    for _ in range(8):
        c=rng.integers(-4,5,2).tolist();a=rng.integers(1,4,2).tolist();rhs=int(rng.integers(2,10))
        k=MathKernel();p=create(k,'OptimizationProblem',variables=['x','y'],c=c,A_ub=[a],b_ub=[rhs],upper=[3,3],integrality=[1,1])
        r=k.apply(p,'certify_milp',{'mode':'numeric'})
        assert r.ok and r.data['details']['accepted'],r.model_dump()
        expected=min(c[0]*x+c[1]*y for x,y in itertools.product(range(4),repeat=2) if a[0]*x+a[1]*y<=rhs)
        assert r.data['value']==str(expected)


def test_branch_search_resource_governance():
    k=MathKernel(Settings(max_milp_nodes=3));p=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    assert not k.apply(p,'certify_milp',{'max_nodes':5}).ok
    assert not k.apply(p,'certify_milp',{'max_nodes':True}).ok


def test_milp_replay_never_invokes_a_solver(monkeypatch, isolated_patch):
    from mathkernel import optimization
    import sympy.solvers.simplex as simplex
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    proof=k.apply(p,'certify_milp')
    def forbidden(*args,**kwargs):
        raise AssertionError('certificate replay must not search')
    monkeypatch.setattr(optimization,'solve',forbidden)
    monkeypatch.setattr(simplex,'linprog',forbidden)
    isolated_patch('mathkernel.optimization', 'solve', forbidden)
    isolated_patch('sympy.solvers.simplex', 'linprog', forbidden)
    result=k.apply(p,'verify_milp_certificate',{'certificate_id':proof.data['object_id']})
    assert result.ok and result.data['details']['accepted']


def test_mixed_integer_ray_requires_integral_anchor_and_step():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[-1],integrality=[1])
    for anchor,ray,accepted in [(0,1,True),('1/2',1,False),(0,'1/2',False)]:
        result=k.apply(p,'verify_certificate',{'certificate':{'kind':'unbounded','primal':[anchor],'ray':[ray]}})
        assert result.ok and result.data['details']['accepted'] is accepted
