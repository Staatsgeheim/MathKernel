import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, TrustLevel, ResultStatus
from mathkernel.optimization import exact_psd
from test_engineering_signal import create

LP=dict(variables=['x','y'],c=[-3,-2],A_ub=[[1,1],[1,0],[0,1]],b_ub=[4,2,3])


@pytest.mark.parametrize('mode',['exact','numeric'])
def test_lp_global_optimum_has_original_data_certificate(mode):
    k=MathKernel();src=create(k,'OptimizationProblem',**LP)
    result=k.apply(src,'solve',{'mode':mode})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.EXACT
    assert result.semantic_status==ResultStatus.VERIFIED_EXACT
    assert result.data['value']=='-10'
    assert result.data['details']['accepted']
    certificate=result.data['details']['certificate']
    verified=k.apply(src,'verify_certificate',{'certificate':certificate})
    assert verified.data['details']['accepted']
    certificate['primal'][0]='3'
    rejected=k.apply(src,'verify_certificate',{'certificate':certificate})
    assert not rejected.data['details']['accepted']
    assert rejected.status=='candidate'
    assert rejected.semantic_status==ResultStatus.CANDIDATE


def test_equality_free_variable_negative_bound_and_maximum_cases():
    cases=[
        (dict(variables=['x','y'],c=[1,2],A_eq=[[1,1]],b_eq=[1]),'1'),
        (dict(variables=['x'],c=[1],lower=[-2],upper=[3]),'-2'),
        (dict(variables=['x'],c=[1],A_eq=[[1]],b_eq=[-5],lower=[None]),'-5'),
        (dict(variables=['x'],c=[1],lower=[None],upper=[3],sense='max'),'3'),
    ]
    for definition,answer in cases:
        k=MathKernel();src=create(k,'OptimizationProblem',**definition)
        result=k.apply(src,'solve')
        assert result.ok,result.errors
        assert result.data['details']['accepted'],result.data
        assert result.data['value']==answer


def test_qp_exact_psd_and_numeric_search_certificate():
    k=MathKernel();src=create(k,'OptimizationProblem',variables=['x','y'],c=[-2,-4],Q=[[2,0],[0,2]])
    result=k.apply(src,'solve',{'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.EXACT
    assert result.data['details']['accepted']
    assert result.data['value']=='-5'
    assert result.data['details']['certificate_checks']['convexity']


@pytest.mark.parametrize('matrix,psd',[([[1,1],[1,1]],True),([[0,1],[1,0]],False),([[0,0],[0,1]],True),([[1,2],[2,1]],False)])
def test_psd_singular_and_indefinite_cases(matrix,psd):
    assert exact_psd(sp.Matrix(matrix))[0] is psd


def test_stationary_maximum_is_not_certified_as_minimum():
    k=MathKernel();src=create(k,'OptimizationProblem',variables=['x'],c=[0],Q=[[-2]],lower=[None])
    r=k.apply(src,'verify_certificate',{'certificate':{'primal':[0]}})
    assert r.ok
    assert r.data['details']['certificate_checks']['stationarity']
    assert not r.data['details']['certificate_checks']['convexity']
    assert not r.data['details']['accepted']
    assert r.semantic_status==ResultStatus.CANDIDATE


def test_decimal_original_problem_and_nested_witness_cannot_launder_trust():
    k=MathKernel()
    approximate=create(k,'OptimizationProblem',variables=['x'],c=['1.0'])
    result=k.apply(approximate,'solve',{'mode':'numeric'})
    assert result.trust==TrustLevel.NUMERIC
    assert result.status=='candidate'
    exact=create(k,'OptimizationProblem',variables=['x'],c=[1])
    r=k.apply(exact,'verify_certificate',{'certificate':{'primal':['0.0'],'inequality_dual':[1]}})
    assert r.trust==TrustLevel.NUMERIC
    assert not r.data['details']['accepted']
    cert=create(k,'OptimizationCertificate',primal=['0.0'],inequality_dual=[1])
    r=k.apply(exact,'verify_certificate',{'certificate_id':cert})
    assert r.trust==TrustLevel.NUMERIC
    assert len(r.data['provenance']['required_object_inputs'])==2


def test_milp_solver_bound_is_not_a_certificate():
    k=MathKernel();src=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    r=k.apply(src,'solve',{'mode':'numeric'})
    assert r.ok,r.errors
    assert r.status=='candidate'
    assert r.trust==TrustLevel.NUMERIC
    assert r.data['value']==-2
    assert r.data['details']['solver_bound_verified'] is False
    assert r.data['details']['global_optimality_certified'] is False


@pytest.mark.parametrize('definition',[dict(variables=['x'],c=[1],lower=[2],upper=[1]),dict(variables=['x'],c=[-1])])
def test_solver_outcomes_require_independent_witnesses(definition):
    k=MathKernel();src=create(k,'OptimizationProblem',**definition)
    r=k.apply(src,'solve',{'mode':'numeric'})
    assert r.ok
    expected = 'infeasible' if 'upper' in definition else 'unbounded'
    assert r.semantic_status==ResultStatus(expected)
    assert r.data['details']['conclusion']=='certified_'+expected
    assert r.data['details']['accepted'] is True
    assert all(r.data['details']['certificate_checks'].values())


def test_random_bounded_lps_match_vertex_enumeration():
    rng=np.random.default_rng(431)
    for _ in range(12):
        c=rng.integers(-5,6,2).tolist()
        weights=rng.integers(1,5,2).tolist()
        capacity=int(rng.integers(2,10))
        k=MathKernel();src=create(k,'OptimizationProblem',variables=['x','y'],c=c,A_ub=[weights],b_ub=[capacity])
        result=k.apply(src,'solve',{'mode':'numeric'})
        assert result.ok,result.errors
        assert result.data['details']['accepted'],result.data
        expected=min(sp.S.Zero,sp.Rational(c[0]*capacity,weights[0]),sp.Rational(c[1]*capacity,weights[1]))
        assert result.data['value']==str(expected)


def test_invalid_certificate_controls_and_oversized_inputs_rejected():
    k=MathKernel();src=create(k,'OptimizationProblem',**LP)
    assert not k.apply(src,'verify_certificate',{'certificate':{'primal':[0]}}).ok
    assert not k.apply(src,'solve',{'max_iterations':1.5}).ok
    assert not k.apply(src,'solve',{'mode':'numeric','tolerance':float('nan')}).ok
