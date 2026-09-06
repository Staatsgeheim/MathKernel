import copy
import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel,Settings,TrustLevel
from test_engineering_signal import create


@pytest.mark.parametrize('mode',['exact','numeric'])
@pytest.mark.parametrize('b,a',[([1],[1,'-1/2']),([1,2,1],[1]),([1,2],[2,1]),([2],[1])])
def test_chunked_filter_matches_one_shot_and_preserves_all_sources(mode,b,a):
    k=MathKernel();f=create(k,'Filter',numerator=b,denominator=a,sample_rate=8)
    initial=k.apply(f,'initial_state',{'mode':mode,'unit':'V'})
    assert initial.ok,initial.errors
    state_id=initial.data['object_id'];before=k.object_get(state_id).model_dump(mode='json')
    data=[1,3,-2,4,0,-1,2,1]
    outputs=[]
    for offset,chunk in [(0,data[:3]),(3,data[3:5]),(5,data[5:])]:
        s=create(k,'Signal',samples=chunk,sample_rate=8,start=str(sp.Rational(offset,8)),unit='V')
        result=k.apply(state_id,'process',{'signal_id':s})
        assert result.ok,result.errors
        assert all(result.data['verification'].values())
        for child in result.data['object_ids'].values():
            assert {state_id,s}.issubset(set(k.object_get(child).data['sources']))
        output=k.math_objects[result.data['object_ids']['output']]['value'].samples
        outputs.extend(output)
        state_id=result.data['object_ids']['state']
    s=create(k,'Signal',samples=data,sample_rate=8,unit='V')
    expected=k.apply(f,'apply_signal',{'signal_id':s,'mode':mode})
    actual=k.math_objects[expected.data['object_id']]['value'].samples
    if mode=='exact':assert outputs==list(actual)
    else:np.testing.assert_allclose(list(map(complex,outputs)),list(map(complex,actual)),atol=1e-14)
    assert k.object_get(initial.data['object_id']).model_dump(mode='json')==before
    assert k.math_objects[state_id]['value'].samples_processed==8


def test_sos_streaming_roundtrip_across_restart(tmp_path):
    settings=Settings(store_path=str(tmp_path/'stream.sqlite'))
    k=MathKernel(settings);d=create(k,'FilterDesign',cutoff=[10],sample_rate=100)
    f=k.apply(d,'design',{'mode':'numeric'}).data['object_id']
    initial=k.apply(f,'initial_state',{'mode':'numeric'})
    s=create(k,'Signal',samples=[1]+[0]*49,sample_rate=100)
    first=k.apply(initial.data['object_id'],'process',{'signal_id':s})
    restarted=MathKernel(settings)
    s2=create(restarted,'Signal',samples=[0]*50,sample_rate=100,start='1/2')
    second=restarted.apply(first.data['object_ids']['state'],'process',{'signal_id':s2})
    assert second.ok,second.errors
    assert second.trust==TrustLevel.NUMERIC
    combined=list(k.math_objects[first.data['object_id']]['value'].samples)+list(restarted.math_objects[second.data['object_id']]['value'].samples)
    whole=create(k,'Signal',samples=[1]+[0]*99,sample_rate=100)
    once=k.apply(f,'apply_signal',{'signal_id':whole,'mode':'numeric'})
    np.testing.assert_allclose(list(map(complex,combined)),list(map(complex,k.math_objects[once.data['object_id']]['value'].samples)),atol=1e-14)


def test_stream_time_units_and_multi_output_capacity_are_enforced():
    k=MathKernel();f=create(k,'Filter',numerator=[1]);state=k.apply(f,'initial_state')
    gap=create(k,'Signal',samples=[1],start=2)
    assert not k.apply(state.data['object_id'],'process',{'signal_id':gap}).ok
    unit=create(k,'Signal',samples=[1],unit='V')
    assert not k.apply(state.data['object_id'],'process',{'signal_id':unit}).ok
    k=MathKernel(Settings(max_math_objects=4));f=create(k,'Filter',numerator=[1]);state=k.apply(f,'initial_state')
    s=create(k,'Signal',samples=[1]);before=len(k.math_objects)
    result=k.apply(state.data['object_id'],'process',{'signal_id':s})
    assert not result.ok
    assert len(k.math_objects)==before


def test_multi_output_state_ancestry_cannot_upgrade_decimal_chunk():
    k=MathKernel();f=create(k,'Filter',numerator=[1]);state=k.apply(f,'initial_state')
    s=create(k,'Signal',samples=['0.1'])
    result=k.apply(state.data['object_id'],'process',{'signal_id':s})
    for child in result.data['object_ids'].values():
        assert k.object_get(child).trust==TrustLevel.NUMERIC


def state(k,**extra):
    definition=dict(A=[[0,1],[0,0]],B=[[0],[1]],C=[[1,0]],D=[[0]])
    definition.update(extra)
    return create(k,'StateSpaceSystem',**definition)


@pytest.mark.parametrize('method',['zoh','bilinear'])
@pytest.mark.parametrize('mode',['exact','numeric'])
def test_discretization_matches_independent_scipy_and_rational_nilpotent_formula(method,mode):
    from scipy.signal import cont2discrete
    k=MathKernel();src=state(k)
    result=k.apply(src,'discretize',{'sample_time':'1/10','method':method,'mode':mode})
    assert result.ok,result.errors
    child=k.math_objects[result.data['object_id']]['value']
    expected=cont2discrete(tuple(np.asarray(v) for v in ([[0,1],[0,0]],[[0],[1]],[[1,0]],[[0]])),.1,method=method)
    for actual,want in zip((child.A,child.B,child.C,child.D),expected[:4]):
        np.testing.assert_allclose(np.array(actual,dtype=complex),want,atol=1e-13)
    assert child.sample_time==sp.Rational(1,10)
    assert result.trust==(TrustLevel.EXACT if mode=='exact' else TrustLevel.NUMERIC)


def test_zoh_scalar_exponential_retains_symbolic_semantics():
    k=MathKernel();src=state(k,A=[[-1]],B=[[1]],C=[[1]],D=[[0]])
    result=k.apply(src,'discretize',{'sample_time':1})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.SYMBOLIC
    child=k.math_objects[result.data['object_id']]['value']
    assert sp.simplify(child.A[0][0]-sp.exp(-1))==0
    assert sp.simplify(child.B[0][0]-(1-sp.exp(-1)))==0
    assert result.data['verification']['exponential_differential_identity']


def test_bilinear_transfer_identity_with_feedthrough():
    k=MathKernel();src=state(k,A=[[-1,1],[0,-2]],B=[[0],[1]],C=[[1,0]],D=[[2]])
    original=k.apply(src,'to_transfer_function')
    discrete=k.apply(src,'discretize',{'sample_time':'1/5','method':'bilinear'})
    transformed=k.apply(discrete.data['object_id'],'to_transfer_function')
    z=sp.Symbol('z')
    h=k.math_objects[original.data['object_id']]['value'].expression(10*(z-1)/(z+1))
    hd=k.math_objects[transformed.data['object_id']]['value'].expression(z)
    assert sp.cancel(h-hd)==0


def test_mimo_transfer_entries_frequency_rank_and_units():
    k=MathKernel();src=state(k,A=[[-1,0],[0,-2]],B=[[1,0],[0,1]],C=[[1,0],[0,1]],D=[[0,1],[0,0]],
        state_units=['m','m/s'],input_units=['V','A'],output_units=['m','m/s'])
    transfer=k.apply(src,'to_transfer_function');assert transfer.ok,transfer.errors
    assert transfer.data['object_type']=='TransferMatrix'
    entry=k.apply(transfer.data['object_id'],'entry',{'output':0,'input':1})
    assert entry.ok,entry.errors
    assert k.math_objects[entry.data['object_id']]['value'].numerator==(sp.S.One,)
    response=k.apply(src,'frequency_response',{'frequencies':[0,1,2],'mode':'numeric'})
    assert response.ok,response.errors
    assert response.data['details']['shape']==[3,2,2]
    units=k.apply(src,'coefficient_units')
    assert units.data['value']['A'][0][1]['dimension']=='1'
    assert k.apply(src,'controllability').data['value']['rank']==2
    assert k.apply(src,'observability').data['value']['rank']==2
    assert not k.apply(transfer.data['object_id'],'entry',{'output':True,'input':0}).ok


def test_exact_controller_and_observer_gains_and_error_stability():
    k=MathKernel();src=state(k)
    control=k.apply(src,'place_poles',{'poles':[-1,-2]})
    assert control.ok,control.errors
    assert control.data['details']['gain']==[['2','3']]
    assert control.trust==TrustLevel.EXACT
    assert k.apply(control.data['object_id'],'stability').data['value'] is True
    observer=k.apply(src,'observer',{'poles':[-3,-4]})
    assert observer.ok,observer.errors
    assert observer.data['details']['gain']==[['7'],['12']]
    assert k.apply(observer.data['object_ids']['error'],'stability').data['value'] is True
    model=k.math_objects[observer.data['object_id']]['value']
    assert model.B==((0,7),(1,12))


def test_feedback_updates_output_feedthrough_and_decimal_ancestry():
    k=MathKernel();src=state(k,D=[[2]])
    result=k.apply(src,'state_feedback',{'gain':[['1.0',2]]})
    assert result.ok,result.errors
    model=k.math_objects[result.data['object_id']]['value']
    assert float(model.C[0][0])==-1 and model.C[0][1]==-4
    assert result.trust==TrustLevel.NUMERIC


def test_numeric_mimo_controller_poles_match_requested_polynomial():
    k=MathKernel();src=state(k,A=[[-1,0],[0,-2]],B=[[1,0],[0,1]],C=[[1,0]],D=[[0,0]])
    result=k.apply(src,'place_poles',{'poles':[-3,-4],'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.NUMERIC
    assert result.data['verification']['characteristic_polynomial_numeric']
    model=k.math_objects[result.data['object_id']]['value']
    np.testing.assert_allclose(sorted(np.linalg.eigvals(np.array(model.A,dtype=float))),[-4,-3])


def test_control_adversarial_inputs_do_not_imply_success():
    k=MathKernel();src=state(k,B=[[0],[0]])
    assert not k.apply(src,'place_poles',{'poles':[-1,-2]}).ok
    src=state(k)
    assert not k.apply(src,'place_poles',{'poles':['complex(1,1)',-2]}).ok
    assert not k.apply(src,'discretize',{'sample_time':0}).ok
    unstable=state(k,A=[[2]],B=[[1]],C=[[1]],D=[[0]])
    assert not k.apply(unstable,'discretize',{'sample_time':1,'method':'bilinear'}).ok
    assert not k.apply(src,'coefficient_units').ok
