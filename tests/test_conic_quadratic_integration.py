import asyncio
import copy
import json
import numpy as np
import pytest
from mathkernel import MathKernel, Settings, TrustLevel, ResultStatus
from mathkernel_artifacts import extract_evidence
from test_engineering_signal import create
from test_conic import soc_definition, psd_definition
from test_quadratic_optimization import ball


@pytest.mark.parametrize('typ,definition',[('ConicProblem',psd_definition()),('QCQP',ball())])
def test_new_models_and_witnesses_roundtrip_and_keep_original_ancestry(tmp_path,typ,definition):
    settings=Settings(store_path=str(tmp_path/'conic.sqlite'))
    k=MathKernel(settings);p=create(k,typ,**definition)
    before=k.object_get(p).model_dump(mode='json')
    result=k.apply(p,'solve',{'mode':'numeric'})
    assert result.ok and result.data['details']['accepted']
    restarted=MathKernel(settings)
    assert restarted.object_get(p).model_dump(mode='json')==before
    proof=result.data['object_id'];record=restarted.object_get(proof)
    assert record.ok and record.data['sources']==[p]
    replay=restarted.apply(p,'verify_certificate',{'certificate_id':proof})
    assert replay.ok and replay.trust==TrustLevel.EXACT and replay.data['details']['accepted']
    assert len(replay.data['plan']['obligations'])==4
    bundle,claims=extract_evidence(json.loads(replay.model_dump_json()))
    assert bundle.conservative_trust()=='exact'
    assert all(c.conservative_trust()=='exact' for c in claims.values())


def test_decimal_original_and_derived_witness_stay_numeric_after_restart(tmp_path):
    settings=Settings(store_path=str(tmp_path/'decimal.sqlite'))
    k=MathKernel(settings);definition=psd_definition();definition['b'][1]='1.0';definition['b'][2]='1.0'
    p=create(k,'ConicProblem',**definition);r=k.apply(p,'solve',{'mode':'numeric'})
    restarted=MathKernel(settings)
    assert restarted.object_get(r.data['object_id']).trust==TrustLevel.NUMERIC
    replay=restarted.apply(p,'verify_certificate',{'certificate_id':r.data['object_id']})
    assert replay.ok and replay.trust==TrustLevel.NUMERIC and not replay.data['details']['accepted']


def test_structured_witness_schema_ignores_enum_not_decimal_mathematics():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[1])
    r=k.apply(p,'verify_certificate',{'certificate':{'kind':'optimal','primal':[0],'inequality_dual':[1]}})
    assert r.ok and r.trust==TrustLevel.EXACT and r.data['details']['accepted']
    integer=create(k,'OptimizationProblem',variables=['x'],c=[-1],upper=['5/2'],integrality=[1])
    proof=k.apply(integer,'certify_milp').data['details']['certificate']
    r=k.apply(integer,'verify_milp_certificate',{'certificate':proof})
    assert r.ok and r.trust==TrustLevel.EXACT
    altered=copy.deepcopy(proof);altered['incumbent']=['2.0']
    r=k.apply(integer,'verify_milp_certificate',{'certificate':altered})
    assert r.ok and r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']


def test_missing_optional_solver_preserves_exact_checker(monkeypatch):
    from mathkernel import engines
    monkeypatch.setattr(engines, 'run_in_subprocess',
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError('clarabel deliberately unavailable')))
    k=MathKernel();p=create(k,'ConicProblem',**soc_definition())
    solve=k.apply(p,'solve',{'mode':'numeric'})
    assert not solve.ok and solve.semantic_status==ResultStatus.UNSUPPORTED
    exact=k.apply(p,'verify_certificate',{'certificate':{'primal':[5],'cone_dual':[1,'-3/5','-4/5']}})
    assert exact.ok and exact.trust==TrustLevel.EXACT


def test_fake_conic_solver_flags_are_not_proofs(monkeypatch):
    from mathkernel import engines
    from mathkernel.engineering import EngineeringResult
    monkeypatch.setattr(engines, 'run_in_subprocess', lambda *args, **kwargs: EngineeringResult(
        operation='solve', status='unknown', trust='unknown',
        details={'accepted': False, 'conclusion': 'unverified_solver_report'}))
    k=MathKernel();p=create(k,'ConicProblem',**soc_definition())
    r=k.apply(p,'solve',{'mode':'numeric'})
    assert r.ok and r.semantic_status==ResultStatus.UNKNOWN
    assert not r.data['details']['accepted']
    assert r.data['details']['conclusion']=='unverified_solver_report'


def test_live_mcp_conic_and_quadratic_workflows():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def go():
        async with Client(mcp) as client:
            discovery=(await client.call_tool('math_capability_query',{'domain':'optimization','input_type':'ConicProblem'})).data
            assert {c['operation'] for c in discovery['capabilities']}=={'solve','verify_certificate'}
            for typ,definition in [('ConicProblem',soc_definition()),('QCQP',ball())]:
                source=(await client.call_tool('math_object_create',{'object_type':typ,'definition':definition})).data
                assert source['ok'],source
                result=(await client.call_tool('math_apply',{'object_id':source['data']['object_id'],'operation':'solve','parameters':{'mode':'numeric'}})).data
                assert result['ok'] and result['data']['details']['accepted'],result
                replay=(await client.call_tool('math_apply',{'object_id':source['data']['object_id'],'operation':'verify_certificate','parameters':{'certificate_id':result['data']['object_id']}})).data
                assert replay['ok'] and replay['trust']=='exact'
    asyncio.run(go())


@pytest.mark.parametrize('stored',[False,True])
@pytest.mark.parametrize('typ,definition,cert_type,cert',[
    ('ConicProblem',dict(variables=['x'],c=[1],A=[[-1]],b=[0],cones=[dict(kind='nonnegative',dimension=1)]),
        'ConicCertificate',{'primal':['1.0-1.0'],'cone_dual':[1]}),
    ('OptimizationProblem',dict(variables=['x'],c=[1]),
        'OptimizationCertificate',{'primal':['1.0-1.0'],'inequality_dual':[1]}),
    ('QCQP',dict(variables=['x'],c=[0],Q=[[2]],lower=[None],quadratics=[dict(Q=[[2]],a=[0],r=-1)]),
        'QuadraticCertificate',{'primal':['1.0-1.0'],'quadratic_dual':[0]}),
])
def test_simplified_decimal_witness_never_sets_accepted(typ,definition,cert_type,cert,stored):
    k=MathKernel();p=create(k,typ,**definition)
    params={'certificate_id':create(k,cert_type,**cert)} if stored else {'certificate':cert}
    r=k.apply(p,'verify_certificate',params)
    assert r.ok and r.trust==TrustLevel.NUMERIC
    assert not r.data['details']['accepted'] and r.semantic_status==ResultStatus.CANDIDATE


def test_simplified_decimal_milp_incumbent_cannot_certify():
    k=MathKernel();p=create(k,'OptimizationProblem',variables=['x'],c=[1],integrality=[1])
    proof=k.apply(p,'certify_milp').data['details']['certificate']
    proof['incumbent']=['1.0-1.0']
    r=k.apply(p,'verify_milp_certificate',{'certificate':proof})
    assert r.ok and r.trust==TrustLevel.NUMERIC and not r.data['details']['accepted']
