import asyncio
import json
import pytest
from mathkernel import MathKernel, Settings, TrustLevel
from test_engineering_signal import create


def test_all_engineering_models_roundtrip_replay_and_preserve_source_nodes(tmp_path):
    settings=Settings(store_path=str(tmp_path/'state.sqlite'))
    k=MathKernel(settings)
    cases=[
        ('Signal',dict(samples=[1,2,3,4],sample_rate=4),'dft',{}),
        ('Spectrum',dict(bins=[10,'complex(-2,2)',-2,'complex(-2,-2)'],sample_rate=4),'idft',{}),
        ('Filter',dict(numerator=[1],denominator=[1,'-1/2']),'to_transfer_function',{}),
        ('FilterDesign',dict(cutoff=[10],sample_rate=100),'design',{'mode':'numeric'}),
        ('TransferFunction',dict(numerator=[1],denominator=[1,1]),'stability',{}),
        ('StateSpaceSystem',dict(A=[[-1]],B=[[1]],C=[[1]],D=[[0]]),'to_transfer_function',{}),
        ('OptimizationProblem',dict(variables=['x'],c=[1]),'solve',{}),
    ]
    for typ,definition,operation,parameters in cases:
        source=create(k,typ,**definition)
        before=k.object_get(source).model_dump(mode='json')
        first=k.apply(source,operation,parameters)
        assert first.ok,first.errors
        restarted=MathKernel(settings)
        assert restarted.object_get(source).model_dump(mode='json')==before
        replay=restarted.apply(source,operation,parameters)
        assert replay.ok,replay.errors
        assert replay.data['value']==first.data['value']
        assert replay.trust==first.trust
        if 'object_id' in first.data:
            child=restarted.object_get(first.data['object_id'])
            assert child.ok and child.trust==first.trust
    cert=create(k,'OptimizationCertificate',primal=[0],inequality_dual=[1])
    restarted=MathKernel(settings)
    assert restarted.object_get(cert).ok
    problem=create(k,'OptimizationProblem',variables=['x'],c=[1])
    result=restarted.apply(problem,'verify_certificate',{'certificate_id':cert})
    assert result.data['details']['accepted']


def test_filtered_numeric_result_stays_numeric_across_restart(tmp_path):
    settings=Settings(store_path=str(tmp_path/'numeric.sqlite'))
    k=MathKernel(settings)
    filt=create(k,'Filter',numerator=[1])
    source=create(k,'Signal',samples=['0.5',1])
    result=k.apply(filt,'apply_signal',{'signal_id':source})
    assert result.trust==TrustLevel.NUMERIC
    restarted=MathKernel(settings)
    obj=restarted.object_get(result.data['object_id'])
    assert set(obj.data['sources'])=={source,filt}
    follow=restarted.apply(result.data['object_id'],'dft',{'mode':'numeric'})
    assert follow.trust==TrustLevel.NUMERIC


def test_engineering_capability_discovery_and_live_mcp():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def go():
        async with Client(mcp) as client:
            discovered=await client.call_tool('math_capability_query',{'domain':'optimization','object_type':'OptimizationProblem','operation':'solve'})
            source=await client.call_tool('math_object_create',{'object_type':'OptimizationProblem','definition':{'variables':['x'],'c':[1]}})
            result=await client.call_tool('math_apply',{'object_id':source.data['data']['object_id'],'operation':'solve','parameters':{}})
            return discovered.data,result.data
    discovered,result=asyncio.run(go())
    assert discovered['count']==1
    assert result['ok']
    assert result['data']['details']['conclusion']=='certified_global_optimum'


def test_result_is_compatible_with_shared_artifact_evidence():
    from mathkernel_artifacts import extract_evidence
    k=MathKernel();source=create(k,'Signal',samples=[1,2,3,4])
    result=k.apply(source,'dft',{'mode':'numeric'})
    serialized=json.loads(result.model_dump_json())
    bundle,claims=extract_evidence(serialized)
    assert bundle.conservative_trust()=='numeric'
    assert all(claim.conservative_trust()=='numeric' for claim in claims.values())
    assert serialized['data']['plan']['obligations'][-1]['state']=='succeeded'


def test_optional_scipy_failure_returns_unsupported(monkeypatch):
    # Dependency injection must occur inside the process that runs the solver.
    from mathkernel import engineering_adapter
    original_worker = engineering_adapter._isolated_adapter_apply
    def unavailable_worker(*args, **kwargs):
        import builtins
        original = builtins.__import__
        def blocked(name, *a, **kw):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("deliberately unavailable scipy")
            return original(name, *a, **kw)
        builtins.__import__ = blocked
        try:
            return original_worker(*args, **kwargs)
        finally:
            builtins.__import__ = original
    k=MathKernel();source=create(k,'Signal',samples=[1,2,3])
    monkeypatch.setattr(engineering_adapter, '_isolated_adapter_apply', unavailable_worker)
    out=k.apply(source,'resample',{'mode':'numeric','up':2})
    assert not out.ok
    assert out.semantic_status.value=='unsupported'


def test_operation_timeout_is_reported_without_persisting_a_derived_object(monkeypatch):
    import time
    from mathkernel import engineering_adapter
    k=MathKernel(Settings(solver_timeout_seconds=.005))
    source=create(k,'Signal',samples=[1,2])
    original=engineering_adapter._isolated_adapter_apply
    def slow(*args,**kwargs):
        time.sleep(.02)
        return original(*args,**kwargs)
    monkeypatch.setattr(engineering_adapter,'_isolated_adapter_apply',slow)
    result=k.apply(source,'dft')
    assert not result.ok
    assert 'solver_timeout_seconds' in result.errors[0]
    assert len(k.math_objects)==1
