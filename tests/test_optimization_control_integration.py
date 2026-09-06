import asyncio
import json
import numpy as np
import pytest
from mathkernel import MathKernel, Settings, TrustLevel
from test_engineering_signal import create


def test_proof_matrix_and_observer_outputs_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / 'd2.sqlite'))
    k = MathKernel(settings)
    p = create(k, 'OptimizationProblem', variables=['x'], c=[-1], upper=['5/2'], integrality=[1])
    proof = k.apply(p, 'certify_milp')
    assert proof.ok, proof.errors
    system = create(k, 'StateSpaceSystem', A=[[-1, 0], [0, -2]], B=[[1, 0], [0, 1]],
                    C=[[1, 1]], D=[[0, 0]])
    matrix = k.apply(system, 'to_transfer_function')
    observer = k.apply(system, 'observer', {'poles': [-3, -4]})
    assert observer.ok, observer.errors
    restarted = MathKernel(settings)
    replay = restarted.apply(p, 'verify_milp_certificate', {'certificate_id': proof.data['object_id']})
    assert replay.ok and replay.data['details']['accepted']
    entry = restarted.apply(matrix.data['object_id'], 'entry', {'input': 1, 'output': 0})
    assert entry.ok and entry.trust == TrustLevel.EXACT
    for role, child in observer.data['object_ids'].items():
        restored = restarted.object_get(child)
        assert restored.ok, (role, restored.errors)
        assert restored.data['sources'] == [system]
        assert restarted.apply(child, 'stability').data['value'] is True
    # A replay must not rewrite the immutable problem or proof.
    assert restarted.object_get(p).model_dump(mode='json') == k.object_get(p).model_dump(mode='json')


def test_multichannel_numeric_rank_matches_independent_matrix_rank():
    rng = np.random.default_rng(29)
    for observable in (False, True):
        n = 8
        a = rng.normal(size=(n, n)) / n
        b, c = rng.normal(size=(n, 3)), rng.normal(size=(2, n))
        k = MathKernel()
        source = create(k, 'StateSpaceSystem', A=a.tolist(), B=b.tolist(), C=c.tolist(), D=np.zeros((2, 3)).tolist())
        result = k.apply(source, 'observability' if observable else 'controllability')
        assert result.ok, result.errors
        blocks = [c @ np.linalg.matrix_power(a, j) if observable else np.linalg.matrix_power(a, j) @ b for j in range(n)]
        matrix = np.vstack(blocks) if observable else np.hstack(blocks)
        assert result.data['value']['rank'] == np.linalg.matrix_rank(matrix)
        assert result.trust == TrustLevel.NUMERIC
        assert result.data['details']['rank_is_tolerance_dependent']


def test_live_mcp_stream_and_milp_proof_workflows():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def go():
        async with Client(mcp) as client:
            async def call(name, **args):
                result = (await client.call_tool(name, args)).data
                assert result['ok'], result.get('errors')
                return result['data']
            filt = await call('math_object_create', object_type='Filter', definition={'numerator': [1], 'denominator': [1, '-1/2']})
            state = await call('math_apply', object_id=filt['object_id'], operation='initial_state', parameters={})
            signal = await call('math_object_create', object_type='Signal', definition={'samples': [1, 0, 0]})
            chunk = await call('math_apply', object_id=state['object_id'], operation='process', parameters={'signal_id': signal['object_id']})
            assert chunk['value'] == ['1', '1/2', '1/4']
            assert set(chunk['object_ids']) == {'output', 'state'}
            problem = await call('math_object_create', object_type='OptimizationProblem', definition={'variables': ['x'], 'c': [-1], 'upper': ['5/2'], 'integrality': [1]})
            proof = await call('math_apply', object_id=problem['object_id'], operation='certify_milp', parameters={})
            checked = await call('math_apply', object_id=problem['object_id'], operation='verify_milp_certificate', parameters={'certificate_id': proof['object_id']})
            assert checked['details']['conclusion'] == 'certified_global_optimum'
    asyncio.run(go())
