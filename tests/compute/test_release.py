"""R8 replay and installed-contract release gates; no provider calls."""
import copy
import os
import subprocess
import sys
from unittest.mock import patch
import pytest
from mathkernel_compute import ComputeClient, ComputeRequest, CuboidParameters
from mathkernel_compute.models import AttemptRecord, ComputeResultReceipt
from mathkernel_compute.protocol import canonical, decode, digest
from mathkernel_compute.replay import inspect_replay, export_replay
from mathkernel_compute.schemas import schema_catalog, CONTRACTS


def manifest(tmp_path):
    with ComputeClient(state_dir=tmp_path, reconcile_on_open=False) as c:
        plan = c.plan(ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20')))
        attempt = AttemptRecord(attempt_id='attempt_test', workspace_id=c.workspace_id, spec=plan.spec,
            bundle_digest=plan.spec.bundle_digest, execution_digest=plan.execution_digest)
        status = ComputeResultReceipt(job_id='job_test', execution='RECEIVED', verification='PASSED',
            resources='RELEASE_CONFIRMED', artifacts='QUARANTINED', candidate_ref='c'*64)
        return decode(export_replay(plan, attempt, status))


def test_replay_roundtrip_lists_missing_artifacts_without_grants_or_trust(tmp_path):
    m = manifest(tmp_path)
    with patch.dict(os.environ, {'MK_LAMBDA_API_KEY': 'secret-sentinel', 'MODAL_TOKEN_SECRET': 'secret-sentinel'}):
        result = inspect_replay(canonical(m))
    assert result['external_artifacts'] == ('c'*64,) and result['artifacts_included'] is False
    assert result['mathematical_trust'] == 'not-established-by-replay'
    assert result['execution_authority'].startswith('none')
    assert b'secret-sentinel' not in canonical(m) and b'grant_id' not in canonical(m)


@pytest.mark.parametrize('fault', ['spec', 'attempt', 'bundle', 'workspace', 'artifact', 'authority', 'version'])
def test_replay_tampering_rejected_without_state_mutation(tmp_path, fault):
    m = manifest(tmp_path)
    if fault == 'spec': m['plan']['spec']['target'] = 'other'
    if fault == 'attempt': m['attempt']['execution_digest'] = 'd'*64
    if fault == 'bundle': m['attempt']['bundle_digest'] = 'd'*64
    if fault == 'workspace': m['attempt']['workspace_id'] = 'other'
    if fault == 'artifact': m['external_artifacts'] = []
    if fault == 'authority': m['grant'] = {'approved': True}
    if fault == 'version': m['schema'] = 'mk.compute-replay/999'
    with ComputeClient(state_dir=tmp_path, reconcile_on_open=False) as c:
        with pytest.raises(ValueError): inspect_replay(canonical(m))
        assert not c.list() and not c.journal.all('budget_grants')


def test_replay_old_version_is_inert_and_duplicate_keys_are_rejected(tmp_path):
    m = manifest(tmp_path); m['schema'] = 'mk.compute-replay/1'; del m['external_artifacts']
    assert inspect_replay(canonical(m))['external_artifacts'] == ('c'*64,)
    with pytest.raises(ValueError, match='duplicate'):
        inspect_replay(b'{"schema":"mk.compute-replay/1",' + canonical(m)[1:])
    with pytest.raises(ValueError, match='LIMIT'):
        inspect_replay(b' '*1048577)


def test_schemas_are_exported_from_strict_runtime_models_without_state(tmp_path):
    c = schema_catalog()
    assert set(c['contracts']) == {m.__name__ for m in CONTRACTS}
    for model in CONTRACTS:
        assert c['contracts'][model.__name__] == model.model_json_schema()
        assert c['contracts'][model.__name__]['additionalProperties'] is False
    state = tmp_path/'must-not-exist'
    result = subprocess.run([sys.executable, '-m', 'mathkernel_compute.cli', '--state-dir', str(state), 'schemas'],
                            capture_output=True, check=True)
    assert decode(result.stdout) == c and not state.exists()


def test_cli_replay_inspection_has_no_launch_or_journal_effect(tmp_path):
    m = manifest(tmp_path/'source'); path = tmp_path/'replay.json'; path.write_bytes(canonical(m))
    state = tmp_path/'must-not-exist'
    result = subprocess.run([sys.executable, '-m', 'mathkernel_compute.cli', '--state-dir', str(state),
                             'inspect-replay', str(path)], capture_output=True, check=True)
    assert decode(result.stdout)['artifacts_included'] is False and not state.exists()
