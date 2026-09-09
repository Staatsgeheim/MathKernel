"""Managed-provider lifecycle, budget, data-boundary and real local-worker tests.

Provider services are deterministic fault fixtures. These are not live GPU tests.
"""
import base64
import io
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from mathkernel_compute import ComputeClient, ComputeRequest, CuboidParameters, LocalPolicy, ResourceRequirements
from mathkernel_compute.models import Money, BudgetLimit, ManagedQuote, ManagedResources, AcceleratorProfile
from mathkernel_compute.managed import ModalProfile, RunpodProfile, S3Storage, BridgeCall, bridge_environment
from mathkernel_compute.registry import runtime_profile, execute
from mathkernel_compute.protocol import canonical, decode, digest, parse
from mathkernel_compute.remote import RemoteUnavailable, AttemptRef
from mathkernel_compute.runpod_adapter import RunpodAdapter, deployment_snapshot, NoRedirect
from mathkernel_compute.managed_worker import RunpodWorkerConfig, runpod_handler
from mathkernel_compute.object_store import S3Store, bound_record, terminal_record
from mathkernel_compute.models import RemoteResultEnvelope
from mathkernel_compute.batch import BatchRequest, BatchShard


def usd(amount):
    return Money(currency='USD', amount=amount)


def budget(amount='2.00'):
    return BudgetLimit(budget_id='pilot', account_scope='research', limit=usd(amount))


def profile(adapter='runpod', **changes):
    kw = dict(target_id='managed', account_scope='research', budget_id='pilot', runtime=runtime_profile(),
        resources=ManagedResources(network='broker-only' if adapter == 'runpod' else 'blocked'),
        quote=ManagedQuote(quote_id='quote', reservation=usd('0.50'), valid_until_ms=int(time.time()*1000)+900000,
            source='test quote, not provider pricing', scope='one bounded invocation, output and idle exposure'),
        credential_env_prefix='MK_TEST', source_image='example.test/mathkernel@sha256:'+'a'*64)
    if adapter == 'runpod':
        kw.update(endpoint_id='endpoint', deployment_digest='b'*64, workers_min=0, workers_max=1,
            idle_timeout_seconds=5, storage=S3Storage(bucket='test-bucket', region='us-east-1',
                prefix='mathkernel', credential_env_prefix='MK_STORE'))
    else:
        kw.update(app_name='testapp', app_id='ap-test', environment='main', image_id='im-test',
                  volume_name='testvolume', volume_id='vo-test', region='us-east')
    kw.update(changes)
    return (RunpodProfile if adapter == 'runpod' else ModalProfile)(**kw)


def request(bound='20', target='managed'):
    return ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound=bound), target=target)


def approve(c, p):
    return c._authorize_managed(p.plan_id, plan_digest=p.digest, bundle_digest=p.spec.bundle_digest,
        target_profile_digest=p.spec.managed.target_profile_digest,
        managed_exposure_acknowledged=True, persistent_storage_allowed=True)


class Cloud:
    def __init__(self):
        self.attempts = {}; self.calls = []; self.lost_ack = False; self.offline = False
        self.execution = 'RUNNING'; self.resources = 'ACTIVE'; self.client = None

    def __call__(self, call):
        self.calls.append(call.action)
        a = call.attempt
        if call.action == 'probe':
            return {'protocol': 'mk.managed-bridge/1', 'adapter': call.profile.adapter}
        r = dict(protocol='mk.managed-bridge/1', attempt_id=a.attempt_id,
                 execution_digest=a.execution_digest, handle='native_' + a.attempt_id)
        if call.action == 'submit':
            # The provider can already see committed authority, intent and reservation.
            assert self.client.journal.get('outbox', a.attempt_id)['state'] == 'DISPATCHING'
            assert any(x['amount'] == '0.50' for x in self.client.journal.all('budget_reservations'))
            self.attempts[a.attempt_id] = a
            if self.lost_ack:
                raise RemoteUnavailable('Acknowledgement lost after acceptance')
        if self.offline:
            raise RemoteUnavailable('Disconnected')
        if call.action == 'observe':
            r['observation'] = dict(execution=self.execution, resources=self.resources, message='fixture')
        if call.action == 'fetch':
            r['candidate_b64'] = base64.b64encode(canonical(RemoteResultEnvelope(
                attempt_id=a.attempt_id, workspace_id=a.workspace_id, bundle_digest=a.bundle_digest,
                execution_digest=a.execution_digest, operation='cuboid_sweep', output_schema='mk.cuboid-pairs/1',
                output=execute(a.spec.bundle.request), worker_claims={'trust': 'formal', 'verified': True}))).decode()
        return r


def client(path, p, cloud, limit='2.00', **kw):
    c = ComputeClient(state_dir=path, managed_targets=(p,), budget_limits=(budget(limit),),
        _managed_call=cloud, **kw)
    cloud.client = c
    return c


def launch(c, key='single'):
    p = c.plan(request()); g = approve(c, p)
    return c.submit(plan_id=p.plan_id, authorization_ref=g.grant_id, client_request_id=key), p, g


def test_discovery_and_plan_are_offline_paid_export_is_distinct(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        assert c.targets()[-1].adapter == 'runpod'
        plan = c.plan(request())
        assert not cloud.calls
        assert plan.provider_cost == usd('0.50')
        assert plan.spec.managed.quote.hard_spend_cap_supported is False
        with pytest.raises(ValueError): plan.spec.managed.provider_options.workers_min = 10
        for method in (lambda: c._authorize_local(plan.plan_id), lambda: c._authorize_remote(plan.plan_id,
                plan_digest=plan.digest, bundle_digest=plan.spec.bundle_digest, target_profile_digest=p.profile_digest)):
            with pytest.raises(PermissionError): method()
        with pytest.raises(ValueError):
            c._authorize_managed(plan.plan_id, plan_digest=plan.digest, bundle_digest=plan.spec.bundle_digest,
                target_profile_digest=p.profile_digest, managed_exposure_acknowledged=False, persistent_storage_allowed=True)
        assert not cloud.calls and not c.journal.all('budget_grants')


def test_lost_ack_reopen_recovers_same_handle_without_replacement(tmp_path):
    p = profile(); cloud = Cloud(); cloud.lost_ack = True
    with client(tmp_path, p, cloud, limit='0.50') as c:
        j, plan, grant = launch(c)
        assert c.status(j.job_id).execution == 'SUBMISSION_UNKNOWN'
        assert c.status(j.job_id).cost == 'EXPOSURE_UNKNOWN'
    with client(tmp_path, p, cloud, limit='0.50', reconcile_on_open=False) as c:
        assert c.reconcile(j.job_id).execution == 'RUNNING'
        assert c.submit(plan_id=plan.plan_id, authorization_ref=grant.grant_id, client_request_id='single') == j
        assert cloud.calls.count('submit') == 1
        assert c.budget_status('pilot')['available'] == '0.00'
        with pytest.raises(PermissionError, match='BUDGET'): launch(c, 'second')


def test_cleanup_and_local_admission_do_not_close_paid_reservation(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='0.50') as c:
        j, _, _ = launch(c)
        cloud.execution = 'RECEIVED'; cloud.resources = 'RELEASE_CONFIRMED'
        assert c.reconcile(j.job_id).cost == 'EXPOSURE_UNKNOWN'
        assert c.verify(j.job_id).verification == 'PASSED'
        assert c.accepted_result(j.job_id).trust.value == 'exact'
        assert c.budget_status('pilot')['available'] == '0.00'
        with pytest.raises(ValueError):
            c._reconcile_cost(j.job_id, usd('0.20'), 'reviewed bill', retained_storage_accounted=False)
        c._reconcile_cost(j.job_id, usd('0.20'), 'reviewed bill including storage', retained_storage_accounted=True)
        c._reconcile_cost(j.job_id, usd('0.20'), 'reviewed bill including storage', retained_storage_accounted=True)
        assert c.status(j.job_id).cost == 'USER_RECONCILED'
        assert c.reconcile(j.job_id).cost == 'USER_RECONCILED'
        assert c.budget_status('pilot')['available'] == '0.30'
        with pytest.raises(ValueError, match='IMMUTABLE'):
            c._reconcile_cost(j.job_id, usd('0.00'), 'erase spending', retained_storage_accounted=True)


def test_cancel_after_candidate_still_releases_sandbox_and_retains_facts(tmp_path):
    p = profile('modal'); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c)
        cloud.execution = 'RECEIVED'
        c.reconcile(j.job_id)
        c.cancel(j.job_id)
        assert 'cancel' in cloud.calls and c.status(j.job_id).resources == 'ACTIVE'
        cloud.execution = 'CANCELLED'; cloud.resources = 'RELEASE_CONFIRMED'
        assert c.reconcile(j.job_id).execution == 'RECEIVED'
        assert c.reconcile(j.job_id).resources == 'RELEASE_CONFIRMED'
        assert c.status(j.job_id).cost == 'EXPOSURE_UNKNOWN'


def test_unknown_cleanup_and_actual_overspend_block_more_work(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='1.00') as c:
        j, _, _ = launch(c)
        with pytest.raises(PermissionError, match='EXPOSURE'):
            c._reconcile_cost(j.job_id, usd('0.00'), 'no bill yet', retained_storage_accounted=True)
        cloud.offline = True
        assert c.reconcile(j.job_id).resources == 'CLEANUP_UNKNOWN'
        cloud.offline = False; cloud.execution = 'FAILED'; cloud.resources = 'RELEASE_CONFIRMED'
        c.reconcile(j.job_id)
        c._reconcile_cost(j.job_id, usd('1.20'), 'actual exceeded estimate', retained_storage_accounted=True)
        assert c.budget_status('pilot')['available'] == '-0.20'
        with pytest.raises(PermissionError, match='BUDGET'): launch(c, 'next')


def test_exact_gpu_selection_no_implicit_allocation(tmp_path):
    p = profile()
    accelerator = AcceleratorProfile(cupy_version='test', cuda_runtime=12000, cuda_driver=12000)
    gpu = p.model_copy(update={'runtime': p.runtime.model_copy(update={'accelerator': accelerator}),
                              'resources': p.resources.model_copy(update={'gpu': 'ADA_24'})})
    with client(tmp_path, gpu, Cloud()) as c:
        with pytest.raises(ValueError, match='ACCELERATOR'): c.plan(request())
        req = request().model_copy(update={'parameters': CuboidParameters(bound='20', engine='cuda'),
                                          'resources': ResourceRequirements(gpu_count=1)})
        assert c.plan(req).spec.runtime.accelerator == accelerator
    with pytest.raises(ValueError):
        ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20', engine='cuda'))


def test_expired_quote_and_changed_target_do_not_export(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        plan = c.plan(request()); grant = approve(c, plan)
        c.managed_targets[p.target_id] = p.model_copy(update={'idle_timeout_seconds': 10})
        with pytest.raises(PermissionError):
            c.submit(plan_id=plan.plan_id, authorization_ref=grant.grant_id, client_request_id='changed')
        assert not cloud.calls and not c.journal.all('jobs')
    with client(tmp_path, p.model_copy(update={'quote': p.quote.model_copy(update={'valid_until_ms': 1})}), cloud) as c:
        with pytest.raises(PermissionError, match='QUOTE_EXPIRED'): c.plan(request())


def test_credentials_only_enter_control_bridge_environment():
    p = profile()
    with patch.dict('os.environ', {'MK_TEST_API_KEY': 'api-secret', 'MK_STORE_ACCESS_KEY_ID': 'key',
            'MK_STORE_SECRET_ACCESS_KEY': 'store-secret', 'UNRELATED_SECRET': 'secret', 'HTTPS_PROXY': 'proxy'}):
        env = bridge_environment(p)
        assert env['MK_TEST_API_KEY'] == 'api-secret'
        assert 'UNRELATED_SECRET' not in env and 'HTTPS_PROXY' not in env
        from mathkernel_compute.executor import worker_environment
        assert all(v not in worker_environment().values() for v in ('api-secret', 'store-secret', 'secret'))
    with pytest.raises(RemoteUnavailable): bridge_environment(p)


class StoreError(Exception):
    def __init__(self, code): self.response = {'Error': {'Code': code}}


class MemoryS3:
    def __init__(self): self.objects = {}; self.writes = []; self.last_body = None
    def put_object(self, **kw):
        assert kw['IfNoneMatch'] == '*'
        key = (kw['Bucket'], kw['Key'])
        if key in self.objects: raise StoreError('PreconditionFailed')
        self.objects[key] = kw['Body']; self.writes.append(kw)
        return {}
    def get_object(self, **kw):
        key = (kw['Bucket'], kw['Key'])
        if key not in self.objects: raise StoreError('NoSuchKey')
        raw = self.objects[key]; self.last_body = io.BytesIO(raw)
        return {'ContentLength': len(raw), 'Body': self.last_body}


def endpoint(p):
    return {'id': p.endpoint_id, 'type': 'QUEUE', 'image': p.source_image,
            'workers': {'min': 0, 'max': 1, 'idleTimeout': 5}, 'dataCenterIds': ['US-KS-2'],
            'cpu': [{'type': 'cpu3c', 'count': 1}], 'env': {'SECRET': 'never-return'}}


def staged(tmp_path):
    p = profile(); p = p.model_copy(update={'deployment_digest': digest(deployment_snapshot(endpoint(p)))})
    cloud = Cloud()
    c = client(tmp_path, p, cloud)
    j, plan, _ = launch(c)
    return c, p, c._attempt(c._job(j.job_id))


def test_runpod_real_worker_durable_output_survives_status_expiry_and_duplicate_delivery(tmp_path):
    c, p, a = staged(tmp_path)
    try:
        memory = MemoryS3(); store = S3Store(p.storage, _client=memory)
        store.put(a, 'request', canonical(a))
        config = RunpodWorkerConfig(target_id=p.target_id, account_scope=p.account_scope, endpoint_id=p.endpoint_id,
            runtime=p.runtime, resources=p.resources, source_image=p.source_image, storage=p.storage,
            allowed_workspaces=(a.workspace_id,))
        job = {'id': 'provider_job', 'input': bound_record(a, request_digest=digest(a))}
        assert runpod_handler(job, config, _store=store)['state'] == 'DURABLE_OUTPUT_RETAINED'
        assert runpod_handler(job, config, _store=store)['state'] == 'DUPLICATE_DELIVERY_NO_EXECUTION'
        kinds = [w['Key'].rsplit('/', 1)[1] for w in memory.writes]
        assert kinds.index('claim.json') < kinds.index('candidate.json') < kinds.index('terminal.json')
        def expired(url, token, **kw): raise FileNotFoundError()
        adapter = RunpodAdapter(p, _http=expired, _store=store)
        observed = adapter.observe(a, None)
        assert observed['handle'] is None  # A worker-supplied ID never grants cancel authority.
        assert observed['observation']['execution'] == 'RECEIVED'
        assert observed['observation']['resources'] == 'CLEANUP_UNKNOWN'
        raw = base64.b64decode(adapter.fetch(a, None)['candidate_b64'])
        candidate = parse(RemoteResultEnvelope, raw)
        from mathkernel_compute.verification import verify_candidate
        assert verify_candidate(a, candidate, 'verify').outcome == 'PASSED'
        with pytest.raises(ValueError, match='CONFLICT'):
            runpod_handler(dict(job, id='foreign_job'), config, _store=store)
    finally:
        c.close()


def test_runpod_policy_units_drift_check_and_narrow_cancel(tmp_path):
    c, p, a = staged(tmp_path)
    try:
        memory = MemoryS3(); store = S3Store(p.storage, _client=memory); calls = []
        def http(url, token, **kw):
            calls.append((url, kw))
            if '/serverless/' in url: return endpoint(p)
            return {'id': 'job-1', 'status': 'IN_QUEUE'}
        adapter = RunpodAdapter(p, _http=http, _store=store)
        assert adapter.submit(a)['handle'] == 'job-1'
        data = calls[-1][1]['data']
        assert data['policy'] == {'executionTimeout': 180000, 'ttl': 600000}
        assert set(data['input']) == {'ref', 'request_digest'}
        assert 'SECRET' not in canonical(adapter.probe()).decode()
        adapter.cancel(a, 'job-1')
        assert calls[-1][0].endswith('/cancel/job-1')
        assert not any('/purge' in u or 'DELETE' in k for u, k in calls)
        with pytest.raises(ValueError): adapter.cancel(a, 'foreign/../../endpoint')
        drift = RunpodAdapter(p.model_copy(update={'deployment_digest': 'c'*64}), _http=http, _store=store)
        before = len(memory.writes)
        with pytest.raises(ValueError, match='ENDPOINT_CHANGED'): drift.submit(a)
        assert len(memory.writes) == before
    finally:
        c.close()


def test_object_store_rejects_oversized_stream_and_conflicting_immutable_bytes(tmp_path):
    c, p, a = staged(tmp_path)
    try:
        memory = MemoryS3(); store = S3Store(p.storage, _client=memory)
        store.put(a, 'candidate', b'12345')
        with pytest.raises(ValueError, match='SIZE_LIMIT'): store.get(a, 'candidate', 4)
        assert memory.last_body.closed
        with pytest.raises(ValueError, match='CONFLICT'): store.put(a, 'candidate', b'54321')
        with pytest.raises(ValueError, match='KIND'): store.get(a, '../foreign')
        with pytest.raises(RemoteUnavailable): NoRedirect().redirect_request(None, None, None, None, None, None)
    finally:
        c.close()


def test_runpod_claim_ambiguity_or_foreign_workspace_never_runs_worker(tmp_path):
    c, p, a = staged(tmp_path)
    try:
        memory = MemoryS3(); store = S3Store(p.storage, _client=memory)
        store.put(a, 'request', canonical(a))
        config = RunpodWorkerConfig(target_id=p.target_id, account_scope=p.account_scope, endpoint_id=p.endpoint_id,
            runtime=p.runtime, resources=p.resources, source_image=p.source_image, storage=p.storage,
            allowed_workspaces=(a.workspace_id,))
        job = {'id': 'provider_job', 'input': bound_record(a, request_digest=digest(a))}
        with patch('mathkernel_compute.managed_worker.execute_attempt') as execution:
            with pytest.raises(PermissionError):
                runpod_handler(job, config.model_copy(update={'allowed_workspaces': ('foreign',)}), _store=store)
            with patch.object(store, 'create', side_effect=TimeoutError()):
                with pytest.raises(TimeoutError): runpod_handler(job, config, _store=store)
            execution.assert_not_called()
    finally:
        c.close()


def test_batch_reserves_all_shards_atomically_and_rolls_back_if_over_budget(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='0.50') as c:
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='a', request=request()), BatchShard(shard_id='b', request=request('21')))))
        g = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        with pytest.raises(PermissionError, match='BUDGET'):
            c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch')
        assert not cloud.calls and not c.journal.all('jobs') and not c.journal.all('budget_reservations')
        assert not c.journal.all('budget_grants')
        assert c.journal.get('batch_grants', g.grant_id)['batch_id'] is None


def test_batch_retains_manifest_and_queues_with_no_duplicate_dispatch(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='1.00') as c:
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='a', request=request()), BatchShard(shard_id='b', request=request('21')))))
        g = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        handle = c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch')
        assert c.budget_status('pilot')['reserved_or_unknown'] == '1.00'
        assert cloud.calls.count('submit') == 1
        assert c.status(handle.shards[1].job.job_id).execution == 'PREPARED'
        assert c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch') == handle
        cloud.execution = 'RECEIVED'; cloud.resources = 'RELEASE_CONFIRMED'
        c.reconcile_batch(handle.batch_id)
        assert cloud.calls.count('submit') == 2
        for shard in handle.shards:
            assert c.verify(shard.job.job_id).verification == 'PASSED'
        assert c.batch_status(handle.batch_id)['admitted_shards'] == 2
        assert c.reconcile_batch(handle.batch_id)['admitted_shards'] == 2
        assert cloud.calls.count('submit') == 2
        assert c.budget_status('pilot')['available'] == '0.00'


def test_batch_cancel_pending_shard_never_exports_or_charges_it(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='1.00') as c:
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='a', request=request()), BatchShard(shard_id='b', request=request()))))
        g = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        handle = c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch')
        c.cancel_batch(handle.batch_id)
        assert cloud.calls.count('submit') == 1 and cloud.calls.count('cancel') == 1
        pending = c.status(handle.shards[1].job.job_id)
        assert pending.execution == 'CANCELLED' and pending.cost == 'NO_METERED_ALLOCATION'
        assert c.budget_status('pilot')['available'] == '0.50'


def test_duplicate_logical_shards_rejected():
    with pytest.raises(ValueError, match='DUPLICATE_LOGICAL_SHARD'):
        BatchRequest(shards=(BatchShard(shard_id='same', request=request()), BatchShard(shard_id='same', request=request())))


def test_account_budget_identity_cannot_hide_retained_spend(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        launch(c)
    replacement = budget().model_copy(update={'budget_id': 'fresh'})
    with pytest.raises(ValueError, match='REBINDING'):
        ComputeClient(state_dir=tmp_path, budget_limits=(replacement,), reconcile_on_open=False)
    with client(tmp_path, p, cloud, reconcile_on_open=False) as c:
        assert c.budget_status('pilot')['reserved_or_unknown'] == '0.50'


def test_expired_pending_shard_voids_reservation_despite_active_sibling(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud, limit='1.00') as c:
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='a', request=request()), BatchShard(shard_id='b', request=request()))))
        g = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        handle = c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch')
        with patch('mathkernel_compute.client.now_ms', return_value=plan.expires_ms+1):
            result = c.reconcile(handle.shards[1].job.job_id)
        assert result.execution == 'FAILED' and result.resources == 'NOT_OWNED'
        assert result.cost == 'NO_METERED_ALLOCATION'
        assert cloud.calls.count('submit') == 1


def test_journal_v1_upgrade_retains_existing_local_identity(tmp_path):
    from mathkernel_compute.journal import ComputeJournal
    with ComputeClient(state_dir=tmp_path, reconcile_on_open=False) as c:
        plan = c.plan(request(target='local-cpu'))
        original = canonical(c.journal.get('plans', plan.plan_id))
        c.journal.db.execute('PRAGMA user_version=1')
    with ComputeClient(state_dir=tmp_path, reconcile_on_open=False) as c:
        assert c.journal.db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert canonical(c.journal.get('plans', plan.plan_id)) == original
        assert c.journal.all('provider_handles') == []


def test_confirmed_cleanup_survives_expired_provider_status(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c)
        cloud.execution = 'RECEIVED'; cloud.resources = 'RELEASE_CONFIRMED'
        c.reconcile(j.job_id)
        cloud.execution = 'SUBMISSION_UNKNOWN'; cloud.resources = 'CLEANUP_UNKNOWN'
        assert c.reconcile(j.job_id).resources == 'RELEASE_CONFIRMED'
        assert c.reconcile(j.job_id).execution == 'RECEIVED'


def test_worker_receipt_cannot_redirect_cancel_to_another_provider_job(tmp_path):
    c, p, a = staged(tmp_path)
    try:
        memory = MemoryS3(); store = S3Store(p.storage, _client=memory)
        store.put(a, 'receipt', canonical(bound_record(a, provider_job_id='unrelated-job')))
        calls = []
        adapter = RunpodAdapter(p, _store=store, _http=lambda *args, **kwargs: calls.append(args))
        assert adapter.cancel(a, None)['handle'] is None
        assert not calls
        with pytest.raises(ValueError, match='BINDING_CONFLICT'): adapter.cancel(a, 'our-job')
        assert not calls
    finally:
        c.close()


def test_batch_shard_swap_is_rejected_before_result_use(tmp_path):
    p = profile(); cloud = Cloud()
    with client(tmp_path, p, cloud) as c:
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='a', request=request()), BatchShard(shard_id='b', request=request('21')))))
        g = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        handle = c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=g.grant_id, client_request_id='batch')
        record = c.journal.get('batches', handle.batch_id)
        record['handle']['shards'][1]['job'] = record['handle']['shards'][0]['job']
        with c.journal.transaction(): c.journal.put('batches', handle.batch_id, record)
        with pytest.raises(ValueError, match='DUPLICATE_SHARD_JOB'): c.batch_status(handle.batch_id)


def test_cli_paid_approval_refuses_nonterminal_and_starts_no_provider(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    p = profile(); cloud = Cloud()
    with client(tmp_path / 'state', p, cloud) as c:
        plan = c.plan(request())
    config = tmp_path/'profile.json'; config.write_bytes(canonical(p))
    limits = tmp_path/'budget.json'; limits.write_bytes(canonical(budget()))
    result = subprocess.run([sys.executable, '-m', 'mathkernel_compute.cli', '--state-dir', str(tmp_path/'state'),
        '--managed-profile', str(config), '--budget-limit', str(limits), 'approve-managed', plan.plan_id,
        '--acknowledge-exposure', '--allow-storage'], input=plan.digest+'\n', capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and 'Explicit terminal confirmation required' in result.stderr
    with client(tmp_path/'state', p, cloud, reconcile_on_open=False) as c:
        assert not c.journal.all('budget_grants') and not c.journal.all('jobs')


def test_definite_setup_failure_releases_invocation_ownership_not_unknown_storage_cost(tmp_path):
    from mathkernel_compute.managed import ManagedNotSubmitted
    p = profile(); cloud = Cloud()
    def fail(call):
        assert call.action == 'submit'
        raise ManagedNotSubmitted('Setup failed before invocation')
    with client(tmp_path, p, cloud) as c:
        c._managed_call = fail
        j, _, _ = launch(c)
        assert c.status(j.job_id).execution == 'FAILED'
        assert c.status(j.job_id).resources == 'NOT_OWNED'
        assert c.status(j.job_id).cost == 'EXPOSURE_UNKNOWN'
        assert c.reconcile(j.job_id).resources == 'NOT_OWNED'
        assert c.budget_status('pilot')['reserved_or_unknown'] == '0.50'


def test_s3_sdk_cannot_redirect_export_to_another_host_or_region():
    from mathkernel_compute.object_store import guard_s3_request
    storage = profile().storage
    guard_s3_request(storage, SimpleNamespace(url='https://test-bucket.s3.us-east-1.amazonaws.com/mathkernel/key'))
    for url in ('https://test-bucket.s3.eu-west-1.amazonaws.com/key', 'https://attacker.test/key',
                'http://test-bucket.s3.us-east-1.amazonaws.com/key',
                'https://test-bucket.s3.us-east-1.amazonaws.com.attacker.test/key'):
        with pytest.raises(ValueError, match='S3_REGION_OR_HOST_CHANGED'):
            guard_s3_request(storage, SimpleNamespace(url=url))
