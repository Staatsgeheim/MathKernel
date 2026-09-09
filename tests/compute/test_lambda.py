"""R7 fault contracts. Lambda is simulated; the SSH/worker journey is real locally."""
import copy
from contextlib import closing
import io
import os
import time
from unittest.mock import patch
import pytest
from mathkernel_compute import ComputeClient, ComputeRequest, CuboidParameters, LambdaProfile, Money, BudgetLimit, ManagedQuote
from mathkernel_compute.lambda_api import LambdaAPI, LambdaRejected, RemoteUnavailable, instance_snapshot, https_json, NoRedirect
from mathkernel_compute.watchdog import LambdaWatchdog
from mathkernel_compute.journal import ComputeJournal
from mathkernel_compute.protocol import canonical, decode, digest, write_frame, read_frame
from mathkernel_compute.registry import runtime_profile
from mathkernel_compute.executor import worker_environment


def profile(**changes):
    fields = dict(target_id='lambda-pilot', account_scope='lambda-account', budget_id='cloud',
        region_name='us-west-1', instance_type_name='gpu_1x_test', image_id='a'*32,
        ssh_key_name='operator-key', firewall_ruleset_id='b'*32, runtime=runtime_profile(),
        cleanup_policy='managed-exposure', quote=ManagedQuote(quote_id='reviewed', reservation=Money(currency='USD', amount='1.00'),
            valid_until_ms=time.time_ns()//1_000_000+900000, source='fixture only, not provider pricing',
            scope='one VM including startup, runtime, idle, transfer and residual exposure'))
    fields.update(changes)
    return LambdaProfile(**fields)


def client(root, cloud, p=None, **changes):
    p = p or profile()
    c = ComputeClient(state_dir=root, lambda_profiles=(p,), reconcile_on_open=False, _lambda_http=cloud,
        budget_limits=(BudgetLimit(budget_id='cloud', account_scope='lambda-account', limit=Money(currency='USD', amount='2.00')),), **changes)
    cloud.client = c
    return c


def approve(c, p):
    return c.vm._authorize(p.lease_id, plan_digest=p.digest, managed_exposure_acknowledged=True,
        broad_key_privileges_acknowledged=True, network_egress_acknowledged=True,
        possible_output_loss_at_deadline_acknowledged=True)


def launch(c):
    p = c.vm.plan('lambda-pilot'); g = approve(c, p)
    return c.vm.launch(p.lease_id, authorization_ref=g.grant_id, client_request_id='once'), p, g


class Cloud:
    def __init__(self):
        self.rows = {}; self.calls = []; self.lost_ack = False; self.reject = False
        self.lost_termination = False; self.offline = False; self.client = None

    def __call__(self, path, data=None):
        self.calls.append((path, copy.deepcopy(data)))
        if self.offline:
            raise RemoteUnavailable('network fixture')
        if path == 'instance-operations/launch':
            lease_id = data['name']
            assert self.client.journal.get('vm_leases', lease_id)['launch_state'] == 'DISPATCHING'
            assert self.client.journal.get('budget_reservations', lease_id)['amount'] == '1.00'
            assert self.client.journal.all('vm_grants')[-1]['consumed'] is True
            if self.reject:
                raise LambdaRejected('fixture rejection')
            identity = 'c'*32
            self.rows[identity] = dict(id=identity, name=data['name'], status='active', ip='127.0.0.1',
                image=data['image'], region={'name': data['region_name']}, instance_type={'name': data['instance_type_name']},
                ssh_key_names=data['ssh_key_names'], file_system_names=[], tags=data['tags'],
                firewall_rulesets=data['firewall_rulesets'], jupyter_token='do-not-retain',
                jupyter_url='https://secret.test/?token=do-not-retain')
            if self.lost_ack:
                raise RemoteUnavailable('launch accepted, acknowledgement lost')
            return {'data': {'instance_ids': [identity]}}
        if path == 'instances':
            return {'data': list(self.rows.values())}
        if path.startswith('instances/'):
            if path[10:] not in self.rows:
                raise FileNotFoundError('missing')
            return {'data': self.rows[path[10:]]}
        if path == 'instance-operations/terminate':
            assert len(data['instance_ids']) == 1
            row = self.rows[data['instance_ids'][0]]
            if self.client:
                saved = self.client.journal.get('vm_leases', row['name'])
                assert saved['resources'] == 'RELEASE_REQUESTED'
            row['status'] = 'terminating'
            if self.lost_termination:
                raise RemoteUnavailable('termination ack lost')
            return {'data': {'terminated_instances': [row]}}
        raise AssertionError('Unsupported endpoint ' + path)


def test_offline_plan_and_strict_policy_refuse_vm_without_grant(tmp_path):
    cloud = Cloud()
    with client(tmp_path, cloud, profile(cleanup_policy='strict')) as c:
        p = c.vm.plan('lambda-pilot')
        assert not p.provisioning_allowed and not p.native_expiry_supported
        with pytest.raises(PermissionError, match='STRICT_CLEANUP'):
            approve(c, p)
        assert not cloud.calls and not c.journal.all('vm_grants')


def test_grants_are_separate_and_scope_checked(tmp_path):
    cloud = Cloud()
    with client(tmp_path, cloud) as c:
        p = c.vm.plan('lambda-pilot')
        local = c.plan(ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20')))
        grant = c._authorize_local(local.plan_id)
        with pytest.raises(KeyError):
            c.vm.launch(p.lease_id, authorization_ref=grant.grant_id, client_request_id='bad')
        with pytest.raises(ValueError):
            c.vm._authorize(p.lease_id, plan_digest=p.digest, managed_exposure_acknowledged=True,
                broad_key_privileges_acknowledged=False, network_egress_acknowledged=True,
                possible_output_loss_at_deadline_acknowledged=True)
        assert not cloud.calls and not c.journal.all('vm_leases')


def test_lost_launch_ack_recovers_after_restart_without_second_launch(tmp_path):
    cloud = Cloud(); cloud.lost_ack = True; p = profile()
    with client(tmp_path, cloud, p) as c:
        row, plan, grant = launch(c)
        assert row['resources'] == 'CLEANUP_UNKNOWN' and row['instance_id'] is None
        assert c.budget_status('cloud')['reserved_or_unknown'] == '1.00'
    with client(tmp_path, cloud, p) as c:
        row = c.vm.observe(plan.lease_id)
        assert row['resources'] == 'ACTIVE' and row['instance_id'] == 'c'*32
        c.vm.launch(plan.lease_id, authorization_ref=grant.grant_id, client_request_id='once')
        assert sum(path.endswith('/launch') for path, _ in cloud.calls) == 1
        assert 'do-not-retain' not in canonical(c.journal.all('vm_events')).decode()


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'image', 'owner', 'filesystem', 'firewall'])
def test_ambiguous_ownership_never_terminates_foreign_resource(tmp_path, fault):
    cloud = Cloud(); cloud.lost_ack = True
    with client(tmp_path, cloud) as c:
        row, p, _ = launch(c)
        original = cloud.rows['c'*32]
        if fault == 'missing': cloud.rows.clear()
        if fault == 'duplicate': cloud.rows['d'*32] = {**original, 'id': 'd'*32}
        if fault == 'image': original['image'] = {'id': 'd'*32}
        if fault == 'owner': original['tags'][0]['value'] = 'foreign'
        if fault == 'filesystem': original['file_system_names'] = ['shared-data']
        if fault == 'firewall': original['firewall_rulesets'] = []
        assert c.vm.reconcile(p.lease_id, cancel=True)['resources'] == 'CLEANUP_UNKNOWN'
        assert not any(path.endswith('/terminate') for path, _ in cloud.calls)
        assert c.budget_status('cloud')['reserved_or_unknown'] == '1.00'


def test_guest_exit_and_termination_ack_do_not_stop_billing(tmp_path):
    cloud = Cloud()
    with client(tmp_path, cloud) as c:
        _, p, _ = launch(c)
        cloud.rows['c'*32]['status'] = 'unhealthy'  # Guest poweroff is not provider termination.
        assert c.vm.observe(p.lease_id)['resources'] == 'CLEANUP_UNKNOWN'
        cloud.rows['c'*32]['status'] = 'active'
        assert c.vm.reconcile(p.lease_id, cancel=True)['resources'] == 'RELEASE_REQUESTED'
        assert c.vm.observe(p.lease_id)['resources'] == 'RELEASE_REQUESTED'
        with pytest.raises(PermissionError, match='EXPOSURE'):
            c.vm._reconcile_cost(p.lease_id, Money(currency='USD', amount='0.20'), 'bill', retained_storage_accounted=True)
        cloud.rows['c'*32]['status'] = 'terminated'
        assert c.vm.observe(p.lease_id)['resources'] == 'RELEASE_CONFIRMED'
        cloud.offline = True
        assert c.vm.observe(p.lease_id)['cost'] == 'EXPOSURE_UNKNOWN'
        c.vm._reconcile_cost(p.lease_id, Money(currency='USD', amount='0.20'), 'reviewed bill', retained_storage_accounted=True)
        assert c.vm.status(p.lease_id)['cost'] == 'USER_RECONCILED'
        assert c.budget_status('cloud')['available'] == '1.80'


def test_termination_timeout_retains_exact_lease_then_recovers(tmp_path):
    cloud = Cloud(); cloud.lost_termination = True; p = profile()
    with client(tmp_path, cloud, p) as c:
        _, plan, _ = launch(c)
        assert c.vm.reconcile(plan.lease_id, cancel=True)['resources'] == 'CLEANUP_UNKNOWN'
        assert c.vm.status(plan.lease_id)['instance_id'] == 'c'*32
    with client(tmp_path, cloud, p) as c:
        assert c.vm.reconcile(plan.lease_id)['resources'] == 'RELEASE_REQUESTED'
        assert sum(path.endswith('/terminate') for path, _ in cloud.calls) == 1
        cloud.rows['c'*32]['status'] = 'terminated'
        assert c.vm.observe(plan.lease_id)['resources'] == 'RELEASE_CONFIRMED'


def test_definite_rejection_voids_but_expiry_never_reauthorizes(tmp_path):
    cloud = Cloud(); cloud.reject = True
    with client(tmp_path, cloud) as c:
        row, p, g = launch(c)
        assert row['resources'] == 'NOT_OWNED' and c.budget_status('cloud')['reserved_or_unknown'] == '0.00'
        p2 = c.vm.plan('lambda-pilot'); g2 = approve(c, p2)
        with patch('mathkernel_compute.provisioning.now_ms', return_value=p2.launch_before_ms):
            with pytest.raises(PermissionError, match='EXPIRED'):
                c.vm.launch(p2.lease_id, authorization_ref=g2.grant_id, client_request_id='expired')
        assert sum(path.endswith('/launch') for path, _ in cloud.calls) == 1


def test_watchdog_prearmed_on_independent_journal_recovers_lost_ack(tmp_path):
    cloud = Cloud(); cloud.lost_ack = True; p = profile()
    watchdog_journal = ComputeJournal(tmp_path/'watchdog')
    try:
        watchdog = LambdaWatchdog(watchdog_journal, {p.target_id: p}, _http=cloud)
        with client(tmp_path/'controller', cloud, p) as c:
            plan = c.vm.plan('lambda-pilot'); ticket = c.vm.watchdog_ticket(plan.lease_id)
            with pytest.raises(PermissionError):
                watchdog._arm(ticket, ticket_digest=ticket.digest, independent_host_acknowledged=False)
            watchdog._arm(ticket, ticket_digest=ticket.digest, independent_host_acknowledged=True)
            assert watchdog.tick()[0]['termination_intents'] == 0
            g = approve(c, plan)
            c.vm.launch(plan.lease_id, authorization_ref=g.grant_id, client_request_id='once')
        cloud.client = None  # The main controller is gone; watchdog owns separate durable state.
        with patch('mathkernel_compute.watchdog.now_ms', return_value=plan.cleanup_after_ms + 1):
            row = watchdog.tick()[0]
            assert row['instance_id'] == 'c'*32 and row['resources'] == 'RELEASE_REQUESTED'
            assert row['termination_intents'] == 1
            assert watchdog_journal.get('watchdog_leases', plan.lease_id)['termination_intents'] == 1
            cloud.rows['c'*32]['status'] = 'terminated'
            assert watchdog.tick()[0]['resources'] == 'RELEASE_CONFIRMED'
        assert not watchdog_journal.all('vm_grants') and not watchdog_journal.all('vm_leases')
    finally:
        watchdog_journal.close()


def test_watchdog_absent_instance_and_changed_ticket_keep_unknown(tmp_path):
    cloud = Cloud(); p = profile()
    with client(tmp_path, cloud, p) as c:
        plan = c.vm.plan('lambda-pilot'); ticket = c.vm.watchdog_ticket(plan.lease_id)
        watchdog = LambdaWatchdog(c.journal, c.vm.profiles, _http=cloud)
        watchdog._arm(ticket, ticket_digest=ticket.digest, independent_host_acknowledged=True)
        altered = ticket.model_copy(update={'plan': plan.model_copy(update={'workspace_id': 'other'})})
        with pytest.raises(PermissionError, match='IMMUTABLE'):
            watchdog._arm(altered, ticket_digest=altered.digest, independent_host_acknowledged=True)
        with patch('mathkernel_compute.watchdog.now_ms', return_value=plan.cleanup_after_ms + 1):
            assert watchdog.tick()[0]['resources'] == 'CLEANUP_UNKNOWN'
        assert not any(path.endswith('/launch') or path.endswith('/terminate') for path, _ in cloud.calls)


def test_controller_key_only_enters_control_bridge_not_worker_or_plan(tmp_path):
    p = profile()
    with patch.dict(os.environ, {p.credential_env: 'fake-secret-sentinel'}):
        assert 'fake-secret-sentinel' not in canonical(worker_environment()).decode()
        def run(argv, data, **kw):
            assert argv[-1] == 'mathkernel_compute.lambda_api'
            assert kw['env']['MK_LAMBDA_CONTROL_KEY'] == 'fake-secret-sentinel'
            assert 'fake-secret-sentinel' not in data.decode()
            output = io.BytesIO(); write_frame(output, {'code': 'ok', 'value': {'instances': []}})
            return output.getvalue()
        with patch('mathkernel_compute.lambda_api.run_bounded', side_effect=run):
            assert LambdaAPI(p).call('list') == {'instances': []}
    assert 'fake-secret-sentinel' not in canonical(p).decode()


def test_control_bridge_rechecks_expired_launch_before_network():
    p = profile()
    with patch.dict(os.environ, {p.credential_env: 'fake-never-transmitted'}):
        with pytest.raises(LambdaRejected):
            LambdaAPI(p).call('launch', {}, launch_before_ms=1)


def test_vm_reservation_shares_aggregate_budget_with_managed_provider(tmp_path):
    from test_managed import profile as managed_profile, Cloud as ManagedCloud, approve as managed_approve
    from mathkernel_compute.batch import BatchRequest, BatchShard
    from mathkernel_compute import LocalPolicy
    cloud = Cloud(); mcloud = ManagedCloud(); p = profile()
    managed = managed_profile(account_scope='lambda-account', budget_id='cloud')
    with client(tmp_path, cloud, p, managed_targets=(managed,), _managed_call=mcloud,
                policy=LocalPolicy(max_concurrent_jobs=2)) as c:
        mcloud.client = c
        launch(c)
        req = ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20'), target='managed')
        plan = c.plan_batch(BatchRequest(shards=tuple(BatchShard(shard_id='s'+str(i), request=req) for i in range(3))))
        grant = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        with pytest.raises(PermissionError, match='BUDGET'):
            c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=grant.grant_id, client_request_id='too-many')
        assert not c.list() and not mcloud.calls
        assert c.budget_status('cloud')['reserved_or_unknown'] == '1.00'


def test_paid_batch_waits_for_vm_capacity_before_dispatch(tmp_path):
    from test_managed import profile as managed_profile, Cloud as ManagedCloud
    from mathkernel_compute.batch import BatchRequest, BatchShard
    cloud = Cloud(); mcloud = ManagedCloud(); p = profile()
    managed = managed_profile(account_scope='lambda-account', budget_id='cloud')
    with client(tmp_path, cloud, p, managed_targets=(managed,), _managed_call=mcloud) as c:
        mcloud.client = c
        _, vm_plan, _ = launch(c)
        req = ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20'), target='managed')
        plan = c.plan_batch(BatchRequest(shards=(BatchShard(shard_id='s1', request=req),)))
        grant = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=True,
            managed_exposure_acknowledged=True, persistent_storage_allowed=True)
        batch = c.submit_batch(plan_id=plan.batch_plan_id, authorization_ref=grant.grant_id, client_request_id='queued')
        assert not mcloud.calls
        cloud.rows['c'*32]['status'] = 'terminated'; c.vm.observe(vm_plan.lease_id)
        c.reconcile_batch(batch.batch_id)
        assert mcloud.calls.count('submit') == 1


def test_lambda_api_bounds_redirects_and_secret_projection():
    import urllib.error
    from email.message import Message
    with pytest.raises(RemoteUnavailable, match='REDIRECT'):
        NoRedirect().redirect_request(None, None, None, None, None, None)
    class Response:
        headers = {'Content-Length': '9999999999'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): raise AssertionError('must reject oversized header first')
    with patch('urllib.request.OpenerDirector.open', return_value=Response()):
        with pytest.raises(RemoteUnavailable, match='LIMIT'):
            https_json('instances', 'fake')
    error = urllib.error.HTTPError('https://cloud.lambda.ai', 503, 'unavailable', Message(), io.BytesIO(b'secret'))
    with patch('urllib.request.OpenerDirector.open', side_effect=error):
        with pytest.raises(RemoteUnavailable): https_json('instance-operations/launch', 'fake', {})
    value = instance_snapshot({'id': 'a'*32, 'status': 'terminated', 'jupyter_token': 'secret', 'jupyter_url': 'secret'})
    assert 'secret' not in canonical(value).decode()


def test_real_ssh_job_retained_and_locally_verified_before_vm_termination(tmp_path):
    from test_remote import LoopbackSSH
    from mathkernel_compute.remote import WorkerConfig
    cloud = Cloud(); p = profile()
    with client(tmp_path/'controller', cloud, p) as c:
        root = tmp_path/'remote'; root.mkdir()
        (root/'spool').mkdir(mode=0o700)
        config = WorkerConfig(target_id='lambda-pilot', spool_root=str(root/'spool'), allowed_workspaces=(c.workspace_id,))
        with closing(LoopbackSSH(root, config)) as server:
            c.remote_targets[p.target_id] = server.profile
            _, plan, _ = launch(c)
            c.vm.attach(plan.lease_id, p.target_id)
            req = ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20'), target=p.target_id)
            math_plan = c.plan(req)
            grant = c._authorize_remote(math_plan.plan_id, plan_digest=math_plan.digest,
                bundle_digest=math_plan.spec.bundle_digest, target_profile_digest=server.profile.profile_digest)
            job = c.submit(plan_id=math_plan.plan_id, authorization_ref=grant.grant_id, client_request_id='math')
            # A finished job with temporarily missing output must keep its VM.
            with patch('mathkernel_compute.remote.SSHExecutor.fetch', side_effect=FileNotFoundError('retention delay')):
                assert c.wait(job.job_id, timeout_s=30).execution == 'RECEIVED'
                waiting = c.vm.reconcile(plan.lease_id)
                assert not waiting['release_requested'] and waiting['retained_candidates'] == []
                assert not any(path.endswith('/terminate') for path, _ in cloud.calls)
            deadline = time.monotonic()+30
            while time.monotonic()<deadline:
                lease = c.vm.reconcile(plan.lease_id)
                if lease['release_requested']: break
                time.sleep(.05)
            assert lease['resources'] == 'RELEASE_REQUESTED'
            assert lease['retained_candidates'] == [c.status(job.job_id).candidate_ref]
            assert c.store.get(lease['retained_candidates'][0])
            # Provider termination is independent of mathematical admission.
            cloud.rows['c'*32]['status'] = 'terminated'; c.vm.observe(plan.lease_id)
            assert c.verify(job.job_id).verification == 'PASSED'
            assert c.accepted_result(job.job_id).trust.value == 'exact'
            assert c.vm.status(plan.lease_id)['cost'] == 'EXPOSURE_UNKNOWN'
            with pytest.raises(PermissionError, match='LEASE'):
                c._authorize_remote(math_plan.plan_id, plan_digest=math_plan.digest,
                    bundle_digest=math_plan.spec.bundle_digest, target_profile_digest=server.profile.profile_digest)


def test_old_schema_migrates_without_rewriting_payloads(tmp_path):
    with ComputeClient(state_dir=tmp_path) as c:
        c.journal.put('identity', 'preserved', {'payload': 'bytes remain stable'})
        before = c.journal.db.execute("SELECT value,sha256 FROM identity WHERE ref='preserved'").fetchone()
        c.journal.db.execute('PRAGMA user_version=2')
    with ComputeClient(state_dir=tmp_path) as c:
        assert c.journal.db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert c.journal.db.execute("SELECT value,sha256 FROM identity WHERE ref='preserved'").fetchone() == before
