"""Experimental job-owned Lambda leases, sharing the coordinator's budget ledger.

Provisioning and export are distinct approvals. A newly created VM is deliberately
not trusted for SSH until the operator supplies independently authenticated host
keys and a pinned, installed gateway. No bootstrap command comes from a request.
"""
from __future__ import annotations
from functools import wraps
import ipaddress
import secrets
import time
from . import budgets
from .lambda_api import LambdaAPI, LambdaRejected
from .lambda_models import LambdaProfile, VMPlan, VMGrant, WatchdogTicket
from .protocol import canonical, digest, parse
from .remote import SSHExecutor, RemoteUnavailable


def now_ms():
    return time.time_ns() // 1_000_000


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.client._coordinator_lock:
            return method(self, *args, **kwargs)
    return call


def check_instance(plan, row, expected_id=None):
    """Authenticated control-plane identity, never names/IPs/worker claims alone."""
    p = plan.profile
    tags = row.get('tags', [])
    if (not isinstance(tags, list) or any(not isinstance(x, dict) or set(x) != {'key', 'value'} for x in tags)
            or len({x['key'] for x in tags}) != len(tags)):
        raise ValueError('LAMBDA_OWNER_TAGS_INVALID')
    tag_map = {x['key']: x['value'] for x in tags}
    if (any(tag_map.get(k) != v for k, v in plan.tags.items())
            or row.get('name') != plan.lease_id
            or (expected_id and row['id'] != expected_id)
            or row.get('region', {}).get('name') != p.region_name
            or row.get('instance_type', {}).get('name') != p.instance_type_name
            or row.get('ssh_key_names') != [p.ssh_key_name]
            or row.get('file_system_names') != [] or row.get('file_system_mounts', []) != []):
        raise ValueError('LAMBDA_OWNERSHIP_MISMATCH')
    # The API explicitly omits image metadata on some terminated instances.
    if row.get('status') != 'terminated' or 'image' in row:
        if row.get('image', {}).get('id') != p.image_id:
            raise ValueError('LAMBDA_IMAGE_MISMATCH')
    if row.get('status') != 'terminated' and row.get('firewall_rulesets') != [{'id': p.firewall_ruleset_id}]:
        raise ValueError('LAMBDA_FIREWALL_MISMATCH')
    return row


def find_instance(api, plan, identity=None):
    if identity:
        return check_instance(plan, api.call('get', {'instance_id': identity}), identity)
    rows = api.call('list')['instances']
    # A missing list entry is not proof that a timed-out launch was rejected.
    matches = [r for r in rows if any(t == {'key': 'mk-lease', 'value': plan.lease_id}
                                     for t in r.get('tags', []))]
    if len(matches) != 1:
        raise RemoteUnavailable('LAMBDA_OWNERSHIP_UNRESOLVED')
    return check_instance(plan, matches[0])


class LambdaProvisioner:
    def __init__(self, client, profiles=(), *, _http=None):
        self.client = client
        self.journal = client.journal
        parsed = [parse(LambdaProfile, canonical(p)) for p in profiles]
        self.profiles = {p.target_id: p for p in parsed}
        if len(self.profiles) != len(parsed) or set(self.profiles) & set(client.managed_targets):
            raise ValueError('DUPLICATE_TARGET_ALIAS')
        self.http = _http

    def _api(self, plan):
        # Cleanup uses the retained profile even when a new launch quote expires.
        current = self.profiles.get(plan.profile.target_id)
        if current is None or current.account_scope != plan.profile.account_scope:
            raise RemoteUnavailable('LAMBDA_ACCOUNT_NOT_CONFIGURED')
        return LambdaAPI(current, _http=self.http, journal=self.journal)

    def _plan(self, lease_id):
        return parse(VMPlan, canonical(self.journal.get('vm_plans', lease_id)))

    def _check_plan(self, plan):
        if (not self.client.policy.enabled or plan.workspace_id != self.client.workspace_id or now_ms() >= plan.launch_before_ms
                or self.profiles.get(plan.profile.target_id) != plan.profile
                or self.client.budget_limits.get(plan.budget.budget_id) != plan.budget):
            raise PermissionError('VM_PLAN_EXPIRED_OR_CHANGED')
        if plan.profile.cleanup_policy != 'managed-exposure':
            raise PermissionError('STRICT_CLEANUP_UNSUPPORTED: no qualified independent expiry guarantee')

    @locked
    def plan(self, target):
        p = self.profiles[target]
        created = now_ms()
        plan = VMPlan(lease_id='vm_' + secrets.token_hex(16), workspace_id=self.client.workspace_id,
            profile=p, budget=self.client.budget_limits[p.budget_id], created_ms=created,
            launch_before_ms=min(created + 300000, p.quote.valid_until_ms),
            cleanup_after_ms=min(created + 300000, p.quote.valid_until_ms) + p.lease_ms)
        with self.journal.transaction():
            if len(self.journal.all('vm_plans')) >= self.client.policy.max_retained_jobs:
                raise ValueError('JOURNAL_CAPACITY')
            self.journal.put('vm_plans', plan.lease_id, plan)
        return plan

    @locked
    def _authorize(self, lease_id, *, plan_digest, managed_exposure_acknowledged,
                   broad_key_privileges_acknowledged, network_egress_acknowledged,
                   possible_output_loss_at_deadline_acknowledged):
        plan = self._plan(lease_id)
        self._check_plan(plan)
        if plan_digest != plan.digest:
            raise PermissionError('VM_PLAN_DIGEST_MISMATCH')
        grant = VMGrant(grant_id='vmgrant_' + secrets.token_hex(16), lease_id=lease_id,
            plan_digest=plan.digest, workspace_id=plan.workspace_id, expires_ms=plan.launch_before_ms,
            provisioning_allowed=True, managed_exposure_acknowledged=managed_exposure_acknowledged,
            broad_key_privileges_acknowledged=broad_key_privileges_acknowledged,
            network_egress_acknowledged=network_egress_acknowledged,
            possible_output_loss_at_deadline_acknowledged=possible_output_loss_at_deadline_acknowledged)
        with self.journal.transaction():
            if len(self.journal.all('vm_grants')) >= self.client.policy.max_retained_jobs * 4:
                raise ValueError('JOURNAL_CAPACITY')
            self.journal.put('vm_grants', grant.grant_id, {'grant': grant.model_dump(mode='json'), 'consumed': False})
        return grant

    def status(self, lease_id):
        record = self.journal.get('vm_leases', lease_id)
        reservation = self.journal.get('budget_reservations', lease_id)
        return {**record, 'cost': 'USER_RECONCILED' if reservation['state'] == 'SETTLED' else
                'NO_METERED_ALLOCATION' if reservation['state'] == 'VOID' else 'EXPOSURE_UNKNOWN'}

    def _save(self, row, reason):
        row['revision'] += 1
        row['message'] = reason
        self.journal.put('vm_leases', row['lease_id'], row)
        self.journal.put('vm_events', f"{row['lease_id']}_{row['revision']}",
            {'lease_id': row['lease_id'], 'sequence': row['revision'], 'time_ms': now_ms(),
             'resources': row['resources'], 'reason': reason, 'instance_id': row['instance_id']})

    @locked
    def launch(self, lease_id, *, authorization_ref, client_request_id):
        from .models import JobHandle
        JobHandle(job_id='check', attempt_id='check', client_request_id=client_request_id)
        plan = self._plan(lease_id)
        for old in self.journal.all('vm_leases'):
            if old['lease_id'] == lease_id or old['client_request_id'] == client_request_id:
                if (old['lease_id'], old['authorization_ref'], old['client_request_id']) != (lease_id, authorization_ref, client_request_id):
                    raise PermissionError('VM_REQUEST_ID_CONFLICT')
                return self.status(lease_id)  # Never reissue launch after an ambiguous acknowledgement.
        self._check_plan(plan)
        api = self._api(plan)
        api.ready()  # Missing local credentials have no possible allocation effect.
        with self.journal.transaction():
            auth = self.journal.get('vm_grants', authorization_ref)
            grant = parse(VMGrant, canonical(auth['grant']))
            if (auth['consumed'] or grant.plan_digest != plan.digest or grant.lease_id != lease_id
                    or grant.workspace_id != plan.workspace_id or now_ms() >= grant.expires_ms):
                raise PermissionError('VM_AUTHORIZATION_INVALID')
            if any(r['resources'] not in {'NOT_OWNED', 'RELEASE_CONFIRMED'} for r in self.journal.all('vm_leases')):
                raise PermissionError('VM_CONCURRENCY_LIMIT')
            paid_jobs = sum(j['resources'] not in {'NOT_OWNED', 'RELEASE_CONFIRMED'}
                and self.client._attempt(j).spec.managed is not None for j in self.journal.all('jobs'))
            if paid_jobs >= self.client.policy.max_concurrent_jobs:
                raise PermissionError('PAID_CONCURRENCY_LIMIT')
            from types import SimpleNamespace
            budgets.reserve(self.journal, SimpleNamespace(budget=plan.budget, quote=plan.profile.quote))
            auth['consumed'] = True
            self.journal.put('vm_grants', authorization_ref, auth)
            self.journal.put('budget_reservations', lease_id, {'budget_id': plan.budget.budget_id,
                'account_scope': plan.budget.account_scope, 'amount': plan.profile.quote.reservation.amount,
                'currency': 'USD', 'state': 'EXPOSURE_UNKNOWN', 'kind': 'owned-vm'})
            row = {'lease_id': lease_id, 'workspace_id': plan.workspace_id, 'client_request_id': client_request_id,
                'authorization_ref': authorization_ref, 'plan_digest': plan.digest, 'instance_id': None,
                'resources': 'ALLOCATION_INTENT', 'launch_state': 'DISPATCHING', 'revision': 0,
                'release_requested': False, 'cleanup_after_ms': plan.cleanup_after_ms, 'ip': None,
                'output_loss_possible': False, 'retained_candidates': [], 'next_reconcile_ms': 0}
            self._save(row, 'launch_intent_committed')
        try:
            self._check_plan(plan)  # Check immediately before the external launch.
        except PermissionError:
            with self.journal.transaction():
                row.update(resources='NOT_OWNED', launch_state='REJECTED')
                reservation = self.journal.get('budget_reservations', lease_id)
                reservation['state'] = 'VOID'; self.journal.put('budget_reservations', lease_id, reservation)
                self._save(row, 'expired_before_launch')
            return self.status(lease_id)
        p = plan.profile
        try:
            reply = api.call('launch', {'region_name': p.region_name, 'instance_type_name': p.instance_type_name,
                'ssh_key_names': [p.ssh_key_name], 'file_system_names': [], 'name': lease_id,
                'image': {'id': p.image_id}, 'tags': [{'key': k, 'value': v} for k, v in plan.tags.items()],
                'firewall_rulesets': [{'id': p.firewall_ruleset_id}]}, launch_before_ms=grant.expires_ms)
            row.update(instance_id=reply['instance_id'], resources='ALLOCATING', launch_state='OBSERVED')
            reason = 'provider_launch_acknowledged'
        except LambdaRejected:
            row.update(resources='NOT_OWNED', launch_state='REJECTED')
            reason = 'provider_launch_rejected'
        except (OSError, ValueError, KeyError, TypeError):
            row.update(resources='CLEANUP_UNKNOWN', launch_state='SUBMISSION_UNKNOWN')
            reason = 'launch_acknowledgement_unknown_no_retry'
        with self.journal.transaction():
            if row['resources'] == 'NOT_OWNED':
                reservation = self.journal.get('budget_reservations', lease_id)
                reservation['state'] = 'VOID'; self.journal.put('budget_reservations', lease_id, reservation)
            self._save(row, reason)
        return self.status(lease_id)

    @locked
    def observe(self, lease_id):
        row = self.journal.get('vm_leases', lease_id)
        if row['resources'] in {'NOT_OWNED', 'RELEASE_CONFIRMED'}:
            return self.status(lease_id)
        if not self.http and now_ms() < row['next_reconcile_ms']:
            return self.status(lease_id)
        plan = self._plan(lease_id)
        try:
            instance = find_instance(self._api(plan), plan, row['instance_id'])
            row.update(instance_id=instance['id'], launch_state='OBSERVED', ip=instance.get('ip'),
                resources={'active': 'ACTIVE', 'booting': 'ALLOCATING', 'terminated': 'RELEASE_CONFIRMED',
                           'terminating': 'RELEASE_REQUESTED'}.get(instance['status'], 'CLEANUP_UNKNOWN'))
            reason = 'provider_' + instance['status']
            delay = 15000
        except (OSError, ValueError, KeyError, TypeError):
            row.update(resources='CLEANUP_UNKNOWN')
            reason = 'provider_ownership_or_transport_unresolved'
            delay = 30000 + secrets.randbelow(15000)
        row['next_reconcile_ms'] = now_ms() + delay
        with self.journal.transaction():
            self._save(row, reason)
        return self.status(lease_id)

    @locked
    def attach(self, lease_id, target):
        row = self.observe(lease_id)
        plan = self._plan(lease_id)
        profile = self.client.remote_targets[target]
        if (row['resources'] != 'ACTIVE' or row['release_requested'] or now_ms() >= plan.cleanup_after_ms
                or target != plan.profile.target_id or profile.adapter != 'ssh'
                or profile.runtime != plan.profile.runtime
                or ipaddress.ip_address(profile.host) != ipaddress.ip_address(row['ip'])):
            raise PermissionError('VM_SSH_ONBOARDING_MISMATCH')
        profile_digest = profile.profile_digest
        SSHExecutor(profile).probe()  # Strict known-hosts, config/runtime pinning. No trust-on-first-use.
        attachment = {'target': target, 'profile_digest': profile_digest, 'lease_id': lease_id}
        with self.journal.transaction():
            try:
                if self.journal.get('vm_attachments', profile_digest) != attachment:
                    raise PermissionError('VM_ATTACHMENT_IMMUTABLE')
            except KeyError:
                self.journal.put('vm_attachments', profile_digest, attachment)
        return attachment

    def attachment(self, spec):
        if spec.remote is None:
            return None
        try:
            return self.journal.get('vm_attachments', spec.remote.target_profile_digest)
        except KeyError:
            if spec.target in self.profiles or any(a['target'] == spec.target for a in self.journal.all('vm_attachments')):
                raise PermissionError('VM_SSH_ATTACHMENT_REQUIRED') from None
            return None

    def check_execution(self, spec):
        attachment = self.attachment(spec)
        if attachment:
            row = self.journal.get('vm_leases', attachment['lease_id'])
            if (row['resources'] != 'ACTIVE' or row['release_requested']
                    or now_ms() + spec.bundle.request.resources.execution_timeout_ms + 30000 >= row['cleanup_after_ms']):
                raise PermissionError('VM_LEASE_UNAVAILABLE_OR_EXPIRED')
        return attachment

    def bind_job(self, spec, job_id):
        attachment = self.check_execution(spec)
        if attachment:
            try:
                if self.journal.get('vm_jobs', attachment['lease_id'])['job_id'] != job_id:
                    raise PermissionError('VM_SINGLE_JOB_ONLY')
            except KeyError:
                self.journal.put('vm_jobs', attachment['lease_id'], {'job_id': job_id})

    @locked
    def reconcile(self, lease_id, *, cancel=False):
        row = self.journal.get('vm_leases', lease_id)
        if row['resources'] in {'NOT_OWNED', 'RELEASE_CONFIRMED'}:
            return self.status(lease_id)
        expired = now_ms() >= row['cleanup_after_ms']
        if cancel or expired:
            with self.journal.transaction():
                row.update(release_requested=True, output_loss_possible=True)
                self._save(row, 'cancel_intent' if cancel else 'lease_deadline_reached')
        self.observe(lease_id)
        row = self.journal.get('vm_leases', lease_id)
        try:
            job_id = self.journal.get('vm_jobs', lease_id)['job_id']
        except KeyError:
            job_id = None
        if job_id and not row['release_requested']:
            status = self.client.reconcile(job_id)
            if status.execution == 'RECEIVED':
                status = self.client.fetch(job_id)  # Atomic fsync quarantine before provider termination.
                if status.candidate_ref:
                    self.client.store.get(status.candidate_ref)  # Verify retained bytes, even rejected candidates.
                    with self.journal.transaction():
                        row.update(retained_candidates=[status.candidate_ref], release_requested=True)
                        self._save(row, 'candidate_durable_before_termination')
            elif status.execution in {'FAILED', 'TIMED_OUT', 'CANCELLED', 'OUT_OF_MEMORY', 'EXPIRED'}:
                with self.journal.transaction():
                    row.update(release_requested=True, output_loss_possible=True)
                    self._save(row, 'failed_job_cleanup')
        if row['release_requested'] and row['resources'] != 'RELEASE_CONFIRMED':
            self._terminate(row)
        return self.status(lease_id)

    def _terminate(self, row):
        plan = self._plan(row['lease_id'])
        try:
            api = self._api(plan)
            instance = find_instance(api, plan, row['instance_id'])
            row['instance_id'] = instance['id']
            if instance['status'] == 'terminated':
                with self.journal.transaction():
                    row['resources'] = 'RELEASE_CONFIRMED'; self._save(row, 'provider_terminated')
                return
            if instance['status'] == 'terminating':
                with self.journal.transaction():
                    row['resources'] = 'RELEASE_REQUESTED'; self._save(row, 'provider_terminating')
                return
            if not self.http and now_ms() < row.get('terminate_retry_after_ms', 0):
                return
            with self.journal.transaction():
                row.update(resources='RELEASE_REQUESTED', terminate_retry_after_ms=now_ms()+30000+secrets.randbelow(15000))
                self._save(row, 'termination_intent_committed')
            api.call('terminate', {'instance_id': row['instance_id']})
            reason = 'termination_acknowledged_reconciliation_required'
        except (OSError, ValueError, KeyError, TypeError):
            row['resources'] = 'CLEANUP_UNKNOWN'
            reason = 'termination_or_ownership_unresolved'
        with self.journal.transaction():
            self._save(row, reason)

    def watchdog_ticket(self, lease_id):
        return WatchdogTicket(plan=self._plan(lease_id))

    @locked
    def _reconcile_cost(self, lease_id, amount, source, *, retained_storage_accounted):
        row = self.journal.get('vm_leases', lease_id)
        with self.journal.transaction():
            return budgets.settle_resource(self.journal, lease_id, row['resources'], amount, source,
                                           retained_storage_accounted=retained_storage_accounted)


class OwnedSSHExecutor:
    def __init__(self, executor, provisioner, attachment):
        self.executor, self.provisioner, self.attachment = executor, provisioner, attachment

    def submit(self, attempt):
        try:
            self.provisioner.observe(self.attachment['lease_id'])
            self.provisioner.check_execution(attempt.spec)
        except PermissionError as exc:
            raise RemoteUnavailable('VM_LEASE_NOT_READY') from exc
        return self.executor.submit(attempt)

    def observe(self, attempt):
        return self.executor.observe(attempt)

    def cancel(self, attempt):
        return self.executor.cancel(attempt)

    def fetch(self, attempt):
        return self.executor.fetch(attempt)
