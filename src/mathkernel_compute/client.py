"""Durable compute client. Host policy, candidate data and mathematical admission are separate."""
from __future__ import annotations
import io
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import threading
from functools import wraps
from .executor import LocalExecutor, worker_environment
from .files import ArtifactStore
from .journal import ComputeJournal
from .models import (AttemptRecord, AuthorizationGrant, ComputePlan, ComputeRequest, ComputeResultReceipt,
    ComputeTarget, ExecutionSpec, InputBundle, JobHandle, LocalPolicy, ResourceLease, RemoteResultEnvelope,
    TargetCapabilities, VerificationReport, JobRecord, RemoteBinding, Money, BudgetLimit, PaidApproval, BatchBinding)
from .protocol import canonical, digest, parse, read_frame, write_frame
from .registry import runtime_profile
from .verification import admit, check_binding
from .remote import SSHProfile, SSHExecutor, RemoteUnavailable
from .managed import parse_profile, ManagedExecutor, ManagedNotSubmitted
from . import budgets
from .batch import BatchRequest, BatchPlan, PlannedShard, BatchGrant, BatchHandle, ShardHandle

TERMINAL = {'RECEIVED', 'FAILED', 'TIMED_OUT', 'CANCELLED', 'LOST', 'OUT_OF_MEMORY', 'PREEMPTED', 'NODE_FAILED', 'EXPIRED'}


def now_ms():
    return time.time_ns() // 1_000_000


def identifier(prefix):
    return prefix + '_' + secrets.token_hex(16)


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._coordinator_lock:
            return method(self, *args, **kwargs)
    return call


class ComputeClient:
    def __init__(self, *, state_dir, kernel=None, policy=None, remote_targets=(), managed_targets=(), budget_limits=(), reconcile_on_open=True, _executor=None, _managed_call=None):
        self._coordinator_lock = threading.RLock()
        self.kernel = kernel  # No remote method is added to the facade.
        self.policy = policy or LocalPolicy()
        profiles = [parse(SSHProfile, canonical(p)) for p in remote_targets]
        self.remote_targets = {p.target_id: p for p in profiles}
        if len(self.remote_targets) != len(profiles):
            raise ValueError('DUPLICATE_TARGET_ALIAS')
        managed = [parse_profile(canonical(p)) for p in managed_targets]
        self.managed_targets = {p.target_id: p for p in managed}
        limits = [parse(BudgetLimit, canonical(b)) for b in budget_limits]
        self.budget_limits = {b.budget_id: b for b in limits}
        self._managed_call = _managed_call
        if (len(self.managed_targets) != len(managed) or set(self.managed_targets) & set(self.remote_targets)
                or len(self.budget_limits) != len(limits)):
            raise ValueError('DUPLICATE_TARGET_OR_BUDGET')
        self.journal = ComputeJournal(state_dir)
        self.root = self.journal.root
        self.store = ArtifactStore(self.root / 'quarantine')
        self.executor = _executor or LocalExecutor(self.root / 'attempt-spool')
        self.closed = False
        try:
            with self.journal.transaction():
                try:
                    self.workspace_id = self.journal.get('identity', 'workspace')['workspace_id']
                except KeyError:
                    self.workspace_id = identifier('workspace')
                    self.journal.put('identity', 'workspace', {'workspace_id': self.workspace_id})
                # Stable account-to-budget association prevents a new budget ID
                # from hiding this coordinator's retained exposure or spend.
                for budget in self.budget_limits.values():
                    ref = 'budget_account_' + digest(budget.account_scope)
                    binding = {'account_scope': budget.account_scope, 'budget_id': budget.budget_id}
                    try:
                        if self.journal.get('identity', ref) != binding:
                            raise ValueError('BUDGET_ACCOUNT_REBINDING_DENIED')
                    except KeyError:
                        self.journal.put('identity', ref, binding)
                    bid = 'budget_id_' + digest(budget.budget_id)
                    try:
                        if self.journal.get('identity', bid) != binding:
                            raise ValueError('BUDGET_ID_REBINDING_DENIED')
                    except KeyError:
                        self.journal.put('identity', bid, binding)
            with self.journal.transaction():
                for job in self.journal.all('jobs'):
                    if job['verification'] == 'RUNNING':
                        job['verification'] = 'INCONCLUSIVE'
                        self._save(job, 'verifier_controller_restarted')
            if reconcile_on_open:
                self.reconcile()
        except BaseException:
            self.close()
            raise

    def targets(self):
        available = sys.platform == 'linux' and Path('/proc/self/stat').exists()
        flag = 'supported' if available else 'unsupported'
        local = ComputeTarget(available=available,
            capabilities=TargetCapabilities(process_deadline=flag, process_tree_cleanup=flag),
            reason='Linux native supervised worker' if available else 'Local execution requires Linux')
        remote = tuple(ComputeTarget(target_id=p.target_id, adapter=p.adapter, available=True,
            capabilities=TargetCapabilities(process_deadline='supported', process_tree_cleanup='supported',
                output_retention='remote-spool', source='operator-profile'),
            reason='Operator-configured target; live identity and availability require an explicit probe')
            for p in self.remote_targets.values())
        managed = tuple(ComputeTarget(target_id=p.target_id, adapter=p.adapter, available=sys.platform == 'linux',
            capabilities=TargetCapabilities(process_deadline='supported', process_tree_cleanup='supported',
                output_retention='durable-provider-store', source='operator-profile',
                network_isolation='provider-configured' if p.adapter == 'modal' else 'unsupported',
                hard_memory_limit='provider-configured' if p.adapter == 'modal' else 'unsupported'),
            reason='Experimental operator profile; paid approval required; live GPU qualification pending')
            for p in self.managed_targets.values())
        return (local, *remote, *managed)

    def target_probe(self, target='local-cpu'):
        if target == 'local-cpu':
            return self.targets()[0]
        if target in self.managed_targets:
            return self._managed_executor(self.managed_targets[target]).probe()
        try:
            return SSHExecutor(self.remote_targets[target]).probe()
        except KeyError:
            raise ValueError('TARGET_NOT_CONFIGURED') from None

    def _managed_executor(self, profile):
        return ManagedExecutor(profile, self.journal, **({'_call': self._managed_call} if self._managed_call else {}))

    def _executor_for(self, attempt):
        if attempt.spec.managed:
            profile = self.managed_targets.get(attempt.spec.target)
            if profile is None or profile.profile_digest != attempt.spec.managed.target_profile_digest:
                raise RemoteUnavailable('PINNED_TARGET_UNAVAILABLE_OR_CHANGED')
            return self._managed_executor(profile)
        if attempt.spec.remote is None:
            return self.executor
        profile = self.remote_targets.get(attempt.spec.target)
        try:
            if profile is None or profile.profile_digest != attempt.spec.remote.target_profile_digest:
                raise RemoteUnavailable('PINNED_TARGET_UNAVAILABLE_OR_CHANGED')
        except OSError as exc:
            raise RemoteUnavailable('PINNED_TARGET_UNAVAILABLE_OR_CHANGED') from exc
        return SSHExecutor(profile)

    @serialized
    def plan(self, request: ComputeRequest):
        plan = self._build_plan(request)
        with self.journal.transaction():
            self._store_plan(plan)
        return plan

    def _build_plan(self, request):
        request = parse(ComputeRequest, canonical(request))
        target = 'local-cpu' if request.target == 'auto' else request.target
        remote = target != 'local-cpu'
        managed_profile = self.managed_targets.get(target)
        if remote and target not in self.remote_targets and managed_profile is None:
            raise ValueError('TARGET_NOT_CONFIGURED')
        if not remote and not self.targets()[0].available:
            raise ValueError('TARGET_UNSUPPORTED')
        local_runtime = runtime_profile()
        profile = self.remote_targets.get(target)
        if request.resources.gpu_count != int(bool(managed_profile and managed_profile.resources.gpu)):
            raise ValueError('REQUEST_ACCELERATOR_TARGET_MISMATCH')
        managed_binding = None
        if managed_profile:
            budget = self.budget_limits.get(managed_profile.budget_id)
            if budget is None:
                raise PermissionError('MANAGED_BUDGET_NOT_CONFIGURED')
            if sys.platform != 'linux':
                raise ValueError('MANAGED_CONTROLLER_UNQUALIFIED')
            if now_ms() >= managed_profile.quote.valid_until_ms:
                raise PermissionError('MANAGED_QUOTE_EXPIRED')
            managed_binding = managed_profile.binding(local_runtime, budget)
        bundle = InputBundle(request=request, classification='explicit_export' if remote else 'local_only')
        spec = ExecutionSpec(bundle=bundle, bundle_digest=digest(bundle), target=target,
            runtime=managed_profile.runtime if managed_profile else profile.runtime if profile else local_runtime,
            policy_digest=digest(self.policy), managed=managed_binding,
            remote=RemoteBinding(target_profile_digest=profile.profile_digest, verifier_runtime=local_runtime,
                                 adapter=profile.adapter, destination=f'{profile.host}:{profile.port}',
                                 ssh_account=profile.account, allocation=profile.allocation) if profile else None)
        warnings = ('Native worker has no hard memory or network isolation.',
                    'One attempt; no automatic retry, fallback or provisioning.',
                    'Local verification consumes additional CPU; numeric verification is quadratic in input length.',
                    'Supervisor survives disconnect; unreachable ownership retains cleanup uncertainty.')
        if remote:
            warnings += ('Approval exports the exact self-contained bundle to this pinned target.',
                         'Remote spool retains input and output; removal is an operator retention action.',
                         'Queue/start authority expires with this plan; runtime has its own bounded deadline.')
        warnings += (('Institutional allocation usage/pricing is unknown; zero provider estimate is not a quota guarantee.'
                      if profile and profile.adapter == 'slurm' else
                      'Existing unmetered host only; electricity and hardware costs are not measured.'),)
        if managed_profile:
            warnings = (
                'Experimental provider adapter; live GPU, billing and disconnected recovery qualification is pending.',
                'Approval exports the exact bundle and permits one paid invocation using persistent user-owned storage.',
                'USD reservation is an operator-supplied exposure allowance, not a provider hard spend cap.',
                'Shared budget covers this coordinator only. External spending and other controllers are not tracked.',
                'Unknown submission, cleanup, billing or retained storage costs keep the reservation.',
                'No automatic retry, replacement, fallback, endpoint deployment or mutation.',
                'Local independent verification consumes additional CPU and never trusts provider success.',
                'Existing endpoint minimum/maximum workers and idle timeout can cause charges beyond a job.'
                    if managed_profile.adapter == 'runpod' else
                'CPU and memory requests and limits are equal; provider billing and volume retention remain separate.',
                'Strict scientific network isolation is unsupported on Runpod; only the registered worker broker uses artifact egress.'
                    if managed_profile.adapter == 'runpod' else 'Sandbox egress is blocked; Volume v2 sync persists output.')
        plan = ComputePlan(plan_id=identifier('plan'), workspace_id=self.workspace_id, created_ms=now_ms(),
            expires_ms=min(now_ms() + self.policy.plan_lifetime_ms, managed_profile.quote.valid_until_ms) if managed_profile else now_ms() + self.policy.plan_lifetime_ms, spec=spec, execution_digest=digest(spec),
            provider_cost=managed_profile.quote.reservation if managed_profile else None if remote else Money(),
            warnings=warnings, rejected_targets=('Unconfigured providers, VM provisioning, Apptainer and automatic remote selection are unavailable.',))
        return plan

    def _store_plan(self, plan):
        if len(self.journal.all('plans')) >= self.policy.max_retained_jobs * 4:
            raise ValueError('JOURNAL_CAPACITY')
        self.journal.put('plans', plan.plan_id, plan)
        self.journal.put('execution_specs', plan.execution_digest, plan.spec)

    def _plan(self, plan_id):
        plan = parse(ComputePlan, canonical(self.journal.get('plans', plan_id)))
        if (plan.workspace_id != self.workspace_id or digest(plan.spec) != plan.execution_digest
                or digest(plan.spec.bundle) != plan.spec.bundle_digest):
            raise ValueError('PLAN_BINDING_MISMATCH')
        return plan

    def _check_plan(self, plan):
        if not self.policy.enabled:
            raise PermissionError('POLICY_DENIED')
        if now_ms() >= plan.expires_ms:
            raise PermissionError('PLAN_EXPIRED')
        verifier_runtime = plan.spec.endpoint.verifier_runtime if plan.spec.endpoint else plan.spec.runtime
        if plan.spec.policy_digest != digest(self.policy) or verifier_runtime != runtime_profile():
            raise PermissionError('PLAN_CHANGED')
        if plan.spec.managed:
            profile = self.managed_targets.get(plan.spec.target)
            binding = plan.spec.managed
            if (profile is None or profile.profile_digest != binding.target_profile_digest
                    or self.budget_limits.get(binding.budget.budget_id) != binding.budget
                    or now_ms() >= binding.quote.valid_until_ms):
                raise PermissionError('MANAGED_PROFILE_OR_BUDGET_CHANGED')
        if plan.spec.remote:
            profile = self.remote_targets.get(plan.spec.target)
            if profile is None or profile.profile_digest != plan.spec.remote.target_profile_digest:
                raise PermissionError('TARGET_PROFILE_CHANGED')

    @serialized
    def _authorize_local(self, plan_id, *, subject='local-user'):
        """Trusted host entrypoint; deliberately absent from agent-facing CLI/API commands."""
        plan = self._plan(plan_id)
        self._check_plan(plan)
        if plan.spec.endpoint is not None:
            raise PermissionError('EXPLICIT_EXPORT_APPROVAL_REQUIRED')
        return self._issue_grant(plan, subject=subject, export_allowed=False)

    @serialized
    def _authorize_remote(self, plan_id, *, plan_digest, bundle_digest, target_profile_digest, subject='local-user'):
        """Trusted host approval of exact export/target scope; not a worker or model capability."""
        plan = self._plan(plan_id)
        self._check_plan(plan)
        if (plan.spec.remote is None or plan.digest != plan_digest or plan.spec.bundle_digest != bundle_digest
                or plan.spec.remote.target_profile_digest != target_profile_digest):
            raise PermissionError('EXPORT_APPROVAL_SCOPE_MISMATCH')
        return self._issue_grant(plan, subject=subject, export_allowed=True)

    @serialized
    def _authorize_managed(self, plan_id, *, plan_digest, bundle_digest, target_profile_digest,
                           managed_exposure_acknowledged, persistent_storage_allowed, subject='local-user'):
        """Trusted host approval only. Never exposed as an agent authority or worker token."""
        plan = self._plan(plan_id)
        self._check_plan(plan)
        m = plan.spec.managed
        if (m is None or plan.digest != plan_digest or plan.spec.bundle_digest != bundle_digest
                or m.target_profile_digest != target_profile_digest):
            raise PermissionError('PAID_APPROVAL_SCOPE_MISMATCH')
        paid = PaidApproval(budget_id=m.budget.budget_id, reservation=m.quote.reservation,
            managed_exposure_acknowledged=managed_exposure_acknowledged,
            persistent_storage_allowed=persistent_storage_allowed)
        return self._issue_grant(plan, subject=subject, export_allowed=True, paid=paid)

    @staticmethod
    def _new_grant(plan, *, subject, export_allowed, paid=None):
        return AuthorizationGrant(grant_id=identifier('grant'), subject=subject,
            workspace_id=plan.workspace_id, plan_digest=plan.digest, expires_ms=plan.expires_ms, export_allowed=export_allowed, paid=paid,
            max_amount=paid.reservation if paid else Money())

    def _store_grant(self, grant):
        if len(self.journal.all('budget_grants')) >= self.policy.max_retained_jobs * 4:
            raise ValueError('JOURNAL_CAPACITY')
        self.journal.put('budget_grants', grant.grant_id, {'grant': grant.model_dump(mode='json'), 'job_id': None})

    def _issue_grant(self, plan, *, subject, export_allowed, paid=None):
        grant = self._new_grant(plan, subject=subject, export_allowed=export_allowed, paid=paid)
        with self.journal.transaction():
            self._store_grant(grant)
        return grant

    @serialized
    def submit(self, *, plan_id, authorization_ref, client_request_id):
        with self.journal.transaction():
            handle = self._prepare_submission(plan_id, authorization_ref, client_request_id)
        self.reconcile(handle.job_id)
        return handle

    def _prepare_submission(self, plan_id, authorization_ref, client_request_id, *, batch_binding=None):
        # Caller owns the transaction; this function performs no process or provider I/O.
        JobHandle(job_id='check', attempt_id='check', client_request_id=client_request_id)
        plan = self._plan(plan_id)
        fingerprint = digest({'plan_digest': plan.digest, 'authorization_ref': authorization_ref})
        jobs = self.journal.all('jobs')
        for job in jobs:
            if job['client_request_id'] == client_request_id:
                if job['fingerprint'] != fingerprint:
                    raise PermissionError('REQUEST_ID_CONFLICT')
                return self._handle(job)
        self._check_plan(plan)
        record = self.journal.get('budget_grants', authorization_ref)
        grant = parse(AuthorizationGrant, canonical(record['grant']))
        if (record['job_id'] or grant.plan_digest != plan.digest or grant.workspace_id != self.workspace_id
                or now_ms() >= grant.expires_ms or grant.export_allowed != (plan.spec.endpoint is not None)):
            raise PermissionError('AUTHORIZATION_INVALID')
        if plan.spec.managed:
            m = plan.spec.managed
            if (grant.paid is None or grant.paid.budget_id != m.budget.budget_id
                    or grant.paid.reservation != m.quote.reservation or grant.max_amount != m.quote.reservation):
                raise PermissionError('PAID_AUTHORIZATION_INVALID')
            budgets.reserve(self.journal, m)
        elif grant.paid is not None:
            raise PermissionError('PAID_AUTHORIZATION_SCOPE')
        if len(jobs) >= self.policy.max_retained_jobs:
            raise ValueError('JOURNAL_CAPACITY')
        # Unresolved cleanup retains the concurrency reservation.
        active = sum(j['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'} for j in jobs)
        if batch_binding is None and active >= self.policy.max_concurrent_jobs:
            raise PermissionError('CONCURRENCY_LIMIT')
        job_id, attempt_id = identifier('job'), identifier('attempt')
        attempt = AttemptRecord(attempt_id=attempt_id, execution_digest=plan.execution_digest,
            bundle_digest=plan.spec.bundle_digest, workspace_id=self.workspace_id, spec=plan.spec,
            start_deadline_ms=grant.expires_ms if plan.spec.endpoint or batch_binding else None, batch=batch_binding)
        job = dict(job_id=job_id, attempt_id=attempt_id, client_request_id=client_request_id,
            workspace_id=self.workspace_id, plan_id=plan_id, authorization_ref=authorization_ref,
            fingerprint=fingerprint, revision=0, observation_revision=-1, execution='PREPARED',
            verification='NOT_REQUESTED' if plan.spec.bundle.request.verification == 'inspect_only' else 'PENDING',
            artifacts='NONE', resources='ALLOCATION_INTENT', candidate_ref=None, accepted_result_ref=None,
            cancel_requested=False, message='Submit intent committed before any process launch.')
        record['job_id'] = job_id
        self.journal.put('budget_grants', authorization_ref, record)
        self.journal.put('budget_reservations', job_id, self._reservation(plan.spec, 'RESERVED'))
        self.journal.put('attempts', attempt_id, {'job_id': job_id, 'attempt_number': 1, 'attempt': attempt.model_dump(mode='json')})
        self.journal.put('resource_leases', attempt_id, ResourceLease(lease_id=identifier('lease'),
            attempt_id=attempt_id, generation=attempt_id, expiry_ms=(grant.expires_ms if plan.spec.endpoint else now_ms()) +
                (plan.spec.managed.resources.ttl_ms if plan.spec.managed else plan.spec.remote.allocation.walltime_seconds * 1000 if plan.spec.remote and plan.spec.remote.allocation else plan.spec.bundle.request.resources.execution_timeout_ms),
            cleanup='ALLOCATION_INTENT'))
        self.journal.put('outbox', attempt_id, {'job_id': job_id, 'state': 'PENDING', 'kind': 'submit'})
        self._save(job, 'submit_intent')
        return self._handle(job)

    @staticmethod
    def _reservation(spec, state):
        if spec.managed:
            m = spec.managed
            return {'state': state, 'amount': m.quote.reservation.amount, 'currency': 'USD',
                    'budget_id': m.budget.budget_id, 'account_scope': m.account_scope,
                    'quote_id': m.quote.quote_id}
        if spec.remote:
            allocation = spec.remote.allocation
            return {'state': state, 'direct_currency': 'UNKNOWN', 'allocation_units': 'UNKNOWN',
                    'maximum_cpu_seconds': allocation.walltime_seconds if allocation else (spec.bundle.request.resources.execution_timeout_ms + 999) // 1000,
                    'memory_mib': allocation.memory_mib if allocation else None}
        return {'amount': '0.00', 'currency': 'EUR', 'state': state}

    @staticmethod
    def _handle(job):
        return JobHandle(**{k: job[k] for k in ('job_id', 'attempt_id', 'client_request_id')})

    def _save(self, job, reason):
        job['revision'] += 1
        JobRecord.model_validate(job)
        self.journal.put('jobs', job['job_id'], job)
        self.journal.put('events', f"{job['job_id']}_{job['revision']}", {'sequence': job['revision'],
            'time_ms': now_ms(), 'actor': 'local-controller', 'reason': reason, 'execution': job['execution'],
            'verification': job['verification'], 'resources': job['resources']})

    def _attempt(self, job):
        a = parse(AttemptRecord, canonical(self.journal.get('attempts', job['attempt_id'])['attempt']))
        if digest(a.spec) != a.execution_digest or digest(a.spec.bundle) != a.bundle_digest:
            raise ValueError('ATTEMPT_BINDING_MISMATCH')
        return a

    def _job(self, job_id):
        job = parse(JobRecord, canonical(self.journal.get('jobs', job_id))).model_dump(mode='json')
        if job['workspace_id'] != self.workspace_id:
            raise PermissionError('WORKSPACE_MISMATCH')
        return job

    @serialized
    def reconcile(self, job_id=None):
        jobs = [self._job(job_id)] if job_id else self.journal.all('jobs')
        for old in jobs:
            job = self._job(old['job_id'])
            attempt = self._attempt(job)
            outbox = self.journal.get('outbox', attempt.attempt_id)
            if outbox['state'] in {'REJECTED', 'CANCELLED'}:
                continue
            if outbox['state'] == 'PENDING' and job['cancel_requested']:
                with self.journal.transaction():
                    job.update(execution='CANCELLED', resources='NOT_OWNED')
                    outbox['state'] = 'CANCELLED'
                    self.journal.put('outbox', attempt.attempt_id, outbox)
                    self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'VOID'))
                    lease = self.journal.get('resource_leases', attempt.attempt_id)
                    lease['cleanup'] = 'NOT_OWNED'
                    self.journal.put('resource_leases', attempt.attempt_id, lease)
                    self._save(job, 'cancelled_before_dispatch')
                continue
            if outbox['state'] == 'PENDING':
                plan = self._plan(job['plan_id'])
                try:
                    self._check_plan(plan)
                    grant = parse(AuthorizationGrant, canonical(self.journal.get('budget_grants', job['authorization_ref'])['grant']))
                    if now_ms() >= grant.expires_ms:
                        raise PermissionError('AUTHORIZATION_EXPIRED')
                except PermissionError as exc:
                    with self.journal.transaction():
                        job.update(execution='FAILED', resources='NOT_OWNED', message=str(exc))
                        outbox['state'] = 'REJECTED'
                        self.journal.put('outbox', attempt.attempt_id, outbox)
                        self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'VOID'))
                        lease = self.journal.get('resource_leases', attempt.attempt_id)
                        lease['cleanup'] = 'NOT_OWNED'
                        self.journal.put('resource_leases', attempt.attempt_id, lease)
                        self._save(job, 'dispatch_denied')
                    continue
                active = sum(j['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'} and
                    self.journal.get('outbox', j['attempt_id'])['state'] != 'PENDING' for j in self.journal.all('jobs'))
                if active >= self.policy.max_concurrent_jobs:
                    continue
                with self.journal.transaction():
                    outbox['state'] = 'DISPATCHING'
                    job['execution'] = 'SUBMITTING'
                    self.journal.put('outbox', attempt.attempt_id, outbox)
                    self._save(job, 'dispatch_intent')
                try:
                    self._executor_for(attempt).submit(attempt)
                except ManagedNotSubmitted:
                    with self.journal.transaction():
                        job.update(execution='FAILED', resources='NOT_OWNED',
                                   message='Provider invocation was not started; staging/storage costs require review.')
                        outbox['state'] = 'REJECTED'
                        self.journal.put('outbox', attempt.attempt_id, outbox)
                        self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'EXPOSURE_UNKNOWN'))
                        lease = self.journal.get('resource_leases', attempt.attempt_id)
                        lease['cleanup'] = 'NOT_OWNED'
                        self.journal.put('resource_leases', attempt.attempt_id, lease)
                        self._save(job, 'managed_invocation_not_started')
                    continue
                except Exception:
                    with self.journal.transaction():
                        job.update(execution='SUBMISSION_UNKNOWN', message='Submission acknowledgement unavailable; reconcile without replacement.')
                        if attempt.spec.managed:
                            self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'EXPOSURE_UNKNOWN'))
                        self._save(job, 'submission_unknown')
                    continue
            try:
                observation = self._executor_for(attempt).observe(attempt)
            except (RemoteUnavailable, FileNotFoundError):
                if attempt.spec.endpoint is None:
                    raise
                self._unreachable(job)
                continue
            if job.get('transport') == 'UNAVAILABLE':
                with self.journal.transaction():
                    job['transport'] = 'AVAILABLE'
                    self._save(job, 'transport_restored')
            if not isinstance(observation, dict) or type(observation.get('revision')) is not int or observation['revision'] < 0:
                raise ValueError('OBSERVATION_SCHEMA_INVALID')
            if attempt.spec.endpoint and not {'attempt_id', 'execution_digest'} <= set(observation):
                raise ValueError('OBSERVATION_BINDING_MISSING')
            if 'attempt_id' in observation and (observation['attempt_id'] != attempt.attempt_id
                    or observation['execution_digest'] != attempt.execution_digest):
                raise ValueError('OBSERVATION_BINDING_MISMATCH')
            if attempt.spec.managed and job['execution'] in TERMINAL:
                observation = dict(observation, execution=job['execution'])
                if job['resources'] == 'RELEASE_CONFIRMED' and observation['resources'] == 'CLEANUP_UNKNOWN':
                    observation['resources'] = 'RELEASE_CONFIRMED'
            if observation['revision'] == job['observation_revision'] and (
                    observation['execution'] != job['execution'] or observation['resources'] != job['resources']):
                raise ValueError('OBSERVATION_CONFLICT: unchanged revision rewrites confirmed facts')
            if observation['revision'] > job['observation_revision']:
                if job['execution'] in TERMINAL and observation['execution'] != job['execution']:
                    if not attempt.spec.managed:
                        continue
                    observation = dict(observation, execution=job['execution'])
                with self.journal.transaction():
                    job.update(execution=observation['execution'], resources=observation['resources'],
                        observation_revision=observation['revision'], message=observation.get('message', 'Observed supervisor.'))
                    lease = self.journal.get('resource_leases', attempt.attempt_id)
                    lease['cleanup'] = job['resources']
                    self.journal.put('resource_leases', attempt.attempt_id, lease)
                    if job['resources'] == 'RELEASE_CONFIRMED' and not attempt.spec.managed:
                        self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'CLOSED'))
                        self.journal.put('usage_entries', job['job_id'], observation.get('usage', {'existing_host_cost': 'UNKNOWN', 'allocation_units': 'UNKNOWN'} if attempt.spec.remote else {'provider_amount': '0.00', 'currency': 'EUR', 'local_energy': 'UNKNOWN'}))
                    if attempt.spec.managed:
                        reservation = self.journal.get('budget_reservations', job['job_id'])
                        if reservation['state'] not in {'SETTLED', 'VOID'} and job['resources'] in {'RELEASE_CONFIRMED', 'CLEANUP_UNKNOWN'}:
                            reservation['state'] = 'EXPOSURE_UNKNOWN'
                            self.journal.put('budget_reservations', job['job_id'], reservation)
                    if job['execution'] in TERMINAL:
                        outbox['state'] = 'OBSERVED'
                        self.journal.put('outbox', attempt.attempt_id, outbox)
                    self._save(job, 'observation')
            if job['cancel_requested'] and job['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'}:
                try:
                    self._executor_for(attempt).cancel(attempt)
                except RemoteUnavailable:
                    self._unreachable(job)
            if job['execution'] == 'RECEIVED' and not job['candidate_ref']:
                self.fetch(job['job_id'])
        return self.status(job_id) if job_id else self.list()

    def _unreachable(self, job):
        if job.get('transport') != 'UNAVAILABLE':
            with self.journal.transaction():
                job['transport'] = 'UNAVAILABLE'
                job['message'] = 'Remote observation unavailable; last execution facts retained and cleanup unresolved.'
                reservation = self.journal.get('budget_reservations', job['job_id'])
                if reservation.get('budget_id') and reservation['state'] not in {'VOID', 'SETTLED'}:
                    reservation['state'] = 'EXPOSURE_UNKNOWN'
                    self.journal.put('budget_reservations', job['job_id'], reservation)
                self._save(job, 'remote_unavailable')

    def status(self, job_id):
        j = self._job(job_id)
        values = {k: j[k] for k in ('job_id', 'execution', 'verification', 'artifacts',
                                  'resources', 'candidate_ref', 'accepted_result_ref')}
        attempt = self._attempt(j)
        if attempt.spec.managed:
            state = self.journal.get('budget_reservations', job_id)['state']
            values['cost'] = 'USER_RECONCILED' if state == 'SETTLED' else 'NO_METERED_ALLOCATION' if state == 'VOID' else 'EXPOSURE_UNKNOWN' if state == 'EXPOSURE_UNKNOWN' else 'RESERVED'
        if attempt.spec.remote:
            values['cost'] = 'ALLOCATION_USAGE_UNKNOWN' if attempt.spec.remote.adapter == 'slurm' else 'EXISTING_HOST_COST_UNMEASURED'
        if j.get('transport') == 'UNAVAILABLE' and j['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'}:
            values['resources'] = 'CLEANUP_UNKNOWN'
        values['transport'] = j.get('transport', 'AVAILABLE')
        return ComputeResultReceipt(**values)

    def list(self, *, limit=25, offset=0):
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 100 or offset < 0:
            raise ValueError('Invalid page bounds')
        return tuple(self.status(j['job_id']) for j in self.journal.all('jobs')[offset:offset+limit])

    @serialized
    def cancel(self, job_id):
        with self.journal.transaction():
            job = self._job(job_id)
            if job['execution'] in TERMINAL and job['resources'] in {'RELEASE_CONFIRMED', 'NOT_OWNED'}:
                return self.status(job_id)
            job['cancel_requested'] = True
            job['message'] = 'Cancellation intent retained; stopping/cleanup not yet confirmed.'
            self._save(job, 'cancel_intent')
        if self.journal.get('outbox', job['attempt_id'])['state'] == 'PENDING':
            return self.reconcile(job_id)
        try:
            self._executor_for(self._attempt(job)).cancel(self._attempt(job))
        except RemoteUnavailable:
            self._unreachable(job)
        return self.status(job_id)

    @serialized
    def fetch(self, job_id):
        job = self._job(job_id)
        attempt = self._attempt(job)
        try:
            raw = self._executor_for(attempt).fetch(attempt)
        except RemoteUnavailable:
            self._unreachable(job)
            return self.status(job_id)
        except FileNotFoundError:
            with self.journal.transaction():
                job['artifacts'] = 'UNAVAILABLE'
                self._save(job, 'output_missing')
            return self.status(job_id)
        # Raw candidate bytes enter quarantine first, never MathResult validators.
        ref = self.store.put(raw, attempt.spec.bundle.request.resources.max_output_bytes)
        if job['candidate_ref'] and job['candidate_ref'] != ref:
            raise ValueError('CONFLICTING_OUTPUT: immutable candidate already retained')
        try:
            candidate = parse(RemoteResultEnvelope, raw)
            check_binding(attempt, candidate)
        except ValueError:
            with self.journal.transaction():
                job.update(candidate_ref=ref, artifacts='REJECTED', verification='FAILED')
                self._save(job, 'candidate_rejected')
            return self.status(job_id)
        with self.journal.transaction():
            if job['candidate_ref'] and job['candidate_ref'] != ref:
                raise ValueError('CONFLICTING_OUTPUT: an attempt cannot replace its candidate')
            job.update(candidate_ref=ref, artifacts='QUARANTINED')
            self.journal.put('artifact_manifests', ref, {'digest': ref, 'bytes': len(raw), 'schema': candidate.output_schema})
            self.journal.put('artifact_refs', attempt.attempt_id, {'candidate_ref': ref, 'workspace_id': self.workspace_id})
            self._save(job, 'candidate_quarantined')
        return self.status(job_id)

    def candidate(self, job_id):
        job = self._job(job_id)
        if not job['candidate_ref'] or job['artifacts'] == 'REJECTED':
            raise ValueError('CANDIDATE_UNAVAILABLE')
        return parse(RemoteResultEnvelope, self.store.get(job['candidate_ref']))

    @serialized
    def verify(self, job_id):
        job = self._job(job_id)
        if job['accepted_result_ref']:
            return self.status(job_id)
        attempt, candidate = self._attempt(job), self.candidate(job_id)
        if (attempt.spec.endpoint.verifier_runtime if attempt.spec.endpoint else attempt.spec.runtime) != runtime_profile():
            raise PermissionError('VERIFIER_RUNTIME_CHANGED: retained candidate requires explicit requalification')
        if job['cancel_requested']:
            raise PermissionError('Late output retained for inspection; cancelled jobs are not automatically admitted')
        report_id = identifier('verification')
        with self.journal.transaction():
            job['verification'] = 'RUNNING'
            self._save(job, 'verification_started')
        stream = io.BytesIO()
        write_frame(stream, {'attempt': attempt.model_dump(mode='json'), 'candidate': candidate.model_dump(mode='json'), 'report_id': report_id})
        process = subprocess.Popen([sys.executable, '-m', 'mathkernel_compute.worker', 'verify'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, env=worker_environment(), close_fds=True)
        try:
            output, _ = process.communicate(stream.getvalue(), timeout=attempt.spec.bundle.request.resources.verification_timeout_ms/1000)
            if process.returncode:
                raise ValueError('Local verifier exited without a complete report')
            report = parse(VerificationReport, read_frame(io.BytesIO(output), 65536), 65536)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            if process.poll() is None:
                if sys.platform == 'linux':
                    from .worker import terminate_owned
                    terminate_owned(process)
                else:
                    process.kill(); process.wait()
            process.communicate()
            report = VerificationReport(report_id=report_id, verifier='local-verification-supervisor',
                candidate_digest=digest(candidate), bundle_digest=attempt.bundle_digest,
                execution_digest=attempt.execution_digest, claim=attempt.spec.bundle.request.required_claim,
                outcome='INCONCLUSIVE', trust='unknown', detail=type(exc).__name__ + ': verifier budget or execution failure')
        with self.journal.transaction():
            self.journal.put('verification_reports', job_id, report)
            job['verification'] = report.outcome
            if report.outcome == 'PASSED':
                result = admit(attempt, candidate, report)
                raw = canonical(result)
                accepted = self.store.put(raw)
                job['accepted_result_ref'] = accepted
                self.journal.put('artifact_refs', job_id, {'accepted_result_ref': accepted, 'report_id': report_id})
            self._save(job, 'verification_finished')
        return self.status(job_id)

    def result(self, job_id):
        return self.status(job_id)

    def accepted_result(self, job_id):
        job = self._job(job_id)
        if not job['accepted_result_ref']:
            raise ValueError('RESULT_NOT_ADMITTED')
        # Only records created by admit(), referenced by this journal, reach MathResult.
        from mathkernel.models import MathResult
        raw = self.store.get(job['accepted_result_ref'])
        from .protocol import decode
        return MathResult.model_validate(decode(raw))

    def wait(self, job_id, *, timeout_s):
        if isinstance(timeout_s, bool) or not 0 <= timeout_s <= 3600:
            raise ValueError('A bounded wait is required')
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.reconcile(job_id)
            if status.execution in TERMINAL:
                return status
            if time.monotonic() >= deadline:
                return status
            time.sleep(.5 if self._attempt(self._job(job_id)).spec.endpoint else .05)

    def budget_status(self, budget_id):
        return budgets.summary(self.journal, self.budget_limits[budget_id])

    @serialized
    def plan_batch(self, request):
        request = parse(BatchRequest, canonical(request))
        shards = tuple(PlannedShard(shard_id=s.shard_id, plan=self._build_plan(s.request)) for s in request.shards)
        if any(s.plan.spec.remote or (s.plan.spec.managed and s.plan.spec.managed.adapter != 'runpod') for s in shards):
            raise ValueError('Independent batches support local workers and existing Runpod endpoints')
        plan = BatchPlan(batch_plan_id=identifier('batchplan'), workspace_id=self.workspace_id,
                         shards=shards, expires_ms=min(s.plan.expires_ms for s in shards))
        if len(canonical(plan)) > 1_048_576:
            raise ValueError('BATCH_MANIFEST_LIMIT')
        with self.journal.transaction():
            if len(self.journal.all('batch_plans')) >= self.policy.max_retained_jobs:
                raise ValueError('JOURNAL_CAPACITY')
            for shard in shards:
                self._store_plan(shard.plan)
            self.journal.put('batch_plans', plan.batch_plan_id, plan)
        return plan

    def _batch_plan(self, plan_id):
        plan = parse(BatchPlan, canonical(self.journal.get('batch_plans', plan_id)))
        if plan.workspace_id != self.workspace_id or len({s.shard_id for s in plan.shards}) != len(plan.shards):
            raise ValueError('BATCH_PLAN_BINDING')
        for shard in plan.shards:
            if self._plan(shard.plan.plan_id) != shard.plan:
                raise ValueError('BATCH_CHILD_PLAN_CHANGED')
        return plan

    @serialized
    def _authorize_batch(self, plan_id, *, plan_digest, export_allowed=False,
                         managed_exposure_acknowledged=False, persistent_storage_allowed=False, subject='local-user'):
        """Host approval covers the entire immutable set, never an open-ended shard generator."""
        plan = self._batch_plan(plan_id)
        if plan.digest != plan_digest or now_ms() >= plan.expires_ms:
            raise PermissionError('BATCH_APPROVAL_SCOPE')
        for shard in plan.shards:
            self._check_plan(shard.plan)
            if shard.plan.spec.managed:
                if not export_allowed:
                    raise PermissionError('BATCH_EXPORT_APPROVAL_REQUIRED')
                PaidApproval(budget_id=shard.plan.spec.managed.budget.budget_id,
                    reservation=shard.plan.spec.managed.quote.reservation,
                    managed_exposure_acknowledged=managed_exposure_acknowledged,
                    persistent_storage_allowed=persistent_storage_allowed)
        grant = BatchGrant(grant_id=identifier('batchgrant'), workspace_id=self.workspace_id,
            batch_plan_digest=plan.digest, subject=subject, expires_ms=plan.expires_ms,
            export_allowed=export_allowed, managed_exposure_acknowledged=managed_exposure_acknowledged,
            persistent_storage_allowed=persistent_storage_allowed)
        with self.journal.transaction():
            if len(self.journal.all('batch_grants')) >= self.policy.max_retained_jobs:
                raise ValueError('JOURNAL_CAPACITY')
            self.journal.put('batch_grants', grant.grant_id, {'grant': grant.model_dump(mode='json'), 'batch_id': None})
        return grant

    @serialized
    def submit_batch(self, *, plan_id, authorization_ref, client_request_id):
        BatchHandle(batch_id='check', client_request_id=client_request_id,
                    shards=(ShardHandle(shard_id='check', job=JobHandle(job_id='check', attempt_id='check', client_request_id='check')),))
        plan = self._batch_plan(plan_id)
        fingerprint = digest({'plan_digest': plan.digest, 'authorization_ref': authorization_ref})
        with self.journal.transaction():
            for old in self.journal.all('batches'):
                if old['handle']['client_request_id'] == client_request_id:
                    if old['fingerprint'] != fingerprint:
                        raise PermissionError('BATCH_REQUEST_ID_CONFLICT')
                    return parse(BatchHandle, canonical(old['handle']))
            record = self.journal.get('batch_grants', authorization_ref)
            grant = parse(BatchGrant, canonical(record['grant']))
            if (record['batch_id'] or grant.workspace_id != self.workspace_id or
                    grant.batch_plan_digest != plan.digest or now_ms() >= grant.expires_ms):
                raise PermissionError('BATCH_GRANT_INVALID')
            batch_id = identifier('batch')
            handles = []
            for shard in plan.shards:
                m = shard.plan.spec.managed
                paid = PaidApproval(budget_id=m.budget.budget_id, reservation=m.quote.reservation,
                    managed_exposure_acknowledged=grant.managed_exposure_acknowledged,
                    persistent_storage_allowed=grant.persistent_storage_allowed) if m else None
                if m and not grant.export_allowed:
                    raise PermissionError('BATCH_EXPORT_APPROVAL_REQUIRED')
                child_grant = self._new_grant(shard.plan, subject=grant.subject, export_allowed=bool(m), paid=paid)
                child_grant = child_grant.model_copy(update={'expires_ms': min(child_grant.expires_ms, grant.expires_ms)})
                self._store_grant(child_grant)
                key = digest({'batch_id': batch_id, 'shard_id': shard.shard_id})
                job = self._prepare_submission(shard.plan.plan_id, child_grant.grant_id, key,
                    batch_binding=BatchBinding(batch_id=batch_id, shard_id=shard.shard_id, batch_plan_digest=plan.digest))
                handles.append(ShardHandle(shard_id=shard.shard_id, job=job))
            handle = BatchHandle(batch_id=batch_id, client_request_id=client_request_id, shards=tuple(handles))
            record['batch_id'] = batch_id
            self.journal.put('batch_grants', authorization_ref, record)
            self.journal.put('batches', batch_id, {'handle': handle.model_dump(mode='json'), 'plan_id': plan_id,
                             'fingerprint': fingerprint})
        # All child jobs, grants, reservations, and the exact manifest are committed before dispatch.
        self.reconcile_batch(batch_id)
        return handle

    def _batch(self, batch_id):
        record = self.journal.get('batches', batch_id)
        handle = parse(BatchHandle, canonical(record['handle']))
        plan = self._batch_plan(record['plan_id'])
        if handle.batch_id != batch_id or tuple(s.shard_id for s in handle.shards) != tuple(s.shard_id for s in plan.shards):
            raise ValueError('BATCH_COVERAGE_MISMATCH')
        if len({s.job.job_id for s in handle.shards}) != len(handle.shards):
            raise ValueError('DUPLICATE_SHARD_JOB')
        for shard, planned in zip(handle.shards, plan.shards):
            job = self._job(shard.job.job_id)
            attempt = self._attempt(job)
            if (self._handle(job) != shard.job or job['plan_id'] != planned.plan.plan_id or
                    attempt.batch != BatchBinding(batch_id=batch_id, shard_id=shard.shard_id, batch_plan_digest=plan.digest)):
                raise ValueError('BATCH_SHARD_BINDING')
        return handle

    def batch_status(self, batch_id):
        handle = self._batch(batch_id)
        shards = tuple({'shard_id': s.shard_id, 'receipt': self.status(s.job.job_id).model_dump(mode='json')} for s in handle.shards)
        return {'batch_id': batch_id, 'expected_shards': len(shards),
                'admitted_shards': sum(s['receipt']['accepted_result_ref'] is not None for s in shards),
                'shards': shards, 'semantics': 'Per-shard locally admitted results only; no aggregate mathematical claim.'}

    @serialized
    def reconcile_batch(self, batch_id):
        for shard in self._batch(batch_id).shards:
            self.reconcile(shard.job.job_id)
        return self.batch_status(batch_id)

    @serialized
    def cancel_batch(self, batch_id):
        for shard in self._batch(batch_id).shards:
            self.cancel(shard.job.job_id)
        return self.batch_status(batch_id)

    @serialized
    def _reconcile_cost(self, job_id, amount, source, *, retained_storage_accounted):
        """Trusted host attestation of billing, not a provider or model claim."""
        with self.journal.transaction():
            record = budgets.settle(self.journal, job_id, amount, source,
                                    retained_storage_accounted=retained_storage_accounted)
            self._save(self._job(job_id), 'cost_user_reconciled')
        return record

    def export_replay(self, job_id):
        job = self._job(job_id)
        # Data only: no grant, shell command, credentials, absolute paths or automatic replay.
        return canonical({'schema': 'mk.compute-replay/1', 'plan': self._plan(job['plan_id']).model_dump(mode='json'),
                          'attempt': self._attempt(job).model_dump(mode='json'), 'status': self.status(job_id).model_dump(mode='json')})

    def close(self):
        if not self.closed:
            self.closed = True
            self.journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
