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
    TargetCapabilities, VerificationReport, JobRecord, RemoteBinding, Money)
from .protocol import canonical, digest, parse, read_frame, write_frame
from .registry import runtime_profile
from .verification import admit, check_binding
from .remote import SSHProfile, SSHExecutor, RemoteUnavailable

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
    def __init__(self, *, state_dir, kernel=None, policy=None, remote_targets=(), reconcile_on_open=True, _executor=None):
        self._coordinator_lock = threading.RLock()
        self.kernel = kernel  # No remote method is added to the facade.
        self.policy = policy or LocalPolicy()
        profiles = [parse(SSHProfile, canonical(p)) for p in remote_targets]
        self.remote_targets = {p.target_id: p for p in profiles}
        if len(self.remote_targets) != len(profiles):
            raise ValueError('DUPLICATE_TARGET_ALIAS')
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
        return (local, *remote)

    def target_probe(self, target='local-cpu'):
        if target == 'local-cpu':
            return self.targets()[0]
        try:
            return SSHExecutor(self.remote_targets[target]).probe()
        except KeyError:
            raise ValueError('TARGET_NOT_CONFIGURED') from None

    def _executor_for(self, attempt):
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
        request = parse(ComputeRequest, canonical(request))
        target = 'local-cpu' if request.target == 'auto' else request.target
        remote = target != 'local-cpu'
        if remote and target not in self.remote_targets:
            raise ValueError('TARGET_NOT_CONFIGURED')
        if not remote and not self.targets()[0].available:
            raise ValueError('TARGET_UNSUPPORTED')
        local_runtime = runtime_profile()
        profile = self.remote_targets[target] if remote else None
        bundle = InputBundle(request=request, classification='explicit_export' if remote else 'local_only')
        spec = ExecutionSpec(bundle=bundle, bundle_digest=digest(bundle), target=target,
            runtime=profile.runtime if profile else local_runtime, policy_digest=digest(self.policy),
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
        plan = ComputePlan(plan_id=identifier('plan'), workspace_id=self.workspace_id, created_ms=now_ms(),
            expires_ms=now_ms() + self.policy.plan_lifetime_ms, spec=spec, execution_digest=digest(spec),
            provider_cost=None if remote else Money(),
            warnings=warnings, rejected_targets=('Managed/GPU providers, Apptainer and automatic selection of remote targets are not enabled.',))
        with self.journal.transaction():
            if len(self.journal.all('plans')) >= self.policy.max_retained_jobs * 4:
                raise ValueError('JOURNAL_CAPACITY')
            self.journal.put('plans', plan.plan_id, plan)
            self.journal.put('execution_specs', plan.execution_digest, spec)
        return plan

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
        verifier_runtime = plan.spec.remote.verifier_runtime if plan.spec.remote else plan.spec.runtime
        if plan.spec.policy_digest != digest(self.policy) or verifier_runtime != runtime_profile():
            raise PermissionError('PLAN_CHANGED')
        if plan.spec.remote:
            profile = self.remote_targets.get(plan.spec.target)
            if profile is None or profile.profile_digest != plan.spec.remote.target_profile_digest:
                raise PermissionError('TARGET_PROFILE_CHANGED')

    @serialized
    def _authorize_local(self, plan_id, *, subject='local-user'):
        """Trusted host entrypoint; deliberately absent from agent-facing CLI/API commands."""
        plan = self._plan(plan_id)
        self._check_plan(plan)
        if plan.spec.remote is not None:
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

    def _issue_grant(self, plan, *, subject, export_allowed):
        grant = AuthorizationGrant(grant_id=identifier('grant'), subject=subject,
            workspace_id=self.workspace_id, plan_digest=plan.digest, expires_ms=plan.expires_ms, export_allowed=export_allowed)
        with self.journal.transaction():
            if len(self.journal.all('budget_grants')) >= self.policy.max_retained_jobs * 4:
                raise ValueError('JOURNAL_CAPACITY')
            self.journal.put('budget_grants', grant.grant_id, {'grant': grant.model_dump(mode='json'), 'job_id': None})
        return grant

    @serialized
    def submit(self, *, plan_id, authorization_ref, client_request_id):
        # Validate request ID before consulting or modifying any journal state.
        JobHandle(job_id='check', attempt_id='check', client_request_id=client_request_id)
        plan = self._plan(plan_id)
        fingerprint = digest({'plan_digest': plan.digest, 'authorization_ref': authorization_ref})
        with self.journal.transaction():
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
                    or now_ms() >= grant.expires_ms or grant.export_allowed != (plan.spec.remote is not None)):
                raise PermissionError('AUTHORIZATION_INVALID')
            if len(jobs) >= self.policy.max_retained_jobs:
                raise ValueError('JOURNAL_CAPACITY')
            # Unresolved cleanup retains the concurrency reservation.
            active = sum(j['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'} for j in jobs)
            if active >= self.policy.max_concurrent_jobs:
                raise PermissionError('CONCURRENCY_LIMIT')
            job_id, attempt_id = identifier('job'), identifier('attempt')
            attempt = AttemptRecord(attempt_id=attempt_id, execution_digest=plan.execution_digest,
                bundle_digest=plan.spec.bundle_digest, workspace_id=self.workspace_id, spec=plan.spec,
                start_deadline_ms=grant.expires_ms if plan.spec.remote else None)
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
                attempt_id=attempt_id, generation=attempt_id, expiry_ms=(grant.expires_ms if plan.spec.remote else now_ms()) +
                    (plan.spec.remote.allocation.walltime_seconds * 1000 if plan.spec.remote and plan.spec.remote.allocation else plan.spec.bundle.request.resources.execution_timeout_ms),
                cleanup='ALLOCATION_INTENT'))
            self.journal.put('outbox', attempt_id, {'job_id': job_id, 'state': 'PENDING', 'kind': 'submit'})
            self._save(job, 'submit_intent')
        self.reconcile(job_id)
        return self._handle(job)

    @staticmethod
    def _reservation(spec, state):
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
            if outbox['state'] == 'PENDING' and job['cancel_requested']:
                with self.journal.transaction():
                    job.update(execution='CANCELLED', resources='NOT_OWNED')
                    outbox['state'] = 'CANCELLED'
                    self.journal.put('outbox', attempt.attempt_id, outbox)
                    self._save(job, 'cancelled_before_dispatch')
                continue
            if outbox['state'] == 'PENDING':
                plan = self._plan(job['plan_id'])
                try:
                    self._check_plan(plan)
                except PermissionError as exc:
                    with self.journal.transaction():
                        job.update(execution='FAILED', resources='NOT_OWNED', message=str(exc))
                        outbox['state'] = 'REJECTED'
                        self.journal.put('outbox', attempt.attempt_id, outbox)
                        self._save(job, 'dispatch_denied')
                    continue
                with self.journal.transaction():
                    outbox['state'] = 'DISPATCHING'
                    job['execution'] = 'SUBMITTING'
                    self.journal.put('outbox', attempt.attempt_id, outbox)
                    self._save(job, 'dispatch_intent')
                try:
                    self._executor_for(attempt).submit(attempt)
                except Exception:
                    with self.journal.transaction():
                        job.update(execution='SUBMISSION_UNKNOWN', message='Submission acknowledgement unavailable; reconcile without replacement.')
                        self._save(job, 'submission_unknown')
                    continue
            try:
                observation = self._executor_for(attempt).observe(attempt)
            except (RemoteUnavailable, FileNotFoundError):
                if attempt.spec.remote is None:
                    raise
                self._unreachable(job)
                continue
            if job.get('transport') == 'UNAVAILABLE':
                with self.journal.transaction():
                    job['transport'] = 'AVAILABLE'
                    self._save(job, 'transport_restored')
            if not isinstance(observation, dict) or type(observation.get('revision')) is not int or observation['revision'] < 0:
                raise ValueError('OBSERVATION_SCHEMA_INVALID')
            if attempt.spec.remote and not {'attempt_id', 'execution_digest'} <= set(observation):
                raise ValueError('OBSERVATION_BINDING_MISSING')
            if 'attempt_id' in observation and (observation['attempt_id'] != attempt.attempt_id
                    or observation['execution_digest'] != attempt.execution_digest):
                raise ValueError('OBSERVATION_BINDING_MISMATCH')
            if observation['revision'] == job['observation_revision'] and (
                    observation['execution'] != job['execution'] or observation['resources'] != job['resources']):
                raise ValueError('OBSERVATION_CONFLICT: unchanged revision rewrites confirmed facts')
            if observation['revision'] > job['observation_revision']:
                if job['execution'] in TERMINAL and observation['execution'] != job['execution']:
                    continue
                with self.journal.transaction():
                    job.update(execution=observation['execution'], resources=observation['resources'],
                        observation_revision=observation['revision'], message=observation.get('message', 'Observed supervisor.'))
                    lease = self.journal.get('resource_leases', attempt.attempt_id)
                    lease['cleanup'] = job['resources']
                    self.journal.put('resource_leases', attempt.attempt_id, lease)
                    if job['resources'] == 'RELEASE_CONFIRMED':
                        self.journal.put('budget_reservations', job['job_id'], self._reservation(attempt.spec, 'CLOSED'))
                        self.journal.put('usage_entries', job['job_id'], observation.get('usage', {'existing_host_cost': 'UNKNOWN', 'allocation_units': 'UNKNOWN'} if attempt.spec.remote else {'provider_amount': '0.00', 'currency': 'EUR', 'local_energy': 'UNKNOWN'}))
                    if job['execution'] in TERMINAL:
                        outbox['state'] = 'OBSERVED'
                        self.journal.put('outbox', attempt.attempt_id, outbox)
                    self._save(job, 'observation')
            if job['cancel_requested'] and job['execution'] not in TERMINAL:
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
                self._save(job, 'remote_unavailable')

    def status(self, job_id):
        j = self._job(job_id)
        values = {k: j[k] for k in ('job_id', 'execution', 'verification', 'artifacts',
                                  'resources', 'candidate_ref', 'accepted_result_ref')}
        attempt = self._attempt(j)
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
            if job['execution'] in TERMINAL:
                return self.status(job_id)
            job['cancel_requested'] = True
            job['message'] = 'Cancellation intent retained; stopping/cleanup not yet confirmed.'
            self._save(job, 'cancel_intent')
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
        if (attempt.spec.remote.verifier_runtime if attempt.spec.remote else attempt.spec.runtime) != runtime_profile():
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
            time.sleep(.5 if self._attempt(self._job(job_id)).spec.remote else .05)

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
