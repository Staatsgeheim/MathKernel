"""Native single-job Slurm backend used only by the installed SSH gateway.

Submission, scheduler visibility, accounting and candidate availability are
independent. Missing accounting never proves release; a lost sbatch reply never
causes a replacement submission. The login node executes no operation handler.
"""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import sys
import time
from .executor import LocalExecutor, worker_environment
from .files import atomic_write, bounded_read
from .protocol import canonical, decode, digest
from .remote import run_bounded, RemoteUnavailable

ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED', 'RESIZING', 'SIGNALING', 'STAGE_OUT'}
END_STATES = {'COMPLETED': 'RECEIVED', 'FAILED': 'FAILED', 'TIMEOUT': 'TIMED_OUT',
    'CANCELLED': 'CANCELLED', 'OUT_OF_MEMORY': 'OUT_OF_MEMORY', 'PREEMPTED': 'PREEMPTED',
    'NODE_FAIL': 'NODE_FAILED', 'BOOT_FAIL': 'NODE_FAILED', 'DEADLINE': 'EXPIRED', 'REVOKED': 'FAILED'}


def scheduler_environment():
    # Do not inherit SBATCH_*, SLURM_*, proxy credentials, or a user's login environment.
    return {'PATH': os.defpath, 'LANG': 'C', 'LC_ALL': 'C', 'TZ': 'UTC', 'SLURM_TIME_FORMAT': 'standard'}


def parse_rows(raw, *, accounting=False):
    """Explicit pipe-delimited columns, bounded before parsing; no shell/table guessing."""
    try:
        lines = raw.decode('utf-8', errors='strict').splitlines()
    except UnicodeError as exc:
        raise RemoteUnavailable('SCHEDULER_ENCODING') from exc
    rows = []
    for line in lines:
        parts = [part.strip() for part in line.split('|')]
        if len(parts) != (13 if accounting else 6):
            raise RemoteUnavailable('SCHEDULER_FIELDS_UNSUPPORTED')
        job_id, name, uid, state, submitted, comment = parts[:6]
        if accounting and '.' in job_id:
            continue  # Batch/extern steps are not a parent allocation release record.
        if not re.fullmatch(r'[1-9][0-9]{0,19}', job_id) or not uid.isdecimal():
            raise RemoteUnavailable('SCHEDULER_JOB_ID_UNSUPPORTED')
        state = state.split(' ', 1)[0].rstrip('+')
        row = dict(job_id=job_id, name=name, uid=uid, state=state, submitted=submitted, comment=comment)
        if accounting:
            if not re.fullmatch(r'[0-9]+:[0-9]+', parts[6]):
                raise RemoteUnavailable('SCHEDULER_EXIT_CODE_INVALID')
            if not parts[12].isdecimal():
                raise RemoteUnavailable('SCHEDULER_RESTART_COUNT_INVALID')
            row['restarts'] = int(parts[12])
            row['usage'] = dict(exit_code=parts[6], elapsed_seconds=parts[7], allocated_cpus=parts[8],
                allocated_tres=parts[9], start=parts[10], end=parts[11],
                allocation_units='UNKNOWN', direct_currency='UNKNOWN', source='sacct')
        rows.append(row)
    return rows


class SlurmExecutor(LocalExecutor):
    def __init__(self, root, config):
        super().__init__(root)
        self.config = config

    @staticmethod
    def name(attempt):
        return 'mk_' + attempt.attempt_id.removeprefix('attempt_')

    def command(self, argv, data=b''):
        return run_bounded(argv, data, timeout=30, limit=262144, env=scheduler_environment())

    def submit(self, attempt):
        directory = self.directory(attempt.attempt_id)
        directory.mkdir(mode=0o700, exist_ok=True)
        intent_path = directory / 'slurm-intent.json'
        if intent_path.exists():
            if decode(bounded_read(intent_path))['attempt_digest'] != digest(attempt):
                raise ValueError('SLURM_ATTEMPT_CONFLICT')
            return
        atomic_write(directory / 'request.json', canonical(attempt))
        c = self.config
        argv = [c.sbatch, '--parsable', '--no-requeue', '--export=NIL', '--nodes=1', '--ntasks=1',
            '--cpus-per-task=1', '--mem=' + str(c.memory_mib) + 'M',
            '--time=00:' + str(c.walltime_seconds // 60).zfill(2) + ':' + str(c.walltime_seconds % 60).zfill(2),
            '--partition=' + c.partition, '--account=' + c.account,
            '--job-name=' + self.name(attempt), '--comment=' + digest(attempt),
            '--chdir=' + str(directory), '--output=/dev/null', '--error=/dev/null']
        if c.qos:
            argv.append('--qos=' + c.qos)
        if c.constraint:
            argv.append('--constraint=' + c.constraint)
        # Fixed worker command. Only trusted installation paths/environment are rendered.
        # Input data remains in request.json; no request expression appears in this script.
        env_words = [k + '=' + v for k, v in worker_environment().items()]
        script = '#!/bin/sh\nexec /usr/bin/env -i MK_SLURM_JOB_ID="$SLURM_JOB_ID" MK_SLURM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}" ' + shlex.join([*env_words,
            sys.executable, '-m', 'mathkernel_compute.worker', 'slurm-supervise', str(directory)]) + '\n'
        atomic_write(intent_path, canonical({'attempt_digest': digest(attempt), 'uid': str(os.getuid()),
            'submitted_after_ms': time.time_ns() // 1_000_000, 'name': self.name(attempt)}))
        # Any exception after the durable intent is ambiguous, including an invalid reply.
        reply = self.command(argv, script.encode()).decode('ascii').strip()
        if not re.fullmatch(r'[1-9][0-9]{0,19}(;[A-Za-z0-9_-]{1,128})?', reply):
            raise RemoteUnavailable('SBATCH_ACK_UNKNOWN')
        job_id, _, cluster = reply.partition(';')
        # This adapter targets the gateway's single configured local cluster, no federation.
        atomic_write(directory / 'slurm-ack.json', canonical({'job_id': job_id, 'cluster': cluster}))

    def accounting_start(self, attempt):
        path = self.directory(attempt.attempt_id) / 'slurm-intent.json'
        stamp = decode(bounded_read(path))['submitted_after_ms'] / 1000 - 60
        return datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')

    def rows(self, attempt):
        c = self.config
        selector = ['--name=' + self.name(attempt), '--user=' + str(os.getuid())]
        queue, account = None, None
        try:
            queue = parse_rows(self.command([c.squeue, '--local', '--noheader', *selector,
                '--format=%i|%j|%U|%T|%V|%k']))
        except (OSError, ValueError):
            pass
        try:
            account = parse_rows(self.command([c.sacct, '--local', '--noheader', '--parsable2', '--duplicates',
                '--starttime=' + self.accounting_start(attempt) + '', *selector,
                '--format=JobIDRaw,JobName%128,UID,State%32,Submit,Comment%128,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES%512,Start,End,Restarts']),
                accounting=True)
        except (OSError, ValueError):
            pass
        return queue, account

    def owned(self, attempt, queue, account):
        directory = self.directory(attempt.attempt_id)
        intent = decode(bounded_read(directory / 'slurm-intent.json'))
        rows = (queue or []) + (account or [])
        rows = [r for r in rows if r['name'] == self.name(attempt)]
        if not rows:
            return None
        # The exact UUID name, authenticated UID, comment digest and submit incarnation must agree.
        for row in rows:
            if row.get('restarts', 0) != 0:
                raise RemoteUnavailable('SLURM_REQUEUE_NOT_SUPPORTED')
            if row['uid'] != intent['uid'] or row['comment'] != digest(attempt):
                raise RemoteUnavailable('SLURM_OWNERSHIP_MISMATCH')
            try:
                stamp = int(datetime.fromisoformat(row['submitted']).replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError as exc:
                raise RemoteUnavailable('SLURM_SUBMIT_TIME_UNKNOWN') from exc
            if not intent['submitted_after_ms'] - 5000 <= stamp <= time.time_ns() // 1_000_000 + 5000:
                raise RemoteUnavailable('SLURM_INCARNATION_MISMATCH')
        identities = {(r['job_id'], r['submitted']) for r in rows}
        if len(identities) != 1:
            raise RemoteUnavailable('SLURM_DUPLICATE_OR_REQUEUED')
        identity = dict(job_id=rows[0]['job_id'], submitted=rows[0]['submitted'], uid=intent['uid'],
                        name=intent['name'], comment=digest(attempt))
        ack = directory / 'slurm-ack.json'
        if ack.exists() and decode(bounded_read(ack))['job_id'] != identity['job_id']:
            raise RemoteUnavailable('SLURM_ACK_IDENTITY_MISMATCH')
        record = directory / 'slurm-owner.json'
        if record.exists() and decode(bounded_read(record)) != identity:
            raise RemoteUnavailable('SLURM_INCARNATION_CHANGED')
        atomic_write(record, canonical(identity))
        return identity

    def observe(self, attempt):
        directory = self.directory(attempt.attempt_id)
        observation_path = directory / 'scheduler-observation.json'
        previous = decode(bounded_read(observation_path)) if observation_path.exists() else None
        result = dict(execution='SUBMISSION_UNKNOWN', resources='CLEANUP_UNKNOWN',
            message='Scheduler visibility/accounting pending; allocation and cost remain unresolved.')
        queue, account = self.rows(attempt)
        identity = self.owned(attempt, queue, account)
        if identity:
            live = next((r for r in (queue or []) if r['job_id'] == identity['job_id']), None)
            recorded = next((r for r in (account or []) if r['job_id'] == identity['job_id']), None)
            if live and live['state'] in ACTIVE:
                result.update(execution='QUEUED' if live['state'] == 'PENDING' else 'RUNNING',
                              resources='ACTIVE', message='Allocation observed by squeue; mathematical result unverified.')
            elif recorded and recorded['state'] in END_STATES:
                outcome = END_STATES[recorded['state']]
                if outcome == 'RECEIVED' and recorded['usage']['exit_code'] != '0:0':
                    outcome = 'FAILED'
                if outcome == 'RECEIVED':
                    # Scheduler success alone does not assert either a result or a mathematical claim.
                    terminal = directory / 'terminal.json'
                    if terminal.exists():
                        supervisor = decode(bounded_read(terminal))
                        if supervisor.get('attempt_id') != attempt.attempt_id or supervisor.get('execution_digest') != attempt.execution_digest:
                            raise RemoteUnavailable('SUPERVISOR_BINDING_MISMATCH')
                        outcome = supervisor['execution']
                    else:
                        outcome = 'SUBMISSION_UNKNOWN'
                result.update(execution=outcome, resources='RELEASE_CONFIRMED',
                    message='Terminal allocation accounting retained; allocation-unit pricing is unknown.', usage=recorded['usage'])
            elif recorded and recorded['state'] in ACTIVE:
                result.update(execution='QUEUED' if recorded['state'] == 'PENDING' else 'RUNNING',
                              resources='ACTIVE', message='Active allocation observed by sacct.')
        if previous and previous['execution'] in set(END_STATES.values()) | {'EXPIRED'}:
            # Retain confirmed terminal facts through accounting retention/visibility changes.
            result = {k: v for k, v in previous.items() if k not in {'revision', 'attempt_id', 'execution_digest'}}
        same = previous and all(previous.get(k) == v for k, v in result.items()) and set(result) == set(previous) - {'revision', 'attempt_id', 'execution_digest'}
        result.update(revision=previous['revision'] if same else (previous['revision'] + 1 if previous else 0),
                      attempt_id=attempt.attempt_id, execution_digest=attempt.execution_digest)
        atomic_write(observation_path, canonical(result))
        return result

    def cancel(self, attempt):
        # A request/acknowledgement is not a termination observation.
        queue, account = self.rows(attempt)
        identity = self.owned(attempt, queue, account)
        if identity is None:
            raise RemoteUnavailable('SLURM_CANCEL_OWNERSHIP_UNKNOWN')
        row = next((r for r in (queue or []) if r['job_id'] == identity['job_id']), None)
        if row is None:
            if any(r['state'] in END_STATES for r in (account or [])):
                return
            raise RemoteUnavailable('SLURM_CANCEL_VISIBILITY_UNKNOWN')
        # The scheduler also filters UID/name, protecting against reuse between query and cancellation.
        self.command([self.config.scancel, '--user=' + identity['uid'], '--name=' + identity['name'], identity['job_id']])
