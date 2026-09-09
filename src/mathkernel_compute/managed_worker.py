"""Fixed managed worker brokers. Scientific subprocesses receive no provider tokens."""
from __future__ import annotations
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from pydantic import Field
from .models import Contract, Identifier, Digest, RuntimeProfile, ManagedResources, AttemptRecord
from .managed import S3Storage, RUNPOD_VERSION
from .remote import AttemptRef
from .object_store import S3Store, bound_record, check_record, terminal_record
from .protocol import canonical, digest, parse
from .files import atomic_write, bounded_read
from .executor import LocalExecutor, worker_environment


class RunpodWorkerConfig(Contract):
    target_id: Identifier
    account_scope: Identifier
    endpoint_id: Identifier
    runtime: RuntimeProfile
    resources: ManagedResources
    source_image: str = Field(pattern=r'^[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}$', max_length=256)
    storage: S3Storage
    allowed_workspaces: tuple[Identifier, ...] = Field(min_length=1, max_length=100)


class WorkerInput(Contract):
    ref: AttemptRef
    request_digest: Digest


def execute_attempt(attempt, directory, *, cancelled=lambda: False):
    """Keep provider I/O outside the independently supervised scientific process."""
    executor = LocalExecutor(directory)
    executor.submit(attempt)
    path = executor.directory(attempt.attempt_id)
    deadline = time.monotonic() + attempt.spec.bundle.request.resources.execution_timeout_ms / 1000 + 30
    next_cancel_check = 0
    while not (path / 'terminal.json').exists():
        if time.monotonic() >= next_cancel_check:
            if cancelled():
                executor.cancel(attempt)
            next_cancel_check = time.monotonic() + 1
        if time.monotonic() >= deadline:
            executor.cancel(attempt)
            # The supervisor's own deadline still applies; do not fabricate stop confirmation.
            raise TimeoutError('MANAGED_SUPERVISOR_UNAVAILABLE')
        time.sleep(.05)
    terminal = bounded_read(path / 'terminal.json', 16384)
    candidate = bounded_read(path / 'candidate.json', attempt.spec.bundle.request.resources.max_output_bytes) if (path / 'candidate.json').exists() else None
    return terminal, candidate


def modal_worker(mount):
    # Volume v2, mounted at an attempt-only subdirectory. No SDK/account credentials needed.
    mount = Path(mount)
    if mount != Path('/mk-data'):
        raise ValueError('FIXED_MODAL_MOUNT_REQUIRED')
    attempt = parse(AttemptRecord, bounded_read(mount / 'request.json'))
    if attempt.spec.managed is None or attempt.spec.managed.adapter != 'modal':
        raise ValueError('MODAL_WORKER_BINDING')
    if digest(attempt.spec) != attempt.execution_digest or digest(attempt.spec.bundle) != attempt.bundle_digest:
        raise ValueError('EXECUTION_BINDING_MISMATCH')
    # The controller sends one Sandbox.create. A retained claim also prevents sequential replay.
    fd = os.open(mount / 'claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as output:
        output.write(canonical(bound_record(attempt, claimed=True)))
        output.flush(); os.fsync(output.fileno())
    subprocess.run(['/usr/bin/sync', '/mk-data'], check=True, timeout=15, env=worker_environment())
    with tempfile.TemporaryDirectory(prefix='mk-managed-') as directory:
        terminal, candidate = execute_attempt(attempt, directory)
    if candidate is not None:
        atomic_write(mount / 'candidate.json', candidate)
        subprocess.run(['/usr/bin/sync', '/mk-data'], check=True, timeout=15, env=worker_environment())
    atomic_write(mount / 'terminal.json', terminal_record(attempt, terminal, candidate))
    subprocess.run(['/usr/bin/sync', '/mk-data'], check=True, timeout=15, env=worker_environment())


def runpod_handler(job, config, *, _store=None):
    payload = parse(WorkerInput, canonical(job['input']))
    provider_id = job.get('id')
    import re
    if not isinstance(provider_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', provider_id):
        raise ValueError('RUNPOD_JOB_ID_INVALID')
    if payload.ref.workspace_id not in config.allowed_workspaces:
        raise PermissionError('WORKSPACE_NOT_ALLOWED')
    store = _store or S3Store(config.storage)
    attempt = parse(AttemptRecord, store.get(payload.ref, 'request'))
    m = attempt.spec.managed
    if (AttemptRef.from_attempt(attempt) != payload.ref or digest(attempt) != payload.request_digest
            or digest(attempt.spec) != attempt.execution_digest or digest(attempt.spec.bundle) != attempt.bundle_digest
            or m is None or m.adapter != 'runpod' or attempt.spec.remote is not None
            or attempt.spec.target != config.target_id or attempt.spec.runtime != config.runtime
            or m.account_scope != config.account_scope or m.destination != config.endpoint_id
            or m.image_identity != config.source_image or m.resources != config.resources
            or m.storage_scope != config.storage.scope):
        raise ValueError('RUNPOD_WORKER_BINDING')
    store.put(attempt, 'receipt', canonical(bound_record(attempt, provider_job_id=provider_id)))
    if not store.create(attempt, 'claim', canonical(bound_record(attempt, claimed=True))):
        return {'attempt_id': attempt.attempt_id, 'state': 'DUPLICATE_DELIVERY_NO_EXECUTION'}
    def cancelled():
        try:
            return check_record(attempt, store.get(attempt, 'cancel', 16384)).get('cancel') is True
        except FileNotFoundError:
            return False
    with tempfile.TemporaryDirectory(prefix='mk-runpod-') as directory:
        terminal, candidate = execute_attempt(attempt, directory, cancelled=cancelled)
    if candidate is not None:
        store.put(attempt, 'candidate', candidate)
    store.put(attempt, 'terminal', terminal_record(attempt, terminal, candidate))
    return {'attempt_id': attempt.attempt_id, 'state': 'DURABLE_OUTPUT_RETAINED'}


def main():
    if sys.argv[1:] == ['modal', '/mk-data']:
        modal_worker('/mk-data')
    elif sys.argv[1:] == ['runpod']:
        if importlib.metadata.version('runpod') != RUNPOD_VERSION:
            raise ValueError('RUNPOD_WORKER_SDK_UNQUALIFIED')
        config = parse(RunpodWorkerConfig, bounded_read('/etc/mathkernel/runpod-worker.json', 65536), 65536)
        if os.environ.get('RUNPOD_REALTIME_PORT', '0') != '0':
            raise ValueError('QUEUE_WORKER_REQUIRED')
        import runpod
        # Registered, non-concurrent handler; no user-defined callable or SDK result deserialization.
        runpod.serverless.start({'handler': lambda job: runpod_handler(job, config)})
    else:
        raise SystemExit('Use the fixed modal /mk-data or runpod worker entrypoint.')


if __name__ == '__main__':
    main()
