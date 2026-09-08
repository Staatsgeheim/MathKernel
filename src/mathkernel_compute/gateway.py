"""Installed, fixed-command Linux gateway. One bounded request per SSH connection.

The operator creates the private spool and installs the config out of band. This
entrypoint cannot install services, select handlers, execute shell input, or power
off a host. Its lock serializes acceptance across independent SSH sessions.
"""
from __future__ import annotations
from contextlib import contextmanager
import os
import base64
from pathlib import Path
import re
import stat
import sys
import time
from .executor import LocalExecutor
from .files import atomic_write, bounded_read
from .models import AttemptRecord
from .protocol import canonical, decode, digest, parse, read_frame, write_frame, MAX_MESSAGE
from .registry import runtime_profile
from .remote import AttemptRef, GatewayRequest, WorkerConfig


class Gateway:
    def __init__(self, config):
        if sys.platform != 'linux':
            raise ValueError('Linux remote supervisors are required')
        self.config = config
        self.root = Path(config.spool_root)
        if self.root.resolve() != self.root or not self.root.is_dir():
            raise ValueError('SPOOL_MUST_BE_PRECREATED_WITHOUT_SYMLINKS')
        info = self.root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('SPOOL_MUST_BE_PRIVATE_AND_OWNED')

    @contextmanager
    def locked(self):
        import fcntl
        fd = os.open(self.root / 'gateway.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def directory(self, ref):
        if ref.workspace_id not in self.config.allowed_workspaces:
            raise PermissionError('WORKSPACE_DENIED')
        if not re.fullmatch(r'attempt_[a-f0-9]{32}', ref.attempt_id):
            raise ValueError('ATTEMPT_ID_INVALID')
        workspace = self.root / ref.workspace_id
        directory = workspace / ref.attempt_id
        if workspace.is_symlink() or directory.is_symlink():
            raise ValueError('SPOOL_PATH_DENIED')
        return directory

    def executor(self, ref):
        root = self.directory(ref).parent
        if self.config.adapter == 'slurm':
            from .slurm import SlurmExecutor
            return SlurmExecutor(root, self.config.slurm)
        return LocalExecutor(root)

    def load(self, ref):
        attempt = parse(AttemptRecord, bounded_read(self.directory(ref) / 'intent.json'))
        if AttemptRef.from_attempt(attempt) != ref:
            raise PermissionError('ATTEMPT_IDENTITY_CONFLICT')
        return attempt

    def capacity(self):
        total = active = 0
        for workspace in self.root.iterdir():
            if not workspace.is_dir() or workspace.is_symlink():
                continue
            for directory in workspace.iterdir():
                if not directory.is_dir() or directory.is_symlink():
                    continue
                total += 1
                try:
                    attempt = parse(AttemptRecord, bounded_read(directory / 'intent.json'))
                    observation = self.executor(AttemptRef.from_attempt(attempt)).observe(attempt)
                    active += observation['resources'] not in {'RELEASE_CONFIRMED', 'NOT_OWNED'}
                except (OSError, ValueError, KeyError):
                    active += 1
        if total >= self.config.max_retained_attempts or active >= self.config.max_active_attempts:
            raise PermissionError('REMOTE_CAPACITY')

    def submit(self, attempt):
        ref = AttemptRef.from_attempt(attempt)
        directory = self.directory(ref)
        intent = directory / 'intent.json'
        if intent.exists():
            if bounded_read(intent) != canonical(attempt):
                raise PermissionError('ATTEMPT_CONFLICT')
            return {'accepted': True, 'attempt_id': attempt.attempt_id}  # Never relaunch an ambiguous intent.
        if (attempt.spec.remote is None or attempt.spec.remote.adapter != self.config.adapter
                or attempt.spec.target != self.config.target_id
                or attempt.spec.remote.allocation != (self.config.slurm.allocation() if self.config.slurm else None)
                or attempt.spec.bundle.classification != 'explicit_export'
                or attempt.execution_digest != digest(attempt.spec)
                or attempt.bundle_digest != digest(attempt.spec.bundle)
                or attempt.spec.bundle_digest != attempt.bundle_digest):
            raise PermissionError('EXPORT_OR_EXECUTION_BINDING_INVALID')
        now = time.time_ns() // 1_000_000
        if attempt.start_deadline_ms is None or not now < attempt.start_deadline_ms <= now + self.config.max_start_delay_ms:
            raise PermissionError('START_AUTHORIZATION_EXPIRED_OR_UNBOUNDED')
        if attempt.spec.runtime != runtime_profile():
            raise PermissionError('RUNTIME_CHANGED')
        self.capacity()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write(intent, canonical(attempt))  # Durable intent precedes every process/scheduler effect.
        self.executor(ref).submit(attempt)
        return {'accepted': True, 'attempt_id': attempt.attempt_id}

    def dispatch(self, request):
        if request.command == 'hello':
            if self.config.adapter == 'slurm':
                # Verify required CLI flag profiles without allocating a job or importing an operation.
                from .remote import run_bounded
                from .slurm import scheduler_environment
                required = {'sbatch': ('--parsable', '--no-requeue', '--export'),
                            'squeue': ('--format', '--local', '--noheader'),
                            'sacct': ('--parsable2', '--duplicates', '--format', '--local'),
                            'scancel': ('--user', '--name')}
                for name, flags in required.items():
                    run_bounded([getattr(self.config.slurm, name), '--version'], timeout=10,
                                limit=4096, env=scheduler_environment())
                    help_text = run_bounded([getattr(self.config.slurm, name), '--help'], timeout=10,
                                            limit=65536, env=scheduler_environment())
                    if any(flag.encode() not in help_text for flag in flags):
                        raise ValueError('SLURM_CLI_PROFILE_UNSUPPORTED')
            return {'protocol': 'mk.gateway/1', 'target_id': self.config.target_id,
                'adapter': self.config.adapter, 'spool_root': self.config.spool_root,
                'config_digest': digest(self.config),
                'allocation': self.config.slurm.allocation().model_dump(mode='json') if self.config.slurm else None,
                'runtime': runtime_profile().model_dump(mode='json')}
        with self.locked():
            if request.command == 'submit':
                return self.submit(request.attempt)
            attempt = self.load(request.ref)
            executor = self.executor(request.ref)
            if request.command == 'observe':
                observed = executor.observe(attempt)
                return dict(observed, attempt_id=attempt.attempt_id, execution_digest=attempt.execution_digest)
            if request.command == 'cancel':
                executor.cancel(attempt)
                return {'cancel_requested': True}
            return {'candidate_base64': base64.b64encode(executor.fetch(attempt)).decode('ascii')}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('A single installed worker configuration path is required.')
    try:
        config = parse(WorkerConfig, bounded_read(sys.argv[1], 65536), 65536)
        request = parse(GatewayRequest, read_frame(sys.stdin.buffer))
        if sys.stdin.buffer.read(1):
            raise ValueError('TRAILING_INPUT')
        value = Gateway(config).dispatch(request)
        write_frame(sys.stdout.buffer, {'ok': True, 'value': value}, 2 * MAX_MESSAGE if request.command == 'fetch' else MAX_MESSAGE)
    except FileNotFoundError:
        write_frame(sys.stdout.buffer, {'ok': False, 'value': 'NOT_FOUND'})
    except Exception:
        # No provider text, traceback, path, credential or partial candidate in error responses.
        write_frame(sys.stdout.buffer, {'ok': False, 'value': 'REFUSED'})


if __name__ == '__main__':
    main()
