"""Durable local submission with fixed commands and generation-checked observations."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from .files import atomic_write, bounded_read
from .protocol import canonical, decode


def worker_environment():
    # No credentials, proxies, inherited kernel settings or arbitrary worker options.
    return {'PATH': os.defpath, 'PYTHONPATH': os.pathsep.join(p for p in sys.path if p),
            'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
            'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        fields = raw[raw.rindex(')') + 2:].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # starttime, field 22 (PID reuse fence)
    except (FileNotFoundError, ProcessLookupError):
        return None


class LocalExecutor:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.children = []

    def directory(self, attempt_id):
        if not attempt_id.startswith('attempt_') or not attempt_id[8:].isalnum():
            raise ValueError('Invalid attempt ID')
        return self.root / attempt_id

    def submit(self, attempt):
        directory = self.directory(attempt.attempt_id)
        directory.mkdir(mode=0o700, exist_ok=True)
        request = directory / 'request.json'
        raw = canonical(attempt)
        if request.exists():
            if bounded_read(request) != raw:
                raise ValueError('ATTEMPT_CONFLICT')
            # Submission outcome can be unknown. Never start a replacement here.
            return
        atomic_write(request, raw)
        process = subprocess.Popen([sys.executable, '-m', 'mathkernel_compute.worker', 'supervise', str(directory)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, env=worker_environment())
        self.children = [p for p in self.children if p.poll() is None]
        self.children.append(process)
        threading.Thread(target=process.wait, daemon=True).start()

    def observe(self, attempt):
        directory = self.directory(attempt.attempt_id)
        terminal = directory / 'terminal.json'
        if terminal.exists():
            return decode(bounded_read(terminal))
        marker = directory / 'owner.json'
        if marker.exists():
            owner = decode(bounded_read(marker))
            if owner['attempt_id'] != attempt.attempt_id or owner['execution_digest'] != attempt.execution_digest:
                raise ValueError('OWNER_IDENTITY_CONFLICT')
            if process_identity(owner['pid']) == owner['starttime']:
                return {'execution': 'RUNNING', 'resources': 'ACTIVE', 'revision': 1}
            if terminal.exists():
                return decode(bounded_read(terminal))  # Supervisor may have exited during the first observation.
            return {'execution': 'LOST', 'resources': 'CLEANUP_UNKNOWN', 'revision': 2,
                    'message': 'Supervisor unavailable; retained process identity requires reconciliation.'}
        return {'execution': 'SUBMISSION_UNKNOWN', 'resources': 'ALLOCATION_INTENT', 'revision': 0}

    def cancel(self, attempt):
        # The recorded supervisor performs the kill; no PID from a worker is executed by the controller.
        directory = self.directory(attempt.attempt_id)
        atomic_write(directory / 'cancel', canonical({'attempt_id': attempt.attempt_id,
                                                       'execution_digest': attempt.execution_digest}))

    def fetch(self, attempt):
        return bounded_read(self.directory(attempt.attempt_id) / 'candidate.json',
                            attempt.spec.bundle.request.resources.max_output_bytes)
