"""Fixed worker/supervisor entrypoint. The wire format contains no Python objects."""
from __future__ import annotations
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .executor import process_identity, worker_environment
from .files import atomic_write, bounded_read
from .models import AttemptRecord, RemoteResultEnvelope
from .protocol import canonical, decode, digest, parse, read_frame, write_frame
from .registry import OPERATIONS, execute, runtime_profile


def enable_subreaper():
    # Linux supervisor adopts grandchildren so cleanup can wait for their exit.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot enable process-tree supervision')


def terminate_owned(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        try:
            while os.waitpid(-process.pid, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired('owned process group cleanup', 5)
        time.sleep(.01)


def run():
    import resource
    attempt = parse(AttemptRecord, read_frame(sys.stdin.buffer))
    if digest(attempt.spec) != attempt.execution_digest or digest(attempt.spec.bundle) != attempt.bundle_digest:
        raise ValueError('EXECUTION_BINDING_MISMATCH')
    if runtime_profile(gpu=attempt.spec.runtime.accelerator is not None) != attempt.spec.runtime:
        raise ValueError('RUNTIME_CHANGED')
    ceiling = attempt.spec.bundle.request.resources.max_output_bytes + 4
    resource.setrlimit(resource.RLIMIT_FSIZE, (ceiling, ceiling))
    if attempt.start_deadline_ms is not None and time.time_ns() // 1_000_000 >= attempt.start_deadline_ms:
        raise SystemExit(42)
    operation_started = time.monotonic_ns()
    output = execute(attempt.spec.bundle.request)
    operation_finished = time.monotonic_ns()
    envelope = RemoteResultEnvelope(attempt_id=attempt.attempt_id, workspace_id=attempt.workspace_id,
        execution_digest=attempt.execution_digest, bundle_digest=attempt.bundle_digest,
        operation=attempt.spec.bundle.request.operation,
        output_schema=OPERATIONS[attempt.spec.bundle.request.operation]['output_schema'], output=output,
        worker_claims={'engine': attempt.spec.bundle.request.parameters.engine,
            'timing_observation': {'operation_started_ns': str(operation_started),
                                   'operation_finished_ns': str(operation_finished)}})
    write_frame(sys.stdout.buffer, envelope, ceiling-4)


def supervise(directory, *, slurm=False):
    supervisor_started = time.monotonic_ns()
    directory = Path(directory).resolve()
    attempt = parse(AttemptRecord, bounded_read(directory / 'request.json'))
    if slurm:
        job_id = os.environ.get('MK_SLURM_JOB_ID', '')
        if not job_id.isdecimal() or os.environ.get('MK_SLURM_RESTART_COUNT', '0') != '0':
            raise ValueError('SLURM_INCARNATION_INVALID')
        ack = directory / 'slurm-ack.json'
        if ack.exists() and decode(bounded_read(ack))['job_id'] != job_id:
            raise ValueError('SLURM_JOB_ID_MISMATCH')
    try:
        fd = os.open(directory / 'accepted', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return  # At most one supervisor can own this attempt.
    os.close(fd)
    if slurm:
        atomic_write(directory / 'batch-incarnation.json', canonical({'job_id': job_id, 'attempt_id': attempt.attempt_id}))
    atomic_write(directory / 'owner.json', canonical({'attempt_id': attempt.attempt_id,
        'execution_digest': attempt.execution_digest, 'pid': os.getpid(), 'starttime': process_identity(os.getpid())}))
    process = None
    child_launch = None
    outcome, message = 'FAILED', 'Worker failed before producing a complete envelope.'
    resources = 'RELEASE_CONFIRMED'
    try:
        enable_subreaper()
        if attempt.start_deadline_ms is not None and time.time_ns() // 1_000_000 >= attempt.start_deadline_ms:
            outcome, message = 'EXPIRED', 'Start authorization expired; no mathematical execution.'
        elif (directory / 'cancel').exists():
            outcome, message = 'CANCELLED', 'Cancelled before worker launch.'
        else:
            with open(directory / 'stdout.frame', 'w+b') as output:
                child_launch = time.monotonic_ns()
                process = subprocess.Popen([sys.executable, '-m', 'mathkernel_compute.worker', 'run'],
                    stdin=subprocess.PIPE, stdout=output, stderr=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True, env=worker_environment())
                atomic_write(directory / 'child.json', canonical({'pid': process.pid,
                    'starttime': process_identity(process.pid), 'attempt_id': attempt.attempt_id}))
                write_frame(process.stdin, attempt); process.stdin.close()
                deadline = time.monotonic() + attempt.spec.bundle.request.resources.execution_timeout_ms / 1000
                while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                    if (directory / 'cancel').exists():
                        outcome, message = 'CANCELLED', 'Cancellation requested; owned process group stopped.'
                        break
                    if time.monotonic() >= deadline:
                        outcome, message = 'TIMED_OUT', 'Execution deadline exceeded; no mathematical conclusion.'
                        break
                    if os.fstat(output.fileno()).st_size > attempt.spec.bundle.request.resources.max_output_bytes + 4:
                        raise ValueError('OUTPUT_LIMIT')
                    time.sleep(.02)
                else:
                    exit_status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT).si_status
                    if exit_status == 42:
                        outcome, message = 'EXPIRED', 'Start authorization expired before the operation handler.'
                    elif exit_status == 0:
                        output.seek(0)
                        raw = read_frame(output, attempt.spec.bundle.request.resources.max_output_bytes)
                        if output.read(1):
                            raise ValueError('OUTPUT_INCOMPLETE: trailing frame bytes')
                        # Candidates are validated/admitted only by the local controller.
                        atomic_write(directory / 'candidate.json', raw)
                        outcome, message = 'RECEIVED', 'Candidate retained; not yet mathematically admitted.'
    except Exception as exc:
        message = type(exc).__name__ + ': ' + str(exc)[:500]
    finally:
        cleanup_started = time.monotonic_ns()
        if process is not None:
            try:
                terminate_owned(process)
            except (OSError, subprocess.TimeoutExpired):
                resources = 'CLEANUP_UNKNOWN'
        # Local diagnostic telemetry, not evidence or billing. Absolute monotonic
        # timestamps are strings (nanoseconds can exceed JSON's safe integer range).
        atomic_write(directory / 'timing.json', canonical({'schema': 'mk.local-timing/1',
            'supervisor_started_ns': str(supervisor_started),
            'child_launch_ns': str(child_launch) if child_launch is not None else None,
            'cleanup_started_ns': str(cleanup_started), 'cleanup_finished_ns': str(time.monotonic_ns())}))
        atomic_write(directory / 'terminal.json', canonical({'execution': outcome, 'resources': resources,
            'revision': 2, 'attempt_id': attempt.attempt_id, 'execution_digest': attempt.execution_digest,
            'message': message}))


def verify():
    from .verification import verify_candidate
    payload = decode(read_frame(sys.stdin.buffer))
    if set(payload) != {'attempt', 'candidate', 'report_id'}:
        raise ValueError('VERIFICATION_INPUT_SCHEMA')
    attempt = parse(AttemptRecord, canonical(payload['attempt']))
    candidate = parse(RemoteResultEnvelope, canonical(payload['candidate']))
    report = verify_candidate(attempt, candidate, payload['report_id'])
    write_frame(sys.stdout.buffer, report)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'supervise':
        supervise(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'slurm-supervise':
        supervise(sys.argv[2], slurm=True)
    elif sys.argv[1:] == ['run']:
        run()
    elif sys.argv[1:] == ['verify']:
        verify()
    else:
        raise SystemExit('Use a fixed supervisor/run/verify protocol operation.')


if __name__ == '__main__':
    main()
