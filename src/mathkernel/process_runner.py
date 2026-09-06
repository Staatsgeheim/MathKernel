# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded reusable subprocesses for trusted Python computational callables.

Each leased worker handles one request. A timeout kills that worker's process
(tree on Windows, process group on POSIX) and frees its slot, so timed-out work
cannot starve a later request. Callable serialization is PRIVATE IPC: never
accept it from an MCP request, a saved artifact, or another untrusted source.
"""
from __future__ import annotations
import atexit
from collections import deque
import math
import os
import queue
import signal
import struct
import subprocess
import sys
import threading
import time

import cloudpickle

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 15.0
_IN_WORKER = False


def in_worker() -> bool:
    return _IN_WORKER


def read_frame(stream) -> bytes:
    header = stream.read(8)
    if not header:
        raise EOFError("solver worker closed its output")
    if len(header) != 8:
        raise ValueError("incomplete solver frame")
    size = struct.unpack("!Q", header)[0]
    if size > MAX_MESSAGE_BYTES:
        raise ValueError("isolated solver response exceeded output limit")
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("incomplete solver payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_frame(stream, payload: bytes) -> None:
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("isolated solver message exceeded resource limit")
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def _descendant_pids(parent_pid: int) -> list[int]:
    """POSIX process ancestry includes children that started a new session.

    Native candidate solvers intentionally use a fresh process group. Killing
    only the outer lease's group would miss those children on timeout.
    """
    try:
        result = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True,
                                text=True, timeout=5, check=True)
        children = {}
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) != 2: continue
            pid, ppid = map(int, fields)
            children.setdefault(ppid, []).append(pid)
        descendants, pending = [], [parent_pid]
        seen = {parent_pid}
        while pending:
            for pid in children.get(pending.pop(), []):
                if pid not in seen:
                    seen.add(pid); descendants.append(pid); pending.append(pid)
        return descendants
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def terminate_process_tree(process) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        # /T includes solver descendants, unlike Popen.kill on Windows.
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        # Freeze the root while discovering descendants, preventing it from
        # launching a new native candidate between enumeration and termination.
        try:
            os.kill(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass
        for pid in reversed(_descendant_pids(process.pid)):
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class _Worker:
    def __init__(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([p for p in sys.path if p] + env.get("PYTHONPATH", "").split(os.pathsep)))
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        self.process = subprocess.Popen([sys.executable, "-m", "mathkernel.call_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=env, start_new_session=os.name != "nt")
        self.commands = queue.Queue()
        self.responses = queue.Queue()
        self.thread = threading.Thread(target=self._io, daemon=True, name="mathkernel-worker-io")
        self.thread.start()
        try:
            ready = self.responses.get(timeout=STARTUP_TIMEOUT_SECONDS)
            if isinstance(ready, BaseException):
                raise ready
            if cloudpickle.loads(ready) != {"ready": True}:
                raise ValueError("invalid solver startup response")
        except BaseException:
            self.close()
            raise

    def _io(self):
        try:
            self.responses.put(read_frame(self.process.stdout))
            while True:
                payload = self.commands.get()
                if payload is None:
                    return
                write_frame(self.process.stdin, payload)
                self.responses.put(read_frame(self.process.stdout))
        except BaseException as exc:
            self.responses.put(exc)

    def call(self, payload: bytes, timeout: float):
        self.commands.put(payload)
        response = self.responses.get(timeout=timeout)
        if isinstance(response, BaseException):
            raise response
        return cloudpickle.loads(response)

    def close(self):
        terminate_process_tree(self.process)
        self.commands.put(None)
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class _Pool:
    def __init__(self, maximum=4):
        self.maximum = maximum
        self.condition = threading.Condition()
        self.idle = deque()
        self.total = 0
        self.live = set()

    def acquire(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.idle and self.total >= self.maximum:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self.condition.wait(remaining)
            if self.idle:
                return self.idle.pop()  # Keep sequential immutable-cache workloads on the warmest worker.
            self.total += 1
        try:
            worker = _Worker()
        except BaseException:
            with self.condition:
                self.total -= 1
                self.condition.notify()
            raise
        with self.condition:
            self.live.add(worker)
        return worker

    def release(self, worker, healthy):
        if not healthy:
            worker.close()
        with self.condition:
            if healthy:
                self.idle.append(worker)
            else:
                self.live.discard(worker)
                self.total -= 1
            self.condition.notify()

    def shutdown(self):
        with self.condition:
            workers = list(self.live)
            self.live.clear()
            self.idle.clear()
            self.total = 0
        for worker in workers:
            worker.close()


_POOL = _Pool()
atexit.register(_POOL.shutdown)


def run_isolated_callable(fn, timeout_seconds, *args, **kwargs):
    from .engines import SolverTimeoutError, IsolatedSolverError
    if timeout_seconds is None or (not isinstance(timeout_seconds, bool) and timeout_seconds == 0):
        return fn(*args, **kwargs)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite (or None to disable)")
    if _IN_WORKER:
        # The outer process lease is already hard-bounded; no nested process
        # proliferation or thread-pool deadlock is necessary.
        return fn(*args, **kwargs)
    name = getattr(fn, "__qualname__", type(fn).__name__)
    try:
        payload = cloudpickle.dumps((fn, args, kwargs), protocol=5)
    except Exception as exc:
        raise IsolatedSolverError(f"{name} cannot be serialized for isolated execution: {exc}") from exc
    if len(payload) > MAX_MESSAGE_BYTES:
        raise IsolatedSolverError("isolated solver request exceeded input limit")
    worker = None
    healthy = False
    try:
        worker = _POOL.acquire(timeout_seconds)
        response = worker.call(payload, timeout_seconds)
        healthy = True
    except queue.Empty:
        raise SolverTimeoutError(f"{name} exceeded solver_timeout_seconds={timeout_seconds}; isolated worker terminated or queue lease expired") from None
    except (EOFError, OSError, ValueError) as exc:
        raise IsolatedSolverError(f"isolated solver transport failed: {exc}") from exc
    finally:
        if worker is not None:
            _POOL.release(worker, healthy)
    if response.get("ok"):
        return response["value"]
    exc = response.get("exception")
    if isinstance(exc, Exception):
        raise exc
    raise IsolatedSolverError(response.get("message", "isolated solver failed"))
