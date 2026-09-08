"""Operator-owned target profiles and bounded, data-only OpenSSH transport.

A profile is configuration supplied by the host application, never part of an
agent request. A probe authenticates the host and reports identity; it runs no math.
"""
from __future__ import annotations
import io
import base64
import hashlib
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Literal
from pydantic import Field, field_validator, model_validator
from .files import bounded_read
from .models import Contract, Identifier, Digest, RuntimeProfile, AttemptRecord, SlurmAllocation
from .protocol import canonical, digest, parse, read_frame, write_frame, MAX_MESSAGE


def posix_path(value):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+', value) or '..' in value.split('/') or '//' in value:
        raise ValueError('An absolute, shell-inert POSIX path is required')
    if value == '/':
        raise ValueError('A dedicated path is required')
    return value


class SSHProfile(Contract):
    target_id: Identifier
    adapter: Literal['ssh', 'slurm'] = 'ssh'
    host: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9.:-]{0,252}$')
    port: int = Field(default=22, ge=1, le=65535)
    account: str = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_-]{0,63}$')
    known_hosts: str
    identity_file: str
    worker_executable: str
    worker_config: str
    spool_root: str
    runtime: RuntimeProfile
    # Digest of the separately installed WorkerConfig, obtained through trusted onboarding.
    worker_config_digest: Digest
    allocation: SlurmAllocation | None = None
    ssh_executable: str = 'ssh'
    connect_timeout_s: int = Field(default=10, ge=1, le=30)
    rpc_timeout_s: int = Field(default=30, ge=1, le=120)

    @field_validator('worker_executable', 'worker_config', 'spool_root')
    @classmethod
    def remote_path(cls, value):
        return posix_path(value)

    @field_validator('known_hosts', 'identity_file')
    @classmethod
    def local_path(cls, value):
        # OpenSSH interprets %-tokens and lists inside configuration option values.
        if not Path(value).is_absolute() or any(c.isspace() or c in '%\x00"\'' for c in value):
            raise ValueError('Use an absolute identity/known-hosts path without SSH expansion tokens or whitespace')
        return value

    @field_validator('target_id')
    @classmethod
    def remote_alias(cls, value):
        if value in {'auto', 'local-cpu'}:
            raise ValueError('Reserved target alias')
        return value

    @model_validator(mode='after')
    def allocation_scope(self):
        if (self.adapter == 'slurm') != (self.allocation is not None):
            raise ValueError('Slurm target requires its reviewed allocation shape; SSH has none')
        return self

    @property
    def profile_digest(self):
        return digest({'profile': self.model_dump(mode='json'),
                       'known_hosts_digest': hashlib.sha256(bounded_read(self.known_hosts)).hexdigest()})


class SlurmConfig(Contract):
    """One operator-approved native single-job allocation shape; no caller directives."""
    partition: Identifier
    account: Identifier
    qos: Identifier | None = None
    constraint: Identifier | None = None
    memory_mib: int = Field(default=1024, ge=256, le=65536)
    # Includes interpreter start and supervisor cleanup, beyond the operation deadline.
    walltime_seconds: int = Field(default=180, ge=120, le=900)
    sbatch: str = '/usr/bin/sbatch'
    squeue: str = '/usr/bin/squeue'
    sacct: str = '/usr/bin/sacct'
    scancel: str = '/usr/bin/scancel'

    @field_validator('sbatch', 'squeue', 'sacct', 'scancel')
    @classmethod
    def command_path(cls, value):
        return posix_path(value)


    def allocation(self):
        return SlurmAllocation(partition=self.partition, allocation_account=self.account, qos=self.qos,
            constraint=self.constraint, memory_mib=self.memory_mib, walltime_seconds=self.walltime_seconds)


class WorkerConfig(Contract):
    target_id: Identifier
    adapter: Literal['ssh', 'slurm'] = 'ssh'
    spool_root: str
    allowed_workspaces: tuple[Identifier, ...] = Field(min_length=1, max_length=100)
    max_retained_attempts: int = Field(default=1000, ge=1, le=10000)
    max_active_attempts: int = Field(default=1, ge=1, le=64)
    max_start_delay_ms: int = Field(default=900000, ge=1000, le=900000)
    slurm: SlurmConfig | None = None

    @field_validator('spool_root')
    @classmethod
    def root_path(cls, value):
        return posix_path(value)

    @model_validator(mode='after')
    def scheduler(self):
        if (self.adapter == 'slurm') != (self.slurm is not None):
            raise ValueError('Slurm adapter requires exactly one site-approved Slurm profile')
        return self


class AttemptRef(Contract):
    attempt_id: Identifier
    workspace_id: Identifier
    execution_digest: Digest
    bundle_digest: Digest

    @classmethod
    def from_attempt(cls, attempt):
        return cls(**{k: getattr(attempt, k) for k in cls.model_fields})


class GatewayRequest(Contract):
    protocol: Literal['mk.gateway/1'] = 'mk.gateway/1'
    command: Literal['hello', 'submit', 'observe', 'cancel', 'fetch']
    ref: AttemptRef | None = None
    attempt: AttemptRecord | None = None

    @model_validator(mode='after')
    def shape(self):
        if self.command == 'hello':
            valid = self.ref is None and self.attempt is None
        elif self.command == 'submit':
            valid = self.ref is None and self.attempt is not None
        else:
            valid = self.ref is not None and self.attempt is None
        if not valid:
            raise ValueError('GATEWAY_REQUEST_SHAPE')
        return self


class RemoteUnavailable(OSError):
    """Transport/protocol unavailable. Never evidence that an allocation was released."""


def run_bounded(argv, data=b'', *, timeout=30, limit=MAX_MESSAGE + 4, env=None):
    """Drain while enforcing a byte ceiling, including for a hostile remote endpoint."""
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=env, close_fds=True)
    output = bytearray()
    overflow = threading.Event()
    def drain():
        try:
            while chunk := process.stdout.read(4096):
                if len(output) + len(chunk) > limit:
                    overflow.set()
                    process.kill()
                    return
                output.extend(chunk)
        finally:
            process.stdout.close()
    def feed():
        try:
            process.stdin.write(data)
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            process.stdin.close()
    reader = threading.Thread(target=drain, daemon=True)
    writer = threading.Thread(target=feed, daemon=True)
    reader.start(); writer.start()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()
        raise RemoteUnavailable('REMOTE_TIMEOUT') from None
    finally:
        reader.join(timeout=2); writer.join(timeout=2)
    if overflow.is_set() or reader.is_alive():
        raise RemoteUnavailable('REMOTE_OUTPUT_LIMIT')
    if process.returncode:
        raise RemoteUnavailable('REMOTE_COMMAND_FAILED')
    return bytes(output)


class SSHExecutor:
    def __init__(self, profile):
        self.profile = profile

    def argv(self):
        p = self.profile
        # -F none prevents ambient ProxyCommand/LocalCommand/forwarding configuration.
        # The two remote words are operator-pinned, strictly shell-inert paths.
        return [p.ssh_executable, '-F', 'none', '-T', '-a', '-k',
            '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
            '-oUserKnownHostsFile=' + p.known_hosts, '-oGlobalKnownHostsFile=none',
            '-oIdentitiesOnly=yes', '-oIdentityAgent=none', '-oClearAllForwardings=yes',
            '-oPermitLocalCommand=no', '-oRequestTTY=no', '-oUpdateHostKeys=no',
            '-oControlMaster=no', '-oControlPath=none', '-oCanonicalizeHostname=no',
            '-oConnectTimeout=' + str(p.connect_timeout_s), '-i', p.identity_file,
            '-p', str(p.port), '-l', p.account, p.host, p.worker_executable + ' ' + p.worker_config]

    def rpc(self, request):
        stream = io.BytesIO(); write_frame(stream, request)
        reply_limit = 2 * MAX_MESSAGE if request.command == 'fetch' else MAX_MESSAGE
        raw = run_bounded(self.argv(), stream.getvalue(), timeout=self.profile.rpc_timeout_s, limit=reply_limit + 4)
        stream = io.BytesIO(raw)
        from .protocol import decode
        try:
            response = decode(read_frame(stream, reply_limit), reply_limit)
            if stream.read(1) or not isinstance(response, dict) or set(response) != {'ok', 'value'}:
                raise ValueError('GATEWAY_RESPONSE_SHAPE')
            if response['ok'] is not True:
                if response['value'] == 'NOT_FOUND':
                    raise FileNotFoundError('REMOTE_ARTIFACT_UNAVAILABLE')
                raise RemoteUnavailable('REMOTE_REQUEST_REFUSED')
            return response['value']
        except (ValueError, EOFError, KeyError) as exc:
            raise RemoteUnavailable('REMOTE_PROTOCOL_INVALID') from exc

    def probe(self):
        result = self.rpc(GatewayRequest(command='hello'))
        expected = {'protocol': 'mk.gateway/1', 'target_id': self.profile.target_id,
            'adapter': self.profile.adapter, 'spool_root': self.profile.spool_root,
            'config_digest': self.profile.worker_config_digest,
            'allocation': self.profile.allocation.model_dump(mode='json') if self.profile.allocation else None,
            'runtime': self.profile.runtime.model_dump(mode='json')}
        if result != expected:
            raise RemoteUnavailable('REMOTE_PROFILE_MISMATCH')
        return result

    def submit(self, attempt):
        if attempt.spec.remote is None or attempt.spec.remote.target_profile_digest != self.profile.profile_digest:
            raise PermissionError('TARGET_PROFILE_CHANGED')
        self.probe()  # No inputs disclosed until the authenticated runtime/configuration matches.
        value = self.rpc(GatewayRequest(command='submit', attempt=attempt))
        if (not isinstance(value, dict) or set(value) != {'accepted', 'attempt_id'}
                or value['accepted'] is not True or value['attempt_id'] != attempt.attempt_id):
            raise RemoteUnavailable('REMOTE_ACCEPTANCE_IDENTITY_INVALID')

    def observe(self, attempt):
        return self.rpc(GatewayRequest(command='observe', ref=AttemptRef.from_attempt(attempt)))

    def cancel(self, attempt):
        value = self.rpc(GatewayRequest(command='cancel', ref=AttemptRef.from_attempt(attempt)))
        if not isinstance(value, dict) or set(value) != {'cancel_requested'} or value['cancel_requested'] is not True:
            raise RemoteUnavailable('REMOTE_CANCELLATION_ACK_INVALID')

    def fetch(self, attempt):
        # A fixed candidate DTO, never a worker URL, filename, archive, pickle or MathResult.
        value = self.rpc(GatewayRequest(command='fetch', ref=AttemptRef.from_attempt(attempt)))
        if not isinstance(value, dict) or set(value) != {'candidate_base64'} or not isinstance(value['candidate_base64'], str):
            raise RemoteUnavailable('REMOTE_ARTIFACT_ENCODING')
        try:
            raw = base64.b64decode(value['candidate_base64'], validate=True)
        except ValueError as exc:
            raise RemoteUnavailable('REMOTE_ARTIFACT_ENCODING') from exc
        if len(raw) > attempt.spec.bundle.request.resources.max_output_bytes:
            raise RemoteUnavailable('REMOTE_ARTIFACT_LIMIT')
        return raw
