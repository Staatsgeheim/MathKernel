"""Operator-installed managed targets and a bounded, data-only provider bridge.

Profiles are host configuration. They never come from a mathematical request.
No SDK is imported, credential read, or network connection opened by discovery.
"""
from __future__ import annotations
import base64
import io
import os
import sys
from typing import Annotated, Literal
from pydantic import Field, field_validator, model_validator
from .models import (Contract, Identifier, Digest, RuntimeProfile, ManagedQuote,
                     ManagedResources, ManagedBinding, AttemptRecord, ModalOptions, RunpodOptions)
from .protocol import canonical, decode, digest, parse, read_frame, write_frame
from .remote import RemoteUnavailable, run_bounded
from .executor import worker_environment

MODAL_VERSION = '1.5.5'
BOTO3_VERSION = '1.43.90'
RUNPOD_VERSION = '1.12.0'


class ManagedNotSubmitted(RemoteUnavailable):
    """The local bridge did not start an invocation call. Storage charges may exist."""


class S3Storage(Contract):
    bucket: str = Field(pattern=r'^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$')
    region: str = Field(pattern=r'^[a-z]{2}-[a-z]+-[1-9]$')
    prefix: Identifier
    credential_env_prefix: str = Field(pattern=r'^MK_[A-Z0-9_]{1,64}$')

    @property
    def scope(self):
        return f's3://{self.bucket}/{self.prefix} ({self.region})'


class ManagedProfile(Contract):
    target_id: Identifier
    account_scope: Identifier
    budget_id: Identifier
    runtime: RuntimeProfile
    resources: ManagedResources
    quote: ManagedQuote
    credential_env_prefix: str = Field(pattern=r'^MK_[A-Z0-9_]{1,64}$')
    # Worker executable is fixed by the image contract, never a supplied command.
    source_image: str = Field(pattern=r'^[A-Za-z0-9._:/-]+@sha256:[a-f0-9]{64}$', max_length=256)

    @field_validator('target_id')
    @classmethod
    def alias(cls, value):
        if value in {'auto', 'local-cpu'}:
            raise ValueError('Reserved target alias')
        return value

    @model_validator(mode='after')
    def accelerator_scope(self):
        if bool(self.resources.gpu) != bool(self.runtime.accelerator):
            raise ValueError('GPU profile requires a pinned accelerator runtime')
        if self.runtime.platform != 'linux':
            raise ValueError('Managed workers require the Linux supervisor')
        return self

    @property
    def profile_digest(self):
        return digest(self)

    def binding(self, verifier_runtime, budget):
        if budget.budget_id != self.budget_id or budget.account_scope != self.account_scope:
            raise ValueError('BUDGET_SCOPE_MISMATCH')
        return ManagedBinding(adapter=self.adapter, target_profile_digest=self.profile_digest,
            verifier_runtime=verifier_runtime, account_scope=self.account_scope,
            destination=self.destination, image_identity=self.image_identity,
            storage_scope=self.storage_scope, source_image=self.source_image,
            provider_options=self.provider_options, resources=self.resources, budget=budget, quote=self.quote)


class ModalProfile(ManagedProfile):
    adapter: Literal['modal'] = 'modal'
    app_name: Identifier
    app_id: str = Field(pattern=r'^ap-[A-Za-z0-9]+$')
    environment: Identifier
    image_id: str = Field(pattern=r'^im-[A-Za-z0-9]+$')
    volume_name: Identifier
    volume_id: str = Field(pattern=r'^vo-[A-Za-z0-9]+$')
    region: str = Field(pattern=r'^[a-z0-9-]{1,40}$')
    sdk_version: Literal['1.5.5'] = MODAL_VERSION
    volume_version: Literal[2] = 2

    @model_validator(mode='after')
    def isolated(self):
        if self.resources.network != 'blocked':
            raise ValueError('Modal pilot requires block_network=True')
        return self

    @property
    def destination(self):
        return f'{self.app_id}/{self.environment}/{self.region}'

    @property
    def image_identity(self):
        return self.image_id

    @property
    def storage_scope(self):
        return f'modal-volume-v2:{self.volume_id}'

    @property
    def provider_options(self):
        return ModalOptions(app_name=self.app_name, environment=self.environment, region=self.region,
                            sdk_version=self.sdk_version, volume_version=2)


class RunpodProfile(ManagedProfile):
    adapter: Literal['runpod'] = 'runpod'
    endpoint_id: Identifier
    deployment_digest: Digest
    storage: S3Storage
    workers_min: int = Field(ge=0, le=16)
    workers_max: int = Field(ge=1, le=16)
    idle_timeout_seconds: int = Field(ge=1, le=600)

    @model_validator(mode='after')
    def existing_endpoint(self):
        if self.resources.network != 'broker-only' or self.workers_min > self.workers_max:
            raise ValueError('Runpod requires an existing bounded endpoint and broker artifact egress')
        return self

    @property
    def destination(self):
        return self.endpoint_id

    @property
    def image_identity(self):
        return self.source_image

    @property
    def storage_scope(self):
        return self.storage.scope

    @property
    def provider_options(self):
        return RunpodOptions(deployment_digest=self.deployment_digest, workers_min=self.workers_min,
                             workers_max=self.workers_max, idle_timeout_seconds=self.idle_timeout_seconds)


Profile = Annotated[ModalProfile | RunpodProfile, Field(discriminator='adapter')]


class ManagedObservation(Contract):
    execution: Literal['SUBMISSION_UNKNOWN', 'QUEUED', 'RUNNING', 'RECEIVED', 'FAILED',
                       'TIMED_OUT', 'CANCELLED', 'LOST', 'EXPIRED']
    resources: Literal['ALLOCATION_INTENT', 'ACTIVE', 'NOT_OWNED', 'RELEASE_CONFIRMED', 'CLEANUP_UNKNOWN']
    message: str = Field(max_length=2000)


class BridgeCall(Contract):
    protocol: Literal['mk.managed-bridge/1'] = 'mk.managed-bridge/1'
    action: Literal['probe', 'submit', 'observe', 'cancel', 'fetch']
    profile: Profile
    attempt: AttemptRecord | None = None
    handle: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,160}$')

    @model_validator(mode='after')
    def bound(self):
        if (self.action == 'probe') != (self.attempt is None):
            raise ValueError('BRIDGE_CALL_SHAPE')
        if self.attempt:
            a = self.attempt
            if (a.spec.managed is None or a.spec.remote is not None
                    or a.spec.target != self.profile.target_id
                    or a.spec.managed.target_profile_digest != self.profile.profile_digest
                    or a.spec.runtime != self.profile.runtime
                    or digest(a.spec) != a.execution_digest or digest(a.spec.bundle) != a.bundle_digest):
                raise ValueError('MANAGED_ATTEMPT_BINDING')
            m = a.spec.managed
            if m != self.profile.binding(m.verifier_runtime, m.budget):
                raise ValueError('MANAGED_PROFILE_BINDING')
        return self


def parse_profile(raw):
    # The discriminator is validated by a strict contract, not an import/module name.
    return parse(BridgeCall, canonical({'action': 'probe', 'profile': decode(raw)})).profile


def bridge_environment(profile):
    env = worker_environment()
    env.update(MODAL_SANDBOX_V2='false', MODAL_CONFIG_PATH='/dev/null',
               AWS_EC2_METADATA_DISABLED='true')
    names = ([profile.credential_env_prefix + '_TOKEN_ID', profile.credential_env_prefix + '_TOKEN_SECRET']
             if profile.adapter == 'modal' else [profile.credential_env_prefix + '_API_KEY'])
    if profile.adapter == 'runpod':
        names += [profile.storage.credential_env_prefix + suffix for suffix in
                  ('_ACCESS_KEY_ID', '_SECRET_ACCESS_KEY')]
        token = profile.storage.credential_env_prefix + '_SESSION_TOKEN'
        if os.environ.get(token):
            names.append(token)
    for name in names:
        value = os.environ.get(name)
        if not value or len(value) > 8192:
            raise RemoteUnavailable('MANAGED_CREDENTIALS_UNAVAILABLE')
        env[name] = value
    return env


def bridge_call(call):
    if sys.platform != 'linux':
        error = ManagedNotSubmitted if call.action == 'submit' else RemoteUnavailable
        raise error('Managed SDK bridge is qualified only for Linux resource limits')
    try:
        env = bridge_environment(call.profile)
    except RemoteUnavailable:
        if call.action == 'submit':
            raise ManagedNotSubmitted('MANAGED_CREDENTIALS_UNAVAILABLE') from None
        raise
    stream = io.BytesIO()
    write_frame(stream, call)
    raw = run_bounded([sys.executable, '-m', 'mathkernel_compute.provider_bridge'],
        stream.getvalue(), env=env, timeout=45, limit=1_500_000)
    from .protocol import decode
    response = decode(read_frame(io.BytesIO(raw), 1_499_996), 1_499_996)
    if not isinstance(response, dict) or response.get('protocol') != 'mk.managed-bridge/1':
        raise RemoteUnavailable('MANAGED_PROTOCOL_INVALID')
    if response.get('missing') is True:
        raise FileNotFoundError('Durable output not available')
    if response.get('error'):
        if call.action == 'submit' and response.get('not_submitted') is True:
            raise ManagedNotSubmitted('MANAGED_INVOCATION_NOT_STARTED')
        raise RemoteUnavailable('MANAGED_PROVIDER_UNAVAILABLE')  # Provider errors may contain secrets.
    return response


class ManagedExecutor:
    def __init__(self, profile, journal, *, _call=bridge_call):
        self.profile, self.journal, self.call = profile, journal, _call

    def probe(self):
        return self.call(BridgeCall(action='probe', profile=self.profile))

    def _request(self, action, attempt):
        try:
            record = self.journal.get('provider_handles', attempt.attempt_id)
            if record['execution_digest'] != attempt.execution_digest or record['profile_digest'] != self.profile.profile_digest:
                raise ValueError('PROVIDER_HANDLE_BINDING')
            handle = record['handle']
        except KeyError:
            handle = None
        response = self.call(BridgeCall(action=action, profile=self.profile, attempt=attempt, handle=handle))
        if response.get('attempt_id') != attempt.attempt_id or response.get('execution_digest') != attempt.execution_digest:
            raise ValueError('MANAGED_RESPONSE_BINDING')
        if response.get('handle') is not None:
            # Validate provider IDs before journaling them or using them in URLs.
            BridgeCall(action=action, profile=self.profile, attempt=attempt, handle=response['handle'])
            if handle is not None and handle != response['handle']:
                raise ValueError('PROVIDER_HANDLE_CONFLICT')
            with self.journal.transaction():
                scope = digest({'adapter': self.profile.adapter, 'account_scope': self.profile.account_scope,
                    'destination': self.profile.app_id if self.profile.adapter == 'modal' else self.profile.endpoint_id})
                self.journal.put('provider_handles', attempt.attempt_id, dict(handle=response['handle'], adapter_scope=scope,
                    execution_digest=attempt.execution_digest, profile_digest=self.profile.profile_digest))
        return response

    def submit(self, attempt):
        self._request('submit', attempt)

    def observe(self, attempt):
        response = self._request('observe', attempt)
        observation = parse(ManagedObservation, canonical(response['observation']), 16384).model_dump(mode='json')
        # Persist a local monotonic revision; provider timestamps are not versions.
        try:
            old = self.journal.get('provider_observations', attempt.attempt_id)
        except KeyError:
            old = {'revision': -1, 'observation': None}
        revision = old['revision'] + int(old['observation'] != observation)
        with self.journal.transaction():
            self.journal.put('provider_observations', attempt.attempt_id,
                             {'revision': revision, 'observation': observation})
        return dict(observation, revision=revision, attempt_id=attempt.attempt_id,
                    execution_digest=attempt.execution_digest)

    def cancel(self, attempt):
        self._request('cancel', attempt)

    def fetch(self, attempt):
        response = self._request('fetch', attempt)
        raw = base64.b64decode(response['candidate_b64'], validate=True)
        if len(raw) > attempt.spec.bundle.request.resources.max_output_bytes:
            raise ValueError('MANAGED_OUTPUT_LIMIT')
        return raw
