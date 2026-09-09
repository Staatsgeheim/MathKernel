"""Pinned Modal Sandbox control API. Never calls a remote Python function."""
from __future__ import annotations
import base64
import importlib.metadata
import re
from .managed import MODAL_VERSION
from .object_store import ModalVolumeStore, attempt_key, check_record
from .protocol import canonical


def tags(attempt):
    return {'mk_workspace': attempt.workspace_id, 'mk_attempt': attempt.attempt_id,
            'mk_execution': attempt.execution_digest,
            'mk_profile': attempt.spec.managed.target_profile_digest}


def stable_sandbox_id(value):
    # Modal 1.5.5 routes all non-v1-shaped IDs to its beta v2 backend.
    if not isinstance(value, str) or not re.fullmatch(r'sb-[A-Za-z0-9]{22}', value):
        raise ValueError('MODAL_SANDBOX_BACKEND_UNQUALIFIED')
    return value


class ModalAdapter:
    def __init__(self, profile, *, _modal=None, _client=None):
        self.profile = profile
        if _modal is None:
            if importlib.metadata.version('modal') != MODAL_VERSION:
                raise ValueError('MODAL_SDK_VERSION_UNQUALIFIED')
            import modal as _modal
        self.modal = _modal
        if _client is None:
            import os
            prefix = profile.credential_env_prefix
            _client = _modal.Client.from_credentials(os.environ[prefix + '_TOKEN_ID'],
                                                      os.environ[prefix + '_TOKEN_SECRET'])
        self.client = _client
        self.app = _modal.App.lookup(profile.app_name, client=_client,
            environment_name=profile.environment, create_if_missing=False)
        if self.app.app_id != profile.app_id:
            raise ValueError('MODAL_APP_CHANGED')
        self.volume = _modal.Volume.from_name(profile.volume_name, client=_client,
            environment_name=profile.environment, create_if_missing=False, version=2)
        self.volume.hydrate()
        if self.volume.object_id != profile.volume_id:
            raise ValueError('MODAL_VOLUME_CHANGED')
        self.store = ModalVolumeStore(self.volume, missing_errors=(FileNotFoundError, _modal.exception.NotFoundError))

    def probe(self):
        return {'adapter': 'modal', 'app_id': self.app.app_id, 'volume_id': self.volume.object_id,
                'volume_version': 2, 'sdk_version': MODAL_VERSION,
                'qualification': 'experimental-not-live-qualified'}

    def prepare(self, attempt):
        self.store.put(attempt, 'request', canonical(attempt))

    def submit(self, attempt):
        self.prepare(attempt)
        return self.invoke(attempt)

    def invoke(self, attempt):
        p = self.profile
        image = self.modal.Image.from_id(p.image_id, client=self.client)
        r = p.resources
        sb = self.modal.Sandbox.create('/opt/venv/bin/python', '-m', 'mathkernel_compute.managed_worker',
            'modal', '/mk-data', app=self.app, name=attempt.attempt_id, tags=tags(attempt),
            image=image, client=self.client, environment_name=p.environment,
            env={'PYTHONUNBUFFERED': '1'}, secrets=[], include_oidc_identity_token=False,
            timeout=(r.lifetime_ms + 999) // 1000, cpu=(r.cpu_millicores / 1000, r.cpu_millicores / 1000),
            memory=(r.memory_mib, r.memory_mib), gpu=r.gpu, region=p.region,
            block_network=True, encrypted_ports=[], unencrypted_ports=[], h2_ports=[],
            volumes={'/mk-data': self.volume.with_mount_options(
                sub_path='/' + attempt_key(attempt, 'request').rsplit('/', 1)[0])})
        # SDK RPC retries use one idempotency key; a new create is never issued on reconcile.
        return {'handle': stable_sandbox_id(sb.object_id)}

    def sandbox(self, attempt, handle):
        if handle:
            stable_sandbox_id(handle)
        try:
            sb = (self.modal.Sandbox.from_id(handle, client=self.client) if handle else
                  self.modal.Sandbox.from_name(self.profile.app_name, attempt.attempt_id,
                    environment_name=self.profile.environment, client=self.client))
        except self.modal.exception.NotFoundError:
            return None
        stable_sandbox_id(sb.object_id)
        if sb.get_tags() != tags(attempt):
            raise ValueError('MODAL_SANDBOX_OWNERSHIP_MISMATCH')
        return sb

    def observe(self, attempt, handle):
        sb = self.sandbox(attempt, handle)
        code = sb.poll() if sb else None
        try:
            record = check_record(attempt, self.store.get(attempt, 'terminal', 16384))
            terminal = record['terminal']
        except FileNotFoundError:
            terminal = None
        execution = terminal['execution'] if terminal else ('RUNNING' if sb and code is None else
                    'FAILED' if sb else 'SUBMISSION_UNKNOWN')
        resources = 'RELEASE_CONFIRMED' if sb and code is not None else 'ACTIVE' if sb else 'CLEANUP_UNKNOWN'
        return {'handle': sb.object_id if sb else handle, 'observation': {
            'execution': execution, 'resources': resources,
            'message': 'Sandbox process status and durable artifacts observed; billing remains unreconciled.'}}

    def cancel(self, attempt, handle):
        sb = self.sandbox(attempt, handle)
        if sb:
            sb.terminate(wait=False)
        return {'handle': sb.object_id if sb else handle}

    def fetch(self, attempt, handle):
        import hashlib
        record = check_record(attempt, self.store.get(attempt, 'terminal', 16384))
        raw = self.store.get(attempt, 'candidate', attempt.spec.bundle.request.resources.max_output_bytes)
        if hashlib.sha256(raw).hexdigest() != record['candidate_digest']:
            raise ValueError('DURABLE_CANDIDATE_DIGEST')
        return {'handle': handle, 'candidate_b64': base64.b64encode(raw).decode('ascii')}
