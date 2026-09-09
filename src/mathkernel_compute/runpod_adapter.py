"""Existing queue endpoint adapter. Bounded HTTPS JSON; no endpoint mutations."""
from __future__ import annotations
import base64
import hashlib
import os
import re
import urllib.error
import urllib.request
from .object_store import S3Store, bound_record, check_record
from .protocol import canonical, decode, digest
from .remote import RemoteUnavailable


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RemoteUnavailable('Provider redirects are not permitted')


def https_json(url, token, *, data=None):
    # Only constants + validated endpoint/job IDs constructed below reach this function.
    request = urllib.request.Request(url, data=canonical(data) if data is not None else None,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/json', 'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=15) as response:
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdecimal() or int(length) > 65536):
                raise ValueError('PROVIDER_RESPONSE_SIZE')
            return decode(response.read(65537), 65536)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        if code == 404:
            raise FileNotFoundError('Provider job or endpoint unavailable') from None
        raise RemoteUnavailable('RUNPOD_HTTP_UNAVAILABLE') from None


def deployment_snapshot(value):
    """Drop secrets/irrelevant metadata; retain the allocation and image identity."""
    if not isinstance(value, dict):
        raise ValueError('RUNPOD_ENDPOINT_SCHEMA')
    fields = ('id', 'type', 'image', 'workers', 'gpu', 'cpu', 'dataCenterIds',
              'args', 'disk', 'ports', 'networkVolumes', 'timeout', 'flashboot', 'scaling')
    snapshot = {k: value[k] for k in fields if k in value}
    if not {'id', 'type', 'image', 'workers', 'dataCenterIds'} <= snapshot.keys():
        raise ValueError('RUNPOD_ENDPOINT_SCHEMA')
    canonical(snapshot)  # Reject duplicate/unsafe numbers through the decoder, bound structured metadata.
    return snapshot


class RunpodAdapter:
    def __init__(self, profile, *, _http=None, _store=None):
        self.profile = profile
        self.token = os.environ.get(profile.credential_env_prefix + '_API_KEY', '')
        if _http is None and not self.token:
            raise RemoteUnavailable('MANAGED_CREDENTIALS_UNAVAILABLE')
        self.http = _http or https_json
        self.store = _store or S3Store(profile.storage)

    def probe(self, *, enforce=False):
        p = self.profile
        raw = self.http('https://api.runpod.io/v2/serverless/' + p.endpoint_id, self.token)
        snapshot = deployment_snapshot(raw)
        if enforce:
            workers = snapshot['workers']
            if (digest(snapshot) != p.deployment_digest or snapshot['id'] != p.endpoint_id
                    or snapshot['type'] != 'QUEUE' or snapshot['image'] != p.source_image
                    or workers.get('min') != p.workers_min or workers.get('max') != p.workers_max
                    or workers.get('idleTimeout') != p.idle_timeout_seconds):
                raise ValueError('RUNPOD_ENDPOINT_CHANGED')
            gpu = snapshot.get('gpu')
            if p.resources.gpu:
                if not isinstance(gpu, dict) or gpu.get('count') != 1 or gpu.get('pools') != [p.resources.gpu]:
                    raise ValueError('RUNPOD_GPU_ALLOCATION_MISMATCH')
            elif gpu or not snapshot.get('cpu'):
                raise ValueError('RUNPOD_CPU_ALLOCATION_MISMATCH')
        return {'adapter': 'runpod', 'deployment': snapshot, 'deployment_digest': digest(snapshot),
                'qualification': 'experimental-not-live-qualified'}

    def url(self, action, handle=None):
        if action not in {'run', 'status', 'cancel'}:
            raise ValueError('RUNPOD_ACTION_UNSUPPORTED')
        if handle is not None and not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', handle):
            raise ValueError('RUNPOD_HANDLE_INVALID')
        return 'https://api.runpod.ai/v2/' + self.profile.endpoint_id + '/' + action + ('/' + handle if handle else '')

    def prepare(self, attempt):
        self.probe(enforce=True)  # Drift denial precedes export or invocation.
        self.store.put(attempt, 'request', canonical(attempt))

    def submit(self, attempt):
        self.prepare(attempt)
        return self.invoke(attempt)

    def invoke(self, attempt):
        response = self.http(self.url('run'), self.token, data={
            'input': bound_record(attempt, request_digest=digest(attempt)),
            'policy': {'executionTimeout': self.profile.resources.lifetime_ms,
                       'ttl': self.profile.resources.ttl_ms}})
        self.url('status', response['id'])
        return {'handle': response['id']}

    def recover_handle(self, attempt, handle):
        try:
            record = check_record(attempt, self.store.get(attempt, 'receipt', 16384))
        except FileNotFoundError:
            return handle
        reported = record['provider_job_id']
        self.url('status', reported)
        if handle and handle != reported:
            raise ValueError('RUNPOD_JOB_BINDING_CONFLICT')
        # A compromised worker can write a foreign job ID into its receipt. The
        # queue API has no qualified lookup by our attempt key. Only the original
        # provider submission acknowledgement authorizes status/cancel by ID.
        # Durable outputs remain retrievable after lost acknowledgement, but
        # cleanup/exposure stay unknown until an authoritative handle is available.
        return handle

    def observe(self, attempt, handle):
        handle = self.recover_handle(attempt, handle)
        try:
            native = self.http(self.url('status', handle), self.token) if handle else {}
            if handle and native.get('id') != handle:
                raise ValueError('RUNPOD_STATUS_IDENTITY')
        except FileNotFoundError:
            native = {}  # TTL/retention expiry is not proof of non-execution or zero spend.
        try:
            terminal = check_record(attempt, self.store.get(attempt, 'terminal', 16384))['terminal']
        except FileNotFoundError:
            terminal = None
        status = native.get('status')
        execution = (terminal['execution'] if terminal else
            {'IN_QUEUE': 'QUEUED', 'IN_PROGRESS': 'RUNNING', 'FAILED': 'FAILED',
             'CANCELLED': 'CANCELLED', 'TIMED_OUT': 'TIMED_OUT'}.get(status, 'SUBMISSION_UNKNOWN'))
        released = (terminal and terminal['resources'] == 'RELEASE_CONFIRMED' and
                    status in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'})
        resources = 'RELEASE_CONFIRMED' if released else 'ACTIVE' if status == 'IN_PROGRESS' else 'CLEANUP_UNKNOWN'
        return {'handle': handle, 'observation': {'execution': execution, 'resources': resources,
            'message': 'Job process and durable artifacts observed. Existing endpoint/storage costs continue independently.'}}

    def cancel(self, attempt, handle):
        self.store.put(attempt, 'cancel', canonical(bound_record(attempt, cancel=True)))
        handle = self.recover_handle(attempt, handle)
        if handle:
            try:
                self.http(self.url('cancel', handle), self.token, data={})
            except FileNotFoundError:
                pass
        return {'handle': handle}  # An acknowledgement makes no release claim.

    def fetch(self, attempt, handle):
        record = check_record(attempt, self.store.get(attempt, 'terminal', 16384))
        raw = self.store.get(attempt, 'candidate', attempt.spec.bundle.request.resources.max_output_bytes)
        if hashlib.sha256(raw).hexdigest() != record['candidate_digest']:
            raise ValueError('DURABLE_CANDIDATE_DIGEST')
        return {'handle': handle, 'candidate_b64': base64.b64encode(raw).decode('ascii')}
