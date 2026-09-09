"""Lambda Cloud API 1.10.0 subset, checked against the live reference 2026-09-09.

Only instance launch/list/get/terminate. No guest shutdown, credentials on guests,
cloud-init, filesystem allocation, arbitrary URLs, redirects or automatic retries.
"""
from __future__ import annotations
import io
import os
import sys
import time
import urllib.error
import urllib.request
from pydantic import TypeAdapter
from .lambda_models import ProviderID
from .protocol import canonical, decode, read_frame, write_frame, MAX_MESSAGE
from .remote import RemoteUnavailable, run_bounded
from .executor import worker_environment


class LambdaRejected(RemoteUnavailable):
    """A definite launch rejection, before an allocation is owned."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RemoteUnavailable('LAMBDA_REDIRECT_REFUSED')


def provider_id(value):
    return TypeAdapter(ProviderID).validate_python(value, strict=True)


def https_json(path, token, data=None):
    request = urllib.request.Request('https://cloud.lambda.ai/api/v1/' + path,
        data=canonical(data) if data is not None else None,
        headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/json', 'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=10) as response:
            size = response.headers.get('Content-Length')
            if size is not None and (not size.isdecimal() or int(size) > MAX_MESSAGE):
                raise RemoteUnavailable('LAMBDA_RESPONSE_LIMIT')
            return decode(response.read(MAX_MESSAGE + 1))
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()  # Never persist server error bodies, Jupyter URLs or account secrets.
        if code == 404 and path.startswith('instances/'):
            raise FileNotFoundError('LAMBDA_INSTANCE_UNAVAILABLE') from None
        if path == 'instance-operations/launch' and code in {400, 401, 403, 404, 429}:
            raise LambdaRejected('LAMBDA_LAUNCH_REJECTED') from None
        raise RemoteUnavailable('LAMBDA_HTTP_UNAVAILABLE') from None


def instance_snapshot(raw):
    """Retain only ownership/readiness fields. The API also returns Jupyter secrets."""
    if not isinstance(raw, dict):
        raise ValueError('LAMBDA_INSTANCE_SCHEMA')
    value = {k: raw[k] for k in ('id', 'name', 'status', 'ip', 'ssh_key_names', 'file_system_names',
                               'file_system_mounts', 'tags', 'firewall_rulesets') if k in raw}
    value['id'] = provider_id(value.get('id'))
    for field in ('region', 'instance_type', 'image'):
        if field in raw:
            key = 'id' if field == 'image' else 'name'
            value[field] = {key: raw[field][key]}
    if value.get('status') not in {'booting', 'active', 'unhealthy', 'terminated', 'terminating', 'preempted'}:
        raise ValueError('LAMBDA_STATUS_UNSUPPORTED')
    return value


def call_api(http, action, body):
    if action == 'launch':
        response = http('instance-operations/launch', data=body)
        ids = response['data']['instance_ids']
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError('LAMBDA_LAUNCH_IDENTITY_AMBIGUOUS')
        return {'instance_id': provider_id(ids[0])}
    if action == 'list':
        response = http('instances')
        rows = response['data']
        if not isinstance(rows, list) or len(rows) > 1000:
            raise ValueError('LAMBDA_LIST_LIMIT')
        return {'instances': [instance_snapshot(x) for x in rows]}
    identity = provider_id(body['instance_id'])
    if action == 'get':
        return instance_snapshot(http('instances/' + identity)['data'])
    if action == 'terminate':
        rows = http('instance-operations/terminate', data={'instance_ids': [identity]})['data']['terminated_instances']
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get('id') != identity:
            raise ValueError('LAMBDA_TERMINATION_IDENTITY')
        # Even the example response says 'terminating': acknowledgement is not release.
        return instance_snapshot(rows[0])
    raise ValueError('LAMBDA_ACTION_UNSUPPORTED')


class LambdaAPI:
    def __init__(self, profile, *, _http=None, journal=None):
        self.profile = profile
        self.http = _http
        self.journal = journal

    def ready(self):
        if self.http is None and not os.environ.get(self.profile.credential_env):
            raise LambdaRejected('LAMBDA_CREDENTIALS_UNAVAILABLE')

    def call(self, action, body=None, *, launch_before_ms=None):
        self.ready()
        if self.http is not None:
            if action == 'launch' and (launch_before_ms is None or time.time_ns() // 1_000_000 >= launch_before_ms):
                raise LambdaRejected('VM_START_AUTHORITY_EXPIRED')
            return call_api(self.http, action, body)
        if self.journal is not None:
            # Provider guidance: one request/second, launch at most every 12 seconds.
            # Persistent timestamps also cover repeated one-shot CLI invocations.
            from .protocol import digest
            key = 'lambda_rate_' + digest(self.profile.account_scope)
            try:
                previous = self.journal.get('identity', key)
            except KeyError:
                previous = {'request_ms': 0, 'launch_ms': 0}
            now = time.time_ns() // 1_000_000
            due = max(previous['request_ms'] + 1050, previous['launch_ms'] + 12100 if action == 'launch' else 0)
            if due > now:
                time.sleep(min((due - now) / 1000, 12.1))
            previous['request_ms'] = time.time_ns() // 1_000_000
            if action == 'launch':
                previous['launch_ms'] = previous['request_ms']
            with self.journal.transaction():
                self.journal.put('identity', key, previous)
        env = worker_environment()
        env['MK_LAMBDA_CONTROL_KEY'] = os.environ[self.profile.credential_env]
        data = io.BytesIO(); write_frame(data, {'action': action, 'body': body, 'launch_before_ms': launch_before_ms})
        # Includes DNS/TLS/connect/slow body parsing; a socket timeout alone is insufficient.
        raw = run_bounded([sys.executable, '-m', 'mathkernel_compute.lambda_api'], data.getvalue(),
                          env=env, timeout=30, limit=MAX_MESSAGE + 4)
        stream = io.BytesIO(raw); reply = decode(read_frame(stream))
        if stream.read(1) or not isinstance(reply, dict) or set(reply) != {'code', 'value'}:
            raise RemoteUnavailable('LAMBDA_BRIDGE_SCHEMA')
        if reply['code'] == 'rejected':
            raise LambdaRejected('LAMBDA_LAUNCH_REJECTED')
        if reply['code'] == 'missing':
            raise FileNotFoundError('LAMBDA_INSTANCE_UNAVAILABLE')
        if reply['code'] != 'ok':
            raise RemoteUnavailable('LAMBDA_CONTROL_UNAVAILABLE')
        return reply['value']


def main():
    try:
        data = decode(read_frame(sys.stdin.buffer))
        if sys.stdin.buffer.read(1) or set(data) != {'action', 'body', 'launch_before_ms'}:
            raise ValueError('LAMBDA_BRIDGE_SCHEMA')
        if data['action'] == 'launch' and (type(data['launch_before_ms']) is not int
                or time.time_ns() // 1_000_000 >= data['launch_before_ms']):
            raise LambdaRejected('VM_START_AUTHORITY_EXPIRED')
        def http(path, data=None):
            return https_json(path, os.environ['MK_LAMBDA_CONTROL_KEY'], data)
        reply = {'code': 'ok', 'value': call_api(http, data['action'], data['body'])}
    except LambdaRejected:
        reply = {'code': 'rejected', 'value': None}
    except FileNotFoundError:
        reply = {'code': 'missing', 'value': None}
    except Exception:
        reply = {'code': 'unavailable', 'value': None}
    write_frame(sys.stdout.buffer, reply)


if __name__ == '__main__':
    main()
