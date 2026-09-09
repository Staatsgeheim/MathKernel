"""Narrow durable stores: fixed attempt keys, bounded bytes, no Python objects."""
from __future__ import annotations
import importlib.metadata
import hashlib
import io
import os
from urllib.parse import urlsplit
from .protocol import canonical, decode, digest
from .managed import BOTO3_VERSION
from .remote import AttemptRef

KINDS = frozenset({'request', 'claim', 'receipt', 'candidate', 'terminal', 'cancel'})


def guard_s3_request(config, request):
    """Reject SDK endpoint overrides/region redirects before transmitting data."""
    url = urlsplit(request.url)
    expected = f'{config.bucket}.s3.{config.region}.amazonaws.com'
    if (url.scheme != 'https' or url.hostname != expected or url.port not in (None, 443)
            or url.username is not None or url.password is not None):
        raise ValueError('S3_REGION_OR_HOST_CHANGED')


def attempt_key(attempt, kind):
    if kind not in KINDS:
        raise ValueError('ARTIFACT_KIND_INVALID')
    ref = AttemptRef.from_attempt(attempt)
    return f'{ref.workspace_id}/{ref.attempt_id}/{kind}.json'


def bound_record(attempt, **data):
    return dict(ref=AttemptRef.from_attempt(attempt).model_dump(mode='json'), **data)


def check_record(attempt, raw):
    record = decode(raw)
    if not isinstance(record, dict) or record.get('ref') != AttemptRef.from_attempt(attempt).model_dump(mode='json'):
        raise ValueError('DURABLE_RECORD_BINDING')
    return record


class S3Store:
    def __init__(self, config, *, _client=None):
        self.config = config
        if _client is None:
            if importlib.metadata.version('boto3') != BOTO3_VERSION:
                raise ValueError('S3_SDK_VERSION_UNQUALIFIED')
            import boto3
            from botocore.config import Config
            prefix = config.credential_env_prefix
            # No default credentials, instance role discovery, custom endpoint, or retry.
            self.client = boto3.client('s3', region_name=config.region,
                aws_access_key_id=os.environ[prefix + '_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ[prefix + '_SECRET_ACCESS_KEY'],
                aws_session_token=os.environ.get(prefix + '_SESSION_TOKEN'),
                config=Config(connect_timeout=5, read_timeout=10, retries={'total_max_attempts': 1},
                    proxies={}, signature_version='s3v4', ignore_configured_endpoint_urls=True,
                    s3={'addressing_style': 'virtual', 'us_east_1_regional_endpoint': 'regional'}))
            self.client.meta.events.register('before-send.s3',
                lambda request, **kwargs: guard_s3_request(config, request))
        else:
            self.client = _client

    def key(self, attempt, kind):
        return self.config.prefix + '/' + attempt_key(attempt, kind)

    def get(self, attempt, kind, limit=1_048_576):
        try:
            response = self.client.get_object(Bucket=self.config.bucket, Key=self.key(attempt, kind))
        except Exception as exc:
            if getattr(exc, 'response', {}).get('Error', {}).get('Code') in {'NoSuchKey', '404'}:
                raise FileNotFoundError(kind) from None
            raise
        body = response['Body']
        try:
            size = response.get('ContentLength')
            if type(size) is not int or not 0 <= size <= limit:
                raise ValueError('ARTIFACT_SIZE_LIMIT')
            raw = body.read(limit + 1)
            if len(raw) != size or len(raw) > limit:
                raise ValueError('ARTIFACT_SIZE_MISMATCH')
            return raw
        finally:
            body.close()

    def create(self, attempt, kind, raw):
        """Conditional, immutable creation. A failed claim must never run mathematics."""
        if len(raw) > 1_048_576:
            raise ValueError('ARTIFACT_SIZE_LIMIT')
        try:
            self.client.put_object(Bucket=self.config.bucket, Key=self.key(attempt, kind),
                Body=raw, ContentType='application/json', IfNoneMatch='*')
            return True
        except Exception as exc:
            if getattr(exc, 'response', {}).get('Error', {}).get('Code') in {'PreconditionFailed', '412'}:
                return False
            raise  # Ambiguous claim outcomes never permit execution.

    def put(self, attempt, kind, raw):
        if not self.create(attempt, kind, raw) and self.get(attempt, kind) != raw:
            raise ValueError('IMMUTABLE_ARTIFACT_CONFLICT')


class ModalVolumeStore:
    """Used only inside the memory/deadline limited SDK bridge subprocess.

    Modal 1.5.5 read_file prefetches whole blocks. The subprocess's RLIMIT_AS
    contains that allocation before the bounded data reaches the controller.
    """
    def __init__(self, volume, *, missing_errors=(FileNotFoundError,)):
        self.volume = volume
        self.missing_errors = missing_errors

    def get(self, attempt, kind, limit=1_048_576):
        try:
            return self._get(attempt, kind, limit)
        except self.missing_errors:
            raise FileNotFoundError(kind) from None

    def _get(self, attempt, kind, limit):
        key = attempt_key(attempt, kind)
        entries = iter(self.volume.iterdir(key, recursive=False))
        try:
            entry = next(entries)
        except StopIteration:
            raise FileNotFoundError(kind) from None
        # Do not list an entire remote directory or trust a size returned in a result.
        if next(entries, None) is not None or not 0 <= entry.size <= limit:
            raise ValueError('ARTIFACT_SIZE_LIMIT')
        result = bytearray()
        for chunk in self.volume.read_file(key):
            if len(result) + len(chunk) > limit:
                raise ValueError('ARTIFACT_SIZE_LIMIT')
            result.extend(chunk)
        if len(result) != entry.size:
            raise ValueError('ARTIFACT_SIZE_MISMATCH')
        return bytes(result)

    def put(self, attempt, kind, raw):
        try:
            if self.get(attempt, kind) != raw:
                raise ValueError('IMMUTABLE_ARTIFACT_CONFLICT')
            return
        except FileNotFoundError:
            pass
        if len(raw) > 1_048_576:
            raise ValueError('ARTIFACT_SIZE_LIMIT')
        # No overwrite, user-selected path, artifact URL, or automatic volume creation.
        with self.volume.batch_upload(force=False) as upload:
            upload.put_file(io.BytesIO(raw), '/' + attempt_key(attempt, kind))


def terminal_record(attempt, raw, candidate=None):
    terminal = decode(raw)
    if terminal.get('attempt_id') != attempt.attempt_id or terminal.get('execution_digest') != attempt.execution_digest:
        raise ValueError('WORKER_TERMINAL_BINDING')
    return canonical(bound_record(attempt, terminal=terminal,
                                  candidate_digest=hashlib.sha256(candidate).hexdigest() if candidate is not None else None))
