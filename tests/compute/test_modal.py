import base64
import io
import inspect
import subprocess
import sys
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from test_managed import profile, client, Cloud, launch
from mathkernel_compute.modal_adapter import ModalAdapter, tags
from mathkernel_compute.object_store import attempt_key, terminal_record
from mathkernel_compute.protocol import canonical, decode, read_frame, write_frame
from mathkernel_compute.models import RemoteResultEnvelope
from mathkernel_compute.registry import execute
from mathkernel_compute.managed import BridgeCall, bridge_call
from mathkernel_compute.remote import RemoteUnavailable


class Missing(Exception): pass


class Volume:
    object_id = 'vo-test'
    def __init__(self): self.files = {}; self.calls = []
    def hydrate(self): self.calls.append('hydrate')
    def iterdir(self, key, *, recursive):
        assert recursive is False
        if key not in self.files: raise Missing()
        yield SimpleNamespace(size=len(self.files[key]))
    def read_file(self, key): yield self.files[key]
    def with_mount_options(self, **kw):
        self.calls.append(kw)
        return ('subdirectory', kw['sub_path'])
    def batch_upload(self, *, force):
        assert force is False
        return self
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def put_file(self, stream, key):
        key = key.lstrip('/')
        assert key not in self.files
        self.files[key] = stream.read()


class Sandbox:
    object_id = 'sb-' + '1'*22
    def __init__(self): self.tags = {}; self.code = None; self.terminations = 0
    def get_tags(self): return self.tags
    def poll(self): return self.code
    def terminate(self, *, wait):
        assert wait is False
        self.terminations += 1


class Modal:
    def __init__(self):
        self.volume = Volume(); self.sb = Sandbox(); self.calls = []; self.running_visible = True
        self.exception = SimpleNamespace(NotFoundError=Missing)
        self.App = SimpleNamespace(lookup=self.lookup)
        self.Volume = SimpleNamespace(from_name=self.volume_name)
        self.Image = SimpleNamespace(from_id=self.image_id)
        self.Sandbox = SimpleNamespace(create=self.create, from_id=self.from_id, from_name=self.from_name)
    def lookup(self, name, **kw):
        assert kw['create_if_missing'] is False
        self.calls.append(('app', name, kw))
        return SimpleNamespace(app_id='ap-test')
    def volume_name(self, name, **kw):
        assert kw['create_if_missing'] is False and kw['version'] == 2
        return self.volume
    def image_id(self, name, **kw):
        self.calls.append(('image', name, kw))
        return ('existing-image', name)
    def create(self, *args, **kw):
        self.calls.append(('create', args, kw)); self.sb.tags = kw['tags']
        return self.sb
    def from_id(self, identity, **kw):
        assert identity == self.sb.object_id
        return self.sb
    def from_name(self, app, name, **kw):
        if not self.running_visible: raise Missing()
        return self.sb


def test_modal_fixed_sandbox_shape_identity_and_durable_artifacts(tmp_path):
    p = profile('modal'); cloud = Cloud(); sdk = Modal()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c); a = c._attempt(c._job(j.job_id))
        adapter = ModalAdapter(p, _modal=sdk, _client=object())
        assert adapter.submit(a) == {'handle': 'sb-1111111111111111111111'}
        name, argv, kwargs = sdk.calls[-1]
        assert argv == ('/opt/venv/bin/python', '-m', 'mathkernel_compute.managed_worker', 'modal', '/mk-data')
        assert kwargs['cpu'] == (1.0, 1.0) and kwargs['memory'] == (2048, 2048)
        assert kwargs['timeout'] == 180 and kwargs['gpu'] is None
        assert kwargs['block_network'] is True and kwargs['secrets'] == []
        assert kwargs['include_oidc_identity_token'] is False
        assert kwargs['volumes']['/mk-data'] == ('subdirectory', '/' + a.workspace_id + '/' + a.attempt_id)
        assert canonical(a) == sdk.volume.files[attempt_key(a, 'request')]
        assert adapter.observe(a, None)['handle'] == 'sb-1111111111111111111111'
        raw = canonical(RemoteResultEnvelope(attempt_id=a.attempt_id, workspace_id=a.workspace_id,
            execution_digest=a.execution_digest, bundle_digest=a.bundle_digest, operation='cuboid_sweep',
            output_schema='mk.cuboid-pairs/1', output=execute(a.spec.bundle.request)))
        sdk.volume.files[attempt_key(a, 'candidate')] = raw
        sdk.volume.files[attempt_key(a, 'terminal')] = terminal_record(a,
            canonical(dict(attempt_id=a.attempt_id, execution_digest=a.execution_digest,
                           execution='RECEIVED', resources='RELEASE_CONFIRMED', revision=2)), raw)
        sdk.sb.code = 0; sdk.running_visible = False
        assert adapter.observe(a, 'sb-1111111111111111111111')['observation']['resources'] == 'RELEASE_CONFIRMED'
        assert base64.b64decode(adapter.fetch(a, 'sb-1111111111111111111111')['candidate_b64']) == raw
        # Durable output without an authoritative handle cannot settle resource ownership.
        assert adapter.observe(a, None)['observation']['resources'] == 'CLEANUP_UNKNOWN'
        sdk.volume.files[attempt_key(a, 'candidate')] = b'{}'
        with pytest.raises(ValueError, match='DIGEST'): adapter.fetch(a, 'sb-1111111111111111111111')


def test_modal_foreign_tags_prevent_cancellation_and_ack_is_not_release(tmp_path):
    p = profile('modal'); cloud = Cloud(); sdk = Modal()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c); a = c._attempt(c._job(j.job_id))
        adapter = ModalAdapter(p, _modal=sdk, _client=object()); adapter.submit(a)
        adapter.cancel(a, 'sb-1111111111111111111111')
        assert sdk.sb.terminations == 1
        assert adapter.observe(a, 'sb-1111111111111111111111')['observation']['resources'] == 'ACTIVE'
        sdk.sb.tags = dict(sdk.sb.tags, mk_workspace='foreign')
        with pytest.raises(ValueError, match='OWNERSHIP'): adapter.cancel(a, 'sb-1111111111111111111111')
        assert sdk.sb.terminations == 1


def test_modal_profile_volume_rebinding_and_output_limits(tmp_path):
    p = profile('modal'); cloud = Cloud(); sdk = Modal()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c); a = c._attempt(c._job(j.job_id))
        sdk.volume.object_id = 'vo-foreign'
        with pytest.raises(ValueError, match='VOLUME_CHANGED'): ModalAdapter(p, _modal=sdk, _client=object())
        sdk.volume.object_id = 'vo-test'
        adapter = ModalAdapter(p, _modal=sdk, _client=object())
        sdk.volume.files[attempt_key(a, 'candidate')] = b'x'*10
        with pytest.raises(ValueError, match='SIZE'): adapter.store.get(a, 'candidate', 5)
        assert not any(v[0] == 'create' for v in sdk.calls)


def test_provider_bridge_rejects_python_object_bytes_without_importing_sdk():
    # The production bridge sets Linux memory/CPU limits before reading a frame.
    p = subprocess.run([sys.executable, '-m', 'mathkernel_compute.provider_bridge'],
        input=b'\x00\x00\x00\x02\x80\x04', capture_output=True, timeout=15)
    assert p.returncode == 0
    assert decode(read_frame(io.BytesIO(p.stdout))) == {
        'protocol': 'mk.managed-bridge/1', 'error': 'PROVIDER_CALL_UNAVAILABLE'}


def test_installed_sdk_method_contracts():
    modal = pytest.importorskip('modal', reason='optional pinned Modal SDK is not installed')
    import importlib.metadata
    assert importlib.metadata.version('modal') == '1.5.5'
    required = {'app', 'name', 'tags', 'image', 'env', 'secrets', 'timeout', 'gpu', 'cpu', 'memory',
                'block_network', 'volumes', 'client', 'region', 'include_oidc_identity_token'}
    assert required <= inspect.signature(modal.Sandbox.create).parameters.keys()
    assert {'version', 'create_if_missing', 'client'} <= inspect.signature(modal.Volume.from_name).parameters.keys()
    assert 'sub_path' in inspect.signature(modal.Volume.with_mount_options).parameters
    assert callable(modal.Sandbox.from_id) and callable(modal.Sandbox.poll) and callable(modal.Sandbox.get_tags)
    # Image.from_id is lazy, references an existing ID and does not create/build anything.
    assert modal.Image.from_id('im-audited').object_id == 'im-audited'


def test_installed_s3_sdk_supports_immutable_writes():
    pytest.importorskip('boto3', reason='optional pinned S3 SDK is not installed')
    import importlib.metadata
    from botocore.session import Session
    assert importlib.metadata.version('boto3') == '1.43.90'
    assert 'IfNoneMatch' in Session().get_service_model('s3').operation_model('PutObject').input_shape.members


def test_real_modal_sdk_treats_result_metadata_as_data_not_python():
    modal = pytest.importorskip('modal')
    from modal_proto import api_pb2
    from google.protobuf.empty_pb2 import Empty
    class Stub:
        async def SandboxWait(self, req):
            return api_pb2.SandboxWaitResponse(result=api_pb2.GenericResult(status=1,
                exitcode=0, data=b'\x80\x04untrusted-python-object', exception='hostile exception text'))
        async def SandboxTagsGet(self, req):
            return api_pb2.SandboxTagsGetResponse()
        async def SandboxTerminate(self, req):
            return Empty()
    with patch('modal._serialization.deserialize', side_effect=AssertionError('untrusted deserialization')) as loads:
        sb = modal.Sandbox.from_id('sb-1111111111111111111111', client=SimpleNamespace(stub=Stub()))
        assert sb.poll() == 0
        assert sb.get_tags() == {}
        sb.terminate(wait=False)
        loads.assert_not_called()


def test_beta_sandbox_handle_rejected_before_sdk_lookup(tmp_path):
    p = profile('modal'); cloud = Cloud(); sdk = Modal()
    with client(tmp_path, p, cloud) as c:
        j, _, _ = launch(c); a = c._attempt(c._job(j.job_id))
        adapter = ModalAdapter(p, _modal=sdk, _client=object())
        with pytest.raises(ValueError, match='BACKEND_UNQUALIFIED'): adapter.cancel(a, 'sb-beta')
        assert sdk.sb.terminations == 0
