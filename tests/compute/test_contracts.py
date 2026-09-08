import io
import struct
import unittest
from pydantic import ValidationError
from mathkernel_compute import ComputeRequest, CuboidParameters, ConvolutionParameters, ResourceRequirements
from mathkernel_compute.models import RemoteResultEnvelope
from mathkernel_compute.protocol import canonical, decode, digest, read_frame, write_frame, parse


class ContractTests(unittest.TestCase):
    def test_request_rejects_authority_and_arbitrary_dispatch(self):
        base = {'operation': 'cuboid_sweep', 'parameters': {'bound': '20'}}
        for extra in [{'approved': True}, {'trust_remote': True}, {'target': 'ssh:attacker'},
                      {'operation': 'python'}, {'parameters': {'bound': '20', 'module': 'os'}}]:
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                ComputeRequest.model_validate({**base, **extra})

    def test_exact_values_strict_limits_and_nested_freezing(self):
        for value in [True, 20, 2.0, '020', '0', '2001', '9007199254740993']:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CuboidParameters(bound=value)
        for value in [True, 3.5, '1000']:
            with self.assertRaises(ValidationError):
                ResourceRequirements(execution_timeout_ms=value)
        values = ['1.5', '2.5']
        p = parse(ConvolutionParameters, canonical({'left': values, 'right': ['2']}))
        values[0] = '999'
        self.assertEqual(p.left, ('1.5', '2.5'))
        with self.assertRaises(ValidationError):
            p.left = ('9',)

    def test_rfc8785_utf16_sorting_and_unicode_preservation(self):
        value = {'\ue000': 'bmp', '\U0001f600': 'astral', 'e\u0301': 'decomposed', 'é': 'composed'}
        raw = canonical(value)
        self.assertLess(raw.index('😀'.encode()), raw.index('\ue000'.encode()))
        self.assertEqual(decode(raw), value)
        self.assertNotEqual(digest({'x': 'é'}), digest({'x': 'e\u0301'}))

    def test_hostile_json_and_frames(self):
        for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e3}',
                    b'{"x":9007199254740993}', b'{"__proto__":{}}', b'{"x":"\\ud800"}',
                    b'['*70+b'0'+b']'*70, b'pickle\x80']:
            with self.subTest(raw=raw[:40]), self.assertRaises(ValueError):
                decode(raw)
        for frame in [struct.pack('!I', 2**31), struct.pack('!I', 100)+b'{}', b'']:
            with self.assertRaises(ValueError):
                read_frame(io.BytesIO(frame))
        stream = io.BytesIO(); write_frame(stream, {'x': '9007199254740993'}); stream.seek(0)
        self.assertEqual(decode(read_frame(stream)), {'x': '9007199254740993'})

    def test_unknown_protocol_and_untrusted_metadata(self):
        d = dict(attempt_id='attempt_x', workspace_id='w', bundle_digest='0'*64, execution_digest='1'*64,
                 operation='cuboid_sweep', output_schema='mk.cuboid-pairs/1', output=(('3','4'),))
        with self.assertRaises(ValidationError):
            RemoteResultEnvelope(**d, protocol_version='2.0')
        with self.assertRaises(ValidationError):
            RemoteResultEnvelope(**d, trust='formal')
        c = RemoteResultEnvelope(**d, worker_claims={'trust':'formal','verified':True,'image_hash':'forged'})
        self.assertNotIn('mathkernel.models', RemoteResultEnvelope.__module__)
        self.assertEqual(c.worker_claims['trust'], 'formal')

    def test_base_import_does_not_load_compute_or_contact_network(self):
        import os, subprocess, sys
        code="""import socket,sys
from unittest.mock import patch
with patch.object(socket.socket,'connect',side_effect=AssertionError('unexpected network')):
 import mathkernel
 assert 'mathkernel_compute' not in sys.modules
 assert 'rfc8785' not in sys.modules
 import mathkernel_compute
 assert 'mathkernel_compute.client' not in sys.modules
 assert 'rfc8785' not in sys.modules
 print('offline imports passed')
"""
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(x for x in sys.path if x)
        value=subprocess.check_output([sys.executable,'-c',code],env=env,text=True,timeout=30)
        self.assertIn('offline imports passed',value)
