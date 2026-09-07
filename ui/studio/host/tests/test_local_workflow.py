"""Authenticated HTTP -> real workflow process -> durable result integration."""
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT / 'ui/studio/host/src')]
from mathkernel import MathKernel
from mathkernel_workflow import WorkflowRuntime
from mathkernel_workflow.contracts import binding
from mathkernel_studio.server import StudioServer
from mathkernel_studio.local import LocalWorkflowSource
from test_workflow_runtime import graph

class LocalHTTPTests(unittest.TestCase):

    def test_browser_protocol_executes_real_graph_and_publishes_original_result(self):
        with tempfile.TemporaryDirectory() as directory, WorkflowRuntime(directory) as runtime:
            source = LocalWorkflowSource(MathKernel(), runtime)
            with StudioServer(source, port=0, service=runtime) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
                    client.request('POST', '/studio/api/session', json.dumps({'code': server.connect_code}), {'Origin': server.origin, 'Content-Type': 'application/json', 'X-Studio-Action': 'connect'})
                    r = client.getresponse()
                    self.assertEqual(r.status, 200)
                    cookie = r.getheader('Set-Cookie').split(';')[0]
                    r.read()

                    def request(path, payload=None, request_id='read'):
                        headers = {'Cookie': cookie, 'X-Studio-Request': request_id}
                        if payload is not None:
                            headers.update({'Origin': server.origin, 'Content-Type': 'application/json', 'X-Studio-Action': path})
                        client.request('POST' if payload is not None else 'GET', '/studio/api/' + path, json.dumps(payload) if payload is not None else None, headers)
                        response = client.getresponse()
                        body = response.read()
                        self.assertEqual(response.status, 200, body)
                        value = json.loads(body)
                        self.assertEqual(value['request_id'], request_id)
                        return value['payload']
                    handshake = request('handshake')
                    self.assertFalse(handshake['test_host'])
                    self.assertTrue(handshake['features']['workflow_execute'])
                    catalog = request('catalog')
                    self.assertTrue(any((e['operation_ref'] == 'workflow.parse' for e in catalog['entries'])))
                    doc = graph()
                    frozen = binding(doc)
                    validation = request('workflow/validate', {'document': doc, 'binding': frozen}, 'validation')
                    self.assertTrue(validation['valid'])
                    plan = request('workflow/plan', {'validation_ref': validation['validation_ref'], 'binding': frozen, 'scope': 'workflow_outputs', 'selected_nodes': []}, 'plan')
                    p = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']}
                    challenge = request('approval/challenge', p, 'challenge')
                    approval = request('approval/confirm', {**p, 'challenge_ref': challenge['challenge_ref'], 'decision': 'approve'}, 'approve')
                    payload = {**p, 'authority_ref': approval['authority_ref'], 'client_request_id': 'submit'}
                    reply = request('workflow/submit', payload, 'submit')
                    self.assertEqual(reply['outcome'], 'accepted')
                    client.close()
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=10)
                    self.assertEqual(request('requests/submit'), reply)
                    limit = time.monotonic() + 30
                    while time.monotonic() < limit:
                        run = request('runs/' + reply['run_ref'])
                        if not run['actions']:
                            break
                        time.sleep(0.25)
                    self.assertEqual(run['execution'], 'completed', run)
                    result = request('results/' + run['attempts'][-1]['result_ref'])
                    self.assertEqual(result['admission'], 'host')
                    self.assertEqual(result['binding']['draft_revision'], 1)
                    expected = source.kernel.differentiate(source.kernel.parse('x^3 + 9007199254740993').data['expr_id'], 'x').model_dump(mode='json')
                    self.assertEqual(result['result']['data']['result'], expected['data']['result'])
                    for key in ['trust', 'semantic_status', 'side_conditions', 'assumptions_used']:
                        self.assertEqual(result['result'][key], expected[key])
                    self.assertEqual(result, request('results/' + result['binding']['result_ref'] + '/admission'))
                    self.assertEqual(request('workflow/submit', payload, 'submit'), reply)
                    self.assertEqual(len(request('runs')['runs']), 1)
                    client.close()
                finally:
                    server.shutdown()
                    thread.join(3)
