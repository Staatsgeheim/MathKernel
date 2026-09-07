"""Offline contract/security tests; full MathKernel integration is separately gated."""
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mathkernel_studio.server import StudioServer, CSP
from mathkernel_studio.source import KernelSource, InspectionRecord, descriptor
from mathkernel_studio.testing import FixtureSource, FAULTS


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class HostTests(unittest.TestCase):
    def setUp(self):
        self.assets = tempfile.TemporaryDirectory()
        Path(self.assets.name, 'index.html').write_text('<div>Studio fixture assets</div>')
        self.server = StudioServer(FixtureSource(), port=0, assets=self.assets.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = None

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(3); self.assets.cleanup()

    def request(self, path='/studio/api/handshake', method='GET', body=None, **headers):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        defaults = {'Host':self.server.authority, 'X-Studio-Request':'test-request'}
        if self.cookie: defaults['Cookie'] = self.cookie
        defaults.update(headers)
        conn.request(method, path, body, defaults)
        r = conn.getresponse(); data=r.read(); status=r.status; response_headers=dict(r.getheaders()); conn.close()
        return status, response_headers, data

    def login(self):
        status, headers, _ = self.request('/studio/api/session','POST',json.dumps({'code':self.server.connect_code}),
            **{'Origin':self.server.origin,'Content-Type':'application/json','X-Studio-Action':'connect'})
        self.assertEqual(status,200)
        self.cookie=headers['Set-Cookie'].split(';')[0]
        return headers

    def test_auth_required_for_handshake_and_results(self):
        for route in ('handshake','catalog','results','results/fixture-result'):
            self.assertEqual(self.request('/studio/api/'+route)[0],401)

    def test_session_is_one_time_http_only_same_site_and_not_in_url(self):
        code=self.server.connect_code; headers=self.login()
        self.assertIn('HttpOnly',headers['Set-Cookie']); self.assertIn('SameSite=Strict',headers['Set-Cookie'])
        self.assertEqual(self.request('/studio/api/session','POST',json.dumps({'code':code}),**{'Origin':self.server.origin,'Content-Type':'application/json','X-Studio-Action':'connect'})[0],401)
        status,_,data=self.request(); self.assertEqual(status,200); self.assertNotIn(code.encode(),data)

    def test_wrong_origin_and_dns_rebinding_hosts_rejected(self):
        self.login()
        for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'},{'Sec-Fetch-Site':'cross-site'},{'Sec-Fetch-Site':'same-site'}):
            self.assertEqual(self.request(**headers)[0],403)

    def test_cross_origin_or_missing_origin_bootstrap_rejected(self):
        for origin in (None,'http://evil.example'):
            headers={'Content-Type':'application/json','X-Studio-Action':'connect'}
            if origin: headers['Origin']=origin
            self.assertEqual(self.request('/studio/api/session','POST',json.dumps({'code':self.server.connect_code}),**headers)[0],403)

    def test_api_has_no_execution_or_generic_method_dispatch(self):
        self.login()
        for name in ('workflow/submit','operation/invoke','settings','export','eval','jobs/submit'):
            self.assertEqual(self.request('/studio/api/'+name,'POST','{}',**{'Origin':self.server.origin})[0],405)
            self.assertEqual(self.request('/studio/api/'+name)[0],404)

    def test_cookie_expiration_stops_private_reads(self):
        self.login(); self.server.session_expires=time.monotonic()-1
        self.assertEqual(self.request()[0],401)

    def test_stale_connection_code_is_rejected(self):
        self.server.code_expires=time.monotonic()-1
        self.assertEqual(self.request('/studio/api/session','POST',json.dumps({'code':self.server.connect_code}),**{'Origin':self.server.origin,'Content-Type':'application/json','X-Studio-Action':'connect'})[0],401)

    def test_scope_and_correlation_are_in_every_successful_api_response(self):
        self.login()
        for route in ('handshake','catalog','results','results/fixture-result'):
            status,headers,data=self.request('/studio/api/'+route); e=json.loads(data)
            self.assertEqual(status,200); self.assertEqual(e['host_instance_id'],'fixture-host'); self.assertEqual(e['request_id'],'test-request'); self.assertIn('observed_at',e)
            self.assertEqual(headers['Cache-Control'],'no-store')

    def test_csp_disallows_eval_inline_script_frames_and_external_connections(self):
        _,headers,_=self.request('/studio/')
        self.assertIn("script-src 'self'",headers['Content-Security-Policy']); self.assertNotIn('unsafe-eval',CSP)
        self.assertIn("frame-ancestors 'none'",CSP); self.assertIn("connect-src 'self'",CSP)
        self.assertEqual(headers['X-Content-Type-Options'],'nosniff')

    def test_static_traversal_and_unrecognized_paths_are_inert(self):
        for path in ('/studio/assets/%2e%2e/%2e%2e/server.py','/studio/assets/../../server.py','/studio/unknown.js','/studio?code=secret'):
            status,_,data=self.request(path)
            if path.startswith('/studio?'): self.assertNotIn(b'secret',data)
            else: self.assertEqual(status,404)

    def test_negative_and_duplicate_offsets_rejected(self):
        self.login()
        for query in ('offset=-1','offset=0&offset=1','url=https://example.org','offset=Infinity'):
            self.assertEqual(self.request('/studio/api/catalog?'+query)[0],400)

    def test_protocol_and_permission_faults_are_explicit(self):
        self.login(); self.server.source=FixtureSource('denied'); self.assertEqual(self.request()[0],403)
        self.server.source=FixtureSource('incompatible'); self.assertEqual(json.loads(self.request()[2])['payload']['ui_protocol'],'studio-host/99')

    def test_forged_candidate_badge_is_never_admitted(self):
        self.login(); self.server.source=FixtureSource('candidate')
        p=json.loads(self.request('/studio/api/results/fixture-result')[2])['payload']
        self.assertEqual(p['result']['trust'],'formal'); self.assertEqual(p['admission'],'candidate'); self.assertEqual(p['claim_trust'],{})

    def test_fixture_receipt_pages_roundtrip_with_digest(self):
        self.login(); self.server.source=FixtureSource('receipt'); pieces=[]; offset=0
        while True:
            p=json.loads(self.request(f'/studio/api/results/fixture-result/page?offset={offset}')[2])['payload']; pieces.append(p['content'])
            if p['next_offset'] is None: break
            offset=p['next_offset']
        text=''.join(pieces); self.assertEqual(len(text),p['total_bytes']); self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),p['sha256'])
        self.assertEqual(json.loads(text)['trust'],'numeric')

    def test_expired_result_does_not_trigger_recomputation(self):
        self.login(); self.server.source=FixtureSource('expired'); self.assertEqual(self.request('/studio/api/results/fixture-result')[0],404)

    def test_hostile_labels_are_plain_json_text(self):
        self.login(); self.server.source=FixtureSource('hostile_text')
        status,headers,data=self.request('/studio/api/catalog'); self.assertEqual(status,200); self.assertEqual(headers['Content-Type'],'application/json')
        self.assertIn('<img',json.loads(data)['payload']['entries'][0]['description'])


class AdapterTests(unittest.TestCase):
    def test_real_registry_discovery_keeps_incomplete_metadata_and_no_ports(self):
        capabilities=load_file('studio_test_capabilities', ROOT/'src/mathkernel/capabilities.py')
        discovery=load_file('studio_test_discovery', ROOT/'src/mathkernel/discovery.py')
        registry=capabilities.CapabilityRegistry([capabilities.Capability(name='example.operation',domain='example',input_types=('Matrix',),output_types=('Result',),parameter_schema={'precision':'integer?'})])
        c=registry.manifest()[0]; c['parameter_json_schema']=discovery.parameter_json_schema(c['parameter_schema'])
        d=descriptor(c,'1.3.0')
        self.assertEqual(d['input_ports'],[]);self.assertEqual(d['composition'],'partial');self.assertEqual(d['availability'],'unknown')
        self.assertNotIn('required',d['parameter_schema']);self.assertEqual(d['operation_ref'],'example.operation')

    def test_inspection_records_are_not_browser_admission_endpoints(self):
        self.assertNotIn('do_PUT',vars(__import__('mathkernel_studio.server',fromlist=['Handler']).Handler))
        record=InspectionRecord(b'{}',b'{}',b'{}')
        with self.assertRaises(Exception): record.result_json=b'changed'

    def test_test_host_never_advertises_execution(self):
        for fault in FAULTS:
            if fault=='denied':continue
            h=FixtureSource(fault).handshake()
            self.assertTrue(h['test_host']);self.assertFalse(h['features']['workflow_execute']);self.assertFalse(h['features']['operation_invoke'])


if __name__=='__main__': unittest.main()
