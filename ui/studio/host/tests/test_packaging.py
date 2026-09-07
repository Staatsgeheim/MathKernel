import hashlib
import http.client
import json
from pathlib import Path
import re
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mathkernel_studio.server import ASSETS, StudioServer
from mathkernel_studio.testing import FixtureSource


@unittest.skipUnless(ASSETS.joinpath('asset-manifest.json').is_file(), 'Run npm run build first')
class PackagingTests(unittest.TestCase):
    def test_manifest_covers_exactly_the_bundled_local_files(self):
        manifest=json.loads(ASSETS.joinpath('asset-manifest.json').read_text())
        expected={e['path'] for e in manifest['files']} | {'asset-manifest.json'}
        actual={str(p.relative_to(ASSETS)).replace('\\','/') for p in ASSETS.rglob('*') if p.is_file()}
        self.assertEqual(actual,expected)
        for e in manifest['files']:
            data=ASSETS.joinpath(e['path']).read_bytes()
            self.assertEqual(len(data),e['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),e['sha256'])

    def test_built_index_uses_only_bundled_non_inline_scripts(self):
        html=ASSETS.joinpath('index.html').read_text()
        scripts=re.findall(r'<script([^>]*)>(.*?)</script>',html,re.S)
        self.assertTrue(scripts)
        for attrs,body in scripts:
            self.assertEqual(body.strip(),'');self.assertIn('src="/studio/assets/',attrs)
        self.assertNotIn('https://',html)
        self.assertTrue(ASSETS.joinpath('THIRD_PARTY_NOTICES.txt').is_file())

    def test_packaged_host_serves_real_assets_and_nested_shell_without_node(self):
        with StudioServer(FixtureSource(),port=0) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                urls=['/studio/','/studio/workspaces/w/documents/d']
                manifest=json.loads(ASSETS.joinpath('asset-manifest.json').read_text())
                urls += ['/studio/'+e['path'] for e in manifest['files'] if e['path'].startswith('assets/')]
                for url in urls:
                    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
                    connection.request('GET',url);response=connection.getresponse();data=response.read()
                    self.assertEqual(response.status,200,url);self.assertTrue(data);self.assertEqual(response.getheader('Cache-Control'),'no-store');connection.close()
            finally:server.shutdown();thread.join(3)


if __name__=='__main__':unittest.main()
