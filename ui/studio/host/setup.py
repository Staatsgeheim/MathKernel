"""Build only a complete, current static bundle; never ship stale cached JS."""
import hashlib
import json
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildStudio(build_py):
    def run(self):
        assets = Path(__file__).parent / 'src' / 'mathkernel_studio' / 'assets'
        if not (assets / 'asset-manifest.json').is_file():
            raise RuntimeError('Build Studio assets with npm ci && npm run build before packaging.')
        manifest = json.loads((assets / 'asset-manifest.json').read_text())
        expected = {entry['path'] for entry in manifest['files']} | {'asset-manifest.json'}
        actual = {str(p.relative_to(assets)).replace('\\', '/') for p in assets.rglob('*') if p.is_file()}
        if expected != actual:
            raise RuntimeError('Studio assets differ from the build manifest. Rebuild them.')
        for entry in manifest['files']:
            data = (assets / entry['path']).read_bytes()
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise RuntimeError('Studio asset integrity mismatch. Rebuild them.')
        previous = Path(self.build_lib) / 'mathkernel_studio' / 'assets'
        if previous.exists():
            shutil.rmtree(previous)
        super().run()


setup(cmdclass={'build_py': BuildStudio})
