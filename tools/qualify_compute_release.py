"""Inspect distributions and qualify base/per-extra installs outside the checkout.

Uses a supplied wheelhouse so dependency resolution is repeatable and offline.
No provider credentials or paid resources are needed. Outputs JSON measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

EXTRAS = ('base', 'compute', 'compute-modal', 'compute-runpod', 'compute-runpod-worker')


def inspect_distributions(wheel, sdist):
    with zipfile.ZipFile(wheel) as z:
        files = z.namelist()
        for name in ('lambda_models.py','lambda_api.py','provisioning.py','watchdog.py','schemas.py','replay.py',
                     'worker.py','gateway.py','LAMBDA.md','RELEASE.md'):
            assert 'mathkernel_compute/'+name in files, name
        assert any(n.endswith('entry_points.txt') for n in files)
        entries = z.read(next(n for n in files if n.endswith('entry_points.txt'))).decode()
        assert 'mathkernel-compute = mathkernel_compute.cli:main' in entries
        assert 'mathkernel-compute-gateway = mathkernel_compute.gateway:main' in entries
    with tarfile.open(sdist) as t:
        names = t.getnames()
        assert any(n.endswith('/tests/compute/test_lambda.py') for n in names)
        assert any(n.endswith('/examples/compute/measure_local.py') for n in names)
        assert any(n.endswith('/tools/qualify_compute_release.py') for n in names)
        assert not any(m.issym() or m.islnk() for m in t.getmembers())
    for name in files + names:
        parts = Path(name).parts
        assert not {'__pycache__', '.git', 'attempt-spool', 'quarantine', 'node_modules'} & set(parts), name
        assert not name.endswith(('.pyc','.sqlite3','.sqlite3-wal','.pem','.key')), name
        assert not any(p in {'id_rsa', 'id_ed25519', '.env', 'terminal.json', 'candidate.json', 'owner.json'} for p in parts), name
    return {'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
            'sdist_sha256': hashlib.sha256(sdist.read_bytes()).hexdigest(),
            'artifact_hygiene': 'passed: no bytecode, private keys, jobs, journals or symlinks'}


def qualify(wheel, sdist, wheelhouse, output):
    output.mkdir(parents=True, exist_ok=True)
    result = inspect_distributions(wheel, sdist)
    # Existing package, user-site and cloud credential environments cannot influence installs.
    env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'TMPDIR', 'TEMP', 'TMP') if k in os.environ}
    env.update(PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CONFIG_FILE=os.devnull)
    installs = []
    for label, extra in [(x,x) for x in EXTRAS] + [('sdist-compute','compute')]:
        directory = output/label
        if directory.exists():
            raise ValueError('Use an empty qualification output directory for clean installs')
        venv.EnvBuilder(with_pip=True).create(directory)
        python = directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        def run(args, **kwargs):
            return subprocess.run([str(python), *args], cwd=output, env=env, check=True, capture_output=True, **kwargs)
        requirement = str(sdist if label=='sdist-compute' else wheel) + (f'[{extra}]' if extra != 'base' else '')
        installed = run(['-m','pip','install','--no-index','--find-links',str(wheelhouse),requirement])
        (output/(label+'-install.log')).write_bytes(installed.stdout+installed.stderr)
        run(['-m','pip','check'])
        code = "from mathkernel import MathKernel; k=MathKernel(); r=k.parse('x+1'); assert r.ok; print('base mathematics OK')"
        run(['-c', code])
        run(['-c', "from pathlib import Path; import mathkernel_compute; assert Path(mathkernel_compute.__file__).resolve().is_relative_to(Path("+repr(str(directory))+"))"])
        if extra == 'base':
            run(['-c', "import sys; import mathkernel_compute; assert not {'modal','boto3','runpod','scipy','rfc8785'} & set(sys.modules)"])
        else:
            catalog = run(['-m','mathkernel_compute.cli','schemas'])
            assert 'LambdaProfile' in json.loads(catalog.stdout)['contracts']
            run(['-m','mathkernel_compute.cli','--help'])
            if extra=='compute-modal': run(['-c', "import importlib.metadata as m; assert m.version('modal')=='1.5.5'"])
            if extra.startswith('compute-runpod'): run(['-c', "import importlib.metadata as m; assert m.version('boto3')=='1.43.90'"])
            if extra=='compute-runpod-worker': run(['-c', "import importlib.metadata as m; assert m.version('runpod')=='1.12.0'; import runpod"])
        freeze = run(['-m','pip','freeze']).stdout
        (output/(label+'-freeze.txt')).write_bytes(freeze)
        installs.append({'extra': extra, 'distribution': 'sdist' if label=='sdist-compute' else 'wheel',
                         'dependency_resolution': 'clean-offline-wheelhouse', 'pip_check': 'passed'})
    # Run sdist examples against installed wheel code, with no checkout PYTHONPATH.
    unpack = output/'source'; unpack.mkdir(exist_ok=True)
    with tarfile.open(sdist) as t:
        t.extractall(unpack, filter='data')
    source = next(unpack.iterdir())
    python = output/'compute'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    example = source/'examples/compute/measure_local.py'
    measured = subprocess.run([str(python),str(example),'--authorize-local'], cwd=output, env=env,
                              check=True, capture_output=True, timeout=180)
    result['measurements'] = json.loads(measured.stdout)
    result['installs'] = installs
    (output/'qualification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', type=Path, required=True)
    p.add_argument('--sdist', type=Path, required=True)
    p.add_argument('--wheelhouse', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    qualify(*(getattr(args, x).resolve() for x in ('wheel','sdist','wheelhouse','output')))
