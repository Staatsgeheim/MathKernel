"""Offline fault tests for explicit installation and local proof execution."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

import pytest

from mathkernel import MathKernel
from mathkernel import lean_bootstrap as lean


@pytest.fixture(autouse=True)
def isolated_lean(monkeypatch, tmp_path):
    for key in ('MATHKERNEL_ELAN_HOME', 'MATHKERNEL_LEAN_WORKSPACE', 'MATHKERNEL_LEAN_BINARY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('MATHKERNEL_LEAN_CACHE', str(tmp_path / 'cache'))
    monkeypatch.delenv('MATHKERNEL_SKIP_LEAN_INSTALL', raising=False)
    lean._HEALTH_CACHE.clear()


def installed_layout(home, workspace):
    spec = lean._spec(home, workspace)
    for path in (spec.lean, spec.lake, spec.lean.parent.parent / 'lib/lean/Init.olean',
                 workspace / '.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    lean._copy_workspace(workspace)
    (workspace / 'lake-manifest.json').write_text('{}')
    return spec


@pytest.fixture
def fake_installer(monkeypatch):
    calls = []
    def install(home, timeout):
        calls.append('download')
        home.mkdir(parents=True)
    def lake_step(workspace, home, args, timeout):
        calls.append(args)
        if args == ['build']:
            installed_layout(home, workspace)
    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, 'Lean (version 4.33.0)\n', '')
    monkeypatch.setattr(lean, '_install_elan', install)
    monkeypatch.setattr(lean, '_run_lake', lake_step)
    monkeypatch.setattr(lean, '_run_process', run)
    return calls


def test_empty_binaries_and_mathlib_directory_are_not_ready(tmp_path):
    home, workspace = tmp_path / 'elan', tmp_path / 'workspace'
    spec = installed_layout(home, workspace)
    assert not spec.ready  # non-executable fixtures, despite a complete layout


@pytest.mark.parametrize('returncode,output', [(1, 'libInit_shared.so missing'), (0, 'Lean (version 4.32.0)')])
def test_health_checks_runtime_and_pinned_version(tmp_path, monkeypatch, returncode, output):
    spec = installed_layout(tmp_path / 'elan', tmp_path / 'workspace')
    monkeypatch.setattr(lean, '_run_process', lambda args, **kw: subprocess.CompletedProcess(args, returncode, output, output))
    assert not spec.ready


def test_health_requires_successful_mathlib_proof_replay(tmp_path, monkeypatch):
    spec = installed_layout(tmp_path / 'elan', tmp_path / 'workspace')
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0 if '--version' in args else 1,
                                           'Lean (version 4.33.0)', 'bad olean')
    monkeypatch.setattr(lean, '_run_process', run)
    assert not spec.ready
    assert list((spec.workspace / '.mathkernel-proofs').iterdir()) == []


def test_discovery_never_executes_an_elan_proxy(tmp_path, monkeypatch):
    home = lean.elan_home()
    for name in ('lean', 'lake'):
        path = lean._bin(home, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('proxy that might download')
    lean._copy_workspace(lean.workspace_dir())
    def forbidden(*args, **kwargs):
        pytest.fail('Discovery invoked an elan proxy')
    monkeypatch.setattr(lean, '_run_process', forbidden)
    assert lean.resolve_lean_toolchain() is None


@pytest.mark.parametrize('skip', [None, '0', '1'])
def test_math_and_mcp_startup_never_install(monkeypatch, skip):
    if skip is not None:
        monkeypatch.setenv('MATHKERNEL_SKIP_LEAN_INSTALL', skip)
    def forbidden(*args, **kwargs):
        pytest.fail('Implicit network/install operation')
    monkeypatch.setattr(lean, 'ensure_lean_toolchain', forbidden)
    monkeypatch.setattr(lean, '_download', forbidden)
    kernel = MathKernel()
    assert kernel.prove_equivalence('x+x', '2*x').status == 'verified'
    assert kernel.prove(kernel.parse('x+x=2*x').data['expr_id']).data['lean']['status'] == 'unavailable'
    with pytest.raises(FileNotFoundError, match='explicitly'):
        lean.run_lean_script('import Mathlib', 1)
    pytest.importorskip('fastmcp')
    from mathkernel_mcp import server
    called = []
    class Transport:
        def run(self): called.append(True)
    monkeypatch.setattr(server, 'mcp', Transport())
    server.main()
    assert called == [True]


def test_successful_setup_activates_only_a_healthy_generation(fake_installer):
    spec = lean.ensure_lean_toolchain(min_free_bytes=0)
    assert spec.ready and lean.resolve_lean_toolchain() == spec
    pointer = json.loads((lean.cache_root() / 'lean-active.json').read_text())
    assert spec.elan_home.parent.name == pointer['generation']
    assert not (spec.elan_home.parent / '.incomplete').exists()
    before = list(fake_installer)
    assert lean.ensure_lean_toolchain() == spec
    assert fake_installer == before


def test_failed_repair_preserves_active_install_and_removes_partial(fake_installer, monkeypatch):
    old = lean.ensure_lean_toolchain(min_free_bytes=0)
    pointer = (lean.cache_root() / 'lean-active.json').read_bytes()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ['lake', 'update'])
    monkeypatch.setattr(lean, '_run_lake', fail)
    with pytest.raises(subprocess.CalledProcessError):
        lean.ensure_lean_toolchain(force=True, min_free_bytes=0)
    assert (lean.cache_root() / 'lean-active.json').read_bytes() == pointer
    assert lean.resolve_lean_toolchain() == old
    assert list((lean.cache_root() / 'lean-installs').iterdir()) == [old.elan_home.parent]


def test_interrupted_generation_is_not_discovered_and_next_setup_cleans_it(fake_installer):
    abandoned = lean.cache_root() / 'lean-installs/install-crashed'
    abandoned.mkdir(parents=True)
    (abandoned / '.incomplete').touch()
    (abandoned / 'partial-download').write_bytes(b'partial')
    assert lean.resolve_lean_toolchain() is None
    assert lean.ensure_lean_toolchain(min_free_bytes=0).ready
    assert not abandoned.exists()


def test_health_failure_never_publishes_generation(fake_installer, monkeypatch):
    monkeypatch.setattr(lean, '_healthy', lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='health check failed'):
        lean.ensure_lean_toolchain(min_free_bytes=0)
    assert not (lean.cache_root() / 'lean-active.json').exists()
    assert list((lean.cache_root() / 'lean-installs').iterdir()) == []


def test_low_space_fails_before_download(fake_installer, monkeypatch):
    usage = type('Usage', (), {'free': 1})()
    monkeypatch.setattr(lean.shutil, 'disk_usage', lambda path: usage)
    with pytest.raises(RuntimeError, match='GiB free'):
        lean.ensure_lean_toolchain()
    assert 'download' not in fake_installer


def test_cache_failure_does_not_silently_build_from_source(fake_installer, monkeypatch):
    original = lean._run_lake
    def lake_step(workspace, home, args, timeout):
        if args == ['exe', 'cache', 'get']:
            raise subprocess.CalledProcessError(1, ['lake'])
        original(workspace, home, args, timeout)
    monkeypatch.setattr(lean, '_run_lake', lake_step)
    with pytest.raises(RuntimeError, match='--build-from-source'):
        lean.ensure_lean_toolchain(min_free_bytes=0)
    assert ['build'] not in fake_installer
    assert lean.ensure_lean_toolchain(min_free_bytes=0, build_from_source=True).ready


def test_setup_time_budget_is_passed_to_each_step(fake_installer, monkeypatch):
    original = lean._run_lake
    budgets = []
    def lake_step(workspace, home, args, timeout):
        budgets.append(timeout)
        original(workspace, home, args, timeout)
    monkeypatch.setattr(lean, '_run_lake', lake_step)
    assert lean.ensure_lean_toolchain(min_free_bytes=0, timeout=10).ready
    assert all(0 < t <= 10 for t in budgets)
    assert budgets == sorted(budgets, reverse=True)


def test_custom_paths_are_not_overwritten(monkeypatch, tmp_path, fake_installer):
    workspace = tmp_path / 'operator-workspace'
    workspace.mkdir()
    original = workspace / 'lakefile.toml'
    original.write_text('operator config')
    monkeypatch.setenv('MATHKERNEL_LEAN_WORKSPACE', str(workspace))
    with pytest.raises(RuntimeError, match='not overwritten'):
        lean.ensure_lean_toolchain(force=True, min_free_bytes=0)
    assert original.read_text() == 'operator config'
    assert 'download' not in fake_installer


@pytest.mark.parametrize('kind', ['traversal', 'symlink', 'hardlink', 'oversize'])
def test_tar_extraction_rejects_unsafe_members(tmp_path, monkeypatch, kind):
    archive, dest = tmp_path / 'elan.tar.gz', tmp_path / 'out'
    dest.mkdir()
    with tarfile.open(archive, 'w:gz') as handle:
        member = tarfile.TarInfo('../escape' if kind == 'traversal' else 'installer')
        if kind in ('symlink', 'hardlink'):
            member.type = tarfile.SYMTYPE if kind == 'symlink' else tarfile.LNKTYPE
            member.linkname = '../escape'
        else:
            member.size = 3
        handle.addfile(member, io.BytesIO(b'abc'))
    if kind == 'oversize': monkeypatch.setattr(lean, 'MAX_EXTRACT_BYTES', 1)
    with pytest.raises(RuntimeError): lean._extract_archive(archive, dest)
    assert not (tmp_path / 'escape').exists()


def test_zip_extraction_rejects_traversal(tmp_path):
    archive, dest = tmp_path / 'elan.zip', tmp_path / 'out'
    dest.mkdir()
    with zipfile.ZipFile(archive, 'w') as handle: handle.writestr('../escape', 'bad')
    with pytest.raises(RuntimeError): lean._extract_archive(archive, dest)


def test_bounded_download(tmp_path, monkeypatch):
    monkeypatch.setattr(lean.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'12345'))
    monkeypatch.setattr(lean, 'MAX_DOWNLOAD_BYTES', 4)
    with pytest.raises(RuntimeError, match='size or time limit'):
        lean._download('https://fixture.invalid/archive', tmp_path / 'archive')


def test_cli_check_is_offline_and_reports_unavailable(monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('check attempted setup')
    monkeypatch.setattr(lean, 'ensure_lean_toolchain', forbidden)
    monkeypatch.setattr(sys, 'argv', ['mathkernel-lean-setup', '--check'])
    with pytest.raises(SystemExit, match='missing or unhealthy'):
        lean.main()


def test_lock_excludes_other_process_and_recovers_after_owner_exit():
    code = 'from mathkernel.lean_bootstrap import _setup_lock, cache_root\nwith _setup_lock(cache_root()): print("locked", flush=True)'
    with lean._setup_lock(lean.cache_root()):
        blocked = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
        assert blocked.returncode != 0 and 'installation lock' in blocked.stderr
    recovered = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=10)
    assert recovered.returncode == 0 and recovered.stdout.strip() == 'locked'
    crash = code.replace('print("locked", flush=True)', '__import__("os")._exit(12)')
    assert subprocess.run([sys.executable, '-c', crash], timeout=10).returncode == 12
    with lean._setup_lock(lean.cache_root()):
        pass  # An abrupt owner exit leaves no stale OS lock.


@pytest.mark.skipif(os.name == 'nt', reason='POSIX subprocess-group assertion')
def test_process_timeout_terminates_installer_children(tmp_path):
    child_file = tmp_path / 'child-started'
    escaped = tmp_path / 'child-escaped'
    child = ('import sys,time\nfrom pathlib import Path\n'
             'Path(sys.argv[1]).touch()\ntime.sleep(1.5)\nPath(sys.argv[2]).touch()')
    code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
            'subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2],sys.argv[3]])\ntime.sleep(60)')
    with pytest.raises(subprocess.TimeoutExpired):
        lean._run_process([sys.executable, '-c', code, child, str(child_file), str(escaped)], timeout=1)
    assert child_file.exists()
    # Observe behavior instead of /proc: test runners can use PID namespaces
    # whose PIDs do not refer to the same processes in the mounted /proc.
    time.sleep(1.6)
    assert not escaped.exists()


def test_default_binary_setting_still_uses_managed_install(fake_installer, monkeypatch):
    monkeypatch.setenv('MATHKERNEL_LEAN_BINARY', 'lean')
    spec = lean.ensure_lean_toolchain(min_free_bytes=0)
    assert lean.resolve_lean_toolchain() == spec


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_setup_budget_is_rejected(timeout):
    with pytest.raises(ValueError): lean.ensure_lean_toolchain(timeout=timeout)
