"""Hostile fixtures: source scans are not theorem proofs; replay must fail closed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mathkernel import MathKernel, TrustLevel
from mathkernel.formal_audit import (
    AuditLimits, ComparatorRequest, FormalProjectSpec, FormalTarget,
    audit_lean_project, inventory, lean_probe, verify_with_comparator,
)
from mathkernel.formal_audit.models import ComparatorReport, STANDARD_AXIOMS
from mathkernel.formal_audit.source import (
    bounded_bytes, declarations, hash_file, imports, mask_lean, safe_relative,
    validate_comparator_config,
)
from mathkernel.formal_audit.access import authorized_root
from mathkernel.formal_audit import replay
from mathkernel.formal_audit.__main__ import main

PIN = 'leanprover/lean4:v4.34.0-rc2'


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding='utf-8')
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'submission'
    put(root, 'lean-toolchain', PIN + '\n')
    put(root, 'lake-manifest.json', '{"packages": []}')
    put(root, 'lakefile.toml', 'name = "example"\n')
    put(root, 'Definitions.lean', 'import Mathlib\nnamespace Demo\ndef x : Nat := 1\nend Demo\n')
    put(root, 'Solution.lean', 'import Definitions\nnamespace Demo\ntheorem target : x = 1 := by rfl\nend Demo\n')
    put(root, 'Challenge.lean', 'import Definitions\nnamespace Demo\ntheorem target : x = 1 := by sorry\nend Demo\n')
    return root


@pytest.fixture
def spec():
    return FormalProjectSpec(targets=(FormalTarget(module='Solution', declaration='Demo.target'),),
                             expected_toolchain=PIN)


def config(spec):
    return dict(challenge_module='Challenge', solution_module='Solution', enable_nanoda=True,
                theorem_names=[t.declaration for t in spec.targets], permitted_axioms=list(STANDARD_AXIOMS))


@pytest.mark.parametrize('source', [
    '-- sorry\ndef t := 1\n', '/- nested /- sorry -/ admit -/\ndef t := 1',
    'def s := "sorry \\" still a string"', 'def s := r##"sorry " nested"##',
    'def «sorry» := 1',
])
def test_literals_and_comments_are_not_placeholders(source):
    assert 'sorry' not in mask_lean(source)
    assert len(source) == len(mask_lean(source))
    assert source.count('\n') == mask_lean(source).count('\n')


@pytest.mark.parametrize('source', ['/- x', 'def x := "x', 'def «x'])
def test_unterminated_syntax_fails_closed(source):
    with pytest.raises(ValueError):
        mask_lean(source)


def test_multiline_imports_and_namespace_index():
    text = 'import A B\npublic import C\nnamespace Demo\nsection unnamed\ntheorem target : True := by trivial\nend unnamed\nend Demo'
    masked = mask_lean(text)
    assert imports(masked) == ('A', 'B', 'C')
    assert declarations(masked)['Demo.target'] == 'theorem'


def test_static_scan_never_runs_code_and_keeps_challenge_outside_closure(project, spec, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('source scan ran a process'))
    scan = audit_lean_project(project, spec)
    assert scan.trust == 'unknown' and scan.theorem_verification == 'not_run'
    assert scan.target_import_closure == ('Definitions', 'Solution')
    assert scan.unresolved_imports == ('Mathlib',)
    assert [f.path for f in scan.findings if f.code == 'PLACEHOLDER'] == ['Challenge.lean']
    assert all(f.scope == 'source_only' for f in scan.findings)


def test_imported_placeholder_is_a_warning_not_a_formal_refutation(project, spec):
    put(project, 'Solution.lean', 'import Challenge\nnamespace Demo\ntheorem target := Demo.target\nend Demo')
    put(project, 'audit.json', json.dumps(config(spec)))
    scan = audit_lean_project(project, spec.model_copy(update={'comparator_configs': ('audit.json',)}))
    assert any(f.code == 'CHALLENGE_IMPORTED' for f in scan.findings)
    assert scan.trust == 'unknown'


def test_target_definition_is_not_proof(project, spec):
    put(project, 'Solution.lean', 'namespace Demo\ndef target : Prop := True\nend Demo')
    assert 'TARGET_NOT_THEOREM' in {f.code for f in audit_lean_project(project, spec).findings}


def test_fingerprint_pins_all_nonexcluded_files(project, spec):
    first = inventory(project)[2]
    put(project, '.lake/build/garbage', 'irrelevant compiled output')
    assert inventory(project)[2] == first
    put(project, 'README.md', 'changed non-Lean source')
    assert inventory(project)[2] != first
    scan = audit_lean_project(project, spec.model_copy(update={'expected_source_sha256': first}))
    assert 'SOURCE_PIN_MISMATCH' in {f.code for f in scan.findings}


@pytest.mark.parametrize('pin', ['stable', '', 'leanprover/lean4:v4.33.0'])
def test_toolchain_missing_or_mismatch(project, spec, pin):
    put(project, 'lean-toolchain', pin)
    assert 'TOOLCHAIN_MISMATCH' in {f.code for f in audit_lean_project(project, spec).findings}


@pytest.mark.parametrize('lock,code', [
    ('{"packages":[{"type":"git","rev":"main"}]}', 'DEPENDENCY_UNPINNED'),
    ('{"packages":{}}', 'LOCKFILE_INVALID'), ('{', 'LOCKFILE_INVALID'),
    ('[]', 'LOCKFILE_INVALID'),
])
def test_lockfiles_fail_closed(project, lock, code):
    put(project, 'lake-manifest.json', lock)
    assert code in {f.code for f in audit_lean_project(project).findings}


def test_pinned_dependency_is_recorded(project):
    put(project, 'lake-manifest.json', json.dumps({'packages': [{'type':'git','name':'mathlib','rev':'1'*40}]}))
    scan = audit_lean_project(project)
    assert scan.dependency_pins[0]['rev'] == '1'*40
    assert not any(f.code == 'DEPENDENCY_UNPINNED' for f in scan.findings)


@pytest.mark.parametrize('name', ['', '.', '..', '../x', '/x', 'a//b', 'a/./b', 'a\\b', 'C:/a', 'x\n'])
def test_unsafe_relative_paths(name):
    with pytest.raises(ValueError): safe_relative(name)


@pytest.mark.parametrize('name', ['A;run', 'A\nimport B', '../X', '«Foo»', 'A B', ''])
def test_lean_identifier_injection_rejected(name):
    with pytest.raises(ValidationError): FormalTarget(module=name, declaration='x')


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink/FIFO fixture")
def test_symlink_and_fifo_rejected(project, tmp_path):
    path = project / 'escape.lean'
    path.symlink_to(tmp_path)
    with pytest.raises(ValueError, match='Symlink'): inventory(project)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match='regular'): inventory(project)


@pytest.mark.parametrize('limits', [dict(max_files=1), dict(max_file_bytes=1), dict(max_total_bytes=1)])
def test_budget_abort_is_not_partial_success(project, limits):
    with pytest.raises(ValueError): inventory(project, AuditLimits(**limits))


def test_findings_limit_marks_incomplete(project):
    put(project, 'Many.lean', '\n'.join('theorem t%s : True := by sorry' % i for i in range(10)))
    scan = audit_lean_project(project, limits=AuditLimits(max_findings=1))
    assert scan.status == 'incomplete'
    assert scan.findings[-1].code == 'FINDINGS_TRUNCATED'


def test_kernel_override_and_elaborator_flagged(project):
    put(project, 'Danger.lean', 'set_option debug.skipKernelTC true\nrun_elab pure ()\n')
    scan = audit_lean_project(project)
    assert {'KERNEL_CHECK_OVERRIDE', 'ELABORATOR_EXTENSION'} <= {f.code for f in scan.findings}


@pytest.mark.parametrize('mutation', [
    {'enable_nanoda': False}, {'theorem_names': []}, {'theorem_names': [{}]},
    {'theorem_names': ['Demo.target','Demo.target']}, {'theorem_names':['Other.target']},
    {'permitted_axioms': ['sorryAx']}, {'permitted_axioms': [{}]},
    {'external_kernels': {'nanoda': ['echo','accepted']}}, {'definition_names': ['x']},
    {'challenge_module':'Solution'}, {'solution_module':'Bad; command'},
])
def test_untrusted_comparator_policies_rejected(spec, mutation):
    candidate = config(spec); candidate.update(mutation)
    assert validate_comparator_config(candidate, spec)


def test_good_config_and_probe(spec):
    assert not validate_comparator_config(config(spec), spec)
    assert '#print axioms Demo.target' in lean_probe(spec)
    with pytest.raises(ValueError): lean_probe(FormalProjectSpec())


def test_read_allowlist_is_required_and_contains_children(project, tmp_path):
    assert authorized_root(str(project), (str(tmp_path),)) == project
    with pytest.raises(PermissionError): authorized_root(str(project), ())
    other = tmp_path / 'other'; other.mkdir()
    with pytest.raises(PermissionError): authorized_root(str(other), (str(project),))


def test_submission_lakefile_lean_is_never_staged(project, tmp_path):
    put(project, 'lakefile.lean', 'run_elab panic! "not a solution module"')
    destination = tmp_path / 'stage'; destination.mkdir()
    replay._copy_inventory(project, destination, inventory(project)[1], only_lean=True)
    assert (destination / 'Solution.lean').exists()
    assert not (destination / 'lakefile.lean').exists()
    assert not (destination / 'lakefile.toml').exists()


def test_staging_rechecks_source_hash(project, tmp_path):
    files = inventory(project)[1]
    put(project, 'Solution.lean', 'changed')
    with pytest.raises(ValueError, match='changed'):
        replay._copy_inventory(project, tmp_path / 'stage', files, only_lean=True)


@pytest.fixture
def replay_request(project, tmp_path, spec):
    if sys.platform != 'linux':
        pytest.skip('Comparator replay is Linux-only')
    reference = tmp_path / 'reference'
    put(reference, 'lean-toolchain', PIN)
    put(reference, 'lakefile.toml', 'name = "trusted"\n')
    put(reference, 'lake-manifest.json', '{"packages": []}')
    put(reference, 'Challenge.lean', (project/'Challenge.lean').read_text())
    put(reference, 'Definitions.lean', (project/'Definitions.lean').read_text())
    put(reference, 'config.json', json.dumps(config(spec)))
    tools = {}
    for name in ('lean','lake','comparator','landrun','lean4export','nanoda','systemd_run','systemctl','env'):
        path = put(tmp_path/'tools/bin', name, '#!/bin/sh\nexit 0\n'); path.chmod(0o700)
        tools[name] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    put(tmp_path/'tools', 'lib/lean/Init.olean', 'fixture-not-real-lean')
    return ComparatorRequest(submission_root=str(project), reference_root=str(reference),
        reference_tree_sha256=inventory(reference, include_dependencies=True)[2], config='config.json',
        tools=tools, spec=spec)


def test_execution_requires_authorization(replay_request, monkeypatch):
    monkeypatch.setattr(replay, 'run_bounded', lambda *a, **k: pytest.fail('unauthorized execution'))
    assert verify_with_comparator(replay_request).status == 'blocked'


def test_root_is_blocked(replay_request, monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    report = verify_with_comparator(replay_request, authorize_execution=True)
    assert report.status == 'blocked' and 'root' in report.errors[0]


def test_reference_pin_mismatch_prevents_execution(replay_request, monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(replay, 'run_bounded', lambda *a, **k: pytest.fail('pin mismatch executed'))
    bad = replay_request.model_copy(update={'reference_tree_sha256':'0'*64})
    assert 'fingerprint' in verify_with_comparator(bad, authorize_execution=True).errors[0]


def test_changed_binary_pin_is_blocked(replay_request, monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    Path(replay_request.tools.nanoda.path).write_text('changed')
    assert 'fingerprint' in verify_with_comparator(replay_request, authorize_execution=True).errors[0]


@pytest.mark.parametrize('returncode,status,trust', [(0,'accepted','formal'), (1,'not_accepted','unknown')])
def test_mocked_checker_contract_receipts_not_real_lean_execution(replay_request, monkeypatch, returncode,status,trust):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    commands=[]
    def mocked_run(command, **kw):
        commands.append(command)
        if command[-1] == '--version':
            return dict(status='finished',returncode=0,stdout='Lean (version 4.34.0-rc2, fixture)',stderr='',elapsed_seconds=0.01)
        if command[1:3] == ['--user','stop']:
            return dict(status='finished',returncode=0,stdout='',stderr='',elapsed_seconds=0.01)
        assert '--property=RestrictAddressFamilies=~AF_UNIX' in command
        assert '--property=PrivateNetwork=yes' in command
        assert '-i' in command  # clear the systemd manager environment
        assert not any('cache get' in word for word in command)
        return dict(status='finished',returncode=returncode,stdout='mocked checker',stderr='',elapsed_seconds=0.01)
    monkeypatch.setattr(replay, 'run_bounded', mocked_run)
    report = verify_with_comparator(replay_request, authorize_execution=True)
    assert (report.status, report.trust) == (status, trust), report.errors
    assert report.semantic_alignment == 'not_established'
    assert len(commands) == (2 if returncode == 0 else 3)  # version, Comparator, cleanup on failure; never pre-build


def test_mocked_service_timeout_stops_unit(replay_request, monkeypatch):
    monkeypatch.setattr(os, 'geteuid', lambda: 1000)
    commands=[]
    def mocked(command, **kw):
        commands.append(command)
        if command[-1] == '--version':
            return dict(status='finished',returncode=0,stdout='Lean (version 4.34.0-rc2, fixture)',stderr='',elapsed_seconds=0)
        return dict(status='timeout' if len(commands)==2 else 'finished',returncode=-9 if len(commands)==2 else 0,
                    stdout='',stderr='',elapsed_seconds=0)
    monkeypatch.setattr(replay, 'run_bounded', mocked)
    report=verify_with_comparator(replay_request, authorize_execution=True)
    assert report.status == 'timeout' and report.trust == 'unknown'
    assert commands[-1][1:3] == ['--user','stop']


def test_real_bounded_child_normal(tmp_path):
    run=replay.run_bounded([sys.executable,'-c','print("child")'],cwd=tmp_path,env={},timeout=3,output_limit=1024)
    assert run['status']=='finished' and run['returncode']==0 and run['stdout']=='child\n'


@pytest.mark.parametrize('script,expected', [('import time; time.sleep(20)','timeout'),
    ('import sys; sys.stdout.write("x"*1000000); sys.stdout.flush()','output_limit')])
def test_real_child_resource_limits(tmp_path, script, expected):
    run=replay.run_bounded([sys.executable,'-c',script],cwd=tmp_path,env={},timeout=0.3 if expected=='timeout' else 3,output_limit=1024)
    assert run['status']==expected
    assert len(run['stdout'])+len(run['stderr']) <= 1024
    assert run['elapsed_seconds'] < 3


def test_invalid_receipt_cannot_claim_formal(spec):
    with pytest.raises(ValidationError):
        ComparatorReport(status='blocked',trust='formal',reference_tree_sha256='0'*64,
                         targets=spec.targets,permitted_axioms=STANDARD_AXIOMS)


def test_facade_derivation_trust_and_discovery(project, spec):
    kernel=MathKernel()
    result=kernel.formal_project_audit(str(project), spec.model_dump(mode='json'))
    assert result.ok and result.trust==TrustLevel.UNKNOWN
    assert result.derivation and result.derivation[0].trust==TrustLevel.UNKNOWN
    assert not result.evidence_bundle.proof
    probe=kernel.formal_project_probe(spec.model_dump(mode='json'))
    assert probe.data['checked'] is False and probe.trust==TrustLevel.UNKNOWN
    caps=kernel.capability_query(domain='formal_project')['capabilities']
    assert len(caps)==2 and all(c['trust_levels']==['unknown'] for c in caps)
    assert all(c['operation']!='formal_project_verify' for c in caps)


def test_cli_and_no_overwrite(project, spec, tmp_path, capsys):
    specpath=put(tmp_path,'spec.json', spec.model_dump_json())
    output=tmp_path/'report.json'
    assert main(['inspect',str(project),'--spec',str(specpath),'--output',str(output)])==0
    assert json.loads(output.read_text())['trust']=='unknown'
    assert main(['inspect',str(project),'--output',str(output)])==2
    assert 'error' in capsys.readouterr().err


def test_mcp_audit_is_read_only_allowlisted_and_does_not_register_replay(project, spec, monkeypatch):
    """Registration contract using a tiny FastMCP test double, not live RPC."""
    import importlib
    import types
    class FakeMCP:
        def __init__(self, *a, **kw): self.tools={}
        def tool(self, fn=None, **kw):
            def register(f): self.tools[f.__name__]=f; return f
            return register(fn) if fn is not None else register
        def resource(self, *a, **kw): return lambda fn:fn
        def prompt(self, fn=None, **kw): return fn if fn is not None else lambda f:f
    monkeypatch.setitem(sys.modules,'fastmcp',types.SimpleNamespace(FastMCP=FakeMCP))
    monkeypatch.setenv('MATHKERNEL_FORMAL_PROJECT_ROOTS',str(project))
    loadspec=importlib.util.spec_from_file_location('_mathkernel_audit_mcp_contract', Path(__file__).parents[1]/'src/mathkernel_mcp/server.py')
    module=importlib.util.module_from_spec(loadspec)
    loadspec.loader.exec_module(module)
    assert 'math_formal_project_verify' not in module.mcp.tools
    result=module.mcp.tools['math_formal_project_audit'](str(project),spec.model_dump(mode='json'))
    assert result['ok'] and result['trust']=='unknown'
    denied=module.mcp.tools['math_formal_project_audit'](str(project.parent),None)
    assert not denied['ok'] and denied['trust']=='unknown'
    # The allowlist is captured at startup, not changed by a later env mutation.
    monkeypatch.setenv('MATHKERNEL_FORMAL_PROJECT_ROOTS',str(project.parent))
    assert not module.mcp.tools['math_formal_project_audit'](str(project.parent),None)['ok']


def test_proof_receipt_in_facade_preserves_only_the_verified_claim(replay_request, monkeypatch):
    """Mocked typed receipt projection, NOT a real proof-checker execution."""
    import mathkernel.formal_audit as api
    checked=ComparatorReport(status='accepted',trust='formal',independent_kernel='accepted',
        reference_tree_sha256=replay_request.reference_tree_sha256,
        source_sha256='1'*64,log_sha256='2'*64,returncode=0,
        tool_hashes={'mock':'3'*64},targets=replay_request.spec.targets,permitted_axioms=STANDARD_AXIOMS)
    monkeypatch.setattr(api,'verify_with_comparator',lambda *a,**k:checked)
    result=MathKernel().formal_project_verify(replay_request.model_dump(mode='json'),authorize_execution=True)
    assert result.trust==result.reconciled_trust()==TrustLevel.FORMAL
    assert result.evidence_bundle.proof[0].verified
    assert result.data['semantic_alignment']=='not_established'
    assert result.derivation[0].trust==TrustLevel.FORMAL
