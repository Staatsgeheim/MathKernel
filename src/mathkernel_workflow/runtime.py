"""Transactional workflow owner with local policy, scoped approvals and bounded workers."""
from __future__ import annotations
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from .contracts import CATALOG, VERSION, canonical, digest, identifier
from .validation import compile_document, select_scope
from .worker import execute

def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def expires(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat().replace('+00:00', 'Z')

def expired(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')) <= datetime.now(timezone.utc)

def fresh(prefix):
    return prefix + '_' + secrets.token_hex(12)

@dataclass(frozen=True)
class LocalPolicy:
    max_nodes: int = 100
    wall_seconds: int = 60
    max_parallel: int = 2
    memory_mb: int = 0
    approval_seconds: int = 300
    enabled: bool = True

    def __post_init__(self):
        for key, lo, hi in [('max_nodes', 1, 100), ('wall_seconds', 1, 3600), ('max_parallel', 1, 8), ('memory_mb', 0, 65536), ('approval_seconds', 1, 900)]:
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('Invalid local policy ' + key)
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be boolean')
        if self.memory_mb and os.name != 'posix':
            raise ValueError('Address-space caps require a POSIX resource adapter')

class WorkflowRuntime:

    def __init__(self, directory, *, workspace_id='local', host_id=None, policy=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=448)
        self.lockfile = open(self.directory / 'owner.lock', 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.lockfile.seek(0)
                self.lockfile.write(b'0')
                self.lockfile.flush()
                self.lockfile.seek(0)
                msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.lockfile.close()
            raise RuntimeError('This state directory already has an active workflow owner')
        self.lock = threading.RLock()
        self.policy = policy or LocalPolicy()
        from mathkernel import __version__
        self.kernel_version = __version__
        self.closed = False
        self.active = {}
        self.threads = []
        self.db = sqlite3.connect(self.directory / 'workflow.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS items(kind TEXT, ref TEXT, scope TEXT, value TEXT, PRIMARY KEY(kind,ref))')
        stored = self.db.execute("SELECT value FROM items WHERE kind='identity'").fetchone()
        identity = json.loads(stored[0]) if stored else {'host': host_id or fresh('local'), 'workspace': workspace_id}
        if host_id and host_id != identity['host'] or workspace_id != identity['workspace']:
            self.db.close()
            self.lockfile.close()
            raise ValueError('State directory belongs to a different host/workspace')
        self.host_id = identifier(identity['host'])
        self.workspace_id = identifier(identity['workspace'])
        self.scope_key = self.host_id + ':' + self.workspace_id
        self._put('identity', 'identity', identity)
        for run in self._all('runs'):
            if run['execution'] in {'queued', 'running', 'cancel requested'}:
                run.update(execution='interrupted', resources='previous process ownership lost; no automatic replay', actions=[], warnings=run['warnings'] + ['Host restarted before completion; inspect saved partial results.'])
                for attempt in run['attempts']:
                    if attempt['state'] in {'pending', 'running'}:
                        attempt.update(state='interrupted', revision=attempt['revision'] + 1, observed_at=now())
                self._event(run, 'interrupted', 'Supervisor restarted; execution will not be replayed.')
                self._put('runs', run['run_ref'], run)
        self.db.commit()

    def _scope(self, scope):
        if scope.host_instance_id != self.host_id or scope.workspace_id != self.workspace_id:
            raise PermissionError('Host/workspace mismatch')

    def _put(self, kind, ref, value):
        self.db.execute('INSERT OR REPLACE INTO items VALUES(?,?,?,?)', (kind, ref, self.scope_key, canonical(value).decode()))

    def _get(self, kind, ref):
        identifier(ref)
        row = self.db.execute('SELECT scope,value FROM items WHERE kind=? AND ref=?', (kind, ref)).fetchone()
        if not row:
            raise KeyError('Reference unavailable')
        if row[0] != self.scope_key:
            raise PermissionError('Scope mismatch')
        return json.loads(row[1])

    def _all(self, kind):
        return [json.loads(row[0]) for row in self.db.execute('SELECT value FROM items WHERE kind=? AND scope=? ORDER BY rowid DESC', (kind, self.scope_key))]

    def _event(self, run, kind, message):
        run['revision'] += 1
        run['cursor'] = f"snapshot-{run['revision']}"
        run['observed_at'] = now()
        run['events'] = (run['events'] + [{'event_id': fresh('event'), 'revision': run['revision'], 'kind': kind, 'message': message, 'observed_at': run['observed_at']}])[-500:]

    def capabilities(self):
        return {'features': {'workflow_validate': True, 'workflow_execute': True, 'run_observe': True, 'compute_review': True, 'approval_interact': True, 'result_inspect': True}, 'extensions': {'contract': 'studio-workflow/1', 'scopes': ['workflow_outputs', 'selected_subgraph'], 'objects': False, 'subworkflows': True}}

    def command(self, action, payload, request_id, scope):
        identifier(request_id)
        self._scope(scope)
        if 'client_request_id' in payload and payload['client_request_id'] != request_id:
            raise ValueError('Request correlation mismatch')
        with self.lock:
            if self.closed:
                raise RuntimeError('Workflow owner closed')
            fingerprint = digest({'action': action, 'payload': payload})
            if action in {'workflow/submit', 'runs/action'}:
                try:
                    recorded = self._get('requests', request_id)
                except KeyError:
                    recorded = None
                if recorded:
                    if recorded['fingerprint'] != fingerprint:
                        raise PermissionError('Request ID reused for different intent')
                    return recorded['reply']
            try:
                if action in {'workflow/validate', 'workflow/plan', 'workflow/submit', 'approval/challenge'} and (sum((f.stat().st_size for f in self.directory.glob('workflow.sqlite3*'))) > 256 * 1024 * 1024 or self.db.execute('SELECT COUNT(*) FROM items').fetchone()[0] > 20000):
                    raise ValueError('Local state budget reached; archive the state directory before accepting new work')
                reply = self._command(action, copy.deepcopy(payload), request_id, scope)
                if action in {'workflow/submit', 'runs/action'}:
                    self._put('requests', request_id, {'fingerprint': fingerprint, 'reply': reply})
                self.db.commit()
            except (ValueError, PermissionError, KeyError) as exc:
                self.db.rollback()
                if action not in {'workflow/submit', 'runs/action'}:
                    raise
                plan_ref = payload.get('plan_ref')
                if not plan_ref:
                    try:
                        plan_ref = self._get('runs', payload['run_ref'])['plan_ref']
                    except KeyError:
                        raise exc
                reply = {'client_request_id': request_id, 'plan_ref': plan_ref, 'outcome': 'rejected', 'run_ref': None, 'message': str(exc)[:2000]}
                self._put('requests', request_id, {'fingerprint': fingerprint, 'reply': reply})
                self.db.commit()
                return reply
            except BaseException:
                self.db.rollback()
                raise
            if action == 'workflow/submit':
                cancel = threading.Event()
                self.active[reply['run_ref']] = cancel
                thread = threading.Thread(target=self._execute, args=(reply['run_ref'],), daemon=True)
                self.threads = [t for t in self.threads if t.is_alive()]
                try:
                    thread.start()
                    self.threads.append(thread)
                except RuntimeError:
                    self.active.pop(reply['run_ref'], None)
                    run = self._get('runs', reply['run_ref'])
                    run.update(execution='failed', resources='process not started', actions=[])
                    self._event(run, 'launch_failed', 'The host could not start its supervisor thread.')
                    self._put('runs', run['run_ref'], run)
                    self.db.commit()
            return copy.deepcopy(reply)

    def _plan_check(self, p):
        record = self._get('plans', p['plan_ref'])
        plan = record['public']
        if p['plan_digest'] != plan['digest'] or expired(plan['expires_at']) or record['policy'] != asdict(self.policy) or record.get('kernel_version') != self.kernel_version or (not self.policy.enabled) or plan['policy_denied']:
            raise PermissionError('Plan expired, changed, or denied by policy')
        return (record, plan)

    def _command(self, action, p, request_id, scope):
        if action == 'workflow/validate':
            ref = fresh('validation')
            try:
                compiled = compile_document(p['document'], p['binding'], self.policy.max_nodes, {v['public']['reference']: v for v in self._all('subworkflows') if v.get('kernel_version') == self.kernel_version})
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                return {'binding': p['binding'], 'validation_ref': ref, 'valid': False, 'diagnostics': [{'code': 'INVALID_DRAFT', 'severity': 'error', 'message': str(exc)[:2000], 'node_id': None, 'field': None}]}
            self._put('validations', ref, compiled)
            return {'binding': p['binding'], 'validation_ref': ref, 'valid': True, 'diagnostics': [{'code': 'KERNEL_CHECKS_AT_EXECUTION', 'severity': 'information', 'message': 'Typed graph accepted. Kernel execution still checks mathematical domains and preserves its own evidence.', 'node_id': None, 'field': None}]}
        if action == 'workflow/publish':
            compiled = self._get('validations', p['validation_ref'])
            if p['binding'] != compiled['binding']:
                raise ValueError('Frozen publication binding changed')
            operations, roots = select_scope(compiled, 'workflow_outputs', [])
            if len(roots) != 1:
                raise ValueError('Reusable workflows require exactly one desired output')
            content = digest({'operations': operations, 'root': roots[0], 'title': compiled['document']['identity']['title'], 'contract': VERSION, 'kernel_version': self.kernel_version})
            ref = 'saved_' + content[:32]
            public = {'reference': ref, 'revision': '1', 'digest': content, 'title': compiled['document']['identity']['title'], 'read_only': True, 'required_capabilities': [VERSION], 'boundary': [{'external_port': 'value', 'internal_node': roots[0], 'internal_port': 'value'}], 'description': 'Immutable, self-contained local workflow. No external inputs or automatic version rebinding. Publishing does not execute it.'}
            self._put('subworkflows', ref, {'public': public, 'operations': operations, 'root': roots[0], 'output_type': compiled['types'][roots[0]], 'kernel_version': self.kernel_version})
            return public
        if action == 'workflow/plan':
            compiled = self._get('validations', p['validation_ref'])
            if p['binding'] != compiled['binding']:
                raise ValueError('Validation binding changed')
            ops, roots = select_scope(compiled, p['scope'], p['selected_nodes'])
            ref = fresh('plan')
            policy = asdict(self.policy)
            body = {'binding': compiled['binding'], 'operations': ops, 'outputs': roots, 'scope': p['scope'], 'policy': policy, 'contract': VERSION, 'kernel_version': self.kernel_version}
            plan = {'binding': compiled['binding'], 'plan_ref': ref, 'digest': digest(body), 'expires_at': expires(self.policy.approval_seconds), 'scope': p['scope'], 'selected_nodes': [n['id'] for n in ops], 'outputs': [n + ':value' for n in roots], 'operations': [{'node_id': n['id'], 'operation_ref': 'workflow.' + n['method'], 'version': VERSION + '/' + self.kernel_version, 'engine': 'MathKernel', 'target': 'local-cpu'} for n in ops], 'assumptions': ['No additional context assumptions injected by workflow host.'], 'arithmetic': ['Kernel-native exact/symbolic semantics; inspect each returned MathResult.'], 'evidence_requirements': ['Preserve original MathResult status, trust, assumptions and per-claim support paths.'], 'resources': [{'name': 'Run wall-time limit', 'amount': str(self.policy.wall_seconds), 'unit': 'seconds', 'source': 'Enforced supervisor policy'}, {'name': 'Process address-space limit', 'amount': str(self.policy.memory_mb) if self.policy.memory_mb else None, 'unit': 'MiB', 'source': 'POSIX RLIMIT_AS' if self.policy.memory_mb else 'No memory hard cap configured'}], 'cost': {'amount': '0', 'currency': 'EUR', 'source': 'Local adapter provisions no billable provider', 'observed_at': now(), 'uncertainty': 'Provider charge is zero; local electricity and hardware costs are not measured.', 'exclusions': ['Electricity', 'Hardware ownership and utilization']}, 'exports': [], 'alternatives': [{'target': 'local-cpu', 'accepted': self.policy.enabled, 'reason': 'Operator-enabled local process policy' if self.policy.enabled else 'Operator disabled execution'}, {'target': 'remote-or-gpu', 'accepted': False, 'reason': 'No remote/GPU target adapter configured; no automatic fallback'}], 'authorization_required': True, 'policy_denied': not self.policy.enabled, 'warnings': ['Closing the browser does not stop this host-owned run.', 'The host executes only explicit adapters in an isolated process.']}
            self._put('plans', ref, {'public': plan, 'operations': ops, 'policy': policy, 'kernel_version': self.kernel_version})
            return plan
        if action in {'approval/challenge', 'approval/confirm', 'workflow/submit'}:
            record, plan = self._plan_check(p)
            if action == 'approval/challenge':
                ref = fresh('challenge')
                reply = {'challenge_ref': ref, 'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'expires_at': plan['expires_at'], 'disclosures': [f"Execute {len(record['operations'])} nodes on this computer for at most {self.policy.wall_seconds} seconds.", 'No remote provider, network export, automatic retry, or external charge is authorized.'], 'can_confirm': True}
                self._put('challenges', ref, {'public': reply, 'session': scope.session_id, 'used': False})
                return reply
            if action == 'approval/confirm':
                c = self._get('challenges', p['challenge_ref'])
                if c['session'] != scope.session_id or c['used'] or c['public']['plan_ref'] != plan['plan_ref'] or (c['public']['plan_digest'] != plan['digest']) or (expired(c['public']['expires_at'])):
                    raise PermissionError('Challenge unavailable or not bound to this session and plan')
                if p['decision'] not in {'approve', 'deny'}:
                    raise ValueError('Invalid decision')
                c['used'] = True
                self._put('challenges', p['challenge_ref'], c)
                approved = p['decision'] == 'approve'
                ref = fresh('authority') if approved else None
                if approved:
                    self._put('authorities', ref, {'session': scope.session_id, 'plan': plan['plan_ref'], 'digest': plan['digest'], 'used': False})
                return {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'authority_ref': ref, 'decision': 'approved' if approved else 'denied', 'expires_at': plan['expires_at']}
            authority = self._get('authorities', p['authority_ref'])
            if authority['session'] != scope.session_id or authority['plan'] != plan['plan_ref'] or authority['digest'] != plan['digest'] or authority['used']:
                raise PermissionError('Authority is not valid for this submission')
            if len(self.active) >= self.policy.max_parallel:
                raise ValueError('Local concurrency limit reached; no command accepted')
            authority['used'] = True
            self._put('authorities', p['authority_ref'], authority)
            ref = fresh('run')
            observed = now()
            run = {'run_ref': ref, 'binding': plan['binding'], 'plan_ref': plan['plan_ref'], 'revision': 0, 'cursor': 'snapshot-0', 'observed_at': observed, 'execution': 'queued', 'verification': 'not evaluated', 'artifacts': 'none', 'resources': 'process not started', 'cost': 'No provider charge; local usage unmeasured', 'warnings': [], 'attempts': [{'node_id': n['id'], 'attempt_id': fresh('attempt'), 'revision': 0, 'state': 'pending', 'result_ref': None, 'progress': None, 'progress_kind': 'indeterminate', 'observed_at': observed} for n in record['operations']], 'events': [], 'actions': ['cancel']}
            self._event(run, 'accepted', 'Frozen plan accepted; no browser scheduler is involved.')
            self._put('runs', ref, run)
            return {'client_request_id': request_id, 'plan_ref': plan['plan_ref'], 'outcome': 'accepted', 'run_ref': ref, 'message': 'Host accepted the immutable plan.'}
        if action == 'runs/action':
            run = self._get('runs', p['run_ref'])
            if p['action'] != 'cancel' or 'cancel' not in run['actions'] or p['revision'] != run['revision']:
                raise ValueError('Stale or unavailable run action')
            run['execution'] = 'cancel requested'
            run['actions'] = []
            self._event(run, 'cancel_requested', 'Cancellation requested; process exit is not yet confirmed.')
            self._put('runs', run['run_ref'], run)
            if run['run_ref'] in self.active:
                self.active[run['run_ref']].set()
            return {'client_request_id': request_id, 'plan_ref': run['plan_ref'], 'outcome': 'accepted', 'run_ref': run['run_ref'], 'message': 'Cancellation requested. Observe cleanup separately.'}
        raise KeyError('Unsupported command')

    def _execute(self, ref):
        ctx = multiprocessing.get_context('spawn')
        reader, writer = ctx.Pipe(duplex=False)
        process = None
        outcome = 'failed'
        message = 'Worker exited without a completion record'
        try:
            with self.lock:
                run = self._get('runs', ref)
                record = self._get('plans', run['plan_ref'])
                cancel = self.active[ref]
            if cancel.is_set():
                outcome = 'cancelled'
                message = 'Cancelled before process launch.'
                return
            process = ctx.Process(target=execute, args=(writer, record['operations'], self.policy.memory_mb), daemon=True)
            process.start()
            writer.close()
            start = time.monotonic()
            with self.lock:
                run = self._get('runs', ref)
                run.update(execution='running' if not cancel.is_set() else 'cancel requested', resources='local process running')
                self._event(run, 'started', 'Isolated kernel process started.')
                self._put('runs', ref, run)
                self.db.commit()
            while True:
                if cancel.is_set():
                    outcome = 'cancelled'
                    message = 'Cancellation stopped the owned process.'
                    break
                if time.monotonic() - start >= self.policy.wall_seconds:
                    outcome = 'timed out'
                    message = 'Host wall-time policy stopped execution; no mathematical disproof is implied.'
                    break
                if reader.poll(0.05):
                    try:
                        event = json.loads(reader.recv_bytes(1600000))
                    except EOFError:
                        break
                    if event['kind'] == 'finished':
                        outcome = 'completed'
                        message = 'All requested nodes returned successful kernel results.'
                        break
                    if event['kind'] == 'error':
                        message = event['message']
                        break
                    with self.lock:
                        run = self._get('runs', ref)
                        attempt = next((a for a in run['attempts'] if a['node_id'] == event['node']))
                        attempt['revision'] += 1
                        attempt['observed_at'] = now()
                        if event['kind'] == 'started':
                            attempt['state'] = 'running'
                        elif event['kind'] == 'result':
                            rref = fresh('result')
                            attempt.update(state='completed' if event['result']['ok'] else 'failed', result_ref=rref)
                            observation = {'binding': {'result_ref': rref, 'source_ref': ref, 'source_revision': str(attempt['revision']), 'run_ref': ref, 'document_id': run['binding']['document_id'], 'draft_revision': run['binding']['draft_revision'], 'node_id': attempt['node_id']}, 'admission': 'host', 'result': event['result'], 'claim_trust': event['claim_trust']}
                            self._put('results', rref, observation)
                            run['artifacts'] = 'result snapshots available'
                            run['verification'] = 'Per-node kernel results available; no aggregate trust promotion'
                            if not event['result']['ok']:
                                outcome = 'failed'
                                message = 'A kernel operation returned failure; downstream nodes were not executed.'
                        self._event(run, event['kind'], f"Node {attempt['node_id']}: {attempt['state']}")
                        self._put('runs', ref, run)
                        self.db.commit()
                elif not process.is_alive():
                    break
        except BaseException as exc:
            message = str(exc)[:2000]
        finally:
            if process is not None and process.pid is not None:
                process.join(0.2)
                if process.is_alive():
                    process.terminate()
                process.join(3)
                if process.is_alive():
                    process.kill()
                    process.join(3)
            reader.close()
            writer.close()
            with self.lock:
                run = self._get('runs', ref)
                run.update(execution=outcome, resources='owned process exited' if process is None or not process.is_alive() else 'cleanup unconfirmed', actions=[])
                for a in run['attempts']:
                    if a['state'] in {'pending', 'running'}:
                        a.update(state='skipped' if a['state'] == 'pending' else outcome, revision=a['revision'] + 1, observed_at=now())
                if outcome != 'completed':
                    run['warnings'].append(message)
                self._event(run, 'terminal', message)
                self._put('runs', ref, run)
                self.db.commit()
                self.active.pop(ref, None)

    def saved_descriptors(self):
        with self.lock:
            return [{'descriptor_id': v['public']['reference'], 'operation_ref': v['public']['reference'], 'operation_version': v['public']['revision'], 'schema_digest': v['public']['digest'], 'title': v['public']['title'], 'domain': 'Saved local workflows', 'description': v['public']['description'], 'availability': 'available' if v.get('kernel_version') == self.kernel_version else 'unavailable', 'composition': 'complete', 'input_ports': [], 'output_ports': [{'id': 'value', 'direction': 'output', 'label': 'value', 'type_ref': v['output_type'], 'cardinality': 'one', 'required': True, 'shape': None}], 'input_types': [], 'output_types': [v['output_type']], 'engines': ['MathKernel'], 'trust_levels': [], 'verification_methods': [], 'parameter_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}, 'coverage': 'Immutable self-contained workflow; expand and validate on the host.'} for v in self._all('subworkflows')]

    def read(self, kind, reference, offset, scope):
        self._scope(scope)
        if type(offset) is not int or offset < 0:
            raise ValueError('Invalid offset')
        with self.lock:
            if kind == 'subworkflows':
                return self._get(kind, reference)['public']
            if kind == 'requests':
                return self._get(kind, reference)['reply']
            if kind in {'runs', 'results'}:
                if reference:
                    return self._get(kind, reference)
                rows = self._all(kind)
                page = rows[offset:offset + 25]
                if kind == 'runs':
                    page = [{k: r[k] for k in ['run_ref', 'execution']} | {'document_id': r['binding']['document_id'], 'draft_revision': r['binding']['draft_revision']} for r in page]
                else:
                    page = [r['binding'] for r in page]
                return {kind: page, 'next_offset': offset + 25 if offset + 25 < len(rows) else None}
            raise KeyError('Unsupported resource')

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for cancel in self.active.values():
                cancel.set()
        for thread in self.threads:
            thread.join(self.policy.wall_seconds + 8)
        self.db.close()
        self.lockfile.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
