"""In-memory UI protocol fixtures. NOT a MathKernel workflow implementation.

No operators, schedulers, providers, verifiers, or real authorization are called.
All returned observations explicitly identify synthetic fixture values.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import secrets
import copy
from .source import canonical

def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
def expires(): return (datetime.now(timezone.utc)+timedelta(minutes=5)).isoformat().replace('+00:00', 'Z')
def fresh(prefix): return prefix+'_'+secrets.token_hex(8)

class FixtureWorkflowService:
    def __init__(self, fault='none'):
        self.fault = fault
        self.validations, self.plans, self.challenges, self.authorities, self.requests, self.runs = ({ } for _ in range(6))

    def capabilities(self):
        return {'features': {'workflow_validate': True, 'workflow_execute': True, 'run_observe': True, 'compute_review': True, 'approval_interact': True},
            'extensions': {'contract': 'studio-workflow/1', 'scopes': ['workflow_outputs', 'selected_subgraph'], 'objects': True, 'subworkflows': True}}

    def command(self, action, payload, request_id, scope):
        # Fixture models the host idempotency contract, including lost acknowledgements.
        if not hasattr(self, 'request_fingerprints'): self.request_fingerprints = {}
        fingerprint = (action, canonical(payload), scope)
        if request_id in self.requests:
            if self.request_fingerprints.get(request_id) != fingerprint:
                raise PermissionError('Request ID reused with a different command or scope')
            return copy.deepcopy(self.requests[request_id][0])
        try:
            return copy.deepcopy(self._command(action, copy.deepcopy(payload), request_id, scope))
        finally:
            if request_id in self.requests: self.request_fingerprints[request_id] = fingerprint

    def _command(self, action, payload, request_id, scope):
        if sum(map(len, (self.validations, self.plans, self.requests))) > 100: raise ValueError('Fixture budget exhausted; restart the test host')
        if action == 'workflow/validate':
            doc, binding = payload['document'], payload['binding']
            if doc.get('schema') != 'mk.studio/1' or binding['document_id'] != doc['identity']['document_id'] or binding['draft_revision'] != doc['authoring']['revision']: raise ValueError('Binding mismatch')
            ref = fresh('fixture_validation')
            self.validations[ref] = (binding, doc, (scope.host_instance_id, scope.workspace_id, scope.session_id))
            return {'binding': binding, 'validation_ref': ref, 'valid': True, 'diagnostics': [{'code': 'TEST_ONLY', 'severity': 'information', 'message': 'Synthetic validation; mathematical admissibility was not checked.', 'node_id': None, 'field': None}]}
        if action == 'workflow/plan':
            binding, doc, session = self.validations[payload['validation_ref']]
            if session != (scope.host_instance_id, scope.workspace_id, scope.session_id) or payload['binding'] != binding: raise PermissionError('Session/binding mismatch')
            ref = fresh('fixture_plan')
            nodes = payload['selected_nodes'] if payload['scope'] == 'selected_subgraph' else [n['id'] for n in doc['authoring']['nodes']]
            plan = {'binding': binding, 'plan_ref': ref, 'digest': hashlib.sha256(canonical(payload)).hexdigest(), 'expires_at': expires(),
                'scope': payload['scope'], 'selected_nodes': nodes, 'outputs': [f"{o['node_id']}:{o['port_id']}" for o in doc['authoring']['desired_outputs']],
                'operations': [{'node_id': n['id'], 'operation_ref': n['operation_ref'] or 'fixture.input', 'version': n['operation_version'] or 'fixture', 'engine': 'fixture-no-execution', 'target': 'fixture-no-resources'} for n in doc['authoring']['nodes'] if n['id'] in nodes],
                'assumptions': ['TEST ONLY: no mathematical meaning was resolved'], 'arithmetic': ['No arithmetic performed'], 'evidence_requirements': ['No evidence generated'],
                'resources': [{'name': 'Per-device GPU memory', 'amount': None, 'unit': 'bytes/device', 'source': 'Fixture unknown value'}],
                'cost': {'amount': None, 'currency': None, 'source': 'Synthetic fixture', 'observed_at': now(), 'uncertainty': 'Unknown; no money or resources are used', 'exclusions': []},
                'exports': [], 'alternatives': [{'target': 'fixture', 'accepted': True, 'reason': 'UI protocol demonstration only'}],
                'authorization_required': True, 'policy_denied': self.fault == 'policy_denied', 'warnings': ['TEST HOST: this plan cannot execute mathematics.']}
            if self.fault == 'stale_plan': plan['expires_at'] = '2000-01-01T00:00:00Z'
            self.plans[ref] = (plan, session)
            return plan
        if action in {'approval/challenge', 'approval/confirm', 'workflow/submit'}:
            plan, session = self.plans[payload['plan_ref']]
            if session != (scope.host_instance_id, scope.workspace_id, scope.session_id) or payload['plan_digest'] != plan['digest'] or plan['expires_at'] <= now() or plan['policy_denied']: raise PermissionError('Plan expired, denied, changed or not session bound')
            if action == 'approval/challenge':
                ref = fresh('fixture_challenge')
                c = {'challenge_ref': ref, 'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'expires_at': expires(), 'disclosures': ['Approve this synthetic fixture only. No execution, data export, provisioning, retries or spending.'], 'can_confirm': True}
                self.challenges[ref] = c
                return c
            if action == 'approval/confirm':
                c = self.challenges.pop(payload['challenge_ref'])
                if c['plan_ref'] != plan['plan_ref'] or c['expires_at'] <= now(): raise PermissionError('Challenge expired or mismatched')
                approved = payload['decision'] == 'approve'
                ref = fresh('fixture_authority') if approved else None
                if ref: self.authorities[ref] = (plan['plan_ref'], session)
                return {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'authority_ref': ref, 'decision': 'approved' if approved else 'denied', 'expires_at': plan['expires_at']}
            if self.authorities.get(payload['authority_ref']) != (plan['plan_ref'], session): raise PermissionError('Authority mismatch')
            if request_id in self.requests: return self.requests[request_id][0]
            ref = fresh('fixture_run')
            observed = now()
            self.runs[ref] = ({'run_ref': ref, 'binding': plan['binding'], 'plan_ref': plan['plan_ref'], 'revision': 1, 'cursor': 'fixture-cursor-1', 'observed_at': observed,
                'execution': 'output ready (synthetic)', 'verification': 'inconclusive (synthetic)', 'artifacts': 'fixture result available',
                'resources': 'cleanup unknown (synthetic)', 'cost': 'unknown (synthetic)', 'warnings': ['TEST HOST — result readiness does not establish cleanup or cost settlement.'],
                'attempts': [{'node_id': n, 'attempt_id': fresh('fixture_attempt'), 'revision': 1, 'state': 'synthetic output ready', 'result_ref': 'fixture-result', 'progress': None, 'progress_kind': 'indeterminate', 'observed_at': observed} for n in plan['selected_nodes']],
                'events': [{'event_id': 'fixture-event-1', 'revision': 1, 'kind': 'synthetic_observation', 'message': 'UI fixture; no computation occurred.', 'observed_at': observed}], 'actions': ['cancel']}, session)
            reply = {'client_request_id': request_id, 'plan_ref': plan['plan_ref'], 'outcome': 'accepted', 'run_ref': ref, 'message': 'Synthetic request accepted; no real execution.'}
            self.requests[request_id] = (reply, session)
            if self.fault == 'submission_unknown': raise RuntimeError('Simulated acknowledgement loss after recording')
            return reply
        if action == 'runs/action':
            run, session = self.runs[payload['run_ref']]
            if session != (scope.host_instance_id, scope.workspace_id, scope.session_id): raise PermissionError('Session mismatch')
            if request_id in self.requests: return self.requests[request_id][0]
            if payload['revision'] != run['revision'] or payload['action'] not in run['actions']: raise ValueError('Stale or unavailable run action')
            run = {**run, 'revision': run['revision']+1, 'execution': 'cancel requested (synthetic)', 'observed_at': now(), 'actions': []}
            self.runs[run['run_ref']] = (run, session)
            reply = {'client_request_id': request_id, 'plan_ref': run['plan_ref'], 'outcome': 'accepted', 'run_ref': run['run_ref'], 'message': 'Synthetic cancellation requested; cleanup is still unknown.'}
            self.requests[request_id] = (reply, session)
            return reply
        raise KeyError('Unsupported action')

    def read(self, kind, reference, offset, scope):
        if kind == 'requests':
            reply, session = self.requests[reference]
            if session != (scope.host_instance_id, scope.workspace_id, scope.session_id): raise PermissionError('Session mismatch')
            return reply
        if kind == 'runs':
            if reference:
                run, session = self.runs[reference]
                if session != (scope.host_instance_id, scope.workspace_id, scope.session_id): raise PermissionError('Session mismatch')
                return run
            runs = [r for r, session in self.runs.values() if session == (scope.host_instance_id, scope.workspace_id, scope.session_id)]
            page = runs[offset:offset+25]
            return {'runs': [{'run_ref': r['run_ref'], 'document_id': r['binding']['document_id'], 'draft_revision': r['binding']['draft_revision'], 'execution': r['execution']} for r in page], 'next_offset': offset+25 if offset+25 < len(runs) else None}
        if kind == 'objects': return {'objects': [{'object_id': 'fixture-matrix', 'revision': 'fixture-1', 'type_ref': 'Matrix', 'summary': 'Synthetic exact 2×2 matrix; no host object exists', 'shape': [2, 2]}] if offset == 0 else [], 'next_offset': None}
        if kind == 'subworkflows' and reference == 'fixture-subworkflow': return {'reference': reference, 'revision': 'fixture-1', 'digest': 'a'*64, 'title': 'Synthetic composition boundary', 'read_only': True, 'required_capabilities': ['fixture-only'], 'boundary': [{'external_port': 'value', 'internal_node': 'fixture-node', 'internal_port': 'value'}], 'description': 'Read-only fixture boundary. No executable subworkflow exists.'}
        raise KeyError('Unavailable reference')
