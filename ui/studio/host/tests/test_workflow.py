"""Optional facade invariants; synthetic protocol evidence, never a live runtime claim."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mathkernel_studio.services import SessionScope, parse_command
from mathkernel_studio.workflow_testing import FixtureWorkflowService
import test_host as host_tests

class WorkflowServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = FixtureWorkflowService()
        self.scope = SessionScope('fixture-host', 'fixture-workspace', 'session-a')
        self.doc = {'schema': 'mk.studio/1', 'identity': {'document_id': 'd', 'title': 'Draft'}, 'authoring': {'revision': 1, 'nodes': [], 'edges': [], 'desired_outputs': []}, 'presentation': {'revision': 0, 'node_positions': {}, 'groups': [], 'viewport': {'x': 0, 'y': 0, 'zoom': 1}}, 'host_binding': None}
        self.binding = {'document_id': 'd', 'draft_revision': 1, 'document_digest': 'a'*64}

    def command(self, action, payload, request='request-1', scope=None):
        return self.service.command(action, parse_command(json.dumps(payload).encode(), action), request, scope or self.scope)

    def plan(self):
        v = self.command('workflow/validate', {'document': self.doc, 'binding': self.binding})
        return self.command('workflow/plan', {'validation_ref': v['validation_ref'], 'binding': self.binding, 'scope': 'workflow_outputs', 'selected_nodes': []})

    def authorize(self, plan):
        p = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']}
        c = self.command('approval/challenge', p)
        a = self.command('approval/confirm', {**p, 'challenge_ref': c['challenge_ref'], 'decision': 'approve'})
        return c, a

    def test_lost_acknowledgement_reconciles_same_request_without_second_run(self):
        plan = self.plan(); _, a = self.authorize(plan)
        p = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'authority_ref': a['authority_ref'], 'client_request_id': 'request-submit'}
        self.service.fault = 'submission_unknown'
        with self.assertRaises(RuntimeError): self.command('workflow/submit', p, 'request-submit')
        self.assertEqual(len(self.service.runs), 1)
        result = self.service.read('requests', 'request-submit', 0, self.scope)
        self.assertEqual(result['outcome'], 'accepted')
        self.assertEqual(self.command('workflow/submit', p, 'request-submit'), result)
        self.assertEqual(len(self.service.runs), 1)

    def test_plan_digest_session_expiry_and_challenge_replay_are_enforced_by_fixture_host(self):
        p = self.plan(); c, a = self.authorize(p)
        payload = {'plan_ref': p['plan_ref'], 'plan_digest': p['digest'], 'challenge_ref': c['challenge_ref'], 'decision': 'approve'}
        with self.assertRaises(KeyError): self.command('approval/confirm', payload)
        with self.assertRaises(PermissionError): self.command('approval/challenge', {'plan_ref': p['plan_ref'], 'plan_digest': 'b'*64})
        with self.assertRaises(PermissionError): self.command('approval/challenge', {'plan_ref': p['plan_ref'], 'plan_digest': p['digest']}, scope=SessionScope('h', 'w', 'other-session'))
        self.service.plans[p['plan_ref']][0]['expires_at'] = '2000-01-01T00:00:00Z'
        with self.assertRaises(PermissionError): self.command('approval/challenge', {'plan_ref': p['plan_ref'], 'plan_digest': p['digest']})

    def test_policy_denial_cannot_be_overridden_by_confirmation(self):
        self.service.fault = 'policy_denied'; p = self.plan(); self.assertTrue(p['policy_denied'])
        with self.assertRaises(PermissionError): self.authorize(p)

    def test_cancellation_leaves_cleanup_and_cost_unknown(self):
        p = self.plan(); _, a = self.authorize(p)
        reply = self.command('workflow/submit', {'plan_ref': p['plan_ref'], 'plan_digest': p['digest'], 'authority_ref': a['authority_ref'], 'client_request_id': 'submit'}, 'submit')
        ref = reply['run_ref']; self.command('runs/action', {'run_ref': ref, 'revision': 1, 'action': 'cancel', 'client_request_id': 'cancel'}, 'cancel')
        run = self.service.read('runs', ref, 0, self.scope)
        self.assertIn('unknown', run['resources']); self.assertIn('unknown', run['cost']); self.assertIn('cancel requested', run['execution'])

    def test_parser_rejects_duplicate_keys_nonfinite_large_numbers_and_unknown_fields(self):
        for raw in (b'{"plan_ref":"p","plan_ref":"q","plan_digest":"x"}', b'{"__proto__":{}}', b'{"revision":9007199254740993}', b'{"revision":NaN}', b'{"revision":1e400}', b'{"approved":true}'):
            with self.assertRaises(ValueError): parse_command(raw, 'workflow/submit')

class OptionalFacadeTests(unittest.TestCase):
    setUp = host_tests.HostTests.setUp
    tearDown = host_tests.HostTests.tearDown
    request = host_tests.HostTests.request
    login = host_tests.HostTests.login
    def test_optional_service_is_explicit_and_origin_session_guarded(self):
        self.server.service = FixtureWorkflowService()
        self.assertEqual(self.request('/studio/api/workflow/validate', 'POST', '{}', **{'Origin': self.server.origin})[0], 401)
        self.login()
        status, _, raw = self.request(); self.assertEqual(status, 200); self.assertEqual(json.loads(raw)['payload']['extensions']['contract'], 'studio-workflow/1')
        self.assertEqual(self.request('/studio/api/workflow/validate', 'POST', '{}', **{'Origin': 'http://hostile.example'})[0], 403)
        self.assertEqual(self.request('/studio/api/workflow/validate', 'POST', '{}', **{'Origin': self.server.origin, 'Content-Type': 'application/json', 'X-Studio-Action': 'workflow/validate'})[0], 400)
        self.assertEqual(self.request('/studio/api/objects')[0], 200)

    def test_disconnect_revokes_session_and_viewer_origin_cannot_read_api(self):
        self.login()
        self.assertEqual(self.request('/studio/api/handshake', **{'Origin': 'null'})[0], 403)
        self.assertEqual(self.request('/studio/api/session/disconnect', 'POST', '{}', **{'Origin': self.server.origin, 'X-Studio-Action': 'disconnect'})[0], 200)
        self.assertEqual(self.request()[0], 401)

    def test_read_rate_limit_is_bounded(self):
        self.login(); self.server.rate_count = 120
        self.assertEqual(self.request()[0], 429)

if __name__ == '__main__': unittest.main()
