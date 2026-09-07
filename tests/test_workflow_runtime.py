"""Real-kernel integration and workflow authority/recovery tests."""
import copy
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
import unittest
from mathkernel_workflow import WorkflowRuntime, LocalPolicy
from mathkernel_workflow.contracts import CATALOG, binding

@dataclass
class Scope:
    host_instance_id: str
    workspace_id: str
    session_id: str

def node(key, method, params=None):
    c = CATALOG['workflow.' + method]
    import json
    return {'id': key, 'kind': 'operation', 'label': method, 'operation_ref': c['operation_ref'], 'operation_version': c['operation_version'], 'schema_digest': c['schema_digest'], 'parameter_drafts': {k: {'encoding': 'json_text' if not isinstance(v, str) else 'text', 'text': json.dumps(v) if not isinstance(v, str) else v} for k, v in (params or {}).items()}, 'input_bindings': {}}

def graph(matrix=False):
    nodes = [node('a', 'matrix_create', {'rows': [['2', '0'], ['0', '3']]}), node('b', 'matrix_det')] if matrix else [node('a', 'parse', {'expression': 'x^3 + 9007199254740993'}), node('b', 'differentiate', {'variable': 'x'})]
    return {'schema': 'mk.studio/1', 'identity': {'document_id': 'doc', 'title': 'Real test'}, 'authoring': {'revision': 1, 'nodes': nodes, 'edges': [{'id': 'edge', 'source_node': 'a', 'source_port': 'value', 'target_node': 'b', 'target_port': 'matrix_id' if matrix else 'expr_id'}], 'desired_outputs': [{'node_id': 'b', 'port_id': 'value'}]}, 'presentation': {'revision': 0, 'node_positions': {}, 'groups': [], 'viewport': {'x': 0, 'y': 0, 'zoom': 1}}, 'host_binding': None}

class WorkflowTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime = WorkflowRuntime(self.tmp.name)
        self.scope = Scope(self.runtime.host_id, self.runtime.workspace_id, 'session')

    def tearDown(self):
        self.runtime.close()
        self.tmp.cleanup()

    def call(self, action, payload, request='test'):
        return self.runtime.command(action, payload, request, self.scope)

    def plan(self, doc=None):
        doc = doc or graph()
        b = binding(doc)
        validation = self.call('workflow/validate', {'document': doc, 'binding': b})
        self.assertTrue(validation['valid'], validation)
        return self.call('workflow/plan', {'validation_ref': validation['validation_ref'], 'binding': b, 'scope': 'workflow_outputs', 'selected_nodes': []})

    def submit(self, plan, request='submit'):
        p = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']}
        c = self.call('approval/challenge', p)
        a = self.call('approval/confirm', {**p, 'challenge_ref': c['challenge_ref'], 'decision': 'approve'})
        payload = {**p, 'authority_ref': a['authority_ref'], 'client_request_id': request}
        return (self.call('workflow/submit', payload, request), payload)

    def terminal(self, ref):
        limit = time.monotonic() + 30
        while time.monotonic() < limit:
            run = self.runtime.read('runs', ref, 0, self.scope)
            if not run['actions'] and run['execution'] != 'cancel requested':
                return run
            time.sleep(0.05)
        self.fail('Run did not finish within test deadline')

    def test_real_expression_run_and_durable_results(self):
        plan = self.plan()
        reply, payload = self.submit(plan)
        run = self.terminal(reply['run_ref'])
        self.assertEqual(run['execution'], 'completed', run)
        self.assertEqual(run['resources'], 'owned process exited')
        self.assertEqual(len(run['attempts']), 2)
        first = self.runtime.read('results', run['attempts'][0]['result_ref'], 0, self.scope)
        self.assertIn('9007199254740993', str(first['result']))
        result = self.runtime.read('results', run['attempts'][1]['result_ref'], 0, self.scope)
        self.assertEqual(result['admission'], 'host')
        self.assertEqual(result['binding']['node_id'], 'b')
        self.assertIn('3', str(result['result']['data']))
        self.assertEqual(self.call('workflow/submit', payload, 'submit'), reply)
        self.runtime.close()
        self.runtime = WorkflowRuntime(self.tmp.name)
        self.scope.session_id = 'new-session'
        self.assertEqual(self.runtime.read('requests', 'submit', 0, self.scope), reply)
        self.assertEqual(self.runtime.read('results', result['binding']['result_ref'], 0, self.scope), result)

    def test_real_matrix_graph(self):
        reply, _ = self.submit(self.plan(graph(True)))
        run = self.terminal(reply['run_ref'])
        self.assertEqual(run['execution'], 'completed', run)
        result = self.runtime.read('results', run['attempts'][-1]['result_ref'], 0, self.scope)
        self.assertIn('6', str(result['result']['data']))

    def test_digest_tampering_and_missing_ports_are_refused(self):
        doc = graph()
        b = binding(doc)
        doc['authoring']['nodes'][0]['parameter_drafts']['expression']['text'] = 'x'
        self.assertFalse(self.call('workflow/validate', {'document': doc, 'binding': b})['valid'])
        doc = graph()
        doc['authoring']['edges'] = []
        self.assertFalse(self.call('workflow/validate', {'document': doc, 'binding': binding(doc)})['valid'])

    def test_wrong_types_unknown_operations_and_cycles_refused(self):
        for mutate in [lambda d: d['authoring']['edges'][0].update(target_port='wrong'), lambda d: d['authoring']['nodes'][0].update(operation_ref='execute_code'), lambda d: d['authoring']['nodes'][0].update(schema_digest='0' * 64)]:
            doc = graph()
            mutate(doc)
            self.assertFalse(self.call('workflow/validate', {'document': doc, 'binding': binding(doc)})['valid'])

    def test_host_enforces_policy_session_expiry_and_approval_consumption(self):
        plan = self.plan()
        self.runtime.policy = LocalPolicy(enabled=False)
        with self.assertRaises(PermissionError):
            self.call('approval/challenge', {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']})
        self.runtime.policy = LocalPolicy()
        p = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']}
        c = self.call('approval/challenge', p)
        self.scope.session_id = 'other'
        with self.assertRaises(PermissionError):
            self.call('approval/confirm', {**p, 'challenge_ref': c['challenge_ref'], 'decision': 'approve'})
        self.scope.session_id = 'session'
        a = self.call('approval/confirm', {**p, 'challenge_ref': c['challenge_ref'], 'decision': 'approve'})
        with self.assertRaises(PermissionError):
            self.call('approval/confirm', {**p, 'challenge_ref': c['challenge_ref'], 'decision': 'approve'})

    def test_cancellation_has_separate_process_cleanup(self):
        reply, _ = self.submit(self.plan())
        run = self.runtime.read('runs', reply['run_ref'], 0, self.scope)
        response = self.call('runs/action', {'run_ref': run['run_ref'], 'revision': run['revision'], 'action': 'cancel', 'client_request_id': 'cancel'}, 'cancel')
        self.assertEqual(response['outcome'], 'accepted')
        run = self.terminal(run['run_ref'])
        self.assertEqual(run['execution'], 'cancelled')
        self.assertEqual(run['resources'], 'owned process exited')

    def test_publish_and_execute_immutable_subworkflow(self):
        doc = graph()
        b = binding(doc)
        validation = self.call('workflow/validate', {'document': doc, 'binding': b})
        saved = self.call('workflow/publish', {'validation_ref': validation['validation_ref'], 'binding': b})
        descriptor = self.runtime.saved_descriptors()[0]
        outer = graph()
        outer['authoring']['nodes'] = [{'id': 'saved', 'kind': 'operation', 'label': 'Saved expression', 'operation_ref': saved['reference'], 'operation_version': saved['revision'], 'schema_digest': saved['digest'], 'parameter_drafts': {}, 'input_bindings': {}}, node('next', 'simplify')]
        outer['authoring']['edges'] = [{'id': 'edge', 'source_node': 'saved', 'source_port': 'value', 'target_node': 'next', 'target_port': 'expr_id'}]
        outer['authoring']['desired_outputs'] = [{'node_id': 'next', 'port_id': 'value'}]
        self.assertEqual(descriptor['schema_digest'], saved['digest'])
        reply, _ = self.submit(self.plan(outer))
        run = self.terminal(reply['run_ref'])
        self.assertEqual(run['execution'], 'completed', run)
        self.assertEqual(len(run['attempts']), 3)
        self.assertEqual(self.runtime.read('subworkflows', saved['reference'], 0, self.scope), saved)
        outer['authoring']['nodes'][0]['schema_digest'] = '0' * 64
        self.assertFalse(self.call('workflow/validate', {'document': outer, 'binding': binding(outer)})['valid'])

    def test_rejected_submission_is_durable_and_cannot_be_replayed_as_new_intent(self):
        plan = self.plan()
        payload = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest'], 'authority_ref': 'nonexistent', 'client_request_id': 'rejected'}
        reply = self.call('workflow/submit', payload, 'rejected')
        self.assertEqual(reply['outcome'], 'rejected')
        self.assertEqual(self.runtime.read('requests', 'rejected', 0, self.scope), reply)
        self.assertEqual(self.call('workflow/submit', payload, 'rejected'), reply)
        with self.assertRaises(PermissionError):
            self.call('workflow/submit', {**payload, 'authority_ref': 'changed'}, 'rejected')

    def test_timeout_is_not_a_mathematical_failure_claim(self):
        self.runtime.policy = LocalPolicy(wall_seconds=1)
        reply, _ = self.submit(self.plan())
        run = self.terminal(reply['run_ref'])
        self.assertEqual(run['execution'], 'timed out')
        self.assertEqual(run['verification'], 'not evaluated')
        self.assertEqual(run['resources'], 'owned process exited')

    def test_every_advertised_operation_executes_with_real_kernel_results(self):
        doc=graph()
        nodes=[node('expr','parse',{'expression':'x^2-1'}),node('matrix','matrix_create',{'rows':[['2','0'],['0','3']]}),node('rhs','matrix_create',{'rows':[['4'],['9']]})]
        edges=[]
        from mathkernel_workflow.contracts import OPERATIONS
        for method,op in OPERATIONS.items():
            if method in {'parse','matrix_create'}:continue
            params={'variable':'x'} if method in {'differentiate','integrate','solve'} else {}
            nodes.append(node(method,method,params))
            for port,kind in op.inputs:
                source='expr' if kind=='Expression' else 'rhs' if port=='rhs_id' else 'matrix'
                edges.append({'id':method+'-'+port,'source_node':source,'source_port':'value','target_node':method,'target_port':port})
        doc['authoring'].update(nodes=nodes,edges=edges,desired_outputs=[{'node_id':n['id'],'port_id':'value'} for n in nodes])
        reply,_=self.submit(self.plan(doc));run=self.terminal(reply['run_ref'])
        self.assertEqual(run['execution'],'completed',run)
        self.assertEqual(len(run['attempts']),len(nodes))
        for attempt in run['attempts']:
            result=self.runtime.read('results',attempt['result_ref'],0,self.scope)
            self.assertTrue(result['result']['ok'],result)
            self.assertEqual(result['admission'],'host')

    def test_restart_marks_unfinished_run_without_reexecution(self):
        reply,_=self.submit(self.plan());run=self.terminal(reply['run_ref'])
        run.update(execution='cancel requested',actions=[])
        run['attempts'][0]['state']='running'
        self.runtime._put('runs',run['run_ref'],run);self.runtime.db.commit()
        before=self.runtime.db.execute("SELECT COUNT(*) FROM items WHERE kind='results'").fetchone()[0]
        self.runtime.close();self.runtime=WorkflowRuntime(self.tmp.name)
        recovered=self.runtime.read('runs',run['run_ref'],0,self.scope)
        self.assertEqual(recovered['execution'],'interrupted');self.assertEqual(recovered['actions'],[])
        self.assertFalse(self.runtime.active)
        self.assertEqual(recovered['attempts'][0]['state'], 'interrupted')
        self.assertEqual(self.runtime.db.execute("SELECT COUNT(*) FROM items WHERE kind='results'").fetchone()[0],before)

    def test_expired_or_different_kernel_plan_cannot_authorize(self):
        plan = self.plan()
        payload = {'plan_ref': plan['plan_ref'], 'plan_digest': plan['digest']}
        record = self.runtime._get('plans', plan['plan_ref'])
        record['kernel_version'] = 'changed-kernel'
        self.runtime._put('plans', plan['plan_ref'], record)
        self.runtime.db.commit()
        with self.assertRaises(PermissionError):
            self.call('approval/challenge', payload)
        record['kernel_version'] = self.runtime.kernel_version
        record['public']['expires_at'] = '2000-01-01T00:00:00Z'
        self.runtime._put('plans', plan['plan_ref'], record)
        self.runtime.db.commit()
        with self.assertRaises(PermissionError):
            self.call('approval/challenge', payload)

    def test_state_budget_rejection_reconciles_and_preserves_old_request(self):
        reply, payload = self.submit(self.plan())
        self.terminal(reply['run_ref'])
        self.runtime.db.executemany('INSERT INTO items VALUES(?,?,?,?)',
            [('budget', str(i), self.runtime.scope_key, '{}') for i in range(20001)])
        self.runtime.db.commit()
        self.assertEqual(self.call('workflow/submit', payload, 'submit'), reply)
        new_payload = {**payload, 'client_request_id': 'new-submit'}
        rejected = self.call('workflow/submit', new_payload, 'new-submit')
        self.assertEqual(rejected['outcome'], 'rejected')
        self.assertIn('budget', rejected['message'])
        self.assertEqual(self.runtime.read('requests', 'new-submit', 0, self.scope), rejected)

    def test_second_process_owner_is_refused(self):
        with self.assertRaises(RuntimeError):
            WorkflowRuntime(self.tmp.name)

    def test_scope_isolation(self):
        p = self.plan()
        other = Scope('other', self.scope.workspace_id, self.scope.session_id)
        with self.assertRaises(PermissionError):
            self.runtime.read('runs', None, 0, other)
        with self.assertRaises(PermissionError):
            self.runtime.command('approval/challenge', {'plan_ref': p['plan_ref'], 'plan_digest': p['digest']}, 'q', other)
if __name__ == '__main__':
    unittest.main()
