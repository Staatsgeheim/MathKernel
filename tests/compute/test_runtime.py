import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from mathkernel_compute import ComputeClient, ComputeRequest, CuboidParameters, ConvolutionParameters, ResourceRequirements, LocalPolicy
from mathkernel_compute.protocol import canonical
from mathkernel_compute.registry import execute, OPERATIONS
from mathkernel_compute.models import RemoteResultEnvelope


def request(bound='20', **kw):
    return ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound=bound), **kw)


def launch(client, req=None, key='request'):
    plan=client.plan(req or request())
    grant=client._authorize_local(plan.plan_id)
    return client.submit(plan_id=plan.plan_id,authorization_ref=grant.grant_id,client_request_id=key), plan, grant


class FaultProvider:
    """Test-only stateful provider: acceptance/visibility/cleanup faults are independent."""
    def __init__(self):
        self.accepted={};self.submissions=0;self.visibility=False;self.lost_ack=False
        self.revision=1;self.execution='RUNNING';self.resources='ACTIVE';self.expired=False
        self.cancellations=0;self.alter=False

    def submit(self,a):
        self.submissions+=1
        self.accepted[a.attempt_id]=a
        if self.lost_ack:raise TimeoutError('ack lost after acceptance')

    def observe(self,a):
        if not self.visibility:return {'revision':0,'execution':'SUBMISSION_UNKNOWN','resources':'ALLOCATION_INTENT'}
        return {'revision':self.revision,'execution':self.execution,'resources':self.resources}

    def cancel(self,a):
        self.cancellations+=1

    def fetch(self,a):
        if self.expired:raise FileNotFoundError('retention expired')
        output=execute(a.spec.bundle.request)
        if self.alter:output=()
        return canonical(RemoteResultEnvelope(attempt_id=a.attempt_id,workspace_id=a.workspace_id,
            execution_digest=a.execution_digest,bundle_digest=a.bundle_digest,operation=a.spec.bundle.request.operation,
            output_schema=OPERATIONS[a.spec.bundle.request.operation]['output_schema'],output=output,
            worker_claims={'trust':'formal','verified':True,'image_hash':'forged'}))


class RuntimeTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory()
    def tearDown(self):self.temp.cleanup()

    def test_real_exact_end_to_end_and_duplicate_submission(self):
        with ComputeClient(state_dir=self.temp.name) as c:
            j,p,g=launch(c)
            receipt=c.wait(j.job_id,timeout_s=30)
            self.assertEqual(receipt.execution,'RECEIVED',receipt)
            self.assertEqual(receipt.resources,'RELEASE_CONFIRMED')
            self.assertEqual(receipt.verification,'PENDING')
            with self.assertRaises(ValueError):c.accepted_result(j.job_id)
            self.assertEqual(c.verify(j.job_id).verification,'PASSED')
            result=c.accepted_result(j.job_id)
            self.assertEqual(result.trust.value,'exact')
            self.assertIn(['3','4'],result.data['output'])
            self.assertEqual(c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='request'),j)
            self.assertEqual(len(c.journal.all('attempts')),1)
            replay=c.export_replay(j.job_id)
            self.assertNotIn(g.grant_id.encode(),replay)
        with ComputeClient(state_dir=self.temp.name) as c:
            self.assertEqual(c.accepted_result(j.job_id).data,result.data)

    def test_real_numeric_end_to_end(self):
        req=ComputeRequest(operation='signal_convolve',parameters=ConvolutionParameters(left=('1.5','2.5'),right=('2','-1')),required_claim='numeric_convolution')
        with ComputeClient(state_dir=self.temp.name) as c:
            j,_,_=launch(c,req)
            self.assertEqual(c.wait(j.job_id,timeout_s=30).execution,'RECEIVED')
            self.assertEqual(c.verify(j.job_id).verification,'PASSED')
            result=c.accepted_result(j.job_id)
            self.assertEqual(result.trust.value,'numeric')
            self.assertEqual(result.data['output'],['3','3.5','-2.5'])

    def test_controller_process_exit_does_not_cancel_durable_worker(self):
        code="""from mathkernel_compute import ComputeClient,ComputeRequest,CuboidParameters
import sys
with ComputeClient(state_dir=sys.argv[1]) as c:
 p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='100')))
 g=c._authorize_local(p.plan_id)
 j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='restart')
 print(j.job_id,flush=True)
"""
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(x for x in sys.path if x)
        job_id=subprocess.check_output([sys.executable,'-c',code,self.temp.name],env=env,text=True,timeout=30).strip()
        with ComputeClient(state_dir=self.temp.name) as c:
            self.assertEqual(c.wait(job_id,timeout_s=30).execution,'RECEIVED')
            self.assertEqual(c.verify(job_id).verification,'PASSED')

    def test_real_cancel_and_timeout(self):
        with ComputeClient(state_dir=self.temp.name) as c:
            j,_,_=launch(c,request('2000'))
            c.cancel(j.job_id)
            status=c.wait(j.job_id,timeout_s=30)
            self.assertEqual(status.execution,'CANCELLED')
            self.assertEqual(status.resources,'RELEASE_CONFIRMED')
            j,_,_=launch(c,request(resources=ResourceRequirements(execution_timeout_ms=100)),key='timeout')
            status=c.wait(j.job_id,timeout_s=30)
            self.assertEqual(status.execution,'TIMED_OUT')
            self.assertNotEqual(status.verification,'FAILED')
            self.assertEqual(status.resources,'RELEASE_CONFIRMED')

    def test_verifier_timeout_is_inconclusive(self):
        with ComputeClient(state_dir=self.temp.name) as c:
            j,_,_=launch(c,request(resources=ResourceRequirements(verification_timeout_ms=100)))
            self.assertEqual(c.wait(j.job_id,timeout_s=30).execution,'RECEIVED')
            status=c.verify(j.job_id)
            self.assertEqual(status.verification,'INCONCLUSIVE')
            self.assertIsNone(status.accepted_result_ref)

    def test_lost_ack_eventual_visibility_and_failed_cleanup_keep_reservation(self):
        fake=FaultProvider();fake.lost_ack=True
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            j,_,_=launch(c,request('5'))
            self.assertEqual(c.status(j.job_id).execution,'SUBMISSION_UNKNOWN')
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            c.reconcile(j.job_id);self.assertEqual(fake.submissions,1)
            fake.visibility=True;fake.execution='RECEIVED';fake.resources='CLEANUP_UNKNOWN';fake.revision=3
            c.reconcile(j.job_id)
            self.assertEqual(c.verify(j.job_id).verification,'PASSED')
            self.assertEqual(c.status(j.job_id).resources,'CLEANUP_UNKNOWN')
            with self.assertRaises(PermissionError):launch(c,key='blocked')
            fake.resources='RELEASE_CONFIRMED';fake.revision=4;c.reconcile(j.job_id)
            self.assertEqual(c.status(j.job_id).resources,'RELEASE_CONFIRMED')
            self.assertEqual(fake.submissions,1)

    def test_out_of_order_expired_output_and_duplicate_candidate(self):
        fake=FaultProvider();fake.visibility=True
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            j,_,_=launch(c,request('5'))
            fake.execution='RECEIVED';fake.resources='RELEASE_CONFIRMED';fake.revision=4;fake.expired=True
            status=c.reconcile(j.job_id);self.assertEqual(status.artifacts,'UNAVAILABLE')
            fake.execution='RUNNING';fake.revision=2
            self.assertEqual(c.reconcile(j.job_id).execution,'RECEIVED')
            fake.expired=False;c.fetch(j.job_id)
            ref=c.status(j.job_id).candidate_ref
            c.fetch(j.job_id);self.assertEqual(c.status(j.job_id).candidate_ref,ref)
            fake.alter=True
            with self.assertRaises(ValueError):c.fetch(j.job_id)
            self.assertEqual(c.status(j.job_id).candidate_ref,ref)

    def test_policy_digest_expiry_scope_grant_and_request_conflicts(self):
        with ComputeClient(state_dir=self.temp.name,_executor=FaultProvider()) as c:
            p=c.plan(request())
            c.policy=LocalPolicy(enabled=False)
            with self.assertRaises(PermissionError):c._authorize_local(p.plan_id)
            c.policy=LocalPolicy()
            g=c._authorize_local(p.plan_id)
            with patch('mathkernel_compute.client.now_ms',return_value=p.expires_ms+1):
                with self.assertRaises(PermissionError):c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='expired')
            with self.assertRaises(KeyError):c.submit(plan_id=p.plan_id,authorization_ref='forged',client_request_id='bad')
            j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='ok')
            with self.assertRaises(PermissionError):c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='new')
            with self.assertRaises(PermissionError):c.submit(plan_id=p.plan_id,authorization_ref='different',client_request_id='ok')
            with tempfile.TemporaryDirectory() as other, ComputeClient(state_dir=other) as d:
                with self.assertRaises(KeyError):d.status(j.job_id)

    def test_journal_hash_schema_and_exclusive_owner(self):
        with ComputeClient(state_dir=self.temp.name) as c:
            p=c.plan(request())
            with self.assertRaises(RuntimeError):ComputeClient(state_dir=self.temp.name)
            c.journal.db.execute("UPDATE plans SET value=? WHERE ref=?",(b'{}',p.plan_id))
            with self.assertRaises(ValueError):c._plan(p.plan_id)
            c.journal.db.execute('PRAGMA user_version=999')
        with self.assertRaises(ValueError):ComputeClient(state_dir=self.temp.name)

    def test_planning_performs_no_submission_or_input_staging(self):
        fake=FaultProvider()
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            c.targets();c.plan(request())
            self.assertEqual(fake.submissions,0)
            self.assertEqual(c.journal.all('outbox'),[])
            self.assertEqual(list((Path(self.temp.name)/'quarantine').iterdir()),[])

    def test_process_tree_cleanup_preserves_unrelated_process(self):
        # Run the supervisor check in its own process so subreaper adoption is scoped.
        code="""import subprocess,sys,os,time
from mathkernel_compute.worker import enable_subreaper,terminate_owned
enable_subreaper()
unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
child=subprocess.Popen([sys.executable,'-c',"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);print('ready',flush=True);time.sleep(30)"],start_new_session=True,stdout=subprocess.PIPE,text=True)
try:
 assert child.stdout.readline().strip()=='ready'
 terminate_owned(child)
 assert unrelated.poll() is None
 try: os.killpg(child.pid,0)
 except ProcessLookupError: pass
 else: raise AssertionError('owned descendants survived')
 print('owned group gone; unrelated process alive')
finally:
 child.stdout.close()
 unrelated.kill();unrelated.wait()
"""
        env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(x for x in sys.path if x)
        value=subprocess.check_output([sys.executable,'-c',code],env=env,text=True,timeout=20)
        self.assertIn('unrelated process alive',value)

    def test_protocol_output_budget_fails_without_admitting_partial_output(self):
        with ComputeClient(state_dir=self.temp.name) as c:
            j,_,_=launch(c,request('2000',resources=ResourceRequirements(max_output_bytes=4096)))
            status=c.wait(j.job_id,timeout_s=30)
            self.assertEqual(status.execution,'FAILED')
            self.assertIsNone(status.accepted_result_ref)
            self.assertEqual(status.resources,'RELEASE_CONFIRMED')

    def test_concurrent_same_request_has_one_attempt_and_dispatch(self):
        from concurrent.futures import ThreadPoolExecutor
        fake=FaultProvider()
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            p=c.plan(request('5'));g=c._authorize_local(p.plan_id)
            def submit():return c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='same')
            with ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(lambda _:submit(),range(4)))
            self.assertEqual(len({j.job_id for j in results}),1)
            self.assertEqual(fake.submissions,1)

    def test_submit_transaction_rolls_back_before_external_effect(self):
        fake=FaultProvider()
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            p=c.plan(request('5'));g=c._authorize_local(p.plan_id)
            with patch.object(c,'_save',side_effect=OSError('simulated journal write failure')):
                with self.assertRaises(OSError):
                    c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='atomic')
            self.assertEqual(fake.submissions,0)
            self.assertEqual(c.journal.all('jobs'),[])
            self.assertEqual(c.journal.all('outbox'),[])
            self.assertIsNone(c.journal.get('budget_grants',g.grant_id)['job_id'])
            c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='atomic')
            self.assertEqual(fake.submissions,1)

    def test_crash_after_dispatch_is_reconciled_without_resubmission(self):
        fake=FaultProvider()
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            with patch.object(fake,'observe',side_effect=OSError('observation unavailable')):
                with self.assertRaises(OSError):launch(c,request('5'))
            job_id=c.journal.all('jobs')[0]['job_id']
        fake.visibility=True
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            self.assertEqual(c.status(job_id).execution,'RUNNING')
            self.assertEqual(fake.submissions,1)
            c.cancel(job_id)
            self.assertGreaterEqual(fake.cancellations,1)
            self.assertEqual(c.status(job_id).resources,'ACTIVE')

    def test_candidate_file_corruption_and_symlinks_are_refused(self):
        from mathkernel_compute.files import ArtifactStore
        with tempfile.TemporaryDirectory() as root:
            store=ArtifactStore(Path(root)/'quarantine')
            ref=store.put(b'original')
            (store.root/ref).write_bytes(b'changed')
            with self.assertRaises(ValueError):store.get(ref)
            owned=Path(root)/'owned-fixture';owned.write_bytes(b'fixture')
            target=store.root/('a'*64);target.symlink_to(owned)
            with self.assertRaises(OSError):store.get('a'*64)

    def test_same_revision_cannot_change_execution_facts(self):
        fake=FaultProvider();fake.visibility=True
        with ComputeClient(state_dir=self.temp.name,_executor=fake) as c:
            j,_,_=launch(c,request('5'))
            fake.execution='RECEIVED'
            with self.assertRaises(ValueError):c.reconcile(j.job_id)
            self.assertEqual(c.status(j.job_id).execution,'RUNNING')
