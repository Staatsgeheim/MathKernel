"""Slurm protocol/scheduler simulator tests, not a live-cluster qualification.

Fixtures emit explicit CLI records and execute the generated batch script only
when an allocation is granted. They also retain ambiguous submit/cleanup states.
"""
from datetime import datetime, timezone, timedelta
import os
import subprocess
import time
from unittest.mock import patch
import pytest
from mathkernel_compute.files import atomic_write, bounded_read
from mathkernel_compute.gateway import Gateway
from mathkernel_compute.protocol import canonical, decode, digest, parse
from mathkernel_compute.remote import SlurmConfig, WorkerConfig, RemoteUnavailable, GatewayRequest, AttemptRef
from mathkernel_compute.slurm import SlurmExecutor, parse_rows, scheduler_environment, END_STATES
from mathkernel_compute.models import RemoteResultEnvelope
from mathkernel_compute.verification import verify_candidate
from test_remote import remote_attempt


class Scheduler:
    def __init__(self):
        self.calls=[]; self.row=None; self.lost_ack=False; self.queue=True; self.account=True
        self.process=None; self.script=None; self.cancelled=0; self.submit_count=0; self.restarts='0'; self.exit_code='0:0'

    def command(self, argv, data=b''):
        self.calls.append((argv,data))
        name=argv[0].rsplit('/',1)[-1]
        if name=='sbatch':
            self.submit_count+=1
            self.script=data
            get=lambda flag: next(v.split('=',1)[1] for v in argv if v.startswith(flag+'='))
            self.directory=get('--chdir')
            self.row=['321',get('--job-name'),str(os.getuid()),'PENDING',
                      datetime.now(timezone.utc).replace(tzinfo=None,microsecond=0).isoformat(),get('--comment')]
            if self.lost_ack: raise RemoteUnavailable('ack lost after scheduler acceptance')
            return b'321;fixture\n'
        if name=='scancel':
            self.cancelled+=1
            return b''  # Acknowledgement deliberately does not terminate the job.
        if self.process is not None and self.process.poll() is not None:
            self.row[3]='COMPLETED' if self.process.returncode==0 else 'FAILED'
        if name=='squeue':
            if not self.queue: raise RemoteUnavailable('squeue unavailable')
            return ('|'.join(self.row)+'\n').encode() if self.row and self.row[3] in {'PENDING','RUNNING','COMPLETING'} else b''
        if name=='sacct':
            if not self.account: return b''
            return ('|'.join(self.row+[self.exit_code,'3','1','cpu=1,mem=1024M','Unknown','Unknown',self.restarts])+'\n').encode() if self.row else b''
        raise AssertionError(argv)

    def start(self, *, restarts='0'):
        self.row[3]='RUNNING'
        self.process=subprocess.Popen(['/bin/sh'],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            env={'PATH':os.defpath,'SLURM_JOB_ID':'321','SLURM_RESTART_COUNT':restarts},start_new_session=True)
        self.process.stdin.write(self.script);self.process.stdin.close()

    def close(self):
        if self.process is not None and self.process.poll() is None:
            atomic_write(__import__('pathlib').Path(self.directory)/'cancel',b'{}')
            self.process.wait(timeout=20)


@pytest.fixture
def slurm(tmp_path):
    config=SlurmConfig(partition='cpu',account='research',qos='short')
    executor=SlurmExecutor(tmp_path/'spool',config)
    scheduler=Scheduler()
    executor.command=scheduler.command
    attempt=remote_attempt(adapter='slurm')
    try: yield executor,scheduler,attempt
    finally: scheduler.close()


def test_login_node_only_stages_and_scheduler_runs_actual_worker(slurm):
    executor,scheduler,a=slurm
    executor.submit(a)
    directory=executor.directory(a.attempt_id)
    assert not (directory/'accepted').exists()
    assert executor.observe(a)['execution']=='QUEUED'
    argv,script=scheduler.calls[0]
    assert '--parsable' in argv and '--no-requeue' in argv and '--export=NIL' in argv
    assert '--mem=1024M' in argv and '--cpus-per-task=1' in argv
    assert '--partition=cpu' in argv and '--account=research' in argv
    assert b'slurm-supervise' in script
    assert b'cuboid_sweep' not in script
    scheduler.start(); scheduler.process.wait(timeout=30)
    observation=executor.observe(a)
    assert observation['execution']=='RECEIVED',observation
    assert observation['resources']=='RELEASE_CONFIRMED'
    assert observation['usage']['allocation_units']=='UNKNOWN'
    candidate=parse(RemoteResultEnvelope,executor.fetch(a))
    assert verify_candidate(a,candidate,'verification_test').outcome=='PASSED'


def test_ambiguous_sbatch_ack_is_reconciled_without_replacement(slurm):
    executor,scheduler,a=slurm;scheduler.lost_ack=True
    with pytest.raises(RemoteUnavailable): executor.submit(a)
    executor.submit(a)
    assert scheduler.submit_count==1
    assert executor.observe(a)['execution']=='QUEUED'
    restored=SlurmExecutor(executor.root,executor.config);restored.command=scheduler.command
    restored.submit(a)
    assert scheduler.submit_count==1
    assert restored.observe(a)['execution']=='QUEUED'


def test_missing_queue_and_delayed_accounting_never_fabricate_release(slurm):
    executor,scheduler,a=slurm;executor.submit(a)
    assert executor.observe(a)['resources']=='ACTIVE'
    scheduler.queue=False;scheduler.account=False
    obs=executor.observe(a)
    assert obs['resources']=='CLEANUP_UNKNOWN'
    assert obs['execution']=='SUBMISSION_UNKNOWN'
    with pytest.raises(RemoteUnavailable): executor.cancel(a)
    assert scheduler.cancelled==0


def test_scancel_ack_is_only_intent_and_filter_protects_reused_id(slurm):
    executor,scheduler,a=slurm;executor.submit(a)
    scheduler.row[3]='RUNNING'
    executor.cancel(a)
    assert scheduler.cancelled==1
    assert executor.observe(a)['resources']=='ACTIVE'
    cancel=[argv for argv,_ in scheduler.calls if argv[0].endswith('scancel')][0]
    assert '--name='+executor.name(a) in cancel
    assert '--user='+str(os.getuid()) in cancel
    assert cancel[-1]=='321'
    scheduler.row[1]='unrelated-job'
    with pytest.raises(RemoteUnavailable): executor.cancel(a)
    assert scheduler.cancelled==1


def test_slurm_submit_incarnation_change_blocks_cancel_and_duplicate_admission(slurm):
    executor,scheduler,a=slurm;executor.submit(a);executor.observe(a)
    stamp=datetime.fromisoformat(scheduler.row[4])-timedelta(seconds=1)
    scheduler.row[4]=stamp.isoformat()
    with pytest.raises(RemoteUnavailable,match='INCARNATION_CHANGED'): executor.cancel(a)
    with pytest.raises(RemoteUnavailable,match='INCARNATION_CHANGED'): executor.observe(a)
    assert scheduler.cancelled==0


@pytest.mark.parametrize('state,expected',list(END_STATES.items())[1:])
def test_scheduler_failure_states_are_distinct(slurm,state,expected):
    executor,scheduler,a=slurm;executor.submit(a);scheduler.row[3]=state
    obs=executor.observe(a)
    assert obs['execution']==expected
    assert obs['resources']=='RELEASE_CONFIRMED'
    assert obs['usage']['source']=='sacct'


def test_completion_without_candidate_is_not_math_success(slurm):
    executor,scheduler,a=slurm;executor.submit(a);scheduler.row[3]='COMPLETED'
    obs=executor.observe(a)
    assert obs['execution']=='SUBMISSION_UNKNOWN'
    assert obs['resources']=='RELEASE_CONFIRMED'
    with pytest.raises(FileNotFoundError): executor.fetch(a)


def test_worker_refuses_expired_queue_start_and_requeue(slurm):
    executor,scheduler,a=slurm
    a=a.model_copy(update={'start_deadline_ms':1})
    executor.submit(a);scheduler.start();scheduler.process.wait(timeout=30)
    assert executor.observe(a)['execution']=='EXPIRED'
    assert not (executor.directory(a.attempt_id)/'child.json').exists()
    assert not (executor.directory(a.attempt_id)/'candidate.json').exists()
    # Forced requeue may incur a new scheduler allocation, but cannot rerun mathematics.
    accepted=(executor.directory(a.attempt_id)/'accepted').stat().st_mtime_ns
    scheduler.start(restarts='1');scheduler.process.wait(timeout=30)
    assert scheduler.process.returncode!=0
    assert (executor.directory(a.attempt_id)/'accepted').stat().st_mtime_ns==accepted


def test_explicit_fields_and_site_allowlist_reject_hostile_values():
    for value in ['cpu\n#SBATCH --wrap=evil','cpu; touch /tmp/pwned','../other','-oProxyCommand=evil']:
        with pytest.raises(ValueError): SlurmConfig(partition=value,account='research')
    for raw in [b'123 RUNNING human formatted\n',b'1_2|name|0|RUNNING|date|comment\n',b'1|name|0|RUNNING|date|comment|extra\n']:
        with pytest.raises(RemoteUnavailable): parse_rows(raw)
    with patch.dict(os.environ,{'SBATCH_WRAP':'evil','SSH_AUTH_SOCK':'secret','SLURM_CONF':'untrusted'}):
        assert not {'SBATCH_WRAP','SSH_AUTH_SOCK','SLURM_CONF'} & set(scheduler_environment())


def test_accounted_requeue_and_conflicting_exit_are_not_success(slurm):
    executor,scheduler,a=slurm;executor.submit(a)
    scheduler.restarts='1'
    with pytest.raises(RemoteUnavailable,match='REQUEUE'): executor.observe(a)
    with pytest.raises(RemoteUnavailable,match='REQUEUE'): executor.cancel(a)
    assert scheduler.cancelled==0
    scheduler.restarts='0';scheduler.row[3]='COMPLETED';scheduler.exit_code='1:0'
    assert executor.observe(a)['execution']=='FAILED'


def test_duplicate_accounting_incarnations_keep_ownership_unknown(slurm):
    executor,scheduler,a=slurm;executor.submit(a)
    rows=executor.rows(a)
    duplicate=dict(rows[1][0],submitted=(datetime.fromisoformat(scheduler.row[4])-timedelta(seconds=1)).isoformat())
    with pytest.raises(RemoteUnavailable,match='DUPLICATE_OR_REQUEUED'):
        executor.owned(a,rows[0],rows[1]+[duplicate])
