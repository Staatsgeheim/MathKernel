"""R3 contract and real OpenSSH-client/loopback SSH transport tests.

The optional Paramiko dependency is test-only. The server dispatches the actual
installed gateway in a subprocess; cryptographic host authentication is not mocked.
This is not qualification of an external OpenSSH daemon or a distributed host.
"""
import io
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import pytest
from mathkernel_compute import ComputeClient, ComputeRequest, CuboidParameters, ResourceRequirements
from mathkernel_compute.executor import LocalExecutor, worker_environment
from mathkernel_compute.gateway import Gateway
from mathkernel_compute.models import AttemptRecord, ExecutionSpec, InputBundle, RemoteBinding
from mathkernel_compute.protocol import canonical, digest, decode, write_frame
from mathkernel_compute.registry import runtime_profile
from mathkernel_compute.remote import SSHProfile, SSHExecutor, WorkerConfig, GatewayRequest, AttemptRef, RemoteUnavailable, run_bounded


def remote_attempt(workspace='workspace_test', *, adapter='ssh', expires=None):
    bundle = InputBundle(request=ComputeRequest(operation='cuboid_sweep', parameters=CuboidParameters(bound='20'), target='test-host'), classification='explicit_export')
    runtime = runtime_profile()
    from mathkernel_compute.remote import SlurmConfig
    spec = ExecutionSpec(bundle=bundle, bundle_digest=digest(bundle), target='test-host', runtime=runtime,
        policy_digest='a'*64, remote=RemoteBinding(target_profile_digest='b'*64, verifier_runtime=runtime, adapter=adapter, destination='localhost:22', ssh_account='compute',
            allocation=SlurmConfig(partition='cpu',account='research',qos='short').allocation() if adapter=='slurm' else None))
    return AttemptRecord(attempt_id='attempt_'+'a'*32, workspace_id=workspace, spec=spec,
        bundle_digest=digest(bundle), execution_digest=digest(spec), start_deadline_ms=expires or time.time_ns()//1_000_000+60000)


@pytest.fixture
def gateway(tmp_path):
    root = tmp_path/'spool'; root.mkdir(mode=0o700)
    return Gateway(WorkerConfig(target_id='test-host', spool_root=str(root), allowed_workspaces=('workspace_test',)))


def test_gateway_duplicate_conflict_scope_and_symlink(gateway, tmp_path):
    attempt = remote_attempt()
    with patch.object(LocalExecutor, 'submit') as submit:
        command = GatewayRequest(command='submit', attempt=attempt)
        gateway.dispatch(command); gateway.dispatch(command)
        assert submit.call_count == 1
        with pytest.raises(PermissionError):
            gateway.dispatch(GatewayRequest(command='submit', attempt=attempt.model_copy(update={'start_deadline_ms': attempt.start_deadline_ms+1})))
        with pytest.raises(PermissionError):
            gateway.dispatch(GatewayRequest(command='submit', attempt=remote_attempt('workspace_intruder')))
    other = tmp_path/'unrelated'; other.mkdir()
    work = gateway.root/'workspace_test'
    linked = work/('attempt_'+'b'*32); linked.symlink_to(other)
    with pytest.raises(ValueError):
        gateway.directory(AttemptRef.from_attempt(attempt).model_copy(update={'attempt_id': linked.name}))
    assert list(other.iterdir()) == []


def test_gateway_expired_and_local_bundles_rejected_before_staging(gateway):
    with pytest.raises(PermissionError):
        gateway.dispatch(GatewayRequest(command='submit', attempt=remote_attempt(expires=1)))
    attempt=remote_attempt()
    local_bundle=attempt.spec.bundle.model_copy(update={'classification':'local_only'})
    with pytest.raises(PermissionError):
        gateway.dispatch(GatewayRequest(command='submit', attempt=attempt.model_copy(update={'spec':attempt.spec.model_copy(update={'bundle':local_bundle})})))
    assert not (gateway.root/'workspace_test').exists()


def test_gateway_probe_has_no_submission_effect(gateway):
    with patch.object(LocalExecutor, 'submit', side_effect=AssertionError('must not execute')):
        hello=gateway.dispatch(GatewayRequest(command='hello'))
    assert hello['runtime'] == runtime_profile().model_dump(mode='json')
    assert list(gateway.root.iterdir()) == []


def test_bounded_process_output_and_timeout():
    with pytest.raises(RemoteUnavailable, match='OUTPUT_LIMIT'):
        run_bounded([sys.executable,'-c',"import sys;sys.stdout.write('x'*1000000)"],limit=4096)
    with pytest.raises(RemoteUnavailable, match='TIMEOUT'):
        run_bounded([sys.executable,'-c','import time;time.sleep(10)'],timeout=.1)


def test_native_canonical_records_remain_backward_compatible():
    from mathkernel_compute.models import ComputePlan
    from mathkernel_compute.protocol import parse
    request=ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'))
    bundle=InputBundle(request=request)
    spec=ExecutionSpec(bundle=bundle,bundle_digest=digest(bundle),runtime=runtime_profile(),policy_digest='a'*64)
    assert 'remote' not in decode(canonical(spec))
    a=AttemptRecord(attempt_id='attempt_old',workspace_id='old',execution_digest=digest(spec),bundle_digest=digest(bundle),spec=spec)
    assert 'start_deadline_ms' not in decode(canonical(a))
    assert digest(parse(AttemptRecord,canonical(a))) == digest(a)


class LoopbackSSH:
    def __init__(self, root, config):
        paramiko=pytest.importorskip('paramiko')
        self.p=paramiko
        self.root=root
        self.key=paramiko.RSAKey.generate(2048)
        self.client_key=paramiko.RSAKey.generate(2048)
        self.identity=root/'identity'; self.client_key.write_private_key_file(str(self.identity)); self.identity.chmod(0o600)
        self.config_path=root/'worker.json'; self.config_path.write_bytes(canonical(config))
        self.wrapper=root/'gateway'
        self.wrapper.write_text('#!/bin/sh\nexec '+sys.executable+' -m mathkernel_compute.gateway "$@"\n'); self.wrapper.chmod(0o700)
        self.expected=f'{self.wrapper} {self.config_path}'.encode()
        self.socket=socket.socket(); self.socket.bind(('127.0.0.1',0)); self.socket.listen(); self.socket.settimeout(.1)
        self.port=self.socket.getsockname()[1]
        self.known=root/'known_hosts'; self.known.write_text(f'[127.0.0.1]:{self.port} {self.key.get_name()} {self.key.get_base64()}\n')
        self.commands=[]; self.stop=threading.Event(); self.transports=[]; self.threads=[]
        self.thread=threading.Thread(target=self.accept,daemon=True); self.thread.start()
        self.profile=SSHProfile(target_id=config.target_id,adapter=config.adapter,host='127.0.0.1',port=self.port,account='compute',
            known_hosts=str(self.known),identity_file=str(self.identity),worker_executable=str(self.wrapper),worker_config=str(self.config_path),
            spool_root=config.spool_root,runtime=runtime_profile(),worker_config_digest=digest(config),rpc_timeout_s=10)

    def accept(self):
        while not self.stop.is_set():
            try: connection,_=self.socket.accept()
            except socket.timeout: continue
            except OSError: break
            thread=threading.Thread(target=self.handle,args=(connection,),daemon=True); self.threads.append(thread); thread.start()

    def handle(self, connection):
        p=self.p; owner=self; event=threading.Event()
        class Server(p.ServerInterface):
            def check_auth_publickey(self,username,key):
                return p.AUTH_SUCCESSFUL if username=='compute' and key==owner.client_key else p.AUTH_FAILED
            def get_allowed_auths(self,username): return 'publickey'
            def check_channel_request(self,kind,chanid): return p.OPEN_SUCCEEDED if kind=='session' else p.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED
            def check_channel_exec_request(self,channel,command):
                if command != owner.expected: return False
                owner.commands.append(command); event.set(); return True
        transport=p.Transport(connection); self.transports.append(transport)
        try:
            transport.add_server_key(self.key); transport.start_server(server=Server())
            channel=transport.accept(10)
            if channel is None or not event.wait(10): return
            data=bytearray()
            while chunk:=channel.recv(4096):
                data.extend(chunk)
                if len(data)>1_048_580: return
            # Fixed test dispatch; use the same subprocess and framing as the installed entrypoint.
            process=subprocess.run([sys.executable,'-m','mathkernel_compute.gateway',str(self.config_path)],input=bytes(data),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=worker_environment(),timeout=30)
            channel.sendall(process.stdout); channel.send_exit_status(process.returncode); channel.close()
            transport.join(5)  # Let the OpenSSH client consume channel close before closing TCP.
        except (EOFError,OSError,p.SSHException): pass
        finally: transport.close()

    def close(self):
        self.stop.set(); self.socket.close(); self.thread.join(2)
        for transport in self.transports: transport.close()
        for thread in self.threads: thread.join(2)


@pytest.fixture
def ssh_host(tmp_path):
    if shutil.which('ssh') is None: pytest.skip('OpenSSH client unavailable')
    state=tmp_path/'controller'
    with ComputeClient(state_dir=state) as client: workspace=client.workspace_id
    spool=tmp_path/'spool'; spool.mkdir(mode=0o700)
    config=WorkerConfig(target_id='test-host',spool_root=str(spool),allowed_workspaces=(workspace,))
    server=LoopbackSSH(tmp_path,config)
    try: yield server,state
    finally: server.close()


def approve(client,plan):
    return client._authorize_remote(plan.plan_id,plan_digest=plan.digest,bundle_digest=plan.spec.bundle_digest,
                                    target_profile_digest=plan.spec.remote.target_profile_digest)


def test_real_ssh_exact_export_execution_reconnect_and_verification(ssh_host):
    server,state=ssh_host
    with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
        p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'),target='test-host'))
        assert server.commands == []  # Planning does not connect or stage inputs.
        with pytest.raises(PermissionError): c._authorize_local(p.plan_id)
        with pytest.raises(PermissionError):
            c._authorize_remote(p.plan_id,plan_digest=p.digest,bundle_digest='f'*64,target_profile_digest=p.spec.remote.target_profile_digest)
        g=approve(c,p)
        j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='ssh-real')
        assert c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='ssh-real') == j
    # All SSH connections have ended and the original controller has closed.
    with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
        result=c.wait(j.job_id,timeout_s=30)
        assert result.execution=='RECEIVED',result
        assert result.resources=='RELEASE_CONFIRMED'
        assert result.accepted_result_ref is None
        assert c.verify(j.job_id).verification=='PASSED'
        assert c.accepted_result(j.job_id).trust.value=='exact'
        assert result.cost=='EXISTING_HOST_COST_UNMEASURED'
        a=c._attempt(c._job(j.job_id))
        before=(Path(server.profile.spool_root)/c.workspace_id/a.attempt_id/'owner.json').read_bytes()
        SSHExecutor(server.profile).submit(a)  # Gateway deduplicates independently of the controller.
        assert (Path(server.profile.spool_root)/c.workspace_id/a.attempt_id/'owner.json').read_bytes()==before


def test_real_ssh_missing_and_changed_host_keys_fail_closed(ssh_host):
    server,_=ssh_host
    server.known.write_text('')
    with pytest.raises(RemoteUnavailable): SSHExecutor(server.profile).probe()
    other=server.p.RSAKey.generate(2048)
    server.known.write_text(f'[127.0.0.1]:{server.port} {other.get_name()} {other.get_base64()}\n')
    with pytest.raises(RemoteUnavailable): SSHExecutor(server.profile).probe()
    assert server.commands==[]


def test_real_ssh_interrupted_upload_and_injection_are_inert(ssh_host):
    server,_=ssh_host
    raw=run_bounded(SSHExecutor(server.profile).argv(),b'\x00\x00\x10\x00{"command":',timeout=10)
    assert b'REFUSED' in raw
    assert list(Path(server.profile.spool_root).iterdir())==[]
    for field,value in [('worker_executable','/bin/true;touch/tmp/pwned'),('worker_config','/tmp/../etc/passwd'),('host','-oProxyCommand=evil')]:
        data=server.profile.model_dump();data[field]=value
        with pytest.raises(ValueError): SSHProfile.model_validate(data)
    with pytest.raises(ValueError): CuboidParameters(bound='20; touch /tmp/pwned')
    argv=SSHExecutor(server.profile).argv()
    assert argv[-1].encode()==server.expected
    assert argv[1:3]==['-F','none']


def test_real_ssh_cancel_deadline_and_unrelated_process_survival(ssh_host):
    server,state=ssh_host
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
            for key,resources in [('cancel',ResourceRequirements()),('deadline',ResourceRequirements(execution_timeout_ms=100))]:
                p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='2000'),target='test-host',resources=resources))
                g=approve(c,p);j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id=key)
                if key=='cancel': c.cancel(j.job_id)
                status=c.wait(j.job_id,timeout_s=30)
                assert status.execution==('CANCELLED' if key=='cancel' else 'TIMED_OUT'),status
                assert status.resources=='RELEASE_CONFIRMED'
                assert unrelated.poll() is None
    finally: unrelated.kill();unrelated.wait()


def test_missing_target_on_restart_keeps_facts_and_cleanup_uncertainty(ssh_host):
    server,state=ssh_host
    with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
        p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'),target='test-host'))
        g=approve(c,p)
        with patch.object(SSHExecutor,'submit',side_effect=RemoteUnavailable('lost')):
            j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='missing')
    with ComputeClient(state_dir=state) as c:
        receipt=c.status(j.job_id)
        assert receipt.execution=='SUBMISSION_UNKNOWN'
        assert receipt.resources=='CLEANUP_UNKNOWN'
        assert receipt.transport=='UNAVAILABLE'
        assert len(c.journal.all('attempts'))==1


def test_raw_remote_candidate_is_retained_before_mathematical_decoding(gateway):
    import base64
    a=remote_attempt()
    with patch.object(LocalExecutor,'submit'):
        gateway.dispatch(GatewayRequest(command='submit',attempt=a))
    raw=b'{ "__proto__": {}, "trust":"formal", "duplicate":1,"duplicate":2 }'
    directory=gateway.directory(AttemptRef.from_attempt(a))
    (directory/'candidate.json').write_bytes(raw)
    result=gateway.dispatch(GatewayRequest(command='fetch',ref=AttemptRef.from_attempt(a)))
    assert base64.b64decode(result['candidate_base64'])==raw


def test_gateway_cannot_cancel_a_reused_pid(gateway):
    a=remote_attempt()
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    try:
        with patch.object(LocalExecutor,'submit'):
            gateway.dispatch(GatewayRequest(command='submit',attempt=a))
        directory=gateway.directory(AttemptRef.from_attempt(a))
        (directory/'owner.json').write_bytes(canonical({'attempt_id':a.attempt_id,'execution_digest':a.execution_digest,
                                                        'pid':unrelated.pid,'starttime':'impossible-old-incarnation'}))
        assert gateway.dispatch(GatewayRequest(command='observe',ref=AttemptRef.from_attempt(a)))['execution']=='LOST'
        gateway.dispatch(GatewayRequest(command='cancel',ref=AttemptRef.from_attempt(a)))
        assert unrelated.poll() is None
        assert (directory/'cancel').exists()
    finally: unrelated.kill();unrelated.wait()


def test_cli_discovery_can_open_without_recovery_effects(tmp_path):
    with patch.object(ComputeClient,'reconcile',side_effect=AssertionError('read-only open must not dispatch')):
        with ComputeClient(state_dir=tmp_path,reconcile_on_open=False) as c:
            c.targets()
            c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20')))


def test_real_ssh_controller_process_exit_preserves_remote_attempt(ssh_host):
    server,state=ssh_host
    profile_file=state.parent/'profile.json';profile_file.write_bytes(canonical(server.profile))
    code='''import os,sys
from mathkernel_compute import ComputeClient,ComputeRequest,CuboidParameters
from mathkernel_compute.remote import SSHProfile
from mathkernel_compute.protocol import parse
from pathlib import Path
c=ComputeClient(state_dir=sys.argv[1],remote_targets=(parse(SSHProfile,Path(sys.argv[2]).read_bytes()),))
p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'),target='test-host'))
g=c._authorize_remote(p.plan_id,plan_digest=p.digest,bundle_digest=p.spec.bundle_digest,target_profile_digest=p.spec.remote.target_profile_digest)
j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='controller-exit')
print(j.job_id,flush=True)
os._exit(0)
'''
    job_id=subprocess.check_output([sys.executable,'-c',code,str(state),str(profile_file)],env=worker_environment(),text=True,timeout=30).strip()
    with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
        assert c.wait(job_id,timeout_s=30).execution=='RECEIVED'
        assert c.verify(job_id).verification=='PASSED'
        assert len(c.journal.all('attempts'))==1


def test_remote_plan_displays_destination_allocation_and_reservation(tmp_path):
    from mathkernel_compute.remote import SlurmConfig
    from mathkernel_compute.models import SlurmAllocation
    known=tmp_path/'known';known.write_text('test-only profile fixture')
    allocation=SlurmConfig(partition='cpu',account='research').allocation()
    with pytest.raises(ValueError): SlurmAllocation.model_validate(dict(allocation.model_dump(),cpus=True))
    profile=SSHProfile(target_id='institution',adapter='slurm',host='login.example.org',account='researcher',
        known_hosts=str(known),identity_file=str(tmp_path/'identity'),worker_executable='/opt/mk/bin/gateway',
        worker_config='/opt/mk/worker.json',spool_root='/spool/mk',runtime=runtime_profile(),
        worker_config_digest='a'*64,allocation=allocation)
    with ComputeClient(state_dir=tmp_path/'state',remote_targets=(profile,)) as c:
        p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'),target='institution'))
        assert p.spec.remote.destination=='login.example.org:22'
        assert p.spec.remote.ssh_account=='researcher'
        assert p.spec.remote.allocation.memory_mib==1024
        assert p.spec.remote.allocation.allocation_account=='research'
        assert p.provider_cost is None
        g=approve(c,p)
        with patch.object(c,'reconcile'):  # No connection or allocation during this journal contract test.
            j=c.submit(plan_id=p.plan_id,authorization_ref=g.grant_id,client_request_id='allocation-review')
        reservation=c.journal.get('budget_reservations',j.job_id)
        assert reservation['maximum_cpu_seconds']==180
        assert reservation['allocation_units']=='UNKNOWN'
        assert c.status(j.job_id).cost=='ALLOCATION_USAGE_UNKNOWN'


def test_remote_profile_mismatch_exports_no_input(ssh_host):
    server,_=ssh_host
    changed=server.profile.model_copy(update={'worker_config_digest':'f'*64})
    with pytest.raises(RemoteUnavailable,match='PROFILE_MISMATCH'): SSHExecutor(changed).probe()
    assert list(Path(server.profile.spool_root).iterdir())==[]


def test_remote_ack_requires_exact_attempt_and_boolean(ssh_host):
    server,state=ssh_host
    with ComputeClient(state_dir=state,remote_targets=(server.profile,)) as c:
        p=c.plan(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='20'),target='test-host',resources=ResourceRequirements(execution_timeout_ms=100)))
        assert c._reservation(p.spec,'RESERVED')['maximum_cpu_seconds']==1
        a=AttemptRecord(attempt_id='attempt_'+'a'*32,workspace_id=c.workspace_id,spec=p.spec,
            execution_digest=p.execution_digest,bundle_digest=p.spec.bundle_digest,start_deadline_ms=p.expires_ms)
        executor=SSHExecutor(server.profile)
        with patch.object(executor,'probe'),patch.object(executor,'rpc',return_value={'accepted':True,'attempt_id':'wrong'}):
            with pytest.raises(RemoteUnavailable,match='ACCEPTANCE'): executor.submit(a)
        with patch.object(executor,'rpc',return_value={'cancel_requested':1}):
            with pytest.raises(RemoteUnavailable,match='CANCELLATION'): executor.cancel(a)
