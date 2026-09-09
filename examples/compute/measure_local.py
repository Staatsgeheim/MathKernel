"""Local CPU qualification measurement, never a provider benchmark or speedup claim.

Run outside the checkout with the installed compute extra. The explicit flag
authorizes only the two fixed local requests below, once each. No cloud is used.
"""
import argparse
import json
import platform
from pathlib import Path
import tempfile
import time
from mathkernel_compute import ComputeClient, ComputeRequest
from mathkernel_compute.protocol import parse, decode
from mathkernel_compute.registry import runtime_profile


def measure(request):
    with tempfile.TemporaryDirectory(prefix='mk-measure-') as root:
        started = time.monotonic_ns()
        with ComputeClient(state_dir=root, reconcile_on_open=False) as client:
            plan = client.plan(request)
            planned = time.monotonic_ns()
            grant = client._authorize_local(plan.plan_id)  # Explicit --authorize-local host policy below.
            submitted = time.monotonic_ns()
            job = client.submit(plan_id=plan.plan_id, authorization_ref=grant.grant_id, client_request_id='measurement')
        # A fresh client must observe the original attempt and retrieve its output.
        with ComputeClient(state_dir=root, reconcile_on_open=False) as client:
            status = client.wait(job.job_id, timeout_s=60)
            completed = time.monotonic_ns()
            if status.execution != 'RECEIVED' or status.resources != 'RELEASE_CONFIRMED':
                raise RuntimeError('Local qualification did not complete: ' + str(status))
            client.fetch(job.job_id)
            fetched = time.monotonic_ns()
            client.verify(job.job_id)
            verified = time.monotonic_ns()
            result = client.accepted_result(job.job_id)
            candidate = client.candidate(job.job_id)
            worker = candidate.worker_claims['timing_observation']
            timing = decode((Path(root)/'attempt-spool'/job.attempt_id/'timing.json').read_bytes())
            child = int(timing['child_launch_ns'])
            op_start, op_end = int(worker['operation_started_ns']), int(worker['operation_finished_ns'])
            def milliseconds(ns): return format(ns / 1000000, '.3f')
            return {'operation': request.operation, 'engine': request.parameters.engine,
                'local_restart_recovery': True, 'trust': result.trust.value,
                'unit': 'milliseconds', 'phases': {
                    'plan_and_open': milliseconds(planned-started), 'host_authorization': milliseconds(submitted-planned),
                    'submit_and_supervisor_startup': milliseconds(child-submitted),
                    'worker_startup_and_validation': milliseconds(op_start-child),
                    'registered_handler_including_lazy_imports': milliseconds(op_end-op_start),
                    'cleanup': milliseconds(int(timing['cleanup_finished_ns'])-int(timing['cleanup_started_ns'])),
                    'retrieve_from_local_spool': milliseconds(fetched-completed),
                    'local_verification_and_admission': milliseconds(verified-fetched),
                    'end_to_end': milliseconds(verified-started)},
                'provider_queue': 'not applicable: local direct dispatch', 'provisioning': 'not applicable',
                'network_transfer': 'not applicable: local files',
                'notes': 'End-to-end includes polling, serialization and reopen overhead not separately attributed. '
                         'Worker timestamps are local diagnostic observations, never mathematical evidence. '
                         'No remote, GPU, cold-image, warm-GPU or speedup measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-local', action='store_true')
    args = parser.parse_args()
    if not args.authorize_local:
        parser.error('Explicit --authorize-local is required to run the two fixed CPU requests')
    root = Path(__file__).resolve().parent
    requests = [parse(ComputeRequest, (root/name).read_bytes()) for name in ('exact-request.json','numeric-request.json')]
    if any(r.target not in {'auto','local-cpu'} for r in requests):
        raise ValueError('This measurement authorizes local CPU only')
    print(json.dumps({'schema': 'mk.local-measurement/1', 'platform': platform.platform(),
        'runtime': runtime_profile().model_dump(mode='json'), 'runs': [measure(r) for r in requests]}, indent=2))


if __name__ == '__main__':
    main()
