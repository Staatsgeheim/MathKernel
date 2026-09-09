"""Explicit host management commands. Grants require human terminal confirmation."""
import argparse
import sys
from pathlib import Path
from .client import ComputeClient
from .models import ComputeRequest, BudgetLimit, Money
from .managed import parse_profile
from .batch import BatchRequest
from .remote import SSHProfile
from .lambda_models import LambdaProfile, WatchdogTicket
from .protocol import canonical, parse, digest
from .files import bounded_read


def main():
    p = argparse.ArgumentParser(description='MathKernel compute — local, SSH, Slurm and experimental cloud adapters')
    p.add_argument('--state-dir', type=Path, default=Path.home()/'.mathkernel'/'compute')
    p.add_argument('--target-profile', type=Path, action='append', default=[], help='Operator-installed SSH target JSON')
    p.add_argument('--managed-profile', type=Path, action='append', default=[], help='Operator-installed Modal/Runpod JSON')
    p.add_argument('--lambda-profile', type=Path, action='append', default=[], help='Operator-installed owned-VM provisioning JSON')
    p.add_argument('--budget-limit', type=Path, action='append', default=[], help='Host budget policy JSON; native USD')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('targets')
    sub.add_parser('workspace')
    sub.add_parser('schemas')
    q = sub.add_parser('inspect-replay'); q.add_argument('manifest', type=Path)
    q = sub.add_parser('probe'); q.add_argument('target')
    sub.add_parser('list')
    q = sub.add_parser('plan'); q.add_argument('request', type=Path)
    q = sub.add_parser('approve-local'); q.add_argument('plan_id')
    q = sub.add_parser('approve-remote'); q.add_argument('plan_id')
    q = sub.add_parser('approve-managed'); q.add_argument('plan_id')
    q.add_argument('--acknowledge-exposure', action='store_true'); q.add_argument('--allow-storage', action='store_true')
    q = sub.add_parser('budget'); q.add_argument('budget_id')
    q = sub.add_parser('reconcile-cost'); q.add_argument('job_id'); q.add_argument('--usd', required=True)
    q.add_argument('--source', required=True); q.add_argument('--storage-accounted', action='store_true')
    q = sub.add_parser('plan-batch'); q.add_argument('request', type=Path)
    q = sub.add_parser('approve-batch'); q.add_argument('plan_id'); q.add_argument('--allow-export', action='store_true')
    q.add_argument('--acknowledge-exposure', action='store_true'); q.add_argument('--allow-storage', action='store_true')
    q = sub.add_parser('submit-batch'); q.add_argument('plan_id'); q.add_argument('--grant', required=True); q.add_argument('--request-id', required=True)
    for command in ('batch-status', 'reconcile-batch', 'cancel-batch'):
        q = sub.add_parser(command); q.add_argument('batch_id')
    q = sub.add_parser('submit'); q.add_argument('plan_id'); q.add_argument('--grant', required=True); q.add_argument('--request-id', required=True)
    q = sub.add_parser('vm-plan'); q.add_argument('target')
    q = sub.add_parser('vm-approve'); q.add_argument('lease_id')
    for flag in ('acknowledge-exposure', 'acknowledge-broad-key', 'acknowledge-egress', 'acknowledge-output-loss'):
        q.add_argument('--' + flag, action='store_true')
    q = sub.add_parser('vm-launch'); q.add_argument('lease_id'); q.add_argument('--grant', required=True); q.add_argument('--request-id', required=True)
    for command in ('vm-status', 'vm-observe', 'vm-reconcile', 'vm-cancel', 'vm-watchdog-ticket'):
        q = sub.add_parser(command); q.add_argument('lease_id')
    q = sub.add_parser('vm-attach'); q.add_argument('lease_id'); q.add_argument('target')
    q = sub.add_parser('vm-reconcile-cost'); q.add_argument('lease_id'); q.add_argument('--usd', required=True)
    q.add_argument('--source', required=True); q.add_argument('--storage-accounted', action='store_true')
    q = sub.add_parser('watchdog-arm'); q.add_argument('ticket', type=Path)
    q.add_argument('--acknowledge-independent-host', action='store_true')
    sub.add_parser('watchdog-tick')
    for command in ('status', 'reconcile', 'cancel', 'fetch', 'verify', 'result', 'candidate', 'replay'):
        q = sub.add_parser(command); q.add_argument('job_id')
    args = p.parse_args()
    try:
        if args.command == 'schemas':
            from .schemas import schema_catalog
            print(canonical(schema_catalog()).decode())
            return
        if args.command == 'inspect-replay':
            from .replay import inspect_replay
            print(canonical(inspect_replay(bounded_read(args.manifest))).decode())
            return
        with ComputeClient(state_dir=args.state_dir, reconcile_on_open=False,
                remote_targets=tuple(parse(SSHProfile, bounded_read(p, 65536), 65536) for p in args.target_profile),
                managed_targets=tuple(parse_profile(bounded_read(p, 65536)) for p in args.managed_profile),
                lambda_profiles=tuple(parse(LambdaProfile, bounded_read(p, 65536), 65536) for p in args.lambda_profile),
                budget_limits=tuple(parse(BudgetLimit, bounded_read(p, 65536), 65536) for p in args.budget_limit)) as c:
            if args.command.startswith('vm-') or args.command.startswith('watchdog-'):
                value = vm_command(c, args)
            elif args.command == 'plan':
                plan = c.plan(parse(ComputeRequest, bounded_read(args.request)))
                value = {'plan': plan.model_dump(mode='json'), 'plan_digest': plan.digest}
            elif args.command == 'approve-local':
                plan = c._plan(args.plan_id)
                print(canonical(plan).decode(), file=sys.stderr)
                print('Local execution only. Type the plan digest to authorize one attempt:', plan.digest, file=sys.stderr)
                if not sys.stdin.isatty() or input().strip() != plan.digest:
                    raise PermissionError('Explicit terminal confirmation required')
                value = c._authorize_local(plan.plan_id)
            elif args.command == 'approve-remote':
                plan = c._plan(args.plan_id)
                if plan.spec.remote is None:
                    raise ValueError('A remote plan is required')
                print(canonical(plan).decode(), file=sys.stderr)
                print('This exports the exact bundle and permits one job on the pinned existing target. '
                      'Retention and allocation costs are described above. Type the plan digest:', plan.digest, file=sys.stderr)
                if not sys.stdin.isatty() or input().strip() != plan.digest:
                    raise PermissionError('Explicit terminal confirmation required')
                value = c._authorize_remote(plan.plan_id, plan_digest=plan.digest,
                    bundle_digest=plan.spec.bundle_digest, target_profile_digest=plan.spec.remote.target_profile_digest)
            elif args.command == 'workspace':
                value = {'workspace_id': c.workspace_id}
            elif args.command == 'approve-managed':
                plan = c._plan(args.plan_id)
                if plan.spec.managed is None:
                    raise ValueError('A managed plan is required')
                confirm(plan, plan.digest, 'Authorize one paid invocation, export, persistent storage and the described residual exposure.')
                value = c._authorize_managed(plan.plan_id, plan_digest=plan.digest,
                    bundle_digest=plan.spec.bundle_digest, target_profile_digest=plan.spec.managed.target_profile_digest,
                    managed_exposure_acknowledged=args.acknowledge_exposure, persistent_storage_allowed=args.allow_storage)
            elif args.command == 'budget':
                value = c.budget_status(args.budget_id)
            elif args.command == 'reconcile-cost':
                proposed = {'job_id': args.job_id, 'amount': Money(currency='USD', amount=args.usd).model_dump(mode='json'),
                            'source': args.source, 'retained_storage_accounted': args.storage_accounted}
                confirm(proposed, digest(proposed), 'Record your reviewed billing and storage accounting; this is not provider-verified billing.')
                value = c._reconcile_cost(args.job_id, Money(currency='USD', amount=args.usd), args.source,
                                          retained_storage_accounted=args.storage_accounted)
            elif args.command == 'plan-batch':
                plan = c.plan_batch(parse(BatchRequest, bounded_read(args.request)))
                value = {'plan': plan.model_dump(mode='json'), 'plan_digest': plan.digest}
            elif args.command == 'approve-batch':
                plan = c._batch_plan(args.plan_id)
                confirm(plan, plan.digest, 'Authorize the full immutable shard set and its aggregate reservations.')
                value = c._authorize_batch(plan.batch_plan_id, plan_digest=plan.digest, export_allowed=args.allow_export,
                    managed_exposure_acknowledged=args.acknowledge_exposure, persistent_storage_allowed=args.allow_storage)
            elif args.command == 'submit-batch':
                value = c.submit_batch(plan_id=args.plan_id, authorization_ref=args.grant, client_request_id=args.request_id)
            elif args.command in ('batch-status', 'reconcile-batch', 'cancel-batch'):
                value = getattr(c, args.command.replace('-', '_'))(args.batch_id)
            elif args.command == 'probe':
                value = c.target_probe(args.target)
            elif args.command == 'submit':
                value = c.submit(plan_id=args.plan_id, authorization_ref=args.grant, client_request_id=args.request_id)
            elif args.command == 'result':
                status = c.status(args.job_id)
                value = c.accepted_result(args.job_id) if status.accepted_result_ref else status
            elif args.command == 'replay':
                sys.stdout.buffer.write(c.export_replay(args.job_id) + b'\n')
                return
            elif args.command in ('targets', 'list'):
                value = getattr(c, args.command)()
                value = [v.model_dump(mode='json') for v in value]
            else:
                value = getattr(c, args.command)(args.job_id)
            print(canonical(value).decode())
    except (ValueError, PermissionError, KeyError, RuntimeError, OSError) as exc:
        print(f'Compute command refused: {exc}', file=sys.stderr)
        raise SystemExit(2)


def confirm(value, expected, message):
    print(canonical(value).decode(), file=sys.stderr)
    print(message + ' Type the digest: ' + expected, file=sys.stderr)
    if not sys.stdin.isatty() or input().strip() != expected:
        raise PermissionError('Explicit terminal confirmation required')


def vm_command(client, args):
    vm = client.vm
    command = args.command
    if command == 'vm-plan':
        plan = vm.plan(args.target)
        return {'plan': plan.model_dump(mode='json'), 'plan_digest': plan.digest}
    if command == 'vm-approve':
        plan = vm._plan(args.lease_id)
        confirm(plan, plan.digest, 'Authorize one paid VM, its provider termination, broad controller key privileges, '
            'network egress and possible output loss at lease expiry. No hard spending or disconnected cleanup guarantee.')
        return vm._authorize(args.lease_id, plan_digest=plan.digest,
            managed_exposure_acknowledged=args.acknowledge_exposure,
            broad_key_privileges_acknowledged=args.acknowledge_broad_key,
            network_egress_acknowledged=args.acknowledge_egress,
            possible_output_loss_at_deadline_acknowledged=args.acknowledge_output_loss)
    if command == 'vm-launch':
        return vm.launch(args.lease_id, authorization_ref=args.grant, client_request_id=args.request_id)
    if command == 'vm-cancel':
        return vm.reconcile(args.lease_id, cancel=True)
    if command == 'vm-attach':
        return vm.attach(args.lease_id, args.target)
    if command == 'vm-reconcile-cost':
        proposed = {'lease_id': args.lease_id, 'usd': args.usd, 'source': args.source, 'storage_accounted': args.storage_accounted}
        confirm(proposed, digest(proposed), 'Record reviewed VM billing and any retained storage; this is a user attestation.')
        return vm._reconcile_cost(args.lease_id, Money(currency='USD', amount=args.usd), args.source,
                                  retained_storage_accounted=args.storage_accounted)
    if command.startswith('vm-'):
        return getattr(vm, command[3:].replace('-', '_'))(args.lease_id)
    from .watchdog import LambdaWatchdog
    watchdog = LambdaWatchdog(client.journal, vm.profiles)
    if command == 'watchdog-arm':
        ticket = parse(WatchdogTicket, bounded_read(args.ticket))
        confirm(ticket, ticket.digest, 'Authorize exact owned-VM termination at this fixed deadline on an independently installed host. '
                'The broad API key stays on this control host. Unretrieved output can be lost at expiry.')
        return watchdog._arm(ticket, ticket_digest=ticket.digest,
                             independent_host_acknowledged=args.acknowledge_independent_host)
    return watchdog.tick()


if __name__ == '__main__':
    main()
