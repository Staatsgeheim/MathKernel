"""Explicit local management commands; no cloud credentials or provisioning surface."""
import argparse
import sys
from pathlib import Path
from .client import ComputeClient
from .models import ComputeRequest
from .protocol import canonical, parse
from .files import bounded_read


def main():
    p = argparse.ArgumentParser(description='MathKernel compute — local protocol pilot')
    p.add_argument('--state-dir', type=Path, default=Path.home()/'.mathkernel'/'compute')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('targets')
    sub.add_parser('list')
    q = sub.add_parser('plan'); q.add_argument('request', type=Path)
    q = sub.add_parser('approve-local'); q.add_argument('plan_id')
    q = sub.add_parser('submit'); q.add_argument('plan_id'); q.add_argument('--grant', required=True); q.add_argument('--request-id', required=True)
    for command in ('status', 'reconcile', 'cancel', 'fetch', 'verify', 'result', 'candidate', 'replay'):
        q = sub.add_parser(command); q.add_argument('job_id')
    args = p.parse_args()
    try:
        with ComputeClient(state_dir=args.state_dir) as c:
            if args.command == 'plan':
                plan = c.plan(parse(ComputeRequest, bounded_read(args.request)))
                value = {'plan': plan.model_dump(mode='json'), 'plan_digest': plan.digest}
            elif args.command == 'approve-local':
                plan = c._plan(args.plan_id)
                print(canonical(plan).decode(), file=sys.stderr)
                print('Local execution only. Type the plan digest to authorize one attempt:', plan.digest, file=sys.stderr)
                if not sys.stdin.isatty() or input().strip() != plan.digest:
                    raise PermissionError('Explicit terminal confirmation required')
                value = c._authorize_local(plan.plan_id)
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
    except (ValueError, PermissionError, KeyError, RuntimeError) as exc:
        print(f'Compute command refused: {exc}', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
