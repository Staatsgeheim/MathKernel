"""Single-call subprocess: bounded memory/time/output around provider SDK internals."""
import sys


def main():
    import resource
    # Set before SDK imports. No child command comes from the provider or the request.
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    from .managed import BridgeCall
    from .protocol import parse, read_frame, write_frame
    response = {'protocol': 'mk.managed-bridge/1'}
    call, invocation_may_exist = None, False
    try:
        call = parse(BridgeCall, read_frame(sys.stdin.buffer))
        if sys.stdin.buffer.read(1):
            raise ValueError('BRIDGE_TRAILING_INPUT')
        if call.profile.adapter == 'modal':
            from .modal_adapter import ModalAdapter
            adapter = ModalAdapter(call.profile)
        else:
            from .runpod_adapter import RunpodAdapter
            adapter = RunpodAdapter(call.profile)
        if call.action == 'probe':
            response.update(adapter.probe())
        else:
            response.update(attempt_id=call.attempt.attempt_id, execution_digest=call.attempt.execution_digest)
            if call.action == 'submit':
                adapter.prepare(call.attempt)
                invocation_may_exist = True
                response.update(adapter.invoke(call.attempt))
            else:
                response.update(getattr(adapter, call.action)(call.attempt, call.handle))
    except FileNotFoundError:
        if call and call.action == 'fetch':
            response['missing'] = True
        else:
            response['error'] = 'PROVIDER_CALL_UNAVAILABLE'
    except Exception:
        # SDK exception messages can contain signed URLs, credentials, or provider log text.
        response['error'] = 'PROVIDER_CALL_UNAVAILABLE'
    if response.get('error') and call and call.action == 'submit':
        response['not_submitted'] = not invocation_may_exist
    write_frame(sys.stdout.buffer, response, 1_499_996)


if __name__ == '__main__':
    main()
