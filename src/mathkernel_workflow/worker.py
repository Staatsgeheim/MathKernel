"""One isolated kernel per run. No browser dependency and no persisted replay."""
from __future__ import annotations
import os
import threading
import time
from .contracts import canonical

def execute(connection, operations, memory_mb):
    from multiprocessing import parent_process

    def watch_parent():
        while True:
            parent = parent_process()
            if parent is not None and (not parent.is_alive()):
                os._exit(72)
            time.sleep(0.5)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        if memory_mb:
            import resource
            ceiling = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
        from mathkernel import MathKernel, Settings
        kernel = MathKernel(Settings(max_workers=1, store_path=None))
        values = {}
        for item in operations:
            connection.send_bytes(canonical({'kind': 'started', 'node': item['id']}))
            params = dict(item['parameters'])
            for port, source in item['inputs'].items():
                params[port] = values[source]
            result = getattr(kernel, item['method'])(**params)
            from mathkernel.models import MathResult
            if not isinstance(result, MathResult):
                raise TypeError('Adapter did not return a MathResult')
            record = {'kind': 'result', 'node': item['id'], 'result': result.model_dump(mode='json'), 'claim_trust': {name: bundle.conservative_trust() for name, bundle in result.claim_evidence.items()}}
            wire = canonical(record)
            if len(wire) > 1500000:
                raise ValueError('Result exceeds local control budget; no evidence was relabelled')
            connection.send_bytes(wire)
            if not result.ok:
                return
            if item['output_key']:
                value = result.data.get(item['output_key'])
                if not isinstance(value, str):
                    raise ValueError('Kernel output did not supply its contracted object reference')
                values[item['id']] = value
        connection.send_bytes(canonical({'kind': 'finished'}))
    except BaseException as exc:
        try:
            connection.send_bytes(canonical({'kind': 'error', 'message': str(exc)[:2000]}))
        except BaseException:
            pass
    finally:
        connection.close()
