# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Private persistent-process callable worker. Not an external RPC endpoint."""
from __future__ import annotations
import os
import sys
import cloudpickle


def main():
    from . import process_runner as runner
    runner._IN_WORKER = True
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", closefd=True)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.close(devnull)
    runner.write_frame(protocol, cloudpickle.dumps({"ready": True}, protocol=5))
    while True:
        try:
            request = runner.read_frame(sys.stdin.buffer)
        except EOFError:
            break
        try:
            fn, args, kwargs = cloudpickle.loads(request)
            value = fn(*args, **kwargs)
            envelope = {"ok": True, "value": value}
        except BaseException as exc:
            envelope = {"ok": False, "exception": exc, "message": str(exc)[:8192]}
        try:
            payload = cloudpickle.dumps(envelope, protocol=5)
            if len(payload) > runner.MAX_MESSAGE_BYTES:
                raise ValueError("isolated solver response exceeded output limit")
        except Exception as exc:
            payload = cloudpickle.dumps({"ok": False, "message": str(exc)[:8192]}, protocol=5)
        runner.write_frame(protocol, payload)
    protocol.close()


if __name__ == "__main__":
    main()
