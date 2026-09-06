# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Private fresh-interpreter entry point for hard-killable native solvers."""
from __future__ import annotations
import importlib
import os
import pickle
import sys


def main():
    # Preserve one private protocol descriptor, then silence Python and native
    # solver stdout so logging cannot corrupt the framed pickle response.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", closefd=True)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.close(devnull)
    try:
        request = pickle.loads(sys.stdin.buffer.read())
        module_name = request["module"]
        function_name = request["function"]
        if not module_name.startswith("mathkernel.") or not function_name.startswith("_isolated_"):
            raise ValueError("worker target is not allowlisted")
        function = getattr(importlib.import_module(module_name), function_name)
        value = function(*request["args"], **request["kwargs"])
        envelope = {"ok": True, "value": value}
    except BaseException as exc:
        envelope = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)[:8192]}
    output = pickle.dumps(envelope, protocol=5)
    limit = int(request.get("max_output_bytes", 64*1024*1024)) if "request" in locals() else 64*1024*1024
    if len(output) > limit:
        output = pickle.dumps({"ok": False, "error_type": "IsolatedSolverError",
                               "message": "isolated solver response exceeded output limit"}, protocol=5)
    protocol.write(output)
    protocol.flush()
    protocol.close()


if __name__ == "__main__":
    main()
