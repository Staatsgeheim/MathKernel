"""Operator-only, fail-closed Comparator/nanoda replay in a fresh workspace.

Do NOT pre-build the submission. Comparator owns challenge-first compilation.
This follows Comparator's trusted-reference and sandbox boundary, including
blocking AF_UNIX in the service. A clean copy is not itself a security sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from .models import (AuditLimits, ComparatorReport, ComparatorRequest, PinnedBinary, TOOLCHAIN)
from .source import (audit_lean_project, bounded_bytes, checked_root, inventory,
                     safe_relative, validate_comparator_config, hash_file)


class ReplayBlocked(ValueError):
    pass


def binary_path(binary: PinnedBinary, forbidden: tuple[Path, ...]) -> Path:
    path = Path(binary.path)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ReplayBlocked(f"Pinned executable is unavailable: {path}")
    resolved = path.resolve()
    if any(c in str(resolved) for c in ("%", "$", "\n", "\r")):
        raise ReplayBlocked("Executable path contains unsupported service expansion characters")
    if any(resolved.is_relative_to(root) for root in forbidden):
        raise ReplayBlocked("Verifier/tool binaries must be outside both project trees")
    if hash_file(resolved, 512 * 1024**2)[1] != binary.sha256:
        raise ReplayBlocked(f"Binary fingerprint mismatch: {path}")
    return resolved


def run_bounded(command: list[str], *, cwd: Path, env: dict[str, str],
                timeout: float, output_limit: int) -> dict:
    """Bound wall time and combined output; terminate the complete local group.

    The caller must additionally stop a systemd unit: terminating systemd-run's
    process group alone does not guarantee termination of the transient service.
    """
    start = time.monotonic()
    if timeout <= 0 or output_limit < 1:
        raise ValueError("Positive process limits are required")
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=(os.name == "posix"))
    buffers = [bytearray(), bytearray()]
    count = 0
    lock = threading.Lock()
    overflow = threading.Event()
    def reader(stream, index):
        nonlocal count
        try:
            while data := stream.read(8192):
                with lock:
                    remaining = max(0, output_limit - count)
                    buffers[index].extend(data[:remaining])
                    count += len(data)
                    if count > output_limit: overflow.set()
        finally:
            stream.close()
    threads = [threading.Thread(target=reader, args=(proc.stdout, 0), daemon=True),
               threading.Thread(target=reader, args=(proc.stderr, 1), daemon=True)]
    for thread in threads: thread.start()
    status = "finished"
    try:
        while True:
            if overflow.is_set():
                status = "output_limit"
                break
            if time.monotonic() - start >= timeout:
                status = "timeout"
                break
            if proc.poll() is not None and not any(t.is_alive() for t in threads):
                break
            time.sleep(0.01)
    finally:
        if status != "finished" or proc.poll() is None:
            if os.name == "posix":
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            else:
                # The supported Comparator runner is Linux-only. This fallback
                # only serves direct unit tests / library subprocess callers.
                proc.kill()
        try: proc.wait(timeout=5)
        except subprocess.TimeoutExpired: proc.kill()
        for thread in threads: thread.join(timeout=1)
    if overflow.is_set(): status = "output_limit"
    return {"status": status, "returncode": proc.returncode,
            "stdout": bytes(buffers[0]).decode("utf-8", errors="replace"),
            "stderr": bytes(buffers[1]).decode("utf-8", errors="replace"),
            "elapsed_seconds": time.monotonic() - start}


def service_command(request: ComparatorRequest, tools: dict[str, Path], workspace: Path,
                    config: Path, unit: str, home: Path) -> list[str]:
    # No shell, no submission-controlled command vector, no network, no UNIX
    # sockets in the service, and limits on the whole service cgroup.
    command = [str(tools["systemd_run"]), "--user", "--pipe", "--wait", "--collect",
               "--service-type=exec", "--unit=" + unit,
               "--working-directory=" + str(workspace),
               "--property=RestrictAddressFamilies=~AF_UNIX",
               "--property=PrivateNetwork=yes", "--property=NoNewPrivileges=yes",
               "--property=KillMode=control-group", "--property=TasksMax=128",
               "--property=MemoryMax=" + str(request.memory_bytes),
               "--property=RuntimeMaxSec=" + str(request.timeout_seconds)]
    environment = {
        "HOME": str(home), "PATH": str(tools["lean"].parent) + ":/usr/bin:/bin",
        "COMPARATOR_LANDRUN": str(tools["landrun"]),
        "COMPARATOR_LEAN4EXPORT": str(tools["lean4export"]),
        "COMPARATOR_NANODA": str(tools["nanoda"]),
        "LANG": "C.UTF-8",
    }
    # env -i clears the service manager's environment too, not merely ours.
    # The verifier must run under a dedicated account without valuable secrets.
    command += ["--", str(tools["env"]), "-i"]
    command += [key + "=" + value for key, value in environment.items()]
    command += [str(tools["lake"]), "env", str(tools["comparator"]), str(config)]
    return command


def _copy_inventory(source: Path, destination: Path, files, *, only_lean=False,
                    skip_paths=frozenset()):
    for item in files:
        if item.path in skip_paths or (only_lean and (not item.path.endswith(".lean")
                or Path(item.path).name == "lakefile.lean")):
            # lakefile.lean is executable build configuration, not a solution module.
            continue
        target = destination / item.path
        if target.exists():
            raise ReplayBlocked(f"Submission would overwrite a trusted file: {item.path}")
        data = bounded_bytes(source / item.path, item.size)
        if hashlib.sha256(data).hexdigest() != item.sha256:
            raise ReplayBlocked(f"Source changed during staging: {item.path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        # Needed for trusted dependency utilities. Submission files stay data.
        if not only_lean and (source / item.path).stat().st_mode & 0o111:
            target.chmod(0o700)


def verify_with_comparator(request: ComparatorRequest, *, authorize_execution: bool = False) -> ComparatorReport:
    """Run the pinned verifier on a fresh, reference-controlled project.

    Linux + unprivileged user + working systemd user service + Landrun are
    mandatory. There is deliberately no unsafe local fallback. No automatic
    downloads, tool installation, or `lake build` precede Comparator.
    """
    # Capture caller-owned mutable mappings before making an execution decision.
    request = ComparatorRequest.model_validate(request.model_dump(mode="json"))
    common = dict(reference_tree_sha256=request.reference_tree_sha256,
                  targets=request.spec.targets, permitted_axioms=request.spec.permitted_axioms)
    if not authorize_execution:
        return ComparatorReport(status="blocked", errors=("Explicit operator execution authorization is required",), **common)
    command: list[str] = []
    hashes: dict[str, str] = {}
    digest = None
    try:
        if os.name != "posix" or not hasattr(os, "uname") or os.uname().sysname != "Linux":
            raise ReplayBlocked("Comparator replay requires Linux with its supported sandbox")
        if os.geteuid() == 0:
            raise ReplayBlocked("Comparator must not run as root")
        if not request.spec.targets:
            raise ReplayBlocked("A nonempty target list is mandatory")
        submission = checked_root(request.submission_root)
        reference = checked_root(request.reference_root)
        if submission.is_relative_to(reference) or reference.is_relative_to(submission):
            raise ReplayBlocked("Submission and trusted reference must be separate, nonnested trees")
        scan = audit_lean_project(submission, request.spec)
        digest = scan.source_sha256
        # Placeholders are not a rejection by themselves: Comparator checks
        # transitive *theorem* dependencies. Structural errors do block replay.
        fatal_codes = {"SOURCE_PIN_MISMATCH", "TOOLCHAIN_UNPINNED", "TOOLCHAIN_MISMATCH",
                       "DEPENDENCY_UNPINNED", "LOCKFILE_INVALID", "TARGET_MODULE_MISSING",
                       "SOURCE_UNPARSED", "SOURCE_CHANGED", "FINDINGS_TRUNCATED", "DEPENDENCIES_UNLOCKED"}
        fatal = [f.message for f in scan.findings if f.code in fatal_codes]
        if fatal: raise ReplayBlocked("; ".join(fatal))
        limits = AuditLimits(max_files=200000, max_file_bytes=1024**3, max_total_bytes=32 * 1024**3)
        _, reference_files, reference_digest = inventory(reference, limits, include_dependencies=True)
        if reference_digest != request.reference_tree_sha256:
            raise ReplayBlocked("Trusted reference workspace fingerprint mismatch")
        if (reference / ".lake" / "build").exists():
            raise ReplayBlocked("Trusted reference must have no top-level .lake/build; use a fresh workspace")
        toolchain = bounded_bytes(reference / "lean-toolchain", 1024).decode().strip()
        if not TOOLCHAIN.fullmatch(toolchain) or toolchain != scan.toolchain:
            raise ReplayBlocked("Reference and submission must use the same concrete Lean version")
        if not (reference / "lake-manifest.json").is_file():
            raise ReplayBlocked("Trusted reference requires a reviewed dependency lockfile")
        ref_scan = audit_lean_project(reference)
        if any(f.code in {"DEPENDENCY_UNPINNED", "LOCKFILE_INVALID"} for f in ref_scan.findings):
            raise ReplayBlocked("Trusted reference dependency lock is invalid or unpinned")
        config_rel = safe_relative(request.config)
        config = json.loads(bounded_bytes(reference / config_rel, 1024**2))
        config_errors = validate_comparator_config(config, request.spec)
        if config_errors: raise ReplayBlocked("; ".join(config_errors))
        challenge_path = config["challenge_module"].replace(".", "/") + ".lean"
        if not (reference / challenge_path).is_file():
            raise ReplayBlocked("Trusted challenge source is missing")
        solution_path = config["solution_module"].replace(".", "/") + ".lean"
        if (reference / solution_path).exists():
            raise ReplayBlocked("The trusted reference must not already contain the submission module")
        tools = {}
        for name, binary in request.tools:
            tools[name] = binary_path(binary, (submission, reference))
            hashes[name] = binary.sha256
        if tools["lean"].parent != tools["lake"].parent or not (
                tools["lean"].parent.parent / "lib" / "lean" / "Init.olean").is_file():
            raise ReplayBlocked("Use direct binaries from one real Lean installation, not elan proxies")
        # Only the trusted binary's version query is run outside the sandbox;
        # it neither imports nor executes any submission code.
        outer_env = {k: os.environ[k] for k in ("PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "LANG") if k in os.environ}
        outer_env["PATH"] = "/usr/bin:/bin"
        with tempfile.TemporaryDirectory(prefix="mathkernel-formal-") as temporary:
            temporary = Path(temporary)
            workspace, home = temporary / "project", temporary / "home"
            workspace.mkdir(); home.mkdir()
            version = run_bounded([str(tools["lean"]), "--version"], cwd=home, env=outer_env,
                                  timeout=min(15, request.timeout_seconds), output_limit=8192)
            wanted = toolchain.split(":v", 1)[1]
            if (version["status"] != "finished" or version["returncode"] != 0
                    or not re.search(r"\bversion " + re.escape(wanted) + r"(?:[,\s]|$)", version["stdout"])):
                raise ReplayBlocked("The actual Lean binary version does not match the project pin")
            _copy_inventory(reference, workspace, reference_files)
            # The reference, not the submitter, owns challenge/config/Lake files.
            trusted_files = {f.path: f for f in reference_files}
            trusted_paths = set(trusted_files)
            skip = {challenge_path}
            for f in scan.source_files:
                if (f.path.endswith(".lean") and Path(f.path).name != "lakefile.lean"
                        and f.path in trusted_paths and f.path != challenge_path):
                    other = trusted_files[f.path]
                    if f.sha256 != other.sha256:
                        raise ReplayBlocked(f"Submission attempts to change a reference dependency: {f.path}")
                    skip.add(f.path)
            _copy_inventory(submission, workspace, scan.source_files, only_lean=True, skip_paths=skip)
            # Top-level build configs from the submission were never copied.
            before = inventory(workspace)[2]
            unit = "mathkernel-audit-" + uuid.uuid4().hex
            command = service_command(request, tools, workspace, workspace / config_rel, unit, home)
            run = None
            try:
                run = run_bounded(command, cwd=home, env=outer_env,
                                  timeout=request.timeout_seconds + 15, output_limit=request.max_output_bytes)
            finally:
                # A failed/interrupted client is not evidence the service stopped.
                # Stop its entire cgroup even when the local runner raised.
                if run is None or run["status"] != "finished" or run["returncode"] != 0:
                    cleanup = run_bounded([str(tools["systemctl"]), "--user", "stop", unit],
                        cwd=home, env=outer_env, timeout=10, output_limit=8192)
                    if run is not None and cleanup["returncode"] != 0:
                        run["stderr"] += "\nUnit stop did not succeed; RuntimeMaxSec remains the hard service watchdog."
            after = inventory(workspace)[2]
            # Recheck the trusted executables before recording an acceptance.
            for name, binary in request.tools:
                binary_path(binary, (submission, reference))
            log_digest = hashlib.sha256((run["stdout"] + "\0" + run["stderr"]).encode()).hexdigest()
            fields = dict(common, source_sha256=digest, command=tuple(command), tool_hashes=hashes,
                          returncode=run["returncode"], elapsed_seconds=run["elapsed_seconds"],
                          stdout=run["stdout"], stderr=run["stderr"], log_sha256=log_digest)
            if before != after:
                return ComparatorReport(status="error", independent_kernel="required_by_pinned_comparator", errors=("Staged source/configuration changed during verification",), **fields)
            if run["status"] != "finished":
                return ComparatorReport(status=run["status"], independent_kernel="required_by_pinned_comparator", **fields)
            if run["returncode"] != 0:
                return ComparatorReport(status="not_accepted", independent_kernel="required_by_pinned_comparator", **fields)
            return ComparatorReport(status="accepted", trust="formal", independent_kernel="accepted", **fields)
    except (OSError, ValueError, UnicodeError, KeyError, TypeError) as exc:
        return ComparatorReport(status="blocked", errors=(str(exc),), source_sha256=digest,
                                command=tuple(command), tool_hashes=hashes,
                                independent_kernel="required_by_pinned_comparator" if command else "not_run", **common)
