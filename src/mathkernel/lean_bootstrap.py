# =============================================================================
# MathKernel - lean bootstrap
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Explicit, transactional setup of the optional Lean 4 + Mathlib toolchain.

Discovery and proof execution never install anything. Only the setup command
(or an explicit ``ensure_lean_toolchain`` call) may download dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

LEAN_TOOLCHAIN = "leanprover/lean4:v4.33.0"
MATHLIB_REV = "v4.33.0"
_WORKSPACE_FILES = ("lean-toolchain", "lakefile.toml", "MathkernelLean.lean")
MIN_FREE_BYTES = 8 * 1024**3
MAX_DOWNLOAD_BYTES = 64 * 1024**2
MAX_EXTRACT_BYTES = 256 * 1024**2
SETUP_TIMEOUT = 3600
_HEALTH_CACHE: dict[tuple, float] = {}
_ELAN_RELEASE = "https://github.com/leanprover/elan/releases/latest/download"


@dataclass(frozen=True)
class LeanToolchain:
    lean: Path
    lake: Path
    workspace: Path
    elan_home: Path
    toolchain: str = LEAN_TOOLCHAIN

    @property
    def bin_dir(self) -> Path:
        return self.lean.parent

    @property
    def ready(self) -> bool:
        return _healthy(self)


def skip_install() -> bool:
    return os.environ.get("MATHKERNEL_SKIP_LEAN_INSTALL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def cache_root() -> Path:
    override = os.environ.get("MATHKERNEL_LEAN_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "mathkernel"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "mathkernel"
    return Path.home() / ".cache" / "mathkernel"


def packaged_workspace() -> Path:
    return Path(resources.files("mathkernel") / "lean_workspace")


def elan_home() -> Path:
    override = os.environ.get("MATHKERNEL_ELAN_HOME")
    if override:
        return Path(override)
    return cache_root() / "elan"


def workspace_dir() -> Path:
    override = os.environ.get("MATHKERNEL_LEAN_WORKSPACE")
    if override:
        return Path(override)
    return cache_root() / "lean-workspace"


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _bin(home: Path, name: str) -> Path:
    return home / "bin" / _exe(name)


def _env_with_elan(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ELAN_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    return env


def _managed_generation() -> Path | None:
    try:
        value = json.loads((cache_root() / "lean-active.json").read_text())
        name = value["generation"]
        if not isinstance(name, str) or not name.startswith("install-") or Path(name).name != name:
            return None
        path = cache_root() / "lean-installs" / name
        if path.is_symlink():
            return None
        return path
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _custom_paths() -> bool:
    return (bool(os.environ.get("MATHKERNEL_ELAN_HOME") or os.environ.get("MATHKERNEL_LEAN_WORKSPACE"))
            or os.environ.get("MATHKERNEL_LEAN_BINARY", "").strip() not in {"", "lean"})


def _direct_binary(home: Path, name: str) -> Path:
    # Never probe an elan proxy: even `lean --version` can trigger a download.
    directory = LEAN_TOOLCHAIN.replace("/", "--").replace(":", "---")
    return home / "toolchains" / directory / "bin" / _exe(name)


def _spec(home: Path, workspace: Path) -> LeanToolchain:
    return LeanToolchain(_direct_binary(home, "lean"),
                         _direct_binary(home, "lake"), workspace, home)


def _proof_env(spec: LeanToolchain) -> dict[str, str]:
    env = _env_with_elan(spec.elan_home)
    env["PATH"] = str(spec.bin_dir) + os.pathsep + env["PATH"]
    env["ELAN_TOOLCHAIN"] = LEAN_TOOLCHAIN
    for key in ("LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT"):
        env.pop(key, None)
    return env


def _healthy(spec: LeanToolchain, timeout: float = 75) -> bool:
    deadline = time.monotonic() + timeout
    try:
        if (spec.workspace / "lean-toolchain").read_text().strip() != LEAN_TOOLCHAIN:
            return False
        mathlib = spec.workspace / ".lake" / "packages" / "mathlib"
        files = (spec.lean, spec.lake, spec.lean.parent.parent / "lib" / "lean" / "Init.olean",
                 spec.workspace / "lean-toolchain", spec.workspace / "lakefile.toml",
                 spec.workspace / "lake-manifest.json", mathlib / ".lake" / "build" / "lib" / "lean" / "Mathlib.olean")
        signature = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files)
        # Briefly cache only successful probes. A fresh process always checks;
        # broken dynamic libraries or oleans are detected again within a minute.
        if _HEALTH_CACHE.get(signature, 0) > time.monotonic():
            return True
        version = _run_process([str(spec.lean), "--version"], capture_output=True,
                               text=True, timeout=min(15, timeout), env=_proof_env(spec))
        if version.returncode or "version 4.33.0" not in version.stdout:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        checked = _execute_script(spec, "import Mathlib\nexample (x y : ℚ) : (x+y)^2 = x^2+2*x*y+y^2 := by ring\n", min(60, remaining))
        if checked.returncode:
            return False
        _HEALTH_CACHE.clear()
        _HEALTH_CACHE[signature] = time.monotonic() + 60
        return True
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def resolve_lean_toolchain() -> LeanToolchain | None:
    """Locate a working, pinned local toolchain without downloads or repairs."""
    generation = None if _custom_paths() else _managed_generation()
    home = generation / "elan" if generation else elan_home()
    workspace = generation / "workspace" if generation else workspace_dir()
    spec = _spec(home, workspace)
    configured = os.environ.get("MATHKERNEL_LEAN_BINARY", "").strip()
    if configured and configured != "lean":
        lean = Path(configured).expanduser().resolve()
        # Explicit standalone installs must have the ordinary Lean layout.
        # Elan shims lack the adjacent lib/lean tree and are never executed.
        spec = LeanToolchain(lean, lean.parent / _exe("lake"), workspace, home)
    return spec if spec.ready else None


def _copy_workspace(dest: Path) -> None:
    src = packaged_workspace()
    dest.mkdir(parents=True, exist_ok=True)
    for name in _WORKSPACE_FILES:
        target = dest / name
        target.write_bytes((src / name).read_bytes())


def _download(url: str, dest: Path, timeout: float = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if timeout <= 0:
        raise TimeoutError("Download time budget is exhausted")
    deadline = time.monotonic() + timeout
    count = 0
    with urllib.request.urlopen(url, timeout=min(30, timeout)) as response, dest.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > MAX_DOWNLOAD_BYTES or time.monotonic() > deadline:
                raise RuntimeError("elan download exceeded its size or time limit")
            handle.write(block)


def _extract_archive(archive: Path, dest: Path) -> None:
    """Extract only bounded ordinary files/directories; reject links and traversal."""
    total = 0
    def target(name: str, size: int) -> Path:
        nonlocal total
        total += size
        path = dest / name
        if (total > MAX_EXTRACT_BYTES or "\\" in name or
                Path(name).is_absolute() or ".." in Path(name).parts or
                not path.resolve().is_relative_to(dest.resolve())):
            raise RuntimeError("Unsafe or oversized elan archive")
        return path
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for item in zf.infolist():
                path = target(item.filename, item.file_size)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise RuntimeError("Unsupported elan archive member")
                if item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(item) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            for item in tf:
                path = target(item.name, item.size)
                if item.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(item) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise RuntimeError("Unsupported elan archive member")


@contextmanager
def _setup_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    # OS-owned locks are released on process exit, including crashes. Never
    # infer ownership from a stale PID or unlink a live lock's inode.
    with (root / "lean-setup.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another mathkernel-lean-setup process holds the installation lock") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _elan_archive_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return "elan-x86_64-pc-windows-msvc.zip"
    if system == "Darwin":
        arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
        return f"elan-{arch}-apple-darwin.tar.gz"
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    return f"elan-{arch}-unknown-linux-gnu.tar.gz"


def _run_process(args, *, timeout: float, check: bool = False,
                 capture_output: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Bound the whole subprocess group, including installer/lake children."""
    if timeout <= 0:
        raise TimeoutError("Subprocess time budget is exhausted")
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.name != "nt":
        kwargs["start_new_session"] = True
    with subprocess.Popen(args, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            if os.name == "nt":
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                   capture_output=True, timeout=10, check=False)
                finally:
                    process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            raise
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result


def _install_elan(home: Path, timeout: float = SETUP_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    home.mkdir(parents=True, exist_ok=True)
    env = _env_with_elan(home)
    archive_name = _elan_archive_name()
    with tempfile.TemporaryDirectory(prefix="download-", dir=home.parent) as td:
        tmp = Path(td)
        archive = tmp / archive_name
        _download(f"{_ELAN_RELEASE}/{archive_name}", archive, timeout=min(120, timeout))
        extracted = tmp / "extracted"
        extracted.mkdir()
        _extract_archive(archive, extracted)
        installer = next(extracted.rglob("elan-init*"), None)
        if installer is None:
            installer = next(extracted.rglob("elan*"), None)
        if installer is None:
            raise RuntimeError(f"elan archive {archive_name} did not contain an installer")
        if os.name != "nt":
            installer.chmod(installer.stat().st_mode | stat.S_IEXEC)
        cmd = [str(installer), "-y", "--no-modify-path",
               "--default-toolchain", LEAN_TOOLCHAIN]
        _run_process(cmd, check=True, env=env, timeout=deadline - time.monotonic())


def _run_lake(workspace: Path, home: Path, args: list[str], timeout: float = 3600) -> None:
    spec = _spec(home, workspace)
    lake = spec.lake
    if not lake.is_file():
        raise FileNotFoundError("Pinned lake is not available after elan install")
    _run_process(
        [str(lake), *args],
        cwd=str(workspace),
        check=True,
        env=_proof_env(spec),
        timeout=timeout,
    )


def ensure_lean_toolchain(*, force: bool = False, timeout: float = SETUP_TIMEOUT,
                          min_free_bytes: int = MIN_FREE_BYTES,
                          build_from_source: bool = False) -> LeanToolchain:
    """Explicit setup only. Activate a staged generation after a real proof check.

    A failed repair leaves the old active generation intact. An interrupted
    generation is never discovered as installed and is pruned on the next setup.
    Custom installations are read-only here; repair those with their own manager.
    """
    if not math.isfinite(timeout) or timeout <= 0 or min_free_bytes < 0:
        raise ValueError("timeout must be positive and min_free_bytes nonnegative")
    root = cache_root().expanduser().resolve()
    with _setup_lock(root):
        existing = resolve_lean_toolchain()
        if existing is not None and not force:
            return existing
        if skip_install() and not force:
            raise RuntimeError("MATHKERNEL_SKIP_LEAN_INSTALL is set; explicit --force overrides it")
        if _custom_paths():
            raise RuntimeError("Custom Lean paths are not overwritten. Repair them separately or unset the path overrides to use managed setup.")
        if shutil.which("git") is None:
            raise RuntimeError("git is required to fetch Mathlib via lake")
        installs = root / "lean-installs"
        installs.mkdir(exist_ok=True)
        active = _managed_generation()
        for abandoned in installs.glob("install-*"):
            if abandoned != active and not abandoned.is_symlink() and (abandoned / ".incomplete").is_file():
                shutil.rmtree(abandoned)
        if shutil.disk_usage(root).free < min_free_bytes:
            raise RuntimeError(f"Lean setup needs at least {min_free_bytes / 1024**3:g} GiB free before downloading; choose a larger MATHKERNEL_LEAN_CACHE")
        generation = Path(tempfile.mkdtemp(prefix="install-", dir=installs))
        (generation / ".incomplete").touch()
        home, workspace = generation / "elan", generation / "workspace"
        deadline = time.monotonic() + timeout
        def remaining():
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("Lean setup exceeded its time budget")
            return seconds
        pointer = root / "lean-active.json.tmp"
        activated = False
        try:
            _install_elan(home, timeout=remaining())
            _copy_workspace(workspace)
            _run_lake(workspace, home, ["update"], timeout=remaining())
            try:
                _run_lake(workspace, home, ["exe", "cache", "get"], timeout=remaining())
            except subprocess.CalledProcessError:
                if not build_from_source:
                    raise RuntimeError("Mathlib cache fetch failed; source builds require explicit --build-from-source") from None
            _run_lake(workspace, home, ["build"], timeout=remaining())
            spec = _spec(home, workspace)
            _HEALTH_CACHE.clear()
            if not _healthy(spec, timeout=remaining()):
                raise RuntimeError("Lean/Mathlib health check failed (version, runtime libraries, or proof replay)")
            remaining()
            pointer.write_text(json.dumps({"generation": generation.name, "toolchain": LEAN_TOOLCHAIN, "mathlib": MATHLIB_REV}))
            with pointer.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(pointer, root / "lean-active.json")
            activated = True
            (generation / ".incomplete").unlink(missing_ok=True)
            return spec
        finally:
            pointer.unlink(missing_ok=True)
            if not activated:
                shutil.rmtree(generation, ignore_errors=True)


def run_lean_script(script: str, timeout: float) -> subprocess.CompletedProcess:
    spec = resolve_lean_toolchain()
    if spec is None:
        raise FileNotFoundError("Lean/Mathlib is unavailable; run mathkernel-lean-setup explicitly to install or repair it")
    return _execute_script(spec, script, timeout)


def _execute_script(spec: LeanToolchain, script: str, timeout: float) -> subprocess.CompletedProcess:
    proof_dir = spec.workspace / ".mathkernel-proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(suffix=".lean", prefix="proof-", dir=proof_dir)
    path = Path(raw)
    try:
        os.close(fd)
        path.write_text(script, encoding="utf-8")
        return _run_process(
            [str(spec.lake), "env", str(spec.lean), str(path)],
            cwd=str(spec.workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=_proof_env(spec),
        )
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly install the optional pinned Lean/Mathlib toolchain (several GiB).")
    parser.add_argument("--check", action="store_true", help="Check local health only; never download or repair")
    parser.add_argument("--force", "--repair", action="store_true", help="Stage a fresh managed installation, preserving the active one until verified")
    parser.add_argument("--timeout", type=float, default=SETUP_TIMEOUT, help="Setup time budget in seconds")
    parser.add_argument("--min-free-gib", type=float, default=MIN_FREE_BYTES / 1024**3, help="Minimum free cache space before download (not a disk quota)")
    parser.add_argument("--build-from-source", action="store_true", help="Allow a source build if the Mathlib binary cache cannot be fetched")
    args = parser.parse_args()
    try:
        spec = resolve_lean_toolchain() if args.check else ensure_lean_toolchain(
            force=args.force, timeout=args.timeout, min_free_bytes=int(args.min_free_gib * 1024**3),
            build_from_source=args.build_from_source)
        if spec is None:
            raise RuntimeError("Lean/Mathlib is missing or unhealthy; run mathkernel-lean-setup to install or --repair to replace it")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"mathkernel-lean-setup: {exc}") from exc
    print(f"Lean toolchain ready: {spec.lean}")
    print(f"Lake workspace: {spec.workspace}")
    print(f"Pinned: {spec.toolchain} + mathlib {MATHLIB_REV}")


if __name__ == "__main__":
    main()
