# =============================================================================
# MathKernel - lean bootstrap
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Install and locate the default Lean 4 + Mathlib toolchain.

Lean is not a Python wheel. The default install copies a pinned lake
workspace, installs elan into a MathKernel cache, then fetches Mathlib
oleans via ``lake exe cache get``. Set MATHKERNEL_SKIP_LEAN_INSTALL=1 to
skip network work (CI, wheel smoke).
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
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
        return (
            self.lean.is_file()
            and self.lake.is_file()
            and (self.workspace / "lean-toolchain").is_file()
            and (self.workspace / ".lake" / "packages" / "mathlib").is_dir()
        )


def skip_install() -> bool:
    return os.environ.get("MATHKERNEL_SKIP_LEAN_INSTALL", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def cache_root() -> Path:
    override = os.environ.get("MATHKERNEL_LEAN_CACHE")
    if override:
        return Path(override)
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


def _which_on_path(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def _env_with_elan(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ELAN_HOME"] = str(home)
    env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
    return env


def resolve_lean_toolchain() -> LeanToolchain | None:
    """Locate an already-installed MathKernel Lean + Mathlib workspace."""
    home = elan_home()
    configured = os.environ.get("MATHKERNEL_LEAN_BINARY", "").strip()
    lean = Path(configured) if configured and configured != "lean" else _bin(home, "lean")
    if not lean.is_file():
        found = _which_on_path("lean")
        lean = found if found else lean
    lake = _bin(home, "lake")
    if not lake.is_file():
        found_lake = _which_on_path("lake")
        lake = found_lake if found_lake else lake
    spec = LeanToolchain(lean=lean, lake=lake, workspace=workspace_dir(), elan_home=home)
    if spec.ready:
        return spec
    return None


def _copy_workspace(dest: Path) -> None:
    src = packaged_workspace()
    dest.mkdir(parents=True, exist_ok=True)
    for name in _WORKSPACE_FILES:
        target = dest / name
        target.write_bytes((src / name).read_bytes())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


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


def _install_elan(home: Path) -> None:
    if _bin(home, "lean").is_file() and _bin(home, "lake").is_file():
        return
    home.mkdir(parents=True, exist_ok=True)
    env = _env_with_elan(home)
    archive_name = _elan_archive_name()
    with tempfile.TemporaryDirectory(prefix="mathkernel-elan-") as td:
        tmp = Path(td)
        archive = tmp / archive_name
        _download(f"{_ELAN_RELEASE}/{archive_name}", archive)
        extracted = tmp / "extracted"
        extracted.mkdir()
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extracted)
        else:
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(extracted)
        installer = next(extracted.rglob("elan-init*"), None)
        if installer is None:
            installer = next(extracted.rglob("elan*"), None)
        if installer is None:
            raise RuntimeError(f"elan archive {archive_name} did not contain an installer")
        if os.name != "nt":
            installer.chmod(installer.stat().st_mode | stat.S_IEXEC)
        cmd = [str(installer), "-y", "--no-modify-path",
               "--default-toolchain", LEAN_TOOLCHAIN]
        subprocess.run(cmd, check=True, env=env)


def _run_lake(workspace: Path, home: Path, args: list[str], timeout: float = 3600) -> None:
    lake = _bin(home, "lake")
    if not lake.is_file():
        found = _which_on_path("lake")
        if found is None:
            raise FileNotFoundError("lake is not available after elan install")
        lake = found
    subprocess.run(
        [str(lake), *args],
        cwd=str(workspace),
        check=True,
        env=_env_with_elan(home),
        timeout=timeout,
    )


def ensure_lean_toolchain(*, force: bool = False) -> LeanToolchain:
    """Install Lean 4 + Mathlib unless already present or explicitly skipped."""
    existing = resolve_lean_toolchain()
    if existing is not None and not force:
        _copy_workspace(existing.workspace)
        return existing
    if skip_install() and not force:
        raise RuntimeError(
            "Lean/Mathlib is not installed and MATHKERNEL_SKIP_LEAN_INSTALL is set"
        )
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch Mathlib via lake")
    home = elan_home()
    workspace = workspace_dir()
    _install_elan(home)
    _copy_workspace(workspace)
    _run_lake(workspace, home, ["update"])
    try:
        _run_lake(workspace, home, ["exe", "cache", "get"])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"mathkernel: Mathlib cache fetch failed ({exc}); building from source",
              file=sys.stderr)
    _run_lake(workspace, home, ["build"])
    lean = _bin(home, "lean") if _bin(home, "lean").is_file() else _which_on_path("lean")
    lake = _bin(home, "lake") if _bin(home, "lake").is_file() else _which_on_path("lake")
    if lean is None or lake is None:
        raise RuntimeError("elan finished but lean/lake binaries were not found")
    spec = LeanToolchain(lean=lean, lake=lake, workspace=workspace, elan_home=home)
    if not spec.ready:
        raise RuntimeError("Lean/Mathlib install finished but the workspace is not ready")
    return spec


def run_lean_script(script: str, timeout: float) -> subprocess.CompletedProcess:
    spec = resolve_lean_toolchain()
    if spec is None:
        if skip_install():
            raise FileNotFoundError(
                "Lean/Mathlib toolchain is not installed and MATHKERNEL_SKIP_LEAN_INSTALL is set"
            )
        spec = ensure_lean_toolchain()
    proof_dir = spec.workspace / ".mathkernel-proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(suffix=".lean", prefix="proof-", dir=proof_dir)
    path = Path(raw)
    try:
        os.close(fd)
        path.write_text(script, encoding="utf-8")
        return subprocess.run(
            [str(spec.lake), "env", str(spec.lean), str(path)],
            cwd=str(spec.workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=_env_with_elan(spec.elan_home),
        )
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    force = "--force" in sys.argv[1:]
    spec = ensure_lean_toolchain(force=force)
    print(f"Lean toolchain ready: {spec.lean}")
    print(f"Lake workspace: {spec.workspace}")
    print(f"Pinned: {spec.toolchain} + mathlib {MATHLIB_REV}")


if __name__ == "__main__":
    main()
