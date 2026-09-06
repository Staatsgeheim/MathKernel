# =============================================================================
# MathKernel - lean bootstrap tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from pathlib import Path

import pytest

from mathkernel import MathKernel
from mathkernel.lean_bootstrap import (
    LEAN_TOOLCHAIN,
    packaged_workspace,
    resolve_lean_toolchain,
    skip_install,
)


def test_packaged_lean_workspace_is_complete():
    root = packaged_workspace()
    assert (root / "lean-toolchain").read_text(encoding="utf-8").strip() == LEAN_TOOLCHAIN
    lakefile = (root / "lakefile.toml").read_text(encoding="utf-8")
    assert 'name = "mathlib"' in lakefile
    assert "v4.33.0" in lakefile
    assert (root / "MathkernelLean.lean").is_file()


def test_skip_install_env(monkeypatch):
    monkeypatch.delenv("MATHKERNEL_SKIP_LEAN_INSTALL", raising=False)
    assert skip_install() is False
    monkeypatch.setenv("MATHKERNEL_SKIP_LEAN_INSTALL", "1")
    assert skip_install() is True


def test_resolve_without_cache_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("MATHKERNEL_LEAN_CACHE", str(tmp_path / "missing"))
    monkeypatch.delenv("MATHKERNEL_LEAN_BINARY", raising=False)
    monkeypatch.delenv("MATHKERNEL_ELAN_HOME", raising=False)
    monkeypatch.delenv("MATHKERNEL_LEAN_WORKSPACE", raising=False)
    assert resolve_lean_toolchain() is None
    assert MathKernel().lean.available is False


def test_elan_archive_name_is_platform_specific(monkeypatch):
    from mathkernel.lean_bootstrap import _elan_archive_name
    monkeypatch.setattr("mathkernel.lean_bootstrap.platform.system", lambda: "Windows")
    assert _elan_archive_name() == "elan-x86_64-pc-windows-msvc.zip"
    monkeypatch.setattr("mathkernel.lean_bootstrap.platform.system", lambda: "Linux")
    monkeypatch.setattr("mathkernel.lean_bootstrap.platform.machine", lambda: "x86_64")
    assert _elan_archive_name() == "elan-x86_64-unknown-linux-gnu.tar.gz"
    monkeypatch.setattr("mathkernel.lean_bootstrap.platform.system", lambda: "Darwin")
    monkeypatch.setattr("mathkernel.lean_bootstrap.platform.machine", lambda: "arm64")
    assert _elan_archive_name() == "elan-aarch64-apple-darwin.tar.gz"


def test_ensure_respects_skip(monkeypatch, tmp_path):
    from mathkernel.lean_bootstrap import ensure_lean_toolchain
    monkeypatch.setenv("MATHKERNEL_SKIP_LEAN_INSTALL", "1")
    monkeypatch.setenv("MATHKERNEL_LEAN_CACHE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="SKIP_LEAN_INSTALL"):
        ensure_lean_toolchain()
