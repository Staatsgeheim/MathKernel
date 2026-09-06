# =============================================================================
# MathKernel - YOLO-mode settings mutation
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import os

import pytest

from mathkernel import MathKernel
from mathkernel.settings import (Settings, apply_setting_updates, parse_setting_value,
                                 setting_schema, yolo_mode)


@pytest.fixture(autouse=True)
def _isolate_mathkernel_env():
    """YOLO writes process env; restore MATHKERNEL_* after each test."""
    snapshot = {key: value for key, value in os.environ.items()
                if key.startswith("MATHKERNEL_")}
    yield
    for key in list(os.environ):
        if key.startswith("MATHKERNEL_"):
            del os.environ[key]
    os.environ.update(snapshot)


def test_yolo_mode_defaults_false(monkeypatch):
    monkeypatch.delenv("MATHKERNEL_YOLO_MODE", raising=False)
    assert yolo_mode() is False
    kernel = MathKernel()
    result = kernel.yolo_settings({"max_ode_steps": 50})
    assert not result.ok
    assert "MATHKERNEL_YOLO_MODE=true" in result.errors[0]


def test_schema_covers_live_and_process_env_keys():
    schema = setting_schema()
    assert schema["MATHKERNEL_MAX_ODE_STEPS"]["type"] == "int"
    assert schema["MATHKERNEL_TOLERANCE"]["type"] == "float"
    assert schema["MATHKERNEL_ENABLE_EXECUTION"]["type"] == "bool"
    assert schema["MATHKERNEL_LEAN_BINARY"]["type"] == "str"
    assert schema["MATHKERNEL_STORE_PATH"]["type"] == "str"
    assert schema["MATHKERNEL_YOLO_MODE"]["scope"] == "process_env"
    assert "MATHKERNEL_CODEGEN_LANGUAGES" not in schema


@pytest.mark.parametrize("key,value,expected", [
    ("MATHKERNEL_MAX_ODE_STEPS", "200", 200),
    ("max_ode_steps", 200.0, 200),
    ("MATHKERNEL_TOLERANCE", "1e-8", 1e-8),
    ("enable_execution", "true", True),
    ("MATHKERNEL_ENABLE_PARALLEL", "off", False),
    ("lean_binary", "lean-custom", "lean-custom"),
    ("MATHKERNEL_STORE_PATH", "", None),
    ("MATHKERNEL_STORE_PATH", None, None),
])
def test_parse_setting_value_types(key, value, expected):
    env, field, typed = parse_setting_value(key, value)
    assert env.startswith("MATHKERNEL_")
    assert typed == expected


def test_parse_rejects_wrong_types():
    with pytest.raises(ValueError, match="integer"):
        parse_setting_value("max_ode_steps", True)
    with pytest.raises(ValueError, match="integer"):
        parse_setting_value("max_ode_steps", 1.5)
    with pytest.raises(ValueError, match="boolean"):
        parse_setting_value("enable_execution", "maybe")
    with pytest.raises(ValueError, match="unknown"):
        parse_setting_value("MATHKERNEL_NOT_A_SETTING", 1)


def test_yolo_apply_updates_live_settings_and_engines(monkeypatch):
    monkeypatch.setenv("MATHKERNEL_YOLO_MODE", "true")
    kernel = MathKernel(Settings(max_ode_steps=10, solver_timeout_seconds=5.0,
                                 z3_timeout_ms=100, lean_timeout_seconds=9.0))
    listed = kernel.yolo_settings()
    assert listed.ok
    assert listed.data["yolo_mode"] is True
    assert "MATHKERNEL_MAX_ODE_STEPS" in listed.data["schema"]

    result = kernel.yolo_settings({
        "MATHKERNEL_MAX_ODE_STEPS": "250",
        "solver_timeout_seconds": 12,
        "MATHKERNEL_Z3_TIMEOUT_MS": 2500,
        "MATHKERNEL_LEAN_TIMEOUT_SECONDS": "11",
        "MATHKERNEL_ENABLE_EXECUTION": True,
        "MATHKERNEL_TOLERANCE": "1e-6",
    })
    assert result.ok, result.errors
    assert kernel.settings.max_ode_steps == 250
    assert kernel.settings.enable_execution is True
    assert kernel.settings.tolerance == pytest.approx(1e-6)
    assert kernel.sympy.timeout_seconds == 12.0
    assert kernel.z3.timeout_ms == 2500
    assert kernel.lean.timeout == 11.0
    assert result.data["applied"]["MATHKERNEL_MAX_ODE_STEPS"] == 250


def test_yolo_store_path_opens_and_clears(monkeypatch, tmp_path):
    monkeypatch.setenv("MATHKERNEL_YOLO_MODE", "true")
    kernel = MathKernel()
    path = str(tmp_path / "yolo.sqlite")
    opened = kernel.yolo_settings({"MATHKERNEL_STORE_PATH": path})
    assert opened.ok, opened.errors
    assert kernel._store is not None
    assert kernel.store_status().data["enabled"] is True
    cleared = kernel.yolo_settings({"store_path": None})
    assert cleared.ok, cleared.errors
    assert kernel._store is None
    assert kernel.settings.store_path is None


def test_yolo_rejects_invalid_update_without_partial_apply(monkeypatch):
    monkeypatch.setenv("MATHKERNEL_YOLO_MODE", "true")
    kernel = MathKernel(Settings(max_ode_steps=11))
    result = kernel.yolo_settings({"max_ode_steps": 12, "tolerance": "nope"})
    assert not result.ok
    assert kernel.settings.max_ode_steps == 11


def test_yolo_can_disable_itself(monkeypatch):
    monkeypatch.setenv("MATHKERNEL_YOLO_MODE", "true")
    kernel = MathKernel()
    result = kernel.yolo_settings({"MATHKERNEL_YOLO_MODE": False})
    assert result.ok, result.errors
    assert yolo_mode() is False
    blocked = kernel.yolo_settings({"max_ode_steps": 99})
    assert not blocked.ok


def test_apply_setting_updates_validates_positive_limits():
    with pytest.raises(ValueError, match="positive"):
        apply_setting_updates(Settings(), {"max_ode_steps": 0})
    with pytest.raises(ValueError, match="1000"):
        apply_setting_updates(Settings(), {"max_output_size_bytes": 10})


def test_mcp_tool_registered_and_gated_by_default(monkeypatch):
    pytest.importorskip("fastmcp")
    monkeypatch.delenv("MATHKERNEL_YOLO_MODE", raising=False)
    from mathkernel_mcp.server import mcp, kernel as server_kernel
    names = {t.name for t in mcp._tool_manager._tools.values()} \
        if hasattr(mcp, "_tool_manager") else set()
    if names:
        assert "math_yolo_settings" in names
    # The imported server kernel is independent of this process env if the
    # module was already loaded; the gate is read live from os.environ.
    blocked = server_kernel.yolo_settings({"max_ode_steps": 1})
    if not yolo_mode():
        assert not blocked.ok
