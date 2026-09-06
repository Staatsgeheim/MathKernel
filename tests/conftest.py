"""Test-only dependency injection across the hard process boundary.

A monkeypatch in the test process alone cannot constrain a fresh interpreter.
These guards execute and restore the patch inside the solver worker, retaining
meaningful negative tests for solver-free certificate replay.
"""
import importlib
import pytest


@pytest.fixture
def isolated_patch(monkeypatch):
    def install(module_name, dotted_attribute, replacement):
        from mathkernel import engineering_adapter
        original_entry = engineering_adapter._isolated_adapter_apply
        def patched_entry(*args, **kwargs):
            target = importlib.import_module(module_name)
            parts = dotted_attribute.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            attribute = parts[-1]
            old = getattr(target, attribute)
            setattr(target, attribute, replacement)
            try:
                return original_entry(*args, **kwargs)
            finally:
                setattr(target, attribute, old)
        monkeypatch.setattr(engineering_adapter, "_isolated_adapter_apply", patched_entry)
    return install
