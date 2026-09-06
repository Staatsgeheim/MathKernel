# =============================================================================
# MathKernel - codeverify
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import sympy as sp

from .settings import Settings


class _SymPyOps:
    """Concrete ops implementation backed by SymPy, used for symbolic roundtrips."""

    def add(self, a, b): return a + b
    def mul(self, a, b): return a * b
    def neg(self, a): return -a
    def inv(self, a): return sp.Pow(a, sp.Integer(-1))
    def pow_int(self, a, n): return sp.Pow(a, sp.Integer(n))
    def pow(self, a, b): return sp.Pow(a, b)
    def sqrt(self, a): return sp.sqrt(a)
    def exp(self, a): return sp.exp(a)
    def log(self, a): return sp.log(a)
    def sin(self, a): return sp.sin(a)
    def cos(self, a): return sp.cos(a)
    def tan(self, a): return sp.tan(a)
    def abs(self, a): return sp.Abs(a)
    def from_int(self, n): return sp.Integer(n)
    def equals(self, a, b): return sp.Eq(a, b)
    def not_equals(self, a, b): return sp.Ne(a, b)
    def less_than(self, a, b): return sp.Lt(a, b)
    def less_eq(self, a, b): return sp.Le(a, b)
    def greater_than(self, a, b): return sp.Gt(a, b)
    def greater_eq(self, a, b): return sp.Ge(a, b)


def _load_generated_python(source: str, function_name: str):
    """Exec server-generated Python with a minimal namespace.

    The source is produced deterministically from validated MathIR (never raw
    user text), but we still strip the typing import and exec without builtins
    to keep the attack surface minimal.
    """
    import builtins
    from typing import Protocol, TypeVar

    def _safe_import(name, *args, **kwargs):
        # typing.Protocol subclass creation lazily imports already-loaded
        # modules; block anything that is not already in sys.modules.
        if name in sys.modules:
            return builtins.__import__(name, *args, **kwargs)
        raise ImportError(f"Import of {name!r} is not allowed in generated-code verification")

    stripped = "\n".join(
        line for line in source.splitlines()
        if not line.startswith("from typing import")
    )
    namespace: dict[str, Any] = {
        "Protocol": Protocol,
        "TypeVar": TypeVar,
        "__name__": "<generated>",
        "__builtins__": {"__build_class__": builtins.__build_class__, "__import__": _safe_import},
    }
    exec(compile(stripped, "<generated>", "exec"), namespace)  # noqa: S102 - server-generated code only
    fn = namespace.get(function_name)
    if fn is None:
        raise ValueError(f"Generated module does not define {function_name}")
    return fn


def symbolic_roundtrip(record: dict) -> dict:
    """Evaluate the generated Python against SymPy and compare with the source expression."""
    artifact = record["artifact"]
    if artifact["language"] != "python":
        return {"check": "symbolic_roundtrip", "status": "unavailable",
                "detail": "Symbolic roundtrip is currently implemented for Python artifacts only."}
    source = artifact["files"][0]["content"]
    try:
        fn = _load_generated_python(source, artifact["function_name"])
        args = [sp.Symbol(p) for p in record["params"]]
        produced = fn(*args, _SymPyOps())
    except Exception as exc:
        return {"check": "symbolic_roundtrip", "status": "failed", "detail": f"Generated code raised: {exc}"}

    expected = record["expected"]
    try:
        if record["target"] == "constraint":
            same = sp.simplify(produced) == sp.simplify(expected[0])
            return {"check": "symbolic_roundtrip", "status": "passed" if same else "failed",
                    "detail": f"produced={produced} expected={expected[0]}"}
        produced_list = list(produced) if isinstance(produced, (tuple, list)) else [produced]
        if len(produced_list) != len(expected):
            return {"check": "symbolic_roundtrip", "status": "failed",
                    "detail": f"Arity mismatch: produced {len(produced_list)} values, expected {len(expected)}."}
        residuals = [sp.simplify(p - e) for p, e in zip(produced_list, expected)]
        bad = [str(r) for r in residuals if r != 0]
        if bad:
            return {"check": "symbolic_roundtrip", "status": "failed",
                    "detail": f"Non-zero residuals after simplification: {bad}"}
        return {"check": "symbolic_roundtrip", "status": "passed",
                "detail": "Generated function matches the symbolic expression exactly."}
    except Exception as exc:
        return {"check": "symbolic_roundtrip", "status": "unknown", "detail": str(exc)}


def typecheck_artifact(artifact: dict, timeout: float = 60.0) -> dict:
    language = artifact["language"]
    files = artifact["files"]
    if language == "python":
        for f in files:
            try:
                compile(f["content"], f["path"], "exec")
            except SyntaxError as exc:
                return {"check": "typecheck", "status": "failed", "detail": f"{f['path']}: {exc}"}
        return {"check": "typecheck", "status": "passed", "detail": "Python sources compile."}
    if language == "typescript":
        tsc = shutil.which("tsc")
        if not tsc:
            return {"check": "typecheck", "status": "unavailable", "detail": "tsc is not installed."}
        with tempfile.TemporaryDirectory(prefix="mathkernel-tsc-") as td:
            for f in files:
                (Path(td) / f["path"]).write_text(f["content"], encoding="utf-8")
            try:
                p = subprocess.run([tsc, "--noEmit", "--strict",
                                    *[f["path"] for f in files]],
                                   cwd=td, capture_output=True, text=True, timeout=timeout, shell=False)
            except subprocess.TimeoutExpired:
                return {"check": "typecheck", "status": "unknown", "detail": "tsc timed out"}
        if p.returncode == 0:
            return {"check": "typecheck", "status": "passed", "detail": "tsc --strict accepted the artifact."}
        return {"check": "typecheck", "status": "failed", "detail": (p.stdout + p.stderr)[-4000:]}
    if language == "rust":
        rustc = shutil.which("rustc")
        if not rustc:
            return {"check": "typecheck", "status": "unavailable", "detail": "rustc is not installed."}
        with tempfile.TemporaryDirectory(prefix="mathkernel-rustc-") as td:
            for f in files:
                (Path(td) / f["path"]).write_text(f["content"], encoding="utf-8")
            try:
                p = subprocess.run([rustc, "--edition", "2021", "--crate-type", "lib",
                                    files[0]["path"]],
                                   cwd=td, capture_output=True, text=True, timeout=timeout, shell=False)
            except subprocess.TimeoutExpired:
                return {"check": "typecheck", "status": "unknown", "detail": "rustc timed out"}
        if p.returncode == 0:
            return {"check": "typecheck", "status": "passed", "detail": "rustc accepted the artifact."}
        return {"check": "typecheck", "status": "failed", "detail": (p.stderr or p.stdout)[-4000:]}
    return {"check": "typecheck", "status": "unavailable", "detail": f"Unknown language: {language}"}


_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import json
    import math
    import sys


    class FloatOps:
        def add(self, a, b): return a + b
        def mul(self, a, b): return a * b
        def neg(self, a): return -a
        def inv(self, a): return 1 / a
        def pow_int(self, a, n): return a ** n
        def pow(self, a, b): return a ** b
        def sqrt(self, a): return math.sqrt(a)
        def exp(self, a): return math.exp(a)
        def log(self, a): return math.log(a)
        def sin(self, a): return math.sin(a)
        def cos(self, a): return math.cos(a)
        def tan(self, a): return math.tan(a)
        def abs(self, a): return abs(a)
        def from_int(self, n): return n
        def equals(self, a, b): return a == b
        def not_equals(self, a, b): return a != b
        def less_than(self, a, b): return a < b
        def less_eq(self, a, b): return a <= b
        def greater_than(self, a, b): return a > b
        def greater_eq(self, a, b): return a >= b


    def norm(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return str(x)


    spec = importlib.util.spec_from_file_location("artifact", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inputs = json.loads(sys.argv[2])
    fn = getattr(mod, sys.argv[3])
    params = json.loads(sys.argv[4])
    result = fn(*[inputs[p] for p in params], FloatOps())
    print(json.dumps({"result": norm(result)}))
    """
).strip()


def execute_artifact(record: dict, inputs: dict[str, Any], settings: Settings) -> dict:
    """Run a generated Python artifact in an isolated subprocess.

    Disabled unless MATHKERNEL_ENABLE_EXECUTION=1. The subprocess runs with
    -I (isolated mode), a hard timeout, and only the artifact temp directory.
    """
    if not settings.enable_execution:
        raise PermissionError(
            "Code execution is disabled. Set MATHKERNEL_ENABLE_EXECUTION=1 to opt in."
        )
    artifact = record["artifact"]
    if artifact["language"] != "python":
        raise ValueError("Sandboxed execution currently supports Python artifacts only.")
    params = record["params"]
    unknown = [k for k in inputs if k not in params]
    if unknown:
        raise ValueError(f"Unknown input(s): {unknown}; expected subset of {params}")
    missing = [p for p in params if p not in inputs]
    if missing:
        raise ValueError(f"Missing input(s): {missing}")
    clean: dict[str, float | int] = {}
    for k, v in inputs.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"Input {k} must be a number, got {type(v).__name__}")
        clean[k] = v
    with tempfile.TemporaryDirectory(prefix="mathkernel-exec-") as td:
        artifact_path = Path(td) / artifact["files"][0]["path"]
        artifact_path.write_text(artifact["files"][0]["content"], encoding="utf-8")
        runner_path = Path(td) / "_runner.py"
        runner_path.write_text(_RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner_path), str(artifact_path),
                 json.dumps(clean), artifact["function_name"], json.dumps(params)],
                capture_output=True, text=True, timeout=settings.execution_timeout_seconds,
                cwd=td, shell=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timed_out": True, "result": None, "stdout": "", "stderr": "",
                    "duration_ms": settings.execution_timeout_seconds * 1000}
    if proc.returncode != 0:
        return {"ok": False, "timed_out": False, "result": None,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "duration_ms": None}
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {"ok": False, "timed_out": False, "result": None,
                "stdout": proc.stdout[-2000:], "stderr": f"Runner output was not JSON: {exc}",
                "duration_ms": None}
    return {"ok": True, "timed_out": False, "result": payload["result"],
            "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "duration_ms": None}
