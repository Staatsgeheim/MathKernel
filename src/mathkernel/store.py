# =============================================================================
# MathKernel - Optional SQLite-backed persistence for expressions, contexts, plans,
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Optional SQLite-backed persistence for expressions, contexts, plans,
derivation steps, and certificates, plus deterministic replay of the
derivation DAG.

Opt-in via MATHKERNEL_STORE_PATH. Writes are append-only and idempotent
(INSERT OR REPLACE keyed by id), so a store can be shared across sessions.
Replay reconstructs the provenance chain of a derivation step in topological
order and validates DAG integrity (no cycles, no missing parents).
"""

from __future__ import annotations

import json
import hashlib
import hmac
import sqlite3
import threading

_SCHEMA = """
CREATE TABLE IF NOT EXISTS expressions (
    expr_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS contexts (
    context_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS derivations (
    step_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS certificates (
    cert_id TEXT PRIMARY KEY, engine TEXT, payload TEXT NOT NULL,
    integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS math_objects (
    object_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS result_resources (
    resource_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY, payload TEXT NOT NULL, integrity TEXT NOT NULL);
"""

_TABLE_KEYS = {
    "expressions": "expr_id", "contexts": "context_id", "plans": "plan_id",
    "derivations": "step_id", "certificates": "cert_id",
    "math_objects": "object_id", "executions": "execution_id",
    "result_resources": "resource_id",
}


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KernelStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
            # Upgrade stores created before payload integrity was introduced.
            # Existing JSON is checksummed byte-for-byte, avoiding a lossy
            # decode/re-encode migration of mathematical provenance.
            for table, key_col in _TABLE_KEYS.items():
                columns = {
                    row[1] for row in self._db.execute(
                        f"PRAGMA table_info({table})").fetchall()
                }
                if "integrity" not in columns:
                    self._db.execute(
                        f"ALTER TABLE {table} ADD COLUMN integrity TEXT")
                    rows = self._db.execute(
                        f"SELECT {key_col}, payload FROM {table} "
                        "WHERE integrity IS NULL").fetchall()
                    self._db.executemany(
                        f"UPDATE {table} SET integrity = ? WHERE {key_col} = ?",
                        [(_digest(payload), key) for key, payload in rows],
                    )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _put(self, table: str, key_col: str, key: str, payload: dict,
             extra: dict | None = None) -> None:
        payload_text = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        integrity = _digest(payload_text)
        cols = ([key_col, "payload", "integrity"] if extra is None else
                [key_col, *extra, "payload", "integrity"])
        vals = ([key, payload_text, integrity] if extra is None else
                [key, *extra.values(), payload_text, integrity])
        with self._lock, self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})", vals)

    def _get(self, table: str, key_col: str, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                f"SELECT payload, integrity FROM {table} WHERE {key_col} = ?",
                (key,)).fetchone()
        if row is None:
            return None
        payload, integrity = row
        if not isinstance(integrity, str) or not hmac.compare_digest(
                integrity, _digest(payload)):
            raise ValueError(f"persisted {table} record failed integrity verification")
        return json.loads(payload)

    def put_expression(self, expr_id: str, payload: dict) -> None:
        self._put("expressions", "expr_id", expr_id, payload)

    def get_expression(self, expr_id: str) -> dict | None:
        return self._get("expressions", "expr_id", expr_id)

    def find_expression_producer(self, expr_id: str) -> dict | None:
        """Recover provenance from pre-dev29 stores without reparsing a display."""
        with self._lock:
            rows = self._db.execute("SELECT step_id FROM derivations WHERE payload LIKE ?", ("%" + expr_id + "%",)).fetchall()
        matches = [self.get_derivation(row[0]) for row in rows]
        matches = [step for step in matches if step and step.get("output_expr_id") == expr_id]
        if len(matches) > 1:
            raise ValueError("ambiguous persisted expression provenance")
        return matches[0] if matches else None

    def put_result_resource(self, resource_id: str, payload: dict) -> None:
        self._put("result_resources", "resource_id", resource_id, payload)

    def get_result_resource(self, resource_id: str) -> dict | None:
        return self._get("result_resources", "resource_id", resource_id)

    def put_context(self, context_id: str, payload: dict) -> None:
        self._put("contexts", "context_id", context_id, payload)

    def get_context(self, context_id: str) -> dict | None:
        return self._get("contexts", "context_id", context_id)

    def put_plan(self, plan_id: str, payload: dict) -> None:
        self._put("plans", "plan_id", plan_id, payload)

    def get_plan(self, plan_id: str) -> dict | None:
        return self._get("plans", "plan_id", plan_id)

    def put_derivation(self, step_id: str, payload: dict) -> None:
        self._put("derivations", "step_id", step_id, payload)

    def get_derivation(self, step_id: str) -> dict | None:
        return self._get("derivations", "step_id", step_id)

    def put_certificate(self, cert_id: str, engine: str, payload: dict) -> None:
        self._put("certificates", "cert_id", cert_id, payload, {"engine": engine})

    def get_certificate(self, cert_id: str) -> dict | None:
        return self._get("certificates", "cert_id", cert_id)

    def put_math_object(self, object_id: str, payload: dict) -> None:
        self._put("math_objects", "object_id", object_id, payload)

    def get_math_object(self, object_id: str) -> dict | None:
        return self._get("math_objects", "object_id", object_id)

    def put_execution(self, execution_id: str, payload: dict) -> None:
        self._put("executions", "execution_id", execution_id, payload)

    def get_execution(self, execution_id: str) -> dict | None:
        return self._get("executions", "execution_id", execution_id)

    def replay(self, step_id: str) -> dict:
        """Reconstruct the derivation DAG leading to `step_id` in topological
        order. Raises ValueError on missing parents or cycles — a store that
        fails validation must not be treated as a provenance record."""
        order: list[dict] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(sid: str) -> None:
            if sid in seen:
                return
            if sid in visiting:
                raise ValueError(f"derivation DAG has a cycle at {sid}")
            step = self.get_derivation(sid)
            if step is None:
                raise ValueError(f"missing derivation step {sid}")
            visiting.add(sid)
            for parent in step.get("parents", []):
                visit(parent)
            visiting.discard(sid)
            seen.add(sid)
            order.append(step)

        visit(step_id)
        return {"root": step_id, "steps": order, "depth": len(order)}
