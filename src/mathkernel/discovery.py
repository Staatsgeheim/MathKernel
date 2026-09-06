# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Machine-readable *declared* parameter hints, without invented obligations.

These schemas describe known primitive types. They intentionally do not pretend
that free-form legacy registry descriptions capture every mathematical invariant
or required argument. Adapters remain authoritative for domain validation.
"""
from __future__ import annotations
import re


def _hint_schema(hint: str) -> dict:
    raw = hint.strip()
    text = raw.rstrip("?").strip()
    head = text.split(" (", 1)[0]
    primitives = {"boolean": "boolean", "bool": "boolean", "integer": "integer",
                  "int": "integer", "float": "number", "number": "number",
                  "str": "string", "string": "string", "dict": "object"}
    if head in primitives:
        schema = {"type": primitives[head]}
    elif head in {"list", "array"}:
        schema = {"type": "array"}
    elif head.endswith("[]"):
        schema = {"type": "array", "items": _hint_schema(head[:-2])}
    elif head.startswith("list[") and head.endswith("]"):
        schema = {"type": "array", "items": _hint_schema(head[5:-1])}
    elif head == "MathIR":
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "object"}]}
    elif head == "exact|numeric":
        schema = {"type": "string", "enum": ["exact", "numeric"]}
    else:
        # Unknown notation is a documented unconstrained schema, not a guessed
        # type. For example "element" can mean several mathematical objects.
        schema = {"x-unresolved-type-hint": True}
    schema["description"] = raw
    return schema


def parameter_json_schema(hints: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {name: (_hint_schema(hint) if isinstance(hint, str) else {"description": str(hint)})
                       for name, hint in hints.items()},
        "additionalProperties": True,
        "x-validation-scope": "declared primitive hints; required arguments and mathematical invariants are checked by the adapter",
        "x-requiredness": "not inferred from legacy prose",
    }
