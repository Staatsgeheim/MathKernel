# =============================================================================
# MathKernel Projection - MathResult/typed-object adapter registry
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Registry mapping MathKernel result/typed-object shapes to projections.

Adapters are pure functions over JSON-like dicts: they extract plottable
arrays from an already-produced result or typed object and choose a canonical
projection kind.  They never recompute mathematics, never upgrade trust, and
must declare any sampling/reduction in ``information_loss``.

Dispatch order in :func:`adapt_result`:

1. ``__pydantic_model__`` marker (``"module:ClassName"`` or bare class name),
2. engine name from the result envelope,
3. data-shape predicates (first match wins).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ProjectionSpec:
    """Everything needed to build a canonical projection from a result."""
    kind: str
    payload: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    information_loss: list[str] = field(default_factory=list)
    information_loss_notes: list[str] = field(default_factory=list)
    coordinate_names: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    title: str | None = None


@dataclass
class AdapterContext:
    """Read-only context handed to adapters.

    ``resolve`` fetches a stored typed object's encoded value by object id;
    it is ``None`` when no kernel store is available (pure library use).
    Adapters that need a referenced object must return ``None`` when the id
    cannot be resolved, never invent the missing structure.
    """
    trust: str = "unknown"
    engine: str | None = None
    resolve: Callable[[str], dict | None] | None = None

    def resolve_value(self, object_id: Any) -> dict | None:
        if not isinstance(object_id, str) or self.resolve is None:
            return None
        return self.resolve(object_id)


AdapterFn = Callable[[dict[str, Any], AdapterContext], ProjectionSpec | None]
ShapePredicate = Callable[[dict[str, Any]], bool]

_MODEL_ADAPTERS: dict[str, AdapterFn] = {}
_ENGINE_ADAPTERS: dict[str, AdapterFn] = {}
_SHAPE_ADAPTERS: list[tuple[ShapePredicate, AdapterFn]] = []


def register_model_adapter(model: str, fn: AdapterFn) -> None:
    """Register an adapter for a typed model ("module:ClassName" or name)."""
    if not model:
        raise ValueError("model key must be non-empty")
    _MODEL_ADAPTERS[model] = fn


def register_engine_adapter(engine: str, fn: AdapterFn) -> None:
    """Register an adapter for all results produced by an engine."""
    if not engine:
        raise ValueError("engine key must be non-empty")
    _ENGINE_ADAPTERS[engine] = fn


def register_shape_adapter(predicate: ShapePredicate, fn: AdapterFn) -> None:
    """Register a fallback adapter guarded by a data-shape predicate."""
    _SHAPE_ADAPTERS.append((predicate, fn))


def encoded_fields(value: Any) -> dict[str, Any]:
    """Normalize an encoded typed model (or plain dict) to its field dict."""
    if not isinstance(value, dict):
        return {}
    fields = value.get("fields")
    return fields if isinstance(fields, dict) else value


def adapt_result(data: Any, ctx: AdapterContext) -> ProjectionSpec | None:
    """Try registered adapters against a result data payload.

    Returns ``None`` when no adapter claims the shape; callers then fall
    back to generic shape inference.  Adapter exceptions propagate as
    ``ValueError`` so callers surface typed errors instead of silent
    misprojections.
    """
    if not isinstance(data, dict):
        return None
    marker = data.get("__pydantic_model__")
    if isinstance(marker, str):
        for key in (marker, marker.rsplit(":", 1)[-1]):
            fn = _MODEL_ADAPTERS.get(key)
            if fn is not None:
                spec = fn(encoded_fields(data), ctx)
                if spec is not None:
                    return spec
    if ctx.engine:
        fn = _ENGINE_ADAPTERS.get(ctx.engine)
        if fn is not None:
            spec = fn(data, ctx)
            if spec is not None:
                return spec
    for predicate, fn in _SHAPE_ADAPTERS:
        try:
            if not predicate(data):
                continue
        except (KeyError, TypeError, IndexError):
            continue
        spec = fn(data, ctx)
        if spec is not None:
            return spec
    return None


def _register_builtin() -> None:
    from . import result_adapters_builtin as _builtin  # noqa: F401


_register_builtin()
