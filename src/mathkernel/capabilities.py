# =============================================================================
# MathKernel - capabilities
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

_TRUST_CAPABILITIES = {
    "unknown", "heuristic", "empirical", "numeric",
    "numeric_high_precision", "interval_certified", "symbolic", "exact",
    "formal",
}


@dataclass(frozen=True)
class Capability:
    name: str
    description: str = ""
    domain: str = "core"
    operation: str | None = None
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    trust_levels: tuple[str, ...] = ()
    verification_methods: tuple[str, ...] = ()
    cost_dimensions: tuple[str, ...] = ()
    handler: str | None = None
    parameter_schema: dict | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.domain:
            raise ValueError("capability name and domain must be non-empty")
        if not self.trust_levels:
            object.__setattr__(
                self,
                "trust_levels",
                tuple(sorted(set(self.evidence) & _TRUST_CAPABILITIES)),
            )
        if not self.verification_methods:
            object.__setattr__(
                self,
                "verification_methods",
                tuple(sorted(set(self.evidence) - _TRUST_CAPABILITIES)),
            )
        unknown = set(self.trust_levels) - _TRUST_CAPABILITIES
        if unknown:
            raise ValueError(
                f"unknown capability trust level: {sorted(unknown)[0]}")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "operation": self.operation or self.name,
            "input_types": list(self.input_types),
            "output_types": list(self.output_types),
            "evidence": list(self.evidence),
            "engines": list(self.engines),
            "trust_levels": list(self.trust_levels),
            "verification_methods": list(self.verification_methods),
            "cost_dimensions": list(self.cost_dimensions),
            "handler": self.handler,
            "parameter_schema": self.parameter_schema or {},
        }


class CapabilityRegistry:
    """Queryable semantic registry used by planners and discovery surfaces."""

    def __init__(self, capabilities: list[Capability] | None = None):
        self._capabilities: dict[tuple, Capability] = {}
        for capability in capabilities or []:
            self.register(capability)

    @staticmethod
    def _key(capability: Capability) -> tuple:
        return (
            capability.domain,
            capability.operation or capability.name,
            capability.input_types,
            capability.output_types,
        )

    def register(self, capability: Capability) -> None:
        key = self._key(capability)
        existing = self._capabilities.get(key)
        if existing is None:
            self._capabilities[key] = capability
            return
        if (
            existing.handler is not None
            and capability.handler is not None
            and existing.handler != capability.handler
        ):
            raise ValueError(
                f"conflicting handlers for capability {capability.name}")
        self._capabilities[key] = Capability(
            name=existing.name,
            description=existing.description or capability.description,
            domain=existing.domain,
            operation=existing.operation or capability.operation,
            input_types=existing.input_types,
            output_types=existing.output_types,
            evidence=tuple(sorted(set(existing.evidence) | set(capability.evidence))),
            engines=tuple(sorted(set(existing.engines) | set(capability.engines))),
            trust_levels=tuple(sorted(
                set(existing.trust_levels) | set(capability.trust_levels))),
            verification_methods=tuple(sorted(
                set(existing.verification_methods)
                | set(capability.verification_methods))),
            cost_dimensions=tuple(sorted(
                set(existing.cost_dimensions) | set(capability.cost_dimensions))),
            handler=existing.handler or capability.handler,
            parameter_schema=(
                existing.parameter_schema or capability.parameter_schema),
        )

    def query(
        self,
        *,
        domain: str | None = None,
        object_type: str | None = None,
        input_type: str | None = None,
        output_type: str | None = None,
        operation: str | None = None,
        evidence: str | None = None,
        trust: str | None = None,
        verification_method: str | None = None,
        engine: str | None = None,
    ) -> list[Capability]:
        matches = []
        for capability in self._capabilities.values():
            if domain is not None and capability.domain != domain:
                continue
            if operation is not None and (capability.operation or capability.name) != operation:
                continue
            if object_type is not None and object_type not in (
                capability.input_types + capability.output_types
            ):
                continue
            if input_type is not None and input_type not in capability.input_types:
                continue
            if output_type is not None and output_type not in capability.output_types:
                continue
            if evidence is not None and evidence not in capability.evidence:
                continue
            if trust is not None and trust not in capability.trust_levels:
                continue
            if (
                verification_method is not None
                and verification_method not in capability.verification_methods
            ):
                continue
            if engine is not None and engine not in capability.engines:
                continue
            matches.append(capability)
        return sorted(matches, key=lambda item: (
            item.domain, item.operation or item.name, item.name
        ))

    def manifest(self, **filters: str | None) -> list[dict]:
        return [capability.as_dict() for capability in self.query(**filters)]

    def resolve(self, *, input_type: str, operation: str) -> Capability:
        """Resolve one operation for an input type without output-type matches."""
        matches = self.query(input_type=input_type, operation=operation)
        if not matches:
            raise LookupError(
                f"no capability applies {operation} to {input_type}")
        if len(matches) > 1:
            names = ", ".join(item.name for item in matches)
            raise LookupError(
                f"ambiguous capability for {operation} on {input_type}: {names}")
        return matches[0]


class Engine(Protocol):
    name: str
    capabilities: set[str]

    @property
    def available(self) -> bool: ...


class CapabilityRouter:
    """Registry + deterministic engine selection by semantic capability."""

    def __init__(self, engines: list[Engine]):
        self._engines = {engine.name: engine for engine in engines}
        self.registry = CapabilityRegistry()
        for engine in engines:
            for operation in engine.capabilities:
                self.registry.register(Capability(
                    name=operation,
                    domain="engine",
                    operation=operation,
                    engines=(engine.name,),
                    handler=f"router:{engine.name}",
                ))

    def engines(self) -> list[Engine]:
        return list(self._engines.values())

    def engine(self, name: str) -> Engine:
        return self._engines[name]

    def providers(self, capability: str, *, available_only: bool = True) -> list[Engine]:
        out = [e for e in self._engines.values() if capability in e.capabilities]
        if available_only:
            out = [e for e in out if e.available]
        return out

    def choose(self, capability: str, preferred: tuple[str, ...] = ()) -> Engine:
        providers = self.providers(capability)
        by_name = {e.name: e for e in providers}
        for name in preferred:
            if name in by_name:
                return by_name[name]
        if not providers:
            raise RuntimeError(f"No available engine provides capability: {capability}")
        return providers[0]

    def manifest(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "available": e.available,
                "capabilities": sorted(e.capabilities),
            }
            for e in self._engines.values()
        ]
