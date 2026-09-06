# =============================================================================
# MathKernel Artifacts - typed evidence bundles and conservative reconciliation
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer


TRUST_RANK: dict[str, int] = {
    "unknown": 0,
    "heuristic": 1,
    "empirical": 2,
    "numeric": 3,
    "numeric_high_precision": 4,
    "interval_certified": 5,
    "symbolic": 6,
    "exact": 7,
    "formal": 8,
}

EvidenceTrust = Literal[
    "unknown",
    "heuristic",
    "empirical",
    "numeric",
    "numeric_high_precision",
    "interval_certified",
    "symbolic",
    "exact",
    "formal",
]


EvidenceRole = Literal[
    "required",
    "cross_check",
    "diagnostic",
]


def _json_evidence(value: Any) -> Any:
    """Serialize engine-native witnesses without executing or importing them."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            str(key): _json_evidence(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_evidence(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [
            _json_evidence(item)
            for item in sorted(value, key=str)
        ]
    return str(value)


class _EvidenceModel(BaseModel):
    @field_serializer("*", when_used="json", check_fields=False)
    def serialize_engine_values(self, value: Any) -> Any:
        return _json_evidence(value)


class _EvidenceItemModel(_EvidenceModel):
    """One auditable piece of evidence.

    ``role`` decides whether the item is needed to establish the claim.
    Required items sharing a ``support_path`` are conjunctive; distinct paths
    are alternative derivations.  Cross-checks and diagnostics are retained in
    the audit trail but cannot weaken a complete required path.
    """

    role: EvidenceRole = "required"
    support_path: str = "primary"


class ComputationEvidence(_EvidenceItemModel):
    engine: str
    method: str
    arithmetic: str
    precision: int | None = None
    rounding: str | None = None
    deterministic: bool = True
    trust: EvidenceTrust = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProofEvidence(_EvidenceItemModel):
    proposition: str
    method: str
    engine: str
    certificate: Any | None = None
    assumptions: list[str] = Field(default_factory=list)
    side_conditions: list[str] = Field(default_factory=list)
    verified: bool = False
    trust: EvidenceTrust = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CertificateEvidence(_EvidenceItemModel):
    certificate_type: str
    claim: str
    witness: Any
    verifier: str
    verified: bool
    trust: EvidenceTrust = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class NumericalEvidence(_EvidenceItemModel):
    precision: int | None = None
    absolute_error: str | float | None = None
    relative_error: str | float | None = None
    residual: Any | None = None
    interval: Any | None = None
    convergence: Any | None = None
    conditioning: Any | None = None
    trust: EvidenceTrust = "numeric"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelEvidence(_EvidenceItemModel):
    assumptions: list[str] = Field(default_factory=list)
    diagnostics: list[Any] = Field(default_factory=list)
    violations: list[Any] = Field(default_factory=list)
    goodness_of_fit: Any | None = None
    sensitivity: Any | None = None
    trust: EvidenceTrust = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmpiricalEvidence(_EvidenceItemModel):
    sample_size: int
    sampling_method: str | None = None
    uncertainty: Any | None = None
    replication: Any | None = None
    diagnostics: list[Any] = Field(default_factory=list)
    trust: EvidenceTrust = "empirical"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(_EvidenceModel):
    """Typed evidence supporting one claim or a compatible group of claims."""

    computation: list[ComputationEvidence] = Field(default_factory=list)
    proof: list[ProofEvidence] = Field(default_factory=list)
    certificate: list[CertificateEvidence] = Field(default_factory=list)
    numerical: list[NumericalEvidence] = Field(default_factory=list)
    model: list[ModelEvidence] = Field(default_factory=list)
    empirical: list[EmpiricalEvidence] = Field(default_factory=list)
    justified_trust: EvidenceTrust | None = None

    def is_empty(self) -> bool:
        return not any((
            self.computation, self.proof, self.certificate, self.numerical,
            self.model, self.empirical,
        ))

    def conservative_trust(self) -> str:
        """Return the strongest complete support path, conservatively capped.

        Required evidence in one ``support_path`` is conjunctive. Distinct
        paths are alternative derivations. Cross-check and diagnostic evidence
        remain auditable but never downgrade an already established claim.
        """
        by_path: dict[str, list[str]] = {}

        def add(item: Any, level: str) -> None:
            if item.role != "required":
                return
            by_path.setdefault(item.support_path or "primary", []).append(level)

        for group in (self.computation, self.numerical, self.model, self.empirical):
            for item in group:
                add(item, item.trust)
        for item in self.proof:
            add(item, item.trust if item.verified else "unknown")
        for item in self.certificate:
            add(item, item.trust if item.verified else "unknown")

        if not by_path:
            return "unknown"
        path_levels = [
            min(levels, key=TRUST_RANK.__getitem__)
            for levels in by_path.values() if levels
        ]
        if not path_levels:
            return "unknown"
        level = max(path_levels, key=TRUST_RANK.__getitem__)
        if self.justified_trust is not None:
            level = min((level, self.justified_trust), key=TRUST_RANK.__getitem__)
        return level


def reconcile_claim_evidence(
    claims: dict[str, EvidenceBundle],
    required_claims: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Conservatively summarize only evidence required by the requested conclusion."""
    names = list(required_claims) if required_claims is not None else list(claims)
    if not names or any(name not in claims for name in names):
        return "unknown"
    levels = [claims[name].conservative_trust() for name in names]
    if any(level not in TRUST_RANK for level in levels):
        return "unknown"
    return min(levels, key=TRUST_RANK.__getitem__)


def merge_evidence_bundles(
    bundles: list[EvidenceBundle] | tuple[EvidenceBundle, ...],
) -> EvidenceBundle:
    """Merge evidence slices without sharing mutable records.

    A bundle-level ``justified_trust`` is a *global claim ceiling* (for
    example, inherited numeric input ancestry).  Ceilings therefore compose
    conjunctively and MUST survive merges; dropping them would allow an
    alternative strong proof path to launder a weaker required input.
    """
    merged = EvidenceBundle()
    ceilings: list[str] = []
    for bundle in bundles:
        merged.computation.extend(
            item.model_copy(deep=True) for item in bundle.computation)
        merged.proof.extend(
            item.model_copy(deep=True) for item in bundle.proof)
        merged.certificate.extend(
            item.model_copy(deep=True) for item in bundle.certificate)
        merged.numerical.extend(
            item.model_copy(deep=True) for item in bundle.numerical)
        merged.model.extend(
            item.model_copy(deep=True) for item in bundle.model)
        merged.empirical.extend(
            item.model_copy(deep=True) for item in bundle.empirical)
        if bundle.justified_trust is not None:
            ceilings.append(bundle.justified_trust)
    if ceilings:
        merged.justified_trust = min(
            ceilings, key=TRUST_RANK.__getitem__)
    return merged



def tag_evidence_bundle(
    bundle: EvidenceBundle,
    *,
    role: EvidenceRole | None = None,
    support_path: str = "primary",
) -> EvidenceBundle:
    """Deep-copy a bundle while assigning a support path.

    When ``role`` is omitted, existing roles are preserved.  This is the safe
    default when moving an already-structured bundle onto another derivation
    path: optional cross-checks must not become required merely because their
    containing obligation is selected as part of a result path.
    """
    tagged = bundle.model_copy(deep=True)
    for group in (
        tagged.computation, tagged.proof, tagged.certificate,
        tagged.numerical, tagged.model, tagged.empirical,
    ):
        for item in group:
            if role is not None:
                item.role = role
            item.support_path = support_path
    return tagged

def extract_evidence(
    source: Any,
) -> tuple[EvidenceBundle, dict[str, EvidenceBundle]]:
    """Normalize evidence from result models or their serialized form."""
    if isinstance(source, dict):
        raw = source
    elif hasattr(source, "model_dump"):
        raw = source.model_dump(mode="python")
    else:
        raw = {}

    claims = {
        str(name): (
            value.model_copy(deep=True)
            if isinstance(value, EvidenceBundle)
            else EvidenceBundle.model_validate(value)
        )
        for name, value in (raw.get("claim_evidence") or {}).items()
    }
    candidate = raw.get("evidence_bundle")
    if candidate is None:
        legacy_candidate = raw.get("evidence")
        if isinstance(legacy_candidate, EvidenceBundle):
            candidate = legacy_candidate
        elif isinstance(legacy_candidate, dict) and any(
            key in legacy_candidate for key in (
                "computation", "proof", "certificate",
                "numerical", "model", "empirical",
            )
        ):
            candidate = legacy_candidate
    bundle = (
        candidate.model_copy(deep=True)
        if isinstance(candidate, EvidenceBundle)
        else EvidenceBundle.model_validate(candidate or {})
    )
    if bundle.is_empty() and claims:
        bundle = (
            claims["result"].model_copy(deep=True)
            if "result" in claims
            else merge_evidence_bundles(tuple(claims.values()))
        )
    if not claims and not bundle.is_empty():
        claims["result"] = bundle.model_copy(deep=True)
    return bundle, claims


def extract_engine_versions(source: Any) -> dict[str, str]:
    """Read normalized engine versions from a serialized result."""
    if isinstance(source, dict):
        raw = source
    elif hasattr(source, "model_dump"):
        raw = source.model_dump(mode="json")
    else:
        return {}
    versions = (
        raw.get("data", {})
        .get("provenance", {})
        .get("engine_versions", {})
    )
    return {
        str(engine): str(version)
        for engine, version in versions.items()
    } if isinstance(versions, dict) else {}
