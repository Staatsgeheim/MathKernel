"""Immutable contracts for contract-relative semantic correspondence audits.

These models describe explicit source and formal contracts.  They do not claim
that a reviewed contract captures arbitrary natural-language intent.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import to_jsonable_python
from ..models import FormalTarget

SCHEMA_VERSION = "mathkernel.semantic-correspondence/v1"
MANIFEST_SCHEMA_VERSION = "mathkernel.semantic-correspondence-manifest/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
IDENTIFIER_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z"
)
CLAIM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
CODE_PATTERN = re.compile(r"[A-Z][A-Z0-9_]{2,95}\Z")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a model or JSON value deterministically.

    Pydantic's JSON-mode conversion handles tuples, enums, and datetimes before
    the deliberately small JSON encoder is invoked.
    """
    value = to_jsonable_python(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation of a model or JSON value."""
    return sha256_hex(canonical_json_bytes(value))


def _digest(value: str) -> str:
    if not SHA256_PATTERN.fullmatch(value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("Expected a qualified identifier")
    return value


def _unique(values: tuple[Any, ...]) -> tuple[Any, ...]:
    keys = [
        canonical_json_bytes(value).decode("ascii")
        if isinstance(value, BaseModel)
        else repr(value)
        for value in values
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate entries are not allowed")
    return values


class Model(BaseModel):
    """Base for public correspondence contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=False,
        validate_default=True,
    )


class ExpressionKind(StrEnum):
    SYMBOL = "symbol"
    INTEGER = "integer"
    RATIONAL = "rational"
    BOOLEAN = "boolean"
    STRING = "string"
    APPLY = "apply"
    OPERATOR = "operator"
    QUANTIFIER = "quantifier"
    OPAQUE = "opaque"


class BinderKind(StrEnum):
    FORALL = "forall"
    EXISTS = "exists"
    IMPLICIT = "implicit"
    INSTANCE = "instance"


class MappingKind(StrEnum):
    MATERIAL = "material"
    BINDER = "binder"
    DEFINITION = "definition"


class CheckKind(StrEnum):
    BINDERS = "binders"
    ASSUMPTIONS = "assumptions"
    CONCLUSION = "conclusion"
    DOMAINS = "domains"
    REGULARITY = "regularity"
    DEPENDENCIES = "dependencies"
    MEASURE_SCOPE = "measure_scope"
    DEFINITIONS = "definitions"
    DEFINEDNESS = "definedness"


ALL_CHECKS: tuple[CheckKind, ...] = tuple(CheckKind)


class ClaimRelation(StrEnum):
    EQUIVALENT = "equivalent"
    FORMAL_STRONGER = "formal_stronger"
    FORMAL_WEAKER = "formal_weaker"
    INCOMPARABLE = "incomparable"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class TrustLevel(StrEnum):
    UNKNOWN = "unknown"
    EXACT = "exact"
    FORMAL = "formal"


TRUST_RANK = {
    TrustLevel.UNKNOWN: 0,
    TrustLevel.EXACT: 1,
    TrustLevel.FORMAL: 2,
}


class SourceLocator(Model):
    page: str | None = Field(default=None, min_length=1, max_length=64)
    section: str | None = Field(default=None, min_length=1, max_length=256)
    equation: str | None = Field(default=None, min_length=1, max_length=128)
    line_start: int | None = Field(default=None, ge=1, le=10_000_000)
    line_end: int | None = Field(default=None, ge=1, le=10_000_000)
    label: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def valid_range(self) -> SourceLocator:
        if not any(
            value is not None
            for value in (
                self.page,
                self.section,
                self.equation,
                self.line_start,
                self.label,
            )
        ):
            raise ValueError("A source locator must identify at least one location")
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end requires line_start")
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        return self


class SourceAnchor(Model):
    document_sha256: str
    document_uri: str | None = Field(default=None, min_length=1, max_length=2048)
    locator: SourceLocator
    excerpt_sha256: str
    transcription_sha256: str
    transcription: str = Field(min_length=1, max_length=262_144)

    _digests = field_validator(
        "document_sha256", "excerpt_sha256", "transcription_sha256"
    )(_digest)

    @model_validator(mode="after")
    def transcription_is_bound(self) -> SourceAnchor:
        if sha256_hex(self.transcription.encode("utf-8")) != self.transcription_sha256:
            raise ValueError("transcription_sha256 does not bind the UTF-8 transcription")
        return self


class ClaimExpression(Model):
    """A deliberately small, data-only claim expression tree."""

    kind: ExpressionKind
    value: str | int | bool | None = None
    operator: str | None = Field(default=None, min_length=1, max_length=128)
    arguments: tuple[ClaimExpression, ...] = Field(default=(), max_length=256)
    type_name: str | None = Field(default=None, min_length=1, max_length=256)
    opaque_reason: str | None = Field(default=None, min_length=1, max_length=1024)

    @model_validator(mode="after")
    def shape_matches_kind(self) -> ClaimExpression:
        leaf_kinds = {
            ExpressionKind.SYMBOL,
            ExpressionKind.INTEGER,
            ExpressionKind.RATIONAL,
            ExpressionKind.BOOLEAN,
            ExpressionKind.STRING,
        }
        if self.kind in leaf_kinds:
            if self.value is None or self.operator is not None or self.arguments:
                raise ValueError("Leaf expressions require only a value")
            if self.opaque_reason is not None:
                raise ValueError("Only opaque expressions may have opaque_reason")
        elif self.kind == ExpressionKind.OPAQUE:
            if self.opaque_reason is None:
                raise ValueError("Opaque expressions require an explicit reason")
            if self.operator is not None:
                raise ValueError("Opaque expressions cannot name a trusted operator")
        else:
            if self.operator is None:
                raise ValueError("Non-leaf expressions require an operator")
            if self.value is not None or self.opaque_reason is not None:
                raise ValueError("Structured expressions cannot carry leaf/opaque data")
        if self.kind == ExpressionKind.RATIONAL:
            if not isinstance(self.value, str) or not re.fullmatch(
                r"-?(?:0|[1-9][0-9]*)/[1-9][0-9]*", self.value
            ):
                raise ValueError("Rationals must use exact canonical numerator/denominator text")
        if self.kind == ExpressionKind.INTEGER and (
            isinstance(self.value, bool) or not isinstance(self.value, int)
        ):
            raise ValueError("Integer expressions require an integer value")
        if self.kind == ExpressionKind.BOOLEAN and not isinstance(self.value, bool):
            raise ValueError("Boolean expressions require a boolean value")
        if self.kind in {ExpressionKind.SYMBOL, ExpressionKind.STRING} and not isinstance(
            self.value, str
        ):
            raise ValueError("Symbol/string expressions require string values")
        return self

    @property
    def contains_opaque(self) -> bool:
        return self.kind == ExpressionKind.OPAQUE or any(
            argument.contains_opaque for argument in self.arguments
        )


Predicate = ClaimExpression


class Binder(Model):
    name: str
    kind: BinderKind = BinderKind.FORALL
    domain: ClaimExpression
    material: bool = True

    _name = field_validator("name")(_identifier)


class DomainConstraint(Model):
    subject: str
    domain: ClaimExpression
    predicate: ClaimExpression | None = None

    _subject = field_validator("subject")(_identifier)


class RegularityRequirement(Model):
    subject: str
    regularity: ClaimExpression
    order: ClaimExpression | None = None
    domain: ClaimExpression | None = None

    _subject = field_validator("subject")(_identifier)


class Dependency(Model):
    quantity: str
    depends_on: tuple[str, ...] = Field(min_length=1, max_length=256)

    _quantity = field_validator("quantity")(_identifier)

    @field_validator("depends_on")
    @classmethod
    def valid_dependencies(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _identifier(value)
        return _unique(values)


class MeasureScope(Model):
    kind: Literal["pointwise", "almost_everywhere", "everywhere", "on_subset"]
    variable: str
    measure: ClaimExpression | None = None
    domain: ClaimExpression | None = None

    _variable = field_validator("variable")(_identifier)

    @model_validator(mode="after")
    def scope_requirements(self) -> MeasureScope:
        if self.kind == "almost_everywhere" and self.measure is None:
            raise ValueError("Almost-everywhere scope requires an explicit measure")
        if self.kind == "on_subset" and self.domain is None:
            raise ValueError("Subset scope requires an explicit domain")
        return self


class DefinitionBinding(Model):
    symbol: str
    expression: ClaimExpression
    partial: bool = False

    _symbol = field_validator("symbol")(_identifier)


class DefinednessObligation(Model):
    obligation_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
    category: Literal[
        "nonempty",
        "nonzero",
        "invertible",
        "valid_index",
        "existence",
        "uniqueness",
        "partial_domain",
        "convergence",
        "coercion",
        "fallback",
        "choice",
    ]
    expression: ClaimExpression
    required: bool = True
    justification: str | None = Field(default=None, min_length=1, max_length=4096)


def _contract_expressions(contract: ClaimContract) -> tuple[ClaimExpression, ...]:
    values: list[ClaimExpression] = [contract.conclusion]
    values.extend(contract.assumptions)
    values.extend(binder.domain for binder in contract.binders)
    for domain in contract.domains:
        values.append(domain.domain)
        if domain.predicate is not None:
            values.append(domain.predicate)
    for regularity in contract.regularity:
        values.append(regularity.regularity)
        values.extend(
            value
            for value in (regularity.order, regularity.domain)
            if value is not None
        )
    if contract.measure_scope is not None:
        values.extend(
            value
            for value in (contract.measure_scope.measure, contract.measure_scope.domain)
            if value is not None
        )
    values.extend(binding.expression for binding in contract.definitions)
    values.extend(item.expression for item in contract.definedness_obligations)
    return tuple(values)


class ClaimContract(Model):
    claim_id: str
    binders: tuple[Binder, ...] = Field(default=(), max_length=256)
    assumptions: tuple[ClaimExpression, ...] = Field(default=(), max_length=1024)
    conclusion: ClaimExpression
    domains: tuple[DomainConstraint, ...] = Field(default=(), max_length=256)
    regularity: tuple[RegularityRequirement, ...] = Field(default=(), max_length=256)
    parameter_dependencies: tuple[Dependency, ...] = Field(default=(), max_length=256)
    measure_scope: MeasureScope | None = None
    definitions: tuple[DefinitionBinding, ...] = Field(default=(), max_length=256)
    definedness_obligations: tuple[DefinednessObligation, ...] = Field(
        default=(), max_length=256
    )

    @field_validator("claim_id")
    @classmethod
    def valid_claim_id(cls, value: str) -> str:
        if not CLAIM_ID_PATTERN.fullmatch(value):
            raise ValueError("Invalid claim_id")
        return value

    @field_validator(
        "binders",
        "assumptions",
        "domains",
        "regularity",
        "parameter_dependencies",
        "definitions",
        "definedness_obligations",
    )
    @classmethod
    def no_duplicates(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return _unique(values)

    @model_validator(mode="after")
    def unique_names(self) -> ClaimContract:
        for label, values in (
            ("binder", [item.name for item in self.binders]),
            ("definition", [item.symbol for item in self.definitions]),
            (
                "definedness obligation",
                [item.obligation_id for item in self.definedness_obligations],
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} names are not allowed")
        return self

    @property
    def contains_opaque(self) -> bool:
        return any(expression.contains_opaque for expression in _contract_expressions(self))

    @property
    def digest(self) -> str:
        return canonical_sha256(self)


class SymbolMapping(Model):
    source_symbol: str
    formal_symbol: str
    kind: MappingKind = MappingKind.MATERIAL

    _symbols = field_validator("source_symbol", "formal_symbol")(_identifier)


class Attestation(Model):
    attestation_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
    reviewer_id: str = Field(min_length=1, max_length=256)
    reviewed_at: datetime
    decision: Literal["approved", "rejected"]
    document_sha256: str
    excerpt_sha256: str
    transcription_sha256: str
    source_contract_sha256: str
    statement: Literal[
        "reviewed_source_anchor_transcription_symbols_scopes_assumptions_conclusion_and_definedness"
    ] = "reviewed_source_anchor_transcription_symbols_scopes_assumptions_conclusion_and_definedness"

    _digests = field_validator(
        "document_sha256",
        "excerpt_sha256",
        "transcription_sha256",
        "source_contract_sha256",
    )(_digest)

    @field_validator("reviewed_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value


class CorrespondenceSpec(Model):
    source: SourceAnchor
    source_contract: ClaimContract
    formal_target: FormalTarget
    symbol_mapping: tuple[SymbolMapping, ...] = Field(default=(), max_length=1024)
    expected_relation: Literal["equivalent"] = "equivalent"
    required_checks: tuple[CheckKind, ...] = ALL_CHECKS
    reviewer_attestations: tuple[Attestation, ...] = Field(default=(), max_length=32)

    @field_validator("symbol_mapping", "required_checks", "reviewer_attestations")
    @classmethod
    def unique_items(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return _unique(values)

    @model_validator(mode="after")
    def mapping_and_attestations(self) -> CorrespondenceSpec:
        sources = [item.source_symbol for item in self.symbol_mapping]
        targets = [item.formal_symbol for item in self.symbol_mapping]
        if len(sources) != len(set(sources)):
            raise ValueError("Each source symbol may be mapped at most once")
        if len(targets) != len(set(targets)):
            raise ValueError("Each formal symbol may be mapped at most once")

        edges = {
            item.source_symbol: item.formal_symbol
            for item in self.symbol_mapping
            if item.source_symbol != item.formal_symbol
        }
        for start in edges:
            seen: set[str] = set()
            node = start
            while node in edges:
                if node in seen:
                    raise ValueError("Symbol mappings must not contain cycles")
                seen.add(node)
                node = edges[node]

        anchor = self.source
        contract_digest = self.source_contract.digest
        ids = [item.attestation_id for item in self.reviewer_attestations]
        if len(ids) != len(set(ids)):
            raise ValueError("Attestation IDs must be unique")
        for item in self.reviewer_attestations:
            if (
                item.document_sha256 != anchor.document_sha256
                or item.excerpt_sha256 != anchor.excerpt_sha256
                or item.transcription_sha256 != anchor.transcription_sha256
                or item.source_contract_sha256 != contract_digest
            ):
                raise ValueError("Reviewer attestation does not bind this source contract")
        return self

    @property
    def is_reviewed(self) -> bool:
        return any(
            item.decision == "approved" for item in self.reviewer_attestations
        ) and not any(item.decision == "rejected" for item in self.reviewer_attestations)


class CorrespondenceManifest(Model):
    schema_version: Literal["mathkernel.semantic-correspondence-manifest/v1"] = (
        MANIFEST_SCHEMA_VERSION
    )
    review_state: Literal["candidate", "reviewed"] = "candidate"
    specification: CorrespondenceSpec

    @model_validator(mode="after")
    def reviewed_requires_attestation(self) -> CorrespondenceManifest:
        if self.review_state == "reviewed" and not self.specification.is_reviewed:
            raise ValueError("Reviewed manifests require a binding approval attestation")
        if self.review_state == "candidate" and self.specification.is_reviewed:
            raise ValueError("A manifest with effective approval must be marked reviewed")
        return self

    @property
    def digest(self) -> str:
        return canonical_sha256(self)


class FieldComparison(Model):
    check: CheckKind
    outcome: Literal["match", "difference", "unsupported", "not_checked"]
    relation: ClaimRelation = ClaimRelation.UNKNOWN
    source_digest: str | None = None
    formal_digest: str | None = None
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=256)
    opaque: bool = False
    message: str = Field(min_length=1, max_length=4096)

    _digests = field_validator("source_digest", "formal_digest")(
        lambda value: _digest(value) if value is not None else value
    )

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(values)

    @model_validator(mode="after")
    def outcome_relation(self) -> FieldComparison:
        if self.outcome == "match" and self.relation != ClaimRelation.EQUIVALENT:
            raise ValueError("Matching fields must have equivalent relation")
        if self.outcome in {"unsupported", "not_checked"} and self.relation != ClaimRelation.UNKNOWN:
            raise ValueError("Unsupported/unperformed fields must have unknown relation")
        if self.opaque and self.outcome not in {"unsupported", "not_checked"}:
            raise ValueError("Opaque fields cannot be reported as compared")
        return self


class CorrespondenceFinding(Model):
    code: str
    severity: Literal["info", "warning", "error"]
    relation: ClaimRelation = ClaimRelation.UNKNOWN
    check: CheckKind | None = None
    message: str = Field(min_length=1, max_length=4096)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=256)

    @field_validator("code")
    @classmethod
    def valid_code(cls, value: str) -> str:
        if not CODE_PATTERN.fullmatch(value):
            raise ValueError("Finding codes must be stable uppercase identifiers")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(values)


class EvidenceReference(Model):
    evidence_id: str = Field(pattern=r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
    kind: Literal[
        "SOURCE_BYTES_BOUND",
        "SOURCE_CONTRACT_ATTESTED",
        "FORMAL_CONTRACT_EXTRACTED",
        "FORMAL_TARGET_REPLAYED",
        "CONTRACT_FIELDS_COMPARED",
        "DEFINEDNESS_OBLIGATION_CHECKED",
        "MANIFEST_INSPECTED",
    ]
    artifact_sha256: str
    trust: TrustLevel = TrustLevel.UNKNOWN
    parent_ids: tuple[str, ...] = Field(default=(), max_length=256)
    detail: str | None = Field(default=None, min_length=1, max_length=4096)

    _digest = field_validator("artifact_sha256")(_digest)

    @field_validator("parent_ids")
    @classmethod
    def unique_parents(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _unique(values)

    @model_validator(mode="after")
    def evidence_kind_trust(self) -> EvidenceReference:
        if self.kind in {"SOURCE_CONTRACT_ATTESTED", "MANIFEST_INSPECTED"} and (
            self.trust != TrustLevel.UNKNOWN
        ):
            raise ValueError("Source review/inspection evidence must remain unknown trust")
        return self


class CorrespondenceReport(Model):
    schema_version: Literal["mathkernel.semantic-correspondence/v1"] = SCHEMA_VERSION
    mode: Literal["inspection", "comparison"]
    status: Literal["matched", "issues_found", "incomplete", "error"]
    relation: ClaimRelation
    trust: TrustLevel = TrustLevel.UNKNOWN
    contract_alignment: Literal[
        "established_relative_to_manifest", "not_established"
    ] = "not_established"
    semantic_alignment: Literal["not_established"] = "not_established"
    source_document_sha256: str
    source_excerpt_sha256: str
    source_transcription_sha256: str
    source_manifest_sha256: str
    canonical_manifest_sha256: str
    formal_source_sha256: str | None = None
    normalized_source_contract_sha256: str | None = None
    normalized_formal_contract_sha256: str | None = None
    comparison_receipt: dict[str, Any] | None = None
    backend: str | None = Field(default=None, min_length=1, max_length=128)
    formal_target: FormalTarget
    tool_hashes: dict[str, str] = Field(default_factory=dict, max_length=64)
    field_results: tuple[FieldComparison, ...] = Field(default=(), max_length=64)
    findings: tuple[CorrespondenceFinding, ...] = Field(default=(), max_length=4096)
    evidence: tuple[EvidenceReference, ...] = Field(default=(), max_length=4096)
    limitations: tuple[str, ...] = Field(min_length=1, max_length=64)

    _source_digests = field_validator(
        "source_document_sha256",
        "source_excerpt_sha256",
        "source_transcription_sha256",
        "source_manifest_sha256",
        "canonical_manifest_sha256",
        "formal_source_sha256",
        "normalized_source_contract_sha256",
        "normalized_formal_contract_sha256",
    )(lambda value: _digest(value) if value is not None else value)

    @field_validator("tool_hashes")
    @classmethod
    def tool_digests(cls, values: dict[str, str]) -> dict[str, str]:
        for name, digest in values.items():
            if not name or len(name) > 128:
                raise ValueError("Invalid tool hash name")
            _digest(digest)
        return values

    @field_validator("field_results", "findings", "evidence", "limitations")
    @classmethod
    def unique_report_items(cls, values: tuple[Any, ...]) -> tuple[Any, ...]:
        return _unique(values)

    @model_validator(mode="after")
    def conservative_report(self) -> CorrespondenceReport:
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        if len(evidence_by_id) != len(self.evidence):
            raise ValueError("Evidence IDs must be unique")
        for item in self.evidence:
            if item.evidence_id in item.parent_ids:
                raise ValueError("Evidence cannot be its own parent")
            missing = set(item.parent_ids) - evidence_by_id.keys()
            if missing:
                raise ValueError(f"Unknown evidence parents: {sorted(missing)}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(evidence_id: str) -> None:
            if evidence_id in visiting:
                raise ValueError("Evidence ancestry must be acyclic")
            if evidence_id in visited:
                return
            visiting.add(evidence_id)
            for parent in evidence_by_id[evidence_id].parent_ids:
                visit(parent)
            visiting.remove(evidence_id)
            visited.add(evidence_id)

        for evidence_id in evidence_by_id:
            visit(evidence_id)

        for item in self.evidence:
            if item.parent_ids:
                weakest = min(
                    TRUST_RANK[evidence_by_id[parent].trust] for parent in item.parent_ids
                )
                if TRUST_RANK[item.trust] > weakest:
                    raise ValueError("Evidence trust cannot exceed its weakest parent")

        referenced = {
            evidence_id
            for result in self.field_results
            for evidence_id in result.evidence_ids
        } | {
            evidence_id
            for finding in self.findings
            for evidence_id in finding.evidence_ids
        }
        if referenced - evidence_by_id.keys():
            raise ValueError("Report references unknown evidence")

        required_checks = {result.check for result in self.field_results}
        if len(required_checks) != len(self.field_results):
            raise ValueError("Each field check may appear at most once")

        if self.mode == "inspection":
            if (
                self.relation != ClaimRelation.UNKNOWN
                or self.contract_alignment != "not_established"
                or self.trust != TrustLevel.UNKNOWN
                or self.formal_source_sha256 is not None
                or self.normalized_source_contract_sha256 is not None
                or self.normalized_formal_contract_sha256 is not None
                or self.comparison_receipt is not None
                or self.backend is not None
                or self.field_results
                or self.status not in {"incomplete", "error"}
            ):
                raise ValueError("Inspection-only reports cannot claim comparison evidence")
        else:
            if (
                self.formal_source_sha256 is None
                or self.normalized_source_contract_sha256 is None
                or self.normalized_formal_contract_sha256 is None
                or self.comparison_receipt is None
                or self.backend is None
                or self.comparison_receipt.get("source_contract_sha256")
                != self.normalized_source_contract_sha256
                or self.comparison_receipt.get("formal_contract_sha256")
                != self.normalized_formal_contract_sha256
                or self.comparison_receipt.get("backend") != self.backend
                or canonical_sha256({
                    key: value for key, value in self.comparison_receipt.items()
                    if key != "receipt_sha256"
                }) != self.comparison_receipt.get("receipt_sha256")
            ):
                raise ValueError("Comparison reports require a valid bound comparison receipt")

        if self.contract_alignment == "established_relative_to_manifest":
            kinds = {item.kind for item in self.evidence}
            needed = {
                "SOURCE_BYTES_BOUND",
                "SOURCE_CONTRACT_ATTESTED",
                "FORMAL_CONTRACT_EXTRACTED",
                "CONTRACT_FIELDS_COMPARED",
                "DEFINEDNESS_OBLIGATION_CHECKED",
            }
            attested_manifest = any(
                item.kind == "SOURCE_CONTRACT_ATTESTED"
                and item.artifact_sha256 == self.canonical_manifest_sha256
                for item in self.evidence
            )
            if (
                self.mode != "comparison"
                or self.status != "matched"
                or self.relation != ClaimRelation.EQUIVALENT
                or self.trust != TrustLevel.UNKNOWN
                or self.formal_source_sha256 is None
                or self.normalized_source_contract_sha256 is None
                or self.normalized_formal_contract_sha256 is None
                or self.comparison_receipt is None
                or self.backend is None
                or not self.tool_hashes
                or set(CheckKind) != required_checks
                or any(
                    result.outcome != "match"
                    or result.relation != ClaimRelation.EQUIVALENT
                    or result.opaque
                    for result in self.field_results
                )
                or self.comparison_receipt.get("source_contract_sha256")
                != self.normalized_source_contract_sha256
                or self.comparison_receipt.get("formal_contract_sha256")
                != self.normalized_formal_contract_sha256
                or self.comparison_receipt.get("backend") != self.backend
                or canonical_sha256({
                    key: value for key, value in self.comparison_receipt.items()
                    if key != "receipt_sha256"
                }) != self.comparison_receipt.get("receipt_sha256")
                or any(finding.severity == "error" for finding in self.findings)
                or not needed.issubset(kinds)
                or not attested_manifest
            ):
                raise ValueError(
                    "Contract alignment requires complete, non-opaque, attested equivalent evidence"
                )
        elif self.status == "matched":
            raise ValueError("Matched status requires established contract alignment")

        if self.relation == ClaimRelation.UNKNOWN and self.contract_alignment != "not_established":
            raise ValueError("Unknown relation cannot establish contract alignment")
        return self


def correspondence_model_schemas() -> dict[str, dict[str, Any]]:
    """Return JSON Schemas for the public manifest and report contracts."""
    return {
        "manifest": CorrespondenceManifest.model_json_schema(),
        "report": CorrespondenceReport.model_json_schema(),
    }


__all__ = [
    "ALL_CHECKS",
    "Attestation",
    "Binder",
    "BinderKind",
    "CheckKind",
    "ClaimContract",
    "ClaimExpression",
    "ClaimRelation",
    "CorrespondenceFinding",
    "CorrespondenceManifest",
    "CorrespondenceReport",
    "CorrespondenceSpec",
    "DefinednessObligation",
    "DefinitionBinding",
    "Dependency",
    "DomainConstraint",
    "EvidenceReference",
    "ExpressionKind",
    "FieldComparison",
    "FormalTarget",
    "MANIFEST_SCHEMA_VERSION",
    "MappingKind",
    "MeasureScope",
    "Model",
    "Predicate",
    "RegularityRequirement",
    "SCHEMA_VERSION",
    "SourceAnchor",
    "SourceLocator",
    "SymbolMapping",
    "TrustLevel",
    "canonical_json_bytes",
    "canonical_sha256",
    "correspondence_model_schemas",
    "sha256_hex",
]
