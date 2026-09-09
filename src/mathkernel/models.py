# =============================================================================
# MathKernel - models
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from enum import Enum
from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator
from mathkernel_artifacts.evidence import (
    CertificateEvidence,
    ComputationEvidence,
    EmpiricalEvidence,
    EvidenceBundle,
    ModelEvidence,
    NumericalEvidence,
    ProofEvidence,
    TRUST_RANK,
    reconcile_claim_evidence,
)


class TrustLevel(str, Enum):
    FORMAL = "formal"
    EXACT = "exact"
    SYMBOLIC = "symbolic"
    INTERVAL_CERTIFIED = "interval_certified"
    NUMERIC_HIGH_PRECISION = "numeric_high_precision"
    NUMERIC = "numeric"
    EMPIRICAL = "empirical"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"


class ResultStatus(str, Enum):
    PROVED = "proved"
    VERIFIED_EXACT = "verified_exact"
    VERIFIED_SYMBOLIC = "verified_symbolic"
    REFUTED = "refuted"
    CERTIFIED = "certified"
    VERIFIED_NUMERIC = "verified_numeric"
    NUMERIC = "numeric"
    EMPIRICAL = "empirical"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"
    DOES_NOT_EXIST = "does_not_exist"
    UNDEFINED = "undefined"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class OperationStatus(str, Enum):
    """Shared adapter-level outcome before MathResult reconciliation."""
    AVAILABLE = "available"
    VERIFIED = "verified"
    REFUTED = "refuted"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"
    DOES_NOT_EXIST = "does_not_exist"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


class SemanticMeta(BaseModel):
    domain: str | None = None
    value_type: str | None = None
    shape: list[int | None] | None = None
    units: str | None = None
    inferred: bool = False


class IntegerNode(BaseModel):
    kind: Literal["integer"] = "integer"
    value: str
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="integer", value_type="integer"))


class RationalNode(BaseModel):
    kind: Literal["rational"] = "rational"
    numerator: str
    denominator: str
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="rational", value_type="rational"))


class RealNode(BaseModel):
    kind: Literal["real"] = "real"
    value: str
    precision: int | None = None
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="real", value_type="real"))


class AlgebraicNumberNode(BaseModel):
    kind: Literal["algebraic"] = "algebraic"
    minimal_polynomial: "Expr"
    root_index: int
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="algebraic", value_type="algebraic_number"))


class ComplexNode(BaseModel):
    kind: Literal["complex"] = "complex"
    real: "Expr"
    imag: "Expr"
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="complex", value_type="complex"))


class IntervalNode(BaseModel):
    kind: Literal["interval"] = "interval"
    lower: "Expr"
    upper: "Expr"
    lower_closed: bool = True
    upper_closed: bool = True
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(domain="real", value_type="interval"))


# Compatibility alias used by adapters/tests written against v0.3.
# New parser output uses IntegerNode / RealNode / RationalNode explicitly.
class NumberNode(BaseModel):
    kind: Literal["number"] = "number"
    value: str
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="scalar"))


class SymbolNode(BaseModel):
    kind: Literal["symbol"] = "symbol"
    name: str
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


class UnaryNode(BaseModel):
    kind: Literal["neg"] = "neg"
    arg: "Expr"
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


class NaryNode(BaseModel):
    kind: Literal["add", "mul"]
    args: list["Expr"]
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


class BinaryNode(BaseModel):
    kind: Literal["pow", "div"]
    left: "Expr"
    right: "Expr"
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


class CallNode(BaseModel):
    kind: Literal["call"] = "call"
    name: str
    args: list["Expr"]
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


class RelationNode(BaseModel):
    kind: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    left: "Expr"
    right: "Expr"
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="proposition"))


class SetNode(BaseModel):
    """A set: either a named standard set or a finite enumerated set."""
    kind: Literal["set"] = "set"
    name: Literal["naturals", "integers", "rationals", "reals", "complexes", "empty"] | None = None
    elements: list["Expr"] | None = None
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="set"))


class MembershipNode(BaseModel):
    kind: Literal["in"] = "in"
    element: "Expr"
    set: "Expr"
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="proposition"))


class SetOpNode(BaseModel):
    kind: Literal["setop"] = "setop"
    op: Literal["union", "intersect", "difference", "complement"]
    args: list["Expr"]
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="set"))


class BoolNode(BaseModel):
    """Boolean combination of propositions (QE results, negation normal form)."""
    kind: Literal["bool"] = "bool"
    op: Literal["and", "or", "not"]
    args: list["Expr"]
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="proposition"))


class QuantifierNode(BaseModel):
    kind: Literal["quantifier"] = "quantifier"
    quantifier: Literal["forall", "exists"]
    variable: str
    domain: "Expr | None" = None
    body: "Expr"
    meta: SemanticMeta = Field(default_factory=lambda: SemanticMeta(value_type="proposition"))


class BinderNode(BaseModel):
    """Sum or product of a body expression as a variable ranges over a finite set."""
    kind: Literal["binder"] = "binder"
    op: Literal["sum", "product"]
    variable: str
    domain: "Expr"
    body: "Expr"
    meta: SemanticMeta = Field(default_factory=SemanticMeta)


Expr = Annotated[
    IntegerNode | RationalNode | RealNode | AlgebraicNumberNode | ComplexNode | IntervalNode | NumberNode |
    SymbolNode | UnaryNode | NaryNode | BinaryNode | CallNode | RelationNode |
    SetNode | MembershipNode | SetOpNode | BoolNode | QuantifierNode | BinderNode,
    Field(discriminator="kind"),
]

for cls in (AlgebraicNumberNode, ComplexNode, IntervalNode, UnaryNode, NaryNode, BinaryNode, CallNode, RelationNode,
            SetNode, MembershipNode, SetOpNode, BoolNode, QuantifierNode, BinderNode):
    cls.model_rebuild()


class ParseAmbiguity(BaseModel):
    span: str
    reason: str
    candidates: list[str] = Field(default_factory=list)
    severity: Literal["warning", "error"] = "warning"


class Assumption(BaseModel):
    expression: Expr
    source: Literal["asserted", "derived", "solver", "formal"] = "asserted"
    provenance: str | None = None


class DerivedFact(BaseModel):
    subject: str
    predicate: str
    value: str | bool
    source: Literal["assumption_inference", "solver", "formal"] = "assumption_inference"
    provenance: str | None = None


class MathContext(BaseModel):
    context_id: str
    domains: dict[str, str] = Field(default_factory=dict)
    symbol_properties: dict[str, list[str]] = Field(default_factory=dict)
    assumptions: list[Assumption] = Field(default_factory=list)
    derived_facts: list[DerivedFact] = Field(default_factory=list)
    consistency: Literal["unchecked", "consistent", "inconsistent", "unknown"] = "unchecked"


class SolutionSet(BaseModel):
    kind: Literal[
        "finite", "interval", "union", "conditional", "parametric",
        "empty", "universal", "symbolic", "unknown"
    ]
    domain: str
    values: list[str] = Field(default_factory=list)
    intervals: list[dict] = Field(default_factory=list)
    parts: list["SolutionSet"] = Field(default_factory=list)
    condition: str | None = None
    expression: str | None = None
    parameter: str | None = None


SolutionSet.model_rebuild()


class EngineEvidence(BaseModel):
    engine: str
    capability: str
    status: VerificationStatus | Literal["computed", "unavailable", "error"]
    trust: TrustLevel = TrustLevel.UNKNOWN
    detail: dict = Field(default_factory=dict)
    error: str | None = None
    # Defaults preserve conjunctive legacy dependencies. Independent verifier
    # attempts must explicitly declare their role and alternative support path.
    role: Literal["required", "cross_check", "diagnostic"] = "required"
    support_path: str = "primary"


def _legacy_evidence_bundle(
    records: list[EngineEvidence],
    assumptions: list[str] | None = None,
    side_conditions: list[str] | None = None,
) -> EvidenceBundle:
    """Project legacy engine records without strengthening their claims."""
    bundle = EvidenceBundle()
    for record in records:
        level = (TrustLevel.UNKNOWN if record.status in
                 {VerificationStatus.UNKNOWN, "unavailable", "error"} else record.trust)
        bundle.computation.append(ComputationEvidence(
            engine=record.engine,
            method=record.capability,
            arithmetic=level.value,
            deterministic=bool(record.detail.get("deterministic", True)),
            trust=level.value,
            metadata=dict(record.detail),
            role=record.role,
            support_path=record.support_path,
        ))
        if record.status == VerificationStatus.PROVED:
            bundle.proof.append(ProofEvidence(
                proposition=str(record.detail.get(
                    "proposition", record.capability)),
                method=record.capability,
                engine=record.engine,
                certificate=record.detail.get("certificate"),
                assumptions=list(assumptions or []),
                side_conditions=list(side_conditions or []),
                verified=True,
                trust=record.trust.value,
                role=record.role,
                support_path=record.support_path,
            ))
    return bundle


class DerivationStep(BaseModel):
    step_id: str
    operation: str
    inputs: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    output: str | None = None
    output_expr_id: str | None = None
    conditions: list[str] = Field(default_factory=list)
    engine: str | None = None
    trust: TrustLevel = TrustLevel.UNKNOWN
    evidence: list[EngineEvidence] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_compatible_evidence(self) -> "DerivationStep":
        if self.evidence_bundle.is_empty() and (
            self.evidence or self.engine or self.trust != TrustLevel.UNKNOWN
        ):
            records = self.evidence or [EngineEvidence(
                engine=self.engine or "unknown",
                capability=self.operation,
                status="computed",
                trust=self.trust,
            )]
            self.evidence_bundle = _legacy_evidence_bundle(
                records, side_conditions=self.conditions)
        if not self.claim_evidence and not self.evidence_bundle.is_empty():
            self.claim_evidence["result"] = self.evidence_bundle.model_copy(
                deep=True)
        return self


class Obligation(BaseModel):
    obligation_id: str
    kind: Literal["symbolic", "smt", "formal", "numeric", "domain", "number_theory", "verification"]
    action: str
    statement: str
    required_capabilities: list[str] = Field(default_factory=list)
    preferred_engines: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)
    produces: list[str] = Field(default_factory=list)
    capability_ref: str | None = None
    resolved_handler: str | None = None


class ProblemPlan(BaseModel):
    classification: str
    variables: list[str] = Field(default_factory=list)
    domains: dict[str, str | None] = Field(default_factory=dict)
    missing_assumptions: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    strategy: list[str] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    capability_routes: dict[str, list[str]] = Field(default_factory=dict)
    formal_verification_possible: bool = False


class ObligationExecution(BaseModel):
    obligation_id: str
    kind: str
    action: str
    state: Literal["pending", "running", "succeeded", "verified", "refuted", "unknown", "skipped", "failed"] = "pending"
    engine: str | None = None
    trust: TrustLevel = TrustLevel.UNKNOWN
    depends_on: list[str] = Field(default_factory=list)
    assumptions_used: list[str] = Field(default_factory=list)
    side_conditions: list[str] = Field(default_factory=list)
    result: dict = Field(default_factory=dict)
    evidence: list[EngineEvidence] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)
    derivation_step_id: str | None = None
    error: str | None = None
    # Obligations discovered mid-execution; the executor appends them to the
    # DAG after the wave that produced them completes.
    followups: list["Obligation"] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_compatible_evidence(self) -> "ObligationExecution":
        if self.evidence_bundle.is_empty() and self.evidence:
            self.evidence_bundle = _legacy_evidence_bundle(
                self.evidence, self.assumptions_used, self.side_conditions)
        if not self.claim_evidence and not self.evidence_bundle.is_empty():
            self.claim_evidence["result"] = self.evidence_bundle.model_copy(
                deep=True)
        return self


class PlanExecution(BaseModel):
    execution_id: str
    plan_id: str
    expr_id: str
    context_id: str | None = None
    state: Literal["completed", "partial", "refuted", "failed"]
    final_status: Literal["verified", "refuted", "unknown", "error", "ok"]
    final_trust: TrustLevel = TrustLevel.UNKNOWN
    obligations: list[ObligationExecution] = Field(default_factory=list)
    propagated_assumptions: list[str] = Field(default_factory=list)
    propagated_side_conditions: list[str] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    graph: dict = Field(default_factory=dict)
    summary: dict = Field(default_factory=dict)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)


class MathResult(BaseModel):
    ok: bool
    status: Literal[
        "verified", "refuted", "candidate", "unknown", "error", "ok",
    ] = "ok"
    data: dict = Field(default_factory=dict)
    assumptions_used: list[str] = Field(default_factory=list)
    side_conditions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trust: TrustLevel = TrustLevel.UNKNOWN
    engine: str | None = None
    derivation: list[DerivationStep] = Field(default_factory=list)
    evidence: list[EngineEvidence] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)
    semantic_status: ResultStatus | None = None
    engine_versions: dict[str, str] = Field(default_factory=dict)
    mathkernel_version: str | None = None
    arithmetic_transition: dict[str, str] = Field(default_factory=dict)

    def reconciled_trust(self, required_claims: list[str] | None = None) -> TrustLevel:
        """Summarize the evidence needed for a conclusion without changing its claims."""
        if not self.claim_evidence:
            return TrustLevel(self.evidence_bundle.conservative_trust())
        level = reconcile_claim_evidence(self.claim_evidence, required_claims)
        return TrustLevel(level)

    @model_validator(mode="after")
    def populate_compatible_evidence(self) -> "MathResult":
        """Project legacy result metadata into the additive evidence model."""
        if self.evidence_bundle.is_empty() and (
            self.engine or self.evidence or self.trust != TrustLevel.UNKNOWN
        ) and not self.claim_evidence:
            records = self.evidence or [
                EngineEvidence(
                    engine=self.engine or "unknown",
                    capability="legacy_result",
                    status="computed",
                    trust=self.trust,
                )
            ]
            self.evidence_bundle = _legacy_evidence_bundle(
                records, self.assumptions_used, self.side_conditions)
        if self.evidence_bundle.is_empty() and "result" in self.claim_evidence:
            self.evidence_bundle = self.claim_evidence["result"].model_copy(deep=True)
        if not self.claim_evidence and not self.evidence_bundle.is_empty():
            self.claim_evidence["result"] = self.evidence_bundle.model_copy(deep=True)
        else:
            self.evidence_bundle = self.evidence_bundle.model_copy(deep=True)
            self.claim_evidence = {
                name: bundle.model_copy(deep=True)
                for name, bundle in self.claim_evidence.items()
            }
        if "result" in self.claim_evidence:
            supported_level = self.claim_evidence["result"].conservative_trust()
        elif self.claim_evidence:
            supported_level = reconcile_claim_evidence(self.claim_evidence)
        else:
            supported_level = self.evidence_bundle.conservative_trust()
        supported_trust = TrustLevel(supported_level)
        if TRUST_RANK[supported_trust.value] < TRUST_RANK[self.trust.value]:
            self.trust = supported_trust

        if not self.ok:
            derived_status = ResultStatus.ERROR
        elif self.status == "refuted":
            derived_status = ResultStatus.REFUTED
        elif self.status == "unknown":
            derived_status = ResultStatus.UNKNOWN
        elif self.trust == TrustLevel.FORMAL:
            derived_status = ResultStatus.PROVED
        elif self.trust == TrustLevel.EXACT:
            derived_status = ResultStatus.VERIFIED_EXACT
        elif self.trust == TrustLevel.INTERVAL_CERTIFIED:
            derived_status = ResultStatus.CERTIFIED
        elif self.trust == TrustLevel.SYMBOLIC:
            derived_status = (
                ResultStatus.VERIFIED_SYMBOLIC
                if self.status == "verified" else ResultStatus.CANDIDATE
            )
        elif self.trust in {TrustLevel.NUMERIC_HIGH_PRECISION, TrustLevel.NUMERIC}:
            derived_status = (
                ResultStatus.VERIFIED_NUMERIC
                if self.status == "verified" else ResultStatus.NUMERIC
            )
        elif self.trust == TrustLevel.EMPIRICAL:
            derived_status = ResultStatus.EMPIRICAL
        else:
            derived_status = ResultStatus.CANDIDATE

        status_requirements = {
            ResultStatus.PROVED: TrustLevel.FORMAL,
            ResultStatus.VERIFIED_EXACT: TrustLevel.EXACT,
            ResultStatus.VERIFIED_SYMBOLIC: TrustLevel.SYMBOLIC,
            ResultStatus.CERTIFIED: TrustLevel.INTERVAL_CERTIFIED,
            ResultStatus.VERIFIED_NUMERIC: TrustLevel.NUMERIC,
            ResultStatus.NUMERIC: TrustLevel.NUMERIC,
            ResultStatus.EMPIRICAL: TrustLevel.EMPIRICAL,
        }
        required_trust = status_requirements.get(self.semantic_status)
        status_overclaims = (
            required_trust is not None
            and TRUST_RANK[self.trust.value] < TRUST_RANK[required_trust.value]
        )
        mathematical_outcomes = {
            ResultStatus.DOES_NOT_EXIST,
            ResultStatus.UNDEFINED,
            ResultStatus.INFEASIBLE,
            ResultStatus.UNBOUNDED,
            ResultStatus.UNSUPPORTED,
        }
        if (
            self.semantic_status is None
            or status_overclaims
            or (not self.ok and self.semantic_status not in mathematical_outcomes)
            or (
                self.status == "unknown"
                and self.semantic_status not in mathematical_outcomes
            )
        ):
            self.semantic_status = derived_status
        return self

ObligationExecution.model_rebuild()
