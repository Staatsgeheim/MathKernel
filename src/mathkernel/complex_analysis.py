# =============================================================================
# MathKernel - conservative symbolic complex analysis
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Typed, evidence-carrying complex analysis over caller-supplied SymPy objects."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Literal

import sympy as sp
from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator,
)

from mathkernel_artifacts import ComputationEvidence, EvidenceBundle, ProofEvidence
from mathkernel_artifacts.evidence import TRUST_RANK
from .models import OperationStatus


def _json_symbolic(value):
    if isinstance(value, sp.Basic):
        return sp.sstr(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_symbolic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_symbolic(item) for item in value]
    return value


class SymbolicModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer(mode="plain", when_used="json")
    def serialize_symbolic_model(self) -> dict[str, Any]:
        return {
            key: _json_symbolic(value)
            for key, value in self.__dict__.items()
        }


ResultStatus = OperationStatus


@lru_cache(maxsize=512)
def _differentiate_cached(expression: sp.Basic, variable: sp.Symbol) -> sp.Basic:
    return sp.diff(expression, variable)


@lru_cache(maxsize=512)
def _singularities_cached(
    expression: sp.Basic, variable: sp.Symbol,
) -> sp.Set:
    return sp.singularities(expression, variable)


class BranchConvention(SymbolicModel):
    """Explicit choices for the multivalued functions supported by this module."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logarithm_branch: int = 0
    square_root_sign: Literal[1, -1] = 1
    power_log_branch: int = 0
    inverse_trig_branch: int = 0
    lambertw_branch: int = 0
    cuts: list[Any] = Field(default_factory=list)

    @field_validator("cuts")
    @classmethod
    def _cuts_are_sympy_sets(cls, cuts: list[Any]) -> list[Any]:
        if any(not isinstance(cut, sp.Set) for cut in cuts):
            raise TypeError("branch cuts must be caller-supplied SymPy Set objects")
        return cuts


class ComplexDomain(SymbolicModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    variable: Any
    region: Any | None = None
    assumptions: list[Any] = Field(default_factory=list)
    excluded_points: list[Any] = Field(default_factory=list)
    branch: BranchConvention = Field(default_factory=BranchConvention)
    input_trust: str = "symbolic"

    @field_validator("variable")
    @classmethod
    def _variable_is_symbol(cls, value: Any) -> sp.Symbol:
        if not isinstance(value, sp.Symbol):
            raise TypeError("variable must be a caller-supplied SymPy Symbol")
        return value

    @field_validator("assumptions", "excluded_points")
    @classmethod
    def _items_are_sympy(cls, values: list[Any]) -> list[Any]:
        if any(not isinstance(value, sp.Basic) for value in values):
            raise TypeError("domain mathematics must be caller-supplied SymPy objects")
        return values

    @field_validator("region")
    @classmethod
    def _region_is_sympy_set(cls, value: Any | None) -> sp.Set | None:
        if value is not None and not isinstance(value, sp.Set):
            raise TypeError("region must be a caller-supplied SymPy Set")
        return value

    @field_validator("input_trust")
    @classmethod
    def _valid_domain_trust(cls, value: str) -> str:
        if value not in TRUST_RANK:
            raise ValueError(f"unknown input trust: {value}")
        return value


class Contour(SymbolicModel):
    """An explicit, piecewise-linear, closed contour."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vertices: list[Any]
    orientation: Literal["ccw", "cw"] | None = None
    assumptions: list[Any] = Field(default_factory=list)
    input_trust: str = "exact"

    @field_validator("vertices")
    @classmethod
    def _vertices_are_sympy(cls, values: list[Any]) -> list[Any]:
        if len(values) < 4:
            raise ValueError("a closed polygon needs at least three edges")
        if any(not isinstance(value, sp.Basic) for value in values):
            raise TypeError("vertices must be caller-supplied SymPy expressions")
        if sp.simplify(values[0] - values[-1]) != 0:
            raise ValueError("piecewise-linear contour must be explicitly closed")
        return values

    @field_validator("assumptions")
    @classmethod
    def _assumptions_are_sympy(cls, values: list[Any]) -> list[Any]:
        if any(not isinstance(value, sp.Basic) for value in values):
            raise TypeError("assumptions must be caller-supplied SymPy objects")
        return values

    @field_validator("input_trust")
    @classmethod
    def _valid_contour_trust(cls, value: str) -> str:
        if value not in TRUST_RANK:
            raise ValueError(f"unknown input trust: {value}")
        return value


class ComplexFunction(SymbolicModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    expression: Any
    variable: Any
    domain: ComplexDomain | None = None
    branch: BranchConvention = Field(default_factory=BranchConvention)
    input_trust: str = "symbolic"

    @field_validator("expression")
    @classmethod
    def _expression_is_sympy(cls, value: Any) -> sp.Basic:
        if not isinstance(value, sp.Basic):
            raise TypeError("expression must be a caller-supplied SymPy expression")
        return value

    @field_validator("variable")
    @classmethod
    def _function_variable_is_symbol(cls, value: Any) -> sp.Symbol:
        if not isinstance(value, sp.Symbol):
            raise TypeError("variable must be a caller-supplied SymPy Symbol")
        return value

    @model_validator(mode="after")
    def _domain_variable_matches(self) -> "ComplexFunction":
        if self.domain is not None and self.domain.variable != self.variable:
            raise ValueError("function and domain variables differ")
        return self

    @field_validator("input_trust")
    @classmethod
    def _valid_input_trust(cls, value: str) -> str:
        if value not in TRUST_RANK:
            raise ValueError(f"unknown input trust: {value}")
        return value


class Singularity(SymbolicModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    point: Any
    kind: Literal["pole", "removable", "essential", "branch_point", "unknown"] = "unknown"
    order: int | None = Field(default=None, ge=1)
    assumptions: list[Any] = Field(default_factory=list)

    @field_validator("point")
    @classmethod
    def _point_is_sympy(cls, value: Any) -> sp.Basic:
        if not isinstance(value, sp.Basic):
            raise TypeError("point must be a caller-supplied SymPy expression")
        return value

    @model_validator(mode="after")
    def _pole_has_consistent_order(self) -> "Singularity":
        if self.kind == "pole" and self.order is None:
            raise ValueError("a supplied pole must include its order")
        if self.kind != "pole" and self.order is not None:
            raise ValueError("only poles have an order")
        return self


class ComplexResult(SymbolicModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: ResultStatus
    value: Any | None = None
    trust: str = "unknown"
    assumptions: list[Any] = Field(default_factory=list)
    branch: BranchConvention | None = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    diagnostics: list[str] = Field(default_factory=list)


class AnalyticityResult(ComplexResult):
    derivative: Any | None = None
    candidate_singularities: list[Any] = Field(default_factory=list)


class SingularityResult(ComplexResult):
    singularity: Singularity | None = None


class ResidueResult(ComplexResult):
    point: Any
    pole_order: int | None = None
    checks: dict[str, Any] = Field(default_factory=dict)


class LaurentSeriesResult(ComplexResult):
    point: Any
    expansion: Any | None = None
    requested_order: int


class WindingNumberResult(ComplexResult):
    point: Any
    winding_number: int | None = None
    orientation: Literal["ccw", "cw"] | None = None


class ContourIntegralResult(ComplexResult):
    integral: Any | None = None
    orientation: Literal["ccw", "cw"] | None = None
    enclosed_singularities: list[Singularity] = Field(default_factory=list)
    winding_numbers: dict[str, int] = Field(default_factory=dict)
    residues: dict[str, Any] = Field(default_factory=dict)
    branch_cuts: list[Any] = Field(default_factory=list)


class ArgumentPrincipleResult(ComplexResult):
    zero_count: int | None = None
    pole_count: int | None = None
    zero_minus_pole: int | None = None
    logarithmic_derivative_integral: Any | None = None
    zero_multiplicities: dict[str, int] = Field(default_factory=dict)
    pole_multiplicities: dict[str, int] = Field(default_factory=dict)
    winding_numbers: dict[str, int] = Field(default_factory=dict)
    enclosed_zeros: list[Any] = Field(default_factory=list)
    enclosed_poles: list[Singularity] = Field(default_factory=list)
    contour_obligations: list[str] = Field(default_factory=list)
    branch_cuts: list[Any] = Field(default_factory=list)


class AnalyticContinuationResult(ComplexResult):
    continuation: ComplexFunction | None = None
    source_domain: ComplexDomain | None = None
    target_domain: ComplexDomain | None = None
    overlap: Any | None = None
    identity_established: bool = False


class ConformalMapResult(ComplexResult):
    mapped_expression: Any | None = None
    source_domain: ComplexDomain | None = None
    image_domain: Any | None = None
    derivative: Any | None = None
    critical_points: Any | None = None
    nondegenerate_on_domain: bool | None = None


def _cap_symbolic(input_trust: str) -> str:
    return min(("symbolic", input_trust), key=TRUST_RANK.__getitem__)


def _sympy_input_trust(*values: sp.Basic) -> str:
    """Infer the ancestry ceiling of direct SymPy values when no kernel record exists."""
    return "numeric" if any(
        isinstance(value, sp.Float) or bool(getattr(value, "atoms", lambda *_: set())(sp.Float))
        for value in values
    ) else "exact"


def _symbolic_evidence(
    method: str,
    proposition: str | None = None,
    input_trust: str = "symbolic",
) -> EvidenceBundle:
    trust = _cap_symbolic(input_trust)
    evidence = EvidenceBundle(computation=[
        ComputationEvidence(
            engine="caller",
            method="input_ancestry",
            arithmetic="inherited",
            deterministic=True,
            trust=input_trust,
        ),
        ComputationEvidence(
            engine="sympy",
            method=method,
            arithmetic="symbolic",
            deterministic=True,
            trust=trust,
        ),
    ])
    if proposition is not None:
        evidence.proof.append(ProofEvidence(
            proposition=proposition,
            method="independent_symbolic_checks",
            engine="sympy",
            verified=True,
            trust=trust,
        ))
    return evidence


def _exact_evidence(method: str, proposition: str) -> EvidenceBundle:
    return EvidenceBundle(
        computation=[ComputationEvidence(
            engine="mathkernel",
            method=method,
            arithmetic="exact_symbolic",
            deterministic=True,
            trust="exact",
        )],
        proof=[ProofEvidence(
            proposition=proposition,
            method="exact_combinatorial_check",
            engine="mathkernel",
            verified=True,
            trust="exact",
        )],
    )


def _unknown(diagnostic: str, *, branch: BranchConvention | None = None,
             assumptions: list[Any] | None = None) -> ComplexResult:
    return ComplexResult(
        status=ResultStatus.UNKNOWN,
        branch=branch,
        assumptions=assumptions or [],
        diagnostics=[diagnostic],
    )


class ComplexAnalysisEngine:
    """Conservative symbolic operations; no textual mathematics is accepted."""

    _INVERSE_TRIG = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc)

    @staticmethod
    def _branch_families(expression: sp.Basic) -> set[str]:
        families: set[str] = set()
        if expression.has(sp.log):
            families.add("log")
        for power in expression.atoms(sp.Pow):
            if power.exp == sp.S.Half:
                families.add("sqrt")
            elif not power.exp.is_integer:
                families.add("pow")
        if any(expression.has(function) for function in ComplexAnalysisEngine._INVERSE_TRIG):
            families.add("inverse_trig")
        if expression.has(sp.LambertW):
            families.add("lambertw")
        return families

    def branch_compatible(
        self,
        expression: sp.Basic,
        left: BranchConvention,
        right: BranchConvention,
    ) -> ComplexResult:
        if not isinstance(expression, sp.Basic):
            raise TypeError("expression must be a caller-supplied SymPy expression")
        families = self._branch_families(expression)
        mismatches: list[str] = []
        comparisons = {
            "log": ("logarithm_branch", left.logarithm_branch, right.logarithm_branch),
            "sqrt": ("square_root_sign", left.square_root_sign, right.square_root_sign),
            "pow": ("power_log_branch", left.power_log_branch, right.power_log_branch),
            "inverse_trig": (
                "inverse_trig_branch", left.inverse_trig_branch, right.inverse_trig_branch),
            "lambertw": ("lambertw_branch", left.lambertw_branch, right.lambertw_branch),
        }
        for family in families:
            name, lhs, rhs = comparisons[family]
            if lhs != rhs:
                mismatches.append(f"{name}: {lhs} != {rhs}")
        if families and left.cuts != right.cuts:
            mismatches.append("branch cuts differ")
        if mismatches:
            return ComplexResult(
                status=ResultStatus.ERROR,
                value=False,
                trust="exact",
                branch=left,
                evidence=_exact_evidence(
                    "branch_metadata_comparison", "branches are incompatible"),
                diagnostics=["incompatible branch conventions", *mismatches],
            )
        return ComplexResult(
            status=ResultStatus.VERIFIED,
            value=True,
            trust="exact",
            branch=left,
            evidence=_exact_evidence("branch_metadata_comparison", "branches are compatible"),
        )

    def derivative(self, function: ComplexFunction) -> ComplexResult:
        derivative = _differentiate_cached(function.expression, function.variable)
        assumptions = list(function.domain.assumptions) if function.domain else []
        return ComplexResult(
            status=ResultStatus.CANDIDATE,
            value=derivative,
            trust=_cap_symbolic(function.input_trust),
            assumptions=assumptions,
            branch=function.branch,
            evidence=_symbolic_evidence(
                "differentiate", input_trust=function.input_trust),
            diagnostics=["formal analyticity of the ambient domain is not inferred from differentiation"],
        )

    def analyticity_candidate(self, function: ComplexFunction) -> AnalyticityResult:
        derivative = _differentiate_cached(function.expression, function.variable)
        denominator = sp.denom(sp.together(function.expression))
        candidates: list[Any] = []
        if denominator != 1:
            try:
                solved = sp.solveset(denominator, function.variable, domain=sp.S.Complexes)
                if isinstance(solved, sp.FiniteSet):
                    candidates.extend(list(solved))
            except (NotImplementedError, ValueError, TypeError):
                pass
        diagnostics = ["candidate only: branch points and unresolved denominator zeros may remain"]
        families = self._branch_families(function.expression)
        if families:
            diagnostics.append(f"branch-sensitive families: {', '.join(sorted(families))}")
        assumptions = list(function.domain.assumptions) if function.domain else []
        return AnalyticityResult(
            status=ResultStatus.CANDIDATE,
            value=derivative,
            derivative=derivative,
            candidate_singularities=candidates,
            trust=_cap_symbolic(function.input_trust),
            assumptions=assumptions,
            branch=function.branch,
            evidence=_symbolic_evidence(
                "differentiate_and_denominator_candidates",
                input_trust=function.input_trust),
            diagnostics=diagnostics,
        )

    def zeros(self, function: ComplexFunction) -> ComplexResult:
        try:
            zeros = sp.solveset(
                function.expression, function.variable, domain=sp.S.Complexes)
        except (NotImplementedError, ValueError, TypeError) as exc:
            return _unknown(
                f"zero set could not be computed: {exc}", branch=function.branch)
        resolved = isinstance(zeros, sp.FiniteSet)
        return ComplexResult(
            status=ResultStatus.VERIFIED if resolved else ResultStatus.CANDIDATE,
            value=zeros,
            trust=_cap_symbolic(function.input_trust),
            branch=function.branch,
            evidence=_symbolic_evidence(
                "complex_zero_solve", input_trust=function.input_trust),
            diagnostics=[] if resolved else [
                "zero set is symbolic and was not established as finite"],
        )

    def singularities(self, function: ComplexFunction) -> ComplexResult:
        discovered = self._discover_singularities(function)
        if discovered is None:
            return _unknown(
                "complete singularity set was not established",
                branch=function.branch)
        classified = [
            self.classify_singularity(function, point).singularity
            for point in sorted(discovered, key=sp.default_sort_key)
        ]
        return ComplexResult(
            status=(
                ResultStatus.VERIFIED if all(
                    item is not None and item.kind != "unknown"
                    for item in classified)
                else ResultStatus.CANDIDATE
            ),
            value=classified,
            trust=_cap_symbolic(function.input_trust),
            branch=function.branch,
            evidence=_symbolic_evidence(
                "singularity_discovery_and_classification",
                input_trust=function.input_trust),
        )

    def conformal_at(
        self, function: ComplexFunction, point: sp.Basic,
    ) -> ComplexResult:
        if not isinstance(point, sp.Basic):
            raise TypeError("point must be a caller-supplied SymPy expression")
        derivative = _differentiate_cached(
            function.expression, function.variable)
        value = sp.simplify(derivative.subs(function.variable, point))
        if value.is_zero is None:
            return ComplexResult(
                status=ResultStatus.UNKNOWN,
                value=value,
                branch=function.branch,
                diagnostics=["nonzero derivative could not be established"],
            )
        is_conformal = value.is_zero is False
        return ComplexResult(
            status=ResultStatus.VERIFIED,
            value=is_conformal,
            trust=_cap_symbolic(function.input_trust),
            branch=function.branch,
            evidence=_symbolic_evidence(
                "local_nonzero_derivative_test",
                f"derivative at {point} is {'nonzero' if is_conformal else 'zero'}",
                function.input_trust),
        )

    @staticmethod
    def _effective_region(domain: ComplexDomain) -> sp.Set | None:
        if domain.region is None:
            return None
        if not domain.excluded_points:
            return domain.region
        return sp.Complement(domain.region, sp.FiniteSet(*domain.excluded_points))

    @staticmethod
    def _set_is_empty(value: sp.Set) -> bool | None:
        if value == sp.S.EmptySet or value.is_empty is True:
            return True
        if value.is_empty is False:
            return False
        return None

    def conformal_map(
        self,
        function: ComplexFunction,
        source_domain: ComplexDomain | None = None,
    ) -> ConformalMapResult:
        domain = source_domain or function.domain
        derivative = _differentiate_cached(function.expression, function.variable)
        if domain is None:
            return ConformalMapResult(
                status=ResultStatus.UNKNOWN,
                mapped_expression=function.expression,
                derivative=derivative,
                branch=function.branch,
                diagnostics=["a source domain is required to map a domain"],
            )
        if domain.variable != function.variable:
            raise ValueError("function and source-domain variables differ")
        compatibility = self.branch_compatible(
            function.expression, function.branch, domain.branch)
        if compatibility.status != ResultStatus.VERIFIED:
            return ConformalMapResult(
                status=ResultStatus.ERROR,
                mapped_expression=function.expression,
                source_domain=domain,
                derivative=derivative,
                branch=function.branch,
                diagnostics=["function and source-domain branches are incompatible",
                             *compatibility.diagnostics],
            )
        region = self._effective_region(domain)
        if region is None:
            return ConformalMapResult(
                status=ResultStatus.UNKNOWN,
                mapped_expression=function.expression,
                source_domain=domain,
                derivative=derivative,
                branch=function.branch,
                assumptions=list(domain.assumptions),
                diagnostics=["source domain has no explicit SymPy region"],
            )
        image = sp.ImageSet(sp.Lambda(function.variable, function.expression), region)
        try:
            critical = sp.solveset(derivative, function.variable, domain=region)
        except (NotImplementedError, ValueError, TypeError) as exc:
            return ConformalMapResult(
                status=ResultStatus.UNKNOWN,
                mapped_expression=function.expression,
                source_domain=domain,
                image_domain=image,
                derivative=derivative,
                branch=function.branch,
                assumptions=list(domain.assumptions),
                diagnostics=[f"critical-point set could not be established: {exc}"],
            )
        empty = self._set_is_empty(critical)
        if empty is None:
            return ConformalMapResult(
                status=ResultStatus.UNKNOWN,
                mapped_expression=function.expression,
                source_domain=domain,
                image_domain=image,
                derivative=derivative,
                critical_points=critical,
                branch=function.branch,
                assumptions=list(domain.assumptions),
                diagnostics=["SymPy did not decide whether the derivative vanishes in the domain"],
            )
        status = ResultStatus.VERIFIED
        diagnostics: list[str] = []
        if self._branch_families(function.expression):
            status = ResultStatus.CANDIDATE
            diagnostics.append(
                "derivative nondegeneracy was checked, but domain analyticity of the "
                "branch-sensitive expression was not established")
        else:
            discovered = self._discover_singularities(function)
            if discovered is None:
                status = ResultStatus.CANDIDATE
                diagnostics.append(
                    "derivative nondegeneracy was checked, but complete singularity "
                    "exclusion was not established")
            else:
                for point in discovered:
                    membership = region.contains(point)
                    if membership is not sp.S.false:
                        status = ResultStatus.CANDIDATE
                        diagnostics.append(
                            f"source-domain exclusion of singularity {point} was not established")
        return ConformalMapResult(
            status=status,
            value=image,
            mapped_expression=function.expression,
            source_domain=domain,
            image_domain=image,
            derivative=derivative,
            critical_points=critical,
            nondegenerate_on_domain=empty,
            trust=_cap_symbolic(function.input_trust),
            assumptions=list(domain.assumptions),
            branch=function.branch,
            evidence=_symbolic_evidence(
                "imageset_and_domain_critical_point_solve",
                ("derivative is nonzero on the source domain" if empty
                 else "derivative has a zero in the source domain"),
                function.input_trust,
            ),
            diagnostics=diagnostics + ([] if empty else [
                "mapping is not conformal throughout the source domain"]),
        )

    def analytic_continuation(
        self, function: ComplexFunction, target_domain: ComplexDomain,
    ) -> AnalyticContinuationResult:
        if target_domain.variable != function.variable:
            raise ValueError("function and target-domain variables differ")
        source = function.domain
        if source is None:
            return AnalyticContinuationResult(
                status=ResultStatus.UNSUPPORTED,
                target_domain=target_domain,
                branch=function.branch,
                assumptions=list(target_domain.assumptions),
                diagnostics=["identity continuation requires an explicit source domain"],
            )
        source_compatibility = self.branch_compatible(
            function.expression, function.branch, source.branch)
        target_compatibility = self.branch_compatible(
            function.expression, source.branch, target_domain.branch)
        if (source_compatibility.status != ResultStatus.VERIFIED
                or target_compatibility.status != ResultStatus.VERIFIED):
            return AnalyticContinuationResult(
                status=ResultStatus.ERROR,
                source_domain=source,
                target_domain=target_domain,
                branch=function.branch,
                diagnostics=["source and target branches are incompatible",
                             *source_compatibility.diagnostics,
                             *target_compatibility.diagnostics],
            )
        source_region = self._effective_region(source)
        target_region = self._effective_region(target_domain)
        if source_region is None or target_region is None:
            return AnalyticContinuationResult(
                status=ResultStatus.UNKNOWN,
                source_domain=source,
                target_domain=target_domain,
                branch=function.branch,
                assumptions=list(target_domain.assumptions),
                diagnostics=["overlap requires explicit caller-supplied SymPy regions"],
            )
        overlap = source_region.intersect(target_region)
        overlap_empty = self._set_is_empty(overlap)
        if overlap_empty is True:
            return AnalyticContinuationResult(
                status=ResultStatus.UNSUPPORTED,
                source_domain=source,
                target_domain=target_domain,
                overlap=overlap,
                branch=function.branch,
                diagnostics=["source and target domains have no overlap"],
            )
        if overlap_empty is None:
            return AnalyticContinuationResult(
                status=ResultStatus.UNKNOWN,
                source_domain=source,
                target_domain=target_domain,
                overlap=overlap,
                branch=function.branch,
                diagnostics=["SymPy could not establish a nonempty domain overlap"],
            )
        if self._branch_families(function.expression):
            return AnalyticContinuationResult(
                status=ResultStatus.UNSUPPORTED,
                source_domain=source,
                target_domain=target_domain,
                overlap=overlap,
                branch=function.branch,
                diagnostics=[
                    "identity continuation of branch-sensitive expressions is not established"],
            )
        discovered = self._discover_singularities(function)
        if discovered is None:
            return AnalyticContinuationResult(
                status=ResultStatus.UNKNOWN,
                source_domain=source,
                target_domain=target_domain,
                overlap=overlap,
                branch=function.branch,
                diagnostics=["complete singularity set was not established"],
            )
        for point in discovered:
            for name, region in (("source", source_region), ("target", target_region)):
                membership = region.contains(point)
                if membership is sp.S.true:
                    return AnalyticContinuationResult(
                        status=ResultStatus.UNSUPPORTED,
                        source_domain=source,
                        target_domain=target_domain,
                        overlap=overlap,
                        branch=function.branch,
                        diagnostics=[f"{name} domain contains singularity {point}"],
                    )
                if membership is not sp.S.false:
                    return AnalyticContinuationResult(
                        status=ResultStatus.UNKNOWN,
                        source_domain=source,
                        target_domain=target_domain,
                        overlap=overlap,
                        branch=function.branch,
                        diagnostics=[
                            f"membership of singularity {point} in {name} is undecidable"],
                    )
        continuation = function.model_copy(
            update={"domain": target_domain, "branch": target_domain.branch})
        return AnalyticContinuationResult(
            status=ResultStatus.VERIFIED,
            value=continuation,
            continuation=continuation,
            source_domain=source,
            target_domain=target_domain,
            overlap=overlap,
            identity_established=True,
            trust=_cap_symbolic(function.input_trust),
            branch=target_domain.branch,
            assumptions=list(target_domain.assumptions),
            evidence=_symbolic_evidence(
                "identity_on_nonempty_overlap_and_singularity_exclusion",
                "the same expression defines an identity continuation",
                function.input_trust,
            ),
        )

    def classify_singularity(
        self, function: ComplexFunction, point: sp.Basic, max_order: int = 12,
    ) -> SingularityResult:
        if not isinstance(point, sp.Basic):
            raise TypeError("point must be a caller-supplied SymPy expression")
        z, expression = function.variable, function.expression
        assumptions = list(function.domain.assumptions) if function.domain else []
        for atom in expression.atoms(sp.exp, sp.sin, sp.cos):
            argument = atom.args[0]
            try:
                argument_singularities = sp.singularities(argument, z)
            except (NotImplementedError, ValueError, TypeError):
                continue
            if point in argument_singularities:
                singularity = Singularity(point=point, kind="essential")
                return SingularityResult(
                    status=ResultStatus.VERIFIED,
                    singularity=singularity,
                    value=singularity,
                    trust=_cap_symbolic(function.input_trust),
                    assumptions=assumptions,
                    branch=function.branch,
                    evidence=_symbolic_evidence(
                        "essential_composition_structure",
                        input_trust=function.input_trust),
                )
        if self._branch_families(expression):
            # A non-meromorphic local expression must not be mislabeled as a pole.
            for atom in expression.atoms(sp.log, sp.LambertW):
                if atom.has(z) and sp.simplify(atom.args[0].subs(z, point)) == 0:
                    singularity = Singularity(point=point, kind="branch_point")
                    return SingularityResult(
                        status=ResultStatus.VERIFIED,
                        singularity=singularity,
                        value=singularity,
                        trust=_cap_symbolic(function.input_trust),
                        assumptions=assumptions,
                        branch=function.branch,
                        evidence=_symbolic_evidence(
                            "branch_point_structure",
                            input_trust=function.input_trust),
                    )
        for order in range(1, max_order + 1):
            try:
                coefficient = sp.simplify(sp.limit((z - point) ** order * expression, z, point))
            except (NotImplementedError, ValueError, TypeError):
                break
            if coefficient in (sp.oo, -sp.oo, sp.zoo, sp.nan) or coefficient.has(sp.oo, sp.zoo, sp.nan):
                continue
            if coefficient.is_zero is False:
                lower_ok = True
                if order > 1:
                    lower = sp.limit((z - point) ** (order - 1) * expression, z, point)
                    lower_ok = lower in (sp.oo, -sp.oo, sp.zoo) or bool(lower.has(sp.oo, sp.zoo))
                if lower_ok:
                    singularity = Singularity(point=point, kind="pole", order=order)
                    return SingularityResult(
                        status=ResultStatus.VERIFIED,
                        singularity=singularity,
                        value=singularity,
                        trust=_cap_symbolic(function.input_trust),
                        assumptions=assumptions,
                        branch=function.branch,
                        evidence=_symbolic_evidence(
                            "pole_defining_limit", f"pole of order {order} at {point}",
                            function.input_trust),
                    )
        try:
            finite_limit = sp.limit(expression, z, point)
            if finite_limit.is_finite is True:
                singularity = Singularity(point=point, kind="removable")
                return SingularityResult(
                    status=ResultStatus.VERIFIED,
                    singularity=singularity,
                    value=singularity,
                    trust=_cap_symbolic(function.input_trust),
                    assumptions=assumptions,
                    branch=function.branch,
                    evidence=_symbolic_evidence(
                        "finite_limit_removability",
                        input_trust=function.input_trust),
                )
        except (NotImplementedError, ValueError, TypeError):
            pass
        singularity = Singularity(point=point, kind="unknown")
        return SingularityResult(
            status=ResultStatus.UNKNOWN,
            singularity=singularity,
            value=singularity,
            assumptions=assumptions,
            branch=function.branch,
            diagnostics=["SymPy did not establish a pole order or removable limit"],
        )

    def residue(
        self,
        function: ComplexFunction,
        point: sp.Basic,
        singularity: Singularity | None = None,
    ) -> ResidueResult:
        classification = singularity
        if classification is None:
            classified = self.classify_singularity(function, point)
            classification = classified.singularity
        if classification is None or classification.kind != "pole" or classification.order is None:
            return ResidueResult(
                status=ResultStatus.UNKNOWN,
                point=point,
                branch=function.branch,
                assumptions=classification.assumptions if classification else [],
                diagnostics=["residue requires an established pole"],
            )
        z, expression, order = function.variable, function.expression, classification.order
        checks: dict[str, Any] = {}
        try:
            defining = sp.simplify(
                sp.limit(
                    sp.diff((z - point) ** order * expression, z, order - 1),
                    z,
                    point,
                ) / sp.factorial(order - 1)
            )
            checks["defining_limit"] = defining
            laurent = sp.series(expression, z, point, max(2, order + 1)).removeO().expand()
            coefficient = sp.simplify(laurent.coeff(z - point, -1))
            checks["laurent_coefficient"] = coefficient
            try:
                backend = sp.simplify(sp.residue(expression, z, point))
                checks["sympy_residue"] = backend
            except (NotImplementedError, ValueError):
                backend = coefficient
            agrees = sp.simplify(defining - coefficient) == 0 and sp.simplify(defining - backend) == 0
        except (NotImplementedError, ValueError, TypeError) as exc:
            return ResidueResult(
                status=ResultStatus.UNKNOWN,
                point=point,
                pole_order=order,
                branch=function.branch,
                assumptions=classification.assumptions,
                diagnostics=[f"independent residue checks were inconclusive: {exc}"],
            )
        if not agrees:
            return ResidueResult(
                status=ResultStatus.ERROR,
                point=point,
                pole_order=order,
                checks=checks,
                branch=function.branch,
                diagnostics=["independent residue definitions disagree"],
            )
        return ResidueResult(
            status=ResultStatus.VERIFIED,
            value=defining,
            point=point,
            pole_order=order,
            checks=checks,
            trust=_cap_symbolic(function.input_trust),
            assumptions=classification.assumptions,
            branch=function.branch,
            evidence=_symbolic_evidence(
                "residue_defining_limit_and_laurent",
                f"residue at {point} equals {defining}",
                function.input_trust,
            ),
        )

    def laurent_series(
        self, function: ComplexFunction, point: sp.Basic, order: int = 6,
    ) -> LaurentSeriesResult:
        if order < 1:
            raise ValueError("series order must be positive")
        try:
            expansion = sp.series(function.expression, function.variable, point, order)
        except (NotImplementedError, ValueError, TypeError) as exc:
            return LaurentSeriesResult(
                status=ResultStatus.UNKNOWN,
                point=point,
                requested_order=order,
                branch=function.branch,
                diagnostics=[f"Laurent expansion unavailable: {exc}"],
            )
        assumptions = list(function.domain.assumptions) if function.domain else []
        return LaurentSeriesResult(
            status=ResultStatus.VERIFIED,
            value=expansion,
            expansion=expansion,
            point=point,
            requested_order=order,
            trust=_cap_symbolic(function.input_trust),
            assumptions=assumptions,
            branch=function.branch,
            evidence=_symbolic_evidence(
                "laurent_series", input_trust=function.input_trust),
        )

    @staticmethod
    def _real_sign(value: sp.Basic) -> int | None:
        value = sp.simplify(value)
        if value.is_positive:
            return 1
        if value.is_negative:
            return -1
        if value.is_zero:
            return 0
        return None

    def winding_number(self, contour: Contour, point: sp.Basic) -> WindingNumberResult:
        if not isinstance(point, sp.Basic):
            raise TypeError("point must be a caller-supplied SymPy expression")
        winding = 0
        signed_area = sp.S.Zero
        for start, end in zip(contour.vertices, contour.vertices[1:]):
            a, b = sp.simplify(start - point), sp.simplify(end - point)
            ax, ay = sp.re(a).expand(complex=True), sp.im(a).expand(complex=True)
            bx, by = sp.re(b).expand(complex=True), sp.im(b).expand(complex=True)
            signs = [self._real_sign(v) for v in (ax, ay, bx, by)]
            if any(sign is None for sign in signs):
                return WindingNumberResult(
                    status=ResultStatus.UNKNOWN,
                    point=point,
                    assumptions=contour.assumptions,
                    diagnostics=["symbolic vertex ordering could not be established"],
                )
            if (ax == 0 and ay == 0) or (bx == 0 and by == 0):
                return WindingNumberResult(
                    status=ResultStatus.ERROR,
                    point=point,
                    assumptions=contour.assumptions,
                    diagnostics=["winding number is undefined for a point on the contour"],
                )
            cross = sp.simplify(ax * by - bx * ay)
            cross_sign = self._real_sign(cross)
            if cross_sign == 0:
                dot_sign = self._real_sign(sp.simplify(ax * bx + ay * by))
                if dot_sign in (-1, 0):
                    return WindingNumberResult(
                        status=ResultStatus.ERROR,
                        point=point,
                        assumptions=contour.assumptions,
                        diagnostics=[
                            "winding number is undefined for a point on the contour"],
                    )
            if ay <= 0 < by and cross_sign == 1:
                winding += 1
            elif by <= 0 < ay and cross_sign == -1:
                winding -= 1
            signed_area += sp.re(start) * sp.im(end) - sp.re(end) * sp.im(start)
        area_sign = self._real_sign(signed_area)
        orientation = "ccw" if area_sign == 1 else "cw" if area_sign == -1 else None
        if contour.orientation is not None and orientation != contour.orientation:
            return WindingNumberResult(
                status=ResultStatus.ERROR,
                point=point,
                winding_number=winding,
                orientation=orientation,
                assumptions=contour.assumptions,
                diagnostics=["declared contour orientation disagrees with vertex order"],
            )
        point_trust = _sympy_input_trust(point)
        trust = min(
            ("exact", contour.input_trust, point_trust),
            key=TRUST_RANK.__getitem__,
        )
        evidence = EvidenceBundle(computation=[
            ComputationEvidence(
                engine="caller", method="contour_input_ancestry",
                arithmetic="inherited", trust=contour.input_trust,
            ),
            ComputationEvidence(
                engine="caller", method="point_input_ancestry",
                arithmetic="inherited", trust=point_trust,
            ),
            ComputationEvidence(
                engine="complex_analysis", method="exact_polygon_ray_crossing",
                arithmetic="exact", trust="exact",
            ),
        ], proof=[ProofEvidence(
            proposition=f"winding number is {winding}",
            method="polygon_ray_crossing_certificate",
            engine="complex_analysis", verified=True, trust="exact",
        )], justified_trust=trust)
        return WindingNumberResult(
            status=ResultStatus.VERIFIED,
            value=sp.Integer(winding),
            point=point,
            winding_number=winding,
            orientation=orientation,
            trust=trust,
            assumptions=contour.assumptions,
            evidence=evidence,
        )

    def _cuts_clear_contour(
        self, function: ComplexFunction, contour: Contour,
    ) -> tuple[bool | None, str]:
        if not self._branch_families(function.expression):
            return True, ""
        cuts = function.branch.cuts
        if not cuts and function.domain is not None:
            cuts = function.domain.branch.cuts
        if not cuts:
            return None, "branch-sensitive expression has no explicit SymPy branch cuts"
        for cut in cuts:
            for start, end in zip(contour.vertices, contour.vertices[1:]):
                segment = sp.Segment(sp.Point(sp.re(start), sp.im(start)),
                                     sp.Point(sp.re(end), sp.im(end)))
                try:
                    intersection = cut.intersect(segment)
                except (NotImplementedError, ValueError, TypeError):
                    return None, "branch-cut/contour intersection could not be established"
                if intersection is not sp.S.EmptySet and intersection != sp.S.EmptySet:
                    return False, "a declared branch cut intersects the contour"
        return True, ""

    @staticmethod
    def _active_branch_cuts(function: ComplexFunction) -> list[Any]:
        if function.branch.cuts:
            return list(function.branch.cuts)
        if function.domain is not None:
            return list(function.domain.branch.cuts)
        return []

    @staticmethod
    def _discover_singularities(function: ComplexFunction) -> set[sp.Basic] | None:
        """Return all singularities only when SymPy produces a finite exact set."""
        try:
            discovered = _singularities_cached(
                function.expression, function.variable)
        except (NotImplementedError, ValueError, TypeError):
            return None
        if discovered == sp.S.EmptySet:
            return set()
        if isinstance(discovered, sp.FiniteSet):
            return set(discovered)
        return None

    @staticmethod
    def _complete_polynomial_roots(
        expression: sp.Basic, variable: sp.Symbol,
    ) -> dict[sp.Basic, int] | None:
        try:
            polynomial = sp.Poly(expression, variable)
            if polynomial.is_zero:
                return None
            roots = sp.roots(polynomial, variable)
        except (sp.PolynomialError, NotImplementedError, ValueError, TypeError):
            return None
        degree = polynomial.degree()
        if degree == 0:
            return {}
        if sum(int(multiplicity) for multiplicity in roots.values()) != degree:
            return None
        return {root: int(multiplicity) for root, multiplicity in roots.items()}

    def argument_principle(
        self,
        function: ComplexFunction,
        contour: Contour,
    ) -> ArgumentPrincipleResult:
        cuts = self._active_branch_cuts(function)
        obligations = [
            "function is meromorphic on and inside the contour",
            "all zeros and poles, with multiplicity, are completely accounted for",
            "the contour passes through no zero or pole",
            "the declared branch is single-valued on and inside the contour",
        ]
        if self._branch_families(function.expression):
            return ArgumentPrincipleResult(
                status=ResultStatus.UNSUPPORTED,
                branch=function.branch,
                branch_cuts=cuts,
                contour_obligations=obligations,
                diagnostics=[
                    "argument principle is restricted to expressions established as meromorphic; "
                    "branch-sensitive expressions are unsupported"],
            )
        reduced = sp.cancel(function.expression)
        numerator, denominator = sp.fraction(reduced)
        zero_roots = self._complete_polynomial_roots(numerator, function.variable)
        pole_roots = self._complete_polynomial_roots(denominator, function.variable)
        if zero_roots is None or pole_roots is None:
            return ArgumentPrincipleResult(
                status=ResultStatus.UNKNOWN,
                branch=function.branch,
                branch_cuts=cuts,
                contour_obligations=obligations,
                assumptions=list(contour.assumptions),
                diagnostics=[
                    "SymPy did not establish complete finite polynomial zero and pole sets"],
            )
        zero_count = 0
        pole_count = 0
        windings: dict[str, int] = {}
        enclosed_zeros: list[Any] = []
        enclosed_poles: list[Singularity] = []
        for point, multiplicity in [*zero_roots.items(), *pole_roots.items()]:
            winding = self.winding_number(contour, point)
            if winding.status != ResultStatus.VERIFIED or winding.winding_number is None:
                return ArgumentPrincipleResult(
                    status=winding.status,
                    branch=function.branch,
                    branch_cuts=cuts,
                    contour_obligations=obligations,
                    zero_multiplicities={
                        sp.sstr(root): order for root, order in zero_roots.items()},
                    pole_multiplicities={
                        sp.sstr(root): order for root, order in pole_roots.items()},
                    winding_numbers=windings,
                    enclosed_zeros=enclosed_zeros,
                    enclosed_poles=enclosed_poles,
                    diagnostics=winding.diagnostics,
                )
            key = sp.sstr(point)
            windings[key] = winding.winding_number
            if point in zero_roots:
                zero_count += multiplicity * winding.winding_number
                if winding.winding_number:
                    enclosed_zeros.append(point)
            else:
                pole_count += multiplicity * winding.winding_number
                if winding.winding_number:
                    enclosed_poles.append(Singularity(
                        point=point, kind="pole", order=multiplicity))
        difference = zero_count - pole_count
        integral = sp.simplify(2 * sp.pi * sp.I * difference)
        combined_trust = min(
            (_cap_symbolic(function.input_trust), contour.input_trust),
            key=TRUST_RANK.__getitem__,
        )
        evidence = _symbolic_evidence(
            "cancelled_rational_roots_with_exact_polygon_winding",
            f"zero count minus pole count is {difference}",
            function.input_trust,
        )
        evidence.computation.append(ComputationEvidence(
            engine="caller", method="contour_input_ancestry",
            arithmetic="inherited", trust=contour.input_trust,
        ))
        evidence.justified_trust = combined_trust
        return ArgumentPrincipleResult(
            status=ResultStatus.VERIFIED,
            value=sp.Integer(difference),
            zero_count=zero_count,
            pole_count=pole_count,
            zero_minus_pole=difference,
            logarithmic_derivative_integral=integral,
            zero_multiplicities={
                sp.sstr(root): order for root, order in zero_roots.items()},
            pole_multiplicities={
                sp.sstr(root): order for root, order in pole_roots.items()},
            winding_numbers=windings,
            enclosed_zeros=enclosed_zeros,
            enclosed_poles=enclosed_poles,
            contour_obligations=obligations,
            branch_cuts=cuts,
            trust=combined_trust,
            branch=function.branch,
            assumptions=list(contour.assumptions)
            + (list(function.domain.assumptions) if function.domain else []),
            evidence=evidence,
        )

    def contour_integral(
        self,
        function: ComplexFunction,
        contour: Contour,
        singularities: list[Singularity] | None = None,
        *,
        singularities_accounted_for: bool = False,
        accounted_for: bool | None = None,
    ) -> ContourIntegralResult:
        branch_cuts = self._active_branch_cuts(function)
        if accounted_for is not None:
            singularities_accounted_for = accounted_for
        if not singularities_accounted_for:
            return ContourIntegralResult(
                status=ResultStatus.UNKNOWN,
                branch=function.branch,
                branch_cuts=branch_cuts,
                orientation=contour.orientation,
                assumptions=list(contour.assumptions),
                diagnostics=["complete singularity accounting was not asserted"],
            )
        if singularities is None:
            return ContourIntegralResult(
                status=ResultStatus.ERROR,
                branch=function.branch,
                branch_cuts=branch_cuts,
                orientation=contour.orientation,
                diagnostics=["singularities must be supplied, including an explicit empty list"],
            )
        discovered = self._discover_singularities(function)
        supplied_points = {item.point for item in singularities}
        accounting_verified = discovered is not None and supplied_points == discovered
        retained_enclosed: list[Singularity] = []
        retained_windings: dict[str, int] = {}
        retained_orientation = contour.orientation
        for supplied in singularities:
            winding = self.winding_number(contour, supplied.point)
            if winding.status == ResultStatus.VERIFIED and winding.winding_number is not None:
                retained_windings[sp.sstr(supplied.point)] = winding.winding_number
                retained_orientation = retained_orientation or winding.orientation
                if winding.winding_number:
                    retained_enclosed.append(supplied)
        if discovered is not None and supplied_points != discovered:
            return ContourIntegralResult(
                status=ResultStatus.UNKNOWN,
                branch=function.branch,
                branch_cuts=branch_cuts,
                orientation=retained_orientation,
                enclosed_singularities=retained_enclosed,
                winding_numbers=retained_windings,
                diagnostics=[
                    "supplied singularities do not match independently discovered singularities",
                    f"supplied={sorted(map(sp.sstr, supplied_points))}",
                    f"discovered={sorted(map(sp.sstr, discovered))}",
                ],
            )
        cuts_clear, cut_diagnostic = self._cuts_clear_contour(function, contour)
        if cuts_clear is not True:
            return ContourIntegralResult(
                status=ResultStatus.UNKNOWN if cuts_clear is None else ResultStatus.ERROR,
                branch=function.branch,
                branch_cuts=branch_cuts,
                orientation=retained_orientation,
                enclosed_singularities=retained_enclosed,
                winding_numbers=retained_windings,
                diagnostics=[cut_diagnostic],
            )
        total = sp.S.Zero
        enclosed: list[Singularity] = []
        windings: dict[str, int] = {}
        residues: dict[str, Any] = {}
        evidence = EvidenceBundle()
        for supplied in singularities:
            classified = self.classify_singularity(function, supplied.point)
            actual = classified.singularity
            if (classified.status != ResultStatus.VERIFIED or actual is None
                    or actual.kind != supplied.kind or actual.order != supplied.order):
                return ContourIntegralResult(
                    status=ResultStatus.UNKNOWN,
                    branch=function.branch,
                    branch_cuts=branch_cuts,
                    orientation=contour.orientation,
                    enclosed_singularities=retained_enclosed,
                    winding_numbers=retained_windings,
                    residues=residues,
                    diagnostics=[f"supplied singularity at {supplied.point} was not independently verified"],
                )
            winding = self.winding_number(contour, supplied.point)
            if winding.status != ResultStatus.VERIFIED or winding.winding_number is None:
                return ContourIntegralResult(
                    status=winding.status,
                    branch=function.branch,
                    branch_cuts=branch_cuts,
                    orientation=winding.orientation,
                    enclosed_singularities=retained_enclosed,
                    winding_numbers=retained_windings,
                    residues=residues,
                    diagnostics=winding.diagnostics,
                )
            key = sp.sstr(supplied.point)
            windings[key] = winding.winding_number
            if winding.winding_number == 0:
                continue
            residue = self.residue(function, supplied.point, supplied)
            if residue.status != ResultStatus.VERIFIED:
                return ContourIntegralResult(
                    status=residue.status,
                    branch=function.branch,
                    branch_cuts=branch_cuts,
                    orientation=winding.orientation,
                    enclosed_singularities=retained_enclosed,
                    winding_numbers=retained_windings,
                    residues=residues,
                    diagnostics=residue.diagnostics,
                )
            enclosed.append(supplied)
            residues[key] = residue.value
            total += winding.winding_number * residue.value
            evidence.computation.extend(residue.evidence.computation)
            evidence.proof.extend(residue.evidence.proof)
        integral = sp.simplify(2 * sp.pi * sp.I * total)
        evidence.computation.append(ComputationEvidence(
            engine="caller",
            method="input_ancestry",
            arithmetic="inherited",
            deterministic=True,
            trust=function.input_trust,
        ))
        evidence.computation.append(ComputationEvidence(
            engine="caller",
            method="contour_input_ancestry",
            arithmetic="inherited",
            deterministic=True,
            trust=contour.input_trust,
        ))
        evidence.computation.append(ComputationEvidence(
            engine="sympy",
            method="residue_theorem_with_exact_polygon_winding",
            arithmetic="symbolic",
            deterministic=True,
            trust=_cap_symbolic(function.input_trust),
            metadata={
                "singularity_accounting": (
                    "independently-verified" if accounting_verified
                    else "caller-asserted-complete"
                )
            },
        ))
        return ContourIntegralResult(
            status=(
                ResultStatus.VERIFIED if accounting_verified
                else ResultStatus.CANDIDATE
            ),
            value=integral,
            integral=integral,
            trust=evidence.conservative_trust(),
            assumptions=list(contour.assumptions)
            + (list(function.domain.assumptions) if function.domain else []),
            branch=function.branch,
            branch_cuts=branch_cuts,
            orientation=contour.orientation
            or ("ccw" if any(value > 0 for value in windings.values()) else
                "cw" if any(value < 0 for value in windings.values()) else None),
            evidence=evidence,
            enclosed_singularities=enclosed,
            winding_numbers=windings,
            residues=residues,
            diagnostics=(
                ["singularity accounting independently matched the supplied set"]
                if accounting_verified else
                ["candidate result depends on unverified caller assertion that "
                 "supplied singularities are complete"]
            ),
        )
