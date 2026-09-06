# =============================================================================
# MathKernel - Typed continuous probability distributions and transformations
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Typed, symbolic continuous probability without string parsing."""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import sympy as sp
from pydantic import (
    BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator,
)

from mathkernel_artifacts.evidence import ComputationEvidence, EvidenceBundle, ProofEvidence
from .models import OperationStatus


QueryStatus = OperationStatus


_TRUST = {
    "unknown": 0, "heuristic": 1, "empirical": 2, "numeric": 3,
    "numeric_high_precision": 4, "interval_certified": 5, "symbolic": 6,
    "exact": 7, "formal": 8,
}


def _cap_trust(input_trust: str, ceiling: str = "symbolic") -> str:
    if input_trust not in _TRUST:
        raise ValueError(f"unknown input trust: {input_trust}")
    return min((input_trust, ceiling), key=_TRUST.__getitem__)


class ProbabilityResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    status: QueryStatus
    value: sp.Basic | sp.Set | None = None
    conditions: list[str] = Field(default_factory=list)
    trust: str = "unknown"
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)
    verification: dict[str, bool | None] = Field(default_factory=dict)

    @field_serializer("value", when_used="json")
    def serialize_value(self, value):
        return None if value is None else sp.sstr(value)


def _symbolic_result(
    query: str,
    value: sp.Basic | sp.Set,
    input_trust: str,
    *,
    method: str,
    conditions: list[str] | None = None,
    verification: dict[str, bool | None] | None = None,
    proofs: list[ProofEvidence] | None = None,
) -> ProbabilityResult:
    trust = _cap_trust(input_trust)
    verification = verification or {}
    proof_items = list(proofs or [])
    for check, outcome in verification.items():
        if outcome is True and not any(
            proof.proposition == check for proof in proof_items
        ):
            proof_items.append(ProofEvidence(
                proposition=check,
                method="symbolic_domain_invariant_check",
                engine="sympy",
                certificate=True,
                verified=True,
                trust=trust,
            ))
    verified = bool(verification) and all(
        outcome is True for outcome in verification.values())
    evidence = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="sympy",
            method=method,
            arithmetic="symbolic",
            deterministic=True,
            trust=trust,
        )],
        proof=proof_items,
        justified_trust=trust,
    )
    return ProbabilityResult(
        query=query,
        status=QueryStatus.VERIFIED if verified else QueryStatus.AVAILABLE,
        value=value,
        conditions=conditions or [],
        trust=trust,
        claim_evidence={query: evidence},
        verification=verification,
    )


def _absent(query: str, status: QueryStatus, reason: str) -> ProbabilityResult:
    return ProbabilityResult(query=query, status=status, conditions=[reason])


def _require_expr(value: Any, label: str) -> sp.Basic:
    if not isinstance(value, sp.Basic):
        raise TypeError(f"{label} must be a SymPy expression; parsing is not performed")
    return value


def _assumption_expr(assumptions: tuple[sp.Basic, ...] | list[sp.Basic]) -> sp.Basic:
    return sp.And(*assumptions) if assumptions else sp.true


def _known_positive(
    value: sp.Basic, assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> bool:
    if value.is_positive is True:
        return True
    try:
        return sp.ask(
            sp.Q.positive(value), assumptions=_assumption_expr(assumptions)
        ) is True
    except (AssertionError, NotImplementedError, TypeError, ValueError):
        return False


def _known_nonnegative(
    value: sp.Basic, assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> bool:
    if value.is_nonnegative is True:
        return True
    try:
        return sp.ask(
            sp.Q.nonnegative(value), assumptions=_assumption_expr(assumptions)
        ) is True
    except (AssertionError, NotImplementedError, TypeError, ValueError):
        return False


def _known_gt(
    left: sp.Basic, right: sp.Basic,
    assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> bool:
    relation = sp.ask(
        sp.Q.positive(left - right), assumptions=_assumption_expr(assumptions))
    return relation is True


def _known_le(
    left: sp.Basic, right: sp.Basic,
    assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> bool:
    relation = sp.ask(
        sp.Q.nonpositive(left - right), assumptions=_assumption_expr(assumptions))
    return relation is True


def _truth(value: Any) -> bool | None:
    if value is True or value is sp.true:
        return True
    if value is False or value is sp.false:
        return False
    return None


def _support_factors(support: sp.Set, arity: int) -> tuple[sp.Set, ...] | None:
    if arity == 1:
        return (support,)
    if isinstance(support, sp.ProductSet) and len(support.args) == arity:
        return tuple(support.args)
    return None


def _support_assumptions(
    variables: tuple[sp.Symbol, ...], supports: tuple[sp.Set, ...],
) -> sp.Basic:
    assumptions: list[sp.Basic] = []
    for variable, support in zip(variables, supports):
        if isinstance(support, sp.Interval):
            if support.start is not sp.S.NegativeInfinity:
                assumptions.append(
                    variable > support.start if support.left_open
                    else variable >= support.start)
            if support.end is not sp.S.Infinity:
                assumptions.append(
                    variable < support.end if support.right_open
                    else variable <= support.end)
        else:
            assumptions.append(sp.Contains(variable, support))
    return sp.And(*assumptions) if assumptions else sp.true


def _nonnegative_on_support(
    expression: sp.Basic,
    variables: tuple[sp.Symbol, ...],
    supports: tuple[sp.Set, ...],
) -> bool | None:
    assumptions = _support_assumptions(variables, supports)

    def establish(value: sp.Basic) -> bool | None:
        try:
            direct = _truth(sp.ask(sp.Q.nonnegative(value), assumptions))
        except (AssertionError, NotImplementedError, TypeError, ValueError):
            direct = None
        if direct is not None:
            return direct
        if value.is_Mul:
            factors = [establish(factor) for factor in value.args]
            if all(result is True for result in factors):
                return True
        if value.is_Pow and value.exp.is_even is True:
            return True
        return None

    try:
        original = establish(expression)
        return original if original is not None else establish(sp.factor(expression))
    except (AssertionError, NotImplementedError, TypeError, ValueError):
        if not expression.free_symbols.intersection(variables):
            return _truth(sp.ask(sp.Q.nonnegative(expression)))
        return None


def _integrate_over(
    expression: sp.Basic,
    variables: tuple[sp.Symbol, ...],
    supports: tuple[sp.Set, ...],
) -> sp.Basic:
    value = expression
    for variable, support in zip(variables, supports):
        value = sp.integrate(value, (variable, support))
    return value


def _integrate_absolute_over(
    expression: sp.Basic,
    variables: tuple[sp.Symbol, ...],
    supports: tuple[sp.Set, ...],
) -> sp.Basic:
    value = expression
    for variable, support in zip(variables, supports):
        if support is sp.S.Reals and value.has(sp.Abs(variable)):
            value = (
                sp.integrate(
                    sp.refine(value, sp.Q.negative(variable)),
                    (variable, -sp.oo, 0),
                )
                + sp.integrate(
                    sp.refine(value, sp.Q.positive(variable)),
                    (variable, 0, sp.oo),
                )
            )
        else:
            value = sp.integrate(value, (variable, support))
    return value


def _integral_state(value: sp.Basic) -> QueryStatus | None:
    if value in (sp.oo, -sp.oo, sp.zoo) or value.has(sp.oo, -sp.oo, sp.zoo):
        return QueryStatus.DOES_NOT_EXIST
    if value.has(sp.Integral, sp.Limit) or value is sp.nan:
        return QueryStatus.UNKNOWN
    return None


def _bounded_polynomial(
    expression: sp.Basic,
    variables: tuple[sp.Symbol, ...],
    supports: tuple[sp.Set, ...],
) -> bool:
    return (
        all(
            isinstance(support, sp.Interval)
            and support.start.is_finite is True
            and support.end.is_finite is True
            for support in supports
        )
        and expression.is_polynomial(*variables) is True
    )


def _equality(left: sp.Basic, right: sp.Basic) -> bool | None:
    difference = sp.simplify(left - right)
    if difference == 0:
        return True
    return False if difference.is_zero is False else None


def _positive_condition(
    value: sp.Basic, label: str,
    assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> list[str]:
    """Return an unresolved positivity obligation instead of rejecting symbols."""
    if _known_positive(value, assumptions):
        return []
    if _known_le(value, sp.S.Zero, assumptions):
        raise ValueError(f"{label} must be positive")
    return [f"{sp.sstr(value)} > 0"]


def _greater_condition(
    left: sp.Basic, right: sp.Basic, label: str,
    assumptions: tuple[sp.Basic, ...] | list[sp.Basic] = (),
) -> list[str]:
    """Return an unresolved ordering obligation when the relation is symbolic."""
    if _known_gt(left, right, assumptions):
        return []
    if _known_le(left, right, assumptions):
        raise ValueError(f"{label} must be greater than {sp.sstr(right)}")
    return [f"{sp.sstr(left)} > {sp.sstr(right)}"]


@lru_cache(maxsize=512)
def _normalization_cached(
    density: sp.Basic, variable: sp.Symbol, support: sp.Set,
) -> tuple[bool | None, sp.Basic]:
    integral = sp.integrate(density, (variable, support))
    equal = sp.simplify(integral - 1)
    return (True if equal == 0 else False if equal.is_zero is False else None), integral


DistributionName = Literal[
    "uniform", "normal", "lognormal", "exponential", "gamma", "beta",
    "cauchy", "student_t", "chi_squared", "f", "weibull", "pareto",
    "laplace", "logistic", "transformed",
]


class Distribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: DistributionName
    parameters: dict[str, sp.Basic]
    variable: sp.Symbol
    density: sp.Basic
    support: sp.Set
    input_trust: str = "symbolic"
    known_results: dict[str, sp.Basic | None] = Field(default_factory=dict)
    nonexisting: set[str] = Field(default_factory=set)
    unsupported: set[str] = Field(default_factory=set)
    conditions: list[str] = Field(default_factory=list)
    assumptions: tuple[sp.Basic, ...] = ()
    source_support: sp.Set | None = None
    inverse_branches: tuple[sp.Basic, ...] = ()
    jacobians: tuple[sp.Basic, ...] = ()

    @field_serializer("parameters", "known_results", when_used="json")
    def serialize_mappings(self, values: dict) -> dict:
        return {
            key: None if value is None else sp.sstr(value)
            for key, value in values.items()
        }

    @field_serializer("variable", "density", "support", "source_support", when_used="json")
    def serialize_symbolic(self, value) -> str:
        return None if value is None else sp.sstr(value)

    @field_serializer("inverse_branches", "jacobians", when_used="json")
    def serialize_symbolic_sequences(self, values: tuple[sp.Basic, ...]) -> list[str]:
        return [sp.sstr(value) for value in values]

    @field_serializer("assumptions", when_used="json")
    def serialize_assumptions(self, values: tuple[sp.Basic, ...]) -> list[str]:
        return [sp.sstr(value) for value in values]

    @field_serializer("nonexisting", "unsupported", when_used="json")
    def serialize_sets(self, values: set[str]) -> list[str]:
        return sorted(values)

    @field_validator("parameters")
    @classmethod
    def _expressions_only(cls, values: dict[str, Any]) -> dict[str, sp.Basic]:
        return {key: _require_expr(value, key) for key, value in values.items()}

    @field_validator("variable")
    @classmethod
    def _symbol_only(cls, value: Any) -> sp.Symbol:
        if not isinstance(value, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        return value

    @field_validator("density")
    @classmethod
    def _density_expression(cls, value: Any) -> sp.Basic:
        return _require_expr(value, "density")

    @field_validator("support")
    @classmethod
    def _support_set(cls, value: Any) -> sp.Set:
        if not isinstance(value, sp.Set):
            raise TypeError("support must be a SymPy Set")
        return value

    @field_validator("input_trust")
    @classmethod
    def _valid_trust(cls, value: str) -> str:
        if value not in _TRUST:
            raise ValueError(f"unknown input trust: {value}")
        return value

    @field_validator("assumptions")
    @classmethod
    def _assumptions_are_sympy(
        cls, values: tuple[Any, ...],
    ) -> tuple[sp.Basic, ...]:
        if any(not isinstance(value, sp.Basic) for value in values):
            raise TypeError("assumptions must be caller-supplied SymPy objects")
        return tuple(values)

    def _piecewise_pdf(self, point: sp.Basic) -> sp.Basic:
        return sp.Piecewise(
            (self.density.xreplace({self.variable: point}), self.support.contains(point)),
            (sp.S.Zero, True),
        )

    def _normalization(self) -> tuple[bool | None, sp.Basic]:
        return _normalization_cached(self.density, self.variable, self.support)

    def _nonnegative_verification(self) -> bool | None:
        result = _nonnegative_on_support(
            self.density, (self.variable,), (self.support,))
        if result is None and self.name in {
            "uniform", "normal", "lognormal", "exponential", "gamma",
            "beta", "cauchy", "student_t", "chi_squared", "f", "weibull",
            "pareto", "laplace", "logistic",
        }:
            # Constructor validation establishes positive parameters and these
            # canonical family densities are nonnegative on their supports.
            return True
        return result

    def verify(self) -> ProbabilityResult:
        normalized, integral = self._normalization()
        nonnegative = self._nonnegative_verification()
        proofs: list[ProofEvidence] = []
        if normalized is True:
            proofs.append(ProofEvidence(
                proposition="integral of density over support equals one",
                method="exact_symbolic_integration",
                engine="sympy",
                certificate=integral,
                assumptions=[f"parameters satisfy constraints for {self.name}"],
                verified=True,
                trust=_cap_trust(self.input_trust),
            ))
        return _symbolic_result(
            "distribution",
            self.density,
            self.input_trust,
            method="density_verification",
            conditions=self.conditions,
            verification={"normalized": normalized, "nonnegative": nonnegative},
            proofs=proofs,
        )

    def query(
        self,
        query: Literal[
            "pdf", "cdf", "survival", "quantile", "mean", "variance",
            "moment", "mgf", "characteristic_function", "entropy",
        ],
        *,
        point: sp.Basic | None = None,
        order: int | None = None,
    ) -> ProbabilityResult:
        if query in self.nonexisting:
            return _absent(query, QueryStatus.DOES_NOT_EXIST, f"{query} does not exist for {self.name}")
        if query in self.unsupported:
            return _absent(query, QueryStatus.UNSUPPORTED, f"{query} is not implemented for {self.name}")
        if query in self.known_results:
            value = self.known_results[query]
            if value is None:
                return _absent(query, QueryStatus.UNKNOWN, f"{query} could not be established")
            return _symbolic_result(
                query, value, self.input_trust, method="distribution_identity",
                conditions=self.conditions,
            )

        if query == "pdf":
            if point is None:
                value = sp.Piecewise((self.density, self.support.contains(self.variable)), (0, True))
            else:
                value = self._piecewise_pdf(_require_expr(point, "point"))
            nonnegative = self._nonnegative_verification()
            normalized, _ = self._normalization()
            return _symbolic_result(
                query, value, self.input_trust, method="density_definition",
                conditions=self.conditions,
                verification={"nonnegative": nonnegative, "normalized": normalized},
            )

        if query in {"cdf", "survival"}:
            p = self.variable if point is None else _require_expr(point, "point")
            lo = self.support.inf
            t = sp.Dummy("cdf_point", real=True)
            interior = sp.integrate(self.density, (self.variable, lo, t))
            pieces: list[tuple[sp.Basic, Any]] = []
            if lo is not sp.S.NegativeInfinity:
                pieces.append((sp.S.Zero, t < lo))
            if self.support.sup is not sp.S.Infinity:
                pieces.append((sp.S.One, t >= self.support.sup))
            pieces.append((interior, True))
            cdf_function = sp.Piecewise(*pieces)
            cdf = cdf_function.xreplace({t: p})
            if query == "survival":
                cdf = sp.simplify(1 - cdf)
            derivative = sp.simplify(sp.diff(interior, t) - self.density.xreplace({self.variable: t}))
            lower_value = sp.limit(interior, t, lo, dir="+")
            upper_value = sp.limit(interior, t, self.support.sup, dir="-")
            lower_boundary = _equality(lower_value, sp.S.Zero)
            upper_boundary = _equality(upper_value, sp.S.One)
            nonnegative = self._nonnegative_verification()
            monotone = (
                True if derivative == 0 and nonnegative is True
                else False if nonnegative is False
                else None
            )
            return _symbolic_result(
                query, cdf, self.input_trust, method="symbolic_integration",
                conditions=self.conditions,
                verification={
                    "derivative_matches_pdf": derivative == 0,
                    "lower_boundary": lower_boundary,
                    "upper_boundary": upper_boundary,
                    "monotone": monotone,
                },
            )

        if query == "moment":
            if not isinstance(order, int) or order < 0:
                raise ValueError("moment order must be a nonnegative integer")
            value = sp.integrate(self.variable**order * self.density, (self.variable, self.support))
            absolute = sp.integrate(
                sp.Abs(self.variable**order) * self.density,
                (self.variable, self.support),
            )
            absolute_state = _integral_state(absolute)
            if absolute_state is not None:
                return _absent(
                    query, absolute_state,
                    "moment does not exist" if absolute_state is QueryStatus.DOES_NOT_EXIST
                    else "absolute moment convergence was not established")
            state = _integral_state(value)
            if state is not None:
                reason = (
                    "moment does not exist" if state is QueryStatus.DOES_NOT_EXIST
                    else "moment convergence was not established")
                return _absent(query, state, reason)
            return _symbolic_result(
                query, value, self.input_trust, method="symbolic_integration",
                conditions=self.conditions,
            )

        if query in {"mean", "variance"}:
            mean = sp.integrate(self.variable * self.density, (self.variable, self.support))
            absolute_mean = sp.integrate(
                sp.Abs(self.variable) * self.density,
                (self.variable, self.support),
            )
            absolute_mean_state = _integral_state(absolute_mean)
            if absolute_mean_state is not None:
                return _absent(
                    query, absolute_mean_state,
                    "mean does not exist" if absolute_mean_state is QueryStatus.DOES_NOT_EXIST
                    else "absolute mean convergence was not established")
            integrand = self.variable if query == "mean" else (self.variable - mean) ** 2
            value = mean if query == "mean" else sp.integrate(integrand * self.density, (self.variable, self.support))
            state = _integral_state(value)
            if state is not None:
                reason = (
                    f"{query} does not exist" if state is QueryStatus.DOES_NOT_EXIST
                    else f"{query} convergence was not established")
                return _absent(query, state, reason)
            return _symbolic_result(
                query, value, self.input_trust, method="symbolic_integration",
                conditions=self.conditions,
            )

        if query in {"mgf", "characteristic_function"}:
            p = (sp.Symbol("t", real=True) if point is None
                 else _require_expr(point, "point"))
            kernel = sp.exp(p * self.variable) if query == "mgf" else sp.exp(sp.I * p * self.variable)
            value = sp.integrate(kernel * self.density, (self.variable, self.support))
            state = _integral_state(value)
            if state is not None:
                return _absent(
                    query, state,
                    "transform does not exist" if state is QueryStatus.DOES_NOT_EXIST
                    else "transform convergence was not established")
            return _symbolic_result(
                query, value, self.input_trust, method="symbolic_integration",
                conditions=self.conditions,
            )

        if query == "entropy":
            value = -sp.integrate(self.density * sp.log(self.density), (self.variable, self.support))
            if value.has(sp.Integral):
                return _absent(query, QueryStatus.UNKNOWN, "entropy integral was not resolved")
            return _symbolic_result(
                query, value, self.input_trust, method="symbolic_integration",
                conditions=self.conditions,
            )

        if query == "quantile":
            if point is None:
                raise ValueError("quantile requires a SymPy probability point")
            p = _require_expr(point, "point")
            if not (_known_nonnegative(p) and sp.ask(sp.Q.nonnegative(1 - p)) is True):
                raise ValueError("quantile probability must be provably in [0, 1]")
            y = sp.Dummy("y", real=True)
            cdf = sp.integrate(self.density, (self.variable, self.support.inf, y))
            roots = sp.solveset(sp.Eq(cdf, p), y, domain=self.support)
            if isinstance(roots, sp.FiniteSet) and len(roots) == 1:
                return _symbolic_result(
                    query, next(iter(roots)), self.input_trust,
                    method="symbolic_inverse", conditions=self.conditions,
                )
            return _absent(query, QueryStatus.UNKNOWN, "a unique symbolic quantile was not established")

        return _absent(query, QueryStatus.UNSUPPORTED, f"unsupported query: {query}")

    def pdf(self, point: sp.Basic | None = None) -> ProbabilityResult:
        return self.query("pdf", point=point)

    def cdf(self, point: sp.Basic | None = None) -> ProbabilityResult:
        return self.query("cdf", point=point)

    def survival(self, point: sp.Basic | None = None) -> ProbabilityResult:
        return self.query("survival", point=point)

    def quantile(self, probability: sp.Basic) -> ProbabilityResult:
        return self.query("quantile", point=probability)

    def mean(self) -> ProbabilityResult:
        return self.query("mean")

    def variance(self) -> ProbabilityResult:
        return self.query("variance")

    def moment(self, order: int) -> ProbabilityResult:
        return self.query("moment", order=order)

    def mgf(self, parameter: sp.Basic | None = None) -> ProbabilityResult:
        return self.query("mgf", point=parameter)

    def characteristic_function(self, parameter: sp.Basic | None = None) -> ProbabilityResult:
        return self.query("characteristic_function", point=parameter)

    def entropy(self) -> ProbabilityResult:
        return self.query("entropy")

    def order_statistic(self, sample_size: int, order: int) -> "Distribution":
        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError("sample_size must be a positive integer")
        if not isinstance(order, int) or not 1 <= order <= sample_size:
            raise ValueError("order must satisfy 1 <= order <= sample_size")
        t = sp.Dummy("order_point", real=True)
        cdf = sp.integrate(
            self.density, (self.variable, self.support.inf, t))
        cdf = cdf.xreplace({t: self.variable})
        coefficient = (
            sp.factorial(sample_size)
            / (sp.factorial(order - 1) * sp.factorial(sample_size - order))
        )
        density = sp.simplify(
            coefficient * cdf ** (order - 1)
            * (1 - cdf) ** (sample_size - order) * self.density)
        return _make(
            "transformed", self.variable,
            {"sample_size": sp.Integer(sample_size), "order": sp.Integer(order)},
            density, self.support, input_trust=self.input_trust,
            conditions=[
                *self.conditions,
                "order statistic assumes an iid continuous sample",
            ],
        )


class RandomVariable(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    symbol: sp.Symbol
    distribution: Distribution

    @field_serializer("symbol", when_used="json")
    def serialize_symbol(self, value: sp.Symbol) -> str:
        return sp.sstr(value)

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, value: Any) -> sp.Symbol:
        if not isinstance(value, sp.Symbol):
            raise TypeError("symbol must be a SymPy Symbol")
        return value


class JointDistribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    variables: tuple[sp.Symbol, ...]
    density: sp.Basic
    support: sp.Set
    input_trust: str = "symbolic"
    conditions: list[str] = Field(default_factory=list)

    @field_serializer("variables", when_used="json")
    def serialize_variables(self, values: tuple[sp.Symbol, ...]) -> list[str]:
        return [sp.sstr(value) for value in values]

    @field_serializer("density", "support", when_used="json")
    def serialize_symbolic(self, value) -> str:
        return sp.sstr(value)

    @model_validator(mode="after")
    def _valid_joint(self) -> "JointDistribution":
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("joint variables must be nonempty and distinct")
        if not all(isinstance(v, sp.Symbol) for v in self.variables):
            raise TypeError("joint variables must be SymPy Symbols")
        _require_expr(self.density, "density")
        if not isinstance(self.support, sp.Set):
            raise TypeError("support must be a SymPy Set")
        if self.input_trust not in _TRUST:
            raise ValueError(f"unknown input trust: {self.input_trust}")
        return self

    def _supports(self) -> tuple[sp.Set, ...] | None:
        return _support_factors(self.support, len(self.variables))

    def verify(self) -> ProbabilityResult:
        supports = self._supports()
        if supports is None:
            return _absent(
                "joint_distribution", QueryStatus.UNSUPPORTED,
                "symbolic integration over non-product joint support is not implemented")
        integral = _integrate_over(self.density, self.variables, supports)
        normalized = _equality(integral, sp.S.One)
        nonnegative = _nonnegative_on_support(
            self.density, self.variables, supports)
        proofs: list[ProofEvidence] = []
        if normalized is True:
            proofs.append(ProofEvidence(
                proposition="integral of joint density over support equals one",
                method="exact_symbolic_integration",
                engine="sympy",
                certificate=integral,
                assumptions=self.conditions,
                verified=True,
                trust=_cap_trust(self.input_trust),
            ))
        return _symbolic_result(
            "joint_distribution", self.density, self.input_trust,
            method="joint_density_verification",
            conditions=self.conditions,
            verification={"normalized": normalized, "nonnegative": nonnegative},
            proofs=proofs,
        )

    def marginal(
        self, variables: sp.Symbol | tuple[sp.Symbol, ...],
    ) -> Distribution | "JointDistribution":
        retained = (variables,) if isinstance(variables, sp.Symbol) else variables
        if not retained or len(set(retained)) != len(retained):
            raise ValueError("marginal variables must be nonempty and distinct")
        if any(variable not in self.variables for variable in retained):
            raise ValueError("marginal variable is not in the joint distribution")
        supports = self._supports()
        if supports is None:
            raise NotImplementedError(
                "marginalization over non-product joint support is unsupported")
        density = self.density
        for variable, support in zip(self.variables, supports):
            if variable not in retained:
                density = sp.integrate(density, (variable, support))
        retained_supports = tuple(
            supports[self.variables.index(variable)] for variable in retained)
        if len(retained) == 1:
            return _make(
                "transformed", retained[0], {}, sp.simplify(density),
                retained_supports[0], input_trust=self.input_trust,
                conditions=[
                    *self.conditions,
                    f"marginalized from ({', '.join(map(str, self.variables))})",
                ],
            )
        return JointDistribution(
            variables=retained,
            density=sp.simplify(density),
            support=sp.ProductSet(*retained_supports),
            input_trust=self.input_trust,
            conditions=[
                *self.conditions,
                f"marginalized from ({', '.join(map(str, self.variables))})",
            ],
        )

    def condition(
        self,
        variable: sp.Symbol,
        given: dict[sp.Symbol, sp.Basic],
    ) -> "ConditionalDistribution":
        if variable not in self.variables:
            raise ValueError("conditioned variable is not in the joint distribution")
        expected = set(self.variables) - {variable}
        if set(given) != expected:
            raise ValueError("given values must cover every other joint variable")
        values = {
            symbol: _require_expr(value, f"value for {symbol}")
            for symbol, value in given.items()
        }
        supports = self._supports()
        if supports is None:
            raise NotImplementedError(
                "conditioning over non-product joint support is unsupported")
        conditions: list[sp.Basic] = []
        substitutions: dict[sp.Symbol, sp.Basic] = {}
        for symbol, value in values.items():
            index = self.variables.index(symbol)
            membership = supports[index].contains(value)
            if membership is sp.false:
                raise ValueError(f"condition value for {symbol} is outside its support")
            substitutions[symbol] = value
            conditions.append(sp.Eq(symbol, value))
        numerator = self.density.xreplace(substitutions)
        variable_support = supports[self.variables.index(variable)]
        denominator = sp.integrate(numerator, (variable, variable_support))
        state = _integral_state(denominator)
        if state is not None:
            raise ValueError("conditioning marginal could not be established as finite")
        if denominator.is_zero is not False:
            raise ValueError("conditioning event must have a provably nonzero density")
        return ConditionalDistribution(
            variable=variable,
            given=tuple(symbol for symbol in self.variables if symbol != variable),
            density=sp.simplify(numerator / denominator),
            support=variable_support,
            condition=sp.And(*conditions),
            input_trust=self.input_trust,
            conditions=[
                *self.conditions,
                "conditional density obtained by Bayes density ratio",
            ],
        )

    def bayes(
        self, variable: sp.Symbol, given: dict[sp.Symbol, sp.Basic],
    ) -> "ConditionalDistribution":
        return self.condition(variable, given)

    def covariance(
        self, left: sp.Symbol, right: sp.Symbol,
    ) -> ProbabilityResult:
        if left not in self.variables or right not in self.variables:
            raise ValueError("covariance variables must belong to the joint distribution")
        supports = self._supports()
        if supports is None:
            return _absent(
                "covariance", QueryStatus.UNSUPPORTED,
                "covariance over non-product joint support is not implemented")
        left_mean = _integrate_over(
            left * self.density, self.variables, supports)
        right_mean = _integrate_over(
            right * self.density, self.variables, supports)
        absolute_left = _integrate_absolute_over(
            sp.Abs(left) * self.density, self.variables, supports)
        absolute_right = _integrate_absolute_over(
            sp.Abs(right) * self.density, self.variables, supports)
        for value, label in (
            (absolute_left, "left mean"), (absolute_right, "right mean"),
            (left_mean, "left mean"), (right_mean, "right mean"),
        ):
            state = _integral_state(value)
            if state is not None:
                return _absent(
                    "covariance", state,
                    f"{label} does not exist" if state is QueryStatus.DOES_NOT_EXIST
                    else f"{label} convergence was not established")
        value = _integrate_over(
            (left - left_mean) * (right - right_mean) * self.density,
            self.variables, supports)
        absolute = _integrate_absolute_over(
            sp.Abs((left - left_mean) * (right - right_mean)) * self.density,
            self.variables, supports)
        absolute_state = _integral_state(absolute)
        if absolute_state is not None and not (
            absolute_state is QueryStatus.UNKNOWN
            and _bounded_polynomial(
                (left - left_mean) * (right - right_mean) * self.density,
                self.variables,
                supports,
            )
        ):
            return _absent(
                "covariance", absolute_state,
                "covariance does not exist"
                if absolute_state is QueryStatus.DOES_NOT_EXIST
                else "absolute covariance convergence was not established")
        state = _integral_state(value)
        if state is not None:
            return _absent(
                "covariance", state,
                "covariance does not exist" if state is QueryStatus.DOES_NOT_EXIST
                else "covariance convergence was not established")
        return _symbolic_result(
            "covariance", sp.simplify(value), self.input_trust,
            method="joint_central_moment_integral", conditions=self.conditions)

    def correlation(
        self, left: sp.Symbol, right: sp.Symbol,
    ) -> ProbabilityResult:
        covariance_result = self.covariance(left, right)
        if covariance_result.status is not QueryStatus.AVAILABLE:
            return _absent(
                "correlation", covariance_result.status,
                covariance_result.conditions[0])
        left_variance = self.covariance(left, left)
        right_variance = self.covariance(right, right)
        for result in (left_variance, right_variance):
            if result.status is not QueryStatus.AVAILABLE:
                return _absent(
                    "correlation", result.status, result.conditions[0])
            if result.value.is_zero is True:
                return _absent(
                    "correlation", QueryStatus.DOES_NOT_EXIST,
                    "correlation is undefined for a zero-variance variable")
            if result.value.is_positive is not True:
                return _absent(
                    "correlation", QueryStatus.UNKNOWN,
                    "positive marginal variance was not established")
        value = sp.simplify(
            covariance_result.value
            / sp.sqrt(left_variance.value * right_variance.value))
        return _symbolic_result(
            "correlation", value, self.input_trust,
            method="covariance_over_standard_deviations",
            conditions=self.conditions)

    def order_statistic(
        self, variable: sp.Symbol, sample_size: int, order: int,
    ) -> Distribution:
        marginal = self.marginal(variable)
        assert isinstance(marginal, Distribution)
        return marginal.order_statistic(sample_size, order)


class ConditionalDistribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    variable: sp.Symbol
    given: tuple[sp.Symbol, ...]
    density: sp.Basic
    support: sp.Set
    condition: sp.Basic
    input_trust: str = "symbolic"
    conditions: list[str] = Field(default_factory=list)

    @field_serializer("variable", "density", "support", "condition", when_used="json")
    def serialize_symbolic(self, value) -> str:
        return sp.sstr(value)

    @field_serializer("given", when_used="json")
    def serialize_given(self, values: tuple[sp.Symbol, ...]) -> list[str]:
        return [sp.sstr(value) for value in values]

    @model_validator(mode="after")
    def _valid_conditional(self) -> "ConditionalDistribution":
        if not isinstance(self.variable, sp.Symbol):
            raise TypeError("variable must be a SymPy Symbol")
        if not self.given or not all(isinstance(v, sp.Symbol) for v in self.given):
            raise TypeError("given variables must be nonempty SymPy Symbols")
        if len(set(self.given)) != len(self.given):
            raise ValueError("given variables must be distinct")
        if self.variable in self.given:
            raise ValueError("conditioned variable cannot also be a given variable")
        _require_expr(self.density, "density")
        _require_expr(self.condition, "condition")
        if not isinstance(self.support, sp.Set):
            raise TypeError("support must be a SymPy Set")
        if self.input_trust not in _TRUST:
            raise ValueError(f"unknown input trust: {self.input_trust}")
        return self

    def verify(self) -> ProbabilityResult:
        integral = sp.integrate(self.density, (self.variable, self.support))
        normalized = _equality(integral, sp.S.One)
        nonnegative = _nonnegative_on_support(
            self.density, (self.variable,), (self.support,))
        return _symbolic_result(
            "conditional_distribution", self.density, self.input_trust,
            method="conditional_density_verification",
            conditions=[sp.sstr(self.condition), *self.conditions],
            verification={"normalized": normalized, "nonnegative": nonnegative},
        )

    def as_distribution(self) -> Distribution:
        return _make(
            "transformed", self.variable, {}, self.density, self.support,
            input_trust=self.input_trust,
            conditions=[sp.sstr(self.condition), *self.conditions],
        )

    def query(self, query: str, **parameters: Any) -> ProbabilityResult:
        return self.as_distribution().query(query, **parameters)

    def pdf(self, point: sp.Basic | None = None) -> ProbabilityResult:
        return self.as_distribution().pdf(point)

    def cdf(self, point: sp.Basic | None = None) -> ProbabilityResult:
        return self.as_distribution().cdf(point)

    def mean(self) -> ProbabilityResult:
        return self.as_distribution().mean()

    def variance(self) -> ProbabilityResult:
        return self.as_distribution().variance()


def _make(
    name: DistributionName,
    x: sp.Symbol,
    parameters: dict[str, sp.Basic],
    density: sp.Basic,
    support: sp.Set,
    *,
    input_trust: str,
    known: dict[str, sp.Basic | None] | None = None,
    nonexisting: set[str] | None = None,
    unsupported: set[str] | None = None,
    conditions: list[str] | None = None,
    assumptions: tuple[sp.Basic, ...] = (),
    source_support: sp.Set | None = None,
    inverse_branches: tuple[sp.Basic, ...] = (),
    jacobians: tuple[sp.Basic, ...] = (),
) -> Distribution:
    return Distribution(
        name=name, parameters=parameters, variable=x, density=density, support=support,
        input_trust=input_trust, known_results=known or {},
        nonexisting=nonexisting or set(), unsupported=unsupported or set(),
        conditions=conditions or [], assumptions=assumptions,
        source_support=source_support,
        inverse_branches=inverse_branches, jacobians=jacobians,
    )


def Uniform(a: sp.Basic, b: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    a, b = _require_expr(a, "a"), _require_expr(b, "b")
    conditions = _greater_condition(b, a, "b", assumptions)
    x = variable or sp.Symbol("x", real=True)
    return _make("uniform", x, {"a": a, "b": b}, 1 / (b - a), sp.Interval(a, b),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": (a + b) / 2, "variance": (b - a) ** 2 / 12,
                        "entropy": sp.log(b - a)})


def Normal(mu: sp.Basic, sigma: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    mu, sigma = _require_expr(mu, "mu"), _require_expr(sigma, "sigma")
    conditions = _positive_condition(sigma, "sigma", assumptions)
    x = variable or sp.Symbol("x", real=True)
    pdf = sp.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi))
    return _make("normal", x, {"mu": mu, "sigma": sigma}, pdf, sp.S.Reals,
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": mu, "variance": sigma**2,
                        "entropy": sp.log(sigma * sp.sqrt(2 * sp.pi * sp.E))})


def LogNormal(mu: sp.Basic, sigma: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    mu, sigma = _require_expr(mu, "mu"), _require_expr(sigma, "sigma")
    conditions = _positive_condition(sigma, "sigma", assumptions)
    x = variable or sp.Symbol("x", positive=True)
    pdf = sp.exp(-(sp.log(x) - mu) ** 2 / (2 * sigma**2)) / (x * sigma * sp.sqrt(2 * sp.pi))
    return _make("lognormal", x, {"mu": mu, "sigma": sigma}, pdf, sp.Interval.open(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": sp.exp(mu + sigma**2 / 2),
                        "variance": (sp.exp(sigma**2) - 1) * sp.exp(2 * mu + sigma**2),
                        "entropy": mu + sp.log(sigma * sp.sqrt(2 * sp.pi * sp.E))},
                 nonexisting={"mgf"})


def Exponential(rate: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    rate = _require_expr(rate, "rate")
    conditions = _positive_condition(rate, "rate", assumptions)
    x = variable or sp.Symbol("x", nonnegative=True)
    return _make("exponential", x, {"rate": rate}, rate * sp.exp(-rate * x), sp.Interval(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": 1 / rate, "variance": 1 / rate**2, "entropy": 1 - sp.log(rate)})


def Gamma(shape: sp.Basic, rate: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    shape, rate = _require_expr(shape, "shape"), _require_expr(rate, "rate")
    conditions = [*_positive_condition(shape, "shape", assumptions), *_positive_condition(rate, "rate", assumptions)]
    x = variable or sp.Symbol("x", positive=True)
    pdf = rate**shape * x ** (shape - 1) * sp.exp(-rate * x) / sp.gamma(shape)
    entropy = shape - sp.log(rate) + sp.loggamma(shape) + (1 - shape) * sp.polygamma(0, shape)
    return _make("gamma", x, {"shape": shape, "rate": rate}, pdf, sp.Interval.open(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": shape / rate, "variance": shape / rate**2, "entropy": entropy})


def Beta(alpha: sp.Basic, beta: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    alpha, beta = _require_expr(alpha, "alpha"), _require_expr(beta, "beta")
    conditions = [*_positive_condition(alpha, "alpha", assumptions), *_positive_condition(beta, "beta", assumptions)]
    x = variable or sp.Symbol("x", real=True)
    pdf = x ** (alpha - 1) * (1 - x) ** (beta - 1) / sp.beta(alpha, beta)
    entropy = (sp.log(sp.beta(alpha, beta)) - (alpha - 1) * sp.polygamma(0, alpha)
               - (beta - 1) * sp.polygamma(0, beta)
               + (alpha + beta - 2) * sp.polygamma(0, alpha + beta))
    return _make("beta", x, {"alpha": alpha, "beta": beta}, pdf, sp.Interval(0, 1),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": alpha / (alpha + beta),
                        "variance": alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)),
                        "entropy": entropy})


def Cauchy(location: sp.Basic, scale: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    location, scale = _require_expr(location, "location"), _require_expr(scale, "scale")
    conditions = _positive_condition(scale, "scale", assumptions)
    x = variable or sp.Symbol("x", real=True)
    pdf = 1 / (sp.pi * scale * (1 + ((x - location) / scale) ** 2))
    return _make("cauchy", x, {"location": location, "scale": scale}, pdf, sp.S.Reals,
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"entropy": sp.log(4 * sp.pi * scale)},
                 nonexisting={"mean", "variance", "mgf", "moment"})


def StudentT(df: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    df = _require_expr(df, "df"); conditions = _positive_condition(df, "df", assumptions)
    x = variable or sp.Symbol("x", real=True)
    pdf = sp.gamma((df + 1) / 2) / (sp.sqrt(df * sp.pi) * sp.gamma(df / 2)) * (1 + x**2 / df) ** (-(df + 1) / 2)
    known: dict[str, sp.Basic | None] = {}
    nonexisting = {"mgf"}
    if _known_gt(df, 1, assumptions):
        known["mean"] = sp.S.Zero
    elif _known_le(df, 1, assumptions):
        nonexisting.add("mean")
    else:
        known["mean"] = None
    if _known_gt(df, 2, assumptions):
        known["variance"] = df / (df - 2)
    elif _known_le(df, 2, assumptions):
        nonexisting.add("variance")
    else:
        known["variance"] = None
    return _make("student_t", x, {"df": df}, pdf, sp.S.Reals, input_trust=input_trust, conditions=conditions, assumptions=assumptions, known=known, nonexisting=nonexisting)


def ChiSquared(df: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    df = _require_expr(df, "df"); conditions = _positive_condition(df, "df", assumptions)
    x = variable or sp.Symbol("x", positive=True)
    pdf = x ** (df / 2 - 1) * sp.exp(-x / 2) / (2 ** (df / 2) * sp.gamma(df / 2))
    return _make("chi_squared", x, {"df": df}, pdf, sp.Interval.open(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions, known={"mean": df, "variance": 2 * df})


def F(d1: sp.Basic, d2: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    d1, d2 = _require_expr(d1, "d1"), _require_expr(d2, "d2")
    conditions = [*_positive_condition(d1, "d1", assumptions), *_positive_condition(d2, "d2", assumptions)]
    x = variable or sp.Symbol("x", positive=True)
    pdf = sp.sqrt((d1 * x) ** d1 * d2**d2 / (d1 * x + d2) ** (d1 + d2)) / (x * sp.beta(d1 / 2, d2 / 2))
    known: dict[str, sp.Basic | None] = {}
    nonexisting: set[str] = set()
    if _known_gt(d2, 2, assumptions):
        known["mean"] = d2 / (d2 - 2)
    elif _known_le(d2, 2, assumptions):
        nonexisting.add("mean")
    else:
        known["mean"] = None
    if _known_gt(d2, 4, assumptions):
        known["variance"] = 2 * d2**2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4))
    elif _known_le(d2, 4, assumptions):
        nonexisting.add("variance")
    else:
        known["variance"] = None
    return _make("f", x, {"d1": d1, "d2": d2}, pdf, sp.Interval.open(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions, known=known,
                 nonexisting=nonexisting | {"mgf"})


def Weibull(shape: sp.Basic, scale: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    shape, scale = _require_expr(shape, "shape"), _require_expr(scale, "scale")
    conditions = [*_positive_condition(shape, "shape", assumptions), *_positive_condition(scale, "scale", assumptions)]
    x = variable or sp.Symbol("x", nonnegative=True)
    pdf = shape / scale * (x / scale) ** (shape - 1) * sp.exp(-(x / scale) ** shape)
    mean = scale * sp.gamma(1 + 1 / shape)
    return _make("weibull", x, {"shape": shape, "scale": scale}, pdf, sp.Interval(0, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": mean,
                        "variance": scale**2 * (
                            sp.gamma(1 + 2 / shape) - sp.gamma(1 + 1 / shape) ** 2
                        ), "entropy": sp.EulerGamma * (1 - 1 / shape) + sp.log(scale / shape) + 1})


def Pareto(scale: sp.Basic, shape: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    scale, shape = _require_expr(scale, "scale"), _require_expr(shape, "shape")
    conditions = [*_positive_condition(scale, "scale", assumptions), *_positive_condition(shape, "shape", assumptions)]
    x = variable or sp.Symbol("x", positive=True)
    pdf = shape * scale**shape / x ** (shape + 1)
    known: dict[str, sp.Basic | None] = {"entropy": sp.log(scale / shape) + 1 + 1 / shape}
    nonexisting = {"mgf"}
    if _known_gt(shape, 1, assumptions):
        known["mean"] = shape * scale / (shape - 1)
    elif _known_le(shape, 1, assumptions):
        nonexisting.add("mean")
    else:
        known["mean"] = None
    if _known_gt(shape, 2, assumptions):
        known["variance"] = shape * scale**2 / ((shape - 1) ** 2 * (shape - 2))
    elif _known_le(shape, 2, assumptions):
        nonexisting.add("variance")
    else:
        known["variance"] = None
    return _make("pareto", x, {"scale": scale, "shape": shape}, pdf, sp.Interval(scale, sp.oo),
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions, known=known, nonexisting=nonexisting)


def Laplace(location: sp.Basic, scale: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    location, scale = _require_expr(location, "location"), _require_expr(scale, "scale")
    conditions = _positive_condition(scale, "scale", assumptions)
    x = variable or sp.Symbol("x", real=True)
    pdf = sp.exp(-sp.Abs(x - location) / scale) / (2 * scale)
    return _make("laplace", x, {"location": location, "scale": scale}, pdf, sp.S.Reals,
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": location, "variance": 2 * scale**2,
                        "entropy": 1 + sp.log(2 * scale)})


def Logistic(location: sp.Basic, scale: sp.Basic, *, variable: sp.Symbol | None = None, input_trust: str = "symbolic", assumptions: tuple[sp.Basic, ...] = ()) -> Distribution:
    location, scale = _require_expr(location, "location"), _require_expr(scale, "scale")
    conditions = _positive_condition(scale, "scale", assumptions)
    x = variable or sp.Symbol("x", real=True)
    z = sp.exp(-(x - location) / scale)
    pdf = z / (scale * (1 + z) ** 2)
    return _make("logistic", x, {"location": location, "scale": scale}, pdf, sp.S.Reals,
                 input_trust=input_trust, conditions=conditions, assumptions=assumptions,
                 known={"mean": location, "variance": sp.pi**2 * scale**2 / 3,
                        "entropy": 2 + sp.log(scale)})


def transform(
    random_variable: RandomVariable,
    expression: sp.Basic,
    target: sp.Symbol,
    *,
    inverse_branches: list[sp.Basic] | None = None,
    jacobians: list[sp.Basic] | None = None,
) -> RandomVariable:
    expression = _require_expr(expression, "expression")
    if not isinstance(target, sp.Symbol):
        raise TypeError("target must be a SymPy Symbol")
    source = random_variable.symbol
    distribution = random_variable.distribution
    expression = expression.xreplace({distribution.variable: source})
    try:
        image = sp.calculus.util.function_range(
            expression, source, distribution.support)
    except (NotImplementedError, ValueError):
        image = sp.S.Reals

    branches = inverse_branches
    if branches is None:
        derivative = sp.diff(expression, source)
        increasing = sp.ask(sp.Q.positive(derivative)) is True
        decreasing = sp.ask(sp.Q.negative(derivative)) is True
        if not (increasing or decreasing):
            raise ValueError("transformation is not provably monotone; supply every inverse branch")
        solved = sp.solve(sp.Eq(target, expression), source)
        branches = [branch for branch in solved if isinstance(branch, sp.Basic)]
        if len(branches) != 1:
            raise ValueError("a unique inverse branch was not established")
    else:
        branches = [_require_expr(branch, "inverse branch") for branch in branches]
        if any(sp.simplify(expression.xreplace({source: branch}) - target) != 0
               for branch in branches):
            raise ValueError("a supplied inverse branch does not satisfy the transformation")
        solved = sp.solve(sp.Eq(target, expression), source)
        if isinstance(solved, list):
            relevant: list[sp.Basic] = []
            for solution in solved:
                try:
                    source_range = sp.calculus.util.function_range(
                        solution, target, image)
                    overlap = source_range.intersect(distribution.support)
                    contributes = (
                        overlap is not sp.S.EmptySet
                        and not isinstance(overlap, sp.FiniteSet)
                    )
                except (NotImplementedError, ValueError):
                    contributes = True
                if contributes:
                    relevant.append(solution)
            if any(
                not any(sp.simplify(solution - branch) == 0 for branch in branches)
                for solution in relevant
            ):
                raise ValueError("inverse branches lose part of the source support")

    if jacobians is not None and len(jacobians) != len(branches):
        raise ValueError("one Jacobian is required per inverse branch")
    expected_jacobians = [sp.Abs(sp.diff(branch, target)) for branch in branches]
    if jacobians is not None:
        js = [_require_expr(j, "Jacobian") for j in jacobians]
        if any(sp.simplify(actual - expected) != 0
               for actual, expected in zip(js, expected_jacobians)):
            raise ValueError("a supplied Jacobian failed the defining derivative check")
    else:
        js = expected_jacobians
    source_pdf = distribution.density.xreplace({distribution.variable: source})
    terms = [source_pdf.xreplace({source: branch}) * jac for branch, jac in zip(branches, js)]
    density = sp.simplify(sum(terms))
    if image.is_subset(sp.Interval(0, sp.oo)) is True:
        positive_target = sp.Dummy("positive_target", positive=True)
        density = sp.simplify(
            density.xreplace({target: positive_target})
        ).xreplace({positive_target: target})
    transformed = _make(
        "transformed", target, {"expression": expression}, density, image,
        input_trust=distribution.input_trust,
        conditions=[
            *distribution.conditions,
            "density obtained by change of variables over every inverse branch",
        ],
        source_support=distribution.support,
        inverse_branches=tuple(branches),
        jacobians=tuple(js),
    )
    return RandomVariable(symbol=target, distribution=transformed)


def expectation(distribution: Distribution, expression: sp.Basic) -> ProbabilityResult:
    expression = _require_expr(expression, "expression")
    value = sp.integrate(
        expression * distribution.density,
        (distribution.variable, distribution.support),
    )
    if value.has(sp.Integral):
        return _absent(
            "expectation", QueryStatus.UNKNOWN,
            "expectation integral was not resolved")
    return _symbolic_result(
        "expectation", value, distribution.input_trust,
        method="expectation_integral", conditions=distribution.conditions)


def bayes_rule(
    joint: JointDistribution,
    variable: sp.Symbol,
    given: dict[sp.Symbol, sp.Basic],
) -> ConditionalDistribution:
    if not isinstance(joint, JointDistribution):
        raise TypeError("joint must be a JointDistribution")
    return joint.condition(variable, given)


def covariance(
    joint: JointDistribution, left: sp.Symbol, right: sp.Symbol,
) -> ProbabilityResult:
    if not isinstance(joint, JointDistribution):
        raise TypeError("joint must be a JointDistribution")
    return joint.covariance(left, right)


def correlation(
    joint: JointDistribution, left: sp.Symbol, right: sp.Symbol,
) -> ProbabilityResult:
    if not isinstance(joint, JointDistribution):
        raise TypeError("joint must be a JointDistribution")
    return joint.correlation(left, right)


def order_statistic(
    distribution: Distribution, sample_size: int, order: int,
) -> Distribution:
    if not isinstance(distribution, Distribution):
        raise TypeError("distribution must be a Distribution")
    return distribution.order_statistic(sample_size, order)


def mixture(
    weights: list[sp.Basic],
    components: list[Distribution],
    *,
    variable: sp.Symbol | None = None,
) -> Distribution:
    if not components or len(weights) != len(components):
        raise ValueError("mixture requires one weight per component")
    weights = [_require_expr(weight, "weight") for weight in weights]
    if any(sp.ask(sp.Q.nonnegative(weight)) is not True for weight in weights):
        raise ValueError("mixture weights must be provably nonnegative")
    if sp.simplify(sum(weights) - 1) != 0:
        raise ValueError("mixture weights must sum exactly to one")
    x = variable or sp.Symbol("x", real=True)
    density = sp.simplify(sum(
        weight * component.density.xreplace({component.variable: x})
        for weight, component in zip(weights, components)
    ))
    support = sp.Union(*[component.support for component in components])
    trust = min(
        (component.input_trust for component in components),
        key=_TRUST.__getitem__,
    )
    return _make(
        "transformed", x,
        {f"weight_{index}": weight for index, weight in enumerate(weights)},
        density, support, input_trust=trust,
        conditions=["mixture weights are nonnegative and sum to one"])


def truncate(
    distribution: Distribution,
    lower: sp.Basic,
    upper: sp.Basic,
) -> Distribution:
    lower = _require_expr(lower, "lower")
    upper = _require_expr(upper, "upper")
    bound_conditions = _greater_condition(upper, lower, "upper")
    support = distribution.support.intersect(sp.Interval(lower, upper))
    normalization = sp.integrate(
        distribution.density, (distribution.variable, support))
    if normalization.has(sp.Integral) or normalization.is_zero is not False:
        raise ValueError("truncation normalization was not established as nonzero")
    return _make(
        "transformed", distribution.variable,
        {**distribution.parameters, "lower": lower, "upper": upper},
        sp.simplify(distribution.density / normalization),
        support,
        input_trust=distribution.input_trust,
        conditions=[
            *distribution.conditions,
            *bound_conditions,
            "density renormalized on the truncated support",
        ],
    )


def convolve(
    left: Distribution,
    right: Distribution,
    *,
    variable: sp.Symbol | None = None,
) -> Distribution:
    target = variable or sp.Symbol("x", real=True)
    integration_variable = sp.Dummy("convolution", real=True)
    left_density = sp.Piecewise(
        (left.density.xreplace({left.variable: integration_variable}),
         left.support.contains(integration_variable)),
        (0, True),
    )
    shifted = target - integration_variable
    right_density = sp.Piecewise(
        (right.density.xreplace({right.variable: shifted}),
         right.support.contains(shifted)),
        (0, True),
    )
    density = sp.integrate(
        left_density * right_density,
        (integration_variable, -sp.oo, sp.oo),
    )
    if isinstance(left.support, sp.Interval) and isinstance(right.support, sp.Interval):
        support = sp.Interval(
            left.support.start + right.support.start,
            left.support.end + right.support.end,
            left_open=left.support.left_open or right.support.left_open,
            right_open=left.support.right_open or right.support.right_open,
        )
    else:
        support = sp.S.Reals
    trust = min((left.input_trust, right.input_trust), key=_TRUST.__getitem__)
    return _make(
        "transformed", target, {}, density, support, input_trust=trust,
        unsupported={"quantile"} if density.has(sp.Integral) else set(),
        conditions=[
            *left.conditions, *right.conditions,
            "convolution assumes independent summands",
        ],
    )


def cross_entropy(
    source: Distribution, reference: Distribution,
) -> ProbabilityResult:
    x = source.variable
    q = reference.density.xreplace({reference.variable: x})
    value = -sp.integrate(
        source.density * sp.log(q), (x, source.support))
    if value.has(sp.Integral):
        return _absent(
            "cross_entropy", QueryStatus.UNKNOWN,
            "cross-entropy integral was not resolved")
    trust = min(
        (source.input_trust, reference.input_trust), key=_TRUST.__getitem__)
    return _symbolic_result(
        "cross_entropy", value, trust, method="cross_entropy_integral")


def kl_divergence(
    source: Distribution, reference: Distribution,
) -> ProbabilityResult:
    entropy = cross_entropy(source, reference)
    source_entropy = source.entropy()
    if entropy.status is not QueryStatus.AVAILABLE \
            or source_entropy.status is not QueryStatus.AVAILABLE:
        return _absent(
            "kl_divergence", QueryStatus.UNKNOWN,
            "KL divergence components were not resolved")
    trust = min(
        (source.input_trust, reference.input_trust), key=_TRUST.__getitem__)
    return _symbolic_result(
        "kl_divergence",
        sp.simplify(entropy.value - source_entropy.value),
        trust,
        method="cross_entropy_minus_entropy",
    )


__all__ = [
    "Beta", "Cauchy", "ChiSquared", "ConditionalDistribution", "Distribution",
    "Exponential", "F", "Gamma", "JointDistribution", "Laplace", "LogNormal",
    "Logistic", "Normal", "Pareto", "ProbabilityResult", "QueryStatus",
    "RandomVariable", "StudentT", "Uniform", "Weibull", "transform",
    "bayes_rule", "convolve", "correlation", "covariance", "cross_entropy",
    "expectation", "kl_divergence", "mixture", "order_statistic", "truncate",
]
