# =============================================================================
# MathKernel - integral transforms
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from sympy.integrals.transforms import IntegralTransform

from mathkernel_artifacts.evidence import (
    ComputationEvidence,
    EvidenceBundle,
    ProofEvidence,
    TRUST_RANK,
)

from .models import TrustLevel


TransformKind = Literal[
    "laplace",
    "inverse_laplace",
    "fourier",
    "inverse_fourier",
    "mellin",
    "inverse_mellin",
    "z",
    "inverse_z",
]
TransformFamily = Literal["laplace", "fourier", "mellin", "z"]


@lru_cache(maxsize=256)
def _laplace_cached(expression, variable, target):
    return sp.laplace_transform(expression, variable, target, noconds=False)


@lru_cache(maxsize=256)
def _inverse_laplace_cached(expression, variable, target):
    return sp.inverse_laplace_transform(expression, variable, target)


@lru_cache(maxsize=256)
def _fourier_angular_cached(expression, variable, target):
    frequency = sp.Dummy("cycles_frequency")
    value = sp.fourier_transform(expression, variable, frequency)
    return value.subs(frequency, target / (2 * sp.pi))


@lru_cache(maxsize=256)
def _inverse_fourier_angular_cached(expression, variable, target):
    frequency = sp.Dummy("cycles_frequency")
    spectrum = expression.subs(variable, 2 * sp.pi * frequency)
    return sp.inverse_fourier_transform(spectrum, frequency, target)


@lru_cache(maxsize=256)
def _mellin_cached(expression, variable, target):
    return sp.mellin_transform(expression, variable, target)


@lru_cache(maxsize=256)
def _inverse_mellin_cached(expression, variable, target, strip):
    return sp.inverse_mellin_transform(expression, variable, target, strip)


class TransformConvention(BaseModel):
    """Explicit kernel and normalization metadata for a transform pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    family: TransformFamily
    forward_kernel: str
    inverse_kernel: str
    forward_normalization: sp.Expr
    inverse_normalization: sp.Expr

    @field_serializer("forward_normalization", "inverse_normalization", when_used="json")
    def serialize_normalization(self, value: sp.Expr) -> str:
        return sp.sstr(value)

    @classmethod
    def laplace_standard(cls) -> "TransformConvention":
        return cls(
            family="laplace",
            forward_kernel="exp(-s*t)",
            inverse_kernel="exp(s*t)",
            forward_normalization=sp.S.One,
            inverse_normalization=1 / (2 * sp.pi * sp.I),
        )

    @classmethod
    def fourier_angular_frequency(cls) -> "TransformConvention":
        return cls(
            family="fourier",
            forward_kernel="exp(-I*omega*t)",
            inverse_kernel="exp(I*omega*t)",
            forward_normalization=sp.S.One,
            inverse_normalization=1 / (2 * sp.pi),
        )

    @classmethod
    def mellin_standard(cls) -> "TransformConvention":
        return cls(
            family="mellin",
            forward_kernel="x**(s - 1)",
            inverse_kernel="x**(-s)",
            forward_normalization=sp.S.One,
            inverse_normalization=1 / (2 * sp.pi * sp.I),
        )

    @classmethod
    def z_bilateral(cls) -> "TransformConvention":
        return cls(
            family="z",
            forward_kernel="z**(-n)",
            inverse_kernel="z**n",
            forward_normalization=sp.S.One,
            inverse_normalization=1 / (2 * sp.pi * sp.I),
        )


class TransformProblem(BaseModel):
    """A fully specified transform request over caller-owned SymPy objects."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transform: TransformKind
    expression: sp.Expr
    variable: sp.Symbol
    transform_variable: sp.Symbol
    domain: sp.Basic | tuple[sp.Expr, sp.Expr]
    assumptions: tuple[sp.Basic, ...]
    convention: TransformConvention

    @field_serializer("expression", "variable", "transform_variable", "domain",
                      "assumptions", when_used="json")
    def serialize_symbolic(self, value):
        if isinstance(value, tuple):
            return [sp.sstr(item) for item in value]
        return sp.sstr(value)

    @model_validator(mode="after")
    def validate_semantics(self) -> "TransformProblem":
        family = self.transform.removeprefix("inverse_")
        if self.convention.family != family:
            raise ValueError(
                f"{self.convention.family!r} convention cannot be used for "
                f"{self.transform!r}"
            )
        if self.variable == self.transform_variable:
            raise ValueError("variable and transform_variable must be distinct")
        expected_domains = {
            "laplace": sp.Interval(0, sp.oo),
            "fourier": sp.S.Reals,
            "mellin": sp.Interval.open(0, sp.oo),
        }
        expected = expected_domains.get(self.transform)
        if expected is not None and self.domain != expected:
            raise ValueError(
                f"{self.transform} convention requires domain {expected}, "
                f"got {self.domain}")
        if self.transform == "inverse_mellin" and not (
            isinstance(self.domain, tuple) and len(self.domain) == 2
        ):
            raise ValueError("inverse Mellin transform requires an explicit convergence strip")
        if self.transform == "inverse_z":
            if not isinstance(self.domain, tuple) or len(self.domain) != 2:
                raise ValueError("inverse Z-transform requires an explicit annulus")
            inner, outer = self.domain
            if inner.is_extended_nonnegative is False:
                raise ValueError("inverse Z-transform annulus has a negative inner radius")
            ordered = sp.ask(sp.Q.lt(inner, outer), assumptions=sp.And(*self.assumptions))
            if ordered is False:
                raise ValueError("inverse Z-transform annulus must have inner < outer")
        return self


class VerifiedCheck(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    status: Literal["verified", "unknown"]
    residual: sp.Expr | None = None
    trust: TrustLevel = TrustLevel.UNKNOWN

    @field_serializer("residual", when_used="json")
    def serialize_residual(self, value: sp.Expr | None) -> str | None:
        return None if value is None else sp.sstr(value)


class TransformResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: sp.Expr | None
    roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None = None
    assumptions: list[sp.Basic] = Field(default_factory=list)
    branch_information: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    side_conditions: list[sp.Basic | str] = Field(default_factory=list)
    verified_checks: list[VerifiedCheck] = Field(default_factory=list)
    # This is intentionally separate from ``status``.  A non-empty reported
    # ROC is only a consistency check until an independent convergence proof
    # establishes that the region is mathematically complete/correct.
    roc_verification: Literal[
        "not_applicable", "consistent_only", "verified", "unknown"
    ] = "not_applicable"
    status: Literal["verified", "candidate", "unknown", "unsupported"]
    trust: TrustLevel

    @field_serializer("value", "roc", "assumptions", "side_conditions", when_used="json")
    def serialize_symbolic(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [sp.sstr(item) for item in value]
        return sp.sstr(value)


class TransformEngine:
    """Conservative adapter over SymPy's integral-transform APIs."""

    _SUPPORTED_CONVENTIONS = {
        "laplace": TransformConvention.laplace_standard(),
        "fourier": TransformConvention.fourier_angular_frequency(),
        "mellin": TransformConvention.mellin_standard(),
        "z": TransformConvention.z_bilateral(),
    }

    def transform(
        self,
        problem: TransformProblem,
        *,
        input_trust: TrustLevel = TrustLevel.SYMBOLIC,
        verify: bool = True,
    ) -> TransformResult:
        self._validate_convention(problem.convention)

        try:
            value, roc, side_conditions = self._compute(problem)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return self._unknown(problem, input_trust, str(exc))

        if self._contains_unevaluated_transform(value):
            return self._candidate_for_unevaluated(
                problem, value, roc, side_conditions, input_trust
            )

        checks = self._verification_checks(problem, value, roc) if verify else []
        identity_verified = any(
            check.name == "forward_inverse_round_trip"
            and check.status == "verified"
            and check.residual == 0
            for check in checks
        )
        unresolved_roc = any(
            "not independently established" in str(condition)
            or "unresolved" in str(condition).lower()
            for condition in side_conditions
        )
        if problem.transform in {
            "laplace", "inverse_laplace", "fourier", "inverse_fourier",
            "mellin", "inverse_mellin", "z", "inverse_z",
        }:
            identity_verified = (
                identity_verified
                and roc is not None
                and not unresolved_roc
                and any(
                    check.name == "roc_nonempty_consistency"
                    and check.status == "verified"
                    for check in checks
                ))
        # Same-engine symbolic round-tripping is a useful consistency check,
        # but it is not independent proof and cannot promote a claim to EXACT.
        derived_trust = (
            TrustLevel.HEURISTIC
            if problem.transform == "z" and roc is None
            else TrustLevel.SYMBOLIC
        )
        trust = self._minimum_trust(derived_trust, input_trust)
        roc_check = next((
            check for check in checks
            if check.name == "roc_nonempty_consistency"
        ), None)
        if roc is None:
            roc_verification = (
                "unknown"
                if problem.transform in {"z", "inverse_z"}
                else "not_applicable"
            )
        elif roc_check is not None and roc_check.status == "verified":
            roc_verification = "consistent_only"
        else:
            roc_verification = "unknown"
        status: Literal["verified", "candidate"] = (
            "verified" if identity_verified else "candidate"
        )
        evidence = self._evidence(
            problem, input_trust, trust, identity_verified=identity_verified
        )
        return TransformResult(
            value=value,
            roc=roc,
            assumptions=list(problem.assumptions),
            branch_information=self._branch_information(value),
            evidence_bundle=evidence,
            side_conditions=side_conditions,
            verified_checks=checks,
            roc_verification=roc_verification,
            status=status,
            trust=trust,
        )

    def _compute(
        self, problem: TransformProblem
    ) -> tuple[
        sp.Expr,
        sp.Basic | tuple[sp.Expr, sp.Expr] | None,
        list[sp.Basic | str],
    ]:
        expression = problem.expression
        variable = problem.variable
        target = problem.transform_variable

        if problem.transform == "laplace":
            value, plane, condition = _laplace_cached(
                expression, variable, target)
            return value, sp.re(target) > plane, self._condition_list(condition)
        if problem.transform == "inverse_laplace":
            roc = self._inverse_laplace_roc(expression, variable)
            conditions = [] if roc is not None else [
                "The inverse Laplace convergence half-plane was unresolved."
            ]
            return (
                _inverse_laplace_cached(expression, variable, target),
                roc,
                conditions,
            )
        if problem.transform == "fourier":
            convergence = sp.Lt(
                sp.Integral(sp.Abs(expression), (variable, -sp.oo, sp.oo)),
                sp.oo, evaluate=False)
            return _fourier_angular_cached(
                expression, variable, target), convergence, []
        if problem.transform == "inverse_fourier":
            convergence = sp.Lt(
                sp.Integral(sp.Abs(expression), (variable, -sp.oo, sp.oo)),
                sp.oo, evaluate=False)
            return _inverse_fourier_angular_cached(
                expression, variable, target), convergence, []
        if problem.transform == "mellin":
            value, strip, condition = _mellin_cached(
                expression, variable, target)
            return value, strip, self._condition_list(condition)
        if problem.transform == "inverse_mellin":
            if not isinstance(problem.domain, tuple) or len(problem.domain) != 2:
                raise ValueError("inverse Mellin transform requires a two-bound strip")
            value = _inverse_mellin_cached(
                expression, variable, target, problem.domain)
            return value, problem.domain, []
        if problem.transform == "z":
            bounds = problem.domain if isinstance(problem.domain, tuple) else (-sp.oo, sp.oo)
            value = sp.summation(
                expression * target ** (-variable),
                (variable, bounds[0], bounds[1]),
            )
            return value, None, [
                "The bilateral Z-transform region of convergence was not independently established."
            ]
        if problem.transform == "inverse_z":
            value, resolved = self._inverse_z_laurent(problem)
            conditions = [
                "Inverse Z-transform uses the Laurent coefficient [z**(-n)] X(z), "
                "equivalently the contour residue of X(z)*z**(n - 1).",
                "The supplied annulus must contain the inversion contour and no poles.",
            ]
            if not resolved:
                conditions.append(
                    "Pole placement or Laurent coefficient extraction was unresolved."
                )
            return value, problem.domain, conditions
        raise NotImplementedError(f"Unsupported transform: {problem.transform}")

    @staticmethod
    def _inverse_laplace_roc(
        expression: sp.Expr, variable: sp.Symbol,
    ) -> sp.Basic | None:
        """Infer a right half-plane from explicit isolated singularities."""
        try:
            singularities = sp.singularities(expression, variable)
        except (NotImplementedError, ValueError):
            return None
        if not isinstance(singularities, sp.FiniteSet):
            return None
        if not singularities:
            return sp.S.true
        boundaries = [sp.re(point) for point in singularities]
        if any(boundary.has(sp.re) for boundary in boundaries):
            return None
        boundary = boundaries[0] if len(boundaries) == 1 else sp.Max(*boundaries)
        return sp.re(variable) > boundary

    def _verification_checks(
        self,
        problem: TransformProblem,
        value: sp.Expr,
        roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None,
    ) -> list[VerifiedCheck]:
        checks = self._verify_round_trip(problem, value, roc)
        checks.extend(self._verify_linearity(problem, value))

        if problem.expression.has(sp.Integral):
            checks.append(self._verify_convolution(problem, value))
        if problem.expression.has(sp.Derivative):
            checks.append(self._verify_differentiation(problem, value))
        if problem.transform == "laplace":
            checks.extend(self._verify_laplace_value_theorems(problem, value))
        elif problem.transform == "z":
            checks.extend([
                VerifiedCheck(name="initial_value_theorem", status="unknown"),
                VerifiedCheck(name="final_value_theorem", status="unknown"),
            ])
        if roc is not None or problem.transform in {"z", "inverse_z"}:
            checks.append(self._verify_roc_nonempty_consistency(problem, roc))
        return checks

    def _verify_differentiation(
        self, problem: TransformProblem, value: sp.Expr,
    ) -> VerifiedCheck:
        derivative = problem.expression
        if not isinstance(derivative, sp.Derivative):
            return VerifiedCheck(
                name="differentiation_theorem", status="unknown")
        count = sum(
            amount for symbol, amount in derivative.variable_count
            if symbol == problem.variable)
        if count < 1 or any(
            symbol != problem.variable
            for symbol, _ in derivative.variable_count
        ):
            return VerifiedCheck(
                name="differentiation_theorem", status="unknown")
        try:
            base = derivative.expr
            base_transform = self._compute(
                problem.model_copy(update={"expression": base}))[0]
            if problem.transform == "laplace":
                expected = problem.transform_variable ** count * base_transform
                expected -= sum(
                    problem.transform_variable ** (count - 1 - order)
                    * sp.limit(
                        sp.diff(base, problem.variable, order),
                        problem.variable, 0, dir="+")
                    for order in range(count)
                )
            elif problem.transform == "fourier":
                expected = (
                    sp.I * problem.transform_variable) ** count * base_transform
            else:
                return VerifiedCheck(
                    name="differentiation_theorem", status="unknown")
            residual = sp.simplify(value - expected)
            verified = residual == 0
            return VerifiedCheck(
                name="differentiation_theorem",
                status="verified" if verified else "unknown",
                residual=residual,
                trust=(
                    TrustLevel.SYMBOLIC if verified
                    else TrustLevel.UNKNOWN),
            )
        except (ValueError, TypeError, NotImplementedError):
            return VerifiedCheck(
                name="differentiation_theorem", status="unknown")

    def _verify_convolution(
        self, problem: TransformProblem, value: sp.Expr,
    ) -> VerifiedCheck:
        if problem.transform not in {"laplace", "fourier"}:
            return VerifiedCheck(name="convolution_theorem", status="unknown")
        integrals = list(problem.expression.atoms(sp.Integral))
        if len(integrals) != 1:
            return VerifiedCheck(name="convolution_theorem", status="unknown")
        integral = integrals[0]
        if len(integral.limits) != 1 or not isinstance(
            integral.function, sp.Mul
        ):
            return VerifiedCheck(name="convolution_theorem", status="unknown")
        tau, *bounds = integral.limits[0]
        factors = list(integral.function.args)
        if len(factors) != 2 or len(bounds) != 2:
            return VerifiedCheck(name="convolution_theorem", status="unknown")
        expected_bounds = (
            (0, problem.variable)
            if problem.transform == "laplace"
            else (-sp.oo, sp.oo)
        )
        if tuple(bounds) != expected_bounds:
            return VerifiedCheck(name="convolution_theorem", status="unknown")
        try:
            pure = [
                factor for factor in factors
                if factor.has(tau) and not factor.has(problem.variable)
            ]
            shifted = [
                factor for factor in factors
                if factor.has(tau) and factor.has(problem.variable)
            ]
            if len(pure) != 1 or len(shifted) != 1:
                return VerifiedCheck(
                    name="convolution_theorem", status="unknown")
            first = pure[0].subs(tau, problem.variable)
            second = shifted[0].subs(tau, 0)
            if second.has(tau):
                return VerifiedCheck(
                    name="convolution_theorem", status="unknown")
            first_transform = self._compute(
                problem.model_copy(update={"expression": first}))[0]
            second_transform = self._compute(
                problem.model_copy(update={"expression": second}))[0]
            direct_transform = self._compute(problem.model_copy(
                update={"expression": integral.doit()}))[0]
            residual = sp.simplify(
                direct_transform - first_transform * second_transform)
            verified = residual == 0
            return VerifiedCheck(
                name="convolution_theorem",
                status="verified" if verified else "unknown",
                residual=residual,
                trust=(
                    TrustLevel.SYMBOLIC if verified
                    else TrustLevel.UNKNOWN),
            )
        except (ValueError, TypeError, NotImplementedError):
            return VerifiedCheck(name="convolution_theorem", status="unknown")

    @staticmethod
    def _verify_laplace_value_theorems(
        problem: TransformProblem, value: sp.Expr,
    ) -> list[VerifiedCheck]:
        target = problem.transform_variable
        variable = problem.variable
        source = problem.expression.doit()
        checks: list[VerifiedCheck] = []
        try:
            initial_residual = sp.simplify(
                sp.limit(target * value, target, sp.oo)
                - sp.limit(source, variable, 0, dir="+"))
            initial_verified = initial_residual == 0
        except (ValueError, TypeError, NotImplementedError):
            initial_residual = None
            initial_verified = False
        checks.append(VerifiedCheck(
            name="initial_value_theorem",
            status="verified" if initial_verified else "unknown",
            residual=initial_residual,
            trust=(
                TrustLevel.SYMBOLIC if initial_verified
                else TrustLevel.UNKNOWN),
        ))
        final_verified = False
        final_residual = None
        try:
            poles = sp.singularities(target * value, target)
            stable = isinstance(poles, sp.FiniteSet) and all(
                sp.ask(sp.Q.negative(sp.re(pole))) is True for pole in poles)
            if stable:
                final_residual = sp.simplify(
                    sp.limit(target * value, target, 0, dir="+")
                    - sp.limit(source, variable, sp.oo))
                final_verified = final_residual == 0
        except (ValueError, TypeError, NotImplementedError):
            pass
        checks.append(VerifiedCheck(
            name="final_value_theorem",
            status="verified" if final_verified else "unknown",
            residual=final_residual,
            trust=(
                TrustLevel.SYMBOLIC if final_verified
                else TrustLevel.UNKNOWN),
        ))
        return checks

    def _verify_round_trip(
        self,
        problem: TransformProblem,
        value: sp.Expr,
        roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None,
    ) -> list[VerifiedCheck]:
        try:
            recovered = self._round_trip_expression(problem, value, roc)
            if recovered is None or self._contains_unevaluated_transform(recovered):
                return [
                    VerifiedCheck(
                        name="forward_inverse_round_trip",
                        status="unknown",
                    )
                ]
            residual = sp.simplify(recovered - problem.expression)
            verified = residual == 0
            return [
                VerifiedCheck(
                    name="forward_inverse_round_trip",
                    status="verified" if verified else "unknown",
                    residual=residual,
                    trust=TrustLevel.SYMBOLIC if verified else TrustLevel.UNKNOWN,
                )
            ]
        except (ValueError, TypeError, NotImplementedError):
            return [
                VerifiedCheck(
                    name="forward_inverse_round_trip",
                    status="unknown",
                )
            ]

    def _round_trip_expression(
        self,
        problem: TransformProblem,
        value: sp.Expr,
        roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None,
    ) -> sp.Expr | None:
        variable = problem.variable
        target = problem.transform_variable
        if problem.transform == "laplace":
            return _inverse_laplace_cached(value, target, variable)
        if problem.transform == "inverse_laplace":
            return _laplace_cached(value, target, variable)[0]
        if problem.transform == "fourier":
            return _inverse_fourier_angular_cached(value, target, variable)
        if problem.transform == "inverse_fourier":
            return _fourier_angular_cached(value, target, variable)
        if problem.transform == "mellin":
            if not isinstance(roc, tuple):
                return None
            return _inverse_mellin_cached(value, target, variable, roc)
        if problem.transform == "inverse_mellin":
            return _mellin_cached(value, target, variable)[0]
        if problem.transform in {"z", "inverse_z"}:
            return None
        return None

    def _verify_linearity(
        self, problem: TransformProblem, value: sp.Expr
    ) -> list[VerifiedCheck]:
        if not isinstance(problem.expression, sp.Add):
            return [VerifiedCheck(name="linearity", status="unknown")]
        try:
            term_values = [
                self._compute(problem.model_copy(update={"expression": term}))[0]
                for term in problem.expression.args
            ]
            if any(self._contains_unevaluated_transform(item) for item in term_values):
                return [VerifiedCheck(name="linearity", status="unknown")]
            residual = sp.simplify(value - sp.Add(*term_values))
            verified = residual == 0
            return [
                VerifiedCheck(
                    name="linearity",
                    status="verified" if verified else "unknown",
                    residual=residual,
                    trust=TrustLevel.SYMBOLIC if verified else TrustLevel.UNKNOWN,
                )
            ]
        except (ValueError, TypeError, NotImplementedError):
            return [VerifiedCheck(name="linearity", status="unknown")]

    @staticmethod
    def _verify_roc_nonempty_consistency(
        problem: TransformProblem,
        roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None,
    ) -> VerifiedCheck:
        """Check only that a reported ROC is not internally empty.

        This is deliberately *not* a proof that the reported region is the
        mathematically complete/correct region of convergence.  The name is
        kept explicit so this diagnostic cannot be mistaken for ROC proof.
        """
        if isinstance(roc, tuple) and len(roc) == 2:
            ordered = sp.ask(
                sp.Q.lt(roc[0], roc[1]), assumptions=sp.And(*problem.assumptions)
            )
            if ordered is True:
                return VerifiedCheck(
                    name="roc_nonempty_consistency",
                    status="verified",
                    residual=sp.S.Zero,
                    trust=TrustLevel.SYMBOLIC,
                )
        if isinstance(roc, sp.Basic) and roc is not sp.S.false:
            try:
                satisfiable = sp.satisfiable(roc)
            except (NotImplementedError, ValueError):
                satisfiable = False
            if satisfiable is not False:
                return VerifiedCheck(
                    name="roc_nonempty_consistency",
                    status="verified",
                    residual=sp.S.Zero,
                    trust=TrustLevel.SYMBOLIC,
                )
        return VerifiedCheck(name="roc_nonempty_consistency", status="unknown")

    def _inverse_z_laurent(
        self, problem: TransformProblem
    ) -> tuple[sp.Expr, bool]:
        """Extract a rational Laurent coefficient only with a decidable annulus."""
        z = problem.variable
        n = problem.transform_variable
        inner, outer = problem.domain
        if n.is_integer is not True or not problem.expression.is_rational_function(z):
            return sp.Function("InverseZTransform")(problem.expression, z, n), False

        numerator, denominator = sp.cancel(problem.expression).as_numer_denom()
        try:
            quotient, remainder = sp.div(
                numerator, denominator, z, domain="EX"
            )
            roots = sp.roots(denominator, z)
        except sp.PolynomialError:
            return sp.Function("InverseZTransform")(problem.expression, z, n), False
        if sum(roots.values()) != sp.degree(denominator, z):
            return sp.Function("InverseZTransform")(problem.expression, z, n), False

        assumptions = sp.And(*problem.assumptions)
        coefficient = self._laurent_polynomial_coefficient(quotient, z, n)
        proper = sp.cancel(remainder / denominator)
        for pole, multiplicity in roots.items():
            placement = self._pole_placement(pole, inner, outer, assumptions)
            if placement is None:
                return sp.Function("InverseZTransform")(problem.expression, z, n), False
            regularized = sp.cancel((z - pole) ** multiplicity * proper)
            for order in range(1, multiplicity + 1):
                derivative_order = multiplicity - order
                amplitude = sp.diff(
                    regularized, z, derivative_order
                ).subs(z, pole) / sp.factorial(derivative_order)
                coefficient += self._pole_laurent_coefficient(
                    amplitude, pole, order, n, placement
                )
        return sp.simplify(coefficient), True

    @staticmethod
    def _laurent_polynomial_coefficient(
        polynomial: sp.Expr, z: sp.Symbol, n: sp.Symbol
    ) -> sp.Expr:
        result = sp.S.Zero
        for term in sp.Add.make_args(sp.expand(polynomial)):
            power = term.as_powers_dict().get(z, sp.S.Zero)
            amplitude = sp.cancel(term / z**power)
            result += sp.Piecewise((amplitude, sp.Eq(n, -power)), (0, True))
        return result

    @staticmethod
    def _pole_laurent_coefficient(
        amplitude: sp.Expr,
        pole: sp.Expr,
        order: int,
        n: sp.Symbol,
        placement: Literal["inside", "outside"],
    ) -> sp.Expr:
        if pole == 0:
            return sp.Piecewise((amplitude, sp.Eq(n, order)), (0, True))
        if placement == "inside":
            value = (
                amplitude
                * sp.binomial(n - 1, order - 1)
                * pole ** (n - order)
            )
            return sp.Piecewise((value, n >= order), (0, True))
        value = (
            amplitude
            * (-1) ** order
            * sp.binomial(order - n - 1, -n)
            * pole ** (n - order)
        )
        return sp.Piecewise((value, n <= 0), (0, True))

    @staticmethod
    def _pole_placement(
        pole: sp.Expr,
        inner: sp.Expr,
        outer: sp.Expr,
        assumptions: sp.Basic,
    ) -> Literal["inside", "outside"] | None:
        magnitude = sp.Abs(pole)
        if sp.ask(sp.Q.le(magnitude, inner), assumptions=assumptions) is True:
            return "inside"
        if sp.ask(sp.Q.ge(magnitude, outer), assumptions=assumptions) is True:
            return "outside"
        return None

    @classmethod
    def _validate_convention(cls, convention: TransformConvention) -> None:
        expected = cls._SUPPORTED_CONVENTIONS[convention.family]
        same_metadata = (
            convention.forward_kernel == expected.forward_kernel
            and convention.inverse_kernel == expected.inverse_kernel
        )
        same_normalization = (
            sp.simplify(
                convention.forward_normalization - expected.forward_normalization
            )
            == 0
            and sp.simplify(
                convention.inverse_normalization - expected.inverse_normalization
            )
            == 0
        )
        if not same_metadata or not same_normalization:
            raise ValueError(
                f"Unsupported {convention.family} transform convention"
            )

    @staticmethod
    def _condition_list(condition: sp.Basic) -> list[sp.Basic | str]:
        return [] if condition is sp.S.true else [condition]

    @staticmethod
    def _contains_unevaluated_transform(value: sp.Expr) -> bool:
        return (
            isinstance(value, (IntegralTransform, sp.Integral, sp.Sum))
            or value.has(IntegralTransform, sp.Integral, sp.Sum)
            or any(
                getattr(node.func, "__name__", "") == "InverseZTransform"
                for node in sp.preorder_traversal(value)
            )
        )

    @staticmethod
    def _branch_information(value: sp.Expr) -> list[str]:
        functions: set[str] = set()
        if value.has(sp.log):
            functions.add("log")
        if any(
            isinstance(node, sp.Pow) and node.exp.is_integer is not True
            for node in sp.preorder_traversal(value)
        ):
            functions.add("non-integer power")
        if not functions:
            return []
        return [
            "Principal SymPy branches are retained for: " + ", ".join(sorted(functions))
        ]

    @staticmethod
    def _minimum_trust(left: TrustLevel, right: TrustLevel) -> TrustLevel:
        return min((left, right), key=lambda item: TRUST_RANK[item.value])

    @staticmethod
    def _evidence(
        problem: TransformProblem,
        input_trust: TrustLevel,
        result_trust: TrustLevel,
        *,
        identity_verified: bool,
    ) -> EvidenceBundle:
        computation = [
            ComputationEvidence(
                engine="caller",
                method="input_ancestry",
                arithmetic="inherited",
                trust=input_trust.value,
            ),
            ComputationEvidence(
                engine="sympy",
                method=problem.transform,
                arithmetic="symbolic",
                trust=result_trust.value,
                metadata={
                    "domain": sp.sstr(problem.domain),
                    "convention": problem.convention.model_dump(mode="json"),
                },
            ),
        ]
        proof = []
        if identity_verified:
            proof.append(
                ProofEvidence(
                    proposition="Forward/inverse transform round trip",
                    method="exact_simplification_to_zero",
                    engine="sympy",
                    assumptions=[sp.sstr(item) for item in problem.assumptions],
                    verified=True,
                    trust=TrustLevel.SYMBOLIC.value,
                )
            )
        return EvidenceBundle(computation=computation, proof=proof)

    def _candidate_for_unevaluated(
        self,
        problem: TransformProblem,
        value: sp.Expr,
        roc: sp.Basic | tuple[sp.Expr, sp.Expr] | None,
        side_conditions: list[sp.Basic | str],
        input_trust: TrustLevel,
    ) -> TransformResult:
        return TransformResult(
            value=value,
            roc=roc,
            assumptions=list(problem.assumptions),
            branch_information=[],
            evidence_bundle=self._evidence(
                problem, input_trust, TrustLevel.UNKNOWN, identity_verified=False
            ),
            side_conditions=side_conditions,
            verified_checks=self._verification_checks(problem, value, roc),
            status="candidate",
            trust=TrustLevel.UNKNOWN,
        )

    def _unsupported(
        self,
        problem: TransformProblem,
        input_trust: TrustLevel,
        reason: str,
    ) -> TransformResult:
        return TransformResult(
            value=None,
            assumptions=list(problem.assumptions),
            evidence_bundle=self._evidence(
                problem, input_trust, TrustLevel.UNKNOWN, identity_verified=False
            ),
            side_conditions=[reason],
            status="unsupported",
            trust=TrustLevel.UNKNOWN,
        )

    def _unknown(
        self,
        problem: TransformProblem,
        input_trust: TrustLevel,
        reason: str,
    ) -> TransformResult:
        return TransformResult(
            value=None,
            assumptions=list(problem.assumptions),
            evidence_bundle=self._evidence(
                problem, input_trust, TrustLevel.UNKNOWN, identity_verified=False
            ),
            side_conditions=[reason],
            status="unknown",
            trust=TrustLevel.UNKNOWN,
        )
