# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed Phase G.1 PDE representation, classification, and compatibility."""
from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import ComputationEvidence, EvidenceBundle, ModelEvidence

from .engineering import EngineeringModel, EngineeringResult, cap_trust


CompatibilityStatus = Literal["compatible", "incompatible", "undetermined"]
PDEClass = Literal[
    "elliptic", "parabolic", "hyperbolic", "conditional", "nonlinear",
    "system", "first_order", "algebraic", "undetermined",
]


def _expression(value: sp.Expr, name: str) -> None:
    if not isinstance(value, sp.Expr):
        raise ValueError(f"{name} must be a parsed scalar expression")
    if value.has(sp.nan, sp.oo, -sp.oo, sp.zoo, sp.I) or value.is_finite is False:
        raise ValueError(f"{name} must be finite and real-valued")


class PDETerm(EngineeringModel):
    coefficient: sp.Expr
    field: str
    derivative: tuple[int, ...]
    power: int = 1

    @model_validator(mode="after")
    def validate_term(self):
        _expression(self.coefficient, "PDE coefficient")
        if not self.field or not self.field.isidentifier():
            raise ValueError("PDE term field must be a valid identifier")
        if not self.derivative or any(
                isinstance(order, bool) or not isinstance(order, int) or order < 0
                for order in self.derivative):
            raise ValueError("PDE derivative multi-index must be nonnegative integers")
        if isinstance(self.power, bool) or not isinstance(self.power, int) or self.power < 1:
            raise ValueError("PDE term power must be a positive integer")
        return self

    @property
    def order(self) -> int:
        return sum(self.derivative)


class PDEEquation(EngineeringModel):
    terms: tuple[PDETerm, ...]
    source: sp.Expr = sp.S.Zero
    label: str | None = None

    @model_validator(mode="after")
    def validate_equation(self):
        if not self.terms:
            raise ValueError("PDE equations must contain at least one term")
        _expression(self.source, "PDE source")
        if self.label is not None and not self.label.strip():
            raise ValueError("PDE equation label must be nonempty")
        return self

    @property
    def order(self) -> int:
        return max(term.order for term in self.terms)

    @property
    def linear(self) -> bool:
        return all(term.power == 1 for term in self.terms)


class PDEBoundaryCondition(EngineeringModel):
    field: str
    kind: Literal["dirichlet", "neumann", "robin", "periodic"]
    coordinate: str
    side: Literal["lower", "upper"]
    value: sp.Expr = sp.S.Zero
    alpha: sp.Expr = sp.S.One
    beta: sp.Expr = sp.S.Zero
    paired_side: Literal["lower", "upper"] | None = None

    @model_validator(mode="after")
    def validate_condition(self):
        if not self.field.isidentifier() or not self.coordinate.isidentifier():
            raise ValueError("boundary field and coordinate must be valid identifiers")
        for value, name in ((self.value, "boundary value"),
                            (self.alpha, "Robin alpha"),
                            (self.beta, "Robin beta")):
            _expression(value, name)
        if self.kind == "periodic":
            if self.paired_side is None or self.paired_side == self.side:
                raise ValueError("periodic conditions require the opposite paired_side")
        elif self.paired_side is not None:
            raise ValueError("paired_side applies only to periodic conditions")
        if self.kind == "dirichlet" and (self.alpha != 1 or self.beta != 0):
            raise ValueError("Dirichlet conditions use alpha=1 and beta=0")
        if self.kind == "neumann" and (self.alpha != 1 or self.beta != 0):
            raise ValueError("Neumann conditions use the normal-derivative value directly")
        if self.kind == "robin" and self.alpha == 0 and self.beta == 0:
            raise ValueError("Robin alpha and beta cannot both be zero")
        return self


class PDEInitialCondition(EngineeringModel):
    field: str
    derivative_order: int = 0
    time: sp.Expr
    value: sp.Expr

    @model_validator(mode="after")
    def validate_condition(self):
        if not self.field.isidentifier():
            raise ValueError("initial-condition field must be a valid identifier")
        if self.derivative_order not in {0, 1}:
            raise ValueError("initial derivative_order must be zero or one")
        _expression(self.time, "initial time")
        _expression(self.value, "initial value")
        if self.time.free_symbols:
            raise ValueError("initial time must be concrete")
        return self


class PDEProblem(EngineeringModel):
    fields: tuple[str, ...]
    independent_variables: tuple[str, ...]
    time_variable: str | None = None
    domain: tuple[tuple[str, sp.Expr, sp.Expr], ...]
    equations: tuple[PDEEquation, ...]
    boundary_conditions: tuple[PDEBoundaryCondition, ...] = ()
    initial_conditions: tuple[PDEInitialCondition, ...] = ()
    parameters: tuple[tuple[str, sp.Expr | None], ...] = ()
    assumptions: tuple[str, ...] = ()
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_problem(self):
        if (not self.fields or len(set(self.fields)) != len(self.fields) or
                any(not name.isidentifier() for name in self.fields)):
            raise ValueError("PDE fields must be unique valid identifiers")
        variables = self.independent_variables
        if (not variables or len(set(variables)) != len(variables) or
                any(not name.isidentifier() for name in variables)):
            raise ValueError("PDE independent variables must be unique valid identifiers")
        if set(self.fields) & set(variables):
            raise ValueError("PDE fields and independent variables must be distinct")
        if self.time_variable is not None and self.time_variable not in variables:
            raise ValueError("time_variable must name an independent variable")
        if tuple(name for name, _, _ in self.domain) != variables:
            raise ValueError("domain bounds must follow independent_variables")
        if not self.equations:
            raise ValueError("PDE problems require at least one equation")
        for _, lower, upper in self.domain:
            _expression(lower, "domain lower bound"); _expression(upper, "domain upper bound")
            if lower.free_symbols or upper.free_symbols:
                raise ValueError("G.1 rectangular domain bounds must be concrete")
            difference = sp.simplify(upper - lower)
            if difference.is_positive is not True:
                raise ValueError("every PDE domain interval requires lower < upper")
        parameter_names = tuple(name for name, _ in self.parameters)
        if (len(set(parameter_names)) != len(parameter_names) or
                any(not name.isidentifier() for name in parameter_names) or
                set(parameter_names) & (set(self.fields) | set(variables))):
            raise ValueError("PDE parameters must be unique, valid, and distinct")
        allowed = {sp.Symbol(name) for name in (*variables, *parameter_names)}
        for _, value in self.parameters:
            if value is not None:
                _expression(value, "PDE parameter value")
                if value.free_symbols:
                    raise ValueError("assigned PDE parameter values must be concrete")
        for equation in self.equations:
            for term in equation.terms:
                if term.field not in self.fields or len(term.derivative) != len(variables):
                    raise ValueError("PDE term field or derivative dimension is invalid")
                if not term.coefficient.free_symbols <= allowed:
                    raise ValueError("PDE coefficient contains an undeclared symbol")
            if not equation.source.free_symbols <= allowed:
                raise ValueError("PDE source contains an undeclared symbol")
        for condition in self.boundary_conditions:
            if condition.field not in self.fields or condition.coordinate not in variables:
                raise ValueError("boundary condition references an undeclared field or coordinate")
            if self.time_variable is not None and condition.coordinate == self.time_variable:
                raise ValueError("use initial_conditions rather than a boundary on time_variable")
            for value in (condition.value, condition.alpha, condition.beta):
                if not value.free_symbols <= allowed:
                    raise ValueError("boundary condition contains an undeclared symbol")
        for condition in self.initial_conditions:
            if condition.field not in self.fields or self.time_variable is None:
                raise ValueError("initial conditions require a declared field and time_variable")
            if not condition.value.free_symbols <= allowed:
                raise ValueError("initial condition contains an undeclared symbol")
        if any(not assumption.strip() for assumption in self.assumptions):
            raise ValueError("PDE assumptions must be nonempty strings")
        return self


class PDEClassification(EngineeringModel):
    problem_id: str
    equation_index: int
    variable_pair: tuple[str, ...]
    differential_order: int
    linearity: Literal["linear", "nonlinear"]
    classification: PDEClass
    principal_matrix: tuple[tuple[sp.Expr, ...], ...] = ()
    discriminant: sp.Expr | None = None
    classification_cases: tuple[tuple[str, str], ...] = ()
    convention: str = "A*u_xx + B*u_xy + C*u_yy; discriminant B^2 - 4*A*C"
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_classification(self):
        if not self.problem_id or self.equation_index < 0 or self.differential_order < 0:
            raise ValueError("PDE classification provenance is invalid")
        if self.principal_matrix:
            size = len(self.principal_matrix)
            if any(len(row) != size for row in self.principal_matrix):
                raise ValueError("principal matrix must be square")
        return self


class PDECompatibilityCheck(EngineeringModel):
    left: str
    right: str
    status: CompatibilityStatus
    residual: sp.Expr


class PDECompatibilityReport(EngineeringModel):
    problem_id: str
    compatibility: CompatibilityStatus
    checks: tuple[PDECompatibilityCheck, ...]
    incompatible_count: int
    undetermined_count: int
    completeness_checked: bool = False
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_report(self):
        if not self.problem_id:
            raise ValueError("compatibility report requires problem provenance")
        if self.incompatible_count != sum(
                check.status == "incompatible" for check in self.checks):
            raise ValueError("incompatible_count does not reconcile")
        if self.undetermined_count != sum(
                check.status == "undetermined" for check in self.checks):
            raise ValueError("undetermined_count does not reconcile")
        return self


def _bundle(problem: PDEProblem, method: str, trust: str, *, diagnostic: str) -> EvidenceBundle:
    return EvidenceBundle(
        computation=[ComputationEvidence(
            engine="partial_differential_equations", method=method,
            arithmetic=trust, deterministic=True, trust=trust)],
        model=[ModelEvidence(
            assumptions=list(problem.assumptions),
            diagnostics=[diagnostic, "PDE existence and uniqueness are not established",
                         "boundary/initial completeness and well-posedness are not established",
                         "no continuous or discrete solution is claimed"],
            role="diagnostic", trust="unknown",
            metadata={"well_posedness": "not_established"})],
        justified_trust=trust,
    )


def _parameters(problem: PDEProblem) -> dict[sp.Symbol, sp.Expr]:
    return {sp.Symbol(name): value for name, value in problem.parameters if value is not None}


def _classification_data(problem: PDEProblem, equation_index: int,
                         variable_pair: tuple[str, ...] | None):
    if not 0 <= equation_index < len(problem.equations):
        raise ValueError("equation_index is outside the stored PDE system")
    equation = problem.equations[equation_index]
    pair = variable_pair or tuple(problem.independent_variables[:2])
    if any(name not in problem.independent_variables for name in pair) or len(set(pair)) != len(pair):
        raise ValueError("variable_pair must contain distinct independent variables")
    linearity = "linear" if equation.linear else "nonlinear"
    common = dict(problem_id="", equation_index=equation_index,
                  variable_pair=pair, differential_order=equation.order,
                  linearity=linearity, input_trust=cap_trust(problem.input_trust, "symbolic"))
    if not equation.linear:
        return {**common, "classification": "nonlinear"}
    equation_fields = {term.field for term in equation.terms}
    if len(problem.fields) != 1 or len(equation_fields) != 1:
        return {**common, "classification": "system"}
    if equation.order == 0:
        return {**common, "classification": "algebraic"}
    if equation.order == 1:
        return {**common, "classification": "first_order"}
    if equation.order != 2 or len(problem.independent_variables) != 2 or len(pair) != 2:
        return {**common, "classification": "undetermined"}
    indices = [problem.independent_variables.index(name) for name in pair]
    a = sp.S.Zero; b = sp.S.Zero; c = sp.S.Zero
    for term in equation.terms:
        if term.order != 2:
            continue
        derivative = term.derivative
        if any(order for index, order in enumerate(derivative) if index not in indices):
            continue
        left, right = derivative[indices[0]], derivative[indices[1]]
        if (left, right) == (2, 0):
            a += term.coefficient
        elif (left, right) == (1, 1):
            b += term.coefficient
        elif (left, right) == (0, 2):
            c += term.coefficient
    substitutions = _parameters(problem)
    a, b, c = (sp.simplify(value.subs(substitutions)) for value in (a, b, c))
    discriminant = sp.simplify(b ** 2 - 4 * a * c)
    matrix = ((a, sp.simplify(b / 2)), (sp.simplify(b / 2), c))
    if discriminant.is_negative is True:
        classification = "elliptic"
        cases = ()
    elif discriminant == 0 or discriminant.is_zero is True:
        classification = "parabolic"
        cases = ()
    elif discriminant.is_positive is True:
        classification = "hyperbolic"
        cases = ()
    else:
        classification = "conditional"
        text = sp.sstr(discriminant)
        cases = (("elliptic", f"{text} < 0"),
                 ("parabolic", f"{text} = 0"),
                 ("hyperbolic", f"{text} > 0"))
    return {**common, "classification": classification,
            "principal_matrix": matrix, "discriminant": discriminant,
            "classification_cases": cases}


def verify_problem(problem: PDEProblem, problem_id: str) -> EngineeringResult:
    compatibility = _compatibility(problem, problem_id)
    initial_at_start = all(
        sp.simplify(condition.time - dict(
            (name, lower) for name, lower, _ in problem.domain
        )[problem.time_variable]) == 0
        for condition in problem.initial_conditions
    ) if problem.initial_conditions else True
    checks = {
        "field_and_variable_scope": True,
        "rectangular_domain_ordered": True,
        "equation_terms_well_scoped": True,
        "conditions_well_scoped": True,
        "initial_conditions_at_domain_start": initial_at_start,
        "known_condition_conflicts_absent": compatibility.incompatible_count == 0,
    }
    trust = cap_trust(problem.input_trust, "symbolic")
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust, value={"equation_count": len(problem.equations),
                            "maximum_order": max(eq.order for eq in problem.equations),
                            "compatibility": compatibility.compatibility},
        details={"well_posedness": "not_established",
                 "condition_completeness": "not_checked"},
        verification=checks,
        claim_evidence={"verify": _bundle(
            problem, "pde_schema_scope_and_known_compatibility", trust,
            diagnostic="schema verification does not prove PDE solvability")})


def classify(problem: PDEProblem, problem_id: str, equation_index: int = 0,
             variable_pair: tuple[str, ...] | None = None):
    data = _classification_data(problem, equation_index, variable_pair)
    output = PDEClassification(**{**data, "problem_id": problem_id})
    principal_collected = None if output.classification == "undetermined" else True
    checks = {"equation_index_in_range": True,
              "principal_part_collected": principal_collected,
              "classification_is_conditional_when_sign_unknown": not (
                  output.discriminant is not None and
                  output.discriminant.is_positive is None and
                  output.discriminant.is_negative is None and
                  output.discriminant.is_zero is None) or
                  output.classification == "conditional"}
    trust = cap_trust(problem.input_trust, "symbolic")
    return EngineeringResult(
        operation="classify", status=("unknown" if principal_collected is None else "verified"), trust=trust,
        value={"classification": output.classification,
               "differential_order": output.differential_order,
               "linearity": output.linearity,
               "principal_matrix": output.principal_matrix,
               "discriminant": output.discriminant,
               "classification_cases": output.classification_cases},
        details={"classification_scope": "principal_part_only",
                 "global_type": "not_established" if output.classification == "conditional"
                 else output.classification,
                 "well_posedness": "not_established"},
        conditions=[condition for _, condition in output.classification_cases],
        verification=checks,
        claim_evidence={"classify": _bundle(
            problem, "second_order_principal_discriminant", trust,
            diagnostic="classification concerns the represented principal part only")}), output


def verify_classification(problem: PDEProblem, result: PDEClassification):
    candidate = PDEClassification(**{
        **_classification_data(problem, result.equation_index, result.variable_pair),
        "problem_id": result.problem_id})
    checks = {
        "order_recomputed": candidate.differential_order == result.differential_order,
        "linearity_recomputed": candidate.linearity == result.linearity,
        "principal_matrix_recomputed": candidate.principal_matrix == result.principal_matrix,
        "discriminant_recomputed": candidate.discriminant == result.discriminant,
        "classification_recomputed": candidate.classification == result.classification,
        "conditions_recomputed": candidate.classification_cases == result.classification_cases,
    }
    trust = cap_trust(problem.input_trust, result.input_trust)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust, value={"checks": checks}, verification=checks,
        details={"well_posedness": "not_established"},
        claim_evidence={"verify": _bundle(
            problem, "classification_replay", trust,
            diagnostic="classification replay does not establish well-posedness")})


def _status(residual: sp.Expr) -> CompatibilityStatus:
    residual = sp.simplify(residual)
    if residual == 0 or residual.is_zero is True:
        return "compatible"
    if residual.is_zero is False:
        return "incompatible"
    return "undetermined"


def _compatibility(problem: PDEProblem, problem_id: str) -> PDECompatibilityReport:
    bounds = {name: (lower, upper) for name, lower, upper in problem.domain}
    substitutions = _parameters(problem)
    checks: list[PDECompatibilityCheck] = []
    dirichlet = [item for item in problem.boundary_conditions
                 if item.kind == "dirichlet"]
    for index, left in enumerate(dirichlet):
        for right_index, right in enumerate(dirichlet[index + 1:], start=index + 1):
            if left.field != right.field:
                continue
            if left.coordinate == right.coordinate and left.side != right.side:
                continue
            left_bound = bounds[left.coordinate][left.side == "upper"]
            right_bound = bounds[right.coordinate][right.side == "upper"]
            left_value = left.value.subs(substitutions).subs(
                {sp.Symbol(left.coordinate): left_bound,
                 sp.Symbol(right.coordinate): right_bound})
            right_value = right.value.subs(substitutions).subs(
                {sp.Symbol(left.coordinate): left_bound,
                 sp.Symbol(right.coordinate): right_bound})
            residual = sp.simplify(left_value - right_value)
            checks.append(PDECompatibilityCheck(
                left=f"boundary[{index}]", right=f"boundary[{right_index}]",
                status=_status(residual), residual=residual))
    if problem.time_variable is not None:
        time_symbol = sp.Symbol(problem.time_variable)
        spatial_boundaries = list(enumerate(dirichlet))
        for initial_index, initial in enumerate(problem.initial_conditions):
            if initial.derivative_order != 0:
                continue
            for boundary_index, boundary in spatial_boundaries:
                if initial.field != boundary.field:
                    continue
                bound = bounds[boundary.coordinate][boundary.side == "upper"]
                left = initial.value.subs(substitutions).subs(
                    {sp.Symbol(boundary.coordinate): bound})
                right = boundary.value.subs(substitutions).subs(
                    {time_symbol: initial.time,
                     sp.Symbol(boundary.coordinate): bound})
                residual = sp.simplify(left - right)
                checks.append(PDECompatibilityCheck(
                    left=f"initial[{initial_index}]", right=f"boundary[{boundary_index}]",
                    status=_status(residual), residual=residual))
    incompatible = sum(item.status == "incompatible" for item in checks)
    undetermined = sum(item.status == "undetermined" for item in checks)
    overall: CompatibilityStatus = (
        "incompatible" if incompatible else
        "undetermined" if undetermined else "compatible")
    return PDECompatibilityReport(
        problem_id=problem_id, compatibility=overall, checks=tuple(checks),
        incompatible_count=incompatible, undetermined_count=undetermined,
        input_trust=cap_trust(problem.input_trust, "symbolic"))


def boundary_compatibility(problem: PDEProblem, problem_id: str):
    output = _compatibility(problem, problem_id)
    checks = {"no_known_conflict": output.incompatible_count == 0,
              "all_comparisons_decidable": output.undetermined_count == 0}
    trust = cap_trust(problem.input_trust, "symbolic")
    return EngineeringResult(
        operation="boundary_compatibility",
        status="refuted" if output.compatibility == "incompatible" else
               "unknown" if output.compatibility == "undetermined" else "verified",
        trust=trust, value={"compatibility": output.compatibility,
                            "checks": output.checks,
                            "incompatible_count": output.incompatible_count,
                            "undetermined_count": output.undetermined_count},
        details={"completeness_checked": False,
                 "well_posedness": "not_established"},
        verification=checks,
        claim_evidence={"boundary_compatibility": _bundle(
            problem, "dirichlet_trace_pairwise_compatibility", trust,
            diagnostic="only represented Dirichlet trace intersections are compared")}), output


def verify_compatibility(problem: PDEProblem, result: PDECompatibilityReport):
    candidate = _compatibility(problem, result.problem_id)
    checks = {"status_recomputed": candidate.compatibility == result.compatibility,
              "comparisons_recomputed": candidate.checks == result.checks,
              "counts_recomputed": (
                  candidate.incompatible_count == result.incompatible_count and
                  candidate.undetermined_count == result.undetermined_count)}
    trust = cap_trust(problem.input_trust, result.input_trust)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust, value={"checks": checks}, verification=checks,
        details={"well_posedness": "not_established",
                 "completeness_checked": False},
        claim_evidence={"verify": _bundle(
            problem, "compatibility_replay", trust,
            diagnostic="replay verifies only the stored compatibility scope")})
