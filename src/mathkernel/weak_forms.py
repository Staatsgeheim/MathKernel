# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed Phase G.2 weak forms and replayable integration by parts."""
from __future__ import annotations

from typing import Literal

import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import ComputationEvidence, EvidenceBundle, ModelEvidence

from .engineering import EngineeringModel, EngineeringResult, cap_trust
from .partial_differential_equations import PDEProblem, _expression, _parameters


SpaceFamily = Literal["L2", "H1", "H1_D", "H1_0", "H2", "custom"]


class PDEFunctionSpace(EngineeringModel):
    name: str
    role: Literal["trial", "test"]
    field: str
    family: SpaceFamily
    variables: tuple[str, ...]
    regularity_order: int
    trace_boundary_indices: tuple[int, ...] = ()
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_space(self):
        if not self.name.isidentifier() or not self.field.isidentifier():
            raise ValueError("function-space name and field must be identifiers")
        if (not self.variables or len(set(self.variables)) != len(self.variables) or
                any(not name.isidentifier() for name in self.variables)):
            raise ValueError("function-space variables must be unique identifiers")
        if (isinstance(self.regularity_order, bool) or
                not isinstance(self.regularity_order, int) or
                self.regularity_order < 0):
            raise ValueError("function-space regularity_order must be nonnegative")
        minimum = {"L2": 0, "H1": 1, "H1_D": 1, "H1_0": 1, "H2": 2}.get(self.family, 0)
        if self.regularity_order < minimum:
            raise ValueError(f"{self.family} requires regularity_order >= {minimum}")
        if (len(set(self.trace_boundary_indices)) != len(self.trace_boundary_indices) or
                any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                    for index in self.trace_boundary_indices)):
            raise ValueError("trace boundary indices must be unique nonnegative integers")
        if any(not item.strip() for item in self.assumptions):
            raise ValueError("function-space assumptions must be nonempty")
        return self


class PDEMeasure(EngineeringModel):
    kind: Literal["volume", "boundary"]
    variables: tuple[str, ...]
    coordinate: str | None = None
    side: Literal["lower", "upper"] | None = None
    outward_orientation: Literal[-1, 1] | None = None

    @model_validator(mode="after")
    def validate_measure(self):
        if self.kind == "volume":
            if self.coordinate is not None or self.side is not None or self.outward_orientation is not None:
                raise ValueError("volume measures cannot carry boundary metadata")
        else:
            if (self.coordinate is None or self.side is None or
                    self.outward_orientation not in {-1, 1}):
                raise ValueError("boundary measures require coordinate, side and orientation")
            expected = -1 if self.side == "lower" else 1
            if self.outward_orientation != expected:
                raise ValueError("boundary orientation does not match rectangular-domain side")
        return self


class WeakIntegralTerm(EngineeringModel):
    side: Literal["lhs", "rhs"]
    coefficient: sp.Expr
    field: str | None
    field_derivative: tuple[int, ...]
    field_power: int
    test_function: str
    test_derivative: tuple[int, ...]
    measure: PDEMeasure
    origin: Literal[
        "equation_term", "equation_source", "integration_by_parts",
        "coefficient_derivative", "boundary_trace",
    ]
    origin_index: int | None = None
    boundary_condition_index: int | None = None
    vanishes_by_trace: bool = False

    @model_validator(mode="after")
    def validate_term(self):
        _expression(self.coefficient, "weak-form coefficient")
        if self.field is not None and not self.field.isidentifier():
            raise ValueError("weak-form field must be an identifier")
        if not self.test_function.isidentifier():
            raise ValueError("weak-form test function must be an identifier")
        if (len(self.field_derivative) != len(self.test_derivative) or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (*self.field_derivative, *self.test_derivative))):
            raise ValueError("weak-form derivative multi-indices must align and be nonnegative")
        if self.field is None and (any(self.field_derivative) or self.field_power != 0):
            raise ValueError("source weak terms cannot carry a field factor")
        if (isinstance(self.field_power, bool) or not isinstance(self.field_power, int) or
                (self.field is not None and self.field_power < 1)):
            raise ValueError("field weak terms require a positive power")
        if self.origin_index is not None and self.origin_index < 0:
            raise ValueError("weak-form origin_index must be nonnegative")
        if self.boundary_condition_index is not None and self.boundary_condition_index < 0:
            raise ValueError("boundary-condition index must be nonnegative")
        if self.vanishes_by_trace and self.measure.kind != "boundary":
            raise ValueError("only boundary terms can vanish by a declared trace")
        return self


class IntegrationByPartsStep(EngineeringModel):
    equation_term_index: int
    coordinate: str
    derivative_before: tuple[int, ...]
    derivative_after: tuple[int, ...]
    test_derivative_after: tuple[int, ...]
    coefficient_before: sp.Expr
    coefficient_derivative: sp.Expr
    generated_volume_term_indices: tuple[int, ...]
    generated_boundary_term_indices: tuple[int, int]
    identity: str = "integral c*D_i(w)*v = boundary n_i*c*w*v - integral c*w*D_i(v) - integral D_i(c)*w*v"

    @model_validator(mode="after")
    def validate_step(self):
        if self.equation_term_index < 0 or not self.coordinate.isidentifier():
            raise ValueError("integration-by-parts provenance is invalid")
        _expression(self.coefficient_before, "integration coefficient")
        _expression(self.coefficient_derivative, "coefficient derivative")
        if (len(self.derivative_before) != len(self.derivative_after) or
                len(self.derivative_before) != len(self.test_derivative_after)):
            raise ValueError("integration-by-parts derivative dimensions must agree")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (*self.derivative_before, *self.derivative_after,
                             *self.test_derivative_after)):
            raise ValueError("integration-by-parts derivatives must be nonnegative integers")
        if len(self.generated_boundary_term_indices) != 2:
            raise ValueError("rectangular integration by parts emits two oriented faces")
        return self


class WeakForm(EngineeringModel):
    problem_id: str
    equation_index: int
    integration_variables: tuple[str, ...]
    trial_spaces: tuple[PDEFunctionSpace, ...]
    test_space: PDEFunctionSpace
    volume_measure: PDEMeasure
    volume_terms: tuple[WeakIntegralTerm, ...]
    boundary_terms: tuple[WeakIntegralTerm, ...]
    derivation_steps: tuple[IntegrationByPartsStep, ...]
    essential_boundary_indices: tuple[int, ...]
    natural_boundary_indices: tuple[int, ...]
    periodic_boundary_indices: tuple[int, ...]
    conditions: tuple[str, ...]
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_weak_form(self):
        if not self.problem_id or self.equation_index < 0:
            raise ValueError("weak form requires PDE source provenance")
        if self.volume_measure.kind != "volume":
            raise ValueError("weak form volume_measure must be a volume measure")
        if self.volume_measure.variables != self.integration_variables:
            raise ValueError("volume measure must match integration variables")
        if self.test_space.role != "test" or any(space.role != "trial" for space in self.trial_spaces):
            raise ValueError("weak-form space roles are inconsistent")
        for indices in (self.essential_boundary_indices,
                        self.natural_boundary_indices,
                        self.periodic_boundary_indices):
            if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
                raise ValueError("weak-form boundary partitions require unique indices")
        sets = [set(self.essential_boundary_indices), set(self.natural_boundary_indices),
                set(self.periodic_boundary_indices)]
        if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("weak-form boundary partitions must be disjoint")
        return self


def _bundle(problem: PDEProblem, method: str, trust: str, conditions: tuple[str, ...]) -> EvidenceBundle:
    return EvidenceBundle(
        computation=[ComputationEvidence(
            engine="weak_forms", method=method, arithmetic=trust,
            deterministic=True, trust=trust)],
        model=[ModelEvidence(
            assumptions=[*problem.assumptions, *conditions], role="diagnostic",
            trust="unknown", diagnostics=[
                "weak identity is conditional on the declared space, trace, and regularity assumptions",
                "PDE existence, uniqueness, regularity, and well-posedness are not established",
                "no mesh, discretization, assembly, continuous solution, or discrete solution is claimed",
            ], metadata={"well_posedness": "not_established",
                         "discretization": "not_performed"})],
        justified_trust=trust,
    )


def _spatial_order(derivative: tuple[int, ...], problem: PDEProblem,
                   variables: tuple[str, ...]) -> int:
    return sum(derivative[problem.independent_variables.index(name)] for name in variables)


def _check_spaces(problem: PDEProblem, equation_index: int,
                  integration_variables: tuple[str, ...],
                  trial_spaces: tuple[PDEFunctionSpace, ...],
                  test_space: PDEFunctionSpace,
                  selections: tuple[tuple[int, str], ...]) -> None:
    if not 0 <= equation_index < len(problem.equations):
        raise ValueError("equation_index is outside the PDE system")
    if (not integration_variables or len(set(integration_variables)) != len(integration_variables) or
            any(name not in problem.independent_variables for name in integration_variables)):
        raise ValueError("integration_variables must be distinct independent variables")
    equation = problem.equations[equation_index]
    fields = tuple(dict.fromkeys(term.field for term in equation.terms))
    if tuple(space.field for space in trial_spaces) != fields:
        raise ValueError("trial_spaces must follow the equation's first field occurrence")
    spaces = {space.field: space for space in trial_spaces}
    if (any(space.variables != integration_variables for space in trial_spaces) or
            test_space.variables != integration_variables or
            test_space.field not in fields):
        raise ValueError("all spaces must use the integration variables and equation fields")
    if len(set(index for index, _ in selections)) != len(selections):
        raise ValueError("G.2 permits at most one integration-by-parts step per equation term")
    selected = dict(selections)
    for term_index, coordinate in selections:
        if not 0 <= term_index < len(equation.terms) or coordinate not in integration_variables:
            raise ValueError("integration-by-parts selection is outside the equation or measure")
        term = equation.terms[term_index]
        axis = problem.independent_variables.index(coordinate)
        if term.derivative[axis] < 1 or term.power != 1:
            raise ValueError("integration by parts requires a linear differentiated field term")
    for term_index, term in enumerate(equation.terms):
        required = _spatial_order(term.derivative, problem, integration_variables)
        if term_index in selected:
            required -= 1
        if spaces[term.field].regularity_order < required:
            raise ValueError("trial-space regularity is insufficient for a represented volume term")
    if selections and test_space.regularity_order < 1:
        raise ValueError("integration by parts requires a first-order test space")
    for space in (*trial_spaces, test_space):
        for boundary_index in space.trace_boundary_indices:
            if not 0 <= boundary_index < len(problem.boundary_conditions):
                raise ValueError("space trace references an unavailable boundary condition")
            condition = problem.boundary_conditions[boundary_index]
            if (condition.coordinate not in integration_variables or
                    condition.field != space.field or condition.kind != "dirichlet"):
                raise ValueError("space zero trace must reference a Dirichlet condition for its field")
        required_traces = {
            index for index, condition in enumerate(problem.boundary_conditions)
            if condition.field == space.field and condition.kind == "dirichlet" and
            condition.coordinate in integration_variables
        }
        if set(space.trace_boundary_indices) != required_traces:
            raise ValueError("trial prescribed traces and test zero traces must cover every essential boundary")


def _boundary_partition_for_fields(problem: PDEProblem,
                                   integration_variables: tuple[str, ...],
                                   fields: set[str]):
    eligible = [(index, condition) for index, condition in enumerate(problem.boundary_conditions)
                if condition.coordinate in integration_variables and condition.field in fields]
    essential = tuple(index for index, item in eligible if item.kind == "dirichlet")
    natural = tuple(index for index, item in eligible if item.kind in {"neumann", "robin"})
    periodic = tuple(index for index, item in eligible if item.kind == "periodic")
    return essential, natural, periodic


def _derive_data(problem: PDEProblem, problem_id: str, equation_index: int,
                 integration_variables: tuple[str, ...],
                 trial_spaces: tuple[PDEFunctionSpace, ...],
                 test_space: PDEFunctionSpace,
                 selections: tuple[tuple[int, str], ...]) -> WeakForm:
    _check_spaces(problem, equation_index, integration_variables,
                  trial_spaces, test_space, selections)
    equation = problem.equations[equation_index]
    dimensions = len(problem.independent_variables)
    zero = (0,) * dimensions
    measure = PDEMeasure(kind="volume", variables=integration_variables)
    volume: list[WeakIntegralTerm] = []
    boundary: list[WeakIntegralTerm] = []
    steps: list[IntegrationByPartsStep] = []
    selected = dict(selections)
    substitutions = _parameters(problem)
    trace_indices = set(test_space.trace_boundary_indices)
    for term_index, term in enumerate(equation.terms):
        coefficient = sp.simplify(term.coefficient.subs(substitutions))
        if term_index not in selected:
            volume.append(WeakIntegralTerm(
                side="lhs", coefficient=coefficient, field=term.field,
                field_derivative=term.derivative, field_power=term.power,
                test_function=test_space.name, test_derivative=zero,
                measure=measure, origin="equation_term", origin_index=term_index))
            continue
        coordinate = selected[term_index]
        axis = problem.independent_variables.index(coordinate)
        after = list(term.derivative); after[axis] -= 1; after = tuple(after)
        test_derivative = tuple(1 if index == axis else 0 for index in range(dimensions))
        generated_volume = []
        volume.append(WeakIntegralTerm(
            side="lhs", coefficient=sp.simplify(-coefficient), field=term.field,
            field_derivative=after, field_power=1,
            test_function=test_space.name, test_derivative=test_derivative,
            measure=measure, origin="integration_by_parts", origin_index=term_index))
        generated_volume.append(len(volume) - 1)
        coefficient_derivative = sp.simplify(sp.diff(coefficient, sp.Symbol(coordinate)))
        if coefficient_derivative != 0:
            volume.append(WeakIntegralTerm(
                side="lhs", coefficient=sp.simplify(-coefficient_derivative),
                field=term.field, field_derivative=after, field_power=1,
                test_function=test_space.name, test_derivative=zero,
                measure=measure, origin="coefficient_derivative", origin_index=term_index))
            generated_volume.append(len(volume) - 1)
        boundary_start = len(boundary)
        bounds = {name: (lower, upper) for name, lower, upper in problem.domain}
        for side, orientation in (("lower", -1), ("upper", 1)):
            condition_index = next((index for index, condition in enumerate(problem.boundary_conditions)
                                    if condition.field == term.field and
                                    condition.coordinate == coordinate and condition.side == side), None)
            face_coefficient = sp.simplify(coefficient.subs(
                {sp.Symbol(coordinate): bounds[coordinate][side == "upper"]}) * orientation)
            boundary.append(WeakIntegralTerm(
                side="lhs", coefficient=face_coefficient, field=term.field,
                field_derivative=after, field_power=1,
                test_function=test_space.name, test_derivative=zero,
                measure=PDEMeasure(
                    kind="boundary", variables=tuple(
                        name for name in integration_variables if name != coordinate),
                    coordinate=coordinate, side=side,
                    outward_orientation=orientation),
                origin="boundary_trace", origin_index=term_index,
                boundary_condition_index=condition_index,
                vanishes_by_trace=(condition_index in trace_indices)))
        steps.append(IntegrationByPartsStep(
            equation_term_index=term_index, coordinate=coordinate,
            derivative_before=term.derivative, derivative_after=after,
            test_derivative_after=test_derivative,
            coefficient_before=coefficient,
            coefficient_derivative=coefficient_derivative,
            generated_volume_term_indices=tuple(generated_volume),
            generated_boundary_term_indices=(boundary_start, boundary_start + 1)))
    source = sp.simplify(equation.source.subs(substitutions))
    volume.append(WeakIntegralTerm(
        side="rhs", coefficient=source, field=None,
        field_derivative=zero, field_power=0,
        test_function=test_space.name, test_derivative=zero,
        measure=measure, origin="equation_source"))
    essential, natural, periodic = _boundary_partition_for_fields(
        problem, integration_variables, {term.field for term in equation.terms})
    conditions = tuple(dict.fromkeys([
        *(item for space in (*trial_spaces, test_space) for item in space.assumptions),
        "declared Sobolev regularity and traces exist on the rectangular domain",
        "integration-by-parts identities hold for the represented coefficients and fields",
    ]))
    return WeakForm(
        problem_id=problem_id, equation_index=equation_index,
        integration_variables=integration_variables, trial_spaces=trial_spaces,
        test_space=test_space, volume_measure=measure,
        volume_terms=tuple(volume), boundary_terms=tuple(boundary),
        derivation_steps=tuple(steps), essential_boundary_indices=essential,
        natural_boundary_indices=natural, periodic_boundary_indices=periodic,
        conditions=conditions, input_trust=cap_trust(problem.input_trust, "symbolic"))


def derive_weak_form(problem: PDEProblem, problem_id: str, equation_index: int,
                     integration_variables: tuple[str, ...],
                     trial_spaces: tuple[PDEFunctionSpace, ...],
                     test_space: PDEFunctionSpace,
                     selections: tuple[tuple[int, str], ...]):
    output = _derive_data(problem, problem_id, equation_index, integration_variables,
                          trial_spaces, test_space, selections)
    checks = {
        "spaces_match_equation_and_measure": True,
        "selected_derivatives_transfer_once": True,
        "coefficient_product_rule_retained": True,
        "both_oriented_boundary_faces_retained": all(
            len(step.generated_boundary_term_indices) == 2 for step in output.derivation_steps),
        "boundary_partition_explicit": True,
    }
    trust = output.input_trust
    return EngineeringResult(
        operation="derive_weak_form", status="verified", trust=trust,
        value={"equation_index": equation_index,
               "volume_term_count": len(output.volume_terms),
               "boundary_term_count": len(output.boundary_terms),
               "integration_by_parts_steps": len(output.derivation_steps),
               "essential_boundary_indices": output.essential_boundary_indices,
               "natural_boundary_indices": output.natural_boundary_indices,
               "periodic_boundary_indices": output.periodic_boundary_indices},
        details={"identity_scope": "represented_integral_identity",
                 "well_posedness": "not_established",
                 "discretization": "not_performed",
                 "solution": "not_claimed"},
        conditions=list(output.conditions), verification=checks,
        claim_evidence={"derive_weak_form": _bundle(
            problem, "explicit_product_rule_integration_by_parts", trust,
            output.conditions)}), output


def verify_weak_form(problem: PDEProblem, result: WeakForm):
    selections = tuple((step.equation_term_index, step.coordinate)
                       for step in result.derivation_steps)
    candidate = _derive_data(
        problem, result.problem_id, result.equation_index,
        result.integration_variables, result.trial_spaces,
        result.test_space, selections)
    checks = {
        "spaces_recomputed": (
            candidate.trial_spaces == result.trial_spaces and
            candidate.test_space == result.test_space),
        "volume_terms_recomputed": candidate.volume_terms == result.volume_terms,
        "boundary_terms_recomputed": candidate.boundary_terms == result.boundary_terms,
        "derivation_steps_recomputed": candidate.derivation_steps == result.derivation_steps,
        "boundary_partition_recomputed": (
            candidate.essential_boundary_indices == result.essential_boundary_indices and
            candidate.natural_boundary_indices == result.natural_boundary_indices and
            candidate.periodic_boundary_indices == result.periodic_boundary_indices),
        "conditions_recomputed": candidate.conditions == result.conditions,
    }
    trust = cap_trust(problem.input_trust, result.input_trust)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust, value={"checks": checks}, verification=checks,
        details={"identity_scope": "represented_integral_identity",
                 "well_posedness": "not_established",
                 "discretization": "not_performed"},
        conditions=list(result.conditions),
        claim_evidence={"verify": _bundle(
            problem, "weak_form_full_derivation_replay", trust,
            result.conditions)})
