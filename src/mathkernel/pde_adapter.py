# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted typed adapter for Phase G partial differential equations."""
from __future__ import annotations

from .engineering import arithmetic_trust, cap_trust
from .models import TrustLevel


TYPES = {
    "pdeproblem": "PDEProblem",
    "pde_problem": "PDEProblem",
}

OPERATIONS = {
    "PDEProblem": {
        "verify": {},
        "classify": {
            "equation_index": "nonnegative integer (default 0)",
            "variable_pair": "two distinct independent-variable names?",
        },
        "boundary_compatibility": {},
        "derive_weak_form": {
            "equation_index": "nonnegative integer (default 0)",
            "integration_variables": "nonempty independent-variable names",
            "trial_spaces": "ordered trial-space specifications",
            "test_space": "test-space specification",
            "integration_by_parts": "term-index/coordinate selections (default empty)",
        },
    },
    "PDEClassification": {"verify": {}},
    "PDECompatibilityReport": {"verify": {}},
    "WeakForm": {"verify": {}},
}

DERIVED_OUTPUTS = {
    "classify": "PDEClassification",
    "boundary_compatibility": "PDECompatibilityReport",
    "derive_weak_form": "WeakForm",
}

from .fem_adapter import (DERIVED_OUTPUTS as FEM_DERIVED_OUTPUTS,
                          OPERATIONS as FEM_OPERATIONS, TYPES as FEM_TYPES)
TYPES.update(FEM_TYPES)
OPERATIONS.update(FEM_OPERATIONS)
DERIVED_OUTPUTS.update(FEM_DERIVED_OUTPUTS)


def _space(raw, *, role, variables, settings):
    from .weak_forms import PDEFunctionSpace

    if not isinstance(raw, dict):
        raise ValueError(f"{role}_space must be an object")
    allowed = {"name", "field", "family", "regularity_order",
               "trace_boundary_indices", "assumptions"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown function-space field: {sorted(unknown)[0]}")
    if any(not isinstance(raw.get(name), str) for name in ("name", "field", "family")):
        raise ValueError("function-space name, field, and family must be strings")
    regularity = raw.get("regularity_order")
    if (isinstance(regularity, bool) or not isinstance(regularity, int) or
            not 0 <= regularity <= settings.max_pde_space_order):
        raise ValueError("function-space regularity exceeds max_pde_space_order")
    traces = raw.get("trace_boundary_indices", ())
    assumptions = raw.get("assumptions", ())
    if (not isinstance(traces, (list, tuple)) or
            any(isinstance(index, bool) or not isinstance(index, int) for index in traces)):
        raise ValueError("trace_boundary_indices must be integers")
    if (not isinstance(assumptions, (list, tuple)) or
            any(not isinstance(item, str) for item in assumptions)):
        raise ValueError("function-space assumptions must be strings")
    if sum(len(item) for item in [raw["name"], raw["field"], raw["family"], *assumptions]) > settings.max_input_length:
        raise ValueError("function-space metadata exceeds max_input_length")
    return PDEFunctionSpace(
        name=raw["name"], role=role, field=raw["field"], family=raw["family"],
        variables=variables, regularity_order=regularity,
        trace_boundary_indices=tuple(traces), assumptions=tuple(assumptions))


def _weak_parameters(problem, parameters, settings):
    equation_index = parameters.get("equation_index", 0)
    if isinstance(equation_index, bool) or not isinstance(equation_index, int):
        raise ValueError("equation_index must be an integer")
    raw_variables = parameters.get("integration_variables")
    if (not isinstance(raw_variables, (list, tuple)) or not raw_variables or
            len(raw_variables) > settings.max_pde_dimensions or
            any(not isinstance(name, str) for name in raw_variables)):
        raise ValueError("integration_variables must be a bounded nonempty name sequence")
    variables = tuple(raw_variables)
    raw_trials = parameters.get("trial_spaces")
    if (not isinstance(raw_trials, (list, tuple)) or not raw_trials or
            len(raw_trials) > settings.max_pde_spaces):
        raise ValueError("trial_spaces exceed max_pde_spaces or are empty")
    trials = tuple(_space(item, role="trial", variables=variables, settings=settings)
                   for item in raw_trials)
    test = _space(parameters.get("test_space"), role="test",
                  variables=variables, settings=settings)
    raw_steps = parameters.get("integration_by_parts", ())
    if (not isinstance(raw_steps, (list, tuple)) or
            len(raw_steps) > settings.max_pde_ibp_steps):
        raise ValueError("integration_by_parts exceeds max_pde_ibp_steps")
    selections = []
    for item in raw_steps:
        if not isinstance(item, dict) or set(item) != {"term_index", "coordinate"}:
            raise ValueError("each integration-by-parts selection needs term_index and coordinate")
        index, coordinate = item["term_index"], item["coordinate"]
        if (isinstance(index, bool) or not isinstance(index, int) or
                not isinstance(coordinate, str)):
            raise ValueError("integration-by-parts selection types are invalid")
        selections.append((index, coordinate))
    term_count = len(problem.equations[equation_index].terms) if 0 <= equation_index < len(problem.equations) else 0
    projected_terms = term_count + 1 + 4 * len(selections)
    if projected_terms > settings.max_pde_weak_terms:
        raise ValueError("derived weak terms exceed max_pde_weak_terms")
    work = projected_terms * (len(problem.independent_variables) + len(problem.boundary_conditions) + 1)
    if work > settings.max_pde_weak_work:
        raise ValueError("weak-form derivation exceeds max_pde_weak_work")
    return equation_index, variables, trials, test, tuple(selections)


def construct(kernel, kind, definition):
    if kind in FEM_TYPES:
        from .fem_adapter import construct as construct_fem
        return construct_fem(kernel, kind, definition)
    from .partial_differential_equations import (
        PDEBoundaryCondition, PDEEquation, PDEInitialCondition, PDEProblem,
        PDETerm,
    )

    data = dict(definition)
    if data.pop("context_id", None) is not None:
        raise ValueError("PDEProblem declares its own symbol scope")
    allowed = {"fields", "independent_variables", "time_variable", "domain",
               "equations", "boundary_conditions", "initial_conditions",
               "parameters", "assumptions"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown PDEProblem field: {sorted(unknown)[0]}")
    settings = kernel.settings
    fields = data.get("fields")
    variables = data.get("independent_variables")
    if (not isinstance(fields, (list, tuple)) or
            not 1 <= len(fields) <= settings.max_pde_fields or
            not isinstance(variables, (list, tuple)) or
            not 1 <= len(variables) <= settings.max_pde_dimensions or
            any(not isinstance(name, str) for name in [*fields, *variables])):
        raise ValueError("PDE fields or independent variables exceed configured limits")
    domain = data.get("domain")
    if not isinstance(domain, dict) or set(domain) != set(variables):
        raise ValueError("domain must map every independent variable to [lower, upper]")
    equations = data.get("equations")
    if (not isinstance(equations, (list, tuple)) or
            not 1 <= len(equations) <= settings.max_pde_equations):
        raise ValueError("equations exceed max_pde_equations or are empty")
    boundaries = data.get("boundary_conditions", ())
    initials = data.get("initial_conditions", ())
    if (not isinstance(boundaries, (list, tuple)) or
            not isinstance(initials, (list, tuple)) or
            len(boundaries) + len(initials) > settings.max_pde_conditions):
        raise ValueError("PDE conditions exceed max_pde_conditions")
    parameters = data.get("parameters", {})
    if not isinstance(parameters, dict) or any(not isinstance(name, str) for name in parameters):
        raise ValueError("PDE parameters must map names to MathIR scalars or null")
    assumptions = data.get("assumptions", ())
    if not isinstance(assumptions, (list, tuple)) or any(
            not isinstance(item, str) for item in assumptions):
        raise ValueError("PDE assumptions must be strings")
    metadata = [*fields, *variables, *parameters, *assumptions]
    if data.get("time_variable") is not None:
        metadata.append(data["time_variable"])
    if any(not isinstance(item, str) for item in metadata) or sum(
            len(item) for item in metadata) > settings.max_input_length:
        raise ValueError("PDE names and assumptions exceed max_input_length")

    raw: list[str] = []
    def scalar(item, name):
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(f"{name} must be a MathIR scalar")
        raw.append(str(item))

    for name in variables:
        interval = domain[name]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each PDE domain entry must be [lower, upper]")
        scalar(interval[0], "domain bound"); scalar(interval[1], "domain bound")
    concrete_parameters = []
    for name, value in parameters.items():
        if value is not None:
            scalar(value, "parameter value")
            concrete_parameters.append(name)

    normalized_equations = []
    term_count = 0
    for equation in equations:
        if not isinstance(equation, dict):
            raise ValueError("each PDE equation must be an object")
        unknown = set(equation) - {"terms", "source", "label"}
        if unknown:
            raise ValueError(f"unknown PDE equation field: {sorted(unknown)[0]}")
        terms = equation.get("terms")
        if not isinstance(terms, (list, tuple)) or not terms:
            raise ValueError("each PDE equation requires nonempty terms")
        term_count += len(terms)
        if term_count > settings.max_pde_terms:
            raise ValueError("PDE terms exceed max_pde_terms")
        normalized_terms = []
        for term in terms:
            if not isinstance(term, dict):
                raise ValueError("each PDE term must be an object")
            unknown = set(term) - {"coefficient", "field", "derivative", "power"}
            if unknown:
                raise ValueError(f"unknown PDE term field: {sorted(unknown)[0]}")
            derivative = term.get("derivative")
            if isinstance(derivative, dict):
                if set(derivative) - set(variables):
                    raise ValueError("PDE derivative names must be independent variables")
                orders = tuple(derivative.get(name, 0) for name in variables)
            elif isinstance(derivative, (list, tuple)) and len(derivative) == len(variables):
                orders = tuple(derivative)
            else:
                raise ValueError("PDE derivative must be a variable-order map or aligned sequence")
            if any(isinstance(order, bool) or not isinstance(order, int) or order < 0
                   for order in orders) or sum(orders) > settings.max_pde_derivative_order:
                raise ValueError("PDE derivative exceeds max_pde_derivative_order")
            power = term.get("power", 1)
            if (isinstance(power, bool) or not isinstance(power, int) or
                    not 1 <= power <= settings.max_pde_nonlinear_power):
                raise ValueError("PDE term power exceeds max_pde_nonlinear_power")
            field = term.get("field")
            if not isinstance(field, str):
                raise ValueError("PDE term field must be a string")
            scalar(term.get("coefficient", 1), "PDE coefficient")
            normalized_terms.append((field, orders, power))
        scalar(equation.get("source", 0), "PDE source")
        label = equation.get("label")
        if label is not None and not isinstance(label, str):
            raise ValueError("PDE equation label must be a string")
        normalized_equations.append((normalized_terms, label))

    normalized_boundaries = []
    for condition in boundaries:
        if not isinstance(condition, dict):
            raise ValueError("each boundary condition must be an object")
        unknown = set(condition) - {"field", "kind", "coordinate", "side", "value",
                                    "alpha", "beta", "paired_side"}
        if unknown:
            raise ValueError(f"unknown boundary-condition field: {sorted(unknown)[0]}")
        for name in ("field", "kind", "coordinate", "side"):
            if not isinstance(condition.get(name), str):
                raise ValueError(f"boundary {name} must be a string")
        scalar(condition.get("value", 0), "boundary value")
        scalar(condition.get("alpha", 1), "boundary alpha")
        scalar(condition.get("beta", 0), "boundary beta")
        normalized_boundaries.append(condition)

    normalized_initials = []
    time_variable = data.get("time_variable")
    default_time = domain.get(time_variable, (None,))[0] if time_variable in domain else None
    for condition in initials:
        if not isinstance(condition, dict):
            raise ValueError("each initial condition must be an object")
        unknown = set(condition) - {"field", "derivative_order", "time", "value"}
        if unknown:
            raise ValueError(f"unknown initial-condition field: {sorted(unknown)[0]}")
        if not isinstance(condition.get("field"), str):
            raise ValueError("initial-condition field must be a string")
        derivative_order = condition.get("derivative_order", 0)
        if isinstance(derivative_order, bool) or not isinstance(derivative_order, int):
            raise ValueError("initial derivative_order must be an integer")
        initial_time = condition.get("time", default_time)
        if initial_time is None:
            raise ValueError("initial condition requires a time")
        scalar(initial_time, "initial time"); scalar(condition.get("value"), "initial value")
        normalized_initials.append((condition, derivative_order))

    work = term_count * len(variables) + len(boundaries) ** 2 + len(initials) * len(boundaries)
    if work > settings.max_pde_work:
        raise ValueError("PDE representation exceeds max_pde_work")
    if sum(len(item) for item in raw) > settings.max_input_length:
        raise ValueError("PDE mathematical input exceeds max_input_length")
    _, parsed, parsed_trust = kernel._typed_parse_many(raw, None)
    iterator = iter(parsed)
    parsed_domain = tuple((name, next(iterator), next(iterator)) for name in variables)
    parsed_parameters = []
    for name, value in parameters.items():
        parsed_parameters.append((name, next(iterator) if value is not None else None))
    parsed_equations = []
    for normalized_terms, label in normalized_equations:
        terms = tuple(PDETerm(
            coefficient=next(iterator), field=field, derivative=orders, power=power)
            for field, orders, power in normalized_terms)
        parsed_equations.append(PDEEquation(terms=terms, source=next(iterator), label=label))
    parsed_boundaries = tuple(PDEBoundaryCondition(
        field=condition["field"], kind=condition["kind"].lower(),
        coordinate=condition["coordinate"], side=condition["side"].lower(),
        value=next(iterator), alpha=next(iterator), beta=next(iterator),
        paired_side=(condition.get("paired_side") or None))
        for condition in normalized_boundaries)
    parsed_initials = tuple(PDEInitialCondition(
        field=condition["field"], derivative_order=derivative_order,
        time=next(iterator), value=next(iterator))
        for condition, derivative_order in normalized_initials)
    trust = TrustLevel(cap_trust(
        parsed_trust.value, arithmetic_trust(parsed, parsed_trust.value)))
    value = PDEProblem(
        fields=tuple(fields), independent_variables=tuple(variables),
        time_variable=time_variable, domain=parsed_domain,
        equations=tuple(parsed_equations), boundary_conditions=parsed_boundaries,
        initial_conditions=parsed_initials, parameters=tuple(parsed_parameters),
        assumptions=tuple(assumptions), input_trust=trust.value)
    return "PDEProblem", value, trust, []


def apply(kernel, value, operation, parameters, object_id):
    if type(value).__name__ in FEM_OPERATIONS:
        from .fem_adapter import apply as apply_fem
        return apply_fem(kernel, value, operation, parameters, object_id)
    from . import partial_differential_equations as pde

    object_type = type(value).__name__
    allowed = set(OPERATIONS[object_type][operation]) | {"context_id"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unsupported operation parameter: {sorted(unknown)[0]}")
    if parameters.get("context_id") is not None:
        raise ValueError("PDE operations use the problem's declared symbol scope")

    record = kernel._get_math_object_record(object_id)
    term_count = 0
    if object_type == "PDEProblem":
        problem = value
        term_count = sum(len(equation.terms) for equation in problem.equations)
    else:
        source_id = record["sources"][0] if record and record.get("sources") else None
        source = kernel._get_math_object_record(source_id) if source_id else None
        if source is None or source["object_type"] != "PDEProblem":
            raise ValueError("PDE derived-result source is unavailable")
        problem = source["value"]
        term_count = sum(len(equation.terms) for equation in problem.equations)
        if value.problem_id != source_id:
            raise ValueError("PDE derived-result ancestry does not match problem_id")
    condition_count = len(problem.boundary_conditions) + len(problem.initial_conditions)
    work = term_count * len(problem.independent_variables) + condition_count ** 2
    if work > kernel.settings.max_pde_work:
        raise ValueError("PDE operation exceeds current max_pde_work")

    if object_type == "PDEProblem":
        if operation == "verify":
            return pde.verify_problem(problem, object_id)
        if operation == "boundary_compatibility":
            return pde.boundary_compatibility(problem, object_id)
        if operation == "derive_weak_form":
            from . import weak_forms
            args = _weak_parameters(problem, parameters, kernel.settings)
            return weak_forms.derive_weak_form(problem, object_id, *args)
        index = parameters.get("equation_index", 0)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("equation_index must be an integer")
        raw_pair = parameters.get("variable_pair")
        if raw_pair is not None and (
                not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2 or
                any(not isinstance(name, str) for name in raw_pair)):
            raise ValueError("variable_pair must contain two variable names")
        return pde.classify(problem, object_id, index,
                            tuple(raw_pair) if raw_pair is not None else None)
    if object_type == "PDEClassification":
        return pde.verify_classification(problem, value)
    if object_type == "PDECompatibilityReport":
        return pde.verify_compatibility(problem, value)
    from . import weak_forms
    projected = (len(value.volume_terms) + len(value.boundary_terms)) * (
        len(problem.independent_variables) + len(problem.boundary_conditions) + 1)
    if (len(value.trial_spaces) + 1 > kernel.settings.max_pde_spaces or
            len(value.volume_terms) + len(value.boundary_terms) > kernel.settings.max_pde_weak_terms or
            len(value.derivation_steps) > kernel.settings.max_pde_ibp_steps or
            projected > kernel.settings.max_pde_weak_work):
        raise ValueError("weak-form replay exceeds current max_pde_weak_work")
    return weak_forms.verify_weak_form(problem, value)
