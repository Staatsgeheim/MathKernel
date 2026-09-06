import asyncio

import pytest
import sympy as sp

from mathkernel import (
    MathKernel, PDEClassification, PDECompatibilityReport, PDEProblem, Settings,
)


def problem_definition(*, coefficients=(1, 0, 1), parameter=None):
    a, b, c = coefficients
    parameters = {} if parameter is None else {parameter: None}
    return {
        "fields": ["u"], "independent_variables": ["x", "y"],
        "domain": {"x": [0, 1], "y": [0, 1]}, "parameters": parameters,
        "equations": [{"label": "principal equation", "terms": [
            {"coefficient": a, "field": "u", "derivative": {"x": 2}},
            {"coefficient": b, "field": "u", "derivative": {"x": 1, "y": 1}},
            {"coefficient": c, "field": "u", "derivative": {"y": 2}},
        ]}],
    }


def create(kernel, definition):
    made = kernel.object_create("PDEProblem", definition)
    assert made.ok, made.errors
    return made.data["object_id"]


def classify(kernel, definition):
    object_id = create(kernel, definition)
    result = kernel.apply(object_id, "classify")
    assert result.ok, result.errors
    return object_id, result


@pytest.mark.parametrize(("coefficients", "expected", "discriminant"), [
    ((1, 0, 1), "elliptic", "-4"),
    ((1, 2, 1), "parabolic", "0"),
    ((1, 0, -1), "hyperbolic", "4"),
])
def test_scalar_second_order_classification(coefficients, expected, discriminant):
    kernel = MathKernel(); source_id, made = classify(
        kernel, problem_definition(coefficients=coefficients))
    assert made.status == "verified"
    assert made.data["value"]["classification"] == expected
    assert made.data["value"]["discriminant"] == discriminant
    output = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(output, PDEClassification) and output.problem_id == source_id
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"


def test_symbolic_discriminant_returns_explicit_conditional_cases():
    kernel = MathKernel(); _, made = classify(
        kernel, problem_definition(coefficients=(1, 0, "kappa"), parameter="kappa"))
    assert made.data["value"]["classification"] == "conditional"
    assert made.data["value"]["classification_cases"] == [
        ["elliptic", "-4*kappa < 0"],
        ["parabolic", "-4*kappa = 0"],
        ["hyperbolic", "-4*kappa > 0"],
    ]
    assert made.data["details"]["global_type"] == "not_established"


def test_assigned_parameter_is_substituted_before_classification():
    definition = problem_definition(coefficients=(1, 0, "kappa"), parameter="kappa")
    definition["parameters"]["kappa"] = -2
    _, made = classify(MathKernel(), definition)
    assert made.data["value"]["classification"] == "hyperbolic"
    assert made.data["value"]["discriminant"] == "8"


def test_heat_equation_representation_and_initial_boundary_compatibility():
    kernel = MathKernel(); object_id = create(kernel, {
        "fields": ["u"], "independent_variables": ["x", "t"],
        "time_variable": "t", "domain": {"x": [0, 1], "t": [0, 2]},
        "parameters": {"alpha": 1}, "equations": [{"terms": [
            {"coefficient": 1, "field": "u", "derivative": {"t": 1}},
            {"coefficient": "-alpha", "field": "u", "derivative": {"x": 2}},
        ]}],
        "boundary_conditions": [
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "upper", "value": 0},
        ],
        "initial_conditions": [{"field": "u", "value": "x*(1-x)"}],
        "assumptions": ["alpha is positive"],
    })
    problem = kernel.math_objects[object_id]["value"]
    assert isinstance(problem, PDEProblem) and problem.equations[0].order == 2
    verified = kernel.apply(object_id, "verify")
    typed = kernel.apply(object_id, "classify")
    compatible = kernel.apply(object_id, "boundary_compatibility")
    assert verified.status == "verified"
    assert typed.data["value"]["classification"] == "parabolic"
    assert compatible.status == "verified"
    assert len(compatible.data["value"]["checks"]) == 2
    assert compatible.data["details"]["completeness_checked"] is False
    assert "not established" in str(verified.data["claim_evidence"]).lower()


def test_corner_and_initial_trace_conflicts_are_refuted_but_reported():
    kernel = MathKernel(); definition = problem_definition()
    definition.update({
        "time_variable": "y",
        "boundary_conditions": [
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "upper", "value": 1},
        ],
        "initial_conditions": [{"field": "u", "time": 0, "value": 2}],
    })
    object_id = create(kernel, definition)
    made = kernel.apply(object_id, "boundary_compatibility")
    assert made.ok and made.status == "refuted"
    assert made.data["value"]["compatibility"] == "incompatible"
    assert made.data["value"]["incompatible_count"] == 2
    report = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(report, PDECompatibilityReport)
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"
    assert kernel.apply(object_id, "verify").status == "refuted"


def test_nonlinear_system_first_order_and_unsupported_high_dimensional_type_are_explicit():
    kernel = MathKernel()
    nonlinear = problem_definition(); nonlinear["equations"][0]["terms"][0]["power"] = 2
    assert kernel.apply(create(kernel, nonlinear), "classify").data["value"]["classification"] == "nonlinear"
    first = problem_definition(); first["equations"][0]["terms"] = [
        {"coefficient": 1, "field": "u", "derivative": {"x": 1}}]
    assert kernel.apply(create(kernel, first), "classify").data["value"]["classification"] == "first_order"
    system = problem_definition(); system["fields"] = ["u", "v"]
    assert kernel.apply(create(kernel, system), "classify").data["value"]["classification"] == "system"
    high = problem_definition(); high["independent_variables"] = ["x", "y", "z"]
    high["domain"]["z"] = [0, 1]
    for term in high["equations"][0]["terms"]:
        term["derivative"]["z"] = 0
    outcome = kernel.apply(create(kernel, high), "classify")
    assert outcome.status == "unknown"
    assert outcome.data["value"]["classification"] == "undetermined"


@pytest.mark.parametrize("mutate", [
    lambda d: d["equations"][0]["terms"][0].update(coefficient="rogue"),
    lambda d: d["equations"][0]["terms"][0].update(field="v"),
    lambda d: d["domain"].update(x=[1, 1]),
    lambda d: d["equations"][0]["terms"][0].update(derivative={"x": 5}),
    lambda d: d.update(boundary_conditions=[{
        "field": "u", "kind": "dirichlet", "coordinate": "z",
        "side": "lower", "value": 0}]),
])
def test_invalid_problem_definitions_are_atomic(mutate):
    kernel = MathKernel(); definition = problem_definition(); mutate(definition)
    before = len(kernel.math_objects)
    result = kernel.object_create("PDEProblem", definition)
    assert not result.ok and len(kernel.math_objects) == before


def test_initial_condition_must_be_at_start_to_verify():
    kernel = MathKernel(); definition = problem_definition()
    definition.update({"time_variable": "y", "initial_conditions": [
        {"field": "u", "time": 1, "value": 0}]})
    checked = kernel.apply(create(kernel, definition), "verify")
    assert checked.status == "refuted"
    assert checked.data["verification"]["initial_conditions_at_domain_start"] is False


def test_derived_results_cannot_be_constructed_and_tampering_is_detected():
    kernel = MathKernel()
    assert not kernel.object_create("PDEClassification", {}).ok
    source_id, made = classify(kernel, problem_definition())
    result_id = made.data["object_id"]
    original = kernel.math_objects[result_id]["value"]
    kernel.math_objects[result_id]["value"] = original.model_copy(
        update={"classification": "hyperbolic"})
    assert kernel.apply(result_id, "verify").status == "refuted"
    assert kernel.math_objects[result_id]["sources"] == [source_id]


def test_persistence_lineage_and_replay_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "pde.sqlite"))
    kernel = MathKernel(settings); source_id, made = classify(kernel, problem_definition())
    result_id = made.data["object_id"]
    report_id = kernel.apply(source_id, "boundary_compatibility").data["object_id"]
    restarted = MathKernel(settings)
    assert restarted.object_get(result_id).data["sources"] == [source_id]
    assert restarted.object_get(report_id).data["sources"] == [source_id]
    assert restarted.apply(result_id, "verify").status == "verified"
    assert restarted.apply(report_id, "verify").status == "verified"


def test_decimal_ancestry_and_current_resource_limits_are_preserved(tmp_path):
    path = tmp_path / "limited.sqlite"
    kernel = MathKernel(Settings(store_path=str(path)))
    definition = problem_definition(); definition["domain"]["x"] = ["0.0", 1]
    source_id = create(kernel, definition)
    made = kernel.apply(source_id, "classify")
    assert kernel.math_objects[source_id]["input_trust"].value == "numeric"
    assert made.trust.value == "numeric"
    limited = MathKernel(Settings(store_path=str(path), max_pde_work=1))
    assert limited.object_get(source_id).ok
    before = len(limited.math_objects)
    refused = limited.apply(source_id, "classify")
    assert not refused.ok and len(limited.math_objects) == before


def test_capability_manifest_and_legacy_pde_overview_are_complete():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="pde")
    assert manifest["count"] >= 5
    assert {
        "pde.PDEProblem.verify", "pde.PDEProblem.classify",
        "pde.PDEProblem.boundary_compatibility",
        "pde.PDEClassification.verify", "pde.PDECompatibilityReport.verify",
    }.issubset({item["name"] for item in manifest["capabilities"]})
    assert all(item["cost_dimensions"] == [
        "field_equation_terms", "condition_pairs", "expression_size"]
        for item in manifest["capabilities"]
        if item["input_types"][0] in {"PDEProblem", "PDEClassification",
                                      "PDECompatibilityReport", "WeakForm"})
    overview = kernel.capabilities()["pde"]
    assert "pde_heat_1d" in overview["operations"]
    assert overview["max_pde_work"] == 2_000_000


def test_generic_mcp_tools_expose_the_complete_pde_representation_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            source = await client.call_tool("math_object_create", {
                "object_type": "PDEProblem", "definition": problem_definition()})
            made = await client.call_tool("math_apply", {
                "object_id": source.data["data"]["object_id"],
                "operation": "classify", "parameters": {}})
            replay = await client.call_tool("math_apply", {
                "object_id": made.data["data"]["object_id"],
                "operation": "verify", "parameters": {}})
            return made.data, replay.data

    made, replay = asyncio.run(scenario())
    assert made["ok"] and made["data"]["object_type"] == "PDEClassification"
    assert replay["ok"] and replay["status"] == "verified"


