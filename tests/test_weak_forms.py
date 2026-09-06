import asyncio

import pytest
import sympy as sp

from mathkernel import MathKernel, Settings, WeakForm


def poisson_definition(*, coefficient=-1, decimal=False):
    zero = "0.0" if decimal else 0
    return {
        "fields": ["u"], "independent_variables": ["x", "y"],
        "domain": {"x": [zero, 1], "y": [0, 1]},
        "parameters": {"f": None},
        "equations": [{"label": "Poisson", "terms": [
            {"coefficient": coefficient, "field": "u", "derivative": {"x": 2}},
            {"coefficient": -1, "field": "u", "derivative": {"y": 2}},
        ], "source": "f"}],
        "boundary_conditions": [
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "upper", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "lower", "value": 0},
            {"field": "u", "kind": "dirichlet", "coordinate": "y", "side": "upper", "value": 0},
        ],
    }


def weak_parameters(*, trace=(0, 1, 2, 3), regularity=1, steps=None):
    return {
        "integration_variables": ["x", "y"],
        "trial_spaces": [{
            "name": "U", "field": "u", "family": "H1",
            "regularity_order": regularity,
            "trace_boundary_indices": list(trace),
            "assumptions": ["u has the declared weak derivatives"],
        }],
        "test_space": {
            "name": "v", "field": "u", "family": "H1_0",
            "regularity_order": regularity,
            "trace_boundary_indices": list(trace),
            "assumptions": ["v has the declared zero trace"],
        },
        "integration_by_parts": steps if steps is not None else [
            {"term_index": 0, "coordinate": "x"},
            {"term_index": 1, "coordinate": "y"},
        ],
    }


def create(kernel, definition=None):
    made = kernel.object_create("PDEProblem", definition or poisson_definition())
    assert made.ok, made.errors
    return made.data["object_id"]


def derive(kernel, definition=None, parameters=None):
    source_id = create(kernel, definition)
    made = kernel.apply(source_id, "derive_weak_form", parameters or weak_parameters())
    assert made.ok, made.errors
    return source_id, made


def test_poisson_weak_form_retains_all_volume_and_oriented_boundary_terms():
    kernel = MathKernel(); source_id, made = derive(kernel)
    assert made.status == "verified" and made.data["object_type"] == "WeakForm"
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(weak, WeakForm) and weak.problem_id == source_id
    assert [term.coefficient for term in weak.volume_terms] == [1, 1, sp.Symbol("f")]
    assert [term.side for term in weak.volume_terms] == ["lhs", "lhs", "rhs"]
    assert [term.field_derivative for term in weak.volume_terms[:2]] == [(1, 0), (0, 1)]
    assert [term.test_derivative for term in weak.volume_terms[:2]] == [(1, 0), (0, 1)]
    assert len(weak.boundary_terms) == 4
    assert [term.measure.outward_orientation for term in weak.boundary_terms] == [-1, 1, -1, 1]
    assert all(term.vanishes_by_trace for term in weak.boundary_terms)
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"


def test_variable_coefficient_product_rule_term_is_not_dropped():
    kernel = MathKernel(); definition = poisson_definition(coefficient="-x")
    parameters = weak_parameters(regularity=2, steps=[{"term_index": 0, "coordinate": "x"}])
    _, made = derive(kernel, definition, parameters)
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    first = [term for term in weak.volume_terms if term.origin == "integration_by_parts"]
    product = [term for term in weak.volume_terms if term.origin == "coefficient_derivative"]
    assert len(first) == len(product) == 1
    assert first[0].coefficient == sp.Symbol("x")
    assert product[0].coefficient == 1
    step = weak.derivation_steps[0]
    assert step.coefficient_before == -sp.Symbol("x")
    assert step.coefficient_derivative == -1
    assert len(step.generated_volume_term_indices) == 2
    assert [term.coefficient for term in weak.boundary_terms] == [0, -1]


def test_time_parameterized_heat_form_integrates_only_over_space():
    definition = {
        "fields": ["u"], "independent_variables": ["x", "t"],
        "time_variable": "t", "domain": {"x": [0, 1], "t": [0, 2]},
        "parameters": {"alpha": 1, "f": None}, "equations": [{"terms": [
            {"coefficient": 1, "field": "u", "derivative": {"t": 1}},
            {"coefficient": "-alpha", "field": "u", "derivative": {"x": 2}},
        ], "source": "f"}],
        "boundary_conditions": [
            {"field": "u", "kind": "dirichlet", "coordinate": "x", "side": "lower", "value": 0},
            {"field": "u", "kind": "neumann", "coordinate": "x", "side": "upper", "value": 0},
        ],
        "initial_conditions": [{"field": "u", "value": 0}],
    }
    parameters = {
        "integration_variables": ["x"],
        "trial_spaces": [{"name": "U", "field": "u", "family": "H1",
                          "regularity_order": 1, "trace_boundary_indices": [0]}],
        "test_space": {"name": "v", "field": "u", "family": "H1_D",
                       "regularity_order": 1, "trace_boundary_indices": [0]},
        "integration_by_parts": [{"term_index": 1, "coordinate": "x"}],
    }
    kernel = MathKernel(); _, made = derive(kernel, definition, parameters)
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    assert weak.integration_variables == ("x",)
    assert weak.volume_terms[0].field_derivative == (0, 1)
    assert weak.volume_terms[0].test_derivative == (0, 0)
    assert weak.essential_boundary_indices == (0,)
    assert weak.natural_boundary_indices == (1,)
    assert weak.boundary_terms[0].vanishes_by_trace
    assert not weak.boundary_terms[1].vanishes_by_trace


def test_no_ibp_preserves_strong_volume_term_when_space_is_h2():
    kernel = MathKernel(); _, made = derive(
        kernel, parameters=weak_parameters(regularity=2, steps=[]))
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    assert len(weak.derivation_steps) == 0 and len(weak.boundary_terms) == 0
    assert weak.volume_terms[0].field_derivative == (2, 0)
    assert weak.volume_terms[0].test_derivative == (0, 0)


@pytest.mark.parametrize(("parameters", "message"), [
    (weak_parameters(steps=[]), "regularity"),
    (weak_parameters(steps=[{"term_index": 0, "coordinate": "z"}]), "selection"),
    (weak_parameters(steps=[{"term_index": 0, "coordinate": "x"},
                            {"term_index": 0, "coordinate": "x"}]), "at most one"),
    (weak_parameters(trace=()), "zero trace"),
])
def test_invalid_derivation_contracts_fail_atomically(parameters, message):
    kernel = MathKernel(); source_id = create(kernel); before = len(kernel.math_objects)
    result = kernel.apply(source_id, "derive_weak_form", parameters)
    assert not result.ok and message in result.errors[0].lower()
    assert len(kernel.math_objects) == before


def test_nonlinear_term_cannot_be_selected_for_linear_ibp_identity():
    definition = poisson_definition(); definition["equations"][0]["terms"][0]["power"] = 2
    kernel = MathKernel(); source_id = create(kernel, definition); before = len(kernel.math_objects)
    result = kernel.apply(source_id, "derive_weak_form", weak_parameters())
    assert not result.ok and "linear differentiated" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_space_scope_family_and_trace_validation_fail_closed():
    kernel = MathKernel(); source_id = create(kernel)
    cases = []
    wrong_field = weak_parameters(); wrong_field["trial_spaces"][0]["field"] = "w"; cases.append(wrong_field)
    bad_family = weak_parameters(); bad_family["trial_spaces"][0]["family"] = "C99"; cases.append(bad_family)
    bad_trace = weak_parameters(); bad_trace["trial_spaces"][0]["trace_boundary_indices"] = [99]; cases.append(bad_trace)
    for parameters in cases:
        before = len(kernel.math_objects)
        assert not kernel.apply(source_id, "derive_weak_form", parameters).ok
        assert len(kernel.math_objects) == before


def test_boundary_partition_ignores_conditions_for_unrelated_equation_fields():
    definition = poisson_definition(); definition["fields"] = ["u", "w"]
    definition["boundary_conditions"].append(
        {"field": "w", "kind": "neumann", "coordinate": "x", "side": "lower", "value": 0})
    kernel = MathKernel(); _, made = derive(kernel, definition)
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    assert weak.essential_boundary_indices == (0, 1, 2, 3)
    assert weak.natural_boundary_indices == ()


def test_robin_and_periodic_conditions_are_partitioned_without_being_erased():
    definition = poisson_definition()
    definition["boundary_conditions"] = [
        {"field": "u", "kind": "robin", "coordinate": "x", "side": "lower",
         "value": 0, "alpha": 1, "beta": 1},
        {"field": "u", "kind": "periodic", "coordinate": "y", "side": "lower",
         "value": 0, "paired_side": "upper"},
        {"field": "u", "kind": "periodic", "coordinate": "y", "side": "upper",
         "value": 0, "paired_side": "lower"},
    ]
    parameters = weak_parameters(trace=())
    kernel = MathKernel(); _, made = derive(kernel, definition, parameters)
    weak = kernel.math_objects[made.data["object_id"]]["value"]
    assert weak.essential_boundary_indices == ()
    assert weak.natural_boundary_indices == (0,)
    assert weak.periodic_boundary_indices == (1, 2)
    assert not any(term.vanishes_by_trace for term in weak.boundary_terms)


def test_creation_limits_preflight_before_weak_form_persistence():
    kernel = MathKernel(Settings(max_pde_ibp_steps=1)); source_id = create(kernel)
    before = len(kernel.math_objects)
    result = kernel.apply(source_id, "derive_weak_form", weak_parameters())
    assert not result.ok and "max_pde_ibp_steps" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_source_ancestry_substitution_fails_closed():
    kernel = MathKernel(); source_id, made = derive(kernel)
    other_id = create(kernel)
    weak_id = made.data["object_id"]
    kernel.math_objects[weak_id]["sources"] = [other_id]
    before = len(kernel.math_objects)
    result = kernel.apply(weak_id, "verify")
    assert not result.ok and len(kernel.math_objects) == before
    assert source_id != other_id


def test_weak_form_is_output_only_and_tampering_is_refuted():
    kernel = MathKernel(); assert not kernel.object_create("WeakForm", {}).ok
    _, made = derive(kernel); result_id = made.data["object_id"]
    original = kernel.math_objects[result_id]["value"]
    bad_term = original.volume_terms[0].model_copy(update={"coefficient": sp.Integer(99)})
    kernel.math_objects[result_id]["value"] = original.model_copy(
        update={"volume_terms": (bad_term, *original.volume_terms[1:])})
    assert kernel.apply(result_id, "verify").status == "refuted"


def test_persistence_lineage_and_replay_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "weak.sqlite"))
    kernel = MathKernel(settings); source_id, made = derive(kernel)
    weak_id = made.data["object_id"]
    restarted = MathKernel(settings)
    fetched = restarted.object_get(weak_id)
    assert fetched.ok and fetched.data["sources"] == [source_id]
    assert restarted.apply(weak_id, "verify").status == "verified"


def test_decimal_ancestry_caps_weak_form_trust():
    kernel = MathKernel(); source_id, made = derive(kernel, poisson_definition(decimal=True))
    assert kernel.math_objects[source_id]["input_trust"].value == "numeric"
    assert made.trust.value == "numeric"
    assert kernel.math_objects[made.data["object_id"]]["input_trust"].value == "numeric"


def test_current_replay_limits_are_enforced_without_partial_output(tmp_path):
    settings = Settings(store_path=str(tmp_path / "limits.sqlite"))
    kernel = MathKernel(settings); _, made = derive(kernel); weak_id = made.data["object_id"]
    limited = MathKernel(Settings(store_path=settings.store_path, max_pde_weak_terms=1))
    fetched = limited.object_get(weak_id)
    assert fetched.ok and limited.object_get(fetched.data["sources"][0]).ok
    before = len(limited.math_objects)
    result = limited.apply(weak_id, "verify")
    assert not result.ok and len(limited.math_objects) == before


def test_capability_manifest_exposes_weak_forms_without_a_solve_claim():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="pde")
    assert manifest["count"] >= 7
    names = {item["name"] for item in manifest["capabilities"]}
    assert "pde.PDEProblem.derive_weak_form" in names
    assert "pde.WeakForm.verify" in names
    assert "pde.WeakForm.solve" not in names
    derive_cap = next(item for item in manifest["capabilities"]
                      if item["name"] == "pde.PDEProblem.derive_weak_form")
    assert derive_cap["output_types"] == ["EngineeringResult", "WeakForm"]
    assert "oriented_boundary_term_retention" in derive_cap["verification_methods"]
    overview = kernel.capabilities()["pde"]
    assert overview["max_pde_weak_work"] == 5_000_000


def test_generic_mcp_tools_run_and_replay_weak_form():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            source = await client.call_tool("math_object_create", {
                "object_type": "PDEProblem", "definition": poisson_definition()})
            made = await client.call_tool("math_apply", {
                "object_id": source.data["data"]["object_id"],
                "operation": "derive_weak_form", "parameters": weak_parameters()})
            replay = await client.call_tool("math_apply", {
                "object_id": made.data["data"]["object_id"],
                "operation": "verify", "parameters": {}})
            return made.data, replay.data

    made, replay = asyncio.run(scenario())
    assert made["ok"] and made["data"]["object_type"] == "WeakForm"
    assert replay["ok"] and replay["status"] == "verified"


