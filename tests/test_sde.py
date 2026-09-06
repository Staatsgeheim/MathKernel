import asyncio
import math

import numpy as np
import pytest
import sympy as sp

from mathkernel import (
    MathKernel, SDEConvergenceStudy, SDESimulation, Settings,
    StochasticDifferentialEquation,
)


def create(kernel, definition):
    made = kernel.object_create("StochasticDifferentialEquation", definition)
    assert made.ok, made.errors
    return made.data["object_id"]


def gbm(kernel, *, decimal=False):
    return create(kernel, {
        "state_variables": ["x"], "time_variable": "t",
        "drift": ["mu*x"], "diffusion": [["sigma*x"]],
        "initial_state": ["1.0" if decimal else 1],
        "start_time": 0, "end_time": 1,
        "parameters": {"mu": "1/10", "sigma": "1/5"},
        "assumptions": ["coefficients satisfy the required regularity"],
    })


def test_typed_sde_preserves_equation_and_symbol_scope():
    kernel = MathKernel(); sde_id = gbm(kernel)
    model = kernel.math_objects[sde_id]["value"]
    assert isinstance(model, StochasticDifferentialEquation)
    assert model.drift == (sp.Symbol("mu") * sp.Symbol("x"),)
    assert model.diffusion == ((sp.Symbol("sigma") * sp.Symbol("x"),),)
    checked = kernel.apply(sde_id, "verify")
    assert checked.status == "verified"
    assert checked.data["value"] == {
        "state_dimension": 1, "noise_dimension": 1, "interval": ["0", "1"]}
    assert checked.data["details"]["existence_uniqueness"] == "not_established"


@pytest.mark.parametrize("change", [
    {"drift": ["alpha*x"]},
    {"start_time": 1, "end_time": 1},
    {"start_time": 2, "end_time": 1},
    {"interpretation": "stratonovich"},
    {"drift": ["I*x"]},
])
def test_invalid_sde_definitions_are_refused(change):
    definition = {
        "state_variables": ["x"], "drift": ["x"], "diffusion": [[1]],
        "initial_state": [0], "start_time": 0, "end_time": 1,
    }
    definition.update(change)
    assert not MathKernel().object_create("sde", definition).ok


def test_euler_maruyama_vector_state_full_diffusion_is_reproducible():
    kernel = MathKernel(); sde_id = create(kernel, {
        "state_variables": ["x", "y"], "drift": ["-x", "-2*y"],
        "diffusion": [[1, "1/2"], [0, 1]], "initial_state": [0, 1],
        "start_time": 0, "end_time": 2,
    })
    parameters = {"scheme": "euler_maruyama", "steps": 12,
                  "paths": 5, "seed": 1234}
    first = kernel.apply(sde_id, "simulate", parameters)
    second = kernel.apply(sde_id, "simulate", parameters)
    assert first.ok and second.ok and first.trust.value == "empirical"
    left = kernel.math_objects[first.data["object_id"]]["value"]
    right = kernel.math_objects[second.data["object_id"]]["value"]
    assert isinstance(left, SDESimulation)
    assert left.values == right.values and left.state_dimension == left.noise_dimension == 2
    assert len(left.values) == 5 and len(left.values[0]) == 13
    assert "values" not in first.data["object"]
    assert first.data["object"]["stored_fields"] == ["times", "values"]
    path = kernel.apply(first.data["object_id"], "path", {
        "path_index": 2, "start_step": 3, "count": 4})
    terminal = kernel.apply(first.data["object_id"], "terminal_values", {
        "start_path": 1, "count": 3})
    assert path.data["value"]["values"] == [
        list(state) for state in left.values[2][3:7]]
    assert terminal.data["value"]["values"] == [
        list(item) for item in (left.values[index][-1] for index in range(1, 4))]
    assert kernel.apply(first.data["object_id"], "verify").status == "verified"


def test_different_seeds_produce_different_paths():
    kernel = MathKernel(); sde_id = gbm(kernel)
    common = {"scheme": "euler_maruyama", "steps": 8, "paths": 2}
    left = kernel.apply(sde_id, "simulate", {**common, "seed": 1})
    right = kernel.apply(sde_id, "simulate", {**common, "seed": 2})
    assert (kernel.math_objects[left.data["object_id"]]["value"].values !=
            kernel.math_objects[right.data["object_id"]]["value"].values)


def test_one_step_milstein_matches_the_defining_update():
    kernel = MathKernel(); sde_id = gbm(kernel)
    made = kernel.apply(sde_id, "simulate", {
        "scheme": "milstein", "steps": 1, "paths": 1, "seed": 42})
    output = kernel.math_objects[made.data["object_id"]]["value"]
    dw = np.random.Generator(np.random.PCG64(42)).normal(size=(1, 1, 1))[0, 0, 0]
    expected = 1 + 0.1 + 0.2 * dw + 0.5 * 0.2 * 0.2 * (dw * dw - 1)
    assert math.isclose(float(output.values[0][1][0]), expected, rel_tol=1e-14)
    assert output.nominal_strong_order == 1
    assert output.nominal_weak_order == 1


def test_multidimensional_milstein_is_refused_without_downgrade_or_output():
    kernel = MathKernel(); sde_id = create(kernel, {
        "state_variables": ["x", "y"], "drift": [0, 0],
        "diffusion": [[1, 0], [0, 1]], "initial_state": [0, 0],
        "start_time": 0, "end_time": 1,
    })
    before = len(kernel.math_objects)
    result = kernel.apply(sde_id, "simulate", {
        "scheme": "milstein", "steps": 5, "paths": 2, "seed": 1})
    assert not result.ok and "only for scalar" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_additive_brownian_terminal_moments_are_reasonable_but_empirical():
    kernel = MathKernel(); sde_id = create(kernel, {
        "state_variables": ["x"], "drift": [0], "diffusion": [[1]],
        "initial_state": [0], "start_time": 0, "end_time": 1,
    })
    made = kernel.apply(sde_id, "simulate", {
        "scheme": "euler_maruyama", "steps": 20, "paths": 4000, "seed": 99})
    output = kernel.math_objects[made.data["object_id"]]["value"]
    assert abs(float(output.terminal_mean[0])) < 0.05
    assert abs(float(output.terminal_variance[0]) - 1) < 0.06
    evidence = made.data["claim_evidence"]["simulate"]
    assert evidence["empirical"][0]["role"] == "required"
    assert evidence["empirical"][0]["replication"] == {
        "seed": 99, "algorithm": "PCG64", "stream": 0}
    assert evidence["numerical"][0]["metadata"]["error_bound"] is False


def test_decimal_ancestry_is_preserved_below_empirical_simulation():
    kernel = MathKernel(); sde_id = gbm(kernel, decimal=True)
    assert kernel.math_objects[sde_id]["input_trust"].value == "numeric"
    made = kernel.apply(sde_id, "simulate", {
        "scheme": "euler_maruyama", "steps": 2, "paths": 1, "seed": 0})
    assert made.trust.value == "empirical"
    assert made.data["provenance"]["required_object_inputs"][0]["trust"] == "numeric"


def test_coupled_three_level_convergence_study_and_replay():
    kernel = MathKernel(); sde_id = gbm(kernel)
    made = kernel.apply(sde_id, "convergence_study", {
        "scheme": "milstein", "base_steps": 8, "paths": 512, "seed": 123})
    assert made.ok and made.data["object_type"] == "SDEConvergenceStudy"
    study = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(study, SDEConvergenceStudy)
    assert study.levels == (8, 16, 32)
    assert study.coupled_brownian_paths and study.observed_strong_order is not None
    assert all(float(value) > 0 for value in study.rms_terminal_differences)
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"
    assert made.data["details"]["nominal_order_is_assumption"] is True


@pytest.mark.parametrize("parameters,needle", [
    ({"scheme": "bogus", "steps": 2, "paths": 1, "seed": 0}, "scheme"),
    ({"scheme": "euler_maruyama", "steps": 0, "paths": 1, "seed": 0}, "max_sde_steps"),
    ({"scheme": "euler_maruyama", "steps": 2, "paths": 0, "seed": 0}, "max_sde_paths"),
    ({"scheme": "euler_maruyama", "steps": 2, "paths": 1, "seed": -1}, "uint64"),
    ({"scheme": "euler_maruyama", "steps": 2, "paths": 1, "seed": 2**64}, "uint64"),
])
def test_invalid_simulation_parameters_are_refused(parameters, needle):
    kernel = MathKernel(); sde_id = gbm(kernel)
    result = kernel.apply(sde_id, "simulate", parameters)
    assert not result.ok and needle in result.errors[0]


def test_resource_preflight_happens_before_simulation_output():
    kernel = MathKernel(Settings(max_sde_simulation_cells=100,
                                 max_sde_work=1000))
    sde_id = gbm(kernel); before = len(kernel.math_objects)
    result = kernel.apply(sde_id, "simulate", {
        "scheme": "euler_maruyama", "steps": 10, "paths": 10, "seed": 1})
    assert not result.ok and "max_sde_simulation_cells" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_convergence_finest_level_obeys_step_limit():
    kernel = MathKernel(Settings(max_sde_steps=10)); sde_id = gbm(kernel)
    result = kernel.apply(sde_id, "convergence_study", {
        "scheme": "milstein", "base_steps": 3, "paths": 1, "seed": 1})
    assert not result.ok and "finest convergence level" in result.errors[0]


def test_replay_obeys_current_resource_limits_after_restart(tmp_path):
    path = tmp_path / "bounded-replay.sqlite"
    kernel = MathKernel(Settings(store_path=str(path)))
    sde_id = gbm(kernel)
    made = kernel.apply(sde_id, "simulate", {
        "scheme": "euler_maruyama", "steps": 10, "paths": 10, "seed": 1})
    limited = MathKernel(Settings(store_path=str(path), max_sde_simulation_cells=10))
    result = limited.apply(made.data["object_id"], "verify")
    assert not result.ok and "current simulation limits" in result.errors[0]


def test_simulation_queries_are_paginated_by_value_budget():
    kernel = MathKernel(Settings(max_sde_query_values=2))
    sde_id = gbm(kernel)
    made = kernel.apply(sde_id, "simulate", {
        "scheme": "euler_maruyama", "steps": 3, "paths": 3, "seed": 1})
    assert not kernel.apply(made.data["object_id"], "path", {
        "path_index": 0, "start_step": 0, "count": 3}).ok
    assert kernel.apply(made.data["object_id"], "path", {
        "path_index": 0, "start_step": 0, "count": 2}).ok
    assert not kernel.apply(made.data["object_id"], "terminal_values", {
        "start_path": 0, "count": 3}).ok


def test_dimension_limits_are_enforced_at_construction():
    kernel = MathKernel(Settings(max_sde_state_dimension=1,
                                 max_sde_noise_dimension=1))
    made = kernel.object_create("sde", {
        "state_variables": ["x", "y"], "drift": [0, 0],
        "diffusion": [[1], [1]], "initial_state": [0, 0],
        "start_time": 0, "end_time": 1})
    assert not made.ok and "max_sde_state_dimension" in made.errors[0]


def test_sde_results_are_output_only_and_persist_with_source(tmp_path):
    settings = Settings(store_path=str(tmp_path / "sde.sqlite"))
    kernel = MathKernel(settings); sde_id = gbm(kernel)
    simulation = kernel.apply(sde_id, "simulate", {
        "scheme": "milstein", "steps": 5, "paths": 3, "seed": 10})
    study = kernel.apply(sde_id, "convergence_study", {
        "scheme": "milstein", "base_steps": 2, "paths": 8, "seed": 11})
    assert not kernel.object_create("SDESimulation", {}).ok
    assert not kernel.object_create("SDEConvergenceStudy", {}).ok
    reopened = MathKernel(settings)
    assert reopened.object_get(sde_id).data["object_type"] == "StochasticDifferentialEquation"
    assert reopened.apply(simulation.data["object_id"], "verify").status == "verified"
    assert reopened.apply(study.data["object_id"], "verify").status == "verified"


def test_sde_capabilities_version_and_limits():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert entries[("StochasticDifferentialEquation", "simulate")]["output_types"] == [
        "EngineeringResult", "SDESimulation"]
    assert entries[("StochasticDifferentialEquation", "convergence_study")]["output_types"] == [
        "EngineeringResult", "SDEConvergenceStudy"]
    assert entries[("SDESimulation", "verify")]["trust_levels"] == ["empirical"]
    assert "seeded_stream_replay" in entries[("StochasticDifferentialEquation", "simulate")]["verification_methods"]
    assert kernel.capabilities()["version"] == __import__("mathkernel").__version__
    for name in ("max_sde_state_dimension", "max_sde_noise_dimension",
                 "max_sde_steps", "max_sde_paths", "max_sde_simulation_cells",
                 "max_sde_work", "max_sde_query_values"):
        assert name in kernel.capabilities()["limits"]


def test_live_mcp_sde_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            model = await client.call_tool("math_object_create", {
                "object_type": "sde", "definition": {
                    "state_variables": ["x"], "drift": [0],
                    "diffusion": [[1]], "initial_state": [0],
                    "start_time": 0, "end_time": 1}})
            made = await client.call_tool("math_apply", {
                "object_id": model.data["data"]["object_id"],
                "operation": "simulate", "parameters": {
                    "scheme": "euler_maruyama", "steps": 3,
                    "paths": 2, "seed": 42}})
            assert made.data["ok"] and made.data["data"]["object_type"] == "SDESimulation"

    asyncio.run(scenario())


