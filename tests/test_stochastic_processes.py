import asyncio
import math

import numpy as np
import pytest
import sympy as sp

from mathkernel import (
    CTMCTransition, ContinuousTimeMarkovChain, FiniteDimensionalDistribution,
    GaussianProcess, GaussianProcessPosterior, MathKernel, PoissonProcess,
    Settings, WienerProcess,
)


def create(kernel, object_type, definition):
    made = kernel.object_create(object_type, definition)
    assert made.ok, made.errors
    return made.data["object_id"]


def test_poisson_exact_queries_and_assumption_scope():
    kernel = MathKernel()
    process_id = create(kernel, "PoissonProcess", {
        "rate": "3/2", "start_time": 1,
        "assumptions": ["events follow the configured process law"],
    })
    assert isinstance(kernel.math_objects[process_id]["value"], PoissonProcess)
    moments = kernel.apply(process_id, "moments", {"time": 5})
    mass = kernel.apply(process_id, "pmf", {"time": 3, "count": 2})
    assert moments.ok and moments.data["value"]["mean"] == "6"
    assert mass.ok and sp.sympify(mass.data["value"]["probability"]) == 9 * sp.exp(-3) / 2
    model = mass.data["claim_evidence"]["pmf"]["model"][0]
    assert model["trust"] == "unknown"
    assert model["metadata"]["model_validity"] == "not_established"


def test_poisson_increment_is_replayable_and_keeps_increment_parameter():
    kernel = MathKernel(); process_id = create(kernel, "poisson_process", {"rate": 2})
    made = kernel.apply(process_id, "increment_distribution", {"start": 1, "end": 4})
    assert made.ok and made.data["object_type"] == "FiniteDimensionalDistribution"
    output = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(output, FiniteDimensionalDistribution)
    assert output.increment_parameters == (sp.Integer(6),)
    assert output.independent_increments is True
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"


@pytest.mark.parametrize("definition", [{"rate": 0}, {"rate": -1}, {"rate": "x"}])
def test_poisson_invalid_or_symbolic_rate_is_refused(definition):
    made = MathKernel().object_create("PoissonProcess", definition)
    assert not made.ok


def test_wiener_finite_dimensional_law_is_exact_and_replayable():
    kernel = MathKernel(); process_id = create(kernel, "brownian_process", {
        "drift": "1/2", "diffusion": 2, "initial": 3, "start_time": 0,
    })
    assert isinstance(kernel.math_objects[process_id]["value"], WienerProcess)
    made = kernel.apply(process_id, "finite_dimensional", {"times": [0, 1, 3]})
    output = kernel.math_objects[made.data["object_id"]]["value"]
    assert output.means == (sp.Integer(3), sp.Rational(7, 2), sp.Rational(9, 2))
    assert output.covariance == ((0, 0, 0), (0, 4, 4), (0, 4, 12))
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"


def test_wiener_increment_and_time_domains():
    kernel = MathKernel(); process_id = create(kernel, "WienerProcess", {
        "drift": -1, "diffusion": "1/2", "start_time": 2,
    })
    increment = kernel.apply(process_id, "increment_distribution", {"start": 3, "end": 5})
    assert increment.data["value"]["mean"] == "-2"
    assert increment.data["value"]["variance"] == "1/2"
    assert not kernel.apply(process_id, "finite_dimensional", {"times": [1, 3]}).ok
    assert not kernel.apply(process_id, "increment_distribution", {"start": 4, "end": 4}).ok


@pytest.mark.parametrize("kernel_name", ["rbf", "matern32", "linear", "brownian"])
def test_gaussian_process_kernel_families_have_psd_finite_laws(kernel_name):
    kernel = MathKernel(); process_id = create(kernel, "GaussianProcess", {
        "kernel": kernel_name, "variance": 2, "length_scale": "3/2",
    })
    assert isinstance(kernel.math_objects[process_id]["value"], GaussianProcess)
    made = kernel.apply(process_id, "finite_dimensional", {"times": [0, 1, 2]})
    assert made.ok, made.errors
    output = kernel.math_objects[made.data["object_id"]]["value"]
    matrix = np.asarray([[float(value) for value in row] for row in output.covariance])
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"


def test_gp_conditioning_is_bounded_numeric_and_replayable():
    kernel = MathKernel(); process_id = create(kernel, "GaussianProcess", {
        "kernel": "rbf", "length_scale": 2, "observation_noise": "1/100",
    })
    prior = kernel.apply(process_id, "finite_dimensional", {"times": [0, 1]})
    made = kernel.apply(process_id, "condition", {
        "observation_times": [0, 1], "observation_values": [1, 2],
        "prediction_times": [0, 1],
    })
    assert made.ok and made.trust.value == "numeric"
    posterior = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(posterior, GaussianProcessPosterior)
    prior_cov = kernel.math_objects[prior.data["object_id"]]["value"].covariance
    assert all(float(posterior.covariance[i][i]) < float(prior_cov[i][i]) for i in range(2))
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"
    assert made.data["details"]["hyperparameters_fitted"] is False


def test_gp_singular_conditioning_refuses_without_noise_but_jitter_is_explicit():
    kernel = MathKernel(); process_id = create(kernel, "GaussianProcess", {"kernel": "rbf"})
    parameters = {"observation_times": [0, 0], "observation_values": [1, 1],
                  "prediction_times": [0, 1]}
    assert not kernel.apply(process_id, "condition", parameters).ok
    repaired = kernel.apply(process_id, "condition", {**parameters, "jitter": 1e-8})
    assert repaired.ok
    assert float(repaired.data["details"]["jitter"]) == 1e-8


def test_ctmc_generator_transition_distribution_and_stationary_law():
    kernel = MathKernel(); chain_id = create(kernel, "ctmc", {
        "states": ["a", "b"], "generator": [[-2, 2], [1, -1]],
        "initial_distribution": [1, 0],
    })
    assert isinstance(kernel.math_objects[chain_id]["value"], ContinuousTimeMarkovChain)
    assert kernel.apply(chain_id, "verify").status == "verified"
    made = kernel.apply(chain_id, "transition_matrix", {"time": 1})
    transition = kernel.math_objects[made.data["object_id"]]["value"]
    assert isinstance(transition, CTMCTransition)
    matrix = np.asarray([[float(value) for value in row] for row in transition.matrix])
    assert np.allclose(matrix.sum(axis=1), 1)
    assert np.all(matrix >= 0)
    assert kernel.apply(made.data["object_id"], "verify").status == "verified"
    stationary = kernel.apply(chain_id, "stationary_distribution")
    assert stationary.data["value"]["probabilities"] == ["1/3", "2/3"]
    distribution = kernel.apply(chain_id, "distribution", {"time": 1})
    assert distribution.ok and math.isclose(
        sum(float(value) for value in distribution.data["value"]["probabilities"]), 1)


def test_invalid_ctmc_is_refuted_and_cannot_be_exponentiated():
    kernel = MathKernel(); chain_id = create(kernel, "ContinuousTimeMarkovChain", {
        "states": ["a", "b"], "generator": [[-1, -1], [1, -1]],
        "initial_distribution": [1, 0],
    })
    assert kernel.apply(chain_id, "verify").status == "refuted"
    assert not kernel.apply(chain_id, "transition_matrix", {"time": 1}).ok


def test_nonunique_ctmc_stationary_distribution_is_unknown():
    kernel = MathKernel(); chain_id = create(kernel, "ctmc", {
        "states": ["a", "b"], "generator": [[0, 0], [0, 0]],
        "initial_distribution": ["1/2", "1/2"],
    })
    result = kernel.apply(chain_id, "stationary_distribution")
    assert result.data["status"] == "unknown" and result.trust.value == "unknown"


def test_stochastic_resource_guards_fire_before_outputs_are_stored():
    kernel = MathKernel(Settings(max_stochastic_time_points=2,
                                 max_stochastic_matrix_entries=4,
                                 max_stochastic_work=8))
    process_id = create(kernel, "WienerProcess", {})
    before = len(kernel.math_objects)
    result = kernel.apply(process_id, "finite_dimensional", {"times": [0, 1, 2]})
    assert not result.ok and len(kernel.math_objects) == before


def test_stochastic_types_round_trip_through_sqlite(tmp_path):
    path = tmp_path / "stochastic.sqlite"
    kernel = MathKernel(Settings(store_path=str(path)))
    process_id = create(kernel, "PoissonProcess", {"rate": "7/3"})
    made = kernel.apply(process_id, "increment_distribution", {"start": 0, "end": 3})
    output_id = made.data["object_id"]
    reopened = MathKernel(Settings(store_path=str(path)))
    assert reopened.object_get(process_id).data["object_type"] == "PoissonProcess"
    assert reopened.object_get(output_id).data["object_type"] == "FiniteDimensionalDistribution"
    assert reopened.apply(output_id, "verify").status == "verified"


def test_stochastic_capability_manifest_and_version_are_complete():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert entries[("GaussianProcess", "condition")]["output_types"] == [
        "EngineeringResult", "GaussianProcessPosterior"]
    assert entries[("ContinuousTimeMarkovChain", "transition_matrix")]["output_types"] == [
        "EngineeringResult", "CTMCTransition"]
    assert "process_assumption_scope" in entries[("PoissonProcess", "verify")]["verification_methods"]
    assert kernel.capabilities()["version"] == __import__("mathkernel").__version__
    for name in ("max_stochastic_states", "max_stochastic_time_points",
                 "max_gp_conditioning_points", "max_stochastic_matrix_entries",
                 "max_gp_condition_number", "max_stochastic_work"):
        assert name in kernel.capabilities()["limits"]



def test_stochastic_output_types_cannot_be_created_directly():
    kernel = MathKernel()
    for name in ("FiniteDimensionalDistribution", "GaussianProcessPosterior",
                 "CTMCTransition"):
        assert not kernel.object_create(name, {}).ok


def test_live_mcp_stochastic_surface():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            made = await client.call_tool("math_object_create", {
                "object_type": "PoissonProcess", "definition": {"rate": 2}})
            result = await client.call_tool("math_apply", {
                "object_id": made.data["data"]["object_id"], "operation": "pmf",
                "parameters": {"time": 1, "count": 0}})
            assert result.data["ok"]
            assert result.data["data"]["value"]["probability"] == "exp(-2)"

    asyncio.run(scenario())
