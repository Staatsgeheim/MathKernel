import asyncio

import pytest
import sympy as sp

from mathkernel import (
    CovarianceMatrix, DescriptiveSummary, EmpiricalDistribution, MathKernel,
    Settings, StatisticalSample, TrustLevel,
)


ROWS = [[1, 2], [2, 4], [3, 8], [4, 10]]


def create(kernel, definition=None):
    result = kernel.object_create("StatisticalSample", definition or {
        "variables": ["x", "y"], "observations": ROWS,
        "observation_ids": ["a", "b", "c", "d"],
        "sampling_method": "simple_random", "population": "target cohort",
        "design_assumptions": ["selection probabilities were equal"],
    })
    assert result.ok, result.errors
    return result.data["object_id"]


def test_statistical_sample_is_typed_immutable_and_exact_for_rational_data():
    k = MathKernel(); sample_id = create(k)
    sample = k.math_objects[sample_id]["value"]
    assert isinstance(sample, StatisticalSample)
    assert sample.input_trust == "exact"
    assert sample.observations[0] == (1, 2)
    with pytest.raises(Exception):
        sample.population = "other"


def test_describe_computes_exact_sample_facts_and_five_number_summary():
    k = MathKernel(); result = k.apply(create(k), "describe")
    assert result.ok and result.status == "verified" and result.trust == TrustLevel.EXACT
    summary = k.math_objects[result.data["object_id"]]["value"]
    assert isinstance(summary, DescriptiveSummary)
    x, y = summary.summaries
    assert (x.mean, x.variance_population, x.variance_sample) == (
        sp.Rational(5, 2), sp.Rational(5, 4), sp.Rational(5, 3))
    assert (x.minimum, x.q1, x.median, x.q3, x.maximum) == (
        1, sp.Rational(7, 4), sp.Rational(5, 2), sp.Rational(13, 4), 4)
    assert y.mean == 6 and y.variance_sample == sp.Rational(40, 3)
    assert summary.inference_scope == "sample_only"


def test_statistical_evidence_keeps_computation_model_and_empirical_roles_separate():
    k = MathKernel(); result = k.apply(create(k), "evidence_profile")
    bundle = result.claim_evidence["evidence_profile"]
    assert bundle.conservative_trust() == "exact"
    assert bundle.computation[0].role == "required"
    assert bundle.empirical[0].role == "diagnostic"
    assert bundle.empirical[0].sample_size == 4
    assert bundle.model[0].role == "diagnostic"
    assert bundle.model[0].trust == "unknown"
    assert result.data["value"]["population_generalization"] == "not_established"
    assert result.data["value"]["model_validity"] == "not_established"


def test_asserted_sampling_design_is_not_mislabeled_as_verified_model_evidence():
    k = MathKernel(); result = k.apply(create(k), "describe")
    assert result.data["details"]["sampling_method_asserted_not_verified"] == "simple_random"
    model = result.claim_evidence["describe"].model[0]
    assert model.metadata["population_claim"] == "not_established"
    assert "population generalization is not established" in model.diagnostics


def test_covariance_supports_exact_sample_and_population_normalizations():
    k = MathKernel(); sample_id = create(k)
    sample = k.apply(sample_id, "covariance")
    population = k.apply(sample_id, "covariance", {"normalization": "population"})
    left = k.math_objects[sample.data["object_id"]]["value"]
    right = k.math_objects[population.data["object_id"]]["value"]
    assert isinstance(left, CovarianceMatrix)
    assert left.matrix == ((sp.Rational(5, 3), sp.Rational(14, 3)),
                           (sp.Rational(14, 3), sp.Rational(40, 3)))
    assert right.matrix == ((sp.Rational(5, 4), sp.Rational(7, 2)),
                            (sp.Rational(7, 2), 10))
    assert left.matrix[0][1] == left.matrix[1][0]


def test_empirical_distribution_retains_exact_counts_and_probabilities():
    k = MathKernel(); sample_id = create(k, {
        "variables": ["value"], "observations": [[2], [1], [2], [4], [1]]})
    result = k.apply(sample_id, "empirical_distribution", {"variable": "value"})
    distribution = k.math_objects[result.data["object_id"]]["value"]
    assert isinstance(distribution, EmpiricalDistribution)
    assert distribution.support == (1, 2, 4)
    assert distribution.counts == (2, 2, 1)
    assert distribution.probabilities == (
        sp.Rational(2, 5), sp.Rational(2, 5), sp.Rational(1, 5))
    assert distribution.inference_scope == "sample_empirical_distribution"


def test_decimal_observations_remain_numeric_through_every_derived_object():
    k = MathKernel(); sample_id = create(k, {
        "variables": ["x"], "observations": [["0.1"], ["0.2"], ["0.3"]]})
    for operation, parameters in (("describe", {}), ("covariance", {}),
                                  ("empirical_distribution", {"variable": "x"})):
        result = k.apply(sample_id, operation, parameters)
        assert result.ok and result.trust == TrustLevel.NUMERIC
        assert k.math_objects[result.data["object_id"]]["input_trust"] == TrustLevel.NUMERIC


@pytest.mark.parametrize("definition, message", [
    ({"variables": ["x", "x"], "observations": [[1, 2]]}, "unique"),
    ({"variables": ["x", "y"], "observations": [[1]]}, "variable count"),
    ({"variables": ["x"], "observations": [["z"]]}, "concrete"),
    ({"variables": ["x"], "observations": [["I"]]}, "concrete"),
    ({"variables": ["x"], "observations": [[True]]}, "MathIR strings"),
    ({"variables": ["x"], "observations": [[1], [2]],
      "observation_ids": ["same", "same"]}, "unique"),
])
def test_invalid_sample_definitions_fail_closed(definition, message):
    result = MathKernel().object_create("StatisticalSample", definition)
    assert not result.ok and message in result.errors[0]


def test_single_observation_refuses_sample_covariance_but_allows_population_covariance():
    k = MathKernel(); sample_id = create(k, {"variables": ["x"], "observations": [[7]]})
    assert not k.apply(sample_id, "covariance").ok
    result = k.apply(sample_id, "covariance", {"normalization": "population"})
    assert result.ok and result.data["value"]["matrix"] == [["0"]]


def test_resource_limits_apply_before_sample_and_quadratic_covariance_work():
    construction = MathKernel(Settings(max_statistical_cells=3)).object_create(
        "StatisticalSample", {"variables": ["x", "y"],
                              "observations": [[1, 2], [3, 4]]})
    assert not construction.ok and "max_statistical_cells" in construction.errors[0]
    k = MathKernel(Settings(max_statistical_work=7)); sample_id = create(k)
    before = len(k.math_objects)
    result = k.apply(sample_id, "covariance")
    assert not result.ok and "max_statistical_work" in result.errors[0]
    assert len(k.math_objects) == before


def test_statistical_metadata_uses_the_global_input_budget():
    result = MathKernel(Settings(max_input_length=5)).object_create(
        "StatisticalSample", {"variables": ["long_name"], "observations": [[1]]})
    assert not result.ok and "max_input_length" in result.errors[0]


def test_statistical_objects_and_evidence_survive_restart(tmp_path):
    settings = Settings(store_path=str(tmp_path / "statistics.sqlite"))
    k = MathKernel(settings); sample_id = create(k)
    summary = k.apply(sample_id, "describe")
    restarted = MathKernel(settings)
    assert restarted.object_get(sample_id).ok
    restored = restarted.object_get(summary.data["object_id"])
    assert restored.ok and restored.data["sources"] == [sample_id]
    assert restored.claim_evidence["describe"].model[0].role == "diagnostic"


def test_statistical_derived_types_are_output_only_and_publicly_exported():
    k = MathKernel()
    for kind in ("DescriptiveSummary", "VariableSummary", "CovarianceMatrix",
                 "EmpiricalDistribution"):
        assert not k.object_create(kind, {}).ok
    assert all(item is not None for item in
               (StatisticalSample, DescriptiveSummary, CovarianceMatrix,
                EmpiricalDistribution))


def test_statistics_capability_and_limit_surfaces_are_truthful():
    k = MathKernel(); manifest = k.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {item["operation"]: item for item in manifest["capabilities"]}
    assert entries["describe"]["output_types"] == ["EngineeringResult", "DescriptiveSummary"]
    assert entries["covariance"]["cost_dimensions"] == [
        "observation_count", "variable_count", "arithmetic"]
    assert set(entries["evidence_profile"]["verification_methods"]) == {
        "claim_scope_audit", "population_non_inference"}
    caps = k.capabilities()["statistics_inference"]
    assert k.capabilities()["version"] == __import__("mathkernel").__version__
    for name in ("max_statistical_variables", "max_statistical_observations",
                 "max_statistical_cells", "max_statistical_work"):
        assert name in Settings().limits()


def test_live_mcp_statistical_evidence_workflow():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def run():
        async with Client(mcp) as client:
            sample = (await client.call_tool("math_object_create", {
                "object_type": "StatisticalSample", "definition": {
                    "variables": ["x"], "observations": [[1], [2], [3]],
                    "sampling_method": "unspecified"}})).data["data"]["object_id"]
            return (await client.call_tool("math_apply", {
                "object_id": sample, "operation": "evidence_profile",
                "parameters": {}})).data

    result = asyncio.run(run())
    assert result["ok"] and result["trust"] == "exact"
    assert result["data"]["value"]["population_generalization"] == "not_established"
