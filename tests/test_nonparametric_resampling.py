import asyncio
from pathlib import Path

import pytest
import sympy as sp

from mathkernel import (
    MathKernel, NonparametricTestResult, ResamplingResult, Settings, TrustLevel,
)


def make_sample(kernel, variables, rows, **metadata):
    result = kernel.object_create("StatisticalSample", {
        "variables": variables, "observations": rows, **metadata})
    assert result.ok, result.errors
    return result.data["object_id"]


def ranked_sample(kernel):
    return make_sample(kernel, ["g", "x", "y"], [
        [0, 1, 1], [0, 2, 2], [0, 3, 3],
        [1, 6, 6], [1, 7, 7], [1, 8, 8],
    ], sampling_method="experiment")


def apply_derived(kernel, sample_id, operation, parameters):
    result = kernel.apply(sample_id, operation, parameters)
    assert result.ok, result.errors
    object_id = result.data["object_id"]
    return result, object_id, kernel.math_objects[object_id]["value"]


def test_mann_whitney_exact_enumerates_all_label_assignments():
    kernel = MathKernel(); sample_id = ranked_sample(kernel)
    result, _, output = apply_derived(kernel, sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1})
    assert isinstance(output, NonparametricTestResult)
    assert output.statistic == 0 and output.p_value == sp.Rational(1, 10)
    assert output.method == "exact_permutation" and output.enumeration_count == 20
    assert result.trust == TrustLevel.EXACT
    assert result.claim_evidence["mann_whitney"].model[0].role == "diagnostic"


def test_wilcoxon_exact_sign_enumeration_removes_zero_pairs_explicitly():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["before", "after"], [
        [5, 3], [7, 4], [4, 4], [9, 8], [10, 6]])
    result, _, output = apply_derived(kernel, sample_id, "wilcoxon", {
        "left": "before", "right": "after"})
    assert output.method == "exact_permutation"
    assert output.sample_sizes == (4,) and output.enumeration_count == 16
    assert output.statistic == 10 and output.p_value == sp.Rational(1, 8)
    assert result.data["verification"]["zero_differences_removed_explicitly"] is True


def test_kruskal_wallis_exact_partition_and_tie_correction():
    kernel = MathKernel(); sample_id = ranked_sample(kernel)
    _, _, output = apply_derived(kernel, sample_id, "kruskal_wallis", {
        "value": "x", "group": "g"})
    assert output.statistic == sp.Rational(27, 7)
    assert output.p_value == sp.Rational(1, 10)
    assert output.sample_sizes == (3, 3) and output.tie_correction == 1


def test_ks_exact_permutation_uses_ecdf_distance():
    kernel = MathKernel(); sample_id = ranked_sample(kernel)
    _, _, output = apply_derived(kernel, sample_id, "ks_2samp", {
        "left": "x", "right": "y"})
    assert output.statistic == 0 and output.p_value == 1
    shifted = make_sample(kernel, ["left", "right"], [
        [1, 6], [2, 7], [3, 8]])
    _, _, separated = apply_derived(kernel, shifted, "ks_2samp", {
        "left": "left", "right": "right"})
    assert separated.statistic == 1 and separated.p_value == sp.Rational(1, 10)


@pytest.mark.parametrize("operation", ["spearman", "kendall"])
def test_exact_association_permutation_for_perfect_order(operation):
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["x", "y"], [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    _, result_id, output = apply_derived(kernel, sample_id, operation, {
        "left": "x", "right": "y"})
    assert output.statistic == 1
    assert output.p_value == sp.Rational(1, 60)
    assert output.enumeration_count == 120
    replay = kernel.apply(result_id, "verify")
    assert replay.ok and replay.status == "verified"


def test_ties_are_ranked_exactly_instead_of_silently_broken():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["g", "x", "y"], [
        [0, 1, 1], [0, 1, 2], [1, 1, 2], [1, 2, 3]])
    _, _, mann = apply_derived(kernel, sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "method": "exact"})
    _, _, rho = apply_derived(kernel, sample_id, "spearman", {
        "left": "x", "right": "y", "method": "exact"})
    assert mann.tie_correction > 0 and mann.p_value.is_Rational
    assert -1 <= rho.statistic <= 1 and rho.p_value.is_Rational


@pytest.mark.parametrize("operation,parameters,expected_method", [
    ("mann_whitney", {"value": "x", "group": "g", "group_a": 0,
                      "group_b": 1}, "asymptotic_normal"),
    ("wilcoxon", {"left": "x", "right": "y"}, "asymptotic_normal"),
    ("kruskal_wallis", {"value": "x", "group": "g"}, "asymptotic_chi_square"),
    ("ks_2samp", {"left": "x", "right": "y"}, "asymptotic_kolmogorov"),
    ("spearman", {"left": "x", "right": "y"}, "asymptotic_student_t"),
    ("kendall", {"left": "x", "right": "y"}, "asymptotic_normal"),
])
def test_every_classical_test_has_an_explicit_asymptotic_path(
        operation, parameters, expected_method):
    kernel = MathKernel()
    rows = [[i % 2, i, (i * 7) % 23] for i in range(30)]
    sample_id = make_sample(kernel, ["g", "x", "y"], rows)
    parameters["method"] = "asymptotic"
    result, result_id, output = apply_derived(kernel, sample_id, operation, parameters)
    assert output.method == expected_method
    assert result.trust == TrustLevel.NUMERIC_HIGH_PRECISION
    bundle = result.claim_evidence[operation]
    assert bundle.numerical[0].metadata["error_bound"] is False
    assert "approximation" in bundle.numerical[0].metadata
    assert kernel.apply(result_id, "verify").status == "verified"


def test_decimal_parameter_ancestry_caps_exact_enumeration_trust():
    kernel = MathKernel()
    sample_id = make_sample(kernel, ["g", "x"], [[0, 1], [0, 2], [1, 4], [1, 5]])
    result, _, output = apply_derived(kernel, sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": "0.0", "group_b": "1.0"})
    assert output.method == "exact_permutation"
    assert result.trust == TrustLevel.NUMERIC and output.input_trust == "numeric"


def test_generic_permutation_exact_result_and_replay():
    kernel = MathKernel(); sample_id = ranked_sample(kernel)
    result, result_id, output = apply_derived(kernel, sample_id, "permutation_test", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "statistic": "difference_in_means", "method": "exact"})
    assert isinstance(output, ResamplingResult)
    assert output.observed_statistic == -5
    assert output.p_value == sp.Rational(1, 10)
    assert output.method == "exact_enumeration" and output.resamples == 20
    assert output.seed is None and output.rng_algorithm is None
    assert result.trust == TrustLevel.EXACT
    assert kernel.apply(result_id, "verify").status == "verified"


def test_monte_carlo_permutation_requires_seed_and_is_replayable():
    kernel = MathKernel(Settings(max_exact_resampling_states=5))
    sample_id = ranked_sample(kernel)
    missing = kernel.apply(sample_id, "permutation_test", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "method": "auto", "resamples": 250})
    assert not missing.ok and "explicit seed" in missing.errors[0]
    parameters = {"value": "x", "group": "g", "group_a": 0, "group_b": 1,
                  "method": "monte_carlo", "resamples": 250, "seed": 12345}
    first, first_id, output = apply_derived(kernel, sample_id, "permutation_test", parameters)
    second, _, second_output = apply_derived(kernel, sample_id, "permutation_test", parameters)
    assert output.p_value == second_output.p_value
    assert output.extreme_count == second_output.extreme_count
    assert output.rng_algorithm == "PCG64" and output.seed == 12345
    assert first.trust == TrustLevel.EMPIRICAL
    required = [item for item in first.claim_evidence["permutation_test"].empirical
                if item.role == "required"]
    assert required[0].replication["seed"] == 12345
    assert kernel.apply(first_id, "verify").status == "verified"


def test_percentile_bootstrap_is_seeded_batched_and_exactly_replayable():
    kernel = MathKernel(Settings(max_resampling_batch_cells=25))
    sample_id = make_sample(kernel, ["x"], [[1], [2], [3], [4], [5]])
    result, result_id, output = apply_derived(kernel, sample_id, "bootstrap", {
        "variable": "x", "statistic": "mean", "confidence_level": "9/10",
        "resamples": 500, "seed": 42})
    assert isinstance(output, ResamplingResult)
    assert output.method == "percentile_bootstrap"
    assert output.observed_statistic == 3
    assert output.interval[0] <= 3 <= output.interval[1]
    assert output.standard_error > 0
    assert output.confidence_level == sp.Float(.9, 15)
    assert output.seed == 42 and output.rng_algorithm == "PCG64"
    assert result.trust == TrustLevel.EMPIRICAL
    replay = kernel.apply(result_id, "verify")
    assert replay.ok and replay.status == "verified"
    assert all(replay.data["verification"].values())


@pytest.mark.parametrize("operation,parameters,message", [
    ("mann_whitney", {"value": "x", "group": "g", "group_a": 0,
                      "group_b": 2}, "outside"),
    ("wilcoxon", {"left": "x", "right": "x"}, "nonzero"),
    ("kruskal_wallis", {"value": "x", "group": "g", "groups": [0, 0]}, "unique"),
    ("ks_2samp", {"left": "missing", "right": "y"}, "stored sample"),
    ("spearman", {"left": "c", "right": "x"}, "constant"),
    ("kendall", {"left": "c", "right": "c"}, "constant"),
    ("bootstrap", {"variable": "x", "resamples": 10}, "seed"),
    ("bootstrap", {"variable": "x", "resamples": 10, "seed": -1}, "uint64"),
])
def test_invalid_nonparametric_and_resampling_requests_fail_closed(
        operation, parameters, message):
    kernel = MathKernel(); sample_id = make_sample(kernel, ["g", "x", "y", "c"], [
        [0, 1, 1, 9], [0, 2, 2, 9], [1, 4, 4, 9], [1, 5, 5, 9]])
    before = len(kernel.math_objects)
    result = kernel.apply(sample_id, operation, parameters)
    assert not result.ok and message in result.errors[0]
    assert len(kernel.math_objects) == before


def test_exact_enumeration_and_resampling_work_limits_preflight():
    exact_kernel = MathKernel(Settings(max_resampling_work=10))
    sample_id = ranked_sample(exact_kernel)
    exact = exact_kernel.apply(sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "method": "exact"})
    assert not exact.ok and "max_resampling_work" in exact.errors[0]
    bootstrap_kernel = MathKernel(Settings(max_resampling_work=100))
    sample_id = make_sample(bootstrap_kernel, ["x"], [[1], [2], [3], [4]])
    boot = bootstrap_kernel.apply(sample_id, "bootstrap", {
        "variable": "x", "resamples": 100, "seed": 1})
    assert not boot.ok and "max_resampling_work" in boot.errors[0]


def test_auto_method_accounts_for_work_budget_before_exact_enumeration():
    kernel = MathKernel(Settings(max_resampling_work=100))
    sample_id = ranked_sample(kernel)
    _, _, mann = apply_derived(kernel, sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "method": "auto"})
    assert mann.method == "asymptotic_normal"
    _, _, permutation = apply_derived(kernel, sample_id, "permutation_test", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1,
        "method": "auto", "resamples": 10, "seed": 23})
    assert permutation.method == "monte_carlo"
    assert permutation.seed == 23 and permutation.resamples == 10


def test_nonparametric_and_resampling_outputs_are_derived_only_and_persistent(tmp_path):
    settings = Settings(store_path=str(tmp_path / "f3.sqlite"))
    kernel = MathKernel(settings); sample_id = ranked_sample(kernel)
    _, test_id, _ = apply_derived(kernel, sample_id, "mann_whitney", {
        "value": "x", "group": "g", "group_a": 0, "group_b": 1})
    _, bootstrap_id, _ = apply_derived(kernel, sample_id, "bootstrap", {
        "variable": "x", "resamples": 100, "seed": 7})
    assert not kernel.object_create("NonparametricTestResult", {}).ok
    assert not kernel.object_create("ResamplingResult", {}).ok
    restarted = MathKernel(settings)
    assert restarted.object_get(test_id).data["sources"] == [sample_id]
    assert restarted.object_get(bootstrap_id).data["sources"] == [sample_id]
    assert restarted.apply(test_id, "verify").status == "verified"
    assert restarted.apply(bootstrap_id, "verify").status == "verified"


def test_nonparametric_capability_limit_and_version_surfaces():
    kernel = MathKernel(); manifest = kernel.capability_query(domain="statistics")
    assert manifest["count"] == 63
    entries = {(item["input_types"][0], item["operation"]): item
               for item in manifest["capabilities"]}
    assert entries[("StatisticalSample", "bootstrap")]["output_types"] == [
        "EngineeringResult", "ResamplingResult"]
    assert entries[("StatisticalSample", "bootstrap")]["trust_levels"] == ["empirical"]
    assert "exact_permutation" in entries[("StatisticalSample", "mann_whitney")]["verification_methods"]
    assert entries[("ResamplingResult", "verify")]["output_types"] == ["EngineeringResult"]
    assert kernel.capabilities()["version"] == __import__("mathkernel").__version__
    for setting in ("max_nonparametric_groups", "max_exact_resampling_states",
                    "max_resamples", "max_resampling_batch_cells",
                    "max_resampling_work"):
        assert setting in Settings().limits()
