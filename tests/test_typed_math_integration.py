# =============================================================================
# MathKernel - typed compositional mathematics integration tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import time

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.complex_analysis import ComplexAnalysisEngine
from mathkernel.models import MathResult, ResultStatus


def test_transform_problem_is_typed_compositional_and_conditioned():
    kernel = MathKernel()
    context = kernel.create_context(
        {"t": "positive", "s": "positive", "a": "real"}, [])
    expression = kernel.parse("exp(a*t)")
    created = kernel.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
        "context_id": context.context_id,
    })
    assert created.ok and created.data["object_type"] == "TransformProblem"
    result = kernel.apply(created.data["object_id"], "apply")
    assert result.ok and result.trust == TrustLevel.SYMBOLIC
    assert result.data["value"] == "1/(-a + s)"
    assert "s > a" in result.data["roc"]
    assert result.data["plan"]["capability"]["domain"] == "transform"
    assert result.data["plan"]["obligations"][-1]["state"] == "succeeded"
    execution = result.data["plan"]["execution"]
    assert len(execution["obligations"]) == 4
    assert execution["obligations"][1]["evidence_bundle"]["computation"]
    assert result.data["plan"]["dag"]["obligations"][1][
        "resolved_handler"] == "module:integral_transforms"


def test_transform_does_not_launder_decimal_input():
    kernel = MathKernel()
    expression = kernel.parse("0.7*exp(-t)")
    created = kernel.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
    })
    result = kernel.apply(created.data["object_id"], "apply")
    assert result.trust == TrustLevel.NUMERIC
    assert result.evidence_bundle.conservative_trust() == "numeric"


def test_distribution_contract_and_nonexistence_status():
    kernel = MathKernel()
    normal = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"], "variable": "x",
    })
    assert normal.ok
    mean = kernel.apply(normal.data["object_id"], "mean")
    assert mean.ok and mean.data["value"] == "0"
    assert mean.status == "ok"
    assert mean.semantic_status == ResultStatus.CANDIDATE
    cauchy = kernel.object_create("Distribution", {
        "family": "cauchy", "parameters": ["0", "1"],
    })
    absent = kernel.apply(cauchy.data["object_id"], "mean")
    assert absent.ok
    assert absent.semantic_status == ResultStatus.DOES_NOT_EXIST


def test_probability_verification_drives_status_and_obligation_state():
    kernel = MathKernel()
    normal = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"], "variable": "x",
    })
    result = kernel.apply(normal.data["object_id"], "verify")
    assert result.ok and result.status == "verified"
    assert result.semantic_status == ResultStatus.VERIFIED_SYMBOLIC
    verification = next(
        obligation for obligation in result.data["plan"]["obligations"]
        if obligation["action"] == "verify_domain_invariants")
    assert verification["state"] == "verified"
    restored = MathResult.model_validate(result.model_dump(mode="json"))
    assert set(restored.claim_evidence) == {"distribution"}
    assert restored.status == "verified"


def test_typed_objects_and_apply_evidence_survive_store_restart(tmp_path):
    store_path = str(tmp_path / "typed.sqlite")
    first = MathKernel(Settings(store_path=store_path))
    created = first.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"], "variable": "x",
    })
    object_id = created.data["object_id"]
    applied = first.apply(object_id, "pdf")
    root_step = applied.derivation[-1].step_id
    plan_id = applied.data["plan_id"]
    execution_id = applied.data["execution_id"]
    expression = first.parse("exp(-t)")
    transform = first.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
    })
    function_expression = first.parse("1/z")
    function = first.object_create("ComplexFunction", {
        "expression_id": function_expression.data["expr_id"],
        "variable": "z",
    })
    contour = first.object_create("Contour", {
        "vertices": [
            "complex(-1,-1)", "complex(1,-1)", "complex(1,1)",
            "complex(-1,1)", "complex(-1,-1)",
        ],
        "orientation": "ccw",
    })
    first._store.close()

    second = MathKernel(Settings(store_path=store_path))
    restored = second.object_get(object_id)
    assert restored.ok
    assert restored.data["object"]["name"] == "normal"
    # Applying a query must not rewrite the scientific source node.  The
    # distribution keeps its construction evidence across persistence.
    assert set(restored.claim_evidence) == {"construction"}
    assert restored.semantic_status == ResultStatus.CANDIDATE
    assert second.plan_get(plan_id).ok
    assert second.execution_get(execution_id).ok
    assert second.execute_plan(plan_id).status == "verified"
    assert second.apply(object_id, "mean").data["value"] == "0"
    transformed = second.apply(transform.data["object_id"], "apply")
    assert transformed.data["value"] == "1/(s + 1)"
    contour_result = second.apply(
        function.data["object_id"], "contour_integral", {
            "contour_id": contour.data["object_id"],
            "singularities": [{"point": "0", "kind": "pole", "order": 1}],
            "singularities_accounted_for": True,
        })
    assert contour_result.status == "verified"
    replayed = second.replay(root_step)
    assert replayed.ok
    assert replayed.data["steps"][-1]["claim_evidence"]["pdf"]["proof"]
    second._store.close()


def test_complex_contour_integral_requires_and_verifies_accounting():
    kernel = MathKernel()
    expression = kernel.parse("1/z")
    function = kernel.object_create("ComplexFunction", {
        "expression_id": expression.data["expr_id"], "variable": "z",
    })
    contour = kernel.object_create("Contour", {
        "vertices": [
            "complex(-1,-1)", "complex(1,-1)", "complex(1,1)",
            "complex(-1,1)", "complex(-1,-1)",
        ],
        "orientation": "ccw",
    })
    result = kernel.apply(function.data["object_id"], "contour_integral", {
        "contour_id": contour.data["object_id"],
        "singularities": [{"point": "0", "kind": "pole", "order": 1}],
        "singularities_accounted_for": True,
    })
    assert result.ok and result.status == "verified"
    assert result.data["integral"] == "2*I*pi"
    assert "independently matched" in result.warnings[0]


def test_capability_registry_exposes_typed_math_objects():
    kernel = MathKernel()
    found = kernel.capability_query(
        domain="transform", object_type="TransformProblem", operation="apply")
    assert found["count"] == 1
    assert "symbolic_identity" in found["capabilities"][0]["evidence"]
    assert "numeric_cross_check" not in found["capabilities"][0]["evidence"]
    assert found["capabilities"][0]["parameter_schema"] == {
        "verify": "boolean?"}
    capabilities = kernel.capabilities()
    assert capabilities["prototype"] is False
    assert capabilities["limits"]["max_joint_dimensions"] == 8
    assert capabilities["limits"]["max_symbolic_series_order"] == 128
    assert capabilities["limits"]["max_obligation_steps"] == 128


def test_composed_distribution_and_complex_operations_use_registry_plan():
    kernel = MathKernel()
    normal = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"],
    })
    truncated = kernel.apply(normal.data["object_id"], "truncate", {
        "lower": "-1", "upper": "1",
    })
    assert truncated.ok and truncated.data["object_type"] == "Distribution"
    assert truncated.data["plan"]["capability"]["operation"] == "truncate"

    expression = kernel.parse("z^2-1")
    function = kernel.object_create("ComplexFunction", {
        "expression_id": expression.data["expr_id"], "variable": "z",
    })
    zeros = kernel.apply(function.data["object_id"], "zeros")
    assert zeros.ok and zeros.data["value"] == "{-1, 1}"
    assert zeros.data["plan"]["obligations"][2]["state"] == "verified"


def test_complex_operations_honor_solver_timeout(monkeypatch):
    kernel = MathKernel(Settings(solver_timeout_seconds=0.01))
    expression = kernel.parse("1/z")
    function = kernel.object_create("ComplexFunction", {
        "expression_id": expression.data["expr_id"], "variable": "z",
    })

    def slow_residue(self, function, point):
        time.sleep(0.05)
        raise AssertionError("timeout should return before this result")

    monkeypatch.setattr(ComplexAnalysisEngine, "residue", slow_residue)
    result = kernel.apply(function.data["object_id"], "residue", {"point": "0"})
    assert not result.ok
    assert "solver_timeout_seconds" in result.errors[0]


def test_transform_solve_skips_verification_only_when_not_requested():
    kernel = MathKernel()
    expression = kernel.parse("exp(-t)")
    created = kernel.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
    })
    candidate = kernel.apply(created.data["object_id"], "solve")
    checked = kernel.apply(
        created.data["object_id"], "solve", {"verify": True})
    assert candidate.semantic_status == ResultStatus.CANDIDATE
    assert not candidate.data["verified_checks"]
    assert "explicitly skipped" in candidate.side_conditions[-1]
    assert checked.data["verified_checks"]
    assert checked.data["verified_checks"][0][
        "name"] == "forward_inverse_round_trip"
    assert checked.semantic_status == ResultStatus.CANDIDATE


def test_complex_domain_argument_continuation_and_conformal_map():
    kernel = MathKernel()
    capability = kernel.capability_query(
        input_type="ComplexFunction", operation="argument_principle")
    contour_mismatch = kernel.capability_query(
        input_type="Contour", operation="contour_integral")
    domain = kernel.object_create("ComplexDomain", {
        "variable": "z", "region": "complexes",
    })
    expression = kernel.parse("z^2 - 1")
    function = kernel.object_create("ComplexFunction", {
        "expression_id": expression.data["expr_id"],
        "variable": "z",
        "domain_id": domain.data["object_id"],
    })
    contour = kernel.object_create("Contour", {
        "vertices": [
            "complex(-2,-2)", "complex(2,-2)", "complex(2,2)",
            "complex(-2,2)", "complex(-2,-2)",
        ],
        "orientation": "ccw",
    })
    count = kernel.apply(function.data["object_id"], "argument_principle", {
        "contour_id": contour.data["object_id"],
    })
    continuation = kernel.apply(
        function.data["object_id"], "analytic_continuation", {
            "target_domain_id": domain.data["object_id"],
        })
    linear_expression = kernel.parse("z + 1")
    linear = kernel.object_create("ComplexFunction", {
        "expression_id": linear_expression.data["expr_id"],
        "variable": "z",
        "domain_id": domain.data["object_id"],
    })
    mapped = kernel.apply(linear.data["object_id"], "conformal_map", {
        "source_domain_id": domain.data["object_id"],
    })
    assert count.data["zero_minus_pole"] == 2
    assert continuation.data["identity_established"] is True
    assert mapped.data["nondegenerate_on_domain"] is True
    assert capability["capabilities"][0]["output_types"] == [
        "ArgumentPrincipleResult"]
    assert contour_mismatch["count"] == 0


def test_joint_and_conditional_distribution_composition():
    kernel = MathKernel()
    density = kernel.parse("1")
    joint = kernel.object_create("JointDistribution", {
        "variables": ["x", "y"],
        "density_expression_id": density.data["expr_id"],
        "supports": [["0", "1"], ["0", "1"]],
    })
    assert joint.ok, joint.errors
    verified = kernel.apply(joint.data["object_id"], "verify")
    covariance = kernel.apply(joint.data["object_id"], "covariance", {
        "left": "x", "right": "y",
    })
    marginal = kernel.apply(joint.data["object_id"], "marginal", {
        "variable": "x",
    })
    conditional = kernel.apply(joint.data["object_id"], "condition", {
        "variable": "x", "given": {"y": "1/2"},
    })
    conditional_mean = kernel.apply(
        conditional.data["object_id"], "mean")
    order = kernel.apply(joint.data["object_id"], "order_statistic", {
        "variable": "x", "sample_size": 3, "order": 2,
    })
    assert verified.data["verification"] == {
        "normalized": True, "nonnegative": True}
    assert covariance.data["value"] == "0"
    assert marginal.data["object_type"] == "Distribution"
    assert conditional.data["object_type"] == "ConditionalDistribution"
    assert conditional_mean.data["value"] == "1/2"
    assert order.data["object_type"] == "Distribution"


def test_math_object_limit_uses_existing_settings_contract():
    kernel = MathKernel(Settings(max_math_objects=1))
    first = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["0", "1"],
    })
    second = kernel.object_create("Distribution", {
        "family": "normal", "parameters": ["1", "1"],
    })
    assert first.ok
    assert not second.ok
    assert "math object limit (1) reached" in second.errors[0]


def test_distribution_composes_into_conditioned_integral_transform():
    kernel = MathKernel()
    distribution = kernel.object_create("Distribution", {
        "family": "exponential", "parameters": ["2"], "variable": "x",
    })
    result = kernel.apply(
        distribution.data["object_id"], "integral_transform", {
            "transform": "laplace",
            "transform_variable": "s",
            "convention": "laplace_standard",
        })
    assert result.ok
    assert result.data["value"] == "2/(s + 2)"
    assert result.data["plan"]["capability"]["domain"] == "composition"
    assert any(
        item.method == "distribution_density_source"
        for item in result.claim_evidence["result"].computation
    )


def test_typed_resource_limits_are_enforced_before_symbolic_work():
    kernel = MathKernel(Settings(
        max_contour_vertices=3,
        max_joint_dimensions=1,
        max_order_statistic_sample_size=2,
    ))
    contour = kernel.object_create("Contour", {
        "vertices": ["0", "1", "0", "1"],
    })
    density = kernel.parse("1")
    joint = kernel.object_create("JointDistribution", {
        "variables": ["x", "y"],
        "density_expression_id": density.data["expr_id"],
        "supports": [["0", "1"], ["0", "1"]],
    })
    uniform = kernel.object_create("Distribution", {
        "family": "uniform", "parameters": ["0", "1"],
    })
    order = kernel.apply(uniform.data["object_id"], "order_statistic", {
        "sample_size": 3, "order": 1,
    })
    assert not contour.ok and "max_contour_vertices=3" in contour.errors[0]
    assert not joint.ok and "max_joint_dimensions=1" in joint.errors[0]
    assert not order.ok and "sample_size must be in 1..2" in order.errors[0]


def test_typed_results_record_versions_and_arithmetic_transition():
    kernel = MathKernel()
    expression = kernel.parse("0.5*exp(-t)")
    problem = kernel.object_create("TransformProblem", {
        "expression_id": expression.data["expr_id"],
        "transform": "laplace",
        "variable": "t",
        "transform_variable": "s",
        "convention": "laplace_standard",
    })
    result = kernel.apply(problem.data["object_id"], "apply")
    provenance = result.data["provenance"]
    assert provenance["engine_versions"]["mathkernel"] == __import__("mathkernel").__version__
    assert provenance["engine_versions"]["sympy"]
    assert provenance["arithmetic_transition"] == {
        "input_trust": "numeric",
        "computation": "symbolic",
        "required_input_trust": "numeric",
        "justified_output_trust": "numeric",
    }
    assert any(
        item.metadata.get("engine_versions", {}).get("sympy")
        for item in result.evidence_bundle.computation
    )
