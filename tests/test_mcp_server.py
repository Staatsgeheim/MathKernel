# =============================================================================
# MathKernel - test mcp server
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import asyncio

import pytest

fastmcp = pytest.importorskip("fastmcp")

from fastmcp import Client

from mathkernel_mcp.server import mcp


def run(coro):
    return asyncio.run(coro)


def test_mcp_tool_surface():
    async def go():
        async with Client(mcp) as client:
            tools = {t.name for t in await client.list_tools()}
            templates = {str(t.uriTemplate) for t in await client.list_resource_templates()}
            prompts = {p.name for p in await client.list_prompts()}
            return tools, templates, prompts

    tools, templates, prompts = run(go())
    expected = {
        "math_capabilities", "math_parse", "math_get", "math_substitute", "math_analyze",
        "math_capability_query", "math_object_create", "math_object_get", "math_apply",
        "math_infer_structure", "math_plan", "math_plan_get", "math_execute_plan", "math_reason",
        "math_execution_get", "math_simplify", "math_solve", "math_prove_equivalence",
        "math_counterexample", "math_interval_evaluate", "math_integer_analyze",
        "math_integer_compute", "math_context_create", "math_context_check", "math_context_infer",
        "math_derivation_get", "math_derivation_trace", "math_codegen", "math_verify_code",
        "math_execute_code",
    }
    assert expected <= tools
    assert "mathkernel://expression/{expr_id}" in templates
    assert "mathkernel://artifact/{artifact_id}" in templates
    assert {"solve_and_codegen", "prove_identity"} <= prompts


def test_mcp_end_to_end_quadratic_workflow():
    async def go():
        async with Client(mcp) as client:
            parsed = await client.call_tool("math_parse", {"expression": "a*x^2 + b*x + c = 0"})
            expr_id = parsed.data["data"]["expr_id"]

            structure = await client.call_tool("math_infer_structure", {"expr_id": expr_id})
            assert structure.data["ok"]

            ctx = await client.call_tool(
                "math_context_create",
                {"domains": {"x": "real", "a": "real", "b": "real", "c": "real"},
                 "assumptions": ["a != 0"]})
            ctx_id = ctx.data["context_id"]

            reasoning = await client.call_tool(
                "math_reason", {"expr_id": expr_id, "context_id": ctx_id,
                                "formal": False, "solve_for": "x"})
            assert reasoning.data["status"] == "verified"

            gen = await client.call_tool(
                "math_codegen", {"expr_id": expr_id, "language": "python",
                                 "target": "solve", "variable": "x", "context_id": ctx_id})
            assert gen.data["ok"]
            assert gen.data["data"]["function_name"] == "solve_for_x"
            assert any(c.startswith("a != 0") for c in gen.data["data"]["constraints"])

            verified = await client.call_tool(
                "math_verify_code", {"artifact_id": gen.data["data"]["artifact_id"]})
            checks = {c["check"]: c["status"] for c in verified.data["data"]["checks"]}
            assert checks["typecheck"] == "passed"
            assert checks["symbolic_roundtrip"] == "passed"

            resource = await client.read_resource("mathkernel://expression/" + expr_id)
            assert expr_id in resource[0].text

    run(go())


def test_mcp_compositional_probability_workflow():
    async def go():
        async with Client(mcp) as client:
            discovered = await client.call_tool("math_capability_query", {
                "input_type": "Distribution",
                "operation": "variance",
                "trust": "symbolic",
                "verification_method": "normalization_check",
            })
            created = await client.call_tool("math_object_create", {
                "object_type": "Distribution",
                "definition": {
                    "family": "normal", "parameters": ["0", "1"],
                    "variable": "x",
                },
            })
            object_id = created.data["data"]["object_id"]
            result = await client.call_tool("math_apply", {
                "object_id": object_id,
                "operation": "variance",
                "parameters": {},
            })
            await client.call_tool("math_apply", {
                "object_id": object_id,
                "operation": "verify",
                "parameters": {},
            })
            restored = await client.call_tool("math_object_get", {
                "object_id": object_id,
            })
            return discovered.data, result.data, restored.data

    discovered, result, restored = run(go())
    assert discovered["count"] == 1
    assert discovered["capabilities"][0]["trust_levels"] == ["symbolic"]
    assert discovered["capabilities"][0]["handler"] == \
        "module:continuous_probability"
    assert result["ok"]
    assert result["data"]["value"] == "1"
    assert result["evidence_bundle"]["computation"]
    # Operations carry their proof; the immutable source retains construction evidence.
    assert restored["status"] == "ok"
    assert restored["semantic_status"] == "candidate"
    assert restored["claim_evidence"]["construction"]["computation"]


def test_mcp_joint_probability_workflow():
    async def go():
        async with Client(mcp) as client:
            parsed = await client.call_tool(
                "math_parse", {"expression": "1"})
            created = await client.call_tool("math_object_create", {
                "object_type": "JointDistribution",
                "definition": {
                    "variables": ["x", "y"],
                    "density_expression_id": parsed.data["data"]["expr_id"],
                    "supports": [["0", "1"], ["0", "1"]],
                },
            })
            result = await client.call_tool("math_apply", {
                "object_id": created.data["data"]["object_id"],
                "operation": "covariance",
                "parameters": {"left": "x", "right": "y"},
            })
            return result.data

    result = run(go())
    assert result["ok"]
    assert result["data"]["value"] == "0"
    assert result["data"]["plan"]["execution"]["obligations"]


def test_mcp_cross_domain_distribution_transform_workflow():
    async def go():
        async with Client(mcp) as client:
            discovered = await client.call_tool(
                "math_capability_query", {
                    "domain": "composition",
                    "input_type": "Distribution",
                    "operation": "integral_transform",
                })
            created = await client.call_tool("math_object_create", {
                "object_type": "Distribution",
                "definition": {
                    "family": "exponential",
                    "parameters": ["2"],
                    "variable": "x",
                },
            })
            result = await client.call_tool("math_apply", {
                "object_id": created.data["data"]["object_id"],
                "operation": "integral_transform",
                "parameters": {
                    "transform": "laplace",
                    "transform_variable": "s",
                    "convention": "laplace_standard",
                },
            })
            return discovered.data, result.data

    discovered, result = run(go())
    assert discovered["count"] == 1
    capability = discovered["capabilities"][0]
    assert capability["handler"] == \
        "composition:distribution_integral_transform"
    assert "integration_complexity" in capability["cost_dimensions"]
    assert result["ok"] and result["data"]["value"] == "2/(s + 2)"
    assert result["data"]["provenance"]["engine_versions"]["sympy"]
