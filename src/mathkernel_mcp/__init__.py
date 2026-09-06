# =============================================================================
# MathKernel - MCP server layer for MathKernel
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""MCP server layer for MathKernel.

The core library lives in the ``mathkernel`` package; this package only adds
the FastMCP tool surface. Imports are re-exported here for backwards
compatibility with code written against mathkernel_mcp <= 0.13.
"""
from mathkernel import (  # noqa: F401
    MathContext,
    MathKernel,
    MathResult,
    ObligationExecution,
    PlanExecution,
    ProblemPlan,
    Settings,
    SolutionSet,
    TrustLevel,
    VerificationStatus,
    ambiguity_diagnostics,
    parse_math,
)
