"""Durable, local workflow execution built on existing MathKernel operations.

This package has no dependency on Studio, browser sessions, cloud providers or MCP.
Importing it starts no worker and performs no mathematical operation.
"""
from .runtime import WorkflowRuntime, LocalPolicy
__all__ = ['WorkflowRuntime', 'LocalPolicy']
