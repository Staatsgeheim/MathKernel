# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Immutable Python expression handles over the normal kernel contract."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING
from .models import MathResult
if TYPE_CHECKING:
    from .kernel import MathKernel


class KernelOperationError(ValueError):
    def __init__(self, result: MathResult):
        self.result = result
        super().__init__("; ".join(result.errors) or result.status)


def complete_result(kernel: "MathKernel", result: MathResult) -> MathResult:
    """Resolve a delivery receipt without confusing its trust with the result."""
    if not result.data.get("truncated") or "resource_id" not in result.data:
        return result
    chunks, offset, digest = [], 0, None
    while True:
        page = kernel.result_resource_get(result.data["resource_id"], offset)
        if "content" not in page:
            raise ValueError("Result resource is unavailable; cannot recover mathematical evidence")
        chunks.append(page["content"])
        if digest is not None and page["sha256"] != digest:
            raise ValueError("Result resource changed between pages")
        digest = page["sha256"]
        next_offset = page["next_offset"]
        if next_offset is None:
            break
        if next_offset <= offset:
            raise ValueError("Result resource paging did not advance")
        offset = next_offset
    payload = "".join(chunks).encode("ascii")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Result resource integrity check failed")
    return MathResult.model_validate(json.loads(payload))


@dataclass(frozen=True)
class ExpressionHandle:
    """Chaining changes the handle, never the source mathematical expression."""
    kernel: "MathKernel" = field(repr=False, compare=False)
    expression_id: str
    last_result: MathResult = field(repr=False, compare=False)

    @classmethod
    def from_result(cls, kernel: "MathKernel", result: MathResult) -> "ExpressionHandle":
        result = complete_result(kernel, result)
        if not result.ok:
            raise KernelOperationError(result)
        eid = result.data.get("result_expr_id") or result.data.get("expr_id")
        if not eid:
            raise ValueError("Result has no representable expression handle; inspect its MathResult instead")
        return cls(kernel, eid, result)

    @property
    def result(self) -> MathResult:
        return complete_result(self.kernel, self.kernel.get_expression(self.expression_id))

    @property
    def display(self) -> str:
        return self.result.data["display"]

    def simplify(self, mode: str = "simplify", context_id: str | None = None) -> "ExpressionHandle":
        return self.from_result(self.kernel, self.kernel.simplify(self.expression_id, mode, context_id))

    def substitute(self, substitutions: dict[str, str]) -> "ExpressionHandle":
        return self.from_result(self.kernel, self.kernel.substitute(self.expression_id, substitutions))

    def differentiate(self, variable: str, order: int = 1, context_id: str | None = None) -> "ExpressionHandle":
        return self.from_result(self.kernel, self.kernel.differentiate(self.expression_id, variable, order, context_id))

    def solve(self, variable: str, context_id: str | None = None) -> MathResult:
        return complete_result(self.kernel, self.kernel.solve(self.expression_id, variable, context_id))

    def __str__(self) -> str:
        return self.display
