# =============================================================================
# MathKernel - intervals
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import mpmath as mp
from .models import BinaryNode, CallNode, Expr, IntegerNode, NaryNode, NumberNode, RationalNode, RealNode, SymbolNode, UnaryNode


class IntervalEngine:
    name = "mpmath_interval"
    capabilities = {"interval_evaluate", "certified_enclosure"}

    @property
    def available(self) -> bool:
        return True

    def _convert(self, node: Expr, env: dict[str, object]):
        iv = mp.iv
        if isinstance(node, (IntegerNode, RealNode, NumberNode)): return iv.mpf(node.value)
        if isinstance(node, RationalNode): return iv.mpf(node.numerator) / iv.mpf(node.denominator)
        if isinstance(node, SymbolNode):
            if node.name not in env: raise ValueError(f"Missing interval for symbol {node.name}")
            return env[node.name]
        if isinstance(node, UnaryNode): return -self._convert(node.arg, env)
        if isinstance(node, NaryNode):
            vals=[self._convert(x,env) for x in node.args]
            if node.kind == "add":
                out=iv.mpf(0)
                for v in vals: out += v
                return out
            out=iv.mpf(1)
            for v in vals: out *= v
            return out
        if isinstance(node, BinaryNode):
            a,b=self._convert(node.left,env),self._convert(node.right,env)
            return a / b if node.kind == "div" else a ** b
        if isinstance(node, CallNode):
            funcs={"sqrt":iv.sqrt,"sin":iv.sin,"cos":iv.cos,"exp":iv.exp,"log":iv.ln,"abs":abs,
                   "pi":lambda: iv.pi}
            if node.name not in funcs: raise ValueError(f"Interval backend does not support {node.name}")
            return funcs[node.name](*[self._convert(x,env) for x in node.args])
        raise ValueError("Interval evaluation requires an arithmetic expression, not a relation")

    def evaluate(self, node: Expr, bounds: dict[str, tuple[str|float, str|float]], dps: int = 50) -> dict:
        if dps < 15 or dps > 500: raise ValueError("dps must be between 15 and 500")
        old = mp.iv.dps
        mp.iv.dps = dps
        try:
            env={name:mp.iv.mpf([str(lo),str(hi)]) for name,(lo,hi) in bounds.items()}
            result=self._convert(node,env)
            # interval endpoints are interval objects themselves; str() is stable and preserves enclosure.
            return {"enclosure": str(result), "lower": str(result.a), "upper": str(result.b), "dps": dps}
        finally:
            mp.iv.dps = old
