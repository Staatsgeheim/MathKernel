# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Shared expression execution contract for the public kernel facade.

Resolve before computing; retain context and literal ancestry after computing.
This module deliberately does not implement algebra. Exact construction and
formal-proof evidence remain distinct from uncertainty in required inputs.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import functools
import inspect
import types

from mathkernel_artifacts import ComputationEvidence, EvidenceBundle
from .models import DerivationStep, MathResult, TrustLevel
from .rendering import render_expr


@dataclass
class OperationFrame:
    kernel: object
    operation: str
    input_ids: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ceiling: TrustLevel | None = None


_FRAME: ContextVar[OperationFrame | None] = ContextVar("mathkernel_operation", default=None)
# Methods whose output describes syntax rather than asserting its value.
# Retrieval still reports the original expression's trust, never a fresh label.
_STRUCTURAL = {"analyze", "infer_structure", "plan", "get_expression", "object_get",
               "get_derivation", "trace_derivation", "capabilities", "capability_query"}
_EXPRESSION_KEYS = {"expr_id", "expr_ids", "expression_id", "density_expression_id",
                    "rhs_id", "rhs_ids", "objective_id", "constraint_ids", "set_id", "set_ids"}


def current_frame(kernel) -> OperationFrame | None:
    frame = _FRAME.get()
    return frame if frame is not None and frame.kernel is kernel else None


def expression_references(arguments: dict) -> list[str]:
    out: list[str] = []
    def visit(value, key=""):
        if isinstance(value, str):
            if (key in _EXPRESSION_KEYS and key not in {"rhs_id", "rhs_ids"}) or ((key.endswith("_id") or key.endswith("_ids")) and value.startswith("expr_")):
                if value not in out:
                    out.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key)
        elif isinstance(value, dict):
            for name, item in value.items():
                visit(item, str(name))
    for name, value in arguments.items():
        if name != "self":
            visit(value, name)
    return out


def expression_trust(kernel, expression_id: str) -> TrustLevel:
    ir, error = kernel._get_expr(expression_id, "provenance")
    if error:
        raise ValueError(error.errors[0])
    record = kernel.expression_provenance.get(expression_id, {})
    return kernel._trust_min(kernel._expr_trust(ir), TrustLevel(record.get("trust", kernel._expr_trust(ir))))


def _cap_bundle(bundle: EvidenceBundle, ceiling: TrustLevel, inputs: list[str]) -> None:
    old = bundle.justified_trust
    from .execution import _TRUST_RANK
    bundle.justified_trust = min((old, ceiling.value), key=lambda t: _TRUST_RANK[TrustLevel(t)]) if old else ceiling.value
    if not any(item.method == "required_expression_ancestry" for item in bundle.computation):
        bundle.computation.append(ComputationEvidence(
            engine="mathir", method="required_expression_ancestry",
            arithmetic="inherited input semantics", trust=ceiling.value,
            metadata={"expression_ids": list(inputs), "not_an_independent_exactness_proof": True}))


def cap_evidence(value, ceiling: TrustLevel | None, inputs: list[str]):
    if ceiling is None:
        return value
    from .execution import _TRUST_RANK
    if _TRUST_RANK[value.trust] > _TRUST_RANK[ceiling]:
        value.trust = ceiling
    _cap_bundle(value.evidence_bundle, ceiling, inputs)
    for bundle in value.claim_evidence.values():
        _cap_bundle(bundle, ceiling, inputs)
    if isinstance(value, MathResult):
        # Reconcile semantic_status as well as the backward-compatible trust.
        value = MathResult.model_validate(value.model_dump(mode="python"))
    return value


def prepare(kernel, bound, operation: str) -> OperationFrame:
    refs = expression_references(bound.arguments)
    frame = OperationFrame(kernel=kernel, operation=operation, input_ids=refs)
    inherited: list[str] = []
    levels = []
    for eid in refs:
        ir, error = kernel._get_expr(eid, "execution_contract")
        if error:
            raise ValueError(error.errors[0])
        provenance = kernel.expression_provenance.get(eid, {})
        if provenance.get("legacy_unverified") and provenance.get("assumptions") and operation not in _STRUCTURAL:
            raise ValueError("Legacy expression has unrecoverable context provenance; re-derive it from the original source and explicit context")
        inherited.extend(provenance.get("context_ids", []))
        frame.assumptions.extend(provenance.get("assumptions", []))
        level = expression_trust(kernel, eid)
        # Symbolic syntax is not uncertain input. Numeric/heuristic ancestry is.
        from .execution import _TRUST_RANK
        if _TRUST_RANK[level] < _TRUST_RANK[TrustLevel.SYMBOLIC]:
            levels.append(level)
    context_id = bound.arguments.get("context_id")
    nested_contexts = []
    def find_contexts(value):
        if isinstance(value, dict):
            if value.get("context_id") is not None:
                nested_contexts.append(value["context_id"])
            for item in value.values(): find_contexts(item)
        elif isinstance(value, (list, tuple)):
            for item in value: find_contexts(item)
    find_contexts({key: value for key, value in bound.arguments.items() if key != "self"})
    contexts = list(dict.fromkeys([*inherited, *nested_contexts, *([context_id] if context_id is not None else [])]))
    loaded = []
    for cid in contexts:
        ctx, error = kernel._get_context(cid, "execution_contract")
        if error:
            raise ValueError(error.errors[0])
        loaded.append(ctx)
    if len(loaded) > 1:
        # A context-dependent simplified handle is not a fresh unconstrained
        # expression. Do not reinterpret it under a different assumption set.
        first = loaded[0].model_dump(exclude={"context_id", "consistency", "derived_facts"})
        if any(ctx.model_dump(exclude={"context_id", "consistency", "derived_facts"}) != first for ctx in loaded[1:]):
            raise ValueError("Incompatible contexts on derived expressions; use the original expression under the desired context")
    if loaded:
        ctx = loaded[-1]
        if ctx.consistency == "inconsistent" and operation not in {"check_context", "infer_context"}:
            raise ValueError("Context is inconsistent; no unconditional result can be derived")
        if operation not in {"check_context", "infer_context"}:
            try:
                from .engines import symbol_env
                symbol_env(sorted(set(ctx.domains) | set(ctx.symbol_properties)), ctx.domains, ctx.symbol_properties)
            except ValueError as exc:
                raise ValueError(f"Inconsistent context assumptions: {exc}") from exc
        frame.context_ids = contexts
        frame.assumptions.extend(render_expr(a.expression) for a in ctx.assumptions)
        if "context_id" in bound.signature.parameters and context_id is None:
            bound.arguments["context_id"] = ctx.context_id
        for assumption in ctx.assumptions:
            if kernel._expr_trust(assumption.expression) == TrustLevel.NUMERIC:
                levels.append(TrustLevel.NUMERIC)
    if operation in {"integrate", "limit", "series", "summation", "product"}:
        from .parser import parse_math
        for key in ("lower", "upper", "point"):
            literal = bound.arguments.get(key)
            if isinstance(literal, str):
                try:
                    if kernel._expr_trust(parse_math(literal)) == TrustLevel.NUMERIC:
                        levels.append(TrustLevel.NUMERIC)
                except ValueError:
                    pass  # The operation reports invalid/unsupported syntax.
    frame.assumptions = list(dict.fromkeys(frame.assumptions))
    frame.ceiling = kernel._trust_min(*levels) if levels else None
    return frame


def persist_expression(kernel, eid: str, step: DerivationStep | None = None, *,
                       frame: OperationFrame | None = None, trust: TrustLevel | None = None) -> None:
    ir = kernel.expressions[eid]
    frame = frame or current_frame(kernel)
    old = kernel.expression_provenance.get(eid, {})
    level = trust or (step.trust if step else TrustLevel(old.get("trust", kernel._expr_trust(ir))))
    level = kernel._trust_min(level, kernel._expr_trust(ir))
    if frame and frame.ceiling is not None:
        level = kernel._trust_min(level, frame.ceiling)
    producer = step.step_id if step else old.get("producer_step_id")
    assumptions = list(dict.fromkeys([*old.get("assumptions", []), *(frame.assumptions if frame else []),
                                       *(step.conditions if step else [])]))
    provenance = {"version": 1, "trust": level.value, "producer_step_id": producer,
                  "context_ids": list(dict.fromkeys([*old.get("context_ids", []), *(frame.context_ids if frame else [])])),
                  "assumptions": assumptions, "input_ids": list(frame.input_ids if frame else (step.inputs if step else []))}
    kernel.expression_provenance[eid] = provenance
    if producer:
        kernel.expression_producers[eid] = producer
    if kernel._store is not None:
        kernel._store.put_expression(eid, {"ir": ir.model_dump(mode="json"),
            "source": kernel.expression_sources.get(eid, render_expr(ir)), "provenance": provenance})


def finalize_step(kernel, step: DerivationStep) -> DerivationStep:
    frame = current_frame(kernel)
    if frame is None:
        return step
    step.conditions = list(dict.fromkeys([*step.conditions, *frame.assumptions]))
    if frame.operation not in _STRUCTURAL:
        step = cap_evidence(step, frame.ceiling, frame.input_ids)
    if step.output_expr_id:
        step.parents = list(dict.fromkeys([*step.parents, *kernel._parents_for_exprs(frame.input_ids)]))
    return step


def finalize_result(kernel, result, frame: OperationFrame):
    if not isinstance(result, MathResult):
        return result
    if kernel._store is not None:
        for cid in frame.context_ids:
            kernel._store.put_context(cid, kernel.contexts[cid].model_dump(mode="json"))
    result.assumptions_used = list(dict.fromkeys([*result.assumptions_used, *frame.assumptions]))
    if frame.context_ids:
        result.data.setdefault("required_context_ids", frame.context_ids)
        ctx = kernel.contexts[frame.context_ids[-1]]
        result.data.setdefault("required_domains", dict(ctx.domains))
    if frame.operation not in _STRUCTURAL:
        result = cap_evidence(result, frame.ceiling, frame.input_ids)
    if result.ok:
        # Covers methods returning several expression handles without a single
        # DerivationStep.output_expr_id (e.g. solve_system/matrix_solve).
        refs = expression_references(result.data)
        for step in result.derivation:
            if step.output_expr_id:
                refs.append(step.output_expr_id)
        for eid in dict.fromkeys(refs):
            if eid in kernel.expressions and eid not in frame.input_ids:
                step = next((s for s in result.derivation if s.output_expr_id == eid),
                            result.derivation[-1] if result.derivation else None)
                persist_expression(kernel, eid, step, frame=frame, trust=result.trust)
    return result


def apply_execution_contract(cls):
    """Install the same pre/postconditions on every public facade operation."""
    for name, fn in list(vars(cls).items()):
        if name.startswith("_") or not isinstance(fn, types.FunctionType):
            continue
        signature = inspect.signature(fn)
        @functools.wraps(fn)
        def wrapper(*args, __fn=fn, __name=name, __signature=signature, **kwargs):
            kernel = args[0]
            parent = current_frame(kernel)
            bound = __signature.bind(*args, **kwargs)
            try:
                frame = prepare(kernel, bound, __name)
            except ValueError as exc:
                result = MathResult(ok=False, status="error", errors=[str(exc)], engine="execution_contract")
                from .output_policy import enforce_output_budget
                return result if parent else enforce_output_budget(kernel, result)
            token = _FRAME.set(frame)
            try:
                from .engines import SolverTimeoutError
                try:
                    result = __fn(*bound.args, **bound.kwargs)
                except SolverTimeoutError as exc:
                    result = MathResult(ok=False, status="error", errors=[str(exc)], engine="isolated_solver")
                result = finalize_result(kernel, result, frame)
            finally:
                _FRAME.reset(token)
            if parent is None:
                from .output_policy import enforce_output_budget
                result = enforce_output_budget(kernel, result)
            return result
        setattr(cls, name, wrapper)
