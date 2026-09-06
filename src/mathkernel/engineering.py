# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Shared immutable engineering models and claim-specific evidence helpers.

This module accepts already parsed mathematical expressions. Public strings
enter through MathKernel's restricted MathIR parser, never a CAS parser.
"""
from __future__ import annotations

from typing import Any
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from mathkernel_artifacts import (CertificateEvidence, ComputationEvidence,
                                  EvidenceBundle, NumericalEvidence)
from mathkernel_artifacts.evidence import TRUST_RANK


def json_math(value):
    if isinstance(value, sp.Basic):
        return sp.sstr(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): json_math(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_math(v) for v in value]
    return value


class EngineeringModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")

    @field_serializer("*", when_used="json", check_fields=False)
    def serialize_math(self, value):
        return json_math(value)


class EngineeringResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    operation: str
    status: str = "candidate"
    trust: str = "unknown"
    value: Any = None
    details: dict[str, Any] = Field(default_factory=dict)
    conditions: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    # Only checks necessary for the requested conclusion go here. Feasibility
    # diagnostics must not accidentally certify an optimizer's global claim.
    verification: dict[str, bool | None] = Field(default_factory=dict)
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)

    @field_serializer("value", "details", when_used="json")
    def serialize_math(self, value):
        return json_math(value)


def cap_trust(*levels):
    return min(levels, key=TRUST_RANK.__getitem__)


def arithmetic_trust(values, input_trust="exact"):
    values = tuple(values)
    if any(v.has(sp.Float) for v in values):
        return cap_trust(input_trust, "numeric_high_precision" if input_trust == "numeric_high_precision" else "numeric")
    return cap_trust(input_trust, "symbolic" if any(v.free_symbols for v in values) else "exact")


def validate_scalars(values, *, real=False):
    for value in values:
        if not isinstance(value, sp.Expr):
            raise ValueError("engineering values must be parsed scalar expressions")
        if value.has(sp.nan, sp.oo, -sp.oo, sp.zoo) or value.is_finite is False:
            raise ValueError("engineering values must be finite")
        if real and value.is_real is not True:
            raise ValueError("this operation requires finite real coefficients")


def checked_result(operation, value, *, method, trust, checks=None,
                   witness=None, details=None, conditions=(), residual=None,
                   precision=None, candidate=False):
    checks = checks or {}
    accepted = bool(checks) and all(v is True for v in checks.values())
    status = "candidate" if candidate else "verified" if accepted else "available"
    bundle = EvidenceBundle(computation=[ComputationEvidence(
        engine="engineering", method=method, arithmetic=trust,
        precision=precision, trust=trust,
    )])
    if witness is not None:
        bundle.certificate.append(CertificateEvidence(
            certificate_type=method, claim=operation, witness=witness,
            verifier=f"{method}_checker", verified=accepted,
            trust=trust if accepted else "unknown",
            role="required" if accepted else "diagnostic",
        ))
    if residual is not None:
        bundle.numerical.append(NumericalEvidence(
            precision=precision, residual=json_math(residual), trust=trust,
            metadata={"error_bound": False},
        ))
    return EngineeringResult(
        operation=operation, status=status, trust=trust, value=value,
        details=details or {}, conditions=list(conditions),
        verification={} if candidate else checks,
        claim_evidence={operation: bundle},
    )


def unsupported(operation, reason):
    return EngineeringResult(operation=operation, status="unsupported", conditions=[reason])


def numeric_array(values, *, real=False):
    import numpy as np
    validate_scalars(values, real=real)
    if any(v.free_symbols for v in values):
        raise ValueError("numeric execution requires concrete values")
    out = np.asarray([float(v) if real else complex(v.evalf(17)) for v in values],
                     dtype=np.float64 if real else np.complex128)
    if not np.all(np.isfinite(out)):
        raise ValueError("input exceeds float64 range; use exact or high-precision execution")
    return out


def sympy_samples(values, digits=17):
    import math
    out = []
    for value in values:
        value = complex(value)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("numerical computation produced non-finite output")
        out.append(sp.Float(value.real, digits) + sp.I * sp.Float(value.imag, digits))
    return tuple(out)


def positive_integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def exact_zero(expression):
    """Decide algebraic equality exactly; failed simplification is not refutation."""
    reduced = sp.simplify(expression)
    if reduced == 0:
        return True
    if reduced.is_algebraic is True:
        return sp.polys.numberfields.to_number_field(reduced).as_expr() == 0
    return reduced.is_zero
