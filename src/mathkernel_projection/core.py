# =============================================================================
# MathKernel Projection - constructors and MathResult inference
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import hashlib
import json
from typing import Any

from mathkernel_artifacts import SourceRef, Transformation
from .models import MultimodalProjection, validate_payload


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def create_projection(kind: str, payload: dict[str, Any], *, title: str | None = None,
                      source_ref: SourceRef | dict | None = None,
                      source_id: str | None = None, trust: str = "unknown",
                      parameters: dict[str, Any] | None = None,
                      coordinate_names: list[str] | None = None,
                      units: dict[str, str] | None = None,
                      assumptions: list[str] | None = None,
                      evidence_refs: list[str] | None = None,
                      information_loss: list[str] | None = None,
                      information_loss_notes: list[str] | None = None,
                      metadata: dict[str, Any] | None = None) -> MultimodalProjection:
    """Create a canonical multimodal projection with explicit lineage/loss."""
    body = dict(payload)
    validate_payload(kind, body)
    digest = _hash({"kind": kind, "payload": body, "parameters": parameters or {}})
    if source_ref is None:
        src = SourceRef(source_id=source_id or f"projection-source:sha256:{digest}",
                        kind="dataset", sha256=_hash(body), trust=trust,
                        label=title or kind)
    else:
        src = source_ref if isinstance(source_ref, SourceRef) else SourceRef.model_validate(source_ref)
    pid = f"projection:sha256:{digest}"
    transform = Transformation(
        transformation_id=f"{pid}:construct",
        operation=f"project:{kind}",
        inputs=[src.source_id], outputs=[pid],
        parameters=dict(parameters or {}), purpose="projection",
        trust=trust, notes="Presentation projection only; no new mathematical claim.")
    return MultimodalProjection(
        projection_id=pid, kind=kind, title=title or kind.replace("_", " ").title(),
        source_ref=src, payload=body, coordinate_names=list(coordinate_names or []),
        units=dict(units or {}), parameters=dict(parameters or {}),
        transformations=[transform], assumptions=list(assumptions or []),
        evidence_refs=list(evidence_refs or []), trust=trust,
        information_loss=list(information_loss or []),
        information_loss_notes=list(information_loss_notes or []),
        metadata=dict(metadata or {}))


def from_mathresult(result: Any, *, kind: str | None = None,
                    title: str | None = None,
                    parameters: dict[str, Any] | None = None,
                    resolve: Any = None) -> MultimodalProjection:
    """Infer a conservative canonical projection from a MathResult-like object.

    Registered domain adapters (``result_adapters``) get first claim on the
    payload; remaining shapes fall back to shape-based inference.  Inference
    never changes mathematics or silently invents a high-dimensional
    projection.  Call ``create_projection`` with an explicit kind/parameters
    when several representations are possible.
    """
    d = result if isinstance(result, dict) else result.model_dump(mode="json")
    data = dict(d.get("data") or {})
    trust = getattr(d.get("trust", "unknown"), "value", d.get("trust", "unknown"))
    source_id = str(data.get("result_id") or d.get("result_id") or f"math-result:sha256:{_hash(d)}")
    src = SourceRef(source_id=source_id, kind="math_result", result_id=d.get("result_id"),
                    sha256=_hash(d), trust=str(trust), label=title or "MathKernel result")

    if kind is None:
        from .result_adapters import AdapterContext, adapt_result
        ctx = AdapterContext(
            trust=str(trust),
            engine=d.get("engine") or data.get("engine"),
            resolve=resolve)
        spec = adapt_result(data, ctx)
        if spec is not None:
            return create_projection(
                spec.kind, spec.payload, title=title or spec.title,
                source_ref=src, trust=str(trust),
                parameters={**spec.parameters, **dict(parameters or {})},
                coordinate_names=spec.coordinate_names or None,
                units=spec.units or None,
                assumptions=list(d.get("assumptions_used", [])),
                information_loss=list(spec.information_loss),
                information_loss_notes=list(spec.information_loss_notes),
                metadata={"engine": d.get("engine"), "status": d.get("status"),
                          "adapter": "registry"})

    if kind is None:
        if "matrix" in data:
            kind = "matrix"
        elif "grid" in data:
            kind = "scalar_field"
        elif "trajectory" in data:
            kind = "ode_solution"
        elif "eigenvalues" in data or "spectrum" in data:
            kind = "spectrum"
        elif "nodes" in data and ("evidence" in data or "claim_evidence" in d):
            kind = "evidence_graph"
        elif "nodes" in data:
            kind = "graph"
        elif "points" in data:
            dims = len(data["points"][0]) if data["points"] else 0
            kind = "point_cloud" if dims >= 3 else "point_set"
        elif "values" in data:
            kind = "sequence"
        elif "roots" in data and data["roots"] and isinstance(data["roots"][0], dict):
            # Certified enclosures become a region-like object, not a fake
            # point estimate.
            payload = {"values": data["roots"]}
            kind = "set"
            data = payload
        else:
            raise ValueError("could not infer a canonical projection from this MathResult; provide kind explicitly")

    payload = data
    # Normalize a few result-native shapes into canonical payloads.
    if kind == "spectrum" and "spectrum" in payload and "values" not in payload:
        payload = {**payload, "values": payload["spectrum"]}
    if kind == "ode_solution" and "trajectory" in payload and "t" not in payload:
        payload = {"trajectory": payload["trajectory"]}
    return create_projection(kind, payload, title=title, source_ref=src,
                             trust=str(trust), parameters=parameters,
                             assumptions=list(d.get("assumptions_used", [])),
                             metadata={"engine": d.get("engine"), "status": d.get("status")})


def projection_catalog() -> list[dict[str, Any]]:
    """Human/tool-readable catalog of canonical projection families."""
    from .models import REQUIRED_PAYLOAD_KEYS
    descriptions = {
        "scalar_field": "Scalar values over a grid or coordinates.",
        "vector_field": "Vectors anchored at coordinates.",
        "point_set": "Finite 1D/2D point set.",
        "point_cloud": "Finite 3D point cloud.",
        "curve": "Parametric or sampled curve.",
        "function": "Sampled scalar mathematical function.",
        "trajectory": "Ordered path through state/configuration space.",
        "intervals": "Exact or certified interval/enclosure collection; entries accept 'lower'/'upper' or 'lo'/'hi' keys.",
        "surface": "Surface grid or sampled surface.",
        "sequence": "Ordered scalar sequence/time series.",
        "distribution": "PDF/CDF/PMF or empirical distribution.",
        "matrix": "Two-dimensional matrix.",
        "tensor": "Tensor with explicit shape and optional slice.",
        "graph": "Graph nodes and edges.",
        "spectrum": "Ordered spectral coefficients/eigenvalues/frequencies.",
        "region": "Boundary/polygon representation of a region.",
        "implicit_set": "Sampled level/zero set or occupancy grid.",
        "evidence_graph": "Claim/evidence/assumption dependency graph.",
        "mesh": "Vertices and cells/elements.",
        "complex_field": "Complex values represented by magnitude/phase or real/imaginary parts.",
        "optimization": "Feasible geometry, objective values, or optimization trace.",
        "statistical_inference": "Sampling/null/bootstrap distribution and observed statistic.",
        "dynamical_system": "State graph, orbit, or sampled evolution.",
        "pde_solution": "PDE scalar/vector solution sampled in space/time.",
        "ode_solution": "ODE trajectory/state history.",
        "finite_field": "Finite-field/GF(2) matrix, state graph, or ordered values.",
        "relation_geometry": "Relation matrix, information eigensystem, or sensitivity geometry.",
        "prng_analysis": "PRNG lag/harmonic/anomaly research projection.",
        "set": "Finite or bounded set representation.",
        "partition": "Explicit parts/cells of a partition.",
        "piecewise": "Piecewise branches with domains.",
        "expression_tree": "Symbolic expression tree.",
        "certificate_tree": "Proof/certificate dependency tree.",
        "quantity": "Scalar quantity with units/uncertainty metadata.",
        "geometric_complex": "Simplicial/cubical/cell complex.",
        "high_dimensional": "Explicit 1D/2D/3D projection of >3D data; projection method required.",
        "ensemble": "Collection of comparable trajectories/curves/samples.",
    }
    return [{"kind": k, "description": descriptions[k],
             "payload_alternatives": [list(x) for x in v]}
            for k, v in REQUIRED_PAYLOAD_KEYS.items()]
