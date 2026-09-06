# =============================================================================
# MathKernel Projection - shared typed multimodal projection IR
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Canonical, evidence-preserving projections for visualization/sonification.

A projection is not a new mathematical claim.  It records a deterministic (or
explicitly seeded) representation of an already established mathematical
object together with its source lineage, assumptions, evidence references and
information loss.  Visualization and sonification consume the same IR.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

from mathkernel_artifacts import SourceRef, Transformation, TRUST_RANK

PROJECTION_SCHEMA = "mathkernel-projection/1.0"

ProjectionKind = Literal[
    "scalar_field", "vector_field", "point_set", "point_cloud", "curve",
    "function", "trajectory", "intervals", "surface", "sequence",
    "distribution", "matrix", "tensor", "graph",
    "spectrum", "region", "implicit_set", "evidence_graph", "mesh",
    "complex_field", "optimization", "statistical_inference",
    "dynamical_system", "pde_solution", "ode_solution", "finite_field",
    "relation_geometry", "prng_analysis", "set", "partition", "piecewise",
    "expression_tree", "certificate_tree", "quantity", "geometric_complex",
    "high_dimensional", "ensemble",
]

InformationLoss = Literal[
    "none", "sampling", "aggregation", "slice", "projection", "quantization",
    "ordering", "flattening", "summary", "other",
]


class MultimodalProjection(BaseModel):
    """Renderer-neutral projection of one mathematical object.

    ``payload`` is intentionally declarative JSON-like data.  ``parameters``
    records choices such as tensor slice, graph traversal, coordinate basis,
    PCA matrix, quantiles, scan ordering, or sampling grid.  ``information_loss``
    must describe any lossy step rather than hiding it in a renderer.
    """

    schema_version: str = "1.0"
    projection_schema: str = PROJECTION_SCHEMA
    projection_id: str = ""
    kind: ProjectionKind
    title: str = "MathKernel projection"
    source_ref: SourceRef
    payload: dict[str, Any] = Field(default_factory=dict)
    coordinate_names: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    transformations: list[Transformation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    trust: str = "unknown"
    information_loss: list[InformationLoss] = Field(default_factory=list)
    information_loss_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_projection(self) -> "MultimodalProjection":
        if self.source_ref.trust != "unknown" and self.trust == "unknown":
            self.trust = self.source_ref.trust
        # A presentation layer may preserve or weaken source trust, never
        # silently strengthen it.
        if self.source_ref.trust in TRUST_RANK and self.trust in TRUST_RANK:
            if TRUST_RANK[self.trust] > TRUST_RANK[self.source_ref.trust]:
                self.trust = self.source_ref.trust
        if self.kind == "high_dimensional":
            in_dim = int(self.parameters.get("input_dimension", 0) or 0)
            if in_dim <= 0:
                pts = self.payload.get("points") or []
                if pts and isinstance(pts[0], (list, tuple)):
                    in_dim = len(pts[0])
                    self.parameters["input_dimension"] = in_dim
            out_dim = int(self.parameters.get("output_dimension", 0) or 0)
            if in_dim > 3:
                if out_dim not in (1, 2, 3):
                    raise ValueError("high_dimensional projection requires output_dimension 1, 2, or 3")
                method = str(self.parameters.get("method", "")).strip()
                if not method:
                    raise ValueError("high_dimensional projection requires an explicit method")
                if "projection" not in self.information_loss:
                    raise ValueError("high_dimensional projection must declare information_loss='projection'")
        return self


# Minimum payload keys that make each canonical object meaningful.  Adapters
# accept richer payloads but do not invent missing mathematical structure.
REQUIRED_PAYLOAD_KEYS: dict[str, tuple[tuple[str, ...], ...]] = {
    "scalar_field": (("grid",), ("points", "values")),
    "vector_field": (("origins", "vectors"),),
    "point_set": (("points",),),
    "point_cloud": (("points",),),
    "curve": (("points",), ("x", "y")),
    "function": (("x", "y"), ("values",)),
    "trajectory": (("states",), ("points",)),
    "intervals": (("intervals",),),
    "surface": (("grid",), ("points",)),
    "sequence": (("values",),),
    "distribution": (("x", "pdf"), ("x", "cdf"), ("values",)),
    "matrix": (("matrix",),),
    "tensor": (("values", "shape"), ("slice",)),
    "graph": (("nodes", "edges"), ("nodes",)),
    "spectrum": (("values",), ("frequency", "amplitude")),
    "region": (("boundary",), ("polygons",)),
    "implicit_set": (("points",), ("grid",)),
    "evidence_graph": (("nodes",),),
    "mesh": (("vertices", "cells"),),
    "complex_field": (("magnitude", "phase"), ("real", "imag"), ("points",)),
    "optimization": (("trace",), ("points", "objective"), ("feasible_boundary",)),
    "statistical_inference": (("values",), ("x", "observed"), ("samples",)),
    "dynamical_system": (("nodes", "edges"), ("trajectory",), ("values",)),
    "pde_solution": (("grid",), ("points", "values")),
    "ode_solution": (("t", "states"), ("trajectory",)),
    "finite_field": (("matrix",), ("nodes", "edges"), ("values",)),
    "relation_geometry": (("eigenvalues",), ("matrix",), ("points",)),
    "prng_analysis": (("values",), ("spectrum",), ("matrix",)),
    "set": (("points",), ("boundary",), ("values",)),
    "partition": (("parts",),),
    "piecewise": (("pieces",),),
    "expression_tree": (("nodes",),),
    "certificate_tree": (("nodes",),),
    "quantity": (("value",),),
    "geometric_complex": (("vertices", "cells"), ("nodes", "edges")),
    "high_dimensional": (("points",),),
    "ensemble": (("members",), ("values",)),
}


def validate_payload(kind: str, payload: dict[str, Any]) -> None:
    alternatives = REQUIRED_PAYLOAD_KEYS.get(kind)
    if not alternatives:
        raise ValueError(f"unsupported projection kind {kind!r}")
    if any(all(key in payload for key in required) for required in alternatives):
        return
    expected = " or ".join("+".join(keys) for keys in alternatives)
    raise ValueError(f"projection kind {kind!r} requires payload keys {expected}")
