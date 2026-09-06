# =============================================================================
# MathKernel Viz - automatic view/renderer selection
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Choose a sensible view from a MathResult's shape.  The user may always
override; "auto" never silently upgrades trust."""
from __future__ import annotations

from typing import Any

_HEATMAP_MIN = 16          # matrices at least this many cells -> heatmap
_CLOUD_3D_MIN_DIM = 3


def select_view(result: Any) -> str:
    d = result if isinstance(result, dict) else result.model_dump(mode="json")
    data = d.get("data", {})
    if d.get("derivation") and not data:
        return "dag"
    if isinstance(data.get("matrix"), list) and data["matrix"]:
        m = data["matrix"]
        if len(m) * (len(m[0]) if m else 0) >= _HEATMAP_MIN:
            return "heatmap"
    if isinstance(data.get("roots"), list) and data["roots"] \
            and isinstance(data["roots"][0], dict) and "lower" in data["roots"][0]:
        return "intervals"
    pts = data.get("points")
    if isinstance(pts, list) and pts:
        return "point_cloud_3d" if len(pts[0]) >= _CLOUD_3D_MIN_DIM else "plot2d"
    if isinstance(data.get("trajectory"), list):
        return "trajectory_3d"
    if isinstance(data.get("grid"), list):
        return "surface_3d"
    if d.get("derivation"):
        return "dag"
    return "summary"


def select_renderer(view: str) -> str:
    return {
        "dag": "svg",
        "plot2d": "svg",
        "intervals": "svg",
        "heatmap": "svg",
        "point_cloud_3d": "threejs",
        "trajectory_3d": "threejs",
        "surface_3d": "threejs",
        "vector_field_3d": "threejs",
    }.get(view, "svg")
