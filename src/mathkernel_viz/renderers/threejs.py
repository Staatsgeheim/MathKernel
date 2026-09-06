# =============================================================================
# MathKernel Viz - Three.js scene specifications
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Three.js renderer: builds renderer-neutral 3D scene specs into a
VisualizationDocument.  GL execution happens entirely in the browser viewer;
this module only shapes data."""
from __future__ import annotations

from typing import Sequence

from ..adapters import (points3d_document, surface_document,
                        trajectory3d_document, vector_field3d_document)
from ..document import VisualizationDocument


def phase_space(points: Sequence[Sequence], **kw) -> VisualizationDocument:
    """3D phase-space point cloud."""
    kw.setdefault("title", "Phase space")
    return points3d_document(points, **kw)


def trajectory(states: Sequence[Sequence], **kw) -> VisualizationDocument:
    """Dynamical trajectory through state space."""
    return trajectory3d_document(states, **kw)


def point_cloud(points: Sequence[Sequence], **kw) -> VisualizationDocument:
    """Spectral point cloud / Koopman mode geometry / manifold projection."""
    kw.setdefault("title", "Point cloud")
    return points3d_document(points, **kw)


def surface(grid: Sequence[Sequence], **kw) -> VisualizationDocument:
    """ODE/PDE solution surface, bifurcation surface, optimization landscape."""
    kw.setdefault("title", "Solution surface")
    return surface_document(grid, **kw)


def vector_field(origins: Sequence[Sequence], vectors: Sequence[Sequence],
                 **kw) -> VisualizationDocument:
    return vector_field3d_document(origins, vectors, **kw)
