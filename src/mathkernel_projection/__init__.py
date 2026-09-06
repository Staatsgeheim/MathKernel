# =============================================================================
# MathKernel Projection - shared multimodal projection layer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Shared typed projection objects consumed by mathkernel-viz and sonification."""
from .models import (PROJECTION_SCHEMA, InformationLoss, MultimodalProjection,
                     ProjectionKind, REQUIRED_PAYLOAD_KEYS, validate_payload)
from .core import create_projection, from_mathresult, projection_catalog
from .result_adapters import (AdapterContext, ProjectionSpec, adapt_result,
                              register_engine_adapter, register_model_adapter,
                              register_shape_adapter)

__all__ = [
    "PROJECTION_SCHEMA", "ProjectionKind", "InformationLoss",
    "MultimodalProjection", "REQUIRED_PAYLOAD_KEYS", "validate_payload",
    "create_projection", "from_mathresult", "projection_catalog",
    "AdapterContext", "ProjectionSpec", "adapt_result",
    "register_engine_adapter", "register_model_adapter",
    "register_shape_adapter",
]
