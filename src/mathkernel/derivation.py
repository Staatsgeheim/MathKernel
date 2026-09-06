# =============================================================================
# MathKernel - derivation
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from .models import DerivationStep


def trace_dag(step_id: str, store: dict[str, DerivationStep]) -> dict:
    if step_id not in store:
        raise KeyError(step_id)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    visiting: set[str] = set()

    def walk(sid: str):
        if sid in nodes: return
        if sid in visiting: raise RuntimeError("derivation cycle detected")
        visiting.add(sid)
        step=store[sid]
        nodes[sid]=step.model_dump(mode="json")
        for parent in step.parents:
            if parent in store:
                edges.append({"from":parent,"to":sid})
                walk(parent)
        visiting.remove(sid)

    walk(step_id)
    return {"root":step_id,"nodes":list(nodes.values()),"edges":edges}
