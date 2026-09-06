# =============================================================================
# MathKernel - context
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from .models import DerivedFact, IntegerNode, MathContext, NumberNode, RationalNode, RealNode, RelationNode, SymbolNode

_ORDER_RELATIONS = {"lt", "le", "gt", "ge"}


def infer_context(ctx: MathContext) -> MathContext:
    facts: list[DerivedFact] = []
    props: dict[str, set[str]] = {k: set(v) for k, v in ctx.symbol_properties.items()}

    for a in ctx.assumptions:
        e = a.expression
        if not isinstance(e, RelationNode):
            continue
        # Ordered comparisons imply an ordered/real interpretation unless a stronger domain exists.
        for side in (e.left, e.right):
            if isinstance(side, SymbolNode) and e.kind in _ORDER_RELATIONS and side.name not in ctx.domains:
                ctx.domains[side.name] = "real"
                facts.append(DerivedFact(subject=side.name, predicate="domain", value="real", provenance=a.provenance))
        # Common unary sign facts against zero.
        if isinstance(e.left, SymbolNode) and isinstance(e.right, (IntegerNode, RealNode, NumberNode)) and e.right.value in {"0", "0.0"}:
            name = e.left.name
            mapping = {"gt": "positive", "ge": "nonnegative", "lt": "negative", "le": "nonpositive", "ne": "nonzero"}
            if e.kind in mapping:
                p = mapping[e.kind]; props.setdefault(name, set()).add(p)
                facts.append(DerivedFact(subject=name, predicate="property", value=p, provenance=a.provenance))
                if p in {"positive", "negative"}:
                    props[name].add("nonzero")
                    facts.append(DerivedFact(subject=name, predicate="property", value="nonzero", provenance=a.provenance))
        if isinstance(e.right, SymbolNode) and isinstance(e.left, (IntegerNode, RealNode, NumberNode)) and e.left.value in {"0", "0.0"}:
            name = e.right.name
            mapping = {"lt": "positive", "le": "nonnegative", "gt": "negative", "ge": "nonpositive", "ne": "nonzero"}
            if e.kind in mapping:
                p = mapping[e.kind]; props.setdefault(name, set()).add(p)
                facts.append(DerivedFact(subject=name, predicate="property", value=p, provenance=a.provenance))
                if p in {"positive", "negative"}:
                    props[name].add("nonzero")
                    facts.append(DerivedFact(subject=name, predicate="property", value="nonzero", provenance=a.provenance))

    ctx.symbol_properties = {k: sorted(v) for k, v in props.items()}
    # Deduplicate deterministic fact keys.
    seen = set(); unique=[]
    for f in facts:
        key=(f.subject,f.predicate,str(f.value),f.source)
        if key not in seen:
            seen.add(key); unique.append(f)
    ctx.derived_facts = unique
    return ctx
