# =============================================================================
# MathKernel Viz - dataset encoding tiers
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Typed-array dataset encoding for portable artifacts.

Tier selection:
    small   -> inline JSON
    medium  -> zlib-compressed base64 typed array
    large   -> chunked zlib+base64 blocks (lazy-decoded by the viewer)

Large numeric arrays are never expanded into millions of JSON numbers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from typing import Iterable

from .document import Dataset

JSON_MAX_ELEMENTS = 512
CHUNK_ELEMENTS = 65536
MAX_DATASET_BYTES = 64 * 1024 * 1024     # hard safety limit per dataset


def _pack(values: list, kind: str) -> bytes:
    fmt = "<%dd" % len(values) if kind == "f64" else "<%dq" % len(values)
    return struct.pack(fmt, *values)


def _unpack(blob: bytes, kind: str) -> list:
    n = len(blob) // 8
    fmt = "<%dd" % n if kind == "f64" else "<%dq" % n
    return list(struct.unpack(fmt, blob))


def _is_int_series(values: list) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def encode_dataset(dataset_id: str, values: Iterable, *, label: str = "",
                   trust: str = "unknown", role: str = "data",
                   source: str = "", shape: list[int] | None = None) -> Dataset:
    vals = list(values)
    if not vals:
        return Dataset(dataset_id=dataset_id, label=label, kind="json",
                       shape=shape or [0], encoding="inline", data=[],
                       trust=trust, role=role, source=source)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        raise TypeError("datasets must be numeric (int/float); use metadata for strings")

    if len(vals) <= JSON_MAX_ELEMENTS:
        raw = json.dumps(vals, separators=(",", ":")).encode()
        return Dataset(dataset_id=dataset_id, label=label, kind="json",
                       shape=shape or [len(vals)], encoding="inline", data=vals,
                       byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                       trust=trust, role=role, source=source)

    kind = "i64" if _is_int_series(vals) else "f64"
    raw = _pack([int(v) if kind == "i64" else float(v) for v in vals], kind)
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError(f"dataset {dataset_id!r} exceeds the {MAX_DATASET_BYTES}-byte limit")
    digest = hashlib.sha256(raw).hexdigest()

    if len(vals) <= CHUNK_ELEMENTS:
        blob = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
        return Dataset(dataset_id=dataset_id, label=label, kind=kind,
                       shape=shape or [len(vals)], encoding="zlib+base64",
                       data=blob, byte_length=len(raw), sha256=digest,
                       trust=trust, role=role, source=source)

    chunks = []
    for off in range(0, len(vals), CHUNK_ELEMENTS):
        block = _pack([int(v) if kind == "i64" else float(v)
                       for v in vals[off:off + CHUNK_ELEMENTS]], kind)
        chunks.append(base64.b64encode(zlib.compress(block, 9)).decode("ascii"))
    return Dataset(dataset_id=dataset_id, label=label, kind=kind,
                   shape=shape or [len(vals)], encoding="zlib+base64+chunked",
                   data=chunks, byte_length=len(raw), sha256=digest,
                   trust=trust, role=role, source=source)


def decode_dataset(ds: Dataset) -> list:
    """Reference decoder (used by tests and by the web exporter)."""
    if ds.encoding == "inline":
        return list(ds.data or [])
    if ds.encoding == "zlib+base64":
        return _unpack(zlib.decompress(base64.b64decode(ds.data)), ds.kind)
    if ds.encoding == "zlib+base64+chunked":
        out: list = []
        for chunk in ds.data:
            out.extend(_unpack(zlib.decompress(base64.b64decode(chunk)), ds.kind))
        return out
    raise ValueError(f"unknown dataset encoding {ds.encoding!r}")


def verify_dataset(ds: Dataset) -> bool:
    """Recompute the payload hash; False means the artifact was altered."""
    if ds.encoding == "inline":
        raw = json.dumps(list(ds.data or []), separators=(",", ":")).encode()
    else:
        raw = _pack([int(v) if ds.kind == "i64" else float(v)
                     for v in decode_dataset(ds)], ds.kind)
    return hashlib.sha256(raw).hexdigest() == ds.sha256
