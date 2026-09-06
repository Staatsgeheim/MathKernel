# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Budget the complete serialized response; preserve large results as resources."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pydantic import BaseModel
from .models import MathResult, TrustLevel


def wire_payload(value):
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def wire_bytes(value) -> bytes:
    # Default JSON separators/ASCII escaping deliberately match ordinary client
    # serialization. UTF-8 bytes, not character count or just result.data.
    return json.dumps(wire_payload(value), ensure_ascii=True, allow_nan=False, default=str).encode("utf-8")


def _save_resource(kernel, payload: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(payload).hexdigest()
    rid = "result_" + digest[:24]
    with kernel._record_lock:
        if not hasattr(kernel, "_result_resources"):
            kernel._result_resources = OrderedDict()
        resources = kernel._result_resources
        previous = resources.pop(rid, None)
        kernel._result_resource_bytes = getattr(kernel, "_result_resource_bytes", 0) - (len(previous) if previous else 0)
        resources[rid] = payload.decode("ascii")
        kernel._result_resource_bytes += len(payload)
        resources.move_to_end(rid)
        # Retain at most 100 results / 64 MiB, but never immediately evict
        # the one result whose receipt is being returned. Configure SQLite for
        # durable paging; in-memory resources expire on eviction/restart.
        while len(resources) > 1 and (len(resources) > 100 or kernel._result_resource_bytes > 64 * 1024 * 1024):
            _, expired = resources.popitem(last=False)
            kernel._result_resource_bytes -= len(expired)
    if kernel._store is not None:
        kernel._store.put_result_resource(rid, {"content": payload.decode("ascii"), "sha256": digest})
    return rid, digest


def enforce_output_budget(kernel, result):
    if not isinstance(result, (BaseModel, dict, list, tuple)):
        return result
    # Non-finite floats were historically emitted by some numerical tools.
    # Preserve the existing serializer for such results, but still count bytes.
    try:
        payload = wire_bytes(result)
    except (TypeError, ValueError):
        payload = json.dumps(wire_payload(result), ensure_ascii=True, default=str).encode("utf-8")
    limit = kernel.settings.max_output_size_bytes
    if len(payload) <= limit:
        return result
    rid, digest = _save_resource(kernel, payload)
    descriptor = {"truncated": True, "resource_id": rid, "size_bytes": len(payload),
                  "limit_bytes": limit, "sha256": digest,
                  "summary": "Full result, assumptions and evidence retained in result_resource_get."}
    if isinstance(result, MathResult):
        descriptor["original_trust"] = result.trust.value
        # A delivery receipt is not the mathematical result, so do not project
        # new exact/formal evidence from the original trust onto the summary.
        small = MathResult(ok=result.ok, status="ok" if result.ok else "error",
            data=descriptor, trust=TrustLevel.UNKNOWN,
            warnings=["Full response exceeded output budget; retrieve the result resource."])
        if not result.ok:
            small.errors = ["Full error details retained in result resource."]
    else:
        small = descriptor
    if len(wire_bytes(small)) > limit:
        target = small.data if isinstance(small, MathResult) else small
        target.pop("summary", None)
        target.pop("sha256", None)
    if len(wire_bytes(small)) > limit:
        raise ValueError(f"max_output_size_bytes={limit} is too small for the minimal response envelope; use at least 1000")
    return small


def result_resource_get(kernel, resource_id: str, offset: int = 0, length: int = 8192) -> dict:
    """Page ASCII JSON bytes; concatenate content fields to reconstruct exactly."""
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (offset, length)) or offset < 0 or length < 1:
        raise ValueError("offset must be a nonnegative integer and length a positive integer")
    content = getattr(kernel, "_result_resources", {}).get(resource_id)
    if content is None and kernel._store is not None:
        record = kernel._store.get_result_resource(resource_id)
        if record:
            content = record["content"]
    if content is None:
        return {"ok": False, "error": "unknown_or_expired_resource", "resource_id": resource_id}
    if offset > len(content):
        raise ValueError("offset exceeds resource length")
    digest = hashlib.sha256(content.encode("ascii")).hexdigest()
    end = min(len(content), offset + length)
    def page(end):
        return {"resource_id": resource_id, "content": content[offset:end], "offset": offset,
                "next_offset": end if end < len(content) else None,
                "total_bytes": len(content), "sha256": digest, "media_type": "application/json"}
    low, high = offset, end
    while low < high:
        mid = (low + high + 1) // 2
        if len(wire_bytes(page(mid))) <= kernel.settings.max_output_size_bytes:
            low = mid
        else:
            high = mid - 1
    result = page(low)
    if len(wire_bytes(result)) > kernel.settings.max_output_size_bytes or (low == offset and offset < len(content)):
        raise ValueError("output budget is too small for a resource page; use at least 1000")
    return result
