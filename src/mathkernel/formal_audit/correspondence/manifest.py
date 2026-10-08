"""Bounded, duplicate-safe loading and inspection of correspondence manifests."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from .models import (
    Attestation,
    ClaimRelation,
    CorrespondenceFinding,
    CorrespondenceManifest,
    CorrespondenceReport,
    EvidenceReference,
    Model,
    TrustLevel,
    canonical_json_bytes,
    canonical_sha256,
    sha256_hex,
)


class ManifestLimits(Model):
    """Resource limits applied before Pydantic validation."""

    max_bytes: int = Field(default=2 * 1024**2, ge=1, le=64 * 1024**2)
    max_depth: int = Field(default=64, ge=1, le=256)
    max_string_chars: int = Field(default=262_144, ge=1, le=4 * 1024**2)
    max_collection_items: int = Field(default=16_384, ge=1, le=1_000_000)
    max_object_keys: int = Field(default=4096, ge=1, le=100_000)
    max_total_nodes: int = Field(default=100_000, ge=1, le=2_000_000)


class LoadedManifest(Model):
    """Validated manifest plus hashes of both submitted and canonical bytes."""

    manifest: CorrespondenceManifest
    source_manifest_sha256: str
    canonical_manifest_sha256: str
    source_size: int = Field(ge=1)

    _digests = field_validator(
        "source_manifest_sha256", "canonical_manifest_sha256"
    )(lambda value: _validated_digest(value))

    @model_validator(mode="after")
    def canonical_digest_matches(self) -> LoadedManifest:
        if canonical_sha256(self.manifest) != self.canonical_manifest_sha256:
            raise ValueError("canonical_manifest_sha256 does not bind the manifest")
        return self


class SourceBinding(Model):
    """Receipt for source bytes supplied independently of the manifest."""

    document_sha256: str
    excerpt_sha256: str
    transcription_sha256: str
    document_verified: bool
    excerpt_verified: bool
    transcription_verified: bool

    _digests = field_validator(
        "document_sha256", "excerpt_sha256", "transcription_sha256"
    )(lambda value: _validated_digest(value))

    @property
    def complete(self) -> bool:
        return (
            self.document_verified
            and self.excerpt_verified
            and self.transcription_verified
        )


def _validated_digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _bounded_nesting(text: str, maximum: int) -> None:
    """Bound JSON nesting before invoking the recursive standard decoder."""
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                raise ValueError("Manifest exceeds max_depth")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("Malformed JSON nesting")
    if in_string or depth != 0:
        raise ValueError("Malformed JSON nesting")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _check_value_bounds(root: Any, limits: ManifestLimits) -> None:
    stack: list[tuple[Any, int]] = [(root, 1)]
    nodes = 0
    collection_items = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > limits.max_total_nodes:
            raise ValueError("Manifest exceeds max_total_nodes")
        if depth > limits.max_depth:
            raise ValueError("Manifest exceeds max_depth")
        if isinstance(value, str):
            if len(value) > limits.max_string_chars:
                raise ValueError("Manifest string exceeds max_string_chars")
        elif isinstance(value, dict):
            if len(value) > limits.max_object_keys:
                raise ValueError("Manifest object exceeds max_object_keys")
            collection_items += len(value)
            for key, child in value.items():
                if len(key) > limits.max_string_chars:
                    raise ValueError("Manifest object key exceeds max_string_chars")
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            collection_items += len(value)
            stack.extend((child, depth + 1) for child in value)
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError("Manifest contains a non-JSON value")
        if collection_items > limits.max_collection_items:
            raise ValueError("Manifest exceeds max_collection_items")


def _coerce_input(data: bytes | bytearray | memoryview | str, maximum: int) -> bytes:
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
    else:
        raise TypeError("Manifest input must be UTF-8 bytes or text")
    if not raw:
        raise ValueError("Manifest is empty")
    if len(raw) > maximum:
        raise ValueError("Manifest exceeds max_bytes")
    return raw


def loads_manifest(
    data: bytes | bytearray | memoryview | str,
    limits: ManifestLimits | None = None,
    *,
    require_reviewed: bool = False,
) -> LoadedManifest:
    """Parse a bounded JSON manifest, rejecting duplicate keys and coercions."""
    limits = limits or ManifestLimits()
    raw = _coerce_input(data, limits.max_bytes)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Manifest must be valid UTF-8") from exc
    _bounded_nesting(text, limits.max_depth)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"Invalid manifest JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Manifest root must be a JSON object")
    _check_value_bounds(value, limits)

    # JSON-mode validation preserves strict scalar behavior while permitting
    # JSON arrays to populate immutable tuple fields.
    canonical_input = canonical_json_bytes(value)
    manifest = CorrespondenceManifest.model_validate_json(canonical_input, strict=True)
    if require_reviewed:
        require_reviewed_manifest(manifest)
    return LoadedManifest(
        manifest=manifest,
        source_manifest_sha256=sha256_hex(raw),
        canonical_manifest_sha256=canonical_sha256(manifest),
        source_size=len(raw),
    )


def _read_regular_file(path: str | Path, maximum: int) -> bytes:
    candidate = Path(path).absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {component}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError("Expected a bounded regular file")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if len(data) > maximum:
        raise ValueError("File exceeds size limit")
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise ValueError("File changed while it was being read")
    return data


def load_manifest(
    path: str | Path,
    limits: ManifestLimits | None = None,
    *,
    require_reviewed: bool = False,
) -> LoadedManifest:
    """Read and parse a regular, non-symlink manifest file exactly once."""
    limits = limits or ManifestLimits()
    return loads_manifest(
        _read_regular_file(path, limits.max_bytes),
        limits,
        require_reviewed=require_reviewed,
    )


def require_reviewed_manifest(
    manifest: CorrespondenceManifest,
) -> tuple[Attestation, ...]:
    """Gate the checked path on explicit, binding reviewer approval."""
    approved = tuple(
        attestation
        for attestation in manifest.specification.reviewer_attestations
        if attestation.decision == "approved"
    )
    rejected = tuple(
        attestation
        for attestation in manifest.specification.reviewer_attestations
        if attestation.decision == "rejected"
    )
    if manifest.review_state != "reviewed" or not approved or rejected:
        raise ValueError(
            "Checked correspondence requires a reviewed manifest with approval and no rejection"
        )
    return approved


def require_audit_ready(
    loaded: LoadedManifest,
    source_binding: SourceBinding,
) -> tuple[Attestation, ...]:
    """Gate comparison on review plus independently bound source material."""
    approved = require_reviewed_manifest(loaded.manifest)
    anchor = loaded.manifest.specification.source
    if (
        source_binding.document_sha256 != anchor.document_sha256
        or source_binding.excerpt_sha256 != anchor.excerpt_sha256
        or source_binding.transcription_sha256 != anchor.transcription_sha256
        or not source_binding.complete
    ):
        raise ValueError(
            "Checked correspondence requires verified document, excerpt, and transcription bytes"
        )
    return approved


def verify_source_binding(
    manifest: CorrespondenceManifest,
    *,
    document_bytes: bytes | bytearray | memoryview | None = None,
    excerpt_bytes: bytes | bytearray | memoryview | None = None,
) -> SourceBinding:
    """Compare independently supplied bytes with all source anchor digests."""
    anchor = manifest.specification.source
    transcription_verified = (
        sha256_hex(anchor.transcription.encode("utf-8"))
        == anchor.transcription_sha256
    )
    document_verified = (
        document_bytes is not None
        and sha256_hex(bytes(document_bytes)) == anchor.document_sha256
    )
    excerpt_verified = (
        excerpt_bytes is not None
        and sha256_hex(bytes(excerpt_bytes)) == anchor.excerpt_sha256
    )
    return SourceBinding(
        document_sha256=anchor.document_sha256,
        excerpt_sha256=anchor.excerpt_sha256,
        transcription_sha256=anchor.transcription_sha256,
        document_verified=document_verified,
        excerpt_verified=excerpt_verified,
        transcription_verified=transcription_verified,
    )


def verify_source_files(
    manifest: CorrespondenceManifest,
    *,
    document_path: str | Path,
    excerpt_path: str | Path,
    max_document_bytes: int = 512 * 1024**2,
    max_excerpt_bytes: int = 2 * 1024**2,
) -> SourceBinding:
    """Read bounded immutable snapshots and verify source/excerpt hashes."""
    if max_document_bytes < 1 or max_excerpt_bytes < 1:
        raise ValueError("Source byte limits must be positive")
    return verify_source_binding(
        manifest,
        document_bytes=_read_regular_file(document_path, max_document_bytes),
        excerpt_bytes=_read_regular_file(excerpt_path, max_excerpt_bytes),
    )


def inspection_report(
    loaded: LoadedManifest,
    *,
    source_binding: SourceBinding | None = None,
) -> CorrespondenceReport:
    """Build a read-only inspection receipt; it never compares formal claims."""
    manifest = loaded.manifest
    spec = manifest.specification
    anchor = spec.source
    findings: list[CorrespondenceFinding] = []
    reviewed = manifest.review_state == "reviewed" and spec.is_reviewed
    if not reviewed:
        findings.append(
            CorrespondenceFinding(
                code="SC_MANIFEST_REVIEW_REQUIRED",
                severity="warning",
                relation=ClaimRelation.UNKNOWN,
                message="Candidate manifest cannot enter the checked correspondence path.",
            )
        )
    if source_binding is None or not source_binding.document_verified:
        findings.append(
            CorrespondenceFinding(
                code="SC_SOURCE_DOCUMENT_UNBOUND",
                severity="warning",
                relation=ClaimRelation.UNKNOWN,
                message="Source document bytes were not independently verified.",
            )
        )
    if source_binding is None or not source_binding.excerpt_verified:
        findings.append(
            CorrespondenceFinding(
                code="SC_SOURCE_EXCERPT_UNBOUND",
                severity="warning",
                relation=ClaimRelation.UNKNOWN,
                message="Source excerpt bytes were not independently verified.",
            )
        )

    evidence = (
        EvidenceReference(
            evidence_id="manifest-inspection",
            kind="MANIFEST_INSPECTED",
            artifact_sha256=loaded.canonical_manifest_sha256,
            trust=TrustLevel.UNKNOWN,
            detail="Schema, bounds, invariants, and canonical digest were inspected.",
        ),
    )
    return CorrespondenceReport(
        mode="inspection",
        status="incomplete",
        relation=ClaimRelation.UNKNOWN,
        trust=TrustLevel.UNKNOWN,
        source_document_sha256=anchor.document_sha256,
        source_excerpt_sha256=anchor.excerpt_sha256,
        source_transcription_sha256=anchor.transcription_sha256,
        source_manifest_sha256=loaded.source_manifest_sha256,
        canonical_manifest_sha256=loaded.canonical_manifest_sha256,
        formal_target=spec.formal_target,
        findings=tuple(findings),
        evidence=evidence,
        limitations=(
            "Inspection validates explicit contracts and fingerprints only; no Lean extraction or comparison was run.",
            "Reviewer attestation records a trust dependency and is not a proof of natural-language fidelity.",
            "Contract alignment and semantic alignment remain not established.",
        ),
    )


def manifest_json_schema() -> dict[str, Any]:
    """Return the standalone correspondence-manifest JSON Schema."""
    return CorrespondenceManifest.model_json_schema()


__all__ = [
    "LoadedManifest",
    "ManifestLimits",
    "SourceBinding",
    "inspection_report",
    "load_manifest",
    "loads_manifest",
    "manifest_json_schema",
    "require_audit_ready",
    "require_reviewed_manifest",
    "verify_source_binding",
    "verify_source_files",
]
