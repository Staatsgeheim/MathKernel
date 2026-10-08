"""Operator-controlled import of data-only Lean declaration signatures.

This layer deliberately produces a conservative extraction scaffold.  It binds
the elaborated signature to a pinned project, target, toolchain, executable
pair, and bundled helper source, but it does not claim that pretty-printed Lean
syntax is a normalized semantic contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..models import (
    SHA256,
    TOOLCHAIN,
    FormalProjectSpec,
    FormalTarget,
    PinnedBinary,
    lean_name,
)
from ..replay import binary_path, run_bounded
from ..source import audit_lean_project, bounded_bytes, hash_file, inventory

RECEIPT_SCHEMA = "mathkernel.semantic-correspondence.lean-extraction/v1"
HELPER_SCHEMA = "mathkernel.semantic-correspondence.lean-helper/v1"
HELPER_FILENAME = "MathKernelLeanExtractor.lean"
TARGET_IMPORT_MARKER = "-- MATHKERNEL_TARGET_IMPORT"
DEFAULT_LIMITATIONS = (
    "The bundled helper emits a conservative elaborated-signature scaffold, not a normalized claim contract.",
    "Pretty-printed expressions are toolchain-relative and must not be treated as proof of semantic correspondence.",
    "Operator authorization attests permission to execute; it does not make the project or its elaborators trustworthy.",
    "The bounded subprocess limits wall time and output, but run_bounded is not an OS network or filesystem sandbox.",
)
_FATAL_SCAN_CODES = frozenset({
    "SOURCE_PIN_MISMATCH",
    "TOOLCHAIN_UNPINNED",
    "TOOLCHAIN_MISMATCH",
    "DEPENDENCY_UNPINNED",
    "LOCKFILE_INVALID",
    "TARGET_MODULE_MISSING",
    "SOURCE_UNPARSED",
    "SOURCE_CHANGED",
    "FINDINGS_TRUNCATED",
    "DEPENDENCIES_UNLOCKED",
})


class LeanExtractionBlocked(ValueError):
    """The extractor could not cross its fail-closed execution boundary."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LeanExtractorTools(_Model):
    """Operator-pinned direct executables; elan shims are not accepted."""

    lean: PinnedBinary
    lake: PinnedBinary


class LeanExtractionLimits(_Model):
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_output_bytes: int = Field(default=1024 * 1024, ge=1024, le=8 * 1024**2)
    max_receipt_bytes: int = Field(default=2 * 1024**2, ge=1024, le=16 * 1024**2)
    max_json_depth: int = Field(default=32, ge=4, le=128)
    max_collection_items: int = Field(default=10000, ge=1, le=100000)
    max_string_chars: int = Field(default=1024 * 1024, ge=1, le=8 * 1024**2)


class LeanExtractionRequest(_Model):
    """An operator-owned request. Never populate it from project metadata."""

    project_root: str
    target: FormalTarget
    expected_project_sha256: str
    expected_toolchain: str
    extractor_source_sha256: str
    tools: LeanExtractorTools
    limits: LeanExtractionLimits = Field(default_factory=LeanExtractionLimits)

    @field_validator("expected_project_sha256", "extractor_source_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value

    @field_validator("expected_toolchain")
    @classmethod
    def toolchain(cls, value: str) -> str:
        if not TOOLCHAIN.fullmatch(value):
            raise ValueError("A concrete leanprover/lean4:vX.Y.Z[-rcN] pin is required")
        return value


class LeanBinderReceipt(_Model):
    name: str
    binder_info: Literal["default", "implicit", "strictImplicit", "instImplicit", "unknown"]
    type: str = Field(min_length=1)

    @field_validator("name", "type")
    @classmethod
    def text(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\t\n" for char in value):
            raise ValueError("Control characters are not allowed in helper text")
        return value


class LeanHelperPayload(_Model):
    schema_version: Literal["mathkernel.semantic-correspondence.lean-helper/v1"]
    status: Literal["scaffold"]
    module: str
    declaration: str
    declaration_kind: Literal[
        "axiom", "definition", "theorem", "opaque", "quotient", "inductive", "constructor", "recursor", "unknown"
    ]
    signature: str = Field(min_length=1)
    binders: tuple[LeanBinderReceipt, ...] = Field(max_length=10000)
    conclusion: str = Field(min_length=1)
    referenced_constants: tuple[str, ...] = Field(default=(), max_length=10000)
    opaque_reasons: tuple[str, ...] = Field(min_length=1, max_length=100)

    @field_validator("module", "declaration")
    @classmethod
    def names(cls, value: str) -> str:
        return lean_name(value)

    @field_validator("referenced_constants")
    @classmethod
    def unique_constants(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(values))) != values:
            raise ValueError("referenced_constants must be sorted and duplicate-free")
        return values


class LeanExtractionReceipt(_Model):
    """Versioned, data-only receipt accepted by the correspondence layer."""

    schema_version: Literal["mathkernel.semantic-correspondence.lean-extraction/v1"] = RECEIPT_SCHEMA
    status: Literal["scaffold", "blocked", "timeout", "output_limit", "error"]
    project_sha256: str
    target: FormalTarget
    toolchain: str
    extractor_source_sha256: str
    driver_sha256: str | None = None
    tool_hashes: dict[str, str] = Field(default_factory=dict)
    returncode: int | None = None
    elapsed_seconds: float = Field(default=0, ge=0)
    stdout_sha256: str | None = None
    stderr_sha256: str | None = None
    payload: LeanHelperPayload | None = None
    errors: tuple[str, ...] = Field(default=(), max_length=100)
    limitations: tuple[str, ...] = DEFAULT_LIMITATIONS

    @field_validator(
        "project_sha256", "extractor_source_sha256", "driver_sha256",
        "stdout_sha256", "stderr_sha256",
    )
    @classmethod
    def optional_digest(cls, value: str | None) -> str | None:
        if value is not None and not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value

    @field_validator("toolchain")
    @classmethod
    def concrete_toolchain(cls, value: str) -> str:
        if not TOOLCHAIN.fullmatch(value):
            raise ValueError("Receipt toolchain is not a concrete Lean pin")
        return value

    @field_validator("tool_hashes")
    @classmethod
    def hashes(cls, values: dict[str, str]) -> dict[str, str]:
        if set(values) not in (set(), {"lean", "lake"}):
            raise ValueError("tool_hashes must be empty or contain exactly lean and lake")
        if any(not SHA256.fullmatch(value) for value in values.values()):
            raise ValueError("Invalid executable digest")
        return dict(sorted(values.items()))

    @field_validator("limitations")
    @classmethod
    def fixed_limitations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != DEFAULT_LIMITATIONS:
            raise ValueError("Receipt limitations are protocol-mandated")
        return values

    @model_validator(mode="after")
    def consistent(self) -> LeanExtractionReceipt:
        if self.status == "scaffold":
            if (
                self.payload is None
                or self.payload.status != "scaffold"
                or self.returncode != 0
                or self.errors
                or set(self.tool_hashes) != {"lean", "lake"}
                or not self.driver_sha256
                or not self.stdout_sha256
                or not self.stderr_sha256
            ):
                raise ValueError("A scaffold receipt requires a complete successful extraction binding")
            if (
                self.payload.module != self.target.module
                or self.payload.declaration != self.target.declaration
            ):
                raise ValueError("Helper payload does not match the receipt target")
        else:
            if self.payload is not None:
                raise ValueError("Unsuccessful receipts cannot contain an extracted payload")
            if not self.errors:
                raise ValueError("Unsuccessful receipts require an explicit error")
        return self


def bundled_helper_path() -> Path:
    return Path(__file__).with_name("lean") / HELPER_FILENAME


def bundled_helper_sha256() -> str:
    path = bundled_helper_path()
    return hash_file(path, 2 * 1024**2)[1]


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _bounded_json_value(
    value: Any,
    limits: LeanExtractionLimits,
    *,
    depth: int = 0,
    items: list[int] | None = None,
) -> None:
    if depth > limits.max_json_depth:
        raise ValueError("JSON nesting limit exceeded")
    if items is None:
        items = [0]
    if isinstance(value, str):
        if len(value) > limits.max_string_chars:
            raise ValueError("JSON string limit exceeded")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, dict):
        items[0] += len(value)
        children = value.items()
    elif isinstance(value, list):
        items[0] += len(value)
        children = enumerate(value)
    else:
        raise TypeError("Unsupported JSON value")
    if items[0] > limits.max_collection_items:
        raise ValueError("JSON collection item limit exceeded")
    for key, child in children:
        if isinstance(value, dict):
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            _bounded_json_value(key, limits, depth=depth + 1, items=items)
        _bounded_json_value(child, limits, depth=depth + 1, items=items)


def parse_data_json(data: bytes | str, limits: LeanExtractionLimits | None = None) -> Any:
    """Parse bounded JSON while rejecting duplicate keys and non-finite numbers."""

    limits = limits or LeanExtractionLimits()
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if len(raw) > limits.max_receipt_bytes:
        raise ValueError("JSON byte limit exceeded")
    text = raw.decode("utf-8")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=_reject_constant)
    _bounded_json_value(value, limits)
    return value


def validate_helper_payload(
    data: bytes | str | dict[str, Any],
    *,
    limits: LeanExtractionLimits | None = None,
) -> LeanHelperPayload:
    limits = limits or LeanExtractionLimits()
    value = parse_data_json(data, limits) if isinstance(data, (bytes, str)) else data
    _bounded_json_value(value, limits)
    return LeanHelperPayload.model_validate(value)


def validate_extraction_receipt(
    data: bytes | str | dict[str, Any],
    *,
    limits: LeanExtractionLimits | None = None,
) -> LeanExtractionReceipt:
    limits = limits or LeanExtractionLimits()
    value = parse_data_json(data, limits) if isinstance(data, (bytes, str)) else data
    _bounded_json_value(value, limits)
    return LeanExtractionReceipt.model_validate(value)


def receipt_json(receipt: LeanExtractionReceipt) -> bytes:
    """Return deterministic JSON bytes suitable for hashing or persistence."""

    receipt = LeanExtractionReceipt.model_validate(receipt.model_dump(mode="json"))
    return json.dumps(
        receipt.model_dump(mode="json"),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _base_receipt(request: LeanExtractionRequest, status: str, error: str) -> LeanExtractionReceipt:
    return LeanExtractionReceipt(
        status=status,
        project_sha256=request.expected_project_sha256,
        target=request.target,
        toolchain=request.expected_toolchain,
        extractor_source_sha256=request.extractor_source_sha256,
        errors=(error,),
    )


def _driver_source(helper: str, target: FormalTarget) -> str:
    marker_count = helper.count(TARGET_IMPORT_MARKER)
    if marker_count != 1:
        raise LeanExtractionBlocked("Bundled helper import marker is missing or ambiguous")
    return helper.replace(TARGET_IMPORT_MARKER, f"import {target.module}", 1) + (
        f'\n#mathkernel_extract "{target.module}" {target.declaration}\n'
    )


def extract_lean_signature(
    request: LeanExtractionRequest,
    *,
    authorize_execution: bool = False,
) -> LeanExtractionReceipt:
    """Run the pinned bundled helper after explicit operator authorization.

    No command, path, module, or declaration is read from project metadata.
    The command vector is constructed solely from validated operator input.
    """

    request = LeanExtractionRequest.model_validate(request.model_dump(mode="json"))
    if not authorize_execution:
        return _base_receipt(
            request, "blocked", "Explicit operator execution authorization is required"
        )

    run: dict[str, Any] | None = None
    driver_digest: str | None = None
    verified_hashes: dict[str, str] = {}
    try:
        project, _, project_digest = inventory(request.project_root)
        if project_digest != request.expected_project_sha256:
            raise LeanExtractionBlocked("Lean project fingerprint does not match the operator pin")

        spec = FormalProjectSpec(
            targets=(request.target,),
            expected_toolchain=request.expected_toolchain,
            expected_source_sha256=request.expected_project_sha256,
        )
        scan = audit_lean_project(project, spec)
        fatal = [finding.message for finding in scan.findings if finding.code in _FATAL_SCAN_CODES]
        if fatal:
            raise LeanExtractionBlocked("; ".join(fatal))
        if scan.toolchain != request.expected_toolchain:
            raise LeanExtractionBlocked("Lean project toolchain does not match the operator pin")

        helper_path = bundled_helper_path()
        helper_bytes = bounded_bytes(helper_path, 2 * 1024**2)
        helper_digest = hashlib.sha256(helper_bytes).hexdigest()
        if helper_digest != request.extractor_source_sha256:
            raise LeanExtractionBlocked("Bundled extractor source fingerprint mismatch")
        helper = helper_bytes.decode("utf-8")

        tools = {
            "lean": binary_path(request.tools.lean, (project,)),
            "lake": binary_path(request.tools.lake, (project,)),
        }
        verified_hashes = {
            "lean": request.tools.lean.sha256,
            "lake": request.tools.lake.sha256,
        }
        if tools["lean"].parent != tools["lake"].parent:
            raise LeanExtractionBlocked("Lean and Lake must be direct binaries from one installation")
        lean_lib = tools["lean"].parent.parent / "lib" / "lean" / "Init.olean"
        if not lean_lib.is_file():
            raise LeanExtractionBlocked("Direct Lean installation not found; proxies are not accepted")

        environment = {
            "HOME": "",
            "LANG": "C.UTF-8",
            "PATH": str(tools["lean"].parent),
        }
        if os.name == "nt" and "SystemRoot" in os.environ:
            environment["SystemRoot"] = os.environ["SystemRoot"]

        with tempfile.TemporaryDirectory(prefix="mathkernel-lean-extract-") as temporary:
            temporary_path = Path(temporary)
            environment["HOME"] = str(temporary_path)
            version = run_bounded(
                [str(tools["lean"]), "--version"],
                cwd=temporary_path,
                env=environment,
                timeout=min(15, request.limits.timeout_seconds),
                output_limit=8192,
            )
            wanted = request.expected_toolchain.split(":v", 1)[1]
            if (
                version["status"] != "finished"
                or version["returncode"] != 0
                or not re.search(
                    r"\bversion " + re.escape(wanted) + r"(?:[,\s]|$)",
                    version["stdout"],
                )
            ):
                raise LeanExtractionBlocked("Pinned Lean executable version does not match the project")

            driver = _driver_source(helper, request.target).encode("utf-8")
            driver_digest = hashlib.sha256(driver).hexdigest()
            driver_path = temporary_path / "MathKernelCorrespondenceDriver.lean"
            driver_path.write_bytes(driver)
            run = run_bounded(
                [str(tools["lake"]), "env", str(tools["lean"]), str(driver_path)],
                cwd=project,
                env=environment,
                timeout=request.limits.timeout_seconds,
                output_limit=request.limits.max_output_bytes,
            )

        # Rebind the source tree and executables after project elaboration.
        if inventory(project)[2] != project_digest:
            raise LeanExtractionBlocked("Lean project source changed during extraction")
        binary_path(request.tools.lean, (project,))
        binary_path(request.tools.lake, (project,))

        common = {
            "project_sha256": project_digest,
            "target": request.target,
            "toolchain": request.expected_toolchain,
            "extractor_source_sha256": helper_digest,
            "driver_sha256": driver_digest,
            "tool_hashes": verified_hashes,
            "returncode": run["returncode"],
            "elapsed_seconds": run["elapsed_seconds"],
            "stdout_sha256": _digest_text(run["stdout"]),
            "stderr_sha256": _digest_text(run["stderr"]),
        }
        if run["status"] in {"timeout", "output_limit"}:
            return LeanExtractionReceipt(
                status=run["status"],
                errors=(f"Extractor terminated with status {run['status']}",),
                **common,
            )
        if run["status"] != "finished" or run["returncode"] != 0:
            return LeanExtractionReceipt(
                status="error",
                errors=("Pinned Lean extractor did not complete successfully",),
                **common,
            )
        payload = validate_helper_payload(run["stdout"], limits=request.limits)
        return LeanExtractionReceipt(status="scaffold", payload=payload, **common)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
        fields: dict[str, Any] = {
            "status": "blocked",
            "project_sha256": request.expected_project_sha256,
            "target": request.target,
            "toolchain": request.expected_toolchain,
            "extractor_source_sha256": request.extractor_source_sha256,
            "driver_sha256": driver_digest,
            "tool_hashes": verified_hashes,
            "errors": (str(exc),),
        }
        if run is not None:
            fields.update(
                returncode=run.get("returncode"),
                elapsed_seconds=run.get("elapsed_seconds", 0),
                stdout_sha256=_digest_text(run.get("stdout", "")),
                stderr_sha256=_digest_text(run.get("stderr", "")),
            )
        return LeanExtractionReceipt(**fields)
