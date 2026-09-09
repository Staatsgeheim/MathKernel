"""Typed, claim-scoped results for external Lean project audits.

A source scan is never a proof. Comparator acceptance is relative to the
operator's trusted reference, toolchain and explicit axiom policy.
"""
from __future__ import annotations

import re
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STANDARD_AXIOMS = ("propext", "Quot.sound", "Classical.choice")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\Z")
TOOLCHAIN = re.compile(r"leanprover/lean4:v\d+\.\d+\.\d+(?:-rc\d+)?\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def lean_name(value: str) -> str:
    if not NAME.fullmatch(value):
        raise ValueError("Expected a supported, injection-safe qualified Lean identifier")
    return value


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuditLimits(Model):
    max_files: int = Field(default=20000, ge=1, le=200000)
    max_file_bytes: int = Field(default=16 * 1024**2, ge=1, le=1024**3)
    max_total_bytes: int = Field(default=512 * 1024**2, ge=1, le=32 * 1024**3)
    max_findings: int = Field(default=2000, ge=1, le=100000)


class FormalTarget(Model):
    module: str
    declaration: str
    _names = field_validator("module", "declaration")(lean_name)


class FormalProjectSpec(Model):
    targets: tuple[FormalTarget, ...] = Field(default=(), max_length=100)
    expected_toolchain: str | None = None
    expected_source_sha256: str | None = None
    comparator_configs: tuple[str, ...] = Field(default=(), max_length=20)
    permitted_axioms: tuple[str, ...] = STANDARD_AXIOMS

    @field_validator("expected_toolchain")
    @classmethod
    def toolchain(cls, value):
        if value is not None and not TOOLCHAIN.fullmatch(value):
            raise ValueError("A concrete leanprover/lean4:vX.Y.Z[-rcN] pin is required")
        return value

    @field_validator("expected_source_sha256")
    @classmethod
    def digest(cls, value):
        if value is not None and not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value

    @field_validator("targets", "comparator_configs", "permitted_axioms")
    @classmethod
    def unique(cls, values):
        keys = [v.model_dump_json() if isinstance(v, Model) else v for v in values]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate entries are not allowed")
        return values

    @field_validator("permitted_axioms")
    @classmethod
    def axioms(cls, values):
        for value in values:
            lean_name(value)
        if "sorryAx" in values or "Lean.ofReduceBool" in values:
            raise ValueError("Proof placeholders/native-evaluation axioms are not permitted")
        return values


class SourceFile(Model):
    path: str
    size: int
    sha256: str


class AuditFinding(Model):
    code: str
    severity: Literal["info", "warning", "error"]
    path: str
    line: int = 0
    message: str
    scope: str = "source_only"


class FormalAuditReport(Model):
    schema_version: str = "mathkernel.formal-audit/v1"
    status: Literal["inspected", "issues_found", "incomplete"]
    trust: Literal["unknown"] = "unknown"
    theorem_verification: Literal["not_run"] = "not_run"
    semantic_alignment: Literal["not_established"] = "not_established"
    source_sha256: str
    source_files: tuple[SourceFile, ...]
    toolchain: str | None
    local_imports: dict[str, tuple[str, ...]]
    target_import_closure: tuple[str, ...]
    unresolved_imports: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    dependency_pins: tuple[dict, ...] = ()
    limitations: tuple[str, ...] = (
        "Lexical source inspection is not Lean elaboration or a proof-dependency audit.",
        "A placeholder in a reachable module is not necessarily used by a target theorem.",
        "External dependency source and compiled artifacts are outside this source inventory.",
        "No project code, Lean, Lake, shell, or network operation was executed.",
        "Hash equality binds inspected bytes; it does not establish mathematical validity.",
    )


class PinnedBinary(Model):
    path: str
    sha256: str

    @field_validator("sha256")
    @classmethod
    def digest(cls, value):
        if not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value


class ComparatorTools(Model):
    lean: PinnedBinary
    lake: PinnedBinary
    comparator: PinnedBinary
    landrun: PinnedBinary
    lean4export: PinnedBinary
    nanoda: PinnedBinary
    systemd_run: PinnedBinary
    systemctl: PinnedBinary
    env: PinnedBinary


class ComparatorRequest(Model):
    """Operator-only replay request. Never populated from a submission's metadata.

    reference_root is a separately reviewed workspace containing the challenge,
    trusted Lake configuration and preinstalled, trusted dependency caches.
    Its fingerprint is provided by the operator, not inferred as trustworthy.
    """
    submission_root: str
    reference_root: str
    reference_tree_sha256: str
    config: str
    tools: ComparatorTools
    timeout_seconds: int = Field(default=600, ge=1, le=7200)
    memory_bytes: int = Field(default=4 * 1024**3, ge=128 * 1024**2, le=64 * 1024**3)
    max_output_bytes: int = Field(default=8 * 1024**2, ge=1024, le=64 * 1024**2)
    spec: FormalProjectSpec

    @field_validator("reference_tree_sha256")
    @classmethod
    def digest(cls, value):
        if not SHA256.fullmatch(value):
            raise ValueError("Expected a lowercase SHA-256 digest")
        return value


class ComparatorReport(Model):
    schema_version: str = "mathkernel.comparator-replay/v1"
    status: Literal["accepted", "not_accepted", "blocked", "timeout", "output_limit", "error"]
    trust: Literal["unknown", "formal"] = "unknown"
    claim: str = "listed_theorems_match_trusted_Lean_reference_under_permitted_axioms"
    semantic_alignment: Literal["not_established"] = "not_established"
    independent_kernel: Literal["not_run", "required_by_pinned_comparator", "accepted"] = "not_run"
    source_sha256: str | None = None
    reference_tree_sha256: str
    targets: tuple[FormalTarget, ...]
    permitted_axioms: tuple[str, ...]
    tool_hashes: dict[str, str] = Field(default_factory=dict)
    command: tuple[str, ...] = ()
    returncode: int | None = None
    elapsed_seconds: float = 0
    stdout: str = ""
    stderr: str = ""
    log_sha256: str | None = None
    errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (
        "Acceptance is conditional on the operator-trusted reference, dependencies, binaries and OS sandbox.",
        "A compiler/verifier rejection or timeout is not a mathematical disproof.",
        "No equivalence to a paper, physical model or Millennium problem is inferred.",
    )

    @model_validator(mode="after")
    def consistent_claim(self):
        if self.status == "accepted":
            if (self.trust != "formal" or self.independent_kernel != "accepted"
                    or self.returncode != 0 or not self.targets or self.errors
                    or not self.source_sha256 or not self.log_sha256 or not self.tool_hashes):
                raise ValueError("An acceptance requires a successful, bound, independent-checker receipt")
        elif self.trust != "unknown" or self.independent_kernel == "accepted":
            raise ValueError("An unaccepted run cannot claim formal evidence")
        return self
