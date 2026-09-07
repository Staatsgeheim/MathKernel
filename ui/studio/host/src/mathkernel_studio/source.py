"""Read-only projection of existing host facts; no executable workflow semantics."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

PROTOCOL = "studio-host/1"
FEATURES = {key: key in {"catalog_read", "authoring_draft"} for key in (
    "catalog_read", "authoring_draft", "operation_invoke", "workflow_validate",
    "workflow_execute", "run_observe", "result_inspect", "artifact_view",
    "compute_review", "approval_interact")}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def check_id(value: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {"constructor", "prototype", "__proto__"}:
        raise ValueError("Invalid scoped identifier")
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def descriptor(capability: dict, version: str) -> dict:
    """Legacy type lists are semantic hints, never synthesized individual ports."""
    identity = {key: capability.get(key) for key in ("name", "domain", "operation", "input_types", "output_types")}
    schema = capability.get("parameter_json_schema", {})
    digest = hashlib.sha256(canonical(capability)).hexdigest()
    return dict(descriptor_id="cap_" + hashlib.sha256(canonical(identity)).hexdigest()[:24],
        operation_ref=check_id(capability["name"]), operation_version=version, schema_digest=digest,
        title=capability["name"], domain=capability["domain"], description=capability.get("description", ""),
        availability="unknown", composition="partial", input_ports=[], output_ports=[],
        input_types=capability.get("input_types", []), output_types=capability.get("output_types", []),
        engines=capability.get("engines", []), trust_levels=capability.get("trust_levels", []),
        verification_methods=capability.get("verification_methods", []), parameter_schema=schema,
        coverage="Declared discovery hints only. Requiredness, individual ports and mathematical invariants require host validation.")


@dataclass(frozen=True)
class InspectionRecord:
    """Operator-supplied snapshot of an existing result, with explicit source identity.

    Construct via from_result in trusted host code. No browser endpoint accepts a
    record or calls MathResult validation to turn imported labels into evidence.
    """
    binding_json: bytes
    result_json: bytes
    claim_trust_json: bytes
    resource_id: str | None = None

    @classmethod
    def from_result(cls, result, *, result_ref: str, source_ref: str, source_revision: str,
                    run_ref: str | None = None, document_id: str | None = None,
                    draft_revision: int | None = None, node_id: str | None = None):
        from mathkernel.models import MathResult
        if not isinstance(result, MathResult):
            raise TypeError("Only existing host MathResult objects can be admitted")
        for value in (result_ref, source_ref, source_revision, run_ref, document_id, node_id):
            if value is not None:
                check_id(value)
        if draft_revision is not None and (type(draft_revision) is not int or not 0 <= draft_revision <= 2**53 - 1):
            raise ValueError("Invalid frozen draft revision")
        if run_ref is not None and any(v is None for v in (document_id, draft_revision, node_id)):
            raise ValueError("Run-linked results require a complete frozen node mapping")
        payload = result.model_dump(mode="json")
        data = payload.get("data", {})
        resource = data.get("resource_id") if data.get("truncated") is True else None
        if resource is not None:
            check_id(resource)
        # Delegate to the existing evidence model; do not duplicate reconciliation.
        claims = {} if resource else {name: bundle.conservative_trust() for name, bundle in result.claim_evidence.items()}
        return cls(canonical(dict(result_ref=result_ref, source_ref=source_ref, source_revision=source_revision,
            run_ref=run_ref, document_id=document_id, draft_revision=draft_revision, node_id=node_id)),
            canonical(payload), canonical(claims), resource)


class KernelSource:
    test_host = False

    def __init__(self, kernel, *, host_id: str, workspace_id: str, records: tuple[InspectionRecord, ...] = (), version: str | None = None):
        from importlib.metadata import version as package_version
        self.kernel = kernel
        self.host_id, self.workspace_id = check_id(host_id), check_id(workspace_id)
        self.version = version or package_version("mathkernel")
        self.entries: list[dict] = []
        offset = 0
        while True:
            page = kernel.capability_query(offset=offset, limit=25, include_schema=True)
            if not isinstance(page, dict) or "capabilities" not in page:
                raise ValueError("Catalog discovery returned a receipt; increase the host output budget")
            self.entries.extend(descriptor(c, self.version) for c in page["capabilities"])
            following = page["next_offset"]
            if following is None:
                break
            if following <= offset or len(self.entries) > 20000:
                raise ValueError("Invalid catalog paging")
            offset = following
        self.catalog_revision = "catalog_" + hashlib.sha256(canonical(self.entries)).hexdigest()[:24]
        self.records = {json.loads(r.binding_json)["result_ref"]: r for r in records}
        if len(self.records) != len(records):
            raise ValueError("Duplicate result reference")

    def handshake(self) -> dict:
        return dict(host_instance_id=self.host_id, workspace_id=self.workspace_id,
            mathkernel_version=self.version, ui_protocol=PROTOCOL, catalog_revision=self.catalog_revision,
            test_host=self.test_host, authenticated=True,
            features={**FEATURES, "result_inspect": bool(self.records)},
            limits={"control_bytes": 2 * 1024 * 1024, "catalog_page": 25})

    def catalog(self, offset: int) -> dict:
        if offset > len(self.entries):
            raise KeyError("Invalid catalog offset")
        end = min(offset + 25, len(self.entries))
        return {"revision": self.catalog_revision, "entries": self.entries[offset:end],
                "total": len(self.entries), "next_offset": end if end < len(self.entries) else None}

    def results(self, offset: int) -> dict:
        records = list(self.records.values())
        end = min(offset + 25, len(records))
        return {"results": [json.loads(r.binding_json) for r in records[offset:end]],
                "next_offset": end if end < len(records) else None}

    def result(self, ref: str) -> dict:
        record = self.records[ref]
        return dict(binding=json.loads(record.binding_json), admission="host",
                    result=json.loads(record.result_json), claim_trust=json.loads(record.claim_trust_json))

    def result_page(self, ref: str, offset: int) -> dict:
        record = self.records[ref]
        if not record.resource_id:
            raise KeyError("Result has no resource")
        # Reference is resolved from the published record, never a caller-selected resource ID.
        result = self.kernel.result_resource_get(record.resource_id, offset=offset, length=8192)
        if result.get("ok") is False:
            raise KeyError("Result resource expired")
        return result
