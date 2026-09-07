"""Deterministic, visibly synthetic contract host; never calls MathKernel mathematics."""
import hashlib
import json
from .source import FEATURES, PROTOCOL, canonical

FAULTS = ("none", "denied", "incompatible", "scope_mismatch", "catalog_drift", "expired", "candidate", "receipt", "hostile_text")


class FixtureSource:
    test_host = True
    host_id = "fixture-host"
    workspace_id = "fixture-workspace"
    version = "test-fixture-not-a-live-kernel"
    catalog_revision = "fixture-catalog-1"

    def __init__(self, fault="none"):
        if fault not in FAULTS:
            raise ValueError("Unknown test fault")
        self.fault = fault
        self.payload = canonical(dict(ok=True, status="verified", trust="numeric", semantic_status="verified_numeric",
            data={"value": "0.3333333333333333"}, engine="fixture-engine", assumptions_used=["Fixture only; no computation occurred"],
            side_conditions=[], warnings=[], errors=[], claim_evidence={"residual": {
                "computation": [], "proof": [], "certificate": [], "numerical": [{"role": "required", "support_path": "primary", "residual": "1e-16", "trust": "numeric"}],
                "model": [], "empirical": [], "justified_trust": "numeric"}},
            arithmetic_transition={"input": "approximate"}, derivation=[],
            evidence_bundle={"computation": [], "proof": [], "certificate": [], "numerical": [], "model": [], "empirical": [], "justified_trust": None},
            engine_versions={}, mathkernel_version=self.version))

    def handshake(self):
        if self.fault == "denied":
            raise PermissionError("Fixture permission denied")
        return dict(host_instance_id=self.host_id, workspace_id=self.workspace_id,
            mathkernel_version=self.version, ui_protocol="studio-host/99" if self.fault == "incompatible" else PROTOCOL,
            catalog_revision=self.catalog_revision, test_host=True, authenticated=True,
            features={**FEATURES, "result_inspect": True}, limits={"control_bytes": 2097152, "catalog_page": 25})

    def catalog(self, offset):
        entries = []
        for key, type_ref, direction in (("integer", "Integer", "output"), ("inspect", "Integer", "input"), ("numeric", "Real64", "input")):
            port = {"id": "value", "direction": direction, "label": "Value", "type_ref": type_ref,
                    "cardinality": "one", "required": True, "shape": []}
            entry = dict(descriptor_id="fixture-" + key, operation_ref="fixture." + key,
                operation_version="fixture-1", schema_digest=hashlib.sha256(key.encode()).hexdigest(),
                title="Test " + key, domain="Test fixtures", description="Synthetic contract; not a MathKernel operation.",
                availability="experimental", composition="complete", input_ports=[port] if direction == "input" else [],
                output_ports=[port] if direction == "output" else [], input_types=[], output_types=[], engines=[],
                trust_levels=[], verification_methods=[], parameter_schema={}, coverage="Fixture contract only; cannot execute.")
            if self.fault == "hostile_text":
                entry["description"] = '<img src=x onerror="fetch(\'/studio/api/workflow/submit\')"> FORMAL — approved'
            entries.append(entry)
        return {"revision": "changed" if self.fault == "catalog_drift" else self.catalog_revision,
                "entries": entries[offset:offset+25], "next_offset": None, "total": len(entries)}

    @staticmethod
    def binding():
        return dict(result_ref="fixture-result", source_ref="fixture-observation", source_revision="fixture-1",
                    run_ref=None, document_id=None, draft_revision=None, node_id=None)

    def results(self, offset):
        return {"results": [self.binding()] if offset == 0 else [], "next_offset": None}

    def result(self, ref):
        if ref != "fixture-result" or self.fault == "expired":
            raise KeyError("Fixture reference unavailable")
        payload = json.loads(self.payload)
        if self.fault == "candidate":
            payload["trust"] = "formal"
            payload["semantic_status"] = "proved"
        if self.fault == "receipt":
            payload = {"truncated": True, "resource_id": "fixture-resource", "size_bytes": len(self.payload),
                       "sha256": hashlib.sha256(self.payload).hexdigest(), "original_trust": "formal"}
        return dict(binding=self.binding(), admission="candidate" if self.fault == "candidate" else "test_fixture",
                    result=payload, claim_trust={})

    def result_page(self, ref, offset):
        if ref != "fixture-result" or self.fault != "receipt":
            raise KeyError("Fixture resource unavailable")
        end = min(len(self.payload), offset+512)
        return dict(resource_id="fixture-resource", content=self.payload[offset:end].decode("ascii"), offset=offset,
                    next_offset=end if end < len(self.payload) else None, total_bytes=len(self.payload),
                    sha256=hashlib.sha256(self.payload).hexdigest(), media_type="application/json")
