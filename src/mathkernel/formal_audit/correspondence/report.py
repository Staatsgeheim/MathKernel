"""Evidence-bound assembly for semantic-correspondence reports."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from mathkernel_artifacts import CertificateEvidence, ComputationEvidence, EvidenceBundle

from .compare import ComparisonResult, compare_contracts, verify_comparison_result
from .lean_extract import LeanExtractionReceipt
from .manifest import (
    LoadedManifest,
    SourceBinding,
    inspection_report,
    load_manifest,
    require_reviewed_manifest,
    verify_source_files,
)
from .models import (
    ALL_CHECKS,
    CheckKind,
    ClaimContract,
    ClaimExpression,
    ClaimRelation,
    CorrespondenceFinding,
    CorrespondenceManifest,
    CorrespondenceReport,
    CorrespondenceSpec,
    EvidenceReference,
    ExpressionKind,
    FieldComparison,
    TrustLevel,
    canonical_sha256,
    canonical_json_bytes,
    sha256_hex,
)

LIMITATIONS = (
    "Correspondence is exact only relative to the reviewed manifest and extracted contract.",
    "Reviewer attestation is not a mechanical proof that the source prose means the manifest.",
    "Lean acceptance or replay cannot establish natural-language semantic alignment.",
    "Unsupported or opaque required material prevents equivalence.",
)


def _evidence_id(kind: str, digest: str) -> str:
    return kind.lower().replace("_", "-") + "-" + digest[:16]


def _opaque_contract(manifest: CorrespondenceManifest, reason: str) -> ClaimContract:
    """Represent an extractor receipt which cannot enter the supported fragment."""
    return ClaimContract(
        claim_id=manifest.specification.source_contract.claim_id + ".formal",
        conclusion=ClaimExpression(kind=ExpressionKind.OPAQUE, opaque_reason=reason),
    )


def _formal_contract_from_extractor(
    extractor: Callable[..., Any] | None,
    *,
    project_root: str | Path,
    formal_spec: Any,
    manifest: CorrespondenceManifest,
) -> tuple[ClaimContract, str, dict[str, str], tuple[str, ...]]:
    if extractor is None:
        raise ValueError("Checked correspondence requires an operator-controlled extractor")
    value = extractor(
        project_root=project_root,
        formal_spec=formal_spec,
        formal_target=manifest.specification.formal_target,
    )
    if isinstance(value, ClaimContract):
        digest = canonical_sha256(value)
        return value, digest, {"extractor": digest}, ()
    if isinstance(value, tuple) and len(value) == 2:
        contract = (
            value[0]
            if isinstance(value[0], ClaimContract)
            else ClaimContract.model_validate_json(canonical_json_bytes(value[0]))
        )
        metadata = dict(value[1])
        digest = canonical_sha256(contract)
        tools = {str(k): str(v) for k, v in metadata.get("tool_hashes", {}).items()}
        tools.setdefault("extractor", str(metadata.get("extractor_sha256", digest)))
        return contract, str(metadata.get("formal_source_sha256", digest)), tools, tuple(
            str(item) for item in metadata.get("limitations", ())
        )
    receipt = (
        value
        if isinstance(value, LeanExtractionReceipt)
        else LeanExtractionReceipt.model_validate(value)
    )
    tools = dict(receipt.tool_hashes)
    tools["extractor"] = receipt.extractor_source_sha256
    if receipt.status != "scaffold" or receipt.payload is None:
        raise ValueError("Lean extraction did not produce a bound receipt")
    # The version-stable bundled helper deliberately marks its textual scaffold
    # opaque. It can support inspection but not manufacture a semantic AST.
    contract = _opaque_contract(
        manifest,
        "; ".join(receipt.payload.opaque_reasons)
        or "Extractor did not produce a supported expression tree",
    )
    return contract, receipt.project_sha256, tools, receipt.limitations


def _field_model(item: Any, evidence_id: str) -> FieldComparison:
    return item.model_copy(update={"evidence_ids": (evidence_id,)})


def _finding_model(item: Any, evidence_id: str) -> CorrespondenceFinding:
    return item.model_copy(update={"evidence_ids": (evidence_id,)})


def assemble_correspondence_report(
    loaded: LoadedManifest,
    source_binding: SourceBinding,
    formal_contract: ClaimContract,
    *,
    formal_source_sha256: str,
    tool_hashes: Mapping[str, str],
    extractor_limitations: tuple[str, ...] = (),
    backend: str = "python-reference",
) -> CorrespondenceReport:
    """Compare two explicit contracts and assemble independently checkable ancestry."""
    manifest = loaded.manifest
    require_reviewed_manifest(manifest)
    if not source_binding.complete:
        raise ValueError("Checked correspondence requires bound document and excerpt bytes")
    spec = manifest.specification
    mapping = spec.symbol_mapping
    compared: ComparisonResult = compare_contracts(
        spec.source_contract,
        formal_contract,
        symbol_mapping=mapping,
        backend=backend,
    )
    if not verify_comparison_result(compared):
        raise ValueError("Comparison receipt failed independent integrity verification")

    source_bytes_digest = canonical_sha256(source_binding)
    attestation_digest = canonical_sha256(spec.reviewer_attestations)
    formal_digest = canonical_sha256(formal_contract)
    receipt_digest = compared.receipt.receipt_sha256
    e_source = _evidence_id("SOURCE_BYTES_BOUND", source_bytes_digest)
    e_attest = _evidence_id("SOURCE_CONTRACT_ATTESTED", attestation_digest)
    e_formal = _evidence_id("FORMAL_CONTRACT_EXTRACTED", formal_digest)
    e_compare = _evidence_id("CONTRACT_FIELDS_COMPARED", receipt_digest)
    e_defined = _evidence_id("DEFINEDNESS_OBLIGATION_CHECKED", receipt_digest)
    evidence = (
        EvidenceReference(
            evidence_id=e_source,
            kind="SOURCE_BYTES_BOUND",
            artifact_sha256=source_bytes_digest,
            trust=TrustLevel.UNKNOWN,
            detail="Document, excerpt, and transcription hashes were independently rebound.",
        ),
        EvidenceReference(
            evidence_id=e_attest,
            kind="SOURCE_CONTRACT_ATTESTED",
            artifact_sha256=loaded.canonical_manifest_sha256,
            trust=TrustLevel.UNKNOWN,
            parent_ids=(e_source,),
            detail=f"Binding reviewer attestations digest: {attestation_digest}",
        ),
        EvidenceReference(
            evidence_id=e_formal,
            kind="FORMAL_CONTRACT_EXTRACTED",
            artifact_sha256=formal_digest,
            trust=TrustLevel.EXACT,
            detail="A pinned extractor supplied the explicit formal contract.",
        ),
        EvidenceReference(
            evidence_id=e_compare,
            kind="CONTRACT_FIELDS_COMPARED",
            artifact_sha256=receipt_digest,
            trust=TrustLevel.UNKNOWN,
            parent_ids=(e_attest, e_formal),
            detail=f"Deterministic comparator backend: {compared.backend}",
        ),
        EvidenceReference(
            evidence_id=e_defined,
            kind="DEFINEDNESS_OBLIGATION_CHECKED",
            artifact_sha256=hashlib.sha256(
                repr([item.model_dump(mode="json") for item in compared.findings]).encode("utf-8")
            ).hexdigest(),
            trust=TrustLevel.UNKNOWN,
            parent_ids=(e_compare,),
        ),
    )
    field_results = [_field_model(item, e_compare) for item in compared.field_results]
    defined_findings = [item for item in compared.findings if item.check == CheckKind.DEFINEDNESS]
    for index, result in enumerate(field_results):
        if result.check == CheckKind.DEFINEDNESS:
            field_results[index] = result.model_copy(update={"evidence_ids": (e_defined,)})
            break
    else:
        field_results.append(
            FieldComparison(
                check=CheckKind.DEFINEDNESS,
                outcome="match" if not defined_findings else "difference",
                relation=ClaimRelation.EQUIVALENT if not defined_findings else ClaimRelation.MISMATCH,
                source_digest=canonical_sha256(spec.source_contract.definedness_obligations),
                formal_digest=canonical_sha256(formal_contract.definedness_obligations),
                evidence_ids=(e_defined,),
                message="Definedness obligations are discharged." if not defined_findings
                else "One or more definedness obligations are unresolved.",
            )
        )
    # Ensure optional checks still have an explicit result; v1 reports never
    # silently omit a material field.
    present = {item.check for item in field_results}
    for check in ALL_CHECKS:
        if check not in present:
            field_results.append(
                FieldComparison(
                    check=check,
                    outcome="not_checked",
                    relation=ClaimRelation.UNKNOWN,
                    evidence_ids=(e_compare,),
                    message="This required field was not checked.",
                )
            )
    field_results.sort(key=lambda item: list(CheckKind).index(item.check))
    findings = tuple(_finding_model(item, e_defined if item.check == CheckKind.DEFINEDNESS else e_compare)
                     for item in compared.findings)
    relation = ClaimRelation(compared.relation)
    matched = (
        relation == ClaimRelation.EQUIVALENT
        and compared.complete
        and not spec.source_contract.contains_opaque
        and not formal_contract.contains_opaque
        and all(item.outcome == "match" for item in field_results)
    )
    return CorrespondenceReport(
        mode="comparison",
        status="matched" if matched else "incomplete" if relation == ClaimRelation.UNKNOWN else "issues_found",
        relation=relation,
        trust=TrustLevel.UNKNOWN,
        contract_alignment="established_relative_to_manifest" if matched else "not_established",
        source_document_sha256=spec.source.document_sha256,
        source_excerpt_sha256=spec.source.excerpt_sha256,
        source_transcription_sha256=spec.source.transcription_sha256,
        source_manifest_sha256=loaded.source_manifest_sha256,
        canonical_manifest_sha256=loaded.canonical_manifest_sha256,
        formal_source_sha256=formal_source_sha256,
        normalized_source_contract_sha256=compared.receipt.source_contract_sha256,
        normalized_formal_contract_sha256=compared.receipt.formal_contract_sha256,
        comparison_receipt=json.loads(json.dumps(asdict(compared.receipt))),
        backend=compared.backend,
        formal_target=spec.formal_target,
        tool_hashes=dict(tool_hashes),
        field_results=tuple(field_results),
        findings=findings,
        evidence=evidence,
        limitations=LIMITATIONS + tuple(extractor_limitations),
    )


def audit_correspondence(
    *,
    project_root: str | Path,
    formal_spec: Any,
    correspondence_spec: str | Path | LoadedManifest | CorrespondenceManifest | CorrespondenceSpec | Mapping[str, Any],
    extractor: Callable[..., Any] | None = None,
    document_path: str | Path | None = None,
    excerpt_path: str | Path | None = None,
    inspection_only: bool = False,
    backend: str = "python-reference",
    limits: Any = None,
) -> CorrespondenceReport:
    """Public read-only correspondence API.

    The function never installs tools, fetches dependencies, or mutates the Lean
    project. The supplied extractor remains responsible for crossing the
    separately authorized execution boundary.
    """
    if isinstance(correspondence_spec, LoadedManifest):
        loaded = correspondence_spec
    elif isinstance(correspondence_spec, (str, Path)):
        loaded = load_manifest(correspondence_spec, limits)
    else:
        if isinstance(correspondence_spec, CorrespondenceSpec):
            manifest = CorrespondenceManifest(
                review_state="reviewed" if correspondence_spec.is_reviewed else "candidate",
                specification=correspondence_spec,
            )
        elif isinstance(correspondence_spec, CorrespondenceManifest):
            manifest = correspondence_spec
        else:
            manifest = CorrespondenceManifest.model_validate_json(
                canonical_json_bytes(correspondence_spec)
            )
        encoded = canonical_json_bytes(manifest)
        loaded = LoadedManifest(
            manifest=manifest,
            source_manifest_sha256=sha256_hex(encoded),
            canonical_manifest_sha256=canonical_sha256(manifest),
            source_size=len(encoded),
        )
    target = loaded.manifest.specification.formal_target.model_dump(mode="json")
    if formal_spec is not None:
        formal_value = formal_spec.model_dump(mode="json") if hasattr(formal_spec, "model_dump") else formal_spec
        targets = formal_value.get("targets", ()) if isinstance(formal_value, Mapping) else ()
        if targets and target not in targets:
            raise ValueError("Correspondence target is not present in the formal project specification")
    if inspection_only:
        binding = None
        if document_path is not None and excerpt_path is not None:
            binding = verify_source_files(
                loaded.manifest, document_path=document_path, excerpt_path=excerpt_path
            )
        return inspection_report(loaded, source_binding=binding)
    if document_path is None or excerpt_path is None:
        raise ValueError("Checked correspondence requires document_path and excerpt_path")
    binding = verify_source_files(
        loaded.manifest, document_path=document_path, excerpt_path=excerpt_path
    )
    formal, source_digest, tool_hashes, extractor_limits = _formal_contract_from_extractor(
        extractor,
        project_root=project_root,
        formal_spec=formal_spec,
        manifest=loaded.manifest,
    )
    return assemble_correspondence_report(
        loaded,
        binding,
        formal,
        formal_source_sha256=source_digest,
        tool_hashes=tool_hashes,
        extractor_limitations=extractor_limits,
        backend=backend,
    )


def replay_correspondence_report(data: bytes | str) -> CorrespondenceReport:
    """Strict persistence decoder; validators reject tampering/inconsistent claims."""
    return CorrespondenceReport.model_validate_json(data)


def correspondence_evidence_bundle(report: CorrespondenceReport) -> EvidenceBundle:
    """Project report ancestry into MathKernel's shared evidence vocabulary."""
    bundle = EvidenceBundle()
    for item in report.evidence:
        metadata = {
            "kind": item.kind,
            "artifact_sha256": item.artifact_sha256,
            "parents": list(item.parent_ids),
            "semantic_alignment": report.semantic_alignment,
            "contract_alignment": report.contract_alignment,
        }
        if item.kind in {"CONTRACT_FIELDS_COMPARED", "DEFINEDNESS_OBLIGATION_CHECKED"}:
            bundle.certificate.append(CertificateEvidence(
                certificate_type=item.kind.lower(),
                claim="explicit_contract_correspondence_relative_to_reviewed_manifest",
                witness=report.comparison_receipt or {"artifact_sha256": item.artifact_sha256},
                verifier="mathkernel.formal_audit.correspondence",
                verified=True,
                trust=item.trust.value,
                support_path="semantic-correspondence",
                metadata=metadata,
            ))
        else:
            bundle.computation.append(ComputationEvidence(
                engine="formal_correspondence",
                method=item.kind.lower(),
                arithmetic="canonical_sha256",
                deterministic=True,
                trust=item.trust.value,
                support_path="semantic-correspondence",
                metadata=metadata,
            ))
    return bundle


__all__ = [
    "LIMITATIONS",
    "assemble_correspondence_report",
    "audit_correspondence",
    "correspondence_evidence_bundle",
    "replay_correspondence_report",
]
