"""Regression tests for the bounded semantic-correspondence subsystem."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
from pydantic import ValidationError

from mathkernel import MathKernel
from mathkernel.formal_audit import FormalProjectSpec
from mathkernel.formal_audit.__main__ import main as formal_audit_main
from mathkernel.formal_audit.correspondence import (
    Attestation,
    ClaimContract,
    ClaimExpression,
    ClaimRelation,
    CorrespondenceManifest,
    CorrespondenceSpec,
    ExpressionKind,
    FormalTarget,
    LeanExtractionRequest,
    LeanExtractorTools,
    ManifestLimits,
    SourceAnchor,
    SourceLocator,
    SymbolMapping,
    assemble_correspondence_report,
    audit_correspondence,
    compare_contracts,
    compare_contracts_batch,
    correspondence_evidence_bundle,
    extract_lean_signature,
    inspection_report,
    loads_manifest,
    replay_correspondence_report,
    require_reviewed_manifest,
    sha256_hex,
    verify_comparison_result,
    verify_source_binding,
)
from mathkernel.formal_audit.models import (
    FormalTarget as AuditFormalTarget,
    PinnedBinary,
)


FIXTURES = Path(__file__).parent / "fixtures" / "correspondence"
ZERO = "0" * 64
ONE = "1" * 64


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _base_contract(claim_id: str = "claim") -> ClaimContract:
    return ClaimContract(
        claim_id=claim_id,
        conclusion=ClaimExpression(kind=ExpressionKind.BOOLEAN, value=True),
    )


def _anchor(document: bytes = b"document", excerpt: bytes = b"excerpt") -> SourceAnchor:
    transcription = "Every reviewed claim is explicit."
    return SourceAnchor(
        document_sha256=sha256_hex(document),
        excerpt_sha256=sha256_hex(excerpt),
        transcription_sha256=sha256_hex(transcription.encode("utf-8")),
        transcription=transcription,
        locator=SourceLocator(section="fixture"),
    )


def _manifest(*, reviewed: bool) -> CorrespondenceManifest:
    document = b"document"
    excerpt = b"excerpt"
    anchor = _anchor(document, excerpt)
    contract = _base_contract()
    attestations = ()
    if reviewed:
        attestations = (
            Attestation(
                attestation_id="review-1",
                reviewer_id="tests",
                reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                decision="approved",
                document_sha256=anchor.document_sha256,
                excerpt_sha256=anchor.excerpt_sha256,
                transcription_sha256=anchor.transcription_sha256,
                source_contract_sha256=contract.digest,
            ),
        )
    return CorrespondenceManifest(
        review_state="reviewed" if reviewed else "candidate",
        specification=CorrespondenceSpec(
            source=anchor,
            source_contract=contract,
            formal_target=FormalTarget(module="Fixture", declaration="Fixture.claim"),
            reviewer_attestations=attestations,
        ),
    )


def _loaded(*, reviewed: bool = True):
    manifest = _manifest(reviewed=reviewed)
    return loads_manifest(manifest.model_dump_json())


def _finding_codes(result) -> set[str]:
    return {finding.code for finding in result.findings}


def _semantic_projection(result) -> tuple:
    return (
        result.relation,
        result.complete,
        tuple(
            (
                str(item.check),
                item.outcome,
                str(item.relation),
                item.source_digest,
                item.formal_digest,
                item.opaque,
            )
            for item in result.field_results
        ),
        tuple(
            (item.code, item.severity, str(item.relation), str(item.check), item.message)
            for item in result.findings
        ),
    )


def test_public_models_are_strict_frozen_and_extra_forbidden():
    expression = ClaimExpression(kind=ExpressionKind.SYMBOL, value="x")
    with pytest.raises(ValidationError, match="frozen"):
        expression.value = "y"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ClaimExpression.model_validate(
            {"kind": "symbol", "value": "x", "invented": True}
        )
    with pytest.raises(ValidationError):
        ClaimExpression(kind=ExpressionKind.INTEGER, value=True)
    with pytest.raises(ValidationError):
        SourceLocator()


def test_contract_and_mapping_uniqueness_invariants():
    expression = ClaimExpression(kind=ExpressionKind.SYMBOL, value="P")
    with pytest.raises(ValidationError, match="Duplicate entries"):
        ClaimContract(
            claim_id="duplicate",
            assumptions=(expression, expression),
            conclusion=expression,
        )
    manifest = _manifest(reviewed=False)
    spec = manifest.specification
    with pytest.raises(ValidationError, match="source symbol"):
        CorrespondenceSpec(
            source=spec.source,
            source_contract=spec.source_contract,
            formal_target=spec.formal_target,
            symbol_mapping=(
                SymbolMapping(source_symbol="a", formal_symbol="x"),
                SymbolMapping(source_symbol="a", formal_symbol="y"),
            ),
        )
    with pytest.raises(ValidationError, match="cycles"):
        CorrespondenceSpec(
            source=spec.source,
            source_contract=spec.source_contract,
            formal_target=spec.formal_target,
            symbol_mapping=(
                SymbolMapping(source_symbol="a", formal_symbol="b"),
                SymbolMapping(source_symbol="b", formal_symbol="a"),
            ),
        )


def test_manifest_rejects_duplicate_json_keys_before_validation():
    duplicate = (
        '{"schema_version":"mathkernel.semantic-correspondence-manifest/v1",'
        '"schema_version":"mathkernel.semantic-correspondence-manifest/v1"}'
    )
    with pytest.raises(ValueError, match="Duplicate JSON object key"):
        loads_manifest(duplicate)


@pytest.mark.parametrize(
    ("payload", "limits", "message"),
    [
        ("{}", ManifestLimits(max_bytes=1), "max_bytes"),
        ('{"x":[[[[]]]]}', ManifestLimits(max_depth=3), "max_depth"),
        ('{"x":"abcd"}', ManifestLimits(max_string_chars=3), "max_string_chars"),
        ('{"x":[1,2]}', ManifestLimits(max_collection_items=1), "max_collection_items"),
        ('{"a":1,"b":2}', ManifestLimits(max_object_keys=1), "max_object_keys"),
    ],
)
def test_manifest_resource_bounds_fail_before_schema_acceptance(payload, limits, message):
    with pytest.raises(ValueError, match=message):
        loads_manifest(payload, limits)


def test_candidate_review_gate_and_inspection_only_contract():
    loaded = _loaded(reviewed=False)
    with pytest.raises(ValueError, match="reviewed manifest"):
        require_reviewed_manifest(loaded.manifest)
    report = inspection_report(loaded)
    assert report.mode == "inspection"
    assert report.status == "incomplete"
    assert report.relation == ClaimRelation.UNKNOWN
    assert report.contract_alignment == report.semantic_alignment == "not_established"
    assert report.formal_source_sha256 is None
    assert not report.field_results
    assert "SC_MANIFEST_REVIEW_REQUIRED" in _finding_codes(report)


def test_inspection_api_never_invokes_extractor(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(reviewed=False).model_dump_json(), encoding="utf-8")

    def forbidden_extractor(**_):
        pytest.fail("inspection-only mode invoked an extractor")

    report = audit_correspondence(
        project_root=tmp_path,
        formal_spec=None,
        correspondence_spec=manifest_path,
        extractor=forbidden_extractor,
        inspection_only=True,
    )
    assert report.mode == "inspection" and report.relation == ClaimRelation.UNKNOWN


def test_fixture_a_m_plus_4_vs_m_plus_5_is_formal_weaker():
    fixture = _fixture("fixture_a_derivative_loss.json")
    result = compare_contracts(fixture["source"], fixture["formal"])
    assert result.relation == "formal_weaker"
    assert _finding_codes(result) == {"SC_REGULARITY_ORDER_MISMATCH"}
    assert verify_comparison_result(result)


def test_fixture_b_dependency_and_exponent_are_separate_findings():
    fixture = _fixture("fixture_b_pressure_flux.json")
    result = compare_contracts(fixture["source"], fixture["formal"])
    assert result.relation == "incomparable"
    assert {"SC_DEPENDENCY_ADDED", "SC_SCALE_EXPONENT_MISMATCH"} <= _finding_codes(
        result
    )


def test_fixture_c_almost_everywhere_vs_pointwise_is_not_equal():
    fixture = _fixture("fixture_c_time_scope.json")
    result = compare_contracts(fixture["source"], fixture["formal"])
    assert result.relation == "formal_stronger"
    assert "SC_QUANTIFIER_SCOPE_MISMATCH" in _finding_codes(result)


def test_fixture_d_nat_sinf_unresolved_then_discharged():
    fixture = _fixture("fixture_d_sinf.json")
    unresolved = compare_contracts(fixture["source"], fixture["formal_unresolved"])
    assert unresolved.relation == "mismatch"
    assert "SC_DEFINEDNESS_TOTALIZED_EMPTY_SET" in _finding_codes(unresolved)
    discharged = compare_contracts(fixture["source"], fixture["formal_discharged"])
    assert discharged.relation == "equivalent"
    assert discharged.complete
    assert "SC_DEFINEDNESS_TOTALIZED_EMPTY_SET" not in _finding_codes(discharged)


def test_fixture_e_unrelated_true_predicates_remain_mismatch():
    fixture = _fixture("fixture_e_unrelated_true.json")
    result = compare_contracts(fixture["source"], fixture["formal"])
    assert result.relation == "mismatch"
    assert "SC_CONCLUSION_MISMATCH" in _finding_codes(result)
    assert verify_comparison_result(result)


def test_opaque_required_material_yields_unknown_not_equivalence():
    source = {
        "claim_id": "opaque-source",
        "conclusion": {
            "kind": "opaque",
            "opaque_reason": "unsupported higher-order expression",
        },
    }
    formal = {
        "claim_id": "opaque-formal",
        "conclusion": {"kind": "boolean", "value": True},
    }
    result = compare_contracts(source, formal)
    assert result.relation == "unknown"
    assert not result.complete
    assert "SC_OPAQUE_REQUIRED_FIELD" in _finding_codes(result)
    assert any(item.opaque and item.outcome == "unsupported" for item in result.field_results)


def test_serial_and_process_batch_have_identical_semantic_results():
    names = (
        "fixture_a_derivative_loss.json",
        "fixture_b_pressure_flux.json",
        "fixture_c_time_scope.json",
        "fixture_e_unrelated_true.json",
    )
    pairs = [
        (_fixture(name)["source"], _fixture(name)["formal"])
        for name in names
    ]
    serial = compare_contracts_batch(pairs, workers=1, min_parallel=1)
    parallel = compare_contracts_batch(pairs, workers=2, min_parallel=2)
    assert [_semantic_projection(item) for item in serial.results] == [
        _semantic_projection(item) for item in parallel.results
    ]
    assert all(verify_comparison_result(item) for item in parallel.results)
    assert parallel.backend in {"process-map", "python-reference"}


def test_matched_report_replays_and_claim_tampering_is_rejected():
    loaded = _loaded(reviewed=True)
    binding = verify_source_binding(
        loaded.manifest,
        document_bytes=b"document",
        excerpt_bytes=b"excerpt",
    )
    report = assemble_correspondence_report(
        loaded,
        binding,
        loaded.manifest.specification.source_contract,
        formal_source_sha256=ONE,
        tool_hashes={"extractor": ZERO},
    )
    assert report.status == "matched"
    assert report.contract_alignment == "established_relative_to_manifest"
    assert report.semantic_alignment == "not_established"
    assert replay_correspondence_report(report.model_dump_json()) == report
    assert correspondence_evidence_bundle(report).conservative_trust() == "unknown"

    mutations = (
        {"semantic_alignment": "established"},
        {"relation": "mismatch"},
        {"status": "issues_found"},
        {"trust": "formal"},
        {"formal_source_sha256": None},
        {"tool_hashes": {}},
        {
            "comparison_receipt": {
                **report.comparison_receipt,
                "receipt_sha256": ZERO,
            }
        },
    )
    base = report.model_dump(mode="json")
    for mutation in mutations:
        tampered = dict(base)
        tampered.update(mutation)
        with pytest.raises(ValidationError):
            replay_correspondence_report(json.dumps(tampered))


def test_comparison_receipt_rejects_field_and_receipt_tampering():
    fixture = _fixture("fixture_a_derivative_loss.json")
    result = compare_contracts(fixture["source"], fixture["formal"])
    assert verify_comparison_result(result)
    assert not verify_comparison_result(replace(result, relation="equivalent"))
    bad_receipt = replace(result.receipt, receipt_sha256=ZERO)
    assert not verify_comparison_result(replace(result, receipt=bad_receipt))


def test_unauthorized_lean_extractor_never_starts_subprocess(tmp_path, monkeypatch):
    import mathkernel.formal_audit.correspondence.lean_extract as extraction

    monkeypatch.setattr(
        extraction,
        "run_bounded",
        lambda *_args, **_kwargs: pytest.fail("unauthorized extractor started a process"),
    )
    tool = PinnedBinary(path=str(tmp_path / "never-run.exe"), sha256=ZERO)
    request = LeanExtractionRequest(
        project_root=str(tmp_path),
        target=AuditFormalTarget(module="Fixture", declaration="Fixture.claim"),
        expected_project_sha256=ONE,
        expected_toolchain="leanprover/lean4:v4.34.0-rc2",
        extractor_source_sha256=ZERO,
        tools=LeanExtractorTools(lean=tool, lake=tool),
    )
    receipt = extract_lean_signature(request, authorize_execution=False)
    assert receipt.status == "blocked"
    assert receipt.tool_hashes == {}
    assert "authorization" in receipt.errors[0].lower()


def test_cli_correspond_inspection_is_windows_safe_and_nonexecuting(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_manifest(reviewed=False).model_dump_json(), encoding="utf-8")
    formal_spec = tmp_path / "formal.json"
    formal_spec.write_text(FormalProjectSpec().model_dump_json(), encoding="utf-8")
    output = tmp_path / "inspection.json"
    code = formal_audit_main(
        [
            "correspond",
            "--project",
            str(project),
            "--formal-spec",
            str(formal_spec),
            "--manifest",
            str(manifest),
            "--inspection-only",
            "--output",
            str(output),
        ]
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert data["mode"] == "inspection"
    assert data["relation"] == "unknown"
    assert data["semantic_alignment"] == "not_established"


def test_capability_discovery_exposes_only_read_only_inspection():
    kernel = MathKernel()
    capabilities = kernel.capability_query(domain="formal_correspondence")[
        "capabilities"
    ]
    assert [item["operation"] for item in capabilities] == [
        "formal_correspondence_inspect"
    ]
    assert capabilities[0]["trust_levels"] == ["unknown"]
    description = capabilities[0]["description"].lower()
    assert "read-only" in description and "never runs lean" in description


def test_mcp_does_not_expose_correspondence_or_extractor(tmp_path, monkeypatch):
    class FakeMCP:
        def __init__(self, *_args, **_kwargs):
            self.tools = {}

        def tool(self, fn=None, **_kwargs):
            def register(function):
                self.tools[function.__name__] = function
                return function

            return register(fn) if fn is not None else register

        def resource(self, *_args, **_kwargs):
            return lambda fn: fn

        def prompt(self, fn=None, **_kwargs):
            return fn if fn is not None else lambda function: function

    monkeypatch.setitem(sys.modules, "fastmcp", types.SimpleNamespace(FastMCP=FakeMCP))
    load_spec = importlib.util.spec_from_file_location(
        "_mathkernel_correspondence_mcp_contract",
        Path(__file__).parents[1] / "src" / "mathkernel_mcp" / "server.py",
    )
    module = importlib.util.module_from_spec(load_spec)
    assert load_spec.loader is not None
    load_spec.loader.exec_module(module)
    names = set(module.mcp.tools)
    assert not any("correspondence" in name for name in names)
    assert not any("extract" in name and "lean" in name for name in names)
