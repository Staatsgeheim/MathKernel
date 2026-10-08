"""Contract-relative semantic correspondence auditing."""
from .compare import (
    BatchComparisonResult,
    ComparisonResult,
    compare_claim_contracts,
    compare_contracts,
    compare_contracts_batch,
    verify_comparison_receipt,
    verify_comparison_result,
)
from .definedness import audit_definedness, check_definedness
from .lean_extract import (
    LeanExtractionLimits,
    LeanExtractionReceipt,
    LeanExtractionRequest,
    LeanExtractorTools,
    bundled_helper_sha256,
    extract_lean_signature,
    validate_extraction_receipt,
)
from .manifest import (
    LoadedManifest,
    ManifestLimits,
    SourceBinding,
    inspection_report,
    load_manifest,
    loads_manifest,
    require_reviewed_manifest,
    verify_source_binding,
    verify_source_files,
)
from .models import *  # re-export the public schema types
from .models import __all__ as _model_exports
from .normalize import normalize_contract, normalize_expression, verify_normalization
from .report import (
    assemble_correspondence_report,
    audit_correspondence,
    correspondence_evidence_bundle,
    replay_correspondence_report,
)

__all__ = list(_model_exports) + [
    "BatchComparisonResult",
    "ComparisonResult",
    "LeanExtractionLimits",
    "LeanExtractionReceipt",
    "LeanExtractionRequest",
    "LeanExtractorTools",
    "LoadedManifest",
    "ManifestLimits",
    "SourceBinding",
    "assemble_correspondence_report",
    "audit_correspondence",
    "audit_definedness",
    "bundled_helper_sha256",
    "check_definedness",
    "compare_claim_contracts",
    "compare_contracts",
    "compare_contracts_batch",
    "correspondence_evidence_bundle",
    "extract_lean_signature",
    "inspection_report",
    "load_manifest",
    "loads_manifest",
    "normalize_contract",
    "normalize_expression",
    "replay_correspondence_report",
    "require_reviewed_manifest",
    "validate_extraction_receipt",
    "verify_comparison_receipt",
    "verify_comparison_result",
    "verify_normalization",
    "verify_source_binding",
    "verify_source_files",
]
