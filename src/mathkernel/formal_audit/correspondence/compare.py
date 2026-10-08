"""Field-wise, fail-closed comparison of normalized claim contracts."""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

from mathkernel.parallel import process_map, resolve_workers

from .definedness import DefinednessFinding, audit_definedness
from .models import (CheckKind, ClaimRelation, CorrespondenceFinding,
                     FieldComparison)
from .normalize import (NORMALIZER_VERSION, NormalizationLimits, NormalizationResult,
                        canonical_json, normalize_contract, normalized_digest)


COMPARATOR_VERSION = "mathkernel.correspondence-comparator/v1"
FIELD_ORDER = (
    "binders", "domains", "assumptions", "conclusion", "regularity",
    "parameter_dependencies", "measure_scope", "definitions",
)


@dataclass(frozen=True)
class ComparisonReceipt:
    schema_version: str
    comparator_version: str
    normalizer_version: str
    source_contract_sha256: str
    formal_contract_sha256: str
    field_results_sha256: str
    findings_sha256: str
    relation: str
    backend: str
    backend_metadata: tuple[tuple[str, str], ...]
    receipt_sha256: str


@dataclass(frozen=True)
class ComparisonResult:
    relation: str
    field_results: tuple[FieldComparison, ...]
    findings: tuple[CorrespondenceFinding, ...]
    receipt: ComparisonReceipt
    complete: bool
    backend: str
    backend_metadata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BatchComparisonResult:
    results: tuple[ComparisonResult, ...]
    backend: str
    backend_metadata: tuple[tuple[str, str], ...]


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json", exclude_none=False))
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name))
                for field in dataclass_fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _field(contract: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "binders": ("binders", "quantifiers"),
        "domains": ("domains", "domain_constraints"),
        "assumptions": ("assumptions", "hypotheses"),
        "conclusion": ("conclusion", "claim", "predicate"),
        "regularity": ("regularity", "regularity_requirements"),
        "parameter_dependencies": ("parameter_dependencies", "dependencies"),
        "measure_scope": ("measure_scope", "scope"),
        "definitions": ("definitions", "definition_bindings"),
    }[name]
    for alias in aliases:
        if alias in contract:
            return contract[alias]
    return () if name not in {"conclusion", "measure_scope"} else None


def _digest(value: Any) -> str:
    return normalized_digest(value)


def _items(value: Any) -> dict[bytes, Any]:
    if value is None:
        return {}
    values = value if isinstance(value, list) else [value]
    return {canonical_json(item): item for item in values}


def _finding(code: str, relation: str, field: str, message: str,
             severity: str = "error") -> CorrespondenceFinding:
    return CorrespondenceFinding(
        code=code, severity=severity, relation=ClaimRelation(relation),
        check=_check_kind(field), message=message,
    )


def _check_kind(field: str) -> CheckKind | None:
    if field == "parameter_dependencies":
        return CheckKind.DEPENDENCIES
    if field == "definedness":
        return CheckKind.DEFINEDNESS
    if field == "contract":
        return None
    return CheckKind(field)


def _compare_set_field(field: str, source: Any, formal: Any
                       ) -> tuple[str, str, list[CorrespondenceFinding]]:
    left, right = _items(source), _items(formal)
    if left.keys() == right.keys():
        return "equal", "equivalent", []
    added, removed = right.keys() - left.keys(), left.keys() - right.keys()
    if field == "assumptions":
        findings = []
        if added:
            findings.append(_finding("SC_HYPOTHESIS_ADDED", "formal_weaker", field,
                                     "Formal contract has additional hypotheses."))
        if removed:
            findings.append(_finding("SC_HYPOTHESIS_REMOVED", "formal_stronger", field,
                                     "Formal contract omits source hypotheses."))
        relation = ("incomparable" if added and removed else
                    "formal_weaker" if added else "formal_stronger")
        return "different", relation, findings
    if field == "parameter_dependencies":
        findings = []
        if added:
            findings.append(_finding("SC_DEPENDENCY_ADDED", "incomparable", field,
                                     "Formal result has additional parameter dependencies."))
        if removed:
            findings.append(_finding("SC_DEPENDENCY_REMOVED", "incomparable", field,
                                     "Formal result omits source parameter dependencies."))
        return "different", "incomparable", findings
    code = {
        "domains": "SC_DOMAIN_MISMATCH",
        "binders": "SC_QUANTIFIER_MISMATCH",
        "definitions": "SC_DEFINITION_MISMATCH",
    }.get(field, "SC_FIELD_MISMATCH")
    return "different", "mismatch", [
        _finding(code, "mismatch", field, f"Source and formal {field} differ.")
    ]


def _scope_name(value: Any) -> str:
    if value is None:
        return "unspecified"
    if isinstance(value, str):
        return value.lower().replace("-", "_").replace(" ", "_")
    if isinstance(value, Mapping):
        for key in ("kind", "scope", "quantifier", "name"):
            if isinstance(value.get(key), str):
                return _scope_name(value[key])
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _compare_scope(source: Any, formal: Any
                   ) -> tuple[str, str, list[CorrespondenceFinding]]:
    left, right = _scope_name(source), _scope_name(formal)
    if left == right:
        return "equal", "equivalent", []
    ae = {"almost_everywhere", "ae", "almosteverywhere"}
    pointwise = {"pointwise", "everywhere", "forall", "every"}
    if left in ae and right in pointwise:
        relation = "formal_stronger"
    elif left in pointwise and right in ae:
        relation = "formal_weaker"
    else:
        relation = "incomparable"
    return "different", relation, [
        _finding("SC_QUANTIFIER_SCOPE_MISMATCH", relation, "measure_scope",
                 "Pointwise and measure-qualified scopes are not interchangeable.")
    ]


def _linear_order(value: Any) -> tuple[bytes, Fraction] | None:
    """Extract a symbolic base plus exact additive offset, without algebra guessing."""
    if isinstance(value, Mapping):
        kind = value.get("kind", value.get("op"))
        operation = value.get("operator") if kind == "operator" else kind
        if kind in {"integer", "nat", "int"}:
            try:
                return b"", Fraction(int(str(value["value"])))
            except (KeyError, ValueError):
                return None
        if kind == "rational":
            try:
                if isinstance(value.get("value"), str):
                    return b"", Fraction(value["value"])
                return b"", Fraction(int(str(value["numerator"])),
                                     int(str(value["denominator"])))
            except (KeyError, ValueError, ZeroDivisionError):
                return None
        if operation in {"add", "+"}:
            args = value.get("args", value.get("arguments", value.get("operands")))
            if not isinstance(args, list):
                args = [value.get("left"), value.get("right")]
            constant, symbolic = Fraction(), []
            for arg in args:
                part = _linear_order(arg)
                if part is None:
                    symbolic.append(arg)
                elif part[0]:
                    symbolic.append(arg)
                else:
                    constant += part[1]
            return canonical_json(symbolic), constant
        if kind in {"symbol", "identifier", "var"}:
            return canonical_json(value), Fraction()
    if isinstance(value, int):
        return b"", Fraction(value)
    return None


def _order_value(record: Any) -> Any:
    if isinstance(record, Mapping):
        for key in ("order", "derivative_order", "degree", "value"):
            if key in record:
                return record[key]
    return record


def _compare_regularity(source: Any, formal: Any
                        ) -> tuple[str, str, list[CorrespondenceFinding]]:
    if canonical_json(source) == canonical_json(formal):
        return "equal", "equivalent", []
    source_values = source if isinstance(source, list) else [source]
    formal_values = formal if isinstance(formal, list) else [formal]
    if len(source_values) == len(formal_values):
        directions = []
        for left, right in zip(source_values, formal_values):
            a, b = _linear_order(_order_value(left)), _linear_order(_order_value(right))
            if a is None or b is None or a[0] != b[0]:
                directions.append(None)
            else:
                directions.append((b[1] > a[1]) - (b[1] < a[1]))
        if directions and all(direction == 1 for direction in directions):
            relation = "formal_weaker"
        elif directions and all(direction == -1 for direction in directions):
            relation = "formal_stronger"
        else:
            relation = "incomparable"
    else:
        relation = "incomparable"
    return "different", relation, [
        _finding("SC_REGULARITY_ORDER_MISMATCH", relation, "regularity",
                 "Source and formal regularity requirements differ.")
    ]


def _collect_named(value: Any, keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in keys and isinstance(child, str):
                found.add(child)
            if key == "operator" and isinstance(child, str) and any(
                    token in child.lower() for token in ("norm", "sobolev", "space")):
                found.add(child)
            found.update(_collect_named(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_named(child, keys))
    return found


def _collect_exponents(value: Any) -> set[bytes]:
    found: set[bytes] = set()
    if isinstance(value, Mapping):
        kind = value.get("kind", value.get("op"))
        operation = value.get("operator") if kind == "operator" else kind
        if operation in {"pow", "^", "power"}:
            args = value.get("arguments", ())
            exponent = value.get("right", value.get(
                "exponent", args[1] if isinstance(args, list) and len(args) == 2 else None))
            found.add(canonical_json(exponent))
        for child in value.values():
            found.update(_collect_exponents(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_exponents(child))
    return found


def _compare_conclusion(source: Any, formal: Any
                        ) -> tuple[str, str, list[CorrespondenceFinding]]:
    if canonical_json(source) == canonical_json(formal):
        return "equal", "equivalent", []
    findings: list[CorrespondenceFinding] = []
    norm_keys = {"norm", "space", "norm_type", "function_space"}
    if _collect_named(source, norm_keys) != _collect_named(formal, norm_keys):
        findings.append(_finding("SC_NORM_SPACE_MISMATCH", "mismatch", "conclusion",
                                 "Source and formal norm or function space differ."))
    if _collect_exponents(source) != _collect_exponents(formal):
        findings.append(_finding("SC_SCALE_EXPONENT_MISMATCH", "incomparable", "conclusion",
                                 "Source and formal scale exponents differ."))
    if not findings:
        findings.append(_finding("SC_CONCLUSION_MISMATCH", "mismatch", "conclusion",
                                 "Source and formal conclusions differ structurally."))
    relation = "mismatch" if any(f.relation == "mismatch" for f in findings) else "incomparable"
    return "different", relation, findings


def _has_opaque(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("kind") in {"opaque", "unsupported", "unknown"} or value.get("opaque") is True:
            return True
        return any(_has_opaque(child) for child in value.values())
    if isinstance(value, list):
        return any(_has_opaque(child) for child in value)
    return False


def _material_symbols(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind in {"symbol", "identifier", "var"}:
            symbol = value.get("value", value.get("name", value.get("symbol")))
            if isinstance(symbol, str) and not symbol.startswith("_b"):
                found.add(symbol)
        for key in ("subject", "quantity", "symbol"):
            item = value.get(key)
            if isinstance(item, str) and not item.startswith("_b"):
                found.add(item)
        dependencies = value.get("depends_on")
        if isinstance(dependencies, list):
            found.update(item for item in dependencies
                         if isinstance(item, str) and not item.startswith("_b"))
        for child in value.values():
            found.update(_material_symbols(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_material_symbols(child))
    return found


def compare_fields(source: Mapping[str, Any], formal: Mapping[str, Any]
                   ) -> tuple[tuple[FieldComparison, ...],
                              tuple[CorrespondenceFinding, ...]]:
    results: list[FieldComparison] = []
    findings: list[CorrespondenceFinding] = []
    for field in FIELD_ORDER:
        left, right = _field(source, field), _field(formal, field)
        if _has_opaque(left) or _has_opaque(right):
            status, relation = "unknown", "unknown"
            current = [_finding("SC_OPAQUE_REQUIRED_FIELD", "unknown", field,
                                f"Opaque material prevents comparison of {field}.")]
        elif field == "measure_scope":
            status, relation, current = _compare_scope(left, right)
        elif field == "regularity":
            status, relation, current = _compare_regularity(left, right)
        elif field == "conclusion":
            status, relation, current = _compare_conclusion(left, right)
        else:
            status, relation, current = _compare_set_field(field, left, right)
        results.append(FieldComparison(
            check=_check_kind(field),
            outcome=("match" if status == "equal" else
                     "unsupported" if status == "unknown" else "difference"),
            relation=ClaimRelation(relation),
            source_digest=_digest(left), formal_digest=_digest(right),
            opaque=status == "unknown",
            message="Fields match." if status == "equal" else current[0].message,
        ))
        findings.extend(current)
    return tuple(results), tuple(findings)


def classify_relation(field_results: Iterable[FieldComparison],
                      findings: Iterable[CorrespondenceFinding] = ()) -> str:
    relations = {str(item.relation) for item in field_results
                 if str(item.relation) != "equivalent"}
    relations.update(str(item.relation) for item in findings
                     if str(item.relation) != "equivalent")
    if "mismatch" in relations:
        return "mismatch"
    if "unknown" in relations:
        return "unknown"
    directional = relations & {"formal_stronger", "formal_weaker"}
    if "incomparable" in relations or len(directional) > 1:
        return "incomparable"
    if directional:
        return next(iter(directional))
    return "equivalent"


def _receipt_payload(**values: Any) -> dict[str, Any]:
    return {key: values[key] for key in (
        "schema_version", "comparator_version", "normalizer_version",
        "source_contract_sha256", "formal_contract_sha256",
        "field_results_sha256", "findings_sha256", "relation", "backend",
        "backend_metadata",
    )}


def _make_receipt(source: NormalizationResult, formal: NormalizationResult,
                  fields: tuple[FieldComparison, ...],
                  findings: tuple[CorrespondenceFinding, ...], relation: str,
                  backend: str, metadata: tuple[tuple[str, str], ...]) -> ComparisonReceipt:
    values = dict(
        schema_version="mathkernel.correspondence-comparison-receipt/v1",
        comparator_version=COMPARATOR_VERSION,
        normalizer_version=NORMALIZER_VERSION,
        source_contract_sha256=source.sha256,
        formal_contract_sha256=formal.sha256,
        field_results_sha256=_digest([_plain(item) for item in fields]),
        findings_sha256=_digest([_plain(item) for item in findings]),
        relation=relation,
        backend=backend,
        backend_metadata=metadata,
    )
    receipt_sha256 = hashlib.sha256(canonical_json(_receipt_payload(**values))).hexdigest()
    return ComparisonReceipt(receipt_sha256=receipt_sha256, **values)


def compare_contracts(source_contract: Any, formal_contract: Any, *,
                      symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                      source_symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                      formal_symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                      definitions: Mapping[str, Any] | Sequence[Any] | None = None,
                      unfold_allowlist: Sequence[str] = (),
                      limits: NormalizationLimits | None = None,
                      backend: str = "python-reference",
                      backend_metadata: Mapping[str, Any] | None = None) -> ComparisonResult:
    """Normalize and compare two contracts without attempting theorem proving."""
    common = symbol_mapping
    source = normalize_contract(
        source_contract, symbol_mapping=source_symbol_mapping or common,
        definitions=definitions, unfold_allowlist=unfold_allowlist, limits=limits,
    )
    formal = normalize_contract(
        formal_contract, symbol_mapping=formal_symbol_mapping or common,
        definitions=definitions, unfold_allowlist=unfold_allowlist, limits=limits,
    )
    fields, findings = compare_fields(source.value, formal.value)
    if ((bool(symbol_mapping) or bool(source_symbol_mapping)
         or bool(formal_symbol_mapping))
            and _material_symbols(source.value) != _material_symbols(formal.value)):
        findings += (_finding(
            "SC_UNMAPPED_SYMBOL", "unknown", "contract",
            "Material source and formal symbols differ after applying the recorded mapping.",
        ),)
    definedness = audit_definedness(source.value, formal.value, limits=limits)
    definedness_relation = (
        "equivalent" if definedness.complete else
        "mismatch" if any(item.relation == "mismatch"
                          for item in definedness.findings) else "unknown"
    )
    fields += (FieldComparison(
        check=CheckKind.DEFINEDNESS,
        outcome=("match" if definedness.complete else
                 "difference" if definedness_relation == "mismatch" else "unsupported"),
        relation=ClaimRelation(definedness_relation),
        source_digest=_digest(source.value.get("definedness_obligations", [])),
        formal_digest=_digest(formal.value.get("definedness_obligations", [])),
        opaque=definedness_relation == "unknown",
        message=("Definedness obligations are discharged." if definedness.complete
                 else definedness.findings[0].message),
    ),)
    findings = findings + tuple(
        _finding(item.code, item.relation, "definedness", item.message,
                 item.severity)
        for item in definedness.findings
    )
    if source.opaque_paths or formal.opaque_paths:
        # Field checks normally emit the specific finding; this catches opaque
        # extension fields so equivalence still fails closed.
        if not any(item.code == "SC_OPAQUE_REQUIRED_FIELD" for item in findings):
            findings += (_finding("SC_OPAQUE_REQUIRED_FIELD", "unknown", "contract",
                                  "Opaque contract material prevents equivalence."),)
    relation = classify_relation(fields, findings)
    metadata = tuple(sorted((str(k), str(v)) for k, v in
                            (backend_metadata or {"workers": "1"}).items()))
    receipt = _make_receipt(source, formal, fields, findings, relation, backend, metadata)
    return ComparisonResult(
        relation=relation, field_results=fields, findings=findings, receipt=receipt,
        complete=relation != "unknown" and definedness.complete,
        backend=backend, backend_metadata=metadata,
    )


def verify_comparison_receipt(receipt: ComparisonReceipt | Mapping[str, Any], *,
                              source_contract: Any | None = None,
                              formal_contract: Any | None = None,
                              symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                              source_symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                              formal_symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                              definitions: Mapping[str, Any] | Sequence[Any] | None = None,
                              unfold_allowlist: Sequence[str] = (),
                              limits: NormalizationLimits | None = None) -> bool:
    """Verify receipt integrity and, when supplied, both normalized input digests."""
    value = _plain(receipt)
    if not isinstance(value, Mapping):
        return False
    try:
        expected = hashlib.sha256(canonical_json(_receipt_payload(**value))).hexdigest()
    except (KeyError, TypeError, ValueError):
        return False
    if (value.get("receipt_sha256") != expected
            or value.get("comparator_version") != COMPARATOR_VERSION
            or value.get("normalizer_version") != NORMALIZER_VERSION):
        return False
    if source_contract is not None:
        if normalize_contract(
            source_contract, symbol_mapping=source_symbol_mapping or symbol_mapping,
            definitions=definitions, unfold_allowlist=unfold_allowlist, limits=limits,
        ).sha256 != value.get("source_contract_sha256"):
            return False
    if formal_contract is not None:
        if normalize_contract(
            formal_contract, symbol_mapping=formal_symbol_mapping or symbol_mapping,
            definitions=definitions, unfold_allowlist=unfold_allowlist, limits=limits,
        ).sha256 != value.get("formal_contract_sha256"):
            return False
    return True


def comparison_digest(result: ComparisonResult | Mapping[str, Any]) -> str:
    """Digest the complete immutable comparison result."""
    return hashlib.sha256(canonical_json(_plain(result))).hexdigest()


def _compare_job(job: tuple[Any, Any, dict[str, Any]]) -> ComparisonResult:
    source, formal, kwargs = job
    return compare_contracts(source, formal, **kwargs)


def compare_contracts_batch(pairs: Sequence[tuple[Any, Any]], *, workers: int | None = None,
                            min_parallel: int = 4, **kwargs: Any) -> BatchComparisonResult:
    """Order-preserving process tier with an automatic reference fallback."""
    jobs = [(source, formal, dict(kwargs)) for source, formal in pairs]
    resolved = resolve_workers(workers)
    if resolved <= 1 or len(jobs) < min_parallel:
        metadata = (("workers", "1"), ("fallback", "not_needed"))
        results = tuple(compare_contracts(source, formal, backend="python-reference",
                                          backend_metadata=dict(metadata), **kwargs)
                        for source, formal in pairs)
        return BatchComparisonResult(results, "python-reference", metadata)
    try:
        metadata = (("workers", str(min(resolved, len(jobs)))),
                    ("fallback", "not_needed"))
        parallel_kwargs = dict(kwargs, backend="process-map",
                               backend_metadata=dict(metadata))
        jobs = [(source, formal, parallel_kwargs) for source, formal in pairs]
        results = tuple(process_map(_compare_job, jobs, workers=resolved,
                                    min_parallel=min_parallel))
        # The parallel tier invokes the same exact comparator and receipt
        # checker as the serial tier.  A malformed worker result fails closed.
        if not all(verify_comparison_result(item) for item in results):
            raise RuntimeError("parallel comparison receipt verification failed")
        return BatchComparisonResult(results, "process-map", metadata)
    except Exception as exc:
        metadata = (("workers", "1"), ("fallback", type(exc).__name__))
        results = tuple(compare_contracts(source, formal, backend="python-reference",
                                          backend_metadata=dict(metadata), **kwargs)
                        for source, formal in pairs)
        return BatchComparisonResult(results, "python-reference", metadata)


def verify_comparison_result(result: ComparisonResult | Mapping[str, Any]) -> bool:
    """Verify a complete result, including field/finding digests and receipt."""
    value = _plain(result)
    if not isinstance(value, Mapping):
        return False
    receipt = value.get("receipt")
    if not verify_comparison_receipt(receipt):
        return False
    receipt = _plain(receipt)
    fields = value.get("field_results")
    findings = value.get("findings")
    try:
        return (
            _digest(fields) == receipt["field_results_sha256"]
            and _digest(findings) == receipt["findings_sha256"]
            and value.get("relation") == receipt["relation"]
            and value.get("backend") == receipt["backend"]
            and tuple(tuple(item) for item in value.get("backend_metadata", ()))
            == tuple(tuple(item) for item in receipt["backend_metadata"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def compare_claim_contracts(source_contract: Any, formal_contract: Any, **kwargs: Any
                            ) -> ComparisonResult:
    """Compatibility alias for the primary comparison API."""
    return compare_contracts(source_contract, formal_contract, **kwargs)
