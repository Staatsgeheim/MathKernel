"""Conservative presupposition and risky-totalization checks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Any, Iterable, Mapping, Sequence

from mathkernel.parallel import process_map, resolve_workers
from .normalize import NormalizationLimits, normalize_value


@dataclass(frozen=True)
class DefinednessFinding:
    code: str
    severity: str
    relation: str
    message: str
    path: str
    construct: str
    obligation_kind: str


@dataclass(frozen=True)
class ObligationCheck:
    obligation_kind: str
    construct: str
    path: str
    discharged: bool
    discharge: str | None = None


@dataclass(frozen=True)
class DefinednessResult:
    complete: bool
    checks: tuple[ObligationCheck, ...]
    findings: tuple[DefinednessFinding, ...]
    backend: str = "python-reference"
    backend_metadata: tuple[tuple[str, str], ...] = (("workers", "1"),)


@dataclass(frozen=True)
class DefinednessBatchResult:
    results: tuple[DefinednessResult, ...]
    backend: str
    backend_metadata: tuple[tuple[str, str], ...]


_RISKS: dict[str, tuple[str, str, str]] = {
    "Nat.sInf": (
        "nonempty",
        "SC_DEFINEDNESS_TOTALIZED_EMPTY_SET",
        "Formal definition supplies a value for the empty case, while the source "
        "contract requires existence of a least witness.",
    ),
    "sInf": ("nonempty", "SC_DEFINEDNESS_INFIMUM_NONEMPTY",
             "Infimum requires a mapped non-emptiness/boundedness obligation."),
    "sSup": ("nonempty", "SC_DEFINEDNESS_SUPREMUM_NONEMPTY",
             "Supremum requires a mapped non-emptiness/boundedness obligation."),
    "min'": ("nonempty", "SC_DEFINEDNESS_MINIMUM_NONEMPTY",
             "Minimum requires a mapped non-emptiness obligation."),
    "max'": ("nonempty", "SC_DEFINEDNESS_MAXIMUM_NONEMPTY",
             "Maximum requires a mapped non-emptiness obligation."),
    "Classical.choose": ("exists", "SC_DEFINEDNESS_ARBITRARY_CHOICE",
                         "Choice requires a mapped existence obligation."),
    "Exists.choose": ("exists", "SC_DEFINEDNESS_ARBITRARY_CHOICE",
                      "Choice requires a mapped existence obligation."),
    "Classical.choice": ("exists", "SC_DEFINEDNESS_ARBITRARY_CHOICE",
                         "Choice requires a mapped existence obligation."),
    "Classical.epsilon": ("exists", "SC_DEFINEDNESS_ARBITRARY_CHOICE",
                          "Indefinite description requires a mapped existence obligation."),
    "Classical.indefiniteDescription": (
        "exists", "SC_DEFINEDNESS_ARBITRARY_CHOICE",
        "Indefinite description requires mapped existence and uniqueness obligations.",
    ),
    "Option.getD": ("no_fallback", "SC_DEFINEDNESS_FALLBACK_DEFAULT",
                    "Fallback semantics require proof that the fallback case is unreachable."),
    "getD": ("no_fallback", "SC_DEFINEDNESS_FALLBACK_DEFAULT",
             "Fallback semantics require proof that the fallback case is unreachable."),
    "get!": ("valid_index", "SC_DEFINEDNESS_INDEX_FALLBACK",
             "Fallback indexing requires a mapped valid-index obligation."),
    "default": ("no_default", "SC_DEFINEDNESS_DEFAULT_INHABITANT",
                "A default inhabitant may totalize a source-side partial construction."),
    "Inhabited.default": ("no_default", "SC_DEFINEDNESS_DEFAULT_INHABITANT",
                          "A default inhabitant may totalize a source-side partial construction."),
    "inv": ("nonzero", "SC_DEFINEDNESS_NONZERO_REQUIRED",
            "Inverse requires a mapped nonzero/invertibility obligation."),
}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)
    return value


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    value = _plain(value)
    yield path, value
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _walk(value[key], f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _construct(node: Any) -> str | None:
    if not isinstance(node, Mapping):
        return None
    kind = node.get("kind", node.get("op"))
    name = node.get("name", node.get("function", node.get("operator")))
    if isinstance(name, str):
        if name in _RISKS:
            return name
        # Lean extractor names may be fully qualified beyond the stable suffix.
        for candidate in _RISKS:
            if name.endswith("." + candidate):
                return candidate
    if kind in {"div", "/"}:
        return "division"
    if kind in {"inv", "inverse"}:
        return "inv"
    if kind in {"index_default", "get_default", "array_get!"}:
        return "get!"
    if kind in {"default", "fallback"}:
        return "default"
    return None


def detect_risky_constructs(formal: Any, *,
                            limits: NormalizationLimits | None = None
                            ) -> tuple[tuple[str, str, str], ...]:
    """Return deterministic ``(path, construct, obligation_kind)`` records."""
    normalized = normalize_value(formal, limits=limits).value
    found: list[tuple[str, str, str]] = []
    for path, node in _walk(normalized):
        construct = _construct(node)
        if construct is None:
            continue
        obligation = "nonzero" if construct == "division" else _RISKS[construct][0]
        found.append((path, construct, obligation))
    return tuple(found)


def _obligation_records(contract: Any) -> list[Mapping[str, Any]]:
    contract = _plain(contract)
    if not isinstance(contract, Mapping):
        return []
    values = contract.get("definedness_obligations", contract.get("obligations", ())) or ()
    records = []
    for value in values:
        value = _plain(value)
        if isinstance(value, Mapping):
            records.append(value)
    return records


def _assumption_text(contract: Any) -> str:
    contract = _plain(contract)
    if not isinstance(contract, Mapping):
        return ""
    relevant = {
        "assumptions": contract.get("assumptions", ()),
        "hypotheses": contract.get("hypotheses", ()),
        "proof_dependencies": contract.get("proof_dependencies", ()),
    }
    return json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str).lower()


def _record_kind(record: Mapping[str, Any]) -> str:
    return str(record.get("kind", record.get("category",
               record.get("obligation_kind", record.get("type", ""))))).lower()


def _canonical_kind(record: Mapping[str, Any]) -> str:
    value = _record_kind(record)
    for canonical, aliases in {
        "nonempty": {"nonempty", "non_empty"},
        "exists": {"existence", "exists", "choice"},
        "nonzero": {"nonzero", "non_zero", "invertible", "denominator_nonzero"},
        "valid_index": {"valid_index", "index_bounds", "in_bounds"},
        "no_fallback": {"fallback", "no_fallback", "defined", "is_some"},
        "partial_domain": {"partial_domain"},
        "convergence": {"convergence"},
        "coercion": {"coercion"},
        "uniqueness": {"uniqueness", "unique_existence"},
    }.items():
        if value in aliases:
            return canonical
    return value


def _record_discharged(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "")).lower()
    return bool(record.get("discharged") is True or record.get("proved") is True
                or bool(record.get("justification"))
                or status in {"discharged", "proved", "verified", "satisfied"})


def _kind_matches(record: Mapping[str, Any], kind: str) -> bool:
    aliases = {
        "nonempty": {"nonempty", "non_empty", "existence", "exists"},
        "exists": {"existence", "exists", "unique_existence"},
        "nonzero": {"nonzero", "non_zero", "invertible", "denominator_nonzero"},
        "valid_index": {"valid_index", "index_bounds", "in_bounds"},
        "no_fallback": {"no_fallback", "defined", "is_some"},
        "no_default": {"no_default", "defined", "inhabited_not_used"},
    }
    return _record_kind(record) in aliases.get(kind, {kind})


def _matches(record: Mapping[str, Any], kind: str, construct: str) -> bool:
    if not _kind_matches(record, kind):
        return False
    target = str(record.get("construct", record.get("target",
                 record.get("subject", ""))))
    return not target or target == construct or target.endswith("." + construct)


def _assumption_discharge(kind: str, construct: str, text: str) -> str | None:
    tokens = {
        "nonempty": ("nonempty", "set.nonempty"),
        "exists": ("exists_witness", "has_witness"),
        "nonzero": ("ne_zero", "nonzero", "invertible"),
        "valid_index": ("inbounds", "in_bounds", "valid_index"),
        "no_fallback": ("is_some",),
        "no_default": (),
    }[kind]
    if any(token in text for token in tokens):
        return "mapped formal assumption/proof dependency"
    return None


def audit_definedness(source_contract: Any, formal_contract: Any, *,
                      limits: NormalizationLimits | None = None) -> DefinednessResult:
    """Check source obligations against risky constructs in the formal contract.

    Merely seeing a risky operation is not called a mistranslation.  A finding is
    emitted only when no explicit discharged obligation or recognizable mapped
    formal hypothesis/proof dependency supports the required presupposition.
    """
    risks = detect_risky_constructs(formal_contract, limits=limits)
    source_obligations = _obligation_records(source_contract)
    formal_obligations = _obligation_records(formal_contract)
    formal_text = _assumption_text(formal_contract)
    checks: list[ObligationCheck] = []
    findings: list[DefinednessFinding] = []
    for path, construct, kind in risks:
        # Source obligations usually name the source set/predicate rather than
        # the Lean implementation primitive, so category matching is the sound
        # cross-language key here.  Formal discharge records may name the exact
        # risky construct and are matched more narrowly.
        source_records = [r for r in source_obligations if _kind_matches(r, kind)]
        formal_records = [r for r in formal_obligations if _matches(r, kind, construct)]
        discharge = next(("explicit formal obligation" for r in formal_records
                          if _record_discharged(r)), None)
        discharge = discharge or _assumption_discharge(kind, construct, formal_text)
        discharged = discharge is not None
        checks.append(ObligationCheck(kind, construct, path, discharged, discharge))
        if discharged:
            continue
        # If the source explicitly requires this presupposition, totalization is
        # a direct mismatch.  Otherwise it remains an unresolved error: absence
        # from a manifest cannot be interpreted as permission for fallback.
        if construct == "Nat.sInf" and source_records:
            _, code, message = _RISKS[construct]
            relation = "mismatch"
        elif construct in _RISKS:
            _, code, message = _RISKS[construct]
            relation = "mismatch" if source_records else "unknown"
        else:
            code = "SC_DEFINEDNESS_NONZERO_REQUIRED"
            message = "Division requires a mapped nonzero denominator obligation."
            relation = "mismatch" if source_records else "unknown"
        findings.append(DefinednessFinding(
            code=code, severity="error", relation=relation, message=message,
            path=path, construct=construct, obligation_kind=kind,
        ))
    # Explicit source presuppositions remain obligations even when the formal
    # expression does not contain one of the initially catalogued risky terms.
    covered: dict[str, int] = {}
    for check in checks:
        covered[check.obligation_kind] = covered.get(check.obligation_kind, 0) + 1
    for index, source_record in enumerate(source_obligations):
        if source_record.get("required", True) is False:
            continue
        kind = _canonical_kind(source_record)
        matching_kind = next((candidate for candidate in covered
                              if _kind_matches(source_record, candidate)
                              and covered[candidate] > 0), None)
        if matching_kind is not None:
            covered[matching_kind] -= 1
            continue
        formal_record = next(
            (record for record in formal_obligations
             if _canonical_kind(record) == kind and _record_discharged(record)),
            None,
        )
        discharge = ("explicit formal obligation" if formal_record is not None
                     else _assumption_discharge(kind, "", formal_text)
                     if kind in {"nonempty", "exists", "nonzero", "valid_index",
                                 "no_fallback", "no_default"} else None)
        discharged = discharge is not None
        path = f"$.definedness_obligations[{index}]"
        checks.append(ObligationCheck(kind, "", path, discharged, discharge))
        if not discharged:
            findings.append(DefinednessFinding(
                code="SC_DEFINEDNESS_OBLIGATION_UNDISCHARGED",
                severity="error",
                relation="unknown",
                message="A required source definedness obligation has no mapped "
                        "formal hypothesis or proof dependency.",
                path=path,
                construct="",
                obligation_kind=kind,
            ))
    return DefinednessResult(complete=not findings, checks=tuple(checks),
                             findings=tuple(findings))


def check_definedness(source_contract: Any, formal_contract: Any, **kwargs: Any
                      ) -> DefinednessResult:
    """Compatibility alias for the public definedness check."""
    return audit_definedness(source_contract, formal_contract, **kwargs)


def _definedness_job(job: tuple[Any, Any, dict[str, Any]]) -> DefinednessResult:
    return audit_definedness(job[0], job[1], **job[2])


def audit_definedness_batch(pairs: Sequence[tuple[Any, Any]], *,
                            workers: int | None = None, min_parallel: int = 4,
                            **kwargs: Any) -> DefinednessBatchResult:
    """Order-preserving process tier with a deterministic serial fallback."""
    resolved = resolve_workers(workers)
    if resolved <= 1 or len(pairs) < min_parallel:
        metadata = (("workers", "1"), ("fallback", "not_needed"))
        results = tuple(replace(audit_definedness(source, formal, **kwargs),
                                backend_metadata=metadata)
                        for source, formal in pairs)
        return DefinednessBatchResult(results, "python-reference", metadata)
    try:
        metadata = (("workers", str(min(resolved, len(pairs)))),
                    ("fallback", "not_needed"))
        raw = process_map(
            _definedness_job,
            [(source, formal, dict(kwargs)) for source, formal in pairs],
            workers=resolved,
            min_parallel=min_parallel,
        )
        results = tuple(replace(item, backend="process-map",
                                backend_metadata=metadata) for item in raw)
        return DefinednessBatchResult(results, "process-map", metadata)
    except Exception as exc:
        metadata = (("workers", "1"), ("fallback", type(exc).__name__))
        results = tuple(replace(audit_definedness(source, formal, **kwargs),
                                backend_metadata=metadata)
                        for source, formal in pairs)
        return DefinednessBatchResult(results, "python-reference", metadata)
