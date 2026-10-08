"""Deterministic, deliberately small normalization for correspondence contracts.

Normalization is syntactic and contract-relative.  It never drops hypotheses,
changes domains, invents symbol mappings, or uses mathematical equivalences
outside the explicit safe fragment in the correspondence design.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping, Sequence

from mathkernel.parallel import process_map, resolve_workers


NORMALIZER_VERSION = "mathkernel.correspondence-normalizer/v1"


@dataclass(frozen=True)
class NormalizationLimits:
    max_depth: int = 128
    max_nodes: int = 100_000
    max_collection: int = 20_000
    max_string: int = 1_000_000
    max_unfold_depth: int = 16


@dataclass(frozen=True)
class NormalizationResult:
    value: Any
    sha256: str
    backend: str = "python-reference"
    backend_metadata: tuple[tuple[str, str], ...] = (("workers", "1"),)
    normalizer_version: str = NORMALIZER_VERSION
    opaque_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizationBatchResult:
    results: tuple[NormalizationResult, ...]
    backend: str
    backend_metadata: tuple[tuple[str, str], ...]


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(value)
    return value


def canonical_json(value: Any) -> bytes:
    """Encode a normalized value without platform- or locale-dependent details."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def normalized_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _mapping_dict(mapping: Mapping[str, str] | Sequence[Any] | None) -> dict[str, str]:
    if mapping is None:
        return {}
    if isinstance(mapping, Mapping):
        pairs = list(mapping.items())
    else:
        pairs = []
        for item in mapping:
            item = _plain(item)
            if isinstance(item, Mapping):
                source = next((item.get(k) for k in
                               ("source_symbol", "source", "from_symbol", "formal_symbol")
                               if item.get(k) is not None), None)
                target = next((item.get(k) for k in
                               ("canonical_symbol", "target", "to_symbol", "formal_symbol")
                               if item.get(k) is not None and item.get(k) != source), None)
                if target is None and "formal_symbol" in item:
                    target = item["formal_symbol"]
                pairs.append((source, target))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                raise ValueError("Symbol mappings must be pairs or typed mapping records")
    result: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for source, target in pairs:
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target:
            raise ValueError("Symbol mappings require nonempty string symbols")
        if source == target:
            continue
        if source in result and result[source] != target:
            raise ValueError(f"Conflicting symbol mapping for {source!r}")
        if target in reverse and reverse[target] != source:
            raise ValueError(f"Non-injective symbol mapping to {target!r}")
        result[source], reverse[target] = target, source
    # A mapping is one recorded substitution, not a rewrite system.  Cycles are
    # nevertheless rejected because they almost always indicate an ambiguous spec.
    for start in result:
        seen: set[str] = set()
        current = start
        while current in result:
            if current in seen:
                raise ValueError("Cyclic symbol mapping")
            seen.add(current)
            current = result[current]
    return result


def _definition_table(definitions: Mapping[str, Any] | Sequence[Any] | None,
                      allowlist: Sequence[str]) -> dict[str, tuple[tuple[str, ...], Any]]:
    if not definitions:
        return {}
    records = definitions.items() if isinstance(definitions, Mapping) else (
        (None, item) for item in definitions
    )
    allowed = set(allowlist)
    table: dict[str, tuple[tuple[str, ...], Any]] = {}
    for key, record in records:
        value = _plain(record)
        if isinstance(value, Mapping):
            name = key or value.get("name") or value.get("symbol")
            body = value.get("body", value.get("expression", value.get("value")))
            params = value.get("parameters", value.get("params", ()))
            params = tuple(
                p if isinstance(p, str) else _plain(p).get("name", _plain(p).get("symbol"))
                for p in params
            )
        else:
            name, body, params = key, value, ()
        if name not in allowed:
            continue
        if not isinstance(name, str) or body is None or not all(isinstance(p, str) for p in params):
            raise ValueError("Malformed allowlisted definition")
        if name in table:
            raise ValueError(f"Duplicate definition {name!r}")
        table[name] = (params, body)
    return table


class _Normalizer:
    def __init__(self, mapping: dict[str, str], definitions, limits: NormalizationLimits):
        self.mapping = mapping
        self.definitions = definitions
        self.limits = limits
        self.nodes = 0
        self.opaque: list[str] = []

    def run(self, value: Any, *, path: str = "$", env: Mapping[str, str] | None = None,
            depth: int = 0, unfolding: tuple[str, ...] = ()) -> Any:
        if depth > self.limits.max_depth:
            raise ValueError("Normalization depth limit exceeded")
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise ValueError("Normalization node limit exceeded")
        value = _plain(value)
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            raise ValueError("Inexact floating-point values are not normalization-safe")
        if isinstance(value, str):
            if len(value) > self.limits.max_string:
                raise ValueError("Normalization string limit exceeded")
            return value
        if isinstance(value, (tuple, list, set, frozenset)):
            if len(value) > self.limits.max_collection:
                raise ValueError("Normalization collection limit exceeded")
            items = [self.run(v, path=f"{path}[{i}]", env=env, depth=depth + 1,
                              unfolding=unfolding) for i, v in enumerate(value)]
            if isinstance(value, (set, frozenset)):
                items.sort(key=canonical_json)
            return items
        if not isinstance(value, Mapping):
            raise ValueError(f"Unsupported normalization value at {path}: {type(value).__name__}")
        if len(value) > self.limits.max_collection:
            raise ValueError("Normalization mapping limit exceeded")

        node = dict(value)
        kind = str(node.get("kind", node.get("op", "")))
        original_call_name = (node.get("name", node.get("operator"))
                              if kind in {"call", "apply", "application"} else None)
        if kind in {"opaque", "unsupported", "unknown"} or node.get("opaque") is True:
            self.opaque.append(path)

        # Exact integer/rational canonicalization.
        if kind in {"integer", "nat", "int"} and "value" in node:
            try:
                canonical_integer = int(str(node["value"]))
                node["value"] = (canonical_integer if isinstance(node["value"], int)
                                 else str(canonical_integer))
            except ValueError as exc:
                raise ValueError(f"Invalid exact integer at {path}") from exc
        if kind == "rational" and isinstance(node.get("value"), str):
            try:
                rational = Fraction(node["value"])
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"Invalid exact rational at {path}") from exc
            node["value"] = f"{rational.numerator}/{rational.denominator}"
        elif kind == "rational" or ("numerator" in node and "denominator" in node):
            try:
                rational = Fraction(int(str(node["numerator"])), int(str(node["denominator"])))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"Invalid exact rational at {path}") from exc
            node["numerator"], node["denominator"] = str(rational.numerator), str(rational.denominator)

        local_env = dict(env or {})
        binder_key = next((key for key in ("variable", "binder", "bound_symbol")
                           if isinstance(node.get(key), str)), None)
        if (kind in {"quantifier", "forall", "exists", "binder", "lambda"}
                and binder_key and "body" in node):
            old = node[binder_key]
            fresh = f"_b{len(local_env)}"
            node[binder_key] = fresh
            local_env[old] = fresh

        # Symbols are changed only by alpha-renaming or an explicit mapping.
        if kind in {"symbol", "identifier", "var"}:
            key = ("name" if "name" in node else "symbol" if "symbol" in node
                   else "value" if isinstance(node.get("value"), str) else None)
            if key and isinstance(node[key], str):
                node[key] = local_env.get(node[key], self.mapping.get(node[key], node[key]))
        for key in ("subject", "quantity", "variable", "type_name"):
            if isinstance(node.get(key), str):
                node[key] = local_env.get(node[key], self.mapping.get(node[key], node[key]))
        if "symbol" in node and kind not in {"symbol", "identifier", "var"}:
            if isinstance(node["symbol"], str):
                node["symbol"] = local_env.get(
                    node["symbol"], self.mapping.get(node["symbol"], node["symbol"]))
        if isinstance(node.get("depends_on"), (list, tuple)):
            node["depends_on"] = [
                local_env.get(item, self.mapping.get(item, item))
                if isinstance(item, str) else item for item in node["depends_on"]
            ]
        if kind in {"call", "apply", "application"} and isinstance(node.get("operator"), str):
            node["operator"] = self.mapping.get(node["operator"], node["operator"])

        # Transparent unfolding is opt-in and only substitutes exact parameters.
        call_name = (original_call_name if original_call_name in self.definitions
                     else node.get("name", node.get("operator"))
                     if kind in {"call", "apply", "application"} else None)
        if call_name in self.definitions:
            if call_name in unfolding or len(unfolding) >= self.limits.max_unfold_depth:
                raise ValueError(f"Cyclic or excessive unfolding of {call_name!r}")
            params, body = self.definitions[call_name]
            args = node.get("args", node.get("arguments", ()))
            if len(params) != len(args):
                raise ValueError(f"Arity mismatch while unfolding {call_name!r}")
            substitutions = dict(zip(params, args))
            body = self._substitute(body, substitutions, path)
            return self.run(body, path=path, env=local_env, depth=depth + 1,
                            unfolding=unfolding + (call_name,))

        result: dict[str, Any] = {}
        for key in sorted(node):
            child_env = local_env if key in {"body", "conclusion", "predicate", "expression"} else env
            result[key] = self.run(node[key], path=f"{path}.{key}", env=child_env,
                                   depth=depth + 1, unfolding=unfolding)

        # Normalize inequality direction without changing strictness.
        result_kind = result.get("kind", result.get("op"))
        operation = result.get("operator") if result_kind == "operator" else result_kind
        if operation in {"gt", "ge", ">", ">="}:
            replacement = {"gt": "lt", "ge": "le", ">": "<", ">=": "<="}[operation]
            if "kind" in result:
                if result_kind == "operator":
                    result["operator"] = replacement
                else:
                    result["kind"] = replacement
            else:
                result["op"] = replacement
            if "left" in result and "right" in result:
                result["left"], result["right"] = result["right"], result["left"]
            elif isinstance(result.get("arguments"), list) and len(result["arguments"]) == 2:
                result["arguments"][0], result["arguments"][1] = (
                    result["arguments"][1], result["arguments"][0])

        # Only explicitly commutative structures are sorted.
        commutative = (
            operation in {"and", "conjunction", "finite_set"}
            or (result_kind == "bool" and result.get("op") == "and")
            or (result_kind == "set" and result.get("elements") is not None)
            or (result_kind == "setop" and result.get("op") in {"union", "intersect"})
        )
        if commutative:
            for key in ("args", "arguments", "operands", "elements"):
                if isinstance(result.get(key), list):
                    result[key] = sorted(result[key], key=canonical_json)
        return result

    def _substitute(self, value: Any, substitutions: Mapping[str, Any], path: str) -> Any:
        value = _plain(value)
        if isinstance(value, Mapping):
            kind = value.get("kind")
            key = ("name" if "name" in value else "symbol" if "symbol" in value
                   else "value" if isinstance(value.get("value"), str) else None)
            if kind in {"symbol", "identifier", "var"} and key and value[key] in substitutions:
                return substitutions[value[key]]
            return {k: self._substitute(v, substitutions, f"{path}.{k}")
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._substitute(v, substitutions, path) for v in value]
        return value


def normalize_value(value: Any, *, symbol_mapping: Mapping[str, str] | Sequence[Any] | None = None,
                    definitions: Mapping[str, Any] | Sequence[Any] | None = None,
                    unfold_allowlist: Sequence[str] = (),
                    limits: NormalizationLimits | None = None) -> NormalizationResult:
    limits = limits or NormalizationLimits()
    normalizer = _Normalizer(_mapping_dict(symbol_mapping),
                             _definition_table(definitions, unfold_allowlist), limits)
    normalized = normalizer.run(value)
    return NormalizationResult(value=normalized, sha256=normalized_digest(normalized),
                               opaque_paths=tuple(normalizer.opaque))


def normalize_expression(expression: Any, **kwargs: Any) -> Any:
    """Return only the canonical expression; use normalize_value for its receipt."""
    return normalize_value(expression, **kwargs).value


def normalize_contract(contract: Any, **kwargs: Any) -> NormalizationResult:
    """Normalize a complete claim contract and bind its canonical bytes."""
    limits = kwargs.pop("limits", None) or NormalizationLimits()
    mapping = _mapping_dict(kwargs.pop("symbol_mapping", None))
    definitions = _definition_table(kwargs.pop("definitions", None),
                                    kwargs.pop("unfold_allowlist", ()))
    if kwargs:
        raise TypeError("Unexpected normalization options: " + ", ".join(sorted(kwargs)))
    normalizer = _Normalizer(mapping, definitions, limits)
    value = _plain(contract)
    env: dict[str, str] = {}
    if isinstance(value, Mapping):
        value = dict(value)
        binders_key = next((key for key in ("binders", "quantifiers")
                            if isinstance(value.get(key), (list, tuple))), None)
        if binders_key:
            binders = []
            for index, original in enumerate(value[binders_key]):
                binder = _plain(original)
                if not isinstance(binder, Mapping):
                    raise ValueError("Contract binders must be typed records")
                binder = dict(binder)
                name_key = next((key for key in ("name", "symbol", "variable")
                                 if isinstance(binder.get(key), str)), None)
                if name_key:
                    old, fresh = binder[name_key], f"_b{index}"
                    binder[name_key] = fresh
                    env[old] = fresh
                binders.append(binder)
            value[binders_key] = binders
    normalized = normalizer.run(value, env=env)
    return NormalizationResult(value=normalized, sha256=normalized_digest(normalized),
                               opaque_paths=tuple(normalizer.opaque))


def verify_normalization(result: NormalizationResult | Mapping[str, Any]) -> bool:
    result = _plain(result)
    if isinstance(result, Mapping):
        value, digest = result.get("value"), result.get("sha256")
        version = result.get("normalizer_version", NORMALIZER_VERSION)
    else:
        return False
    return (version == NORMALIZER_VERSION and isinstance(digest, str)
            and normalized_digest(value) == digest)


def _normalize_contract_job(job: tuple[Any, dict[str, Any]]) -> NormalizationResult:
    return normalize_contract(job[0], **job[1])


def normalize_contracts_batch(contracts: Sequence[Any], *, workers: int | None = None,
                              min_parallel: int = 4,
                              **kwargs: Any) -> NormalizationBatchResult:
    """Order-preserving process tier with exact receipt checks and fallback."""
    resolved = resolve_workers(workers)
    if resolved <= 1 or len(contracts) < min_parallel:
        metadata = (("workers", "1"), ("fallback", "not_needed"))
        results = tuple(replace(normalize_contract(contract, **kwargs),
                                backend_metadata=metadata)
                        for contract in contracts)
        return NormalizationBatchResult(results, "python-reference", metadata)
    try:
        metadata = (("workers", str(min(resolved, len(contracts)))),
                    ("fallback", "not_needed"))
        raw = process_map(
            _normalize_contract_job,
            [(contract, dict(kwargs)) for contract in contracts],
            workers=resolved,
            min_parallel=min_parallel,
        )
        results = tuple(replace(item, backend="process-map",
                                backend_metadata=metadata) for item in raw)
        if not all(verify_normalization(item) for item in results):
            raise RuntimeError("parallel normalization receipt verification failed")
        return NormalizationBatchResult(results, "process-map", metadata)
    except Exception as exc:
        metadata = (("workers", "1"), ("fallback", type(exc).__name__))
        results = tuple(replace(normalize_contract(contract, **kwargs),
                                backend_metadata=metadata)
                        for contract in contracts)
        return NormalizationBatchResult(results, "python-reference", metadata)


def normalize_claim_contract(contract: Any, **kwargs: Any) -> NormalizationResult:
    """Compatibility alias for typed claim contracts."""
    return normalize_contract(contract, **kwargs)
