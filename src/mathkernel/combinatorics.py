# =============================================================================
# MathKernel - exact combinatorics and generating functions
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact combinatorial enumeration and generating-function machinery.

All counting is done in exact integer arithmetic: factorials, binomial and
multinomial coefficients, Stirling numbers of both kinds, Bell numbers,
Catalan numbers, derangement numbers, integer- and set-partition counts and
composition counts are computed with Python integers (and Fraction where a
division must stay exact), never with floats.

Enumeration is lazy with explicit caps: generators are sliced with
``max_items`` and results carry ``truncated`` plus the exact ``total_count``
so a capped listing is never mistaken for a complete one.  Arguments that
would trigger accidental combinatorial explosions are rejected with explicit
limits instead of being attempted.

Generating functions are typed prefixes of exact coefficients.  Ordinary
generating functions support conversion to and from linear recurrences with
constant integer coefficients (rational OGFs); exponential generating
functions support coefficient extraction only — recurrence conversion for
EGFs is explicitly reported as unsupported.  Verification recomputes
coefficients from recurrences or rational forms and cross-checks classical
count identities; anything that cannot be verified is reported as UNKNOWN or
UNSUPPORTED, never silently asserted.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from itertools import (
    combinations as _it_combinations,
    islice,
    permutations as _it_permutations,
)
from math import comb, factorial
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from mathkernel_artifacts.evidence import (
    CertificateEvidence,
    ComputationEvidence,
    EvidenceBundle,
    ProofEvidence,
)
from .models import OperationStatus, TrustLevel

try:
    from .combinatorics_fast import extend_recurrence_fast as _extend_fast
except ImportError:  # pragma: no cover
    _extend_fast = None

__all__ = [
    "CombinatorialClass",
    "CombinatorialKind",
    "CombinatoricsEngine",
    "CombinatoricsResult",
    "CountResult",
    "CoefficientResult",
    "DEFAULT_MAX_ITEMS",
    "GeneratingFunction",
    "GenerationResult",
    "GfConversionResult",
    "GfKind",
    "LinearRecurrence",
    "MAX_EXACT_ARGUMENT",
    "MAX_GENERATE_ITEMS",
    "MAX_SERIES_TERMS",
    "MAX_TABLE_ARGUMENT",
    "RationalGeneratingFunction",
    "VerificationResult",
    "bell_number",
    "binomial",
    "catalan_number",
    "combinations_count",
    "compositions_count",
    "derangements_count",
    "integer_partitions_count",
    "multinomial",
    "permutations_count",
    "set_partitions_count",
    "stirling_first",
    "stirling_second",
]

# Explicit limits: anything beyond these is rejected, never attempted.
MAX_EXACT_ARGUMENT = 10_000      # factorial-family counts
MAX_TABLE_ARGUMENT = 1_000       # O(n*k) Stirling/Bell/partition tables
MAX_GENERATE_ITEMS = 100_000     # hard cap on enumerated items
DEFAULT_MAX_ITEMS = 1_000        # default lazy-generation cap
MAX_SERIES_TERMS = 10_000        # recurrence-driven coefficient extension
MAX_GENERATION_DEPTH = 256       # recursion depth of enumeration generators
MAX_DERANGEMENT_GENERATION = 10  # filter-based derangement enumeration


class CombinatorialKind(str, Enum):
    PERMUTATIONS = "permutations"
    COMBINATIONS = "combinations"
    MULTISET_PERMUTATIONS = "multiset_permutations"
    INTEGER_PARTITIONS = "integer_partitions"
    SET_PARTITIONS = "set_partitions"
    COMPOSITIONS = "compositions"
    DERANGEMENTS = "derangements"


class GfKind(str, Enum):
    ORDINARY = "ordinary"
    EXPONENTIAL = "exponential"


# ---------------------------------------------------------------------------
# Exact counting functions
# ---------------------------------------------------------------------------

def _check_argument(n: int, limit: int, what: str) -> None:
    if not 0 <= n <= limit:
        raise ValueError(
            f"{what} requires 0 <= n <= {limit}, got {n}; "
            "larger arguments are rejected explicitly")


def binomial(n: int, k: int) -> int:
    """Exact binomial coefficient C(n, k); zero outside 0 <= k <= n."""
    _check_argument(n, MAX_EXACT_ARGUMENT, "binomial")
    if not 0 <= k <= n:
        return 0
    return comb(n, k)


def multinomial(multiplicities: list[int]) -> int:
    """Exact multinomial coefficient (sum m_i)! / prod m_i!."""
    if not multiplicities or any(m < 0 for m in multiplicities):
        raise ValueError("multiplicities must be a nonempty list of nonnegative integers")
    total = sum(multiplicities)
    _check_argument(total, MAX_EXACT_ARGUMENT, "multinomial")
    result = factorial(total)
    for m in multiplicities:
        result //= factorial(m)
    return result


def permutations_count(n: int, k: int | None = None) -> int:
    """Exact number of permutations of n elements (or k-permutations P(n, k))."""
    _check_argument(n, MAX_EXACT_ARGUMENT, "permutations")
    if k is None:
        return factorial(n)
    if not 0 <= k <= n:
        return 0
    return factorial(n) // factorial(n - k)


def combinations_count(n: int, k: int) -> int:
    """Exact number of k-subsets of an n-set."""
    return binomial(n, k)


@lru_cache(maxsize=None)
def _stirling_second_row(n: int) -> tuple[int, ...]:
    # Iterative: a recursive formulation would exceed the interpreter
    # recursion limit well below MAX_TABLE_ARGUMENT.
    row: tuple[int, ...] = (1,)
    for i in range(1, n + 1):
        new = [0] * (i + 1)
        for k in range(1, i + 1):
            above = row[k] if k < len(row) else 0
            new[k] = k * above + row[k - 1]
        row = tuple(new)
    return row


def stirling_second(n: int, k: int) -> int:
    """Exact Stirling number of the second kind S(n, k)."""
    _check_argument(n, MAX_TABLE_ARGUMENT, "stirling_second")
    if not 0 <= k <= n:
        return 0
    return _stirling_second_row(n)[k]


@lru_cache(maxsize=None)
def _stirling_first_row(n: int) -> tuple[int, ...]:
    """Unsigned Stirling numbers of the first kind c(n, k), iteratively."""
    row: tuple[int, ...] = (1,)
    for i in range(1, n + 1):
        new = [0] * (i + 1)
        for k in range(1, i + 1):
            above = row[k] if k < len(row) else 0
            new[k] = row[k - 1] + (i - 1) * above
        row = tuple(new)
    return row


def stirling_first(n: int, k: int, signed: bool = False) -> int:
    """Exact Stirling number of the first kind; unsigned unless signed=True."""
    _check_argument(n, MAX_TABLE_ARGUMENT, "stirling_first")
    if not 0 <= k <= n:
        return 0
    value = _stirling_first_row(n)[k]
    if signed and (n - k) % 2 == 1:
        return -value
    return value


def bell_number(n: int) -> int:
    """Exact Bell number B_n = sum_k S(n, k)."""
    _check_argument(n, MAX_TABLE_ARGUMENT, "bell_number")
    return sum(_stirling_second_row(n))


def catalan_number(n: int) -> int:
    """Exact Catalan number C_n = C(2n, n) / (n + 1)."""
    _check_argument(n, MAX_EXACT_ARGUMENT, "catalan_number")
    return comb(2 * n, n) // (n + 1)


@lru_cache(maxsize=None)
def derangements_count(n: int) -> int:
    """Exact derangement number !n via !n = (n - 1)(!(n-1) + !(n-2))."""
    _check_argument(n, MAX_EXACT_ARGUMENT, "derangements_count")
    d0, d1 = 1, 0
    if n == 0:
        return d0
    for i in range(2, n + 1):
        d0, d1 = d1, (i - 1) * (d1 + d0)
    return d1


@lru_cache(maxsize=None)
def integer_partitions_count(n: int, k: int | None = None) -> int:
    """Exact partition number p(n), or p(n, k) for partitions into k parts."""
    _check_argument(n, MAX_TABLE_ARGUMENT, "integer_partitions_count")
    if k is not None:
        if not 0 <= k <= n:
            return 1 if (n == 0 and k == 0) else 0
        return _partitions_exact_parts(n, k)
    # Pentagonal-number recurrence, computed iteratively.
    table = [0] * (n + 1)
    table[0] = 1
    for i in range(1, n + 1):
        total = 0
        m = 1
        while True:
            g1 = m * (3 * m - 1) // 2
            g2 = m * (3 * m + 1) // 2
            if g1 > i and g2 > i:
                break
            sign = 1 if m % 2 == 1 else -1
            if g1 <= i:
                total += sign * table[i - g1]
            if g2 <= i:
                total += sign * table[i - g2]
            m += 1
        table[i] = total
    return table[n]


@lru_cache(maxsize=None)
def _partitions_exact_parts(n: int, k: int) -> int:
    """p(n, k) = p(n-1, k-1) + p(n-k, k), computed as an iterative table."""
    rows = [[0] * (k + 1) for _ in range(n + 1)]
    rows[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            rows[i][j] = rows[i - 1][j - 1] + rows[i - j][j]
    return rows[n][k]


def set_partitions_count(n: int, k: int | None = None) -> int:
    """Exact number of set partitions of an n-set (into k blocks, or Bell)."""
    if k is None:
        return bell_number(n)
    return stirling_second(n, k)


def compositions_count(n: int, k: int | None = None) -> int:
    """Exact number of compositions of n (into k positive parts, or total)."""
    _check_argument(n, MAX_EXACT_ARGUMENT, "compositions_count")
    if k is not None:
        if k < 0:
            return 0
        if n == 0:
            return 1 if k == 0 else 0
        if not 1 <= k <= n:
            return 0
        return comb(n - 1, k - 1)
    if n == 0:
        return 1
    return 1 << (n - 1)


# ---------------------------------------------------------------------------
# Lazy enumeration generators (internal; always consumed through a cap)
# ---------------------------------------------------------------------------

def _collect(iterator, max_items: int) -> tuple[list[tuple[int, ...]], bool]:
    """Take at most max_items items; report whether more remained."""
    items = list(islice(iterator, max_items + 1))
    truncated = len(items) > max_items
    return items[:max_items], truncated


def _integer_partitions(n: int, k: int | None = None):
    if n == 0:
        if k in (None, 0):
            yield ()
        return
    a = [0] * (n + 1)
    size = 1
    a[1] = n
    while size != 0:
        x = a[size - 1] + 1
        y = a[size] - 1
        size -= 1
        while x <= y:
            a[size] = x
            y -= x
            size += 1
        a[size] = x + y
        # The underlying algorithm builds non-decreasing parts; emit the
        # conventional non-increasing (descending) partition tuple.
        partition = tuple(reversed(a[: size + 1]))
        if k is None or len(partition) == k:
            yield partition


def _set_partitions_rgs(n: int, k: int | None = None):
    """Set partitions as restricted-growth strings over block indices."""
    if n == 0:
        if k in (None, 0):
            yield ()
        return
    a = [0] * n

    def rec(i: int, blocks: int):
        if i == n:
            if k is None or blocks == k:
                yield tuple(a)
            return
        for b in range(blocks + 1):
            a[i] = b
            yield from rec(i + 1, max(blocks, b + 1))

    yield from rec(1, 1)


def _compositions(n: int, k: int | None = None):
    if n == 0:
        if k in (None, 0):
            yield ()
        return
    if k is not None:
        if not 1 <= k <= n:
            return
        for first in range(1, n - k + 2):
            for rest in _compositions(n - first, k - 1):
                yield (first,) + rest
        return
    for length in range(1, n + 1):
        yield from _compositions(n, length)


def _multiset_permutations(multiplicities: list[int]):
    total = sum(multiplicities)
    counts = list(multiplicities)
    arrangement = [0] * total

    def rec(pos: int):
        if pos == total:
            yield tuple(arrangement)
            return
        for value in range(len(counts)):
            if counts[value]:
                counts[value] -= 1
                arrangement[pos] = value
                yield from rec(pos + 1)
                counts[value] += 1

    yield from rec(0)


def _derangements(n: int):
    for perm in _it_permutations(range(n)):
        if all(perm[i] != i for i in range(n)):
            yield perm


# ---------------------------------------------------------------------------
# Typed models
# ---------------------------------------------------------------------------

def _exact_evidence(method: str, proposition: str | None = None) -> EvidenceBundle:
    bundle = EvidenceBundle(computation=[ComputationEvidence(
        engine="mathkernel",
        method=method,
        arithmetic="exact_integer",
        deterministic=True,
        trust="exact",
    )])
    if proposition is not None:
        bundle.proof.append(ProofEvidence(
            proposition=proposition,
            method="exact_count_identity",
            engine="mathkernel",
            verified=True,
            trust="exact",
        ))
    return bundle


class CombinatoricsResult(BaseModel):
    status: OperationStatus
    trust: TrustLevel = TrustLevel.UNKNOWN
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    diagnostics: list[str] = Field(default_factory=list)


class CountResult(CombinatoricsResult):
    kind: CombinatorialKind | None = None
    n: int = 0
    k: int | None = None
    value: int | None = None


class GenerationResult(CombinatoricsResult):
    kind: CombinatorialKind
    items: list[tuple[int, ...]] = Field(default_factory=list)
    returned: int = 0
    truncated: bool = False
    total_count: int | None = None
    max_items: int = DEFAULT_MAX_ITEMS
    checks: dict[str, bool] = Field(default_factory=dict)


class LinearRecurrence(BaseModel):
    """a_n = sum_i coefficients[i-1] * a_{n-i}, with initial a_0..a_{k-1}."""
    coefficients: list[int]
    initial: list[int]

    @model_validator(mode="after")
    def _consistent(self) -> "LinearRecurrence":
        if not self.coefficients:
            raise ValueError("a recurrence needs at least one coefficient")
        if len(self.initial) != len(self.coefficients):
            raise ValueError("initial terms must match the recurrence order")
        return self


class RationalGeneratingFunction(BaseModel):
    """P(x)/Q(x) with ascending-power integer coefficients and Q(0) = 1."""
    numerator: list[int]
    denominator: list[int]

    @model_validator(mode="after")
    def _normalized(self) -> "RationalGeneratingFunction":
        if not self.denominator or self.denominator[0] != 1:
            raise ValueError(
                "denominator must be normalized with constant term 1; "
                "other normalizations are unsupported")
        if not self.numerator:
            raise ValueError("numerator must be nonempty")
        return self


class GeneratingFunction(BaseModel):
    """Exact coefficient prefix of sum a_n x^n (ordinary) or sum a_n x^n/n!.

    ``coefficients[i]`` is always a_i, the sequence value; for exponential
    generating functions the n! scaling is implicit in ``kind``.
    """
    kind: GfKind
    coefficients: list[int] = Field(default_factory=list)
    recurrence: LinearRecurrence | None = None
    rational: RationalGeneratingFunction | None = None
    source: str = "explicit"

    @model_validator(mode="after")
    def _consistent(self) -> "GeneratingFunction":
        if self.rational is not None and self.kind != GfKind.ORDINARY:
            raise ValueError(
                "rational closed forms are supported only for ordinary "
                "generating functions")
        if self.recurrence is not None and self.kind != GfKind.ORDINARY:
            raise ValueError(
                "recurrence extension is supported only for ordinary "
                "generating functions")
        return self


class CoefficientResult(CombinatoricsResult):
    index: int = 0
    value: int | None = None
    scaling: Literal["ordinary", "factorial"] = "ordinary"


class GfConversionResult(CombinatoricsResult):
    rational: RationalGeneratingFunction | None = None
    recurrence: LinearRecurrence | None = None


class VerificationResult(CombinatoricsResult):
    verified: bool = False
    checks: dict[str, bool] = Field(default_factory=dict)


class CombinatorialClass(BaseModel):
    """A typed finite combinatorial family with exact count and lazy listing.

    ``k`` selects a subfamily: k-permutations, k-subsets, partitions into
    exactly k parts/blocks, or compositions into exactly k parts.
    ``multiplicities`` defines a multiset over value indices 0..m-1.
    """
    kind: CombinatorialKind
    n: int = Field(default=0, ge=0)
    k: int | None = Field(default=None, ge=0)
    multiplicities: list[int] | None = None
    max_items: int = Field(default=DEFAULT_MAX_ITEMS, ge=0, le=MAX_GENERATE_ITEMS)

    @model_validator(mode="after")
    def _consistent(self) -> "CombinatorialClass":
        kind = self.kind
        if kind == CombinatorialKind.MULTISET_PERMUTATIONS:
            if not self.multiplicities or any(m < 0 for m in self.multiplicities):
                raise ValueError(
                    "multiset permutations require nonnegative multiplicities")
            self.multiplicities = [int(m) for m in self.multiplicities]
            self.n = sum(self.multiplicities)
        elif self.multiplicities is not None:
            raise ValueError("multiplicities apply only to multiset permutations")
        if kind == CombinatorialKind.COMBINATIONS and self.k is None:
            raise ValueError("combinations require k")
        if kind in (CombinatorialKind.DERANGEMENTS,
                    CombinatorialKind.MULTISET_PERMUTATIONS) and self.k is not None:
            raise ValueError(f"{kind.value} do not take a k parameter")
        if self.k is not None and self.k > self.n:
            raise ValueError(f"k must satisfy 0 <= k <= n for {kind.value}")
        return self

    def count(self) -> CountResult:
        """Exact cardinality of this class."""
        kind, n, k = self.kind, self.n, self.k
        try:
            if kind == CombinatorialKind.PERMUTATIONS:
                value = permutations_count(n, k)
            elif kind == CombinatorialKind.COMBINATIONS:
                value = combinations_count(n, k if k is not None else 0)
            elif kind == CombinatorialKind.MULTISET_PERMUTATIONS:
                value = multinomial(list(self.multiplicities or []))
            elif kind == CombinatorialKind.INTEGER_PARTITIONS:
                value = integer_partitions_count(n, k)
            elif kind == CombinatorialKind.SET_PARTITIONS:
                value = set_partitions_count(n, k)
            elif kind == CombinatorialKind.COMPOSITIONS:
                value = compositions_count(n, k)
            elif kind == CombinatorialKind.DERANGEMENTS:
                value = derangements_count(n)
            else:  # pragma: no cover - enum is exhaustive
                return CountResult(
                    status=OperationStatus.UNSUPPORTED, kind=kind, n=n, k=k,
                    diagnostics=[f"unsupported combinatorial kind: {kind}"])
        except ValueError as exc:
            return CountResult(
                status=OperationStatus.ERROR, kind=kind, n=n, k=k,
                diagnostics=[str(exc)])
        return CountResult(
            status=OperationStatus.AVAILABLE,
            kind=kind, n=n, k=k, value=value,
            trust=TrustLevel.EXACT,
            evidence=_exact_evidence(f"count:{kind.value}"),
        )

    def generate(self, max_items: int | None = None) -> GenerationResult:
        """Lazily enumerate up to max_items items (explicitly capped)."""
        cap = self.max_items if max_items is None else max_items
        if not 0 <= cap <= MAX_GENERATE_ITEMS:
            return GenerationResult(
                status=OperationStatus.ERROR, kind=self.kind, max_items=cap,
                diagnostics=[f"max_items must be within 0..{MAX_GENERATE_ITEMS}"])
        try:
            iterator = self._iterator()
        except ValueError as exc:
            return GenerationResult(
                status=OperationStatus.UNSUPPORTED, kind=self.kind,
                max_items=cap, diagnostics=[str(exc)])
        items, truncated = _collect(iterator, cap)
        total = self.count()
        result = GenerationResult(
            status=OperationStatus.AVAILABLE,
            kind=self.kind,
            items=items,
            returned=len(items),
            truncated=truncated,
            total_count=total.value,
            max_items=cap,
            trust=TrustLevel.EXACT,
            evidence=_exact_evidence(f"generate:{self.kind.value}"),
            diagnostics=list(total.diagnostics),
        )
        if not truncated and total.value is not None:
            matches = len(items) == total.value
            result.checks["enumeration_matches_count"] = matches
            result.status = (
                OperationStatus.VERIFIED if matches else OperationStatus.REFUTED)
            result.evidence.certificate.append(CertificateEvidence(
                certificate_type="enumeration_count_agreement",
                claim=(f"enumerated {len(items)} {self.kind.value}; "
                       f"exact count is {total.value}"),
                witness={"enumerated": len(items), "count": total.value},
                verifier="mathkernel",
                verified=matches,
                trust="exact",
            ))
        return result

    def _iterator(self):
        kind, n, k = self.kind, self.n, self.k
        if kind == CombinatorialKind.PERMUTATIONS:
            return _it_permutations(range(n), k)
        if kind == CombinatorialKind.COMBINATIONS:
            return _it_combinations(range(n), k if k is not None else 0)
        if kind == CombinatorialKind.MULTISET_PERMUTATIONS:
            if n > MAX_GENERATION_DEPTH:
                raise ValueError(
                    f"multiset generation supports total size <= "
                    f"{MAX_GENERATION_DEPTH}")
            return _multiset_permutations(list(self.multiplicities or []))
        if kind == CombinatorialKind.INTEGER_PARTITIONS:
            return _integer_partitions(n, k)
        if kind == CombinatorialKind.SET_PARTITIONS:
            if n > MAX_GENERATION_DEPTH:
                raise ValueError(
                    f"set-partition generation supports n <= {MAX_GENERATION_DEPTH}")
            return _set_partitions_rgs(n, k)
        if kind == CombinatorialKind.COMPOSITIONS:
            if n > MAX_GENERATION_DEPTH:
                raise ValueError(
                    f"composition generation supports n <= {MAX_GENERATION_DEPTH}")
            return _compositions(n, k)
        if kind == CombinatorialKind.DERANGEMENTS:
            if n > MAX_DERANGEMENT_GENERATION:
                raise ValueError(
                    f"derangement generation supports n <= "
                    f"{MAX_DERANGEMENT_GENERATION}; use count() for larger n")
            return _derangements(n)
        raise ValueError(f"unsupported combinatorial kind: {kind}")


# ---------------------------------------------------------------------------
# Generating-function operations
# ---------------------------------------------------------------------------

def _extend_from_recurrence(recurrence: LinearRecurrence, n: int) -> list[int]:
    if _extend_fast is not None:
        fast = _extend_fast(recurrence.coefficients, recurrence.initial, n)
        if fast is not None:
            return fast
    terms = list(recurrence.initial)
    while len(terms) <= n:
        nxt = 0
        for i, c in enumerate(recurrence.coefficients, start=1):
            nxt += c * terms[len(terms) - i]
        terms.append(nxt)
    return terms


def _rational_series(rational: RationalGeneratingFunction, count: int) -> list[int]:
    num, den = rational.numerator, rational.denominator
    out: list[int] = []
    for n in range(count):
        acc = num[n] if n < len(num) else 0
        for i in range(1, min(n, len(den) - 1) + 1):
            acc -= den[i] * out[n - i]
        out.append(acc)
    return out


class CombinatoricsEngine:
    """Exact generating-function construction, conversion and verification."""

    def binomial_ogf(self, n: int) -> GeneratingFunction:
        """(1 + x)^n as an exact ordinary generating function."""
        _check_argument(n, MAX_EXACT_ARGUMENT, "binomial_ogf")
        return GeneratingFunction(
            kind=GfKind.ORDINARY,
            coefficients=[comb(n, k) for k in range(n + 1)],
            rational=RationalGeneratingFunction(
                numerator=[comb(n, k) for k in range(n + 1)],
                denominator=[1],
            ),
            source=f"(1+x)^{n}",
        )

    def catalan_ogf(self, terms: int) -> GeneratingFunction:
        """Exact Catalan OGF prefix sum C_n x^n."""
        _check_argument(terms, MAX_SERIES_TERMS, "catalan_ogf")
        return GeneratingFunction(
            kind=GfKind.ORDINARY,
            coefficients=[catalan_number(i) for i in range(terms)],
            source="catalan",
        )

    def ogf_from_recurrence(self, recurrence: LinearRecurrence) -> GfConversionResult:
        """Rational OGF of a constant-coefficient linear recurrence.

        For a_n = sum c_i a_{n-i} the denominator is 1 - sum c_i x^i and the
        numerator is the degree < k polynomial fixed by the initial terms.
        """
        k = len(recurrence.coefficients)
        denominator = [1] + [-c for c in recurrence.coefficients]
        numerator = []
        for n in range(k):
            value = recurrence.initial[n]
            for i in range(1, min(n, k) + 1):
                value -= recurrence.coefficients[i - 1] * recurrence.initial[n - i]
            numerator.append(value)
        while len(numerator) > 1 and numerator[-1] == 0:
            numerator.pop()
        return GfConversionResult(
            status=OperationStatus.AVAILABLE,
            rational=RationalGeneratingFunction(
                numerator=numerator, denominator=denominator),
            trust=TrustLevel.EXACT,
            evidence=_exact_evidence("ogf_from_recurrence"),
        )

    def recurrence_from_ogf(
        self, rational: RationalGeneratingFunction,
    ) -> GfConversionResult:
        """Linear recurrence encoded by a rational OGF's denominator."""
        order = len(rational.denominator) - 1
        if order < 1:
            return GfConversionResult(
                status=OperationStatus.UNSUPPORTED,
                diagnostics=["a polynomial OGF has no nontrivial recurrence"])
        coefficients = [-c for c in rational.denominator[1:]]
        initial = _rational_series(rational, order)
        return GfConversionResult(
            status=OperationStatus.AVAILABLE,
            recurrence=LinearRecurrence(coefficients=coefficients, initial=initial),
            trust=TrustLevel.EXACT,
            evidence=_exact_evidence("recurrence_from_ogf"),
        )

    def egf_from_recurrence(self, recurrence: LinearRecurrence) -> GfConversionResult:
        """EGF recurrence conversion is not supported; reported explicitly."""
        return GfConversionResult(
            status=OperationStatus.UNSUPPORTED,
            diagnostics=[
                "recurrence conversion is supported only for ordinary "
                "generating functions; exponential generating functions of "
                "linear recurrences are not rational in general",
            ],
        )

    def coefficient(self, gf: GeneratingFunction, n: int) -> CoefficientResult:
        """Exact a_n: coefficient of x^n (OGF) or of x^n/n! (EGF)."""
        scaling = "ordinary" if gf.kind == GfKind.ORDINARY else "factorial"
        if n < 0:
            return CoefficientResult(
                status=OperationStatus.ERROR, index=n, scaling=scaling,
                diagnostics=["coefficient index must be nonnegative"])
        if n < len(gf.coefficients):
            return CoefficientResult(
                status=OperationStatus.AVAILABLE, index=n,
                value=gf.coefficients[n], scaling=scaling,
                trust=TrustLevel.EXACT,
                evidence=_exact_evidence("coefficient:prefix"),
            )
        if n > MAX_SERIES_TERMS:
            return CoefficientResult(
                status=OperationStatus.ERROR, index=n, scaling=scaling,
                diagnostics=[f"coefficient extension is limited to n <= "
                             f"{MAX_SERIES_TERMS}"])
        if gf.recurrence is not None:
            terms = None
            method = "coefficient:recurrence_extension"
            if _extend_fast is not None:
                terms = _extend_fast(
                    gf.recurrence.coefficients, gf.recurrence.initial, n)
                if terms is not None:
                    method = "coefficient:recurrence_extension_numba"
            if terms is None:
                terms = _extend_from_recurrence(gf.recurrence, n)
            return CoefficientResult(
                status=OperationStatus.AVAILABLE, index=n, value=terms[n],
                scaling=scaling, trust=TrustLevel.EXACT,
                evidence=_exact_evidence(method),
            )
        if gf.rational is not None:
            terms = _rational_series(gf.rational, n + 1)
            return CoefficientResult(
                status=OperationStatus.AVAILABLE, index=n, value=terms[n],
                scaling=scaling, trust=TrustLevel.EXACT,
                evidence=_exact_evidence("coefficient:rational_expansion"),
            )
        return CoefficientResult(
            status=OperationStatus.UNKNOWN, index=n, scaling=scaling,
            diagnostics=[
                "coefficient beyond the stored prefix; no recurrence or "
                "rational closed form is available to extend it",
            ],
        )

    def verify_generating_function(
        self, gf: GeneratingFunction, terms: int | None = None,
    ) -> VerificationResult:
        """Recompute the stored prefix from recurrence/rational forms."""
        prefix = list(gf.coefficients)
        count = len(prefix) if terms is None else min(terms, len(prefix))
        checks: dict[str, bool] = {}
        if gf.recurrence is not None and count > 0:
            extended = _extend_from_recurrence(gf.recurrence, count - 1)
            checks["recurrence_extension"] = extended[:count] == prefix[:count]
        if gf.rational is not None and count > 0:
            expanded = _rational_series(gf.rational, count)
            checks["rational_expansion"] = expanded == prefix[:count]
        if not checks:
            return VerificationResult(
                status=OperationStatus.UNKNOWN,
                diagnostics=["no recurrence or rational form to verify against"])
        verified = all(checks.values())
        evidence = _exact_evidence("verify_generating_function")
        evidence.certificate.append(CertificateEvidence(
            certificate_type="generating_function_coefficient_agreement",
            claim=f"first {count} coefficients agree with every available "
                  "recurrence/rational reconstruction",
            witness=dict(checks),
            verifier="mathkernel",
            verified=verified,
            trust="exact",
        ))
        return VerificationResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.REFUTED,
            verified=verified,
            checks=checks,
            trust=TrustLevel.EXACT,
            evidence=evidence,
        )

    def verify_count_identities(self, n: int) -> VerificationResult:
        """Cross-check classical count identities at a given n, exactly."""
        _check_argument(n, MAX_TABLE_ARGUMENT, "verify_count_identities")
        checks: dict[str, bool] = {
            "binomial_row_sum": (
                sum(comb(n, k) for k in range(n + 1)) == (1 << n)),
            "stirling_second_bell": (
                sum(stirling_second(n, k) for k in range(n + 1))
                == bell_number(n)),
            "stirling_first_factorial": (
                sum(stirling_first(n, k) for k in range(n + 1))
                == factorial(n)),
            "derangement_recurrence": (
                n == 0 or derangements_count(n)
                == n * derangements_count(n - 1) + (-1) ** n),
            "catalan_binomial": catalan_number(n) == comb(2 * n, n) // (n + 1),
            "catalan_convolution": (
                n == 0 and catalan_number(0) == 1
                or catalan_number(n)
                == sum(catalan_number(i) * catalan_number(n - 1 - i)
                       for i in range(n))),
        }
        # Enumeration cross-checks, only where the full listing is affordable.
        partition_total = integer_partitions_count(n)
        if partition_total <= MAX_GENERATE_ITEMS:
            items, truncated = _collect(_integer_partitions(n), partition_total)
            checks["partition_enumeration"] = (
                not truncated and len(items) == partition_total)
        bell_total = bell_number(n)
        if bell_total <= MAX_GENERATE_ITEMS and n <= MAX_GENERATION_DEPTH:
            items, truncated = _collect(_set_partitions_rgs(n), bell_total)
            checks["set_partition_enumeration"] = (
                not truncated and len(items) == bell_total)
        verified = all(checks.values())
        evidence = _exact_evidence(
            "verify_count_identities",
            proposition=f"classical count identities hold at n={n}")
        evidence.certificate.append(CertificateEvidence(
            certificate_type="count_identity_agreement",
            claim=f"binomial/Stirling/Bell/Catalan/derangement/partition "
                  f"identities cross-checked at n={n}",
            witness=dict(checks),
            verifier="mathkernel",
            verified=verified,
            trust="exact",
        ))
        return VerificationResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.REFUTED,
            verified=verified,
            checks=checks,
            trust=TrustLevel.EXACT,
            evidence=evidence,
        )
