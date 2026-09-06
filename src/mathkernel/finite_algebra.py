# =============================================================================
# MathKernel - exact finite rings, finite fields, modules and normal forms
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Typed, evidence-carrying finite algebra over exact integer arithmetic.

Stage C core: Z/nZ rings, GF(p^m) fields with explicit defining polynomials
(irreducibility proved by Rabin's test before construction succeeds), typed
module presentations, and exact Smith/Hermite normal forms for integer
matrices with independently checkable certificates.  All arithmetic is exact
integer/polynomial arithmetic, so every successful result carries
``TrustLevel.EXACT`` evidence and a certificate whose witness is re-verified
by the public ``verify_*`` functions.  Out-of-scope requests (composite
modulus modules, oversized inputs, non-prime characteristics) return an
explicit ``OperationStatus.UNSUPPORTED``/``REFUTED`` result instead of
failing silently.
"""
from __future__ import annotations

from enum import Enum
from math import gcd, isqrt
from typing import Any, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, field_serializer, field_validator,
    model_validator,
)

from mathkernel_artifacts import (
    CertificateEvidence, ComputationEvidence, EvidenceBundle, ProofEvidence,
)
from .models import OperationStatus, TrustLevel

try:
    from .finite_algebra_fast import poly_mulmod_fast as _poly_mulmod_fast
except ImportError:  # pragma: no cover
    _poly_mulmod_fast = None

ResultStatus = OperationStatus

ENGINE = "mathkernel.finite_algebra"
EXACT = TrustLevel.EXACT.value

# Explicit limits: requests beyond them return UNSUPPORTED, never a silent
# approximation or an unbounded computation.
MAX_MODULUS_BITS = 4096
MAX_PRIME_BITS = 63          # deterministic Miller-Rabin range (< 2^64)
MAX_FIELD_DEGREE = 128
MAX_MATRIX_DIM = 64
MAX_MATRIX_ENTRY_BITS = 2048

LIMITS = {
    "max_modulus_bits": MAX_MODULUS_BITS,
    "max_prime_bits": MAX_PRIME_BITS,
    "max_field_degree": MAX_FIELD_DEGREE,
    "max_matrix_dim": MAX_MATRIX_DIM,
    "max_matrix_entry_bits": MAX_MATRIX_ENTRY_BITS,
}


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class FiniteAlgebraResult(BaseModel):
    """Typed outcome of a finite-algebra query, with evidence and limits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str
    status: OperationStatus
    value: Any = None
    trust: str = TrustLevel.UNKNOWN.value
    conditions: list[str] = Field(default_factory=list)
    verification: dict[str, bool | None] = Field(default_factory=dict)
    limits: dict[str, int] = Field(default_factory=lambda: dict(LIMITS))
    claim_evidence: dict[str, EvidenceBundle] = Field(default_factory=dict)

    @field_serializer("value", when_used="json")
    def serialize_value(self, value: Any) -> Any:
        return _json_value(value)


def _exact_result(
    query: str,
    value: Any,
    *,
    method: str,
    verification: dict[str, bool | None] | None = None,
    certificates: list[CertificateEvidence] | None = None,
    proofs: list[ProofEvidence] | None = None,
    conditions: list[str] | None = None,
) -> FiniteAlgebraResult:
    verification = verification or {}
    verified = bool(verification) and all(
        outcome is True for outcome in verification.values())
    evidence = EvidenceBundle(
        computation=[ComputationEvidence(
            engine=ENGINE,
            method=method,
            arithmetic="exact",
            deterministic=True,
            trust=EXACT,
        )],
        certificate=list(certificates or []),
        proof=list(proofs or []),
        justified_trust=EXACT,
    )
    return FiniteAlgebraResult(
        query=query,
        status=OperationStatus.VERIFIED if verified else OperationStatus.AVAILABLE,
        value=value,
        trust=EXACT,
        conditions=conditions or [],
        verification=verification,
        claim_evidence={query: evidence},
    )


def _absent(
    query: str, status: OperationStatus, reason: str,
) -> FiniteAlgebraResult:
    return FiniteAlgebraResult(query=query, status=status, conditions=[reason])


def _unsupported(query: str, reason: str) -> FiniteAlgebraResult:
    return _absent(query, OperationStatus.UNSUPPORTED, reason)


# ---------------------------------------------------------------------------
# Exact primality (deterministic Miller-Rabin, valid below 2^64)
# ---------------------------------------------------------------------------

_MR_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def _miller_rabin_passes(n: int, bases: tuple[int, ...]) -> bool:
    if n < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % small == 0:
            return n == small
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _is_prime(n: int) -> bool:
    """Exact primality for 0 <= n < 2^64 only."""
    if n >= 1 << 64:
        raise ValueError("exact deterministic primality is limited to n < 2^64")
    return _miller_rabin_passes(n, _MR_BASES_64)


def _classify_primality(n: int) -> tuple[str, bool | None, bool, str]:
    """Return (status, is_prime, exact, method) without overstating large n.

    A failed Miller-Rabin round is an exact compositeness witness. Passing the
    fixed bases above 2^64 is only probable-prime evidence and is never promoted
    to an exact primality claim.
    """
    if n < 2:
        return "composite", False, True, "definition"
    if n < (1 << 64):
        prime = _is_prime(n)
        return (
            "prime" if prime else "composite",
            prime, True, "deterministic_miller_rabin_64",
        )
    passes = _miller_rabin_passes(n, _MR_BASES_64)
    if not passes:
        return "composite", False, True, "miller_rabin_compositeness_witness"
    return "probable_prime", None, False, "miller_rabin_probable_prime"


def _prime_divisors(m: int) -> list[int]:
    """Distinct prime divisors of m by exact trial division."""
    out: list[int] = []
    rest = m
    d = 2
    while d * d <= rest:
        if rest % d == 0:
            out.append(d)
            while rest % d == 0:
                rest //= d
        d += 1 if d == 2 else 2
    if rest > 1:
        out.append(rest)
    return out


# ---------------------------------------------------------------------------
# Polynomial arithmetic over GF(p): coefficient lists, index = power
# ---------------------------------------------------------------------------

def _poly_trim(a: list[int]) -> list[int]:
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _poly_norm(a: list[int], p: int) -> list[int]:
    return _poly_trim([c % p for c in a])


def _poly_add(a: list[int], b: list[int], p: int) -> list[int]:
    n = max(len(a), len(b))
    return _poly_trim([
        ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p
        for i in range(n)
    ])


def _poly_sub(a: list[int], b: list[int], p: int) -> list[int]:
    n = max(len(a), len(b))
    return _poly_trim([
        ((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p
        for i in range(n)
    ])


def _poly_mul(a: list[int], b: list[int], p: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    out[i + j] = (out[i + j] + ai * bj) % p
    return _poly_trim(out)


def _poly_divmod(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]:
    if b == [0]:
        raise ZeroDivisionError("polynomial division by zero")
    rem = list(a)
    deg_b = len(b) - 1
    inv_lead = pow(b[-1], -1, p)
    quo = [0] * max(1, len(a) - deg_b)
    while len(rem) - 1 >= deg_b and rem != [0]:
        k = len(rem) - 1 - deg_b
        c = rem[-1] * inv_lead % p
        quo[k] = c
        for i in range(deg_b + 1):
            rem[k + i] = (rem[k + i] - c * b[i]) % p
        _poly_trim(rem)
    return _poly_trim(quo), rem


def _poly_mod(a: list[int], f: list[int], p: int) -> list[int]:
    return _poly_divmod(a, f, p)[1]


def _poly_mulmod(a: list[int], b: list[int], f: list[int], p: int) -> list[int]:
    if _poly_mulmod_fast is not None:
        fast = _poly_mulmod_fast(a, b, f, p)
        if fast is not None:
            return fast
    return _poly_mod(_poly_mul(a, b, p), f, p)


def _poly_powmod(base: list[int], e: int, f: list[int], p: int) -> list[int]:
    acc = [1]
    cur = _poly_mod(base, f, p)
    while e:
        if e & 1:
            acc = _poly_mulmod(acc, cur, f, p)
        cur = _poly_mulmod(cur, cur, f, p)
        e >>= 1
    return acc


def _poly_gcd(a: list[int], b: list[int], p: int) -> list[int]:
    a, b = _poly_trim(list(a)), _poly_trim(list(b))
    while b != [0]:
        a, b = b, _poly_mod(a, b, p)
    if a[-1] != 1:
        a = [c * pow(a[-1], -1, p) % p for c in a]
    return a


def _poly_xgcd(
    a: list[int], b: list[int], p: int,
) -> tuple[list[int], list[int], list[int]]:
    """Monic g = gcd(a, b) with s, t such that s*a + t*b = g over GF(p)."""
    s, s_new = [1], [0]
    t, t_new = [0], [1]
    r, r_new = _poly_trim(list(a)), _poly_trim(list(b))
    while r_new != [0]:
        quo, rem = _poly_divmod(r, r_new, p)
        r, r_new = r_new, rem
        s, s_new = s_new, _poly_sub(s, _poly_mul(quo, s_new, p), p)
        t, t_new = t_new, _poly_sub(t, _poly_mul(quo, t_new, p), p)
    if r[-1] != 1:
        inv = pow(r[-1], -1, p)
        r = [c * inv % p for c in r]
        s = [c * inv % p for c in s]
        t = [c * inv % p for c in t]
    return r, s, t


def _rabin_irreducible(f: list[int], p: int, m: int) -> tuple[bool, dict]:
    """Rabin's irreducibility test for a monic degree-m polynomial over GF(p).

    Returns (is_irreducible, witness) where the witness records each checked
    gcd and the final Frobenius residue so the verdict is auditable.
    """
    x = [0, 1]
    witness: dict[str, Any] = {"prime_divisors_of_degree": _prime_divisors(m)}
    for q in witness["prime_divisors_of_degree"]:
        h = x
        for _ in range(m // q):
            h = _poly_powmod(h, p, f, p)
        g = _poly_gcd(f, _poly_sub(h, x, p), p)
        witness[f"gcd_x_pow_p^{m // q}_minus_x"] = g
        if g != [1]:
            witness["failed_check"] = f"gcd(f, x^(p^{m // q}) - x) = {g}"
            return False, witness
    x_mod_f = _poly_mod([0, 1], f, p)
    for _ in range(m):
        x = _poly_powmod(x, p, f, p)
    witness["x_pow_p^m_mod_f"] = x
    if x != x_mod_f:
        witness["failed_check"] = "x^(p^m) - x != 0 (mod f)"
        return False, witness
    return True, witness


# ---------------------------------------------------------------------------
# Typed finite rings Z/nZ
# ---------------------------------------------------------------------------

class FiniteRingSpec(BaseModel):
    """The residue ring Z/nZ for an explicit modulus n >= 2.

    Ring construction is exact for every accepted modulus. Primality is a
    separate claim: above the deterministic <2^64 range, probable-prime status
    is recorded without pretending it is an exact proof.
    """

    kind: Literal["zmod"] = "zmod"
    modulus: int = Field(ge=2)
    modulus_is_prime: bool | None = None
    primality_status: Literal["prime", "composite", "probable_prime", "unknown"] = "unknown"
    primality_exact: bool = False
    primality_method: str = "unknown"
    order: int

    @model_validator(mode="after")
    def _consistent(self) -> "FiniteRingSpec":
        if self.order != self.modulus:
            raise ValueError("order of Z/nZ must equal the modulus")
        status, is_prime, exact, method = _classify_primality(self.modulus)
        object.__setattr__(self, "primality_status", status)
        object.__setattr__(self, "modulus_is_prime", is_prime)
        object.__setattr__(self, "primality_exact", exact)
        object.__setattr__(self, "primality_method", method)
        return self

    @property
    def is_field(self) -> bool | None:
        return self.modulus_is_prime


FiniteRing = FiniteRingSpec


class RingElement(BaseModel):
    """An element of Z/nZ, always stored reduced to 0 <= value < n."""

    ring: FiniteRingSpec
    value: int

    @model_validator(mode="after")
    def _reduced(self) -> "RingElement":
        object.__setattr__(self, "value", self.value % self.ring.modulus)
        return self


def create_finite_ring(modulus: int) -> FiniteAlgebraResult:
    """Construct the typed ring Z/nZ (a field exactly when n is prime)."""
    query = f"create_finite_ring({modulus})"
    n = int(modulus)
    if n < 2:
        return _absent(query, OperationStatus.ERROR, "modulus must be >= 2")
    if n.bit_length() > MAX_MODULUS_BITS:
        return _unsupported(
            query, f"modulus exceeds {MAX_MODULUS_BITS}-bit limit")
    ring = FiniteRingSpec(modulus=n, order=n)
    result = _exact_result(
        query, ring, method="zmod_construction",
        verification={"modulus_ge_2": True, "ring_construction_exact": True},
        conditions=[
            "Z/nZ is a field exactly when n is prime",
            f"modulus primality status: {ring.primality_status} via {ring.primality_method}",
        ],
    )
    # Primality is a separate claim from exact ring construction. A large
    # probable prime is deliberately not represented as exact evidence.
    if ring.primality_exact:
        primality_trust = "exact"
    elif ring.primality_status == "probable_prime":
        primality_trust = "heuristic"
    else:
        primality_trust = "unknown"
    result.claim_evidence["modulus_primality"] = EvidenceBundle(computation=[
        ComputationEvidence(
            engine=ENGINE, method=ring.primality_method, arithmetic="integer",
            deterministic=True, trust=primality_trust,
            metadata={"status": ring.primality_status},
        )
    ], justified_trust=primality_trust)
    return result


def _require_ring(ring: FiniteRingSpec, query: str) -> FiniteAlgebraResult | None:
    if not isinstance(ring, FiniteRingSpec):
        return _absent(query, OperationStatus.ERROR,
                       "ring must be a FiniteRingSpec (Z/nZ)")
    return None


def ring_add(ring: FiniteRingSpec, a: int, b: int) -> FiniteAlgebraResult:
    query = f"ring_add({a}, {b}) in Z/{ring.modulus}Z"
    if (err := _require_ring(ring, query)) is not None:
        return err
    element = RingElement(ring=ring, value=int(a) + int(b))
    return _exact_result(query, element, method="zmod_add",
                         verification={"reduced": True})


def ring_mul(ring: FiniteRingSpec, a: int, b: int) -> FiniteAlgebraResult:
    query = f"ring_mul({a}, {b}) in Z/{ring.modulus}Z"
    if (err := _require_ring(ring, query)) is not None:
        return err
    element = RingElement(ring=ring, value=int(a) * int(b))
    return _exact_result(query, element, method="zmod_mul",
                         verification={"reduced": True})


def ring_neg(ring: FiniteRingSpec, a: int) -> FiniteAlgebraResult:
    query = f"ring_neg({a}) in Z/{ring.modulus}Z"
    if (err := _require_ring(ring, query)) is not None:
        return err
    element = RingElement(ring=ring, value=-int(a))
    return _exact_result(query, element, method="zmod_neg",
                         verification={"reduced": True})


def ring_pow(ring: FiniteRingSpec, a: int, exponent: int) -> FiniteAlgebraResult:
    query = f"ring_pow({a}, {exponent}) in Z/{ring.modulus}Z"
    if (err := _require_ring(ring, query)) is not None:
        return err
    e = int(exponent)
    if e < 0:
        inv = ring_inverse(ring, a)
        if inv.status != OperationStatus.VERIFIED or inv.value is None:
            return _absent(query, OperationStatus.DOES_NOT_EXIST,
                           "negative exponent requires a unit base")
        return ring_pow(ring, inv.value.value, -e)
    element = RingElement(ring=ring, value=pow(int(a), e, ring.modulus))
    return _exact_result(query, element, method="zmod_pow",
                         verification={"reduced": True})


def ring_inverse(ring: FiniteRingSpec, a: int) -> FiniteAlgebraResult:
    """Multiplicative inverse in Z/nZ; DOES_NOT_EXIST for non-units."""
    query = f"ring_inverse({a}) in Z/{ring.modulus}Z"
    if (err := _require_ring(ring, query)) is not None:
        return err
    value = int(a) % ring.modulus
    g = gcd(value, ring.modulus)
    if g != 1:
        return _absent(
            query, OperationStatus.DOES_NOT_EXIST,
            f"gcd({value}, {ring.modulus}) = {g} > 1: not a unit")
    element = RingElement(ring=ring, value=pow(value, -1, ring.modulus))
    verified = (element.value * value) % ring.modulus == 1
    return _exact_result(query, element, method="zmod_inverse_extended_gcd",
                         verification={"a_times_inverse_is_one": verified})


# ---------------------------------------------------------------------------
# Typed finite fields GF(p^m)
# ---------------------------------------------------------------------------

class FiniteFieldSpec(BaseModel):
    """GF(p^m) presented as GF(p)[x]/(f) with an explicit monic irreducible f.

    ``modulus_coeffs[i]`` is the coefficient of x^i; the polynomial must be
    monic of degree ``degree`` and is accepted only after Rabin's exact
    irreducibility test passes.  ``basis`` names the power basis
    1, x, ..., x^{m-1} used by element coefficient lists.
    """

    kind: Literal["gf"] = "gf"
    prime: int = Field(ge=2)
    degree: int = Field(ge=1)
    modulus_coeffs: list[int]
    basis: list[str]
    order: int
    irreducibility_verified: bool = False
    irreducibility_witness: dict[str, Any] = Field(default_factory=dict)

    @field_validator("modulus_coeffs")
    @classmethod
    def _coeffs_reduced(cls, coeffs: list[int]) -> list[int]:
        return [int(c) for c in coeffs]

    @model_validator(mode="after")
    def _consistent(self) -> "FiniteFieldSpec":
        if len(self.modulus_coeffs) != self.degree + 1:
            raise ValueError("defining polynomial must have degree + 1 coefficients")
        if self.order != self.prime ** self.degree:
            raise ValueError("order must be p^m")
        expected_basis = ["1"] + [
            "x" if k == 1 else f"x^{k}" for k in range(1, self.degree)
        ]
        if self.basis != expected_basis:
            raise ValueError("basis must be the power basis 1, x, ..., x^{m-1}")
        return self


FiniteField = FiniteFieldSpec


class FieldElement(BaseModel):
    """An element of GF(p^m) in the power basis, coefficients reduced mod p."""

    field: FiniteFieldSpec
    coeffs: list[int]

    @model_validator(mode="after")
    def _reduced(self) -> "FieldElement":
        if len(self.coeffs) != self.field.degree:
            raise ValueError("element must have exactly `degree` coefficients")
        object.__setattr__(
            self, "coeffs", [int(c) % self.field.prime for c in self.coeffs])
        return self


def create_finite_field(prime: int, modulus_coeffs: list[int]) -> FiniteAlgebraResult:
    """Construct GF(p^m) = GF(p)[x]/(f), proving f irreducible first.

    Returns REFUTED (with the failing Rabin witness) when p is not prime or
    f is reducible, and UNSUPPORTED beyond the explicit size limits.
    """
    query = f"create_finite_field(p={prime}, f={list(modulus_coeffs)})"
    p = int(prime)
    coeffs = [int(c) for c in modulus_coeffs]
    if p.bit_length() > MAX_PRIME_BITS:
        return _unsupported(
            query, f"characteristic exceeds {MAX_PRIME_BITS}-bit exact limit")
    if not _is_prime(p):
        return _absent(query, OperationStatus.REFUTED,
                       f"{p} is not prime (exact deterministic test)")
    m = len(coeffs) - 1
    if m < 1:
        return _absent(query, OperationStatus.ERROR,
                       "defining polynomial must have degree >= 1")
    if m > MAX_FIELD_DEGREE:
        return _unsupported(
            query, f"degree {m} exceeds limit {MAX_FIELD_DEGREE}")
    coeffs = [c % p for c in coeffs]
    if coeffs[-1] != 1:
        return _absent(query, OperationStatus.ERROR,
                       "defining polynomial must be monic")
    irreducible, witness = _rabin_irreducible(coeffs, p, m)
    if not irreducible:
        result = _absent(query, OperationStatus.REFUTED,
                         f"defining polynomial is reducible over GF({p}): "
                         + witness.get("failed_check", "Rabin test failed"))
        result.verification["irreducible"] = False
        return result
    field = FiniteFieldSpec(
        prime=p, degree=m, modulus_coeffs=coeffs,
        basis=["1"] + ["x" if k == 1 else f"x^{k}" for k in range(1, m)],
        order=p ** m,
        irreducibility_verified=True,
        irreducibility_witness=witness,
    )
    proof = ProofEvidence(
        proposition=f"f is irreducible over GF({p}), so GF(p)[x]/(f) is a field",
        method="rabin_irreducibility",
        engine=ENGINE,
        certificate=witness,
        verified=True,
        trust=EXACT,
    )
    return _exact_result(
        query, field, method="gf_construction_rabin",
        verification={"prime_characteristic": True, "monic": True,
                      "irreducible": True},
        proofs=[proof],
    )


def _require_field(
    field: FiniteFieldSpec, query: str,
) -> FiniteAlgebraResult | None:
    if not isinstance(field, FiniteFieldSpec):
        return _absent(query, OperationStatus.ERROR,
                       "field must be a FiniteFieldSpec")
    if not field.irreducibility_verified:
        return _absent(query, OperationStatus.ERROR,
                       "field construction was not irreducibility-verified")
    return None


def _field_coeffs(
    field: FiniteFieldSpec, coeffs: list[int], query: str,
) -> list[int] | FiniteAlgebraResult:
    if len(coeffs) != field.degree:
        return _absent(query, OperationStatus.ERROR,
                       f"element needs exactly {field.degree} coefficients")
    return [int(c) % field.prime for c in coeffs]


def field_add(
    field: FiniteFieldSpec, a: list[int], b: list[int],
) -> FiniteAlgebraResult:
    query = "field_add"
    if (err := _require_field(field, query)) is not None:
        return err
    ca, cb = _field_coeffs(field, a, query), _field_coeffs(field, b, query)
    if isinstance(ca, FiniteAlgebraResult):
        return ca
    if isinstance(cb, FiniteAlgebraResult):
        return cb
    total = _poly_add(ca, cb, field.prime)
    total += [0] * (field.degree - len(total))
    element = FieldElement(field=field, coeffs=total)
    return _exact_result(query, element, method="gf_add",
                         verification={"reduced": True})


def field_mul(
    field: FiniteFieldSpec, a: list[int], b: list[int],
) -> FiniteAlgebraResult:
    query = "field_mul"
    if (err := _require_field(field, query)) is not None:
        return err
    ca, cb = _field_coeffs(field, a, query), _field_coeffs(field, b, query)
    if isinstance(ca, FiniteAlgebraResult):
        return ca
    if isinstance(cb, FiniteAlgebraResult):
        return cb
    product = None
    method = "gf_mul_mod_defining_poly"
    if _poly_mulmod_fast is not None:
        product = _poly_mulmod_fast(
            ca, cb, field.modulus_coeffs, field.prime)
        if product is not None:
            method = "gf_mul_mod_defining_poly_numba"
    if product is None:
        product = _poly_mulmod(ca, cb, field.modulus_coeffs, field.prime)
    product += [0] * (field.degree - len(product))
    element = FieldElement(field=field, coeffs=product)
    return _exact_result(query, element, method=method,
                         verification={"reduced": True})


def field_neg(field: FiniteFieldSpec, a: list[int]) -> FiniteAlgebraResult:
    query = "field_neg"
    if (err := _require_field(field, query)) is not None:
        return err
    ca = _field_coeffs(field, a, query)
    if isinstance(ca, FiniteAlgebraResult):
        return ca
    element = FieldElement(field=field, coeffs=[(-c) % field.prime for c in ca])
    return _exact_result(query, element, method="gf_neg",
                         verification={"reduced": True})


def field_pow(
    field: FiniteFieldSpec, a: list[int], exponent: int,
) -> FiniteAlgebraResult:
    query = f"field_pow(exponent={exponent})"
    if (err := _require_field(field, query)) is not None:
        return err
    ca = _field_coeffs(field, a, query)
    if isinstance(ca, FiniteAlgebraResult):
        return ca
    e = int(exponent)
    if e < 0:
        inv = field_inverse(field, a)
        if inv.status != OperationStatus.VERIFIED or inv.value is None:
            return _absent(query, OperationStatus.DOES_NOT_EXIST,
                           "negative exponent of zero is undefined")
        return field_pow(field, inv.value.coeffs, -e)
    value = _poly_powmod(ca, e, field.modulus_coeffs, field.prime)
    value += [0] * (field.degree - len(value))
    element = FieldElement(field=field, coeffs=value)
    return _exact_result(query, element, method="gf_pow_square_and_multiply",
                         verification={"reduced": True})


def field_inverse(field: FiniteFieldSpec, a: list[int]) -> FiniteAlgebraResult:
    """Inverse via extended Euclid; DOES_NOT_EXIST for the zero element."""
    query = "field_inverse"
    if (err := _require_field(field, query)) is not None:
        return err
    ca = _field_coeffs(field, a, query)
    if isinstance(ca, FiniteAlgebraResult):
        return ca
    if all(c == 0 for c in ca):
        return _absent(query, OperationStatus.DOES_NOT_EXIST,
                       "zero has no multiplicative inverse")
    g, s, _ = _poly_xgcd(ca, field.modulus_coeffs, field.prime)
    if g != [1]:
        # Unreachable for a verified irreducible modulus; guard anyway.
        return _absent(query, OperationStatus.ERROR,
                       "gcd(element, defining polynomial) != 1: "
                       "defining polynomial is not irreducible")
    inv = _poly_mod(s, field.modulus_coeffs, field.prime)
    inv += [0] * (field.degree - len(inv))
    element = FieldElement(field=field, coeffs=inv)
    check = _poly_mulmod(ca, element.coeffs, field.modulus_coeffs, field.prime)
    verified = check == [1]
    return _exact_result(query, element, method="gf_inverse_extended_gcd",
                         verification={"a_times_inverse_is_one": verified})


# ---------------------------------------------------------------------------
# Exact integer matrix helpers
# ---------------------------------------------------------------------------

def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _matmul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    rows, inner, cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(inner))
             for j in range(cols)] for i in range(rows)]


def _det_bareiss(M: list[list[int]]) -> int:
    """Exact determinant by fraction-free Bareiss elimination."""
    n = len(M)
    if n == 0:
        return 1
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def _check_matrix(matrix: list[list[int]], query: str) -> FiniteAlgebraResult | None:
    if not matrix or not matrix[0]:
        return _absent(query, OperationStatus.ERROR, "matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        return _absent(query, OperationStatus.ERROR, "matrix rows must align")
    if len(matrix) > MAX_MATRIX_DIM or width > MAX_MATRIX_DIM:
        return _unsupported(
            query, f"matrix exceeds {MAX_MATRIX_DIM}x{MAX_MATRIX_DIM} limit")
    if any(int(v).bit_length() > MAX_MATRIX_ENTRY_BITS
           for row in matrix for v in row):
        return _unsupported(
            query, f"entries exceed {MAX_MATRIX_ENTRY_BITS}-bit limit")
    return None


# ---------------------------------------------------------------------------
# Smith normal form with unimodular transforms
# ---------------------------------------------------------------------------

def _smith_normal_form(
    A: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Exact SNF: returns (D, U, V) with U*A*V = D, U and V unimodular."""
    m, n = len(A), len(A[0])
    D = [row[:] for row in A]
    U = _identity(m)
    V = _identity(n)

    def swap_rows(i: int, j: int) -> None:
        D[i], D[j] = D[j], D[i]
        U[i], U[j] = U[j], U[i]

    def swap_cols(i: int, j: int) -> None:
        for row in D:
            row[i], row[j] = row[j], row[i]
        for row in V:
            row[i], row[j] = row[j], row[i]

    def row_add(i: int, j: int, c: int) -> None:
        for k in range(n):
            D[i][k] += c * D[j][k]
        for k in range(m):
            U[i][k] += c * U[j][k]

    def col_add(i: int, j: int, c: int) -> None:
        for row in D:
            row[i] += c * row[j]
        for row in V:
            row[i] += c * row[j]

    def row_negate(i: int) -> None:
        D[i] = [-v for v in D[i]]
        U[i] = [-v for v in U[i]]

    t = 0
    while t < m and t < n:
        pivot = next(
            ((i, j) for i in range(t, m) for j in range(t, n) if D[i][j]),
            None,
        )
        if pivot is None:
            break
        swap_rows(t, pivot[0])
        swap_cols(t, pivot[1])
        while True:
            for i in range(t + 1, m):
                while D[i][t] != 0:
                    q = D[i][t] // D[t][t]
                    row_add(i, t, -q)
                    if D[i][t] != 0:
                        swap_rows(i, t)
            for j in range(t + 1, n):
                while D[t][j] != 0:
                    q = D[t][j] // D[t][t]
                    col_add(j, t, -q)
                    if D[t][j] != 0:
                        swap_cols(j, t)
            if any(D[i][t] for i in range(t + 1, m)) or \
               any(D[t][j] for j in range(t + 1, n)):
                continue
            bad = next(
                ((i, j) for i in range(t + 1, m) for j in range(t + 1, n)
                 if D[i][j] % D[t][t] != 0),
                None,
            )
            if bad is None:
                break
            # Inject a non-divisible entry into the pivot row; the next
            # reduction pass strictly shrinks |D[t][t]|, so this terminates.
            row_add(t, bad[0], 1)
        if D[t][t] < 0:
            row_negate(t)
        t += 1
    return D, U, V


def verify_snf(
    A: list[list[int]], D: list[list[int]],
    U: list[list[int]], V: list[list[int]],
) -> dict[str, bool]:
    """Independently verify a Smith normal form certificate."""
    m, n = len(A), len(A[0])
    product = _matmul(_matmul(U, A), V)
    diagonal_ok = all(
        D[i][j] == 0 for i in range(m) for j in range(n) if i != j)
    diag = [D[i][i] for i in range(min(m, n))]
    nonnegative = all(d >= 0 for d in diag)
    nonzero = [d for d in diag if d != 0]
    divisibility = all(
        nonzero[i] % nonzero[i - 1] == 0 for i in range(1, len(nonzero)))
    zeros_last = all(d == 0 for d in diag[len(nonzero):])
    return {
        "U_A_V_equals_D": product == D,
        "diagonal": diagonal_ok,
        "diagonal_nonnegative": nonnegative,
        "diagonal_divisibility_chain": divisibility,
        "zero_diagonal_entries_trail": zeros_last,
        "U_unimodular": abs(_det_bareiss(U)) == 1,
        "V_unimodular": abs(_det_bareiss(V)) == 1,
    }


def smith_normal_form(matrix: list[list[int]]) -> FiniteAlgebraResult:
    """Exact Smith normal form of an integer matrix, with certificate.

    The value carries D, U, V with U*A*V = D; the certificate witness is
    re-checked by ``verify_snf`` and every check is recorded in
    ``verification``.
    """
    query = f"smith_normal_form({len(matrix)}x{len(matrix[0]) if matrix else 0})"
    if (err := _check_matrix(matrix, query)) is not None:
        return err
    A = [[int(v) for v in row] for row in matrix]
    D, U, V = _smith_normal_form(A)
    verification = verify_snf(A, D, U, V)
    certificate = CertificateEvidence(
        certificate_type="smith_normal_form",
        claim="U*A*V = D with D diagonal, d_i >= 0, d_i | d_{i+1}, "
              "U and V unimodular",
        witness={"U": U, "D": D, "V": V},
        verifier=f"{ENGINE}.verify_snf",
        verified=all(verification.values()),
        trust=EXACT,
    )
    return _exact_result(
        query, {"D": D, "U": U, "V": V},
        method="smith_normal_form_elementary_operations",
        verification=verification, certificates=[certificate],
    )


# ---------------------------------------------------------------------------
# Hermite normal form (row style) with unimodular transform
# ---------------------------------------------------------------------------

def _hermite_normal_form(
    A: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Exact row HNF: returns (H, U) with U*A = H, U unimodular.

    H has zero rows at the bottom; each nonzero row i has its first nonzero
    entry (pivot) in column j_i with j_0 < j_1 < ..., pivot positive, and
    entries above each pivot reduced to 0 <= h[k][j_i] < pivot.
    """
    m, n = len(A), len(A[0])
    H = [row[:] for row in A]
    U = _identity(m)

    def swap_rows(i: int, j: int) -> None:
        H[i], H[j] = H[j], H[i]
        U[i], U[j] = U[j], U[i]

    def row_add(i: int, j: int, c: int) -> None:
        for k in range(n):
            H[i][k] += c * H[j][k]
        for k in range(m):
            U[i][k] += c * U[j][k]

    def row_negate(i: int) -> None:
        H[i] = [-v for v in H[i]]
        U[i] = [-v for v in U[i]]

    r = 0
    for j in range(n):
        if r >= m:
            break
        if all(H[i][j] == 0 for i in range(r, m)):
            continue
        while True:
            smallest = min(
                (i for i in range(r, m) if H[i][j] != 0),
                key=lambda i: abs(H[i][j]),
            )
            swap_rows(r, smallest)
            for i in range(r + 1, m):
                if H[i][j] != 0:
                    row_add(i, r, -(H[i][j] // H[r][j]))
            if all(H[i][j] == 0 for i in range(r + 1, m)):
                break
        if H[r][j] < 0:
            row_negate(r)
        for i in range(r):
            q = H[i][j] // H[r][j]
            if q:
                row_add(i, r, -q)
        r += 1
    return H, U


def verify_hnf(
    A: list[list[int]], H: list[list[int]], U: list[list[int]],
) -> dict[str, bool]:
    """Independently verify a row Hermite normal form certificate."""
    m, n = len(A), len(A[0])
    product_ok = _matmul(U, A) == H
    pivots: list[int] = []
    pivot_columns_ok = True
    pivots_positive = True
    reduced_ok = True
    nonzero_rows = [
        i for i in range(m) if any(H[i][j] != 0 for j in range(n))]
    zero_rows_bottom = all(
        i < z for i in nonzero_rows for z in range(m)
        if not any(H[z][j] != 0 for j in range(n)))
    for i in nonzero_rows:
        row = H[i]
        first = next(j for j in range(n) if row[j] != 0)
        if pivots and first <= pivots[-1]:
            pivot_columns_ok = False
        if row[first] <= 0:
            pivots_positive = False
        for k in range(i):
            if not (0 <= H[k][first] < row[first]):
                reduced_ok = False
        pivots.append(first)
    return {
        "U_A_equals_H": product_ok,
        "zero_rows_at_bottom": zero_rows_bottom,
        "pivot_columns_strictly_increasing": pivot_columns_ok,
        "pivots_positive": pivots_positive,
        "entries_above_pivots_reduced": reduced_ok,
        "U_unimodular": abs(_det_bareiss(U)) == 1,
    }


def hermite_normal_form(matrix: list[list[int]]) -> FiniteAlgebraResult:
    """Exact row Hermite normal form of an integer matrix, with certificate."""
    query = f"hermite_normal_form({len(matrix)}x{len(matrix[0]) if matrix else 0})"
    if (err := _check_matrix(matrix, query)) is not None:
        return err
    A = [[int(v) for v in row] for row in matrix]
    H, U = _hermite_normal_form(A)
    verification = verify_hnf(A, H, U)
    certificate = CertificateEvidence(
        certificate_type="hermite_normal_form_row",
        claim="U*A = H with H in row Hermite normal form (zero rows at "
              "bottom, strictly increasing positive pivots, entries above "
              "pivots reduced) and U unimodular",
        witness={"U": U, "H": H},
        verifier=f"{ENGINE}.verify_hnf",
        verified=all(verification.values()),
        trust=EXACT,
    )
    return _exact_result(
        query, {"H": H, "U": U},
        method="hermite_normal_form_elementary_row_operations",
        verification=verification, certificates=[certificate],
    )


# ---------------------------------------------------------------------------
# Typed module presentations and finitely generated abelian groups
# ---------------------------------------------------------------------------

class ModulePresentation(BaseModel):
    """A finitely presented module: generators modulo integer relations.

    Over ``base_ring = "Z"`` the relation matrix presents a finitely
    generated abelian group Z^n / im(A^T).  Modules over Z/nZ are
    representable but normal-form decomposition over them is explicitly
    unsupported here.
    """

    base_ring: Literal["Z", "zmod"] = "Z"
    modulus: int | None = Field(default=None, ge=2)
    generators: list[str]
    relations: list[list[int]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> "ModulePresentation":
        if self.base_ring == "zmod" and self.modulus is None:
            raise ValueError("zmod modules require an explicit modulus")
        if self.base_ring == "Z" and self.modulus is not None:
            raise ValueError("Z modules do not take a modulus")
        if not self.generators:
            raise ValueError("a presentation needs at least one generator")
        width = len(self.generators)
        if any(len(row) != width for row in self.relations):
            raise ValueError("each relation must have one coefficient per generator")
        return self


Module = ModulePresentation


class AbelianGroupDecomposition(BaseModel):
    """G isomorphic to Z^free_rank x Z/d_1 x ... x Z/d_r with d_i | d_{i+1}."""

    free_rank: int = Field(ge=0)
    torsion_invariants: list[int] = Field(default_factory=list)
    invariant_factors: list[int] = Field(default_factory=list)
    num_generators: int = Field(ge=0)
    num_relations: int = Field(ge=0)
    structure: str

    @model_validator(mode="after")
    def _consistent(self) -> "AbelianGroupDecomposition":
        if any(d < 2 for d in self.torsion_invariants):
            raise ValueError("torsion invariants must be >= 2")
        if any(d % c for c, d in zip(
                self.torsion_invariants, self.torsion_invariants[1:])):
            raise ValueError("torsion invariants must form a divisibility chain")
        return self


def decompose_abelian_group(
    presentation: ModulePresentation,
) -> FiniteAlgebraResult:
    """Decompose a finitely generated abelian group via Smith normal form.

    For a presentation with n generators and relation matrix A, the group is
    Z^n / im(A^T); the SNF diagonal gives the invariant factors, ones are
    dropped and zero diagonal entries contribute free rank.
    """
    query = (f"decompose_abelian_group({len(presentation.generators)} generators, "
             f"{len(presentation.relations)} relations)")
    if not isinstance(presentation, ModulePresentation):
        return _absent(query, OperationStatus.ERROR,
                       "expected a ModulePresentation")
    if presentation.base_ring != "Z":
        return _unsupported(
            query, "normal-form decomposition is implemented over Z only; "
                   "Z/nZ module decomposition is out of scope")
    n = len(presentation.generators)
    relations = presentation.relations or [[0] * n]
    if (err := _check_matrix(relations, query)) is not None:
        return err
    snf = smith_normal_form(relations)
    if snf.status not in (OperationStatus.VERIFIED, OperationStatus.AVAILABLE):
        return _absent(query, snf.status,
                       "underlying Smith normal form failed: "
                       + "; ".join(snf.conditions))
    D = snf.value["D"]
    diag = [D[i][i] for i in range(min(len(D), n))]
    nonzero = [d for d in diag if d != 0]
    free_rank = n - len(nonzero)
    torsion = [d for d in nonzero if d > 1]
    parts = []
    if free_rank:
        parts.append(f"Z^{free_rank}" if free_rank > 1 else "Z")
    parts.extend(f"Z/{d}Z" for d in torsion)
    decomposition = AbelianGroupDecomposition(
        free_rank=free_rank,
        torsion_invariants=torsion,
        invariant_factors=nonzero,
        num_generators=n,
        num_relations=len(presentation.relations),
        structure=" x ".join(parts) if parts else "0",
    )
    verification = dict(snf.verification)
    verification["rank_plus_torsion_covers_generators"] = (
        free_rank + len(nonzero) == n)
    certificate = CertificateEvidence(
        certificate_type="abelian_group_decomposition",
        claim="G ~= Z^free_rank x prod Z/d_i with d_i | d_{i+1}, "
              "derived from the Smith normal form of the relation matrix",
        witness={"snf": snf.value, "decomposition": decomposition.model_dump()},
        verifier=f"{ENGINE}.verify_snf",
        verified=all(verification.values()),
        trust=EXACT,
    )
    return _exact_result(
        query, decomposition,
        method="abelian_group_decomposition_via_snf",
        verification=verification,
        certificates=[certificate],
        conditions=["presentation: Z^n / im(relations) with one relation per row"],
    )
