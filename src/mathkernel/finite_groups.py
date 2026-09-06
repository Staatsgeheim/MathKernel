# =============================================================================
# MathKernel - Stage C: exact finite group core
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Typed, evidence-carrying finite group theory over explicit representations.

Three exact representations are supported:

* ``FiniteGroup`` — a finite Cayley table over element indices 0..n-1. The
  constructor re-verifies closure, the Latin (cancellation) property, the
  declared identity, inverses, and full associativity; invalid tables are
  rejected, so every live instance is a verified group.
* ``PermutationGroup`` — a generator set inside Sym(0..degree-1), delegated to
  ``sympy.combinatorics``. Schreier-Sims stabilizer chains are returned as
  checkable certificates: the group order is independently recomputed as the
  product of the basic transversal sizes and only then trusted as exact.
* ``FiniteAbelianGroup`` — invariant-factor form, derived from a Cayley table
  by exact p-primary element counting and re-verified against the table.

All enumeration is deterministic (index order), all arithmetic is exact
integer arithmetic, and every claim carries an EvidenceBundle whose trust is
"exact" only when the associated check actually passed. Explicit limits bound
Cayley order, subgroup enumeration, permutation degree, and action size.
"""
from __future__ import annotations

from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from mathkernel_artifacts.evidence import (
    CertificateEvidence,
    ComputationEvidence,
    EvidenceBundle,
    ProofEvidence,
)

from .models import OperationStatus, TrustLevel

try:
    from .finite_groups_fast import validate_cayley_fast as _validate_cayley_fast
except ImportError:  # pragma: no cover
    _validate_cayley_fast = None

__all__ = [
    "MAX_CAYLEY_ORDER",
    "MAX_SUBGROUP_ENUM_ORDER",
    "MAX_SUBGROUPS",
    "MAX_PERM_DEGREE",
    "MAX_ACTION_POINTS",
    "LIMITS",
    "GroupResult",
    "OrderResult",
    "SubgroupInfo",
    "SubgroupResult",
    "SubgroupsResult",
    "CosetsResult",
    "NormalityResult",
    "QuotientResult",
    "ConjugacyClassesResult",
    "OrbitsResult",
    "MembershipResult",
    "StabilizerChainResult",
    "AbelianAnalysisResult",
    "FiniteGroup",
    "PermutationGroup",
    "FiniteAbelianGroup",
    "GroupHomomorphism",
    "GroupAction",
]

# Explicit limits -------------------------------------------------------------
MAX_CAYLEY_ORDER = 256        # full associativity re-check is O(n^3)
MAX_SUBGROUP_ENUM_ORDER = 64  # complete subgroup lattice enumeration
MAX_SUBGROUPS = 1024          # hard cap on enumerated subgroups
MAX_PERM_DEGREE = 512         # permutation group degree
MAX_ACTION_POINTS = 512       # size of a set a group acts on

LIMITS = {
    "max_cayley_order": MAX_CAYLEY_ORDER,
    "max_subgroup_enum_order": MAX_SUBGROUP_ENUM_ORDER,
    "max_subgroups": MAX_SUBGROUPS,
    "max_perm_degree": MAX_PERM_DEGREE,
    "max_action_points": MAX_ACTION_POINTS,
}


# Evidence helpers ------------------------------------------------------------
def _computation(method: str, *, engine: str = "mathkernel",
                 trust: str = "exact",
                 metadata: dict[str, Any] | None = None) -> ComputationEvidence:
    return ComputationEvidence(
        engine=engine,
        method=method,
        arithmetic="exact_integer",
        deterministic=True,
        trust=trust,
        metadata=dict(metadata or {}),
    )


def _proof(proposition: str, method: str, *, engine: str = "mathkernel",
           verified: bool = True) -> ProofEvidence:
    return ProofEvidence(
        proposition=proposition,
        method=method,
        engine=engine,
        verified=verified,
        trust="exact" if verified else "unknown",
    )


def _certificate(certificate_type: str, claim: str, witness: Any,
                 verifier: str, *, verified: bool = True) -> CertificateEvidence:
    return CertificateEvidence(
        certificate_type=certificate_type,
        claim=claim,
        witness=witness,
        verifier=verifier,
        verified=verified,
        trust="exact" if verified else "unknown",
    )


def _bundle(method: str, *, engine: str = "mathkernel",
            proposition: str | None = None,
            certificates: Iterable[CertificateEvidence] = (),
            metadata: dict[str, Any] | None = None,
            trust: str = "exact") -> EvidenceBundle:
    bundle = EvidenceBundle(computation=[
        _computation(method, engine=engine, trust=trust, metadata=metadata)])
    if proposition is not None:
        bundle.proof.append(_proof(proposition, method, engine=engine,
                                   verified=trust == "exact"))
    bundle.certificate.extend(certificates)
    return bundle


# Result models ---------------------------------------------------------------
class GroupResult(BaseModel):
    """Base adapter-level outcome for finite group computations."""

    status: OperationStatus
    trust: TrustLevel = TrustLevel.UNKNOWN
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    diagnostics: list[str] = Field(default_factory=list)
    limits: dict[str, int] = Field(default_factory=lambda: dict(LIMITS))


class OrderResult(GroupResult):
    value: int | None = None


class SubgroupInfo(BaseModel):
    elements: list[int]
    order: int


class SubgroupResult(GroupResult):
    elements: list[int] = Field(default_factory=list)
    order: int = 0


class SubgroupsResult(GroupResult):
    subgroups: list[SubgroupInfo] = Field(default_factory=list)
    complete: bool = False


class CosetsResult(GroupResult):
    subgroup: list[int] = Field(default_factory=list)
    side: Literal["left", "right"] = "left"
    cosets: list[list[int]] = Field(default_factory=list)


class NormalityResult(GroupResult):
    subgroup: list[int] = Field(default_factory=list)
    is_normal: bool = False


class QuotientResult(GroupResult):
    subgroup: list[int] = Field(default_factory=list)
    quotient: "FiniteGroup | None" = None


class ConjugacyClassesResult(GroupResult):
    classes: list[list[int]] = Field(default_factory=list)
    class_sizes: list[int] = Field(default_factory=list)


class OrbitsResult(GroupResult):
    orbits: list[list[int]] = Field(default_factory=list)


class MembershipResult(GroupResult):
    element: list[int] = Field(default_factory=list)
    contains: bool = False


class StabilizerChainResult(GroupResult):
    base: list[int] = Field(default_factory=list)
    strong_generators: list[list[int]] = Field(default_factory=list)
    basic_orbits: list[list[int]] = Field(default_factory=list)
    transversal_sizes: list[int] = Field(default_factory=list)
    order: int = 0
    verified: bool = False


class AbelianAnalysisResult(GroupResult):
    value: "FiniteAbelianGroup | None" = None
    elementary_divisors: list[int] = Field(default_factory=list)


# FiniteGroup -----------------------------------------------------------------
class FiniteGroup(BaseModel):
    """A finite group as an explicit Cayley table over indices 0..order-1.

    Construction re-verifies the group axioms exactly; a table that fails
    closure, cancellation, the identity law, inverses, or associativity is
    rejected with ValueError.
    """

    order: int = Field(ge=1, le=MAX_CAYLEY_ORDER)
    identity: int = Field(ge=0)
    cayley_table: list[list[int]]
    element_names: list[str] | None = None
    inverses: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_group_axioms(self) -> "FiniteGroup":
        n = self.order
        table = self.cayley_table
        if len(table) != n or any(len(row) != n for row in table):
            raise ValueError("cayley_table must be order x order")
        if self.identity >= n:
            raise ValueError("identity must be an element index")
        fast = _validate_cayley_fast(table, self.identity) \
            if _validate_cayley_fast is not None else None
        if fast is not None:
            code, a, b, c, inverses = fast
            if code == 1:
                raise ValueError("cayley_table entries must be element indices")
            if code == 2:
                raise ValueError(f"declared identity fails at element {a}")
            if code == 3:
                raise ValueError(
                    f"row {a} is not a permutation (closure/cancellation)")
            if code == 4:
                raise ValueError(
                    f"column {a} is not a permutation (cancellation)")
            if code == 5:
                raise ValueError(f"associativity fails at ({a}, {b}, {c})")
            if code == 6:
                raise ValueError(f"element {a} has no two-sided inverse")
            self.inverses = inverses
        else:
            for row in table:
                for value in row:
                    if not isinstance(value, int) or isinstance(value, bool) \
                            or not 0 <= value < n:
                        raise ValueError(
                            "cayley_table entries must be element indices")
            e = self.identity
            for g in range(n):
                if table[e][g] != g or table[g][e] != g:
                    raise ValueError(f"declared identity fails at element {g}")
            want = list(range(n))
            for a in range(n):
                if sorted(table[a]) != want:
                    raise ValueError(
                        f"row {a} is not a permutation (closure/cancellation)")
            for b in range(n):
                if sorted(table[a][b] for a in range(n)) != want:
                    raise ValueError(
                        f"column {b} is not a permutation (cancellation)")
            for a in range(n):
                row_a = table[a]
                for b in range(n):
                    ab = row_a[b]
                    row_b = table[b]
                    for c in range(n):
                        if table[ab][c] != row_a[row_b[c]]:
                            raise ValueError(
                                f"associativity fails at ({a}, {b}, {c})")
            inverses = []
            for g in range(n):
                h = table[g].index(e)  # unique: rows are permutations
                if table[h][g] != e:
                    raise ValueError(f"element {g} has no two-sided inverse")
                inverses.append(h)
            self.inverses = inverses
        if self.element_names is not None and len(self.element_names) != n:
            raise ValueError("element_names must match the group order")
        return self

    # -- constructors ---------------------------------------------------------
    @staticmethod
    def cyclic(n: int) -> "FiniteGroup":
        """The cyclic group Z_n on indices 0..n-1 with addition mod n."""
        if not 1 <= n <= MAX_CAYLEY_ORDER:
            raise ValueError(f"cyclic order must be 1..{MAX_CAYLEY_ORDER}")
        return FiniteGroup(
            order=n,
            identity=0,
            cayley_table=[[(i + j) % n for j in range(n)] for i in range(n)],
        )

    # -- primitive operations --------------------------------------------------
    def _check_element(self, g: int) -> int:
        if not isinstance(g, int) or isinstance(g, bool) \
                or not 0 <= g < self.order:
            raise ValueError(f"not an element index: {g!r}")
        return g

    def multiply(self, a: int, b: int) -> int:
        return self.cayley_table[self._check_element(a)][self._check_element(b)]

    def inverse(self, a: int) -> int:
        return self.inverses[self._check_element(a)]

    def power(self, a: int, k: int) -> int:
        a = self._check_element(a)
        if k < 0:
            return self.power(self.inverses[a], -k)
        result = self.identity
        base = a
        while k:
            if k & 1:
                result = self.cayley_table[result][base]
            base = self.cayley_table[base][base]
            k >>= 1
        return result

    def element_order(self, a: int) -> int:
        a = self._check_element(a)
        order = 1
        current = a
        while current != self.identity:
            current = self.cayley_table[current][a]
            order += 1
        return order

    def is_abelian(self) -> bool:
        n = self.order
        table = self.cayley_table
        return all(table[a][b] == table[b][a]
                   for a in range(n) for b in range(a + 1, n))

    # -- evidence-carrying results ---------------------------------------------
    def order_result(self) -> OrderResult:
        return OrderResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            value=self.order,
            evidence=_bundle(
                "cayley_table_axiom_check",
                proposition="carrier is a group of the declared order",
                metadata={"order": self.order}),
        )

    def closure(self, generators: Iterable[int]) -> list[int]:
        """Sorted elements of the subgroup generated by `generators`."""
        gens = [self._check_element(g) for g in generators]
        seen = {self.identity}
        stack = [self.identity]
        table = self.cayley_table
        while stack:
            x = stack.pop()
            row = table[x]
            for g in gens:
                y = row[g]
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return sorted(seen)

    def generated_subgroup(self, generators: Iterable[int]) -> SubgroupResult:
        gens = [self._check_element(g) for g in generators]
        elements = self.closure(gens)
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "subset_closure_enumeration",
                proposition="result is the subgroup generated by the inputs",
                certificates=[_certificate(
                    "subgroup_witness",
                    "elements are closed under the group operation",
                    {"generators": gens, "elements": elements},
                    "mathkernel.finite_groups.FiniteGroup.closure")],
                metadata={"generators": gens}),
        )

    def _require_subgroup(self, subset: Iterable[int]) -> list[int]:
        elements = sorted({self._check_element(g) for g in subset})
        if not elements:
            raise ValueError("subgroup must be nonempty")
        member = set(elements)
        if self.identity not in member:
            raise ValueError("subset does not contain the identity")
        table = self.cayley_table
        for a in elements:
            if self.inverses[a] not in member:
                raise ValueError("subset is not closed under inverses")
            for b in elements:
                if table[a][b] not in member:
                    raise ValueError("subset is not closed under the operation")
        return elements

    def all_subgroups(self) -> SubgroupsResult:
        """Complete subgroup lattice via joins of cyclic subgroups (bounded)."""
        if self.order > MAX_SUBGROUP_ENUM_ORDER:
            return SubgroupsResult(
                status=OperationStatus.UNSUPPORTED,
                trust=TrustLevel.UNKNOWN,
                diagnostics=[
                    f"order {self.order} exceeds max_subgroup_enum_order "
                    f"{MAX_SUBGROUP_ENUM_ORDER}"],
            )
        known: set[frozenset[int]] = set()
        subs: list[frozenset[int]] = []

        def add(s: frozenset[int]) -> bool:
            if s in known:
                return False
            known.add(s)
            subs.append(s)
            return True

        add(frozenset({self.identity}))
        for g in range(self.order):
            add(frozenset(self.closure([g])))
        capped = False
        i = 0
        while i < len(subs) and not capped:
            for j in range(i, len(subs)):
                join = frozenset(self.closure(subs[i] | subs[j]))
                if join not in known:
                    if len(subs) >= MAX_SUBGROUPS:
                        capped = True
                        break
                    add(join)
            i += 1
        infos = [SubgroupInfo(elements=sorted(s), order=len(s)) for s in subs]
        infos.sort(key=lambda info: (info.order, info.elements))
        if capped:
            return SubgroupsResult(
                status=OperationStatus.UNKNOWN,
                trust=TrustLevel.UNKNOWN,
                subgroups=infos,
                complete=False,
                diagnostics=[
                    f"subgroup cap {MAX_SUBGROUPS} reached; "
                    "enumeration is incomplete and not claimed complete"],
                evidence=_bundle(
                    "subgroup_lattice_join_closure",
                    metadata={"found": len(infos), "capped": True}),
            )
        return SubgroupsResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            subgroups=infos,
            complete=True,
            evidence=_bundle(
                "subgroup_lattice_join_closure",
                proposition=(
                    "every subgroup is a join of cyclic subgroups; the "
                    "enumeration is closed under joins and therefore complete"),
                metadata={"count": len(infos)}),
        )

    def cosets(self, subgroup: Iterable[int],
               side: Literal["left", "right"] = "left") -> CosetsResult:
        H = self._require_subgroup(subgroup)
        table = self.cayley_table
        seen: set[frozenset[int]] = set()
        cosets: list[list[int]] = []
        for g in range(self.order):
            if side == "left":
                coset = frozenset(table[g][h] for h in H)
            else:
                coset = frozenset(table[h][g] for h in H)
            if coset not in seen:
                seen.add(coset)
                cosets.append(sorted(coset))
        return CosetsResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            subgroup=H,
            side=side,
            cosets=cosets,
            evidence=_bundle(
                "coset_enumeration",
                proposition="cosets partition the group into |H|-element blocks",
                metadata={"subgroup_order": len(H),
                          "index": len(cosets), "side": side}),
        )

    def is_normal(self, subgroup: Iterable[int]) -> NormalityResult:
        H = self._require_subgroup(subgroup)
        member = set(H)
        table = self.cayley_table
        normal = all(
            table[table[g][h]][self.inverses[g]] in member
            for g in range(self.order) for h in H
        )
        return NormalityResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            subgroup=H,
            is_normal=normal,
            evidence=_bundle(
                "conjugation_closure_check",
                proposition=(
                    "subgroup is normal: gHg^-1 = H for all g"
                    if normal else
                    "subgroup is not normal: some conjugate leaves H"),
                certificates=[_certificate(
                    "normality_check",
                    "conjugation closure decided by exhaustive exact check",
                    {"subgroup": H, "is_normal": normal},
                    "mathkernel.finite_groups.FiniteGroup.is_normal")]),
        )

    def quotient(self, subgroup: Iterable[int]) -> QuotientResult:
        """G/H as a verified FiniteGroup; defined only when H is normal."""
        H = self._require_subgroup(subgroup)
        normality = self.is_normal(H)
        if not normality.is_normal:
            return QuotientResult(
                status=OperationStatus.DOES_NOT_EXIST,
                trust=TrustLevel.EXACT,
                subgroup=H,
                quotient=None,
                diagnostics=["subgroup is not normal; quotient is undefined"],
                evidence=normality.evidence,
            )
        cosets = self.cosets(H, "left").cosets
        index_of: dict[int, int] = {}
        for i, coset in enumerate(cosets):
            for x in coset:
                index_of[x] = i
        reps = [coset[0] for coset in cosets]
        m = len(cosets)
        table = self.cayley_table
        quotient_table = [
            [index_of[table[reps[i]][reps[j]]] for j in range(m)]
            for i in range(m)
        ]
        # The constructor re-verifies all axioms of the quotient table.
        quotient_group = FiniteGroup(
            order=m,
            identity=index_of[self.identity],
            cayley_table=quotient_table,
            element_names=[f"coset_{i}" for i in range(m)],
        )
        return QuotientResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            subgroup=H,
            quotient=quotient_group,
            evidence=_bundle(
                "quotient_by_normal_subgroup",
                proposition=(
                    "coset multiplication is well-defined and the quotient "
                    "table satisfies the group axioms (re-verified)"),
                certificates=[_certificate(
                    "quotient_table",
                    "quotient Cayley table re-validated by FiniteGroup",
                    {"subgroup": H, "quotient_order": m},
                    "mathkernel.finite_groups.FiniteGroup.__init__")],
                metadata={"index": m}),
        )

    def center(self) -> SubgroupResult:
        n = self.order
        table = self.cayley_table
        elements = [g for g in range(n)
                    if all(table[g][x] == table[x][g] for x in range(n))]
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "center_by_exhaustive_commutation",
                proposition="center = {g : gx = xg for all x}"),
        )

    def centralizer(self, subset: Iterable[int]) -> SubgroupResult:
        S = sorted({self._check_element(g) for g in subset})
        table = self.cayley_table
        elements = [g for g in range(self.order)
                    if all(table[g][s] == table[s][g] for s in S)]
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "centralizer_by_exhaustive_commutation",
                proposition="centralizer = {g : gs = sg for all s in subset}",
                metadata={"subset": S}),
        )

    def conjugacy_classes(self) -> ConjugacyClassesResult:
        table = self.cayley_table
        seen = [False] * self.order
        classes: list[list[int]] = []
        for g in range(self.order):
            if seen[g]:
                continue
            orbit = sorted({
                table[table[x][g]][self.inverses[x]]
                for x in range(self.order)
            })
            for member in orbit:
                seen[member] = True
            classes.append(orbit)
        return ConjugacyClassesResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            classes=classes,
            class_sizes=[len(c) for c in classes],
            evidence=_bundle(
                "conjugacy_class_enumeration",
                proposition="conjugacy classes partition the group",
                metadata={"class_sizes": [len(c) for c in classes]}),
        )

    def commutator_subgroup(self) -> SubgroupResult:
        table = self.cayley_table
        inv = self.inverses
        commutators = sorted({
            table[table[table[a][b]][inv[a]]][inv[b]]
            for a in range(self.order) for b in range(self.order)
        })
        elements = self.closure(commutators)
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "commutator_subgroup_closure",
                proposition=(
                    "derived subgroup [G, G] is the closure of all "
                    "commutators aba^-1b^-1"),
                metadata={"commutators": commutators}),
        )


# Group actions ---------------------------------------------------------------
class GroupAction(BaseModel):
    """A left action of a FiniteGroup on the points 0..n_points-1.

    action[g][x] is the image of point x under group element g. Construction
    re-verifies that every row is a permutation and that the action law
    action[g*h] = action[g] o action[h] holds exactly.
    """

    group: FiniteGroup
    n_points: int = Field(ge=1, le=MAX_ACTION_POINTS)
    action: list[list[int]]

    @model_validator(mode="after")
    def _validate_action(self) -> "GroupAction":
        n = self.group.order
        if len(self.action) != n:
            raise ValueError("action must define one permutation per element")
        want = list(range(self.n_points))
        for g in range(n):
            row = self.action[g]
            if len(row) != self.n_points or sorted(row) != want:
                raise ValueError(f"action row {g} is not a permutation")
        identity_row = list(range(self.n_points))
        if self.action[self.group.identity] != identity_row:
            raise ValueError("identity element must act trivially")
        table = self.group.cayley_table
        for g in range(n):
            row_g = self.action[g]
            for h in range(n):
                row_gh = self.action[table[g][h]]
                row_h = self.action[h]
                for x in range(self.n_points):
                    if row_gh[x] != row_g[row_h[x]]:
                        raise ValueError(
                            f"action law fails at ({g}, {h}, {x})")
        return self

    def orbit_of(self, point: int) -> list[int]:
        if not 0 <= point < self.n_points:
            raise ValueError("point out of range")
        return sorted({self.action[g][point] for g in range(self.group.order)})

    def orbits(self) -> OrbitsResult:
        seen = [False] * self.n_points
        orbits: list[list[int]] = []
        for x in range(self.n_points):
            if seen[x]:
                continue
            orbit = self.orbit_of(x)
            for y in orbit:
                seen[y] = True
            orbits.append(orbit)
        return OrbitsResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            orbits=orbits,
            evidence=_bundle(
                "orbit_enumeration",
                proposition="orbits partition the point set",
                metadata={"n_points": self.n_points,
                          "n_orbits": len(orbits)}),
        )

    def stabilizer(self, point: int) -> SubgroupResult:
        orbit = self.orbit_of(point)
        elements = [g for g in range(self.group.order)
                    if self.action[g][point] == point]
        verified = len(orbit) * len(elements) == self.group.order
        return SubgroupResult(
            status=OperationStatus.VERIFIED if verified else OperationStatus.UNKNOWN,
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "stabilizer_enumeration",
                proposition=(
                    "orbit-stabilizer: |orbit| * |stabilizer| = |G|"
                    if verified else
                    "orbit-stabilizer check failed; result not trusted"),
                metadata={"point": point, "orbit": orbit,
                          "orbit_size": len(orbit),
                          "group_order": self.group.order},
                trust="exact" if verified else "unknown"),
        )


# Homomorphisms ---------------------------------------------------------------
class GroupHomomorphism(BaseModel):
    """An explicit map between FiniteGroups, verified as a homomorphism.

    images[g] is the codomain index of the image of domain element g. The
    constructor checks the homomorphism law f(ab) = f(a)f(b) on all pairs.
    """

    domain: FiniteGroup
    codomain: FiniteGroup
    images: list[int]

    @model_validator(mode="after")
    def _validate_homomorphism(self) -> "GroupHomomorphism":
        if len(self.images) != self.domain.order:
            raise ValueError("images must map every domain element")
        for value in self.images:
            if not isinstance(value, int) or isinstance(value, bool) \
                    or not 0 <= value < self.codomain.order:
                raise ValueError("images must be codomain element indices")
        dom_table = self.domain.cayley_table
        cod_table = self.codomain.cayley_table
        images = self.images
        for a in range(self.domain.order):
            row = dom_table[a]
            fa = images[a]
            for b in range(self.domain.order):
                if images[row[b]] != cod_table[fa][images[b]]:
                    raise ValueError(
                        f"homomorphism law fails at ({a}, {b})")
        return self

    def kernel(self) -> SubgroupResult:
        e = self.codomain.identity
        elements = [g for g in range(self.domain.order)
                    if self.images[g] == e]
        normality = self.domain.is_normal(elements)
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=elements,
            order=len(elements),
            evidence=_bundle(
                "homomorphism_kernel",
                proposition="kernel is a normal subgroup of the domain",
                certificates=[_certificate(
                    "kernel_normality",
                    "kernel normality re-checked by conjugation",
                    {"kernel": elements, "is_normal": normality.is_normal},
                    "mathkernel.finite_groups.FiniteGroup.is_normal",
                    verified=normality.is_normal)]),
        )

    def image(self) -> SubgroupResult:
        elements = sorted(set(self.images))
        # Re-verify the image is a subgroup of the codomain, not just a subset.
        checked = self.codomain._require_subgroup(elements)
        return SubgroupResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            elements=checked,
            order=len(checked),
            evidence=_bundle(
                "homomorphism_image",
                proposition="image is a subgroup of the codomain (re-verified)"),
        )

    def is_injective(self) -> bool:
        return self.kernel().order == 1

    def is_surjective(self) -> bool:
        return len(set(self.images)) == self.codomain.order

    def is_isomorphism(self) -> bool:
        return self.is_injective() and self.is_surjective()


# Permutation groups (sympy.combinatorics) --------------------------------------
def _import_sympy_combinatorics():
    try:
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.perm_groups import (
            PermutationGroup as SympyPermutationGroup,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PermutationGroup requires sympy.combinatorics") from exc
    return Permutation, SympyPermutationGroup


class PermutationGroup(BaseModel):
    """A subgroup of Sym(0..degree-1) given by generators in image-list form.

    generators[k][i] is the image of point i under generator k. All structural
    computation is delegated to sympy.combinatorics (deterministic exact
    Schreier-Sims); the stabilizer chain is returned as a certificate whose
    order claim is independently re-verified from the transversal sizes.
    """

    degree: int = Field(ge=1, le=MAX_PERM_DEGREE)
    generators: list[list[int]] = Field(min_length=1)

    _sympy_group: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_generators(self) -> "PermutationGroup":
        want = list(range(self.degree))
        for k, images in enumerate(self.generators):
            if len(images) != self.degree or sorted(images) != want:
                raise ValueError(
                    f"generator {k} is not a permutation of 0..{self.degree - 1}")
        return self

    # -- constructors ---------------------------------------------------------
    @staticmethod
    def symmetric(n: int) -> "PermutationGroup":
        """S_n generated by an adjacent transposition and an n-cycle."""
        if not 1 <= n <= MAX_PERM_DEGREE:
            raise ValueError(f"degree must be 1..{MAX_PERM_DEGREE}")
        if n == 1:
            return PermutationGroup(degree=1, generators=[[0]])
        transposition = [1, 0] + list(range(2, n))
        cycle = list(range(1, n)) + [0]
        return PermutationGroup(degree=n,
                                generators=[transposition, cycle])

    @staticmethod
    def cyclic(n: int) -> "PermutationGroup":
        """C_n as the group generated by one n-cycle."""
        if not 1 <= n <= MAX_PERM_DEGREE:
            raise ValueError(f"degree must be 1..{MAX_PERM_DEGREE}")
        return PermutationGroup(
            degree=n, generators=[list(range(1, n)) + [0]])

    @staticmethod
    def dihedral(n: int) -> "PermutationGroup":
        """D_n (order 2n) as symmetries of the n-gon on vertices 0..n-1."""
        if not 2 <= n <= MAX_PERM_DEGREE:
            raise ValueError(f"gon size must be 2..{MAX_PERM_DEGREE}")
        rotation = list(range(1, n)) + [0]
        reflection = [(-i) % n for i in range(n)]
        return PermutationGroup(degree=n, generators=[rotation, reflection])

    # -- sympy bridge -----------------------------------------------------------
    def _sympy(self):
        if self._sympy_group is None:
            Permutation, SympyPermutationGroup = _import_sympy_combinatorics()
            self._sympy_group = SympyPermutationGroup(
                *[Permutation(images) for images in self.generators])
        return self._sympy_group

    @staticmethod
    def _images_of(sympy_perm, degree: int) -> list[int]:
        images = [int(v) for v in sympy_perm.array_form]
        if len(images) < degree:
            images.extend(range(len(images), degree))
        return images

    # -- evidence-carrying results ---------------------------------------------
    def stabilizer_chain(self) -> StabilizerChainResult:
        """Schreier-Sims stabilizer chain with re-verified order certificate."""
        group = self._sympy()
        group.schreier_sims()
        base = [int(b) for b in group.base]
        strong = [self._images_of(g, self.degree) for g in group.strong_gens]
        transversal_sizes = [len(t) for t in group.basic_transversals]
        basic_orbits = [sorted(int(x) for x in orbit)
                        for orbit in group.basic_orbits]
        order = int(group.order())
        product = 1
        for size in transversal_sizes:
            product *= size
        verified = product == order
        witness = {
            "base": base,
            "transversal_sizes": transversal_sizes,
            "basic_orbits": basic_orbits,
            "order": order,
            "transversal_product": product,
        }
        return StabilizerChainResult(
            status=(OperationStatus.VERIFIED if verified
                    else OperationStatus.UNKNOWN),
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            base=base,
            strong_generators=strong,
            basic_orbits=basic_orbits,
            transversal_sizes=transversal_sizes,
            order=order,
            verified=verified,
            evidence=_bundle(
                "schreier_sims",
                engine="sympy",
                proposition=(
                    "|G| equals the product of the basic transversal sizes"
                    if verified else
                    "transversal product does not match sympy order; "
                    "chain not trusted"),
                certificates=[_certificate(
                    "stabilizer_chain",
                    "base and strong generating set with order re-verified "
                    "from transversal sizes",
                    witness,
                    "mathkernel.finite_groups:transversal_product_check",
                    verified=verified)],
                metadata={"degree": self.degree},
                trust="exact" if verified else "unknown"),
        )

    def order_result(self) -> OrderResult:
        chain = self.stabilizer_chain()
        return OrderResult(
            status=chain.status,
            trust=chain.trust,
            value=chain.order if chain.verified else None,
            evidence=chain.evidence,
            diagnostics=list(chain.diagnostics),
        )

    def contains(self, images: list[int]) -> MembershipResult:
        """Membership test via the deterministic Schreier-Sims chain."""
        if len(images) != self.degree \
                or sorted(images) != list(range(self.degree)):
            raise ValueError("element is not a permutation of the degree")
        Permutation, _ = _import_sympy_combinatorics()
        group = self._sympy()
        group.schreier_sims()
        member = bool(group.contains(Permutation(images)))
        return MembershipResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            element=[int(v) for v in images],
            contains=member,
            evidence=_bundle(
                "schreier_sims_coset_factor",
                engine="sympy",
                proposition=(
                    "element strips through the stabilizer chain to the "
                    "identity" if member else
                    "element does not strip through the stabilizer chain"),
                metadata={"degree": self.degree}),
        )

    def orbit_of(self, point: int) -> OrbitsResult:
        if not 0 <= point < self.degree:
            raise ValueError("point out of range")
        orbit = sorted(int(x) for x in self._sympy().orbit(point))
        return OrbitsResult(
            status=OperationStatus.VERIFIED,
            trust=TrustLevel.EXACT,
            orbits=[orbit],
            evidence=_bundle(
                "orbit_transversal_enumeration",
                engine="sympy",
                metadata={"point": point, "orbit_size": len(orbit)}),
        )

    def stabilizer_of(self, point: int) -> SubgroupResult:
        if not 0 <= point < self.degree:
            raise ValueError("point out of range")
        group = self._sympy()
        stabilizer = group.stabilizer(point)
        orbit_size = len(group.orbit(point))
        stab_order = int(stabilizer.order())
        group_order = int(group.order())
        verified = orbit_size * stab_order == group_order
        generators = [self._images_of(g, self.degree)
                      for g in stabilizer.generators]
        return SubgroupResult(
            status=(OperationStatus.VERIFIED if verified
                    else OperationStatus.UNKNOWN),
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            elements=[],
            order=stab_order,
            evidence=_bundle(
                "stabilizer_schreier",
                engine="sympy",
                proposition=(
                    "orbit-stabilizer: |orbit| * |stabilizer| = |G|"
                    if verified else
                    "orbit-stabilizer check failed; result not trusted"),
                metadata={"point": point,
                          "stabilizer_generators": generators,
                          "orbit_size": orbit_size,
                          "group_order": group_order},
                trust="exact" if verified else "unknown"),
            diagnostics=[] if verified else [
                "orbit-stabilizer verification failed"],
        )


# Finite abelian groups ---------------------------------------------------------
def _prime_factorization(n: int) -> list[tuple[int, int]]:
    factors: list[tuple[int, int]] = []
    rest = n
    p = 2
    while p * p <= rest:
        if rest % p == 0:
            k = 0
            while rest % p == 0:
                rest //= p
                k += 1
            factors.append((p, k))
        p += 1 if p == 2 else 2
    if rest > 1:
        factors.append((rest, 1))
    return factors


def _verify_abelian_witness(group: FiniteGroup,
                            elementary_divisors: list[int]) -> bool:
    """Re-check that p-primary solution counts match the claimed divisors."""
    product = 1
    for d in elementary_divisors:
        product *= d
    if product != group.order:
        return False
    orders = [group.element_order(g) for g in range(group.order)]
    by_prime: dict[int, list[int]] = {}
    for d in elementary_divisors:
        factors = _prime_factorization(d)
        if len(factors) != 1:
            return False
        p, a = factors[0]
        by_prime.setdefault(p, []).append(a)
    for p, exponents in by_prime.items():
        for k in range(1, max(exponents) + 1):
            q = p ** k
            actual = sum(1 for o in orders if q % o == 0)
            predicted = 1
            for a in exponents:
                predicted *= p ** min(k, a)
            if actual != predicted:
                return False
    return True


class FiniteAbelianGroup(BaseModel):
    """A finite abelian group in invariant-factor form Z_d1 x ... x Z_dk.

    invariant_factors satisfies d_i >= 2 and d_i | d_{i+1}; the empty list is
    the trivial group. Use ``from_finite_group`` to derive this form from a
    Cayley table with a re-verified certificate.
    """

    invariant_factors: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_factors(self) -> "FiniteAbelianGroup":
        for d in self.invariant_factors:
            if not isinstance(d, int) or isinstance(d, bool) or d < 2:
                raise ValueError("invariant factors must be integers >= 2")
        for a, b in zip(self.invariant_factors, self.invariant_factors[1:]):
            if b % a != 0:
                raise ValueError("invariant factors must satisfy d_i | d_{i+1}")
        return self

    @property
    def order(self) -> int:
        order = 1
        for d in self.invariant_factors:
            order *= d
        return order

    @property
    def exponent(self) -> int:
        return self.invariant_factors[-1] if self.invariant_factors else 1

    @property
    def is_cyclic(self) -> bool:
        return len(self.invariant_factors) <= 1

    @classmethod
    def from_finite_group(cls, group: FiniteGroup) -> AbelianAnalysisResult:
        """Invariant-factor decomposition of an abelian Cayley-table group.

        For each prime p | |G| the p-primary exponents are read off the exact
        counts c_k = #{x : x^(p^k) = e}; the claimed elementary divisors are
        then re-verified against those counts before exact trust is granted.
        """
        if not group.is_abelian():
            return AbelianAnalysisResult(
                status=OperationStatus.DOES_NOT_EXIST,
                trust=TrustLevel.EXACT,
                value=None,
                diagnostics=["group is not abelian; invariant-factor form "
                             "is undefined"],
                evidence=_bundle(
                    "abelianness_check",
                    proposition="group is not abelian (commutation check "
                                "failed)"),
            )
        n = group.order
        orders = [group.element_order(g) for g in range(n)]
        elementary: list[int] = []
        for p, _ in _prime_factorization(n):
            p_part = 1
            while n % (p_part * p) == 0:
                p_part *= p
            s_prev = 0
            s_values: list[int] = []
            k = 0
            while True:
                k += 1
                q = p ** k
                c_k = sum(1 for o in orders if q % o == 0)
                s_k = 0
                rest = c_k
                while rest % p == 0 and rest > 1:
                    rest //= p
                    s_k += 1
                s_values.append(s_k)
                if c_k == p_part:
                    break
            diffs = [s_values[i] - (s_values[i - 1] if i else s_prev)
                     for i in range(len(s_values))]
            diffs.append(0)
            for k in range(1, len(s_values) + 1):
                multiplicity = diffs[k - 1] - diffs[k]
                elementary.extend([p ** k] * multiplicity)
        elementary.sort()
        # Combine elementary divisors into invariant factors (right-aligned).
        by_prime: dict[int, list[int]] = {}
        for d in elementary:
            p, a = _prime_factorization(d)[0]
            by_prime.setdefault(p, []).append(a)
        depth = max((len(v) for v in by_prime.values()), default=0)
        invariant_factors: list[int] = []
        for j in range(depth):
            factor = 1
            for p, exponents in by_prime.items():
                offset = len(exponents) - depth + j
                if offset >= 0:
                    factor *= p ** exponents[offset]
            invariant_factors.append(factor)
        invariant_factors = [d for d in invariant_factors if d > 1]
        verified = _verify_abelian_witness(group, elementary)
        value = cls(invariant_factors=invariant_factors) if verified else None
        return AbelianAnalysisResult(
            status=(OperationStatus.VERIFIED if verified
                    else OperationStatus.UNKNOWN),
            trust=TrustLevel.EXACT if verified else TrustLevel.UNKNOWN,
            value=value,
            elementary_divisors=elementary,
            diagnostics=[] if verified else [
                "elementary-divisor witness failed re-verification"],
            evidence=_bundle(
                "abelian_p_primary_counting",
                proposition=(
                    "elementary divisors reproduce all p-primary solution "
                    "counts #{x : x^(p^k) = e}" if verified else
                    "elementary-divisor witness did not reproduce the "
                    "p-primary counts"),
                certificates=[_certificate(
                    "abelian_elementary_divisors",
                    "elementary divisors re-verified against element orders",
                    {"elementary_divisors": elementary,
                     "invariant_factors": invariant_factors},
                    "mathkernel.finite_groups:_verify_abelian_witness",
                    verified=verified)],
                metadata={"order": n},
                trust="exact" if verified else "unknown"),
        )


# Resolve forward references introduced by quoted annotations.
QuotientResult.model_rebuild()
AbelianAnalysisResult.model_rebuild()
