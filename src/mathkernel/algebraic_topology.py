# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Finite exact chain complexes and homology for Phase E.4.

Public definitions are validated by :mod:`mathkernel.topology_adapter`.  This
module works only with finite integral boundary matrices.  Homology is computed
over Z directly and by exact base change over Q or GF(p).
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import TypeAlias

import sympy as sp
from pydantic import model_validator

from .engineering import EngineeringModel, checked_result
from .finite_algebra import _smith_normal_form, verify_snf


IntMatrix: TypeAlias = tuple[tuple[int, ...], ...]
Simplex: TypeAlias = tuple[int, ...]
Interval: TypeAlias = tuple[int, int]
Cube: TypeAlias = tuple[Interval, ...]


def _matrix_rows(matrix: IntMatrix, rows: int, cols: int) -> list[list[int]]:
    if rows == 0:
        return []
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"boundary matrix must have shape {rows}x{cols}")
    return [list(row) for row in matrix]


def _matmul(left: list[list[int]], right: list[list[int]],
            left_rows: int, shared: int, right_cols: int) -> list[list[int]]:
    if left_rows == 0:
        return []
    if right_cols == 0:
        return [[] for _ in range(left_rows)]
    return [[sum(left[i][k] * right[k][j] for k in range(shared))
             for j in range(right_cols)] for i in range(left_rows)]


def _zero_matrix(rows: int, cols: int) -> IntMatrix:
    return tuple(tuple(0 for _ in range(cols)) for _ in range(rows))


@lru_cache(maxsize=32)
def _cached_snf(matrix: IntMatrix):
    """Cache immutable SNF certificates for repeated homology/base queries."""
    source = [list(row) for row in matrix]
    D, U, V = _smith_normal_form(source)
    checks = verify_snf(source, D, U, V)
    freeze = lambda value: tuple(tuple(row) for row in value)
    U_inv = [[int(value) for value in row] for row in sp.Matrix(U).inv().tolist()]
    V_inv = [[int(value) for value in row] for row in sp.Matrix(V).inv().tolist()]
    return (freeze(D), freeze(U), freeze(V), freeze(U_inv), freeze(V_inv),
            tuple(checks.items()))


def cache_info():
    info = _cached_snf.cache_info()
    return {"smith_normal_forms": {name: getattr(info, name) for name in
            ("hits", "misses", "maxsize", "currsize")}}


def clear_caches():
    _cached_snf.cache_clear()


class ChainComplex(EngineeringModel):
    """A finite chain complex of free abelian groups in degrees 0..n."""

    ranks: tuple[int, ...]
    boundaries: tuple[IntMatrix, ...]
    basis_labels: tuple[tuple[str, ...], ...] = ()
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_chain_complex(self):
        if not self.ranks or any(isinstance(rank, bool) or rank < 0
                                 for rank in self.ranks):
            raise ValueError("chain ranks must be nonnegative integers")
        if len(self.boundaries) != len(self.ranks) - 1:
            raise ValueError("one boundary matrix is required for each positive degree")
        for degree, matrix in enumerate(self.boundaries, start=1):
            _matrix_rows(matrix, self.ranks[degree - 1], self.ranks[degree])
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for row in matrix for value in row):
                raise ValueError("integral chain-complex entries must be integers")
        if self.basis_labels:
            if len(self.basis_labels) != len(self.ranks) or any(
                    len(labels) != rank for labels, rank in zip(self.basis_labels, self.ranks)):
                raise ValueError("basis labels must match every chain rank")
            if any(len(set(labels)) != len(labels) for labels in self.basis_labels):
                raise ValueError("basis labels must be unique within each degree")
        if self.input_trust != "exact":
            raise ValueError("finite integral chain complexes require exact trust")
        return self


class SimplicialComplex(EngineeringModel):
    vertices: tuple[str, ...]
    simplices: tuple[tuple[Simplex, ...], ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_simplicial_complex(self):
        if not self.vertices or len(set(self.vertices)) != len(self.vertices):
            raise ValueError("simplicial vertices must be nonempty and uniquely labelled")
        if not self.simplices:
            raise ValueError("simplicial complex must contain cells")
        seen: set[Simplex] = set()
        for degree, layer in enumerate(self.simplices):
            if len(set(layer)) != len(layer):
                raise ValueError("duplicate simplices are not allowed")
            for simplex in layer:
                if len(simplex) != degree + 1 or tuple(sorted(simplex)) != simplex:
                    raise ValueError("simplices must be canonically ordered by dimension")
                if any(isinstance(i, bool) or not isinstance(i, int)
                       or i < 0 or i >= len(self.vertices) for i in simplex):
                    raise ValueError("simplex vertex index is outside the vertex range")
                seen.add(simplex)
        if any((face not in seen) for simplex in seen if len(simplex) > 1
               for face in combinations(simplex, len(simplex) - 1)):
            raise ValueError("simplices must be closed under faces")
        if self.input_trust != "exact":
            raise ValueError("finite combinatorial complexes require exact trust")
        return self


class CubicalComplex(EngineeringModel):
    ambient_dimension: int
    cells: tuple[tuple[Cube, ...], ...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_cubical_complex(self):
        if isinstance(self.ambient_dimension, bool) or self.ambient_dimension < 1:
            raise ValueError("cubical ambient dimension must be positive")
        if not self.cells:
            raise ValueError("cubical complex must contain cells")
        seen: set[Cube] = set()
        for degree, layer in enumerate(self.cells):
            if len(set(layer)) != len(layer):
                raise ValueError("duplicate cubes are not allowed")
            for cube in layer:
                if len(cube) != self.ambient_dimension:
                    raise ValueError("cube coordinate count must match ambient dimension")
                if any(isinstance(lo, bool) or isinstance(hi, bool)
                       or not isinstance(lo, int) or not isinstance(hi, int)
                       or hi - lo not in {0, 1} for lo, hi in cube):
                    raise ValueError("elementary cube intervals must be [a,a] or [a,a+1]")
                if sum(hi - lo for lo, hi in cube) != degree:
                    raise ValueError("cube dimension does not match its cell layer")
                seen.add(cube)
        if any(face not in seen for cube in seen for _, face in _cube_faces(cube)):
            raise ValueError("cubes must be closed under faces")
        if self.input_trust != "exact":
            raise ValueError("finite combinatorial complexes require exact trust")
        return self


class HomologyGroup(EngineeringModel):
    degree: int
    coefficient_domain: str
    prime: int | None = None
    chain_rank: int
    cycle_rank: int
    boundary_rank: int
    betti_number: int
    torsion_coefficients: tuple[int, ...] = ()
    cycle_representatives: tuple[tuple[sp.Expr, ...], ...] = ()
    torsion_representatives: tuple[tuple[int, ...], ...] = ()
    structure: str

    @model_validator(mode="after")
    def validate_homology_group(self):
        if any(isinstance(value, bool) or value < 0 for value in
               (self.degree, self.chain_rank, self.cycle_rank,
                self.boundary_rank, self.betti_number)):
            raise ValueError("homology ranks and degree must be nonnegative integers")
        if self.betti_number != self.cycle_rank - self.boundary_rank:
            raise ValueError("Betti number must equal cycle rank minus boundary rank")
        if any(value < 2 for value in self.torsion_coefficients) or any(
                right % left for left, right in zip(
                    self.torsion_coefficients, self.torsion_coefficients[1:])):
            raise ValueError("torsion coefficients must form a divisibility chain")
        if any(len(vector) != self.chain_rank for vector in
               self.cycle_representatives + self.torsion_representatives):
            raise ValueError("homology representatives must use the stored chain basis")
        if self.coefficient_domain not in {"Z", "Q", "GF(p)"}:
            raise ValueError("homology coefficient domain is invalid")
        if self.coefficient_domain == "GF(p)":
            if not _is_prime(self.prime):
                raise ValueError("GF(p) homology requires a prime")
        elif self.prime is not None:
            raise ValueError("only GF(p) homology has a prime")
        if self.coefficient_domain != "Z" and self.torsion_coefficients:
            raise ValueError("field homology cannot carry integer torsion coefficients")
        return self


class Homology(EngineeringModel):
    coefficient_domain: str
    prime: int | None = None
    groups: tuple[HomologyGroup, ...]
    euler_characteristic: int | None = None
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_homology(self):
        if not self.groups or len({group.degree for group in self.groups}) != len(self.groups):
            raise ValueError("homology must contain uniquely graded groups")
        if any(group.coefficient_domain != self.coefficient_domain or
               group.prime != self.prime for group in self.groups):
            raise ValueError("homology groups must share one coefficient domain")
        if self.input_trust != "exact":
            raise ValueError("finite homology results require exact trust")
        return self


def simplicial_closure(vertex_count: int, maximal_simplices, *,
                       max_cells: int | None = None) -> tuple[tuple[Simplex, ...], ...]:
    maximal: set[Simplex] = set()
    for raw in maximal_simplices:
        simplex = tuple(sorted(raw))
        if not simplex or len(set(simplex)) != len(simplex):
            raise ValueError("maximal simplices require distinct vertex indices")
        if any(i < 0 or i >= vertex_count for i in simplex):
            raise ValueError("simplex vertex index is outside the vertex range")
        maximal.add(simplex)
    if not maximal:
        raise ValueError("at least one maximal simplex is required")
    closure: set[Simplex] = set()
    for simplex in maximal:
        for size in range(1, len(simplex) + 1):
            closure.update(combinations(simplex, size))
            if max_cells is not None and len(closure) > max_cells:
                raise ValueError("simplicial closure exceeds max_topology_cells")
    maximum = max(len(simplex) for simplex in closure)
    return tuple(tuple(sorted(simplex for simplex in closure if len(simplex) == size))
                 for size in range(1, maximum + 1))


def _cube_faces(cube: Cube) -> tuple[tuple[int, Cube], ...]:
    faces = []
    position = 0
    for axis, (lo, hi) in enumerate(cube):
        if hi == lo:
            continue
        lower = list(cube); upper = list(cube)
        lower[axis] = (lo, lo); upper[axis] = (hi, hi)
        sign = -1 if position % 2 == 0 else 1
        faces.append((sign, tuple(lower)))
        faces.append((-sign, tuple(upper)))
        position += 1
    return tuple(faces)


def cubical_closure(ambient_dimension: int, maximal_cells, *,
                    max_cells: int | None = None) -> tuple[tuple[Cube, ...], ...]:
    pending = [tuple(tuple(interval) for interval in cell) for cell in maximal_cells]
    if not pending:
        raise ValueError("at least one maximal cube is required")
    closure: set[Cube] = set()
    while pending:
        cube = pending.pop()
        if len(cube) != ambient_dimension:
            raise ValueError("cube coordinate count must match ambient dimension")
        if any(isinstance(lo, bool) or isinstance(hi, bool)
               or not isinstance(lo, int) or not isinstance(hi, int)
               or hi - lo not in {0, 1} for lo, hi in cube):
            raise ValueError("elementary cube intervals must be [a,a] or [a,a+1]")
        if cube in closure:
            continue
        closure.add(cube)
        if max_cells is not None and len(closure) > max_cells:
            raise ValueError("cubical closure exceeds max_topology_cells")
        pending.extend(face for _, face in _cube_faces(cube))
    maximum = max(sum(hi - lo for lo, hi in cube) for cube in closure)
    return tuple(tuple(sorted(cube for cube in closure
                              if sum(hi - lo for lo, hi in cube) == degree))
                 for degree in range(maximum + 1))


def simplicial_chain_complex(complex_: SimplicialComplex) -> ChainComplex:
    ranks = tuple(len(layer) for layer in complex_.simplices)
    boundaries = []
    for degree in range(1, len(ranks)):
        lower_index = {simplex: i for i, simplex in enumerate(complex_.simplices[degree - 1])}
        matrix = [[0] * ranks[degree] for _ in range(ranks[degree - 1])]
        for column, simplex in enumerate(complex_.simplices[degree]):
            for removed in range(len(simplex)):
                face = simplex[:removed] + simplex[removed + 1:]
                matrix[lower_index[face]][column] = -1 if removed % 2 else 1
        boundaries.append(tuple(tuple(row) for row in matrix))
    labels = tuple(tuple("[" + ",".join(complex_.vertices[i] for i in simplex) + "]"
                         for simplex in layer) for layer in complex_.simplices)
    return ChainComplex(ranks=ranks, boundaries=tuple(boundaries),
                        basis_labels=labels, input_trust="exact")


def _cube_label(cube: Cube) -> str:
    return "x".join(f"[{lo}]" if lo == hi else f"[{lo},{hi}]" for lo, hi in cube)


def cubical_chain_complex(complex_: CubicalComplex) -> ChainComplex:
    ranks = tuple(len(layer) for layer in complex_.cells)
    boundaries = []
    for degree in range(1, len(ranks)):
        lower_index = {cube: i for i, cube in enumerate(complex_.cells[degree - 1])}
        matrix = [[0] * ranks[degree] for _ in range(ranks[degree - 1])]
        for column, cube in enumerate(complex_.cells[degree]):
            for sign, face in _cube_faces(cube):
                matrix[lower_index[face]][column] += sign
        boundaries.append(tuple(tuple(row) for row in matrix))
    labels = tuple(tuple(_cube_label(cube) for cube in layer) for layer in complex_.cells)
    return ChainComplex(ranks=ranks, boundaries=tuple(boundaries),
                        basis_labels=labels, input_trust="exact")


def triangulation_to_simplicial_complex(triangulation, *, max_cells: int,
                                        max_matrix_entries: int,
                                        max_work: int):
    """Forget exact geometric coordinates only after certifying the realization."""
    from .computational_geometry import verify_triangulation

    if triangulation.input_trust != "exact":
        raise ValueError("conversion to exact combinatorial topology requires exact coordinates")
    verified = verify_triangulation(triangulation)
    if not all(value is True for value in verified.verification.values()):
        raise ValueError("conversion requires a verified nonoverlapping triangulation")
    labels = tuple(f"v{index}:{','.join(sp.sstr(value) for value in point)}"
                   for index, point in enumerate(triangulation.points))
    simplices = simplicial_closure(len(labels), triangulation.triangles,
                                   max_cells=max_cells)
    ranks = tuple(len(layer) for layer in simplices)
    matrix_entries = sum(ranks[k - 1] * ranks[k]
                         for k in range(1, len(ranks)))
    if matrix_entries > max_matrix_entries:
        raise ValueError("derived boundary matrices exceed max_topology_matrix_entries")
    verification_work = max((ranks[k - 2] * ranks[k - 1] * ranks[k]
                             for k in range(2, len(ranks))), default=0)
    if max(sum(ranks), matrix_entries, verification_work) > max_work:
        raise ValueError("triangulation conversion exceeds max_topology_work")
    output = SimplicialComplex(vertices=labels, simplices=simplices,
                               input_trust="exact")
    chain = simplicial_chain_complex(output)
    chain_checks = verify_chain_complex(chain).verification
    checks = {"triangulation_verified": True, "face_closure": True,
              "boundary_squared_zero": all(chain_checks.values())}
    return checked_result("to_simplicial_complex",
        {"cell_counts": tuple(len(layer) for layer in simplices)},
        method="certified_triangulation_forgetful_map", trust="exact",
        checks=checks, witness={"triangles": triangulation.triangles,
                                "boundaries": chain.boundaries},
        details={"identity_checks": checks}), output


def verify_chain_complex(chain: ChainComplex):
    checks = {}
    products = {}
    for degree in range(2, len(chain.ranks)):
        left = _matrix_rows(chain.boundaries[degree - 2], chain.ranks[degree - 2],
                            chain.ranks[degree - 1])
        right = _matrix_rows(chain.boundaries[degree - 1], chain.ranks[degree - 1],
                             chain.ranks[degree])
        product = _matmul(left, right, chain.ranks[degree - 2],
                          chain.ranks[degree - 1], chain.ranks[degree])
        products[f"d{degree-1}_d{degree}"] = product
        checks[f"boundary_squared_zero_degree_{degree}"] = all(
            value == 0 for row in product for value in row)
    if not checks:
        checks["boundary_squared_zero_vacuous"] = True
    checks["integral_finite_free_chain_groups"] = True
    return checked_result("verify", {"ranks": chain.ranks, "products": products},
        method="exact_boundary_composition", trust="exact", checks=checks,
        witness={"boundaries": chain.boundaries, "products": products},
        details={"identity_checks": checks})


def verify_simplicial_complex(complex_: SimplicialComplex):
    chain = simplicial_chain_complex(complex_)
    verified = verify_chain_complex(chain)
    checks = {"canonical_orientation": all(tuple(sorted(simplex)) == simplex
              for layer in complex_.simplices for simplex in layer),
              "closed_under_faces": True,
              "boundary_squared_zero": all(verified.verification.values())}
    return checked_result("verify", {"dimension": len(complex_.simplices) - 1,
        "cell_counts": tuple(len(layer) for layer in complex_.simplices)},
        method="simplicial_face_closure", trust="exact", checks=checks,
        witness={"boundaries": chain.boundaries}, details={"identity_checks": checks})


def verify_cubical_complex(complex_: CubicalComplex):
    chain = cubical_chain_complex(complex_)
    verified = verify_chain_complex(chain)
    checks = {"elementary_cubes": True, "closed_under_faces": True,
              "boundary_squared_zero": all(verified.verification.values())}
    return checked_result("verify", {"dimension": len(complex_.cells) - 1,
        "cell_counts": tuple(len(layer) for layer in complex_.cells)},
        method="cubical_face_closure", trust="exact", checks=checks,
        witness={"boundaries": chain.boundaries}, details={"identity_checks": checks})


def boundary_matrix(chain: ChainComplex, degree: int):
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree < len(chain.ranks):
        raise ValueError("degree must index a chain group")
    matrix = (_zero_matrix(0, chain.ranks[0]) if degree == 0
              else chain.boundaries[degree - 1])
    checks = {"shape_verified": (degree == 0 or
        len(matrix) == chain.ranks[degree - 1] and
        all(len(row) == chain.ranks[degree] for row in matrix))}
    return checked_result("boundary_matrix", {"degree": degree, "matrix": matrix,
        "shape": (0 if degree == 0 else chain.ranks[degree - 1], chain.ranks[degree])},
        method="stored_integral_boundary", trust="exact", checks=checks)


def _is_prime(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 2 and bool(sp.isprime(value))


def _mod_rref(matrix: list[list[int]], rows: int, cols: int, prime: int):
    out = [[value % prime for value in row] for row in matrix]
    pivots = []
    pivot_row = 0
    for column in range(cols):
        pivot = next((row for row in range(pivot_row, rows) if out[row][column]), None)
        if pivot is None:
            continue
        out[pivot_row], out[pivot] = out[pivot], out[pivot_row]
        inverse = pow(out[pivot_row][column], -1, prime)
        out[pivot_row] = [(value * inverse) % prime for value in out[pivot_row]]
        for row in range(rows):
            if row == pivot_row or not out[row][column]:
                continue
            factor = out[row][column]
            out[row] = [(left - factor * right) % prime
                        for left, right in zip(out[row], out[pivot_row])]
        pivots.append(column); pivot_row += 1
        if pivot_row == rows:
            break
    return out, tuple(pivots)


def _mod_nullspace(matrix: list[list[int]], rows: int, cols: int, prime: int):
    rref, pivots = _mod_rref(matrix, rows, cols, prime)
    free = [column for column in range(cols) if column not in pivots]
    basis = []
    for column in free:
        vector = [0] * cols; vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = (-rref[row][column]) % prime
        basis.append(vector)
    return basis, len(pivots), rref, pivots


def _independent_quotient_representatives(image_columns, cycle_columns,
                                          field: str, prime: int | None):
    if not cycle_columns:
        return []
    dimension = len(cycle_columns[0])
    selected = []
    span = list(image_columns)
    def rank(columns):
        if not columns:
            return 0
        rows = [[columns[j][i] for j in range(len(columns))] for i in range(dimension)]
        if field == "Q":
            return int(sp.Matrix(rows).rank())
        return len(_mod_rref(rows, dimension, len(columns), prime)[1])
    current = rank(span)
    for vector in cycle_columns:
        candidate = span + [vector]
        next_rank = rank(candidate)
        if next_rank > current:
            selected.append(vector); span = candidate; current = next_rank
    return selected


def _integer_homology(chain: ChainComplex, degree: int):
    n = chain.ranks[degree]
    if degree == 0:
        A = []
        m = 0
        D = []
        U = []
        V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        rank_a = 0
        snf_a_checks = {"degree_zero_kernel": True}
    else:
        m = chain.ranks[degree - 1]
        A = _matrix_rows(chain.boundaries[degree - 1], m, n)
        if m == 0 or n == 0:
            D = [row[:] for row in A]
            U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
            V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            rank_a = 0
            snf_a_checks = {"empty_boundary_snf": True}
        else:
            (frozen_D, frozen_U, frozen_V, frozen_U_inv, frozen_V_inv,
             frozen_checks) = _cached_snf(
                tuple(tuple(row) for row in A))
            D, U, V = map(lambda value: [list(row) for row in value],
                          (frozen_D, frozen_U, frozen_V))
            snf_a_checks = dict(frozen_checks)
            rank_a = sum(D[i][i] != 0 for i in range(min(m, n)))
    nullity = n - rank_a
    kernel_basis = [[V[row][column] for column in range(rank_a, n)] for row in range(n)]
    next_rank = chain.ranks[degree + 1] if degree + 1 < len(chain.ranks) else 0
    B = (_matrix_rows(chain.boundaries[degree], n, next_rank)
         if degree + 1 < len(chain.ranks) else [[] for _ in range(n)])
    if n:
        V_inv = ([list(row) for row in frozen_V_inv] if degree > 0 and m and n
                 else [[int(value) for value in row] for row in sp.Matrix(V).inv().tolist()])
        transformed = _matmul(V_inv, B, n, n, next_rank)
    else:
        transformed = []
    image_in_kernel = all(transformed[row][column] == 0
                          for row in range(rank_a) for column in range(next_rank))
    coordinates = [row[:] for row in transformed[rank_a:]]
    if nullity == 0 or next_rank == 0:
        relation_rank = 0
        relation_diag = []
        relation_snf = {"empty_relation_snf": True}
        relation_D = [row[:] for row in coordinates]
        relation_U = [[1 if i == j else 0 for j in range(nullity)] for i in range(nullity)]
        relation_V = [[1 if i == j else 0 for j in range(next_rank)] for i in range(next_rank)]
    else:
        (frozen_D, frozen_U, frozen_V, frozen_U_inv, frozen_V_inv,
         frozen_checks) = _cached_snf(
            tuple(tuple(row) for row in coordinates))
        relation_D, relation_U, relation_V = map(
            lambda value: [list(row) for row in value],
            (frozen_D, frozen_U, frozen_V))
        relation_snf = dict(frozen_checks)
        relation_diag = [relation_D[i][i] for i in range(min(nullity, next_rank))]
        relation_rank = sum(value != 0 for value in relation_diag)
    torsion = tuple(value for value in relation_diag if value > 1)
    free_rank = nullity - relation_rank
    if nullity:
        U_inv = ([list(row) for row in frozen_U_inv]
                 if nullity and next_rank else
                 [[int(value) for value in row] for row in sp.Matrix(relation_U).inv().tolist()])
        quotient_coordinates = [[U_inv[row][column] for row in range(nullity)]
                                for column in range(nullity)]
        representatives = [tuple(sum(kernel_basis[row][j] * vector[j]
                                     for j in range(nullity)) for row in range(n))
                           for vector in quotient_coordinates]
    else:
        representatives = []
    torsion_reps = tuple(representatives[i] for i, value in enumerate(relation_diag)
                         if value > 1)
    free_reps = tuple(tuple(sp.Integer(value) for value in representatives[i])
                      for i in range(relation_rank, nullity))
    reconstructed_image = _matmul(kernel_basis, coordinates, n, nullity, next_rank)
    checks = {"boundary_image_in_cycles": image_in_kernel,
              "kernel_coordinates_reconstruct_image": reconstructed_image == B,
              "kernel_snf_verified": all(snf_a_checks.values()),
              "quotient_snf_verified": all(relation_snf.values()),
              "rank_decomposition": free_rank + relation_rank == nullity,
              "torsion_divisibility_chain": all(right % left == 0 for left, right
                  in zip(torsion, torsion[1:]))}
    witness = {"boundary_k_snf": {"D": D, "U": U, "V": V},
               "kernel_basis": kernel_basis, "image_coordinates": coordinates,
               "quotient_snf": {"D": relation_D, "U": relation_U,
                                "V": relation_V},
               "kernel_snf_checks": snf_a_checks,
               "quotient_snf_checks": relation_snf}
    return HomologyGroup(degree=degree, coefficient_domain="Z", chain_rank=n,
        cycle_rank=nullity, boundary_rank=relation_rank, betti_number=free_rank,
        torsion_coefficients=torsion, cycle_representatives=free_reps,
        torsion_representatives=torsion_reps,
        structure=_structure("Z", free_rank, torsion, None)), checks, witness


def _field_homology(chain: ChainComplex, degree: int, field: str, prime: int | None):
    n = chain.ranks[degree]
    m = chain.ranks[degree - 1] if degree else 0
    A = (_matrix_rows(chain.boundaries[degree - 1], m, n) if degree else [])
    next_rank = chain.ranks[degree + 1] if degree + 1 < len(chain.ranks) else 0
    B = (_matrix_rows(chain.boundaries[degree], n, next_rank)
         if degree + 1 < len(chain.ranks) else [[] for _ in range(n)])
    if field == "Q":
        matrix_a = sp.Matrix(A) if m and n else sp.zeros(m, n)
        matrix_b = sp.Matrix(B) if n and next_rank else sp.zeros(n, next_rank)
        cycles = [list(vector) for vector in matrix_a.nullspace()]
        image = [list(vector) for vector in matrix_b.columnspace()]
        rank_a, rank_b = int(matrix_a.rank()), int(matrix_b.rank())
        composition_zero = matrix_a * matrix_b == sp.zeros(m, next_rank)
    else:
        cycles, rank_a, _, _ = _mod_nullspace(A, m, n, prime)
        _, pivot_columns = _mod_rref(B, n, next_rank, prime)
        image = [[B[row][column] % prime for row in range(n)] for column in pivot_columns]
        rank_b = len(pivot_columns)
        composition = _matmul(A, B, m, n, next_rank)
        composition_zero = all(value % prime == 0 for row in composition for value in row)
    reps = _independent_quotient_representatives(image, cycles, field, prime)
    betti = n - rank_a - rank_b
    checks = {"boundary_squared_zero": composition_zero,
              "rank_nullity": len(cycles) == n - rank_a,
              "quotient_dimension": len(reps) == betti}
    converted = tuple(tuple(sp.Rational(value) if field == "Q" else sp.Integer(value)
                            for value in vector) for vector in reps)
    group = HomologyGroup(degree=degree, coefficient_domain=field, prime=prime,
        chain_rank=n, cycle_rank=n-rank_a, boundary_rank=rank_b,
        betti_number=betti, cycle_representatives=converted,
        structure=_structure(field, betti, (), prime))
    witness = {"cycle_representatives": converted, "boundary_rank": rank_b,
               "cycle_rank": n-rank_a}
    return group, checks, witness


def _structure(field: str, betti: int, torsion, prime):
    if field == "Z":
        parts = (["Z" if betti == 1 else f"Z^{betti}"] if betti else [])
        parts.extend(f"Z/{value}Z" for value in torsion)
        return " x ".join(parts) if parts else "0"
    label = "Q" if field == "Q" else f"GF({prime})"
    return "0" if betti == 0 else label if betti == 1 else f"{label}^{betti}"


def homology(chain: ChainComplex, coefficient: str = "Z", prime: int | None = None,
             degree: int | None = None):
    verified = verify_chain_complex(chain)
    if not all(value is True for value in verified.verification.values()):
        raise ValueError("homology requires a verified chain complex with boundary squared zero")
    coefficient = coefficient.upper().replace(" ", "")
    if coefficient in {"GF(P)", "GF"}:
        coefficient = "GF(p)"
    elif coefficient == "Z":
        coefficient = "Z"
    elif coefficient == "Q":
        coefficient = "Q"
    else:
        raise ValueError("coefficient must be Z, Q, or GF(p)")
    if coefficient == "GF(p)" and not _is_prime(prime):
        raise ValueError("GF(p) homology requires a prime integer")
    if coefficient != "GF(p)" and prime is not None:
        raise ValueError("prime is only valid with GF(p) coefficients")
    if degree is not None and (isinstance(degree, bool) or not isinstance(degree, int)
                               or not 0 <= degree < len(chain.ranks)):
        raise ValueError("degree must index a chain group")
    degrees = range(len(chain.ranks)) if degree is None else (degree,)
    groups = []
    checks = {"chain_complex_verified": True}
    witnesses = {}
    for current in degrees:
        if coefficient == "Z":
            group, local_checks, witness = _integer_homology(chain, current)
        else:
            group, local_checks, witness = _field_homology(chain, current, coefficient, prime)
        groups.append(group)
        checks.update({f"H{current}_{name}": value for name, value in local_checks.items()})
        witnesses[f"H{current}"] = witness
    euler = None
    if degree is None:
        chain_euler = sum((-1) ** i * rank for i, rank in enumerate(chain.ranks))
        homology_euler = sum((-1) ** group.degree * group.betti_number for group in groups)
        checks["euler_poincare"] = chain_euler == homology_euler
        euler = chain_euler
    output = Homology(coefficient_domain=coefficient, prime=prime,
                      groups=tuple(groups), euler_characteristic=euler)
    return checked_result("homology", output.model_dump(),
        method=("smith_kernel_quotient" if coefficient == "Z" else
                "exact_rational_rank_quotient" if coefficient == "Q" else
                "modular_rank_quotient"), trust="exact", checks=checks,
        witness=witnesses, details={"identity_checks": checks}), output


def euler_characteristic(chain: ChainComplex):
    value = sum((-1) ** degree * rank for degree, rank in enumerate(chain.ranks))
    return checked_result("euler_characteristic", value,
        method="alternating_chain_rank", trust="exact",
        checks={"finite_chain_groups": True}, witness={"ranks": chain.ranks})
