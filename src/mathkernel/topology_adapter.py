# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted public boundary for Phase E.4 algebraic topology."""
from __future__ import annotations

from .engineering import checked_result
from .models import TrustLevel


TYPES = {
    "simplicialcomplex": "SimplicialComplex",
    "simplicial_complex": "SimplicialComplex",
    "cubicalcomplex": "CubicalComplex",
    "cubical_complex": "CubicalComplex",
    "chaincomplex": "ChainComplex",
    "chain_complex": "ChainComplex",
}

_HOMOLOGY_PARAMETERS = {
    "coefficient": "Z|Q|GF(p) (Z default)",
    "prime": "prime integer required for GF(p)?",
    "degree": "nonnegative chain degree?",
}

OPERATIONS = {
    "SimplicialComplex": {"verify": {}, "chain_complex": {},
                          "boundary_matrix": {"degree": "nonnegative chain degree"},
                          "homology": _HOMOLOGY_PARAMETERS},
    "CubicalComplex": {"verify": {}, "chain_complex": {},
                       "boundary_matrix": {"degree": "nonnegative chain degree"},
                       "homology": _HOMOLOGY_PARAMETERS},
    "ChainComplex": {"verify": {},
                     "boundary_matrix": {"degree": "nonnegative chain degree"},
                     "homology": _HOMOLOGY_PARAMETERS,
                     "euler_characteristic": {}},
}

DERIVED_OUTPUTS = {"chain_complex": "ChainComplex", "homology": "Homology"}


def _integer(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds configured limit {maximum}")
    return value


def construct(kernel, kind, definition):
    from .algebraic_topology import (ChainComplex, CubicalComplex,
        SimplicialComplex, cubical_closure, simplicial_closure)

    typ = TYPES[kind]
    data = dict(definition)
    context_id = data.pop("context_id", None)
    if context_id is not None:
        raise ValueError("combinatorial topology definitions do not accept MathIR context")
    settings = kernel.settings
    if typ == "SimplicialComplex":
        allowed = {"vertices", "maximal_simplices"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown SimplicialComplex field: {sorted(unknown)[0]}")
        vertices = data.get("vertices", ())
        maximal = data.get("maximal_simplices", ())
        if not isinstance(vertices, (list, tuple)) or not vertices:
            raise ValueError("vertices must be a nonempty sequence of labels")
        if len(vertices) > settings.max_topology_cells:
            raise ValueError("vertex count exceeds max_topology_cells")
        if any(not isinstance(label, str) or not label for label in vertices):
            raise ValueError("vertex labels must be nonempty strings")
        if (not isinstance(maximal, (list, tuple)) or
                len(maximal) > settings.max_topology_cells or any(
                not isinstance(simplex, (list, tuple)) or
                any(isinstance(i, bool) or not isinstance(i, int) for i in simplex)
                for simplex in maximal)):
            raise ValueError("maximal_simplices must be a bounded sequence of integer index sequences")
        if any(len(simplex) > settings.max_topology_dimension + 1 for simplex in maximal):
            raise ValueError("simplex exceeds max_topology_dimension")
        closure = simplicial_closure(len(vertices), maximal,
                                     max_cells=settings.max_topology_cells)
        if sum(map(len, closure)) > settings.max_topology_cells:
            raise ValueError("simplicial closure exceeds max_topology_cells")
        return typ, SimplicialComplex(vertices=tuple(vertices), simplices=closure), TrustLevel.EXACT, []
    if typ == "CubicalComplex":
        allowed = {"ambient_dimension", "maximal_cells"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown CubicalComplex field: {sorted(unknown)[0]}")
        ambient = _integer(data.get("ambient_dimension"), "ambient_dimension",
                           minimum=1, maximum=settings.max_topology_dimension)
        maximal = data.get("maximal_cells", ())
        if not isinstance(maximal, (list, tuple)) or len(maximal) > settings.max_topology_cells:
            raise ValueError("maximal_cells must be a bounded sequence")
        if any(not isinstance(cell, (list, tuple)) or
               any(not isinstance(interval, (list, tuple)) or len(interval) != 2
                   for interval in cell) for cell in maximal):
            raise ValueError("each cube must contain [lower, upper] intervals")
        closure = cubical_closure(ambient, maximal,
                                  max_cells=settings.max_topology_cells)
        if sum(map(len, closure)) > settings.max_topology_cells:
            raise ValueError("cubical closure exceeds max_topology_cells")
        return typ, CubicalComplex(ambient_dimension=ambient, cells=closure), TrustLevel.EXACT, []
    allowed = {"ranks", "boundaries", "basis_labels"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown ChainComplex field: {sorted(unknown)[0]}")
    ranks = data.get("ranks", ())
    boundaries = data.get("boundaries", ())
    if not isinstance(ranks, (list, tuple)) or not ranks or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 for rank in ranks):
        raise ValueError("ranks must be a nonempty sequence of nonnegative integers")
    if len(ranks) - 1 > settings.max_topology_dimension:
        raise ValueError("chain complex exceeds max_topology_dimension")
    if sum(ranks) > settings.max_topology_cells:
        raise ValueError("chain ranks exceed max_topology_cells")
    if not isinstance(boundaries, (list, tuple)) or len(boundaries) != len(ranks) - 1:
        raise ValueError("boundaries must provide one matrix per positive degree")
    matrices = []
    entries = 0
    for degree, raw in enumerate(boundaries, start=1):
        if not isinstance(raw, (list, tuple)):
            raise ValueError("each boundary must be a matrix")
        if ranks[degree - 1] and len(raw) != ranks[degree - 1]:
            raise ValueError("boundary row count must match the lower chain rank")
        if any(not isinstance(row, (list, tuple)) or len(row) != ranks[degree]
               for row in raw):
            raise ValueError("boundary column count must match the chain rank")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for row in raw for value in row):
            raise ValueError("boundary entries must be exact integers")
        if any(abs(value).bit_length() > settings.max_topology_entry_bits
               for row in raw for value in row):
            raise ValueError("boundary entry exceeds max_topology_entry_bits")
        entries += len(raw) * ranks[degree]
        matrices.append(tuple(tuple(value for value in row) for row in raw))
    if entries > settings.max_topology_matrix_entries:
        raise ValueError("boundary matrices exceed max_topology_matrix_entries")
    labels = data.get("basis_labels", ())
    if labels and (not isinstance(labels, (list, tuple)) or any(
            not isinstance(layer, (list, tuple)) or
            any(not isinstance(label, str) or not label for label in layer)
            for layer in labels)):
        raise ValueError("basis_labels must be nonempty strings grouped by degree")
    return typ, ChainComplex(ranks=tuple(ranks), boundaries=tuple(matrices),
        basis_labels=tuple(tuple(layer) for layer in labels)), TrustLevel.EXACT, []


def _chain(value):
    from .algebraic_topology import (ChainComplex, cubical_chain_complex,
                                     simplicial_chain_complex)
    if isinstance(value, ChainComplex):
        return value
    if type(value).__name__ == "SimplicialComplex":
        return simplicial_chain_complex(value)
    return cubical_chain_complex(value)


def apply(kernel, value, operation, parameters, object_id):
    from . import algebraic_topology as topology

    if type(value).__name__ == "SimplicialComplex":
        ranks = tuple(len(layer) for layer in value.simplices)
    elif type(value).__name__ == "CubicalComplex":
        ranks = tuple(len(layer) for layer in value.cells)
    else:
        ranks = value.ranks
    total_cells = sum(ranks)
    matrix_entries = sum(ranks[k - 1] * ranks[k]
                         for k in range(1, len(ranks)))
    if matrix_entries > kernel.settings.max_topology_matrix_entries:
        raise ValueError("boundary matrices exceed max_topology_matrix_entries")
    composition_work = max((ranks[k - 2] * ranks[k - 1] * ranks[k]
                            for k in range(2, len(ranks))), default=0)
    work = max(1, total_cells, matrix_entries, composition_work)
    if operation == "homology":
        work = max(work, sum(rank ** 3 for rank in ranks))
        coefficient = parameters.get("coefficient", "Z")
        if str(coefficient).upper().replace(" ", "") == "Z" and any(
                rank > kernel.settings.max_normal_form_dim for rank in ranks):
            raise ValueError("integer homology exceeds max_normal_form_dim")
    if work > kernel.settings.max_topology_work:
        raise ValueError(f"{operation} exceeds max_topology_work")
    chain = _chain(value)
    if operation == "verify":
        if type(value).__name__ == "SimplicialComplex":
            return topology.verify_simplicial_complex(value)
        if type(value).__name__ == "CubicalComplex":
            return topology.verify_cubical_complex(value)
        return topology.verify_chain_complex(value)
    if operation == "chain_complex":
        checks = {"boundary_squared_zero": all(
            topology.verify_chain_complex(chain).verification.values())}
        return checked_result("chain_complex", {"ranks": chain.ranks},
            method="oriented_cellular_boundary", trust="exact", checks=checks,
            witness={"boundaries": chain.boundaries},
            details={"identity_checks": checks}), chain
    if operation == "boundary_matrix":
        return topology.boundary_matrix(chain, parameters.get("degree"))
    if operation == "homology":
        return topology.homology(chain, parameters.get("coefficient", "Z"),
                                 parameters.get("prime"), parameters.get("degree"))
    if operation == "euler_characteristic":
        return topology.euler_characteristic(chain)
    raise NotImplementedError(f"unsupported algebraic topology operation {operation}")
