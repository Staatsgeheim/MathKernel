# =============================================================================
# MathKernel - Exact branching Markov tensors and observation-rank transfer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact finite branching-Markov tensors, flattenings, and observation transfer.

The module provides a finite, auditable realization of four related facts.

1. A general Markov process on a rooted tree produces a leaf tensor by a
   finite sum-product contraction over the hidden internal states.
2. Arbitrary stochastic leaf observations act locally: an output function is
   pulled back through its observation channel, after which the universal
   observation-transfer contraction applies without modification.
3. If an edge separates leaves ``A|B`` and the hidden state at the boundary
   has cardinality ``k``, the corresponding leaf flattening factors through a
   ``k``-dimensional space and therefore has rank at most ``k``.
4. Local observation channels transform a flattening as ``L F R^T``.  Such
   channels cannot increase rank; full-latent-rank channels preserve it and
   admit exact left-inverse recovery.  Rank-deficient channels have explicit
   distribution-collision witnesses.

All probabilistic and algebraic identities are evaluated with
:class:`fractions.Fraction`.  Numerical singular-value diagnostics are kept
separate and use NumPy only after the exact matrix identity has been checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from itertools import combinations, product
from math import prod
from typing import Iterable, Mapping, Sequence, TypeVar

import numpy as np

from .cumulants import connected_statistic
from .stochastic_koopman import FiniteJointLaw, arbitrary_joint_contraction

Scalar = TypeVar("Scalar")
RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _as_fraction(value: Fraction | int | str | float) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _sum_terms(values: Iterable[Scalar], zero: Scalar | Fraction = Fraction(0)):
    result = None
    for value in values:
        result = value if result is None else result + value
    return zero if result is None else result


def _product_values(values: Iterable[Scalar]):
    result = None
    for value in values:
        result = value if result is None else result * value
    return Fraction(1) if result is None else result


def _normalize_matrix(
    matrix: Sequence[Sequence[Fraction | int | str | float]],
    *,
    name: str = "matrix",
) -> RationalMatrix:
    rows = tuple(tuple(_as_fraction(value) for value in row) for row in matrix)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be nonempty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must not be ragged")
    return rows


def matrix_transpose(matrix: Sequence[Sequence[Fraction]]) -> RationalMatrix:
    rows = _normalize_matrix(matrix)
    return tuple(tuple(rows[i][j] for i in range(len(rows))) for j in range(len(rows[0])))


def matrix_multiply(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> RationalMatrix:
    a = _normalize_matrix(left, name="left matrix")
    b = _normalize_matrix(right, name="right matrix")
    if len(a[0]) != len(b):
        raise ValueError("matrix dimensions do not agree")
    return tuple(
        tuple(
            sum((a[i][k] * b[k][j] for k in range(len(b))), Fraction(0))
            for j in range(len(b[0]))
        )
        for i in range(len(a))
    )


def identity_matrix(size: int) -> RationalMatrix:
    if size < 1:
        raise ValueError("identity size must be positive")
    return tuple(
        tuple(Fraction(int(i == j), 1) for j in range(size)) for i in range(size)
    )


def exact_matrix_rank(matrix: Sequence[Sequence[Fraction | int | str | float]]) -> int:
    """Return matrix rank by exact rational row reduction."""

    work = [list(row) for row in _normalize_matrix(matrix)]
    n_rows = len(work)
    n_cols = len(work[0])
    rank = 0
    for column in range(n_cols):
        pivot = next((row for row in range(rank, n_rows) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for row in range(n_rows):
            if row == rank:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    work[row][j] - factor * work[rank][j] for j in range(n_cols)
                ]
        rank += 1
        if rank == n_rows:
            break
    return rank


def inverse_matrix(matrix: Sequence[Sequence[Fraction | int | str | float]]) -> RationalMatrix:
    """Invert a square rational matrix exactly."""

    source = _normalize_matrix(matrix)
    n = len(source)
    if len(source[0]) != n:
        raise ValueError("matrix must be square")
    work = [list(source[i]) + list(identity_matrix(n)[i]) for i in range(n)]
    for column in range(n):
        pivot = next((row for row in range(column, n) if work[row][column]), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        work[column], work[pivot] = work[pivot], work[column]
        pivot_value = work[column][column]
        work[column] = [value / pivot_value for value in work[column]]
        for row in range(n):
            if row == column:
                continue
            factor = work[row][column]
            if factor:
                work[row] = [
                    work[row][j] - factor * work[column][j] for j in range(2 * n)
                ]
    return tuple(tuple(work[i][n:]) for i in range(n))


def kronecker_product(
    *matrices: Sequence[Sequence[Fraction | int | str | float]],
) -> RationalMatrix:
    """Exact Kronecker product, returning ``[[1]]`` for an empty family."""

    result: RationalMatrix = ((Fraction(1),),)
    for raw in matrices:
        matrix = _normalize_matrix(raw)
        result = tuple(
            tuple(
                result[i][j] * matrix[k][ell]
                for j in range(len(result[0]))
                for ell in range(len(matrix[0]))
            )
            for i in range(len(result))
            for k in range(len(matrix))
        )
    return result


def _mixed_radix_index(values: Sequence[int], sizes: Sequence[int]) -> int:
    if len(values) != len(sizes):
        raise ValueError("values and sizes must have equal length")
    index = 0
    for value, size in zip(values, sizes, strict=True):
        if not 0 <= value < size:
            raise ValueError("mixed-radix digit out of range")
        index = index * size + value
    return index


def _dense_law_vector(law: FiniteJointLaw) -> tuple[Fraction, ...]:
    return tuple(
        law.probabilities.get(state, Fraction(0))
        for state in product(*(range(size) for size in law.state_sizes))
    )


def _matrix_times_vector(
    matrix: Sequence[Sequence[Fraction]], vector: Sequence[Fraction]
) -> tuple[Fraction, ...]:
    source = _normalize_matrix(matrix)
    if len(source[0]) != len(vector):
        raise ValueError("matrix and vector dimensions do not agree")
    return tuple(
        sum((source[i][j] * vector[j] for j in range(len(vector))), Fraction(0))
        for i in range(len(source))
    )


@dataclass(frozen=True, slots=True)
class FiniteObservationChannel:
    """Row-stochastic channel from latent states to observed states."""

    conditional: RationalMatrix
    name: str = "observation"

    def __post_init__(self) -> None:
        rows = _normalize_matrix(self.conditional, name="observation channel")
        for row in rows:
            if any(value < 0 for value in row) or sum(row, Fraction(0)) != 1:
                raise ValueError("every observation-channel row must be a probability vector")
        object.__setattr__(self, "conditional", rows)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Sequence[Fraction | int | str | float]],
        *,
        name: str = "observation",
    ) -> "FiniteObservationChannel":
        return cls(_normalize_matrix(rows, name="observation channel"), name=name)

    @classmethod
    def deterministic(
        cls,
        mapping: Sequence[int],
        *,
        output_states: int | None = None,
        name: str = "deterministic observation",
    ) -> "FiniteObservationChannel":
        targets = tuple(int(value) for value in mapping)
        if not targets or any(value < 0 for value in targets):
            raise ValueError("mapping must contain nonnegative output labels")
        m = max(targets) + 1 if output_states is None else int(output_states)
        if m < 1 or any(value >= m for value in targets):
            raise ValueError("output state count does not cover the mapping")
        return cls.from_rows(
            [
                [Fraction(int(column == target), 1) for column in range(m)]
                for target in targets
            ],
            name=name,
        )

    @property
    def latent_states(self) -> int:
        return len(self.conditional)

    @property
    def output_states(self) -> int:
        return len(self.conditional[0])

    @property
    def operator(self) -> RationalMatrix:
        """Column-vector operator ``M^T`` mapping latent to observed masses."""

        return matrix_transpose(self.conditional)

    @property
    def rank(self) -> int:
        return exact_matrix_rank(self.conditional)

    @property
    def has_full_latent_rank(self) -> bool:
        return self.rank == self.latent_states

    def pullback(self, output_function: Sequence[Scalar]) -> tuple[Scalar, ...]:
        """Return ``x -> E[f(Y)|X=x]``."""

        if len(output_function) != self.output_states:
            raise ValueError("output function has the wrong cardinality")
        return tuple(
            _sum_terms(
                (
                    self.conditional[x][y] * output_function[y]
                    for y in range(self.output_states)
                    if self.conditional[x][y]
                ),
                Fraction(0),
            )
            for x in range(self.latent_states)
        )

    def left_inverse_operator(self) -> RationalMatrix:
        """Exact left inverse of ``operator`` when latent information is injective."""

        if not self.has_full_latent_rank:
            raise ValueError("observation channel is not injective on latent distributions")
        operator = self.operator
        transpose = matrix_transpose(operator)
        gram = matrix_multiply(transpose, operator)
        left_inverse = matrix_multiply(inverse_matrix(gram), transpose)
        if matrix_multiply(left_inverse, operator) != identity_matrix(self.latent_states):
            raise RuntimeError("internal error constructing channel left inverse")
        return left_inverse

    @classmethod
    def jointly_conditionally_independent(
        cls,
        channels: Sequence["FiniteObservationChannel"],
        *,
        name: str = "joint observation",
    ) -> "FiniteObservationChannel":
        """Combine sensors observed independently conditional on one latent state."""

        items = tuple(channels)
        if not items:
            raise ValueError("at least one channel is required")
        k = items[0].latent_states
        if any(channel.latent_states != k for channel in items):
            raise ValueError("all channels must share the same latent alphabet")
        output_ranges = tuple(range(channel.output_states) for channel in items)
        rows = []
        for state in range(k):
            rows.append(
                tuple(
                    _product_values(
                        items[j].conditional[state][output[j]]
                        for j in range(len(items))
                    )
                    for output in product(*output_ranges)
                )
            )
        return cls.from_rows(rows, name=name)

    def collision_witness(self) -> dict[str, object]:
        """Give distinct latent distributions with the same observed law.

        A witness exists exactly when the channel lacks full latent rank.
        """

        if self.has_full_latent_rank:
            raise ValueError("full-latent-rank channel has no distribution collision")
        matrix = [list(row) for row in self.operator]
        n_rows = len(matrix)
        n_cols = len(matrix[0])
        pivot_columns: list[int] = []
        pivot_row = 0
        for column in range(n_cols):
            pivot = next(
                (row for row in range(pivot_row, n_rows) if matrix[row][column]),
                None,
            )
            if pivot is None:
                continue
            matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
            value = matrix[pivot_row][column]
            matrix[pivot_row] = [entry / value for entry in matrix[pivot_row]]
            for row in range(n_rows):
                if row == pivot_row:
                    continue
                factor = matrix[row][column]
                if factor:
                    matrix[row] = [
                        matrix[row][j] - factor * matrix[pivot_row][j]
                        for j in range(n_cols)
                    ]
            pivot_columns.append(column)
            pivot_row += 1
            if pivot_row == n_rows:
                break
        free_columns = [column for column in range(n_cols) if column not in pivot_columns]
        if not free_columns:
            raise RuntimeError("rank calculation and nullspace calculation disagree")
        free = free_columns[0]
        direction = [Fraction(0)] * n_cols
        direction[free] = Fraction(1)
        for row, column in enumerate(pivot_columns):
            direction[column] = -matrix[row][free]
        if all(value == 0 for value in direction):
            raise RuntimeError("internal nullspace construction failed")
        base = [Fraction(1, n_cols)] * n_cols
        positive_limit = min(
            (base[i] / -direction[i] for i in range(n_cols) if direction[i] < 0),
            default=None,
        )
        if positive_limit is None:
            direction = [-value for value in direction]
            positive_limit = min(
                base[i] / -direction[i]
                for i in range(n_cols)
                if direction[i] < 0
            )
        epsilon = positive_limit / 2
        alternative = tuple(base[i] + epsilon * direction[i] for i in range(n_cols))
        base_tuple = tuple(base)
        observed_base = tuple(
            sum(
                (base_tuple[x] * self.conditional[x][y] for x in range(n_cols)),
                Fraction(0),
            )
            for y in range(self.output_states)
        )
        observed_alternative = tuple(
            sum(
                (alternative[x] * self.conditional[x][y] for x in range(n_cols)),
                Fraction(0),
            )
            for y in range(self.output_states)
        )
        return {
            "first": base_tuple,
            "second": alternative,
            "null_direction": tuple(direction),
            "observed_first": observed_base,
            "observed_second": observed_alternative,
            "distinct": base_tuple != alternative,
            "collision_exact": observed_base == observed_alternative,
        }


@dataclass(frozen=True, slots=True)
class MarkovTreeEdge:
    parent: int
    child: int
    transition: RationalMatrix

    @classmethod
    def make(
        cls,
        parent: int,
        child: int,
        transition: Sequence[Sequence[Fraction | int | str | float]],
    ) -> "MarkovTreeEdge":
        return cls(int(parent), int(child), _normalize_matrix(transition, name="edge transition"))


@dataclass(frozen=True)
class FiniteMarkovTree:
    """Exact rooted finite-state general Markov model on a tree.

    Vertices are numbered ``0 .. n-1``.  ``transitions[v]`` is the row-stochastic
    matrix on the edge ``parent[v] -> v`` and is ``None`` at the root.
    State cardinalities may differ between vertices.
    """

    state_sizes: tuple[int, ...]
    root: int
    root_distribution: tuple[Fraction, ...]
    parent: tuple[int | None, ...]
    transitions: tuple[RationalMatrix | None, ...]
    leaf_order: tuple[int, ...]
    name: str = "finite Markov tree"
    _children: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)
    _topological_order: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _leaf_positions: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.state_sizes)
        n = len(sizes)
        if n == 0 or any(size < 1 for size in sizes):
            raise ValueError("state_sizes must contain positive cardinalities")
        root = int(self.root)
        if not 0 <= root < n:
            raise ValueError("root is out of range")
        parents = tuple(None if value is None else int(value) for value in self.parent)
        if len(parents) != n or parents[root] is not None:
            raise ValueError("parent must have one entry per vertex and None at the root")
        if any(parents[v] is None for v in range(n) if v != root):
            raise ValueError("every non-root vertex must have one parent")
        if any(
            parents[v] is not None and (not 0 <= parents[v] < n or parents[v] == v)
            for v in range(n)
        ):
            raise ValueError("parent index is invalid")
        children: list[list[int]] = [[] for _ in range(n)]
        for child, parent in enumerate(parents):
            if parent is not None:
                children[parent].append(child)
        order: list[int] = []
        stack = [root]
        seen: set[int] = set()
        while stack:
            vertex = stack.pop()
            if vertex in seen:
                raise ValueError("parent relation contains a cycle")
            seen.add(vertex)
            order.append(vertex)
            stack.extend(reversed(children[vertex]))
        if len(seen) != n:
            raise ValueError("parent relation is not a connected rooted tree")

        root_distribution = tuple(_as_fraction(value) for value in self.root_distribution)
        if len(root_distribution) != sizes[root]:
            raise ValueError("root distribution has the wrong cardinality")
        if any(value < 0 for value in root_distribution) or sum(
            root_distribution, Fraction(0)
        ) != 1:
            raise ValueError("root distribution must be nonnegative and sum to one")

        transitions = tuple(self.transitions)
        if len(transitions) != n or transitions[root] is not None:
            raise ValueError("transitions must have one entry per vertex and None at root")
        normalized_transitions: list[RationalMatrix | None] = [None] * n
        for child in range(n):
            if child == root:
                continue
            matrix = _normalize_matrix(transitions[child], name=f"transition to {child}")  # type: ignore[arg-type]
            parent = parents[child]
            assert parent is not None
            if len(matrix) != sizes[parent] or len(matrix[0]) != sizes[child]:
                raise ValueError(f"transition to {child} has incompatible shape")
            for row in matrix:
                if any(value < 0 for value in row) or sum(row, Fraction(0)) != 1:
                    raise ValueError(f"transition to {child} is not row stochastic")
            normalized_transitions[child] = matrix

        actual_leaves = {vertex for vertex in range(n) if not children[vertex]}
        leaves = tuple(int(vertex) for vertex in self.leaf_order)
        if len(set(leaves)) != len(leaves) or set(leaves) != actual_leaves:
            raise ValueError("leaf_order must list every and only leaf exactly once")

        object.__setattr__(self, "state_sizes", sizes)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "root_distribution", root_distribution)
        object.__setattr__(self, "parent", parents)
        object.__setattr__(self, "transitions", tuple(normalized_transitions))
        object.__setattr__(self, "leaf_order", leaves)
        object.__setattr__(self, "_children", tuple(tuple(row) for row in children))
        object.__setattr__(self, "_topological_order", tuple(order))
        object.__setattr__(self, "_leaf_positions", {leaf: i for i, leaf in enumerate(leaves)})

    @classmethod
    def from_edges(
        cls,
        state_sizes: Sequence[int],
        *,
        root: int,
        root_distribution: Sequence[Fraction | int | str | float],
        edges: Sequence[
            MarkovTreeEdge
            | tuple[int, int, Sequence[Sequence[Fraction | int | str | float]]]
        ],
        leaf_order: Sequence[int] | None = None,
        name: str = "finite Markov tree",
    ) -> "FiniteMarkovTree":
        sizes = tuple(int(size) for size in state_sizes)
        n = len(sizes)
        parents: list[int | None] = [None] * n
        transitions: list[RationalMatrix | None] = [None] * n
        for raw in edges:
            edge = raw if isinstance(raw, MarkovTreeEdge) else MarkovTreeEdge.make(*raw)
            if not 0 <= edge.parent < n or not 0 <= edge.child < n:
                raise ValueError("edge endpoint out of range")
            if edge.child == root:
                raise ValueError("root cannot have an incoming edge")
            if parents[edge.child] is not None:
                raise ValueError("a vertex cannot have multiple parents")
            parents[edge.child] = edge.parent
            transitions[edge.child] = edge.transition
        if leaf_order is None:
            parent_vertices = {
                edge[0] if not isinstance(edge, MarkovTreeEdge) else edge.parent for edge in edges
            }
            leaf_order = tuple(vertex for vertex in range(n) if vertex not in parent_vertices)
        return cls(
            sizes,
            int(root),
            tuple(_as_fraction(value) for value in root_distribution),
            tuple(parents),
            tuple(transitions),
            tuple(leaf_order),
            name=name,
        )

    @property
    def n_vertices(self) -> int:
        return len(self.state_sizes)

    @property
    def leaves(self) -> tuple[int, ...]:
        return self.leaf_order

    @property
    def topological_order(self) -> tuple[int, ...]:
        return self._topological_order

    def children(self, vertex: int) -> tuple[int, ...]:
        if not 0 <= vertex < self.n_vertices:
            raise ValueError("vertex out of range")
        return self._children[vertex]

    @cached_property
    def descendant_leaves(self) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = [tuple() for _ in range(self.n_vertices)]
        for vertex in reversed(self.topological_order):
            if vertex in self._leaf_positions:
                result[vertex] = (vertex,)
            else:
                leaf_set = {
                    leaf for child in self._children[vertex] for leaf in result[child]
                }
                result[vertex] = tuple(leaf for leaf in self.leaf_order if leaf in leaf_set)
        return tuple(result)

    @cached_property
    def _conditional_tables(
        self,
    ) -> tuple[tuple[dict[tuple[int, ...], Fraction], ...], ...]:
        """Conditional descendant-leaf pattern laws for every vertex state."""

        tables: list[tuple[dict[tuple[int, ...], Fraction], ...] | None] = [
            None
        ] * self.n_vertices
        global_position = self._leaf_positions
        for vertex in reversed(self.topological_order):
            if vertex in global_position:
                tables[vertex] = tuple(
                    {(state,): Fraction(1)} for state in range(self.state_sizes[vertex])
                )
                continue
            node_leaves = self.descendant_leaves[vertex]
            node_positions = {leaf: i for i, leaf in enumerate(node_leaves)}
            state_tables: list[dict[tuple[int, ...], Fraction]] = []
            for parent_state in range(self.state_sizes[vertex]):
                combined: dict[tuple[int, ...], Fraction] = {
                    (-1,) * len(node_leaves): Fraction(1)
                }
                for child in self._children[vertex]:
                    transition = self.transitions[child]
                    child_tables = tables[child]
                    assert transition is not None and child_tables is not None
                    child_leaves = self.descendant_leaves[child]
                    child_mixture: dict[tuple[int, ...], Fraction] = {}
                    for child_state in range(self.state_sizes[child]):
                        edge_probability = transition[parent_state][child_state]
                        if not edge_probability:
                            continue
                        for pattern, probability in child_tables[child_state].items():
                            child_mixture[pattern] = (
                                child_mixture.get(pattern, Fraction(0))
                                + edge_probability * probability
                            )
                    updated: dict[tuple[int, ...], Fraction] = {}
                    child_slots = tuple(node_positions[leaf] for leaf in child_leaves)
                    for partial, partial_probability in combined.items():
                        for child_pattern, child_probability in child_mixture.items():
                            merged = list(partial)
                            for slot, value in zip(child_slots, child_pattern, strict=True):
                                merged[slot] = value
                            key = tuple(merged)
                            updated[key] = updated.get(key, Fraction(0)) + (
                                partial_probability * child_probability
                            )
                    combined = updated
                if any(any(value < 0 for value in pattern) for pattern in combined):
                    raise RuntimeError("internal error merging subtree patterns")
                if sum(combined.values(), Fraction(0)) != 1:
                    raise RuntimeError("conditional subtree law is not normalized")
                state_tables.append(combined)
            tables[vertex] = tuple(state_tables)
        return tuple(table for table in tables if table is not None)

    def conditional_descendant_law(self, vertex: int, state: int) -> FiniteJointLaw:
        if not 0 <= vertex < self.n_vertices:
            raise ValueError("vertex out of range")
        if not 0 <= state < self.state_sizes[vertex]:
            raise ValueError("state out of range")
        leaves = self.descendant_leaves[vertex]
        return FiniteJointLaw(
            tuple(self.state_sizes[leaf] for leaf in leaves),
            dict(self._conditional_tables[vertex][state]),
        )

    def leaf_joint_law(self) -> FiniteJointLaw:
        masses: dict[tuple[int, ...], Fraction] = {}
        root_tables = self._conditional_tables[self.root]
        for root_state, root_probability in enumerate(self.root_distribution):
            if not root_probability:
                continue
            for pattern, conditional in root_tables[root_state].items():
                masses[pattern] = masses.get(pattern, Fraction(0)) + (
                    root_probability * conditional
                )
        return FiniteJointLaw(
            tuple(self.state_sizes[leaf] for leaf in self.leaf_order), masses
        )

    def joint_law(
        self,
        vertices: Sequence[int],
        *,
        max_assignments: int = 2_000_000,
    ) -> FiniteJointLaw:
        """Exact marginal on arbitrary vertices by full hidden-state enumeration."""

        requested = tuple(int(vertex) for vertex in vertices)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("vertices must be a nonempty unique sequence")
        if any(not 0 <= vertex < self.n_vertices for vertex in requested):
            raise ValueError("requested vertex out of range")
        blank = (-1,) * self.n_vertices
        dynamic: dict[tuple[int, ...], Fraction] = {}
        for state, probability in enumerate(self.root_distribution):
            if probability:
                assignment = list(blank)
                assignment[self.root] = state
                dynamic[tuple(assignment)] = probability
        for child in self.topological_order[1:]:
            parent = self.parent[child]
            transition = self.transitions[child]
            assert parent is not None and transition is not None
            updated: dict[tuple[int, ...], Fraction] = {}
            for assignment, probability in dynamic.items():
                parent_state = assignment[parent]
                for child_state, edge_probability in enumerate(transition[parent_state]):
                    if not edge_probability:
                        continue
                    target = list(assignment)
                    target[child] = child_state
                    key = tuple(target)
                    updated[key] = updated.get(key, Fraction(0)) + (
                        probability * edge_probability
                    )
            if len(updated) > max_assignments:
                raise ValueError("hidden assignment budget exceeded")
            dynamic = updated
        masses: dict[tuple[int, ...], Fraction] = {}
        for assignment, probability in dynamic.items():
            pattern = tuple(assignment[vertex] for vertex in requested)
            masses[pattern] = masses.get(pattern, Fraction(0)) + probability
        return FiniteJointLaw(tuple(self.state_sizes[v] for v in requested), masses)

    def leaf_moment(
        self, variables: Sequence[tuple[int, Sequence[Scalar]]]
    ):
        """Exact product moment by tree sum-product, without materializing the leaf law."""

        local: dict[int, tuple[Scalar, ...]] = {}
        for leaf, function in variables:
            leaf = int(leaf)
            if leaf not in self._leaf_positions:
                raise ValueError("all variables must be attached to leaves")
            values = tuple(function)
            if len(values) != self.state_sizes[leaf]:
                raise ValueError("leaf function has the wrong state cardinality")
            if leaf in local:
                local[leaf] = tuple(
                    local[leaf][state] * values[state]
                    for state in range(self.state_sizes[leaf])
                )
            else:
                local[leaf] = values

        messages: list[tuple[Scalar, ...] | None] = [None] * self.n_vertices
        for vertex in reversed(self.topological_order):
            values: list[Scalar] = []
            for state in range(self.state_sizes[vertex]):
                value = (
                    local[vertex][state]
                    if vertex in local
                    else Fraction(1)
                )
                for child in self._children[vertex]:
                    transition = self.transitions[child]
                    child_message = messages[child]
                    assert transition is not None and child_message is not None
                    child_expectation = _sum_terms(
                        (
                            transition[state][child_state]
                            * child_message[child_state]
                            for child_state in range(self.state_sizes[child])
                            if transition[state][child_state]
                        ),
                        Fraction(0),
                    )
                    value = value * child_expectation
                values.append(value)
            messages[vertex] = tuple(values)
        root_message = messages[self.root]
        assert root_message is not None
        return _sum_terms(
            (
                self.root_distribution[state] * root_message[state]
                for state in range(self.state_sizes[self.root])
                if self.root_distribution[state]
            ),
            Fraction(0),
        )

    def leaf_cumulant(
        self, variables: Sequence[tuple[int, Sequence[Scalar]]]
    ):
        items = tuple(variables)
        if not items:
            raise ValueError("at least one leaf variable is required")

        def raw(block: tuple[int, ...]):
            if not block:
                return Fraction(1)
            return self.leaf_moment([items[index] for index in block])

        return connected_statistic(raw, len(items))

    def observed_leaf_moment(
        self,
        variables: Sequence[
            tuple[int, FiniteObservationChannel, Sequence[Scalar]]
        ],
    ):
        leaves = [int(leaf) for leaf, _, _ in variables]
        if len(set(leaves)) != len(leaves):
            raise ValueError("observed variables must use distinct leaves")
        pulled = []
        for leaf, channel, function in variables:
            if channel.latent_states != self.state_sizes[leaf]:
                raise ValueError("channel latent alphabet and leaf state space disagree")
            pulled.append((leaf, channel.pullback(function)))
        return self.leaf_moment(pulled)

    def observed_leaf_cumulant(
        self,
        variables: Sequence[
            tuple[int, FiniteObservationChannel, Sequence[Scalar]]
        ],
    ):
        items = tuple(variables)
        leaves = [int(leaf) for leaf, _, _ in items]
        if not items or len(set(leaves)) != len(leaves):
            raise ValueError("observed cumulant requires distinct leaf variables")

        def raw(block: tuple[int, ...]):
            if not block:
                return Fraction(1)
            return self.observed_leaf_moment([items[index] for index in block])

        return connected_statistic(raw, len(items))


@dataclass(frozen=True, slots=True)
class TensorFlattening:
    left_coordinates: tuple[int, ...]
    right_coordinates: tuple[int, ...]
    left_sizes: tuple[int, ...]
    right_sizes: tuple[int, ...]
    matrix: RationalMatrix

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.matrix), len(self.matrix[0])

    @property
    def exact_rank(self) -> int:
        return exact_matrix_rank(self.matrix)

    def singular_values(self) -> tuple[float, ...]:
        values = np.asarray(
            [[float(value) for value in row] for row in self.matrix], dtype=float
        )
        return tuple(float(value) for value in np.linalg.svd(values, compute_uv=False))

    def tail_frobenius(self, rank_bound: int) -> float:
        if rank_bound < 0:
            raise ValueError("rank bound must be nonnegative")
        singular = self.singular_values()
        return float(np.sqrt(sum(value * value for value in singular[rank_bound:])))


def tensor_flattening(
    law: FiniteJointLaw, left_coordinates: Sequence[int]
) -> TensorFlattening:
    left = tuple(int(index) for index in left_coordinates)
    if not left or len(set(left)) != len(left):
        raise ValueError("left coordinates must be a nonempty unique sequence")
    if any(not 0 <= index < law.dimension for index in left):
        raise ValueError("flattening coordinate out of range")
    right = tuple(index for index in range(law.dimension) if index not in set(left))
    if not right:
        raise ValueError("flattening requires a nonempty right side")
    left_sizes = tuple(law.state_sizes[index] for index in left)
    right_sizes = tuple(law.state_sizes[index] for index in right)
    matrix = [
        [Fraction(0) for _ in range(prod(right_sizes))]
        for _ in range(prod(left_sizes))
    ]
    for state, probability in law.probabilities.items():
        row = _mixed_radix_index(tuple(state[index] for index in left), left_sizes)
        column = _mixed_radix_index(tuple(state[index] for index in right), right_sizes)
        matrix[row][column] += probability
    return TensorFlattening(
        left,
        right,
        left_sizes,
        right_sizes,
        tuple(tuple(row) for row in matrix),
    )


def transform_joint_law(
    law: FiniteJointLaw,
    channels: Sequence[FiniteObservationChannel],
    *,
    max_terms: int = 5_000_000,
) -> FiniteJointLaw:
    """Push a joint law through independent local observation channels."""

    items = tuple(channels)
    if len(items) != law.dimension:
        raise ValueError("one observation channel is required per coordinate")
    for index, channel in enumerate(items):
        if channel.latent_states != law.state_sizes[index]:
            raise ValueError("channel latent state size does not match joint law")
    output_sizes = tuple(channel.output_states for channel in items)
    if prod(output_sizes) * max(1, len(law.probabilities)) > max_terms:
        raise ValueError("observation transform budget exceeded")
    masses: dict[tuple[int, ...], Fraction] = {}
    output_ranges = tuple(range(size) for size in output_sizes)
    for latent, latent_probability in law.probabilities.items():
        for output in product(*output_ranges):
            conditional = _product_values(
                items[j].conditional[latent[j]][output[j]]
                for j in range(law.dimension)
            )
            if conditional:
                masses[output] = masses.get(output, Fraction(0)) + (
                    latent_probability * conditional
                )
    return FiniteJointLaw(output_sizes, masses)


def branching_observation_contraction(
    tree: FiniteMarkovTree,
    variables: Sequence[
        tuple[int, FiniteObservationChannel, Sequence[Fraction | int | str | float]]
    ],
    bases: Sequence[
        Sequence[Sequence[Fraction | int | str | float]]
    ],
    *,
    connected: bool = True,
):
    """Tree-level observation-transfer contraction for selected leaves."""

    items = tuple(variables)
    leaves = tuple(int(leaf) for leaf, _, _ in items)
    if not items or len(set(leaves)) != len(leaves) or len(bases) != len(items):
        raise ValueError("variables must be nonempty, use distinct leaves, and match bases")
    leaf_law = tree.leaf_joint_law()
    positions = tuple(tree._leaf_positions[leaf] for leaf in leaves)
    marginal = leaf_law.marginal(positions)
    pulled = []
    for index, (leaf, channel, output_function) in enumerate(items):
        if channel.latent_states != tree.state_sizes[leaf]:
            raise ValueError("channel and leaf state size disagree")
        values = channel.pullback(output_function)
        if len(bases[index]) != len(values):
            raise ValueError("basis cardinality does not match latent leaf")
        pulled.append(values)
    return arbitrary_joint_contraction(
        marginal, pulled, bases, connected=connected
    )


@dataclass(frozen=True, slots=True)
class EdgeFlatteningCertificate:
    parent: int
    child: int
    left_leaves: tuple[int, ...]
    right_leaves: tuple[int, ...]
    flattening: TensorFlattening
    parent_factor: RationalMatrix
    edge_transition: RationalMatrix
    child_factor: RationalMatrix
    three_factor_reconstructed: RationalMatrix
    left_factor: RationalMatrix
    right_factor: RationalMatrix
    reconstructed: RationalMatrix
    parent_state_size: int
    boundary_state_size: int
    transition_rank: int
    sharp_rank_bound: int
    flattening_rank: int
    parent_factor_rank: int
    child_factor_rank: int
    left_factor_rank: int
    right_factor_rank: int
    three_factorization_exact: bool
    factorization_exact: bool
    rank_bound_holds: bool


def edge_flattening_certificate(
    tree: FiniteMarkovTree,
    child: int,
    *,
    max_assignments: int = 2_000_000,
) -> EdgeFlatteningCertificate:
    """Certify the exact edge factorization and its sharp finite-state bound.

    For the directed edge ``parent -> child`` and induced leaf split ``A|B``,
    the flattening admits both

    ``Flat(A|B) = P(A, X_child) P(B | X_child)``

    and the more informative three-factor representation

    ``Flat(A|B) = P(A, X_parent) M_edge P(B | X_child)``.

    Consequently its rank is bounded by the rank of the edge transition and,
    in particular, by both endpoint state cardinalities.  In the homogeneous
    ``k``-state general Markov model this recovers the familiar ``rank <= k``
    edge-flattening constraint.
    """

    child = int(child)
    if not 0 <= child < tree.n_vertices or child == tree.root:
        raise ValueError("child must identify a non-root edge endpoint")
    right_leaves = tree.descendant_leaves[child]
    right_set = set(right_leaves)
    left_leaves = tuple(leaf for leaf in tree.leaf_order if leaf not in right_set)
    if not left_leaves or not right_leaves:
        raise ValueError("edge does not induce a nontrivial leaf split")
    leaf_law = tree.leaf_joint_law()
    left_coordinates = tuple(tree._leaf_positions[leaf] for leaf in left_leaves)
    flattening = tensor_flattening(leaf_law, left_coordinates)

    parent = tree.parent[child]
    transition = tree.transitions[child]
    assert parent is not None and transition is not None

    # P(A, X_parent).  The outside leaves A and the descendant subtree B are
    # conditionally independent once the edge is cut and the parent state is
    # fixed, with M_edge carrying the state transfer across the cut.
    parent_joint = tree.joint_law(
        (*left_leaves, parent), max_assignments=max_assignments
    )
    left_sizes = tuple(tree.state_sizes[leaf] for leaf in left_leaves)
    parent_factor = [
        [Fraction(0) for _ in range(tree.state_sizes[parent])]
        for _ in range(prod(left_sizes))
    ]
    for state, probability in parent_joint.probabilities.items():
        row = _mixed_radix_index(state[:-1], left_sizes)
        parent_factor[row][state[-1]] += probability

    joint = tree.joint_law((*left_leaves, child), max_assignments=max_assignments)
    left_factor = [
        [Fraction(0) for _ in range(tree.state_sizes[child])]
        for _ in range(prod(left_sizes))
    ]
    for state, probability in joint.probabilities.items():
        row = _mixed_radix_index(state[:-1], left_sizes)
        left_factor[row][state[-1]] += probability

    right_sizes = tuple(tree.state_sizes[leaf] for leaf in right_leaves)
    child_factor = [
        [Fraction(0) for _ in range(prod(right_sizes))]
        for _ in range(tree.state_sizes[child])
    ]
    for boundary_state in range(tree.state_sizes[child]):
        conditional = tree.conditional_descendant_law(child, boundary_state)
        for pattern, probability in conditional.probabilities.items():
            column = _mixed_radix_index(pattern, right_sizes)
            child_factor[boundary_state][column] += probability

    parent_matrix = tuple(tuple(row) for row in parent_factor)
    left_matrix = tuple(tuple(row) for row in left_factor)
    child_matrix = tuple(tuple(row) for row in child_factor)
    right_matrix = child_matrix
    three_factor_reconstructed = matrix_multiply(
        matrix_multiply(parent_matrix, transition), child_matrix
    )
    reconstructed = matrix_multiply(left_matrix, right_matrix)
    rank = flattening.exact_rank
    transition_rank = exact_matrix_rank(transition)
    bound = min(
        tree.state_sizes[parent],
        tree.state_sizes[child],
        transition_rank,
    )
    return EdgeFlatteningCertificate(
        parent=parent,
        child=child,
        left_leaves=left_leaves,
        right_leaves=right_leaves,
        flattening=flattening,
        parent_factor=parent_matrix,
        edge_transition=transition,
        child_factor=child_matrix,
        three_factor_reconstructed=three_factor_reconstructed,
        left_factor=left_matrix,
        right_factor=right_matrix,
        reconstructed=reconstructed,
        parent_state_size=tree.state_sizes[parent],
        boundary_state_size=tree.state_sizes[child],
        transition_rank=transition_rank,
        sharp_rank_bound=bound,
        flattening_rank=rank,
        parent_factor_rank=exact_matrix_rank(parent_matrix),
        child_factor_rank=exact_matrix_rank(child_matrix),
        left_factor_rank=exact_matrix_rank(left_matrix),
        right_factor_rank=exact_matrix_rank(right_matrix),
        three_factorization_exact=three_factor_reconstructed == flattening.matrix,
        factorization_exact=reconstructed == flattening.matrix,
        rank_bound_holds=rank <= bound,
    )


@dataclass(frozen=True, slots=True)
class ObservationFlatteningCertificate:
    latent: TensorFlattening
    observed: TensorFlattening
    left_operator: RationalMatrix
    right_operator: RationalMatrix
    transformed: RationalMatrix
    identity_exact: bool
    latent_rank: int
    observed_rank: int
    left_operator_rank: int
    right_operator_rank: int
    full_latent_rank_channels: bool
    rank_nonincrease: bool
    rank_preserved_when_injective: bool


def observation_flattening_certificate(
    latent_law: FiniteJointLaw,
    channels: Sequence[FiniteObservationChannel],
    left_coordinates: Sequence[int],
) -> ObservationFlatteningCertificate:
    """Certify ``Flat(Y)=L Flat(X) R^T`` for local observation channels."""

    items = tuple(channels)
    latent = tensor_flattening(latent_law, left_coordinates)
    observed_law = transform_joint_law(latent_law, items)
    observed = tensor_flattening(observed_law, latent.left_coordinates)
    left_operator = kronecker_product(
        *(items[index].operator for index in latent.left_coordinates)
    )
    right_operator = kronecker_product(
        *(items[index].operator for index in latent.right_coordinates)
    )
    transformed = matrix_multiply(
        matrix_multiply(left_operator, latent.matrix),
        matrix_transpose(right_operator),
    )
    latent_rank = latent.exact_rank
    observed_rank = observed.exact_rank
    full = all(channel.has_full_latent_rank for channel in items)
    return ObservationFlatteningCertificate(
        latent=latent,
        observed=observed,
        left_operator=left_operator,
        right_operator=right_operator,
        transformed=transformed,
        identity_exact=transformed == observed.matrix,
        latent_rank=latent_rank,
        observed_rank=observed_rank,
        left_operator_rank=exact_matrix_rank(left_operator),
        right_operator_rank=exact_matrix_rank(right_operator),
        full_latent_rank_channels=full,
        rank_nonincrease=observed_rank <= latent_rank,
        rank_preserved_when_injective=(not full) or observed_rank == latent_rank,
    )


def recover_latent_joint_law(
    observed_law: FiniteJointLaw,
    channels: Sequence[FiniteObservationChannel],
    *,
    max_entries: int = 2_000_000,
) -> FiniteJointLaw:
    """Recover a latent joint law exactly through full-rank local channels."""

    items = tuple(channels)
    if len(items) != observed_law.dimension:
        raise ValueError("one channel is required per observed coordinate")
    if any(
        observed_law.state_sizes[index] != channel.output_states
        for index, channel in enumerate(items)
    ):
        raise ValueError("observed law and channel output alphabets disagree")
    if not all(channel.has_full_latent_rank for channel in items):
        raise ValueError("every channel must have full latent rank")
    latent_sizes = tuple(channel.latent_states for channel in items)
    if prod(latent_sizes) * prod(observed_law.state_sizes) > max_entries:
        raise ValueError("exact recovery matrix budget exceeded")
    recovery = kronecker_product(
        *(channel.left_inverse_operator() for channel in items)
    )
    vector = _matrix_times_vector(recovery, _dense_law_vector(observed_law))
    if any(value < 0 for value in vector) or sum(vector, Fraction(0)) != 1:
        raise ValueError("observed law is not in the exact image of the supplied channels")
    return FiniteJointLaw.from_dense(latent_sizes, vector)


def minimum_collectively_injective_channel_sets(
    channels: Sequence[FiniteObservationChannel],
) -> tuple[tuple[int, ...], ...]:
    """Return all smallest sensor subsets whose joint channel has full latent rank."""

    items = tuple(channels)
    if not items:
        raise ValueError("at least one channel is required")
    k = items[0].latent_states
    if any(channel.latent_states != k for channel in items):
        raise ValueError("all channels must share one latent alphabet")
    for size in range(1, len(items) + 1):
        winners = []
        for subset in combinations(range(len(items)), size):
            joint = FiniteObservationChannel.jointly_conditionally_independent(
                [items[index] for index in subset]
            )
            if joint.has_full_latent_rank:
                winners.append(subset)
        if winners:
            return tuple(winners)
    return tuple()


def singular_value_observation_bounds(
    certificate: ObservationFlatteningCertificate,
    *,
    rank_bound: int,
    tolerance: float = 1e-10,
) -> dict[str, object]:
    """Numerically certify singular-value and low-rank-tail transfer bounds.

    For ``G=L F R^T`` and full-column-rank ``L,R``:

    ``sigma_min(L)sigma_min(R) E_r(F) <= E_r(G)``
    ``E_r(G) <= ||L||_2 ||R||_2 E_r(F)``,

    where ``E_r`` is the Frobenius distance to matrices of rank at most ``r``.
    The upper bound remains valid without injectivity; the lower bound is then
    reported as zero.
    """

    if rank_bound < 0:
        raise ValueError("rank bound must be nonnegative")

    def array(matrix: RationalMatrix) -> np.ndarray:
        return np.asarray([[float(value) for value in row] for row in matrix], dtype=float)

    latent = array(certificate.latent.matrix)
    observed = array(certificate.observed.matrix)
    left = array(certificate.left_operator)
    right = array(certificate.right_operator)
    singular_latent = np.linalg.svd(latent, compute_uv=False)
    singular_observed = np.linalg.svd(observed, compute_uv=False)
    singular_left = np.linalg.svd(left, compute_uv=False)
    singular_right = np.linalg.svd(right, compute_uv=False)
    latent_tail = float(np.linalg.norm(singular_latent[rank_bound:]))
    observed_tail = float(np.linalg.norm(singular_observed[rank_bound:]))
    left_max = float(singular_left[0])
    right_max = float(singular_right[0])
    left_min = (
        float(singular_left[-1])
        if certificate.left_operator_rank == len(certificate.left_operator[0])
        else 0.0
    )
    right_min = (
        float(singular_right[-1])
        if certificate.right_operator_rank == len(certificate.right_operator[0])
        else 0.0
    )
    lower = left_min * right_min * latent_tail
    upper = left_max * right_max * latent_tail
    scale = max(1.0, observed_tail, upper)
    return {
        "rank_bound": rank_bound,
        "latent_singular_values": tuple(float(value) for value in singular_latent),
        "observed_singular_values": tuple(float(value) for value in singular_observed),
        "left_operator_singular_min": left_min,
        "left_operator_singular_max": left_max,
        "right_operator_singular_min": right_min,
        "right_operator_singular_max": right_max,
        "latent_tail_frobenius": latent_tail,
        "observed_tail_frobenius": observed_tail,
        "lower_bound": lower,
        "upper_bound": upper,
        "lower_bound_applicable": left_min > tolerance and right_min > tolerance,
        "bounds_hold": observed_tail + tolerance * scale >= lower
        and observed_tail <= upper + tolerance * scale,
        "conditioning_amplification_ceiling": (
            (left_max / left_min) * (right_max / right_min)
            if left_min > tolerance and right_min > tolerance
            else float("inf")
        ),
    }
