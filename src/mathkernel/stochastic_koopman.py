# =============================================================================
# MathKernel - Exact stochastic observation transfer and Koopman dilations
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact finite stochastic observation-transfer and Koopman-dilation tools.

This module separates three layers that are easy to conflate:

1. **Arbitrary latent joint laws.**  Basis expansion plus cumulant
   multilinearity gives an exact observation-transfer contraction without any
   dynamical assumption.
2. **Finite Markov paths.**  Multi-time moments require alternating Markov
   propagation and pointwise multiplication.  Merely replacing deterministic
   Koopman powers by a Markov matrix is generally wrong because Markov
   expectation operators need not preserve products.
3. **Deterministic dilations.**  A rational finite Markov kernel is the exact
   average of deterministic maps driven by a finite uniform noise alphabet.
   Finite-horizon path laws can therefore be reproduced by a deterministic
   augmented system.

All probability calculations use :class:`fractions.Fraction`.  Observable and
basis values may be any scalar type that interoperates with Fraction and
supports addition and multiplication (for example int, Fraction, complex, or
SymPy expressions).  Basis-coordinate solving is intentionally restricted to
rational matrices so that its certificate is exact and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from typing import Iterable, Sequence, TypeVar

from .cumulants import connected_statistic

Scalar = TypeVar("Scalar")
StateTuple = tuple[int, ...]


def _as_fraction(value: Fraction | int | str | float) -> Fraction:
    """Convert a probability-like value without silently using binary floats."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _sum_terms(terms: Iterable[Scalar], zero=Fraction(0)):
    total = None
    for term in terms:
        total = term if total is None else total + term
    return zero if total is None else total


def _validate_probability_vector(
    probabilities: Sequence[Fraction | int | str | float],
    *,
    name: str,
) -> tuple[Fraction, ...]:
    result = tuple(_as_fraction(value) for value in probabilities)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if any(value < 0 for value in result) or sum(result, Fraction(0)) != 1:
        raise ValueError(f"{name} must be nonnegative and sum exactly to one")
    return result


def _identity_matrix(n: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(int(i == j), 1) for j in range(n)) for i in range(n)
    )


def _matrix_multiply(
    left: Sequence[Sequence[Fraction]],
    right: Sequence[Sequence[Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    rows = len(left)
    inner = len(right)
    if rows == 0 or inner == 0 or any(len(row) != inner for row in left):
        raise ValueError("incompatible or empty left matrix")
    columns = len(right[0])
    if columns == 0 or any(len(row) != columns for row in right):
        raise ValueError("incompatible or ragged right matrix")
    return tuple(
        tuple(
            sum(
                (_as_fraction(left[i][k]) * _as_fraction(right[k][j]) for k in range(inner)),
                Fraction(0),
            )
            for j in range(columns)
        )
        for i in range(rows)
    )


def _matrix_power(
    matrix: Sequence[Sequence[Fraction]], steps: int
) -> tuple[tuple[Fraction, ...], ...]:
    if steps < 0:
        raise ValueError("matrix power must be nonnegative")
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    result = _identity_matrix(n)
    base = tuple(tuple(_as_fraction(value) for value in row) for row in matrix)
    exponent = steps
    while exponent:
        if exponent & 1:
            result = _matrix_multiply(result, base)
        exponent >>= 1
        if exponent:
            base = _matrix_multiply(base, base)
    return result


def _row_times_matrix(
    row: Sequence[Scalar], matrix: Sequence[Sequence[Fraction]]
) -> tuple[Scalar, ...]:
    if len(row) != len(matrix) or not matrix:
        raise ValueError("row and matrix dimensions do not agree")
    columns = len(matrix[0])
    if any(len(matrix_row) != columns for matrix_row in matrix):
        raise ValueError("matrix must not be ragged")
    return tuple(
        _sum_terms(
            (row[i] * matrix[i][j] for i in range(len(row)) if matrix[i][j]),
            Fraction(0),
        )
        for j in range(columns)
    )


def _matrix_times_column(
    matrix: Sequence[Sequence[Fraction]], column: Sequence[Scalar]
) -> tuple[Scalar, ...]:
    if not matrix or any(len(row) != len(column) for row in matrix):
        raise ValueError("matrix and column dimensions do not agree")
    return tuple(
        _sum_terms(
            (matrix[i][j] * column[j] for j in range(len(column)) if matrix[i][j]),
            Fraction(0),
        )
        for i in range(len(matrix))
    )


def _lcm(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return abs(left * right) // gcd(left, right)


@dataclass(frozen=True)
class FiniteJointLaw:
    """Exact joint law of finitely many finite latent variables.

    ``state_sizes[j]`` is the cardinality of variable ``X_j`` and
    ``probabilities[x_tuple]`` is its exact joint mass.  Zero-mass tuples may be
    omitted.
    """

    state_sizes: tuple[int, ...]
    probabilities: dict[StateTuple, Fraction]

    def __post_init__(self) -> None:
        sizes = tuple(int(size) for size in self.state_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("state_sizes must contain positive cardinalities")
        normalized: dict[StateTuple, Fraction] = {}
        for state, probability in self.probabilities.items():
            key = tuple(int(value) for value in state)
            if len(key) != len(sizes):
                raise ValueError("every state tuple must match state_sizes")
            if any(not 0 <= key[j] < sizes[j] for j in range(len(sizes))):
                raise ValueError(f"state tuple {key!r} is out of range")
            mass = _as_fraction(probability)
            if mass < 0:
                raise ValueError("joint probabilities must be nonnegative")
            if mass:
                normalized[key] = normalized.get(key, Fraction(0)) + mass
        if sum(normalized.values(), Fraction(0)) != 1:
            raise ValueError("joint probabilities must sum exactly to one")
        object.__setattr__(self, "state_sizes", sizes)
        object.__setattr__(self, "probabilities", normalized)

    @property
    def dimension(self) -> int:
        return len(self.state_sizes)

    @classmethod
    def from_dense(
        cls,
        state_sizes: Sequence[int],
        probabilities: Sequence[Fraction | int | str | float],
    ) -> "FiniteJointLaw":
        """Build a law from lexicographically ordered dense masses."""

        sizes = tuple(int(size) for size in state_sizes)
        expected = 1
        for size in sizes:
            expected *= size
        if len(probabilities) != expected:
            raise ValueError(f"expected {expected} dense probabilities")
        masses = (_as_fraction(value) for value in probabilities)
        mapping = {
            state: mass
            for state, mass in zip(product(*(range(size) for size in sizes)), masses, strict=True)
            if mass
        }
        return cls(sizes, mapping)

    def moment(self, functions: Sequence[Sequence[Scalar]]):
        """Return ``E[prod_j f_j(X_j)]`` exactly."""

        if len(functions) != self.dimension:
            raise ValueError("one function is required for every joint coordinate")
        for j, function in enumerate(functions):
            if len(function) != self.state_sizes[j]:
                raise ValueError(f"function {j} has the wrong state cardinality")
        return _sum_terms(
            (
                probability
                * _product_values(functions[j][state[j]] for j in range(self.dimension))
                for state, probability in self.probabilities.items()
            ),
            Fraction(0),
        )

    def cumulant(self, functions: Sequence[Sequence[Scalar]]):
        """Return the algebraic joint cumulant of the supplied coordinates."""

        if len(functions) != self.dimension:
            raise ValueError("one function is required for every joint coordinate")
        for j, function in enumerate(functions):
            if len(function) != self.state_sizes[j]:
                raise ValueError(f"function {j} has the wrong state cardinality")

        def raw(block: tuple[int, ...]):
            if not block:
                return Fraction(1)
            return _sum_terms(
                (
                    probability
                    * _product_values(functions[j][state[j]] for j in block)
                    for state, probability in self.probabilities.items()
                ),
                Fraction(0),
            )

        return connected_statistic(raw, self.dimension)

    def marginal(self, coordinates: Sequence[int]) -> "FiniteJointLaw":
        """Return an exact marginal, preserving the requested coordinate order."""

        coords = tuple(int(index) for index in coordinates)
        if not coords:
            raise ValueError("at least one coordinate is required")
        if any(not 0 <= index < self.dimension for index in coords):
            raise ValueError("marginal coordinate out of range")
        masses: dict[StateTuple, Fraction] = {}
        for state, probability in self.probabilities.items():
            key = tuple(state[index] for index in coords)
            masses[key] = masses.get(key, Fraction(0)) + probability
        return FiniteJointLaw(tuple(self.state_sizes[index] for index in coords), masses)


def _product_values(values: Iterable[Scalar]):
    result = None
    for value in values:
        result = value if result is None else result * value
    return Fraction(1) if result is None else result


def pullback_observable(
    observation: Sequence[int], output_function: Sequence[Scalar]
) -> tuple[Scalar, ...]:
    """Pull an output-space function back to a finite latent state space."""

    if not observation:
        raise ValueError("observation map must be nonempty")
    if any(index < 0 or index >= len(output_function) for index in observation):
        raise ValueError("observation map refers outside the output function")
    return tuple(output_function[index] for index in observation)


def rational_basis_coordinates(
    function: Sequence[Fraction | int | str | float],
    basis: Sequence[Sequence[Fraction | int | str | float]],
) -> tuple[Fraction, ...]:
    """Solve ``f(x)=sum_a c[a] basis[a][x]`` by exact Gauss-Jordan elimination.

    The rows of ``basis`` are basis functions and must form a square, linearly
    independent rational matrix.
    """

    n = len(function)
    if n == 0 or len(basis) != n or any(len(row) != n for row in basis):
        raise ValueError("basis must be square and match the function domain")
    # A c = f with A[x, a] = basis[a][x].
    augmented = [
        [_as_fraction(basis[column][row]) for column in range(n)]
        + [_as_fraction(function[row])]
        for row in range(n)
    ]
    for pivot_column in range(n):
        pivot_row = next(
            (row for row in range(pivot_column, n) if augmented[row][pivot_column]),
            None,
        )
        if pivot_row is None:
            raise ValueError("basis is singular")
        if pivot_row != pivot_column:
            augmented[pivot_column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_column],
            )
        pivot = augmented[pivot_column][pivot_column]
        augmented[pivot_column] = [value / pivot for value in augmented[pivot_column]]
        for row in range(n):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            if factor:
                augmented[row] = [
                    augmented[row][column] - factor * augmented[pivot_column][column]
                    for column in range(n + 1)
                ]
    return tuple(augmented[row][-1] for row in range(n))


def observation_transfer_coefficients(
    observation: Sequence[int],
    output_function: Sequence[Fraction | int | str | float],
    basis: Sequence[Sequence[Fraction | int | str | float]],
) -> tuple[Fraction, ...]:
    """Exact coordinates of ``output_function o observation`` in ``basis``."""

    return rational_basis_coordinates(pullback_observable(observation, output_function), basis)


def latent_joint_tensor_value(
    law: FiniteJointLaw,
    bases: Sequence[Sequence[Sequence[Scalar]]],
    alphas: Sequence[int],
    *,
    connected: bool = True,
):
    """Raw or connected latent tensor entry for arbitrary finite joint variables."""

    if len(bases) != law.dimension or len(alphas) != law.dimension:
        raise ValueError("bases and alpha labels must match the joint dimension")
    functions = []
    for j, (basis, alpha) in enumerate(zip(bases, alphas, strict=True)):
        if not 0 <= alpha < len(basis):
            raise ValueError(f"basis index {alpha} out of range at coordinate {j}")
        if len(basis[alpha]) != law.state_sizes[j]:
            raise ValueError(f"basis {j} has the wrong state cardinality")
        functions.append(basis[alpha])
    return law.cumulant(functions) if connected else law.moment(functions)


def arbitrary_joint_contraction(
    law: FiniteJointLaw,
    pulled_observables: Sequence[Sequence[Fraction | int | str | float]],
    bases: Sequence[Sequence[Sequence[Fraction | int | str | float]]],
    *,
    connected: bool = True,
):
    """Evaluate the universal observation-transfer contraction exactly.

    The direct statistic is a cumulant (or moment) of ``pulled_observables``.
    Each pulled observable is expanded in its coordinate-specific basis and
    contracted against the corresponding latent raw/connected tensor.
    """

    if len(pulled_observables) != law.dimension or len(bases) != law.dimension:
        raise ValueError("observables and bases must match the joint dimension")
    coefficients = [
        rational_basis_coordinates(pulled_observables[j], bases[j])
        for j in range(law.dimension)
    ]
    total = None
    for alphas in product(*(range(len(basis)) for basis in bases)):
        transfer = _product_values(
            coefficients[j][alphas[j]] for j in range(law.dimension)
        )
        if not transfer:
            continue
        latent = latent_joint_tensor_value(
            law, bases, alphas, connected=connected
        )
        if latent:
            term = transfer * latent
            total = term if total is None else total + term
    return total if total is not None else Fraction(0)


@dataclass(frozen=True)
class FiniteMarkovKernel:
    """Exact row-stochastic transition kernel on ``{0, ..., n-1}``."""

    transition: tuple[tuple[Fraction, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(
            tuple(_as_fraction(value) for value in row) for row in self.transition
        )
        n = len(rows)
        if n == 0 or any(len(row) != n for row in rows):
            raise ValueError("transition must be nonempty and square")
        for row in rows:
            if any(value < 0 for value in row) or sum(row, Fraction(0)) != 1:
                raise ValueError("each transition row must be a probability vector")
        object.__setattr__(self, "transition", rows)

    @property
    def n(self) -> int:
        return len(self.transition)

    @classmethod
    def from_rows(
        cls, rows: Sequence[Sequence[Fraction | int | str | float]]
    ) -> "FiniteMarkovKernel":
        return cls(tuple(tuple(_as_fraction(value) for value in row) for row in rows))

    @classmethod
    def deterministic(cls, successor: Sequence[int]) -> "FiniteMarkovKernel":
        targets = tuple(int(value) for value in successor)
        n = len(targets)
        if n == 0 or any(not 0 <= target < n for target in targets):
            raise ValueError("successor must map a nonempty state set to itself")
        rows = tuple(
            tuple(Fraction(int(column == target), 1) for column in range(n))
            for target in targets
        )
        return cls(rows)

    def power(self, steps: int) -> tuple[tuple[Fraction, ...], ...]:
        return _matrix_power(self.transition, steps)

    def apply(self, function: Sequence[Scalar], steps: int = 1) -> tuple[Scalar, ...]:
        """Apply the Markov expectation operator ``P^steps`` to a function."""

        if len(function) != self.n:
            raise ValueError("function must be defined on every state")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return _matrix_times_column(self.power(steps), function)

    def distribution_after(
        self,
        initial: Sequence[Fraction | int | str | float],
        steps: int,
    ) -> tuple[Fraction, ...]:
        distribution = _validate_probability_vector(initial, name="initial distribution")
        if len(distribution) != self.n:
            raise ValueError("initial distribution has the wrong state cardinality")
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return tuple(_row_times_matrix(distribution, self.power(steps)))

    def path_moment(
        self,
        initial: Sequence[Fraction | int | str | float],
        times: Sequence[int],
        functions: Sequence[Sequence[Scalar]],
    ):
        """Exact ordered Markov-path moment.

        For ``t_0 <= ... <= t_{d-1}``, this evaluates

        ``mu P^t0 D_f0 P^(t1-t0) D_f1 ... D_f{d-1} 1``.

        The multiplication operators ``D_f`` are essential.  A stochastic
        Markov expectation operator is generally not an algebra homomorphism.
        """

        times_tuple = _validate_times(times)
        if len(functions) != len(times_tuple):
            raise ValueError("one function is required for every requested time")
        for function in functions:
            if len(function) != self.n:
                raise ValueError("every function must be defined on every state")
        weighted: tuple[Scalar, ...] = tuple(
            self.distribution_after(initial, times_tuple[0])
        )
        for index, function in enumerate(functions):
            weighted = tuple(weighted[state] * function[state] for state in range(self.n))
            if index + 1 < len(times_tuple):
                delta = times_tuple[index + 1] - times_tuple[index]
                weighted = _row_times_matrix(weighted, self.power(delta))
        return _sum_terms(weighted, Fraction(0))

    def path_cumulant(
        self,
        initial: Sequence[Fraction | int | str | float],
        times: Sequence[int],
        functions: Sequence[Sequence[Scalar]],
    ):
        """Exact multi-time cumulant obtained from ordered path block moments."""

        times_tuple = _validate_times(times)
        if len(functions) != len(times_tuple):
            raise ValueError("one function is required for every requested time")

        def raw(block: tuple[int, ...]):
            return self.path_moment(
                initial,
                [times_tuple[index] for index in block],
                [functions[index] for index in block],
            )

        return connected_statistic(raw, len(times_tuple))

    def path_joint_law(
        self,
        initial: Sequence[Fraction | int | str | float],
        times: Sequence[int],
        *,
        max_dynamic_states: int = 2_000_000,
    ) -> FiniteJointLaw:
        """Construct the exact joint law at requested times by dynamic programming."""

        times_tuple = _validate_times(times)
        distribution = _validate_probability_vector(initial, name="initial distribution")
        if len(distribution) != self.n:
            raise ValueError("initial distribution has the wrong state cardinality")
        requested: dict[int, list[int]] = {}
        for position, time in enumerate(times_tuple):
            requested.setdefault(time, []).append(position)

        # State: (current Markov state, partially filled observation tuple).
        blank = (-1,) * len(times_tuple)
        dynamic: dict[tuple[int, StateTuple], Fraction] = {}
        for state, probability in enumerate(distribution):
            if probability:
                record = list(blank)
                for position in requested.get(0, []):
                    record[position] = state
                dynamic[(state, tuple(record))] = probability

        for time in range(1, times_tuple[-1] + 1):
            next_dynamic: dict[tuple[int, StateTuple], Fraction] = {}
            for (state, record), probability in dynamic.items():
                for target, transition_probability in enumerate(self.transition[state]):
                    if not transition_probability:
                        continue
                    updated = list(record)
                    for position in requested.get(time, []):
                        updated[position] = target
                    key = (target, tuple(updated))
                    next_dynamic[key] = (
                        next_dynamic.get(key, Fraction(0))
                        + probability * transition_probability
                    )
            if len(next_dynamic) > max_dynamic_states:
                raise ValueError("joint-law dynamic-state budget exceeded")
            dynamic = next_dynamic

        masses: dict[StateTuple, Fraction] = {}
        for (_, record), probability in dynamic.items():
            if any(value < 0 for value in record):
                raise RuntimeError("internal error: requested state was not recorded")
            masses[record] = masses.get(record, Fraction(0)) + probability
        return FiniteJointLaw((self.n,) * len(times_tuple), masses)

    def multiplicativity_defect(
        self, function: Sequence[Scalar], other: Sequence[Scalar]
    ) -> tuple[Scalar, ...]:
        """Return ``P(fg) - (Pf)(Pg)`` statewise.

        At each state this is the conditional covariance of ``f(X_1)`` and
        ``g(X_1)`` under the corresponding transition row.
        """

        if len(function) != self.n or len(other) != self.n:
            raise ValueError("functions must be defined on every state")
        product_function = tuple(function[i] * other[i] for i in range(self.n))
        joint = self.apply(product_function)
        first = self.apply(function)
        second = self.apply(other)
        return tuple(joint[i] - first[i] * second[i] for i in range(self.n))

    def is_deterministic(self) -> bool:
        return all(
            sum(value == 1 for value in row) == 1
            and all(value in (0, 1) for value in row)
            for row in self.transition
        )

    def deterministic_successor(self) -> tuple[int, ...]:
        if not self.is_deterministic():
            raise ValueError("kernel is not deterministic on its original state space")
        return tuple(row.index(Fraction(1)) for row in self.transition)

    def algebra_homomorphism_certificate(self) -> dict:
        """Certify the finite criterion ``P(fg)=Pf Pg`` iff ``P`` is deterministic.

        It suffices to test idempotent point indicators ``e_y``.  Their defect
        at state ``x`` is ``p_xy - p_xy^2``.  A row-stochastic kernel has all
        such defects zero exactly when every entry is 0 or 1, hence every row
        is a point mass.
        """

        witnesses = []
        for state, row in enumerate(self.transition):
            for target, probability in enumerate(row):
                defect = probability - probability * probability
                if defect:
                    witnesses.append(
                        {
                            "state": state,
                            "target": target,
                            "probability": probability,
                            "indicator_defect": defect,
                        }
                    )
        deterministic = self.is_deterministic()
        return {
            "deterministic": deterministic,
            "multiplicative_on_point_indicators": not witnesses,
            "equivalence_verified": deterministic == (not witnesses),
            "witnesses": witnesses,
        }


def _validate_times(times: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(time) for time in times)
    if not result:
        raise ValueError("at least one time is required")
    if any(time < 0 for time in result):
        raise ValueError("times must be nonnegative")
    if any(result[index] > result[index + 1] for index in range(len(result) - 1)):
        raise ValueError("times must be nondecreasing")
    return result


def markov_latent_tensor_value(
    kernel: FiniteMarkovKernel,
    initial: Sequence[Fraction | int | str | float],
    times: Sequence[int],
    basis: Sequence[Sequence[Scalar]],
    alphas: Sequence[int],
    *,
    connected: bool = True,
):
    """Dynamics-specific latent tensor generated by a finite Markov path."""

    if len(times) != len(alphas):
        raise ValueError("times and alpha labels must have equal length")
    functions = []
    for alpha in alphas:
        if not 0 <= alpha < len(basis):
            raise ValueError("basis index out of range")
        if len(basis[alpha]) != kernel.n:
            raise ValueError("basis has the wrong state cardinality")
        functions.append(basis[alpha])
    if connected:
        return kernel.path_cumulant(initial, times, functions)
    return kernel.path_moment(initial, times, functions)


def markov_observation_contraction(
    kernel: FiniteMarkovKernel,
    initial: Sequence[Fraction | int | str | float],
    times: Sequence[int],
    pulled_observables: Sequence[Sequence[Fraction | int | str | float]],
    basis: Sequence[Sequence[Fraction | int | str | float]],
    *,
    connected: bool = True,
):
    """Contract observation coefficients against a Markov-generated tensor."""

    if len(times) != len(pulled_observables):
        raise ValueError("times and observables must have equal length")
    coefficients = [
        rational_basis_coordinates(observable, basis)
        for observable in pulled_observables
    ]
    total = None
    for alphas in product(range(len(basis)), repeat=len(times)):
        transfer = _product_values(
            coefficients[j][alphas[j]] for j in range(len(times))
        )
        if not transfer:
            continue
        latent = markov_latent_tensor_value(
            kernel,
            initial,
            times,
            basis,
            alphas,
            connected=connected,
        )
        if latent:
            term = transfer * latent
            total = term if total is None else total + term
    return total if total is not None else Fraction(0)


def compressed_transport_moment(
    kernel: FiniteMarkovKernel,
    initial: Sequence[Fraction | int | str | float],
    times: Sequence[int],
    functions: Sequence[Sequence[Scalar]],
):
    """Naive same-initial-state moment of independently propagated observables.

    This equals the true path moment for deterministic kernels because their
    Markov operator is a Koopman composition operator.  For genuinely
    stochastic kernels it is only a diagnostic and generally differs from the
    path moment.
    """

    times_tuple = _validate_times(times)
    distribution = _validate_probability_vector(initial, name="initial distribution")
    if len(distribution) != kernel.n or len(functions) != len(times_tuple):
        raise ValueError("dimensions do not agree")
    propagated = [kernel.apply(function, time) for function, time in zip(functions, times_tuple, strict=True)]
    return _sum_terms(
        (
            distribution[state]
            * _product_values(column[state] for column in propagated)
            for state in range(kernel.n)
        ),
        Fraction(0),
    )


def compressed_transport_cumulant(
    kernel: FiniteMarkovKernel,
    initial: Sequence[Fraction | int | str | float],
    times: Sequence[int],
    functions: Sequence[Sequence[Scalar]],
):
    """Cumulant counterpart of :func:`compressed_transport_moment`."""

    times_tuple = _validate_times(times)
    if len(functions) != len(times_tuple):
        raise ValueError("one function is required for every time")

    def raw(block: tuple[int, ...]):
        return compressed_transport_moment(
            kernel,
            initial,
            [times_tuple[index] for index in block],
            [functions[index] for index in block],
        )

    return connected_statistic(raw, len(times_tuple))


@dataclass(frozen=True)
class RationalMarkovDilation:
    """Finite uniform-noise deterministic realization of a rational kernel."""

    sampler: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(int(target) for target in row) for row in self.sampler)
        n = len(rows)
        if n == 0 or not rows[0]:
            raise ValueError("sampler must have states and a nonempty noise alphabet")
        noise_size = len(rows[0])
        if any(len(row) != noise_size for row in rows):
            raise ValueError("all sampler rows must share one noise alphabet")
        if any(not 0 <= target < n for row in rows for target in row):
            raise ValueError("sampler target out of range")
        object.__setattr__(self, "sampler", rows)

    @property
    def n(self) -> int:
        return len(self.sampler)

    @property
    def noise_size(self) -> int:
        return len(self.sampler[0])

    @classmethod
    def from_kernel(
        cls,
        kernel: FiniteMarkovKernel,
        *,
        max_noise_size: int = 1_000_000,
    ) -> "RationalMarkovDilation":
        noise_size = 1
        for row in kernel.transition:
            for probability in row:
                noise_size = _lcm(noise_size, probability.denominator)
                if noise_size > max_noise_size:
                    raise ValueError("required finite noise alphabet exceeds budget")
        rows = []
        for row in kernel.transition:
            targets = []
            for target, probability in enumerate(row):
                count = probability * noise_size
                if count.denominator != 1:
                    raise RuntimeError("internal error constructing rational dilation")
                targets.extend([target] * count.numerator)
            if len(targets) != noise_size:
                raise RuntimeError("internal error: sampler row has wrong length")
            rows.append(tuple(targets))
        return cls(tuple(rows))

    def step(self, state: int, noise: int) -> int:
        if not 0 <= state < self.n or not 0 <= noise < self.noise_size:
            raise ValueError("state or noise value out of range")
        return self.sampler[state][noise]

    def induced_kernel(self) -> FiniteMarkovKernel:
        rows = []
        for state in range(self.n):
            counts = [0] * self.n
            for target in self.sampler[state]:
                counts[target] += 1
            rows.append(
                tuple(Fraction(count, self.noise_size) for count in counts)
            )
        return FiniteMarkovKernel(tuple(rows))

    def compression_apply(self, function: Sequence[Scalar]) -> tuple[Scalar, ...]:
        """Average one deterministic lifted step over the hidden noise."""

        if len(function) != self.n:
            raise ValueError("function must be defined on every state")
        return tuple(
            _sum_terms(
                (function[self.step(state, noise)] for noise in range(self.noise_size)),
                Fraction(0),
            )
            / self.noise_size
            for state in range(self.n)
        )

    def path_joint_law(
        self,
        initial: Sequence[Fraction | int | str | float],
        times: Sequence[int],
        *,
        max_dynamic_states: int = 2_000_000,
    ) -> FiniteJointLaw:
        """Exact finite-horizon law generated by deterministic state/noise updates."""

        times_tuple = _validate_times(times)
        distribution = _validate_probability_vector(initial, name="initial distribution")
        if len(distribution) != self.n:
            raise ValueError("initial distribution has the wrong state cardinality")
        requested: dict[int, list[int]] = {}
        for position, time in enumerate(times_tuple):
            requested.setdefault(time, []).append(position)
        blank = (-1,) * len(times_tuple)
        dynamic: dict[tuple[int, StateTuple], Fraction] = {}
        for state, probability in enumerate(distribution):
            if probability:
                record = list(blank)
                for position in requested.get(0, []):
                    record[position] = state
                dynamic[(state, tuple(record))] = probability

        noise_probability = Fraction(1, self.noise_size)
        for time in range(1, times_tuple[-1] + 1):
            next_dynamic: dict[tuple[int, StateTuple], Fraction] = {}
            for (state, record), probability in dynamic.items():
                for noise in range(self.noise_size):
                    target = self.step(state, noise)
                    updated = list(record)
                    for position in requested.get(time, []):
                        updated[position] = target
                    key = (target, tuple(updated))
                    next_dynamic[key] = (
                        next_dynamic.get(key, Fraction(0))
                        + probability * noise_probability
                    )
            if len(next_dynamic) > max_dynamic_states:
                raise ValueError("dilation dynamic-state budget exceeded")
            dynamic = next_dynamic

        masses: dict[StateTuple, Fraction] = {}
        for (_, record), probability in dynamic.items():
            if any(value < 0 for value in record):
                raise RuntimeError("internal error: requested state was not recorded")
            masses[record] = masses.get(record, Fraction(0)) + probability
        return FiniteJointLaw((self.n,) * len(times_tuple), masses)


def deterministic_koopman_equivalence(
    kernel: FiniteMarkovKernel,
    initial: Sequence[Fraction | int | str | float],
    times: Sequence[int],
    functions: Sequence[Sequence[Scalar]],
) -> dict:
    """Compare true Markov and same-state Koopman-style path statistics."""

    true_moment = kernel.path_moment(initial, times, functions)
    compressed_moment = compressed_transport_moment(kernel, initial, times, functions)
    true_cumulant = kernel.path_cumulant(initial, times, functions)
    compressed_cumulant = compressed_transport_cumulant(
        kernel, initial, times, functions
    )
    return {
        "kernel_deterministic": kernel.is_deterministic(),
        "true_moment": true_moment,
        "compressed_moment": compressed_moment,
        "moment_equal": true_moment == compressed_moment,
        "true_cumulant": true_cumulant,
        "compressed_cumulant": compressed_cumulant,
        "cumulant_equal": true_cumulant == compressed_cumulant,
    }
