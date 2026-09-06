# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""SISO control models, exact conversion identities and stability criteria.

Transfer functions describe zero-state input/output behavior. State-space
stability examines all internal modes, including hidden unstable modes.
"""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import model_validator
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust,
    checked_result, numeric_array, sympy_samples, validate_scalars)
from .units import parse_unit


def polynomial(coefficients, variable):
    # Horner evaluation avoids intermediate expanded expressions.
    out = sp.S.Zero
    for coefficient in coefficients:
        out = out*variable+coefficient
    return sp.expand(out)


class TransferFunction(EngineeringModel):
    numerator: tuple[sp.Expr, ...]  # descending powers of s or z
    denominator: tuple[sp.Expr, ...]
    time_domain: Literal["continuous", "discrete"] = "continuous"
    sample_time: sp.Expr | None = None  # seconds, required for discrete
    input_unit: str = ""
    output_unit: str = ""
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_transfer(self):
        if not self.numerator or not self.denominator:
            raise ValueError("transfer function requires coefficient sequences")
        validate_scalars((*self.numerator, *self.denominator))
        if self.denominator[0].is_zero is not False:
            raise ValueError("leading denominator coefficient must be provably nonzero")
        if len(self.numerator) > 1 and self.numerator[0].is_zero is not False:
            raise ValueError("remove leading zero numerator coefficients")
        if self.time_domain == "discrete":
            if self.sample_time is None or self.sample_time.is_positive is not True:
                raise ValueError("discrete systems require positive sample_time in seconds")
        elif self.sample_time is not None:
            raise ValueError("continuous systems cannot specify sample_time")
        parse_unit(self.input_unit)
        parse_unit(self.output_unit)
        cap_trust(self.input_trust)
        return self

    def expression(self, variable=None):
        variable = variable if variable is not None else sp.Symbol("s" if self.time_domain == "continuous" else "z")
        return polynomial(self.numerator, variable)/polynomial(self.denominator, variable)


class StateSpaceSystem(EngineeringModel):
    A: tuple[tuple[sp.Expr, ...], ...]
    B: tuple[tuple[sp.Expr, ...], ...]
    C: tuple[tuple[sp.Expr, ...], ...]
    D: tuple[tuple[sp.Expr, ...], ...]
    time_domain: Literal["continuous", "discrete"] = "continuous"
    sample_time: sp.Expr | None = None
    input_unit: str = ""
    output_unit: str = ""
    input_trust: str = "exact"
    state_units: tuple[str, ...] = ()
    input_units: tuple[str, ...] = ()
    output_units: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_state(self):
        n = len(self.A)
        if not n or any(len(r) != n for r in self.A):
            raise ValueError("A must be a nonempty square matrix")
        m = len(self.B[0]) if self.B else 0
        p = len(self.C)
        if not m or len(self.B) != n or any(len(row) != m for row in self.B):
            raise ValueError("B must be n by m with at least one input")
        if not p or any(len(row) != n for row in self.C):
            raise ValueError("C must be p by n with at least one output")
        if len(self.D) != p or any(len(row) != m for row in self.D):
            raise ValueError("D must be p by m")
        for units, count in ((self.state_units,n),(self.input_units,m),(self.output_units,p)):
            if units and len(units) != count:
                raise ValueError("channel/state unit count does not match matrix dimensions")
            for unit in units:
                parse_unit(unit)
        validate_scalars(x for matrix in (self.A, self.B, self.C, self.D) for row in matrix for x in row)
        if self.time_domain == "discrete":
            if self.sample_time is None or self.sample_time.is_positive is not True:
                raise ValueError("discrete systems require positive sample_time")
        elif self.sample_time is not None:
            raise ValueError("continuous systems cannot specify sample_time")
        parse_unit(self.input_unit)
        parse_unit(self.output_unit)
        cap_trust(self.input_trust)
        return self


class DiscreteControlSystem(StateSpaceSystem):
    """Explicit discrete-time state-space representation."""
    time_domain: Literal["discrete"] = "discrete"
    sample_time: sp.Expr


def _matrix_tuple(matrix):
    return tuple(tuple(matrix[i, j] for j in range(matrix.cols)) for i in range(matrix.rows))


def _system_trust(system):
    if isinstance(system, TransferFunction):
        entries = (*system.numerator, *system.denominator)
    else:
        entries = tuple(x for m in (system.A, system.B, system.C, system.D) for r in m for x in r)
    if system.sample_time is not None:
        entries += (system.sample_time,)
    return arithmetic_trust(entries, system.input_trust)


def to_state_space(system):
    n = len(system.denominator)-1
    if n < 1 or len(system.numerator) > n+1:
        raise ValueError("state-space conversion requires a proper dynamic transfer function")
    a = [sp.cancel(x/system.denominator[0]) for x in system.denominator]
    b = [sp.S.Zero]*(n+1-len(system.numerator))+[sp.cancel(x/system.denominator[0]) for x in system.numerator]
    A = sp.zeros(n)
    for i in range(n-1):
        A[i, i+1] = 1
    for j in range(n):
        A[n-1, j] = -a[n-j]
    B = sp.zeros(n, 1)
    B[n-1, 0] = 1
    C = sp.Matrix([[sp.cancel(b[n-j]-b[0]*a[n-j]) for j in range(n)]])
    D = sp.Matrix([[b[0]]])
    trust = _system_trust(system)
    derived = StateSpaceSystem(A=_matrix_tuple(A), B=_matrix_tuple(B), C=_matrix_tuple(C), D=_matrix_tuple(D),
        time_domain=system.time_domain, sample_time=system.sample_time,
        input_unit=system.input_unit, output_unit=system.output_unit, input_trust=trust)
    z = sp.Symbol("s" if system.time_domain == "continuous" else "z")
    # Solve, don't invert, and check the matrix equation independently.
    v = (z*sp.eye(n)-A).solve(B, method="GJ")
    residual = (z*sp.eye(n)-A)*v-B
    checks = {"resolvent_equation": all(sp.cancel(x) == 0 for x in residual),
              "transfer_identity": sp.cancel((C*v+D)[0]-system.expression(z)) == 0}
    return checked_result("to_state_space", derived, method="companion_realization_identity", trust=trust,
        checks=checks, witness={"A": A.tolist(), "B": B.tolist(), "C": C.tolist(), "D": D.tolist()},
        details={"realization": "controllable_companion", "minimality_claimed": False}), derived


def to_transfer_function(system):
    from .control_design import TransferMatrix, channel_units
    A, B, C, D = (sp.Matrix(m) for m in (system.A, system.B, system.C, system.D))
    z = sp.Symbol("s" if system.time_domain == "continuous" else "z")
    resolvent = z*sp.eye(A.rows)-A
    v = resolvent.solve(B, method="GJ")
    expression = (C*v+D).applyfunc(sp.cancel)
    trust = _system_trust(system)
    inputs, outputs = channel_units(system)
    rows = []
    for i in range(C.rows):
        row = []
        for j in range(B.cols):
            num, den = sp.fraction(expression[i,j])
            row.append(TransferFunction(numerator=tuple(sp.Poly(num,z).all_coeffs()),
                denominator=tuple(sp.Poly(den,z).all_coeffs()), time_domain=system.time_domain,
                sample_time=system.sample_time, input_unit=inputs[j], output_unit=outputs[i], input_trust=trust))
        rows.append(tuple(row))
    derived = rows[0][0] if C.rows == B.cols == 1 else TransferMatrix(entries=tuple(rows),input_trust=trust)
    checks = {"resolvent_equation": all(sp.cancel(x) == 0 for x in resolvent*v-B),
              "output_identity": all(sp.cancel(rows[i][j].expression(z)-expression[i,j]) == 0
                                     for i in range(C.rows) for j in range(B.cols))}
    return checked_result("to_transfer_function", derived, method="state_space_resolvent_identity", trust=trust,
        checks=checks, witness={"resolvent_solution": v.tolist()},
        details={"scope":"zero_state_input_output", "shape":[C.rows,B.cols],
                 "internal_characteristic_polynomial":A.charpoly(z).as_expr(),"hidden_modes_may_cancel":True}),derived


def rank_analysis(system, *, observability=False):
    A = sp.Matrix(system.A)
    seed = sp.Matrix(system.C if observability else system.B)
    if any(x.free_symbols for x in (*A, *seed)):
        raise NotImplementedError("parameter-dependent rank needs explicit parameter cases")
    operation = "observability" if observability else "controllability"
    if any(x.has(sp.Float) for x in (*A, *seed)):
        import numpy as np
        a = numeric_array(tuple(A)).reshape(A.shape)
        block = numeric_array(tuple(seed)).reshape(seed.shape)
        blocks = []
        for index in range(A.rows):
            blocks.append(block)
            if index + 1 < A.rows:
                block = block @ a if observability else a @ block
        numeric = np.vstack(blocks) if observability else np.hstack(blocks)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("numeric Krylov recurrence overflowed")
        singular_values = np.linalg.svd(numeric, compute_uv=False)
        tolerance = max(numeric.shape)*np.finfo(float).eps*float(singular_values[0])
        rank = int(np.sum(singular_values > tolerance))
        return checked_result(operation, {"rank": rank, "dimension": A.rows, operation: rank == A.rows},
            method="krylov_svd_numeric_rank", trust=cap_trust(_system_trust(system), "numeric"),
            precision=53, details={"singular_values": singular_values.tolist(), "rank_tolerance": tolerance,
                                    "rank_is_tolerance_dependent": True})
    blocks, block = [], seed
    # Reuse Krylov products; do not compute unused powers or form an inverse.
    for index in range(A.rows):
        blocks.append(block)
        if index + 1 < A.rows:
            block = block*A if observability else A*block
    matrix = sp.Matrix.vstack(*blocks) if observability else sp.Matrix.hstack(*blocks)
    rank = matrix.rank()
    trust = _system_trust(system)
    # DomainMatrix exact elimination is the computation, witness stores the matrix.
    return checked_result(operation, {"rank": rank, "dimension": A.rows, operation: rank == A.rows},
        method="krylov_rank", trust=trust, details={"matrix": matrix.tolist(), "parameter_cases": "concrete"})


def stability(system):
    z = sp.Symbol("z")
    if isinstance(system, StateSpaceSystem):
        coefficients = tuple(sp.Matrix(system.A).charpoly(z).all_coeffs())
        scope = "internal_asymptotic_stability"
    else:
        # BIBO stability must use the reduced transfer function. Internal modes
        # of a physical realization cannot be inferred from cancelled factors.
        denominator = sp.fraction(sp.cancel(system.expression(z)))[1]
        coefficients = tuple(sp.Poly(denominator, z).all_coeffs())
        scope = "bibo_stability"
    validate_scalars(coefficients, real=True)
    a = [sp.cancel(x/coefficients[0]) for x in coefficients]
    n = len(a)-1
    if system.time_domain == "continuous":
        H = sp.Matrix(n, n, lambda i, j: a[2*j-i+1] if 0 <= 2*j-i+1 <= n else 0)
        tests = [sp.factor(H[:k, :k].det(method="domain-ge")) for k in range(1, n+1)]
        conditions = [sp.Gt(x, 0) for x in tests]
        witness = {"hurwitz_matrix": H.tolist(), "leading_principal_minors": tests}
        method = "routh_hurwitz_principal_minors"
    else:
        tests, rows = [], []
        while len(a) > 1:
            rows.append(tuple(a))
            tests.append(sp.factor(a[0]**2-a[-1]**2))
            if tests[-1].is_positive is not True:
                break
            a = [sp.cancel(a[0]*a[i]-a[-1]*a[-i-1]) for i in range(len(a)-1)]
            lead = a[0]
            a = [sp.cancel(x/lead) for x in a]
        conditions = [sp.Gt(x, 0) for x in tests]
        witness = {"schur_rows": rows, "strict_margins": tests}
        method = "schur_cohn_real_recursion"
    outcomes = [True if c is sp.true else False if c is sp.false else None for c in conditions]
    stable = False if False in outcomes else True if all(v is True for v in outcomes) else None
    trust = _system_trust(system)
    result = checked_result("stability", stable, method=method, trust=trust,
        checks={"criterion_decided": stable is not None} if stable is not None else {}, witness=witness,
        conditions=[sp.sstr(c) for c in conditions if c not in (sp.true, sp.false)],
        details={"scope": scope, "classification": "asymptotically_stable" if stable else
                 "not_asymptotically_stable" if stable is False else "conditional",
                 "coefficients": coefficients, "criterion": method})
    if stable is None:
        result.status = "candidate"
    return result


def frequency_response(system, frequencies):
    # Explicit numerical boundary. Frequency is always rad/s, including discrete systems.
    import numpy as np
    omega = numeric_array(frequencies, real=True)
    if np.any(omega < 0):
        raise ValueError("frequencies must be nonnegative rad/s")
    if system.time_domain == "discrete":
        dt = float(system.sample_time)
        if np.any(omega > np.pi/dt):
            raise ValueError("discrete frequencies exceed Nyquist (pi/sample_time)")
        points = np.exp(1j*omega*dt)
    else:
        points = 1j*omega
    b, a = numeric_array(system.numerator), numeric_array(system.denominator)
    den = np.polyval(a, points)
    if np.any(abs(den) == 0):
        raise ValueError("frequency response undefined at a denominator zero")
    response = np.polyval(b, points)/den
    values = sympy_samples(response)
    return checked_result("frequency_response", values, method="vectorized_horner_response", trust=cap_trust(_system_trust(system), "numeric"),
        details={"angular_frequencies": frequencies, "frequency_unit": "rad/s", "time_domain": system.time_domain,
                 "magnitude": [float(x) for x in abs(response)],
                 "phase_radians": [float(x) for x in np.unwrap(np.angle(response))],
                 "denominator_magnitude": [float(x) for x in abs(den)]}, precision=53)


def response(system, *, step=False):
    if system.time_domain != "continuous":
        raise NotImplementedError("discrete time responses use to_filter followed by apply_signal")
    s, t = sp.symbols("s t", positive=True)
    target = system.expression(s)/(s if step else 1)
    value = sp.inverse_laplace_transform(target, s, t)
    from .control_analysis import TimeResponse
    trust = cap_trust(_system_trust(system), "symbolic")
    if value.has(sp.InverseLaplaceTransform):
        return checked_result("step_response" if step else "impulse_response", value,
            method="inverse_laplace", trust=trust, candidate=True,
            details={"typed_response_available": False, "initial_state": "zero", "causality": "causal; t > 0"})
    reconstructed = sp.laplace_transform(value, t, s, noconds=True)
    same = sp.simplify(reconstructed-target) == 0
    derived = TimeResponse(expression=value, response="step" if step else "impulse",
        input_unit=system.input_unit, output_unit=system.output_unit, input_trust=trust)
    return checked_result("step_response" if step else "impulse_response", derived,
        method="laplace_response_round_trip", trust=trust,
        checks={"laplace_identity": same}, conditions=["causal, zero initial state; t > 0"],
        details={"variable": "t", "time_unit": "s", "initial_state": "zero",
                 "input_class": "unit step" if step else "Dirac impulse",
                 "input_unit": system.input_unit, "output_unit": system.output_unit,
                 "causality": "causal; t > 0", "distribution_at_t_zero_included": False}), derived


def interconnect(left, right, *, feedback=False):
    if left.time_domain != right.time_domain or left.sample_time != right.sample_time:
        raise ValueError("interconnected systems require identical time domains and sample times")
    # No hidden unit conversion: scaling must already agree.
    lu, lo = parse_unit(left.input_unit), parse_unit(left.output_unit)
    ru, ro = parse_unit(right.input_unit), parse_unit(right.output_unit)
    if feedback:
        if lo.dimension != ru.dimension or lo.scale != ru.scale or ro.dimension != lu.dimension or ro.scale != lu.scale:
            raise ValueError("feedback path input/output units do not close the loop")
    elif lo.dimension != ru.dimension or lo.scale != ru.scale:
        raise ValueError("series interface units differ; convert explicitly")
    z = sp.Symbol("s" if left.time_domain == "continuous" else "z")
    a, b = left.expression(z), right.expression(z)
    expression = sp.cancel(a/(1+a*b) if feedback else a*b)
    if expression.has(sp.zoo, sp.nan):
        raise ValueError("feedback interconnection is singular")
    num, den = sp.fraction(expression)
    trust = cap_trust(_system_trust(left), _system_trust(right))
    derived = TransferFunction(numerator=tuple(sp.Poly(num, z).all_coeffs()), denominator=tuple(sp.Poly(den, z).all_coeffs()),
        time_domain=left.time_domain, sample_time=left.sample_time, input_unit=left.input_unit,
        output_unit=left.output_unit if feedback else right.output_unit, input_trust=trust)
    check = sp.cancel(derived.expression(z)*(1+a*b)-a) == 0 if feedback else sp.cancel(derived.expression(z)-a*b) == 0
    return checked_result("feedback" if feedback else "series", derived, method="interconnection_polynomial_identity",
        trust=trust, checks={"identity": check}, details={"feedback_sign": "negative" if feedback else None,
                "scope": "zero_state_input_output", "internal_stability_claimed": False}), derived


def to_filter(system):
    from .signal_processing import Filter
    if system.time_domain != "discrete" or len(system.numerator) > len(system.denominator):
        raise ValueError("only proper discrete transfer functions convert to causal filters")
    b = (sp.S.Zero,)*(len(system.denominator)-len(system.numerator))+system.numerator
    trust = _system_trust(system)
    derived = Filter(numerator=b, denominator=system.denominator,
                     sample_rate=1/system.sample_time, input_trust=trust)
    z = sp.Symbol("z")
    expression = sum(v*z**(-i) for i, v in enumerate(b))/sum(v*z**(-i) for i, v in enumerate(system.denominator))
    return checked_result("to_filter", derived, method="lag_polynomial_identity", trust=trust,
                          checks={"transfer_identity": sp.cancel(expression-system.expression(z)) == 0}), derived


def filter_to_transfer(filt):
    n = max(len(filt.numerator), len(filt.denominator))
    # Multiplying lag polynomials by z^(n-1) appends trailing, not leading zeros.
    b = filt.numerator+(sp.S.Zero,)*(n-len(filt.numerator))
    a = filt.denominator+(sp.S.Zero,)*(n-len(filt.denominator))
    while len(b) > 1 and b[0] == 0:
        b = b[1:]
    trust = arithmetic_trust((*filt.numerator, *filt.denominator, filt.sample_rate), filt.input_trust)
    derived = TransferFunction(numerator=b, denominator=a, time_domain="discrete",
                                sample_time=1/filt.sample_rate, input_trust=trust)
    z = sp.Symbol("z")
    original = sum(v*z**(-i) for i, v in enumerate(filt.numerator))/sum(v*z**(-i) for i, v in enumerate(filt.denominator))
    return checked_result("to_transfer_function", derived, method="lag_polynomial_identity", trust=trust,
                           checks={"transfer_identity": sp.cancel(original-derived.expression(z)) == 0}), derived
