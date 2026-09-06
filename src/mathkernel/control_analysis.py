# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded control representations and analysis with scoped evidence."""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import model_validator
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust,
    checked_result, numeric_array, sympy_samples, validate_scalars)
from .units import parse_unit


class ZeroPoleGain(EngineeringModel):
    zeros: tuple[sp.Expr, ...] = ()
    poles: tuple[sp.Expr, ...] = ()
    gain: sp.Expr = sp.Integer(1)
    time_domain: Literal["continuous", "discrete"] = "continuous"
    sample_time: sp.Expr | None = None
    input_unit: str = ""
    output_unit: str = ""
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_zpk(self):
        validate_scalars((*self.zeros, *self.poles, self.gain))
        if self.time_domain == "discrete":
            if self.sample_time is None or self.sample_time.is_positive is not True:
                raise ValueError("discrete ZPK requires positive sample_time")
        elif self.sample_time is not None:
            raise ValueError("continuous ZPK cannot specify sample_time")
        parse_unit(self.input_unit); parse_unit(self.output_unit); cap_trust(self.input_trust)
        return self


class FrequencyResponse(EngineeringModel):
    angular_frequencies: tuple[sp.Expr, ...]
    response: tuple[sp.Expr, ...]
    magnitude: tuple[sp.Expr, ...]
    phase_radians: tuple[sp.Expr, ...]
    magnitude_db: tuple[sp.Expr | None, ...]  # None denotes -infinity dB at an exact sampled zero
    analysis: Literal["frequency_response", "bode", "nyquist"]
    time_domain: Literal["continuous", "discrete"]
    sample_time: sp.Expr | None = None
    input_unit: str = ""
    output_unit: str = ""
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_response(self):
        n = len(self.angular_frequencies)
        if not n or any(len(v) != n for v in (self.response, self.magnitude, self.phase_radians, self.magnitude_db)):
            raise ValueError("frequency-response arrays must be nonempty and aligned")
        validate_scalars((*self.angular_frequencies, *self.response, *self.magnitude,
                          *self.phase_radians, *(v for v in self.magnitude_db if v is not None)))
        if any(v.is_real is not True for v in (*self.angular_frequencies, *self.magnitude, *self.phase_radians)):
            raise ValueError("frequency, magnitude and phase arrays must be real")
        if any((b-a).is_nonnegative is not True for a, b in zip(self.angular_frequencies, self.angular_frequencies[1:])):
            raise ValueError("angular frequencies must be nondecreasing")
        if self.time_domain == "discrete":
            if self.sample_time is None or self.sample_time.is_positive is not True:
                raise ValueError("discrete response requires positive sample_time")
        elif self.sample_time is not None:
            raise ValueError("continuous response cannot specify sample_time")
        parse_unit(self.input_unit); parse_unit(self.output_unit); cap_trust(self.input_trust)
        return self


class RootLocus(EngineeringModel):
    gains: tuple[sp.Expr, ...]
    branches: tuple[tuple[sp.Expr, ...], ...]  # gain-major ordered roots
    characteristic: Literal["D(q)+k*N(q)=0"] = "D(q)+k*N(q)=0"
    variable: Literal["s", "z"] = "s"
    input_trust: str = "numeric"

    @model_validator(mode="after")
    def validate_locus(self):
        if not self.gains or len(self.branches) != len(self.gains) or not self.branches:
            raise ValueError("root locus requires aligned gain and root arrays")
        width = len(self.branches[0])
        if not width or any(len(row) != width for row in self.branches):
            raise ValueError("root-locus branch width must be constant")
        validate_scalars((*self.gains, *(x for row in self.branches for x in row)))
        if any(v.is_real is not True or v.is_nonnegative is not True for v in self.gains):
            raise ValueError("root-locus gains must be nonnegative real values")
        if any((b-a).is_nonnegative is not True for a, b in zip(self.gains, self.gains[1:])):
            raise ValueError("root-locus gains must be nondecreasing")
        cap_trust(self.input_trust)
        return self


class TimeResponse(EngineeringModel):
    expression: sp.Expr
    variable: str = "t"
    response: Literal["step", "impulse"]
    initial_state: Literal["zero"] = "zero"
    causality: Literal["causal; t > 0"] = "causal; t > 0"
    input_unit: str = ""
    output_unit: str = ""
    time_unit: Literal["s"] = "s"
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_time_response(self):
        if not self.variable.isidentifier() or self.variable.startswith("_"):
            raise ValueError("variable must be a public identifier")
        validate_scalars((self.expression,))
        if self.expression.free_symbols - {sp.Symbol(self.variable, positive=True)}:
            # SymPy symbols compare by assumptions; accept same-name time symbols.
            if any(v.name != self.variable for v in self.expression.free_symbols):
                raise ValueError("time response contains undeclared symbols")
        parse_unit(self.input_unit); parse_unit(self.output_unit); cap_trust(self.input_trust)
        return self


def _trust(model):
    entries = (*model.zeros, *model.poles, model.gain)
    if model.sample_time is not None: entries += (model.sample_time,)
    return arithmetic_trust(entries, model.input_trust)


def zpk_to_transfer(model):
    from .control_systems import TransferFunction, polynomial
    q = sp.Symbol("s" if model.time_domain == "continuous" else "z")
    numerator = sp.Poly(sp.expand(model.gain*sp.prod(q-z for z in model.zeros)), q).all_coeffs()
    denominator = sp.Poly(sp.expand(sp.prod(q-p for p in model.poles)), q).all_coeffs()
    derived = TransferFunction(numerator=tuple(numerator), denominator=tuple(denominator),
        time_domain=model.time_domain, sample_time=model.sample_time, input_unit=model.input_unit,
        output_unit=model.output_unit, input_trust=_trust(model))
    identity = sp.cancel(derived.expression(q) - model.gain*sp.prod(q-z for z in model.zeros)/sp.prod(q-p for p in model.poles)) == 0
    return checked_result("to_transfer_function", derived, method="factored_polynomial_identity",
        trust=_trust(model), checks={"factor_identity": identity},
        details={"multiplicity_preserved": True, "cancellation_performed": False}), derived


def transfer_to_zpk(system, *, mode="exact"):
    from .control_systems import polynomial, _system_trust
    q = sp.Symbol("s" if system.time_domain == "continuous" else "z")
    if mode == "exact":
        if any(x.free_symbols for x in (*system.numerator, *system.denominator)):
            raise NotImplementedError("exact ZPK conversion requires concrete coefficients")
        numerator, denominator = sp.Poly(polynomial(system.numerator, q), q), sp.Poly(polynomial(system.denominator, q), q)
        zeros = () if numerator.is_zero else tuple(numerator.all_roots())
        poles = tuple(denominator.all_roots())
        gain = sp.S.Zero if numerator.is_zero else sp.cancel(numerator.LC()/denominator.LC())
        trust, residual = _system_trust(system), None
    elif mode == "numeric":
        import numpy as np
        b, a = numeric_array(system.numerator), numeric_array(system.denominator)
        zeros = tuple(sympy_samples(np.roots(b))) if not np.all(b == 0) and len(b) > 1 else ()
        poles = tuple(sympy_samples(np.roots(a))) if len(a) > 1 else ()
        gain = sp.Float(0) if np.all(b == 0) else sympy_samples([b[0]/a[0]])[0]
        rb = np.asarray(np.poly([complex(z) for z in zeros])*complex(gain) if zeros else [complex(gain)])
        ra = np.asarray(np.poly([complex(p) for p in poles]) if poles else [1.])
        target_b, target_a = b/a[0], a/a[0]
        rb = np.pad(rb, (len(target_b)-len(rb), 0))
        residual = float(max(np.max(abs(rb-target_b))/max(1., float(np.max(abs(target_b)))),
                             np.max(abs(ra-target_a))/max(1., float(np.max(abs(target_a))))))
        trust = cap_trust(_system_trust(system), "numeric")
    else:
        raise ValueError("mode must be exact or numeric")
    derived = ZeroPoleGain(zeros=zeros, poles=poles, gain=gain, time_domain=system.time_domain,
        sample_time=system.sample_time, input_unit=system.input_unit, output_unit=system.output_unit,
        input_trust=trust)
    rebuilt = model_expression(derived, q)
    exact_check = sp.cancel(rebuilt-system.expression(q)) == 0 if mode == "exact" else None
    checks = {"factor_identity": exact_check} if mode == "exact" else {"coefficient_reconstruction_numeric": residual <= 1e-9}
    return checked_result("to_zero_pole_gain", derived, method=f"{mode}_polynomial_roots",
        trust=trust, checks=checks, residual=residual, precision=53 if mode == "numeric" else None,
        details={"multiplicity_preserved": True, "cancellation_performed": False,
                 "verification_absolute_tolerance": 1e-9 if mode == "numeric" else None}), derived


def model_expression(model, q):
    return model.gain*sp.prod(q-z for z in model.zeros)/sp.prod(q-p for p in model.poles)


def poles_zeros(system, *, poles):
    from .control_systems import StateSpaceSystem, _system_trust, to_transfer_function
    q = sp.Symbol("s" if system.time_domain == "continuous" else "z")
    if isinstance(system, StateSpaceSystem) and poles:
        poly = sp.Matrix(system.A).charpoly(q)
        if any(x.free_symbols for x in poly.all_coeffs()):
            raise NotImplementedError("exact poles require concrete coefficients")
        values = tuple(poly.all_roots())
        scope = "internal_state_modes"
    else:
        if isinstance(system, StateSpaceSystem):
            if len(system.B[0]) != 1 or len(system.C) != 1:
                raise NotImplementedError("exact transmission zeros currently require SISO state space")
            _, system = to_transfer_function(system)
        coeff = system.denominator if poles else system.numerator
        poly = sp.Poly(sum(c*q**(len(coeff)-i-1) for i,c in enumerate(coeff)), q)
        if any(x.free_symbols for x in poly.all_coeffs()):
            raise NotImplementedError("exact roots require concrete coefficients")
        values = () if poly.is_zero else tuple(poly.all_roots())
        scope = "transfer_denominator_roots" if poles else "transfer_numerator_roots"
    checks = {"root_multiplicity_count": len(values) == (poly.degree() if not poly.is_zero else 0),
              "root_substitution": all(sp.simplify(poly.as_expr().subs(q, v)) == 0 for v in values)}
    return checked_result("poles" if poles else "zeros", values, method="exact_polynomial_roots",
        trust=_system_trust(system), checks=checks,
        details={"scope": scope, "multiplicity_preserved": True, "cancellation_performed": False})


def sampled_frequency_analysis(system, frequencies, *, analysis):
    from .control_systems import frequency_response, _system_trust
    import numpy as np
    base = frequency_response(system, frequencies)
    omega = numeric_array(frequencies, real=True)
    response = np.asarray([complex(v) for v in base.value])
    if analysis == "nyquist":
        if any(value.is_real is not True for value in (*system.numerator, *system.denominator)):
            raise ValueError("Nyquist conjugate completion requires real coefficients")
        omega = np.concatenate((-omega[:0:-1], omega))
        response = np.concatenate((np.conjugate(response[:0:-1]), response))
    magnitude = np.abs(response)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Bode/Nyquist magnitude contains nonfinite values")
    db = np.full(magnitude.shape, np.nan)
    positive = magnitude > 0
    db[positive] = 20*np.log10(magnitude[positive])
    db_values = tuple(sympy_samples(db[positive]))
    db_iterator = iter(db_values)
    typed_db = tuple(next(db_iterator) if present else None for present in positive)
    derived = FrequencyResponse(angular_frequencies=sympy_samples(omega), response=sympy_samples(response),
        magnitude=sympy_samples(magnitude), phase_radians=sympy_samples(np.unwrap(np.angle(response))),
        magnitude_db=typed_db, analysis=analysis, time_domain=system.time_domain,
        sample_time=system.sample_time, input_unit=system.input_unit, output_unit=system.output_unit,
        input_trust=cap_trust(_system_trust(system), "numeric"))
    result = checked_result(analysis, derived, method="vectorized_horner_frequency_analysis",
        trust=derived.input_trust, checks=base.verification, precision=53,
        details={"frequency_unit": "rad/s", "phase_unit": "radian", "magnitude_db_reference": "20*log10(abs(H))",
                 "angular_frequencies": tuple(sympy_samples(omega)),
                 "magnitude": [float(v) for v in magnitude],
                 "phase_radians": [float(v) for v in np.unwrap(np.angle(response))],
                 "magnitude_db": [float(v) if present else None for v, present in zip(db, positive)],
                 "zero_magnitude_db_representation": None,
                 "negative_frequency_completion": "complex conjugate for real coefficients" if analysis == "nyquist" else None,
                 "nyquist_encirclement_claimed": False, "closed_loop_stability_claimed": False})
    return result, derived


def root_locus(system, gains):
    from .control_systems import _system_trust
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    k = numeric_array(gains, real=True)
    if np.any(k < 0) or np.any(np.diff(k) < 0):
        raise ValueError("root-locus gains must be nonnegative and nondecreasing")
    a, b = numeric_array(system.denominator), numeric_array(system.numerator)
    if len(b) > len(a):
        raise ValueError("root locus requires a proper transfer function")
    b = np.pad(b, (len(a)-len(b), 0))
    rows, worst = [], 0.
    previous = None
    for gain in k:
        coefficients = a + gain*b
        first = np.flatnonzero(abs(coefficients) > 0)
        if not len(first): raise ValueError("characteristic polynomial vanishes identically")
        roots = np.roots(coefficients[first[0]:])
        if previous is not None and len(roots) == len(previous):
            assignment = linear_sum_assignment(abs(previous[:,None]-roots[None,:]))[1]
            roots = roots[assignment]
        previous = roots
        scale = max(1., float(np.sum(abs(coefficients))))
        worst = max(worst, float(np.max(abs(np.polyval(coefficients, roots)), initial=0))/scale)
        rows.append(sympy_samples(roots))
    tolerance = float(512*np.finfo(float).eps*max(1, len(a)))
    trust = cap_trust(_system_trust(system), "numeric")
    derived = RootLocus(gains=sympy_samples(k), branches=tuple(rows),
        variable="s" if system.time_domain == "continuous" else "z", input_trust=trust)
    result = checked_result("root_locus", derived, method="polynomial_roots_hungarian_branch_tracking",
        trust=trust, checks={"characteristic_residual": worst <= tolerance}, residual=worst, precision=53,
        details={"feedback_sign": "negative", "branch_ordering": "minimum-distance heuristic",
                 "branch_ordering_certified": False, "stability_intervals_claimed": False,
                 "verification_relative_tolerance": tolerance})
    return result, derived
