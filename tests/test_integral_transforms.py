# =============================================================================
# MathKernel - test integral transforms
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import pytest
import mpmath as mp
import sympy as sp

from mathkernel.integral_transforms import (
    TransformConvention,
    TransformEngine,
    TransformProblem,
)
from mathkernel.models import TrustLevel


def test_laplace_value_differentially_matches_mpmath_quadrature():
    t, s = sp.symbols("t s", positive=True)
    result = TransformEngine().transform(TransformProblem(
        transform="laplace",
        expression=sp.exp(-3 * t),
        variable=t,
        transform_variable=s,
        domain=sp.Interval(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    ))
    symbolic_value = float(result.value.subs(s, 2))
    independent_value = mp.quad(lambda x: mp.exp(-5 * x), [0, mp.inf])
    assert symbolic_value == pytest.approx(float(independent_value), rel=1e-12)


def test_laplace_exponential_retains_region_of_convergence():
    t, s = sp.symbols("t s", positive=True)
    a = sp.symbols("a")
    problem = TransformProblem(
        transform="laplace",
        expression=sp.exp(a * t),
        variable=t,
        transform_variable=s,
        domain=sp.Interval(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )

    result = TransformEngine().transform(problem)

    assert sp.simplify(result.value - 1 / (s - a)) == 0
    assert result.roc == (sp.re(s) > sp.re(a))
    assert result.verified_checks[0].status == "verified"
    assert result.trust == TrustLevel.SYMBOLIC
    assert result.model_dump(mode="json")["roc"] == "s > re(a)"


def test_inverse_laplace_transform():
    s, t = sp.symbols("s t", positive=True)
    problem = TransformProblem(
        transform="inverse_laplace",
        expression=1 / (s + 1),
        variable=s,
        transform_variable=t,
        domain=sp.S.Complexes,
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )

    result = TransformEngine().transform(problem)

    assert sp.simplify(result.value - sp.exp(-t)) == 0
    assert result.roc == (sp.re(s) > -1)
    assert result.status == "verified"


def test_unresolved_inverse_laplace_roc_prevents_verified_status():
    s, t = sp.symbols("s t")
    unknown_spectrum = sp.Function("F")(s)
    problem = TransformProblem(
        transform="inverse_laplace",
        expression=unknown_spectrum,
        variable=s,
        transform_variable=t,
        domain=sp.S.Complexes,
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )
    result = TransformEngine().transform(problem)
    assert result.status != "verified"
    assert any("unresolved" in str(item).lower()
               for item in result.side_conditions)


def test_laplace_property_obligations_execute_when_decidable():
    t, s, tau = sp.symbols("t s tau", positive=True)
    derivative_problem = TransformProblem(
        transform="laplace",
        expression=sp.Derivative(sp.exp(-t), t),
        variable=t,
        transform_variable=s,
        domain=sp.Interval(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )
    derivative = TransformEngine().transform(derivative_problem)
    checks = {check.name: check.status for check in derivative.verified_checks}
    assert checks["differentiation_theorem"] == "verified"
    assert checks["initial_value_theorem"] == "verified"
    assert checks["final_value_theorem"] == "verified"

    convolution_problem = derivative_problem.model_copy(update={
        "expression": sp.Integral(
            sp.exp(-tau) * (t - tau), (tau, 0, t)),
    })
    convolution = TransformEngine().transform(convolution_problem)
    convolution_checks = {
        check.name: check.status for check in convolution.verified_checks}
    assert convolution_checks["convolution_theorem"] == "verified"


def test_fourier_convention_rejection():
    t, omega = sp.symbols("t omega", real=True)
    unsupported = TransformConvention(
        family="fourier",
        forward_kernel="exp(-2*pi*I*f*t)",
        inverse_kernel="exp(2*pi*I*f*t)",
        forward_normalization=sp.S.One,
        inverse_normalization=sp.S.One,
    )
    problem = TransformProblem(
        transform="fourier",
        expression=sp.exp(-t**2),
        variable=t,
        transform_variable=omega,
        domain=sp.S.Reals,
        assumptions=(),
        convention=unsupported,
    )

    with pytest.raises(ValueError, match="Unsupported fourier"):
        TransformEngine().transform(problem)


def test_numeric_input_ancestry_caps_transform_trust():
    t, s = sp.symbols("t s", positive=True)
    problem = TransformProblem(
        transform="laplace",
        expression=sp.exp(-t),
        variable=t,
        transform_variable=s,
        domain=sp.Interval(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )

    result = TransformEngine().transform(
        problem, input_trust=TrustLevel.NUMERIC
    )

    assert result.verified_checks[0].status == "verified"
    assert result.trust == TrustLevel.NUMERIC
    assert result.evidence_bundle.conservative_trust() == "numeric"


def test_finite_bilateral_z_transform_is_candidate_without_invented_roc():
    n = sp.symbols("n", integer=True)
    z = sp.symbols("z")
    problem = TransformProblem(
        transform="z",
        expression=n,
        variable=n,
        transform_variable=z,
        domain=(sp.Integer(0), sp.Integer(2)),
        assumptions=(),
        convention=TransformConvention.z_bilateral(),
    )
    result = TransformEngine().transform(problem)
    assert sp.simplify(result.value - (1 / z + 2 / z**2)) == 0
    assert result.roc is None
    assert result.status == "candidate"
    assert result.trust == TrustLevel.HEURISTIC


def test_fourier_and_mellin_conventions_produce_expected_pairs():
    t, omega = sp.symbols("t omega", real=True)
    fourier = TransformProblem(
        transform="fourier",
        expression=sp.exp(-t**2),
        variable=t,
        transform_variable=omega,
        domain=sp.S.Reals,
        assumptions=(),
        convention=TransformConvention.fourier_angular_frequency(),
    )
    fourier_result = TransformEngine().transform(fourier)
    expected = sp.sqrt(sp.pi) * sp.exp(-omega**2 / 4)
    assert sp.simplify(fourier_result.value - expected) == 0

    x = sp.Symbol("x", positive=True)
    s = sp.Symbol("s")
    mellin = TransformProblem(
        transform="mellin",
        expression=sp.exp(-x),
        variable=x,
        transform_variable=s,
        domain=sp.Interval.open(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.mellin_standard(),
    )
    mellin_result = TransformEngine().transform(mellin)
    assert mellin_result.value == sp.gamma(s)
    assert mellin_result.roc == (0, sp.oo)


def test_inverse_z_extracts_right_sided_laurent_coefficients():
    z = sp.Symbol("z")
    n = sp.Symbol("n", integer=True)
    problem = TransformProblem(
        transform="inverse_z",
        expression=z / (z - 2),
        variable=z,
        transform_variable=n,
        domain=(sp.Integer(2), sp.oo),
        assumptions=(),
        convention=TransformConvention.z_bilateral(),
    )

    result = TransformEngine().transform(problem)

    assert sp.simplify(result.value.subs(n, 0) - 1) == 0
    assert sp.simplify(result.value.subs(n, 4) - 16) == 0
    assert result.value.subs(n, -1) == 0
    assert result.roc == (2, sp.oo)
    assert result.trust == TrustLevel.SYMBOLIC
    assert result.status == "candidate"
    checks = {check.name: check for check in result.verified_checks}
    assert checks["roc_nonempty_consistency"].status == "verified"
    assert checks["forward_inverse_round_trip"].status == "unknown"


def test_inverse_z_uses_outer_pole_for_left_sided_sequence():
    z = sp.Symbol("z")
    n = sp.Symbol("n", integer=True)
    problem = TransformProblem(
        transform="inverse_z",
        expression=z / (z - 2),
        variable=z,
        transform_variable=n,
        domain=(sp.Integer(0), sp.Integer(2)),
        assumptions=(),
        convention=TransformConvention.z_bilateral(),
    )

    result = TransformEngine().transform(problem)

    assert result.value.subs(n, 0) == 0
    assert sp.simplify(result.value.subs(n, -2) + sp.Rational(1, 4)) == 0
    assert not result.value.has(sp.Function("InverseZTransform"))


def test_inverse_z_repeated_pole_uses_laurent_coefficient_formula():
    z = sp.Symbol("z")
    n = sp.Symbol("n", integer=True)
    problem = TransformProblem(
        transform="inverse_z",
        expression=z**2 / (z - 1) ** 2,
        variable=z,
        transform_variable=n,
        domain=(sp.Integer(1), sp.oo),
        assumptions=(),
        convention=TransformConvention.z_bilateral(),
    )

    result = TransformEngine().transform(problem)

    assert result.value.subs(n, -1) == 0
    assert sp.simplify(result.value.subs(n, 0) - 1) == 0
    assert sp.simplify(result.value.subs(n, 3) - 4) == 0


def test_inverse_z_ambiguous_annulus_remains_serializable_candidate():
    z, a = sp.symbols("z a")
    n = sp.Symbol("n", integer=True)
    problem = TransformProblem(
        transform="inverse_z",
        expression=z / (z - a),
        variable=z,
        transform_variable=n,
        domain=(sp.Integer(1), sp.Integer(3)),
        assumptions=(),
        convention=TransformConvention.z_bilateral(),
    )

    result = TransformEngine().transform(problem)
    serialized = result.model_dump(mode="json")

    assert result.status == "candidate"
    assert result.trust == TrustLevel.UNKNOWN
    assert "InverseZTransform" in serialized["value"]
    assert serialized["roc"] == ["1", "3"]
    assert {item["name"] for item in serialized["verified_checks"]} >= {
        "forward_inverse_round_trip",
        "linearity",
        "roc_nonempty_consistency",
    }


def test_inverse_z_rejects_adversarial_annuli():
    z = sp.Symbol("z")
    n = sp.Symbol("n", integer=True)
    common = {
        "transform": "inverse_z",
        "expression": 1 / (z - 1),
        "variable": z,
        "transform_variable": n,
        "assumptions": (),
        "convention": TransformConvention.z_bilateral(),
    }

    with pytest.raises(ValueError, match="negative inner"):
        TransformProblem(domain=(sp.Integer(-1), sp.Integer(2)), **common)
    with pytest.raises(ValueError, match="inner < outer"):
        TransformProblem(domain=(sp.Integer(2), sp.Integer(2)), **common)


def test_laplace_verification_obligations_do_not_overclaim_theorems():
    t, s = sp.symbols("t s", positive=True)
    problem = TransformProblem(
        transform="laplace",
        expression=sp.exp(-t) + sp.exp(-2 * t),
        variable=t,
        transform_variable=s,
        domain=sp.Interval(0, sp.oo),
        assumptions=(),
        convention=TransformConvention.laplace_standard(),
    )

    result = TransformEngine().transform(problem)
    checks = {check.name: check for check in result.verified_checks}

    assert checks["linearity"].status == "verified"
    assert checks["linearity"].trust == TrustLevel.SYMBOLIC
    assert checks["initial_value_theorem"].status == "verified"
    assert checks["final_value_theorem"].status == "verified"
    assert checks["roc_nonempty_consistency"].trust in {
        TrustLevel.SYMBOLIC,
        TrustLevel.UNKNOWN,
    }
