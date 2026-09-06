# =============================================================================
# MathKernel - Focused tests for typed continuous probability
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import mpmath as mp
import pytest
import sympy as sp

from mathkernel.continuous_probability import (
    Beta,
    Cauchy,
    ChiSquared,
    ConditionalDistribution,
    Distribution,
    Exponential,
    F,
    Gamma,
    JointDistribution,
    Laplace,
    LogNormal,
    Logistic,
    Normal,
    Pareto,
    QueryStatus,
    RandomVariable,
    StudentT,
    Uniform,
    Weibull,
    bayes_rule,
    convolve,
    correlation,
    covariance,
    expectation,
    kl_divergence,
    mixture,
    order_statistic,
    transform,
    truncate,
)


def test_normal_density_differentially_normalizes_with_mpmath():
    x = sp.Symbol("x", real=True)
    distribution = Normal(sp.S.Zero, sp.S.One, variable=x)
    numeric_density = sp.lambdify(x, distribution.density, "mpmath")
    integral = mp.quad(numeric_density, [-mp.inf, mp.inf])
    assert float(integral) == pytest.approx(1.0, rel=1e-12)


def test_all_initial_distribution_constructors_have_explicit_support():
    distributions = [
        Uniform(sp.Integer(-1), sp.Integer(2)),
        Normal(sp.Integer(0), sp.Integer(1)),
        LogNormal(sp.Integer(0), sp.Integer(1)),
        Exponential(sp.Integer(2)),
        Gamma(sp.Integer(2), sp.Integer(3)),
        Beta(sp.Integer(2), sp.Integer(3)),
        Cauchy(sp.Integer(0), sp.Integer(1)),
        StudentT(sp.Integer(5)),
        ChiSquared(sp.Integer(4)),
        F(sp.Integer(5), sp.Integer(8)),
        Weibull(sp.Integer(2), sp.Integer(3)),
        Pareto(sp.Integer(2), sp.Integer(3)),
        Laplace(sp.Integer(0), sp.Integer(1)),
        Logistic(sp.Integer(0), sp.Integer(1)),
    ]
    assert [distribution.name for distribution in distributions] == [
        "uniform", "normal", "lognormal", "exponential", "gamma", "beta",
        "cauchy", "student_t", "chi_squared", "f", "weibull", "pareto",
        "laplace", "logistic",
    ]
    assert all(isinstance(distribution.support, sp.Set) for distribution in distributions)
    assert all(distribution.density is not None for distribution in distributions)


@pytest.mark.parametrize(
    ("constructor", "arguments"),
    [
        (Uniform, (sp.Integer(1), sp.Integer(1))),
        (Normal, (sp.Integer(0), sp.Integer(0))),
        (LogNormal, (sp.Integer(0), sp.Integer(-1))),
        (Exponential, (sp.Integer(0),)),
        (Gamma, (sp.Integer(-1), sp.Integer(1))),
        (Beta, (sp.Integer(1), sp.Integer(0))),
        (Cauchy, (sp.Integer(0), sp.Integer(0))),
        (StudentT, (sp.Integer(0),)),
        (ChiSquared, (sp.Integer(-1),)),
        (F, (sp.Integer(1), sp.Integer(0))),
        (Weibull, (sp.Integer(0), sp.Integer(1))),
        (Pareto, (sp.Integer(1), sp.Integer(0))),
        (Laplace, (sp.Integer(0), sp.Integer(-1))),
        (Logistic, (sp.Integer(0), sp.Integer(0))),
    ],
)
def test_constructor_parameter_constraints(constructor, arguments):
    with pytest.raises(ValueError):
        constructor(*arguments)


def test_inputs_are_sympy_only_and_are_never_parsed():
    with pytest.raises(TypeError, match="SymPy expression"):
        Normal("0", sp.Integer(1))
    unconstrained = sp.Symbol("sigma")
    conditional = Normal(sp.Integer(0), unconstrained)
    assert conditional.conditions == ["sigma > 0"]
    sigma = sp.Symbol("sigma", positive=True)
    assert Normal(sp.Integer(0), sigma).parameters["sigma"] is sigma


def test_parameter_dependent_existence_remains_unknown_without_an_assumption():
    positive = sp.Symbol("nu", positive=True)
    assert StudentT(positive).mean().status is QueryStatus.UNKNOWN
    assert F(sp.Integer(2), positive).variance().status is QueryStatus.UNKNOWN
    assert Pareto(sp.Integer(1), positive).mean().status is QueryStatus.UNKNOWN


def test_cauchy_distinguishes_nonexistence_from_unsupported_and_unknown():
    distribution = Cauchy(sp.Integer(0), sp.Integer(1))
    for query in ("mean", "variance", "mgf"):
        result = distribution.query(query)
        assert result.status is QueryStatus.DOES_NOT_EXIST
        assert result.value is None
    assert F(sp.Integer(5), sp.Integer(8)).mgf().status is QueryStatus.DOES_NOT_EXIST


def test_normalization_pdf_and_cdf_verification_are_claim_specific():
    x = sp.Symbol("x", real=True)
    distribution = Normal(sp.Integer(0), sp.Integer(1), variable=x, input_trust="exact")
    pdf = distribution.pdf()
    assert pdf.status is QueryStatus.VERIFIED
    assert pdf.verification == {"nonnegative": True, "normalized": True}
    assert pdf.trust == "symbolic"
    assert set(pdf.claim_evidence) == {"pdf"}
    assert pdf.claim_evidence["pdf"].conservative_trust() == "symbolic"

    cdf_at_zero = distribution.cdf(sp.Integer(0))
    assert sp.simplify(cdf_at_zero.value - sp.Rational(1, 2)) == 0
    assert cdf_at_zero.verification["derivative_matches_pdf"] is True
    encoded = distribution.model_dump(mode="json")
    assert encoded["parameters"] == {"mu": "0", "sigma": "1"}
    assert pdf.model_dump(mode="json")["value"]


def test_exact_normalization_across_representative_supports():
    distributions = [
        Uniform(sp.Integer(-2), sp.Integer(3)),
        Exponential(sp.Integer(2)),
        Gamma(sp.Integer(3), sp.Integer(2)),
        Cauchy(sp.Integer(0), sp.Integer(2)),
        StudentT(sp.Integer(5)),
        ChiSquared(sp.Integer(4)),
        Weibull(sp.Integer(2), sp.Integer(3)),
        Pareto(sp.Integer(2), sp.Integer(3)),
        Laplace(sp.Integer(0), sp.Integer(2)),
        Logistic(sp.Integer(0), sp.Integer(2)),
    ]
    assert all(distribution.verify().verification["normalized"] is True for distribution in distributions)


def test_means_variances_and_tail_dependent_existence():
    assert Normal(sp.Integer(3), sp.Integer(2)).mean().value == 3
    assert Normal(sp.Integer(3), sp.Integer(2)).variance().value == 4
    assert StudentT(sp.Integer(1)).mean().status is QueryStatus.DOES_NOT_EXIST
    assert StudentT(sp.Integer(3)).variance().status is QueryStatus.AVAILABLE
    assert Pareto(sp.Integer(2), sp.Integer(1)).mean().status is QueryStatus.DOES_NOT_EXIST
    assert Pareto(sp.Integer(2), sp.Integer(3)).variance().status is QueryStatus.AVAILABLE


def test_typed_joint_and_conditional_models_reject_bad_semantics():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        variables=(x, y),
        density=sp.exp(-x**2 - y**2) / sp.pi,
        support=sp.ProductSet(sp.S.Reals, sp.S.Reals),
    )
    assert joint.variables == (x, y)
    conditional = ConditionalDistribution(
        variable=x,
        given=(y,),
        density=sp.exp(-(x - y) ** 2) / sp.sqrt(sp.pi),
        support=sp.S.Reals,
        condition=sp.Eq(y, sp.Integer(0)),
    )
    assert conditional.variable == x
    with pytest.raises(ValueError):
        JointDistribution(variables=(x, x), density=sp.Integer(1), support=sp.S.Reals)
    with pytest.raises(ValueError):
        ConditionalDistribution(
            variable=x, given=(x,), density=sp.Integer(1),
            support=sp.S.Reals, condition=sp.true,
        )


def test_monotone_transformation_uses_inverse_jacobian():
    x, y = sp.symbols("x y", real=True)
    source = RandomVariable(
        symbol=x,
        distribution=Exponential(sp.Integer(2), variable=x),
    )
    transformed = transform(source, 3 * x + 1, y)
    assert transformed.distribution.support == sp.Interval(1, sp.oo)
    expected = sp.Rational(2, 3) * sp.exp(-sp.Rational(2, 3) * (y - 1))
    assert sp.simplify(transformed.distribution.density - expected) == 0
    assert transformed.distribution.verify().verification["normalized"] is True


def test_nonmonotone_transformation_detects_missing_branches():
    x, y = sp.symbols("x y", real=True)
    source = RandomVariable(symbol=x, distribution=Normal(sp.Integer(0), sp.Integer(1), variable=x))
    with pytest.raises(ValueError, match="not provably monotone"):
        transform(source, x**2, y)
    with pytest.raises(ValueError, match="lose part"):
        transform(source, x**2, y, inverse_branches=[sp.sqrt(y)])

    transformed = transform(
        source,
        x**2,
        y,
        inverse_branches=[-sp.sqrt(y), sp.sqrt(y)],
    )
    expected = sp.exp(-y / 2) / sp.sqrt(2 * sp.pi * y)
    assert transformed.distribution.support == sp.Interval(0, sp.oo)
    assert sp.simplify(transformed.distribution.density - expected) == 0


def test_distribution_composition_retains_normalization_and_evidence():
    x = sp.Symbol("x", real=True)
    left = Normal(sp.Integer(0), sp.Integer(1), variable=x)
    right = Normal(sp.Integer(2), sp.Integer(1), variable=x)
    mixed = mixture(
        [sp.Rational(1, 4), sp.Rational(3, 4)], [left, right], variable=x)
    assert mixed.verify().verification["normalized"] is True
    truncated = truncate(left, sp.Integer(-1), sp.Integer(1))
    assert truncated.verify().verification["normalized"] is True
    assert expectation(left, x**2).value == 1
    assert kl_divergence(left, left).value == 0


def test_convolution_produces_typed_distribution_without_claiming_exactness():
    x = sp.Symbol("x", real=True)
    first = Normal(sp.Integer(0), sp.Integer(1), variable=x)
    second = Normal(sp.Integer(0), sp.Integer(1), variable=x)
    result = convolve(first, second, variable=x)
    assert result.name == "transformed"
    assert result.input_trust == "symbolic"
    assert sp.simplify(result.variance().value - 2) == 0


def test_adversarial_density_verification_disproves_invalid_claims():
    x = sp.Symbol("x", real=True)
    negative = Distribution(
        name="transformed",
        parameters={},
        variable=x,
        density=-sp.Integer(1),
        support=sp.Interval(0, 1),
    )
    result = negative.verify()
    assert result.status is QueryStatus.AVAILABLE
    assert result.verification == {"normalized": False, "nonnegative": False}
    assert result.claim_evidence["distribution"].proof == []

    over_normalized = negative.model_copy(
        update={"density": sp.Integer(2)})
    assert over_normalized.verify().verification == {
        "normalized": False, "nonnegative": True}


def test_cdf_verification_checks_both_boundaries_and_monotonicity():
    x = sp.Symbol("x", real=True)
    result = Exponential(sp.Integer(2), variable=x).cdf()
    assert result.verification == {
        "derivative_matches_pdf": True,
        "lower_boundary": True,
        "upper_boundary": True,
        "monotone": True,
    }


def test_joint_marginals_bayes_covariance_and_correlation():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        variables=(x, y),
        density=1 + (2 * x - 1) * (2 * y - 1),
        support=sp.ProductSet(sp.Interval(0, 1), sp.Interval(0, 1)),
        conditions=["unit-square Farlie-Gumbel-Morgenstern density"],
    )
    assert joint.verify().verification["normalized"] is True
    assert joint.marginal(x).density == 1
    assert joint.marginal((y, x)).variables == (y, x)
    assert covariance(joint, x, y).value == sp.Rational(1, 36)
    assert correlation(joint, x, y).value == sp.Rational(1, 3)

    conditional = bayes_rule(joint, x, {y: sp.Rational(1, 2)})
    assert conditional.condition == sp.Eq(y, sp.Rational(1, 2))
    assert conditional.density == 1
    assert conditional.verify().verification == {
        "normalized": True, "nonnegative": True}
    assert conditional.mean().value == sp.Rational(1, 2)


def test_joint_failure_semantics_separate_unknown_unsupported_and_nonexistence():
    x, y = sp.symbols("x y", real=True)
    unsupported = JointDistribution(
        variables=(x, y),
        density=sp.exp(-x**2 - y**2) / sp.pi,
        support=sp.S.Reals,
    )
    assert unsupported.verify().status is QueryStatus.UNSUPPORTED
    assert unsupported.covariance(x, y).status is QueryStatus.UNSUPPORTED

    unresolved = JointDistribution(
        variables=(x, y),
        density=sp.Function("f")(x, y),
        support=sp.ProductSet(sp.Interval(0, 1), sp.Interval(0, 1)),
    )
    assert unresolved.covariance(x, y).status is QueryStatus.UNKNOWN

    cauchy_product = JointDistribution(
        variables=(x, y),
        density=1 / (sp.pi**2 * (1 + x**2) * (1 + y**2)),
        support=sp.ProductSet(sp.S.Reals, sp.S.Reals),
    )
    assert cauchy_product.verify().verification["normalized"] is True
    assert cauchy_product.covariance(x, y).status is QueryStatus.DOES_NOT_EXIST
    assert cauchy_product.correlation(x, y).status is QueryStatus.DOES_NOT_EXIST


def test_continuous_order_statistics_are_normalized_and_validate_indices():
    x = sp.Symbol("x", real=True)
    uniform = Uniform(sp.Integer(0), sp.Integer(1), variable=x)
    median = order_statistic(uniform, 3, 2)
    assert sp.simplify(median.density - 6 * x * (1 - x)) == 0
    assert median.support == uniform.support
    assert median.verify().verification == {
        "normalized": True, "nonnegative": True}
    with pytest.raises(ValueError):
        uniform.order_statistic(3, 0)
    with pytest.raises(ValueError):
        uniform.order_statistic(2, 3)


def test_transformation_and_multivariate_serialization_retain_domain_evidence():
    x, y = sp.symbols("x y", real=True)
    source = RandomVariable(
        symbol=x,
        distribution=Uniform(sp.Integer(0), sp.Integer(1), variable=x),
    )
    transformed = transform(source, x**2, y, inverse_branches=[sp.sqrt(y)])
    encoded = transformed.distribution.model_dump(mode="json")
    assert encoded["support"] == "Interval(0, 1)"
    assert encoded["source_support"] == "Interval(0, 1)"
    assert encoded["inverse_branches"] == ["sqrt(y)"]
    assert encoded["jacobians"] == ["1/(2*Abs(sqrt(y)))"]
    assert encoded["conditions"]

    joint = JointDistribution(
        variables=(x, y),
        density=sp.Integer(1),
        support=sp.ProductSet(sp.Interval(0, 1), sp.Interval(0, 1)),
        conditions=["closed unit square"],
    )
    dumped = joint.model_dump(mode="json")
    assert dumped["variables"] == ["x", "y"]
    assert dumped["support"] == "ProductSet(Interval(0, 1), Interval(0, 1))"
    assert dumped["conditions"] == ["closed unit square"]
