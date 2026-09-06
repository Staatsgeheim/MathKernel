# =============================================================================
# MathKernel - focused conservative complex-analysis tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import mpmath as mp
import sympy as sp
import pytest

from mathkernel.complex_analysis import (
    ArgumentPrincipleResult,
    BranchConvention,
    ComplexAnalysisEngine,
    ComplexDomain,
    ComplexFunction,
    ConformalMapResult,
    Contour,
    ResultStatus,
    Singularity,
)


z = sp.Symbol("z")
engine = ComplexAnalysisEngine()


def test_residue_theorem_differentially_matches_numeric_contour_quadrature():
    vertices = [
        -1 - sp.I, 1 - sp.I, 1 + sp.I, -1 + sp.I, -1 - sp.I,
    ]
    contour = Contour(vertices=vertices, orientation="ccw")
    result = engine.contour_integral(
        ComplexFunction(expression=1 / z, variable=z),
        contour,
        [Singularity(point=sp.S.Zero, kind="pole", order=1)],
        singularities_accounted_for=True,
    )
    numeric = 0j
    for start, end in zip(vertices, vertices[1:]):
        a, b = complex(start), complex(end)
        delta = b - a
        numeric += complex(mp.quad(
            lambda parameter: delta / (a + delta * parameter),
            [0, 1],
        ))
    assert complex(result.integral.evalf()) == pytest.approx(
        numeric, rel=1e-10, abs=1e-10)


def test_simple_pole_residue_has_independent_agreeing_checks():
    function = ComplexFunction(expression=sp.exp(z) / z, variable=z)

    result = engine.residue(function, sp.S.Zero)

    assert result.status == ResultStatus.VERIFIED
    assert sp.simplify(result.value - 1) == 0
    assert result.pole_order == 1
    assert result.checks["defining_limit"] == result.checks["laurent_coefficient"]
    assert result.evidence.conservative_trust() == "symbolic"
    assert result.model_dump(mode="json")["value"] == "1"


def test_double_pole_residue_has_independent_agreeing_checks():
    function = ComplexFunction(expression=sp.exp(z) / z**2, variable=z)

    result = engine.residue(function, sp.S.Zero)

    assert result.status == ResultStatus.VERIFIED
    assert sp.simplify(result.value - 1) == 0
    assert result.pole_order == 2
    assert result.checks["defining_limit"] == result.checks["laurent_coefficient"]


def test_incompatible_log_branches_are_rejected():
    left = BranchConvention(logarithm_branch=0)
    right = BranchConvention(logarithm_branch=1)

    result = engine.branch_compatible(sp.log(z), left, right)

    assert result.status == ResultStatus.ERROR
    assert result.value is False
    assert any("logarithm_branch" in diagnostic for diagnostic in result.diagnostics)


def test_residue_theorem_on_ccw_polygon_records_accounting():
    function = ComplexFunction(expression=1 / z, variable=z)
    contour = Contour(
        vertices=[
            -1 - sp.I,
            1 - sp.I,
            1 + sp.I,
            -1 + sp.I,
            -1 - sp.I,
        ],
        orientation="ccw",
    )
    pole = Singularity(point=sp.S.Zero, kind="pole", order=1)

    result = engine.contour_integral(
        function,
        contour,
        singularities=[pole],
        singularities_accounted_for=True,
    )

    assert result.status == ResultStatus.VERIFIED
    assert sp.simplify(result.integral - 2 * sp.pi * sp.I) == 0
    assert result.winding_numbers["0"] == 1
    assert result.enclosed_singularities == [pole]
    assert result.residues["0"] == 1
    assert result.trust == "symbolic"


def test_missing_singularity_accounting_stays_unknown():
    function = ComplexFunction(expression=1 / z, variable=z)
    contour = Contour(
        vertices=[
            -1 - sp.I,
            1 - sp.I,
            1 + sp.I,
            -1 + sp.I,
            -1 - sp.I,
        ],
    )

    result = engine.contour_integral(function, contour, singularities=[])

    assert result.status == ResultStatus.UNKNOWN
    assert result.integral is None
    assert result.trust == "unknown"


def test_numeric_input_ancestry_caps_complex_result_trust():
    function = ComplexFunction(
        expression=sp.Float("0.7") / z,
        variable=z,
        input_trust="numeric",
    )
    result = engine.residue(function, sp.S.Zero)
    assert result.status == ResultStatus.VERIFIED
    assert result.trust == "numeric"
    assert result.evidence.conservative_trust() == "numeric"


def test_zero_singularity_and_local_conformal_operations_are_typed():
    polynomial = ComplexFunction(expression=z**2 - 1, variable=z)
    zeros = engine.zeros(polynomial)
    assert zeros.status == ResultStatus.VERIFIED
    assert zeros.value == sp.FiniteSet(-1, 1)
    assert engine.conformal_at(polynomial, sp.Integer(1)).value is True
    assert engine.conformal_at(polynomial, sp.Integer(0)).value is False

    essential = ComplexFunction(expression=sp.exp(1 / z), variable=z)
    classified = engine.classify_singularity(essential, sp.S.Zero)
    assert classified.status == ResultStatus.VERIFIED
    assert classified.singularity.kind == "essential"


def _square(radius=2):
    radius = sp.Integer(radius)
    return Contour(vertices=[
        -radius - radius * sp.I,
        radius - radius * sp.I,
        radius + radius * sp.I,
        -radius + radius * sp.I,
        -radius - radius * sp.I,
    ], orientation="ccw")


def test_argument_principle_counts_zeros_minus_poles_with_multiplicity():
    function = ComplexFunction(expression=(z - 1) ** 2 / z, variable=z)

    result = engine.argument_principle(function, _square())

    assert isinstance(result, ArgumentPrincipleResult)
    assert result.status == ResultStatus.VERIFIED
    assert result.zero_count == 2
    assert result.pole_count == 1
    assert result.zero_minus_pole == 1
    assert result.value == 1
    assert result.zero_multiplicities == {"1": 2}
    assert result.pole_multiplicities == {"0": 1}
    assert sp.simplify(
        result.logarithmic_derivative_integral - 2 * sp.pi * sp.I) == 0
    assert len(result.contour_obligations) == 4


def test_argument_principle_rejects_boundary_zero_and_branch_expression():
    boundary = ComplexFunction(expression=z - 2, variable=z)
    boundary_result = engine.argument_principle(boundary, _square())
    assert boundary_result.status == ResultStatus.ERROR
    assert any("undefined" in item for item in boundary_result.diagnostics)

    cut = sp.Segment(sp.Point(-2, 0), sp.Point(0, 0))
    branch = BranchConvention(cuts=[cut])
    logarithm = ComplexFunction(expression=sp.log(z), variable=z, branch=branch)
    branch_result = engine.argument_principle(logarithm, _square())
    assert branch_result.status == ResultStatus.UNSUPPORTED
    assert branch_result.branch_cuts == [cut]


def test_identity_analytic_continuation_requires_provable_overlap_and_analyticity():
    source = ComplexDomain(variable=z, region=sp.Interval(-1, 1))
    target = ComplexDomain(variable=z, region=sp.Interval(0, 2))
    function = ComplexFunction(
        expression=z**2 + 1, variable=z, domain=source, branch=source.branch)

    result = engine.analytic_continuation(function, target)

    assert result.status == ResultStatus.VERIFIED
    assert result.identity_established is True
    assert result.continuation.expression == function.expression
    assert result.continuation.domain == target
    assert result.overlap == sp.Interval(0, 1)


def test_analytic_continuation_never_fabricates_branch_or_disjoint_continuation():
    branch = BranchConvention(logarithm_branch=0)
    source = ComplexDomain(variable=z, region=sp.Interval(1, 2), branch=branch)
    target = ComplexDomain(variable=z, region=sp.Interval(3, 4), branch=branch)
    function = ComplexFunction(
        expression=sp.log(z), variable=z, domain=source, branch=branch)

    disjoint = engine.analytic_continuation(function, target)
    assert disjoint.status == ResultStatus.UNSUPPORTED
    assert disjoint.continuation is None

    overlapping = ComplexDomain(
        variable=z, region=sp.Interval(sp.Rational(3, 2), 3), branch=branch)
    branch_result = engine.analytic_continuation(function, overlapping)
    assert branch_result.status == ResultStatus.UNSUPPORTED
    assert branch_result.continuation is None


def test_conformal_map_returns_typed_image_and_domain_nondegeneracy():
    domain = ComplexDomain(variable=z, region=sp.Interval(1, 2))
    function = ComplexFunction(
        expression=z**2, variable=z, domain=domain, branch=domain.branch)

    result = engine.conformal_map(function)

    assert isinstance(result, ConformalMapResult)
    assert result.status == ResultStatus.VERIFIED
    assert result.mapped_expression == z**2
    assert result.derivative == 2 * z
    assert result.critical_points == sp.S.EmptySet
    assert result.nondegenerate_on_domain is True
    assert result.image_domain is not None

    degenerate_domain = ComplexDomain(variable=z, region=sp.Interval(-1, 1))
    degenerate = engine.conformal_map(function, degenerate_domain)
    assert degenerate.status == ResultStatus.VERIFIED
    assert degenerate.nondegenerate_on_domain is False


def test_contour_failure_retains_declared_cuts_and_enclosed_singularities():
    cut = sp.Segment(sp.Point(-2, 0), sp.Point(0, 0))
    branch = BranchConvention(cuts=[cut])
    function = ComplexFunction(expression=sp.log(z), variable=z, branch=branch)
    branch_point = Singularity(point=sp.S.Zero, kind="branch_point")

    result = engine.contour_integral(
        function, _square(), [branch_point], singularities_accounted_for=True)

    assert result.status == ResultStatus.ERROR
    assert result.branch_cuts == [cut]
    assert result.enclosed_singularities == [branch_point]
    assert result.winding_numbers == {"0": 1}


def test_complex_results_serialize_and_restrict_non_sympy_inputs():
    result = engine.argument_principle(
        ComplexFunction(expression=(z - 1) / z, variable=z), _square())
    dumped = result.model_dump(mode="json")
    assert dumped["zero_minus_pole"] == 0
    assert dumped["logarithmic_derivative_integral"] == "0"
    assert dumped["enclosed_zeros"] == ["1"]
    assert dumped["enclosed_poles"][0]["point"] == "0"

    with pytest.raises(TypeError):
        ComplexDomain(variable=z, region="not a SymPy set")
    with pytest.raises(TypeError):
        engine.conformal_at(
            ComplexFunction(expression=z, variable=z), "not a SymPy expression")
