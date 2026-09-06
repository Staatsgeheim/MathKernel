# =============================================================================
# MathKernel - kernel
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import functools
import importlib
import json
import platform
import sys
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import sympy as sp
from pydantic import BaseModel, TypeAdapter
from mathkernel_artifacts import (
    ComputationEvidence,
    EvidenceBundle,
    extract_evidence,
)

from .bigint import decimal_to_int
from .capabilities import Capability, CapabilityRouter
from .codegen import EMITTERS, CodegenError, build_artifact
from .codeverify import execute_artifact, symbolic_roundtrip, typecheck_artifact
from .collatz import sieve_cycle_classes
from .context import infer_context
from .coordinator import VerificationCoordinator
from .derivation import trace_dag
from .engines import LeanEngine, SymPyEngine, Z3Engine, run_with_timeout, symbol_env
from .execution import _TRUST_RANK, ObligationExecutor
from .integers import IntegerEngine
from .intervals import IntervalEngine
from .models import (
    Assumption,
    DerivationStep,
    Expr,
    IntegerNode,
    MathContext,
    MathResult,
    MembershipNode,
    NaryNode,
    Obligation,
    ObligationExecution,
    OperationStatus,
    PlanExecution,
    ProblemPlan,
    QuantifierNode,
    RationalNode,
    RelationNode,
    SetNode,
    SetOpNode,
    SymbolNode,
    ResultStatus,
    TrustLevel,
    UnaryNode,
    VerificationStatus,
)
from .parallel import resolve_workers
from .parser import ambiguity_diagnostics, parse_math
from .reasoning import _walk as _scan_ir
from .reasoning import plan_problem
from .rendering import render_expr
from .settings import Settings, apply_setting_updates, setting_schema, yolo_mode
from .safe_sympy import decode_srepr
from .solution import serialize_solution_set
from .structure import extract_constraints, infer_capabilities, suggest_structure


from ._version import __version__ as MATHKERNEL_VERSION


_TYPED_SOURCE_FIELDS = {
    "GeneralizedLinearModel": "sample_id",
    "GLMFit": "model_id",
    "SurvivalDataset": "sample_id",
    "KaplanMeierEstimate": "dataset_id",
    "CoxProportionalHazardsModel": "dataset_id",
    "CoxPHFit": "model_id",
    "TimeSeriesDataset": "sample_id",
    "TimeSeriesAnalysis": "dataset_id",
    "TimeSeriesModel": "dataset_id",
    "TimeSeriesFit": "model_id",
    "TimeSeriesForecast": "fit_id",
    "FiniteDimensionalDistribution": "process_id",
    "GaussianProcessPosterior": "process_id",
    "CTMCTransition": "process_id",
    "SDESimulation": "sde_id",
    "SDEConvergenceStudy": "sde_id",
    "PDEClassification": "problem_id",
    "PDECompatibilityReport": "problem_id",
    "WeakForm": "problem_id",
    "FEMMesh": "weak_form_id",
    "ReferenceElement": "mesh_id",
    "BasisFunctionSet": "reference_element_id",
    "QuadratureRule": "reference_element_id",
    "FiniteElementSpace": "mesh_id",
    "AssembledSystem": "finite_element_space_id",
    "FEMSolution": "assembled_system_id",
    "FEMErrorEstimate": "solution_id",
    "RefinementMarking": "error_estimate_id",
    "RefinedMesh": "marking_id",
    "MeshTransfer": "marking_id",
    "FEMConvergenceObservation": "fine_estimate_id",
}


def _encode_typed_value(value):
    """Encode trusted typed-domain models into deterministic JSON data."""
    if isinstance(value, sp.Basic):
        return {"__sympy_srepr__": sp.srepr(value)}
    if isinstance(value, BaseModel):
        return {
            "__pydantic_model__": (
                f"{value.__class__.__module__}:{value.__class__.__name__}"),
            "fields": {
                name: _encode_typed_value(getattr(value, name))
                for name in type(value).model_fields
            },
        }
    if isinstance(value, Fraction):
        return {"__fraction__": str(value)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_typed_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [
            _encode_typed_value(item)
            for item in sorted(value, key=str)
        ]}
    if isinstance(value, list):
        return [_encode_typed_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_typed_value(item)
            for key, item in value.items()
        }
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _decode_typed_value(value):
    """Decode only MathKernel-owned model types from the local store."""
    if isinstance(value, list):
        return [_decode_typed_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__sympy_srepr__" in value:
        return decode_srepr(value["__sympy_srepr__"])
    if "__fraction__" in value:
        return Fraction(value["__fraction__"])
    if "__tuple__" in value:
        return tuple(_decode_typed_value(item) for item in value["__tuple__"])
    if "__set__" in value:
        return set(_decode_typed_value(item) for item in value["__set__"])
    model_ref = value.get("__pydantic_model__")
    if model_ref:
        module_name, class_name = model_ref.split(":", 1)
        # Persistence is an untrusted input boundary.  Do not instantiate an
        # arbitrary Pydantic class merely because it lives in a MathKernel
        # module: future validators could otherwise become a deserialisation
        # gadget.  Persisted model types are an explicit wire-format allowlist.
        allowed_models = {
            "mathkernel.integral_transforms": frozenset({
                "TransformConvention", "TransformProblem", "VerifiedCheck",
                "TransformResult",
            }),
            "mathkernel.complex_analysis": frozenset({
                "BranchConvention", "ComplexDomain", "Contour",
                "ComplexFunction", "Singularity", "ComplexResult",
                "AnalyticityResult", "SingularityResult", "ResidueResult",
                "LaurentSeriesResult", "WindingNumberResult",
                "ContourIntegralResult", "ArgumentPrincipleResult",
                "AnalyticContinuationResult", "ConformalMapResult",
            }),
            "mathkernel.continuous_probability": frozenset({
                "ProbabilityResult", "Distribution", "RandomVariable",
                "JointDistribution", "ConditionalDistribution",
            }),
            "mathkernel.graph_theory": frozenset({
                "GraphEdge", "Graph", "DirectedGraph", "WeightedGraph",
                "MultiGraph", "SearchLimits", "GraphResult",
                "TraversalResult", "ComponentsResult", "ShortestPathResult",
                "SpanningTreeResult", "EdgeFlow", "FlowVerification",
                "MaxFlowResult", "MatchingResult", "EulerResult",
                "ColoringResult", "TopologicalSortResult", "CycleResult",
                "CentralityResult", "IsomorphismResult",
            }),
            "mathkernel.combinatorics": frozenset({
                "CombinatoricsResult", "CountResult", "GenerationResult",
                "LinearRecurrence", "RationalGeneratingFunction",
                "GeneratingFunction", "CoefficientResult",
                "GfConversionResult", "VerificationResult",
                "CombinatorialClass",
            }),
            "mathkernel.finite_groups": frozenset({
                "GroupResult", "OrderResult", "SubgroupInfo",
                "SubgroupResult", "SubgroupsResult", "CosetsResult",
                "NormalityResult", "QuotientResult",
                "ConjugacyClassesResult", "OrbitsResult",
                "MembershipResult", "StabilizerChainResult",
                "AbelianAnalysisResult", "FiniteGroup", "GroupAction",
                "GroupHomomorphism", "PermutationGroup",
                "FiniteAbelianGroup",
            }),
            "mathkernel.signal_processing": frozenset({"ContinuousSignal", "DiscreteSignal", "Spectrum", "Filter", "FilterDesign"}),
            "mathkernel.control_systems": frozenset({"TransferFunction", "StateSpaceSystem", "DiscreteControlSystem"}),
            "mathkernel.control_analysis": frozenset({"ZeroPoleGain", "FrequencyResponse", "RootLocus", "TimeResponse"}),
            "mathkernel.control_design": frozenset({"TransferMatrix"}),
            "mathkernel.streaming": frozenset({"FilterState"}),
            "mathkernel.optimization_proofs": frozenset({"MILPCertificate", "MILPProofNode"}),
            "mathkernel.control_sequential": frozenset({"FiniteHorizonLQR", "KalmanState"}),
            "mathkernel.mpc": frozenset({"MPCPlan"}),
            "mathkernel.riccati": frozenset({"RiccatiCertificate"}),
            "mathkernel.conic": frozenset({"ConicProblem", "ConicCertificate", "ConeBlock"}),
            "mathkernel.quadratic_constraints": frozenset({"QuadraticallyConstrainedProblem", "QuadraticCertificate", "QuadraticConstraint"}),
            "mathkernel.optimization": frozenset({"OptimizationProblem", "OptimizationCertificate"}),
            "mathkernel.differential_geometry": frozenset({
                "Manifold", "Chart", "Metric", "GeometryTensor",
                "Connection", "GeodesicSystem", "CoordinateMap", "JacobianMap",
                "TensorField", "FormTerm", "DifferentialForm",
            }),
            "mathkernel.computational_geometry": frozenset({
                "Point", "PointSet", "Polygon", "HalfSpace", "Polytope",
                "Triangulation", "VoronoiRay", "VoronoiDiagram",
            }),
            "mathkernel.algebraic_topology": frozenset({
                "SimplicialComplex", "CubicalComplex", "ChainComplex",
                "HomologyGroup", "Homology",
            }),
            "mathkernel.statistical_inference": frozenset({
                "StatisticalSample", "VariableSummary", "DescriptiveSummary",
                "CovarianceMatrix", "EmpiricalDistribution",
                "GeneralizedLinearModel", "GLMFit",
                "NonparametricTestResult", "ResamplingResult",
                "SurvivalDataset", "KaplanMeierEstimate",
                "CoxProportionalHazardsModel", "CoxPHFit",
                "TimeSeriesDataset", "TimeSeriesAnalysis", "TimeSeriesModel",
                "TimeSeriesFit", "TimeSeriesForecast",
            }),
            "mathkernel.stochastic_processes": frozenset({
                "PoissonProcess", "WienerProcess", "GaussianProcess",
                "ContinuousTimeMarkovChain", "FiniteDimensionalDistribution",
                "GaussianProcessPosterior", "CTMCTransition",
            }),
            "mathkernel.stochastic_differential_equations": frozenset({
                "StochasticDifferentialEquation", "SDESimulation",
                "SDEConvergenceStudy",
            }),
            "mathkernel.partial_differential_equations": frozenset({
                "PDETerm", "PDEEquation", "PDEBoundaryCondition",
                "PDEInitialCondition", "PDEProblem", "PDEClassification",
                "PDECompatibilityCheck", "PDECompatibilityReport",
            }),
            "mathkernel.weak_forms": frozenset({
                "PDEFunctionSpace", "PDEMeasure", "WeakIntegralTerm",
                "IntegrationByPartsStep", "WeakForm",
            }),
            "mathkernel.finite_elements": frozenset({
                "FEMMesh", "ReferenceElement", "BasisFunctionSet",
                "QuadratureRule", "FiniteElementSpace",
            }),
            "mathkernel.finite_element_assembly": frozenset({
                "SparseMatrixEntry", "LocalElementContribution",
                "NaturalBoundaryContribution", "EssentialConstraint",
                "AssembledSystem", "FEMSolution",
            }),
            "mathkernel.finite_element_adaptivity": frozenset({
                "CellErrorIndicator", "FEMErrorEstimate", "RefinementMarking",
                "RefinedMesh", "MeshTransfer", "FEMConvergenceObservation",
            }),
            "mathkernel.finite_algebra": frozenset({
                "FiniteAlgebraResult", "FiniteRingSpec", "RingElement",
                "FiniteFieldSpec", "FieldElement", "ModulePresentation",
                "AbelianGroupDecomposition",
            }),
            "mathkernel_artifacts.evidence": frozenset({
                "ComputationEvidence", "ProofEvidence",
                "CertificateEvidence", "NumericalEvidence",
                "ModelEvidence", "EmpiricalEvidence", "EvidenceBundle",
            }),
        }
        if class_name not in allowed_models.get(module_name, frozenset()):
            raise ValueError(f"stored model type is not allowed: {model_ref}")
        model_type = getattr(importlib.import_module(module_name), class_name, None)
        if (
            not isinstance(model_type, type)
            or not issubclass(model_type, BaseModel)
        ):
            raise ValueError(f"stored model type is not a Pydantic model: {model_ref}")
        fields = {
            name: _decode_typed_value(item)
            for name, item in value["fields"].items()
        }
        return model_type.model_validate(fields)
    return {
        key: _decode_typed_value(item)
        for key, item in value.items()
    }


_CHILD_ATTRS = ("arg", "left", "right", "real", "imag", "lower", "upper", "minimal_polynomial",
                "element", "set", "domain", "body")


def _arb_available() -> bool:
    try:
        from .certified import arb_available
        return arb_available()
    except Exception:
        return False


def _count_nodes(node: Expr) -> int:
    children = getattr(node, "args", None) or []
    total = 1 + sum(_count_nodes(c) for c in children)
    elements = getattr(node, "elements", None) or []
    total += sum(_count_nodes(e) for e in elements)
    for attr in _CHILD_ATTRS:
        child = getattr(node, attr, None)
        if child is not None:
            total += _count_nodes(child)
    return total


def _substitute_ir(node: Expr, replacements: dict[str, Expr]) -> Expr:
    if isinstance(node, SymbolNode) and node.name in replacements:
        return replacements[node.name]
    bound = getattr(node, "variable", None)  # QuantifierNode/BinderNode bind their variable
    if bound is not None and bound in replacements:
        replacements = {k: v for k, v in replacements.items() if k != bound}
    children = getattr(node, "args", None)
    if children is not None:
        node.args = [_substitute_ir(c, replacements) for c in children]
    elements = getattr(node, "elements", None)
    if elements is not None:
        node.elements = [_substitute_ir(e, replacements) for e in elements]
    for attr in _CHILD_ATTRS:
        child = getattr(node, attr, None)
        if child is not None:
            setattr(node, attr, _substitute_ir(child, replacements))
    return node


def _symbol_names(*irs: Expr) -> set[str]:
    names: set[str] = set()
    for ir in irs:
        _scan_ir(ir, names, set(), set())
    return names


_INFINITY_ALIASES = {"oo", "inf", "+oo", "+inf", "infinity", "+infinity"}
_NEG_INFINITY_ALIASES = {"-oo", "-inf", "-infinity"}


def _summarize_data(data) -> dict | str:
    """Compact per-key summary for an over-budget payload."""
    if not isinstance(data, dict):
        return f"<{type(data).__name__}>"
    summary = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            summary[key] = f"<list of {len(value)}>"
        elif isinstance(value, dict):
            summary[key] = f"<dict with {len(value)} keys>"
        else:
            text = str(value)
            summary[key] = text if len(text) <= 120 else text[:117] + "..."
    return summary


# Public pre/postconditions live outside the mathematical facade.
from .contracts import apply_execution_contract as _apply_output_budget
from .contracts import (finalize_step, persist_expression, expression_trust,
                        current_frame)
from .output_policy import enforce_output_budget as _enforce_output_budget


class MathKernel:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.sympy = SymPyEngine(timeout_seconds=self.settings.solver_timeout_seconds)
        self.z3 = Z3Engine(timeout_ms=self.settings.z3_timeout_ms)
        self.lean = LeanEngine(executable=self.settings.lean_binary,
                               timeout=self.settings.lean_timeout_seconds)
        self.interval = IntervalEngine()
        self.integer = IntegerEngine()
        self.router = CapabilityRouter([self.sympy, self.z3, self.lean, self.interval, self.integer])
        for capability in (
            Capability(
                name="transform.apply", domain="transform", operation="apply",
                input_types=("TransformProblem",), output_types=("TransformResult",),
                evidence=("symbolic", "symbolic_identity"),
                engines=("integral_transforms", "sympy"),
                cost_dimensions=("expression_size", "solver_time"),
                handler="module:integral_transforms",
                parameter_schema={"verify": "boolean?"},
            ),
            Capability(
                name="probability.query", domain="probability", operation="query",
                input_types=("Distribution",), output_types=("DistributionResult",),
                evidence=("symbolic", "normalization_check", "cdf_identity"),
                engines=("continuous_probability", "sympy"),
                cost_dimensions=("expression_size", "integration_complexity",
                                 "solver_time"),
                handler="module:continuous_probability",
            ),
        ):
            self.router.registry.register(capability)
        complex_result_types = {
            "argument_principle": "ArgumentPrincipleResult",
            "analytic_continuation": "AnalyticContinuationResult",
            "conformal_map": "ConformalMapResult",
        }
        probability_parameters = {
            "query": {"query": "string", "point": "MathIR expression?",
                      "order": "integer?"},
            "pdf": {"point": "MathIR expression?"},
            "cdf": {"point": "MathIR expression?"},
            "survival": {"point": "MathIR expression?"},
            "quantile": {"point": "MathIR expression"},
            "moment": {"order": "integer"},
            "expectation": {"expression_id": "Expression object id"},
            "truncate": {"lower": "MathIR expression",
                         "upper": "MathIR expression"},
            "convolve": {"other_id": "Distribution object id"},
            "cross_entropy": {"other_id": "Distribution object id"},
            "kl_divergence": {"other_id": "Distribution object id"},
            "order_statistic": {"sample_size": "integer", "order": "integer"},
        }
        for operation in (
            "pdf", "cdf", "survival", "quantile", "mean", "variance",
            "moment", "mgf", "characteristic_function", "entropy", "verify",
            "expectation", "truncate", "convolve", "cross_entropy",
            "kl_divergence", "order_statistic",
        ):
            self.router.registry.register(Capability(
                name=f"probability.{operation}",
                domain="probability",
                operation=operation,
                input_types=("Distribution",),
                output_types=("ProbabilityResult",),
                evidence=("symbolic", "normalization_check", "cdf_identity"),
                engines=("continuous_probability", "sympy"),
                cost_dimensions=("expression_size", "integration_complexity",
                                 "solver_time"),
                handler="module:continuous_probability",
                parameter_schema=probability_parameters.get(operation, {}),
            ))
        complex_parameters = {
            "zeros": {"region": "set specification?"},
            "classify_singularity": {"point": "MathIR expression"},
            "residue": {"point": "MathIR expression"},
            "laurent_series": {"point": "MathIR expression", "order": "integer"},
            "conformal_at": {"point": "MathIR expression"},
            "contour_integral": {
                "contour_id": "Contour object id",
                "singularities_accounted_for": "boolean",
            },
            "argument_principle": {"contour_id": "Contour object id"},
            "analytic_continuation": {"domain_id": "ComplexDomain object id"},
            "conformal_map": {"domain_id": "ComplexDomain object id"},
        }
        for operation in (
            "derivative", "analyticity", "zeros", "singularities",
            "classify_singularity", "residue", "laurent_series",
            "conformal_at", "contour_integral", "argument_principle",
            "analytic_continuation", "conformal_map",
        ):
            input_types = ("ComplexFunction",)
            evidence = (
                ("symbolic", "residue_certificate")
                if operation == "contour_integral"
                else ("symbolic", "defining_identity")
            )
            self.router.registry.register(Capability(
                name=f"complex.{operation}",
                domain="complex",
                operation=operation,
                input_types=input_types,
                output_types=(complex_result_types.get(
                    operation, "ComplexResult"),),
                evidence=evidence,
                engines=("complex_analysis", "sympy"),
                cost_dimensions=("expression_size", "singularity_count",
                                 "solver_time"),
                handler="module:complex_analysis",
                parameter_schema=complex_parameters.get(operation, {}),
            ))
        for object_type, operations in (
            ("JointDistribution", (
                "verify", "marginal", "condition", "bayes",
                "covariance", "correlation", "order_statistic",
            )),
            ("ConditionalDistribution", (
                "verify", "pdf", "cdf", "mean", "variance",
            )),
        ):
            for operation in operations:
                self.router.registry.register(Capability(
                    name=f"probability.{object_type}.{operation}",
                    domain="probability",
                    operation=operation,
                    input_types=(object_type,),
                    output_types=("ProbabilityResult",),
                    evidence=(
                        "symbolic", "normalization_check",
                        "support_nonnegativity",
                    ),
                    engines=("continuous_probability", "sympy"),
                    cost_dimensions=(
                        "dimension", "expression_size",
                        "integration_complexity", "solver_time",
                    ),
                    handler="module:continuous_probability",
                    parameter_schema={
                        "marginal": {"variables": "symbol or list[symbol]"},
                        "condition": {
                            "condition": "MathIR relation",
                            "conditioned_variables": "list[symbol]",
                        },
                        "bayes": {"event": "MathIR relation",
                                  "given": "MathIR relation"},
                        "covariance": {"left": "symbol", "right": "symbol"},
                        "correlation": {"left": "symbol", "right": "symbol"},
                        "order_statistic": {
                            "variable": "symbol", "sample_size": "integer",
                            "order": "integer",
                        },
                        "pdf": {"point": "MathIR expression?"},
                        "cdf": {"point": "MathIR expression?"},
                    }.get(operation, {}),
                ))
        self.router.registry.register(Capability(
            name="complex.winding_number", domain="complex",
            operation="winding_number", input_types=("Contour",),
            output_types=("WindingNumberResult",),
            evidence=("exact", "combinatorial_certificate"),
            engines=("complex_analysis",),
            cost_dimensions=("contour_vertices",),
            handler="module:complex_analysis",
            parameter_schema={"point": "MathIR expression"},
        ))
        self.router.registry.register(Capability(
            name="probability.transform", domain="probability",
            operation="transform", input_types=("RandomVariable",),
            output_types=("RandomVariable",),
            evidence=("symbolic", "jacobian_identity"),
            engines=("continuous_probability", "sympy"),
            cost_dimensions=("expression_size", "inverse_branch_count",
                             "solver_time"),
            handler="module:continuous_probability",
            parameter_schema={
                "expression_id": "Expression object id",
                "inverse_branch": "integer?",
            },
        ))
        self.router.registry.register(Capability(
            name="probability.mixture", domain="probability",
            operation="mixture", input_types=("Distribution",),
            output_types=("Distribution",),
            evidence=("symbolic", "normalization_check"),
            engines=("continuous_probability", "sympy"),
            cost_dimensions=("component_count", "expression_size",
                             "solver_time"),
            handler="module:continuous_probability",
            parameter_schema={
                "other_ids": "list[Distribution object id]",
                "weights": "list[MathIR expression]",
            },
        ))
        self.router.registry.register(Capability(
            name="composition.distribution_integral_transform",
            domain="composition",
            operation="integral_transform",
            input_types=("Distribution",),
            output_types=("TransformResult",),
            evidence=("symbolic", "symbolic_identity", "roc_nonempty_consistency"),
            engines=("integral_transforms", "sympy"),
            cost_dimensions=("expression_size", "integration_complexity",
                             "solver_time"),
            handler="composition:distribution_integral_transform",
            parameter_schema={
                "transform": "laplace | fourier | mellin",
                "convention": "explicit convention name",
                "transform_variable": "symbol?",
                "assumptions": "list[MathIR predicate]?",
                "verify": "boolean?",
            },
        ))
        presentation_capabilities = (
            Capability(
                name="projection.catalog", domain="projection",
                operation="catalog",
                output_types=("ProjectionCatalog",),
                evidence=("exact",), engines=("projection",),
                cost_dimensions=("catalog_size",),
                handler="kernel:projection_catalog",
            ),
            Capability(
                name="projection.create", domain="projection",
                operation="create",
                input_types=("dict",), output_types=("MultimodalProjection",),
                evidence=("exact",), engines=("projection",),
                cost_dimensions=("payload_size",),
                handler="kernel:projection_create",
                parameter_schema={
                    "kind": "projection kind", "payload": "dict",
                    "trust": "trust level?", "parameters": "dict?",
                    "information_loss": "list[str]?",
                },
            ),
            Capability(
                name="projection.describe", domain="projection",
                operation="describe",
                input_types=("MultimodalProjection",), output_types=("dict",),
                evidence=("exact",), engines=("projection",),
                cost_dimensions=("payload_size",),
                handler="kernel:projection_describe",
                parameter_schema={"projection_id": "Projection object id"},
            ),
            Capability(
                name="visualization.create", domain="visualization",
                operation="create",
                input_types=("dict", "MathResult"),
                output_types=("VisualizationDocument",),
                evidence=("numeric", "exact", "interval_certified"),
                engines=("viz",),
                cost_dimensions=("payload_size", "block_count"),
                handler="kernel:viz_create",
                parameter_schema={
                    "view": "auto|plot2d|heatmap|dag|...",
                    "renderer": "auto|svg|threejs",
                            "data": "inline block spec?", "object_id": "str?",
                },
            ),
            Capability(
                name="visualization.projection", domain="visualization",
                operation="projection",
                input_types=("MultimodalProjection",),
                output_types=("VisualizationDocument",),
                evidence=("numeric", "exact", "interval_certified"),
                engines=("viz",),
                cost_dimensions=("payload_size",),
                handler="kernel:viz_projection",
                parameter_schema={"projection_id": "Projection object id"},
            ),
            Capability(
                name="visualization.export", domain="visualization",
                operation="export",
                input_types=("VisualizationDocument",), output_types=("ArtifactFile",),
                evidence=("numeric", "exact", "interval_certified"),
                engines=("viz",),
                cost_dimensions=("payload_size",),
                handler="kernel:viz_export",
                parameter_schema={"viz_id": "str", "path": "str", "mode": "portable?"},
            ),
            Capability(
                name="sonification.create", domain="sonification",
                operation="create",
                input_types=("list[float]",),
                output_types=("SonificationDocument",),
                evidence=("numeric",), engines=("sonify",),
                cost_dimensions=("sequence_length",),
                handler="kernel:sonify_create",
                parameter_schema={"data": "list[float]", "mode": "auto|harmonic|scan?"},
            ),
            Capability(
                name="sonification.compare", domain="sonification",
                operation="compare",
                input_types=("list[float]", "list[float]"),
                output_types=("SonificationDocument",),
                evidence=("numeric",), engines=("sonify",),
                cost_dimensions=("sequence_length",),
                handler="kernel:sonify_compare",
                parameter_schema={
                    "prediction": "list[float]", "observation": "list[float]",
                    "mode": "stereo|residual?",
                },
            ),
            Capability(
                name="sonification.projection", domain="sonification",
                operation="projection",
                input_types=("MultimodalProjection",),
                output_types=("SonificationDocument",),
                evidence=("numeric", "exact", "interval_certified"),
                engines=("sonify",),
                cost_dimensions=("payload_size",),
                handler="kernel:sonify_projection",
                parameter_schema={"projection_id": "Projection object id", "mode": "auto?"},
            ),
            Capability(
                name="sonification.export", domain="sonification",
                operation="export",
                input_types=("SonificationDocument",), output_types=("ArtifactFile",),
                evidence=("numeric", "exact", "interval_certified"),
                engines=("sonify",),
                cost_dimensions=("duration", "sample_rate"),
                handler="kernel:sonify_export",
                parameter_schema={"sonification_id": "str", "path": "str"},
            ),
        )
        for capability in presentation_capabilities:
            self.router.registry.register(capability)
        for operation in ("solve", "verify"):
            self.router.registry.register(Capability(
                name=f"transform.{operation}", domain="transform",                operation=operation, input_types=("TransformProblem",),
                output_types=("TransformResult",),
                evidence=("symbolic", "symbolic_identity"),
                engines=("integral_transforms", "sympy"),
                cost_dimensions=("expression_size", "solver_time"),
                handler="module:integral_transforms",
                parameter_schema={"verify": "boolean?"},
            ))
        graph_schemas = {
            "verify": {},
            "bfs": {"source": "node label"},
            "dfs": {"source": "node label"},
            "connected_components": {},
            "strongly_connected_components": {},
            "shortest_path": {"source": "node label", "target": "node label?"},
            "minimum_spanning_tree": {},
            "maximum_flow": {"source": "node label", "sink": "node label"},
            "minimum_cut": {"source": "node label", "sink": "node label"},
            "matching": {},
            "euler_path": {},
            "coloring": {
                "max_operations": "integer?",
                "exact_coloring_max_vertices": "integer?",
            },
            "topological_sort": {},
            "cycle_detection": {},
            "centrality": {},
            "isomorphic_to": {"other_id": "graph object id"},
        }
        graph_operations = {
            "Graph": (
                "verify", "bfs", "dfs", "connected_components",
                "shortest_path", "matching", "euler_path", "coloring",
                "cycle_detection", "centrality", "isomorphic_to",
            ),
            "DirectedGraph": (
                "verify", "bfs", "dfs", "strongly_connected_components",
                "shortest_path", "topological_sort", "cycle_detection",
                "centrality",
            ),
            "WeightedGraph": (
                "verify", "bfs", "dfs", "connected_components",
                "strongly_connected_components", "shortest_path",
                "minimum_spanning_tree", "maximum_flow", "minimum_cut",
                "matching", "euler_path", "coloring", "topological_sort",
                "cycle_detection", "centrality",
            ),
            "MultiGraph": (
                "verify", "bfs", "dfs", "connected_components",
                "shortest_path", "matching", "euler_path", "coloring",
                "cycle_detection", "centrality",
            ),
        }
        for object_type, operations in graph_operations.items():
            for operation in operations:
                schema = graph_schemas[operation]
                self.router.registry.register(Capability(
                    name=f"graph.{object_type}.{operation}",
                    domain="graph", operation=operation,
                    input_types=(object_type,), output_types=("GraphResult",),
                    evidence=("exact", "witness_certificate"),
                    engines=("graph_theory",),
                    trust_levels=("exact", "unknown"),
                    verification_methods=(
                        "witness_check", "optimality_certificate"),
                    cost_dimensions=(
                        "vertex_count", "edge_count", "search_space"),
                    handler="module:graph_theory",
                    parameter_schema=schema,
                ))
        for object_type, operations in {
            "CombinatorialClass": ("count", "generate", "verify"),
            "GeneratingFunction": ("coefficient", "recurrence", "verify"),
        }.items():
            for operation in operations:
                self.router.registry.register(Capability(
                    name=f"combinatorics.{object_type}.{operation}",
                    domain="combinatorics", operation=operation,
                    input_types=(object_type,),
                    output_types=("CombinatoricsResult",),
                    evidence=("exact", "enumeration_check"),
                    engines=("combinatorics", "sympy"),
                    trust_levels=("exact", "symbolic", "unknown"),
                    verification_methods=(
                        "enumeration_check", "recurrence_check"),
                    cost_dimensions=("parameter_size", "item_count"),
                    handler="module:combinatorics",
                    parameter_schema={
                        "limit": "integer?", "n": "integer?",
                        "order": "integer?",
                    },
                ))
        group_operations = {
            "verify": {}, "order": {}, "closure": {"elements": "list?"},
            "generated_subgroup": {"generators": "list"},
            "subgroups": {}, "cosets": {"subgroup": "list"},
            "normality": {"subgroup": "list"}, "quotient": {"subgroup": "list"},
            "center": {}, "centralizer": {"element": "element"},
            "conjugacy_classes": {}, "commutator_subgroup": {},
            "orbits": {"action": "operation?"},
            "stabilizers": {"point": "element?"},
        }
        for operation, schema in group_operations.items():
            self.router.registry.register(Capability(
                name=f"group.FiniteGroup.{operation}",
                domain="finite_group", operation=operation,
                input_types=("FiniteGroup",), output_types=("GroupResult",),
                evidence=("exact", "group_axiom_check"),
                engines=("finite_groups",),
                trust_levels=("exact", "unknown"),
                verification_methods=("axiom_check", "membership_check"),
                cost_dimensions=("group_order",),
                handler="module:finite_groups", parameter_schema=schema,
            ))
        for operation, schema in {
            "verify": {}, "order": {}, "contains": {"images": "list[int]"},
            "orbits": {"point": "integer"},
            "stabilizers": {"point": "integer"},
            "stabilizer_chain": {},
        }.items():
            self.router.registry.register(Capability(
                name=f"group.PermutationGroup.{operation}",
                domain="finite_group", operation=operation,
                input_types=("PermutationGroup",), output_types=("GroupResult",),
                evidence=("exact", "schreier_sims"),
                engines=("finite_groups", "sympy"),
                trust_levels=("exact", "unknown"),
                verification_methods=("membership_check", "stabilizer_chain"),
                cost_dimensions=("degree", "generator_count"),
                handler="module:finite_groups", parameter_schema=schema,
            ))
        for operation in ("verify", "order"):
            self.router.registry.register(Capability(
                name=f"group.FiniteAbelianGroup.{operation}",
                domain="finite_group", operation=operation,
                input_types=("FiniteAbelianGroup",),
                output_types=("GroupResult",),
                evidence=("exact", "invariant_factor_check"),
                engines=("finite_groups",),
                trust_levels=("exact",),
                verification_methods=("invariant_factor_check",),
                cost_dimensions=("group_order",),
                handler="module:finite_groups",
            ))
        for operation in ("verify", "kernel", "image"):
            self.router.registry.register(Capability(
                name=f"group.homomorphism.{operation}",
                domain="finite_group", operation=operation,
                input_types=("GroupHomomorphism",),
                output_types=("GroupResult",),
                evidence=("exact", "homomorphism_check"),
                engines=("finite_groups",),
                trust_levels=("exact", "unknown"),
                verification_methods=("homomorphism_identity",),
                cost_dimensions=("group_order",),
                handler="module:finite_groups",
            ))
        for object_type in ("FiniteRing", "FiniteField"):
            for operation in ("verify", "add", "multiply", "inverse"):
                self.router.registry.register(Capability(
                    name=f"algebra.{object_type}.{operation}",
                    domain="finite_algebra", operation=operation,
                    input_types=(object_type,),
                    output_types=("FiniteAlgebraResult",),
                    evidence=("exact", "arithmetic_certificate"),
                    engines=("finite_algebra",),
                    trust_levels=("exact", "unknown"),
                    verification_methods=(
                        "ring_axiom_check", "irreducibility_check"),
                    cost_dimensions=("modulus", "field_degree"),
                    handler="module:finite_algebra",
                    parameter_schema={
                        "left": "element?", "right": "element?",
                        "element": "element?",
                    },
                ))
        for operation in (
            "verify", "smith_normal_form", "hermite_normal_form",
            "abelian_group",
        ):
            self.router.registry.register(Capability(
                name=f"algebra.module.{operation}",
                domain="finite_algebra", operation=operation,
                input_types=("Module",), output_types=("FiniteAlgebraResult",),
                evidence=("exact", "normal_form_certificate"),
                engines=("finite_algebra", "sympy"),
                trust_levels=("exact", "unknown"),
                verification_methods=(
                    "unimodular_transform", "normal_form_invariants"),
                cost_dimensions=("matrix_dimension", "entry_size"),
                handler="module:finite_algebra",
            ))
        from .engineering_adapter import register as register_engineering
        register_engineering(self.router.registry)
        self.verifier = VerificationCoordinator(self.router)
        self.executor = ObligationExecutor(self)
        self.contexts: dict[str, MathContext] = {}
        self.expressions: dict[str, Expr] = {}
        self.expression_sources: dict[str, str] = {}
        self.expression_producers: dict[str, str] = {}
        self.expression_provenance: dict[str, dict] = {}
        self.derivations: dict[str, DerivationStep] = {}
        self.plans: dict[str, ProblemPlan] = {}
        self.plan_sources: dict[str, tuple[str, str | None]] = {}
        self.typed_plan_requests: dict[str, dict] = {}
        self.executions: dict[str, PlanExecution] = {}
        self.artifacts: dict[str, dict] = {}
        self.matrices: dict[str, list[list[Expr]]] = {}
        self.matrix_producers: dict[str, str] = {}
        self.gf2m_fields: dict[str, object] = {}
        self.finite_systems: dict[str, object] = {}
        self.random_variables: dict[str, object] = {}
        self.tensors: dict[str, object] = {}
        self.jobs: dict[str, dict] = {}
        self.projections: dict[str, object] = {}
        self.viz_documents: dict[str, object] = {}
        self.sonification_documents: dict[str, object] = {}
        self.research_artifacts: dict[str, object] = {}
        self.math_objects: dict[str, dict] = {}
        self._job_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mathkernel-job")
        self._record_lock = threading.Lock()
        self._store = None
        if self.settings.store_path:
            from .store import KernelStore
            self._store = KernelStore(self.settings.store_path)

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:16]}"

    def _engine_availability(self) -> dict:
        engines = {e.name: {"available": bool(e.available)} for e in self.router.engines()}
        from .lean_bootstrap import LEAN_TOOLCHAIN, MATHLIB_REV, resolve_lean_toolchain
        spec = resolve_lean_toolchain()
        engines["lean"] = {
            "available": spec is not None,
            "toolchain": LEAN_TOOLCHAIN,
            "mathlib": MATHLIB_REV,
            "workspace": str(spec.workspace) if spec is not None else None,
            "install": "default",
        }
        return engines

    def capabilities(self, detail: str = "full") -> dict:
        """Discover the kernel; use detail='summary' before paged queries.

        The Python default remains full for compatibility. MCP defaults to the
        compact summary, and complete oversized outputs are retrievable resources.
        """
        if detail not in {"summary", "full"}:
            raise ValueError("detail must be 'summary' or 'full'")
        if detail == "summary":
            from collections import Counter
            records = self.router.registry.query()
            return {
                "version": MATHKERNEL_VERSION,
                "detail": "summary",
                "capability_count": len(records),
                "domains": dict(sorted(Counter(c.domain for c in records).items())),
                "engines": self._engine_availability(),
                "discovery": {"next": "capability_query", "default_page_size": 25,
                    "filters": ["domain", "object_type", "operation", "trust", "verification_method", "engine"],
                    "parameter_schema": "include_schema=True (declared type hints; domain validation remains at runtime)"},
                "output": {"maximum_bytes": self.settings.max_output_size_bytes,
                           "resource_reader": "result_resource_get", "encoding": "ASCII JSON, byte offsets"},
                "research_modules": {"robust_relation_inference": "standalone Python API, not a registered MCP operation"},
                "yolo_mode": yolo_mode(),
            }
        return {
            "prototype": False,
            "version": MATHKERNEL_VERSION,
            "engines": self.router.manifest(),
            "capability_registry": self.router.registry.manifest(),
            "operations": ["parse", "parse_latex", "get", "substitute", "analyze", "infer_structure",
                           "object_create", "object_get", "apply", "capability_query", "result_resource_get",
                           "plan", "plan_get", "execute_plan", "reason", "execution_get",
                           "simplify", "solve", "solve_system",
                           "differentiate", "integrate", "limit", "series", "summation", "product",
                           "prove_equivalence", "counterexample", "interval_evaluate", "numeric_evaluate",
                           "matrix_create", "matrix_get", "matrix_det", "matrix_inverse", "matrix_transpose",
                           "matrix_multiply", "matrix_rank", "matrix_rref", "matrix_eigenvalues", "matrix_solve",
                           "integer_analyze", "integer_compute", "integer_batch", "context_create",
                           "context_check", "context_infer", "derivation_get", "derivation_trace",
                           "codegen", "verify_code", "execute_code", "yolo_settings",
                           "collatz_sieve", "cuboid_sweep",
                           "job_submit", "job_status", "job_result", "job_list",
                           "gf2m_create", "gf2m_from_transition", "gf2m_compute", "gf2m_coords",
                           "gf2m_root_jump_rows", "gf2m_closure_roots", "gf2m_jump_rows",
                           "gf2_rank", "gf2_nullspace", "gf2_carryfree_cols", "gf2_minpoly",
                           "fwht",
                           "cumulant_compute", "finite_system_create", "koopman_matrix",
                           "koopman_transfer", "koopman_visibility", "koopman_lagged",
                           "koopman_observed", "koopman_diagnostics",
                           "finite_fourier_compute", "closure_search",
                           "set_create", "set_op", "set_membership", "quantifier_check",
                           "quantifier_eliminate", "quantifier_eliminate_batch",
                           "poly_groebner", "poly_divide", "poly_resultant",
                           "poly_discriminant", "poly_factor", "ideal_membership",
                           "poly_groebner_batch",
                           "prob_rv_create", "prob_expectation", "prob_variance",
                           "prob_covariance", "prob_bayes", "prob_markov_stationary",
                           "prob_markov_hitting_time", "prob_sample", "prob_distribution",
                           "stats_moments", "stats_order", "stats_regression",
                           "stats_correlation", "stats_ttest", "stats_chi2",
                           "stats_confidence_interval", "stats_batch_moments",
                           "tensor_create", "tensor_get", "tensor_contract", "tensor_solve",
                           "root_find", "root_scan", "quadrature",
                           "ode_solve", "ode_solve_numeric", "ode_ensemble", "pde_heat_1d",
                           "optimize_critical_points", "optimize_kkt", "lp_solve",
                           "optimize_minimize", "optimize_multistart",
                           "unit_check", "unit_convert", "unit_simplify",
                           "store_status", "replay", "fuzz_differential",
                           "certified_enclose",
                           "prove", "prove_batch", "prove_replay",
                           "conditioned_access_solve", "conditioned_symmetry_access",
                           "conditioned_access_compose", "conditioned_closure",
                           "symbolic_conditioned_access", "affine_conditioned_access",
                           "gf2_conditioned_access", "gf2_predictive_closure",
                           "synthesize_conditioned_closures",
                           "synthesize_gf2_vector_conditioned_access",
                           "discover_structural_conditioned_closure",
                           "discover_factor_swap_conditioned_closure",
                           "viz_create", "viz_dag", "viz_koopman", "viz_export"],
            "numerics": {"root_finding": ["ridder/secant (mpmath)", "brent float64",
                                          "interval isolation (mpmath.iv)"],
                         "quadrature": ["tanh-sinh + gauss-legendre cross-check"],
                         "engines": ["numeric_high_precision", "interval_certified",
                                     "numeric (float64)"],
                         "parallel": ["root_scan over subintervals"]},
            "ode": {"symbolic": "sympy.dsolve + classification",
                    "numeric": ["adaptive RK45 (mpmath)", "float64 RK4 (njit)"],
                    "ensemble": ["cuda-rawkernel" if self._numeric_device() == "gpu"
                                 else "process-pool", "process-pool"],
                    "pde": ["1D heat FTCS (njit/cupy)"],
                    "max_ode_steps": self.settings.max_ode_steps},
            "pde": {"operations": ["pde_heat_1d", "pde_heat_2d", "pde_wave_1d",
                                   "pde_advect_1d", "pde_ensemble", "pde_mol_heat"],
                    "typed_operations": ["verify", "classify", "boundary_compatibility",
                                         "derive_weak_form", "reference_element", "basis",
                                         "quadrature", "finite_element_space", "assemble", "solve",
                                         "estimate_error", "mark", "refine", "compare"],
                    "typed_objects": ["PDEProblem", "PDEClassification",
                                      "PDECompatibilityReport", "WeakForm", "FEMMesh",
                                      "ReferenceElement", "BasisFunctionSet", "QuadratureRule",
                                      "FiniteElementSpace", "AssembledSystem", "FEMSolution",
                                      "FEMErrorEstimate", "RefinementMarking", "RefinedMesh",
                                      "MeshTransfer", "FEMConvergenceObservation"],
                    "discovery": "capability_query(domain='pde')",
                    "representation": "bounded rectangular domains; derivative multi-indices; scalar or system equations",
                    "conditions": ["Dirichlet", "Neumann", "Robin", "periodic", "initial"],
                    "classification": "linear scalar two-variable second-order principal-part discriminant; explicit conditional outcomes",
                    "weak_forms": "typed trial/test spaces and measures; one explicit product-rule integration-by-parts transfer per selected term; oriented boundary terms retained",
                    "finite_elements": "oriented simplex meshes; canonical P1 reference elements and nodal bases; exact degree-1/2 reference quadrature where supported; vertex-DOF C0 spaces",
                    "assembly": "sparse affine-simplex P1 assembly for scalar linear stationary weak forms; explicit Neumann/Robin integration and symmetric Dirichlet elimination; exact or quadrature-defined replay",
                    "algebraic_solve": "bounded exact rank/solve or SciPy sparse numeric solve with residual, rank and conditioning diagnostics",
                    "adaptivity": "P1 residual-jump indicators for constant diagonal diffusion; explicit marking, conforming triangle red refinement, nested nodal transfer, and empirical two-mesh estimator rates",
                    "claims_excluded": ["space membership", "analytic regularity",
                                        "existence", "uniqueness", "well-posedness",
                                        "condition completeness", "geometric mesh coverage or non-overlap",
                                        "continuous PDE solution", "continuum error bound"],
                    "schemes": ["FTCS (heat 1D/2D)", "leapfrog (wave)",
                                "upwind (advection)", "method-of-lines + RK4"],
                    "stability": ["heat-1d r<=1/2", "heat-2d r<=1/4",
                                  "wave/advect CFL<=1"],
                    "tiers": ["python-fallback", "njit", "numeric-gpu"],
                    "ensemble": "batched CuPy stencil / process pool",
                    "max_pde_grid": self.settings.max_pde_grid,
                    "max_pde_fields": self.settings.max_pde_fields,
                    "max_pde_dimensions": self.settings.max_pde_dimensions,
                    "max_pde_equations": self.settings.max_pde_equations,
                    "max_pde_terms": self.settings.max_pde_terms,
                    "max_pde_conditions": self.settings.max_pde_conditions,
                    "max_pde_derivative_order": self.settings.max_pde_derivative_order,
                    "max_pde_nonlinear_power": self.settings.max_pde_nonlinear_power,
                    "max_pde_work": self.settings.max_pde_work,
                    "max_pde_spaces": self.settings.max_pde_spaces,
                    "max_pde_space_order": self.settings.max_pde_space_order,
                    "max_pde_weak_terms": self.settings.max_pde_weak_terms,
                    "max_pde_ibp_steps": self.settings.max_pde_ibp_steps,
                    "max_pde_weak_work": self.settings.max_pde_weak_work,
                    "max_fem_points": self.settings.max_fem_points,
                    "max_fem_cells": self.settings.max_fem_cells,
                    "max_fem_dofs": self.settings.max_fem_dofs,
                    "max_fem_work": self.settings.max_fem_work,
                    "max_fem_assembly_nnz": self.settings.max_fem_assembly_nnz,
                    "max_fem_assembly_work": self.settings.max_fem_assembly_work,
                    "max_fem_exact_solve_dofs": self.settings.max_fem_exact_solve_dofs,
                    "max_fem_numeric_solve_dofs": self.settings.max_fem_numeric_solve_dofs,
                    "max_fem_estimator_work": self.settings.max_fem_estimator_work,
                    "max_fem_refined_cells": self.settings.max_fem_refined_cells,
                    "phase": "G.5 complete; Phase G closed with typed estimation/refinement and no continuum-error or convergence-theorem claim"},
            "engineering": {
                "objects": ["ContinuousSignal", "DiscreteSignal", "Spectrum", "Filter", "FilterDesign",
                            "TransferFunction", "ZeroPoleGain", "StateSpaceSystem", "DiscreteControlSystem", "TransferMatrix", "FrequencyResponse", "RootLocus", "TimeResponse", "FilterState", "RiccatiCertificate", "FiniteHorizonLQR", "KalmanState", "MPCPlan",
                            "OptimizationProblem", "OptimizationCertificate", "MILPCertificate",
                            "ConicProblem", "ConicCertificate", "QuadraticallyConstrainedProblem", "QuadraticCertificate"],
                "discovery": "capability_query(domain='signal'|'control'|'optimization')",
                "arithmetic": "exact by default; explicit mode='numeric' for approximate execution",
                "optimization_certification": "original-data KKT, product-cone, Farkas/ray, MILP tree and quadratic-Lagrangian certificates",
                "native_solver_isolation": "fresh hard-killable interpreter per external candidate search",
                "phase": "D.8 complete; Phase D engineering baseline closed",
            },
            "geometry": {
                "objects": ["Manifold", "Chart", "Metric", "GeometryTensor",
                            "Connection", "GeodesicSystem", "CoordinateMap",
                            "JacobianMap", "TensorField", "DifferentialForm",
                            "Point", "PointSet", "Polygon", "Polytope",
                            "Triangulation", "VoronoiDiagram",
                            "SimplicialComplex", "CubicalComplex",
                            "ChainComplex", "Homology"],
                "discovery": "capability_query(domain='geometry')",
                "arithmetic": "exact/symbolic coordinate computation through restricted MathIR",
                "verification": ["inverse identity", "metric compatibility",
                                 "torsion-free symmetry", "Riemann symmetries",
                                 "first Bianchi identity", "Jacobian composition",
                                 "graded commutativity", "d squared = 0",
                                 "pullback commutes with d", "filtered orientation",
                                 "incircle and intersection predicates",
                                 "hull containment", "empty circumcircle",
                                 "triangulation incidence", "face closure",
                                 "boundary squared = 0", "Smith kernel quotient",
                                 "exact field rank", "Euler-Poincare identity"],
                "limits": {"max_geometry_dimension": self.settings.max_geometry_dimension,
                           "max_geometry_rank": self.settings.max_geometry_rank,
                           "max_geometry_points": self.settings.max_geometry_points,
                           "max_geometry_simplices": self.settings.max_geometry_simplices,
                           "max_geometry_work": self.settings.max_geometry_work,
                           "max_topology_dimension": self.settings.max_topology_dimension,
                           "max_topology_cells": self.settings.max_topology_cells,
                           "max_topology_matrix_entries": self.settings.max_topology_matrix_entries,
                           "max_topology_entry_bits": self.settings.max_topology_entry_bits,
                           "max_topology_work": self.settings.max_topology_work},
                "phase": "E.5 complete; Phase E geometry/topology baseline closed",
            },
            "statistics_inference": {
                "objects": ["StatisticalSample", "DescriptiveSummary",
                            "CovarianceMatrix", "EmpiricalDistribution",
                            "GeneralizedLinearModel", "GLMFit"],
                "nonparametric_objects": ["NonparametricTestResult",
                                           "ResamplingResult"],
                "survival_objects": ["SurvivalDataset", "KaplanMeierEstimate",
                                     "CoxProportionalHazardsModel", "CoxPHFit"],
                "time_series_objects": ["TimeSeriesDataset", "TimeSeriesAnalysis",
                                        "TimeSeriesModel", "TimeSeriesFit",
                                        "TimeSeriesForecast"],
                "discovery": "capability_query(domain='statistics')",
                "arithmetic": "exact or numeric sample computation; arithmetic trust is distinct from empirical/model support",
                "verification": ["sample schema", "moment identity",
                                 "order statistics", "covariance symmetry",
                                 "frequency normalization",
                                 "GLM response domain and design rank",
                                 "normal equations and score residual",
                                 "IRLS convergence and observed information",
                                 "separation and conditioning refusal",
                                 "exact permutation enumeration",
                                 "tie-corrected asymptotic approximation",
                                 "seeded PCG64 resampling replay",
                                 "right-censoring and delayed-entry risk sets",
                                 "Kaplan-Meier product-limit identity",
                                 "Greenwood log-log uncertainty",
                                 "Cox partial-likelihood score and information",
                                 "Breslow/Efron tie accounting",
                                 "baseline hazard and Schoenfeld diagnostics",
                                 "strict stored time ordering and regular spacing",
                                 "exact ACF and Durbin-Levinson PACF",
                                 "ADF regression with explicit asymptotic critical values",
                                 "ARMA conditional residual replay",
                                 "AR/MA root stationarity and invertibility",
                                 "GARCH positivity and persistence constraint",
                                 "Ljung-Box and Gaussian residual diagnostics",
                                 "analytic forecast recursion and intervals",
                                 "population non-inference"],
                "limits": {
                    "max_statistical_variables": self.settings.max_statistical_variables,
                    "max_statistical_observations": self.settings.max_statistical_observations,
                    "max_statistical_cells": self.settings.max_statistical_cells,
                    "max_statistical_work": self.settings.max_statistical_work,
                    "max_glm_parameters": self.settings.max_glm_parameters,
                    "max_glm_iterations": self.settings.max_glm_iterations,
                    "max_glm_prediction_rows": self.settings.max_glm_prediction_rows,
                    "max_glm_work": self.settings.max_glm_work,
                    "max_nonparametric_groups": self.settings.max_nonparametric_groups,
                    "max_exact_resampling_states": self.settings.max_exact_resampling_states,
                    "max_resamples": self.settings.max_resamples,
                    "max_resampling_batch_cells": self.settings.max_resampling_batch_cells,
                    "max_resampling_work": self.settings.max_resampling_work,
                    "max_survival_strata": self.settings.max_survival_strata,
                    "max_survival_timeline_points": self.settings.max_survival_timeline_points,
                    "max_cox_parameters": self.settings.max_cox_parameters,
                    "max_cox_iterations": self.settings.max_cox_iterations,
                    "max_cox_prediction_rows": self.settings.max_cox_prediction_rows,
                    "max_cox_information_condition": self.settings.max_cox_information_condition,
                    "max_survival_work": self.settings.max_survival_work,
                    "max_time_series_lag": self.settings.max_time_series_lag,
                    "max_time_series_difference": self.settings.max_time_series_difference,
                    "max_time_series_parameters": self.settings.max_time_series_parameters,
                    "max_time_series_iterations": self.settings.max_time_series_iterations,
                    "max_time_series_forecast_steps": self.settings.max_time_series_forecast_steps,
                    "max_time_series_work": self.settings.max_time_series_work,
                },
                "phase": "F.5 complete; time-series models",
            },
            "optimize": {"exact": ["rational simplex (Bland)"],
                         "symbolic": ["critical points", "KKT conditions"],
                         "numeric": ["nelder-mead", "multistart (process pool)",
                                     "njit simplex tableau"],
                         "max_iterations": self.settings.max_iterations,
                         "tolerance": self.settings.tolerance},
            "units": {"operations": ["unit_check", "unit_convert", "unit_simplify"],
                      "dimensions": "exact 7-vector SI (L,M,T,I,Th,N,J) with Fraction exponents",
                      "registry": "SI base + derived + common prefixed units",
                      "trust": "exact"},
            "persistence": {"enabled": self._store is not None,
                            "path": self.settings.store_path,
                            "operations": ["store_status", "replay"]},
            "certified": {"arb_available": _arb_available(),
                          "operations": ["certified_enclose"],
                          "fallback": "mpmath.iv"},
            "fuzzing": {"operations": ["fuzz_differential"],
                        "engines": ["mpmath-40dps", "float64-compiled"],
                        "parallel": "process pool"},
            "lean": {"tactic_coverage": self.lean.tactic_coverage(),
                     "replay": "replay_certificate",
                     **self._engine_availability()["lean"]},
            "proving": {"operations": ["prove", "prove_batch", "prove_replay"],
                        "smt_portfolio": ["direct", "qe-light", "lia"],
                        "portfolio_size": self.settings.prove_portfolio_size,
                        "lean_tactics": list(self.lean.tactic_coverage()),
                        "parallel": ["encoding portfolio (threads)",
                                     "prove_batch (process pool)"]},
            "exact_discrete": {
                "graphs": {
                    "objects": ["Graph", "DirectedGraph", "WeightedGraph", "MultiGraph"],
                    "operations": [
                        "bfs", "dfs", "connected_components",
                        "strongly_connected_components", "shortest_path",
                        "minimum_spanning_tree", "maximum_flow", "minimum_cut",
                        "matching", "euler_path", "coloring",
                        "topological_sort", "cycle_detection", "centrality",
                        "isomorphic_to",
                    ],
                    "certificates": [
                        "bfs_tree", "shortest_path_optimality",
                        "mst_cycle_property", "max_flow_min_cut",
                        "konig_cover", "proper_coloring",
                        "isomorphism_permutation",
                    ],
                    "max_vertices": self.settings.max_graph_vertices,
                    "max_edges": self.settings.max_graph_edges,
                },
                "combinatorics": {
                    "objects": ["CombinatorialClass", "GeneratingFunction"],
                    "lazy_generation": True,
                    "max_items": self.settings.max_combinatorial_items,
                },
                "finite_algebra": {
                    "objects": [
                        "FiniteGroup", "PermutationGroup", "FiniteAbelianGroup",
                        "FiniteRing", "FiniteField", "Module",
                    ],
                    "normal_forms": ["smith", "hermite"],
                    "max_group_elements": self.settings.max_group_elements,
                    "max_field_degree": self.settings.max_field_degree,
                    "max_normal_form_dim": self.settings.max_normal_form_dim,
                },
            },
            "probability": {"discrete": "exact rational (Fraction) pmfs",
                            "markov": ["stationary", "hitting_times"],
                            "max_markov_states": 512, "max_support": 100_000,
                            "symbolic_bridge": "sympy.stats",
                            "engines": ["exact", "symbolic (sympy.stats)", "numeric (sampling)"]},
            "statistics": {"exact": ["moments", "order_statistics", "regression",
                                     "correlation r^2"],
                           "numeric_high_precision": ["ttest", "chi2", "confidence_interval"],
                           "engines": ["exact", "numpy vectorized", "process-pool batch"]},
            "tensors": {"storage": "dict-of-keys sparse",
                        "operations": ["create", "get", "contract", "solve"],
                        "engines": ["exact", "njit", "numeric-cpu",
                                    f"numeric-gpu" if self._numeric_device() == "gpu"
                                    else "numeric-gpu (unavailable)"],
                        "max_entries": 5_000_000},
            "logic": {"set_kinds": ["finite", "naturals", "integers", "rationals", "reals",
                                    "complexes", "empty"],
                      "set_ops": ["union", "intersect", "difference", "complement"],
                      "quantifiers": ["forall", "exists"],
                      "quantifier_engine": "z3" if self.z3.available else "unavailable",
                      "quantifier_elimination": {"tactic": "z3-qe",
                          "fragments": ["LRA", "LIA"],
                          "max_variables": self.settings.max_qe_variables,
                          "batch": "process pool"},
                      "boolean_connectives": ["and", "or", "not"],
                      "binders": ["sumover", "productover"]},
            "polynomial": {"operations": ["groebner", "divide", "resultant", "discriminant",
                                          "factor", "ideal_membership", "groebner_batch"],
                           "monomial_orders": ["lex", "grlex", "grevlex", "ilex", "igrlex",
                                               "igrevlex"],
                           "max_polys": 64, "max_variables": 16,
                           "engines": ["exact", "process-pool batch"]},
            "finite_dynamics": {
                "max_states": self.settings.max_finite_states,
                "max_cumulant_order": self.settings.max_cumulant_order,
                "max_closure_results": self.settings.max_closure_results,
                "bases": ["walsh", "cyclic"],
                "closure_kinds": ["cyclic", "binary", "cyclic_order", "binary_order"],
                "engines": {
                    "koopman": ["exact", f"numeric-{self._numeric_device()}"],
                    "closure_search": ["njit" if self._relations_njit() else "python",
                                       "python-fallback"]}},
            "transforms": {"fwht": {"normalization": "unnormalized",
                                    "max_size": self.settings.max_fwht_size}},
            "gf2m": {"max_degree": 1024, "irreducibility": "rabin",
                     "operations": ["add", "sub", "mul", "div", "pow", "inv", "sqrt", "trace",
                                    "quadratic_roots"]},
            "jobs": {"kinds": sorted(self._JOB_RUNNERS), "max_retained": self.settings.max_jobs_retained},
            "latex": {"engine": "sympy.parsing.latex"},
            "matrices": {"max_dim": self.settings.max_matrix_dim,
                         "operations": ["det", "inverse", "transpose", "multiply", "rank",
                                        "rref", "eigenvalues", "solve"]},
            "simplify_modes": ["simplify", "normal", "expand", "factor", "cancel", "trig", "rational", "normal_form"],
            "codegen": {"languages": sorted(EMITTERS), "modes": ["generic"],
                        "targets": ["evaluate", "solve", "constraint"]},
            "trust_levels": [x.value for x in TrustLevel],
            "result_statuses": [x.value for x in ResultStatus],
            "solution_set_kinds": ["finite", "interval", "union", "conditional", "parametric", "empty", "universal", "symbolic", "unknown"],
            "numeric_node_kinds": ["integer", "rational", "real", "algebraic", "complex", "interval"],
            "limits": self.settings.limits(),
            "security": {"string_to_sympy_eval": False, "lean_shell": False,
                         "ambiguous_notation_is_not_guessed": True,
                         "code_execution_enabled": self.settings.enable_execution,
                         "yolo_mode": yolo_mode()},
            "execution": {"dependency_aware_obligation_dag": True, "assumption_propagation": True,
                          "conflict_detection": True, "result_reconciliation": "conservative",
                          "parallel_waves": self.settings.enable_parallel},
            "parallel": {"enabled": self.settings.enable_parallel,
                         "max_workers": self.settings.max_workers,
                         "integer_batch": "processes",
                         "obligation_waves": "threads"},
        }

    def capability_query(
        self,
        domain: str | None = None,
        object_type: str | None = None,
        input_type: str | None = None,
        output_type: str | None = None,
        operation: str | None = None,
        evidence: str | None = None,
        trust: str | None = None,
        verification_method: str | None = None,
        engine: str | None = None,
        offset: int = 0,
        limit: int | None = None,
        include_schema: bool = False,
    ) -> dict:
        capabilities = self.router.registry.manifest(
            domain=domain,
            object_type=object_type,
            input_type=input_type,
            output_type=output_type,
            operation=operation,
            evidence=evidence,
            trust=trust,
            verification_method=verification_method,
            engine=engine,
        )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100):
            raise ValueError("limit must be an integer from 1 to 100")
        total = len(capabilities)
        filters = (domain, object_type, input_type, output_type, operation, evidence, trust, verification_method, engine)
        page_limit = 25 if limit is None and all(v is None for v in filters) else limit
        page = capabilities[offset:offset + page_limit if page_limit is not None else None]
        if include_schema:
            from .discovery import parameter_json_schema
            for item in page:
                item["parameter_json_schema"] = parameter_json_schema(item["parameter_schema"])
        next_offset = offset + len(page) if offset + len(page) < total else None
        return {
            "capabilities": page,
            "count": len(page),
            "total_count": total,
            "offset": offset,
            "next_offset": next_offset,
            "filters": {
                "domain": domain,
                "object_type": object_type,
                "input_type": input_type,
                "output_type": output_type,
                "operation": operation,
                "evidence": evidence,
                "trust": trust,
                "verification_method": verification_method,
                "engine": engine,
            },
        }

    def expression(self, text: str):
        """Parse into an immutable Python ExpressionHandle; no new evaluator."""
        from .python_api import ExpressionHandle
        return ExpressionHandle.from_result(self, self.parse(text))

    def result_resource_get(self, resource_id: str, offset: int = 0, length: int = 8192) -> dict:
        """Retrieve a byte-budgeted page of a previously oversized result."""
        from .output_policy import result_resource_get
        return result_resource_get(self, resource_id, offset, length)

    def _typed_parse_many(
        self, texts: list[str], context_id: str | None,
    ) -> tuple[list[Expr], list[sp.Basic], TrustLevel]:
        ctx = self.contexts.get(context_id) if context_id else None
        if context_id and ctx is None:
            raise ValueError(f"Unknown context_id: {context_id}")
        if any(len(text) > self.settings.max_input_length for text in texts):
            raise ValueError("Typed expression exceeds max_input_length")
        # Repeated coefficients/sample literals share immutable parsing and CAS
        # conversion within this call; no cross-context cache can leak assumptions.
        unique_irs = {text: parse_math(text) for text in dict.fromkeys(texts)}
        irs = [unique_irs[text] for text in texts]
        oversized = [
            text for text, ir in zip(texts, irs)
            if _count_nodes(ir) > self.settings.max_expression_nodes
        ]
        if oversized:
            raise ValueError(
                "Typed expression exceeds "
                f"max_expression_nodes={self.settings.max_expression_nodes}")
        env = self._symbol_env(ctx, *irs)
        converted = {text: self.sympy.to_sympy(ir, env) for text, ir in unique_irs.items()}
        values = [converted[text] for text in texts]
        trust = (
            self._trust_min(*(self._expr_trust(ir) for ir in irs))
            if irs else TrustLevel.EXACT
        )
        return irs, values, trust

    def _typed_context_assumptions(
        self, context_id: str | None,
    ) -> tuple[sp.Basic, ...]:
        """Return explicit context relations as SymPy objects for typed domains.

        Symbol properties are already reflected in ``_symbol_env``.  Relational
        assumptions such as ``b > a`` need to be passed explicitly to domain
        objects so parameter constraints can be discharged without erasing the
        assumption from provenance.
        """
        if not context_id:
            return ()
        ctx = self.contexts.get(context_id)
        if ctx is None:
            raise ValueError(f"Unknown context_id: {context_id}")
        irs = [item.expression for item in ctx.assumptions]
        if not irs:
            return ()
        env = self._symbol_env(ctx, *irs)
        return tuple(self.sympy.to_sympy(ir, env) for ir in irs)

    @staticmethod
    def _typed_record_assumptions(record: dict) -> list[str]:
        """Extract persisted mathematical assumptions from a typed object."""
        value = record.get("value")
        assumptions = list(getattr(value, "assumptions", ()) or ())
        domain = getattr(value, "domain", None)
        assumptions.extend(list(getattr(domain, "assumptions", ()) or ()))
        out: list[str] = []
        for assumption in assumptions:
            text = sp.sstr(assumption)
            if text not in out:
                out.append(text)
        return out

    @staticmethod
    def _typed_domain(value, *, transform: str | None = None):
        if isinstance(value, list) and len(value) == 2:
            return tuple(value)
        names = {
            "reals": sp.S.Reals,
            "complexes": sp.S.Complexes,
            "integers": sp.S.Integers,
            "nonnegative_integers": sp.S.Naturals0,
        }
        if isinstance(value, str) and value.lower() in names:
            return names[value.lower()]
        if transform == "laplace":
            return sp.Interval(0, sp.oo)
        if transform == "fourier":
            return sp.S.Reals
        if transform == "mellin":
            return sp.Interval.open(0, sp.oo)
        if transform == "z":
            return (-sp.oo, sp.oo)
        return sp.S.Complexes

    @staticmethod
    def _transform_convention(name: str):
        from .integral_transforms import TransformConvention
        conventions = {
            "laplace_standard": TransformConvention.laplace_standard,
            "fourier_angular_frequency":
                TransformConvention.fourier_angular_frequency,
            "mellin_standard": TransformConvention.mellin_standard,
            "z_bilateral": TransformConvention.z_bilateral,
        }
        if name not in conventions:
            raise ValueError(
                f"explicit convention required; choose from "
                f"{sorted(conventions)}")
        return conventions[name]()

    def _typed_set(
        self, value, context_id: str | None = None,
    ) -> tuple[sp.Set, TrustLevel]:
        """Build a set from named domains or restricted interval bounds."""
        if isinstance(value, str):
            if value.lower() not in {
                "reals", "complexes", "integers", "nonnegative_integers",
            }:
                raise ValueError(f"unknown named set: {value}")
            domain = self._typed_domain(value)
            if not isinstance(domain, sp.Set):
                raise ValueError("set specification did not produce a SymPy Set")
            return domain, TrustLevel.EXACT
        if isinstance(value, (list, tuple)) and len(value) == 2:
            specification = {"lower": value[0], "upper": value[1]}
        elif isinstance(value, dict):
            specification = value
        else:
            raise ValueError(
                "set must be a named domain or an interval specification")
        _, bounds, trust = self._typed_parse_many(
            [str(specification["lower"]), str(specification["upper"])],
            context_id,
        )
        return sp.Interval(
            bounds[0], bounds[1],
            left_open=bool(specification.get("left_open", False)),
            right_open=bool(specification.get("right_open", False)),
        ), trust

    @staticmethod
    def _bounded_positive_int(value, name: str, maximum: int) -> int:
        parsed = int(value)
        if parsed < 1 or parsed > maximum:
            raise ValueError(f"{name} must be in 1..{maximum}")
        return parsed

    def _store_math_object(
        self, object_type: str, value, *, input_trust: TrustLevel,
        sources: list[str] | None = None,
    ) -> str:
        if len(self.math_objects) >= self.settings.max_math_objects:
            raise ValueError(
                f"math object limit ({self.settings.max_math_objects}) reached")
        object_id = self._id("obj")
        construction = EvidenceBundle(computation=[
            ComputationEvidence(
                engine="math_object",
                method="typed_object_construction",
                arithmetic="inherited",
                deterministic=True,
                trust=input_trust.value,
                metadata={
                    "object_type": object_type,
                    "sources": list(sources or []),
                },
            )
        ])
        record = {
            "object_type": object_type,
            "value": value,
            "input_trust": input_trust,
            "sources": list(sources or []),
            "evidence_bundle": construction.model_copy(deep=True),
            "claim_evidence": {
                "construction": construction.model_copy(deep=True),
            },
            "semantic_status": ResultStatus.CANDIDATE,
        }
        self.math_objects[object_id] = record
        self._persist_math_object(object_id, record)
        return object_id

    def _persist_math_object(self, object_id: str, record: dict) -> None:
        if self._store is None:
            return
        self._validate_math_object_record(record)
        self._store.put_math_object(object_id, {
            "object_type": record["object_type"],
            "value": _encode_typed_value(record["value"]),
            "input_trust": record["input_trust"].value,
            "sources": list(record["sources"]),
            "evidence_bundle": _encode_typed_value(
                record.get("evidence_bundle", EvidenceBundle())),
            "claim_evidence": _encode_typed_value(
                record.get("claim_evidence", {})),
            "semantic_status": (
                record.get("semantic_status", ResultStatus.CANDIDATE).value
                if hasattr(
                    record.get("semantic_status", ResultStatus.CANDIDATE),
                    "value")
                else str(record.get(
                    "semantic_status", ResultStatus.CANDIDATE.value))
            ),
        })

    @staticmethod
    def _validate_math_object_record(record: dict) -> None:
        """Reject inconsistent typed-object envelopes before use or writeback."""
        object_type = record.get("object_type")
        value = record.get("value")
        if not isinstance(object_type, str) or value is None:
            raise ValueError("typed object record is missing its type or value")
        actual_type = type(value).__name__
        # A few older domains intentionally use public wire names that differ
        # from the implementation class.  Phase F uses canonical class names,
        # so its complete surface can be checked without weakening aliases.
        canonical_typed_types = {
            "StatisticalSample", "VariableSummary", "DescriptiveSummary",
            "CovarianceMatrix", "EmpiricalDistribution",
            "GeneralizedLinearModel", "GLMFit", "NonparametricTestResult",
            "ResamplingResult", "SurvivalDataset", "KaplanMeierEstimate",
            "CoxProportionalHazardsModel", "CoxPHFit", "TimeSeriesDataset",
            "TimeSeriesAnalysis", "TimeSeriesModel", "TimeSeriesFit",
            "TimeSeriesForecast", "PoissonProcess", "WienerProcess",
            "GaussianProcess", "ContinuousTimeMarkovChain",
            "FiniteDimensionalDistribution", "GaussianProcessPosterior",
            "CTMCTransition", "StochasticDifferentialEquation",
            "SDESimulation", "SDEConvergenceStudy",
            "PDEProblem", "PDEClassification", "PDECompatibilityReport",
            "WeakForm", "FEMMesh", "ReferenceElement", "BasisFunctionSet",
            "QuadratureRule", "FiniteElementSpace", "AssembledSystem", "FEMSolution",
            "FEMErrorEstimate", "RefinementMarking", "RefinedMesh", "MeshTransfer",
            "FEMConvergenceObservation",
        }
        if object_type in canonical_typed_types and actual_type != object_type:
            raise ValueError(
                "typed object record type does not match its decoded value")
        sources = record.get("sources")
        if not isinstance(sources, list) or any(
                not isinstance(item, str) or not item for item in sources):
            raise ValueError("typed object sources must be object-id strings")
        source_field = _TYPED_SOURCE_FIELDS.get(object_type)
        if source_field is not None:
            internal_source = getattr(value, source_field, None)
            if not sources or sources[0] != internal_source:
                raise ValueError(
                    f"{object_type} source ancestry does not match {source_field}")
        if object_type == "FEMMesh":
            triangulation_id = getattr(value, "triangulation_id", None)
            expected = ([getattr(value, "weak_form_id")] +
                        ([triangulation_id] if triangulation_id is not None else []))
            if sources != expected:
                raise ValueError("FEMMesh source ancestry does not reconcile")
        if object_type in {"ReferenceElement", "BasisFunctionSet", "QuadratureRule"}:
            if len(sources) != 1:
                raise ValueError(f"{object_type} must have exactly one source")
        if object_type == "FiniteElementSpace":
            expected = [value.mesh_id, value.reference_element_id, value.basis_id]
            if sources != expected:
                raise ValueError("FiniteElementSpace source ancestry does not reconcile")
        if object_type == "AssembledSystem":
            expected = [value.finite_element_space_id, value.quadrature_id]
            if sources != expected:
                raise ValueError("AssembledSystem source ancestry does not reconcile")
        if object_type == "FEMSolution" and sources != [value.assembled_system_id]:
            raise ValueError("FEMSolution source ancestry does not reconcile")
        if object_type == "FEMErrorEstimate" and sources != [value.solution_id]:
            raise ValueError("FEMErrorEstimate source ancestry does not reconcile")
        if object_type == "RefinementMarking" and sources != [value.error_estimate_id]:
            raise ValueError("RefinementMarking source ancestry does not reconcile")
        if object_type in {"RefinedMesh", "MeshTransfer"} and sources != [value.marking_id]:
            raise ValueError(f"{object_type} source ancestry does not reconcile")
        if object_type == "FEMConvergenceObservation":
            if sources != [value.fine_estimate_id, value.coarse_estimate_id]:
                raise ValueError("FEMConvergenceObservation source ancestry does not reconcile")

    def _get_math_object_record(self, object_id: str) -> dict | None:
        record = self.math_objects.get(object_id)
        if record is not None:
            self._validate_math_object_record(record)
            return record
        if self._store is None:
            return record
        payload = self._store.get_math_object(object_id)
        if payload is None:
            return None
        record = {
            "object_type": payload["object_type"],
            "value": _decode_typed_value(payload["value"]),
            "input_trust": TrustLevel(payload["input_trust"]),
            "sources": list(payload.get("sources", [])),
            "evidence_bundle": _decode_typed_value(
                payload.get(
                    "evidence_bundle",
                    _encode_typed_value(EvidenceBundle()))),
            "claim_evidence": _decode_typed_value(
                payload.get("claim_evidence", {})),
            "semantic_status": ResultStatus(
                payload.get("semantic_status", "candidate")),
        }
        self._validate_math_object_record(record)
        self.math_objects[object_id] = record
        return record

    def _typed_operation_dependencies(
        self, object_id: str, capability: Capability, parameters: dict,
    ) -> tuple[list[tuple[str, dict]], TrustLevel]:
        """Resolve every typed-object/MathIR input that can limit a result.

        Object references are discovered from explicit ``*_id``/``*_ids``
        parameters and capability schemas. Mathematical expression parameters
        declared as MathIR are parsed through the restricted parser so decimal
        ancestry cannot be lost before a domain adapter sees it.
        """
        dependencies: list[tuple[str, dict]] = []
        seen: set[str] = set()

        def add_ref(candidate) -> None:
            if candidate is None:
                return
            ref = str(candidate)
            if ref in seen:
                return
            record = self._get_math_object_record(ref)
            if record is not None:
                seen.add(ref)
                dependencies.append((ref, record))

        add_ref(object_id)
        schema = capability.parameter_schema or {}
        for key, value in parameters.items():
            descriptor = str(schema.get(key, "")).lower()
            is_ref = (
                key.endswith("_id") or key.endswith("_ids")
                or "object id" in descriptor
            )
            if is_ref:
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        add_ref(item)
                else:
                    add_ref(value)

        math_texts: list[str] = []
        def collect_math(value):
            if isinstance(value, bool):
                raise ValueError("boolean is not a mathematical scalar")
            if isinstance(value, (str, int, float)):
                math_texts.append(str(value))
            elif isinstance(value, dict):
                for item in value.values():
                    collect_math(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect_math(item)
        def collect_declared_math(value, descriptor):
            # Structured object schemas separate mathematical witnesses from
            # enum labels and tree indices. Never parse metadata as an expression.
            if isinstance(descriptor, dict) and "properties" in descriptor:
                if not isinstance(value, dict):
                    raise ValueError("structured mathematical parameter must be an object")
                for name, child_schema in descriptor["properties"].items():
                    if name in value and value[name] is not None:
                        collect_declared_math(value[name], child_schema)
            elif isinstance(descriptor, dict) and "items" in descriptor:
                if not isinstance(value, (list, tuple)):
                    raise ValueError("structured mathematical parameter must be an array")
                for item in value:
                    collect_declared_math(item, descriptor["items"])
            elif "mathir" in str(descriptor).lower():
                collect_math(value)
        for key, descriptor_value in schema.items():
            if key in parameters:
                collect_declared_math(parameters[key], descriptor_value)
        context_id = parameters.get("context_id")
        parameter_trust = TrustLevel.EXACT
        if math_texts:
            _, _, parameter_trust = self._typed_parse_many(
                math_texts, str(context_id) if context_id else None)
        return dependencies, parameter_trust

    def _typed_construction_result(
        self, result, operation: str, engine: str,
    ) -> MathResult:
        """Wrap a typed construction verdict that did not produce an object."""
        data = result.model_dump(mode="json")
        raw_status = getattr(result, "status", "unknown")
        raw_status = (
            raw_status.value if hasattr(raw_status, "value") else str(raw_status))
        raw_trust = getattr(result, "trust", "unknown")
        raw_trust = (
            raw_trust.value if hasattr(raw_trust, "value") else str(raw_trust))
        trust = (
            TrustLevel(raw_trust)
            if raw_trust in {item.value for item in TrustLevel}
            else TrustLevel.UNKNOWN
        )
        bundle, claim_evidence = extract_evidence(result)
        semantic_status = {
            "verified": ResultStatus.VERIFIED_EXACT,
            "refuted": ResultStatus.REFUTED,
            "does_not_exist": ResultStatus.DOES_NOT_EXIST,
            "unsupported": ResultStatus.UNSUPPORTED,
            "candidate": ResultStatus.CANDIDATE,
        }.get(raw_status)
        ok = raw_status not in {"error", "unsupported"}
        return MathResult(
            ok=ok,
            status=(
                "verified" if raw_status == "verified" else
                "refuted" if raw_status == "refuted" else
                "candidate" if raw_status == "candidate" else
                "unknown" if ok else "error"
            ),
            semantic_status=semantic_status,
            data=data,
            errors=[] if ok else list(data.get("conditions", [])),
            warnings=list(data.get("diagnostics", [])),
            trust=trust,
            engine=engine,
            evidence_bundle=bundle,
            claim_evidence=claim_evidence,
            mathkernel_version=MATHKERNEL_VERSION,
        )

    def object_create(self, object_type: str, definition: dict) -> MathResult:
        """Create one typed mathematical object through restricted MathIR."""
        kind = object_type.strip().lower()
        from .engineering_adapter import TYPES as engineering_types, construct as construct_engineering
        context_id = definition.get("context_id")
        try:
            context_assumptions = self._typed_context_assumptions(
                str(context_id) if context_id else None)
            if kind in engineering_types:
                stored_type, value, trust, sources = construct_engineering(self, kind, definition)
            elif kind in {"transform_problem", "transformproblem"}:
                from .integral_transforms import TransformProblem
                expression_id = str(definition["expression_id"])
                ir, err = self._get_expr(expression_id, "integral_transforms")
                if err:
                    return err
                texts = [
                    str(definition["variable"]),
                    str(definition["transform_variable"]),
                    *[str(item) for item in definition.get("assumptions", [])],
                ]
                _, parsed, extra_trust = self._typed_parse_many(texts, context_id)
                variable, target, *assumptions = parsed
                transform = str(definition["transform"]).lower()
                convention_name = definition.get("convention")
                convention = self._transform_convention(str(convention_name))
                env = self._symbol_env(self.contexts.get(context_id) if context_id else None,
                                       ir, *[parse_math(text) for text in texts])
                expression = self.sympy.to_sympy(ir, env)
                domain_raw = definition.get("domain")
                if isinstance(domain_raw, list):
                    _, domain_values, domain_trust = self._typed_parse_many(
                        [str(item) for item in domain_raw], context_id)
                    domain = tuple(domain_values)
                else:
                    domain_trust = TrustLevel.EXACT
                    domain = self._typed_domain(domain_raw, transform=transform.removeprefix("inverse_"))
                trust = self._trust_min(
                    self._expr_trust(ir), extra_trust, domain_trust)
                value = TransformProblem(
                    transform=transform,
                    expression=expression,
                    variable=variable,
                    transform_variable=target,
                    domain=domain,
                    assumptions=tuple(dict.fromkeys(
                        [*assumptions, *context_assumptions])),
                    convention=convention,
                )
                sources = [expression_id]
                stored_type = "TransformProblem"
            elif kind in {"complex_domain", "complexdomain"}:
                from .complex_analysis import BranchConvention, ComplexDomain
                assumption_texts = [
                    str(item) for item in definition.get("assumptions", [])]
                excluded_texts = [
                    str(item) for item in definition.get("excluded_points", [])]
                _, parsed, parsed_trust = self._typed_parse_many(
                    [str(definition.get("variable", "z")),
                     *assumption_texts, *excluded_texts],
                    context_id,
                )
                split = 1 + len(assumption_texts)
                region, region_trust = self._typed_set(
                    definition.get("region", "complexes"), context_id)
                trust = self._trust_min(parsed_trust, region_trust)
                value = ComplexDomain(
                    variable=parsed[0],
                    region=region,
                    assumptions=list(dict.fromkeys(
                        [*parsed[1:split], *context_assumptions])),
                    excluded_points=parsed[split:],
                    branch=BranchConvention.model_validate(
                        definition.get("branch") or {}),
                    input_trust=trust.value,
                )
                sources = []
                stored_type = "ComplexDomain"
            elif kind in {"complex_function", "complexfunction"}:
                from .complex_analysis import (
                    BranchConvention, ComplexDomain, ComplexFunction,
                )
                expression_id = str(definition["expression_id"])
                ir, err = self._get_expr(expression_id, "complex_analysis")
                if err:
                    return err
                assumption_texts = [str(item) for item in definition.get("assumptions", [])]
                excluded_texts = [str(item) for item in definition.get("excluded_points", [])]
                texts = [str(definition["variable"]), *assumption_texts, *excluded_texts]
                parsed_irs, parsed, extra_trust = self._typed_parse_many(texts, context_id)
                variable = parsed[0]
                split = 1 + len(assumption_texts)
                assumptions, excluded = parsed[1:split], parsed[split:]
                env = self._symbol_env(self.contexts.get(context_id) if context_id else None,
                                       ir, *parsed_irs)
                expression = self.sympy.to_sympy(ir, env)
                branch_data = definition.get("branch")
                domain_id = definition.get("domain_id")
                from .complex_analysis import ComplexAnalysisEngine
                if (
                    ComplexAnalysisEngine._branch_families(expression)
                    and branch_data is None
                    and domain_id is None
                ):
                    raise ValueError("branch-sensitive complex functions require explicit branch metadata")
                branch = BranchConvention.model_validate(branch_data or {})
                if domain_id is not None:
                    domain_record = self._get_math_object_record(str(domain_id))
                    if (
                        not domain_record
                        or domain_record["object_type"] != "ComplexDomain"
                    ):
                        raise ValueError(
                            f"Unknown ComplexDomain object: {domain_id}")
                    domain = domain_record["value"]
                    if domain.variable != variable:
                        raise ValueError(
                            "ComplexFunction and ComplexDomain variables differ")
                    if branch_data is None:
                        branch = domain.branch
                    domain_trust = domain_record["input_trust"]
                else:
                    region, region_trust = self._typed_set(
                        definition.get("region", "complexes"), context_id)
                    domain = ComplexDomain(
                        variable=variable, region=region,
                        assumptions=list(dict.fromkeys(
                            [*assumptions, *context_assumptions])),
                        excluded_points=excluded, branch=branch,
                        input_trust=region_trust.value)
                    domain_trust = region_trust
                trust = self._trust_min(self._expr_trust(ir), extra_trust)
                trust = self._trust_min(trust, domain_trust)
                value = ComplexFunction(
                    expression=expression, variable=variable, domain=domain,
                    branch=branch, input_trust=trust.value)
                sources = [expression_id] + (
                    [str(domain_id)] if domain_id is not None else [])
                stored_type = "ComplexFunction"
            elif kind == "contour":
                from .complex_analysis import Contour
                point_texts = [str(item) for item in definition.get("vertices", [])]
                if len(point_texts) > self.settings.max_contour_vertices:
                    raise ValueError(
                        "contour exceeds max_contour_vertices="
                        f"{self.settings.max_contour_vertices}")
                _, vertices, trust = self._typed_parse_many(point_texts, context_id)
                value = Contour(
                    vertices=vertices,
                    orientation=definition.get("orientation"),
                    assumptions=list(context_assumptions),
                    input_trust=trust.value,
                )
                sources = []
                stored_type = "Contour"
            elif kind == "distribution":
                from . import continuous_probability as cp
                family = str(definition["family"]).lower()
                variable_text = str(definition.get("variable", "x"))
                if family == "mixture":
                    component_ids = [
                        str(item) for item in definition.get("components", [])]
                    if (
                        len(component_ids)
                        > self.settings.max_distribution_components
                    ):
                        raise ValueError(
                            "mixture exceeds max_distribution_components="
                            f"{self.settings.max_distribution_components}")
                    component_records = [
                        self._get_math_object_record(item)
                        for item in component_ids]
                    if not component_records or any(
                        record is None or record["object_type"] != "Distribution"
                        for record in component_records
                    ):
                        raise ValueError(
                            "mixture components must reference Distribution objects")
                    weight_texts = [
                        str(item) for item in definition.get("weights", [])]
                    _, parsed, parsed_trust = self._typed_parse_many(
                        [variable_text, *weight_texts], context_id)
                    component_trust = self._trust_min(*[
                        record["input_trust"] for record in component_records
                    ])
                    trust = self._trust_min(component_trust, parsed_trust)
                    value = cp.mixture(
                        parsed[1:],
                        [record["value"] for record in component_records],
                        variable=parsed[0],
                    )
                    value.input_trust = trust.value
                    sources = component_ids
                    stored_type = "Distribution"
                    object_id = self._store_math_object(
                        stored_type, value, input_trust=trust, sources=sources)
                    return MathResult(
                        ok=True,
                        data={
                            "object_id": object_id,
                            "object_type": stored_type,
                            "object": value.model_dump(mode="json"),
                        },
                        trust=self._trust_min(TrustLevel.SYMBOLIC, trust),
                        engine="continuous_probability",
                    )
                parameter_texts = [
                    str(item) for item in definition.get("parameters", [])]
                _, parsed, trust = self._typed_parse_many(
                    [variable_text, *parameter_texts], context_id)
                variable, parameters = parsed[0], parsed[1:]
                constructors = {
                    "uniform": cp.Uniform, "normal": cp.Normal,
                    "lognormal": cp.LogNormal, "exponential": cp.Exponential,
                    "gamma": cp.Gamma, "beta": cp.Beta, "cauchy": cp.Cauchy,
                    "student_t": cp.StudentT, "chi_squared": cp.ChiSquared,
                    "f": cp.F, "weibull": cp.Weibull, "pareto": cp.Pareto,
                    "laplace": cp.Laplace, "logistic": cp.Logistic,
                }
                if family not in constructors:
                    raise ValueError(f"unsupported distribution: {family}")
                value = constructors[family](
                    *parameters, variable=variable, input_trust=trust.value,
                    assumptions=context_assumptions)
                sources = []
                stored_type = "Distribution"
            elif kind in {"joint_distribution", "jointdistribution"}:
                from .continuous_probability import JointDistribution
                expression_id = str(definition["density_expression_id"])
                ir, err = self._get_expr(
                    expression_id, "continuous_probability")
                if err:
                    return err
                variable_texts = [
                    str(item) for item in definition.get("variables", [])]
                if len(variable_texts) > self.settings.max_joint_dimensions:
                    raise ValueError(
                        "joint distribution exceeds max_joint_dimensions="
                        f"{self.settings.max_joint_dimensions}")
                variable_irs, variables, variable_trust = (
                    self._typed_parse_many(variable_texts, context_id))
                support_specs = definition.get("supports", [])
                if len(support_specs) != len(variables):
                    raise ValueError(
                        "supports must contain one set per joint variable")
                supports_with_trust = [
                    self._typed_set(item, context_id)
                    for item in support_specs
                ]
                supports = [item[0] for item in supports_with_trust]
                support_trust = self._trust_min(
                    *(item[1] for item in supports_with_trust))
                env = self._symbol_env(
                    self.contexts.get(context_id) if context_id else None,
                    ir, *variable_irs)
                density = self.sympy.to_sympy(ir, env)
                trust = self._trust_min(
                    self._expr_trust(ir), variable_trust, support_trust)
                value = JointDistribution(
                    variables=tuple(variables),
                    density=density,
                    support=sp.ProductSet(*supports),
                    input_trust=trust.value,
                    conditions=[
                        str(item)
                        for item in definition.get("conditions", [])
                    ],
                )
                sources = [expression_id]
                stored_type = "JointDistribution"
            elif kind in {
                "conditional_distribution", "conditionaldistribution",
            }:
                from .continuous_probability import ConditionalDistribution
                expression_id = str(definition["density_expression_id"])
                ir, err = self._get_expr(
                    expression_id, "continuous_probability")
                if err:
                    return err
                texts = [
                    str(definition["variable"]),
                    *[str(item) for item in definition.get("given", [])],
                    str(definition["condition"]),
                ]
                parsed_irs, parsed, parsed_trust = self._typed_parse_many(
                    texts, context_id)
                support, support_trust = self._typed_set(
                    definition["support"], context_id)
                env = self._symbol_env(
                    self.contexts.get(context_id) if context_id else None,
                    ir, *parsed_irs)
                density = self.sympy.to_sympy(ir, env)
                trust = self._trust_min(
                    self._expr_trust(ir), parsed_trust, support_trust)
                value = ConditionalDistribution(
                    variable=parsed[0],
                    given=tuple(parsed[1:-1]),
                    density=density,
                    support=support,
                    condition=parsed[-1],
                    input_trust=trust.value,
                    conditions=[
                        str(item)
                        for item in definition.get("conditions", [])
                    ],
                )
                sources = [expression_id]
                stored_type = "ConditionalDistribution"
            elif kind in {"random_variable", "randomvariable"}:
                from .continuous_probability import RandomVariable
                distribution_id = str(definition["distribution_id"])
                distribution_record = self._get_math_object_record(
                    distribution_id)
                if not distribution_record or distribution_record["object_type"] != "Distribution":
                    raise ValueError(f"Unknown Distribution object: {distribution_id}")
                _, parsed, symbol_trust = self._typed_parse_many(
                    [str(definition.get("symbol", "X"))], context_id)
                trust = self._trust_min(
                    distribution_record["input_trust"], symbol_trust)
                value = RandomVariable(
                    symbol=parsed[0], distribution=distribution_record["value"])
                sources = [distribution_id]
                stored_type = "RandomVariable"
            elif kind in {
                "graph", "directed_graph", "directedgraph",
                "weighted_graph", "weightedgraph", "multi_graph", "multigraph",
            }:
                from .graph_theory import (
                    DirectedGraph, Graph, MultiGraph, WeightedGraph,
                )
                models = {
                    "graph": Graph,
                    "directed_graph": DirectedGraph,
                    "directedgraph": DirectedGraph,
                    "weighted_graph": WeightedGraph,
                    "weightedgraph": WeightedGraph,
                    "multi_graph": MultiGraph,
                    "multigraph": MultiGraph,
                }
                graph_definition = dict(definition)
                if "nodes" in graph_definition and "vertices" not in graph_definition:
                    graph_definition["vertices"] = graph_definition.pop("nodes")
                value = models[kind].model_validate(graph_definition)
                if len(value.vertices) > self.settings.max_graph_vertices:
                    raise ValueError(
                        "graph exceeds max_graph_vertices="
                        f"{self.settings.max_graph_vertices}")
                if len(value.edges) > self.settings.max_graph_edges:
                    raise ValueError(
                        "graph exceeds max_graph_edges="
                        f"{self.settings.max_graph_edges}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = type(value).__name__
            elif kind in {"combinatorial_class", "combinatorialclass"}:
                from .combinatorics import CombinatorialClass
                value = CombinatorialClass.model_validate(definition)
                if value.max_items > self.settings.max_combinatorial_items:
                    raise ValueError(
                        "combinatorial generation exceeds "
                        f"max_combinatorial_items="
                        f"{self.settings.max_combinatorial_items}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "CombinatorialClass"
            elif kind in {"generating_function", "generatingfunction"}:
                from .combinatorics import GeneratingFunction
                value = GeneratingFunction.model_validate(definition)
                if len(value.coefficients) > self.settings.max_combinatorial_items:
                    raise ValueError(
                        "generating-function prefix exceeds "
                        f"max_combinatorial_items="
                        f"{self.settings.max_combinatorial_items}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "GeneratingFunction"
            elif kind in {"finite_group", "finitegroup"}:
                from .finite_groups import FiniteGroup
                if "cayley_table" not in definition and definition.get("cyclic"):
                    value = FiniteGroup.cyclic(int(definition["cyclic"]))
                else:
                    value = FiniteGroup.model_validate(definition)
                if value.order > self.settings.max_group_elements:
                    raise ValueError(
                        "finite group exceeds max_group_elements="
                        f"{self.settings.max_group_elements}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "FiniteGroup"
            elif kind in {"permutation_group", "permutationgroup"}:
                from .finite_groups import PermutationGroup
                family = definition.get("family")
                if family == "symmetric":
                    value = PermutationGroup.symmetric(int(definition["degree"]))
                elif family == "cyclic":
                    value = PermutationGroup.cyclic(int(definition["degree"]))
                elif family == "dihedral":
                    value = PermutationGroup.dihedral(int(definition["degree"]))
                else:
                    value = PermutationGroup.model_validate(definition)
                if value.degree > self.settings.max_group_elements:
                    raise ValueError(
                        "permutation degree exceeds max_group_elements="
                        f"{self.settings.max_group_elements}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "PermutationGroup"
            elif kind in {"finite_abelian_group", "finiteabeliangroup"}:
                from .finite_groups import FiniteAbelianGroup
                value = FiniteAbelianGroup.model_validate(definition)
                if value.order > self.settings.max_group_elements:
                    raise ValueError(
                        "finite abelian group exceeds max_group_elements="
                        f"{self.settings.max_group_elements}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "FiniteAbelianGroup"
            elif kind in {"group_homomorphism", "grouphomomorphism"}:
                from .finite_groups import GroupHomomorphism
                domain_id = str(definition["domain_id"])
                codomain_id = str(definition["codomain_id"])
                domain_record = self._get_math_object_record(domain_id)
                codomain_record = self._get_math_object_record(codomain_id)
                if (
                    not domain_record
                    or domain_record["object_type"] != "FiniteGroup"
                    or not codomain_record
                    or codomain_record["object_type"] != "FiniteGroup"
                ):
                    raise ValueError(
                        "group homomorphisms require FiniteGroup domain_id "
                        "and codomain_id")
                value = GroupHomomorphism(
                    domain=domain_record["value"],
                    codomain=codomain_record["value"],
                    images=[int(item) for item in definition["images"]],
                )
                trust = TrustLevel.EXACT
                sources = [domain_id, codomain_id]
                stored_type = "GroupHomomorphism"
            elif kind in {"finite_ring", "finitering"}:
                from .finite_algebra import create_finite_ring
                constructed = create_finite_ring(int(definition["modulus"]))
                if constructed.status.value not in {"verified", "available"}:
                    return self._typed_construction_result(
                        constructed, "finite_ring", "finite_algebra")
                value = constructed.value
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "FiniteRing"
            elif kind in {"finite_field", "finitefield"}:
                from .finite_algebra import create_finite_field
                modulus = [
                    int(item) for item in definition["modulus_coeffs"]]
                if len(modulus) - 1 > self.settings.max_field_degree:
                    raise ValueError(
                        "finite field exceeds max_field_degree="
                        f"{self.settings.max_field_degree}")
                constructed = create_finite_field(
                    int(definition["prime"]), modulus)
                if constructed.status.value not in {"verified", "available"}:
                    return self._typed_construction_result(
                        constructed, "finite_field", "finite_algebra")
                value = constructed.value
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "FiniteField"
            elif kind in {"module", "module_presentation", "modulepresentation"}:
                from .finite_algebra import ModulePresentation
                value = ModulePresentation.model_validate(definition)
                if (
                    len(value.generators) > self.settings.max_normal_form_dim
                    or any(
                        len(row) > self.settings.max_normal_form_dim
                        for row in value.relations)
                ):
                    raise ValueError(
                        "module presentation exceeds max_normal_form_dim="
                        f"{self.settings.max_normal_form_dim}")
                trust = TrustLevel.EXACT
                sources = []
                stored_type = "Module"
            else:
                raise ValueError(
                    "object_type must be TransformProblem, ComplexDomain, "
                    "ComplexFunction, Contour, Distribution, "
                    "JointDistribution, ConditionalDistribution, "
                    "RandomVariable, Graph, DirectedGraph, WeightedGraph, "
                    "MultiGraph, CombinatorialClass, GeneratingFunction, "
                    "FiniteGroup, PermutationGroup, FiniteAbelianGroup, "
                    "GroupHomomorphism, FiniteRing, FiniteField, Module, DiscreteSignal, "
                    "Spectrum, Filter, TransferFunction, StateSpaceSystem, "
                    "OptimizationProblem, or OptimizationCertificate")
            object_id = self._store_math_object(
                stored_type, value, input_trust=trust, sources=sources)
        except (KeyError, TypeError, ValueError) as exc:
            return MathResult(
                ok=False, status="error", errors=[str(exc)], engine="math_object")
        return MathResult(
            ok=True,
            data={
                "object_id": object_id,
                "object_type": stored_type,
                "object": value.model_dump(mode="json"),
            },
            trust=trust,
            engine="math_object",
        )

    def object_get(self, object_id: str) -> MathResult:
        try:
            record = self._get_math_object_record(object_id)
        except (TypeError, ValueError) as exc:
            return MathResult(
                ok=False, status="error", errors=[str(exc)],
                engine="math_object")
        if record is None:
            return MathResult(
                ok=False, status="error", errors=[f"Unknown object_id: {object_id}"],
                engine="math_object")
        return MathResult(
            ok=True,
            status=(
                "verified"
                if record["semantic_status"] in {
                    ResultStatus.PROVED,
                    ResultStatus.VERIFIED_EXACT,
                    ResultStatus.VERIFIED_SYMBOLIC,
                    ResultStatus.CERTIFIED,
                    ResultStatus.VERIFIED_NUMERIC,
                }
                else "ok"
            ),
            data={
                "object_id": object_id,
                "object_type": record["object_type"],
                "object": self._typed_object_payload(record["value"]),
                "sources": record["sources"],
                "semantic_status": record["semantic_status"].value,
            },
            trust=record["input_trust"],
            engine="math_object",
            semantic_status=record["semantic_status"],
            evidence_bundle=record["evidence_bundle"],
            claim_evidence=record["claim_evidence"],
        )

    @staticmethod
    def _typed_object_payload(value) -> dict:
        """Return a bounded public representation of a stored typed object."""
        payload = value.model_dump(mode="json")
        if type(value).__name__ == "SDESimulation":
            payload = {key: item for key, item in payload.items()
                       if key not in {"times", "values"}}
            payload.update({
                "time_range": [str(value.times[0]), str(value.times[-1])],
                "stored_fields": ["times", "values"],
                "query_operations": ["path", "terminal_values"],
            })
        return payload

    def _typed_result(
        self, result, operation: str, object_id: str, engine: str,
    ) -> MathResult:
        data = result.model_dump(mode="json")
        raw_trust = getattr(result, "trust", "unknown")
        trust_text = raw_trust.value if hasattr(raw_trust, "value") else str(raw_trust)
        trust = TrustLevel(trust_text) if trust_text in {item.value for item in TrustLevel} \
            else TrustLevel.UNKNOWN
        raw_status = getattr(result, "status", "unknown")
        raw_status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        semantic_status = None
        if raw_status == "infeasible":
            semantic_status = ResultStatus.INFEASIBLE
        elif raw_status == "unbounded":
            semantic_status = ResultStatus.UNBOUNDED
        elif raw_status == "does_not_exist":
            semantic_status = ResultStatus.DOES_NOT_EXIST
        elif raw_status == "refuted":
            semantic_status = ResultStatus.REFUTED
        elif raw_status == "unsupported":
            semantic_status = ResultStatus.UNSUPPORTED
        elif raw_status == "candidate":
            semantic_status = ResultStatus.CANDIDATE
        ok = raw_status not in {"error", "unsupported"}
        outward = (
            "error" if not ok else
            "verified" if raw_status in {"verified", "infeasible", "unbounded"} else
            "refuted" if raw_status == "refuted" else
            "ok" if raw_status == "available" else
            "candidate" if raw_status == "candidate" else
            "unknown"
        )
        bundle, claim_evidence = extract_evidence(result)
        engine_versions = {
            "mathkernel": MATHKERNEL_VERSION,
            "sympy": sp.__version__,
            "python": platform.python_version(),
        }
        if engine == "engineering":
            for backend in ("numpy", "scipy", "mpmath", "clarabel"):
                module = sys.modules.get(backend)
                version = getattr(module, "__version__", None)
                if version is not None:
                    engine_versions[backend] = str(version)
        for evidence in [
            bundle,
            *claim_evidence.values(),
        ]:
            for computation in evidence.computation:
                computation.metadata.setdefault(
                    "adapter", engine)
                computation.metadata.setdefault(
                    "implementation", computation.method)
                computation.metadata.setdefault(
                    "engine_versions", dict(engine_versions))
                computation.metadata.setdefault(
                    "platform", platform.platform())
        source_record = self._get_math_object_record(object_id) or {}
        source_trust = source_record.get(
            "input_trust", TrustLevel.UNKNOWN)
        arithmetic = next(
            (
                evidence.arithmetic
                for evidence in bundle.computation
                if evidence.arithmetic
            ),
            "symbolic",
        )
        if arithmetic == "inherited":
            arithmetic = "symbolic"
        data["provenance"] = {
            "mathkernel_version": MATHKERNEL_VERSION,
            "engine_versions": engine_versions,
            "adapter": engine,
            "operation": operation,
            "arithmetic_transition": {
                "input_trust": source_trust.value,
                "computation": arithmetic,
                "justified_output_trust": trust.value,
            },
        }
        conditions = []
        for condition in [
            *data.get("conditions", []),
            *data.get("side_conditions", []),
        ]:
            if condition not in conditions:
                conditions.append(condition)
        step = self._record(DerivationStep(
            step_id=self._id("step"),
            operation=operation,
            inputs=[object_id],
            parents=[],
            output=str(data.get("value", raw_status)),
            engine=engine,
            trust=trust,
            conditions=conditions,
            evidence_bundle=bundle.model_copy(deep=True),
            claim_evidence={
                name: evidence.model_copy(deep=True)
                for name, evidence in claim_evidence.items()
            },
        ))
        return MathResult(
            ok=ok,
            status=outward,
            semantic_status=semantic_status,
            data=data,
            side_conditions=conditions,
            warnings=list(data.get("diagnostics", [])),
            trust=trust,
            engine=engine,
            derivation=[step],
            evidence_bundle=bundle,
            claim_evidence=claim_evidence,
            engine_versions=engine_versions,
            mathkernel_version=MATHKERNEL_VERSION,
            arithmetic_transition=data["provenance"][
                "arithmetic_transition"],
        )

    def apply(
        self, object_id: str, operation: str, parameters: dict | None = None,
    ) -> MathResult:
        """Execute a typed operation through the shared obligation executor."""
        return self.executor.execute_typed_apply(
            object_id=object_id,
            operation=operation,
            parameters=dict(parameters or {}),
            compute=lambda: self._apply_typed_operation(
                object_id, operation, parameters),
        )

    def _apply_typed_operation(
        self, object_id: str, operation: str, parameters: dict | None = None,
    ) -> MathResult:
        """Domain-adapter candidate computation used by the obligation executor."""
        record = self._get_math_object_record(object_id)
        if record is None:
            return MathResult(
                ok=False, status="error", errors=[f"Unknown object_id: {object_id}"],
                engine="math_object")
        parameters = parameters or {}
        object_type, value = record["object_type"], record["value"]
        context_id = parameters.get("context_id")
        try:
            capability = self.router.registry.resolve(
                input_type=object_type, operation=operation)
        except LookupError as exc:
            return MathResult(
                ok=False,
                status="error",
                semantic_status=ResultStatus.UNSUPPORTED,
                errors=[str(exc)],
                engine="capability_registry",
            )
        supported_handlers = {
            "module:integral_transforms",
            "module:continuous_probability",
            "module:complex_analysis",
            "module:graph_theory",
            "module:combinatorics",
            "module:finite_groups",
            "module:finite_algebra",
            "module:engineering",
            "composition:distribution_integral_transform",
        }
        if capability.handler not in supported_handlers:
            return MathResult(
                ok=False, status="error",
                semantic_status=ResultStatus.UNSUPPORTED,
                errors=[
                    f"Capability {capability.name} has no executable typed "
                    f"handler: {capability.handler!r}"],
                engine="capability_registry")
        def finish(result, name: str, engine_name: str) -> MathResult:
            return self._typed_result(result, name, object_id, engine_name)

        def finish_distribution(
            distribution, sources: list[str], method: str,
            stored_object_type: str = "Distribution",
        ) -> MathResult:
            derived_trust = TrustLevel(distribution.input_trust)
            new_id = self._store_math_object(
                stored_object_type, distribution, input_trust=derived_trust,
                sources=sources)
            trust = self._trust_min(TrustLevel.SYMBOLIC, derived_trust)
            bundle = EvidenceBundle(computation=[ComputationEvidence(
                engine="continuous_probability",
                method=method,
                arithmetic="symbolic",
                deterministic=True,
                trust=trust.value,
            )])
            step = self._record(DerivationStep(
                step_id=self._id("step"), operation=f"probability:{operation}",
                inputs=sources, output=new_id, engine="continuous_probability",
                trust=trust, evidence_bundle=bundle.model_copy(deep=True),
                claim_evidence={
                    "distribution": bundle.model_copy(deep=True),
                }))
            wrapped = MathResult(
                ok=True,
                data={
                    "object_id": new_id,
                    "object_type": stored_object_type,
                    "object": distribution.model_dump(mode="json"),
                },
                trust=trust,
                engine="continuous_probability",
                derivation=[step],
                evidence_bundle=bundle.model_copy(deep=True),
                claim_evidence={
                    "distribution": bundle.model_copy(deep=True),
                },
            )
            derived_record = self.math_objects[new_id]
            derived_record["evidence_bundle"] = bundle.model_copy(deep=True)
            derived_record["claim_evidence"] = {
                "distribution": bundle.model_copy(deep=True),
            }
            derived_record["semantic_status"] = (
                wrapped.semantic_status or ResultStatus.CANDIDATE)
            self._persist_math_object(new_id, derived_record)
            return wrapped

        try:
            if capability.handler == "module:engineering":
                from .engineering_adapter import apply as apply_engineering
                computed = apply_engineering(self, value, object_type, operation, parameters, object_id=object_id)
                derived = None
                if isinstance(computed, tuple):
                    computed, derived = computed
                wrapped = finish(computed, f"engineering:{operation}", "engineering")
                if derived is not None and wrapped.ok:
                    outputs = derived if isinstance(derived, dict) else {"output": derived}
                    if len(self.math_objects)+len(outputs) > self.settings.max_math_objects:
                        raise ValueError("multiple derived objects exceed max_math_objects")
                    ids = {}
                    for role, node in outputs.items():
                        derived_sources = [object_id]
                        for key, candidate in parameters.items():
                            if not key.endswith("_id") or candidate is None:
                                continue
                            candidate_id = str(candidate)
                            if (candidate_id not in derived_sources and
                                    self._get_math_object_record(candidate_id) is not None):
                                derived_sources.append(candidate_id)
                        new_id = self._store_math_object(type(node).__name__, node,
                            input_trust=wrapped.trust, sources=derived_sources)
                        ids[role] = new_id
                        new_record = self.math_objects[new_id]
                        new_record["evidence_bundle"] = wrapped.evidence_bundle.model_copy(deep=True)
                        new_record["claim_evidence"] = {name: evidence.model_copy(deep=True)
                                                      for name, evidence in wrapped.claim_evidence.items()}
                        new_record["semantic_status"] = wrapped.semantic_status or ResultStatus.CANDIDATE
                        self._persist_math_object(new_id, new_record)
                    first = "output" if "output" in ids else next(iter(ids))
                    output_type = type(outputs[first]).__name__
                    payload = self._typed_object_payload(outputs[first])
                    wrapped.data.update({"object_id": ids[first], "object_ids": ids,
                        "object_type": output_type, "object": payload})
                return wrapped
            if capability.handler == "module:graph_theory":
                from .graph_theory import (
                    DirectedGraph, Graph, GraphResult, SearchLimits,
                    WeightedGraph, bipartite_matching, breadth_first_search,
                    connected_components, degree_centrality, depth_first_search,
                    euler_trail, find_cycle, graph_coloring, graph_isomorphism,
                    maximum_flow, minimum_spanning_tree, shortest_paths,
                    strongly_connected_components, topological_sort,
                    verify_graph,
                )

                def unsupported(reason: str) -> GraphResult:
                    return GraphResult(
                        status=OperationStatus.UNSUPPORTED,
                        diagnostics=[reason],
                    )

                is_directed = isinstance(value, DirectedGraph) or (
                    isinstance(value, WeightedGraph) and value.directed)
                if operation == "verify":
                    result = verify_graph(value)
                elif operation == "bfs":
                    result = breadth_first_search(value, parameters["source"])
                elif operation == "dfs":
                    result = depth_first_search(value, parameters["source"])
                elif operation == "connected_components":
                    result = (
                        unsupported("connected_components requires an undirected graph")
                        if is_directed else connected_components(value))
                elif operation == "strongly_connected_components":
                    result = (
                        strongly_connected_components(value) if is_directed else
                        unsupported(
                            "strongly_connected_components requires a directed graph"))
                elif operation == "shortest_path":
                    result = run_with_timeout(
                        shortest_paths,
                        self.settings.solver_timeout_seconds,
                        value, parameters["source"], parameters.get("target"))
                elif operation == "minimum_spanning_tree":
                    result = (
                        minimum_spanning_tree(value)
                        if isinstance(value, WeightedGraph) and not is_directed else
                        unsupported(
                            "minimum_spanning_tree requires an undirected "
                            "WeightedGraph"))
                elif operation in {"maximum_flow", "minimum_cut"}:
                    result = (
                        run_with_timeout(
                            maximum_flow,
                            self.settings.solver_timeout_seconds,
                            value, parameters["source"], parameters["sink"])
                        if isinstance(value, WeightedGraph) and is_directed else
                        unsupported(
                            f"{operation} requires a directed WeightedGraph"))
                elif operation == "matching":
                    result = (
                        unsupported("matching requires an undirected graph")
                        if is_directed else bipartite_matching(value))
                elif operation == "euler_path":
                    result = (
                        unsupported("euler_path requires an undirected graph")
                        if is_directed else euler_trail(value))
                elif operation == "coloring":
                    if is_directed:
                        result = unsupported("coloring requires an undirected graph")
                    else:
                        limits = SearchLimits(
                            max_vertices=self.settings.max_graph_vertices,
                            max_operations=int(parameters.get(
                                "max_operations", 100_000)),
                            exact_coloring_max_vertices=int(parameters.get(
                                "exact_coloring_max_vertices", 24)),
                        )
                        result = run_with_timeout(
                            graph_coloring,
                            self.settings.solver_timeout_seconds,
                            value, limits)
                elif operation == "topological_sort":
                    result = (
                        topological_sort(value) if is_directed else
                        unsupported("topological_sort requires a directed graph"))
                elif operation == "cycle_detection":
                    result = find_cycle(value)
                elif operation == "centrality":
                    result = degree_centrality(value)
                elif operation == "isomorphic_to":
                    other_id = str(parameters["other_id"])
                    other = self._get_math_object_record(other_id)
                    if other is None or not isinstance(other["value"], Graph):
                        raise ValueError(
                            f"Unknown simple graph object: {other_id}")
                    limits = SearchLimits(
                        max_vertices=self.settings.max_graph_vertices,
                        max_operations=int(parameters.get(
                            "max_operations", 100_000)),
                    )
                    result = run_with_timeout(
                        graph_isomorphism,
                        self.settings.solver_timeout_seconds,
                        value, other["value"], limits)
                else:
                    raise ValueError(f"Graph does not support {operation}")
                return finish(result, f"graph:{operation}", "graph_theory")
            if capability.handler == "module:combinatorics":
                from .combinatorics import CombinatoricsEngine
                engine = CombinatoricsEngine()
                if object_type == "CombinatorialClass":
                    if operation == "count":
                        result = value.count()
                    elif operation == "generate":
                        limit = int(
                            parameters.get("limit", value.max_items))
                        if limit > self.settings.max_combinatorial_items:
                            raise ValueError(
                                "generation exceeds max_combinatorial_items="
                                f"{self.settings.max_combinatorial_items}")
                        result = value.generate(max_items=limit)
                    elif operation == "verify":
                        result = run_with_timeout(
                            engine.verify_count_identities,
                            self.settings.solver_timeout_seconds,
                            value.n)
                    else:
                        raise ValueError(
                            "CombinatorialClass supports count, generate, or verify")
                elif object_type == "GeneratingFunction":
                    if operation == "coefficient":
                        result = engine.coefficient(
                            value, int(parameters["n"]))
                    elif operation == "recurrence":
                        if value.recurrence is not None:
                            from .combinatorics import GfConversionResult
                            result = GfConversionResult(
                                status=OperationStatus.AVAILABLE,
                                recurrence=value.recurrence,
                                trust=TrustLevel.EXACT,
                                evidence=EvidenceBundle(computation=[
                                    ComputationEvidence(
                                        engine="combinatorics",
                                        method="existing_recurrence",
                                        arithmetic="exact",
                                        deterministic=True,
                                        trust="exact",
                                    )]),
                            )
                        elif value.rational is not None:
                            result = engine.recurrence_from_ogf(value.rational)
                        else:
                            from .combinatorics import GfConversionResult
                            result = GfConversionResult(
                                status=OperationStatus.UNKNOWN,
                                diagnostics=[
                                    "no recurrence or rational form available"],
                            )
                    elif operation == "verify":
                        result = engine.verify_generating_function(
                            value, parameters.get("terms"))
                    else:
                        raise ValueError(
                            "GeneratingFunction supports coefficient, recurrence, "
                            "or verify")
                else:
                    raise ValueError(
                        f"unsupported combinatorics object: {object_type}")
                return finish(
                    result, f"combinatorics:{operation}", "combinatorics")
            if capability.handler == "module:finite_groups":
                if object_type == "FiniteGroup":
                    if operation in {"verify", "order"}:
                        result = value.order_result()
                    elif operation in {"closure", "generated_subgroup"}:
                        result = value.generated_subgroup(
                            [int(item) for item in parameters.get(
                                "elements", parameters.get("generators", []))])
                    elif operation == "subgroups":
                        result = value.all_subgroups()
                    elif operation == "cosets":
                        result = value.cosets(
                            [int(item) for item in parameters["subgroup"]],
                            side=parameters.get("side", "left"))
                    elif operation == "normality":
                        result = value.is_normal(
                            [int(item) for item in parameters["subgroup"]])
                    elif operation == "quotient":
                        result = value.quotient(
                            [int(item) for item in parameters["subgroup"]])
                    elif operation == "center":
                        result = value.center()
                    elif operation == "centralizer":
                        subset = parameters.get("subset", [parameters["element"]])
                        result = value.centralizer(
                            [int(item) for item in subset])
                    elif operation == "conjugacy_classes":
                        result = value.conjugacy_classes()
                    elif operation == "commutator_subgroup":
                        result = value.commutator_subgroup()
                    elif operation in {"orbits", "stabilizers"}:
                        from .finite_groups import GroupAction
                        action = GroupAction(
                            group=value,
                            n_points=int(parameters["n_points"]),
                            action=[
                                [int(point) for point in row]
                                for row in parameters["action"]
                            ],
                        )
                        result = (
                            action.orbits() if operation == "orbits" else
                            action.stabilizer(int(parameters["point"])))
                    else:
                        raise ValueError(
                            f"FiniteGroup does not support {operation}")
                elif object_type == "PermutationGroup":
                    if operation in {"verify", "order"}:
                        result = value.order_result()
                    elif operation == "contains":
                        result = value.contains(
                            [int(item) for item in parameters["images"]])
                    elif operation == "orbits":
                        result = value.orbit_of(int(parameters["point"]))
                    elif operation == "stabilizers":
                        result = value.stabilizer_of(int(parameters["point"]))
                    elif operation == "stabilizer_chain":
                        result = value.stabilizer_chain()
                    else:
                        raise ValueError(
                            f"PermutationGroup does not support {operation}")
                elif object_type == "FiniteAbelianGroup":
                    from .finite_groups import OrderResult
                    bundle = EvidenceBundle(computation=[ComputationEvidence(
                        engine="finite_groups", method="invariant_factor_order",
                        arithmetic="exact", deterministic=True, trust="exact",
                    )])
                    if operation == "verify":
                        result = OrderResult(
                            status=OperationStatus.VERIFIED,
                            trust=TrustLevel.EXACT,
                            value=value.order,
                            evidence=bundle,
                        )
                    elif operation == "order":
                        result = OrderResult(
                            status=OperationStatus.AVAILABLE,
                            trust=TrustLevel.EXACT,
                            value=value.order,
                            evidence=bundle,
                        )
                    else:
                        raise ValueError(
                            f"FiniteAbelianGroup does not support {operation}")
                elif object_type == "GroupHomomorphism":
                    if operation == "verify":
                        result = value.image()
                    elif operation == "kernel":
                        result = value.kernel()
                    elif operation == "image":
                        result = value.image()
                    else:
                        raise ValueError(
                            f"GroupHomomorphism does not support {operation}")
                else:
                    raise ValueError(f"unsupported group object: {object_type}")
                return finish(result, f"finite_group:{operation}", "finite_groups")
            if capability.handler == "module:finite_algebra":
                from . import finite_algebra as algebra
                if object_type == "FiniteRing":
                    if operation == "verify":
                        result = algebra.create_finite_ring(value.modulus)
                    elif operation == "add":
                        result = algebra.ring_add(
                            value, int(parameters["left"]),
                            int(parameters["right"]))
                    elif operation == "multiply":
                        result = algebra.ring_mul(
                            value, int(parameters["left"]),
                            int(parameters["right"]))
                    elif operation == "inverse":
                        element = parameters.get(
                            "element", parameters.get("left"))
                        result = algebra.ring_inverse(value, int(element))
                    else:
                        raise ValueError(
                            f"FiniteRing does not support {operation}")
                elif object_type == "FiniteField":
                    if operation == "verify":
                        result = algebra.create_finite_field(
                            value.prime, value.modulus_coeffs)
                    elif operation == "add":
                        result = algebra.field_add(
                            value, parameters["left"], parameters["right"])
                    elif operation == "multiply":
                        result = algebra.field_mul(
                            value, parameters["left"], parameters["right"])
                    elif operation == "inverse":
                        element = parameters.get(
                            "element", parameters.get("left"))
                        result = algebra.field_inverse(value, element)
                    else:
                        raise ValueError(
                            f"FiniteField does not support {operation}")
                elif object_type == "Module":
                    if operation == "smith_normal_form":
                        result = algebra.smith_normal_form(value.relations)
                    elif operation == "hermite_normal_form":
                        result = algebra.hermite_normal_form(value.relations)
                    elif operation in {"verify", "abelian_group"}:
                        result = algebra.decompose_abelian_group(value)
                    else:
                        raise ValueError(f"Module does not support {operation}")
                else:
                    raise ValueError(
                        f"unsupported finite-algebra object: {object_type}")
                return finish(
                    result, f"finite_algebra:{operation}", "finite_algebra")
            if (
                capability.handler == "module:integral_transforms"
                and object_type == "TransformProblem"
            ):
                from .integral_transforms import TransformEngine
                if operation not in {"apply", "solve", "verify"}:
                    raise ValueError("TransformProblem supports apply, solve, or verify")
                verify_requested = (
                    bool(parameters["verify"])
                    if "verify" in parameters
                    else operation != "solve"
                )
                result = run_with_timeout(
                    TransformEngine().transform,
                    self.settings.solver_timeout_seconds,
                    value,
                    input_trust=record["input_trust"],
                    verify=verify_requested,
                )
                if not verify_requested:
                    result.side_conditions.append(
                        "Verification was explicitly skipped for solve; "
                        "the returned transform is a candidate computation.")
                return finish(
                    result, f"transform:{value.transform}", "integral_transforms")
            if (
                capability.handler in {
                    "module:continuous_probability",
                    "composition:distribution_integral_transform",
                }
                and object_type == "Distribution"
            ):
                from . import continuous_probability as cp
                if operation == "integral_transform":
                    from .integral_transforms import (
                        TransformEngine, TransformProblem,
                    )
                    transform = str(parameters["transform"]).lower()
                    if transform not in {"laplace", "fourier", "mellin"}:
                        raise ValueError(
                            "continuous distributions support laplace, "
                            "fourier, or mellin density transforms")
                    required_support = {
                        "laplace": sp.Interval(0, sp.oo),
                        "fourier": sp.S.Reals,
                        "mellin": sp.Interval.open(0, sp.oo),
                    }[transform]
                    if value.support.is_subset(required_support) is not True:
                        raise ValueError(
                            f"{transform} composition requires support within "
                            f"{required_support}; got {value.support}")
                    supported_density = value.density
                    if value.support != required_support:
                        supported_density = sp.Piecewise(
                            (value.density,
                             value.support.contains(value.variable)),
                            (sp.S.Zero, True),
                        )
                    target_defaults = {
                        "laplace": "s", "fourier": "omega", "mellin": "s",
                    }
                    assumption_texts = [
                        str(item)
                        for item in parameters.get("assumptions", [])
                    ]
                    _, parsed, parameter_trust = self._typed_parse_many(
                        [str(parameters.get(
                            "transform_variable",
                            target_defaults[transform])),
                         *assumption_texts],
                        context_id,
                    )
                    problem = TransformProblem(
                        transform=transform,
                        expression=supported_density,
                        variable=value.variable,
                        transform_variable=parsed[0],
                        domain=required_support,
                        assumptions=tuple(parsed[1:]),
                        convention=self._transform_convention(
                            str(parameters["convention"])),
                    )
                    result = run_with_timeout(
                        TransformEngine().transform,
                        self.settings.solver_timeout_seconds,
                        problem,
                        input_trust=self._trust_min(
                            record["input_trust"], parameter_trust),
                        verify=bool(parameters.get("verify", True)),
                    )
                    result.evidence_bundle.computation.append(
                        ComputationEvidence(
                            engine="continuous_probability",
                            method="distribution_density_source",
                            arithmetic="symbolic",
                            trust=record["input_trust"].value,
                            metadata={
                                "source_object_id": object_id,
                                "support": sp.sstr(value.support),
                            },
                        ))
                    return finish(
                        result, "composition:distribution_integral_transform",
                        "integral_transforms")
                if operation == "order_statistic":
                    sample_size = self._bounded_positive_int(
                        parameters["sample_size"], "sample_size",
                        self.settings.max_order_statistic_sample_size)
                    distribution = run_with_timeout(
                        value.order_statistic,
                        self.settings.solver_timeout_seconds,
                        sample_size,
                        int(parameters["order"]),
                    )
                    return finish_distribution(
                        distribution, [object_id],
                        "order_statistic_density")
                if operation == "mixture":
                    other_ids = [
                        str(item) for item in parameters.get("other_ids", [])]
                    if (
                        len(other_ids) + 1
                        > self.settings.max_distribution_components
                    ):
                        raise ValueError(
                            "mixture exceeds max_distribution_components="
                            f"{self.settings.max_distribution_components}")
                    component_records = [
                        record,
                        *[
                            self._get_math_object_record(item)
                            for item in other_ids
                        ],
                    ]
                    if any(
                        item is None
                        or item["object_type"] != "Distribution"
                        for item in component_records
                    ):
                        raise ValueError(
                            "all mixture components must be Distribution objects")
                    weight_texts = [
                        str(item) for item in parameters.get("weights", [])]
                    _, weights, weight_trust = self._typed_parse_many(
                        weight_texts, context_id)
                    distribution = run_with_timeout(
                        cp.mixture, self.settings.solver_timeout_seconds,
                        weights,
                        [item["value"] for item in component_records],
                        variable=value.variable,
                    )
                    distribution.input_trust = self._trust_min(
                        weight_trust,
                        *(item["input_trust"]
                          for item in component_records),
                    ).value
                    return finish_distribution(
                        distribution, [object_id, *other_ids],
                        "mixture_density_normalization")
                if operation == "expectation":
                    expression_id = str(parameters["expression_id"])
                    ir, err = self._get_expr(expression_id, "continuous_probability")
                    if err:
                        return err
                    env = self._symbol_env(
                        self.contexts.get(context_id) if context_id else None, ir)
                    expression = self.sympy.to_sympy(ir, env)
                    result = run_with_timeout(
                        cp.expectation, self.settings.solver_timeout_seconds,
                        value, expression)
                    expression_trust = self._expr_trust(ir)
                    result.trust = self._trust_min(
                        TrustLevel(result.trust), expression_trust).value
                    for bundle in result.claim_evidence.values():
                        bundle.computation.append(ComputationEvidence(
                            engine="mathir",
                            method="expectation_expression_ancestry",
                            arithmetic="inherited",
                            trust=expression_trust.value,
                        ))
                        bundle.justified_trust = result.trust
                elif operation == "truncate":
                    _, bounds, bound_trust = self._typed_parse_many(
                        [str(parameters["lower"]), str(parameters["upper"])],
                        context_id)
                    distribution = run_with_timeout(
                        cp.truncate, self.settings.solver_timeout_seconds,
                        value, bounds[0], bounds[1])
                    distribution.input_trust = self._trust_min(
                        record["input_trust"], bound_trust).value
                    return finish_distribution(
                        distribution, [object_id], "truncated_density_normalization")
                elif operation == "convolve":
                    other_id = str(parameters["other_id"])
                    other = self._get_math_object_record(other_id)
                    if not other or other["object_type"] != "Distribution":
                        raise ValueError(f"Unknown Distribution object: {other_id}")
                    distribution = run_with_timeout(
                        cp.convolve, self.settings.solver_timeout_seconds,
                        value, other["value"])
                    distribution.input_trust = self._trust_min(
                        record["input_trust"], other["input_trust"]).value
                    return finish_distribution(
                        distribution, [object_id, other_id], "symbolic_convolution")
                elif operation in {"cross_entropy", "kl_divergence"}:
                    other_id = str(parameters["other_id"])
                    other = self._get_math_object_record(other_id)
                    if not other or other["object_type"] != "Distribution":
                        raise ValueError(f"Unknown Distribution object: {other_id}")
                    function = (
                        cp.cross_entropy if operation == "cross_entropy"
                        else cp.kl_divergence)
                    result = run_with_timeout(
                        function, self.settings.solver_timeout_seconds,
                        value, other["value"])
                elif operation == "verify":
                    result = run_with_timeout(
                        value.verify, self.settings.solver_timeout_seconds)
                else:
                    query_operation = (
                        str(parameters["query"])
                        if operation == "query"
                        else operation
                    )
                    point = None
                    if parameters.get("point") is not None:
                        _, parsed, _ = self._typed_parse_many(
                            [str(parameters["point"])], context_id)
                        point = parsed[0]
                    result = run_with_timeout(
                        value.query, self.settings.solver_timeout_seconds,
                        query_operation, point=point,
                        order=parameters.get("order"))
                return finish(
                    result, f"probability:{operation}", "continuous_probability")
            if (
                capability.handler == "module:continuous_probability"
                and object_type == "JointDistribution"
            ):
                from .continuous_probability import Distribution, JointDistribution
                if operation == "verify":
                    result = run_with_timeout(
                        value.verify, self.settings.solver_timeout_seconds)
                    return finish(
                        result, "probability:joint_verify",
                        "continuous_probability")
                if operation == "marginal":
                    variable_texts = parameters.get("variables")
                    if variable_texts is None:
                        variable_texts = [parameters["variable"]]
                    _, variables, _ = self._typed_parse_many(
                        [str(item) for item in variable_texts], context_id)
                    marginal = run_with_timeout(
                        value.marginal, self.settings.solver_timeout_seconds,
                        tuple(variables))
                    output_type = (
                        "Distribution"
                        if isinstance(marginal, Distribution)
                        else "JointDistribution"
                    )
                    return finish_distribution(
                        marginal, [object_id], "joint_marginalization",
                        output_type)
                if operation in {"condition", "bayes"}:
                    target_text = str(parameters["variable"])
                    given_raw = parameters.get("given", {})
                    key_texts = [str(item) for item in given_raw]
                    value_texts = [str(item) for item in given_raw.values()]
                    _, parsed, parameter_trust = self._typed_parse_many(
                        [target_text, *key_texts, *value_texts], context_id)
                    split = 1 + len(key_texts)
                    given = dict(zip(
                        parsed[1:split], parsed[split:], strict=True))
                    function = (
                        value.condition
                        if operation == "condition"
                        else value.bayes
                    )
                    conditioned = run_with_timeout(
                        function, self.settings.solver_timeout_seconds,
                        parsed[0], given)
                    conditioned.input_trust = self._trust_min(
                        record["input_trust"], parameter_trust).value
                    return finish_distribution(
                        conditioned, [object_id],
                        "bayes_conditional_normalization",
                        "ConditionalDistribution")
                if operation in {"covariance", "correlation"}:
                    _, variables, parameter_trust = self._typed_parse_many(
                        [str(parameters["left"]), str(parameters["right"])],
                        context_id)
                    function = (
                        value.covariance
                        if operation == "covariance"
                        else value.correlation
                    )
                    result = run_with_timeout(
                        function, self.settings.solver_timeout_seconds,
                        variables[0], variables[1])
                    if parameter_trust != TrustLevel.EXACT:
                        result.trust = self._trust_min(
                            TrustLevel(result.trust),
                            parameter_trust).value
                    return finish(
                        result, f"probability:{operation}",
                        "continuous_probability")
                if operation == "order_statistic":
                    _, variables, _ = self._typed_parse_many(
                        [str(parameters["variable"])], context_id)
                    sample_size = self._bounded_positive_int(
                        parameters["sample_size"], "sample_size",
                        self.settings.max_order_statistic_sample_size)
                    distribution = run_with_timeout(
                        value.order_statistic,
                        self.settings.solver_timeout_seconds,
                        variables[0], sample_size,
                        int(parameters["order"]))
                    return finish_distribution(
                        distribution, [object_id],
                        "joint_marginal_order_statistic")
            if (
                capability.handler == "module:continuous_probability"
                and object_type == "ConditionalDistribution"
            ):
                if operation == "verify":
                    result = run_with_timeout(
                        value.verify, self.settings.solver_timeout_seconds)
                elif operation in {"pdf", "cdf"}:
                    point = None
                    if parameters.get("point") is not None:
                        _, parsed, _ = self._typed_parse_many(
                            [str(parameters["point"])], context_id)
                        point = parsed[0]
                    result = run_with_timeout(
                        getattr(value, operation),
                        self.settings.solver_timeout_seconds,
                        point)
                elif operation in {"mean", "variance"}:
                    result = run_with_timeout(
                        getattr(value, operation),
                        self.settings.solver_timeout_seconds)
                else:
                    raise ValueError(
                        "ConditionalDistribution supports verify, pdf, cdf, "
                        "mean, or variance")
                return finish(
                    result, f"probability:conditional_{operation}",
                    "continuous_probability")
            if (
                capability.handler == "module:continuous_probability"
                and object_type == "RandomVariable"
                and operation == "transform"
            ):
                from .continuous_probability import transform as transform_rv
                expression_id = str(parameters["expression_id"])
                ir, err = self._get_expr(expression_id, "continuous_probability")
                if err:
                    return err
                target_text = str(parameters.get("target", "Y"))
                branch_texts = [str(item) for item in parameters.get("inverse_branches", [])]
                jacobian_texts = [str(item) for item in parameters.get("jacobians", [])]
                if (
                    len(branch_texts) > self.settings.max_inverse_branches
                    or len(jacobian_texts) > self.settings.max_inverse_branches
                ):
                    raise ValueError(
                        "change of variables exceeds max_inverse_branches="
                        f"{self.settings.max_inverse_branches}")
                all_irs, parsed, parsed_trust = self._typed_parse_many(
                    [target_text, *branch_texts, *jacobian_texts], context_id)
                env = self._symbol_env(self.contexts.get(context_id) if context_id else None,
                                       ir, *all_irs)
                expression = self.sympy.to_sympy(ir, env)
                target = parsed[0]
                split = 1 + len(branch_texts)
                branches = parsed[1:split] or None
                jacobians = parsed[split:] or None
                transformed = run_with_timeout(
                    transform_rv, self.settings.solver_timeout_seconds,
                    value, expression, target,
                    inverse_branches=branches, jacobians=jacobians)
                trust = self._trust_min(
                    record["input_trust"], self._expr_trust(ir), parsed_trust)
                transformed.distribution.input_trust = trust.value
                new_id = self._store_math_object(
                    "RandomVariable", transformed, input_trust=trust,
                    sources=[object_id, expression_id])
                bundle = EvidenceBundle(computation=[
                    ComputationEvidence(
                        engine="continuous_probability",
                        method="change_of_variables_with_jacobian_check",
                        arithmetic="symbolic",
                        deterministic=True,
                        trust=self._trust_min(
                            TrustLevel.SYMBOLIC, trust).value,
                    )
                ])
                step = self._record(DerivationStep(
                    step_id=self._id("step"),
                    operation="probability:transform",
                    inputs=[object_id, expression_id],
                    output=new_id,
                    engine="continuous_probability",
                    trust=self._trust_min(TrustLevel.SYMBOLIC, trust),
                ))
                wrapped = MathResult(
                    ok=True,
                    data={
                        "object_id": new_id,
                        "object_type": "RandomVariable",
                        "object": transformed.model_dump(mode="json"),
                    },
                    trust=self._trust_min(TrustLevel.SYMBOLIC, trust),
                    engine="continuous_probability",
                    derivation=[step],
                    evidence_bundle=bundle.model_copy(deep=True),
                    claim_evidence={
                        "transformed_distribution": bundle.model_copy(
                            deep=True),
                    },
                )
                return wrapped
            if (
                capability.handler == "module:complex_analysis"
                and object_type in {"ComplexFunction", "Contour"}
            ):
                from .complex_analysis import (
                    ComplexAnalysisEngine, Singularity,
                )
                engine = ComplexAnalysisEngine()
                if object_type == "Contour":
                    if operation != "winding_number":
                        raise ValueError("Contour supports winding_number")
                    _, parsed, _ = self._typed_parse_many(
                        [str(parameters["point"])], context_id)
                    result = run_with_timeout(
                        engine.winding_number,
                        self.settings.solver_timeout_seconds,
                        value, parsed[0])
                elif operation == "derivative":
                    result = run_with_timeout(
                        engine.derivative,
                        self.settings.solver_timeout_seconds,
                        value)
                elif operation == "analyticity":
                    result = run_with_timeout(
                        engine.analyticity_candidate,
                        self.settings.solver_timeout_seconds,
                        value)
                elif operation == "zeros":
                    result = run_with_timeout(
                        engine.zeros,
                        self.settings.solver_timeout_seconds,
                        value)
                elif operation == "singularities":
                    result = run_with_timeout(
                        engine.singularities,
                        self.settings.solver_timeout_seconds,
                        value)
                elif operation == "argument_principle":
                    contour_id = str(parameters["contour_id"])
                    contour_record = self._get_math_object_record(contour_id)
                    if (
                        not contour_record
                        or contour_record["object_type"] != "Contour"
                    ):
                        raise ValueError(
                            f"Unknown Contour object: {contour_id}")
                    result = run_with_timeout(
                        engine.argument_principle,
                        self.settings.solver_timeout_seconds,
                        value, contour_record["value"])
                elif operation == "analytic_continuation":
                    domain_id = str(parameters["target_domain_id"])
                    domain_record = self._get_math_object_record(domain_id)
                    if (
                        not domain_record
                        or domain_record["object_type"] != "ComplexDomain"
                    ):
                        raise ValueError(
                            f"Unknown ComplexDomain object: {domain_id}")
                    result = run_with_timeout(
                        engine.analytic_continuation,
                        self.settings.solver_timeout_seconds,
                        value, domain_record["value"])
                elif operation == "conformal_map":
                    domain = None
                    if parameters.get("source_domain_id") is not None:
                        domain_id = str(parameters["source_domain_id"])
                        domain_record = self._get_math_object_record(domain_id)
                        if (
                            not domain_record
                            or domain_record["object_type"] != "ComplexDomain"
                        ):
                            raise ValueError(
                                f"Unknown ComplexDomain object: {domain_id}")
                        domain = domain_record["value"]
                    result = run_with_timeout(
                        engine.conformal_map,
                        self.settings.solver_timeout_seconds,
                        value, domain)
                else:
                    _, parsed, _ = self._typed_parse_many(
                        [str(parameters.get("point", "0"))], context_id)
                    point = parsed[0]
                    if operation == "classify_singularity":
                        max_order = self._bounded_positive_int(
                            parameters.get("max_order", 12), "max_order",
                            self.settings.max_symbolic_series_order)
                        result = run_with_timeout(
                            engine.classify_singularity,
                            self.settings.solver_timeout_seconds,
                            value, point, max_order)
                    elif operation == "residue":
                        result = run_with_timeout(
                            engine.residue,
                            self.settings.solver_timeout_seconds,
                            value, point)
                    elif operation == "laurent_series":
                        order = self._bounded_positive_int(
                            parameters.get("order", 6), "order",
                            self.settings.max_symbolic_series_order)
                        result = run_with_timeout(
                            engine.laurent_series,
                            self.settings.solver_timeout_seconds,
                            value, point, order)
                    elif operation == "conformal_at":
                        result = run_with_timeout(
                            engine.conformal_at,
                            self.settings.solver_timeout_seconds,
                            value, point)
                    elif operation == "contour_integral":
                        contour_id = str(parameters["contour_id"])
                        contour_record = self._get_math_object_record(contour_id)
                        if not contour_record or contour_record["object_type"] != "Contour":
                            raise ValueError(f"Unknown Contour object: {contour_id}")
                        singularities = []
                        for item in parameters.get("singularities", []):
                            _, points, _ = self._typed_parse_many(
                                [str(item["point"])], context_id)
                            singularities.append(Singularity(
                                point=points[0], kind=item.get("kind", "unknown"),
                                order=item.get("order")))
                        result = run_with_timeout(
                            engine.contour_integral,
                            self.settings.solver_timeout_seconds,
                            value, contour_record["value"], singularities,
                            singularities_accounted_for=bool(
                                parameters.get("singularities_accounted_for", False)))
                    else:
                        raise ValueError(
                            "ComplexFunction supports derivative, analyticity, "
                            "zeros, singularities, classify_singularity, residue, "
                            "laurent_series, conformal_at, contour_integral, "
                            "argument_principle, analytic_continuation, or "
                            "conformal_map")
                return finish(
                    result, f"complex:{operation}", "complex_analysis")
            raise ValueError(f"{object_type} does not support operation {operation}")
        except NotImplementedError as exc:
            return MathResult(
                ok=False, status="unknown",
                semantic_status=ResultStatus.UNSUPPORTED,
                errors=[str(exc)], engine="math_object")
        except (KeyError, TypeError, ValueError) as exc:
            return MathResult(
                ok=False, status="error", errors=[str(exc)], engine="math_object")

    def create_context(self, domains: dict[str,str] | None=None, assumptions: list[str] | None=None):
        cid = self._id("ctx")
        parsed = [Assumption(expression=parse_math(a), provenance=f"assertion:{i}") for i,a in enumerate(assumptions or [])]
        ctx = infer_context(MathContext(context_id=cid, domains=domains or {}, assumptions=parsed))
        self.contexts[cid] = ctx
        if self._store is not None:
            self._store.put_context(cid, ctx.model_dump(mode="json"))
        return ctx

    def infer_context(self, context_id: str) -> MathResult:
        ctx, err = self._get_context(context_id, "mathir")
        if err: return err
        ctx=infer_context(ctx)
        return MathResult(ok=True, data={"context":ctx.model_dump(mode="json")}, trust=TrustLevel.EXACT, engine="mathir")

    def check_context(self, context_id: str):
        ctx, err = self._get_context(context_id, "z3")
        if err: return err
        if not self.z3.available:
            return MathResult(ok=True, status="unknown", data={"consistency":"unknown"},
                warnings=["Z3 is unavailable; context consistency was not checked."], trust=TrustLevel.UNKNOWN)
        try:
            consistency = self.z3.check_context([a.expression for a in ctx.assumptions], ctx.domains)
        except (TypeError, ValueError) as exc:
            return MathResult(ok=True, status="unknown", data={"consistency":"unknown"},
                warnings=[f"Context is outside the current Z3 fragment: {exc}"], trust=TrustLevel.UNKNOWN)
        ctx.consistency = consistency
        status = "refuted" if consistency == "inconsistent" else ("verified" if consistency == "consistent" else "unknown")
        return MathResult(ok=True, status=status, data={"consistency":consistency,"context":ctx.model_dump(mode="json")}, trust=TrustLevel.EXACT, engine="z3")

    def _record(self, step: DerivationStep):
        step = finalize_step(self, step)
        with self._record_lock:
            self.derivations[step.step_id] = step
            if self._store is not None:
                self._store.put_derivation(step.step_id, step.model_dump(mode="json"))
        if step.output_expr_id and step.output_expr_id in self.expressions:
            persist_expression(self, step.output_expr_id, step)
        return step

    def _parents_for_exprs(self, expr_ids: list[str]) -> list[str]:
        return [self.expression_producers[eid] for eid in expr_ids if eid in self.expression_producers]

    @staticmethod
    def _expr_trust(ir: Expr) -> TrustLevel:
        """Trust implied by MathIR literals.

        Decimal/RealNode literals are approximate observations, even when a
        symbolic backend can manipulate them algebraically.  Exact integer and
        rational literals remain exact; symbolic variables/operators are
        symbolic unless an approximate real occurs anywhere in the tree.
        """
        kinds: set[str] = set()
        _scan_ir(ir, set(), set(), kinds)
        if "real" in kinds:
            return TrustLevel.NUMERIC
        exact_kinds = {"integer", "rational", "add", "mul", "pow", "div", "neg"}
        return TrustLevel.EXACT if kinds <= exact_kinds else TrustLevel.SYMBOLIC

    def _input_expr_trust(self, expr_ids: list[str]) -> TrustLevel:
        levels = [expression_trust(self, eid) for eid in expr_ids]
        return self._trust_min(*levels) if levels else TrustLevel.UNKNOWN

    def parse(self, expression: str):
        if len(expression) > self.settings.max_input_length:
            return MathResult(ok=False,status="error",
                errors=[f"Input too long (maximum {self.settings.max_input_length} characters)"],
                trust=TrustLevel.UNKNOWN,engine="parser")
        diagnostics=ambiguity_diagnostics(expression)
        fatal=[d for d in diagnostics if d.severity == "error"]
        if fatal:
            return MathResult(ok=False,status="error",data={"ambiguities":[d.model_dump() for d in diagnostics]},
                errors=["Ambiguous notation requires explicit parentheses or multiplication."],trust=TrustLevel.UNKNOWN,engine="parser")
        try:
            ir = parse_math(expression)
        except ValueError as exc:
            return MathResult(ok=False,status="error",data={"ambiguities":[d.model_dump() for d in diagnostics]},errors=[str(exc)],engine="parser")
        nodes=_count_nodes(ir)
        if nodes > self.settings.max_expression_nodes:
            return MathResult(ok=False,status="error",
                errors=[f"Expression has {nodes} nodes; limit is {self.settings.max_expression_nodes}."],
                trust=TrustLevel.UNKNOWN,engine="parser")
        eid = self._id("expr")
        self.expressions[eid] = ir
        self.expression_sources[eid] = expression
        if self._store is not None:
            self._store.put_expression(eid, {"ir": ir.model_dump(mode="json"), "source": expression})
        step=self._record(DerivationStep(step_id=self._id("step"),operation="parse",inputs=[expression],output_expr_id=eid,
            output=expression,engine="mathir",trust=self._expr_trust(ir)))
        self.expression_producers[eid]=step.step_id
        warnings=[f"Potential ambiguity: {d.reason} ({d.span})" for d in diagnostics]
        trust = self._expr_trust(ir)
        return MathResult(ok=True, data={"expr_id":eid,"ir":ir.model_dump(mode="json"),"source":expression,
            "ambiguities":[d.model_dump() for d in diagnostics]}, warnings=warnings, trust=trust, engine="mathir", derivation=[step])

    def analyze(self, expr_id: str, context_id: str | None=None):
        ir, err = self._get_expr(expr_id, "mathir")
        if err: return err
        ctx=self.contexts.get(context_id) if context_id else None
        caps=set(); symbols=set(); node_kinds=set()
        from .models import BinaryNode, CallNode, RelationNode, SymbolNode
        def walk(n):
            node_kinds.add(n.kind)
            if isinstance(n, SymbolNode): symbols.add(n.name)
            if isinstance(n, NaryNode):
                caps.add("Add" if n.kind=="add" else "Mul")
                for a in n.args: walk(a)
            elif isinstance(n, BinaryNode): caps.add("Pow" if n.kind=="pow" else "Inv"); walk(n.left); walk(n.right)
            elif isinstance(n, UnaryNode): caps.add("Neg"); walk(n.arg)
            elif isinstance(n, CallNode): caps.add(n.name.capitalize()); [walk(a) for a in n.args]
            elif isinstance(n, RelationNode): caps.add("Relation"); walk(n.left); walk(n.right)
        walk(ir)
        semantic={s:{"domain":(ctx.domains.get(s) if ctx else None),"properties":(ctx.symbol_properties.get(s,[]) if ctx else [])} for s in sorted(symbols)}
        return MathResult(ok=True,data={"symbols":sorted(symbols),"semantic_symbols":semantic,
            "node_kinds":sorted(node_kinds),"required_capabilities":sorted(caps),"engine_manifest":self.router.manifest()},trust=TrustLevel.EXACT,engine="mathir")

    def plan(self, expr_id: str, context_id: str | None = None, solve_for: str | None = None) -> MathResult:
        ir, err = self._get_expr(expr_id, "reasoning_planner")
        if err: return err
        ctx = self.contexts.get(context_id) if context_id else None
        plan = plan_problem(
            ir,
            ctx,
            self.router.manifest(),
            solve_for=solve_for,
            capability_manifest=self.router.registry.manifest(),
        )
        plan_id = self._id("plan")
        self.plans[plan_id] = plan
        self.plan_sources[plan_id] = (expr_id, context_id)
        self._persist_plan(plan_id)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="plan_problem",
            inputs=[expr_id], parents=self._parents_for_exprs([expr_id]), output=plan.classification,
            engine="reasoning_planner", trust=TrustLevel.HEURISTIC))
        return MathResult(ok=True, data={"plan_id": plan_id, "plan": plan.model_dump(mode="json")},
                          warnings=(["Planner classifications guide routing; they are not proofs."] if plan.missing_assumptions else []),
                          trust=TrustLevel.HEURISTIC, engine="reasoning_planner", derivation=[step])

    def _persist_plan(self, plan_id: str) -> None:
        if self._store is None:
            return
        expr_id, context_id = self.plan_sources[plan_id]
        self._store.put_plan(plan_id, {
            "plan": self.plans[plan_id].model_dump(mode="json"),
            "expr_id": expr_id,
            "context_id": context_id,
            "typed_request": self.typed_plan_requests.get(plan_id),
        })

    def _load_plan(self, plan_id: str) -> bool:
        if plan_id in self.plans:
            return True
        if self._store is None:
            return False
        payload = self._store.get_plan(plan_id)
        if payload is None:
            return False
        self.plans[plan_id] = ProblemPlan.model_validate(payload["plan"])
        self.plan_sources[plan_id] = (
            payload["expr_id"], payload.get("context_id"))
        if payload.get("typed_request") is not None:
            self.typed_plan_requests[plan_id] = payload["typed_request"]
        return True

    def plan_get(self, plan_id: str) -> MathResult:
        if not self._load_plan(plan_id):
            return MathResult(ok=False, status="error", errors=[f"Unknown plan_id: {plan_id}"], engine="reasoning_planner")
        expr_id, context_id = self.plan_sources[plan_id]
        return MathResult(ok=True, data={"plan_id": plan_id, "expr_id": expr_id, "context_id": context_id,
            "plan": self.plans[plan_id].model_dump(mode="json")}, trust=TrustLevel.HEURISTIC, engine="reasoning_planner")

    def execute_plan(self, plan_id: str, formal: bool = True, max_steps: int = 32) -> MathResult:
        if not self._load_plan(plan_id):
            return MathResult(ok=False, status="error", errors=[f"Unknown plan_id: {plan_id}"], engine="obligation_executor")
        if max_steps < 1 or max_steps > self.settings.max_obligation_steps:
            return MathResult(
                ok=False, status="error",
                errors=[
                    f"max_steps must be in 1.."
                    f"{self.settings.max_obligation_steps}"],
                engine="obligation_executor")
        typed_request = self.typed_plan_requests.get(plan_id)
        if typed_request is not None:
            if max_steps < len(self.plans[plan_id].obligations):
                return MathResult(
                    ok=False, status="error",
                    errors=["max_steps is smaller than the typed obligation DAG"],
                    engine="obligation_executor")
            return self.apply(**typed_request)
        expr_id, context_id = self.plan_sources[plan_id]
        try:
            execution = self.executor.execute(plan_id, self.plans[plan_id], expr_id, context_id, formal=formal, max_steps=max_steps)
        except (ValueError, RuntimeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="obligation_executor")
        self.executions[execution.execution_id] = execution
        if self._store is not None:
            self._store.put_execution(
                execution.execution_id,
                execution.model_dump(mode="json"))
        final_steps=[self.derivations[n.derivation_step_id] for n in execution.obligations if n.derivation_step_id in self.derivations]
        evidence=[e for n in execution.obligations for e in n.evidence]
        warnings=[]
        if any(x.startswith("Domain for ") for x in execution.propagated_side_conditions):
            warnings.append("Execution retained unresolved domain assumptions; inspect propagated_side_conditions.")
        if execution.conflicts:
            warnings.append("Independent obligations disagreed; inspect conflicts before using the result.")
        return MathResult(ok=execution.state != "failed", status=execution.final_status,
            data={"execution": execution.model_dump(mode="json")}, assumptions_used=execution.propagated_assumptions,
            side_conditions=execution.propagated_side_conditions, warnings=warnings, trust=execution.final_trust,
            engine="obligation_executor", derivation=final_steps, evidence=evidence,
            evidence_bundle=execution.evidence_bundle,
            claim_evidence=execution.claim_evidence)

    def reason(self, expr_id: str, context_id: str | None = None, formal: bool = True,
               max_steps: int = 32, solve_for: str | None = None) -> MathResult:
        planned=self.plan(expr_id, context_id, solve_for=solve_for)
        if not planned.ok:
            return planned
        return self.execute_plan(planned.data["plan_id"], formal=formal, max_steps=max_steps)

    def execution_get(self, execution_id: str) -> MathResult:
        if execution_id not in self.executions and self._store is not None:
            payload = self._store.get_execution(execution_id)
            if payload is not None:
                self.executions[execution_id] = PlanExecution.model_validate(
                    payload)
        if execution_id not in self.executions:
            return MathResult(ok=False, status="error", errors=[f"Unknown execution_id: {execution_id}"], engine="obligation_executor")
        ex=self.executions[execution_id]
        return MathResult(ok=True, status=ex.final_status, data={"execution":ex.model_dump(mode="json")},
                          assumptions_used=ex.propagated_assumptions, side_conditions=ex.propagated_side_conditions,
                          trust=ex.final_trust, engine="obligation_executor",
                          evidence_bundle=ex.evidence_bundle,
                          claim_evidence=ex.claim_evidence)

    def integer_analyze(self, value: str, factor_limit: int = 100_000) -> MathResult:
        try:
            data = self.integer.analyze(value, factor_limit=factor_limit)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="integer_exact")
        return MathResult(ok=True, data=data, trust=TrustLevel.EXACT, engine="integer_exact")

    def integer_compute(self, operation: str, values: list[str], modulus: str | None = None,
                        moduli: list[str] | None = None, max_output_digits: int = 100_000,
                        factor_limit: int = 100_000) -> MathResult:
        if max_output_digits < 1 or max_output_digits > 1_000_000:
            return MathResult(ok=False, status="error", errors=["max_output_digits must be between 1 and 1,000,000"], engine="integer_exact")
        try:
            data = self.integer.compute(operation, values, modulus=modulus, moduli=moduli,
                                        max_output_digits=max_output_digits, factor_limit=factor_limit)
        except (ValueError, OverflowError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="integer_exact")
        step = self._record(DerivationStep(step_id=self._id("step"), operation=f"integer:{operation}",
            inputs=values, output=str(data.get("result", data)), engine="integer_exact", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data=data, trust=TrustLevel.EXACT, engine="integer_exact", derivation=[step])

    def integer_batch(self, jobs: list[dict], workers: int | None = None) -> MathResult:
        if not jobs:
            return MathResult(ok=False, status="error", errors=["jobs must not be empty"], engine="integer_exact")
        if len(jobs) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                errors=[f"Batch has {len(jobs)} jobs; limit is {self.settings.max_batch_jobs}."],
                engine="integer_exact")
        for i, job in enumerate(jobs):
            if not isinstance(job, dict) or "operation" not in job:
                return MathResult(ok=False, status="error",
                    errors=[f"Job {i} must be an object with an 'operation' field."], engine="integer_exact")
        resolved = resolve_workers(workers, cap=self.settings.max_workers) \
            if self.settings.enable_parallel else 1
        results = self.integer.batch(jobs, workers=resolved)
        failed = sum(1 for r in results if not r.get("ok"))
        step = self._record(DerivationStep(step_id=self._id("step"), operation="integer:batch",
            inputs=[f"{len(jobs)} jobs"], output=f"{len(results) - failed} ok, {failed} failed",
            engine="integer_exact", trust=TrustLevel.EXACT))
        return MathResult(ok=failed < len(results),
            data={"results": results, "job_count": len(jobs), "failed": failed, "workers": resolved},
            trust=TrustLevel.EXACT, engine="integer_exact", derivation=[step])

    def collatz_sieve(self, n_max: int, x_min: str = "1", workers: int | None = None,
                      n_min: int = 1, canonical: bool = True,
                      engine: str = "auto") -> MathResult:
        try:
            xm = decimal_to_int(x_min)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[f"Invalid x_min: {exc}"],
                              engine="collatz_sieve")
        resolved = resolve_workers(workers, cap=self.settings.max_workers) \
            if self.settings.enable_parallel else 1
        try:
            report = sieve_cycle_classes(n_max, x_min=xm, workers=resolved, n_min=n_min,
                                         canonical=canonical, engine=engine)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="collatz_sieve")
        side = []
        if xm > 1:
            side.append(
                f"Refutation covers only cycles whose minimum element is >= {x_min}; "
                "this relies on an external verification floor, not on this kernel.")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="collatz_sieve",
            inputs=[f"n_max={n_max}", f"x_min={x_min}", f"n_min={n_min}"],
            output=f"{report['total_patterns_checked']} patterns, "
                   f"{len(report['nontrivial_cycles'])} non-trivial cycles",
            engine="collatz_sieve", trust=TrustLevel.EXACT))
        found = report["nontrivial_cycles"]
        return MathResult(ok=True, status="refuted" if found else "verified", data=report,
            side_conditions=side,
            warnings=(["Non-trivial Collatz cycle candidate(s) found; inspect nontrivial_cycles."] if found
                      else []),
            trust=TrustLevel.EXACT, engine="collatz_sieve", derivation=[step])

    def derivation_get(self, step_id: str) -> MathResult:
        step = self.derivations.get(step_id)
        if step is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown step_id: {step_id}"], engine="derivation_dag")
        return MathResult(ok=True, data={"step":step.model_dump(mode="json")}, trust=step.trust, engine=step.engine)

    def derivation_trace(self, step_id: str) -> MathResult:
        if step_id not in self.derivations:
            return MathResult(ok=False, status="error", errors=[f"Unknown step_id: {step_id}"], engine="derivation_dag")
        graph=trace_dag(step_id,self.derivations)
        root = self.derivations[step_id]
        return MathResult(
            ok=True,
            data={"graph": graph},
            trust=root.trust,
            engine="derivation_dag",
            derivation=[root.model_copy(deep=True)],
            evidence=[item.model_copy(deep=True) for item in root.evidence],
            evidence_bundle=root.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in root.claim_evidence.items()
            },
        )

    def fuzz_differential(self, n: int = 100, variables: list[str] | None = None,
                          depth: int = 3, samples: int = 8, seed: int = 0,
                          workers: int | None = None) -> MathResult:
        """Cross-engine differential fuzzing: random MathIR over the numeric
        fragment, mpmath high-precision vs float64 compiled path. Disagreements
        are surfaced, never absorbed. Parallel over the process pool."""
        from .fuzz import fuzz_batch
        if n < 1 or n > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                              errors=[f"n must be 1..{self.settings.max_batch_jobs}"],
                              engine="fuzz")
        out = fuzz_batch(n, variables, depth, samples, seed, workers)
        status = "ok" if not out["disagreements"] else "conflict"
        return MathResult(ok=True, status=status, trust=TrustLevel.NUMERIC,
                          data=out, engine="fuzz")

    def certified_enclose(self, expr_id: str, variable: str, lo: str, hi: str) -> MathResult:
        """Interval enclosure of a univariate expression over [lo, hi] using
        Arb ball arithmetic when python-flint is installed (certified extra),
        else mpmath.iv. interval_certified trust."""
        from .certified import arb_available, interval_enclosure
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="certified")
        try:
            lo_s, hi_s = self._numeric_bound(lo), self._numeric_bound(hi)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="certified")
        import sympy as sp
        sym = self.sympy.to_sympy(ir, {variable: sp.Symbol(variable)})
        if arb_available():
            try:
                f = sp.lambdify(sp.Symbol(variable), sym, modules="flint")
            except (KeyError, NameError, TypeError):
                f = None
            if f is not None:
                out = interval_enclosure(f, lo_s, hi_s)
                if out is not None:
                    return MathResult(ok=True, status="ok", trust=TrustLevel.INTERVAL_CERTIFIED,
                                      data=out, engine="arb")
        import mpmath as mp
        f = sp.lambdify(sp.Symbol(variable), sym, modules="mpmath")
        enclosure = f(mp.iv.mpf([mp.mpf(lo_s), mp.mpf(hi_s)]))
        return MathResult(ok=True, status="ok", trust=TrustLevel.INTERVAL_CERTIFIED,
                          data={"engine": "mpmath.iv", "enclosure": mp.nstr(enclosure, 30)},
                          engine="mpmath.iv")

    def prove(self, expr_id: str, context_id: str | None = None,
              formal: bool = True) -> MathResult:
        """General theorem proving: fragment classification -> SMT portfolio
        (raced encodings) -> optional Lean certificate. FORMAL only with a
        checked certificate; EXACT for decisive SMT; UNKNOWN otherwise."""
        from .prove import classify_fragment, prove_smt
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="prove")
        ctx = self.contexts.get(context_id) if context_id else None
        domains = ctx.domains if ctx else {}
        assumptions = [a.expression for a in ctx.assumptions] if ctx else []
        fragment = classify_fragment(ir, domains)
        data: dict = {"fragment": fragment}
        trust = TrustLevel.UNKNOWN
        status = "unknown"

        smt = prove_smt(self.z3, ir, domains, fragment, assumptions,
                        portfolio_size=self.settings.prove_portfolio_size)
        data["smt"] = {k: v for k, v in smt.items() if k != "attempts"}
        data["smt"]["encodings"] = [a["encoding"] for a in smt.get("attempts", [])]
        if smt["status"] == "valid":
            trust, status = TrustLevel.EXACT, "verified"
        elif smt["status"] == "not_valid":
            trust, status = TrustLevel.EXACT, "refuted"
        elif smt["status"] == "unavailable":
            data["smt_unavailable"] = True

        # Lean certificate tier: only for relation goals in the supported
        # fragment, and only worth attempting when SMT says valid.
        if formal and status == "verified" and isinstance(ir, RelationNode) \
                and fragment["logic"] == "qf" and fragment["arithmetic"] != "outside" \
                and not fragment["has_functions"]:
            integer = fragment["sort"] == "int"
            try:
                lstatus, script, tactic, error = run_with_timeout(
                    self.lean.prove_relation, self.settings.solver_timeout_seconds,
                    ir, assumptions, integer)
            except (ValueError, TypeError) as exc:
                lstatus, script, tactic, error = "unsupported", None, None, str(exc)
            data["lean"] = {"status": lstatus, "tactic": tactic}
            if lstatus == "proved":
                trust = TrustLevel.FORMAL
                if self._store is not None:
                    cert_id = self._id("cert")
                    self._store.put_certificate(cert_id, "lean",
                                                {"script": script, "tactic": tactic,
                                                 "expr_id": expr_id})
                    data["certificate_id"] = cert_id
                else:
                    data["certificate"] = script
            elif error:
                data["lean"]["error"] = error[:500]

        step = self._record(DerivationStep(step_id=self._id("step"), operation="prove",
                                           inputs=[expr_id], output=status,
                                           engine="prove", trust=trust))
        return MathResult(ok=True, status=status, trust=trust, data=data,
                          derivation=[step], engine="prove")

    def prove_batch(self, expr_ids: list[str], context_id: str | None = None,
                    workers: int | None = None) -> MathResult:
        """Prove many statements; parallel over the process pool. Workers
        re-parse from source text so no engine state crosses processes."""
        from .parallel import process_map, resolve_workers
        from .prove import _prove_job
        if len(expr_ids) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                              errors=[f"batch exceeds max_batch_jobs={self.settings.max_batch_jobs}"],
                              engine="prove")
        jobs = []
        for eid in expr_ids:
            src = self.expression_sources.get(eid)
            if src is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"Unknown expr_id: {eid}"], engine="prove")
            ctx = self.contexts.get(context_id) if context_id else None
            jobs.append({"expr_id": eid, "source": src,
                         "domains": ctx.domains if ctx else {}})
        results = process_map(_prove_job, jobs, workers=resolve_workers(
            workers, cap=self.settings.max_workers))
        n_valid = sum(1 for r in results if r.get("status") == "valid")
        # weakest-evidence-wins: the batch is EXACT only when every child
        # proof was decisive; any unknown/unavailable child caps the batch
        decisive = all(r.get("status") in ("valid", "not_valid") for r in results)
        trust = TrustLevel.EXACT if results and decisive else TrustLevel.UNKNOWN
        return MathResult(ok=True, status="ok", trust=trust,
                          data={"total": len(results), "valid": n_valid,
                                "decisive": sum(1 for r in results
                                                if r.get("status") in ("valid", "not_valid")),
                                "results": results},
                          engine="prove")

    def prove_replay(self, cert_id: str) -> MathResult:
        """Re-check a stored Lean certificate (requires MATHKERNEL_STORE_PATH)."""
        if self._store is None:
            return MathResult(ok=False, status="error", engine="prove",
                              errors=["persistence is disabled; set MATHKERNEL_STORE_PATH"])
        cert = self._store.get_certificate(cert_id)
        if cert is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown cert_id: {cert_id}"], engine="prove")
        out = self.lean.replay_certificate(cert["script"],
                                           timeout=self.settings.lean_timeout_seconds)
        if out["status"] == "proved":
            trust, status = TrustLevel.FORMAL, "verified"
        elif out["status"] == "error":
            trust, status = TrustLevel.UNKNOWN, "error"
        else:
            trust, status = TrustLevel.UNKNOWN, "unknown"
        return MathResult(ok=True, status=status, trust=trust,
                          data={"cert_id": cert_id, "tactic": cert.get("tactic"),
                                "replay": out}, engine="lean")

    def _install_settings(self, settings: Settings) -> list[str]:
        """Replace live settings and refresh engines/store that cache them."""
        notes: list[str] = []
        previous = self.settings
        self.settings = settings
        self.sympy.timeout_seconds = settings.solver_timeout_seconds
        self.z3.timeout_ms = settings.z3_timeout_ms
        self.lean.executable = settings.lean_binary
        self.lean.timeout = settings.lean_timeout_seconds
        if previous.store_path != settings.store_path:
            if self._store is not None:
                self._store.close()
                self._store = None
            if settings.store_path:
                from .store import KernelStore
                self._store = KernelStore(settings.store_path)
            notes.append(
                "store_path changed; the previous SQLite handle was closed "
                "and a new one opened only if a path is set")
        return notes

    def yolo_settings(self, updates: dict | None = None) -> MathResult:
        """Inspect or mutate MATHKERNEL_* settings. Requires YOLO mode."""
        if not yolo_mode():
            return MathResult(
                ok=False, status="error", engine="settings",
                errors=["YOLO mode is disabled; start the process with "
                        "MATHKERNEL_YOLO_MODE=true to change MATHKERNEL_* "
                        "settings at runtime"])
        schema = setting_schema()
        current = {
            "yolo_mode": True,
            "settings": self.settings.limits(),
            "schema": schema,
        }
        if updates is None:
            return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                              engine="settings", data=current)
        try:
            new_settings, applied, notes = apply_setting_updates(
                self.settings, updates)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", engine="settings",
                              errors=[str(exc)])
        notes.extend(self._install_settings(new_settings))
        return MathResult(
            ok=True, status="ok", trust=TrustLevel.EXACT, engine="settings",
            data={"applied": applied, "settings": self.settings.limits(),
                  "yolo_mode": yolo_mode(), "notes": notes, "schema": schema})

    def store_status(self) -> MathResult:
        """Report whether SQLite persistence is active (MATHKERNEL_STORE_PATH)."""
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT, engine="store",
                          data={"enabled": self._store is not None,
                                "path": self.settings.store_path})

    def replay(self, step_id: str) -> MathResult:
        """Deterministic replay of the derivation DAG from the persistent store:
        reconstructs the provenance chain in topological order and validates
        DAG integrity. Requires MATHKERNEL_STORE_PATH."""
        if self._store is None:
            return MathResult(ok=False, status="error", engine="store",
                              errors=["persistence is disabled; set MATHKERNEL_STORE_PATH"])
        try:
            out = self._store.replay(step_id)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="store")
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                          data=out, engine="store")

    def _symbol_env(self, ctx: MathContext | None, *irs: Expr) -> dict:
        return symbol_env(sorted(_symbol_names(*irs)),
                          ctx.domains if ctx else None,
                          ctx.symbol_properties if ctx else None)

    def _parse_bound(self, text: str, env: dict):
        key = text.strip().lower()
        if key in _INFINITY_ALIASES:
            return sp.oo
        if key in _NEG_INFINITY_ALIASES:
            return -sp.oo
        return self.sympy.to_sympy(parse_math(text), env)

    def _get_expr(self, expr_id: str, engine: str) -> tuple[Expr | None, MathResult | None]:
        ir = self.expressions.get(expr_id)
        if ir is None and self._store is not None:
            payload = self._store.get_expression(expr_id)
            if payload is not None:
                if not isinstance(payload, dict) or "ir" not in payload:
                    return None, MathResult(ok=False, status="error", engine=engine,
                        errors=["Legacy expression has no stored MathIR; reparse the original source explicitly"])
                ir = TypeAdapter(Expr).validate_python(payload["ir"])
                self.expressions[expr_id] = ir
                self.expression_sources[expr_id] = payload.get("source", "")
                provenance = payload.get("provenance")
                if provenance is None:
                    producer = self._store.find_expression_producer(expr_id)
                    provenance = {"version": 0,
                        "trust": "unknown", "legacy_unverified": True,
                        "producer_step_id": producer.get("step_id") if producer else None,
                        "assumptions": producer.get("conditions", []) if producer else [],
                        "context_ids": [], "input_ids": producer.get("inputs", []) if producer else []}
                self.expression_provenance[expr_id] = provenance
                sid = provenance.get("producer_step_id")
                if sid:
                    for raw in self._store.replay(sid)["steps"]:
                        step = DerivationStep.model_validate(raw)
                        self.derivations[step.step_id] = step
                    self.expression_producers[expr_id] = sid
        if ir is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown expr_id: {expr_id}"], engine=engine)
        return ir, None

    def _get_context(self, context_id: str, engine: str) -> tuple[MathContext | None, MathResult | None]:
        ctx = self.contexts.get(context_id)
        if ctx is None and self._store is not None:
            payload = self._store.get_context(context_id)
            if payload is not None:
                ctx = MathContext.model_validate(payload)
                self.contexts[context_id] = ctx
        if ctx is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown context_id: {context_id}"], engine=engine)
        return ctx, None

    def _finish_symbolic(self, out, operation: str, inputs: list[str], parents: list[str],
                         side_conditions: list[str] | None = None,
                         warnings: list[str] | None = None,
                         engine: str = "sympy",
                         trust: TrustLevel = TrustLevel.SYMBOLIC) -> MathResult:
        warnings = list(warnings or [])
        out_eid = None
        try:
            out_ir = self.sympy.from_sympy(out)
            out_eid = self._id("expr")
            self.expressions[out_eid] = out_ir
            self.expression_sources[out_eid] = str(out)
        except ValueError as exc:
            warnings.append(f"Result is outside the representable MathIR fragment; display string only: {exc}")
        display = render_expr(out_ir) if out_eid else str(out)
        step = self._record(DerivationStep(step_id=self._id("step"), operation=operation, inputs=inputs,
            parents=parents, output=display, output_expr_id=out_eid, engine=engine,
            trust=trust, conditions=side_conditions or []))
        if out_eid:
            self.expression_producers[out_eid] = step.step_id
        return MathResult(ok=True, data={"result": display, "result_expr_id": out_eid},
                          side_conditions=side_conditions or [], warnings=warnings,
                          trust=trust, engine=engine, derivation=[step])

    def simplify(self, expr_id: str, mode: str="simplify", context_id: str | None = None):
        ir, err = self._get_expr(expr_id, "sympy")
        if err: return err
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        capability = mode if mode in {"expand","factor","cancel","trig","rational","normal_form"} else "simplify"
        engine = self.router.choose(capability, preferred=("sympy",))
        try:
            out=engine.simplify(ir,mode,env)  # type: ignore[attr-defined]
        except ValueError as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine=engine.name)
        trust = self._trust_min(TrustLevel.SYMBOLIC, expression_trust(self, expr_id))
        return self._finish_symbolic(out, mode, [expr_id], self._parents_for_exprs([expr_id]),
                                     engine=engine.name, trust=trust)

    def solve(self, expr_id: str, variable: str, context_id: str|None=None):
        ir, err = self._get_expr(expr_id, "sympy")
        if err: return err
        ctx = self.contexts.get(context_id) if context_id else None
        domain = (ctx.domains.get(variable, "complex") if ctx else "complex").lower()
        env = self._symbol_env(ctx, ir)
        try:
            from .assumption_constraints import constrained_solve
            sol = constrained_solve(self.sympy, ir, variable, domain, ctx, env)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        structured=serialize_solution_set(sol,domain)
        assumptions=[render_expr(a.expression) for a in ctx.assumptions] if ctx else []
        trust = self._trust_min(TrustLevel.SYMBOLIC, expression_trust(self, expr_id))
        step=self._record(DerivationStep(step_id=self._id("step"),operation="solve",inputs=[expr_id],parents=self._parents_for_exprs([expr_id]),
            output=str(sol),engine="sympy",trust=trust,conditions=assumptions))
        return MathResult(ok=True,data={"variable":variable,"domain":domain,"solution_set":structured.model_dump(mode="json"),"display":str(sol)},
            assumptions_used=assumptions,trust=trust,engine="sympy",derivation=[step])

    def solve_system(self, expr_ids: list[str], variables: list[str], context_id: str|None=None):
        if not expr_ids:
            return MathResult(ok=False, status="error", errors=["expr_ids must not be empty"], engine="sympy")
        if not variables:
            return MathResult(ok=False, status="error", errors=["variables must not be empty"], engine="sympy")
        missing = [e for e in expr_ids if e not in self.expressions]
        if missing:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id(s): {missing}"], engine="sympy")
        irs = [self.expressions[e] for e in expr_ids]
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, *irs)
        env.update({name: sp.Symbol(name) for name in variables})
        try:
            solutions = self.sympy.solve_system(irs, variables, env)
            from .assumption_constraints import constrain_system_solutions
            solutions, candidate_conditions = constrain_system_solutions(self.sympy, solutions, variables, ctx, env)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        out_solutions = []
        for sol, conditions in zip(solutions, candidate_conditions):
            entry = {}
            for var in variables:
                value = sol.get((env or {}).get(var, sp.Symbol(var)))
                if value is None:
                    continue
                entry[var] = {"display": str(value)}
                try:
                    value_ir = self.sympy.from_sympy(value)
                    vid = self._id("expr")
                    self.expressions[vid] = value_ir
                    self.expression_sources[vid] = str(value)
                    entry[var]["expr_id"] = vid
                except ValueError:
                    pass
            if conditions:
                entry["conditions"] = conditions
            out_solutions.append(entry)
        warnings = []
        if not out_solutions:
            warnings.append("No explicit solutions returned; the system may be inconsistent "
                            "or outside the supported fragment.")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="solve_system",
            inputs=list(expr_ids), parents=self._parents_for_exprs(list(expr_ids)),
            output=f"{len(out_solutions)} solution(s) for {variables}", engine="sympy",
            trust=TrustLevel.SYMBOLIC))
        return MathResult(ok=True, data={"variables": variables, "solutions": out_solutions,
            "solution_count": len(out_solutions)}, warnings=warnings,
            trust=TrustLevel.SYMBOLIC, engine="sympy", derivation=[step])

    def differentiate(self, expr_id: str, variable: str, order: int = 1,
                      context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="sympy")
        if order < 1 or order > 64:
            return MathResult(ok=False, status="error", errors=["order must be between 1 and 64"], engine="sympy")
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        try:
            out = self.sympy.differentiate(ir, variable, order, env)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        trust = self._trust_min(TrustLevel.SYMBOLIC, expression_trust(self, expr_id))
        return self._finish_symbolic(out, "differentiate", [expr_id, f"d^{order}/d{variable}^{order}"],
                                     self._parents_for_exprs([expr_id]), trust=trust)

    def integrate(self, expr_id: str, variable: str, lower: str | None = None,
                  upper: str | None = None, context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="sympy")
        if (lower is None) != (upper is None):
            return MathResult(ok=False, status="error",
                errors=["Definite integration requires both lower and upper bounds."], engine="sympy")
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        try:
            lo = self._parse_bound(lower, env) if lower is not None else None
            hi = self._parse_bound(upper, env) if upper is not None else None
            out = self.sympy.integrate(ir, variable, lo, hi, env)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        label = f"integral d{variable}" if lo is None else f"integral [{lower}, {upper}] d{variable}"
        side = [] if lo is not None else ["Indefinite integral is determined up to an additive constant."]
        trust = self._trust_min(TrustLevel.SYMBOLIC, expression_trust(self, expr_id))
        return self._finish_symbolic(out, "integrate", [expr_id, label],
                                     self._parents_for_exprs([expr_id]), side_conditions=side, trust=trust)

    def limit(self, expr_id: str, variable: str, point: str, direction: str = "+-",
              context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="sympy")
        if direction not in {"+-", "+", "-"}:
            return MathResult(ok=False, status="error",
                errors=["direction must be '+-' (two-sided), '+' (from above), or '-' (from below)"], engine="sympy")
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        try:
            pt = self._parse_bound(point, env)
            out = self.sympy.limit(ir, variable, pt, direction, env)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        return self._finish_symbolic(out, "limit", [expr_id, f"{variable} -> {point} ({direction})"],
                                     self._parents_for_exprs([expr_id]))

    def series(self, expr_id: str, variable: str, point: str = "0", order: int = 6,
               context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="sympy")
        if order < 1 or order > 128:
            return MathResult(ok=False, status="error", errors=["order must be between 1 and 128"], engine="sympy")
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        try:
            pt = self._parse_bound(point, env)
            out, had_o_term = self.sympy.series(ir, variable, pt, order, env)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        side = []
        if had_o_term:
            side.append(f"Series truncated at order {order}; the O(({variable} - ({point}))^{order}) term was dropped.")
        return self._finish_symbolic(out, "series", [expr_id, f"{variable} at {point}, order {order}"],
                                     self._parents_for_exprs([expr_id]), side_conditions=side)

    def summation(self, expr_id: str, variable: str, lower: str, upper: str,
                  context_id: str | None = None) -> MathResult:
        return self._sum_or_product("summation", expr_id, variable, lower, upper, context_id)

    def product(self, expr_id: str, variable: str, lower: str, upper: str,
                context_id: str | None = None) -> MathResult:
        return self._sum_or_product("product", expr_id, variable, lower, upper, context_id)

    def _sum_or_product(self, kind: str, expr_id: str, variable: str, lower: str, upper: str,
                        context_id: str | None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="sympy")
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, ir)
        try:
            lo = self._parse_bound(lower, env)
            hi = self._parse_bound(upper, env)
            fn = self.sympy.summation if kind == "summation" else self.sympy.product
            out = fn(ir, variable, lo, hi, env)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy")
        return self._finish_symbolic(out, kind, [expr_id, f"{variable} = {lower}..{upper}"],
                                     self._parents_for_exprs([expr_id]))

    def interval_evaluate(self, expr_id: str, bounds: dict[str, list[float|str]], dps: int=50):
        ir, err = self._get_expr(expr_id, "mpmath_interval")
        if err: return err
        normalized={k:(v[0],v[1]) for k,v in bounds.items() if len(v)==2}
        if len(normalized)!=len(bounds):
            return MathResult(ok=False,status="error",errors=["Every bound must contain exactly [lower, upper]."],engine="mpmath_interval")
        try:
            data=self.interval.evaluate(ir,normalized,dps)
        except (ValueError,ZeroDivisionError) as exc:
            return MathResult(ok=True,status="unknown",warnings=[str(exc)],trust=TrustLevel.UNKNOWN,engine="mpmath_interval")
        step=self._record(DerivationStep(step_id=self._id("step"),operation="interval_evaluate",inputs=[expr_id],parents=self._parents_for_exprs([expr_id]),
            output=data["enclosure"],engine="mpmath_interval",trust=TrustLevel.INTERVAL_CERTIFIED,
            conditions=[f"{k} in [{v[0]}, {v[1]}]" for k,v in normalized.items()]))
        return MathResult(ok=True,data=data,trust=TrustLevel.INTERVAL_CERTIFIED,engine="mpmath_interval",derivation=[step])

    def numeric_evaluate(self, expr_id: str, values: dict[str, str] | None = None, dps: int = 50) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="numeric")
        if dps < 2 or dps > 500:
            return MathResult(ok=False, status="error", errors=["dps must be between 2 and 500"], engine="numeric")
        env = self._symbol_env(None, ir)
        try:
            expr = self.sympy.to_sympy(ir, env)
            if values:
                subs = {env.get(k, sp.Symbol(k)): self._parse_bound(v, env) for k, v in values.items()}
                expr = expr.subs(subs, simultaneous=True)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="numeric")
        free = sorted(str(s) for s in expr.free_symbols)
        if free:
            return MathResult(ok=False, status="error",
                errors=[f"Unbound symbols {free}; provide 'values' for full numeric evaluation."], engine="numeric")
        value = sp.N(expr, dps)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="numeric_evaluate",
            inputs=[expr_id, *[f"{k}={v}" for k, v in (values or {}).items()]],
            parents=self._parents_for_exprs([expr_id]), output=str(value), engine="numeric",
            trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data={"value": str(value), "dps": dps,
            "certified": False, "note": "Use interval_evaluate for a certified enclosure."},
            trust=TrustLevel.NUMERIC, engine="numeric", derivation=[step])

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    @staticmethod
    def _matrix_trust(cells: list[list[Expr]]) -> TrustLevel:
        kinds: set[str] = set()
        for row in cells:
            for cell in row:
                _scan_ir(cell, set(), set(), kinds)
        if "real" in kinds:
            return TrustLevel.NUMERIC
        if kinds <= {"integer", "rational", "add", "mul", "pow", "div", "neg"}:
            return TrustLevel.EXACT
        return TrustLevel.SYMBOLIC

    @staticmethod
    def _trust_min(*levels: TrustLevel) -> TrustLevel:
        """Weakest link: a derived result can never be trusted more than its
        least-trusted evidentiary input."""
        return min(levels, key=lambda level: _TRUST_RANK[level])

    @staticmethod
    def _matrix_display(cells: list[list[Expr]]) -> str:
        return "[" + ", ".join("[" + ", ".join(render_expr(c) for c in row) + "]" for row in cells) + "]"

    def _to_sp_matrix(self, matrix_id: str, ctx: MathContext | None = None):
        cells = self.matrices[matrix_id]
        env = self._symbol_env(ctx, *[c for row in cells for c in row])
        return sp.Matrix([[self.sympy.to_sympy(c, env) for c in row] for row in cells])

    def _store_matrix(self, cells: list[list[Expr]], operation: str, inputs: list[str],
                      parents: list[str], trust: TrustLevel) -> MathResult:
        mid = self._id("mat")
        self.matrices[mid] = cells
        display = self._matrix_display(cells)
        step = self._record(DerivationStep(step_id=self._id("step"), operation=operation, inputs=inputs,
            parents=parents, output=display, engine="linalg", trust=trust))
        self.matrix_producers[mid] = step.step_id
        return MathResult(ok=True, data={"matrix_id": mid, "rows": len(cells), "cols": len(cells[0]),
            "display": display}, trust=trust, engine="linalg", derivation=[step])

    def matrix_create(self, rows: list[list[str]]) -> MathResult:
        if not rows or not all(isinstance(r, list) and r for r in rows):
            return MathResult(ok=False, status="error", errors=["rows must be a non-empty list of non-empty lists"], engine="linalg")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            return MathResult(ok=False, status="error", errors=["Matrix rows must all have the same length"], engine="linalg")
        if len(rows) > self.settings.max_matrix_dim or width > self.settings.max_matrix_dim:
            return MathResult(ok=False, status="error",
                errors=[f"Matrix dimensions must be <= {self.settings.max_matrix_dim}"], engine="linalg")
        try:
            cells = [[parse_math(c) for c in row] for row in rows]
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[f"Invalid cell: {exc}"], engine="linalg")
        return self._store_matrix(cells, "matrix_create",
                                  [f"{len(rows)}x{width}"], [], self._matrix_trust(cells))

    def matrix_get(self, matrix_id: str) -> MathResult:
        cells = self.matrices.get(matrix_id)
        if cells is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown matrix_id: {matrix_id}"], engine="linalg")
        return MathResult(ok=True, data={"matrix_id": matrix_id, "rows": len(cells), "cols": len(cells[0]),
            "display": self._matrix_display(cells),
            "cells": [[render_expr(c) for c in row] for row in cells],
            "producer_step_id": self.matrix_producers.get(matrix_id)},
            trust=self._matrix_trust(cells), engine="linalg")

    def _matrix_op(self, matrix_id: str) -> MathResult | None:
        if matrix_id not in self.matrices:
            return MathResult(ok=False, status="error", errors=[f"Unknown matrix_id: {matrix_id}"], engine="linalg")
        return None

    def matrix_det(self, matrix_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            out = self._to_sp_matrix(matrix_id, ctx).det()
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        return self._finish_symbolic(out, "matrix_det", [matrix_id], [], engine="linalg",
                                     trust=self._matrix_trust(self.matrices[matrix_id]))

    def matrix_inverse(self, matrix_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            inv = self._to_sp_matrix(matrix_id, ctx).inv()
            cells = [[self.sympy.from_sympy(inv[i, j]) for j in range(inv.cols)] for i in range(inv.rows)]
        except sp.matrices.exceptions.NonInvertibleMatrixError:
            return MathResult(ok=False, status="error", errors=["Matrix is singular (det = 0)"], engine="linalg")
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        return self._store_matrix(cells, "matrix_inverse", [matrix_id], [],
                                  self._matrix_trust(self.matrices[matrix_id]))

    def matrix_transpose(self, matrix_id: str) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        cells = self.matrices[matrix_id]
        out = [list(row) for row in zip(*cells)]
        return self._store_matrix(out, "matrix_transpose", [matrix_id], [], self._matrix_trust(cells))

    def matrix_multiply(self, left_id: str, right_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(left_id)): return err
        if (err := self._matrix_op(right_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        a, b = self.matrices[left_id], self.matrices[right_id]
        if len(a[0]) != len(b):
            return MathResult(ok=False, status="error",
                errors=[f"Dimension mismatch: {len(a)}x{len(a[0])} cannot multiply {len(b)}x{len(b[0])}"], engine="linalg")
        try:
            product = self._to_sp_matrix(left_id, ctx) * self._to_sp_matrix(right_id, ctx)
            cells = [[self.sympy.from_sympy(product[i, j]) for j in range(product.cols)] for i in range(product.rows)]
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        trust = self._trust_min(self._matrix_trust(a), self._matrix_trust(b))
        return self._store_matrix(cells, "matrix_multiply", [left_id, right_id], [], trust)

    def matrix_rank(self, matrix_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            rank = self._to_sp_matrix(matrix_id, ctx).rank()
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        trust = self._matrix_trust(self.matrices[matrix_id])
        step = self._record(DerivationStep(step_id=self._id("step"), operation="matrix_rank",
            inputs=[matrix_id], output=str(rank), engine="linalg", trust=trust))
        return MathResult(ok=True, data={"rank": rank}, trust=trust, engine="linalg", derivation=[step])

    def matrix_rref(self, matrix_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            reduced, pivots = self._to_sp_matrix(matrix_id, ctx).rref()
            cells = [[self.sympy.from_sympy(reduced[i, j]) for j in range(reduced.cols)] for i in range(reduced.rows)]
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        result = self._store_matrix(cells, "matrix_rref", [matrix_id], [],
                                    self._matrix_trust(self.matrices[matrix_id]))
        result.data["pivot_columns"] = list(pivots)
        return result

    def matrix_eigenvalues(self, matrix_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            eigenvals = self._to_sp_matrix(matrix_id, ctx).eigenvals()
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        out = []
        warnings = []
        for value, multiplicity in sorted(eigenvals.items(), key=lambda kv: str(kv[0])):
            entry = {"display": str(value), "multiplicity": multiplicity}
            try:
                value_ir = self.sympy.from_sympy(value)
                vid = self._id("expr")
                self.expressions[vid] = value_ir
                self.expression_sources[vid] = str(value)
                entry["expr_id"] = vid
            except ValueError:
                warnings.append(f"Eigenvalue {value} is outside the MathIR fragment; display only.")
            out.append(entry)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="matrix_eigenvalues",
            inputs=[matrix_id], output=", ".join(e["display"] for e in out), engine="linalg",
            trust=self._matrix_trust(self.matrices[matrix_id])))
        return MathResult(ok=True, data={"eigenvalues": out}, warnings=warnings,
            trust=self._matrix_trust(self.matrices[matrix_id]), engine="linalg", derivation=[step])

    def matrix_solve(self, matrix_id: str, rhs_id: str, context_id: str | None = None) -> MathResult:
        if (err := self._matrix_op(matrix_id)): return err
        if (err := self._matrix_op(rhs_id)): return err
        a, b = self.matrices[matrix_id], self.matrices[rhs_id]
        if len(a) != len(a[0]):
            return MathResult(ok=False, status="error", errors=["Coefficient matrix must be square"], engine="linalg")
        if len(b) != len(a):
            return MathResult(ok=False, status="error",
                errors=[f"Right-hand side must have {len(a)} rows"], engine="linalg")
        ctx = self.contexts.get(context_id) if context_id else None
        try:
            solution = self._to_sp_matrix(matrix_id, ctx).LUsolve(self._to_sp_matrix(rhs_id, ctx))
            cells = [[self.sympy.from_sympy(solution[i, j]) for j in range(solution.cols)] for i in range(solution.rows)]
        except sp.matrices.exceptions.NonInvertibleMatrixError:
            return MathResult(ok=False, status="error",
                errors=["System has no unique solution (singular coefficient matrix)"], engine="linalg")
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="linalg")
        return self._store_matrix(cells, "matrix_solve", [matrix_id, rhs_id], [],
                                  self._trust_min(self._matrix_trust(a), self._matrix_trust(b)))

    # ------------------------------------------------------------------
    # Sets, membership, quantifiers (MathIR v2)
    # ------------------------------------------------------------------

    def _store_expr(self, ir: Expr, operation: str, inputs: list[str],
                    trust: TrustLevel, engine: str, source: str | None = None) -> MathResult:
        eid = self._id("expr")
        self.expressions[eid] = ir
        self.expression_sources[eid] = source if source is not None else render_expr(ir)
        step = self._record(DerivationStep(step_id=self._id("step"), operation=operation,
            inputs=inputs, output=render_expr(ir), output_expr_id=eid, engine=engine, trust=trust))
        self.expression_producers[eid] = step.step_id
        return MathResult(ok=True, data={"expr_id": eid, "display": render_expr(ir)},
                          trust=trust, engine=engine, derivation=[step])

    def set_create(self, elements: list[str] | None = None, name: str | None = None) -> MathResult:
        """Create a set: a named standard set (naturals/integers/rationals/reals/complexes/empty)
        or a finite enumerated set from element expressions."""
        if (name is None) == (elements is None):
            return MathResult(ok=False, status="error",
                errors=["Provide exactly one of name= or elements="], engine="sets")
        if name is not None:
            named = {"naturals", "integers", "rationals", "reals", "complexes", "empty"}
            if name not in named:
                return MathResult(ok=False, status="error",
                    errors=[f"Unknown set name: {name}; choose from {sorted(named)}"], engine="sets")
            return self._store_expr(SetNode(name=name), "set_create", [name], TrustLevel.EXACT, "sets")
        try:
            element_ir = [parse_math(e) for e in elements]
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sets")
        kinds: set[str] = set()
        for e in element_ir:
            _scan_ir(e, set(), set(), kinds)
        trust = TrustLevel.EXACT if "real" not in kinds else TrustLevel.NUMERIC
        return self._store_expr(SetNode(elements=element_ir), "set_create",
                                [render_expr(e) for e in element_ir], trust, "sets")

    def set_op(self, op: str, set_ids: list[str]) -> MathResult:
        """Set algebra: union/intersect/difference/complement over stored sets."""
        if op not in {"union", "intersect", "difference", "complement"}:
            return MathResult(ok=False, status="error",
                errors=[f"Unknown set operation: {op}"], engine="sets")
        sets = []
        for sid in set_ids:
            ir = self.expressions.get(sid)
            if ir is None:
                return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {sid}"], engine="sets")
            sets.append(ir)
        if op in {"union", "intersect"} and len(sets) < 2:
            return MathResult(ok=False, status="error", errors=[f"{op} expects at least two sets"], engine="sets")
        if op == "difference" and len(sets) != 2:
            return MathResult(ok=False, status="error", errors=["difference expects exactly two sets"], engine="sets")
        if op == "complement" and len(sets) != 2:
            return MathResult(ok=False, status="error",
                errors=["complement expects (set, universe)"], engine="sets")
        try:
            out = self.sympy.to_sympy(SetOpNode(op=op, args=sets))
            simplified = out.simplify() if hasattr(out, "simplify") else out
            out_ir = self.sympy.from_sympy(simplified)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sets")
        return self._store_expr(out_ir, f"set_op:{op}", set_ids, TrustLevel.EXACT, "sets")

    def set_membership(self, element: str, set_id: str) -> MathResult:
        """Decide element ∈ set. Exact for finite/named sets over exact elements."""
        ir = self.expressions.get(set_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {set_id}"], engine="sets")
        try:
            element_ir = parse_math(element)
            contains = self.sympy.to_sympy(MembershipNode(element=element_ir, set=ir))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sets")
        decided = contains if isinstance(contains, bool) else None
        if decided is None:
            try:
                decided = bool(contains)
            except TypeError:
                decided = None
        if decided is None:
            return MathResult(ok=True, status="unknown", data={"member": None},
                warnings=["Membership could not be decided symbolically."],
                trust=TrustLevel.UNKNOWN, engine="sets")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="set_membership",
            inputs=[element, set_id], output=str(decided), engine="sets", trust=TrustLevel.EXACT))
        return MathResult(ok=True, status="verified" if decided else "refuted",
            data={"member": decided}, trust=TrustLevel.EXACT, engine="sets", derivation=[step])

    def quantifier_check(self, expr_id: str, context_id: str | None = None) -> MathResult:
        """Validity check for a quantified proposition via Z3.

        valid (EXACT) when Z3 proves the negation unsatisfiable; invalid
        (EXACT, with countermodel) when the negation is satisfiable; unknown
        on timeout or when Z3/the fragment is unavailable."""
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="z3")
        if not isinstance(ir, QuantifierNode):
            return MathResult(ok=False, status="error",
                errors=["quantifier_check expects a forall(...)/exists(...) expression"], engine="z3")
        if not self.z3.available:
            return MathResult(ok=True, status="unknown", data={"validity": "unknown"},
                warnings=["Z3 is unavailable; the quantified statement was not checked."],
                trust=TrustLevel.UNKNOWN, engine="z3")
        ctx = self.contexts.get(context_id) if context_id else None
        domains = ctx.domains if ctx else {}
        z3 = self.z3.z3
        try:
            # Truth of the closed quantified statement: Z3 decides the whole
            # formula natively (arbitrary alternation depth). sat means the
            # statement is true, unsat means false.
            env = self.z3._env([ir], domains)
            goal = self.z3.to_z3(ir, env)
            solver = self.z3._new_solver()
            solver.add(goal)
            answer = solver.check()
        except (ValueError, TypeError) as exc:
            return MathResult(ok=True, status="unknown", data={"validity": "unknown"},
                warnings=[f"Statement is outside the current Z3 fragment: {exc}"],
                trust=TrustLevel.UNKNOWN, engine="z3")
        display = render_expr(ir)
        if answer == z3.unknown:
            return MathResult(ok=True, status="unknown", data={"validity": "unknown"},
                warnings=["Z3 returned unknown (timeout or undecidable fragment)."],
                trust=TrustLevel.UNKNOWN, engine="z3")
        is_valid = answer == z3.sat
        # Per-level witness/countermodel extraction: strip leading
        # existentials of the statement (sat-preserving) when it holds, or of
        # its NNF negation (= leading foralls) when it fails, so each level's
        # Skolem constant materializes in the model.
        from .qe import negate, strip_existentials
        witness = None
        levels: list[str] = []
        try:
            source = ir if is_valid else negate(ir)
            constraints, remaining, levels, senv = strip_existentials(
                self.z3, source, domains)
            if levels:
                senv = {**self.z3._env([source], domains), **senv}
                s2 = self.z3._new_solver()
                for c in constraints:
                    s2.add(c)
                s2.add(self.z3.to_z3(remaining, senv))
                if s2.check() == z3.sat:
                    model = s2.model()
                    witness = {name: str(model.eval(senv[name], model_completion=True))
                               for name in levels}
        except (ValueError, TypeError):
            witness = None
        step = self._record(DerivationStep(step_id=self._id("step"), operation="quantifier_check",
            inputs=[expr_id], output=f"{'valid' if is_valid else 'invalid'}: {display}"
            + (f"; witness {witness}" if witness else ""),
            engine="z3", trust=TrustLevel.EXACT))
        data = {"validity": "valid" if is_valid else "invalid",
                "skolem_levels": levels}
        if witness:
            data["witness" if is_valid else "countermodel"] = witness
        return MathResult(ok=True, status="verified" if is_valid else "refuted",
            data=data, trust=TrustLevel.EXACT, engine="z3", derivation=[step])

    def quantifier_eliminate(self, expr_id: str,
                             context_id: str | None = None) -> MathResult:
        """True quantifier elimination via Z3's qe tactic (LRA/LIA): returns an
        equivalent quantifier-free MathIR formula that renders and re-parses.
        EXACT on success; structured UNKNOWN (with fragment classification)
        on undecidable fragments or failure."""
        from .qe import quantifier_eliminate
        from .rendering import render_expr as _render
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="z3")
        ctx = self.contexts.get(context_id) if context_id else None
        domains = ctx.domains if ctx else {}
        out = quantifier_eliminate(self.z3, ir, domains,
                                   self.settings.max_qe_variables)
        if out["status"] != "eliminated":
            return MathResult(ok=True, status="unknown", trust=TrustLevel.UNKNOWN,
                              data=out, engine="z3")
        formula = out.pop("formula")
        source = _render(formula)
        fid = self._id("expr")
        self.expressions[fid] = formula
        self.expression_sources[fid] = source
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="quantifier_eliminate", inputs=[expr_id], output=source,
            engine="z3", trust=TrustLevel.EXACT))
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                          data={**out, "expr_id": fid, "formula": source},
                          engine="z3", derivation=[step])

    def quantifier_eliminate_batch(self, expr_ids: list[str],
                                   context_id: str | None = None,
                                   workers: int | None = None) -> MathResult:
        """Batch QE over the process pool (QE is single-threaded inside Z3,
        so parallelism is across problems)."""
        from .qe import quantifier_eliminate_batch
        if len(expr_ids) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                              errors=[f"batch exceeds max_batch_jobs={self.settings.max_batch_jobs}"],
                              engine="z3")
        ctx = self.contexts.get(context_id) if context_id else None
        domains = ctx.domains if ctx else {}
        jobs = []
        for eid in expr_ids:
            src = self.expression_sources.get(eid)
            if src is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"Unknown expr_id: {eid}"], engine="z3")
            jobs.append({"expr_id": eid, "source": src, "domains": domains,
                         "timeout_ms": self.settings.z3_timeout_ms,
                         "max_variables": self.settings.max_qe_variables})
        results = quantifier_eliminate_batch(jobs, workers)
        n_ok = sum(1 for r in results if r["status"] == "eliminated")
        trust = TrustLevel.EXACT if n_ok == len(results) and results else TrustLevel.UNKNOWN
        return MathResult(ok=True, status="ok", trust=trust,
                          data={"total": len(results), "eliminated": n_ok,
                                "results": results}, engine="z3")

    # ------------------------------------------------------------------
    # Exact polynomial algebra
    # ------------------------------------------------------------------

    def _poly_operands(self, expr_ids: list[str], context_id: str | None):
        from . import polynomial as _poly  # noqa: F401  (import check)
        irs = []
        for eid in expr_ids:
            ir = self.expressions.get(eid)
            if ir is None:
                return None, None, MathResult(ok=False, status="error",
                    errors=[f"Unknown expr_id: {eid}"], engine="polynomial")
            irs.append(ir)
        ctx = self.contexts.get(context_id) if context_id else None
        env = self._symbol_env(ctx, *irs)
        try:
            return [self.sympy.to_sympy(ir, env) for ir in irs], env, None
        except (ValueError, TypeError) as exc:
            return None, None, MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")

    def _poly_finish(self, out, operation: str, inputs: list[str]) -> MathResult:
        return self._finish_symbolic(out, operation, inputs, self._parents_for_exprs(inputs),
                                     engine="polynomial", trust=TrustLevel.EXACT)

    def poly_groebner(self, expr_ids: list[str], variables: list[str],
                      order: str = "lex", context_id: str | None = None) -> MathResult:
        """Reduced Gröbner basis of the ideal generated by the given polynomials. Exact."""
        from .polynomial import groebner_basis
        polys, _env, err = self._poly_operands(expr_ids, context_id)
        if err: return err
        try:
            syms = [sp.Symbol(v) for v in variables]
            basis = run_with_timeout(groebner_basis, self.settings.solver_timeout_seconds,
                                     polys, syms, order)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        basis_ids = []
        displays = []
        for g in basis:
            r = self._poly_finish(g, "poly_groebner:basis_element", expr_ids)
            basis_ids.append(r.data["result_expr_id"])
            displays.append(r.data["result"])
        step = self._record(DerivationStep(step_id=self._id("step"), operation="poly_groebner",
            inputs=expr_ids, output=f"basis of {len(basis)} element(s), order={order}",
            engine="polynomial", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"basis": displays, "basis_expr_ids": basis_ids,
            "order": order, "variables": variables}, trust=TrustLevel.EXACT,
            engine="polynomial", derivation=[step])

    def poly_divide(self, dividend_id: str, divisor_ids: list[str], variables: list[str],
                    order: str = "lex", context_id: str | None = None) -> MathResult:
        """Exact multivariate polynomial division: quotients and remainder."""
        from .polynomial import poly_divide
        polys, _env, err = self._poly_operands([dividend_id, *divisor_ids], context_id)
        if err: return err
        try:
            syms = [sp.Symbol(v) for v in variables]
            quotients, remainder = run_with_timeout(
                poly_divide, self.settings.solver_timeout_seconds, polys[0], polys[1:], syms, order)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        qr = self._poly_finish(remainder, "poly_divide:remainder", [dividend_id, *divisor_ids])
        step = self._record(DerivationStep(step_id=self._id("step"), operation="poly_divide",
            inputs=[dividend_id, *divisor_ids], output=f"remainder {qr.data['result']}",
            engine="polynomial", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"quotients": [str(q) for q in quotients],
            "remainder": qr.data["result"], "remainder_expr_id": qr.data["result_expr_id"]},
            trust=TrustLevel.EXACT, engine="polynomial", derivation=[step])

    def poly_resultant(self, a_id: str, b_id: str, variable: str,
                       context_id: str | None = None) -> MathResult:
        from .polynomial import resultant_poly
        polys, _env, err = self._poly_operands([a_id, b_id], context_id)
        if err: return err
        try:
            out = run_with_timeout(resultant_poly, self.settings.solver_timeout_seconds,
                                   polys[0], polys[1], sp.Symbol(variable))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        return self._poly_finish(out, "poly_resultant", [a_id, b_id])

    def poly_discriminant(self, expr_id: str, variable: str,
                          context_id: str | None = None) -> MathResult:
        from .polynomial import discriminant_poly
        polys, _env, err = self._poly_operands([expr_id], context_id)
        if err: return err
        try:
            out = run_with_timeout(discriminant_poly, self.settings.solver_timeout_seconds,
                                   polys[0], sp.Symbol(variable))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        return self._poly_finish(out, "poly_discriminant", [expr_id])

    def poly_factor(self, expr_id: str, extension: str | None = None,
                    context_id: str | None = None) -> MathResult:
        """Exact factorization over ZZ/QQ, optionally over an algebraic extension."""
        from .polynomial import factor_poly
        polys, _env, err = self._poly_operands([expr_id], context_id)
        if err: return err
        try:
            constant, factors = run_with_timeout(
                factor_poly, self.settings.solver_timeout_seconds, polys[0], None, extension)
        except (ValueError, TypeError, sp.SympifyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="poly_factor",
            inputs=[expr_id], output=f"{len(factors)} factor(s)", engine="polynomial",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"constant": str(constant),
            "factors": [{"factor": str(f), "multiplicity": m} for f, m in factors]},
            trust=TrustLevel.EXACT, engine="polynomial", derivation=[step])

    def ideal_membership(self, expr_id: str, generator_ids: list[str], variables: list[str],
                         order: str = "lex", context_id: str | None = None) -> MathResult:
        """Exact ideal membership: is expr in <generators>? (Gröbner remainder test.)"""
        from .polynomial import ideal_membership as _membership
        polys, _env, err = self._poly_operands([expr_id, *generator_ids], context_id)
        if err: return err
        try:
            syms = [sp.Symbol(v) for v in variables]
            is_member, remainder = run_with_timeout(
                _membership, self.settings.solver_timeout_seconds, polys[0], polys[1:], syms, order)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="polynomial")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="ideal_membership",
            inputs=[expr_id, *generator_ids], output=str(is_member), engine="polynomial",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, status="verified" if is_member else "refuted",
            data={"member": is_member, "remainder": str(remainder)},
            trust=TrustLevel.EXACT, engine="polynomial", derivation=[step])

    def poly_groebner_batch(self, jobs: list[dict], workers: int | None = None) -> MathResult:
        """Independent Gröbner bases across worker processes. Each job:
        {"polys": [str, ...], "variables": [str, ...], "order": "lex"?}."""
        from .polynomial import groebner_batch
        if not jobs:
            return MathResult(ok=False, status="error", errors=["jobs must not be empty"], engine="polynomial")
        if len(jobs) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                errors=[f"Batch has {len(jobs)} jobs; limit is {self.settings.max_batch_jobs}."],
                engine="polynomial")
        resolved = resolve_workers(workers, cap=self.settings.max_workers)
        results = groebner_batch(jobs, workers=resolved)
        failed = sum(1 for r in results if not r.get("ok"))
        step = self._record(DerivationStep(step_id=self._id("step"), operation="poly_groebner_batch",
            inputs=[f"{len(jobs)} jobs"], output=f"{len(results) - failed} ok, {failed} failed",
            engine="polynomial", trust=TrustLevel.EXACT))
        return MathResult(ok=failed < len(results),
            data={"results": results, "job_count": len(results), "failed": failed,
                  "workers": resolved},
            trust=TrustLevel.EXACT, engine="polynomial", derivation=[step])

    # ------------------------------------------------------------------
    # Probability and statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _frac_or_error(text: str) -> Fraction:
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"not an exact rational: {text!r}") from exc

    def _get_rv(self, rv_id: str):
        rv = self.random_variables.get(rv_id)
        if rv is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown rv_id: {rv_id}"], engine="probability")
        return rv, None

    def prob_rv_create(self, values: list[str], probabilities: list[str]) -> MathResult:
        """Create a discrete random variable with exact rational probabilities."""
        from .probability import DiscreteRV
        try:
            rv = DiscreteRV([self._frac_or_error(v) for v in values],
                            [self._frac_or_error(p) for p in probabilities])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        rv_id = self._id("rv")
        self.random_variables[rv_id] = rv
        step = self._record(DerivationStep(step_id=self._id("step"), operation="prob_rv_create",
            inputs=[f"{len(values)} outcomes"], output=f"rv_id={rv_id}",
            engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"rv_id": rv_id, "support": len(values)},
                          trust=TrustLevel.EXACT, engine="probability", derivation=[step])

    def prob_expectation(self, rv_id: str, power: int = 1) -> MathResult:
        rv, err = self._get_rv(rv_id)
        if err: return err
        try:
            value = rv.expectation(power)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="prob_expectation" if power == 1 else f"prob_moment:{power}",
            inputs=[rv_id], output=str(value), engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"value": str(value)}, trust=TrustLevel.EXACT,
                          engine="probability", derivation=[step])

    def prob_variance(self, rv_id: str) -> MathResult:
        rv, err = self._get_rv(rv_id)
        if err: return err
        value = rv.variance()
        step = self._record(DerivationStep(step_id=self._id("step"), operation="prob_variance",
            inputs=[rv_id], output=str(value), engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"value": str(value)}, trust=TrustLevel.EXACT,
                          engine="probability", derivation=[step])

    def prob_covariance(self, x_values: list[str], y_values: list[str],
                        joint_probabilities: list[str]) -> MathResult:
        """Exact covariance from a joint pmf over paired outcomes."""
        from .probability import covariance
        try:
            value = covariance([self._frac_or_error(v) for v in x_values],
                               [self._frac_or_error(v) for v in y_values],
                               [self._frac_or_error(p) for p in joint_probabilities])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="prob_covariance",
            inputs=[f"{len(x_values)} joint outcomes"], output=str(value),
            engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"value": str(value)}, trust=TrustLevel.EXACT,
                          engine="probability", derivation=[step])

    def prob_bayes(self, prior: list[str], likelihood: list[str]) -> MathResult:
        """Exact Bayes posterior: posterior_i ∝ likelihood_i * prior_i."""
        from .probability import bayes
        try:
            posterior = bayes([self._frac_or_error(p) for p in prior],
                              [self._frac_or_error(l) for l in likelihood])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="prob_bayes",
            inputs=[f"{len(prior)} hypotheses"], output=str([str(p) for p in posterior]),
            engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"posterior": [str(p) for p in posterior]},
                          trust=TrustLevel.EXACT, engine="probability", derivation=[step])

    def _matrix_fractions(self, matrix_id: str):
        entry = self.matrices.get(matrix_id)
        if entry is None:
            return None, MathResult(ok=False, status="error",
                errors=[f"Unknown matrix_id: {matrix_id}"], engine="probability")
        grid = []
        for row in entry:
            frow = []
            for cell in row:
                if isinstance(cell, IntegerNode):
                    frow.append(Fraction(cell.value))
                elif isinstance(cell, RationalNode):
                    frow.append(Fraction(int(cell.numerator), int(cell.denominator)))
                else:
                    return None, MathResult(ok=False, status="error",
                        errors=["Markov chain tools require exact integer/rational matrix entries"],
                        engine="probability")
            grid.append(frow)
        return grid, None

    def prob_markov_stationary(self, matrix_id: str) -> MathResult:
        """Exact stationary distribution of a rational row-stochastic matrix."""
        from .probability import stationary_distribution
        grid, err = self._matrix_fractions(matrix_id)
        if err: return err
        try:
            pi = run_with_timeout(stationary_distribution,
                                  self.settings.solver_timeout_seconds, grid)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="prob_markov_stationary", inputs=[matrix_id],
            output=str([str(p) for p in pi]), engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"stationary": [str(p) for p in pi]},
                          trust=TrustLevel.EXACT, engine="probability", derivation=[step])

    def prob_markov_hitting_time(self, matrix_id: str, targets: list[int]) -> MathResult:
        """Exact expected hitting times to the target state set (null = unreachable)."""
        from .probability import hitting_times
        grid, err = self._matrix_fractions(matrix_id)
        if err: return err
        try:
            h = run_with_timeout(hitting_times, self.settings.solver_timeout_seconds,
                                 grid, targets)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="prob_markov_hitting_time", inputs=[matrix_id, f"targets={targets}"],
            output=str([str(x) if x is not None else None for x in h]),
            engine="probability", trust=TrustLevel.EXACT))
        return MathResult(ok=True,
            data={"hitting_times": [str(x) if x is not None else None for x in h]},
            trust=TrustLevel.EXACT, engine="probability", derivation=[step])

    def prob_sample(self, rv_id: str, n: int, seed: int | None = None) -> MathResult:
        """Draw n samples (numeric evidence; seeded for reproducibility)."""
        rv, err = self._get_rv(rv_id)
        if err: return err
        try:
            draws = rv.sample(n, seed)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="probability")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="prob_sample",
            inputs=[rv_id, f"n={n}", f"seed={seed}"], output=f"{n} samples",
            engine="probability", trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data={"samples": [str(d) for d in draws], "n": n},
                          trust=TrustLevel.NUMERIC, engine="probability", derivation=[step])

    def prob_distribution(self, distribution: str, parameters: list[str], query: str,
                          point: str | None = None) -> MathResult:
        """Legacy SymPy bridge; prefer object_create + apply for evidence.
        query: expectation | variance | std | density | cdf (density/cdf need point)."""
        from .probability import sympy_stats_query
        try:
            out = run_with_timeout(sympy_stats_query, self.settings.solver_timeout_seconds,
                                   distribution, parameters, query, point)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sympy.stats")
        result = self._finish_symbolic(
            out, f"prob_distribution:{query}", [distribution, *parameters],
            [], engine="sympy.stats")
        result.warnings.append(
            "prob_distribution is a legacy compatibility bridge; use the "
            "typed Distribution object with apply for claim-specific evidence.")
        return result

    def stats_moments(self, values: list[str], max_order: int = 4) -> MathResult:
        """Exact sample moments (mean, variances, central moments) via Fraction."""
        from .statistics import sample_moments
        try:
            m = sample_moments([self._frac_or_error(v) for v in values], max_order)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_moments",
            inputs=[f"n={m['n']}"], output=f"mean={m['mean']}", engine="statistics",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={
            "n": m["n"], "mean": str(m["mean"]),
            "variance_population": str(m["variance_population"]),
            "variance_sample": str(m["variance_sample"]) if m["variance_sample"] is not None else None,
            "central_moments": {str(k): str(v) for k, v in m["central_moments"].items()}},
            trust=TrustLevel.EXACT, engine="statistics", derivation=[step])

    def stats_order(self, values: list[str]) -> MathResult:
        """Exact order statistics: sorted sample, min/max, median, quartiles."""
        from .statistics import order_statistics
        try:
            o = order_statistics([self._frac_or_error(v) for v in values])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_order",
            inputs=[f"n={len(o['sorted'])}"], output=f"median={o['median']}",
            engine="statistics", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={
            "sorted": [str(x) for x in o["sorted"]], "min": str(o["min"]),
            "max": str(o["max"]), "median": str(o["median"]),
            "q1": str(o["q1"]), "q3": str(o["q3"])},
            trust=TrustLevel.EXACT, engine="statistics", derivation=[step])

    def stats_regression(self, x_values: list[str], y_values: list[str]) -> MathResult:
        """Exact rational least-squares regression: slope, intercept, R²."""
        from .statistics import linear_regression
        try:
            r = linear_regression([self._frac_or_error(v) for v in x_values],
                                  [self._frac_or_error(v) for v in y_values])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_regression",
            inputs=[f"n={r['n']}"], output=f"slope={r['slope']}", engine="statistics",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"slope": str(r["slope"]),
            "intercept": str(r["intercept"]),
            "r_squared": str(r["r_squared"]) if r["r_squared"] is not None else None,
            "n": r["n"]}, trust=TrustLevel.EXACT, engine="statistics", derivation=[step])

    def stats_correlation(self, x_values: list[str], y_values: list[str]) -> MathResult:
        """Pearson correlation: exact r² and covariance; r itself is high-precision
        numeric (square root), so overall trust is numeric_high_precision."""
        from .statistics import correlation
        try:
            r = correlation([self._frac_or_error(v) for v in x_values],
                            [self._frac_or_error(v) for v in y_values])
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_correlation",
            inputs=[f"n={len(x_values)}"], output=f"r^2={r['r_squared']}",
            engine="statistics", trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data={"r_squared": str(r["r_squared"]), "r": r["r"],
            "covariance_population": str(r["covariance_population"])},
            trust=TrustLevel.NUMERIC_HIGH_PRECISION, engine="statistics", derivation=[step])

    def stats_ttest(self, values: list[str], mu0: str) -> MathResult:
        """One-sample t-test (mpmath; numeric_high_precision, not proof)."""
        from .statistics import t_test_1samp
        try:
            r = t_test_1samp([self._frac_or_error(v) for v in values], mu0)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_ttest",
            inputs=[f"n={len(values)}", f"mu0={mu0}"], output=f"t={r['t']}",
            engine="statistics", trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data=r, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="statistics", derivation=[step])

    def stats_chi2(self, observed: list[str], expected: list[str] | None = None) -> MathResult:
        """Pearson chi-square goodness-of-fit (mpmath; numeric_high_precision)."""
        from .statistics import chi_square_test
        try:
            r = chi_square_test([self._frac_or_error(v) for v in observed],
                                [self._frac_or_error(v) for v in expected] if expected else None)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_chi2",
            inputs=[f"k={len(observed)}"], output=f"chi2={r['chi2']}",
            engine="statistics", trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data=r, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="statistics", derivation=[step])

    def stats_confidence_interval(self, values: list[str], confidence: str = "0.95") -> MathResult:
        """t-based confidence interval for the mean (numeric_high_precision)."""
        from .statistics import mean_confidence_interval
        try:
            r = mean_confidence_interval([self._frac_or_error(v) for v in values], confidence)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="stats_confidence_interval", inputs=[f"n={len(values)}", confidence],
            output=f"[{r['lower']}, {r['upper']}]", engine="statistics",
            trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data=r, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="statistics", derivation=[step])

    def stats_batch_moments(self, columns: list[list[str]], workers: int | None = None,
                            numeric: bool = False) -> MathResult:
        """Moments for many sample columns across the process pool.
        numeric=True takes the vectorized float64 path (trust numeric)."""
        from .statistics import batch_moments
        if not columns:
            return MathResult(ok=False, status="error", errors=["columns must not be empty"],
                              engine="statistics")
        if len(columns) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                errors=[f"Batch has {len(columns)} columns; limit is {self.settings.max_batch_jobs}."],
                engine="statistics")
        try:
            prepared = [[self._frac_or_error(v) for v in col] if not numeric
                        else [float(v) for v in col] for col in columns]
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="statistics")
        resolved = resolve_workers(workers, cap=self.settings.max_workers)
        results = batch_moments(prepared, workers=resolved, numeric=numeric)
        trust = TrustLevel.NUMERIC if numeric else TrustLevel.EXACT
        step = self._record(DerivationStep(step_id=self._id("step"), operation="stats_batch_moments",
            inputs=[f"{len(columns)} columns"], output=f"workers={resolved}",
            engine="statistics", trust=trust))
        return MathResult(ok=True, data={"results": results, "workers": resolved},
                          trust=trust, engine="statistics", derivation=[step])

    # ------------------------------------------------------------------
    # Sparse tensors
    # ------------------------------------------------------------------

    def _get_tensor(self, tensor_id: str):
        t = self.tensors.get(tensor_id)
        if t is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown tensor_id: {tensor_id}"], engine="tensors")
        return t, None

    def tensor_create(self, shape: list[int], entries: dict[str, str]) -> MathResult:
        """Create a sparse tensor. entries maps 'i,j,k' index strings to exact
        rational values ('2', '1/3') or float literals ('0.5' -> numeric trust)."""
        from .tensors import SparseTensor
        try:
            parsed = {}
            numeric = False
            for key, value in entries.items():
                idx = tuple(int(p) for p in key.split(","))
                if any(c in value for c in ".eE") and "/" not in value:
                    parsed[idx] = float(value)
                    numeric = True
                else:
                    parsed[idx] = self._frac_or_error(value)
            tensor = SparseTensor(tuple(shape), parsed)
        except (ValueError, AttributeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="tensors")
        tid = self._id("tensor")
        self.tensors[tid] = tensor
        trust = TrustLevel.NUMERIC if numeric else TrustLevel.EXACT
        step = self._record(DerivationStep(step_id=self._id("step"), operation="tensor_create",
            inputs=[f"shape={shape}", f"nnz={tensor.nnz}"], output=tid, engine="tensors",
            trust=trust))
        return MathResult(ok=True, data={"tensor_id": tid, **tensor.summary()},
                          trust=trust, engine="tensors", derivation=[step])

    def tensor_get(self, tensor_id: str) -> MathResult:
        """Fetch a tensor: full entries when small, shape/nnz summary when large."""
        t, err = self._get_tensor(tensor_id)
        if err: return err
        data = {"tensor_id": tensor_id, **t.summary()}
        if t.nnz <= 256:
            data["entries"] = {",".join(str(i) for i in k): str(v) for k, v in t.entries.items()}
        else:
            data["entries_omitted"] = True
        trust = TrustLevel.NUMERIC if any(isinstance(v, float) for v in t.entries.values()) \
            else TrustLevel.EXACT
        return MathResult(ok=True, data=data, trust=trust, engine="tensors")

    def tensor_contract(self, spec: str, tensor_ids: list[str],
                        exact: bool = True) -> MathResult:
        """Einstein-style contraction, e.g. 'ij,jk->ik'. exact=True uses exact
        rational arithmetic; exact=False dispatches to njit/CuPy sparse tiers."""
        from .tensors import contract, contract_numeric
        tensors = []
        for tid in tensor_ids:
            t, err = self._get_tensor(tid)
            if err: return err
            tensors.append(t)
        input_numeric = any(isinstance(v, float) for t in tensors for v in t.entries.values())
        try:
            if exact and not input_numeric:
                out = contract(spec, *tensors)
                tier = "exact"
                trust = TrustLevel.EXACT
            else:
                out, tier = contract_numeric(spec, *tensors)
                trust = TrustLevel.NUMERIC
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="tensors")
        tid = self._id("tensor")
        self.tensors[tid] = out
        step = self._record(DerivationStep(step_id=self._id("step"), operation="tensor_contract",
            inputs=tensor_ids, output=f"{spec} -> {out.summary()}", engine="tensors", trust=trust))
        return MathResult(ok=True, data={"tensor_id": tid, "engine_tier": tier,
            **out.summary()}, trust=trust, engine="tensors", derivation=[step])

    def tensor_solve(self, a_id: str, b_id: str) -> MathResult:
        """Exact sparse solve A x = b over the rationals (sparse Gauss-Jordan)."""
        from .tensors import sparse_solve_exact
        a, err = self._get_tensor(a_id)
        if err: return err
        b, err = self._get_tensor(b_id)
        if err: return err
        for tid, t in ((a_id, a), (b_id, b)):
            if any(isinstance(v, float) for v in t.entries.values()):
                return MathResult(ok=False, status="error",
                    errors=[f"{tid} has numeric entries; exact solve requires rationals"],
                    engine="tensors")
        try:
            x = run_with_timeout(sparse_solve_exact, self.settings.solver_timeout_seconds, a, b)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="tensors")
        tid = self._id("tensor")
        self.tensors[tid] = x
        step = self._record(DerivationStep(step_id=self._id("step"), operation="tensor_solve",
            inputs=[a_id, b_id], output=f"x {x.summary()}", engine="tensors",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"tensor_id": tid, **x.summary()},
                          trust=TrustLevel.EXACT, engine="tensors", derivation=[step])

    # ------------------------------------------------------------------
    # Certified numerics: roots, quadrature, ODE/PDE, optimization
    # ------------------------------------------------------------------

    def _numeric_bound(self, text: str) -> str:
        """Parse a bound as MathIR (supports pi, fractions, expressions) and
        render it as a high-precision decimal string for mpmath."""
        import mpmath as mp
        value = self._parse_bound(text, {})
        # bare `pi`/`e` parse as symbols in the restricted grammar; bind them
        # to the constants for numeric evaluation
        value = value.subs({sp.Symbol("pi"): sp.pi, sp.Symbol("e"): sp.E})
        return mp.nstr(mp.mpf(str(sp.N(value, 60))), 50)

    def _mpmath_fn(self, expr_id: str, variable: str):
        ir = self.expressions.get(expr_id)
        if ir is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown expr_id: {expr_id}"], engine="numerics")
        try:
            sym = self.sympy.to_sympy(ir, self._symbol_env(None, ir))
            import sympy as sp
            from .numerics import MpmathExpressionCallable
            return MpmathExpressionCallable(sym, sp.Symbol(variable)), None
        except (ValueError, TypeError) as exc:
            return None, MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")

    def unit_check(self, expr_id: str, units: dict[str, str] | None = None) -> MathResult:
        """Dimensional analysis of an expression. `units` maps free symbols to
        unit expressions ("m/s^2"). On success the dimension is recorded in the
        expression's SemanticMeta.units. Dimensional errors are hard errors."""
        from .units import dimension_of
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="units")
        try:
            dim = dimension_of(ir, units or {})
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="units")
        ir.meta.units = str(dim)
        ir.meta.inferred = True
        step = self._record(DerivationStep(step_id=self._id("step"), operation="unit_check",
                                           inputs=[expr_id], outputs=[expr_id],
                                           method="si-dimension-vector", engine="units"))
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                          data={"dimension": str(dim)}, derivation=[step], engine="units")

    def unit_convert(self, value: str, from_unit: str, to_unit: str) -> MathResult:
        """Exact rational unit conversion; dimensional mismatch is an error."""
        from .units import convert
        try:
            out = convert(value, from_unit, to_unit)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="units")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="unit_convert",
                                           inputs=[f"{value} {from_unit}"],
                                           outputs=[f"{out['value']} {to_unit}"],
                                           method="exact-rational-scale", engine="units"))
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                          data=out, derivation=[step], engine="units")

    def unit_simplify(self, unit: str) -> MathResult:
        """Reduce a unit expression to its SI dimension and exact scale factor."""
        from .units import simplify_unit
        try:
            out = simplify_unit(unit)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="units")
        return MathResult(ok=True, status="ok", trust=TrustLevel.EXACT,
                          data=out, engine="units")

    def root_find(self, expr_id: str, variable: str, a: str | None = None,
                  b: str | None = None, x0: str | None = None,
                  certified: bool = False, dps: int = 50,
                  fast: bool = False) -> MathResult:
        """Root finding. certified=True: interval isolation via mpmath.iv
        (interval_certified). Default: arbitrary-precision Brent/secant
        (numeric_high_precision). fast=True: float64 Brent (numeric)."""
        from .numerics import (brent_float64, compile_float64, find_root_mpmath,
                               isolate_roots_interval)
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="numerics")
        try:
            a = self._numeric_bound(a) if a is not None else None
            b = self._numeric_bound(b) if b is not None else None
            x0 = self._numeric_bound(x0) if x0 is not None else None
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
        if certified:
            if a is None or b is None:
                return MathResult(ok=False, status="error",
                    errors=["certified isolation requires a bracket [a, b]"], engine="numerics")
            f_iv, err = self._mpmath_fn(expr_id, variable)
            if err: return err
            try:
                roots = isolate_roots_interval(f_iv, a, b)
            except (ValueError, TypeError) as exc:
                return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
            step = self._record(DerivationStep(step_id=self._id("step"),
                operation="root_find:certified", inputs=[expr_id, f"[{a}, {b}]"],
                output=f"{len(roots)} certified interval(s)", engine="numerics",
                trust=TrustLevel.INTERVAL_CERTIFIED))
            return MathResult(ok=True, data={"certified_intervals": roots, "count": len(roots)},
                              trust=TrustLevel.INTERVAL_CERTIFIED, engine="numerics",
                              derivation=[step])
        if fast:
            if a is None or b is None:
                return MathResult(ok=False, status="error",
                    errors=["the float64 fast path requires a bracket [a, b]"], engine="numerics")
            try:
                fn = compile_float64(ir, [variable], njit=True) or compile_float64(ir, [variable])
                root = brent_float64(fn, float(a), float(b), tol=self.settings.tolerance,
                                     max_iter=self.settings.max_iterations)
            except (ValueError, TypeError) as exc:
                return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
            step = self._record(DerivationStep(step_id=self._id("step"),
                operation="root_find:float64", inputs=[expr_id, f"[{a}, {b}]"],
                output=repr(root), engine="numerics", trust=TrustLevel.NUMERIC))
            return MathResult(ok=True, data={"root": repr(root), "method": "brent-float64"},
                              trust=TrustLevel.NUMERIC, engine="numerics", derivation=[step])
        f, err = self._mpmath_fn(expr_id, variable)
        if err: return err
        try:
            out = run_with_timeout(find_root_mpmath, self.settings.solver_timeout_seconds,
                                   f, a, b, x0, dps)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="root_find",
            inputs=[expr_id], output=out["root"], engine="numerics",
            trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="numerics", derivation=[step])

    def root_scan(self, expr_id: str, variable: str, a: str, b: str,
                  intervals: int = 64, dps: int = 50,
                  workers: int | None = None) -> MathResult:
        """Parallel scan of [a, b] for sign-changing roots (process pool)."""
        from .numerics import root_scan
        source = self.expression_sources.get(expr_id)
        if source is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="numerics")
        try:
            a, b = self._numeric_bound(a), self._numeric_bound(b)
            resolved = resolve_workers(workers, cap=self.settings.max_workers)
            results = run_with_timeout(root_scan, self.settings.solver_timeout_seconds,
                                       source, variable, a, b, intervals, dps, resolved)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
        found = [r for r in results if r.get("ok")]
        step = self._record(DerivationStep(step_id=self._id("step"), operation="root_scan",
            inputs=[expr_id, f"[{a}, {b}] x {intervals}"],
            output=f"{len(found)} root(s)", engine="numerics",
            trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data={"roots": found, "intervals_scanned": intervals,
            "workers": resolved}, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
            engine="numerics", derivation=[step])

    def quadrature(self, expr_id: str, variable: str, a: str, b: str,
                   dps: int = 50, cross_check: bool = True) -> MathResult:
        """Definite integral via tanh-sinh at arbitrary precision, cross-checked
        with Gauss-Legendre. Disagreement surfaces as a conflict."""
        from .numerics import quadrature_mpmath
        f, err = self._mpmath_fn(expr_id, variable)
        if err: return err
        try:
            a, b = self._numeric_bound(a), self._numeric_bound(b)
            out = run_with_timeout(quadrature_mpmath, self.settings.solver_timeout_seconds,
                                   f, a, b, dps, cross_check)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="numerics")
        conflict = out.pop("conflict", False)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="quadrature",
            inputs=[expr_id, f"[{a}, {b}]"], output=out["value"], engine="numerics",
            trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        warnings = ["tanh-sinh and Gauss-Legendre disagree at the requested "
                    "precision; treat the value as unreliable."] if conflict else []
        return MathResult(ok=True, status="conflict" if conflict else "ok", data=out,
                          warnings=warnings, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="numerics", derivation=[step])

    def ode_solve(self, rhs_id: str, y_var: str = "y", x_var: str = "x",
                  ics: dict[str, str] | None = None) -> MathResult:
        """Symbolic dy/dx = rhs(x, y) via sympy.dsolve with classification."""
        from .ode import dsolve_symbolic
        ir = self.expressions.get(rhs_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {rhs_id}"], engine="ode")
        try:
            env = self._symbol_env(None, ir)
            env[x_var] = sp.Symbol(x_var)
            env[y_var] = sp.Symbol(y_var)  # y may appear in the rhs as a symbol
            rhs = self.sympy.to_sympy(ir, env)
            input_trust = self._expr_trust(ir)
            parsed_ics = None
            if ics:
                f_ = sp.Function(y_var)
                parsed_ics = {}
                ic_levels = [input_trust]
                for raw_x, raw_y in ics.items():
                    x_ir = parse_math(str(raw_x))
                    y_ir = parse_math(str(raw_y))
                    parsed_ics[f_(self.sympy.to_sympy(x_ir, env))] = (
                        self.sympy.to_sympy(y_ir, env)
                    )
                    ic_levels.extend([self._expr_trust(x_ir), self._expr_trust(y_ir)])
                input_trust = self._trust_min(*ic_levels)
            solution, classification = run_with_timeout(
                dsolve_symbolic, self.settings.solver_timeout_seconds,
                rhs, y_var, x_var, parsed_ics)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="ode")
        trust = self._trust_min(TrustLevel.SYMBOLIC, input_trust)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="ode_solve",
            inputs=[rhs_id], output=str(solution), engine="ode", trust=trust))
        return MathResult(ok=True, data={"solution": str(solution),
            "classification": classification}, trust=trust,
            engine="ode", derivation=[step])

    def ode_solve_numeric(self, rhs_ids: list[str], t_span: list[str], y0: list[str],
                          tol: float | None = None, dps: int = 50,
                          fast: bool = False) -> MathResult:
        """Numeric IVP for a first-order system. Default: adaptive RK45 on
        mpmath (numeric_high_precision). fast=True: float64 RK4 (numeric)."""
        from .ode import compile_rhs_float64, rk45_mpmath, rk4_float64
        irs = []
        for rid in rhs_ids:
            ir = self.expressions.get(rid)
            if ir is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"Unknown expr_id: {rid}"], engine="ode")
            irs.append(ir)
        tol = tol if tol is not None else self.settings.tolerance
        y_vars = [f"y{i}" for i in range(len(irs))]
        try:
            t_span = [self._numeric_bound(t) for t in t_span]
            y0 = [self._numeric_bound(v) for v in y0]
            if fast:
                f = compile_rhs_float64(irs, y_vars, njit=True) or \
                    compile_rhs_float64(irs, y_vars)
                y = rk4_float64(f, float(t_span[0]), [float(v) for v in y0],
                                float(t_span[1]), 10_000)
                step = self._record(DerivationStep(step_id=self._id("step"),
                    operation="ode_solve_numeric:float64", inputs=rhs_ids,
                    output=str(y), engine="ode", trust=TrustLevel.NUMERIC))
                return MathResult(ok=True, data={"t": t_span[1], "y": [repr(v) for v in y],
                    "method": "rk4-float64"}, trust=TrustLevel.NUMERIC, engine="ode",
                    derivation=[step])
            f_py = compile_rhs_float64(irs, y_vars)
            import mpmath as mp
            def f_mp(t, y):
                out = [mp.mpf(0)] * len(y)
                f_py(t, y, out)
                return [mp.mpf(v) for v in out]
            out = run_with_timeout(rk45_mpmath, self.settings.solver_timeout_seconds,
                                   f_mp, t_span[0], y0, t_span[1], tol,
                                   self.settings.max_ode_steps, dps)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="ode")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="ode_solve_numeric", inputs=rhs_ids,
            output=f"y({out['t']}) = {out['y']}", engine="ode",
            trust=TrustLevel.NUMERIC_HIGH_PRECISION))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC_HIGH_PRECISION,
                          engine="ode", derivation=[step])

    def ode_ensemble(self, rhs_ids: list[str], t_span: list[str],
                     y0s: list[list[float]], steps: int = 1000,
                     prefer_gpu: bool = True,
                     workers: int | None = None) -> MathResult:
        """Many-trajectory IVP batch: GPU RawKernel (one thread per trajectory)
        when CuPy works, else a process pool of float64 RK4. Numeric trust."""
        from .ode import ensemble_rk4_cpu, ensemble_rk4_gpu
        irs = []
        for rid in rhs_ids:
            ir = self.expressions.get(rid)
            if ir is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"Unknown expr_id: {rid}"], engine="ode")
            irs.append(ir)
        y_vars = [f"y{i}" for i in range(len(irs))]
        if steps < 1 or steps > 10_000_000:
            return MathResult(ok=False, status="error",
                              errors=["steps must be between 1 and 10,000,000"], engine="ode")
        try:
            if prefer_gpu:
                try:
                    out = ensemble_rk4_gpu(irs, y_vars, float(t_span[0]),
                                           float(t_span[1]), y0s, steps)
                except ValueError:
                    sources = [self.expression_sources[r] for r in rhs_ids]
                    out = ensemble_rk4_cpu(sources, y_vars, float(t_span[0]),
                                           float(t_span[1]), y0s, steps,
                                           resolve_workers(workers, cap=self.settings.max_workers))
            else:
                sources = [self.expression_sources[r] for r in rhs_ids]
                out = ensemble_rk4_cpu(sources, y_vars, float(t_span[0]),
                                       float(t_span[1]), y0s, steps,
                                       resolve_workers(workers, cap=self.settings.max_workers))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="ode")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="ode_ensemble",
            inputs=rhs_ids, output=f"{out['trajectories']} trajectories via {out['engine_tier']}",
            engine="ode", trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC,
                          engine="ode", derivation=[step])

    def pde_heat_1d(self, u0: list[float], alpha: float, dx: float, dt: float,
                    steps: int, prefer_gpu: bool = True) -> MathResult:
        """1D heat equation u_t = alpha u_xx via explicit FTCS finite
        differences (numeric evidence only). Stability r <= 1/2 is enforced."""
        from .ode import heat_ftcs
        try:
            out = heat_ftcs(u0, alpha, dx, dt, steps, prefer_gpu=prefer_gpu)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="ode")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="pde_heat_1d",
            inputs=[f"grid={len(u0)}", f"steps={steps}"],
            output=f"tier={out['engine_tier']}, r={out['stability_r']:.4f}",
            engine="ode", trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC,
                          engine="ode", derivation=[step])

    def _pde_guard(self, cells: int, steps: int) -> MathResult | None:
        if cells > self.settings.max_pde_grid:
            return MathResult(ok=False, status="error", engine="pde",
                errors=[f"grid has {cells} cells; limit is {self.settings.max_pde_grid}"])
        if steps > self.settings.max_ode_steps:
            return MathResult(ok=False, status="error", engine="pde",
                errors=[f"steps={steps} exceeds max_ode_steps={self.settings.max_ode_steps}"])
        return None

    def _pde_result(self, operation: str, out: dict, detail: str) -> MathResult:
        step = self._record(DerivationStep(step_id=self._id("step"), operation=operation,
            inputs=[detail], output=f"tier={out['engine_tier']}",
            engine="pde", trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC,
                          engine="pde", derivation=[step])

    def pde_heat_2d(self, u0: list[list[float]], alpha: float, dx: float,
                    dt: float, steps: int, prefer_gpu: bool = True) -> MathResult:
        """2D heat equation u_t = alpha*(u_xx + u_yy), FTCS, Dirichlet-zero
        boundaries. Stability r <= 1/4 enforced. Numeric evidence only."""
        from .pde import heat_2d
        guard = self._pde_guard(len(u0) * len(u0[0]), steps)
        if guard: return guard
        try:
            out = heat_2d(u0, alpha, dx, dt, steps, prefer_gpu)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="pde")
        return self._pde_result("pde_heat_2d", out,
                                f"grid={len(u0)}x{len(u0[0])}, steps={steps}")

    def pde_wave_1d(self, u0: list[float], v0: list[float], c: float,
                   dx: float, dt: float, steps: int,
                   prefer_gpu: bool = True) -> MathResult:
        """1D wave equation u_tt = c^2 u_xx, leapfrog. CFL c*dt/dx <= 1
        enforced. Numeric evidence only."""
        from .pde import wave_1d
        guard = self._pde_guard(len(u0), steps)
        if guard: return guard
        try:
            out = wave_1d(u0, v0, c, dx, dt, steps, prefer_gpu)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="pde")
        return self._pde_result("pde_wave_1d", out, f"n={len(u0)}, steps={steps}")

    def pde_advect_1d(self, u0: list[float], c: float, dx: float, dt: float,
                      steps: int, prefer_gpu: bool = True) -> MathResult:
        """1D advection u_t + c u_x = 0, first-order upwind (c >= 0).
        CFL enforced. First-order upwind is numerically diffusive."""
        from .pde import advect_1d
        guard = self._pde_guard(len(u0), steps)
        if guard: return guard
        try:
            out = advect_1d(u0, c, dx, dt, steps, prefer_gpu)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="pde")
        return self._pde_result("pde_advect_1d", out, f"n={len(u0)}, steps={steps}")

    def pde_ensemble(self, u0: list[list[float]], alphas: list[float],
                     dx: float, dt: float, steps: int,
                     prefer_gpu: bool = True,
                     workers: int | None = None) -> MathResult:
        """2D heat parameter sweep over diffusivities: one batched CuPy
        stencil on GPU, process pool on CPU. Numeric evidence only."""
        from .pde import heat_2d_ensemble
        guard = self._pde_guard(len(u0) * len(u0[0]) * len(alphas), steps)
        if guard: return guard
        try:
            out = heat_2d_ensemble(u0, alphas, dx, dt, steps, prefer_gpu, workers)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="pde")
        return self._pde_result("pde_ensemble", out,
                                f"grid={len(u0)}x{len(u0[0])}, batch={len(alphas)}")

    def pde_mol_heat(self, u0: list[float], alpha: float, dx: float,
                     t1: float, steps: int = 1000) -> MathResult:
        """Method of lines: semidiscretized heat equation integrated by the
        existing RK4 float64 solver. Numeric evidence only."""
        from .pde import mol_heat_1d
        guard = self._pde_guard(len(u0), steps)
        if guard: return guard
        out = mol_heat_1d(u0, alpha, dx, t1, steps)
        return self._pde_result("pde_mol_heat", out, f"n={len(u0)}, t1={t1}")

    def optimize_critical_points(self, expr_id: str, variables: list[str],
                                 context_id: str | None = None) -> MathResult:
        """Symbolic critical points: solves grad f = 0. Trust follows the solve
        evidence (symbolic unless independently verified)."""
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="optimize")
        try:
            env = self._symbol_env(self.contexts.get(context_id) if context_id else None, ir)
            f = self.sympy.to_sympy(ir, env)
            grads = [sp.diff(f, sp.Symbol(v)) for v in variables]
            solutions = run_with_timeout(sp.solve, self.settings.solver_timeout_seconds,
                                         grads, [sp.Symbol(v) for v in variables],
                                         dict=True)
        except (ValueError, TypeError, NotImplementedError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="optimize")
        points = [{v: str(sol[sp.Symbol(v)]) for v in variables if sp.Symbol(v) in sol}
                  for sol in solutions]
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="optimize_critical_points", inputs=[expr_id],
            output=f"{len(points)} point(s)", engine="optimize", trust=TrustLevel.SYMBOLIC))
        return MathResult(ok=True, data={"critical_points": points,
            "count": len(points)}, trust=TrustLevel.SYMBOLIC, engine="optimize",
            derivation=[step])

    def optimize_kkt(self, objective_id: str, constraint_ids: list[str],
                     variables: list[str], context_id: str | None = None) -> MathResult:
        """KKT conditions for min f s.t. g_i(x) <= 0, returned symbolically:
        stationarity, primal/dual feasibility, complementary slackness."""
        ir = self.expressions.get(objective_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {objective_id}"], engine="optimize")
        constraints = []
        for cid in constraint_ids:
            cir = self.expressions.get(cid)
            if cir is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"Unknown expr_id: {cid}"], engine="optimize")
            constraints.append(cir)
        try:
            env = self._symbol_env(self.contexts.get(context_id) if context_id else None,
                                   ir, *constraints)
            f = self.sympy.to_sympy(ir, env)
            gs = [self.sympy.to_sympy(c, env) for c in constraints]
            lambdas = sp.symbols(f"lambda0:{len(gs)}")
            lagrangian = f + sum(l * g for l, g in zip(lambdas, gs))
            stationarity = [str(sp.Eq(sp.diff(lagrangian, sp.Symbol(v)), 0))
                            for v in variables]
            conditions = {
                "lagrangian": str(lagrangian),
                "stationarity": stationarity,
                "primal_feasibility": [f"({g}) <= 0" for g in gs],
                "dual_feasibility": [f"lambda{i} >= 0" for i in range(len(gs))],
                "complementary_slackness": [f"lambda{i} * ({g}) = 0"
                                            for i, g in enumerate(gs)],
            }
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="optimize")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="optimize_kkt",
            inputs=[objective_id, *constraint_ids], output="KKT conditions",
            engine="optimize", trust=TrustLevel.SYMBOLIC))
        return MathResult(ok=True, data=conditions, trust=TrustLevel.SYMBOLIC,
                          engine="optimize", derivation=[step])

    def lp_solve(self, c: list[str], a: list[list[str]], b: list[str],
                 exact: bool = True) -> MathResult:
        """LP: max c^T x s.t. A x <= b, x >= 0, b >= 0. exact=True: rational
        simplex (exact). exact=False: njit/float64 tableau (numeric)."""
        from .optimize import simplex_exact, simplex_numeric
        try:
            if exact:
                result = run_with_timeout(simplex_exact,
                    self.settings.solver_timeout_seconds,
                    [self._frac_or_error(v) for v in c],
                    [[self._frac_or_error(v) for v in row] for row in a],
                    [self._frac_or_error(v) for v in b])
                result = {**result,
                          "x": [str(v) for v in result.get("x", [])],
                          "objective": str(result["objective"]) if "objective" in result else None}
                tier, trust = "exact", TrustLevel.EXACT
            else:
                raw, tier = simplex_numeric(c, a, b)
                result = raw
                trust = TrustLevel.NUMERIC
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="optimize")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="lp_solve",
            inputs=[f"{len(c)} vars, {len(a)} constraints"],
            output=str(result.get("status")), engine="optimize", trust=trust))
        return MathResult(ok=True, data={**result, "engine_tier": tier},
                          trust=trust, engine="optimize", derivation=[step])

    def optimize_minimize(self, expr_id: str, variables: list[str],
                          start: list[float], tol: float | None = None,
                          max_iter: int | None = None) -> MathResult:
        """Local minimization via Nelder-Mead (float64, numeric trust)."""
        from .numerics import compile_float64
        from .optimize import nelder_mead
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="optimize")
        tol = tol if tol is not None else self.settings.tolerance
        max_iter = max_iter or self.settings.max_iterations
        try:
            f = compile_float64(ir, variables)
            result = nelder_mead(lambda point: f(*point), list(start), tol, max_iter)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="optimize")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="optimize_minimize", inputs=[expr_id, f"start={start}"],
            output=f"f={result['f']:.6g}", engine="optimize", trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data=result, trust=TrustLevel.NUMERIC,
                          engine="optimize", derivation=[step])

    def optimize_multistart(self, expr_id: str, variables: list[str],
                            starts: list[list[float]], workers: int | None = None) -> MathResult:
        """Multi-start Nelder-Mead across the process pool (numeric trust)."""
        from .optimize import multistart_minimize
        source = self.expression_sources.get(expr_id)
        if source is None:
            return MathResult(ok=False, status="error",
                              errors=[f"Unknown expr_id: {expr_id}"], engine="optimize")
        if len(starts) > self.settings.max_batch_jobs:
            return MathResult(ok=False, status="error",
                errors=[f"{len(starts)} starts exceeds limit {self.settings.max_batch_jobs}"],
                engine="optimize")
        try:
            resolved = resolve_workers(workers, cap=self.settings.max_workers)
            out = multistart_minimize(source, variables, starts,
                                      self.settings.tolerance,
                                      self.settings.max_iterations, resolved)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="optimize")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="optimize_multistart", inputs=[expr_id, f"{len(starts)} starts"],
            output=f"best f={out['best']['f']:.6g}", engine="optimize",
            trust=TrustLevel.NUMERIC))
        return MathResult(ok=True, data=out, trust=TrustLevel.NUMERIC,
                          engine="optimize", derivation=[step])

    # ------------------------------------------------------------------
    # Exact transforms
    # ------------------------------------------------------------------

    def fwht(self, values: list[str]) -> MathResult:
        from .transforms import fwht as _fwht
        if not values:
            return MathResult(ok=False, status="error", errors=["values must not be empty"],
                              engine="transforms")
        if len(values) > self.settings.max_fwht_size:
            return MathResult(ok=False, status="error",
                errors=[f"fwht length {len(values)} exceeds limit {self.settings.max_fwht_size} "
                        "(MATHKERNEL_MAX_FWHT_SIZE)"], engine="transforms")
        try:
            nums = [decimal_to_int(v) for v in values]
            out = _fwht(nums)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="transforms")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="fwht",
            inputs=[f"{len(nums)} values"], output=f"{len(out)} Walsh coefficients",
            engine="transforms", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"values": [str(v) for v in out], "length": len(out),
            "variables": len(out).bit_length() - 1}, trust=TrustLevel.EXACT,
            engine="transforms", derivation=[step])

    # ------------------------------------------------------------------
    # Binary fields GF(2^m)
    # ------------------------------------------------------------------

    def _get_gf2m(self, field_id: str) -> tuple[object | None, MathResult | None]:
        entry = self.gf2m_fields.get(field_id)
        if entry is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"Unknown field_id: {field_id}"], engine="gf2m")
        return entry, None

    @staticmethod
    def _parse_hex(value: str, m: int) -> int:
        try:
            out = int(value, 16)
        except ValueError as exc:
            raise ValueError(f"Expected a hex string, got {value!r}") from exc
        if out >= (1 << m):
            raise ValueError(f"Value {value!r} does not fit in {m} bits")
        return out

    def gf2m_create(self, degree: int, reduction: str) -> MathResult:
        from .gf2m import GF2mField
        try:
            red = self._parse_hex(reduction, degree)
            field = GF2mField(degree, red)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        if not field.verify_irreducible():
            return MathResult(ok=False, status="error",
                errors=[f"x^{degree} + {red:0{degree // 4}x} is reducible over GF(2); "
                        "the modulus does not define a field"], engine="gf2m")
        fid = self._id("gf2m")
        self.gf2m_fields[fid] = field
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m_create",
            inputs=[f"degree={degree}", f"reduction={reduction}"],
            output=f"GF(2^{degree}), irreducibility verified (Rabin)", engine="gf2m",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"field_id": fid, "degree": degree,
            "reduction": f"{red:0{degree // 4}x}", "irreducible": True,
            "basis": "polynomial"}, trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2m_from_transition(self, columns: list[str]) -> MathResult:
        from .gf2m import TransitionField
        if not columns:
            return MathResult(ok=False, status="error", errors=["columns must not be empty"], engine="gf2m")
        try:
            cols = [self._parse_hex(c, len(columns)) for c in columns]
            tf = TransitionField(cols)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        fid = self._id("gf2m")
        self.gf2m_fields[fid] = tf
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m_from_transition",
            inputs=[f"{len(columns)} columns"],
            output=f"GF(2^{tf.m}), reduction x^{tf.m} + {tf.red:0{tf.m // 4}x}",
            engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"field_id": fid, "degree": tf.m,
            "reduction": f"{tf.red:0{tf.m // 4}x}", "irreducible": True,
            "basis": "transition_dual_orbit"}, trust=TrustLevel.EXACT, engine="gf2m",
            derivation=[step])

    def gf2m_compute(self, field_id: str, operation: str, values: list[str],
                     exponent: str | None = None) -> MathResult:
        entry, err = self._get_gf2m(field_id)
        if err: return err
        field = entry.field if hasattr(entry, "field") else entry
        m = field.m
        try:
            vals = [self._parse_hex(v, m) for v in values]
            if operation in {"add", "sub", "mul", "div"}:
                if len(vals) != 2: raise ValueError(f"{operation} expects exactly 2 values")
                out = {"add": field.add, "sub": field.add, "mul": field.mul,
                       "div": field.div}[operation](vals[0], vals[1])
                result = f"{out:0{m // 4}x}"
            elif operation == "pow":
                if len(vals) != 1 or exponent is None:
                    raise ValueError("pow expects 1 value and an exponent")
                result = f"{field.power(vals[0], int(exponent, 16)):0{m // 4}x}"
            elif operation in {"inv", "sqrt", "trace"}:
                if len(vals) != 1: raise ValueError(f"{operation} expects exactly 1 value")
                out = {"inv": field.inv, "sqrt": field.sqrt, "trace": field.trace}[operation](vals[0])
                result = f"{out:0{m // 4}x}" if operation != "trace" else str(out)
            elif operation == "quadratic_roots":
                if len(vals) != 3: raise ValueError("quadratic_roots expects [c0, c1, c2]")
                roots = field.quadratic_roots(*vals)
                step = self._record(DerivationStep(step_id=self._id("step"),
                    operation="gf2m:quadratic_roots", inputs=values,
                    output=f"{len(roots)} root(s)", engine="gf2m", trust=TrustLevel.EXACT))
                return MathResult(ok=True, data={"roots": [f"{r:0{m // 4}x}" for r in roots],
                    "root_count": len(roots)}, trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])
            else:
                raise ValueError(f"Unsupported gf2m operation: {operation}; choose from add, sub, "
                                 "mul, div, pow, inv, sqrt, trace, quadratic_roots")
        except (ValueError, ZeroDivisionError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation=f"gf2m:{operation}",
            inputs=values, output=result, engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"result": result}, trust=TrustLevel.EXACT,
                          engine="gf2m", derivation=[step])

    def gf2m_coords(self, field_id: str, row: str) -> MathResult:
        entry, err = self._get_gf2m(field_id)
        if err: return err
        if not hasattr(entry, "coords"):
            return MathResult(ok=False, status="error",
                errors=["coords requires a transition-derived field (gf2m_from_transition)"], engine="gf2m")
        try:
            value = self._parse_hex(row, entry.m)
            coords = entry.coords(value)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m:coords",
            inputs=[row], output=f"{coords:0{entry.m // 4}x}", engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"coords": f"{coords:0{entry.m // 4}x}"},
                          trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2m_root_jump_rows(self, field_id: str, root: str) -> MathResult:
        entry, err = self._get_gf2m(field_id)
        if err: return err
        if not hasattr(entry, "root_jump_rows"):
            return MathResult(ok=False, status="error",
                errors=["root_jump_rows requires a transition-derived field"], engine="gf2m")
        try:
            r = self._parse_hex(root, entry.m)
            rows = entry.root_jump_rows(r)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m:root_jump_rows",
            inputs=[root], output=f"{len(rows)} rows", engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"rows": [f"{x:0{entry.m // 4}x}" for x in rows]},
                          trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2m_closure_roots(self, field_id: str, row_sets: list[list[str]]) -> MathResult:
        entry, err = self._get_gf2m(field_id)
        if err: return err
        if not hasattr(entry, "closure_roots"):
            return MathResult(ok=False, status="error",
                errors=["closure_roots requires a transition-derived field"], engine="gf2m")
        try:
            parsed = [[self._parse_hex(r, entry.m) for r in rows] for rows in row_sets]
            for rows in parsed:
                if len(rows) not in {2, 3}:
                    raise ValueError("each row set must contain 2 (linear) or 3 (quadratic) rows")
            results = entry.closure_roots(parsed)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        m4 = entry.m // 4
        candidates = [{"coefficients": [f"{c:0{m4}x}" for c in r["coefficients"]],
                       "roots": [f"{x:0{m4}x}" for x in r["roots"]],
                       "residuals": [f"{x:0{m4}x}" for x in r["residuals"]],
                       "closure_valid": all(x == 0 for x in r["residuals"])}
                      for r in results]
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m:closure_roots",
            inputs=[f"{len(row_sets)} row sets"],
            output=f"{sum(len(c['roots']) for c in candidates)} roots, "
                   f"{sum(1 for c in candidates if c['closure_valid'])} valid closures",
            engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"candidates": candidates,
            "all_closures_valid": all(c["closure_valid"] for c in candidates)},
            trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2m_jump_rows(self, field_id: str, k: str) -> MathResult:
        entry, err = self._get_gf2m(field_id)
        if err: return err
        if not hasattr(entry, "jump_rows"):
            return MathResult(ok=False, status="error",
                errors=["jump_rows requires a transition-derived field"], engine="gf2m")
        try:
            steps = int(k, 0) if isinstance(k, str) else int(k)
            if steps < 0: raise ValueError("k must be non-negative")
            rows = entry.jump_rows(steps)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2m:jump_rows",
            inputs=[f"k={steps}"], output=f"{len(rows)} rows (alpha^{steps} action)",
            engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"rows": [f"{x:0{entry.m // 4}x}" for x in rows],
            "k": str(steps)}, trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2_rank(self, rows: list[str], width: int) -> MathResult:
        from .gf2m import gf2_rank as _rank
        try:
            parsed = [self._parse_hex(r, width) for r in rows]
            rank, nullity = _rank(parsed, width)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2:rank",
            inputs=[f"{len(rows)} rows", f"width={width}"],
            output=f"rank={rank}, nullity={nullity}", engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"rank": rank, "nullity": nullity, "width": width},
                          trust=TrustLevel.EXACT, engine="gf2m", derivation=[step])

    def gf2_nullspace(self, rows: list[str], width: int, side: str = "right") -> MathResult:
        from .gf2m import gf2_left_nullspace, gf2_nullspace
        try:
            parsed = [self._parse_hex(r, width) for r in rows]
            if side == "right":
                basis = gf2_nullspace(parsed, width)
            elif side == "left":
                basis = gf2_left_nullspace(parsed, width)
            else:
                raise ValueError("side must be 'right' or 'left'")
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation=f"gf2:nullspace:{side}", inputs=[f"{len(rows)} rows", f"width={width}"],
            output=f"dimension {len(basis)}", engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"basis": [f"{v:0{width // 4}x}" for v in basis],
            "dimension": len(basis), "side": side}, trust=TrustLevel.EXACT,
            engine="gf2m", derivation=[step])

    def gf2_carryfree_cols(self, c: str, width: int) -> MathResult:
        from .gf2m import carryfree_cols
        try:
            value = int(c, 0) if isinstance(c, str) else int(c)
            if value <= 0: raise ValueError("c must be positive")
            cols = carryfree_cols(value, width)
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2:carryfree_cols",
            inputs=[f"c={value}", f"width={width}"], output=f"{len(cols)} columns",
            engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"columns": [f"{x:0{width // 4}x}" for x in cols],
            "c": str(value), "width": width}, trust=TrustLevel.EXACT, engine="gf2m",
            derivation=[step])

    def gf2_minpoly(self, bits_hex: str, nbits: int, verify_from: int | None = None) -> MathResult:
        from .gf2m import gf2_berlekamp_massey, gf2_sequence_satisfies
        try:
            if nbits < 1 or nbits > 4_000_000:
                raise ValueError("nbits must be between 1 and 4,000,000")
            bits = int(bits_hex, 16)
            if bits >> nbits:
                raise ValueError("bits_hex exceeds nbits")
            conn = gf2_berlekamp_massey(bits, nbits)
            deg = conn.bit_length() - 1
            data: dict = {
                "connection_polynomial": f"{conn:x}",
                "degree": deg,
                "weight": conn.bit_count(),
            }
            if verify_from is not None and verify_from < nbits:
                data["violations"] = gf2_sequence_satisfies(bits, nbits, conn, start=verify_from)
                data["verify_from"] = verify_from
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="gf2m")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="gf2:minpoly",
            inputs=[f"nbits={nbits}"], output=f"degree={deg}, weight={conn.bit_count()}",
            engine="gf2m", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data=data, trust=TrustLevel.EXACT, engine="gf2m",
                          derivation=[step])

    # ------------------------------------------------------------------
    # Finite dynamical systems: cumulants, Koopman transport, Fourier
    # transfer, closure-relation search
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_scalar(v):
        """JSON scalar -> exact value: "p/q" | int | {"re":..,"im":..} | {"cyclotomic":..}."""
        from fractions import Fraction
        from .finite_fourier import CyclotomicNumber
        if isinstance(v, (int, Fraction)):
            return Fraction(v)
        if isinstance(v, str):
            try:
                return Fraction(v)
            except ValueError as exc:
                raise ValueError(f"expected an exact rational string, got {v!r}") from exc
        if isinstance(v, dict):
            if "cyclotomic" in v:
                spec = v["cyclotomic"]
                return CyclotomicNumber(int(spec["order"]),
                                        {int(e): Fraction(c) for e, c in spec["terms"].items()})
            if "re" in v or "im" in v:
                return complex(float(v.get("re", 0)), float(v.get("im", 0)))
        raise ValueError(f"unsupported scalar encoding: {v!r}")

    @staticmethod
    def _encode_scalar(v):
        from fractions import Fraction
        from .finite_fourier import CyclotomicNumber
        if isinstance(v, CyclotomicNumber):
            return {"cyclotomic": {"order": v.L,
                                   "terms": {str(e): f"{c.numerator}/{c.denominator}"
                                             for e, c in sorted(v.coeffs.items())}}}
        if isinstance(v, Fraction):
            return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, complex):
            return {"re": repr(v.real), "im": repr(v.imag)}
        return str(v)

    @staticmethod
    def _mask_key(mask: int) -> str:
        return ",".join(str(j) for j in range(mask.bit_length()) if mask >> j & 1)

    @staticmethod
    def _key_mask(key: str) -> int:
        mask = 0
        for part in str(key).split(","):
            part = part.strip()
            if part:
                mask |= 1 << int(part)
        return mask

    def cumulant_compute(self, op: str, order: int, values: dict | None = None,
                         columns: list | None = None) -> MathResult:
        from . import cumulants as cum
        try:
            if order < 1 or order > self.settings.max_cumulant_order:
                raise ValueError(f"order must be 1..{self.settings.max_cumulant_order}")
            if op == "cumulant":
                if values is None:
                    raise ValueError("values (subset moments) required")
                moments = {self._key_mask(k): self._decode_scalar(v) for k, v in values.items()}
                out = cum.cumulant_from_moments(moments, order)
            elif op == "moment":
                if values is None:
                    raise ValueError("values (subset cumulants) required")
                cums = {self._key_mask(k): self._decode_scalar(v) for k, v in values.items()}
                out = cum.moment_from_cumulants(cums, order)
            elif op == "samples":
                if not columns:
                    raise ValueError("columns (sample vectors) required")
                cols = [[self._decode_scalar(v) for v in col] for col in columns]
                if len(cols) != order:
                    raise ValueError("number of columns must equal order")
                out = cum.joint_cumulant_from_samples(cols)
            else:
                raise ValueError("op must be 'cumulant', 'moment', or 'samples'")
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="cumulants")
        step = self._record(DerivationStep(step_id=self._id("step"), operation=f"cumulant:{op}",
            inputs=[f"order={order}"], output=str(self._encode_scalar(out)), engine="cumulants",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"op": op, "order": order,
                          "result": self._encode_scalar(out),
                          "partitions": len(cum.set_partitions(order))},
                          trust=TrustLevel.EXACT, engine="cumulants", derivation=[step])

    def finite_system_create(self, mu, transition: list[int],
                             observation: list[int] | None = None) -> MathResult:
        from fractions import Fraction
        from .koopman import FiniteSystem
        try:
            if isinstance(mu, str) and mu == "uniform":
                n = len(transition)
                mu_v = [Fraction(1, n)] * n
            else:
                mu_v = [self._decode_scalar(v) for v in mu]
            if len(mu_v) > self.settings.max_finite_states:
                raise ValueError(f"state count exceeds limit {self.settings.max_finite_states}")
            system = FiniteSystem(mu_v, list(transition), observation)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        sid = self._id("fsys")
        self.finite_systems[sid] = system
        step = self._record(DerivationStep(step_id=self._id("step"), operation="finite_system:create",
            inputs=[f"{system.n} states", f"{system.n_outputs} outputs"],
            output=sid, engine="koopman", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"system_id": sid, "states": system.n,
                          "outputs": system.n_outputs, "support": len(system.support),
                          "stationary": system.is_stationary()},
                          trust=TrustLevel.EXACT, engine="koopman", derivation=[step])

    def _get_system(self, system_id: str):
        system = self.finite_systems.get(system_id)
        if system is None:
            return None, MathResult(ok=False, status="error",
                                    errors=[f"unknown system_id {system_id!r}"], engine="koopman")
        return system, None

    def _build_basis(self, spec: dict):
        from .koopman import cyclic_basis, walsh_basis
        kind = spec.get("kind")
        if kind == "walsh":
            return walsh_basis(int(spec["r"]))
        if kind == "cyclic":
            return cyclic_basis(int(spec["m"]))
        raise ValueError("basis kind must be 'walsh' or 'cyclic'")

    def _koopman_common(self, system_id: str, basis_spec: dict):
        system, err = self._get_system(system_id)
        if err:
            return None, None, None, err
        basis, norm_sq = self._build_basis(basis_spec)
        if len(basis) != system.n:
            return None, None, None, MathResult(ok=False, status="error",
                errors=[f"basis has {len(basis)} modes but the system has {system.n} states"],
                engine="koopman")
        return system, basis, norm_sq, None

    _numeric_device_cache: str | None = None

    @classmethod
    def _numeric_device(cls) -> str:
        if cls._numeric_device_cache is None:
            from .koopman import _xp
            cls._numeric_device_cache = (
                "gpu" if _xp(True).__name__ == "cupy" else "cpu")
        return cls._numeric_device_cache

    @staticmethod
    def _relations_njit() -> bool:
        from .relations import HAVE_NUMBA
        return HAVE_NUMBA

    def koopman_matrix(self, system_id: str, basis_spec: dict,
                       exact: bool = True) -> MathResult:
        from .koopman import koopman_matrix, koopman_matrix_numeric
        try:
            system, basis, norm_sq, err = self._koopman_common(system_id, basis_spec)
            if err:
                return err
            if exact:
                Q = koopman_matrix(system, basis, norm_sq)
            else:
                Q = koopman_matrix_numeric(system, basis, norm_sq)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        step = self._record(DerivationStep(step_id=self._id("step"), operation="koopman:matrix",
            inputs=[system_id], output=f"{len(Q)}x{len(Q)}", engine=engine, trust=trust))
        return MathResult(ok=True, data={"matrix": [[self._encode_scalar(v) for v in row]
                          for row in Q], "modes": len(Q), "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def koopman_transfer(self, system_id: str, output_functions: list, basis_spec: dict,
                         exact: bool = True) -> MathResult:
        from .koopman import transfer_matrix, transfer_matrix_numeric
        try:
            system, basis, norm_sq, err = self._koopman_common(system_id, basis_spec)
            if err:
                return err
            funcs = [[self._decode_scalar(v) for v in phi] for phi in output_functions]
            if exact:
                C = transfer_matrix(system, funcs, basis, norm_sq)
            else:
                C = transfer_matrix_numeric(system, funcs, basis, norm_sq)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        step = self._record(DerivationStep(step_id=self._id("step"), operation="koopman:transfer",
            inputs=[system_id, f"{len(output_functions)} output functions"],
            output=f"{len(C)}x{len(C[0]) if C else 0}", engine=engine, trust=trust))
        return MathResult(ok=True, data={"matrix": [[self._encode_scalar(v) for v in row]
                          for row in C], "functions": len(C), "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def koopman_visibility(self, system_id: str, basis_spec: dict,
                           exact: bool = True) -> MathResult:
        from .koopman import mode_visibility, mode_visibility_numeric
        try:
            system, basis, norm_sq, err = self._koopman_common(system_id, basis_spec)
            if err:
                return err
            if exact:
                vis = mode_visibility(system, basis, norm_sq)
            else:
                vis = mode_visibility_numeric(system, basis, norm_sq)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        tol = 0 if exact else 1e-9
        invisible = [a for a, v in enumerate(vis) if abs(v) <= tol]
        full = [a for a, v in enumerate(vis) if abs(v - 1) <= tol]
        step = self._record(DerivationStep(step_id=self._id("step"), operation="koopman:visibility",
            inputs=[system_id], output=f"{len(invisible)} invisible, {len(full)} fully visible",
            engine=engine, trust=trust))
        return MathResult(ok=True, data={"visibility": [self._encode_scalar(v) for v in vis],
                          "invisible_modes": invisible, "fully_visible_modes": full,
                          "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def koopman_lagged(self, system_id: str, basis_spec: dict, tau: list[int],
                       alphas: list[int], connected: bool = True,
                       exact: bool = True) -> MathResult:
        from .koopman import lagged_tensor_numeric, lagged_tensor_value
        try:
            system, basis, norm_sq, err = self._koopman_common(system_id, basis_spec)
            if err:
                return err
            if len(tau) > self.settings.max_cumulant_order:
                raise ValueError(f"order exceeds limit {self.settings.max_cumulant_order}")
            if exact:
                value = lagged_tensor_value(system, basis, list(tau), list(alphas), connected)
            else:
                value = lagged_tensor_numeric(system, basis, list(tau), list(alphas), connected)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        kind = "connected" if connected else "raw"
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation=f"koopman:lagged:{kind}", inputs=[system_id, f"tau={tau}"],
            output=str(self._encode_scalar(value)), engine=engine, trust=trust))
        return MathResult(ok=True, data={"value": self._encode_scalar(value), "tau": tau,
                          "alphas": alphas, "connected": connected, "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def koopman_observed(self, system_id: str, output_functions: list, h_tuple: list[int],
                         tau: list[int], connected: bool = True,
                         exact: bool = True) -> MathResult:
        from .koopman import observed_statistic, observed_statistic_numeric
        try:
            system, err = self._get_system(system_id)
            if err:
                return err
            if len(tau) > self.settings.max_cumulant_order:
                raise ValueError(f"order exceeds limit {self.settings.max_cumulant_order}")
            funcs = [[self._decode_scalar(v) for v in phi] for phi in output_functions]
            if exact:
                value = observed_statistic(system, funcs, list(h_tuple), list(tau), connected)
            else:
                value = observed_statistic_numeric(system, funcs, list(h_tuple),
                                                   list(tau), connected)
        except (ValueError, TypeError, IndexError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        kind = "connected" if connected else "raw"
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation=f"koopman:observed:{kind}",
            inputs=[system_id, f"h={h_tuple}", f"tau={tau}"],
            output=str(self._encode_scalar(value)), engine=engine, trust=trust))
        return MathResult(ok=True, data={"value": self._encode_scalar(value),
                          "h_tuple": h_tuple, "tau": tau, "connected": connected,
                          "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def koopman_diagnostics(self, system_id: str, basis_spec: dict,
                            exact: bool = True) -> MathResult:
        from .koopman import (koopman_matrix, koopman_matrix_numeric,
                              transport_diagnostics, transport_diagnostics_numeric)
        try:
            system, basis, norm_sq, err = self._koopman_common(system_id, basis_spec)
            if err:
                return err
            if exact:
                diag = transport_diagnostics(koopman_matrix(system, basis, norm_sq))
            else:
                diag = transport_diagnostics_numeric(
                    koopman_matrix_numeric(system, basis, norm_sq))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="koopman")
        trust = TrustLevel.EXACT if exact else TrustLevel.NUMERIC
        engine = "koopman" if exact else f"koopman-numeric-{self._numeric_device()}"
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="koopman:diagnostics", inputs=[system_id],
            output=f"{diag['modes']} modes", engine=engine, trust=trust))
        return MathResult(ok=True, data={
                          "ipr": [self._encode_scalar(v) for v in diag["ipr"]],
                          "entropy": diag["entropy"], "modes": diag["modes"],
                          "stationary": system.is_stationary(), "exact": exact},
                          trust=trust, engine=engine, derivation=[step])

    def finite_fourier_compute(self, op: str, **params) -> MathResult:
        from . import finite_fourier as ff
        try:
            if op == "transfer":
                F = [int(v) for v in params["F"]]
                out = ff.transfer_transform(F, int(params["N"]), int(params["h"]))
                data = {"coefficients": [self._encode_scalar(v) for v in out], "M": len(F)}
            elif op == "two_point":
                F = [int(v) for v in params["F"]]
                out = ff.two_point_difference(F, int(params["N"]), int(params["m"]),
                                              int(params["a_k"]), int(params["b_k"]))
                data = {"value": self._encode_scalar(out)}
            elif op == "measure":
                mu = [self._decode_scalar(v) for v in params["mu"]]
                out = ff.measure_fourier(mu)
                data = {"coefficients": [self._encode_scalar(v) for v in out], "M": len(mu)}
            elif op == "dft":
                values = [self._decode_scalar(v) for v in params["values"]]
                out = ff.dft_zm(values)
                data = {"coefficients": [self._encode_scalar(v) for v in out], "M": len(values)}
            elif op == "orbit_correction":
                closed = self._decode_scalar(params["closed_sum"])
                total = self._decode_scalar(params["total_sum"])
                out = ff.orbit_correction(closed, total, int(params["orbit_size"]))
                data = {"value": self._encode_scalar(out)}
            else:
                raise ValueError("op must be 'transfer', 'two_point', 'measure', 'dft', "
                                 "or 'orbit_correction'")
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="finite_fourier")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation=f"finite_fourier:{op}", inputs=[str(sorted(params))],
            engine="finite_fourier", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"op": op, **data}, trust=TrustLevel.EXACT,
                          engine="finite_fourier", derivation=[step])

    def conditioned_access_solve(self, system_id: str, target: list[int]) -> MathResult:
        """Exact least nonnegative state-conditioned lags T^k(x)=target[x]."""
        from .conditioned_dynamics import solve_access_to_target
        system, err = self._get_system(system_id)
        if err: return err
        try:
            access = solve_access_to_target(system.transition, list(target))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="conditioned-dynamics")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:access_solve", inputs=[system_id],
            output=f"{len(access.lags)} exact lags", engine="conditioned-dynamics",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"lags": list(access.lags), "target": list(target)},
                          trust=TrustLevel.EXACT, engine="conditioned-dynamics", derivation=[step])

    def conditioned_symmetry_access(self, system_id: str, transform: list[int]) -> MathResult:
        """Prove O(S(x))=O(x), then solve T^k(x)=S(x) exactly."""
        from .conditioned_dynamics import observable_symmetry, solve_access_to_target
        system, err = self._get_system(system_id)
        if err: return err
        try:
            if not observable_symmetry(system.observation, transform):
                raise ValueError("transform is not a symmetry of the observation")
            access = solve_access_to_target(system.transition, list(transform))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="conditioned-dynamics")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:symmetry_to_access", inputs=[system_id],
            output="proved observation symmetry and orbit access", engine="conditioned-dynamics",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"symmetry": True, "transform": list(transform),
                          "lags": list(access.lags)},
                          trust=TrustLevel.EXACT, engine="conditioned-dynamics", derivation=[step])

    def conditioned_access_compose(self, system_id: str, first: list[int],
                                   second: list[int]) -> MathResult:
        """Compose access maps using k★lambda=k+lambda(T^k(x)x)."""
        from .conditioned_dynamics import AccessMap
        system, err = self._get_system(system_id)
        if err: return err
        try:
            out = AccessMap(tuple(first)).compose(AccessMap(tuple(second)), system.transition)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="conditioned-dynamics")
        return MathResult(ok=True, data={"lags": list(out.lags), "law": "k(x)+lambda(T^k(x)x)"},
                          trust=TrustLevel.EXACT, engine="conditioned-dynamics")

    def conditioned_closure(self, system_id: str, accesses: list[list[int]],
                            coefficients: list[int], modulus: int,
                            constant: int = 0) -> MathResult:
        """Exact exhaustive additive closure over state-conditioned access maps."""
        from .conditioned_dynamics import AccessMap, conditioned_closure
        system, err = self._get_system(system_id)
        if err: return err
        try:
            maps=[AccessMap(tuple(a)) for a in accesses]
            out=conditioned_closure(system.observation, system.transition, maps,
                                    coefficients, int(modulus), int(constant))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="conditioned-dynamics")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:closure", inputs=[system_id, f"order={len(accesses)}"],
            output="exact closure" if out["holds"] else "counterexample found",
            engine="conditioned-dynamics", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data=out, trust=TrustLevel.EXACT,
                          engine="conditioned-dynamics", derivation=[step])

    def symbolic_conditioned_access(self, modulus: int, increment: int,
                                    target_kind: str, target_constant: int,
                                    variable: str = "x") -> MathResult:
        """Derive a compact exact symbolic state-conditioned affine access map."""
        from .conditioned_dynamics import symbolic_affine_access
        try:
            a=symbolic_affine_access(int(modulus),int(increment),
                                     target_kind=target_kind,
                                     target_constant=int(target_constant),
                                     variable=variable)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="conditioned-dynamics")
        return MathResult(ok=True,data={
            "variable":a.variable,"modulus":str(a.modulus),"increment":str(a.increment),
            "target":{"kind":a.target_kind,"constant":str(a.target_constant)},
            "expression":a.expression},
            trust=TrustLevel.EXACT,engine="conditioned-dynamics")

    def synthesize_gf2_vector_conditioned_access(self, outputs: list[dict],
                                                 word_bits: int, constants: list[int],
                                                 columns: list[int], state_words: list[int],
                                                 max_lag: int, probe_samples: int=2048,
                                                 top_k: int=64) -> MathResult:
        """Synthesize two-word output symmetries and solve their GF(2) orbit access."""
        from .structural_discovery import Var,Const,Op,synthesize_vector_symmetries,apply_vector_transform
        from .gf2_conditioned import solve_orbit_access_bounded
        def cv(n):
            op=n["op"]
            if op=="var": return Var(n["name"])
            if op=="const": return Const(int(n["value"]))
            return Op(op,*(cv(a) for a in n.get("args",[])))
        try:
            exprs=[cv(e) for e in outputs]
            d=synthesize_vector_symmetries(exprs,int(word_bits),[int(c) for c in constants],
                                           int(probe_samples),16,int(top_k))
            mask=(1<<int(word_bits))-1
            sw=tuple(int(x)&mask for x in state_words)
            if len(sw)!=2: raise ValueError("state_words must contain exactly two words")
            packed=sw[0]|(sw[1]<<int(word_bits))
            accesses=[]
            # At full widths, ranking is numeric; every orbit access found is exactly
            # verified, but output symmetry remains a candidate unless separately proved.
            for sym in d["ranked"]:
                if sym["score"]<1.0: continue
                tw=apply_vector_transform(sym["transform"],sw,int(word_bits))
                target=tw[0]|(tw[1]<<int(word_bits))
                acc=solve_orbit_access_bounded([int(c) for c in columns],packed,target,int(max_lag))
                if acc:
                    accesses.append({"symmetry":sym["name"],"symmetry_score":sym["score"],
                        "target_words":[str(tw[0]),str(tw[1])],"lag":str(acc.lag),
                        "orbit_access_trust":"exact",
                        "symmetry_trust":"exact" if any(e["name"]==sym["name"] for e in d["exact_symmetries"]) else "numeric"})
            out={"candidate_count":d["candidate_count"],"ranked_candidates":d["ranked"],
                 "exact_symmetries":d["exact_symmetries"],"conditioned_accesses":accesses}
        except (ValueError,TypeError,KeyError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="gf2-vector-synthesis")
        return MathResult(ok=True,data=out,trust=TrustLevel.NUMERIC,engine="gf2-vector-synthesis")

    def gf2_predictive_closure(self, columns: list[int], lag: int,
                               sparse_term_limit: int=16) -> MathResult:
        """Derive an exact GF(2) giant-lag predictive closure."""
        from .gf2_conditioned import (gf2_predictive_closure,
            gf2_jump_polynomial_support, verify_sparse_predictive_closure)
        try:
            c=gf2_predictive_closure([int(x) for x in columns],int(lag))
            support=gf2_jump_polynomial_support([int(x) for x in columns],int(lag))
            sparse=len(support)<=int(sparse_term_limit)
            verified=verify_sparse_predictive_closure(
                [int(x) for x in columns],int(lag),support) if sparse else False
            data={"lag":str(c.lag),"relation":c.relation,
                  "jump_columns":[str(x) for x in c.jump_columns],
                  "polynomial_support":[str(x) for x in support],
                  "sparse":sparse,"sparse_identity_verified":verified,
                  "closure_kind":"predictive"}
        except (ValueError,TypeError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],
                              engine="gf2-predictive-closure")
        return MathResult(ok=True,data=data,trust=TrustLevel.EXACT,
                          engine="gf2-predictive-closure")

    def gf2_conditioned_access(self, columns: list[int], state: int, target: int,
                               max_lag: int) -> MathResult:
        """Exactly solve T^k(state)=target for a bounded lag under cyclic GF(2) dynamics."""
        from .gf2_conditioned import solve_orbit_access_bounded
        try:
            r=solve_orbit_access_bounded([int(c) for c in columns],int(state),
                                         int(target),int(max_lag))
        except (ValueError,TypeError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="gf2-conditioned")
        if r is None:
            return MathResult(ok=True,data={"found":False,"max_lag":str(max_lag)},
                              trust=TrustLevel.EXACT,engine="gf2-conditioned")
        return MathResult(ok=True,data={"found":True,"lag":str(r.lag),
                          "state":str(r.state),"target":str(r.target),
                          "degree":r.degree,"verified":r.verified},
                          trust=TrustLevel.EXACT,engine="gf2-conditioned")

    def synthesize_conditioned_closures(self, expression: dict, word_bits: int,
                                        increment: int, constants: list[int],
                                        variable: str="x", max_depth: int=2,
                                        probe_samples: int=1024,
                                        proof_top_k: int=64) -> MathResult:
        """Synthesize candidate word symmetries, rank numerically, prove exact closures."""
        from .structural_discovery import Var, Const, Op, synthesize_symmetries, classify_simple_transform
        from .conditioned_dynamics import symbolic_affine_access
        def cv(n):
            op=n["op"]
            if op=="var": return Var(n.get("name",variable))
            if op=="const": return Const(int(n["value"]))
            return Op(op,*(cv(a) for a in n.get("args",[])))
        try:
            expr=cv(expression)
            d=synthesize_symmetries(expr,int(word_bits),[int(c) for c in constants],
                                    variable,int(max_depth),int(probe_samples),int(proof_top_k))
            closures=[]
            for sym in d["exact_symmetries"]:
                cls=classify_simple_transform(sym["transform"],variable)
                if cls and cls["kind"] in ("xor","add"):
                    a=symbolic_affine_access(1<<int(word_bits),int(increment),
                        target_kind=cls["kind"],target_constant=cls["constant"],variable=variable)
                    closures.append({"symmetry":sym["name"],"score":sym["score"],
                        "access_expression":a.expression,
                        "closure":"g(T^kappa(x)(x)) = g(x)","trust":"exact"})
            out={"candidate_count":d["candidate_count"],
                 "ranked_candidates":d["ranked"],
                 "exact_symmetries":d["exact_symmetries"],
                 "conditioned_closures":closures,
                 "ranking_trust":"numeric","proof_trust":"exact" if closures else "none"}
        except (ValueError,TypeError,KeyError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="structural-synthesis")
        # Overall result is NUMERIC because candidate prioritization was sampled;
        # individual closure records explicitly carry exact proof trust.
        step=self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:symmetry_synthesis",
            inputs=[f"w={word_bits}",f"candidates={d['candidate_count']}"],
            output=f"{len(closures)} exact closure(s) after numeric ranking",
            engine="structural-synthesis",trust=TrustLevel.NUMERIC))
        return MathResult(ok=True,data=out,trust=TrustLevel.NUMERIC,
                          engine="structural-synthesis",derivation=[step])

    def discover_structural_conditioned_closure(self, expression: dict,
                                                word_bits: int, increment: int,
                                                constants: list[int],
                                                variable: str = "x") -> MathResult:
        """Discover exact word-expression symmetries, then derive affine access maps.

        Expression schema: {"op":"fold|mul|xor|add|rotl|var|const", ...}.
        Discovery is constrained and proof-producing: only canonical rewrite
        identities are returned as EXACT.
        """
        from .structural_discovery import Var, Const, Op, discover_symmetries
        from .conditioned_dynamics import symbolic_affine_access
        def cv(n):
            op=n["op"]
            if op=="var": return Var(n.get("name",variable))
            if op=="const": return Const(int(n["value"]))
            return Op(op,*(cv(a) for a in n.get("args",[])))
        try:
            expr=cv(expression)
            d=discover_symmetries(expr,int(word_bits),[int(c) for c in constants],variable)
            closures=[]
            for sym in d["symmetries"]:
                t=sym["transform"]
                if t[0]=="xor" and len(t)==3:
                    # canonical xor arguments may place the constant first.
                    consts=[a[1] for a in t[1:] if a[0]=="const"]
                    vars_=[a for a in t[1:] if a[0]=="var" and a[1]==variable]
                    if len(consts)==1 and len(vars_)==1:
                        a=symbolic_affine_access(1<<int(word_bits),int(increment),
                            target_kind="xor",target_constant=consts[0],variable=variable)
                        closures.append({"symmetry":sym["name"],
                            "access_expression":a.expression,
                            "closure":"g(T^kappa(x)(x)) = g(x)","trust":"exact"})
            out={"searched_candidates":d["searched"],"symmetries":d["symmetries"],
                 "conditioned_closures":closures}
        except (ValueError,TypeError,KeyError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="structural-discovery")
        step=self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:structural_discovery",
            inputs=[f"w={word_bits}",f"candidates={d['searched']}"],
            output=f"{len(closures)} exact conditioned closure(s)",
            engine="structural-discovery",trust=TrustLevel.EXACT))
        return MathResult(ok=True,data=out,trust=TrustLevel.EXACT,
                          engine="structural-discovery",derivation=[step])

    def discover_factor_swap_conditioned_closure(self, word_bits: int, increment: int,
                                                  xor_constant: int) -> MathResult:
        """Discover/prove the fold(x*(x xor C)) symmetry and derive kappa(x)."""
        from .conditioned_dynamics import prove_factor_swap_conditioned_closure
        try:
            out=prove_factor_swap_conditioned_closure(int(word_bits),int(increment),int(xor_constant))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False,status="error",errors=[str(exc)],engine="conditioned-dynamics")
        access=out.pop("access")
        step=self._record(DerivationStep(step_id=self._id("step"),
            operation="conditioned:factor_swap_discovery",
            inputs=[f"w={word_bits}",f"A={increment}",f"C={xor_constant}"],
            output=out["closure"],engine="conditioned-dynamics",trust=TrustLevel.EXACT))
        return MathResult(ok=True,data=out,trust=TrustLevel.EXACT,
                          engine="conditioned-dynamics",derivation=[step])

    def affine_conditioned_access(self, modulus: int, increment: int,
                                  state: int, target_state: int) -> MathResult:
        """Solve x+k*A=target (mod M) exactly; supports arbitrary-size integers."""
        from .conditioned_dynamics import affine_cyclic_access_formula
        try:
            k=affine_cyclic_access_formula(int(modulus), int(increment),
                                           int(state), int(target_state))
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="conditioned-dynamics")
        return MathResult(ok=True, data={"lag": str(k), "modulus": str(modulus),
                          "verification": str((int(state)+k*int(increment))%int(modulus))},
                          trust=TrustLevel.EXACT, engine="conditioned-dynamics")

    # ------------------------------------------------------------------
    # Visualization artifacts (mathkernel_viz).  The artifact layer packages
    # already-produced results; it never recomputes mathematics.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Canonical multimodal projections.  A projection records how a
    # mathematical object/result is represented for sensory frontends; it is
    # presentation lineage, never a new mathematical claim.
    # ------------------------------------------------------------------

    def _object_value_dict(self, object_id: str) -> tuple[dict | None, dict | None]:
        """Resolve a stored typed object to (record, encoded value dict).

        The encoded dict carries a ``__pydantic_model__`` marker so the
        projection adapter registry can dispatch on the model class.
        """
        record = self._get_math_object_record(object_id)
        if record is None:
            rv = self.random_variables.get(object_id)
            if rv is not None and hasattr(rv, "values") and hasattr(rv, "probs"):
                return {"object_type": "DiscreteRV",
                        "input_trust": TrustLevel.EXACT, "sources": []}, {
                    "__pydantic_model__": "mathkernel.probability:DiscreteRV",
                    "fields": {"values": [str(v) for v in rv.values],
                               "probabilities": [str(p) for p in rv.probs]},
                }
            return None, None
        value = record["value"]
        if isinstance(value, BaseModel):
            try:
                encoded = value.model_dump(mode="json")
            except (ValueError, TypeError):
                encoded = _encode_typed_value(value)["fields"]
            return record, {
                "__pydantic_model__": (
                    f"{type(value).__module__}:{type(value).__name__}"),
                "fields": encoded,
            }
        return record, _encode_typed_value(value)

    def _adapt_object_projection(self, object_id: str):
        """Build a canonical projection for a stored typed object, or None."""
        import mathkernel_projection as mkp
        from mathkernel_projection.result_adapters import AdapterContext, adapt_result, encoded_fields
        record, encoded = self._object_value_dict(object_id)
        if record is None:
            return None, f"unknown object_id {object_id}"
        trust = record["input_trust"].value
        ctx = AdapterContext(
            trust=trust,
            engine=None,
            resolve=lambda oid: (lambda pair: pair[1])(self._object_value_dict(oid)))
        try:
            spec = adapt_result(encoded, ctx)
        except (ValueError, TypeError, KeyError) as exc:
            return None, str(exc)
        if spec is None:
            return None, (f"no projection adapter registered for object type "
                          f"{record['object_type']!r}")
        from mathkernel_artifacts import SourceRef
        src = SourceRef(
            source_id=object_id, kind="other",
            label=record["object_type"], trust=trust,
            metadata={"object_type": record["object_type"],
                      "sources": list(record.get("sources", []))})
        try:
            proj = mkp.create_projection(
                spec.kind, spec.payload, title=spec.title, source_ref=src,
                trust=trust, parameters=spec.parameters,
                coordinate_names=spec.coordinate_names or None,
                units=spec.units or None,
                information_loss=list(spec.information_loss),
                information_loss_notes=list(spec.information_loss_notes),
                metadata={"adapter": "registry",
                          "object_type": record["object_type"]})
        except (ValueError, TypeError, KeyError) as exc:
            return None, str(exc)
        return proj, None

    def projection_catalog(self) -> MathResult:
        import mathkernel_projection as mkp
        return MathResult(ok=True, data={
            "projection_schema": mkp.PROJECTION_SCHEMA,
            "kinds": mkp.projection_catalog(),
        }, trust=TrustLevel.EXACT, engine="projection")

    def projection_create(self, kind: str | None = None, payload: dict | None = None, *,
                          title: str | None = None,
                          source_object_id: str | None = None,
                          source_ref: dict | None = None,
                          trust: str = "unknown",
                          parameters: dict | None = None,
                          coordinate_names: list[str] | None = None,
                          units: dict[str, str] | None = None,
                          assumptions: list[str] | None = None,
                          evidence_refs: list[str] | None = None,
                          information_loss: list[str] | None = None,
                          information_loss_notes: list[str] | None = None,
                          metadata: dict | None = None) -> MathResult:
        import mathkernel_projection as mkp
        from mathkernel_artifacts import SourceRef
        try:
            src = source_ref
            resolved_trust = str(trust)
            if source_object_id is not None and not payload:
                proj, err = self._adapt_object_projection(source_object_id)
                if proj is None:
                    return MathResult(ok=False, status="error",
                                      errors=[err], engine="projection")
                if title is not None:
                    proj.title = title
                title = title or proj.title
                payload = dict(proj.payload)
                kind = proj.kind
                src = proj.source_ref
                resolved_trust = proj.trust
                parameters = {**proj.parameters, **dict(parameters or {})}
                coordinate_names = coordinate_names or proj.coordinate_names
                units = units or proj.units
                information_loss = information_loss or list(proj.information_loss)
                information_loss_notes = (information_loss_notes
                                          or list(proj.information_loss_notes))
                metadata = {**proj.metadata, **dict(metadata or {})}
            if source_object_id is not None:
                record = self._get_math_object_record(source_object_id)
                if record is None:
                    record, _ = self._object_value_dict(source_object_id)
                if record is None:
                    return MathResult(ok=False, status="error",
                                      errors=[f"unknown object_id {source_object_id}"],
                                      engine="projection")
                object_trust = record["input_trust"].value
                if resolved_trust == "unknown":
                    resolved_trust = object_trust
                elif resolved_trust in TrustLevel._value2member_map_:
                    resolved_trust = self._trust_min(
                        TrustLevel(resolved_trust), record["input_trust"]).value
                src = SourceRef(
                    source_id=source_object_id, kind="other",
                    label=record["object_type"], trust=object_trust,
                    metadata={"object_type": record["object_type"],
                              "sources": list(record.get("sources", []))})
            if kind is None:
                return MathResult(ok=False, status="error",
                                  errors=["provide kind+payload, or a "
                                          "source_object_id with a registered "
                                          "projection adapter"],
                                  engine="projection")
            proj = mkp.create_projection(
                kind, payload or {}, title=title, source_ref=src,
                trust=resolved_trust, parameters=parameters,
                coordinate_names=coordinate_names, units=units,
                assumptions=assumptions, evidence_refs=evidence_refs,
                information_loss=information_loss,
                information_loss_notes=information_loss_notes,
                metadata=metadata)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)],
                              engine="projection")
        pid = self._id("projection")
        # Session handle is intentionally distinct from the content-addressed
        # projection_id inside the object.
        self.projections[pid] = proj
        level = (TrustLevel(proj.trust)
                 if proj.trust in TrustLevel._value2member_map_
                 else TrustLevel.UNKNOWN)
        step = self._record(DerivationStep(
            step_id=self._id("step"), operation=f"projection:create:{kind}",
            inputs=[source_object_id or proj.source_ref.source_id],
            output=pid, engine="projection", trust=level,
            conditions=list(proj.assumptions)))
        return MathResult(ok=True, data={
            "projection_id": pid,
            "content_projection_id": proj.projection_id,
            "kind": proj.kind,
            "projection_schema": proj.projection_schema,
            "trust": proj.trust,
            "information_loss": proj.information_loss,
        }, trust=level, engine="projection", derivation=[step])

    def projection_describe(self, projection_id: str) -> MathResult:
        proj = self.projections.get(projection_id)
        if proj is None:
            return MathResult(ok=False, status="error",
                              errors=[f"unknown projection_id {projection_id}"],
                              engine="projection")
        level = (TrustLevel(proj.trust)
                 if proj.trust in TrustLevel._value2member_map_
                 else TrustLevel.UNKNOWN)
        return MathResult(ok=True, data=proj.model_dump(mode="json"),
                          trust=level, engine="projection")

    def viz_projection(self, projection_id: str, title: str | None = None) -> MathResult:
        import mathkernel_viz as viz
        proj = self.projections.get(projection_id)
        if proj is None:
            return MathResult(ok=False, status="error",
                              errors=[f"unknown projection_id {projection_id}"],
                              engine="viz")
        try:
            doc = viz.from_projection(proj, title=title)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="viz")
        vid = self._id("viz")
        self.viz_documents[vid] = doc
        level = (TrustLevel(doc.trust)
                 if doc.trust in TrustLevel._value2member_map_
                 else TrustLevel.UNKNOWN)
        step = self._record(DerivationStep(
            step_id=self._id("step"), operation="viz:projection",
            inputs=[projection_id], output=vid, engine="viz", trust=level,
            conditions=list(doc.assumptions)))
        return MathResult(ok=True, data={
            "viz_id": vid, "projection_id": projection_id,
            "blocks": len(doc.blocks), "block_kinds": [b.kind for b in doc.blocks],
            "trust": doc.trust, "artifact_schema": doc.artifact_schema,
        }, trust=level, engine="viz", derivation=[step])

    def sonify_projection(self, projection_id: str, mode: str = "auto",
                          title: str | None = None,
                          options: dict | None = None) -> MathResult:
        import mathkernel_sonify as son
        proj = self.projections.get(projection_id)
        if proj is None:
            return MathResult(ok=False, status="error",
                              errors=[f"unknown projection_id {projection_id}"],
                              engine="sonify")
        opts = dict(options or {})
        if title is not None:
            # projection_sonification reads title from the projection; clone to
            # keep the original immutable-by-convention session object intact.
            proj = proj.model_copy(deep=True)
            proj.title = title
        try:
            doc = son.projection_sonification(proj, mode=mode, options=opts)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sonify")
        sid = self._id("sonify")
        self.sonification_documents[sid] = doc
        level = (TrustLevel(doc.trust)
                 if doc.trust in TrustLevel._value2member_map_
                 else TrustLevel.UNKNOWN)
        step = self._record(DerivationStep(
            step_id=self._id("step"), operation="sonify:projection",
            inputs=[projection_id], output=sid, engine="sonify", trust=level,
            conditions=list(doc.assumptions)))
        return MathResult(ok=True, data={
            "sonification_id": sid, "projection_id": projection_id,
            "tracks": len(doc.tracks), "duration": doc.duration(),
            "mappings": list(doc.mappings), "trust": doc.trust,
            "artifact_schema": doc.artifact_schema,
            "acoustic_extraction": doc.metadata.get("projection", {}).get("acoustic_extraction"),
        }, trust=level, engine="sonify", derivation=[step])

    def viz_create(self, view: str = "auto", renderer: str = "auto",
                   title: str | None = None, matrix_id: str | None = None,
                   step_id: str | None = None, system_id: str | None = None,
                   object_id: str | None = None,
                   data: dict | None = None) -> MathResult:
        """Build a VisualizationDocument from kernel-held objects or inline data."""
        import mathkernel_viz as viz
        try:
            if matrix_id is not None:
                cells = self.matrices.get(matrix_id)
                if cells is None:
                    return MathResult(ok=False, status="error",
                                      errors=[f"unknown matrix_id {matrix_id}"], engine="viz")
                from .rendering import render_expr
                from mathkernel_viz.adapters import _num
                rows = [[_num(render_expr(c)) for c in row] for row in cells]
                doc = viz.heatmap_document(rows, title=title or f"Matrix {matrix_id}",
                                           trust=self._matrix_trust(cells).value,
                                           engine="linalg")
            elif step_id is not None:
                trace = self.derivation_trace(step_id)
                if not trace.ok:
                    return MathResult(ok=False, status="error",
                                      errors=trace.errors, engine="viz")
                graph = trace.data.get("graph", {})
                steps = list(graph.get("nodes", []))
                if not steps and step_id in self.derivations:
                    steps = [self.derivations[step_id]]
                doc = viz.dag_document(steps, title=title or f"Derivation {step_id}")
                doc.linked_result = trace.model_dump(mode="json")
            elif system_id is not None:
                system, err = self._get_system(system_id)
                if err:
                    return err
                obs = list(getattr(system, "observation", range(len(system.transition))))
                doc = viz.plot_document(list(range(len(obs))), [float(o) for o in obs],
                                        title=title or f"Observation series {system_id}",
                                        x_label="state", y_label="observation",
                                        trust="exact", engine="finite-system")
            elif object_id is not None:
                proj, err = self._adapt_object_projection(object_id)
                if proj is None:
                    return MathResult(ok=False, status="error",
                                      errors=[err], engine="viz")
                doc = viz.from_projection(proj, title=title)
            elif data:
                doc = self._viz_from_inline(data, title)
                if doc is None:
                    return MathResult(ok=False, status="error",
                                      errors=["unsupported inline data shape"], engine="viz")
            else:
                return MathResult(ok=False, status="error",
                                  errors=["provide matrix_id, step_id, system_id, "
                                          "object_id or data"],
                                  engine="viz")
            doc.metadata["view"] = view
            doc.metadata["renderer"] = viz.select_renderer(view) if renderer == "auto" else renderer
            vid = self._id("viz")
            self.viz_documents[vid] = doc
            step = self._record(DerivationStep(step_id=self._id("step"),
                operation="viz:create", inputs=[matrix_id or step_id or system_id or object_id or "inline"],
                output=f"{len(doc.blocks)} block(s)", engine="viz",
                trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_
                else TrustLevel.UNKNOWN))
            return MathResult(ok=True, data={
                "viz_id": vid, "blocks": len(doc.blocks),
                "block_kinds": [b.kind for b in doc.blocks], "trust": doc.trust,
                "series": list(doc.series), "datasets": list(doc.datasets),
                "artifact_schema": doc.artifact_schema},
                trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_
                else TrustLevel.UNKNOWN, engine="viz", derivation=[step])
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="viz")

    @staticmethod
    def _viz_from_inline(data: dict, title: str | None):
        import mathkernel_viz as viz
        trust = data.get("trust", "numeric")
        if "blocks" in data:
            return MathKernel._viz_compose(data, title)
        # Registered domain adapters get first claim on typed/engine-tagged
        # inline payloads before generic shape heuristics.
        if "__pydantic_model__" in data or "engine" in data:
            from mathkernel_projection.result_adapters import (
                AdapterContext, adapt_result)
            import mathkernel_projection as mkp
            spec = adapt_result(data, AdapterContext(
                trust=str(trust), engine=data.get("engine")))
            if spec is not None:
                proj = mkp.create_projection(
                    spec.kind, spec.payload, title=title or spec.title,
                    trust=str(trust), parameters=spec.parameters,
                    coordinate_names=spec.coordinate_names or None,
                    units=spec.units or None,
                    information_loss=list(spec.information_loss),
                    information_loss_notes=list(spec.information_loss_notes))
                return viz.from_projection(proj, title=title)
        if "matrix" in data:
            return viz.heatmap_document(data["matrix"], title=title or "Matrix", trust=trust)
        if "grid" in data:
            return viz.surface_document(data["grid"], title=title or "Surface", trust=trust)
        if "trajectory" in data:
            return viz.trajectory3d_document(data["trajectory"],
                                             title=title or "Trajectory", trust=trust)
        if "points" in data:
            pts = data["points"]
            if pts and len(pts[0]) >= 3:
                return viz.points3d_document(pts, title=title or "Point cloud", trust=trust)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return viz.plot_document(xs, ys, title=title or "Plot", trust=trust)
        if "xs" in data and "ys" in data:
            return viz.plot_document(data["xs"], data["ys"], title=title or "Plot",
                                     trust=trust)
        if "enclosures" in data:
            return viz.interval_plot_document(data["enclosures"],
                                              title=title or "Certified enclosures")
        return None

    @staticmethod
    def _viz_compose(data: dict, title: str | None):
        """Compose a document from a generic block specification.

        data: {"title": str, "layout": {"cols": int}, "blocks": [
            {"kind": <block kind>, "title"?: str, "span"?: int,
             "config"?: dict, "bindings"?: dict, "data"?: dict} ...]}

        ``data`` per kind (all inline, all optional depending on kind):
          plot2d:         {"series": [{"x": [...], "y": [...], "label"?,
                                       "trust"?, "role"?, "marker"?,
                                       "color"?, "stack"?}]}
          point_cloud_3d: {"series": [{"points": [[x,y,z]...], ...}]}
                          or {"values": [...1D...], "embed": {"lags": [0,k,2k]}}
          trajectory_3d:  {"states": [[x,y,z]...]}
          surface_3d:     {"grid": [[...]]}
          vector_field_3d: {"origins": [[...]], "vectors": [[...]]}
          histogram:      {"values": [...], "bins"?: int}
          heatmap:        {"matrix": [[...]]}
          dag:            {"nodes": [{"id","label","parents"?,"trust"?}]}
                          (omit to render the document provenance)
          metric_grid:    {"entries": [{"label","value","hint"?}]}
          data_table:     {"columns": [{"label","values":[...]}]} or {"rows": [[...]]}
          text:           {"text": str}
          select:         config {"param", "options": [{"label","value"}]}
        """
        import mathkernel_viz as viz
        layout = data.get("layout") or {}
        doc = viz.dashboard(data.get("title") or title or "Composition",
                            cols=int(layout.get("cols", 2)))
        for spec in data["blocks"]:
            kind = spec.get("kind")
            cfg = dict(spec.get("config") or {})
            raw_data = spec.get("data")
            if raw_data is None:
                raw_data = {}
            elif isinstance(raw_data, str) and kind == "text":
                raw_data = {"text": raw_data}
            if not isinstance(raw_data, dict):
                raise ValueError(
                    f"block {kind!r} requires 'data' as an object; "
                    f"got {type(raw_data).__name__}")
            d = dict(raw_data)
            kw = {"title": spec.get("title", ""), "span": int(spec.get("span", 1)),
                  "block_id": spec.get("block_id")}
            bindings = spec.get("bindings")
            trust = d.get("trust", "numeric")
            if kind == "plot2d":
                block = None
                first_kw = dict(kw)
                first_kw.pop("block_id", None)
                for s in d.get("series", []):
                    block = viz.add_plot(
                        doc, s.get("x", []), s.get("y", []),
                        label=s.get("label", ""), trust=s.get("trust", trust),
                        role=s.get("role", "data"), source=s.get("source", ""),
                        marker=s.get("marker"), color=s.get("color"),
                        stack=s.get("stack"), config=cfg, bindings=bindings,
                        **(first_kw if block is None
                           else {"block_id": block.block_id}))
            elif kind == "point_cloud_3d":
                if "values" in d:
                    cfg["embed"] = dict(d.get("embed") or {"lags": [0, 1, 2]})
                    did = f"seq{len(doc.datasets) + 1}"
                    doc.add_dataset(viz.encode_dataset(
                        did, [float(v) for v in d["values"]],
                        label=d.get("label", "sequence"), trust=trust))
                    cfg["embed"]["dataset"] = did
                    block = doc.add_block("point_cloud_3d", config=cfg,
                                          datasets=[did], bindings=bindings,
                                          **kw)
                else:
                    for s in d.get("series", []):
                        block = viz.add_point_cloud(
                            doc, s.get("points", []), label=s.get("label", ""),
                            trust=s.get("trust", trust),
                            role=s.get("role", "data"), source=s.get("source", ""),
                            color=s.get("color"), config=cfg, bindings=bindings,
                            **kw)
            elif kind == "trajectory_3d":
                viz.add_trajectory_3d(doc, d.get("states", []), trust=trust,
                                      config=cfg, **kw)
            elif kind == "surface_3d":
                viz.add_surface_3d(doc, d.get("grid", []), trust=trust,
                                   config=cfg, **kw)
            elif kind == "vector_field_3d":
                viz.add_vector_field_3d(doc, d.get("origins", []),
                                        d.get("vectors", []), trust=trust,
                                        config=cfg, **kw)
            elif kind == "histogram":
                viz.add_histogram(doc, d.get("values", []),
                                  bins=int(d.get("bins", cfg.get("bins", 64))),
                                  trust=trust, config=cfg, bindings=bindings, **kw)
            elif kind == "heatmap":
                viz.add_heatmap(doc, d.get("matrix", []),
                                trust=d.get("trust", "exact"), config=cfg, **kw)
            elif kind == "dag":
                viz.add_dag(doc, nodes=d.get("nodes"), config=cfg, **kw)
            elif kind == "metric_grid":
                viz.add_metric_grid(doc, d.get("entries"), config=cfg,
                                    bindings=bindings, **kw)
            elif kind == "data_table":
                viz.add_data_table(doc, d.get("columns"), rows=d.get("rows"),
                                   config=cfg, bindings=bindings, **kw)
            elif kind == "text":
                viz.add_text(doc, d.get("text", ""), config=cfg, **kw)
            elif kind == "select":
                viz.add_select(doc, cfg.get("param", "param"),
                               cfg.get("options", []), label=cfg.get("label", ""),
                               default=int(cfg.get("default", 0)), **kw)
            else:
                raise ValueError(f"unknown block kind {kind!r}")
        doc.trust = doc.weakest_trust()
        return doc

    def viz_dag(self, step_id: str, title: str | None = None) -> MathResult:
        """Provenance/obligation DAG document for a derivation step."""
        return self.viz_create(step_id=step_id, title=title)

    def viz_koopman(self, system_id: str, basis_spec: dict, exact: bool = True,
                    title: str | None = None) -> MathResult:
        """Koopman mode-visibility plot: MathKernel computes, viz packages."""
        import mathkernel_viz as viz
        res = self.koopman_visibility(system_id, basis_spec, exact=exact)
        if not res.ok:
            return res
        vis = [float(v) for v in res.data["visibility"]]
        doc = viz.plot_document(list(range(len(vis))), vis,
                                title=title or f"Koopman mode visibility {system_id}",
                                x_label="mode", y_label="rho_O",
                                trust=res.trust.value, engine=res.engine)
        doc.provenance = viz.adapters.provenance_from_steps(res.derivation)
        doc.linked_result = res.model_dump(mode="json")
        doc.trust = doc.weakest_trust()
        vid = self._id("viz")
        self.viz_documents[vid] = doc
        return MathResult(ok=True, data={"viz_id": vid,
                          "blocks": len(doc.blocks),
                          "block_kinds": [b.kind for b in doc.blocks],
                          "modes": len(vis), "trust": doc.trust,
                          "artifact_schema": doc.artifact_schema},
                          trust=res.trust, engine="viz", derivation=res.derivation,
                          evidence_bundle=res.evidence_bundle,
                          claim_evidence=res.claim_evidence)

    def viz_export(self, viz_id: str, path: str, mode: str = "portable",
                   include_provenance: bool = True,
                   include_reproducibility: bool = True,
                   deterministic: bool = True) -> MathResult:
        """Write a portable HTML artifact; returns metadata, never the payload."""
        import mathkernel_viz as viz
        doc = self.viz_documents.get(viz_id)
        if doc is None:
            return MathResult(ok=False, status="error",
                              errors=[f"unknown viz_id {viz_id}"], engine="viz")
        try:
            info = viz.export_html(doc, path, mode=mode,
                                   include_provenance=include_provenance,
                                   include_reproducibility=include_reproducibility,
                                   mathkernel_version=self.capabilities()["version"],
                                   deterministic=deterministic)
        except (ValueError, TypeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="viz")
        # Export is a presentation step: it must carry the document's trust,
        # never upgrade it.
        doc_trust = (TrustLevel(doc.trust)
                     if doc.trust in TrustLevel._value2member_map_
                     else TrustLevel.UNKNOWN)
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="viz:export", inputs=[viz_id], output=info["path"],
            engine="viz", trust=doc_trust))
        return MathResult(ok=True, data={"viz_id": viz_id, **info},
                          trust=doc_trust, engine="viz", derivation=[step])

    # ------------------------------------------------------------------
    # Scientific sonification (mathkernel_sonify). Perception is candidate
    # generation only; mappings remain explicit provenance.
    # ------------------------------------------------------------------

    def sonify_create(self, data: list[float], mode: str = "auto",
                      title: str | None = None, options: dict | None = None) -> MathResult:
        import mathkernel_sonify as son
        opts = dict(options or {})
        if title is not None:
            opts["title"] = title
        try:
            doc = son.sonify(data, mode=mode, **opts)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sonify")
        sid = self._id("sonify")
        self.sonification_documents[sid] = doc
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="sonify:create", inputs=["inline"],
            output=f"{len(doc.tracks)} track(s)", engine="sonify",
            trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN))
        return MathResult(ok=True, data={"sonification_id": sid, "tracks": len(doc.tracks),
            "duration": doc.duration(), "mappings": list(doc.mappings),
            "artifact_schema": doc.artifact_schema, "trust": doc.trust},
            trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN,
            engine="sonify", derivation=[step])

    def sonify_compare(self, prediction: list[float], observation: list[float],
                       mode: str = "stereo", title: str | None = None,
                       options: dict | None = None) -> MathResult:
        import mathkernel_sonify as son
        opts = dict(options or {})
        if title is not None: opts["title"] = title
        try:
            doc = son.sonify_compare(prediction, observation, mode=mode, **opts)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sonify")
        sid=self._id("sonify"); self.sonification_documents[sid]=doc
        return MathResult(ok=True,data={"sonification_id":sid,"tracks":len(doc.tracks),
            "duration":doc.duration(),"mode":mode,"artifact_schema":doc.artifact_schema},
            trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN,engine="sonify")

    def sonify_describe(self, sonification_id: str) -> MathResult:
        doc=self.sonification_documents.get(sonification_id)
        if doc is None: return MathResult(ok=False,status="error",errors=[f"unknown sonification_id {sonification_id}"],engine="sonify")
        return MathResult(ok=True,data=doc.model_dump(mode="json"),
            trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN,engine="sonify")

    def sonify_export(self, sonification_id: str, path: str) -> MathResult:
        import mathkernel_sonify as son
        doc=self.sonification_documents.get(sonification_id)
        if doc is None: return MathResult(ok=False,status="error",errors=[f"unknown sonification_id {sonification_id}"],engine="sonify")
        try: info=son.write_wav(doc,path)
        except (ValueError,TypeError,OSError) as exc: return MathResult(ok=False,status="error",errors=[str(exc)],engine="sonify")
        return MathResult(ok=True,data={"sonification_id":sonification_id,**info},trust=TrustLevel(doc.trust) if doc.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN,engine="sonify")

    # ------------------------------------------------------------------
    # Unified multimodal research artifacts (mathkernel_multimodal).  One
    # portable HTML file carrying visualizations, sonifications, evidence,
    # lineage and cross-modal synchronization links.
    # ------------------------------------------------------------------

    def research_artifact_create(self, title: str = "MathKernel research artifact",
                                 viz_ids: list[str] | None = None,
                                 sonification_ids: list[str] | None = None,
                                 result: dict | None = None,
                                 synchronize: bool = True) -> MathResult:
        import mathkernel_multimodal as mkm
        viz_docs = []
        for vid in (viz_ids or []):
            doc = self.viz_documents.get(vid)
            if doc is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"unknown viz_id {vid}"], engine="multimodal")
            viz_docs.append(doc)
        son_docs = []
        for sid in (sonification_ids or []):
            doc = self.sonification_documents.get(sid)
            if doc is None:
                return MathResult(ok=False, status="error",
                                  errors=[f"unknown sonification_id {sid}"],
                                  engine="multimodal")
            son_docs.append(doc)
        if not viz_docs and not son_docs:
            return MathResult(ok=False, status="error",
                              errors=["at least one viz_id or sonification_id is "
                                      "required"], engine="multimodal")
        try:
            artifact = mkm.build_artifact(
                title=title, visualizations=viz_docs, sonifications=son_docs,
                result=result, mathkernel_version=self.capabilities()["version"],
                auto_synchronize=synchronize)
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)],
                              engine="multimodal")
        aid = self._id("artifact")
        artifact.artifact_id = aid
        self.research_artifacts[aid] = artifact
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="multimodal:create",
            inputs=list(viz_ids or []) + list(sonification_ids or []),
            output=f"{len(viz_docs)} visualization(s), {len(son_docs)} sonification(s)",
            engine="multimodal",
            trust=TrustLevel(artifact.trust) if artifact.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN))
        return MathResult(ok=True, data={
            "artifact_id": aid, "artifact_schema": artifact.artifact_schema,
            "visualizations": len(artifact.visualizations),
            "sonifications": len(artifact.sonifications),
            "synchronization_links": len(artifact.synchronization),
            "trust": artifact.trust},
            trust=TrustLevel(artifact.trust) if artifact.trust in TrustLevel._value2member_map_ else TrustLevel.UNKNOWN,
            engine="multimodal", derivation=[step])

    def research_artifact_export(self, artifact_id: str, path: str) -> MathResult:
        import mathkernel_multimodal as mkm
        artifact = self.research_artifacts.get(artifact_id)
        if artifact is None:
            return MathResult(ok=False, status="error",
                              errors=[f"unknown artifact_id {artifact_id}"],
                              engine="multimodal")
        try:
            info = mkm.export_html(artifact, path)
        except (ValueError, TypeError, OSError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)],
                              engine="multimodal")
        art_trust = (TrustLevel(artifact.trust)
                     if artifact.trust in TrustLevel._value2member_map_
                     else TrustLevel.UNKNOWN)
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation="multimodal:export", inputs=[artifact_id],
            output=info["path"], engine="multimodal", trust=art_trust))
        return MathResult(ok=True, data={"artifact_id": artifact_id, **info},
                          trust=art_trust, engine="multimodal",
                          derivation=[step])

    def closure_search(self, kind: str, **params) -> MathResult:
        from . import relations as rel
        max_results = min(int(params.get("max_results", self.settings.max_closure_results)),
                          self.settings.max_closure_results)
        try:
            if kind == "cyclic":
                if "multipliers" in params:
                    a_ks = [int(a) for a in params["multipliers"]]
                else:
                    a_k, d = int(params["a_k"]), int(params["d"])
                    a_ks = [pow(a_k, j, int(params["m"])) for j in range(d)]
                m = int(params["m"])
                tuples = rel.cyclic_closure(a_ks, m, int(params["weight_bound"]), max_results)
                if params.get("irreducible_only"):
                    tuples = [t for t in tuples if rel.is_irreducible_cyclic(t, a_ks, m)]
                data = {"tuples": [list(t) for t in tuples], "count": len(tuples), "modulus": str(m)}
            elif kind == "cyclic_order":
                out = rel.cyclic_closure_order(int(params["a_k"]), int(params["m"]),
                                               int(params["weight_bound"]),
                                               int(params.get("d_min", 2)),
                                               int(params.get("d_max", 6)))
                data = out
            elif kind == "binary":
                width = int(params["width"])
                rows = [self._parse_hex(r0, width) for r0 in params["rows"]]
                tuples = rel.binary_closure(rows, width, int(params["d"]),
                                            int(params.get("k_step", 1)),
                                            int(params["max_weight"]), max_results)
                if params.get("irreducible_only"):
                    tuples = [t for t in tuples
                              if rel.is_irreducible_binary(t, rows, width,
                                                           int(params.get("k_step", 1)))]
                data = {"tuples": [[f"{w:x}" for w in t] for t in tuples],
                        "count": len(tuples), "width": width}
            elif kind == "binary_order":
                width = int(params["width"])
                rows = [self._parse_hex(r0, width) for r0 in params["rows"]]
                data = rel.binary_closure_order(rows, width, int(params.get("k_step", 1)),
                                                int(params["max_weight"]),
                                                int(params.get("d_min", 2)),
                                                int(params.get("d_max", 4)))
            else:
                raise ValueError("kind must be 'cyclic', 'cyclic_order', 'binary', "
                                 "or 'binary_order'")
        except (ValueError, TypeError, KeyError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="relations")
        step = self._record(DerivationStep(step_id=self._id("step"),
            operation=f"closure:{kind}", inputs=[str(sorted(params))],
            output=str(data.get("count", data.get("order"))), engine="relations",
            trust=TrustLevel.EXACT))
        return MathResult(ok=True, data={"kind": kind, **data}, trust=TrustLevel.EXACT,
                          engine="relations", derivation=[step])

    # ------------------------------------------------------------------
    # Search sweeps and asynchronous jobs
    # ------------------------------------------------------------------

    def cuboid_sweep(self, bound: int, engine: str = "auto", workers: int | None = None) -> MathResult:
        from .cuboid import sweep_leg_pairs
        resolved = resolve_workers(workers, cap=self.settings.max_workers) \
            if self.settings.enable_parallel else 1
        try:
            report = sweep_leg_pairs(bound, engine=engine, workers=resolved)
        except (ValueError, RuntimeError) as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="cuboid_sweep")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="cuboid_sweep",
            inputs=[f"bound={bound}"], output=f"{report['pair_count']} Pythagorean leg pairs",
            engine="cuboid_sweep", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data=report, trust=TrustLevel.EXACT,
                          engine="cuboid_sweep", derivation=[step])

    _JOB_RUNNERS = {
        "collatz_sieve": ("collatz_sieve", {"n_max", "x_min", "workers", "n_min", "canonical", "engine"}),
        "cuboid_sweep": ("cuboid_sweep", {"bound", "engine", "workers"}),
    }

    def job_submit(self, kind: str, params: dict | None = None) -> MathResult:
        spec = self._JOB_RUNNERS.get(kind)
        if spec is None:
            return MathResult(ok=False, status="error",
                errors=[f"Unknown job kind: {kind}; choose from {sorted(self._JOB_RUNNERS)}"], engine="jobs")
        method_name, allowed = spec
        params = dict(params or {})
        unknown = sorted(set(params) - allowed)
        if unknown:
            return MathResult(ok=False, status="error",
                errors=[f"Unknown parameter(s) for {kind}: {unknown}; allowed: {sorted(allowed)}"], engine="jobs")
        job_id = self._id("job")
        self.jobs[job_id] = {"job_id": job_id, "kind": kind, "params": params, "status": "queued",
                             "submitted_at": time.time(), "started_at": None, "finished_at": None,
                             "result": None, "error": None}

        def _run():
            job = self.jobs[job_id]
            job["status"] = "running"
            job["started_at"] = time.time()
            try:
                result = getattr(self, method_name)(**params)
                job["result"] = result.model_dump(mode="json")
                job["status"] = "done" if result.ok else "failed"
                if not result.ok:
                    job["error"] = "; ".join(result.errors)
            except Exception as exc:  # job isolation: report, never crash the pool
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                job["finished_at"] = time.time()

        self._job_pool.submit(_run)
        self._evict_finished_jobs()
        return MathResult(ok=True, data={"job_id": job_id, "kind": kind, "status": "queued"},
                          trust=TrustLevel.EXACT, engine="jobs")

    def _evict_finished_jobs(self) -> None:
        cap = self.settings.max_jobs_retained
        if len(self.jobs) <= cap:
            return
        finished = sorted((j for j in self.jobs.values() if j["status"] in {"done", "failed"}),
                          key=lambda j: j["finished_at"] or 0)
        for job in finished[:len(self.jobs) - cap]:
            del self.jobs[job["job_id"]]

    def job_list(self, status: str | None = None) -> MathResult:
        jobs = sorted(self.jobs.values(), key=lambda j: j["submitted_at"], reverse=True)
        if status is not None:
            jobs = [j for j in jobs if j["status"] == status]
        out = [{"job_id": j["job_id"], "kind": j["kind"], "status": j["status"],
                "submitted_at": j["submitted_at"], "finished_at": j["finished_at"]} for j in jobs]
        return MathResult(ok=True, data={"jobs": out, "count": len(out)},
                          trust=TrustLevel.EXACT, engine="jobs")

    def job_status(self, job_id: str) -> MathResult:
        job = self.jobs.get(job_id)
        if job is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown job_id: {job_id}"], engine="jobs")
        data = {k: job[k] for k in ("job_id", "kind", "params", "status", "submitted_at", "started_at", "finished_at", "error")}
        if job["started_at"]:
            end = job["finished_at"] or time.time()
            data["elapsed_seconds"] = round(end - job["started_at"], 3)
        return MathResult(ok=True, data=data, trust=TrustLevel.EXACT, engine="jobs")

    def job_result(self, job_id: str) -> MathResult:
        job = self.jobs.get(job_id)
        if job is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown job_id: {job_id}"], engine="jobs")
        if job["status"] in {"queued", "running"}:
            return MathResult(ok=False, status="error",
                errors=[f"Job is still {job['status']}; poll job_status."], engine="jobs")
        if job["result"] is None:
            return MathResult(ok=False, status="error",
                errors=[job["error"] or "Job failed without a result"], engine="jobs")
        stored = job["result"]
        underlying = MathResult.model_validate(stored)
        return MathResult(
            ok=underlying.ok,
            status=underlying.status,
            data=stored,
            assumptions_used=list(underlying.assumptions_used),
            side_conditions=list(underlying.side_conditions),
            warnings=list(underlying.warnings),
            errors=list(underlying.errors),
            trust=underlying.trust,
            engine="jobs",
            derivation=[step.model_copy(deep=True) for step in underlying.derivation],
            evidence=[item.model_copy(deep=True) for item in underlying.evidence],
            evidence_bundle=underlying.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in underlying.claim_evidence.items()
            },
            semantic_status=underlying.semantic_status,
        )

    def parse_latex(self, latex: str) -> MathResult:
        if len(latex) > self.settings.max_input_length:
            return MathResult(ok=False, status="error",
                errors=[f"Input too long (maximum {self.settings.max_input_length} characters)"],
                trust=TrustLevel.UNKNOWN, engine="latex")
        try:
            from sympy.parsing.latex import parse_latex as _sp_parse_latex
        except ImportError:
            return MathResult(ok=False, status="error",
                errors=["LaTeX parsing requires antlr4-python3-runtime==4.11 (pip install antlr4-python3-runtime==4.11)"],
                engine="latex")
        try:
            expr = _sp_parse_latex(latex)
        except Exception as exc:
            return MathResult(ok=False, status="error", errors=[f"LaTeX parse error: {exc}"],
                              trust=TrustLevel.UNKNOWN, engine="latex")
        # sympy's LaTeX parser returns \pi as a plain symbol; restore constant semantics.
        expr = expr.subs({sp.Symbol("pi"): sp.pi}, simultaneous=True)
        try:
            ir = self.sympy.from_sympy(expr)
        except ValueError as exc:
            return MathResult(ok=False, status="error",
                errors=[f"LaTeX parsed but is outside the MathIR fragment: {exc}"],
                trust=TrustLevel.UNKNOWN, engine="latex")
        nodes = _count_nodes(ir)
        if nodes > self.settings.max_expression_nodes:
            return MathResult(ok=False, status="error",
                errors=[f"Expression has {nodes} nodes; limit is {self.settings.max_expression_nodes}."],
                trust=TrustLevel.UNKNOWN, engine="latex")
        eid = self._id("expr")
        self.expressions[eid] = ir
        self.expression_sources[eid] = latex
        display = render_expr(ir)
        trust = self._expr_trust(ir)
        step = self._record(DerivationStep(step_id=self._id("step"), operation="parse_latex",
            inputs=[latex], output_expr_id=eid, output=display, engine="latex", trust=trust))
        self.expression_producers[eid] = step.step_id
        return MathResult(ok=True, data={"expr_id": eid, "ir": ir.model_dump(mode="json"),
            "source": latex, "display": display}, trust=trust, engine="latex", derivation=[step])

    def prove_equivalence(self,left: str,right: str, context_id: str|None=None, formal: bool=True):
        l,r=parse_math(left),parse_math(right)
        input_trust = self._trust_min(self._expr_trust(l), self._expr_trust(r))
        approximate = input_trust == TrustLevel.NUMERIC
        ctx = self.contexts.get(context_id) if context_id else None
        # Formal backends commonly encode decimal syntax as exact rationals. That
        # proves a different statement from one containing approximate RealNodes,
        # so never issue a formal certificate for approximate inputs.
        status, trust, evidence, detail = self.verifier.equivalence(l, r, ctx, formal=(formal and not approximate))
        trust = self._trust_min(trust, input_trust)
        if approximate:
            detail.pop("lean_certificate", None)
            detail.pop("lean_tactic", None)
            evidence = [e for e in evidence if getattr(e, "engine", "") != "lean"]
        outward = {VerificationStatus.PROVED:"verified", VerificationStatus.DISPROVED:"refuted", VerificationStatus.UNKNOWN:"unknown"}[status]
        step=self._record(DerivationStep(step_id=self._id("step"),operation="prove_equivalence",output=detail.get("symbolic_difference"),
            engine="verification_coordinator",trust=trust,evidence=evidence))
        warnings=[]
        if approximate:
            warnings.append("Approximate decimal input: equivalence is not an exact/formal certificate; use exact rationals or interval certification for stronger evidence.")
        if status is VerificationStatus.UNKNOWN: warnings.append("No backend established or refuted the statement in its supported fragment.")
        return MathResult(ok=True,status=outward,data=detail,warnings=warnings,trust=trust,engine="verification_coordinator",derivation=[step],evidence=evidence)

    def counterexample(self,left: str,right: str, context_id: str|None=None):
        l,r=parse_math(left),parse_math(right)
        if self._trust_min(self._expr_trust(l), self._expr_trust(r)) == TrustLevel.NUMERIC:
            return MathResult(ok=True,status="unknown",
                warnings=["Exact SMT counterexamples are disabled for approximate decimal inputs; use exact rationals or interval methods."],
                trust=TrustLevel.UNKNOWN,engine="z3")
        ctx = self.contexts.get(context_id) if context_id else None
        if not self.z3.available: return MathResult(ok=False,status="error",errors=["z3-solver is not installed"],engine="z3")
        assumptions = [a.expression for a in ctx.assumptions] if ctx else []
        domains = ctx.domains if ctx else {}
        try: status, model = self.z3.counterexample_equivalence(l,r,assumptions,domains)
        except (TypeError, ValueError) as exc:
            return MathResult(ok=True,status="unknown",warnings=[str(exc)],trust=TrustLevel.UNKNOWN,engine="z3")
        if status == "disproved": return MathResult(ok=True,status="refuted",data={"counterexample":model},trust=TrustLevel.EXACT,engine="z3")
        if status == "proved": return MathResult(ok=True,status="verified",data={"counterexample":None,"reason":"negation is unsatisfiable"},trust=TrustLevel.EXACT,engine="z3")
        return MathResult(ok=True,status="unknown",trust=TrustLevel.UNKNOWN,engine="z3")

    def get_expression(self, expr_id: str) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="mathir")
        return MathResult(ok=True, data={"expr_id": expr_id, "ir": ir.model_dump(mode="json"),
            "source": self.expression_sources.get(expr_id), "display": render_expr(ir),
            "producer_step_id": self.expression_producers.get(expr_id)},
            trust=expression_trust(self, expr_id), engine="mathir",
            assumptions_used=self.expression_provenance.get(expr_id, {}).get("assumptions", []))

    def substitute(self, expr_id: str, substitutions: dict[str, str]) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="mathir")
        try:
            replacements = {name: parse_math(text) for name, text in substitutions.items()}
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[f"Invalid substitution: {exc}"], engine="parser")
        trust = self._trust_min(expression_trust(self, expr_id),
                                *(self._expr_trust(r) for r in replacements.values()))
        new_ir = _substitute_ir(ir.model_copy(deep=True), replacements)
        out_eid = self._id("expr")
        self.expressions[out_eid] = new_ir
        display = render_expr(new_ir)
        self.expression_sources[out_eid] = display
        step = self._record(DerivationStep(step_id=self._id("step"), operation="substitute",
            inputs=[expr_id, *[f"{k} := {v}" for k, v in substitutions.items()]],
            parents=self._parents_for_exprs([expr_id]), output=display, output_expr_id=out_eid,
            engine="mathir", trust=trust))
        self.expression_producers[out_eid] = step.step_id
        return MathResult(ok=True, data={"expr_id": out_eid, "result": display,
            "ir": new_ir.model_dump(mode="json")}, trust=trust, engine="mathir", derivation=[step])

    def infer_structure(self, expr_id: str, context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="mathir")
        ctx = self.contexts.get(context_id) if context_id else None
        caps = infer_capabilities(ir)
        constraints = extract_constraints(ir)
        assumptions = [render_expr(a.expression) for a in ctx.assumptions] if ctx else []
        for a in assumptions:
            if a not in constraints:
                constraints.append(a)
        data = {"required_capabilities": sorted(caps),
                "suggested_structure": suggest_structure(caps),
                "constraints": constraints,
                "display": render_expr(ir)}
        step = self._record(DerivationStep(step_id=self._id("step"), operation="infer_structure",
            inputs=[expr_id], parents=self._parents_for_exprs([expr_id]),
            output=data["suggested_structure"], engine="mathir", trust=TrustLevel.EXACT))
        return MathResult(ok=True, data=data, assumptions_used=assumptions,
            trust=TrustLevel.EXACT, engine="mathir", derivation=[step])

    def codegen(self, expr_id: str, language: str = "typescript", mode: str = "generic",
                target: str = "evaluate", variable: str | None = None,
                context_id: str | None = None) -> MathResult:
        ir = self.expressions.get(expr_id)
        if ir is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown expr_id: {expr_id}"], engine="codegen")
        if mode != "generic":
            return MathResult(ok=False, status="error",
                errors=[f"Unsupported codegen mode: {mode}; only 'generic' is implemented."], engine="codegen")
        emitter_cls = EMITTERS.get(language)
        if emitter_cls is None:
            return MathResult(ok=False, status="error",
                errors=[f"Unsupported language: {language}; choose from {sorted(EMITTERS)}."], engine="codegen")
        ctx = self.contexts.get(context_id) if context_id else None
        assumptions = [render_expr(a.expression) for a in ctx.assumptions] if ctx else []
        source = self.expression_sources.get(expr_id, render_expr(ir))
        emitter = emitter_cls()
        expected: list = []
        symbols: set[str] = set(); calls: set[str] = set(); kinds: set[str] = set()
        _scan_ir(ir, symbols, calls, kinds)
        try:
            if target == "evaluate":
                if ir.kind in {"eq", "ne", "lt", "le", "gt", "ge"}:
                    return MathResult(ok=False, status="error",
                        errors=["target='evaluate' requires a plain expression, not a relation."], engine="codegen")
                params = sorted(symbols)
                bodies = [ir]
                expected = [self.sympy.to_sympy(ir)]
                function_name = "evaluate_expression"
                returns_relation = False
            elif target == "solve":
                if ir.kind != "eq":
                    return MathResult(ok=False, status="error",
                        errors=["target='solve' requires an equality (e.g. 'a*x^2 + b*x + c = 0')."], engine="codegen")
                if not variable:
                    return MathResult(ok=False, status="error",
                        errors=["target='solve' requires the 'variable' to solve for."], engine="codegen")
                domain = (ctx.domains.get(variable, "complex") if ctx else "complex").lower()
                sol = self.sympy.solve(ir, variable, domain)
                domain_fallback = False
                if not isinstance(sol, sp.FiniteSet) and domain != "complex":
                    retry = self.sympy.solve(ir, variable, "complex")
                    if isinstance(retry, sp.FiniteSet):
                        sol = retry
                        domain_fallback = True
                if not isinstance(sol, sp.FiniteSet):
                    return MathResult(ok=False, status="error",
                        errors=[f"Solution set is {type(sol).__name__}; codegen requires a finite explicit solution set."],
                        engine="codegen")
                symbols.discard(variable)
                params = sorted(symbols)
                bodies = [self.sympy.from_sympy(s) for s in sorted(sol, key=sp.default_sort_key)]
                expected = sorted(sol, key=sp.default_sort_key)
                function_name = f"solve_for_{variable}"
                returns_relation = False
            elif target == "constraint":
                if ir.kind not in {"eq", "ne", "lt", "le", "gt", "ge"}:
                    return MathResult(ok=False, status="error",
                        errors=["target='constraint' requires a relation."], engine="codegen")
                params = sorted(symbols)
                bodies = [ir]
                expected = [self.sympy.to_sympy(ir)]
                function_name = "check_constraint"
                returns_relation = True
            else:
                return MathResult(ok=False, status="error",
                    errors=[f"Unsupported codegen target: {target}; choose from evaluate, solve, constraint."],
                    engine="codegen")
            extra_constraints = []
            warnings: list[str] = []
            if target == "solve" and domain_fallback:
                extra_constraints.append(
                    f"Solutions were computed over the complex field; existence in the "
                    f"declared domain '{domain}' depends on the parameters.")
                warnings.append(extra_constraints[-1])
            artifact = build_artifact(emitter=emitter, function_name=function_name, params=params,
                bodies=bodies, source=source, assumptions=assumptions, target=target,
                returns_relation=returns_relation, extra_constraints=extra_constraints)
        except CodegenError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="codegen")
        except ValueError as exc:
            return MathResult(ok=False, status="error",
                errors=[f"Solution is outside the representable fragment: {exc}"], engine="codegen")
        artifact_id = self._id("artifact")
        artifact["artifact_id"] = artifact_id
        artifact["mode"] = mode
        self.artifacts[artifact_id] = {"artifact": artifact, "expr_id": expr_id, "target": target,
                                       "variable": variable, "params": params, "expected": expected}
        step = self._record(DerivationStep(step_id=self._id("step"), operation=f"codegen:{language}:{target}",
            inputs=[expr_id], parents=self._parents_for_exprs([expr_id]), output=artifact_id,
            engine="codegen", trust=TrustLevel.EXACT, conditions=artifact["constraints"]))
        return MathResult(ok=True, data=artifact, assumptions_used=assumptions,
            side_conditions=artifact["constraints"], warnings=warnings,
            trust=TrustLevel.EXACT, engine="codegen", derivation=[step])

    def verify_code(self, artifact_id: str, checks: list[str] | None = None) -> MathResult:
        record = self.artifacts.get(artifact_id)
        if record is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown artifact_id: {artifact_id}"], engine="codegen")
        checks = checks or ["typecheck", "symbolic_roundtrip"]
        results = []
        for check in checks:
            if check == "typecheck":
                results.append(typecheck_artifact(record["artifact"]))
            elif check == "symbolic_roundtrip":
                results.append(symbolic_roundtrip(record))
            else:
                results.append({"check": check, "status": "unavailable",
                                "detail": f"Unknown check: {check}; supported: typecheck, symbolic_roundtrip."})
        failed = [r for r in results if r["status"] == "failed"]
        unavailable = [r for r in results if r["status"] == "unavailable"]
        warnings = [f"Check '{r['check']}' unavailable: {r['detail']}" for r in unavailable]
        # Trust is the minimum over the requested evidentiary checks: a failed
        # check refutes the claim, and an unavailable required check means the
        # verification is partial — never exact.
        if failed:
            trust = TrustLevel.UNKNOWN
        elif unavailable:
            trust = TrustLevel.SYMBOLIC
            warnings.append("Verification is partial: unavailable checks were not "
                            "performed, so the result is not exact evidence.")
        else:
            trust = TrustLevel.EXACT
        step = self._record(DerivationStep(step_id=self._id("step"), operation="verify_code",
            inputs=[artifact_id], output="failed" if failed else "passed",
            engine="codegen", trust=trust))
        return MathResult(ok=not failed, status="error" if failed else "ok",
            data={"artifact_id": artifact_id, "checks": results,
                  "all_passed": not failed and all(r["status"] == "passed" for r in results)},
            warnings=warnings, trust=trust, engine="codegen", derivation=[step])

    def execute_code(self, artifact_id: str, inputs: dict) -> MathResult:
        record = self.artifacts.get(artifact_id)
        if record is None:
            return MathResult(ok=False, status="error", errors=[f"Unknown artifact_id: {artifact_id}"], engine="sandbox")
        try:
            outcome = execute_artifact(record, inputs, self.settings)
        except PermissionError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sandbox")
        except ValueError as exc:
            return MathResult(ok=False, status="error", errors=[str(exc)], engine="sandbox")
        step = self._record(DerivationStep(step_id=self._id("step"), operation="execute_code",
            inputs=[artifact_id, *[f"{k}={v}" for k, v in inputs.items()]],
            output=str(outcome.get("result")), engine="sandbox", trust=TrustLevel.NUMERIC))
        warnings = ["Sandboxed execution runs in an isolated subprocess with a timeout; "
                    "it is numeric, not symbolic, evidence."]
        return MathResult(ok=outcome["ok"], status="ok" if outcome["ok"] else "error",
            data=outcome, warnings=warnings, trust=TrustLevel.NUMERIC, engine="sandbox", derivation=[step])


_apply_output_budget(MathKernel)
