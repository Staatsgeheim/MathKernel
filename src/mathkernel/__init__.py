# =============================================================================
# MathKernel -   init  
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from ._version import __version__
from .kernel import MathKernel
from .parser import ambiguity_diagnostics, parse_math
from .settings import Settings
from .models import (MathResult, MathContext, SolutionSet, TrustLevel,
                     VerificationStatus, ResultStatus, OperationStatus,
                     ProblemPlan, PlanExecution, ObligationExecution)
from mathkernel_artifacts.evidence import (
    CertificateEvidence, ComputationEvidence, EmpiricalEvidence, EvidenceBundle,
    ModelEvidence, NumericalEvidence, ProofEvidence, extract_evidence,
    merge_evidence_bundles, reconcile_claim_evidence,
)
from .integral_transforms import (
    TransformConvention, TransformEngine, TransformProblem, TransformResult,
)
from .complex_analysis import (
    AnalyticContinuationResult, ArgumentPrincipleResult, BranchConvention,
    ComplexAnalysisEngine, ComplexDomain, ComplexFunction, ConformalMapResult,
    Contour, Singularity,
)
from .continuous_probability import (
    ConditionalDistribution, Distribution, JointDistribution, RandomVariable,
)
from .graph_theory import (
    DirectedGraph, Graph, GraphEdge, MultiGraph, WeightedGraph,
)
from .combinatorics import (
    CombinatorialClass, CombinatoricsEngine, GeneratingFunction,
    LinearRecurrence, RationalGeneratingFunction,
)
from .finite_groups import (
    FiniteAbelianGroup, FiniteGroup, GroupAction, GroupHomomorphism,
    PermutationGroup,
)
from .finite_algebra import (
    FieldElement, FiniteField, FiniteRing, Module, RingElement,
)

__all__ = ["MathKernel", "Settings", "parse_math", "ambiguity_diagnostics",
           "MathResult", "MathContext", "SolutionSet", "TrustLevel",
           "VerificationStatus", "ResultStatus", "OperationStatus",
           "ProblemPlan", "PlanExecution",
           "ObligationExecution", "EvidenceBundle", "ComputationEvidence",
           "ProofEvidence", "CertificateEvidence", "NumericalEvidence",
           "ModelEvidence", "EmpiricalEvidence", "extract_evidence",
           "merge_evidence_bundles", "reconcile_claim_evidence"]
__all__ += [
    "TransformConvention", "TransformEngine", "TransformProblem", "TransformResult",
    "BranchConvention", "ComplexAnalysisEngine", "ComplexDomain",
    "ComplexFunction", "Contour", "Singularity", "ArgumentPrincipleResult",
    "AnalyticContinuationResult", "ConformalMapResult", "Distribution",
    "RandomVariable", "JointDistribution", "ConditionalDistribution",
    "Graph", "DirectedGraph", "WeightedGraph", "MultiGraph", "GraphEdge",
    "CombinatorialClass", "CombinatoricsEngine", "GeneratingFunction",
    "LinearRecurrence", "RationalGeneratingFunction", "FiniteGroup",
    "PermutationGroup", "FiniteAbelianGroup", "GroupAction",
    "GroupHomomorphism", "FiniteRing", "FiniteField", "RingElement",
    "FieldElement", "Module",
]

from .signal_processing import ContinuousSignal, DiscreteSignal, Spectrum, Filter, FilterDesign
from .control_systems import TransferFunction, StateSpaceSystem, DiscreteControlSystem
from .control_analysis import FrequencyResponse, RootLocus, TimeResponse, ZeroPoleGain
from .optimization import OptimizationProblem, OptimizationCertificate
__all__ += ["ContinuousSignal", "DiscreteSignal", "Spectrum", "Filter", "FilterDesign", "TransferFunction",
            "ZeroPoleGain", "StateSpaceSystem", "DiscreteControlSystem", "FrequencyResponse",
            "RootLocus", "TimeResponse", "OptimizationProblem", "OptimizationCertificate"]

from .streaming import FilterState
from .control_design import TransferMatrix
from .optimization_proofs import MILPCertificate, MILPProofNode
__all__ += ["FilterState", "TransferMatrix", "MILPCertificate", "MILPProofNode"]

from .conic import ConicProblem, ConicCertificate, ConeBlock
from .quadratic_constraints import QuadraticallyConstrainedProblem, QuadraticCertificate, QuadraticConstraint
__all__ += ["ConicProblem", "ConicCertificate", "ConeBlock", "QuadraticallyConstrainedProblem", "QuadraticCertificate", "QuadraticConstraint"]

from .riccati import RiccatiCertificate
from .control_sequential import FiniteHorizonLQR, KalmanState
from .mpc import MPCPlan
__all__ += ["RiccatiCertificate", "FiniteHorizonLQR", "KalmanState", "MPCPlan"]

from .differential_geometry import (
    Chart, Connection, CoordinateMap, DifferentialForm, FormTerm,
    GeodesicSystem, GeometryTensor, JacobianMap, Manifold, Metric, TensorField,
)
__all__ += ["Manifold", "Chart", "Metric", "GeometryTensor", "Connection",
            "GeodesicSystem", "CoordinateMap", "JacobianMap", "TensorField",
            "FormTerm", "DifferentialForm"]

from .computational_geometry import (
    HalfSpace, Point, PointSet, Polygon, Polytope, Triangulation,
    VoronoiDiagram, VoronoiRay,
)
__all__ += ["Point", "PointSet", "Polygon", "HalfSpace", "Polytope",
            "Triangulation", "VoronoiRay", "VoronoiDiagram"]

from .algebraic_topology import (
    ChainComplex, CubicalComplex, Homology, HomologyGroup, SimplicialComplex,
)
__all__ += ["SimplicialComplex", "CubicalComplex", "ChainComplex",
            "HomologyGroup", "Homology"]

from .statistical_inference import (
    CovarianceMatrix, DescriptiveSummary, EmpiricalDistribution, GLMFit,
    CoxPHFit, CoxProportionalHazardsModel, GeneralizedLinearModel,
    KaplanMeierEstimate, NonparametricTestResult, ResamplingResult,
    StatisticalSample, SurvivalDataset, VariableSummary,
    TimeSeriesAnalysis, TimeSeriesDataset, TimeSeriesFit, TimeSeriesForecast,
    TimeSeriesModel,
)
__all__ += ["StatisticalSample", "VariableSummary", "DescriptiveSummary",
            "CovarianceMatrix", "EmpiricalDistribution"]
__all__ += ["GeneralizedLinearModel", "GLMFit"]
__all__ += ["NonparametricTestResult", "ResamplingResult"]
__all__ += ["SurvivalDataset", "KaplanMeierEstimate",
            "CoxProportionalHazardsModel", "CoxPHFit"]
__all__ += ["TimeSeriesDataset", "TimeSeriesAnalysis", "TimeSeriesModel",
            "TimeSeriesFit", "TimeSeriesForecast"]

from .stochastic_processes import (
    CTMCTransition, ContinuousTimeMarkovChain, FiniteDimensionalDistribution,
    GaussianProcess, GaussianProcessPosterior, PoissonProcess, WienerProcess,
)
__all__ += ["PoissonProcess", "WienerProcess", "GaussianProcess",
            "ContinuousTimeMarkovChain", "FiniteDimensionalDistribution",
            "GaussianProcessPosterior", "CTMCTransition"]

from .stochastic_differential_equations import (
    SDEConvergenceStudy, SDESimulation, StochasticDifferentialEquation,
)
__all__ += ["StochasticDifferentialEquation", "SDESimulation",
            "SDEConvergenceStudy"]

from .partial_differential_equations import (
    PDEBoundaryCondition, PDEClassification, PDECompatibilityCheck,
    PDECompatibilityReport, PDEEquation, PDEInitialCondition, PDEProblem,
    PDETerm,
)
__all__ += ["PDETerm", "PDEEquation", "PDEBoundaryCondition",
            "PDEInitialCondition", "PDEProblem", "PDEClassification",
            "PDECompatibilityCheck", "PDECompatibilityReport"]

from .weak_forms import (
    IntegrationByPartsStep, PDEFunctionSpace, PDEMeasure, WeakForm,
    WeakIntegralTerm,
)
__all__ += ["PDEFunctionSpace", "PDEMeasure", "WeakIntegralTerm",
            "IntegrationByPartsStep", "WeakForm"]

from .finite_elements import (
    BasisFunctionSet, FEMMesh, FiniteElementSpace, QuadratureRule,
    ReferenceElement,
)
__all__ += ["FEMMesh", "ReferenceElement", "BasisFunctionSet",
            "QuadratureRule", "FiniteElementSpace"]

from .finite_element_assembly import (
    AssembledSystem, EssentialConstraint, FEMSolution,
    LocalElementContribution, NaturalBoundaryContribution, SparseMatrixEntry,
)
__all__ += ["SparseMatrixEntry", "LocalElementContribution",
            "NaturalBoundaryContribution", "EssentialConstraint",
            "AssembledSystem", "FEMSolution"]

from .finite_element_adaptivity import (
    CellErrorIndicator, FEMConvergenceObservation, FEMErrorEstimate,
    MeshTransfer, RefinedMesh, RefinementMarking,
)
__all__ += ["CellErrorIndicator", "FEMErrorEstimate", "RefinementMarking",
            "RefinedMesh", "MeshTransfer", "FEMConvergenceObservation"]

from .stochastic_koopman import (
    FiniteJointLaw, FiniteMarkovKernel, RationalMarkovDilation,
    arbitrary_joint_contraction, compressed_transport_cumulant,
    compressed_transport_moment, deterministic_koopman_equivalence,
    latent_joint_tensor_value, markov_latent_tensor_value,
    markov_observation_contraction, observation_transfer_coefficients,
    pullback_observable, rational_basis_coordinates,
)
__all__ += [
    "FiniteJointLaw", "FiniteMarkovKernel", "RationalMarkovDilation",
    "arbitrary_joint_contraction", "latent_joint_tensor_value",
    "markov_latent_tensor_value", "markov_observation_contraction",
    "compressed_transport_moment", "compressed_transport_cumulant",
    "deterministic_koopman_equivalence", "pullback_observable",
    "rational_basis_coordinates", "observation_transfer_coefficients",
]

from .phylogenetic_tensor import (
    EdgeFlatteningCertificate,
    FiniteMarkovTree,
    FiniteObservationChannel,
    MarkovTreeEdge,
    ObservationFlatteningCertificate,
    TensorFlattening,
    branching_observation_contraction,
    edge_flattening_certificate,
    exact_matrix_rank,
    identity_matrix,
    inverse_matrix,
    kronecker_product,
    matrix_multiply,
    matrix_transpose,
    minimum_collectively_injective_channel_sets,
    observation_flattening_certificate,
    recover_latent_joint_law,
    singular_value_observation_bounds,
    tensor_flattening,
    transform_joint_law,
)
__all__ += [
    "FiniteObservationChannel", "MarkovTreeEdge", "FiniteMarkovTree",
    "TensorFlattening", "EdgeFlatteningCertificate",
    "ObservationFlatteningCertificate", "matrix_transpose",
    "matrix_multiply", "identity_matrix", "exact_matrix_rank",
    "inverse_matrix", "kronecker_product", "tensor_flattening",
    "transform_joint_law", "branching_observation_contraction",
    "edge_flattening_certificate", "observation_flattening_certificate",
    "recover_latent_joint_law", "minimum_collectively_injective_channel_sets",
    "singular_value_observation_bounds",
]

from .phylogenetic_inference import (
    ChannelRecoveryResult,
    FisherCovarianceResult,
    LatentClassFit,
    QuartetSplitScore,
    RankWaldResult,
    RegularizationSelection,
    fit_known_channel_distribution,
    fit_nonnegative_rank,
    fit_ridge_channel_distribution,
    flatten_count_tensor,
    known_channel_fisher_covariance,
    local_observation_operator,
    multinomial_covariance,
    normalize_probability_vector,
    project_probability_simplex,
    propagate_linear_covariance,
    rank_tail_frobenius,
    rank_wald_test,
    score_quartet_splits,
    select_known_channel_regularization,
    split_score_minimizers,
    split_score_winners,
)
__all__ += [
    "FisherCovarianceResult", "ChannelRecoveryResult",
    "RegularizationSelection", "LatentClassFit", "RankWaldResult",
    "QuartetSplitScore", "normalize_probability_vector",
    "project_probability_simplex", "local_observation_operator",
    "multinomial_covariance", "propagate_linear_covariance",
    "known_channel_fisher_covariance", "fit_known_channel_distribution",
    "fit_ridge_channel_distribution", "select_known_channel_regularization",
    "fit_nonnegative_rank", "rank_tail_frobenius", "rank_wald_test",
    "flatten_count_tensor", "score_quartet_splits",
    "split_score_minimizers", "split_score_winners",
]

from .phylogenetic_benchmark import (
    AlignmentPartition,
    BenchmarkProtocol,
    BootstrapMethodSummary,
    CANONICAL_QUARTET_SPLITS,
    DatasetAudit,
    DatasetManifest,
    FrozenBenchmarkManifest,
    NUCLEOTIDE_ALPHABET,
    NUCLEOTIDE_CODE,
    NewickNode,
    NewickTree,
    QuartetBenchmarkResult,
    QuartetCase,
    QuartetSiteData,
    SequenceAlignment,
    SiteRange,
    audit_dataset,
    canonical_json,
    canonical_sha256,
    circular_block_bootstrap_indices,
    complete_nucleotide_quartet,
    count_quartet_states,
    filtered_partition_groups,
    infer_alignment_format,
    load_alignment,
    load_newick,
    logdet_scores,
    pairwise_logdet_distance,
    pairwise_p_distance,
    parse_fasta,
    parse_newick,
    parse_nexus,
    parse_phylip,
    partition_block_bootstrap_indices,
    partition_stratified_bootstrap_indices,
    p_distance_scores,
    rank_tail_scores,
    resample_indices,
    run_quartet_benchmark,
    sample_reference_quartets,
    score_margin,
    score_minimizers,
    score_quartet_methods,
    sha256_bytes,
    sha256_file,
    site_bootstrap_indices,
    split_name,
    summarize_corpus,
)
__all__ += [
    "NUCLEOTIDE_ALPHABET", "NUCLEOTIDE_CODE", "CANONICAL_QUARTET_SPLITS",
    "SiteRange", "AlignmentPartition", "SequenceAlignment", "QuartetSiteData",
    "NewickNode", "NewickTree", "DatasetManifest", "BenchmarkProtocol",
    "FrozenBenchmarkManifest", "QuartetCase", "DatasetAudit",
    "BootstrapMethodSummary", "QuartetBenchmarkResult", "sha256_bytes",
    "sha256_file", "canonical_json", "canonical_sha256", "parse_fasta",
    "parse_phylip", "parse_nexus", "infer_alignment_format", "load_alignment",
    "complete_nucleotide_quartet", "parse_newick", "load_newick",
    "sample_reference_quartets", "audit_dataset", "split_name",
    "count_quartet_states", "rank_tail_scores", "pairwise_p_distance",
    "p_distance_scores", "pairwise_logdet_distance", "logdet_scores",
    "score_quartet_methods", "score_minimizers", "score_margin",
    "site_bootstrap_indices", "circular_block_bootstrap_indices",
    "filtered_partition_groups", "partition_stratified_bootstrap_indices",
    "partition_block_bootstrap_indices", "resample_indices",
    "run_quartet_benchmark", "summarize_corpus",
]

from .relation_detection import (
    ConditionalExpectationSpectrum,
    DetectionSampleBounds,
    ExactModeTransfer,
    ExactPureInteractionCertificate,
    ModeTransfer,
    PureInteractionInformation,
    binary_parity_information,
    binary_symmetric_channel,
    conditional_expectation_spectrum,
    exact_chi_square,
    exact_mode_transfer,
    exact_pure_interaction_certificate,
    exact_pure_interaction_law,
    joint_conditionally_independent_channel,
    mode_transfer,
    numerical_chi_square,
    numerical_pure_interaction_law,
    pure_interaction_information,
    pure_relation_sample_bounds,
    push_joint_law_through_local_channels,
    standardize_centered_mode,
)
__all__ += [
    "ConditionalExpectationSpectrum", "ModeTransfer", "ExactModeTransfer",
    "ExactPureInteractionCertificate", "PureInteractionInformation",
    "DetectionSampleBounds", "standardize_centered_mode",
    "conditional_expectation_spectrum", "mode_transfer",
    "exact_mode_transfer", "exact_pure_interaction_law",
    "exact_pure_interaction_certificate", "exact_chi_square",
    "numerical_pure_interaction_law",
    "push_joint_law_through_local_channels", "numerical_chi_square",
    "pure_interaction_information", "pure_relation_sample_bounds",
    "binary_symmetric_channel", "binary_parity_information",
    "joint_conditionally_independent_channel",
]

from .relation_subspace import (
    DependentRelationInformation,
    RelationCollisionCertificate,
    RelationSubspaceTransfer,
    SensorOptimizationResult,
    SensorSubsetEvaluation,
    analytic_ar1_long_run_covariance,
    chi_square_divergence,
    collision_certificate,
    dependent_relation_information,
    direction_information_retention,
    direction_visibility,
    evaluate_sensor_subset,
    linear_relation_law,
    newey_west_long_run_covariance,
    optimize_sensor_subsets,
    push_finite_law,
    relation_chi_square,
    relation_subspace_from_partition,
    relation_subspace_transfer,
)
__all__ += [
    "RelationSubspaceTransfer", "RelationCollisionCertificate",
    "SensorSubsetEvaluation", "SensorOptimizationResult",
    "DependentRelationInformation", "relation_subspace_transfer",
    "relation_subspace_from_partition",
    "direction_information_retention", "direction_visibility",
    "linear_relation_law", "push_finite_law", "chi_square_divergence",
    "relation_chi_square", "collision_certificate",
    "evaluate_sensor_subset", "optimize_sensor_subsets",
    "newey_west_long_run_covariance", "dependent_relation_information",
    "analytic_ar1_long_run_covariance",
]

from .information_geometry import (
    DirectionTestingBounds,
    SimplexTangentGeometry,
    SubspaceMinimaxBounds,
    VisibilitySpectrumBootstrap,
    bootstrap_empirical_relation_spectrum,
    direction_testing_bounds,
    mixture_chart_jacobian,
    push_simplex_tangents,
    score_subspace_geometry,
    scores_to_tangents,
    simplex_fisher_inner,
    simplex_tangent_geometry,
    softmax_chart_jacobian,
    subspace_minimax_bounds,
    tangent_information_retention,
    tangents_to_scores,
)
__all__ += [
    "SimplexTangentGeometry", "DirectionTestingBounds",
    "SubspaceMinimaxBounds", "VisibilitySpectrumBootstrap",
    "simplex_fisher_inner", "scores_to_tangents",
    "tangents_to_scores", "push_simplex_tangents",
    "mixture_chart_jacobian", "softmax_chart_jacobian",
    "simplex_tangent_geometry", "score_subspace_geometry",
    "tangent_information_retention", "direction_testing_bounds",
    "subspace_minimax_bounds", "bootstrap_empirical_relation_spectrum",
]

from .composite_relation_inference import (
    CompositeDetectionBounds,
    CompositeScoreGeometry,
    CompositeScoreTestResult,
    EigenspaceBootstrapUncertainty,
    EigenspaceClusterRegion,
    NuisanceAdjustedGeometry,
    RobustCompositeDetectionBound,
    StudentizedSpectrumBootstrap,
    bootstrap_eigenspace_regions,
    composite_detection_bounds,
    composite_score_geometry,
    composite_score_test,
    estimate_circular_block_length,
    estimate_relation_block_length,
    nuisance_adjusted_score_geometry,
    nuisance_adjusted_tangent_geometry,
    nuisance_composite_score_geometry,
    robust_composite_detection_bound,
    studentized_bootstrap_relation_spectrum,
)
__all__ += [
    "CompositeScoreGeometry", "CompositeScoreTestResult",
    "CompositeDetectionBounds", "NuisanceAdjustedGeometry",
    "RobustCompositeDetectionBound", "EigenspaceClusterRegion",
    "EigenspaceBootstrapUncertainty", "StudentizedSpectrumBootstrap",
    "composite_score_geometry", "nuisance_composite_score_geometry",
    "composite_score_test", "composite_detection_bounds",
    "nuisance_adjusted_tangent_geometry", "nuisance_adjusted_score_geometry",
    "robust_composite_detection_bound", "bootstrap_eigenspace_regions",
    "estimate_circular_block_length", "estimate_relation_block_length",
    "studentized_bootstrap_relation_spectrum",
]

# Phase 10: model-scoped quadratic inference and learned nuisances.
from .robust_relation_inference import (
    QuadraticMinimaxBounds, LongRunCovarianceEstimate, CrossFittedRelations,
    quadratic_minimax_bounds, gaussian_quadratic_test, quadratic_u_test,
    prewhitened_long_run_covariance, quadratic_moment_test, relation_folds,
    crossfit_nuisance_projection, crossfit_residual_relations,
)
__all__ += [
    "QuadraticMinimaxBounds", "LongRunCovarianceEstimate", "CrossFittedRelations",
    "quadratic_minimax_bounds", "gaussian_quadratic_test", "quadratic_u_test",
    "prewhitened_long_run_covariance", "quadratic_moment_test", "relation_folds",
    "crossfit_nuisance_projection", "crossfit_residual_relations",
]

from .python_api import ExpressionHandle, KernelOperationError, complete_result
__all__ += ["__version__", "ExpressionHandle", "KernelOperationError", "complete_result"]
