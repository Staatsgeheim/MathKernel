# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Capability registration and restricted MathIR boundary for engineering domains."""
from __future__ import annotations
import math
from .capabilities import Capability
from .engineering import cap_trust, positive_integer, unsupported
from .models import TrustLevel

TYPES = {
    "signal": "DiscreteSignal", "discretesignal": "DiscreteSignal", "discrete_signal": "DiscreteSignal",
    "continuoussignal": "ContinuousSignal", "continuous_signal": "ContinuousSignal",
    "filterdesign": "FilterDesign", "filter_design": "FilterDesign",
    "spectrum": "Spectrum", "filter": "Filter", "transferfunction": "TransferFunction",
    "transfer_function": "TransferFunction", "statespacesystem": "StateSpaceSystem",
    "state_space_system": "StateSpaceSystem", "optimizationproblem": "OptimizationProblem",
    "zeropolegain": "ZeroPoleGain", "zero_pole_gain": "ZeroPoleGain", "zpk": "ZeroPoleGain",
    "discretecontrolsystem": "DiscreteControlSystem", "discrete_control_system": "DiscreteControlSystem",
    "optimization_problem": "OptimizationProblem", "optimizationcertificate": "OptimizationCertificate",
    "optimization_certificate": "OptimizationCertificate",
    "milpcertificate": "MILPCertificate", "milp_certificate": "MILPCertificate",
}

from .optimization_extended_adapter import TYPES as EXTENDED_TYPES, OPERATIONS as EXTENDED_OPERATIONS
TYPES.update(EXTENDED_TYPES)
TYPES.update({"riccaticertificate":"RiccatiCertificate", "riccati_certificate":"RiccatiCertificate"})
from .geometry_adapter import TYPES as GEOMETRY_TYPES, OPERATIONS as GEOMETRY_OPERATIONS
TYPES.update(GEOMETRY_TYPES)
from .topology_adapter import TYPES as TOPOLOGY_TYPES, OPERATIONS as TOPOLOGY_OPERATIONS
TYPES.update(TOPOLOGY_TYPES)
from .statistics_adapter import TYPES as STATISTICS_TYPES, OPERATIONS as STATISTICS_OPERATIONS
TYPES.update(STATISTICS_TYPES)
from .pde_adapter import TYPES as PDE_TYPES, OPERATIONS as PDE_OPERATIONS
TYPES.update(PDE_TYPES)

# All mathematical parameters are declared, including nested certificate entries.
OPERATIONS = {
    "ContinuousSignal": {"sample": {"sample_rate": "MathIR Hz", "count": "positive integer"}},
    "DiscreteSignal": {
        "dft": {"normalization": "backward|forward|ortho"},
        "convolution": {"other_id": "DiscreteSignal object id"},
        "correlation": {"other_id": "DiscreteSignal object id"},
        "autocorrelation": {}, "cross_spectrum": {"other_id": "DiscreteSignal object id"}, "window": {"kind": "hann|hamming|boxcar", "periodic": "bool"},
        "stft": {"size": "integer", "overlap": "integer", "window": "hann|hamming|boxcar"},
        "resample": {"up": "integer", "down": "integer"},
    },
    "FilterDesign": {"design": {}},
    "Spectrum": {"idft": {}},
    "FilterState": {"process": {"signal_id": "DiscreteSignal object id"}},
    "Filter": {"initial_state": {"start": "MathIR seconds?", "unit": "unit string?"}, "apply_signal": {"signal_id": "DiscreteSignal object id"}, "to_transfer_function": {}},
    "TransferFunction": {"to_state_space": {}, "to_zero_pole_gain": {}, "stability": {}, "to_filter": {},
        "poles": {}, "zeros": {}, "bode": {"frequencies": "MathIR[] rad/s"},
        "nyquist": {"frequencies": "MathIR[] nonnegative rad/s"},
        "root_locus": {"gains": "MathIR[] nonnegative and nondecreasing"},
        "frequency_response": {"frequencies": "MathIR[] rad/s"}, "step_response": {}, "impulse_response": {},
        "series": {"other_id": "TransferFunction object id"}, "feedback": {"other_id": "TransferFunction object id"}},
    "ZeroPoleGain": {"to_transfer_function": {}, "poles": {}, "zeros": {},
        "bode": {"frequencies": "MathIR[] rad/s"}, "nyquist": {"frequencies": "MathIR[] nonnegative rad/s"}},
    "TransferMatrix": {"entry": {"output": "zero-based integer", "input": "zero-based integer"}},
    "StateSpaceSystem": {"to_transfer_function": {}, "to_discrete_control": {}, "controllability": {}, "observability": {}, "stability": {},
        "poles": {}, "zeros": {}, "bode": {"frequencies": "MathIR[] rad/s"}, "nyquist": {"frequencies": "MathIR[] nonnegative rad/s"},
        "coefficient_units": {}, "discretize": {"sample_time": "MathIR seconds", "method": "zoh|bilinear"},
        "frequency_response": {"frequencies": "MathIR[] rad/s"},
        "state_feedback": {"gain": "MathIR[][] inputs by states"},
        "place_poles": {"poles": "MathIR[]", "max_iterations": "integer", "rtol": "positive tolerance"},
        "observer": {"poles": "MathIR[]", "max_iterations": "integer", "rtol": "positive tolerance"}},
    "DiscreteControlSystem": {"to_state_space": {}, "to_transfer_function": {}, "controllability": {}, "observability": {}, "stability": {},
        "poles": {}, "zeros": {}, "frequency_response": {"frequencies": "MathIR[] rad/s"},
        "bode": {"frequencies": "MathIR[] rad/s"}, "nyquist": {"frequencies": "MathIR[] nonnegative rad/s"}},
    "OptimizationProblem": {"certify_milp": {"max_nodes": "positive integer"},
        "verify_milp_certificate": {"certificate_id": "MILPCertificate object id?", "certificate": "nested MathIR witness?"},
        "solve": {"max_iterations": "integer", "tolerance": "positive float",
        "reconstruction_denominator": "positive integer"},
        "verify_certificate": {"certificate_id": "OptimizationCertificate object id?", "certificate": "nested MathIR witness?"}},
}


from .optimization_extended_adapter import witness_schema
_linear_witness = witness_schema('primal', 'inequality_dual', 'equality_dual', 'ray')
OPERATIONS['OptimizationProblem']['verify_certificate']['certificate'] = _linear_witness
OPERATIONS['OptimizationProblem']['verify_milp_certificate']['certificate'] = {
    'type': 'object', 'properties': {'incumbent': 'MathIR[]', 'nodes': {'type': 'array',
        'items': {'type': 'object', 'properties': {'split': 'MathIR', 'certificate': _linear_witness}}}}}
OPERATIONS.update(EXTENDED_OPERATIONS)
OPERATIONS.update(GEOMETRY_OPERATIONS)
OPERATIONS.update(TOPOLOGY_OPERATIONS)
OPERATIONS.update(STATISTICS_OPERATIONS)
OPERATIONS.update(PDE_OPERATIONS)
OPERATIONS['OptimizationProblem']['to_conic'] = {}
for _name, _weights in (("lqr", ("Q", "R")), ("kalman", ("W", "V"))):
    OPERATIONS['StateSpaceSystem'][_name] = {
        **{key: 'MathIR[][]' for key in _weights},
        'certificate_id': 'RiccatiCertificate object id?',
        'certificate': {'type':'object','properties':{'P':'MathIR[][]'}},
        'tolerance': 'positive finite tolerance', 'reconstruction_denominator': 'positive integer'}


from .control_sequential_adapter import OPERATIONS as SEQUENTIAL_OPERATIONS
for _typ, _operations in SEQUENTIAL_OPERATIONS.items():
    OPERATIONS.setdefault(_typ, {}).update(_operations)
from .mpc_adapter import OPERATIONS as MPC_OPERATIONS
for _typ, _operations in MPC_OPERATIONS.items():
    OPERATIONS.setdefault(_typ, {}).update(_operations)


MODE_OPERATIONS = {"mpc", "lqg", "finite_lqr", "kalman_state", "lqr", "kalman","dft", "idft", "convolution", "correlation", "autocorrelation", "cross_spectrum",
                   "window", "apply_signal", "stft", "resample", "sample", "design", "frequency_response", "bode", "nyquist", "root_locus", "to_zero_pole_gain", "solve", "certify_milp", "initial_state", "discretize", "place_poles", "observer"}


def parameter_schema(operation, schema):
    out = {"context_id": "context id?", **schema}
    if operation in MODE_OPERATIONS and not (
            operation == "solve" and "condition_limit" in schema):
        out["mode"] = "exact|numeric (exact default; numeric operations require explicit choice)"
    if operation in {"dft", "idft", "cross_spectrum"}:
        out["precision"] = "integer bits (53 default; 54..4096 uses mpmath)"
    return out


def register(registry):
    derived_outputs = {"mpc":"MPCPlan", "finite_lqr": "FiniteHorizonLQR", "kalman_state": "KalmanState",
        "sample": "DiscreteSignal", "to_zero_pole_gain": "ZeroPoleGain", "to_discrete_control": "DiscreteControlSystem",
        "bode": "FrequencyResponse", "nyquist": "FrequencyResponse", "root_locus": "RootLocus",
        "update": "KalmanState", "predict": "KalmanState", "to_conic": "ConicProblem","dft": "Spectrum", "idft": "DiscreteSignal", "convolution": "DiscreteSignal",
        "correlation": "DiscreteSignal", "autocorrelation": "DiscreteSignal", "cross_spectrum": "Spectrum",
        "window": "DiscreteSignal", "resample": "DiscreteSignal", "apply_signal": "DiscreteSignal",
        "discretize": "StateSpaceSystem", "place_poles": "StateSpaceSystem", "observer": "StateSpaceSystem",
        "state_feedback": "StateSpaceSystem", "entry": "TransferFunction", "initial_state": "FilterState", "process": "DiscreteSignal", "certify_milp": "MILPCertificate", "design": "Filter", "to_filter": "Filter", "to_state_space": "StateSpaceSystem",
        "to_transfer_function": "TransferFunction", "series": "TransferFunction", "feedback": "TransferFunction",
        "step_response": "TimeResponse", "impulse_response": "TimeResponse"}
    from .geometry_adapter import DERIVED_OUTPUTS as GEOMETRY_DERIVED_OUTPUTS
    derived_outputs.update(GEOMETRY_DERIVED_OUTPUTS)
    from .topology_adapter import DERIVED_OUTPUTS as TOPOLOGY_DERIVED_OUTPUTS
    derived_outputs.update(TOPOLOGY_DERIVED_OUTPUTS)
    from .statistics_adapter import DERIVED_OUTPUTS as STATISTICS_DERIVED_OUTPUTS
    derived_outputs.update(STATISTICS_DERIVED_OUTPUTS)
    from .pde_adapter import DERIVED_OUTPUTS as PDE_DERIVED_OUTPUTS
    derived_outputs.update(PDE_DERIVED_OUTPUTS)
    for typ, operations in OPERATIONS.items():
        domain = ("signal" if typ in {"ContinuousSignal", "DiscreteSignal", "Spectrum", "Filter", "FilterDesign", "FilterState"}
                  else "optimization" if typ in {"OptimizationProblem", "ConicProblem", "QuadraticallyConstrainedProblem"}
                  else "geometry" if typ in GEOMETRY_OPERATIONS or typ in TOPOLOGY_OPERATIONS
                  else "statistics" if typ in STATISTICS_OPERATIONS
                  else "pde" if typ in PDE_OPERATIONS else "control")
        for operation, schema in operations.items():
            numeric_only = operation in {"design", "stft", "resample", "frequency_response", "bode", "nyquist", "root_locus"}
            levels = ("numeric", "unknown") if numeric_only else ("exact", "symbolic", "numeric", "unknown")
            engines = ("sympy",)
            if typ in TOPOLOGY_OPERATIONS:
                levels = ("exact",)
                engines = ("exact_integer", "sympy")
            if typ in STATISTICS_OPERATIONS:
                levels = ("exact", "numeric", "numeric_high_precision")
                engines = ("statistical_inference", "sympy", "numpy")
                if operation == "fit" and typ == "GeneralizedLinearModel":
                    levels = ("exact", "numeric")
                elif operation == "bootstrap":
                    levels = ("empirical",)
                elif operation == "permutation_test":
                    levels = ("exact", "empirical")
                elif typ in {"SurvivalDataset", "KaplanMeierEstimate"}:
                    levels = ("exact", "numeric", "numeric_high_precision")
                    engines = ("survival_analysis", "sympy", "mpmath")
                elif typ in {"CoxProportionalHazardsModel", "CoxPHFit"}:
                    levels = ("numeric",)
                    engines = ("survival_analysis", "numpy")
                elif typ in {"TimeSeriesDataset", "TimeSeriesAnalysis",
                              "TimeSeriesModel", "TimeSeriesFit",
                              "TimeSeriesForecast"}:
                    levels = (("exact", "numeric") if operation in {"acf", "pacf", "verify"}
                              and typ in {"TimeSeriesDataset", "TimeSeriesAnalysis"}
                              else ("numeric",))
                    engines = (("time_series", "sympy", "numpy")
                               if operation in {"acf", "pacf", "verify"}
                               else ("time_series", "numpy", "scipy", "mpmath"))
                elif typ in {"PoissonProcess", "WienerProcess",
                              "FiniteDimensionalDistribution"}:
                    levels = ("exact", "symbolic", "numeric")
                    engines = ("stochastic_processes", "sympy")
                elif typ in {"GaussianProcess", "GaussianProcessPosterior"}:
                    levels = (("numeric",) if operation == "condition" or
                              typ == "GaussianProcessPosterior" else
                              ("exact", "symbolic", "numeric"))
                    engines = (("stochastic_processes", "numpy", "scipy")
                               if operation == "condition" or typ == "GaussianProcessPosterior"
                               else ("stochastic_processes", "sympy", "numpy"))
                elif typ in {"ContinuousTimeMarkovChain", "CTMCTransition"}:
                    levels = (("numeric",) if operation in {"transition_matrix", "distribution"}
                              or typ == "CTMCTransition" else ("exact", "numeric"))
                    engines = (("stochastic_processes", "numpy", "scipy")
                               if "numeric" in levels and "exact" not in levels
                               else ("stochastic_processes", "sympy"))
                elif typ in {"StochasticDifferentialEquation", "SDESimulation",
                              "SDEConvergenceStudy"}:
                    levels = (("exact", "symbolic", "numeric")
                              if typ == "StochasticDifferentialEquation" and operation == "verify"
                              else ("empirical",))
                    engines = (("stochastic_differential_equations", "sympy")
                               if typ == "StochasticDifferentialEquation" and operation == "verify"
                               else ("stochastic_differential_equations", "numpy"))
            if typ in PDE_OPERATIONS:
                levels = ("exact", "symbolic", "numeric")
                engines = ("partial_differential_equations", "sympy")
                if operation == "compare" or typ == "FEMConvergenceObservation":
                    levels = ("empirical",)
                if typ in {"FEMMesh", "ReferenceElement", "BasisFunctionSet",
                           "QuadratureRule", "FiniteElementSpace",
                           "AssembledSystem", "FEMSolution", "FEMErrorEstimate",
                           "RefinementMarking", "RefinedMesh", "MeshTransfer",
                           "FEMConvergenceObservation"}:
                    engines = (("finite_element_adaptivity", "sympy")
                               if typ in {"FEMErrorEstimate", "RefinementMarking",
                                          "RefinedMesh", "MeshTransfer",
                                          "FEMConvergenceObservation"} or
                                  operation in {"estimate_error", "mark", "refine", "compare"}
                               else ("finite_element_assembly", "sympy", "scipy")
                               if typ in {"AssembledSystem", "FEMSolution"} or
                                  operation in {"assemble", "solve"}
                               else ("finite_elements", "sympy"))
            if operation == "to_simplicial_complex":
                # Crossing from geometric predicates into finite combinatorics is
                # sound only after an exact triangulation has been re-verified.
                levels = ("exact",)
            if operation in MODE_OPERATIONS and not (
                    domain == "pde" and typ == "AssembledSystem"):
                engines += ("scipy", "numpy")
            if operation in {"dft", "idft", "cross_spectrum"}:
                engines = ("sympy", "numpy", "mpmath")
                levels += ("numeric_high_precision",)
            derived_output = derived_outputs.get(operation)
            if typ == "CoxProportionalHazardsModel" and operation == "fit":
                derived_output = "CoxPHFit"
            elif typ == "TimeSeriesModel" and operation == "fit":
                derived_output = "TimeSeriesFit"
            output = ("EngineeringResult",) + ((derived_output,) if derived_output else ())
            if typ in {"StateSpaceSystem", "DiscreteControlSystem"} and operation == "to_transfer_function":
                output += ("TransferMatrix",)
            if typ == "TransferFunction" and operation == "frequency_response":
                output += ("FrequencyResponse",)
            if operation == "solve" and typ in {"ConicProblem", "QuadraticallyConstrainedProblem"}:
                output += ("ConicCertificate" if typ == "ConicProblem" else "QuadraticCertificate",)
                if typ == "ConicProblem":
                    engines = ("clarabel", "sympy")
            if operation == "process":
                output += ("FilterState",)
            if operation in {"lqr", "kalman", "lqg"}:
                output += ("RiccatiCertificate", "StateSpaceSystem")
            if operation == "mpc":
                output += ("OptimizationProblem", "OptimizationCertificate", "StateSpaceSystem")
            verification_methods = (("defining_identity", "residual_check") if domain == "signal" else
                ("original_data_kkt", "primal_dual_gap", "exact_psd", "farkas_witness", "recession_ray", "integer_partition_tree") if domain == "optimization" else
                ("polynomial_identity", "hurwitz_schur_criteria", "rank", "resolvent_residual", "characteristic_polynomial"))
            if operation in {"lqr", "kalman", "lqg"}:
                verification_methods = ("riccati_identity", "exact_psd", "hurwitz_schur_criteria")
            if operation in SEQUENTIAL_OPERATIONS.get(typ, {}):
                verification_methods = ("bellman_identity", "covariance_identity", "separation_identity")
                engines = ("sympy", "numpy", "scipy")
            if operation in MPC_OPERATIONS.get(typ, {}):
                verification_methods = ("trajectory_constraints", "original_data_kkt", "farkas_witness", "terminal_box_invariance")
                engines = ("sympy", "numpy", "scipy")
            if typ == "ConicProblem":
                verification_methods = ("product_cone_membership", "primal_dual_gap", "farkas_witness", "recession_ray", "exact_psd")
            elif typ == "QuadraticallyConstrainedProblem":
                verification_methods = ("quadratic_lagrangian", "primal_dual_gap", "exact_psd")
            elif domain == "geometry":
                verification_methods = ("inverse_identity", "metric_compatibility",
                    "torsion_free", "riemann_symmetries", "first_bianchi")
                geometry_checks = {
                    "jacobian": ("symbolic_differentiation", "jacobian_nonsingular"),
                    "verify": ("coordinate_composition", "jacobian_nonsingular"),
                    "covariant_derivative": ("levi_civita_formula", "metric_compatibility"),
                    "lie_derivative": ("coordinate_lie_formula",),
                    "wedge": ("antisymmetric_shuffle", "graded_commutativity"),
                    "exterior_derivative": ("coordinate_derivative", "d_squared_zero"),
                    "interior_product": ("coordinate_contraction",),
                    "pullback": ("jacobian_minors", "pullback_commutes_with_d"),
                    "hodge_star": ("metric_volume", "raised_components"),
                    "distance_to": ("squared_distance_identity",),
                    "orientation": ("adaptive_determinant_filter",),
                    "incircle": ("adaptive_incircle_filter",),
                    "segment_intersection": ("adaptive_orientation_intersection",),
                    "convex_hull": ("monotone_chain", "containment_witness"),
                    "nearest_neighbor": ("squared_distance_scan",),
                    "delaunay": ("empty_circumcircle", "planar_triangle_count"),
                    "voronoi": ("delaunay_dual",),
                    "contains": ("winding_or_halfspace_predicates",),
                    "intersection": ("convex_clipping", "containment_witness"),
                    "triangulate": ("ear_clipping", "area_partition"),
                    "to_simplicial_complex": ("triangulation_topology",
                                               "boundary_squared_zero"),
                }
                verification_methods = geometry_checks.get(operation, verification_methods)
                if operation == "verify" and typ == "Polygon":
                    verification_methods = ("edge_intersections", "turn_checks")
                elif operation == "verify" and typ == "Polytope":
                    verification_methods = ("halfspace_feasibility",)
                elif operation == "verify" and typ == "Triangulation":
                    verification_methods = ("orientation", "edge_incidence",
                                            "nonoverlapping_interiors")
                elif typ in TOPOLOGY_OPERATIONS:
                    topology_checks = {
                        "verify": ("boundary_squared_zero", "face_closure"),
                        "chain_complex": ("oriented_cellular_boundary", "boundary_squared_zero"),
                        "boundary_matrix": ("basis_order", "matrix_shape"),
                        "homology": ("smith_kernel_quotient", "exact_field_rank",
                                     "euler_poincare"),
                        "euler_characteristic": ("alternating_chain_rank",),
                    }
                    verification_methods = topology_checks[operation]
            elif domain == "statistics":
                statistical_checks = {
                    "describe": ("moment_identity", "order_statistics"),
                    "covariance": ("centered_cross_product", "matrix_symmetry"),
                    "empirical_distribution": ("frequency_reconciliation", "probability_normalization"),
                    "evidence_profile": ("claim_scope_audit", "population_non_inference"),
                    "fit": ("design_rank", "normal_equations", "score_residual",
                            "irls_convergence", "observed_information", "separation_check"),
                    "verify": ("stored_result_recomputation",),
                    "diagnostics": ("deviance", "covariance_symmetry", "conditioning"),
                    "predict": ("row_dimension", "finite_conditional_mean"),
                    "mann_whitney": ("rank_sum", "exact_permutation", "tie_correction"),
                    "wilcoxon": ("signed_rank_sum", "exact_sign_enumeration", "tie_correction"),
                    "kruskal_wallis": ("rank_partition", "tie_correction", "exact_permutation"),
                    "ks_2samp": ("ecdf_distance", "exact_permutation", "kolmogorov_limit"),
                    "spearman": ("average_ranks", "exact_permutation", "student_t_approximation"),
                    "kendall": ("concordant_discordant_pairs", "exact_permutation", "tie_correction"),
                    "permutation_test": ("label_enumeration", "seeded_monte_carlo", "draw_reconciliation"),
                    "bootstrap": ("seeded_empirical_resampling", "percentile_interval", "stream_replay"),
                    "kaplan_meier": ("risk_set_accounting", "product_limit_identity",
                                      "greenwood_log_log_interval"),
                    "survival_at": ("right_continuous_step_lookup",),
                    "predict_partial_hazard": ("row_dimension", "finite_positive_hazard"),
                    "acf": ("centered_cross_product", "lag_zero_identity"),
                    "pacf": ("durbin_levinson", "lag_zero_identity"),
                    "stationarity_test": ("adf_regression", "asymptotic_critical_values"),
                    "forecast": ("recursive_conditional_mean", "psi_weight_variance",
                                 "interval_ordering"),
                    "pmf": ("poisson_parameter", "mass_formula"),
                    "moments": ("poisson_mean_variance",),
                    "increment_distribution": ("increment_parameter", "positive_interval"),
                    "finite_dimensional": ("mean_vector", "covariance_symmetry", "positive_semidefinite"),
                    "condition": ("cholesky_factorization", "posterior_covariance_psd", "conditioning_guard"),
                    "transition_matrix": ("generator_exponential", "row_stochasticity", "semigroup_identity"),
                    "distribution": ("initial_law_propagation", "probability_normalization"),
                    "stationary_distribution": ("generator_left_nullspace", "probability_normalization"),
                    "simulate": ("seeded_stream_replay", "time_grid", "finite_path_check"),
                    "convergence_study": ("coupled_brownian_paths", "nested_step_grid",
                                          "terminal_rms_difference"),
                    "path": ("stored_path_slice", "index_bounds"),
                    "terminal_values": ("stored_terminal_slice", "index_bounds"),
                }
                verification_methods = statistical_checks[operation]
                if operation == "verify" and typ == "GeneralizedLinearModel":
                    verification_methods = ("response_domain", "design_rank",
                                            "link_family_compatibility")
                elif operation == "verify" and typ == "GLMFit":
                    verification_methods = ("response_domain", "design_rank",
                                            "score_recomputation")
                elif operation == "verify" and typ == "NonparametricTestResult":
                    verification_methods = ("statistic_recomputation",
                                            "p_value_replay", "source_ancestry")
                elif operation == "verify" and typ == "ResamplingResult":
                    verification_methods = ("seed_or_enumeration_replay",
                                            "draw_reconciliation", "source_ancestry")
                elif operation == "verify" and typ == "SurvivalDataset":
                    verification_methods = ("time_domain", "binary_event_indicator",
                                            "risk_interval_validation")
                elif operation == "verify" and typ == "KaplanMeierEstimate":
                    verification_methods = ("risk_set_recomputation",
                                            "product_limit_recomputation",
                                            "uncertainty_recomputation")
                elif operation == "verify" and typ == "CoxProportionalHazardsModel":
                    verification_methods = ("predictor_rank", "event_count",
                                            "risk_interval_validation")
                elif operation == "verify" and typ == "CoxPHFit":
                    verification_methods = ("partial_likelihood_replay",
                                            "score_recomputation", "baseline_hazard_replay")
                elif operation == "fit" and typ == "CoxProportionalHazardsModel":
                    verification_methods = ("partial_likelihood_score",
                                            "observed_information", "tie_accounting",
                                            "baseline_hazard")
                elif operation == "diagnostics" and typ == "CoxPHFit":
                    verification_methods = ("covariance_symmetry", "score_residual",
                                            "schoenfeld_time_correlation")
                elif operation == "verify" and typ == "TimeSeriesDataset":
                    verification_methods = ("strict_time_order", "spacing_reconciliation",
                                            "missing_policy")
                elif operation == "verify" and typ == "TimeSeriesModel":
                    verification_methods = ("regular_spacing", "order_family_contract",
                                            "effective_sample_size")
                elif operation == "verify" and typ == "TimeSeriesFit":
                    verification_methods = ("residual_replay", "variance_replay",
                                            "source_length")
                elif operation == "verify" and typ == "TimeSeriesForecast":
                    verification_methods = ("forecast_replay", "uncertainty_replay",
                                            "time_grid")
                elif operation == "fit" and typ == "TimeSeriesModel":
                    verification_methods = ("conditional_residual_replay", "root_conditions",
                                            "variance_positivity", "optimizer_convergence")
                elif operation == "diagnostics" and typ == "TimeSeriesFit":
                    verification_methods = ("residual_acf", "ljung_box", "jarque_bera")
                elif operation == "verify" and typ == "PoissonProcess":
                    verification_methods = ("rate_domain", "process_assumption_scope")
                elif operation == "verify" and typ == "WienerProcess":
                    verification_methods = ("diffusion_domain", "process_assumption_scope")
                elif operation == "verify" and typ == "GaussianProcess":
                    verification_methods = ("kernel_parameter_domain", "kernel_family_identity")
                elif operation == "verify" and typ == "ContinuousTimeMarkovChain":
                    verification_methods = ("generator_row_sums", "off_diagonal_rates", "initial_probability")
                elif operation == "verify" and typ == "FiniteDimensionalDistribution":
                    verification_methods = ("source_replay", "mean_covariance_recomputation")
                elif operation == "verify" and typ == "GaussianProcessPosterior":
                    verification_methods = ("source_replay", "cholesky_recomputation")
                elif operation == "verify" and typ == "CTMCTransition":
                    verification_methods = ("source_replay", "matrix_exponential_recomputation")
                elif operation == "verify" and typ == "StochasticDifferentialEquation":
                    verification_methods = ("shape_contract", "symbol_scope",
                                            "time_interval", "interpretation")
                elif operation == "verify" and typ == "SDESimulation":
                    verification_methods = ("source_replay", "pcg64_stream_replay",
                                            "path_recomputation")
                elif operation == "verify" and typ == "SDEConvergenceStudy":
                    verification_methods = ("source_replay", "coupled_stream_replay",
                                            "error_order_recomputation")
            elif domain == "pde":
                pde_checks = {
                    ("PDEProblem", "verify"): ("shape_contract", "symbol_scope",
                        "rectangular_domain_order", "known_condition_compatibility"),
                    ("PDEProblem", "classify"): ("differential_order",
                        "principal_matrix", "discriminant_sign_or_conditions"),
                    ("PDEProblem", "boundary_compatibility"): (
                        "dirichlet_trace_intersections", "initial_boundary_traces"),
                    ("PDEProblem", "derive_weak_form"): (
                        "space_scope_and_regularity", "product_rule_integration_by_parts",
                        "oriented_boundary_term_retention"),
                    ("PDEClassification", "verify"): ("source_replay",
                        "principal_part_recomputation", "classification_recomputation"),
                    ("PDECompatibilityReport", "verify"): ("source_replay",
                        "trace_recomputation", "conflict_count_reconciliation"),
                    ("WeakForm", "verify"): ("source_replay",
                        "volume_boundary_term_recomputation", "derivation_step_recomputation"),
                    ("FEMMesh", "verify"): ("simplex_determinants", "orientation",
                        "facet_incidence", "source_replay"),
                    ("FEMMesh", "reference_element"): ("canonical_simplex",),
                    ("FEMMesh", "finite_element_space"): (
                        "source_replay", "local_to_global_range", "essential_dof_mapping"),
                    ("ReferenceElement", "verify"): ("canonical_simplex_replay",),
                    ("ReferenceElement", "basis"): (
                        "nodal_kronecker_property", "partition_of_unity", "gradient_partition"),
                    ("ReferenceElement", "quadrature"): (
                        "weight_sum", "positive_weights", "monomial_exactness"),
                    ("BasisFunctionSet", "verify"): ("basis_symbolic_replay",),
                    ("QuadratureRule", "verify"): ("quadrature_moment_replay",),
                    ("FiniteElementSpace", "verify"): (
                        "multi_source_replay", "dof_numbering", "essential_dof_recomputation"),
                    ("FiniteElementSpace", "assemble"): (
                        "local_element_replay", "sparse_coalescing",
                        "natural_boundary_integration", "symmetric_essential_elimination"),
                    ("AssembledSystem", "verify"): (
                        "multi_source_replay", "local_to_global_reassembly",
                        "boundary_transformation_replay"),
                    ("AssembledSystem", "solve"): (
                        "rank", "augmented_rank", "residual", "conditioning"),
                    ("FEMSolution", "verify"): (
                        "solver_replay", "rank_recomputation", "residual_recomputation"),
                    ("FEMSolution", "estimate_error"): (
                        "solution_ancestry", "cell_residuals", "interior_flux_jumps",
                        "global_estimator_reconciliation"),
                    ("FEMErrorEstimate", "verify"): (
                        "source_replay", "local_indicator_recomputation", "global_norm"),
                    ("FEMErrorEstimate", "mark"): (
                        "marking_policy", "threshold_reconciliation"),
                    ("FEMErrorEstimate", "compare"): (
                        "direct_refinement_lineage", "empirical_rate_recomputation"),
                    ("RefinementMarking", "verify"): (
                        "source_replay", "marking_policy_recomputation"),
                    ("RefinementMarking", "refine"): (
                        "conforming_closure", "parent_child_lineage", "nested_transfer"),
                    ("RefinedMesh", "verify"): (
                        "refinement_replay", "parent_child_lineage", "facet_incidence"),
                    ("RefinedMesh", "reference_element"): ("canonical_simplex",),
                    ("RefinedMesh", "finite_element_space"): (
                        "source_replay", "local_to_global_range", "essential_dof_mapping"),
                    ("MeshTransfer", "verify"): (
                        "source_replay", "interpolation_row_recomputation"),
                    ("FEMConvergenceObservation", "verify"): (
                        "source_replay", "empirical_rate_recomputation"),
                }
                verification_methods = pde_checks[(typ, operation)]
            registry.register(Capability(name=f"{domain}.{typ}.{operation}", domain=domain, operation=operation,
                input_types=(typ,), output_types=output, engines=engines, trust_levels=levels,
                verification_methods=verification_methods,
                cost_dimensions=("sample_count", "precision_bits") if domain == "signal" else
                                ("matrix_dimension", "constraint_count", "entry_size") if domain == "optimization" else
                                ("cell_count", "boundary_entries", "coefficient_domain") if typ in TOPOLOGY_OPERATIONS else
                                ("point_count", "dimension", "predicate_precision") if domain == "geometry" and typ in {"Point", "PointSet", "Polygon", "Polytope", "Triangulation"} else
                                ("chart_dimension", "expression_size") if domain == "geometry" else
                                ("observation_count", "resample_count", "enumeration_states") if domain == "statistics" and operation in {"mann_whitney", "wilcoxon", "kruskal_wallis", "ks_2samp", "spearman", "kendall", "permutation_test", "bootstrap"} else
                                ("observation_count", "event_count", "risk_set_work") if domain == "statistics" and typ in {"SurvivalDataset", "KaplanMeierEstimate", "CoxProportionalHazardsModel", "CoxPHFit"} else
                                ("observation_count", "lag_or_horizon", "parameter_count") if domain == "statistics" and typ in {"TimeSeriesDataset", "TimeSeriesAnalysis", "TimeSeriesModel", "TimeSeriesFit", "TimeSeriesForecast"} else
                                ("state_or_time_count", "matrix_entries", "factorization_work") if domain == "statistics" and typ in {"PoissonProcess", "WienerProcess", "GaussianProcess", "ContinuousTimeMarkovChain", "FiniteDimensionalDistribution", "GaussianProcessPosterior", "CTMCTransition"} else
                                ("path_count", "step_count", "state_noise_dimension") if domain == "statistics" and typ in {"StochasticDifferentialEquation", "SDESimulation", "SDEConvergenceStudy"} else
                                ("observation_count", "variable_count", "arithmetic") if domain == "statistics" else
                                ("mesh_points", "simplex_cells", "sparse_nnz_or_solve_dofs") if domain == "pde" and typ in {"FEMMesh", "ReferenceElement", "BasisFunctionSet", "QuadratureRule", "FiniteElementSpace", "AssembledSystem", "FEMSolution", "FEMErrorEstimate", "RefinementMarking", "RefinedMesh", "MeshTransfer", "FEMConvergenceObservation"} else
                                ("field_equation_terms", "condition_pairs", "expression_size") if domain == "pde" else
                                ("system_order", "entry_size"),
                handler="module:engineering", parameter_schema=parameter_schema(operation, schema)))


def _parse_tree(kernel, tree, context_id):
    texts = []
    def collect(item):
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                collect(child)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError("mathematical values must be MathIR strings or numeric literals")
        else:
            texts.append(str(item))
    collect(tree)
    _, values, trust = kernel._typed_parse_many(texts, context_id)
    iterator = iter(values)
    def build(item):
        if item is None:
            return None
        if isinstance(item, dict):
            return {key: build(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return tuple(build(value) for value in item)
        return next(iterator)
    return build(tree), trust


def construct(kernel, kind, definition):
    if kind in PDE_TYPES:
        from .pde_adapter import construct as construct_pde
        return construct_pde(kernel, kind, definition)
    if kind in STATISTICS_TYPES:
        from .statistics_adapter import construct as construct_statistics
        return construct_statistics(kernel, kind, definition)
    if kind in TOPOLOGY_TYPES:
        from .topology_adapter import construct as construct_topology
        return construct_topology(kernel, kind, definition)
    if kind in GEOMETRY_TYPES:
        from .geometry_adapter import construct as construct_geometry
        return construct_geometry(kernel, kind, definition)
    if TYPES.get(kind) == 'RiccatiCertificate':
        from .riccati import RiccatiCertificate
        data = dict(definition); context = data.pop('context_id', None)
        if set(data) != {'P'}:
            raise ValueError('RiccatiCertificate accepts only P and context_id')
        matrix = data['P']
        if not isinstance(matrix, (list, tuple)) or len(matrix) > kernel.settings.max_control_order or any(not isinstance(row, (list, tuple)) or len(row) > kernel.settings.max_control_order for row in matrix):
            raise ValueError('P exceeds control dimension limit')
        parsed, trust = _parse_tree(kernel, matrix, context)
        return 'RiccatiCertificate', RiccatiCertificate(P=parsed), trust, []
    if kind in EXTENDED_TYPES:
        from .optimization_extended_adapter import construct as construct_extended
        return construct_extended(kernel, kind, definition)
    from .signal_processing import ContinuousSignal, DiscreteSignal, Spectrum, Filter, FilterDesign
    from .control_systems import TransferFunction, StateSpaceSystem, DiscreteControlSystem
    from .control_analysis import ZeroPoleGain
    from .optimization import OptimizationProblem, OptimizationCertificate
    from .optimization_proofs import MILPCertificate, MILPProofNode
    typ = TYPES[kind]
    cls = {c.__name__: c for c in (ContinuousSignal, DiscreteSignal, Spectrum, Filter, FilterDesign, TransferFunction, StateSpaceSystem, DiscreteControlSystem, ZeroPoleGain,
                                  OptimizationProblem, OptimizationCertificate, MILPCertificate)}[typ]
    data = dict(definition)
    if typ == "Filter" and "sos" in data:
        raise ValueError("SOS realizations are generated by FilterDesign; supply coefficients for manual filters")
    context_id = data.pop("context_id", None)
    requested_trust = data.get("input_trust", "exact")
    if typ in {"OptimizationCertificate", "MILPCertificate"}:
        if "input_trust" in data:
            raise ValueError("certificate trust is derived from its mathematical entries")
    s = kernel.settings
    if typ == "MILPCertificate":
        if not 1 <= len(data.get("nodes", ())) <= s.max_milp_nodes:
            raise ValueError("MILP certificate exceeds max_milp_nodes")
        incumbent, trust = _parse_tree(kernel, data.get("incumbent", ()), context_id)
        nodes = []
        for raw in data["nodes"]:
            node = dict(raw)
            if node.get("split") is not None:
                node["split"], local_trust = _parse_tree(kernel, node["split"], context_id)
                trust = TrustLevel(cap_trust(trust.value, local_trust.value))
            if node.get("certificate") is not None:
                _, node["certificate"], local_trust, _ = construct(kernel, "optimization_certificate", node["certificate"])
                trust = TrustLevel(cap_trust(trust.value, local_trust.value))
            nodes.append(MILPProofNode(**node))
        extra = set(data)-{"nodes", "incumbent"}
        if extra:
            raise ValueError(f"unknown MILP certificate field: {sorted(extra)[0]}")
        return typ, MILPCertificate(incumbent=incumbent, nodes=tuple(nodes)), trust, []
    if typ == "ContinuousSignal":
        fields = ("expression", "start", "end")
    elif typ in {"DiscreteSignal", "Spectrum"}:
        key = "samples" if typ == "DiscreteSignal" else "bins"
        if not 1 <= len(data.get(key, ())) <= s.max_signal_samples:
            raise ValueError(f"{key} must contain 1..max_signal_samples={s.max_signal_samples} entries")
        fields = (key, "sample_rate", "start")
    elif typ == "FilterDesign":
        fields = ("cutoff", "sample_rate", "passband_ripple", "stopband_attenuation")
    elif typ in {"TransferFunction", "Filter"}:
        for key in ("numerator", "denominator"):
            if len(data.get(key, ())) > s.max_control_order+1:
                raise ValueError(f"coefficient count exceeds max_control_order={s.max_control_order}")
        fields = ("numerator", "denominator", "sample_time") if typ == "TransferFunction" else ("numerator", "denominator", "sample_rate", "sos")
    elif typ == "ZeroPoleGain":
        for key in ("zeros", "poles"):
            if len(data.get(key, ())) > s.max_control_order:
                raise ValueError(f"root count exceeds max_control_order={s.max_control_order}")
        fields = ("zeros", "poles", "gain", "sample_time")
    elif typ in {"StateSpaceSystem", "DiscreteControlSystem"}:
        for key in ("A", "B", "C", "D"):
            matrix = data.get(key, ())
            if len(matrix) > s.max_control_order or any(len(row) > s.max_control_order for row in matrix):
                raise ValueError(f"matrix exceeds max_control_order={s.max_control_order}")
        fields = ("A", "B", "C", "D", "sample_time")
    elif typ == "OptimizationProblem":
        if len(data.get("c", ())) > s.max_optimization_variables:
            raise ValueError("objective exceeds max_optimization_variables")
        for key in ("A_ub", "A_eq", "Q"):
            matrix = data.get(key, ())
            if len(matrix) > s.max_optimization_constraints or any(len(row) > s.max_optimization_variables for row in matrix):
                raise ValueError("optimization matrix exceeds configured resource limits")
        fields = ("c", "Q", "A_ub", "b_ub", "A_eq", "b_eq", "lower", "upper")
    else:
        fields = ("primal", "inequality_dual", "equality_dual", "ray")
        if any(len(data.get(key, ())) > s.max_optimization_constraints+2*s.max_optimization_variables for key in fields):
            raise ValueError("certificate exceeds configured resource limits")
    parsed, trust = _parse_tree(kernel, {key: data[key] for key in fields if key in data}, context_id)
    data.update(parsed)
    trust = TrustLevel(cap_trust(trust.value, requested_trust))
    if typ != "OptimizationCertificate":
        data["input_trust"] = trust.value
    return typ, cls(**data), trust, []


def _isolated_adapter_apply(value, object_type, operation, parameters, settings, records, contexts,
                            object_id, native_runner=None):
    from dataclasses import replace
    from .kernel import MathKernel
    from . import engines
    kernel = MathKernel(replace(settings, store_path=None))
    kernel.math_objects.update(records)
    kernel.contexts.update(contexts)
    original_runner = engines.run_in_subprocess
    if native_runner is not None:
        engines.run_in_subprocess = native_runner
    try:
        return apply(kernel, value, object_type, operation, parameters, object_id=object_id)
    finally:
        engines.run_in_subprocess = original_runner


def _dependency_snapshot(kernel, value, parameters, object_id):
    """Only reachable mathematical records cross IPC, including transitive sources.

    A fit depends on its model, which depends on its dataset and original sample;
    transferring only explicit operation parameters would lose that lineage.
    Live locks, SQLite connections and session executors never cross IPC.
    """
    from pydantic import BaseModel
    pending = [object_id] if object_id else []
    def find_refs(value):
        if isinstance(value, str) and value.startswith("obj_"):
            pending.append(value)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                find_refs(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values(): find_refs(item)
        elif isinstance(value, (list, tuple)):
            for item in value: find_refs(item)
    find_refs(value)
    find_refs(parameters)
    records, visited = {}, set()
    while pending:
        key = pending.pop()
        if key in visited: continue
        visited.add(key)
        record = kernel._get_math_object_record(key)
        if record is not None:
            records[key] = record
            find_refs(record.get("sources", []))
            find_refs(record.get("value"))
    return records


def apply(kernel, value, object_type, operation, parameters, *, object_id=None):
    from .process_runner import in_worker
    if not in_worker() and kernel.settings.solver_timeout_seconds:
        from .engines import run_with_timeout, run_in_subprocess
        records = _dependency_snapshot(kernel, value, parameters, object_id)
        return run_with_timeout(_isolated_adapter_apply, kernel.settings.solver_timeout_seconds,
            value, object_type, operation, parameters, kernel.settings, records,
            dict(kernel.contexts), object_id, run_in_subprocess)
    from . import signal_processing as signal
    from . import control_systems as control
    from . import optimization as opt
    from .engines import run_with_timeout
    p, s = parameters, kernel.settings
    allowed = parameter_schema(operation, OPERATIONS[object_type][operation])
    unknown = set(p)-set(allowed)
    if unknown:
        raise ValueError(f"unsupported operation parameter: {sorted(unknown)[0]}")
    def reference(name, typ):
        record = kernel._get_math_object_record(str(p[name]))
        if record is None or record["object_type"] != typ:
            raise ValueError(f"{name} must reference a stored {typ}")
        return record["value"]
    def integer(name, default, maximum):
        return positive_integer(p.get(name, default), name, maximum)
    mode = p.get("mode", "exact")
    if mode not in {"exact", "numeric"}:
        raise ValueError("mode must be exact or numeric")
    precision = integer("precision", 53, 4096)
    if precision != 53 and operation not in {"dft", "idft", "cross_spectrum"}:
        raise NotImplementedError("arbitrary precision currently applies to DFT/IDFT only")
    def compute():
        if object_type in PDE_OPERATIONS:
            from .pde_adapter import apply as apply_pde
            return apply_pde(kernel, value, operation, p, object_id)
        if object_type in STATISTICS_OPERATIONS:
            from .statistics_adapter import apply as apply_statistics
            return apply_statistics(kernel, value, operation, p, object_id)
        if object_type in TOPOLOGY_OPERATIONS:
            from .topology_adapter import apply as apply_topology
            return apply_topology(kernel, value, operation, p, object_id)
        if object_type in GEOMETRY_OPERATIONS:
            from .geometry_adapter import apply as apply_geometry
            return apply_geometry(kernel, value, operation, p, object_id)
        if operation in MPC_OPERATIONS.get(object_type, {}):
            from .mpc_adapter import apply as apply_mpc
            return apply_mpc(kernel,value,object_type,operation,p,reference,object_id)
        if operation in SEQUENTIAL_OPERATIONS.get(object_type, {}):
            from .control_sequential_adapter import apply as apply_sequential
            return apply_sequential(kernel, value, object_type, operation, p, reference, object_id)
        if object_type == 'StateSpaceSystem' and operation in {'lqr', 'kalman'}:
            from .riccati import synthesize
            keys = ('W', 'V') if operation == 'kalman' else ('Q', 'R')
            for key in keys:
                matrix = p.get(key, ())
                if not isinstance(matrix, (list, tuple)) or len(matrix) > s.max_control_order or any(not isinstance(row, (list, tuple)) or len(row) > s.max_control_order for row in matrix):
                    raise ValueError('weights exceed control dimension limit')
            parsed, weight_trust = _parse_tree(kernel, {key:p[key] for key in keys}, p.get('context_id'))
            if 'certificate' in p and 'certificate_id' in p:
                raise ValueError('supply only one certificate or certificate_id')
            cert=None;cert_trust=TrustLevel.EXACT
            if 'certificate_id' in p:
                cert=reference('certificate_id', 'RiccatiCertificate')
                cert_trust=kernel._get_math_object_record(str(p['certificate_id']))['input_trust']
            elif 'certificate' in p:
                definition=dict(p['certificate'])
                if p.get('context_id') is not None: definition.setdefault('context_id',p['context_id'])
                _, cert, cert_trust, _ = construct(kernel, 'riccati_certificate', definition)
            tolerance=p.get('tolerance',1e-9)
            if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or not 0<tolerance<1:
                raise ValueError('tolerance must be finite and in (0,1)')
            return synthesize(value,parsed[keys[0]],parsed[keys[1]],kalman=operation=='kalman',mode=mode,
                certificate=cert,input_trust=weight_trust.value,certificate_trust=cert_trust.value,
                tolerance=tolerance,max_exact_order=s.max_exact_control_order,
                reconstruction_denominator=integer('reconstruction_denominator',1000000,1000000000),
                time_limit=s.solver_timeout_seconds*.9)
        if object_type in EXTENDED_OPERATIONS:
            from .optimization_extended_adapter import apply as apply_extended
            return apply_extended(kernel, value, object_type, operation, p, reference)
        if object_type == "OptimizationProblem" and operation == "to_conic":
            from .conic import from_linear_problem
            return from_linear_problem(value, max_rows=s.max_optimization_constraints)
        if object_type == "ContinuousSignal":
            if operation == "sample":
                rate, _ = _parse_tree(kernel, p["sample_rate"], p.get("context_id"))
                return signal.sample_continuous(value, sample_rate=rate,
                    count=integer("count", 1, s.max_signal_samples), mode=mode, max_output=s.max_signal_samples)
        if object_type in {"DiscreteSignal", "Spectrum", "Filter", "FilterDesign", "FilterState"}:
            if operation == "initial_state":
                from .streaming import initial_state
                start, _ = _parse_tree(kernel, p.get("start", 0), p.get("context_id"))
                return initial_state(value, object_id, mode=mode, start=start, unit=p.get("unit", ""))
            if operation == "process":
                from .streaming import process
                filter_record = kernel._get_math_object_record(value.filter_id)
                if filter_record is None or filter_record["object_type"] != "Filter":
                    raise ValueError("stream filter reference is unavailable")
                return process(value, filter_record["value"], reference("signal_id", "DiscreteSignal"), max_work=s.max_engineering_work)
            if operation == "design":
                if mode != "numeric":
                    raise ValueError("filter design requires explicit mode='numeric'")
                return signal.design_filter(value)
            if operation == "cross_spectrum":
                return signal.cross_spectrum(value, reference("other_id", "DiscreteSignal"), mode=mode,
                    precision=precision, max_exact=s.max_exact_dft_size, max_high_precision=s.max_high_precision_dft_size)
            if operation in {"dft", "idft"}:
                return signal.dft(value, mode=mode, inverse=operation == "idft", precision=precision,
                    normalization=p.get("normalization", "backward"), max_exact=s.max_exact_dft_size,
                    max_high_precision=s.max_high_precision_dft_size)
            if operation in {"convolution", "correlation", "autocorrelation"}:
                return signal.convolve(value, value if operation == "autocorrelation" else reference("other_id", "DiscreteSignal"),
                    mode=mode, correlation=operation != "convolution", max_work=s.max_engineering_work)
            if operation == "window":
                periodic = p.get("periodic", True)
                if not isinstance(periodic, bool):
                    raise ValueError("periodic must be a boolean")
                if mode == "exact" and len(value.samples) > s.max_exact_window_size:
                    raise ValueError("exact window exceeds max_exact_window_size")
                return signal.window(value, kind=p.get("kind", "hann"), periodic=periodic, mode=mode)
            if operation == "apply_signal":
                return signal.apply_filter(value, reference("signal_id", "DiscreteSignal"), mode=mode, max_work=s.max_engineering_work)
            if operation == "to_transfer_function":
                return control.filter_to_transfer(value)
            if operation == "stft":
                if mode != "numeric":
                    raise ValueError("STFT requires explicit mode='numeric'")
                overlap = p.get("overlap", 128)
                if isinstance(overlap, bool) or not isinstance(overlap, int):
                    raise ValueError("overlap must be an integer")
                return signal.stft(value, size=integer("size", 256, s.max_signal_samples), overlap=overlap,
                    window_kind=p.get("window", "hann"), max_output=s.max_signal_samples)
            if operation == "resample":
                if mode != "numeric":
                    raise ValueError("resampling requires explicit mode='numeric'")
                return signal.resample(value, up=integer("up", 1, s.max_signal_samples),
                    down=integer("down", 1, s.max_signal_samples), max_output=s.max_signal_samples)
        if object_type == "TransferMatrix":
            from .engineering import checked_result
            i, j = p.get("output"), p.get("input")
            if (isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int)
                    or not 0 <= i < len(value.entries) or not 0 <= j < len(value.entries[0])):
                raise ValueError("entry indices out of range")
            selected = value.entries[i][j]
            return checked_result("entry", selected, method="transfer_matrix_entry", trust=value.input_trust), selected
        if object_type == "ZeroPoleGain":
            from . import control_analysis as analysis
            if operation == "to_transfer_function": return analysis.zpk_to_transfer(value)
            if operation in {"poles", "zeros"}:
                values = value.poles if operation == "poles" else value.zeros
                from .engineering import checked_result
                return checked_result(operation, values, method="stored_factored_representation", trust=value.input_trust,
                    checks={"representation_read": True}, details={"multiplicity_preserved": True})
            _, tf = analysis.zpk_to_transfer(value)
            frequencies, _ = _parse_tree(kernel, p["frequencies"], p.get("context_id"))
            return analysis.sampled_frequency_analysis(tf, frequencies, analysis=operation)
        if object_type in {"TransferFunction", "StateSpaceSystem", "DiscreteControlSystem"}:
            if operation in {"coefficient_units", "discretize", "state_feedback", "place_poles", "observer"}:
                from . import control_design as design
                if operation == "coefficient_units":
                    return design.coefficient_units(value)
                if operation == "discretize":
                    dt, _ = _parse_tree(kernel, p["sample_time"], p.get("context_id"))
                    return design.discretize(value, dt, method=p.get("method", "zoh"), mode=mode,
                                             max_exact_order=s.max_exact_control_order)
                if operation == "state_feedback":
                    if len(p.get("gain", ())) > s.max_control_order or any(len(row) > s.max_control_order for row in p.get("gain", ())):
                        raise ValueError("gain exceeds control dimension limit")
                    gain, _ = _parse_tree(kernel, p["gain"], p.get("context_id"))
                    return design.state_feedback(value, gain)
                if not 1 <= len(p.get("poles", ())) <= s.max_control_order:
                    raise ValueError("pole array exceeds control dimension limit")
                poles, _ = _parse_tree(kernel, p["poles"], p.get("context_id"))
                rtol = p.get("rtol", 1e-3)
                if isinstance(rtol, bool) or not isinstance(rtol, (int,float)) or not math.isfinite(rtol) or not 0 < rtol < 1:
                    raise ValueError("rtol must be finite and in (0,1)")
                return design.place_poles(value, poles, mode=mode, observer=operation == "observer",
                    max_iterations=integer("max_iterations", s.max_iterations, s.max_iterations), rtol=rtol,
                    time_limit=s.solver_timeout_seconds*.9)
            if operation == "to_state_space":
                if object_type == "DiscreteControlSystem":
                    from .engineering import checked_result
                    derived = control.StateSpaceSystem(**value.model_dump(mode="python"))
                    return checked_result("to_state_space", derived, method="representation_identity", trust=control._system_trust(value), checks={"fields_identical": True}), derived
                return control.to_state_space(value)
            if operation == "to_discrete_control":
                if value.time_domain != "discrete": raise ValueError("to_discrete_control requires a discrete state-space system")
                from .engineering import checked_result
                derived = control.DiscreteControlSystem(**value.model_dump(mode="python"))
                return checked_result(operation, derived, method="representation_identity", trust=control._system_trust(value), checks={"fields_identical": True}), derived
            if operation == "to_transfer_function":
                return control.to_transfer_function(value)
            if operation == "to_filter":
                if value.input_unit or value.output_unit:
                    raise NotImplementedError("unitful transfer-to-filter conversion requires explicit amplitude normalization")
                return control.to_filter(value)
            if operation in {"controllability", "observability"}:
                return control.rank_analysis(value, observability=operation == "observability")
            if operation == "stability":
                return control.stability(value)
            if operation in {"poles", "zeros"}:
                from .control_analysis import poles_zeros
                return poles_zeros(value, poles=operation == "poles")
            if operation in {"step_response", "impulse_response"}:
                return control.response(value, step=operation == "step_response")
            if operation in {"series", "feedback"}:
                return control.interconnect(value, reference("other_id", "TransferFunction"), feedback=operation == "feedback")
            if operation in {"frequency_response", "bode", "nyquist", "root_locus", "to_zero_pole_gain"}:
                from . import control_analysis as analysis
                if operation == "to_zero_pole_gain": return analysis.transfer_to_zpk(value, mode=mode)
                if operation == "root_locus":
                    if mode != "numeric": raise ValueError("root_locus requires explicit mode='numeric'")
                    gains, _ = _parse_tree(kernel, p["gains"], p.get("context_id"))
                    return analysis.root_locus(value, gains)
                if mode != "numeric":
                    raise ValueError(f"{operation} requires explicit mode='numeric'")
                if not 1 <= len(p.get("frequencies", ())) <= s.max_signal_samples:
                    raise ValueError("frequency array exceeds resource limit or is empty")
                parsed, _ = _parse_tree(kernel, p["frequencies"], p.get("context_id"))
                if operation in {"bode", "nyquist"}:
                    source = value
                    if object_type in {"StateSpaceSystem", "DiscreteControlSystem"}:
                        if len(value.B[0]) != 1 or len(value.C) != 1:
                            raise NotImplementedError("Bode/Nyquist typed curves currently require SISO systems")
                        _, source = control.to_transfer_function(value)
                    return analysis.sampled_frequency_analysis(source, parsed, analysis=operation)
                if object_type in {"StateSpaceSystem", "DiscreteControlSystem"}:
                    from .control_design import state_frequency_response
                    return state_frequency_response(value, parsed, max_work=s.max_engineering_work)
                return analysis.sampled_frequency_analysis(value, parsed, analysis="frequency_response")
        if operation == "certify_milp":
            from .optimization_proofs import solve_milp
            return solve_milp(value, mode=mode, max_nodes=integer("max_nodes", s.max_milp_nodes, s.max_milp_nodes),
                              time_limit=s.solver_timeout_seconds*.9)
        if operation == "verify_milp_certificate":
            from .optimization_proofs import verify_milp
            if ("certificate_id" in p) == ("certificate" in p):
                raise ValueError("supply exactly one certificate_id or inline certificate")
            if "certificate_id" in p:
                cert = reference("certificate_id", "MILPCertificate")
                cert_trust = kernel._get_math_object_record(str(p["certificate_id"]))["input_trust"]
            else:
                _, cert, cert_trust, _ = construct(kernel, "milp_certificate", p["certificate"])
            return verify_milp(value, cert, max_nodes=s.max_milp_nodes, certificate_trust=cert_trust.value)
        if operation == "verify_certificate":
            if ("certificate_id" in p) == ("certificate" in p):
                raise ValueError("supply exactly one certificate_id or inline certificate")
            if "certificate_id" in p:
                certificate = reference("certificate_id", "OptimizationCertificate")
                cert_trust = kernel._get_math_object_record(str(p["certificate_id"]))["input_trust"]
            else:
                _, certificate, cert_trust, _ = construct(kernel, "optimization_certificate", p["certificate"])
            return opt.verify_certificate(value, certificate, certificate_trust=cert_trust.value)
        if operation == "solve":
            tolerance = p.get("tolerance", 1e-9)
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("tolerance must be positive and finite")
            return opt.solve(value, mode=mode, max_iterations=integer("max_iterations", s.max_iterations, s.max_iterations),
                tolerance=tolerance, time_limit=s.solver_timeout_seconds,
                reconstruction_denominator=integer("reconstruction_denominator", 1_000_000, 1_000_000_000))
        raise NotImplementedError(f"unsupported engineering operation {operation}")
    try:
        return run_with_timeout(compute, s.solver_timeout_seconds)
    except ImportError as exc:
        return unsupported(operation, f"optional numeric engine unavailable: {exc}; install the conic extra for conic search, sci for other numerical engineering")
