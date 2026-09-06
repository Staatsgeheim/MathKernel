# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted typed adapter for Phase F statistical inference."""
from __future__ import annotations

from .models import TrustLevel


TYPES = {
    "statisticalsample": "StatisticalSample",
    "statistical_sample": "StatisticalSample",
    "generalizedlinearmodel": "GeneralizedLinearModel",
    "generalized_linear_model": "GeneralizedLinearModel",
    "glm": "GeneralizedLinearModel",
    "survivaldataset": "SurvivalDataset",
    "survival_dataset": "SurvivalDataset",
    "survivaldata": "SurvivalDataset",
    "survival_data": "SurvivalDataset",
    "coxproportionalhazardsmodel": "CoxProportionalHazardsModel",
    "cox_proportional_hazards_model": "CoxProportionalHazardsModel",
    "coxphmodel": "CoxProportionalHazardsModel",
    "cox_ph_model": "CoxProportionalHazardsModel",
    "timeseriesdataset": "TimeSeriesDataset",
    "time_series_dataset": "TimeSeriesDataset",
    "timeseries": "TimeSeriesDataset",
    "timeseriesmodel": "TimeSeriesModel",
    "time_series_model": "TimeSeriesModel",
    "poissonprocess": "PoissonProcess",
    "poisson_process": "PoissonProcess",
    "wienerprocess": "WienerProcess",
    "wiener_process": "WienerProcess",
    "brownianprocess": "WienerProcess",
    "brownian_process": "WienerProcess",
    "gaussianprocess": "GaussianProcess",
    "gaussian_process": "GaussianProcess",
    "continuoustimemarkovchain": "ContinuousTimeMarkovChain",
    "continuous_time_markov_chain": "ContinuousTimeMarkovChain",
    "ctmc": "ContinuousTimeMarkovChain",
    "stochasticdifferentialequation": "StochasticDifferentialEquation",
    "stochastic_differential_equation": "StochasticDifferentialEquation",
    "sde": "StochasticDifferentialEquation",
}

OPERATIONS = {
    "StatisticalSample": {
        "describe": {},
        "covariance": {"normalization": "sample|population"},
        "empirical_distribution": {"variable": "stored variable name"},
        "evidence_profile": {},
        "mann_whitney": {"value": "stored variable name", "group": "stored variable name",
                         "group_a": "MathIR scalar", "group_b": "MathIR scalar",
                         "alternative": "two_sided|less|greater", "method": "auto|exact|asymptotic"},
        "wilcoxon": {"left": "stored variable name", "right": "stored variable name",
                     "alternative": "two_sided|less|greater", "method": "auto|exact|asymptotic"},
        "kruskal_wallis": {"value": "stored variable name", "group": "stored variable name",
                           "groups": "MathIR scalar[]?", "method": "auto|exact|asymptotic"},
        "ks_2samp": {"left": "stored variable name", "right": "stored variable name",
                     "alternative": "two_sided|less|greater", "method": "auto|exact|asymptotic"},
        "spearman": {"left": "stored variable name", "right": "stored variable name",
                     "alternative": "two_sided|less|greater", "method": "auto|exact|asymptotic"},
        "kendall": {"left": "stored variable name", "right": "stored variable name",
                    "alternative": "two_sided|less|greater", "method": "auto|exact|asymptotic"},
        "permutation_test": {"value": "stored variable name", "group": "stored variable name",
                             "group_a": "MathIR scalar", "group_b": "MathIR scalar",
                             "statistic": "difference_in_means|difference_in_medians",
                             "alternative": "two_sided|less|greater",
                             "method": "auto|exact|monte_carlo", "resamples": "positive integer",
                             "seed": "uint64 (required for monte_carlo)"},
        "bootstrap": {"variable": "stored variable name", "statistic": "mean|median",
                      "confidence_level": "MathIR scalar in (0,1)",
                      "resamples": "positive integer", "seed": "uint64"},
    },
    "GeneralizedLinearModel": {
        "verify": {},
        "fit": {"max_iterations": "positive integer",
                "tolerance": "positive finite float"},
    },
    "GLMFit": {
        "verify": {},
        "diagnostics": {},
        "predict": {"rows": "MathIR[][] in predictor order"},
    },
    "NonparametricTestResult": {"verify": {}},
    "ResamplingResult": {"verify": {}},
    "SurvivalDataset": {
        "verify": {},
        "kaplan_meier": {"stratum": "MathIR scalar?",
                           "confidence_level": "MathIR scalar in (0,1)"},
    },
    "KaplanMeierEstimate": {
        "verify": {},
        "survival_at": {"time": "concrete MathIR scalar"},
    },
    "CoxProportionalHazardsModel": {
        "verify": {},
        "fit": {"max_iterations": "positive integer",
                "tolerance": "positive finite float"},
    },
    "CoxPHFit": {
        "verify": {},
        "diagnostics": {},
        "predict_partial_hazard": {"rows": "MathIR[][] in predictor order"},
    },
    "TimeSeriesDataset": {
        "verify": {},
        "acf": {"max_lag": "nonnegative integer"},
        "pacf": {"max_lag": "nonnegative integer"},
        "stationarity_test": {"method": "adf", "max_lag": "nonnegative integer"},
    },
    "TimeSeriesAnalysis": {"verify": {}},
    "TimeSeriesModel": {
        "verify": {},
        "fit": {"max_iterations": "positive integer", "tolerance": "positive finite float"},
    },
    "TimeSeriesFit": {
        "verify": {},
        "diagnostics": {"max_lag": "positive integer"},
        "forecast": {"horizon": "positive integer",
                     "confidence_level": "MathIR scalar in (0,1)"},
    },
    "TimeSeriesForecast": {"verify": {}},
    "PoissonProcess": {
        "verify": {}, "pmf": {"time": "MathIR scalar", "count": "nonnegative integer"},
        "moments": {"time": "MathIR scalar"},
        "increment_distribution": {"start": "MathIR scalar", "end": "MathIR scalar"},
    },
    "WienerProcess": {
        "verify": {}, "finite_dimensional": {"times": "strictly increasing MathIR scalar[]"},
        "increment_distribution": {"start": "MathIR scalar", "end": "MathIR scalar"},
    },
    "GaussianProcess": {
        "verify": {}, "finite_dimensional": {"times": "MathIR scalar[]"},
        "condition": {"observation_times": "MathIR scalar[]", "observation_values": "MathIR scalar[]",
                      "prediction_times": "MathIR scalar[]", "jitter": "nonnegative finite float"},
    },
    "FiniteDimensionalDistribution": {"verify": {}},
    "GaussianProcessPosterior": {"verify": {}},
    "ContinuousTimeMarkovChain": {
        "verify": {}, "transition_matrix": {"time": "nonnegative MathIR scalar"},
        "distribution": {"time": "nonnegative MathIR scalar"},
        "stationary_distribution": {},
    },
    "CTMCTransition": {"verify": {}},
    "StochasticDifferentialEquation": {
        "verify": {},
        "simulate": {"scheme": "euler_maruyama|milstein", "steps": "positive integer",
                     "paths": "positive integer", "seed": "uint64"},
        "convergence_study": {"scheme": "euler_maruyama|milstein",
                              "base_steps": "positive integer",
                              "paths": "positive integer", "seed": "uint64"},
    },
    "SDESimulation": {
        "verify": {},
        "path": {"path_index": "nonnegative integer", "start_step": "nonnegative integer",
                 "count": "positive integer"},
        "terminal_values": {"start_path": "nonnegative integer", "count": "positive integer"},
    },
    "SDEConvergenceStudy": {"verify": {}},
}

DERIVED_OUTPUTS = {
    "describe": "DescriptiveSummary",
    "covariance": "CovarianceMatrix",
    "empirical_distribution": "EmpiricalDistribution",
    "fit": "GLMFit",
    "mann_whitney": "NonparametricTestResult",
    "wilcoxon": "NonparametricTestResult",
    "kruskal_wallis": "NonparametricTestResult",
    "ks_2samp": "NonparametricTestResult",
    "spearman": "NonparametricTestResult",
    "kendall": "NonparametricTestResult",
    "permutation_test": "ResamplingResult",
    "bootstrap": "ResamplingResult",
    "kaplan_meier": "KaplanMeierEstimate",
    "acf": "TimeSeriesAnalysis",
    "pacf": "TimeSeriesAnalysis",
    "stationarity_test": "TimeSeriesAnalysis",
    "forecast": "TimeSeriesForecast",
    "finite_dimensional": "FiniteDimensionalDistribution",
    "increment_distribution": "FiniteDimensionalDistribution",
    "condition": "GaussianProcessPosterior",
    "transition_matrix": "CTMCTransition",
    "simulate": "SDESimulation",
    "convergence_study": "SDEConvergenceStudy",
}


def construct(kernel, kind, definition):
    from .statistical_inference import (
        CoxProportionalHazardsModel, GeneralizedLinearModel, StatisticalSample,
        SurvivalDataset, TimeSeriesDataset, TimeSeriesModel, _glm_columns,
        _response_domain,
    )

    target = TYPES[kind]
    if target == "StochasticDifferentialEquation":
        from .stochastic_differential_equations import StochasticDifferentialEquation
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("SDE specifications declare their own symbol scope")
        allowed = {"state_variables", "time_variable", "drift", "diffusion",
                   "initial_state", "start_time", "end_time", "parameters",
                   "interpretation", "assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown StochasticDifferentialEquation field: {sorted(unknown)[0]}")
        states = data.get("state_variables")
        drift = data.get("drift")
        diffusion = data.get("diffusion")
        initial = data.get("initial_state")
        if not isinstance(states, (list, tuple)) or not 1 <= len(states) <= kernel.settings.max_sde_state_dimension:
            raise ValueError("state_variables exceed max_sde_state_dimension or are empty")
        dimension = len(states)
        if (not isinstance(drift, (list, tuple)) or len(drift) != dimension or
                not isinstance(initial, (list, tuple)) or len(initial) != dimension or
                not isinstance(diffusion, (list, tuple)) or len(diffusion) != dimension or
                any(not isinstance(row, (list, tuple)) for row in diffusion)):
            raise ValueError("SDE drift, diffusion, and initial_state must match state dimension")
        noise_dimension = len(diffusion[0]) if diffusion and diffusion[0] else 0
        if (not 1 <= noise_dimension <= kernel.settings.max_sde_noise_dimension or
                any(len(row) != noise_dimension for row in diffusion)):
            raise ValueError("diffusion exceeds max_sde_noise_dimension or is ragged")
        if dimension * noise_dimension > kernel.settings.max_stochastic_matrix_entries:
            raise ValueError("SDE diffusion exceeds max_stochastic_matrix_entries")
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict) or any(not isinstance(name, str) for name in parameters):
            raise ValueError("SDE parameters must be a name-to-MathIR mapping")
        assumptions = data.get("assumptions", ())
        metadata = [*states, data.get("time_variable", "t"), *parameters, *assumptions]
        if (not isinstance(assumptions, (list, tuple)) or
                any(not isinstance(item, str) for item in metadata) or
                sum(len(item) for item in metadata) > kernel.settings.max_input_length):
            raise ValueError("SDE names and assumptions must be bounded strings")
        start, end = data.get("start_time"), data.get("end_time")
        raw = [*drift, *[item for row in diffusion for item in row], *initial,
               start, end, *parameters.values()]
        if (start is None or end is None or any(
                isinstance(item, bool) or not isinstance(item, (str, int, float)) for item in raw)):
            raise ValueError("SDE coefficients, initial state, interval, and parameters must be MathIR scalars")
        _, parsed, trust = kernel._typed_parse_many([str(item) for item in raw], None)
        cursor = 0
        parsed_drift = tuple(parsed[cursor:cursor + dimension]); cursor += dimension
        parsed_diffusion = tuple(tuple(parsed[cursor + row * noise_dimension + column]
                                       for column in range(noise_dimension))
                                 for row in range(dimension))
        cursor += dimension * noise_dimension
        parsed_initial = tuple(parsed[cursor:cursor + dimension]); cursor += dimension
        parsed_start, parsed_end = parsed[cursor:cursor + 2]; cursor += 2
        parsed_parameters = tuple((name, parsed[cursor + index])
                                  for index, name in enumerate(parameters))
        value = StochasticDifferentialEquation(
            state_variables=tuple(states), time_variable=data.get("time_variable", "t"),
            drift=parsed_drift, diffusion=parsed_diffusion,
            initial_state=parsed_initial, start_time=parsed_start, end_time=parsed_end,
            parameters=parsed_parameters, interpretation=data.get("interpretation", "ito"),
            assumptions=tuple(assumptions), input_trust=trust.value)
        return target, value, trust, []
    if target in {"PoissonProcess", "WienerProcess", "GaussianProcess",
                  "ContinuousTimeMarkovChain"}:
        from .stochastic_processes import (
            ContinuousTimeMarkovChain, GaussianProcess, PoissonProcess,
            WienerProcess,
        )
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("stochastic-process specifications do not accept symbolic context")
        assumptions = data.pop("assumptions", ())
        if not isinstance(assumptions, (list, tuple)) or any(
                not isinstance(item, str) or not item.strip() for item in assumptions):
            raise ValueError("assumptions must be nonempty strings")
        if sum(len(item) for item in assumptions) > kernel.settings.max_input_length:
            raise ValueError("stochastic assumptions exceed max_input_length")

        def parse_fields(defaults):
            unknown = set(data) - set(defaults)
            if unknown:
                raise ValueError(f"unknown {target} field: {sorted(unknown)[0]}")
            raw = [data.get(name, default) for name, default in defaults.items()]
            if any(isinstance(item, bool) or not isinstance(item, (str, int, float))
                   for item in raw):
                raise ValueError("stochastic parameters must be MathIR scalars")
            _, parsed, trust = kernel._typed_parse_many([str(item) for item in raw], None)
            return dict(zip(defaults, parsed)), trust

        if target == "PoissonProcess":
            if data.get("rate") is None:
                raise ValueError("rate is required")
            fields, trust = parse_fields({"rate": None, "start_time": 0})
            value = PoissonProcess(**fields, assumptions=tuple(assumptions),
                                   input_trust=trust.value)
        elif target == "WienerProcess":
            fields, trust = parse_fields({"drift": 0, "diffusion": 1,
                                          "initial": 0, "start_time": 0})
            value = WienerProcess(**fields, assumptions=tuple(assumptions),
                                  input_trust=trust.value)
        elif target == "GaussianProcess":
            kernel_name = data.pop("kernel", None)
            if not isinstance(kernel_name, str):
                raise ValueError("kernel is required and must be a string")
            fields, trust = parse_fields({"mean": 0, "variance": 1,
                                          "length_scale": 1,
                                          "observation_noise": 0,
                                          "start_time": 0})
            value = GaussianProcess(kernel=kernel_name.lower(), **fields,
                                    assumptions=tuple(assumptions),
                                    input_trust=trust.value)
        else:
            allowed = {"states", "generator", "initial_distribution"}
            unknown = set(data) - allowed
            if unknown:
                raise ValueError(f"unknown {target} field: {sorted(unknown)[0]}")
            states = data.get("states")
            generator = data.get("generator")
            initial = data.get("initial_distribution")
            if not isinstance(states, (list, tuple)) or not 1 <= len(states) <= kernel.settings.max_stochastic_states:
                raise ValueError("states exceed max_stochastic_states or are empty")
            if any(not isinstance(item, str) for item in states):
                raise ValueError("CTMC state labels must be strings")
            n = len(states)
            if (not isinstance(generator, (list, tuple)) or len(generator) != n or
                    any(not isinstance(row, (list, tuple)) or len(row) != n for row in generator) or
                    not isinstance(initial, (list, tuple)) or len(initial) != n):
                raise ValueError("CTMC generator and initial distribution must match states")
            if n * n > kernel.settings.max_stochastic_matrix_entries:
                raise ValueError("CTMC generator exceeds max_stochastic_matrix_entries")
            raw = [*[_ for row in generator for _ in row], *initial]
            if any(isinstance(item, bool) or not isinstance(item, (str, int, float)) for item in raw):
                raise ValueError("CTMC entries must be MathIR scalars")
            _, parsed, trust = kernel._typed_parse_many([str(item) for item in raw], None)
            matrix = tuple(tuple(parsed[i * n + j] for j in range(n)) for i in range(n))
            value = ContinuousTimeMarkovChain(
                states=tuple(states), generator=matrix,
                initial_distribution=tuple(parsed[n * n:]),
                assumptions=tuple(assumptions), input_trust=trust.value)
        return target, value, trust, []
    if target == "TimeSeriesDataset":
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("time-series specifications do not accept symbolic context")
        allowed = {"sample_id", "time", "value", "missing_policy", "time_assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown TimeSeriesDataset field: {sorted(unknown)[0]}")
        sample_id = data.get("sample_id")
        record = kernel._get_math_object_record(sample_id) if isinstance(sample_id, str) else None
        if record is None or record["object_type"] != "StatisticalSample":
            raise ValueError("sample_id must reference a stored StatisticalSample")
        assumptions = data.get("time_assumptions", ())
        metadata = [data.get("time"), data.get("value"), *assumptions]
        if not isinstance(assumptions, (list, tuple)) or any(
                not isinstance(item, str) for item in metadata):
            raise ValueError("time-series columns and assumptions must be strings")
        if sum(len(item) for item in metadata) > kernel.settings.max_input_length:
            raise ValueError("time-series metadata exceeds max_input_length")
        value = TimeSeriesDataset(
            sample_id=sample_id, time=data["time"], value=data["value"],
            missing_policy=data.get("missing_policy", "reject"),
            time_assumptions=tuple(assumptions), input_trust=record["input_trust"].value)
        from . import time_series
        verified = time_series.validate_dataset(record["value"], value)
        if verified.status != "verified":
            raise ValueError("time-series timestamps must be strictly increasing in stored row order")
        return "TimeSeriesDataset", value, record["input_trust"], [sample_id]

    if target == "TimeSeriesModel":
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("time-series model specifications do not accept symbolic context")
        allowed = {"dataset_id", "family", "p", "d", "q", "include_constant",
                   "innovation_distribution", "initialization", "model_assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown TimeSeriesModel field: {sorted(unknown)[0]}")
        dataset_id = data.get("dataset_id")
        record = kernel._get_math_object_record(dataset_id) if isinstance(dataset_id, str) else None
        if record is None or record["object_type"] != "TimeSeriesDataset":
            raise ValueError("dataset_id must reference a stored TimeSeriesDataset")
        orders = [data.get(name, 0) for name in ("p", "d", "q")]
        if any(isinstance(item, bool) or not isinstance(item, int) for item in orders):
            raise ValueError("time-series orders must be integers")
        if orders[0] + orders[2] > kernel.settings.max_time_series_parameters:
            raise ValueError("time-series orders exceed max_time_series_parameters")
        if orders[1] > kernel.settings.max_time_series_difference:
            raise ValueError("d exceeds max_time_series_difference")
        assumptions = data.get("model_assumptions", ())
        if not isinstance(assumptions, (list, tuple)) or any(
                not isinstance(item, str) for item in assumptions):
            raise ValueError("model_assumptions must be strings")
        if not isinstance(data.get("include_constant", True), bool):
            raise ValueError("include_constant must be boolean")
        value = TimeSeriesModel(
            dataset_id=dataset_id, family=str(data.get("family", "")).lower(),
            p=orders[0], d=orders[1], q=orders[2],
            include_constant=data.get("include_constant", True),
            innovation_distribution=data.get("innovation_distribution", "gaussian"),
            initialization=data.get("initialization", "conditional_sum_squares"),
            model_assumptions=tuple(assumptions), input_trust=record["input_trust"].value)
        return "TimeSeriesModel", value, record["input_trust"], [dataset_id]

    if target == "SurvivalDataset":
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("survival specifications do not accept symbolic context")
        allowed = {"sample_id", "duration", "event", "entry", "strata",
                   "censoring", "survival_assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown SurvivalDataset field: {sorted(unknown)[0]}")
        sample_id = data.get("sample_id")
        record = kernel._get_math_object_record(sample_id) if isinstance(sample_id, str) else None
        if record is None or record["object_type"] != "StatisticalSample":
            raise ValueError("sample_id must reference a stored StatisticalSample")
        assumptions = data.get("survival_assumptions", ())
        labels = [data.get("duration"), data.get("event")]
        labels.extend(data.get(name) for name in ("entry", "strata")
                      if data.get(name) is not None)
        if not isinstance(assumptions, (list, tuple)) or any(
                not isinstance(item, str) for item in [*labels, *assumptions]):
            raise ValueError("survival column labels and assumptions must be strings")
        if sum(len(item) for item in [*labels, *assumptions]) > kernel.settings.max_input_length:
            raise ValueError("survival metadata exceeds max_input_length")
        value = SurvivalDataset(
            sample_id=sample_id, duration=data["duration"], event=data["event"],
            entry=data.get("entry"), strata=data.get("strata"),
            censoring=data.get("censoring", "right"),
            survival_assumptions=tuple(assumptions),
            input_trust=record["input_trust"].value)
        from . import survival_analysis
        verified = survival_analysis.validate_survival_dataset(record["value"], value)
        if verified.status != "verified":
            raise ValueError("survival dataset validation failed")
        if value.strata is not None:
            strata_count = verified.value["strata"]
            if strata_count > kernel.settings.max_survival_strata:
                raise ValueError("survival strata exceed max_survival_strata")
        return "SurvivalDataset", value, record["input_trust"], [sample_id]

    if target == "CoxProportionalHazardsModel":
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("Cox specifications do not accept symbolic context")
        allowed = {"dataset_id", "predictors", "tie_method", "model_assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown Cox model field: {sorted(unknown)[0]}")
        dataset_id = data.get("dataset_id")
        record = kernel._get_math_object_record(dataset_id) if isinstance(dataset_id, str) else None
        if record is None or record["object_type"] != "SurvivalDataset":
            raise ValueError("dataset_id must reference a stored SurvivalDataset")
        predictors = data.get("predictors", ())
        assumptions = data.get("model_assumptions", ())
        if not isinstance(predictors, (list, tuple)) or not isinstance(assumptions, (list, tuple)):
            raise ValueError("predictors and model_assumptions must be lists")
        if not 1 <= len(predictors) <= kernel.settings.max_cox_parameters:
            raise ValueError("predictors exceed max_cox_parameters or are empty")
        metadata = [*predictors, *assumptions, data.get("tie_method", "efron")]
        if any(not isinstance(item, str) for item in metadata):
            raise ValueError("Cox labels, tie method, and assumptions must be strings")
        if sum(len(item) for item in metadata) > kernel.settings.max_input_length:
            raise ValueError("Cox metadata exceeds max_input_length")
        value = CoxProportionalHazardsModel(
            dataset_id=dataset_id, predictors=tuple(predictors),
            tie_method=data.get("tie_method", "efron").lower(),
            model_assumptions=tuple(assumptions),
            input_trust=record["input_trust"].value)
        return "CoxProportionalHazardsModel", value, record["input_trust"], [dataset_id]

    if target == "GeneralizedLinearModel":
        data = dict(definition)
        if data.pop("context_id", None) is not None:
            raise ValueError("GLM specifications do not accept symbolic context")
        allowed = {"sample_id", "response", "predictors", "family", "link",
                   "include_intercept", "model_assumptions"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown GeneralizedLinearModel field: {sorted(unknown)[0]}")
        sample_id = data.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("sample_id must reference a stored StatisticalSample")
        record = kernel._get_math_object_record(sample_id)
        if record is None or record["object_type"] != "StatisticalSample":
            raise ValueError("sample_id must reference a stored StatisticalSample")
        predictors = data.get("predictors", ())
        assumptions = data.get("model_assumptions", ())
        if not isinstance(predictors, (list, tuple)) or not isinstance(assumptions, (list, tuple)):
            raise ValueError("predictors and model_assumptions must be lists")
        if len(predictors) + bool(data.get("include_intercept", True)) > kernel.settings.max_glm_parameters:
            raise ValueError("GLM design exceeds max_glm_parameters")
        metadata = [data.get("response"), data.get("family"), data.get("link"),
                    *predictors, *assumptions]
        if any(not isinstance(item, str) for item in metadata):
            raise ValueError("GLM labels, family, link, and assumptions must be strings")
        if sum(len(item) for item in metadata) > kernel.settings.max_input_length:
            raise ValueError("GLM metadata exceeds max_input_length")
        if not isinstance(data.get("include_intercept", True), bool):
            raise ValueError("include_intercept must be boolean")
        value = GeneralizedLinearModel(
            sample_id=sample_id, response=data["response"],
            predictors=tuple(predictors), family=data["family"].lower(),
            link=data["link"].lower(),
            include_intercept=data.get("include_intercept", True),
            model_assumptions=tuple(assumptions),
            input_trust=record["input_trust"].value)
        # Resolve variables and response-domain errors at construction, while
        # leaving rank/refutability visible to the explicit verify operation.
        _, _, response = _glm_columns(record["value"], value)
        if not _response_domain(record["value"], value, response):
            raise ValueError(f"response violates the {value.family} family domain or is degenerate")
        return "GeneralizedLinearModel", value, record["input_trust"], [sample_id]

    data = dict(definition)
    if data.pop("context_id", None) is not None:
        raise ValueError("statistical observations do not accept symbolic context")
    allowed = {"variables", "observations", "observation_ids", "sampling_method",
               "population", "design_assumptions"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown StatisticalSample field: {sorted(unknown)[0]}")
    variables = data.get("variables", ())
    observations = data.get("observations", ())
    settings = kernel.settings
    if not isinstance(variables, (list, tuple)) or not 1 <= len(variables) <= settings.max_statistical_variables:
        raise ValueError("variables exceed max_statistical_variables or are empty")
    if not isinstance(observations, (list, tuple)) or not 1 <= len(observations) <= settings.max_statistical_observations:
        raise ValueError("observations exceed max_statistical_observations or are empty")
    if len(variables) * len(observations) > settings.max_statistical_cells:
        raise ValueError("sample exceeds max_statistical_cells")
    observation_ids = data.get("observation_ids", ())
    assumptions = data.get("design_assumptions", ())
    population = data.get("population")
    metadata = [*variables, *observation_ids, *assumptions]
    if population is not None:
        metadata.append(population)
    if any(not isinstance(item, str) for item in metadata):
        raise ValueError("statistical labels and assumptions must be strings")
    if sum(len(item) for item in metadata) > settings.max_input_length:
        raise ValueError("statistical metadata exceeds max_input_length")
    if any(not isinstance(row, (list, tuple)) or len(row) != len(variables)
           for row in observations):
        raise ValueError("every observation must match the variable count")
    if any(isinstance(item, bool) or not isinstance(item, (str, int, float))
           for row in observations for item in row):
        raise ValueError("observations must use MathIR strings or numeric literals")
    parsed, trust = kernel._typed_parse_many(
        [str(value) for row in observations for value in row], None)[1:]
    rows = tuple(tuple(parsed[row * len(variables) + column]
                       for column in range(len(variables)))
                 for row in range(len(observations)))
    value = StatisticalSample(
        variables=tuple(variables), observations=rows,
        observation_ids=tuple(observation_ids),
        sampling_method=data.get("sampling_method", "unspecified"),
        population=population,
        design_assumptions=tuple(assumptions),
        input_trust=trust.value)
    return "StatisticalSample", value, trust, []


def apply(kernel, value, operation, parameters, object_id):
    from . import statistical_inference as statistics

    object_type = type(value).__name__
    allowed = set(OPERATIONS[object_type][operation]) | {"context_id"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unsupported operation parameter: {sorted(unknown)[0]}")
    if parameters.get("context_id") is not None:
        raise ValueError("statistical sample operations do not accept symbolic context")
    stochastic_types = {
        "PoissonProcess", "WienerProcess", "GaussianProcess",
        "ContinuousTimeMarkovChain", "FiniteDimensionalDistribution",
        "GaussianProcessPosterior", "CTMCTransition",
        "StochasticDifferentialEquation", "SDESimulation", "SDEConvergenceStudy",
    }
    if object_type in stochastic_types:
        from . import stochastic_processes as processes

        def scalar(name):
            raw = parameters.get(name)
            if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                raise ValueError(f"{name} must be a MathIR scalar")
            return kernel._typed_parse_many([str(raw)], None)[1][0]

        def vector(name, maximum=None):
            raw = parameters.get(name)
            maximum = maximum or kernel.settings.max_stochastic_time_points
            if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= maximum:
                raise ValueError(f"{name} exceeds its configured limit or is empty")
            if any(isinstance(item, bool) or not isinstance(item, (str, int, float))
                   for item in raw):
                raise ValueError(f"{name} must contain MathIR scalars")
            return tuple(kernel._typed_parse_many([str(item) for item in raw], None)[1])

        def source(expected):
            record = kernel._get_math_object_record(object_id)
            source_id = record["sources"][0] if record and record.get("sources") else None
            source_record = kernel._get_math_object_record(source_id) if source_id else None
            if source_record is None or source_record["object_type"] != expected:
                raise ValueError(f"stochastic result source {expected} is unavailable")
            return source_record["value"]

        if object_type in {"StochasticDifferentialEquation", "SDESimulation",
                           "SDEConvergenceStudy"}:
            from . import stochastic_differential_equations as sde
            if object_type == "SDESimulation":
                model = source("StochasticDifferentialEquation")
                if operation == "verify":
                    replay_work = (value.path_count * value.steps * value.state_dimension *
                                   (value.noise_dimension + 1))
                    replay_cells = ((value.steps + 1) + value.path_count *
                                    (value.steps + 1) * value.state_dimension +
                                    value.path_count * value.steps * value.noise_dimension)
                    if replay_work > kernel.settings.max_sde_work or \
                            replay_cells > kernel.settings.max_sde_simulation_cells:
                        raise ValueError("SDE replay exceeds current simulation limits")
                    return sde.verify_simulation(model, value)
                if operation == "path":
                    index = parameters.get("path_index")
                    start = parameters.get("start_step", 0)
                    count = parameters.get("count", value.steps + 1 - start
                                           if isinstance(start, int) else 0)
                    if any(isinstance(item, bool) or not isinstance(item, int)
                           for item in (index, start, count)):
                        raise ValueError("path_index, start_step, and count must be integers")
                    if not 0 <= index < value.path_count or not 0 <= start <= value.steps or count <= 0:
                        raise ValueError("simulation path query is outside the stored artifact")
                    if start + count > value.steps + 1:
                        raise ValueError("simulation path query exceeds the stored time grid")
                    if count * value.state_dimension > kernel.settings.max_sde_query_values:
                        raise ValueError("simulation path query exceeds max_sde_query_values")
                    return sde.simulation_path(model, value, index, start, count)
                start = parameters.get("start_path", 0)
                count = parameters.get("count", value.path_count - start
                                       if isinstance(start, int) else 0)
                if any(isinstance(item, bool) or not isinstance(item, int)
                       for item in (start, count)):
                    raise ValueError("start_path and count must be integers")
                if not 0 <= start < value.path_count or count <= 0 or start + count > value.path_count:
                    raise ValueError("terminal_values query is outside the stored artifact")
                if count * value.state_dimension > kernel.settings.max_sde_query_values:
                    raise ValueError("terminal_values query exceeds max_sde_query_values")
                return sde.simulation_terminal_values(model, value, start, count)
            if object_type == "SDEConvergenceStudy":
                model = source("StochasticDifferentialEquation")
                replay_work = (value.path_count * value.base_steps * 7 *
                               model.state_dimension * (model.noise_dimension + 1))
                replay_cells = value.path_count * value.base_steps * 7 * model.noise_dimension
                if replay_work > kernel.settings.max_sde_work or \
                        replay_cells > kernel.settings.max_sde_simulation_cells:
                    raise ValueError("SDE convergence replay exceeds current simulation limits")
                return sde.verify_convergence_study(
                    model, value)
            if operation == "verify":
                return sde.verify_sde(value)
            scheme = parameters.get("scheme", "euler_maruyama")
            if not isinstance(scheme, str):
                raise ValueError("scheme must be euler_maruyama or milstein")
            scheme = scheme.lower()
            count_name = "steps" if operation == "simulate" else "base_steps"
            steps = parameters.get(count_name)
            paths = parameters.get("paths")
            seed = parameters.get("seed")
            if isinstance(steps, bool) or not isinstance(steps, int) or not (
                    1 <= steps <= kernel.settings.max_sde_steps):
                raise ValueError(f"{count_name} exceeds max_sde_steps or is not positive")
            if operation == "convergence_study" and 4 * steps > kernel.settings.max_sde_steps:
                raise ValueError("finest convergence level exceeds max_sde_steps")
            if isinstance(paths, bool) or not isinstance(paths, int) or not (
                    1 <= paths <= kernel.settings.max_sde_paths):
                raise ValueError("paths exceeds max_sde_paths or is not positive")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
                raise ValueError("seed must be a uint64")
            multiplier = 1 if operation == "simulate" else 7
            work = (paths * steps * multiplier * value.state_dimension *
                    (value.noise_dimension + 1))
            stored = (paths * (steps + 1) * value.state_dimension +
                      (steps + 1 if operation == "simulate" else 0))
            random_cells = paths * steps * (value.noise_dimension if operation == "simulate"
                                             else 7 * value.noise_dimension)
            if work > kernel.settings.max_sde_work:
                raise ValueError("SDE request exceeds max_sde_work")
            if ((stored if operation == "simulate" else 0) + random_cells) > \
                    kernel.settings.max_sde_simulation_cells:
                raise ValueError("SDE request exceeds max_sde_simulation_cells")
            if operation == "simulate":
                return sde.simulate(value, object_id, scheme, steps, paths, seed)
            return sde.convergence_study(value, object_id, scheme, steps, paths, seed)

        if object_type == "PoissonProcess":
            if operation == "verify":
                return processes.verify_poisson(value)
            if operation == "pmf":
                count = parameters.get("count")
                return processes.poisson_pmf(value, scalar("time"), count)
            if operation == "moments":
                return processes.poisson_moments(value, scalar("time"))
            return processes.poisson_increment(
                value, object_id, scalar("start"), scalar("end"))
        if object_type == "WienerProcess":
            if operation == "verify":
                return processes.verify_wiener(value)
            if operation == "finite_dimensional":
                times = vector("times")
                if len(times) ** 2 > kernel.settings.max_stochastic_matrix_entries:
                    raise ValueError("finite-dimensional covariance exceeds max_stochastic_matrix_entries")
                return processes.wiener_finite(value, object_id, times)
            return processes.wiener_increment(
                value, object_id, scalar("start"), scalar("end"))
        if object_type == "GaussianProcess":
            if operation == "verify":
                return processes.verify_gp(value)
            if operation == "finite_dimensional":
                times = vector("times")
                if len(times) ** 3 > kernel.settings.max_stochastic_work or \
                        len(times) ** 2 > kernel.settings.max_stochastic_matrix_entries:
                    raise ValueError("GP finite-dimensional query exceeds stochastic limits")
                return processes.gp_finite(value, object_id, times)
            observations = vector("observation_times", kernel.settings.max_gp_conditioning_points)
            observed = vector("observation_values", kernel.settings.max_gp_conditioning_points)
            predictions = vector("prediction_times")
            n, m = len(observations), len(predictions)
            if n != len(observed):
                raise ValueError("observation_times and observation_values must align")
            if n * n + n * m + m * m > kernel.settings.max_stochastic_matrix_entries or \
                    n ** 3 + n * n * m > kernel.settings.max_stochastic_work:
                raise ValueError("GP conditioning exceeds stochastic limits")
            jitter = parameters.get("jitter", 0.0)
            import math
            if isinstance(jitter, bool) or not isinstance(jitter, (int, float)) or \
                    not math.isfinite(jitter) or jitter < 0:
                raise ValueError("jitter must be nonnegative and finite")
            return processes.gp_condition(
                value, object_id, observations, observed, predictions,
                kernel.settings.max_gp_condition_number, float(jitter))
        if object_type == "ContinuousTimeMarkovChain":
            if operation == "verify":
                return processes.verify_ctmc(value)
            if operation == "stationary_distribution":
                return processes.ctmc_stationary(value)
            if len(value.states) ** 3 > kernel.settings.max_stochastic_work:
                raise ValueError("CTMC matrix exponential exceeds max_stochastic_work")
            time = scalar("time")
            if operation == "distribution":
                return processes.ctmc_distribution(value, time)
            return processes.ctmc_transition(value, object_id, time)
        if object_type == "FiniteDimensionalDistribution":
            expected = ("PoissonProcess" if value.family == "poisson_increment" else
                        "WienerProcess" if value.family.startswith("wiener") else
                        "GaussianProcess")
            return processes.verify_distribution(source(expected), value)
        if object_type == "GaussianProcessPosterior":
            return processes.verify_posterior(
                source("GaussianProcess"), value,
                kernel.settings.max_gp_condition_number)
        return processes.verify_transition(
            source("ContinuousTimeMarkovChain"), value)
    if object_type in {"NonparametricTestResult", "ResamplingResult"}:
        record = kernel._get_math_object_record(object_id)
        source_id = record["sources"][0] if record and record.get("sources") else None
        sample_record = kernel._get_math_object_record(source_id) if source_id else None
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("statistical result source sample is unavailable")
        sample = sample_record["value"]
        work = len(sample.observations) * max(1, value.resamples if object_type == "ResamplingResult" else
                                             value.enumeration_count or 1)
        if work > kernel.settings.max_resampling_work:
            raise ValueError("verification replay exceeds max_resampling_work")
        from . import nonparametric
        if object_type == "NonparametricTestResult":
            return nonparametric.verify_nonparametric(
                sample, value, kernel.settings.max_exact_resampling_states,
                kernel.settings.max_resampling_work,
                kernel.settings.max_nonparametric_groups)
        return nonparametric.verify_resampling(
            sample, value, kernel.settings.max_exact_resampling_states,
            kernel.settings.max_resampling_batch_cells,
            kernel.settings.max_resampling_work)
    if object_type == "TimeSeriesDataset":
        sample_record = kernel._get_math_object_record(value.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("time-series source sample is unavailable")
        sample = sample_record["value"]
        from . import time_series
        if operation == "verify":
            return time_series.validate_dataset(sample, value)
        max_lag = parameters.get("max_lag", min(20, len(sample.observations) - 1))
        if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not (
                0 <= max_lag <= kernel.settings.max_time_series_lag):
            raise ValueError("max_lag exceeds max_time_series_lag or is negative")
        work = len(sample.observations) * (max_lag + 1)
        if operation == "pacf":
            work += max_lag ** 3
        if work > kernel.settings.max_time_series_work:
            raise ValueError("time-series analysis exceeds max_time_series_work")
        if operation == "acf":
            return time_series.autocorrelation(sample, value, object_id, max_lag)
        if operation == "pacf":
            return time_series.partial_autocorrelation(sample, value, object_id, max_lag)
        if parameters.get("method", "adf") != "adf":
            raise ValueError("stationarity_test method must be adf")
        return time_series.adf_test(sample, value, object_id, max_lag)
    if object_type == "TimeSeriesAnalysis":
        dataset_record = kernel._get_math_object_record(value.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "TimeSeriesDataset":
            raise ValueError("time-series analysis source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("time-series analysis source sample is unavailable")
        from . import time_series
        return time_series.verify_analysis(
            sample_record["value"], dataset, value,
            kernel.settings.max_time_series_work)
    if object_type == "TimeSeriesModel":
        dataset_record = kernel._get_math_object_record(value.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "TimeSeriesDataset":
            raise ValueError("time-series model source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("time-series model source sample is unavailable")
        sample = sample_record["value"]
        from . import time_series
        if operation == "verify":
            return time_series.validate_model(sample, dataset, value)
        max_iterations = parameters.get("max_iterations", 200)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not (
                1 <= max_iterations <= kernel.settings.max_time_series_iterations):
            raise ValueError("max_iterations exceeds max_time_series_iterations or is not positive")
        tolerance = parameters.get("tolerance", 1e-9)
        import math
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not (
                math.isfinite(tolerance) and tolerance > 0):
            raise ValueError("tolerance must be positive and finite")
        parameters_count = int(value.include_constant) + value.p + value.q + (
            1 if value.family == "garch" else 0)
        if len(sample.observations) * max(parameters_count, 1) * max_iterations > \
                kernel.settings.max_time_series_work:
            raise ValueError("time-series fit exceeds max_time_series_work")
        return time_series.fit_model(
            sample, dataset, value, object_id, max_iterations, float(tolerance))
    if object_type == "TimeSeriesFit":
        model_record = kernel._get_math_object_record(value.model_id)
        if model_record is None or model_record["object_type"] != "TimeSeriesModel":
            raise ValueError("time-series fit source model is unavailable")
        model = model_record["value"]
        dataset_record = kernel._get_math_object_record(model.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "TimeSeriesDataset":
            raise ValueError("time-series fit source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("time-series fit source sample is unavailable")
        sample = sample_record["value"]
        from . import time_series
        if operation == "verify":
            parameter_count = int(model.include_constant) + model.p + model.q + (
                1 if model.family == "garch" else 0)
            if len(sample.observations) * max(parameter_count, 1) > \
                    kernel.settings.max_time_series_work:
                raise ValueError("time-series fit replay exceeds max_time_series_work")
            return time_series.verify_fit(sample, dataset, model, value)
        if operation == "diagnostics":
            max_lag = parameters.get("max_lag", min(20, len(value.residuals) - 1))
            if isinstance(max_lag, bool) or not isinstance(max_lag, int) or not (
                    1 <= max_lag <= kernel.settings.max_time_series_lag):
                raise ValueError("diagnostic max_lag is outside configured limits")
            return time_series.diagnostics(value, max_lag)
        horizon = parameters.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not (
                1 <= horizon <= kernel.settings.max_time_series_forecast_steps):
            raise ValueError("horizon exceeds max_time_series_forecast_steps or is not positive")
        raw_level = parameters.get("confidence_level", "19/20")
        if isinstance(raw_level, bool) or not isinstance(raw_level, (str, int, float)):
            raise ValueError("confidence_level must be a MathIR scalar")
        _, parsed, _ = kernel._typed_parse_many([str(raw_level)], None)
        level = parsed[0]
        if level.free_symbols or level.is_real is not True or not 0 < float(level) < 1:
            raise ValueError("confidence_level must be concrete and in (0,1)")
        if horizon * max(value.order[0] + value.order[2], 1) > kernel.settings.max_time_series_work:
            raise ValueError("time-series forecast exceeds max_time_series_work")
        return time_series.forecast(
            sample, dataset, model, value, object_id, horizon, float(level))
    if object_type == "TimeSeriesForecast":
        fit_record = kernel._get_math_object_record(value.fit_id)
        if fit_record is None or fit_record["object_type"] != "TimeSeriesFit":
            raise ValueError("forecast source fit is unavailable")
        fit = fit_record["value"]
        model_record = kernel._get_math_object_record(fit.model_id)
        model = model_record["value"] if model_record else None
        dataset_record = kernel._get_math_object_record(model.dataset_id) if model else None
        dataset = dataset_record["value"] if dataset_record else None
        sample_record = kernel._get_math_object_record(dataset.sample_id) if dataset else None
        if (model is None or dataset is None or sample_record is None or
                model_record["object_type"] != "TimeSeriesModel" or
                dataset_record["object_type"] != "TimeSeriesDataset" or
                sample_record["object_type"] != "StatisticalSample"):
            raise ValueError("forecast source chain is unavailable")
        from . import time_series
        if value.horizon * max(fit.order[0] + fit.order[2], 1) > \
                kernel.settings.max_time_series_work:
            raise ValueError("forecast replay exceeds max_time_series_work")
        return time_series.verify_forecast(
            sample_record["value"], dataset, model, fit, value)
    if object_type == "SurvivalDataset":
        sample_record = kernel._get_math_object_record(value.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("survival source sample is unavailable")
        sample = sample_record["value"]
        from . import survival_analysis
        if operation == "verify":
            return survival_analysis.validate_survival_dataset(sample, value)
        raw_level = parameters.get("confidence_level", "19/20")
        if isinstance(raw_level, bool) or not isinstance(raw_level, (str, int, float)):
            raise ValueError("confidence_level must be a MathIR scalar")
        raw_values = [str(raw_level)]
        if parameters.get("stratum") is not None:
            raw_stratum = parameters["stratum"]
            if isinstance(raw_stratum, bool) or not isinstance(raw_stratum, (str, int, float)):
                raise ValueError("stratum must be a MathIR scalar")
            raw_values.append(str(raw_stratum))
        _, parsed, parameter_trust = kernel._typed_parse_many(raw_values, None)
        level = parsed[0]
        if level.free_symbols or level.is_real is not True or not 0 < float(level) < 1:
            raise ValueError("confidence_level must be concrete and in (0,1)")
        stratum = parsed[1] if len(parsed) > 1 else None
        return survival_analysis.kaplan_meier(
            sample, value, object_id, stratum=stratum,
            confidence_level=float(level),
            max_timeline=kernel.settings.max_survival_timeline_points,
            max_work=kernel.settings.max_survival_work,
            parameter_trust=parameter_trust.value)
    if object_type == "KaplanMeierEstimate":
        dataset_record = kernel._get_math_object_record(value.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "SurvivalDataset":
            raise ValueError("Kaplan-Meier source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("Kaplan-Meier source sample is unavailable")
        from . import survival_analysis
        if operation == "verify":
            return survival_analysis.verify_kaplan_meier(
                sample_record["value"], dataset, value,
                kernel.settings.max_survival_timeline_points,
                kernel.settings.max_survival_work)
        raw_time = parameters.get("time")
        if isinstance(raw_time, bool) or not isinstance(raw_time, (str, int, float)):
            raise ValueError("time must be a MathIR scalar")
        _, parsed, query_trust = kernel._typed_parse_many([str(raw_time)], None)
        result = survival_analysis.survival_at(value, parsed[0])
        from .engineering import cap_trust
        result.trust = cap_trust(result.trust, query_trust.value)
        return result
    if object_type == "CoxProportionalHazardsModel":
        dataset_record = kernel._get_math_object_record(value.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "SurvivalDataset":
            raise ValueError("Cox source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("Cox source sample is unavailable")
        sample = sample_record["value"]
        rows = len(sample.observations); columns = len(value.predictors)
        from . import survival_analysis
        if operation == "verify":
            if rows * columns ** 2 > kernel.settings.max_survival_work:
                raise ValueError("Cox verify exceeds max_survival_work")
            return survival_analysis.validate_cox_model(sample, dataset, value)
        max_iterations = parameters.get("max_iterations", 100)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not (
                1 <= max_iterations <= kernel.settings.max_cox_iterations):
            raise ValueError("max_iterations exceeds max_cox_iterations or is not positive")
        tolerance = parameters.get("tolerance", 1e-9)
        import math
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if rows * rows * columns ** 2 * max_iterations > kernel.settings.max_survival_work:
            raise ValueError("Cox fit exceeds max_survival_work")
        return survival_analysis.fit_cox(
            sample, dataset, value, object_id, max_iterations=max_iterations,
            tolerance=float(tolerance),
            max_condition=kernel.settings.max_cox_information_condition)
    if object_type == "CoxPHFit":
        model_record = kernel._get_math_object_record(value.model_id)
        if model_record is None or model_record["object_type"] != "CoxProportionalHazardsModel":
            raise ValueError("Cox fit source model is unavailable")
        model = model_record["value"]
        dataset_record = kernel._get_math_object_record(model.dataset_id)
        if dataset_record is None or dataset_record["object_type"] != "SurvivalDataset":
            raise ValueError("Cox fit source dataset is unavailable")
        dataset = dataset_record["value"]
        sample_record = kernel._get_math_object_record(dataset.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("Cox fit source sample is unavailable")
        sample = sample_record["value"]
        from . import survival_analysis
        if operation == "verify":
            rows = len(sample.observations); columns = len(model.predictors)
            if rows * rows * columns ** 2 * value.iterations > kernel.settings.max_survival_work:
                raise ValueError("Cox replay exceeds max_survival_work")
            return survival_analysis.verify_cox_fit(
                sample, dataset, model, value,
                max_iterations=max(value.iterations, 1), tolerance=1e-9,
                max_condition=kernel.settings.max_cox_information_condition)
        if operation == "diagnostics":
            return survival_analysis.cox_diagnostics(value)
        raw_rows = parameters.get("rows")
        if not isinstance(raw_rows, (list, tuple)) or not 1 <= len(raw_rows) <= kernel.settings.max_cox_prediction_rows:
            raise ValueError("rows exceed max_cox_prediction_rows or are empty")
        if any(not isinstance(row, (list, tuple)) or len(row) != len(model.predictors)
               for row in raw_rows):
            raise ValueError("every Cox prediction row must match the predictor count")
        if len(raw_rows) * len(model.predictors) > kernel.settings.max_survival_work:
            raise ValueError("Cox prediction exceeds max_survival_work")
        if any(isinstance(item, bool) or not isinstance(item, (str, int, float))
               for row in raw_rows for item in row):
            raise ValueError("Cox prediction rows must use MathIR scalars")
        _, parsed, row_trust = kernel._typed_parse_many(
            [str(item) for row in raw_rows for item in row], None)
        parsed_rows = tuple(tuple(parsed[i * len(model.predictors) + j]
                                  for j in range(len(model.predictors)))
                            for i in range(len(raw_rows)))
        return survival_analysis.predict_partial_hazard(value, parsed_rows,
                                                        row_trust.value)
    if object_type == "GeneralizedLinearModel":
        sample_record = kernel._get_math_object_record(value.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("GLM source sample is unavailable")
        sample = sample_record["value"]
        rows = len(sample.observations)
        columns = len(value.predictors) + bool(value.include_intercept)
        if operation == "verify":
            work = rows * columns ** 2 + columns ** 3
            if work > kernel.settings.max_glm_work:
                raise ValueError("verify exceeds max_glm_work")
            return statistics.validate_glm_specification(sample, value)
        max_iterations = parameters.get("max_iterations", 100)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not (
                1 <= max_iterations <= kernel.settings.max_glm_iterations):
            raise ValueError("max_iterations exceeds max_glm_iterations or is not positive")
        tolerance = parameters.get("tolerance", 1e-9)
        import math
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        iterations = 1 if value.family == "gaussian" else max_iterations
        work = rows * columns ** 2 * iterations + columns ** 3 * iterations
        if work > kernel.settings.max_glm_work:
            raise ValueError("fit exceeds max_glm_work")
        return statistics.fit_glm(sample, value, object_id,
                                  max_iterations=max_iterations,
                                  tolerance=float(tolerance))

    if object_type == "GLMFit":
        model_record = kernel._get_math_object_record(value.model_id)
        if model_record is None or model_record["object_type"] != "GeneralizedLinearModel":
            raise ValueError("GLM fit source model is unavailable")
        model = model_record["value"]
        sample_record = kernel._get_math_object_record(model.sample_id)
        if sample_record is None or sample_record["object_type"] != "StatisticalSample":
            raise ValueError("GLM fit source sample is unavailable")
        sample = sample_record["value"]
        work = len(sample.observations) * len(value.coefficients) ** 2
        if operation in {"verify", "diagnostics"}:
            if work > kernel.settings.max_glm_work:
                raise ValueError(f"{operation} exceeds max_glm_work")
            return (statistics.verify_glm_fit(sample, model, value) if operation == "verify"
                    else statistics.glm_diagnostics(sample, model, value))
        raw_rows = parameters.get("rows")
        if not isinstance(raw_rows, (list, tuple)) or not 1 <= len(raw_rows) <= kernel.settings.max_glm_prediction_rows:
            raise ValueError("rows exceed max_glm_prediction_rows or are empty")
        if any(not isinstance(row, (list, tuple)) or len(row) != len(model.predictors)
               for row in raw_rows):
            raise ValueError("every prediction row must match the predictor count")
        if len(raw_rows) * len(value.coefficients) > kernel.settings.max_glm_work:
            raise ValueError("predict exceeds max_glm_work")
        if any(isinstance(item, bool) or not isinstance(item, (str, int, float))
               for row in raw_rows for item in row):
            raise ValueError("prediction rows must use MathIR strings or numeric literals")
        flattened = [str(item) for row in raw_rows for item in row]
        if flattened:
            _, parsed, row_trust = kernel._typed_parse_many(flattened, None)
            parsed_rows = tuple(tuple(parsed[i * len(model.predictors) + j]
                                      for j in range(len(model.predictors)))
                                for i in range(len(raw_rows)))
        else:
            # Intercept-only models accept empty rows without invoking the parser.
            parsed_rows = tuple(() for _ in raw_rows)
            row_trust = TrustLevel.EXACT
        return statistics.glm_predict(sample, model, value, parsed_rows,
                                      row_trust.value)

    rows = len(value.observations); columns = len(value.variables)
    if operation == "covariance":
        work = rows * columns ** 2
    elif operation == "describe":
        work = rows * columns * max(1, (rows - 1).bit_length())
    elif operation == "empirical_distribution":
        work = rows * max(1, (rows - 1).bit_length())
    elif operation == "evidence_profile":
        work = 1
    else:
        work = rows * max(1, (rows - 1).bit_length())
    if work > kernel.settings.max_statistical_work:
        raise ValueError(f"{operation} exceeds max_statistical_work")
    if operation == "describe":
        return statistics.describe(value)
    if operation == "covariance":
        return statistics.covariance(value, parameters.get("normalization", "sample"))
    if operation == "empirical_distribution":
        variable = parameters.get("variable")
        if not isinstance(variable, str):
            raise ValueError("variable must be a string")
        return statistics.empirical_distribution(value, variable)
    if operation == "evidence_profile":
        return statistics.evidence_profile(value)
    from . import nonparametric
    def text(name, default=None):
        item = parameters.get(name, default)
        if not isinstance(item, str):
            raise ValueError(f"{name} must be a string")
        return item
    def parsed_labels(*names):
        raw = [parameters.get(name) for name in names]
        if any(isinstance(item, bool) or not isinstance(item, (str, int, float)) for item in raw):
            raise ValueError("group labels must be MathIR strings or numeric literals")
        return kernel._typed_parse_many([str(item) for item in raw], None)[1]
    def positive(name, default):
        item = parameters.get(name, default)
        if isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= kernel.settings.max_resamples:
            raise ValueError(f"{name} must be in 1..max_resamples")
        return item
    def seed(required):
        item = parameters.get("seed")
        if item is None and not required:
            return None
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 2 ** 64:
            raise ValueError("seed must be an explicit uint64 integer")
        return item
    common = {"alternative": text("alternative", "two_sided"),
              "method": text("method", "auto"),
              "max_exact_states": kernel.settings.max_exact_resampling_states,
              "max_work": kernel.settings.max_resampling_work}
    if operation == "mann_whitney":
        group_a, group_b = parsed_labels("group_a", "group_b")
        return nonparametric.mann_whitney(value, text("value"), text("group"),
                                          group_a, group_b, **common)
    if operation == "wilcoxon":
        return nonparametric.wilcoxon(value, text("left"), text("right"), **common)
    if operation == "kruskal_wallis":
        groups = parameters.get("groups")
        parsed_groups = None
        if groups is not None:
            if not isinstance(groups, (list, tuple)) or not 2 <= len(groups) <= kernel.settings.max_nonparametric_groups:
                raise ValueError("groups must contain 2..max_nonparametric_groups labels")
            if any(isinstance(item, bool) or not isinstance(item, (str, int, float)) for item in groups):
                raise ValueError("groups must use MathIR strings or numeric literals")
            parsed_groups = tuple(kernel._typed_parse_many([str(item) for item in groups], None)[1])
        return nonparametric.kruskal_wallis(
            value, text("value"), text("group"), groups=parsed_groups,
            method=common["method"], max_exact_states=common["max_exact_states"],
            max_work=common["max_work"],
            max_groups=kernel.settings.max_nonparametric_groups)
    if operation in {"ks_2samp", "spearman", "kendall"}:
        function = getattr(nonparametric, operation)
        return function(value, text("left"), text("right"), **common)
    if operation == "permutation_test":
        group_a, group_b = parsed_labels("group_a", "group_b")
        requested_method = text("method", "auto")
        resamples = positive("resamples", 10_000)
        random_seed = seed(requested_method == "monte_carlo")
        if rows * resamples > kernel.settings.max_resampling_work and requested_method == "monte_carlo":
            raise ValueError("permutation_test exceeds max_resampling_work")
        return nonparametric.permutation_test(
            value, text("value"), text("group"), group_a, group_b,
            statistic=text("statistic", "difference_in_means"),
            alternative=text("alternative", "two_sided"), method=requested_method,
            resamples=resamples, seed=random_seed,
            max_exact_states=kernel.settings.max_exact_resampling_states,
            max_work=kernel.settings.max_resampling_work)
    if operation == "bootstrap":
        resamples = positive("resamples", 10_000)
        if rows * resamples > kernel.settings.max_resampling_work:
            raise ValueError("bootstrap exceeds max_resampling_work")
        raw_level = parameters.get("confidence_level", "0.95")
        if isinstance(raw_level, bool) or not isinstance(raw_level, (str, int, float)):
            raise ValueError("confidence_level must be a MathIR scalar")
        level = kernel._typed_parse_many([str(raw_level)], None)[1][0]
        if level.free_symbols or level.is_real is not True or not 0 < float(level) < 1:
            raise ValueError("confidence_level must be concrete and in (0,1)")
        return nonparametric.bootstrap(
            value, text("variable"), statistic=text("statistic", "mean"),
            confidence_level=float(level), resamples=resamples, seed=seed(True),
            batch_cells=kernel.settings.max_resampling_batch_cells)
    raise NotImplementedError(f"unsupported statistics operation {operation}")
