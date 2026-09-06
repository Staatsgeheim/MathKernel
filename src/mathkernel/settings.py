# =============================================================================
# MathKernel - settings
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, get_args, get_origin, get_type_hints


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def yolo_mode() -> bool:
    """Runtime gate for live MATHKERNEL_* mutation. Default false."""
    return _env_bool("MATHKERNEL_YOLO_MODE", False)


# Settings fields that are not environment-driven.
_NON_ENV_FIELDS = frozenset({"codegen_languages"})

# MATHKERNEL_* variables that are process-env only (not Settings fields).
PROCESS_ENV_SETTINGS: dict[str, str] = {
    "MATHKERNEL_YOLO_MODE": "bool",
    "MATHKERNEL_SKIP_LEAN_INSTALL": "bool",
    "MATHKERNEL_LEAN_CACHE": "str",
}


def env_name_for_field(name: str) -> str:
    return "MATHKERNEL_" + name.upper()


def _annotation_kind(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"
    if origin is type(None):
        return "str"
    # str | None
    if origin is None and args:
        pass
    allowed = {a for a in ((args or ()) + (annotation,)) if a is not type(None)}
    if origin is not None:
        allowed = {a for a in args if a is not type(None)}
    if allowed == {str} or str in allowed and len(allowed) == 1:
        return "str"
    if allowed == {bool}:
        return "bool"
    if allowed == {int}:
        return "int"
    if allowed == {float} or allowed == {int, float}:
        return "float"
    if str in allowed:
        return "str"
    return "str"


def setting_schema() -> dict[str, dict[str, str]]:
    """Declared types for every MATHKERNEL_* knob the YOLO tool may write."""
    hints = get_type_hints(Settings)
    schema: dict[str, dict[str, str]] = {}
    for item in fields(Settings):
        if item.name in _NON_ENV_FIELDS:
            continue
        schema[env_name_for_field(item.name)] = {
            "field": item.name,
            "type": _annotation_kind(hints.get(item.name, item.type)),
            "scope": "live",
        }
    for env_name, kind in PROCESS_ENV_SETTINGS.items():
        schema[env_name] = {
            "field": env_name.removeprefix("MATHKERNEL_").lower(),
            "type": kind,
            "scope": "process_env",
        }
    return schema


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1) and not isinstance(value, bool):
        # Reject bare 0/1 integers? User asked for typed correctly.
        # Accept 0/1 as bool only if they came as int 0/1 — common LLM habit.
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean (true/false, 1/0, yes/no, on/off)")


def _parse_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer() and math.isfinite(value):
            return int(value)
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            if any(c in text for c in ".eE") and not text.startswith("0x"):
                number = float(text)
                if number.is_integer() and math.isfinite(number):
                    return int(number)
                raise ValueError
            return int(text, 10)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    raise ValueError(f"{name} must be an integer, got {type(value).__name__}")


def _parse_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a float, not a boolean")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be a float, got {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number
    raise ValueError(f"{name} must be a float, got {type(value).__name__}")


def _parse_str(value: Any, name: str, *, optional: bool = False) -> str | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} must be a string")
    if isinstance(value, str):
        text = value.strip()
        if optional and text == "":
            return None
        if not text:
            raise ValueError(f"{name} must be a non-empty string")
        return text
    if isinstance(value, (bool, int, float, list, dict)):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return str(value)


def parse_setting_value(env_or_field: str, value: Any) -> tuple[str, str, Any]:
    """Coerce one YOLO update. Returns (env_name, field_or_env_key, typed_value)."""
    key = str(env_or_field).strip()
    schema = setting_schema()
    env_name = key if key in schema else env_name_for_field(key.lower())
    if env_name not in schema:
        raise ValueError(
            f"unknown MATHKERNEL setting {key!r}; use a Settings field "
            "or MATHKERNEL_* name from the schema")
    info = schema[env_name]
    kind = info["type"]
    optional = env_name == "MATHKERNEL_STORE_PATH" or env_name == "MATHKERNEL_LEAN_CACHE"
    if kind == "bool":
        typed = _parse_bool(value, env_name)
    elif kind == "int":
        typed = _parse_int(value, env_name)
    elif kind == "float":
        typed = _parse_float(value, env_name)
    else:
        typed = _parse_str(value, env_name, optional=optional)
    return env_name, info["field"], typed


def write_process_env(env_name: str, value: Any) -> None:
    if value is None:
        os.environ.pop(env_name, None)
        return
    if isinstance(value, bool):
        os.environ[env_name] = "true" if value else "false"
        return
    os.environ[env_name] = str(value)


def apply_setting_updates(current: Settings, updates: dict[str, Any]
                          ) -> tuple[Settings, dict[str, Any], list[str]]:
    """Return a validated Settings plus applied values and process-env notes."""
    if not isinstance(updates, dict) or not updates:
        raise ValueError("updates must be a non-empty object of MATHKERNEL_* keys")
    schema = setting_schema()
    live: dict[str, Any] = {}
    applied: dict[str, Any] = {}
    notes: list[str] = []
    pending_env: list[tuple[str, Any]] = []
    for raw_key, raw_value in updates.items():
        env_name, field_name, typed = parse_setting_value(str(raw_key), raw_value)
        info = schema[env_name]
        pending_env.append((env_name, typed))
        applied[env_name] = typed
        if info["scope"] == "live":
            live[field_name] = typed
        elif env_name == "MATHKERNEL_YOLO_MODE":
            notes.append(
                "MATHKERNEL_YOLO_MODE written to process env; a false value "
                "disables further YOLO calls in this process")
        else:
            notes.append(
                f"{env_name} written to process env only; it takes effect on "
                "the next Lean bootstrap or process restart")
    new = current
    if live:
        merged = replace(current, **live)
        payload = {item.name: getattr(merged, item.name) for item in fields(Settings)}
        new = Settings(**payload)
    for env_name, typed in pending_env:
        write_process_env(env_name, typed)
    return new, applied, notes


@dataclass(frozen=True)
class Settings:
    """Environment-driven configuration (prefix MATHKERNEL_).

    Execution of generated code is disabled by default and must be opted into
    explicitly via MATHKERNEL_ENABLE_EXECUTION=1.
    """

    max_input_length: int = 100_000
    max_expression_nodes: int = 10_000
    max_output_size_bytes: int = 256_000_000
    solver_timeout_seconds: float = 30.0
    enable_execution: bool = False
    execution_timeout_seconds: float = 10.0
    lean_binary: str = "lean"
    lean_timeout_seconds: float = 90.0
    z3_timeout_ms: int = 10_000
    default_target_language: str = "typescript"
    codegen_languages: tuple[str, ...] = field(default=("typescript", "python", "rust"))
    enable_parallel: bool = True
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 4)
    max_batch_jobs: int = 10_000
    max_matrix_dim: int = 128
    max_jobs_retained: int = 100
    max_math_objects: int = 10_000
    max_contour_vertices: int = 4_096
    max_joint_dimensions: int = 8
    max_distribution_components: int = 256
    max_symbolic_series_order: int = 128
    max_order_statistic_sample_size: int = 1_024
    max_inverse_branches: int = 256
    max_obligation_steps: int = 128
    max_graph_vertices: int = 4_096
    max_graph_edges: int = 65_536
    max_combinatorial_items: int = 10_000
    max_group_elements: int = 4_096
    max_field_degree: int = 64
    max_normal_form_dim: int = 128
    max_signal_samples: int = 65536
    max_exact_dft_size: int = 64
    max_high_precision_dft_size: int = 256
    max_exact_window_size: int = 256
    max_control_horizon: int = 256
    max_exact_control_horizon: int = 32
    max_mpc_horizon: int = 32
    max_exact_control_order: int = 8
    max_control_order: int = 32
    max_optimization_variables: int = 128
    max_optimization_constraints: int = 512
    max_psd_cone_order: int = 16
    max_quadratic_constraints: int = 64
    max_milp_nodes: int = 255
    max_engineering_work: int = 1000000
    max_geometry_dimension: int = 8
    max_geometry_rank: int = 6
    max_geometry_points: int = 10_000
    max_geometry_simplices: int = 100_000
    max_geometry_work: int = 1_000_000
    max_topology_dimension: int = 16
    max_topology_cells: int = 10_000
    max_topology_matrix_entries: int = 1_000_000
    max_topology_entry_bits: int = 4_096
    max_topology_work: int = 2_000_000
    max_statistical_variables: int = 256
    max_statistical_observations: int = 100_000
    max_statistical_cells: int = 1_000_000
    max_statistical_work: int = 2_000_000
    max_glm_parameters: int = 64
    max_glm_iterations: int = 200
    max_glm_prediction_rows: int = 100_000
    max_glm_work: int = 20_000_000
    max_nonparametric_groups: int = 64
    max_exact_resampling_states: int = 100_000
    max_resamples: int = 1_000_000
    max_resampling_batch_cells: int = 1_000_000
    max_resampling_work: int = 20_000_000
    max_survival_strata: int = 64
    max_survival_timeline_points: int = 100_000
    max_cox_parameters: int = 64
    max_cox_iterations: int = 200
    max_cox_prediction_rows: int = 100_000
    max_cox_information_condition: int = 1_000_000_000_000
    max_survival_work: int = 20_000_000
    max_time_series_lag: int = 1_000
    max_time_series_difference: int = 2
    max_time_series_parameters: int = 32
    max_time_series_iterations: int = 500
    max_time_series_forecast_steps: int = 10_000
    max_time_series_work: int = 50_000_000
    max_stochastic_states: int = 256
    max_stochastic_time_points: int = 10_000
    max_gp_conditioning_points: int = 2_000
    max_stochastic_matrix_entries: int = 1_000_000
    max_gp_condition_number: int = 1_000_000_000_000
    max_stochastic_work: int = 50_000_000
    max_sde_state_dimension: int = 32
    max_sde_noise_dimension: int = 32
    max_sde_steps: int = 1_000_000
    max_sde_paths: int = 100_000
    max_sde_simulation_cells: int = 5_000_000
    max_sde_work: int = 50_000_000
    max_sde_query_values: int = 20_000
    max_fwht_size: int = 1 << 20
    max_finite_states: int = 4096
    max_cumulant_order: int = 8
    max_closure_results: int = 10_000
    max_iterations: int = 1_000
    tolerance: float = 1e-12
    max_ode_steps: int = 100_000
    store_path: str | None = None  # SQLite persistence; opt-in
    prove_portfolio_size: int = 3
    max_pde_grid: int = 1_000_000  # cells
    max_pde_fields: int = 16
    max_pde_dimensions: int = 8
    max_pde_equations: int = 32
    max_pde_terms: int = 1_024
    max_pde_conditions: int = 1_024
    max_pde_derivative_order: int = 4
    max_pde_nonlinear_power: int = 8
    max_pde_work: int = 2_000_000
    max_pde_spaces: int = 64
    max_pde_space_order: int = 8
    max_pde_weak_terms: int = 4_096
    max_pde_ibp_steps: int = 256
    max_pde_weak_work: int = 5_000_000
    max_fem_points: int = 100_000
    max_fem_cells: int = 200_000
    max_fem_dofs: int = 200_000
    max_fem_work: int = 20_000_000
    max_fem_assembly_nnz: int = 2_000_000
    max_fem_assembly_work: int = 50_000_000
    max_fem_exact_solve_dofs: int = 256
    max_fem_numeric_solve_dofs: int = 100_000
    max_fem_estimator_work: int = 50_000_000
    max_fem_refined_cells: int = 500_000
    max_qe_variables: int = 16

    def __post_init__(self):
        if isinstance(self.max_output_size_bytes, int) and self.max_output_size_bytes < 1000:
            raise ValueError("max_output_size_bytes must be at least 1000 to fit a complete response envelope")
        for name, value in vars(self).items():
            if name.startswith("max_") or name in {"z3_timeout_ms", "prove_portfolio_size"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        for name in ("solver_timeout_seconds", "execution_timeout_seconds",
                     "lean_timeout_seconds", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            max_input_length=_env_int("MATHKERNEL_MAX_INPUT_LENGTH", 100_000),
            max_expression_nodes=_env_int("MATHKERNEL_MAX_EXPRESSION_NODES", 10_000),
            max_output_size_bytes=_env_int("MATHKERNEL_MAX_OUTPUT_SIZE_BYTES", 256_000_000),
            solver_timeout_seconds=_env_float("MATHKERNEL_SOLVER_TIMEOUT_SECONDS", 30.0),
            enable_execution=_env_bool("MATHKERNEL_ENABLE_EXECUTION", False),
            execution_timeout_seconds=_env_float("MATHKERNEL_EXECUTION_TIMEOUT_SECONDS", 10.0),
            lean_binary=os.environ.get("MATHKERNEL_LEAN_BINARY", "lean"),
            lean_timeout_seconds=_env_float("MATHKERNEL_LEAN_TIMEOUT_SECONDS", 90.0),
            z3_timeout_ms=_env_int("MATHKERNEL_Z3_TIMEOUT_MS", 10_000),
            default_target_language=os.environ.get("MATHKERNEL_DEFAULT_TARGET_LANGUAGE", "typescript"),
            enable_parallel=_env_bool("MATHKERNEL_ENABLE_PARALLEL", True),
            max_workers=_env_int("MATHKERNEL_MAX_WORKERS", os.cpu_count() or 4),
            max_batch_jobs=_env_int("MATHKERNEL_MAX_BATCH_JOBS", 10_000),
            max_matrix_dim=_env_int("MATHKERNEL_MAX_MATRIX_DIM", 128),
            max_jobs_retained=_env_int("MATHKERNEL_MAX_JOBS_RETAINED", 100),
            max_math_objects=_env_int("MATHKERNEL_MAX_MATH_OBJECTS", 10_000),
            max_contour_vertices=_env_int(
                "MATHKERNEL_MAX_CONTOUR_VERTICES", 4_096),
            max_joint_dimensions=_env_int(
                "MATHKERNEL_MAX_JOINT_DIMENSIONS", 8),
            max_distribution_components=_env_int(
                "MATHKERNEL_MAX_DISTRIBUTION_COMPONENTS", 256),
            max_symbolic_series_order=_env_int(
                "MATHKERNEL_MAX_SYMBOLIC_SERIES_ORDER", 128),
            max_order_statistic_sample_size=_env_int(
                "MATHKERNEL_MAX_ORDER_STATISTIC_SAMPLE_SIZE", 1_024),
            max_inverse_branches=_env_int(
                "MATHKERNEL_MAX_INVERSE_BRANCHES", 256),
            max_obligation_steps=_env_int(
                "MATHKERNEL_MAX_OBLIGATION_STEPS", 128),
            max_graph_vertices=_env_int(
                "MATHKERNEL_MAX_GRAPH_VERTICES", 4_096),
            max_graph_edges=_env_int(
                "MATHKERNEL_MAX_GRAPH_EDGES", 65_536),
            max_combinatorial_items=_env_int(
                "MATHKERNEL_MAX_COMBINATORIAL_ITEMS", 10_000),
            max_group_elements=_env_int(
                "MATHKERNEL_MAX_GROUP_ELEMENTS", 4_096),
            max_field_degree=_env_int(
                "MATHKERNEL_MAX_FIELD_DEGREE", 64),
            max_normal_form_dim=_env_int(
                "MATHKERNEL_MAX_NORMAL_FORM_DIM", 128),
            max_signal_samples=_env_int("MATHKERNEL_MAX_SIGNAL_SAMPLES", 65536),
            max_exact_dft_size=_env_int("MATHKERNEL_MAX_EXACT_DFT_SIZE", 64),
            max_high_precision_dft_size=_env_int("MATHKERNEL_MAX_HIGH_PRECISION_DFT_SIZE", 256),
            max_exact_window_size=_env_int("MATHKERNEL_MAX_EXACT_WINDOW_SIZE", 256),
            max_control_horizon=_env_int("MATHKERNEL_MAX_CONTROL_HORIZON", 256),
            max_exact_control_horizon=_env_int("MATHKERNEL_MAX_EXACT_CONTROL_HORIZON", 32),
            max_mpc_horizon=_env_int("MATHKERNEL_MAX_MPC_HORIZON", 32),
            max_exact_control_order=_env_int("MATHKERNEL_MAX_EXACT_CONTROL_ORDER", 8),
            max_control_order=_env_int("MATHKERNEL_MAX_CONTROL_ORDER", 32),
            max_optimization_variables=_env_int("MATHKERNEL_MAX_OPTIMIZATION_VARIABLES", 128),
            max_optimization_constraints=_env_int("MATHKERNEL_MAX_OPTIMIZATION_CONSTRAINTS", 512),
            max_psd_cone_order=_env_int("MATHKERNEL_MAX_PSD_CONE_ORDER", 16),
            max_quadratic_constraints=_env_int("MATHKERNEL_MAX_QUADRATIC_CONSTRAINTS", 64),
            max_milp_nodes=_env_int("MATHKERNEL_MAX_MILP_NODES", 255),
            max_engineering_work=_env_int("MATHKERNEL_MAX_ENGINEERING_WORK", 1000000),
            max_geometry_dimension=_env_int("MATHKERNEL_MAX_GEOMETRY_DIMENSION", 8),
            max_geometry_rank=_env_int("MATHKERNEL_MAX_GEOMETRY_RANK", 6),
            max_geometry_points=_env_int("MATHKERNEL_MAX_GEOMETRY_POINTS", 10_000),
            max_geometry_simplices=_env_int("MATHKERNEL_MAX_GEOMETRY_SIMPLICES", 100_000),
            max_geometry_work=_env_int("MATHKERNEL_MAX_GEOMETRY_WORK", 1_000_000),
            max_topology_dimension=_env_int("MATHKERNEL_MAX_TOPOLOGY_DIMENSION", 16),
            max_topology_cells=_env_int("MATHKERNEL_MAX_TOPOLOGY_CELLS", 10_000),
            max_topology_matrix_entries=_env_int("MATHKERNEL_MAX_TOPOLOGY_MATRIX_ENTRIES", 1_000_000),
            max_topology_entry_bits=_env_int("MATHKERNEL_MAX_TOPOLOGY_ENTRY_BITS", 4_096),
            max_topology_work=_env_int("MATHKERNEL_MAX_TOPOLOGY_WORK", 2_000_000),
            max_statistical_variables=_env_int("MATHKERNEL_MAX_STATISTICAL_VARIABLES", 256),
            max_statistical_observations=_env_int("MATHKERNEL_MAX_STATISTICAL_OBSERVATIONS", 100_000),
            max_statistical_cells=_env_int("MATHKERNEL_MAX_STATISTICAL_CELLS", 1_000_000),
            max_statistical_work=_env_int("MATHKERNEL_MAX_STATISTICAL_WORK", 2_000_000),
            max_glm_parameters=_env_int("MATHKERNEL_MAX_GLM_PARAMETERS", 64),
            max_glm_iterations=_env_int("MATHKERNEL_MAX_GLM_ITERATIONS", 200),
            max_glm_prediction_rows=_env_int("MATHKERNEL_MAX_GLM_PREDICTION_ROWS", 100_000),
            max_glm_work=_env_int("MATHKERNEL_MAX_GLM_WORK", 20_000_000),
            max_nonparametric_groups=_env_int("MATHKERNEL_MAX_NONPARAMETRIC_GROUPS", 64),
            max_exact_resampling_states=_env_int("MATHKERNEL_MAX_EXACT_RESAMPLING_STATES", 100_000),
            max_resamples=_env_int("MATHKERNEL_MAX_RESAMPLES", 1_000_000),
            max_resampling_batch_cells=_env_int("MATHKERNEL_MAX_RESAMPLING_BATCH_CELLS", 1_000_000),
            max_resampling_work=_env_int("MATHKERNEL_MAX_RESAMPLING_WORK", 20_000_000),
            max_survival_strata=_env_int("MATHKERNEL_MAX_SURVIVAL_STRATA", 64),
            max_survival_timeline_points=_env_int("MATHKERNEL_MAX_SURVIVAL_TIMELINE_POINTS", 100_000),
            max_cox_parameters=_env_int("MATHKERNEL_MAX_COX_PARAMETERS", 64),
            max_cox_iterations=_env_int("MATHKERNEL_MAX_COX_ITERATIONS", 200),
            max_cox_prediction_rows=_env_int("MATHKERNEL_MAX_COX_PREDICTION_ROWS", 100_000),
            max_cox_information_condition=_env_int("MATHKERNEL_MAX_COX_INFORMATION_CONDITION", 1_000_000_000_000),
            max_survival_work=_env_int("MATHKERNEL_MAX_SURVIVAL_WORK", 20_000_000),
            max_time_series_lag=_env_int("MATHKERNEL_MAX_TIME_SERIES_LAG", 1_000),
            max_time_series_difference=_env_int("MATHKERNEL_MAX_TIME_SERIES_DIFFERENCE", 2),
            max_time_series_parameters=_env_int("MATHKERNEL_MAX_TIME_SERIES_PARAMETERS", 32),
            max_time_series_iterations=_env_int("MATHKERNEL_MAX_TIME_SERIES_ITERATIONS", 500),
            max_time_series_forecast_steps=_env_int("MATHKERNEL_MAX_TIME_SERIES_FORECAST_STEPS", 10_000),
            max_time_series_work=_env_int("MATHKERNEL_MAX_TIME_SERIES_WORK", 50_000_000),
            max_stochastic_states=_env_int("MATHKERNEL_MAX_STOCHASTIC_STATES", 256),
            max_stochastic_time_points=_env_int("MATHKERNEL_MAX_STOCHASTIC_TIME_POINTS", 10_000),
            max_gp_conditioning_points=_env_int("MATHKERNEL_MAX_GP_CONDITIONING_POINTS", 2_000),
            max_stochastic_matrix_entries=_env_int("MATHKERNEL_MAX_STOCHASTIC_MATRIX_ENTRIES", 1_000_000),
            max_gp_condition_number=_env_int("MATHKERNEL_MAX_GP_CONDITION_NUMBER", 1_000_000_000_000),
            max_stochastic_work=_env_int("MATHKERNEL_MAX_STOCHASTIC_WORK", 50_000_000),
            max_sde_state_dimension=_env_int("MATHKERNEL_MAX_SDE_STATE_DIMENSION", 32),
            max_sde_noise_dimension=_env_int("MATHKERNEL_MAX_SDE_NOISE_DIMENSION", 32),
            max_sde_steps=_env_int("MATHKERNEL_MAX_SDE_STEPS", 1_000_000),
            max_sde_paths=_env_int("MATHKERNEL_MAX_SDE_PATHS", 100_000),
            max_sde_simulation_cells=_env_int("MATHKERNEL_MAX_SDE_SIMULATION_CELLS", 5_000_000),
            max_sde_work=_env_int("MATHKERNEL_MAX_SDE_WORK", 50_000_000),
            max_sde_query_values=_env_int("MATHKERNEL_MAX_SDE_QUERY_VALUES", 20_000),
            max_fwht_size=_env_int("MATHKERNEL_MAX_FWHT_SIZE", 1 << 20),
            max_finite_states=_env_int("MATHKERNEL_MAX_FINITE_STATES", 4096),
            max_cumulant_order=_env_int("MATHKERNEL_MAX_CUMULANT_ORDER", 8),
            max_closure_results=_env_int("MATHKERNEL_MAX_CLOSURE_RESULTS", 10_000),
            max_iterations=_env_int("MATHKERNEL_MAX_ITERATIONS", 1_000),
            tolerance=_env_float("MATHKERNEL_TOLERANCE", 1e-12),
            max_ode_steps=_env_int("MATHKERNEL_MAX_ODE_STEPS", 100_000),
            store_path=os.environ.get("MATHKERNEL_STORE_PATH") or None,
            prove_portfolio_size=_env_int("MATHKERNEL_PROVE_PORTFOLIO_SIZE", 3),
            max_pde_grid=_env_int("MATHKERNEL_MAX_PDE_GRID", 1_000_000),
            max_pde_fields=_env_int("MATHKERNEL_MAX_PDE_FIELDS", 16),
            max_pde_dimensions=_env_int("MATHKERNEL_MAX_PDE_DIMENSIONS", 8),
            max_pde_equations=_env_int("MATHKERNEL_MAX_PDE_EQUATIONS", 32),
            max_pde_terms=_env_int("MATHKERNEL_MAX_PDE_TERMS", 1_024),
            max_pde_conditions=_env_int("MATHKERNEL_MAX_PDE_CONDITIONS", 1_024),
            max_pde_derivative_order=_env_int("MATHKERNEL_MAX_PDE_DERIVATIVE_ORDER", 4),
            max_pde_nonlinear_power=_env_int("MATHKERNEL_MAX_PDE_NONLINEAR_POWER", 8),
            max_pde_work=_env_int("MATHKERNEL_MAX_PDE_WORK", 2_000_000),
            max_pde_spaces=_env_int("MATHKERNEL_MAX_PDE_SPACES", 64),
            max_pde_space_order=_env_int("MATHKERNEL_MAX_PDE_SPACE_ORDER", 8),
            max_pde_weak_terms=_env_int("MATHKERNEL_MAX_PDE_WEAK_TERMS", 4_096),
            max_pde_ibp_steps=_env_int("MATHKERNEL_MAX_PDE_IBP_STEPS", 256),
            max_pde_weak_work=_env_int("MATHKERNEL_MAX_PDE_WEAK_WORK", 5_000_000),
            max_fem_points=_env_int("MATHKERNEL_MAX_FEM_POINTS", 100_000),
            max_fem_cells=_env_int("MATHKERNEL_MAX_FEM_CELLS", 200_000),
            max_fem_dofs=_env_int("MATHKERNEL_MAX_FEM_DOFS", 200_000),
            max_fem_work=_env_int("MATHKERNEL_MAX_FEM_WORK", 20_000_000),
            max_fem_assembly_nnz=_env_int("MATHKERNEL_MAX_FEM_ASSEMBLY_NNZ", 2_000_000),
            max_fem_assembly_work=_env_int("MATHKERNEL_MAX_FEM_ASSEMBLY_WORK", 50_000_000),
            max_fem_exact_solve_dofs=_env_int("MATHKERNEL_MAX_FEM_EXACT_SOLVE_DOFS", 256),
            max_fem_numeric_solve_dofs=_env_int("MATHKERNEL_MAX_FEM_NUMERIC_SOLVE_DOFS", 100_000),
            max_fem_estimator_work=_env_int("MATHKERNEL_MAX_FEM_ESTIMATOR_WORK", 50_000_000),
            max_fem_refined_cells=_env_int("MATHKERNEL_MAX_FEM_REFINED_CELLS", 500_000),
            max_qe_variables=_env_int("MATHKERNEL_MAX_QE_VARIABLES", 16),
        )

    def limits(self) -> dict:
        return {
            "max_input_length": self.max_input_length,
            "max_expression_nodes": self.max_expression_nodes,
            "max_output_size_bytes": self.max_output_size_bytes,
            "solver_timeout_seconds": self.solver_timeout_seconds,
            "enable_execution": self.enable_execution,
            "execution_timeout_seconds": self.execution_timeout_seconds,
            "lean_available_configured": self.lean_binary,
            "z3_timeout_ms": self.z3_timeout_ms,
            "enable_parallel": self.enable_parallel,
            "max_workers": self.max_workers,
            "max_batch_jobs": self.max_batch_jobs,
            "max_matrix_dim": self.max_matrix_dim,
            "max_jobs_retained": self.max_jobs_retained,
            "max_math_objects": self.max_math_objects,
            "max_contour_vertices": self.max_contour_vertices,
            "max_joint_dimensions": self.max_joint_dimensions,
            "max_distribution_components": self.max_distribution_components,
            "max_symbolic_series_order": self.max_symbolic_series_order,
            "max_order_statistic_sample_size":
                self.max_order_statistic_sample_size,
            "max_inverse_branches": self.max_inverse_branches,
            "max_obligation_steps": self.max_obligation_steps,
            "max_graph_vertices": self.max_graph_vertices,
            "max_graph_edges": self.max_graph_edges,
            "max_combinatorial_items": self.max_combinatorial_items,
            "max_group_elements": self.max_group_elements,
            "max_field_degree": self.max_field_degree,
            "max_normal_form_dim": self.max_normal_form_dim,
            "max_signal_samples": self.max_signal_samples,
            "max_exact_dft_size": self.max_exact_dft_size,
            "max_high_precision_dft_size": self.max_high_precision_dft_size,
            "max_exact_window_size": self.max_exact_window_size,
            "max_control_horizon": self.max_control_horizon,
            "max_exact_control_horizon": self.max_exact_control_horizon,
            "max_mpc_horizon": self.max_mpc_horizon,
            "max_exact_control_order": self.max_exact_control_order,
            "max_control_order": self.max_control_order,
            "max_optimization_variables": self.max_optimization_variables,
            "max_optimization_constraints": self.max_optimization_constraints,
            "max_psd_cone_order": self.max_psd_cone_order,
            "max_quadratic_constraints": self.max_quadratic_constraints,
            "max_milp_nodes": self.max_milp_nodes,
            "max_engineering_work": self.max_engineering_work,
            "max_geometry_dimension": self.max_geometry_dimension,
            "max_geometry_rank": self.max_geometry_rank,
            "max_geometry_points": self.max_geometry_points,
            "max_geometry_simplices": self.max_geometry_simplices,
            "max_geometry_work": self.max_geometry_work,
            "max_topology_dimension": self.max_topology_dimension,
            "max_topology_cells": self.max_topology_cells,
            "max_topology_matrix_entries": self.max_topology_matrix_entries,
            "max_topology_entry_bits": self.max_topology_entry_bits,
            "max_topology_work": self.max_topology_work,
            "max_statistical_variables": self.max_statistical_variables,
            "max_statistical_observations": self.max_statistical_observations,
            "max_statistical_cells": self.max_statistical_cells,
            "max_statistical_work": self.max_statistical_work,
            "max_glm_parameters": self.max_glm_parameters,
            "max_glm_iterations": self.max_glm_iterations,
            "max_glm_prediction_rows": self.max_glm_prediction_rows,
            "max_glm_work": self.max_glm_work,
            "max_nonparametric_groups": self.max_nonparametric_groups,
            "max_exact_resampling_states": self.max_exact_resampling_states,
            "max_resamples": self.max_resamples,
            "max_resampling_batch_cells": self.max_resampling_batch_cells,
            "max_resampling_work": self.max_resampling_work,
            "max_survival_strata": self.max_survival_strata,
            "max_survival_timeline_points": self.max_survival_timeline_points,
            "max_cox_parameters": self.max_cox_parameters,
            "max_cox_iterations": self.max_cox_iterations,
            "max_cox_prediction_rows": self.max_cox_prediction_rows,
            "max_cox_information_condition": self.max_cox_information_condition,
            "max_survival_work": self.max_survival_work,
            "max_time_series_lag": self.max_time_series_lag,
            "max_time_series_difference": self.max_time_series_difference,
            "max_time_series_parameters": self.max_time_series_parameters,
            "max_time_series_iterations": self.max_time_series_iterations,
            "max_time_series_forecast_steps": self.max_time_series_forecast_steps,
            "max_time_series_work": self.max_time_series_work,
            "max_stochastic_states": self.max_stochastic_states,
            "max_stochastic_time_points": self.max_stochastic_time_points,
            "max_gp_conditioning_points": self.max_gp_conditioning_points,
            "max_stochastic_matrix_entries": self.max_stochastic_matrix_entries,
            "max_gp_condition_number": self.max_gp_condition_number,
            "max_stochastic_work": self.max_stochastic_work,
            "max_sde_state_dimension": self.max_sde_state_dimension,
            "max_sde_noise_dimension": self.max_sde_noise_dimension,
            "max_sde_steps": self.max_sde_steps,
            "max_sde_paths": self.max_sde_paths,
            "max_sde_simulation_cells": self.max_sde_simulation_cells,
            "max_sde_work": self.max_sde_work,
            "max_sde_query_values": self.max_sde_query_values,
            "max_fwht_size": self.max_fwht_size,
            "max_finite_states": self.max_finite_states,
            "max_cumulant_order": self.max_cumulant_order,
            "max_closure_results": self.max_closure_results,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "max_ode_steps": self.max_ode_steps,
            "store_path": self.store_path,
            "prove_portfolio_size": self.prove_portfolio_size,
            "max_pde_grid": self.max_pde_grid,
            "max_pde_fields": self.max_pde_fields,
            "max_pde_dimensions": self.max_pde_dimensions,
            "max_pde_equations": self.max_pde_equations,
            "max_pde_terms": self.max_pde_terms,
            "max_pde_conditions": self.max_pde_conditions,
            "max_pde_derivative_order": self.max_pde_derivative_order,
            "max_pde_nonlinear_power": self.max_pde_nonlinear_power,
            "max_pde_work": self.max_pde_work,
            "max_pde_spaces": self.max_pde_spaces,
            "max_pde_space_order": self.max_pde_space_order,
            "max_pde_weak_terms": self.max_pde_weak_terms,
            "max_pde_ibp_steps": self.max_pde_ibp_steps,
            "max_pde_weak_work": self.max_pde_weak_work,
            "max_fem_points": self.max_fem_points,
            "max_fem_cells": self.max_fem_cells,
            "max_fem_dofs": self.max_fem_dofs,
            "max_fem_work": self.max_fem_work,
            "max_fem_assembly_nnz": self.max_fem_assembly_nnz,
            "max_fem_assembly_work": self.max_fem_assembly_work,
            "max_fem_exact_solve_dofs": self.max_fem_exact_solve_dofs,
            "max_fem_numeric_solve_dofs": self.max_fem_numeric_solve_dofs,
            "max_fem_estimator_work": self.max_fem_estimator_work,
            "max_fem_refined_cells": self.max_fem_refined_cells,
            "max_qe_variables": self.max_qe_variables,
        }
