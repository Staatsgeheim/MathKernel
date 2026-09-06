# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed, bounded Phase F.7 Itô SDE representation and simulation."""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import sympy as sp
from pydantic import model_validator

from mathkernel_artifacts import (
    ComputationEvidence, EmpiricalEvidence, EvidenceBundle, ModelEvidence,
    NumericalEvidence,
)

from .engineering import EngineeringModel, EngineeringResult, cap_trust, validate_scalars
from .statistical_inference import _check_scalar, _compare


Scheme = Literal["euler_maruyama", "milstein"]


class StochasticDifferentialEquation(EngineeringModel):
    state_variables: tuple[str, ...]
    time_variable: str = "t"
    drift: tuple[sp.Expr, ...]
    diffusion: tuple[tuple[sp.Expr, ...], ...]
    initial_state: tuple[sp.Expr, ...]
    start_time: sp.Expr
    end_time: sp.Expr
    parameters: tuple[tuple[str, sp.Expr], ...] = ()
    interpretation: Literal["ito"] = "ito"
    assumptions: tuple[str, ...] = ()
    input_trust: str = "symbolic"

    @model_validator(mode="after")
    def validate_equation(self):
        dimension = len(self.state_variables)
        if not dimension or len(set(self.state_variables)) != dimension:
            raise ValueError("state_variables must be nonempty and unique")
        if any(not name.isidentifier() for name in self.state_variables):
            raise ValueError("state_variables must be valid identifiers")
        if not self.time_variable.isidentifier() or self.time_variable in self.state_variables:
            raise ValueError("time_variable must be a distinct valid identifier")
        if len(self.drift) != dimension or len(self.initial_state) != dimension:
            raise ValueError("drift and initial_state must match state dimension")
        if len(self.diffusion) != dimension or not self.diffusion or not self.diffusion[0]:
            raise ValueError("diffusion must have one nonempty row per state")
        noise_dimension = len(self.diffusion[0])
        if any(len(row) != noise_dimension for row in self.diffusion):
            raise ValueError("diffusion rows must have equal noise dimension")
        parameter_names = tuple(name for name, _ in self.parameters)
        if (len(set(parameter_names)) != len(parameter_names) or
                any(not name.isidentifier() for name in parameter_names) or
                set(parameter_names) & (set(self.state_variables) | {self.time_variable})):
            raise ValueError("parameter names must be unique, valid, and distinct")
        for value in (*self.initial_state, self.start_time, self.end_time,
                      *(value for _, value in self.parameters)):
            _check_scalar(value)
        if _compare(self.start_time, self.end_time) >= 0:
            raise ValueError("SDE requires start_time < end_time")
        expressions = (*self.drift, *(value for row in self.diffusion for value in row))
        validate_scalars(expressions)
        if any(expression.is_real is False or expression.has(sp.I)
               for expression in expressions):
            raise ValueError("SDE coefficients must be real on the declared domain")
        allowed = {sp.Symbol(name) for name in
                   (*self.state_variables, self.time_variable, *parameter_names)}
        unknown = set().union(*(expression.free_symbols for expression in expressions)) - allowed
        if unknown:
            raise ValueError(f"SDE expression contains undeclared symbol: {sorted(map(str, unknown))[0]}")
        if any(not assumption.strip() for assumption in self.assumptions):
            raise ValueError("assumptions must be nonempty strings")
        return self

    @property
    def state_dimension(self):
        return len(self.state_variables)

    @property
    def noise_dimension(self):
        return len(self.diffusion[0])


class SDESimulation(EngineeringModel):
    sde_id: str
    scheme: Scheme
    times: tuple[sp.Expr, ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    steps: int
    path_count: int
    state_dimension: int
    noise_dimension: int
    step_size: sp.Expr
    seed: int
    random_algorithm: Literal["PCG64"] = "PCG64"
    random_stream: int = 0
    replay_from_seed: bool = True
    nominal_strong_order: sp.Expr
    nominal_weak_order: sp.Expr = sp.S.One
    terminal_mean: tuple[float, ...]
    terminal_variance: tuple[float, ...]
    input_trust: str = "empirical"

    @model_validator(mode="after")
    def validate_simulation(self):
        if not self.sde_id or len(self.times) != self.steps + 1:
            raise ValueError("simulation time grid must match steps")
        if len(self.values) != self.path_count or any(
                len(path) != len(self.times) for path in self.values):
            raise ValueError("simulation paths must match path_count and time grid")
        if any(len(state) != self.state_dimension for path in self.values for state in path):
            raise ValueError("simulation states must match state_dimension")
        if len(self.terminal_mean) != self.state_dimension or len(
                self.terminal_variance) != self.state_dimension:
            raise ValueError("terminal summaries must match state_dimension")
        return self


class SDEConvergenceStudy(EngineeringModel):
    sde_id: str
    scheme: Scheme
    base_steps: int
    levels: tuple[int, int, int]
    path_count: int
    seed: int
    random_algorithm: Literal["PCG64"] = "PCG64"
    random_stream: int = 0
    coupled_brownian_paths: bool = True
    step_sizes: tuple[sp.Expr, sp.Expr, sp.Expr]
    rms_terminal_differences: tuple[float, float]
    observed_strong_order: float | None
    nominal_strong_order: sp.Expr
    input_trust: str = "empirical"

    @model_validator(mode="after")
    def validate_study(self):
        if (not self.sde_id or self.levels != (
                self.base_steps, 2 * self.base_steps, 4 * self.base_steps)):
            raise ValueError("convergence levels must be base, 2*base, and 4*base")
        if self.path_count <= 0 or len(self.step_sizes) != 3:
            raise ValueError("convergence study dimensions are invalid")
        return self


def _float(value):
    return sp.Float(repr(float(value)), 17)


def _nominal_order(scheme):
    return sp.Rational(1, 2) if scheme == "euler_maruyama" else sp.S.One


def _bundle(model, method, *, paths=None, seed=None, details=None):
    metadata = {"error_bound": False, "method": method, **(details or {})}
    empirical = []
    if paths is not None:
        empirical.append(EmpiricalEvidence(
            sample_size=paths, sampling_method="PCG64",
            replication={"seed": seed, "algorithm": "PCG64", "stream": 0},
            role="required", trust="empirical",
            metadata={"scope": "simulated paths"}))
    return EvidenceBundle(
        computation=[ComputationEvidence(
            engine="stochastic_differential_equations", method=method,
            arithmetic="numeric", deterministic=True, trust="numeric")],
        numerical=[NumericalEvidence(
            precision=53, residual=None, role="required", trust="numeric",
            metadata=metadata)],
        empirical=empirical,
        model=[ModelEvidence(
            assumptions=["Itô interpretation", *model.assumptions],
            diagnostics=["existence and uniqueness conditions are asserted, not proved",
                         "nominal convergence order requires regularity assumptions",
                         "simulation does not establish model validity",
                         "population generalization and causality are not established"],
            role="diagnostic", trust="unknown",
            metadata={"model_validity": "not_established"})],
        justified_trust="empirical" if paths is not None else "numeric")


def verify_sde(model):
    allowed = set(model.state_variables) | {model.time_variable} | {
        name for name, _ in model.parameters}
    expressions = (*model.drift, *(item for row in model.diffusion for item in row))
    checks = {
        "state_shape": len(model.drift) == len(model.initial_state) == model.state_dimension,
        "diffusion_shape": len(model.diffusion) == model.state_dimension and all(
            len(row) == model.noise_dimension for row in model.diffusion),
        "time_interval_positive": _compare(model.start_time, model.end_time) < 0,
        "symbol_scope": all({str(symbol) for symbol in expression.free_symbols} <= allowed
                            for expression in expressions),
        "ito_interpretation_explicit": model.interpretation == "ito",
    }
    trust = cap_trust(model.input_trust, "symbolic")
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="stochastic_differential_equations", method="sde_schema_and_symbol_scope",
            arithmetic=trust, deterministic=True, trust=trust)],
        model=[ModelEvidence(
            assumptions=["Itô interpretation", *model.assumptions],
            diagnostics=["existence, uniqueness and model validity are not established"],
            role="diagnostic", trust="unknown")], justified_trust=trust)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=trust,
        value={"state_dimension": model.state_dimension,
               "noise_dimension": model.noise_dimension,
               "interval": (model.start_time, model.end_time)},
        details={"existence_uniqueness": "not_established",
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"verify": bundle})


def _compiled(model):
    state_symbols = tuple(sp.Symbol(name) for name in model.state_variables)
    time_symbol = sp.Symbol(model.time_variable)
    substitutions = {sp.Symbol(name): value for name, value in model.parameters}
    drift = [sp.lambdify((time_symbol, *state_symbols), expression.subs(substitutions),
                         modules="numpy") for expression in model.drift]
    diffusion = [[sp.lambdify((time_symbol, *state_symbols), expression.subs(substitutions),
                              modules="numpy") for expression in row]
                 for row in model.diffusion]
    derivative = None
    if model.state_dimension == model.noise_dimension == 1:
        expression = model.diffusion[0][0].subs(substitutions)
        derivative = sp.lambdify((time_symbol, *state_symbols),
                                 sp.diff(expression, state_symbols[0]), modules="numpy")
    return drift, diffusion, derivative


def _evaluate(function, time, state, paths):
    value = np.asarray(function(time, *(state[:, index]
                         for index in range(state.shape[1]))), dtype=float)
    if value.ndim == 0:
        return np.full(paths, float(value))
    value = np.ravel(value)
    if value.size != paths:
        raise ValueError("SDE coefficient did not evaluate once per path")
    return value


def _simulate_array(model, scheme, steps, paths, *, seed=None, increments=None,
                    keep_path=True):
    if scheme not in {"euler_maruyama", "milstein"}:
        raise ValueError("scheme must be euler_maruyama or milstein")
    if scheme == "milstein" and not (
            model.state_dimension == 1 and model.noise_dimension == 1):
        raise ValueError("Milstein is supported only for scalar state and scalar noise")
    start = float(model.start_time); end = float(model.end_time)
    step = (end - start) / steps
    if not math.isfinite(step) or step <= 0:
        raise ValueError("SDE step size is not positive and finite")
    if increments is None:
        rng = np.random.Generator(np.random.PCG64(seed))
        increments = rng.normal(0.0, math.sqrt(step),
                                size=(paths, steps, model.noise_dimension))
    elif increments.shape != (paths, steps, model.noise_dimension):
        raise ValueError("coupled Brownian increments have the wrong shape")
    drift, diffusion, derivative = _compiled(model)
    state = np.broadcast_to(np.asarray([float(value) for value in model.initial_state]),
                            (paths, model.state_dimension)).copy()
    stored = np.empty((paths, steps + 1, model.state_dimension)) if keep_path else None
    if stored is not None:
        stored[:, 0, :] = state
    for index in range(steps):
        time = start + index * step
        previous = state.copy()
        drift_values = np.column_stack([
            _evaluate(function, time, previous, paths) for function in drift])
        diffusion_values = np.empty((paths, model.state_dimension, model.noise_dimension))
        for row in range(model.state_dimension):
            for column in range(model.noise_dimension):
                diffusion_values[:, row, column] = _evaluate(
                    diffusion[row][column], time, previous, paths)
        state = previous + drift_values * step + np.einsum(
            "pdm,pm->pd", diffusion_values, increments[:, index, :], optimize=True)
        if scheme == "milstein":
            derivative_values = _evaluate(derivative, time, previous, paths)
            brownian = increments[:, index, 0]
            state[:, 0] += 0.5 * diffusion_values[:, 0, 0] * derivative_values * (
                brownian * brownian - step)
        if not np.all(np.isfinite(state)):
            raise ValueError("SDE simulation produced a nonfinite state")
        if stored is not None:
            stored[:, index + 1, :] = state
    return stored, state, increments, step


def simulate(model, sde_id, scheme, steps, paths, seed):
    stored, terminal, _, step = _simulate_array(
        model, scheme, steps, paths, seed=seed, keep_path=True)
    exact_step = sp.simplify((model.end_time - model.start_time) / steps)
    times = tuple(sp.simplify(model.start_time + index * exact_step)
                  for index in range(steps + 1))
    output = SDESimulation(
        sde_id=sde_id, scheme=scheme, times=times,
        values=tuple(tuple(tuple(float(value) for value in state) for state in path)
                     for path in stored),
        steps=steps, path_count=paths, state_dimension=model.state_dimension,
        noise_dimension=model.noise_dimension, step_size=exact_step, seed=seed,
        nominal_strong_order=_nominal_order(scheme),
        terminal_mean=tuple(float(value) for value in np.mean(terminal, axis=0)),
        terminal_variance=tuple(float(value) for value in np.var(terminal, axis=0)),
        input_trust="empirical")
    checks = {"finite_paths": True, "grid_matches_interval": times[0] == model.start_time
              and times[-1] == model.end_time, "seed_recorded": output.seed == seed,
              "scheme_not_downgraded": output.scheme == scheme}
    return EngineeringResult(
        operation="simulate", status="verified", trust="empirical",
        value={"scheme": scheme, "steps": steps, "path_count": paths,
               "step_size": exact_step, "terminal_mean": output.terminal_mean,
               "terminal_variance": output.terminal_variance},
        details={"random_algorithm": "PCG64", "seed": seed, "random_stream": 0,
                 "replay_from_seed": True,
                 "nominal_strong_order": _nominal_order(scheme),
                 "nominal_weak_order": 1, "error_bound": False,
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"simulate": _bundle(
            model, scheme, paths=paths, seed=seed,
            details={"steps": steps, "step_size": str(exact_step),
                     "nominal_strong_order": str(_nominal_order(scheme))})}), output


def convergence_study(model, sde_id, scheme, base_steps, paths, seed):
    finest_steps = 4 * base_steps
    rng = np.random.Generator(np.random.PCG64(seed))
    finest_step = float(model.end_time - model.start_time) / finest_steps
    fine_increments = rng.normal(0.0, math.sqrt(finest_step),
                                 size=(paths, finest_steps, model.noise_dimension))
    medium_increments = fine_increments.reshape(
        paths, 2 * base_steps, 2, model.noise_dimension).sum(axis=2)
    coarse_increments = fine_increments.reshape(
        paths, base_steps, 4, model.noise_dimension).sum(axis=2)
    terminals = []
    for steps, increments in ((base_steps, coarse_increments),
                              (2 * base_steps, medium_increments),
                              (finest_steps, fine_increments)):
        terminals.append(_simulate_array(
            model, scheme, steps, paths, increments=increments,
            keep_path=False)[1])
    errors = tuple(float(np.sqrt(np.mean(np.sum(
        (terminals[index] - terminals[index + 1]) ** 2, axis=1))))
                   for index in range(2))
    observed = (math.log(errors[0] / errors[1], 2)
                if errors[0] > 0 and errors[1] > 0 else None)
    exact_steps = tuple(sp.simplify((model.end_time - model.start_time) / count)
                        for count in (base_steps, 2 * base_steps, finest_steps))
    output = SDEConvergenceStudy(
        sde_id=sde_id, scheme=scheme, base_steps=base_steps,
        levels=(base_steps, 2 * base_steps, finest_steps), path_count=paths,
        seed=seed, step_sizes=exact_steps,
        rms_terminal_differences=errors,
        observed_strong_order=observed,
        nominal_strong_order=_nominal_order(scheme), input_trust="empirical")
    checks = {"brownian_paths_coupled": True, "levels_nested": True,
              "finite_errors": all(math.isfinite(value) for value in errors),
              "seed_recorded": True}
    return EngineeringResult(
        operation="convergence_study", status="verified", trust="empirical",
        value={"levels": output.levels, "step_sizes": output.step_sizes,
               "rms_terminal_differences": output.rms_terminal_differences,
               "observed_strong_order": output.observed_strong_order},
        details={"nominal_strong_order": output.nominal_strong_order,
                 "nominal_order_is_assumption": True, "error_bound": False,
                 "random_algorithm": "PCG64", "seed": seed,
                 "model_validity": "not_established"}, verification=checks,
        claim_evidence={"convergence_study": _bundle(
            model, f"coupled_{scheme}_convergence", paths=paths, seed=seed,
            details={"levels": list(output.levels), "coupled_brownian_paths": True})}), output


def verify_simulation(model, simulation):
    _, candidate = simulate(model, simulation.sde_id, simulation.scheme,
                            simulation.steps, simulation.path_count,
                            simulation.seed)
    checks = {"time_grid_recomputed": candidate.times == simulation.times,
              "paths_recomputed": candidate.values == simulation.values,
              "stream_reconciled": candidate.random_algorithm == simulation.random_algorithm
              and candidate.random_stream == simulation.random_stream,
              "summaries_recomputed": candidate.terminal_mean == simulation.terminal_mean
              and candidate.terminal_variance == simulation.terminal_variance}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="empirical", value={"checks": checks}, verification=checks,
        details={"replay_from_seed": True, "model_validity": "not_established"},
        claim_evidence={"verify": _bundle(
            model, f"replay_{simulation.scheme}", paths=simulation.path_count,
            seed=simulation.seed, details={"steps": simulation.steps})})


def simulation_path(model, simulation, path_index, start_step, count):
    stop = start_step + count
    checks = {"path_index_in_range": 0 <= path_index < simulation.path_count,
              "step_slice_in_range": 0 <= start_step < stop <= simulation.steps + 1}
    return EngineeringResult(
        operation="path", status="verified", trust="empirical",
        value={"path_index": path_index, "start_step": start_step,
               "times": simulation.times[start_step:stop],
               "values": simulation.values[path_index][start_step:stop]},
        details={"stored_simulation_query": True,
                 "random_algorithm": simulation.random_algorithm,
                 "seed": simulation.seed, "model_validity": "not_established"},
        verification=checks,
        claim_evidence={"path": _bundle(
            model, "stored_sde_path_slice", paths=simulation.path_count,
            seed=simulation.seed, details={"path_index": path_index,
                                           "start_step": start_step,
                                           "count": count})})


def simulation_terminal_values(model, simulation, start_path, count):
    stop = start_path + count
    values = tuple(path[-1] for path in simulation.values[start_path:stop])
    checks = {"path_slice_in_range": 0 <= start_path < stop <= simulation.path_count,
              "terminal_dimension": all(len(item) == simulation.state_dimension
                                        for item in values)}
    return EngineeringResult(
        operation="terminal_values", status="verified", trust="empirical",
        value={"start_path": start_path, "count": count,
               "time": simulation.times[-1], "values": values},
        details={"stored_simulation_query": True,
                 "random_algorithm": simulation.random_algorithm,
                 "seed": simulation.seed, "model_validity": "not_established"},
        verification=checks,
        claim_evidence={"terminal_values": _bundle(
            model, "stored_sde_terminal_slice", paths=simulation.path_count,
            seed=simulation.seed, details={"start_path": start_path,
                                           "count": count})})


def verify_convergence_study(model, study):
    _, candidate = convergence_study(model, study.sde_id, study.scheme,
                                     study.base_steps, study.path_count,
                                     study.seed)
    checks = {"levels_recomputed": candidate.levels == study.levels,
              "errors_recomputed": candidate.rms_terminal_differences ==
              study.rms_terminal_differences,
              "observed_order_recomputed": candidate.observed_strong_order ==
              study.observed_strong_order,
              "stream_reconciled": candidate.random_algorithm == study.random_algorithm
              and candidate.random_stream == study.random_stream}
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust="empirical", value={"checks": checks}, verification=checks,
        details={"coupled_brownian_paths": True, "error_bound": False},
        claim_evidence={"verify": _bundle(
            model, f"replay_coupled_{study.scheme}", paths=study.path_count,
            seed=study.seed, details={"levels": list(study.levels)})})
