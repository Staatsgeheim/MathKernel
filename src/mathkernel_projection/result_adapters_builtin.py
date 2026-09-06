# =============================================================================
# MathKernel Projection - built-in result adapters
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Built-in adapters from MathKernel typed objects/results to projections.

Every adapter is a pure function over the JSON-encoded object fields.  Any
sampling of symbolic expressions is declared as ``information_loss=
['sampling']`` with the grid recorded in ``parameters`` — presentation never
claims the underlying exactness.  Adapters return ``None`` when required
structure is absent; they never invent missing mathematics.
"""
from __future__ import annotations

import re
from typing import Any

from .coerce import sample_symbolic, to_complex, to_float, to_float_list
from .result_adapters import (AdapterContext, ProjectionSpec,
                              register_model_adapter, register_shape_adapter)

_DEFAULT_SAMPLES = 200


def _sampled_function(expression: Any, variable: Any, lo: Any, hi: Any, *,
                      title: str, samples: int = _DEFAULT_SAMPLES,
                      x_label: str = "x") -> ProjectionSpec:
    a, b = to_float(lo), to_float(hi)
    if not b > a:
        raise ValueError(f"sampling range must be increasing, got [{a}, {b}]")
    xs, ys = sample_symbolic(expression, variable, a, b, samples)
    return ProjectionSpec(
        kind="function", title=title,
        payload={"x": xs, "y": ys},
        parameters={"range": [a, b], "samples": samples,
                    "expression": str(expression), "variable": str(variable)},
        information_loss=["sampling"],
        information_loss_notes=[
            f"Symbolic expression sampled on a uniform grid of {samples} "
            f"points over [{a}, {b}]; the plot is a numeric presentation, "
            "not the exact expression."],
        coordinate_names=[x_label, title])


# ----------------------------------------------------------------------
# Signal processing
# ----------------------------------------------------------------------

def _discrete_signal(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    samples = to_float_list(fields.get("samples"), what="signal samples")
    rate = to_float(fields.get("sample_rate", 1))
    start = to_float(fields.get("start", 0))
    xs = [start + i / rate for i in range(len(samples))]
    unit = str(fields.get("unit") or "")
    return ProjectionSpec(
        kind="function", title="Discrete signal",
        payload={"x": xs, "y": samples},
        parameters={"sample_rate_hz": rate, "start_s": start,
                    "ordering": "sample order"},
        coordinate_names=["t", "amplitude"],
        units={"t": "s", "amplitude": unit} if unit else {"t": "s"})


def _continuous_signal(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    return _sampled_function(fields["expression"], fields.get("variable", "t"),
                             fields.get("start", 0), fields["end"],
                             title="Continuous signal", x_label="t")


def _spectrum(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    raw_bins = fields.get("bins")
    if not isinstance(raw_bins, (list, tuple)):
        raise ValueError("spectrum bins must be a sequence")
    bins = [abs(to_complex(b)) for b in raw_bins]
    rate = to_float(fields["sample_rate"]) if "sample_rate" in fields else None
    n = len(bins)
    if rate:
        freqs = [i * rate / n for i in range(n)]
        payload = {"frequency": freqs, "amplitude": bins}
    else:
        payload = {"values": bins}
    return ProjectionSpec(
        kind="spectrum", title="Spectrum",
        payload=payload,
        parameters={"normalization": fields.get("normalization", "backward"),
                    "ordering": "bin order", "bin_value": "magnitude"},
        information_loss=["summary"],
        information_loss_notes=[
            "Complex bins are presented as magnitudes; phase is retained in "
            "the source object."],
        coordinate_names=["frequency", "amplitude"],
        units={"frequency": "Hz"} if rate else {})


register_model_adapter("DiscreteSignal", _discrete_signal)
register_model_adapter("ContinuousSignal", _continuous_signal)
register_model_adapter("Spectrum", _spectrum)


# ----------------------------------------------------------------------
# Graph theory
# ----------------------------------------------------------------------

def _graph_payload(fields: dict) -> tuple[list[str], list[dict]] | None:
    vertices = fields.get("vertices")
    if not isinstance(vertices, list):
        return None
    nodes = [str(v) for v in vertices]
    edges = []
    for e in fields.get("edges") or []:
        if isinstance(e, dict):
            edges.append({"source": str(e.get("source")),
                          "target": str(e.get("target"))})
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            edges.append({"source": str(e[0]), "target": str(e[1])})
    return nodes, edges


def _graph_like(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    payload = _graph_payload(fields)
    if payload is None:
        return None
    nodes, edges = payload
    return ProjectionSpec(
        kind="graph", title="Graph",
        payload={"nodes": nodes, "edges": edges})


def _parent_tree(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    """Traversal/shortest-path results carry a parent/predecessor map — a
    spanning tree of the explored graph, exact by construction."""
    parents = data.get("parent") or data.get("predecessors")
    if not isinstance(parents, dict) or not parents:
        return None
    nodes = sorted({str(k) for k in parents} |
                   {str(v) for v in parents.values() if v is not None})
    edges = [{"source": str(p), "target": str(v)}
             for v, p in parents.items() if p is not None]
    params: dict[str, Any] = {}
    if data.get("order"):
        params["visit_order"] = [str(v) for v in data["order"]]
    if data.get("path") is not None:
        params["highlighted_path"] = [str(v) for v in data["path"]]
    return ProjectionSpec(
        kind="graph", title="Traversal tree",
        payload={"nodes": nodes, "edges": edges},
        parameters=params)


register_model_adapter("Graph", _graph_like)
register_model_adapter("DirectedGraph", _graph_like)
register_model_adapter("WeightedGraph", _graph_like)
register_model_adapter("MultiGraph", _graph_like)
register_shape_adapter(
    lambda d: isinstance(d.get("parent") or d.get("predecessors"), dict)
    and bool(d.get("parent") or d.get("predecessors"))
    and ("order" in d or "path" in d or "distances" in d),
    _parent_tree)


# ----------------------------------------------------------------------
# Probability and statistics
# ----------------------------------------------------------------------

_INTERVAL_RE = re.compile(r"Interval\(\s*([^,]+),\s*([^)]+)\)")


def _distribution_range(fields: dict) -> tuple[float, float, str]:
    """Pick a plotting window for a density; the choice is recorded."""
    support = str(fields.get("support") or "")
    m = _INTERVAL_RE.search(support)
    if m:
        try:
            a, b = to_float(m.group(1)), to_float(m.group(2))
            if b > a and abs(a) < 1e15 and abs(b) < 1e15:
                return a, b, "support"
        except ValueError:
            pass
    known = fields.get("known_results") or {}
    try:
        mean = to_float(known.get("mean"))
        var = to_float(known.get("variance"))
        if var > 0:
            sd = var ** 0.5
            return mean - 4 * sd, mean + 4 * sd, "mean +/- 4 standard deviations"
    except (ValueError, TypeError):
        pass
    return -5.0, 5.0, "default window (support not finite or unknown)"


def _distribution(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    density = fields.get("density")
    variable = fields.get("variable", "x")
    if density is None:
        return None
    lo, hi, how = _distribution_range(fields)
    spec = _sampled_function(density, variable, lo, hi,
                             title=f"PDF of {fields.get('name', 'distribution')}")
    spec.kind = "distribution"
    spec.payload = {"x": spec.payload["x"], "pdf": spec.payload["y"]}
    spec.parameters["range_basis"] = how
    spec.information_loss_notes.append(f"Plot window chosen from {how}.")
    return spec


def _empirical_distribution(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    support = to_float_list(fields.get("support"), what="support")
    probs = to_float_list(fields.get("probabilities"), what="probabilities")
    if len(support) != len(probs):
        raise ValueError("support and probabilities must have equal length")
    return ProjectionSpec(
        kind="distribution", title="Empirical distribution",
        payload={"x": support, "pdf": probs},
        parameters={"representation": "probability mass per support point"})


def _statistical_sample(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    obs = to_float_list(fields.get("observations"), what="observations")
    return ProjectionSpec(
        kind="statistical_inference", title="Statistical sample",
        payload={"values": obs},
        parameters={"n": len(obs)})


def _covariance_matrix(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    matrix = [[to_float(v) for v in row] for row in fields["matrix"]]
    return ProjectionSpec(kind="matrix", title="Covariance matrix",
                          payload={"matrix": matrix})


def _glm_fit(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    fitted = to_float_list(fields.get("fitted_means"), what="fitted means")
    return ProjectionSpec(
        kind="function", title="GLM fitted means",
        payload={"x": list(range(len(fitted))), "y": fitted},
        parameters={"ordering": "observation order",
                    "coefficients": fields.get("coefficients")})


register_model_adapter("Distribution", _distribution)
register_model_adapter("EmpiricalDistribution", _empirical_distribution)
register_model_adapter("StatisticalSample", _statistical_sample)
register_model_adapter("CovarianceMatrix", _covariance_matrix)
register_model_adapter("GLMFit", _glm_fit)


def _samples_result(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    return ProjectionSpec(
        kind="distribution", title="Random samples",
        payload={"values": to_float_list(data["samples"], what="samples")},
        parameters={"n": data.get("n", len(data["samples"]))})


def _sequence_result(key: str, title: str):
    def adapt(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
        values = to_float_list(data[key], what=key)
        return ProjectionSpec(
            kind="sequence", title=title,
            payload={"values": values},
            parameters={"ordering": "index order"})
    return adapt


def _confidence_interval(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    return ProjectionSpec(
        kind="intervals", title="Confidence interval",
        payload={"intervals": [{"lower": to_float(data["lower"]),
                                "upper": to_float(data["upper"])}]},
        parameters={"confidence": data.get("confidence"),
                    "mean": data.get("mean")})


register_shape_adapter(
    lambda d: isinstance(d.get("samples"), list) and "n" in d,
    _samples_result)
register_shape_adapter(
    lambda d: isinstance(d.get("posterior"), list),
    _sequence_result("posterior", "Posterior distribution"))
register_shape_adapter(
    lambda d: isinstance(d.get("stationary"), list),
    _sequence_result("stationary", "Stationary distribution"))
register_shape_adapter(
    lambda d: isinstance(d.get("hitting_times"), list),
    _sequence_result("hitting_times", "Hitting times"))
register_shape_adapter(
    lambda d: isinstance(d.get("sorted"), list) and "median" in d,
    _sequence_result("sorted", "Order statistics"))
register_shape_adapter(
    lambda d: "lower" in d and "upper" in d and "confidence" in d,
    _confidence_interval)


# ----------------------------------------------------------------------
# Survival analysis and time series
# ----------------------------------------------------------------------

def _kaplan_meier(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    timeline = to_float_list(fields.get("timeline"), what="timeline")
    survival = to_float_list(fields.get("survival"), what="survival")
    if len(timeline) != len(survival):
        raise ValueError("timeline and survival must have equal length")
    params: dict[str, Any] = {"step_function": True}
    for key in ("at_risk", "events"):
        if key in fields:
            params[key] = fields[key]
    return ProjectionSpec(
        kind="function", title="Kaplan-Meier estimate",
        payload={"x": timeline, "y": survival},
        parameters=params,
        information_loss_notes=[
            "Kaplan-Meier curves are right-continuous step functions; the "
            "plot connects event times linearly."])


def _time_series_analysis(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    lags = to_float_list(fields.get("lags"), what="lags")
    values = to_float_list(fields.get("values"), what="values")
    label = fields.get("analysis") or fields.get("kind") or "Time series analysis"
    return ProjectionSpec(
        kind="function", title=str(label).upper(),
        payload={"x": lags, "y": values},
        coordinate_names=["lag", "value"])


def _forecast_intervals(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    times = to_float_list(data["times"], what="times")
    lower = to_float_list(data["lower"], what="lower")
    upper = to_float_list(data["upper"], what="upper")
    entries = [{"x": t, "lower": lo, "upper": hi}
               for t, lo, hi in zip(times, lower, upper)]
    params: dict[str, Any] = {}
    if "means" in data:
        params["means"] = to_float_list(data["means"], what="means")
    return ProjectionSpec(
        kind="intervals", title="Forecast intervals",
        payload={"intervals": entries}, parameters=params)


def _gp_posterior_intervals(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    times = to_float_list(data["times"], what="times")
    means = to_float_list(data["means"], what="means")
    cov = data["covariance"]
    sds = [to_float(cov[i][i]) ** 0.5 for i in range(len(means))]
    entries = [{"x": t, "lower": m - 2 * s, "upper": m + 2 * s}
               for t, m, s in zip(times, means, sds)]
    return ProjectionSpec(
        kind="intervals", title="Gaussian process posterior",
        payload={"intervals": entries},
        parameters={"band": "mean +/- 2 standard deviations",
                    "means": means},
        information_loss=["summary"],
        information_loss_notes=[
            "Full covariance matrix reduced to marginal +/-2sd bands; "
            "cross-covariances are not shown."])


register_model_adapter("KaplanMeierEstimate", _kaplan_meier)
register_model_adapter("TimeSeriesAnalysis", _time_series_analysis)
register_shape_adapter(
    lambda d: all(isinstance(d.get(k), list)
                  for k in ("times", "means", "lower", "upper")),
    _forecast_intervals)
register_shape_adapter(
    lambda d: isinstance(d.get("times"), list)
    and isinstance(d.get("means"), list)
    and isinstance(d.get("covariance"), list),
    _gp_posterior_intervals)


# ----------------------------------------------------------------------
# Control systems
# ----------------------------------------------------------------------

def _frequency_response(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    freqs = to_float_list(fields.get("angular_frequencies"),
                          what="angular frequencies")
    mag_key = "magnitude_db" if "magnitude_db" in fields else "magnitude"
    mag = to_float_list(fields.get(mag_key), what="magnitude")
    if len(freqs) != len(mag):
        raise ValueError("frequencies and magnitude must have equal length")
    return ProjectionSpec(
        kind="function", title="Frequency response",
        payload={"x": freqs, "y": mag},
        parameters={"magnitude": mag_key,
                    "phase_available": "phase_radians" in fields},
        information_loss=["summary"],
        information_loss_notes=[
            "Only the magnitude channel is plotted; phase is retained in the "
            "source object."],
        coordinate_names=["omega", mag_key],
        units={"omega": "rad/s", mag_key: "dB" if mag_key == "magnitude_db" else ""})


def _root_locus(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    branches = fields.get("branches") or []
    points: list[list[float]] = []
    for row in branches:
        roots = row if isinstance(row, (list, tuple)) else [row]
        for r in roots:
            z = to_complex(r)
            points.append([z.real, z.imag])
    if not points:
        return None
    return ProjectionSpec(
        kind="complex_field", title="Root locus",
        payload={"points": points},
        parameters={"gains": fields.get("gains"),
                    "ordering": "branch-major, gain-increasing"},
        coordinate_names=["Re", "Im"])


def _time_response(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    expression = fields.get("expression")
    if expression is None:
        return None
    spec = _sampled_function(expression, fields.get("variable", "t"),
                             fields.get("start", 0), fields.get("end", 10),
                             title=str(fields.get("kind") or "Time response"),
                             x_label="t")
    spec.information_loss_notes.append(
        "Default time window used; the source object carries the symbolic "
        "response, not a sampled grid.")
    return spec


register_model_adapter("FrequencyResponse", _frequency_response)
register_model_adapter("RootLocus", _root_locus)
register_model_adapter("TimeResponse", _time_response)
register_shape_adapter(
    lambda d: isinstance(d.get("angular_frequencies"), list)
    and ("magnitude_db" in d or "magnitude" in d),
    _frequency_response)
register_shape_adapter(
    lambda d: isinstance(d.get("branches"), list) and "gains" in d,
    _root_locus)


# ----------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------

def _minimize_result(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    x = to_float_list(data["x"], what="minimizer")
    points = [[float(i), v] for i, v in enumerate(x)]
    return ProjectionSpec(
        kind="optimization", title="Optimization result",
        payload={"points": points,
                 "objective": to_float(data.get("f", data.get("objective")))},
        parameters={"converged": data.get("converged"),
                    "iterations": data.get("iterations"),
                    "coordinate": "component index"},
        information_loss=["summary"],
        information_loss_notes=[
            "Only the final iterate is shown; the solver does not record an "
            "iteration trace."])


def _multistart_result(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    runs = data["runs"]
    trace = [to_float(r.get("f", r.get("objective"))) for r in runs]
    return ProjectionSpec(
        kind="optimization", title="Multistart runs",
        payload={"trace": trace},
        parameters={"runs": len(runs), "ordering": "run order"})


register_shape_adapter(
    lambda d: isinstance(d.get("runs"), list) and "best" in d,
    _multistart_result)
register_shape_adapter(
    lambda d: isinstance(d.get("x"), list)
    and ("f" in d or "objective" in d)
    and ("iterations" in d or "converged" in d or d.get("status") == "optimal"),
    _minimize_result)


# ----------------------------------------------------------------------
# ODE / SDE / stochastic processes
# ----------------------------------------------------------------------

def _ode_ensemble(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    trajectories = data.get("trajectories")
    if isinstance(trajectories, list) and trajectories:
        members = [[to_float(v) for v in traj] for traj in trajectories]
        return ProjectionSpec(
            kind="ensemble", title="ODE ensemble",
            payload={"members": members},
            parameters={"ordering": "integration step order"})
    endpoints = [[to_float(c) for c in pt] for pt in data["endpoints"]]
    return ProjectionSpec(
        kind="point_cloud", title="ODE ensemble endpoints",
        payload={"points": endpoints},
        information_loss=["summary"],
        information_loss_notes=["Only trajectory endpoints are shown."])


def _sde_simulation(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    values = fields.get("values")
    times = fields.get("times")
    if not isinstance(values, list) or not values:
        return None
    # paths x time x state; present the first state component per path.
    members = [[to_float(v[0] if isinstance(v, (list, tuple)) else v)
                for v in path] for path in values]
    return ProjectionSpec(
        kind="ensemble", title="SDE simulation paths",
        payload={"members": members},
        parameters={"times": [to_float(t) for t in times] if times else None,
                    "state_component": 0,
                    "paths": len(members)},
        information_loss=["slice"] if any(
            isinstance(v, (list, tuple)) and len(v) > 1
            for path in values for v in path[:1]) else [],
        information_loss_notes=[
            "Only state component 0 of each path is shown."])


register_model_adapter("SDESimulation", _sde_simulation)
register_shape_adapter(
    lambda d: isinstance(d.get("endpoints"), list)
    or isinstance(d.get("trajectories"), list),
    _ode_ensemble)


# ----------------------------------------------------------------------
# FEM and PDE
# ----------------------------------------------------------------------

def _fem_mesh(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    points = [[to_float(c) for c in p] for p in fields["points"]]
    cells = [[int(i) for i in cell] for cell in fields.get("cells") or []]
    return ProjectionSpec(kind="mesh", title="FEM mesh",
                          payload={"vertices": points, "cells": cells})


def _fem_solution(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    values = to_float_list(fields.get("values"), what="solution values")
    system_id = fields.get("assembled_system_id")
    system = ctx.resolve_value(system_id)
    mesh_id = (system or {}).get("mesh_id")
    mesh = ctx.resolve_value(mesh_id)
    points = (mesh or {}).get("points")
    if isinstance(points, list) and points:
        pts = [[to_float(c) for c in p] for p in points]
        return ProjectionSpec(
            kind="pde_solution", title="FEM solution",
            payload={"points": pts, "values": values},
            parameters={"mesh_id": mesh_id})
    return ProjectionSpec(
        kind="function", title="FEM solution (nodal order)",
        payload={"x": list(range(len(values))), "y": values},
        parameters={"ordering": "node index"},
        information_loss=["ordering"],
        information_loss_notes=[
            "Mesh geometry could not be resolved; values are plotted against "
            "node index."])


def _pde_grid(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    u = data["u"]
    if u and isinstance(u[0], (list, tuple)):
        grid = [[to_float(v) for v in row] for row in u]
        return ProjectionSpec(kind="pde_solution", title="PDE solution",
                              payload={"grid": grid})
    values = to_float_list(u, what="u")
    return ProjectionSpec(
        kind="function", title="PDE solution",
        payload={"x": list(range(len(values))), "y": values},
        parameters={"ordering": "grid point order"})


def _pde_ensemble(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    members = [[to_float(v) for v in sol] for sol in data["solutions"]]
    return ProjectionSpec(
        kind="ensemble", title="PDE ensemble",
        payload={"members": members},
        parameters={"ordering": "grid point order"})


register_model_adapter("FEMMesh", _fem_mesh)
register_model_adapter("FEMSolution", _fem_solution)
register_shape_adapter(
    lambda d: isinstance(d.get("solutions"), list),
    _pde_ensemble)
register_shape_adapter(
    lambda d: isinstance(d.get("u"), list),
    _pde_grid)


# ----------------------------------------------------------------------
# Computational geometry
# ----------------------------------------------------------------------

def _point_set(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    points = [[to_float(c) for c in p] for p in fields["points"]]
    kind = "point_cloud" if points and len(points[0]) >= 3 else "point_set"
    return ProjectionSpec(kind=kind, title="Point set",
                          payload={"points": points})


def _polygon(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    boundary = [[to_float(c) for c in v] for v in fields["vertices"]]
    return ProjectionSpec(kind="region", title="Polygon",
                          payload={"boundary": boundary})


def _triangulation(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    points = [[to_float(c) for c in p] for p in fields["points"]]
    cells = [[int(i) for i in t] for t in fields.get("triangles") or []]
    return ProjectionSpec(kind="mesh", title="Triangulation",
                          payload={"vertices": points, "cells": cells})


def _voronoi(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    vertices = [[to_float(c) for c in v] for v in fields.get("vertices") or []]
    if not vertices:
        return None
    nodes = [str(v) for v in vertices]
    edges = []
    for e in fields.get("edges") or []:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            try:
                a, b = int(e[0]), int(e[1])
                edges.append({"source": nodes[a], "target": nodes[b]})
            except (TypeError, ValueError, IndexError):
                continue
    return ProjectionSpec(
        kind="graph", title="Voronoi diagram",
        payload={"nodes": nodes, "edges": edges},
        parameters={"layout": "coordinates in node labels"},
        information_loss=["summary"],
        information_loss_notes=[
            "Unbounded rays are not represented in the graph projection."])


register_model_adapter("PointSet", _point_set)
register_model_adapter("Polygon", _polygon)
register_model_adapter("Triangulation", _triangulation)
register_model_adapter("VoronoiDiagram", _voronoi)


# ----------------------------------------------------------------------
# Combinatorics and finite groups
# ----------------------------------------------------------------------

def _generating_function(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    coeffs = to_float_list(fields.get("coefficients"), what="coefficients")
    return ProjectionSpec(
        kind="function", title="Generating function coefficients",
        payload={"x": list(range(len(coeffs))), "y": coeffs},
        parameters={"ordering": "coefficient index"},
        coordinate_names=["n", "a_n"])


def _finite_group(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    table = [[int(v) for v in row] for row in fields["cayley_table"]]
    return ProjectionSpec(
        kind="matrix", title=f"Cayley table (order {fields.get('order', len(table))})",
        payload={"matrix": table},
        parameters={"entry_meaning": "product index"})


register_model_adapter("GeneratingFunction", _generating_function)
register_model_adapter("FiniteGroup", _finite_group)


# ----------------------------------------------------------------------
# Complex analysis and units
# ----------------------------------------------------------------------

def _singularity_points(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    raw = data.get("enclosed_singularities") or data.get("residues") or {}
    points: list[list[float]] = []
    if isinstance(raw, dict):
        keys = raw.keys()
    elif isinstance(raw, list):
        keys = raw
    else:
        return None
    for item in keys:
        point = item.get("point") if isinstance(item, dict) else item
        try:
            z = to_complex(point)
        except ValueError:
            continue
        points.append([z.real, z.imag])
    if not points:
        return None
    return ProjectionSpec(
        kind="complex_field", title="Enclosed singularities",
        payload={"points": points},
        coordinate_names=["Re", "Im"])


def _unit_quantity(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    return ProjectionSpec(
        kind="quantity", title="Quantity",
        payload={"value": to_float(data["value"]),
                 "unit": str(data.get("dimension", ""))})


register_shape_adapter(
    lambda d: isinstance(d.get("enclosed_singularities") or d.get("residues"),
                         (dict, list))
    and bool(d.get("enclosed_singularities") or d.get("residues")),
    _singularity_points)
register_shape_adapter(
    lambda d: "value" in d and "dimension" in d,
    _unit_quantity)


# ----------------------------------------------------------------------
# Tail tier: control models, survival/time-series fits, FEM diagnostics,
# group combinatorics, contours, discrete random variables
# ----------------------------------------------------------------------

def _zero_pole_gain(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    zeros = [to_complex(z) for z in fields.get("zeros") or []]
    poles = [to_complex(p) for p in fields.get("poles") or []]
    if not zeros and not poles:
        return None
    points = [[z.real, z.imag] for z in zeros] + [[p.real, p.imag] for p in poles]
    return ProjectionSpec(
        kind="complex_field", title="Pole-zero map",
        payload={"points": points},
        parameters={"zeros": len(zeros), "poles": len(poles),
                    "ordering": "zeros first, then poles",
                    "time_domain": fields.get("time_domain", "continuous")},
        coordinate_names=["Re", "Im"])


def _filter_taps(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    taps = to_float_list(fields.get("numerator"), what="numerator coefficients")
    return ProjectionSpec(
        kind="sequence", title="Filter numerator coefficients",
        payload={"values": taps},
        parameters={"ordering": "coefficient order (b[0], b[1], ...)",
                    "denominator": fields.get("denominator")},
        information_loss=["summary"],
        information_loss_notes=[
            "Only the numerator (feed-forward) coefficients are plotted; the "
            "denominator is retained in the source object."])


def _cox_ph_fit(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    timeline = to_float_list(fields.get("baseline_timeline"),
                             what="baseline timeline")
    hazard = to_float_list(fields.get("baseline_cumulative_hazard"),
                           what="baseline cumulative hazard")
    return ProjectionSpec(
        kind="function", title="Cox baseline cumulative hazard",
        payload={"x": timeline, "y": hazard},
        parameters={"predictors": list(fields.get("predictors") or []),
                    "coefficients": fields.get("coefficients"),
                    "concordance_index": fields.get("concordance_index")},
        coordinate_names=["t", "H0(t)"])


def _time_series_fit(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    residuals = to_float_list(fields.get("residuals"), what="residuals")
    return ProjectionSpec(
        kind="function", title=f"{fields.get('family', 'time-series').upper()} residuals",
        payload={"x": list(range(len(residuals))), "y": residuals},
        parameters={"ordering": "observation order",
                    "family": fields.get("family"),
                    "order": fields.get("order"),
                    "converged": fields.get("converged")},
        information_loss=["summary"],
        information_loss_notes=[
            "Residuals are plotted; fitted values and conditional variance "
            "are retained in the source object."])


def _contour(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    vertices = fields.get("vertices") or []
    boundary = []
    for v in vertices:
        z = to_complex(v)
        boundary.append([z.real, z.imag])
    if len(boundary) < 3:
        return None
    return ProjectionSpec(
        kind="region", title="Contour",
        payload={"boundary": boundary},
        parameters={"orientation": fields.get("orientation"),
                    "closed": True},
        coordinate_names=["Re", "Im"])


def _assembled_system_spy(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    entries = fields.get("raw_matrix_entries") or []
    points = [[float(e["row"]), float(e["column"])] for e in entries
              if isinstance(e, dict) and "row" in e and "column" in e]
    if not points:
        return None
    return ProjectionSpec(
        kind="point_set", title="Assembled system sparsity pattern",
        payload={"points": points},
        parameters={"dof_count": fields.get("dof_count"),
                    "coordinate": "(row, column)"},
        information_loss=["summary"],
        information_loss_notes=[
            "Only the sparsity pattern is shown; entry values are retained "
            "in the source object."])


def _fem_error_estimate(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    indicators = fields.get("cell_indicators") or []
    pairs = sorted(
        ((int(e["cell_index"]), to_float(e["total_indicator_squared"]))
         for e in indicators
         if isinstance(e, dict) and "cell_index" in e),
        key=lambda pair: pair[0])
    if not pairs:
        return None
    return ProjectionSpec(
        kind="function", title="FEM cell error indicators",
        payload={"x": [float(i) for i, _ in pairs],
                 "y": [v for _, v in pairs]},
        parameters={"quantity": "total_indicator_squared",
                    "ordering": "cell index",
                    "global_estimator": fields.get("global_estimator")},
        information_loss_notes=[
            "Squared indicators are plotted exactly as stored; no square "
            "root is applied."],
        coordinate_names=["cell", "indicator^2"])


def _fem_convergence_observation(fields: dict,
                                 ctx: AdapterContext) -> ProjectionSpec | None:
    rate = fields.get("observed_estimator_rate")
    if rate is None:
        return None
    return ProjectionSpec(
        kind="quantity", title="Observed estimator rate",
        payload={"value": to_float(rate)},
        parameters={"coarse_cell_count": fields.get("coarse_cell_count"),
                    "fine_cell_count": fields.get("fine_cell_count"),
                    "estimator_ratio": fields.get("estimator_ratio"),
                    "interpretation": fields.get("interpretation")})


def _subgroups_partition(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    subgroups = data["subgroups"]
    parts = [list(sg.get("elements", [])) if isinstance(sg, dict) else list(sg)
             for sg in subgroups]
    return ProjectionSpec(
        kind="partition", title="Subgroups",
        payload={"parts": parts},
        parameters={"complete": data.get("complete"),
                    "part_meaning": "subgroup element lists"})


def _list_partition(key: str, title: str, meaning: str):
    def adapt(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
        parts = [list(part) for part in data[key]]
        return ProjectionSpec(
            kind="partition", title=title,
            payload={"parts": parts},
            parameters={"part_meaning": meaning})
    return adapt


def _count_result(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    if data.get("value") is None:
        return None
    return ProjectionSpec(
        kind="quantity", title=f"Count ({data.get('kind', 'combinatorial')})",
        payload={"value": to_float(data["value"])},
        parameters={"n": data.get("n"), "k": data.get("k"),
                    "exact_value": str(data["value"])})


def _coefficient_result(data: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    if data.get("value") is None:
        return None
    return ProjectionSpec(
        kind="quantity", title="Generating function coefficient",
        payload={"value": to_float(data["value"])},
        parameters={"index": data.get("index"),
                    "scaling": data.get("scaling"),
                    "exact_value": str(data["value"])})


def _discrete_rv(fields: dict, ctx: AdapterContext) -> ProjectionSpec | None:
    values = to_float_list(fields.get("values"), what="values")
    probs = to_float_list(fields.get("probabilities"), what="probabilities")
    if len(values) != len(probs):
        raise ValueError("values and probabilities must have equal length")
    return ProjectionSpec(
        kind="distribution", title="Discrete random variable",
        payload={"x": values, "pdf": probs},
        parameters={"representation": "probability mass per support point"})


register_model_adapter("ZeroPoleGain", _zero_pole_gain)
register_model_adapter("Filter", _filter_taps)
register_model_adapter("CoxPHFit", _cox_ph_fit)
register_model_adapter("TimeSeriesFit", _time_series_fit)
register_model_adapter("Contour", _contour)
register_model_adapter("AssembledSystem", _assembled_system_spy)
register_model_adapter("FEMErrorEstimate", _fem_error_estimate)
register_model_adapter("FEMConvergenceObservation", _fem_convergence_observation)
register_model_adapter("DiscreteRV", _discrete_rv)
register_shape_adapter(
    lambda d: isinstance(d.get("subgroups"), list) and bool(d["subgroups"]),
    _subgroups_partition)
register_shape_adapter(
    lambda d: isinstance(d.get("cosets"), list) and bool(d["cosets"]),
    _list_partition("cosets", "Cosets", "coset element lists"))
register_shape_adapter(
    lambda d: isinstance(d.get("orbits"), list) and bool(d["orbits"]),
    _list_partition("orbits", "Orbits", "orbit element lists"))
register_shape_adapter(
    lambda d: isinstance(d.get("classes"), list) and bool(d["classes"]),
    _list_partition("classes", "Conjugacy classes", "class element lists"))
register_shape_adapter(
    lambda d: d.get("value") is not None and "n" in d and "kind" in d,
    _count_result)
register_shape_adapter(
    lambda d: d.get("value") is not None and "index" in d and "scaling" in d,
    _coefficient_result)
