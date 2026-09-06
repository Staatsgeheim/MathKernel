# =============================================================================
# MathKernel - regression tests for the projection adapter registry
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Adapters must convert real MathKernel result/object shapes into canonical
projections without inventing structure, upgrading trust, or hiding sampling.
"""
import pytest

from mathkernel import MathKernel
from mathkernel_projection import from_mathresult
from mathkernel_projection.result_adapters import (AdapterContext, adapt_result,
                                                   encoded_fields)

CTX = AdapterContext(trust="exact")


def adapt_model(name, fields, ctx=CTX):
    spec = adapt_result({"__pydantic_model__": f"mathkernel.x:{name}",
                         "fields": fields}, ctx)
    assert spec is not None, f"no adapter claimed {name}"
    return spec


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------

def test_registry_dispatches_on_bare_class_name():
    spec = adapt_model("DiscreteSignal", {"samples": ["1", "2"], "sample_rate": "2",
                                          "start": "0"})
    assert spec.kind == "function"
    assert spec.payload["x"] == [0.0, 0.5]
    assert spec.payload["y"] == [1.0, 2.0]


def test_registry_returns_none_for_unknown_shape():
    assert adapt_result({"unrelated": 1}, CTX) is None


def test_encoded_fields_unwraps_envelope():
    enc = {"__pydantic_model__": "m:C", "fields": {"a": 1}}
    assert encoded_fields(enc) == {"a": 1}
    assert encoded_fields({"a": 1}) == {"a": 1}


def test_from_mathresult_prefers_registry():
    result = {"data": {"__pydantic_model__": "mathkernel.signal_processing:Spectrum",
                       "fields": {"bins": ["1", "0"], "sample_rate": "4"}},
              "trust": "exact", "engine": "signal"}
    proj = from_mathresult(result)
    assert proj.kind == "spectrum"
    assert proj.metadata.get("adapter") == "registry"


def test_from_mathresult_falls_back_to_shapes():
    result = {"data": {"matrix": [[1, 2], [3, 4]]}, "trust": "exact"}
    proj = from_mathresult(result)
    assert proj.kind == "matrix"


# ---------------------------------------------------------------------------
# Signal / control
# ---------------------------------------------------------------------------

def test_continuous_signal_declares_sampling():
    spec = adapt_model("ContinuousSignal", {"expression": "sin(t)", "variable": "t",
                                            "start": "0", "end": "1"})
    assert spec.kind == "function"
    assert "sampling" in spec.information_loss
    assert spec.parameters["range"] == [0.0, 1.0]
    assert len(spec.payload["x"]) == spec.parameters["samples"]


def test_frequency_response_uses_db_channel():
    spec = adapt_model("FrequencyResponse",
                       {"angular_frequencies": ["1", "10"], "magnitude_db": ["0", "-3"],
                        "phase_radians": ["0", "-1"]})
    assert spec.kind == "function"
    assert spec.payload["y"] == [0.0, -3.0]
    assert "summary" in spec.information_loss


def test_root_locus_flattens_complex_branches():
    spec = adapt_model("RootLocus", {"gains": ["0", "1"],
                                     "branches": [["-1 + I", "-1 - I"],
                                                  ["-2 + 2*I", "-2 - 2*I"]]})
    assert spec.kind == "complex_field"
    assert [-2.0, 2.0] in spec.payload["points"]
    assert len(spec.payload["points"]) == 4


def test_time_response_samples_symbolic_expression():
    spec = adapt_model("TimeResponse", {"expression": "exp(-t)", "variable": "t",
                                        "start": "0", "end": "2"})
    assert spec.kind == "function"
    assert "sampling" in spec.information_loss
    assert spec.payload["y"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Probability / statistics / survival / time series
# ---------------------------------------------------------------------------

def test_distribution_pdf_sampling_declared():
    spec = adapt_model("Distribution", {
        "name": "normal", "density": "exp(-x**2/2)/sqrt(2*pi)",
        "variable": "x", "support": "Reals",
        "known_results": {"mean": "0", "variance": "1"}})
    assert spec.kind == "distribution"
    assert "pdf" in spec.payload
    assert "sampling" in spec.information_loss
    assert spec.parameters["range"] == [-4.0, 4.0]
    assert spec.parameters["range_basis"] == "mean +/- 4 standard deviations"


def test_distribution_finite_support_window():
    spec = adapt_model("Distribution", {
        "name": "uniform", "density": "1/2", "variable": "x",
        "support": "Interval(0, 2)", "known_results": {}})
    assert spec.parameters["range"] == [0.0, 2.0]
    assert spec.parameters["range_basis"] == "support"


def test_empirical_distribution_pmf():
    spec = adapt_model("EmpiricalDistribution",
                       {"support": ["1", "2"], "probabilities": ["1/3", "2/3"]})
    assert spec.kind == "distribution"
    assert spec.payload["pdf"] == pytest.approx([1 / 3, 2 / 3])


def test_statistical_sample_histogram():
    spec = adapt_model("StatisticalSample", {"observations": ["1", "2", "3"]})
    assert spec.kind == "statistical_inference"
    assert spec.payload["values"] == [1.0, 2.0, 3.0]


def test_kaplan_meier_step_note():
    spec = adapt_model("KaplanMeierEstimate",
                       {"timeline": ["0", "1"], "survival": ["1", "1/2"]})
    assert spec.kind == "function"
    assert spec.payload["y"] == [1.0, 0.5]
    assert any("step" in n for n in spec.information_loss_notes)


def test_time_series_analysis_lags():
    spec = adapt_model("TimeSeriesAnalysis",
                       {"kind": "acf", "lags": ["0", "1"], "values": ["1", "1/2"]})
    assert spec.kind == "function"
    assert spec.payload["x"] == [0.0, 1.0]


def test_prob_sample_shape_adapter():
    spec = adapt_result({"samples": ["1", "2", "2"], "n": 3}, CTX)
    assert spec.kind == "distribution"
    assert spec.payload["values"] == [1.0, 2.0, 2.0]


def test_confidence_interval_shape_adapter():
    spec = adapt_result({"confidence": "19/20", "mean": "0", "lower": "-1",
                         "upper": "1"}, CTX)
    assert spec.kind == "intervals"
    assert spec.payload["intervals"] == [{"lower": -1.0, "upper": 1.0}]


def test_forecast_intervals_shape_adapter():
    spec = adapt_result({"times": ["0", "1"], "means": ["2", "3"],
                         "lower": ["1", "2"], "upper": ["3", "4"]}, CTX)
    assert spec.kind == "intervals"
    assert spec.payload["intervals"][0] == {"x": 0.0, "lower": 1.0, "upper": 3.0}


def test_gp_posterior_band_declared_summary():
    spec = adapt_result({"times": ["0"], "means": ["1"],
                         "covariance": [["1/4"]]}, CTX)
    assert spec.kind == "intervals"
    assert spec.payload["intervals"][0]["lower"] == pytest.approx(0.0)
    assert "summary" in spec.information_loss


# ---------------------------------------------------------------------------
# Graph / optimization / ODE / SDE
# ---------------------------------------------------------------------------

def test_graph_model_adapter():
    spec = adapt_model("Graph", {"vertices": ["a", "b"],
                                 "edges": [{"source": "a", "target": "b"}]})
    assert spec.kind == "graph"
    assert spec.payload["edges"] == [{"source": "a", "target": "b"}]


def test_traversal_parent_tree_shape_adapter():
    spec = adapt_result({"source": "a", "order": ["a", "b"],
                         "parent": {"a": None, "b": "a"}}, CTX)
    assert spec.kind == "graph"
    assert spec.payload["edges"] == [{"source": "a", "target": "b"}]
    assert spec.parameters["visit_order"] == ["a", "b"]


def test_minimize_result_shape_adapter():
    spec = adapt_result({"x": ["1", "2"], "f": "3", "iterations": 5,
                         "converged": True}, CTX)
    assert spec.kind == "optimization"
    assert spec.payload["objective"] == 3.0
    assert "summary" in spec.information_loss


def test_multistart_trace_shape_adapter():
    spec = adapt_result({"best": {"x": ["0"], "f": "1"},
                         "runs": [{"x": ["0"], "f": "1"},
                                  {"x": ["2"], "f": "5"}]}, CTX)
    assert spec.kind == "optimization"
    assert spec.payload["trace"] == [1.0, 5.0]


def test_ode_ensemble_endpoints():
    spec = adapt_result({"endpoints": [["1", "2"], ["3", "4"]], "steps": 10}, CTX)
    assert spec.kind == "point_cloud"
    assert spec.payload["points"] == [[1.0, 2.0], [3.0, 4.0]]


def test_sde_simulation_first_component_slice():
    spec = adapt_model("SDESimulation", {
        "times": ["0", "1"],
        "values": [[["0", "9"], ["1", "8"]], [["2", "7"], ["3", "6"]]]})
    assert spec.kind == "ensemble"
    assert spec.payload["members"] == [[0.0, 1.0], [2.0, 3.0]]
    assert "slice" in spec.information_loss


# ---------------------------------------------------------------------------
# FEM / PDE / geometry / combinatorics / groups / complex / units
# ---------------------------------------------------------------------------

def test_fem_mesh_adapter():
    spec = adapt_model("FEMMesh", {"points": [["0", "0"], ["1", "0"], ["0", "1"]],
                                   "cells": [[0, 1, 2]]})
    assert spec.kind == "mesh"
    assert spec.payload["cells"] == [[0, 1, 2]]


def test_fem_solution_resolves_mesh_geometry():
    store = {
        "sys1": {"mesh_id": "mesh1"},
        "mesh1": {"points": [["0", "0"], ["1", "0"]]},
    }
    ctx = AdapterContext(trust="exact", resolve=store.get)
    spec = adapt_model("FEMSolution", {"values": ["2", "3"],
                                       "assembled_system_id": "sys1"}, ctx)
    assert spec.kind == "pde_solution"
    assert spec.payload["values"] == [2.0, 3.0]
    assert spec.payload["points"] == [[0.0, 0.0], [1.0, 0.0]]


def test_fem_solution_without_mesh_falls_back_to_nodal_order():
    spec = adapt_model("FEMSolution", {"values": ["2", "3"],
                                       "assembled_system_id": "missing"})
    assert spec.kind == "function"
    assert "ordering" in spec.information_loss


def test_pde_2d_grid_shape_adapter():
    spec = adapt_result({"u": [["1", "2"], ["3", "4"]], "engine_tier": "python"}, CTX)
    assert spec.kind == "pde_solution"
    assert spec.payload["grid"] == [[1.0, 2.0], [3.0, 4.0]]


def test_pde_1d_shape_adapter():
    spec = adapt_result({"u": ["1", "2", "3"]}, CTX)
    assert spec.kind == "function"
    assert spec.payload["y"] == [1.0, 2.0, 3.0]


def test_point_set_and_polygon_and_triangulation():
    assert adapt_model("PointSet", {"points": [["0", "0"]]}).kind == "point_set"
    poly = adapt_model("Polygon", {"vertices": [["0", "0"], ["1", "0"], ["0", "1"]]})
    assert poly.kind == "region" and len(poly.payload["boundary"]) == 3
    tri = adapt_model("Triangulation", {"points": [["0", "0"], ["1", "0"], ["0", "1"]],
                                        "triangles": [[0, 1, 2]]})
    assert tri.kind == "mesh"


def test_generating_function_coefficients():
    spec = adapt_model("GeneratingFunction", {"coefficients": ["1", "1", "2", "3"]})
    assert spec.kind == "function"
    assert spec.payload["y"] == [1.0, 1.0, 2.0, 3.0]


def test_finite_group_cayley_table():
    spec = adapt_model("FiniteGroup", {"order": 2,
                                       "cayley_table": [[0, 1], [1, 0]]})
    assert spec.kind == "matrix"
    assert spec.payload["matrix"] == [[0, 1], [1, 0]]


def test_contour_singularities_shape_adapter():
    spec = adapt_result({"integral": "2*pi*I",
                         "enclosed_singularities": {"I": "2*pi*I"}}, CTX)
    assert spec.kind == "complex_field"
    assert spec.payload["points"] == [[0.0, 1.0]]


def test_unit_convert_quantity_shape_adapter():
    spec = adapt_result({"value": "1000", "dimension": "meter"}, CTX)
    assert spec.kind == "quantity"
    assert spec.payload["value"] == 1000.0


# ---------------------------------------------------------------------------
# Kernel integration: object_id paths
# ---------------------------------------------------------------------------

@pytest.fixture
def kernel():
    return MathKernel()


def test_viz_create_from_signal_object(kernel):
    obj = kernel.object_create("signal", {"samples": ["1", "2", "3", "4"],
                                          "sample_rate": "4"})
    assert obj.ok, obj.errors
    oid = obj.data["object_id"]
    viz = kernel.viz_create(object_id=oid)
    assert viz.ok, viz.errors
    assert viz.data["blocks"] >= 1
    proj = kernel.projection_create(source_object_id=oid)
    assert proj.ok, proj.errors
    assert proj.data["kind"] == "function"
    assert proj.data["trust"] in {"exact", "numeric"}


def test_projection_create_requires_kind_or_object(kernel):
    result = kernel.projection_create()
    assert not result.ok
    assert "kind" in result.errors[0]


def test_viz_create_unknown_object_type_errors(kernel):
    viz = kernel.viz_create(object_id="does-not-exist")
    assert not viz.ok


def test_viz_create_unregistered_object_type_errors(kernel):
    obj = kernel.object_create("transfer_function",
                               {"numerator": ["1"], "denominator": ["1", "1"]})
    assert obj.ok, obj.errors
    viz = kernel.viz_create(object_id=obj.data["object_id"])
    assert not viz.ok
    assert "no projection adapter" in viz.errors[0]


# ---------------------------------------------------------------------------
# Phase 2 tail tier
# ---------------------------------------------------------------------------

def test_zero_pole_gain_pole_zero_map():
    spec = adapt_model("ZeroPoleGain", {"zeros": ["-1"], "poles": ["-2 + I", "-2 - I"],
                                        "gain": "1"})
    assert spec.kind == "complex_field"
    assert spec.payload["points"] == [[-1.0, 0.0], [-2.0, 1.0], [-2.0, -1.0]]
    assert spec.parameters["zeros"] == 1 and spec.parameters["poles"] == 2


def test_filter_taps_declared_summary():
    spec = adapt_model("Filter", {"numerator": ["1/2", "1/2"], "denominator": ["1"]})
    assert spec.kind == "sequence"
    assert spec.payload["values"] == [0.5, 0.5]
    assert "summary" in spec.information_loss


def test_cox_ph_baseline_hazard():
    spec = adapt_model("CoxPHFit", {
        "baseline_timeline": ["0", "1"], "baseline_cumulative_hazard": ["0", "1/4"],
        "predictors": ["age"], "coefficients": ["1/10"]})
    assert spec.kind == "function"
    assert spec.payload["y"] == [0.0, 0.25]


def test_time_series_fit_residuals():
    spec = adapt_model("TimeSeriesFit", {"family": "ar", "order": [1, 0, 0],
                                         "residuals": ["1", "-1"],
                                         "converged": True})
    assert spec.kind == "function"
    assert spec.payload["y"] == [1.0, -1.0]
    assert "summary" in spec.information_loss


def test_time_series_analysis_uses_analysis_field():
    spec = adapt_model("TimeSeriesAnalysis",
                       {"analysis": "pacf", "lags": ["0", "1"], "values": ["1", "1/2"]})
    assert spec.title == "PACF"


def test_contour_region_boundary():
    spec = adapt_model("Contour", {"vertices": ["0", "1", "1 + I", "I"],
                                   "orientation": "ccw"})
    assert spec.kind == "region"
    assert [1.0, 1.0] in spec.payload["boundary"]


def test_assembled_system_spy_pattern():
    spec = adapt_model("AssembledSystem", {
        "raw_matrix_entries": [{"row": 0, "column": 1, "value": "2"}],
        "dof_count": 2})
    assert spec.kind == "point_set"
    assert spec.payload["points"] == [[0.0, 1.0]]
    assert "summary" in spec.information_loss


def test_fem_error_estimate_indicators():
    spec = adapt_model("FEMErrorEstimate", {
        "cell_indicators": [{"cell_index": 1, "total_indicator_squared": "1/4"},
                            {"cell_index": 0, "total_indicator_squared": "1"}],
        "global_estimator": "5/4"})
    assert spec.kind == "function"
    assert spec.payload["x"] == [0.0, 1.0]
    assert spec.payload["y"] == [1.0, 0.25]


def test_fem_convergence_observation_quantity():
    spec = adapt_model("FEMConvergenceObservation",
                       {"observed_estimator_rate": 0.5, "coarse_cell_count": 4,
                        "fine_cell_count": 16})
    assert spec.kind == "quantity"
    assert spec.payload["value"] == 0.5


def test_subgroups_partition():
    spec = adapt_result({"subgroups": [{"elements": [0, 1], "order": 2}],
                         "complete": True}, CTX)
    assert spec.kind == "partition"
    assert spec.payload["parts"] == [[0, 1]]


def test_cosets_and_orbits_partitions():
    assert adapt_result({"cosets": [[0, 1], [2, 3]]}, CTX).kind == "partition"
    assert adapt_result({"orbits": [[0, 1], [2]]}, CTX).kind == "partition"


def test_count_and_coefficient_quantities():
    count = adapt_result({"kind": "binomial", "n": 5, "k": 2, "value": 10}, CTX)
    assert count.kind == "quantity" and count.payload["value"] == 10.0
    coeff = adapt_result({"index": 4, "value": 5, "scaling": "ordinary"}, CTX)
    assert coeff.kind == "quantity" and coeff.parameters["index"] == 4


def test_discrete_rv_resolution(kernel):
    rv = kernel.prob_rv_create(["0", "1"], ["1/2", "1/2"])
    assert rv.ok, rv.errors
    viz = kernel.viz_create(object_id=rv.data["rv_id"])
    assert viz.ok, viz.errors
    assert viz.data["blocks"] >= 1
    proj = kernel.projection_create(source_object_id=rv.data["rv_id"])
    assert proj.ok, proj.errors
    assert proj.data["kind"] == "distribution"
    assert proj.data["trust"] == "exact"


def test_spectrum_complex_bins_present_magnitude():
    spec = adapt_model("Spectrum", {"bins": ["0", "-4*I", "0", "4*I"],
                                    "sample_rate": "8"})
    assert spec.kind == "spectrum"
    assert spec.payload["amplitude"] == [0.0, 4.0, 0.0, 4.0]
    assert "summary" in spec.information_loss
