import numpy as np
import pytest
import sympy as sp

from mathkernel import MathKernel, Settings, TrustLevel
from test_engineering_signal import create


def test_continuous_signal_exact_and_vectorized_sampling_metadata():
    k = MathKernel()
    context = k.create_context(domains={"t": "real"})
    source = create(k, "ContinuousSignal", expression="t^2", start=0, end=1,
                    unit="V", context_id=context.context_id)
    exact = k.apply(source, "sample", {"sample_rate": 4, "count": 5})
    assert exact.ok, exact.errors
    assert exact.data["value"] == ["0", "1/16", "1/4", "9/16", "1"]
    assert exact.data["details"]["bandlimited_assumption"] is False
    numeric = k.apply(source, "sample", {"sample_rate": 4, "count": 5, "mode": "numeric"})
    assert numeric.ok, numeric.errors
    assert numeric.trust == TrustLevel.NUMERIC
    assert numeric.data["verification"]["scalar_substitution_consistency"]


def test_continuous_signal_rejects_undeclared_variables_and_out_of_interval_samples():
    k = MathKernel()
    context = k.create_context(domains={"t": "real", "x": "real"})
    assert not k.object_create("ContinuousSignal", {"expression": "t+x", "end": 1,
        "context_id": context.context_id}).ok
    source = create(k, "ContinuousSignal", expression="t", end=1, context_id=context.context_id)
    assert not k.apply(source, "sample", {"sample_rate": 2, "count": 4}).ok


@pytest.mark.parametrize("mode", ["exact", "numeric"])
def test_zpk_transfer_roundtrip_preserves_unreduced_multiplicity(mode):
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=[2, 6, 4], denominator=[4, 12, 8])
    zpk = k.apply(source, "to_zero_pole_gain", {"mode": mode})
    assert zpk.ok, zpk.errors
    model = k.math_objects[zpk.data["object_id"]]["value"]
    assert len(model.zeros) == len(model.poles) == 2
    back = k.apply(zpk.data["object_id"], "to_transfer_function")
    assert back.ok, back.errors
    original = k.math_objects[source]["value"].expression()
    restored = k.math_objects[back.data["object_id"]]["value"].expression()
    if mode == "exact":
        assert sp.cancel(original-restored) == 0
    else:
        assert complex((original-restored).subs(sp.Symbol("s"), 3).evalf()).real == pytest.approx(0, abs=1e-12)
        assert zpk.trust == TrustLevel.NUMERIC


def test_direct_zpk_construction_and_discrete_timing_rules():
    k = MathKernel()
    zpk = create(k, "ZeroPoleGain", zeros=[-1], poles=[-2], gain=3)
    tf = k.apply(zpk, "to_transfer_function")
    assert tf.ok and tf.data["verification"]["factor_identity"]
    assert not k.object_create("ZeroPoleGain", {"poles": [0], "time_domain": "discrete"}).ok
    discrete = create(k, "ZeroPoleGain", poles=["1/2"], time_domain="discrete", sample_time="1/10")
    assert k.apply(discrete, "poles").data["value"] == ["1/2"]


def test_internal_poles_and_transfer_zeros_have_explicit_scopes():
    k = MathKernel()
    state = create(k, "StateSpaceSystem", A=[[-1, 0], [0, 2]], B=[[1], [0]], C=[[1, 0]], D=[[0]])
    poles = k.apply(state, "poles")
    zeros = k.apply(state, "zeros")
    assert set(poles.data["value"]) == {"-1", "2"}
    assert poles.data["details"]["scope"] == "internal_state_modes"
    assert zeros.data["details"]["scope"] == "transfer_numerator_roots"


def test_bode_nyquist_and_frequency_response_are_typed_and_cross_consistent():
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, 1])
    parameters = {"frequencies": [0, 1, 2], "mode": "numeric"}
    frequency = k.apply(source, "frequency_response", parameters)
    bode = k.apply(source, "bode", parameters)
    nyquist = k.apply(source, "nyquist", parameters)
    assert frequency.data["object_type"] == bode.data["object_type"] == "FrequencyResponse"
    np.testing.assert_allclose(frequency.data["details"]["magnitude"], bode.data["details"]["magnitude"])
    curve = k.math_objects[nyquist.data["object_id"]]["value"]
    assert len(curve.response) == 5
    assert complex(curve.response[0]) == pytest.approx(complex(curve.response[-1]).conjugate())
    assert nyquist.data["details"]["nyquist_encirclement_claimed"] is False


def test_discrete_frequency_range_is_bounded_by_physical_nyquist():
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, "-1/2"],
                    time_domain="discrete", sample_time="1/10")
    assert k.apply(source, "bode", {"frequencies": [0, 10], "mode": "numeric"}).ok
    assert not k.apply(source, "bode", {"frequencies": [32], "mode": "numeric"}).ok


def test_bode_represents_sampled_zero_and_nyquist_rejects_complex_completion():
    k = MathKernel()
    zero = create(k, "TransferFunction", numerator=[1, 0], denominator=[1, 1])
    bode = k.apply(zero, "bode", {"frequencies": [0, 1], "mode": "numeric"})
    assert bode.ok and bode.data["details"]["magnitude_db"][0] is None
    complex_tf = create(k, "TransferFunction", numerator=["complex(1,1)"], denominator=[1, 1])
    assert not k.apply(complex_tf, "nyquist", {"frequencies": [0, 1], "mode": "numeric"}).ok


def test_root_locus_residuals_and_branch_semantics():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, 3, 2])
    result = k.apply(source, "root_locus", {"gains": [0, 1, 10], "mode": "numeric"})
    assert result.ok, result.errors
    assert result.data["verification"]["characteristic_residual"]
    assert result.data["details"]["branch_ordering_certified"] is False
    locus = k.math_objects[result.data["object_id"]]["value"]
    np.testing.assert_allclose(sorted(complex(v).real for v in locus.branches[0]), [-2, -1])
    assert not k.apply(source, "root_locus", {"gains": [1, 0], "mode": "numeric"}).ok
    assert not k.apply(source, "root_locus", {"gains": [0, 1]}).ok


def test_time_response_is_typed_and_complete_about_conventions():
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, 1],
                    input_unit="V", output_unit="A")
    result = k.apply(source, "step_response")
    assert result.ok and result.data["object_type"] == "TimeResponse"
    assert result.data["details"]["initial_state"] == "zero"
    assert result.data["details"]["distribution_at_t_zero_included"] is False
    response = k.math_objects[result.data["object_id"]]["value"]
    assert response.output_unit == "A" and response.time_unit == "s"


def test_explicit_discrete_control_representation_and_identity_conversion():
    k = MathKernel()
    discrete = create(k, "DiscreteControlSystem", A=[["1/2"]], B=[[1]], C=[[1]], D=[[0]], sample_time="1/10")
    converted = k.apply(discrete, "to_state_space")
    assert converted.ok and converted.data["verification"]["fields_identical"]
    standard = create(k, "StateSpaceSystem", A=[["1/2"]], B=[[1]], C=[[1]], D=[[0]],
                      time_domain="discrete", sample_time="1/10")
    explicit = k.apply(standard, "to_discrete_control")
    assert explicit.ok and explicit.data["object_type"] == "DiscreteControlSystem"
    continuous = create(k, "StateSpaceSystem", A=[[-1]], B=[[1]], C=[[1]], D=[[0]])
    assert not k.apply(continuous, "to_discrete_control").ok


def test_windowed_fir_family_has_symmetric_coefficients_and_direct_execution():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k = MathKernel()
    design = create(k, "FilterDesign", family="fir_window", order=8, cutoff=[10],
                    sample_rate=100, window="blackman")
    result = k.apply(design, "design", {"mode": "numeric"})
    assert result.ok, result.errors
    filt = k.math_objects[result.data["object_id"]]["value"]
    assert len(filt.numerator) == 9 and len(filt.denominator) == 1 and float(filt.denominator[0]) == 1
    np.testing.assert_allclose([float(v) for v in filt.numerator], [float(v) for v in reversed(filt.numerator)], atol=1e-14)
    assert result.data["details"]["implementation"] == "linear_phase_windowed_fir"


def test_signal_analysis_objects_survive_restricted_persistence(tmp_path):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    settings = Settings(store_path=str(tmp_path/"d7.sqlite"))
    k = MathKernel(settings)
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, 1])
    objects = [k.apply(source, "to_zero_pole_gain").data["object_id"],
               k.apply(source, "bode", {"frequencies": [0, 1], "mode": "numeric"}).data["object_id"],
               k.apply(source, "root_locus", {"gains": [0, 1], "mode": "numeric"}).data["object_id"],
               k.apply(source, "step_response").data["object_id"]]
    restarted = MathKernel(settings)
    assert all(restarted.object_get(object_id).ok for object_id in objects)


def test_exact_high_degree_algebraic_zpk_root_survives_safe_decoder(tmp_path):
    settings = Settings(store_path=str(tmp_path/"rootof.sqlite"), max_control_order=8)
    k = MathKernel(settings)
    source = create(k, "TransferFunction", numerator=[1], denominator=[1, 0, 0, 0, -1, 1])
    result = k.apply(source, "to_zero_pole_gain")
    assert result.ok and len(k.math_objects[result.data["object_id"]]["value"].poles) == 5
    assert MathKernel(settings).object_get(result.data["object_id"]).ok


def test_signal_analysis_capability_discovery_and_public_exports():
    from mathkernel import (ContinuousSignal, DiscreteControlSystem, FrequencyResponse,
                            RootLocus, TimeResponse, ZeroPoleGain)
    assert all((ContinuousSignal, DiscreteControlSystem, FrequencyResponse, RootLocus, TimeResponse, ZeroPoleGain))
    k = MathKernel()
    assert k.capability_query(domain="signal", input_type="ContinuousSignal", operation="sample")["count"] == 1
    assert k.capability_query(domain="control", input_type="TransferFunction", operation="root_locus")["count"] == 1


def test_signal_analysis_numeric_outputs_keep_decimal_ancestry():
    k = MathKernel()
    source = create(k, "TransferFunction", numerator=["0.5"], denominator=[1, 1])
    result = k.apply(source, "bode", {"frequencies": [0, 1], "mode": "numeric"})
    assert result.trust == TrustLevel.NUMERIC
    child = k.object_get(result.data["object_id"])
    assert child.trust == TrustLevel.NUMERIC and child.data["sources"] == [source]


def test_signal_analysis_models_are_immutable():
    from pydantic import ValidationError
    from mathkernel.control_analysis import ZeroPoleGain
    model = ZeroPoleGain(poles=(sp.Integer(-1),))
    with pytest.raises(ValidationError):
        model.gain = sp.Integer(2)


def test_mimo_state_zeros_and_curves_are_explicitly_unsupported_but_poles_work():
    k = MathKernel()
    source = create(k, "StateSpaceSystem", A=[[-1]], B=[[1, 0]], C=[[1], [0]], D=[[0, 0], [0, 0]])
    assert k.apply(source, "poles").ok
    assert not k.apply(source, "zeros").ok
    assert not k.apply(source, "bode", {"frequencies": [0, 1], "mode": "numeric"}).ok


def test_signal_analysis_resource_limits_cover_roots_samples_and_curve_points():
    k = MathKernel(Settings(max_control_order=2, max_signal_samples=3))
    assert not k.object_create("ZeroPoleGain", {"poles": [-1, -2, -3]}).ok
    context = k.create_context(domains={"t": "real"})
    signal = create(k, "ContinuousSignal", expression="t", end=2, context_id=context.context_id)
    assert not k.apply(signal, "sample", {"sample_rate": 2, "count": 4}).ok
    tf = create(k, "TransferFunction", numerator=[1], denominator=[1, 1])
    assert not k.apply(tf, "bode", {"frequencies": [0, 1, 2, 3], "mode": "numeric"}).ok


@pytest.mark.parametrize("kind,order", [("lowpass", 8), ("highpass", 8), ("bandpass", 8), ("bandstop", 8)])
def test_windowed_fir_common_response_kinds(kind, order):
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    k = MathKernel()
    cutoff = [10, 20] if kind.startswith("band") else [10]
    spec = create(k, "FilterDesign", family="fir_window", order=order, cutoff=cutoff,
                  sample_rate=100, kind=kind)
    result = k.apply(spec, "design", {"mode": "numeric"})
    assert result.ok, result.errors
