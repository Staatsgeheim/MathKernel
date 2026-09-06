import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import sympy as sp

from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.engines import (IsolatedSolverError, SolverTimeoutError,
    _isolated_blob, _isolated_probe, run_in_subprocess)
from test_engineering_signal import create


def test_isolated_worker_runs_in_a_distinct_fresh_process():
    child = run_in_subprocess(_isolated_probe, 5.0)
    assert child != os.getpid()


def test_timeout_hard_kills_the_native_worker(monkeypatch):
    from mathkernel import engines
    original = engines.subprocess.Popen
    children = []
    def tracked(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(engines.subprocess, "Popen", tracked)
    with pytest.raises(SolverTimeoutError, match="isolated process terminated"):
        run_in_subprocess(_isolated_probe, 0.15, 5.0)
    assert len(children) >= 1 and children[0].poll() is not None


def test_isolated_worker_target_and_transport_are_bounded():
    with pytest.raises(ValueError, match="private module-level"):
        run_in_subprocess(lambda: None, 1.0)
    with pytest.raises(ValueError, match="isolated probe failure"):
        run_in_subprocess(_isolated_probe, 5.0, fail=True)
    with pytest.raises(IsolatedSolverError, match="output limit"):
        run_in_subprocess(_isolated_blob, 5.0, 4096, max_output_bytes=256)
    with pytest.raises(IsolatedSolverError, match="input limit"):
        run_in_subprocess(_isolated_blob, 5.0, b"x"*4096, max_input_bytes=256)
    with pytest.raises(ValueError, match="positive and finite"):
        run_in_subprocess(_isolated_probe, float("nan"))
    with pytest.raises(ValueError, match="max_output_bytes"):
        run_in_subprocess(_isolated_probe, 1.0, max_output_bytes=0)


def test_concurrent_isolated_jobs_do_not_share_solver_processes():
    with ThreadPoolExecutor(max_workers=2) as pool:
        pids = list(pool.map(lambda _: run_in_subprocess(_isolated_probe, 5.0, .05), range(2)))
    assert len(set(pids)) == 2 and os.getpid() not in pids


def test_exact_certificate_replay_never_invokes_candidate_subprocess(monkeypatch):
    from mathkernel import engines
    monkeypatch.setattr(engines, "run_in_subprocess", lambda *a, **k: (_ for _ in ()).throw(AssertionError("candidate invoked")))
    k = MathKernel()
    problem = create(k, "OptimizationProblem", variables=["x"], c=[1])
    result = k.apply(problem, "solve")
    assert result.ok and result.trust == TrustLevel.EXACT
    assert result.data["details"]["accepted"]


def test_numeric_lp_riccati_and_pole_search_report_hard_isolation():
    k = MathKernel()
    problem = create(k, "OptimizationProblem", variables=["x"], c=[-1], upper=[2])
    lp = k.apply(problem, "solve", {"mode": "numeric"})
    assert lp.ok
    lp_details = lp.data["details"].get("search_diagnostics", lp.data["details"])
    assert lp_details["process_isolation"] == "fresh_interpreter_hard_killable"

    system = create(k, "StateSpaceSystem", A=[[-1]], B=[[1]], C=[[1]], D=[[0]])
    lqr = k.apply(system, "lqr", {"Q": [[1]], "R": [[1]], "mode": "numeric"})
    assert lqr.ok and lqr.data["details"]["process_isolation"] == "fresh_interpreter_hard_killable"
    placement = k.apply(system, "place_poles", {"poles": [-2], "mode": "numeric"})
    assert placement.ok
    assert placement.data["details"]["process_isolation"] == "fresh_interpreter_hard_killable"


def test_native_timeout_is_structured_and_creates_no_derived_object(monkeypatch):
    from mathkernel import engines
    def timeout(*args, **kwargs):
        raise SolverTimeoutError("isolated process terminated")
    monkeypatch.setattr(engines, "run_in_subprocess", timeout)
    k = MathKernel()
    problem = create(k, "OptimizationProblem", variables=["x"], c=[-1], upper=[2])
    result = k.apply(problem, "solve", {"mode": "numeric"})
    assert not result.ok and "isolated process terminated" in result.errors[0]
    assert len(k.math_objects) == 1


def test_conic_and_qcqp_native_search_report_hard_isolation():
    k = MathKernel()
    conic = create(k, "ConicProblem", variables=["x"], c=[1], A=[[1]], b=[1],
                   cones=[{"kind": "nonnegative", "dimension": 1}])
    conic_result = k.apply(conic, "solve", {"mode": "numeric"})
    assert conic_result.ok
    assert conic_result.data["details"]["process_isolation"] == "fresh_interpreter_hard_killable"

    qcqp = create(k, "QuadraticallyConstrainedProblem", variables=["x"], c=[-1],
                  quadratics=[{"Q": [[2]], "a": [0], "r": -1}])
    qcqp_result = k.apply(qcqp, "solve", {"mode": "numeric", "initial": [0]})
    assert qcqp_result.ok
    assert qcqp_result.data["details"]["process_isolation"] == "fresh_interpreter_hard_killable"


@pytest.mark.parametrize("field,value", [
    ("max_engineering_work", 0), ("max_control_order", -1),
    ("solver_timeout_seconds", 0), ("tolerance", float("nan")),
])
def test_invalid_resource_configuration_fails_closed(field, value):
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_discrete_timing_units_and_type_survive_restart_and_conversion(tmp_path):
    settings = Settings(store_path=str(tmp_path/"audit.sqlite"))
    k = MathKernel(settings)
    source = create(k, "StateSpaceSystem", A=[["1/2"]], B=[[1]], C=[[1]], D=[[0]],
                    time_domain="discrete", sample_time="1/10", input_unit="V", output_unit="A")
    explicit = k.apply(source, "to_discrete_control")
    restarted = MathKernel(settings)
    stored = restarted.object_get(explicit.data["object_id"])
    assert stored.ok and stored.data["object_type"] == "DiscreteControlSystem"
    transfer = restarted.apply(explicit.data["object_id"], "to_transfer_function")
    model = restarted.math_objects[transfer.data["object_id"]]["value"]
    assert model.sample_time == sp.Rational(1, 10) and model.input_unit == "V" and model.output_unit == "A"


def test_capability_outputs_match_runtime_derived_types():
    k = MathKernel()
    frequency = k.capability_query(domain="control", input_type="TransferFunction", operation="frequency_response")
    assert frequency["count"] == 1 and "FrequencyResponse" in frequency["capabilities"][0]["output_types"]
    transfer = k.capability_query(domain="control", input_type="DiscreteControlSystem", operation="to_transfer_function")
    assert "TransferMatrix" in transfer["capabilities"][0]["output_types"]


def test_live_mcp_numeric_solver_keeps_isolation_metadata():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp
    async def run():
        async with Client(mcp) as client:
            source = await client.call_tool("math_object_create", {"object_type": "OptimizationProblem",
                "definition": {"variables": ["x"], "c": [-1], "upper": [2]}})
            return await client.call_tool("math_apply", {"object_id": source.data["data"]["object_id"],
                "operation": "solve", "parameters": {"mode": "numeric"}})
    result = asyncio.run(run()).data
    details = result["data"]["details"].get("search_diagnostics", result["data"]["details"])
    assert result["ok"] and details["process_isolation"] == "fresh_interpreter_hard_killable"
