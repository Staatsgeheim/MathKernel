import asyncio
import hashlib
import json
import sqlite3

import pytest

from mathkernel import MathKernel, Settings
from mathkernel.statistics_adapter import DERIVED_OUTPUTS, OPERATIONS, TYPES
from mathkernel.store import KernelStore


def _create(kernel, object_type, definition):
    result = kernel.object_create(object_type, definition)
    assert result.ok, result.errors
    return result.data["object_id"]


def _sde(kernel, drift="0"):
    return _create(kernel, "sde", {
        "state_variables": ["x"], "drift": [drift],
        "diffusion": [[1]], "initial_state": [0],
        "start_time": 0, "end_time": 1,
    })


def _rewrite_payload(path, object_id, transform, *, refresh_integrity=False):
    connection = sqlite3.connect(path)
    payload = connection.execute(
        "SELECT payload FROM math_objects WHERE object_id = ?", (object_id,)
    ).fetchone()[0]
    data = json.loads(payload)
    transform(data)
    changed = json.dumps(data, sort_keys=True, separators=(",", ":"))
    if refresh_integrity:
        integrity = hashlib.sha256(changed.encode("utf-8")).hexdigest()
        connection.execute(
            "UPDATE math_objects SET payload = ?, integrity = ? WHERE object_id = ?",
            (changed, integrity, object_id),
        )
    else:
        connection.execute(
            "UPDATE math_objects SET payload = ? WHERE object_id = ?",
            (changed, object_id),
        )
    connection.commit()
    connection.close()


def test_store_integrity_rejects_tampered_persisted_result(tmp_path):
    path = tmp_path / "tampered.sqlite"
    kernel = MathKernel(Settings(store_path=str(path)))
    sde_id = _sde(kernel)
    made = kernel.apply(sde_id, "simulate", {
        "steps": 3, "paths": 2, "seed": 7,
    })
    simulation_id = made.data["object_id"]
    kernel._store.close()
    _rewrite_payload(
        path, simulation_id,
        lambda payload: payload.__setitem__("semantic_status", "verified"),
    )

    reopened = MathKernel(Settings(store_path=str(path)))
    before = len(reopened.math_objects)
    fetched = reopened.object_get(simulation_id)
    replay = reopened.apply(simulation_id, "verify")
    assert not fetched.ok and "integrity verification" in fetched.errors[0]
    assert not replay.ok and "integrity verification" in replay.errors[0]
    assert len(reopened.math_objects) == before


def test_store_integrity_migrates_legacy_rows_once_and_rejects_null_marker(tmp_path):
    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE expressions (expr_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.execute(
        "INSERT INTO expressions VALUES (?, ?)", ("expr_old", '{"value":1}'))
    connection.commit(); connection.close()

    store = KernelStore(str(path))
    assert store.get_expression("expr_old") == {"value": 1}
    store.close()
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE expressions SET payload = ?, integrity = NULL WHERE expr_id = ?",
        ('{"value":2}', "expr_old"))
    connection.commit(); connection.close()
    reopened = KernelStore(str(path))
    with pytest.raises(ValueError, match="integrity verification"):
        reopened.get_expression("expr_old")


def test_record_envelope_rejects_relabeling_even_with_fresh_checksum(tmp_path):
    path = tmp_path / "relabel.sqlite"
    kernel = MathKernel(Settings(store_path=str(path)))
    sde_id = _sde(kernel)
    kernel._store.close()
    _rewrite_payload(
        path, sde_id,
        lambda payload: payload.__setitem__("object_type", "WienerProcess"),
        refresh_integrity=True,
    )
    fetched = MathKernel(Settings(store_path=str(path))).object_get(sde_id)
    assert not fetched.ok
    assert "type does not match" in fetched.errors[0]


def test_cross_type_source_substitution_fails_before_output():
    kernel = MathKernel()
    sample_id = _create(kernel, "StatisticalSample", {
        "variables": ["y", "x"], "observations": [[1, 0], [2, 1], [3, 2]],
    })
    model_id = _create(kernel, "glm", {
        "sample_id": sample_id, "response": "y", "predictors": ["x"],
        "family": "gaussian", "link": "identity",
    })
    wrong_id = _sde(kernel)
    record = kernel.math_objects[model_id]
    record["value"] = record["value"].model_copy(update={"sample_id": wrong_id})
    record["sources"] = [wrong_id]
    before = len(kernel.math_objects)
    result = kernel.apply(model_id, "fit")
    assert not result.ok and "GLM source sample" in result.errors[0]
    assert len(kernel.math_objects) == before


def test_internal_source_and_envelope_ancestry_must_agree():
    kernel = MathKernel(); first = _sde(kernel); second = _sde(kernel, drift="x")
    made = kernel.apply(first, "simulate", {"steps": 2, "paths": 1, "seed": 1})
    simulation_id = made.data["object_id"]
    kernel.math_objects[simulation_id]["sources"] = [second]
    result = kernel.apply(simulation_id, "verify")
    assert not result.ok and "source ancestry" in result.errors[0]


def test_tampered_derived_numeric_result_is_refuted_by_replay():
    kernel = MathKernel(); sde_id = _sde(kernel)
    made = kernel.apply(sde_id, "simulate", {"steps": 2, "paths": 1, "seed": 3})
    simulation_id = made.data["object_id"]
    record = kernel.math_objects[simulation_id]
    values = [[list(state) for state in path] for path in record["value"].values]
    values[0][-1][0] += 1.0
    record["value"] = record["value"].model_copy(update={
        "values": tuple(tuple(tuple(state) for state in path) for path in values)
    })
    replay = kernel.apply(simulation_id, "verify")
    assert replay.status == "refuted"
    assert replay.data["verification"]["paths_recomputed"] is False


def test_sde_object_get_cannot_bypass_paginated_output_limit():
    kernel = MathKernel(Settings(max_sde_query_values=1))
    sde_id = _sde(kernel)
    made = kernel.apply(sde_id, "simulate", {"steps": 8, "paths": 8, "seed": 1})
    fetched = kernel.object_get(made.data["object_id"])
    payload = fetched.data["object"]
    assert fetched.ok and "values" not in payload and "times" not in payload
    assert payload["stored_fields"] == ["times", "values"]
    assert payload["query_operations"] == ["path", "terminal_values"]


def test_statistics_capability_contract_is_unique_complete_and_executable():
    kernel = MathKernel()
    manifest = kernel.capability_query(domain="statistics")
    capabilities = manifest["capabilities"]
    names = [item["name"] for item in capabilities]
    assert manifest["count"] == len(capabilities) == 63
    assert len(names) == len(set(names))
    for item in capabilities:
        object_type = item["input_types"][0]
        assert item["operation"] in OPERATIONS[object_type]
        assert item["handler"] == "module:engineering"
        assert item["verification_methods"]
        assert item["trust_levels"]
    output_only = set(DERIVED_OUTPUTS.values()) | {
        "CoxPHFit", "TimeSeriesFit",
    }
    assert output_only.isdisjoint(set(TYPES.values()))


def test_mcp_reports_bounded_simulation_metadata():
    from fastmcp import Client
    from mathkernel_mcp.server import mcp

    async def scenario():
        async with Client(mcp) as client:
            model = await client.call_tool("math_object_create", {
                "object_type": "sde", "definition": {
                    "state_variables": ["x"], "drift": [0],
                    "diffusion": [[1]], "initial_state": [0],
                    "start_time": 0, "end_time": 1,
                }})
            made = await client.call_tool("math_apply", {
                "object_id": model.data["data"]["object_id"],
                "operation": "simulate",
                "parameters": {"steps": 2, "paths": 2, "seed": 5},
            })
            fetched = await client.call_tool("math_object_get", {
                "object_id": made.data["data"]["object_id"],
            })
            return fetched.data

    result = asyncio.run(scenario())
    assert result["ok"]
    assert "values" not in result["data"]["object"]


