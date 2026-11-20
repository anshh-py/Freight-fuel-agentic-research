"""M6.2 — tests for the thin FastAPI HTTP layer (``src.api.server``).

Scope (this file only covers the new HTTP surface; the gateway contract itself
is already covered by ``test_integration_gateway.py``):

* /health
* feature catalog endpoint
* session create / get / delete lifecycle
* feature-set validation on session creation (400 on unknown set)
* invalid message shape (422 via pydantic)
* a genuine end-to-end prediction request through the real gateway
* synthetic replay endpoint

The real-model tests load the saved M1.2 artifacts once; they never retrain.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app, get_gateway

client = TestClient(app)

_FULL_INPUTS = {
    "vehicle_class": "40T",
    "gross_vehicle_weight_kg": 36000.0,
    "payload_kg": 12000.0,
    "vehicle_mass_kg": 24000.0,
    "speed_kph": 80.0,
    "acceleration_mps2": 0.2,
    "road_grade_pct": 1.5,
    "ignition_status": 1,
    "engine_rpm": 1400.0,
    "engine_load_pct": 55.0,
    "throttle_position_pct": 40.0,
    "torque_nm": 1600.0,
    "coolant_temp_c": 88.0,
    "intake_air_temp_c": 30.0,
    "ambient_temp_c": 22.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["synthetic_only"] is True


def test_feature_catalog():
    r = client.get("/api/v1/features")
    assert r.status_code == 200
    body = r.json()
    assert "full_operational" in body["feature_sets"]
    assert "telematics_only" in body["feature_sets"]
    assert body["fields"]["speed_kph"]["kind"] == "numeric"
    assert body["fields"]["vehicle_class"]["options"] == ["32T", "40T", "49T"]


def test_session_create_get_delete():
    r = client.post("/api/v1/sessions", json={"feature_set": "full_operational"})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert sid

    g = client.get(f"/api/v1/sessions/{sid}")
    assert g.status_code == 200
    assert g.json()["feature_set"] == "full_operational"

    d = client.delete(f"/api/v1/sessions/{sid}")
    assert d.status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404


def test_session_bad_feature_set_400():
    r = client.post("/api/v1/sessions", json={"feature_set": "bogus"})
    assert r.status_code == 400


def test_message_invalid_shape_422():
    # Missing required "query" field triggers pydantic 422.
    r = client.post("/api/v1/sessions/whatever/messages", json={"structured_inputs": {}})
    assert r.status_code == 422


def test_message_unknown_session_404():
    r = client.post(
        "/api/v1/sessions/does-not-exist/messages",
        json={"query": "fuel consumption"},
    )
    assert r.status_code == 404


def test_real_prediction_end_to_end():
    # Session + a real model prediction through the HTTP layer.
    sid = client.post("/api/v1/sessions", json={"feature_set": "full_operational"}).json()["session_id"]
    r = client.post(
        f"/api/v1/sessions/{sid}/messages",
        json={"query": "fuel consumption", "structured_inputs": _FULL_INPUTS},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["error"] is None
    assert body["delivered"] is True, body["driver_feedback"]
    assert "Estimated fuel rate" in body["driver_feedback"]
    assert body["technical_report"]["synthetic_only"] is True
    assert body["technical_report"]["prediction"]["predicted_fuel_rate_lph"] > 0
    # The prediction value must be preserved in the validated driver text.
    val = body["technical_report"]["prediction"]["predicted_fuel_rate_lph"]
    assert str(val) in body["driver_feedback"]


def test_replay_samples_synthetic_telemetry():
    r = client.get("/api/v1/replay", params={"trip_id": "trip_000001", "sample": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "synthetic_telemetry_replay"
    assert body["trip_id"] == "trip_000001"
    assert len(body["points"]) <= 50
    assert "fuel_rate_lph" in body["points"][0]
    assert "engine_rpm" in body["points"][0]
