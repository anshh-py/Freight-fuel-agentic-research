"""M6.3 — tests for synthetic telemetry replay integration.

Scope (this file only covers the M6.3 replay surface added on top of the
M6.2 HTTP layer):

* /api/v1/replay/catalog returns trucks and their trips from the metadata CSVs
* /api/v1/replay returns chronological, in-range telemetry points drawn from
  the real synthetic dataset (bounded sampling, never the full 963,697 rows)
* replay points carry the fields displayed by the cockpit
  (speed, rpm, load, throttle, grade, fuel rate)
* an invalid/unknown trip returns 404
* the replay data matches the underlying source dataset exactly
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.features.spec import PROJECT_ROOT, load_feature_spec

client = TestClient(app)

EXPECTED_POINT_FIELDS = {
    "timestamp",
    "speed_kph",
    "engine_rpm",
    "engine_load_pct",
    "throttle_position_pct",
    "fuel_rate_lph",
    "road_grade_pct",
}


def test_replay_catalog_returns_trucks_and_trips():
    r = client.get("/api/v1/replay/catalog")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "synthetic_telemetry_replay"
    trucks = body["trucks"]
    assert len(trucks) >= 29
    # Each truck has its own trips and at least one trip.
    for t in trucks:
        assert t["truck_id"]
        assert t["vehicle_class"] in {"32T", "40T", "49T"}
        assert t["trips"], f"truck {t['truck_id']} has no trips"
        for trip in t["trips"]:
            assert trip["trip_id"]
            assert trip["route_type"]
            assert trip["record_count"] > 0


def test_replay_catalog_trip_ids_match_metadata():
    # The catalog is derived from trip_metadata.csv, so every truck's trips
    # must resolve through the replay endpoint.
    cat = client.get("/api/v1/replay/catalog").json()
    first = cat["trucks"][0]
    trip_id = first["trips"][0]["trip_id"]
    r = client.get("/api/v1/replay", params={"trip_id": trip_id, "sample": 10})
    assert r.status_code == 200
    assert r.json()["trip_id"] == trip_id


def test_replay_sampled_chronological_and_bounded():
    r = client.get("/api/v1/replay", params={"trip_id": "trip_000001", "sample": 30})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) <= 30
    assert body["record_count_total"] > 0
    assert body["record_count_sampled"] == len(body["points"])

    # Chronological ordering: timestamps are non-decreasing.
    stamps = [p["timestamp"] for p in body["points"]]
    assert stamps == sorted(stamps)
    assert stamps[0] < stamps[-1]


def test_replay_point_fields_present_and_in_range():
    r = client.get("/api/v1/replay", params={"trip_id": "trip_000001", "sample": 50})
    assert r.status_code == 200
    for p in r.json()["points"]:
        assert set(p.keys()) == EXPECTED_POINT_FIELDS
        assert 0 <= p["speed_kph"] <= 92
        assert 0 <= p["engine_rpm"] <= 2200
        assert 0 <= p["engine_load_pct"] <= 100
        assert 0 <= p["throttle_position_pct"] <= 100
        assert -6 <= p["road_grade_pct"] <= 6
        assert 0 <= p["fuel_rate_lph"] <= 220


def test_replay_matches_source_dataset():
    # Compare the backend's sampled point values against the source telemetry
    # master CSV (rounded to the same precision as the endpoint).
    spec = load_feature_spec()
    path = PROJECT_ROOT / spec["telemetry_path"]
    df = pd.read_csv(path)
    df = df[df["ignition_status"] == 1]
    trip = df[df["trip_id"] == "trip_000001"]

    r = client.get("/api/v1/replay", params={"trip_id": "trip_000001", "sample": 10})
    assert r.status_code == 200
    for p in r.json()["points"]:
        row = trip[trip["timestamp_utc"] == p["timestamp"]]
        assert len(row) == 1, f"timestamp {p['timestamp']} not found in source"
        src = row.iloc[0]
        assert abs(round(float(src["speed_kph"]), 2) - p["speed_kph"]) < 0.011
        assert abs(round(float(src["engine_rpm"]), 1) - p["engine_rpm"]) < 0.11
        assert abs(round(float(src["engine_load_pct"]), 2) - p["engine_load_pct"]) < 0.011


def test_replay_unknown_trip_404():
    r = client.get("/api/v1/replay", params={"trip_id": "trip_999999"})
    assert r.status_code == 404


def test_replay_negative_sample_safe():
    r = client.get("/api/v1/replay", params={"trip_id": "trip_000001", "sample": -5})
    assert r.status_code == 200
    assert len(r.json()["points"]) >= 2
