"""M6.5.1 — demo-environment readiness check (read-only).

Hits the live backend endpoints and prints a compact readiness report.
Usage: ``python scripts/verify_demo_env.py [base_url]``
Base URL defaults to ``http://127.0.0.1:8000``.

This is a verification/diagnostic script; it does not modify model behavior,
datasets, or agent logic.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def request(path: str, method: str = "GET", body: Any = None) -> Any:
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        return e.code, (json.loads(raw) if raw else None)


def main() -> int:
    checks: list[tuple[str, str, Any]] = []

    status, health = request("/health")
    checks.append(("health", "ok" if health.get("status") == "ok" else "FAIL", health))

    status, features = request("/api/v1/features")
    fs = list(features.get("feature_sets", {}).keys())
    checks.append(
        ("features", "ok" if "full_operational" in fs and "telematics_only" in fs else "FAIL",
         {"feature_sets": fs, "field_count": len(features.get("fields", {}))})
    )

    status, sess = request("/api/v1/sessions", "POST", {"feature_set": "full_operational"})
    session_id = sess.get("session_id") if sess else None
    checks.append(("create_session", "ok" if session_id else "FAIL", sess))

    status, cat = request("/api/v1/replay/catalog")
    trucks = cat.get("trucks", []) if cat else []
    trip_count = sum(len(t.get("trips", [])) for t in trucks)
    first_truck = trucks[0].get("truck_id") if trucks else None
    checks.append(
        ("replay_catalog", "ok" if trucks else "FAIL",
         {"trucks": len(trucks), "trips": trip_count, "first_truck": first_truck})
    )

    status, trip = request("/api/v1/replay?trip_id=trip_000001&sample=20")
    pts = trip.get("points", []) if trip else []
    checks.append(
        ("replay_trip", "ok" if pts else "FAIL",
         {"trip_id": trip.get("trip_id"), "points": len(pts),
          "truck_id": trip.get("truck_id")})
    )

    # One deterministic message turn on the created session (offline/synthetic).
    status, resp = request(
        f"/api/v1/sessions/{session_id}/messages",
        "POST",
        {
            "query": "predict fuel rate and explain what drives it",
            "structured_inputs": {
                "vehicle_class": "40T", "gross_vehicle_weight_kg": 36000,
                "payload_kg": 12000, "vehicle_mass_kg": 24000,
                "speed_kph": 12.0, "acceleration_mps2": 0.2,
                "road_grade_pct": -0.06, "ignition_status": 1,
                "engine_rpm": 1041.9, "engine_load_pct": 23.42,
                "throttle_position_pct": 28.75, "torque_nm": 1600,
                "coolant_temp_c": 88, "intake_air_temp_c": 30,
                "ambient_temp_c": 22,
            },
        },
    )
    tr = resp.get("technical_report", {}) if resp else {}
    expl = tr.get("explanation", {}) if isinstance(tr.get("explanation"), dict) else {}
    fuel_rate = expl.get("predicted_fuel_rate_lph")
    checks.append(
        ("message_turn", "ok" if resp and resp.get("delivered") else "FAIL",
         {"status": status, "delivered": resp.get("delivered") if resp else None,
          "fuel_rate_lph": fuel_rate,
          "expected_value": expl.get("expected_value"),
          "model": expl.get("model_name")})
    )

    print("Demo-environment readiness report")
    print("=" * 60)
    ok = True
    for name, verdict, payload in checks:
        print(f"[{verdict}] {name}: {json.dumps(payload, default=str)}")
        if verdict != "ok":
            ok = False
    print("=" * 60)
    print(f"OVERALL: {'READY' if ok else 'NOT READY'} (session {session_id})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
