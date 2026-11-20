"""M6.2 — thin FastAPI HTTP layer over the M6.1 ``FuelRateFeedbackGateway``.

This module exposes the existing gateway through a small, conventional REST
surface. It is an adapter only: it does not re-implement session management,
routing, prediction, explanation, retrieval, or validation — all of that is
delegated to the frozen M6.1 gateway and the underlying M1–M5 pipeline.

Design rules enforced here (mirroring the gateway contract):

* requests are validated with pydantic (shape) then again by the gateway
  (semantic/feature-set validation),
* every successful message turn returns the serialized ``APIResponse`` (which
  always carries ``synthetic_only=True``) — never a raw internal object,
* boundary errors from the gateway are mapped to appropriate HTTP status codes
  and returned as JSON, never as stack traces or internal paths,
* sessions are isolated per ``session_id`` (registry-owned),
* a second orchestration system is never created.

All telemetry, prediction targets, and retrieval corpus are synthetic. No
endpoint here connects to a real truck, OBD hardware, or external service.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.gateway import FuelRateFeedbackGateway, InvalidInputError
from src.api.schemas import APIResponse

# ---------------------------------------------------------------------------
# Request/response models (shape validation only).
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    feature_set: str = "full_operational"


class MessageRequest(BaseModel):
    query: str
    structured_inputs: Optional[dict[str, Any]] = None


class ReplayRequest(BaseModel):
    trip_id: Optional[str] = None
    sample: int = 120


# ---------------------------------------------------------------------------
# Gateway singleton (lazy so unit tests of other modules stay fast).
# ---------------------------------------------------------------------------

_gateway: Optional[FuelRateFeedbackGateway] = None
_gateway_lock = threading.Lock()


def get_gateway() -> FuelRateFeedbackGateway:
    """Return the process-wide gateway, constructing it on first use."""
    global _gateway
    with _gateway_lock:
        if _gateway is None:
            _gateway = FuelRateFeedbackGateway()
        return _gateway


# ---------------------------------------------------------------------------
# Feature / replay helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def feature_catalog() -> dict[str, Any]:
    """Return the canonical feature-set + field metadata for the UI."""
    from src.features.spec import load_feature_spec
    from src.tools.validation import FIELD_RANGES, VEHICLE_CLASSES, IGNITION_ALLOWED

    spec = load_feature_spec()
    sets = spec.get("feature_sets", {})

    labels = {
        "vehicle_class": "Vehicle class",
        "gross_vehicle_weight_kg": "Gross vehicle weight",
        "payload_kg": "Payload",
        "vehicle_mass_kg": "Vehicle mass",
        "speed_kph": "Speed",
        "acceleration_mps2": "Acceleration",
        "road_grade_pct": "Road grade",
        "ignition_status": "Ignition",
        "engine_rpm": "Engine RPM",
        "engine_load_pct": "Engine load",
        "throttle_position_pct": "Throttle position",
        "torque_nm": "Torque",
        "coolant_temp_c": "Coolant temperature",
        "intake_air_temp_c": "Intake air temperature",
        "ambient_temp_c": "Ambient temperature",
    }
    units = {
        "gross_vehicle_weight_kg": "kg",
        "payload_kg": "kg",
        "vehicle_mass_kg": "kg",
        "speed_kph": "km/h",
        "acceleration_mps2": "m/s²",
        "road_grade_pct": "%",
        "engine_rpm": "rpm",
        "engine_load_pct": "%",
        "throttle_position_pct": "%",
        "torque_nm": "N·m",
        "coolant_temp_c": "°C",
        "intake_air_temp_c": "°C",
        "ambient_temp_c": "°C",
    }
    steps = {
        "speed_kph": 1.0,
        "engine_rpm": 50.0,
        "engine_load_pct": 1.0,
        "throttle_position_pct": 1.0,
        "road_grade_pct": 0.1,
        "acceleration_mps2": 0.05,
        "gross_vehicle_weight_kg": 100.0,
        "payload_kg": 100.0,
        "vehicle_mass_kg": 100.0,
        "torque_nm": 10.0,
        "coolant_temp_c": 1.0,
        "intake_air_temp_c": 1.0,
        "ambient_temp_c": 1.0,
    }

    fields: dict[str, dict[str, Any]] = {}
    for name in spec.get("feature_sets", {}).get("full_operational", {}).get("features", []):
        meta: dict[str, Any] = {
            "name": name,
            "label": labels.get(name, name),
            "unit": units.get(name, ""),
        }
        if name == "vehicle_class":
            meta.update({"kind": "categorical", "options": list(VEHICLE_CLASSES)})
        elif name == "ignition_status":
            meta.update({"kind": "binary", "options": list(IGNITION_ALLOWED)})
        else:
            lo, hi = FIELD_RANGES[name]
            meta.update({"kind": "numeric", "min": lo, "max": hi, "step": steps.get(name, 1.0)})
        fields[name] = meta

    return {
        "feature_sets": {k: v.get("features", []) for k, v in sets.items()},
        "descriptions": {k: v.get("description", "") for k, v in sets.items()},
        "fields": fields,
        "synthetic_only": True,
    }


_replay_catalog: Optional[dict[str, Any]] = None
_replay_catalog_lock = threading.Lock()


def get_replay_catalog() -> dict[str, Any]:
    """Return the truck/trip catalog for replay selection (synthetic only)."""
    global _replay_catalog
    with _replay_catalog_lock:
        if _replay_catalog is None:
            import pandas as pd

            from src.features.spec import PROJECT_ROOT, load_feature_spec

            spec = load_feature_spec()
            base = (PROJECT_ROOT / spec.get("telemetry_path")).resolve().parent
            trucks_df = pd.read_csv(base / "truck_metadata.csv")
            trips_df = pd.read_csv(base / "trip_metadata.csv")

            trucks: list[dict[str, Any]] = []
            for _, t in trucks_df.iterrows():
                trip_rows = trips_df[trips_df["truck_id"] == t["truck_id"]]
                trips: list[dict[str, Any]] = []
                for _, r in trip_rows.iterrows():
                    trips.append(
                        {
                            "trip_id": str(r["trip_id"]),
                            "route_type": str(r["route_type"]),
                            "record_count": int(r["record_count"]),
                            "duration_seconds": int(r["duration_seconds"]),
                            "distance_km": round(float(r["distance_km"]), 4),
                            "start_timestamp_utc": str(r["start_timestamp_utc"]),
                            "end_timestamp_utc": str(r["end_timestamp_utc"]),
                        }
                    )
                trucks.append(
                    {
                        "truck_id": str(t["truck_id"]),
                        "vehicle_class": str(t["vehicle_class"]),
                        "gross_vehicle_weight_kg": int(t["gross_vehicle_weight_kg"]),
                        "trips": trips,
                    }
                )

            _replay_catalog = {
                "source": "synthetic_telemetry_replay",
                "trucks": trucks,
            }
        return _replay_catalog


def _load_replay_index() -> dict[str, list[dict[str, Any]]]:
    """Load telemetry once and index rows by trip_id (synthetic replay only)."""
    import pandas as pd

    from src.features.spec import PROJECT_ROOT, load_feature_spec

    spec = load_feature_spec()
    path = spec.get("telemetry_path")
    full = (PROJECT_ROOT / path).resolve()

    df = pd.read_csv(full)
    df = df[df["ignition_status"] == 1]  # only driving/rolling records
    index: dict[str, list[dict[str, Any]]] = {}
    for trip_id, group in df.groupby("trip_id", sort=False):
        index[str(trip_id)] = group.to_dict(orient="records")
    return index


_replay_index: Optional[dict[str, list[dict[str, Any]]]] = None
_replay_lock = threading.Lock()


def get_replay_trip(trip_id: str, sample: int) -> dict[str, Any]:
    global _replay_index
    with _replay_lock:
        if _replay_index is None:
            _replay_index = _load_replay_index()

    if trip_id not in _replay_index:
        raise HTTPException(status_code=404, detail=f"unknown replay trip: {trip_id}")

    rows = _replay_index[trip_id]
    n = max(2, min(int(sample), len(rows)))
    step = max(1, len(rows) // n)
    chosen = rows[::step][:n]

    points = []
    for r in chosen:
        points.append(
            {
                "timestamp": r.get("timestamp_utc"),
                "speed_kph": round(float(r.get("speed_kph") or 0.0), 2),
                "engine_rpm": round(float(r.get("engine_rpm") or 0.0), 1),
                "engine_load_pct": round(float(r.get("engine_load_pct") or 0.0), 2),
                "throttle_position_pct": round(float(r.get("throttle_position_pct") or 0.0), 2),
                "fuel_rate_lph": round(float(r.get("fuel_rate_lph") or 0.0), 3),
                "road_grade_pct": round(float(r.get("road_grade_pct") or 0.0), 2),
            }
        )

    first = chosen[0]
    return {
        "trip_id": trip_id,
        "truck_id": first.get("truck_id"),
        "vehicle_class": first.get("vehicle_class"),
        "source": "synthetic_telemetry_replay",
        "record_count_total": len(rows),
        "record_count_sampled": len(points),
        "points": points,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Truck Fuel Agentic Cockpit API",
    version="6.2.0",
    description=(
        "M6.2 HTTP front-end over the M6.1 gateway. All telemetry, prediction "
        "targets, and retrieval corpus are synthetic."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    gw = get_gateway()
    return {
        "status": "ok",
        "service": "truck-fuel-gateway",
        "milestone": "M6.2",
        "synthetic_only": True,
        "active_sessions": gw.registry.active_count(),
    }


@app.get("/api/v1/features")
def features() -> dict[str, Any]:
    return feature_catalog()


@app.post("/api/v1/sessions", status_code=201)
def create_session(req: CreateSessionRequest) -> dict[str, Any]:
    gw = get_gateway()
    try:
        session_id = gw.start_session(req.feature_set)
    except InvalidInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": session_id, "feature_set": req.feature_set}


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    gw = get_gateway()
    try:
        return gw.session_state(session_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="unknown or expired session") from exc


@app.delete("/api/v1/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    gw = get_gateway()
    if not gw.registry.is_live(session_id):
        raise HTTPException(status_code=404, detail="unknown or expired session")
    with gw.registry._guard:
        gw.registry._entries.pop(session_id, None)


_STATUS_FOR_CODE = {
    "session_not_found": 404,
    "session_expired": 404,
    "invalid_input": 400,
    "gateway_error": 500,
    "internal_error": 500,
}


@app.post("/api/v1/sessions/{session_id}/messages")
def send_message(session_id: str, req: MessageRequest) -> dict[str, Any]:
    gw = get_gateway()
    resp: APIResponse = gw.send(session_id, req.query, req.structured_inputs)
    payload = resp.to_dict()
    if resp.error is not None:
        code = resp.error.get("code", "internal_error")
        status = _STATUS_FOR_CODE.get(code, 500)
        raise HTTPException(status_code=status, detail=resp.error.get("message"))
    return payload


@app.get("/api/v1/replay/catalog")
def replay_catalog() -> dict[str, Any]:
    return get_replay_catalog()


@app.get("/api/v1/replay")
def replay(trip_id: str = "trip_000001", sample: int = 120) -> dict[str, Any]:
    return get_replay_trip(trip_id, sample)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
