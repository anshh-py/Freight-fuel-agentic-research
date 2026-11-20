"""Generate the complete synthetic heavy-duty truck research dataset suite.

Outputs (into ``data/synthetic_heavy_truck_research/``):
    * telemetry_master_963697.csv.gz   (exactly 963,697 rows)
    * truck_metadata.csv
    * trip_metadata.csv
    * truckwise_split_manifest.csv
    * telemetry_replay_manifest.csv
    * agent_evaluation_queries.csv
    * rag_knowledge_corpus.jsonl
    * dataset_card.json
    * data_dictionary.csv
    * generation_config.json
    * README.md

Usage:
    python gen_telemetry.py [--seed 20240617] [--out DIR]

Everything is synthetic. This script makes no claim that any record represents
real measured truck telemetry. All randomness is seeded so the suite is fully
reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_config import (
    AGENT_CATEGORIES,
    AIR_DENSITY_KG_M3,
    CLASS_SPECS,
    DIESEL_ENERGY_J_PER_L,
    DRIVETRAIN_EFFICIENCY,
    EXCLUDED_FEATURES,
    FLEET_CLASSES,
    GRAVITY_MPS2,
    NUM_TRUCKS,
    OUTPUT_DIR,
    POWER_FEASIBILITY_TOLERANCE,
    RECOMMENDED_FEATURES,
    ROUTE_NAMES,
    ROUTE_PROBABILITIES,
    ROUTE_SPECS,
    SAMPLING_INTERVAL_S,
    SEED,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TARGET_COLUMN,
    TELEMETRY_COLUMNS,
    TOTAL_ROWS,
)
from synthetic_model import TripGenerator

# ---------------------------------------------------------------------------
# Truck fleet construction
# ---------------------------------------------------------------------------

def build_trucks(rng: np.random.Generator, out_dir: Path) -> pd.DataFrame:
    """Create 29 synthetic trucks. Static truck attributes are fixed here and
    reused identically across all of a truck's records (consistency)."""
    records = []
    spec = CLASS_SPECS
    for i in range(NUM_TRUCKS):
        cls = FLEET_CLASSES[i]
        s = spec[cls]
        gvw = s["gvw_kg"]
        curb_frac = float(rng.uniform(0.54, 0.63))
        curb = round(gvw * curb_frac)
        truck_id = f"truck_{i + 1:03d}"
        age = round(float(rng.uniform(0.5, 12.0)), 1)
        efficiency = round(float(rng.uniform(0.85, 1.08)), 4)
        # Deterministic per-truck powertrain variation within a class, so that
        # displacement and rated power differ between trucks while staying
        # physically consistent with the class GVW (bigger class -> bigger
        # engine, and a truck's displacement correlates with its rated power).
        disp = round(s["displacement_l"] * float(rng.uniform(0.92, 1.08)), 1)
        rated_power = int(round(s["rated_power_kw"] * float(rng.uniform(0.95, 1.05))))
        records.append({
            "truck_id": truck_id,
            "vehicle_class": cls,
            "gross_vehicle_weight_kg": gvw,
            "curb_mass_kg": curb,
            "engine_displacement_l": disp,
            "rated_power_kw": rated_power,
            "vehicle_age_years": age,
            "synthetic_efficiency_factor": efficiency,
            "cda_m2": round(s["cda_m2"] * float(rng.uniform(0.96, 1.04)), 3),
            "crr": round(s["crr"] * float(rng.uniform(0.95, 1.05)), 5),
        })
    df = pd.DataFrame(records)
    # The public metadata file exposes the documented columns.
    public_cols = [
        "truck_id", "vehicle_class", "gross_vehicle_weight_kg",
        "engine_displacement_l", "rated_power_kw", "vehicle_age_years",
        "synthetic_efficiency_factor",
    ]
    df[public_cols].to_csv(out_dir / "truck_metadata.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Record / trip allocation
# ---------------------------------------------------------------------------

def allocate_truck_targets(rng: np.random.Generator, trucks: pd.DataFrame) -> np.ndarray:
    """Allocate the exact TOTAL_ROWS across the 29 trucks."""
    mult = {"32T": 1.00, "40T": 1.08, "49T": 1.16}
    raw = np.array([
        float(rng.uniform(0.85, 1.15)) * mult[c] for c in trucks["vehicle_class"]
    ])
    frac = raw / raw.sum()
    tgt = np.floor(TOTAL_ROWS * frac).astype(np.int64)
    tgt[-1] += TOTAL_ROWS - int(tgt.sum())
    assert tgt.sum() == TOTAL_ROWS
    return tgt


def allocate_trips_for_truck(rng: np.random.Generator, target_recs: int
                             ) -> list[int]:
    """Split a truck's record budget into trip durations (sum exact)."""
    if target_recs < 80:
        return [target_recs]
    k = int(rng.integers(12, 27))
    avg = target_recs / k
    raw = rng.lognormal(math.log(max(avg, 120.0)), 0.45, size=k)
    raw = np.maximum(raw, 20.0)
    scaled = raw / raw.sum() * target_recs
    dur = np.round(scaled).astype(np.int64)
    diff = target_recs - int(dur.sum())
    if diff > 0:
        dur[int(np.argmin(dur))] += diff
    elif diff < 0:
        dur[int(np.argmax(dur))] += diff
    dur = np.maximum(dur, 1)
    # re-fix rounding drift caused by the max clamp
    diff2 = target_recs - int(dur.sum())
    if diff2 != 0:
        dur[-1] += diff2
    assert int(dur.sum()) == target_recs
    return dur.tolist()


def choose_payload(rng: np.random.Generator, truck: pd.Series) -> int:
    max_payload = truck["gross_vehicle_weight_kg"] - truck["curb_mass_kg"] - 600
    if max_payload < 100:
        max_payload = 100
    low = max(100, int(0.15 * max_payload))
    return int(rng.integers(low, max(int(0.9 * max_payload), low + 1)))


# ---------------------------------------------------------------------------
# Telemetry + trip metadata generation
# ---------------------------------------------------------------------------

def generate_telemetry(trucks: pd.DataFrame, out_dir: Path, rng: np.random.Generator):
    targets = allocate_truck_targets(rng, trucks)
    rid_cursor = [0]
    trip_gen = TripGenerator(rng, rid_cursor)

    trip_rows = []
    telemetry_path = out_dir / "telemetry_master_963697.csv.gz"
    truck_start_base = pd.Timestamp("2023-01-02T00:00:00Z")

    total_written = 0
    trip_counter = 0

    # Deterministic gzip: mtime=0 and no embedded filename so the compressed
    # bytes are byte-for-byte reproducible from the same seed.
    with open(telemetry_path, "wb") as _raw, \
         gzip.GzipFile(filename="", mode="wb", fileobj=_raw, mtime=0) as _gz, \
         io.TextIOWrapper(_gz, encoding="utf-8") as fh:
        writer = fh
        writer.write(",".join(TELEMETRY_COLUMNS) + "\n")

        for truck_idx in range(NUM_TRUCKS):
            truck = trucks.iloc[truck_idx]
            truck_id = truck["truck_id"]
            # truck-specific timeline start (days offset + hour of day)
            start = truck_start_base + pd.Timedelta(
                days=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 24 * 3600)),
            )
            durations = allocate_trips_for_truck(rng, int(targets[truck_idx]))
            truck_frames = []

            for d in durations:
                trip_counter += 1
                trip_id = f"trip_{trip_counter:06d}"
                route = str(rng.choice(ROUTE_NAMES, p=ROUTE_PROBABILITIES))
                payload = choose_payload(rng, truck)
                truck_dict = {
                    "truck_id": truck_id,
                    "vehicle_class": truck["vehicle_class"],
                    "gross_vehicle_weight_kg": int(truck["gross_vehicle_weight_kg"]),
                    "curb_mass_kg": int(truck["curb_mass_kg"]),
                    "payload_kg": payload,
                    "vehicle_mass_kg": int(truck["curb_mass_kg"]) + payload,
                    "cda_m2": float(truck["cda_m2"]),
                    "crr": float(truck["crr"]),
                    "rated_power_kw": float(truck["rated_power_kw"]),
                    "rated_rpm": 1900,
                    "idle_rpm": CLASS_SPECS[truck["vehicle_class"]]["idle_rpm"],
                    "displacement_l": float(truck["engine_displacement_l"]),
                    "synthetic_efficiency_factor": float(truck["synthetic_efficiency_factor"]),
                }

                df, agg = trip_gen.generate(
                    truck_dict, trip_id, start, route, int(d),
                )
                truck_frames.append(df)
                trip_rows.append(agg)
                start = start + pd.Timedelta(
                    seconds=int(d) + int(rng.integers(1800, 28800)),
                )

            # flush this truck's chunk to the gzip stream
            chunk = pd.concat(truck_frames, ignore_index=True)
            chunk.to_csv(fh, index=False, header=False)
            total_written += len(chunk)
            truck_frames.clear()

    trip_df = pd.DataFrame(trip_rows)
    trip_df.to_csv(out_dir / "trip_metadata.csv", index=False)

    # quick internal sanity before returning
    assert total_written == TOTAL_ROWS, (total_written, TOTAL_ROWS)
    assert trip_df["record_count"].sum() == TOTAL_ROWS
    assert trip_df["trip_id"].nunique() == len(trip_df)
    assert trip_df.groupby("trip_id")["truck_id"].nunique().max() == 1
    return trip_df, total_written


# ---------------------------------------------------------------------------
# Truck-wise split manifest (leakage-safe)
# ---------------------------------------------------------------------------

def make_splits(trucks: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign trucks to train/val/test with zero overlap, stratified by class.

    All trucks appear in exactly one split; no individual rows are split. The
    ratio is 20 train / 4 val / 5 test. Within each class the truck order is
    shuffled, then each truck is greedily assigned to the split that is most
    below its target share, so every class is represented in every split.
    """
    split_names = ["train", "val", "test"]
    target_share = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}
    counts = {"train": 0, "val": 0, "test": 0}

    rows = []
    classes = list(dict.fromkeys(trucks["vehicle_class"]))
    for cls in classes:
        members = list(trucks.index[trucks["vehicle_class"] == cls])
        rng.shuffle(members)
        for idx in members:
            best = min(split_names, key=lambda s: counts[s] / target_share[s])
            counts[best] += 1
            rows.append({
                "truck_id": trucks.at[idx, "truck_id"],
                "vehicle_class": trucks.at[idx, "vehicle_class"],
                "split": best,
            })

    manifest = pd.DataFrame(rows)
    assert manifest["split"].value_counts().to_dict()["train"] == SPLIT_TRAIN
    assert manifest["split"].value_counts().to_dict()["val"] == SPLIT_VAL
    assert manifest["split"].value_counts().to_dict()["test"] == SPLIT_TEST
    assert manifest["truck_id"].nunique() == len(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Replay manifest
# ---------------------------------------------------------------------------

def make_replay_manifest(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, tr in trips.iterrows():
        multiplier = 2.0 if tr["route_type"] == "highway" and tr["duration_seconds"] > 2400 else 1.0
        mode = "fast_forward" if multiplier > 1.0 else "sequential_timestamp"
        rows.append({
            "trip_id": tr["trip_id"],
            "truck_id": tr["truck_id"],
            "start_timestamp_utc": tr["start_timestamp_utc"],
            "end_timestamp_utc": tr["end_timestamp_utc"],
            "record_count": int(tr["record_count"]),
            "replay_speed_multiplier": multiplier,
            "replay_mode": mode,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agent evaluation queries (grounded in real trip/truck references)
# ---------------------------------------------------------------------------

# Each template is a (query, expected_primary_tool, required_evidence_or_fields,
# expected_behaviour, difficulty) tuple with {} placeholders filled from actual
# generated trips/trucks so the cases are grounded, not merely fluent.
AGENT_TEMPLATES = {
    "current_telemetry_retrieval": [
        ("What is the current speed, engine load and fuel rate for the {truck_id}?",
         "telemetry_retrieval", "speed_kph, engine_load_pct, fuel_rate_lph",
         "Return the latest telemetry row with the requested fields and timestamps.", "easy"),
        ("Show me the last 5 minutes of coolant and intake temperatures for {truck_id} in trip {trip_id}.",
         "telemetry_retrieval", "coolant_temp_c, intake_air_temp_c, timestamp_utc",
         "Return time-ordered rows within the requested window.", "medium"),
        ("Was the engine of {truck_id} running at the end of trip {trip_id}?",
         "telemetry_retrieval", "ignition_status, timestamp_utc",
         "Report the engine-off status from the final telemetry row.", "easy"),
    ],
    "fuel_rate_prediction": [
        ("Predict the fuel rate for {truck_id} at its current operating point.",
         "fuel_rate_model", "speed_kph, engine_load_pct, engine_rpm, torque_nm, payload_kg, road_grade_pct",
         "Return a scalar fuel_rate_lph prediction plus model confidence and input snapshot.", "medium"),
        ("Estimate fuel rate for {truck_id} on a 4% uphill at full payload and 40 kph.",
         "fuel_rate_model", "road_grade_pct, payload_kg, speed_kph",
         "Return a forecast with the assumed inputs stated explicitly.", "hard"),
    ],
    "fuel_consumption_analysis": [
        ("What is the total fuel consumed during trip {trip_id}?",
         "fuel_analysis", "fuel_consumed_l_increment, cumulative_fuel_l_trip",
         "Sum increments within the trip (or read the final cumulative value) and report litres.", "easy"),
        ("Compare the average fuel rate of {truck_id} across its highway trips.",
         "fuel_analysis", "fuel_rate_lph, route_type",
         "Aggregate per-route mean fuel rate and return a comparison.", "medium"),
    ],
    "trip_summarization": [
        ("Summarize trip {trip_id}: distance, duration, average speed, fuel used and payload.",
         "trip_summary", "distance_km, duration_seconds, speed_kph, fuel_consumed_l_increment, payload_kg",
         "Return a concise summary with aggregate values from telemetry.", "medium"),
        ("How many stops (speed-zero periods) did trip {trip_id} have?",
         "trip_summary", "speed_kph, ignition_status",
         "Count distinct stationary episodes while ignition is on.", "hard"),
    ],
    "cross_truck_comparison": [
        ("How does the average fuel rate of {truck_id} compare to the fleet 40T average?",
         "fleet_comparison", "fuel_rate_lph, vehicle_class",
         "Compare to fleet baseline and return a clear difference.", "medium"),
        ("Which truck in the fleet is most fuel-efficient on highway routes?",
         "fleet_comparison", "fuel_rate_lph, route_type, truck_id",
         "Rank trucks by mean highway fuel rate and return the leader.", "medium"),
    ],
    "shap_explanation": [
        ("Why did the model predict a high fuel rate for the current state of {truck_id}?",
         "shap", "road_grade_pct, payload_kg, speed_kph, engine_load_pct",
         "Return the top SHAP contributors with signed attributions.", "hard"),
        ("Which features drive the fuel-rate prediction for {truck_id} on trip {trip_id}?",
         "shap", "shap_values, feature_names",
         "Return a ranked list of most important features for the given trip.", "hard"),
    ],
    "driver_feedback": [
        ("Give the driver of {truck_id} one actionable tip to reduce fuel use on the current trip.",
         "driver_feedback", "speed_kph, acceleration_mps2, throttle_position_pct, fuel_rate_lph",
         "Return a safe, specific recommendation grounded in observed behaviour.", "medium"),
        ("Is aggressive acceleration increasing fuel use for {truck_id} right now?",
         "driver_feedback", "acceleration_mps2, throttle_position_pct, fuel_rate_lph",
         "Assess recent acceleration episodes and respond with evidence.", "hard"),
    ],
    "rag_evidence_retrieval": [
        ("Explain how road grade affects fuel consumption on heavy-duty trucks.",
         "rag_retrieval", "document_id, topic=road_grade_effects",
         "Return relevant corpus documents with citations, not invented figures.", "easy"),
        ("What does engine load have to do with fuel demand?",
         "rag_retrieval", "document_id, topic=engine_load_fuel_demand",
         "Return corpus passages and paraphrase them accurately.", "easy"),
    ],
    "model_reliability_limitations": [
        ("Can I trust a fuel-rate prediction from this model on a route it has never seen?",
         "limitations", "model_card, validation_summary",
         "State the synthetic-data limitation and unseen-truck caveat honestly.", "easy"),
        ("How accurate is the model on unseen trucks?",
         "limitations", "validation_summary, split_manifest",
         "Report validation metrics only for the unseen-truck split; do not overclaim.", "hard"),
    ],
    "missing_or_invalid_telemetry": [
        ("The telemetry for {truck_id} at this timestamp has no engine_rpm. What should I do?",
         "data_quality", "engine_rpm, missing_value_handling",
         "Flag the missing field, do not fabricate data, and explain the fallback.", "medium"),
        ("Speed is negative in a row for {truck_id}. How should this be handled?",
         "data_quality", "speed_kph, valid_range",
         "Identify the invalid value and recommend exclusion or repair policy.", "medium"),
    ],
}


def make_agent_queries(trucks: pd.DataFrame, trips: pd.DataFrame,
                       rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    truck_ids = trucks["truck_id"].tolist()
    trip_ids = trips["trip_id"].tolist()
    per_category = 55  # 10 categories -> 550 cases
    case_counter = 0
    for cat in AGENT_CATEGORIES:
        templates = AGENT_TEMPLATES.get(cat, [])
        if not templates:
            continue
        for j in range(per_category):
            q, tool, fields, behav, diff = templates[j % len(templates)]
            truck_id = str(rng.choice(truck_ids))
            trip_id = str(rng.choice(trip_ids))
            # prefer a trip belonging to the chosen truck when possible for grounding
            cand = trips[trips["truck_id"] == truck_id]["trip_id"]
            if len(cand):
                trip_id = str(rng.choice(cand.tolist()))
            case_counter += 1
            rows.append({
                "case_id": f"case_{case_counter:04d}",
                "user_query": q.format(truck_id=truck_id, trip_id=trip_id),
                "expected_primary_tool": tool,
                "required_evidence_or_fields": fields,
                "expected_behaviour": behav,
                "difficulty": diff,
                "truck_id": truck_id,
                "trip_id": trip_id,
            })
    df = pd.DataFrame(rows)
    assert len(df) >= 500
    return df


# ---------------------------------------------------------------------------
# RAG knowledge corpus (synthetic, clearly non-OEM)
# ---------------------------------------------------------------------------

RAG_DOCS = [
    {"topic": "fuel_consumption_fundamentals", "title": "What drives heavy-duty fuel consumption",
     "content": "Fuel consumption of a heavy-duty truck is primarily driven by the power demanded at the wheels, which depends on the vehicle mass, speed, acceleration, aerodynamic drag, rolling resistance and road gradient. Higher demanded power requires more fuel per second. The fuel rate (L/h) is therefore an immediate proxy for engine load and driver demand, not a fixed vehicle property."},
    {"topic": "fuel_consumption_fundamentals", "title": "Fuel rate versus total fuel used",
     "content": "Instantaneous fuel rate (L/h) describes how fast fuel is consumed at an instant, while total fuel used is the integral of the rate over time. When analyzing a trip, prefer summing per-second consumption increments; the cumulative trip fuel resets at each trip boundary."},
    {"topic": "engine_load_fuel_demand", "title": "Engine load and fuel demand",
     "content": "Engine load (percent) reflects how much of the engine's rated power is being demanded. Higher load means the engine must inject more fuel to produce more torque and power. Fuel rate generally rises with engine load, roughly linearly at moderate loads and with lower efficiency at very high load."},
    {"topic": "engine_load_fuel_demand", "title": "Idle fuel use",
     "content": "When a truck is stationary with the engine running, fuel rate does not drop to zero; it remains at an idle level because the engine must keep rotating and support auxiliaries. Long idling is a real source of fuel loss in haulage operations."},
    {"topic": "rpm_torque_relationships", "title": "RPM and torque relationship",
     "content": "Engine power is the product of torque and rotational speed. For a given power demand, the torque required falls as RPM rises. Drivers and automated systems generally aim to operate the engine in a favourable torque band around the rated RPM where efficiency is best; lugging at very low RPM can reduce efficiency."},
    {"topic": "rpm_torque_relationships", "title": "Why RPM matters for fuel",
     "content": "Higher engine RPM usually increases internal friction and pumping losses, which raises fuel use for the same output power. Therefore, cruising at a moderate RPM in the correct gear tends to use less fuel than running the same speed at a much higher RPM."},
    {"topic": "payload_effects", "title": "Payload and fuel consumption",
     "content": "Adding payload increases vehicle mass, which raises rolling resistance and the force needed to accelerate and climb. Heavier loads therefore increase fuel rate, especially during acceleration and on grades. The effect is roughly proportional to mass on flat ground and stronger on gradients."},
    {"topic": "payload_effects", "title": "Gross vehicle weight limits",
     "content": "A truck's gross vehicle weight (GVW) is a rated maximum. Operating near GVW increases fuel use and drivetrain stress; operating at very low load improves fuel economy but reduces productivity per trip."},
    {"topic": "road_grade_effects", "title": "How road grade affects fuel and",
     "content": "Climbing a grade requires extra force equal to the component of vehicle weight along the slope, which directly increases engine load and fuel rate. Descending can allow engine braking or coasting with little or no fuel. Steep, sustained uphill sections sharply increase fuel per kilometre."},
    {"topic": "road_grade_effects", "title": "Grade and hill climbing behaviour",
     "content": "On uphill sections, drivers often shift down and increase throttle, raising RPM and load; fuel rate spikes accordingly. Good predictive control can 'roll into' a hill with momentum, reducing peak fuel demand."},
    {"topic": "speed_variability_driving_behaviour", "title": "Speed variability and driving behaviour",
     "content": "Frequent acceleration and braking wastes fuel because energy put into accelerating is then dissipated as heat in braking. Smooth driving with steady speed, gentle throttle and anticipation of stops lowers average fuel use relative to aggressive stop-and-go driving."},
    {"topic": "speed_variability_driving_behaviour", "title": "Aerodynamic drag grows with speed",
     "content": "Aerodynamic drag rises with the square of speed, so the fuel cost per kilometre climbs quickly above about 70-80 kph. High-speed cruising consumes disproportionately more fuel than moderate-speed cruising for a heavy truck."},
    {"topic": "fuel_rate_interpretation", "title": "Interpreting a fuel-rate measurement",
     "content": "A single fuel-rate reading is only meaningful in context: it depends on the current speed, load, grade, payload and throttle. Do not judge a driver or truck on one instantaneous value; use trip-level statistics and fixed conditions."},
    {"topic": "fuel_rate_interpretation", "title": "Fuel-rate units and scale",
     "content": "In the datasets in this project, fuel rate is reported in litres per hour (L/h). Values range from near idle (a few L/h) up to well over 100 L/h during hard acceleration or climbing under load. Always convert to consistent units before comparing."},
    {"topic": "shap_interpretation", "title": "Reading SHAP explanations for fuel rate",
     "content": "SHAP attribution tells you how much each input feature pushed a prediction away from a baseline. For fuel-rate predictions, large positive attributions usually come from high engine load, steep uphill grade, heavy payload and high throttle or speed. Negative attributions indicate conditions that lower fuel demand."},
    {"topic": "shap_interpretation", "title": "SHAP is about attribution, not causation",
     "content": "SHAP values reflect the model's learned associations, not guaranteed physical causation. They help diagnose which inputs mattered for a specific prediction but should be interpreted alongside physical reasoning and the known limitations of the model."},
    {"topic": "driver_feedback_safety", "title": "Safe driver feedback principles",
     "content": "Driver feedback on fuel use should be constructive, specific and non-judgemental. It must never encourage unsafe behaviour such as aggressive coasting downhill, exceeding speed limits, or shutting off the engine in traffic. Safety always outranks fuel economy."},
    {"topic": "driver_feedback_safety", "title": "Delivering actionable feedback",
     "content": "Actionable feedback ties an observation to a specific, safe behaviour, for example: 'strong acceleration was detected in the last 5 minutes; smoother throttle reduced fuel use in similar trips'. Feedback should reference concrete measured evidence."},
    {"topic": "model_limitations", "title": "Limitations of synthetic fuel-rate models",
     "content": "Models trained on synthetic data can be internally consistent and useful for engineering development, but their accuracy on real trucks is not established. Synthetic targets are formula-generated, so performance on this dataset does not predict real-world fuel-prediction accuracy."},
    {"topic": "model_limitations", "title": "Unseen-truck generalization caveats",
     "content": "A model evaluated on trucks it never saw during training is a stronger test than same-truck evaluation. Even so, results depend on how representative the training fleet is, and cannot guarantee performance on other fleets, routes or weather."},
    {"topic": "model_limitations", "title": "Do not use target-derived features",
     "content": "Do not feed consumed-fuel increments or cumulative trip fuel into a model that predicts instantaneous fuel rate: these are integrals of the target and leak the answer. Exclude them, and never give the model future timestamps or future telemetry."},
]


def make_rag_corpus() -> None:
    with open(Path(OUTPUT_DIR) / "rag_knowledge_corpus.jsonl", "w", encoding="utf-8") as fh:
        for i, d in enumerate(RAG_DOCS):
            doc = {
                "document_id": f"doc_{i + 1:03d}",
                "title": d["title"],
                "topic": d["topic"],
                "content": d["content"],
                "source_type": "synthetic_educational",
                "source_reference": "Synthetic RAG corpus (non-OEM educational text; not manufacturer guidance)",
                "is_oem_manual": False,
            }
            fh.write(json.dumps(doc) + "\n")


# ---------------------------------------------------------------------------
# data_dictionary.csv
# ---------------------------------------------------------------------------

DATA_DICTIONARY = [
    # (name, type, unit, description, role, freq, valid_range, notes)
    ("record_id", "int64", "-", "Globally unique telemetry record identifier.", "identifier", 1.0, "0..963696", "Sequential across all trips."),
    ("timestamp_utc", "datetime64[ns]", "ISO-8601 UTC", "Record timestamp (ens 1 Hz grid within each trip).", "time", 1.0, "2023-01..2023-03", "UTC; constant interval within a trip."),
    ("truck_id", "object", "-", "Synthetic truck identifier.", "grouping_key", 1.0, "truck_001..truck_029", "Matches truck_metadata.csv."),
    ("trip_id", "object", "-", "Synthetic trip identifier.", "grouping_key", 1.0, "trip_000001..", "Matches trip_metadata.csv."),
    ("vehicle_class", "object", "-", "Weight class of the truck.", "feature", 1.0, "32T | 40T | 49T", "Constant per truck."),
    ("gross_vehicle_weight_kg", "int64", "kg", "Rated gross vehicle weight.", "feature", 1.0, "32000/40000/49000", "Constant per truck; equals class GVW."),
    ("payload_kg", "int64", "kg", "Cargo mass carried on this trip.", "feature", 1.0, ">0 .. GVW-curb", "Constant within a trip; varies across trips."),
    ("vehicle_mass_kg", "int64", "kg", "Curb mass + payload.", "feature", 1.0, "~17000..49000", "Constant within a trip."),
    ("speed_kph", "float64", "km/h", "Ground speed.", "feature", 1.0, "0..92", "0 when stationary."),
    ("acceleration_mps2", "float64", "m/s^2", "Rate of change of speed.", "feature", 1.0, "~-1.6..0.85", "Derived from speed."),
    ("road_grade_pct", "float64", "%", "Longitudinal road slope.", "feature", 1.0, "-6..6", "Smooth profile per route."),
    ("latitude", "float64", "deg", "Synthetic latitude.", "feature", 1.0, "23..31", "Moves with the trip; not real geo."),
    ("longitude", "float64", "deg", "Synthetic longitude.", "feature", 1.0, "72..80", "Moves with the trip; not real geo."),
    ("ignition_status", "int8", "-", "1 = engine on, 0 = engine off.", "feature", 1.0, "0 | 1", "Engine-off rows have zero engine/fuel values."),
    ("engine_rpm", "float64", "rpm", "Engine rotational speed.", "feature", 1.0, "0..2200", "0 when engine off; ~620 idle."),
    ("engine_load_pct", "float64", "%", "Engine load relative to rated power.", "feature", 1.0, "0..100", "0 when engine off."),
    ("throttle_position_pct", "float64", "%", "Throttle position.", "feature", 1.0, "0..100", "0 when engine off or braking."),
    ("torque_nm", "float64", "N m", "Engine torque (capped at rated torque).", "feature", 1.0, "0..~2100", "0 when engine off."),
    ("coolant_temp_c", "float64", "degC", "Coolant temperature.", "feature", 1.0, "ambient..95", "Smooth warm-up / cool-down."),
    ("intake_air_temp_c", "float64", "degC", "Intake air temperature.", "feature", 1.0, "ambient..ambient+30", "Higher while engine running."),
    ("ambient_temp_c", "float64", "degC", "Outside ambient temperature.", "feature", 1.0, "~0..40", "Slow smooth drift."),
    ("fuel_rate_lph", "float64", "L/h", "Instantaneous fuel rate (TARGET).", "target", 1.0, "0..220", "The prediction target."),
    ("fuel_consumed_l_increment", "float64", "L", "Fuel consumed in this sampling interval.", "excluded", 1.0, ">=0", "= fuel_rate_lph / 3600. EXCLUDED as feature."),
    ("cumulative_fuel_l_trip", "float64", "L", "Cumulative fuel used since trip start.", "excluded", 1.0, ">=0", "Resets at trip boundaries. EXCLUDED as feature."),
]


def make_data_dictionary() -> None:
    cols = ["column_name", "dtype", "unit", "description", "role", "fill_rate", "valid_range", "notes"]
    rows = []
    for name, dtype, unit, desc, role, fill, rng, notes in DATA_DICTIONARY:
        rows.append({
            "column_name": name,
            "dtype": dtype,
            "unit": unit,
            "description": desc,
            "role": role,
            "fill_rate": fill,
            "valid_range": rng,
            "notes": notes,
        })
    pd.DataFrame(rows)[cols].to_csv(Path(OUTPUT_DIR) / "data_dictionary.csv", index=False)


# ---------------------------------------------------------------------------
# generation_config.json
# ---------------------------------------------------------------------------

def make_generation_config() -> None:
    cfg = {
        "purpose": "Synthetic heavy-duty truck research telemetry suite",
        "synthetic_disclaimer": "All data are synthetic and do not represent real measured truck telemetry.",
        "seed": SEED,
        "target_rows": TOTAL_ROWS,
        "sampling_interval_seconds": SAMPLING_INTERVAL_S,
        "num_trucks": int(NUM_TRUCKS),
        "fleet_classes": FLEET_CLASSES,
        "class_specs": CLASS_SPECS,
        "route_specs": ROUTE_SPECS,
        "route_probabilities": ROUTE_PROBABILITIES,
        "split_sizes": {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST},
        "physical_constants": {
            "gravity_mps2": GRAVITY_MPS2,
            "air_density_kg_m3": AIR_DENSITY_KG_M3,
            "diesel_energy_j_per_l": DIESEL_ENERGY_J_PER_L,
            "drivetrain_efficiency": DRIVETRAIN_EFFICIENCY,
            "accel_max_mps2": 0.85,
            "decel_max_mps2": 1.60,
        },
        "model_notes": {
            "grade_fraction_to_angle": "sin(atan(percent/100))",
            "motion_power_limited": True,
            "power_feasibility_tolerance": POWER_FEASIBILITY_TOLERANCE,
            "highway_grade_mode": "segments (deterministic smooth uphill/downhill)",
            "engine_load_range_pct": "0..100 (definitional fraction of rated power)",
        },
        "target_column": TARGET_COLUMN,
        "excluded_features": EXCLUDED_FEATURES,
        "recommended_features": RECOMMENDED_FEATURES,
        "generator": "scripts/synthetic_heavy_truck/gen_telemetry.py",
        "reproducibility_note": "Fixed seed reproduces identical outputs given the same numpy/pandas versions.",
    }
    with open(Path(OUTPUT_DIR) / "generation_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# dataset_card.json
# ---------------------------------------------------------------------------

def make_dataset_card(total_rows: int, written_rows: int) -> None:
    card = {
        "dataset_name": "synthetic_heavy_truck_research",
        "version": "1.0",
        "provenance": "Synthetically generated by scripts/synthetic_heavy_truck/gen_telemetry.py (seed=%d)." % SEED,
        "synthetic_status": "SYNTHETIC. No claim that any record represents real measured truck telemetry.",
        "intended_use": [
            "Fuel-rate prediction (supervised regression)",
            "Cross-truck generalization studies (truck-wise splits)",
            "Explainability (SHAP) evaluations",
            "Telemetry replay / agentic AI evaluation",
        ],
        "not_for": [
            "Claiming real-world fuel-prediction accuracy",
            "Manufacturer comparisons or OEM benchmarking",
        ],
        "row_count_expected": int(TOTAL_ROWS),
        "row_count_written": int(written_rows),
        "sampling_interval_seconds": SAMPLING_INTERVAL_S,
        "num_trucks": int(NUM_TRUCKS),
        "num_trips": None,  # filled after trip metadata is read
        "target_column": TARGET_COLUMN,
        "recommended_features": RECOMMENDED_FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "leakage_safeguards": [
            "No truck appears in more than one split (truck-wise splitter)",
            "Target-derived fields (fuel increment, cumulative fuel) excluded from inputs",
            "No future timestamps or future telemetry used",
        ],
        "validation": "See validation_summary.json (produced by validate_dataset.py) "
                    "- includes power-feasibility and highway-grade-coverage checks.",
        "model_physics": {
            "motion_power_limited": True,
            "power_feasibility_tolerance": "0.02 (documented numerical tolerance)",
            "grade_fraction_to_angle": "sin(atan(percent/100))",
            "inertia": "signed (braking assists, overrun fuel drop)",
            "highway_grade_mode": "deterministic smooth uphill/downhill segments",
        },
        "limitations": [
            "Target is formula-generated; metrics on this dataset do not establish real-world accuracy.",
            "Physics is simplified (steady-state efficiency, no weather/tyre/road-surface detail).",
            "Synthetic fleet is not a sample of any real population of trucks.",
            "Coordinates are synthetic and not geographic reality.",
        ],
        "reproducibility": "Regenerate with the same seed; see README.md.",
        "license_note": "Generated for this project; no external dataset redistribution is claimed.",
    }
    with open(Path(OUTPUT_DIR) / "dataset_card.json", "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------

def make_readme(trips: pd.DataFrame, written_rows: int) -> None:
    n_trips = len(trips)
    readme = f"""# Synthetic Heavy-Duty Truck Research Dataset Suite

Synthetic telemetry designed for **fuel-rate prediction**, **cross-truck
generalization**, **explainability**, **telemetry replay** and **agentic AI
evaluation** for heavy-duty trucks.

> ## Synthetic status
> All records in this suite are **synthetically generated**. None of them
> represent real measured truck telemetry, and the suite makes no claim to do
> so. The generator is reproducible from a fixed seed.

## 1. Dataset provenance

Generated by `scripts/synthetic_heavy_truck/gen_telemetry.py` with a fixed seed
(`{SEED}`). The physical model is a simplified, internally consistent
longitudinal vehicle model (rolling resistance + aerodynamic drag + grade +
inertia -> tractive power -> engine load -> fuel rate). Motion is
**power-limited**: commanded acceleration is constrained to what the engine can
already deliver at the current speed (solved implicitly at the end-of-step
speed), so no record ever demands more than rated power (verified by a dedicated
validation check within a documented 2% numerical tolerance). Road grade uses
the correct `sin(atan(%/100))` conversion; highway routes use smooth,
deterministic uphill/downhill segments rather than per-record random grades.
Braking/overrun uses signed inertia (momentum assists, so fuel drops on
deceleration). No real-world telemetry was used. Coordinates, identifiers and
every value are invented.

## 2. Files

| File | Description |
| --- | --- |
| `telemetry_master_963697.csv.gz` | Master telemetry, exactly {TOTAL_ROWS} rows |
| `truck_metadata.csv` | 29 synthetic trucks (+ static attributes |
| `trip_metadata.csv` | Trip-level aggregates agreeing with telemetry |
| `truckwise_split_manifest.csv` | Leakage-safe truck-wise split (20/4/5) |
| `telemetry_replay_manifest.csv` | Replay instructions per trip |
| `agent_evaluation_queries.csv` | {n_trips} grounded evaluation cases |
| `rag_knowledge_corpus.jsonl` | Synthetic educational RAG corpus |
| `dataset_card.json` | Dataset card (intended use, limitations) |
| `data_dictionary.csv` | Column dictionary with units and roles |
| `generation_config.json` | Exact generation parameters and seed |
| `validation_summary.json` | Automated validation results (separate script) |
| `README.md` | This document |

## 3. Feature definitions and units

See `data_dictionary.csv` for the full schema. Key points:

* `fuel_rate_lph` is the **target** (instantaneous fuel rate, L/h).
* `fuel_consumed_l_increment` = `fuel_rate_lph / 3600` (litres per second at 1 Hz).
* `cumulative_fuel_l_trip` is the within-trip cumulative sum, resetting at each
  trip boundary.
* Engine-off records (`ignition_status == 0`) have zero speed, engine and fuel
  values; temperatures evolve smoothly.
* `vehicle_mass_kg` = curb mass + `payload_kg`; `gross_vehicle_weight_kg` is the
  rated class GVW.

## 4. Target variable

`fuel_rate_lph` (L/h) is the regression target. It is **formula-generated** from
the physical model, so model metrics on this dataset are engineering/self-
consistency checks and **do not establish real-world fuel-prediction accuracy**.

## 5. Intended use

* Supervised fuel-rate prediction (regression, tabular time series).
* Cross-truck generalization: train on `train` trucks, validate on `val`,
  evaluate on `test` (unseen trucks).
* Explainability (SHAP) and agent evaluation on grounded queries.
* Telemetry replay for pipeline/UI development.

## 6. Leakage risks and safeguards

The following are **excluded from model inputs** for predicting
`fuel_rate_lph`:

* `fuel_consumed_l_increment` (an integral of the target),
* `cumulative_fuel_l_trip` (the target's cumulative sum),
* any direct mathematical transformation of the target,
* any future timestamp or future telemetry.

The recommended input allowlist is in `generation_config.json` and
`dataset_card.json`.

## 7. Train/validation/test strategy

Split at the **truck level** (`truckwise_split_manifest.csv`): 20 train, 4
validation, 5 unseen-truck test. No truck appears in more than one split, and no
individual rows are randomly split. Join training rows from master telemetry by
`truck_id IN (train trucks)`; evaluation must use only test trucks.

## 8. How to regenerate

```
python scripts/synthetic_heavy_truck/gen_telemetry.py --seed 20240617
python scripts/synthetic_heavy_truck/validate_dataset.py
```

The same seed reproduces the same files (given the same numpy/pandas versions).

## 9. Limitations

* Synthetic, simplified physics; no weather, tyre, road-surface or traffic detail.
* Fleet is not a sample of any real population.
* Coordinates are not real geography.
* Synthetic targets mean validation on this suite cannot predict real-world fuel
  accuracy.
* Highway profiles include deterministic climb/descent segments; sustained uphill
  cruise can run near rated power, so fuel economy is load- and grade-sensitive.

## 10. Generation summary

Rows written: {written_rows:,} (expected {TOTAL_ROWS:,}) · Trips: {n_trips:,}.
Wall-clock generation time is intentionally not embedded here so that every
artifact is byte-for-byte reproducible from the fixed seed.
"""
    (Path(OUTPUT_DIR) / "README.md").write_text(readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate synthetic heavy-truck dataset suite")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=str, default=OUTPUT_DIR)
    ap.add_argument("--no-telemetry", action="store_true",
                    help="Reuse an existing telemetry file (dev/re-run mode)")
    args = ap.parse_args()

    # Allow an --out override; this updates the module-level constant used by
    # the writer helpers below (defaults to synthetic_config.OUTPUT_DIR).
    globals()['OUTPUT_DIR'] = args.out
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    trucks = build_trucks(rng, out_dir)

    if args.no_telemetry:
        trips = pd.read_csv(out_dir / "trip_metadata.csv")
        total_written = int(trips["record_count"].sum())
    else:
        trips, total_written = generate_telemetry(trucks, out_dir, rng)

    splits = make_splits(trucks, rng)
    splits.to_csv(out_dir / "truckwise_split_manifest.csv", index=False)

    replay = make_replay_manifest(trips)
    replay.to_csv(out_dir / "telemetry_replay_manifest.csv", index=False)

    agent = make_agent_queries(trucks, trips, rng)
    agent.to_csv(out_dir / "agent_evaluation_queries.csv", index=False)

    make_rag_corpus()
    make_data_dictionary()
    make_generation_config()
    make_dataset_card(TOTAL_ROWS, total_written)

    elapsed = time.time() - t0  # console-only timing; never embedded in artifacts
    make_readme(trips, total_written)

    print(f"telemetry rows written : {total_written:,} (expected {TOTAL_ROWS:,})")
    print(f"trips                 : {len(trips):,}")
    print(f"trucks                : {len(trucks)}")
    print(f"agent evaluation cases : {len(agent)}")
    print(f"split counts           : {splits['split'].value_counts().to_dict()}")
    print(f"generation time        : {elapsed:.1f}s")
    print(f"output directory       : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

