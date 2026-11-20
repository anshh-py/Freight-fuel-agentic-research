"""Automated validation for the synthetic heavy-duty truck dataset suite.

Loads the generated artifacts (telemetry, metadata, manifests) and runs the
documented checks. Writes ``validation_summary.json`` into the output directory
and prints a concise report. Honest: failures are reported, never silently
ignored.

Optional reproducibility check:
    python validate_dataset.py --repro
regenerates the whole suite with the same seed into a temporary directory and
compares file hashes against the existing outputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_config import (
    AIR_DENSITY_KG_M3,
    DRIVETRAIN_EFFICIENCY,
    EXCLUDED_FEATURES,
    GRAVITY_MPS2,
    OUTPUT_DIR,
    POWER_FEASIBILITY_TOLERANCE,
    RECOMMENDED_FEATURES,
    SEED,
    TARGET_COLUMN,
    TELEMETRY_COLUMNS,
    TOTAL_ROWS,
)
from gen_telemetry import build_trucks

CHECKS: list[dict] = []


def record(name: str, passed: bool, detail: str) -> None:
    CHECKS.append({"check": name, "passed": bool(passed), "detail": detail})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_telemetry(df: pd.DataFrame, trips: pd.DataFrame, trucks: pd.DataFrame):
    # 1. exactly 963,697 rows
    record("telemetry_row_count", len(df) == TOTAL_ROWS,
           f"rows={len(df):,} expected={TOTAL_ROWS:,}")

    # 1b. exact schema (columns == TELEMETRY_COLUMNS, in order)
    record("telemetry_schema_exact", list(df.columns) == TELEMETRY_COLUMNS,
           f"n_cols={len(df.columns)} expected={len(TELEMETRY_COLUMNS)}")

    # 2. no duplicate record ids
    record("record_ids_unique", df["record_id"].is_unique,
           f"n_unique={df['record_id'].nunique():,}")

    # 3. no missing required values
    missing = df.isna().sum()
    any_missing = bool((missing > 0).any())
    record("no_missing_required", not any_missing,
           "columns with NAs: " + ", ".join(missing[missing > 0].index) if any_missing else "none")

    # 4. all truck_ids exist in truck metadata
    meta_trucks = set(trucks["truck_id"])
    bad_trucks = set(df["truck_id"]) - meta_trucks
    record("truck_ids_in_metadata", not bad_trucks,
           f"unknown trucks={sorted(bad_trucks)}")

    # 5. all trip_ids exist in trip metadata (and vice-versa)
    meta_trips = set(trips["trip_id"])
    tele_trips = set(df["trip_id"])
    record("trip_ids_in_metadata", (tele_trips == meta_trips) and (tele_trips - meta_trips) == set(),
           f"in_telemetry_not_meta={len(tele_trips - meta_trips)} in_meta_not_telemetry={len(meta_trips - tele_trips)}")

    # 6. each trip belongs to exactly one truck
    per_trip_trucks = df.groupby("trip_id")["truck_id"].nunique()
    record("trip_single_truck", bool((per_trip_trucks == 1).all()),
           f"max_trucks_per_trip={per_trip_trucks.max()}")

    # 8+9. fuel increments consistent with rate & interval; cumulative resets
    inc_ok = np.allclose(
        df["fuel_consumed_l_increment"].values,
        df["fuel_rate_lph"].values / 3600.0,
        atol=1e-9,
    )
    record("fuel_increment_consistency", bool(inc_ok),
           "increment == fuel_rate_lph / 3600")

    g = df.groupby("trip_id", sort=False)
    cum_ok = True
    reset_ok = True
    for _, sub in g:
        inc = sub["fuel_consumed_l_increment"].to_numpy()
        cum = sub["cumulative_fuel_l_trip"].to_numpy()
        if not np.allclose(cum, np.cumsum(inc), atol=1e-9):
            cum_ok = False
            break
        if not (abs(cum[0]) < 0.05):
            reset_ok = False
            break
    record("cumulative_within_trip", bool(cum_ok), "cumulative == cumsum(increment) per trip")
    record("cumulative_resets_at_trip_start", bool(reset_ok),
           "each trip starts at ~0 cumulative (<= one interval's fuel)")

    # 10. ranges reasonable
    bounds = {
        "speed_kph": (0.0, 100.0),
        "acceleration_mps2": (-3.0, 3.0),
        "road_grade_pct": (-8.0, 8.0),
        "engine_rpm": (0.0, 2300.0),
        "engine_load_pct": (0.0, 120.0),
        "throttle_position_pct": (0.0, 100.0),
        "torque_nm": (0.0, 3000.0),
        "fuel_rate_lph": (0.0, 250.0),
    }
    bad = []
    for col, (lo, hi) in bounds.items():
        cmin, cmax = df[col].min(), df[col].max()
        if not (lo <= cmin and cmax <= hi):
            bad.append(f"{col}(min={cmin:.2f},max={cmax:.2f})")
    record("value_ranges_reasonable", not bad, "within bounds" if not bad else "; ".join(bad))

    # 11. engine-off records consistent
    off = df["ignition_status"] == 0
    off_engine_ok = bool(off.sum() == 0 or (
        (df.loc[off, "speed_kph"] == 0).all() and
        (df.loc[off, "engine_rpm"] == 0).all() and
        (df.loc[off, "engine_load_pct"] == 0).all() and
        (df.loc[off, "throttle_position_pct"] == 0).all() and
        (df.loc[off, "torque_nm"] == 0).all() and
        (df.loc[off, "fuel_rate_lph"] == 0).all() and
        (df.loc[off, "fuel_consumed_l_increment"] == 0).all()
    ))
    record("engine_off_consistent", off_engine_ok,
           f"engine-off rows={off.sum():,} (zero speed/engine/fuel expected)")

    # 12. trip aggregates match telemetry within tolerance
    tele_agg = df.groupby("trip_id", sort=False).agg(
        record_count=("record_id", "size"),
        distance_km=("speed_kph", lambda s: s.sum() / 3600.0),
        payload_kg=("payload_kg", "mean"),
    )
    m = trips.set_index("trip_id")
    ok_dist, ok_count, ok_payload = True, True, True
    for trip_id, row in tele_agg.iterrows():
        meta = m.loc[trip_id]
        if abs(row["distance_km"] - meta["distance_km"]) > 0.05:
            ok_dist = False
        if row["record_count"] != meta["record_count"]:
            ok_count = False
        if abs(float(row["payload_kg"]) - float(meta["payload_kg"])) > 1.0:
            ok_payload = False
    record("trip_distance_matches", ok_dist, "telemetry-derived distance vs trip_metadata")
    record("trip_record_count_matches", ok_count, "telemetry row count vs trip_metadata")
    record("trip_payload_matches", ok_payload, "telemetry mean payload vs trip_metadata")

    # vehicle-class / GVW consistency per truck
    per_truck_classes = df.groupby("truck_id")["vehicle_class"].nunique()
    record("truck_class_consistent", bool((per_truck_classes == 1).all()),
           "one vehicle_class per truck")

    # highway grade coverage & smooth transitions (final correction: the
    # deterministic uphill/downhill segments must actually appear at cruise
    # speeds, and grade must be continuous - no per-record independent jumps)
    hw_trips = set(trips.loc[trips["route_type"] == "highway", "trip_id"])
    hw_on = df[df["trip_id"].isin(hw_trips) & (df["ignition_status"] == 1)]
    pos = bool((hw_on["road_grade_pct"] > 1.5).any())
    neg = bool((hw_on["road_grade_pct"] < -1.5).any())
    record("highway_grade_coverage", pos and neg,
           f"uphill_gt1.5pc_present={pos} downhill_lt-1.5pc_present={neg} "
           f"(highway_on_records={len(hw_on):,})")
    if len(hw_on):
        dmax = hw_on.groupby("trip_id")["road_grade_pct"].apply(
            lambda s: s.diff().abs().max()).dropna()
        smax = float(dmax.max()) if len(dmax) else 0.0
        record("highway_grade_smooth_transitions", smax <= 0.5,
               f"max_abs_dgrade_per_s={smax:.3f} (limit 0.5)")
    else:
        record("highway_grade_smooth_transitions", False,
               "no highway ignition-on records")

    # 8b. leakage: recommended feature allowlist present; target-derived fields
    # and the target itself must NOT be among recommended inputs.
    colset = set(df.columns)
    missing_rec = [c for c in RECOMMENDED_FEATURES if c not in colset]
    record("recommended_features_present", not missing_rec,
           "all recommended features exist" if not missing_rec else f"missing={missing_rec}")
    leak_in_rec = [c for c in EXCLUDED_FEATURES if c in RECOMMENDED_FEATURES]
    record("no_target_derived_input_features", not leak_in_rec,
           "target-derived fields excluded from inputs" if not leak_in_rec
           else f"leak={leak_in_rec}")
    target_leak = TARGET_COLUMN in RECOMMENDED_FEATURES
    record("target_not_in_input_features", not target_leak,
           "target column absent from recommended inputs" if not target_leak
           else "target IS in recommended inputs")


def validate_manifests(trucks: pd.DataFrame, trips: pd.DataFrame,
                       splits: pd.DataFrame, replay: pd.DataFrame,
                       agent: pd.DataFrame):
    # 7. truck-wise split zero overlap
    total_assigned = len(splits)
    record("split_all_trucks_assigned", total_assigned == 29,
           f"assigned={total_assigned}")
    record("split_zero_overlap", splits["truck_id"].is_unique,
           "every truck appears exactly once")
    split_counts = splits["split"].value_counts().to_dict()
    record("split_sizes", split_counts.get("train") == 20 and split_counts.get("val") == 4 and split_counts.get("test") == 5,
           f"counts={split_counts}")

    # replay manifest: trips in 1:1 correspondence with trip metadata
    record("replay_trips_match", set(replay["trip_id"]) == set(trips["trip_id"]),
           f"replay_trips={len(replay)} meta_trips={len(trips)}")

    # agent queries: >= 500 cases, grounded in real truck/trip ids
    known_trucks = set(trucks["truck_id"])
    known_trips = set(trips["trip_id"])
    agent_ok = (len(agent) >= 500 and
                set(agent["truck_id"]).issubset(known_trucks) and
                set(agent["trip_id"]).issubset(known_trips) and
                agent["case_id"].is_unique)
    record("agent_queries_grounded", bool(agent_ok),
           f"cases={len(agent)} (>=500, grounded ids, unique case_id)")


def validate_power_feasibility(df: pd.DataFrame) -> None:
    """Verify no record's tractive-power demand exceeds rated power by more than
    the documented numerical tolerance (POWER_FEASIBILITY_TOLERANCE).

    Reconstructs the internal per-truck parameters (crr, cda, rated power) by
    calling ``build_trucks`` with a fresh seeded RNG - it consumes exactly the
    same drawn sequence as generation, so it reproduces the identical trucks.
    The demand uses the same force equations as ``engine_telemetry``.
    """
    with tempfile.TemporaryDirectory() as td:
        trucks_full = build_trucks(np.random.default_rng(SEED), Path(td))
    tf = trucks_full.set_index("truck_id")

    v = df["speed_kph"].to_numpy() / 3.6
    m = df["vehicle_mass_kg"].to_numpy().astype(float)
    crr = df["truck_id"].map(tf["crr"]).to_numpy()
    cda = df["truck_id"].map(tf["cda_m2"]).to_numpy()
    rated = df["truck_id"].map(tf["rated_power_kw"]).to_numpy()

    f_aero = 0.5 * AIR_DENSITY_KG_M3 * cda * v ** 2
    f_roll = crr * m * GRAVITY_MPS2
    f_grade = m * GRAVITY_MPS2 * np.sin(
        np.arctan(np.clip(df["road_grade_pct"].to_numpy(), -89.0, 89.0) / 100.0))
    # signed inertia - must match the generator's engine model (negative
    # acceleration reduces tractive demand; momentum assists during braking)
    f_inertia = m * df["acceleration_mps2"].to_numpy()
    need = np.maximum(f_roll + f_aero + f_grade + f_inertia, 0.0)
    p_eng = need * v / 1000.0 / DRIVETRAIN_EFFICIENCY

    tol = POWER_FEASIBILITY_TOLERANCE
    exceed = p_eng > rated * (1.0 + tol)
    ratio = np.divide(p_eng, rated, out=np.zeros_like(p_eng), where=rated > 0)
    maxratio = float(ratio.max()) if len(ratio) else 0.0
    record("power_feasibility", int(exceed.sum()) == 0,
           f"records_exceeding_rated_x{1 + tol:.3f}={int(exceed.sum()):,} "
           f"(tol={tol:.0%}) max_demand_x_rated={maxratio:.4f}")


def validate_repro(out_dir: Path):
    tmp = out_dir / "_repro_run"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cmd = [sys.executable, str(Path(__file__).with_name("gen_telemetry.py")),
           "--seed", str(SEED), "--out", str(tmp)]
    try:
        subprocess.run(cmd, check=True, timeout=1200, capture_output=True)
    except subprocess.CalledProcessError as e:
        record("reproducibility", False,
               f"regeneration failed: {e.stderr.decode()[-800:]}")
        shutil.rmtree(tmp, ignore_errors=True)
        return

    files = ["telemetry_master_963697.csv.gz", "truck_metadata.csv",
             "trip_metadata.csv", "truckwise_split_manifest.csv",
             "telemetry_replay_manifest.csv", "agent_evaluation_queries.csv",
             "rag_knowledge_corpus.jsonl", "data_dictionary.csv",
             "generation_config.json", "dataset_card.json"]
    diffs = []
    for f in files:
        a, b = out_dir / f, tmp / f
        if not b.exists():
            diffs.append(f"{f}:missing_in_regen")
            continue
        if _sha256(a) != _sha256(b):
            diffs.append(f"{f}:hash_differs")
        # For the gzip telemetry, also compare decompressed content directly so
        # byte-level header differences can never mask a content difference.
        if f.endswith(".csv.gz"):
            if gzip.decompress(a.read_bytes()) != gzip.decompress(b.read_bytes()):
                diffs.append(f"{f}:decompressed_content_differs")
    record("reproducibility", not diffs,
           "identical hashes + decompressed content" if not diffs else "; ".join(diffs))
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate synthetic heavy-truck dataset suite")
    ap.add_argument("--out", type=str, default=OUTPUT_DIR)
    ap.add_argument("--repro", action="store_true",
                    help="Regenerate with the same seed and compare hashes")
    args = ap.parse_args()
    out_dir = Path(args.out)

    trucks = pd.read_csv(out_dir / "truck_metadata.csv")
    trips = pd.read_csv(out_dir / "trip_metadata.csv")
    splits = pd.read_csv(out_dir / "truckwise_split_manifest.csv")
    replay = pd.read_csv(out_dir / "telemetry_replay_manifest.csv")
    agent = pd.read_csv(out_dir / "agent_evaluation_queries.csv")

    df = pd.read_csv(out_dir / "telemetry_master_963697.csv.gz")
    validate_telemetry(df, trips, trucks)
    validate_power_feasibility(df)
    validate_manifests(trucks, trips, splits, replay, agent)
    del df

    if args.repro:
        validate_repro(out_dir)

    failures = [c for c in CHECKS if not c["passed"]]
    summary = {
        "dataset": "synthetic_heavy_truck_research",
        "generated_seed": SEED,
        "total_checks": len(CHECKS),
        "passed": len(CHECKS) - len(failures),
        "failed": len(failures),
        "all_passed": not failures,
        "checks": CHECKS,
        "failed_checks": [c["check"] for c in failures],
        "honesty_note": "Failures are reported above and are never silently ignored. "
                         "This summary is deterministic (no wall-clock timestamps).",
    }
    with open(out_dir / "validation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nValidation complete: {summary['passed']}/{summary['total_checks']} passed, "
          f"{summary['failed']} failed.")
    for c in failures:
        print(f"  FAIL  {c['check']}: {c['detail']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
