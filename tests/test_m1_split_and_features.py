"""Focused tests for M1.1: split enforcement and feature validation.

These use small synthetic fixtures (a few trucks / rows), not the full
963K-row dataset, so the suite stays fast and isolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import enforce_split
from src.features.build import FeatureValidationError, build_feature_matrix
from src.features.spec import (
    get_denylist,
    get_feature_set,
    get_target_column,
    load_feature_spec,
    validate_spec,
)

SPEC = load_feature_spec()


def _make_telemetry(trucks: list[str], rows_per_truck: int = 4) -> pd.DataFrame:
    """Build a minimal telemetry-like frame covering all schema columns."""
    cols = {
        "record_id": [],
        "timestamp_utc": [],
        "truck_id": [],
        "trip_id": [],
        "vehicle_class": [],
        "gross_vehicle_weight_kg": [],
        "payload_kg": [],
        "vehicle_mass_kg": [],
        "speed_kph": [],
        "acceleration_mps2": [],
        "road_grade_pct": [],
        "latitude": [],
        "longitude": [],
        "ignition_status": [],
        "engine_rpm": [],
        "engine_load_pct": [],
        "throttle_position_pct": [],
        "torque_nm": [],
        "coolant_temp_c": [],
        "intake_air_temp_c": [],
        "ambient_temp_c": [],
        "fuel_rate_lph": [],
        "fuel_consumed_l_increment": [],
        "cumulative_fuel_l_trip": [],
    }
    classes = {"truck_001": "32T", "truck_002": "32T", "truck_003": "40T",
               "truck_004": "40T", "truck_005": "49T", "truck_006": "49T",
               "truck_007": "32T", "truck_008": "40T", "truck_009": "49T"}
    n = 0
    for t in trucks:
        for i in range(rows_per_truck):
            n += 1
            cols["record_id"].append(n)
            cols["timestamp_utc"].append(f"2026-01-01T00:00:0{n % 10}Z")
            cols["truck_id"].append(t)
            cols["trip_id"].append(f"{t}_trip_{i}")
            cls = classes.get(t, "40T")
            cols["vehicle_class"].append(cls)
            cols["gross_vehicle_weight_kg"].append(32000 if cls == "32T"
                                                   else 40000 if cls == "40T"
                                                   else 49000)
            cols["payload_kg"].append(1000 * i)
            cols["vehicle_mass_kg"].append(20000 + 1000 * i)
            cols["speed_kph"].append(50.0)
            cols["acceleration_mps2"].append(0.1)
            cols["road_grade_pct"].append(0.0)
            cols["latitude"].append(20.0 + i * 0.001)
            cols["longitude"].append(73.0 + i * 0.001)
            cols["ignition_status"].append(1)
            cols["engine_rpm"].append(1400.0)
            cols["engine_load_pct"].append(40.0 + i)
            cols["throttle_position_pct"].append(30.0)
            cols["torque_nm"].append(1000.0)
            cols["coolant_temp_c"].append(80.0)
            cols["intake_air_temp_c"].append(30.0)
            cols["ambient_temp_c"].append(27.0)
            cols["fuel_rate_lph"].append(25.0 + i)
            cols["fuel_consumed_l_increment"].append(0.007)
            cols["cumulative_fuel_l_trip"].append(10.0 + i)
    return pd.DataFrame(cols)


def _make_manifest(assign: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"truck_id": t, "split": s} for t, s in assign.items()]
    )


# --------------------------------------------------------------------------
# Split isolation and manifest enforcement
# --------------------------------------------------------------------------

def test_enforce_split_disjoint_and_full_accounting():
    trucks = [f"truck_{i:03d}" for i in range(1, 10)]
    assign = {t: ("train" if i < 4 else "val" if i < 6 else "test")
              for i, t in enumerate(trucks)}
    telemetry = _make_telemetry(trucks)
    manifest = _make_manifest(assign)

    res = enforce_split(telemetry, manifest, expected_counts=(4, 2, 3))

    assert set(res.train_trucks) == set(trucks[:4])
    assert set(res.val_trucks) == set(trucks[4:6])
    assert set(res.test_trucks) == set(trucks[6:])
    assert res.total_rows == len(telemetry)
    # row accounting equals full size
    assert len(res.train) + len(res.val) + len(res.test) == len(telemetry)

    # per-frame trucks entirely disjoint
    all_tr = set(res.train["truck_id"]) & set(res.val["truck_id"])
    all_tt = set(res.train["truck_id"]) & set(res.test["truck_id"])
    all_vt = set(res.val["truck_id"]) & set(res.test["truck_id"])
    assert not all_tr and not all_tt and not all_vt


def test_enforce_split_rejects_truck_in_two_splits():
    trucks = [f"truck_{i:03d}" for i in range(1, 7)]
    telemetry = _make_telemetry(trucks)
    # truck_001 appears in both train and test (manifest-level overlap)
    rows = [
        {"truck_id": "truck_001", "split": "train"},
        {"truck_id": "truck_001", "split": "test"},
        {"truck_id": "truck_002", "split": "train"},
        {"truck_id": "truck_003", "split": "train"},
        {"truck_id": "truck_004", "split": "val"},
        {"truck_id": "truck_005", "split": "test"},
        {"truck_id": "truck_006", "split": "test"},
    ]
    manifest = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="multiple splits"):
        enforce_split(telemetry, manifest, expected_counts=None)


def test_enforce_split_rejects_mismatched_counts():
    trucks = [f"truck_{i:03d}" for i in range(1, 7)]
    telemetry = _make_telemetry(trucks)
    assign = {"truck_001": "train", "truck_002": "train",
              "truck_003": "train", "truck_004": "val",
              "truck_005": "test", "truck_006": "test"}
    manifest = _make_manifest(assign)
    with pytest.raises(ValueError, match="counts"):
        enforce_split(telemetry, manifest, expected_counts=(4, 1, 1))


def test_enforce_split_rejects_unmapped_truck():
    trucks = [f"truck_{i:03d}" for i in range(1, 7)]
    telemetry = _make_telemetry(trucks)
    # truck_005 not in manifest at all
    assign = {"truck_001": "train", "truck_002": "train",
              "truck_003": "val", "truck_004": "val",
              "truck_006": "test"}
    manifest = _make_manifest(assign)
    with pytest.raises(ValueError, match="absent from manifest"):
        enforce_split(telemetry, manifest, expected_counts=None)


# --------------------------------------------------------------------------
# Feature allowlist / denylist
# --------------------------------------------------------------------------

def test_feature_spec_wellformed():
    validate_spec(SPEC)  # should not raise
    assert get_target_column(SPEC) == "fuel_rate_lph"


def test_feature_sets_have_expected_columns():
    full = get_feature_set(SPEC, "full_operational")
    tele = get_feature_set(SPEC, "telematics_only")
    assert len(full) == 15
    assert len(tele) == 11
    # telematics_only is a strict subset of full_operational
    assert set(tele).issubset(set(full))
    # engine-internal signals only in full set
    assert "engine_rpm" in full and "engine_rpm" not in tele


def test_feature_sets_exclude_all_forbidden():
    denied = set(get_denylist(SPEC))
    for set_name in ("full_operational", "telematics_only"):
        feats = set(get_feature_set(SPEC, set_name))
        assert feats.isdisjoint(denied), (set_name, feats & denied)


def test_build_feature_matrix_full_exact_columns():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    X, y, feats = build_feature_matrix(telemetry, "full_operational", spec=SPEC)
    assert list(X.columns) == get_feature_set(SPEC, "full_operational")
    assert list(y.index) == list(X.index)
    assert y.name == "fuel_rate_lph"
    assert len(X) == len(telemetry)


def test_build_feature_matrix_telematics():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    X, _, feats = build_feature_matrix(telemetry, "telematics_only", spec=SPEC)
    assert list(X.columns) == get_feature_set(SPEC, "telematics_only")


# --------------------------------------------------------------------------
# Target leakage prevention
# --------------------------------------------------------------------------

def test_target_derived_rejected():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 6)])
    # full set never includes target-derived; assert via denylist
    denied = set(get_denylist(SPEC))
    assert "fuel_consumed_l_increment" in denied
    assert "cumulative_fuel_l_trip" in denied
    for set_name in ("full_operational", "telematics_only"):
        feats = set(get_feature_set(SPEC, set_name))
        assert feats.isdisjoint({"fuel_consumed_l_increment",
                                 "cumulative_fuel_l_trip"})


def test_target_separated_never_in_x():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    X, y, _ = build_feature_matrix(telemetry, "full_operational", spec=SPEC)
    assert "fuel_rate_lph" not in X.columns
    assert list(y.values) == list(telemetry["fuel_rate_lph"].values)


def test_forbidden_extra_column_rejected():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    # inject a target-derived column into the telemetry frame
    telemetry["fuel_consumed_l_increment_dup"] = telemetry["fuel_rate_lph"] * 2
    with pytest.raises(FeatureValidationError):
        build_feature_matrix(telemetry, "full_operational", spec=SPEC)


def test_missing_required_feature_rejected():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    telemetry = telemetry.drop(columns=["engine_rpm"])
    with pytest.raises(FeatureValidationError, match="missing"):
        build_feature_matrix(telemetry, "full_operational", spec=SPEC)


def test_unknown_feature_set_rejected():
    telemetry = _make_telemetry([f"truck_{i:03d}" for i in range(1, 4)])
    with pytest.raises(ValueError, match="Unknown feature set"):
        build_feature_matrix(telemetry, "does_not_exist", spec=SPEC)
