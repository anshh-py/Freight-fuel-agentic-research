"""Focused tests for the M3.2 operating-condition robustness analysis.

Covers mask definitions and coverage accounting, payload-bin boundaries and
invalid-GVW handling, trip-metadata join integrity, row/trip/truck counts,
insufficient-support flags, error-concentration denominator/share
reconciliation, and per-truck breakdown correctness — without loading the
963K-row dataset or saved artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.m3_condition_robustness import (
    INFORMATIVE_CONDITIONS,
    MIN_SUPPORT_ROWS,
    MIN_SUPPORT_TRUCKS,
    attach_operating_columns,
    build_condition_report,
    error_concentration,
    make_extended_condition_masks,
    payload_load_factor,
    payload_regime_label,
    per_truck_condition,
    route_type_table,
    subgroup_metrics,
)


def _make_frame(
    trip_ids,
    trucks,
    y_true,
    y_pred,
    *,
    route_type=None,
    vehicle_class="40T",
    speed=None,
    grade=None,
    accel=None,
    ignition=None,
    payload=None,
    gvw=None,
    index=None,
) -> pd.DataFrame:
    n = len(y_true)
    defaults = {
        "route_type": ["highway"] * n,
        "speed_kph": [50.0] * n,
        "road_grade_pct": [0.0] * n,
        "acceleration_mps2": [0.0] * n,
        "ignition_status": [1] * n,
        "payload_kg": [10000] * n,
        "gross_vehicle_weight_kg": [30000] * n,
    }
    data = {
        "trip_id": trip_ids,
        "truck_id": trucks,
        "vehicle_class": [vehicle_class] * n,
        "route_type": route_type if route_type is not None else defaults["route_type"],
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "speed_kph": speed if speed is not None else defaults["speed_kph"],
        "road_grade_pct": grade if grade is not None else defaults["road_grade_pct"],
        "acceleration_mps2": accel if accel is not None else defaults["acceleration_mps2"],
        "ignition_status": ignition if ignition is not None else defaults["ignition_status"],
        "payload_kg": payload if payload is not None else defaults["payload_kg"],
        "gross_vehicle_weight_kg": gvw if gvw is not None else defaults["gross_vehicle_weight_kg"],
    }
    idx = index if index is not None else pd.RangeIndex(n)
    return pd.DataFrame(data, index=idx)


def _simple_frame():
    """8 rows, 2 trucks, 2 trips — realistic-ish but small."""
    return _make_frame(
        trip_ids=["t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2"],
        trucks=["a"] * 4 + ["b"] * 4,
        y_true=[10, 20, 30, 40, 10, 20, 30, 40],
        y_pred=[11, 19, 31, 39, 12, 18, 32, 38],
        route_type=["highway", "highway", "city", "city",
                    "highway", "highway", "city", "city"],
        speed=[70, 70, 25, 25, 70, 70, 25, 25],
        grade=[0.0, 0.0, 1.8, 1.8, 0.0, 0.0, 1.8, 1.8],
        accel=[0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8],
        ignition=[1, 1, 1, 1, 1, 1, 1, 1],
    )


# --------------------------------------------------------------------------
# 1. Payload load-factor bins and invalid GVW handling
# --------------------------------------------------------------------------

def test_payload_load_factor_bin_boundaries():
    # Boundaries: light <= 0.33; medium (0.33, 0.66]; heavy > 0.66.
    payload = pd.Series([3000, 9900, 9901, 19800, 19801, 30000])
    gvw = pd.Series([30000, 30000, 30000, 30000, 30000, 30000])
    lf = payload_load_factor(payload, gvw)
    labels = payload_regime_label(lf).tolist()
    assert labels[0] == "payload_light_le_0.33"      # 0.10
    assert labels[1] == "payload_light_le_0.33"      # 0.33 exactly
    assert labels[2] == "payload_medium_gt0.33_le0.66"  # 0.330033... > 0.33
    assert labels[3] == "payload_medium_gt0.33_le0.66"  # 0.66 exactly
    assert labels[4] == "payload_heavy_gt0.66"      # 0.6600... > 0.66
    assert labels[5] == "payload_heavy_gt0.66"      # 1.0


def test_payload_load_factor_invalid_gvw_bucketed_not_dropped():
    payload = pd.Series([1000, 2000, 3000])
    gvw = pd.Series([0, np.nan, 30000])  # 0, NaN, valid
    lf = payload_load_factor(payload, gvw)
    labels = payload_regime_label(lf).tolist()
    assert labels[0] == "payload_invalid_gvw"
    assert labels[1] == "payload_invalid_gvw"
    assert labels[2] == "payload_light_le_0.33"  # 0.1


# --------------------------------------------------------------------------
# 2. Mask definitions and coverage accounting
# --------------------------------------------------------------------------

def test_extended_masks_idle_vs_engine_off_stopped():
    frame = _simple_frame()
    # idle: engine on AND speed < 0.5. None of the simple rows are stopped, so
    # add a dedicated stopped row.
    frame = pd.concat(
        [
            frame,
            _make_frame(
                ["t9"], ["c"], [5.0], [5.0],
                speed=[0.0], ignition=[1], route_type=["city"],
            ),
            _make_frame(
                ["t10"], ["c"], [5.0], [6.0],
                speed=[0.0], ignition=[0], route_type=["city"],
            ),
        ],
        ignore_index=True,
    )
    masks = make_extended_condition_masks(frame)
    idle = int(masks["idle_engine_on_stopped"].sum())
    off_stopped = int(masks["engine_off_stopped"].sum())
    assert idle == 1
    assert off_stopped == 1
    # Partition covers all rows with no overlap.
    not_ = int(masks["not_idle_not_engine_off_stopped"].sum())
    assert idle + off_stopped + not_ == len(frame)


def test_extended_masks_combined_conditions():
    frame = _simple_frame()
    masks = make_extended_condition_masks(frame)
    # highway+heavy payload (highway rows: speed 70, payload 10000/30000=0.33)
    assert int(masks["combined_highway_heavy_payload"].sum()) == 0  # not heavy
    # uphill+accelerating: grade>=0.3 & accel>=0.5
    uph_accel = int(masks["combined_uphill_accelerating"].sum())
    # rows with grade 1.8 & accel 0.8: indices 2 and 6
    assert uph_accel == 2
    # downhill+braking: grade<=-0.3 & accel<=-0.5 -> none in simple frame
    assert int(masks["combined_downhill_braking"].sum()) == 0
    # steep uphill + heavy payload: none (payload not heavy)
    assert int(masks["combined_steep_uphill_heavy_payload"].sum()) == 0


def test_masks_full_payload_coverage():
    # All rows must belong to exactly one payload regime mask.
    frame = _simple_frame()
    masks = make_extended_condition_masks(frame)
    labels = ["payload_light_le_0.33", "payload_medium_gt0.33_le0.66",
              "payload_heavy_gt0.66", "payload_invalid_gvw"]
    total = sum(int(masks[l].sum()) for l in labels)
    assert total == len(frame)


# --------------------------------------------------------------------------
# 3. Trip-metadata join integrity / row preservation / counts
# --------------------------------------------------------------------------

def test_attach_operating_columns_preserves_rows_and_index():
    telemetry = _simple_frame()  # has operating columns
    frame = _make_frame(
        ["t1", "t1", "t2", "t2"],
        ["a", "a", "b", "b"],
        [10, 20, 10, 20],
        [10, 20, 10, 20],
        route_type=["highway", "highway", "city", "city"],
    )
    # Use the same index as telemetry rows for alignment.
    frame.index = telemetry.index[:4]
    out = attach_operating_columns(frame, telemetry)
    assert len(out) == len(frame)
    assert out.index.equals(frame.index)
    for col in ("speed_kph", "road_grade_pct", "acceleration_mps2",
                "ignition_status", "payload_kg",
                "gross_vehicle_weight_kg"):
        assert col in out.columns
    assert out["speed_kph"].isna().sum() == 0


def test_attach_operating_columns_rejects_missing():
    telemetry = _simple_frame().drop(columns=["speed_kph"])
    frame = _simple_frame()
    with pytest.raises(ValueError, match="missing operating columns"):
        attach_operating_columns(frame, telemetry)


def test_route_type_table_counts():
    frame = _simple_frame()
    rows = route_type_table(frame)
    by = {r["route_type"]: r for r in rows}
    assert set(by) == {"highway", "city"}
    assert by["highway"]["n_rows"] == 4
    assert by["highway"]["n_trips"] == 2
    assert by["highway"]["n_trucks"] == 2
    assert by["city"]["n_rows"] == 4
    # highway MAE: |10-11|+|20-19|+|10-12|+|20-18| = 1+1+2+2 = 6 / 4 = 1.5
    assert by["highway"]["mae"] == pytest.approx(1.5)


# --------------------------------------------------------------------------
# 4. Insufficient-support flags
# --------------------------------------------------------------------------

def test_subgroup_metrics_insufficient_under_min_rows():
    frame = _simple_frame()
    mask = frame["truck_id"] == "a"  # 4 rows, 1 truck
    m = subgroup_metrics(frame, mask)
    assert m["insufficient_support"] is True  # 1 truck < 2
    # Justify: MIN_SUPPORT_TRUCKS = 2
    assert MIN_SUPPORT_TRUCKS == 2
    assert MIN_SUPPORT_ROWS == 500


def test_subgroup_metrics_medium_like_limited_coverage_not_insufficient():
    # Regression: a subgroup with enough rows and >= MIN_SUPPORT_TRUCKS trucks
    # but fewer than the full 5-truck test set is DESCRIPTIVE with limited
    # truck coverage — it must NOT be flagged insufficient/unusable.
    n = 600
    frame = _make_frame(
        trip_ids=[f"t{i}" for i in range(n)],
        trucks=["a", "b", "c"] * (n // 3),
        y_true=list(range(n)),
        y_pred=[v + 1 for v in range(n)],
    )
    mask = pd.Series(True, index=frame.index)
    m = subgroup_metrics(frame, mask)
    assert m["n_rows"] == n
    assert m["n_trucks"] == 3
    assert m["insufficient_support"] is False
    assert m["limited_truck_coverage"] is True
    assert m["support_status"] == "descriptive_limited_truck_coverage"


def test_subgroup_metrics_full_coverage_status():
    # 5 trucks, enough rows -> full coverage, no limited-coverage warning.
    n = 600
    frame = _make_frame(
        trip_ids=[f"t{i}" for i in range(n)],
        trucks=["a", "b", "c", "d", "e"] * (n // 5),
        y_true=list(range(n)),
        y_pred=[v + 1 for v in range(n)],
    )
    mask = pd.Series(True, index=frame.index)
    m = subgroup_metrics(frame, mask)
    assert m["n_trucks"] == 5
    assert m["insufficient_support"] is False
    assert m["limited_truck_coverage"] is False
    assert m["support_status"] == "descriptive_full_truck_coverage"


def test_subgroup_metrics_empty_status():
    # Empty subgroup -> truly insufficient, flagged appropriately.
    frame = _simple_frame()
    mask = pd.Series(False, index=frame.index)
    m = subgroup_metrics(frame, mask)
    assert m["n_rows"] == 0
    assert m["insufficient_support"] is True
    assert m["support_status"] == "empty"


def test_subgroup_metrics_supported_with_enough_rows_and_trucks():
    # Build a synthetic frame with 2 trucks and > MIN_SUPPORT_ROWS rows.
    n = 600
    frame = _make_frame(
        trip_ids=[f"t{i}" for i in range(n)],
        trucks=["a"] * (n // 2) + ["b"] * (n // 2),
        y_true=list(range(n)),
        y_pred=[v + 1 for v in range(n)],
    )
    mask = pd.Series(True, index=frame.index)
    m = subgroup_metrics(frame, mask)
    assert m["n_rows"] == n
    assert m["n_trucks"] == 2
    assert m["insufficient_support"] is False


# --------------------------------------------------------------------------
# 5. Error-concentration denominator/share reconciliation
# --------------------------------------------------------------------------

def test_error_concentration_shares_reconcile():
    frame = _simple_frame()
    masks = make_extended_condition_masks(frame)
    conc = error_concentration(frame, masks, INFORMATIVE_CONDITIONS)
    # Shares are fractions of total squared error; they must be <= 1 and
    # non-negative.
    for e in conc:
        assert 0.0 <= e["squared_error_share"] <= 1.0
        assert 0.0 <= e["row_share"] <= 1.0
    # The "all" rows represent the full denominator; recompute total to check.
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    manual_total = float(((y - p) ** 2).sum())
    for e in conc:
        manual_share = float(((y[masks[e["condition"]].to_numpy()] -
                              p[masks[e["condition"]].to_numpy()]) ** 2).sum())
        assert e["squared_error_share"] == pytest.approx(
            manual_share / manual_total
        )
        assert e["row_share"] == pytest.approx(
            int(masks[e["condition"]].sum()) / len(frame)
        )


def test_error_concentration_returns_mae_rmse():
    frame = _simple_frame()
    masks = make_extended_condition_masks(frame)
    conc = error_concentration(frame, masks, ["all"])
    assert conc[0]["mae"] == pytest.approx(
        float(np.abs(frame["y_true"] - frame["y_pred"]).mean())
    )
    assert conc[0]["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((frame["y_true"] - frame["y_pred"]) ** 2)))
    )


# --------------------------------------------------------------------------
# 6. Per-truck breakdown correctness
# --------------------------------------------------------------------------

def test_per_truck_condition_breakdown():
    frame = _simple_frame()
    masks = make_extended_condition_masks(frame)
    out = per_truck_condition(frame, masks, ["all"])
    assert "all" in out
    assert set(out["all"]["per_truck"]) == {"a", "b"}
    # truck a has 4 rows
    assert out["all"]["per_truck"]["a"]["n_rows"] == 4
    assert out["all"]["per_truck"]["b"]["n_rows"] == 4


# --------------------------------------------------------------------------
# 7. Integration: build_condition_report structure
# --------------------------------------------------------------------------

def test_build_condition_report_structure():
    frame = _simple_frame()
    report = build_condition_report(frame)
    assert report["n_rows"] == len(frame)
    assert report["n_trips"] == frame["trip_id"].nunique()
    assert report["n_trucks"] == frame["truck_id"].nunique()
    assert isinstance(report["route_type_table"], list)
    assert isinstance(report["conditions"], list)
    assert isinstance(report["error_concentration"], list)
    assert isinstance(report["per_truck_condition"], dict)
    # every condition entry reports coverage
    for c in report["conditions"]:
        assert {"n_rows", "n_trips", "n_trucks"} <= set(c)
    assert "payload_bin_edges" in report
    assert report["payload_bin_edges"] == [0.33, 0.66]
