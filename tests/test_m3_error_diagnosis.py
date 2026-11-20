"""Focused tests for the M3.3 cross-truck / cross-condition error diagnosis.

Covers the residual sign convention and bias percentages, error quantiles and
zero-median handling, truck-condition grouping and sample counts, trip
recurrence and worst-trip error contribution, paired model alignment and
delta sign convention, insufficient-support handling, and reconciliation
against independently calculated metrics — without loading the 963K-row
dataset or saved artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.m3_error_diagnosis import (
    APPROX_UNBIASED_TOL_LPH,
    DELTA_SIGN_CONVENTION,
    HIGH_ERROR_RATIO,
    RESIDUAL_CONVENTION,
    build_diagnosis_report,
    cell_support_status,
    error_distribution,
    paired_condition_comparison,
    residual_direction_table,
    signed_residual_stats,
    supported_cell_error_distribution,
    trip_recurrence,
    truck_condition_cells,
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


# --------------------------------------------------------------------------
# 1. Residual sign convention and bias percentages
# --------------------------------------------------------------------------

def test_residual_definition_positive_is_overprediction():
    # Convention: residual = y_pred - y_true. Positive => overprediction.
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 12.0, 8.0, 8.0])  # over by 2, under by 2
    s = signed_residual_stats(y_true, y_pred)
    assert s["mean_signed_error"] == pytest.approx(0.0)
    assert s["median_signed_error"] == pytest.approx(0.0)
    assert s["overprediction_pct"] == pytest.approx(50.0)
    assert s["underprediction_pct"] == pytest.approx(50.0)
    assert s["n"] == 4


def test_residual_all_overpredicted():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 22.0])  # both over
    s = signed_residual_stats(y_true, y_pred)
    assert s["overprediction_pct"] == pytest.approx(100.0)
    assert s["underprediction_pct"] == pytest.approx(0.0)
    assert s["mean_signed_error"] == pytest.approx(1.5)


def test_approx_unbiased_tolerance_documented():
    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([10.02, 10.03, 9.99])  # within 0.05
    s = signed_residual_stats(y_true, y_pred)
    assert s["approximately_unbiased_pct"] == pytest.approx(100.0)
    assert s["approx_unbiased_tolerance_lph"] == APPROX_UNBIASED_TOL_LPH


# --------------------------------------------------------------------------
# 2. Error quantiles and zero-median handling
# --------------------------------------------------------------------------

def test_error_distribution_quantiles():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 41.0, 49.0])  # abs err [1,1,1,1,1]
    d = error_distribution(y_true, y_pred)
    assert d["mae"] == pytest.approx(1.0)
    assert d["rmse"] == pytest.approx(1.0)
    assert d["median_absolute_error"] == pytest.approx(1.0)
    assert d["p90_absolute_error"] == pytest.approx(1.0)
    assert d["p95_absolute_error"] == pytest.approx(1.0)
    assert d["max_absolute_error"] == pytest.approx(1.0)
    assert d["p95_to_median_ratio"] == pytest.approx(1.0)
    assert d["zero_median"] is False


def test_error_distribution_zero_median_safe():
    # All errors zero -> median = 0; ratio must be None, not a divide-by-zero.
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    d = error_distribution(y_true, y_pred)
    assert d["median_absolute_error"] == 0.0
    assert d["p95_to_median_ratio"] is None
    assert d["zero_median"] is True


def test_error_distribution_ratio_reflects_tail():
    # Widespread small errors + a sizable set of big extremes -> high ratio.
    # 90 small errors (0.2) + 10 large errors (10): P95 falls in the tail.
    y_true = np.concatenate([np.full(90, 10.0), np.full(10, 10.0)])
    y_pred = np.concatenate([np.full(90, 10.2), np.full(10, 20.0)])
    d = error_distribution(y_true, y_pred)
    assert d["p95_to_median_ratio"] > 5.0
    assert d["median_absolute_error"] == pytest.approx(0.2)


# --------------------------------------------------------------------------
# 3. Truck-condition grouping and sample counts
# --------------------------------------------------------------------------

def test_truck_condition_cells_counts_and_support():
    frame = _make_frame(
        trip_ids=["t1"] * 4 + ["t2"] * 4,
        trucks=["a"] * 4 + ["b"] * 4,
        y_true=[10, 20, 30, 40, 10, 20, 30, 40],
        y_pred=[11, 21, 31, 41, 11, 21, 31, 41],
        speed=[70, 70, 70, 70, 5, 5, 5, 5],
        grade=[0.0] * 8,
        accel=[0.0] * 8,
        ignition=[1] * 8,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    labels = ["speed_ge_60_highway", "idle_engine_on_stopped"]
    cells = truck_condition_cells(frame, masks, labels)
    # truck a / highway -> 4 rows; truck b / highway -> 0 rows (empty cell kept)
    cell = next(c for c in cells if c["truck_id"] == "a" and c["condition"] == "speed_ge_60_highway")
    assert cell["n_rows"] == 4
    assert cell["n_trips"] == 1
    assert cell["n_trucks"] == 1
    # 4 rows < 500 -> insufficient_rows (row deficiency checked before trips)
    assert cell["support_status"] == "insufficient_rows"
    # empty cell preserved, not dropped
    empty = [c for c in cells if c["truck_id"] == "b" and c["condition"] == "speed_ge_60_highway"]
    assert len(empty) == 1
    assert empty[0]["support_status"] == "empty"
    assert empty[0]["n_rows"] == 0


def test_truck_condition_cells_mae_correct():
    frame = _make_frame(
        trip_ids=["t1"] * 4 + ["t2"] * 4,
        trucks=["a"] * 4 + ["b"] * 4,
        y_true=[10, 20, 30, 40, 10, 20, 30, 40],
        y_pred=[12, 22, 32, 42, 10, 20, 30, 40],  # truck a over by 2
        speed=[70] * 8,
        grade=[0.0] * 8,
        accel=[0.0] * 8,
        ignition=[1] * 8,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    cells = truck_condition_cells(frame, masks, ["speed_ge_60_highway"])
    cell = next(c for c in cells if c["truck_id"] == "a")
    assert cell["mae"] == pytest.approx(2.0)
    assert cell["rmse"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 4. Trip recurrence and worst-trip error contribution
# --------------------------------------------------------------------------

def test_trip_recurrence_recurring_vs_dominated():
    # Truck a: 3 trips of 200 rows, each with modest, even errors -> recurring.
    # Truck b: 3 trips of 200 rows, but trip 3 has huge errors -> dominated.
    n = 200
    a_true = np.full(3 * n, 10.0)
    a_pred = np.full(3 * n, 10.5)
    b_true = np.full(3 * n, 10.0)
    b_pred = np.concatenate(
        [np.full(n, 10.5), np.full(n, 10.5), np.full(n, 30.0)]
    )
    frame = _make_frame(
        trip_ids=[f"a{i}" for i in range(1, 4) for _ in range(n)] +
                 [f"b{i}" for i in range(1, 4) for _ in range(n)],
        trucks=["a"] * (3 * n) + ["b"] * (3 * n),
        y_true=a_true.tolist() + b_true.tolist(),
        y_pred=a_pred.tolist() + b_pred.tolist(),
        speed=[70] * (6 * n),
        grade=[0.0] * (6 * n),
        accel=[0.0] * (6 * n),
        ignition=[1] * (6 * n),
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    rec = trip_recurrence(frame, masks, ["speed_ge_60_highway"])
    rec_a = next(r for r in rec if r["truck_id"] == "a")
    rec_b = next(r for r in rec if r["truck_id"] == "b")
    assert rec_a["n_trips"] == 3
    assert rec_a["pattern"] == "recurring_pattern"
    assert rec_a["worst_trip_squared_error_share"] <= 0.5
    assert rec_b["n_trips"] == 3
    assert rec_b["pattern"] == "dominated_by_worst_trip"
    assert rec_b["worst_trip_squared_error_share"] > 0.5


def test_trip_recurrence_single_trip_insufficient():
    frame = _make_frame(
        trip_ids=["t1"] * 4,
        trucks=["a"] * 4,
        y_true=[10, 20, 30, 40],
        y_pred=[11, 21, 31, 41],
        speed=[70] * 4,
        grade=[0.0] * 4,
        accel=[0.0] * 4,
        ignition=[1] * 4,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    rec = trip_recurrence(frame, masks, ["speed_ge_60_highway"])
    assert rec[0]["pattern"] == "insufficient"
    assert rec[0]["support_status"] == "insufficient_rows"


# --------------------------------------------------------------------------
# 5. Paired model alignment and delta sign convention
# --------------------------------------------------------------------------

def test_paired_requires_aligned_frames():
    full = _make_frame(
        ["t1"] * 4, ["a"] * 4, [10, 20, 30, 40], [10, 20, 30, 40],
        index=pd.RangeIndex(4),
    )
    # Same length but a different index -> not aligned (must raise).
    tel = _make_frame(
        ["t1"] * 4, ["a"] * 4, [10, 20, 30, 40], [10, 20, 30, 40],
        index=pd.RangeIndex(4, 8),
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(full)
    with pytest.raises(ValueError):
        paired_condition_comparison(full, tel, masks, ["all"])


def test_paired_delta_sign_positive_when_full_better():
    full = _make_frame(
        ["t1"] * 4 + ["t2"] * 4,
        ["a"] * 4 + ["b"] * 4,
        [10, 20, 30, 40, 10, 20, 30, 40],
        [10, 20, 30, 40, 10, 20, 30, 40],  # perfect
        speed=[70] * 8,
        grade=[0.0] * 8,
        accel=[0.0] * 8,
        ignition=[1] * 8,
    )
    tel = full.copy()
    tel["y_pred"] = tel["y_pred"] + 1.0  # telematics worse by 1 everywhere
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(full)
    out = paired_condition_comparison(full, tel, masks, ["all"])
    pooled = out["pooled"]
    # delta = MAE_telematics - MAE_full = 1.0 - 0.0 = +1.0 (full better)
    assert pooled["delta_mae"] == pytest.approx(1.0)
    assert pooled["delta_rmse"] == pytest.approx(1.0)
    assert DELTA_SIGN_CONVENTION in out["delta_sign_convention"]
    assert out["delta_sign_convention"].startswith("delta = metric_telematics_only")


def test_paired_delta_sign_positive_means_full_lower_error():
    # Explicit: positive delta => full-operational has lower (better) error.
    full = _make_frame(["t1"] * 2, ["a"] * 2, [10, 20], [10, 20])
    tel = _make_frame(["t1"] * 2, ["a"] * 2, [10, 20], [12, 22])
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(full)
    out = paired_condition_comparison(full, tel, masks, ["all"])
    assert out["pooled"]["delta_mae"] == pytest.approx(2.0)  # tel - full = 2 - 0


# --------------------------------------------------------------------------
# 6. Insufficient-support handling
# --------------------------------------------------------------------------

def test_cell_support_status():
    assert cell_support_status(0, 0) == "empty"
    assert cell_support_status(10, 1) == "insufficient_rows"  # < 500 rows
    assert cell_support_status(600, 1) == "insufficient_trip_support"
    assert cell_support_status(600, 3) == "descriptive"


def test_supported_cell_error_distribution_only_descriptive():
    frame = _make_frame(
        trip_ids=["t1"] * 2 + ["t2"] * 2,
        trucks=["a"] * 4,
        y_true=[10, 20, 30, 40],
        y_pred=[11, 21, 31, 41],
        speed=[70] * 4,
        grade=[0.0] * 4,
        accel=[0.0] * 4,
        ignition=[1] * 4,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    out = supported_cell_error_distribution(frame, masks, ["speed_ge_60_highway"])
    # 4 rows < 500 -> not descriptive -> filtered out.
    assert out == []


# --------------------------------------------------------------------------
# 7. Reconciliation against independent metrics
# --------------------------------------------------------------------------

def test_reconciliation_against_independent_metrics():
    frame = _make_frame(
        trip_ids=["t1"] * 4 + ["t2"] * 4,
        trucks=["a"] * 4 + ["b"] * 4,
        y_true=[10, 20, 30, 40, 10, 20, 30, 40],
        y_pred=[11, 19, 31, 39, 12, 18, 32, 38],
        speed=[70, 70, 70, 70, 70, 70, 70, 70],
        grade=[0.0] * 8,
        accel=[0.0] * 8,
        ignition=[1] * 8,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    cells = truck_condition_cells(frame, masks, ["speed_ge_60_highway"])
    # Independently compute pooled highway MAE/RMSE and check the truck-level
    # baseline reconciliation.
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    manual_mae = float(np.abs(y - p).mean())
    manual_rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    pooled_cells = [c for c in cells if c["support_status"] != "empty"]
    # truck-column baselines should match per-truck pooled MAE.
    for truck in ["a", "b"]:
        mask = frame["truck_id"] == truck
        manual_truck_mae = float(np.abs(y[mask] - p[mask]).mean())
        cell = next(c for c in pooled_cells if c["truck_id"] == truck)
        assert cell["truck_baseline_mae"] == pytest.approx(manual_truck_mae)
    # The column baseline equals the manual pooled MAE (both trucks, all rows).
    first = pooled_cells[0]
    assert first["condition_baseline_mae"] == pytest.approx(manual_mae)


def test_residual_direction_table_includes_trucks_conditions_cells():
    frame = _make_frame(
        trip_ids=["t1"] * 4 + ["t2"] * 4,
        trucks=["a"] * 4 + ["b"] * 4,
        y_true=[10, 20, 30, 40, 10, 20, 30, 40],
        y_pred=[11, 21, 31, 41, 11, 21, 31, 41],
        speed=[70] * 8,
        grade=[0.0] * 8,
        accel=[0.0] * 8,
        ignition=[1] * 8,
    )
    from src.evaluation.m3_condition_robustness import make_extended_condition_masks
    masks = make_extended_condition_masks(frame)
    out = residual_direction_table(frame, masks, ["speed_ge_60_highway"])
    assert out["residual_convention"] == RESIDUAL_CONVENTION
    assert {r["truck_id"] for r in out["per_truck"]} == {"a", "b"}
    # per-cell only includes descriptive cells; 8 rows/2 trips but only 1
    # truck, < 500 rows -> none descriptive.
    assert out["per_cell"] == []
