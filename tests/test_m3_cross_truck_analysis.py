"""Focused tests for the M3.1 cross-truck generalization analysis.

Covers the pure, unit-testable diagnostics (trip grouping, paired alignment,
bootstrap determinism, delta sign, tail quantiles, degenerate-trip handling,
variance-decomposition accounting) without loading the 963K-row dataset or
the saved artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.m3_cross_truck_analysis import (
    _assert_aligned,
    paired_grouped_delta,
    per_trip_metrics,
    per_truck_tail,
    pooled_paired_delta,
    safe_r2,
    tail_quantiles,
    trip_clustered_bootstrap,
    variance_decomposition,
)


def _make_frame(
    trip_ids,
    trucks,
    y_true,
    y_pred,
    route="highway",
    vehicle_class="40T",
) -> pd.DataFrame:
    n = len(y_true)
    return pd.DataFrame(
        {
            "trip_id": trip_ids,
            "truck_id": trucks,
            "vehicle_class": [vehicle_class] * n,
            "route_type": [route] * n,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        },
        index=pd.RangeIndex(n),
    )


# --------------------------------------------------------------------------
# 1. Per-trip grouping
# --------------------------------------------------------------------------

def test_per_trip_metrics_groups_and_includes_metadata():
    frame = _make_frame(
        trip_ids=["t1", "t1", "t2", "t2", "t1", "t2"],
        trucks=["trk1", "trk1", "trk2", "trk2", "trk1", "trk2"],
        y_true=[10, 20, 10, 20, 30, 40],
        y_pred=[9, 21, 8, 22, 29, 39],
    )
    rows = per_trip_metrics(frame)
    by = {r["trip_id"]: r for r in rows}
    assert set(by) == {"t1", "t2"}
    assert by["t1"]["n"] == 3
    assert by["t1"]["truck_id"] == "trk1"
    assert by["t1"]["route_type"] == "highway"
    assert by["t1"]["vehicle_class"] == "40T"
    # MAE: |10-9|+|20-21|+|30-29| = 3 / 3 = 1.0
    # RMSE: errors 1, -1, 1 -> MSE = (1+1+1)/3 = 1 -> RMSE = 1.0
    assert by["t1"]["mae"] == pytest.approx(1.0)
    assert by["t1"]["rmse"] == pytest.approx(1.0)
    assert by["t1"]["r2"] is not None and by["t1"]["r2_flag"] == "ok"


# --------------------------------------------------------------------------
# 2. Paired alignment
# --------------------------------------------------------------------------

def test_assert_aligned_rejects_misaligned_frames():
    full = _make_frame(["t1"], ["a"], [1.0], [1.0])
    tel = _make_frame(["t1"], ["a"], [1.0], [2.0])
    tel.index = pd.RangeIndex(1, 2)
    with pytest.raises(ValueError, match="index-aligned"):
        _assert_aligned(full, tel)


def test_paired_delta_sign_positive_when_full_better():
    # full predictions are correct; telematics overpredicts -> delta > 0
    full = _make_frame(
        ["t1"] * 4, ["a"] * 4, [10, 20, 30, 40], [10, 20, 30, 40]
    )
    tel = _make_frame(
        ["t1"] * 4, ["a"] * 4, [10, 20, 30, 40], [12, 22, 32, 42]
    )
    d = pooled_paired_delta(full, tel)
    assert d["delta_mae"] == pytest.approx(2.0)
    assert d["delta_mae"] > 0  # positive => full-operational lower error
    assert d["mae_full_operational"] == pytest.approx(0.0)
    assert d["mae_telematics_only"] == pytest.approx(2.0)


def test_paired_grouped_delta_reports_per_truck():
    full = _make_frame(
        ["t1", "t1", "t2", "t2"],
        ["a", "a", "b", "b"],
        [10, 20, 10, 20],
        [10, 20, 10, 20],
    )
    tel = _make_frame(
        ["t1", "t1", "t2", "t2"],
        ["a", "a", "b", "b"],
        [10, 20, 10, 20],
        [11, 21, 10.5, 20.5],
    )
    per = paired_grouped_delta(full, tel, group_col="truck_id")
    assert set(per) == {"a", "b"}
    assert per["a"]["delta_mae"] == pytest.approx(1.0)
    assert per["b"]["delta_mae"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 3. Bootstrap determinism and finiteness
# --------------------------------------------------------------------------

def _bootstrap_frames():
    rng = np.random.default_rng(0)
    trips, trucks, yt, pf, pt = [], [], [], [], []
    for i in range(10):
        n = rng.integers(5, 12)
        trips.extend([f"t{i}"] * n)
        trucks.extend([f"trk{i % 3}"] * n)
        yy = rng.normal(30, 8, n)
        yt.extend(yy)
        pf.extend(yy + rng.normal(0, 1, n))
        pt.extend(yy + rng.normal(0, 2, n))
    full = _make_frame(trips, trucks, yt, pf)
    tel = _make_frame(trips, trucks, yt, pt)
    return full, tel


def test_bootstrap_is_deterministic_and_finite():
    full, tel = _bootstrap_frames()
    b1 = trip_clustered_bootstrap(full, tel, n_resamples=200, seed=42)
    b2 = trip_clustered_bootstrap(full, tel, n_resamples=200, seed=42)
    for k in ("mae_full", "rmse_full", "mae_tel", "rmse_tel", "delta_mae"):
        assert b1[k] == b2[k]
        assert np.isfinite(b1[k]["ci95_lower"])
        assert np.isfinite(b1[k]["ci95_upper"])
        assert b1[k]["ci95_lower"] <= b1[k]["ci95_upper"]
    assert b1["seed"] == 42
    assert b1["resampling_unit"] == "trip_id"
    assert b1["n_trips"] == 10
    assert b1["conditional_on_five_test_trucks"] is True


def test_bootstrap_resamples_trips_not_rows():
    full, tel = _bootstrap_frames()
    b = trip_clustered_bootstrap(full, tel, n_resamples=50, seed=1)
    assert b["n_resamples_used"] == 50
    # n_trips is the number of trips, not rows
    assert b["n_trips"] == full["trip_id"].nunique()


# --------------------------------------------------------------------------
# 4. Tail quantiles
# --------------------------------------------------------------------------

def test_tail_quantiles_values():
    ae = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    t = tail_quantiles(ae)
    assert t["p90"] == pytest.approx(9.0)
    assert t["p95"] == pytest.approx(9.5)
    assert t["p99"] == pytest.approx(9.9)
    assert t["max"] == pytest.approx(10.0)


def test_per_truck_tail_includes_n_and_max():
    frame = _make_frame(
        ["t1"] * 5 + ["t2"] * 5,
        ["a"] * 5 + ["b"] * 5,
        [1, 2, 3, 4, 5] * 2,
        [2, 4, 6, 8, 10] * 2,
    )
    out = per_truck_tail(frame)
    assert set(out) == {"a", "b"}
    assert out["a"]["n"] == 5
    assert out["a"]["max"] == pytest.approx(5.0)
    # truck b abs errors = [1,2,3,4,5]; quantile(0.9) interpolates -> 4.6
    assert out["b"]["p90"] == pytest.approx(4.6)


# --------------------------------------------------------------------------
# 5. Degenerate-trip handling
# --------------------------------------------------------------------------

def test_safe_r2_constant_target_flagged():
    # constant target -> undefined R², must not silently emit a number
    y = np.array([5.0] * 4)
    p = np.array([4.0, 5.0, 6.0, 5.0])
    r2, flag = safe_r2(y, p)
    assert r2 is None
    assert flag == "constant_target"


def test_safe_r2_insufficient_rows_flagged():
    r2, flag = safe_r2(np.array([1.0]), np.array([1.0]))
    assert r2 is None
    assert flag == "insufficient_rows_lt_2"


def test_per_trip_metrics_carries_degenerate_flag():
    frame = _make_frame(
        ["t1"] * 4,
        ["a"] * 4,
        [5, 5, 5, 5],
        [4, 5, 6, 5],
        route="city",
    )
    rows = per_trip_metrics(frame)
    assert rows[0]["r2"] is None
    assert rows[0]["r2_flag"] == "constant_target"
    # MAE still defined and correct (abs errors 1,0,1,0 -> mean 0.5)
    assert rows[0]["mae"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 6. Variance decomposition accounting
# --------------------------------------------------------------------------

def test_variance_decomposition_partitions_total():
    frame = _make_frame(
        ["t1"] * 4 + ["t2"] * 4,
        ["a"] * 4 + ["b"] * 4,
        [10, 20, 30, 40, 10, 20, 30, 40],
        [12, 18, 32, 38, 14, 16, 34, 36],
    )
    vd = variance_decomposition(frame)
    shares = vd["share_of_squared_error"]
    total_share = shares["between_truck"] + shares["between_trip_within_truck"] \
        + shares["within_trip"]
    assert vd["n_trucks"] == 2
    assert vd["n_trips"] == 2
    assert vd["total_squared_error"] == pytest.approx(
        vd["between_truck_ss"]
        + vd["between_trip_within_truck_ss"]
        + vd["within_trip_ss"]
    )
    assert total_share == pytest.approx(1.0)
    assert "unstable" in vd["note"] or "descriptive" in vd["note"]
