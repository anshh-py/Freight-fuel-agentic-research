"""Focused tests for M1.3 final test evaluation logic.

Covers the helpers that are unit-testable without running the full pipeline
(which requires the 963K-row dataset and the saved M1.2 artifacts):

  * validation-based model selection uses validation evidence only,
  * operating-condition masks are explicit and cover every record (nothing is
    silently discarded),
  * per-truck / per-class aggregation and the descriptive macro-average,
  * residual sign semantics,
  * early rejection of unknown models / feature sets before any evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.test_evaluation import (
    EXPECTED_TEST_TRUCKS,
    evaluate_test_set,
    macro_average,
    make_condition_masks,
    per_class_metrics,
    per_truck_metrics,
    residual_stats,
    select_model_from_validation,
)


# --------------------------------------------------------------------------
# 1. Model selection uses validation evidence only
# --------------------------------------------------------------------------

def _minimal_metrics_report() -> dict:
    return {
        "feature_sets": {
            "full_operational": {
                "models": {
                    "ridge": {"metrics": {"mae": 5.2}},
                    "xgboost": {"metrics": {"mae": 3.6}},
                    "random_forest": {"metrics": {"mae": 3.8}},
                }
            },
            "telematics_only": {
                "models": {
                    "ridge": {"metrics": {"mae": 19.2}},
                    "xgboost": {"metrics": {"mae": 4.3}},
                }
            },
        }
    }


def test_select_model_from_validation_picks_min_mae():
    sel = select_model_from_validation(_minimal_metrics_report())
    assert sel["full_operational"]["model"] == "xgboost"
    assert sel["telematics_only"]["model"] == "xgboost"
    assert sel["full_operational"]["validation_mae"] == 3.6


def test_selection_basis_is_validation_evidence():
    sel = select_model_from_validation(_minimal_metrics_report())
    for fs in sel:
        assert "validation" in sel[fs]["basis"].lower()
        assert sel[fs]["basis"] == "validation MAE (M1.2) — test data not used"


# --------------------------------------------------------------------------
# 2. Operating-condition masks: explicit and exhaustive
# --------------------------------------------------------------------------

def _condition_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "truck_id": ["tuck_004"] * 6,
            "vehicle_class": ["32T", "32T", "40T", "40T", "49T", "49T"],
            "ignition_status": [1, 1, 0, 1, 1, 1],
            "speed_kph": [0.0, 10.0, 25.0, 45.0, 70.0, 85.0],
            "road_grade_pct": [-2.0, -1.0, 0.0, 0.5, 1.6, 2.5],
            "acceleration_mps2": [-1.2, -0.2, 0.1, 0.6, 0.4, -0.6],
        }
    )


def test_condition_masks_cover_all_rows():
    df = _condition_df()
    masks = make_condition_masks(df)
    n = len(df)
    for prefix, span in [
        ("speed_", 5),   # five speed bands
        ("grade_", 5),   # five grade bands
        ("accel_", 3),   # three accel bands
    ]:
        subset = [m for k, m in masks.items() if k.startswith(prefix)]
        total = sum(m.sum() for m in subset)
        assert total == n, (prefix, total, n)
    # engine on/off partition
    assert (masks["engine_on"].sum() + masks["engine_off"].sum()) == n
    # all rows are in "all"
    assert masks["all"].all()
    # every class has a mask
    for cls in df["vehicle_class"].unique():
        assert f"class_{cls}" in masks


def test_condition_masks_engine_state_correct():
    df = _condition_df()
    masks = make_condition_masks(df)
    assert masks["engine_on"].tolist() == [True, True, False, True, True, True]
    assert masks["engine_off"].tolist() == [False, False, True, False, False, False]


# --------------------------------------------------------------------------
# 3. Per-truck / per-class aggregation and descriptive macro-average
# --------------------------------------------------------------------------

def test_per_truck_metrics_returns_each_truck():
    df = pd.DataFrame(
        {
            "truck_id": ["t1", "t1", "t2", "t2", "t1", "t2"],
            "vehicle_class": ["32T", "40T", "32T", "49T", "49T", "40T"],
        }
    )
    y = pd.Series([10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    pred = pd.Series([9.0, 9.0, 19.0, 19.0, 29.0, 29.0])
    per = per_truck_metrics(df, y, pred)
    assert set(per) == {"t1", "t2"}
    assert per["t1"]["mae"] == pytest.approx(1.0)
    perc = per_class_metrics(df, y, pred)
    assert set(perc) == {"32T", "40T", "49T"}


def test_macro_average_is_descriptive_mean():
    per = {
        "t1": {"mae": 1.0, "rmse": 1.2, "r2": 0.9, "mape": 10.0},
        "t2": {"mae": 3.0, "rmse": 3.2, "r2": 0.7, "mape": 20.0},
    }
    macro = macro_average(per)
    assert macro["mae"] == pytest.approx(2.0)
    assert macro["r2"] == pytest.approx(0.8)
    assert "not robust" in macro["basis"]


# --------------------------------------------------------------------------
# 4. Residual sign semantics
# --------------------------------------------------------------------------

def test_residual_stats_positive_is_underpredicted():
    y = pd.Series([10.0, 20.0, 30.0])
    pred = pd.Series([8.0, 18.0, 28.0])  # always underpredicted
    r = residual_stats(y, pred)
    assert r["mean_residual"] == pytest.approx(2.0)
    assert r["fraction_underpredicted"] == 1.0
    assert r["fraction_overpredicted"] == 0.0


# --------------------------------------------------------------------------
# 5. Early validation before any evaluation
# --------------------------------------------------------------------------

class _FakeSplit:
    train_trucks = ["truck_100", "truck_101"]
    val_trucks = ["truck_102", "truck_103", "truck_104", "truck_105"]
    test_trucks = list(EXPECTED_TEST_TRUCKS)


def test_evaluate_test_set_rejects_unknown_model(monkeypatch):
    import src.evaluation.test_evaluation as te
    monkeypatch.setattr(te, "enforce_split", lambda *a, **k: _FakeSplit())

    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="unknown model"):
        evaluate_test_set(telemetry=df, manifest=df, models=["bogus"])


def test_evaluate_test_set_rejects_unknown_feature_set(monkeypatch):
    import src.evaluation.test_evaluation as te
    monkeypatch.setattr(te, "enforce_split", lambda *a, **k: _FakeSplit())

    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="unknown feature set"):
        evaluate_test_set(telemetry=df, manifest=df, feature_sets=["nope"])


def test_evaluate_test_set_rejects_train_val_leak(monkeypatch):
    import src.evaluation.test_evaluation as te

    class LeakySplit(_FakeSplit):
        test_trucks = ["truck_100"] + list(EXPECTED_TEST_TRUCKS)

    monkeypatch.setattr(te, "enforce_split", lambda *a, **k: LeakySplit())
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="non-test truck"):
        evaluate_test_set(telemetry=df, manifest=df)
