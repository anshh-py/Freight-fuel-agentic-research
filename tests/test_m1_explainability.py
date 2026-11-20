"""Focused tests for M1.4 explainability logic.

Covers the unit-testable helpers without depending on the 963K-row dataset or
the saved M1.2 artifacts (those are exercised by running the real pipeline):

  * readable feature-name mapping is total and labels one-hot columns,
  * deterministic sampling keeps all trucks and a fixed per-truck size,
  * SHAP reconstruction validation (pass on exact, fail on shifted),
  * per-truck aggregation accounts for each sampled record,
  * individual-exclamation record selection is deterministic and uses the
    documented residual sign convention (actual − predicted).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.explainability import (
    FEATURE_SETS,
    load_selected_xgb,
    make_explanation_sample,
    map_readable_names,
    mean_abs_shap,
    select_explanation_records,
    top_shap_contributions,
    validate_reconstruction,
)


# --------------------------------------------------------------------------
# 1. Readable feature-name mapping
# --------------------------------------------------------------------------

def test_map_readable_names_is_total_and_labels_onehot():
    cols = ["speed_kph", "vehicle_class_40T", "vehicle_class_49T", "raw_col"]
    mapping = map_readable_names(cols)
    assert set(mapping) == set(cols)
    assert mapping["speed_kph"] == "Speed (kph)"
    assert mapping["vehicle_class_40T"] == "Vehicle class (40T)"
    # unknown columns map to themselves (never dropped)
    assert mapping["raw_col"] == "raw_col"


# --------------------------------------------------------------------------
# 2. Deterministic sampling
# --------------------------------------------------------------------------

def _truck_df():
    trucks = ["truck_001", "truck_004", "truck_017", "truck_020", "truck_028"]
    frames = []
    for i, t in enumerate(trucks):
        frames.append(pd.DataFrame(
            {"truck_id": [t] * 20,
             "vehicle_class": [["32T", "40T", "49T"][i % 3]] * 20,
             "speed_kph": np.arange(20.0) + i}
        ))
    return pd.concat(frames, ignore_index=True)


def test_deterministic_sample_all_trucks_fixed_count():
    df = _truck_df()
    s1 = make_explanation_sample(df, sample_per_truck=10, seed=42)
    s2 = make_explanation_sample(df, sample_per_truck=10, seed=42)
    # deterministic
    pd.testing.assert_frame_equal(s1, s2)
    # all five trucks represented, exactly 10 each
    counts = s1.groupby("truck_id").size().to_dict()
    assert set(counts) == {"truck_001", "truck_004", "truck_017",
                            "truck_020", "truck_028"}
    assert all(v == 10 for v in counts.values())
    assert len(s1) == 50
    # sampling is a subset of the original rows (index preserved)
    assert set(s1.index) <= set(df.index)


def test_sample_caps_at_truck_size():
    df = _truck_df()  # 20 rows per truck
    s = make_explanation_sample(df, sample_per_truck=1000, seed=1)
    assert len(s) == len(df)


# --------------------------------------------------------------------------
# 3. SHAP reconstruction validation
# --------------------------------------------------------------------------

def test_validate_reconstruction_passes_on_exact():
    ev = 0.5
    shap_values = np.array([[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]])
    preds = ev + shap_values.sum(axis=1)
    res = validate_reconstruction(ev, shap_values, preds)
    assert res["pass"] is True
    assert res["fraction_within_tolerance"] == pytest.approx(1.0)
    assert res["max_abs_deviation"] < 1e-12


def test_validate_reconstruction_fails_on_shift():
    ev = 0.5
    shap_values = np.array([[0.2, -0.1, 0.4], [0.3, 0.1, -0.2]])
    preds = ev + shap_values.sum(axis=1) + 10.0
    res = validate_reconstruction(ev, shap_values, preds)
    assert res["pass"] is False


# --------------------------------------------------------------------------
# 4. Mean-abs-importance ranking
# --------------------------------------------------------------------------

def test_mean_abs_shap_ranks_descending():
    columns = ["a", "b", "c"]
    readable = {"a": "A", "b": "B", "c": "C"}
    # feature b has the largest mean abs
    shap_values = np.array([
        [1.0, 5.0, 2.0],
        [1.0, 5.0, 2.0],
    ])
    imp = mean_abs_shap(shap_values, columns, readable)
    assert imp["column"].tolist() == ["b", "c", "a"]
    assert imp.iloc[0]["readable_label"] == "B"
    assert imp["mean_abs_shap"].iloc[0] == pytest.approx(5.0)


def test_top_shap_contributions_split_signs():
    columns = ["a", "b", "c", "d"]
    readable = {c: c.upper() for c in columns}
    shap_values = np.array([[-2.0, 1.0, -0.5, 3.0]])
    pos, neg = top_shap_contributions(shap_values, columns, readable, 0, k=2)
    assert [p["column"] for p in pos] == ["d", "b"]
    assert [n["column"] for n in neg] == ["a", "c"]
    assert all(p["shap"] > 0 for p in pos)
    assert all(n["shap"] < 0 for n in neg)


# --------------------------------------------------------------------------
# 5. Individual explanation record selection (deterministic + residual sign)
# --------------------------------------------------------------------------

def test_select_explanation_records_rules_and_sign():
    sample = pd.DataFrame(
        {"truck_id": ["t1"] * 10, "vehicle_class": ["32T"] * 10}
    )
    y_true = pd.Series(np.arange(10.0, 20.0), index=sample.index)
    y_pred = pd.Series(np.arange(9.0, 19.0), index=sample.index)  # underpred
    recs = select_explanation_records(sample, y_true, y_pred, per_truck=3)
    rules = {r["rule"] for r in recs}
    assert rules == {"largest_abs_residual", "median_abs_residual",
                     "closest_to_zero_residual"}
    for r in recs:
        assert r["residual"] == pytest.approx(
            y_true.loc[r["index"]] - y_pred.loc[r["index"]]
        )
        # since pred < actual, residual is positive (underpredicted)
        assert r["residual"] > 0


def test_select_explanation_records_spans_error_spectrum():
    sample = pd.DataFrame(
        {"truck_id": ["t1"] * 6, "vehicle_class": ["40T"] * 6}
    )
    y_true = pd.Series([10.0] * 6, index=sample.index)
    y_pred = pd.Series([9.0, 9.0, 9.0, 9.0, 9.0, 9.0], index=sample.index)
    recs = select_explanation_records(sample, y_true, y_pred, per_truck=3)
    assert len(recs) == 3  # all three rules picked from the single truck


# --------------------------------------------------------------------------
# 6. Loading selected models (unknown set / missing artifacts)
# --------------------------------------------------------------------------

def test_load_selected_xgb_rejects_unknown_set():
    with pytest.raises(ValueError, match="unknown feature set"):
        load_selected_xgb(feature_sets=["nope"])


def test_load_selected_xgb_missing_artifact():
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        # empty report dir => artifact missing (model checked first)
        with pytest.raises(FileNotFoundError, match="artifact missing"):
            load_selected_xgb(report_dir=tmp, feature_sets=["full_operational"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
