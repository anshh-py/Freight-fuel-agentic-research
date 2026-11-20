"""Focused tests for the M3.4 telemetry-robustness analysis.

Covers numeric-feature selection (categorical/binary preservation), the
scenario catalog, determinism and row alignment of corruption, every
corruption mechanism (missingness, complete sensor failure, noise, bias,
spikes, clipping, combined), clean-vs-corrupted metrics and MAE degradation,
and prediction-failure accounting — without loading the 963K-row dataset or
saved artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.m3_telemetry_robustness import (
    CLIP_BOUNDS,
    MISSINGNESS_FRACTIONS,
    PRESERVED_BINARY,
    PRESERVED_CATEGORICAL,
    SEED_BASE,
    SPIKE_EXTREME_MULTIPLIER,
    SPIKE_FRACTION,
    apply_corruption,
    build_scenarios,
    evaluate_corrupted,
    numeric_features,
    regression_metrics_eval,
)

FULL_FEATURES = [
    "vehicle_class",
    "gross_vehicle_weight_kg",
    "payload_kg",
    "vehicle_mass_kg",
    "speed_kph",
    "acceleration_mps2",
    "road_grade_pct",
    "ignition_status",
    "engine_rpm",
    "engine_load_pct",
    "throttle_position_pct",
    "torque_nm",
    "coolant_temp_c",
    "intake_air_temp_c",
    "ambient_temp_c",
]

TELEMATICS_FEATURES = [
    "vehicle_class",
    "gross_vehicle_weight_kg",
    "payload_kg",
    "vehicle_mass_kg",
    "speed_kph",
    "acceleration_mps2",
    "road_grade_pct",
    "ignition_status",
    "coolant_temp_c",
    "intake_air_temp_c",
    "ambient_temp_c",
]

NUMERIC_COLS = [c for c in FULL_FEATURES if c not in ("vehicle_class", "ignition_status")]


def _make_raw(n=100, features=FULL_FEATURES):
    rng = np.random.default_rng(0)
    data = {}
    for c in features:
        if c == "vehicle_class":
            data[c] = ["40T", "32T", "49T"] * (n // 3 + 1)
            data[c] = data[c][:n]
        elif c == "ignition_status":
            data[c] = rng.integers(0, 2, size=n)
        elif c in ("gross_vehicle_weight_kg", "vehicle_mass_kg"):
            data[c] = rng.uniform(18000, 40000, size=n)
        elif c == "payload_kg":
            data[c] = rng.uniform(0, 20000, size=n)
        elif c == "speed_kph":
            data[c] = rng.uniform(0, 86, size=n)
        elif c == "acceleration_mps2":
            data[c] = rng.uniform(-1.6, 0.85, size=n)
        elif c == "road_grade_pct":
            data[c] = rng.uniform(-2.9, 2.9, size=n)
        elif c == "engine_rpm":
            data[c] = rng.uniform(600, 3500, size=n)
        elif c in ("engine_load_pct", "throttle_position_pct"):
            data[c] = rng.uniform(0, 100, size=n)
        elif c == "torque_nm":
            data[c] = rng.uniform(0, 3000, size=n)
        elif c in ("coolant_temp_c", "intake_air_temp_c", "ambient_temp_c"):
            data[c] = rng.uniform(-10, 100, size=n)
        else:
            data[c] = rng.uniform(0, 100, size=n)
    return pd.DataFrame(data, index=pd.RangeIndex(1000, 1000 + n))


# --------------------------------------------------------------------------
# 1. Numeric feature selection (categorical / binary preserved)
# --------------------------------------------------------------------------

def test_numeric_features_excludes_categorical_and_binary():
    num = numeric_features(FULL_FEATURES)
    assert "vehicle_class" not in num
    assert "ignition_status" not in num
    for c in ("speed_kph", "engine_rpm", "coolant_temp_c"):
        assert c in num
    assert set(num) == set(NUMERIC_COLS)


# --------------------------------------------------------------------------
# 2. Scenario catalog
# --------------------------------------------------------------------------

def test_build_scenarios_full_covers_all_kinds():
    sc = build_scenarios(FULL_FEATURES)
    kinds = {s["kind"] for s in sc}
    assert kinds == {
        "random_missingness",
        "complete_sensor_failure",
        "gaussian_noise",
        "fixed_bias",
        "spikes",
        "clipping",
        "combined",
    }
    missing = sorted(s["fraction"] for s in sc if s["kind"] == "random_missingness")
    assert missing == sorted(MISSINGNESS_FRACTIONS)
    # every scenario has a documented name
    for s in sc:
        assert s["name"]
        assert s["columns"]


def test_build_scenarios_telematics_omits_engine_sensors():
    sc = build_scenarios(TELEMATICS_FEATURES)
    affected = {c for s in sc for c in s["columns"]}
    assert "engine_rpm" not in affected
    assert "throttle_position_pct" not in affected
    assert "torque_nm" not in affected
    assert "speed_kph" in affected
    assert "coolant_temp_c" in affected


# --------------------------------------------------------------------------
# 3. Determinism and row alignment
# --------------------------------------------------------------------------

def test_corruption_is_deterministic_and_preserves_alignment():
    X = _make_raw()
    scen = {"name": "t", "kind": "random_missingness",
            "fraction": 0.2, "columns": NUMERIC_COLS, "seed": 123}
    A1, aff1, cnt1 = apply_corruption(X, scen)
    A2, aff2, cnt2 = apply_corruption(X, scen)
    pd.testing.assert_frame_equal(A1, A2)
    assert A1.index.equals(X.index)
    assert A1.columns.equals(X.columns)
    assert len(A1) == len(X)
    assert aff1 == NUMERIC_COLS
    assert cnt1 == cnt2


def test_corruption_preserves_categorical_and_binary_columns():
    X = _make_raw()
    scen = {"name": "t", "kind": "random_missingness",
            "fraction": 0.3, "columns": NUMERIC_COLS, "seed": 9}
    A, _, _ = apply_corruption(X, scen)
    pd.testing.assert_series_equal(A["vehicle_class"], X["vehicle_class"])
    pd.testing.assert_series_equal(A["ignition_status"], X["ignition_status"])
    # every numeric column has some missingness introduced
    for c in NUMERIC_COLS:
        assert A[c].isna().sum() > 0


def test_corruption_does_not_modify_source():
    X = _make_raw()
    src_copy = X.copy()
    scen = {"name": "t", "kind": "random_missingness",
            "fraction": 0.2, "columns": NUMERIC_COLS, "seed": 5}
    apply_corruption(X, scen)
    pd.testing.assert_frame_equal(X, src_copy)


# --------------------------------------------------------------------------
# 4. Individual corruption mechanisms
# --------------------------------------------------------------------------

def test_random_missingness_fraction_respected():
    X = _make_raw(n=200)
    scen = {"name": "m", "kind": "random_missingness",
            "fraction": 0.2, "columns": ["speed_kph"], "seed": 1}
    A, _, cnt = apply_corruption(X, scen)
    assert cnt == 40
    assert int(A["speed_kph"].isna().sum()) == 40


def test_complete_sensor_failure_all_nan():
    X = _make_raw()
    scen = {"name": "f", "kind": "complete_sensor_failure",
            "columns": ["engine_rpm"]}
    A, aff, cnt = apply_corruption(X, scen)
    assert aff == ["engine_rpm"]
    assert A["engine_rpm"].isna().all()
    assert cnt == len(X)
    # other columns untouched
    pd.testing.assert_series_equal(A["speed_kph"], X["speed_kph"])


def test_gaussian_noise_is_feature_scaled_and_additive():
    X = _make_raw()
    X["speed_kph"] = np.full(len(X), 50.0)  # constant -> sigma=0, not noisy
    X["coolant_temp_c"] = np.linspace(20, 80, len(X))  # varying feature
    scen = {"name": "n", "kind": "gaussian_noise", "severity": 0.5,
            "columns": ["speed_kph", "coolant_temp_c"], "seed": 4}
    # Training-only statistics (what the runner derives from the train split).
    stats = {
        "speed_kph": {"std": 0.0, "median": 50.0},
        "coolant_temp_c": {
            "std": float(X["coolant_temp_c"].std()),
            "median": float(X["coolant_temp_c"].median()),
        },
    }
    A, _, _ = apply_corruption(X, scen, stats)
    # constant column unaffected (sigma=0); varying column changed
    pd.testing.assert_series_equal(A["speed_kph"], X["speed_kph"])
    assert not A["coolant_temp_c"].equals(X["coolant_temp_c"])
    # empirical noise std should be near 0.5 * clean std = 0.5 * ~17.5
    noise_std = float((A["coolant_temp_c"] - X["coolant_temp_c"]).std())
    assert abs(noise_std - 0.5 * float(X["coolant_temp_c"].std())) < 2.0


def test_fixed_bias_shifts_by_documented_amount():
    X = _make_raw()
    scen = {"name": "b", "kind": "fixed_bias", "bias": 200.0,
            "columns": ["engine_rpm"]}
    A, _, _ = apply_corruption(X, scen)
    pd.testing.assert_series_equal(A["engine_rpm"], X["engine_rpm"] + 200.0)


def test_isolated_extreme_spikes_exceed_normal_range():
    X = _make_raw()
    scen = {"name": "s", "kind": "spikes", "spike_fraction": SPIKE_FRACTION,
            "spike_multiplier": SPIKE_EXTREME_MULTIPLIER,
            "columns": ["speed_kph"], "seed": 3}
    stats = {"speed_kph": {
        "std": float(X["speed_kph"].std()),
        "median": float(X["speed_kph"].median()),
    }}
    A, _, cnt = apply_corruption(X, scen, stats)
    expected = int(round(SPIKE_FRACTION * len(X)))
    assert cnt == expected
    med = float(X["speed_kph"].median())
    std = float(X["speed_kph"].std())
    spike_val = med + SPIKE_EXTREME_MULTIPLIER * std
    assert int((A["speed_kph"] == pytest.approx(spike_val)).sum()) == expected
    # spikes are extreme: beyond the 99.9th percentile of clean data
    assert spike_val > float(X["speed_kph"].quantile(0.999))


def test_clipping_applies_documented_bounds():
    X = _make_raw(n=200)
    X["speed_kph"] = np.concatenate([np.full(100, -20.0), np.full(100, 500.0)])
    lo, hi = CLIP_BOUNDS["speed_kph"]
    scen = {"name": "c", "kind": "clipping",
            "columns": ["speed_kph"]}
    A, _, cnt = apply_corruption(X, scen)
    assert A["speed_kph"].min() >= lo
    assert A["speed_kph"].max() <= hi
    assert cnt == 200  # all 200 clipped to bounds
    # column not in scenario unchanged
    pd.testing.assert_series_equal(A["coolant_temp_c"], X["coolant_temp_c"])


def test_combined_missingness_plus_noise():
    X = _make_raw()
    scen = {"name": "comb", "kind": "combined", "fraction": 0.2,
            "severity": 0.25, "columns": NUMERIC_COLS, "seed": SEED_BASE + 301}
    stats = {c: {"std": 1.0, "median": 0.0} for c in NUMERIC_COLS}
    A, _, cnt = apply_corruption(X, scen, stats)
    assert cnt > len(X)  # missingness cells + noise cells
    assert A[NUMERIC_COLS[0]].isna().sum() > 0


# --------------------------------------------------------------------------
# 5. Metrics
# --------------------------------------------------------------------------

def test_regression_metrics_eval_basic():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.5, 20.5, 30.5])
    m = regression_metrics_eval(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)
    assert m["n_rows"] == 3
    # R² < 1 when predictions are offset; perfect fit gives R² = 1
    m_perfect = regression_metrics_eval(y_true, y_true)
    assert m_perfect["r2"] == pytest.approx(1.0)
    m_off = regression_metrics_eval(y_true, y_pred)
    assert m_off["r2"] == pytest.approx(0.99625, abs=1e-4)


def test_regression_metrics_eval_empty_and_constant():
    m = regression_metrics_eval(np.array([]), np.array([]))
    assert m["n_rows"] == 0
    assert m["mae"] is None
    m2 = regression_metrics_eval(np.array([1.0, 1.0, 1.0]),
                                 np.array([1.0, 1.0, 1.0]))
    assert m2["r2"] is None
    assert m2["r2_flag"] == "constant_target"


# --------------------------------------------------------------------------
# 6. Prediction-failure accounting
# --------------------------------------------------------------------------

class _IdentityPreprocessor:
    _fitted = True

    def transform(self, X):
        return X


class _NanAwareEstimator:
    """Predicts 10.0; any numeric row with a NaN feature -> NaN (a failure)."""

    def predict(self, X):
        num = X.select_dtypes(include=[np.number])
        arr = num.to_numpy(dtype=float)
        out = np.full(len(X), 10.0)
        bad = np.isnan(arr).any(axis=1)
        out[bad] = np.nan
        return out


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_evaluate_corrupted_counts_nonfinite_predictions_as_failures(bad_value):
    """NaN, +inf and -inf predictions are all counted as failures."""
    n = 100
    X = _make_raw(n=n)
    y_true = np.linspace(0, 20, n)
    trucks = np.array(["a"] * (n // 2) + ["b"] * (n // 2))

    clean_pred = np.full(n, 10.0)
    clean_metrics = regression_metrics_eval(y_true, clean_pred)
    clean_per_truck = {"a": {"n_rows": 50, "mae": 0.0},
                       "b": {"n_rows": 50, "mae": 0.0}}

    class _NonFiniteEstimator:
        def predict(self, X):
            out = np.full(len(X), 10.0)
            out[:20] = bad_value
            return out

    res = evaluate_corrupted(
        y_true, trucks, _IdentityPreprocessor(), _NonFiniteEstimator(),
        X, clean_metrics, clean_per_truck,
    )
    assert res["prediction_aborted"] is False
    assert res["n_prediction_failures"] == 20
    assert res["prediction_failure_rate"] == pytest.approx(0.2)
    # metrics computed only over the finite surviving rows (rows 20..99)
    assert res["n_rows_evaluated"] == 80
    expected_mae = float(np.abs(y_true[20:] - 10.0).mean())
    assert res["mae"] == pytest.approx(expected_mae)
    # per-truck table reports degradation vs clean baseline and row counts
    assert res["per_truck"]["a"]["n_rows"] == 30  # 50 rows, 20 failed
    assert res["per_truck"]["b"]["n_rows"] == 50
    for d in res["per_truck"].values():
        assert d["clean_mae"] == 0.0
        assert d["delta_mae"] == pytest.approx(d["mae"])


def test_evaluate_corrupted_no_failures_when_clean():
    n = 50
    X = _make_raw(n=n)
    y_true = np.linspace(0, 10, n)
    trucks = np.array(["a"] * n)
    clean_metrics = regression_metrics_eval(y_true, np.full(n, 5.0))
    clean_per_truck = {"a": {"n_rows": n, "mae": 0.0}}
    res = evaluate_corrupted(
        y_true, trucks, _IdentityPreprocessor(), _NanAwareEstimator(), X,
        clean_metrics, clean_per_truck,
    )
    assert res["prediction_aborted"] is False
    assert res["n_prediction_failures"] == 0
    assert res["prediction_failure_rate"] == 0.0
    assert res["n_rows_evaluated"] == n

def test_evaluate_corrupted_aborted_prediction_reported():
    class _BoomEstimator:
        def predict(self, X):
            raise ValueError("cannot predict")

    n = 20
    X = _make_raw(n=n)
    y_true = np.zeros(n)
    trucks = np.array(["a"] * n)
    clean_metrics = regression_metrics_eval(y_true, np.zeros(n))
    clean_per_truck = {"a": {"n_rows": n, "mae": 0.0}}

    res = evaluate_corrupted(
        y_true, trucks, _IdentityPreprocessor(), _BoomEstimator(), X,
        clean_metrics, clean_per_truck,
    )
    assert res["prediction_aborted"] is True
    assert res["n_prediction_failures"] == n
    assert res["prediction_failure_rate"] == 1.0
    assert res["mae"] is None
    assert "prediction_error" in res


def test_shared_columns_receive_identical_corruption_across_feature_sets():
    """A shared column gets identical corruption in both feature sets.

    Regression for the paired-corruption audit finding: corruption must be
    identical for the same scenario/column/rows regardless of which feature
    set it belongs to (per-column deterministic seeding).
    """
    X_full = _make_raw(n=200, features=FULL_FEATURES)
    X_tele = X_full[TELEMATICS_FEATURES]
    shared = sorted(set(NUMERIC_COLS) & set(TELEMATICS_FEATURES))

    for kind in ("gaussian_noise", "spikes"):
        if kind == "gaussian_noise":
            scen = {"name": "n", "kind": "gaussian_noise",
                    "severity": 0.25, "columns": shared,
                    "seed": SEED_BASE + 101}
        else:
            scen = {"name": "s", "kind": "spikes",
                    "spike_fraction": SPIKE_FRACTION,
                    "spike_multiplier": SPIKE_EXTREME_MULTIPLIER,
                    "columns": shared, "seed": SEED_BASE + 201}
        stats = {c: {"std": 1.0, "median": 0.0} for c in shared}

        Af, _, _ = apply_corruption(X_full, scen, stats)
        At, _, _ = apply_corruption(X_tele, scen, stats)
        for c in shared:
            pd.testing.assert_series_equal(Af[c], At[c])
