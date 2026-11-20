"""M3.4 — missing / noisy / corrupted telemetry robustness (frozen test split).

This module measures how the saved M1.2 models degrade when the *model-input
feature matrix* is corrupted, reusing the frozen five-truck test split and the
saved M1.2 artifacts (preprocessors + validation-selected XGBoost models)
read-only.

Scope (per the milestone requirements):

  * Synthetic corruption is applied to COPIES of the final model-input feature
    matrices, after feature construction and before preprocessing. Raw
    telemetry and raw feature columns are never modified; no feature column is
    removed (sensor failure is simulated with missing values while preserving
    the model's expected schema).
  * Corruption is initially restricted to numeric model-input features.
    Categorical (``vehicle_class``) and binary (``ignition_status``) columns
    are preserved untouched.
  * Deterministic experiments: random missingness (5/10/20/30%), complete
    numeric sensor-column failure (all NaN), feature-scaled Gaussian noise,
    fixed sensor bias, isolated extreme spikes, clipping with documented
    bounds, and one combined missingness+noise scenario.
  * Deterministic seeds and preserved row alignment throughout.
  * Metrics: MAE, RMSE, R², MAE degradation relative to the corresponding
    clean baseline, prediction failure count/rate, and per-truck degradation.
  * Prediction failures are counted (never silently discarded) and the
    per-truck results reconcile with the pooled results.

Protocol guarantees (mirroring M1.3 / M3.1–3.3):
  * models/preprocessors are loaded from disk and never refitted;
  * the test set is evaluation-only — never used for fitting/tuning/selection;
  * no source, split manifest, M1/M2/M3.1/M3.2 artifact is modified;
  * synthetic-corruption results are kept explicitly distinct from real-world
    sensor validation (there is no observed sensor-failure data here);
  * all metrics are descriptive — no hypothesis tests or significance claims.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.data.loader import enforce_split, load_split_manifest, load_telemetry
from src.evaluation.m3_cross_truck_analysis import (
    DEFAULT_REPORT_DIR,
    EXPECTED_TEST_TRUCKS,
    FEATURE_SETS,
    safe_r2,
)
from src.evaluation.test_evaluation import select_model_from_validation
from src.features.build import build_feature_matrix
from src.features.spec import load_feature_spec

# --------------------------------------------------------------------------
# Documented corruption conventions
# --------------------------------------------------------------------------

#: Deterministic seed base shared by all corruptions. Each scenario adds a
#: fixed, documented offset so every run is reproducible.
SEED_BASE = 20240617

#: Extreme-spike magnitude, expressed as a multiple of the feature's standard
#: deviation added to its median (documented in the report).
SPIKE_EXTREME_MULTIPLIER = 8.0

#: Columns that are NEVER corrupted: the categorical vehicle class and the
#: binary ignition status. Corruption is restricted to numeric model-input
#: features (per the M3.4 restriction).
PRESERVED_CATEGORICAL = ("vehicle_class",)
PRESERVED_BINARY = ("ignition_status",)
PRESERVED_COLUMNS = frozenset((*PRESERVED_CATEGORICAL, *PRESERVED_BINARY))

#: Explicit physical clipping bounds per feature (documented). Rows already
#: inside the bounds are untouched; only out-of-bound values are clamped, so
#: ``clip_count`` records how many cells actually changed.
CLIP_BOUNDS = {
    "gross_vehicle_weight_kg": (15000.0, 60000.0),
    "payload_kg": (0.0, 40000.0),
    "vehicle_mass_kg": (15000.0, 60000.0),
    "speed_kph": (0.0, 90.0),
    "acceleration_mps2": (-5.0, 5.0),
    "road_grade_pct": (-10.0, 10.0),
    "engine_rpm": (0.0, 5000.0),
    "engine_load_pct": (0.0, 100.0),
    "throttle_position_pct": (0.0, 100.0),
    "torque_nm": (0.0, 4000.0),
    "coolant_temp_c": (0.0, 120.0),
    "intake_air_temp_c": (-40.0, 100.0),
    "ambient_temp_c": (-40.0, 60.0),
}

#: Documented, feature-scaled Gaussian-noise severities (fraction of each
#: feature's clean standard deviation used as the additive-noise sigma).
GAUSSIAN_NOISE_SEVERITIES = (0.25, 0.50)

#: Random-missingness fractions evaluated (documented).
MISSINGNESS_FRACTIONS = (0.05, 0.10, 0.20, 0.30)

#: Spike fraction (share of rows per affected column set to an extreme value).
SPIKE_FRACTION = 0.005


def numeric_features(features) -> list[str]:
    """The numeric model-input features (categorical/binary preserved)."""
    return [f for f in features if f not in PRESERVED_COLUMNS]


# --------------------------------------------------------------------------
# Scenario catalog (built per feature set, applied to raw feature matrices)
# --------------------------------------------------------------------------

def build_scenarios(features) -> list[dict]:
    """Return the concrete M3.4 corruption scenarios for a feature set.

    ``features`` is the ordered feature list of one feature set (from the
    spec). Scenarios whose target columns do not exist in this feature set
    (e.g. engine sensors in ``telematics_only``) are omitted. Every scenario
    records its config, affected columns, a documented seed, and the expected
    corruption mechanism so the JSON report is fully self-describing.
    """
    numeric = numeric_features(features)
    if not numeric:
        raise ValueError("feature set has no numeric model-input features to corrupt")

    def present(candidates) -> list[str]:
        return [c for c in candidates if c in numeric]

    all_numeric = list(numeric)

    scenarios: list[dict] = []

    # Random missingness (all numeric features).
    for i, frac in enumerate(MISSINGNESS_FRACTIONS):
        pct = int(round(frac * 100))
        scenarios.append(
            {
                "name": f"random_missingness_{pct:02d}pct",
                "kind": "random_missingness",
                "description": (
                    f"{frac:.2f} of cells set to NaN per affected numeric "
                    "feature, selected uniformly at random per column"
                ),
                "fraction": frac,
                "columns": all_numeric,
                "seed": SEED_BASE + 1 + i,
            }
        )

    # Complete numeric sensor-column failure (all values NaN).
    for col, j in (("engine_rpm", 0), ("throttle_position_pct", 1),
                   ("speed_kph", 2), ("coolant_temp_c", 3)):
        if col in numeric:
            scenarios.append(
                {
                    "name": f"complete_sensor_failure_{col}",
                    "kind": "complete_sensor_failure",
                    "description": (
                        f"entire column '{col}' set to NaN (simulated complete "
                        "sensor failure); column is kept so the schema is "
                        "preserved"
                    ),
                    "columns": [col],
                    "seed": None,
                }
            )

    # Feature-scaled Gaussian noise.
    for j, sev in enumerate(GAUSSIAN_NOISE_SEVERITIES):
        scenarios.append(
            {
                "name": f"gaussian_noise_severity_{int(sev*100):02d}pct",
                "kind": "gaussian_noise",
                "description": (
                    f"additive Gaussian noise with sigma = {sev:.2f} x each "
                    "affected feature's clean standard deviation (feature-"
                    "scaled severity)"
                ),
                "severity": sev,
                "columns": all_numeric,
                "seed": SEED_BASE + 101 + j,
            }
        )

    # Fixed sensor bias (physical units, documented).
    for col, bias, j in (
        ("engine_rpm", 200.0, 0),
        ("ambient_temp_c", 5.0, 1),
        ("speed_kph", 10.0, 2),
    ):
        if col in numeric:
            scenarios.append(
                {
                    "name": f"fixed_bias_{col}_plus{bias:g}",
                    "kind": "fixed_bias",
                    "description": (
                        f"fixed additive bias of +{bias:g} applied to column "
                        f"'{col}' (physical units)"
                    ),
                    "bias": bias,
                    "columns": [col],
                    "seed": None,
                }
            )

    # Isolated extreme spikes.
    for col, j in (("engine_load_pct", 0), ("speed_kph", 1),
                   ("torque_nm", 2)):
        if col in numeric:
            scenarios.append(
                {
                    "name": f"isolated_extreme_spikes_{col}",
                    "kind": "spikes",
                    "description": (
                        f"{SPIKE_FRACTION:.3f} of rows in column '{col}' set "
                        "to an extreme value = median + "
                        f"{SPIKE_EXTREME_MULTIPLIER:g} x std (an isolated "
                        "6+ sigma-style spike)"
                    ),
                    "spike_fraction": SPIKE_FRACTION,
                    "spike_multiplier": SPIKE_EXTREME_MULTIPLIER,
                    "columns": [col],
                    "seed": SEED_BASE + 201 + j,
                }
            )

    # Clipping with explicit documented bounds.
    clip_cols = present(sorted(CLIP_BOUNDS))
    if clip_cols:
        scenarios.append(
            {
                "name": "clipping_physical_bounds",
                "kind": "clipping",
                "description": (
                    "clip affected numeric features to their documented "
                    "physical bounds (see CLIP_BOUNDS)"
                ),
                "columns": clip_cols,
                "seed": None,
            }
        )

    # Combined missingness + noise.
    scenarios.append(
        {
            "name": "combined_missingness20pct_noise0.25",
            "kind": "combined",
            "description": (
                "combined scenario: random missingness 20% + Gaussian noise "
                "severity 0.25 on the same numeric features"
            ),
            "fraction": 0.20,
            "severity": 0.25,
            "columns": all_numeric,
            "seed": SEED_BASE + 301,
        }
    )

    return scenarios


# --------------------------------------------------------------------------
# Corruption primitives (each returns a fresh copy of the raw feature matrix)
# --------------------------------------------------------------------------

def _rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        raise ValueError("scenario requires a seed for randomization")
    return np.random.default_rng(int(seed))


def _per_column_seed(seed: int, column: str) -> int:
    """Deterministic per-column seed.

    Combines the scenario seed with a stable (crc32) hash of the column name,
    so that every column draws from its own independent RNG stream. Because the
    derivation depends only on the scenario seed + column name, a given column
    receives *identical* corruption across feature sets that share it (same
    scenario, same row indices). The crc32 hash is stable across processes
    (unlike builtin ``hash``).
    """
    return int(seed) + zlib.crc32(str(column).encode("utf-8"))


def _corrupt_missingness(X, cols, fraction, seed) -> int:
    n = len(X)
    changed = 0
    for c in cols:
        rng = _rng(_per_column_seed(seed, c))
        k = int(round(fraction * n))
        if k <= 0:
            continue
        idx = rng.choice(n, size=k, replace=False)
        X.loc[X.index[idx], c] = np.nan
        changed += k
    return changed


def _corrupt_noise(X, cols, severity, seed, statistics) -> int:
    """Additive Gaussian noise, sigma = severity x *training-derived* std.

    ``statistics`` maps column -> {"std": ..., "median": ...} derived
    exclusively from the training split (never validation/test). Each column
    uses an independent deterministic RNG via ``_per_column_seed``.
    """
    n = len(X)
    for c in cols:
        st = statistics.get(c)
        if st is None:
            continue
        rng = _rng(_per_column_seed(seed, c))
        sigma = severity * st["std"]
        col = pd.to_numeric(X[c], errors="coerce")
        X[c] = col.to_numpy() + rng.normal(0.0, sigma, size=n)
    return n * len(cols)


def _corrupt_spikes(X, cols, fraction, multiplier, seed, statistics) -> int:
    """Set a fraction of rows to median + multiplier x (training-derived) std.

    Reference statistics (median, std) come from the training split only.
    Each column uses an independent deterministic RNG.
    """
    n = len(X)
    changed = 0
    for c in cols:
        st = statistics.get(c)
        if st is None:
            continue
        rng = _rng(_per_column_seed(seed, c))
        k = int(round(fraction * n))
        if k <= 0:
            continue
        idx = rng.choice(n, size=k, replace=False)
        X.loc[X.index[idx], c] = st["median"] + multiplier * st["std"]
        changed += k
    return changed


def _corrupt_clipping(X, cols) -> int:
    changed = 0
    for c in cols:
        lo, hi = CLIP_BOUNDS[c]
        vals = pd.to_numeric(X[c], errors="coerce")
        clamped = vals.clip(lower=lo, upper=hi)
        changed += int((clamped.to_numpy() != vals.to_numpy()).sum())
        X[c] = clamped
    return changed


def apply_corruption(
    X_raw: pd.DataFrame,
    scenario: dict,
    statistics: Optional[dict] = None,
) -> tuple[pd.DataFrame, list[str], int]:
    """Return ``(corrupted_copy, affected_columns, cells_changed)``.

    Operates on a *copy* of the raw model-input feature matrix (after feature
    construction, before preprocessing). Row indices and column order are
    preserved exactly; categorical and binary columns are never touched. The
    original ``X_raw`` and the underlying telemetry are never modified.

    ``statistics`` (training-only per-feature std/median) is required for the
    Gaussian-noise, spike and combined scenarios; it is never derived from the
    validation/test matrix being corrupted.
    """
    cols = [c for c in scenario.get("columns", []) if c in X_raw.columns]
    cols = [c for c in cols if c not in PRESERVED_COLUMNS]
    if not cols:
        return X_raw.copy(), [], 0

    out = X_raw.copy()
    kind = scenario["kind"]

    if kind == "random_missingness":
        changed = _corrupt_missingness(
            out, cols, float(scenario["fraction"]), scenario.get("seed")
        )
    elif kind == "complete_sensor_failure":
        out[cols] = np.nan
        changed = len(out) * len(cols)
    elif kind == "gaussian_noise":
        if statistics is None:
            raise ValueError("gaussian_noise requires training statistics")
        changed = _corrupt_noise(
            out, cols, float(scenario["severity"]), scenario.get("seed"),
            statistics,
        )
    elif kind == "fixed_bias":
        bias = float(scenario["bias"])
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce") + bias
        changed = len(out) * len(cols)
    elif kind == "spikes":
        if statistics is None:
            raise ValueError("spikes requires training statistics")
        changed = _corrupt_spikes(
            out,
            cols,
            float(scenario["spike_fraction"]),
            float(scenario["spike_multiplier"]),
            scenario.get("seed"),
            statistics,
        )
    elif kind == "clipping":
        changed = _corrupt_clipping(out, cols)
    elif kind == "combined":
        changed_miss = _corrupt_missingness(
            out, cols, float(scenario["fraction"]), scenario.get("seed")
        )
        if statistics is None:
            raise ValueError("combined requires training statistics")
        changed_noise = _corrupt_noise(
            out, cols, float(scenario["severity"]),
            int(scenario.get("seed")) + 7, statistics,
        )
        changed = changed_miss + changed_noise
    else:
        raise ValueError(f"unknown corruption kind: {kind!r}")

    return out, cols, changed


# --------------------------------------------------------------------------
# Predictions with explicit failure accounting
# --------------------------------------------------------------------------

def predict_safe(
    preprocessor,
    estimator,
    X_input: pd.DataFrame,
) -> tuple[Optional[pd.Series], dict]:
    """Transform + predict; never silently drop a failure.

    Returns ``(pred_series, info)``. If transformation or prediction raises
    (e.g. a corruption the preprocessor or estimator cannot consume),
    ``info["aborted"]`` is True and ``pred_series`` is None. Otherwise
    ``pred_series`` is a float Series index-aligned with ``X_input`` (it may
    contain NaN values, which callers count as prediction failures rather than
    discarding silently). The preprocessor is only ever ``transform``-ed —
    never refitted.
    """
    try:
        Xt = preprocessor.transform(X_input)
        if len(Xt) != len(X_input):
            raise RuntimeError("preprocessor changed row count")
        pred = pd.Series(estimator.predict(Xt), index=X_input.index, dtype=float)
    except Exception as exc:  # noqa: BLE001 - captured & reported explicitly
        return None, {"aborted": True, "error": str(exc)[:300]}
    return pred, {"aborted": False}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def regression_metrics_eval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """MAE / RMSE / R² with explicit degenerate handling.

    ``y_true``/``y_pred`` may be empty; R² is None (with a reason flag) when
    undefined (fewer than two rows, constant target or constant prediction).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = int(len(yt))
    if n == 0:
        return {"n_rows": 0, "mae": None, "rmse": None, "r2": None,
                "r2_flag": "empty"}
    mae = float(np.abs(yt - yp).mean())
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2_val, r2_flag = safe_r2(yt, yp)
    return {
        "n_rows": n,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": r2_val,
        "r2_flag": r2_flag,
    }


def pool_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    return regression_metrics_eval(y_true, y_pred)


def per_truck_mae(
    y_true: np.ndarray,
    trucks: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict]:
    """Per-truck MAE (and row counts) used for per-truck degradation."""
    df = pd.DataFrame(
        {"truck_id": trucks, "y_true": y_true, "y_pred": y_pred}
    )
    out: dict[str, dict] = {}
    for truck, g in df.groupby("truck_id", sort=True):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        out[str(truck)] = {
            "n_rows": int(len(yt)),
            "mae": round(float(np.abs(yt - yp).mean()), 6),
        }
    return out


# --------------------------------------------------------------------------
# Per-scenario evaluation
# --------------------------------------------------------------------------

def evaluate_corrupted(
    y_true: np.ndarray,
    trucks: np.ndarray,
    preprocessor,
    estimator,
    X_corr: pd.DataFrame,
    clean_metrics: dict,
    clean_per_truck: dict,
) -> dict:
    """Evaluate one corrupted matrix vs the clean baseline.

    Prediction failures (aborted predictions or non-finite NaN/±inf
    predictions) are counted and reported — never silently discarded. Metrics
    are computed on the finite non-failed rows, and the failure count/rate is
    reported alongside so they are read together.
    """
    n_total = int(len(y_true))
    pred, info = predict_safe(preprocessor, estimator, X_corr)

    if info["aborted"]:
        return {
            "prediction_aborted": True,
            "prediction_error": info["error"],
            "n_prediction_failures": n_total,
            "prediction_failure_rate": 1.0,
            "n_rows_evaluated": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "delta_mae": None,
            "mae_degradation_ratio": None,
            "per_truck": {},
        }

    # NaN, +inf and -inf are all reported as prediction failures.
    finite = np.isfinite(pred.to_numpy(dtype=float))
    n_fail = int((~finite).sum())
    rate = n_fail / n_total if n_total else 0.0

    yt = y_true[finite]
    pp = pred.to_numpy(dtype=float)[finite]
    tr = trucks[finite]

    metrics = regression_metrics_eval(yt, pp)
    clean_mae = clean_metrics["mae"]

    if clean_mae is not None and np.isfinite(clean_mae) and clean_mae > 0:
        ratio = (metrics["mae"] / clean_mae) if metrics["mae"] is not None else None
        delta = (metrics["mae"] - clean_mae) if metrics["mae"] is not None else None
    else:
        ratio, delta = None, None

    per_truck = per_truck_mae(yt, tr, pp)
    per_truck_degradation = {}
    for truck, d in per_truck.items():
        base = clean_per_truck.get(truck, {})
        base_mae = base.get("mae")
        per_truck_degradation[truck] = {
            "n_rows": d["n_rows"],
            "mae": d["mae"],
            "clean_mae": base_mae,
            "delta_mae": (
                round(d["mae"] - base_mae, 6)
                if base_mae is not None else None
            ),
        }

    return {
        "prediction_aborted": False,
        "n_prediction_failures": n_fail,
        "prediction_failure_rate": round(rate, 6),
        "n_rows_evaluated": metrics["n_rows"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "delta_mae": round(delta, 6) if delta is not None else None,
        "mae_degradation_ratio": (
            round(ratio, 6) if ratio is not None else None
        ),
        "per_truck": per_truck_degradation,
    }


# --------------------------------------------------------------------------
# Training-only corruption statistics
# --------------------------------------------------------------------------

def _derive_training_statistics(train_df, feature_set, spec) -> dict:
    """Per-feature training-only statistics (std, median) for corruption.

    Used to scale Gaussian-noise sigma and spike magnitudes. Derived
    exclusively from the training split of the frozen manifest — never from
    validation or test data.
    """
    X_train, _, features = build_feature_matrix(train_df, feature_set, spec=spec)
    stats: dict = {}
    for c in numeric_features(features):
        col = pd.to_numeric(X_train[c], errors="coerce")
        finite = col.dropna().to_numpy(dtype=float)
        if len(finite) == 0:
            continue
        s = float(np.std(finite))
        m = float(np.median(finite))
        if np.isfinite(s) and s > 0:
            stats[c] = {"std": s, "median": m}
    return stats


# --------------------------------------------------------------------------
# Orchestration (read-only)
# --------------------------------------------------------------------------

def load_test_resources(
    report_dir: Path = DEFAULT_REPORT_DIR,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> dict:
    """Load the frozen test split and saved artifacts; verify the split.

    Mirrors ``M3.1.prepare_test_predictions`` but also exposes the raw
    model-input feature matrices (before preprocessing) so corruption can be
    applied to copies. Nothing is trained or refitted.
    """
    if not report_dir.exists():
        raise FileNotFoundError(f"report dir not found: {report_dir}")

    telemetry = load_telemetry()
    manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)

    test_trucks = sorted(split.test_trucks)
    if set(test_trucks) != set(EXPECTED_TEST_TRUCKS):
        raise ValueError(
            "test truck set mismatch for M3.4: "
            f"manifest={test_trucks} expected={sorted(EXPECTED_TEST_TRUCKS)}"
        )

    with open(report_dir / "metrics_report.json", "r", encoding="utf-8") as fh:
        metrics_report = json.load(fh)
    selected = select_model_from_validation(metrics_report)

    spec = load_feature_spec()
    resources: dict = {}

    for fs in feature_sets:
        artifacts_dir = report_dir / "artifacts" / fs
        pre_path = artifacts_dir / "preprocessor.joblib"
        if not pre_path.exists():
            raise FileNotFoundError(f"preprocessor artifact missing: {pre_path}")
        preprocessor = joblib.load(pre_path)
        if not getattr(preprocessor, "_fitted", False):
            raise RuntimeError(f"preprocessor {pre_path} is not fitted")

        model_name = selected[fs]["model"]
        model_path = artifacts_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"model artifact missing: {model_path}")
        estimator = joblib.load(model_path)

        X_raw, y, features = build_feature_matrix(split.test, fs, spec=spec)

        # Corruption magnitudes (noise sigma, spike refs) come from the
        # training split only — never from validation/test data.
        train_stats = _derive_training_statistics(split.train, fs, spec=spec)

        resources[fs] = {
            "preprocessor": preprocessor,
            "estimator": estimator,
            "model_name": model_name,
            "X_raw": X_raw,
            "y_true": y.astype(float).to_numpy(),
            "trucks": split.test["truck_id"].to_numpy(),
            "trips": split.test["trip_id"].to_numpy(),
            "features": features,
            "numeric_features": numeric_features(features),
            "train_stats": train_stats,
        }

    return {
        "resources": resources,
        "selected_models": selected,
        "test_rows": int(len(split.test)),
        "test_trucks": test_trucks,
        "n_test_trips": int(split.test["trip_id"].nunique()),
    }


def run_telemetry_robustness(
    report_dir: Path = DEFAULT_REPORT_DIR,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> dict:
    """Run all M3.4 corruption experiments and build the report body."""
    prep = load_test_resources(report_dir, feature_sets)
    resources = prep["resources"]

    feature_sets_out: dict = {}
    clean_baselines: dict = {}
    reconciliation: dict = {}

    for fs in feature_sets:
        r = resources[fs]
        y_true = r["y_true"]
        trucks = r["trucks"]
        preprocessor = r["preprocessor"]
        estimator = r["estimator"]

        # Clean baseline (uncorrupted).
        p_clean, info_clean = predict_safe(preprocessor, estimator, r["X_raw"])
        if info_clean["aborted"]:
            raise RuntimeError(
                f"clean baseline prediction failed for {fs}: "
                f"{info_clean['error']}"
            )
        if not np.all(np.isfinite(p_clean.to_numpy(dtype=float))):
            raise RuntimeError(
                f"clean baseline produced non-finite predictions for {fs}"
            )
        p_clean_np = p_clean.to_numpy()
        clean_metrics = regression_metrics_eval(y_true, p_clean_np)
        clean_per_truck = per_truck_mae(y_true, trucks, p_clean_np)
        clean_baselines[fs] = {
            "model": r["model_name"],
            "n_rows": int(len(y_true)),
            **clean_metrics,
        }

        # Reconciliation: per-truck MAE weighted by n_rows must equal pooled MAE.
        n_total = int(len(y_true))
        w_mae = (
            sum(float(d["mae"]) * int(d["n_rows"]) for d in clean_per_truck.values())
            / n_total
        )
        reconciliation[fs] = {
            "pooled_mae": clean_metrics["mae"],
            "per_truck_weighted_mae": round(w_mae, 6),
            "max_abs_diff": round(abs(w_mae - clean_metrics["mae"]), 9),
            "reconciles": bool(abs(w_mae - clean_metrics["mae"]) < 1e-6),
        }

        scenarios = build_scenarios(r["features"])
        scenario_results = []
        for scenario in scenarios:
            X_corr, affected, changed = apply_corruption(
                r["X_raw"], scenario, r["train_stats"]
            )
            if not affected:
                scenario_results.append(
                    {
                        **scenario,
                        "applied": False,
                        "reason": (
                            "no numeric target column present in this feature set"
                        ),
                    }
                )
                continue

            result = evaluate_corrupted(
                y_true, trucks, preprocessor, estimator, X_corr,
                clean_metrics, clean_per_truck,
            )
            scenario_results.append(
                {
                    **scenario,
                    "applied": True,
                    "affected_columns": affected,
                    "cells_changed": changed,
                    "n_rows_target": int(len(r["X_raw"])),
                    **result,
                }
            )

        feature_sets_out[fs] = {
            "n_rows": int(len(y_true)),
            "model": r["model_name"],
            "features": r["features"],
            "numeric_features": r["numeric_features"],
            "clean_baseline": clean_baselines[fs],
            "scenarios": scenario_results,
        }

    return {
        "experiment": "M3.4_missing_noisy_corrupted_telemetry_robustness",
        "milestone": "M3.4",
        "protocol": (
            "read-only: reuse frozen 5-truck test split and saved M1.2 "
            "artifacts; corruption applied to copies of the final model-input "
            "feature matrices after feature construction and before "
            "preprocessing; no training/refitting/tuning"
        ),
        "test_rows": prep["test_rows"],
        "test_trucks": prep["test_trucks"],
        "n_test_trips": prep["n_test_trips"],
        "selected_models_from_validation": prep["selected_models"],
        "clean_baselines": clean_baselines,
        "feature_sets": feature_sets_out,
        "per_truck_reconciliation": reconciliation,
        "corruption_conventions": {
            "seed_base": SEED_BASE,
            "preserved_categorical": list(PRESERVED_CATEGORICAL),
            "preserved_binary": list(PRESERVED_BINARY),
            "spike_multiplier": SPIKE_EXTREME_MULTIPLIER,
            "spike_fraction": SPIKE_FRACTION,
            "gaussian_noise_severities": list(GAUSSIAN_NOISE_SEVERITIES),
            "missingness_fractions": list(MISSINGNESS_FRACTIONS),
            "clip_bounds": CLIP_BOUNDS,
            "training_statistics_provenance": {
                "source": "training_split_only",
                "description": (
                    "Gaussian-noise sigma and spike reference statistics "
                    "(median, std) are derived exclusively from the training "
                    "split of the frozen split manifest; never from "
                    "validation or test data. Severity multipliers, bias "
                    "magnitudes and clipping bounds are predefined documented "
                    "constants."
                ),
            },
            "random_stream": {
                "kind": "per_column_deterministic",
                "description": (
                    "Each affected column uses an independent RNG seeded "
                    "deterministically from the scenario seed plus a stable "
                    "hash of the column name; shared columns receive identical "
                    "corruption across feature sets (same scenario, same row "
                    "indices)."
                ),
            },
            "combined_scenario": {
                "note": (
                    "The combined missingness+noise scenario uses an "
                    "independent realization from the standalone 20% "
                    "missingness and 0.25-noise scenarios (separate seeds), "
                    "not a superset of them."
                ),
            },
        },
    }
