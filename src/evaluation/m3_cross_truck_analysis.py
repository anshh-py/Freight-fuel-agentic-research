"""M3.1 — cross-truck generalization diagnostics for the frozen test split.

This module builds on the M1.3 final evaluation without duplicating it. It
reuses the *saved* M1.2 artifacts (preprocessors + validation-selected
XGBoost models) and the frozen five-truck test split, and adds genuinely new
diagnostics that M1.3 did not compute:

  * per-trip performance (MAE / RMSE, with explicit degenerate-trip handling),
  * tail-error analysis (absolute-error p90/p95/p99/max at trip and truck
    level),
  * a **paired** comparison of full-operational versus telematics-only
    predictions on exactly the same test rows, with an explicit delta sign,
  * trip-clustered bootstrap 95% intervals (resampling the 96 test trips,
    jointly for both models — not individual autocorrelated telemetry rows),
  * a descriptive variance decomposition of prediction error across truck /
    trip-within-truck / row-within-trip levels.

Protocol guarantees (mirroring M1.3):
  * models and preprocessors are loaded from disk and never refitted;
  * the test set is used for evaluation only — never for fitting, tuning or
    selection;
  * no dataset, split manifest, M1 artifact or frozen milestone is modified;
  * the analysis is explicitly *conditional on these five test trucks* — it
    does not establish population-level generalization across trucks;
  * trip-level inference resamples trips, not individual telemetry rows, to
    avoid treating repeated rows (autocorrelated within a trip) as
    independent observations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.data.loader import (
    enforce_split,
    load_split_manifest,
    load_telemetry,
)
from src.evaluation.test_evaluation import (
    EXPECTED_TEST_TRUCKS,
    select_model_from_validation,
)
from src.features.build import build_feature_matrix
from src.features.spec import load_feature_spec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"
TRIP_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_heavy_truck_research"
    / "trip_metadata.csv"
)

FEATURE_SETS = ("full_operational", "telematics_only")
TRAIL_QUANTILES = (0.90, 0.95, 0.99)
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 20240617
#: The test set has exactly 96 trips across the five frozen test trucks.
EXPECTED_TEST_TRIPS = 96


# --------------------------------------------------------------------------
# Trip metadata (route_type lives here, not in telemetry)
# --------------------------------------------------------------------------

def load_trip_metadata(path: Optional[Path] = None) -> pd.DataFrame:
    """Load trip-level metadata (route_type, truck, vehicle class counts)."""
    p = Path(path) if path is not None else TRIP_METADATA_PATH
    if not p.exists():
        raise FileNotFoundError(f"Trip metadata not found: {p}")
    return pd.read_csv(p)


def _route_type_map(telemetry: pd.DataFrame) -> dict[str, str]:
    """Map ``trip_id`` -> ``route_type`` from trip metadata.

    Telemetry does not carry route_type, so it is joined from the frozen
    trip_metadata.csv. Any test trip missing from the metadata raises (we
    never silently drop a trip).
    """
    tm = load_trip_metadata()
    missing_cols = {"trip_id", "route_type"} - set(tm.columns)
    if missing_cols:
        raise ValueError(f"trip metadata missing columns: {sorted(missing_cols)}")
    mapping = tm.set_index("trip_id")["route_type"].to_dict()
    test_trips = set(telemetry["trip_id"].unique())
    absent = test_trips - set(mapping)
    if absent:
        raise ValueError(
            f"test trip(s) missing route_type in metadata: {sorted(absent)[:5]}"
        )
    return mapping


# --------------------------------------------------------------------------
# R² with explicit degenerate handling (never silently misleading)
# --------------------------------------------------------------------------

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[Optional[float], str]:
    """Return ``(r2, flag)``.

    ``r2`` is ``None`` (with a reason flag) when R² is undefined: fewer than
    two rows, or a constant target or constant prediction (division by zero
    in the denominator). This ensures degenerate trips are handled explicitly
    rather than silently producing a misleading R².
    """
    if len(y_true) < 2:
        return None, "insufficient_rows_lt_2"
    denom = float(np.var(y_true))
    if not np.isfinite(denom) or denom == 0.0:
        return None, "constant_target"
    if float(np.var(y_pred)) == 0.0:
        return None, "constant_prediction"
    return round(float(r2_score(y_true, y_pred)), 6), "ok"


# --------------------------------------------------------------------------
# Tail-error quantiles
# --------------------------------------------------------------------------

def tail_quantiles(
    absolute_error: np.ndarray,
    quantiles: tuple[float, ...] = TRAIL_QUANTILES,
) -> dict[str, float]:
    """Absolute-error p90/p95/p99 and max for a set of rows."""
    ae = np.asarray(absolute_error, dtype=float)
    out: dict[str, float] = {}
    for q in quantiles:
        out[f"p{int(round(q * 100))}"] = round(float(np.quantile(ae, q)), 6)
    out["max"] = round(float(ae.max()), 6) if len(ae) else None
    return out


def per_truck_tail(
    frame: pd.DataFrame,
    quantiles: tuple[float, ...] = TRAIL_QUANTILES,
) -> dict[str, dict]:
    """Tail-error stats per test truck."""
    out: dict[str, dict] = {}
    for truck, g in frame.groupby("truck_id", sort=True):
        ae = np.abs(g["y_true"].astype(float) - g["y_pred"].astype(float)).to_numpy()
        out[str(truck)] = {**tail_quantiles(ae, quantiles), "n": int(len(ae))}
    return out


# --------------------------------------------------------------------------
# Per-trip metrics
# --------------------------------------------------------------------------

def per_trip_metrics(
    frame: pd.DataFrame,
    quantiles: tuple[float, ...] = TRAIL_QUANTILES,
) -> list[dict]:
    """One entry per trip: MAE/RMSE, explicit R² handling, tail quantiles.

    Each entry includes the truck, route type, vehicle class and row count so
    per-trip diagnostics stay fully traceable back to the frozen split.
    """
    rows: list[dict] = []
    for trip_id, g in frame.groupby("trip_id", sort=True):
        t = g["y_true"].to_numpy(dtype=float)
        p = g["y_pred"].to_numpy(dtype=float)
        ae = np.abs(t - p)
        r2_val, r2_flag = safe_r2(t, p)
        entry: dict = {
            "trip_id": str(trip_id),
            "truck_id": str(g["truck_id"].iloc[0]),
            "vehicle_class": str(g["vehicle_class"].iloc[0]),
            "route_type": str(g["route_type"].iloc[0]),
            "n": int(len(t)),
            "mae": round(float(ae.mean()), 6),
            "rmse": round(float(np.sqrt(np.mean((t - p) ** 2))), 6),
            "r2": r2_val,
            "r2_flag": r2_flag,
        }
        entry.update(tail_quantiles(ae, quantiles))
        rows.append(entry)
    return rows


# --------------------------------------------------------------------------
# Paired comparison (full-operational vs telematics-only)
# --------------------------------------------------------------------------

#: Delta sign convention: delta = MAE_telematics - MAE_full_operational.
#: A positive delta means full-operational has *lower* error.
DELTA_SIGN = (
    "delta = MAE_telematics_only - MAE_full_operational; "
    "positive => full-operational has lower error (better)"
)


def _assert_aligned(full: pd.DataFrame, tel: pd.DataFrame) -> None:
    """Both feature-set frames must share exactly the same test rows."""
    if not full.index.equals(tel.index):
        raise ValueError("full and telematics-only frames are not index-aligned")
    if len(full) != len(tel):
        raise ValueError("full and telematics-only frames have different lengths")


def pooled_paired_delta(
    full: pd.DataFrame,
    tel: pd.DataFrame,
) -> dict:
    """Row-level pooled paired delta over all test rows."""
    _assert_aligned(full, tel)
    y = full["y_true"].to_numpy(dtype=float)
    pf = full["y_pred"].to_numpy(dtype=float)
    pt = tel["y_pred"].to_numpy(dtype=float)
    ae_full = np.abs(y - pf)
    ae_tel = np.abs(y - pt)
    mae_full = float(ae_full.mean())
    mae_tel = float(ae_tel.mean())
    rmse_full = float(np.sqrt(np.mean((y - pf) ** 2)))
    rmse_tel = float(np.sqrt(np.mean((y - pt) ** 2)))
    return {
        "n_rows": int(len(y)),
        "mae_full_operational": round(mae_full, 6),
        "mae_telematics_only": round(mae_tel, 6),
        "delta_mae": round(mae_tel - mae_full, 6),
        "rmse_full_operational": round(rmse_full, 6),
        "rmse_telematics_only": round(rmse_tel, 6),
        "delta_rmse": round(rmse_tel - rmse_full, 6),
        "sign_convention": DELTA_SIGN,
    }


def paired_grouped_delta(
    full: pd.DataFrame,
    tel: pd.DataFrame,
    group_col: str = "trip_id",
) -> dict[str, dict]:
    """Paired MAE/RMSE deltas per group (trip or truck)."""
    _assert_aligned(full, tel)
    y = full["y_true"].to_numpy(dtype=float)
    pf = full["y_pred"].to_numpy(dtype=float)
    pt = tel["y_pred"].to_numpy(dtype=float)
    groups = full[group_col].to_numpy()

    out: dict[str, dict] = {}
    for gv in np.unique(groups):
        mask = groups == gv
        yy, gpf, gpt = y[mask], pf[mask], pt[mask]
        mae_full = float(np.mean(np.abs(yy - gpf)))
        mae_tel = float(np.mean(np.abs(yy - gpt)))
        rmse_full = float(np.sqrt(np.mean((yy - gpf) ** 2)))
        rmse_tel = float(np.sqrt(np.mean((yy - gpt) ** 2)))
        out[str(gv)] = {
            "n": int(mask.sum()),
            "mae_full_operational": round(mae_full, 6),
            "mae_telematics_only": round(mae_tel, 6),
            "delta_mae": round(mae_tel - mae_full, 6),
            "rmse_full_operational": round(rmse_full, 6),
            "rmse_telematics_only": round(rmse_tel, 6),
            "delta_rmse": round(rmse_tel - rmse_full, 6),
        }
    return out


# --------------------------------------------------------------------------
# Trip-clustered bootstrap
# --------------------------------------------------------------------------

def trip_clustered_bootstrap(
    full: pd.DataFrame,
    tel: pd.DataFrame,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    trip_col: str = "trip_id",
) -> dict:
    """95% percentile intervals via trip-clustered resampling.

    Resamples whole trips (with replacement, same size as the 96 test trips)
    and recomputes pooled MAE/RMSE for each model plus the paired MAE delta on
    exactly the same resampled trips for both models. This keeps repeated rows
    of a trip together, so autocorrelation within a trip is not treated as
    independence.

    IMPORTANT: the resulting intervals estimate uncertainty *conditional on
    the five frozen test trucks* -- they are not a population-level
    generalization statement across trucks.
    """
    _assert_aligned(full, tel)
    trips = np.asarray(sorted(set(full[trip_col])))
    # Precompute per-trip aggregate error sums (exactly equivalent to
    # recomputing pooled metrics on the rows of the chosen trips).
    g = full.groupby(trip_col, sort=True)
    n = g.size().to_numpy(dtype=float)
    y_t = full["y_true"].to_numpy(dtype=float)
    pf_t = full["y_pred"].to_numpy(dtype=float)
    pt_t = tel["y_pred"].to_numpy(dtype=float)
    abs_full = g.apply(lambda d: float(np.abs(d["y_true"] - d["y_pred"]).sum())).to_numpy()
    abs_tel = g.apply(lambda d: float(np.abs(d["y_true"] - tel.loc[d.index, "y_pred"]).sum())).to_numpy()
    sq_full = g.apply(lambda d: float(((d["y_true"] - d["y_pred"]) ** 2).sum())).to_numpy()
    sq_tel = g.apply(lambda d: float(((d["y_true"] - tel.loc[d.index, "y_pred"]) ** 2).sum())).to_numpy()

    rng = np.random.default_rng(seed)
    # (n_resamples, n_trips) matrix of chosen trip indices — fully vectorized.
    chosen = rng.choice(len(trips), size=(n_resamples, len(trips)), replace=True)

    total_n = n[chosen].sum(axis=1)
    sum_abs_full = abs_full[chosen].sum(axis=1)
    sum_abs_tel = abs_tel[chosen].sum(axis=1)
    sum_sq_full = sq_full[chosen].sum(axis=1)
    sum_sq_tel = sq_tel[chosen].sum(axis=1)

    samples = {
        "mae_full": sum_abs_full / total_n,
        "rmse_full": np.sqrt(sum_sq_full / total_n),
        "mae_tel": sum_abs_tel / total_n,
        "rmse_tel": np.sqrt(sum_sq_tel / total_n),
        "delta_mae": (sum_abs_tel - sum_abs_full) / total_n,
    }

    out: dict = {
        "method": (
            "trip-clustered percentile bootstrap; resample the 96 test trips "
            "with replacement, jointly for both models; 95% percentile interval"
        ),
        "resampling_unit": "trip_id",
        "n_trips": int(len(trips)),
        "n_resamples": n_resamples,
        "n_resamples_used": int(len(samples["mae_full"])),
        "seed": seed,
        "conditional_on_five_test_trucks": True,
        "note": (
            "estimates uncertainty conditional on these five test trucks; "
            "does NOT establish population-level generalization across trucks"
        ),
    }
    for k, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[k] = {
            "ci95_lower": round(float(lo), 6),
            "ci95_upper": round(float(hi), 6),
        }
    return out


# --------------------------------------------------------------------------
# Variance decomposition (descriptive)
# --------------------------------------------------------------------------

def variance_decomposition(frame: pd.DataFrame) -> dict:
    """Decompose squared prediction error into truck / trip / row levels.

    Uses a one-way nested sum-of-squares decomposition of the residuals
    ``e = y_true - y_pred``:

      SS_total = SS_between_trucks + SS_between_trips_within_truck
                + SS_within_trips

    with the share of total squared error attributable to each level reported.
    The truck-level component rests on only five trucks and is therefore
    labelled unstable/descriptive; no causal interpretation is attached.
    """
    df = frame.copy()
    df["resid"] = df["y_true"].astype(float) - df["y_pred"].astype(float)
    n = len(df)
    grand = df["resid"].mean()
    truck_mean = df.groupby("truck_id")["resid"].transform("mean")
    trip_mean = df.groupby("trip_id")["resid"].transform("mean")

    ss_total = float(((df["resid"] - grand) ** 2).sum())
    ss_truck = float(((truck_mean - grand) ** 2).sum())
    ss_trip_within = float(((trip_mean - truck_mean) ** 2).sum())
    ss_within = ss_total - ss_truck - ss_trip_within

    n_trucks = int(df["truck_id"].nunique())
    n_trips = int(df["trip_id"].nunique())

    def _share(ss: float) -> float:
        return round(ss / ss_total, 6) if ss_total > 0 else None

    return {
        "n_rows": int(n),
        "n_trucks": n_trucks,
        "n_trips": n_trips,
        "total_squared_error": round(ss_total, 6),
        "between_truck_ss": round(ss_truck, 6),
        "between_trip_within_truck_ss": round(ss_trip_within, 6),
        "within_trip_ss": round(ss_within, 6),
        "share_of_squared_error": {
            "between_truck": _share(ss_truck),
            "between_trip_within_truck": _share(ss_trip_within),
            "within_trip": _share(ss_within),
        },
        "note": (
            "5-truck sample: truck-level variance component is descriptive and "
            "unstable; a formal ANOVA ICC is NOT reported because n=5 trucks is "
            "too small for a defensible between-truck random-effect estimate. "
            "No causal interpretation."
        ),
    }


# --------------------------------------------------------------------------
# Prediction loading (read-only)
# --------------------------------------------------------------------------

def prepare_test_predictions(
    telemetry: Optional[pd.DataFrame] = None,
    manifest: Optional[pd.DataFrame] = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    feature_sets: tuple[str, ...] = FEATURE_SETS,
) -> dict:
    """Load saved artifacts and produce aligned test predictions per feature set.

    Returns ``{"frames": {fs: DataFrame}, "selected_models": {...},
    "test_rows": int, "test_trucks": [...], "n_trips": int}``.

    Each frame carries columns ``trip_id, truck_id, vehicle_class, route_type,
    y_true, y_pred`` with the telemetry row index preserved, so the two
    feature-set frames are exactly index-aligned (same rows).
    """
    if telemetry is None:
        telemetry = load_telemetry()
    if manifest is None:
        manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)

    test_trucks = sorted(split.test_trucks)
    if set(test_trucks) != set(EXPECTED_TEST_TRUCKS):
        raise ValueError(
            "test truck set mismatch for M3.1: "
            f"manifest={test_trucks} expected={sorted(EXPECTED_TEST_TRUCKS)}"
        )

    # Model selection from validation evidence only (same as M1.3).
    with open(report_dir / "metrics_report.json", "r", encoding="utf-8") as fh:
        metrics_report = json.load(fh)
    selected = select_model_from_validation(metrics_report)

    spec = load_feature_spec()
    route_map = _route_type_map(split.test)

    frames: dict[str, pd.DataFrame] = {}
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

        X_raw, y, _ = build_feature_matrix(split.test, fs, spec=spec)
        X_test = preprocessor.transform(X_raw)
        pred = pd.Series(estimator.predict(X_test), index=X_test.index, dtype=float)

        frames[fs] = pd.DataFrame(
            {
                "trip_id": split.test["trip_id"].values,
                "truck_id": split.test["truck_id"].values,
                "vehicle_class": split.test["vehicle_class"].values,
                "route_type": split.test["trip_id"].map(route_map).values,
                "y_true": y.astype(float).values,
                "y_pred": pred.values,
            },
            index=split.test.index,
        )

    n_trips = int(frames[feature_sets[0]]["trip_id"].nunique())
    if n_trips != EXPECTED_TEST_TRIPS:
        raise ValueError(
            f"expected {EXPECTED_TEST_TRIPS} test trips, got {n_trips}"
        )

    return {
        "frames": frames,
        "selected_models": selected,
        "test_rows": int(len(split.test)),
        "test_trucks": test_trucks,
        "n_trips": n_trips,
    }
