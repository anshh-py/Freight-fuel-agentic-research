"""M1.3 — unseen-truck final evaluation of the M1.2 baseline models.

This module loads the *saved* M1.2 artifacts (models and preprocessors),
evaluates them on the five held-out test trucks only, and produces:

  * pooled test metrics (MAE, RMSE, R², MAPE over ``fuel_rate > 1``),
  * per-truck metrics,
  * per-class metrics (32T / 40T / 49T),
  * a macro-average across test trucks (reported as descriptive only),
  * operating-condition breakdowns (engine on/off, speed bands, road-grade
    bands, acceleration/braking, vehicle class),
  * residual analysis (bias, distribution, error vs actual / speed / engine
    load, per-truck bias) and diagnostic plots.

Protocol guarantees (plan §4.3):
  * the test set is evaluated exactly once, only here, for final evaluation;
  * test data is never used to fit, tune or select models (selection is made
    from the M1.2 validation metrics via ``select_model_from_validation``);
  * preprocessors are loaded from disk and **never refitted**;
  * no dataset or model artifact is modified by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.features.build import build_feature_matrix
from src.features.spec import load_feature_spec
from src.data.loader import enforce_split, load_split_manifest, load_telemetry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"

# Informational only — the split manifest is the source of truth and is
# verified against this at runtime by ``evaluate_test_set``.
EXPECTED_TEST_TRUCKS = [
    "truck_004",
    "truck_001",
    "truck_020",
    "truck_017",
    "truck_028",
]

# Models with saved M1.2 artifacts that we evaluate on test. The dummy
# baselines are included because their artifacts are available (their
# predictions are feature-independent, which the report notes explicitly).
EVALUATION_MODELS = [
    "mean_dummy",
    "median_dummy",
    "ridge",
    "random_forest",
    "xgboost",
]

FEATURE_SETS = ("full_operational", "telematics_only")


# --------------------------------------------------------------------------
# Operating-condition filters (all defined explicitly; nothing is silently
# discarded — every record belongs to at least the "all" bucket, and the
# band definitions below are documented in the summary).
# --------------------------------------------------------------------------

def make_condition_masks(test_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return ``{condition_label: boolean mask}`` over ``test_df``.

    Band edges are derived from the dataset distributions (speed 0–86.6 kph,
    grade ±2.9 %, acceleration −1.6..0.85 m/s²) and documented in the report.
    """
    speed = test_df["speed_kph"]
    grade = test_df["road_grade_pct"]
    accel = test_df["acceleration_mps2"]
    ign = test_df["ignition_status"]

    masks: dict[str, pd.Series] = {
        "all": pd.Series(True, index=test_df.index),
        # engine state
        "engine_on": ign == 1,
        "engine_off": ign == 0,
        # speed bands (kph)
        "speed_0_0.5_stopped": speed < 0.5,
        "speed_0.5_20_slow": (speed >= 0.5) & (speed < 20),
        "speed_20_40_urban": (speed >= 20) & (speed < 40),
        "speed_40_60_suburban": (speed >= 40) & (speed < 60),
        "speed_ge_60_highway": speed >= 60,
        # road-grade bands (%)
        "grade_le_-1.5_steep_downhill": grade <= -1.5,
        "grade_-1.5_-0.3_downhill": (grade > -1.5) & (grade <= -0.3),
        "grade_-0.3_0.3_flat": (grade > -0.3) & (grade < 0.3),
        "grade_0.3_1.5_uphill": (grade >= 0.3) & (grade < 1.5),
        "grade_ge_1.5_steep_uphill": grade >= 1.5,
        # acceleration / braking (m/s²)
        "accel_le_-0.5_braking": accel <= -0.5,
        "accel_-0.5_0.5_cruise": (accel > -0.5) & (accel < 0.5),
        "accel_ge_0.5_accelerating": accel >= 0.5,
    }

    # vehicle class (the third operating dimension)
    for cls in sorted(test_df["vehicle_class"].unique()):
        masks[f"class_{cls}"] = test_df["vehicle_class"] == cls

    return masks


# --------------------------------------------------------------------------
# Metric aggregation helpers
# --------------------------------------------------------------------------

def _subset_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mask: pd.Series,
) -> dict:
    """Compute the metric set over the rows selected by ``mask``."""
    yt = np.asarray(y_true[mask].astype(float))
    yp = np.asarray(y_pred[mask].astype(float))
    base = regression_metrics(yt, yp)
    return {"count": int(len(yt)), **base}


def per_truck_metrics(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, dict]:
    """Compute metrics independently for each test truck."""
    out: dict[str, dict] = {}
    for truck, _ in test_df.groupby("truck_id", sort=True):
        out[truck] = _subset_metrics(
            y_true, y_pred, test_df["truck_id"] == truck
        )
    return out


def per_class_metrics(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, dict]:
    """Compute metrics per vehicle class."""
    out: dict[str, dict] = {}
    for cls, _ in test_df.groupby("vehicle_class", sort=True):
        out[cls] = _subset_metrics(
            y_true, y_pred, test_df["vehicle_class"] == cls
        )
    return out


def macro_average(per_metrics: dict[str, dict]) -> dict:
    """Arithmetic mean of each metric across the per-truck table.

    This is a **descriptive** aggregate over the five test trucks only — it
    is not a statistically robust population estimate (n = 5 trucks) and is
    labelled as such in the report.
    """
    keys = ("mae", "rmse", "r2", "mape")
    out: dict = {}
    for k in keys:
        vals = [m[k] for m in per_metrics.values() if np.isfinite(m[k])]
        out[k] = round(float(np.mean(vals)), 6) if vals else None
    out["basis"] = "descriptive macro-average over 5 test trucks (not robust)"
    return out


def residual_stats(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Residual summaries: bias, spread, and under/over-prediction rates."""
    yt = np.asarray(y_true.astype(float))
    yp = np.asarray(y_pred.astype(float))
    resid = yt - yp  # positive => underpredicted, negative => overpredicted
    n = int(len(resid))
    return {
        "mean_residual": round(float(resid.mean()), 6),
        "median_residual": round(float(np.median(resid)), 6),
        "std_residual": round(float(resid.std()), 6),
        "max_abs_residual": round(float(np.abs(resid).max()), 6),
        "fraction_underpredicted": round(float((resid > 0).mean()), 6),
        "fraction_overpredicted": round(float((resid < 0).mean()), 6),
        "n": n,
    }


# --------------------------------------------------------------------------
# Model selection from **validation** evidence (M1.2). The test set is never
# used for selection, tuning or feature choice.
# --------------------------------------------------------------------------

def select_model_from_validation(metrics_report: dict) -> dict[str, dict]:
    """Pick the best model per feature set using validation MAE.

    This uses the M1.2 *validation* report only. The low-MAE model is
    chosen; ties fall to the first encountered in registry order.
    """
    selected: dict[str, dict] = {}
    for fs, fsr in metrics_report["feature_sets"].items():
        best_name = None
        best_mae = float("inf")
        val_mae = {}
        for name, m in fsr["models"].items():
            mae = float(m["metrics"]["mae"])
            val_mae[name] = mae
            if mae < best_mae:
                best_mae = mae
                best_name = name
        selected[fs] = {
            "model": best_name,
            "basis": "validation MAE (M1.2) — test data not used",
            "validation_mae": round(best_mae, 6),
            "validation_mae_all_models": {
                k: round(v, 6) for k, v in val_mae.items()
            },
        }
    return selected


# --------------------------------------------------------------------------
# Plot generation (matplotlib, Agg backend — non-interactive)
# --------------------------------------------------------------------------

def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _binned_mean(df: pd.DataFrame, xcol: str, ycol: str, bins) -> pd.DataFrame:
    """Mean of ``ycol`` per bin of ``xcol`` (for diagnostic scatter lines)."""
    binned = df.groupby(pd.cut(df[xcol], bins), observed=False)[ycol].agg(
        ["mean", "count"]
    )
    return binned.dropna()


def _bin_midpoints(index) -> list:
    """Interval midpoints for a (possibly Categorical) bin index."""
    return [float(iv.mid) for iv in index]


def generate_residual_plots(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    feature_set: str,
    plots_dir: Path,
    selected_model_label: str,
) -> list[str]:
    """Write residual diagnostic plots; return the list of file names."""
    import io

    plt = _import_pyplot()
    resid = y_true.astype(float) - y_pred.astype(float)
    df = pd.DataFrame(
        {
            "actual": y_true.astype(float),
            "pred": y_pred.astype(float),
            "resid": resid,
            "speed": test_df["speed_kph"].astype(float),
            "engine_load": test_df["engine_load_pct"].astype(float),
            "grade": test_df["road_grade_pct"].astype(float),
            "truck": test_df["truck_id"].astype(str),
        },
        index=test_df.index,
    )

    files: list[str] = []

    # 1. residual histogram
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(resid, bins=100, color="#4472C4", edgecolor="none")
    ax.axvline(0.0, color="black", lw=1.2, ls="--")
    ax.set_title(f"Residual distribution — {feature_set} ({selected_model_label})")
    ax.set_xlabel("Residual (actual − predicted, L/h)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    name = f"residuals_hist_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)

    # 2. residual vs actual fuel rate (binned mean + scatter)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(df["actual"], df["resid"], s=2, alpha=0.08, color="#8FAADC")
    bins = np.linspace(df["actual"].min(), df["actual"].quantile(0.99), 40)
    bm = _binned_mean(df, "actual", "resid", bins)
    ax.plot(_bin_midpoints(bm.index), bm["mean"], color="black", lw=2)
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.set_title(f"Residual vs actual fuel rate — {feature_set}")
    ax.set_xlabel("Actual fuel rate (L/h)")
    ax.set_ylabel("Residual (L/h)")
    ax.set_xlim(df["actual"].min(), df["actual"].quantile(0.995))
    fig.tight_layout()
    name = f"residual_vs_actual_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)

    # 3. error vs speed (binned mean + spread)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    bm = _binned_mean(df, "speed", "resid", bins)
    ax.plot(_bin_midpoints(bm.index), bm["mean"], marker="o", color="#C00000")
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.set_title(f"Mean residual vs speed — {feature_set}")
    ax.set_xlabel("Speed (kph)")
    ax.set_ylabel("Mean residual (L/h)")
    fig.tight_layout()
    name = f"error_vs_speed_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)

    # 4. error vs engine load (full_operational only; telematics lacks it)
    if "engine_load" in df and df["engine_load"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.linspace(
            df["engine_load"].min(), df["engine_load"].quantile(0.99), 40
        )
        bm = _binned_mean(df, "engine_load", "resid", bins)
        ax.plot(_bin_midpoints(bm.index), bm["mean"], color="#70AD47")
        ax.axhline(0.0, color="gray", lw=1, ls="--")
        ax.set_title(f"Mean residual vs engine load — {feature_set}")
        ax.set_xlabel("Engine load (%)")
        ax.set_ylabel("Mean residual (L/h)")
        fig.tight_layout()
        name = f"error_vs_engine_load_{feature_set}.png"
        fig.savefig(plots_dir / name, dpi=130)
        plt.close(fig)
        files.append(name)

    # 5. per-truck mean residual
    fig, ax = plt.subplots(figsize=(7, 4.5))
    per_truck = df.groupby("truck")["resid"].mean().sort_values()
    colors = ["#C00000" if v > 0 else "#4472C4" for v in per_truck.values]
    ax.bar(per_truck.index, per_truck.values, color=colors)
    ax.axhline(0.0, color="black", lw=1.2)
    ax.set_title(f"Mean residual per test truck — {feature_set}")
    ax.set_ylabel("Mean residual (L/h)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    name = f"per_truck_mean_residual_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)

    return files


# --------------------------------------------------------------------------
# Main evaluator
# --------------------------------------------------------------------------

def _env_versions() -> dict[str, str]:
    import platform
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": "1.9.1",
        "xgboost": "3.4.1",
    }


def _artifact_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else -1


def evaluate_test_set(
    telemetry: Optional[pd.DataFrame] = None,
    manifest: Optional[pd.DataFrame] = None,
    models: Optional[list[str]] = None,
    feature_sets: Optional[list[str]] = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    plots_dir: Optional[Path] = None,
) -> dict:
    """Evaluate the saved M1.2 models on the five held-out test trucks.

    Reads only: telemetry, split manifest, and the M1.2 artifacts under
    ``report_dir/artifacts``. Models and preprocessors are loaded, never
    refitted. Test data is used for evaluation only — never for selection.
    """
    if telemetry is None:
        telemetry = load_telemetry()
    if manifest is None:
        manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)

    test_trucks = sorted(split.test_trucks)
    # Source of truth is the manifest; cross-check against the expected list.
    expected_set = set(EXPECTED_TEST_TRUCKS)
    test_set = set(test_trucks)
    train_val_set = set(split.train_trucks) | set(split.val_trucks)

    #: Never evaluate a train/val truck on test.
    leak_trucks = sorted(test_set & train_val_set)
    if leak_trucks:
        raise ValueError(
            f"test evaluation would include non-test truck(s): {leak_trucks}"
        )
    if test_set != expected_set:
        raise ValueError(
            f"test truck set mismatch: manifest={sorted(test_set)} "
            f"expected={sorted(expected_set)}"
        )

    models = list(models) if models is not None else list(EVALUATION_MODELS)
    unknown = set(models) - set(EVALUATION_MODELS)
    if unknown:
        raise ValueError(f"unknown model(s) for evaluation: {sorted(unknown)}")

    feature_sets = (
        list(feature_sets) if feature_sets is not None else list(FEATURE_SETS)
    )
    unknown_fs = set(feature_sets) - set(FEATURE_SETS)
    if unknown_fs:
        raise ValueError(f"unknown feature set(s): {sorted(unknown_fs)}")

    # Validation-based model selection (test data never used).
    metrics_report_path = report_dir / "metrics_report.json"
    with open(metrics_report_path, "r", encoding="utf-8") as fh:
        metrics_report = json.load(fh)
    selected = select_model_from_validation(metrics_report)

    if plots_dir is None:
        plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    spec = load_feature_spec()
    report: dict = {
        "experiment": "M1.3_unseen_truck_final_evaluation",
        "milestone": "M1.3",
        "protocol": (
            "evaluate saved M1.2 models on the 5 held-out test trucks only; "
            "test data never used for fitting/tuning/selection"
        ),
        "selected_models_from_validation": selected,
        "environment": _env_versions(),
        "verification": {
            "test_trucks_evaluated": test_trucks,
            "all_expected_test_trucks_present": test_set == expected_set,
            "train_or_val_truck_in_test_evaluation": False,
            "test_rows": int(len(split.test)),
            "preprocessors_loaded_from_disk_not_refitted": True,
            "generated_outputs_under": str(report_dir),
        },
        "feature_sets": {},
    }

    for fs in feature_sets:
        artifacts_dir = report_dir / "artifacts" / fs
        pre_path = artifacts_dir / "preprocessor.joblib"
        if not pre_path.exists():
            raise FileNotFoundError(f"preprocessor artifact missing: {pre_path}")

        # Load the fitted (training) encoder and transform test — never fit.
        preprocessor = joblib.load(pre_path)
        if not getattr(preprocessor, "_fitted", False):
            raise RuntimeError(f"preprocessor {pre_path} is not fitted")

        X_raw, y, names = build_feature_matrix(split.test, fs, spec=spec)
        X_test = preprocessor.transform(X_raw)

        fs_result: dict = {
            "feature_set": fs,
            "columns": X_test.columns.tolist(),
            "n_features": int(X_test.shape[1]),
            "test_n_rows": int(len(X_test)),
            "models": {},
            "plots": {},
        }

        cond_masks = make_condition_masks(split.test)

        for name in models:
            artifact_path = artifacts_dir / f"{name}.joblib"
            if not artifact_path.exists():
                raise FileNotFoundError(f"model artifact missing: {artifact_path}")
            estimator = joblib.load(artifact_path)

            pred = pd.Series(
                estimator.predict(X_test), index=X_test.index, dtype=float
            )

            pooled = regression_metrics(y, pred)
            pooled["count"] = int(len(y))
            per_trk = per_truck_metrics(split.test, y, pred)
            per_cls = per_class_metrics(split.test, y, pred)
            macro = macro_average(per_trk)

            conditions = {
                label: _subset_metrics(y, pred, mask)
                for label, mask in cond_masks.items()
            }

            entry: dict = {
                "artifact": str(artifact_path),
                "artifact_size_bytes": _artifact_size_bytes(artifact_path),
                "pooled": pooled,
                "per_truck_metrics": per_trk,
                "per_class_metrics": per_cls,
                "macro_average_across_test_trucks": macro,
                "conditions": conditions,
                "residual_stats": residual_stats(y, pred),
            }
            # Dummy regressors ignore features entirely: their metrics are
            # identical across feature sets by construction (documented).
            if name in ("mean_dummy", "median_dummy"):
                entry["note"] = (
                    "feature-independent prediction (constant); metrics are "
                    "identical across feature sets by design"
                )
            fs_result["models"][name] = entry

            # Residual plots only for the validation-selected model.
            if name == selected[fs]["model"]:
                fs_result["plots"] = {
                    "files": generate_residual_plots(
                        split.test,
                        y,
                        pred,
                        fs,
                        plots_dir,
                        selected[fs]["model"],
                    )
                }

        report["feature_sets"][fs] = fs_result

    return report
