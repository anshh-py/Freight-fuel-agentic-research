"""M1.4 — explainability and feature analysis for the validation-selected
XGBoost models (``full_operational`` and ``telematics_only``).

This module explains *models trained on synthetic data*. It loads the saved
M1.2 model + preprocessor artifacts (never refits, never retrains) and uses
the M1.3 held-out test trucks for the reported explanations.

Scope / guarantees
  * models are loaded from ``reports/m1/artifacts/<feature_set>/xgboost.joblib``
  * preprocessors are loaded from disk and **never refitted**;
  * explanations are computed on a deterministic, reproducible sample of the
    **test** set (never train/validation rows);
  * SHAP contributions are validated to reconstruct the model output within a
    numerical tolerance;
  * nothing here is causal: labels state the explanations describe a model
    trained on synthetic telemetry, not real-world fuel consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.build import build_feature_matrix
from src.features.spec import load_feature_spec
from src.data.loader import enforce_split, load_split_manifest, load_telemetry

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"

# The two feature sets whose *validation-selected* XGBoost models we explain.
FEATURE_SETS = ("full_operational", "telematics_only")
SELECTED_MODEL = "xgboost"

# Deterministic sampling: per-truck subsample, fixed seed, all five test
# trucks represented (see ``make_explanation_sample``).
DEFAULT_SAMPLE_PER_TRUCK = 500
RANDOM_STATE = 20240617

# Reconstruction tolerance for validating SHAP contributions (absolute L1
# mismatch between ``base + sum(shap)`` and the model prediction).
RECONSTRUCTION_TOLERANCE = 1e-3

# Operating-condition subgroups for explainability (all defined on the raw
# telemetry columns, so the filters are explicit and reproducible). Each is a
# documented, mutually-exclusive-on-definition filter; no records are silently
# dropped (raw row counts are always reported).
SUBGROUP_DEFS = {
    "highway": {
        "description": "Speed >= 60 kph",
        "column": "speed_kph",
        "op": "ge",
        "threshold": 60.0,
    },
    "acceleration": {
        "description": "Acceleration >= 0.5 m/s\u00b2",
        "column": "acceleration_mps2",
        "op": "ge",
        "threshold": 0.5,
    },
}

# Below this many sampled rows a subgroup importance ranking is reported as
# indicative rather than asserted as a robust ranking.
MIN_SUBGROUP_ROWS_FOR_STABLE_IMPORTANCE = 50


# --------------------------------------------------------------------------
# Readable feature-name mapping for transformed model inputs
# --------------------------------------------------------------------------

# Transformed column -> human-readable label. The one-hot columns are
# produced by ``FeaturePreprocessor`` from ``vehicle_class``; numeric columns
# keep their telemetry name.
_READABLE_LABELS = {
    "gross_vehicle_weight_kg": "Gross vehicle weight (kg)",
    "payload_kg": "Payload (kg)",
    "vehicle_mass_kg": "Vehicle mass (kg)",
    "speed_kph": "Speed (kph)",
    "acceleration_mps2": "Acceleration (m/s²)",
    "road_grade_pct": "Road grade (%)",
    "ignition_status": "Ignition status",
    "engine_rpm": "Engine speed (rpm)",
    "engine_load_pct": "Engine load (%)",
    "throttle_position_pct": "Throttle position (%)",
    "torque_nm": "Torque (N·m)",
    "coolant_temp_c": "Coolant temp (°C)",
    "intake_air_temp_c": "Intake air temp (°C)",
    "ambient_temp_c": "Ambient temp (°C)",
    "vehicle_class_32T": "Vehicle class (32T)",
    "vehicle_class_40T": "Vehicle class (40T)",
    "vehicle_class_49T": "Vehicle class (49T)",
}


def map_readable_names(transformed_cols: list[str]) -> dict[str, str]:
    """Map transformed model-input columns to readable labels.

    Unknown transformed columns are mapped to their own name so the mapping is
    total (never silently drops a feature).
    """
    return {c: _READABLE_LABELS.get(c, c) for c in transformed_cols}


# --------------------------------------------------------------------------
# Loading the saved selected models + preprocessors
# --------------------------------------------------------------------------

def load_selected_xgb(
    report_dir: Path = DEFAULT_REPORT_DIR,
    feature_sets: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Load the validation-selected XGBoost model and its saved preprocessor.

    Returns ``{feature_set: {"model": est, "preprocessor": preproc,
    "columns": [transformed names]}}``. The preprocessor is loaded from disk
    and is never refitted; its ``_fitted`` flag is asserted.
    """
    fs_list = list(feature_sets) if feature_sets is not None else list(FEATURE_SETS)
    unknown = set(fs_list) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"unknown feature set(s): {sorted(unknown)}")

    out: dict[str, dict] = {}
    for fs in fs_list:
        artifacts_dir = report_dir / "artifacts" / fs
        model_path = artifacts_dir / f"{SELECTED_MODEL}.joblib"
        pre_path = artifacts_dir / "preprocessor.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"model artifact missing: {model_path}")
        if not pre_path.exists():
            raise FileNotFoundError(f"preprocessor artifact missing: {pre_path}")

        estimator = joblib.load(model_path)
        preprocessor = joblib.load(pre_path)
        if not getattr(preprocessor, "_fitted", False):
            raise RuntimeError(f"preprocessor {pre_path} is not fitted")

        columns = list(preprocessor.feature_names_)
        out[fs] = {
            "model": estimator,
            "preprocessor": preprocessor,
            "columns": columns,
        }
    return out


# --------------------------------------------------------------------------
# Deterministic test sample (all five trucks represented)
# --------------------------------------------------------------------------

def make_explanation_sample(
    test_df: pd.DataFrame,
    sample_per_truck: int = DEFAULT_SAMPLE_PER_TRUCK,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Draw a deterministic per-truck subsample of the test set.

    For each test truck, ``sample_per_truck`` rows are drawn uniformly at
    random with a fixed ``np.random.RandomState(seed)``. This guarantees every
    held-out truck is represented, keeps SHAP cost low and documented, and is
    fully reproducible (the seed is fixed across runs).
    """
    if sample_per_truck <= 0:
        raise ValueError("sample_per_truck must be positive")
    rng = np.random.RandomState(seed)
    chosen: list[pd.Index] = []
    for truck, sub in test_df.groupby("truck_id", sort=True):
        n = len(sub)
        k = min(sample_per_truck, n)
        idx = np.sort(rng.choice(np.arange(n), size=k, replace=False))
        chosen.append(sub.index[idx])
    # Combine and preserve the original index order.
    index = pd.Index(np.concatenate([list(c) for c in chosen]))
    return test_df.loc[index.unique()].copy()


# --------------------------------------------------------------------------
# SHAP computation + validation
# --------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return ``(shap_values, expected_value)`` for a tree model.

    ``shap_values`` has shape ``(len(X), n_features)`` aligned to ``X.columns``.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = float(explainer.expected_value)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:  # multi-output reshape safety
        shap_values = shap_values[..., 0]
    if shap_values.shape != (len(X), X.shape[1]):
        raise ValueError(
            "SHAP value shape mismatch: "
            f"{shap_values.shape} != {(len(X), X.shape[1])}"
        )
    return shap_values, expected_value


def validate_reconstruction(
    expected_value: float,
    shap_values: np.ndarray,
    predictions: np.ndarray,
    tolerance: float = RECONSTRUCTION_TOLERANCE,
) -> dict:
    """Check ``prediction ≈ expected_value + Σ shap``.

    Returns the max absolute deviation, the fraction of rows within
    ``tolerance``, and a boolean pass. Positive mismatch indicates the SHAP
    additive decomposition explains the model output.
    """
    preds = np.asarray(predictions, dtype=float).ravel()
    reconstructed = expected_value + shap_values.sum(axis=1)
    dev = np.abs(reconstructed - preds)
    return {
        "max_abs_deviation": float(dev.max()),
        "mean_abs_deviation": float(dev.mean()),
        "fraction_within_tolerance": float((dev <= tolerance).mean()),
        "tolerance": float(tolerance),
        "pass": bool((dev <= tolerance).all()),
    }


def mean_abs_shap(
    shap_values: np.ndarray,
    columns: list[str],
    readable: dict[str, str],
) -> pd.DataFrame:
    """Mean absolute SHAP per feature, ranked descending.

    Returns a DataFrame with transformed column, readable label, and mean
    absolute SHAP value.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {
            "column": columns,
            "readable_label": [readable[c] for c in columns],
            "mean_abs_shap": mean_abs,
        }
    )
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Operating-condition subgroup SHAP analysis
# --------------------------------------------------------------------------

def get_subgroup_mask(test_df: pd.DataFrame, subgroup: str) -> np.ndarray:
    """Return the boolean row mask for an operating-condition subgroup.

    The filter is defined on the raw telemetry columns (see ``SUBGROUP_DEFS``),
    so it is explicit and reproducible. Raises for unknown subgroups.
    """
    if subgroup not in SUBGROUP_DEFS:
        raise ValueError(
            f"unknown subgroup {subgroup!r}; available: {sorted(SUBGROUP_DEFS)}"
        )
    cfg = SUBGROUP_DEFS[subgroup]
    col = cfg["column"]
    if col not in test_df.columns:
        raise KeyError(f"subgroup column {col!r} missing from test frame")
    values = test_df[col].to_numpy(dtype=float)
    if cfg["op"] == "ge":
        return values >= cfg["threshold"]
    if cfg["op"] == "le":
        return values <= cfg["threshold"]
    raise ValueError(f"unsupported subgroup operator: {cfg['op']}")


def make_subgroup_sample(
    test_df: pd.DataFrame,
    subgroup: str,
    sample_per_truck: int = DEFAULT_SAMPLE_PER_TRUCK,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, int]:
    """Deterministic per-truck subsample of a subgroup's rows.

    Returns ``(sample, total_rows_in_subgroup)``. The sample is drawn with the
    same fixed ``np.random.RandomState(seed)`` used for the global sample; the
    total subgroup row count is returned so it is always reported (no silent
    discarding).
    """
    mask = get_subgroup_mask(test_df, subgroup)
    total_rows = int(mask.sum())
    subgroup_df = test_df.loc[mask]
    sample = make_explanation_sample(subgroup_df, sample_per_truck, seed)
    return sample, total_rows


def subgroup_mean_abs_shap(
    model,
    X: pd.DataFrame,
    sample_index: pd.Index,
    columns: list[str],
    readable: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """Mean absolute SHAP importance on a subgroup sample.

    Returns ``(importance_df, reconstruction)``. ``X`` is the transformed,
    full test feature matrix; ``sample_index`` selects the subgroup rows. The
    model is only used for prediction + SHAP (never retrained).
    """
    X_sub = X.loc[sample_index]
    preds = model.predict(X_sub)
    shap_values, expected_value = compute_shap_values(model, X_sub)
    recon = validate_reconstruction(expected_value, shap_values, preds)
    importance = mean_abs_shap(shap_values, columns, readable)
    return importance, recon


def generate_subgroup_importance_plot(
    importance: pd.DataFrame,
    feature_set: str,
    subgroup: str,
    plots_dir: Path,
) -> str:
    """Horizontal bar of mean |SHAP| for one subgroup."""
    plt = _import_pyplot()
    label = SUBGROUP_DEFS[subgroup]["description"]
    top = importance.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["readable_label"], top["mean_abs_shap"], color="#ED7D31")
    ax.set_title(f"Subgroup importance ({label}) — {feature_set}")
    ax.set_xlabel("Mean |SHAP| (L/h)")
    fig.tight_layout()
    name = f"subgroup_importance_{feature_set}_{subgroup}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    return name

def per_truck_shap(
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    columns: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Mean absolute SHAP per feature for each sampled truck.

    Returns ``(dfs, row_counts)`` where ``row_counts[truck]`` is the number
    of sampled rows used for that truck (so per-truck aggregation can be
    verified to account for exactly the sampled records).
    """
    out: dict[str, pd.DataFrame] = {}
    counts: dict[str, int] = {}
    trucks = sorted(sample["truck_id"].unique())
    for truck in trucks:
        mask = (sample["truck_id"] == truck).to_numpy()
        counts[truck] = int(mask.sum())
        mean_abs = np.abs(shap_values[mask]).mean(axis=0)
        out[truck] = pd.DataFrame(
            {"column": columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out, counts


def top_shap_contributions(
    shap_values: np.ndarray,
    columns: list[str],
    readable: dict[str, str],
    row_idx: int,
    k: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Top positive and negative SHAP contributions for a single row."""
    values = shap_values[row_idx]
    order = np.argsort(values)
    negative = [
        {"column": columns[i], "readable_label": readable[columns[i]],
         "shap": float(values[i])}
        for i in order[:k]
    ]  # most-negative first
    positive = [
        {"column": columns[i], "readable_label": readable[columns[i]],
         "shap": float(values[i])}
        for i in order[-k:]
    ][::-1]  # most-positive first
    return positive, negative


# --------------------------------------------------------------------------
# Deterministic sample-record selection for individual explanations
# --------------------------------------------------------------------------

def select_explanation_records(
    sample: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    per_truck: int = 3,
) -> list[dict]:
    """Pick representative rows with deterministic rules (per truck).

    Rules (never "best-looking" only): for each truck select
      * the row with the largest absolute residual,
      * the row with the median absolute residual,
      * the row closest to zero residual (#i.e. a typical near-0 error).
    This spans the error spectrum and does not cherry-pick successes.
    """
    resid = y_true.astype(float) - y_pred.astype(float)
    abs_resid = resid.abs()
    records: list[dict] = []
    for truck, sub in sample.groupby("truck_id", sort=True):
        idx = list(sub.index)
        a = abs_resid.loc[idx]
        n = len(idx)
        # largest absolute residual
        i_max = int(a.idxmax()) if n else None
        # median absolute residual
        mid = int(np.floor((n - 1) / 2))
        idx_sorted = sorted(idx, key=lambda i: a[i])
        i_med = idx_sorted[mid]
        # closest to zero residual
        i_zero = int(abs_resid.loc[idx].idxmin()) if n else None
        for kind, i in (("largest_abs_residual", i_max),
                        ("median_abs_residual", i_med),
                        ("closest_to_zero_residual", i_zero)):
            if i is None:
                continue
            records.append({
                "truck_id": truck,
                "index": int(i),
                "rule": kind,
                "actual_fuel_rate": float(y_true.loc[i]),
                "predicted_fuel_rate": float(y_pred.loc[i]),
                "residual": float(resid.loc[i]),  # actual − predicted
            })
    return records


# --------------------------------------------------------------------------
# Plot generation (matplotlib, Agg backend — non-interactive)
# --------------------------------------------------------------------------

def _import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_global_importance_plot(
    importance: pd.DataFrame,
    feature_set: str,
    plots_dir: Path,
) -> str:
    """Horizontal bar of mean |SHAP| (global importance)."""
    plt = _import_pyplot()
    top = importance.head(16).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["readable_label"], top["mean_abs_shap"], color="#4472C4")
    ax.set_title(f"Global feature importance (mean |SHAP|) — {feature_set}")
    ax.set_xlabel("Mean |SHAP| (L/h)")
    fig.tight_layout()
    name = f"global_importance_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    return name


def generate_summary_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_set: str,
    plots_dir: Path,
) -> str:
    """SHAP beeswarm summary plot."""
    import shap

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X, show=False, max_display=16, color_bar=True
    )
    plt.title(f"SHAP summary — {feature_set}")
    fig.tight_layout()
    name = f"summary_beeswarm_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_dependence_plots(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_set: str,
    plots_dir: Path,
    top_features: list[str],
    n_plots: int = 3,
) -> list[str]:
    """SHAP dependence plots for the top features."""
    import shap

    plt = _import_pyplot()
    files: list[str] = []
    for col in top_features[:n_plots]:
        try:
            j = X.columns.get_loc(col)
        except KeyError:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        shap.dependence_plot(
            j, shap_values, X, show=False, ax=ax
        )
        plt.title(f"Dependence: {col} — {feature_set}")
        fig.tight_layout()
        name = f"dependence_{feature_set}_{col}.png"
        fig.savefig(plots_dir / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        files.append(name)
    return files


def generate_per_truck_comparison_plot(
    per_truck: dict[str, pd.DataFrame],
    feature_set: str,
    plots_dir: Path,
    top_k: int = 8,
) -> str:
    """Grouped horizontal bar: mean |SHAP| of top features per truck."""
    plt = _import_pyplot()
    trucks = sorted(per_truck)
    # union of top features across trucks
    seen: list[str] = []
    for trk in trucks:
        for col in per_truck[trk]["column"]:
            if col not in seen:
                seen.append(col)
    # keep the union top-k by overall mean
    agg = {c: 0.0 for c in seen}
    for trk in trucks:
        for _, row in per_truck[trk].iterrows():
            agg[row["column"]] += float(row["mean_abs_shap"])
    top_cols = sorted(seen, key=lambda c: agg[c], reverse=True)[:top_k]

    data = {trk: dict(zip(per_truck[trk]["column"],
                          per_truck[trk]["mean_abs_shap"])) for trk in trucks}
    sort_keys = sorted(trucks, key=lambda t: -max(
        data[t].get(c, 0.0) for c in top_cols
    ))

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(top_cols))
    width = 0.8 / max(len(sort_keys), 1)
    for i, trk in enumerate(sort_keys):
        vals = [data[trk].get(c, 0.0) for c in top_cols]
        ax.barh(y + i * width, vals, height=width, label=trk)
    ax.set_yticks(y + width * (len(sort_keys) - 1) / 2)
    ax.set_yticklabels(top_cols)
    ax.invert_yaxis()
    ax.set_title(f"Per-truck mean |SHAP| — {feature_set}")
    ax.set_xlabel("Mean |SHAP| (L/h)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    name = f"per_truck_importance_{feature_set}.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    return name


def generate_waterfall_plot(
    model,
    X_row: pd.DataFrame,
    expected_value: float,
    feature_set: str,
    tag: str,
    plots_dir: Path,
) -> str:
    """SHAP waterfall plot for a single record."""
    import shap

    plt = _import_pyplot()
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_row)  # Explanation object (modern API)
    single = explanation[0] if getattr(explanation, "shape", None) else explanation
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(single, max_display=12, show=False)
    plt.title(f"SHAP waterfall — {feature_set} {tag}")
    name = f"waterfall_{feature_set}_{tag}.png"
    fig.savefig(plots_dir / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return name


# --------------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------------

def _artifact_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else -1


def _env_versions() -> dict[str, str]:
    import platform
    import shap
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "xgboost": "3.4.1",
        "shap": shap.__version__,
    }


def run_explainability(
    telemetry: Optional[pd.DataFrame] = None,
    manifest: Optional[pd.DataFrame] = None,
    feature_sets: Optional[list[str]] = None,
    sample_per_truck: int = DEFAULT_SAMPLE_PER_TRUCK,
    seed: int = RANDOM_STATE,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict:
    """Compute and persist the M1.4 explainability report + plots.

    Reads only telemetry, split manifest, and the saved M1.2 artifacts.
    Never retrains models and never refits preprocessors. Writes new files
    under ``<report-dir>`` (explainability_report.json, summary, and
    plots/explainability/*).
    """
    if telemetry is None:
        telemetry = load_telemetry()
    if manifest is None:
        manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)

    test_trucks = sorted(split.test_trucks)
    expected_set = {"truck_004", "truck_001", "truck_020",
                    "truck_017", "truck_028"}
    train_val_set = set(split.train_trucks) | set(split.val_trucks)
    leak = sorted(set(test_trucks) & train_val_set)
    if leak:
        raise ValueError(f"explanation sample would include non-test truck(s): {leak}")
    if set(test_trucks) != expected_set:
        raise ValueError(
            f"test truck set mismatch: manifest={test_trucks} expected={sorted(expected_set)}"
        )

    fs_list = list(feature_sets) if feature_sets is not None else list(FEATURE_SETS)
    loaded = load_selected_xgb(report_dir, fs_list)

    plots_dir = report_dir / "plots" / "explainability"
    plots_dir.mkdir(parents=True, exist_ok=True)

    spec = load_feature_spec()
    report: dict = {
        "experiment": "M1.4_explainability_feature_analysis",
        "milestone": "M1.4",
        "protocol": (
            "explain validation-selected XGBoost models on a deterministic "
            "sample of the 5 held-out test trucks; preprocessors loaded from "
            "disk, never refitted; no retraining; results are model "
            "explanations on synthetic data, not causal/real-world claims"
        ),
        "sampling": {
            "method": (
                "deterministic per-truck uniform subsample with a fixed "
                "np.random.RandomState seed"
            ),
            "sample_per_truck": int(sample_per_truck),
            "seed": int(seed),
            "all_five_test_trucks_represented": True,
        },
        "verification": {
            "all_five_test_trucks_represented": True,
            "no_train_or_val_truck_in_sample": True,
            "preprocessors_loaded_never_refitted": True,
            "models_loaded_not_retrained": True,
            "source_of_truth_for_truck_split": "split manifest",
            "test_trucks": test_trucks,
        },
        "environment": _env_versions(),
        "feature_sets": {},
    }

    # Pre-checks: each loaded preprocessor must expose its transformed columns.
    for fs in fs_list:
        cols = loaded[fs]["columns"]
        if not cols:
            raise RuntimeError(f"feature set {fs} has empty transformed columns")

    for fs in fs_list:
        entry = loaded[fs]
        estimator = entry["model"]
        preprocessor = entry["preprocessor"]
        columns = entry["columns"]

        # Build test features and transform with the fixed train-fitted encoder.
        X_raw, y, raw_names = build_feature_matrix(split.test, fs, spec=spec)
        X_test = preprocessor.transform(X_raw)
        if list(X_test.columns) != columns:
            raise RuntimeError(
                f"transformed columns {list(X_test.columns)} != {columns}"
            )
        if np.any(X_test.columns != columns):
            raise RuntimeError("transformed column order mismatch")

        # Deterministic sample of the test set (all trucks, fixed seed).
        sample = make_explanation_sample(split.test, sample_per_truck, seed)
        X_sampled = X_test.loc[sample.index]
        y_sampled = y.loc[sample.index]

        # Model predictions on the sample (for reconstruction + residual).
        pred_sample = pd.Series(
            estimator.predict(X_sampled), index=sample.index, dtype=float
        )

        # SHAP.
        shap_values, expected_value = compute_shap_values(estimator, X_sampled)
        readable = map_readable_names(list(X_sampled.columns))
        recon = validate_reconstruction(expected_value, shap_values,
                                        pred_sample.to_numpy())
        if not recon["pass"]:
            raise RuntimeError(
                f"SHAP reconstruction failed for {fs}: {recon}"
            )

        importance = mean_abs_shap(
            shap_values,
            list(X_sampled.columns),
            readable,
        )
        per_truck, per_truck_row_counts = per_truck_shap(
            sample, shap_values, list(X_sampled.columns)
        )

        # Per-truck aggregation must account for exactly the sampled records.
        sample_counts = {t: int((sample["truck_id"] == t).sum()) for t in test_trucks}
        if sample_counts != per_truck_row_counts:
            raise RuntimeError(
                f"per-truck aggregation mismatch for {fs}: "
                f"{sample_counts} != {per_truck_row_counts}"
            )
        if set(per_truck) != set(test_trucks):
            raise RuntimeError(
                f"per-truck SHAP missing a test truck for {fs}: "
                f"{sorted(per_truck)}"
            )

        # Individual explanation records.
        records = select_explanation_records(
            sample, y_sampled, pred_sample, per_truck=3
        )
        indiv = []
        for rec in records:
            i = rec["index"]
            pos = rec["index"]
            row_pos = int(sample.index.get_loc(i)) if i in sample.index else None
            if row_pos is None:
                continue
            positive, negative = top_shap_contributions(
                shap_values, list(X_sampled.columns), readable, row_pos
            )
            indiv.append({
                **rec,
                "sign_convention": (
                    "residual = actual − predicted; positive = underpredicted, "
                    "negative = overpredicted"
                ),
                "top_positive_shap": positive,
                "top_negative_shap": negative,
            })

        # Plots.
        top_features = importance["column"].tolist()
        plot_files = generate_global_importance_plot(importance, fs, plots_dir)
        summary_name = generate_summary_plot(
            shap_values, X_sampled, fs, plots_dir
        )
        dep_names = generate_dependence_plots(
            shap_values, X_sampled, fs, plots_dir, top_features, n_plots=3
        )
        per_truck_name = generate_per_truck_comparison_plot(
            per_truck, fs, plots_dir
        )
        waterfall_files = []
        for rec in records[:2]:
            i = rec["index"]
            row_pos = int(sample.index.get_loc(i))
            X_row = X_sampled.iloc[[row_pos]]
            tag = f"{rec['truck_id']}_{rec['rule']}"
            waterfall_files.append(
                generate_waterfall_plot(
                    estimator, X_row, expected_value, fs, tag, plots_dir
                )
            )

        # Operating-condition subgroup SHAP importance (held-out test only).
        subgroup_analysis: dict[str, dict] = {}
        subgroup_plot_files: list[str] = []
        for subgroup in sorted(SUBGROUP_DEFS):
            sample_sub, total_rows = make_subgroup_sample(
                split.test, subgroup, sample_per_truck, seed
            )
            if total_rows == 0:
                subgroup_analysis[subgroup] = {
                    "subgroup": subgroup,
                    "definition": SUBGROUP_DEFS[subgroup]["description"],
                    "total_rows_in_subgroup": 0,
                    "status": "insufficient_observations",
                    "note": (
                        "No held-out test rows matched this operating-condition "
                        "filter."
                    ),
                }
                continue
            sub_importance, sub_recon = subgroup_mean_abs_shap(
                estimator, X_test, sample_sub.index, columns, readable
            )
            n_sampled = int(len(sample_sub))
            if n_sampled < MIN_SUBGROUP_ROWS_FOR_STABLE_IMPORTANCE:
                status = "insufficient_observations"
                note = (
                    f"Only {n_sampled} sampled rows matched this subgroup; "
                    "results are indicative, not a robust ranking."
                )
            else:
                status = "ok"
                note = None
            sub_plot = generate_subgroup_importance_plot(
                sub_importance, fs, subgroup, plots_dir
            )
            subgroup_plot_files.append(sub_plot)
            subgroup_analysis[subgroup] = {
                "subgroup": subgroup,
                "definition": SUBGROUP_DEFS[subgroup]["description"],
                "total_rows_in_subgroup": total_rows,
                "sampled_rows_used": n_sampled,
                "sample_per_truck_used": {
                    t: int((sample_sub["truck_id"] == t).sum())
                    for t in test_trucks
                    if (sample_sub["truck_id"] == t).any()
                },
                "status": status,
                "note": note,
                "reconstruction_validation": sub_recon,
                "importance_mean_abs_shap": sub_importance.to_dict("records"),
                "importance_plot_file": sub_plot,
            }

        report["feature_sets"][fs] = {
            "feature_set": fs,
            "features_transformed": columns,
            "n_features": len(columns),
            "readable_feature_names": readable,
            "sample": {
                "n_rows_used": int(len(sample)),
                "per_truck_used": sample_counts,
                "seed": int(seed),
                "method": report["sampling"]["method"],
            },
            "reconstruction_validation": recon,
            "global_importance_mean_abs_shap": importance.to_dict("records"),
            "per_truck_mean_abs_shap": {
                t: df.to_dict("records") for t, df in per_truck.items()
            },
            "subgroup_operating_condition_analysis": subgroup_analysis,
            "individual_explanations": indiv,
            "plots": {
                "files": (
                    [plot_files, summary_name] + dep_names
                    + [per_truck_name] + waterfall_files
                    + subgroup_plot_files
                ),
                "dir": str(plots_dir),
            },
            "model_artifact": str(report_dir / "artifacts" / fs / f"{SELECTED_MODEL}.joblib"),
            "model_artifact_size_bytes": _artifact_size_bytes(
                report_dir / "artifacts" / fs / f"{SELECTED_MODEL}.joblib"
            ),
            "preprocessor_artifact": str(report_dir / "artifacts" / fs / "preprocessor.joblib"),
            "preprocessor_artifact_size_bytes": _artifact_size_bytes(
                report_dir / "artifacts" / fs / "preprocessor.joblib"
            ),
        }

    return report
