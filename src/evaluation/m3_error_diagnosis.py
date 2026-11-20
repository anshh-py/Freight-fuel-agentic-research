"""M3.3 — cross-truck and cross-condition error diagnosis (frozen test split).

This module adds descriptive error-diagnosis diagnostics that M1.3,
M3.1 and M3.2 did not compute, reusing the frozen five-truck test split and
the saved M1.2 artifacts read-only.

Scope per milestone (descriptive only — no causal claims, no hypothesis
tests, no test-set-driven model changes):

  * Experiment 1 — truck × condition error table: MAE / RMSE / median-abs /
    P90 / P95 per truck-condition cell, compared descriptively against its
    truck-level and pooled-condition baselines; all cells preserved with
    explicit support flags (no cell is silently dropped).
  * Experiment 2 — trip-level recurrence: per truck-condition cell, per-trip
    MAE/RMSE, number of contributing trips, median and spread of trip errors,
    and the share of the cell's total squared error contributed by its worst
    trip. Used to distinguish *recurring* patterns from cases dominated by a
    single problematic trip (recurrence is never inferred from row count).
  * Experiment 3 — error distribution: median-abs, P90/P95/max, MAE/RMSE,
    and a P95-to-median ratio (with safe zero-median handling) to separate
    widespread moderate errors from concentrated extreme errors.
  * Experiment 4 — residual direction: an explicit, documented convention
    (residual = y_pred - y_true) with mean/median signed error and
    overprediction / underprediction / approximately-unbiased percentages per
    truck, condition, and supported truck-condition combination.
  * Experiment 5 — paired model comparison: full-operational versus
    telematics-only on identical rows, with a consistent delta sign
    (positive => full-operational has lower error), per truck, condition, and
    truck-condition where support permits.

Protocol guarantees:
  * models/preprocessors are loaded from disk and never refitted (reuses the
    M3.1 ``prepare_test_predictions`` interface);
  * the test set is evaluation-only — never used for fitting, tuning or
    selection;
  * no source, split manifest, M1/M2/M3.1/M3.2 artifact is modified;
  * every subgroup reports n_rows / n_trips / n_trucks and an explicit
    support classification;
  * metrics are descriptive; no hypothesis tests or significance claims;
  * synthetic data stays explicitly synthetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.m3_condition_robustness import (
    INFORMATIVE_CONDITIONS,
    MIN_SUPPORT_ROWS,
    make_extended_condition_masks,
)
from src.evaluation.m3_cross_truck_analysis import _assert_aligned

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"

# --------------------------------------------------------------------------
# Documented conventions
# --------------------------------------------------------------------------

#: Explicit residual convention for the whole M3.3 analysis.
#: residual = y_pred - y_true  =>  positive = overprediction,
#: negative = underprediction.
RESIDUAL_CONVENTION = (
    "residual = y_pred - y_true; positive => overprediction; "
    "negative => underprediction. Overprediction, underprediction and "
    "approximately-unbiased percentages overlap by definition (an "
    "approximately-unbiased row is also counted in one of the other two "
    "bins), so they do not sum to 100%."
)

#: Paired-delta sign convention (consistent with M3.1).
#: delta = MAE_telematics_only - MAE_full_operational; positive =>
#: full-operational has lower error (better). Same for RMSE.
DELTA_SIGN_CONVENTION = (
    "delta = metric_telematics_only - metric_full_operational; "
    "positive => full-operational has lower error (better)"
)

#: Tolerance used to label a row "approximately unbiased" (|residual| <= tol).
APPROX_UNBIASED_TOL_LPH = 0.05

#: Descriptive threshold for flagging an unusually high truck-condition cell:
#: cell MAE must exceed BOTH its truck-level baseline and its pooled-condition
#: baseline by at least this factor. This is a descriptive heuristic, not a
#: statistical interaction test (see the M3.3 caution).
HIGH_ERROR_RATIO = 1.25

#: Minimum number of contributing trips before a pattern may be called
#: "recurring" (a single trip can never establish recurrence).
RECURRING_MIN_TRIPS = 3

#: Worst-trip squared-error share above which a pattern is considered
#: dominated by a single problematic trip.
WORST_TRIP_DOMINANCE_SHARE = 0.5


# --------------------------------------------------------------------------
# Shared metric helpers
# --------------------------------------------------------------------------

def signed_residual_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tol: float = APPROX_UNBIASED_TOL_LPH,
) -> dict:
    """Residual direction statistics under the documented convention.

    ``residual = y_pred - y_true`` (positive => overprediction). Reports
    mean/median signed error and the percentages of rows that are
    overpredicted, underpredicted, and approximately unbiased (|residual| <=
    tol). The three percentages do not partition the data (an approximately
    unbiased row is usually also counted in over/under) — the tolerance is
    documented so the figure is interpretable.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    resid = yp - yt
    n = int(len(resid))
    if n == 0:
        return {"n": 0, "mean_signed_error": None,
                "median_signed_error": None, "overprediction_pct": None,
                "underprediction_pct": None,
                "approximately_unbiased_pct": None,
                "approx_unbiased_tolerance_lph": tol}
    over = int((resid > 0).sum())
    under = int((resid < 0).sum())
    unbiased = int((np.abs(resid) <= tol).sum())
    return {
        "n": n,
        "mean_signed_error": round(float(resid.mean()), 6),
        "median_signed_error": round(float(np.median(resid)), 6),
        "overprediction_pct": round(100.0 * over / n, 6),
        "underprediction_pct": round(100.0 * under / n, 6),
        "approximately_unbiased_pct": round(100.0 * unbiased / n, 6),
        "approx_unbiased_tolerance_lph": tol,
    }


def error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Error-shape stats: median-abs, P90/P95/max + P95/median ratio.

    ``p95_to_median_ratio`` is ``None`` (and ``zero_median=True``) when the
    median absolute error is zero — never a silent division by zero.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = int(len(yt))
    if n == 0:
        return {"n_rows": 0, "mae": None, "rmse": None,
                "median_absolute_error": None, "p90_absolute_error": None,
                "p95_absolute_error": None, "max_absolute_error": None,
                "p95_to_median_ratio": None, "zero_median": None}
    ae = np.abs(yp - yt)
    median = float(np.median(ae))
    p90 = float(np.quantile(ae, 0.90))
    p95 = float(np.quantile(ae, 0.95))
    mx = float(ae.max())
    if median > 0:
        ratio = p95 / median
        zero_median = False
    else:
        ratio = None
        zero_median = True
    return {
        "n_rows": n,
        "mae": round(float(ae.mean()), 6),
        "rmse": round(float(np.sqrt(np.mean((yp - yt) ** 2))), 6),
        "median_absolute_error": round(median, 6),
        "p90_absolute_error": round(p90, 6),
        "p95_absolute_error": round(p95, 6),
        "max_absolute_error": round(mx, 6),
        "p95_to_median_ratio": round(ratio, 6) if ratio is not None else None,
        "zero_median": zero_median,
    }


def cell_support_status(n_rows: int, n_trips: int, min_rows: int = MIN_SUPPORT_ROWS) -> str:
    """Support classification for a truck-condition cell.

    Truck-condition cells always span a single truck, so truck *count* is not
    used for support (n_trucks=1 is intrinsic and noted separately). Support
    is judged on the number of rows and the number of trips (a cell must have
    at least two trips before any recurrence statement is meaningful).
    """
    if n_rows == 0:
        return "empty"
    if n_rows < min_rows:
        return "insufficient_rows"
    if n_trips < 2:
        return "insufficient_trip_support"
    return "descriptive"


# --------------------------------------------------------------------------
# Experiment 1 — truck × condition error table
# --------------------------------------------------------------------------

def truck_condition_cells(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
    min_rows: int = MIN_SUPPORT_ROWS,
) -> list[dict]:
    """MAE/RMSE/distribution per (truck, condition) cell + baselines.

    Each cell is compared descriptively against:
      * its truck-level baseline (that truck's pooled MAE over all rows),
      * its pooled-condition baseline (that condition's pooled MAE over all
        trucks).
    ``flagged_high_error`` marks cells whose MAE exceeds BOTH baselines by at
    least ``HIGH_ERROR_RATIO``. This is a descriptive heuristic — it is NOT a
    statistical interaction test (the M3.3 caution).
    """
    cells: list[dict] = []

    # Truck-level baselines (pooled over all conditions).
    truck_baseline: dict[str, float] = {}
    for truck, g in frame.groupby("truck_id", sort=True):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        truck_baseline[str(truck)] = float(np.abs(yt - yp).mean())

    # Pooled-condition baselines (pooled over all trucks).
    cond_baseline: dict[str, float] = {}
    for label in labels:
        mask = masks[label]
        sub = frame.loc[mask]
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        cond_baseline[label] = (
            float(np.abs(yt - yp).mean()) if len(yt) else float("nan")
        )

    for label in labels:
        cond_mask = masks[label]
        for truck, g in frame.groupby("truck_id", sort=True):
            mask = cond_mask & (frame["truck_id"] == truck)
            n_rows = int(mask.sum())
            if n_rows == 0:
                cells.append(
                    {
                        "truck_id": str(truck),
                        "condition": label,
                        "support_status": "empty",
                        "n_rows": 0,
                        "n_trips": 0,
                        "n_trucks": 1,
                        "mae": None,
                        "rmse": None,
                        "median_absolute_error": None,
                        "p90_absolute_error": None,
                        "p95_absolute_error": None,
                        "max_absolute_error": None,
                        "truck_baseline_mae": round(truck_baseline[str(truck)], 6),
                        "condition_baseline_mae": (
                            round(cond_baseline[label], 6)
                            if np.isfinite(cond_baseline[label]) else None
                        ),
                        "mae_ratio_to_truck_baseline": None,
                        "mae_ratio_to_condition_baseline": None,
                        "flagged_high_error": False,
                    }
                )
                continue

            sub = frame.loc[mask]
            yt = sub["y_true"].to_numpy(dtype=float)
            yp = sub["y_pred"].to_numpy(dtype=float)
            n_trips = int(sub["trip_id"].nunique())
            dist = error_distribution(yt, yp)
            mae = dist["mae"]

            t_ratio = mae / truck_baseline[str(truck)] if truck_baseline[str(truck)] > 0 else None
            c_base = cond_baseline[label]
            c_ratio = (mae / c_base) if (np.isfinite(c_base) and c_base > 0) else None

            flagged = bool(
                t_ratio is not None
                and t_ratio >= HIGH_ERROR_RATIO
                and c_ratio is not None
                and c_ratio >= HIGH_ERROR_RATIO
            )

            cells.append(
                {
                    "truck_id": str(truck),
                    "condition": label,
                    "support_status": cell_support_status(n_rows, n_trips, min_rows),
                    "n_rows": n_rows,
                    "n_trips": n_trips,
                    "n_trucks": 1,
                    "mae": mae,
                    "rmse": dist["rmse"],
                    "median_absolute_error": dist["median_absolute_error"],
                    "p90_absolute_error": dist["p90_absolute_error"],
                    "p95_absolute_error": dist["p95_absolute_error"],
                    "max_absolute_error": dist["max_absolute_error"],
                    "truck_baseline_mae": round(truck_baseline[str(truck)], 6),
                    "condition_baseline_mae": (
                        round(c_base, 6) if np.isfinite(c_base) else None
                    ),
                    "mae_ratio_to_truck_baseline": (
                        round(t_ratio, 6) if t_ratio is not None else None
                    ),
                    "mae_ratio_to_condition_baseline": (
                        round(c_ratio, 6) if c_ratio is not None else None
                    ),
                    "flagged_high_error": flagged,
                }
            )

    # Preserve all cells; sort descriptive ones by MAE descending for review.
    cells.sort(key=lambda c: (c["mae"] is not None, c["mae"] if c["mae"] is not None else -1.0), reverse=True)
    return cells


# --------------------------------------------------------------------------
# Experiment 2 — trip-level recurrence
# --------------------------------------------------------------------------

def _per_trip_stats(sub: pd.DataFrame) -> dict:
    """Per-trip MAE/RMSE, counts, and worst-trip squared-error share."""
    g = sub.groupby("trip_id", sort=True)
    trip_errors: list[dict] = []
    trip_ss: list[float] = []
    for trip_id, gg in g:
        yt = gg["y_true"].to_numpy(dtype=float)
        yp = gg["y_pred"].to_numpy(dtype=float)
        mae = float(np.abs(yt - yp).mean())
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        ss = float(((yp - yt) ** 2).sum())
        trip_errors.append({"trip_id": str(trip_id), "mae": mae,
                            "rmse": rmse, "n_rows": int(len(yt))})
        trip_ss.append(ss)
    n_trips = len(trip_errors)
    total_ss = float(sum(trip_ss)) if trip_ss else 0.0
    worst_share = (
        max(trip_ss) / total_ss if total_ss > 0 else None
    )
    mae_vals = np.array([t["mae"] for t in trip_errors], dtype=float)
    return {
        "n_trips": n_trips,
        "trips": trip_errors,
        "median_trip_mae": round(float(np.median(mae_vals)), 6) if n_trips else None,
        "mean_trip_mae": round(float(mae_vals.mean()), 6) if n_trips else None,
        "std_trip_mae": round(float(mae_vals.std()), 6) if n_trips else None,
        "min_trip_mae": round(float(mae_vals.min()), 6) if n_trips else None,
        "max_trip_mae": round(float(mae_vals.max()), 6) if n_trips else None,
        "worst_trip_squared_error_share": (
            round(worst_share, 6) if worst_share is not None else None
        ),
    }


def trip_recurrence(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
    min_rows: int = MIN_SUPPORT_ROWS,
) -> list[dict]:
    """Trip-level recurrence per truck-condition cell.

    A pattern is classified as:
      * ``recurring_pattern`` — >= RECURRING_MIN_TRIPS contributing trips and
        the worst trip contributes <= 50% of the cell's squared error (the
        error is spread across multiple trips, not one trip). This describes
        recurrence across at least RECURRING_MIN_TRIPS trips with no single trip
        contributing > 50% of the cell's squared error; it does NOT imply high
        error magnitude (that is the separate ``flagged_high_error`` signal);
      * ``dominated_by_worst_trip`` — the worst trip contributes > 50% of the
        cell's squared error (one problematic trip drives the cell), or fewer
        than RECURRING_MIN_TRIPS trips;
      * ``insufficient`` — too few rows/trips to say anything.
    This never treats row count alone as evidence of recurrence.
    """
    out: list[dict] = []
    for label in labels:
        cond_mask = masks[label]
        for truck, g in frame.groupby("truck_id", sort=True):
            mask = cond_mask & (frame["truck_id"] == truck)
            n_rows = int(mask.sum())
            if n_rows == 0:
                continue
            sub = frame.loc[mask]
            n_trips = int(sub["trip_id"].nunique())
            status = cell_support_status(n_rows, n_trips, min_rows)
            per = _per_trip_stats(sub)

            if status in ("empty", "insufficient_rows", "insufficient_trip_support"):
                pattern = "insufficient"
            else:
                worst = per["worst_trip_squared_error_share"]
                if worst is None:
                    pattern = "insufficient"
                elif n_trips >= RECURRING_MIN_TRIPS and worst <= WORST_TRIP_DOMINANCE_SHARE:
                    pattern = "recurring_pattern"
                else:
                    pattern = "dominated_by_worst_trip"

            out.append(
                {
                    "truck_id": str(truck),
                    "condition": label,
                    "n_rows": n_rows,
                    "n_trips": n_trips,
                    "n_trucks": 1,
                    "support_status": status,
                    "pattern": pattern,
                    "median_trip_mae": per["median_trip_mae"],
                    "mean_trip_mae": per["mean_trip_mae"],
                    "std_trip_mae": per["std_trip_mae"],
                    "min_trip_mae": per["min_trip_mae"],
                    "max_trip_mae": per["max_trip_mae"],
                    "worst_trip_squared_error_share": per["worst_trip_squared_error_share"],
                    "trips": per["trips"],
                }
            )
    return out


# --------------------------------------------------------------------------
# Experiment 3 — error distribution (pooled conditions + supported cells)
# --------------------------------------------------------------------------

def pooled_condition_error_distribution(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
) -> list[dict]:
    """Error-shape stats per pooled condition (all trucks)."""
    out: list[dict] = []
    for label in labels:
        sub = frame.loc[masks[label]]
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        dist = error_distribution(yt, yp)
        out.append(
            {
                "condition": label,
                "n_rows": dist["n_rows"],
                "n_trips": int(sub["trip_id"].nunique()),
                "n_trucks": int(sub["truck_id"].nunique()),
                **dist,
            }
        )
    return out


def supported_cell_error_distribution(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
    min_rows: int = MIN_SUPPORT_ROWS,
) -> list[dict]:
    """Error-shape stats for descriptive (supported) truck-condition cells."""
    out: list[dict] = []
    for label in labels:
        cond_mask = masks[label]
        for truck, g in frame.groupby("truck_id", sort=True):
            mask = cond_mask & (frame["truck_id"] == truck)
            n_rows = int(mask.sum())
            sub = frame.loc[mask]
            n_trips = int(sub["trip_id"].nunique())
            if cell_support_status(n_rows, n_trips, min_rows) != "descriptive":
                continue
            yt = sub["y_true"].to_numpy(dtype=float)
            yp = sub["y_pred"].to_numpy(dtype=float)
            dist = error_distribution(yt, yp)
            out.append({"truck_id": str(truck), "condition": label, **dist})
    return out


# --------------------------------------------------------------------------
# Experiment 4 — residual direction (per truck / condition / cell)
# --------------------------------------------------------------------------

def residual_direction_table(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
    min_rows: int = MIN_SUPPORT_ROWS,
) -> dict:
    """Residual direction per truck, pooled condition, and supported cells.

    Returns ``{"per_truck": [...], "per_condition": [...],
    "per_cell": [...]}``. Only descriptive cells are included in ``per_cell``
    (others have too little support for a stable bias figure). Bias reversals
    are identified descriptively via the sign of the median signed error.
    """
    per_truck: list[dict] = []
    for truck, g in frame.groupby("truck_id", sort=True):
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        s = signed_residual_stats(yt, yp)
        per_truck.append({"truck_id": str(truck), "n_rows": s["n"], **s})

    per_condition: list[dict] = []
    for label in labels:
        sub = frame.loc[masks[label]]
        yt = sub["y_true"].to_numpy(dtype=float)
        yp = sub["y_pred"].to_numpy(dtype=float)
        s = signed_residual_stats(yt, yp)
        per_condition.append(
            {
                "condition": label,
                "n_rows": s["n"],
                "n_trips": int(sub["trip_id"].nunique()),
                "n_trucks": int(sub["truck_id"].nunique()),
                **s,
            }
        )

    per_cell: list[dict] = []
    for label in labels:
        cond_mask = masks[label]
        for truck, g in frame.groupby("truck_id", sort=True):
            mask = cond_mask & (frame["truck_id"] == truck)
            n_rows = int(mask.sum())
            sub = frame.loc[mask]
            n_trips = int(sub["trip_id"].nunique())
            if cell_support_status(n_rows, n_trips, min_rows) != "descriptive":
                continue
            yt = sub["y_true"].to_numpy(dtype=float)
            yp = sub["y_pred"].to_numpy(dtype=float)
            s = signed_residual_stats(yt, yp)
            per_cell.append({"truck_id": str(truck), "condition": label,
                             "n_rows": s["n"], "n_trips": n_trips,
                             "n_trucks": 1, **s})

    return {
        "residual_convention": RESIDUAL_CONVENTION,
        "per_truck": per_truck,
        "per_condition": per_condition,
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Experiment 5 — paired model comparison (full vs telematics-only)
# --------------------------------------------------------------------------

def paired_condition_comparison(
    full: pd.DataFrame,
    tel: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
    min_rows: int = MIN_SUPPORT_ROWS,
) -> dict:
    """Paired full-operational vs telematics-only deltas per truck/condition.

    ``delta = metric_telematics_only - metric_full_operational``; positive =>
    full-operational has lower error. Only identical rows are compared
    (``_assert_aligned``). Nonzero deltas are flagged with
    ``paired_delta_nonzero``; this is a descriptive comparison, not a
    superiority claim.
    """
    _assert_aligned(full, tel)
    y = full["y_true"].to_numpy(dtype=float)
    pf = full["y_pred"].to_numpy(dtype=float)
    pt = tel["y_pred"].to_numpy(dtype=float)

    def _paired(sub_mask: np.ndarray, label: str, truck: Optional[str]) -> dict:
        yy = y[sub_mask]
        ppf = pf[sub_mask]
        ppt = pt[sub_mask]
        mae_full = float(np.abs(yy - ppf).mean()) if len(yy) else float("nan")
        mae_tel = float(np.abs(yy - ppt).mean()) if len(yy) else float("nan")
        rmse_full = float(np.sqrt(np.mean((ppf - yy) ** 2))) if len(yy) else float("nan")
        rmse_tel = float(np.sqrt(np.mean((ppt - yy) ** 2))) if len(yy) else float("nan")
        d_mae = mae_tel - mae_full
        d_rmse = rmse_tel - rmse_full
        return {
            "group": label,
            "truck_id": truck,
            "n_rows": int(len(yy)),
            "mae_full_operational": round(mae_full, 6),
            "mae_telematics_only": round(mae_tel, 6),
            "delta_mae": round(d_mae, 6),
            "rmse_full_operational": round(rmse_full, 6),
            "rmse_telematics_only": round(rmse_tel, 6),
            "delta_rmse": round(d_rmse, 6),
            "paired_delta_nonzero": bool(abs(d_mae) > 1e-6),
        }

    pooled = _paired(np.ones(len(y), dtype=bool), "all", None)

    per_truck = [_paired(y == t, str(t), str(t)) for t in np.unique(full["truck_id"])]

    per_condition = []
    for label in labels:
        mask = masks[label].to_numpy(dtype=bool)
        per_condition.append(_paired(mask, label, None))

    per_cell = []
    for label in labels:
        cond_mask = masks[label]
        for truck in np.unique(full["truck_id"]):
            mask = (cond_mask & (full["truck_id"] == truck)).to_numpy(dtype=bool)
            n_rows = int(mask.sum())
            sub = full.loc[mask]
            n_trips = int(sub["trip_id"].nunique()) if n_rows else 0
            if cell_support_status(n_rows, n_trips, min_rows) != "descriptive":
                continue
            per_cell.append(_paired(mask, label, str(truck)))

    return {
        "delta_sign_convention": DELTA_SIGN_CONVENTION,
        "pooled": pooled,
        "per_truck": per_truck,
        "per_condition": per_condition,
        "per_cell": per_cell,
    }


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def build_diagnosis_report(
    frames: dict[str, pd.DataFrame],
    feature_sets: tuple[str, ...],
) -> dict:
    """Build the full M3.3 report for all feature sets.

    ``frames`` must already have operating columns attached (via the M3.2
    ``attach_operating_columns``), so which ``make_extended_condition_masks``
    can run. Returns a per-feature-set diagnostic body plus the paired
    comparison (which needs both feature sets together).
    """
    per_fs: dict = {}
    for fs in feature_sets:
        frame = frames[fs]
        masks = make_extended_condition_masks(frame)
        labels = list(INFORMATIVE_CONDITIONS)
        per_fs[fs] = {
            "n_rows": int(len(frame)),
            "n_trips": int(frame["trip_id"].nunique()),
            "n_trucks": int(frame["truck_id"].nunique()),
            "truck_condition_cells": truck_condition_cells(frame, masks, labels),
            "trip_recurrence": trip_recurrence(frame, masks, labels),
            "pooled_condition_error_distribution": (
                pooled_condition_error_distribution(frame, masks, labels)
            ),
            "supported_cell_error_distribution": (
                supported_cell_error_distribution(frame, masks, labels)
            ),
            "residual_direction": residual_direction_table(frame, masks, labels),
            "high_error_ratio": HIGH_ERROR_RATIO,
            "recurring_min_trips": RECURRING_MIN_TRIPS,
            "worst_trip_dominance_share": WORST_TRIP_DOMINANCE_SHARE,
            "approx_unbiased_tolerance_lph": APPROX_UNBIASED_TOL_LPH,
        }

    full = frames[feature_sets[0]]
    tel = frames[feature_sets[1]]
    masks = make_extended_condition_masks(full)
    labels = list(INFORMATIVE_CONDITIONS)
    paired = paired_condition_comparison(full, tel, masks, labels)

    return {
        "feature_sets": per_fs,
        "paired_comparison": paired,
        "residual_convention": RESIDUAL_CONVENTION,
    }
