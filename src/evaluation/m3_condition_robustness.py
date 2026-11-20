"""M3.2 — operating-condition robustness diagnostics (frozen test split).

This module adds operating-condition diagnostics that M1.3 and M3.1 did not
compute, reusing the frozen five-truck test split and the saved M1.2
artifacts (preprocessors + validation-selected XGBoost models) read-only.

What M1.3 already measured (payload duplicated here):
  * pooled / per-truck / per-class / macro metrics,
  * row-level speed / road-grade / acceleration bands (pooled across trucks),
  * engine on/off, residual diagnostics.

What M3.1 added (not duplicated):
  * per-trip metrics, tail quantiles, paired full-vs-telematics deltas,
    trip-clustered bootstrap, variance decomposition.

What M3.2 genuinely adds:
  * trip-level route-type metrics (highway / city / urban / mountainous /
    mixed) — route_type lives in trip metadata, never used before,
  * idle versus engine-off-stopped disambiguation,
  * payload load-factor regimes (payload_kg / gross_vehicle_weight_kg),
  * combined operating conditions (highway+heavy payload, uphill+accelerating,
    downhill+braking, steep-uphill+heavy payload),
  * per-truck × condition robustness (to expose pooled masking),
  * error concentration (squared-error share, row share) by condition and
    route type.

Protocol guarantees:
  * no training / refitting / tuning / test-set-driven selection;
  * frozen test split and M1/M3.1 artifacts are never modified;
  * every subgroup reports n_rows / n_trips / n_trucks and an explicit
    insufficient-support flag (row count alone is never treated as proof);
  * all metrics are descriptive — no hypothesis tests, no significance claims;
  * synthetic data is kept explicitly synthetic (no real-world claims).

Band edges for speed / grade / acceleration reuse the M1.3 definitions
verbatim (see ``make_condition_masks`` in ``test_evaluation.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.test_evaluation import make_condition_masks

# --------------------------------------------------------------------------
# Sample-size safeguards
# --------------------------------------------------------------------------

#: Minimum rows for a subgroup to be considered "supported".
MIN_SUPPORT_ROWS = 500
#: Minimum distinct trucks for a subgroup to be considered "supported"
#: (a single truck must never back a generalization claim).
MIN_SUPPORT_TRUCKS = 2

#: Number of held-out test trucks in the frozen split (5). A subgroup that
#: spans fewer than this many trucks has only partial truck coverage.
FULL_TEST_TRUCKS = 5

#: Support status note. Support is classified (not a single boolean) so that
#: "too few rows" is never conflated with "only limited truck coverage":
#:   * empty                          -> insufficient_support=True
#:   * insufficient_rows              -> insufficient_support=True
#:   * insufficient_truck_coverage    -> insufficient_support=True
#:   * descriptive_limited_truck_coverage -> insufficient_support=False,
#:     limited_truck_coverage=True (reported descriptively, with a warning)
#:   * descriptive_full_truck_coverage -> insufficient_support=False
INSUFFICIENT_SUPPORT_NOTE = (
    "Support is classified, not a single boolean. A subgroup is 'insufficient' "
    "(insufficient_support=True) when it is empty, has fewer than "
    f"{MIN_SUPPORT_ROWS} rows, or spans fewer than {MIN_SUPPORT_TRUCKS} distinct "
    "trucks. A subgroup with enough rows and at least "
    f"{MIN_SUPPORT_TRUCKS} trucks but fewer than all {FULL_TEST_TRUCKS} test "
    "trucks is reported DESCRIPTIVELY with limited_truck_coverage=True and an "
    "explicit warning: it does not support population-level generalization. "
    "Row count alone never proves reliability."
)

# --------------------------------------------------------------------------
# Payload load-factor regimes
# --------------------------------------------------------------------------

#: Load-factor bins (payload_kg / gross_vehicle_weight_kg). Boundaries are
#: intentionally explicit and mutually exclusive:
#:   light: lf <= 0.33
#:   medium: 0.33 < lf <= 0.66
#:   heavy: 0.66 < lf
#: Rows with an invalid/missing denominator (gvw <= 0 or NaN) fall into a
#: separate ``payload_invalid_gvw`` bucket and are never silently dropped.
PAYLOAD_BIN_EDGES = (0.33, 0.66)


def payload_load_factor(
    payload_kg: pd.Series,
    gross_vehicle_weight_kg: pd.Series,
) -> pd.Series:
    """Return load factor ``payload / gvw``; invalid denominators -> NaN."""
    payload = pd.to_numeric(payload_kg, errors="coerce").astype(float)
    gvw = pd.to_numeric(gross_vehicle_weight_kg, errors="coerce").astype(float)
    lf = payload / gvw
    lf[(gvw <= 0) | gvw.isna()] = np.nan
    return lf


def payload_regime_label(lf: pd.Series) -> pd.Series:
    """Map a load-factor series to regime labels (invalid -> ``invalid_gvw``)."""
    labels = pd.Series("payload_invalid_gvw", index=lf.index, dtype=object)
    labels[lf <= PAYLOAD_BIN_EDGES[0]] = "payload_light_le_0.33"
    medium = (lf > PAYLOAD_BIN_EDGES[0]) & (lf <= PAYLOAD_BIN_EDGES[1])
    labels[medium] = "payload_medium_gt0.33_le0.66"
    labels[lf > PAYLOAD_BIN_EDGES[1]] = "payload_heavy_gt0.66"
    return labels


# --------------------------------------------------------------------------
# Operating-condition masks (M1 bands reused verbatim + new conditions)
# --------------------------------------------------------------------------

def make_extended_condition_masks(
    frame: pd.DataFrame,
) -> dict[str, pd.Series]:
    """Return ``{label: boolean mask}`` over ``frame``.

    Includes the M1.3 masks verbatim (via ``make_condition_masks``), plus the
    new M3.2 masks: idle, engine-off-stopped, payload regimes, and combined
    conditions.
    """
    masks: dict[str, pd.Series] = dict(make_condition_masks(frame))

    ign = frame["ignition_status"]
    speed = frame["speed_kph"]
    grade = frame["road_grade_pct"]
    accel = frame["acceleration_mps2"]

    # Idle vs engine-off-stopped (speed < 0.5 km/h reused from M1 band edge).
    stopped = speed < 0.5
    masks["idle_engine_on_stopped"] = (ign == 1) & stopped
    masks["engine_off_stopped"] = (ign == 0) & stopped
    masks["not_idle_not_engine_off_stopped"] = ~(
        masks["idle_engine_on_stopped"] | masks["engine_off_stopped"]
    )

    # Payload regimes.
    if "payload_load_factor" not in frame.columns:
        lf = payload_load_factor(
            frame.get("payload_kg"), frame.get("gross_vehicle_weight_kg")
        )
        frame = frame.assign(payload_load_factor=lf)
    labels = payload_regime_label(frame["payload_load_factor"])
    for reg in ("payload_light_le_0.33", "payload_medium_gt0.33_le0.66",
                "payload_heavy_gt0.66", "payload_invalid_gvw"):
        masks[reg] = labels == reg

    # Combined conditions (reusing M1 band edges).
    masks["combined_highway_heavy_payload"] = (
        (speed >= 60) & (labels == "payload_heavy_gt0.66")
    )
    masks["combined_uphill_accelerating"] = (
        (grade >= 0.3) & (accel >= 0.5)
    )
    masks["combined_downhill_braking"] = (
        (grade <= -0.3) & (accel <= -0.5)
    )
    masks["combined_steep_uphill_heavy_payload"] = (
        (grade >= 1.5) & (labels == "payload_heavy_gt0.66")
    )

    return masks


# --------------------------------------------------------------------------
# Subgroup metrics with coverage + insufficient-support flag
# --------------------------------------------------------------------------

def subgroup_metrics(
    frame: pd.DataFrame,
    mask: pd.Series,
    min_rows: int = MIN_SUPPORT_ROWS,
    min_trucks: int = MIN_SUPPORT_TRUCKS,
) -> dict:
    """Metrics for one subgroup: MAE/RMSE + coverage + support flag."""
    sub = frame.loc[mask]
    y = sub["y_true"].to_numpy(dtype=float)
    p = sub["y_pred"].to_numpy(dtype=float)
    n = int(len(y))
    mae = float(np.abs(y - p).mean()) if n else float("nan")
    rmse = float(np.sqrt(np.mean((y - p) ** 2))) if n else float("nan")
    n_trucks = int(sub["truck_id"].nunique()) if n else 0
    n_trips = int(sub["trip_id"].nunique()) if n else 0

    if n == 0:
        status = "empty"
    elif n < min_rows:
        status = "insufficient_rows"
    elif n_trucks < min_trucks:
        status = "insufficient_truck_coverage"
    elif n_trucks < FULL_TEST_TRUCKS:
        status = "descriptive_limited_truck_coverage"
    else:
        status = "descriptive_full_truck_coverage"

    insufficient_support = status in (
        "empty",
        "insufficient_rows",
        "insufficient_truck_coverage",
    )
    limited_truck_coverage = status == "descriptive_limited_truck_coverage"

    return {
        "n_rows": n,
        "n_trips": n_trips,
        "n_trucks": n_trucks,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "support_status": status,
        "limited_truck_coverage": limited_truck_coverage,
        "insufficient_support": insufficient_support,
    }


def _metrics_for_label(frame: pd.DataFrame, label: str, mask: pd.Series) -> dict:
    entry = subgroup_metrics(frame, mask)
    return {"condition": label, **entry}


# --------------------------------------------------------------------------
# Route-type metrics (trip level)
# --------------------------------------------------------------------------

def route_type_table(frame: pd.DataFrame) -> list[dict]:
    """MAE/RMSE per route_type with coverage (rows/trips/trucks)."""
    rows: list[dict] = []
    if "route_type" not in frame.columns:
        return rows
    for route, g in frame.groupby("route_type", sort=True):
        sub = g
        y = sub["y_true"].to_numpy(dtype=float)
        p = sub["y_pred"].to_numpy(dtype=float)
        rows.append(
            {
                "route_type": str(route),
                "n_rows": int(len(y)),
                "n_trips": int(sub["trip_id"].nunique()),
                "n_trucks": int(sub["truck_id"].nunique()),
                "mae": round(float(np.abs(y - p).mean()), 6),
                "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 6),
            }
        )
    return sorted(rows, key=lambda r: r["mae"], reverse=True)


# --------------------------------------------------------------------------
# Error concentration
# --------------------------------------------------------------------------

def error_concentration(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
) -> list[dict]:
    """Squared-error share + row share per condition.

    Denominator is the pooled total squared error over ALL test rows, so each
    share is interpretable as the fraction of total test error that lies in
    that condition. Row share is reported alongside so the two are read
    together (a condition can hold a large error share simply because it
    holds a large row share).
    """
    y = frame["y_true"].to_numpy(dtype=float)
    p = frame["y_pred"].to_numpy(dtype=float)
    sq_err = (y - p) ** 2
    total_ss = float(sq_err.sum())
    total_n = int(len(sq_err))

    out: list[dict] = []
    for label in labels:
        mask = masks[label]
        sub_ss = float(sq_err[mask].sum())
        n = int(mask.sum())
        out.append(
            {
                "condition": label,
                "squared_error_share": round(sub_ss / total_ss, 6)
                if total_ss > 0
                else None,
                "row_share": round(n / total_n, 6) if total_n else None,
                "squared_error_sum": round(sub_ss, 6),
                "n_rows": n,
                "mae": round(float(np.abs(y[mask] - p[mask]).mean()), 6)
                if n
                else None,
                "rmse": round(float(np.sqrt(np.mean(sq_err[mask]))), 6)
                if n
                else None,
            }
        )
    return sorted(out, key=lambda r: r["squared_error_share"] or -1.0, reverse=True)


# --------------------------------------------------------------------------
# Per-truck × condition |
# --------------------------------------------------------------------------

def per_truck_condition(
    frame: pd.DataFrame,
    masks: dict[str, pd.Series],
    labels: list[str],
) -> dict[str, dict]:
    """MAE/RMSE per truck for each selected condition (exposes pooled masking)."""
    out: dict[str, dict] = {}
    for label in labels:
        sub = frame.loc[masks[label]]
        per_truck: dict[str, dict] = {}
        for truck, g in sub.groupby("truck_id", sort=True):
            y = g["y_true"].to_numpy(dtype=float)
            p = g["y_pred"].to_numpy(dtype=float)
            per_truck[str(truck)] = {
                "n_rows": int(len(y)),
                "mae": round(float(np.abs(y - p).mean()), 6),
                "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 6),
            }
        out[label] = {"n_rows": int(len(sub)), "per_truck": per_truck}
    return out


# --------------------------------------------------------------------------
# High-level " table that reuses the M3.1 prepared frames
# --------------------------------------------------------------------------

#: Telemetry columns required for operating-condition diagnosis. The M3.1
#: prepared frames carry only ``trip_id/truck_id/vehicle_class/route_type/
#: y_true/y_pred``; these columns are attached from telemetry by index.
OPERATING_COLUMNS = [
    "speed_kph",
    "road_grade_pct",
    "acceleration_mps2",
    "ignition_status",
    "payload_kg",
    "gross_vehicle_weight_kg",
]


def attach_operating_columns(
    frame: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """Attach operating-condition columns from ``telemetry`` by index.

    The prepared frames share the telemetry index, so we align on index and
    assert no missing values appear in the attached columns (a missing value
    here would silently change subgroup coverage).
    """
    missing = set(OPERATING_COLUMNS) - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"telemetry missing operating columns: {sorted(missing)}"
        )
    out = frame.copy()
    attached = telemetry.loc[frame.index, OPERATING_COLUMNS]
    if len(attached) != len(frame):
        raise ValueError("operating-column attach changed row count")
    if attached.isna().any().any():
        raise ValueError(
            "operating columns contain missing values on test rows "
            "(would silently change subgroup coverage)"
        )
    for col in OPERATING_COLUMNS:
        out[col] = attached[col].to_numpy()
    return out


INFORMATIVE_CONDITIONS = [
    "speed_ge_60_highway",
    "idle_engine_on_stopped",
    "engine_off_stopped",
    "accel_ge_0.5_accelerating",
    "accel_le_-0.5_braking",
    "grade_ge_1.5_steep_uphill",
    "payload_heavy_gt0.66",
    "combined_highway_heavy_payload",
    "combined_uphill_accelerating",
    "combined_downhill_braking",
    "combined_steep_uphill_heavy_payload",
]


def build_condition_report(frame: pd.DataFrame) -> dict:
    """Compute the full set of M3.2 diagnostics for one feature set's frame."""
    masks = make_extended_condition_masks(frame)

    condensed_metric_labels = [
        "all",
        "engine_on",
        "engine_off",
        "speed_ge_60_highway",
        "idle_engine_on_stopped",
        "engine_off_stopped",
        "accel_ge_0.5_accelerating",
        "accel_le_-0.5_braking",
        "grade_ge_1.5_steep_uphill",
        "payload_light_le_0.33",
        "payload_medium_gt0.33_le0.66",
        "payload_heavy_gt0.66",
        "payload_invalid_gvw",
        "combined_highway_heavy_payload",
        "combined_uphill_accelerating",
        "combined_downhill_braking",
        "combined_steep_uphill_heavy_payload",
    ]

    primary_table = [
        _metrics_for_label(frame, label, masks[label])
        for label in condensed_metric_labels
    ]

    return {
        "n_rows": int(len(frame)),
        "n_trips": int(frame["trip_id"].nunique()),
        "n_trucks": int(frame["truck_id"].nunique()),
        "route_type_table": route_type_table(frame),
        "conditions": primary_table,
        "error_concentration": error_concentration(
            frame, masks, INFORMATIVE_CONDITIONS
        ),
        "per_truck_condition": per_truck_condition(
            frame, masks, INFORMATIVE_CONDITIONS
        ),
        "insufficient_support_note": INSUFFICIENT_SUPPORT_NOTE,
        "payload_bin_edges": list(PAYLOAD_BIN_EDGES),
        "note": (
            "All metrics are descriptive; no hypothesis tests. The combined "
            "conditions reuse M1.3 band edges (speed>=60, grade>=0.3 / "
            "<=-0.3 / >=1.5, accel>=0.5 / <=-0.5). Payload load factor = "
            "payload_kg / gross_vehicle_weight_kg; invalid denominators are "
            "bucketed separately (never dropped)."
        ),
    }


def run_m3_condition_analysis(
    frames: dict,
    feature_sets: tuple[str, ...],
) -> dict:
    """Wrap per-feature-set condition reports into one M3.2 report body."""
    return {
        feature_set: build_condition_report(frames[feature_set])
        for feature_set in feature_sets
    }
