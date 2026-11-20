"""M3.1 runner — cross-truck generalization diagnostics (read-only).

Loads the frozen five-truck test split and the saved M1.2 artifacts, then
computes and writes the M3.1 diagnostics:

  * ``reports/m3/cross_truck_generalization_report.json``,
  * ``reports/m3/cross_truck_generalization_summary.md``.

No model is trained/refitted/tuned; no dataset or frozen artifact is modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.m3_cross_truck_analysis import (
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_REPORT_DIR,
    EXPECTED_TEST_TRUCKS,
    FEATURE_SETS,
    paired_grouped_delta,
    per_trip_metrics,
    per_truck_tail,
    pooled_paired_delta,
    prepare_test_predictions,
    tail_quantiles,
    trip_clustered_bootstrap,
    variance_decomposition,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M1_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"
M3_REPORT_DIR = PROJECT_ROOT / "reports" / "m3"


def _pooled_tail(
    frame: pd.DataFrame,
) -> dict:
    """Pooled tail-error quantiles over all test rows of a feature set."""
    ae = np.abs(frame["y_true"].astype(float) - frame["y_pred"].astype(float)).to_numpy()
    return {
        **tail_quantiles(ae),
        "n_rows": int(len(ae)),
        "mae": round(float(ae.mean()), 6),
    }


def _worst_trips(per_trip: list[dict], k: int = 5) -> list[dict]:
    """Top-k trips by MAE (worst first)."""
    return sorted(per_trip, key=lambda r: r["mae"], reverse=True)[:k]


def build_report() -> dict:
    prep = prepare_test_predictions(report_dir=M1_REPORT_DIR)
    frames = prep["frames"]
    full = frames["full_operational"]
    tel = frames["telematics_only"]

    # --- per-feature-set diagnostics --------------------------------
    feature_sets_report: dict[str, dict] = {}
    for fs in FEATURE_SETS:
        frame = frames[fs]
        per_trip = per_trip_metrics(frame)
        feature_sets_report[fs] = {
            "test_rows": int(len(frame)),
            "n_trips": int(frame["trip_id"].nunique()),
            "pooled": _pooled_tail(frame),
            "per_truck_tail": per_truck_tail(frame),
            "per_trip": per_trip,
            "worst_trips_by_mae": _worst_trips(per_trip),
            "variance_decomposition": variance_decomposition(frame),
        }

    # --- paired comparison (same rows) ------------------------------
    pooled_delta = pooled_paired_delta(full, tel)
    per_trip_delta = paired_grouped_delta(full, tel, group_col="trip_id")
    per_truck_delta = paired_grouped_delta(full, tel, group_col="truck_id")

    # --- bootstrap ----------------------------------------------------
    bootstrap = trip_clustered_bootstrap(full, tel)

    # --- verification / reconciliation ------------------------------
    verification = {
        "test_rows": prep["test_rows"],
        "test_trucks": prep["test_trucks"],
        "all_expected_test_trucks_present": set(prep["test_trucks"])
        == set(EXPECTED_TEST_TRUCKS),
        "n_test_trips": prep["n_trips"],
        "n_trips_expected": 96,
        "full_and_telematics_same_rows": int(len(full)) == int(len(tel))
        and full.index.equals(tel.index),
        "per_feature_set_row_counts_match": all(
            len(frames[fs]) == prep["test_rows"] for fs in FEATURE_SETS
        ),
        "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "preprocessors_loaded_not_refitted": True,
        "models_loaded_trained": True,
    }

    report: dict = {
        "experiment": "M3.1_cross_truck_generalization",
        "milestone": "M3.1",
        "protocol": (
            "read-only: reuse frozen 5-truck test split and saved M1.2 "
            "artifacts; no training/refitting/tuning; test used only for "
            "evaluation; trip-level resampling (not telemetry rows)"
        ),
        "analysis_frame": (
            "conditional on the five frozen test trucks (truck_001, truck_004, "
            "truck_017, truck_020, truck_028); does NOT establish population-"
            "level generalization across trucks"
        ),
        "selected_models_from_validation": prep["selected_models"],
        "verification": verification,
        "pooled_paired_delta": pooled_delta,
        "per_truck_paired_delta": per_truck_delta,
        "per_trip_paired_delta": per_trip_delta,
        "bootstrap": bootstrap,
        "feature_sets": feature_sets_report,
        "limitations": [
            "Synthetic telemetry: results are pipeline validation, not observed "
            "real-world Indian truck telemetry.",
            "Only five test trucks: truck-level estimates (tail, variance "
            "decomposition, macro) are descriptive and unstable; not population "
            "estimates.",
            "Trip-clustered bootstrap resamples the 96 test trips and therefore "
            "reflects uncertainty conditional on these five trucks only.",
            "Repeated trips of the same truck are not fully independent; we "
            "mitigate this by resampling at the trip level rather than rows.",
            "49T class is represented by a single truck; no class-level "
            "generalization claims are made.",
        ],
    }
    return report


def _write_json(report: dict) -> Path:
    report_path = M3_REPORT_DIR / "cross_truck_generalization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path


def _fmt_ci(entry: dict) -> str:
    return f"[{entry['ci95_lower']:.4f}, {entry['ci95_upper']:.4f}]"


def _write_markdown(report: dict) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# M3.1 Cross-Truck Generalization Analysis\n")
    add(f"**Milestone:** {report['milestone']}\n")
    add(f"**Analysis frame:** {report['analysis_frame']}\n")
    add(f"**Protocol:** {report['protocol']}\n")
    add("**Synthetic-data caveat:** results are pipeline validation; they do not "
        "represent observed real-world Indian truck telemetry.\n")

    add("## Selected models (from M1.2 validation, not test)\n")
    add("| Feature set | Selected model | Validation MAE |")
    add("| --- | --- | --- |")
    for fs, sel in report["selected_models_from_validation"].items():
        add(f"| {fs} | {sel['model']} | {sel['validation_mae']} |")
    add("")

    add("## Verification / reconciliation\n")
    v = report["verification"]
    add("- Test trucks: " + ", ".join(v["test_trucks"]) + "\n")
    add(f"- Test rows: {v['test_rows']}")
    add(f"- Test trips: {v['n_test_trips']} (expected {v['n_trips_expected']})")
    add(f"- All expected test trucks present: {v['all_expected_test_trucks_present']}")
    add(f"- Full vs telematics same rows: {v['full_and_telematics_same_rows']}")
    add(f"- Per-feature-set row counts match: {v['per_feature_set_row_counts_match']}")
    add(f"- Bootstrap resamples: {v['bootstrap_n_resamples']} (seed {v['bootstrap_seed']})")
    add("")

    # Pooled paired delta
    add("## Pooled paired comparison (full-operational vs telematics-only)\n")
    pd_ = report["pooled_paired_delta"]
    add(f"Sign convention: {pd_['sign_convention']}\n")
    add("| Metric | Full-operational | Telematics-only | Delta |")
    add("| --- | --- | --- | --- |")
    add(f"| MAE | {pd_['mae_full_operational']:.4f} | "
        f"{pd_['mae_telematics_only']:.4f} | {pd_['delta_mae']:.4f} |")
    add(f"| RMSE | {pd_['rmse_full_operational']:.4f} | "
        f"{pd_['rmse_telematics_only']:.4f} | {pd_['delta_rmse']:.4f} |")
    add("")

    # Per-truck paired delta
    add("## Per-truck paired MAE delta\n")
    add("| Truck | n | MAE full | MAE telematics | Δ MAE |")
    add("| --- | --- | --- | --- | --- |")
    for truck, d in report["per_truck_paired_delta"].items():
        add(f"| {truck} | {d['n']} | {d['mae_full_operational']:.4f} | "
            f"{d['mae_telematics_only']:.4f} | {d['delta_mae']:.4f} |")
    add("")

    # Bootstrap
    add("## Trip-clustered bootstrap (95% percentile intervals)\n")
    b = report["bootstrap"]
    add(f"Method: {b['method']}\n")
    add(f"Resampling unit: {b['resampling_unit']}; n trips = {b['n_trips']}; "
        f"resamples = {b['n_resamples']}; seed = {b['seed']}\n")
    add(f"**{b['note']}**\n")
    add("| Quantity | 95% CI |")
    add("| --- | --- |")
    for k in ("mae_full", "rmse_full", "mae_tel", "rmse_tel", "delta_mae"):
        add(f"| {k} | {_fmt_ci(b[k])} |")
    add("")

    # Feature sets: tail, variance decomposition, per-trip.
    for fs, fsr in report["feature_sets"].items():
        add(f"## Feature set: `{fs}`\n")
        pooled = fsr["pooled"]
        add(f"Pooled MAE = {pooled['mae']:.4f}; rows = {pooled['n_rows']}; "
            f"trips = {fsr['n_trips']}\n")
        add("Pooled tail error (L/h):")
        add("| Level | p90 | p95 | p99 | max |")
        add("| --- | --- | --- | --- | --- |")
        add(f"| pooled | {pooled['p90']:.3f} | {pooled['p95']:.3f} | "
            f"{pooled['p99']:.3f} | {pooled['max']:.3f} |")
        add("")
        add("Per-truck tail error (L/h):")
        add("| Truck | n | p90 | p95 | p99 | max |")
        add("| --- | --- | --- | --- | --- | --- |")
        for truck, tl in fsr["per_truck_tail"].items():
            add(f"| {truck} | {tl['n']} | {tl['p90']:.3f} | {tl['p95']:.3f} | "
                f"{tl['p99']:.3f} | {tl['max']:.3f} |")
        add("")
        vd = fsr["variance_decomposition"]
        add("Variance decomposition of squared error (descriptive):")
        add(f"- Between-truck share: {vd['share_of_squared_error']['between_truck']}")
        add(f"- Between-trip (within truck) share: "
            f"{vd['share_of_squared_error']['between_trip_within_truck']}")
        add(f"- Within-trip share: {vd['share_of_squared_error']['within_trip']}")
        add(f"- Note: {vd['note']}\n")
        add("Worst 5 trips by MAE:")
        add("| Trip | Truck | Route | Class | n | MAE | RMSE | p95 |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in fsr["worst_trips_by_mae"]:
            add(f"| {r['trip_id']} | {r['truck_id']} | {r['route_type']} | "
                f"{r['vehicle_class']} | {r['n']} | {r['mae']:.3f} | "
                f"{r['rmse']:.3f} | {r['p95']:.3f} |")
        add("")

    add("## Limitations\n")
    for lim in report["limitations"]:
        add(f"- {lim}")

    md_path = M3_REPORT_DIR / "cross_truck_generalization_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return md_path


def main() -> None:
    report = build_report()
    json_path = _write_json(report)
    md_path = _write_markdown(report)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")

    # Quick console summary of the key numbers.
    pd_ = report["pooled_paired_delta"]
    b = report["bootstrap"]
    print("\nPooled paired delta MAE (full − telematics): "
          f"{pd_['delta_mae']:.4f} "
          f"(positive => full-operational lower error)")
    print(f"Bootstrap 95% CI for paired MAE delta: {_fmt_ci(b['delta_mae'])}")
    for fs in FEATURE_SETS:
        pooled = report["feature_sets"][fs]["pooled"]
        print(f"[{fs}] pooled MAE={pooled['mae']:.4f} p95={pooled['p95']:.3f} "
              f"max={pooled['max']:.3f}")


if __name__ == "__main__":
    main()
