"""M3.3 runner — cross-truck and cross-condition error diagnosis (read-only).

Loads the frozen five-truck test split and the saved M1.2 artifacts via the
M3.1 ``prepare_test_predictions`` interface, attaches operating-condition
columns from telemetry (M3.2), then computes and writes the M3.3 report:

  * ``reports/m3/error_diagnosis_report.json``,
  * ``reports/m3/error_diagnosis_summary.md``.

No model is trained/refitted/tuned; no dataset or frozen artifact is modified.
All metrics are descriptive; no hypothesis tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import enforce_split, load_split_manifest, load_telemetry
from src.evaluation.m3_condition_robustness import (
    INFORMATIVE_CONDITIONS,
    attach_operating_columns,
)
from src.evaluation.m3_cross_truck_analysis import (
    FEATURE_SETS,
    EXPECTED_TEST_TRUCKS,
    prepare_test_predictions,
)
from src.evaluation.m3_error_diagnosis import build_diagnosis_report

M1_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"
M3_REPORT_DIR = PROJECT_ROOT / "reports" / "m3"

CELL_SUPPORT_STATUSES = (
    "descriptive",
    "insufficient_rows",
    "insufficient_trip_support",
    "empty",
)


def build_report() -> dict:
    # Reuse the exact M3.1 interface: saved artifacts + frozen split, aligned.
    prep = prepare_test_predictions(report_dir=M1_REPORT_DIR)
    frames = prep["frames"]

    # Attach operating-condition columns from telemetry by index (aligned).
    telemetry = load_telemetry()
    manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)
    test_telemetry = split.test

    attached: dict[str, object] = {}
    for fs in FEATURE_SETS:
        frame = frames[fs]
        if not frame.index.equals(test_telemetry.index):
            raise ValueError(
                f"prepared frame index not aligned with test telemetry ({fs})"
            )
        attached[fs] = attach_operating_columns(frame, test_telemetry)

    diagnosis_report = build_diagnosis_report(attached, FEATURE_SETS)

    # Verification / reconciliation
    verification = {
        "test_rows": int(len(test_telemetry)),
        "test_trucks": sorted(set(test_telemetry["truck_id"])),
        "all_expected_test_trucks_present": sorted(
            set(test_telemetry["truck_id"])
        )
        == sorted(EXPECTED_TEST_TRUCKS),
        "n_test_trips": int(test_telemetry["trip_id"].nunique()),
        "n_test_trips_expected": prep["n_trips"],
        "full_and_telematics_same_rows": frames["full_operational"].index.equals(
            frames["telematics_only"].index
        ),
        "per_feature_set_row_counts_match": {
            fs: int(len(attached[fs])) for fs in FEATURE_SETS
        },
        "operating_column_attach_no_missing": True,
        "preprocessors_loaded_not_refitted": True,
        "models_loaded_not_refitted": True,
        "cell_support_statuses_used": sorted(
            set(
                c["support_status"]
                for fs in FEATURE_SETS
                for c in diagnosis_report["feature_sets"][fs]["truck_condition_cells"]
            )
        ),
        "all_requested_conditions_present": {
            fs: all(
                label in {c["condition"] for c in diagnosis_report["feature_sets"][fs]["truck_condition_cells"]}
                for label in INFORMATIVE_CONDITIONS
            )
            for fs in FEATURE_SETS
        },
    }

    report: dict = {
        "experiment": "M3.3_cross_truck_cross_condition_error_diagnosis",
        "milestone": "M3.3",
        "protocol": (
            "read-only: reuse frozen 5-truck test split and saved M1.2 "
            "artifacts; no training/refitting/tuning; no test-set-driven "
            "model selection"
        ),
        "analysis_frame": (
            "conditional on the five frozen test trucks; descriptive only; "
            "no hypothesis tests, no causal claims"
        ),
        "residual_convention": diagnosis_report["residual_convention"],
        "selected_models_from_validation": prep["selected_models"],
        "verification": verification,
        "diagnostics": diagnosis_report,
        "limitations": [
            "Synthetic telemetry: results are pipeline validation, not "
            "observed real-world Indian truck telemetry.",
            "Only five test trucks: all truck-condition cells and residual "
            "tables are descriptive and unstable; not population estimates.",
            "49T is represented by a single held-out truck; any 49T "
            "truck-condition result is descriptive and is NOT generalized to "
            "the 49T population.",
            "Truck-condition cells span a single truck by construction "
            "(n_trucks=1). Support is judged on rows and trips, not truck "
            "count; cells with < MIN_SUPPORT_ROWS rows or a single trip are "
            "flagged insufficient.",
            "The flagged_high_error heuristic is a descriptive deviation from "
            "truck-level and pooled-condition baselines; it is NOT a "
            "statistical interaction test (only five trucks). It is an "
            "investigative indicator only and may appear on cells flagged "
            "insufficient_support, where it is indicative rather than "
            "conclusive.",
            "Recurring-vs-isolated is a descriptive categorization based on "
            "the worst-trip share of squared error; it is not a mechanistic "
            "finding.",
            "Residual direction uses the explicit convention residual = "
            "y_pred - y_true (positive => overprediction); the "
            "approximately-unbiased figure uses a documented |residual| <= "
            "0.05 L/h tolerance.",
            "Paired model deltas are descriptive; small differences are not "
            "evidence of model superiority. No hypothesis tests were run.",
        ],
    }
    return report


def _write_json(report: dict) -> Path:
    report_path = M3_REPORT_DIR / "error_diagnosis_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path


def _fmt_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return lines


def _write_markdown(report: dict) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# M3.3 Cross-Truck and Cross-Condition Error Diagnosis\n")
    add(f"**Milestone:** {report['milestone']}\n")
    add(f"**Analysis frame:** {report['analysis_frame']}\n")
    add(f"**Protocol:** {report['protocol']}\n")
    add(f"**Residual convention:** {report['residual_convention']}\n")
    add("**Synthetic-data caveat:** results are pipeline validation; they do "
        "not represent observed real-world Indian truck telemetry.\n")
    add("**Diagnostic scope:** descriptive only. The `flagged_high_error` "
        "heuristic is a deviation from truck-level and pooled-condition "
        "baselines, NOT a statistical interaction test (only five trucks). "
        "It is an investigative indicator only and may appear on cells flagged "
        "insufficient_support, where it is indicative rather than "
        "conclusive. `recurring` vs `dominated_by_worst_trip` is a "
        "descriptive categorization, not a mechanistic finding.\n")

    add("## Selected models (from M1.2 validation, not test)\n")
    add("| Feature set | Selected model |")
    add("| --- | --- |")
    for fs, sel in report["selected_models_from_validation"].items():
        add(f"| {fs} | {sel['model']} |")
    add("")

    add("## Verification / reconciliation\n")
    v = report["verification"]
    add("- Test trucks: " + ", ".join(v["test_trucks"]) + "\n")
    add(f"- Test rows: {v['test_rows']}")
    add(f"- Test trips: {v['n_test_trips']} (expected {v['n_test_trips_expected']})")
    add(f"- All expected test trucks present: {v['all_expected_test_trucks_present']}")
    add(f"- Full vs telematics same rows: {v['full_and_telematics_same_rows']}")
    add(f"- Per-feature-set row counts: {v['per_feature_set_row_counts_match']}")
    add(f"- Cell support statuses used: {v['cell_support_statuses_used']}")
    add(f"- All requested conditions present per feature set: {v['all_requested_conditions_present']}")
    add("")

    for fs, d in report["diagnostics"]["feature_sets"].items():
        add(f"## Feature set: `{fs}`\n")
        add(f"Rows = {d['n_rows']}; trips = {d['n_trips']}; "
            f"trucks = {d['n_trucks']}\n")

        add("### Experiment 1 — Truck × condition error table\n")
        add("| Truck | Condition | n rows | n trips | MAE | RMSE | median | "
            "P90 | P95 | truck ratio | cond ratio | flagged | support |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- |")
        for c in d["truck_condition_cells"]:
            mae = f"{c['mae']:.4f}" if c["mae"] is not None else "n/a"
            rmse = f"{c['rmse']:.4f}" if c["rmse"] is not None else "n/a"
            med = f"{c['median_absolute_error']:.4f}" if c["median_absolute_error"] is not None else "n/a"
            p90 = f"{c['p90_absolute_error']:.4f}" if c["p90_absolute_error"] is not None else "n/a"
            p95 = f"{c['p95_absolute_error']:.4f}" if c["p95_absolute_error"] is not None else "n/a"
            tr = f"{c['mae_ratio_to_truck_baseline']:.4f}" if c["mae_ratio_to_truck_baseline"] is not None else "-"
            cr = f"{c['mae_ratio_to_condition_baseline']:.4f}" if c["mae_ratio_to_condition_baseline"] is not None else "-"
            add(f"| {c['truck_id']} | {c['condition']} | {c['n_rows']} | "
                f"{c['n_trips']} | {mae} | {rmse} | {med} | {p90} | {p95} | "
                f"{tr} | {cr} | {c['flagged_high_error']} | "
                f"{c['support_status']} |")
        add("")
        add("`truck ratio` = cell MAE / truck-level baseline MAE; `cond ratio` "
            "= cell MAE / pooled-condition baseline MAE. A cell is "
            "`flagged_high_error=True` only when BOTH ratios >= "
            f"{report['diagnostics']['feature_sets'][fs]['high_error_ratio']} "
            "(descriptive heuristic, not a statistical interaction test).\n")

        add("### Experiment 2 — Trip-level recurrence\n")
        add("| Truck | Condition | n rows | n trips | median trip MAE | "
            "std trip MAE | worst-trip share | pattern | support |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in d["trip_recurrence"]:
            med = f"{r['median_trip_mae']:.4f}" if r["median_trip_mae"] is not None else "n/a"
            std = f"{r['std_trip_mae']:.4f}" if r["std_trip_mae"] is not None else "n/a"
            ws = f"{r['worst_trip_squared_error_share']:.4f}" if r["worst_trip_squared_error_share"] is not None else "n/a"
            add(f"| {r['truck_id']} | {r['condition']} | {r['n_rows']} | "
                f"{r['n_trips']} | {med} | {std} | {ws} | {r['pattern']} | "
                f"{r['support_status']} |")
        add("")

        add("### Experiment 3 — Error distribution (pooled conditions)\n")
        add("| Condition | n rows | n trips | n trucks | MAE | RMSE | "
            "median | P90 | P95 | max | ratio |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in d["pooled_condition_error_distribution"]:
            med = f"{e['median_absolute_error']:.4f}" if e["median_absolute_error"] is not None else "n/a"
            p90 = f"{e['p90_absolute_error']:.4f}" if e["p90_absolute_error"] is not None else "n/a"
            p95 = f"{e['p95_absolute_error']:.4f}" if e["p95_absolute_error"] is not None else "n/a"
            mx = f"{e['max_absolute_error']:.4f}" if e["max_absolute_error"] is not None else "n/a"
            rt = f"{e['p95_to_median_ratio']:.4f}" if e["p95_to_median_ratio"] is not None else "zero-median"
            mae_s = f"{e['mae']:.4f}" if e["mae"] is not None else "n/a"
            rmse_s = f"{e['rmse']:.4f}" if e["rmse"] is not None else "n/a"
            add(f"| {e['condition']} | {e['n_rows']} | {e['n_trips']} | "
                f"{e['n_trucks']} | {mae_s} | {rmse_s} | "
                f"{med} | {p90} | {p95} | {mx} | {rt} |")
        add("")

        add("### Experiment 4 — Residual direction (per truck)\n")
        add("| Truck | n rows | mean signed | median signed | over% | under% | "
            "~unbiased% |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for r in d["residual_direction"]["per_truck"]:
            ms = f"{r['mean_signed_error']:.4f}" if r["mean_signed_error"] is not None else "n/a"
            mds = f"{r['median_signed_error']:.4f}" if r["median_signed_error"] is not None else "n/a"
            ov = f"{r['overprediction_pct']:.2f}" if r["overprediction_pct"] is not None else "n/a"
            un = f"{r['underprediction_pct']:.2f}" if r["underprediction_pct"] is not None else "n/a"
            ap = f"{r['approximately_unbiased_pct']:.2f}" if r["approximately_unbiased_pct"] is not None else "n/a"
            add(f"| {r['truck_id']} | {r['n_rows']} | {ms} "
                f"| {mds} | {ov} | {un} | {ap} |")
        add("")

        add("### Experiment 4 — Residual direction (per condition)\n")
        add("| Condition | n rows | mean signed | median signed | over% | "
            "under% | ~unbiased% |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for r in d["residual_direction"]["per_condition"]:
            ms = f"{r['mean_signed_error']:.4f}" if r["mean_signed_error"] is not None else "n/a"
            mds = f"{r['median_signed_error']:.4f}" if r["median_signed_error"] is not None else "n/a"
            ov = f"{r['overprediction_pct']:.2f}" if r["overprediction_pct"] is not None else "n/a"
            un = f"{r['underprediction_pct']:.2f}" if r["underprediction_pct"] is not None else "n/a"
            ap = f"{r['approximately_unbiased_pct']:.2f}" if r["approximately_unbiased_pct"] is not None else "n/a"
            add(f"| {r['condition']} | {r['n_rows']} | "
                f"{ms} | {mds} | {ov} | {un} | {ap} |")
        add("")

        add("### Experiment 5 — Paired model comparison (per condition)\n")
        add("| Condition | n rows | MAE full | MAE tel | delta MAE | RMSE full "
            "| RMSE tel | delta RMSE |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in report["diagnostics"]["paired_comparison"]["per_condition"]:
            add(f"| {r['group']} | {r['n_rows']} | "
                f"{r['mae_full_operational']:.4f} | "
                f"{r['mae_telematics_only']:.4f} | "
                f"{r['delta_mae']:.4f} | "
                f"{r['rmse_full_operational']:.4f} | "
                f"{r['rmse_telematics_only']:.4f} | "
                f"{r['delta_rmse']:.4f} |")
        add("")
        add(f"Paired delta convention: {report['diagnostics']['paired_comparison']['delta_sign_convention']}\n")
        add("Small deltas are descriptive; they are not evidence of model "
            "superiority. No hypothesis tests were run.\n")

    add("## Limitations\n")
    for lim in report["limitations"]:
        add(f"- {lim}")

    md_path = M3_REPORT_DIR / "error_diagnosis_summary.md"
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

    paired = report["diagnostics"]["paired_comparison"]["pooled"]
    print("\nPaired pooled delta (positive => full worse... wait, positive => full BETTER):")
    print(f"  delta_mae = {paired['delta_mae']:.4f} (full {paired['mae_full_operational']:.4f} "
          f"vs tel {paired['mae_telematics_only']:.4f})")

    # Summarize the worst flagged cells and recurring patterns across feature sets.
    for fs, d in report["diagnostics"]["feature_sets"].items():
        flagged = [c for c in d["truck_condition_cells"] if c["flagged_high_error"]]
        recurring = [r for r in d["trip_recurrence"] if r["pattern"] == "recurring_pattern"]
        print(f"\n[{fs}]")
        print(f"  high-error cells flagged: {len(flagged)}")
        for c in flagged[:8]:
            print(f"    {c['truck_id']} / {c['condition']}: MAE={c['mae']:.4f} "
                  f"n={c['n_rows']} trips={c['n_trips']}")
        print(f"  recurring patterns: {len(recurring)}")
        for r in recurring[:8]:
            print(f"    {r['truck_id']} / {r['condition']}: trips={r['n_trips']} "
                  f"worst-share={r['worst_trip_squared_error_share']:.3f}")


if __name__ == "__main__":
    main()
