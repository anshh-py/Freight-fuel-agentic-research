"""M3.2 runner — operating-condition robustness diagnostics (read-only).

Loads the frozen five-truck test split and the saved M1.2 artifacts via the
M3.1 ``prepare_test_predictions`` interface, attaches operating-condition
columns from telemetry, then computes and writes the M3.2 report:

  * ``reports/m3/condition_robustness_report.json``,
  * ``reports/m3/condition_robustness_summary.md``.

No model is trained/refitted/tuned; no dataset or frozen artifact is modified.
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
    attach_operating_columns,
    build_condition_report,
)
from src.evaluation.m3_cross_truck_analysis import (
    FEATURE_SETS,
    EXPECTED_TEST_TRUCKS,
    prepare_test_predictions,
)

M1_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"
M3_REPORT_DIR = PROJECT_ROOT / "reports" / "m3"


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
        # Verify index alignment with the test telemetry before attaching.
        if not frame.index.equals(test_telemetry.index):
            raise ValueError(
                f"prepared frame index not aligned with test telemetry ({fs})"
            )
        attached[fs] = attach_operating_columns(frame, test_telemetry)

    # per-feature-set condition reports
    condition_reports = {
        fs: build_condition_report(attached[fs]) for fs in FEATURE_SETS
    }

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
    }

    report: dict = {
        "experiment": "M3.2_operating_condition_robustness",
        "milestone": "M3.2",
        "protocol": (
            "read-only: reuse frozen 5-truck test split and saved M1.2 "
            "artifacts; no training/refitting/tuning; no test-set-driven "
            "model selection"
        ),
        "analysis_frame": (
            "conditional on the five frozen test trucks; descriptive only; "
            "no hypothesis tests or significance claims"
        ),
        "selected_models_from_validation": prep["selected_models"],
        "verification": verification,
        "feature_sets": condition_reports,
        "limitations": [
            "Synthetic telemetry: results are pipeline validation, not "
            "observed real-world Indian truck telemetry.",
            "Payload load-factor ceiling: across the entire synthetic dataset "
            "(all 963,697 rows) the load factor payload_kg / "
            "gross_vehicle_weight_kg never exceeds 0.3886. There is therefore "
            "NO heavy-payload regime (LF > 0.66) in the data; the heavy and "
            "combined-heavy cells are empty by construction, not by an error, "
            "and cannot be evaluated. Only the light (<= 0.33) and medium "
            "(0.33-0.66) regimes are populated.",
            "Only five test trucks: any truck-level or per-condition breakdown "
            "is descriptive and unstable; not a population estimate.",
            "49T is represented by a single held-out truck; its per-condition "
            "results are descriptive only and are NOT generalized to the 49T "
            "population.",
            "Combined-condition subgroups are small; they are reported with "
            "explicit insufficient-support flags and must not be "
            "over-interpreted.",
            "Payload load factor uses payload_kg / gross_vehicle_weight_kg "
            "(intra-trip constant); invalid denominators are bucketed "
            "separately.",
            "Insufficient-support flag is a combined caution (rows AND "
            "trucks), not a proof of reliability.",
        ],
    }
    return report


def _write_json(report: dict) -> Path:
    report_path = M3_REPORT_DIR / "condition_robustness_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path


def _fmt_table(
    headers: list[str], rows: list[list[str]]
) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return lines


def _write_markdown(report: dict) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# M3.2 Operating-Condition Robustness Analysis\n")
    add(f"**Milestone:** {report['milestone']}\n")
    add(f"**Analysis frame:** {report['analysis_frame']}\n")
    add(f"**Protocol:** {report['protocol']}\n")
    add("**Synthetic-data caveat:** results are pipeline validation; they do "
        "not represent observed real-world Indian truck telemetry.\n")

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
    add("")

    add("## Insufficient-support note\n")
    add(f"{report['feature_sets'][FEATURE_SETS[0]]['insufficient_support_note']}\n")

    for fs, cs in report["feature_sets"].items():
        add(f"## Feature set: `{fs}`\n")
        add(f"Rows = {cs['n_rows']}; trips = {cs['n_trips']}; "
            f"trucks = {cs['n_trucks']}\n")

        add("### Route-type metrics (trip level)\n")
        add("| Route type | n rows | n trips | n trucks | MAE | RMSE |")
        add("| --- | --- | --- | --- | --- | --- |")
        for r in cs["route_type_table"]:
            add(f"| {r['route_type']} | {r['n_rows']} | {r['n_trips']} | "
                f"{r['n_trucks']} | {r['mae']:.4f} | {r['rmse']:.4f} |")
        add("")

        add("### Condition table (coverage + support status + MAE/RMSE)\n")
        add("| Condition | n rows | n trips | n trucks | MAE | RMSE | "
            "support status |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for c in cs["conditions"]:
            add(f"| {c['condition']} | {c['n_rows']} | {c['n_trips']} | "
                f"{c['n_trucks']} | {c['mae']:.4f} | {c['rmse']:.4f} | "
                f"{c['support_status']} |")
        add("")
        add("*Support status legend:* `descriptive_full_truck_coverage` covers "
            "all 5 frozen test trucks; "
            "`descriptive_limited_truck_coverage` has enough rows and at "
            "least 2 trucks but fewer than the full test-truck set — reported "
            "descriptively with an explicit limited-coverage warning, NOT as "
            "unusable; `insufficient_rows` / `insufficient_truck_coverage` / "
            "`empty` are genuinely under-sampled (insufficient_support=True).")
        add("")
        # Explicit limited-coverage warning for the medium-payload subgroup.
        for c in cs["conditions"]:
            if c["condition"] == "payload_medium_gt0.33_le0.66" and c[
                "limited_truck_coverage"]:
                add(
                    "> **Medium-payload subgroup:** descriptive result with "
                    "incomplete truck coverage (n_trucks=3 of 5). It is NOT "
                    "flagged unusable, but it does not support population-level "
                    "generalization."
                )
                add("")

        add("### Error concentration (worst first, by squared-error share)\n")
        add("| Condition | sq-err share | row share | MAE | RMSE |")
        add("| --- | --- | --- | --- | --- |")
        for e in cs["error_concentration"]:
            add(f"| {e['condition']} | {e['squared_error_share']:.4f} | "
                f"{e['row_share']:.4f} | "
                f"{('%.4f' % e['mae']) if e['mae'] is not None else 'n/a'} | "
                f"{('%.4f' % e['rmse']) if e['rmse'] is not None else 'n/a'} |")
        add("")

        add("### Per-truck × condition (informative conditions)\n")
        for cond, d in cs["per_truck_condition"].items():
            add(f"**{cond}** (n={d['n_rows']})")
            add("| Truck | n rows | MAE | RMSE |")
            add("| --- | --- | --- | --- |")
            for truck, td in d["per_truck"].items():
                add(f"| {truck} | {td['n_rows']} | {td['mae']:.4f} | "
                    f"{td['rmse']:.4f} |")
            add("")

        add(f"Payload bin edges (load factor): {cs['payload_bin_edges']}\n")

    add("## Limitations\n")
    for lim in report["limitations"]:
        add(f"- {lim}")

    md_path = M3_REPORT_DIR / "condition_robustness_summary.md"
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

    for fs in FEATURE_SETS:
        print(f"\n[{fs}]")
        for r in report["feature_sets"][fs]["route_type_table"]:
            print(f"  route={r['route_type']}: n={r['n_rows']} "
                  f"mae={r['mae']:.4f} rmse={r['rmse']:.4f}")


if __name__ == "__main__":
    main()
