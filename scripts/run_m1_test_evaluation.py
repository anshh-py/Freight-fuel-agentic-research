"""M1.3 — unseen-truck final evaluation of M1.2 baseline models.

Reproducible entry point:

    python scripts/run_m1_test_evaluation.py                 # default
    python scripts/run_m1_test_evaluation.py --report-dir D

Reads (never modifies):
  * telemetry + truck-wise split manifest (source of truth for truck split),
  * the M1.2 artifacts under ``<report-dir>/artifacts/<feature_set>/``.

Writes (all new files, under ``<report-dir>``):
  * ``test_metrics_report.json``          machine-readable test metrics
  * ``test_evaluation_summary.md``        human-readable summary
  * ``plots/*.png``                       residual diagnostic plots

Protocol guarantees: test data is evaluated once here for final evaluation;
it is never used for fitting, tuning, feature selection or model selection
(selection comes from the M1.2 validation report).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.test_evaluation import evaluate_test_set  # noqa: E402

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        if np.isnan(x):
            return "n/a"
    except TypeError:
        pass
    return f"{x:.4f}"


def _render_model_summary(entry: dict) -> list[str]:
    lines: list[str] = []
    p = entry["pooled"]
    lines.append(
        f"| pooled | {_fmt(p['mae'])} | {_fmt(p['rmse'])} | {_fmt(p['r2'])} "
        f"| {_fmt(p['mape'])} | {p['count']} |"
    )
    macro = entry["macro_average_across_test_trucks"]
    lines.append(
        f"| macro-avg | {_fmt(macro['mae'])} | {_fmt(macro['rmse'])} "
        f"| {_fmt(macro['r2'])} | {_fmt(macro['mape'])} | — |"
    )
    # per-truck
    rows = []
    for truck, m in entry["per_truck_metrics"].items():
        rows.append(
            f"| {truck} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} "
            f"| {_fmt(m['r2'])} | {_fmt(m['mape'])} | {m['count']} |"
        )
    lines.append("| **Per-truck** | | | | | |")
    lines.append("| truck | MAE | RMSE | R² | MAPE (>1) | rows |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    lines.extend(rows)
    # per-class
    lines.append("")
    lines.append("| **Per-class** | | | | | |")
    lines.append("| class | MAE | RMSE | R² | MAPE (>1) | rows |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cls, m in entry["per_class_metrics"].items():
        lines.append(
            f"| {cls} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} "
            f"| {_fmt(m['r2'])} | {_fmt(m['mape'])} | {m['count']} |"
        )
    # residual bias
    r = entry["residual_stats"]
    lines.append("")
    lines.append(
        f"| **Residual bias** | mean={r['mean_residual']:.3f}, "
        f"median={r['median_residual']:.3f}, std={r['std_residual']:.3f}, "
        f"underpredicted={r['fraction_underpredicted']*100:.1f}%, "
        f"overpredicted={r['fraction_overpredicted']*100:.1f}% |"
    )
    return lines


def render_summary(report: dict) -> str:
    lines: list[str] = [
        "# M1.3 Unseen-Truck Final Evaluation",
        "",
        f"Protocol: {report['protocol']}",
        "",
        f"**Selected models (from M1.2 validation, not test):**",
        "",
        "| Feature set | Selected model | Validation MAE |",
        "| --- | --- | --- |",
    ]
    for fs, sel in report["selected_models_from_validation"].items():
        lines.append(
            f"| `{fs}` | `{sel['model']}` | {_fmt(sel['validation_mae'])} |"
        )
    lines.append("")
    lines.append(
        "Model selection used **validation** evidence only; the test set was "
        "not used for tuning, feature choice, or retrospective selection."
    )
    lines.append("")

    ver = report["verification"]
    lines.append("## Verification")
    lines.append("")
    lines.append("- " + ", ".join(ver["test_trucks_evaluated"]))
    lines.append(
        f"- All expected test trucks present: {ver['all_expected_test_trucks_present']}"
    )
    lines.append(
        f"- Train/validation truck in test evaluation: "
        f"{ver['train_or_val_truck_in_test_evaluation']}"
    )
    lines.append(f"- Test rows: {ver['test_rows']}")
    lines.append(
        f"- Preprocessors loaded from saved M1.2 artifacts, never refitted: "
        f"{ver['preprocessors_loaded_from_disk_not_refitted']}"
    )
    lines.append("")

    for fs, fsr in report["feature_sets"].items():
        lines.append(f"## Feature set: `{fs}`")
        lines.append("")
        lines.append(f"- Test rows: {fsr['test_n_rows']}, features: {fsr['n_features']}")
        lines.append("- Plots: " + ", ".join(f"plots/{f}" for f in fsr["plots"].get("files", [])))
        lines.append("")
        for name, entry in fsr["models"].items():
            sel = report["selected_models_from_validation"][fs]["model"]
            marker = " ✅ selected" if name == sel else ""
            lines.append(f"### `{name}`{marker}")
            lines.append("")
            if "note" in entry:
                lines.append("> " + entry["note"])
                lines.append("")
            lines.append("| metric | MAE | RMSE | R² | MAPE (>1) | rows |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            lines.extend(_render_model_summary(entry))
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Synthetic data**: metrics reflect the formula-generated suite; "
        "they do **not** establish real-world predictive performance."
    )
    lines.append(
        "- **Macro-average** is a descriptive average over the five test "
        "trucks only (n=5); it is not a statistically robust population estimate."
    )
    lines.append(
        "- **Small class samples**: per-class test estimates come from 2×32T, "
        "2×40T and 1×49T trucks and should be read accordingly."
    )
    lines.append(
        "- **Rows within a truck/trip are autocorrelated**; per-truck metrics "
        "are the more meaningful unit than row-level aggregates."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M1.3 test evaluation")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--models", default=None, help="comma-separated models")
    parser.add_argument("--feature-sets", default=None, help="comma-separated sets")
    args = parser.parse_args(argv)

    report_dir = args.report_dir.expanduser().resolve()

    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    feature_sets = (
        [f.strip() for f in args.feature_sets.split(",") if f.strip()]
        if args.feature_sets
        else None
    )

    t_start = time.perf_counter()
    report = evaluate_test_set(
        models=models,
        feature_sets=feature_sets,
        report_dir=report_dir,
    )
    report["total_clock_seconds"] = round(time.perf_counter() - t_start, 3)

    _write_json(report_dir / "test_metrics_report.json", report)
    (report_dir / "test_evaluation_summary.md").write_text(
        render_summary(report), encoding="utf-8"
    )

    print(f"M1.3 complete. Wrote to: {report_dir}")
    print(f"Total clock: {report['total_clock_seconds']}s")
    print(f"Test trucks evaluated: {report['verification']['test_trucks_evaluated']}")
    for fs, sel in report["selected_models_from_validation"].items():
        print(f"  selected ({fs}): {sel['model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
