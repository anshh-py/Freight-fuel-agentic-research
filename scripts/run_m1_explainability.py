"""M1.4 — explainability and feature analysis for the validation-selected
XGBoost models.

Reproducible entry point:

    python scripts/run_m1_explainability.py                 # default
    python scripts/run_m1_explainability.py --report-dir D

Reads (never modifies):
  * telemetry + truck-wise split manifest (source of truth for truck split),
  * the M1.2 artifacts (xgboost model + preprocessor) per feature set.

Writes (all new files, under ``<report-dir>``):
  * ``explainability_report.json``       machine-readable explainability results
  * ``explainability_summary.md``        human-readable summary
  * ``plots/explainability/*.png``       SHAP importance/summary/dependence/
                                         per-truck/waterfall plots

Guarantees: SHAP explanations use a deterministic sample of the five held-out
test trucks (never train/validation rows); preprocessors are loaded from disk
and never refitted; models are loaded, never retrained. The results explain
models trained on synthetic data and are not real-world causal claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.explainability import (  # noqa: E402
    DEFAULT_REPORT_DIR,
    DEFAULT_SAMPLE_PER_TRUCK,
    RANDOM_STATE,
    run_explainability,
)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def render_summary(report: dict) -> str:
    lines: list[str] = [
        "# M1.4 Explainability & Feature Analysis",
        "",
        f"Protocol: {report['protocol']}",
        "",
        "**Scope:** validation-selected XGBoost models, both feature sets, "
        "explained on a deterministic sample of the five held-out test trucks.",
        "",
        f"## Sampling ({report['sampling']['method']})",
        "",
        f"- Sample size per truck: {report['sampling']['sample_per_truck']}",
        f"- Seed: {report['sampling']['seed']}",
        f"- All five test trucks represented: "
        f"{report['sampling']['all_five_test_trucks_represented']}",
        "",
        "## Verification",
        "",
        f"- All five test trucks represented: "
        f"{report['verification']['all_five_test_trucks_represented']}",
        f"- No train/validation truck in sample: "
        f"{report['verification']['no_train_or_val_truck_in_sample']}",
        f"- Preprocessors loaded, never refitted: "
        f"{report['verification']['preprocessors_loaded_never_refitted']}",
        f"- Models loaded, not retrained: "
        f"{report['verification']['models_loaded_not_retrained']}",
        f"- Source of truth for truck split: "
        f"{report['verification']['source_of_truth_for_truck_split']}",
        "",
        "## Important caveat",
        "",
        "These explanations describe models trained on the **synthetic** "
        "telemetry suite. They are model-behavior explanations, not causal "
        "claims and not evidence of real-world fuel-consumption drivers.",
        "",
    ]

    for fs, fsr in report["feature_sets"].items():
        lines.append(f"## Feature set: `{fs}`")
        lines.append("")
        lines.append(
            f"- Sample rows: {fsr['sample']['n_rows_used']} "
            f"(per truck: {fsr['sample']['per_truck_used']})"
        )
        recon = fsr["reconstruction_validation"]
        lines.append(
            f"- SHAP reconstruction: max|dev|={recon['max_abs_deviation']:.2e}, "
            f"fraction within tol={recon['fraction_within_tolerance']:.4f}, "
            f"pass={recon['pass']}"
        )
        lines.append(
            f"- Model artifact: {fsr['model_artifact_size_bytes']} bytes; "
            f"preprocessor: {fsr['preprocessor_artifact_size_bytes']} bytes"
        )
        lines.append("")
        lines.append("### Global feature importance (mean |SHAP|)")
        lines.append("")
        lines.append("| # | Feature | Mean |SHAP| (L/h) |")
        lines.append("| --- | --- | --- |")
        for i, row in enumerate(fsr["global_importance_mean_abs_shap"]):
            lines.append(
                f"| {i+1} | {row['readable_label']} | {row['mean_abs_shap']:.4f} |"
            )
        lines.append("")
        lines.append("### Per-truck mean |SHAP| (top 5 per truck)")
        lines.append("")
        trucks = sorted(fsr["per_truck_mean_abs_shap"])
        lines.append("| Truck | Top features (mean |SHAP|) |")
        lines.append("| --- | --- |")
        for t in trucks:
            top = fsr["per_truck_mean_abs_shap"][t][:5]
            txt = "; ".join(
                f"{r['column']}={r['mean_abs_shap']:.3f}" for r in top
            )
            lines.append(f"| {t} | {txt} |")
        lines.append("")
        lines.append("### Individual explanation records (deterministic rules)")
        lines.append("")
        lines.append("| Truck | Rule | Actual L/h | Pred L/h | Residual |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rec in fsr["individual_explanations"]:
            lines.append(
                f"| {rec['truck_id']} | {rec['rule']} | "
                f"{rec['actual_fuel_rate']:.3f} | "
                f"{rec['predicted_fuel_rate']:.3f} | "
                f"{rec['residual']:+.3f} |"
            )
        lines.append("")
        lines.append("### Plots")
        lines.append("")
        for f in fsr["plots"]["files"]:
            lines.append(f"- plots/explainability/{f}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Synthetic data**: SHAP explains models trained on the formula-"
        "generated suite; do not interpret as real-world fuel-consumption "
        "drivers."
    )
    lines.append(
        "- **Per-truck importance** is over a small deterministic sample per "
        "truck and reflects that truck's own operating profile, not a "
        "population estimate."
    )
    lines.append(
        "- SHAP is an additive model-explanations tool; feature attributions "
        "are not causal effects."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M1.4 explainability")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--feature-sets", default=None,
                        help="comma-separated sets")
    parser.add_argument("--sample-per-truck", type=int,
                        default=DEFAULT_SAMPLE_PER_TRUCK)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args(argv)

    report_dir = args.report_dir.expanduser().resolve()
    feature_sets = (
        [f.strip() for f in args.feature_sets.split(",") if f.strip()]
        if args.feature_sets
        else None
    )

    t_start = time.perf_counter()
    report = run_explainability(
        feature_sets=feature_sets,
        sample_per_truck=args.sample_per_truck,
        seed=args.seed,
        report_dir=report_dir,
    )
    report["total_clock_seconds"] = round(time.perf_counter() - t_start, 3)

    _write_json(report_dir / "explainability_report.json", report)
    (report_dir / "explainability_summary.md").write_text(
        render_summary(report), encoding="utf-8"
    )

    print(f"M1.4 complete. Wrote to: {report_dir}")
    print(f"Total clock: {report['total_clock_seconds']}s")
    for fs, fsr in report["feature_sets"].items():
        print(f"  {fs}: sample={fsr['sample']['n_rows_used']} rows, "
              f"reconstruction pass={fsr['reconstruction_validation']['pass']}, "
              f"plots={len(fsr['plots']['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
