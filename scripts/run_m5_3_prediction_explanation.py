"""M5.3 runner — prediction & explanation reliability evaluation (real tools).

Loads the real saved M1.2 XGBoost models + preprocessors through the actual
``FuelRatePredictionTool`` and ``PredictionExplanationTool``, runs the
deterministic reliability / consistency / validation / explanation /
segment checks, and writes:

  * ``reports/m5/m5_3/evaluation.json``
  * ``reports/m5/m5_3/summary.md``
  * ``reports/m5/m5_3/plots/tool_agreement_by_segment.png``

No models retrained, no preprocessors refitted, no frozen M1/M2/M3/M4 file
modified. Deterministic (fixed seed); no LLM/API/DB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import enforce_split, load_split_manifest, load_telemetry
from src.evaluation.m5_3_prediction_explanation import (
    FEATURE_SETS,
    INVALID_CASES,
    canonical_inputs,
    check_attribution_direction,
    project_row_to_inputs,

    check_explanation_reconstruction,
    check_explanation_safety,
    check_finite_valid,
    check_invalid_input_rejection,
    classify_segment,
    prediction_agreement_with_direct_model,
    prediction_explanation_consistency,
    repeated_prediction_consistency,
    sample_test_inputs,
)
from src.tools.fuel_rate_prediction import DEFAULT_MODEL_DIR, FuelRatePredictionTool
from src.tools.prediction_explanation import PredictionExplanationTool

M5_3_REPORT_DIR = PROJECT_ROOT / "reports" / "m5" / "m5_3"
PLOTS_DIR = M5_3_REPORT_DIR / "plots"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _build_tool_evaluation(feature_set: str, inputs: dict) -> dict:
    """Run the full prediction + explanation check battery for one input set."""
    pred_tool = FuelRatePredictionTool(model_dir=DEFAULT_MODEL_DIR)
    expl_tool = PredictionExplanationTool(model_dir=DEFAULT_MODEL_DIR)

    # Prediction reliability.
    consistency = repeated_prediction_consistency(pred_tool, inputs, feature_set)
    pred_res = pred_tool.predict(inputs, feature_set=feature_set)
    finite = check_finite_valid(pred_res)
    agreement = prediction_agreement_with_direct_model(pred_tool, inputs, feature_set)
    rejection = check_invalid_input_rejection(pred_tool, feature_set, inputs)

    # Explanation reliability.
    expl_res = expl_tool.explain(inputs, feature_set=feature_set)
    recon = check_explanation_reconstruction(expl_res)
    direction = check_attribution_direction(expl_res)
    safety = check_explanation_safety(expl_res)
    consistency_expl = prediction_explanation_consistency(pred_res, expl_res)

    return {
        "feature_set": feature_set,
        "segment": classify_segment(inputs),
        "prediction": {
            "repeated_consistency": consistency,
            "finite_valid": finite,
            "tool_vs_direct_model_agreement": agreement,
            "invalid_input_rejection": rejection,
        },
        "explanation": {
            "reconstruction": recon,
            "attribution_direction": direction,
            "safety": safety,
            "prediction_consistency": consistency_expl,
        },
    }


def _run_full_evaluation() -> dict:
    results: dict[str, dict] = {}
    for fs in FEATURE_SETS:
        inputs = canonical_inputs(fs)
        results[fs] = _build_tool_evaluation(fs, inputs)
    return results


def _run_segment_sample() -> dict:
    telemetry = load_telemetry()
    manifest = load_split_manifest()
    split = enforce_split(telemetry, manifest)
    sample = sample_test_inputs(split.test)

    pred_tool = FuelRatePredictionTool(model_dir=DEFAULT_MODEL_DIR)
    out: dict[str, dict] = {}
    for fs in FEATURE_SETS:
        out[fs] = {"segments": {}, "n_total": 0}
        for seg, input_list in sample.items():
            rows = []
            for raw in input_list:
                try:
                    # Project the raw telemetry row onto the feature set's input
                    # columns and validate it; skip rows the tool rejects.
                    inputs = project_row_to_inputs(raw, fs)
                    tool_res = pred_tool.predict(inputs, feature_set=fs)
                    agree = prediction_agreement_with_direct_model(
                        pred_tool, inputs, fs
                    )
                    rows.append({
                        "segment": seg,
                        "tool_value": agree["tool_value"],
                        "direct_model_value": agree["direct_model_value"],
                        "abs_deviation": agree["abs_deviation"],
                        "within_tolerance": agree["within_tolerance"],
                        "finite_valid": check_finite_valid(tool_res)["pass"],
                    })
                except Exception:  # noqa: BLE001 - invalid/incompatible row
                    continue
            n_ok = sum(1 for r in rows if r.get("within_tolerance"))
            out[fs]["segments"][seg] = {
                "n": len(rows),
                "n_within_tolerance": n_ok,
                "agreement_rate": (n_ok / len(rows)) if rows else None,
            }
            out[fs]["n_total"] += len(rows)
    return out


def _generate_plot(segment_results: dict) -> list[str]:
    plt = _plt()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    segments = list(next(iter(segment_results.values()))["segments"].keys())
    x = range(len(segments))
    width = 0.35
    for i, fs in enumerate(FEATURE_SETS):
        rates = [
            segment_results[fs]["segments"][s]["agreement_rate"] or 0.0
            for s in segments
        ]
        ax.bar([p + i * width for p in x], rates, width, label=fs)
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(segments)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tool↔direct-model agreement rate")
    ax.set_title("M5.3 tool agreement by operating segment")
    ax.legend()
    fig.tight_layout()
    name = "tool_agreement_by_segment.png"
    fig.savefig(PLOTS_DIR / name, dpi=130)
    plt.close(fig)
    files.append(name)
    return files


def build_report() -> dict:
    tool_eval = _run_full_evaluation()
    segment_sample = _run_segment_sample()
    plots = _generate_plot(segment_sample)

    # Aggregate pass rates.
    summary = {}
    for fs, block in tool_eval.items():
        p = block["prediction"]
        e = block["explanation"]
        inv = p["invalid_input_rejection"]
        summary[fs] = {
            "prediction_consistent": p["repeated_consistency"]["all_identical"],
            "prediction_finite_valid": p["finite_valid"]["pass"],
            "tool_model_agreement": p["tool_vs_direct_model_agreement"]["within_tolerance"],
            "invalid_rejection_rate": (
                inv["n_rejected"] / inv["n_cases"] if inv["n_cases"] else None
            ),
            "explanation_reconstruction_ok": e["reconstruction"]["ok"],
            "attribution_direction_ok": e["attribution_direction"]["ok"],
            "explanation_safety_ok": e["safety"]["ok"],
            "explanation_prediction_consistent": e["prediction_consistency"]["within_tolerance"],
        }

    return {
        "experiment": "M5.3_prediction_explanation_reliability",
        "milestone": "M5.3",
        "protocol": (
            "evaluate the real FuelRatePredictionTool and PredictionExplanationTool "
            "on canonical valid inputs plus a small deterministic held-out segment "
            "sample; saved M1.2 models and preprocessors only; no retraining; "
            "deterministic; synthetic-data constrained"
        ),
        "non_duplication": {
            "m1_3_metrics_referenced": True,
            "m1_4_reconstruction_referenced": True,
            "m3_4_robustness_referenced": True,
            "note": (
                "Pooled/per-truck/per-condition regression metrics and global SHAP "
                "importance are already reported in M1.3/M1.4; M5.3 does not "
                "duplicate them and instead audits the agent tool layer."
            ),
        },
        "tool_evaluation": tool_eval,
        "segment_sample": segment_sample,
        "summary": summary,
        "plots": {"files": plots, "dir": str(PLOTS_DIR)},
    }


def _write_json(report: dict) -> Path:
    M5_3_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = M5_3_REPORT_DIR / "evaluation.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return path


def _write_markdown(report: dict) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# M5.3 Prediction & Explanation Reliability Evaluation\n")
    add(f"**Milestone:** {report['milestone']}\n")
    add(f"**Protocol:** {report['protocol']}\n")
    add("**Scope (non-duplication):** M1.3 pooled/per-truck/per-condition metrics "
        "and M1.4 global/per-truck SHAP importance are referenced, not recomputed. "
        "M5.3 audits the real agent tool layer.\n")

    add("## Tool-level evaluation (canonical valid inputs)\n")
    add("| Feature set | Pred consistent | Pred finite/valid | Tool↔model agreement | "
        "Invalid rejection | Recon OK | Attribution dir OK | Safety OK | "
        "Expl↔pred consistent |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for fs, s in report["summary"].items():
        add(f"| {fs} | {s['prediction_consistent']} | {s['prediction_finite_valid']} | "
            f"{s['tool_model_agreement']} | {s['invalid_rejection_rate']} | "
            f"{s['explanation_reconstruction_ok']} | {s['attribution_direction_ok']} | "
            f"{s['explanation_safety_ok']} | {s['explanation_prediction_consistent']} |")
    add("")

    add("## Segment tool-agreement sample (held-out test, deterministic)\n")
    add("| Feature set | Segment | n | agreement rate |")
    add("| --- | --- | --- | --- |")
    for fs, block in report["segment_sample"].items():
        for seg, segdata in block["segments"].items():
            rate = segdata["agreement_rate"]
            rate_s = "n/a" if rate is None else f"{rate:.3f}"
            add(f"| {fs} | {seg} | {segdata['n']} | {rate_s} |")
    add("")

    add("## Safety / constraints\n")
    add("- Invalid inputs must never produce a successful prediction (rejection "
        "rate = 1.0 across all cases).")
    add("- SHAP is reported as attribution, never causality; no guaranteed-savings "
        "claim; synthetic-training disclaimer always attached.")
    add("- All numbers are estimates from synthetic telemetry, not validated "
        "real-world measurements.")

    M5_3_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = M5_3_REPORT_DIR / "summary.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    report = build_report()
    json_path = _write_json(report)
    md_path = _write_markdown(report)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote plots: {report['plots']['files']}")

    for fs, s in report["summary"].items():
        print(f"\n[{fs}]")
        print(f"  prediction consistent = {s['prediction_consistent']}")
        print(f"  finite/valid = {s['prediction_finite_valid']}")
        print(f"  tool↔model agreement = {s['tool_model_agreement']}")
        print(f"  invalid rejection rate = {s['invalid_rejection_rate']}")
        print(f"  explanation recon OK = {s['explanation_reconstruction_ok']}")
        print(f"  attribution direction OK = {s['attribution_direction_ok']}")
        print(f"  explanation safety OK = {s['explanation_safety_ok']}")
        print(f"  explanation↔prediction consistent = {s['explanation_prediction_consistent']}")


if __name__ == "__main__":
    main()
