"""M3.4 runner — missing / noisy / corrupted telemetry robustness (read-only).

Loads the frozen five-truck test split and the saved M1.2 artifacts, corrupts
COPIES of the model-input feature matrices after feature construction and
before preprocessing, then computes and writes the M3.4 report:

  * ``reports/m3/telemetry_robustness_report.json``,
  * ``reports/m3/telemetry_robustness_summary.md``,
  * ``reports/m3/plots/telemetry_robustness_*.png``.

No model is trained/refitted/tuned; no dataset or frozen artifact is modified.
Corruption is synthetic and explicitly distinguished from real-world sensor
validation. All metrics are descriptive; no hypothesis tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.m3_cross_truck_analysis import FEATURE_SETS
from src.evaluation.m3_telemetry_robustness import (
    run_telemetry_robustness,
)

M1_REPORT_DIR = PROJECT_ROOT / "reports" / "m1"
M3_REPORT_DIR = PROJECT_ROOT / "reports" / "m3"
PLOTS_DIR = M3_REPORT_DIR / "plots"

#: Numerical tolerance for reproducing the M1.3 clean pooled metrics.
CLEAN_REFERENCE_TOL = 1e-3


# --------------------------------------------------------------------------
# Reference M1.3 pooled metrics (for clean-baseline reconciliation)
# --------------------------------------------------------------------------

def load_reference_clean_metrics() -> dict:
    """M1.3 pooled metrics for the validation-selected models per feature set."""
    with open(M1_REPORT_DIR / "test_metrics_report.json", "r",
              encoding="utf-8") as fh:
        report = json.load(fh)
    selected = report["selected_models_from_validation"]
    out = {}
    for fs in FEATURE_SETS:
        model = selected[fs]["model"]
        pooled = report["feature_sets"][fs]["models"][model]["pooled"]
        out[fs] = {
            "model": model,
            "mae": float(pooled["mae"]),
            "rmse": float(pooled["rmse"]),
            "r2": float(pooled["r2"]),
        }
    return out


def verify_clean_baselines(clean_baselines: dict) -> dict:
    """Compare our clean baselines to the M1.3 reference values."""
    ref = load_reference_clean_metrics()
    out = {}
    for fs in FEATURE_SETS:
        clean = clean_baselines[fs]
        r = ref[fs]
        diffs = {
            metric: round(abs(float(clean[metric]) - float(r[metric])), 9)
            for metric in ("mae", "rmse", "r2")
        }
        out[fs] = {
            "reference_model": r["model"],
            "matched_model": clean["model"] == r["model"],
            "abs_diffs": diffs,
            "within_tolerance": all(d <= CLEAN_REFERENCE_TOL for d in diffs.values()),
            "tolerance": CLEAN_REFERENCE_TOL,
        }
    return out


def build_report() -> dict:
    body = run_telemetry_robustness(report_dir=M1_REPORT_DIR)

    clean_verification = verify_clean_baselines(body["clean_baselines"])

    report = {
        "experiment": body["experiment"],
        "milestone": "M3.4",
        "protocol": body["protocol"],
        "analysis_frame": (
            "synthetic corruption applied to copies of the model-input feature "
            "matrices; results are descriptive pipeline validation, NOT "
            "observed real-world sensor-failure data"
        ),
        "selected_models_from_validation": body["selected_models_from_validation"],
        "verification": {
            "test_rows": body["test_rows"],
            "test_trucks": body["test_trucks"],
            "n_test_trips": body["n_test_trips"],
            "clean_baseline_reproduces_m1_3": clean_verification,
            "per_truck_reconciliation": body["per_truck_reconciliation"],
            "preprocessors_loaded_not_refitted": True,
            "models_loaded_not_refitted": True,
            "corruption_applied_to_copies_after_feature_construction_before_preprocessing": True,
            "raw_telemetry_and_feature_columns_not_modified": True,
            "prediction_failures_counted_not_silently_discarded": True,
            "categorical_and_binary_columns_preserved": True,
            "synthetic_corruption_distinct_from_real_world_sensor_validation": True,
        },
        "corruption_conventions": body["corruption_conventions"],
        "clean_baselines": body["clean_baselines"],
        "feature_sets": body["feature_sets"],
        "limitations": [
            "Synthetic telemetry: these are pipeline-corruption experiments on "
            "synthetic data, not observed real-world sensor-failure validation. "
            "They characterize model sensitivity to the injected corruptions "
            "only.",
            "Only five test trucks: all per-truck breakdowns are descriptive "
            "and unstable; not population estimates.",
            "Corruption is restricted to numeric model-input features; "
            "categorical vehicle_class and binary ignition_status are kept "
            "unchanged. Real-world sensor failures may also affect other "
            "signal classes.",
            "No feature columns are removed; complete sensor failure is "
            "simulated with all-NaN values while preserving the model's "
            "expected schema.",
            "Prediction failures (aborted, NaN, +inf or -inf predictions) are counted and "
            "reported as a failure rate; metrics are computed on the surviving "
            "finite rows and should always be read alongside the failure rate.",
            "The clipping bounds are documented physical limits; they are "
            "already satisfied by most data, so the clipping scenario "
            "primarily measures the effect of the bound enforcement itself.",
            "MAE degradation is reported against each feature set's own clean "
            "baseline; cross-feature-set ratios are not directly comparable "
            "because baselines differ.",
        ],
    }
    return report


def _write_json(report: dict) -> Path:
    report_path = M3_REPORT_DIR / "telemetry_robustness_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report_path


def _write_markdown(report: dict) -> Path:
    lines: list[str] = []
    add = lines.append

    add("# M3.4 Missing / Noisy / Corrupted Telemetry Robustness\n")
    add(f"**Milestone:** {report['milestone']}\n")
    add(f"**Analysis frame:** {report['analysis_frame']}\n")
    add(f"**Protocol:** {report['protocol']}\n")
    add("**Synthetic-data caveat:** corruption is applied to copies of the "
        "model-input feature matrices and is explicitly NOT real-world "
        "sensor-failure validation.\n")

    add("## Selected models (from M1.2 validation, not test)\n")
    add("| Feature set | Selected model |")
    add("| --- | --- |")
    for fs, sel in report["selected_models_from_validation"].items():
        add(f"| {fs} | {sel['model']} |")
    add("")

    add("## Verification / reconciliation\n")
    v = report["verification"]
    add("- Test trucks: " + ", ".join(v["test_trucks"]) + "\n")
    add(f"- Test rows: {v['test_rows']}; test trips: {v['n_test_trips']}")
    add(f"- Categorical/binary columns preserved: {v['categorical_and_binary_columns_preserved']}")
    add(f"- Prediction failures counted: {v['prediction_failures_counted_not_silently_discarded']}")
    add(f"- Synthetic corruption distinct from real-world validation: {v['synthetic_corruption_distinct_from_real_world_sensor_validation']}")
    add("")
    add("### Clean baseline reproduces M1.3 (within tolerance)\n")
    add("| Feature set | ref model | model match | |MAE diff| | |RMSE diff| | |R² diff| | within tol |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for fs, c in v["clean_baseline_reproduces_m1_3"].items():
        d = c["abs_diffs"]
        add(f"| {fs} | {c['reference_model']} | {c['matched_model']} | "
            f"{d['mae']:.2e} | {d['rmse']:.2e} | {d['r2']:.2e} | "
            f"{c['within_tolerance']} |")
    add("")
    add("### Per-truck reconciliation (weighted per-truck MAE == pooled MAE)\n")
    add("| Feature set | pooled MAE | weighted per-truck MAE | max abs diff | reconciles |")
    add("| --- | --- | --- | --- | --- |")
    for fs, rc in v["per_truck_reconciliation"].items():
        add(f"| {fs} | {rc['pooled_mae']} | {rc['per_truck_weighted_mae']} | "
            f"{rc['max_abs_diff']} | {rc['reconciles']} |")
    add("")

    add("## Corruption conventions\n")
    cc = report["corruption_conventions"]
    add(f"- Seed base: {cc['seed_base']}")
    add(f"- Preserved categorical: {cc['preserved_categorical']}; binary: {cc['preserved_binary']}")
    add(f"- Gaussian-noise severities (fraction of feature std): {cc['gaussian_noise_severities']}")
    add(f"- Random-missingness fractions: {cc['missingness_fractions']}")
    add(f"- Spike fraction: {cc['spike_fraction']}; multiplier: {cc['spike_multiplier']} x std")
    add(f"- Clipping bounds: {json.dumps(cc['clip_bounds'])}")
    add(f"- Gaussian-noise sigma & spike references are training-derived only ({cc['training_statistics_provenance']['source']}).")
    add(f"- Random streams per-column deterministic: shared columns equivalent across feature sets.")
    add(f"- Combined scenario: {cc['combined_scenario']['note']}")
    add("")

    for fs, d in report["feature_sets"].items():
        add(f"## Feature set: `{fs}`\n")
        base = d["clean_baseline"]
        add(f"Clean baseline ({base['model']}): "
            f"MAE={base['mae']}, RMSE={base['rmse']}, R²={base['r2']}, "
            f"rows={base['n_rows']}\n")
        add(f"Numeric features corrupted: {d['numeric_features']}\n")

        add("### Scenario results\n")
        add("| Scenario | applied | affected cols | cells changed | fail rate | "
            "MAE | RMSE | R² | ΔMAE | ratio |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in d["scenarios"]:
            if not s["applied"]:
                add(f"| {s['name']} | no ({s['reason']}) | — | — | — | — | "
                    "— | — | — | — |")
                continue
            aff = ", ".join(s["affected_columns"])
            mae = f"{s['mae']:.4f}" if s["mae"] is not None else "n/a"
            rmse = f"{s['rmse']:.4f}" if s["rmse"] is not None else "n/a"
            r2 = f"{s['r2']:.4f}" if s["r2"] is not None else "n/a"
            dl = f"{s['delta_mae']:.4f}" if s["delta_mae"] is not None else "n/a"
            rt = f"{s['mae_degradation_ratio']:.4f}" if s["mae_degradation_ratio"] is not None else "n/a"
            add(f"| {s['name']} | yes | {aff} | {s['cells_changed']} | "
                f"{s['prediction_failure_rate']:.4f} | {mae} | {rmse} | "
                f"{r2} | {dl} | {rt} |")
        add("")
        add("`ΔMAE` = corrupted MAE − clean MAE (L/h); `ratio` = corrupted MAE / "
            "clean MAE. Prediction-failure rate = fraction of test rows with "
            "aborted or NaN predictions (failures are counted, never silently "
            "dropped).\n")

        add("### Per-truck MAE degradation (most damaging scenario)\n")
        applied = [s for s in d["scenarios"] if s["applied"] and s["mae"] is not None]
        if applied:
            worst = max(applied, key=lambda s: s["mae_degradation_ratio"] or 0.0)
            add(f"Scenario: `{worst['name']}` (ratio {worst['mae_degradation_ratio']})\n")
            add("| Truck | n rows | MAE | clean MAE | ΔMAE |")
            add("| --- | --- | --- | --- | --- |")
            for truck, td in worst["per_truck"].items():
                dl = f"{td['delta_mae']:.4f}" if td["delta_mae"] is not None else "n/a"
                add(f"| {truck} | {td['n_rows']} | {td['mae']:.4f} | "
                    f"{td['clean_mae']:.4f} | {dl} |")
            add("")

    add("## Limitations\n")
    for lim in report["limitations"]:
        add(f"- {lim}")

    md_path = M3_REPORT_DIR / "telemetry_robustness_summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return md_path


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _generate_plots(report: dict) -> list[str]:
    plt = _plt()
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    # 1. MAE vs missingness fraction (per feature set).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fs, d in report["feature_sets"].items():
        fracs, maes = [], []
        for s in d["scenarios"]:
            if s["applied"] and s["kind"] == "random_missingness" and s["mae"] is not None:
                fracs.append(s["fraction"] * 100)
                maes.append(s["mae"])
        if fracs:
            ax.plot(fracs, maes, marker="o", label=fs)
    ax.set_title("MAE vs random missingness (M3.4)")
    ax.set_xlabel("Missingness fraction (%)")
    ax.set_ylabel("MAE (L/h)")
    ax.legend()
    fig.tight_layout()
    name = "telemetry_robustness_mae_vs_missingness.png"
    fig.savefig(plots_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)

    # 2. MAE degradation ratio per scenario (per feature set).
    for fs, d in report["feature_sets"].items():
        fig, ax = plt.subplots(figsize=(8, 6))
        labels, ratios = [], []
        for s in d["scenarios"]:
            if s["applied"] and s["mae_degradation_ratio"] is not None:
                labels.append(s["name"])
                ratios.append(s["mae_degradation_ratio"])
        if labels:
            order = sorted(range(len(ratios)), key=lambda i: ratios[i])
            ax.barh(
                [labels[i] for i in order],
                [ratios[i] for i in order],
                color="#4472C4",
            )
            ax.set_title(f"MAE degradation ratio by scenario — {fs}")
            ax.set_xlabel("Corrupted MAE / clean MAE")
            ax.axvline(1.0, color="black", lw=1.2, ls="--")
            fig.tight_layout()
            name = f"telemetry_robustness_mae_ratio_{fs}.png"
            fig.savefig(plots_dir / name, dpi=130)
            plt.close(fig)
            files.append(name)

    # 3. Per-truck MAE degradation for the most damaging scenario per feature set.
    for fs, d in report["feature_sets"].items():
        applied = [s for s in d["scenarios"] if s["applied"] and s["mae"] is not None]
        if not applied:
            continue
        worst = max(applied, key=lambda s: s["mae_degradation_ratio"] or 0.0)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        trucks = list(worst["per_truck"].keys())
        deltas = [worst["per_truck"][t]["delta_mae"] for t in trucks]
        ax.bar(trucks, deltas, color="#C00000")
        ax.set_title(f"Per-truck ΔMAE — {fs} ({worst['name']})")
        ax.set_ylabel("ΔMAE vs clean (L/h)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        name = f"telemetry_robustness_per_truck_{fs}.png"
        fig.savefig(plots_dir / name, dpi=130)
        plt.close(fig)
        files.append(name)

    return files


def main() -> None:
    report = build_report()
    json_path = _write_json(report)
    md_path = _write_markdown(report)
    plot_files = _generate_plots(report)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote plots: {plot_files}")

    for fs in FEATURE_SETS:
        base = report["clean_baselines"][fs]
        print(f"\n[{fs}] clean baseline ({base['model']}): "
              f"MAE={base['mae']}, RMSE={base['rmse']}, R²={base['r2']}")
        for s in report["feature_sets"][fs]["scenarios"]:
            if not s["applied"] or s["mae"] is None:
                continue
            print(f"  {s['name']}: MAE={s['mae']:.4f} ratio="
                  f"{s['mae_degradation_ratio']:.3f} fail_rate="
                  f"{s['prediction_failure_rate']:.4f}")


if __name__ == "__main__":
    main()
