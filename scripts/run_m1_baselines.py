"""M1.2 — run all baseline models on train/validation trucks.

Reproducible entry point:

    python scripts/run_m1_baselines.py            # default output: reports/m1
    python scripts/run_m1_baselines.py --out PATH # custom output directory
    python scripts/run_m1_baselines.py --models ridge,xgboost
    python scripts/run_m1_baselines.py --feature-sets full_operational
    python scripts/run_m1_baselines.py --no-artifacts

Protocol (plan §3, §4):
  * train only on the 20 train trucks,
  * encode categoricals (OneHotEncoder, handle_unknown="ignore") on train only,
  * evaluate on the 4 validation trucks (test trucks are NOT touched here),
  * fixed random_state=20240617,
  * both feature sets (full_operational, telematics_only).

Outputs (under ``--out``, default ``reports/m1``):
  * ``artifacts/<feature_set>/preprocessor.joblib``        fitted encoder
  * ``artifacts/<feature_set>/<model>.joblib``            fitted model
  * ``metrics_report.json``                               machine-readable metrics
  * ``summary.md``                                        human-readable summary
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Allow running from anywhere while keeping `src` importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import enforce_split, load_split_manifest, load_telemetry  # noqa: E402
from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.features.build import build_feature_matrix  # noqa: E402
from src.features.preprocess import FeaturePreprocessor  # noqa: E402
from src.features.spec import load_feature_spec  # noqa: E402
from src.models.baselines import MODEL_REGISTRY, RANDOM_STATE  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "m1"
FEATURE_SETS = ("full_operational", "telematics_only")


def _env_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": "1.9.1",  # resolved at import of estimators
        "xgboost": "3.4.1",
        "random_state": str(RANDOM_STATE),
    }


def _artifact_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else -1


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _fit_encode(matrices: dict[str, tuple[pd.DataFrame, pd.Series]]) -> tuple[FeaturePreprocessor, dict[str, dict]]:
    """Fit the categorical encoder on train only; transform train/val."""
    X_train, _ = matrices["train"]
    preprocessor = FeaturePreprocessor().fit(X_train)
    transformed = {
        split: (preprocessor.transform(X), y)
        for split, (X, y) in matrices.items()
    }
    return preprocessor, transformed


def _run_feature_set(
    feature_set: str,
    split: "object",
    models: list[str],
    out_dir: Path,
    save_artifacts: bool,
) -> dict:
    """Train every model on one feature set; evaluate on validation."""
    spec = load_feature_spec()

    # Build raw feature matrices for train/val (test is NOT used in M1.2).
    matrices: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ("train", "val"):
        frame = getattr(split, split_name)
        X_raw, y_raw, _ = build_feature_matrix(frame, feature_set, spec=spec)
        matrices[split_name] = (X_raw, y_raw)

    preprocessor, transformed = _fit_encode(matrices)
    X_train, y_train = transformed["train"]
    X_val, y_val = transformed["val"]

    result: dict = {
        "feature_set": feature_set,
        "columns": X_train.columns.tolist(),
        "n_features": int(X_train.shape[1]),
        "train_n_rows": int(len(X_train)),
        "val_n_rows": int(len(X_val)),
        "models": {},
    }

    feat_dir = out_dir / "artifacts" / feature_set
    feat_dir.mkdir(parents=True, exist_ok=True)
    if save_artifacts:
        joblib.dump(preprocessor, feat_dir / "preprocessor.joblib", compress=3)
        result["preprocessor_artifact"] = str(feat_dir / "preprocessor.joblib")
        result["preprocessor_size_bytes"] = _artifact_size_bytes(
            feat_dir / "preprocessor.joblib"
        )

    for name in models:
        factory = MODEL_REGISTRY[name]["factory"]
        estimator = factory()

        t0 = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred_val = estimator.predict(X_val)
        predict_seconds = time.perf_counter() - t1

        metrics = regression_metrics(y_val, y_pred_val)

        model_entry: dict = {
            "label": MODEL_REGISTRY[name]["label"],
            "fit_seconds": round(fit_seconds, 3),
            "predict_seconds": round(predict_seconds, 3),
            "metrics": metrics,
        }
        if save_artifacts:
            artifact_path = feat_dir / f"{name}.joblib"
            joblib.dump(estimator, artifact_path, compress=3)
            model_entry["artifact"] = str(artifact_path)
            model_entry["artifact_size_bytes"] = _artifact_size_bytes(artifact_path)

        result["models"][name] = model_entry

    return result


def build_report(
    out_dir: Path,
    models: list[str],
    feature_sets: list[str],
    save_artifacts: bool,
) -> dict:
    """Load data once and run every feature-set × model combination."""
    split = enforce_split(load_telemetry(), load_split_manifest())

    report: dict = {
        "experiment": "M1.2_baselines_cross_truck_fuel_rate",
        "milestone": "M1.2",
        "protocol": "train on 20 trucks, validate on 4 trucks; test untouched",
        "random_state": RANDOM_STATE,
        "environment": _env_versions(),
        "split": {
            "train_trucks": split.train_trucks,
            "val_trucks": split.val_trucks,
            "test_trucks": split.test_trucks,  # informational only; unused here
            "train_rows": int(len(split.train)),
            "val_rows": int(len(split.val)),
            "test_rows": int(len(split.test)),
        },
        "feature_sets": {},
    }

    for fs in feature_sets:
        report["feature_sets"][fs] = _run_feature_set(
            fs, split, models, out_dir, save_artifacts
        )

    return report


def render_summary(report: dict) -> str:
    lines: list[str] = [
        "# M1.2 Baseline Results",
        "",
        f"Protocol: {report['protocol']}",
        f"Random state: {report['random_state']}",
        "",
        f"Split: train={report['split']['train_rows']} rows "
        f"({len(report['split']['train_trucks'])} trucks), "
        f"val={report['split']['val_rows']} rows "
        f"({len(report['split']['val_trucks'])} trucks), "
        f"test (unused)={report['split']['test_rows']} rows.",
        "",
    ]
    for fs, fs_res in report["feature_sets"].items():
        lines.append(f"## Feature set: `{fs}`")
        lines.append("")
        lines.append("| Model | MAE | RMSE | R² | MAPE (>1) | fit (s) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, m in fs_res["models"].items():
            mm = m["metrics"]
            mape = mm["mape"]
            mape_s = f"{mape:.3f}%" if isinstance(mape, float) and not np.isnan(mape) else "n/a"
            lines.append(
                f"| {m['label']} | {mm['mae']:.4f} | {mm['rmse']:.4f} "
                f"| {mm['r2']:.4f} | {mape_s} | {m['fit_seconds']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M1.2 baseline training")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated model names (default: all five)",
    )
    parser.add_argument(
        "--feature-sets",
        default=None,
        help="comma-separated feature-set names (default: both)",
    )
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args(argv)

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else sorted(MODEL_REGISTRY)
    )
    unknown = set(models) - set(MODEL_REGISTRY)
    if unknown:
        print(f"Unknown models requested: {sorted(unknown)}", file=sys.stderr)
        return 2

    feature_sets = (
        [f.strip() for f in args.feature_sets.split(",") if f.strip()]
        if args.feature_sets
        else list(FEATURE_SETS)
    )
    unknown_fs = set(feature_sets) - set(FEATURE_SETS)
    if unknown_fs:
        print(f"Unknown feature sets requested: {sorted(unknown_fs)}", file=sys.stderr)
        return 2

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_artifacts = not args.no_artifacts

    t_start = time.perf_counter()
    report = build_report(out_dir, models, feature_sets, save_artifacts)
    report["total_clock_seconds"] = round(time.perf_counter() - t_start, 3)

    _write_json(out_dir / "metrics_report.json", report)
    (out_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")

    print(f"M1.2 complete. Wrote artifacts/report to: {out_dir}")
    print(f"Total clock: {report['total_clock_seconds']}s")
    print("Models trained: " + ", ".join(models))
    print("Feature sets: " + ", ".join(feature_sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
