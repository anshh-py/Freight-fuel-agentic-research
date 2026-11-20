"""M5.5 runner — generate reports/m5/m5_5/evaluation.json + summary.md.

Usage:
    python scripts/run_m5_5_ablation_study.py

Runs the full deterministic ablation study on the real service and writes the
reports. No LLM, no API, no retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.m5_5_ablation_study import build_report  # noqa: E402


def main() -> None:
    report = build_report()
    print("M5.5 ablation study complete.")
    ev = report["variants"]["A_full_pipeline"]["results"]
    print(f"  baseline overall: {ev['overall_score']:.3f} "
          f"({ev['passed_criteria']}/{ev['applicable_criteria']})")
    for vid, diff in report["differences"].items():
        d = diff["overall"]["delta_pp"]
        print(f"  {vid}: overall Δpp = {d:+.2f}" if d is not None
              else f"  {vid}: overall Δpp = n/a")
    print(f"  reports written to "
          f"{PROJECT_ROOT / 'reports' / 'm5' / 'm5_5'}")


if __name__ == "__main__":
    main()
