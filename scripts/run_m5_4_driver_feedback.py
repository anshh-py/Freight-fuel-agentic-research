"""Run the M5.4 driver-feedback usefulness & safety evaluation.

Generates ``reports/m5/m5_4/evaluation.json`` and ``summary.md`` using the
real DriverConversationService (real saved M1.2 models / SHAP / synthetic RAG).
Deterministic. No LLM / API / database / retraining.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.m5_4_driver_feedback import build_report  # noqa: E402


def main() -> int:
    report = build_report()
    ev = report["evaluation"]
    print("[M5.4] evaluation written to reports/m5/m5_4/evaluation.json")
    print(f"[M5.4] overall score = {ev['overall_score']:.3f} "
          f"({ev['passed_criteria']}/{ev['applicable_criteria']} applicable)")
    print("[M5.4] safety-critical scenario count = "
          f"{ev['scenarios_with_safety_critical_failures']}")
    for dim, d in ev["dimensions"].items():
        score = f"{d['score']:.3f}" if d["score"] is not None else "n/a"
        print(f"[M5.4] {dim}: {score} ({d['passed']}/{d['applicable']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
