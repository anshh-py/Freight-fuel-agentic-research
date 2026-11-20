"""M4.5 runner — end-to-end driver interaction demo & evaluation.

Builds the real M2 tool stack (saved XGBoost models, SHAP explanation,
synthetic RAG retrieval) behind the M4.4 ``DriverConversationService``,
runs the deterministic multi-turn driver scenarios, and writes:

  * ``reports/m4/driver_interaction_report.json``
  * ``reports/m4/driver_interaction_summary.md``
  * ``reports/m4/plots/driver_interaction_scenarios.png``

No models are retrained, no preprocessors refitted, and no frozen M1/M2/M3
artifact is modified. The demo is synthetic-data-grounded and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.driver_conversation import DriverConversationService
from src.evaluation.m4_5_driver_interaction import DEFAULT_REPORT_DIR, build_report


def main() -> None:
    # Real M2 tools: saved models + SHAP + synthetic RAG corpus.
    service = DriverConversationService(default_feature_set="full_operational")
    report = build_report(service=service, report_dir=DEFAULT_REPORT_DIR)

    print(f"Wrote: {DEFAULT_REPORT_DIR / 'driver_interaction_report.json'}")
    print(f"Wrote: {DEFAULT_REPORT_DIR / 'driver_interaction_summary.md'}")
    print(f"Wrote: {DEFAULT_REPORT_DIR / 'plots' / 'driver_interaction_scenarios.png'}")

    ev = report["evaluation"]
    print(f"\nEvaluation: {ev['passed_checks']}/{ev['total_checks']} checks passed; "
          f"{ev['scenarios_completed']}/{ev['scenarios_total']} scenarios completed")

    for sc in report["scenarios"]:
        e = sc["evaluation"]
        print(f"  - {sc['name']}: {e['passed_checks']}/{e['total_checks']} "
              f"(completed={e['completion_ok']})")
        for t in sc["turns"]:
            print(f"      turn {t['turn']}: intent={t['intent']} "
                  f"tools={t['executed_tools']} delivered={t['delivered']}")


if __name__ == "__main__":
    main()
