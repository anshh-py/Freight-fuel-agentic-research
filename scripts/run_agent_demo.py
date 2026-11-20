"""M2.2 — run the agent locally and print example responses.

Usage: ``python scripts/run_agent_demo.py``

This is a demonstrator, not a deployment. It builds the default orchestrator
(real saved M1.2 models + synthetic corpus retrieval + SHAP explanation) and
runs a small set of representative queries, writing the structured responses
to ``reports/m2/agent_demo_examples.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent.orchestrator import AgentOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "reports" / "m2" / "agent_demo_examples.json"

FULL_INPUTS = {
    "vehicle_class": "32T",
    "gross_vehicle_weight_kg": 40000.0,
    "payload_kg": 10000.0,
    "vehicle_mass_kg": 30000.0,
    "speed_kph": 75.0,
    "acceleration_mps2": 0.1,
    "road_grade_pct": 1.0,
    "ignition_status": 1,
    "engine_rpm": 1500.0,
    "engine_load_pct": 60.0,
    "throttle_position_pct": 40.0,
    "torque_nm": 1200.0,
    "coolant_temp_c": 85.0,
    "intake_air_temp_c": 30.0,
    "ambient_temp_c": 20.0,
}

DEMO_QUERIES = [
    {
        "tag": "prediction",
        "query": "Predict fuel consumption for this operating condition.",
        "inputs": FULL_INPUTS,
    },
    {
        "tag": "knowledge",
        "query": "Why does road grade affect fuel consumption on heavy trucks?",
        "inputs": None,
    },
    {
        "tag": "combined_knowledge_prediction",
        "query": "Predict fuel consumption and explain how engine load affects it.",
        "inputs": FULL_INPUTS,
    },
    {
        "tag": "explanation",
        "query": "Explain this prediction.",
        "inputs": FULL_INPUTS,
    },
    {
        "tag": "unsupported",
        "query": "What is the official manufacturer torque limit for this truck?",
        "inputs": None,
    },
    {
        "tag": "ambiguous",
        "query": "banana split",
        "inputs": None,
    },
]


def main() -> None:
    orchestrator = AgentOrchestrator()
    examples: dict = {
        "pipeline": "M2.2 agent demo",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "routing_backend": "deterministic",
        "is_llm": False,
        "examples": [],
    }
    for ex in DEMO_QUERIES:
        resp = orchestrator.handle(ex["query"], ex["inputs"])
        examples["examples"].append(
            {
                "tag": ex["tag"],
                "query": ex["query"],
                "has_structured_inputs": bool(ex["inputs"]),
                "response": resp.to_dict(),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(examples, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for ex in examples["examples"]:
        r = ex["response"]
        print(
            f"- [{ex['tag']}] intent={r['detected_intent']} "
            f"tools={r['selected_tools']} -> {r['final_response'][:90]}"
        )


if __name__ == "__main__":
    main()
