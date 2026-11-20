"""M5.1 — build the expanded driver-conversation evaluation benchmark.

Deterministic builder. Reads the curated scenario data
(:mod:`src.evaluation.m5_1_scenarios`), assembles the benchmark document,
validates schema/labels/duplicates/split-leakage, computes category and split
counts, and writes:

* ``reports/m5/benchmark/driver_conversation_benchmark.json`` — full dataset.
* ``reports/m5/benchmark/summary.md`` — category/split counts + methodology.

No model is loaded, trained, or invoked. No frozen M1/M2/M3/M4 file is
modified. All claims are limited to the synthetic research environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.m5_1_driver_benchmark import (
    CATEGORIES,
    PROJECT_ROOT,
    validate_benchmark,
)
from src.evaluation.m5_1_scenarios import SCENARIOS

DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "m5" / "benchmark"
SEED_NOTE = "deterministic (fixed scenario data; no randomness)"


def build_benchmark() -> dict:
    """Assemble the full benchmark document and validate it."""
    cases = list(SCENARIOS)
    dev_count = sum(1 for c in cases if c["split"] == "dev")
    held_count = sum(1 for c in cases if c["split"] == "heldout")

    category_counts: dict[str, int] = {}
    for c in cases:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1

    turn_counts = {"single_turn": 0, "multi_turn": 0}
    for c in cases:
        key = "single_turn" if c["turn_count"] == 1 else "multi_turn"
        turn_counts[key] += 1

    doc = {
        "dataset": "M5.1 driver-conversation evaluation benchmark",
        "milestone": "M5.1",
        "evaluation_target": "M4 deterministic driver-feedback agent",
        "generated_by": "scripts/run_m5_1_build_benchmark.py",
        "generation_methodology": (
            "Curated, hand-authored labeled scenarios (no LLM). Deterministic; "
            "scenario ids and dev/held-out split are fixed. Coverage spans the "
            "fifteen milestone categories. Held-out cases target semantically "
            "distinct edges/phrasings to avoid near-duplicate leakage."
        ),
        "limitations": (
            "All scenarios are synthetic research-environment cases. They do "
            "not represent real-world driver behaviour, and the agent's "
            "outputs are model estimates / synthetic-corpus knowledge, not "
            "real-world observations or manufacturer guidance."
        ),
        "schema": {
            "case": [
                "scenario_id", "split (dev|heldout)", "category", "turn_count",
                "turns[]", "generation_note",
            ],
            "turn": [
                "turn_index", "speaker", "text", "provided_inputs",
                "expected_intent", "expected_tools", "expected_required_inputs",
                "expected_clarification", "expected_abstention",
                "expected_grounding", "expected_safety_outcome", "rationale",
            ],
        },
        "categories": dict(sorted(CATEGORIES.items())),
        "split_counts": {"dev": dev_count, "heldout": held_count},
        "category_counts": dict(sorted(category_counts.items())),
        "turn_counts": turn_counts,
        "n_cases": len(cases),
        "n_turns": sum(c["turn_count"] for c in cases),
        "cases": cases,
    }

    errors = validate_benchmark(doc)
    if errors:
        raise RuntimeError(f"benchmark validation failed ({len(errors)}):\n" + "\n".join(errors[:20]))
    return doc


def _markdown_summary(doc: dict) -> str:
    lines: list[str] = []
    lines.append("# M5.1 Driver-Conversation Evaluation Benchmark")
    lines.append("")
    lines.append("**Milestone:** M5.1")
    lines.append("**Evaluation target:** M4 deterministic driver-feedback agent (no LLM)")
    lines.append("**Generation:** curated, hand-authored, deterministic (no LLM; fixed ids/split)")
    lines.append("")
    lines.append("## Size")
    lines.append("")
    lines.append(f"- Cases: **{doc['n_cases']}** ({doc['split_counts']['dev']} dev / {doc['split_counts']['heldout']} held-out)")
    lines.append(f"- Turns: **{doc['n_turns']}**")
    lines.append(f"- Single-turn cases: {doc['turn_counts']['single_turn']}; multi-turn: {doc['turn_counts']['multi_turn']}")
    lines.append("")
    lines.append("## Category coverage")
    lines.append("")
    lines.append("| Category | Cases |")
    lines.append("| --- | --- |")
    for cat, note in doc["categories"].items():
        lines.append(f"| {cat} ({note}) | {doc['category_counts'].get(cat, 0)} |")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(doc["generation_methodology"])
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(doc["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = build_benchmark()

    json_path = out_dir / "driver_conversation_benchmark.json"
    json_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_path = out_dir / "summary.md"
    md_path.write_text(_markdown_summary(doc), encoding="utf-8")

    # Report category/split counts to the console.
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"cases={doc['n_cases']} (dev={doc['split_counts']['dev']}, "
          f"heldout={doc['split_counts']['heldout']}) turns={doc['n_turns']}")
    print("category counts:", doc["category_counts"])
    print("validation: PASSED (0 errors)")


if __name__ == "__main__":
    main()
