"""M5.1 — expanded and diversified driver-conversation evaluation benchmark.

This module defines a *new* benchmark for the M4 deterministic driver-feedback
agent. It is deliberately broader than the existing artifacts:

* M2.4 (:code:`reports/m2/agent_evaluation/benchmark.json`, 49 scenarios)
  targets single-turn intent routing and response-level safety.
* The 550 dataset query rows are single-turn tool-level cases (English only).
* M4.5 is a 3-scenario end-to-end demo.

M5.1 adds *multi-turn* driver-conversation cases with per-turn expected labels,
diverse wording (Hindi / Hinglish-style, typos, shorthand, realistic incomplete
requests), and the fourteen coverage dimensions required by the milestone:

  prediction requests, partial inputs, invalid inputs, multi-turn
  clarification/context retention, explanation requests, knowledge/maintenance
  guidance, combined intents, unsupported requests, ambiguous requests,
  contradictory inputs, out-of-range/malformed values, abstention, safety-
  sensitive questions, fuel-saving-guarantee elicitation, and OEM-specific
  claim elicitation from the synthetic corpus.

Design guarantees:

* Deterministic: scenario IDs and the dev/held-out split are fixed in this
  module (no randomness, no wall-clock). Rerunning the builder reproduces the
  exact same JSON.
* Dev and held-out cases are separated *semantically*: held-out cases target
  distinct coverage edges and phrasings so they are not near-duplicates of
  development cases.
* The benchmark is a static labeled dataset; it runs no models and never calls
  the frozen M1/M2/M3/M4 implementations.
* All claims are explicitly limited to the synthetic research environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Canonical vehicle/trip inputs (valid against the M2 strict validator)
# ---------------------------------------------------------------------------

#: A complete valid input set for the ``full_operational`` feature set.
VALID_FULL: dict[str, Any] = {
    "vehicle_class": "40T",
    "gross_vehicle_weight_kg": 40000.0,
    "payload_kg": 12000.0,
    "vehicle_mass_kg": 28000.0,
    "speed_kph": 75.0,
    "acceleration_mps2": 0.3,
    "road_grade_pct": 1.0,
    "ignition_status": 1,
    "engine_rpm": 1400.0,
    "engine_load_pct": 60.0,
    "throttle_position_pct": 45.0,
    "torque_nm": 1200.0,
    "coolant_temp_c": 85.0,
    "intake_air_temp_c": 25.0,
    "ambient_temp_c": 18.0,
}

_TELE_FIELDS = {
    "vehicle_class", "gross_vehicle_weight_kg", "payload_kg",
    "vehicle_mass_kg", "speed_kph", "acceleration_mps2",
    "road_grade_pct", "ignition_status", "coolant_temp_c",
    "intake_air_temp_c", "ambient_temp_c",
}

#: A complete valid input set for the ``telematics_only`` feature set.
VALID_TELE: dict[str, Any] = {
    k: v for k, v in VALID_FULL.items() if k in _TELE_FIELDS
}

#: Fields that are REQUIRED for a full_operational prediction (all inputs).
FULL_REQUIRED: list[str] = list(VALID_FULL.keys())

#: Fields that are REQUIRED for a telematics_only prediction.
TELE_REQUIRED: list[str] = list(VALID_TELE.keys())


def subset(fields: list[str], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a partial input dict containing only ``fields`` (driver-supplied)."""
    src = base if base is not None else VALID_FULL
    return {k: src[k] for k in fields if k in src}


# ---------------------------------------------------------------------------
# Categories registry (canonical coverage dimensions)
# ---------------------------------------------------------------------------

CATEGORIES: dict[str, str] = {
    "prediction_request": "Direct fuel-rate prediction request with complete inputs.",
    "partial_input": "Driver supplies only a subset of required inputs.",
    "invalid_input": "Driver supplies present but invalid values (range/type).",
    "multi_turn_clarification": "Multi-turn dialogue: clarify -> supply -> predict.",
    "explanation_request": "Request for why a prediction is what it is (SHAP).",
    "knowledge_guidance": "Knowledge / maintenance / practical efficiency guidance.",
    "combined_intent": "Request that needs more than one tool.",
    "unsupported": "Out-of-scope request that must stay unsupported.",
    "ambiguous": "Under-specified request that must ask a targeted question.",
    "contradictory_input": "Driver gives conflicting/contradictory values.",
    "out_of_range_malformed": "Out-of-range or malformed field values.",
    "abstention": "System must decline/abstain rather than guess.",
    "safety_sensitive": "Question touching physical/legal/operational safety.",
    "fuel_saving_guarantee": "Attempt to elicit an unsupported savings guarantee.",
    "oem_claim": "Attempt to get OEM/manufacturer specs from the synthetic corpus.",
}

#: Intent values the labeled agent can emit (must match DriverIntent values).
VALID_INTENTS: frozenset[str] = frozenset(
    {
        "report_behavior", "predict_fuel", "explain_prediction",
        "knowledge_query", "combined_request", "clarifying",
        "unsupported", "ambiguous",
    }
)

#: Tool names the agent can execute (must match the M2 tool registry).
VALID_TOOLS: frozenset[str] = frozenset(
    {"fuel_rate_prediction", "prediction_explanation", "knowledge_retrieval"}
)

#: Grounding expectations that may appear on a turn.
VALID_GROUNDING: frozenset[str] = frozenset(
    {
        "synthetic_disclaimer", "source_attribution", "synthetic_corpus_notice",
        "non_causal_stated", "prediction_value_preserved", "evidence_sufficiency",
        "insufficient_evidence_stated", "oem_not_claimed",
    }
)

#: Safety outcomes for a turn.
VALID_SAFETY: frozenset[str] = frozenset(
    {
        "no_tool_execution_while_incomplete",
        "unsupported_refused",
        "ambiguous_clarified",
        "delivered_grounded",
        "not_delivered_validation_failed",
        "abstention_stated",
        "no_fabrication",
        "no_causal_reading_of_shap",
        "no_oem_claim_from_synthetic_corpus",
        "no_fuel_savings_guarantee",
    }
)


# ---------------------------------------------------------------------------
# Benchmark document schema validation
# ---------------------------------------------------------------------------

_REQUIRED_CASE_FIELDS = (
    "scenario_id", "split", "category", "turn_count", "turns",
    "generation_note",
)
_REQUIRED_TURN_FIELDS = (
    "turn_index", "speaker", "text", "provided_inputs",
    "user_intent", "expected_intent", "expected_tools",
    "expected_required_inputs",
    "expected_clarification", "expected_abstention", "expected_grounding",
    "expected_safety_outcome", "rationale",
)


def validate_benchmark(doc: dict[str, Any]) -> list[str]:
    """Return a list of schema/label/leakage errors (empty when valid).

    Deterministic and side-effect free. Validates schema, label consistency,
    duplicate text, and dev/held-out split leakage.
    """
    errors: list[str] = []
    cases = doc.get("cases", [])
    if not isinstance(cases, list) or not cases:
        errors.append("document has no 'cases' list")
    seen_ids: dict[str, int] = {}
    seen_text: dict[str, str] = {}   # cross-case, whole benchmark
    dev_text: dict[str, str] = {}
    held_text: dict[str, str] = {}
    seen_text_in_case: dict[str, set[str]] = {}  # case-local repeats are allowed
    for case in cases:
        if not isinstance(case, dict):
            errors.append("case is not a dict")
            continue
        cid = case.get("scenario_id")
        if not cid:
            errors.append("case missing scenario_id")
            continue
        if cid in seen_ids:
            errors.append(f"duplicate scenario_id: {cid}")
        seen_ids[cid] = seen_ids.get(cid, 0) + 1
        for f in _REQUIRED_CASE_FIELDS:
            if f not in case:
                errors.append(f"{cid}: missing required case field {f!r}")
        split = case.get("split")
        if split not in ("dev", "heldout"):
            errors.append(f"{cid}: invalid split {split!r} (expected dev|heldout)")
        cat = case.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{cid}: unknown category {cat!r}")
        turns = case.get("turns", [])
        if not isinstance(turns, list):
            errors.append(f"{cid}: 'turns' is not a list")
            turns = []
        if case.get("turn_count") != len(turns):
            errors.append(f"{cid}: turn_count mismatch")
        for idx, turn in enumerate(turns, start=1):
            prefix = f"{cid}/turn{idx}"
            if not isinstance(turn, dict):
                errors.append(f"{prefix}: turn is not a dict")
                continue
            for tf in _REQUIRED_TURN_FIELDS:
                if tf not in turn:
                    errors.append(f"{prefix}: missing turn field {tf!r}")
            if turn.get("turn_index") != idx:
                errors.append(f"{prefix}: turn_index mismatch")
            u_intent = turn.get("user_intent")
            if u_intent not in VALID_INTENTS:
                errors.append(f"{prefix}: invalid user_intent {u_intent!r}")
            intent = turn.get("expected_intent")
            if intent not in VALID_INTENTS:
                errors.append(f"{prefix}: invalid expected_intent {intent!r}")
            tools = turn.get("expected_tools", [])
            if not isinstance(tools, list):
                errors.append(f"{prefix}: expected_tools not a list")
                tools = []
            for t in tools:
                if t not in VALID_TOOLS:
                    errors.append(f"{prefix}: invalid tool {t!r}")
            rec_model_inputs = intent in ("predict_fuel", "explain_prediction",
                                          "combined_request", "report_behavior")
            if rec_model_inputs and not tools:
                errors.append(f"{prefix}: model-input intent {intent!r} has no tools")
            if intent in ("unsupported", "ambiguous", "clarifying") and tools:
                errors.append(f"{prefix}: {intent!r} must not select tools")
            clar = turn.get("expected_clarification")
            abst = turn.get("expected_abstention")
            if not isinstance(clar, bool) or not isinstance(abst, bool):
                errors.append(f"{prefix}: clarification/abstention must be bool")
            if clar and abst:
                errors.append(f"{prefix}: clarification and abstention both True")
            if intent == "clarifying" and not clar:
                errors.append(f"{prefix}: intent='clarifying' but expected_clarification=False")
            if intent == "unsupported" and not abst and not clar:
                errors.append(f"{prefix}: unsupported must abstain or clarify")
            if intent == "ambiguous" and not clar:
                errors.append(f"{prefix}: ambiguous must clarify")
            req = turn.get("expected_required_inputs")
            if req is not None and not isinstance(req, list):
                errors.append(f"{prefix}: expected_required_inputs not a list|null")
            for g in turn.get("expected_grounding", []):
                if g not in VALID_GROUNDING:
                    errors.append(f"{prefix}: invalid grounding {g!r}")
            saf = turn.get("expected_safety_outcome")
            if saf not in VALID_SAFETY:
                errors.append(f"{prefix}: invalid safety outcome {saf!r}")
            text = turn.get("text", "")
            norm = _normalize(text)
            if norm:
                # intra-scenario repeats are legitimate (same message retried)
                if norm not in seen_text_in_case.setdefault(cid, set()):
                    if norm in seen_text:
                        errors.append(
                            f"cross-case near-duplicate text: {cid} vs "
                            f"{seen_text[norm]} -> {text!r}"
                        )
                    seen_text[norm] = cid
                    seen_text_in_case[cid].add(norm)
                    if split == "dev":
                        dev_text[norm] = cid
                    elif split == "heldout":
                        held_text[norm] = cid
    for norm, hcid in held_text.items():
        if norm in dev_text:
            errors.append(f"split leakage: held-out {hcid} duplicates dev {dev_text[norm]}")
    dev_count = sum(1 for c in cases if c.get("split") == "dev")
    held_count = sum(1 for c in cases if c.get("split") == "heldout")
    if doc.get("split_counts", {}).get("dev") != dev_count:
        errors.append("split_counts.dev mismatch")
    if doc.get("split_counts", {}).get("heldout") != held_count:
        errors.append("split_counts.heldout mismatch")
    if doc.get("category_counts") is not None:
        doc_cat = doc["category_counts"]
        real_cat: dict[str, int] = {}
        for c in cases:
            real_cat[c.get("category", "")] = real_cat.get(c.get("category", ""), 0) + 1
        if doc_cat != real_cat:
            errors.append("category_counts mismatch")
    return errors


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation for duplicate detection."""
    import re
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

