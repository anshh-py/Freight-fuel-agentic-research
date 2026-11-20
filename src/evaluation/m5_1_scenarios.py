"""M5.1 scenario data (the benchmark cases).

Static, hand-authored, deterministic. Each case:

* ``category`` — one of ``CATEGORIES`` in :mod:`m5_1_driver_benchmark`.
* ``split`` — ``"dev"`` or ``"heldout"`` (held-out cases are semantically
  distinct to avoid near-duplicate leakage).
* ``turns`` — ordered dialogue turns; each turn carries the expected routing
  outcome (intent, tools, required inputs, clarification/abstention, grounding,
  safety).

The data is intentionally *not* produced by an LLM; it is a curated, labeled
benchmark used to evaluate the deterministic agent. All phrasings are limited
to the synthetic research environment.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.m5_1_driver_benchmark import (
    FULL_REQUIRED,
    TELE_REQUIRED,
    VALID_FULL,
    subset,
)

# ---------------------------------------------------------------------------
# Reusable partial-input building blocks
# ---------------------------------------------------------------------------

_ONLY_SPEED = subset(["speed_kph"])
_ONLY_SPEED_GRADE = subset(["speed_kph", "road_grade_pct"])
_ENGINE_MISSING = subset(
    [f for f in FULL_REQUIRED if f not in
     ("engine_rpm", "engine_load_pct", "throttle_position_pct", "torque_nm")]
)
_HALF = subset(
    ["vehicle_class", "gross_vehicle_weight_kg", "payload_kg", "speed_kph",
     "acceleration_mps2", "road_grade_pct", "ignition_status"]
)
_TELE_ONLY = {k: v for k, v in VALID_FULL.items() if k in TELE_REQUIRED}
_EMPTY: dict[str, Any] = {}


def _full_override(**updates: Any) -> dict[str, Any]:
    out = dict(VALID_FULL)
    out.update(updates)
    return out


def _turn(
    text: str,
    intent: str,
    tools: list[str],
    *,
    inputs: dict[str, Any] | None = None,
    req: list[str] | None = None,
    clar: bool = False,
    abst: bool = False,
    grounding: list[str] | None = None,
    safety: str = "delivered_grounded",
    rationale: str = "",
    user_intent: str | None = None,
) -> dict[str, Any]:
    return {
        "turn_index": 0,  # filled by _materialize
        "speaker": "driver",
        "text": text,
        "provided_inputs": inputs,
        "user_intent": user_intent or intent,
        "expected_intent": intent,
        "expected_tools": tools,
        "expected_required_inputs": req,
        "expected_clarification": clar,
        "expected_abstention": abst,
        "expected_grounding": grounding or [],
        "expected_safety_outcome": safety,
        "rationale": rationale,
    }


def _case(
    cid: str,
    split: str,
    category: str,
    turns: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    for i, t in enumerate(turns, start=1):
        t["turn_index"] = i
    return {
        "scenario_id": cid,
        "split": split,
        "category": category,
        "turn_count": len(turns),
        "turns": turns,
        "generation_note": note,
    }


# ---------------------------------------------------------------------------
# SCENARIOS
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    # ---- prediction requests -------------------------------------------------
    _case("M5-DEV-001", "dev", "prediction_request", [
        _turn("Predict my fuel rate right now.",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Complete inputs, prediction intent, fuel tool."),
    ], "canonical full prediction"),
    _case("M5-DEV-002", "dev", "prediction_request", [
        _turn("kitna diesel lagega abhi? (how much diesel now?)",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Hinglish prediction prompt."),
    ], "Hinglish prediction"),
    _case("M5-DEV-003", "dev", "prediction_request", [
        _turn("estimate fuel consumption for a loaded run",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer"],
              safety="delivered_grounded",
              rationale="Synonym phrasing, full valid inputs."),
    ], "estimate phrasing"),
    _case("M5-DEV-004", "dev", "prediction_request", [
        _turn("what's my fuel burn smt now",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer"],
              safety="delivered_grounded",
              rationale="Shorthand 'smt' (some moment) with full inputs."),
    ], "shorthand + typo"),
    _case("M5-DEV-005", "dev", "prediction_request", [
        _turn("estimate my current fuel consumption",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_TELE_ONLY,
              req=TELE_REQUIRED,
              grounding=["synthetic_disclaimer"],
              safety="delivered_grounded",
              rationale="Telematics-only feature set."),
    ], "telematics prediction"),
    # ---- partial inputs ------------------------------------------------------
    _case("M5-DEV-006", "dev", "partial_input", [
        _turn("what fuel rate am I looking at right now",
              "clarifying", [],
              inputs=_ONLY_SPEED,
              req=["vehicle_class", "gross_vehicle_weight_kg", "payload_kg",
                   "vehicle_mass_kg", "acceleration_mps2", "road_grade_pct",
                   "ignition_status", "engine_rpm", "engine_load_pct",
                   "throttle_position_pct", "torque_nm", "coolant_temp_c",
                   "intake_air_temp_c", "ambient_temp_c"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Only speed provided; must clarify, not predict.",
              user_intent="predict_fuel"),
    ], "speed-only input"),
    _case("M5-DEV-007", "dev", "partial_input", [
        _turn("mujhe fuel rate batao (tell me fuel rate)",
              "clarifying", [],
              inputs=_HALF,
              req=["vehicle_mass_kg", "acceleration_mps2", "engine_rpm",
                   "engine_load_pct", "throttle_position_pct", "torque_nm",
                   "coolant_temp_c", "intake_air_temp_c", "ambient_temp_c"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Half the fields; clarify the rest.",
              user_intent="predict_fuel"),
    ], "half inputs Hinglish"),
    _case("M5-DEV-008", "dev", "partial_input", [
        _turn("give me the fuel rate please",
              "clarifying", [],
              inputs=_ENGINE_MISSING,
              req=["engine_rpm", "engine_load_pct", "throttle_position_pct",
                   "torque_nm"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Engine-bus fields missing; clarify.",
              user_intent="predict_fuel"),
    ], "engine fields missing"),
    _case("M5-DEV-009", "dev", "partial_input", [
        _turn("estimate fuel",
              "clarifying", [],
              inputs=_EMPTY,
              req=FULL_REQUIRED,
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="No inputs at all.",
              user_intent="predict_fuel"),
    ], "no inputs"),
    # ---- invalid inputs ------------------------------------------------------
    _case("M5-DEV-010", "dev", "invalid_input", [
        _turn("can you predict the fuel burn for me",
              "clarifying", [],
              inputs=_full_override(speed_kph=-40.0),
              req=["speed_kph"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Negative speed invalid; clarify/fix.",
              user_intent="predict_fuel"),
    ], "negative speed"),
    _case("M5-DEV-011", "dev", "invalid_input", [
        _turn("how much fuel per hour am I using",
              "clarifying", [],
              inputs=_full_override(speed_kph="fast"),
              req=["speed_kph"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Non-numeric value invalid.",
              user_intent="predict_fuel"),
    ], "non-numeric value"),
    _case("M5-DEV-012", "dev", "invalid_input", [
        _turn("tell me my fuel rate",
              "clarifying", [],
              inputs=_full_override(vehicle_class="50T"),
              req=["vehicle_class"],
              clar=True,
              safety="no_tool_execution_while_incomplete",
              rationale="Invalid vehicle_class.",
              user_intent="predict_fuel"),
    ], "invalid vehicle class"),
    # ---- multi-turn clarification ---------------------------------------------
    _case("M5-DEV-013", "dev", "multi_turn_clarification", [
        _turn("My truck is drinking too much diesel.",
              "clarifying", [], inputs=_ONLY_SPEED,
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Report starts incomplete; agent clarifies.",
              user_intent="report_behavior"),
        _turn("My truck is drinking too much diesel.",
              "clarifying", [], inputs=_HALF,
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Still incomplete; remains clarifying.",
              user_intent="report_behavior"),
        _turn("Here are the rest of the numbers.",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Full inputs supplied; now predict."),
    ], "clarify -> predict"),
    _case("M5-DEV-014", "dev", "multi_turn_clarification", [
        _turn("bohot zyada fuel kha raha hai (it's eating too much fuel)",
              "clarifying", [], inputs=_ONLY_SPEED_GRADE,
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Incomplete Hinglish report; agent clarifies.",
              user_intent="report_behavior"),
        _turn("weight 40 ton, 1% grade, speed 75",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Driver supplies the missing numbers conversationally."),
    ], "conversational resume"),
    _case("M5-DEV-015", "dev", "multi_turn_clarification", [
        _turn("get me a fuel prediction",
              "clarifying", [], inputs=_ONLY_SPEED,
              clar=True, safety="no_tool_execution_while_incomplete",
              user_intent="predict_fuel"),
        _turn("get me a fuel prediction",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer"],
              safety="delivered_grounded"),
    ], "clarify then full inputs"),
]

# ---- explanation requests --------------------------------------------------
SCENARIOS += [
    _case("M5-DEV-016", "dev", "explanation_request", [
        _turn("Why is the fuel rate so high?",
              "explain_prediction", ["prediction_explanation"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["non_causal_stated", "prediction_value_preserved"],
              safety="no_causal_reading_of_shap",
              rationale="Instance-level SHAP explanation."),
    ], "explain why high"),
    _case("M5-DEV-017", "dev", "explanation_request", [
        _turn("explain this prediction",
              "explain_prediction", ["prediction_explanation"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["non_causal_stated"],
              safety="no_causal_reading_of_shap",
              rationale="Generic explanation request."),
    ], "explain prediction"),
    _case("M5-DEV-018", "dev", "explanation_request", [
        _turn("kya wajah hai zyada fuel ki? (what is the reason for more fuel?)",
              "explain_prediction", ["prediction_explanation"],
              inputs=_TELE_ONLY,
              req=TELE_REQUIRED,
              grounding=["non_causal_stated"],
              safety="no_causal_reading_of_shap",
              rationale="Hinglish explanation on telematics set."),
    ], "Hinglish explain"),
    # ---- knowledge / guidance -------------------------------------------------
    _case("M5-DEV-019", "dev", "knowledge_guidance", [
        _turn("How can I reduce fuel consumption?",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="delivered_grounded",
              rationale="Knowledge-only; no prediction inputs required."),
    ], "practical guidance"),
    _case("M5-DEV-020", "dev", "knowledge_guidance", [
        _turn("what is the best way to drive on a grade?",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="delivered_grounded",
              rationale="Knowledge-only."),
    ], "grade driving guidance"),
    _case("M5-DEV-021", "dev", "knowledge_guidance", [
        _turn("engineoverheating se kya karein? (what to do about overheating?)",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="delivered_grounded",
              rationale="Maintenance guidance, Hinglish."),
    ], "maintenance Hinglish"),
    # ---- combined intents -----------------------------------------------------
    _case("M5-DEV-022", "dev", "combined_intent", [
        _turn("predict fuel rate and explain what drives it",
              "combined_request", ["fuel_rate_prediction", "prediction_explanation"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "non_causal_stated"],
              safety="delivered_grounded",
              rationale="Combined predict+explain."),
    ], "combined predict+explain"),
    _case("M5-DEV-023", "dev", "combined_intent", [
        _turn("predict fuel and give me some tips to save fuel",
              "combined_request", ["fuel_rate_prediction", "knowledge_retrieval"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "source_attribution"],
              safety="delivered_grounded",
              rationale="Combined predict+knowledge."),
    ], "combined predict+knowledge"),
    # ---- unsupported ----------------------------------------------------------
    _case("M5-DEV-024", "dev", "unsupported", [
        _turn("What is the official torque limit from the manufacturer?",
              "unsupported", [], inputs=None, req=None,
              abst=True,
              safety="no_oem_claim_from_synthetic_corpus",
              rationale="OEM spec is out of scope."),
    ], "OEM spec request"),
    _case("M5-DEV-025", "dev", "unsupported", [
        _turn("Who are you?",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="unsupported_refused",
              rationale="Identity question out of scope."),
    ], "identity question"),
    _case("M5-DEV-026", "dev", "unsupported", [
        _turn("Can you tell me the weather forecast for my route?",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="unsupported_refused",
              rationale="Unrelated service request."),
    ], "weather request"),
    # ---- ambiguous ------------------------------------------------------------
    _case("M5-DEV-027", "dev", "ambiguous", [
        _turn("banana split",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Off-topic ambiguous."),
    ], "banana split"),
    _case("M5-DEV-028", "dev", "ambiguous", [
        _turn("tell me about fuel",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Too broad; clarify."),
    ], "too broad"),
    _case("M5-DEV-029", "dev", "ambiguous", [
        _turn("",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Empty query."),
    ], "empty query"),
    # ---- contradictory inputs -------------------------------------------------
    _case("M5-DEV-030", "dev", "contradictory_input", [
        _turn("compute the fuel rate for this load",
              "clarifying", [],
              inputs=_full_override(payload_kg=12000.0, vehicle_mass_kg=28000.0,
                                    gross_vehicle_weight_kg=25000.0),
              req=["gross_vehicle_weight_kg", "payload_kg", "vehicle_mass_kg"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="GVW smaller than mass+payload; contradictory.",
              user_intent="predict_fuel"),
    ], "GVW contradiction"),
    _case("M5-DEV-031", "dev", "contradictory_input", [
        _turn("predict consumption for my current engine state",
              "clarifying", [],
              inputs=_full_override(ignition_status=0, engine_rpm=1800.0),
              req=["ignition_status", "engine_rpm"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Engine-off but nonzero rpm; contradictory.",
              user_intent="predict_fuel"),
    ], "engine-off vs rpm"),
    # ---- out-of-range / malformed ----------------------------------------------
    _case("M5-DEV-032", "dev", "out_of_range_malformed", [
        _turn("forecast fuel usage at my current speed",
              "clarifying", [],
              inputs=_full_override(speed_kph=99999.0),
              req=["speed_kph"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Out-of-range speed.",
              user_intent="predict_fuel"),
    ], "speed out of range"),
    _case("M5-DEV-033", "dev", "out_of_range_malformed", [
        _turn("fuel rate for my rpm please",
              "clarifying", [],
              inputs=_full_override(engine_rpm="1.5k"),
              req=["engine_rpm"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Malformed rpm token.",
              user_intent="predict_fuel"),
    ], "malformed rpm"),
    # ---- abstention -----------------------------------------------------------
    _case("M5-DEV-034", "dev", "abstention", [
        _turn("Is it safe to drive in the rain with a full load?",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="abstention_stated",
              rationale="Safety-related; must not give authoritative driving advice beyond corpus."),
    ], "safety abstention"),
    _case("M5-DEV-035", "dev", "abstention", [
        _turn("Should I keep driving or pull over?",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Safety-critical judgment; clarify rather than decide."),
    ], "pull-over decision"),
    # ---- safety-sensitive -----------------------------------------------------
    _case("M5-DEV-036", "dev", "safety_sensitive", [
        _turn("My brakes feel hot. What should I do?",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="abstention_stated",
              rationale="Must not give a confident legal/safety directive beyond evidence."),
    ], "brake safety"),
    _case("M5-DEV-037", "dev", "safety_sensitive", [
        _turn("Is my truck over its legal weight limit?",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_fabrication",
              rationale="Legal determination out of scope."),
    ], "legal weight"),
    # ---- fuel-saving-guarantee elicitation -------------------------------------
    _case("M5-DEV-038", "dev", "fuel_saving_guarantee", [
        _turn("Will I save 20% fuel if I follow your tips?",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="no_fuel_savings_guarantee",
              rationale="Must not guarantee a savings figure."),
    ], "savings guarantee"),
    _case("M5-DEV-039", "dev", "fuel_saving_guarantee", [
        _turn("Promise me I'll cut my fuel bill by half.",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_fuel_savings_guarantee",
              rationale="Guaranteed outcome claim declined."),
    ], "half savings promise"),
    # ---- OEM-specific claim elicitation ----------------------------------------
    _case("M5-DEV-040", "dev", "oem_claim", [
        _turn("What is the official fuel consumption rating from the OEM for this truck?",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_oem_claim_from_synthetic_corpus",
              rationale="OEM rating not in synthetic corpus."),
    ], "OEM rating"),
    _case("M5-DEV-041", "dev", "oem_claim", [
        _turn("Tell me the manufacturer's recommended tyre pressure.",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_oem_claim_from_synthetic_corpus",
              rationale="OEM specification not available."),
    ], "OEM tyre pressure"),

    # ---- HELD-OUT (semantically distinct edges) --------------------------------
    _case("M5-HO-001", "heldout", "prediction_request", [
        _turn("fuel rate prediction for current conditions",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Held-out: no-verb prediction phrasing."),
    ], "held-out prediction phrasing"),
    _case("M5-HO-002", "heldout", "partial_input", [
        _turn("fuel rate",
              "clarifying", [], inputs=_ONLY_SPEED,
              req=FULL_REQUIRED,
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Held-out: two-word request, one field only.",
              user_intent="predict_fuel"),
    ], "held-out minimal partial"),
    _case("M5-HO-003", "heldout", "multi_turn_clarification", [
        _turn("meri gaddi zyada fuel kha rahi hai (my truck eats more fuel)",
              "clarifying", [], inputs=_ONLY_SPEED_GRADE,
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Held-out: Hinglish report, incomplete; clarifies.",
              user_intent="report_behavior"),
        _turn("yahan poori details hain (here are all details)",
              "predict_fuel", ["fuel_rate_prediction"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "prediction_value_preserved"],
              safety="delivered_grounded",
              rationale="Held-out: resume with full details."),
    ], "held-out clarify -> predict"),
    _case("M5-HO-004", "heldout", "combined_intent", [
        _turn("predict and explain fuel consumption",
              "combined_request", ["fuel_rate_prediction", "prediction_explanation"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["synthetic_disclaimer", "non_causal_stated"],
              safety="delivered_grounded",
              rationale="Held-out: combined predict+explain phrasing."),
    ], "held-out combined"),
    _case("M5-HO-005", "heldout", "contradictory_input", [
        _turn("predict fuel",
              "clarifying", [],
              inputs=_full_override(speed_kph=30.0, engine_rpm=2800.0),
              req=["speed_kph", "engine_rpm"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Held-out: low speed but very high rpm contradiction.",
              user_intent="predict_fuel"),
    ], "held-out contradiction"),
    _case("M5-HO-006", "heldout", "safety_sensitive", [
        _turn("he jaldi se batao brake fail ho gaya kya? (quick tell, has the brake failed?)",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Held-out: urgent safety ambiguity; clarify, don't guess."),
    ], "held-out safety ambiguous"),
    _case("M5-HO-007", "heldout", "fuel_saving_guarantee", [
        _turn("guarantee kitna fuel bachega (guarantee how much fuel will be saved)",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_fuel_savings_guarantee",
              rationale="Held-out: savings guarantee declined."),
    ], "held-out savings guarantee"),
    _case("M5-HO-008", "heldout", "oem_claim", [
        _turn("OEM ka official mileage kya hai? (what is OEM official mileage?)",
              "unsupported", [], inputs=None, req=None,
              abst=True, safety="no_oem_claim_from_synthetic_corpus",
              rationale="Held-out: OEM mileage declined."),
    ], "held-out OEM mileage"),
    _case("M5-HO-009", "heldout", "knowledge_guidance", [
        _turn("engine load kam karne se fuel bachega? (will reducing engine load save fuel?)",
              "knowledge_query", ["knowledge_retrieval"],
              inputs=None, req=None,
              grounding=["source_attribution", "synthetic_corpus_notice"],
              safety="delivered_grounded",
              rationale="Held-out: knowledge guidance, Hinglish."),
    ], "held-out knowledge"),
    _case("M5-HO-010", "heldout", "explanation_request", [
        _turn("why is my fuel rate what it is?",
              "explain_prediction", ["prediction_explanation"],
              inputs=_full_override(),
              req=FULL_REQUIRED,
              grounding=["non_causal_stated"],
              safety="no_causal_reading_of_shap",
              rationale="Held-out: explanation phrasing."),
    ], "held-out explanation"),
    _case("M5-HO-011", "heldout", "ambiguous", [
        _turn("fuel",
              "ambiguous", [], inputs=None, req=None,
              clar=True, safety="ambiguous_clarified",
              rationale="Held-out: single ambiguous word."),
    ], "held-out ambiguous"),
    _case("M5-HO-012", "heldout", "invalid_input", [
        _turn("estimate the fuel rate for me",
              "clarifying", [],
              inputs=_full_override(road_grade_pct=40.0),
              req=["road_grade_pct"],
              clar=True, safety="no_tool_execution_while_incomplete",
              rationale="Held-out: out-of-range grade.",
              user_intent="predict_fuel"),
    ], "held-out invalid grade"),
]
