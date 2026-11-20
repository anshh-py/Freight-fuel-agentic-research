"""M4.3 — focused tests for the driver intent router and clarification flow.

Covers driver-intent classification, clarification when required inputs are
missing, resuming a pending request, knowledge-only requests that do not
require prediction inputs, unsupported/ambiguous handling, the guarantee that
no tool is executed while clarification is required, and compatibility with
the existing M2 routing interface shape.
"""

from __future__ import annotations

from src.agent.driver_intents import (
    DriverIntent,
    DriverIntentResult,
    DriverIntentRouter,
)
from src.agent.driver_session import DriverSession
from src.agent.intents import TOOL_EXPLAIN, TOOL_PREDICT, TOOL_RETRIEVAL


def _valid_full_inputs() -> dict:
    return {
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


def _router() -> DriverIntentRouter:
    return DriverIntentRouter()


# ---------------------------------------------------------------------------
# Driver-intent classification
# ---------------------------------------------------------------------------


def test_report_behavior_with_complete_inputs_resolves_to_predict():
    result = _router().route(
        "My truck is consuming too much diesel.",
        structured_inputs=_valid_full_inputs(),
    )
    assert isinstance(result, DriverIntentResult)
    assert result.intent == DriverIntent.PREDICT_FUEL
    assert result.is_ready
    assert result.selected_tools == [TOOL_PREDICT]


def test_report_behavior_without_inputs_requires_clarification():
    result = _router().route("I'm driving uphill with a heavy load.")
    assert result.intent == DriverIntent.CLARIFYING
    assert not result.is_ready
    assert result.selected_tools == []
    assert "payload_kg" in result.missing_fields


def test_explain_prediction_with_complete_inputs():
    result = _router().route(
        "Explain the prediction.",
        structured_inputs=_valid_full_inputs(),
    )
    assert result.intent == DriverIntent.EXPLAIN_PREDICTION
    assert result.selected_tools == [TOOL_EXPLAIN]
    assert result.is_ready


def test_explain_requiring_inputs_clarifies_when_missing():
    result = _router().route("Why is fuel consumption increasing?", {})
    assert result.intent == DriverIntent.CLARIFYING
    assert result.selected_tools == []


def test_knowledge_query_does_not_require_prediction_inputs():
    result = _router().route(
        "What should I do to reduce unnecessary fuel consumption?"
    )
    assert result.intent == DriverIntent.KNOWLEDGE_QUERY
    assert result.selected_tools == [TOOL_RETRIEVAL]
    assert result.is_ready
    assert result.missing_fields == []


def test_unsupported_intent_preserved():
    result = _router().route("real-world validation of my truck")
    assert result.intent == DriverIntent.UNSUPPORTED
    assert result.selected_tools == []


def test_ambiguous_empty_query():
    result = _router().route("")
    assert result.intent == DriverIntent.AMBIGUOUS
    assert result.selected_tools == []


# ---------------------------------------------------------------------------
# Clarification flow
# ---------------------------------------------------------------------------


def test_clarification_returns_missing_fields_and_no_tools():
    result = _router().route(
        "how much fuel will I use",
        {"speed_kph": 75.0},
    )
    assert result.intent == DriverIntent.CLARIFYING
    assert result.selected_tools == []
    assert "payload_kg" in result.missing_fields
    assert "speed_kph" not in result.missing_fields  # already supplied


def test_clarification_records_invalid_fields():
    result = _router().route(
        "predict fuel rate",
        {"speed_kph": 250.0},
    )
    assert result.intent == DriverIntent.CLARIFYING
    assert result.selected_tools == []
    assert "speed_kph" in result.invalid_fields


def test_resume_pending_when_inputs_complete():
    session = DriverSession()
    session.apply_partial_inputs({"vehicle_class": "40T", "speed_kph": 75.0})
    session.record_request("how much fuel on this grade?", intent="predict_fuel")

    # Still missing fields -> stays clarifying.
    pre = _router().resume_pending(session)
    assert pre.intent == DriverIntent.CLARIFYING

    # Supply the rest of the required fields via the session context.
    rest = {k: v for k, v in _valid_full_inputs().items() if k not in session.truck_context}
    session.apply_partial_inputs(rest)

    post = _router().resume_pending(session)
    assert post.intent == DriverIntent.PREDICT_FUEL
    assert post.selected_tools == [TOOL_PREDICT]
    assert post.is_ready
    assert post.resumes_pending


def test_no_tool_execution_while_clarifying():
    result = _router().route("predict", {"speed_kph": 75.0})
    assert result.intent == DriverIntent.CLARIFYING
    assert result.selected_tools == []


# ---------------------------------------------------------------------------
# Qualitative observation safety
# ---------------------------------------------------------------------------


def test_qualitative_report_never_invents_telemetry():
    # "heavy load" is qualitative and must not become a numeric payload.
    result = _router().route("I'm driving uphill with a heavy load.")
    assert result.intent == DriverIntent.CLARIFYING
    assert result.selected_tools == []
    assert "payload_kg" in result.missing_fields
    assert "payload_kg" not in getattr(result, "validated_inputs", {})


# ---------------------------------------------------------------------------
# Compatibility with existing routing interfaces
# ---------------------------------------------------------------------------


def test_result_is_shape_compatible_with_intent_result():
    result = _router().route("predict", _valid_full_inputs())
    d = result.to_dict()
    assert d["driver_intent"] == "predict_fuel"
    assert "confidence" in d
    assert "reason" in d
    assert "backend" in d
    assert "selected_tools" in d
    # Mirrors IntentResult's scalar attributes.
    assert result.intent is not None
    assert isinstance(result.confidence, float)
    assert isinstance(result.reason, str)
    assert result.backend == "deterministic_driver"


def test_router_exposes_route_with_intent_router_signature():
    r = _router()
    # Must be callable with (query, structured_inputs) like the M2 protocol.
    result = r.route("predict fuel rate", _valid_full_inputs())
    assert isinstance(result, DriverIntentResult)
