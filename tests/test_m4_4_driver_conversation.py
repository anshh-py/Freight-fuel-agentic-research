"""M4.4 — focused tests for the grounded conversational driver feedback service.

Covers multi-turn clarification -> input -> prediction, prediction -> explanation
follow-up, knowledge-only interactions, unsupported requests, missing-input
safety (no tool execution), response-validation failure blocking, and session
context / history preservation.

Tool stubs mimic the result *shapes* returned by the real M2 tools so the real
``ResponseGenerator`` and ``validate_grounded_response`` run without loading any
model artifacts.
"""

from __future__ import annotations

from src.agent.driver_conversation import (
    DriverConversationResponse,
    DriverConversationService,
)
from src.agent.driver_session import ConversationState, DriverSession
from src.agent.response_generator import GroundedResponse

# Required fields for the full_operational feature set.
FULL_INPUTS = {
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


# ---------------------------------------------------------------------------
# Recording tool stubs
# ---------------------------------------------------------------------------
class RecordingPredictStub:
    def __init__(self):
        self.calls = 0

    def predict(self, inputs, feature_set=None):
        self.calls += 1
        return {
            "status": "ok",
            "predicted_fuel_rate_lph": 12.3456,
            "units": "L/h",
            "feature_set": feature_set or "full_operational",
            "model": {"name": "xgboost", "artifact": "stub"},
            "preprocessor": {"artifact": "stub", "transformed_columns": []},
            "input_fields_used": dict(inputs),
            "provenance": {"synthetic_training": True, "model_is_estimate": True},
        }


class RecordingExplainStub:
    def __init__(self):
        self.calls = 0

    def explain(self, inputs, feature_set=None, top_features=None):
        self.calls += 1
        return {
            "status": "ok",
            "predicted_fuel_rate_lph": 12.3456,
            "feature_set": feature_set or "full_operational",
            "model": {"name": "xgboost", "artifact": "stub"},
            "explanation_kind": "instance_level_shap",
            "summary_statistic": "per-instance SHAP",
            "top_features": top_features or 5,
            "expected_value": 10.0,
            "reconstruction": {"pass": True, "max_abs_deviation": 0.0, "tolerance": 0.01},
            "most_influential_features": {
                "top_positive_shap": [
                    {"readable_label": "Road grade", "column": "road_grade_pct", "shap": 1.25},
                ],
                "top_negative_shap": [
                    {"readable_label": "Speed", "column": "speed_kph", "shap": -0.5},
                ],
            },
            "limitations": {"shap_is_not_causal": True},
            "provenance": {"synthetic_training": True},
        }


class RecordingRetrieveStub:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, k=None, min_score=None):
        self.calls += 1
        return {
            "status": "ok",
            "query": query,
            "n_results": 1,
            "passages": [
                {
                    "rank": 1,
                    "chunk_id": "c1",
                    "document_id": "doc_001",
                    "title": "Driving tips",
                    "topic": "driving",
                    "source_type": "synthetic",
                    "source_reference": "synthetic-corpus",
                    "is_oem_manual": False,
                    "score": 0.8,
                    "content_excerpt": "Shift gears smoothly and avoid hard acceleration.",
                },
            ],
            "evidence_sufficiency": {
                "insufficient": False,
                "reason": "top passage meets similarity floor",
                "min_score_floor": 0.45,
            },
            "provenance": {"corpus_synthetic": True},
        }


def _service():
    predict = RecordingPredictStub()
    explain = RecordingExplainStub()
    retrieve = RecordingRetrieveStub()
    svc = DriverConversationService(
        tools={
            "fuel_rate_prediction": predict,
            "prediction_explanation": explain,
            "knowledge_retrieval": retrieve,
        }
    )
    return svc, predict, explain, retrieve


# ---------------------------------------------------------------------------
# Multi-turn: clarification -> input -> prediction
# ---------------------------------------------------------------------------
def test_multi_turn_clarification_then_prediction():
    svc, predict, _, _ = _service()
    session = DriverSession()

    # Turn 1: report without enough inputs -> clarification, no tools.
    r1 = svc.handle(session, "My truck is consuming too much diesel.")
    assert isinstance(r1, DriverConversationResponse)
    assert r1.driver_intent == "clarifying"
    assert r1.clarification_request is not None
    assert r1.executed_tools == []
    assert r1.delivered is False
    assert predict.calls == 0
    assert "payload_kg" in r1.missing_fields

    # Turn 2: driver supplies the missing inputs -> prediction runs.
    r2 = svc.handle(session, "My truck is consuming too much diesel.", FULL_INPUTS)
    assert r2.driver_intent == "predict_fuel"
    assert r2.executed_tools == ["fuel_rate_prediction"]
    assert predict.calls == 1
    assert r2.delivered is True
    assert r2.grounded is not None
    assert r2.validation_summary["all_passed"] is True
    assert "12.3456" in r2.driver_feedback


# ---------------------------------------------------------------------------
# Prediction -> explanation follow-up
# ---------------------------------------------------------------------------
def test_prediction_then_explanation_followup():
    svc, predict, explain, _ = _service()
    session = DriverSession()

    r1 = svc.handle(session, "predict fuel rate", FULL_INPUTS)
    assert r1.driver_intent == "predict_fuel"
    assert predict.calls == 1 and explain.calls == 0

    r2 = svc.handle(session, "Explain the prediction.")
    assert r2.driver_intent == "explain_prediction"
    assert r2.executed_tools == ["prediction_explanation"]
    assert explain.calls == 1
    assert r2.grounded is not None and r2.grounded.explanation is not None


# ---------------------------------------------------------------------------
# Knowledge-only interaction
# ---------------------------------------------------------------------------
def test_knowledge_only_executes_only_retrieval():
    svc, predict, explain, retrieve = _service()
    session = DriverSession()

    r = svc.handle(session, "What should I do to reduce unnecessary fuel consumption?")
    assert r.driver_intent == "knowledge_query"
    assert r.executed_tools == ["knowledge_retrieval"]
    assert retrieve.calls == 1
    assert predict.calls == 0 and explain.calls == 0
    assert r.delivered is True
    assert r.grounded is not None and r.grounded.evidence


# ---------------------------------------------------------------------------
# Unsupported requests
# ---------------------------------------------------------------------------
def test_unsupported_request_remains_unsupported():
    svc, predict, explain, retrieve = _service()
    session = DriverSession()

    r = svc.handle(session, "real-world validation of my truck")
    assert r.driver_intent == "unsupported"
    assert r.executed_tools == []
    assert predict.calls == 0 and explain.calls == 0 and retrieve.calls == 0
    assert session.conversation_state == ConversationState.UNSUPPORTED
    assert "can't answer" in r.driver_feedback.lower()


# ---------------------------------------------------------------------------
# Missing-input safety
# ---------------------------------------------------------------------------
def test_missing_input_safety_no_prediction_tools():
    svc, predict, explain, explore = _service()
    session = DriverSession()

    r = svc.handle(session, "how much fuel will I use", {"speed_kph": 75.0})
    assert r.driver_intent == "clarifying"
    assert r.executed_tools == []
    assert predict.calls == 0 and explain.calls == 0
    assert "payload_kg" in r.missing_fields


# ---------------------------------------------------------------------------
# Response validation failure handling
# ---------------------------------------------------------------------------
class _FakeBadGenerator:
    """A generator that emits a response missing the required synthetic disclaimer."""

    def generate(self, response):
        return GroundedResponse(
            intent="fuel_prediction",
            detected_intent="fuel_prediction",
            router_backend="deterministic",
            selected_tools=["fuel_rate_prediction"],
            prediction={"predicted_fuel_rate_lph": 12.3, "feature_set": "full_operational"},
            driver_version="Predicted fuel rate: 12.3 L/h.",
            technical_version={"note": "missing synthetic disclaimer"},
        )


def test_validation_failure_blocks_delivery():
    svc, predict, _, _ = _service()
    svc.generator = _FakeBadGenerator()
    session = DriverSession()

    r = svc.handle(session, "predict fuel rate", FULL_INPUTS)
    assert predict.calls == 1  # tool ran, but the assembled response fails validation
    assert r.validation_blocked is True
    assert r.delivered is False
    failed_ids = {i.check_id for i in r.validation_issues if not i.passed}
    assert "synthetic_disclaimer" in failed_ids
    assert "not being presented" in r.driver_feedback.lower()


# ---------------------------------------------------------------------------
# Session context / history preservation
# ---------------------------------------------------------------------------
def test_session_context_and_history_preserved():
    svc, _, _, _ = _service()
    session = DriverSession()

    # Turn with a qualitative observation and a numeric field.
    svc.handle(
        session,
        "I'm driving uphill with a heavy load.",
        {"speed_kph": 75.0, "driver_observation": "heavy trailer today"},
    )
    assert session.qualitative_observations.get("driver_observation") == "heavy trailer today"
    assert session.truck_context.get("speed_kph") == 75.0
    assert "driver_observation" not in session.truck_context
    assert session.conversation_state == ConversationState.CLARIFYING

    # Complete all inputs in a second turn, then assert history is preserved.
    r = svc.handle(session, "I'm driving uphill with a heavy load.", FULL_INPUTS)
    assert r.delivered is True
    assert len(session.history) >= 2
    assert session.last_request is not None
    assert session.last_tool_results  # stored after tool execution
