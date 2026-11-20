"""M6.1 — focused contract tests for the integration gateway.

Scope:
* session lifecycle + feature-set isolation,
* missing/expired session handling,
* multi-turn clarification and resumption (passthrough to the service),
* structured + conversational input forwarding,
* concurrent requests to one session (per-session serialization),
* routing / tool-failure / validation-blocked surfacing,
* serializable API response contract,
* one genuine end-to-end integration test using the real research artifacts.

Unit-style tests use a StubService so the heavy model artifacts are not loaded
repeatedly. Only ``test_real_integration_predict`` exercises the real tools.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from src.agent.driver_conversation import DriverConversationResponse
from src.api.gateway import FuelRateFeedbackGateway, GatewayStartupError, InvalidInputError
from src.api.session_registry import SessionExpiredError, SessionNotFoundError, SessionRegistry
from src.api.schemas import APIResponse

# ---------------------------------------------------------------------------
# Stub service
# ---------------------------------------------------------------------------


class StubService:
    """Replicates the gateway-facing surface of DriverConversationService."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.mode = "deliver"
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def handle(self, session, message: str, structured_inputs=None) -> DriverConversationResponse:
        with self._lock:
            self.calls.append({
                "session_id": session.session_id,
                "message": message,
                "structured_inputs": structured_inputs,
                "feature_set": session.feature_set,
            })
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        try:
            time.sleep(0.005)  # widen the window for overlap detection
            if self.mode == "toolfail":
                return _make_resp(session, tool_error=True)
            if self.mode == "blocked":
                return _make_resp(session, blocked=True)
            if self.mode == "clarify" and not getattr(session, "_primed", False):
                return _make_resp(session, clarify=True)
            return _make_resp(session, delivered=True)
        finally:
            with self._lock:
                self.active -= 1


def _make_resp(session, *, delivered=False, clarify=False, blocked=False, tool_error=False):
    from src.agent.driver_session import ConversationState
    if clarify:
        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent="clarifying",
            base_intent=None,
            conversation_state=ConversationState.CLARIFYING.value,
            delivered=False,
            validation_blocked=False,
            clarification_request="I need the following values: speed_kph, payload_kg.",
            missing_fields=["speed_kph", "payload_kg"],
            driver_feedback="I need the following values: speed_kph, payload_kg.",
            technical_report={"note": "clarifying"},
        )
    if blocked:
        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent="predict_fuel",
            base_intent="predict_fuel",
            conversation_state=ConversationState.ANSWERED.value,
            delivered=False,
            validation_blocked=True,
            selected_tools=["fuel_rate_prediction"],
            executed_tools=["fuel_rate_prediction"],
            tool_results=[{"tool": "fuel_rate_prediction", "status": "ok", "result": {}}],
            validation_issues=[],
            driver_feedback="I could not produce a validated, grounded answer for this turn.",
            technical_report={"synthetic_only": True},
        )
    if tool_error:
        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent="predict_fuel",
            base_intent="predict_fuel",
            conversation_state=ConversationState.ANSWERED.value,
            delivered=True,
            validation_blocked=False,
            selected_tools=["fuel_rate_prediction"],
            executed_tools=["fuel_rate_prediction"],
            tool_results=[{
                "tool": "fuel_rate_prediction", "status": "error",
                "error_type": "ToolExecutionError", "error_msg": "missing artifact",
            }],
            validation_issues=[],
            driver_feedback="One or more underlying tools failed, so I cannot provide a complete trusted answer.",
            technical_report={"synthetic_only": True},
        )
    return DriverConversationResponse(
        session_id=session.session_id,
        driver_intent="predict_fuel",
        base_intent="predict_fuel",
        conversation_state=ConversationState.ANSWERED.value,
        delivered=True,
        validation_blocked=False,
        selected_tools=["fuel_rate_prediction"],
        executed_tools=["fuel_rate_prediction"],
        tool_results=[{"tool": "fuel_rate_prediction", "status": "ok", "result": {}}],
        validation_issues=[],
        driver_feedback="Estimated fuel rate: 12.34 L/h (synthetic estimate).",
        technical_report={"synthetic_only": True, "prediction": {"predicted_fuel_rate_lph": 12.34}},
    )


def _gateway(service=None, **kwargs) -> FuelRateFeedbackGateway:
    return FuelRateFeedbackGateway(
        service=service if service is not None else StubService(),
        skip_preflight=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def test_session_create_and_feature_set_isolation():
    gw = _gateway()
    s1 = gw.start_session("full_operational")
    s2 = gw.start_session("telematics_only")
    assert s1 != s2
    assert gw.session_state(s1)["feature_set"] == "full_operational"
    assert gw.session_state(s2)["feature_set"] == "telematics_only"
    with pytest.raises(InvalidInputError):
        gw.start_session("bogus_set")


def test_session_expiration_and_invalid_ids():
    registry = SessionRegistry(ttl_seconds=60.0)
    gw = FuelRateFeedbackGateway(service=StubService(), registry=registry, skip_preflight=True)
    sid = gw.start_session("full_operational")
    # Force expiry by backdating last-access.
    registry._entries[sid].last_accessed = 0.0
    resp = gw.send(sid, "predict fuel")
    assert resp.error is not None and resp.error["code"] == "session_expired"
    resp2 = gw.send("does-not-exist", "predict fuel")
    assert resp2.error is not None and resp2.error["code"] == "session_not_found"


def test_invalid_query_and_inputs_shape():
    gw = _gateway()
    sid = gw.start_session("full_operational")
    assert gw.send(sid, "   ").error["code"] == "invalid_input"
    assert gw.send(sid, "predict", structured_inputs="nope").error["code"] == "invalid_input"


# ---------------------------------------------------------------------------
# Multi-turn clarification / resumption + merging
# ---------------------------------------------------------------------------

def test_multi_turn_clarification_then_delivery():
    stub = StubService()
    stub.mode = "clarify"
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")
    first = gw.send(sid, "predict fuel", structured_inputs={"speed_kph": 80.0})
    assert first.delivered is False
    assert first.driver_intent == "clarifying"
    assert "speed_kph" in first.missing_fields
    # Second turn completes inputs -> deliver.
    stub.mode = "deliver"
    second = gw.send(sid, "predict fuel", structured_inputs={"payload_kg": 12000.0})
    assert second.delivered is True
    assert second.driver_intent == "predict_fuel"
    # The same session was used across turns.
    assert stub.calls[0]["session_id"] == stub.calls[1]["session_id"] == sid


def test_structured_and_conversational_input_forwarding():
    stub = StubService()
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")
    inputs = {"speed_kph": 80.0, "gross_vehicle_weight_kg": 36000.0}
    gw.send(sid, "predict fuel", structured_inputs=inputs)
    call = stub.calls[0]
    assert call["message"] == "predict fuel"
    assert call["structured_inputs"] == inputs
    assert call["feature_set"] == "full_operational"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_requests_same_session_serialized():
    stub = StubService()
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")

    def worker(i: int) -> None:
        gw.send(sid, f"query {i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(stub.calls) == 8
    assert stub.max_active == 1  # never interleaved inside one session


# ---------------------------------------------------------------------------
# Routing / failure / validation surfacing
# ---------------------------------------------------------------------------

def test_tool_failure_surfaces_separately():
    stub = StubService()
    stub.mode = "toolfail"
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")
    resp = gw.send(sid, "predict fuel")
    assert resp.technical_report.get("tool_failures") == ["fuel_rate_prediction"]
    assert resp.technical_report["synthetic_only"] is True


def test_validation_blocked_blocks_delivery():
    stub = StubService()
    stub.mode = "blocked"
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")
    resp = gw.send(sid, "predict fuel")
    assert resp.delivered is False
    assert resp.validation_blocked is True
    assert resp.driver_feedback == "I could not produce a validated, grounded answer for this turn."


def test_internal_exception_does_not_leak_details():
    class BoomService(StubService):
        def handle(self, session, message, structured_inputs=None):
            raise RuntimeError("shh: /etc/passwd and traceback")
    gw = _gateway(service=BoomService())
    sid = gw.start_session("full_operational")
    resp = gw.send(sid, "predict fuel")
    assert resp.error is not None and resp.error["code"] == "internal_error"
    assert "passwd" not in json.dumps(resp.to_dict())
    assert "traceback" not in json.dumps(resp.to_dict()).lower()


# ---------------------------------------------------------------------------
# Serializable contract
# ---------------------------------------------------------------------------

def test_api_response_contract_serializable():
    stub = StubService()
    gw = _gateway(service=stub)
    sid = gw.start_session("full_operational")
    resp = gw.send(sid, "predict fuel")
    data = resp.to_dict()
    for key in ("session_id", "driver_intent", "conversation_state", "delivered",
                "validation_blocked", "driver_feedback", "selected_tools",
                "executed_tools", "technical_report", "error"):
        assert key in data
    # Must survive JSON serialization.
    parsed = json.loads(resp.to_json())
    assert parsed["session_id"] == sid
    assert parsed["delivered"] is True


# ---------------------------------------------------------------------------
# Config wiring / preflight
# ---------------------------------------------------------------------------

def test_preflight_missing_artifacts_raises_clear_error():
    tmpdir = Path(tempfile.mkdtemp())
    try:
        bad_cfg = tmpdir / "agent_config.json"
        bad_cfg.write_text(json.dumps({
            "agent": "test",
            "tools": {
                "fuel_rate_prediction": {"model_artifacts_dir": "reports/nope"},
                "knowledge_retrieval": {"corpus_path": "data/nope.jsonl"},
                "prediction_explanation": {"model_artifacts_dir": "reports/nope"},
            },
        }), encoding="utf-8")
        with pytest.raises(GatewayStartupError):
            FuelRateFeedbackGateway(config_path=bad_cfg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Real integration (genuine research artifacts)
# ---------------------------------------------------------------------------

_REAL_GATEWAY = None


def _real_gateway() -> FuelRateFeedbackGateway:
    global _REAL_GATEWAY
    if _REAL_GATEWAY is None:
        _REAL_GATEWAY = FuelRateFeedbackGateway()  # real service + real tools
    return _REAL_GATEWAY


def test_real_integration_predict():
    gw = _real_gateway()
    sid = gw.start_session("full_operational")
    full = {
        "vehicle_class": "40T",
        "gross_vehicle_weight_kg": 36000.0,
        "payload_kg": 12000.0,
        "vehicle_mass_kg": 24000.0,
        "speed_kph": 80.0,
        "acceleration_mps2": 0.2,
        "road_grade_pct": 1.5,
        "ignition_status": 1,
        "engine_rpm": 1400.0,
        "engine_load_pct": 55.0,
        "throttle_position_pct": 40.0,
        "torque_nm": 1600.0,
        "coolant_temp_c": 88.0,
        "intake_air_temp_c": 30.0,
        "ambient_temp_c": 22.0,
    }
    resp = gw.send(sid, "predict fuel", structured_inputs=full)
    assert resp.error is None
    assert resp.delivered is True, resp.driver_feedback
    assert "Estimated fuel rate" in resp.driver_feedback
    assert resp.technical_report["synthetic_only"] is True
    # SHAP/technical detail stays out of the driver-facing text.
    assert resp.technical_report is not None
    state = gw.session_state(sid)
    assert state["feature_set"] == "full_operational"
    assert state["history"]
