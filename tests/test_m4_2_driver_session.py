"""M4.2 — focused tests for DriverSession and partial-input validation.

Covers session state, partial/carry-over inputs, invalid-input rejection,
missing-field detection, and the guarantee that neither fabricated numeric
telemetry nor qualitative-to-measurement conversion ever occurs.
"""

from __future__ import annotations

import pytest

from src.agent.driver_session import (
    ConversationState,
    DriverSession,
    PartialValidationResult,
    validate_partial_vehicle_inputs,
)
from src.features.spec import get_feature_set, load_feature_spec
from src.tools.validation import ToolInputError, validate_vehicle_inputs

SPEC = load_feature_spec()
FULL_FEATURES = get_feature_set(SPEC, "full_operational")
TELE_FEATURES = get_feature_set(SPEC, "telematics_only")

#: Required fields shared by both feature sets (non-engine engine signals).
ENGINE_FEATURES = ["engine_rpm", "engine_load_pct", "throttle_position_pct", "torque_nm"]


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


def _valid_telematics_inputs() -> dict:
    return {k: v for k, v in _valid_full_inputs().items() if k not in ENGINE_FEATURES}


# ---------------------------------------------------------------------------
# Partial-validation adapter
# ---------------------------------------------------------------------------


def test_partial_missing_fields_for_full_feature_set():
    result = validate_partial_vehicle_inputs({"speed_kph": 75.0})
    assert isinstance(result, PartialValidationResult)
    assert result.feature_set == "full_operational"
    # Only speed_kph supplied -> every other feature is missing.
    assert result.missing_fields == [f for f in FULL_FEATURES if f != "speed_kph"]
    assert result.validated_inputs == {"speed_kph": 75.0}
    assert result.invalid_fields == {}
    assert not result.is_ready
    assert result.readiness == "needs_input"


def test_partial_missing_fields_for_telematics_feature_set():
    result = validate_partial_vehicle_inputs(
        {"vehicle_class": "32T", "ignition_status": 1}, feature_set="telematics_only"
    )
    assert result.feature_set == "telematics_only"
    expected_missing = [f for f in TELE_FEATURES if f not in ("vehicle_class", "ignition_status")]
    assert result.missing_fields == expected_missing
    assert result.validated_inputs == {"vehicle_class": "32T", "ignition_status": 1}


def test_partial_ready_when_all_required_present():
    result = validate_partial_vehicle_inputs(_valid_full_inputs())
    assert result.is_ready
    assert result.readiness == "ready"
    assert result.missing_fields == []
    assert result.invalid_fields == {}
    # validated inputs reproduce the cleaned dict of the strict validator.
    _, cleaned = validate_vehicle_inputs(_valid_full_inputs(), "full_operational")
    assert result.validated_inputs == cleaned


def test_partial_rejects_out_of_range_value():
    result = validate_partial_vehicle_inputs({"speed_kph": 250.0})
    assert "speed_kph" in result.invalid_fields
    assert "outside permissible range" in result.invalid_fields["speed_kph"]
    assert "speed_kph" not in result.validated_inputs
    assert not result.is_ready


def test_partial_rejects_non_numeric_qualitative_as_telemetry():
    # A qualitative description must never be silently converted into a number.
    result = validate_partial_vehicle_inputs({"payload_kg": "heavy"})
    assert "payload_kg" in result.invalid_fields
    assert "not a valid number" in result.invalid_fields["payload_kg"]
    assert "payload_kg" not in result.validated_inputs


def test_partial_rejects_boolean_numeric():
    result = validate_partial_vehicle_inputs({"speed_kph": True})
    assert "speed_kph" in result.invalid_fields
    assert "boolean" in result.invalid_fields["speed_kph"]


def test_partial_rejects_unknown_field():
    result = validate_partial_vehicle_inputs({"horsepower": 500})
    assert "horsepower" in result.invalid_fields
    assert "unexpected field" in result.invalid_fields["horsepower"]
    assert not result.is_ready


def test_partial_keeps_qualitative_observations_separate():
    inputs = {"speed_kph": 75.0, "driver_observation": "climbing a long grade"}
    result = validate_partial_vehicle_inputs(inputs)
    assert result.qualitative_observations == {"driver_observation": "climbing a long grade"}
    # Not treated as telemetry, not missing, not invalid.
    assert "driver_observation" not in result.validated_inputs
    assert "driver_observation" not in result.missing_fields
    assert "driver_observation" not in result.invalid_fields


def test_partial_carryover_context_counts_toward_ready():
    first = validate_partial_vehicle_inputs(
        {"vehicle_class": "40T", "speed_kph": 75.0}
    )
    assert not first.is_ready
    second = validate_partial_vehicle_inputs(
        {"payload_kg": 12000.0},
        existing_context=first.validated_inputs,
    )
    # Still not all present.
    assert not second.is_ready
    remaining = {f: _valid_full_inputs()[f] for f in FULL_FEATURES if f not in second.validated_inputs}
    third = validate_partial_vehicle_inputs(remaining, existing_context=second.validated_inputs)
    assert third.is_ready
    assert third.validated_inputs == _valid_full_inputs()


def test_partial_to_strict_only_when_ready():
    partial = validate_partial_vehicle_inputs({"speed_kph": 75.0})
    with pytest.raises(ToolInputError):
        partial.to_strict_validated()
    ready = validate_partial_vehicle_inputs(_valid_full_inputs())
    fs, cleaned = ready.to_strict_validated()
    assert fs == "full_operational"
    assert cleaned == _valid_full_inputs()


def test_partial_never_invents_telemetry():
    result = validate_partial_vehicle_inputs({"speed_kph": 75.0})
    supplied = {"speed_kph"}
    assert set(result.validated_inputs) == supplied
    assert set(result.validated_inputs) <= supplied
    # Missing numeric fields are reported, never defaulted.
    assert "payload_kg" in result.missing_fields
    assert "payload_kg" not in result.validated_inputs


# ---------------------------------------------------------------------------
# DriverSession model
# ---------------------------------------------------------------------------


def test_driver_session_initial_state():
    s = DriverSession()
    assert s.session_id
    assert s.feature_set == "full_operational"
    assert s.conversation_state == ConversationState.AWAITING_INPUT
    assert s.truck_context == {}
    assert s.pending_clarification == []
    assert s.last_request is None
    assert s.history == []
    d = s.to_dict()
    assert d["session_id"] == s.session_id
    assert d["conversation_state"] == "awaiting_input"


def test_driver_session_partial_input_sets_clarifying_state():
    s = DriverSession()
    result = s.apply_partial_inputs({"speed_kph": 75.0})
    assert not result.is_ready
    assert s.conversation_state == ConversationState.CLARIFYING
    assert s.pending_clarification == result.missing_fields
    assert s.truck_context == {"speed_kph": 75.0}


def test_driver_session_qualitative_stored_separately():
    s = DriverSession()
    s.apply_partial_inputs(
        {"speed_kph": 75.0, "driver_observation": "heavy trailer today"}
    )
    assert s.qualitative_observations == {"driver_observation": "heavy trailer today"}
    assert "driver_observation" not in s.truck_context


def test_driver_session_reaches_ready_after_all_fields():
    s = DriverSession()
    # Feed fields one batch at a time; context carries them forward.
    batch1 = {k: v for k, v in _valid_full_inputs().items() if k in TELE_FEATURES}
    r1 = s.apply_partial_inputs(batch1)
    assert not r1.is_ready
    remaining = {f: _valid_full_inputs()[f] for f in FULL_FEATURES if f not in s.truck_context}
    r2 = s.apply_partial_inputs(remaining)
    assert r2.is_ready
    assert s.conversation_state == ConversationState.READY_PREDICT
    assert s.pending_clarification == []
    # finalize delegates to the strict validator.
    fs, cleaned = s.finalize_for_prediction()
    assert fs == "full_operational"
    assert cleaned == _valid_full_inputs()


def test_driver_session_finalize_raises_when_missing():
    s = DriverSession()
    s.apply_partial_inputs({"speed_kph": 75.0})
    with pytest.raises(ToolInputError):
        s.finalize_for_prediction()


def test_driver_session_records_request_and_tool_results():
    s = DriverSession()
    s.record_request("how much fuel on this grade?", intent="fuel_prediction")
    assert s.last_request and s.last_request["intent"] == "fuel_prediction"
    assert len(s.history) == 1
    assert s.history[0]["role"] == "user"
    s.record_tool_result([{"tool": "fuel_rate_prediction", "status": "ok"}])
    assert s.last_tool_results == [{"tool": "fuel_rate_prediction", "status": "ok"}]
    assert len(s.history) == 2
    assert s.history[1]["role"] == "tool"


def test_driver_session_state_transitions():
    s = DriverSession()
    s.mark_answered()
    assert s.conversation_state == ConversationState.ANSWERED
    s.mark_unsupported()
    assert s.conversation_state == ConversationState.UNSUPPORTED
