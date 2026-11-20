"""M4.2 — DriverSession and partial-input validation.

This module provides the conversational session state (``DriverSession``) and
the partial-input validation adapter (``validate_partial_vehicle_inputs``) for
the M4 driver-feedback system.

Design rules (enforced here):

* Only genuine driver-supplied numeric/categorical telemetry is accepted. The
  adapter **never invents missing values** and **never converts qualitative
  wording into an exact measurement** (e.g. "heavy" is never parsed into a
  ``payload_kg`` number).
* Qualitative driver observations (free-text remarks) are kept **separately**
  from numeric telemetry and are never fed to the prediction/explanation tools.
* The existing M2 validation rules (``src.tools.validation``) are reused for the
  fields that *are* supplied; the strict validator
  (``validate_vehicle_inputs``) is delegated to only once every required field
  for the selected feature set is present.

Safety/provenance framing: this pipeline consumes driver context for the
synthetic fuel-rate guidance system. It does not represent real-world sensor
observations and never fabricates data to make a prediction possible.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.features.spec import get_feature_set, load_feature_spec
from src.tools.validation import (
    FIELD_RANGES,
    IGNITION_ALLOWED,
    VEHICLE_CLASSES,
    ToolInputError,
    normalize_feature_set,
    validate_vehicle_inputs,
)

#: Free-text keys treated as qualitative driver observations (never telemetry).
QUALITATIVE_FIELDS: frozenset[str] = frozenset(
    {
        "driver_observation",
        "driving_condition",
        "route_context",
        "comment",
    }
)


class ConversationState(Enum):
    """Conversational states tracked by a :class:`DriverSession`."""

    AWAITING_INPUT = "awaiting_input"
    CLARIFYING = "clarifying"
    READY_PREDICT = "ready_predict"
    READY_EXPLAIN = "ready_explain"
    ANSWERED = "answered"
    UNSUPPORTED = "unsupported"


# ---------------------------------------------------------------------------
# Partial-input validation adapter
# ---------------------------------------------------------------------------


def _coerce_numeric(value: Any, field: str) -> float:
    """Parse a numeric token exactly like ``validation._as_float``.

    Mirrors the strict M2 coercion rules so partial validation agrees with the
    strict validator: booleans are rejected, and non-numeric strings are
    rejected (this is what prevents "heavy" from becoming a number).
    """
    if isinstance(value, bool):
        raise ToolInputError(f"field {field!r} must be numeric, got boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            raise ToolInputError(
                f"field {field!r} is not a valid number: {value!r}"
            ) from None
    raise ToolInputError(
        f"field {field!r} has unsupported type {type(value).__name__}"
    )


def _validate_field(field: str, raw: Any) -> tuple[bool, Any]:
    """Validate a single supplied field.

    Returns ``(ok, value)`` where ``value`` is the cleaned value when ``ok``
    is true, otherwise a human-readable reason string.
    """
    if field == "vehicle_class":
        if raw not in VEHICLE_CLASSES:
            return False, f"must be one of {list(VEHICLE_CLASSES)}; got {raw!r}"
        return True, raw

    if field == "ignition_status":
        try:
            num = _coerce_numeric(raw, field)
        except ToolInputError as exc:
            return False, str(exc)
        if int(num) not in IGNITION_ALLOWED:
            return False, "must be 0 or 1"
        return True, int(num)

    try:
        num = _coerce_numeric(raw, field)
    except ToolInputError as exc:
        return False, str(exc)

    lo, hi = FIELD_RANGES[field]
    if not (lo <= num <= hi):
        return False, f"value {num} outside permissible range [{lo}, {hi}]"
    return True, num


@dataclass
class PartialValidationResult:
    """Outcome of validating a partial set of driver-provided inputs."""

    feature_set: str
    provided_fields: list[str]
    validated_inputs: dict[str, Any]
    missing_fields: list[str]
    invalid_fields: dict[str, str]
    qualitative_observations: dict[str, Any]
    is_ready: bool

    @property
    def readiness(self) -> str:
        return "ready" if self.is_ready else "needs_input"

    def to_strict_validated(self) -> tuple[str, dict[str, Any]]:
        """Delegate to the strict M2 validator once all fields are available."""
        if not self.is_ready:
            raise ToolInputError(
                "cannot finalize to strict validator: missing required field(s) "
                f"{self.missing_fields}; invalid field(s): {self.invalid_fields}"
            )
        return validate_vehicle_inputs(self.validated_inputs, self.feature_set)


def validate_partial_vehicle_inputs(
    inputs: dict[str, Any],
    feature_set: str = "full_operational",
    existing_context: dict[str, Any] | None = None,
    qualitative_fields: frozenset[str] = QUALITATIVE_FIELDS,
) -> PartialValidationResult:
    """Validate incomplete driver-provided inputs against a feature set.

    ``existing_context`` carries previously-validated fields (e.g. carry-over
    vehicle/trip context already stored in the session) so the driver is not
    re-asked for unchanged inputs. Driver-supplied ``inputs`` override context.

    Rules:
      * unknown (non-spec, non-qualitative) fields are recorded as invalid,
      * qualitative keys are kept separately and never treated as telemetry,
      * a field is ``missing`` only if absent from both inputs and context,
      * numeric/categorical fields are validated with the existing M2 rules,
      * ``is_ready`` is true iff no field is missing or invalid.
    """
    fs = normalize_feature_set(feature_set)
    spec = load_feature_spec()
    features = get_feature_set(spec, fs)

    if not isinstance(inputs, dict):
        raise ToolInputError(
            f"inputs must be a dict, got {type(inputs).__name__}"
        )
    context = existing_context or {}
    if not isinstance(context, dict):
        raise ToolInputError(
            f"existing_context must be a dict, got {type(context).__name__}"
        )

    candidate: dict[str, Any] = {**context, **inputs}  # inputs override context

    unexpected = set(candidate) - set(features) - set(qualitative_fields)
    invalid: dict[str, str] = {
        f: "unexpected field (not in feature set)" for f in sorted(unexpected)
    }

    qualitative: dict[str, Any] = {}
    for f in sorted(qualitative_fields):
        if f in candidate:
            qualitative[f] = candidate[f]

    validated: dict[str, Any] = {}
    missing: list[str] = []
    for field in features:
        if field not in candidate:
            missing.append(field)
            continue
        ok, val = _validate_field(field, candidate[field])
        if ok:
            validated[field] = val
        else:
            invalid[field] = val

    is_ready = (not missing) and (not invalid)
    return PartialValidationResult(
        feature_set=fs,
        provided_fields=sorted(set(inputs)),
        validated_inputs=validated,
        missing_fields=missing,
        invalid_fields=invalid,
        qualitative_observations=qualitative,
        is_ready=is_ready,
    )


# ---------------------------------------------------------------------------
# DriverSession model
# ---------------------------------------------------------------------------


@dataclass
class DriverSession:
    """In-memory conversation state for a single driver session.

    Stores validated vehicle/trip context separately from qualitative
    observations, tracks pending clarification fields, keeps the last request
    and tool results, and maintains a compact conversation history. No
    database or external service is used — persistence is in-memory for this
    milestone.
    """

    feature_set: str = "full_operational"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    truck_context: dict[str, Any] = field(default_factory=dict)
    qualitative_observations: dict[str, Any] = field(default_factory=dict)
    pending_clarification: list[str] = field(default_factory=list)
    last_request: dict[str, Any] | None = None
    last_tool_results: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    conversation_state: ConversationState = ConversationState.AWAITING_INPUT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # -- lifecycle --------------------------------------------------------
    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    # -- input handling ---------------------------------------------------
    def apply_partial_inputs(
        self,
        inputs: dict[str, Any],
        feature_set: str | None = None,
    ) -> PartialValidationResult:
        """Validate driver inputs against the session's carried context.

        Updates the session's validated context, qualitative observations,
        pending clarification fields, and conversation state in place.
        """
        fs = feature_set or self.feature_set
        result = validate_partial_vehicle_inputs(
            inputs,
            feature_set=fs,
            existing_context=self.truck_context,
        )
        self.feature_set = fs
        self.truck_context.update(result.validated_inputs)
        self.qualitative_observations.update(result.qualitative_observations)
        self.pending_clarification = list(result.missing_fields)

        if result.is_ready:
            self.conversation_state = ConversationState.READY_PREDICT
        else:
            self.conversation_state = ConversationState.CLARIFYING
        self._touch()
        return result

    # -- request / tool tracking -----------------------------------------
    def record_request(
        self,
        query: str,
        structured_inputs: dict[str, Any] | None = None,
        intent: str | None = None,
    ) -> None:
        """Record the user's latest request and append a compact history turn."""
        self.last_request = {
            "query": query,
            "structured_inputs": structured_inputs,
            "intent": intent,
        }
        self.history.append(
            {
                "role": "user",
                "query": query,
                "intent": intent,
                "timestamp": self._now(),
            }
        )
        self._touch()

    def record_tool_result(self, results: list[dict[str, Any]] | dict[str, Any]) -> None:
        """Store the latest tool results and append a history turn."""
        items = results if isinstance(results, list) else [results]
        self.last_tool_results = list(items)
        self.history.append(
            {"role": "tool", "tool_results": self.last_tool_results, "timestamp": self._now()}
        )
        self._touch()

    def mark_answered(self) -> None:
        self.conversation_state = ConversationState.ANSWERED
        self._touch()

    def mark_unsupported(self) -> None:
        self.conversation_state = ConversationState.UNSUPPORTED
        self._touch()

    # -- finalization ----------------------------------------------------
    def finalize_for_prediction(self) -> tuple[str, dict[str, Any]]:
        """Return strict-validated inputs once all required fields are present.

        Raises ``ToolInputError`` if any required field is still missing or
        was invalid. This is the only path that delegates to the M2 strict
        validator, satisfying "call the existing strict validator only when
        all required inputs are available".
        """
        spec = load_feature_spec()
        features = get_feature_set(spec, self.feature_set)
        missing = [f for f in features if f not in self.truck_context]
        if missing:
            raise ToolInputError(
                f"missing required field(s): {missing} "
                f"(feature set {self.feature_set!r})"
            )
        return validate_vehicle_inputs(self.truck_context, self.feature_set)

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "feature_set": self.feature_set,
            "conversation_state": self.conversation_state.value,
            "truck_context": self.truck_context,
            "qualitative_observations": self.qualitative_observations,
            "pending_clarification": self.pending_clarification,
            "last_request": self.last_request,
            "last_tool_results": self.last_tool_results,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
