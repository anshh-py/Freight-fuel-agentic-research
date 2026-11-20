"""M6.1 — stable API response schema for the driver-feedback gateway.

Defines the serializable contract returned by :class:`src.api.gateway.FuelRateFeedbackGateway`.
It is a *view* over the service's ``DriverConversationResponse``: it selects the
stable fields a UI/fleet operator consumes, keeps the technical report separate
from the driver-facing feedback, and adds an explicit boundary-error channel.

All telemetry and prediction targets are synthetic; the response always carries
``synthetic_only=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class APIResponse:
    """Serializable response returned for one gateway ``send`` call.

    Fields are intentionally stable and UI-friendly. The technical report is
    kept separate from the driver-facing feedback text (which is always the
    validated ``driver_feedback`` string produced by the frozen pipeline).
    """

    session_id: str
    driver_intent: str
    conversation_state: str
    delivered: bool
    validation_blocked: bool
    driver_feedback: str
    # Optional clarification surface.
    clarification_request: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: dict[str, str] = field(default_factory=dict)
    # Executed tool surface.
    selected_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    # Technical / fleet-operator view (kept separate from driver feedback).
    technical_report: dict = field(default_factory=dict)
    # Boundary error channel — populated only on gateway/request errors, never
    # on validated conversation outcomes. Intentionally never contains stack
    # traces, secrets, or internal paths.
    error: Optional[dict] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "driver_intent": self.driver_intent,
            "conversation_state": self.conversation_state,
            "delivered": self.delivered,
            "validation_blocked": self.validation_blocked,
            "driver_feedback": self.driver_feedback,
            "clarification_request": self.clarification_request,
            "missing_fields": list(self.missing_fields),
            "invalid_fields": dict(self.invalid_fields),
            "selected_tools": list(self.selected_tools),
            "executed_tools": list(self.executed_tools),
            "technical_report": self.technical_report,
            "error": self.error,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, ensure_ascii=False)


def api_error_response(
    session_id: str,
    *,
    code: str,
    message: str,
    driver_intent: str = "error",
    conversation_state: str = "error",
) -> APIResponse:
    """Build an ``APIResponse`` carrying a boundary error (no internals)."""
    return APIResponse(
        session_id=session_id,
        driver_intent=driver_intent,
        conversation_state=conversation_state,
        delivered=False,
        validation_blocked=False,
        driver_feedback="The request could not be processed.",
        error={"code": code, "message": message},
    )
