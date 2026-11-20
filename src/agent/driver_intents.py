"""M4.3 — driver-facing intent router and clarification flow.

This module adds a conversational routing layer on top of the frozen M2 agent
architecture. It is a *new adapter* built to the existing ``IntentRouter``
shape: it reuses ``DeterministicIntentRouter`` for standard intent signal,
maps the result onto driver-facing intent categories, and adds a clarification
pass that consults the M4.2 ``DriverSession`` / partial-validation adapter.

Design rules (enforced here):

* No M2 source is modified — the router consumes ``IntentRouter``,
  ``DeterministicIntentRouter``, ``INTENT_TOOLS`` and the M2 tools through
  their existing public interfaces.
* It never invents numeric telemetry: if a driver intent needs model inputs
  (predict / explain / combined / report-of-behavior) and the required fields
  are not present in the session context or the supplied inputs, the router
  returns ``CLARIFYING`` with the missing fields and **no** selected tools.
* Qualitative driver observations remain separate and are never converted into
  exact payload / weight / speed / grade measurements.
* Knowledge-only intents do not require prediction inputs.
* Deterministic, no LLM, no database, no external service.
* No prediction / SHAP / RAG logic is duplicated here — tool execution still
  goes through the existing tool interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agent.driver_session import (
    ConversationState,
    DriverSession,
    validate_partial_vehicle_inputs,
)
from src.agent.intents import (
    TOOL_EXPLAIN,
    TOOL_PREDICT,
    TOOL_RETRIEVAL,
    DeterministicIntentRouter,
    Intent,
    IntentResult,
)


class DriverIntent(Enum):
    """Driver-facing intent categories for the M4 conversation."""

    REPORT_BEHAVIOR = "report_behavior"
    PREDICT_FUEL = "predict_fuel"
    EXPLAIN_PREDICTION = "explain_prediction"
    KNOWLEDGE_QUERY = "knowledge_query"
    COMBINED_REQUEST = "combined_request"
    CLARIFYING = "clarifying"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


#: Driver intents that require a complete validated model-input set before any
#: prediction / explanation tool may be invoked.
REQUIRES_MODEL_INPUTS = frozenset(
    {
        DriverIntent.REPORT_BEHAVIOR,
        DriverIntent.PREDICT_FUEL,
        DriverIntent.EXPLAIN_PREDICTION,
        DriverIntent.COMBINED_REQUEST,
    }
)

#: Driver intent -> underlying M2 intent that provides the same tool signal.
_DRIVER_TO_BASE: dict[DriverIntent, tuple[Intent, ...]] = {
    DriverIntent.PREDICT_FUEL: (Intent.PREDICT_FUEL,),
    DriverIntent.KNOWLEDGE_QUERY: (Intent.KNOWLEDGE_QUERY,),
    DriverIntent.EXPLAIN_PREDICTION: (Intent.EXPLAIN_PREDICTION,),
    DriverIntent.COMBINED_REQUEST: (
        Intent.COMBINED_PREDICT_EXPLAIN,
        Intent.COMBINED_KNOWLEDGE_PREDICT,
    ),
    DriverIntent.UNSUPPORTED: (Intent.UNSUPPORTED,),
    DriverIntent.AMBIGUOUS: (Intent.AMBIGUOUS,),
}


def _driver_tools(driver: DriverIntent, base: Intent | None = None) -> list[str]:
    """Resolve the tool list for a driver intent (empty when none allowed)."""
    if driver == DriverIntent.PREDICT_FUEL:
        return [TOOL_PREDICT]
    if driver == DriverIntent.EXPLAIN_PREDICTION:
        return [TOOL_EXPLAIN]
    if driver == DriverIntent.KNOWLEDGE_QUERY:
        return [TOOL_RETRIEVAL]
    if driver == DriverIntent.COMBINED_REQUEST:
        if base == Intent.COMBINED_KNOWLEDGE_PREDICT:
            return [TOOL_RETRIEVAL, TOOL_PREDICT]
        return [TOOL_PREDICT, TOOL_EXPLAIN]
    return []  # report (needs clarification), clarifying, unsupported, ambiguous


# ---------------------------------------------------------------------------
# Driver-language keyword banks (deterministic, no LLM)
# ---------------------------------------------------------------------------

_REPORT_BEHAVIOR_WORDS = (
    "consuming too much",
    "too much diesel",
    "using too much",
    "burning too much",
    "driving uphill",
    "climbing uphill",
    "heavy load",
    "running heavy",
    "hauling",
    "fully loaded",
    "my truck",
    "i'm driving",
    "i am driving",
    "report",
    "behavio",
)

_EXPLAIN_WORDS = (
    "explain",
    "explanation",
    "why is",
    "why did",
    "why has",
    "why does",
    "contribute",
    "factor",
    "what drives",
)

_KNOWLEDGE_ADVISORY_WORDS = (
    "what should i do",
    "how to reduce",
    "reduce fuel",
    "save fuel",
    "tips",
    "advice",
    "recommend",
    "what can i do",
    "improve",
    "best way",
    "more efficient",
    "fuel efficient",
    "how do i",
    "to reduce unnecessary",
)


def _count_words(text: str, bank: tuple[str, ...]) -> int:
    return sum(1 for w in bank if w in text)


@dataclass
class DriverIntentResult:
    """Outcome of routing one driver turn.

    Structurally mirrors ``IntentResult`` (``intent``, ``confidence``,
    ``reason``, ``backend``, ``to_dict``) so it can be used anywhere the M2
    ``IntentRouter`` protocol is expected, while adding the M4 fields needed
    by the clarification flow.
    """

    intent: DriverIntent
    confidence: float
    reason: str
    backend: str = "deterministic"
    selected_tools: list[str] = field(default_factory=list)
    base_intent: Intent | None = None
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: dict[str, str] = field(default_factory=dict)
    is_ready: bool = True
    requires_inputs: bool = False
    resumes_pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_intent": self.intent.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "backend": self.backend,
            "selected_tools": list(self.selected_tools),
            "base_intent": self.base_intent.value if self.base_intent else None,
            "missing_fields": list(self.missing_fields),
            "invalid_fields": dict(self.invalid_fields),
            "is_ready": self.is_ready,
            "requires_inputs": self.requires_inputs,
            "resumes_pending": self.resumes_pending,
        }


class DriverIntentRouter:
    """Deterministic driver-facing router built on the M2 router.

    Implements the ``IntentRouter`` call shape::

        def route(self, query, structured_inputs=None, session=None)

    ``session`` is optional (M4.2). When provided, the session's validated
    context is used as carry-over context for the clarification decision and
    the driver intent can resume a pending request once inputs are complete.
    """

    BACKEND = "deterministic_driver"

    def __init__(
        self,
        base_router: DeterministicIntentRouter | None = None,
        default_feature_set: str = "full_operational",
    ) -> None:
        self.base_router = base_router or DeterministicIntentRouter()
        self.default_feature_set = default_feature_set

    # -- public routing --------------------------------------------------
    def route(
        self,
        query: str,
        structured_inputs: dict[str, Any] | None = None,
        session: DriverSession | None = None,
    ) -> DriverIntentResult:
        q = (query or "").strip().lower()

        # 0. Safety-first: out-of-scope / unsupported requests terminate here,
        #    before any driver-specific detection. This guarantees e.g.
        #    "real-world validation of my truck" stays UNSUPPORTED even though
        #    it mentions "my truck".
        base_early = self.base_router.route(query, structured_inputs)
        if base_early.intent == Intent.UNSUPPORTED:
            return DriverIntentResult(
                intent=DriverIntent.UNSUPPORTED,
                confidence=base_early.confidence,
                reason=base_early.reason,
                backend=self.BACKEND,
                base_intent=Intent.UNSUPPORTED,
                is_ready=True,
            )

        # 1. Driver-specific behavior-report detection happens before falling
        #    back to the M2 router, because a report is a distinct driver
        #    preamble that must be captured in its own category.
        if q and _count_words(q, _REPORT_BEHAVIOR_WORDS) > 0:
            return self._resolve_requires_inputs(
                DriverIntent.REPORT_BEHAVIOR,
                0.7,
                "driver reports driving/consumption behaviour",
                structured_inputs,
                session,
            )

        # 2. Explanation / advisory-knowledge intents are recognised before the
        #    generic M2 fallback so driver phrasing maps to the right category.
        if q and _count_words(q, _EXPLAIN_WORDS) > 0:
            return self._resolve_requires_inputs(
                DriverIntent.EXPLAIN_PREDICTION,
                0.75,
                "driver asks why/explanation for fuel rate",
                structured_inputs,
                session,
            )

        if q and _count_words(q, _KNOWLEDGE_ADVISORY_WORDS) > 0:
            return DriverIntentResult(
                intent=DriverIntent.KNOWLEDGE_QUERY,
                confidence=0.8,
                reason="driver asks for knowledge / advisory guidance",
                backend=self.BACKEND,
                selected_tools=[TOOL_RETRIEVAL],
                base_intent=Intent.KNOWLEDGE_QUERY,
                is_ready=True,
            )

        # 3. Fall back to the existing M2 deterministic router for the standard
        #    intent signal, then map onto a driver-facing category.
        base: IntentResult = self.base_router.route(query, structured_inputs)
        driver = self._from_base(base.intent)
        reason = base.reason
        confidence = base.confidence

        # Unsupported/ambiguous terminate immediately (no input requirements).
        if driver in (DriverIntent.UNSUPPORTED, DriverIntent.AMBIGUOUS):
            return DriverIntentResult(
                intent=driver,
                confidence=confidence,
                reason=reason,
                backend=self.BACKEND,
                base_intent=base.intent,
                is_ready=True,
            )

        return self._resolve_requires_inputs(
            driver,
            confidence,
            reason,
            structured_inputs,
            session,
            base_intent=base.intent,
        )

    def resume_pending(
        self,
        session: DriverSession,
        structured_inputs: dict[str, Any] | None = None,
    ) -> DriverIntentResult:
        """Continue a previously-clarified pending request.

        Uses the session's recorded last request to recover the actionable
        driver intent, then re-runs the clarification check with the
        session's (now possibly complete) carry-over context. If inputs are
        still incomplete the result is ``CLARIFYING`` with the remaining
        missing fields; otherwise the actionable intent with
        ``resumes_pending=True``.
        """
        last = session.last_request or {}
        intent_name = last.get("intent")
        if not intent_name:
            return DriverIntentResult(
                intent=DriverIntent.AMBIGUOUS,
                confidence=0.3,
                reason="no pending request recorded to resume",
                backend=self.BACKEND,
                is_ready=False,
            )

        driver = self._driver_from_value(intent_name)
        return self._resolve_requires_inputs(
            driver,
            0.7,
            f"resuming pending {driver.value} request",
            structured_inputs if structured_inputs else last.get("structured_inputs"),
            session,
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _from_base(base_intent: Intent) -> DriverIntent:
        for driver, bases in _DRIVER_TO_BASE.items():
            if base_intent in bases:
                return driver
        return DriverIntent.AMBIGUOUS

    @staticmethod
    def _driver_from_value(value: str) -> DriverIntent:
        for driver in DriverIntent:
            if driver.value == value:
                return driver
        return DriverIntent.AMBIGUOUS

    def _resolve_requires_inputs(
        self,
        driver: DriverIntent,
        confidence: float,
        reason: str,
        structured_inputs: dict[str, Any] | None,
        session: DriverSession | None,
        base_intent: Intent | None = None,
    ) -> DriverIntentResult:
        if driver not in REQUIRES_MODEL_INPUTS:
            return DriverIntentResult(
                intent=driver,
                confidence=confidence,
                reason=reason,
                backend=self.BACKEND,
                selected_tools=_driver_tools(driver, base_intent),
                base_intent=base_intent,
                is_ready=True,
            )

        fs = session.feature_set if session is not None else self.default_feature_set
        carryover = session.truck_context.copy() if (session is not None and session.truck_context) else None

        check = validate_partial_vehicle_inputs(
            structured_inputs or {},
            feature_set=fs,
            existing_context=carryover,
        )

        if check.is_ready:
            # The request is resumable: all required inputs are now available.
            # A behavior * is itself not directly actionable; it resolves to
            # the prediction intent (M4.1 routing table: report -> predict).
            resolved = (
                DriverIntent.PREDICT_FUEL
                if driver == DriverIntent.REPORT_BEHAVIOR
                else driver
            )
            resolved_base = (
                Intent.PREDICT_FUEL
                if driver == DriverIntent.REPORT_BEHAVIOR
                else base_intent
            )
            return DriverIntentResult(
                intent=resolved,
                confidence=confidence,
                reason=reason + " (inputs complete)",
                backend=self.BACKEND,
                selected_tools=_driver_tools(resolved, resolved_base),
                base_intent=resolved_base,
                is_ready=True,
                requires_inputs=True,
                resumes_pending=True,
            )

        # Missing/invalid required inputs -> clarify, never invoke tools.
        return DriverIntentResult(
            intent=DriverIntent.CLARIFYING,
            confidence=0.6,
            reason="required model inputs incomplete; asking driver for clarification",
            backend=self.BACKEND,
            selected_tools=[],
            base_intent=base_intent,
            missing_fields=list(check.missing_fields),
            invalid_fields=dict(check.invalid_fields),
            is_ready=False,
            requires_inputs=True,
            resumes_pending=False,
        )
