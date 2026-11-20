"""M4.4 — grounded conversational driver feedback service.

This module wires the M4 conversation loop together on top of the frozen M2
architecture:

    DriverSession  (M4.2)  -> session context, partial validation, history
    DriverIntentRouter (M4.3) -> driver intent + clarification flow
    M2 tools                -> FuelRatePredictionTool / PredictionExplanationTool
                               / KnowledgeRetrievalTool
    ResponseGenerator        -> deterministic grounded driver/technical response
    validate_grounded_response -> guardrail battery on every generated response

Design rules (enforced here):

* The service only ever executes the tools that the resolved driver intent
  selected. A ``CLARIFYING`` intent runs **no** prediction/explanation tool.
* Required model inputs are finalized through ``DriverSession.finalize_for_prediction``
  (which delegates to the strict M2 validator) — never silently invented.
* Knowledge-only requests run retrieval without requiring prediction inputs.
* Requests generated response is passed through the existing M2 validation
  battery. If any error-severity check fails, the response is **not**
  presented as a successful grounded answer (``delivered=False``).
* Qualitative driver observations are kept separate and never converted into
  exact measurements. No fuel savings, OEM specifications or telemetry values
  are ever invented. SHAP is always presented as attribution, never causality.
* Deterministic. No LLM, no external API, no database, no persistence beyond
  the in-memory ``DriverSession``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.driver_intents import (
    DriverIntent,
    DriverIntentResult,
    DriverIntentRouter,
)
from src.agent.driver_session import (
    ConversationState,
    DriverSession,
)
from src.agent.intents import (
    TOOL_EXPLAIN,
    TOOL_PREDICT,
    TOOL_RETRIEVAL,
    Intent,
)
from src.agent.orchestrator import AgentResponse
from src.agent.response_generator import ResponseGenerator, GroundedResponse
from src.agent.response_validation import (
    ValidationIssue,
    summary as validation_summary,
    validate_grounded_response,
)
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool

#: Driver intent -> underlying M2 intent used for the grounded-response object.
_DRIVER_TO_M2_INTENT: dict[DriverIntent, Intent] = {
    DriverIntent.PREDICT_FUEL: Intent.PREDICT_FUEL,
    DriverIntent.EXPLAIN_PREDICTION: Intent.EXPLAIN_PREDICTION,
    DriverIntent.KNOWLEDGE_QUERY: Intent.KNOWLEDGE_QUERY,
    DriverIntent.COMBINED_REQUEST: Intent.COMBINED_PREDICT_EXPLAIN,
    DriverIntent.UNSUPPORTED: Intent.UNSUPPORTED,
    DriverIntent.AMBIGUOUS: Intent.AMBIGUOUS,
}

#: Tool registry name -> callable used to execute a selected tool.
_TOOL_EFFECTORS = {
    "fuel_rate_prediction": "predict",
    "prediction_explanation": "explain",
    "knowledge_retrieval": "retrieve",
}


def _m2_intent_for(result: DriverIntentResult) -> Intent:
    """Return the M2 intent to attach to the grounded-response object."""
    if result.base_intent is not None:
        return result.base_intent
    return _DRIVER_TO_M2_INTENT.get(result.intent, Intent.AMBIGUOUS)


@dataclass
class DriverConversationResponse:
    """The full result of processing one driver turn."""

    session_id: str
    driver_intent: str
    base_intent: str | None
    conversation_state: str
    selected_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    grounded: GroundedResponse | None = None
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    validation_summary: dict = field(default_factory=dict)
    validation_blocked: bool = False
    delivered: bool = False
    clarification_request: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: dict[str, str] = field(default_factory=dict)
    driver_feedback: str = ""
    technical_report: dict = field(default_factory=dict)


class DriverConversationService:
    """Deterministic conversational front-end over the M2 tools."""

    def __init__(
        self,
        router: DriverIntentRouter | None = None,
        tools: dict[str, Any] | None = None,
        generator: ResponseGenerator | None = None,
        default_feature_set: str = "full_operational",
    ) -> None:
        self.router = router or DriverIntentRouter(
            default_feature_set=default_feature_set
        )
        if tools is None:
            tools = {
                "fuel_rate_prediction": FuelRatePredictionTool(),
                "knowledge_retrieval": KnowledgeRetrievalTool(),
                "prediction_explanation": PredictionExplanationTool(),
            }
        self.tools = tools
        self.generator = generator or ResponseGenerator()
        self.default_feature_set = default_feature_set

    # -- public entry ----------------------------------------------------
    def handle(
        self,
        session: DriverSession,
        message: str,
        structured_inputs: dict[str, Any] | None = None,
    ) -> DriverConversationResponse:
        """Process one driver turn and produce a grounded response."""
        result = self.router.route(message, structured_inputs, session=session)

        # Record the user turn (context + history).
        session.record_request(message, structured_inputs, intent=result.intent.value)

        # Persist validated inputs / qualitative observations / pending fields.
        if structured_inputs:
            session.apply_partial_inputs(structured_inputs)

        # ---- CLARIFYING: never execute prediction tools -----------------
        if result.intent == DriverIntent.CLARIFYING:
            return self._clarify(session, result, message, structured_inputs)

        # ---- UNSUPPORTED / AMBIGUOUS ------------------------------------
        if result.intent == DriverIntent.UNSUPPORTED:
            return self._unsupported(session, result, message)
        if result.intent == DriverIntent.AMBIGUOUS:
            return self._ambiguous(session, result)

        # ---- Actionable intents -----------------------------------------
        if result.intent in (
            DriverIntent.PREDICT_FUEL,
            DriverIntent.EXPLAIN_PREDICTION,
            DriverIntent.COMBINED_REQUEST,
        ):
            return self._actionable(session, result, message)
        if result.intent == DriverIntent.KNOWLEDGE_QUERY:
            return self._knowledge_only(session, result, message)

        # Defensive fallback (should not be reachable).
        return self._ambiguous(session, result)

    # -- branch handlers -------------------------------------------------
    def _clarify(
        self,
        session: DriverSession,
        result: DriverIntentResult,
        message: str,
        structured_inputs: dict[str, Any] | None,
    ) -> DriverConversationResponse:
        session.conversation_state = ConversationState.CLARIFYING
        session.pending_clarification = list(result.missing_fields)

        missing = list(result.missing_fields)
        invalid = dict(result.invalid_fields)
        prompt_parts: list[str] = []
        if missing:
            prompt_parts.append("I need the following vehicle/trip values: " + ", ".join(missing) + ".")
        if invalid:
            prompt_parts.append(
                "Some values were not acceptable: "
                + "; ".join(f"{k} ({v})" for k, v in invalid.items())
                + "."
            )
        if not prompt_parts:
            prompt_parts.append("Please give me more information about your truck and trip.")

        ask = " ".join(prompt_parts)

        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent=DriverIntent.CLARIFYING.value,
            base_intent=result.base_intent.value if result.base_intent else None,
            conversation_state=ConversationState.CLARIFYING.value,
            selected_tools=[],
            executed_tools=[],
            validation_blocked=False,
            delivered=False,
            clarification_request=ask,
            missing_fields=missing,
            invalid_fields=invalid,
            driver_feedback=ask,
            technical_report={
                "driver_intent": DriverIntent.CLARIFYING.value,
                "message": message,
                "structured_inputs": structured_inputs or {},
                "missing_fields": missing,
                "invalid_fields": invalid,
                "note": "no tools executed while inputs are incomplete",
            },
        )

    def _unsupported(
        self,
        session: DriverSession,
        result: DriverIntentResult,
        message: str,
    ) -> DriverConversationResponse:
        session.mark_unsupported()
        feedback = (
            "I can't answer that within this project's scope. The driver "
            "feedback system works with synthetic fuel-rate prediction, "
            "retrieval over a synthetic knowledge corpus, and model "
            "explanation only; it does not provide real-world, manufacturer, "
            "or maintenance information."
        )
        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent=DriverIntent.UNSUPPORTED.value,
            base_intent=result.base_intent.value if result.base_intent else None,
            conversation_state=ConversationState.UNSUPPORTED.value,
            selected_tools=[],
            executed_tools=[],
            validation_blocked=False,
            delivered=True,
            driver_feedback=feedback,
            technical_report={
                "driver_intent": DriverIntent.UNSUPPORTED.value,
                "message": message,
                "note": "unsupported request remains unsupported (no advice invented)",
            },
        )

    def _ambiguous(
        self,
        session: DriverSession,
        result: DriverIntentResult,
    ) -> DriverConversationResponse:
        feedback = (
            "I couldn't pin down exactly what you'd like. Please ask for a "
            "fuel prediction (with vehicle inputs), a knowledge question, an "
            "explanation, or a combination."
        )
        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent=DriverIntent.AMBIGUOUS.value,
            base_intent=result.base_intent.value if result.base_intent else None,
            conversation_state=ConversationState.CLARIFYING.value,
            selected_tools=[],
            executed_tools=[],
            validation_blocked=False,
            delivered=False,
            clarification_request=feedback,
            driver_feedback=feedback,
            technical_report={"note": "ambiguous request; asking a targeted clarifying question"},
        )

    def _actionable(
        self,
        session: DriverSession,
        result: DriverIntentResult,
        message: str,
    ) -> DriverConversationResponse:
        """Predict / explain / combined: finalize inputs, run selected tools."""
        if not result.is_ready:
            # Safety net: never run prediction tools while inputs are incomplete.
            return self._clarify(session, result, message, None)

        # This is the only path that delegates to the strict M2 validator.
        try:
            _, validated = session.finalize_for_prediction()
        except Exception as exc:  # noqa: BLE001
            # Defensive: if finalization fails, treat as clarification.
            fake = DriverIntentResult(
                intent=DriverIntent.CLARIFYING,
                confidence=0.6,
                reason=str(exc),
                backend=result.backend,
                missing_fields=list(session.pending_clarification),
                is_ready=False,
            )
            return self._clarify(session, fake, message, None)

        tools = list(result.selected_tools)
        results = self._execute_tools(tools, message, validated, session)
        return self._assemble(session, result, tools, results)

    def _knowledge_only(
        self,
        session: DriverSession,
        result: DriverIntentResult,
        message: str,
    ) -> DriverConversationResponse:
        tools = [TOOL_RETRIEVAL]
        results = self._execute_tools(tools, message, None, session)
        return self._assemble(session, result, tools, results)

    # -- tool execution --------------------------------------------------
    def _execute_tools(
        self,
        tool_names: list[str],
        message: str,
        validated_inputs: dict | None,
        session: DriverSession,
    ) -> list[dict]:
        results: list[dict] = []
        for name in tool_names:
            tool = self.tools[name]
            effector = _TOOL_EFFECTORS[name]
            try:
                if name == "knowledge_retrieval":
                    res = tool.retrieve(message)
                elif name == "fuel_rate_prediction":
                    res = tool.predict(validated_inputs or {})
                elif name == "prediction_explanation":
                    res = tool.explain(validated_inputs or {})
                else:  # pragma: no cover - defensive
                    res = getattr(tool, effector)(message)
                results.append({"tool": name, "status": "ok", "result": res})
            except Exception as exc:  # noqa: BLE001 - surfaces tool failures
                results.append(
                    {
                        "tool": name,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_msg": str(exc),
                    }
                )
        session.record_tool_result(results)
        return results

    # -- grounded response assembly --------------------------------------
    def _assemble(
        self,
        session: DriverSession,
        result: DriverIntentResult,
        executed_tools: list[str],
        results: list[dict],
    ) -> DriverConversationResponse:
        # Build a response object in the same shape the M2 generator expects.
        resp = AgentResponse()
        resp.intent = _m2_intent_for(result)
        resp.router = result
        resp.selected_tools = list(executed_tools)
        resp.tool_results = list(results)
        resp.limitations = [
            "Prediction is an estimate from a synthetic-trained model; not a "
            "validated real-world fuel measurement.",
        ]

        grounded = self.generator.generate(resp)

        # Run the existing validation battery.
        source_prediction = self._first_ok(results, TOOL_PREDICT)
        source_explanation = self._first_ok(results, TOOL_EXPLAIN)
        known_doc_ids = self._known_doc_ids(results)
        issues = validate_grounded_response(
            grounded,
            source_prediction=source_prediction,
            source_explanation=source_explanation,
            known_doc_ids=known_doc_ids,
        )
        summ = validation_summary(issues)

        # Error-severity failures block presentation as a successful answer.
        blocked = any((not i.passed) and i.severity == "error" for i in issues)

        session.mark_answered() if not blocked else None

        feedback = (
            grounded.driver_version
            if (not blocked and grounded.driver_version)
            else (
                "I could not produce a validated, grounded answer for this "
                "turn; the internal response failed safety checks and is not "
                "being presented to you."
            )
        )

        technical = dict(grounded.technical_version)
        technical["validation"] = summ
        technical["delivered"] = not blocked
        technical["invented_telemetry"] = False
        technical["shap_is_attribution_not_causality"] = True
        technical["synthetic_only"] = True

        return DriverConversationResponse(
            session_id=session.session_id,
            driver_intent=result.intent.value,
            base_intent=result.base_intent.value if result.base_intent else None,
            conversation_state=session.conversation_state.value,
            selected_tools=list(result.selected_tools),
            executed_tools=executed_tools,
            tool_results=results,
            grounded=grounded,
            validation_issues=issues,
            validation_summary=summ,
            validation_blocked=blocked,
            delivered=not blocked,
            driver_feedback=feedback,
            technical_report=technical,
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _first_ok(results: list[dict], tool: str) -> dict | None:
        for r in results:
            if r.get("tool") == tool and r.get("status") == "ok" and isinstance(r.get("result"), dict):
                return r["result"]
        return None

    @staticmethod
    def _known_doc_ids(results: list[dict]) -> list[str]:
        for r in results:
            if r.get("tool") == TOOL_RETRIEVAL and r.get("status") == "ok":
                passages = r.get("result", {}).get("passages") or []
                return [p.get("document_id") for p in passages if p.get("document_id")]
        return []
