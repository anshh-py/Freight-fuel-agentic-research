"""Intent definition and routing for the M2.2 agent.

The orchestrator receives a user query plus optional structured vehicle inputs
and must decide *which* tool(s) to call. This module defines the intent enum,
the router interface, and a deterministic rule-based router that is the
current baseline.

The deterministic router is intentionally modest: it matches keyword sets and
returns an intent with a human-readable reason. It is NOT LLM reasoning and is
NOT claimed to be autonomous agent intelligence. It exists so the orchestrator
keeps a stable intent interface that a future LLM-based router can implement
without rewriting the tools (Phase 4 boundary).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

# Canonical tool registry names (must match the orchestrator's registry keys).
TOOL_PREDICT = "fuel_rate_prediction"
TOOL_RETRIEVAL = "knowledge_retrieval"
TOOL_EXPLAIN = "prediction_explanation"


class Intent(Enum):
    PREDICT_FUEL = "fuel_prediction"
    KNOWLEDGE_QUERY = "knowledge_query"
    EXPLAIN_PREDICTION = "prediction_explanation"
    COMBINED_PREDICT_EXPLAIN = "combined_prediction_explanation"
    COMBINED_KNOWLEDGE_PREDICT = "combined_knowledge_prediction"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


# Intent -> tools to execute (empty for unsupported/ambiguous).
INTENT_TOOLS: dict[Intent, list[str]] = {
    Intent.PREDICT_FUEL: [TOOL_PREDICT],
    Intent.KNOWLEDGE_QUERY: [TOOL_RETRIEVAL],
    Intent.EXPLAIN_PREDICTION: [TOOL_EXPLAIN],
    Intent.COMBINED_PREDICT_EXPLAIN: [TOOL_PREDICT, TOOL_EXPLAIN],
    Intent.COMBINED_KNOWLEDGE_PREDICT: [TOOL_RETRIEVAL, TOOL_PREDICT],
    Intent.UNSUPPORTED: [],
    Intent.AMBIGUOUS: [],
}


class IntentResult:
    """Outcome of routing one query."""

    __slots__ = ("intent", "confidence", "reason", "backend")

    def __init__(
        self,
        intent: Intent,
        confidence: float,
        reason: str,
        backend: str = "deterministic",
    ) -> None:
        self.intent = intent
        self.confidence = confidence
        self.reason = reason
        self.backend = backend

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "backend": self.backend,
        }


class IntentRouter(Protocol):
    """Interface every router (deterministic or LLM-based) must satisfy."""

    def route(self, query: str, structured_inputs: dict | None) -> IntentResult:
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Keyword banks (deterministic baseline)
# ---------------------------------------------------------------------------

_PREDICT_WORDS = (
    "fuel rate", "consumption", "fuel demand", "how much fuel", "predict",
    "estimate", "gallons", "mpg", "l/h", "fuel consumption", "rate",
)
_KNOWLEDGE_WORDS = (
    "why", "how does", "affect", "relates", "relationship", "engine",
    "cause", "mechanism", "difference between", "vs", "compare", "what role",
    "what causes", "behind", "torque", "rpm", "grade", "transmission", "gear",
    "how it", "in what way",
)
_EXPLAIN_WORDS = (
    "explain", "why did", "contribute", "which factor", "factor", "importance",
    "important", "influence", "feature importance", "what drives", "why is",
    "explanation",
)
_OUT_OF_SCOPE_WORDS = (
    "actual truck", "real-world validation", "real world", "official",
    "oem spec", "manufacturer limit", "warranty", "maintenance limit", "recall",
    "compliance", "billing",
)
_UNSUPPORTED_WORDS = (
    "hello", "hi ", "thanks", "who are you", "good morning", "good afternoon",
    "weather", "joke", "price of", "buy", "song", "write a poem",
)


def _count_words(text: str, bank: tuple[str, ...]) -> int:
    return sum(1 for w in bank if w in text)


class DeterministicIntentRouter:
    """Rule-based intent routing baseline (Phase 4 deterministic fallback)."""

    BACKEND = "deterministic"

    def route(
        self,
        query: str,
        structured_inputs: dict | None = None,
    ) -> IntentResult:
        q = (query or "").strip().lower()
        if not q:
            return IntentResult(
                Intent.AMBIGUOUS, 0.3,
                "empty query; cannot determine intent", self.BACKEND,
            )

        has_inputs = bool(structured_inputs)

        # Out-of-scope (safety): never answer manufacturer/real-world questions.
        if _count_words(q, _OUT_OF_SCOPE_WORDS) > 0:
            return IntentResult(
                Intent.UNSUPPORTED, 0.9,
                "query asks for out-of-scope real-world/manufacturer information "
                "that the synthetic project cannot provide",
                self.BACKEND,
            )

        p = _count_words(q, _PREDICT_WORDS)
        kc = _count_words(q, _KNOWLEDGE_WORDS)
        e = _count_words(q, _EXPLAIN_WORDS)
        u = _count_words(q, _UNSUPPORTED_WORDS)

        if has_inputs:
            # A knowledge-style question alongside inputs (e.g. "why engine load
            # affects consumption") is a combined knowledge+prediction request.
            if kc > 0:
                return IntentResult(
                    Intent.COMBINED_KNOWLEDGE_PREDICT, 0.8,
                    "structured vehicle inputs plus a knowledge-style question",
                    self.BACKEND,
                )
            if e > 0:
                return IntentResult(
                    Intent.EXPLAIN_PREDICTION, 0.85,
                    "structured inputs present and explanation requested",
                    self.BACKEND,
                )
            if p > 0:
                return IntentResult(
                    Intent.PREDICT_FUEL, 0.85,
                    "structured vehicle inputs with a prediction request",
                    self.BACKEND,
                )
            return IntentResult(
                Intent.AMBIGUOUS, 0.4,
                "inputs present but no clear prediction/knowledge/explanation intent",
                self.BACKEND,
            )

        # No structured inputs.
        if kc > 0:
            return IntentResult(
                Intent.KNOWLEDGE_QUERY, 0.8,
                "conceptual knowledge question without vehicle inputs",
                self.BACKEND,
            )
        if p > 0 and e > 0:
            return IntentResult(
                Intent.COMBINED_PREDICT_EXPLAIN, 0.75,
                "request for prediction and explanation",
                self.BACKEND,
            )
        if p > 0:
            return IntentResult(
                Intent.PREDICT_FUEL, 0.75,
                "prediction request but no vehicle inputs supplied",
                self.BACKEND,
            )
        if e > 0:
            return IntentResult(
                Intent.EXPLAIN_PREDICTION, 0.6,
                "explanation requested but no vehicle inputs supplied",
                self.BACKEND,
            )
        if u > 0:
            return IntentResult(
                Intent.UNSUPPORTED, 0.8,
                "query appears off-topic / unsupported",
                self.BACKEND,
            )
        return IntentResult(
            Intent.AMBIGUOUS, 0.4,
            "no decisive intent signal", self.BACKEND,
        )
