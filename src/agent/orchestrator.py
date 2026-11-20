"""M2.2 — modular agent orchestrator.

The orchestrator:
  1. routes the user query to an intent (Phase 4 routing boundary),
  2. selects the tool(s) for that intent from a registry,
  3. executes each selected tool through its explicit callable interface,
  4. handles invalid arguments / tool failures / missing evidence,
  5. returns a structured response (schema below) with safety provenance.

The orchestrator is independent of any LLM provider. Routing is currently the
deterministic baseline (``DeterministicIntentRouter``); a future LLM-based
router can implement ``IntentRouter`` without touching the tools.
"""

from __future__ import annotations

from typing import Any

from src.agent.intents import (
    INTENT_TOOLS,
    Intent,
    IntentResult,
    IntentRouter,
    DeterministicIntentRouter,
)
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool
from src.tools.validation import ToolInputError


class AgentResponse:
    """Structured response schema for the orchestrator."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_intent": self.intent.value,
            "router": self.router.to_dict() if self.router else None,
            "selected_tools": list(self.selected_tools),
            "tool_execution_results": self.tool_results,
            "supporting_evidence": self.supporting_evidence,
            "limitations_and_warnings": self.limitations,
            "final_response": self.final_response,
            "agent_mode": self.agent_mode,
        }

    def __init__(self) -> None:
        self.intent: Intent = Intent.AMBIGUOUS
        self.router: IntentResult | None = None
        self.selected_tools: list[str] = []
        self.tool_results: list[dict] = []
        self.supporting_evidence: list[dict] = []
        self.limitations: list[str] = []
        self.final_response = ""
        self.agent_mode = "deterministic_routing__not_llm"


class AgentOrchestrator:
    """Routes and executes tools for a user query."""

    #: Explicit tool registry name -> tool instance.
    TOOL_NAMES = {
        "fuel_rate_prediction": "FuelRatePredictionTool",
        "knowledge_retrieval": "KnowledgeRetrievalTool",
        "prediction_explanation": "PredictionExplanationTool",
    }

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        router: IntentRouter | None = None,
    ) -> None:
        if tools is None:
            tools = {
                "fuel_rate_prediction": FuelRatePredictionTool(),
                "knowledge_retrieval": KnowledgeRetrievalTool(),
                "prediction_explanation": PredictionExplanationTool(),
            }
        self.tools = tools
        self.router = router or DeterministicIntentRouter()

    # -- tool execution ---------------------------------------------------
    def _execute_tool(
        self, name: str, query: str, inputs: dict | None, resp: AgentResponse
    ) -> dict:
        tool = self.tools[name]
        if name == "fuel_rate_prediction":
            result = tool.predict(inputs or {})
            kind = "prediction"
        elif name == "knowledge_retrieval":
            result = tool.retrieve(query)
            kind = "retrieval"
        elif name == "prediction_explanation":
            result = tool.explain(inputs or {})
            kind = "explanation"
        else:  # pragma: no cover - defensive
            raise KeyError(f"unknown tool {name!r}")

        resp.supporting_evidence.append(
            {
                "tool": name,
                "kind": kind,
                "summary": self._evidence_summary(name, result),
            }
        )
        return {"tool": name, "status": "ok", "result": result}

    def _evidence_summary(self, name: str, result: dict) -> dict:
        if name == "knowledge_retrieval":
            return {
                "n_passages": result.get("n_results"),
                "insufficient": result.get("evidence_sufficiency", {}).get("insufficient"),
                "top_document_id": (
                    result["passages"][0]["document_id"]
                    if result.get("passages") else None
                ),
            }
        if name in ("fuel_rate_prediction", "prediction_explanation"):
            return {
                "feature_set": result.get("feature_set"),
                "predicted_fuel_rate_lph": result.get("predicted_fuel_rate_lph"),
            }
        return {}

    # -- response assembly ------------------------------------------------
    def _warn_from_results(self, intent: Intent, resp: AgentResponse) -> None:
        resp.limitations.append(
            "Routing is a deterministic rule-based baseline, not LLM reasoning "
            "or autonomous agent intelligence."
        )
        for r in resp.tool_results:
            if r["status"] != "ok":
                resp.limitations.append(
                    f"tool {r['tool']} failed: {r['error_msg']}"
                )
            if r["tool"] == "fuel_rate_prediction":
                resp.limitations.append(
                    "Prediction is an estimate from a synthetic-trained model; "
                    "not a validated real-world fuel measurement."
                )
            if r["tool"] == "prediction_explanation":
                resp.limitations.append(
                    "SHAP attribution is not causality."
                )
            if r["tool"] == "knowledge_retrieval" and r["status"] == "ok":
                res = r["result"]
                if res.get("evidence_sufficiency", {}).get("insufficient"):
                    resp.limitations.append(
                        "Retrieved evidence was insufficient for this query."
                    )
                resp.limitations.append(
                    "Retrieval similarity is not proof of factual correctness; "
                    "corpus is synthetic educational material, not OEM manuals."
                )

    def _final_response(self, intent: Intent, resp: AgentResponse) -> str:
        if intent == Intent.UNSUPPORTED:
            return (
                "I can't answer that within this project's scope. The agent "
                "works on synthetic fuel-rate prediction, retrieval over a "
                "synthetic knowledge corpus, and model-explanation only; it "
                "does not provide real-world, manufacturer, or maintenance "
                "information."
            )
        if intent == Intent.AMBIGUOUS:
            return (
                "I couldn't pin down a single intent. Please ask for a fuel "
                "prediction (with vehicle inputs), a knowledge question, an "
                "explanation, or a combination."
            )

        pieces: list[str] = []

        def _ok(name: str):
            return next(
                (r for r in resp.tool_results if r["tool"] == name and r["status"] == "ok"),
                None,
            )

        pred = _ok("fuel_rate_prediction")
        if pred:
            res = pred["result"]
            pieces.append(
                f"Predicted fuel rate: {res['predicted_fuel_rate_lph']} L/h "
                f"(feature set: {res['feature_set']}, model: "
                f"{res['model']['name']}). This is a synthetic-model estimate."
            )

        expl = _ok("prediction_explanation")
        if expl:
            res = expl["result"]
            top = res["most_influential_features"]["top_positive_shap"]
            neg = res["most_influential_features"]["top_negative_shap"]
            top_s = ", ".join(
                f"{t['readable_label']} (+{t['shap']:.2f})" for t in top
            )
            neg_s = ", ".join(
                f"{t['readable_label']} ({t['shap']:.2f})" for t in neg
            )
            pieces.append(
                f"Instance-level SHAP for the predicted {res['predicted_fuel_rate_lph']} "
                f"L/h: main positive drivers: {top_s}; main negative drivers: {neg_s}. "
                "SHAP is attribution for the model, not causality."
            )

        ret = _ok("knowledge_retrieval")
        if ret:
            res = ret["result"]
            insuff = res["evidence_sufficiency"]["insufficient"]
            if insuff:
                pieces.append(
                    "No sufficiently supportive corpus passage was retrieved for this "
                    "question (the synthetic corpus may lack the information)."
                )
            else:
                top = res["passages"][0]
                pieces.append(
                    f"Most relevant retrieved passage: "
                    f"[{top['document_id']}] {top['title']} (topic: "
                    f"{top['topic']}, similarity {top['score']:.3f}, synthetic source)."
                )

        return " ".join(pieces) if pieces else "No tool output produced."

    # -- main entry point -------------------------------------------------
    def handle(
        self,
        query: str,
        structured_inputs: dict | None = None,
    ) -> AgentResponse:
        resp = AgentResponse()
        intent_result = self.router.route(query, structured_inputs)
        resp.intent = intent_result.intent
        resp.router = intent_result
        resp.selected_tools = list(INTENT_TOOLS[intent_result.intent])

        for name in resp.selected_tools:
            try:
                resp.tool_results.append(
                    self._execute_tool(name, query, structured_inputs, resp)
                )
            except (ToolInputError, KeyError) as exc:
                resp.tool_results.append(
                    {"tool": name, "status": "error", "error_type": "input",
                     "error_msg": str(exc)}
                )
            except Exception as exc:  # noqa: BLE001 - surfaces tool failures
                resp.tool_results.append(
                    {"tool": name, "status": "error", "error_type": "runtime",
                     "error_msg": f"{type(exc).__name__}: {exc}"}
                )

        self._warn_from_results(intent_result.intent, resp)
        resp.final_response = self._final_response(intent_result.intent, resp)
        return resp
