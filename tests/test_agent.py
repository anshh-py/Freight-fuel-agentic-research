"""Focused tests for the M2.2 agent router and orchestrator.

Covers intent routing for representative queries (prediction, knowledge,
explanation, combined, unsupported, ambiguous), tool selection, invalid-input
and tool-failure handling, insufficient-evidence warnings, and end-to-end runs
of a prediction query, a knowledge query, and a combined query using the real
saved models and real corpus/vector store.
"""

from __future__ import annotations

import copy
import pytest

from src.agent.intents import (
    DeterministicIntentRouter,
    Intent,
)
from src.agent.orchestrator import AgentOrchestrator
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool

from tests.test_tools import FULL_INPUTS


# ---------------------------------------------------------------------------
# Routing (deterministic baseline)
# ---------------------------------------------------------------------------

class TestRouter:
    def test_router_prediction_intent(self):
        r = DeterministicIntentRouter().route("predict fuel consumption", FULL_INPUTS)
        assert r.intent == Intent.PREDICT_FUEL
        assert r.backend == "deterministic"

    def test_router_knowledge_intent(self):
        r = DeterministicIntentRouter().route("why does engine load affect fuel consumption")
        assert r.intent == Intent.KNOWLEDGE_QUERY

    def test_router_explain_intent(self):
        r = DeterministicIntentRouter().route("explain this prediction", FULL_INPUTS)
        assert r.intent == Intent.EXPLAIN_PREDICTION

    def test_router_combined_knowledge_prediction(self):
        r = DeterministicIntentRouter().route(
            "predict fuel consumption and explain why engine load affects it",
            FULL_INPUTS,
        )
        assert r.intent == Intent.COMBINED_KNOWLEDGE_PREDICT

    def test_router_combined_predict_explain(self):
        r = DeterministicIntentRouter().route("predict and explain the fuel consumption")
        assert r.intent == Intent.COMBINED_PREDICT_EXPLAIN

    def test_router_unsupported_out_of_scope(self):
        r = DeterministicIntentRouter().route("what is the official manufacturer torque limit")
        assert r.intent == Intent.UNSUPPORTED

    def test_router_ambiguous(self):
        r = DeterministicIntentRouter().route("banana split")
        assert r.intent == Intent.AMBIGUOUS


# ---------------------------------------------------------------------------
# Orchestrator end-to-end (real models + corpus)
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def _orchestrator(self, retrieval_store, knowledge_min_score=None):
        tools = {
            "fuel_rate_prediction": FuelRatePredictionTool(),
            "knowledge_retrieval": KnowledgeRetrievalTool(
                store=retrieval_store, min_score=knowledge_min_score
            ),
            "prediction_explanation": PredictionExplanationTool(),
        }
        return AgentOrchestrator(tools=tools)

    def test_e2e_prediction(self, retrieval_store):
        orch = self._orchestrator(retrieval_store)
        resp = orch.handle("predict fuel consumption", FULL_INPUTS)
        assert resp.intent == Intent.PREDICT_FUEL
        assert resp.selected_tools == ["fuel_rate_prediction"]
        assert resp.tool_results[0]["status"] == "ok"
        assert "L/h" in resp.final_response or "L/h" in repr(resp.final_response)
        assert resp.tool_results[0]["result"]["predicted_fuel_rate_lph"] >= 0.0
        assert any("deterministic" in w for w in resp.limitations)

    def test_e2e_knowledge(self, retrieval_store):
        orch = self._orchestrator(retrieval_store)
        resp = orch.handle("why does road grade affect fuel consumption on trucks")
        assert resp.intent == Intent.KNOWLEDGE_QUERY
        assert resp.selected_tools == ["knowledge_retrieval"]
        assert resp.tool_results[0]["status"] == "ok"
        assert resp.tool_results[0]["result"]["n_results"] > 0

    def test_e2e_combined_knowledge_prediction(self, retrieval_store):
        orch = self._orchestrator(retrieval_store)
        resp = orch.handle(
            "predict fuel consumption and explain why engine load affects it",
            FULL_INPUTS,
        )
        assert resp.intent == Intent.COMBINED_KNOWLEDGE_PREDICT
        assert resp.selected_tools == ["knowledge_retrieval", "fuel_rate_prediction"]
        assert all(r["status"] == "ok" for r in resp.tool_results)
        assert resp.supporting_evidence

    def test_invalid_inputs_failure_recorded(self, retrieval_store):
        orch = self._orchestrator(retrieval_store)
        bad = copy.deepcopy(FULL_INPUTS)
        bad["speed_kph"] = -40.0
        resp = orch.handle("predict fuel consumption", bad)
        assert resp.intent == Intent.PREDICT_FUEL
        assert resp.tool_results[0]["status"] == "error"
        assert resp.tool_results[0]["error_type"] == "input"
        assert any("failed" in w for w in resp.limitations)
        # The agent still returns a structured response (no crash).
        assert resp.final_response

    def test_unsupported_selects_no_tools(self, retrieval_store):
        orch = self._orchestrator(retrieval_store)
        resp = orch.handle("what is the official manufacturer torque limit")
        assert resp.intent == Intent.UNSUPPORTED
        assert resp.selected_tools == []
        assert resp.tool_results == []

    def test_insufficient_evidence_warning(self, retrieval_store):
        orch = self._orchestrator(retrieval_store, knowledge_min_score=2.0)
        resp = orch.handle("why does road grade affect fuel consumption")
        assert resp.intent == Intent.KNOWLEDGE_QUERY
        assert resp.tool_results[0]["status"] == "ok"
        assert resp.tool_results[0]["result"]["evidence_sufficiency"]["insufficient"] is True
        assert any("insufficient" in w for w in resp.limitations)
