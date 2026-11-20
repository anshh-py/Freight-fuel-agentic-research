"""M2.3 — focused tests for grounded response generation and validation.

Covers each supported response type, prediction/SHAP value preservation,
citation and source attribution, synthetic provenance disclosure, non-causal
language, insufficient-evidence handling, unsupported questions, tool failures,
contradictory outputs, and the driver-facing vs technical formats.

Real local integration: the generation tests exercise the existing M2.2 tools
with the saved M1.2 models and the synthetic corpus/vector store. No retraining
and no retrieval-parameter changes are made here.
"""

from __future__ import annotations

import copy

import pytest

from src.agent.intents import Intent
from src.agent.orchestrator import AgentOrchestrator
from src.agent.response_generator import GroundedResponse, ResponseGenerator
from src.agent.response_validation import validate_grounded_response
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool
from tests.test_tools import FULL_INPUTS, TELE_INPUTS


def _orchestrator(retrieval_store):
    tools = {
        "fuel_rate_prediction": FuelRatePredictionTool(),
        "knowledge_retrieval": KnowledgeRetrievalTool(store=retrieval_store),
        "prediction_explanation": PredictionExplanationTool(),
    }
    return AgentOrchestrator(tools=tools)


@pytest.fixture(scope="session")
def orch(retrieval_store):
    """Shared orchestrator over the real saved models + synthetic corpus store."""
    return _orchestrator(retrieval_store)


def _grounded(intent="prediction", **kw):
    g = GroundedResponse(
        intent=intent,
        detected_intent=intent,
        router_backend="deterministic",
        selected_tools=kw.get("selected_tools", []),
        prediction=kw.get("prediction"),
        explanation=kw.get("explanation"),
        evidence=kw.get("evidence", []),
        evidence_insufficient=kw.get("evidence_insufficient", False),
        tool_failures=kw.get("tool_failures", []),
        limitations=kw.get("limitations", []),
        driver_version=kw.get("driver_version", ""),
    )
    return g


# ---------------------------------------------------------------------------
# Response generation (real tools)
# ---------------------------------------------------------------------------

class TestResponseGeneration:
    def test_prediction_response_real(self, orch):
        resp = orch.handle("predict fuel consumption", FULL_INPUTS)
        out = ResponseGenerator(inputs=FULL_INPUTS).generate(resp)

        assert out.intent == Intent.PREDICT_FUEL.value
        assert "fuel_rate_prediction" in out.selected_tools
        # A. prediction preserved with provenance
        assert out.prediction["predicted_fuel_rate_lph"] == pytest.approx(
            resp.tool_results[0]["result"]["predicted_fuel_rate_lph"]
        )
        assert out.prediction["feature_set"] == "full_operational"
        assert out.prediction["model_name"] == "xgboost"
        assert out.prediction["synthetic_model_estimate"] is True
        # synthetic disclaimer present in driver version
        assert "SYNTHETIC" in out.driver_version.upper()
        assert "estimate" in out.driver_version.lower()
        assert "52.5773" in out.driver_version or "L/h" in out.driver_version

    def test_knowledge_response_real(self, orch):
        resp = orch.handle("why does road grade affect fuel consumption on trucks")
        out = ResponseGenerator().generate(resp)

        assert out.intent == Intent.KNOWLEDGE_QUERY.value
        assert out.evidence, "expected retrieved passages"
        doc_id = out.evidence[0]["document_id"]
        assert doc_id in out.driver_version          # source cited
        assert "SYNTHETIC EDUCATIONAL MATERIAL" in out.driver_version.upper()
        assert out.evidence[0]["source_reference"]
        assert out.evidence[0]["is_oem_manual"] is False

    def test_combined_prediction_explanation_real(self):
        pred = FuelRatePredictionTool().predict(FULL_INPUTS, "full_operational")
        expl = PredictionExplanationTool().explain(FULL_INPUTS, "full_operational")
        r = type("R", (), {})()
        r.intent = Intent.COMBINED_PREDICT_EXPLAIN
        r.selected_tools = ["fuel_rate_prediction", "prediction_explanation"]
        r.tool_results = [
            {"tool": "fuel_rate_prediction", "status": "ok", "result": pred},
            {"tool": "prediction_explanation", "status": "ok", "result": expl},
        ]
        r.router = None
        r.limitations = ["SHAP attribution is not causality."]

        out = ResponseGenerator(inputs=FULL_INPUTS).generate(r)
        assert out.explanation is not None
        pos = out.explanation["positive_contributions"]
        neg = out.explanation["negative_contributions"]
        assert pos and neg
        # SHAP values preserved (signed) in driver version
        top_pos = pos[0]["shap"]
        assert f"+{top_pos:.2f}" in out.driver_version
        # non-causal disclaimer present
        assert "NOT a causal explanation" in out.driver_version
        # feature values attached where available
        assert pos[0]["actual_input_value"] is not None

    def test_driver_vs_technical_formats(self):
        pred = FuelRatePredictionTool().predict(FULL_INPUTS, "full_operational")
        expl = PredictionExplanationTool().explain(FULL_INPUTS, "full_operational")
        r = type("R", (), {})()
        r.intent = Intent.COMBINED_PREDICT_EXPLAIN
        r.selected_tools = ["fuel_rate_prediction", "prediction_explanation"]
        r.tool_results = [
            {"tool": "fuel_rate_prediction", "status": "ok", "result": pred},
            {"tool": "prediction_explanation", "status": "ok", "result": expl},
        ]
        r.router = None
        r.limitations = []
        out = ResponseGenerator(inputs=FULL_INPUTS).generate(r)

        # driver is prose; technical is a structured dict
        assert isinstance(out.driver_version, str) and len(out.driver_version) > 0
        assert isinstance(out.technical_version, dict)
        assert out.technical_version["prediction"]["predicted_fuel_rate_lph"] == \
            out.prediction["predicted_fuel_rate_lph"]
        # technical lists columns + signed SHAP, and provenance claims
        column = out.explanation["positive_contributions"][0]["column"]
        assert column in str(out.technical_version["explanation"])
        claims = out.technical_version["claims_and_provenance"]
        assert claims["shap_is_attribution_not_causality"] is True
        assert claims["corpus_is_synthetic_educational_material"] is True
        assert claims["no_invented_driver_or_truck_identity"] is True

    def test_insufficient_evidence_stated(self, retrieval_store):
        tools = {
            "fuel_rate_prediction": FuelRatePredictionTool(),
            "knowledge_retrieval": KnowledgeRetrievalTool(
                store=retrieval_store, min_score=0.99),
            "prediction_explanation": PredictionExplanationTool(),
        }
        orch = AgentOrchestrator(tools=tools)
        resp = orch.handle("why does road grade affect fuel consumption")
        out = ResponseGenerator().generate(resp)

        assert out.evidence_insufficient is True
        assert "no sufficiently supportive passage" in out.driver_version
        # No evidence-backed recommendations should be offered on weak evidence
        assert out.recommendations_evidence_backed == []
        assert out.recommendations_unsupported

    def test_tool_failure_disclosed(self, orch):
        bad = copy.deepcopy(FULL_INPUTS)
        bad["speed_kph"] = -40.0
        resp = orch.handle("predict fuel consumption", bad)
        out = ResponseGenerator(inputs=bad).generate(resp)

        assert "fuel_rate_prediction" in out.tool_failures
        assert "failed" in out.driver_version.lower()
        # must not fabricate a value
        assert out.prediction is None

    def test_unsupported_query_declined(self, orch):
        resp = orch.handle("what is the official manufacturer torque limit for this truck")
        out = ResponseGenerator().generate(resp)
        assert out.intent == Intent.UNSUPPORTED.value
        assert "can't answer" in out.driver_version.lower()
        assert out.selected_tools == []

    def test_ambiguous_query(self, orch):
        resp = orch.handle("banana split")
        out = ResponseGenerator().generate(resp)
        assert out.intent == Intent.AMBIGUOUS.value
        assert "pin down" in out.driver_version.lower()


# ---------------------------------------------------------------------------
# Response validation (deterministic checks)
# ---------------------------------------------------------------------------

class TestResponseValidation:
    def test_causal_claim_flagged(self):
        g = _grounded(
            intent="prediction", selected_tools=["prediction_explanation"],
            driver_version="The torque caused the higher fuel rate; reducing torque "
                           "will reduce fuel consumption by 20 L/h.")
        issues = validate_grounded_response(g)
        causal = next(i for i in issues if i.check_id == "causal_language")
        assert causal.passed is False

    def test_protective_disclaimer_not_flagged(self):
        # The generator's own anti-causal disclaimer must never be flagged.
        g = _grounded(
            intent="prediction", selected_tools=["prediction_explanation"],
            driver_version="Estimated fuel rate: 52.5773 L/h. "
                           "SHAP attribution describes how the model composes its "
                           "estimate. It is NOT a causal explanation: it does not "
                           "prove that changing a feature changes real fuel "
                           "consumption.")
        issues = validate_grounded_response(g)
        causal = next(i for i in issues if i.check_id == "causal_language")
        assert causal.passed is True

    def test_negated_frame_does_not_prove_not_flagged(self):
        # "does not prove" is protective phrasing, not an assertion frame.
        g = _grounded(
            intent="prediction", selected_tools=["prediction_explanation"],
            driver_version="SHAP does not prove that changing a feature changes "
                           "real-world fuel consumption.")
        issues = validate_grounded_response(g)
        causal = next(i for i in issues if i.check_id == "causal_language")
        assert causal.passed is True

    def test_nonnegated_frame_proves_flagged(self):
        # A non-negated assertion frame is still a prohibited causal claim.
        g = _grounded(
            intent="prediction", selected_tools=["prediction_explanation"],
            driver_version="SHAP proves this feature increases fuel consumption.")
        issues = validate_grounded_response(g)
        causal = next(i for i in issues if i.check_id == "causal_language")
        assert causal.passed is False

    def test_oem_mislabeling_flagged(self):
        g = _grounded(
            intent="knowledge", selected_tools=["knowledge_retrieval"],
            evidence=[{"document_id": "doc_001", "title": "T", "score": 0.72}],
            driver_version="Per the manufacturer specification in doc_001, grade "
                           "raises fuel use.")
        issues = validate_grounded_response(g, known_doc_ids=["doc_001"])
        oem = next(i for i in issues if i.check_id == "oem_mislabeling")
        assert oem.passed is False

    def test_synthetic_disclaimer_missing_flagged(self):
        g = _grounded(
            intent="prediction",
            prediction={"predicted_fuel_rate_lph": 52.5773,
                        "feature_set": "full_operational"},
            driver_version="Estimated fuel rate: 52.5773 L/h.")
        issues = validate_grounded_response(
            g, source_prediction={"predicted_fuel_rate_lph": 52.5773})
        chk = next(i for i in issues if i.check_id == "synthetic_disclaimer")
        assert chk.passed is False

    def test_fabricated_source_flagged(self):
        g = _grounded(
            intent="knowledge", selected_tools=["knowledge_retrieval"],
            evidence=[{"document_id": "doc_001", "title": "T", "score": 0.72}],
            driver_version="Per [doc_999], grades raise fuel use.")
        issues = validate_grounded_response(g, known_doc_ids=["doc_001"])
        src = next(i for i in issues if i.check_id == "source_attribution")
        assert src.passed is False

    def test_unsupported_spec_fabrication_flagged(self):
        g = _grounded(intent="unsupported",
                      driver_version="The official torque limit is 89 L/h.")
        issues = validate_grounded_response(g)
        chk = next(i for i in issues if i.check_id == "unsupported_not_answered")
        assert chk.passed is False

    def test_insufficient_not_stated_flagged(self):
        g = _grounded(intent="knowledge", evidence=[], evidence_insufficient=True,
                      driver_version="Here is the answer: grades double fuel use.")
        issues = validate_grounded_response(g)
        chk = next(i for i in issues if i.check_id == "insufficient_evidence_stated")
        assert chk.passed is False

    def test_failure_presented_as_success_flagged(self):
        g = _grounded(intent="prediction", selected_tools=["fuel_rate_prediction"],
                      tool_failures=["fuel_rate_prediction"],
                      prediction={"predicted_fuel_rate_lph": 52.5773},
                      driver_version="Estimated fuel rate: 52.5773 L/h.")
        issues = validate_grounded_response(g)
        chk = next(i for i in issues if i.check_id == "failure_not_presented_as_success")
        assert chk.passed is False

    def test_prediction_value_dropped_flagged(self):
        g = _grounded(
            intent="prediction",
            prediction={"predicted_fuel_rate_lph": 52.5773},
            driver_version="We estimated fuel consumption but hid the number.")
        issues = validate_grounded_response(
            g, source_prediction={"predicted_fuel_rate_lph": 52.5773})
        chk = next(i for i in issues if i.check_id == "prediction_value_preserved")
        assert chk.passed is False

    def test_shap_value_dropped_flagged(self):
        g = _grounded(
            intent="prediction",
            explanation={"positive_contributions": [{"shap": 20.1063}]},
            driver_version="The explanation omitted the top SHAP magnitude.")
        issues = validate_grounded_response(
            g, source_explanation={"most_influential_features": {
                "top_positive_shap": [{"shap": 20.1063}]}})
        chk = next(i for i in issues if i.check_id == "shap_value_preserved")
        assert chk.passed is False

    def test_contradictory_passages_with_provenance(self):
        g = _grounded(
            intent="knowledge",
            evidence=[
                {"document_id": "doc_001", "title": "A", "score": 0.72},
                {"document_id": "doc_002", "title": "B", "score": 0.60},
            ],
            driver_version="Relevant knowledge (retrieved passages):\n"
                           "  - [doc_001] A (similarity 0.720).\n"
                           "  - [doc_002] B (similarity 0.600).\n"
                           "Retrieved passages are SYNTHETIC educational material "
                           "(not OEM or manufacturer documentation). Retrieval "
                           "similarity is not proof of factual correctness.")
        issues = validate_grounded_response(g, known_doc_ids=["doc_001", "doc_002"])
        for cid in ("source_attribution", "synthetic_corpus_notice", "oem_mislabeling"):
            chk = next(i for i in issues if i.check_id == cid)
            assert chk.passed is True, cid
