"""M4.5 — focused tests for the end-to-end driver interaction demo.

Separates:
  * unit tests of the evaluation layer (stub tools, no model artifacts), and
  * an integration test that exercises the real saved XGBoost prediction tool
    and the real M4.5 demo report builder.

The integration test is marked ``integration`` and is the only test that
touches real artifacts; all other tests run with stub tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.driver_conversation import DriverConversationService
from src.agent.driver_session import DriverSession
from src.evaluation.m4_5_driver_interaction import (
    DEFAULT_FEATURE_SET,
    FULL_INPUTS,
    PARTIAL_INPUTS,
    TurnRecord,
    build_report,
    run_demo,
    _evaluate_scenario,
)


# ---------------------------------------------------------------------------
# Stub tools (mirror the real tool result shapes)
# ---------------------------------------------------------------------------

class StubPredictTool:
    def predict(self, inputs, feature_set=None):
        return {
            "status": "ok",
            "predicted_fuel_rate_lph": 13.25,
            "units": "L/h",
            "feature_set": feature_set or DEFAULT_FEATURE_SET,
            "model": {"name": "xgboost", "artifact": "stub"},
            "preprocessor": {"artifact": "stub", "transformed_columns": []},
            "input_fields_used": dict(inputs),
            "provenance": {"synthetic_training": True, "model_is_estimate": True},
        }


class StubExplainTool:
    def explain(self, inputs, feature_set=None, top_features=None):
        return {
            "status": "ok",
            "predicted_fuel_rate_lph": 13.25,
            "feature_set": feature_set or DEFAULT_FEATURE_SET,
            "model": {"name": "xgboost", "artifact": "stub"},
            "explanation_kind": "instance_level_shap",
            "summary_statistic": "per-instance SHAP",
            "top_features": top_features or 5,
            "expected_value": 11.0,
            "reconstruction": {"pass": True, "max_abs_deviation": 0.0, "tolerance": 0.01},
            "most_influential_features": {
                "top_positive_shap": [
                    {"readable_label": "Payload", "column": "payload_kg", "shap": 1.6},
                ],
                "top_negative_shap": [
                    {"readable_label": "Speed", "column": "speed_kph", "shap": -0.4},
                ],
            },
            "limitations": {"shap_is_not_causal": True},
            "provenance": {"synthetic_training": True},
        }


class StubRetrieveTool:
    def retrieve(self, query, k=None, min_score=None):
        return {
            "status": "ok",
            "query": query,
            "n_results": 1,
            "passages": [
                {
                    "rank": 1,
                    "chunk_id": "c1",
                    "document_id": "doc_001",
                    "title": "Driving tips",
                    "topic": "driving",
                    "source_type": "synthetic",
                    "source_reference": "synthetic-corpus",
                    "is_oem_manual": False,
                    "score": 0.82,
                    "content_excerpt": "Shift gears smoothly and avoid hard acceleration.",
                },
            ],
            "evidence_sufficiency": {
                "insufficient": False,
                "reason": "top passage meets similarity floor",
                "min_score_floor": 0.45,
            },
            "provenance": {"corpus_synthetic": True},
        }


def _stub_service():
    return DriverConversationService(
        tools={
            "fuel_rate_prediction": StubPredictTool(),
            "prediction_explanation": StubExplainTool(),
            "knowledge_retrieval": StubRetrieveTool(),
        },
        default_feature_set=DEFAULT_FEATURE_SET,
    )


# ---------------------------------------------------------------------------
# Unit tests of the evaluation layer (stub tools)
# ---------------------------------------------------------------------------

def test_run_demo_with_stubs_produces_full_report():
    body = run_demo(service=_stub_service())
    assert body["experiment"] == "M4.5_end_to_end_driver_interaction_demo"
    assert body["feature_set"] == DEFAULT_FEATURE_SET
    assert len(body["scenarios"]) == 3
    ev = body["evaluation"]
    assert ev["scenarios_total"] == 3
    assert ev["all_passed"] is True
    assert ev["scenarios_completed"] == 3


def test_high_fuel_scenario_has_expected_turn_count():
    body = run_demo(service=_stub_service())
    sc = body["scenarios"][0]
    assert sc["name"] == "high_fuel_report_full_journey"
    assert len(sc["turns"]) == 5
    intents = [t["intent"] for t in sc["turns"]]
    assert intents == [
        "clarifying",
        "clarifying",
        "predict_fuel",
        "explain_prediction",
        "knowledge_query",
    ]


def test_clarification_turns_execute_no_tools():
    body = run_demo(service=_stub_service())
    sc = body["scenarios"][0]
    for t in sc["turns"][:2]:
        assert t["executed_tools"] == []
        assert t["intent"] == "clarifying"
        assert t["missing_fields"]  # must report what's missing


def test_unsupported_and_ambiguous_are_safe():
    body = run_demo(service=_stub_service())
    by_name = {s["name"]: s for s in body["scenarios"]}
    uns = by_name["unsupported_request"]["turns"][0]
    amb = by_name["ambiguous_request"]["turns"][0]
    assert uns["intent"] == "unsupported"
    assert uns["executed_tools"] == []
    assert uns["delivered"] is True  # safe scope-decline is delivered
    assert amb["intent"] == "ambiguous"
    assert amb["executed_tools"] == []
    assert amb["delivered"] is False  # asks a targeted question


def test_turn_record_checks():
    rec = TurnRecord(
        scenario="s",
        turn=1,
        message="m",
        expected_intent="predict_fuel",
        expected_tools=["fuel_rate_prediction"],
        intent="predict_fuel",
        executed_tools=["fuel_rate_prediction"],
        delivered=True,
        validation_all_passed=True,
        prediction_ok=True,
        explanation_recon_pass=None,
        retrieval_insufficient=None,
    )
    checks = rec.checks()
    assert checks["intent_correct"] is True
    assert checks["tool_selection_correct"] is True
    assert checks["prediction_success"] is True
    assert checks["delivery_validated"] is True


def test_scenario_evaluation_aggregates():
    rec = TurnRecord(
        scenario="s",
        turn=1,
        message="m",
        expected_intent="clarifying",
        expected_tools=[],
        intent="clarifying",
        executed_tools=[],
        delivered=False,
        missing_fields=["payload_kg"],
    )
    ev = _evaluate_scenario("s", [rec])
    assert ev.total_checks == len(rec.checks())
    assert ev.passed_checks == ev.total_checks
    assert ev.completion_ok is True


def test_build_report_writes_files():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        report = build_report(service=_stub_service(), report_dir=Path(tmp))
        assert (Path(tmp) / "driver_interaction_report.json").exists()
        assert (Path(tmp) / "driver_interaction_summary.md").exists()
        assert (Path(tmp) / "plots" / "driver_interaction_scenarios.png").exists()
    assert report["verification"]["real_artifacts_used"] is True
    assert report["verification"]["no_fabricated_telemetry"] is True
    assert not report["is_llm"]


# ---------------------------------------------------------------------------
# Integration test (real saved model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_prediction_tool_runs_through_demo():
    """Exercise the real saved XGBoost prediction artifact via the demo flow."""
    # Use the real service (loads the saved M1.2 model + preprocessor).
    service = DriverConversationService(default_feature_set=DEFAULT_FEATURE_SET)
    session = DriverSession(feature_set=DEFAULT_FEATURE_SET)

    # A full-input prediction turn should run the **real** model path.
    resp = service.handle(session, "My truck is consuming too much diesel.", FULL_INPUTS)
    assert resp.driver_intent == "predict_fuel"
    assert resp.executed_tools == ["fuel_rate_prediction"]
    assert resp.delivered is True
    # The real tool returns a finite predicted fuel rate.
    pred_results = [r for r in resp.tool_results if r["tool"] == "fuel_rate_prediction"]
    assert pred_results and pred_results[0]["status"] == "ok"
    value = pred_results[0]["result"]["predicted_fuel_rate_lph"]
    assert isinstance(value, (int, float)) and value > 0
