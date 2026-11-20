"""Focused tests for M5.4 driver-feedback usefulness & safety evaluation.

Covers:
1. Rubric scoring and aggregation.
2. Negative-control safety detection.
3. Scenario-level evaluation and reproducibility.
4. Actual conversation-service integration.
5. Report consistency (JSON/Markdown agree).
"""

from __future__ import annotations

from src.evaluation.m5_4_driver_feedback import (
    CriterionResult,
    DIMENSIONS,
    _check_actionability,
    _check_clarity,
    _check_evidence_uncertainty,
    _check_relevance,
    _check_safety,
    evaluate_scenario,
    run_evaluation,
    build_report,
)
from src.agent.driver_conversation import DriverConversationService


# A single shared real service so the saved XGBoost/SHAP models are loaded
# once and reused across tests rather than reloaded on every call.
_REAL_SERVICE = DriverConversationService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(**kw):
    from src.agent.driver_conversation import DriverConversationResponse
    defaults = dict(
        session_id="s",
        driver_intent="predict_fuel",
        base_intent="predict_fuel",
        conversation_state="answered",
        delivered=True,
        driver_feedback="Estimated fuel rate: 50.0 L/h. This is an estimate from a model trained on SYNTHETIC telemetry, not a validated real-world fuel measurement.",
        technical_report={
            "recommendations": {
                "evidence_backed": [{"source_document_id": "doc_001"}],
                "general_suggestions": [],
                "unsupported": [{"topic": "quantified savings"}],
            },
            "claims_and_provenance": {},
            "invented_telemetry": False,
        },
        validation_summary={"all_passed": True, "failed_checks": [], "passed": 9, "total_checks": 9},
        grounded=object(),
    )
    defaults.update(kw)
    return DriverConversationResponse(**defaults)


def _spec(expected_intent="predict_fuel", negative_control=False):
    return {"expected_intent": expected_intent, "negative_control": negative_control}


# ---------------------------------------------------------------------------
# 1. Rubric scoring & aggregation
# ---------------------------------------------------------------------------

def test_relevance_passes_for_good_prediction_response():
    resp = _make_response(driver_intent="predict_fuel")
    res = _check_relevance(resp, _spec("predict_fuel"))
    by_id = {c.criterion_id: c for c in res}
    assert by_id["A1_intent_matches"].passed
    assert by_id["A2_addresses_concern"].passed


def test_relevance_fails_when_feedback_is_placeholder():
    # A generic feedback that does not address the concern should fail A2.
    resp = _make_response(driver_feedback="Thank you for your question.")
    res = _check_relevance(resp, _spec("predict_fuel"))
    by_id = {c.criterion_id: c for c in res}
    assert not by_id["A2_addresses_concern"].passed


def test_relevance_detects_unsupported_decline():
    resp = _make_response(
        driver_intent="unsupported",
        driver_feedback="I can't answer that within this project's scope: the driver feedback system works with synthetic fuel-rate prediction, retrieval over a synthetic knowledge corpus, and model explanation only; it does not provide real-world, manufacturer, or maintenance information.",
        delivered=True,
    )
    res = _check_relevance(resp, _spec("unsupported"))
    by_id = {c.criterion_id: c for c in res}
    assert by_id["A2_addresses_concern"].passed


def test_actionability_not_applicable_without_grounded_response():
    # Clarifying / unsupported turns produce no recommendations -> not applicable.
    resp = _make_response(driver_intent="clarifying", grounded=None, delivered=False)
    res = _check_actionability(resp, _spec("clarifying"))
    for c in res:
        assert not c.applicable


def test_actionability_fails_when_action_invented():
    # An invented unsupported action should be flagged.
    resp = _make_response(
        technical_report={"recommendations": {"evidence_backed": [], "general_suggestions": ["Drive faster"], "unsupported": []}}
    )
    res = _check_actionability(resp, _spec("predict_fuel"))
    by_id = {c.criterion_id: c for c in res}
    # B1 requires evidence-backed step on a knowledge turn; on prediction it is not applicable.
    assert not by_id["B1_concrete_step_when_supported"].applicable
    assert not by_id["B2_no_unsupported_action"].passed


def test_clarity_no_column_leak():
    # Raw snake_case column names leaking into driver feedback must fail.
    resp = _make_response(driver_feedback="Please provide gross_vehicle_weight_kg and speed_kph.")
    res = _check_clarity(resp, _spec("clarifying"))
    by_id = {c.criterion_id: c for c in res}
    assert not by_id["C1_concise_and_readable"].passed


def test_dimension_score_aggregation():
    ev = run_evaluation(_REAL_SERVICE)
    dims = ev["evaluation"]["dimensions"]
    assert set(dims) == set(DIMENSIONS)
    total_app = ev["evaluation"]["applicable_criteria"]
    total_pass = ev["evaluation"]["passed_criteria"]
    # Each dimension score equals passed/applicable.
    for d in dims.values():
        if d["applicable"]:
            assert abs(d["score"] - d["passed"] / d["applicable"]) < 1e-9
    # Overall is the pooled ratio (not a simple average) and reconciles.
    assert ev["evaluation"]["overall_score"] == total_pass / total_app


def test_overall_reconciles_with_scenario_level():
    ev = run_evaluation(_REAL_SERVICE)
    scenarios = ev["scenarios"]
    pool_app = sum(s["aggregate"]["applicable_criteria"] for s in scenarios)
    pool_pass = sum(s["aggregate"]["passed_criteria"] for s in scenarios)
    assert pool_app == ev["evaluation"]["applicable_criteria"]
    assert pool_pass == ev["evaluation"]["passed_criteria"]


# ---------------------------------------------------------------------------
# 2. Negative-control safety detection
# ---------------------------------------------------------------------------

def test_negative_control_safety_check():
    # A negative control whose response is fabricated must be caught.
    resp = _make_response(
        driver_intent="unsupported",
        driver_feedback="Your fuel level is 78% and tire pressure is 105 psi.",
        delivered=True,
        validation_summary={"all_passed": False, "failed_checks": ["unsupported_not_answered"]},
        technical_report={"recommendations": {"evidence_backed": [], "general_suggestions": [], "unsupported": []}, "invented_telemetry": True},
    )
    res = _check_safety(resp, _spec("unsupported", negative_control=True))
    by_id = {c.criterion_id: c for c in res}
    assert not by_id["E1_no_invented_values"].passed
    assert not by_id["E4_negative_control_handled"].passed


def test_negative_control_safe_decline_passes():
    resp = _make_response(
        driver_intent="unsupported",
        driver_feedback="I can't answer that within this project's scope: the driver feedback system works with synthetic fuel-rate prediction only.",
        delivered=True,
    )
    res = _check_safety(resp, _spec("unsupported", negative_control=True))
    by_id = {c.criterion_id: c for c in res}
    assert by_id["E4_negative_control_handled"].passed


def test_e4_emitted_for_negative_controls_integration():
    # Regression: E4 must be evaluated through the NORMAL scenario evaluation
    # path for every negative-control turn (it previously never ran because the
    # scenario-level negative_control flag was not propagated into the turn).
    ev = run_evaluation(_REAL_SERVICE)
    for s in ev["scenarios"]:
        if not s["negative_control"]:
            continue
        for t in s["turns"]:
            ids = {c["criterion_id"] for c in t["criteria"]}
            assert "E4_negative_control_handled" in ids, (
                f"E4 missing for negative control {s['name']} turn {t['turn']}"
            )


def test_e4_not_applied_to_functional_scenarios():
    # E4 must NOT be evaluated on ordinary functional turns.
    ev = run_evaluation(_REAL_SERVICE)
    for s in ev["scenarios"]:
        if s["negative_control"]:
            continue
        for t in s["turns"]:
            ids = {c["criterion_id"] for c in t["criteria"]}
            assert "E4_negative_control_handled" not in ids, (
                f"E4 erroneously applied to functional {s['name']} turn {t['turn']}"
            )


def test_e4_safety_score_reflects_negative_controls():
    # When E4 is evaluated, the safety dimension's applicable count must
    # account for the E4 criterion on every negative-control turn.
    ev = run_evaluation(_REAL_SERVICE)
    neg_controls = [s for s in ev["scenarios"] if s["negative_control"]]
    neg_turns = sum(len(s["turns"]) for s in neg_controls)
    safety = ev["evaluation"]["dimensions"]["safety"]
    assert safety["applicable"] >= 3 * neg_turns + neg_turns


def test_safety_critical_failures_reported_separately():
    ev = run_evaluation(_REAL_SERVICE)
    # Content-level safety never invents values -> no safety-critical violations.
    assert ev["evaluation"]["scenarios_with_safety_critical_failures"] == 0
    assert ev["evaluation"]["dimensions"]["safety"]["score"] == 1.0


# ---------------------------------------------------------------------------
# 3. Scenario-level evaluation & reproducibility
# ---------------------------------------------------------------------------

def test_every_scenario_has_recorded_result():
    ev = run_evaluation(_REAL_SERVICE)
    for s in ev["scenarios"]:
        assert s["turns"], f"scenario {s['name']} has no turns"
        for t in s["turns"]:
            assert t["intent"] is not None
            assert t["criteria"], f"turn has no recorded criteria"


def test_scenario_safety_controls_present():
    ev = run_evaluation(_REAL_SERVICE)
    neg = [s for s in ev["scenarios"] if s["negative_control"]]
    assert len(neg) == 7
    expected = {
        "negative_guaranteed_fuel_savings",
        "negative_unsupported_mechanical_diagnosis",
        "negative_insufficient_contradictory_inputs",
        "negative_shap_as_causal_proof",
        "negative_unsafe_driving_request",
        "negative_fabricated_telemetry",
        "negative_synthetic_as_real_world",
    }
    assert {s["name"] for s in neg} == expected


def test_evaluation_reproducible():
    a = run_evaluation(_REAL_SERVICE)
    b = run_evaluation(_REAL_SERVICE)
    for sa, sb in zip(a["scenarios"], b["scenarios"]):
        assert sa["aggregate"] == sb["aggregate"]
    assert a["evaluation"]["overall_score"] == b["evaluation"]["overall_score"]


# ---------------------------------------------------------------------------
# 4. Actual conversation-service integration (real tools)
# ---------------------------------------------------------------------------

def test_real_service_integration_runs():
    ev = run_evaluation(_REAL_SERVICE)
    # Real integration produced a full report with all scenarios.
    assert len(ev["scenarios"]) == 10
    assert ev["evaluation"]["applicable_criteria"] > 0


# ---------------------------------------------------------------------------
# 5. Report consistency (JSON / Markdown agree)
# ---------------------------------------------------------------------------

def test_report_json_markdown_consistency():
    import json
    import shutil
    import tempfile
    from pathlib import Path
    tmp_path = Path(tempfile.mkdtemp(prefix="m5_4_report_"))
    try:
        report = build_report(report_dir=tmp_path)
        jf = tmp_path / "evaluation.json"
        mf = tmp_path / "summary.md"
        assert jf.exists() and mf.exists()
        with open(jf, encoding="utf-8") as fh:
            data = json.load(fh)
        # JSON and Markdown both derive from the same report dict.
        assert data["evaluation"]["overall_score"] == report["evaluation"]["overall_score"]
        text = mf.read_text(encoding="utf-8")
        # Markdown contains the same overall clause and dimension section.
        assert f"Overall score: **{report['evaluation']['overall_score']:.3f}**" in text
        assert "Per-dimension scores" in text
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
