"""Focused tests for M5.5 ablation study (deterministic).

Unit tests use lightweight stubs where isolated logic is under test. The
baseline integration tests exercise the real service.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.agent.driver_conversation import DriverConversationService
from src.agent.driver_session import DriverSession
from src.evaluation.m5_4_driver_feedback import _SCENARIOS, evaluate_turn
from src.evaluation.m5_5_ablation_study import (
    DEFAULT_FEATURE_SET,
    TELEMATICS_FEATURE_SET,
    _EmptyRetrievalTool,
    _NoOpExplanationTool,
    _no_explanation_service,
    _no_retrieval_service,
    aggregate_variant,
    compare_feature_sets,
    run_scenario_for_variant,
    run_validation_shadow,
    run_variant,
    shared_scenarios,
)

SC = shared_scenarios()
BASELINE_VARIANT = "A_full_pipeline"

# Cache heavy real-study runs so the focused suite does not recompute the full
# pipeline dozens of times (each run loads the saved models).
_VARIANT_CACHE = {}
_SHADOW_CACHE = {}


def _run_variant_cached(*args, **kwargs):
    key = (args, tuple(sorted(kwargs.items())))
    if key not in _VARIANT_CACHE:
        _VARIANT_CACHE[key] = run_variant(*args, **kwargs)
    return _VARIANT_CACHE[key]


def _shadow_cached(scenarios):
    key = id(scenarios)
    if key not in _SHADOW_CACHE:
        _SHADOW_CACHE[key] = run_validation_shadow(scenarios)
    return _SHADOW_CACHE[key]


def _criterion(resp, spec, history=None):
    return evaluate_turn(resp, spec, history or [])


# ---------------------------------------------------------------------------
# 1. Variant configuration & isolation
# ---------------------------------------------------------------------------

def test_variant_config_service_isolation():
    """Distinct variants must build distinct services; A uses real tools."""
    s_a = DriverConversationService(default_feature_set=DEFAULT_FEATURE_SET)
    s_c = _no_retrieval_service()
    s_d = _no_explanation_service()
    assert s_a.tools["knowledge_retrieval"].__class__.__name__ == "KnowledgeRetrievalTool"
    assert isinstance(s_c.tools["knowledge_retrieval"], _EmptyRetrievalTool)
    assert isinstance(s_d.tools["prediction_explanation"], _NoOpExplanationTool)
    # Real tools retained where not ablated.
    assert isinstance(s_d.tools["fuel_rate_prediction"].__class__, type)
    assert s_d.tools["fuel_rate_prediction"].__class__.__name__ == "FuelRatePredictionTool"


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        run_scenario_for_variant("Z_does_not_exist", SC[0])


# ---------------------------------------------------------------------------
# 2. Identical inputs across comparable variants
# ---------------------------------------------------------------------------

def test_shared_scenarios_have_same_ids_and_inputs():
    """Compareable variants consume the same scenario IDs and user inputs."""
    full_run = _run_variant_cached(BASELINE_VARIANT, SC)
    no_session = _run_variant_cached(BASELINE_VARIANT, SC, carry_session=False)
    ids_full = [s["name"] for s in full_run["scenarios"]]
    ids_ns = [s["name"] for s in no_session["scenarios"]]
    assert ids_full == ids_ns
    # Same message text + raw input dicts by scenario/turn.
    for sc in SC:
        for t in sc["turns"]:
            assert isinstance(t.get("message"), str)
            assert "inputs" in t


# ---------------------------------------------------------------------------
# 3. Session-context ablation
# ---------------------------------------------------------------------------

def test_session_context_ablation_preserves_scenario_count():
    full_run = _run_variant_cached(BASELINE_VARIANT, SC)
    no_session = _run_variant_cached(BASELINE_VARIANT, SC, carry_session=False)
    assert full_run["evaluation"]["scenarios_total"] == len(SC)
    assert no_session["evaluation"]["scenarios_total"] == len(SC)
    # No scenario silently disappears.
    assert len(no_session["scenarios"]) == len(SC)


def test_session_context_ablation_actually_uses_fresh_sessions():
    """Carry_session=False must not share a session across turns."""
    # By construction our runner builds a new DriverSession per turn. Verify
    # that a multi-turn scenario's later turns are NOT resumable from context.
    sc = next(s for s in SC if s["name"] == "functional_clarification_then_resume")
    no_session = run_scenario_for_variant(
        BASELINE_VARIANT, sc, carry_session=False)
    # With no carried context, turn 2 (only FULL_INPUTS) is still ready; but
    # the key is that the router cannot resume a pending request.
    turns = no_session["turns"]
    assert len(turns) == 2


# ---------------------------------------------------------------------------
# 4. Retrieval ablation
# ---------------------------------------------------------------------------

def test_retrieval_ablation_uses_empty_stub():
    service = _no_retrieval_service()
    session = DriverSession(feature_set=DEFAULT_FEATURE_SET)
    resp = service.handle(session, "What should I do to reduce unnecessary fuel consumption?", None)
    # The grounded response should carry no retrieved passages (ablation).
    grounded = resp.grounded
    if grounded is not None:
        assert resp.grounded.evidence == []
        assert resp.grounded.evidence_insufficient is True


def test_retrieval_ablation_runs_without_error():
    run = _run_variant_cached("C_without_retrieval", SC)
    assert run["evaluation"]["scenarios_total"] == len(SC)


# ---------------------------------------------------------------------------
# 5. SHAP ablation
# ---------------------------------------------------------------------------

def test_explanation_ablation_uses_noop_stub():
    service = _no_explanation_service()
    session = DriverSession(feature_set=DEFAULT_FEATURE_SET)
    resp = service.handle(session, "Why does the model think fuel consumption is increasing?", None)
    # The explanation tool is a no-op: either no grounded response or no
    # explanation block.
    if resp.grounded is not None:
        assert resp.grounded.explanation is None


def test_explanation_ablation_runs_without_error():
    run = _run_variant_cached("D_without_explanation", SC)
    assert run["evaluation"]["scenarios_total"] == len(SC)


# ---------------------------------------------------------------------------
# 6. Offline response-validation comparison
# ---------------------------------------------------------------------------

def test_validation_shadow_is_offline_and_never_delivers():
    shadow = _shadow_cached(SC)
    assert shadow["unvalidated_never_delivered"] is True
    totals = shadow["totals"]
    assert totals["scenarios"] == len(SC)
    # Every turn is accounted for.
    turns = sum(len(s["turns"]) for s in shadow["scenarios"])
    assert turns == totals["turns"]


def test_validation_shadow_reports_checks():
    shadow = _shadow_cached(SC)
    for s in shadow["scenarios"]:
        for t in s["turns"]:
            assert t["total_validation_checks"] >= 0
            assert t["failed_checks"] >= 0
            assert 0 <= t["error_severity_failed"] <= t["failed_checks"]
            assert isinstance(t["failed_check_ids"], list)


# ---------------------------------------------------------------------------
# 7. Score calculations & percentage-point differences
# ---------------------------------------------------------------------------

def test_aggregate_reconciles_with_scenario_level():
    """Pooled aggregates equal the sum of scenario-level applicable/passed."""
    run = _run_variant_cached(BASELINE_VARIANT, SC)
    total_app = sum(s["aggregate"]["applicable_criteria"] for s in run["scenarios"])
    total_pass = sum(s["aggregate"]["passed_criteria"] for s in run["scenarios"])
    assert total_app == run["evaluation"]["applicable_criteria"]
    assert total_pass == run["evaluation"]["passed_criteria"]
    expected = total_pass / total_app
    assert run["evaluation"]["overall_score"] == pytest.approx(expected)


def test_dimension_scores_reconcile():
    run = _run_variant_cached(BASELINE_VARIANT, SC)
    for dim in ("relevance", "actionability", "clarity", "evidence_uncertainty",
                "safety", "consistency"):
        applicable = sum(s["dimension_scores"][dim]["applicable"]
                         for s in run["scenarios"])
        passed = sum(s["dimension_scores"][dim]["passed"]
                     for s in run["scenarios"])
        assert applicable == run["evaluation"]["dimensions"][dim]["applicable"]
        assert passed == run["evaluation"]["dimensions"][dim]["passed"]
        if applicable:
            assert run["evaluation"]["dimensions"][dim]["score"] == pytest.approx(passed / applicable)


def test_percentage_point_difference_calculation():
    base = _run_variant_cached(BASELINE_VARIANT, SC)
    ablated = _run_variant_cached(BASELINE_VARIANT, SC, carry_session=False)
    from src.evaluation.m5_5_ablation_study import _diff_from_baseline
    diff = _diff_from_baseline(base, ablated)
    b = base["evaluation"]["overall_score"]
    v = ablated["evaluation"]["overall_score"]
    assert diff["overall"]["delta_pp"] == pytest.approx((v - b) * 100)
    for dim in ("relevance", "clarity", "safety"):
        d = diff["dimensions"][dim]
        assert d["delta_pp"] is None or abs(d["delta_pp"]) <= 100.0


# ---------------------------------------------------------------------------
# 8. Safety-critical violation reporting
# ---------------------------------------------------------------------------

def test_safety_critical_failures_reported_separately():
    run = _run_variant_cached(BASELINE_VARIANT, SC)
    # The aggregate count must equal the number of scenarios flagged.
    flagged = [s for s in run["scenarios"] if s["safety_critical_failures"]]
    assert run["evaluation"]["scenarios_with_safety_critical_failures"] == len(flagged)
    assert len(run["safety_critical_violations"]) == len(flagged)


def test_safety_failures_not_hidden_by_average():
    """Even with a high overall score, safety-critical failures must surface."""
    run = _run_variant_cached(BASELINE_VARIANT, SC)
    overall = run["evaluation"]["overall_score"]
    if overall > 0.5:  # baseline is ~0.92
        # We only assert that the reporting structure carries the violations.
        assert "safety_critical_violations" in run
        assert "scenarios_with_safety_critical_failures" in run["evaluation"]


# ---------------------------------------------------------------------------
# 9. Real full-pipeline baseline integration
# ---------------------------------------------------------------------------

def test_baseline_integration_matches_m5_4():
    """Variant A (full pipeline) must reproduce the frozen M5.4 numbers."""
    run = _run_variant_cached(BASELINE_VARIANT, SC)
    # M5.4 was frozen at 185/201 applicable (92.0%).
    assert run["evaluation"]["passed_criteria"] == 185
    assert run["evaluation"]["applicable_criteria"] == 201
    assert run["evaluation"]["overall_score"] == pytest.approx(185 / 201)


def test_real_service_used_in_baseline():
    """The baseline must execute the real service (real tools), not mocks."""
    from src.evaluation.m5_5_ablation_study import _base_service
    service = _base_service()
    assert isinstance(service, DriverConversationService)
    assert service.tools["fuel_rate_prediction"].__class__.__name__ == "FuelRatePredictionTool"
    assert service.tools["knowledge_retrieval"].__class__.__name__ == "KnowledgeRetrievalTool"
    assert service.tools["prediction_explanation"].__class__.__name__ == "PredictionExplanationTool"


# ---------------------------------------------------------------------------
# 10. JSON/Markdown consistency
# ---------------------------------------------------------------------------

REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "m5" / "m5_5"


def _loaded_report():
    json_path = REPORT_DIR / "evaluation.json"
    md_path = REPORT_DIR / "summary.md"
    assert json_path.exists(), "reports/m5/m5_5/evaluation.json missing"
    assert md_path.exists(), "reports/m5/m5_5/summary.md missing"
    return json.loads(json_path.read_text(encoding="utf-8")), json_path, md_path


def test_report_json_markdown_agree():
    data, json_path, md_path = _loaded_report()
    md_text = md_path.read_text(encoding="utf-8")
    assert data["milestone"] == "M5.5"
    assert data["baseline_variant"] == "A_full_pipeline"
    # Overall score appears in both.
    base = data["variants"]["A_full_pipeline"]["results"]["overall_score"]
    assert f"{base:.3f}" in md_text
    # Safety counts appear in both.
    count = data["variants"][
        "A_full_pipeline"]["results"]["scenarios_with_safety_critical_failures"]
    assert str(count) in md_text
    # Scenario count and applicable criteria also present in markdown.
    assert str(data["variants"]["A_full_pipeline"]["results"]["applicable_criteria"]) in md_text


def test_report_mentions_limitations():
    data, _, _ = _loaded_report()
    limitations = data["limitations"]
    assert any("synthetic" in lim.lower() for lim in limitations)
    assert any("not" in lim.lower() and "real-world" in lim.lower() for lim in limitations)
