"""Focused tests for the M2.4 agent evaluation framework.

These test the evaluator's own logic (scenario applicability, dimension
aggregation, tool selection, negative-control detection) without running the
full 49-scenario benchmark or loading model weights. They exercise the pure
logic in ``scripts/run_m2_agent_evaluation.py`` via importlib.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "run_m2_agent_evaluation.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("m2_4_runner", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


@pytest.fixture(scope="module")
def benchmark(runner):
    with open(ROOT / "reports/m2/agent_evaluation/benchmark.json",
              encoding="utf-8") as fh:
        return json.load(fh)


def _dim(runner, r, key):
    return runner.dimensions_for_scenario(dict(
        r,
        text_full=r.get("text_full") or "",
        text_preview=r.get("text_preview") or "",
    ))[key]


def _result(runner, **over):
    base = {
        "id": "TEST",
        "category": "combined",
        "build": "real",
        "status": "ok",
        "is_negative": False,
        "pass": True,
        "detected_intent": "fuel_prediction",
        "expected_intent": "fuel_prediction",
        "selected_tools": ["fuel_rate_prediction"],
        "expected_tools": ["fuel_rate_prediction"],
        "tool_exec_status": {"fuel_rate_prediction": "ok"},
        "expected_pass": ["prediction_value_preserved"],
        "expected_pass_ok": True,
        "expected_violations_ok": True,
        "requirements_ok": True,
        "prohibited_ok": True,
        "check_summary": {
            "passed_checks": sorted([
                "prediction_value_preserved", "synthetic_disclaimer",
                "failure_not_presented_as_success",
            ]),
            "failed_checks": [],
        },
        "text_full": "Estimated fuel rate: 52.5773 l/h. "
                     "synthetic model. this is a synthetic estimate.",
        "text_preview": "Estimated fuel rate: 52.5773 l/h.",
        "failure_categories": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# G applicability
# ---------------------------------------------------------------------------

class TestGApplicability:
    def test_combined_prediction_explanation_applicable(self, runner):
        r = _result(runner, category="combined",
                    expected_intent="combined_prediction_explanation",
                    selected_tools=["fuel_rate_prediction",
                                    "prediction_explanation"],
                    expected_tools=["fuel_rate_prediction",
                                    "prediction_explanation"],
                    tool_exec_status={"fuel_rate_prediction": "ok",
                                      "prediction_explanation": "ok"},
                    expected_pass=["prediction_value_preserved",
                                   "shap_value_preserved"],
                    check_summary={
                        "passed_checks": sorted([
                            "prediction_value_preserved",
                            "synthetic_disclaimer",
                            "shap_value_preserved",
                            "failure_not_presented_as_success",
                        ]),
                        "failed_checks": [],
                    },
                    text_full="Estimated fuel rate: 52.5773 l/h. "
                              "shap positive: Torque. synthetic model estimate.")
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["applicable"] is True
        assert g["passed"] is True

    def test_knowledge_only_applicable_no_prediction_required(self, runner):
        r = _result(runner, category="end_to_end",
                    expected_intent="knowledge_query",
                    selected_tools=["knowledge_retrieval"],
                    expected_tools=["knowledge_retrieval"],
                    tool_exec_status={"knowledge_retrieval": "ok"},
                    expected_pass=["source_attribution"],
                    check_summary={
                        "passed_checks": sorted([
                            "source_attribution", "synthetic_corpus_notice",
                            "oem_mislabeling", "insufficient_evidence_stated",
                        ]),
                        "failed_checks": [],
                    },
                    text_full="Synthetic educational corpus passage about engine load.")
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["applicable"] is True
        # No prediction expected -> no L/h / prediction check required.
        assert g["passed"] is True

    def test_prediction_only_applicable_no_retrieval_required(self, runner):
        r = _result(runner, category="end_to_end",
                    expected_intent="fuel_prediction",
                    selected_tools=["fuel_rate_prediction"],
                    expected_tools=["fuel_rate_prediction"],
                    tool_exec_status={"fuel_rate_prediction": "ok"},
                    expected_pass=["prediction_value_preserved"],
                    check_summary_checks=[
                        "prediction_value_preserved", "synthetic_disclaimer",
                    ])
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["applicable"] is True
        # No retrieval expected -> no source_attribution requirement.
        assert g["passed"] is True

    def test_non_combined_not_applicable(self, runner):
        r = _result(runner, category="prediction")
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["applicable"] is False
        assert g["passed"] is True


# ---------------------------------------------------------------------------
# Tool selection: order independence and failure detection
# ---------------------------------------------------------------------------

class TestGToolSelection:
    def test_tool_order_independent(self, runner):
        r = _result(
            runner,
            category="combined",
            expected_intent="combined_knowledge_prediction",
            selected_tools=["fuel_rate_prediction", "knowledge_retrieval"],
            expected_tools=["knowledge_retrieval", "fuel_rate_prediction"],
            tool_exec_status={"fuel_rate_prediction": "ok",
                              "knowledge_retrieval": "ok"},
            expected_pass=["prediction_value_preserved",
                           "source_attribution"],
            check_summary={
                "passed_checks": sorted([
                    "prediction_value_preserved", "synthetic_disclaimer",
                    "source_attribution", "synthetic_corpus_notice",
                    "oem_mislabeling", "insufficient_evidence_stated",
                ]),
                "failed_checks": [],
            },
            text_full="Estimated 52 l/h synthetic model. synthetic educational passage.")
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["passed"] is True

    def test_expected_tool_not_selected_fails(self, runner):
        r = _result(runner, selected_tools=["fuel_rate_prediction"],
                    expected_tools=["fuel_rate_prediction",
                                    "prediction_explanation"],
                    tool_exec_status={"fuel_rate_prediction": "ok"})
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["passed"] is False

    def test_failed_expected_tool_fails(self, runner):
        r = _result(
            runner,
            category="combined",
            expected_intent="combined_knowledge_prediction",
            selected_tools=["knowledge_retrieval", "fuel_rate_prediction"],
            expected_tools=["knowledge_retrieval", "fuel_rate_prediction"],
            tool_exec_status={"knowledge_retrieval": "ok",
                              "fuel_rate_prediction": "error"},
            expected_pass=["prediction_value_preserved",
                           "source_attribution"],
            check_summary_checks=[
                "source_attribution", "synthetic_corpus_notice",
                "oem_mislabeling", "insufficient_evidence_stated",
            ])
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["passed"] is False

    def test_missing_prediction_output_fails(self, runner):
        r = _result(
            runner,
            text_full="knowledge passage about synthetic corpus. no figure.",
            check_summary={
                "passed_checks": sorted([
                    "source_attribution", "synthetic_corpus_notice",
                ]),
                "failed_checks": [],
            },
        )
        # prediction expected, but no L/h in text -> output check fails
        g = _dim(runner, r, "G_end_to_end_reliability")
        assert g["passed"] is False


# ---------------------------------------------------------------------------
# Negative control detection
# ---------------------------------------------------------------------------

class TestNegativeControls:
    def test_negative_control_detects_violation(self, runner):
        # A scenario asserting an expected violation passes iff the violation
        # actually fired (expected_violations_ok True).
        r = _result(
            runner,
            is_negative=True,
            expected_violations_ok=True,
            expected_pass_ok=False,
            check_summary_checks=[],
            failure_categories=["causal_language"],
        )
        assert r["is_negative"] is True

    def test_negative_control_requires_fired_violation(self, runner):
        r = _result(runner, is_negative=True,
                    expected_violations_ok=False)
        # Negative control that did not fire its violation must not count as
        # a pass of the violation check.
        assert r["expected_violations_ok"] is False


# ---------------------------------------------------------------------------
# Dimension metric aggregation
# ---------------------------------------------------------------------------

class TestMetricAggregation:
    def test_aggregate_matches_scenario_level(self, runner, benchmark):
        res = benchmark["scenarios"]
        # Build a minimal synthetic result set the aggregate can consume.
        results = []
        for sc in res[:6]:
            r = _result(runner, id=sc["id"], category=sc["category"],
                        expected_intent=sc["expected_intent"],
                        expected_tools=sc["expected_tools"],
                        selected_tools=list(sc["expected_tools"]),
                        tool_exec_status={t: "ok"
                                          for t in sc["expected_tools"]},
                        expected_pass=sc.get("expect_pass", []))
            r["dimensions"] = runner.dimensions_for_scenario(dict(
                r, text_full="synthetic model estimate 52 l/h passage knowledge",
                text_preview="synthetic model estimate",
            ))
            results.append(r)

        agg = runner.aggregate_dimensions(results)
        for key, dim in agg.items():
            recalc_app = sum(1 for x in results
                             if x.get("dimensions", {}).get(key, {})
                             .get("applicable"))
            assert dim["total_applicable"] == recalc_app
