"""M5.3 — focused tests for prediction & explanation reliability evaluation.

Separates:
  * unit tests of the metric/check helpers (stub-style result dicts, no model
    artifacts), and
  * an integration test (marked ``integration``) that exercises the real saved
    XGBoost prediction + explanation tools on canonical inputs.
"""

from __future__ import annotations

import pytest

from src.evaluation.m5_3_prediction_explanation import (
    FEATURE_SETS,
    INVALID_CASES,
    canonical_inputs,
    check_attribution_direction,
    check_explanation_reconstruction,
    check_explanation_safety,
    check_finite_valid,
    check_invalid_input_rejection,
    classify_segment,
    prediction_agreement_with_direct_model,
    prediction_explanation_consistency,
    repeated_prediction_consistency,
    sample_test_inputs,
)
from src.tools.validation import ToolInputError


# ---------------------------------------------------------------------------
# Helper checks
# ---------------------------------------------------------------------------

class TestFiniteValid:
    def test_ok_finite_nonneg_passes(self):
        res = {"status": "ok", "predicted_fuel_rate_lph": 13.25}
        assert check_finite_valid(res)["pass"] is True

    def test_nan_rejected(self):
        res = {"status": "ok", "predicted_fuel_rate_lph": float("nan")}
        assert check_finite_valid(res)["pass"] is False
        assert check_finite_valid(res)["finite"] is False

    def test_negative_rejected(self):
        res = {"status": "ok", "predicted_fuel_rate_lph": -1.0}
        assert check_finite_valid(res)["pass"] is False
        assert check_finite_valid(res)["non_negative"] is False

    def test_bad_status_rejected(self):
        res = {"status": "error", "predicted_fuel_rate_lph": 13.25}
        assert check_finite_valid(res)["pass"] is False


class TestInconsistentRejection:
    def test_all_invalid_cases_raise(self):
        # Simulate with a stub tool that fails through validate_vehicle_inputs.
        class _Tool:
            def predict(self, inputs, feature_set=None):
                # Mirror the real tool's usage of validation.
                from src.tools.validation import validate_vehicle_inputs
                validate_vehicle_inputs(inputs, feature_set)
                return {"status": "ok", "predicted_fuel_rate_lph": 1.0}

        for fs in FEATURE_SETS:
            base = canonical_inputs(fs)
            res = check_invalid_input_rejection(_Tool(), fs, base)
            assert res["n_rejected"] == res["n_cases"]
            assert res["n_cases"] == len(INVALID_CASES)

    def test_canonical_inputs_valid_for_both_fs(self):
        for fs in FEATURE_SETS:
            inputs = canonical_inputs(fs)
            from src.tools.validation import validate_vehicle_inputs
            _, cleaned = validate_vehicle_inputs(inputs, fs)
            assert cleaned


class TestRepeatedConsistency:
    def test_identical_values(self):
        class _Tool:
            def predict(self, inputs, feature_set=None):
                return {"predicted_fuel_rate_lph": 7.5}

        res = repeated_prediction_consistency(_Tool(), {}, "full_operational")
        assert res["all_identical"] is True

    def test_different_values_detected(self):
        class _Tool:
            def __init__(self):
                self.i = 0

            def predict(self, inputs, feature_set=None):
                self.i += 1
                return {"predicted_fuel_rate_lph": float(self.i)}

        res = repeated_prediction_consistency(_Tool(), {}, "full_operational")
        assert res["all_identical"] is False


class TestAttributionDirection:
    def test_valid_direction(self):
        res = {
            "most_influential_features": {
                "top_positive_shap": [{"shap": 1.2}, {"shap": 0.4}],
                "top_negative_shap": [{"shap": -1.1}, {"shap": -0.2}],
            }
        }
        assert check_attribution_direction(res)["ok"] is True

    def test_invalid_sign_detected(self):
        res = {
            "most_influential_features": {
                "top_positive_shap": [{"shap": -0.5}],
                "top_negative_shap": [{"shap": 0.5}],
            }
        }
        assert check_attribution_direction(res)["ok"] is False

    def test_empty_lists_rejected(self):
        res = {
            "most_influential_features": {
                "top_positive_shap": [],
                "top_negative_shap": [],
            }
        }
        assert check_attribution_direction(res)["ok"] is False


class TestExplanationReconstruction:
    def test_pass_within_tolerance(self):
        res = {"reconstruction": {"pass": True,
                                   "max_abs_deviation": 1e-5,
                                   "tolerance": 1e-3}}
        assert check_explanation_reconstruction(res)["ok"] is True

    def test_fail_detected(self):
        res = {"reconstruction": {"pass": False,
                                   "max_abs_deviation": 0.5,
                                   "tolerance": 1e-3}}
        assert check_explanation_reconstruction(res)["ok"] is False


class TestExplanationSafety:
    def test_safe_result(self):
        res = {
            "limitations": {"shap_is_not_causal": True, "instance_scope": "x"},
            "provenance": {"synthetic_training": True},
            "text": "This is a model explanation; not a guarantee.",
        }
        assert check_explanation_safety(res)["ok"] is True

    def test_causal_claim_detected(self):
        res = {
            "limitations": {"shap_is_not_causal": False, "instance_scope": "x"},
            "provenance": {"synthetic_training": True},
            "text": "Torque causes higher fuel use.",
        }
        assert check_explanation_safety(res)["ok"] is False


class TestPredictionExplanationConsistency:
    def test_matching_values(self):
        pred = {"predicted_fuel_rate_lph": 10.0}
        expl = {"predicted_fuel_rate_lph": 10.0}
        assert prediction_explanation_consistency(pred, expl)["within_tolerance"] is True

    def test_mismatch_detected(self):
        pred = {"predicted_fuel_rate_lph": 10.0}
        expl = {"predicted_fuel_rate_lph": 99.0}
        assert prediction_explanation_consistency(pred, expl)["within_tolerance"] is False


class TestSegmentClassification:
    def test_highway(self):
        assert classify_segment({"speed_kph": 80, "acceleration_mps2": 0.1,
                                 "road_grade_pct": 0.0, "ignition_status": 1}) == "highway"

    def test_acceleration(self):
        assert classify_segment({"speed_kph": 40, "acceleration_mps2": 0.8,
                                 "road_grade_pct": 0.0, "ignition_status": 1}) == "acceleration"

    def test_idle_stopped(self):
        assert classify_segment({"speed_kph": 0.0, "acceleration_mps2": 0.0,
                                 "road_grade_pct": 0.0, "ignition_status": 1}) == "idle_stopped"

    def test_uphill(self):
        assert classify_segment({"speed_kph": 30, "acceleration_mps2": 0.1,
                                 "road_grade_pct": 1.0, "ignition_status": 1}) == "uphill"


class TestSegmentSample:
    def test_sample_returns_expected_segments(self):
        import pandas as pd
        df = pd.DataFrame({
            "speed_kph": [0.0, 80.0, 40.0, 30.0],
            "acceleration_mps2": [0.0, 0.1, 0.8, 0.1],
            "road_grade_pct": [0.0, 0.0, 0.0, 1.0],
            "ignition_status": [1, 1, 1, 1],
        })
        sample = sample_test_inputs(df, prices_segment=2, seed=1)
        assert set(sample.keys()) == {"highway", "acceleration", "idle_stopped", "uphill"}
        for seg, rows in sample.items():
            assert len(rows) <= 2


# ---------------------------------------------------------------------------
# Integration test (real saved model artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_tools_agree_and_explain():
    from src.tools.fuel_rate_prediction import FuelRatePredictionTool
    from src.tools.prediction_explanation import PredictionExplanationTool
    from src.evaluation.m5_3_prediction_explanation import direct_model_prediction

    fs = "full_operational"
    inputs = canonical_inputs(fs)

    pred_tool = FuelRatePredictionTool()
    expl_tool = PredictionExplanationTool()

    pred_res = pred_tool.predict(inputs, feature_set=fs)
    assert pred_res["status"] == "ok"
    assert check_finite_valid(pred_res)["pass"] is True

    # Tool agrees with the direct saved-model prediction.
    agree = prediction_agreement_with_direct_model(pred_tool, inputs, fs)
    assert agree["within_tolerance"] is True

    # Explanation reconstructs and is consistent with the prediction.
    expl_res = expl_tool.explain(inputs, feature_set=fs)
    assert check_explanation_reconstruction(expl_res)["ok"] is True
    assert check_attribution_direction(expl_res)["ok"] is True
    assert check_explanation_safety(expl_res)["ok"] is True
    assert prediction_explanation_consistency(pred_res, expl_res)["within_tolerance"] is True
