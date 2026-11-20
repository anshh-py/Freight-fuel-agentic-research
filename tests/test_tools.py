"""Focused tests for the M2.2 agent tools.

Covers each tool's input validation and output schema, loading/invoking the
saved M1.2 models without retraining, retrieval source provenance, the
instance-level explanation output and its provenance flags, and insufficient-
evidence handling. Includes real local integration checks (saved XGBoost
models + the synthetic corpus/vector store).
"""

from __future__ import annotations

import copy
import pytest

from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.knowledge_retrieval import KnowledgeRetrievalTool
from src.tools.prediction_explanation import PredictionExplanationTool
from src.tools.validation import ToolInputError

FULL_INPUTS = {
    "vehicle_class": "32T",
    "gross_vehicle_weight_kg": 40000.0,
    "payload_kg": 10000.0,
    "vehicle_mass_kg": 30000.0,
    "speed_kph": 75.0,
    "acceleration_mps2": 0.1,
    "road_grade_pct": 1.0,
    "ignition_status": 1,
    "engine_rpm": 1500.0,
    "engine_load_pct": 60.0,
    "throttle_position_pct": 40.0,
    "torque_nm": 1200.0,
    "coolant_temp_c": 85.0,
    "intake_air_temp_c": 30.0,
    "ambient_temp_c": 20.0,
}

TELE_INPUTS = {
    k: v for k, v in FULL_INPUTS.items()
    if k in {
        "vehicle_class", "gross_vehicle_weight_kg", "payload_kg",
        "vehicle_mass_kg", "speed_kph", "acceleration_mps2",
        "road_grade_pct", "ignition_status", "coolant_temp_c",
        "intake_air_temp_c", "ambient_temp_c",
    }
}


# ---------------------------------------------------------------------------
# Tool 1 — fuel-rate prediction (real saved models)
# ---------------------------------------------------------------------------

class TestPredictionTool:
    def test_predict_full_operational_schema(self):
        tool = FuelRatePredictionTool()
        out = tool.predict(FULL_INPUTS, "full_operational")
        assert out["status"] == "ok"
        assert isinstance(out["predicted_fuel_rate_lph"], float)
        assert out["predicted_fuel_rate_lph"] >= 0.0
        assert out["units"] == "L/h"
        assert out["feature_set"] == "full_operational"
        assert out["model"]["name"] == "xgboost"
        assert "xgboost.joblib" in out["model"]["artifact"]
        assert out["provenance"]["synthetic_training"] is True
        assert out["provenance"]["model_is_estimate"] is True

    def test_predict_telematics_only_schema(self):
        tool = FuelRatePredictionTool()
        out = tool.predict(TELE_INPUTS, "telematics_only")
        assert out["feature_set"] == "telematics_only"
        assert out["predicted_fuel_rate_lph"] >= 0.0
        assert set(out["input_fields_used"]) == set(TELE_INPUTS)

    def test_predict_rejects_missing_field(self):
        tool = FuelRatePredictionTool()
        bad = copy.deepcopy(FULL_INPUTS)
        del bad["torque_nm"]
        with pytest.raises(ToolInputError):
            tool.predict(bad, "full_operational")

    def test_predict_rejects_out_of_range(self):
        tool = FuelRatePredictionTool()
        bad = copy.deepcopy(FULL_INPUTS)
        bad["speed_kph"] = -5.0
        with pytest.raises(ToolInputError):
            tool.predict(bad, "full_operational")

    def test_predict_rejects_unknown_field(self):
        tool = FuelRatePredictionTool()
        bad = copy.deepcopy(FULL_INPUTS)
        bad["not_a_feature"] = 1.0
        with pytest.raises(ToolInputError):
            tool.predict(bad, "full_operational")

    def test_predict_rejects_invalid_vehicle_class(self):
        tool = FuelRatePredictionTool()
        bad = copy.deepcopy(FULL_INPUTS)
        bad["vehicle_class"] = "99T"
        with pytest.raises(ToolInputError):
            tool.predict(bad, "full_operational")

    def test_predict_rejects_unknown_feature_set(self):
        tool = FuelRatePredictionTool()
        with pytest.raises(ToolInputError):
            tool.predict(FULL_INPUTS, "brain_control")


# ---------------------------------------------------------------------------
# Tool 2 — knowledge retrieval (real corpus / vector store)
# ---------------------------------------------------------------------------

class TestRetrievalTool:
    def test_retrieval_provenance(self, retrieval_store):
        tool = KnowledgeRetrievalTool(store=retrieval_store)
        out = tool.retrieve("how does road grade affect fuel consumption on trucks")
        assert out["status"] == "ok"
        assert out["n_results"] > 0
        p = out["passages"][0]
        assert p["document_id"]
        assert p["source_reference"]
        assert p["is_oem_manual"] is False
        assert p["topic"]
        assert 0.0 <= p["score"] <= 1.0
        assert out["provenance"]["corpus_synthetic"] is True
        assert out["provenance"]["similarity_not_proof_of_correctness"] is True

    def test_retrieval_empty_query_rejected(self, retrieval_store):
        tool = KnowledgeRetrievalTool(store=retrieval_store)
        with pytest.raises(ToolInputError):
            tool.retrieve("   ")

    def test_retrieval_insufficient_evidence_flag(self, retrieval_store):
        tool = KnowledgeRetrievalTool(store=retrieval_store, min_score=0.99)
        out = tool.retrieve("what is the weather on mars")
        assert out["evidence_sufficiency"]["insufficient"] is True
        assert out["status"] == "ok"


# ---------------------------------------------------------------------------
# Tool 3 — prediction explanation (real saved model + SHAP)
# ---------------------------------------------------------------------------

class TestExplanationTool:
    def test_explain_instance_level_schema(self):
        tool = PredictionExplanationTool()
        out = tool.explain(FULL_INPUTS, "full_operational")
        assert out["status"] == "ok"
        assert out["explanation_kind"] == "instance_level_shap"
        assert out["reconstruction"]["pass"] is True
        assert out["predicted_fuel_rate_lph"] >= 0.0
        mi = out["most_influential_features"]
        for key in ("top_positive_shap", "top_negative_shap"):
            assert mi[key], f"{key} should not be empty"
            for item in mi[key]:
                assert "column" in item
                assert "readable_label" in item
                assert "shap" in item
        assert out["limitations"]["shap_is_not_causal"] is True
        assert "not a causal explanation" in out["limitations"]["note"]

    def test_explain_rejects_missing_field(self):
        tool = PredictionExplanationTool()
        bad = copy.deepcopy(FULL_INPUTS)
        del bad["engine_rpm"]
        with pytest.raises(ToolInputError):
            tool.explain(bad, "full_operational")
