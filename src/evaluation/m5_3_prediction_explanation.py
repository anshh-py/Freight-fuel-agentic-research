"""M5.3 — prediction and explanation reliability evaluation (deterministic).

This module audits the reliability, consistency, numerical validity and
explanation correctness of the prediction-and-explanation pipeline at the
TOOL layer, using the real saved M1.2 XGBoost models + preprocessors behind
``FuelRatePredictionTool`` and ``PredictionExplanationTool``.

Scope / non-duplication
  * M1.3 already reports pooled / per-truck / per-condition regression metrics
    (MAE, RMSE, R²) on the frozen 5-truck test set, and M1.4 already reports
    SHAP reconstruction and global/per-truck importance on a full sample.
    This module deliberately does NOT recompute those aggregate tables.
  * M5.3 adds what the batch pipelines do not cover: feeding raw validated
    inputs through the real agent tools and checking
      - repeated identical-input prediction consistency (determinism),
      - finite / valid outputs and correct feature-set handling,
      - invalid-input rejection (a rejected input must never yield a
        "successful" prediction),
      - tool-vs-direct-model prediction agreement,
      - per-instance SHAP reconstruction + attribution sign/direction,
      - prediction vs explanation output consistency for the same inputs,
      - segment (highway/accel/idle/uphill) and per-truck tool agreement on a
        small deterministic sample, and
      - safety flags: no causal claim for SHAP, no guaranteed-savings claim,
        explicit synthetic-data disclaimer.

Design guarantees
  * Uses only saved artifacts; never retrains models or refits preprocessors.
  * Deterministic: fixed seed, fixed sample, no LLM, no external API, no DB.
  * Keep tests/data separate from any tuning; nothing is fit here.
  * Frozen M1/M2/M3/M4 implementation files are not modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.spec import get_feature_set, load_feature_spec
from src.evaluation.explainability import load_selected_xgb
from src.tools.fuel_rate_prediction import FuelRatePredictionTool
from src.tools.prediction_explanation import PredictionExplanationTool
from src.tools.validation import ToolInputError

FEATURE_SETS: tuple[str, ...] = ("full_operational", "telematics_only")
RECONSTRUCTION_TOLERANCE = 1e-3
DEFAULT_REPEATS = 3
SEED = 20250801


# ---------------------------------------------------------------------------
# Canonical valid inputs (full / telematics-only)
# ---------------------------------------------------------------------------

def _canonical_full_inputs() -> dict:
    """A known-valid full_operational input set (within observed ranges)."""
    return {
        "gross_vehicle_weight_kg": 40000.0,
        "payload_kg": 8000.0,
        "vehicle_mass_kg": 32000.0,
        "speed_kph": 72.0,
        "acceleration_mps2": 0.4,
        "road_grade_pct": 0.5,
        "ignition_status": 1,
        "engine_rpm": 1450.0,
        "engine_load_pct": 62.0,
        "throttle_position_pct": 55.0,
        "torque_nm": 1400.0,
        "coolant_temp_c": 82.0,
        "intake_air_temp_c": 28.0,
        "ambient_temp_c": 22.0,
        "vehicle_class": "40T",
    }


def _canonical_telematics_inputs() -> dict:
    """A known-valid telematics_only input set (subset of full)."""
    full = _canonical_full_inputs()
    return {k: v for k, v in full.items() if k != "engine_rpm" and k != "engine_load_pct"
            and k != "throttle_position_pct" and k != "torque_nm"}


def canonical_inputs(feature_set: str) -> dict:
    if feature_set == "full_operational":
        return _canonical_full_inputs()
    if feature_set == "telematics_only":
        return _canonical_telematics_inputs()
    raise ValueError(f"unknown feature set: {feature_set}")


def project_row_to_inputs(row: dict, feature_set: str) -> dict:
    """Project a raw telemetry row onto exactly a feature set's input fields.

    Raw rows carry extra columns (truck_id, trip_id, fuel_rate_lph, ...) that
    the tool validation rejects. This returns only the spec columns for
    ``feature_set``, so it can be fed to the real tools.
    """
    spec = load_feature_spec()
    features = get_feature_set(spec, feature_set)
    return {f: row[f] for f in features}


# ---------------------------------------------------------------------------
# Direct-model prediction (for tool-vs-model agreement)
# ---------------------------------------------------------------------------

def direct_model_prediction(feature_set: str, inputs: dict, model_dir=None) -> float:
    """Predict directly with the saved estimator (same preprocessor path)."""
    tool = FuelRatePredictionTool(model_dir=model_dir) if model_dir is not None \
        else FuelRatePredictionTool()
    entry = tool._ensure_loaded()[feature_set]
    estimator = entry["model"]
    preprocessor = entry["preprocessor"]
    columns = entry["columns"]
    spec = load_feature_spec()
    features = get_feature_set(spec, feature_set)
    row = pd.DataFrame([inputs], columns=features)
    X = preprocessor.transform(row)
    if list(X.columns) != columns:
        raise RuntimeError("transformed column layout mismatch")
    return float(estimator.predict(X)[0])


# ---------------------------------------------------------------------------
# Prediction reliability checks
# ---------------------------------------------------------------------------

def repeated_prediction_consistency(tool, inputs, feature_set, repeats=DEFAULT_REPEATS):
    """Run the tool ``repeats`` times on identical inputs and compare values."""
    values = []
    for _ in range(repeats):
        res = tool.predict(inputs, feature_set=feature_set)
        values.append(res["predicted_fuel_rate_lph"])
    return {
        "repeats": int(repeats),
        "values": values,
        "all_identical": all(v == values[0] for v in values),
        "value": values[0],
    }


def check_finite_valid(result) -> dict:
    """A successful tool result must carry a finite, non-negative prediction."""
    value = result.get("predicted_fuel_rate_lph")
    finite = isinstance(value, (int, float)) and np.isfinite(float(value))
    nonneg = finite and float(value) >= 0.0
    ok_status = result.get("status") == "ok"
    return {
        "status_ok": bool(ok_status),
        "finite": bool(finite),
        "non_negative": bool(nonneg),
        "pass": bool(ok_status and finite and nonneg),
    }


def prediction_agreement_with_direct_model(tool, inputs, feature_set, tolerance=1e-4) -> dict:
    """Tool prediction must equal the direct saved-model prediction."""
    tool_res = tool.predict(inputs, feature_set=feature_set)
    tool_val = tool_res["predicted_fuel_rate_lph"]
    model_val = direct_model_prediction(feature_set, inputs)
    deviation = abs(float(tool_val) - float(model_val))
    return {
        "tool_value": float(tool_val),
        "direct_model_value": float(model_val),
        "abs_deviation": deviation,
        "within_tolerance": deviation <= tolerance,
        "tolerance": tolerance,
    }


INVALID_CASES = {
    "missing_field": lambda fs, i: {k: v for k, v in i.items()
                                     if k != list(i)[0]},
    "out_of_range": lambda fs, i: {**i, "speed_kph": 500.0},
    "non_numeric": lambda fs, i: {**i, "torque_nm": "fast"},
    "boolean_as_number": lambda fs, i: {**i, "payload_kg": True},
    "unknown_field": lambda fs, i: {**i, "not_a_feature": 1.0},
    "non_finite": lambda fs, i: {**i, "engine_rpm": float("nan")},
    "wrong_feature_set_shape": lambda fs, i: canonical_inputs(
        "telematics_only" if fs == "full_operational" else "full_operational"
    ),
}


def check_invalid_input_rejection(tool, feature_set, inputs) -> dict:
    """Every invalid case must raise ToolInputError and never return 'ok'."""
    results = {}
    for name, make in INVALID_CASES.items():
        try:
            bad = make(feature_set, inputs)
            res = tool.predict(bad, feature_set=feature_set)
            # If it somehow returned, it must not be a successful prediction.
            results[name] = {
                "raised_tool_input_error": False,
                "returned_status": res.get("status"),
                "rejected": False,
            }
        except ToolInputError:
            results[name] = {
                "raised_tool_input_error": True,
                "returned_status": None,
                "rejected": True,
            }
        except Exception as exc:  # noqa: BLE001 - classify any other failure
            results[name] = {
                "raised_tool_input_error": False,
                "raised_other": type(exc).__name__,
                "returned_status": None,
                "rejected": True,
            }
    return {
        "n_cases": len(results),
        "n_rejected": sum(1 for r in results.values() if r["rejected"]),
        "n_raised_tool_input_error": sum(
            1 for r in results.values() if r["raised_tool_input_error"]
        ),
        "cases": results,
    }


# ---------------------------------------------------------------------------
# Explanation reliability checks
# ---------------------------------------------------------------------------

def check_explanation_reconstruction(expl_result) -> dict:
    """SHAP reconstruction must pass within the documented tolerance."""
    recon = expl_result.get("reconstruction", {})
    passed = bool(recon.get("pass"))
    deviation = recon.get("max_abs_deviation")
    tol = recon.get("tolerance", RECONSTRUCTION_TOLERANCE)
    within = isinstance(deviation, (int, float)) and float(deviation) <= float(tol)
    return {
        "pass": passed,
        "max_abs_deviation": deviation,
        "tolerance": tol,
        "within_tolerance": bool(within),
        "ok": bool(passed and within),
    }


def check_attribution_direction(expl_result) -> dict:
    """Positive list must be >= 0 and negative list <= 0, both non-empty."""
    pos = expl_result.get("most_influential_features", {}).get("top_positive_shap", [])
    neg = expl_result.get("most_influential_features", {}).get("top_negative_shap", [])
    pos_ok = bool(pos) and all(p.get("shap", 0) >= 0 for p in pos)
    neg_ok = bool(neg) and all(n.get("shap", 0) <= 0 for n in neg)
    return {
        "positive_nonempty": bool(pos),
        "negative_nonempty": bool(neg),
        "positive_sign_ok": bool(pos_ok),
        "negative_sign_ok": bool(neg_ok),
        "ok": bool(pos_ok and neg_ok and pos and neg),
    }


def check_explanation_safety(expl_result) -> dict:
    """Must carry non-causal note, instance scope, synthetic disclaimer, and no
    guaranteed-savings claim."""
    lim = expl_result.get("limitations", {})
    prov = expl_result.get("provenance", {})
    text = str(expl_result)
    no_guarantee = "guarantee" not in text.lower() or "not a guarantee" in text.lower()
    return {
        "shap_is_not_causal": bool(lim.get("shap_is_not_causal")),
        "instance_scope_note": bool(lim.get("instance_scope")),
        "synthetic_training_disclosed": bool(prov.get("synthetic_training")),
        "no_guaranteed_savings_claim": bool(no_guarantee),
        "ok": bool(
            lim.get("shap_is_not_causal")
            and prov.get("synthetic_training")
            and no_guarantee
        ),
    }


def prediction_explanation_consistency(pred_result, expl_result, tolerance=1e-6) -> dict:
    """The same inputs must yield the same predicted fuel rate from both tools."""
    a = pred_result["predicted_fuel_rate_lph"]
    b = expl_result["predicted_fuel_rate_lph"]
    deviation = abs(float(a) - float(b))
    return {
        "prediction_tool_value": float(a),
        "explanation_tool_value": float(b),
        "abs_deviation": deviation,
        "within_tolerance": deviation <= tolerance,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# Segment classification (mirrors M1.3 condition bins for the raw inputs)
# ---------------------------------------------------------------------------

def classify_segment(inputs: dict) -> str:
    """Return an operating segment label from the (validated) raw inputs."""
    speed = float(inputs.get("speed_kph", 0.0))
    accel = float(inputs.get("acceleration_mps2", 0.0))
    grade = float(inputs.get("road_grade_pct", 0.0))
    ign = int(inputs.get("ignition_status", 0))
    if speed < 0.5 and ign == 1:
        return "idle_stopped"
    if grade >= 0.3:
        return "uphill"
    if accel >= 0.5:
        return "acceleration"
    if speed >= 60.0:
        return "highway"
    return "other"


# ---------------------------------------------------------------------------
# Deterministic small sample drawn from the held-out test set
# ---------------------------------------------------------------------------

def sample_test_inputs(test_df: pd.DataFrame, prices_segment: int = 6, seed: int = SEED):
    """Draw a deterministic, small per-segment sample of raw input dicts.

    Returns ``{segment: [inputs_dict, ...]}``. The sample is small by design:
    it is used only for tool-vs-model agreement evidence, not to duplicate the
    M1.3 metric tables.
    """
    rng = np.random.RandomState(seed)
    out: dict[str, list[dict]] = {}
    for seg in ("highway", "acceleration", "idle_stopped", "uphill"):
        sub = test_df[test_df.apply(lambda r: classify_segment({
            "speed_kph": r["speed_kph"],
            "acceleration_mps2": r["acceleration_mps2"],
            "road_grade_pct": r["road_grade_pct"],
            "ignition_status": r["ignition_status"],
        }) == seg, axis=1)]
        n = min(prices_segment, len(sub))
        idx = rng.choice(np.arange(len(sub)), size=n, replace=False)
        rows = sub.iloc[idx]
        out[seg] = [
            {c: row[c] for c in row.index}
            for _, row in rows.iterrows()
        ]
    return out
