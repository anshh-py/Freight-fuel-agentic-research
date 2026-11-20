"""Tool 3 — Instance-level prediction explanation (M2.2).

Explains a single fuel-rate prediction using the existing M1.4 SHAP
implementation (:mod:`src.evaluation.explainability`) and the saved M1.2 model.
It computes true **instance-level** SHAP contributions for the one input record
that the prediction tool validated, then returns the most influential features
with their signed contributions.

Safety contract (never violated here):
  * only instance-level SHAP is returned; global feature importance is never
    substituted or mislabelled as an instance explanation,
  * SHAP attribution is explicitly distinguished from causality,
  * the numeric contributions are computed from the SHAP tree explainer and
    validated to reconstruct the model output — no invented values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.explainability import (
    compute_shap_values,
    load_selected_xgb,
    map_readable_names,
    top_shap_contributions,
    validate_reconstruction,
)
from src.features.spec import get_feature_set, load_feature_spec
from src.tools.fuel_rate_prediction import DEFAULT_MODEL_DIR, ToolExecutionError
from src.tools.validation import validate_vehicle_inputs

DEFAULT_FEATURE_SET = "full_operational"
TOP_FEATURES = 5


class PredictionExplanationTool:
    """Instance-level SHAP explanation of a single prediction."""

    CAUSAL_NOTE = (
        "SHAP attribution describes how the model's prediction is composed from "
        "its input features. It is not a causal explanation and does not "
        "imply a real-world causal effect."
    )
    INSTANCE_NOTE = (
        "This is an instance-level explanation computed from SHAP for the "
        "single input record, not a global feature-importance ranking."
    )

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        feature_set: str = DEFAULT_FEATURE_SET,
        top_features: int = TOP_FEATURES,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.feature_set = feature_set
        self.top_features = top_features
        self._loaded: dict[str, dict] | None = None

    def _ensure_loaded(self) -> dict[str, dict]:
        if self._loaded is None:
            if not self.model_dir.exists():
                raise ToolExecutionError(
                    f"model artifact directory not found: {self.model_dir}"
                )
            try:
                self._loaded = load_selected_xgb(
                    report_dir=self.model_dir.parent,
                    feature_sets=["full_operational", "telematics_only"],
                )
            except FileNotFoundError as exc:
                raise ToolExecutionError(str(exc)) from exc
        return self._loaded

    def explain(
        self,
        inputs: dict[str, Any],
        feature_set: str | None = None,
        top_features: int | None = None,
    ) -> dict[str, Any]:
        """Explain a single prediction for ``inputs`` and return top SHAP contributions."""
        fs = feature_set or self.feature_set
        fs, cleaned = validate_vehicle_inputs(inputs, fs)
        k = top_features or self.top_features

        loaded = self._ensure_loaded()
        entry = loaded[fs]
        estimator = entry["model"]
        preprocessor = entry["preprocessor"]
        columns = entry["columns"]

        spec = load_feature_spec()
        features = get_feature_set(spec, fs)
        row = pd.DataFrame([cleaned], columns=features)
        X = preprocessor.transform(row)
        if list(X.columns) != columns:
            raise ToolExecutionError("transformed column layout mismatch")

        predicted = float(estimator.predict(X)[0])
        shap_values, expected_value = compute_shap_values(estimator, X)
        recon = validate_reconstruction(expected_value, shap_values, np.asarray([predicted]))
        if not recon["pass"]:
            raise ToolExecutionError(f"SHAP reconstruction failed: {recon}")

        readable = map_readable_names(columns)
        positive, negative = top_shap_contributions(
            shap_values, columns, readable, 0, k=k
        )

        return {
            "status": "ok",
            "predicted_fuel_rate_lph": round(predicted, 4),
            "feature_set": fs,
            "model": {
                "name": "xgboost",
                "artifact": str(self.model_dir / fs / "xgboost.joblib"),
            },
            "explanation_kind": "instance_level_shap",
            "summary_statistic": "per-instance SHAP contribution (not mean |SHAP|)",
            "top_features": k,
            "expected_value": round(float(expected_value), 4),
            "reconstruction": {
                "pass": recon["pass"],
                "max_abs_deviation": round(recon["max_abs_deviation"], 6),
                "tolerance": recon["tolerance"],
            },
            "most_influential_features": {
                "top_positive_shap": positive,
                "top_negative_shap": negative,
            },
            "limitations": {
                "shap_is_not_causal": True,
                "note": self.CAUSAL_NOTE,
                "instance_scope": self.INSTANCE_NOTE,
            },
            "provenance": {
                "synthetic_training": True,
                "model_is_estimate": True,
                "shap_sh_library": "shap.TreeExplainer",
            },
        }
