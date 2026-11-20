"""Tool 1 — Fuel-rate prediction (M2.2).

Wraps the M1.2 validation-selected XGBoost models (``full_operational`` and
``telematics_only``) behind an explicit callable interface. It loads the saved
model + preprocessor from disk (never refits, never retrains), validates the
operational inputs via :mod:`src.tools.validation`, and returns a predicted
fuel rate in L/h.

The prediction is an *estimate from a synthetic-trained model*; the tool labels
this explicitly and never presents it as a validated real-world fuel
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.explainability import load_selected_xgb
from src.features.spec import get_feature_set, load_feature_spec
from src.tools.validation import ToolInputError, validate_vehicle_inputs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "reports" / "m1" / "artifacts"

DEFAULT_FEATURE_SET = "full_operational"


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails at runtime (e.g. missing artifact)."""


class FuelRatePredictionTool:
    """Predict fuel rate from validated operational vehicle inputs."""

    #: Safety note attached to every prediction result.
    SAFETY_NOTE = (
        "Predicted fuel rate is an estimate from a model trained on SYNTHETIC "
        "telemetry. It is not a validated real-world fuel measurement and "
        "must not be used for billing, compliance, or vehicle performance "
        "claims."
    )

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
        feature_set: str = DEFAULT_FEATURE_SET,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.feature_set = feature_set
        self._loaded: dict[str, dict] | None = None

    # -- loading ----------------------------------------------------------
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

    def available_feature_sets(self) -> list[str]:
        return ["full_operational", "telematics_only"]

    # -- prediction -------------------------------------------------------
    def predict(
        self,
        inputs: dict[str, Any],
        feature_set: str | None = None,
    ) -> dict[str, Any]:
        """Validate inputs and return a predicted fuel rate (L/h)."""
        fs = feature_set or self.feature_set
        fs, cleaned = validate_vehicle_inputs(inputs, fs)

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

        return {
            "status": "ok",
            "predicted_fuel_rate_lph": round(predicted, 4),
            "units": "L/h",
            "feature_set": fs,
            "model": {
                "name": "xgboost",
                "artifact": str(self.model_dir / fs / "xgboost.joblib"),
            },
            "preprocessor": {
                "artifact": str(self.model_dir / fs / "preprocessor.joblib"),
                "transformed_columns": columns,
            },
            "input_fields_used": {k: v for k, v in cleaned.items()},
            "provenance": {
                "synthetic_training": True,
                "model_is_estimate": True,
                "note": self.SAFETY_NOTE,
            },
        }
