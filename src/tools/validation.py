"""Shared input validation for the agent tools (M2.2).

This module centralises the rules that turn raw user/programmatic inputs into a
validated, typed record that the fuel-rate prediction and explanation tools
can consume. It is intentionally strict: missing or invalid required fields
raise ``ToolInputError`` rather than being silently invented or defaulted.

Validation bounds are grounded in the observed synthetic telemetry ranges (see
``scripts/_derive_feature_bounds.py``) with a small physical margin so that
sensible operational values are accepted while clearly out-of-domain values are
rejected.
"""

from __future__ import annotations

from typing import Any

from src.features.spec import get_feature_set, load_feature_spec

# Canonical feature sets supported by the M1.2 validation-selected models.
FEATURE_SETS: tuple[str, ...] = ("full_operational", "telematics_only")

# Vehicle classes used across the corpus (must match the one-hot encoder's
# training categories so ``handle_unknown="ignore"`` is never exercised).
VEHICLE_CLASSES: tuple[str, ...] = ("32T", "40T", "49T")

# Permissible numeric ranges (min, max), inclusive. Grounded in the synthetic
# telemetry with a small documented margin beyond the observed extrema.
FIELD_RANGES: dict[str, tuple[float, float]] = {
    "gross_vehicle_weight_kg": (32000.0, 49000.0),
    "payload_kg": (0.0, 20000.0),
    "vehicle_mass_kg": (18000.0, 47000.0),
    "speed_kph": (0.0, 100.0),
    "acceleration_mps2": (-2.0, 1.5),
    "road_grade_pct": (-4.0, 4.0),
    "engine_rpm": (0.0, 2200.0),
    "engine_load_pct": (0.0, 100.0),
    "throttle_position_pct": (0.0, 100.0),
    "torque_nm": (0.0, 2400.0),
    "coolant_temp_c": (-10.0, 100.0),
    "intake_air_temp_c": (-10.0, 70.0),
    "ambient_temp_c": (-15.0, 50.0),
}

# Ignition is a binary flag in the telemetry.
IGNITION_ALLOWED: tuple[int, ...] = (0, 1)


class ToolInputError(ValueError):
    """Raised when tool inputs are missing, malformed, or out of range."""


def normalize_feature_set(feature_set: str) -> str:
    """Return a canonical feature-set name or raise ``ToolInputError``."""
    if feature_set not in FEATURE_SETS:
        raise ToolInputError(
            f"unknown feature set {feature_set!r}; available: {list(FEATURE_SETS)}"
        )
    return feature_set


def _as_float(value: Any, field: str) -> float:
    """Parse a numeric token without silently coercing booleans."""
    if isinstance(value, bool):
        raise ToolInputError(f"field {field!r} must be numeric, got boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            raise ToolInputError(
                f"field {field!r} is not a valid number: {value!r}"
            ) from None
    raise ToolInputError(
        f"field {field!r} has unsupported type {type(value).__name__}"
    )


def _check_range(field: str, value: float) -> None:
    lo, hi = FIELD_RANGES[field]
    if not (lo <= value <= hi):
        raise ToolInputError(
            f"field {field!r} value {value} is outside the permissible range "
            f"[{lo}, {hi}]"
        )


def validate_vehicle_inputs(
    inputs: dict[str, Any],
    feature_set: str = "full_operational",
) -> tuple[str, dict[str, Any]]:
    """Validate raw inputs and return ``(feature_set, cleaned_inputs)``.

    ``cleaned_inputs`` contains exactly the feature-set columns in spec order,
    coerced to ``float`` (or the raw validated categorical for
    ``vehicle_class`` / ``ignition_status``).

    Raises ``ToolInputError`` for unknown columns, missing required fields,
    malformed numbers, and out-of-range values.
    """
    fs = normalize_feature_set(feature_set)
    spec = load_feature_spec()
    features = get_feature_set(spec, fs)

    if inputs is None:
        raise ToolInputError("no vehicle inputs provided")
    if not isinstance(inputs, dict):
        raise ToolInputError(f"inputs must be a dict, got {type(inputs).__name__}")

    unknown = set(inputs) - set(features)
    if unknown:
        raise ToolInputError(
            f"unexpected input field(s): {sorted(unknown)}; expected fields for "
            f"feature set {fs!r}: {features}"
        )
    missing = set(features) - set(inputs)
    if missing:
        raise ToolInputError(
            f"missing required input field(s): {sorted(missing)} (feature set {fs!r})"
        )

    cleaned: dict[str, Any] = {}
    for field in features:
        raw = inputs[field]
        if field == "vehicle_class":
            if raw not in VEHICLE_CLASSES:
                raise ToolInputError(
                    f"field 'vehicle_class' must be one of {list(VEHICLE_CLASSES)}; "
                    f"got {raw!r}"
                )
            cleaned[field] = raw
            continue
        if field == "ignition_status":
            val = int(_as_float(raw, field))
            if val not in IGNITION_ALLOWED:
                raise ToolInputError(
                    f"field 'ignition_status' must be 0 or 1; got {val!r}"
                )
            cleaned[field] = val
            continue
        num = _as_float(raw, field)
        _check_range(field, num)
        cleaned[field] = num

    return fs, cleaned
