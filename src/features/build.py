"""Feature-matrix construction with strict allowlist/denylist enforcement.

Builds ``X`` (features) and ``y`` (target) for a given feature set while
guaranteeing:

  * forbidden columns (identifiers, timestamps, GPS, target-derived,
    proxy columns) are never present in ``X``,
  * every required feature from the spec actually exists in the input frame,
  * no additional column is ever silently included,
  * the target ``fuel_rate_lph`` is separated from ``X``.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.features.spec import (
    get_denylist,
    get_feature_set,
    get_target_column,
    validate_spec,
)


class FeatureValidationError(ValueError):
    """Raised when the feature contract is violated."""


def known_column_superset(spec: dict[str, Any]) -> set[str]:
    """The set of all columns we expect to see in a telemetry frame.

    = union of both feature sets + target + denylisted columns. Anything
    outside this set is an unexpected column (rejected).
    """
    known: set[str] = set()
    for set_name, cfg in spec.get("feature_sets", {}).items():
        known.update(cfg.get("features", []))
    known.add(get_target_column(spec))
    known.update(get_denylist(spec))
    return known


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "full_operational",
    spec: Optional[dict[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return ``(X, y, feature_names)`` for a configured feature set.

    Parameters
    ----------
    df : frame of telemetry columns (must include the target).
    feature_set : name in spec["feature_sets"] (default full_operational).
    spec : optional feature spec dict; loaded from disk if None.

    Returns
    -------
    X : DataFrame of exactly the spec'd features, in spec order.
    y : Series of the target, index-aligned with X.
    feature_names : the ordered feature list used.
    """
    if spec is None:
        from src.features.spec import load_feature_spec
        spec = load_feature_spec()
    validate_spec(spec)

    target = get_target_column(spec)
    denied = set(get_denylist(spec))
    features = get_feature_set(spec, feature_set)

    # 1. Unexpected column guard: frame must only contain known columns.
    known = known_column_superset(spec)
    unexpected = [c for c in df.columns if c not in known]
    if unexpected:
        raise FeatureValidationError(
            f"unexpected column(s) in frame: {unexpected}. Refuse to build "
            f"features from a frame containing undeclared columns."
        )

    # 2. Target must be present to separate it.
    if target not in df.columns:
        raise FeatureValidationError(f"target {target!r} not present in frame")

    # 3. The selected feature set must not request any forbidden column.
    bad = [f for f in features if f in denied or f == target]
    if bad:
        raise FeatureValidationError(
            f"forbidden feature(s) requested for set {feature_set!r}: {bad}"
        )

    # 4. All required features must exist.
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise FeatureValidationError(
            f"missing required feature(s) for set {feature_set!r}: {missing}"
        )

    # 5. Build X explicitly from the spec list — never from df.columns.
    X = df[features].copy()
    y = df[target]

    # Final safety: X columns must equal the spec exactly, in order.
    if list(X.columns) != features:
        raise FeatureValidationError(
            f"X columns {list(X.columns)} do not match spec {features}"
        )

    return X, y, features
