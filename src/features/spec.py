"""Feature-spec loading and validation for the M1 synthetic fuel-rate task.

The single source of truth for the feature contract is
``configs/ml_synthetic_feature_spec.json``. This module loads it, exposes
the allowlist / denylist, and provides helpers used by the feature builder
to reject forbidden or unexpected columns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPEC_PATH = PROJECT_ROOT / "configs" / "ml_synthetic_feature_spec.json"


def load_feature_spec(path: Optional[Path] = None) -> dict[str, Any]:
    """Load the feature-spec JSON."""
    p = Path(path) if path is not None else DEFAULT_SPEC_PATH
    if not p.exists():
        raise FileNotFoundError(f"Feature spec not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_feature_set(spec: dict[str, Any], name: str) -> list[str]:
    """Return the ordered feature list for a configured feature set."""
    sets = spec.get("feature_sets", {})
    if name not in sets:
        raise ValueError(
            f"Unknown feature set {name!r}; available: {sorted(sets)}"
        )
    feats = sets[name].get("features", [])
    if not feats:
        raise ValueError(f"Feature set {name!r} has an empty feature list")
    return list(feats)


def get_denylist(spec: dict[str, Any]) -> list[str]:
    """Combine identifiers, target-derived and proxy columns into a denylist."""
    denied: list[str] = []
    denied += list(spec.get("id_columns_excluded", []))
    denied += list(spec.get("target_derived_excluded", []))
    denied += list(spec.get("proxy_excluded", []))
    return list(dict.fromkeys(denied))  # dedupe, preserve order


def get_target_column(spec: dict[str, Any]) -> str:
    target = spec.get("target_column")
    if not target:
        raise ValueError("Feature spec is missing 'target_column'")
    return target


def validate_spec(spec: dict[str, Any]) -> None:
    """Structural checks over the spec so downstream code fails fast."""
    target = get_target_column(spec)
    denied = get_denylist(spec)

    if target in denied:
        raise ValueError(f"target {target!r} must not be on the denylist")

    # A feature must never collide with the denylist (incl. target itself).
    for set_name, feats in spec.get("feature_sets", {}).items():
        for f in feats.get("features", []):
            if f in denied:
                raise ValueError(
                    f"feature {f!r} in set {set_name!r} is on the denylist"
                )
            if f == target:
                raise ValueError(
                    f"feature {f!r} in set {set_name!r} equals the target"
                )
