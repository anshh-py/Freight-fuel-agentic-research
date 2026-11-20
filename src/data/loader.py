"""Dataset loading and truck-wise split enforcement for the synthetic
heavy-truck research suite.

This module only reads data; it never writes or modifies the dataset.
The split manifest is treated as the single source of truth for the
train/validation/test truck assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Project root = three levels up from this file (src/data -> project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_TELEMETRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_heavy_truck_research"
    / "telemetry_master_963697.csv.gz"
)
DEFAULT_SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_heavy_truck_research"
    / "truckwise_split_manifest.csv"
)

# Minimal, cheap dtype map avoids object columns for the numeric signals.
TELEMETRY_DTYPES = {
    "record_id": "int64",
    "truck_id": "str",
    "trip_id": "str",
    "vehicle_class": "str",
    "gross_vehicle_weight_kg": "int64",
    "payload_kg": "int64",
    "vehicle_mass_kg": "int64",
    "speed_kph": "float64",
    "acceleration_mps2": "float64",
    "road_grade_pct": "float64",
    "latitude": "float64",
    "longitude": "float64",
    "ignition_status": "int64",
    "engine_rpm": "float64",
    "engine_load_pct": "float64",
    "throttle_position_pct": "float64",
    "torque_nm": "float64",
    "coolant_temp_c": "float64",
    "intake_air_temp_c": "float64",
    "ambient_temp_c": "float64",
    "fuel_rate_lph": "float64",
    "fuel_consumed_l_increment": "float64",
    "cumulative_fuel_l_trip": "float64",
}


def load_telemetry(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the compressed telemetry CSV.

    ``timestamp_utc`` is loaded as ``str`` on purpose: it is an identifier /
    ordering field, not a numeric feature, and we do not parse it here.
    """
    p = Path(path) if path is not None else DEFAULT_TELEMETRY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Telemetry not found: {p}")
    return pd.read_csv(p, compression="gzip", dtype=TELEMETRY_DTYPES)


def load_split_manifest(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the truck-wise split manifest."""
    p = Path(path) if path is not None else DEFAULT_SPLIT_MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"Split manifest not found: {p}")
    df = pd.read_csv(p)
    required = {"truck_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing)}")
    return df


@dataclass
class SplitResult:
    """Enforced, disjoint truck-wise split of the telemetry."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    manifest: pd.DataFrame
    train_trucks: list[str] = field(default_factory=list)
    val_trucks: list[str] = field(default_factory=list)
    test_trucks: list[str] = field(default_factory=list)

    @property
    def truck_splits(self) -> dict[str, list[str]]:
        return {
            "train": self.train_trucks,
            "val": self.val_trucks,
            "test": self.test_trucks,
        }

    @property
    def total_rows(self) -> int:
        return int(len(self.train) + len(self.val) + len(self.test))


def enforce_split(
    telemetry: pd.DataFrame,
    manifest: Optional[pd.DataFrame] = None,
    expected_counts: Optional[tuple[int, int, int]] = (20, 4, 5),
) -> SplitResult:
    """Split telemetry rows by the truck assignment in the manifest.

    Guarantees:
      * each truck appears in exactly one split (no overlap),
      * every manifest truck exists in telemetry,
      * every telemetry truck is present in the manifest,
      * all rows are accounted for (train+val+test == full size),
      * the per-split truck counts match ``expected_counts`` when provided.

    Raises ``ValueError`` on any violation. Rows are never shuffled across
    split boundaries.
    """
    if manifest is None:
        manifest = load_split_manifest()

    # Build the truck->split mapping, rejecting a truck assigned to more
    # than one split (a manifest-level overlap).
    mapping: dict[str, str] = {}
    for t, s in zip(manifest["truck_id"], manifest["split"]):
        if t in mapping and mapping[t] != s:
            raise ValueError(f"truck {t!r} is assigned to multiple splits")
        mapping[t] = s
    allowed_splits = set(mapping.values())

    # Every telemetry truck must be represented in the manifest.
    telemetry_trucks = set(telemetry["truck_id"].dropna().unique())
    unknown = telemetry_trucks - set(mapping)
    if unknown:
        raise ValueError(
            f"telemetry contains truck(s) absent from manifest: {sorted(unknown)}"
        )

    frames: dict[str, pd.DataFrame] = {"train": None, "val": None, "test": None}
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        if split_name not in allowed_splits:
            raise ValueError(f"Manifest does not contain split '{split_name}'")
        trucks = sorted(t for t in mapping if mapping[t] == split_name)
        mask = telemetry["truck_id"].isin(trucks)
        frames[split_name] = telemetry.loc[mask].copy()
        splits[split_name] = trucks

    # Zero overlap: a truck may not appear in more than one split.
    all_trucks = []
    for split_name in ("train", "val", "test"):
        all_trucks.extend(splits[split_name])
    if len(all_trucks) != len(set(all_trucks)):
        raise ValueError("truck overlap detected across splits (a truck in >1 split)")

    # All rows accounted for.
    total = len(telemetry)
    accounted = sum(len(frames[s]) for s in ("train", "val", "test"))
    if accounted != total:
        raise ValueError(
            f"split row accounting mismatch: {accounted} != dataset {total}"
        )

    # Optional manifest truck-count check.
    if expected_counts is not None:
        exp_train, exp_val, exp_test = expected_counts
        if (
            len(splits["train"]) != exp_train
            or len(splits["val"]) != exp_val
            or len(splits["test"]) != exp_test
        ):
            raise ValueError(
                "manifest truck counts do not match expected "
                f"{expected_counts}: "
                f"train={len(splits['train'])}, val={len(splits['val'])}, "
                f"test={len(splits['test'])}"
            )

    return SplitResult(
        train=frames["train"],
        val=frames["val"],
        test=frames["test"],
        manifest=manifest,
        train_trucks=splits["train"],
        val_trucks=splits["val"],
        test_trucks=splits["test"],
    )
