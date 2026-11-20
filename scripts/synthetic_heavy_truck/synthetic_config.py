"""Shared configuration for the synthetic heavy-duty truck research dataset suite.

This module defines the physical constants, vehicle classes, route profiles and
generation parameters used by the generator and validation scripts. Everything is
synthetic; no value here is claimed to represent real measured truck telemetry.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Global generation parameters
# ---------------------------------------------------------------------------

# Fixed seed -> fully reproducible output. The same seed reproduces the same
# files byte-for-byte (given the same numpy/pandas versions).
SEED = 20240617

# Exact number of telemetry records to produce.
TOTAL_ROWS = 963_697

# Fixed sampling interval (seconds). All telemetry is on a regular 1 Hz grid
# within each trip. Fuel increments are computed from this interval.
SAMPLING_INTERVAL_S = 1

# Fleet size (number of synthetic trucks).
NUM_TRUCKS = 29

# Truck-wise split sizes (must sum to NUM_TRUCKS).
SPLIT_TRAIN = 20
SPLIT_VAL = 4
SPLIT_TEST = 5

# Output directory (workspace-relative).
OUTPUT_DIR = "data/synthetic_heavy_truck_research"

# Physical constants
GRAVITY_MPS2 = 9.81
AIR_DENSITY_KG_M3 = 1.225
DIESEL_ENERGY_J_PER_L = 36.0e6   # approximate lower-heating-value energy density
DRIVETRAIN_EFFICIENCY = 0.85     # mechanical power delivered to wheels / engine power

# Engine internal parasitic power (friction + pumping + auxiliaries) at idle,
# expressed per litre of engine displacement (kW/L). This is the physical
# basis for stationary idle fuel consumption: at idle the engine must still
# overcome these internal losses, producing ~1-3 L/h depending on engine size.
IDLE_PARASITIC_KW_PER_L = 0.45

# ---------------------------------------------------------------------------
# Vehicle classes (synthetic reference values)
# ---------------------------------------------------------------------------
# gvw_kg           : rated gross vehicle weight
# rated_power_kw   : rated (max continuous) engine power
# displacement_l    : engine displacement
# cda_m2           : aerodynamic drag area (Cd * frontal area)
# crr              : rolling-resistance coefficient
# idle_rpm / rated_rpm: engine speed band
#
# Note: fuel at idle is derived from engine internal parasitic power (see
# IDLE_PARASITIC_KW_PER_L), not from a fixed per-litre L/h additive.
CLASS_SPECS = {
    "32T": dict(
        gvw_kg=32_000, rated_power_kw=275, displacement_l=9.0,
        cda_m2=6.9, crr=0.0095,
        idle_rpm=620, rated_rpm=1900,
    ),
    "40T": dict(
        gvw_kg=40_000, rated_power_kw=345, displacement_l=11.0,
        cda_m2=7.9, crr=0.0092,
        idle_rpm=620, rated_rpm=1900,
    ),
    "49T": dict(
        gvw_kg=49_000, rated_power_kw=415, displacement_l=13.0,
        cda_m2=8.7, crr=0.0090,
        idle_rpm=620, rated_rpm=1900,
    ),
}

# Fleet composition: 29 trucks across classes (order is fixed and reproducible).
FLEET_CLASSES = (
    ["32T"] * 10 + ["40T"] * 11 + ["49T"] * 8
)

# ---------------------------------------------------------------------------
# Route profiles (synthetic)
# ---------------------------------------------------------------------------
# cruise_kph     : target cruising speed
# vmax_kph       : route speed cap
# sigma_kph      : speed noise around cruise
# stops_per_hour : expected number of full stops per hour of driving
# gmax           : max |road grade| in %
# gsigma         : grade random-walk volatility in % (random_walk mode)
# grade_mode      : "random_walk" (smooth noise) or "segments" (deterministic
#                   smooth uphill/downhill segments with realistic magnitudes)
# stop_dur       : (min, max) stop duration seconds
ROUTE_SPECS = {
    "highway": dict(
        cruise_kph=82, vmax_kph=92, sigma_kph=1.4, stops_per_hour=1.2,
        gmax=3.0, gsigma=0.22, stop_dur=(15, 45), grade_mode="segments",
    ),
    "urban": dict(
        cruise_kph=48, vmax_kph=60, sigma_kph=2.6, stops_per_hour=7.0,
        gmax=1.5, gsigma=0.35, stop_dur=(15, 50),
    ),
    "city": dict(
        cruise_kph=28, vmax_kph=40, sigma_kph=2.2, stops_per_hour=14.0,
        gmax=1.0, gsigma=0.30, stop_dur=(20, 55),
    ),
    "mountainous": dict(
        cruise_kph=42, vmax_kph=58, sigma_kph=2.2, stops_per_hour=2.5,
        gmax=6.0, gsigma=1.25, stop_dur=(20, 50),
    ),
    "mixed": dict(
        cruise_kph=55, vmax_kph=72, sigma_kph=2.3, stops_per_hour=4.0,
        gmax=3.0, gsigma=0.70, stop_dur=(15, 45),
    ),
}

ROUTE_NAMES = list(ROUTE_SPECS.keys())
# Equal route probability for all classes in this suite (documented).
ROUTE_PROBABILITIES = [1.0 / len(ROUTE_NAMES)] * len(ROUTE_NAMES)

# Runtime tolerance for the power-feasibility validation check: the generated
# tractive-power demand (per record) must not exceed rated power by more than
# this fraction. The motion solver is exact to ~0.1%, so a 2% tolerance only
# absorbs floating-point / discretization noise - never a real over-demand.
POWER_FEASIBILITY_TOLERANCE = 0.02

# ---------------------------------------------------------------------------
# Telemetry schema (ordered as written to the master CSV)
# ---------------------------------------------------------------------------
TELEMETRY_COLUMNS = [
    "record_id",                  # unique integer id
    "timestamp_utc",              # ISO-8601 UTC string
    "truck_id",                   # truck identifier string
    "trip_id",                    # trip identifier string
    "vehicle_class",              # 32T / 40T / 49T
    "gross_vehicle_weight_kg",
    "payload_kg",
    "vehicle_mass_kg",
    "speed_kph",
    "acceleration_mps2",
    "road_grade_pct",
    "latitude",
    "longitude",
    "ignition_status",            # 1 engine on, 0 engine off
    "engine_rpm",
    "engine_load_pct",
    "throttle_position_pct",
    "torque_nm",
    "coolant_temp_c",
    "intake_air_temp_c",
    "ambient_temp_c",
    "fuel_rate_lph",              # TARGET
    "fuel_consumed_l_increment",
    "cumulative_fuel_l_trip",
]

# Target variable
TARGET_COLUMN = "fuel_rate_lph"

# Features that are derived from / leak the target and MUST NOT be used as
# model inputs for instantaneous fuel-rate prediction.
EXCLUDED_FEATURES = [
    "fuel_consumed_l_increment",
    "cumulative_fuel_l_trip",
]

# Recommended input-feature allowlist (no target derivative, no future info).
RECOMMENDED_FEATURES = [
    "truck_id",
    "vehicle_class",
    "gross_vehicle_weight_kg",
    "payload_kg",
    "vehicle_mass_kg",
    "speed_kph",
    "acceleration_mps2",
    "road_grade_pct",
    "latitude",
    "longitude",
    "ignition_status",
    "engine_rpm",
    "engine_load_pct",
    "throttle_position_pct",
    "torque_nm",
    "coolant_temp_c",
    "intake_air_temp_c",
    "ambient_temp_c",
]

# Reference ordering of topic categories for the agent-evaluation dataset.
AGENT_CATEGORIES = [
    "current_telemetry_retrieval",
    "fuel_rate_prediction",
    "fuel_consumption_analysis",
    "trip_summarization",
    "cross_truck_comparison",
    "shap_explanation",
    "driver_feedback",
    "rag_evidence_retrieval",
    "model_reliability_limitations",
    "missing_or_invalid_telemetry",
]
