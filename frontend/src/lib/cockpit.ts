import type { FeatureSetName } from "../api/types";

// Canonical full_operational field order (matches the feature spec).
export const FULL_FIELDS = [
  "vehicle_class",
  "gross_vehicle_weight_kg",
  "payload_kg",
  "vehicle_mass_kg",
  "speed_kph",
  "acceleration_mps2",
  "road_grade_pct",
  "ignition_status",
  "engine_rpm",
  "engine_load_pct",
  "throttle_position_pct",
  "torque_nm",
  "coolant_temp_c",
  "intake_air_temp_c",
  "ambient_temp_c",
] as const;

// The five "manual engine controls" mapped to gauges/sliders.
export const MANUAL_CONTROLS = [
  "engine_rpm",
  "speed_kph",
  "engine_load_pct",
  "throttle_position_pct",
  "road_grade_pct",
] as const;

// Everything else in the full set lives in the Advanced Vehicle Input panel.
export const ADVANCED_FIELDS = FULL_FIELDS.filter(
  (f) => !(MANUAL_CONTROLS as readonly string[]).includes(f),
);

export const DEFAULT_FEATURE_SET: FeatureSetName = "full_operational";

// Complete, in-range default input set (matches the validated test payload).
export const DEFAULT_INPUTS: Record<string, string | number> = {
  vehicle_class: "40T",
  gross_vehicle_weight_kg: 36000,
  payload_kg: 12000,
  vehicle_mass_kg: 24000,
  speed_kph: 80,
  acceleration_mps2: 0.2,
  road_grade_pct: 1.5,
  ignition_status: 1,
  engine_rpm: 1400,
  engine_load_pct: 55,
  throttle_position_pct: 40,
  torque_nm: 1600,
  coolant_temp_c: 88,
  intake_air_temp_c: 30,
  ambient_temp_c: 22,
};

// Deterministic query that triggers the combined predict+explain intent so a
// single "Run AI Prediction" press returns both the fuel-rate estimate and the
// instance-level SHAP explanation.
export const PREDICT_QUERY = "predict fuel rate and explain what drives it";

export interface Scenario {
  id: string;
  label: string;
  description: string;
  overrides: Record<string, number>;
}

// Predefined driving scenarios. They override the *manual controls* (and the
// closely coupled acceleration/grade/torque) and leave the truck's static
// attributes (class, mass, payload, temperatures) at the user's current values.
// All values are within the validated synthetic ranges.
export const SCENARIOS: Scenario[] = [
  {
    id: "idle",
    label: "Idle",
    description: "Engine at idle, stationary on level ground.",
    overrides: {
      engine_rpm: 700,
      speed_kph: 0,
      engine_load_pct: 15,
      throttle_position_pct: 10,
      road_grade_pct: 0,
      acceleration_mps2: 0,
      torque_nm: 400,
    },
  },
  {
    id: "city",
    label: "City driving",
    description: "Low speed, frequent stops, light grade.",
    overrides: {
      engine_rpm: 1100,
      speed_kph: 35,
      engine_load_pct: 45,
      throttle_position_pct: 30,
      road_grade_pct: 0.5,
      acceleration_mps2: 0.3,
      torque_nm: 900,
    },
  },
  {
    id: "highway",
    label: "Highway cruising",
    description: "Steady cruise near legal speed.",
    overrides: {
      engine_rpm: 1500,
      speed_kph: 85,
      engine_load_pct: 60,
      throttle_position_pct: 45,
      road_grade_pct: 0.2,
      acceleration_mps2: 0.05,
      torque_nm: 1500,
    },
  },
  {
    id: "acceleration",
    label: "Acceleration",
    description: "Hard pull from a lower speed.",
    overrides: {
      engine_rpm: 2000,
      speed_kph: 55,
      engine_load_pct: 85,
      throttle_position_pct: 75,
      road_grade_pct: 0,
      acceleration_mps2: 0.9,
      torque_nm: 2200,
    },
  },
  {
    id: "uphill",
    label: "Uphill driving",
    description: "Climbing a significant grade with load.",
    overrides: {
      engine_rpm: 1900,
      speed_kph: 45,
      engine_load_pct: 90,
      throttle_position_pct: 80,
      road_grade_pct: 3.5,
      acceleration_mps2: 0.15,
      torque_nm: 2350,
    },
  },
  {
    id: "downhill",
    label: "Downhill driving",
    description: "Descending; engine braking, light throttle.",
    overrides: {
      engine_rpm: 950,
      speed_kph: 60,
      engine_load_pct: 12,
      throttle_position_pct: 8,
      road_grade_pct: -3,
      acceleration_mps2: -0.4,
      torque_nm: 250,
    },
  },
];

export interface FieldGroup {
  name: string;
  kind: "numeric" | "categorical" | "binary";
  min: number;
  max: number;
  step: number;
}

/** Build the structured model-input payload for the selected feature set. */
export function buildPayload(
  inputs: Record<string, string | number>,
  featureSet: FeatureSetName,
): Record<string, string | number> {
  const allowed = new Set(
    featureSet === "telematics_only" ? FULL_FIELDS.filter(
      (f) => !["engine_rpm", "engine_load_pct", "throttle_position_pct", "torque_nm"].includes(f),
    ) : FULL_FIELDS,
  );
  const out: Record<string, string | number> = {};
  for (const f of FULL_FIELDS) {
    if (allowed.has(f) && f in inputs) {
      out[f] = inputs[f];
    }
  }
  return out;
}

export function formatNumber(value: number, digits = 0): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
