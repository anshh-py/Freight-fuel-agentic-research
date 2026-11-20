// Shared API types mirroring the M6.1 gateway / M6.2 HTTP contract.

export type FeatureSetName = "full_operational" | "telematics_only";

export interface FieldMeta {
  name: string;
  label: string;
  unit: string;
  kind: "numeric" | "categorical" | "binary";
  min?: number;
  max?: number;
  step?: number;
  options?: (string | number)[];
}

export interface FeatureCatalog {
  feature_sets: Record<string, string[]>;
  descriptions: Record<string, string>;
  fields: Record<string, FieldMeta>;
  synthetic_only: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  milestone: string;
  synthetic_only: boolean;
  active_sessions: number;
}

export interface PredictionResult {
  status: string;
  predicted_fuel_rate_lph: number;
  units: string;
  feature_set: string;
  model?: { name?: string; artifact?: string };
  preprocessor?: { artifact?: string; transformed_columns?: string[] };
  input_fields_used?: Record<string, string | number>;
  provenance?: Record<string, unknown>;
}

export interface ShapContribution {
  feature: string;
  shap: number;
  /* Present in the M6.1 gateway's explain-prediction output. */
  column?: string;
  value?: number;
  actual_input_value?: number | null;
  readable?: string;
}

export interface ExplanationResult {
  status: string;
  predicted_fuel_rate_lph: number;
  feature_set: string;
  explanation_kind: string;
  expected_value: number;
  reconstruction?: {
    pass: boolean;
    max_abs_deviation?: number;
    tolerance?: number;
  };
  top_features?: number;
  /** Legacy normalized shape (predict+explain intent). */
  most_influential_features?: {
    top_positive_shap: ShapContribution[];
    top_negative_shap: ShapContribution[];
  };
  /** Canonical gateway explain-prediction shape. */
  positive_contributions?: ShapContribution[];
  negative_contributions?: ShapContribution[];
  limitations?: {
    shap_is_not_causal: boolean;
    note?: string;
    instance_scope?: string;
  };
  provenance?: Record<string, unknown>;
}

export interface ValidationIssue {
  check_id: string;
  passed: boolean;
  detail: string;
}

export interface ValidationSummary {
  total_checks: number;
  passed: number;
  failed: number;
  all_passed: boolean;
  failed_checks: string[];
  issues: ValidationIssue[];
}

export interface TechnicalReport {
  prediction?: PredictionResult | null;
  explanation?: ExplanationResult | null;
  evidence?: unknown[];
  evidence_insufficient?: boolean;
  recommendations?: Record<string, unknown>;
  validation?: ValidationSummary;
  synthetic_only?: boolean;
  invented_telemetry?: boolean;
  shap_is_attribution_not_causality?: boolean;
  tool_failures?: string[];
  driver_intent?: string;
  message?: string;
  note?: string;
}

export interface ApiResponse {
  session_id: string;
  driver_intent: string;
  conversation_state: string;
  delivered: boolean;
  validation_blocked: boolean;
  driver_feedback: string;
  clarification_request?: string | null;
  missing_fields: string[];
  invalid_fields: Record<string, string>;
  selected_tools: string[];
  executed_tools: string[];
  technical_report: TechnicalReport;
  error?: { code: string; message: string } | null;
}

export interface SessionState {
  session_id: string;
  feature_set: string;
  conversation_state: string;
  history?: unknown[];
  truck_context?: Record<string, unknown>;
  pending_clarification?: string[];
  [k: string]: unknown;
}

export interface ReplayPoint {
  timestamp?: string;
  speed_kph: number;
  engine_rpm: number;
  engine_load_pct: number;
  throttle_position_pct?: number;
  fuel_rate_lph: number;
  road_grade_pct?: number;
}

export interface ReplayResponse {
  trip_id: string;
  truck_id?: string;
  vehicle_class?: string;
  source: "synthetic_telemetry_replay";
  record_count_total: number;
  record_count_sampled: number;
  points: ReplayPoint[];
}

export interface ReplayTrip {
  trip_id: string;
  route_type: string;
  record_count: number;
  duration_seconds: number;
  distance_km: number;
  start_timestamp_utc: string;
  end_timestamp_utc: string;
}

export interface ReplayTruck {
  truck_id: string;
  vehicle_class: string;
  gross_vehicle_weight_kg: number;
  trips: ReplayTrip[];
}

export interface ReplayCatalog {
  source: "synthetic_telemetry_replay";
  trucks: ReplayTruck[];
}
