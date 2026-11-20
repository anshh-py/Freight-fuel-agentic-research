import { describe, expect, it } from "vitest";
import type { ApiResponse } from "../api/types";
import type { HistoryPoint } from "../components/AnalyticsCharts";
import { nextHistoryEntry, predictionFuel } from "./predictionHistory";

const SHARED: Omit<ApiResponse, "technical_report" | "delivered"> = {
  session_id: "s1",
  driver_intent: "explain_prediction",
  conversation_state: "answered",
  validation_blocked: false,
  driver_feedback: "",
  missing_fields: [],
  invalid_fields: {},
  selected_tools: ["prediction_explanation"],
  executed_tools: ["prediction_explanation"],
};

function delivered(overrides: Partial<ApiResponse> = {}): ApiResponse {
  return {
    ...SHARED,
    delivered: true,
    technical_report: {
      explanation: {
        status: "ok",
        predicted_fuel_rate_lph: 41.2622,
        feature_set: "full_operational",
        explanation_kind: "instance_level_shap",
        expected_value: 41.5477,
      },
    },
    ...overrides,
  };
}

const INPUTS = { engine_rpm: 1041.9, speed_kph: 11.96, engine_load_pct: 23.42 };

describe("predictionFuel", () => {
  it("extracts the fuel rate from an explanation/prediction", () => {
    expect(predictionFuel(delivered())).toBe(41.2622);
  });

  it("returns null when there is no fuel value", () => {
    expect(
      predictionFuel(
        delivered({
          technical_report: {},
        }),
      ),
    ).toBeNull();
  });
});

describe("nextHistoryEntry — prediction-history integrity", () => {
  it("explicit successful prediction adds exactly one entry", () => {
    const entry = nextHistoryEntry([], delivered(), INPUTS);
    expect(entry).not.toBeNull();
    expect(entry!.seq).toBe(1);
    expect(entry!.fuel).toBe(41.2622);
    expect(entry!.rpm).toBe(1041.9);
    expect(entry!.speed).toBeCloseTo(11.96);
    expect(entry!.load).toBeCloseTo(23.42);
  });

  it("a distinct second prediction appends exactly one more entry", () => {
    const first: HistoryPoint = {
      seq: 1,
      fuel: 41.2622,
      rpm: 1041.9,
      speed: 11.96,
      load: 23.42,
    };
    const secondInputs = { engine_rpm: 1500, speed_kph: 40, engine_load_pct: 60 };
    const second = nextHistoryEntry([first], delivered(), secondInputs);
    expect(second).not.toBeNull();
    expect(second!.seq).toBe(2);
    expect(second!.fuel).toBe(41.2622);
    expect(second!.rpm).toBe(1500);
  });

  it("assistant follow-up / re-explanation of the same prediction adds NO duplicate", () => {
    const existing: HistoryPoint = {
      seq: 1,
      fuel: 41.2622,
      rpm: 1041.9,
      speed: 11.96,
      load: 23.42,
    };
    // Same prediction, same carried inputs (the assistant re-explanation case).
    expect(nextHistoryEntry([existing], delivered(), INPUTS)).toBeNull();
  });

  it("failed responses add no entry", () => {
    expect(
      nextHistoryEntry(
        [],
        { ...delivered(), delivered: false, error: { code: "E", message: "boom" } },
        INPUTS,
      ),
    ).toBeNull();
  });

  it("validation-blocked responses add no entry", () => {
    expect(
      nextHistoryEntry(
        [],
        { ...delivered(), delivered: false, validation_blocked: true },
        INPUTS,
      ),
    ).toBeNull();
  });

  it("clarification responses add no entry", () => {
    expect(
      nextHistoryEntry(
        [],
        {
          ...SHARED,
          delivered: false,
          driver_intent: "clarifying",
          clarification_request: "Need inputs",
          technical_report: { note: "no tools executed" },
        },
        INPUTS,
      ),
    ).toBeNull();
  });

  it("a delivered response without a fuel value adds no entry", () => {
    expect(nextHistoryEntry([], delivered({ technical_report: {} }), INPUTS)).toBeNull();
  });
});
