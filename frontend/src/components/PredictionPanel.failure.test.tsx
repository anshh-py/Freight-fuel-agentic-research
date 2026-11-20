/**
 * M6.4 negative-path regression tests for PredictionPanel rendering.
 *
 * Confirms the UI's failure invariants:
 *   - a clarification shows the missing-fields prompt, never a fuel metric,
 *   - a non-delivered / validation-blocked response never shows the
 *     "delivered" success state or a prediction metric,
 *   - an error message is surfaced in the error box.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import PredictionPanel from "./PredictionPanel";
import type { ApiResponse } from "../api/types";

const BLANK: Omit<ApiResponse, "technical_report" | "delivered" | "driver_intent"> = {
  session_id: "s1",
  conversation_state: "answered",
  validation_blocked: false,
  driver_feedback: "",
  missing_fields: [],
  invalid_fields: {},
  selected_tools: [],
  executed_tools: [],
};

function renderPanel(overrides: Partial<ApiResponse>) {
  const resp = {
    ...BLANK,
    driver_intent: "explain_prediction",
    delivered: true,
    technical_report: {
      prediction: { status: "ok", predicted_fuel_rate_lph: 41.2622, feature_set: "full_operational" },
      validation: { total_checks: 9, passed: 9, failed: 0, all_passed: true, failed_checks: [], issues: [] },
    },
    ...overrides,
  } as ApiResponse;
  render(
    <PredictionPanel running={false} response={resp} errorMsg={null} canRun onRun={vi.fn()} />,
  );
}

describe("PredictionPanel failure-state rendering", () => {
  it("clarification shows the missing-fields prompt, never a fuel metric or delivered badge", () => {
    renderPanel({
      driver_intent: "clarifying",
      delivered: false,
      clarification_request: "I need the following vehicle/trip values: speed_kph.",
      missing_fields: ["speed_kph"],
      technical_report: { note: "no tools executed while inputs are incomplete" },
    });

    expect(screen.getByText(/More vehicle inputs needed/)).toBeInTheDocument();
    expect(screen.getByText("speed_kph")).toBeInTheDocument();
    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
    expect(screen.queryByText(/delivered/)).not.toBeInTheDocument();
  });

  it("a validation-blocked prediction does NOT show a fuel metric or delivered state", () => {
    renderPanel({
      delivered: false,
      validation_blocked: true,
      driver_intent: "explain_prediction",
    });

    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
    expect(screen.queryByText(/delivered/)).not.toBeInTheDocument();
  });

  it("an error message is surfaced in the error box without a prediction metric", () => {
    renderPanel({
      delivered: false,
    });
    render(
      <PredictionPanel
        running={false}
        response={null}
        errorMsg="Backend error (503): Service Unavailable"
        canRun={false}
        onRun={vi.fn()}
      />,
    );
    expect(screen.getByText(/Backend error \(503\): Service Unavailable/)).toBeInTheDocument();
    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
  });

  it("no response and no error renders no success or error framing", () => {
    render(
      <PredictionPanel running={false} response={null} errorMsg={null} canRun onRun={vi.fn()} />,
    );
    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Run AI prediction/ })).toBeInTheDocument();
  });
});
