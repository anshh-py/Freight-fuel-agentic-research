/**
 * M6.4 negative-path regression tests for AssistantPanel.
 *
 * Confirms the assistant's failure invariants:
 *   - clarification renders a clarify bubble, never calls onPredictionSuccess,
 *   - safety decline renders the decline text, never calls onPredictionSuccess,
 *   - backend failure renders an error bubble, never calls onPredictionSuccess,
 *   - only a delivered response with a fuel value calls onPredictionSuccess.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import AssistantPanel from "./AssistantPanel";
import { api } from "../api/client";
import type { ApiResponse } from "../api/types";

vi.mock("../api/client", () => {
  class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  }
  return {
    ApiError,
    api: {
      health: vi.fn(),
      features: vi.fn(),
      createSession: vi.fn(),
      getSession: vi.fn(),
      deleteSession: vi.fn(),
      sendMessage: vi.fn(),
      replayCatalog: vi.fn(),
      replay: vi.fn(),
    },
  };
});

const mockedApi = api as unknown as {
  sendMessage: ReturnType<typeof vi.fn>;
};

function baseResponse(overrides: Partial<ApiResponse>): ApiResponse {
  return {
    session_id: "s1",
    driver_intent: "knowledge_query",
    conversation_state: "answered",
    delivered: true,
    validation_blocked: false,
    driver_feedback: "ok",
    missing_fields: [],
    invalid_fields: {},
    selected_tools: [],
    executed_tools: [],
    technical_report: {},
    ...overrides,
  };
}

function renderPanel() {
  const onPredictionSuccess = vi.fn();
  render(<AssistantPanel sessionId="s1" disabled={false} onPredictionSuccess={onPredictionSuccess} />);
  return { onPredictionSuccess };
}

async function ask(question: string) {
  const textarea = screen.getByLabelText("Message the assistant");
  fireEvent.change(textarea, { target: { value: question } });
  fireEvent.click(screen.getByRole("button", { name: /Send/ }));
  await waitFor(() => expect(screen.queryByText(/typing/)).not.toBeInTheDocument());
}

describe("AssistantPanel failure-path handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("clarification shows a clarify bubble and does not call onPredictionSuccess", async () => {
    mockedApi.sendMessage.mockResolvedValue(
      baseResponse({
        driver_intent: "clarifying",
        conversation_state: "clarifying",
        delivered: false,
        clarification_request: "I need speed_kph.",
        driver_feedback: "I need speed_kph.",
        missing_fields: ["speed_kph"],
        technical_report: { note: "no tools executed while inputs are incomplete" },
      }),
    );
    const { onPredictionSuccess } = renderPanel();
    await ask("predict my fuel rate");

    expect(screen.getByText(/I need speed_kph\./)).toBeInTheDocument();
    expect(onPredictionSuccess).not.toHaveBeenCalled();
  });

  it("safety decline shows the decline text and does not call onPredictionSuccess", async () => {
    mockedApi.sendMessage.mockResolvedValue(
      baseResponse({
        driver_intent: "unsupported",
        conversation_state: "unsupported",
        delivered: true,
        driver_feedback: "I can't answer that within this project's scope.",
        technical_report: { note: "unsupported request remains unsupported" },
      }),
    );
    const { onPredictionSuccess } = renderPanel();
    await ask("what is my truck's OEM warranty?");

    expect(screen.getByText(/I can't answer that within this project's scope\./)).toBeInTheDocument();
    expect(onPredictionSuccess).not.toHaveBeenCalled();
  });

  it("backend failure renders an error bubble and does not call onPredictionSuccess", async () => {
    mockedApi.sendMessage.mockRejectedValue(new Error("fetch failed"));
    const { onPredictionSuccess } = renderPanel();
    await ask("helloo");

    expect(screen.getByText(/Could not reach the backend\./)).toBeInTheDocument();
    expect(onPredictionSuccess).not.toHaveBeenCalled();
  });

  it("a delivered response carrying a fuel value does call onPredictionSuccess", async () => {
    mockedApi.sendMessage.mockResolvedValue(
      baseResponse({
        driver_intent: "explain_prediction",
        technical_report: {
          explanation: { status: "ok", predicted_fuel_rate_lph: 41.2622, feature_set: "full_operational", explanation_kind: "instance_level_shap", expected_value: 41.5477 },
        },
      }),
    );
    const { onPredictionSuccess } = renderPanel();
    await ask("explain the prediction");

    expect(onPredictionSuccess).toHaveBeenCalledTimes(1);
  });
});
