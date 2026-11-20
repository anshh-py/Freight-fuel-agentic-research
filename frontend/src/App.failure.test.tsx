/**
 * M6.4 App-level failure-path regression tests.
 *
 * Covers the two scenarios that span the whole cockpit:
 *   - backend unavailable at startup (offline header, Run disabled, no session),
 *   - invalid session / backend error during predict (error surfaced, no
 *     prediction history entry, no false success).
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import { api, ApiError } from "./api/client";

vi.mock("./api/client", () => {
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
  health: ReturnType<typeof vi.fn>;
  features: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  sendMessage: ReturnType<typeof vi.fn>;
  replayCatalog: ReturnType<typeof vi.fn>;
  replay: ReturnType<typeof vi.fn>;
};

const CATALOG = {
  feature_sets: { full_operational: ["speed_kph"] },
  descriptions: {},
  fields: {},
  synthetic_only: true,
};

function okStartup() {
  mockedApi.health.mockResolvedValue({ status: "ok", service: "g", milestone: "m6", synthetic_only: true, active_sessions: 1 });
  mockedApi.features.mockResolvedValue(CATALOG);
  mockedApi.createSession.mockResolvedValue({ session_id: "s1", feature_set: "full_operational" });
  mockedApi.replayCatalog.mockResolvedValue({ source: "synthetic_telemetry_replay", trucks: [] });
  mockedApi.replay.mockResolvedValue({ trip_id: "t", source: "synthetic_telemetry_replay", record_count_total: 1, record_count_sampled: 1, points: [] });
}

describe("App backend-unavailable / invalid-session failure paths", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("backend unavailable at startup: offline header, Run disabled, no session badge", async () => {
    mockedApi.health.mockRejectedValue(new Error("down"));
    mockedApi.features.mockRejectedValue(new Error("down"));
    mockedApi.createSession.mockRejectedValue(new Error("down"));
    mockedApi.replayCatalog.mockRejectedValue(new Error("down"));

    render(<App />);
    await waitFor(() => expect(screen.getByText(/Backend offline/)).toBeInTheDocument());
    expect(screen.getByText(/no session/)).toBeInTheDocument();

    const run = screen.getByRole("button", { name: /Run AI prediction/ });
    expect((run as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
  });

  it("invalid session / backend error during predict surfaces the error, no history entry, no false success", async () => {
    okStartup();
    mockedApi.sendMessage.mockRejectedValue(new ApiError(404, "Session not found"));

    render(<App />);
    // wait for startup to settle (session "s1" created -> header badge)
    await waitFor(() => expect(screen.getByText(/session s1/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /Run AI prediction/ }));

    await waitFor(() => {
      expect(screen.getByText(/Backend error \(404\): Session not found/)).toBeInTheDocument();
    });
    // No successful prediction metric is shown.
    expect(screen.queryByText(/Predicted fuel rate/)).not.toBeInTheDocument();
    // No synthetic replay target is misrepresented as a model prediction.
    expect(screen.queryByText(/delivered/)).not.toBeInTheDocument();
  });
});
