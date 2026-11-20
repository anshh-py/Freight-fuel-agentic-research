/**
 * M6.4 negative-path regression tests for ReplayPanel.
 *
 * Confirms failure invariants:
 *   - empty / invalid selection degrades safely (disabled selects, no crash),
 *   - replay API failure renders an alert and clears the active point,
 *   - catalog failure renders an alert.
 */
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ReplayPanel from "./ReplayPanel";
import { api } from "../api/client";
import type { ReplayCatalog } from "../api/types";

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
  replayCatalog: ReturnType<typeof vi.fn>;
  replay: ReturnType<typeof vi.fn>;
};

const EMPTY_CATALOG: ReplayCatalog = { source: "synthetic_telemetry_replay", trucks: [] };

function renderPanel() {
  const props = {
    onLoadReplay: vi.fn(),
    onPoint: vi.fn(),
    onProgress: vi.fn(),
    onLoadSampleToInputs: vi.fn(),
  };
  render(<ReplayPanel {...props} />);
  return props;
}

describe("ReplayPanel failure-path handling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("empty catalog degrades safely: selects are present and no crash", async () => {
    mockedApi.replayCatalog.mockResolvedValue(EMPTY_CATALOG);
    const props = renderPanel();

    await waitFor(() => {
      expect(screen.getByText("Truck")).toBeInTheDocument();
    });
    // Truck select is disabled (no trucks to choose from).
    const selects = screen.getAllByRole("combobox");
    for (const s of selects) {
      expect((s as HTMLSelectElement).disabled).toBe(true);
    }
    // No replay data has been loaded, and no real point is ever reported.
    expect(props.onLoadReplay).not.toHaveBeenCalled();
    expect(props.onPoint).not.toHaveBeenCalledWith(expect.objectContaining({ speed_kph: expect.any(Number) }));
  });

  it("replay API failure renders an alert and clears the active point", async () => {
    mockedApi.replayCatalog.mockResolvedValue({
      source: "synthetic_telemetry_replay",
      trucks: [
        {
          truck_id: "truck_001",
          vehicle_class: "Class 8",
          gross_vehicle_weight_kg: 36000,
          trips: [
            {
              trip_id: "trip_000001",
              route_type: "city",
              record_count: 100,
              duration_seconds: 100,
              distance_km: 8,
              start_timestamp_utc: "2023-01-01T00:00:00Z",
              end_timestamp_utc: "2023-01-01T00:01:40Z",
            },
          ],
        },
      ],
    });
    mockedApi.replay.mockRejectedValue(new Error("backend down"));
    const props = renderPanel();

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
    expect(screen.getByText(/Could not load replay data\./)).toBeInTheDocument();
    expect(props.onPoint).toHaveBeenCalledWith(null);
  });

  it("catalog failure renders an alert", async () => {
    mockedApi.replayCatalog.mockRejectedValue(new Error("backend down"));
    renderPanel();
    await waitFor(() => {
      expect(screen.getByText(/Could not load replay catalog\./)).toBeInTheDocument();
    });
  });

  it("empty trip selection (blank tripId) does not fire a replay request", async () => {
    mockedApi.replayCatalog.mockResolvedValue({
      source: "synthetic_telemetry_replay",
      trucks: [
        {
          truck_id: "truck_001",
          vehicle_class: "Class 8",
          gross_vehicle_weight_kg: 36000,
          trips: [],
        },
      ],
    });
    const props = renderPanel();
    await waitFor(() => {
      expect(screen.getByText("Truck")).toBeInTheDocument();
    });
    expect(mockedApi.replay).not.toHaveBeenCalled();
    expect(props.onLoadReplay).not.toHaveBeenCalled();
  });
});
