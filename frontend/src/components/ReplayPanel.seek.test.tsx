/**
 * Focused M6.3 component/integration test for ReplayPanel seeking and
 * end-of-trip behavior. It drives the REAL React onChange handler of the
 * timeline range input (handleSeek) and asserts the rendered state.
 *
 * The headless browser tool cannot dispatch the controlled range events React
 * needs, so this test exercises the actual handler + rendered output instead.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ReplayPanel from "./ReplayPanel";
import { api } from "../api/client";
import type { ReplayCatalog, ReplayResponse } from "../api/types";

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
      replayCatalog: vi.fn(),
      replay: vi.fn(),
      health: vi.fn(),
      features: vi.fn(),
      createSession: vi.fn(),
      getSession: vi.fn(),
      deleteSession: vi.fn(),
      sendMessage: vi.fn(),
    },
  };
});

const mockedApi = api as unknown as {
  replayCatalog: ReturnType<typeof vi.fn>;
  replay: ReturnType<typeof vi.fn>;
};

/** 5-point synthetic replay series with a timestamp for every record. */
const POINTS: ReplayResponse["points"] = [
  { timestamp: "2023-01-28T12:00:00.000Z", speed_kph: 0, engine_rpm: 620, engine_load_pct: 15, throttle_position_pct: 20, fuel_rate_lph: 1.1, road_grade_pct: 0 },
  { timestamp: "2023-01-28T12:00:10.000Z", speed_kph: 20, engine_rpm: 900, engine_load_pct: 25, throttle_position_pct: 30, fuel_rate_lph: 1.4, road_grade_pct: 0.2 },
  { timestamp: "2023-01-28T12:00:20.000Z", speed_kph: 40, engine_rpm: 1200, engine_load_pct: 40, throttle_position_pct: 45, fuel_rate_lph: 1.9, road_grade_pct: 0.5 },
  { timestamp: "2023-01-28T12:00:30.000Z", speed_kph: 60, engine_rpm: 1500, engine_load_pct: 55, throttle_position_pct: 60, fuel_rate_lph: 2.6, road_grade_pct: -0.3 },
  { timestamp: "2023-01-28T12:00:40.000Z", speed_kph: 80, engine_rpm: 1800, engine_load_pct: 70, throttle_position_pct: 75, fuel_rate_lph: 3.4, road_grade_pct: -0.8 },
];

const CATALOG: ReplayCatalog = {
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
          record_count: 5,
          duration_seconds: 40,
          distance_km: 8.2,
          start_timestamp_utc: "2023-01-28T12:00:00Z",
          end_timestamp_utc: "2023-01-28T12:00:40Z",
        },
      ],
    },
  ],
};

const RESPONSE: ReplayResponse = {
  trip_id: "trip_000001",
  truck_id: "truck_001",
  vehicle_class: "Class 8",
  source: "synthetic_telemetry_replay",
  record_count_total: 5,
  record_count_sampled: 5,
  points: POINTS,
};

function renderPanel() {
  const props = {
    onLoadReplay: vi.fn(),
    onPoint: vi.fn(),
    onProgress: vi.fn(),
    onLoadSampleToInputs: vi.fn(),
  };
  const utils = render(<ReplayPanel {...props} />);
  return { props, utils };
}

describe("ReplayPanel seek and end-of-trip", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedApi.replayCatalog.mockResolvedValue(CATALOG);
    mockedApi.replay.mockResolvedValue(RESPONSE);
  });

  it("loads the trip and renders sample 1 (index 0) initially", async () => {
    renderPanel();
    // wait for catalog + replay to load
    await waitFor(() => {
      expect(screen.getByText(/city · 5 records/)).toBeInTheDocument();
    });
    expect(screen.getByText(/1 \/ 5/)).toBeInTheDocument();
    expect(screen.getByText(/0\.0 km\/h/)).toBeInTheDocument();
    expect(screen.getByText(/620 rpm/)).toBeInTheDocument();
  });

  it("seeks to a middle position and reflects that exact record", async () => {
    const { props } = renderPanel();
    await waitFor(() => {
      expect(screen.getByLabelText("Replay timeline seek")).toBeInTheDocument();
    });

    const slider = screen.getByLabelText("Replay timeline seek");
    // middle index = 2
    fireEvent.change(slider, { target: { value: "2" } });

    // timestamp of record index 2
    expect(screen.getByText("2023-01-28 12:00:20")).toBeInTheDocument();
    // progress text: row 3 of 5, 50%
    expect(screen.getByText(/3 \/ 5 · 50%/)).toBeInTheDocument();
    // telemetry matches record index 2
    expect(screen.getByText(/40\.0 km\/h/)).toBeInTheDocument();
    expect(screen.getByText(/1200 rpm/)).toBeInTheDocument();
    expect(screen.getByText(/40\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/45\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/0\.50%/)).toBeInTheDocument();
    expect(screen.getByText(/1\.90 L\/h/)).toBeInTheDocument();

    // onPoint/onProgress received the correct record/index
    expect(props.onPoint).toHaveBeenLastCalledWith(POINTS[2]);
    expect(props.onProgress).toHaveBeenLastCalledWith(2);
  });

  it("seeks to the final position, reports 'trip ended', and stays on the final sample", async () => {
    const { props } = renderPanel();
    await waitFor(() => {
      expect(screen.getByLabelText("Replay timeline seek")).toBeInTheDocument();
    });

    const slider = screen.getByLabelText("Replay timeline seek");
    // final index = 4 (points.length - 1)
    fireEvent.change(slider, { target: { value: "4" } });

    expect(screen.getByText("2023-01-28 12:00:40")).toBeInTheDocument();
    expect(screen.getByText(/5 \/ 5 · 100%/)).toBeInTheDocument();
    // UI reports trip ended (not "paused"/"replaying")
    expect(screen.getByText(/trip ended/)).toBeInTheDocument();
    // final sample remains displayed
    expect(screen.getByText(/80\.0 km\/h/)).toBeInTheDocument();
    expect(screen.getByText(/1800 rpm/)).toBeInTheDocument();
    expect(screen.getByText(/3\.40 L\/h/)).toBeInTheDocument();

    expect(props.onPoint).toHaveBeenLastCalledWith(POINTS[4]);
    expect(props.onProgress).toHaveBeenLastCalledWith(4);
  });

  it("loads the CURRENT paused sample into prediction inputs, not the first one", async () => {
    const { props } = renderPanel();
    await waitFor(() => {
      expect(screen.getByLabelText("Replay timeline seek")).toBeInTheDocument();
    });

    // Move to a middle position (index 2) so the "first sample" is NOT current.
    fireEvent.change(
      screen.getByLabelText("Replay timeline seek"),
      { target: { value: "2" } },
    );
    expect(screen.getByText(/3 \/ 5 · 50%/)).toBeInTheDocument();

    // Click the explicit load-into-inputs button.
    fireEvent.click(
      screen.getByRole("button", { name: /Load this replay sample/ }),
    );

    // The action must carry the CURRENT record (index 2), not points[0].
    expect(props.onLoadSampleToInputs).toHaveBeenLastCalledWith({
      engine_rpm: 1200,
      speed_kph: 40,
      engine_load_pct: 40,
      throttle_position_pct: 45,
      road_grade_pct: 0.5,
    });
    expect(props.onLoadSampleToInputs).not.toHaveBeenLastCalledWith({
      engine_rpm: 620,
      speed_kph: 0,
      engine_load_pct: 15,
      throttle_position_pct: 20,
      road_grade_pct: 0,
    });
  });

  it("clamps out-of-range seeks to the valid range", async () => {
    renderPanel();
    await waitFor(() => {
      expect(screen.getByLabelText("Replay timeline seek")).toBeInTheDocument();
    });
    const slider = screen.getByLabelText("Replay timeline seek");

    // well above range -> clamped to last sample (index 4)
    fireEvent.change(slider, { target: { value: "999" } });
    expect(screen.getByText(/5 \/ 5 · 100%/)).toBeInTheDocument();

    // negative -> clamped to first sample (index 0)
    fireEvent.change(slider, { target: { value: "-5" } });
    expect(screen.getByText(/1 \/ 5 · 0%/)).toBeInTheDocument();
  });
});
