import type { ApiResponse } from "../api/types";
import type { HistoryPoint } from "../components/AnalyticsCharts";

export function predictionFuel(resp: ApiResponse): number | null {
  const tr = resp.technical_report;
  return (
    tr?.prediction?.predicted_fuel_rate_lph ??
    tr?.explanation?.predicted_fuel_rate_lph ??
    null
  );
}

/**
 * Decide whether a delivered response appends a NEW model-prediction history
 * entry, and build it. Returns `null` when nothing should be recorded:
 *   - not delivered (error / validation-blocked / clarification / unsupported),
 *   - no fuel value present in the response, or
 *   - the exact same prediction as the most recent entry (for example an
 *     assistant re-explanation of the same carried inputs). The backend reuses
 *     the session's truck context for a follow-up and returns the identical
 *     fuel, so this guard prevents duplicate chart points for one prediction.
 */
export function nextHistoryEntry(
  history: HistoryPoint[],
  resp: ApiResponse,
  inputs: { engine_rpm?: number; speed_kph?: number; engine_load_pct?: number },
): HistoryPoint | null {
  if (!resp.delivered) return null;
  const fuel = predictionFuel(resp);
  if (fuel == null) return null;

  const entry: HistoryPoint = {
    seq: history.length + 1,
    fuel,
    rpm: Number(inputs.engine_rpm ?? 0),
    speed: Number(inputs.speed_kph ?? 0),
    load: Number(inputs.engine_load_pct ?? 0),
  };

  const last = history[history.length - 1];
  if (
    last &&
    last.fuel === fuel &&
    last.rpm === entry.rpm &&
    last.speed === entry.speed &&
    last.load === entry.load
  ) {
    return null;
  }

  return entry;
}
