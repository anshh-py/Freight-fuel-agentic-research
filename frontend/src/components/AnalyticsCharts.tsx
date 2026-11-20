import React, { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReplayResponse } from "../api/types";

export interface HistoryPoint {
  seq: number;
  fuel: number;
  rpm: number;
  speed: number;
  load: number;
}

interface AnalyticsChartsProps {
  history: HistoryPoint[];
  replay: ReplayResponse | null;
  /** Number of replay points consumed so far (progressive reveal). */
  replayProgress?: number;
}

const TOOLTIP_STYLE = {
  backgroundColor: "#0d1524",
  border: "1px solid #1d2d47",
  borderRadius: 10,
  fontSize: 12,
  color: "#e6edf7",
};

const AXIS = { tick: { fill: "#8da2bd", fontSize: 11 } };
const GRID = { stroke: "#1d2d47", strokeDasharray: "3 3" };

export default function AnalyticsCharts({
  history,
  replay,
  replayProgress,
}: AnalyticsChartsProps) {
  const fuelRows = useMemo(() => {
    const rows: { label: string; modelFuel: number | null; replayFuel: number | null }[] = [];
    const shown = replay?.points.slice(0, replayProgress) ?? [];
    for (const [i, p] of shown.entries()) {
      rows.push({
        label: p.timestamp ? p.timestamp.slice(11, 19) : `r${i + 1}`,
        modelFuel: null,
        replayFuel: p.fuel_rate_lph,
      });
    }
    for (const h of history) {
      rows.push({
        label: `#${h.seq}`,
        modelFuel: h.fuel,
        replayFuel: null,
      });
    }
    return rows;
  }, [history, replay]);

  const hasModel = history.length > 0;
  const hasReplay = (replay?.points?.length ?? 0) > 0;

  const telemetryRows = useMemo(
    () =>
      history.map((h) => ({
        label: `#${h.seq}`,
        rpm: h.rpm,
        speed: h.speed,
        load: h.load,
      })),
    [history],
  );

  return (
    <section className="card area-analytics">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">📈</span> Fuel &amp; telemetry analytics
        </div>
        <div className="legend-inline">
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "#34d399" }} />
            Model prediction
          </span>
          <span className="legend-item">
            <span className="legend-swatch" style={{ background: "#8da2bd" }} />
            Synthetic telemetry replay
          </span>
        </div>
      </div>

      <div className="chart-note">
        Fuel-rate points come from two distinct sources: newly generated model
        predictions (added only after a successful, validated backend response)
        and pre-recorded synthetic telemetry replay. No fuel value is invented in
        the browser. Accuracy vs. actual is not claimed — there are no reference
        labels for live runs.
      </div>

      <div className="chart-grid">
        {/* Fuel rate — dual provenance */}
        <div className="chart-card">
          <div className="card-title" style={{ padding: "0 14px 6px" }}>
            Fuel rate (L/h)
          </div>
          <div style={{ height: 260 }}>
            {hasModel || hasReplay ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={fuelRows} margin={{ top: 10, right: 14, left: 0, bottom: 4 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="label" {...AXIS} />
                  <YAxis {...AXIS} width={40} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {hasReplay && (
                    <Line
                      type="monotone"
                      dataKey="replayFuel"
                      name="Synthetic replay"
                      stroke="#8da2bd"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive
                    />
                  )}
                  {hasModel && (
                    <Line
                      type="monotone"
                      dataKey="modelFuel"
                      name="Model prediction"
                      stroke="#34d399"
                      strokeWidth={2.5}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                      isAnimationActive
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="muted-2" style={{ padding: 20, fontSize: 13 }}>
                No fuel data yet. Run a prediction to add a point, or rely on the
                synthetic replay series.
              </div>
            )}
          </div>
        </div>

        {/* Telemetry trends from model runs */}
        <div className="chart-card">
          <div className="card-title" style={{ padding: "0 14px 6px" }}>
            Cockpit telemetry (model runs)
          </div>
          <div style={{ height: 260 }}>
            {telemetryRows.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={telemetryRows} margin={{ top: 10, right: 14, left: 0, bottom: 4 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="label" {...AXIS} />
                  <YAxis {...AXIS} width={40} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="rpm"
                    name="Engine RPM"
                    stroke="#35c4ff"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="speed"
                    name="Speed (km/h)"
                    stroke="#7c5cff"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="load"
                    name="Engine load (%)"
                    stroke="#fbbf24"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="muted-2" style={{ padding: 20, fontSize: 13 }}>
                Run a prediction to populate telemetry trend history.
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
