import React from "react";
import type { FeatureCatalog, ReplayResponse } from "../api/types";

interface VehicleContextCardProps {
  catalog: FeatureCatalog | null;
  featureSet: string;
  inputs: Record<string, string | number>;
  scenarioLabel: string | null;
  replay: ReplayResponse | null;
  conversationState: string;
}

export default function VehicleContextCard({
  catalog,
  featureSet,
  inputs,
  scenarioLabel,
  replay,
  conversationState,
}: VehicleContextCardProps) {
  const vehicleClass = inputs.vehicle_class ?? "—";
  const mass = Number(inputs.vehicle_mass_kg ?? 0);
  const payload = Number(inputs.payload_kg ?? 0);

  const rows: [string, string][] = [
    ["Vehicle class", String(vehicleClass)],
    ["Mass", `${mass.toLocaleString()} kg`],
    ["Payload", `${payload.toLocaleString()} kg`],
    [
      "Driving scenario",
      scenarioLabel ?? "Manual (custom controls)",
    ],
    ["Feature set", featureSet],
    ["Session state", conversationState || "—"],
    ["Replay trip", replay ? `${replay.trip_id} (${replay.truck_id ?? "?"})` : "—"],
  ];

  return (
    <section className="card area-vehicle">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">🚚</span> Vehicle / trip context
        </div>
        <span className="tag">synthetic</span>
      </div>

      <div className="card-pad">
        <div className="kv-list">
          {rows.map(([k, v]) => (
            <div className="kv" key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ))}
        </div>

        {catalog?.descriptions[featureSet] && (
          <div className="advanced-note" style={{ marginTop: 12 }}>
            <span>ℹ️</span>
            <span>{catalog.descriptions[featureSet]}</span>
          </div>
        )}

        <div className="advanced-note" style={{ marginTop: 10 }}>
          <span>⚠️</span>
          <span>
            This cockpit is a simulator connected to the research gateway. It is
            <strong> not </strong>
            wired to a real truck or live OBD hardware. Every telemetry value,
            model target, and retrieval passage is synthetic.
          </span>
        </div>
      </div>
    </section>
  );
}
