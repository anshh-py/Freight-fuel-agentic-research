import React from "react";
import ControlSlider from "./ControlSlider";
import { MANUAL_CONTROLS, SCENARIOS, type Scenario } from "../lib/cockpit";
import type { FeatureCatalog } from "../api/types";

interface ManualControlsProps {
  catalog: FeatureCatalog | null;
  inputs: Record<string, string | number>;
  activeScenario: string | null;
  onInput: (field: string, value: number) => void;
  onScenario: (s: Scenario) => void;
}

export default function ManualControls({
  catalog,
  inputs,
  activeScenario,
  onInput,
  onScenario,
}: ManualControlsProps) {
  return (
    <section className="card area-controls">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">🎛️</span> Manual engine controls
        </div>
        <span className="tag">linked to gauges</span>
      </div>

      <div className="card-pad">
        <div className="control-group">
          {MANUAL_CONTROLS.map((field) => {
            const meta = catalog?.fields[field];
            if (!meta || meta.kind !== "numeric") return null;
            return (
              <ControlSlider
                key={field}
                label={meta.label}
                unit={meta.unit}
                min={meta.min ?? 0}
                max={meta.max ?? 100}
                step={meta.step ?? 1}
                value={Number(inputs[field] ?? meta.min ?? 0)}
                onChange={(v) => onInput(field, v)}
              />
            );
          })}
        </div>

        <div className="divider" />

        <div style={{ fontSize: "12px", color: "var(--muted)", fontWeight: 600 }}>
          Predefined scenarios
        </div>
        <div className="scenario-grid">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`btn btn-ghost scenario-chip ${activeScenario === s.id ? "active" : ""}`}
              onClick={() => onScenario(s)}
            >
              <span className="t">{s.label}</span>
              <span className="d">{s.description}</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
