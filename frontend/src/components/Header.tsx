import React from "react";
import type { FeatureSetName } from "../api/types";

interface HeaderProps {
  backendStatus: "connecting" | "ok" | "error";
  sessionId: string | null;
  featureSet: FeatureSetName;
  featureSets: string[];
  onFeatureSetChange: (f: FeatureSetName) => void;
}

export default function Header({
  backendStatus,
  sessionId,
  featureSet,
  featureSets,
  onFeatureSetChange,
}: HeaderProps) {
  const dot =
    backendStatus === "ok" ? (
      <span className="dot dot-ok" />
    ) : backendStatus === "error" ? (
      <span className="dot dot-bad" />
    ) : (
      <span className="dot dot-idle pulse" />
    );

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-icon">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M5 8h11l3 4v5h-2m-12-9v9h2m6 0a2 2 0 1 1 4 0 2 2 0 0 1-4 0Zm-8 0a2 2 0 1 1 4 0 2 2 0 0 1-4 0Z"
              stroke="#03141f"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <div className="brand-title">FuelAgent Cockpit</div>
          <div className="brand-sub">Truck Fuel Agentic Research · M6.2</div>
        </div>
      </div>

      <div className="header-spacer" />

      <div className="header-pills">
        <span className="pill">
          <span className="dot dot-ok" /> Truck T-001 · 40T
        </span>

        <select
          className="feature-select"
          value={featureSet}
          onChange={(e) => onFeatureSetChange(e.target.value as FeatureSetName)}
          aria-label="Selected feature set"
        >
          {featureSets.map((f) => (
            <option key={f} value={f}>
              {f === "full_operational" ? "Full operational set" : "Telematics only"}
            </option>
          ))}
        </select>

        <span className="pill">
          {dot}
          {backendStatus === "ok"
            ? "Backend connected"
            : backendStatus === "error"
              ? "Backend offline"
              : "Connecting…"}
        </span>

        <span className="pill">
          <span className="dot dot-ok" />
          {sessionId ? `session ${sessionId.slice(0, 6)}` : "no session"}
        </span>

        <span className="pill pill-synthetic">
          <span className="dot dot-warn" /> Synthetic data
        </span>
      </div>
    </header>
  );
}
