import React, { useState } from "react";
import type { ApiResponse, ExplanationResult, ShapContribution } from "../api/types";

interface PredictionPanelProps {
  running: boolean;
  response: ApiResponse | null;
  errorMsg: string | null;
  canRun: boolean;
  onRun: () => void;
}

function ShapTable({ rows }: { rows?: ShapContribution[] }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div>
      {rows.map((r) => (
        <div className="shap-row" key={r.feature + r.column + r.shap}>
          <span className="feat">{r.readable || r.column || r.feature}</span>
          <span className="pos">{r.shap >= 0 ? "+" : ""}{r.shap.toFixed(2)}</span>
          <span className="muted">value {r.actual_input_value ?? r.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Normalize SHAP contributions across gateway intent shapes. */
function contributionRows(expl?: ExplanationResult): {
  pos: ShapContribution[];
  neg: ShapContribution[];
} {
  const pos =
    expl?.positive_contributions ??
    expl?.most_influential_features?.top_positive_shap ??
    [];
  const neg =
    expl?.negative_contributions ??
    expl?.most_influential_features?.top_negative_shap ??
    [];
  return { pos, neg };
}

export default function PredictionPanel({
  running,
  response,
  errorMsg,
  canRun,
  onRun,
}: PredictionPanelProps) {
  const [showShap, setShowShap] = useState(false);
  const [showValidation, setShowValidation] = useState(false);

  const tech = response?.technical_report;
  const prediction = tech?.prediction;
  const explanation = tech?.explanation;
  const validation = tech?.validation;
  const fuelRate =
    prediction?.predicted_fuel_rate_lph ??
    explanation?.predicted_fuel_rate_lph ??
    null;
  const isClarify = response?.driver_intent === "clarifying";

  return (
    <section className="card area-prediction">
      <div className="card-head">
        <div className="card-title">
          <span className="icon">📊</span> AI prediction
        </div>
        <span className="tag">XGBoost + SHAP</span>
      </div>

      <div className="card-pad">
        <div className="prediction-action">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onRun}
            disabled={running || !canRun}
          >
            {running ? (
              <>
                <span className="spinner" /> Running…
              </>
            ) : (
              <>▶ Run AI prediction</>
            )}
          </button>

          {running && (
            <span className="predicting">
              <span className="bar"><i /></span> computing prediction & explanation
            </span>
          )}
        </div>

        {/* Clarification state */}
        {!running && isClarify && response && (
          <div className="error-box">
            <strong>More vehicle inputs needed.</strong>{" "}
            {response.clarification_request}
            {response.missing_fields.length > 0 && (
              <div style={{ marginTop: 8 }}>
                Missing:{" "}
                {response.missing_fields.map((f) => (
                  <span className="tag" key={f} style={{ marginRight: 6 }}>
                    {f}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Error state */}
        {!running && errorMsg && (
          <div className="error-box">⚠️ {errorMsg}</div>
        )}

        {/* Success state — only a DELIVERED validated prediction may present a fuel metric. */}
        {!running && response && response.delivered && !isClarify && fuelRate != null && (
          <div className="prediction-result">
            <div className="metric-big">
              <div className="k">Predicted fuel rate</div>
              <div className="v">
                {fuelRate.toFixed(2)}
                <span className="unit">L/h</span>
              </div>
              <div style={{ marginTop: 8 }} className="row">
                {response.delivered ? (
                  <span className="status-line status-ok">
                    <span className="dot dot-ok" /> delivered
                  </span>
                ) : (
                  <span className="status-line status-bad">
                    <span className="dot dot-bad" /> validation blocked
                  </span>
                )}
              </div>
            </div>

            <div className="prediction-detail">
              <div className="feedback-text">{response.driver_feedback}</div>

              <div className="kv-list">
                <div className="kv">
                  <span className="k">Driver intent</span>
                  <span className="v">{response.driver_intent}</span>
                </div>
                <div className="kv">
                  <span className="k">Executed tools</span>
                  <span className="v">
                    {response.executed_tools.join(", ") || "—"}
                  </span>
                </div>
                <div className="kv">
                  <span className="k">Validation</span>
                  <span className="v">
                    {validation
                      ? `${validation.passed}/${validation.total_checks} checks passed`
                      : "—"}
                  </span>
                </div>
              </div>

              {tech?.synthetic_only && (
                <div style={{ fontSize: "11.5px", color: "var(--warn)" }}>
                  ⚠️ Synthetic-only: estimate from a synthetic-trained model, not
                  a real-world fuel measurement.
                </div>
              )}
            </div>
          </div>
        )}

        {/* SHAP explanation (expandable) */}
        {!running && explanation && (
          <div
            className={`expandable ${showShap ? "open" : ""}`}
            style={{ marginTop: 14 }}
          >
            <button
              type="button"
              className="expandable-trigger"
              onClick={() => setShowShap((v) => !v)}
              aria-expanded={showShap}
            >
              <span>🔬 Technical SHAP explanation</span>
              <span className="caret">▾</span>
            </button>
            {showShap && (
              <div className="expandable-body">
                <div className="kv-list" style={{ gap: 4, marginBottom: 8 }}>
                  <div className="kv">
                    <span className="k">Kind</span>
                    <span className="v">{explanation.explanation_kind}</span>
                  </div>
                  <div className="kv">
                    <span className="k">Expected value</span>
                    <span className="v">
                      {explanation.expected_value?.toFixed(2)} L/h (base rate)
                    </span>
                  </div>
                  {explanation.reconstruction && (
                    <div className="kv">
                      <span className="k">SHAP reconstruction</span>
                      <span className="v">
                        {explanation.reconstruction.pass
                          ? "passed (max dev "
                            + (explanation.reconstruction.max_abs_deviation ?? 0).toFixed(5)
                            + ")"
                          : "failed"}
                      </span>
                    </div>
                  )}
                </div>

                <div style={{ fontWeight: 700, fontSize: "12px", marginBottom: 2 }}>
                  Top drivers of higher fuel rate
                </div>
                <ShapTable rows={contributionRows(explanation).pos} />
                <div style={{ fontWeight: 700, fontSize: "12px", margin: "10px 0 2px" }}>
                  Top drivers of lower fuel rate
                </div>
                <ShapTable rows={contributionRows(explanation).neg} />

                <div style={{ fontSize: "11px", color: "var(--muted-2)", marginTop: 10 }}>
                  {explanation.limitations?.note ?? "SHAP is attribution, not causality."}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Validation details (expandable) */}
        {!running && validation && (
          <div
            className={`expandable ${showValidation ? "open" : ""}`}
            style={{ marginTop: 10 }}
          >
            <button
              type="button"
              className="expandable-trigger"
              onClick={() => setShowValidation((v) => !v)}
              aria-expanded={showValidation}
            >
              <span>
                ✅ Guardrail validation — {validation.passed}/{validation.total_checks} passed
              </span>
              <span className="caret">▾</span>
            </button>
            {showValidation && (
              <div className="expandable-body">
                <div className="validation-grid">
                  {(validation.issues ?? []).map((issue) => (
                    <div
                      key={issue.check_id}
                      className={`validation-chip ${issue.passed ? "pass" : "fail"}`}
                    >
                      <span className="cid">{issue.check_id}</span>
                      <span className="muted-2 small">{issue.detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
