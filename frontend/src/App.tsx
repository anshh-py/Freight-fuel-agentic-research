import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type {
  ApiResponse,
  FeatureCatalog,
  FeatureSetName,
  ReplayPoint,
  ReplayResponse,
} from "./api/types";
import Header from "./components/Header";
import Gauge from "./components/Gauge";
import ManualControls from "./components/ManualControls";
import AdvancedInputs from "./components/AdvancedInputs";
import PredictionPanel from "./components/PredictionPanel";
import AnalyticsCharts, { type HistoryPoint } from "./components/AnalyticsCharts";
import AssistantPanel from "./components/AssistantPanel";
import VehicleContextCard from "./components/VehicleContextCard";
import ReplayPanel, {
  type ReplayLoadAction,
} from "./components/ReplayPanel";
import {
  buildPayload,
  DEFAULT_FEATURE_SET,
  DEFAULT_INPUTS,
  PREDICT_QUERY,
  type Scenario,
} from "./lib/cockpit";
import { nextHistoryEntry } from "./lib/predictionHistory";

/* Gauge calibration ---------------------------------------------------- */
const TACHO = {
  min: 0,
  max: 2200,
  ticks: [0, 500, 1000, 1500, 2000, 2200],
  segments: [
    { from: 0, to: 900, color: "#35c4ff" },
    { from: 900, to: 1600, color: "#34d399" },
    { from: 1600, to: 2000, color: "#fbbf24" },
    { from: 2000, to: 2200, color: "#fb7185" },
  ],
};

const SPEEDO = {
  min: 0,
  max: 100,
  ticks: [0, 20, 40, 60, 80, 100],
  segments: [
    { from: 0, to: 80, color: "#34d399" },
    { from: 80, to: 90, color: "#fbbf24" },
    { from: 90, to: 100, color: "#fb7185" },
  ],
};

export default function App() {
  const [catalog, setCatalog] = useState<FeatureCatalog | null>(null);
  const [backendStatus, setBackendStatus] = useState<"connecting" | "ok" | "error">("connecting");
  const [featureSet, setFeatureSet] = useState<FeatureSetName>(DEFAULT_FEATURE_SET);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, string | number>>(DEFAULT_INPUTS);
  const [activeScenario, setActiveScenario] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  const [running, setRunning] = useState(false);
  const [predictionResponse, setPredictionResponse] = useState<ApiResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [conversationState, setConversationState] = useState<string>("awaiting_input");

  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [replay, setReplay] = useState<ReplayResponse | null>(null);
  const [replayPoint, setReplayPoint] = useState<ReplayPoint | null>(null);
  const [replayProgress, setReplayProgress] = useState(0);
  const [mirrorReplay, setMirrorReplay] = useState(false);
  const [transferNotice, setTransferNotice] = useState<string | null>(null);

  const initStarted = useRef(false);

  /* startup: health + catalog + session + replay --------------------- */
  useEffect(() => {
    if (initStarted.current) return;
    initStarted.current = true;

    (async () => {
      try {
        await api.health();
        setBackendStatus("ok");
      } catch {
        setBackendStatus("error");
      }

      try {
        const cat = await api.features();
        setCatalog(cat);
        setInputs((prev) => {
          const next = { ...prev };
          for (const f of Object.keys(cat.fields)) {
            if (!(f in next)) {
              const meta = cat.fields[f];
              next[f] = meta.options?.[0] ?? meta.min ?? 0;
            }
          }
          return next;
        });
      } catch {
        /* leave catalog null */
      }

      try {
        const s = await api.createSession(DEFAULT_FEATURE_SET);
        setSessionId(s.session_id);
      } catch {
        /* session creation failed */
      }

    })();
  }, []);

  const handleLoadReplay = useCallback((r: ReplayResponse) => {
    setReplay(r);
  }, []);

  const handleReplayPoint = useCallback((p: ReplayPoint | null) => {
    setReplayPoint(p);
  }, []);

  const handleReplayProgress = useCallback((i: number) => {
    setReplayProgress(i);
  }, []);

  const handleLoadSampleToInputs = useCallback((a: ReplayLoadAction) => {
    setInputs((prev) => ({ ...prev, ...a }));
    setActiveScenario(null);
    setDirty(true);
    setPredictionResponse(null);
    setPredictionError(null);
    setTransferNotice(
      `Loaded replay sample → manual inputs: ` +
        `RPM ${Math.round(a.engine_rpm)}, speed ${a.speed_kph.toFixed(0)} km/h, ` +
        `load ${a.engine_load_pct.toFixed(0)}%, throttle ${a.throttle_position_pct.toFixed(0)}%, ` +
        `grade ${a.road_grade_pct.toFixed(2)}%. Re-run prediction to use these.`,
    );
    window.setTimeout(() => setTransferNotice(null), 6000);
  }, []);

  const recordPrediction = useCallback(
    (resp: ApiResponse) => {
      setHistory((h) => {
        const entry = nextHistoryEntry(h, resp, inputs);
        return entry ? [...h, entry] : h;
      });
      setDirty(false);
    },
    [inputs],
  );

  const handleInput = useCallback((field: string, value: string | number) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
    if (["engine_rpm", "speed_kph", "engine_load_pct", "throttle_position_pct", "road_grade_pct"].includes(field)) {
      setActiveScenario(null);
    }
    setDirty(true);
    setPredictionResponse(null);
    setPredictionError(null);
  }, []);

  const handleScenario = useCallback((s: Scenario) => {
    setInputs((prev) => ({ ...prev, ...s.overrides }));
    setActiveScenario(s.id);
    setDirty(true);
    setPredictionResponse(null);
    setPredictionError(null);
  }, []);

  const handleFeatureSetChange = useCallback(
    (f: FeatureSetName) => {
      setFeatureSet(f);
      setHistory([]);
      setPredictionResponse(null);
      setPredictionError(null);
      if (sessionId) {
        api.deleteSession(sessionId).catch(() => undefined);
      }
      api.createSession(f).then((s) => setSessionId(s.session_id)).catch(() => undefined);
    },
    [sessionId],
  );

  const runPrediction = useCallback(async () => {
    if (!sessionId || running) return;
    const payload = buildPayload(inputs, featureSet);
    setRunning(true);
    setPredictionError(null);
    setPredictionResponse(null);

    try {
      const resp = await api.sendMessage(sessionId, PREDICT_QUERY, payload);
      setPredictionResponse(resp);
      setConversationState(resp.conversation_state);
      if (resp.error) {
        setPredictionError(resp.error.message);
      } else {
        recordPrediction(resp);
        if (!resp.delivered && resp.driver_intent !== "clarifying") {
          setPredictionError(
            "The backend did not produce a validated answer for this turn.",
          );
        }
      }
    } catch (err) {
      setPredictionResponse(null);
      setPredictionError(
        err instanceof ApiError
          ? `Backend error (${err.status}): ${err.message}`
          : "Could not reach the backend gateway.",
      );
    } finally {
      setRunning(false);
    }
  }, [sessionId, running, inputs, featureSet, recordPrediction]);

  const featureSets = catalog ? Object.keys(catalog.feature_sets) : [DEFAULT_FEATURE_SET];
  const gaugeRpm = mirrorReplay && replayPoint ? replayPoint.engine_rpm : Number(inputs.engine_rpm ?? 0);
  const gaugeSpeed = mirrorReplay && replayPoint ? replayPoint.speed_kph : Number(inputs.speed_kph ?? 0);

  return (
    <div className="app">
      <Header
        backendStatus={backendStatus}
        sessionId={sessionId}
        featureSet={featureSet}
        featureSets={featureSets}
        onFeatureSetChange={handleFeatureSetChange}
      />

      <main className="main">
        {/* Cockpit gauges */}
        <section className="card area-cockpit">
          <div className="card-head">
            <div className="card-title">
              <span className="icon">🏎️</span> Live cockpit
            </div>
            {dirty ? (
              <span className="status-line status-warn">
                <span className="dot dot-warn" /> controls changed — rerun prediction
              </span>
            ) : (
              <span className="status-line status-idle">
                <span className="dot dot-idle" /> idle
              </span>
            )}
          </div>
          <div className="card-pad">
            <div className="cockpit-grid">
              <Gauge
                label="Engine RPM"
                unit="rpm"
                min={TACHO.min}
                max={TACHO.max}
                value={gaugeRpm}
                segments={TACHO.segments}
                ticks={TACHO.ticks}
                format={(v) => String(Math.round(v))}
              />
              <Gauge
                label="Speed"
                unit="km/h"
                min={SPEEDO.min}
                max={SPEEDO.max}
                value={gaugeSpeed}
                segments={SPEEDO.segments}
                ticks={SPEEDO.ticks}
                format={(v) => String(Math.round(v))}
              />
            </div>
            <div className="gauge-mirror">
              <label className="switch">
                <input
                  type="checkbox"
                  checked={mirrorReplay}
                  onChange={(e) => setMirrorReplay(e.target.checked)}
                />
                <span>Mirror active replay into gauges</span>
              </label>
              <span className="muted-2">
                {mirrorReplay && replayPoint
                  ? `Showing replay: ${Math.round(replayPoint.engine_rpm)} rpm · ${replayPoint.speed_kph.toFixed(0)} km/h`
                  : "Gauges show manual controls"}
              </span>
            </div>
            <div className="gauge-region">
              Tachometer regions — blue: idle · green: economy · amber: power · red: limit.
              Speed — green: cruise · amber: approach limit · red: limit.
            </div>
          </div>
        </section>

        {/* Synthetic telemetry replay */}
        <ReplayPanel
          onLoadReplay={handleLoadReplay}
          onPoint={handleReplayPoint}
          onProgress={handleReplayProgress}
          onLoadSampleToInputs={handleLoadSampleToInputs}
        />

        {transferNotice && (
          <div className="transfer-notice" role="status">
            <span>🔄</span>
            <span>{transferNotice}</span>
          </div>
        )}

        {/* Manual controls + advanced inputs */}
        <ManualControls
          catalog={catalog}
          inputs={inputs}
          activeScenario={activeScenario}
          onInput={handleInput}
          onScenario={handleScenario}
        />

        <section className="card">
          <div className="card-head">
            <div className="card-title">
              <span className="icon">🧩</span> Advanced vehicle inputs
            </div>
            <span className="tag">validated · not invented</span>
          </div>
          <div className="card-pad">
            {catalog ? (
              <AdvancedInputs
                catalog={catalog}
                featureSet={featureSet}
                inputs={inputs}
                onInput={handleInput}
              />
            ) : (
              <div className="skeleton pulse" style={{ height: 120 }} />
            )}
          </div>
        </section>

        {/* Prediction */}
        <PredictionPanel
          running={running}
          response={predictionResponse}
          errorMsg={predictionError}
          canRun={!!sessionId && !!catalog}
          onRun={runPrediction}
        />

        {/* Vehicle context */}
        <VehicleContextCard
          catalog={catalog}
          featureSet={featureSet}
          inputs={inputs}
          scenarioLabel={activeScenario ? SCENARIO_LABEL(activeScenario) : null}
          replay={replay}
          conversationState={conversationState}
        />

        {/* Analytics */}
        <AnalyticsCharts
          history={history}
          replay={replay}
          replayProgress={replayProgress}
        />

        {/* Assistant */}
        <AssistantPanel
          sessionId={sessionId}
          disabled={!sessionId}
          onPredictionSuccess={recordPrediction}
        />
      </main>
    </div>
  );
}

function SCENARIO_LABEL(id: string): string {
  const map: Record<string, string> = {
    idle: "Idle",
    city: "City driving",
    highway: "Highway cruising",
    acceleration: "Acceleration",
    uphill: "Uphill driving",
    downhill: "Downhill driving",
  };
  return map[id] ?? id;
}
