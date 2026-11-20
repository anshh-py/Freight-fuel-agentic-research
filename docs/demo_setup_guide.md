# Demo Setup Guide — Heavy-Truck Fuel-Rate Prediction & Explainable Agentic Driver-Feedback System

Milestone **M6.5.1** — demo-environment readiness. This guide establishes a
reliable, reproducible local environment for recording the final project
demonstration.

> Scope: local development demo only. Everything is **synthetic**. No real
> truck, no live OBD, no external/LLM service, no deployment, no video yet.

---

## 1. Prerequisites

- Python **3.11+** (verified against 3.13) with the project's installed deps:
  - `fastapi`, `uvicorn`, `pydantic`
  - `pandas`, `numpy`, `scikit-learn`, `joblib`, `xgboost`, `shap`
- Node.js **18+** and npm (for the React/Vite frontend).
- No environment variables, API keys, or external accounts are required.
  The agent backend is **deterministic** (`configs/agent_config.json` →
  `routing.backend = "deterministic"`, `is_llm = false`). No LLM provider is
  configured, so there is nothing to pay for or authenticate against.

No undocumented local-only paths or secrets are required. All data and model
paths are resolved relative to the repository root (`PROJECT_ROOT`).

## 2. Repository layout (what the demo needs)

| Item                              | Location                                                                                                            | Verified                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| Model artifacts                   | `reports/m1/artifacts/{full_operational,telematics_only}/` (contains `xgboost.joblib`, `preprocessor.joblib`, etc.) | present, loads                |
| Replay dataset (master telemetry) | `data/synthetic_heavy_truck_research/telemetry_master_963697.csv.gz`                                                | loads (963,697 rows)          |
| Replay catalog metadata           | `data/synthetic_heavy_truck_research/truck_metadata.csv`, `trip_metadata.csv`                                       | loads (29 trucks / 540 trips) |
| RAG corpus                        | `data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl`                                                    | present                       |
| Feature spec                      | `configs/ml_synthetic_feature_spec.json`                                                                            | loads                         |
| Agent config                      | `configs/agent_config.json`                                                                                         | loads                         |

## 3. Environment configuration

- **No `.env` file is needed.** Defaults are baked into
  `src/features/spec.py` / `src/api/gateway.py`:
  - Model dir → `reports/m1/artifacts`
  - Corpus path → `data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl`
  - Replay data → same dataset folder as the master telemetry
- The frontend Vite dev server proxies `/api` and `/health` to
  `http://127.0.0.1:8000` (`frontend/vite.config.ts`). No CORS config or
  `VITE_API_BASE` override is needed in development.

## 4. Backend startup

> Run **from the repository root** as a module (so `src` is importable):

```bash
python -m src.api.server
```

- Serves on `http://0.0.0.0:8000` (uvicorn, no reload).
- Note: `python src/api/server.py` alone fails with `ModuleNotFoundError: No
module named 'src'` — always use `python -m src.api.server`.

## 5. Frontend startup

```bash
cd frontend
npm install        # first time only
npm run dev
```

- Serves on `http://localhost:5173`.
- The backend must already be running on `127.0.0.1:8000` before the page
  loads, otherwise the cockpit shows **"Backend offline"**.

## 6. Health/readiness checks

```bash
# 1) Backend health
curl.exe -s http://127.0.0.1:8000/health
# => {"status":"ok","service":"truck-fuel-gateway","milestone":"M6.2",
#     "synthetic_only":true,"active_sessions":N}

# 2) Full automated readiness report (creates a throwaway session + one message turn)
python scripts/verify_demo_env.py
```

`scripts/verify_demo_env.py` checks: `/health`, `/api/v1/features`,
session creation, replay catalog, replay trip load, and one delivered message
turn (prediction + SHAP). Verified output:

```
[ok] health:          {"status":"ok", ...}
[ok] features:        {"feature_sets":["full_operational","telematics_only"],"field_count":15}
[ok] create_session:   {"session_id":"...","feature_set":"full_operational"}
[ok] replay_catalog:   {"trucks":29,"trips":540,"first_truck":"truck_001"}
[ok] replay_trip:      {"trip_id":"trip_000001","points":20,"truck_id":"truck_001"}
[ok] message_turn:     {"status":200,"delivered":true,"fuel_rate_lph":41.2622,"expected_value":41.5477,"model":"xgboost"}
OVERALL: READY
```

## 7. Deterministic demo scenario

- **Replay trip:** `trip_000001` (truck `truck_001`, synthetic 32T, city route).
- **Sample index:** `2` (third row of the 360-point downsampled view, timestamp
  `2023-01-28T12:48:17Z`). This is the **only** replay sample for which a
  complete prediction + SHAP result was verified end-to-end (M6.4 live
  session → 41.26 L/h, SHAP expected 41.55). It is an accelerating, non-idle
  driving point. Exact row (from
  `GET /api/v1/replay?trip_id=trip_000001&sample=360`):

| Field                                                        | Value      |
| ------------------------------------------------------------ | ---------- |
| Engine RPM                                                   | 1041.9     |
| Speed                                                        | 11.96 km/h |
| Engine load                                                  | 23.42 %    |
| Throttle position                                            | 28.75 %    |
| Road grade                                                   | −0.06 %    |
| Fuel rate _(synthetic replay target, displayed as "target")_ | 12.222 L/h |

> **Row count clarification.** `trip_metadata.csv` lists `trip_000001` at
> **2,013 records** (all rows). The replay API reports **1,993** as
> `record_count_total` because the server filters to `ignition_status == 1`
> (driving-only) rows before sampling. Both numbers are correct; they count
> different scopes (all telemetry vs. driving rows).

### Demo walkthrough (5 features)

1. **Replay selection & playback** — cockpit opens with `trip_000001`
   selected. Click **▶ Play**; verify the replay readout advances and the
   Speed/Engine RPM gauges mirror the replayed telemetry.
2. **Pause & transfer** — click **⏸ Pause**, then **⬇ Load this replay sample
   into prediction inputs**. The five manual engine controls (Engine RPM,
   Speed, Engine load, Throttle, Road grade) now carry the paused sample's
   values.
3. **Fuel-rate prediction** — set/confirm the Advanced vehicle inputs are
   complete, then click **▶ Run AI prediction**. A validated backend response
   is required; a failed/blocked response shows an error, never a fuel number.
4. **SHAP explanation** — the prediction panel shows the XGBoost + SHAP
   result: predicted fuel rate, expected value (base rate), top positive and
   negative feature contributions, and the reconstruction/validation status.
5. **Assistant interaction** — ask the assistant a follow-up (e.g. _"Explain
   what I should check to reduce this fuel rate"_). The assistant reuses the
   same gateway session and carried inputs; unsupported or safety-relevant
   requests are declined rather than answered with invented facts.

> Verified reference (backend, saved model) **for replay sample index 2**
> (RPM 1041.9, speed 11.96, load 23.42%, throttle 28.75%, grade −0.06%,
> with the cockpit's static defaults): predicted fuel rate **41.2622 L/h**,
> SHAP expected value **41.5477**, model `xgboost`, validation 9/9,
> reconstruction pass.
>
> ⚠️ **Distinct scenario.** The prediction captured in evidence screenshot
> `02_prediction_delivered.png` (**62.61 L/h**) is a **separate manual-input**
> run (user-entered controls), not the replay-sample prediction above. Do not
> present 62.61 as the output of the replay-sample scenario. Exact prediction
> varies with the currently-entered static inputs (vehicle class / masses /
> temperature / torque), which come from the manual controls, not the replay.

## 8. Known limitations

- The replay trip's truck metadata is **32T**, while the cockpit's default
  static vehicle inputs are **40T / 24,000 kg / 12,000 kg**. The replay
  supplies only the five dynamic controls; the static inputs are the user's
  own values. This mismatch is pre-existing and is not corrected in this
  milestone (no model/UI changes).
- Replay fuel-rate values are **formula-generated targets**, not measured
  consumption; they are labelled clearly in the UI and must not be presented
  as model predictions.
- The prediction is an **estimate from a synthetic-trained model**; the UI
  carries the "Synthetic-only" disclosure, and SHAP is presented as attribution
  (not causality).
- The live-UI interactive click-through (play/pause/transfer) was verified in a
  real single-session browser run in M6.4; in this milestone the deterministic
  data, endpoints, and end-to-end prediction path are verified at the API and
  frontend-boot level. A headless-browser automation of the `▶ Play` / transfer
  buttons is not relied on here (the buttons are timing-sensitive).
- No video recording has been started (as required).

## 9. What was verified vs. inspected

| Step                                                           | Method                | Result |
| -------------------------------------------------------------- | --------------------- | ------ |
| Backend starts via `python -m src.api.server`                  | executed              | ✅     |
| Frontend starts via `npm run dev` (proxies → 127.0.0.1:8000)   | executed              | ✅     |
| Frontend connects to backend (session created, catalog loaded) | executed (browser)    | ✅     |
| Model artifacts present & load (xgboost prediction)            | executed              | ✅     |
| Replay catalog (29 trucks / 540 trips) & trip data             | executed              | ✅     |
| Session can be created                                         | executed              | ✅     |
| No undocumented local-only paths / secrets needed              | inspected config/spec | ✅     |
| Deterministic demo scenario (trip_000001, sample 2)            | executed (probe)      | ✅     |
