# M6.5 — Reproducibility & Research Documentation

Project: **Cross-Truck Fuel-Rate Prediction and Explainable Agentic
Driver-Feedback System** (heavy-truck digital cockpit).

> **Scope & integrity statement.** This document distinguishes four types of
> result, and nothing here conflates them:
>
> 1. **Synthetic training/evaluation results** — offline, on synthetic data
>    (e.g. `reports/m1/metrics_report.json`, `reports/m5/m5_5/`).
> 2. **Actual software integration tests** — real code, mocked/real endpoints
>    (frontend vitest + backend pytest).
> 3. **Live-browser verification** — real running app in a real browser
>    (the M6.5 screenshot evidence and the live prediction run recorded below).
> 4. **Real-world predictive validation** — **NOT established.** There is no
>    real truck, no live OBD, no Indian-truck fleet, no validated real-world
>    fuel measurement anywhere in this work.

---

## 1. Architecture & components

| Component                     | Source                                                                            | Role                                                                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Telemetry dataset (synthetic) | `data/synthetic_heavy_truck_research/telemetry_master_963697.csv.gz`              | Feature inputs for prediction (963,697 rows)                                                                                   |
| Feature spec                  | `configs/ml_synthetic_feature_spec.json`                                          | Feature sets (`full_operational`, `telematics_only`), ranges, leakage rules                                                    |
| Saved model artifacts         | `reports/m1/artifacts/{full_operational,telematics_only}/`                        | XGBoost regressor + preprocessor + baselines (joblib)                                                                          |
| RAG corpus (synthetic)        | `data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl`                  | Educational passages (non-OEM) for retrieval grounding                                                                         |
| Fuel-rate tool                | `src/tools/fuel_rate_prediction.py`                                               | Loads saved model; validates; predicts L/h                                                                                     |
| Explanation tool              | `src/tools/prediction_explanation.py`                                             | Instance-level SHAP + reconstruction check; non-causal                                                                         |
| Retrieval tool                | `src/tools/knowledge_retrieval.py`                                                | Deterministic vector store over corpus                                                                                         |
| Session agent                 | `src/agent/` (`driver_session.py`, `driver_intents.py`, `driver_conversation.py`) | Multi-turn state, intent routing, clarification, orchestration                                                                 |
| Response generation           | `src/agent/response_generator.py`                                                 | Driver-facing + technical text                                                                                                 |
| Response validation           | `src/agent/response_validation.py`                                                | Deterministic guardrail battery (synthetic disclaimer, non-causal, no OEM mislabeling, value preservation, failure disclosure) |
| Gateway                       | `src/api/gateway.py`                                                              | Thin adapter over the frozen M6.1 service                                                                                      |
| HTTP API                      | `src/api/server.py`                                                               | FastAPI server (`/health`, sessions, replay, messages)                                                                         |
| Frontend cockpit              | `frontend/src/` (React + Vite)                                                    | Gauge cockpit, replay, manual/advanced inputs, prediction+SHAP panel, analytics, assistant                                     |

No external LLM, no paid service, no API keys, no database. The agent backend is
deterministic (`configs/agent_config.json` → `routing.backend="deterministic"`,
`is_llm=false`). All sessions are in-memory.

## 2. Installation prerequisites

- Python **3.11+** (verified on 3.13) with: `fastapi`, `uvicorn`, `pydantic`,
  `pandas`, `numpy`, `scikit-learn`, `joblib`, `xgboost`, `shap`.
- Node.js **18+** and npm.
- No environment variables, secret keys, or external accounts. No `.env` file
  is required or used. No funds are needed.

## 3. Environment variables & configuration

None required. All paths resolve relative to the repository root
(`PROJECT_ROOT` in `src/features/spec.py`): model dir
`reports/m1/artifacts`, corpus path, and replay data. The frontend Vite dev
server proxies `/api` and `/health` to `http://127.0.0.1:8000`
(`frontend/vite.config.ts`); no `VITE_API_BASE` override is needed in dev.

## 4. Startup commands

```bash
# Backend — run from the repository root as a module:
python -m src.api.server
#   serves on http://0.0.0.0:8000  (uvicorn)
#   NOTE: `python src/api/server.py` alone FAILS (ModuleNotFoundError: 'src');
#   always use `python -m src.api.server`.

# Frontend:
cd frontend
npm install        # first time only
npm run dev        # serves on http://localhost:5173
```

## 5. Health checks & readiness

```bash
curl.exe -s http://127.0.0.1:8000/health
# => {"status":"ok","service":"truck-fuel-gateway","milestone":"M6.2",
#     "synthetic_only":true,"active_sessions":N}

python scripts/verify_demo_env.py
# => OVERALL: READY  (health, features, session, replay catalog/trip,
#     delivered prediction + SHAP all checked)
```

## 6. Model artifacts

`reports/m1/artifacts/` holds, per feature set, `xgboost.joblib` (the demo
model), `preprocessor.joblib`, and baselines (`mean_dummy`, `median_dummy`,
`random_forest`, `ridge`). Verified loading: the gateway constructs tools with
these real artifacts (no mocks in the default path) and a live prediction turn
returns a delivered fuel estimate (see §9).

## 7. Dataset provenance & synthetic-data generation

The prediction and replay data are **synthetic** telemetry generated for this
research project (see `DATA_SOURCES.md` and `src/data/synthetic_generation/`).
The **target variable is `fuel_rate_lph`** — litres of fuel per hour. Replay
fuel-rate values are **formula-generated targets**, not measured consumption.
No real FEAD/truck OBD data is used anywhere in the running demo system.

## 8. Feature definitions & target

- `full_operational` (15 fields): vehicle class (32T/40T/49T), gross vehicle
  weight kg, payload kg, vehicle mass kg, speed kph, acceleration m/s², road
  grade %, ignition status (0/1), engine rpm, engine load %, throttle %, torque
  N·m, coolant °C, intake-air °C, ambient °C (one-hot truck class).
- `telematics_only` drops the four engine-bus fields (engine rpm, load,
  throttle, torque).
- Target: **fuel rate (L/h)**.

## 9. What the demo performs (verified)

### Replay-sample scenario (canonical, sample index 2)

In the verified M6.4 live session and the M6.5 API probe (`trip_000001`,
`truck_001`, synthetic 32T, city route):

1. **Replay selection & playback** — `trip_000001` selected; Play/Pause
   advances the readout.
2. **Sample transfer** — "⬇ Load this replay sample into prediction inputs"
   copies the paused sample into the five manual controls. Canonical sample:
   **index 2** (RPM 1041.9, speed 11.96 km/h, load 23.42%, throttle 28.75%,
   grade −0.06%). A distinct earlier capture (`04_replay_sample_transfer.png`)
   paused at **index 1** (RPM 815, speed 5.8 km/h) — a different genuine
   pause, not the canonical scenario.
3. **Prediction (replay scenario)** — verified reference: **predicted fuel
   rate 41.2622 L/h**, SHAP expected value **41.5477**, model `xgboost`,
   validation 9/9, reconstruction pass.

### Separate manual-input run (evidence screenshots 02/03)

The live browser run that produced evidence screenshot `02_prediction_delivered.png`
(**62.61 L/h**) and `03_shap_explanation.png` (SHAP expected 41.55 L/h,
top positive torque +38.86, top negative `engine_rpm` −8.79) used a
**separate manual-input configuration** (user-entered controls), not the
replay-sample scenario above. These two prediction runs are distinct input
sets and must not be presented as the same scenario.

### Assistant interaction (verified)

A typed question returns a grounded answer or an honest decline; verified live:
"How can I reduce fuel consumption?" was declined ("could not produce a
validated, grounded answer… not delivered") because no sufficiently relevant
evidence was retrieved — the assistant does not invent advice.

## 10. Test commands & verified outputs

Frontend (React/Vite):

```bash
cd frontend
npm test                 # vitest run — all component/lib tests
npm run typecheck        # tsc --noEmit
npm run build            # tsc --noEmit && vite build
```

Verified (M6.4/M6.5 focused suites):

- `vitest run` on prediction/SHAP/assistant/replay/history suites:
  **28/28 passed**.
- `npm run build` → **exit 0** (only a pre-existing >500 kB chunk-size
  warning).

Backend (Python):

```bash
python -m pytest tests/test_m6_2_server.py tests/test_m4_2_driver_session.py \
  tests/test_m4_4_driver_conversation.py -q
```

Verified: **33 passed**. (Targeted suites only; a full-suite audit is out of
scope for M6.5.)

## 11. Agent behavior & evaluation

Behavior: intent routing → clarification when inputs are missing/invalid →
prediction/explanation/retrieval → response generation → guardrail validation →
deliver only if no error-severity failure. The M5.5 ablation (`reports/m5/m5_5/`)
scores the baseline full pipeline **0.920** (185/201 applicable criteria;
safety-critical scenarios 0). This is an **engineering ablation on synthetic
data**, not human-subject or real-world validation.

## 12. Known limitations & unresolved issues

- **No real-world validation.** No Indian-truck fleet, no live truck, no
  validated real-world fuel measurement. All results are synthetic.
- **Replay truck metadata (32T) differs from the cockpit's default static
  inputs (40T / 24,000 / 12,000 kg).** The replay supplies only the five
  dynamic controls; static inputs are the user's own. Pre-existing; not
  corrected.
- **In-memory sessions** are lost on backend restart.
- **The M5.5 evaluation status.** The M5.5 ablation artifacts exist and are
  reported above. However, M5.5 is an **offline engineering ablation** with a
  deterministic rubric; it is **not** independent human-subject evaluation and
  does not establish real-world validity. It is therefore reported as
  **substantively complete as an engineering ablation but not as validation** —
  it is not silently marked as finished real-world validation.
- **Frontend history bookkeeping** (a follow-up re-explanation does not append a
  duplicate chart point) is enforced and regression-tested; this is a UI-level
  guard, not a claim about the underlying model.
- The headless browser capture cannot record video in this environment (no
  screen-capture tool), so the demo video is **unrecorded**; the script is in
  `docs/demo_recording_script.md`.

## 13. Reproduction from a clean environment

1. `git clone` the repository; `cd truck-fuel-agentic-research`.
2. Install backend deps (`pip install -r requirements.txt` per repo config) and
   `cd frontend && npm install`.
3. Verify artifacts exist (`reports/m1/artifacts/`) and datasets are present
   (`data/synthetic_heavy_truck_research/`).
4. Start backend: `python -m src.api.server`.
5. Start frontend: `cd frontend && npm run dev`.
6. Open `http://localhost:5173/`; wait for the green "Backend connected" banner.
7. Run `python scripts/verify_demo_env.py` → `OVERALL: READY`.
8. Follow the demo walkthrough in `docs/demo_setup_guide.md` (scenario:
   `trip_000001`, sample index 2).
9. Run the frontend/backend test commands in §10 to reproduce the verified
   outputs.

## 14. Evidence location

Genuine live-browser screenshots and the evidence index:
`evidence/m6_5_demo/` (see `evidence/m6_5_demo/index.md`). Model/agent metric
artifacts: `reports/m1/metrics_report.json`, `reports/m5/m5_5/`.
