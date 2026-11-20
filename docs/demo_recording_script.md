# M6.5 Demo — Recording Script & Shot List

> ⚠️ **Recording status: UNRECORDED.** This document is the complete, ready-to-use
> recording script and shot list for the end-to-end demonstration video. The
> actual video was **not** recorded in the available environment: no screen-
> capture tool is installed (verified: `ffmpeg` is not available), and the
> headless browser capture does not produce video. **No video file exists.**
> Every claim below is of the _script_, not of a recorded deliverable.

---

## 1. Voiceover / narration outline

The narration must explicitly say (and the on-screen text confirms):

- "Everything you see is a **simulator connected to a research gateway**. There
  is **no real truck**, no live OBD hardware, and no real-world measurement."
- "All telemetry and all prediction target labels are **synthetic**, generated
  for research. They are not from Indian trucks and not from any real fleet."
- "The model is an XGBoost regressor trained on synthetic data. It predicts an
  **instantaneous fuel rate in litres per hour** (L/h)."
- "The agent routes the request, runs the prediction, produces a SHAP
  explanation, checks it against safety rules, and only then presents an answer.
  When it cannot ground an answer, it declines instead of guessing."
- "Results here are **synthetic training/evaluation results** — not real-world
  predictive validation. No real-world validation has been established."

## 2. Shot list (8 shots, single continuous session)

Setup: backend running (`python -m src.api.server`), frontend running
(`npm run dev` in `frontend/`), browser opened to `http://localhost:5173`.
Use a real screen recorder (e.g. OBS or the OS screen recorder) at 1080p.

### Shot 1 — Intro & architecture (system map / repo tree)

- Show `README.md` and `MILESTONES.md` top lines.
- Verbally walk: telemetry dataset → features → saved model artifacts
  (`reports/m1/artifacts/full_operational/xgboost.joblib`) → gateway
  (`src/api/gateway.py`) → frontend cockpit (`frontend/src/App.tsx`).
- Script: "Data is synthetic; the model is pretrained; the gateway is a
  deterministic research service — no external LLM, no API keys."

### Shot 2 — Interactive cockpit loads

- Load `http://localhost:5173/`. Wait for the green "Backend connected" banner
  and a session id.
- Point at the tachometer/speedometer, manual controls, advanced inputs,
  analytics, and the assistant.
- Script: "This is the reusable cockpit. Everything shown is driven by the
  research gateway."

### Shot 3 — Manual controls & presets

- Move a slider (e.g. Engine RPM or Speed) and click a preset (e.g. "Highway
  cruising"). Show the "controls changed — rerun prediction" warning.
- Script: "The five manual controls are linked to the gauges; the advanced
  (static truck) inputs are validated and never silently defaulted."

### Shot 4 — Synthetic replay, playback & transfer

- In the replay panel select trip `trip_000001` (truck `truck_001`), press
  **▶ Play**, let it advance, then **⏸ Pause**.
- ⭐ **Pause at sample index 2** (the third row of the 360-point view): the
  readout/gauges should show roughly **Engine RPM 1041.9, Speed 11.96 km/h,
  engine load 23.42%, throttle 28.75%, grade −0.06%**.
- Click **⬇ Load this replay sample into prediction inputs**; show the transfer
  notice and the updated manual controls.
- Script: "This replay is **synthetic** telemetry. Its fuel-rate column is a
  formula-generated target, not a measurement."
- **Note on indices:** earlier evidence (`04_replay_sample_transfer.png`)
  captured a pause at **sample index 1** (RPM 815, speed 5.8 km/h). The
  canonical demo scenario uses **sample index 2**; both are genuine pauses, but
  they are different sample points. Keep this one at index 2 so the prediction
  matches the verified 41.26 L/h reference.

### Shot 5 — Fuel-rate prediction

- Click **▶ Run AI prediction**.
- Highlight the delivered result (fuel rate, `delivered` badge, validation
  `9/9`).
- Script: "The saved model computes an estimate, and the validator gates it
  before it is shown."

### Shot 6 — SHAP explanation

- Expand **🔬 Technical SHAP explanation**.
- Show expected value, top positive/negative contributions, and the SHAP
  reconstruction "passed" status.
- Script: "SHAP shows which inputs most influenced this estimate — attribution,
  **not** causation. The reconstruction check confirms the numbers are
  self-consistent."

### Shot 7 — Assistant interaction (evidence-grounded & limitations)

- Type "How can I reduce fuel consumption?" (or use the suggested chip).
- Show the assistant's response. Use the _grounded_ knowledge answer if it is
  delivered; otherwise show the honest decline ("could not produce a validated,
  grounded answer", `not delivered`).
- Script: "The assistant answers from retrieved synthetic knowledge and declines
  to invent unsupported advice — it never fabricates a fuel-saving guarantee."

### Shot 8 — Closing / research scope

- Return to the repo tree / `README.md`.
- Script: "This is a research demonstration on synthetic data. It shows a
  reproducible, safety-gated agentic prediction pipeline. It is not a validated
  real-world truck product, and no Indian-truck or live-fleet validation is
  claimed."

## 3. Exact on-screen steps (for the operator)

1. Run backend: `python -m src.api.server`
2. Run frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5173/`; wait for green "Backend connected".
4. (Shot 3) drag any control slider; click "Highway cruising" preset.
5. (Shot 4) select trip `trip_000001`; Play; **Pause when the readout shows
   sample index 2** (≈RPM 1041.9, ≈speed 11.96 km/h); "Load this replay
   sample into prediction inputs".
6. (Shot 5) click "▶ Run AI prediction"; wait for delivered result.
7. (Shot 6) click "🔬 Technical SHAP explanation".
8. (Shot 7) type "How can I reduce fuel consumption?" and press Send.
9. (Shot 8) switch to the code editor showing `README.md`.

## 4. Review checklist before publishing the video

- [ ] No real truck/OBD/Indian-fleet implication anywhere in narration.
- [ ] The words **synthetic**, **estimate**, and **not validated on real-world
      data** are spoken and on screen.
- [ ] The safety decline (if it occurs) is shown rather than edited out.
- [ ] The video matches the live app behavior captured in the screenshot
      evidence (`evidence/m6_5_demo/`).
