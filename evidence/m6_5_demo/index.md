# M6.5 Demo Evidence Index

All images in this folder are **genuine screenshots** captured from the live
application (React/Vite frontend `http://localhost:5173` against the local
FastAPI backend `http://127.0.0.1:8000`) during the M6.5 verification run on
2026-11-20. They are real browser renderings — no staged, illustrated, or
hand-edited states.

> The visible footer/sidebar text in some shots is white-on-dark and may be
> cropped at the viewport edge; the core panel content is the source of truth.

## Files

| File                            | What it demonstrates                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Evidence captured                                                                                                        |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `01_main_cockpit.png`           | Booted cockpit: header (Backend connected, session id, Synthetic data, Truck T-001 · 40T), live tachometer + speedometer, manual engine controls, advanced vehicle inputs, replay panel, analytics, assistant.                                                                                                                                                                                                                                                                           | HTTP 200, title "Truck Fuel Agentic Cockpit"; backend banner green; replay catalog loaded (trip_000001, city, 360/1993). |
| `02_prediction_delivered.png`   | Successful saved-model prediction from a **separate manual-input run** (distinct from the replay-sample scenario): **Predicted fuel rate 62.61 L/h**, **delivered** status, validation **9/9 checks passed**, `explain_prediction` intent, executed tool `prediction_explanation`.                                                                                                                                                                                                       | Live click of "▶ Run AI prediction" produced a validated, delivered response.                                            |
| `03_shap_explanation.png`       | Expandable SHAP explanation: kind `instance_level_shap`, expected value **41.55 L/h** (base rate), reconstruction **passed (max dev 0.00007)**, top positive/negative contributions (torque +38.86, engine_rpm −8.79, …), "SHAP is attribution, not causality" note.                                                                                                                                                                                                                     | Same live session, SHAP accordion expanded.                                                                              |
| `04_replay_sample_transfer.png` | Synthetic telemetry replay playback + sample transfer: replay readout at **sample index 1** (speed 5.8 km/h, RPM 815, load 16.1%, throttle 28.8%, grade −0.06%, target fuel 6.39 L/h) and the transfer notice **"Loaded replay sample → manual inputs: RPM 815, speed 6 km/h, load 16%, throttle 29%, grade -0.06%"** visible with the manual engine controls updated. Note: the canonical demo scenario uses sample index 2 (RPM 1041.9); this screenshot is a different genuine pause. | Live clicking of `▶ Play`, pause, and "⬇ Load this replay sample into prediction inputs".                                |
| `05_assistant_interaction.png`  | Assistant real conversation: user asked **"How can I reduce fuel consumption?"**; assistant answered honestly with a **declined/ungrounded** response ("I could not produce a validated, grounded answer… the internal response failed safety checks and is not being presented to you.") tagged `knowledge_query · not delivered`. Demonstrates the evidence-grounded safety behavior (no invented advice).                                                                             | Live typed message via the assistant panel.                                                                              |

## Evaluation metrics (referenced, not screenshotted)

The model and agent evaluation metrics are machine-readable artifacts in
`reports/` (not rendered in the cockpit UI), so they are cited rather than
screenshotted:

- `reports/m1/metrics_report.json` — cross-truck fuel-rate baseline metrics
  (XGBoost full_operational: MAE 3.62, RMSE 5.88, R² 0.981; telematics_only:
  MAE 4.29, RMSE 6.62, R² 0.976). **Synthetic training set only.**
- `reports/m5/m5_5/evaluation.json` + `summary.md` — M5.5 agent ablation
  study (baseline `A_full_pipeline` overall **0.920** = 185/201 applicable,
  safety-critical scenarios 0). **Engineering ablation; not human-subject;
  not real-world validation.**

## Provenance

Screenshots were saved by the headless browser capture of the live dev server
and copied verbatim into this folder (byte-for-byte PNG, verified signature).
No image was edited or regenerated after capture.
