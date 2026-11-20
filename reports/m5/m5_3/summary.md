# M5.3 Prediction & Explanation Reliability Evaluation

**Milestone:** M5.3

**Protocol:** evaluate the real FuelRatePredictionTool and PredictionExplanationTool on canonical valid inputs plus a small deterministic held-out segment sample; saved M1.2 models and preprocessors only; no retraining; deterministic; synthetic-data constrained

**Scope (non-duplication):** M1.3 pooled/per-truck/per-condition metrics and M1.4 global/per-truck SHAP importance are referenced, not recomputed. M5.3 audits the real agent tool layer.

## Tool-level evaluation (canonical valid inputs)

| Feature set | Pred consistent | Pred finite/valid | Tool↔model agreement | Invalid rejection | Recon OK | Attribution dir OK | Safety OK | Expl↔pred consistent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_operational | True | True | True | 1.0 | True | True | True | True |
| telematics_only | True | True | True | 1.0 | True | True | True | True |

## Segment tool-agreement sample (held-out test, deterministic)

| Feature set | Segment | n | agreement rate |
| --- | --- | --- | --- |
| full_operational | highway | 6 | 1.000 |
| full_operational | acceleration | 6 | 1.000 |
| full_operational | idle_stopped | 6 | 1.000 |
| full_operational | uphill | 6 | 1.000 |
| telematics_only | highway | 6 | 1.000 |
| telematics_only | acceleration | 6 | 1.000 |
| telematics_only | idle_stopped | 6 | 1.000 |
| telematics_only | uphill | 6 | 1.000 |

## Safety / constraints

- Invalid inputs must never produce a successful prediction (rejection rate = 1.0 across all cases).
- SHAP is reported as attribution, never causality; no guaranteed-savings claim; synthetic-training disclaimer always attached.
- All numbers are estimates from synthetic telemetry, not validated real-world measurements.
