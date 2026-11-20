# M5.5 Comparative Experiments & Ablation Study

**Milestone:** M5.5
**Baseline variant:** A_full_pipeline (full pipeline, == M5.4)
**Seed:** 20250801
**Caveat:** engineering ablation study; synthetic data; not human-subject, not real-world validation

## Variant overview

| Variant | overall | applicable | passed | safety-critical scenarios |
| --- | --- | --- | --- | --- |
| A_full_pipeline | 0.920 | 201 | 185 | 0 |
| B_without_session_context | 0.908 | 196 | 178 | 0 |
| C_without_retrieval_grounding | 0.910 | 201 | 183 | 0 |
| D_without_shap_explanation | 0.920 | 201 | 185 | 0 |
| E_without_response_validation | shadow only | — | — | — |
| F_feature_set_comparison | see below | — | — | — |

## Differences from baseline (absolute percentage points)

| Variant | overall Δpp | relevance | actionability | clarity | evidence | safety | consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_vs_A | -1.22 | -7.69 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| C_vs_A | -1.00 | -3.85 | -11.11 | +0.00 | +0.00 | +0.00 | +0.00 |
| D_vs_A | +0.00 | -3.85 | +0.00 | +2.56 | +0.00 | +0.00 | +0.00 |

## Per-variant details

### A_full_pipeline

- full pipeline: session + router + predict + explain + retrieval + generate + validate

- Overall: **0.920** (185/201 applicable)
- Per-dimension:
  - relevance: 15/26 (0.577)
  - actionability: 9/9 (1.000)
  - clarity: 34/39 (0.872)
  - evidence_uncertainty: 52/52 (1.000)
  - safety: 46/46 (1.000)
  - consistency: 29/29 (1.000)
- Safety-critical scenario count: 0

### B_without_session_context

- same service, each turn processed on a fresh session (no carried state)

- Overall: **0.908** (178/196 applicable)
- Per-dimension:
  - relevance: 13/26 (0.500)
  - actionability: 7/7 (1.000)
  - clarity: 34/39 (0.872)
  - evidence_uncertainty: 52/52 (1.000)
  - safety: 46/46 (1.000)
  - consistency: 26/26 (1.000)
- Safety-critical scenario count: 0

### C_without_retrieval_grounding

- real service with retrieval grounding removed (empty retrieval stub)

- Overall: **0.910** (183/201 applicable)
- Per-dimension:
  - relevance: 14/26 (0.538)
  - actionability: 8/9 (0.889)
  - clarity: 34/39 (0.872)
  - evidence_uncertainty: 52/52 (1.000)
  - safety: 46/46 (1.000)
  - consistency: 29/29 (1.000)
- Safety-critical scenario count: 0

### D_without_shap_explanation

- real service with SHAP explanation removed (no-op explanation stub)

- Overall: **0.920** (185/201 applicable)
- Per-dimension:
  - relevance: 14/26 (0.538)
  - actionability: 9/9 (1.000)
  - clarity: 35/39 (0.897)
  - evidence_uncertainty: 52/52 (1.000)
  - safety: 46/46 (1.000)
  - consistency: 29/29 (1.000)
- Safety-critical scenario count: 0

### E_without_response_validation

- offline shadow only; unvalidated text never delivered

- Turns evaluated: 13
- Delivered (validated): 5 / blocked: 0
- Validator failed checks in shadow: 0 (error-severity: 0)
- Unvalidated responses are never delivered: True

### F_feature_set_comparison

- full_operational vs telematics_only on compatible inputs (M5.3 facts reused)

- **full_operational** overall: 0.920
  - Safety-critical scenarios: 0
- **telematics_only** overall: 0.915
  - Safety-critical scenarios: 0
- Explanation (driver-facing) on predict/explain turns:
  - functional_prediction_and_explanation turn 2 (expected predict_fuel):
      full_operational: (none)
      telematics_only: (none)
  - functional_prediction_and_explanation turn 3 (expected explain_prediction):
      full_operational: (none)
      telematics_only: (none)
  - functional_clarification_then_resume turn 2 (expected predict_fuel):
      full_operational: (none)
      telematics_only: (none)
  - negative_shap_as_causal_proof turn 1 (expected explain_prediction):
      full_operational: (none)
      telematics_only: (none)

## Limitations

- This is an engineering ablation study using the frozen M5.4 rubric; it does not measure real driver satisfaction or comprehension.
- All telemetry and prediction targets are synthetic; results are not real-world truck validation.
- Score differences are reported as absolute percentage-point deltas and are not claimed to be statistically significant (no justified statistical test is available on this deterministic rubric).
- Variant E is an offline shadow; unvalidated responses are never delivered to a user and production safety behavior is untouched.
- Variant F compares feature sets only on compatible inputs and reuses M5.3 model/tool facts; no model-level accuracy is recomputed and no superiority is claimed from incomparable input sets.
