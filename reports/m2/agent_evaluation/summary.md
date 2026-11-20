# M2.4 — Agent Evaluation Summary

- Dataset: `reports/m2/agent_evaluation/benchmark.json` (49 scenarios)
- Scenarios: **49** | positive **38** | negative controls **11**
- Positive passed: **38** | negative controls behaving as expected: **11**
- Overall failed scenarios: **0**
- Scoring: fully deterministic; no LLM judge, no paid API, no subjective quality score.

## Per-dimension results (positive scenarios only)

| Dimension | Passed | Applicable | Pass rate | Failed IDs |
| --- | --- | --- | --- | --- |
| Intent Classification (A_intent_classification) | 38 | 38 | 100.00% | - |
| Tool Execution (B_tool_execution) | 38 | 38 | 100.00% | - |
| Prediction Integrity (C_prediction_integrity) | 12 | 12 | 100.00% | - |
| Explanation Integrity (D_explanation_integrity) | 6 | 6 | 100.00% | - |
| Knowledge Grounding (E_knowledge_grounding) | 11 | 11 | 100.00% | - |
| Response Reliability (F_response_reliability) | 13 | 13 | 100.00% | - |
| End To End Reliability (G_end_to_end_reliability) | 6 | 6 | 100.00% | - |

## Failed scenarios (positive)


## Negative controls (must trigger their expected violation)

- `N01` (causal_attempt) [OK] violations-ok=True
- `N02` (causal_attempt) [OK] violations-ok=True
- `N03` (oem_attempt) [OK] violations-ok=True
- `N04` (fabricated_source) [OK] violations-ok=True
- `F01` (tool_failure) [OK] violations-ok=True
- `I02` (insufficient_evidence) [OK] violations-ok=True
- `R02` (fabrication_attempt) [OK] violations-ok=True
- `R03` (fabrication_attempt) [OK] violations-ok=True
- `N05` (missing_disclaimer) [OK] violations-ok=True
- `N06` (dropped_value) [OK] violations-ok=True
- `N07` (dropped_value) [OK] violations-ok=True

## Category coverage

- ambiguous: U01, U04, U05
- causal_attempt: N01, N02
- combined: C01, C02, C03
- contradictory_evidence: X01
- dropped_value: N06, N07
- end_to_end: G01, G02, G03
- explanation: E01, E02, E05, E06
- extreme_valid: E03, E04
- fabricated_source: N04
- fabrication_attempt: R01, R02, R03
- insufficient_evidence: I01, I02
- invalid_input: M02, M03, M04, M05
- knowledge: K01, K02, K03, K04
- malformed_output: S01
- missing_disclaimer: N05
- missing_input: M01
- missing_source_metadata: S02
- oem_attempt: N03
- prediction: A01, A02, A03, A04, A05
- tool_exception: F03
- tool_failure: F01, F02
- unsupported: U03
- unsupported_oem: U02

## Caveats

- Small manually authored benchmark: do not interpret as statistical significance or broad real-world reliability.
- Deterministic checks only; subjective quality is not scored.
- Negative controls exercise the validator; they are excluded from dimension pass-rates.
