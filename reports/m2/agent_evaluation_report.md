# M2.4 — Agent Evaluation and Reliability Report

**Milestone:** M2.4 (deterministic agent evaluation — no LLM judge, no paid API)
**Dataset:** `reports/m2/agent_evaluation/benchmark.json`
**Runner:** `scripts/run_m2_agent_evaluation.py`
**Date:** 2026-11-19

## 1. Overview

M2.4 is a reproducible, fully deterministic evaluation framework for the
integrated Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver-
Feedback System (M2.2 agent + M2.3 response generator). All scoring is rule-based
on the actual tool outputs and generated response text. There is no LLM judge,
no paid API, no new LLM provider, and no subjective quality score.

The benchmark is a separate, manually authored dataset (49 scenarios) whose
expected outcomes are independent of the agent's generated answers.

## 2. Files

| Path | Kind |
| --- | --- |
| `reports/m2/agent_evaluation/benchmark.json` | benchmark (49 scenarios) |
| `reports/m2/agent_evaluation/evaluation.json` | dimension metrics + per-scenario checks |
| `reports/m2/agent_evaluation/scenario_results.json` | full per-scenario results |
| `reports/m2/agent_evaluation/summary.md` | summary table |
| `reports/m2/agent_evaluation_report.md` | this report |
| `scripts/run_m2_agent_evaluation.py` | evaluation runner |
| `tests/test_m2_agent_evaluation.py` | focused tests (11) |

## 3. Methodology

### 3.1 Scenario authoring
49 manually authored scenarios with unique IDs (A01–G03), each with: a user
query, required input, expected intent, expected tool behavior, expected
response properties, prohibited claims, and expected abstention/failure
behavior. Categories cover every requested area:

prediction, knowledge, explanation, combined, ambiguous, unsupported,
missing/invalid input, insufficient evidence, unsupported OEM specs, tool
failure/exception, malformed output, missing source metadata, contradictory
evidence, extreme-but-valid inputs, causal-attempt, and fabrication-attempt.

### 3.2 Build type (real vs direct vs mock)
Each scenario is executed one of three ways, and this is tracked per scenario in
`scenario_results.json` under `build`:

- `real` (30): executed end-to-end through the **actual M2.2 orchestrator**
  (which calls the real M2.2 tools and the M2.3 response generator). These are
  genuine local integrations over the saved M1.2 models and the M2.1 corpus.
- `direct` (1): `A02` (telematics-only prediction) constructs the tool call
  directly because the frozen orchestrator routes `full_operational` by
  default; it still exercises the real prediction tool and response generator.
- `mock` (18): responses are constructed directly (with injected content) to
  test validation/failure handling. **All 11 negative controls are mock**, by
  design — you cannot inject a violation into the frozen agent without
  modifying it. Mock scenarios test the validator's ability to *detect*
  prohibited content, not the real execution path.

Mock scenarios are **never** described as real end-to-end executions. Only the
30 `real` runs are real orchestrator integrations.

### 3.3 Checks and dimensions
The runner runs the M2.3 validator checks on full response text (not truncated
previews) plus scenario-specific checks, and aggregates into seven dimensions:
A Intent Classification, B Tool Execution, C Prediction Integrity,
D Explanation Integrity, E Knowledge Grounding, F Response Reliability,
G End-to-End Reliability. Only **applicable** scenarios count toward a
dimension (e.g. prediction-only scenarios are not scored on knowledge grounding).
Negative controls are excluded from pass-rates and reported separately.

## 4. Results

- Total scenarios: **49** (38 positive, 11 negative controls).
- Positive scenarios passed: **38/38**. Failed positive scenarios: none.
- Negative controls behaving as expected: **11/11**.

### 4.1 Dimension metrics (positive scenarios only)

| Dimension | Passed | Applicable | Pass rate | Failed IDs |
| --- | --- | --- | --- | --- |
| A Intent Classification | 38 | 38 | 100.00% | — |
| B Tool Execution | 38 | 38 | 100.00% | — |
| C Prediction Integrity | 12 | 12 | 100.00% | — |
| D Explanation Integrity | 6 | 6 | 100.00% | — |
| E Knowledge Grounding | 11 | 11 | 100.00% | — |
| F Response Reliability | 13 | 13 | 100.00% | — |
| G End-to-End Reliability | 6 | 6 | 100.00% | — |

These figures were recomputed directly from `scenario_results.json` and match the
summary and evaluation JSON exactly (applies/passed per dimension: A 38/38,
B 38/38, C 12/12, D 6/6, E 11/11, F 13/13, G 6/6).

### 4.2 Negative controls (all detect their intended violation)

| ID | Category | Intended violation detected |
| --- | --- | --- |
| N01 | causal_attempt | `causal_language` fails |
| N02 | causal_attempt | `causal_language` fails |
| N03 | oem_attempt | `oem_mislabeling` fails |
| N04 | fabricated_source | `source_attribution` fails |
| F01 | tool_failure | `failure_not_presented_as_success` fails |
| I02 | insufficient_evidence | `insufficient_evidence_stated` fails |
| R02 | fabrication_attempt | `prediction_value_preserved` fails |
| R03 | fabrication_attempt | `oem_mislabeling` fails |
| N05 | missing_disclaimer | `synthetic_disclaimer` fails |
| N06 | dropped_value | `prediction_value_preserved` fails |
| N07 | dropped_value | `shap_value_preserved` fails |

Each negative control contains the injected violation and the corresponding
check genuinely flags it (verified per-scenario in `scenario_results.json`).

## 5. Verification performed

- Scenario count: benchmark `n_scenarios=49` == 49 result records; all IDs unique.
- Build distribution consistent between benchmark and results
  (real 30 / direct 1 / mock 18).
- Dimension metrics recomputed from scenario-level results == summary/evaluation.
- Negative-control detection confirmed per scenario.
- Full response text used for checks (the evaluator evaluates `text_full`).
- No expected outcomes were weakened to inflate scores; failing/edge scenarios
  (e.g. M01–M05 invalid input, U01–U05 ambiguous/unsupported) assert the
  required abstention/failure behavior.

## 6. Tests

Run: `python -m pytest tests/test_m2_agent_evaluation.py -q` → **11 passed**.
Coverage: scenario dimension applicability, prohibited-claim detection,
expected-tool verification, tool-order independence, negative-control detection,
and dimension metric aggregation.

Integration compatibility (real M2.2 tools + M2.3 generator):
`python -m pytest tests/test_tools.py tests/test_agent.py tests/test_m2_3_response.py -q`
→ **46 passed** (M2.2 tools/agent + M2.3 response suite).

## 7. Safety and scientific boundaries

- All telemetry, fuel-rate labels, and corpus content are synthetic; predictions
  are estimates from synthetic-trained models, not validated real-world
  measurements.
- RAG content is synthetic educational material, not OEM documentation.
- The agent never invents manufacturer specs, maintenance limits, vehicle
  history, driver identity, or route data; missing/invalid inputs are rejected.
- SHAP attribution is model attribution, not causality.
- Retrieval similarity is not proof of factual correctness.
- Deterministic routing is not LLM reasoning or autonomous agent intelligence.
- The benchmark is small and manually authored: results are **not** evidence of
  statistical significance or broad real-world reliability.

## 8. Limitations

- `real` runs exercise the actual orchestrator/tools, but the benchmark is small
  (49 scenarios) and manually authored; no claim of statistical significance.
- Mock scenarios validate the *validator*, not the real tool outputs.
- Deterministic keywords/internt rules can misroute unusual phrasing.
- The evidence-sufficiency and confidence heuristics are fixed thresholds, not
  calibrated.
- No LLM judge or automated readability metric is used; response quality is
  assessed via deterministic checks and manual review criteria.

## 9. How to run

```bash
python scripts/run_m2_agent_evaluation.py            # regenerate evaluation artifacts
python -m pytest tests/test_m2_agent_evaluation.py -q  # focused tests
```
