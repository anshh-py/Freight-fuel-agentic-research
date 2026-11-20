# M2.2 — Agent Architecture and Tool Integration

**Milestone:** M2.2 (engineering milestone — not a dashboard/deployment)
**Agent:** `truck_fuel_rate_agent_v1` (config: `configs/agent_config.json`)
**Date:** 2026-11-19

This document describes the first integrated agent for the heavy-duty truck
fuel-rate research project. It connects the M1.2 prediction models, the M1.4
explainability/SHAP implementation, and the M2.1 RAG retrieval system through a
modular orchestrator. All numeric content is synthetic; provenance is labelled
throughout.

---

## 1. Overview and architecture

The agent is a stateless orchestrator over three independent tools. It routes a
user query to an intent (Phase 4 boundary), selects the tools for that intent,
executes them through explicit callable interfaces, and returns a structured
response.

```
                       user query + optional structured vehicle inputs
                                          |
                                          v
                        ┌───────────────────────────────────────┐
                        │  AgentOrchestrator.handle(...)         │
                        │  ┌─────────────────────────────────┐  │
                        │  │ IntentRouter (Phase 4 boundary)  │  │
                        │  │  DeterministicIntentRouter       │  │
                        │  └───────────────┬─────────────────┘  │
                        │                  v                    │
                        │  intent -> tool selection (registry)  │
                        │       ┌───────┬──────┬────────┐       │
                        │       v       v      v        v       │
                        │  [Prediction] [Retrieval] [Explanation]│
                        │       └───────┴──────┴────────┘       │
                        │  aggregate -> AgentResponse (schema)  │
                        └───────────────────────────────────────┘
```

**Key property:** the orchestrator is independent of any LLM provider. The
current router is deterministic (see §5). Tools are plain callables with typed
I/O, so a future LLM-based router can reuse them unchanged.

---

## 2. Files (what changed)

| Path | Role |
| --- | --- |
| `src/tools/validation.py` | Shared input validation (required fields, ranges, feature-set contract). `ToolInputError`. |
| `src/tools/fuel_rate_prediction.py` | Tool 1 — predicts fuel rate (L/h) from validated inputs. |
| `src/tools/knowledge_retrieval.py` | Tool 2 — wraps the M2.1 vector store + corpus. |
| `src/tools/prediction_explanation.py` | Tool 3 — instance-level SHAP explanation of a single prediction. |
| `src/tools/__init__.py` | Tool registry namespace. |
| `src/agent/intents.py` | `Intent` enum, `IntentRouter` protocol, `DeterministicIntentRouter`. |
| `src/agent/orchestrator.py` | `AgentOrchestrator`, `AgentResponse` schema. |
| `src/agent/__init__.py` | Agent namespace. |
| `configs/agent_config.json` | Agent/tool configuration and provenance flags. |
| `scripts/run_agent_demo.py` | Local run/demo CLI (writes `reports/m2/agent_demo_examples.json`). |
| `tests/conftest.py`, `tests/test_tools.py`, `tests/test_agent.py` | Focused tests. |

No existing modules were modified. The M1 pipeline, M2.1 corpus/retrieval, and
original 55-case benchmark are untouched.

---

## 3. Tool interfaces

### Tool 1 — Fuel-rate prediction (`FuelRatePredictionTool`)

- **Input:** `dict` of operational vehicle inputs + optional `feature_set`
  (`full_operational` | `telematics_only`).
- **Validation:** rejects missing/unknown fields, non-numeric tokens,
  out-of-range values, invalid `vehicle_class`/`ignition_status`, unknown
  feature set. Bounds are grounded in the observed synthetic telemetry (with a
  documented margin). Never invents values.
- **Execution:** loads the saved M1.2 `xgboost.joblib` + `preprocessor.joblib`
  (never refits/retrains), builds the 1-row feature matrix in spec order,
  transforms, predicts.
- **Output:** `predicted_fuel_rate_lph` (L/h), `feature_set`, model artifact
  path, transformed columns, input fields used, provenance note (synthetic
  estimate).

### Tool 2 — Knowledge retrieval (`KnowledgeRetrievalTool`)

- **Input:** natural-language `query`, optional `k`, optional `min_score`.
- **Execution:** uses the existing M2.1 `VectorStore` + `LocalEmbedder`
  (all-MiniLM-L6-v2) over the synthetic JSONL corpus. Corpus/embedder/store
  are unchanged.
- **Output:** ranked passages with `document_id`, `chunk_id`, `title`, `topic`,
  `source_type`, `source_reference`, `is_oem_manual`, similarity `score`, and
  a content excerpt. Includes an `evidence_sufficiency` flag (top similarity
  below a documented floor, or zero results) and provenance that the corpus is
  synthetic educational material and that similarity ≠ factual correctness.

### Tool 3 — Prediction explanation (`PredictionExplanationTool`)

- **Input:** the same validated vehicle inputs (optionally a feature set).
- **Execution:** reuses `src.evaluation.explainability`:
  `load_selected_xgb`, `compute_shap_values`, `top_shap_contributions`,
  `validate_reconstruction`. Computes **instance-level** SHAP for the single
  input row and verifies the additive decomposition reconstructs the model
  output.
- **Output:** `explanation_kind: instance_level_shap`, predicted value,
  `expected_value`, reconstruction result, and the top positive/negative SHAP
  contributions (readable feature + signed value). Explicitly labels the
  explanation as *attribution for the model, not causality*, and distinguishes
  it from global feature importance (no substitution).

---

## 4. Intent routing and actual routing behavior

Intents and tool selection:

| Intent | Tools executed |
| --- | --- |
| `fuel_prediction` | `fuel_rate_prediction` |
| `knowledge_query` | `knowledge_retrieval` |
| `prediction_explanation` | `prediction_explanation` |
| `combined_prediction_explanation` | `fuel_rate_prediction`, `prediction_explanation` |
| `combined_knowledge_prediction` | `knowledge_retrieval`, `fuel_rate_prediction` |
| `unsupported` / `ambiguous` | none (response explains why) |

**Actual behavior** (from `scripts/run_agent_demo.py` →
`reports/m2/agent_demo_examples.json`):

- `"Predict fuel consumption for this operating condition."` + inputs →
  `fuel_prediction` → `Predicted fuel rate: 52.58 L/h (full_operational,
  xgboost)`.
- `"Why does road grade affect fuel consumption on heavy trucks?"` →
  `knowledge_query` → top passage `[doc_001] What drives heavy-duty fuel
  consumption` (similarity 0.697, synthetic source).
- `"Predict fuel consumption and explain how engine load affects it."` +
  inputs → `combined_knowledge_prediction` → retrieval + prediction.
- `"Explain this prediction."` + inputs → `prediction_explanation` → instance
  SHAP: top positive driver `Torque (N·m) +20.1 L/h`, top negative driver
  `Engine speed (rpm) -6.6 L/h`; reconstruction max deviation 1.6e-5 (pass).
- `"What is the official manufacturer torque limit …?"` → `unsupported`
  (out-of-scope safety refusal).
- `"banana split"` → `ambiguous`.

---

## 5. LLM integration boundary

- **Status:** no LLM provider is configured. The repository contains no
  existing LLM integration (searched `src` for openai/anthropic/langchain/llm
  keys — none). No paid API, no hardcoded key, no mandatory external service
  was introduced.
- **Routing backend:** `DeterministicIntentRouter` (keyword/rule scoring) behind
  the `IntentRouter` protocol. It is deliberately **not** claimed to be LLM
  reasoning or autonomous intelligence; every response includes the warning:
  *"Routing is a deterministic rule-based baseline, not LLM reasoning."*
- **Future path:** an LLM-based router only needs to implement
  `IntentRouter.route(...)`. The tools and orchestrator I/O are unchanged, so
  LLM tool-calling can be added without rewriting the tools.

---

## 6. Safety and scientific boundaries

Enforced in tool outputs and assembled warnings:

- All telemetry and fuel-rate labels are **synthetic**.
- Predictions are **estimates from synthetic-trained models**, not validated
  real-world fuel measurements.
- RAG documents are **synthetic educational material**, not OEM manuals
  (`is_oem_manual=false` preserved; `source_reference` states non-OEM).
- The agent **never invents** manufacturer specifications, maintenance limits,
  or missing vehicle data (missing inputs are rejected, not defaulted).
- Retrieval similarity is **not proof of factual correctness**.
- SHAP attribution is **not causality**.
- Prediction output is clearly separated from retrieved evidence (distinct
  response fields + `supporting_evidence`).

---

## 7. Tests

Run: `python -m pytest tests/test_tools.py tests/test_agent.py` → **25 passed**.

| Area | Coverage |
| --- | --- |
| Prediction tool | output schema, both feature sets (real saved models), missing/unknown/out-of-range/invalid-class/unknown-set rejection |
| Retrieval tool | provenance fields, synthetic flag, empty-query rejection, insufficient-evidence flag (real corpus + store) |
| Explanation tool | instance-level schema, reconstruction pass, positive/negative contributions, anti-causal flags, rejection on missing field |
| Router | prediction, knowledge, explanation, combined knowledge+prediction, combined predict+explain, unsupported, ambiguous |
| Orchestrator E2E | prediction query, knowledge query, combined query (real models + corpus) |
| Orchestrator resilience | invalid-input failure recorded, unsupported selects no tools, insufficient-evidence warning |

Mocks are not required for the model/corpus paths because they run against the
real saved artifacts locally; the session-scoped conftest shares the embedding
store to keep tests fast.

---

## 8. Limitations

- Routing is deterministic keyword matching; it can misroute edge phrasing and
  gives no natural-language understanding (explicitly not LLM).
- Confidence values are heuristic, not calibrated.
- The "insufficient evidence" floor is a fixed similarity heuristic; it is
  configurable but not tuned per query.
- Explanations are only as good as the synthetic-trained models; instance SHAP
  reflects the model, not the real world.
- Combined intents run tools sequentially; there is no cross-tool reasoning
  step.

---

## 9. How to run locally

```bash
# run the demo (writes reports/m2/agent_demo_examples.json)
python scripts/run_agent_demo.py

# run the focused test suite
python -m pytest tests/test_tools.py tests/test_agent.py
```

Programmatic use:

```python
from src.agent.orchestrator import AgentOrchestrator

orch = AgentOrchestrator()
resp = orch.handle("Predict fuel consumption", {"speed_kph": 75.0, ...})
print(resp.to_dict())
```
