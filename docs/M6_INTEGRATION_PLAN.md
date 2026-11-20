# M6.1 — Core System Integration Plan

Status: *Plan only — no integration code written.* M6.2 and beyond are not started.

Project: Cross-Truck Fuel-Rate Prediction and Explainable Agentic Driver-Feedback System.
All telemetry, prediction targets, and retrieval corpus are **synthetic**. Nothing in this plan
represents real-world truck validation.

This document is an architecture/integration contract. It does not modify any frozen
M1–M5 implementation file, does not retrain models, and adds no LLM / external API / database.

---

## 1. Actual architecture (read-only inspection)

### Component map and interfaces

| Component | Source | Public interface | Role |
|---|---|---|---|
| Fuel-rate tool | `src/tools/fuel_rate_prediction.py` | `FuelRatePredictionTool()`. `predict(inputs, feature_set=None) -> dict` | Loads saved M1.2 XGBoost + preprocessor; validates inputs via `validate_vehicle_inputs`; returns synthetic estimate in L/h |
| Explanation tool | `src/tools/prediction_explanation.py` | `PredictionExplanationTool()`. `explain(inputs, feature_set=None, top_features=None) -> dict` | Instance-level SHAP on the same validated inputs; verifies SHAP reconstruction; never claims causality |
| Retrieval tool | `src/tools/knowledge_retrieval.py` | `KnowledgeRetrievalTool()`. `retrieve(query, k=None, min_score=None) -> dict` | Builds deterministic vector store over synthetic corpus; returns ranked passages with evidence-sufficiency flag |
| Input validation | `src/tools/validation.py` | `validate_vehicle_inputs(inputs, feature_set)`; `ToolInputError` | Strict field allowlist, ranges, required-field checks |
| Session | `src/agent/driver_session.py` | `DriverSession` dataclass; `apply_partial_inputs`, `record_request`, `record_tool_result`, `finalize_for_prediction`, `to_dict`, `mark_unsupported/answered` | In-memory multi-turn state, partial-input validation, history |
| Intent router | `src/agent/driver_intents.py` | `DriverIntentRouter.route(message, structured_inputs=None, session=None) -> DriverIntentResult` | Deterministic classification into DriverIntent; clarification pass; tool selection |
| Conversation service | `src/agent/driver_conversation.py` | `DriverConversationService(router=, tools=, generator=, default_feature_set=)`. `handle(session, message, structured_inputs=None) -> DriverConversationResponse` | Orchestrates session → route → clarify/actionable → tools → generate → validate → deliver |
| Response generator | `src/agent/response_generator.py` | `ResponseGenerator().generate(agent_response) -> GroundedResponse` | Renders driver-facing + technical versions from tool outputs |
| Response validation | `src/agent/response_validation.py` | `validate_grounded_response(grounded, source_prediction=, source_explanation=, known_doc_ids=) -> list[ValidationIssue]`; `summary(issues)` | Deterministic guardrail battery (synthetic disclaimer, non-causal, no OEM mislabeling, source attribution, value preservation, failure disclosure) |
| M2 orchestrator (frozen baseline) | `src/agent/orchestrator.py` | `AgentOrchestrator().handle(query, structured_inputs=None) -> AgentResponse` | Single-turn agent; reused internally by the conversation service concepts |
| Demo entry point | `scripts/run_agent_demo.py` | `python scripts/run_agent_demo.py` | Demonstrator; prints and writes JSON examples |
| Feature spec | `configs/ml_synthetic_feature_spec.json` | loaded via `src/features/spec.py` | Single source of truth for feature sets / ranges / leakage rules |
| Agent config | `configs/agent_config.json` | — | Tool paths, routing backend, provenance flags |
| Corpus | `data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl` | loaded via `src/rag/corpus.py` | Synthetic educational material (non-OEM) |

### Current call flow (production path)
```
Driver input
 → DriverSession
 → DriverIntentRouter.route (uses session context)
 → session.record_request + apply_partial_inputs
 → CLARIFYING: no tools, return clarification request
 → UNSUPPORTED / AMBIGUOUS: no tools, safe decline / clarify
 → PREDICT/EXPLAIN/COMBINED: session.finalize_for_prediction → execute selected tools
 → KNOWLEDGE_QUERY: execute retrieval only
 → ResponseGenerator.generate → validate_grounded_response → delivered = no error-severity failure
```

### Dependency / data notes
- Tools are constructed with real saved artifacts (`reports/m1/artifacts/`), real synthetic corpus,
  and MiniLM embeddings. No mocks in the default path.
- `DriverConversationService.__init__` accepts an injected `tools` dict and `router`/`generator`,
  which is the natural seam for integration wiring (and was already used by M5.5 ablations).
- Deterministic. No LLM, no external API, no database, no network. Sessions are in-memory only.
- There is **no HTTP server, CLI application shell, or UI contract** today — the only entry points are
  research scripts.

---

## 2. Proposed integration contract

### Application entry point
Recommended: a single programmatic entry facade, `src/api/gateway.py`, exposing one method:

```
class FuelRateFeedbackGateway:
    def __init__(self, service: DriverConversationService | None = None, ...)
    def start_session(self, feature_set: str = "full_operational") -> SessionHandle
    def send(self, handle: SessionHandle, query: str, structured_inputs: dict | None = None) -> APIResponse
    def session_state(self, handle: SessionHandle) -> dict
```

This is intentionally thin: the gateway only adapts `DriverConversationService` into a stable,
UI-friendly contract. It does not change any frozen component.

### Input schema
- `query: str` — required, non-empty.
- `structured_inputs: dict | None = None` — optional; validated per feature set.
- Optional field for the **full_operational** set (canonical): `vehicle_class`, `gross_vehicle_weight_kg`,
  `payload_kg`, `vehicle_mass_kg`, `speed_kph`, `acceleration_mps2`, `road_grade_pct`, `ignition_status`,
  `engine_rpm`, `engine_load_pct`, `throttle_position_pct`, `torque_nm`, `coolant_temp_c`,
  `intake_air_temp_c`, `ambient_temp_c`.
- **telematics_only** drops `engine_rpm`, `engine_load_pct`, `throttle_position_pct`, `torque_nm`.
- `vehicle_class` is categorical (`"32T"`, `"40T"`, `"49T"`); `ignition_status` is `0`/`1`;
  all numeric fields are floats within documented ranges. Unknown fields are rejected.
- Qualitative free-text observations (`driver_observation`, `driving_condition`, `route_context`, `comment`)
  are accepted but never converted into telemetry.

### Session creation and lifecycle
- `start_session` creates a `DriverSession` (in-memory, `session_id` generated).
- Lifecycle state: `awaiting_input → clarifying → ready_predict / ready_explain → answered`,
  plus `unsupported`. Maintained by `DriverSession`.
- No persistence; sessions live only in the process. The plan notes this as a production limitation.

### Partial inputs and clarification
- If a model-input intent arrives with missing/invalid required fields, the router returns `CLARIFYING`
  with `missing_fields` / `invalid_fields`. The service returns `driver_feedback` asking for exactly those
  fields, `delivered=False`, and runs no prediction tools.
- On subsequent turns the driver may supply only the missing fields; the session merges them and proceeds.

### Prediction / explanation invocation
- Only after `session.finalize_for_prediction()` succeeds (strict `validate_vehicle_inputs`).
- Prediction returns `predicted_fuel_rate_lph` (L/h) with synthetic disclaimer.
- Explanation returns instance-level SHAP (top positive/negative), with reconstruction check and
  non-causal disclaimer. Invoked only when the selected intent includes the explanation tool.

### Knowledge handling
- Only `KNOWLEDGE_QUERY` (or combined-with-knowledge) runs retrieval. No prediction inputs required.
- `evidence_sufficiency.insufficient` governs whether evidence-backed recommendations are emitted;
  insufficient evidence is surfaced explicitly, never fabricated into guidance.

### History
- `DriverSession.history` holds user turns (query/intent/timestamp) and tool-result turns. Multi-turn
  clarification and resumption are supported by routing + session context.

### Validation failures surfaced
- `response.validation_issues` (full list), `validation_summary`, `validation_blocked`, and `delivered`.
- If any error-severity check fails, `delivered=False` and the driver-facing feedback is the standard
  "could not produce a validated, grounded answer" message — never the raw failed text.

### Tool errors / missing artifacts
- Tools raise `ToolExecutionError` (missing model dir / malformed transform) or `ToolInputError`.
  The service catches exceptions per tool, records `status="error"` in `tool_results`, and the validation
  battery's `failure_not_presented_as_success` check gates the response accordingly.

### Returned to UI (proposed `APIResponse`)
```
{
  "session_id": str,
  "driver_intent": str,
  "conversation_state": str,
  "selected_tools": [str],
  "executed_tools": [str],
  "delivered": bool,
  "validation_blocked": bool,
  "driver_feedback": str,
  "clarification_request": str | null,
  "missing_fields": [str],
  "invalid_fields": {str: str},
  "technical_report": {   # fleet/operator view
      "prediction": {...} | null,
      "explanation": {...} | null,
      "evidence": [...] , "evidence_insufficient": bool,
      "recommendations": {...},
      "validation": {...},
      "synthetic_only": true,
      "invented_telemetry": false,
      "shap_is_attribution_not_causality": true
  }
}
```
A `SessionHandle` is an opaque key (e.g. `session_id`) mapping to the live in-memory `DriverSession`.

---

## 3. Required new files (integration later, not now)

- `src/api/gateway.py` — adapter facade over `DriverConversationService` (the one new component).
- `src/api/session_registry.py` (optional) — in-memory `session_id → DriverSession` map with TTL/gc;
  small and self-contained. Could be folded into `gateway.py` if preferred.
- `src/api/schemas.py` (optional) — `APIResponse` serialization helpers (validation/coercion).
- `scripts/run_integration_cli.py` — optional thin CLI wrapping the gateway for manual/demo use.
- `tests/test_integration_gateway.py` — contract tests around the gateway (later milestone).
- Documentation: this plan file.

No frozen M1–M5 source is added to or modified.

---

## 4. Existing files to reuse unchanged

- `src/agent/driver_conversation.py` — `DriverConversationService` (the integration core).
- `src/agent/driver_session.py` — `DriverSession`, `validate_partial_vehicle_inputs`.
- `src/agent/driver_intents.py` — `DriverIntentRouter`.
- `src/tools/*` — the three tools.
- `src/agent/response_generator.py`, `src/agent/response_validation.py`.
- `configs/agent_config.json`, `configs/ml_synthetic_feature_spec.json`, `src/features/spec.py`.
- `src/evaluation/m5_4_driver_feedback.py` — frozen rubric (reused only for evaluation, not integration).
- `scripts/run_agent_demo.py` — template for a later demo.

---

## 5. Gap analysis

### Reusable unchanged
All core logic (session, routing, tools, generation, validation) is directly reusable. The conversation
service already exposes dependency injection for `tools`, `router`, `generator`, and `default_feature_set`.

### Missing integration components
1. **No user-facing entry point.** No CLI/HTTP/UI shell; only research scripts.
2. **No session registry.** Sessions are per-object; a UI needs a way to address an ongoing session.
3. **No uniform response serialization** for a UI (only `driver_feedback` + `technical_report` dict; a
   stable `APIResponse` schema is not yet formalized).
4. **No error/exception policy at the boundary** — tool `ToolExecutionError`/`ToolInputError` are caught
   inside the service, but a gateway should normalize into a defined error payload.
5. **No config-driven construction** — tool paths are hardcoded defaults; `agent_config.json` exists but is
   not consumed by `DriverConversationService`. Integration should wire it so artifacts/corpus are explicit.

### Duplicate functionality
- `AgentOrchestrator` (M2.2) and `DriverConversationService` (M4.4) overlap on routing + tool execution.
  The conversation service is the intended integration core; the M2 orchestrator is the frozen research
  baseline and should **not** be the gateway backend.

### Inconsistent interfaces
- Tools differ in key names (`predicted_fuel_rate_lph`), and the service's `_TOOL_EFFECTORS` map is fine,
  but there is no single `ToolResult` schema. Fine internally; a gateway should not need to change it.
- `DriverConversationResponse` has many fields (some optional); a gateway should pick a stable exposed subset.

### Error-handling gaps
- Tool failures are captured as `status="error"` but are not exposed as a distinct top-level error channel
  in the response; the gateway should surface them explicitly.
- Missing model artifacts raise `ToolExecutionError` at first tool use; a gateway should preflight-check
  artifacts at startup and return a clear readiness error.

### Configuration / deployment
- Artifacts: `reports/m1/artifacts/` (XGBoost + preprocessor per feature set), corpus JSONL, embedding model.
- No external services, no network, no secrets. Running is local-only.
- Determinism: fixed seeds + in-memory state; no concurrency model defined yet (registry must be thread-safe
  if the UI is concurrent).

---

## 6. Identified risks

- **Synthetic-data risk (must stay explicit):** predictions, corpus, and any "savings" framing are synthetic;
  the gateway must carry the synthetic disclaimer and never present results as real-world validation.
- **Determinism vs. production:** in-memory sessions are lost on restart; acceptable for this milestone,
  must be documented.
- **Concurrency:** shared `DriverSession` is not thread-safe; a multi-request UI needs per-session locking or
  per-session service instances.
- **Seam risk:** injecting custom tools/router is supported, but behavior should be validated against the
  frozen baseline (M5.4 / M5.5) to ensure the gateway does not alter routing or safety semantics.
- **Scope creep:** the gateway must not assert new claims (e.g. real-world fuel savings). Keep UI text
  identical to validated `driver_feedback`.

---

## 7. Recommended implementation sequence (for later approval)

1. Add `src/api/gateway.py` + `session_registry.py` (thin adapters; no change to M1–M5).
2. Add `src/api/schemas.py` for `APIResponse` serialization.
3. Wire `configs/agent_config.json` into `DriverConversationService` construction (read-only, defaults same).
4. Add `scripts/run_integration_cli.py` demo.
5. Add contract tests `tests/test_integration_gateway.py`; verify gateway results match frozen M5.4/M5.5
   baseline on the same scenarios (no semantic drift).
6. Produce an integration report (later milestone), explicitly synthetic and non-validating.

Nothing above is implemented in this milestone.
