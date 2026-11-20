# M2.3 — Grounded Driver Feedback and Response Generation

**Milestone:** M2.3 (response-generation layer on top of the finished M2.2 agent)
**Status:** complete — 21/21 evaluation scenarios pass, 21/21 tests pass
**Report date:** updated after the M2.3 continuation (see §7 "Fixes applied")

This report documents the response-generation layer, the grounding/claim separation,
the response-validation guardrails, the response-evaluation benchmark, test results,
remaining limitations, and how to run everything locally.

---

## 1. Deliverables

| Path | Role |
| --- | --- |
| `src/agent/response_generator.py` | `ResponseGenerator` (deterministic, no LLM) + `GroundedResponse` schema |
| `src/agent/response_validation.py` | Deterministic validation layer (9 checks) |
| `reports/m2/response_evaluation/benchmark.json` | 21 distinct evaluation scenarios |
| `reports/m2/response_evaluation/evaluation.json` | Per-scenario check-level results |
| `reports/m2/response_evaluation/summary.md` | Human-readable evaluation summary |
| `scripts/run_m2_3_response_eval.py` | Reproducible evaluation runner |
| `tests/test_m2_3_response.py` | Focused tests (21) |
| `reports/m2/response_generation_report.md` | This report |

No M1, M2.1, or M2.2 files were modified; no model retraining, paid API, LLM
provider, or LLM judge was used.

---

## 2. Response types supported

The generator consumes the actual `AgentResponse` + tool outputs from M2.2 and
produces a `GroundedResponse` with `driver_version` (readable) and
`technical_version` (fleet-operator) renderings. Supported response types:

1. Fuel-rate prediction (prediction block + synthetic-model disclaimer).
2. Knowledge question (retrieved-passage block + synthetic-corpus notice).
3. Prediction explanation (positive/negative SHAP contributions + caveat).
4. Combined prediction + explanation.
5. Combined knowledge retrieval + prediction.
6. Unsupported / ambiguous query (explicit abstention).
7. Tool failure / insufficient-evidence response (explicit disclosure).

---

## 3. Grounding and claim separation

Every response keeps four claim categories strictly separated:

- **A. Model prediction** — predicted L/h value, feature set, model identity,
  and the synthetic-trained-model disclaimer (`SYNTHETIC_DISCLAIMER`).
- **B. Model explanation** — positive/negative SHAP contributions with the
  actual input value where available, plus the explicit non-causal caveat
  (`CAUSAL_DISCLAIMER`). SHAP is always described as attribution, never as
  the cause of real-world fuel consumption.
- **C. Retrieved evidence** — document id/title, topic, similarity score, and
  the synthetic-educational provenance notice (`CORPUS_DISCLAIMER`). Never
  presented as an OEM specification or independently verified fact.
- **D. Recommendations** — split into evidence-backed operational
  considerations, general suggestions, and unsupported information. Evidence
  backed recommendations are only produced when retrieval was judged
  *sufficient*; otherwise the response states no evidence-backed recommendation
  can be given.

The generator never invents a driver name, truck identity, route, observed fuel
consumption, manufacturer specification, maintenance limit, or historical
baseline.

---

## 4. Driver versus technical formats

- **Driver-facing (`driver_version`):** low-jargon structure:
  1. **Estimated fuel rate** (L/h),
  2. **What influenced the model** (top positive/negative contributions),
  3. **Relevant knowledge** (retrieved passages, if any),
  4. **Operational considerations** (evidence-backed only),
  5. **Important limitations** (synthetic data + model limits).
- **Technical / fleet operator (`technical_version`):** structured dict with
  feature names, signed SHAP values, source ids, retrieval metadata, claim/
  provenance flags, and detailed caveats.

---

## 5. Validation layer (`src/agent/response_validation.py`)

Deterministic, rule-based checks detect:

- predictions without synthetic-data disclaimers (`synthetic_disclaimer`),
- SHAP contributions described as causal effects (`causal_language`),
- RAG passages presented as OEM documentation (`oem_mislabeling`),
- missing synthetic-corpus provenance on cited passages
  (`synthetic_corpus_notice`),
- fabricated / missing source references (`source_attribution`),
- unsupported queries answered with concrete specs (`unsupported_not_answered`),
- insufficient evidence not stated explicitly (`insufficient_evidence_stated`),
- tool failures presented as success (`failure_not_presented_as_success`),
- dropped predicted L/h value (`prediction_value_preserved`),
- dropped top SHAP contribution (`shap_value_preserved`).

---

## 6. Evaluation results

Dataset: `reports/m2/response_evaluation/benchmark.json` — **21 distinct
scenarios** (unique IDs), deterministic scoring, no LLM judge, no paid API.
Categories: prediction-only (P01, P02), knowledge-only (K01), prediction +
explanation (E01, E02), combined knowledge + prediction (C01), ambiguous
(U01), unsupported OEM spec (U02), tool failure (F01, M01), insufficient
evidence (I01), and contradictory/prohibited negative cases (N01–N09, C02).

| Result | Count |
| --- | --- |
| Scenarios | **21** |
| Passed | **21** |
| Failed | **0** |

**Negative cases genuinely reach the intended validation logic** (each injected
violation is detected — the targeted check fails on the hypothetical bad
response):

| Scenario | Injected violation | Check that fires |
| --- | --- | --- |
| N01 | causal claim (`"The torque caused the higher fuel rate…"`) | `causal_language` |
| N02 | corpus presented as OEM documentation | `oem_mislabeling` |
| N03 | prediction without synthetic disclaimer | `synthetic_disclaimer` |
| N04 | fabricated citation `[doc_999]` | `source_attribution` |
| N05 | unsupported query answered with a spec number | `unsupported_not_answered` |
| N06 | insufficient evidence but answer given | `insufficient_evidence_stated` |
| N07 | failed tool presented as success | `failure_not_presented_as_success` |
| N08 | predicted value omitted | `prediction_value_preserved` |
| N09 | top SHAP contribution omitted | `shap_value_preserved` |

**No fabricated metrics are reported.** Subjective readability is recorded only
as documented manual-review notes (not as a numeric score). The runner does not
claim statistical significance or broad real-world reliability from this small,
manually authored set.

---

## 7. Fixes applied in the M2.3 continuation

Two blockers (reported at the gate) were fixed with minimal, focused changes:

1. **Test assertion — `tests/test_m2_3_response.py`:** the synthetic-provenance
   assertion mixed case (`"SYNTHETIC educational material" in
   out.driver_version.upper()` could never match an all-uppercase haystack). It
   now normalizes both sides consistently (`"SYNTHETIC EDUCATIONAL MATERIAL" in
   out.driver_version.upper()`).
2. **Causal-language validator — `src/agent/response_validation.py`:** the old
   `_is_negated_prefix` only recognized an *immediately adjacent* negation, so
   protective phrasing (`"NOT a causal explanation"`, `"does not prove"`) was
   misread as a causal claim, and assertion frames were treated unconditionally.
   It now uses token-based context detection (`_is_negated_context`) applied to
   **both** frames and word-level causal terms, recognizing a negation even when
   an article/adjective separates it and handling double negatives. This keeps
   genuine claims (`"this feature causes higher fuel consumption"`,
   `"SHAP proves this feature increases consumption"`) flagged while never
   flagging anti-causal disclaimers.
3. **Regression tests added** (`tests/test_m2_3_response.py`):
   `test_negated_frame_does_not_prove_not_flagged` and
   `test_nonnegated_frame_proves_flagged`.

---

## 8. Tests

Run: `python -m pytest tests/test_m2_3_response.py -q`

**21 passed, 0 failed** (previously 19 collected with 2 failing). Coverage
includes each response type, prediction/SHAP value preservation, citation/source
preservation, synthetic provenance disclosure, non-causal language, insufficient-
evidence handling, unsupported questions, tool failures, contradictory tool
outputs, driver vs technical formats, and real local integration against the
M2.2 tools (saved M1.2 models + synthetic corpus/vector store). No retraining,
no retrieval-parameter changes.

---

## 9. Limitations

- Response generation and validation are deterministic; there is no LLM, so
  phrasing is templated, not free-form natural language.
- The causal-language and OEM checks are heuristic regex rules; they can
  over- or under-match on unusual phrasing (this is documented, not hidden).
- Evidence sufficiency uses a fixed similarity floor heuristic, not calibrated
  per query.
- Driver-facing prose is intended for readability, not legal or compliance
  accuracy; all values are synthetic-trained estimates.

---

## 10. How to run locally

```bash
# run the response-evaluation (writes reports/m2/response_evaluation/*)
python scripts/run_m2_3_response_eval.py

# run the focused M2.3 test suite
python -m pytest tests/test_m2_3_response.py
```

Programmatic use:

```python
from src.agent.orchestrator import AgentOrchestrator
from src.agent.response_generator import ResponseGenerator

resp = AgentOrchestrator().handle("predict fuel consumption", {...inputs...})
grounded = ResponseGenerator(inputs={...inputs...}).generate(resp)
print(grounded.driver_version)
print(grounded.technical_version)
```
