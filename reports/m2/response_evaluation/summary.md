# M2.3 — Response Evaluation Summary

- Dataset: `reports/m2/response_evaluation/benchmark.json`
- Scenarios: **21**
- Passed: **21**, Failed: **0**
- Scoring: fully deterministic (`src.agent.response_validation`); no LLM judge, no paid API.

## Categories covered

- ambiguous: U01
- combined_knowledge_prediction: C01
- contradictory_or_prohibited: N01, N02, N03, N04, N05, N06, N07, N08, N09, C02
- insufficient_evidence: I01
- knowledge_only: K01
- prediction_only: P01, P02
- prediction_plus_explanation: E01, E02
- tool_failure: F01, M01
- unsupported_oem_spec: U02

## Positive (expected-pass) check coverage

- `P01` (prediction_only): synthetic_disclaimer, prediction_value_preserved, causal_language, oem_mislabeling, failure_not_presented_as_success
- `P02` (prediction_only): synthetic_disclaimer, prediction_value_preserved, causal_language, oem_mislabeling
- `K01` (knowledge_only): source_attribution, synthetic_corpus_notice, oem_mislabeling, insufficient_evidence_stated, causal_language
- `E01` (prediction_plus_explanation): prediction_value_preserved, shap_value_preserved, synthetic_disclaimer, causal_language, oem_mislabeling
- `C01` (combined_knowledge_prediction): source_attribution, synthetic_corpus_notice, prediction_value_preserved, causal_language, oem_mislabeling
- `U01` (ambiguous): unsupported_not_answered, causal_language, oem_mislabeling
- `U02` (unsupported_oem_spec): unsupported_not_answered, causal_language, oem_mislabeling
- `F01` (tool_failure): failure_not_presented_as_success, causal_language
- `I01` (insufficient_evidence): insufficient_evidence_stated, causal_language, oem_mislabeling
- `C02` (contradictory_or_prohibited): source_attribution, synthetic_corpus_notice, oem_mislabeling
- `M01` (tool_failure): failure_not_presented_as_success, causal_language, oem_mislabeling
- `E02` (prediction_plus_explanation): prediction_value_preserved, shap_value_preserved, synthetic_disclaimer, causal_language, oem_mislabeling

## Negative (expected-violation) coverage

- `N01` (contradictory_or_prohibited): causal_language
- `N02` (contradictory_or_prohibited): oem_mislabeling
- `N03` (contradictory_or_prohibited): synthetic_disclaimer
- `N04` (contradictory_or_prohibited): source_attribution
- `N05` (contradictory_or_prohibited): unsupported_not_answered
- `N06` (contradictory_or_prohibited): insufficient_evidence_stated
- `N07` (contradictory_or_prohibited): failure_not_presented_as_success
- `N08` (contradictory_or_prohibited): prediction_value_preserved
- `N09` (contradictory_or_prohibited): shap_value_preserved

## Manual review notes (documented, not scored)

- `P01`: Check readability of the no-jargon driver summary.
- `P02`: Confirm the telematics feature set is clearly labelled.
- `K01`: Passages are clearly cited with ids and synthetic provenance.
- `E01`: Driver version should show estimated rate, top drivers, and limitations.
- `C01`: Combined response must keep prediction and evidence clearly separated.
- `U01`: Response should ask for clarification, not guess.
- `U02`: Must decline without inventing a torque value.
- `F01`: Failure must be explicit; no predicted value fabricated.
- `I01`: Response must explicitly say evidence is insufficient.
- `N01`: Validation must flag the causal claim.
- `N02`: Validation must flag OEM framing of synthetic corpus.
- `N03`: Validation must detect the missing disclaimer.
- `N04`: Validation must flag the fabricated citation.
- `N05`: Validation must flag fabricated numbers under an unsupported query.
- `N06`: Validation must require explicit insufficiency disclosure.
- `N07`: Validation must flag failure presented as success.
- `N08`: Validation must flag the missing value.
- `N09`: Validation must flag the dropped SHAP value.
- `C02`: Both passages cited with provenance; no fabricated resolution.
- `M01`: Failure must be explicit; no predicted value fabricated from missing data.
- `E02`: Telematics feature set clearly labelled; SHAP caveat present.
