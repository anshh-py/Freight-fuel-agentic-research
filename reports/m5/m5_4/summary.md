# M5.4 Driver-Feedback Usefulness & Safety Evaluation

**Milestone:** M5.4
**Backend:** deterministic driver (no LLM)
**Feature set:** `full_operational`
**Seed:** 20250801
**Caveat:** engineering rubric, not a human-subject study

## Overall

- Overall score: **0.920** (185/201 applicable criteria passed)
- Calculation: pooled across all applicable criteria (passed / applicable), not an unweighted average of dimension scores; dimensions with zero applicable criteria are excluded

## Per-dimension scores

| Dimension | applicable | passed | score |
| --- | --- | --- | --- |
| relevance | 26 | 15 | 0.577 |
| actionability | 9 | 9 | 1.000 |
| clarity | 39 | 34 | 0.872 |
| evidence_uncertainty | 52 | 52 | 1.000 |
| safety | 46 | 46 | 1.000 |
| consistency | 29 | 29 | 1.000 |

## Safety-critical violations

None.

## Scenarios

### functional_prediction_and_explanation (type: functional, score 0.958)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | clarifying | clarifying | False | 13/14 |
| 2 | predict_fuel | predict_fuel | True | 17/17 |
| 3 | explain_prediction | explain_prediction | True | 16/17 |

### functional_knowledge_guidance (type: functional, score 1.000)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | knowledge_query | knowledge_query | True | 17/17 |

### functional_clarification_then_resume (type: functional, score 0.897)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | clarifying | clarifying | False | 13/14 |
| 2 | ambiguous | predict_fuel | False | 13/15 |

### negative_guaranteed_fuel_savings (type: negative_control, score 0.867)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | ambiguous | unsupported | False | 13/15 |

### negative_unsupported_mechanical_diagnosis (type: negative_control, score 0.800)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | clarifying | unsupported | False | 12/15 |

### negative_insufficient_contradictory_inputs (type: negative_control, score 0.933)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | clarifying | clarifying | False | 14/15 |

### negative_shap_as_causal_proof (type: negative_control, score 0.941)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | combined_request | explain_prediction | True | 16/17 |

### negative_unsafe_driving_request (type: negative_control, score 0.867)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | ambiguous | unsupported | False | 13/15 |

### negative_fabricated_telemetry (type: negative_control, score 0.867)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | ambiguous | unsupported | False | 13/15 |

### negative_synthetic_as_real_world (type: negative_control, score 1.000)

| Turn | intent | expected | delivered | passed/applicable |
| --- | --- | --- | --- | --- |
| 1 | unsupported | unsupported | True | 15/15 |

## Limitations

- This is an engineering evaluation using manually authored, deterministic criteria. It does NOT measure actual human driver satisfaction or comprehension, and no semantic comprehension has been experimentally validated.
- No LLM judge or external API is used; criteria are rule-based and observable on the generated text/structure.
- All predictions, explanations, and retrieved material are derived from synthetic telemetry and a synthetic corpus; they are not real-world measurements or manufacturer guidance.
- Resulting scores describe rubric compliance, not real-world effectiveness of driver feedback.
