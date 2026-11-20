# M4.5 End-to-End Driver Interaction Demo & Evaluation

**Milestone:** M4.5
**Routing backend:** deterministic_driver (no LLM)
**Feature set:** `full_operational`
**Seed:** 20250801
**Synthetic-data caveat:** predictions/explanations/retrieval come from synthetic-trained models and a synthetic corpus — not real-world observations.

## Evaluation summary

- Checks: 63/63 passed
- Scenarios completed: 3/3

| Scenario | checks | passed | completed |
| --- | --- | --- | --- |
| high_fuel_report_full_journey | 45 | 45 | True |
| unsupported_request | 9 | 9 | True |
| ambiguous_request | 9 | 9 | True |

## Scenarios

### high_fuel_report_full_journey

Driver reports high fuel consumption -> incomplete context -> clarification -> valid inputs -> prediction -> explanation -> knowledge guidance.

| Turn | intent | expected | tools | expected tools | delivered | missing | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clarifying | clarifying | — | — | False | vehicle_class, gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, acceleration_mps2, road_grade_pct, ignition_status, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | — |
| 2 | clarifying | clarifying | — | — | False | vehicle_class, gross_vehicle_weight_kg, payload_kg, vehicle_mass_kg, acceleration_mps2, road_grade_pct, ignition_status, engine_rpm, engine_load_pct, throttle_position_pct, torque_nm, coolant_temp_c, intake_air_temp_c, ambient_temp_c | — |
| 3 | predict_fuel | predict_fuel | fuel_rate_prediction | fuel_rate_prediction | True | — | OK |
| 4 | explain_prediction | explain_prediction | prediction_explanation | prediction_explanation | True | — | OK |
| 5 | knowledge_query | knowledge_query | knowledge_retrieval | knowledge_retrieval | True | — | OK |

### unsupported_request

Out-of-scope request must remain unsupported and safe.

| Turn | intent | expected | tools | expected tools | delivered | missing | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | unsupported | unsupported | — | — | True | — | — |

### ambiguous_request

Ambiguous request must ask a targeted question, not guess.

| Turn | intent | expected | tools | expected tools | delivered | missing | validation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ambiguous | ambiguous | — | — | False | — | — |

## Verification

- **real_artifacts_used:** True
- **models_loaded_not_refitted:** True
- **preprocessors_loaded_not_refitted:** True
- **shap_explanation_instance_level:** True
- **shap_is_attribution_not_causality:** True
- **rag_corpus_synthetic:** True
- **no_fabricated_telemetry:** True
- **no_oem_claims_from_synthetic_corpus:** True
- **no_unsupported_fuel_savings_claims:** True
- **failed_validation_not_presented_as_success:** True
- **synthetic_corruption_or_environment_note:** The demo runs on a synthetic telemetry-derived model and a synthetic knowledge corpus; outputs are model estimates, not real-world observations.

## Limitations

- Synthetic data: predictions, explanations, and retrieved passages are all derived from synthetic telemetry / a synthetic corpus. They are not validated real-world fuel measurements or manufacturer guidance.
- SHAP contributions describe how the model composes its prediction; they are not causal evidence about real-world fuel consumption.
- The demo exercises the deterministic, rule-based routing path; it is not an LLM and does not generate free-form natural language.
- Only the scenario inputs exercised here are shown; the demo is a representative workflow, not an exhaustive coverage of all driver phrasings.
- Retrieval similarity is not proof of factual correctness; evidence sufficiency is judged against a documented heuristic similarity floor.
