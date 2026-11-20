# M5.1 Driver-Conversation Evaluation Benchmark

**Milestone:** M5.1
**Evaluation target:** M4 deterministic driver-feedback agent (no LLM)
**Generation:** curated, hand-authored, deterministic (no LLM; fixed ids/split)

## Size

- Cases: **53** (41 dev / 12 held-out)
- Turns: **58**
- Single-turn cases: 49; multi-turn: 4

## Category coverage

| Category | Cases |
| --- | --- |
| abstention (System must decline/abstain rather than guess.) | 2 |
| ambiguous (Under-specified request that must ask a targeted question.) | 4 |
| combined_intent (Request that needs more than one tool.) | 3 |
| contradictory_input (Driver gives conflicting/contradictory values.) | 3 |
| explanation_request (Request for why a prediction is what it is (SHAP).) | 4 |
| fuel_saving_guarantee (Attempt to elicit an unsupported savings guarantee.) | 3 |
| invalid_input (Driver supplies present but invalid values (range/type).) | 4 |
| knowledge_guidance (Knowledge / maintenance / practical efficiency guidance.) | 4 |
| multi_turn_clarification (Multi-turn dialogue: clarify -> supply -> predict.) | 4 |
| oem_claim (Attempt to get OEM/manufacturer specs from the synthetic corpus.) | 3 |
| out_of_range_malformed (Out-of-range or malformed field values.) | 2 |
| partial_input (Driver supplies only a subset of required inputs.) | 5 |
| prediction_request (Direct fuel-rate prediction request with complete inputs.) | 6 |
| safety_sensitive (Question touching physical/legal/operational safety.) | 3 |
| unsupported (Out-of-scope request that must stay unsupported.) | 3 |

## Methodology

Curated, hand-authored labeled scenarios (no LLM). Deterministic; scenario ids and dev/held-out split are fixed. Coverage spans the fifteen milestone categories. Held-out cases target semantically distinct edges/phrasings to avoid near-duplicate leakage.

## Limitations

All scenarios are synthetic research-environment cases. They do not represent real-world driver behaviour, and the agent's outputs are model estimates / synthetic-corpus knowledge, not real-world observations or manufacturer guidance.
