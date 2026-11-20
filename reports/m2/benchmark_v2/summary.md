# M2.1-B RAG Benchmark v2 — Summary

Generated: `2026-11-19T18:59:20+05:30`

## Purpose

A diverse, document-level retrieval benchmark over the *synthetic* RAG corpus. It complements the original 55-case topic-level baseline without modifying it, the corpus, or retrieval parameters.

## Corpus (unchanged)

- Documents: 21 (all synthetic educational; `is_oem_manual=false`).
- Chunks: 21 (one per short document).
- **Notice:** all corpus content is synthetic educational material, not OEM/manufacturer documentation.

## Baseline preserved

- 55 original `rag_retrieval` queries kept unchanged.
- Unique query strings: 2 (`road_grade_effects`=28, `engine_load_fuel_demand`=27).

## Benchmark v2

- Total queries: 28 (all unique; no superficial paraphrases).
- Answerable: 24, unanswerable: 4.
- Categories: conceptual, cause-and-effect, operational scenarios, comparisons, practical driver questions, multi-document questions, unanswerable.
- Validation: True (every relevant doc id exists in the corpus; unanswerable queries assign no relevant ids).

## Metrics (mean over answerable queries)

| Metric | Value |
| --- | --- |
| Recall@1 | 0.8333 |
| Recall@3 | 0.9375 |
| Recall@5 | 0.9792 |
| Precision@1 | 1.0 |
| Precision@3 | 0.4028 |
| Precision@5 | 0.2583 |
| MRR | 1.0 |
| nDCG@5 | 0.9692 |
| Zero-relevant results | 0 |

## Per-topic results (answerable queries)

| Topic | n | R@1 | R@3 | R@5 | P@1 | P@3 | P@5 | nDCG@5 | MRR | failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| driver_feedback_safety | 2 | 0.75 | 1.0 | 1.0 | 1.0 | 0.5 | 0.3 | 1.0 | 1.0 | 0 |
| engine_load_fuel_demand | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 0 |
| fuel_consumption_fundamentals | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 0 |
| fuel_rate_interpretation | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 0 |
| model_limitations | 3 | 0.8333 | 0.8333 | 1.0 | 1.0 | 0.3333 | 0.2667 | 0.9501 | 1.0 | 0 |
| payload_effects | 2 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3333 | 0.2 | 1.0 | 1.0 | 0 |
| road_grade_effects | 3 | 0.5 | 0.8333 | 1.0 | 1.0 | 0.5556 | 0.4 | 0.9591 | 1.0 | 0 |
| rpm_torque_relationships | 2 | 0.75 | 0.75 | 0.75 | 1.0 | 0.3333 | 0.2 | 0.8065 | 1.0 | 0 |
| shap_interpretation | 2 | 0.75 | 1.0 | 1.0 | 1.0 | 0.5 | 0.3 | 1.0 | 1.0 | 0 |
| speed_variability_driving_behaviour | 4 | 0.875 | 1.0 | 1.0 | 1.0 | 0.4166 | 0.25 | 0.9799 | 1.0 | 0 |

## Representative successful retrieval

- `bq_01` (fuel_consumption_fundamentals): top relevant at rank 1; relevant=['doc_001'], retrieved_top5=['doc_001', 'doc_012', 'doc_008', 'doc_013', 'doc_004']
- `bq_02` (fuel_rate_interpretation): top relevant at rank 1; relevant=['doc_013'], retrieved_top5=['doc_013', 'doc_002', 'doc_001', 'doc_017', 'doc_021']
- `bq_03` (fuel_consumption_fundamentals): top relevant at rank 1; relevant=['doc_002'], retrieved_top5=['doc_002', 'doc_021', 'doc_013', 'doc_001', 'doc_014']
- `bq_04` (engine_load_fuel_demand): top relevant at rank 1; relevant=['doc_003'], retrieved_top5=['doc_003', 'doc_007', 'doc_001', 'doc_009', 'doc_010']

## Representative failed retrieval (zero relevant in top-5)

- None — every answerable query retrieved at least one relevant document in the top-5.

## Representative imperfect retrieval (not all relevant docs found in top-5)

- `bq_13` (rpm_torque_relationships): recall@5=0.5; relevant=['doc_006', 'doc_005'], retrieved_top5=['doc_006', 'doc_011', 'doc_010', 'doc_012', 'doc_001']

## Caveats

- Relevance labels are assigned from actual document content, not topic metadata alone; every label has a short recorded explanation in `benchmark.json`.
- Unanswerable queries are handled separately and are excluded from metric aggregates; no relevant documents were artificially assigned to them.
- Deterministic pipeline: same model, fixed corpus, fixed queries => reproducible.
- Results describe retrieval over synthetic educational text, not verified manufacturer documentation.
