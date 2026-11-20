# M2.1 Retrieval Evaluation Summary

Generated: `2026-11-19T18:49:40+05:30`

## Corpus

- Documents: 21 (all synthetic educational; `is_oem_manual=false`).
- Chunks: 21 (one per short document; no fragmentation).
- Source references preserved per chunk (`source_reference`, `document_id`).

## Embedding & retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2` (dimension 384), loaded locally; no download required.
- Similarity: cosine (L2-normalized dot product).
- Evaluation top-k: 5; metadata filtering supported.

## Queries & ground truth

- 55 labeled `rag_retrieval` queries; 55 evaluated.
- Ground truth: topic-level mapping validated against the corpus (all topics resolved).

## Metrics (mean over labeled queries)

| Metric | Value |
| --- | --- |
| Recall@1 | 0.2455 |
| Recall@3 | 0.2455 |
| Recall@5 | 0.5 |
| Precision@1 | 0.4909 |
| Precision@3 | 0.1636 |
| Precision@5 | 0.2 |
| MRR | 0.6182 |
| nDCG@5 | 0.7102 |

## Examples

### Successful retrieval (top-5 contains relevant chunk)

- `case_0386` (road_grade_effects): top1_relevant=False
- `case_0387` (engine_load_fuel_demand): top1_relevant=True
- `case_0388` (road_grade_effects): top1_relevant=False
- `case_0389` (engine_load_fuel_demand): top1_relevant=True

### Failed retrieval (top-5 misses all relevant chunks)


## Caveats

- Corpus is synthetic; these numbers describe retrieval over synthetic educational text, not verified manufacturer documentation.
- Relevance is topic-based ground truth, not a human-judged gold standard.
- Deterministic pipeline: same model, fixed corpus, fixed queries => reproducible under the documented configuration.
