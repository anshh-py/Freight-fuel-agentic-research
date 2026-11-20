# M5.2 — Retrieval Quality & Answer-Grounding Evaluation (Summary)

- Pipeline: M5.2 (deterministic; no LLM, no external API, no new dataset).
- Corpus: synthetic (21 docs / 21 chunks) — `is_oem_manual=false` for all.
- Query set: 26 total (19 answerable, 7 unanswerable).

## Retrieval metrics (answerable queries only)

| Metric | Value |
| --- | --- |
| Recall@1 | 0.8158 |
| Recall@3 | 0.9211 |
| Recall@5 | 0.9737 |
| Precision@1 | 1.0 |
| Precision@3 | 0.4035 |
| Precision@5 | 0.2632 |
| MRR | 1.0 |
| nDCG@5 | 0.9667 |
| Zero-relevant results | 0 |

## Grounding (answerable queries)

- Answers fully grounded (no ungrounded sentence): 18/19
- OEM mislabeling avoided: 19/19
- Source attribution correct: 19/19
- Uncertainty communicated: 7/19

## Abstention (unanswerable queries)

- Abstention cases: 7
- Abstained with no fabrication: 7/7
- Invalid abstention category: 0

Abstention categories: unanswerable_general, insufficient_evidence, unsupported_technical_spec, guaranteed_fuel_savings, real_world_validation

- `m52_q20` [unsupported_technical_spec]: abstains_no_fabrication=True
- `m52_q21` [unsupported_technical_spec]: abstains_no_fabrication=True
- `m52_q22` [unanswerable_general]: abstains_no_fabrication=True
- `m52_q23` [unanswerable_general]: abstains_no_fabrication=True
- `m52_q24` [guaranteed_fuel_savings]: abstains_no_fabrication=True
- `m52_q25` [real_world_validation]: abstains_no_fabrication=True
- `m52_q26` [insufficient_evidence]: abstains_no_fabrication=True

## Limitations

- Labeling is curated / hand-authored for the synthetic research environment; it is **not** an independent external human review.
- Retrieval metrics reflect retrieval over synthetic educational text, not verified manufacturer documentation.
- Grounding support uses a token-overlap heuristic (documented), not semantic entailment; it is reported alongside the manual label for transparency.
- All claims are limited to the synthetic research environment.
