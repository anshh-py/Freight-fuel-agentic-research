"""M5.2 — retrieval quality and answer-grounding evaluation runner.

Flow (run from workspace root):

    python scripts/run_m5_2_retrieval_grounding.py

Steps:
  1. Validate the M5.2 labeled query set (schema, IDs, abstention categories).
  2. Build the embedding index / VectorStore over the (unchanged) synthetic corpus.
  3. Compute retrieval metrics on answerable queries (reusing M2.1-B machinery);
     report unanswerable queries separately.
  4. Run deterministic grounding checks on answerable queries and abstention
     checks on unanswerable queries.
  5. Write ``evaluation.json`` and ``summary.md`` under ``reports/m5/m5_2/``.

The M2.1 / M2.4 / M5.1 historical reports are preserved untouched. No model is
retrained; no LLM / external API / new dataset is used. Everything is
deterministic.

Grounding labels are hand-authored for the synthetic environment (see
``src/evaluation/m5_2_query_set.py``). The report documents that they are NOT
an independent external human review.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.corpus import chunk_documents, validate_corpus
from src.rag.embedder import LocalEmbedder
from src.rag.vectorstore import VectorStore
from src.evaluation.m5_2_retrieval_grounding import (
    evaluate_grounding_and_abstention,
    evaluate_retrieval_metrics,
    validate_query_set,
)
from src.evaluation.m5_2_query_set import ABSTENTION_CATEGORIES, QUERIES

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_DIR = "reports/m5/m5_2"


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _assure_report_dir() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def write_summary(report: dict) -> None:
    ev = report["retrieval"]["metrics"]
    gr = report["grounding"]["summary"]
    abst = report["grounding"]["abstention"]
    unc = gr["uncertainty"]
    lines = [
        "# M5.2 — Retrieval Quality & Answer-Grounding Evaluation (Summary)",
        "",
        "- Pipeline: M5.2 (deterministic; no LLM, no external API, no new dataset).",
        f"- Corpus: synthetic (21 docs / 21 chunks) — `is_oem_manual=false` for all.",
        f"- Query set: {report['query_set']['n_queries']} total "
        f"({report['query_set']['n_answerable']} answerable, "
        f"{report['query_set']['n_unanswerable']} unanswerable).",
        "",
        "## Retrieval metrics (answerable queries only)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recall@1 | {ev['recall@1']} |",
        f"| Recall@3 | {ev['recall@3']} |",
        f"| Recall@5 | {ev['recall@5']} |",
        f"| Precision@1 | {ev['precision@1']} |",
        f"| Precision@3 | {ev['precision@3']} |",
        f"| Precision@5 | {ev['precision@5']} |",
        f"| MRR | {ev['mrr']} |",
        f"| nDCG@5 | {ev['ndcg@5']} |",
        f"| Zero-relevant results | {ev['zero_relevant']} |",
        "",
        "## Grounding (answerable queries)",
        "",
        f"- Answers fully grounded (no ungrounded sentence): {gr['n_answers_fully_grounded']}/{gr['n_answerable_grounded']}",
        f"- OEM mislabeling avoided: {gr['n_answers_oem_ok']}/{gr['n_answerable_grounded']}",
        f"- Source attribution correct: {gr['n_answers_attribution_ok']}/{gr['n_answerable_grounded']}",
        "",
        "## Uncertainty communication (corrected)",
        "",
        "Compliance is computed only over answers where communicating uncertainty is required.",
        "",
        f"- Uncertainty required and communicated: {unc['n_uncertainty_required_communicated']}/{unc['n_uncertainty_required']}",
        f"- Uncertainty required but missing: {unc['n_uncertainty_required_missing']} "
        f"({', '.join(unc['required_missing_query_ids']) if unc['required_missing_query_ids'] else 'none'})",
        f"- Uncertainty not required: {unc['n_uncertainty_not_required']}",
        f"- Compliance rate (over required only): {unc['compliance_rate']}",
        "",
        "## Abstention (unanswerable queries)",
        "",
        f"- Abstention cases: {gr['n_abstention']}",
        f"- Abstained with no fabrication: {gr['n_abstention_correct']}/{gr['n_abstention']}",
        f"- Invalid abstention category: {gr['n_abstention_invalid_category']}",
        "",
        "Abstention categories: " + ", ".join(ABSTENTION_CATEGORIES),
        "",
    ]
    for r in abst:
        lines.append(
            f"- `{r['query_id']}` [{r['abstention_category']}]: abstains_no_fabrication={r['abstains_no_fabrication']}"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- Labeling is curated / hand-authored for the synthetic research environment; it is **not** an independent external human review.",
        "- Retrieval metrics reflect retrieval over synthetic educational text, not verified manufacturer documentation.",
        "- Grounding support uses a token-overlap heuristic (documented), not semantic entailment; it is reported alongside the manual label for transparency.",
        "- All claims are limited to the synthetic research environment.",
        "",
    ]
    with open(os.path.join(REPORT_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    _assure_report_dir()

    # 1. Validate labeled query set
    validation_errors = validate_query_set(QUERIES)

    # 2. Corpus (unchanged) + index
    corpus_report = validate_corpus()
    chunks = chunk_documents()
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    store = VectorStore(embedder)
    store.add(chunks)

    # 3. Retrieval metrics (answerable only; reranswerable reported separately)
    retrieval = evaluate_retrieval_metrics(store, QUERIES)

    # 4. Grounding + abstention over actually-retrieved evidence
    doc_ids = sorted({c["document_id"] for c in chunks})
    grounding = evaluate_grounding_and_abstention(store, QUERIES, doc_ids)

    report = {
        "pipeline": "M5.2 retrieval quality and answer-grounding evaluation",
        "generated_at": _iso(),
        "query_set": {
            "n_queries": len(QUERIES),
            "n_answerable": sum(1 for q in QUERIES if q.get("answerable")),
            "n_unanswerable": sum(1 for q in QUERIES if not q.get("answerable")),
            "validation_passed": not validation_errors,
            "validation_errors": validation_errors,
        },
        "corpus": {
            "path": "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl",
            "valid_documents": corpus_report["valid_documents"],
            "chunk_count": len(chunks),
            "synthetic": True,
            "oem_manual_count": corpus_report["oem_manual_count"],
        },
        "embedding": {
            "model": embedder.model_name,
            "dimension": embedder.dimension,
            "normalization": "L2 (cosine similarity)",
            "local": True,
        },
        "retrieval": {
            "kind": "document-level, reusing M2.1-B evaluator",
            "metrics_are_separate_from_grounding": True,
            "metrics": retrieval,
        },
        "grounding": {
            "dimension": "answer-grounding (separate from retrieval metrics)",
            "labeling_notice": (
                "Grounding/relevance labels are curated for the synthetic environment; "
                "not an independent external human review."
            ),
            **grounding,
        },
        "caveats": {
            "corpus_is_synthetic": "All documents are synthetic educational texts (is_oem_manual=false).",
            "retrieval_relevance_document_level": "Relevance is document-level, assigned from content.",
            "grounding_heuristic": "Claim support uses a token-overlap heuristic, reported alongside manual labels.",
            "not_real_world": "No claim describes real-world validation; results are limited to the synthetic research environment.",
            "historical_reports_preserved": "M2.1, M2.4 and M5.1 reports are untouched.",
        },
    }

    eval_path = os.path.join(REPORT_DIR, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    write_summary(report)

    print(f"Done. Wrote {eval_path}")
    print(f"Queries: {report['query_set']['n_queries']} "
          f"(answerable {report['query_set']['n_answerable']}, "
          f"unanswerable {report['query_set']['n_unanswerable']})")
    print(f"Validation passed: {not validation_errors}")
    print(f"Recall@5={retrieval['recall@5']}, Precision@5={retrieval['precision@5']}, "
          f"MRR={retrieval['mrr']}, nDCG@5={retrieval['ndcg@5']}")


if __name__ == "__main__":
    main()
