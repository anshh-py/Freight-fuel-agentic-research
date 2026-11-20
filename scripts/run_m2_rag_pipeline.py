"""Reproducible M2.1 RAG retrieval pipeline.

Steps:
  1. Validate the synthetic corpus and produce ``rag_corpus_validation.json``.
  2. Chunk documents (deterministic, no fragmentation).
  3. Build the embedding index and a VectorStore.
  4. Load the 55 labeled ``rag_retrieval`` queries.
  5. Validate ground-truth topic->document relevance.
  6. Evaluate retrieval and write ``retrieval_evaluation.json`` and the
     ``retrieval_evaluation_summary.md`` report.

The script is run from the workspace root:
    python scripts/run_m2_rag_pipeline.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.rag.corpus import chunk_documents, validate_corpus
from src.rag.embedder import LocalEmbedder
from src.rag.retrieval import (
    evaluate_retrieval,
    load_eval_queries,
    validate_ground_truth,
)
from src.rag.vectorstore import VectorStore

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_DIR = "reports/m2"
TOP_K_EVAL = 5


def _assure_report_dir() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    _assure_report_dir()

    # 1. Corpus validation
    corpus_report = validate_corpus()
    with open(os.path.join(REPORT_DIR, "rag_corpus_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(corpus_report, fh, indent=2, ensure_ascii=False)

    # 2. Chunking
    chunks = chunk_documents()

    # 3. Index
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    store = VectorStore(embedder)
    store.add(chunks)

    # 4. Queries
    queries = load_eval_queries()
    rag_count = len(queries)

    # 5. Ground-truth validation
    gt = validate_ground_truth(chunks, queries)

    # 6. Retrieval evaluation
    eval_result = evaluate_retrieval(store, queries, gt["relevant_documents_per_topic"], ks=(1, 3, 5))

    report = {
        "pipeline": "M2.1 RAG knowledge base retrieval",
        "generated_at": _iso(),
        "embedding": {
            "model": embedder.model_name,
            "dimension": embedder.dimension,
            "pooling": "mean",
            "normalization": "L2 (cosine similarity)",
            "local": True,
            "download_required": False,
        },
        "retrieval": {
            "top_k": TOP_K_EVAL,
            "similarity": "cosine (dot product of L2-normalized embeddings)",
            "metadata_filters": "supported (dict or predicate) in VectorStore.search",
        },
        "corpus": {
            "path": "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl",
            "valid_documents": corpus_report["valid_documents"],
            "chunk_count": len(chunks),
            "synthetic": True,
            "oem_manual_count": corpus_report["oem_manual_count"],
        },
        "ground_truth": gt,
        "evaluation": eval_result["metrics"],
        "examples": {
            "successful": eval_result["successful_examples"],
            "failed": eval_result["failed_examples"],
        },
        "caveats": {
            "corpus_is_synthetic": "All documents are synthetic educational texts (is_oem_manual=false); "
            "results explain retrieval over synthetic content, not verified manufacturer documentation.",
            "relevance_is_topic_based": "Ground truth is a topic-level mapping from query labels to corpus "
            "documents; it is not a human-judged relevance gold standard.",
            "fixed_corpus": "The corpus is fixed; retrieval does not mutate it.",
        },
    }

    eval_path = os.path.join(REPORT_DIR, "retrieval_evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    _write_summary(store, corpus_report, eval_result, rag_count)

    print(f"Done. Wrote {eval_path}")
    print(f"Corpus valid docs: {corpus_report['valid_documents']}, chunks: {len(chunks)}")
    print(f"Embedding: {embedder.model_name}, dim={embedder.dimension}")
    print(f"Queries (labeled rag_retrieval): {rag_count}, evaluated: {eval_result['metrics']['n_queries_evaluated']}")
    print(f"mrr={eval_result['metrics']['mrr']}, recall@5={eval_result['metrics']['recall@5']}, precision@5={eval_result['metrics']['precision@5']}")


def _write_summary(store: VectorStore, corpus_report: dict, eval_result: dict, rag_count: int) -> None:
    m = eval_result["metrics"]
    lines = [
        "# M2.1 Retrieval Evaluation Summary",
        "",
        f"Generated: `{_iso()}`",
        "",
        "## Corpus",
        "",
        f"- Documents: {corpus_report['valid_documents']} (all synthetic educational; `is_oem_manual=false`).",
        f"- Chunks: {store.size} (one per short document; no fragmentation).",
        "- Source references preserved per chunk (`source_reference`, `document_id`).",
        "",
        "## Embedding & retrieval",
        "",
        f"- Model: `{store.embedder.model_name}` (dimension {store.dimension}), loaded locally; no download required.",
        "- Similarity: cosine (L2-normalized dot product).",
        f"- Evaluation top-k: {TOP_K_EVAL}; metadata filtering supported.",
        "",
        "## Queries & ground truth",
        "",
        f"- {rag_count} labeled `rag_retrieval` queries; {m['n_queries_evaluated']} evaluated.",
        "- Ground truth: topic-level mapping validated against the corpus (all topics resolved).",
        "",
        "## Metrics (mean over labeled queries)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Recall@1 | {m['recall@1']} |",
        f"| Recall@3 | {m['recall@3']} |",
        f"| Recall@5 | {m['recall@5']} |",
        f"| Precision@1 | {m['precision@1']} |",
        f"| Precision@3 | {m['precision@3']} |",
        f"| Precision@5 | {m['precision@5']} |",
        f"| MRR | {m['mrr']} |",
        f"| nDCG@5 | {m['ndcg@5']} |",
        "",
        "## Examples",
        "",
        "### Successful retrieval (top-5 contains relevant chunk)",
        "",
    ]
    for ex in eval_result["successful_examples"]:
        lines.append(f"- `{ex['case_id']}` ({ex['topic']}): top1_relevant={ex['top1_relevant']}")
    lines += [
        "",
        "### Failed retrieval (top-5 misses all relevant chunks)",
        "",
    ]
    for ex in eval_result["failed_examples"]:
        lines.append(f"- `{ex['case_id']}` ({ex['topic']}): retrieved={ex['retrieved_ids']}")
    lines += [
        "",
        "## Caveats",
        "",
        "- Corpus is synthetic; these numbers describe retrieval over synthetic educational text, not verified manufacturer documentation.",
        "- Relevance is topic-based ground truth, not a human-judged gold standard.",
        "- Deterministic pipeline: same model, fixed corpus, fixed queries => reproducible under the documented configuration.",
        "",
    ]
    summary_path = os.path.join(REPORT_DIR, "retrieval_evaluation_summary.md")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
