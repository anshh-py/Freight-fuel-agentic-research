"""M2.1-B: diverse RAG benchmark v2 — validation and evaluation.

Flow (run from workspace root):

    python scripts/run_m2_benchmark_v2.py

Steps:
  1. Validate the benchmark file (unique queries, answerable/unanswerable
     consistency, every relevant document id exists in the corpus, no relevant
     id assigned to unanswerable queries).
  2. Build the embedding index / VectorStore over the (unchanged) corpus.
  3. Run document-level retrieval evaluation per query.
  4. Compute per-topic aggregates over answerable queries.
  5. Write ``evaluation.json`` and ``summary.md`` under
     ``reports/m2/benchmark_v2/``.

The original 55-case baseline and its reports under ``reports/m2/`` are not
touched. Retrieval parameters are not tuned against this benchmark.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from src.rag.corpus import chunk_documents, validate_corpus
from src.rag.embedder import LocalEmbedder
from src.rag.retrieval import evaluate_benchmark_v2
from src.rag.vectorstore import VectorStore

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REPORT_DIR = "reports/m2/benchmark_v2"
BENCHMARK_PATH = os.path.join(REPORT_DIR, "benchmark.json")
TOP_K_EVAL = 5


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _assure_report_dir() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)


def validate_benchmark(benchmark: dict, doc_ids_in_corpus: set[str]) -> dict:
    """Structural checks on the benchmark file. Returns a report dict and
    raises ValueError on any hard inconsistency."""
    queries = benchmark["queries"]
    problems: list[str] = []

    # Unique query strings (case-sensitive) — no superficial duplicates.
    seen: dict[str, str] = {}
    for q in queries:
        key = q["user_query"].strip()
        if key in seen:
            problems.append(f"duplicate query string: {key!r} ({q['query_id']} vs {seen[key]})")
        seen[key] = q["query_id"]

    answerable = 0
    unanswerable = 0
    for q in queries:
        rel = q.get("relevant_document_ids", [])
        if q.get("answerable", True):
            answerable += 1
            if not rel:
                problems.append(f"{q['query_id']}: answerable query has no relevant ids")
            for d in rel:
                if d not in doc_ids_in_corpus:
                    problems.append(f"{q['query_id']}: relevant doc {d} not in corpus")
        else:
            unanswerable += 1
            if rel:
                problems.append(f"{q['query_id']}: unanswerable query must not assign relevant ids")

    report = {
        "total_queries": len(queries),
        "unique_query_strings": len(seen),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "validation_passed": not problems,
        "problems": problems,
    }
    if problems:
        raise ValueError("benchmark validation failed: " + "; ".join(problems))
    return report


def _topic_aggregate(per_query: list[dict]) -> dict[str, dict]:
    """Aggregate metrics per topic over answered queries only."""
    topics: dict[str, list[dict]] = {}
    for q in per_query:
        if not q.get("scored", False):
            continue
        topics.setdefault(q.get("topic", "unknown"), []).append(q)

    out: dict[str, dict] = {}
    for topic, items in sorted(topics.items()):
        n = len(items)
        out[topic] = {
            "n_queries": n,
            "recall@1": round(sum(i["recall@1"] for i in items) / n, 4),
            "recall@3": round(sum(i["recall@3"] for i in items) / n, 4),
            "recall@5": round(sum(i["recall@5"] for i in items) / n, 4),
            "precision@1": round(sum(i["precision@1"] for i in items) / n, 4),
            "precision@3": round(sum(i["precision@3"] for i in items) / n, 4),
            "precision@5": round(sum(i["precision@5"] for i in items) / n, 4),
            "ndcg@5": round(sum(i["ndcg@5"] for i in items) / n, 4),
            "mrr": round(
                sum(1.0 / i["first_relevant_rank"] if i.get("first_relevant_rank") else 0.0 for i in items) / n,
                4,
            ),
            "n_failed": sum(0 if i.get("matched", False) else 1 for i in items),
        }
    return out


def main() -> None:
    _assure_report_dir()

    # 1. Corpus (unchanged) + benchmark validation
    corpus_report = validate_corpus()
    chunks = chunk_documents()
    doc_ids = {c["document_id"] for c in chunks}
    with open(BENCHMARK_PATH, encoding="utf-8") as fh:
        benchmark = json.load(fh)
    bm_validation = validate_benchmark(benchmark, doc_ids)

    # 2. Index (same embedder + store as the M2.1 pipeline)
    embedder = LocalEmbedder(EMBEDDING_MODEL)
    store = VectorStore(embedder)
    store.add(chunks)

    # 3. Document-level evaluation
    queries = benchmark["queries"]
    eval_result = evaluate_benchmark_v2(store, queries, ks=(1, 3, 5))
    metrics = eval_result["metrics"]
    per_query = eval_result["per_query"]

    # 4. Per-topic aggregates
    per_topic = _topic_aggregate(per_query)

    # Representative examples (success/failure) among answerable queries.
    scored = [q for q in per_query if q.get("scored", False)]
    successes = [q for q in scored if q.get("matched", False)]
    failures = [q for q in scored if not q.get("matched", False)]
    examples_ok = successes[:4]
    examples_fail = failures[:4]

    # Imperfect retrieval: answerable queries where not all relevant documents
    # appear in the top-5 (recall@5 < 1.0). This is the meaningful limitation
    # signal for a multi-document benchmark.
    imperfect = sorted(
        [q for q in scored if (q.get("recall@5") or 0.0) < 1.0],
        key=lambda q: q["recall@5"],
    )

    report = {
        "pipeline": "M2.1-B RAG Benchmark v2 (diverse, document-level)",
        "generated_at": _iso(),
        "embedding": {
            "model": embedder.model_name,
            "dimension": embedder.dimension,
            "normalization": "L2 (cosine similarity)",
            "local": True,
            "download_required": False,
        },
        "corpus": {
            "path": "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl",
            "valid_documents": corpus_report["valid_documents"],
            "chunk_count": len(chunks),
            "synthetic": True,
            "oem_manual_count": corpus_report["oem_manual_count"],
            "notice": "All corpus content is synthetic educational material, not OEM/manufacturer documentation.",
        },
        "baseline_preserved": benchmark["meta"]["baseline_preserved"],
        "benchmark_validation": bm_validation,
        "evaluation": {
            "n_unique_queries": bm_validation["unique_query_strings"],
            "n_answerable": bm_validation["answerable"],
            "n_unanswerable": bm_validation["unanswerable"],
            "n_answerable_evaluated": metrics["n_answerable_evaluated"],
            "zero_relevant_result_count": metrics["zero_relevant_result_count"],
            "recall@1": metrics["recall@1"],
            "recall@3": metrics["recall@3"],
            "recall@5": metrics["recall@5"],
            "precision@1": metrics["precision@1"],
            "precision@3": metrics["precision@3"],
            "precision@5": metrics["precision@5"],
            "ndcg@1": metrics["ndcg@1"],
            "ndcg@3": metrics["ndcg@3"],
            "ndcg@5": metrics["ndcg@5"],
            "mrr": metrics["mrr"],
        },
        "per_topic": per_topic,
        "per_query": per_query,
        "imperfect_retrieval": [
            {
                "query_id": e["query_id"],
                "topic": e["topic"],
                "query": next(q["user_query"] for q in queries if q["query_id"] == e["query_id"]),
                "relevant_doc_ids": next(q["relevant_document_ids"] for q in queries if q["query_id"] == e["query_id"]),
                "retrieved_doc_ids": e["retrieved_doc_ids"],
                "recall@5": e.get("recall@5"),
            }
            for e in imperfect
        ],
        "examples": {
            "successful_retrieval": [
                {
                    "query_id": e["query_id"],
                    "topic": e["topic"],
                    "query": next(q["user_query"] for q in queries if q["query_id"] == e["query_id"]),
                    "relevant_doc_ids": next(q["relevant_document_ids"] for q in queries if q["query_id"] == e["query_id"]),
                    "retrieved_doc_ids": e["retrieved_doc_ids"],
                    "first_relevant_rank": e.get("first_relevant_rank"),
                }
                for e in examples_ok
            ],
            "failed_retrieval": [
                {
                    "query_id": e["query_id"],
                    "topic": e["topic"],
                    "query": next(q["user_query"] for q in queries if q["query_id"] == e["query_id"]),
                    "relevant_doc_ids": next(q["relevant_document_ids"] for q in queries if q["query_id"] == e["query_id"]),
                    "retrieved_doc_ids": e["retrieved_doc_ids"],
                }
                for e in examples_fail
            ],
        },
        "caveats": {
            "relevance_is_content_based": "Relevance labels are assigned from actual document content (not topic metadata alone); each label carries a short explanation.",
            "unanswerable_treated_separately": "Unanswerable queries have no assigned relevant documents and are excluded from metric aggregates.",
            "fixed_corpus": "The synthetic corpus is unchanged; retrieval parameters were not tuned against this benchmark.",
        },
    }

    eval_path = os.path.join(REPORT_DIR, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    _write_summary(report, corpus_report, store)

    print(f"Done. Wrote {eval_path}")
    print(f"Unique queries: {bm_validation['unique_query_strings']} | "
          f"answerable: {bm_validation['answerable']} | "
          f"unanswerable: {bm_validation['unanswerable']}")
    print(f"Recall@5={metrics['recall@5']}, Precision@5={metrics['precision@5']}, "
          f"MRR={metrics['mrr']}, nDCG@5={metrics['ndcg@5']}")
    print(f"Zero-relevant results: {metrics['zero_relevant_result_count']}")


def _write_summary(report: dict, corpus_report: dict, store: VectorStore) -> None:
    ev = report["evaluation"]
    lines = [
        "# M2.1-B RAG Benchmark v2 — Summary",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Purpose",
        "",
        "A diverse, document-level retrieval benchmark over the *synthetic* RAG corpus. It complements the original 55-case topic-level baseline without modifying it, the corpus, or retrieval parameters.",
        "",
        "## Corpus (unchanged)",
        "",
        f"- Documents: {corpus_report['valid_documents']} (all synthetic educational; `is_oem_manual=false`).",
        f"- Chunks: {store.size} (one per short document).",
        "- **Notice:** all corpus content is synthetic educational material, not OEM/manufacturer documentation.",
        "",
        "## Baseline preserved",
        "",
        "- 55 original `rag_retrieval` queries kept unchanged.",
        f"- Unique query strings: {report['baseline_preserved']['unique_query_strings']} "
        f"(`road_grade_effects`={report['baseline_preserved']['road_grade_effects_count']}, "
        f"`engine_load_fuel_demand`={report['baseline_preserved']['engine_load_fuel_demand_count']}).",
        "",
        "## Benchmark v2",
        "",
        f"- Total queries: {ev['n_unique_queries']} (all unique; no superficial paraphrases).",
        f"- Answerable: {ev['n_answerable']}, unanswerable: {ev['n_unanswerable']}.",
        "- Categories: conceptual, cause-and-effect, operational scenarios, comparisons, practical driver questions, multi-document questions, unanswerable.",
        f"- Validation: {report['benchmark_validation']['validation_passed']} (every relevant doc id exists in the corpus; unanswerable queries assign no relevant ids).",
        "",
        "## Metrics (mean over answerable queries)",
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
        f"| Zero-relevant results | {ev['zero_relevant_result_count']} |",
        "",
        "## Per-topic results (answerable queries)",
        "",
        "| Topic | n | R@1 | R@3 | R@5 | P@1 | P@3 | P@5 | nDCG@5 | MRR | failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for topic, t in report["per_topic"].items():
        lines.append(
            f"| {topic} | {t['n_queries']} | {t['recall@1']} | {t['recall@3']} | {t['recall@5']} | "
            f"{t['precision@1']} | {t['precision@3']} | {t['precision@5']} | {t['ndcg@5']} | "
            f"{t['mrr']} | {t['n_failed']} |"
        )
    lines += [
        "",
        "## Representative successful retrieval",
        "",
    ]
    for e in report["examples"]["successful_retrieval"]:
        lines.append(f"- `{e['query_id']}` ({e['topic']}): top relevant at rank {e['first_relevant_rank']}; "
                     f"relevant={e['relevant_doc_ids']}, retrieved_top5={e['retrieved_doc_ids'][:5]}")
    lines += [
        "",
        "## Representative failed retrieval (zero relevant in top-5)",
        "",
    ]
    if report["examples"]["failed_retrieval"]:
        for e in report["examples"]["failed_retrieval"]:
            lines.append(f"- `{e['query_id']}` ({e['topic']}): relevant={e['relevant_doc_ids']}, retrieved_top5={e['retrieved_doc_ids'][:5]}")
    else:
        lines.append("- None — every answerable query retrieved at least one relevant document in the top-5.")
    lines += [
        "",
        "## Representative imperfect retrieval (not all relevant docs found in top-5)",
        "",
    ]
    if report.get("imperfect_retrieval"):
        for e in report["imperfect_retrieval"]:
            lines.append(f"- `{e['query_id']}` ({e['topic']}): recall@5={e['recall@5']}; relevant={e['relevant_doc_ids']}, retrieved_top5={e['retrieved_doc_ids'][:5]}")
    else:
        lines.append("- None — all relevant documents retrieved for every answerable query.")
    lines += [
        "",
        "## Caveats",
        "",
        "- Relevance labels are assigned from actual document content, not topic metadata alone; every label has a short recorded explanation in `benchmark.json`.",
        "- Unanswerable queries are handled separately and are excluded from metric aggregates; no relevant documents were artificially assigned to them.",
        "- Deterministic pipeline: same model, fixed corpus, fixed queries => reproducible.",
        "- Results describe retrieval over synthetic educational text, not verified manufacturer documentation.",
        "",
    ]
    summary_path = os.path.join(REPORT_DIR, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
