"""Retrieval evaluation for the synthetic RAG knowledge base (M2.1).

Ground truth comes from the ``rag_retrieval`` queries in
``agent_evaluation_queries.csv``: each carries a label of the form
``document_id, topic=road_grade_effects`` in ``required_evidence_or_fields``.
A query is *relevant* to a chunk when the chunk's ``topic`` equals the labeled
topic. This mapping is validated before metrics are computed (the two labeled
topics must resolve to a non-empty set of corpus documents), so we never invent
labels that do not exist in the corpus.

Metrics computed on the 55 labeled queries (binary relevance, k=1/3/5):
  * Recall@k     -- |relevant in top-k| / |relevant total|
  * Precision@k  -- |relevant in top-k| / k
  * MRR          -- mean reciprocal rank of the first relevant chunk
  * nDCG@k       -- normalized discounted cumulative gain (binary gains)
"""

from __future__ import annotations

import csv
import re
from typing import Any, Iterable

import numpy as np

from src.rag.vectorstore import VectorStore

QUERIES_PATH = "data/synthetic_heavy_truck_research/agent_evaluation_queries.csv"
LIT_TOOL = "rag_retrieval"
LABEL_RE = re.compile(r"topic=([A-Za-z0-9_]+)")


# ---------------------------------------------------------------------------
# Query loading and label parsing
# ---------------------------------------------------------------------------
def load_eval_queries(path: str = QUERIES_PATH) -> list[dict[str, Any]]:
    """Load all rows with ``expected_primary_tool == rag_retrieval``."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("expected_primary_tool") == LIT_TOOL:
                out.append(row)
    return out


def parse_topic_label(label: str) -> str | None:
    """Extract the topic from a label like ``document_id, topic=road_grade_effects``."""
    m = LABEL_RE.search(label or "")
    return m.group(1) if m else None


def query_relevant_topics(queries: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each labeled topic to its queries (works as ground-truth validation)."""
    topics: dict[str, list[str]] = {}
    for q in queries:
        t = parse_topic_label(q.get("required_evidence_or_fields", ""))
        if not t:
            continue
        topics.setdefault(t, []).append(q["case_id"])
    return topics


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------
def validate_ground_truth(
    chunks: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate that every labeled topic exists in the corpus and has at least
    one document. Returns metadata plus per-topic relevant chunk IDs. Raises
    ValueError if any topic label cannot be resolved."""
    topic_to_ids: dict[str, list[str]] = {}
    for c in chunks:
        topic_to_ids.setdefault(str(c["topic"]), []).append(c["chunk_id"])

    labeled = query_relevant_topics(queries)
    unresolved = [t for t in labeled if t not in topic_to_ids]
    if unresolved:
        raise ValueError(f"ground-truth topics not present in corpus: {unresolved}")

    relevant: dict[str, list[str]] = {}
    for t, qids in labeled.items():
        relevant[t] = topic_to_ids[t]
    return {
        "labeled_topics": {t: len(qids) for t, qids in sorted(labeled.items())},
        "relevant_documents_per_topic": {t: ids for t, ids in sorted(relevant.items())},
        "all_topics_resolved": True,
    }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def _relevant_ids_for_query(query: dict[str, Any], topic_to_relevant: dict[str, list[str]]) -> set[str]:
    t = parse_topic_label(query.get("required_evidence_or_fields", ""))
    return set(topic_to_relevant.get(t or "", []))


def _dcg_at_k(relevant_rank: int | None, k: int) -> float:
    if relevant_rank is None:
        return 0.0
    return 1.0 / np.log2(relevant_rank + 1)


def evaluate_retrieval(
    store: VectorStore,
    queries: list[dict[str, Any]],
    topic_to_relevant: dict[str, list[str]],
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Run retrieval for every labeled query and aggregate metrics."""
    k_list = list(ks)
    recall: dict[int, list[float]] = {k: [] for k in k_list}
    precision: dict[int, list[float]] = {k: [] for k in k_list}
    ndcg: dict[int, list[float]] = {k: [] for k in k_list}
    reciprocal_ranks: list[float] = []
    max_k = max(k_list)

    examples_ok: list[dict[str, Any]] = []
    examples_fail: list[dict[str, Any]] = []

    for query in queries:
        rel_set = _relevant_ids_for_query(query, topic_to_relevant)
        n_rel = len(rel_set)
        if n_rel == 0:
            continue

        hits = store.search(query.get("user_query", ""), k=max_k)
        retrieved_ids = [h["chunk_id"] for h in hits]

        # First relevant rank (1-based) for MRR
        first_rank = None
        for i, cid in enumerate(retrieved_ids, start=1):
            if cid in rel_set:
                first_rank = i
                break
        if first_rank is not None:
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

        top_k_retrieved = {
            k: retrieved_ids[:k] for k in k_list
        }
        for k in k_list:
            tk = set(top_k_retrieved[k])
            n_hit = len(tk & rel_set)
            recall[k].append(n_hit / n_rel)
            precision[k].append(n_hit / k)
            ndcg[k].append(_dcg_at_k(first_rank if first_rank is not None and first_rank <= k else None, k))

        # Success / failure examples: a query is a "success" if top-k contains
        # at least one relevant chunk for the largest k; otherwise a failure.
        matched = any(cid in rel_set for cid in retrieved_ids)
        example = {
            "case_id": query["case_id"],
            "query": query["user_query"],
            "topic": parse_topic_label(query.get("required_evidence_or_fields", "")),
            "relevant_count": n_rel,
            "top1_relevant": retrieved_ids[0] in rel_set if retrieved_ids else False,
            "retrieved_ids": retrieved_ids,
            "scores": [round(h["score"], 4) for h in hits],
        }
        if matched and len(examples_ok) < 4:
            examples_ok.append(example)
        if not matched and len(examples_fail) < 4:
            examples_fail.append(example)

    agg: dict[str, Any] = {"n_queries_evaluated": len(reciprocal_ranks)}
    for k in k_list:
        agg[f"recall@{k}"] = round(float(np.mean(recall[k])), 4)
        agg[f"precision@{k}"] = round(float(np.mean(precision[k])), 4)
        agg[f"ndcg@{k}"] = round(float(np.mean(ndcg[k])), 4)
    agg["mrr"] = round(float(np.mean(reciprocal_ranks)), 4)
    agg["coverage"] = {
        "n_labeled_queries": len(queries),
        "n_evaluated": len(reciprocal_ranks),
        "topic_out_of_corpus": 0,
    }
    return {
        "metrics": agg,
        "successful_examples": examples_ok,
        "failed_examples": examples_fail,
    }


# ---------------------------------------------------------------------------
# M2.1-B: per-query (document-level) benchmark evaluation
# ---------------------------------------------------------------------------
def _doc_of_chunk(chunk_id: str) -> str:
    """Map a chunk id such as ``doc_007__chunk_000`` back to its document id."""
    return chunk_id.split("__chunk")[0]


def _ndcg_at_k(retrieved_ids: list[str], rel_set: set[str], k: int) -> float:
    """nDCG@k with binary gains over doc-level relevance."""
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in rel_set:
            dcg += 1.0 / np.log2(i + 1)
    n_rel = len(rel_set)
    k_eff = min(n_rel, k)
    idcg = sum(1.0 / np.log2(j + 1) for j in range(1, k_eff + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_benchmark_v2(
    store: VectorStore,
    benchmark_queries: list[dict[str, Any]],
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate the document-level benchmark v2.

    Each query dict provides ``user_query``, ``answerable``, ``topic`` and
    ``relevant_document_ids`` (document ids, matched against chunk ids via
    ``_doc_of_chunk``). Unanswerable queries (or queries with no assigned
    relevant documents) are run through retrieval for reporting but are
    excluded from every metric aggregate.
    """
    k_list = list(ks)
    max_k = max(k_list)

    recall: dict[int, list[float]] = {k: [] for k in k_list}
    precision: dict[int, list[float]] = {k: [] for k in k_list}
    ndcg: dict[int, list[float]] = {k: [] for k in k_list}
    reciprocal_ranks: list[float] = []
    zero_relevant = 0
    answerable_scored = 0
    per_query: list[dict[str, Any]] = []

    for q in benchmark_queries:
        rel_set = {str(d) for d in q.get("relevant_document_ids", [])}
        answerable = bool(q.get("answerable", True))
        n_rel = len(rel_set)

        hits = store.search(q["user_query"], k=max_k)
        retrieved_ids = [h["chunk_id"] for h in hits]
        retrieved_doc_ids = [_doc_of_chunk(c) for c in retrieved_ids]

        entry: dict[str, Any] = {
            "query_id": q.get("query_id"),
            "topic": q.get("topic"),
            "answerable": answerable,
            "n_relevant": n_rel,
            "retrieved_doc_ids": retrieved_doc_ids,
            "scores": [round(h["score"], 4) for h in hits],
        }

        if not answerable or n_rel == 0:
            entry["scored"] = False
            per_query.append(entry)
            continue

        entry["scored"] = True
        answerable_scored += 1

        first_rank = None
        for i, cid in enumerate(retrieved_doc_ids, start=1):
            if cid in rel_set:
                first_rank = i
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        entry["first_relevant_rank"] = first_rank

        matched = any(cid in rel_set for cid in retrieved_doc_ids)
        entry["matched"] = matched
        if not matched:
            zero_relevant += 1

        for k in k_list:
            tk = set(retrieved_doc_ids[:k])
            n_hit = len(tk & rel_set)
            recall[k].append(n_hit / n_rel)
            precision[k].append(n_hit / k)
            ndcg_val = _ndcg_at_k(retrieved_doc_ids, rel_set, k)
            ndcg[k].append(ndcg_val)
            entry[f"recall@{k}"] = round(n_hit / n_rel, 4)
            entry[f"precision@{k}"] = round(n_hit / k, 4)
            entry[f"ndcg@{k}"] = round(ndcg_val, 4)
        per_query.append(entry)

    agg: dict[str, Any] = {"n_answerable_evaluated": answerable_scored}
    for k in k_list:
        agg[f"recall@{k}"] = round(float(np.mean(recall[k])), 4) if recall[k] else None
        agg[f"precision@{k}"] = round(float(np.mean(precision[k])), 4) if precision[k] else None
        agg[f"ndcg@{k}"] = round(float(np.mean(ndcg[k])), 4) if ndcg[k] else None
    agg["mrr"] = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else None
    agg["zero_relevant_result_count"] = zero_relevant
    agg["n_unanswerable"] = sum(1 for q in benchmark_queries if not bool(q.get("answerable", True)))

    return {"metrics": agg, "per_query": per_query}
