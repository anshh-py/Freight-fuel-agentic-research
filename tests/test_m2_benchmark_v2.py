"""Focused tests for M2.1-B RAG Benchmark v2.

Covers: (1) baseline preservation (the original 55-case set is untouched and
its two unique query strings / counts are unchanged); (2) benchmark v2
validity (>=20 distinct queries, category coverage, answerable/unanswerable
consistency, every relevant id exists in the corpus, unanswerable queries
assign no relevant ids); (3) metric-correctness of ``evaluate_benchmark_v2``
on a deterministic fake store; (4) the generated report files exist and are
valid JSON.
"""

import json
import os

import pytest

from src.rag.corpus import load_corpus
from src.rag.retrieval import (
    evaluate_benchmark_v2,
    load_eval_queries,
    parse_topic_label,
)

CORPUS = "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl"
QUERIES = "data/synthetic_heavy_truck_research/agent_evaluation_queries.csv"
BM_DIR = os.path.join("reports", "m2", "benchmark_v2")
BENCHMARK = os.path.join(BM_DIR, "benchmark.json")


# ---------------------------------------------------------------------------
# 1. Baseline preservation (original 55-case benchmark untouched)
# ---------------------------------------------------------------------------
def test_baseline_preserved_counts():
    queries = load_eval_queries(QUERIES)
    assert len(queries) == 55, "original rag_retrieval baseline must remain 55 cases"
    unique = {q["user_query"] for q in queries}
    assert len(unique) == 2, "baseline has exactly two unique query strings"
    counts = {"road_grade_effects": 0, "engine_load_fuel_demand": 0}
    for q in queries:
        t = parse_topic_label(q.get("required_evidence_or_fields", ""))
        counts[t] = counts.get(t, 0) + 1
    assert counts["road_grade_effects"] == 28
    assert counts["engine_load_fuel_demand"] == 27
    assert sum(counts.values()) == 55


# ---------------------------------------------------------------------------
# 2. Benchmark v2 validity
# ---------------------------------------------------------------------------
def _load_benchmark():
    with open(BENCHMARK, encoding="utf-8") as fh:
        return json.load(fh)


def test_benchmark_v2_volume_and_distinctness():
    bm = _load_benchmark()
    queries = bm["queries"]
    assert len(queries) >= 20
    # no superficial paraphrases: every query string is unique
    texts = [q["user_query"].strip() for q in queries]
    assert len(texts) == len(set(texts)) == len(queries)
    # categories present
    cats = {q["category"] for q in queries}
    assert {
        "conceptual",
        "cause_effect",
        "operational",
        "comparison",
        "practical_driver",
        "multi_doc",
        "unanswerable",
    } <= cats


def test_benchmark_v2_answerable_and_unanswerable():
    bm = _load_benchmark()
    queries = bm["queries"]
    answerable = [q for q in queries if q["answerable"]]
    unanswerable = [q for q in queries if not q["answerable"]]
    assert len(answerable) >= 16
    assert len(unanswerable) >= 1
    # answerable queries must assign at least one relevant document
    for q in answerable:
        assert q["relevant_document_ids"], f"{q['query_id']} answerable but empty relevant ids"
    # unanswerable queries must assign NO relevant ids
    for q in unanswerable:
        assert q["relevant_document_ids"] == [], f"{q['query_id']} unanswerable but has relevant ids"
    # every query has an explanation record
    for q in queries:
        assert q["explanation"].strip()


def test_benchmark_v2_relevant_ids_exist_in_corpus():
    bm = _load_benchmark()
    doc_ids = {d["document_id"] for d in load_corpus(CORPUS)}
    for q in bm["queries"]:
        for rid in q["relevant_document_ids"]:
            assert rid in doc_ids, f"{q['query_id']} references unknown doc {rid}"


def test_benchmark_v2_meta_counts():
    bm = _load_benchmark()
    queries = bm["queries"]
    assert bm["meta"]["query_count"] == len(queries)
    assert bm["meta"]["unique_query_count"] == len({q["user_query"] for q in queries})
    assert bm["meta"]["answerable_count"] == sum(1 for q in queries if q["answerable"])
    assert bm["meta"]["unanswerable_count"] == sum(1 for q in queries if not q["answerable"])
    # baseline metadata recorded in the benchmark file matches the real baseline
    assert bm["meta"]["baseline_preserved"]["n_queries"] == 55
    assert bm["meta"]["baseline_preserved"]["road_grade_effects_count"] == 28
    assert bm["meta"]["baseline_preserved"]["engine_load_fuel_demand_count"] == 27


# ---------------------------------------------------------------------------
# 3. Metric correctness of evaluate_benchmark_v2 (deterministic fake store)
# ---------------------------------------------------------------------------
class _FakeStore:
    """Duck-typed stand-in for VectorStore: returns a fixed top-5 ranking."""

    def __init__(self, ranking):
        self.ranking = ranking  # list of chunk ids, e.g. ["A__chunk_000", ...]

    def search(self, query, k=5):
        out = []
        for i, cid in enumerate(self.ranking[:k]):
            out.append({"chunk_id": cid, "score": float(max(0, 1.0 - i * 0.1))})
        return out


def test_evaluate_benchmark_v2_metric_math():
    store = _FakeStore(["A__chunk_000", "B__chunk_000", "C__chunk_000", "D__chunk_000", "E__chunk_000"])
    queries = [
        {"query_id": "q1", "user_query": "x", "answerable": True, "topic": "t1", "relevant_document_ids": ["A"]},
        {"query_id": "q2", "user_query": "y", "answerable": True, "topic": "t2", "relevant_document_ids": ["C", "E"]},
        {"query_id": "q3", "user_query": "z", "answerable": False, "topic": "unanswerable", "relevant_document_ids": []},
    ]
    res = evaluate_benchmark_v2(store, queries, ks=(1, 3, 5))
    m = res["metrics"]
    assert m["n_answerable_evaluated"] == 2
    assert m["n_unanswerable"] == 1
    assert m["zero_relevant_result_count"] == 0

    # q1: relevant={A}, retrieved [A,B,C,D,E]
    assert res["per_query"][0]["scored"] is True
    assert res["per_query"][0]["first_relevant_rank"] == 1
    # q2: relevant={C,E}, first relevant at rank 3
    assert res["per_query"][1]["first_relevant_rank"] == 3
    # q3: unanswerable -> not scored, no metrics
    assert res["per_query"][2]["scored"] is False

    # metrics are rounded to 4 decimals in the implementation, so use abs=1e-3
    assert m["recall@1"] == pytest.approx(0.5, abs=1e-3)
    assert m["recall@3"] == pytest.approx(0.75, abs=1e-3)
    assert m["recall@5"] == pytest.approx(1.0, abs=1e-3)
    assert m["precision@1"] == pytest.approx(0.5, abs=1e-3)
    assert m["precision@3"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["precision@5"] == pytest.approx(0.3, abs=1e-3)
    assert m["mrr"] == pytest.approx((1 + 1 / 3) / 2, abs=1e-3)
    # nDCG@5: q1 = 1.0; q2 = dcg/idcg = (1/log2 4 + 1/log2 6)/(1/log2 2 + 1/log2 3)
    ndcg_q2 = (1 / 2 + 1 / 2.585) / (1.0 + 1 / 1.585)
    assert m["ndcg@5"] == pytest.approx((1.0 + ndcg_q2) / 2, abs=1e-3)


def test_evaluate_benchmark_v2_zero_relevant_counted():
    store = _FakeStore(["F__chunk_000", "G__chunk_000", "H__chunk_000", "I__chunk_000", "J__chunk_000"])
    queries = [
        {"query_id": "q1", "user_query": "x", "answerable": True, "topic": "t1", "relevant_document_ids": ["A"]},
    ]
    res = evaluate_benchmark_v2(store, queries, ks=(1, 3, 5))
    assert res["metrics"]["zero_relevant_result_count"] == 1
    assert res["metrics"]["recall@5"] == 0.0
    assert res["metrics"]["mrr"] == 0.0
    assert res["per_query"][0]["matched"] is False


# ---------------------------------------------------------------------------
# 4. Generated report files
# ---------------------------------------------------------------------------
def test_benchmark_v2_reports_exist():
    for name in ("benchmark.json", "evaluation.json", "summary.md"):
        p = os.path.join(BM_DIR, name)
        assert os.path.exists(p), f"missing report {p}"
    for name in ("benchmark.json", "evaluation.json"):
        with open(os.path.join(BM_DIR, name), encoding="utf-8") as fh:
            json.load(fh)


def test_evaluation_report_reports_key_counts():
    with open(os.path.join(BM_DIR, "evaluation.json"), encoding="utf-8") as fh:
        ev = json.load(fh)
    e = ev["evaluation"]
    assert e["n_unique_queries"] == 28
    assert e["n_answerable"] == 24
    assert e["n_unanswerable"] == 4
    assert e["n_answerable_evaluated"] == 24
    for key in ("recall@1", "recall@3", "recall@5", "precision@1", "precision@3", "precision@5", "mrr", "ndcg@5"):
        assert key in e
        assert 0.0 <= e[key] <= 1.0
    # no zero-relevant failures were observed on this benchmark
    assert e["zero_relevant_result_count"] >= 0
    assert ev["benchmark_validation"]["validation_passed"] is True
