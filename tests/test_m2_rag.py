"""Focused tests for M2.1 RAG retrieval: corpus validation, chunking, vector
store (metadata filters, determinism), ground-truth validation and metrics."""

import json
import os

import pytest

from src.rag.corpus import chunk_documents, load_corpus, validate_corpus
from src.rag.embedder import LocalEmbedder
from src.rag.retrieval import (
    evaluate_retrieval,
    load_eval_queries,
    parse_topic_label,
    query_relevant_topics,
    validate_ground_truth,
)
from src.rag.vectorstore import VectorStore

CORPUS = "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl"
QUERIES = "data/synthetic_heavy_truck_research/agent_evaluation_queries.csv"


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------
def test_corpus_validation_report_shapes():
    rep = validate_corpus(CORPUS)
    assert rep["validation_passed"] is True
    assert rep["valid_documents"] == 21
    assert rep["oem_manual_count"] == 0
    assert rep["malformed"]["malformed_records"] == []
    assert rep["fields"]["missing_fields"] == []
    assert rep["uniqueness"]["duplicate_ids"] == []
    # synthetic status preserved
    assert "SYNTHETIC" in rep["synthetic_status"]


def test_load_corpus_and_required_fields():
    docs = load_corpus(CORPUS)
    assert len(docs) == 21
    for d in docs:
        for f in ("document_id", "title", "topic", "content", "source_type", "source_reference", "is_oem_manual"):
            assert f in d


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def test_chunks_unique_and_traceable():
    chunks = chunk_documents()
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for c in chunks:
        assert c["chunk_id"] == f"{c['document_id']}__chunk_000"
        assert c["source_reference"], "source reference must be preserved"
        assert c["content"], "chunk content must be non-empty"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def test_vectorstore_dimensions_and_size():
    embedder = LocalEmbedder()
    chunks = chunk_documents()
    store = VectorStore(embedder)
    store.add(chunks)
    assert store.size == len(chunks)
    assert store.dimension == embedder.dimension == 384
    assert store._embeddings.shape == (len(chunks), 384)


def test_metadata_filter_works():
    chunks = chunk_documents()
    store = VectorStore(LocalEmbedder(), chunks)
    # filter to road-grade docs only
    hits = store.search("how does road grade affect fuel consumption", k=10, filter_spec={"topic": "road_grade_effects"})
    assert hits, "expected hits"
    for h in hits:
        assert h["chunk"]["topic"] == "road_grade_effects"


def test_retrieval_deterministic():
    chunks = chunk_documents()
    s1 = VectorStore(LocalEmbedder(), chunks)
    s2 = VectorStore(LocalEmbedder(), chunks)
    r1 = s1.search("what does engine load have to do with fuel demand", k=5)
    r2 = s2.search("what does engine load have to do with fuel demand", k=5)
    assert [h["chunk_id"] for h in r1] == [h["chunk_id"] for h in r2]
    assert [h["score"] for h in r1] == pytest.approx([h["score"] for h in r2], abs=1e-6)


def test_search_ranks_and_scores():
    chunks = chunk_documents()
    store = VectorStore(LocalEmbedder(), chunks)
    hits = store.search("road grade and climbing", k=3)
    assert len(hits) == 3
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(len(hits) - 1))


# ---------------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------------
def test_ground_truth_topics_resolve():
    chunks = chunk_documents()
    queries = load_eval_queries(QUERIES)
    assert len(queries) == 55
    gt = validate_ground_truth(chunks, queries)
    assert gt["all_topics_resolved"] is True
    assert set(gt["labeled_topics"]) == {"road_grade_effects", "engine_load_fuel_demand"}
    assert gt["labeled_topics"]["road_grade_effects"] == 28
    assert gt["labeled_topics"]["engine_load_fuel_demand"] == 27
    # each topic maps to at least one document
    for ids in gt["relevant_documents_per_topic"].values():
        assert ids


def test_parse_topic_label():
    assert parse_topic_label("document_id, topic=road_grade_effects") == "road_grade_effects"
    assert parse_topic_label("") is None


def test_query_relevant_topics_counts():
    queries = load_eval_queries(QUERIES)
    topics = query_relevant_topics(queries)
    assert sum(len(v) for v in topics.values()) == 55


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------
def test_evaluation_metrics_produced():
    chunks = chunk_documents()
    store = VectorStore(LocalEmbedder(), chunks)
    queries = load_eval_queries(QUERIES)
    gt = validate_ground_truth(chunks, queries)
    res = evaluate_retrieval(store, queries, gt["relevant_documents_per_topic"], ks=(1, 3, 5))
    m = res["metrics"]
    assert m["n_queries_evaluated"] == 55
    for key in ("recall@1", "recall@3", "recall@5", "precision@1", "precision@3", "precision@5", "mrr", "ndcg@5"):
        assert key in m
        assert 0.0 <= m[key] <= 1.0


def test_reports_write():
    """After running the pipeline, the two JSON reports exist and are valid JSON."""
    for name in ("rag_corpus_validation.json", "retrieval_evaluation.json"):
        p = os.path.join("reports", "m2", name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                json.load(fh)
