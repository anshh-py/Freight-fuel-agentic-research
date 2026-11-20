"""Focused tests for M5.2 retrieval-quality and answer-grounding evaluation.

These tests validate the metric helpers, the relevance/abstention labels, and
the deterministic grounding/abstention checks. They use lightweight synthetic
queries and a tiny in-memory store (real corpus indexing is exercised by the
runner, not here).
"""

from __future__ import annotations

import pytest

from src.evaluation.m5_2_retrieval_grounding import (
    evaluate_grounding_and_abstention,
    evaluate_retrieval_metrics,
    grounding_support_check,
    uncertainty_communicated_check,
    validate_query_set,
)
from src.evaluation.m5_2_query_set import ABSTENTION_CATEGORIES, QUERIES


# ---------------------------------------------------------------------------
# Query-set schema / label consistency
# ---------------------------------------------------------------------------
class TestQuerySetValidation:
    def test_all_required_fields(self):
        for q in QUERIES:
            assert q["query_id"]
            assert q["user_query"].strip()
            assert isinstance(q["answerable"], bool)
            assert "category" in q
            assert "topic" in q

    def test_unique_ids(self):
        ids = [q["query_id"] for q in QUERIES]
        assert len(ids) == len(set(ids)), "duplicate query_id"
        assert len(ids) >= 20

    def test_answerable_have_relevant_docs(self):
        for q in QUERIES:
            if q["answerable"]:
                assert q["relevant_document_ids"], f"{q['query_id']} answerable but no relevant docs"
                # relevant docs must be doc_ prefixed
                assert all(str(d).startswith("doc_") for d in q["relevant_document_ids"])

    def test_unanswerable_assign_no_relevant(self):
        for q in QUERIES:
            if not q["answerable"]:
                assert q["relevant_document_ids"] == [], f"{q['query_id']} unanswerable but has relevant docs"
                assert q["abstention_category"] in ABSTENTION_CATEGORIES

    def test_validate_query_set_clean(self):
        assert validate_query_set(QUERIES) == []

    def test_validate_query_set_detects_bad_abstention(self):
        bad = [
            {
                "query_id": "bad1",
                "user_query": "test",
                "answerable": False,
                "topic": "x",
                "category": "abstention",
                "relevant_document_ids": [],
                "abstention_category": "not_a_category",
                "candidate_answer": "",
            }
        ]
        assert validate_query_set(bad) != []


# ---------------------------------------------------------------------------
# Grounding heuristic
# ---------------------------------------------------------------------------
class TestGroundingSupport:
    def test_grounded_claim(self):
        evidence = ["Higher engine load means the engine injects more fuel to produce torque and power."]
        candidate = "Higher engine load means the engine injects more fuel for torque and power."
        check = grounding_support_check(candidate, evidence)
        assert check["n_ungrounded"] == 0
        assert check["grounded_fraction"] == 1.0

    def test_ungrounded_claim_detected(self):
        evidence = ["Aerodynamic drag rises with the square of speed."]
        candidate = "The factory spark plug gap is 1.1 mm for maximum economy."
        check = grounding_support_check(candidate, evidence)
        assert check["n_ungrounded"] >= 1
        assert check["grounded_fraction"] < 1.0

    def test_empty_evidence_all_ungrounded(self):
        check = grounding_support_check("This truck gives 10% better economy by design.", [])
        assert check["n_ungrounded"] == check["n_sentences"]


class TestUncertainty:
    def test_uncertainty_present(self):
        assert uncertainty_communicated_check("roughly 70-80 kph; not a guarantee") is True

    def test_uncertainty_absent(self):
        assert uncertainty_communicated_check("The exact value is 50.0 L/h precisely.") is False


# ---------------------------------------------------------------------------
# Retrieval metrics reuse (tiny synthetic store)
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    model_name = "fake"
    dimension = 4

    def encode(self, texts, **kwargs):
        import numpy as np
        n = len(texts)
        rng = np.random.default_rng(7)
        v = rng.normal(size=(n, self.dimension)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        return v


class _FakeStore:
    def __init__(self):
        self._chunks = [
            {"chunk_id": "doc_001__chunk_000", "document_id": "doc_001", "content": "fuel consumption drivers mass speed grade"},
            {"chunk_id": "doc_009__chunk_000", "document_id": "doc_009", "content": "climbing uphill raises engine load"},
            {"chunk_id": "doc_015__chunk_000", "document_id": "doc_015", "content": "shap attributions not causation"},
        ]

    def search(self, query, k=5, filter_spec=None):
        return [
            {"chunk": c, "chunk_id": c["chunk_id"], "score": 0.5, "rank": i + 1}
            for i, c in enumerate(self._chunks[:k])
        ]


def _queries_small():
    return [
        {
            "query_id": "t1",
            "user_query": "what determines fuel consumption",
            "topic": "fuel_consumption_fundamentals",
            "category": "answerable",
            "answerable": True,
            "relevant_document_ids": ["doc_001"],
            "abstention_category": None,
            "candidate_answer": "Fuel consumption is driven by mass, speed and grade.",
            "grounding": {"supported_by_evidence": True},
            "rationale": "x",
        },
        {
            "query_id": "t2",
            "user_query": "tyre pressure spec",
            "topic": "unanswerable",
            "category": "abstention",
            "answerable": False,
            "relevant_document_ids": [],
            "abstention_category": "unsupported_technical_spec",
            "candidate_answer": "I cannot answer; no spec in the corpus.",
            "grounding": None,
            "rationale": "x",
        },
    ]


class TestRetrievalMetricsReuse:
    def test_metrics_run_over_answerable(self):
        store = _FakeStore()
        res = evaluate_retrieval_metrics(store, _queries_small())
        assert res["n_answerable"] == 1
        assert res["n_unanswerable"] == 1
        for key in ("recall@1", "recall@5", "precision@5", "mrr", "ndcg@5"):
            assert key in res

    def test_grounding_and_abstention(self):
        store = _FakeStore()
        res = evaluate_grounding_and_abstention(store, _queries_small(), ["doc_001", "doc_009", "doc_015"])
        assert res["summary"]["n_answerable_grounded"] == 1
        assert res["summary"]["n_abstention"] == 1
        assert res["abstention"][0]["abstains_no_fabrication"] is True
