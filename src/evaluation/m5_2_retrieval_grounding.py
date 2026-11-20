"""M5.2 — retrieval quality and answer-grounding evaluation (deterministic).

This module evaluates the synthetic RAG stack on two SEPARATE axes (never
combined into a single score):

1. Retrieval quality — Recall@k / Precision@k / MRR / nDCG@k over the labeled,
   document-level query set. The metric machinery is reused from the existing
   M2.1-B evaluator (``src.rag.retrieval.evaluate_benchmark_v2``); this module
   does not reimplement rank metrics. Answerable and unanswerable queries are
   reported separately, and only answerable queries enter the metric
   aggregates.

2. Answer grounding & abstention — deterministic, rule-based checks over the
   candidate answer and the retrieved evidence:
   * claim support: is each substantive claim present in retrieved evidence,
   * source attribution: are cited doc ids drawn from the retrieved set,
   * OEM mislabeling: is synthetic material ever presented as OEM/manufacturer
     guidance,
   * uncertainty communication: is uncertainty conveyed where estimates are
     qualitative/hedged,
   * abstention: unanswerable queries decline rather than fabricate concrete
     specifications or guaranteed savings.

Labels from ``src.evaluation.m5_2_query_set`` are hand-authored for the
synthetic research environment (documented methodology; NOT an independent
external human review). The evaluator reports both the manual label and the
outcome of the deterministic check, and records agreement/disagreement
transparently.

Design guarantees:
* Deterministic: fixed corpus, fixed embedder, fixed query set, fixed checks.
  No randomness, no LLM, no external API, no web access.
* The frozen M1/M2/M3/M4 implementation files are never modified; this module
  only imports their public, read-only interfaces.
* Retrieval and grounding numbers are kept in separate blocks in the report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.rag.corpus import chunk_documents, validate_corpus
from src.rag.embedder import LocalEmbedder
from src.rag.retrieval import evaluate_benchmark_v2
from src.rag.vectorstore import VectorStore
from src.agent.response_validation import (
    check_oem_mislabeling,
    check_source_attribution,
    check_unsupported_not_answered,
)

from src.evaluation.m5_2_query_set import (
    ABSTENTION_CATEGORIES,
    QUERIES,
    UNCERTAINTY_REQUIRED_BY_ID,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVE = 5

# Uncertainty / hedging phrases that indicate the answer communicates that its
# numbers or estimates are approximate rather than exact guarantees.
_UNCERTAINTY_RE = re.compile(
    r"\b(?:roughly|about|approximately|generally|usually|tends to|can|may|often|"
    r"not established|not guaranteed|not a guarantee|no guarantee|unclear|"
    r"not known|depends on|in general)\b",
    re.I,
)

# Sentences that are provenance/disclaimer metadata rather than substantive claims.
# Mirror of the protective-phrase stripping used in response_validation.py: these
# are anti-claim statements (synthetic-material notices, non-causality warnings)
# and must not be scored as ungrounded claims.
_PROVENANCE_SENTENCE_RE = re.compile(
    r"^.*(?:synthetic educational material|not an? (?:oem|manufacturer|factory)|"
    r"not (?:oem|manufacturer|factory) (?:guidance|documentation|manual)|"
    r"based on synthetic|this follows from synthetic|this is from synthetic|"
    r"not a (?:causal proof|proof of physical causation|guaranteed saving|guarantee)|"
    r"model association, not|cannot[ ]?t (?:answer|provide|validate|guarantee)|"
    r"no sufficient (?:answer|evidence)|"
    r"beyond what the synthetic|"
    r"not established|not guaranteed|no guarantee).*$",
    re.I,
)

# Span tokens that identify a concrete claim to check for evidence support.
_STOP = set(
    "a an the and or but if then than so for of in on at by with from to is are was were be been "
    "this that these those it its as not no will would should can could may might must very more most "
    "when while about after before over under between do does did have has had i you we they he she "
    "my your our their his her me us him them what which who whom how why where i'm you're it's "
    "don't doesn't i'll you'll that's there's here's cannot can't won't wouldn't couldn't shouldn't "
    "wouldn't doesn't isn't aren't wasn't weren't haven't hasn't hadn't i've you've we've they've "
    "one two three per from about".split()
)


def _sentences(text: str) -> list[str]:
    """Split into sentences (deterministic, punctuation-based)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _key_terms(sentence: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9'-]*", sentence.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def _claim_is_grounded(sentence: str, evidence_texts: list[str]) -> bool:
    """A claim is 'grounded' when most of its key terms appear in the evidence.

    This is a documented heuristic (token overlap), not a semantic entailment
    check. It is reported as a heuristic and compared against the manual label.
    """
    terms = _key_terms(sentence)
    if not terms:
        return True  # no substantive claim to verify
    pooled = " ".join(evidence_texts).lower()
    matched = sum(1 for t in terms if t in pooled)
    return matched / len(terms) >= 0.6


def grounding_support_check(candidate: str, evidence_texts: list[str]) -> dict[str, Any]:
    """Return per-sentence grounded/ungrounded split for the candidate answer.

    Provenance/disclaimer sentences (synthetic-material notices, non-causality
    warnings, abstention statements) are not claims and are excluded from the
    ungrounded count; they are tallied separately as ``n_provenance``.
    """
    sentences = _sentences(candidate)
    claims = [s for s in sentences if not _PROVENANCE_SENTENCE_RE.match(s)]
    provenance = [s for s in sentences if _PROVENANCE_SENTENCE_RE.match(s)]
    grounded = [s for s in claims if _claim_is_grounded(s, evidence_texts)]
    ungrounded = [s for s in claims if not _claim_is_grounded(s, evidence_texts)]
    return {
        "n_sentences": len(sentences),
        "n_claims": len(claims),
        "n_provenance": len(provenance),
        "n_grounded": len(grounded),
        "n_ungrounded": len(ungrounded),
        "grounded_fraction": (len(grounded) / len(claims)) if claims else 1.0,
        "ungrounded_sentences": ungrounded,
    }


def uncertainty_communicated_check(text: str) -> bool:
    """True when the answer uses hedged / uncertain phrasing for estimates."""
    return bool(_UNCERTAINTY_RE.search(text or ""))


def evaluate_retrieval_metrics(store: VectorStore, queries: list[dict[str, Any]], ks=(1, 3, 5)) -> dict[str, Any]:
    """Run document-level retrieval metrics (reusing M2.1-B machinery).

    Only answerable queries carry relevance labels and enter the metric
    aggregates; unanswerable queries are counted and reported separately.
    """
    result = evaluate_benchmark_v2(store, queries, ks=ks)
    metrics = result["metrics"]
    per_query = result["per_query"]

    answerable = [q for q in per_query if q.get("scored", False)]
    unanswerable = [q for q in per_query if not q.get("scored", False)]
    return {
        "n_query_total": len(queries),
        "n_answerable": metrics.get("n_answerable_evaluated", 0),
        "n_unanswerable": len(unanswerable),
        "recall@1": metrics.get("recall@1"),
        "recall@3": metrics.get("recall@3"),
        "recall@5": metrics.get("recall@5"),
        "precision@1": metrics.get("precision@1"),
        "precision@3": metrics.get("precision@3"),
        "precision@5": metrics.get("precision@5"),
        "ndcg@1": metrics.get("ndcg@1"),
        "ndcg@3": metrics.get("ndcg@3"),
        "ndcg@5": metrics.get("ndcg@5"),
        "mrr": metrics.get("mrr"),
        "zero_relevant": metrics.get("zero_relevant_result_count", 0),
        "per_query": per_query,
    }


def evaluate_grounding_and_abstention(
    store: VectorStore,
    queries: list[dict[str, Any]],
    known_doc_ids: list[str],
) -> dict[str, Any]:
    """Run grounding checks for answerable queries and abstention checks for
    unanswerable queries over the actually-retrieved evidence."""
    by_id = {q["query_id"]: q for q in queries}
    results: list[dict[str, Any]] = []
    abstention_rows: list[dict[str, Any]] = []
    for query in queries:
        qid = query["query_id"]
        hits = store.search(query["user_query"], k=TOP_K_RETRIEVE)
        evidence_chunks = [h["chunk"] for h in hits]
        evidence_texts = [str(c["content"]) for c in evidence_chunks]
        retrieved_ids = [str(c["chunk_id"]) for c in evidence_chunks]

        row: dict[str, Any] = {"query_id": qid, "user_query": query["user_query"]}
        if query.get("answerable"):
            # --- grounding axis (answerable only) ---
            candidate = query["candidate_answer"]
            support = grounding_support_check(candidate, evidence_texts)
            oem = check_oem_mislabeling(candidate)
            attrib = check_source_attribution(candidate, [d.split("__chunk")[0] for d in retrieved_ids])
            unc = uncertainty_communicated_check(candidate)

            uncertainty_required = bool(UNCERTAINTY_REQUIRED_BY_ID.get(qid, False))
            row.update(
                {
                    "answerable": True,
                    "manual_label": query.get("grounding"),
                    "uncertainty_required": uncertainty_required,
                    "grounding_check": {
                        "claim_support": support,
                        "oem_mislabeling_ok": oem.passed,
                        "source_attribution_ok": attrib.passed,
                        "uncertainty_communicated": unc,
                    },
                    "retrieved_doc_ids": sorted({d.split("__chunk")[0] for d in retrieved_ids}),
                }
            )
            results.append(row)
        else:
            # --- abstention axis (unanswerable only) ---
            category = query.get("abstention_category")
            candidate = query.get("candidate_answer", "")
            minimal = check_unsupported_not_answered(candidate, minimal=True)
            abstention_rows.append(
                {
                    "query_id": qid,
                    "user_query": query["user_query"],
                    "abstention_category": category,
                    "category_valid": category in ABSTENTION_CATEGORIES,
                    "abstains_no_fabrication": minimal.passed,
                    "retrieved_doc_ids": sorted({d.split("__chunk")[0] for d in retrieved_ids}),
                }
            )

    summary = {
        "n_answerable_grounded": len(results),
        "n_abstention": len(abstention_rows),
        "n_abstention_correct": sum(1 for r in abstention_rows if r["abstains_no_fabrication"]),
        "n_abstention_invalid_category": sum(1 for r in abstention_rows if not r["category_valid"]),
        "n_answers_fully_grounded": sum(
            1 for r in results if r["grounding_check"]["claim_support"]["n_ungrounded"] == 0
        ),
        "n_claims_total": sum(r["grounding_check"]["claim_support"]["n_claims"] for r in results),
        "n_claims_ungrounded": sum(r["grounding_check"]["claim_support"]["n_ungrounded"] for r in results),
        "n_answers_oem_ok": sum(1 for r in results if r["grounding_check"]["oem_mislabeling_ok"]),
        "n_answers_attribution_ok": sum(1 for r in results if r["grounding_check"]["source_attribution_ok"]),
        # Corrected uncertainty reporting: compliance is computed ONLY over cases
        # where uncertainty communication is required. The manual grounding label
        # (uncertainty_communicated) is authoritative for "communicated".
        "uncertainty": _uncertainty_summary(results),
    }
    return {"grounding": results, "abstention": abstention_rows, "summary": summary}


def _uncertainty_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the corrected uncertainty-communication breakdown.

    ``required``       answer must hedge (estimate / causal-claim / non-guaranteed
                       recommendation / beyond-evidence generalization).
    ``communicated``   the manual grounding label says uncertainty is conveyed.

    The compliance rate is the share of REQUIRED cases that communicate
    uncertainty. Answers where communication is not required are counted
    separately and deliberately excluded from the denominator.
    """
    n_required = 0
    n_required_communicated = 0
    n_not_required = 0
    missing: list[str] = []
    for r in results:
        required = bool(r.get("uncertainty_required"))
        manual = (r.get("manual_label") or {}).get("uncertainty_communicated", False)
        if required:
            n_required += 1
            if manual:
                n_required_communicated += 1
            else:
                missing.append(r["query_id"])
        else:
            n_not_required += 1
    n_required_missing = n_required - n_required_communicated
    compliance = (
        round(n_required_communicated / n_required, 4)
        if n_required
        else None
    )
    return {
        "n_uncertainty_required": n_required,
        "n_uncertainty_required_communicated": n_required_communicated,
        "n_uncertainty_required_missing": n_required_missing,
        "n_uncertainty_not_required": n_not_required,
        "compliance_rate": compliance,
        "required_missing_query_ids": missing,
    }


def validate_query_set(queries: list[dict[str, Any]] | None = None) -> list[str]:
    """Schema / label consistency checks for the M5.2 query set."""
    queries = queries if queries is not None else QUERIES
    errors: list[str] = []
    seen: set[str] = set()
    for q in queries:
        qid = q.get("query_id")
        if qid in seen:
            errors.append(f"duplicate query_id: {qid}")
        seen.add(qid)
        if not q.get("user_query", "").strip():
            errors.append(f"{qid}: empty user_query")
        answerable = q.get("answerable")
        if not isinstance(answerable, bool):
            errors.append(f"{qid}: answerable must be bool")
        rel = q.get("relevant_document_ids", [])
        if answerable and not isinstance(rel, list):
            errors.append(f"{qid}: relevant_document_ids must be a list")
        if not answerable:
            if q.get("abstention_category") not in ABSTENTION_CATEGORIES:
                errors.append(f"{qid}: invalid/absent abstention_category {q.get('abstention_category')!r}")
            if rel:
                errors.append(f"{qid}: unanswerable query must not assign relevant ids")
    return errors
