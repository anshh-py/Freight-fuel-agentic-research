"""Corpus validation and deterministic document processing for the synthetic
RAG knowledge base (M2.1).

The corpus is ``data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl``.
It is *synthetic*: every document carries ``is_oem_manual=false`` and a
``source_reference`` that states it is a non-OEM educational text. This module
preserves that provenance and never represents it as manufacturer guidance.

Two responsibilities live here:

1. ``validate_corpus`` -- structural validation (IDs, required fields, text
   non-empty, duplicates, malformed records, ambiguous/hedged claims). Returns
   a machine-readable validation report.
2. ``load_corpus`` / ``chunk_documents`` -- load the JSONL, normalize text
   deterministically, and assign chunks. Short documents are deliberately kept
   as a single chunk (no unnecessary fragmentation).
"""

from __future__ import annotations

import json
from typing import Any, Iterator

# Fields every corpus document must contain.
REQUIRED_FIELDS = (
    "document_id",
    "title",
    "topic",
    "content",
    "source_type",
    "source_reference",
    "is_oem_manual",
)

# Words that indicate an approximate / hedged / subjective claim. These are
# flagged as "ambiguous" so downstream users review rather than quote them as
# established fact.
HEDGE_TOKENS = (
    "usually",
    "roughly",
    "tends",
    "can",
    "may",
    "often",
    "generally",
    "approximately",
    "about",
)

# Corpus source path relative to the workspace root.
DEFAULT_CORPUS_PATH = "data/synthetic_heavy_truck_research/rag_knowledge_corpus.jsonl"


def _normalize_text(text: str) -> str:
    """Deterministic normalization: strip outer whitespace and collapse runs of
    whitespace into a single space. Case is preserved (English text)."""
    return " ".join(text.split())


def load_corpus(path: str = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    """Load the JSONL corpus into a list of document dicts. Any line that is
    not a JSON object is reported as malformed by the caller's validation, so
    here we only collect the raw parsed lines (including ``None`` for lines
    that failed to parse)."""
    docs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                docs.append(None)
    return docs


def _uniqueness_report(docs: list[dict[str, Any]]) -> dict[str, Any]:
    ids = []
    for d in docs:
        if isinstance(d, dict) and "document_id" in d:
            ids.append(d["document_id"])
    seen: dict[str, int] = {}
    dupes = []
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    for i, c in seen.items():
        if c > 1:
            dupes.append({"document_id": i, "count": c})
    return {"total_documents": len(ids), "duplicate_ids": dupes}


def _malformed_report(docs: list[dict[str, Any]]) -> dict[str, Any]:
    malformed = []
    for idx, d in enumerate(docs):
        if d is None:
            malformed.append({"line_index": idx, "reason": "invalid_json"})
        elif not isinstance(d, dict):
            malformed.append({"line_index": idx, "reason": "not_an_object"})
    return {"malformed_records": malformed}


def _field_report(docs: list[dict[str, Any]]) -> dict[str, Any]:
    missing = []
    empty_content = []
    for idx, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        for field in REQUIRED_FIELDS:
            if field not in d:
                missing.append({"line_index": idx, "document_id": d.get("document_id"), "field": field})
        content = d.get("content")
        if content is None or not str(content).strip():
            empty_content.append({"line_index": idx, "document_id": d.get("document_id")})
    return {"missing_fields": missing, "empty_content_records": empty_content}


def _claim_report(docs: list[dict[str, Any]]) -> dict[str, Any]:
    ambiguous = []
    for idx, d in enumerate(docs):
        if not isinstance(d, dict):
            continue
        content = str(d.get("content", ""))
        hedges = [t for t in HEDGE_TOKENS if t in content.lower()]
        if hedges:
            ambiguous.append(
                {
                    "document_id": d.get("document_id"),
                    "line_index": idx,
                    "hedge_tokens": hedges,
                    "note": "contains approximating/hedging language; review before quoting as fact",
                }
            )
    return {"ambiguous_claims": ambiguous}


def validate_corpus(path: str = DEFAULT_CORPUS_PATH) -> dict[str, Any]:
    """Run the full corpus validation and return a JSON-serializable report."""
    docs = load_corpus(path)
    report: dict[str, Any] = {
        "corpus_path": path,
        "rows_parsed": len(docs),
        "valid_documents": sum(1 for d in docs if isinstance(d, dict)),
        "synthetic_status": "SYNTHETIC corpus; no document is OEM/manufacturer guidance",
        "oem_manual_count": sum(1 for d in docs if isinstance(d, dict) and d.get("is_oem_manual") is True),
        "source_type_counts": {},
        "topic_counts": {},
    }
    for d in docs:
        if not isinstance(d, dict):
            continue
        st = d.get("source_type", "unknown")
        report["source_type_counts"][st] = report["source_type_counts"].get(st, 0) + 1
        tp = d.get("topic", "unknown")
        report["topic_counts"][tp] = report["topic_counts"].get(tp, 0) + 1
    report["uniqueness"] = _uniqueness_report(docs)
    report["malformed"] = _malformed_report(docs)
    report["fields"] = _field_report(docs)
    report["claims"] = _claim_report(docs)
    report["validation_passed"] = (
        report["malformed"]["malformed_records"] == []
        and report["fields"]["missing_fields"] == []
        and report["fields"]["empty_content_records"] == []
        and report["uniqueness"]["duplicate_ids"] == []
    )
    return report


def chunk_documents(docs: list[dict[str, Any]] | None = None, path: str = DEFAULT_CORPUS_PATH) -> list[dict[str, Any]]:
    """Rec deterministic chunks from the corpus.

    Documents in this corpus are short (1-3 sentences). To honour the
    requirement to avoid unnecessary fragmentation, each document becomes
    exactly one chunk whose ``content`` is the normalized document text. Chunk
    IDs are ``<document_id>__chunk_000`` so they are unique and traceable to
    their source document.
    """
    if docs is None:
        docs = load_corpus(path)
    chunks = []
    for d in docs:
        if not isinstance(d, dict) or "document_id" not in d:
            continue
        doc_id = str(d["document_id"])
        chunk_id = f"{doc_id}__chunk_000"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": d.get("title", ""),
                "topic": d.get("topic", "unknown"),
                "content": _normalize_text(str(d.get("content", ""))),
                "source_type": d.get("source_type", "unknown"),
                "source_reference": d.get("source_reference", ""),
                "is_oem_manual": bool(d.get("is_oem_manual", False)),
            }
        )
    return chunks
