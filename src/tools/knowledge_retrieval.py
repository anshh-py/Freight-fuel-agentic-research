"""Tool 2 — Knowledge retrieval over the synthetic RAG corpus (M2.2).

Wraps the existing M2.1 retrieval implementation (``src.rag.vectorstore.VectorStore``
+ ``src.rag.corpus`` + ``src.rag.embedder.LocalEmbedder``) behind an explicit
callable interface. It preserves corpus provenance (synthetic educational
material, not OEM manuals) and returns ranked passages with document IDs,
source references, topic metadata, and similarity scores.

Retrieval similarity is *not* proof of factual correctness; the tool flags when
the retrieved evidence is insufficient and never fabricates content absent from
the corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.rag.corpus import DEFAULT_CORPUS_PATH, chunk_documents, load_corpus
from src.rag.embedder import DEFAULT_MODEL_NAME, LocalEmbedder
from src.rag.vectorstore import VectorStore
from src.tools.validation import ToolInputError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TOPK = 5
# Heuristic similarity floor below which the top hit is treated as weak
# evidence. Cosine similarity on L2-normalized MiniLM embeddings; typical
# relevant hits in the M2.1 benchmark sit above ~0.55, weak/non-relevant below
# ~0.45. This is a documented heuristic, not a correctness guarantee.
DEFAULT_MIN_SCORE = 0.45


def build_store(
    corpus_path: str = DEFAULT_CORPUS_PATH,
    embedder: LocalEmbedder | None = None,
) -> VectorStore:
    """Build the deterministic vector store for the corpus."""
    if embedder is None:
        embedder = LocalEmbedder(DEFAULT_MODEL_NAME)
    chunks = chunk_documents(load_corpus(corpus_path))
    return VectorStore(embedder, chunks)


class KnowledgeRetrievalTool:
    """Ranked passage retrieval over the synthetic knowledge corpus."""

    CORPUS_NOTICE = (
        "All retrieved passages are SYNTHETIC educational material, not OEM/manufacturer "
        "documentation. Retrieval similarity is not proof of factual correctness."
    )

    def __init__(
        self,
        corpus_path: str = DEFAULT_CORPUS_PATH,
        embedder: LocalEmbedder | None = None,
        store: VectorStore | None = None,
        top_k: int = DEFAULT_TOPK,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.corpus_path = corpus_path
        self.top_k = top_k
        self.min_score = DEFAULT_MIN_SCORE if min_score is None else min_score
        self._store = store
        self._embedder = embedder

    def _ensure_store(self) -> VectorStore:
        if self._store is None:
            self._store = build_store(self.corpus_path, self._embedder)
        return self._store

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Return ranked passages for a natural-language query."""
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query must be a non-empty string")
        q = query.strip()
        k = k or self.top_k
        floor = self.min_score if min_score is None else min_score

        store = self._ensure_store()
        hits = store.search(q, k=k)

        passages = []
        for h in hits:
            chunk = h["chunk"]
            passages.append(
                {
                    "rank": h["rank"],
                    "chunk_id": h["chunk_id"],
                    "document_id": chunk.get("document_id"),
                    "title": chunk.get("title"),
                    "topic": chunk.get("topic"),
                    "source_type": chunk.get("source_type"),
                    "source_reference": chunk.get("source_reference"),
                    "is_oem_manual": chunk.get("is_oem_manual", False),
                    "score": round(float(h["score"]), 4),
                    "content_excerpt": str(chunk.get("content", ""))[:400],
                }
            )

        top_score = passages[0]["score"] if passages else 0.0
        insufficient = (not passages) or (top_score < floor)

        return {
            "status": "ok",
            "query": q,
            "n_results": len(passages),
            "passages": passages,
            "evidence_sufficiency": {
                "insufficient": insufficient,
                "reason": (
                    "no passage retrieved" if not passages
                    else f"top similarity {top_score:.4f} < floor {floor:.4f}"
                    if insufficient else "top passage meets similarity floor"
                ),
                "min_score_floor": floor,
            },
            "provenance": {
                "corpus_synthetic": True,
                "corpus_path": self.corpus_path,
                "embedding_model": store.embedder.model_name,
                "note": self.CORPUS_NOTICE,
                "similarity_not_proof_of_correctness": True,
            },
        }
