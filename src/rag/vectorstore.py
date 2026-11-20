"""Deterministic vector store for RAG retrieval (M2.1).

The store holds chunks produced by ``corpus.chunk_documents`` and the
L2-normalized embeddings of their content from ``embedder.LocalEmbedder``.
Similarity is cosine (dot product of normalized vectors). Retrieval supports a
configurable top-k and a metadata filter (either a ``{field: value}`` dict or a
callable predicate).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.rag.embedder import LocalEmbedder

MetadataFilter = Callable[[dict[str, Any]], bool] | dict[str, Any] | None


def _make_filter(filter_spec: MetadataFilter) -> Callable[[dict[str, Any]], bool]:
    """Normalize a metadata filter spec into a predicate over a chunk dict."""
    if filter_spec is None:
        return lambda _chunk: True
    if isinstance(filter_spec, dict):
        def _dict_filter(chunk: dict[str, Any]) -> bool:
            return all(chunk.get(k) == v for k, v in filter_spec.items())

        return _dict_filter
    if callable(filter_spec):
        return filter_spec
    raise TypeError("filter_spec must be a dict, a callable, or None")


class VectorStore:
    """Cosine-similarity vector store with top-k and metadata filtering."""

    def __init__(self, embedder: LocalEmbedder, chunks: list[dict[str, Any]] | None = None) -> None:
        self.embedder = embedder
        self.chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None
        if chunks:
            self.add(chunks)

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[dict[str, Any]]) -> None:
        """Append chunks and their embeddings to the store. Deterministic for a
        fixed chunk list and model."""
        if not chunks:
            return
        texts = [str(c["content"]) for c in chunks]
        embeds = self.embedder.encode(texts)  # (n, dimension) float32, L2-normalized
        if self._embeddings is None:
            self._embeddings = embeds
        else:
            self._embeddings = np.concatenate([self._embeddings, embeds], axis=0)
        self.chunks.extend(list(chunks))

    def search(
        self,
        query: str,
        k: int = 5,
        filter_spec: MetadataFilter = None,
    ) -> list[dict[str, Any]]:
        """Return the top-``k`` chunks most similar to ``query``.

        Each result is a dict with keys ``chunk`` (the chunk dict), ``score``
        (cosine similarity), ``rank`` (1-based) and ``chunk_id``. Optional
        ``filter_spec`` restricts candidates before ranking. Rows are ordered by
        descending similarity; ties are broken by chunk order (stable).
        """
        if self.size == 0:
            return []
        q = self.embedder.encode([query])[0]
        scores = self._embeddings @ q  # cosine, normalized vectors
        predicate = _make_filter(filter_spec)
        mask = np.zeros(self.size, dtype=bool)
        for i, chunk in enumerate(self.chunks):
            mask[i] = predicate(chunk)
        if not mask.any():
            return []
        indices = np.flatnonzero(mask)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        order = order[: int(k)]
        return [
            {
                "chunk": self.chunks[int(i)],
                "chunk_id": self.chunks[int(i)]["chunk_id"],
                "score": float(scores[int(i)]),
                "rank": idx + 1,
            }
            for idx, i in enumerate(order)
        ]

    def to_state(self) -> dict[str, Any]:
        """Serialize the store to a reproducible state (for reproducibility / docs)."""
        return {
            "dimension": self.dimension,
            "n_chunks": self.size,
            "model_name": self.embedder.model_name,
        }
