"""Reusable embedding interface for RAG retrieval (M2.1).

Uses a locally cached ``sentence-transformers/all-MiniLM-L6-v2`` model
(384-dimensional) so no paid API or network download is required. The model is
deterministic for a fixed input, which keeps retrieval reproducible under the
documented configuration.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    """Sentence-embedding wrapper with a stable, small API."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (inferred from a single probe embedding)."""
        if self._dimension is None:
            probe = self._model.encode(["probe"], normalize_embeddings=True, convert_to_numpy=True)
            self._dimension = int(np.asarray(probe).shape[1])
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into a (n, dimension) float32 array of
        L2-normalized embeddings. Deterministic for fixed input/model."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between two (L2-normalized) embedding arrays."""
        return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T
