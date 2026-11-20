"""Shared session-scoped fixtures for M2.2 agent tests.

The sentence-embedding model is loaded once per session and shared through a
single read-only ``VectorStore`` so retrieval integration tests do not reload
the model for every test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def retrieval_store():
    # Imported lazily so tests that do not touch retrieval never pay for the
    # heavy sentence-transformers / datasets import graph at collection time.
    from src.tools.knowledge_retrieval import build_store

    return build_store()
