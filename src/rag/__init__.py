"""RAG retrieval foundation for the Cross-Truck Fuel-Rate Prediction and
Explainable Agentic Driver-Feedback System (M2.1).

This package is retrieval engineering only. It does not implement agents,
orchestration, or any conversational behaviour. It offers corpus validation,
deterministic document normalization/chunking, a reusable local embedding
interface, vector retrieval with metadata filtering, and retrieval
evaluation against the ground-truth topic labels provided in the evaluation
query set.
"""

from __future__ import annotations
