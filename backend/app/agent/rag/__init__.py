"""Lightweight Hybrid RAG Package for SpecSage.

Exports:
    - EvidenceDocument: Canonical, provenance-bearing evidence unit.
    - ScoredEvidence: Ranked evidence result.
    - ProductEvidenceSummary: Product-isolated evidence package for LLM generation.
    - EvidenceIndex: In-memory catalogue indexer.
    - BM25Retriever: Lexical exact-term retrieval engine.
    - VectorRetriever: Dense semantic vector retrieval engine.
    - EmbeddingProvider, LocalHashEmbeddingProvider, FakeEmbeddingProvider.
    - HybridEvidenceRetriever, get_hybrid_retriever: Unified RRF retrieval orchestrator.
"""

from __future__ import annotations

from app.agent.rag.bm25 import BM25Retriever
from app.agent.rag.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from app.agent.rag.hybrid import (
    HybridEvidenceRetriever,
    get_hybrid_retriever,
)
from app.agent.rag.indexer import (
    EvidenceIndex,
    build_product_evidence_documents,
)
from app.agent.rag.models import (
    EvidenceDocument,
    ProductEvidenceSummary,
    ScoredEvidence,
)
from app.agent.rag.vector import VectorRetriever

__all__ = [
    "EvidenceDocument",
    "ScoredEvidence",
    "ProductEvidenceSummary",
    "EvidenceIndex",
    "build_product_evidence_documents",
    "BM25Retriever",
    "VectorRetriever",
    "EmbeddingProvider",
    "LocalHashEmbeddingProvider",
    "FakeEmbeddingProvider",
    "HybridEvidenceRetriever",
    "get_hybrid_retriever",
]
