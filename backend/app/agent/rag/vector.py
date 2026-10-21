"""Dense Vector Similarity Retriever for SpecSage Hybrid RAG.

Indexes evidence documents in vector space and performs cosine similarity search.
"""

from __future__ import annotations

from typing import Sequence

from app.agent.rag.embeddings import EmbeddingProvider, LocalHashEmbeddingProvider
from app.agent.rag.models import EvidenceDocument, ScoredEvidence
from app.models.product import Category


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute dot product of two L2-normalized dense vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    return sum(a * b for a, b in zip(vec_a, vec_b))


class VectorRetriever:
    """Vector similarity retrieval engine operating on EvidenceDocuments."""

    def __init__(
        self,
        documents: Sequence[EvidenceDocument] | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self._documents: list[EvidenceDocument] = []
        self._vectors: list[list[float]] = []

        if documents:
            self.build_index(documents)

    def build_index(self, documents: Sequence[EvidenceDocument]) -> None:
        """Precompute and store embedding vectors for all documents."""
        self._documents = list(documents)
        texts = [doc.content for doc in self._documents]
        self._vectors = self.embedding_provider.embed_batch(texts)

    def search(
        self,
        query: str,
        top_k: int = 20,
        product_id: str | None = None,
        category: Category | None = None,
    ) -> list[ScoredEvidence]:
        """Perform semantic vector similarity search for query."""
        if not self._documents or not query.strip():
            return []

        query_vec = self.embedding_provider.embed_text(query)
        scored: list[tuple[float, EvidenceDocument]] = []

        for idx, doc in enumerate(self._documents):
            if product_id is not None and doc.product_id != product_id:
                continue
            if category is not None and doc.category != category:
                continue

            sim = cosine_similarity(query_vec, self._vectors[idx])
            scored.append((sim, doc))

        # Sort descending by similarity, tie-break by document_id ascending
        scored.sort(key=lambda item: (-item[0], item[1].document_id))

        results: list[ScoredEvidence] = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                ScoredEvidence(
                    document=doc,
                    score=round(score, 4),
                    retrieval_method="vector",
                    rank=rank,
                )
            )

        return results
