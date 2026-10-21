"""Hybrid Evidence Retriever with Reciprocal Rank Fusion (RRF).

Combines lexical BM25 keyword search with dense vector similarity search to produce
deduplicated, provenance-grounded evidence sets for electronics recommendations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from app.agent.rag.bm25 import BM25Retriever
from app.agent.rag.embeddings import EmbeddingProvider, LocalHashEmbeddingProvider
from app.agent.rag.indexer import EvidenceIndex, build_product_evidence_documents
from app.agent.rag.models import (
    EvidenceDocument,
    ProductEvidenceSummary,
    ScoredEvidence,
)
from app.agent.rag.vector import VectorRetriever
from app.core.logging import get_logger
from app.models.product import CATEGORY_SPEC_SCHEMA, Category, Product
from app.models.requirements import Requirements

logger = get_logger("app.agent.rag.hybrid")

_GLOBAL_RETRIEVER: HybridEvidenceRetriever | None = None


class HybridEvidenceRetriever:
    """Coordinates BM25 and Vector retrievers using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        index: EvidenceIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index = index or EvidenceIndex()
        docs = self.index.get_documents()
        self.bm25 = BM25Retriever(documents=docs, k1=k1, b=b)
        self.vector = VectorRetriever(
            documents=docs,
            embedding_provider=embedding_provider or LocalHashEmbeddingProvider(),
        )

    def rebuild_index(self, products: Sequence[Product]) -> None:
        """Re-index a sequence of Product models across both retrieval backends."""
        self.index.build_from_products(products)
        docs = self.index.get_documents()
        self.bm25.build_index(docs)
        self.vector.build_index(docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        product_id: str | None = None,
        category: Category | None = None,
        rrf_k: int = 60,
        weight_bm25: float = 1.0,
        weight_vector: float = 1.0,
    ) -> list[ScoredEvidence]:
        """Retrieve deduplicated evidence using Reciprocal Rank Fusion (RRF).

        Formula:
            RRF_score(d) = sum_{m in {bm25, vector}} (w_m / (rrf_k + rank_m(d)))

        Evidence is deduplicated by (product_id, source_field), preserving the highest score
        and complete provenance metadata.
        """
        # Fetch candidate sets from both retrieval methods
        candidate_multiplier = max(20, top_k * 3)
        bm25_matches = self.bm25.search(
            query=query,
            top_k=candidate_multiplier,
            product_id=product_id,
            category=category,
        )
        vector_matches = self.vector.search(
            query=query,
            top_k=candidate_multiplier,
            product_id=product_id,
            category=category,
        )

        # Reciprocal Rank Fusion
        fused_scores: dict[str, float] = defaultdict(float)
        doc_store: dict[str, EvidenceDocument] = {}

        for rank, item in enumerate(bm25_matches, start=1):
            doc_id = item.document.document_id
            fused_scores[doc_id] += weight_bm25 / (rrf_k + rank)
            doc_store[doc_id] = item.document

        for rank, item in enumerate(vector_matches, start=1):
            doc_id = item.document.document_id
            fused_scores[doc_id] += weight_vector / (rrf_k + rank)
            doc_store[doc_id] = item.document

        if not fused_scores:
            return []

        # Deduplicate by (product_id, source_field)
        field_dedup: dict[tuple[str, str], tuple[float, EvidenceDocument]] = {}
        for doc_id, score in fused_scores.items():
            doc = doc_store[doc_id]
            dedup_key = (doc.product_id, doc.source_field)
            if dedup_key not in field_dedup or score > field_dedup[dedup_key][0]:
                field_dedup[dedup_key] = (score, doc)

        # Sort descending by fused RRF score, tie-break by document_id
        ranked_items = sorted(
            field_dedup.values(),
            key=lambda item: (-item[0], item[1].document_id),
        )

        results: list[ScoredEvidence] = []
        for rank, (score, doc) in enumerate(ranked_items[:top_k], start=1):
            results.append(
                ScoredEvidence(
                    document=doc,
                    score=round(score, 6),
                    retrieval_method="hybrid_rrf",
                    rank=rank,
                )
            )

        return results

    def get_product_grounding_context(
        self,
        product: Product,
        requirements: Requirements,
        top_k: int = 8,
    ) -> ProductEvidenceSummary:
        """Construct structured, product-isolated evidence package for LLM explanation generation.

        Guarantees that:
            - Retrieved evidence strictly belongs to product.product_id.
            - Specs absent from the catalogue record are explicitly listed as unknown_attributes.
            - Specifications from other products are never included.
        """
        # Formulate contextual search query from user's use cases, category, and preferences
        query_parts: list[str] = []
        if requirements.category:
            query_parts.append(requirements.category.value)
        if requirements.use_cases:
            query_parts.extend(requirements.use_cases)
        if requirements.preferences:
            query_parts.extend(requirements.preferences.keys())

        query = " ".join(query_parts) if query_parts else product.name

        # Retrieve evidence strictly for this product_id
        retrieved = self.retrieve(
            query=query,
            top_k=top_k,
            product_id=product.product_id,
            category=product.category,
        )

        # Staleness guard: the index is built once from data/products.json, while
        # `product` is the live MongoDB record. If they differ (DB seeded from a
        # different file, or a record edited), indexed evidence would contradict
        # the verified specs sent alongside it. Keep only evidence that matches
        # the live record, and fill gaps from the live record itself.
        live_docs = {d.source_field: d for d in build_product_evidence_documents(product)}
        fresh = [s for s in retrieved if (ld := live_docs.get(s.document.source_field)) and ld.content == s.document.content]
        if len(fresh) < len(retrieved):
            logger.warning(
                "Evidence index is stale for %s (%d of %d retrieved documents no longer match the catalogue record); "
                "using the live record. Call rebuild_index() after reseeding.",
                product.product_id, len(retrieved) - len(fresh), len(retrieved),
            )
        if len(fresh) < min(top_k, len(live_docs)):
            used = {s.document.source_field for s in fresh}
            for doc in live_docs.values():
                if len(fresh) >= top_k:
                    break
                if doc.source_field not in used:
                    fresh.append(ScoredEvidence(document=doc, score=0.0, retrieval_method="catalogue_record", rank=len(fresh) + 1))
        retrieved = fresh

        # Determine unknown specifications according to schema
        expected_specs = CATEGORY_SPEC_SCHEMA.get(product.category, set())
        present_specs = set(product.specs.keys())
        unknown_specs = sorted(list(expected_specs - present_specs))

        return ProductEvidenceSummary(
            product_id=product.product_id,
            category=product.category,
            product_name=product.name,
            retrieved_evidence=retrieved,
            verified_specs=dict(product.specs),
            verified_features=list(product.features),
            unknown_attributes=unknown_specs,
        )


def get_hybrid_retriever() -> HybridEvidenceRetriever:
    """Retrieve or initialize the global singleton HybridEvidenceRetriever instance."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        try:
            index = EvidenceIndex.from_catalogue_file()
            _GLOBAL_RETRIEVER = HybridEvidenceRetriever(index=index)
            logger.info("Initialized global HybridEvidenceRetriever with %d evidence documents.", index.document_count)
        except Exception as err:
            logger.warning("Failed to initialize catalogue file index: %s. Using empty index fallback.", err)
            _GLOBAL_RETRIEVER = HybridEvidenceRetriever(index=EvidenceIndex([]))
    return _GLOBAL_RETRIEVER
