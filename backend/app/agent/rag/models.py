"""Data models for Lightweight Hybrid RAG and Evidence Grounding.

Defines:
    - EvidenceDocument: Atomic, provenance-tracked evidence unit extracted from product catalogue records.
    - ScoredEvidence: Retrieved evidence paired with relevance score, retrieval method, and rank.
    - ProductEvidenceSummary: Product-isolated evidence package for grounded LLM explanation generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.product import Category


@dataclass
class EvidenceDocument:
    """Atomic evidence unit derived from verified catalogue product records."""

    document_id: str
    product_id: str
    category: Category
    product_name: str
    source_field: str
    recorded_value: Any
    content: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert evidence document to JSON-serializable dictionary."""
        return {
            "document_id": self.document_id,
            "product_id": self.product_id,
            "category": self.category.value,
            "product_name": self.product_name,
            "source_field": self.source_field,
            "recorded_value": self.recorded_value,
            "content": self.content,
            "provenance": self.provenance,
        }


@dataclass
class ScoredEvidence:
    """Retrieved evidence document annotated with score, rank, and retrieval method."""

    document: EvidenceDocument
    score: float
    retrieval_method: str  # "bm25", "vector", or "hybrid_rrf"
    rank: int = 0

    @property
    def product_id(self) -> str:
        return self.document.product_id

    @property
    def source_field(self) -> str:
        return self.document.source_field

    @property
    def content(self) -> str:
        return self.document.content


@dataclass
class ProductEvidenceSummary:
    """Structured, product-isolated evidence package supplied to LLM explanation generator."""

    product_id: str
    category: Category
    product_name: str
    retrieved_evidence: list[ScoredEvidence] = field(default_factory=list)
    verified_specs: dict[str, Any] = field(default_factory=dict)
    verified_features: list[str] = field(default_factory=list)
    unknown_attributes: list[str] = field(default_factory=list)
