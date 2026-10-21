"""Okapi BM25 Lexical Keyword Retriever for SpecSage Hybrid RAG.

Provides deterministic, zero-dependency lexical search optimized for electronics
catalogue specifications, exact product names, certifications, and feature terms.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from app.agent.rag.models import EvidenceDocument, ScoredEvidence
from app.models.product import Category

# Stopwords that carry minimal discriminative value in technical search
STOPWORDS: set[str] = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "with",
    "on", "at", "by", "from", "is", "are", "was", "were", "it", "its",
    "that", "this", "these", "those", "be", "has", "have", "had",
}

# Regex to capture words, numbers, and technical spec alphanumeric tokens (e.g., 16gb, 144hz, rtx4060)
TOKEN_REGEX = re.compile(r"[a-zA-Z0-9]+(?:[\.\-][a-zA-Z0-9]+)*")
COMPOUND_SPEC_REGEX = re.compile(r"^(\d+)([a-zA-Z]+)$")


def tokenize(text: str) -> list[str]:
    """Tokenize technical text into normalized terms, splitting compounds like 16gb -> [16gb, 16, gb]."""
    raw_tokens = TOKEN_REGEX.findall(text.lower())
    tokens: list[str] = []

    for t in raw_tokens:
        t_clean = t.strip(".-")
        if not t_clean or t_clean in STOPWORDS:
            continue
        tokens.append(t_clean)

        # Compound technical tokens: e.g. "16gb" -> also emit "16" and "gb"
        compound_match = COMPOUND_SPEC_REGEX.match(t_clean)
        if compound_match:
            num_part = compound_match.group(1)
            unit_part = compound_match.group(2)
            if num_part:
                tokens.append(num_part)
            if unit_part and unit_part not in STOPWORDS:
                tokens.append(unit_part)

    return tokens


class BM25Retriever:
    """Okapi BM25 lexical ranking engine."""

    def __init__(
        self,
        documents: Sequence[EvidenceDocument] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._documents: list[EvidenceDocument] = []
        self._doc_map: dict[str, EvidenceDocument] = {}
        self._doc_tokens: list[list[str]] = []
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_len: float = 0.0
        self._idf: dict[str, float] = {}

        if documents:
            self.build_index(documents)

    def build_index(self, documents: Sequence[EvidenceDocument]) -> None:
        """Construct BM25 index over the provided evidence documents."""
        self._documents = list(documents)
        self._doc_map = {d.document_id: d for d in self._documents}
        self._doc_tokens = [tokenize(d.content) for d in self._documents]
        self._doc_term_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]

        num_docs = len(self._documents)
        if num_docs == 0:
            self._avg_doc_len = 0.0
            self._idf = {}
            return

        self._avg_doc_len = sum(self._doc_lengths) / num_docs

        # Calculate Document Frequencies
        df: Counter[str] = Counter()
        for term_counter in self._doc_term_freqs:
            for term in term_counter.keys():
                df[term] += 1

        # Calculate Robertson-Spärck Jones IDF with floor of 0.0
        self._idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF formulation: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf_val = math.log(1.0 + (num_docs - freq + 0.5) / (freq + 0.5))
            self._idf[term] = max(0.001, idf_val)

    def search(
        self,
        query: str,
        top_k: int = 20,
        product_id: str | None = None,
        category: Category | None = None,
    ) -> list[ScoredEvidence]:
        """Rank evidence documents for query using BM25 scoring."""
        if not self._documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, EvidenceDocument]] = []
        avgdl = self._avg_doc_len or 1.0

        for idx, doc in enumerate(self._documents):
            # Apply product or category filters if specified
            if product_id is not None and doc.product_id != product_id:
                continue
            if category is not None and doc.category != category:
                continue

            tf_counter = self._doc_term_freqs[idx]
            doc_len = self._doc_lengths[idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / avgdl)

            score = 0.0
            for q_term in query_tokens:
                if q_term not in tf_counter:
                    continue
                tf = tf_counter[q_term]
                idf = self._idf.get(q_term, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * len_norm
                score += idf * (numerator / denominator)

            if score > 0.0:
                scored.append((score, doc))

        # Sort descending by score, tie-break by document_id ascending for determinism
        scored.sort(key=lambda item: (-item[0], item[1].document_id))

        results: list[ScoredEvidence] = []
        for rank, (s, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                ScoredEvidence(
                    document=doc,
                    score=round(s, 4),
                    retrieval_method="bm25",
                    rank=rank,
                )
            )

        return results
