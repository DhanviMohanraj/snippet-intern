"""Vector Embedding Providers for Lightweight Semantic Retrieval.

Defines:
    - EmbeddingProvider: Abstract interface for text embedding providers.
    - LocalHashEmbeddingProvider: Zero-dependency, deterministic subword hash embedding engine.
    - FakeEmbeddingProvider: Predictable embedding provider for unit testing without network/API calls.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for dense vector embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string into a normalized dense vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings."""
        return [self.embed_text(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vector space."""
        ...


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Deterministic, zero-dependency subword and term-frequency hash embedding provider.

    Projects tokens and character n-grams into a fixed D-dimensional vector space
    using Murmur-style hash projections and L2 normalization.
    Cosine similarity between vectors reflects subword and semantic vocabulary overlap.
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Compute L2-normalized dense embedding vector from text."""
        vec = [0.0] * self._dim
        clean = text.lower().strip()
        if not clean:
            return vec

        # Extract word tokens and subword character n-grams (3-grams and 4-grams)
        words = re.findall(r"[a-z0-9]+", clean)
        features: list[str] = list(words)

        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    features.append(w[i:i + 3])
            if len(w) >= 4:
                for i in range(len(w) - 3):
                    features.append(w[i:i + 4])

        for feat in features:
            # Deterministic hash to dimension index and sign (+1 / -1)
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        # Compute L2 norm
        sq_sum = sum(x * x for x in vec)
        if sq_sum > 0.0:
            norm = math.sqrt(sq_sum)
            vec = [x / norm for x in vec]

        return vec


class FakeEmbeddingProvider(EmbeddingProvider):
    """Test embedding provider with customizable fixed vectors or deterministic mappings."""

    def __init__(
        self,
        dimension: int = 4,
        preset_vectors: dict[str, list[float]] | None = None,
    ) -> None:
        self._dim = dimension
        self.preset_vectors = preset_vectors or {}

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        """Return preset vector if matched; else return deterministic simple vector."""
        text_lower = text.lower().strip()
        for k, v in self.preset_vectors.items():
            if k.lower() in text_lower:
                return v

        # Default unit vector based on length mod dimension
        idx = len(text_lower) % self._dim
        vec = [0.0] * self._dim
        vec[idx] = 1.0
        return vec
