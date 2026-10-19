"""Recommendation agent package.

Exports:
    extract: Asynchronous NLU requirements extractor using LLMClient.
    normalize: Deterministic pure Python requirements normalizer.
    EXTRACTION_EXAMPLES: 15 canonical handwritten extraction test examples.
"""

from app.agent.examples import EXTRACTION_EXAMPLES, ExtractionExample
from app.agent.extract import extract
from app.agent.normalize import normalize

__all__ = ["EXTRACTION_EXAMPLES", "ExtractionExample", "extract", "normalize"]
