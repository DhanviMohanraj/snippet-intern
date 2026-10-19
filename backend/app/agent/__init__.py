"""Recommendation agent package.

Exports:
    extract: Asynchronous NLU requirements extractor using LLMClient.
    normalize: Deterministic pure Python requirements normalizer.
    merge: Pure Python multi-turn requirements merger.
    detect_contradictions: Deterministic contradiction detector.
    EXTRACTION_EXAMPLES: 15 canonical handwritten extraction test examples.
"""

from app.agent.examples import EXTRACTION_EXAMPLES, ExtractionExample
from app.agent.extract import extract
from app.agent.merge import detect_contradictions, merge
from app.agent.normalize import normalize
from app.agent.policy import (
    MAX_CLARIFICATION_TURNS,
    PolicyAction,
    PolicyDecision,
    decide_next_action,
)
from app.agent.ranking import (
    CandidateEvaluation,
    CandidateSearchResult,
    ConstraintCheck,
    ConstraintStatus,
    RankedCandidate,
    ScoreBreakdown,
    evaluate_candidate,
    filter_candidates,
    rank_candidates,
    score_candidate,
    search_and_rank,
)
from app.agent.retrieval import retrieve_candidates

__all__ = [
    "CandidateEvaluation",
    "CandidateSearchResult",
    "ConstraintCheck",
    "ConstraintStatus",
    "EXTRACTION_EXAMPLES",
    "ExtractionExample",
    "MAX_CLARIFICATION_TURNS",
    "PolicyAction",
    "PolicyDecision",
    "RankedCandidate",
    "ScoreBreakdown",
    "decide_next_action",
    "detect_contradictions",
    "evaluate_candidate",
    "extract",
    "filter_candidates",
    "merge",
    "normalize",
    "rank_candidates",
    "retrieve_candidates",
    "score_candidate",
    "search_and_rank",
]

