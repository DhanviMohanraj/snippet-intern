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
from app.agent.respond import generate_recommendation_response
from app.agent.response_validator import (
    build_fallback_product_recommendation,
    build_fallback_recommendation_response,
    check_claim_contradictions,
    validate_recommendation_response,
)
from app.agent.pipeline import PipelineResult, run_pipeline
from app.agent.retrieval import retrieve_candidates
from app.agent.slots import (
    RequirementSlot,
    SLOT_REGISTRY,
    SlotStatus,
    can_proceed_to_recommendation,
    get_next_slot_to_clarify,
    get_unresolved_mandatory_slots,
    is_slot_answered,
    is_slot_resolved,
)
from app.models.response import (
    LLMProductExplanation,
    LLMRecommendationPayload,
    ProductRecommendation,
    RecommendationResponse,
)

__all__ = [
    "CandidateEvaluation",
    "CandidateSearchResult",
    "ConstraintCheck",
    "ConstraintStatus",
    "EXTRACTION_EXAMPLES",
    "ExtractionExample",
    "LLMProductExplanation",
    "LLMRecommendationPayload",
    "MAX_CLARIFICATION_TURNS",
    "PipelineResult",
    "PolicyAction",
    "PolicyDecision",
    "ProductRecommendation",
    "RankedCandidate",
    "RecommendationResponse",
    "ScoreBreakdown",
    "build_fallback_product_recommendation",
    "build_fallback_recommendation_response",
    "check_claim_contradictions",
    "decide_next_action",
    "detect_contradictions",
    "evaluate_candidate",
    "extract",
    "filter_candidates",
    "generate_recommendation_response",
    "merge",
    "normalize",
    "rank_candidates",
    "retrieve_candidates",
    "run_pipeline",
    "score_candidate",
    "search_and_rank",
    "validate_recommendation_response",
]

