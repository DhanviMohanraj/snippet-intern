"""Response generation service for electronics recommendations.

Combines Module 8 search results with Groq LLM natural-language explanations, enforcing:
    1. Authoritative catalogue data separation: Prices, specs, ratings, and IDs are copied from the database.
    2. Deterministic grounding: Generated explanations are validated and factual contradictions are repaired.
    3. Ranking preservation: The authoritative ranking order from Module 8 is strictly maintained.
    4. Deterministic fallbacks: Graceful degradation when LLM is unavailable, times out, or produces invalid output.

Principle: The LLM explains; deterministic code validates and enforces.
"""

from __future__ import annotations

from app.agent.rag import ProductEvidenceSummary, get_hybrid_retriever
from app.agent.response_validator import (
    build_fallback_recommendation_response,
    validate_recommendation_response,
)
from app.core.logging import get_logger
from app.llm.client import LLMClient
from app.llm.prompts import build_recommendation_prompt
from app.models.ranking import CandidateSearchResult
from app.models.requirements import Requirements
from app.models.response import LLMRecommendationPayload, RecommendationResponse

logger = get_logger("app.agent.respond")


async def generate_recommendation_response(
    requirements: Requirements,
    search_result: CandidateSearchResult,
    llm_client: LLMClient | None = None,
) -> RecommendationResponse:
    """Generate a structured, grounded recommendation response for the user.

    Args:
        requirements: Current user requirements specifying use cases, budget, and preferences.
        search_result: CandidateSearchResult produced by Module 8 search_and_rank().
        llm_client: Injected LLMClient (GroqClient or FakeLLMClient). If None, generates fallback response.

    Returns:
        Validated RecommendationResponse containing ordered recommendations and grounded explanations.
    """
    # 1. No-Match Short-Circuit: If no candidates were eligible, return deterministic no-match immediately
    if not search_result.has_matches or not search_result.ranked_candidates:
        logger.info("Search result has no eligible candidates; returning deterministic no-match response.")
        return build_fallback_recommendation_response(
            requirements,
            search_result,
            fallback_reason="No eligible products matched all criteria.",
        )

    # 2. Retrieve Product-Isolated Grounding Evidence via Hybrid RAG Layer
    evidence_summaries: dict[str, ProductEvidenceSummary] = {}
    try:
        retriever = get_hybrid_retriever()
        for cand in search_result.ranked_candidates:
            evidence_summaries[cand.product_id] = retriever.get_product_grounding_context(
                product=cand.product,
                requirements=requirements,
                top_k=6,
            )
    except Exception as err:
        logger.warning(
            "Hybrid RAG evidence retrieval failed: %s. Proceeding with catalogue context.",
            err,
        )

    # 3. Offline / No LLM Fallback
    if llm_client is None:
        logger.info("No LLM client provided; returning deterministic grounded fallback response.")
        return build_fallback_recommendation_response(
            requirements,
            search_result,
            fallback_reason="Offline fallback: LLM client not provided.",
            evidence_summaries=evidence_summaries,
        )

    # 4. Build Prompt Context with Retrieved Evidence and Call LLM
    messages = build_recommendation_prompt(
        requirements,
        search_result.ranked_candidates,
        evidence_summaries=evidence_summaries,
    )

    try:
        payload = await llm_client.complete_json(
            messages=messages,
            schema=LLMRecommendationPayload,
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as err:
        logger.warning(
            "LLM recommendation explanation generation failed: %s. Falling back to deterministic response.",
            err,
        )
        return build_fallback_recommendation_response(
            requirements,
            search_result,
            fallback_reason=f"LLM generation failed: {type(err).__name__}",
            evidence_summaries=evidence_summaries,
        )

    # 5. Deterministic Validation and Grounding Verification
    try:
        response = validate_recommendation_response(
            payload,
            search_result,
            requirements,
            evidence_summaries=evidence_summaries,
        )
        return response
    except Exception as err:
        logger.warning(
            "Response validation encountered error: %s. Falling back to deterministic response.",
            err,
        )
        return build_fallback_recommendation_response(
            requirements,
            search_result,
            fallback_reason=f"Response validation failed: {type(err).__name__}",
            evidence_summaries=evidence_summaries,
        )
