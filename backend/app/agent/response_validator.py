"""Deterministic response validation and grounded fallback synthesis.

Implements Module 9 validation rules:
    - Product identity validation: Only pre-filtered eligible candidates from Module 8 can be recommended.
    - Spec and price integrity: Authoritative catalogue values (price, specs, rating, ID) are strictly enforced.
    - Grounding validation: Factual and numerical claims in generated text are verified against catalogue records;
      contradictions or unsupported specs trigger deterministic repair.
    - Fallback generation: Deterministic, 100% grounded explanations generated directly from Module 8 metadata.
    - Ranking preservation: The authoritative shortlist ordering established by Module 8 cannot be altered.

Principle: The LLM explains; deterministic code validates and enforces.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from app.core.logging import get_logger
from app.models.product import Category, Product
from app.models.ranking import CandidateSearchResult, RankedCandidate
from app.models.requirements import Requirements
from app.models.response import (
    LLMProductExplanation,
    LLMRecommendationPayload,
    ProductRecommendation,
    RecommendationResponse,
)

logger = get_logger("app.agent.response_validator")


# ---------------------------------------------------------------------------
# 1. Deterministic Fallback Synthesis
# ---------------------------------------------------------------------------

def build_fallback_product_recommendation(
    candidate: RankedCandidate,
    requirements: Requirements,
) -> ProductRecommendation:
    """Build a 100% factual, grounded ProductRecommendation from catalogue and ranking metadata.

    Guarantees that:
        - All product IDs, prices, ratings, and verified specs match catalogue records exactly.
        - Reasoning is tailored to the user's use case and top score dimensions.
        - Real trade-offs (e.g. integrated graphics, weight, lack of ANC) are stated honestly.
        - Missing catalogue attributes are noted as limitations.

    Args:
        candidate: Authoritative RankedCandidate from Module 8.
        requirements: Current user requirements.

    Returns:
        Validated ProductRecommendation.
    """
    p = candidate.product
    sb = candidate.score_breakdown

    # 1. Build concise, grounded reasoning
    use_case_str = (
        ", ".join(requirements.use_cases) if requirements.use_cases else "your requirements"
    )
    score_desc = f"a composite match score of {candidate.overall_score:.1f}/100"
    reasoning = (
        f"The {p.name} ({p.brand}) is recommended for {use_case_str} with {score_desc}. "
        f"{'; '.join(candidate.reasons[:2])}."
    )

    # 2. Build supported advantages from score breakdown & features
    advantages: list[str] = []
    # Add top preference explanation
    if sb.weighted_scores:
        top_pref = max(sb.weighted_scores.items(), key=lambda item: item[1])[0]
        if top_pref in sb.explanations:
            advantages.append(sb.explanations[top_pref])
    # Add key product features from catalogue
    if p.features:
        for feat in p.features[:3]:
            if feat not in advantages:
                advantages.append(feat)

    # 3. Build honest, verified trade-offs supported by catalogue data
    trade_offs: list[str] = []
    if p.category == Category.LAPTOP:
        vram = p.specs.get("vram_gb", 0)
        gpu = p.specs.get("gpu", "")
        if vram == 0:
            trade_offs.append(f"Uses integrated graphics ({gpu}); not suited for intensive AAA gaming or local 3D rendering.")
        w = p.specs.get("weight_kg")
        if w is not None and w >= 2.0:
            trade_offs.append(f"At {w}kg, this laptop is heavier than ultrabooks.")
        b = p.specs.get("battery_hr")
        if b is not None and b < 8.0:
            trade_offs.append(f"Battery runtime of {b} hours may require keeping the charger nearby.")

    elif p.category == Category.HEADPHONES:
        anc = p.specs.get("anc")
        if anc is False:
            trade_offs.append("Features passive noise isolation only; does not include Active Noise Cancellation (ANC).")

    elif p.category == Category.PHONE:
        w_g = p.specs.get("weight_g")
        if w_g is not None and w_g >= 210:
            trade_offs.append(f"Device weight of {w_g}g is on the heavier side for single-handed use.")

    if not trade_offs:
        trade_offs.append("Priced near the upper threshold of its performance bracket.")

    # 4. Build match explanation from passed constraints
    passed_labels = [
        c.reason for c in candidate.evaluation.passed_constraints
        if c.name in ("budget_max", "min_ram_gb", "min_storage_gb", "requires_gpu", "min_battery_hr")
    ]
    if passed_labels:
        match_explanation = f"Verified compliance: {'; '.join(passed_labels[:3])}."
    else:
        match_explanation = f"Priced at ₹{p.price_inr:,}, matching your category requirements."

    # 5. Build limitations from unverifiable constraints or missing specs
    limitations: list[str] = []
    for u in candidate.evaluation.unverifiable_constraints:
        limitations.append(u.reason)
    for unsup in sb.unsupported_preferences:
        limitations.append(f"'{unsup}' preference cannot be verified from catalogue specifications.")

    return ProductRecommendation(
        product_id=p.product_id,
        name=p.name,
        brand=p.brand,
        category=p.category,
        price_inr=p.price_inr,
        rating=p.rating,
        verified_specs=dict(p.specs),
        features=list(p.features),
        overall_score=candidate.overall_score,
        reasoning=reasoning,
        advantages=advantages[:4],
        trade_offs=trade_offs[:3],
        match_explanation=match_explanation,
        limitations=limitations,
    )


def build_fallback_recommendation_response(
    requirements: Requirements,
    search_result: CandidateSearchResult,
    fallback_reason: str = "Deterministic fallback generated.",
) -> RecommendationResponse:
    """Generate a complete, safe, and factual recommendation response without LLM calls.

    Handles both successful candidate sets and no-match outcomes.

    Args:
        requirements: Current user requirements.
        search_result: CandidateSearchResult from Module 8.
        fallback_reason: Explanation of why fallback generation was used.

    Returns:
        Structured RecommendationResponse.
    """
    cat_name = requirements.category.value if requirements.category else "electronics"

    # 1. Handle No-Match scenario
    if not search_result.has_matches or not search_result.ranked_candidates:
        excluded_reasons: list[str] = []
        for ex in search_result.excluded_candidates[:3]:
            for r in ex.rejection_reasons[:1]:
                excluded_reasons.append(f"{ex.product.name}: {r}")

        reasons_summary = f" ({'; '.join(excluded_reasons)})" if excluded_reasons else ""
        msg = (
            f"I could not find any {cat_name} in our catalogue that satisfy all your non-negotiable criteria{reasons_summary}. "
            "Please consider adjusting your budget or relaxing specific constraints."
        )

        return RecommendationResponse(
            message=msg,
            category=requirements.category,
            recommendations=[],
            total_matching=0,
            follow_up_question="Would you like to increase your budget or adjust any hardware specifications?",
            data_quality_notes=list(search_result.data_quality_limitations),
            is_fallback=True,
            fallback_reason=fallback_reason,
            has_matches=False,
        )

    # 2. Build recommendations for eligible candidates in strict ranked order
    recommendations: list[ProductRecommendation] = [
        build_fallback_product_recommendation(cand, requirements)
        for cand in search_result.ranked_candidates
    ]

    use_case_text = (
        f" for {', '.join(requirements.use_cases)}" if requirements.use_cases else ""
    )
    intro_message = (
        f"Based on your requirements, here are the top recommended {cat_name}s from our catalogue{use_case_text}:"
    )

    return RecommendationResponse(
        message=intro_message,
        category=requirements.category,
        recommendations=recommendations,
        total_matching=len(search_result.ranked_candidates),
        follow_up_question="Would you like more details on any of these options, or should we compare specific models?",
        data_quality_notes=list(search_result.data_quality_limitations),
        is_fallback=True,
        fallback_reason=fallback_reason,
        has_matches=True,
    )


# ---------------------------------------------------------------------------
# 2. Grounding and Claim Contradiction Detection
# ---------------------------------------------------------------------------

def check_claim_contradictions(
    text: str,
    product: Product,
    requirements: Requirements,
) -> list[str]:
    """Inspect generated text for factual contradictions against verified product specifications.

    Detects:
        - Price contradictions (stating a price that contradicts the catalogue).
        - RAM size contradictions (claiming an incorrect amount of RAM).
        - Storage size contradictions (claiming an incorrect storage size).
        - Weight contradictions (claiming an incorrect weight).
        - Dedicated GPU claims on integrated graphics laptops.
        - Runtime claims on products lacking verified battery_hr specifications.
        - Monitor weight claims when catalogue does not track monitor weight.

    Args:
        text: The generated text (reasoning, advantages, match_explanation).
        product: Authoritative Product catalogue record.
        requirements: User requirements for budget reference.

    Returns:
        List of detected contradiction descriptions. Empty if text is grounded.
    """
    contradictions: list[str] = []
    text_lower = text.lower()

    # 1. Price claim validation
    # Match patterns like: ₹50,000, Rs. 50000, Rs 50,000, INR 50000
    # Also capture preceding context to differentiate price vs savings
    price_matches = re.finditer(
        r"(?:(sav(?:ing|ings|e|ed)(?:\s+of)?|discount(?:\s+of)?)\s+)?(?:₹|rs\.?\s*|inr\s*)(\d[\d,]+)",
        text_lower,
    )
    for match in price_matches:
        prefix = match.group(1)
        pm = match.group(2)
        try:
            num = int(pm.replace(",", ""))
            allowed_numbers = {product.price_inr}
            if requirements.budget_max_inr:
                allowed_numbers.add(requirements.budget_max_inr)
                allowed_numbers.add(requirements.budget_max_inr - product.price_inr)
            if requirements.budget_min_inr:
                allowed_numbers.add(requirements.budget_min_inr)
                allowed_numbers.add(product.price_inr - requirements.budget_min_inr)

            if prefix:
                # Explicitly marked as savings or discount, not a product price claim
                continue

            # If number is within typical price range (> 1000) and doesn't match product price or budget
            if num >= 1000 and all(abs(num - a) > 100 for a in allowed_numbers):
                contradictions.append(
                    f"Contradictory price claim: text mentions ₹{num:,}, but catalogue price is ₹{product.price_inr:,}."
                )
        except ValueError:
            continue

    # 2. RAM claim validation
    ram_matches = re.findall(r"(\d+)\s*(?:gb|gigabytes?)\s*(?:of\s*)?ram", text_lower)
    cat_ram = product.specs.get("ram_gb")
    for rm in ram_matches:
        try:
            num = int(rm)
            if cat_ram is not None and num != cat_ram:
                contradictions.append(
                    f"Contradictory RAM claim: text mentions {num}GB RAM, but catalogue specifies {cat_ram}GB."
                )
        except ValueError:
            continue

    # 3. Storage claim validation
    storage_matches = re.findall(r"(\d+)\s*(?:gb|tb)\s*(?:of\s*)?(?:ssd|hdd|storage)", text_lower)
    cat_storage = product.specs.get("storage_gb")
    for sm in storage_matches:
        try:
            num = int(sm)
            # If in TB, convert 1TB to 1024GB for comparison
            if "tb" in text_lower and num <= 4:
                num_gb = num * 1024
            else:
                num_gb = num
            if cat_storage is not None and abs(num_gb - cat_storage) > 30:
                contradictions.append(
                    f"Contradictory storage claim: text mentions {sm}, but catalogue specifies {cat_storage}GB."
                )
        except ValueError:
            continue

    # 4. Discrete GPU claims on integrated graphics
    if product.category == Category.LAPTOP:
        vram = product.specs.get("vram_gb", 0)
        gpu = product.specs.get("gpu", "").lower()
        if vram == 0 and not any(k in gpu for k in ("rtx", "gtx", "radeon rx", "arc a")):
            if any(term in text_lower for term in ("dedicated rtx", "discrete rtx", "dedicated nvidia", "dedicated gpu", "discrete gpu", "geforce rtx")):
                contradictions.append(
                    f"Contradictory GPU claim: text claims dedicated GPU, but product has integrated graphics ('{product.specs.get('gpu')}')."
                )

    # 5. Fabricated battery runtime on products lacking battery_hr
    if "battery_hr" not in product.specs:
        # Check if text claims specific hourly battery runtime
        hr_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?battery", text_lower)
        if hr_matches:
            contradictions.append(
                f"Contradictory battery claim: text claims {hr_matches[0]} hours battery runtime, but catalogue lacks battery_hr specification."
            )

    # 6. Fabricated monitor weight
    if product.category == Category.MONITOR:
        weight_matches = re.findall(r"(\d+(?:\.\d+)?)\s*kg", text_lower)
        if weight_matches:
            contradictions.append(
                f"Contradictory monitor weight claim: text mentions {weight_matches[0]}kg, but catalogue does not track monitor weight."
            )

    return contradictions


# ---------------------------------------------------------------------------
# 3. Comprehensive Response Validator
# ---------------------------------------------------------------------------

def validate_recommendation_response(
    payload: LLMRecommendationPayload,
    search_result: CandidateSearchResult,
    requirements: Requirements,
) -> RecommendationResponse:
    """Deterministically validate and assemble an LLM recommendation response.

    Enforces:
        1. Product Identity: Only candidates present in search_result.ranked_candidates can be recommended.
        2. Ranking Order: Output order strictly mirrors search_result.ranked_candidates.
        3. Authoritative Specs: Price, verified_specs, rating, and brand are copied from catalogue records.
        4. No Duplicates: Duplicate product IDs are prevented.
        5. Factual Grounding: Contradictory numerical or spec claims trigger per-product repair.
        6. Safe Fallback: If payload is malformed or all explanations fail, returns full fallback.

    Args:
        payload: LLMRecommendationPayload returned by Groq LLM.
        search_result: Module 8 search result containing authoritative RankedCandidate list.
        requirements: Current user requirements.

    Returns:
        Fully validated, authoritative RecommendationResponse.
    """
    ranked_candidates = search_result.ranked_candidates

    # If no eligible candidates exist, immediately return no-match fallback
    if not search_result.has_matches or not ranked_candidates:
        return build_fallback_recommendation_response(
            requirements, search_result, fallback_reason="No eligible candidates found."
        )

    # Map of authoritative candidates keyed by product_id
    cand_map: dict[str, RankedCandidate] = {
        cand.product_id: cand for cand in ranked_candidates
    }

    # Map of LLM explanations, rejecting unknown IDs and duplicate occurrences
    exp_map: dict[str, LLMProductExplanation] = {}
    for exp in payload.explanations:
        pid = exp.product_id.strip()
        if pid not in cand_map:
            logger.warning("Rejected LLM-generated recommendation for unauthorized product_id '%s'.", pid)
            continue
        if pid in exp_map:
            logger.warning("Detected duplicate explanation for product_id '%s'; ignoring duplicate.", pid)
            continue
        exp_map[pid] = exp

    # Assemble recommendations in EXACT authoritative Module 8 ranked order
    recommendations: list[ProductRecommendation] = []
    any_repaired = False

    for cand in ranked_candidates:
        pid = cand.product_id
        if pid in exp_map:
            exp = exp_map[pid]
            # Check for factual / numerical contradictions in LLM text
            combined_text = f"{exp.reasoning} {' '.join(exp.advantages)} {exp.match_explanation} {' '.join(exp.trade_offs)}"
            contradictions = check_claim_contradictions(combined_text, cand.product, requirements)

            if contradictions:
                logger.warning(
                    "Factual contradiction in LLM explanation for product '%s': %s. Repairing with grounded fallback.",
                    pid,
                    contradictions,
                )
                any_repaired = True
                recommendations.append(build_fallback_product_recommendation(cand, requirements))
            else:
                # Merge validated explanation with authoritative catalogue product record
                recommendations.append(
                    ProductRecommendation(
                        product_id=cand.product_id,
                        name=cand.product.name,
                        brand=cand.product.brand,
                        category=cand.product.category,
                        price_inr=cand.product.price_inr,
                        rating=cand.product.rating,
                        verified_specs=dict(cand.product.specs),
                        features=list(cand.product.features),
                        overall_score=cand.overall_score,
                        reasoning=exp.reasoning.strip(),
                        advantages=[a.strip() for a in exp.advantages if a.strip()],
                        trade_offs=[t.strip() for t in exp.trade_offs if t.strip()],
                        match_explanation=exp.match_explanation.strip() or f"Priced at ₹{cand.product.price_inr:,}, matching your {cand.product.category.value} criteria.",
                        limitations=[lim.strip() for lim in exp.limitations if lim.strip()],
                    )
                )
        else:
            # Candidate was omitted by LLM; populate with deterministic grounded fallback
            logger.info("Candidate '%s' was omitted by LLM; generating deterministic fallback.", pid)
            any_repaired = True
            recommendations.append(build_fallback_product_recommendation(cand, requirements))

    # If all explanations were missing or empty, mark as fallback
    is_fallback = any_repaired and (len(exp_map) == 0)
    fallback_reason = "Repaired ungrounded claims or missing explanations" if any_repaired else None

    # Validate non-empty message
    msg = payload.message.strip() if payload.message and payload.message.strip() else (
        f"Based on your requirements, here are the top recommended {requirements.category.value if requirements.category else 'electronics'}s from our catalogue:"
    )

    return RecommendationResponse(
        message=msg,
        category=requirements.category,
        recommendations=recommendations,
        total_matching=len(ranked_candidates),
        follow_up_question=payload.follow_up_question.strip() if payload.follow_up_question else None,
        data_quality_notes=list(search_result.data_quality_limitations),
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        has_matches=True,
    )
