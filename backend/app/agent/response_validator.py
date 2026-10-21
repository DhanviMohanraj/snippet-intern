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


_NEGATION_RE = re.compile(
    r"\b(?:not|no|without|lacks?|lacking|isn't|aren't|doesn't|don't|cannot|can't|"
    r"unverified|unconfirmed|unavailable|missing|absent|neither|nor)\b"
)


def _affirmative_text(text: str) -> str:
    """Return only the sentences that do not negate or hedge a feature.

    Feature-presence checks (ANC, stylus, flicker-free, certifications, colour
    gamut, dedicated GPU) must not fire on honest statements such as
    "does not include ANC" or "sRGB coverage is not verified" — previously those
    triggered a contradiction and discarded a correct LLM explanation.
    """
    sentences = re.split(r"(?<=[.!?;])\s+|\n+", text.lower())
    kept = [s for s in sentences if s.strip() and not _NEGATION_RE.search(s)]
    # Normalise hyphens so "low-blue-light" and "low blue light" compare equal.
    return " " + " ".join(kept).replace("-", " ") + " "


def _as_sentence(text: str) -> str:
    """Return text as exactly one sentence: trimmed, with a single terminal full stop."""
    cleaned = re.sub(r"[\s;,.]+$", "", text.strip())
    return f"{cleaned}." if cleaned else ""


def _collect_unverified_limitations(
    product: Product,
    requirements: Requirements,
    candidate: RankedCandidate | None = None,
) -> list[str]:
    """Collect unverified or missing specifications relevant to user requirements."""
    limitations: list[str] = []
    features_lower = " ".join(product.features).lower()

    if candidate:
        for u in candidate.evaluation.unverifiable_constraints:
            if u.reason not in limitations:
                limitations.append(u.reason)
        for unsup in candidate.score_breakdown.unsupported_preferences:
            msg = f"'{unsup}' preference cannot be verified from catalogue specifications."
            if msg not in limitations:
                limitations.append(msg)

    # Category-specific verification audits
    if product.category == Category.MONITOR:
        if (requirements.hard_constraints.requires_eye_comfort or "eye_comfort" in requirements.preferences) and not any(
            k in features_lower for k in ("low blue light", "eye comfort", "eye care", "tuv")
        ):
            msg = "Eye-comfort / Low blue light certification is not verified in the catalogue for this model."
            if msg not in limitations:
                limitations.append(msg)

        if (requirements.hard_constraints.requires_flicker_free or "flicker_free" in requirements.preferences or "eye_comfort" in requirements.preferences) and not any(
            k in features_lower for k in ("flicker-free", "flicker free", "anti-flicker")
        ):
            msg = "Flicker-free operation is not verified in the catalogue for this model."
            if msg not in limitations:
                limitations.append(msg)

        if any(k in features_lower for k in ("srgb", "dci-p3", "delta e")):
            pass
        else:
            msg = "Exact colour gamut coverage (e.g. sRGB / DCI-P3 percentage) is not verified in the catalogue."
            if msg not in limitations:
                limitations.append(msg)

    elif product.category == Category.TABLET:
        if (requirements.hard_constraints.requires_stylus or "stylus" in requirements.preferences):
            has_bundled = any("includes" in f.lower() and ("pen" in f.lower() or "stylus" in f.lower()) for f in product.features)
            if not has_bundled:
                msg = "Bundled stylus inclusion is not verified in the catalogue (may require separate purchase)."
                if msg not in limitations:
                    limitations.append(msg)

    elif product.category == Category.HEADPHONES:
        if requirements.hard_constraints.requires_anc and product.specs.get("anc") is not True:
            msg = "Active Noise Cancellation (ANC) is not present on this model."
            if msg not in limitations:
                limitations.append(msg)

    elif product.category == Category.PHONE:
        if "battery" in requirements.preferences or any("battery" in u for u in requirements.use_cases):
            mah = product.specs.get("battery_mah")
            msg = f"Hourly battery runtime is not verified in the catalogue (device features a {mah}mAh battery)." if mah else "Battery runtime is not verified in the catalogue."
            if msg not in limitations:
                limitations.append(msg)

    return limitations


def _build_comparison_summary(
    recommendations: Sequence[ProductRecommendation],
    requirements: Requirements,
) -> str | None:
    """Generate a concise, factual comparison across top recommended products."""
    if len(recommendations) < 2:
        return None
    r1, r2 = recommendations[0], recommendations[1]
    p1, p2 = r1.price_inr, r2.price_inr
    diff = abs(p1 - p2)

    parts: list[str] = []
    if diff > 0:
        cheaper = r1 if p1 < p2 else r2
        costlier = r2 if p1 < p2 else r1
        parts.append(f"{cheaper.name} is priced at ₹{cheaper.price_inr:,} (₹{diff:,} more affordable than {costlier.name} at ₹{costlier.price_inr:,})")
    else:
        parts.append(f"Both {r1.name} and {r2.name} are priced identically at ₹{p1:,}")

    # Key spec difference
    if r1.category == Category.MONITOR:
        s1 = r1.verified_specs.get("size_in") or r1.verified_specs.get("screen_in", "standard")
        s2 = r2.verified_specs.get("size_in") or r2.verified_specs.get("screen_in", "standard")
        hz1 = r1.verified_specs.get("refresh_hz", 60)
        hz2 = r2.verified_specs.get("refresh_hz", 60)
        parts.append(f"{r1.name} offers a {s1}\" panel at {hz1}Hz, while {r2.name} offers a {s2}\" panel at {hz2}Hz")
    elif r1.category == Category.LAPTOP:
        ram1 = r1.verified_specs.get("ram_gb", "standard")
        ram2 = r2.verified_specs.get("ram_gb", "standard")
        parts.append(f"{r1.name} provides {ram1}GB RAM compared to {r2.name}'s {ram2}GB RAM")
    elif r1.category == Category.PHONE:
        stor1 = r1.verified_specs.get("storage_gb", "standard")
        stor2 = r2.verified_specs.get("storage_gb", "standard")
        parts.append(f"{r1.name} has {stor1}GB storage versus {r2.name}'s {stor2}GB storage")

    return ". ".join(parts) + "."


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

    # 1. Build concise, grounded reasoning.
    # Each entry in candidate.reasons is already a full sentence ending in ".";
    # joining them with "; " and appending "." produced "budget.; Strong ... score..".
    # The match score is shown on the card itself, so it is not repeated in prose.
    use_cases = [u.replace("_", " ") for u in requirements.use_cases if u.strip()]
    if len(use_cases) > 1:
        use_case_str = f"{', '.join(use_cases[:-1])} and {use_cases[-1]}"
    elif use_cases:
        use_case_str = use_cases[0]
    else:
        use_case_str = ""
    opening = (
        f"A strong fit for {use_case_str} use." if use_case_str
        else "A strong fit for your stated requirements."
    )
    reason_sentences = [_as_sentence(r) for r in candidate.reasons[:2] if r and r.strip()]
    reasoning = " ".join([opening, *reason_sentences])

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

    elif p.category == Category.MONITOR:
        hz = p.specs.get("refresh_hz", 60)
        if hz <= 75 and any("gaming" in u.lower() for u in requirements.use_cases):
            trade_offs.append(f"Refresh rate of {hz}Hz is adequate for office and casual gaming, but not high-frame-rate competitive esports.")

    if not trade_offs:
        # Previously a generic filler ("Priced near the upper threshold of its
        # performance bracket") that no catalogue field supports.
        trade_offs.append("No notable trade-offs identified from the catalogue specifications.")

    # 4. Build match explanation from passed constraints
    passed_labels = [
        c.reason for c in candidate.evaluation.passed_constraints
        if c.name in ("budget_max", "min_ram_gb", "min_storage_gb", "requires_gpu", "min_battery_hr", "screen_size_class")
    ]
    if passed_labels:
        match_explanation = " ".join(_as_sentence(lbl) for lbl in passed_labels[:3])
    else:
        match_explanation = f"Priced at ₹{p.price_inr:,}, matching your category requirements."

    # 5. Build limitations from unverifiable constraints or missing specs
    limitations = _collect_unverified_limitations(p, requirements, candidate)

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
    evidence_summaries: dict[str, Any] | None = None,
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
        # Check if there are candidates that satisfied non-budget constraints but exceeded budget
        over_budget = [
            ex for ex in search_result.excluded_candidates
            if any("budget_max" in r.lower() for r in ex.rejection_reasons)
            and all("budget_max" in r.lower() or "budget_min" in r.lower() for r in ex.rejection_reasons)
        ]

        if over_budget and requirements.budget_max_inr is not None:
            cheapest = min(over_budget, key=lambda x: x.product.price_inr)
            diff = cheapest.product.price_inr - requirements.budget_max_inr
            msg = (
                f"I could not find any {cat_name}s in our catalogue that satisfy all your criteria within your budget of ₹{requirements.budget_max_inr:,}. "
                f"The lowest-priced matching model is the {cheapest.product.name} at ₹{cheapest.product.price_inr:,} (₹{diff:,} above budget). "
                f"Would you like to increase your budget to ₹{cheapest.product.price_inr:,} or consider relaxing specific constraints?"
            )
        else:
            excluded_reasons: list[str] = []
            for ex in search_result.excluded_candidates[:3]:
                for r in ex.rejection_reasons[:1]:
                    excluded_reasons.append(f"{ex.product.name}: {r}")

            reasons_summary = f" ({'; '.join(excluded_reasons)})" if excluded_reasons else ""
            msg = (
                f"I could not find any {cat_name}s in our catalogue that satisfy all your non-negotiable criteria{reasons_summary}. "
                "Please consider adjusting your budget or relaxing specific constraints."
            )

        return RecommendationResponse(
            message=msg,
            category=requirements.category,
            recommendations=[],
            total_matching=0,
            follow_up_question="Would you like to increase your budget or adjust any hardware specifications?",
            data_quality_notes=list(search_result.data_quality_limitations),
            comparison_summary=None,
            remaining_uncertainties=[],
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

    remaining_unc = list(dict.fromkeys(
        lim for rec in recommendations for lim in rec.limitations
    ))

    return RecommendationResponse(
        message=intro_message,
        category=requirements.category,
        recommendations=recommendations,
        total_matching=len(search_result.ranked_candidates),
        follow_up_question="Would you like more details on any of these options, or should we compare specific models?",
        data_quality_notes=list(search_result.data_quality_limitations),
        comparison_summary=_build_comparison_summary(recommendations, requirements),
        remaining_uncertainties=remaining_unc,
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
    evidence_summary: Any | None = None,
    all_candidates: Sequence[RankedCandidate] | None = None,
) -> list[str]:
    """Inspect generated text for factual contradictions against verified product specifications and retrieved evidence.

    Detects:
        - Price contradictions (stating a price that contradicts the catalogue).
        - RAM size contradictions (claiming an incorrect amount of RAM).
        - Storage size contradictions (claiming an incorrect storage size).
        - Weight contradictions (claiming an incorrect weight).
        - Dedicated GPU claims on integrated graphics laptops.
        - Runtime claims on products lacking verified battery_hr specifications.
        - Monitor weight claims when catalogue does not track monitor weight.
        - Active Noise Cancellation claims on passive headphones.
        - Bundled stylus claims on tablets without confirmed stylus inclusion.
        - Color gamut / calibration claims not in product features.
        - Claims asserting attributes known to be unknown or missing in catalogue.
        - Cross-product feature transfer from other recommended candidates.

    Args:
        text: The generated text (reasoning, advantages, match_explanation).
        product: Authoritative Product catalogue record.
        requirements: User requirements for budget reference.
        evidence_summary: Optional ProductEvidenceSummary containing verified facts and unknown attributes.
        all_candidates: Optional list of other ranked candidates to detect cross-product spec transfer.

    Returns:
        List of detected contradiction descriptions. Empty if text is grounded.
    """
    contradictions: list[str] = []
    text_lower = text.lower()
    # Affirmative-only, hyphen-normalised text for feature-presence checks.
    claims = _affirmative_text(text)
    features_norm = " ".join(product.features).lower().replace("-", " ")

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
    # Unit is captured per match: the old check converted every number to TB
    # whenever "tb" appeared anywhere in the text.
    storage_matches = re.findall(r"(\d+)\s*(gb|tb)\s*(?:of\s*)?(?:ssd|hdd|storage)", text_lower)
    cat_storage = product.specs.get("storage_gb")
    for sm, unit in storage_matches:
        num = int(sm)
        num_gb = num * 1024 if unit == "tb" else num
        if cat_storage is not None and abs(num_gb - cat_storage) > 30 and not (
            unit == "tb" and abs(num * 1000 - cat_storage) <= 30
        ):
            contradictions.append(
                f"Contradictory storage claim: text mentions {sm}{unit.upper()}, but catalogue specifies {cat_storage}GB."
            )

    # 4. Discrete GPU claims on integrated graphics
    if product.category == Category.LAPTOP:
        vram = product.specs.get("vram_gb", 0)
        gpu = product.specs.get("gpu", "").lower()
        if vram == 0 and not any(k in gpu for k in ("rtx", "gtx", "radeon rx", "arc a")):
            if any(term in claims for term in ("dedicated rtx", "discrete rtx", "dedicated nvidia", "dedicated gpu", "discrete gpu", "geforce rtx")):
                contradictions.append(
                    f"Contradictory GPU claim: text claims dedicated GPU, but product has integrated graphics ('{product.specs.get('gpu')}')."
                )

    # 5. Fabricated battery runtime on products lacking battery_hr
    if "battery_hr" not in product.specs:
        hr_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s*)?battery", text_lower)
        if not hr_matches:
            hr_matches = re.findall(r"battery\s*(?:life|runtime)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", text_lower)
        if hr_matches:
            if product.category == Category.PHONE:
                contradictions.append(
                    f"Contradictory battery claim: text claims {hr_matches[0]} hours battery runtime, but catalogue lacks battery_hr specification (phone catalogue specifies battery_mah only)."
                )
            else:
                contradictions.append(
                    f"Contradictory battery claim: text claims {hr_matches[0]} hours battery runtime, but catalogue lacks battery_hr specification."
                )
    else:
        # 5b. Battery runtime present in catalogue: the stated hours must match it.
        cat_hr = float(product.specs["battery_hr"])
        hr_claims = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", text_lower)
        for hc in hr_claims:
            val = float(hc)
            # Ignore small numbers that are usage durations ("8 hours of coding daily").
            if val >= 3 and abs(val - cat_hr) > 0.5 and re.search(
                rf"{re.escape(hc)}\s*(?:hours?|hrs?)\s*(?:of\s*)?(?:battery|runtime|backup)|"
                rf"(?:battery|runtime|backup)[^.]{{0,40}}?{re.escape(hc)}\s*(?:hours?|hrs?)",
                text_lower,
            ):
                contradictions.append(
                    f"Contradictory battery claim: text mentions {hc} hours, but catalogue specifies {cat_hr:g} hours."
                )

    # 5c. Weight values must match the catalogue when it records them.
    cat_kg = product.specs.get("weight_kg")
    if cat_kg is not None and product.category != Category.MONITOR:
        for wk in re.findall(r"(\d+(?:\.\d+)?)\s*kg\b", text_lower):
            if abs(float(wk) - float(cat_kg)) > 0.05:
                contradictions.append(
                    f"Contradictory weight claim: text mentions {wk}kg, but catalogue specifies {cat_kg}kg."
                )
    cat_g = product.specs.get("weight_g")
    if cat_g is not None:
        for wg in re.findall(r"(\d{2,4})\s*(?:g|grams)\b", text_lower):
            if abs(float(wg) - float(cat_g)) > 3:
                contradictions.append(
                    f"Contradictory weight claim: text mentions {wg}g, but catalogue specifies {cat_g}g."
                )

    # 6. Fabricated monitor weight
    if product.category == Category.MONITOR:
        weight_matches = re.findall(r"(\d+(?:\.\d+)?)\s*kg", text_lower)
        if weight_matches:
            contradictions.append(
                f"Contradictory monitor weight claim: text mentions {weight_matches[0]}kg, but catalogue does not track monitor weight."
            )

    # 7. Fabricated Active Noise Cancellation (ANC) on headphones lacking ANC
    if product.category == Category.HEADPHONES and product.specs.get("anc") is not True:
        if any(term in claims for term in ("active noise cancellation", "active noise cancelling", " anc ", "with anc", " anc,", " anc.")):
            contradictions.append(
                f"Contradictory ANC claim: text claims active noise cancellation, but catalogue specifies anc={product.specs.get('anc')}."
            )

    # 8. Fabricated bundled stylus on tablets
    if product.category == Category.TABLET:
        if any(term in claims for term in ("includes stylus", "comes with stylus", "bundled stylus", "includes pen", "comes with pen", "bundled pen", "includes apple pencil", "with apple pencil", "with stylus", "stylus included", "pen included")):
            if not any("includes" in f.lower() and ("pen" in f.lower() or "stylus" in f.lower()) for f in product.features):
                contradictions.append(
                    "Contradictory stylus claim: text claims stylus is included, but catalogue does not confirm bundled stylus."
                )

    # 9. Fabricated color gamut calibration claims
    if any(term in claims for term in ("srgb", "dci p3", "adobe rgb", "delta e", "factory calibrated", "color accurate", "colour accurate", "calibrated")):
        spec_str = " ".join(str(v) for v in product.specs.values()).lower().replace("-", " ")
        if not any(term in (features_norm + " " + spec_str) for term in ("srgb", "dci p3", "adobe rgb", "delta e", "calibrat", "color accurate")):
            contradictions.append(
                "Contradictory color / colour accuracy claim: text asserts verified color calibration / gamut, but this is not verified in catalogue features."
            )

    # 10. Eye-comfort / low blue light claims.
    # Any affirmative low-blue-light / TÜV / eye-care claim needs a matching
    # catalogue feature; a *certification* claim additionally needs the feature
    # itself to say "certified". The old check only matched phrases containing
    # "certified", so "low-blue-light certification" slipped through.
    # "eye comfort" alone is not listed: "you prioritised eye comfort" describes the
    # user's preference, not a product claim.
    eye_terms = (
        "low blue light", "blue light filter", " tuv", "tüv", "eye care",
        "eye comfort certif", "eye comfort feature", "reduces eye strain",
        "reduce eye strain", "less eye strain",
    )
    if any(term in claims for term in eye_terms):
        eye_feature_listed = any(k in features_norm for k in ("low blue light", "eye comfort", "eye care", "tuv", "tüv"))
        claims_certification = re.search(r"certifi|tuv|tüv", claims) is not None
        feature_certified = re.search(r"certifi|tuv|tüv", features_norm) is not None
        if not eye_feature_listed or (claims_certification and not feature_certified):
            contradictions.append(
                "Contradictory certification claim: text asserts eye-comfort / low blue light certification, but catalogue features do not verify this certification."
            )

    # 11. Fabricated flicker-free operation claims
    if any(term in claims for term in ("flicker free", "anti flicker", "flicker less")):
        if not any(k in features_norm for k in ("flicker free", "anti flicker")):
            contradictions.append(
                "Contradictory display claim: text asserts flicker-free operation, but catalogue features do not verify flicker-free / anti-flicker operation."
            )

    # 12. Screen size contradiction
    cat_size = product.specs.get("size_in") or product.specs.get("screen_in")
    if cat_size is not None:
        size_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:-?\s*inch|(?:\s*|\b)in\b|(?:\s*|\b)inches\b|\")", text_lower)
        for sm in size_matches:
            try:
                num = float(sm)
                # Ignore sizes outside sensible display range or sizes that refer to something else
                if 5.0 <= num <= 50.0 and abs(num - float(cat_size)) > 0.8:
                    contradictions.append(
                        f"Contradictory screen size claim: text mentions {num}-inch display, but catalogue specifies {cat_size} inches."
                    )
            except ValueError:
                continue

    # 13. Resolution contradiction
    cat_res = str(product.specs.get("resolution", "")).lower()
    if any(term in text_lower for term in ("4k", "uhd", "3840x2160")) and not any(k in cat_res for k in ("4k", "uhd", "3840")):
        contradictions.append(
            f"Contradictory resolution claim: text claims 4K/UHD, but catalogue specifies {product.specs.get('resolution')}."
        )
    if any(term in text_lower for term in ("1440p", "2k", "qhd", "2560x1440")) and not any(k in cat_res for k in ("1440", "2k", "qhd", "2560")):
        contradictions.append(
            f"Contradictory resolution claim: text claims 1440p/2K/QHD, but catalogue specifies {product.specs.get('resolution')}."
        )

    # 14. Refresh rate contradiction
    cat_refresh = product.specs.get("refresh_hz")
    if cat_refresh is not None:
        hz_matches = re.findall(r"(\d+)\s*hz", text_lower)
        for hm in hz_matches:
            try:
                num = int(hm)
                if abs(num - cat_refresh) > 5:
                    contradictions.append(
                        f"Contradictory refresh rate claim: text mentions {num}Hz, but catalogue specifies {cat_refresh}Hz."
                    )
            except ValueError:
                continue

    # 15. Unknown or unverified attribute claims based on RAG evidence summary
    if evidence_summary is not None and getattr(evidence_summary, "unknown_attributes", None):
        for unk in evidence_summary.unknown_attributes:
            if unk == "stylus_support" and any(kw in claims for kw in ("stylus support", "supports stylus", "pen support", "apple pencil", "s pen")):
                contradictions.append(
                    "Contradictory claim: text asserts stylus support, but catalogue record does not specify stylus_support."
                )
            elif unk == "anc" and any(kw in claims for kw in ("active noise cancellation", "with anc", "active noise cancelling")):
                contradictions.append(
                    "Contradictory claim: text asserts Active Noise Cancellation, but catalogue lacks verified anc specification."
                )
            elif unk in ("gpu", "vram_gb") and any(kw in claims for kw in ("dedicated rtx", "discrete gpu", "dedicated gpu", "nvidia geforce")):
                contradictions.append(
                    f"Contradictory claim: text asserts dedicated GPU, but catalogue lacks '{unk}' specification."
                )

    # 16. Cross-product specification transfer validation
    if all_candidates:
        for other in all_candidates:
            if other.product_id == product.product_id:
                continue
            other_name = other.product.name.lower()
            if other_name in text_lower and product.name.lower() not in text_lower:
                contradictions.append(
                    f"Cross-product hallucination: explanation for '{product.product_id}' refers to other product '{other.product.name}'."
                )
            other_gpu = str(other.product.specs.get("gpu", "")).strip().lower()
            this_gpu = str(product.specs.get("gpu", "")).strip().lower()
            if other_gpu and "integrated" in this_gpu and other_gpu in text_lower:
                contradictions.append(
                    f"Cross-product spec transfer: explanation transfers dedicated GPU '{other.product.specs.get('gpu')}' from {other.product_id}."
                )

    return contradictions


# ---------------------------------------------------------------------------
# 3. Comprehensive Response Validator
# ---------------------------------------------------------------------------

def validate_recommendation_response(
    payload: LLMRecommendationPayload,
    search_result: CandidateSearchResult,
    requirements: Requirements,
    evidence_summaries: dict[str, Any] | None = None,
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
        evidence_summaries: Optional mapping of product_id to ProductEvidenceSummary.

    Returns:
        Fully validated, authoritative RecommendationResponse.
    """
    ranked_candidates = search_result.ranked_candidates

    # If no eligible candidates exist, immediately return no-match fallback
    if not search_result.has_matches or not ranked_candidates:
        return build_fallback_recommendation_response(
            requirements,
            search_result,
            fallback_reason="No eligible candidates found.",
            evidence_summaries=evidence_summaries,
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
        ev_summary = (evidence_summaries or {}).get(pid)
        if pid in exp_map:
            exp = exp_map[pid]
            # Check for factual / numerical contradictions in LLM text
            combined_text = f"{exp.reasoning} {' '.join(exp.advantages)} {exp.match_explanation} {' '.join(exp.trade_offs)}"
            contradictions = check_claim_contradictions(
                text=combined_text,
                product=cand.product,
                requirements=requirements,
                evidence_summary=ev_summary,
                all_candidates=ranked_candidates,
            )

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
                # Ensure all unverified limitations are preserved
                unverified_lims = _collect_unverified_limitations(cand.product, requirements, cand)
                merged_limitations = list(dict.fromkeys(
                    [lim.strip() for lim in exp.limitations if lim.strip()] + unverified_lims
                ))

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
                        limitations=merged_limitations,
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

    remaining_unc = list(dict.fromkeys(
        (payload.remaining_uncertainties or []) + [lim for r in recommendations for lim in r.limitations]
    ))

    comparison_summary = payload.comparison_summary.strip() if payload.comparison_summary else _build_comparison_summary(recommendations, requirements)

    return RecommendationResponse(
        message=msg,
        category=requirements.category,
        recommendations=recommendations,
        total_matching=len(ranked_candidates),
        follow_up_question=payload.follow_up_question.strip() if payload.follow_up_question else None,
        data_quality_notes=list(search_result.data_quality_limitations),
        comparison_summary=comparison_summary,
        remaining_uncertainties=remaining_unc,
        is_fallback=is_fallback,
        fallback_reason=fallback_reason,
        has_matches=True,
    )
