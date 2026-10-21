"""Comprehensive regression test suite for SpecSage recommendation quality, evidence grounding,
requested count enforcement, claim validation, deterministic ranking, and live reproduction.

Covers:
    Section 3: Requested count and requirement state (1-7)
    Section 6: Claim validation and product-level evidence isolation (8-18)
    Section 4 & 7: Ranking, hard filtering, unknown attributes, and diversity (19-25)
    Section 5 & 8: Cross-category coverage (laptops, phones, headphones, tablets, monitors) (26-30)
    Section 11: Exact monitor reproduction scenario
"""

from __future__ import annotations

import pytest

from app.agent.extract import _apply_deterministic_fallbacks, extract
from app.agent.followup import (
    FollowUpIntent,
    analyze_catalogue_minimum_budget,
    detect_followup_intent,
)
from app.agent.merge import merge
from app.agent.normalize import normalize
from app.agent.ranking import (
    evaluate_candidate,
    filter_candidates,
    rank_candidates,
    score_candidate,
    search_and_rank,
)
from app.agent.response_validator import (
    build_fallback_product_recommendation,
    build_fallback_recommendation_response,
    check_claim_contradictions,
    validate_recommendation_response,
)
from app.llm.fake import FakeLLMClient
from app.models.product import Category, Product
from app.models.ranking import CandidateSearchResult, ConstraintStatus, RankedCandidate
from app.models.requirements import (
    HardConstraints,
    Requirements,
    RequirementsDelta,
)
from app.models.response import (
    LLMProductExplanation,
    LLMRecommendationPayload,
    ProductRecommendation,
    RecommendationResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor_catalog():
    """Diverse set of monitor fixtures covering verified and unverified specifications."""
    return [
        Product(
            product_id="MON-0005",
            name="Zenit OfficeClear 24",
            brand="Zenit",
            category=Category.MONITOR,
            price_inr=9499,
            rating=4.2,
            specs={
                "size_in": 23.8,
                "resolution": "1920x1080",
                "panel": "IPS",
                "refresh_hz": 75,
                "ports": ["HDMI", "VGA"],
            },
            features=["3-Side Frameless Bezel", "Low Blue Light Certified"],
            intended_use=["coding", "office", "general"],
            in_stock=True,
        ),
        Product(
            product_id="MON-0007",
            name="Nexa Essential 24",
            brand="Nexa",
            category=Category.MONITOR,
            price_inr=7999,
            rating=4.0,
            specs={
                "size_in": 24.0,
                "resolution": "1920x1080",
                "panel": "VA",
                "refresh_hz": 60,
                "ports": ["HDMI", "VGA"],
            },
            features=["Tilt Adjustable Stand", "VESA Compatible"],
            intended_use=["office", "casual"],
            in_stock=True,
        ),
        Product(
            product_id="MON-0015",
            name="Zenit UltraSlim 27",
            brand="Zenit",
            category=Category.MONITOR,
            price_inr=14499,
            rating=4.3,
            specs={
                "size_in": 27.0,
                "resolution": "1920x1080",
                "panel": "IPS",
                "refresh_hz": 100,
                "ports": ["HDMI", "DisplayPort"],
            },
            features=["Ultra-slim Profile", "Built-in Speakers"],
            intended_use=["coding", "multimedia", "casual gaming"],
            in_stock=True,
        ),
        Product(
            product_id="MON-0001",
            name="AeroView Ultra 27 4K",
            brand="AeroView",
            category=Category.MONITOR,
            price_inr=28999,
            rating=4.6,
            specs={
                "size_in": 27.0,
                "resolution": "3840x2160",
                "panel": "IPS",
                "refresh_hz": 60,
                "ports": ["HDMI 2.0", "DisplayPort 1.4", "USB-C 65W PD"],
            },
            features=["100% sRGB", "Anti-Flicker", "Low Blue Light", "Height Adjustable Stand"],
            intended_use=["coding", "content creation", "office"],
            in_stock=True,
        ),
    ]


@pytest.fixture
def laptop_fixture():
    return Product(
        product_id="LAP-TEST-1",
        name="ZenPro 14",
        brand="Zenit",
        category=Category.LAPTOP,
        price_inr=62000,
        rating=4.5,
        specs={
            "ram_gb": 16,
            "storage_gb": 512,
            "cpu": "Intel Core i5-13500H",
            "gpu": "Integrated Intel Iris Xe",
            "vram_gb": 0,
            "battery_hr": 9.5,
            "weight_kg": 1.35,
        },
        features=["IPS Display", "Thunderbolt 4"],
        intended_use=["coding", "productivity"],
        in_stock=True,
    )


@pytest.fixture
def phone_fixture():
    return Product(
        product_id="PHN-TEST-1",
        name="Aura Neo 5G",
        brand="Aura",
        category=Category.PHONE,
        price_inr=24999,
        rating=4.3,
        specs={
            "ram_gb": 8,
            "storage_gb": 128,
            "battery_mah": 5000,
            "screen_in": 6.5,
            "weight_g": 182,
        },
        features=["120Hz AMOLED", "33W Fast Charging"],
        intended_use=["everyday", "movies"],
        in_stock=True,
    )


@pytest.fixture
def headphones_fixture():
    return Product(
        product_id="HDP-TEST-1",
        name="BassWave 300",
        brand="SoundCore",
        category=Category.HEADPHONES,
        price_inr=3999,
        rating=4.1,
        specs={
            "type": "over-ear",
            "wireless": True,
            "anc": False,
            "battery_hr": 30.0,
        },
        features=["Deep Bass", "Foldable Design"],
        intended_use=["music", "casual"],
        in_stock=True,
    )


@pytest.fixture
def tablet_fixture():
    return Product(
        product_id="TAB-TEST-1",
        name="TabAir 11",
        brand="Nova",
        category=Category.TABLET,
        price_inr=29999,
        rating=4.4,
        specs={
            "storage_gb": 128,
            "screen_in": 11.0,
            "stylus_support": True,
        },
        features=["2K 90Hz Display", "Quad Speakers"],
        intended_use=["drawing", "reading", "entertainment"],
        in_stock=True,
    )


# ---------------------------------------------------------------------------
# Section 3: Requested Count & Requirement State Tests (1-7)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_1_requested_top_2_returns_at_most_2(monitor_catalog):
    """1. A request for top 2 returns at most 2 products even when more qualify."""
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=20000,
        requested_result_count=2,
    )
    result = await search_and_rank(req, monitor_catalog, top_n=5)
    assert result.has_matches is True
    assert len(result.ranked_candidates) <= 2


@pytest.mark.asyncio
async def test_2_requested_5_returns_up_to_5_eligible(monitor_catalog):
    """2. A request for 5 returns up to 5 eligible products if available."""
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=35000,
        requested_result_count=5,
    )
    result = await search_and_rank(req, monitor_catalog, top_n=5)
    assert len(result.ranked_candidates) == len(monitor_catalog)


def test_3_changed_result_count_updates_existing_conversation():
    """3. Changing the requested result count updates existing conversation state."""
    existing = Requirements(
        category=Category.MONITOR,
        budget_max_inr=15000,
        requested_result_count=2,
    )
    delta = RequirementsDelta(requested_result_count=4)
    merged, _ = merge(existing, delta)
    assert merged.requested_result_count == 4
    assert merged.budget_max_inr == 15000


@pytest.mark.asyncio
async def test_4_optional_features_remain_soft_preferences():
    """4. 'Eye-comfort would be nice to have' is extracted as a soft preference, not hard constraint."""
    fake = FakeLLMClient(json_responses=[{
        "category": "monitor",
        "preferences": {"eye_comfort": 0.8},
    }])
    delta = await extract(
        "I'm looking for a monitor. Eye-comfort certification would be nice to have.",
        Requirements(),
        [],
        fake,
    )
    assert "eye_comfort" in (delta.preferences or {})
    assert delta.hard is None or delta.hard.requires_eye_comfort is not True


@pytest.mark.asyncio
async def test_5_mandatory_features_remain_hard_constraints():
    """5. 'Eye-comfort is required' is extracted as a hard constraint."""
    fake = FakeLLMClient(json_responses=[{
        "category": "monitor",
        "budget_max_inr": 20000,
        "hard": {"requires_eye_comfort": True},
    }])
    delta = await extract(
        "I need a monitor under 20000. Eye-comfort certification is required.",
        Requirements(),
        [],
        fake,
    )
    assert delta.hard is not None
    assert delta.hard.requires_eye_comfort is True


@pytest.mark.asyncio
async def test_6_negated_requirements_not_misinterpreted():
    """6. Negations ('not for gaming', 'no dedicated gpu') are not treated as positive requirements."""
    fake = FakeLLMClient(json_responses=[{
        "category": "laptop",
        "budget_max_inr": 60000,
        "hard": {"requires_gpu": False},
        "use_cases": ["office"],
    }])
    delta = await extract(
        "I want a laptop under 60000, but not for gaming and no dedicated GPU needed.",
        Requirements(),
        [],
        fake,
    )
    if delta.hard:
        assert delta.hard.requires_gpu is not True
    assert "gaming" not in delta.use_cases


def test_7_requirements_survive_follow_up_turns():
    """7. Requirements survive subsequent turns unless explicitly modified or cleared."""
    state = Requirements(
        category=Category.MONITOR,
        budget_max_inr=15000,
        use_cases=["coding"],
        requested_result_count=2,
    )
    follow_up_delta = RequirementsDelta(
        preferences={"display_quality": 0.8}
    )
    merged, _ = merge(state, follow_up_delta)
    assert merged.category == Category.MONITOR
    assert merged.budget_max_inr == 15000
    assert "programming" in merged.use_cases or "coding" in merged.use_cases
    assert merged.requested_result_count == 2
    assert "display_quality" in merged.preferences


# ---------------------------------------------------------------------------
# Section 6: Claim Validation & Evidence Grounding (8-18)
# ---------------------------------------------------------------------------

def test_8_monitor_without_verified_low_blue_light_rejected(monitor_catalog):
    """8. Monitor without verified low-blue-light certification cannot claim it."""
    mon_nexa = monitor_catalog[1]
    claimed_text = "The Nexa Essential 24 includes certified low blue light protection for eye health."
    contradictions = check_claim_contradictions(claimed_text, mon_nexa, Requirements(category=Category.MONITOR))
    assert any("certification" in c.lower() for c in contradictions)


def test_9_monitor_without_verified_flicker_free_rejected(monitor_catalog):
    """9. Monitor without verified flicker-free operation cannot claim it."""
    mon_zenit = monitor_catalog[0]
    claimed_text = "This monitor features flicker-free backlighting to prevent eye fatigue."
    contradictions = check_claim_contradictions(claimed_text, mon_zenit, Requirements(category=Category.MONITOR))
    assert any("flicker-free" in c.lower() or "anti-flicker" in c.lower() for c in contradictions)


def test_10_missing_certification_reported_as_unknown_not_absent(monitor_catalog):
    """10. Missing certification is reported as not verified in catalogue, not confirmed absent."""
    mon_zenit = monitor_catalog[0]
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=15000,
        hard=HardConstraints(requires_flicker_free=False),
        preferences={"flicker_free": 0.8},
    )
    cands = rank_candidates([mon_zenit], req, top_n=1)
    rec = build_fallback_product_recommendation(cands[0], req)
    assert any("not verified" in lim.lower() for lim in rec.limitations)


def test_11_cross_product_spec_transfer_rejected(laptop_fixture):
    """11. Another product's dedicated GPU cannot be attributed to an integrated graphics laptop."""
    other_laptop = Product(
        product_id="LAP-RTX-2",
        name="GamerX 15",
        brand="GamerTech",
        category=Category.LAPTOP,
        price_inr=85000,
        specs={"gpu": "NVIDIA GeForce RTX 4060", "vram_gb": 8},
        features=["RTX 4060 GPU"],
        intended_use=["gaming"],
        in_stock=True,
    )
    req = Requirements(category=Category.LAPTOP)
    cands = rank_candidates([laptop_fixture, other_laptop], req, top_n=2)

    hallucinated_text = "The ZenPro 14 comes equipped with NVIDIA GeForce RTX 4060 graphics."
    contradictions = check_claim_contradictions(
        hallucinated_text,
        laptop_fixture,
        req,
        all_candidates=cands,
    )
    assert any("gpu" in c.lower() for c in contradictions)


def test_12_phone_battery_hours_claim_rejected(phone_fixture):
    """12. Claiming precise battery hours for a phone with only battery_mah is flagged."""
    text = "The Aura Neo 5G gives you 28 hours of battery runtime on a single charge."
    contradictions = check_claim_contradictions(text, phone_fixture, Requirements(category=Category.PHONE))
    assert any("battery_hr" in c.lower() or "battery" in c.lower() for c in contradictions)


def test_13_missing_anc_data_rejects_anc_claim(headphones_fixture):
    """13. Claiming Active Noise Cancellation on passive headphones is flagged."""
    text = "Enjoy quiet travels with the BassWave 300 Active Noise Cancellation (ANC)."
    contradictions = check_claim_contradictions(text, headphones_fixture, Requirements(category=Category.HEADPHONES))
    assert any("anc" in c.lower() for c in contradictions)


def test_14_missing_stylus_inclusion_rejects_bundled_claim(tablet_fixture):
    """14. Supporting a stylus does not permit claiming that a stylus is bundled."""
    text = "The TabAir 11 includes pen in the box for immediate drawing."
    contradictions = check_claim_contradictions(text, tablet_fixture, Requirements(category=Category.TABLET))
    assert any("stylus" in c.lower() for c in contradictions)


def test_15_unsupported_color_gamut_rejected(monitor_catalog):
    """15. Claiming 100% sRGB on a monitor lacking colour gamut confirmation is rejected."""
    mon_nexa = monitor_catalog[1]
    text = "Provides 100% sRGB color accuracy for professional photo grading."
    contradictions = check_claim_contradictions(text, mon_nexa, Requirements(category=Category.MONITOR))
    assert any("color" in c.lower() or "colour" in c.lower() for c in contradictions)


def test_16_prices_and_scores_remain_consistent(monitor_catalog):
    """16. Displayed prices and scores strictly match catalogue and deterministic ranking."""
    mon = monitor_catalog[0]
    req = Requirements(category=Category.MONITOR, budget_max_inr=15000)
    cands = rank_candidates([mon], req, top_n=1)
    rec = build_fallback_product_recommendation(cands[0], req)
    assert rec.price_inr == mon.price_inr
    assert rec.overall_score == cands[0].overall_score
    assert rec.product_id == mon.product_id


def test_17_18_repaired_response_uses_safe_fallback(monitor_catalog):
    """17 & 18. Factual contradiction triggers repair; repaired response contains grounded data."""
    mon = monitor_catalog[0]
    req = Requirements(category=Category.MONITOR, budget_max_inr=15000)
    cands = rank_candidates([mon], req, top_n=1)
    search_result = CandidateSearchResult(
        category=Category.MONITOR,
        total_retrieved=1,
        total_passed=1,
        total_excluded=0,
        total_unverifiable=0,
        ranked_candidates=cands,
        has_matches=True,
    )
    # LLM hallucinates 4K resolution on a 1080p monitor
    payload = LLMRecommendationPayload(
        message="Here is your top monitor:",
        explanations=[
            LLMProductExplanation(
                product_id=mon.product_id,
                reasoning="This stunning 4K UHD display offers razor sharp text.",
                advantages=["4K display"],
                trade_offs=[],
            )
        ],
    )
    response = validate_recommendation_response(payload, search_result, req)
    assert response.has_matches is True
    assert len(response.recommendations) == 1
    repaired_rec = response.recommendations[0]
    assert "4k" not in repaired_rec.reasoning.lower()
    assert repaired_rec.verified_specs["resolution"] == "1920x1080"


# ---------------------------------------------------------------------------
# Section 4 & 7: Ranking, Filtering & Preference Personalization (19-25)
# ---------------------------------------------------------------------------

def test_19_products_violating_hard_budget_excluded(monitor_catalog):
    """19. Products violating hard budget maximum are excluded."""
    req = Requirements(category=Category.MONITOR, budget_max_inr=15000)
    eligible, excluded = filter_candidates(monitor_catalog, req)
    assert all(p.price_inr <= 15000 for p in eligible)
    assert any(ex.product.product_id == "MON-0001" for ex in excluded)


def test_20_unknown_values_do_not_satisfy_mandatory_requirements(monitor_catalog):
    """20. Mandatory certification excludes products where certification is unverified."""
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=30000,
        hard=HardConstraints(requires_flicker_free=True),
    )
    eligible, excluded = filter_candidates(monitor_catalog, req)
    assert len(eligible) == 1
    assert eligible[0].product_id == "MON-0001"


@pytest.mark.asyncio
async def test_21_explicit_24_inch_preference_affects_monitor_ranking(monitor_catalog):
    """21. Explicit 24-inch preference ranks 23.8\" and 24.0\" higher than 27.0\"."""
    eligible_monitors = [m for m in monitor_catalog if m.price_inr <= 15000]
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=15000,
        hard=HardConstraints(target_screen_in=24.0),
        preferences={"screen_size": 1.0, "refresh_rate": 0.2},
    )
    result = await search_and_rank(req, eligible_monitors, top_n=5)
    ranked_ids = [c.product_id for c in result.ranked_candidates]
    assert ranked_ids.index("MON-0005") < ranked_ids.index("MON-0015")
    assert ranked_ids.index("MON-0007") < ranked_ids.index("MON-0015")


def test_22_unnecessary_high_refresh_rate_does_not_dominate():
    """22. When user prioritizes coding and text, a 165Hz gaming monitor does not automatically outscore an office monitor."""
    office_mon = Product(
        product_id="MON-OFFICE",
        name="OfficePro 24",
        brand="Dell",
        category=Category.MONITOR,
        price_inr=12000,
        rating=4.5,
        specs={"size_in": 24.0, "refresh_hz": 75, "panel": "IPS"},
        features=["Low Blue Light Certified", "Height Adjustable"],
        intended_use=["coding", "office"],
        in_stock=True,
    )
    gaming_mon = Product(
        product_id="MON-GAMING",
        name="SpeedGamer 24",
        brand="Asus",
        category=Category.MONITOR,
        price_inr=12500,
        rating=4.2,
        specs={"size_in": 24.0, "refresh_hz": 165, "panel": "TN"},
        features=["165Hz Refresh"],
        intended_use=["esports", "gaming"],
        in_stock=True,
    )
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=15000,
        use_cases=["coding"],
        preferences={"eye_comfort": 0.9, "refresh_rate": 0.1},
    )
    score_off = score_candidate(office_mon, req).overall_score
    score_gam = score_candidate(gaming_mon, req).overall_score
    assert score_off > score_gam


@pytest.mark.asyncio
async def test_23_user_priorities_change_ranking_deterministically(monitor_catalog):
    """23. User priorities alter deterministic scores and rankings."""
    eligible = [m for m in monitor_catalog if m.price_inr <= 15000]
    # Priority 1: Budget focus
    req_budget = Requirements(category=Category.MONITOR, preferences={"budget": 1.0, "refresh_rate": 0.1})
    res_budget = await search_and_rank(req_budget, eligible, top_n=3)
    assert res_budget.ranked_candidates[0].product_id == "MON-0007"

    # Priority 2: Refresh focus
    req_refresh = Requirements(category=Category.MONITOR, preferences={"budget": 0.1, "refresh_rate": 1.0})
    res_refresh = await search_and_rank(req_refresh, eligible, top_n=3)
    assert res_refresh.ranked_candidates[0].product_id == "MON-0015"


@pytest.mark.asyncio
async def test_24_no_match_response_contains_no_ineligible_recommendations(monitor_catalog):
    """24. No-match response never returns ineligible products under recommendations."""
    req = Requirements(category=Category.MONITOR, budget_max_inr=5000)
    result = await search_and_rank(req, monitor_catalog, top_n=3)
    assert result.has_matches is False
    assert len(result.ranked_candidates) == 0

    response = build_fallback_recommendation_response(req, result)
    assert response.has_matches is False
    assert len(response.recommendations) == 0
    assert "₹5,000" in response.message


@pytest.mark.asyncio
async def test_25_minimum_budget_question_uses_catalogue_data(monitor_catalog):
    """25. Minimum budget follow-up calculates actual cheapest matching price."""
    req = Requirements(
        category=Category.MONITOR,
        budget_max_inr=20000,
        hard=HardConstraints(requires_flicker_free=True),
    )
    result = await search_and_rank(req, monitor_catalog, top_n=5)
    assert result.has_matches is False

    min_budget_msg = await analyze_catalogue_minimum_budget(req, monitor_catalog)
    assert "28,999" in min_budget_msg
    assert "AeroView" in min_budget_msg


# ---------------------------------------------------------------------------
# Section 5 & 8: Cross-Category Verification (26-30)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_26_to_30_cross_category_and_offline_providers(
    laptop_fixture, phone_fixture, headphones_fixture, tablet_fixture, monitor_catalog
):
    """26-30. Verify laptops, phones, headphones, tablets, monitors, offline LLM, and schema compatibility."""
    all_products = [laptop_fixture, phone_fixture, headphones_fixture, tablet_fixture] + monitor_catalog

    for cat in [Category.LAPTOP, Category.PHONE, Category.HEADPHONES, Category.TABLET, Category.MONITOR]:
        cat_prods = [p for p in all_products if p.category == cat]
        req = Requirements(category=cat, requested_result_count=2)
        res = await search_and_rank(req, cat_prods, top_n=2)
        assert res.has_matches is True
        assert len(res.ranked_candidates) <= 2

        response = build_fallback_recommendation_response(req, res)
        assert isinstance(response, RecommendationResponse)
        assert response.total_matching == len(res.ranked_candidates)
        for rec in response.recommendations:
            assert isinstance(rec, ProductRecommendation)
            assert rec.price_inr == rec.verified_specs.get("price_inr", rec.price_inr)
            assert isinstance(rec.limitations, list)


# ---------------------------------------------------------------------------
# Section 11: Exact Reproduced Failure Scenario
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_section_11_exact_reproduction_monitor_query(monitor_catalog):
    """Integration reproduction test matching the exact prompt from Section 11:

    'I'm looking for a monitor under ₹15,000 mainly for coding for 8 hours a day,
    watching movies, and occasional gaming. I want a 24-inch display with sharp
    text and comfortable colours. Eye-comfort certification and flicker-free operation
    would be nice to have, but only recommend products if those features are verified.
    Please give me my top 2 options, explain why each is suitable, and clearly mention
    any specifications that are not available or verified in your catalogue.'
    """
    user_query = (
        "I'm looking for a monitor under ₹15,000 mainly for coding for 8 hours a day, "
        "watching movies, and occasional gaming. I want a 24-inch display with sharp text "
        "and comfortable colours. Eye-comfort certification and flicker-free operation "
        "would be nice to have, but only recommend products if those features are verified. "
        "Please give me my top 2 options, explain why each is suitable, and clearly mention "
        "any specifications that are not available or verified in your catalogue."
    )

    # 1. Delta Extraction with offline client
    fake = FakeLLMClient(json_responses=[{
        "category": "monitor",
        "budget_max_inr": 15000,
        "requested_result_count": 2,
        "use_cases": ["coding", "movies", "gaming"],
        "hard": {"target_screen_in": 24.0},
        "preferences": {"eye_comfort": 0.8, "flicker_free": 0.8},
    }])
    normalized = await extract(user_query, Requirements(), [], fake)

    # Verify extracted criteria
    assert normalized.category == Category.MONITOR
    assert normalized.budget_max_inr == 15000
    assert normalized.requested_result_count == 2
    assert "programming" in normalized.use_cases or "coding" in normalized.use_cases
    assert normalized.hard is not None
    assert normalized.hard.target_screen_in == 24.0 or normalized.hard.screen_size_in == 24.0
    assert "eye_comfort" in normalized.preferences or normalized.hard.requires_eye_comfort is True

    # 2. Merge with clean conversation state
    requirements = Requirements()
    requirements, _ = merge(requirements, normalized)
    assert requirements.budget_max_inr == 15000
    assert requirements.requested_result_count == 2

    # 3. Deterministic Hard-Constraint Filtering and Ranking
    search_result = await search_and_rank(requirements, monitor_catalog, top_n=2)

    # 4. Assert Expected Shortlist Properties
    assert search_result.has_matches is True
    # At most two recommendations
    assert len(search_result.ranked_candidates) <= 2
    assert len(search_result.ranked_candidates) == 2

    ranked_products = [c.product for c in search_result.ranked_candidates]

    # All recommendations satisfy the ₹15,000 hard budget
    for p in ranked_products:
        assert p.price_inr <= 15000

    # 24-inch preference influences ranking: 23.8"/24.0" monitors are prioritized over 27"
    for p in ranked_products:
        size = p.specs.get("size_in")
        assert size in (23.8, 24.0)

    # 5. Assemble grounded response
    response = build_fallback_recommendation_response(
        requirements=requirements,
        search_result=search_result,
        fallback_reason="Section 11 integration test execution",
    )

    # Assert Response Structure
    assert isinstance(response, RecommendationResponse)
    assert len(response.recommendations) == 2
    assert response.comparison_summary is not None
    assert len(response.remaining_uncertainties) > 0

    # Assert Grounding on each recommended product
    for rec in response.recommendations:
        cat_prod = next(p for p in monitor_catalog if p.product_id == rec.product_id)
        # Exact catalogue price and specs
        assert rec.price_inr == cat_prod.price_inr
        assert rec.rating == cat_prod.rating
        assert rec.verified_specs["size_in"] == cat_prod.specs["size_in"]
        assert rec.verified_specs["refresh_hz"] == cat_prod.specs["refresh_hz"]

        # No unsupported eye-comfort certification or flicker-free claim
        combined_text = f"{rec.reasoning} {' '.join(rec.advantages)} {rec.match_explanation}"
        contradictions = check_claim_contradictions(combined_text, cat_prod, requirements)
        assert len(contradictions) == 0

        # Missing certification evidence is clearly identified in limitations
        features_str = " ".join(cat_prod.features).lower()
        if "anti-flicker" not in features_str and "flicker-free" not in features_str:
            assert any("flicker-free" in lim.lower() or "anti-flicker" in lim.lower() for lim in rec.limitations)
        if "low blue light" not in features_str and "eye comfort" not in features_str:
            assert any("eye-comfort" in lim.lower() or "blue light" in lim.lower() for lim in rec.limitations)
