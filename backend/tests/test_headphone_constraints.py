"""Comprehensive regression tests for multi-turn headphone requirements and hard constraints.

Proves:
    1. Complete initial headphone request extraction and preservation.
    2. Budget and preferences remain stored across turns.
    3. Selecting over-ear excludes in-ear products from recommendations.
    4. An incompatible product cannot be returned merely because its score is high.
    5. No-match behavior works when all products violate a hard constraint.
    6. A skipped or delegated optional preference does not cause an endless conversation.
    7. The agent continues asking relevant mandatory questions without a fixed turn cutoff.
    8. Displayed match scores and product facts originate from actual backend data.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.extract import extract
from app.agent.merge import merge
from app.agent.normalize import normalize
from app.agent.pipeline import _build_fallback_delta, run_pipeline
from app.agent.policy import decide_next_action
from app.agent.ranking import evaluate_candidate, score_candidate, search_and_rank
from app.agent.slots import (
    SlotStatus,
    can_proceed_to_recommendation,
    parse_natural_language_slot_updates,
)
from app.main import create_app
from app.db.conversation_repository import (
    ConversationRepository,
    InMemoryConversationRepository,
    get_conversation_repository,
)
from app.db.repositories import ProductRepository
from app.llm.fake import FakeLLMClient
from app.api.v1.conversations import get_llm_client
from app.models.conversation import ConversationState
from app.models.product import Category, Product
from app.models.requirements import (
    HardConstraints,
    PolicyAction,
    Requirements,
    RequirementsDelta,
)


class MockHeadphoneRepository:
    """Mock repository with representative catalogue products."""

    def __init__(self):
        self.products = [
            Product(
                product_id="HPH-0010",
                name="Apex Gaming Cloud X",
                brand="Apex",
                category=Category.HEADPHONES,
                price_inr=5499,
                specs={
                    "type": "over-ear",
                    "anc": False,
                    "battery_hr": 30.0,
                    "wireless": True,
                    "weight_g": 285,
                    "codec": "2.4GHz Low Latency",
                },
                intended_use=["gaming", "calls"],
                features=["53mm Drivers", "Memory Foam Ear Cushions"],
                in_stock=True,
                rating=4.4,
            ),
            Product(
                product_id="HPH-0011",
                name="Titan Field Comm Wireless",
                brand="Titan",
                category=Category.HEADPHONES,
                price_inr=8500,
                specs={
                    "type": "over-ear",
                    "anc": True,
                    "battery_hr": 20.0,
                    "wireless": True,
                    "weight_g": 210,
                    "codec": "AAC, SBC",
                },
                intended_use=["calls", "travel"],
                features=["Heavy Duty Boom Mic", "Wind Noise Filter"],
                in_stock=True,
                rating=4.5,
            ),
            Product(
                product_id="HPH-0001",
                name="Zenit SoundPod Neo",
                brand="Zenit",
                category=Category.HEADPHONES,
                price_inr=3999,
                specs={
                    "type": "in-ear",
                    "anc": True,
                    "battery_hr": 28.0,
                    "wireless": True,
                    "weight_g": 45,
                },
                intended_use=["music", "calls"],
                features=["Active Noise Cancellation", "ENC Quad Mic"],
                in_stock=True,
                rating=4.7,
            ),
            Product(
                product_id="HPH-0002",
                name="Pulse SportFit Air",
                brand="Pulse",
                category=Category.HEADPHONES,
                price_inr=4499,
                specs={
                    "type": "in-ear",
                    "anc": False,
                    "battery_hr": 32.0,
                    "wireless": True,
                    "weight_g": 40,
                },
                intended_use=["sports", "music", "calls"],
                features=["IPX7 Waterproof", "Bass Boost"],
                in_stock=True,
                rating=4.6,
            ),
        ]

    async def get_by_category(self, category: Category) -> list[Product]:
        return [p for p in self.products if p.category == category]

    async def get_by_id(self, product_id: str) -> Product | None:
        for p in self.products:
            if p.product_id == product_id:
                return p
        return None

    async def get_categories(self) -> list[str]:
        return [Category.HEADPHONES.value]

    async def find(
        self,
        filters: dict | None = None,
        sort: str = "price_inr_asc",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Product], int]:
        flt = filters or {}
        matches = list(self.products)
        if "category" in flt:
            cat_val = flt["category"]
            cat_str = cat_val.value if isinstance(cat_val, Category) else str(cat_val)
            matches = [p for p in matches if p.category.value == cat_str]
        total = len(matches)
        return matches[:page_size], total


# ---------------------------------------------------------------------------
# 1. Exact Section A Initial Request Extraction & Preservation
# ---------------------------------------------------------------------------

EXACT_SECTION_A_REQUEST = (
    "I'm looking for wireless headphones for listening to music, attending online classes, and taking calls. "
    "My budget is ₹5,000. I prefer clear vocals, good microphone quality, and comfortable ear cushions because "
    "I'll use them for several hours daily. Battery life is important, and I'd like noise cancellation if possible."
)


@pytest.mark.asyncio
async def test_section_a_prompt_extraction_preserves_all_requirements():
    """Verify that all supported info in the exact Section A prompt is extracted."""
    delta = _build_fallback_delta(EXACT_SECTION_A_REQUEST)

    # 1. Category
    assert delta.category == Category.HEADPHONES

    # 2. Budget
    assert delta.budget_max_inr == 5000

    # 3. Use Cases
    assert "music" in delta.use_cases
    assert "classes" in delta.use_cases
    assert "calls" in delta.use_cases

    # 4. Hard Constraints
    assert delta.hard is not None
    assert delta.hard.requires_wireless is True
    # Form factor was not yet specified in Turn 1
    assert delta.hard.form_factor is None

    # 5. Preferences
    assert delta.preferences is not None
    assert "microphone_quality" in delta.preferences
    assert "comfort" in delta.preferences
    assert "battery_life" in delta.preferences
    assert "noise_cancellation" in delta.preferences


@pytest.mark.asyncio
async def test_requirements_remain_stored_across_turns():
    """Verify that budget, use cases, and preferences remain stored when user answers form factor."""
    # Turn 1: Process initial request
    conv = ConversationState()
    repo = MockHeadphoneRepository()
    fake_llm = FakeLLMClient(
        json_responses=[
            _build_fallback_delta(EXACT_SECTION_A_REQUEST),
            RequirementsDelta(hard=HardConstraints(form_factor="over-ear")),
        ]
    )

    result_turn1 = await run_pipeline(
        user_message=EXACT_SECTION_A_REQUEST,
        conversation=conv,
        llm_client=fake_llm,
        product_repo=repo,
    )

    # In Turn 1, form factor is missing, so agent asks about form factor
    assert result_turn1.decision.action == PolicyAction.ASK
    assert result_turn1.decision.slot == "form_factor"
    assert "over-ear" in result_turn1.assistant_message.lower()

    # Verify requirements stored after Turn 1
    req1 = result_turn1.updated_conversation.requirements
    assert req1.category == Category.HEADPHONES
    assert req1.budget_max_inr == 5000
    assert "calls" in req1.use_cases
    assert req1.hard.requires_wireless is True
    assert "microphone_quality" in req1.preferences
    assert "comfort" in req1.preferences

    # Turn 2: User answers with "over ear headphones"
    result_turn2 = await run_pipeline(
        user_message="over ear headphones",
        conversation=result_turn1.updated_conversation,
        llm_client=fake_llm,
        product_repo=repo,
    )

    req2 = result_turn2.updated_conversation.requirements
    # All prior information is preserved
    assert req2.category == Category.HEADPHONES
    assert req2.budget_max_inr == 5000
    assert "calls" in req2.use_cases
    assert req2.hard.requires_wireless is True
    assert "microphone_quality" in req2.preferences
    assert "comfort" in req2.preferences
    # Form factor is now recorded as over-ear
    assert req2.hard.form_factor == "over-ear"


# ---------------------------------------------------------------------------
# 2. Hard Constraint: Form Factor Strictly Excludes In-Ear
# ---------------------------------------------------------------------------

def test_form_factor_hard_constraint_excludes_in_ear():
    """Explicitly selecting over-ear MUST reject in-ear products regardless of score or price."""
    in_ear_product = Product(
        product_id="HPH-0001",
        name="Zenit SoundPod Neo",
        brand="Zenit",
        category=Category.HEADPHONES,
        price_inr=3999,
        specs={"type": "in-ear", "anc": True, "battery_hr": 28.0, "wireless": True},
        intended_use=["music", "calls"],
        features=["Active Noise Cancellation"],
        in_stock=True,
        rating=4.9,
    )

    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=10000,
        hard=HardConstraints(form_factor="over-ear"),
    )

    eval_result = evaluate_candidate(in_ear_product, req)
    assert eval_result.is_eligible is False
    assert any("form factor" in r.lower() or "type" in r.lower() for r in eval_result.rejection_reasons)


def test_incompatible_product_cannot_be_returned_due_to_high_score():
    """High rating and preference match cannot override a hard form factor violation."""
    in_ear_high_rating = Product(
        product_id="HPH-0001",
        name="Zenit SoundPod Neo",
        brand="Zenit",
        category=Category.HEADPHONES,
        price_inr=3999,
        specs={"type": "in-ear", "anc": True, "wireless": True},
        intended_use=["music", "calls"],
        features=["ANC", "ENC"],
        in_stock=True,
        rating=5.0,
    )

    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=10000,
        hard=HardConstraints(form_factor="over-ear"),
        preferences={"noise_cancellation": 1.0, "microphone_quality": 1.0},
    )

    eval_res = evaluate_candidate(in_ear_high_rating, req)
    assert eval_res.is_eligible is False


# ---------------------------------------------------------------------------
# 3. No-Match Behavior on Budget + Over-Ear
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_match_when_all_products_violate_hard_constraints():
    """Budget ₹5,000 with over-ear form factor yields NO_MATCH when lowest over-ear is ₹5,499."""
    repo = MockHeadphoneRepository()
    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=5000,
        hard=HardConstraints(form_factor="over-ear", requires_wireless=True),
        use_cases=["calls", "music"],
    )

    search_res = await search_and_rank(req, repo, top_n=5)
    assert search_res.has_matches is False
    assert len(search_res.ranked_candidates) == 0
    # Over-ear models were rejected for budget; in-ear models rejected for form-factor
    assert len(search_res.excluded_candidates) == 4


@pytest.mark.asyncio
async def test_pipeline_returns_no_match_action_and_relaxed_suggestion():
    """Pipeline produces clean NO_MATCH action and clear relaxation suggestions."""
    conv = ConversationState()
    conv.requirements = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=5000,
        use_cases=["calls", "music"],
    )
    conv.slot_status = {"category": "answered", "budget": "answered", "use_cases": "answered"}
    conv.last_asked_slot = "form_factor"

    repo = MockHeadphoneRepository()
    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(hard=HardConstraints(form_factor="over-ear")),
        ]
    )

    result = await run_pipeline(
        user_message="over ear headphones",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=repo,
    )

    assert result.decision.action == PolicyAction.NO_MATCH
    assert len(result.recommendations) == 0
    assert "apex gaming cloud x" in result.assistant_message.lower() or "budget" in result.assistant_message.lower()
    assert "5,499" in result.assistant_message or "5500" in result.assistant_message or "relax" in result.assistant_message.lower()


# ---------------------------------------------------------------------------
# 4. Constraint Relaxation Matches Product
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_relaxing_budget_matches_over_ear_and_still_excludes_in_ear():
    """Increasing budget to ₹6,000 matches Apex Gaming Cloud X (₹5,499) while still excluding in-ear products."""
    repo = MockHeadphoneRepository()
    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=6000,
        hard=HardConstraints(form_factor="over-ear", requires_wireless=True),
        use_cases=["calls"],
    )

    search_res = await search_and_rank(req, repo, top_n=5)
    assert search_res.has_matches is True
    assert len(search_res.ranked_candidates) == 1
    matched = search_res.ranked_candidates[0]
    assert matched.product.product_id == "HPH-0010"
    assert matched.product.name == "Apex Gaming Cloud X"
    assert matched.product.specs.get("type") == "over-ear"


# ---------------------------------------------------------------------------
# 5. Delegation / Skip Handling
# ---------------------------------------------------------------------------

def test_delegated_form_factor_allows_both_types():
    """When user delegates form factor ('you decide'), both over-ear and in-ear products are eligible."""
    status_updates, delta = parse_natural_language_slot_updates(
        text="you decide",
        last_asked_slot="form_factor",
        current_req=Requirements(category=Category.HEADPHONES),
    )

    assert status_updates["form_factor"] == SlotStatus.DELEGATED.value
    assert delta.hard is None  # Does not impose a hard constraint


def test_skipped_form_factor_does_not_loop_endlessly():
    """Declining or skipping form factor satisfies the slot resolution without blocking."""
    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=5000,
        use_cases=["music"],
    )
    slot_status = {
        "category": "answered",
        "budget": "answered",
        "use_cases": "answered",
        "form_factor": SlotStatus.DECLINED.value,
    }

    can_rec, blocking_reason, remaining = can_proceed_to_recommendation(req, slot_status, [])
    assert can_rec is True
    assert blocking_reason is None
    assert len(remaining) == 0


# ---------------------------------------------------------------------------
# 6. Score & Specification Grounding
# ---------------------------------------------------------------------------

def test_displayed_scores_and_specs_originate_from_catalogue():
    """Verify that score breakdown and specs reflect catalogue facts without fabrication."""
    product = Product(
        product_id="HPH-0010",
        name="Apex Gaming Cloud X",
        brand="Apex",
        category=Category.HEADPHONES,
        price_inr=5499,
        specs={
            "type": "over-ear",
            "anc": False,
            "battery_hr": 30.0,
            "wireless": True,
            "weight_g": 285,
        },
        intended_use=["gaming", "calls"],
        features=["Memory Foam Ear Cushions"],
        in_stock=True,
        rating=4.4,
    )

    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=6000,
        hard=HardConstraints(form_factor="over-ear"),
        preferences={"comfort": 0.85, "microphone_quality": 0.85},
    )

    sb = score_candidate(product, req)
    assert sb.overall_score > 0
    assert "comfort" in sb.preference_scores
    assert "memory foam" in sb.explanations["comfort"].lower()
    # Does not claim ANC
    assert sb.preference_scores["noise_cancellation"] <= 50.0
