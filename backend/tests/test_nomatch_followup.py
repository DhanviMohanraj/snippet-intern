"""Offline regression tests for No-Match Follow-Up Conversation & Minimum-Budget Reasoning.

Verifies:
    1. A minimum-budget question after NO_MATCH does not repeat the identical response blindly.
    2. Minimum budget is computed from valid catalogue candidates only.
    3. A price increase does not override a missing-storage constraint.
    4. A storage relaxation updates only storage and preserves the remaining requirements.
    5. A budget update preserves the existing storage and other constraints.
    6. No valid candidate produces an honest explanation rather than an invented price.
    7. Alternative products violating hard constraints are not presented as matching products.
    8. The flow works across categories (Laptops, Headphones, Phones) and is not hardcoded specifically for smartphones.
    9. Diagnostics ("Which requirement is preventing a match?") correctly identifies blocking constraints.
"""

from __future__ import annotations

import pytest

from app.agent.followup import (
    FollowUpIntent,
    analyze_catalogue_minimum_budget,
    detect_followup_intent,
    explain_blocking_constraints,
    find_closest_alternatives,
)
from app.agent.pipeline import run_pipeline
from app.llm.fake import FakeLLMClient
from app.models.conversation import ConversationState
from app.models.product import Category, Product
from app.models.requirements import (
    HardConstraints,
    PolicyAction,
    Requirements,
    RequirementsDelta,
)


class MockCatalogueRepository:
    """Mock repository with controllable multi-category catalogue data."""

    def __init__(self, products: list[Product] | None = None):
        self.products = products or []

    async def get_by_category(self, category: Category) -> list[Product]:
        return [p for p in self.products if p.category == category]

    async def get_by_id(self, product_id: str) -> Product | None:
        for p in self.products:
            if p.product_id == product_id:
                return p
        return None

    async def get_categories(self) -> list[str]:
        return list({p.category.value for p in self.products})

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
        return matches[:page_size], len(matches)


@pytest.fixture
def phone_catalogue_repo() -> MockCatalogueRepository:
    """Fixture providing phones with known 64GB, 128GB, and 256GB configurations."""
    products = [
        # Budget 64GB phone
        Product(
            product_id="PHN-0001",
            name="Nexa Core 4G",
            brand="Nexa",
            category=Category.PHONE,
            price_inr=9999,
            specs={"ram_gb": 4, "storage_gb": 64, "camera_mp": 48.0, "battery_mah": 5000},
            intended_use=["general"],
            in_stock=True,
            rating=4.0,
        ),
        # 128GB phones
        Product(
            product_id="PHN-0002",
            name="Zenit Prime 5G",
            brand="Zenit",
            category=Category.PHONE,
            price_inr=17999,
            specs={"ram_gb": 8, "storage_gb": 128, "camera_mp": 50.0, "battery_mah": 5000},
            intended_use=["general", "photography"],
            in_stock=True,
            rating=4.3,
        ),
        Product(
            product_id="PHN-0003",
            name="ByteCraft Vision 1",
            brand="ByteCraft",
            category=Category.PHONE,
            price_inr=31999,
            specs={"ram_gb": 8, "storage_gb": 128, "camera_mp": 64.0, "battery_mah": 4500},
            intended_use=["photography", "general"],
            in_stock=True,
            rating=4.2,
        ),
        # 256GB phone at ₹42,000 (exceeds ₹35,000 budget)
        Product(
            product_id="PHN-0004",
            name="Apex Turbo GT 6",
            brand="Apex",
            category=Category.PHONE,
            price_inr=42000,
            specs={"ram_gb": 12, "storage_gb": 256, "camera_mp": 108.0, "battery_mah": 5000},
            intended_use=["gaming", "photography"],
            in_stock=True,
            rating=4.6,
        ),
    ]
    return MockCatalogueRepository(products)


# ---------------------------------------------------------------------------
# 1. Intent Detection Unit Tests
# ---------------------------------------------------------------------------

def test_detect_followup_intent_variants():
    """Verify follow-up intent classification for varied phrasing."""
    assert detect_followup_intent("what is the minimum budget") == FollowUpIntent.MINIMUM_BUDGET
    assert detect_followup_intent("What is the cheapest option?") == FollowUpIntent.MINIMUM_BUDGET
    assert detect_followup_intent("how much more do I need to spend") == FollowUpIntent.MINIMUM_BUDGET
    assert detect_followup_intent("what if I increase my budget?") == FollowUpIntent.MINIMUM_BUDGET
    assert detect_followup_intent("Keep 256GB but increase my budget") == FollowUpIntent.MINIMUM_BUDGET

    assert detect_followup_intent("Which requirement is preventing a match?") == FollowUpIntent.EXPLAIN_BLOCKING
    assert detect_followup_intent("why are there no matches?") == FollowUpIntent.EXPLAIN_BLOCKING
    assert detect_followup_intent("what is blocking the match?") == FollowUpIntent.EXPLAIN_BLOCKING

    assert detect_followup_intent("Show me the closest alternatives.") == FollowUpIntent.CLOSEST_ALTERNATIVES
    assert detect_followup_intent("can you find a cheaper alternative") == FollowUpIntent.CLOSEST_ALTERNATIVES

    # Concrete updates must NOT be treated as questions
    assert detect_followup_intent("increase budget to 45000") is None
    assert detect_followup_intent("what if I accept 128GB") is None


# ---------------------------------------------------------------------------
# 2. Minimum-Budget Question After NO_MATCH Does Not Repeat Blindly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_minimum_budget_question_after_nomatch_does_not_repeat_blindly(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """When user asks 'what is the minimum budget' after NO_MATCH, agent responds with actual minimum budget."""
    conv = ConversationState(
        conversation_id="conv-nomatch-1",
        requirements=Requirements(
            category=Category.PHONE,
            budget_max_inr=35000,
            use_cases=["photography"],
            hard=HardConstraints(min_storage_gb=256),
        ),
        slot_status={
            "category": "answered",
            "budget": "answered",
            "use_cases": "answered",
            "phone_storage": "answered",
            "phone_priority": "answered",
        },
        turn_count=2,
    )

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(),  # Extraction returns empty delta for inquiry
        ]
    )

    result = await run_pipeline(
        user_message="what is the minimum budget",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=phone_catalogue_repo,
    )

    # 1. Action is NO_MATCH (or clarification) but NOT repeating identical failure text
    assert result.decision.action == PolicyAction.NO_MATCH
    assert len(result.recommendations) == 0

    # 2. Computes the real minimum budget: ₹42,000 for Apex Turbo GT 6
    msg = result.assistant_message
    assert "42,000" in msg or "42000" in msg
    assert "Apex Turbo GT 6" in msg
    assert "7,000" in msg or "7000" in msg  # Difference from current budget (42000 - 35000)
    assert "256GB" in msg


# ---------------------------------------------------------------------------
# 3. Minimum Budget Computed From Valid Catalogue Candidates Only
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_minimum_budget_computed_from_valid_candidates_only(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """Minimum budget ignores 64GB/128GB phones and selects the lowest price among 256GB candidates."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=35000,
        hard=HardConstraints(min_storage_gb=256),
    )

    resp = await analyze_catalogue_minimum_budget(req, phone_catalogue_repo)
    # Nexa Core 4G is ₹9,999 and ByteCraft is ₹31,999, but neither has 256GB
    assert "9,999" not in resp
    assert "31,999" not in resp
    assert "42,000" in resp
    assert "Apex Turbo GT 6" in resp


# ---------------------------------------------------------------------------
# 4. Price Increase Does Not Override Missing Storage Constraint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_price_increase_does_not_override_missing_storage():
    """If catalogue has no 256GB phones at any price, increasing budget cannot produce a match."""
    # Catalogue with ONLY 128GB phones
    repo_only_128 = MockCatalogueRepository([
        Product(
            product_id="PHN-101",
            name="Phone A 128GB",
            brand="Zenit",
            category=Category.PHONE,
            price_inr=20000,
            specs={"storage_gb": 128},
            intended_use=["general"],
            in_stock=True,
        ),
        Product(
            product_id="PHN-102",
            name="Phone B 128GB",
            brand="Pulse",
            category=Category.PHONE,
            price_inr=80000,
            specs={"storage_gb": 128},
            intended_use=["general"],
            in_stock=True,
        ),
    ])

    conv = ConversationState(
        conversation_id="conv-missing-storage",
        requirements=Requirements(
            category=Category.PHONE,
            budget_max_inr=35000,
            hard=HardConstraints(min_storage_gb=256),
        ),
        slot_status={"category": "answered", "budget": "answered", "phone_storage": "answered"},
        turn_count=2,
    )

    fake_llm = FakeLLMClient(json_responses=[RequirementsDelta()])

    result = await run_pipeline(
        user_message="Keep 256GB but increase my budget",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=repo_only_128,
    )

    msg = result.assistant_message.lower()
    # Honest explanation: cannot determine minimum budget because 256GB is unavailable
    assert "cannot determine a minimum budget" in msg or "no" in msg
    assert "256gb" in msg
    assert "maximum available" in msg or "128gb" in msg
    assert "increasing your budget alone will not" in msg
    assert len(result.recommendations) == 0


# ---------------------------------------------------------------------------
# 5. Storage Relaxation Updates Only Storage & Preserves Remaining Requirements
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_storage_relaxation_updates_only_storage_and_preserves_remaining(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """User accepts 128GB: storage updates to 128GB, budget & preferences stay intact, match found."""
    conv = ConversationState(
        conversation_id="conv-relax-storage",
        requirements=Requirements(
            category=Category.PHONE,
            budget_max_inr=35000,
            use_cases=["photography"],
            hard=HardConstraints(min_storage_gb=256),
            preferences={"camera_quality": 0.85},
        ),
        slot_status={
            "category": "answered",
            "budget": "answered",
            "use_cases": "answered",
            "phone_storage": "answered",
            "phone_priority": "answered",
        },
        turn_count=2,
    )

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(hard=HardConstraints(min_storage_gb=128)),
        ]
    )

    result = await run_pipeline(
        user_message="What if I accept 128GB?",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=phone_catalogue_repo,
    )

    # 1. Requirements updated cleanly
    assert result.updated_conversation.requirements.hard.min_storage_gb == 128
    assert result.updated_conversation.requirements.budget_max_inr == 35000
    assert result.updated_conversation.requirements.category == Category.PHONE
    assert "photography" in result.updated_conversation.requirements.use_cases
    assert result.updated_conversation.requirements.preferences["camera_quality"] == 0.85

    # 2. Decision transitions from NO_MATCH into RECOMMEND
    assert result.decision.action == PolicyAction.RECOMMEND
    assert len(result.recommendations) > 0
    # Returned products are 128GB phones (e.g. ByteCraft Vision 1 or Zenit Prime 5G)
    for rec in result.recommendations:
        assert rec.price_inr <= 35000


# ---------------------------------------------------------------------------
# 6. Budget Update Preserves Existing Storage Constraint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_update_preserves_existing_storage(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """Increasing budget to ₹45,000 preserves min_storage_gb=256 and matches Apex Turbo GT 6."""
    conv = ConversationState(
        conversation_id="conv-budget-update",
        requirements=Requirements(
            category=Category.PHONE,
            budget_max_inr=35000,
            use_cases=["photography"],
            hard=HardConstraints(min_storage_gb=256),
        ),
        slot_status={
            "category": "answered",
            "budget": "answered",
            "use_cases": "answered",
            "phone_storage": "answered",
            "phone_priority": "answered",
        },
        turn_count=2,
    )

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(budget_max_inr=45000),
        ]
    )

    result = await run_pipeline(
        user_message="Keep 256GB but increase my budget to 45000",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=phone_catalogue_repo,
    )

    # 1. Budget updated, storage strictly preserved
    assert result.updated_conversation.requirements.budget_max_inr == 45000
    assert result.updated_conversation.requirements.hard.min_storage_gb == 256

    # 2. Recommendations match 256GB phone (Apex Turbo GT 6 at ₹42,000)
    assert result.decision.action == PolicyAction.RECOMMEND
    assert len(result.recommendations) == 1
    assert result.recommendations[0].product_id == "PHN-0004"
    assert result.recommendations[0].name == "Apex Turbo GT 6"


# ---------------------------------------------------------------------------
# 7. No Valid Candidate Produces Honest Explanation (No Invented Prices)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_valid_candidate_produces_honest_explanation_not_invented_price(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """Asking for 1TB phone when max available is 256GB produces honest explanation with no fake price."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=35000,
        hard=HardConstraints(min_storage_gb=1024),
    )

    resp = await analyze_catalogue_minimum_budget(req, phone_catalogue_repo)
    assert "cannot determine a minimum budget" in resp.lower()
    assert "1024gb" in resp.lower()
    assert "256gb" in resp.lower()  # Max available in phone_catalogue_repo is 256GB
    assert "increasing your budget alone will not yield a match" in resp.lower()


# ---------------------------------------------------------------------------
# 8. Alternative Products Violating Hard Constraints Not Presented as Matches
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_closest_alternatives_not_presented_as_matching_cards(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """Closest alternatives endpoint returns empty recommendations list and marks trade-offs in text."""
    conv = ConversationState(
        conversation_id="conv-closest-alts",
        requirements=Requirements(
            category=Category.PHONE,
            budget_max_inr=35000,
            hard=HardConstraints(min_storage_gb=256),
        ),
        slot_status={"category": "answered", "budget": "answered", "phone_storage": "answered"},
        turn_count=2,
    )

    fake_llm = FakeLLMClient(json_responses=[RequirementsDelta()])

    result = await run_pipeline(
        user_message="Show me the closest alternatives",
        conversation=conv,
        llm_client=fake_llm,
        product_repo=phone_catalogue_repo,
    )

    # Hard constraints violated must not be put into recommendation cards!
    assert len(result.recommendations) == 0
    msg = result.assistant_message
    assert "closest alternatives" in msg.lower()
    # Identifies in-budget trade-off (below required storage) and over-budget option
    assert "below your required 256gb" in msg.lower() or "differs" in msg.lower()


# ---------------------------------------------------------------------------
# 9. Cross-Category Verification (Laptops & Headphones)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cross_category_laptop_minimum_budget():
    """Verify minimum budget computation works for Laptops (e.g. 32GB RAM)."""
    laptop_repo = MockCatalogueRepository([
        Product(
            product_id="LAP-001",
            name="Zenit Student 16",
            brand="Zenit",
            category=Category.LAPTOP,
            price_inr=50000,
            specs={"ram_gb": 16, "storage_gb": 512},
            intended_use=["student"],
            in_stock=True,
        ),
        Product(
            product_id="LAP-002",
            name="Apex Pro Creator 32",
            brand="Apex",
            category=Category.LAPTOP,
            price_inr=85000,
            specs={"ram_gb": 32, "storage_gb": 1024},
            intended_use=["programming"],
            in_stock=True,
        ),
    ])

    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=60000,
        hard=HardConstraints(min_ram_gb=32),
    )

    resp = await analyze_catalogue_minimum_budget(req, laptop_repo)
    assert "85,000" in resp
    assert "Apex Pro Creator 32" in resp
    assert "25,000" in resp  # 85000 - 60000 difference
    assert "32GB RAM" in resp


@pytest.mark.asyncio
async def test_cross_category_headphone_minimum_budget():
    """Verify minimum budget computation works for Headphones (e.g. over-ear form factor)."""
    hph_repo = MockCatalogueRepository([
        Product(
            product_id="HPH-001",
            name="Zenit In-Ear Neo",
            brand="Zenit",
            category=Category.HEADPHONES,
            price_inr=2999,
            specs={"type": "in-ear"},
            intended_use=["music"],
            in_stock=True,
        ),
        Product(
            product_id="HPH-002",
            name="Apex Over-Ear Cloud X",
            brand="Apex",
            category=Category.HEADPHONES,
            price_inr=5499,
            specs={"type": "over-ear"},
            intended_use=["gaming"],
            in_stock=True,
        ),
    ])

    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=3000,
        hard=HardConstraints(form_factor="over-ear"),
    )

    resp = await analyze_catalogue_minimum_budget(req, hph_repo)
    assert "5,499" in resp
    assert "Apex Over-Ear Cloud X" in resp
    assert "2,499" in resp  # 5499 - 3000 difference


# ---------------------------------------------------------------------------
# 10. Diagnostics: Which Requirement Is Blocking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explain_blocking_constraints_diagnostics(
    phone_catalogue_repo: MockCatalogueRepository,
):
    """When user asks 'which requirement is preventing a match', agent explains budget vs hardware."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=35000,
        hard=HardConstraints(min_storage_gb=256),
    )

    resp = await explain_blocking_constraints(req, phone_catalogue_repo)
    assert "Budget Limit (₹35,000)" in resp
    assert "42,000" in resp
    assert "Apex Turbo GT 6" in resp
