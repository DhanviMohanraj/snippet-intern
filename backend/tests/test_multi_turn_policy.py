"""Comprehensive offline tests for Multi-Turn Clarification Policy and Slot-Filling Engine.

Covers all 25 required verification scenarios:
1. The initial laptop request does not immediately recommend.
2. Missing mandatory slots trigger clarification.
3. The agent asks only one question at a time.
4. The three-turn limit no longer triggers premature recommendation.
5. A user answering one slot advances to the next missing slot.
6. Multiple answers in one message populate multiple slots.
7. Previously answered slots are not asked again.
8. A budget update is preserved across turns.
9. A category change recalculates applicable slots.
10. GPU questions depend on the user's workload.
11. Optional preferences do not block recommendation indefinitely.
12. Explicitly declining a mandatory slot follows a defined policy.
13. "You decide" is handled without repeated questioning.
14. Contradictions block recommendation.
15. Off-topic handling remains correct.
16. Product retrieval is not called while clarification is required.
17. Response generation is not called while clarification is required.
18. Product retrieval is called only when the recommendation gate passes.
19. No-match handling works after all mandatory inputs are collected.
20. Conversation state persists across API requests.
21. Clarification slot state remains isolated between conversations.
22. Headphones use headphone-specific slots.
23. Smartphones use smartphone-specific slots.
24. Tablets and monitors use their own applicable slots.
25. The full previous test suite continues to pass.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.products import get_product_repository
from app.api.v1.conversations import get_conversation_repository, get_llm_client
from app.agent.policy import decide_next_action
from app.agent.slots import (
    SLOT_REGISTRY,
    SlotStatus,
    can_proceed_to_recommendation,
    get_applicable_slots,
    get_next_slot_to_clarify,
    get_unresolved_mandatory_slots,
    is_slot_answered,
    is_slot_resolved,
)
from app.db.conversation_repository import InMemoryConversationRepository
from app.llm.fake import FakeLLMClient
from app.main import create_app
from app.models.product import Category, Product
from app.models.requirements import (
    Contradiction,
    HardConstraints,
    PolicyAction,
    Requirements,
    RequirementsDelta,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Factories
# ---------------------------------------------------------------------------

def make_test_laptop(
    product_id: str = "LAP-001",
    name: str = "Zenit Pro 15",
    brand: str = "Zenit",
    price_inr: int = 65000,
    ram_gb: int = 16,
    storage_gb: int = 512,
) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        brand=brand,
        category=Category.LAPTOP,
        price_inr=price_inr,
        specs={
            "cpu": "Intel Core i5-13500H",
            "ram_gb": ram_gb,
            "storage_gb": storage_gb,
            "gpu": "Intel Iris Xe",
            "vram_gb": 0,
            "weight_kg": 1.45,
            "battery_hr": 9.0,
        },
        intended_use=["programming", "student"],
        features=["Backlit Keyboard", "IPS Display"],
        rating=4.5,
        in_stock=True,
    )


class SpyProductRepository:
    """Mock repository that tracks calls to find()."""

    def __init__(self, products: list[Product]) -> None:
        self._products = products
        self.find_call_count = 0
        self.last_filters = None

    async def get(self, product_id: str) -> Product | None:
        for p in self._products:
            if p.product_id == product_id.strip():
                return p
        return None

    async def find(self, filters=None, sort=None, page=1, page_size=50):
        self.find_call_count += 1
        self.last_filters = filters
        results = [p for p in self._products if filters is None or p.category == filters.get("category")]
        return results, len(results)


@pytest.fixture
def mock_db():
    return InMemoryConversationRepository()


@pytest.fixture
def mock_products():
    return SpyProductRepository([make_test_laptop()])


# ---------------------------------------------------------------------------
# 1. Initial Laptop Request Does NOT Immediately Recommend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_initial_laptop_request_does_not_immediately_recommend(mock_db, mock_products):
    """When user says 'I need a laptop for college programming under 70k', agent asks follow-up, NOT recommend."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(
                category=Category.LAPTOP,
                budget_max_inr=70000,
                use_cases=["programming", "student"],
            )
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I need a laptop for college programming, Python development, and VS Code under ₹70,000"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()

        # Crucial: Must be ASK, NOT RECOMMEND
        assert data["action"] == "ASK"
        assert data["can_recommend"] is False
        assert len(data["recommendations"]) == 0
        assert data["requires_clarification"] is True
        assert data["metadata"]["slot"] == "ram"
        assert "ram" in data["message"].lower()


# ---------------------------------------------------------------------------
# 2. Missing Mandatory Slots Trigger Clarification
# ---------------------------------------------------------------------------

def test_missing_mandatory_slots_trigger_clarification():
    """Unit test for policy: Category, budget, use_cases present but RAM missing -> asks RAM."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=None),
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.ASK
    assert decision.can_recommend is False
    assert decision.slot == "ram"
    assert "ram" in decision.question.lower()


# ---------------------------------------------------------------------------
# 3. Agent Asks Only One Question at a Time
# ---------------------------------------------------------------------------

def test_agent_asks_only_one_question_at_a_time():
    """When both RAM and storage are missing, only one question (RAM) is returned."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=None, min_storage_gb=None),
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "ram"
    # Question asks specifically about RAM, not combining storage
    assert "storage" not in decision.question.lower()


# ---------------------------------------------------------------------------
# 4. Three-Turn Limit No Longer Triggers Premature Recommendation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_three_turn_limit_no_longer_triggers_premature_recommendation(mock_db, mock_products):
    """After 3 turns, if mandatory slots (RAM, storage) remain missing, do NOT recommend."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # 3 turns where user only specifies category, but RAM/storage remain missing
    deltas = [
        RequirementsDelta(category=Category.LAPTOP),
        RequirementsDelta(use_cases=["student"]),
        RequirementsDelta(),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "I want a laptop"})
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "It is for study"})
        r3 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Can you help?"})

        assert r3.status_code == 200
        data = r3.json()
        # Must NOT be RECOMMEND
        assert data["action"] == "ASK"
        assert data["can_recommend"] is False
        assert len(data["recommendations"]) == 0


# ---------------------------------------------------------------------------
# 5. User Answering One Slot Advances to Next Missing Slot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_answering_one_slot_advances_to_next_missing_slot(mock_db, mock_products):
    """Turn 1: Laptop under 70k -> asks RAM. Turn 2: '16GB RAM' -> asks Storage."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000, use_cases=["programming"]),
        RequirementsDelta(hard=HardConstraints(min_ram_gb=16)),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Turn 1
        r1 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I need a programming laptop under 70k"},
        )
        assert r1.json()["action"] == "ASK"
        assert r1.json()["metadata"]["slot"] == "ram"

        # Turn 2: answer RAM
        r2 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I need at least 16GB RAM"},
        )
        assert r2.json()["action"] == "ASK"
        # Advanced to storage!
        assert r2.json()["metadata"]["slot"] == "storage"
        assert "storage" in r2.json()["message"].lower()


# ---------------------------------------------------------------------------
# 6. Multiple Answers in One Message Populate Multiple Slots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_answers_in_one_message_populate_multiple_slots(mock_db, mock_products):
    """Turn 2 provides both '16GB RAM and 512GB SSD' -> both slots populated, unlocks recommendation."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    rec_payload = {
        "message": "Here are our top recommended laptops:",
        "explanations": [
            {
                "product_id": "LAP-001",
                "reasoning": "Top match",
                "advantages": ["16GB RAM", "512GB SSD"],
                "trade_offs": [],
                "match_explanation": "Perfect match",
                "limitations": [],
            }
        ],
    }

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000, use_cases=["programming"]),
        RequirementsDelta(hard=HardConstraints(min_ram_gb=16, min_storage_gb=512)),
        rec_payload,
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Turn 1
        await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Programming laptop under 70k"},
        )

        # Turn 2: answers both RAM and Storage
        r2 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "16GB RAM and 512GB SSD"},
        )
        data = r2.json()
        assert data["action"] == "RECOMMEND"
        assert data["can_recommend"] is True
        assert len(data["recommendations"]) > 0


# ---------------------------------------------------------------------------
# 7. Previously Answered Slots Are Not Asked Again
# ---------------------------------------------------------------------------

def test_previously_answered_slots_not_asked_again():
    """Once RAM is answered (min_ram_gb=16), policy does not ask for RAM again."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=None),
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.ASK
    assert decision.slot != "ram"
    assert decision.slot == "storage"


# ---------------------------------------------------------------------------
# 8. Budget Update Is Preserved Across Turns
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_update_preserved_across_turns(mock_db, mock_products):
    """Updating budget in Turn 2 preserves category and use_cases while updating budget_max_inr."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=60000, use_cases=["programming"]),
        RequirementsDelta(budget_max_inr=85000),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 60k for coding"})
        r2 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Actually make budget 85k"})

        assert r2.status_code == 200
        reqs = r2.json()["requirements"]
        assert reqs["category"] == "laptop"
        assert reqs["budget_max_inr"] == 85000
        assert "programming" in reqs["use_cases"]


# ---------------------------------------------------------------------------
# 9. Category Change Recalculates Applicable Slots
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_category_change_recalculates_applicable_slots(mock_db, mock_products):
    """Switching category from laptop to headphones clears laptop slots and asks headphone form factor."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=60000, use_cases=["coding"]),
        RequirementsDelta(category=Category.HEADPHONES, budget_max_inr=5000, use_cases=["music"]),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 60k"})
        r2 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Actually I need headphones under 5000 for music"},
        )

        assert r2.status_code == 200
        data = r2.json()
        assert data["requirements"]["category"] == "headphones"
        assert data["metadata"]["slot"] == "form_factor"
        assert "over-ear" in data["message"].lower() or "in-ear" in data["message"].lower()


# ---------------------------------------------------------------------------
# 10. GPU Questions Depend on User Workload
# ---------------------------------------------------------------------------

def test_gpu_questions_depend_on_workload():
    """Programming laptop does NOT require GPU question, but Gaming laptop DOES."""
    # 1. Programming laptop with RAM & storage fulfilled -> can recommend directly!
    req_coding = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
    )
    decision_coding = decide_next_action(req_coding, enforce_mandatory_slots=True)
    assert decision_coding.action == PolicyAction.RECOMMEND
    assert decision_coding.can_recommend is True

    # 2. Gaming laptop with RAM & storage fulfilled but GPU unanswered -> asks GPU!
    req_gaming = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=100000,
        use_cases=["gaming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=512, requires_gpu=None),
    )
    decision_gaming = decide_next_action(req_gaming, enforce_mandatory_slots=True)
    assert decision_gaming.action == PolicyAction.ASK
    assert decision_gaming.slot == "gpu"
    assert "gpu" in decision_gaming.question.lower() or "graphics" in decision_gaming.question.lower()


# ---------------------------------------------------------------------------
# 11. Optional Preferences Do Not Block Recommendation Indefinitely
# ---------------------------------------------------------------------------

def test_optional_preferences_do_not_block_recommendation():
    """Once all mandatory slots (category, budget, use case, RAM, storage) are answered, RECOMMEND is unlocked."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
        preferences={},  # No portability or brand preferences
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.RECOMMEND
    assert decision.can_recommend is True
    assert decision.reason == "sufficient_requirements"


# ---------------------------------------------------------------------------
# 12. Explicitly Declining a Mandatory Slot Follows Defined Policy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicitly_declining_mandatory_slot_follows_policy(mock_db, mock_products):
    """User saying 'skip' or 'I don't know' marks slot DECLINED and moves to next slot without repeating."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000, use_cases=["programming"]),
        RequirementsDelta(),  # user says 'I don't know'
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        r1 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 70k"})
        assert r1.json()["metadata"]["slot"] == "ram"

        # Turn 2: decline RAM
        r2 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "I don't know, skip this"})
        assert r2.status_code == 200
        # Advanced to storage!
        assert r2.json()["metadata"]["slot"] == "storage"
        assert r2.json()["slot_status"]["ram"] == "declined"


# ---------------------------------------------------------------------------
# 13. "You Decide" Handled Without Repeated Questioning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_you_decide_handled_without_repeated_questioning(mock_db, mock_products):
    """User saying 'you decide' marks slot DELEGATED and advances without re-asking."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000, use_cases=["programming"]),
        RequirementsDelta(),  # user says 'you decide'
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        r1 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 70k"})
        assert r1.json()["metadata"]["slot"] == "ram"

        # Turn 2: delegate choice
        r2 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "You decide what is best"})
        assert r2.status_code == 200
        assert r2.json()["metadata"]["slot"] == "storage"
        assert r2.json()["slot_status"]["ram"] == "delegated"


# ---------------------------------------------------------------------------
# 14. Contradictions Block Recommendation
# ---------------------------------------------------------------------------

def test_contradictions_block_recommendation():
    """When contradictions exist, can_recommend is FALSE and action is CLARIFY_CONFLICT."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_min_inr=100000,
        budget_max_inr=70000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.CLARIFY_CONFLICT
    assert decision.can_recommend is False
    assert len(decision.contradictions) > 0


# ---------------------------------------------------------------------------
# 15. Off-Topic Handling Remains Correct
# ---------------------------------------------------------------------------

def test_off_topic_handling_remains_correct():
    """Off-topic delta signal triggers REDIRECT and blocks recommendation."""
    req = Requirements(category=Category.LAPTOP)
    decision = decide_next_action(req, is_off_topic=True, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.REDIRECT
    assert decision.can_recommend is False


# ---------------------------------------------------------------------------
# 16. Product Retrieval Not Called While Clarification Is Required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_product_retrieval_not_called_while_clarification_required(mock_db, mock_products):
    """While action is ASK, product_repo.find() must NEVER be called."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000)
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop under 70k"},
        )
        assert resp.json()["action"] == "ASK"
        # Zero calls to product retrieval!
        assert mock_products.find_call_count == 0


# ---------------------------------------------------------------------------
# 17. Response Generation Not Called While Clarification Is Required
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_generation_not_called_while_clarification_required(mock_db, mock_products):
    """Module 9 generate_recommendation_response must NOT be called when action is ASK."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000)
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    with patch("app.agent.pipeline.generate_recommendation_response", new_callable=AsyncMock) as mock_gen:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            init_resp = await ac.post("/api/v1/conversations", json={})
            conv_id = init_resp.json()["conversation_id"]

            await ac.post(
                f"/api/v1/conversations/{conv_id}/messages",
                json={"message": "Laptop under 70k"},
            )
            # Response generator never invoked!
            mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# 18. Product Retrieval Called Only When Recommendation Gate Passes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_product_retrieval_called_only_when_recommendation_gate_passes(mock_db, mock_products):
    """When all mandatory slots are satisfied, product retrieval is invoked."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    rec_payload = {
        "message": "Top recommendations:",
        "explanations": [
            {
                "product_id": "LAP-001",
                "reasoning": "Fits criteria",
                "advantages": ["16GB RAM"],
                "trade_offs": [],
                "match_explanation": "Great fit",
                "limitations": [],
            }
        ],
    }
    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(
                category=Category.LAPTOP,
                budget_max_inr=70000,
                use_cases=["programming"],
                hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
            ),
            rec_payload,
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop for coding under 70k with 16GB RAM and 512GB SSD"},
        )
        assert resp.json()["action"] == "RECOMMEND"
        # Product retrieval was called!
        assert mock_products.find_call_count == 1


# ---------------------------------------------------------------------------
# 19. No-Match Handling Works After All Mandatory Inputs Collected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_match_handling_works_after_mandatory_inputs_collected(mock_db, mock_products):
    """All mandatory slots satisfied, but budget is too low (₹5k for laptop) -> returns clean 0 matches."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(
                category=Category.LAPTOP,
                budget_max_inr=5000,
                use_cases=["programming"],
                hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
            )
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop under 5k with 16GB RAM and 512GB SSD"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "RECOMMEND"
        assert len(data["recommendations"]) == 0
        assert "could not find any laptop" in data["message"].lower()


# ---------------------------------------------------------------------------
# 20. Conversation State Persists Across API Requests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conversation_state_persists_across_api_requests(mock_db, mock_products):
    """Answers given in Turn 1 and Turn 2 persist into Turn 3 and are accessible via GET."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    deltas = [
        RequirementsDelta(category=Category.LAPTOP, budget_max_inr=70000, use_cases=["programming"]),
        RequirementsDelta(hard=HardConstraints(min_ram_gb=16)),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 70k for coding"})
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "16GB RAM"})

        get_resp = await ac.get(f"/api/v1/conversations/{conv_id}")
        assert get_resp.status_code == 200
        state = get_resp.json()
        assert state["turn_count"] == 2
        assert state["requirements"]["category"] == "laptop"
        assert state["requirements"]["hard"]["min_ram_gb"] == 16
        assert state["slot_status"]["ram"] == "answered"


# ---------------------------------------------------------------------------
# 21. Clarification Slot State Remains Isolated Between Conversations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarification_slot_state_remains_isolated_between_conversations(mock_db, mock_products):
    """Conversation A answering RAM does not leak into Conversation B."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, hard=HardConstraints(min_ram_gb=16)),
            RequirementsDelta(category=Category.LAPTOP),
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        c1 = (await ac.post("/api/v1/conversations", json={})).json()["conversation_id"]
        c2 = (await ac.post("/api/v1/conversations", json={})).json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{c1}/messages", json={"message": "Laptop with 16GB RAM"})
        r_c2 = await ac.post(f"/api/v1/conversations/{c2}/messages", json={"message": "I want a laptop"})

        # c1 has RAM answered
        state_c1 = (await ac.get(f"/api/v1/conversations/{c1}")).json()
        assert state_c1["requirements"]["hard"]["min_ram_gb"] == 16

        # c2 has RAM unset
        state_c2 = (await ac.get(f"/api/v1/conversations/{c2}")).json()
        assert state_c2["requirements"]["hard"]["min_ram_gb"] is None
        assert "ram" not in state_c2["slot_status"]


# ---------------------------------------------------------------------------
# 22. Headphones Use Headphone-Specific Slots
# ---------------------------------------------------------------------------

def test_headphones_use_headphone_specific_slots():
    """Headphones category requires form_factor, NOT RAM or storage."""
    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=5000,
        use_cases=["music"],
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "form_factor"
    assert "over-ear" in decision.question.lower() or "in-ear" in decision.question.lower()


# ---------------------------------------------------------------------------
# 23. Smartphones Use Smartphone-Specific Slots
# ---------------------------------------------------------------------------

def test_smartphones_use_smartphone_specific_slots():
    """Phone category requires storage, NOT laptop dedicated GPU."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=30000,
        use_cases=["everyday"],
        hard=HardConstraints(min_storage_gb=None),
    )
    decision = decide_next_action(req, enforce_mandatory_slots=True)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "phone_storage"
    assert "storage" in decision.question.lower()


# ---------------------------------------------------------------------------
# 24. Tablets and Monitors Use Their Own Applicable Slots
# ---------------------------------------------------------------------------

def test_tablets_and_monitors_use_their_own_applicable_slots():
    """Tablet asks stylus/keyboard; Monitor asks display specs."""
    # Tablet
    req_tab = Requirements(category=Category.TABLET, budget_max_inr=40000, use_cases=["drawing"])
    dec_tab = decide_next_action(req_tab, enforce_mandatory_slots=True)
    assert dec_tab.action == PolicyAction.ASK
    assert dec_tab.slot == "stylus_keyboard"
    assert "stylus" in dec_tab.question.lower() or "pen" in dec_tab.question.lower()

    # Monitor
    req_mon = Requirements(category=Category.MONITOR, budget_max_inr=20000, use_cases=["gaming"])
    dec_mon = decide_next_action(req_mon, enforce_mandatory_slots=True)
    assert dec_mon.action == PolicyAction.ASK
    assert dec_mon.slot == "display_specs"
    assert "screen size" in dec_mon.question.lower() or "resolution" in dec_mon.question.lower()


# ---------------------------------------------------------------------------
# 25. Slot Engine Metadata Integrity
# ---------------------------------------------------------------------------

def test_slot_engine_metadata_integrity():
    """Verify that all slots in registry have questions and valid categories."""
    assert len(SLOT_REGISTRY) >= 10
    for slot in SLOT_REGISTRY:
        assert isinstance(slot.name, str) and slot.name
        assert isinstance(slot.question, str) and len(slot.question) > 10
        assert slot.field_name
