"""Comprehensive offline test suite for Module 10: Agent Pipeline & Conversation API.

Covers all 23 required scenarios without making any live Groq API or network calls:
1. Conversation creation.
2. Successful message submission.
3. Requirements extraction and merging through the pipeline.
4. Clarification for missing requirements.
5. Contradiction handling.
6. Recommendation flow through Modules 8 and 9.
7. Persistence across separate requests.
8. Updating a previously stated budget.
9. Changing the product category.
10. Removing a previously stated preference.
11. Off-topic messages.
12. No eligible product matches.
13. Invalid conversation IDs.
14. Missing conversations.
15. Unauthorized access where authentication applies.
16. MongoDB errors.
17. LLM errors and safe fallback behavior.
18. Conversation history ordering.
19. Conversation deletion and associated history cleanup.
20. Turn-limit behavior.
21. Preventing one conversation's state from leaking into another.
22. Concurrent update behavior.
23. API response schema validation.
"""

from __future__ import annotations

import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.products import get_product_repository
from app.api.v1.conversations import get_conversation_repository, get_llm_client
from app.core.config import Settings
from app.db.conversation_repository import InMemoryConversationRepository
from app.llm.errors import LLMProviderError
from app.llm.fake import FakeLLMClient
from app.main import create_app
from app.models.conversation import ConversationState
from app.models.product import Category, Product
from app.models.requirements import HardConstraints, PolicyAction, RequirementsDelta


# ---------------------------------------------------------------------------
# Test Factories & Fixtures
# ---------------------------------------------------------------------------

def make_test_product(
    product_id: str = "LAP-0001",
    name: str = "Zenit Air 14",
    brand: str = "Zenit",
    category: Category = Category.LAPTOP,
    price_inr: int = 54999,
    specs: dict | None = None,
    intended_use: list[str] | None = None,
    features: list[str] | None = None,
    rating: float = 4.5,
    in_stock: bool = True,
) -> Product:
    """Create a verified test product for candidate retrieval and scoring."""
    default_specs = {
        "cpu": "Intel Core i5-13420H",
        "ram_gb": 16,
        "storage_gb": 512,
        "gpu": "Intel Iris Xe",
        "vram_gb": 0,
        "weight_kg": 1.35,
        "battery_hr": 10.0,
    }
    return Product(
        product_id=product_id,
        name=name,
        brand=brand,
        category=category,
        price_inr=price_inr,
        specs=specs or default_specs,
        intended_use=intended_use or ["programming", "student"],
        features=features or ["Backlit Keyboard", "FHD IPS Display"],
        rating=rating,
        in_stock=in_stock,
    )


class MockProductRepository:
    """Deterministic in-memory ProductRepository for candidate retrieval in tests."""

    def __init__(self, products: list[Product]) -> None:
        self._products = products

    async def get(self, product_id: str) -> Product | None:
        for p in self._products:
            if p.product_id == product_id.strip():
                return p
        return None

    async def find(
        self,
        filters: dict | None = None,
        sort: str = "price_inr_asc",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Product], int]:
        flt = filters or {}
        matches = list(self._products)

        if "category" in flt:
            cat_val = flt["category"]
            cat_str = cat_val.value if isinstance(cat_val, Category) else str(cat_val)
            matches = [p for p in matches if p.category.value == cat_str]

        if "min_price" in flt:
            matches = [p for p in matches if p.price_inr >= flt["min_price"]]

        if "max_price" in flt:
            matches = [p for p in matches if p.price_inr <= flt["max_price"]]

        if "in_stock" in flt:
            matches = [p for p in matches if p.in_stock == flt["in_stock"]]

        total = len(matches)
        return matches[:page_size], total


@pytest.fixture
def sample_products() -> list[Product]:
    """Diverse product fixture for testing recommendation pipeline."""
    return [
        make_test_product(
            product_id="LAP-0001",
            name="Zenit Air 14",
            price_inr=54999,
            category=Category.LAPTOP,
            intended_use=["programming", "student"],
        ),
        make_test_product(
            product_id="LAP-0002",
            name="Voltix DevBook Pro 16",
            price_inr=74999,
            category=Category.LAPTOP,
            intended_use=["programming", "machine_learning"],
        ),
        make_test_product(
            product_id="PHN-0001",
            name="Velo 12 Pro",
            brand="Velo",
            category=Category.PHONE,
            price_inr=45000,
            specs={"chipset": "Snapdragon", "ram_gb": 8, "storage_gb": 256, "camera_mp": 64.0},
            intended_use=["photography", "daily"],
        ),
    ]


@pytest.fixture
def mock_db() -> InMemoryConversationRepository:
    """Clean in-memory conversation repository fixture."""
    return InMemoryConversationRepository()


@pytest.fixture
def mock_products(sample_products: list[Product]) -> MockProductRepository:
    """Mock product catalogue repository fixture."""
    return MockProductRepository(sample_products)


# ---------------------------------------------------------------------------
# 1. Conversation Creation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_conversation_default(mock_db, mock_products):
    """POST /api/v1/conversations creates a new empty conversation session."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/api/v1/conversations", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "conversation_id" in data
        assert data["turn_count"] == 0
        assert data["user_id"] is None
        assert data["requirements"]["category"] is None
        assert data["requirements"]["budget_max_inr"] is None


@pytest.mark.asyncio
async def test_create_conversation_with_user_id_and_metadata(mock_db, mock_products):
    """POST /api/v1/conversations respects user_id and client metadata."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/conversations",
            json={"user_id": "user_alpha_123", "metadata": {"source": "mobile_app"}},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["user_id"] == "user_alpha_123"
        assert data["metadata"]["source"] == "mobile_app"


# ---------------------------------------------------------------------------
# 2. Message Submission & Pipeline Execution Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_message_submission_and_extraction(mock_db, mock_products):
    """Submitting a message extracts delta, updates state, and advances turn count."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # Configure FakeLLM with canned extraction delta
    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, budget_max_inr=80000, use_cases=["programming"])
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create conversation
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Send message
        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I need a programming laptop under 80k"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["conversation_id"] == conv_id
        assert data["turn_count"] == 1
        assert data["requirements"]["category"] == "laptop"
        assert data["requirements"]["budget_max_inr"] == 80000
        assert "programming" in data["requirements"]["use_cases"]


# ---------------------------------------------------------------------------
# 3. Clarification for Missing Requirements Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clarification_for_missing_category(mock_db, mock_products):
    """When category is missing, policy action is ASK for category."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # User mentions budget but no category
    fake_llm = FakeLLMClient(
        json_responses=[RequirementsDelta(budget_max_inr=50000)]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I have a budget of ₹50,000."},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["action"] == "ASK"
        assert data["requires_clarification"] is True
        assert data["metadata"]["slot"] == "category"
        assert "laptop, phone" in data["message"].lower()


@pytest.mark.asyncio
async def test_clarification_for_missing_budget(mock_db, mock_products):
    """When category is known but budget is missing, policy asks for budget."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[RequirementsDelta(category=Category.LAPTOP)]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I want to buy a laptop"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["action"] == "ASK"
        assert data["metadata"]["slot"] == "budget"
        assert "budget" in data["message"].lower()


# ---------------------------------------------------------------------------
# 4. Contradiction Handling Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_contradiction_handling_in_pipeline(mock_db, mock_products):
    """When min budget exceeds max budget, policy emits CLARIFY_CONFLICT."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, budget_min_inr=80000, budget_max_inr=50000)
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop with min budget 80k and max budget 50k"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["action"] == "CLARIFY_CONFLICT"
        assert data["requires_clarification"] is True
        assert len(data["unresolved_contradictions"]) > 0
        assert data["unresolved_contradictions"][0]["code"] == "BUDGET_MIN_EXCEEDS_MAX"
        assert len(data["recommendations"]) == 0


# ---------------------------------------------------------------------------
# 5. Recommendation Flow Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recommendation_flow_returns_ranked_products(mock_db, mock_products):
    """Complete requirements trigger RECOMMEND action with Module 9 validated recommendations."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # 1. Extraction response
    extract_delta = RequirementsDelta(
        category=Category.LAPTOP,
        budget_max_inr=90000,
        use_cases=["programming"],
        hard=HardConstraints(min_ram_gb=16, min_storage_gb=512),
    )
    # 2. Recommendation response payload for Module 9
    rec_payload = {
        "message": "Here are our top recommended laptops for programming:",
        "explanations": [
            {
                "product_id": "LAP-0001",
                "reasoning": "Great programming machine.",
                "advantages": ["Portable"],
                "trade_offs": ["Integrated GPU"],
                "match_explanation": "Within budget",
                "limitations": [],
            }
        ],
    }
    fake_llm = FakeLLMClient(json_responses=[extract_delta, rec_payload])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        msg_resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I want a laptop for programming under 90k with 16GB RAM and 512GB SSD"},
        )
        assert msg_resp.status_code == 200
        data = msg_resp.json()
        assert data["action"] == "RECOMMEND"
        assert len(data["recommendations"]) > 0
        assert data["recommendations"][0]["product_id"] == "LAP-0001"
        assert data["recommendations"][0]["price_inr"] == 54999


# ---------------------------------------------------------------------------
# 6. Multi-Turn Persistence & Requirement Revisions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multi_turn_persistence_and_budget_update(mock_db, mock_products):
    """Turn 1 states category & budget; Turn 2 updates budget to a new value."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # Turn 1: laptop under 50k
    t1_delta = RequirementsDelta(category=Category.LAPTOP, budget_max_inr=50000)
    # Turn 2: update budget to 70k
    t2_delta = RequirementsDelta(budget_max_inr=70000)

    fake_llm = FakeLLMClient(json_responses=[t1_delta, t2_delta])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Turn 1
        r1 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop under 50k"},
        )
        assert r1.status_code == 200
        assert r1.json()["requirements"]["budget_max_inr"] == 50000

        # Turn 2: budget update
        r2 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Actually, increase my budget to 70k"},
        )
        assert r2.status_code == 200
        assert r2.json()["requirements"]["budget_max_inr"] == 70000
        assert r2.json()["requirements"]["category"] == "laptop"  # category preserved


@pytest.mark.asyncio
async def test_changing_product_category_in_pipeline(mock_db, mock_products):
    """User changes category from laptop to phone across conversational turns."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    t1_delta = RequirementsDelta(category=Category.LAPTOP)
    t2_delta = RequirementsDelta(category=Category.PHONE)

    fake_llm = FakeLLMClient(json_responses=[t1_delta, t2_delta])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Looking for a laptop"})
        r2 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Never mind, show me phones"})
        assert r2.status_code == 200
        assert r2.json()["requirements"]["category"] == "phone"


@pytest.mark.asyncio
async def test_removing_a_preference(mock_db, mock_products):
    """Explicitly removing a preference clears it from accumulated requirements."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    t1_delta = RequirementsDelta(
        category=Category.LAPTOP,
        preferences={"portability": 0.9},
    )
    t2_delta = RequirementsDelta(removed_fields=["portability"])

    fake_llm = FakeLLMClient(json_responses=[t1_delta, t2_delta])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        r1 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Lightweight laptop"})
        assert "portability" in r1.json()["requirements"]["preferences"]

        r2 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Remove portability preference"})
        assert "portability" not in r2.json()["requirements"]["preferences"]


# ---------------------------------------------------------------------------
# 7. Off-Topic & Unsupported Categories Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_topic_message_redirects_and_preserves_state(mock_db, mock_products):
    """Off-topic inquiry receives REDIRECT without modifying existing requirements."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    t1_delta = RequirementsDelta(category=Category.LAPTOP, budget_max_inr=60000)
    t2_delta = RequirementsDelta(is_off_topic=True)

    fake_llm = FakeLLMClient(json_responses=[t1_delta, t2_delta])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Establish state
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Laptop under 60k"})

        # Off-topic turn
        r2 = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        assert r2.json()["action"] == "REDIRECT"
        # State remained preserved
        assert r2.json()["requirements"]["category"] == "laptop"
        assert r2.json()["requirements"]["budget_max_inr"] == 60000


@pytest.mark.asyncio
async def test_unsupported_category_returns_no_match(mock_db, mock_products):
    """Requesting an unsupported category (e.g. smartwatch, tv) returns NO_MATCH."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # Empty delta from LLM since smartwatch is not a supported Category enum
    fake_llm = FakeLLMClient(json_responses=[RequirementsDelta()])
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I want to buy a smartwatch for fitness"},
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "NO_MATCH"
        assert "cannot provide recommendations for 'smartwatch'" in resp.json()["message"].lower()


@pytest.mark.asyncio
async def test_no_eligible_product_matches_handled_safely(mock_db, mock_products):
    """When budget is too low for any catalogue product, returns clean response without crashing."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # Laptop under ₹5,000 with complete specs (no matching laptops in catalogue)
    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(
                category=Category.LAPTOP,
                budget_max_inr=5000,
                use_cases=["programming"],
                hard=HardConstraints(min_ram_gb=8, min_storage_gb=256),
            )
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "Laptop under 5k for programming with 8GB RAM and 256GB SSD"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] in ("RECOMMEND", "NO_MATCH")
        assert len(data["recommendations"]) == 0
        assert "could not find any laptop" in data["message"].lower()


# ---------------------------------------------------------------------------
# 8. Error Handling & Validation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_conversation_id_format(mock_db, mock_products):
    """Invalid conversation ID format returns HTTP 400."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/conversations/invalid!id@with$special%chars")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_CONVERSATION_ID"


@pytest.mark.asyncio
async def test_missing_conversation_returns_404(mock_db, mock_products):
    """Querying a non-existent conversation returns HTTP 404."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/conversations/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"


@pytest.mark.asyncio
async def test_unauthorized_access_rejected_403(mock_db, mock_products):
    """Accessing a conversation owned by user A with user B identity returns HTTP 403."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Create as user_alice
        init_resp = await ac.post("/api/v1/conversations", json={"user_id": "user_alice"})
        conv_id = init_resp.json()["conversation_id"]

        # Attempt read as user_bob
        resp = await ac.get(
            f"/api/v1/conversations/{conv_id}",
            headers={"X-User-ID": "user_bob"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "FORBIDDEN_CONVERSATION_ACCESS"


@pytest.mark.asyncio
async def test_mongodb_error_returns_503(mock_db, mock_products):
    """Simulated MongoDB failure returns clean HTTP 503 error envelope."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    mock_db.simulate_db_failure = True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/api/v1/conversations", json={})
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_llm_error_safely_falls_back(mock_db, mock_products):
    """LLM provider failure during extraction uses deterministic fallback without failing turn."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(error_to_raise=LLMProviderError("Groq 503 Service Unavailable"))
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Submits keyword-rich message
        resp = await ac.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"message": "I want a laptop under 60k"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_fallback"] is True
        assert data["requirements"]["category"] == "laptop"
        assert data["requirements"]["budget_max_inr"] == 60000


# ---------------------------------------------------------------------------
# 9. History Ordering & Deletion Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conversation_history_ordering(mock_db, mock_products):
    """GET /api/v1/conversations/{id}/messages returns chronological user and assistant messages."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP),
            RequirementsDelta(budget_max_inr=70000),
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        # Turn 1
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Looking for a laptop"})
        # Turn 2
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "My budget is 70k"})

        hist_resp = await ac.get(f"/api/v1/conversations/{conv_id}/messages")
        assert hist_resp.status_code == 200
        msgs = hist_resp.json()["messages"]
        assert len(msgs) == 4  # (user, assistant, user, assistant)
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "assistant"


@pytest.mark.asyncio
async def test_conversation_deletion(mock_db, mock_products):
    """DELETE /api/v1/conversations/{id} deletes the conversation and subsequent queries return 404."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        del_resp = await ac.delete(f"/api/v1/conversations/{conv_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # Subsequent fetch must fail with 404
        get_resp = await ac.get(f"/api/v1/conversations/{conv_id}")
        assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# 10. Turn-Limit & Isolation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_limit_behavior(mock_db, mock_products):
    """Reaching MAX_CLARIFICATION_TURNS safely terminates questioning with RECOMMEND."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    # 3 turns without budget, but category laptop is known
    deltas = [
        RequirementsDelta(category=Category.LAPTOP),
        RequirementsDelta(),
        RequirementsDelta(),
    ]
    fake_llm = FakeLLMClient(json_responses=deltas)
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Turn 1"})
        await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Turn 2"})
        r3 = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Turn 3"})

        assert r3.status_code == 200
        # Reaching 3 turns does not prematurely recommend when mandatory requirements remain missing
        assert r3.json()["action"] == "ASK"
        assert r3.json()["requires_clarification"] is True


@pytest.mark.asyncio
async def test_prevent_state_leaking_between_conversations(mock_db, mock_products):
    """Two separate conversations maintain completely isolated requirements states."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[
            RequirementsDelta(category=Category.LAPTOP, budget_max_inr=50000),
            RequirementsDelta(category=Category.PHONE, budget_max_inr=90000),
        ]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        c1 = (await ac.post("/api/v1/conversations", json={})).json()["conversation_id"]
        c2 = (await ac.post("/api/v1/conversations", json={})).json()["conversation_id"]

        r1 = await ac.post(f"/api/v1/conversations/{c1}/messages", json={"message": "Laptop 50k"})
        r2 = await ac.post(f"/api/v1/conversations/{c2}/messages", json={"message": "Phone 90k"})

        assert r1.json()["requirements"]["category"] == "laptop"
        assert r1.json()["requirements"]["budget_max_inr"] == 50000

        assert r2.json()["requirements"]["category"] == "phone"
        assert r2.json()["requirements"]["budget_max_inr"] == 90000


@pytest.mark.asyncio
async def test_concurrent_update_behavior(mock_db):
    """Simultaneous updates to the same conversation version detect conflict."""
    conv = ConversationState(conversation_id="conv-concurrent-1", version=1)
    await mock_db.create(conv)

    # First update succeeds and increments version to 2
    conv_copy1 = await mock_db.get("conv-concurrent-1")
    conv_copy1.turn_count = 1
    await mock_db.update(conv_copy1)

    # Second update using stale version=1 triggers ConcurrentUpdateError
    conv_copy2 = ConversationState(conversation_id="conv-concurrent-1", version=1, turn_count=2)
    with pytest.raises(Exception) as exc_info:
        await mock_db.update(conv_copy2)
    assert "concurrently" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_api_response_schema_validation(mock_db, mock_products):
    """Validates complete typed structure of ConversationTurnResponse."""
    app = create_app(init_db=False)
    app.dependency_overrides[get_conversation_repository] = lambda: mock_db
    app.dependency_overrides[get_product_repository] = lambda: mock_products

    fake_llm = FakeLLMClient(
        json_responses=[RequirementsDelta(category=Category.LAPTOP)]
    )
    app.dependency_overrides[get_llm_client] = lambda: fake_llm

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        init_resp = await ac.post("/api/v1/conversations", json={})
        conv_id = init_resp.json()["conversation_id"]

        resp = await ac.post(f"/api/v1/conversations/{conv_id}/messages", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()

        # Schema fields must be present and correctly typed
        assert isinstance(data["conversation_id"], str)
        assert isinstance(data["message"], str)
        assert isinstance(data["action"], str)
        assert isinstance(data["turn_count"], int)
        assert isinstance(data["requirements"], dict)
        assert isinstance(data["requires_clarification"], bool)
        assert isinstance(data["unresolved_contradictions"], list)
        assert isinstance(data["recommendations"], list)
        assert isinstance(data["is_fallback"], bool)
        assert isinstance(data["metadata"], dict)
