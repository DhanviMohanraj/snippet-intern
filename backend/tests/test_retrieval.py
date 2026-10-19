"""Unit and offline tests for Module 8 candidate retrieval engine."""

from __future__ import annotations

import copy
import pytest
from unittest.mock import AsyncMock

from app.agent.retrieval import DEFAULT_CANDIDATE_LIMIT, retrieve_candidates
from app.db.repositories import ProductRepository
from app.models.product import Category, Product
from app.models.requirements import Requirements


# ---------------------------------------------------------------------------
# Fixtures and Fakes
# ---------------------------------------------------------------------------

def make_sample_products() -> list[Product]:
    """Create sample validated in-memory products across multiple categories."""
    return [
        Product(
            product_id="LAP-0001",
            name="Zenit Air 14",
            brand="Zenit",
            category=Category.LAPTOP,
            price_inr=54999,
            specs={
                "cpu": "Intel i5",
                "ram_gb": 16,
                "storage_gb": 512,
                "vram_gb": 0,
                "gpu": "Intel Iris Xe",
                "weight_kg": 1.35,
                "battery_hr": 10.0,
            },
            intended_use=["programming", "student"],
            features=["Backlit Keyboard"],
            in_stock=True,
            rating=4.5,
        ),
        Product(
            product_id="LAP-0002",
            name="Apex Nitro G15",
            brand="Apex",
            category=Category.LAPTOP,
            price_inr=74999,
            specs={
                "cpu": "AMD Ryzen 7",
                "ram_gb": 16,
                "storage_gb": 512,
                "vram_gb": 6,
                "gpu": "NVIDIA GeForce RTX 4050",
                "weight_kg": 2.25,
                "battery_hr": 7.0,
            },
            intended_use=["gaming"],
            features=["144Hz IPS"],
            in_stock=True,
            rating=4.3,
        ),
        Product(
            product_id="PHN-0001",
            name="Nova Phone 12",
            brand="Nova",
            category=Category.PHONE,
            price_inr=29999,
            specs={
                "chipset": "Snapdragon 778G",
                "ram_gb": 8,
                "storage_gb": 128,
                "camera_mp": 50.0,
                "battery_mah": 5000,
                "weight_g": 185.0,
            },
            intended_use=["everyday", "photography"],
            features=["AMOLED 120Hz"],
            in_stock=True,
            rating=4.4,
        ),
    ]


class FakeProductRepository:
    """Deterministic in-memory ProductRepository fake for offline testing."""

    def __init__(self, products: list[Product] | None = None) -> None:
        self.products = products or []
        self.last_filters: dict | None = None
        self.last_page_size: int | None = None

    async def find(
        self,
        filters: dict | None = None,
        sort: str = "price_inr_asc",
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Product], int]:
        self.last_filters = filters
        self.last_page_size = page_size
        results = list(self.products)

        if filters and "category" in filters:
            cat = filters["category"]
            cat_val = cat.value if isinstance(cat, Category) else cat
            results = [p for p in results if p.category.value == cat_val]

        return results[:page_size], len(results)


# ---------------------------------------------------------------------------
# Retrieval Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_candidates_matching_category():
    """retrieve_candidates fetches only products matching the requested category."""
    sample = make_sample_products()
    repo = FakeProductRepository(sample)
    req = Requirements(category=Category.LAPTOP)

    candidates = await retrieve_candidates(req, repo)

    assert len(candidates) == 2
    assert all(c.category == Category.LAPTOP for c in candidates)
    assert candidates[0].product_id == "LAP-0001"
    assert candidates[1].product_id == "LAP-0002"
    assert repo.last_filters == {"category": Category.LAPTOP}


@pytest.mark.asyncio
async def test_retrieve_candidates_empty_when_category_none():
    """retrieve_candidates immediately returns empty list when requirements.category is None."""
    repo = FakeProductRepository(make_sample_products())
    req = Requirements(category=None)

    candidates = await retrieve_candidates(req, repo)

    assert candidates == []
    assert repo.last_filters is None


@pytest.mark.asyncio
async def test_retrieve_candidates_empty_category_results():
    """When the catalogue has zero items for a category, returns empty list without error."""
    repo = FakeProductRepository(make_sample_products())
    req = Requirements(category=Category.MONITOR)

    candidates = await retrieve_candidates(req, repo)

    assert candidates == []
    assert repo.last_filters == {"category": Category.MONITOR}


@pytest.mark.asyncio
async def test_retrieve_candidates_limit_parameter_respected():
    """Candidate retrieval limit is respected and clamped safely."""
    sample = make_sample_products()
    repo = FakeProductRepository(sample)
    req = Requirements(category=Category.LAPTOP)

    # Request limit of 1
    candidates = await retrieve_candidates(req, repo, limit=1)
    assert len(candidates) == 1
    assert repo.last_page_size == 1

    # Request limit > 50 (must clamp to 50 for ProductRepository compatibility)
    await retrieve_candidates(req, repo, limit=100)
    assert repo.last_page_size == 50


@pytest.mark.asyncio
async def test_retrieve_candidates_propagates_repository_errors():
    """Database/repository errors propagate directly and are not masked as empty results."""
    mock_repo = AsyncMock(spec=ProductRepository)
    mock_repo.find.side_effect = RuntimeError("MongoDB connection dropped")
    req = Requirements(category=Category.LAPTOP)

    with pytest.raises(RuntimeError, match="MongoDB connection dropped"):
        await retrieve_candidates(req, mock_repo)


@pytest.mark.asyncio
async def test_retrieve_candidates_does_not_mutate_inputs():
    """retrieve_candidates leaves input requirements state pristine."""
    repo = FakeProductRepository(make_sample_products())
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=80000,
        use_cases=["programming"],
        asked_slots=["category"],
    )
    req_copy = copy.deepcopy(req)

    await retrieve_candidates(req, repo)

    assert req == req_copy
