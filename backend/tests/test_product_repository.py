"""Integration and unit tests for the ProductRepository class against local MongoDB."""

from __future__ import annotations

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import get_settings
from app.db.repositories import ProductRepository
from app.models.product import Category, Product

# Use dedicated test database name to keep development data completely untouched
TEST_DB_NAME = "electronics_agent_test"


@pytest.fixture
async def test_repo():
    """Create a ProductRepository connected to an isolated test MongoDB database."""
    cfg = get_settings()
    client = AsyncIOMotorClient(cfg.mongo_uri, serverSelectionTimeoutMS=2000)
    db = client[TEST_DB_NAME]
    collection = db["test_products"]

    # Clear test collection before test
    await collection.delete_many({})

    # Seed sample products
    sample_products = [
        Product(
            product_id="LAP-0001",
            name="Zenit Air 14",
            brand="Zenit",
            category=Category.LAPTOP,
            price_inr=54999,
            specs={"cpu": "Intel i5", "ram_gb": 16, "storage_gb": 512, "vram_gb": 0, "gpu": "Intel Iris Xe", "weight_kg": 1.35},
            intended_use=["programming", "student"],
            features=["Backlit Keyboard"],
            in_stock=True,
            rating=4.5,
        ),
        Product(
            product_id="LAP-0002",
            name="Voltix DevBook Pro 16",
            brand="Voltix",
            category=Category.LAPTOP,
            price_inr=112000,
            specs={"cpu": "AMD Ryzen 9", "ram_gb": 32, "storage_gb": 1024, "vram_gb": 0, "gpu": "AMD Radeon 780M", "weight_kg": 1.78},
            intended_use=["programming", "machine_learning"],
            features=["120Hz Display"],
            in_stock=True,
            rating=4.8,
        ),
        Product(
            product_id="LAP-0003",
            name="Apex Nitro G15",
            brand="Apex",
            category=Category.LAPTOP,
            price_inr=74999,
            specs={"cpu": "AMD Ryzen 7", "ram_gb": 16, "storage_gb": 512, "vram_gb": 6, "gpu": "NVIDIA GeForce RTX 4050", "weight_kg": 2.25},
            intended_use=["gaming"],
            features=["144Hz Screen"],
            in_stock=False,  # Out of stock
            rating=4.2,
        ),
        Product(
            product_id="PHN-0001",
            name="Velo Ultra 15 Pro",
            brand="Velo",
            category=Category.PHONE,
            price_inr=119999,
            specs={"chipset": "Snapdragon 8 Gen 3", "ram_gb": 16, "storage_gb": 512, "camera_mp": 200.0, "battery_mah": 5000},
            intended_use=["photography", "gaming"],
            features=["200MP Camera"],
            in_stock=True,
            rating=4.8,
        ),
        Product(
            product_id="HPH-0001",
            name="Sonix QuietMaster 100",
            brand="Sonix",
            category=Category.HEADPHONES,
            price_inr=29999,
            specs={"type": "over-ear", "anc": True, "battery_hr": 35.0, "wireless": True, "weight_g": 250},
            intended_use=["music", "travel"],
            features=["ANC"],
            in_stock=True,
            rating=4.7,
        ),
    ]

    for p in sample_products:
        await collection.insert_one(p.model_dump())

    repo = ProductRepository(collection)
    await repo.create_indexes()

    yield repo

    # Cleanup test database
    await client.drop_database(TEST_DB_NAME)
    client.close()


@pytest.mark.asyncio
async def test_get_existing_product(test_repo: ProductRepository):
    """Verify get() returns a validated Product instance when product exists."""
    product = await test_repo.get("LAP-0001")
    assert product is not None
    assert isinstance(product, Product)
    assert product.product_id == "LAP-0001"
    assert product.name == "Zenit Air 14"
    assert product.price_inr == 54999


@pytest.mark.asyncio
async def test_get_nonexistent_and_empty_id(test_repo: ProductRepository):
    """Verify get() returns None for missing or blank IDs."""
    assert await test_repo.get("NONEXISTENT-ID") is None
    assert await test_repo.get("") is None
    assert await test_repo.get("   ") is None


@pytest.mark.asyncio
async def test_find_with_pagination(test_repo: ProductRepository):
    """Verify find() pagination and total count calculation."""
    # Page 1 with page_size=2
    products, total = await test_repo.find(page=1, page_size=2)
    assert total == 5
    assert len(products) == 2

    # Page 2 with page_size=2
    page2_products, total2 = await test_repo.find(page=2, page_size=2)
    assert total2 == 5
    assert len(page2_products) == 2
    assert page2_products[0].product_id != products[0].product_id

    # Page 3 with page_size=2 (only 1 product remaining)
    page3_products, total3 = await test_repo.find(page=3, page_size=2)
    assert total3 == 5
    assert len(page3_products) == 1


@pytest.mark.asyncio
async def test_find_invalid_pagination_parameters(test_repo: ProductRepository):
    """Verify invalid pagination values raise ValueError."""
    with pytest.raises(ValueError):
        await test_repo.find(page=0)

    with pytest.raises(ValueError):
        await test_repo.find(page_size=0)

    with pytest.raises(ValueError):
        await test_repo.find(page_size=51)


@pytest.mark.asyncio
async def test_find_sorting(test_repo: ProductRepository):
    """Verify deterministic sorting on price and rating."""
    # Price ascending
    products_asc, _ = await test_repo.find(sort="price_asc")
    prices_asc = [p.price_inr for p in products_asc]
    assert prices_asc == sorted(prices_asc)

    # Price descending
    products_desc, _ = await test_repo.find(sort="price_desc")
    prices_desc = [p.price_inr for p in products_desc]
    assert prices_desc == sorted(prices_desc, reverse=True)

    # Invalid sort option
    with pytest.raises(ValueError):
        await test_repo.find(sort="invalid_sort_field")


@pytest.mark.asyncio
async def test_find_category_filter(test_repo: ProductRepository):
    """Verify filtering by Category enum and string."""
    laptops, total = await test_repo.find(filters={"category": "laptop"})
    assert total == 3
    assert len(laptops) == 3
    assert all(p.category == Category.LAPTOP for p in laptops)

    phones, total_phn = await test_repo.find(filters={"category": Category.PHONE})
    assert total_phn == 1
    assert phones[0].category == Category.PHONE


@pytest.mark.asyncio
async def test_find_price_range_inclusive(test_repo: ProductRepository):
    """Verify inclusive min_price and max_price boundaries."""
    # 54999 <= price <= 74999 (should match Zenit Air 14 and Apex Nitro G15)
    products, total = await test_repo.find(
        filters={"min_price": 54999, "max_price": 74999}
    )
    assert total == 2
    matched_ids = {p.product_id for p in products}
    assert matched_ids == {"LAP-0001", "LAP-0003"}

    # max_price < min_price raises ValueError
    with pytest.raises(ValueError):
        await test_repo.find(filters={"min_price": 80000, "max_price": 50000})


@pytest.mark.asyncio
async def test_find_use_case_filter(test_repo: ProductRepository):
    """Verify filtering by use_case matching items inside intended_use list."""
    progs, total = await test_repo.find(filters={"use_case": "programming"})
    assert total == 2
    assert all("programming" in p.intended_use for p in progs)

    music_items, total_music = await test_repo.find(filters={"use_case": "music"})
    assert total_music == 1
    assert music_items[0].product_id == "HPH-0001"


@pytest.mark.asyncio
async def test_find_brand_filter(test_repo: ProductRepository):
    """Verify brand filtering with case-insensitivity."""
    zenit_products, total = await test_repo.find(filters={"brand": "zenit"})
    assert total == 1
    assert zenit_products[0].brand == "Zenit"


@pytest.mark.asyncio
async def test_find_in_stock_filter(test_repo: ProductRepository):
    """Verify filtering on in_stock True and in_stock False."""
    in_stock_items, in_stock_count = await test_repo.find(filters={"in_stock": True})
    assert in_stock_count == 4
    assert all(p.in_stock is True for p in in_stock_items)

    out_of_stock_items, oos_count = await test_repo.find(filters={"in_stock": False})
    assert oos_count == 1
    assert out_of_stock_items[0].product_id == "LAP-0003"


@pytest.mark.asyncio
async def test_unsupported_filters_rejected(test_repo: ProductRepository):
    """Verify that submitting unallowed filter parameters raises ValueError."""
    with pytest.raises(ValueError):
        await test_repo.find(filters={"malicious_key": "some_value"})

    with pytest.raises(ValueError):
        await test_repo.find(filters={"$where": "this.price > 1000"})


@pytest.mark.asyncio
async def test_count_matches_find(test_repo: ProductRepository):
    """Verify that count() returns identical count to find()."""
    flt = {"category": "laptop", "in_stock": True}
    _, find_total = await test_repo.find(filters=flt)
    count_total = await test_repo.count(filters=flt)
    assert find_total == count_total == 2


@pytest.mark.asyncio
async def test_categories_summary(test_repo: ProductRepository):
    """Verify categories() aggregation computes exact counts and price bounds."""
    summaries = await test_repo.categories()
    assert len(summaries) == 5

    summary_map = {s.category: s for s in summaries}

    # Laptop: 3 items (54999, 74999, 112000)
    assert summary_map[Category.LAPTOP].product_count == 3
    assert summary_map[Category.LAPTOP].min_price_inr == 54999
    assert summary_map[Category.LAPTOP].max_price_inr == 112000

    # Phone: 1 item (119999)
    assert summary_map[Category.PHONE].product_count == 1
    assert summary_map[Category.PHONE].min_price_inr == 119999
    assert summary_map[Category.PHONE].max_price_inr == 119999

    # Monitor: 0 items in test sample
    assert summary_map[Category.MONITOR].product_count == 0
    assert summary_map[Category.MONITOR].min_price_inr is None
    assert summary_map[Category.MONITOR].max_price_inr is None


@pytest.mark.asyncio
async def test_empty_collection_categories_summary():
    """Verify categories() handles a completely empty collection cleanly."""
    cfg = get_settings()
    client = AsyncIOMotorClient(cfg.mongo_uri, serverSelectionTimeoutMS=2000)
    db = client["empty_test_db"]
    collection = db["empty_products"]
    await collection.delete_many({})

    repo = ProductRepository(collection)
    summaries = await repo.categories()

    assert len(summaries) == 5
    for s in summaries:
        assert s.product_count == 0
        assert s.min_price_inr is None
        assert s.max_price_inr is None

    await client.drop_database("empty_test_db")
    client.close()
