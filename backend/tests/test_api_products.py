"""Automated tests for Product Catalogue API endpoints (/api/v1/products).

Verifies product listing, multi-parameter filtering, pagination boundaries,
deterministic sorting, single product retrieval (200/404), category summaries,
and schema conformity.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.db.repositories import ProductRepository
from app.main import create_app
from app.models.product import Category, Product


@pytest.fixture
def test_products() -> list[Product]:
    """Fixture containing diverse sample products for API testing."""
    return [
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


class MockProductRepository:
    """Mock repository providing in-memory data for deterministic API testing."""

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
        page_size: int = 20,
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

        if "use_case" in flt:
            matches = [p for p in matches if flt["use_case"] in p.intended_use]

        if "brand" in flt:
            matches = [p for p in matches if p.brand.lower() == flt["brand"].lower()]

        if "in_stock" in flt:
            matches = [p for p in matches if p.in_stock == flt["in_stock"]]

        # Sorting
        if sort in ("price_asc", "price_inr_asc"):
            matches.sort(key=lambda p: (p.price_inr, p.product_id))
        elif sort in ("price_desc", "price_inr_desc"):
            matches.sort(key=lambda p: (-p.price_inr, p.product_id))
        elif sort == "rating_desc":
            matches.sort(key=lambda p: (-(p.rating or 0), p.price_inr, p.product_id))
        elif sort == "rating_asc":
            matches.sort(key=lambda p: (p.rating or 0, p.price_inr, p.product_id))
        elif sort == "name_asc":
            matches.sort(key=lambda p: (p.name, p.product_id))
        elif sort == "name_desc":
            matches.sort(key=lambda p: (p.name, p.product_id), reverse=True)

        total = len(matches)
        start = (page - 1) * page_size
        end = start + page_size
        return matches[start:end], total

    async def categories(self):
        from app.models.product import CategorySummary
        from collections import defaultdict
        grouped = defaultdict(list)
        for p in self._products:
            grouped[p.category].append(p)

        summaries = []
        for cat in Category:
            items = grouped[cat]
            if items:
                prices = [p.price_inr for p in items]
                summaries.append(
                    CategorySummary(
                        category=cat,
                        product_count=len(items),
                        min_price_inr=min(prices),
                        max_price_inr=max(prices),
                    )
                )
            else:
                summaries.append(
                    CategorySummary(
                        category=cat,
                        product_count=0,
                        min_price_inr=None,
                        max_price_inr=None,
                    )
                )
        return summaries


@pytest.fixture
def client(test_products: list[Product]):
    """FastAPI test client fixture with injected mock product repository."""
    from app.api.products import get_product_repository

    app = create_app(
        settings=Settings(mongo_uri="mongodb://localhost:27017", mongo_db="test_api"),
        init_db=False,
    )
    mock_repo = MockProductRepository(test_products)
    app.dependency_overrides[get_product_repository] = lambda: mock_repo
    return app


@pytest.mark.asyncio
async def test_get_products_list_default(client):
    """GET /api/v1/products returns default paginated list of items."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["page_size"] == 20
        assert data["total_pages"] == 1
        assert len(data["items"]) == 5
        # Ensure MongoDB _id is not present
        assert all("_id" not in item for item in data["items"])
        assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_get_products_filter_by_category(client):
    """GET /api/v1/products?category=laptop returns only laptops."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?category=laptop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert all(p["category"] == "laptop" for p in data["items"])


@pytest.mark.asyncio
async def test_get_products_filter_by_price_range(client):
    """GET /api/v1/products with min_price and max_price boundaries."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        # min_price=54999, max_price=74999 -> LAP-0001 (54999) & LAP-0003 (74999)
        resp = await ac.get("/api/v1/products?min_price=54999&max_price=74999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        p_ids = {p["product_id"] for p in data["items"]}
        assert p_ids == {"LAP-0001", "LAP-0003"}


@pytest.mark.asyncio
async def test_get_products_min_price_exceeds_max_price(client):
    """GET /api/v1/products with min_price > max_price returns HTTP 422."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?min_price=100000&max_price=50000")
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data


@pytest.mark.asyncio
async def test_get_products_filter_use_case(client):
    """GET /api/v1/products?use_case=gaming returns gaming items."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?use_case=gaming")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert all("gaming" in p["intended_use"] for p in data["items"])


@pytest.mark.asyncio
async def test_get_products_filter_brand(client):
    """GET /api/v1/products?brand=Zenit returns products for that brand."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?brand=zenit")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["brand"] == "Zenit"


@pytest.mark.asyncio
async def test_get_products_filter_in_stock(client):
    """GET /api/v1/products?in_stock=true and in_stock=false filter stock availability."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        # in_stock=true
        resp_true = await ac.get("/api/v1/products?in_stock=true")
        assert resp_true.status_code == 200
        data_true = resp_true.json()
        assert data_true["total"] == 4
        assert all(p["in_stock"] is True for p in data_true["items"])

        # in_stock=false
        resp_false = await ac.get("/api/v1/products?in_stock=false")
        assert resp_false.status_code == 200
        data_false = resp_false.json()
        assert data_false["total"] == 1
        assert data_false["items"][0]["product_id"] == "LAP-0003"
        assert data_false["items"][0]["in_stock"] is False


@pytest.mark.asyncio
async def test_get_products_pagination(client):
    """GET /api/v1/products pagination: total_pages, page, page_size."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["total_pages"] == 3
        assert len(data["items"]) == 2

        # Page 2
        resp2 = await ac.get("/api/v1/products?page=2&page_size=2")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["items"]) == 2
        assert data2["items"][0]["product_id"] != data["items"][0]["product_id"]


@pytest.mark.asyncio
async def test_get_products_sorting(client):
    """GET /api/v1/products?sort=price_desc sorts descending."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?sort=price_desc")
        assert resp.status_code == 200
        data = resp.json()
        prices = [p["price_inr"] for p in data["items"]]
        assert prices == sorted(prices, reverse=True)


@pytest.mark.asyncio
async def test_get_products_empty_results(client):
    """GET /api/v1/products with non-matching filter returns empty list and 0 pages."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products?category=monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["total_pages"] == 0


@pytest.mark.asyncio
async def test_get_products_invalid_validation_errors(client):
    """Verify HTTP 422 on invalid category, negative price, invalid sort, or page boundaries."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        # Invalid category
        r1 = await ac.get("/api/v1/products?category=invalid_cat")
        assert r1.status_code == 422

        # Negative price
        r2 = await ac.get("/api/v1/products?min_price=-10")
        assert r2.status_code == 422

        # Page < 1
        r3 = await ac.get("/api/v1/products?page=0")
        assert r3.status_code == 422

        # Page size > 50
        r4 = await ac.get("/api/v1/products?page_size=51")
        assert r4.status_code == 422

        # Unsupported sort
        r5 = await ac.get("/api/v1/products?sort=invalid_sort")
        assert r5.status_code == 422


@pytest.mark.asyncio
async def test_get_product_by_id_success(client):
    """GET /api/v1/products/{product_id} returns 200 and validated Product."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products/LAP-0001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["product_id"] == "LAP-0001"
        assert data["name"] == "Zenit Air 14"
        assert data["price_inr"] == 54999
        assert "_id" not in data


@pytest.mark.asyncio
async def test_get_product_by_id_not_found(client):
    """GET /api/v1/products/{product_id} returns 404 with error envelope for nonexistent ID."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products/NONEXISTENT-999")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "PRODUCT_NOT_FOUND"
        assert "request_id" in data["error"]


@pytest.mark.asyncio
async def test_get_categories_summary(client):
    """GET /api/v1/products/categories returns summaries for all 5 categories."""
    async with AsyncClient(transport=ASGITransport(app=client), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/products/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 5

        cat_map = {item["category"]: item for item in data}
        assert cat_map["laptop"]["product_count"] == 3
        assert cat_map["laptop"]["min_price_inr"] == 54999
        assert cat_map["laptop"]["max_price_inr"] == 112000
        assert cat_map["monitor"]["product_count"] == 0
        assert cat_map["monitor"]["min_price_inr"] is None
