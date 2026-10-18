"""Automated tests for the product catalogue JSON and seed script idempotency."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import get_settings
from app.models.product import Category, Product
from scripts.seed import DEFAULT_JSON_PATH, load_and_validate_products, seed_catalogue

SEED_TEST_DB = "electronics_agent_seed_test"


def test_catalogue_file_exists_and_total_count():
    """Verify products.json exists and contains exactly 120 records."""
    assert DEFAULT_JSON_PATH.exists(), f"Catalogue missing at {DEFAULT_JSON_PATH}"

    with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    assert len(products) == 120, f"Expected 120 products, got {len(products)}"


def test_catalogue_category_distribution():
    """Verify exact category counts: 40 laptops, 30 phones, 20 headphones, 15 tablets, 15 monitors."""
    with open(DEFAULT_JSON_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    counts: dict[str, int] = {}
    for p in products:
        counts[p["category"]] = counts.get(p["category"], 0) + 1

    assert counts[Category.LAPTOP.value] == 40
    assert counts[Category.PHONE.value] == 30
    assert counts[Category.HEADPHONES.value] == 20
    assert counts[Category.TABLET.value] == 15
    assert counts[Category.MONITOR.value] == 15


def test_catalogue_unique_ids_and_schema_validation():
    """Verify all 120 products have unique IDs and pass Product model validation."""
    products = load_and_validate_products(DEFAULT_JSON_PATH)
    assert len(products) == 120
    unique_ids = {p.product_id for p in products}
    assert len(unique_ids) == 120


def test_catalogue_edge_cases_present():
    """Verify all required edge cases exist in the catalogue dataset."""
    products = load_and_validate_products(DEFAULT_JSON_PATH)

    # 1. Product priced around 80,500 (just above common 80k budget)
    just_above_budget = [p for p in products if p.price_inr == 80500]
    assert len(just_above_budget) >= 1

    # 2. Heavy powerful laptop vs lightweight laptop
    laptops = [p for p in products if p.category == Category.LAPTOP]
    heavy_laptops = [l for l in laptops if l.specs.get("weight_kg", 0) >= 2.8]
    light_laptops = [l for l in laptops if l.specs.get("weight_kg", 99) <= 1.2]
    assert len(heavy_laptops) >= 1
    assert len(light_laptops) >= 1

    # 3. Out-of-stock products
    out_of_stock = [p for p in products if not p.in_stock]
    assert len(out_of_stock) >= 4  # Spread across categories

    # 4. Products with missing specification keys
    missing_specs = [p for p in products if "vram_gb" not in p.specs or "battery_hr" not in p.specs]
    assert len(missing_specs) >= 1

    # 5. Products with unknown/None rating
    unrated_products = [p for p in products if p.rating is None]
    assert len(unrated_products) >= 1

    # 6. Headphones with and without ANC
    headphones = [p for p in products if p.category == Category.HEADPHONES]
    anc_headphones = [h for h in headphones if h.specs.get("anc") is True]
    non_anc_headphones = [h for h in headphones if h.specs.get("anc") is False]
    assert len(anc_headphones) >= 1
    assert len(non_anc_headphones) >= 1

    # 7. Tablets with and without stylus support
    tablets = [p for p in products if p.category == Category.TABLET]
    stylus_tablets = [t for t in tablets if t.specs.get("stylus_support") is True]
    non_stylus_tablets = [t for t in tablets if t.specs.get("stylus_support") is False]
    assert len(stylus_tablets) >= 1
    assert len(non_stylus_tablets) >= 1

    # 8. Monitors with different panels (IPS, VA, OLED) and refresh rates (60, 144, 165, 240)
    monitors = [p for p in products if p.category == Category.MONITOR]
    panels = {m.specs.get("panel") for m in monitors}
    refresh_rates = {m.specs.get("refresh_hz") for m in monitors}
    assert {"IPS", "VA", "OLED"}.issubset(panels)
    assert {60, 144, 240}.issubset(refresh_rates)


def test_seed_invalid_json_fails(tmp_path: Path):
    """Verify loading invalid JSON or duplicate IDs causes clear ValueError."""
    # 1. Invalid JSON syntax
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unclosed json }", encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_validate_products(bad_json)

    # 2. Duplicate product_id
    duplicate_json = tmp_path / "dup.json"
    dup_data = [
        {
            "product_id": "LAP-0001",
            "name": "Laptop A",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 50000,
            "specs": {"cpu": "Intel i5"},
            "intended_use": ["student"],
        },
        {
            "product_id": "LAP-0001",  # duplicate ID
            "name": "Laptop B",
            "brand": "Voltix",
            "category": "laptop",
            "price_inr": 60000,
            "specs": {"cpu": "Intel i7"},
            "intended_use": ["programming"],
        },
    ]
    duplicate_json.write_text(json.dumps(dup_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate product_id detected"):
        load_and_validate_products(duplicate_json)


@pytest.mark.asyncio
async def test_seed_idempotency_and_update(tmp_path: Path):
    """Verify that repeated seed runs do not create duplicates, and modifications update records."""
    cfg = get_settings()
    client = AsyncIOMotorClient(cfg.mongo_uri, serverSelectionTimeoutMS=2000)
    db = client[SEED_TEST_DB]
    collection = db["products"]
    await db.drop_collection("products")

    # Sample small catalogue
    seed_file = tmp_path / "test_products.json"
    sample_data = [
        {
            "product_id": "LAP-0099",
            "name": "Original Name",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 50000,
            "specs": {"cpu": "Intel i5", "ram_gb": 8},
            "intended_use": ["student"],
            "features": ["Feature 1"],
            "in_stock": True,
            "rating": 4.0,
        }
    ]
    seed_file.write_text(json.dumps(sample_data), encoding="utf-8")

    try:
        # First Run: Insert record
        s1 = await seed_catalogue(db_name=SEED_TEST_DB, json_path=seed_file)
        assert s1["loaded"] == 1
        assert s1["inserted"] == 1
        assert s1["modified"] == 0
        assert s1["total_in_db"] == 1

        # Check existing internal _id
        doc1 = await collection.find_one({"product_id": "LAP-0099"})
        orig_object_id = doc1["_id"]

        # Second Run: Unchanged data (idempotent, 0 inserts, 0 modifications)
        s2 = await seed_catalogue(db_name=SEED_TEST_DB, json_path=seed_file)
        assert s2["inserted"] == 0
        assert s2["modified"] == 0
        assert s2["unchanged"] == 1
        assert s2["total_in_db"] == 1

        # Third Run: Modify record name & price
        sample_data[0]["name"] = "Updated Name"
        sample_data[0]["price_inr"] = 52000
        seed_file.write_text(json.dumps(sample_data), encoding="utf-8")

        s3 = await seed_catalogue(db_name=SEED_TEST_DB, json_path=seed_file)
        assert s3["inserted"] == 0
        assert s3["modified"] == 1
        assert s3["total_in_db"] == 1

        # Verify internal _id was preserved
        doc2 = await collection.find_one({"product_id": "LAP-0099"})
        assert doc2["_id"] == orig_object_id
        assert doc2["name"] == "Updated Name"
        assert doc2["price_inr"] == 52000

        # Verify indexes exist
        index_info = await collection.index_information()
        assert "idx_product_id_unique" in index_info
        assert "idx_category_price_inr" in index_info
        assert "idx_intended_use" in index_info

    finally:
        await client.drop_database(SEED_TEST_DB)
        client.close()
