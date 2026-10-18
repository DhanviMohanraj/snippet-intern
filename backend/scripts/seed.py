"""Idempotent MongoDB product catalogue seeding script.

Loads products from backend/data/products.json, validates each record against
the Product schema, creates required MongoDB indexes, and performs bulk upserts.
Preserves existing MongoDB _id identifiers on repeated runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.repositories import ProductRepository
from app.models.product import Product

logger = get_logger("scripts.seed")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = SCRIPT_DIR.parent / "data" / "products.json"


def load_and_validate_products(json_file_path: Path) -> list[Product]:
    """Load JSON product catalogue and validate every record against Product model.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If JSON is invalid, malformed, or has duplicate product IDs.
    """
    if not json_file_path.exists():
        raise FileNotFoundError(f"Product catalogue file not found at: {json_file_path}")

    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        raise ValueError(f"Malformed JSON in catalogue file '{json_file_path}': {err}") from err

    if not isinstance(raw_data, list):
        raise ValueError(f"Catalogue JSON must contain a top-level list of products, got {type(raw_data).__name__}.")

    seen_ids: set[str] = set()
    validated_products: list[Product] = []

    for idx, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(f"Product at index {idx} must be a dictionary object.")

        p_id = item.get("product_id")
        if not p_id or not isinstance(p_id, str) or not p_id.strip():
            raise ValueError(f"Product at index {idx} has missing or empty 'product_id'.")

        clean_id = p_id.strip()
        if clean_id in seen_ids:
            raise ValueError(f"Duplicate product_id detected in catalogue: '{clean_id}' at index {idx}.")
        seen_ids.add(clean_id)

        try:
            product = Product.model_validate(item)
            validated_products.append(product)
        except Exception as err:
            raise ValueError(f"Validation failed for product '{clean_id}' at index {idx}: {err}") from err

    return validated_products


async def seed_catalogue(
    db_name: str | None = None,
    json_path: Path | None = None,
    mongo_uri: str | None = None,
) -> dict[str, int]:
    """Execute idempotent catalogue seeding against the configured MongoDB database.

    Args:
        db_name: Target database name (defaults to Settings.mongo_db).
        json_path: Path to products JSON file (defaults to backend/data/products.json).
        mongo_uri: MongoDB connection URI (defaults to Settings.mongo_uri).

    Returns:
        Dictionary summarizing loaded, inserted, updated, and unchanged record counts.
    """
    setup_logging(log_level="INFO")
    cfg = get_settings()

    target_uri = mongo_uri or cfg.mongo_uri
    target_db_name = db_name or cfg.mongo_db
    file_path = json_path or DEFAULT_JSON_PATH

    logger.info("Loading and validating products from: %s", file_path)
    products = load_and_validate_products(file_path)
    total_loaded = len(products)
    logger.info("Successfully validated %d product records against Product schema.", total_loaded)

    # Dedicated script-owned MongoDB client with short timeout
    client = AsyncIOMotorClient(target_uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)

    try:
        # Verify connectivity
        await client.admin.command("ping")
        db = client[target_db_name]
        collection = db["products"]
        logger.info("Connected to MongoDB database '%s'.", target_db_name)

        # 1. Create required indexes
        repo = ProductRepository(collection)
        await repo.create_indexes()

        # 2. Build bulk upsert operations preserving _id
        operations: list[UpdateOne] = [
            UpdateOne(
                {"product_id": p.product_id},
                {"$set": p.model_dump()},
                upsert=True,
            )
            for p in products
        ]

        # 3. Execute bulk write
        result = await collection.bulk_write(operations, ordered=False)

        inserted_count = result.upserted_count
        modified_count = result.modified_count
        matched_count = result.matched_count
        unchanged_count = matched_count - modified_count
        total_in_db = await collection.count_documents({})

        summary = {
            "loaded": total_loaded,
            "inserted": inserted_count,
            "modified": modified_count,
            "unchanged": unchanged_count,
            "total_in_db": total_in_db,
        }

        logger.info(
            "Catalogue Seed Completed: loaded=%d, inserted=%d, modified=%d, unchanged=%d, total_in_db=%d",
            total_loaded,
            inserted_count,
            modified_count,
            unchanged_count,
            total_in_db,
        )
        return summary

    except PyMongoError as err:
        logger.error("MongoDB error during seeding: %s", err)
        raise
    finally:
        client.close()
        logger.info("Closed seed MongoDB connection client.")


def main() -> None:
    """CLI entrypoint for executing seed script."""
    try:
        summary = asyncio.run(seed_catalogue())
        print("\n" + "=" * 50)
        print("ELECTRONICS RECOMMENDATION AGENT — CATALOGUE SEED")
        print("=" * 50)
        print(f"Products Loaded from JSON: {summary['loaded']}")
        print(f"New Records Inserted:     {summary['inserted']}")
        print(f"Existing Records Updated:  {summary['modified']}")
        print(f"Records Unchanged:         {summary['unchanged']}")
        print(f"Total Products in Database:{summary['total_in_db']}")
        print("=" * 50)
        sys.exit(0)
    except Exception as exc:
        print(f"\nFATAL: Seed execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
