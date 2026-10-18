"""Product catalogue repository for MongoDB persistence and query operations.

Enforces strict filter validation, deterministic sorting, and safe aggregation.
"""

from __future__ import annotations

import re
from typing import Any, Sequence
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from app.core.logging import get_logger
from app.db.mongo import get_mongo_db
from app.models.product import Category, CategorySummary, Product

logger = get_logger("app.db.repositories")

ALLOWED_FILTERS = {"category", "min_price", "max_price", "use_case", "brand", "in_stock"}

SORT_MAPPINGS: dict[str, list[tuple[str, int]]] = {
    "price_asc": [("price_inr", ASCENDING), ("product_id", ASCENDING)],
    "price_inr_asc": [("price_inr", ASCENDING), ("product_id", ASCENDING)],
    "price_desc": [("price_inr", DESCENDING), ("product_id", ASCENDING)],
    "price_inr_desc": [("price_inr", DESCENDING), ("product_id", ASCENDING)],
    "rating_desc": [("rating", DESCENDING), ("price_inr", ASCENDING), ("product_id", ASCENDING)],
    "rating_asc": [("rating", ASCENDING), ("price_inr", ASCENDING), ("product_id", ASCENDING)],
    "name_asc": [("name", ASCENDING), ("product_id", ASCENDING)],
    "name_desc": [("name", DESCENDING), ("product_id", ASCENDING)],
}


class ProductRepository:
    """Asynchronous repository for querying and managing product catalogue records."""

    def __init__(
        self,
        collection_or_db: AsyncIOMotorCollection | AsyncIOMotorDatabase | None = None,
    ) -> None:
        if isinstance(collection_or_db, AsyncIOMotorCollection):
            self._collection = collection_or_db
        elif isinstance(collection_or_db, AsyncIOMotorDatabase):
            self._collection = collection_or_db["products"]
        else:
            self._collection = None

    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Obtain active products collection, falling back to default database dependency."""
        if self._collection is not None:
            return self._collection
        return get_mongo_db()["products"]

    def _build_filter_query(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Validate input filter keys and translate them to safe MongoDB query clauses."""
        if not isinstance(filters, dict):
            raise ValueError("Filters must be provided as a dictionary.")

        unknown_filters = set(filters.keys()) - ALLOWED_FILTERS
        if unknown_filters:
            raise ValueError(
                f"Unsupported filter parameters: {sorted(unknown_filters)}. "
                f"Allowed filters are: {sorted(ALLOWED_FILTERS)}."
            )

        query: dict[str, Any] = {}

        # 1. Category Filter
        if "category" in filters and filters["category"] is not None:
            cat_val = filters["category"]
            if isinstance(cat_val, Category):
                query["category"] = cat_val.value
            elif isinstance(cat_val, str) and cat_val in [c.value for c in Category]:
                query["category"] = cat_val
            else:
                raise ValueError(
                    f"Invalid category filter: '{cat_val}'. Must be one of {[c.value for c in Category]}."
                )

        # 2. Price Boundaries
        min_price = filters.get("min_price")
        max_price = filters.get("max_price")

        price_clause: dict[str, int] = {}
        if min_price is not None:
            if isinstance(min_price, bool) or not isinstance(min_price, int) or min_price < 0:
                raise ValueError("min_price must be a non-negative integer.")
            price_clause["$gte"] = min_price

        if max_price is not None:
            if isinstance(max_price, bool) or not isinstance(max_price, int) or max_price < 0:
                raise ValueError("max_price must be a non-negative integer.")
            if min_price is not None and max_price < min_price:
                raise ValueError(f"max_price ({max_price}) cannot be less than min_price ({min_price}).")
            price_clause["$lte"] = max_price

        if price_clause:
            query["price_inr"] = price_clause

        # 3. Use Case Filter (searches inside intended_use array)
        if "use_case" in filters and filters["use_case"] is not None:
            use_case = filters["use_case"]
            if not isinstance(use_case, str) or not use_case.strip():
                raise ValueError("use_case must be a non-empty string.")
            query["intended_use"] = use_case.strip()

        # 4. Brand Filter (case-insensitive exact match)
        if "brand" in filters and filters["brand"] is not None:
            brand = filters["brand"]
            if not isinstance(brand, str) or not brand.strip():
                raise ValueError("brand must be a non-empty string.")
            escaped = re.escape(brand.strip())
            query["brand"] = {"$regex": f"^{escaped}$", "$options": "i"}

        # 5. In-Stock Filter
        if "in_stock" in filters and filters["in_stock"] is not None:
            in_stock = filters["in_stock"]
            if not isinstance(in_stock, bool):
                raise ValueError("in_stock must be a boolean.")
            query["in_stock"] = in_stock

        return query

    async def get(self, product_id: str) -> Product | None:
        """Retrieve a single validated product by its unique product_id."""
        if not product_id or not isinstance(product_id, str) or not product_id.strip():
            return None

        clean_id = product_id.strip()
        doc = await self.collection.find_one({"product_id": clean_id}, {"_id": 0})
        if not doc:
            return None
        return Product.model_validate(doc)

    async def find(
        self,
        filters: dict[str, Any] | None = None,
        sort: str = "price_inr_asc",
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Product], int]:
        """Find matching products using validated filters, deterministic sorting, and pagination."""
        if page < 1:
            raise ValueError(f"page must be an integer >= 1, got {page}.")
        if page_size < 1 or page_size > 50:
            raise ValueError(f"page_size must be between 1 and 50, got {page_size}.")

        if sort not in SORT_MAPPINGS:
            raise ValueError(
                f"Unsupported sort option '{sort}'. Allowed options: {sorted(SORT_MAPPINGS.keys())}."
            )

        query = self._build_filter_query(filters or {})
        sort_criteria = SORT_MAPPINGS[sort]

        total_count = await self.collection.count_documents(query)
        if total_count == 0:
            return [], 0

        skip = (page - 1) * page_size
        cursor = self.collection.find(query, {"_id": 0}).sort(sort_criteria).skip(skip).limit(page_size)
        raw_docs = await cursor.to_list(length=page_size)

        products = [Product.model_validate(doc) for doc in raw_docs]
        return products, total_count

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count the total number of products matching the provided filters."""
        query = self._build_filter_query(filters or {})
        return await self.collection.count_documents(query)

    async def categories(self) -> list[CategorySummary]:
        """Summarize catalogue product counts and price boundaries across all categories."""
        pipeline = [
            {
                "$group": {
                    "_id": "$category",
                    "product_count": {"$sum": 1},
                    "min_price_inr": {"$min": "$price_inr"},
                    "max_price_inr": {"$max": "$price_inr"},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        raw_summaries: dict[str, dict[str, Any]] = {}
        async for doc in self.collection.aggregate(pipeline):
            raw_summaries[doc["_id"]] = doc

        summaries: list[CategorySummary] = []
        for cat in Category:
            if cat.value in raw_summaries:
                data = raw_summaries[cat.value]
                summaries.append(
                    CategorySummary(
                        category=cat,
                        product_count=data.get("product_count", 0),
                        min_price_inr=data.get("min_price_inr"),
                        max_price_inr=data.get("max_price_inr"),
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

    async def create_indexes(self) -> list[str]:
        """Create required indexes on the products collection."""
        indexes = [
            IndexModel([("product_id", ASCENDING)], unique=True, name="idx_product_id_unique"),
            IndexModel(
                [("category", ASCENDING), ("price_inr", ASCENDING)],
                name="idx_category_price_inr",
            ),
            IndexModel([("intended_use", ASCENDING)], name="idx_intended_use"),
        ]
        created = await self.collection.create_indexes(indexes)
        logger.info("Created MongoDB indexes on collection '%s': %s", self.collection.name, created)
        return created
