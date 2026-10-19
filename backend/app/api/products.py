"""Product catalogue API endpoints.

Provides filterable, paginated product search, category aggregation summaries,
and single product retrieval by product_id.
"""

from __future__ import annotations

from typing import Any, Literal
from fastapi import APIRouter, Depends, Path, Query
from fastapi.exceptions import RequestValidationError

from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.repositories import ProductRepository
from app.models.api import CategorySummary, ProductListResponse
from app.models.product import Category, Product

logger = get_logger("app.api.products")

router = APIRouter(prefix="/products", tags=["Products"])

AllowedSort = Literal[
    "price_asc",
    "price_inr_asc",
    "price_desc",
    "price_inr_desc",
    "rating_desc",
    "rating_asc",
    "name_asc",
    "name_desc",
]


def get_product_repository() -> ProductRepository:
    """Dependency provider returning an active ProductRepository instance."""
    return ProductRepository()


@router.get(
    "",
    summary="List and filter catalogue products",
    description=(
        "Retrieve a paginated list of electronics products with optional filtering "
        "by category, price range, use case persona, brand, and stock status."
    ),
    response_model=ProductListResponse,
    responses={
        200: {"description": "Matching products successfully retrieved."},
        422: {"description": "Validation error in query parameters."},
    },
)
async def list_products(
    category: Category | None = Query(
        default=None,
        description="Filter products by category (laptop, phone, headphones, tablet, monitor)",
    ),
    min_price: int | None = Query(
        default=None,
        ge=0,
        description="Minimum product price in Indian Rupees (inclusive)",
    ),
    max_price: int | None = Query(
        default=None,
        ge=0,
        description="Maximum product price in Indian Rupees (inclusive)",
    ),
    use_case: str | None = Query(
        default=None,
        description="Filter by intended use case (e.g. programming, gaming, photography)",
    ),
    brand: str | None = Query(
        default=None,
        description="Filter by manufacturer brand (case-insensitive)",
    ),
    in_stock: bool | None = Query(
        default=None,
        description="Filter by stock availability (true for in-stock, false for out-of-stock)",
    ),
    sort: AllowedSort = Query(
        default="price_inr_asc",
        description="Sort attribute and direction",
    ),
    page: int = Query(
        default=1,
        ge=1,
        description="Page number (1-indexed)",
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=50,
        description="Number of records per page (1 to 50)",
    ),
    repo: ProductRepository = Depends(get_product_repository),
) -> ProductListResponse:
    """Retrieve filtered and paginated product items."""
    # Cross-field validation: min_price cannot exceed max_price
    if min_price is not None and max_price is not None and min_price > max_price:
        raise RequestValidationError(
            [
                {
                    "loc": ["query", "min_price"],
                    "msg": f"min_price ({min_price}) cannot be greater than max_price ({max_price}).",
                    "type": "value_error",
                }
            ]
        )

    filters: dict[str, Any] = {}
    if category is not None:
        filters["category"] = category
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    if use_case is not None:
        filters["use_case"] = use_case
    if brand is not None:
        filters["brand"] = brand
    if in_stock is not None:
        filters["in_stock"] = in_stock

    products, total = await repo.find(
        filters=filters,
        sort=sort,
        page=page,
        page_size=page_size,
    )

    return ProductListResponse.create(
        items=products,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/categories",
    summary="Get catalogue category summaries",
    description="Retrieve product counts and price boundaries aggregated across all five supported categories.",
    response_model=list[CategorySummary],
    responses={
        200: {"description": "Category summaries successfully retrieved."},
    },
)
async def get_categories_summary(
    repo: ProductRepository = Depends(get_product_repository),
) -> list[CategorySummary]:
    """Retrieve catalogue category statistics."""
    return await repo.categories()


@router.get(
    "/{product_id}",
    summary="Get product details by ID",
    description="Retrieve a single validated product by its unique product identifier.",
    response_model=Product,
    responses={
        200: {"description": "Product found and returned."},
        404: {"description": "Product not found."},
    },
)
async def get_product_by_id(
    product_id: str = Path(
        ...,
        description="Unique product identifier (e.g., LAP-0001, PHN-0001)",
        examples=["LAP-0001"],
    ),
    repo: ProductRepository = Depends(get_product_repository),
) -> Product:
    """Retrieve an individual product record."""
    product = await repo.get(product_id)
    if product is None:
        raise AppError(
            code="PRODUCT_NOT_FOUND",
            status=404,
            message=f"Product with ID '{product_id}' was not found in the catalogue.",
        )
    return product
