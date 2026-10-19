"""API request and response models for public endpoints.

Defines response envelopes, pagination metadata, and dependency health statuses.
"""

from __future__ import annotations

import math
from typing import Literal
from pydantic import BaseModel, Field

from app.models.product import CategorySummary, Product

DependencyStatus = Literal["up", "down"]
OverallHealthStatus = Literal["ok", "degraded"]


class ProductListResponse(BaseModel):
    """Paginated product list response envelope."""

    items: list[Product] = Field(
        default_factory=list,
        description="List of matching validated products for the requested page",
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of matching products in the catalogue before pagination",
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page index (1-based)",
    )
    page_size: int = Field(
        ...,
        ge=1,
        le=50,
        description="Number of items returned per page",
    )
    total_pages: int = Field(
        ...,
        ge=0,
        description="Total calculated pages available based on total and page_size",
    )

    @classmethod
    def create(
        cls,
        items: list[Product],
        total: int,
        page: int,
        page_size: int,
    ) -> ProductListResponse:
        """Construct response and calculate total_pages safely."""
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )


class HealthResponse(BaseModel):
    """Diagnostic health response communicating MongoDB and Groq availability."""

    status: OverallHealthStatus = Field(
        ...,
        description="Overall service state ('ok' if both dependencies are up, else 'degraded')",
    )
    db: DependencyStatus = Field(
        ...,
        description="Local MongoDB database status ('up' or 'down')",
    )
    llm: DependencyStatus = Field(
        ...,
        description="Groq Cloud LLM provider status ('up' or 'down')",
    )


class ErrorDetail(BaseModel):
    """Detailed error object within the standard error envelope."""

    code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable explanation of the error")
    request_id: str | None = Field(default=None, description="Request correlation identifier")


class ErrorResponse(BaseModel):
    """Standardized error envelope returned by the API."""

    error: ErrorDetail


__all__ = [
    "CategorySummary",
    "DependencyStatus",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "OverallHealthStatus",
    "ProductListResponse",
]
