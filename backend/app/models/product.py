"""Product data models and category specification validation.

Defines the core Product model, supported categories, category-specific
specifications, and category aggregation summaries.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Category(str, Enum):
    """Supported product categories in the catalogue."""

    LAPTOP = "laptop"
    PHONE = "phone"
    HEADPHONES = "headphones"
    TABLET = "tablet"
    MONITOR = "monitor"


# Allowed specification keys per category
CATEGORY_SPEC_SCHEMA: dict[Category, set[str]] = {
    Category.LAPTOP: {
        "cpu",
        "ram_gb",
        "storage_gb",
        "gpu",
        "vram_gb",
        "weight_kg",
        "battery_hr",
        "screen_in",
        "os",
    },
    Category.PHONE: {
        "chipset",
        "ram_gb",
        "storage_gb",
        "camera_mp",
        "battery_mah",
        "screen_in",
        "weight_g",
        "os",
    },
    Category.HEADPHONES: {
        "type",
        "anc",
        "battery_hr",
        "wireless",
        "weight_g",
        "codec",
    },
    Category.TABLET: {
        "chipset",
        "ram_gb",
        "storage_gb",
        "screen_in",
        "stylus_support",
        "battery_hr",
        "weight_g",
    },
    Category.MONITOR: {
        "size_in",
        "resolution",
        "refresh_hz",
        "panel",
        "ports",
    },
}


def _validate_non_empty_str(field_name: str, val: Any) -> str:
    """Validate that val is a non-empty string."""
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"'{field_name}' must be a non-empty string.")
    return val.strip()


def _validate_positive_num(
    field_name: str,
    val: Any,
    allow_zero: bool = False,
    must_be_int: bool = False,
) -> int | float:
    """Validate that val is a positive number and not NaN/infinity or boolean."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"'{field_name}' must be a numeric value, not boolean or string.")
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"'{field_name}' cannot be NaN or infinity.")
    if must_be_int and not isinstance(val, int):
        raise ValueError(f"'{field_name}' must be an integer.")
    if allow_zero:
        if val < 0:
            raise ValueError(f"'{field_name}' must be non-negative (>= 0).")
    else:
        if val <= 0:
            raise ValueError(f"'{field_name}' must be strictly positive (> 0).")
    return val


def _validate_bool(field_name: str, val: Any) -> bool:
    """Validate that val is strictly a boolean."""
    if not isinstance(val, bool):
        raise ValueError(f"'{field_name}' must be a boolean.")
    return val


def validate_category_specs(category: Category, specs: dict[str, Any]) -> dict[str, Any]:
    """Validate specifications against the category's expected schema."""
    if not isinstance(specs, dict):
        raise ValueError("Product 'specs' must be a dictionary.")

    allowed_keys = CATEGORY_SPEC_SCHEMA[category]
    unknown_keys = set(specs.keys()) - allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Invalid specification keys for category '{category.value}': {sorted(unknown_keys)}. "
            f"Allowed keys are: {sorted(allowed_keys)}."
        )

    validated: dict[str, Any] = {}

    for k, v in specs.items():
        if v is None:
            # Missing specs are omitted, not set to None
            continue

        if category == Category.LAPTOP:
            if k in ("cpu", "gpu", "os"):
                validated[k] = _validate_non_empty_str(k, v)
            elif k in ("ram_gb", "storage_gb"):
                validated[k] = _validate_positive_num(k, v, must_be_int=True)
            elif k == "vram_gb":
                validated[k] = _validate_positive_num(k, v, allow_zero=True, must_be_int=True)
            elif k in ("weight_kg", "battery_hr", "screen_in"):
                validated[k] = _validate_positive_num(k, v)

        elif category == Category.PHONE:
            if k in ("chipset", "os"):
                validated[k] = _validate_non_empty_str(k, v)
            elif k in ("ram_gb", "storage_gb", "battery_mah"):
                validated[k] = _validate_positive_num(k, v, must_be_int=True)
            elif k in ("camera_mp", "screen_in", "weight_g"):
                validated[k] = _validate_positive_num(k, v)

        elif category == Category.HEADPHONES:
            if k == "type":
                s_val = _validate_non_empty_str(k, v).lower()
                if s_val not in ("over-ear", "in-ear"):
                    raise ValueError(f"Headphone type must be 'over-ear' or 'in-ear', got '{v}'.")
                validated[k] = s_val
            elif k in ("anc", "wireless"):
                validated[k] = _validate_bool(k, v)
            elif k in ("battery_hr", "weight_g"):
                validated[k] = _validate_positive_num(k, v)
            elif k == "codec":
                validated[k] = _validate_non_empty_str(k, v)

        elif category == Category.TABLET:
            if k == "chipset":
                validated[k] = _validate_non_empty_str(k, v)
            elif k in ("ram_gb", "storage_gb"):
                validated[k] = _validate_positive_num(k, v, must_be_int=True)
            elif k == "stylus_support":
                validated[k] = _validate_bool(k, v)
            elif k in ("screen_in", "battery_hr", "weight_g"):
                validated[k] = _validate_positive_num(k, v)

        elif category == Category.MONITOR:
            if k in ("resolution", "panel"):
                validated[k] = _validate_non_empty_str(k, v)
            elif k == "refresh_hz":
                validated[k] = _validate_positive_num(k, v, must_be_int=True)
            elif k == "size_in":
                validated[k] = _validate_positive_num(k, v)
            elif k == "ports":
                if not isinstance(v, list) or not v:
                    raise ValueError("Monitor 'ports' must be a non-empty list of port descriptions.")
                validated[k] = [_validate_non_empty_str(f"ports[{i}]", item) for i, item in enumerate(v)]

    # Consistency checks: if laptop has dedicated GPU with discrete model, vram_gb should not be 0
    if category == Category.LAPTOP and "gpu" in validated and "vram_gb" in validated:
        gpu_name = validated["gpu"].lower()
        vram = validated["vram_gb"]
        is_discrete = any(d in gpu_name for d in ("rtx", "gtx", "radeon rx", "arc a"))
        if is_discrete and vram == 0:
            raise ValueError(
                f"Contradictory laptop specs: discrete GPU '{validated['gpu']}' cannot have vram_gb=0."
            )

    return validated


class Product(BaseModel):
    """Core domain model representing an electronics catalogue item."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    product_id: str = Field(description="Unique product identifier (e.g., LAP-0001)")
    name: str = Field(description="Display name of the product")
    brand: str = Field(description="Manufacturer / brand name")
    category: Category = Field(description="Product category")
    price_inr: int = Field(description="Price in Indian Rupees as a positive integer")
    specs: dict[str, Any] = Field(default_factory=dict, description="Category-specific specifications")
    intended_use: list[str] = Field(description="List of intended user personas / use cases")
    features: list[str] = Field(default_factory=list, description="Key product selling points / features")
    in_stock: bool = Field(default=True, description="Inventory availability flag")
    rating: float | None = Field(default=None, description="Average review score between 0.0 and 5.0")

    @field_validator("product_id", "name", "brand")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        return _validate_non_empty_str(info.field_name, v)

    @field_validator("price_inr", mode="before")
    @classmethod
    def validate_price_inr(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("price_inr must be an integer, not boolean or float.")
        if v <= 0:
            raise ValueError(f"price_inr must be a positive integer (> 0), got {v}.")
        return v

    @field_validator("intended_use", "features")
    @classmethod
    def validate_string_lists(cls, v: Any, info: Any) -> list[str]:
        if not isinstance(v, list) or (info.field_name == "intended_use" and not v):
            raise ValueError(f"{info.field_name} must be a non-empty list of strings.")
        cleaned = [_validate_non_empty_str(f"{info.field_name}[{i}]", item) for i, item in enumerate(v)]
        return cleaned

    @field_validator("rating", mode="before")
    @classmethod
    def validate_rating(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("rating must be a numeric value or None.")
        if math.isnan(v) or math.isinf(v):
            raise ValueError("rating cannot be NaN or infinity.")
        f_val = round(float(v), 2)
        if not (0.0 <= f_val <= 5.0):
            raise ValueError(f"rating must be between 0.0 and 5.0, got {f_val}.")
        return f_val

    @model_validator(mode="after")
    def validate_specs_against_category(self) -> Product:
        """Validate specs according to the assigned product category."""
        object.__setattr__(self, "specs", validate_category_specs(self.category, self.specs))
        return self



class CategorySummary(BaseModel):
    """Aggregated summary of product counts and price boundaries for a category."""

    category: Category = Field(description="Category name")
    product_count: int = Field(description="Total products available in category", ge=0)
    min_price_inr: int | None = Field(default=None, description="Lowest price in INR for this category")
    max_price_inr: int | None = Field(default=None, description="Highest price in INR for this category")
