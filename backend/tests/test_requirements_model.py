"""Unit tests for Requirements and RequirementsDelta domain models.

Verifies Pydantic v2 schemas, strict type enforcement, non-negative budget validation,
preference weight boundaries, allowed removal fields, and serialization integrity.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.product import Category
from app.models.requirements import (
    HardConstraints,
    Message,
    RECOGNIZED_PREFERENCES,
    Requirements,
    RequirementsDelta,
    VALID_REMOVABLE_FIELDS,
)


# ---------------------------------------------------------------------------
# Message Model Tests
# ---------------------------------------------------------------------------

def test_message_valid():
    """Message constructs with valid role and non-empty content."""
    msg = Message(role="user", content="Looking for a laptop")
    assert msg.role == "user"
    assert msg.content == "Looking for a laptop"


def test_message_invalid_empty_content():
    """Message rejects empty or whitespace-only content."""
    with pytest.raises(ValidationError):
        Message(role="user", content="   ")


def test_message_invalid_role():
    """Message rejects unsupported roles."""
    with pytest.raises(ValidationError):
        Message(role="invalid_role", content="hello")  # type: ignore


# ---------------------------------------------------------------------------
# HardConstraints Model Tests
# ---------------------------------------------------------------------------

def test_hard_constraints_valid():
    """HardConstraints accepts valid positive values."""
    hard = HardConstraints(
        min_ram_gb=16,
        min_storage_gb=512,
        min_battery_hr=8.5,
        max_weight_kg=1.8,
        requires_gpu=True,
        min_vram_gb=4,
        exclude_brands=["Apple", "Acer"],
        in_stock=True,
    )
    assert hard.min_ram_gb == 16
    assert hard.min_storage_gb == 512
    assert hard.min_battery_hr == 8.5
    assert hard.max_weight_kg == 1.8
    assert hard.requires_gpu is True
    assert hard.min_vram_gb == 4
    assert hard.exclude_brands == ["Apple", "Acer"]
    assert hard.in_stock is True


def test_hard_constraints_negative_numeric_rejected():
    """HardConstraints rejects negative numbers for ram, storage, weight, battery."""
    with pytest.raises(ValidationError):
        HardConstraints(min_ram_gb=-8)
    with pytest.raises(ValidationError):
        HardConstraints(min_storage_gb=0)
    with pytest.raises(ValidationError):
        HardConstraints(max_weight_kg=-1.5)
    with pytest.raises(ValidationError):
        HardConstraints(min_vram_gb=-2)


def test_hard_constraints_forbids_extra_fields():
    """HardConstraints rejects unsupported arbitrary fields."""
    with pytest.raises(ValidationError):
        HardConstraints(arbitrary_mongo_query={"$gt": 10})  # type: ignore


# ---------------------------------------------------------------------------
# Requirements Model Tests
# ---------------------------------------------------------------------------

def test_requirements_default_construction():
    """Requirements initializes with sensible defaults."""
    req = Requirements()
    assert req.category is None
    assert req.use_cases == []
    assert req.budget_max_inr is None
    assert req.budget_min_inr is None
    assert isinstance(req.hard, HardConstraints)
    assert req.preferences == {}
    assert req.ml_mode is None
    assert req.asked_slots == []


def test_requirements_valid_full():
    """Requirements constructs with comprehensive valid parameters."""
    req = Requirements(
        category=Category.LAPTOP,
        use_cases=["programming", "machine_learning"],
        budget_max_inr=80000,
        budget_min_inr=50000,
        hard=HardConstraints(min_ram_gb=16, requires_gpu=True),
        preferences={"portability": 0.8, "performance": 0.95},
        ml_mode="local",
        asked_slots=["budget"],
    )
    assert req.category == Category.LAPTOP
    assert req.use_cases == ["programming", "machine_learning"]
    assert req.budget_max_inr == 80000
    assert req.budget_min_inr == 50000
    assert req.hard.min_ram_gb == 16
    assert req.preferences["portability"] == 0.8
    assert req.preferences["performance"] == 0.95
    assert req.ml_mode == "local"


def test_requirements_negative_budget_rejected():
    """Requirements rejects negative budget values."""
    with pytest.raises(ValidationError):
        Requirements(budget_max_inr=-1000)
    with pytest.raises(ValidationError):
        Requirements(budget_min_inr=-500)


def test_requirements_invalid_category_rejected():
    """Requirements rejects unsupported category strings."""
    with pytest.raises(ValidationError):
        Requirements(category="smartwatch")  # type: ignore


def test_requirements_invalid_ml_mode_rejected():
    """Requirements rejects unknown ml_mode values."""
    with pytest.raises(ValidationError):
        Requirements(ml_mode="quantum")


def test_requirements_preference_weight_boundaries():
    """Requirements validates that preference weights fall within [0.0, 1.0]."""
    # Valid
    req = Requirements(preferences={"portability": 0.5, "camera_quality": 1.0, "battery_life": 0.0})
    assert req.preferences["portability"] == 0.5

    # Out of bounds (> 1.0)
    with pytest.raises(ValidationError):
        Requirements(preferences={"portability": 1.2})

    # Out of bounds (< 0.0)
    with pytest.raises(ValidationError):
        Requirements(preferences={"portability": -0.1})

    # Non-numeric weight
    with pytest.raises(ValidationError):
        Requirements(preferences={"portability": "high"})  # type: ignore


# ---------------------------------------------------------------------------
# RequirementsDelta Model Tests
# ---------------------------------------------------------------------------

def test_requirements_delta_empty():
    """RequirementsDelta allows all fields to be None/omitted except defaults."""
    delta = RequirementsDelta()
    assert delta.category is None
    assert delta.use_cases is None
    assert delta.budget_max_inr is None
    assert delta.budget_min_inr is None
    assert delta.hard is None
    assert delta.preferences is None
    assert delta.ml_mode is None
    assert delta.removed_fields == []
    assert delta.is_off_topic is False


def test_requirements_delta_with_removed_fields():
    """RequirementsDelta validates known removed_fields entries."""
    delta = RequirementsDelta(
        removed_fields=["portability", "budget_max_inr", "min_ram_gb"]
    )
    assert delta.removed_fields == ["portability", "budget_max_inr", "min_ram_gb"]


def test_requirements_delta_rejects_unknown_removed_field():
    """RequirementsDelta rejects unknown fields in removed_fields."""
    with pytest.raises(ValidationError) as exc_info:
        RequirementsDelta(removed_fields=["non_existent_slot_xyz"])
    assert "Invalid removed_field" in str(exc_info.value)


def test_requirements_delta_off_topic_flag():
    """RequirementsDelta supports explicit off-topic classification."""
    delta = RequirementsDelta(is_off_topic=True)
    assert delta.is_off_topic is True


def test_requirements_delta_serialization_round_trip():
    """RequirementsDelta serializes to JSON and deserializes losslessly."""
    delta = RequirementsDelta(
        category=Category.PHONE,
        budget_max_inr=35000,
        preferences={"camera_quality": 0.9},
        removed_fields=["portability"],
        is_off_topic=False,
    )
    dumped = delta.model_dump()
    assert dumped["category"] == "phone"
    assert dumped["budget_max_inr"] == 35000
    assert dumped["preferences"]["camera_quality"] == 0.9

    reloaded = RequirementsDelta.model_validate(dumped)
    assert reloaded == delta
