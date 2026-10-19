"""Unit tests for deterministic RequirementsDelta normalization.

Covers:
- Indian currency INR parsing (80k, 1.2 lakh, 70 hazaar, ₹80,000).
- Category and use-case aliases.
- Preference weight boundaries and qualitative normalizations.
- Explicit removal handling and preference suppression.
- Normalization idempotency.
- Verification across all 15 handwritten extraction examples.
"""

from __future__ import annotations

import pytest

from app.agent.examples import EXTRACTION_EXAMPLES
from app.agent.normalize import (
    normalize,
    normalize_category,
    normalize_preferences,
    normalize_removed_fields,
    normalize_use_cases,
    parse_inr_budget_expression,
)
from app.models.product import Category
from app.models.requirements import HardConstraints, RequirementsDelta


# ---------------------------------------------------------------------------
# Budget Expression Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_input,expected_amount",
    [
        ("80k", 80000),
        ("80 K", 80000),
        ("80K", 80000),
        ("₹80,000", 80000),
        ("Rs 80,000", 80000),
        ("under 80k", 80000),
        ("below 80k", 80000),
        ("1.2 lakh", 120000),
        ("1.2L", 120000),
        ("1.2 lac", 120000),
        ("1 lakh", 100000),
        ("70 hazaar", 70000),
        ("70 hazar", 70000),
        ("50,000", 50000),
        ("50000", 50000),
        (80000, 80000),
        (80000.0, 80000),
        (None, None),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_inr_budget_expression_valid(raw_input, expected_amount):
    """parse_inr_budget_expression normalizes diverse Indian currency patterns."""
    assert parse_inr_budget_expression(raw_input) == expected_amount


def test_parse_inr_budget_expression_negative_raises():
    """Negative numeric inputs raise ValueError."""
    with pytest.raises(ValueError):
        parse_inr_budget_expression(-5000)


# ---------------------------------------------------------------------------
# Category Alias Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_cat,expected_enum",
    [
        ("laptop", Category.LAPTOP),
        ("notebook", Category.LAPTOP),
        ("lappy", Category.LAPTOP),
        ("macbook", Category.LAPTOP),
        ("phone", Category.PHONE),
        ("smartphone", Category.PHONE),
        ("mobile", Category.PHONE),
        ("cellphone", Category.PHONE),
        ("headphones", Category.HEADPHONES),
        ("headphone", Category.HEADPHONES),
        ("earphones", Category.HEADPHONES),
        ("tws", Category.HEADPHONES),
        ("tablet", Category.TABLET),
        ("tab", Category.TABLET),
        ("ipad", Category.TABLET),
        ("monitor", Category.MONITOR),
        ("screen", Category.MONITOR),
        ("display", Category.MONITOR),
        ("unknown_device", None),
        (None, None),
    ],
)
def test_normalize_category_aliases(raw_cat, expected_enum):
    """normalize_category maps slang, synonyms, and brand types to canonical Category."""
    assert normalize_category(raw_cat) == expected_enum


# ---------------------------------------------------------------------------
# Use Case Alias Tests
# ---------------------------------------------------------------------------

def test_normalize_use_cases_aliases():
    """normalize_use_cases maps synonyms to catalogue canonical use cases."""
    raw = ["coding", "software development", "college", "gaming", "unknown_custom"]
    normalized = normalize_use_cases(raw)
    assert normalized == ["programming", "student", "gaming", "unknown_custom"]


def test_normalize_use_cases_deduplication():
    """normalize_use_cases eliminates duplicates while preserving order."""
    raw = ["coding", "programming", "code", "dev"]
    normalized = normalize_use_cases(raw)
    assert normalized == ["programming"]


# ---------------------------------------------------------------------------
# Preference Normalization Tests
# ---------------------------------------------------------------------------

def test_normalize_preferences_aliases_and_clamping():
    """normalize_preferences clamps weights into [0.0, 1.0] and maps aliases."""
    raw = {
        "weight": 0.85,
        "battery": 1.5,      # should clamp to 1.0
        "camera": 0.0,      # weight <= 0 should be dropped
        "anc": 0.7,
        "unrecognized_key": 0.9,  # unknown key dropped
    }
    normalized = normalize_preferences(raw)
    assert normalized == {
        "portability": 0.85,
        "battery_life": 1.0,
        "noise_cancellation": 0.7,
    }


# ---------------------------------------------------------------------------
# Explicit Removal and Preference Conflict Handling
# ---------------------------------------------------------------------------

def test_normalize_strips_preferences_present_in_removed_fields():
    """normalize removes preferences if they appear in removed_fields."""
    delta = RequirementsDelta(
        category=Category.LAPTOP,
        preferences={"portability": 0.8, "performance": 0.9},
        removed_fields=["portability"],
    )
    norm = normalize(delta)
    assert norm.preferences == {"performance": 0.9}
    assert norm.removed_fields == ["portability"]


def test_normalize_removed_fields_aliases():
    """normalize_removed_fields resolves aliases to canonical removal fields."""
    raw = ["weight", "budget_limit", "anc", "unknown_xyz"]
    normalized = normalize_removed_fields(raw)
    assert normalized == ["portability", "budget_max_inr", "noise_cancellation"]


# ---------------------------------------------------------------------------
# Idempotence Test
# ---------------------------------------------------------------------------

def test_normalize_is_idempotent():
    """Applying normalize multiple times produces identical results."""
    delta = RequirementsDelta(
        category=Category.LAPTOP,
        use_cases=["coding", "college"],
        budget_max_inr=80000,
        hard=HardConstraints(min_ram_gb=16),
        preferences={"weight": 0.8},
        removed_fields=["battery"],
    )
    first_pass = normalize(delta)
    second_pass = normalize(first_pass)
    third_pass = normalize(second_pass)

    assert first_pass == second_pass
    assert second_pass == third_pass


# ---------------------------------------------------------------------------
# Verification Across All 15 Handwritten Examples
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("example", EXTRACTION_EXAMPLES, ids=[f"example_{e.id}" for e in EXTRACTION_EXAMPLES])
def test_normalize_across_all_15_examples(example):
    """Verify that normalize correctly transforms raw LLM outputs to expected deltas."""
    # Construct a delta from raw_llm_payload
    raw_delta = RequirementsDelta.model_validate(example.raw_llm_payload)
    normalized = normalize(raw_delta)

    # Verify category
    assert normalized.category == example.expected_delta.category

    # Verify use cases
    assert normalized.use_cases == example.expected_delta.use_cases

    # Verify budgets
    assert normalized.budget_max_inr == example.expected_delta.budget_max_inr
    assert normalized.budget_min_inr == example.expected_delta.budget_min_inr

    # Verify hard constraints
    assert normalized.hard == example.expected_delta.hard

    # Verify preferences
    assert normalized.preferences == example.expected_delta.preferences

    # Verify ml_mode
    assert normalized.ml_mode == example.expected_delta.ml_mode

    # Verify removals and off-topic
    assert normalized.removed_fields == example.expected_delta.removed_fields
    assert normalized.is_off_topic == example.expected_delta.is_off_topic
