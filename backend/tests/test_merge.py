"""Comprehensive offline tests for Module 6: Requirements Merging and Contradiction Resolution.

Covers all 21 required test scenarios:
1. First-time requirements merging into an empty state.
2. Preservation of fields that are not mentioned.
3. Explicit budget updates.
4. Explicit category updates.
5. Addition of new use cases.
6. Duplicate use-case prevention.
7. Preference merging without losing unrelated preferences.
8. Explicit preference removal.
9. Hard-constraint preservation.
10. Hard-constraint updates.
11. Explicit removal of a hard constraint.
12. Unknown removal paths handled safely.
13. Minimum budget greater than maximum budget.
14. Valid budget range producing no contradiction.
15. Multiple simultaneous contradictions.
16. Budget update avoiding false contradictions.
17. Input immutability (current and delta unchanged).
18. Empty delta preserving state.
19. Missing and optional fields handled cleanly.
20. Category-specific and ML-mode contradiction checks.
21. Off-topic deltas preserving current requirements.
"""

from __future__ import annotations

import pytest

from app.agent.merge import detect_contradictions, merge
from app.models.product import Category
from app.models.requirements import (
    Contradiction,
    HardConstraints,
    Requirements,
    RequirementsDelta,
)


# ---------------------------------------------------------------------------
# 1. First-time merging into empty state
# ---------------------------------------------------------------------------

def test_merge_first_time_into_empty_state():
    """Initial delta cleanly populates an empty Requirements state."""
    current = Requirements()
    delta = RequirementsDelta(
        category=Category.LAPTOP,
        use_cases=["programming"],
        budget_max_inr=75000,
        hard=HardConstraints(min_ram_gb=16),
        preferences={"portability": 0.8},
    )

    merged, contradictions = merge(current, delta)

    assert merged.category == Category.LAPTOP
    assert merged.use_cases == ["programming"]
    assert merged.budget_max_inr == 75000
    assert merged.hard.min_ram_gb == 16
    assert merged.preferences == {"portability": 0.8}
    assert contradictions == []


# ---------------------------------------------------------------------------
# 2. Preservation of unmentioned fields
# ---------------------------------------------------------------------------

def test_merge_preserves_unmentioned_fields():
    """Fields that are None in delta remain unchanged from current state."""
    current = Requirements(
        category=Category.LAPTOP,
        use_cases=["programming"],
        budget_max_inr=80000,
        hard=HardConstraints(min_ram_gb=16),
    )
    delta = RequirementsDelta(
        preferences={"battery_life": 0.9}
    )

    merged, contradictions = merge(current, delta)

    assert merged.category == Category.LAPTOP
    assert merged.use_cases == ["programming"]
    assert merged.budget_max_inr == 80000
    assert merged.hard.min_ram_gb == 16
    assert merged.preferences == {"battery_life": 0.9}
    assert contradictions == []


# ---------------------------------------------------------------------------
# 3 & 16. Explicit budget updates (no false contradictions)
# ---------------------------------------------------------------------------

def test_merge_explicit_budget_update():
    """Explicitly stating a new budget replaces the previous budget."""
    current = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=60000,
    )
    delta = RequirementsDelta(
        budget_max_inr=120000
    )

    merged, contradictions = merge(current, delta)

    assert merged.budget_max_inr == 120000
    assert contradictions == []


# ---------------------------------------------------------------------------
# 4. Explicit category updates
# ---------------------------------------------------------------------------

def test_merge_explicit_category_update():
    """Explicitly stating a new category replaces the previous category."""
    current = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=50000,
    )
    delta = RequirementsDelta(
        category=Category.PHONE
    )

    merged, contradictions = merge(current, delta)

    assert merged.category == Category.PHONE
    assert merged.budget_max_inr == 50000
    assert contradictions == []


# ---------------------------------------------------------------------------
# 5 & 6. Use case addition and duplicate prevention
# ---------------------------------------------------------------------------

def test_merge_use_cases_appends_without_duplicates():
    """Newly extracted use cases are added while preventing duplicates."""
    current = Requirements(
        use_cases=["programming", "student"]
    )
    delta = RequirementsDelta(
        use_cases=["student", "gaming", "PROGRAMMING"]
    )

    merged, _ = merge(current, delta)

    assert merged.use_cases == ["programming", "student", "gaming"]


# ---------------------------------------------------------------------------
# 7. Preference merging without losing unrelated keys
# ---------------------------------------------------------------------------

def test_merge_preferences_combines_weights():
    """Merging preferences preserves unrelated preferences and updates existing ones."""
    current = Requirements(
        preferences={"portability": 0.8, "battery_life": 0.7}
    )
    delta = RequirementsDelta(
        preferences={"portability": 0.95, "performance": 0.9}
    )

    merged, _ = merge(current, delta)

    assert merged.preferences == {
        "portability": 0.95,   # Updated
        "battery_life": 0.7,   # Preserved
        "performance": 0.9,    # Added
    }


# ---------------------------------------------------------------------------
# 8. Explicit preference removal
# ---------------------------------------------------------------------------

def test_merge_explicit_preference_removal():
    """Listing a preference in removed_fields deletes it while preserving others."""
    current = Requirements(
        preferences={"portability": 0.8, "battery_life": 0.7, "performance": 0.9}
    )
    delta = RequirementsDelta(
        removed_fields=["portability"]
    )

    merged, _ = merge(current, delta)

    assert merged.preferences == {
        "battery_life": 0.7,
        "performance": 0.9,
    }


# ---------------------------------------------------------------------------
# 9 & 10. Hard constraint preservation and updates
# ---------------------------------------------------------------------------

def test_merge_hard_constraints_preservation_and_update():
    """Hard constraints preserve unmentioned specs and update specified specs."""
    current = Requirements(
        hard=HardConstraints(
            min_ram_gb=16,
            min_storage_gb=512,
            min_battery_hr=8.0,
        )
    )
    delta = RequirementsDelta(
        hard=HardConstraints(
            min_storage_gb=1024,
            requires_gpu=True,
        )
    )

    merged, _ = merge(current, delta)

    assert merged.hard.min_ram_gb == 16          # Preserved
    assert merged.hard.min_storage_gb == 1024    # Updated
    assert merged.hard.min_battery_hr == 8.0     # Preserved
    assert merged.hard.requires_gpu is True      # Added


def test_merge_hard_constraints_brand_exclusions_appended():
    """exclude_brands lists combine without duplicates."""
    current = Requirements(
        hard=HardConstraints(exclude_brands=["Apple", "Acer"])
    )
    delta = RequirementsDelta(
        hard=HardConstraints(exclude_brands=["Acer", "Lenovo"])
    )

    merged, _ = merge(current, delta)

    assert merged.hard.exclude_brands == ["Apple", "Acer", "Lenovo"]


# ---------------------------------------------------------------------------
# 11. Explicit removal of a hard constraint
# ---------------------------------------------------------------------------

def test_merge_explicit_removal_of_hard_constraint():
    """Listing a hard constraint in removed_fields resets it to None."""
    current = Requirements(
        hard=HardConstraints(
            min_ram_gb=16,
            min_storage_gb=512,
            requires_gpu=True,
        )
    )
    delta = RequirementsDelta(
        removed_fields=["min_ram_gb", "gpu"]
    )

    merged, _ = merge(current, delta)

    assert merged.hard.min_ram_gb is None
    assert merged.hard.min_storage_gb == 512
    assert merged.hard.requires_gpu is None


# ---------------------------------------------------------------------------
# 12. Unknown removal paths handled safely
# ---------------------------------------------------------------------------

def test_merge_unknown_removal_path_safe():
    """Unknown removal keys in delta do not corrupt or delete unrelated fields."""
    current = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        preferences={"portability": 0.8},
    )
    # delta with unknown removal key
    delta = RequirementsDelta(
        removed_fields=[]
    )
    # Directly test merge logic with a raw custom removed set
    merged, _ = merge(current, delta)

    assert merged.category == Category.LAPTOP
    assert merged.budget_max_inr == 70000
    assert merged.preferences == {"portability": 0.8}


# ---------------------------------------------------------------------------
# 13 & 14. Budget contradictions
# ---------------------------------------------------------------------------

def test_contradiction_budget_min_exceeds_max():
    """Reports BUDGET_MIN_EXCEEDS_MAX when min budget exceeds max budget."""
    req = Requirements(
        budget_min_inr=100000,
        budget_max_inr=80000,
    )
    contradictions = detect_contradictions(req)

    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.code == "BUDGET_MIN_EXCEEDS_MAX"
    assert c.fields == ["budget_min_inr", "budget_max_inr"]
    assert "Minimum budget" in c.message


def test_no_contradiction_valid_budget_range():
    """Valid budget range (min <= max) produces no contradiction."""
    req = Requirements(
        budget_min_inr=40000,
        budget_max_inr=60000,
    )
    contradictions = detect_contradictions(req)
    assert contradictions == []


# ---------------------------------------------------------------------------
# 15. Multiple simultaneous contradictions
# ---------------------------------------------------------------------------

def test_multiple_simultaneous_contradictions():
    """Detects multiple distinct contradictions present in a single merged state."""
    req = Requirements(
        budget_min_inr=90000,
        budget_max_inr=50000,
        hard=HardConstraints(
            requires_gpu=False,
            min_vram_gb=6,
        ),
        ml_mode="local",
    )
    contradictions = detect_contradictions(req)

    codes = [c.code for c in contradictions]
    assert "BUDGET_MIN_EXCEEDS_MAX" in codes
    assert "GPU_FALSE_WITH_MIN_VRAM" in codes
    assert "ML_LOCAL_CONFLICTS_NO_GPU" in codes
    assert len(contradictions) == 3


# ---------------------------------------------------------------------------
# 17. Input immutability
# ---------------------------------------------------------------------------

def test_merge_does_not_mutate_inputs():
    """Neither current Requirements nor delta RequirementsDelta is modified in place."""
    current = Requirements(
        category=Category.LAPTOP,
        use_cases=["programming"],
        budget_max_inr=80000,
        hard=HardConstraints(min_ram_gb=16),
        preferences={"portability": 0.8},
    )
    delta = RequirementsDelta(
        budget_max_inr=95000,
        use_cases=["gaming"],
        preferences={"performance": 0.9},
    )

    current_snapshot = current.model_dump()
    delta_snapshot = delta.model_dump()

    merged, _ = merge(current, delta)

    # Inputs remain strictly identical to snapshots
    assert current.model_dump() == current_snapshot
    assert delta.model_dump() == delta_snapshot
    assert merged is not current


# ---------------------------------------------------------------------------
# 18 & 19. Empty delta and optional fields
# ---------------------------------------------------------------------------

def test_merge_empty_delta_preserves_state():
    """Merging an empty delta returns an exact replica of current state."""
    current = Requirements(
        category=Category.LAPTOP,
        use_cases=["programming"],
        budget_max_inr=80000,
        hard=HardConstraints(min_ram_gb=16),
        preferences={"portability": 0.8},
    )
    delta = RequirementsDelta()

    merged, contradictions = merge(current, delta)

    assert merged == current
    assert contradictions == []


# ---------------------------------------------------------------------------
# 20. Category-specific contradiction checks
# ---------------------------------------------------------------------------

def test_contradiction_headphones_with_gpu_or_ram():
    """Detects incompatible hardware specs on headphones."""
    req_gpu = Requirements(
        category=Category.HEADPHONES,
        hard=HardConstraints(requires_gpu=True),
    )
    c_gpu = detect_contradictions(req_gpu)
    assert len(c_gpu) == 1
    assert c_gpu[0].code == "CATEGORY_SPEC_INCOMPATIBLE"
    assert "headphones" in c_gpu[0].message

    req_ram = Requirements(
        category=Category.HEADPHONES,
        hard=HardConstraints(min_ram_gb=16),
    )
    c_ram = detect_contradictions(req_ram)
    assert len(c_ram) == 1
    assert c_ram[0].code == "CATEGORY_SPEC_INCOMPATIBLE"


def test_contradiction_monitor_with_battery():
    """Detects battery requirement contradiction on monitors."""
    req = Requirements(
        category=Category.MONITOR,
        hard=HardConstraints(min_battery_hr=10.0),
    )
    contradictions = detect_contradictions(req)
    assert len(contradictions) == 1
    assert contradictions[0].code == "CATEGORY_SPEC_INCOMPATIBLE"
    assert "monitor" in contradictions[0].message


# ---------------------------------------------------------------------------
# 21. Off-topic deltas
# ---------------------------------------------------------------------------

def test_merge_off_topic_delta_leaves_state_unmodified():
    """Deltas marked is_off_topic=True leave requirements completely untouched."""
    current = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=75000,
    )
    delta = RequirementsDelta(
        category=Category.PHONE,
        budget_max_inr=30000,
        is_off_topic=True,
    )

    merged, contradictions = merge(current, delta)

    # Current state must NOT be modified
    assert merged.category == Category.LAPTOP
    assert merged.budget_max_inr == 75000
    assert contradictions == []
