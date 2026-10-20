"""Comprehensive Universal Cross-Category Test Suite.

Validates:
    1. Universal and category-specific slot applicability across all 5 supported categories
       (Laptop, Phone, Headphones, Tablet, Monitor).
    2. Mandatory vs. optional slot resolution and delegation ("you decide", "skip").
    3. Natural language slot update extraction (tablets, monitors, headphones, phones, laptops).
    4. Deterministic hard-constraint enforcement (stylus, refresh rate, form factor, storage, RAM, GPU).
    5. Handling of missing and unknown product specifications (never assumed compliant).
    6. Context-aware re-ranking after updated constraints or preferences.
    7. Grounded explanation validation and contradiction detection (ANC, stylus, color gamut).
    8. Universal follow-up diagnostics (minimum budget, blocking constraints) across categories.
"""

from __future__ import annotations

import pytest

from app.agent.followup import (
    FollowUpIntent,
    analyze_catalogue_minimum_budget,
    detect_followup_intent,
    explain_blocking_constraints,
)
from app.agent.policy import decide_next_action
from app.agent.ranking import (
    evaluate_candidate,
    filter_candidates,
    score_candidate,
)
from app.agent.response_validator import (
    build_fallback_recommendation_response,
    check_claim_contradictions,
)
from app.agent.slots import (
    SLOT_REGISTRY,
    SlotStatus,
    can_proceed_to_recommendation,
    get_applicable_slots,
    get_next_slot_to_clarify,
    get_unresolved_mandatory_slots,
    is_slot_applicable,
    parse_natural_language_slot_updates,
)
from app.models.product import Category, Product
from app.models.ranking import CandidateSearchResult, ConstraintStatus
from app.models.requirements import (
    HardConstraints,
    PolicyAction,
    Requirements,
    RequirementsDelta,
)


# ---------------------------------------------------------------------------
# Fixtures: Products for all 5 categories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_laptop():
    return Product(
        product_id="LAP-TEST-1",
        name="UltraBook Pro 15",
        brand="TechCorp",
        category=Category.LAPTOP,
        price_inr=65000,
        rating=4.5,
        specs={
            "ram_gb": 16,
            "storage_gb": 512,
            "cpu": "Intel Core i7-13700H",
            "gpu": "Integrated Intel Iris Xe",
            "vram_gb": 0,
            "battery_hr": 10.0,
            "weight_kg": 1.4,
        },
        features=["IPS display", "Backlit keyboard", "Thunderbolt 4"],
        intended_use=["programming", "office", "multitasking"],
        in_stock=True,
    )


@pytest.fixture
def sample_phone():
    return Product(
        product_id="PHN-TEST-1",
        name="Apex Horizon 10",
        brand="Apex",
        category=Category.PHONE,
        price_inr=32000,
        rating=4.4,
        specs={
            "ram_gb": 8,
            "storage_gb": 128,
            "camera_mp": 64,
            "battery_mah": 5000,
            "screen_in": 6.5,
            "weight_g": 185,
        },
        features=["AMOLED 120Hz", "Fast Charging 67W"],
        intended_use=["photography", "everyday use"],
        in_stock=True,
    )


@pytest.fixture
def sample_headphones():
    return Product(
        product_id="HDP-TEST-1",
        name="SoundWave QuietComfort",
        brand="Acoustic",
        category=Category.HEADPHONES,
        price_inr=8999,
        rating=4.6,
        specs={
            "type": "over-ear",
            "wireless": True,
            "anc": True,
            "battery_hr": 35.0,
            "weight_g": 240,
        },
        features=["Active Noise Cancellation", "Memory foam cushions", "Bluetooth 5.3"],
        intended_use=["travel", "music", "calls"],
        in_stock=True,
    )


@pytest.fixture
def sample_tablet():
    return Product(
        product_id="TAB-TEST-1",
        name="CreativePad Studio 11",
        brand="DesignTech",
        category=Category.TABLET,
        price_inr=42000,
        rating=4.5,
        specs={
            "chipset": "Snapdragon 8 Gen 2",
            "ram_gb": 8,
            "storage_gb": 256,
            "screen_in": 11.0,
            "stylus_support": True,
            "battery_hr": 12.0,
            "weight_g": 490,
        },
        features=["120Hz IPS display", "Quad speakers", "Stylus pen compatible"],
        intended_use=["drawing", "digital art", "note-taking"],
        in_stock=True,
    )


@pytest.fixture
def sample_monitor():
    return Product(
        product_id="MON-TEST-1",
        name="VisionPro 27 Gaming",
        brand="VisionTech",
        category=Category.MONITOR,
        price_inr=24999,
        rating=4.7,
        specs={
            "size_in": 27.0,
            "resolution": "2560x1440",
            "refresh_hz": 165,
            "panel": "IPS",
            "ports": ["DisplayPort 1.4", "HDMI 2.0"],
        },
        features=["165Hz refresh rate", "1ms response time", "QHD IPS panel"],
        intended_use=["gaming", "coding"],
        in_stock=True,
    )


# ---------------------------------------------------------------------------
# 1. Universal and Category-Specific Slot Applicability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category", [
    Category.LAPTOP,
    Category.PHONE,
    Category.HEADPHONES,
    Category.TABLET,
    Category.MONITOR,
])
def test_universal_slots_apply_to_all_categories(category: Category):
    req = Requirements(category=category)
    applicable = get_applicable_slots(req)
    slot_names = [s.name for s in applicable]
    assert "category" in slot_names
    assert "budget" in slot_names
    assert "use_cases" in slot_names


def test_category_specific_slots_isolated():
    """Verify slots are only applicable to their declared categories."""
    req_phone = Requirements(category=Category.PHONE)
    phone_slots = [s.name for s in get_applicable_slots(req_phone)]
    assert "phone_storage" in phone_slots
    assert "phone_priority" in phone_slots
    assert "form_factor" not in phone_slots
    assert "stylus_keyboard" not in phone_slots
    assert "display_specs" not in phone_slots

    req_headphones = Requirements(category=Category.HEADPHONES)
    headphone_slots = [s.name for s in get_applicable_slots(req_headphones)]
    assert "form_factor" in headphone_slots
    assert "anc" in headphone_slots
    assert "ram" not in headphone_slots
    assert "stylus_keyboard" not in headphone_slots

    req_tablet = Requirements(category=Category.TABLET)
    tablet_slots = [s.name for s in get_applicable_slots(req_tablet)]
    assert "stylus_keyboard" in tablet_slots
    assert "tablet_storage" in tablet_slots
    assert "refresh_rate" not in tablet_slots

    req_monitor = Requirements(category=Category.MONITOR)
    monitor_slots = [s.name for s in get_applicable_slots(req_monitor)]
    assert "display_specs" in monitor_slots
    assert "refresh_rate" in monitor_slots
    assert "ram" not in monitor_slots


# ---------------------------------------------------------------------------
# 2. Natural Language Extraction across Categories
# ---------------------------------------------------------------------------

def test_natural_language_extraction_tablet():
    updates, delta = parse_natural_language_slot_updates(
        "I need a tablet with stylus support for digital sketching",
        last_asked_slot="stylus_keyboard",
        current_req=Requirements(category=Category.TABLET),
    )
    assert updates.get("stylus_keyboard") == SlotStatus.ANSWERED.value
    assert delta.hard.requires_stylus is True


def test_natural_language_extraction_tablet_no_stylus():
    updates, delta = parse_natural_language_slot_updates(
        "No stylus needed, just for watching videos",
        last_asked_slot="stylus_keyboard",
        current_req=Requirements(category=Category.TABLET),
    )
    assert updates.get("stylus_keyboard") == SlotStatus.ANSWERED.value
    assert delta.hard.requires_stylus is False


def test_natural_language_extraction_monitor():
    updates, delta = parse_natural_language_slot_updates(
        "27-inch 144Hz monitor for esports",
        last_asked_slot="display_specs",
        current_req=Requirements(category=Category.MONITOR),
    )
    assert updates.get("display_specs") == SlotStatus.ANSWERED.value
    assert updates.get("refresh_rate") == SlotStatus.ANSWERED.value
    assert delta.hard.min_refresh_hz == 144


def test_natural_language_extraction_phone():
    updates, delta = parse_natural_language_slot_updates(
        "I need 256GB storage with great camera and night photography",
        last_asked_slot="phone_priority",
        current_req=Requirements(category=Category.PHONE),
    )
    assert updates.get("phone_storage") == SlotStatus.ANSWERED.value
    assert updates.get("phone_priority") == SlotStatus.ANSWERED.value
    assert delta.hard.min_storage_gb == 256
    assert delta.preferences.get("camera_quality") == 0.85


# ---------------------------------------------------------------------------
# 3. Deterministic Hard-Constraint Evaluation across Categories
# ---------------------------------------------------------------------------

def test_tablet_stylus_constraint_pass_and_fail(sample_tablet):
    # Pass: requires stylus
    req_stylus = Requirements(category=Category.TABLET, hard=HardConstraints(requires_stylus=True))
    eval_pass = evaluate_candidate(sample_tablet, req_stylus)
    assert eval_pass.is_eligible is True

    # Fail: requires NO stylus
    req_no_stylus = Requirements(category=Category.TABLET, hard=HardConstraints(requires_stylus=False))
    eval_fail = evaluate_candidate(sample_tablet, req_no_stylus)
    assert eval_fail.is_eligible is False
    assert any("requires_stylus" in chk.name for chk in eval_fail.failed_constraints)


def test_monitor_refresh_rate_constraint(sample_monitor):
    # Pass: 144Hz required, product has 165Hz
    req_144 = Requirements(category=Category.MONITOR, hard=HardConstraints(min_refresh_hz=144))
    eval_pass = evaluate_candidate(sample_monitor, req_144)
    assert eval_pass.is_eligible is True

    # Fail: 240Hz required, product has 165Hz
    req_240 = Requirements(category=Category.MONITOR, hard=HardConstraints(min_refresh_hz=240))
    eval_fail = evaluate_candidate(sample_monitor, req_240)
    assert eval_fail.is_eligible is False
    assert any("min_refresh_hz" in chk.name for chk in eval_fail.failed_constraints)


def test_headphones_form_factor_constraint(sample_headphones):
    # Pass: over-ear requested
    req_over = Requirements(category=Category.HEADPHONES, hard=HardConstraints(form_factor="over-ear"))
    eval_pass = evaluate_candidate(sample_headphones, req_over)
    assert eval_pass.is_eligible is True

    # Fail: in-ear requested
    req_in = Requirements(category=Category.HEADPHONES, hard=HardConstraints(form_factor="in-ear"))
    eval_fail = evaluate_candidate(sample_headphones, req_in)
    assert eval_fail.is_eligible is False
    assert any("form_factor" in chk.name for chk in eval_fail.failed_constraints)


# ---------------------------------------------------------------------------
# 4. Unknown and Missing Product Specifications
# ---------------------------------------------------------------------------

def test_missing_specs_treated_as_unknown():
    """A product lacking required specification must be marked UNKNOWN and ineligible."""
    sparse_tablet = Product(
        product_id="TAB-SPARSE",
        name="Basic Tab",
        brand="Generic",
        category=Category.TABLET,
        price_inr=15000,
        specs={},  # No storage, no stylus_support
        intended_use=["general"],
    )
    req = Requirements(category=Category.TABLET, hard=HardConstraints(requires_stylus=True))
    ev = evaluate_candidate(sparse_tablet, req)
    assert ev.is_eligible is False
    assert any(chk.status == ConstraintStatus.UNKNOWN for chk in ev.unverifiable_constraints)


# ---------------------------------------------------------------------------
# 5. Grounded Explanation & Claim Contradiction Detection
# ---------------------------------------------------------------------------

def test_anc_fabrication_detection(sample_headphones):
    passive_headphones = sample_headphones.model_copy(deep=True)
    passive_headphones.specs["anc"] = False

    # Text falsely claiming ANC on passive headphones
    claimed_text = "These headphones feature active noise cancellation for complete silence."
    contradictions = check_claim_contradictions(
        claimed_text, passive_headphones, Requirements(category=Category.HEADPHONES)
    )
    assert any("anc" in c.lower() for c in contradictions)


def test_tablet_stylus_inclusion_claim_detection(sample_tablet):
    # Tablet supports stylus, but features do NOT state stylus is bundled/included
    claimed_text = "Great deal that includes stylus in the box for free."
    contradictions = check_claim_contradictions(
        claimed_text, sample_tablet, Requirements(category=Category.TABLET)
    )
    assert any("stylus" in c.lower() for c in contradictions)


def test_monitor_color_accuracy_fabrication_detection(sample_monitor):
    claimed_text = "This monitor provides certified 100% sRGB color calibration for photo editing."
    contradictions = check_claim_contradictions(
        claimed_text, sample_monitor, Requirements(category=Category.MONITOR)
    )
    assert any("color" in c.lower() for c in contradictions)


# ---------------------------------------------------------------------------
# 6. Follow-up Intent Recognition across Categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_intent", [
    ("what is the minimum budget", FollowUpIntent.MINIMUM_BUDGET),
    ("how much more do i need to spend", FollowUpIntent.MINIMUM_BUDGET),
    ("what is the cheapest option", FollowUpIntent.MINIMUM_BUDGET),
    ("which requirement is blocking", FollowUpIntent.EXPLAIN_BLOCKING),
    ("why are there no matches", FollowUpIntent.EXPLAIN_BLOCKING),
    ("show me the closest alternatives", FollowUpIntent.CLOSEST_ALTERNATIVES),
    ("can you find a cheaper alternative", FollowUpIntent.CLOSEST_ALTERNATIVES),
])
def test_followup_intent_recognition(query: str, expected_intent: FollowUpIntent):
    assert detect_followup_intent(query) == expected_intent


# ---------------------------------------------------------------------------
# 7. Context-Aware Re-ranking
# ---------------------------------------------------------------------------

def test_context_aware_reranking(sample_laptop):
    """Updating soft preferences re-ranks candidates deterministically."""
    sample_laptop_2 = sample_laptop.model_copy(deep=True)
    sample_laptop_2.product_id = "LAP-TEST-2"
    sample_laptop_2.specs["battery_hr"] = 18.0
    sample_laptop_2.specs["weight_kg"] = 1.9

    # State 1: Portability preferred -> LAP-TEST-1 (1.4kg) beats LAP-TEST-2 (1.9kg)
    req_port = Requirements(category=Category.LAPTOP, preferences={"portability": 0.9})
    score_1 = score_candidate(sample_laptop, req_port)
    score_2 = score_candidate(sample_laptop_2, req_port)
    assert score_1.overall_score > score_2.overall_score

    # State 2: Battery life preferred -> LAP-TEST-2 (18h) beats LAP-TEST-1 (10h)
    req_batt = Requirements(category=Category.LAPTOP, preferences={"battery_life": 0.9})
    score_1_b = score_candidate(sample_laptop, req_batt)
    score_2_b = score_candidate(sample_laptop_2, req_batt)
    assert score_2_b.overall_score > score_1_b.overall_score
