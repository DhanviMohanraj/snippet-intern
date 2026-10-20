"""Deterministic Requirement Slots Registry and Decision Gate.

Defines:
    - Slot definitions and metadata across categories (Laptop, Phone, Headphones, Tablet, Monitor).
    - Classification of Mandatory decision inputs vs. Optional preferences.
    - Slot status tracking: ANSWERED, DECLINED, DELEGATED, NOT_APPLICABLE, UNKNOWN.
    - Intent recognition for natural language answers: "you decide", "skip", "I don't know", and spec phrases.
    - Deterministic recommendation gate (can_recommend).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.agent.normalize import parse_inr_budget_expression
from app.models.product import Category
from app.models.requirements import Contradiction, HardConstraints, Requirements, RequirementsDelta

# ML use case keywords
ML_USE_CASE_KEYWORDS: set[str] = {
    "ml",
    "machine learning",
    "deep learning",
    "ai",
    "artificial intelligence",
    "data science",
    "model training",
}

# Gaming / Graphics use case keywords
GAMING_GRAPHICS_KEYWORDS: set[str] = {
    "gaming",
    "game",
    "games",
    "gamer",
    "gamming",
    "esports",
    "video editing",
    "editing",
    "3d",
    "render",
    "rendering",
    "animation",
    "blender",
}


class SlotStatus(str, Enum):
    """Resolution status of a conversational requirement slot."""

    UNKNOWN = "unknown"
    ANSWERED = "answered"
    DECLINED = "declined"     # user explicitly skipped, said "skip", "I don't know", "no preference"
    DELEGATED = "delegated"   # user said "you decide", "your choice", "whatever you recommend"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class RequirementSlot:
    """Defines a single decision-relevant requirements slot."""

    name: str
    categories: list[Category] | None  # None = applies to all categories
    is_mandatory: bool
    field_name: str
    question: str
    depends_on: str | None = None
    condition_fn: Callable[[Requirements], bool] | None = None
    can_skip: bool = True
    examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Slot Registry Definition
# ---------------------------------------------------------------------------

SLOT_REGISTRY: list[RequirementSlot] = [
    # 1. Universal Slots
    RequirementSlot(
        name="category",
        categories=None,
        is_mandatory=True,
        field_name="category",
        question=(
            "What type of electronics are you looking for? "
            "(e.g., laptop, phone, headphones, tablet, or monitor)"
        ),
        can_skip=False,
    ),
    RequirementSlot(
        name="budget",
        categories=None,
        is_mandatory=True,
        field_name="budget_max_inr",
        question=(
            "What is your budget or maximum price in INR? "
            "(e.g., ₹50,000, ₹70,000, or let me know if your budget is flexible)"
        ),
        depends_on="category",
        can_skip=True,
    ),
    RequirementSlot(
        name="use_cases",
        categories=None,
        is_mandatory=True,
        field_name="use_cases",
        question="What is your primary intended use for this device?",
        depends_on="category",
        can_skip=True,
    ),

    # 2. Laptop Slots
    RequirementSlot(
        name="ram",
        categories=[Category.LAPTOP],
        is_mandatory=True,
        field_name="hard.min_ram_gb",
        question=(
            "How much RAM do you need for your workload? "
            "(e.g., 8GB, 16GB, or 32GB — 16GB is recommended for programming and multitasking)"
        ),
        depends_on="category",
        can_skip=True,
        examples=["8GB", "16GB", "32GB"],
    ),
    RequirementSlot(
        name="storage",
        categories=[Category.LAPTOP],
        is_mandatory=True,
        field_name="hard.min_storage_gb",
        question="What is your preferred storage capacity? (e.g., 512GB SSD or 1TB SSD)",
        depends_on="category",
        can_skip=True,
        examples=["512GB", "1TB"],
    ),
    RequirementSlot(
        name="gpu",
        categories=[Category.LAPTOP],
        is_mandatory=True,
        field_name="hard.requires_gpu",
        question=(
            "Do you need a dedicated graphics card (GPU) for your gaming or 3D work? "
            "(e.g., dedicated NVIDIA RTX series, or is an integrated GPU fine?)"
        ),
        depends_on="category",
        condition_fn=lambda req: any(
            any(kw in uc.lower() for kw in GAMING_GRAPHICS_KEYWORDS)
            for uc in req.use_cases
        ),
        can_skip=True,
    ),
    RequirementSlot(
        name="ml_mode",
        categories=[Category.LAPTOP],
        is_mandatory=True,
        field_name="ml_mode",
        question=(
            "Will you be training machine learning models locally on the laptop (requiring a dedicated GPU), "
            "or using cloud platforms like Google Colab or AWS?"
        ),
        depends_on="category",
        condition_fn=lambda req: any(
            any(kw in uc.lower() for kw in ML_USE_CASE_KEYWORDS)
            for uc in req.use_cases
        ),
        can_skip=True,
    ),
    RequirementSlot(
        name="portability_battery",
        categories=[Category.LAPTOP],
        is_mandatory=False,
        field_name="preferences.portability",
        question="Is lightweight portability (under 1.5kg) or extended battery life a high priority for you?",
        depends_on="category",
        can_skip=True,
    ),

    # 3. Phone Slots
    RequirementSlot(
        name="phone_storage",
        categories=[Category.PHONE],
        is_mandatory=True,
        field_name="hard.min_storage_gb",
        question="How much internal storage capacity do you need? (e.g., 128GB, 256GB, or 512GB)",
        depends_on="category",
        can_skip=True,
    ),
    RequirementSlot(
        name="phone_priority",
        categories=[Category.PHONE],
        is_mandatory=True,
        field_name="preferences",
        question=(
            "What is your top priority for this phone? "
            "(e.g., camera quality, gaming performance, long battery life, or compact design)"
        ),
        depends_on="category",
        can_skip=True,
    ),

    # 4. Headphones Slots
    RequirementSlot(
        name="form_factor",
        categories=[Category.HEADPHONES],
        is_mandatory=True,
        field_name="form_factor",
        question=(
            "Do you prefer over-ear headphones (for maximum comfort and sound isolation) "
            "or in-ear true wireless (TWS) earbuds (for portability)?"
        ),
        depends_on="category",
        can_skip=True,
    ),
    RequirementSlot(
        name="anc",
        categories=[Category.HEADPHONES],
        is_mandatory=False,
        field_name="preferences.noise_cancellation",
        question="Is Active Noise Cancellation (ANC) an essential feature for you?",
        depends_on="category",
        can_skip=True,
    ),

    # 5. Tablet Slots
    RequirementSlot(
        name="stylus_keyboard",
        categories=[Category.TABLET],
        is_mandatory=True,
        field_name="stylus_support",
        question=(
            "Do you require stylus/pen support (for drawing and note-taking) "
            "or an attachable keyboard for typing?"
        ),
        depends_on="category",
        can_skip=True,
    ),
    RequirementSlot(
        name="tablet_storage",
        categories=[Category.TABLET],
        is_mandatory=False,
        field_name="hard.min_storage_gb",
        question="What storage capacity do you prefer for this tablet? (e.g., 64GB, 128GB, or 256GB)",
        depends_on="category",
        can_skip=True,
    ),

    # 6. Monitor Slots
    RequirementSlot(
        name="display_specs",
        categories=[Category.MONITOR],
        is_mandatory=True,
        field_name="preferences.display_quality",
        question=(
            "What screen size and resolution do you prefer? "
            "(e.g., 24-inch 1080p, 27-inch 1440p QHD, or 4K UHD)"
        ),
        depends_on="category",
        can_skip=True,
    ),
    RequirementSlot(
        name="refresh_rate",
        categories=[Category.MONITOR],
        is_mandatory=False,
        field_name="refresh_rate",
        question="Do you need a high refresh rate (e.g., 144Hz, 165Hz, or 240Hz for competitive gaming)?",
        depends_on="category",
        can_skip=True,
    ),
]


# ---------------------------------------------------------------------------
# Natural Language Intent Recognition Helpers
# ---------------------------------------------------------------------------

RE_DELEGATION = re.compile(
    r"\b(you\s+decide|your\s+choice|whatever\s+you\s+(think|recommend)|recommend\s+for\s+me|"
    r"leave\s+it\s+to\s+you|up\s+to\s+you|best\s+available|anything\s+good|you\s+pick)\b",
    re.IGNORECASE,
)

RE_DECLINE_OR_SKIP = re.compile(
    r"\b(skip|i\s+don'?t\s+know|no\s+preference|doesn'?t\s+matter|any(?:\s+is\s+fine)?|"
    r"either(?:\s+is\s+fine)?|not\s+sure|don'?t\s+care|flexible|no\s+budget\s+limit|unlimited)\b",
    re.IGNORECASE,
)

RE_RAM = re.compile(r"\b(4|8|16|24|32|64)\s*(?:gb|gigs?)\s*(?:ram)?\b", re.IGNORECASE)
RE_STORAGE = re.compile(
    r"(?:\b(?:storage|ssd|rom|disk|capacity)\s*(?:to|of|at|is)?\s*(\d+)\s*(?:gb|tb)?\b|"
    r"\b(64|128|256|512|1024|1|2)\s*(?:gb|tb)\s*(?:ssd|storage|rom|hdd|hard\s*drive)?(?!\s*ram)\b)",
    re.IGNORECASE,
)
RE_GPU_NEED = re.compile(r"\b(rtx\s*\d{4}|dedicated\s*gpu|nvidia|graphics\s*card|gtx|radeon)\b", re.IGNORECASE)
RE_GPU_NO_NEED = re.compile(r"\b(integrated|no\s*gpu|don'?t\s*need\s*gpu|not\s*needed|integrated\s*graphics)\b", re.IGNORECASE)
RE_OVER_EAR = re.compile(r"\b(over[-\s]?ear|on[-\s]?ear|overhead)\b", re.IGNORECASE)
RE_IN_EAR = re.compile(r"\b(in[-\s]?ear|tws|earbuds?|earphones?|in[-\s]?canal)\b", re.IGNORECASE)
RE_STYLUS = re.compile(r"\b(stylus|pen|pencil|apple\s*pencil|s-?pen|drawing|notes?|note-?taking|keyboard|typing)\b", re.IGNORECASE)
RE_NO_STYLUS = re.compile(r"\b(no\s+stylus|don'?t\s+need\s+stylus|no\s+pen|without\s+stylus|neither|not\s+needed)\b", re.IGNORECASE)
RE_WIRELESS = re.compile(r"\b(wireless|bluetooth|true\s+wireless)\b", re.IGNORECASE)
RE_ANC = re.compile(r"\b(anc|active\s+noise\s+cancellation|noise\s+cancelling|noise\s+cancellation)\b", re.IGNORECASE)
RE_MIC = re.compile(r"\b(mic|microphone|voice\s*calls?|calling|online\s*classes?|zoom|teams)\b", re.IGNORECASE)
RE_COMFORT = re.compile(r"\b(comfort|comfortable|cushions?|ear\s*cushions?|hours\s*(?:daily|a\s*day))\b", re.IGNORECASE)
RE_BATTERY = re.compile(r"\b(battery|battery\s*life|endurance|playtime)\b", re.IGNORECASE)
RE_CAMERA = re.compile(r"\b(camera|photo|photography|night\s*(?:mode|shots?|photography)?|portrait|zoom)\b", re.IGNORECASE)
RE_PERF = re.compile(r"\b(gaming|speed|fast|performance|heavy\s*multitasking)\b", re.IGNORECASE)
RE_COMPACT = re.compile(r"\b(compact|small\s*size|single[-\s]hand|lightweight)\b", re.IGNORECASE)
RE_REFRESH_HZ = re.compile(r"\b(60|75|90|120|144|165|240|360)\s*hz\b", re.IGNORECASE)
RE_DISPLAY_RES = re.compile(r"\b(1080p|1440p|4k|2k|qhd|uhd|fhd|retina|oled|ips)\b", re.IGNORECASE)
RE_DISPLAY_SIZE = re.compile(r"\b(24|27|32|34|11|12|13|14|15|16)\s*(?:-?\s*inch|\")\b", re.IGNORECASE)


def parse_natural_language_slot_updates(
    text: str,
    last_asked_slot: str | None,
    current_req: Requirements,
) -> tuple[dict[str, str], RequirementsDelta]:
    """Extract slot status updates and concrete requirement values from user message.

    Handles:
        - "you decide" / "up to you" -> status: DELEGATED
        - "I don't know" / "skip" / "no preference" -> status: DECLINED
        - Concrete spec phrases (e.g. "16GB RAM and 512GB SSD") -> updates delta and marks ANSWERED
    """
    text_clean = text.strip()
    status_updates: dict[str, str] = {}
    delta_kwargs: dict[str, Any] = {}
    hard_kwargs: dict[str, Any] = {}
    pref_kwargs: dict[str, float] = {}

    # 1. Check for delegation ("you decide")
    if RE_DELEGATION.search(text_clean):
        if last_asked_slot:
            status_updates[last_asked_slot] = SlotStatus.DELEGATED.value

    # 2. Check for decline / skip ("skip", "I don't know", "no preference")
    if RE_DECLINE_OR_SKIP.search(text_clean):
        if last_asked_slot:
            status_updates[last_asked_slot] = SlotStatus.DECLINED.value
        # If user says flexible / no budget limit
        if "budget" in (last_asked_slot or "") or re.search(r"\b(flexible|no\s+budget|unlimited)\b", text_clean, re.I):
            status_updates["budget"] = SlotStatus.DECLINED.value

    # 3. Direct spec extraction helpers
    # RAM
    ram_match = RE_RAM.search(text_clean)
    if ram_match:
        ram_val = int(ram_match.group(1))
        hard_kwargs["min_ram_gb"] = ram_val
        status_updates["ram"] = SlotStatus.ANSWERED.value

    # Storage
    storage_match = RE_STORAGE.search(text_clean)
    if storage_match:
        val_str = storage_match.group(1) or storage_match.group(2)
        if val_str:
            val = int(val_str)
            # 1TB or 2TB convert to GB
            if "tb" in storage_match.group(0).lower():
                val = val * 1024
            hard_kwargs["min_storage_gb"] = val
            status_updates["storage"] = SlotStatus.ANSWERED.value
            status_updates["phone_storage"] = SlotStatus.ANSWERED.value
            status_updates["tablet_storage"] = SlotStatus.ANSWERED.value

    # GPU
    if RE_GPU_NEED.search(text_clean):
        hard_kwargs["requires_gpu"] = True
        status_updates["gpu"] = SlotStatus.ANSWERED.value
    elif RE_GPU_NO_NEED.search(text_clean):
        hard_kwargs["requires_gpu"] = False
        status_updates["gpu"] = SlotStatus.ANSWERED.value

    # Headphones Form Factor
    if RE_OVER_EAR.search(text_clean) and not RE_IN_EAR.search(text_clean):
        hard_kwargs["form_factor"] = "over-ear"
        status_updates["form_factor"] = SlotStatus.ANSWERED.value
    elif RE_IN_EAR.search(text_clean) and not RE_OVER_EAR.search(text_clean):
        hard_kwargs["form_factor"] = "in-ear"
        status_updates["form_factor"] = SlotStatus.ANSWERED.value

    # Headphones & Audio features
    if RE_WIRELESS.search(text_clean):
        hard_kwargs["requires_wireless"] = True
    if RE_ANC.search(text_clean):
        pref_kwargs["noise_cancellation"] = 0.85
        status_updates["anc"] = SlotStatus.ANSWERED.value
    if RE_MIC.search(text_clean):
        pref_kwargs["microphone_quality"] = 0.85
    if RE_COMFORT.search(text_clean):
        pref_kwargs["comfort"] = 0.85
    if RE_BATTERY.search(text_clean):
        pref_kwargs["battery_life"] = 0.85
        if last_asked_slot == "phone_priority":
            status_updates["phone_priority"] = SlotStatus.ANSWERED.value

    # Phone Priorities
    if RE_CAMERA.search(text_clean):
        pref_kwargs["camera_quality"] = 0.85
        status_updates["phone_priority"] = SlotStatus.ANSWERED.value
    if RE_PERF.search(text_clean):
        pref_kwargs["performance"] = 0.85
        if last_asked_slot == "phone_priority":
            status_updates["phone_priority"] = SlotStatus.ANSWERED.value
    if RE_COMPACT.search(text_clean):
        pref_kwargs["portability"] = 0.85
        if last_asked_slot == "phone_priority":
            status_updates["phone_priority"] = SlotStatus.ANSWERED.value

    # Monitor & Display features
    hz_match = RE_REFRESH_HZ.search(text_clean)
    if hz_match:
        hz_val = int(hz_match.group(1))
        hard_kwargs["min_refresh_hz"] = hz_val
        status_updates["refresh_rate"] = SlotStatus.ANSWERED.value
    if RE_DISPLAY_RES.search(text_clean) or RE_DISPLAY_SIZE.search(text_clean):
        pref_kwargs["display_quality"] = 0.85
        status_updates["display_specs"] = SlotStatus.ANSWERED.value

    # Budget extraction helper
    if (
        last_asked_slot == "budget"
        or re.search(
            r"(?:under|below|budget(?:\s+(?:is|to|of|at))?|max|up to|rs\.?|inr|₹|increase\s+(?:my\s+)?budget\s+to|raise\s+(?:my\s+)?budget\s+to)\s*\d",
            text_clean,
            re.I,
        )
    ):
        parsed_b = parse_inr_budget_expression(text_clean)
        if parsed_b is not None:
            delta_kwargs["budget_max_inr"] = parsed_b
            status_updates["budget"] = SlotStatus.ANSWERED.value

    # Stylus / Tablet
    if RE_NO_STYLUS.search(text_clean):
        hard_kwargs["requires_stylus"] = False
        status_updates["stylus_keyboard"] = SlotStatus.ANSWERED.value
    elif RE_STYLUS.search(text_clean):
        hard_kwargs["requires_stylus"] = True
        status_updates["stylus_keyboard"] = SlotStatus.ANSWERED.value

    if hard_kwargs:
        delta_kwargs["hard"] = HardConstraints(**hard_kwargs)
    if pref_kwargs:
        delta_kwargs["preferences"] = pref_kwargs

    delta = RequirementsDelta(**delta_kwargs) if delta_kwargs else RequirementsDelta()
    return status_updates, delta


# ---------------------------------------------------------------------------
# Slot Evaluation Engine
# ---------------------------------------------------------------------------

def is_slot_applicable(slot: RequirementSlot, req: Requirements) -> bool:
    """Check whether a slot applies to the current requirements."""
    if slot.categories is not None:
        if req.category is None or req.category not in slot.categories:
            return False

    if slot.condition_fn is not None:
        if not slot.condition_fn(req):
            return False

    return True


def is_slot_answered(slot: RequirementSlot, req: Requirements, slot_status: dict[str, str]) -> bool:
    """Determine whether a slot has been answered with a concrete value."""
    # Check explicit slot status
    current_status = slot_status.get(slot.name)
    if current_status == SlotStatus.ANSWERED.value:
        return True

    # Check underlying Requirements object fields
    if slot.name == "category":
        return req.category is not None

    if slot.name == "use_cases":
        return len(req.use_cases) > 0

    if slot.name == "budget":
        return req.budget_max_inr is not None

    if slot.name == "ram":
        return req.hard.min_ram_gb is not None

    if slot.name in ("storage", "phone_storage", "tablet_storage"):
        return req.hard.min_storage_gb is not None

    if slot.name == "gpu":
        return req.hard.requires_gpu is not None

    if slot.name == "ml_mode":
        return req.ml_mode is not None or req.hard.requires_gpu is not None

    if slot.name == "phone_priority":
        return any(
            k in req.preferences
            for k in ("camera_quality", "performance", "battery_life", "display_quality")
        )

    if slot.name == "form_factor":
        return req.hard.form_factor is not None or "form_factor" in slot_status

    if slot.name == "anc":
        return "noise_cancellation" in req.preferences or "anc" in slot_status

    if slot.name == "stylus_keyboard":
        return "stylus_keyboard" in slot_status or req.hard.requires_stylus is not None

    if slot.name == "display_specs":
        return "display_quality" in req.preferences or "display_specs" in slot_status or req.hard.min_refresh_hz is not None

    if slot.name == "refresh_rate":
        return req.hard.min_refresh_hz is not None or "refresh_rate" in slot_status

    return False


def is_slot_resolved(slot: RequirementSlot, req: Requirements, slot_status: dict[str, str]) -> bool:
    """Check if a slot is satisfied, declined, delegated, or not applicable."""
    if not is_slot_applicable(slot, req):
        return True

    current_status = slot_status.get(slot.name)
    if current_status in (SlotStatus.DECLINED.value, SlotStatus.DELEGATED.value, SlotStatus.NOT_APPLICABLE.value):
        return True

    return is_slot_answered(slot, req, slot_status)


def get_applicable_slots(req: Requirements) -> list[RequirementSlot]:
    """Retrieve all slots that are applicable to current requirements state."""
    return [slot for slot in SLOT_REGISTRY if is_slot_applicable(slot, req)]


def get_unresolved_mandatory_slots(
    req: Requirements,
    slot_status: dict[str, str],
) -> list[RequirementSlot]:
    """Get list of mandatory slots that remain unresolved."""
    unresolved: list[RequirementSlot] = []
    for slot in SLOT_REGISTRY:
        if slot.is_mandatory and is_slot_applicable(slot, req):
            if not is_slot_resolved(slot, req, slot_status):
                unresolved.append(slot)
    return unresolved


def get_next_slot_to_clarify(
    req: Requirements,
    slot_status: dict[str, str],
    asked_slots: list[str],
) -> RequirementSlot | None:
    """Deterministically pick the next best mandatory slot to ask."""
    unresolved = get_unresolved_mandatory_slots(req, slot_status)
    if not unresolved:
        return None

    # Filter out slots already asked, unless no other unresolved slot exists
    asked_set = {s.strip().lower() for s in asked_slots if isinstance(s, str)}
    unasked = [s for s in unresolved if s.name.lower() not in asked_set]
    if unasked:
        return unasked[0]

    return unresolved[0]


def can_proceed_to_recommendation(
    req: Requirements,
    slot_status: dict[str, str],
    contradictions: list[Contradiction],
) -> tuple[bool, str | None, list[str]]:
    """Recommendation Gate: Deterministically verifies whether product recommendation is permitted.

    Returns:
        (can_recommend, blocking_reason, remaining_mandatory_slot_names)
    """
    if len(contradictions) > 0:
        return False, "unresolved_contradictions", []

    if req.category is None:
        return False, "missing_category", ["category"]

    unresolved = get_unresolved_mandatory_slots(req, slot_status)
    if unresolved:
        names = [s.name for s in unresolved]
        return False, f"missing_mandatory_slots:{','.join(names)}", names

    return True, None, []
