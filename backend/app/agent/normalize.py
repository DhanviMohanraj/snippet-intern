"""Deterministic normalization module for Requirements and RequirementsDelta.

Provides pure Python normalization of Indian Rupee (INR) currency expressions,
category and use-case aliases, preference names and numeric weights,
and safe removal tracking. No network or LLM operations are performed.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.product import Category
from app.models.requirements import (
    HardConstraints,
    RECOGNIZED_PREFERENCES,
    RequirementsDelta,
    VALID_REMOVABLE_FIELDS,
)

# ---------------------------------------------------------------------------
# Canonical Vocabularies and Alias Maps
# ---------------------------------------------------------------------------

CATEGORY_ALIASES: dict[str, Category] = {
    # Laptop
    "laptop": Category.LAPTOP,
    "notebook": Category.LAPTOP,
    "lappy": Category.LAPTOP,
    "macbook": Category.LAPTOP,
    "ultrabook": Category.LAPTOP,
    "chromebook": Category.LAPTOP,
    "pc": Category.LAPTOP,
    # Phone
    "phone": Category.PHONE,
    "smartphone": Category.PHONE,
    "mobile": Category.PHONE,
    "cellphone": Category.PHONE,
    "android": Category.PHONE,
    "iphone": Category.PHONE,
    # Headphones
    "headphones": Category.HEADPHONES,
    "headphone": Category.HEADPHONES,
    "earphones": Category.HEADPHONES,
    "earphone": Category.HEADPHONES,
    "headset": Category.HEADPHONES,
    "earbuds": Category.HEADPHONES,
    "earbud": Category.HEADPHONES,
    "tws": Category.HEADPHONES,
    # Tablet
    "tablet": Category.TABLET,
    "tab": Category.TABLET,
    "ipad": Category.TABLET,
    # Monitor
    "monitor": Category.MONITOR,
    "screen": Category.MONITOR,
    "display": Category.MONITOR,
}

USE_CASE_ALIASES: dict[str, str] = {
    # Programming
    "coding": "programming",
    "software development": "programming",
    "software": "programming",
    "programming": "programming",
    "developer": "programming",
    "dev": "programming",
    "code": "programming",
    # Machine Learning
    "ml": "machine_learning",
    "machine learning": "machine_learning",
    "machine_learning": "machine_learning",
    "deep learning": "machine_learning",
    "ai": "machine_learning",
    "data science": "machine_learning",
    # Gaming
    "gaming": "gaming",
    "gamming": "gaming",
    "games": "gaming",
    "esports": "gaming",
    "gamer": "gaming",

    # Content Creation
    "content_creation": "content_creation",
    "content creation": "content_creation",
    "video editing": "content_creation",
    "video_editing": "content_creation",
    "editing": "content_creation",
    "creator": "content_creation",
    "game dev": "content_creation",
    "game development": "content_creation",
    # Student / Study
    "college": "student",
    "student": "student",
    "school": "student",
    "university": "student",
    "classes": "student",
    "study": "study",
    "studying": "study",
    "studies": "study",
    # Office
    "office": "office",
    "work": "office",
    "corporate": "office",
    "business": "office",
    # Travel
    "travel": "travel",
    "traveling": "travel",
    "commute": "travel",
    "portable": "travel",
    # Photography
    "photography": "photography",
    "photos": "photography",
    "camera": "photography",
    # Design / Drawing
    "design": "design",
    "graphic design": "design",
    "ui/ux": "design",
    "drawing": "drawing",
    "sketching": "drawing",
    "illustration": "drawing",
    "digital art": "drawing",
    # Music / Audio
    "music": "music",
    "listening": "music",
    "audiophile": "music",
    "audio": "music",
    # Calls / Meetings
    "calls": "calls",
    "calling": "calls",
    "meetings": "calls",
    "zoom": "calls",
    # Note taking
    "note taking": "note_taking",
    "note_taking": "note_taking",
    "notes": "note_taking",
    # Multitasking & General
    "multitasking": "multitasking",
    "general": "general",
    "daily use": "general",
    "casual": "general",
    "browsing": "general",
    "productivity": "productivity",
    "entertainment": "entertainment",
    "movies": "entertainment",
}

PREFERENCE_ALIASES: dict[str, str] = {
    "portability": "portability",
    "portable": "portability",
    "lightweight": "portability",
    "weight": "portability",
    "slim": "portability",
    "performance": "performance",
    "speed": "performance",
    "fast": "performance",
    "battery_life": "battery_life",
    "battery": "battery_life",
    "endurance": "battery_life",
    "camera_quality": "camera_quality",
    "camera": "camera_quality",
    "display_quality": "display_quality",
    "display": "display_quality",
    "screen": "display_quality",
    "screen_quality": "display_quality",
    "gpu_power": "gpu_power",
    "gpu": "gpu_power",
    "graphics": "gpu_power",
    "build_quality": "build_quality",
    "build": "build_quality",
    "durability": "build_quality",
    "noise_cancellation": "noise_cancellation",
    "anc": "noise_cancellation",
}

REMOVAL_FIELD_ALIASES: dict[str, str] = {
    "budget": "budget_max_inr",
    "budget_max": "budget_max_inr",
    "budget_min": "budget_min_inr",
    "budget_limit": "budget_max_inr",
    "max_budget": "budget_max_inr",
    "min_budget": "budget_min_inr",
    "price": "budget_max_inr",
    "price_limit": "budget_max_inr",
    "weight": "portability",
    "lightweight": "portability",
    "portability_preference": "portability",
    "battery": "battery_life",
    "camera": "camera_quality",
    "display": "display_quality",
    "gpu": "requires_gpu",
    "gpu_preference": "gpu_power",
    "ram": "min_ram_gb",
    "storage": "min_storage_gb",
    "anc": "noise_cancellation",
}


# ---------------------------------------------------------------------------
# INR Budget Expression Normalizer
# ---------------------------------------------------------------------------

# Regex for Indian numbers like 80k, 1.2 lakh, 70 hazaar, ₹80,000
LAKH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", re.IGNORECASE)
K_HAZAAR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:k|hazaar|hazar|thousand)\b", re.IGNORECASE)
DIGITS_PATTERN = re.compile(r"\b(\d{1,3}(?:,\d{3})+|\d+)\b")


def parse_inr_budget_expression(raw: str | int | float | None) -> int | None:
    """Normalize a textual or numeric Indian currency expression into an integer INR amount.

    Supports:
        - 80k / 80 K -> 80000
        - 1.2 lakh / 1.2L -> 120000
        - 70 hazaar / 70 hazar -> 70000
        - ₹80,000 / Rs 80,000 -> 80000
        - Plain integers or floats -> rounded non-negative int
    """
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        if raw < 0:
            raise ValueError(f"Budget cannot be negative: {raw}")
        return int(round(raw))

    text = str(raw).strip()
    if not text:
        return None

    # Strip non-currency spec quantities (e.g. 16GB, 256GB, 1TB, 108MP, 5000mAh, 144Hz, 1.5kg)
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb|gigs?|mp|mah|hz|inch|inches|kg|g)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Clean currency symbols and prefixes
    cleaned = re.sub(
        r"[₹,]|(?:\b(?:rs|inr|rupees|under|below|around|approx|upto|up to|max|budget|to|increase)\b)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip()

    # 1. Check for Lakh (1,00,000)
    match_lakh = LAKH_PATTERN.search(cleaned)
    if match_lakh:
        val = float(match_lakh.group(1))
        return int(round(val * 100_000))

    # 2. Check for Thousand / K / Hazaar (1,000)
    match_k = K_HAZAAR_PATTERN.search(cleaned)
    if match_k:
        val = float(match_k.group(1))
        return int(round(val * 1_000))

    # 3. Plain digits (e.g. 80000 or 80,000)
    match_num = DIGITS_PATTERN.search(cleaned.replace(" ", ""))
    if match_num:
        raw_num = match_num.group(1).replace(",", "")
        try:
            val = int(raw_num)
            return val if val >= 0 else None
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Category & Use Case Normalization
# ---------------------------------------------------------------------------

def normalize_category(raw_category: Any) -> Category | None:
    """Normalize raw category string or enum to canonical Category."""
    if raw_category is None:
        return None

    if isinstance(raw_category, Category):
        return raw_category

    s = str(raw_category).strip().lower()
    return CATEGORY_ALIASES.get(s, None)


def normalize_use_cases(raw_use_cases: list[str] | None) -> list[str] | None:
    """Map use-case strings to canonical catalogue vocabulary, deduplicating order."""
    if raw_use_cases is None:
        return None

    result: list[str] = []
    seen: set[str] = set()

    for item in raw_use_cases:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()
        canonical = USE_CASE_ALIASES.get(cleaned, cleaned)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result if result else None


# ---------------------------------------------------------------------------
# Preference Normalization
# ---------------------------------------------------------------------------

def normalize_preferences(raw_prefs: dict[str, Any] | None) -> dict[str, float] | None:
    """Normalize preference keys to canonical names and ensure weights are bounded in [0.0, 1.0]."""
    if raw_prefs is None:
        return None

    result: dict[str, float] = {}

    for key, weight in raw_prefs.items():
        if not isinstance(key, str):
            continue
        k_clean = key.strip().lower()
        canonical_key = PREFERENCE_ALIASES.get(k_clean, k_clean)

        # Enforce canonical preference domain
        if canonical_key not in RECOGNIZED_PREFERENCES:
            continue

        try:
            w_float = float(weight)
        except (ValueError, TypeError):
            continue

        # Clamp weight between 0.0 and 1.0
        w_clamped = max(0.0, min(w_float, 1.0))
        if w_clamped > 0.0:
            result[canonical_key] = round(w_clamped, 2)

    return result if result else None


# ---------------------------------------------------------------------------
# Removed Fields Normalization
# ---------------------------------------------------------------------------

def normalize_removed_fields(raw_removals: list[str] | None) -> list[str]:
    """Map removal descriptions/aliases into validated removable field names."""
    if not raw_removals:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for item in raw_removals:
        if not isinstance(item, str):
            continue
        s = item.strip().lower()
        canonical = REMOVAL_FIELD_ALIASES.get(s, s)
        if canonical in VALID_REMOVABLE_FIELDS and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result


# ---------------------------------------------------------------------------
# Main Normalization Function
# ---------------------------------------------------------------------------

def normalize(delta: RequirementsDelta) -> RequirementsDelta:
    """Perform deterministic, idempotent normalization on a RequirementsDelta instance.

    Transforms:
        - Category strings/aliases to canonical Category enum.
        - Use-case strings to canonical catalogue vocabulary.
        - INR budget amounts from expressions (80k -> 80000, 1.2L -> 120000).
        - Preference keys to canonical vocabulary and bounded [0.0, 1.0] weights.
        - Removals aliases to canonical removable field names.
        - Drops preferences that are marked for explicit removal.

    Args:
        delta: The raw RequirementsDelta to normalize.

    Returns:
        A new normalized, validated RequirementsDelta instance.
    """
    # 1. Normalize Category
    norm_category = normalize_category(delta.category)

    # 2. Normalize Use Cases
    norm_use_cases = normalize_use_cases(delta.use_cases)

    # 3. Normalize Budgets
    norm_budget_max = parse_inr_budget_expression(delta.budget_max_inr)
    norm_budget_min = parse_inr_budget_expression(delta.budget_min_inr)

    # 4. Normalize Removed Fields
    norm_removed_fields = normalize_removed_fields(delta.removed_fields)

    # 5. Normalize Preferences
    norm_preferences = normalize_preferences(delta.preferences)

    # If a preference key is in removed_fields, eliminate it from preferences
    if norm_preferences and norm_removed_fields:
        for removed in norm_removed_fields:
            if removed in norm_preferences:
                del norm_preferences[removed]
        if not norm_preferences:
            norm_preferences = None

    # 6. Normalize Hard Constraints
    norm_hard: HardConstraints | None = None
    if delta.hard is not None:
        exclude_brands = [
            b.strip()
            for b in delta.hard.exclude_brands
            if isinstance(b, str) and b.strip()
        ]
        norm_hard = HardConstraints(
            min_ram_gb=delta.hard.min_ram_gb,
            min_storage_gb=delta.hard.min_storage_gb,
            min_battery_hr=delta.hard.min_battery_hr,
            max_weight_kg=delta.hard.max_weight_kg,
            requires_gpu=delta.hard.requires_gpu,
            min_vram_gb=delta.hard.min_vram_gb,
            exclude_brands=exclude_brands,
            in_stock=delta.hard.in_stock,
            form_factor=delta.hard.form_factor,
            requires_wireless=delta.hard.requires_wireless,
            requires_anc=delta.hard.requires_anc,
            requires_stylus=delta.hard.requires_stylus,
            min_refresh_hz=delta.hard.min_refresh_hz,
            screen_size_in=delta.hard.screen_size_in,
            target_screen_in=delta.hard.target_screen_in,
            requires_eye_comfort=delta.hard.requires_eye_comfort,
            requires_flicker_free=delta.hard.requires_flicker_free,
            requires_verified_features=delta.hard.requires_verified_features,
        )

    # 7. Normalize ML Mode
    norm_ml_mode = delta.ml_mode.strip().lower() if delta.ml_mode else None

    # 8. Normalize Requested Result Count
    norm_requested_count = delta.requested_result_count
    if norm_requested_count is not None:
        try:
            norm_requested_count = max(1, min(20, int(norm_requested_count)))
        except (ValueError, TypeError):
            norm_requested_count = None

    return RequirementsDelta(
        category=norm_category,
        use_cases=norm_use_cases,
        budget_max_inr=norm_budget_max,
        budget_min_inr=norm_budget_min,
        hard=norm_hard,
        preferences=norm_preferences,
        ml_mode=norm_ml_mode,
        asked_slots=delta.asked_slots,
        requested_result_count=norm_requested_count,
        removed_fields=norm_removed_fields,
        is_off_topic=delta.is_off_topic,
    )
