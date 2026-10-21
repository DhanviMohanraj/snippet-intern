"""Deterministic multi-turn requirements merging and contradiction resolution.

Provides pure Python state merging of user requirements deltas and automated detection
of logical contradictions (e.g. min budget exceeding max budget, conflicting GPU requirements).
No network or LLM operations are performed.
"""

from __future__ import annotations

from typing import Any

from app.models.product import Category
from app.models.requirements import (
    Contradiction,
    HardConstraints,
    Requirements,
    RequirementsDelta,
)

# Set of all known removal aliases for max budget
MAX_BUDGET_REMOVAL_KEYS = {
    "budget",
    "budget_max",
    "budget_max_inr",
    "budget_limit",
    "max_budget",
    "price",
    "price_limit",
}

# Set of all known removal aliases for min budget
MIN_BUDGET_REMOVAL_KEYS = {
    "budget_min",
    "budget_min_inr",
    "min_budget",
}

# Map of hard constraint removal aliases to HardConstraints attribute names
HARD_CONSTRAINT_REMOVAL_MAP = {
    "min_ram_gb": "min_ram_gb",
    "ram": "min_ram_gb",
    "min_storage_gb": "min_storage_gb",
    "storage": "min_storage_gb",
    "min_battery_hr": "min_battery_hr",
    "battery_hr": "min_battery_hr",
    "max_weight_kg": "max_weight_kg",
    "max_weight": "max_weight_kg",
    "requires_gpu": "requires_gpu",
    "gpu": "requires_gpu",
    "min_vram_gb": "min_vram_gb",
    "vram": "min_vram_gb",
    "exclude_brands": "exclude_brands",
    "in_stock": "in_stock",
    "form_factor": "form_factor",
    "wireless": "requires_wireless",
    "requires_wireless": "requires_wireless",
    "anc": "requires_anc",
    "requires_anc": "requires_anc",
    "stylus": "requires_stylus",
    "requires_stylus": "requires_stylus",
    "refresh_rate": "min_refresh_hz",
    "min_refresh_hz": "min_refresh_hz",
    "screen_size": "screen_size_in",
    "screen_size_in": "screen_size_in",
    "target_screen_in": "target_screen_in",
    "eye_comfort": "requires_eye_comfort",
    "requires_eye_comfort": "requires_eye_comfort",
    "flicker_free": "requires_flicker_free",
    "requires_flicker_free": "requires_flicker_free",
    "requires_verified_features": "requires_verified_features",
}


def detect_contradictions(req: Requirements) -> list[Contradiction]:
    """Deterministically detect logical contradictions and incompatible states in Requirements.

    Checks:
        1. Budget range contradiction: min_budget > max_budget.
        2. Incompatible GPU specifications: requires_gpu=False with min_vram_gb > 0.
        3. Local ML conflict: ml_mode='local' with requires_gpu=False.
        4. Category specification incompatibilities:
           - Headphones with GPU, RAM, or storage requirements.
           - Monitors with GPU, RAM, storage, or battery requirements.

    Args:
        req: The accumulated Requirements instance to inspect.

    Returns:
        List of Contradiction objects. Empty if no contradictions exist.
    """
    contradictions: list[Contradiction] = []

    # 1. Budget range check
    if req.budget_min_inr is not None and req.budget_max_inr is not None:
        if req.budget_min_inr > req.budget_max_inr:
            contradictions.append(
                Contradiction(
                    code="BUDGET_MIN_EXCEEDS_MAX",
                    fields=["budget_min_inr", "budget_max_inr"],
                    message=(
                        f"Minimum budget (₹{req.budget_min_inr:,}) exceeds "
                        f"maximum budget (₹{req.budget_max_inr:,})."
                    ),
                    severity="error",
                    requires_clarification=True,
                )
            )

    # 2. Incompatible GPU specifications
    if req.hard.requires_gpu is False and req.hard.min_vram_gb is not None and req.hard.min_vram_gb > 0:
        contradictions.append(
            Contradiction(
                code="GPU_FALSE_WITH_MIN_VRAM",
                fields=["requires_gpu", "min_vram_gb"],
                message=(
                    f"Contradictory GPU requirement: 'requires_gpu' is set to False, "
                    f"but 'min_vram_gb' is set to {req.hard.min_vram_gb}GB."
                ),
                severity="error",
                requires_clarification=True,
            )
        )

    # 3. Local ML conflicting with disabled GPU
    if req.ml_mode == "local" and req.hard.requires_gpu is False:
        contradictions.append(
            Contradiction(
                code="ML_LOCAL_CONFLICTS_NO_GPU",
                fields=["ml_mode", "requires_gpu"],
                message=(
                    "Local machine learning execution requires dedicated GPU acceleration, "
                    "but 'requires_gpu' is set to False."
                ),
                severity="error",
                requires_clarification=True,
            )
        )

    # 4. Category-specific specification incompatibilities
    if req.category == Category.HEADPHONES:
        if req.hard.requires_gpu is True or (req.hard.min_vram_gb is not None and req.hard.min_vram_gb > 0):
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "requires_gpu"],
                    message="Product category 'headphones' cannot have a dedicated GPU requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )
        if req.hard.min_ram_gb is not None:
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "min_ram_gb"],
                    message="Product category 'headphones' cannot have a minimum RAM requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )
        if req.hard.min_storage_gb is not None:
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "min_storage_gb"],
                    message="Product category 'headphones' cannot have a minimum storage requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )

    elif req.category == Category.MONITOR:
        if req.hard.requires_gpu is True or (req.hard.min_vram_gb is not None and req.hard.min_vram_gb > 0):
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "requires_gpu"],
                    message="Product category 'monitor' cannot have a dedicated GPU requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )
        if req.hard.min_ram_gb is not None:
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "min_ram_gb"],
                    message="Product category 'monitor' cannot have a minimum RAM requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )
        if req.hard.min_storage_gb is not None:
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "min_storage_gb"],
                    message="Product category 'monitor' cannot have a minimum storage requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )
        if req.hard.min_battery_hr is not None:
            contradictions.append(
                Contradiction(
                    code="CATEGORY_SPEC_INCOMPATIBLE",
                    fields=["category", "min_battery_hr"],
                    message="Product category 'monitor' cannot have a battery runtime requirement.",
                    severity="error",
                    requires_clarification=True,
                )
            )

    return contradictions


def merge(
    current: Requirements,
    delta: RequirementsDelta,
) -> tuple[Requirements, list[Contradiction]]:
    """Apply a RequirementsDelta to current Requirements and detect contradictions.

    Adheres to the following merging rules:
        - Absence/None in delta preserves the existing value.
        - Concrete value in delta replaces previous value (updates are not conflicts).
        - List-valued fields (use_cases, exclude_brands) append without duplicate entries.
        - Preferences merge weights without discarding unrelated preference keys.
        - Hard constraints preserve unset attributes.
        - Explicit removals in removed_fields clear corresponding fields or preferences.
        - Never mutates input arguments.
        - Off-topic deltas preserve current requirements without modifications.

    Args:
        current: The accumulated session Requirements.
        delta: The new turn RequirementsDelta to apply.

    Returns:
        A tuple of (merged Requirements, list of Contradictions).
    """
    # 0. Off-topic short-circuit: do not alter state for off-topic queries
    if delta.is_off_topic:
        return current.model_copy(deep=True), []

    # Prepare safe set of normalized removal keys
    removed_set = {r.strip().lower() for r in delta.removed_fields}

    # 1. Category merging
    if "category" in removed_set:
        new_category = None
    elif delta.category is not None:
        new_category = delta.category
    else:
        new_category = current.category

    # 2. Use cases merging
    if "use_cases" in removed_set:
        new_use_cases: list[str] = []
    else:
        new_use_cases = list(current.use_cases)
        if delta.use_cases is not None:
            for uc in delta.use_cases:
                uc_clean = uc.strip().lower()
                if uc_clean and uc_clean not in new_use_cases:
                    new_use_cases.append(uc_clean)
        # Check if individual use cases are targeted for removal
        new_use_cases = [uc for uc in new_use_cases if uc not in removed_set]

    # 3. Budget merging
    if any(k in removed_set for k in MAX_BUDGET_REMOVAL_KEYS):
        new_budget_max = None
    elif delta.budget_max_inr is not None:
        new_budget_max = delta.budget_max_inr
    else:
        new_budget_max = current.budget_max_inr

    if any(k in removed_set for k in MIN_BUDGET_REMOVAL_KEYS):
        new_budget_min = None
    elif delta.budget_min_inr is not None:
        new_budget_min = delta.budget_min_inr
    else:
        new_budget_min = current.budget_min_inr

    # 4. ML Mode merging
    if "ml_mode" in removed_set:
        new_ml_mode = None
    elif delta.ml_mode is not None:
        new_ml_mode = delta.ml_mode
    else:
        new_ml_mode = current.ml_mode

    # 5. Hard constraints merging
    if "hard" in removed_set:
        new_hard_dict = HardConstraints().model_dump()
    else:
        new_hard_dict = current.hard.model_dump()

        # Handle explicit removal of individual hard constraint fields
        for r_key, attr_name in HARD_CONSTRAINT_REMOVAL_MAP.items():
            if r_key in removed_set:
                if attr_name == "exclude_brands":
                    new_hard_dict["exclude_brands"] = []
                else:
                    new_hard_dict[attr_name] = None

        # Apply new delta hard constraint values if present
        if delta.hard is not None:
            if delta.hard.min_ram_gb is not None:
                new_hard_dict["min_ram_gb"] = delta.hard.min_ram_gb
            if delta.hard.min_storage_gb is not None:
                new_hard_dict["min_storage_gb"] = delta.hard.min_storage_gb
            if delta.hard.min_battery_hr is not None:
                new_hard_dict["min_battery_hr"] = delta.hard.min_battery_hr
            if delta.hard.max_weight_kg is not None:
                new_hard_dict["max_weight_kg"] = delta.hard.max_weight_kg
            if delta.hard.requires_gpu is not None:
                new_hard_dict["requires_gpu"] = delta.hard.requires_gpu
            if delta.hard.min_vram_gb is not None:
                new_hard_dict["min_vram_gb"] = delta.hard.min_vram_gb
            if delta.hard.in_stock is not None:
                new_hard_dict["in_stock"] = delta.hard.in_stock
            if delta.hard.form_factor is not None:
                new_hard_dict["form_factor"] = delta.hard.form_factor
            if delta.hard.requires_wireless is not None:
                new_hard_dict["requires_wireless"] = delta.hard.requires_wireless
            if delta.hard.requires_anc is not None:
                new_hard_dict["requires_anc"] = delta.hard.requires_anc
            if delta.hard.requires_stylus is not None:
                new_hard_dict["requires_stylus"] = delta.hard.requires_stylus
            if delta.hard.min_refresh_hz is not None:
                new_hard_dict["min_refresh_hz"] = delta.hard.min_refresh_hz
            if delta.hard.screen_size_in is not None:
                new_hard_dict["screen_size_in"] = delta.hard.screen_size_in
            if delta.hard.target_screen_in is not None:
                new_hard_dict["target_screen_in"] = delta.hard.target_screen_in
            if delta.hard.requires_eye_comfort is not None:
                new_hard_dict["requires_eye_comfort"] = delta.hard.requires_eye_comfort
            if delta.hard.requires_flicker_free is not None:
                new_hard_dict["requires_flicker_free"] = delta.hard.requires_flicker_free
            if delta.hard.requires_verified_features:
                existing_feats = list(new_hard_dict.get("requires_verified_features", []))
                for f in delta.hard.requires_verified_features:
                    f_clean = f.strip().lower()
                    if f_clean and f_clean not in existing_feats:
                        existing_feats.append(f_clean)
                new_hard_dict["requires_verified_features"] = existing_feats

            if delta.hard.exclude_brands:
                existing_brands = list(new_hard_dict.get("exclude_brands", []))
                for brand in delta.hard.exclude_brands:
                    b_clean = brand.strip()
                    if b_clean and b_clean not in existing_brands:
                        existing_brands.append(b_clean)
                new_hard_dict["exclude_brands"] = existing_brands

    new_hard = HardConstraints(**new_hard_dict)

    # 6. Preferences merging
    if "preferences" in removed_set:
        new_preferences: dict[str, float] = {}
    else:
        new_preferences = dict(current.preferences)

        # Remove specific preferences present in removed_fields
        for r_field in removed_set:
            if r_field in new_preferences:
                del new_preferences[r_field]

        # Merge delta preferences
        if delta.preferences is not None:
            for pref_key, pref_weight in delta.preferences.items():
                p_clean = pref_key.strip().lower()
                if 0.0 <= pref_weight <= 1.0:
                    new_preferences[p_clean] = pref_weight

    # 7. Asked slots merging
    new_asked_slots = list(current.asked_slots)
    if delta.asked_slots is not None:
        for slot in delta.asked_slots:
            s_clean = slot.strip().lower()
            if s_clean and s_clean not in new_asked_slots:
                new_asked_slots.append(s_clean)

    # 8. Requested result count merging
    if "requested_result_count" in removed_set or "count" in removed_set:
        new_requested_count = None
    elif delta.requested_result_count is not None:
        new_requested_count = delta.requested_result_count
    else:
        new_requested_count = current.requested_result_count

    # 9. Build new Requirements object
    merged = Requirements(
        category=new_category,
        use_cases=new_use_cases,
        budget_max_inr=new_budget_max,
        budget_min_inr=new_budget_min,
        hard=new_hard,
        preferences=new_preferences,
        ml_mode=new_ml_mode,
        asked_slots=new_asked_slots,
        requested_result_count=new_requested_count,
    )

    # 9. Deterministic contradiction detection on merged state
    contradictions = detect_contradictions(merged)

    return merged, contradictions
