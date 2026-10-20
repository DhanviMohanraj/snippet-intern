"""Deterministic product candidate filtering, multi-factor scoring, and ranking engine.

Implements Module 8:
    1. Hard-constraint filtering: Evaluates budget, RAM, storage, battery, weight, GPU, VRAM,
       brand exclusions, and availability against actual catalogue product records.
       Missing specifications are treated as UNKNOWN, never as proof of compliance.
    2. Preference-based scoring: Maps normalized user soft preferences (portability, performance,
       battery life, display, camera, GPU power, noise cancellation, build quality) to 0-100 scores.
    3. Deterministic candidate ranking: Sorts eligible candidates by score descending, price ascending,
       and product ID ascending.
    4. Structured search summary: Provides complete visibility into passed, failed, and unverifiable
       candidates with human-readable rationale.

Design Principle:
    The LLM interprets user intent; deterministic Python code filters and ranks actual catalogue records.
    No LLM or network operations are performed.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from app.agent.retrieval import retrieve_candidates
from app.core.logging import get_logger
from app.db.repositories import ProductRepository
from app.models.product import Category, Product
from app.models.ranking import (
    CandidateEvaluation,
    CandidateSearchResult,
    ConstraintCheck,
    ConstraintStatus,
    RankedCandidate,
    ScoreBreakdown,
)
from app.models.requirements import Requirements

logger = get_logger("app.agent.ranking")

# Discrete GPU keywords identifying dedicated graphics in laptop specs
DISCRETE_GPU_KEYWORDS: tuple[str, ...] = ("rtx", "gtx", "radeon rx", "arc a")

# High-tier CPU keywords for performance scoring bonus
FLAGSHIP_CPU_KEYWORDS: tuple[str, ...] = (
    "i9",
    "ryzen 9",
    "m3 max",
    "m2 max",
    "m3 pro",
    "snapdragon 8 gen 3",
    "a17 pro",
)
MID_TIER_CPU_KEYWORDS: tuple[str, ...] = (
    "i7",
    "ryzen 7",
    "m2",
    "m3",
    "snapdragon 8 gen 2",
    "a16",
    "dimensity 9200",
)


# ---------------------------------------------------------------------------
# 1. Hard-Constraint Evaluation
# ---------------------------------------------------------------------------

def evaluate_candidate(product: Product, requirements: Requirements) -> CandidateEvaluation:
    """Evaluate all applicable hard constraints for a single product record.

    Adheres strictly to the three-way outcome convention:
        - PASS: The product demonstrably satisfies the constraint.
        - FAIL: The product demonstrably violates the constraint.
        - UNKNOWN: The product record lacks the specification needed to verify compliance.
                   A product with any UNKNOWN required constraint is NOT eligible.

    Args:
        product: The product catalogue record to evaluate.
        requirements: The user's accumulated Requirements state.

    Returns:
        CandidateEvaluation detailing every evaluated constraint check and eligibility status.
    """
    passed: list[ConstraintCheck] = []
    failed: list[ConstraintCheck] = []
    unverifiable: list[ConstraintCheck] = []
    rejection_reasons: list[str] = []

    # 1. Category check
    if requirements.category is not None:
        if product.category == requirements.category:
            passed.append(
                ConstraintCheck(
                    name="category",
                    status=ConstraintStatus.PASS,
                    required=requirements.category.value,
                    actual=product.category.value,
                    reason=f"Category matches '{product.category.value}'.",
                )
            )
        else:
            chk = ConstraintCheck(
                name="category",
                status=ConstraintStatus.FAIL,
                required=requirements.category.value,
                actual=product.category.value,
                reason=f"Product category '{product.category.value}' does not match requested '{requirements.category.value}'.",
            )
            failed.append(chk)
            rejection_reasons.append(chk.reason)

    # 2. Maximum Budget check
    if requirements.budget_max_inr is not None:
        if product.price_inr <= requirements.budget_max_inr:
            passed.append(
                ConstraintCheck(
                    name="budget_max",
                    status=ConstraintStatus.PASS,
                    required=requirements.budget_max_inr,
                    actual=product.price_inr,
                    reason=f"Price ₹{product.price_inr:,} is within maximum budget ₹{requirements.budget_max_inr:,}.",
                )
            )
        else:
            chk = ConstraintCheck(
                name="budget_max",
                status=ConstraintStatus.FAIL,
                required=requirements.budget_max_inr,
                actual=product.price_inr,
                reason=f"Price ₹{product.price_inr:,} exceeds maximum budget of ₹{requirements.budget_max_inr:,}.",
            )
            failed.append(chk)
            rejection_reasons.append(chk.reason)

    # 3. Minimum Budget check
    if requirements.budget_min_inr is not None:
        if product.price_inr >= requirements.budget_min_inr:
            passed.append(
                ConstraintCheck(
                    name="budget_min",
                    status=ConstraintStatus.PASS,
                    required=requirements.budget_min_inr,
                    actual=product.price_inr,
                    reason=f"Price ₹{product.price_inr:,} meets minimum budget ₹{requirements.budget_min_inr:,}.",
                )
            )
        else:
            chk = ConstraintCheck(
                name="budget_min",
                status=ConstraintStatus.FAIL,
                required=requirements.budget_min_inr,
                actual=product.price_inr,
                reason=f"Price ₹{product.price_inr:,} is below minimum budget of ₹{requirements.budget_min_inr:,}.",
            )
            failed.append(chk)
            rejection_reasons.append(chk.reason)

    # 4. Brand Exclusions check
    if requirements.hard.exclude_brands:
        excluded_brands_lower = {b.strip().lower() for b in requirements.hard.exclude_brands if b.strip()}
        if product.brand.strip().lower() in excluded_brands_lower:
            chk = ConstraintCheck(
                name="exclude_brands",
                status=ConstraintStatus.FAIL,
                required=requirements.hard.exclude_brands,
                actual=product.brand,
                reason=f"Brand '{product.brand}' is in user's excluded brands list.",
            )
            failed.append(chk)
            rejection_reasons.append(chk.reason)
        else:
            passed.append(
                ConstraintCheck(
                    name="exclude_brands",
                    status=ConstraintStatus.PASS,
                    required=requirements.hard.exclude_brands,
                    actual=product.brand,
                    reason=f"Brand '{product.brand}' is not excluded.",
                )
            )

    # 5. In-Stock requirement check
    if requirements.hard.in_stock is not None:
        if requirements.hard.in_stock is True:
            if product.in_stock is True:
                passed.append(
                    ConstraintCheck(
                        name="in_stock",
                        status=ConstraintStatus.PASS,
                        required=True,
                        actual=product.in_stock,
                        reason="Product is confirmed in stock.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="in_stock",
                    status=ConstraintStatus.FAIL,
                    required=True,
                    actual=product.in_stock,
                    reason="Product is currently out of stock.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)

    # 6. Minimum RAM requirement check
    if requirements.hard.min_ram_gb is not None:
        if product.category in (Category.LAPTOP, Category.PHONE, Category.TABLET):
            ram = product.specs.get("ram_gb")
            if ram is None:
                chk = ConstraintCheck(
                    name="min_ram_gb",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.min_ram_gb,
                    actual=None,
                    reason="Product record lacks 'ram_gb' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif ram >= requirements.hard.min_ram_gb:
                passed.append(
                    ConstraintCheck(
                        name="min_ram_gb",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.min_ram_gb,
                        actual=ram,
                        reason=f"RAM {ram}GB satisfies minimum requirement of {requirements.hard.min_ram_gb}GB.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="min_ram_gb",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.min_ram_gb,
                    actual=ram,
                    reason=f"RAM {ram}GB fails minimum requirement of {requirements.hard.min_ram_gb}GB.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="min_ram_gb",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.min_ram_gb,
                actual=None,
                reason=f"Product category '{product.category.value}' does not support RAM specifications.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 7. Minimum Storage requirement check
    if requirements.hard.min_storage_gb is not None:
        if product.category in (Category.LAPTOP, Category.PHONE, Category.TABLET):
            storage = product.specs.get("storage_gb")
            if storage is None:
                chk = ConstraintCheck(
                    name="min_storage_gb",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.min_storage_gb,
                    actual=None,
                    reason="Product record lacks 'storage_gb' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif storage >= requirements.hard.min_storage_gb:
                passed.append(
                    ConstraintCheck(
                        name="min_storage_gb",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.min_storage_gb,
                        actual=storage,
                        reason=f"Storage {storage}GB satisfies minimum requirement of {requirements.hard.min_storage_gb}GB.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="min_storage_gb",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.min_storage_gb,
                    actual=storage,
                    reason=f"Storage {storage}GB fails minimum requirement of {requirements.hard.min_storage_gb}GB.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="min_storage_gb",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.min_storage_gb,
                actual=None,
                reason=f"Product category '{product.category.value}' does not support storage specifications.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 8. Minimum Battery Runtime check
    if requirements.hard.min_battery_hr is not None:
        if product.category in (Category.LAPTOP, Category.HEADPHONES, Category.TABLET):
            battery = product.specs.get("battery_hr")
            if battery is None:
                chk = ConstraintCheck(
                    name="min_battery_hr",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.min_battery_hr,
                    actual=None,
                    reason="Product record lacks 'battery_hr' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif battery >= requirements.hard.min_battery_hr:
                passed.append(
                    ConstraintCheck(
                        name="min_battery_hr",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.min_battery_hr,
                        actual=battery,
                        reason=f"Battery runtime {battery}hr satisfies minimum requirement of {requirements.hard.min_battery_hr}hr.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="min_battery_hr",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.min_battery_hr,
                    actual=battery,
                    reason=f"Battery runtime {battery}hr fails minimum requirement of {requirements.hard.min_battery_hr}hr.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="min_battery_hr",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.min_battery_hr,
                actual=None,
                reason=f"Category '{product.category.value}' specifications do not track battery runtime in hours.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 9. Maximum Weight check
    if requirements.hard.max_weight_kg is not None:
        if product.category == Category.LAPTOP:
            w_kg = product.specs.get("weight_kg")
            if w_kg is None:
                chk = ConstraintCheck(
                    name="max_weight_kg",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.max_weight_kg,
                    actual=None,
                    reason="Product record lacks 'weight_kg' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif w_kg <= requirements.hard.max_weight_kg:
                passed.append(
                    ConstraintCheck(
                        name="max_weight_kg",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.max_weight_kg,
                        actual=w_kg,
                        reason=f"Weight {w_kg}kg satisfies maximum limit of {requirements.hard.max_weight_kg}kg.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="max_weight_kg",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.max_weight_kg,
                    actual=w_kg,
                    reason=f"Weight {w_kg}kg exceeds maximum limit of {requirements.hard.max_weight_kg}kg.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        elif product.category in (Category.PHONE, Category.HEADPHONES, Category.TABLET):
            w_g = product.specs.get("weight_g")
            if w_g is None:
                chk = ConstraintCheck(
                    name="max_weight_kg",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.max_weight_kg,
                    actual=None,
                    reason="Product record lacks 'weight_g' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            else:
                w_kg = round(w_g / 1000.0, 3)
                if w_kg <= requirements.hard.max_weight_kg:
                    passed.append(
                        ConstraintCheck(
                            name="max_weight_kg",
                            status=ConstraintStatus.PASS,
                            required=requirements.hard.max_weight_kg,
                            actual=w_kg,
                            reason=f"Weight {w_kg}kg ({w_g}g) satisfies maximum limit of {requirements.hard.max_weight_kg}kg.",
                        )
                    )
                else:
                    chk = ConstraintCheck(
                        name="max_weight_kg",
                        status=ConstraintStatus.FAIL,
                        required=requirements.hard.max_weight_kg,
                        actual=w_kg,
                        reason=f"Weight {w_kg}kg ({w_g}g) exceeds maximum limit of {requirements.hard.max_weight_kg}kg.",
                    )
                    failed.append(chk)
                    rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="max_weight_kg",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.max_weight_kg,
                actual=None,
                reason=f"Category '{product.category.value}' does not track weight specifications.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 10. Dedicated GPU check
    if requirements.hard.requires_gpu is not None:
        if product.category == Category.LAPTOP:
            gpu_str = product.specs.get("gpu")
            vram = product.specs.get("vram_gb")
            if gpu_str is None:
                chk = ConstraintCheck(
                    name="requires_gpu",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.requires_gpu,
                    actual=None,
                    reason="Product record lacks 'gpu' specification; cannot verify dedicated GPU presence.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            else:
                gpu_lower = gpu_str.lower()
                has_discrete_gpu = (vram is not None and vram > 0) or any(
                    kw in gpu_lower for kw in DISCRETE_GPU_KEYWORDS
                )
                if requirements.hard.requires_gpu is True:
                    if has_discrete_gpu:
                        passed.append(
                            ConstraintCheck(
                                name="requires_gpu",
                                status=ConstraintStatus.PASS,
                                required=True,
                                actual=gpu_str,
                                reason=f"Dedicated GPU verified: '{gpu_str}' with {vram or 0}GB VRAM.",
                            )
                        )
                    else:
                        chk = ConstraintCheck(
                            name="requires_gpu",
                            status=ConstraintStatus.FAIL,
                            required=True,
                            actual=gpu_str,
                            reason=f"Product has integrated graphics ('{gpu_str}'), but dedicated GPU is required.",
                        )
                        failed.append(chk)
                        rejection_reasons.append(chk.reason)
                else:  # requires_gpu is False
                    if not has_discrete_gpu:
                        passed.append(
                            ConstraintCheck(
                                name="requires_gpu",
                                status=ConstraintStatus.PASS,
                                required=False,
                                actual=gpu_str,
                                reason=f"Integrated graphics verified: '{gpu_str}'.",
                            )
                        )
                    else:
                        chk = ConstraintCheck(
                            name="requires_gpu",
                            status=ConstraintStatus.FAIL,
                            required=False,
                            actual=gpu_str,
                            reason=f"Product has a dedicated GPU ('{gpu_str}'), but integrated graphics was required.",
                        )
                        failed.append(chk)
                        rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="requires_gpu",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.requires_gpu,
                actual=None,
                reason=f"Category '{product.category.value}' does not support dedicated GPU selection.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 11. Minimum VRAM check
    if requirements.hard.min_vram_gb is not None:
        if product.category == Category.LAPTOP:
            vram = product.specs.get("vram_gb")
            if vram is None:
                chk = ConstraintCheck(
                    name="min_vram_gb",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.min_vram_gb,
                    actual=None,
                    reason="Product record lacks 'vram_gb' specification; cannot verify compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif vram >= requirements.hard.min_vram_gb:
                passed.append(
                    ConstraintCheck(
                        name="min_vram_gb",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.min_vram_gb,
                        actual=vram,
                        reason=f"VRAM {vram}GB satisfies minimum requirement of {requirements.hard.min_vram_gb}GB.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="min_vram_gb",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.min_vram_gb,
                    actual=vram,
                    reason=f"VRAM {vram}GB fails minimum requirement of {requirements.hard.min_vram_gb}GB.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="min_vram_gb",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.min_vram_gb,
                actual=None,
                reason=f"Category '{product.category.value}' does not track dedicated VRAM specifications.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 12. Form Factor check (Headphones)
    if requirements.hard.form_factor is not None:
        requested_ff = requirements.hard.form_factor.strip().lower()
        if product.category == Category.HEADPHONES:
            actual_type = str(product.specs.get("type", "")).strip().lower()
            if not actual_type:
                chk = ConstraintCheck(
                    name="form_factor",
                    status=ConstraintStatus.UNKNOWN,
                    required=requested_ff,
                    actual=None,
                    reason="Headphone record lacks 'type' specification; cannot verify form factor compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif actual_type == requested_ff:
                passed.append(
                    ConstraintCheck(
                        name="form_factor",
                        status=ConstraintStatus.PASS,
                        required=requested_ff,
                        actual=actual_type,
                        reason=f"Headphone form factor '{actual_type}' matches requested '{requested_ff}'.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="form_factor",
                    status=ConstraintStatus.FAIL,
                    required=requested_ff,
                    actual=actual_type,
                    reason=f"Headphone type '{actual_type}' does not match requested '{requested_ff}'.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="form_factor",
                status=ConstraintStatus.UNKNOWN,
                required=requested_ff,
                actual=None,
                reason=f"Category '{product.category.value}' does not support form factor constraints.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 13. Wireless Connectivity check (Headphones)
    if requirements.hard.requires_wireless is not None:
        if product.category == Category.HEADPHONES:
            actual_wireless = product.specs.get("wireless")
            if actual_wireless is None:
                chk = ConstraintCheck(
                    name="requires_wireless",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.requires_wireless,
                    actual=None,
                    reason="Headphone record lacks 'wireless' specification; cannot verify wireless compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif actual_wireless == requirements.hard.requires_wireless:
                passed.append(
                    ConstraintCheck(
                        name="requires_wireless",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.requires_wireless,
                        actual=actual_wireless,
                        reason=f"Wireless status ({actual_wireless}) satisfies requirement.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="requires_wireless",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.requires_wireless,
                    actual=actual_wireless,
                    reason=f"Product wireless={actual_wireless} does not match required {requirements.hard.requires_wireless}.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)

    # 14. Active Noise Cancellation (ANC) check (Headphones)
    if requirements.hard.requires_anc is True:
        if product.category == Category.HEADPHONES:
            actual_anc = product.specs.get("anc")
            if actual_anc is None:
                chk = ConstraintCheck(
                    name="requires_anc",
                    status=ConstraintStatus.UNKNOWN,
                    required=True,
                    actual=None,
                    reason="Headphone record lacks 'anc' specification; cannot verify ANC compliance.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif actual_anc is True:
                passed.append(
                    ConstraintCheck(
                        name="requires_anc",
                        status=ConstraintStatus.PASS,
                        required=True,
                        actual=True,
                        reason="Active Noise Cancellation (ANC) verified.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="requires_anc",
                    status=ConstraintStatus.FAIL,
                    required=True,
                    actual=actual_anc,
                    reason=f"Active Noise Cancellation (ANC) required, but product has anc={actual_anc}.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)

    # 15. Stylus Input Support check (Tablets)
    if requirements.hard.requires_stylus is not None:
        if product.category == Category.TABLET:
            actual_stylus = product.specs.get("stylus_support")
            if actual_stylus is None:
                chk = ConstraintCheck(
                    name="requires_stylus",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.requires_stylus,
                    actual=None,
                    reason="Tablet record lacks 'stylus_support' specification; cannot verify stylus compatibility.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif actual_stylus == requirements.hard.requires_stylus:
                passed.append(
                    ConstraintCheck(
                        name="requires_stylus",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.requires_stylus,
                        actual=actual_stylus,
                        reason=f"Stylus support ({actual_stylus}) matches requirement.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="requires_stylus",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.requires_stylus,
                    actual=actual_stylus,
                    reason=f"Tablet stylus support={actual_stylus} does not match required {requirements.hard.requires_stylus}.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="requires_stylus",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.requires_stylus,
                actual=None,
                reason=f"Category '{product.category.value}' does not support stylus input constraints.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # 16. Minimum Refresh Rate check (Monitors)
    if requirements.hard.min_refresh_hz is not None:
        if product.category == Category.MONITOR:
            actual_hz = product.specs.get("refresh_hz")
            if actual_hz is None:
                chk = ConstraintCheck(
                    name="min_refresh_hz",
                    status=ConstraintStatus.UNKNOWN,
                    required=requirements.hard.min_refresh_hz,
                    actual=None,
                    reason="Monitor record lacks 'refresh_hz' specification; cannot verify refresh rate.",
                )
                unverifiable.append(chk)
                rejection_reasons.append(chk.reason)
            elif actual_hz >= requirements.hard.min_refresh_hz:
                passed.append(
                    ConstraintCheck(
                        name="min_refresh_hz",
                        status=ConstraintStatus.PASS,
                        required=requirements.hard.min_refresh_hz,
                        actual=actual_hz,
                        reason=f"Refresh rate {actual_hz}Hz satisfies minimum requirement of {requirements.hard.min_refresh_hz}Hz.",
                    )
                )
            else:
                chk = ConstraintCheck(
                    name="min_refresh_hz",
                    status=ConstraintStatus.FAIL,
                    required=requirements.hard.min_refresh_hz,
                    actual=actual_hz,
                    reason=f"Refresh rate {actual_hz}Hz fails minimum requirement of {requirements.hard.min_refresh_hz}Hz.",
                )
                failed.append(chk)
                rejection_reasons.append(chk.reason)
        else:
            chk = ConstraintCheck(
                name="min_refresh_hz",
                status=ConstraintStatus.UNKNOWN,
                required=requirements.hard.min_refresh_hz,
                actual=None,
                reason=f"Category '{product.category.value}' specifications do not track refresh rate in Hz.",
            )
            unverifiable.append(chk)
            rejection_reasons.append(chk.reason)

    # Critical Rule: Product is eligible ONLY if all checks passed, with zero failures and zero unknowns
    is_eligible = (len(failed) == 0 and len(unverifiable) == 0)

    return CandidateEvaluation(
        product_id=product.product_id,
        product=product,
        is_eligible=is_eligible,
        passed_constraints=passed,
        failed_constraints=failed,
        unverifiable_constraints=unverifiable,
        rejection_reasons=rejection_reasons,
    )


def filter_candidates(
    products: Sequence[Product],
    requirements: Requirements,
) -> tuple[list[Product], list[CandidateEvaluation]]:
    """Filter candidates into eligible matches and structured exclusion reports.

    Args:
        products: Sequence of candidate products to filter.
        requirements: Current user requirements defining hard constraints.

    Returns:
        A tuple of (eligible_products, excluded_candidate_evaluations).
    """
    eligible: list[Product] = []
    excluded: list[CandidateEvaluation] = []

    for product in products:
        eval_report = evaluate_candidate(product, requirements)
        if eval_report.is_eligible:
            eligible.append(product)
        else:
            excluded.append(eval_report)

    return eligible, excluded


# ---------------------------------------------------------------------------
# 2. Preference-Based Multi-Factor Scoring
# ---------------------------------------------------------------------------

def _score_portability(product: Product) -> tuple[float, str]:
    """Score portability attribute on a 0-100 scale based on product weight."""
    if product.category == Category.LAPTOP:
        w_kg = product.specs.get("weight_kg")
        if w_kg is not None:
            # 1.0kg -> 100, 1.4kg -> 80, 1.8kg -> 60, 2.2kg -> 40, >= 2.6kg -> 20
            score = max(15.0, min(100.0, 100.0 - (w_kg - 1.0) * 50.0))
            return round(score, 1), f"Weight of {w_kg}kg provides {score:.1f}/100 portability rating."
        return 50.0, "Weight specification unavailable; assigned neutral 50/100 score."

    if product.category in (Category.PHONE, Category.HEADPHONES, Category.TABLET):
        w_g = product.specs.get("weight_g")
        if w_g is not None:
            # Phones: 160g -> 96, 200g -> 80, 250g -> 60
            score = max(20.0, min(100.0, 100.0 - (w_g - 150) * 0.4))
            return round(score, 1), f"Weight of {w_g}g rated {score:.1f}/100 for portability."
        return 50.0, "Weight specification unavailable; assigned neutral 50/100 score."

    return 30.0, "Stationary product category provides limited portability."


def _score_performance(product: Product) -> tuple[float, str]:
    """Score performance on a 0-100 scale using RAM, processor tier, and storage."""
    ram = product.specs.get("ram_gb", 8)
    # Baseline RAM points
    if ram >= 64:
        ram_score = 95.0
    elif ram >= 32:
        ram_score = 88.0
    elif ram >= 16:
        ram_score = 75.0
    elif ram >= 12:
        ram_score = 65.0
    else:
        ram_score = 50.0

    # CPU / Chipset evaluation
    cpu = product.specs.get("cpu", "") or product.specs.get("chipset", "")
    cpu_lower = cpu.lower()
    bonus = 0.0
    if any(k in cpu_lower for k in FLAGSHIP_CPU_KEYWORDS):
        bonus = 10.0
    elif any(k in cpu_lower for k in MID_TIER_CPU_KEYWORDS):
        bonus = 5.0

    score = min(100.0, ram_score + bonus)
    explanation = f"{ram}GB RAM with '{cpu or 'standard processor'}' provides {score:.1f}/100 performance capability."
    return round(score, 1), explanation


def _score_battery_life(product: Product) -> tuple[float, str]:
    """Score battery life on a 0-100 scale using runtime hours or mAh capacity."""
    if product.category in (Category.LAPTOP, Category.HEADPHONES, Category.TABLET):
        battery_hr = product.specs.get("battery_hr")
        if battery_hr is not None:
            # 6hr -> 50, 10hr -> 75, 14hr -> 90, 18hr+ -> 100
            score = min(100.0, max(25.0, (battery_hr / 16.0) * 90.0 + 10.0))
            return round(score, 1), f"{battery_hr} hours battery runtime rated {score:.1f}/100."
        return 50.0, "Battery runtime specification unavailable; assigned neutral score."

    if product.category == Category.PHONE:
        mah = product.specs.get("battery_mah")
        if mah is not None:
            # 4000mAh -> 65, 5000mAh -> 85, 6000mAh -> 100
            score = min(100.0, max(30.0, ((mah - 3000) / 3000.0) * 70.0 + 30.0))
            return round(score, 1), f"{mah}mAh battery capacity rated {score:.1f}/100."
        return 50.0, "Battery capacity specification unavailable; assigned neutral score."

    return 40.0, "Category does not feature portable battery operation."


def _score_display_quality(product: Product) -> tuple[float, str]:
    """Score display quality using panel type, refresh rate, and resolution."""
    features_text = " ".join(product.features).lower()
    specs_text = " ".join(str(v) for v in product.specs.values()).lower()

    score = 65.0
    highlights: list[str] = []

    # Panel check
    panel_val = str(product.specs.get("panel", "")).lower()
    if "oled" in features_text or "oled" in specs_text or "amoled" in features_text or "oled" in panel_val:
        score = 90.0
        highlights.append("OLED panel")
    elif "ips" in features_text or "ips" in specs_text or "ips" in panel_val:
        score = 75.0
        highlights.append("IPS display")

    # Refresh rate check
    refresh_val = product.specs.get("refresh_hz")
    if (refresh_val and refresh_val >= 120) or any(h in features_text for h in ("120hz", "144hz", "165hz", "240hz")):
        score = min(100.0, score + 10.0)
        hz_tag = f"{refresh_val}Hz" if refresh_val else "high"
        highlights.append(f"{hz_tag} refresh rate")

    # Resolution check
    res_val = str(product.specs.get("resolution", "")).lower()
    if "3840" in res_val or "4k" in res_val or "4k" in features_text or "3.2k" in features_text or "retina" in features_text:
        score = min(100.0, score + 10.0)
        highlights.append("high resolution")
    elif "2560" in res_val or "1440" in res_val or "qhd" in features_text:
        score = min(100.0, score + 5.0)
        highlights.append("QHD resolution")

    tag = ", ".join(highlights) if highlights else "standard display"
    return round(score, 1), f"Display features ({tag}) rated {score:.1f}/100."


def _score_camera_quality(product: Product) -> tuple[float, str, bool]:
    """Score camera quality for phones. Returns (score, explanation, is_supported)."""
    if product.category == Category.PHONE:
        mp = product.specs.get("camera_mp")
        if mp is not None:
            # 12MP -> 55, 48MP -> 75, 108MP -> 90, 200MP -> 100
            score = min(100.0, max(35.0, (mp / 100.0) * 60.0 + 40.0))
            return round(score, 1), f"{mp}MP camera rated {score:.1f}/100.", True
        return 50.0, "Camera resolution specification unavailable.", True

    return 50.0, f"Category '{product.category.value}' does not support camera evaluation.", False


def _score_gpu_power(product: Product) -> tuple[float, str, bool]:
    """Score GPU capability for graphics and gaming. Returns (score, explanation, is_supported)."""
    if product.category == Category.LAPTOP:
        vram = product.specs.get("vram_gb", 0)
        gpu = product.specs.get("gpu", "").lower()
        if vram >= 16 or "4090" in gpu:
            score = 100.0
        elif vram >= 12 or "4080" in gpu:
            score = 92.0
        elif vram >= 8 or "4070" in gpu:
            score = 84.0
        elif vram >= 6 or "4060" in gpu:
            score = 75.0
        elif vram >= 4 or "3050" in gpu or "4050" in gpu:
            score = 65.0
        else:
            score = 25.0
        return score, f"Graphics configuration ({product.specs.get('gpu', 'integrated')}) rated {score:.1f}/100.", True

    return 50.0, f"Category '{product.category.value}' does not support dedicated GPU power evaluation.", False


def _score_noise_cancellation(product: Product) -> tuple[float, str, bool]:
    """Score active noise cancellation for headphones. Returns (score, explanation, is_supported)."""
    if product.category == Category.HEADPHONES:
        anc = product.specs.get("anc")
        if anc is True:
            return 95.0, "Active Noise Cancellation (ANC) verified.", True
        if anc is False:
            return 30.0, "Passive noise isolation only (no ANC).", True
        return 50.0, "ANC specification unavailable; assigned neutral score.", True

    return 50.0, f"Category '{product.category.value}' does not support noise cancellation evaluation.", False


def _score_microphone_quality(product: Product) -> tuple[float, str, bool]:
    """Score microphone clarity and call performance for headphones."""
    if product.category == Category.HEADPHONES:
        feat = " ".join(product.features).lower()
        if "enc" in feat or "boom mic" in feat or "dual mic" in feat or "quad mic" in feat:
            return 95.0, "Environmental Noise Cancellation (ENC) dual/boom microphones verified for clear voice calls.", True
        if "mic" in feat or any("call" in u.lower() for u in product.intended_use):
            return 82.0, "Integrated microphone verified for voice calls and online classes.", True
        return 65.0, "Standard microphone for calls.", True

    return 50.0, f"Category '{product.category.value}' does not support dedicated microphone evaluation.", False


def _score_comfort(product: Product) -> tuple[float, str, bool]:
    """Score physical comfort and extended-wear suitability for headphones."""
    if product.category == Category.HEADPHONES:
        feat = " ".join(product.features).lower()
        score = 70.0
        details = []
        if "memory foam" in feat or "plush" in feat or "cushion" in feat:
            score += 15.0
            details.append("plush memory foam ear cushions")
        if "ergonomic" in feat or "lightweight" in feat:
            score += 10.0
            details.append("ergonomic headband")
        w_g = product.specs.get("weight_g")
        if w_g and w_g < 250:
            score = min(100.0, score + 5.0)
            details.append(f"lightweight {w_g}g weight")
        return min(100.0, score), f"Comfort features: {', '.join(details) if details else 'standard earcups'}.", True

    return 50.0, f"Category '{product.category.value}' does not support comfort evaluation.", False


def _score_build_quality(product: Product) -> tuple[float, str]:
    """Score build quality from premium material selling points."""
    feat_text = " ".join(product.features).lower()
    score = 70.0
    materials: list[str] = []

    for mat in ("aluminum", "titanium", "unibody", "gorilla glass", "ip68", "magnesium"):
        if mat in feat_text:
            score = min(100.0, score + 12.0)
            materials.append(mat.title())

    if materials:
        return round(score, 1), f"Premium materials ({', '.join(materials)}) rated {score:.1f}/100."
    return score, "Standard durable construction rated 70/100."


def score_candidate(product: Product, requirements: Requirements) -> ScoreBreakdown:
    """Calculate transparent, multi-factor preference scores for an eligible candidate product.

    Combines:
        1. User-specified preferences (70% weight if present)
        2. Intended use-case persona matching (15% weight, or 50% if no preferences)
        3. Verified customer review rating (15% weight, or 50% if no preferences)

    Args:
        product: The candidate product to score.
        requirements: Current user requirements containing soft preferences and use cases.

    Returns:
        ScoreBreakdown containing overall score, per-preference decomposition, and explanations.
    """
    preference_scores: dict[str, float] = {}
    explanations: dict[str, str] = {}
    unsupported_preferences: list[str] = []

    # 1. Evaluate known preference dimensions
    # Portability
    p_score, p_exp = _score_portability(product)
    preference_scores["portability"] = p_score
    explanations["portability"] = p_exp

    # Performance
    perf_score, perf_exp = _score_performance(product)
    preference_scores["performance"] = perf_score
    explanations["performance"] = perf_exp

    # Battery Life
    bat_score, bat_exp = _score_battery_life(product)
    preference_scores["battery_life"] = bat_score
    explanations["battery_life"] = bat_exp

    # Display Quality
    disp_score, disp_exp = _score_display_quality(product)
    preference_scores["display_quality"] = disp_score
    explanations["display_quality"] = disp_exp

    # Camera Quality
    cam_score, cam_exp, cam_supp = _score_camera_quality(product)
    preference_scores["camera_quality"] = cam_score
    explanations["camera_quality"] = cam_exp
    if not cam_supp and "camera_quality" in requirements.preferences:
        unsupported_preferences.append("camera_quality")

    # GPU Power
    gpu_score, gpu_exp, gpu_supp = _score_gpu_power(product)
    preference_scores["gpu_power"] = gpu_score
    explanations["gpu_power"] = gpu_exp
    if not gpu_supp and "gpu_power" in requirements.preferences:
        unsupported_preferences.append("gpu_power")

    # Noise Cancellation
    anc_score, anc_exp, anc_supp = _score_noise_cancellation(product)
    preference_scores["noise_cancellation"] = anc_score
    explanations["noise_cancellation"] = anc_exp
    if not anc_supp and "noise_cancellation" in requirements.preferences:
        unsupported_preferences.append("noise_cancellation")

    # Microphone Quality
    mic_score, mic_exp, mic_supp = _score_microphone_quality(product)
    preference_scores["microphone_quality"] = mic_score
    explanations["microphone_quality"] = mic_exp
    if not mic_supp and "microphone_quality" in requirements.preferences:
        unsupported_preferences.append("microphone_quality")

    # Comfort
    comf_score, comf_exp, comf_supp = _score_comfort(product)
    preference_scores["comfort"] = comf_score
    explanations["comfort"] = comf_exp
    if not comf_supp and "comfort" in requirements.preferences:
        unsupported_preferences.append("comfort")

    # Build Quality
    bq_score, bq_exp = _score_build_quality(product)
    preference_scores["build_quality"] = bq_score
    explanations["build_quality"] = bq_exp

    # 2. Rating Score component (0-100)
    rating_val = product.rating if product.rating is not None else 3.5
    rating_score = round((rating_val / 5.0) * 100.0, 1)

    # 3. Use-Case Persona Match component (0-100)
    if requirements.use_cases:
        user_ucs = {uc.strip().lower() for uc in requirements.use_cases if uc.strip()}
        prod_ucs = {uc.strip().lower() for uc in product.intended_use if uc.strip()}
        # Count overlapping use cases
        overlap = sum(
            1 for u in user_ucs if any(u in p or p in u for p in prod_ucs)
        )
        ratio = overlap / len(user_ucs)
        use_case_score = round(min(100.0, max(30.0, 40.0 + ratio * 60.0)), 1)
    else:
        use_case_score = 75.0

    # 4. Weighted combination across active user preferences
    active_weights: dict[str, float] = {}
    for key, weight in requirements.preferences.items():
        canonical_key = key.strip().lower()
        if weight > 0 and canonical_key in preference_scores:
            active_weights[canonical_key] = weight

    total_weight = sum(active_weights.values())
    weighted_scores: dict[str, float] = {}

    if total_weight > 0:
        pref_component = 0.0
        # Iterate in sorted order to guarantee deterministic float summation
        for pref_name in sorted(active_weights.keys()):
            w = active_weights[pref_name]
            normalized_w = w / total_weight
            raw_s = preference_scores[pref_name]
            contribution = round(raw_s * normalized_w, 2)
            weighted_scores[pref_name] = contribution
            pref_component += raw_s * normalized_w

        overall_score = round(
            0.70 * pref_component + 0.15 * use_case_score + 0.15 * rating_score, 2
        )
    else:
        overall_score = round(0.50 * use_case_score + 0.50 * rating_score, 2)

    # Ensure strictly within 0.0 to 100.0 boundaries
    overall_score = max(0.0, min(100.0, overall_score))

    return ScoreBreakdown(
        overall_score=overall_score,
        preference_scores=preference_scores,
        weighted_scores=weighted_scores,
        explanations=explanations,
        unsupported_preferences=sorted(unsupported_preferences),
        rating_score=rating_score,
        use_case_score=use_case_score,
    )


# ---------------------------------------------------------------------------
# 3. Deterministic Candidate Ranking
# ---------------------------------------------------------------------------

def _generate_candidate_reasons(
    product: Product,
    score_breakdown: ScoreBreakdown,
    requirements: Requirements,
) -> list[str]:
    """Generate concise, human-understandable selection reasons for a recommended product."""
    reasons: list[str] = []

    # 1. Budget efficiency highlight
    if requirements.budget_max_inr is not None:
        diff = requirements.budget_max_inr - product.price_inr
        if diff >= 5000:
            reasons.append(f"Priced at ₹{product.price_inr:,}, saving ₹{diff:,} under your ₹{requirements.budget_max_inr:,} budget.")
        else:
            reasons.append(f"Comfortably within budget at ₹{product.price_inr:,}.")
    else:
        reasons.append(f"Competitively priced at ₹{product.price_inr:,}.")

    # 2. Top-scoring preference matches
    if score_breakdown.weighted_scores:
        top_pref = max(score_breakdown.weighted_scores.items(), key=lambda item: item[1])
        pref_name = top_pref[0].replace("_", " ").title()
        raw_val = score_breakdown.preference_scores.get(top_pref[0], 0.0)
        reasons.append(f"Excellent {pref_name} rating ({raw_val:.0f}/100).")

    # 3. Customer rating highlight
    if product.rating is not None and product.rating >= 4.0:
        reasons.append(f"Strong customer satisfaction with a {product.rating}/5.0 review score.")

    # 4. Use case persona match
    if requirements.use_cases:
        matched_cases = [u for u in requirements.use_cases if any(u.lower() in p.lower() for p in product.intended_use)]
        if matched_cases:
            cases_str = ", ".join(matched_cases[:2])
            reasons.append(f"Tailored specifically for {cases_str}.")

    return reasons[:4]


def rank_candidates(
    eligible_products: Sequence[Product],
    requirements: Requirements,
    top_n: int = 5,
) -> list[RankedCandidate]:
    """Rank eligible candidates deterministically using multi-factor scores.

    Tie-breaking hierarchy (100% deterministic):
        1. Composite overall_score descending.
        2. price_inr ascending (more economical product preferred).
        3. product_id ascending (lexicographical fallback).

    Args:
        eligible_products: Sequence of products that passed all hard constraints.
        requirements: Requirements object containing soft preference weights.
        top_n: Maximum number of ranked products to return in shortlist.

    Returns:
        Ordered list of RankedCandidate objects.
    """
    candidates: list[RankedCandidate] = []

    for product in eligible_products:
        eval_report = evaluate_candidate(product, requirements)
        score_bd = score_candidate(product, requirements)
        reasons = _generate_candidate_reasons(product, score_bd, requirements)

        candidates.append(
            RankedCandidate(
                product_id=product.product_id,
                product=product,
                overall_score=score_bd.overall_score,
                score_breakdown=score_bd,
                evaluation=eval_report,
                reasons=reasons,
            )
        )

    # Deterministic sort
    candidates.sort(
        key=lambda c: (-c.overall_score, c.product.price_inr, c.product_id)
    )

    return candidates[:top_n]


# ---------------------------------------------------------------------------
# 4. End-to-End Retrieval, Filtering, and Ranking Coordinator
# ---------------------------------------------------------------------------

async def search_and_rank(
    requirements: Requirements,
    repository: ProductRepository,
    candidate_limit: int = 50,
    top_n: int = 5,
) -> CandidateSearchResult:
    """Coordinate candidate retrieval, constraint filtering, and preference ranking.

    Args:
        requirements: Accumulated requirements state from conversation turns.
        repository: Injected ProductRepository for catalogue queries.
        candidate_limit: Maximum products to retrieve from the catalogue.
        top_n: Size of the recommended shortlist.

    Returns:
        CandidateSearchResult summarizing retrieval metrics, exclusions, and ranked candidates.
    """
    # 1. Retrieve candidates
    retrieved = await retrieve_candidates(requirements, repository, limit=candidate_limit)

    # 2. Hard constraint filtering
    eligible, excluded = filter_candidates(retrieved, requirements)

    # 3. Rank eligible candidates
    ranked = rank_candidates(eligible, requirements, top_n=top_n)

    # 4. Synthesize data-quality limitations across excluded candidates
    data_quality_limitations: set[str] = set()
    for ex in excluded:
        for u in ex.unverifiable_constraints:
            data_quality_limitations.add(u.reason)

    total_unverifiable = sum(
        1 for e in excluded if len(e.failed_constraints) == 0 and len(e.unverifiable_constraints) > 0
    )
    total_excluded = len(excluded) - total_unverifiable

    return CandidateSearchResult(
        category=requirements.category,
        total_retrieved=len(retrieved),
        total_passed=len(eligible),
        total_excluded=total_excluded,
        total_unverifiable=total_unverifiable,
        ranked_candidates=ranked,
        excluded_candidates=excluded,
        data_quality_limitations=sorted(data_quality_limitations),
        has_matches=(len(ranked) > 0),
    )
