"""Follow-Up Intent Recognition and Catalogue-Grounded Reasoning Engine.

Handles conversational inquiries following NO_MATCH or during active recommendation sessions:
    - Minimum-budget inquiries ("What is the minimum budget?", "How much more do I need to spend?", "Cheapest option")
    - Blocking-constraint diagnostics ("Which requirement is preventing a match?", "Why are there no matches?")
    - Closest alternatives requests ("Can you find a cheaper alternative?", "Show me the closest alternatives")

Operating Principles:
    1. Grounded in actual catalogue data: Never invent products, prices, or specifications.
    2. Hard constraints preserved: Never silently relax hard requirements.
    3. Category agnostic: Works for Laptops, Phones, Headphones, Tablets, and Monitors.
    4. Honest limitations: If a specification is unavailable in the catalogue at any price,
       never claim that increasing the budget alone will solve the problem.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from app.agent.ranking import evaluate_candidate
from app.core.logging import get_logger
from app.db.repositories import ProductRepository
from app.models.conversation import ConversationState
from app.models.product import Category, Product
from app.models.requirements import Requirements

logger = get_logger("app.agent.followup")


class FollowUpIntent(str, Enum):
    """Categorized user follow-up inquiry intent."""

    MINIMUM_BUDGET = "minimum_budget"
    EXPLAIN_BLOCKING = "explain_blocking"
    CLOSEST_ALTERNATIVES = "closest_alternatives"


RE_MINIMUM_BUDGET = re.compile(
    r"\b("
    r"what(?:'s|\s+is)?\s+(?:the\s+)?min(?:imum)?\s*(?:budget|price|cost)|"
    r"how\s+much\s+(?:more\s+)?(?:do\s+i\s+need\s+to\s+spend|to\s+spend|more)|"
    r"what(?:'s|\s+is)?\s+(?:the\s+)?cheapest(?:\s+option|\s+one|\s+model|\s+alternative|\s+phone|\s+laptop|\s+headphones?|\s+tablet|\s+device)?|"
    r"cheapest\s+(?:option|phone|laptop|headphones?|tablet|monitor|device)|"
    r"what\s+budget\s+(?:do\s+i\s+need|is\s+needed|is\s+required)|"
    r"what\s+if\s+i\s+increase\s+(?:my\s+)?budget|"
    r"keep\s+\d+\s*gb\s+but\s+increase\s+(?:my\s+)?budget|"
    r"increase\s+(?:my\s+)?budget$|"
    r"raise\s+(?:my\s+)?budget$|"
    r"how\s+much\s+does\s+it\s+cost|"
    r"starting\s+price|"
    r"lowest\s+price"
    r")\b",
    re.IGNORECASE,
)

RE_EXPLAIN_BLOCKING = re.compile(
    r"\b("
    r"which\s+(?:requirement|spec|specification|constraint|feature|factor)\s+(?:is|are)?\s*(?:preventing|blocking|causing|the\s*problem)|"
    r"why\s+(?:are\s+there\s+)?no\s+match(?:es)?|"
    r"why\s+(?:did\s+it|was\s+there)\s+fail(?:ure)?|"
    r"what\s+(?:is\s+)?(?:blocking|preventing|the\s+problem|causing\s+no\s+match)"
    r")\b",
    re.IGNORECASE,
)

RE_CLOSEST_ALTERNATIVES = re.compile(
    r"\b("
    r"show\s+(?:me\s+)?(?:the\s+)?(?:closest\s+)?alternatives?|"
    r"closest\s+alternatives?|"
    r"(?:can\s+you\s+)?find\s+(?:a\s+)?cheaper\s+alternative|"
    r"what\s+are\s+the\s+alternatives?|"
    r"other\s+options?"
    r")\b",
    re.IGNORECASE,
)


def detect_followup_intent(
    text: str,
    conversation: ConversationState | None = None,
) -> FollowUpIntent | None:
    """Classify user message into a follow-up inquiry intent if applicable.

    Distinguishes informational inquiries from concrete constraint updates.
    """
    text_clean = text.strip().lower()

    # If the user supplied a concrete budget number (e.g. "increase budget to 40000"),
    # this is a budget update, not an informational inquiry.
    has_budget_number = bool(
        re.search(
            r"(?:to|is|of|under|at)\s*(\d+[\d,]*\s*(?:k|lakh)?|\d+k|\d+\s*lakh)\b",
            text_clean,
            re.I,
        )
    )

    if RE_MINIMUM_BUDGET.search(text_clean):
        # If user explicitly specifies a new budget amount (e.g. "increase budget to 45k"), treat as update
        if has_budget_number and not re.search(r"what\s+if|how\s+much|what\s+is", text_clean):
            return None
        return FollowUpIntent.MINIMUM_BUDGET

    if RE_EXPLAIN_BLOCKING.search(text_clean):
        return FollowUpIntent.EXPLAIN_BLOCKING

    if RE_CLOSEST_ALTERNATIVES.search(text_clean):
        return FollowUpIntent.CLOSEST_ALTERNATIVES

    return None


async def _get_all_category_products(
    category: Category, product_repo: ProductRepository
) -> list[Product]:
    """Retrieve all products in a given category from the repository abstraction."""
    if hasattr(product_repo, "find"):
        prods, _ = await product_repo.find(
            filters={"category": category},
            sort="price_inr_asc",
            page=1,
            page_size=100,
        )
        if prods:
            return prods

    if hasattr(product_repo, "get_by_category"):
        prods = await product_repo.get_by_category(category)
        if prods:
            return sorted(prods, key=lambda p: p.price_inr)

    return []


def _format_product_specs_summary(product: Product) -> str:
    """Format key hardware specifications of a product for conversational presentation."""
    specs = product.specs or {}
    cat = product.category
    parts: list[str] = []

    if cat == Category.PHONE:
        if specs.get("ram_gb"):
            parts.append(f"{specs['ram_gb']}GB RAM")
        if specs.get("storage_gb"):
            parts.append(f"{specs['storage_gb']}GB storage")
        if specs.get("camera_mp"):
            parts.append(f"{int(specs['camera_mp'])}MP camera")
        if specs.get("battery_mah"):
            parts.append(f"{specs['battery_mah']}mAh battery")
    elif cat == Category.LAPTOP:
        if specs.get("ram_gb"):
            parts.append(f"{specs['ram_gb']}GB RAM")
        if specs.get("storage_gb"):
            parts.append(f"{specs['storage_gb']}GB SSD")
        if specs.get("gpu"):
            parts.append(str(specs["gpu"]))
        if specs.get("cpu"):
            parts.append(str(specs["cpu"]))
    elif cat == Category.HEADPHONES:
        if specs.get("type"):
            parts.append(str(specs["type"]))
        if specs.get("wireless"):
            parts.append("wireless")
        if specs.get("anc"):
            parts.append("ANC")
        if specs.get("battery_hr"):
            parts.append(f"{specs['battery_hr']}h battery")
    elif cat == Category.TABLET:
        if specs.get("screen_in"):
            parts.append(f"{specs['screen_in']}-inch screen")
        if specs.get("storage_gb"):
            parts.append(f"{specs['storage_gb']}GB storage")
    elif cat == Category.MONITOR:
        if specs.get("screen_in"):
            parts.append(f"{specs['screen_in']}-inch")
        if specs.get("resolution"):
            parts.append(str(specs["resolution"]))
        if specs.get("refresh_hz"):
            parts.append(f"{specs['refresh_hz']}Hz")

    return ", ".join(parts) if parts else "standard specifications"


async def analyze_catalogue_minimum_budget(
    requirements: Requirements,
    product_repo: ProductRepository,
) -> str:
    """Analyze actual catalogue products to compute minimum budget for active non-budget constraints."""
    category = requirements.category
    if category is None:
        return "Please specify a product category (e.g. laptop, phone, headphones) so I can check pricing in our catalogue."

    products = await _get_all_category_products(category, product_repo)
    if not products:
        return f"No products in category '{category.value}' are currently available in our catalogue."

    cat_name = category.value

    # Build copy of requirements ignoring budget constraint to evaluate purely hardware/spec requirements
    req_no_budget = requirements.model_copy(deep=True)
    req_no_budget.budget_max_inr = None
    req_no_budget.budget_min_inr = None

    valid_candidates: list[Product] = []
    for p in products:
        ev = evaluate_candidate(p, req_no_budget)
        if ev.is_eligible:
            valid_candidates.append(p)

    # CASE A: Valid products exist satisfying all non-budget specifications
    if valid_candidates:
        cheapest_valid = min(valid_candidates, key=lambda p: p.price_inr)
        min_price = cheapest_valid.price_inr
        specs_summary = _format_product_specs_summary(cheapest_valid)

        if requirements.budget_max_inr is not None:
            curr_b = requirements.budget_max_inr
            if min_price > curr_b:
                diff = min_price - curr_b
                return (
                    f"The minimum budget required in our catalogue to satisfy all your specifications "
                    f"is **₹{min_price:,}** for the **{cheapest_valid.name}** "
                    f"(₹{diff:,} more than your current budget of ₹{curr_b:,}). "
                    f"It satisfies your requirements with {specs_summary}. "
                    f"Would you like to increase your budget to ₹{min_price:,} to proceed with recommendations?"
                )
            else:
                return (
                    f"The lowest-priced option in our catalogue meeting your specifications is the "
                    f"**{cheapest_valid.name}** at **₹{min_price:,}** ({specs_summary}), "
                    f"which is within your budget of ₹{curr_b:,}. "
                    f"Would you like to review this recommendation?"
                )
        else:
            return (
                f"The minimum budget required in our catalogue to satisfy all your specifications is "
                f"**₹{min_price:,}** for the **{cheapest_valid.name}** ({specs_summary}). "
                f"Would you like to proceed with this budget?"
            )

    # CASE B: Zero candidates in the entire catalogue satisfy the non-budget hard constraints
    # Identify which specific constraint(s) have 0 matches in the catalogue
    # 1. Check storage
    if requirements.hard.min_storage_gb is not None:
        matching_storage = [
            p for p in products if p.specs.get("storage_gb", 0) >= requirements.hard.min_storage_gb
        ]
        if not matching_storage:
            max_avail_storage = max((p.specs.get("storage_gb", 0) for p in products), default=0)
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no {cat_name}s "
                f"with at least {requirements.hard.min_storage_gb}GB storage at any price (the maximum available "
                f"storage in our catalogue is {max_avail_storage}GB). "
                f"Therefore, increasing your budget alone will not yield a match. "
                f"Would you like to keep {requirements.hard.min_storage_gb}GB as a mandatory requirement, "
                f"or consider relaxing your storage requirement to {max_avail_storage}GB?"
            )

    # 2. Check RAM
    if requirements.hard.min_ram_gb is not None:
        matching_ram = [
            p for p in products if p.specs.get("ram_gb", 0) >= requirements.hard.min_ram_gb
        ]
        if not matching_ram:
            max_avail_ram = max((p.specs.get("ram_gb", 0) for p in products), default=0)
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no {cat_name}s "
                f"with at least {requirements.hard.min_ram_gb}GB RAM at any price (the maximum available "
                f"RAM in our catalogue is {max_avail_ram}GB). "
                f"Therefore, increasing your budget alone will not yield a match. "
                f"Would you consider relaxing your RAM requirement to {max_avail_ram}GB?"
            )

    # 3. Check Headphone Form Factor
    if requirements.hard.form_factor is not None:
        matching_ff = [
            p
            for p in products
            if str(p.specs.get("type", "")).strip().lower() == requirements.hard.form_factor.strip().lower()
        ]
        if not matching_ff:
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no headphones "
                f"with the '{requirements.hard.form_factor}' form factor. "
                f"Would you like to consider a different form factor?"
            )

    # 4. Check Dedicated GPU
    if requirements.hard.requires_gpu is True:
        matching_gpu = [
            p
            for p in products
            if p.specs.get("vram_gb", 0) > 0 or any(kw in str(p.specs.get("gpu", "")).lower() for kw in ("rtx", "gtx", "radeon"))
        ]
        if not matching_gpu:
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no laptops "
                f"with a dedicated GPU. Would you consider models with integrated graphics?"
            )

    # 5. Check Stylus Support (Tablets)
    if requirements.hard.requires_stylus is True:
        matching_stylus = [p for p in products if p.specs.get("stylus_support") is True]
        if not matching_stylus:
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no tablets "
                f"with stylus support. Would you consider tablets without stylus support?"
            )

    # 6. Check Refresh Rate (Monitors)
    if requirements.hard.min_refresh_hz is not None:
        matching_hz = [p for p in products if p.specs.get("refresh_hz", 0) >= requirements.hard.min_refresh_hz]
        if not matching_hz:
            max_avail_hz = max((p.specs.get("refresh_hz", 0) for p in products), default=0)
            return (
                f"I cannot determine a minimum budget because our catalogue currently has no monitors "
                f"with at least {requirements.hard.min_refresh_hz}Hz refresh rate at any price (the maximum available "
                f"in our catalogue is {max_avail_hz}Hz). "
                f"Would you consider relaxing your refresh rate requirement to {max_avail_hz}Hz?"
            )

    # 7. Multi-constraint conflict (e.g. 256GB storage + weight < 170g)
    return (
        f"I cannot determine a minimum budget because no single {cat_name} in our catalogue satisfies "
        f"all your non-budget constraints simultaneously. "
        f"Increasing your budget alone will not resolve this unless conflicting constraints (such as weight, "
        f"RAM, or storage limits) are adjusted. Would you like to review and relax any of these requirements?"
    )


async def explain_blocking_constraints(
    requirements: Requirements,
    product_repo: ProductRepository,
) -> str:
    """Diagnose and clearly explain the specific constraints blocking a match."""
    category = requirements.category
    if category is None:
        return "Please specify a product category so I can diagnose the constraints."

    products = await _get_all_category_products(category, product_repo)
    cat_name = category.value

    req_no_budget = requirements.model_copy(deep=True)
    req_no_budget.budget_max_inr = None

    valid_no_budget = [p for p in products if evaluate_candidate(p, req_no_budget).is_eligible]

    bullets: list[str] = []

    if valid_no_budget:
        cheapest_valid = min(valid_no_budget, key=lambda p: p.price_inr)
        if requirements.budget_max_inr is not None and cheapest_valid.price_inr > requirements.budget_max_inr:
            diff = cheapest_valid.price_inr - requirements.budget_max_inr
            bullets.append(
                f"• **Budget Limit (₹{requirements.budget_max_inr:,})**: Products meeting all your specifications exist starting at **₹{cheapest_valid.price_inr:,}** ({cheapest_valid.name}), which is ₹{diff:,} above your budget."
            )
    else:
        # Hardware constraint is blocking
        if requirements.hard.min_storage_gb is not None:
            max_storage = max((p.specs.get("storage_gb", 0) for p in products), default=0)
            if max_storage < requirements.hard.min_storage_gb:
                bullets.append(
                    f"• **Storage Requirement ({requirements.hard.min_storage_gb}GB)**: Our catalogue currently offers up to {max_storage}GB storage in {cat_name}s."
                )
            else:
                bullets.append(
                    f"• **Storage & Hardware Conflict**: Models with {requirements.hard.min_storage_gb}GB storage are available, but conflict with other constraints (such as weight or budget)."
                )

        if requirements.hard.min_ram_gb is not None:
            max_ram = max((p.specs.get("ram_gb", 0) for p in products), default=0)
            if max_ram < requirements.hard.min_ram_gb:
                bullets.append(
                    f"• **RAM Requirement ({requirements.hard.min_ram_gb}GB)**: Our catalogue offers up to {max_ram}GB RAM in {cat_name}s."
                )

        if requirements.hard.form_factor is not None:
            bullets.append(
                f"• **Form Factor**: No models matching your combination were found for '{requirements.hard.form_factor}'."
            )

        if requirements.hard.requires_stylus is True:
            bullets.append(
                "• **Stylus Requirement**: Requires active stylus/pen input support."
            )

        if requirements.hard.min_refresh_hz is not None:
            bullets.append(
                f"• **Refresh Rate Requirement ({requirements.hard.min_refresh_hz}Hz)**: Higher than currently supported in this price bracket."
            )

    bullet_text = "\n".join(bullets) if bullets else "• Non-negotiable specifications and current budget limit."
    return (
        f"Here are the specific requirements currently preventing a match in our catalogue:\n\n"
        f"{bullet_text}\n\n"
        f"Would you like to adjust your budget, or would you prefer to relax one of these specifications?"
    )


async def find_closest_alternatives(
    requirements: Requirements,
    product_repo: ProductRepository,
) -> str:
    """Find and clearly describe closest alternatives without violating hard constraints."""
    category = requirements.category
    if category is None:
        return "Please specify a product category so I can search for alternatives."

    products = await _get_all_category_products(category, product_repo)
    if not products:
        return "No alternative products could be found in our catalogue."

    cat_name = category.value
    curr_b = requirements.budget_max_inr or 9999999

    # Option 1: Products satisfying all specs, but slightly above budget
    req_no_budget = requirements.model_copy(deep=True)
    req_no_budget.budget_max_inr = None
    over_budget_valid = [
        p for p in products if evaluate_candidate(p, req_no_budget).is_eligible and p.price_inr > curr_b
    ]
    over_budget_valid.sort(key=lambda p: p.price_inr)

    # Option 2: Products within budget, but with closest lower spec (e.g. 128GB instead of 256GB)
    in_budget_alts = [p for p in products if p.price_inr <= curr_b]
    in_budget_alts.sort(key=lambda p: (-(p.specs.get("storage_gb") or 0), -(p.rating or 0)))

    lines = [f"Here are the closest alternatives from our catalogue, along with the specific trade-offs:\n"]
    has_any = False

    if in_budget_alts:
        alt_ib = in_budget_alts[0]
        specs_ib = _format_product_specs_summary(alt_ib)
        diff_spec = ""
        if requirements.hard.min_storage_gb and alt_ib.specs.get("storage_gb"):
            diff_spec = f"has {alt_ib.specs['storage_gb']}GB storage (below your required {requirements.hard.min_storage_gb}GB)"
        else:
            diff_spec = "differs in some specifications"

        lines.append(
            f"1. **{alt_ib.name}** (₹{alt_ib.price_inr:,}) — **Within budget**, but {diff_spec}. Features: {specs_ib}."
        )
        has_any = True

    if over_budget_valid:
        alt_ob = over_budget_valid[0]
        diff_price = alt_ob.price_inr - curr_b
        specs_ob = _format_product_specs_summary(alt_ob)
        lines.append(
            f"2. **{alt_ob.name}** (₹{alt_ob.price_inr:,}) — **Meets all your specifications**, but is ₹{diff_price:,} above your budget. Features: {specs_ob}."
        )
        has_any = True

    if not has_any:
        return (
            f"No close alternatives were found for {cat_name} in our catalogue. "
            f"Please consider adjusting your budget or relaxing specific specifications."
        )

    lines.append(
        "\nWould you like to relax any requirement to consider one of these options, or adjust your budget?"
    )
    return "\n".join(lines)
