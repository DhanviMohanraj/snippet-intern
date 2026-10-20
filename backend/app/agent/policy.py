"""Clarification Policy and Slot-Filling Decision Engine.

Provides deterministic decision-making on conversational product-recommendation turns:
    - ASK: Request targeted essential requirements (category, budget, use case, critical specs).
    - CLARIFY_CONFLICT: Request resolution of logical contradictions.
    - REDIRECT: Politely steer off-topic or conversational requests back to electronics.
    - NO_MATCH: Safely reject unsupported categories or unrecoverable turn limits.
    - RECOMMEND: Signal that requirements are sufficient to proceed to candidate retrieval.

Central Design Principle:
    The LLM extracts meaning; deterministic Python policy decides the next conversational action.
    No network, database, or LLM operations are performed.
"""

from __future__ import annotations

from typing import Any

from app.agent.merge import detect_contradictions
from app.models.product import Category
from app.models.requirements import (
    Contradiction,
    PolicyAction,
    PolicyDecision,
    Requirements,
)

# Configurable maximum clarification turns before proceeding or safely terminating
MAX_CLARIFICATION_TURNS: int = 3

# Question text for missing category
CATEGORY_QUESTION: str = (
    "What type of electronics are you looking for? "
    "(e.g., laptop, phone, headphones, tablet, or monitor)"
)

# Question text for missing budget
BUDGET_QUESTION: str = (
    "What is your budget or maximum price in INR? "
    "(e.g., ₹50,000, ₹1,00,000, or let me know if your budget is flexible)"
)

CATEGORY_BUDGET_QUESTIONS: dict[Category, str] = {
    Category.HEADPHONES: (
        "What is your budget or maximum price in INR? "
        "(e.g., ₹3,000, ₹8,000, or let me know if your budget is flexible)"
    ),
    Category.PHONE: (
        "What is your budget or maximum price in INR? "
        "(e.g., ₹20,000, ₹40,000, or let me know if your budget is flexible)"
    ),
    Category.LAPTOP: (
        "What is your budget or maximum price in INR? "
        "(e.g., ₹50,000, ₹80,000, or let me know if your budget is flexible)"
    ),
    Category.TABLET: (
        "What is your budget or maximum price in INR? "
        "(e.g., ₹25,000, ₹45,000, or let me know if your budget is flexible)"
    ),
    Category.MONITOR: (
        "What is your budget or maximum price in INR? "
        "(e.g., ₹12,000, ₹25,000, or let me know if your budget is flexible)"
    ),
}

# Category-specific use case inquiry questions
CATEGORY_USE_CASE_QUESTIONS: dict[Category, str] = {
    Category.LAPTOP: (
        "What will you primarily use this laptop for? "
        "(e.g., programming, gaming, college, machine learning, or general work)"
    ),
    Category.PHONE: (
        "What is your primary use case for this phone? "
        "(e.g., photography, gaming, everyday use, battery life, or productivity)"
    ),
    Category.HEADPHONES: (
        "What will you primarily use these headphones for? "
        "(e.g., calls, travel, music, studying, or noise cancellation)"
    ),
    Category.TABLET: (
        "What is your main use case for this tablet? "
        "(e.g., note-taking, digital art/drawing, media consumption, or productivity)"
    ),
    Category.MONITOR: (
        "What will you primarily use this monitor for? "
        "(e.g., gaming, office work, coding, or color-accurate content creation)"
    ),
}

# Question text for clarifying machine learning execution mode on laptops
ML_MODE_QUESTION: str = (
    "Will you be training machine learning models locally on the laptop (requiring a dedicated GPU), "
    "or using cloud platforms like Google Colab or AWS?"
)

# Polite redirection prompt when a user message is off-topic
OFF_TOPIC_REDIRECT_MESSAGE: str = (
    "I am your Electronics Recommendation Assistant. I can help you find laptops, phones, "
    "headphones, tablets, or monitors. How can I assist you with your electronics search today?"
)

# Canonical mapping for slot name normalization
SLOT_ALIASES: dict[str, str] = {
    "category": "category",
    "product_category": "category",
    "device": "category",
    "device_type": "category",
    "budget": "budget",
    "budget_max": "budget",
    "budget_min": "budget",
    "budget_max_inr": "budget",
    "budget_min_inr": "budget",
    "price": "budget",
    "price_range": "budget",
    "price_limit": "budget",
    "use_case": "use_cases",
    "use_cases": "use_cases",
    "intended_use": "use_cases",
    "purpose": "use_cases",
    "ml_mode": "ml_mode",
    "machine_learning": "ml_mode",
    "local_ml": "ml_mode",
    "conflict": "conflict",
    "contradiction": "conflict",
    "ram": "ram",
    "min_ram_gb": "ram",
    "memory": "ram",
    "storage": "storage",
    "min_storage_gb": "storage",
    "ssd": "storage",
    "gpu": "gpu",
    "graphics": "gpu",
    "requires_gpu": "gpu",
    "form_factor": "form_factor",
    "phone_priority": "phone_priority",
    "stylus_keyboard": "stylus_keyboard",
    "display_specs": "display_specs",
}

# Keywords indicating machine learning / AI use cases
ML_USE_CASE_KEYWORDS: set[str] = {
    "ml",
    "machine learning",
    "deep learning",
    "ai",
    "artificial intelligence",
    "data science",
    "model training",
}


def normalize_slot_name(slot: str) -> str:
    """Normalize a slot name to its canonical identifier."""
    cleaned = slot.strip().lower()
    return SLOT_ALIASES.get(cleaned, cleaned)


def format_contradiction_question(contradiction: Contradiction, req: Requirements) -> str:
    """Generate a concise, human-understandable clarification question for a contradiction."""
    code = contradiction.code

    if code == "BUDGET_MIN_EXCEEDS_MAX":
        min_str = f"₹{req.budget_min_inr:,}" if req.budget_min_inr is not None else "your minimum budget"
        max_str = f"₹{req.budget_max_inr:,}" if req.budget_max_inr is not None else "your maximum budget"
        return (
            f"Your minimum budget of {min_str} exceeds your maximum budget of {max_str}. "
            "Which budget limit would you like to adjust?"
        )

    if code == "GPU_FALSE_WITH_MIN_VRAM":
        vram_str = f"{req.hard.min_vram_gb}GB" if req.hard.min_vram_gb else "VRAM"
        return (
            f"You specified that a dedicated GPU is not required, but also set a minimum VRAM of {vram_str}. "
            "Would you like a dedicated GPU, or should we remove the VRAM requirement?"
        )

    if code == "ML_LOCAL_CONFLICTS_NO_GPU":
        return (
            "Local machine learning execution requires a dedicated GPU, but GPU requirement is disabled. "
            "Would you like to enable a dedicated GPU or switch to cloud-based ML?"
        )

    if code == "CATEGORY_SPEC_INCOMPATIBLE":
        cat_name = req.category.value if req.category else "this category"
        fields_str = ", ".join(contradiction.fields)
        return (
            f"Your requirements contain specifications ({fields_str}) that are not applicable to {cat_name}. "
            "Would you like to adjust these specifications or choose a different device category?"
        )

    # Fallback to sanitized message on the contradiction object
    return f"We found an inconsistency in your requirements: {contradiction.message} How would you like to resolve this?"


def decide_next_action(
    requirements: Requirements,
    contradictions: list[Contradiction] | None = None,
    asked_slots: list[str] | None = None,
    turn_count: int = 0,
    max_turns: int = MAX_CLARIFICATION_TURNS,
    is_off_topic: bool = False,
    unsupported_category: str | None = None,
    slot_status: dict[str, str] | None = None,
    enforce_mandatory_slots: bool = False,
) -> PolicyDecision:
    """Deterministically decide the next conversational action.

    Priority order:
        1. Off-topic redirection (REDIRECT).
        2. Unsupported category handling (NO_MATCH).
        3. Turn limit enforcement with safe stopping (RECOMMEND or NO_MATCH).
        4. Missing product category (ASK category).
        5. Unresolved logical contradictions (CLARIFY_CONFLICT).
        6. Missing mandatory category-specific requirements (ASK targeted slot).
        7. Sufficient requirements established (RECOMMEND).

    Args:
        requirements: Accumulated requirements state from previous turns.
        contradictions: Optional pre-computed contradictions. If None, detected automatically.
        asked_slots: Optional list of previously asked slot names (merged with requirements.asked_slots).
        turn_count: Number of clarification turns already executed in this conversation.
        max_turns: Maximum allowed clarification turns before stopping.
        is_off_topic: Flag indicating the latest message was off-topic.
        unsupported_category: Name of an explicitly requested unsupported category, if any.
        slot_status: Optional mapping of slot names to SlotStatus values.
        enforce_mandatory_slots: When True, enforces all category-specific mandatory slots.

    Returns:
        PolicyDecision containing action, targeted question, slot, reason, and can_recommend flag.
    """
    from app.agent.slots import (
        can_proceed_to_recommendation,
        get_next_slot_to_clarify,
    )

    # 1. Off-topic check
    if is_off_topic:
        return PolicyDecision(
            action=PolicyAction.REDIRECT,
            question=OFF_TOPIC_REDIRECT_MESSAGE,
            slot=None,
            reason="off_topic",
            can_recommend=False,
            remaining_slots=[],
        )

    # 2. Unsupported category check
    if unsupported_category:
        return PolicyDecision(
            action=PolicyAction.NO_MATCH,
            question=(
                f"Currently, I can only recommend laptops, phones, headphones, tablets, and monitors. "
                f"I cannot provide recommendations for '{unsupported_category}'."
            ),
            slot="category",
            reason="unsupported_category",
            metadata={"unsupported_category": unsupported_category},
            can_recommend=False,
            remaining_slots=[],
        )

    # Combine asked_slots from requirements and explicit parameter
    effective_asked: set[str] = set()
    for s in (requirements.asked_slots or []):
        if isinstance(s, str) and s.strip():
            effective_asked.add(normalize_slot_name(s))
    if asked_slots:
        for s in asked_slots:
            if isinstance(s, str) and s.strip():
                effective_asked.add(normalize_slot_name(s))

    # Evaluate contradictions (detect if not provided)
    if contradictions is None:
        contradictions = detect_contradictions(requirements)

    # 3. Missing product category (highest conversational priority)
    if requirements.category is None:
        if not enforce_mandatory_slots and turn_count >= max_turns:
            return PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question="Unable to provide recommendations because the product category could not be determined within the clarification limit.",
                slot="category",
                reason="turn_limit_missing_category",
                metadata={"turn_count": turn_count, "max_turns": max_turns},
                can_recommend=False,
                remaining_slots=["category"],
            )
        return PolicyDecision(
            action=PolicyAction.ASK,
            question=CATEGORY_QUESTION,
            slot="category",
            reason="missing_category",
            can_recommend=False,
            remaining_slots=["category"],
        )

    # 4. Unresolved contradictions
    if len(contradictions) > 0:
        first_conflict = contradictions[0]
        clarification_q = format_contradiction_question(first_conflict, requirements)
        if not enforce_mandatory_slots and turn_count >= max_turns:
            return PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question="Unable to proceed with recommendations due to unresolved requirement contradictions after reaching the clarification limit.",
                slot="conflict",
                reason="turn_limit_unresolved_contradictions",
                contradictions=list(contradictions),
                metadata={"turn_count": turn_count, "max_turns": max_turns},
                can_recommend=False,
                remaining_slots=[],
            )
        return PolicyDecision(
            action=PolicyAction.CLARIFY_CONFLICT,
            question=clarification_q,
            slot="conflict",
            reason="unresolved_contradiction",
            contradictions=list(contradictions),
            metadata={
                "contradiction_code": first_conflict.code,
                "contradiction_fields": first_conflict.fields,
            },
            can_recommend=False,
            remaining_slots=[],
        )

    # 5. Mandatory Slot-Filling Engine
    if enforce_mandatory_slots:
        effective_status = dict(slot_status or {})
        can_rec, blocking_reason, remaining_mandatory = can_proceed_to_recommendation(
            requirements, effective_status, contradictions
        )

        if can_rec:
            return PolicyDecision(
                action=PolicyAction.RECOMMEND,
                question=None,
                slot=None,
                reason="sufficient_requirements",
                can_recommend=True,
                remaining_slots=[],
            )

        # Retrieve next best mandatory slot to clarify
        next_slot = get_next_slot_to_clarify(requirements, effective_status, list(effective_asked))
        if next_slot is not None:
            q_text = next_slot.question
            if next_slot.name == "use_cases" and requirements.category in CATEGORY_USE_CASE_QUESTIONS:
                q_text = CATEGORY_USE_CASE_QUESTIONS[requirements.category]
            elif next_slot.name == "budget" and requirements.category in CATEGORY_BUDGET_QUESTIONS:
                q_text = CATEGORY_BUDGET_QUESTIONS[requirements.category]

            return PolicyDecision(
                action=PolicyAction.ASK,
                question=q_text,
                slot=next_slot.name,
                reason=f"missing_{next_slot.name}",
                can_recommend=False,
                remaining_slots=remaining_mandatory,
            )

        # Fallback question if mandatory slot exists but was previously queried
        return PolicyDecision(
            action=PolicyAction.ASK,
            question="Could you please provide more details on your requirements so I can find the best match for you?",
            slot=remaining_mandatory[0] if remaining_mandatory else None,
            reason="missing_mandatory_slots",
            can_recommend=False,
            remaining_slots=remaining_mandatory,
        )

    # 6. Legacy Module 7 Fallback Path (for backward-compatible tests)
    # Turn limit enforcement
    if turn_count >= max_turns:
        return PolicyDecision(
            action=PolicyAction.RECOMMEND,
            question=None,
            slot=None,
            reason="turn_limit_reached",
            metadata={"turn_count": turn_count, "max_turns": max_turns},
            can_recommend=True,
            remaining_slots=[],
        )

    # Missing essential budget
    if requirements.budget_max_inr is None and "budget" not in effective_asked:
        return PolicyDecision(
            action=PolicyAction.ASK,
            question=BUDGET_QUESTION,
            slot="budget",
            reason="missing_budget",
            can_recommend=False,
            remaining_slots=["budget"],
        )

    # Missing primary use case
    if len(requirements.use_cases) == 0 and "use_cases" not in effective_asked:
        use_case_question = CATEGORY_USE_CASE_QUESTIONS.get(
            requirements.category,
            "What is your primary intended use for this device?",
        )
        return PolicyDecision(
            action=PolicyAction.ASK,
            question=use_case_question,
            slot="use_cases",
            reason="missing_use_case",
            can_recommend=False,
            remaining_slots=["use_cases"],
        )

    # Critical technical constraints (category-aware clarification)
    if requirements.category == Category.LAPTOP:
        has_ml_use_case = any(
            any(kw in uc.lower() for kw in ML_USE_CASE_KEYWORDS)
            for uc in requirements.use_cases
        )
        if (
            has_ml_use_case
            and requirements.ml_mode is None
            and "ml_mode" not in effective_asked
            and requirements.hard.requires_gpu is None
        ):
            return PolicyDecision(
                action=PolicyAction.ASK,
                question=ML_MODE_QUESTION,
                slot="ml_mode",
                reason="clarify_ml_mode",
                can_recommend=False,
                remaining_slots=["ml_mode"],
            )

    # Sufficient requirements established
    return PolicyDecision(
        action=PolicyAction.RECOMMEND,
        question=None,
        slot=None,
        reason="sufficient_requirements",
        can_recommend=True,
        remaining_slots=[],
    )
