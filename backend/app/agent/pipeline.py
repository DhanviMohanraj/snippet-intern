"""Agent Orchestration Pipeline for conversational product recommendations.

Coordinates:
    - Module 5: Requirements extraction from natural language.
    - Module 6: Multi-turn requirements merging and contradiction detection.
    - Module 7: Clarification policy and slot-filling decisions.
    - Module 8: Product candidate retrieval, filtering, and scoring.
    - Module 9: Response generation, grounded explanation, and validation.

Principle:
    The LLM understands and explains; deterministic Python code decides and enforces.
"""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.agent.extract import extract
from app.agent.followup import (
    FollowUpIntent,
    analyze_catalogue_minimum_budget,
    detect_followup_intent,
    explain_blocking_constraints,
    find_closest_alternatives,
)
from app.agent.merge import merge
from app.agent.normalize import parse_inr_budget_expression
from app.agent.policy import decide_next_action
from app.agent.ranking import search_and_rank
from app.agent.respond import generate_recommendation_response
from app.agent.slots import (
    RE_RAM,
    RE_STORAGE,
    SlotStatus,
    parse_natural_language_slot_updates,
)
from app.core.errors import InvalidMessageError
from app.core.logging import get_logger
from app.db.repositories import ProductRepository
from app.llm.client import LLMClient
from app.models.conversation import ConversationMessage, ConversationState
from app.models.product import Category
from app.models.ranking import CandidateSearchResult
from app.models.requirements import (
    CATEGORY_ALIASES,
    Contradiction,
    HardConstraints,
    Message,
    PolicyAction,
    PolicyDecision,
    Requirements,
    RequirementsDelta,
)
from app.models.response import ProductRecommendation, RecommendationResponse

logger = get_logger("app.agent.pipeline")

# Known unsupported categories to detect and cleanly redirect or reject
UNSUPPORTED_CATEGORIES = {
    "tv",
    "television",
    "smartwatch",
    "watch",
    "refrigerator",
    "fridge",
    "camera",
    "dslr",
    "drone",
    "printer",
    "speaker",
    "soundbar",
    "console",
    "playstation",
    "xbox",
    "ac",
    "air conditioner",
}


def _extract_unsupported_category_mention(text: str) -> str | None:
    """Check if user text explicitly mentions an unsupported electronics category."""
    text_lower = text.lower()
    for cat in UNSUPPORTED_CATEGORIES:
        # Match whole words only
        if re.search(rf"\b{re.escape(cat)}\b", text_lower):
            return cat
    return None


def _build_fallback_delta(text: str) -> RequirementsDelta:
    """Deterministic rule-based fallback extractor when LLM extraction fails."""
    text_lower = text.lower()
    cat: Category | None = None
    for alias, c in CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text_lower):
            cat = c
            break

    # Attempt to extract budget mention (e.g., under 50k, 50000)
    budget: int | None = None
    budget_match = re.search(r"(?:under|below|budget\s*(?:is|of)?|max|up to|rs\.?|inr|₹)\s*(\d[\d,]*\s*(?:k|lakh|thousand)?)", text_lower)
    if budget_match:
        budget = parse_inr_budget_expression(budget_match.group(1))

    # Extract use cases
    use_cases: list[str] = []
    known_use_cases = [
        ("music", "music"),
        ("classes", "classes"),
        ("class", "classes"),
        ("calls", "calls"),
        ("call", "calls"),
        ("calling", "calls"),
        ("gaming", "gaming"),
        ("programming", "programming"),
        ("coding", "programming"),
        ("editing", "video editing"),
        ("studying", "studying"),
        ("study", "studying"),
        ("travel", "travel"),
    ]
    for kw, label in known_use_cases:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            if label not in use_cases:
                use_cases.append(label)

    hard_kwargs: dict[str, Any] = {}
    ram_match = RE_RAM.search(text_lower)
    if ram_match:
        hard_kwargs["min_ram_gb"] = int(ram_match.group(1))

    storage_match = RE_STORAGE.search(text_lower)
    if storage_match:
        val_str = storage_match.group(1) or storage_match.group(2)
        if val_str:
            val = int(val_str)
            if "tb" in storage_match.group(0).lower():
                val = val * 1024
            hard_kwargs["min_storage_gb"] = val

    if re.search(r"\b(over[-\s]?ear|on[-\s]?ear|overhead)\b", text_lower) and not re.search(r"\b(in[-\s]?ear|tws|earbuds?)\b", text_lower):
        hard_kwargs["form_factor"] = "over-ear"
    elif re.search(r"\b(in[-\s]?ear|tws|earbuds?|earphones?)\b", text_lower) and not re.search(r"\b(over[-\s]?ear|on[-\s]?ear)\b", text_lower):
        hard_kwargs["form_factor"] = "in-ear"

    if re.search(r"\b(wireless|bluetooth|true\s+wireless)\b", text_lower):
        hard_kwargs["requires_wireless"] = True

    prefs: dict[str, float] = {}
    if re.search(r"\b(microphone|mic|voice\s*calls?|clear\s*vocals?)\b", text_lower):
        prefs["microphone_quality"] = 0.85
    if re.search(r"\b(comfort|comfortable|cushions?|ear\s*cushions?|hours\s*(?:daily|a\s*day))\b", text_lower):
        prefs["comfort"] = 0.85
    if re.search(r"\b(battery|battery\s*life|endurance)\b", text_lower):
        prefs["battery_life"] = 0.85
    if re.search(r"\b(anc|noise\s+cancellation|noise\s+cancelling)\b", text_lower):
        prefs["noise_cancellation"] = 0.85

    hard = HardConstraints(**hard_kwargs) if hard_kwargs else None
    return RequirementsDelta(
        category=cat,
        budget_max_inr=budget,
        use_cases=use_cases if use_cases else None,
        hard=hard,
        preferences=prefs if prefs else None,
    )


class PipelineResult(BaseModel):
    """Structured result returned by the agent orchestration pipeline."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    updated_conversation: ConversationState = Field(description="Updated conversation state")
    decision: PolicyDecision = Field(description="Policy decision executed on this turn")
    assistant_message: str = Field(description="User-facing response message from assistant")
    recommendations: list[ProductRecommendation] = Field(
        default_factory=list, description="Ranked product recommendations if action is RECOMMEND"
    )
    recommendation_response: RecommendationResponse | None = Field(
        default=None, description="Module 9 recommendation envelope if recommendations were generated"
    )
    unresolved_contradictions: list[Contradiction] = Field(
        default_factory=list, description="Active contradictions if any"
    )
    is_fallback: bool = Field(
        default=False, description="Whether fallback logic was engaged during extraction or recommendation"
    )


async def run_pipeline(
    user_message: str,
    conversation: ConversationState,
    llm_client: LLMClient,
    product_repo: ProductRepository,
) -> PipelineResult:
    """Execute the end-to-end multi-module conversational recommendation pipeline.

    Args:
        user_message: The raw input message text from the user.
        conversation: Current persistent ConversationState.
        llm_client: Injected LLM client for extraction and explanation generation.
        product_repo: Injected ProductRepository for candidate retrieval.

    Returns:
        PipelineResult containing updated conversation state, decisions, and recommendations.

    Raises:
        InvalidMessageError: If message is empty or exceeds 1000 characters.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise InvalidMessageError("User message cannot be empty or whitespace only.")

    cleaned_message = user_message.strip()
    if len(cleaned_message) > 1000:
        raise InvalidMessageError(
            f"User message exceeds maximum permitted length of 1000 characters (received {len(cleaned_message)})."
        )

    is_fallback: bool = False

    # 1. Convert recent conversation history (max 6 messages) for context
    recent_messages = [
        Message(role=m.role, content=m.content)
        for m in conversation.messages[-6:]
    ]

    # 2. Extract rule-based natural language slot updates (you decide, skip, specs)
    nl_status_updates, nl_delta = parse_natural_language_slot_updates(
        text=cleaned_message,
        last_asked_slot=conversation.last_asked_slot,
        current_req=conversation.requirements,
    )

    # 3. Extract requirements delta using Module 5 LLM extractor
    try:
        llm_delta = await extract(
            message=cleaned_message,
            current=conversation.requirements,
            recent=recent_messages,
            llm=llm_client,
        )
    except Exception as err:
        logger.warning(
            "LLM requirements extraction encountered error: %s. Using deterministic fallback extraction.",
            err,
        )
        is_fallback = True
        llm_delta = _build_fallback_delta(cleaned_message)

    # Merge nl_delta and llm_delta
    # Prioritize concrete specs and preferences from either source
    combined_delta_dict = llm_delta.model_dump()
    if nl_delta.category is not None and combined_delta_dict.get("category") is None:
        combined_delta_dict["category"] = nl_delta.category
    if nl_delta.budget_max_inr is not None and combined_delta_dict.get("budget_max_inr") is None:
        combined_delta_dict["budget_max_inr"] = nl_delta.budget_max_inr
    if nl_delta.use_cases:
        existing_ucs = combined_delta_dict.get("use_cases") or []
        for uc in nl_delta.use_cases:
            if uc not in existing_ucs:
                existing_ucs.append(uc)
        combined_delta_dict["use_cases"] = existing_ucs
    if nl_delta.hard is not None:
        if combined_delta_dict.get("hard") is None:
            combined_delta_dict["hard"] = nl_delta.hard.model_dump()
        else:
            for k, v in nl_delta.hard.model_dump().items():
                if v is not None:
                    combined_delta_dict["hard"][k] = v
    if nl_delta.preferences:
        existing_prefs = combined_delta_dict.get("preferences") or {}
        for k, v in nl_delta.preferences.items():
            if k not in existing_prefs:
                existing_prefs[k] = v
        combined_delta_dict["preferences"] = existing_prefs

    combined_delta = RequirementsDelta(**combined_delta_dict)

    # 4. Merge combined delta into accumulated requirements using Module 6
    merged_requirements, contradictions = merge(conversation.requirements, combined_delta)

    # If category changed, reset category-specific slot statuses
    if (
        conversation.requirements.category is not None
        and merged_requirements.category is not None
        and merged_requirements.category != conversation.requirements.category
    ):
        # Keep universal slots (category, budget, use_cases), reset category-specific slots
        universal = {"category", "budget", "use_cases"}
        conversation.slot_status = {
            k: v for k, v in conversation.slot_status.items() if k in universal
        }

    # Update slot_status with new updates
    conversation.slot_status.update(nl_status_updates)

    # Automatically mark slots as ANSWERED if values are now concrete in requirements
    if merged_requirements.category is not None:
        conversation.slot_status["category"] = SlotStatus.ANSWERED.value
    if merged_requirements.budget_max_inr is not None:
        conversation.slot_status["budget"] = SlotStatus.ANSWERED.value
    if len(merged_requirements.use_cases) > 0:
        conversation.slot_status["use_cases"] = SlotStatus.ANSWERED.value
    if merged_requirements.hard.min_ram_gb is not None:
        conversation.slot_status["ram"] = SlotStatus.ANSWERED.value
    if merged_requirements.hard.min_storage_gb is not None:
        conversation.slot_status["storage"] = SlotStatus.ANSWERED.value
        conversation.slot_status["phone_storage"] = SlotStatus.ANSWERED.value
        conversation.slot_status["tablet_storage"] = SlotStatus.ANSWERED.value
    if merged_requirements.hard.requires_gpu is not None:
        conversation.slot_status["gpu"] = SlotStatus.ANSWERED.value
    if merged_requirements.ml_mode is not None:
        conversation.slot_status["ml_mode"] = SlotStatus.ANSWERED.value
    if merged_requirements.hard.form_factor is not None:
        conversation.slot_status["form_factor"] = SlotStatus.ANSWERED.value

    # 5. Check for explicitly mentioned unsupported categories
    unsupported_cat: str | None = None
    if merged_requirements.category is None and combined_delta.category is None:
        unsupported_cat = _extract_unsupported_category_mention(cleaned_message)

    # 5b. Follow-Up Intent Recognition (e.g. "what is the minimum budget", "which requirement is the problem", "closest alternatives")
    followup_intent = detect_followup_intent(cleaned_message, conversation)
    recommendations: list[ProductRecommendation] = []
    rec_response: RecommendationResponse | None = None
    assistant_message: str

    if followup_intent is not None:
        turn_count = conversation.turn_count + 1
        if merged_requirements.category is None:
            followup_message = (
                "Please specify a product category (e.g. laptop, phone, headphones) "
                "so I can check pricing and constraints in our catalogue."
            )
            decision = PolicyDecision(
                action=PolicyAction.ASK,
                question=followup_message,
                slot="category",
                reason="missing_category_for_followup",
                can_recommend=False,
                remaining_slots=["category"],
            )
            conversation.last_asked_slot = "category"
        elif followup_intent == FollowUpIntent.MINIMUM_BUDGET:
            followup_message = await analyze_catalogue_minimum_budget(
                requirements=merged_requirements,
                product_repo=product_repo,
            )
            decision = PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question=followup_message,
                slot="budget",
                reason="followup_minimum_budget",
                can_recommend=False,
                remaining_slots=[],
            )
            conversation.last_asked_slot = "budget"
        elif followup_intent == FollowUpIntent.EXPLAIN_BLOCKING:
            followup_message = await explain_blocking_constraints(
                requirements=merged_requirements,
                product_repo=product_repo,
            )
            decision = PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question=followup_message,
                slot="conflict",
                reason="followup_explain_blocking",
                can_recommend=False,
                remaining_slots=[],
            )
            conversation.last_asked_slot = "conflict"
        elif followup_intent == FollowUpIntent.CLOSEST_ALTERNATIVES:
            followup_message = await find_closest_alternatives(
                requirements=merged_requirements,
                product_repo=product_repo,
            )
            decision = PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question=followup_message,
                slot="budget",
                reason="followup_closest_alternatives",
                can_recommend=False,
                remaining_slots=[],
            )
            conversation.last_asked_slot = "budget"
        else:
            followup_message = "How can I assist you with your electronics search today?"
            decision = PolicyDecision(
                action=PolicyAction.NO_MATCH,
                question=followup_message,
                slot="budget",
                reason="followup_general",
                can_recommend=False,
                remaining_slots=[],
            )

        assistant_message = followup_message
    else:
        # 6. Evaluate policy action using Module 7 with mandatory slot enforcement
        turn_count = conversation.turn_count + 1
        decision = decide_next_action(
            requirements=merged_requirements,
            contradictions=contradictions,
            asked_slots=conversation.asked_slots,
            turn_count=turn_count,
            is_off_topic=combined_delta.is_off_topic,
            unsupported_category=unsupported_cat,
            slot_status=conversation.slot_status,
            enforce_mandatory_slots=True,
        )

        # 7. Execute policy action (Recommendation Gate enforces no premature recommendations)
        if decision.action == PolicyAction.RECOMMEND and decision.can_recommend:
            # Step 7a: Retrieve and rank candidates via Module 8
            search_result = await search_and_rank(
                requirements=merged_requirements,
                repository=product_repo,
                top_n=5,
            )

            # Step 7b: Generate grounded response via Module 9
            rec_response = await generate_recommendation_response(
                requirements=merged_requirements,
                search_result=search_result,
                llm_client=llm_client,
            )

            assistant_message = rec_response.message
            recommendations = rec_response.recommendations
            is_fallback = is_fallback or rec_response.is_fallback
            conversation.last_asked_slot = None

            # Critical: If no products matched hard constraints, convert to NO_MATCH action
            if not search_result.has_matches or not recommendations:
                failed_names: set[str] = set()
                for ex in search_result.excluded_candidates:
                    for chk in ex.failed_constraints:
                        failed_names.add(chk.name)

                cat_label = merged_requirements.category.value if merged_requirements.category else "product"
                if "form_factor" in failed_names:
                    q = f"Would you like to adjust your budget or consider relaxing the form factor requirement for {cat_label}s?"
                elif "min_storage_gb" in failed_names:
                    q = f"Would you like to adjust your budget or consider relaxing your storage requirement?"
                elif "requires_gpu" in failed_names:
                    q = f"Would you like to adjust your budget or consider models with integrated graphics?"
                elif "requires_stylus" in failed_names:
                    q = f"Would you like to adjust your budget or consider tablets without stylus support?"
                elif "min_refresh_hz" in failed_names:
                    q = f"Would you like to adjust your budget or consider a lower refresh rate?"
                else:
                    q = f"Would you like to increase your budget or adjust any hardware specifications for {cat_label}s?"

                decision = PolicyDecision(
                    action=PolicyAction.NO_MATCH,
                    question=q,
                    slot="budget",
                    reason="no_candidates_matched",
                    can_recommend=False,
                    remaining_slots=[],
                )
                conversation.last_asked_slot = "budget"

        elif decision.action == PolicyAction.ASK:
            assistant_message = decision.question or "Could you provide more details about your requirements?"
            conversation.last_asked_slot = decision.slot
            if decision.slot and decision.slot not in conversation.asked_slots:
                conversation.asked_slots.append(decision.slot)

        elif decision.action == PolicyAction.CLARIFY_CONFLICT:
            assistant_message = decision.question or "Please clarify your contradictory requirements."
            conversation.last_asked_slot = "conflict"

        elif decision.action == PolicyAction.REDIRECT:
            assistant_message = decision.question or "I am an electronics recommendation assistant. How can I help you today?"
            conversation.last_asked_slot = None

        elif decision.action == PolicyAction.NO_MATCH:
            assistant_message = decision.question or "No matching products could be found matching your requirements."
            conversation.last_asked_slot = None

        else:
            assistant_message = "How can I assist you with your electronics search today?"
            conversation.last_asked_slot = None

    # 8. Update conversation state and append chronological messages
    conversation.requirements = merged_requirements
    conversation.turn_count = turn_count

    user_msg_record = ConversationMessage(
        role="user",
        content=cleaned_message,
    )
    assistant_msg_record = ConversationMessage(
        role="assistant",
        content=assistant_message,
        action=decision.action,
        recommendations=recommendations,
        clarification_question=decision.question if decision.is_ask or decision.is_clarify_conflict else None,
        contradictions=list(contradictions),
        is_fallback=is_fallback,
    )

    conversation.messages.append(user_msg_record)
    conversation.messages.append(assistant_msg_record)

    return PipelineResult(
        updated_conversation=conversation,
        decision=decision,
        assistant_message=assistant_message,
        recommendations=recommendations,
        recommendation_response=rec_response,
        unresolved_contradictions=list(contradictions),
        is_fallback=is_fallback,
    )
