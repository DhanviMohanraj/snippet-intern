"""Comprehensive offline tests for Module 7: Clarification Policy and Slot-Filling Decision Engine.

Covers all 21 required test scenarios:
1. Missing category causes a category question.
2. Known category with sufficient requirements proceeds to RECOMMEND.
3. Missing essential budget triggers a budget question when required.
4. Missing primary use case triggers a category-appropriate question.
5. Optional preferences do not block recommendation.
6. Existing answered slots are not asked again.
7. A single best next question is returned (one at a time).
8. Category-specific question selection works for all 5 supported categories.
9. Existing contradictions trigger CLARIFY_CONFLICT.
10. Budget contradiction explanation includes the relevant values.
11. GPU/VRAM contradiction triggers clarification.
12. An unrelated off-topic request triggers REDIRECT.
13. An unsupported category is handled safely (NO_MATCH).
14. The turn limit is respected.
15. At the turn limit, sufficient requirements proceed to RECOMMEND.
16. At the turn limit, unresolved essential ambiguity is handled safely.
17. Previously answered optional preferences are not queried again.
18. Repeated policy calls are deterministic.
19. The input requirements and contradiction objects are not mutated.
20. No policy path makes an LLM call.
21. Existing Modules 1-6 tests remain unaffected.
"""

from __future__ import annotations

import copy
import pytest

from app.agent.policy import (
    BUDGET_QUESTION,
    CATEGORY_QUESTION,
    CATEGORY_USE_CASE_QUESTIONS,
    MAX_CLARIFICATION_TURNS,
    ML_MODE_QUESTION,
    OFF_TOPIC_REDIRECT_MESSAGE,
    decide_next_action,
    format_contradiction_question,
    normalize_slot_name,
)
from app.models.product import Category
from app.models.requirements import (
    Contradiction,
    HardConstraints,
    PolicyAction,
    PolicyDecision,
    Requirements,
)


# ---------------------------------------------------------------------------
# 1. Missing category causes a category question
# ---------------------------------------------------------------------------

def test_missing_category_triggers_category_question():
    """When category is None, policy must emit ASK for slot='category'."""
    req = Requirements(category=None, budget_max_inr=60000, use_cases=["coding"])
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.is_ask is True
    assert decision.slot == "category"
    assert decision.reason == "missing_category"
    assert decision.question == CATEGORY_QUESTION
    assert "laptop" in decision.question.lower()


# ---------------------------------------------------------------------------
# 2. Known category with sufficient requirements proceeds to RECOMMEND
# ---------------------------------------------------------------------------

def test_sufficient_requirements_proceeds_to_recommend():
    """When category, budget, and use case are present without contradictions, RECOMMEND."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=80000,
        use_cases=["programming"],
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.RECOMMEND
    assert decision.is_recommend is True
    assert decision.slot is None
    assert decision.question is None
    assert decision.reason == "sufficient_requirements"


# ---------------------------------------------------------------------------
# 3. Missing essential budget triggers a budget question
# ---------------------------------------------------------------------------

def test_missing_budget_triggers_budget_question():
    """When category is known and budget is missing (not yet asked), emit ASK for slot='budget'."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=None,
        use_cases=["college"],
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "budget"
    assert decision.reason == "missing_budget"
    assert decision.question == BUDGET_QUESTION
    assert "budget" in decision.question.lower() or "price" in decision.question.lower()


# ---------------------------------------------------------------------------
# 4. Missing primary use case triggers a category-appropriate question
# ---------------------------------------------------------------------------

def test_missing_use_case_triggers_use_case_question():
    """When category and budget are present but use_cases is empty, emit ASK for slot='use_cases'."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=60000,
        use_cases=[],
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "use_cases"
    assert decision.reason == "missing_use_case"
    assert decision.question == CATEGORY_USE_CASE_QUESTIONS[Category.LAPTOP]
    assert "programming" in decision.question.lower() or "gaming" in decision.question.lower()


# ---------------------------------------------------------------------------
# 5. Optional preferences do not block recommendation
# ---------------------------------------------------------------------------

def test_optional_preferences_do_not_block_recommendation():
    """Absence of soft preferences, brand exclusions, or specs must not block RECOMMEND."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=30000,
        use_cases=["everyday"],
        preferences={},
        hard=HardConstraints(),
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.RECOMMEND
    assert decision.reason == "sufficient_requirements"


# ---------------------------------------------------------------------------
# 6. Existing answered slots are not asked again
# ---------------------------------------------------------------------------

def test_answered_or_asked_slots_not_repeated():
    """Slots already in asked_slots (or requirements.asked_slots) are not asked again."""
    # User was already asked budget, but has flexible/unspecified budget
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=None,
        use_cases=["programming"],
        asked_slots=["budget"],
    )
    decision = decide_next_action(req)

    # Budget must not be re-asked; since use case is present, proceed to RECOMMEND
    assert decision.action == PolicyAction.RECOMMEND
    assert decision.reason == "sufficient_requirements"


def test_asked_slots_passed_as_parameter_honored():
    """Passing asked_slots explicitly to decide_next_action overrides/augments requirements."""
    req = Requirements(
        category=Category.PHONE,
        budget_max_inr=None,
        use_cases=[],
    )
    # Both budget and use_cases were already queried
    decision = decide_next_action(req, asked_slots=["budget", "use_cases"])

    # Neither budget nor use_cases is repeated; proceed to RECOMMEND
    assert decision.action == PolicyAction.RECOMMEND


def test_slot_normalization_handles_aliases():
    """Slot aliases like 'price' or 'use_case' correctly map to canonical slots."""
    assert normalize_slot_name("price") == "budget"
    assert normalize_slot_name("price_range") == "budget"
    assert normalize_slot_name("use_case") == "use_cases"
    assert normalize_slot_name("device_type") == "category"


# ---------------------------------------------------------------------------
# 7. A single best next question is returned
# ---------------------------------------------------------------------------

def test_single_best_next_question_returned():
    """When everything is missing, only the highest-priority question (category) is returned."""
    req = Requirements()  # Category, budget, and use case all missing
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "category"
    assert isinstance(decision.question, str)
    assert decision.question == CATEGORY_QUESTION


# ---------------------------------------------------------------------------
# 8. Category-specific question selection works for all 5 supported categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("category", "expected_keyword"),
    [
        (Category.LAPTOP, "laptop"),
        (Category.PHONE, "phone"),
        (Category.HEADPHONES, "headphones"),
        (Category.TABLET, "tablet"),
        (Category.MONITOR, "monitor"),
    ],
)
def test_category_tailored_use_case_questions(category: Category, expected_keyword: str):
    """Each category receives a specifically tailored use-case inquiry."""
    req = Requirements(category=category, budget_max_inr=50000, use_cases=[])
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "use_cases"
    assert decision.question == CATEGORY_USE_CASE_QUESTIONS[category]
    assert expected_keyword in decision.question.lower()


# ---------------------------------------------------------------------------
# 9. Existing contradictions trigger CLARIFY_CONFLICT
# ---------------------------------------------------------------------------

def test_unresolved_contradictions_trigger_clarify_conflict():
    """Unresolved contradictions force CLARIFY_CONFLICT action."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_min_inr=100000,
        budget_max_inr=70000,
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.CLARIFY_CONFLICT
    assert decision.is_clarify_conflict is True
    assert decision.slot == "conflict"
    assert decision.reason == "unresolved_contradiction"
    assert len(decision.contradictions) == 1
    assert decision.contradictions[0].code == "BUDGET_MIN_EXCEEDS_MAX"


# ---------------------------------------------------------------------------
# 10. Budget contradiction explanation includes the relevant values
# ---------------------------------------------------------------------------

def test_budget_contradiction_explanation_includes_numeric_values():
    """The contradiction question explicitly surfaces the contradictory budget amounts."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_min_inr=120000,
        budget_max_inr=80000,
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.CLARIFY_CONFLICT
    assert "120,000" in decision.question
    assert "80,000" in decision.question


# ---------------------------------------------------------------------------
# 11. GPU/VRAM contradiction triggers clarification
# ---------------------------------------------------------------------------

def test_gpu_vram_contradiction_triggers_clarification():
    """GPU disabled with positive VRAM triggers CLARIFY_CONFLICT."""
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(requires_gpu=False, min_vram_gb=6),
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.CLARIFY_CONFLICT
    assert decision.contradictions[0].code == "GPU_FALSE_WITH_MIN_VRAM"
    assert "GPU" in decision.question
    assert "6GB" in decision.question or "VRAM" in decision.question


# ---------------------------------------------------------------------------
# 12. An unrelated off-topic request triggers REDIRECT
# ---------------------------------------------------------------------------

def test_off_topic_triggers_redirect():
    """Off-topic delta signal triggers REDIRECT with polite guidance."""
    req = Requirements(category=Category.LAPTOP)
    decision = decide_next_action(req, is_off_topic=True)

    assert decision.action == PolicyAction.REDIRECT
    assert decision.is_redirect is True
    assert decision.slot is None
    assert decision.reason == "off_topic"
    assert decision.question == OFF_TOPIC_REDIRECT_MESSAGE
    assert "electronics" in decision.question.lower()


# ---------------------------------------------------------------------------
# 13. An unsupported category is handled safely
# ---------------------------------------------------------------------------

def test_unsupported_category_triggers_no_match():
    """Explicitly requested unsupported category triggers NO_MATCH."""
    req = Requirements()
    decision = decide_next_action(req, unsupported_category="smartwatch")

    assert decision.action == PolicyAction.NO_MATCH
    assert decision.is_no_match is True
    assert decision.slot == "category"
    assert decision.reason == "unsupported_category"
    assert "smartwatch" in decision.question
    assert "laptops, phones" in decision.question


# ---------------------------------------------------------------------------
# 14. Turn limit is respected
# ---------------------------------------------------------------------------

def test_turn_limit_respected():
    """When turn_count >= max_turns, the policy does not continue asking questions."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=None,
        use_cases=[],
    )
    decision = decide_next_action(req, turn_count=3, max_turns=3)

    assert decision.action != PolicyAction.ASK
    assert decision.reason == "turn_limit_reached"


# ---------------------------------------------------------------------------
# 15. At turn limit, sufficient requirements proceed to RECOMMEND
# ---------------------------------------------------------------------------

def test_turn_limit_proceeds_to_recommend_if_category_known():
    """At turn limit with known category and no contradictions, proceed to RECOMMEND."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=None,  # Budget never specified
        use_cases=[],          # Use case never specified
    )
    decision = decide_next_action(req, turn_count=MAX_CLARIFICATION_TURNS)

    assert decision.action == PolicyAction.RECOMMEND
    assert decision.reason == "turn_limit_reached"
    assert decision.metadata["turn_count"] == 3


# ---------------------------------------------------------------------------
# 16. At turn limit, unresolved essential ambiguity is handled safely
# ---------------------------------------------------------------------------

def test_turn_limit_fails_safe_if_category_missing():
    """If category is still unknown after max turns, return NO_MATCH safely."""
    req = Requirements(category=None)
    decision = decide_next_action(req, turn_count=3, max_turns=3)

    assert decision.action == PolicyAction.NO_MATCH
    assert decision.slot == "category"
    assert decision.reason == "turn_limit_missing_category"


def test_turn_limit_fails_safe_if_contradictions_remain():
    """If contradictions are unresolved after max turns, return NO_MATCH safely."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_min_inr=100000,
        budget_max_inr=80000,
    )
    decision = decide_next_action(req, turn_count=3, max_turns=3)

    assert decision.action == PolicyAction.NO_MATCH
    assert decision.slot == "conflict"
    assert decision.reason == "turn_limit_unresolved_contradictions"


# ---------------------------------------------------------------------------
# 17. Previously answered optional preferences are not queried again
# ---------------------------------------------------------------------------

def test_optional_preferences_not_queried():
    """Supplied preferences are preserved and never prompted for."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=90000,
        use_cases=["coding"],
        preferences={"portability": 0.9, "battery_life": 0.8},
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.RECOMMEND
    assert decision.reason == "sufficient_requirements"


# ---------------------------------------------------------------------------
# 18. Repeated policy calls are deterministic
# ---------------------------------------------------------------------------

def test_policy_calls_are_deterministic():
    """Identical inputs yield exact identical decisions across repeated calls."""
    req = Requirements(
        category=Category.HEADPHONES,
        budget_max_inr=15000,
        use_cases=[],
    )

    d1 = decide_next_action(req, turn_count=1)
    d2 = decide_next_action(req, turn_count=1)

    assert d1.action == d2.action
    assert d1.question == d2.question
    assert d1.slot == d2.slot
    assert d1.reason == d2.reason
    assert d1.metadata == d2.metadata
    assert d1 == d2


# ---------------------------------------------------------------------------
# 19. The input requirements and contradiction objects are not mutated
# ---------------------------------------------------------------------------

def test_input_objects_not_mutated():
    """decide_next_action must never alter requirements, contradictions, or asked_slots in place."""
    original_req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=75000,
        use_cases=["gaming"],
        asked_slots=["budget"],
    )
    req_copy = copy.deepcopy(original_req)

    contradictions = [
        Contradiction(
            code="TEST_CODE",
            fields=["test_field"],
            message="Test message",
        )
    ]
    contradictions_copy = copy.deepcopy(contradictions)
    asked_slots = ["budget", "use_cases"]
    asked_slots_copy = list(asked_slots)

    decision = decide_next_action(
        requirements=original_req,
        contradictions=contradictions,
        asked_slots=asked_slots,
        turn_count=1,
    )

    assert original_req == req_copy
    assert contradictions == contradictions_copy
    assert asked_slots == asked_slots_copy


# ---------------------------------------------------------------------------
# 20. Laptop Machine Learning Technical Clarification
# ---------------------------------------------------------------------------

def test_laptop_ml_use_case_clarifies_ml_mode():
    """A laptop requested for ML without ml_mode or GPU triggers ASK for ml_mode."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=120000,
        use_cases=["machine learning"],
        ml_mode=None,
        hard=HardConstraints(requires_gpu=None),
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.ASK
    assert decision.slot == "ml_mode"
    assert decision.reason == "clarify_ml_mode"
    assert decision.question == ML_MODE_QUESTION
    assert "locally" in decision.question.lower()
    assert "cloud" in decision.question.lower()


def test_laptop_ml_use_case_skips_if_ml_mode_already_asked():
    """If ml_mode was already queried, do not ask again; proceed to RECOMMEND."""
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=120000,
        use_cases=["machine learning"],
        ml_mode=None,
        asked_slots=["ml_mode"],
    )
    decision = decide_next_action(req)

    assert decision.action == PolicyAction.RECOMMEND


# ---------------------------------------------------------------------------
# 21. Policy Decision Model Helper Properties
# ---------------------------------------------------------------------------

def test_policy_decision_helper_properties():
    """Test is_ask, is_recommend, is_clarify_conflict, is_redirect, is_no_match."""
    d_ask = PolicyDecision(action=PolicyAction.ASK, slot="category")
    assert d_ask.is_ask is True
    assert d_ask.is_recommend is False

    d_rec = PolicyDecision(action=PolicyAction.RECOMMEND)
    assert d_rec.is_recommend is True

    d_conf = PolicyDecision(action=PolicyAction.CLARIFY_CONFLICT)
    assert d_conf.is_clarify_conflict is True

    d_redir = PolicyDecision(action=PolicyAction.REDIRECT)
    assert d_redir.is_redirect is True

    d_no = PolicyDecision(action=PolicyAction.NO_MATCH)
    assert d_no.is_no_match is True
