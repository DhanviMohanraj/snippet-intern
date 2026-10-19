"""Handwritten extraction examples for natural-language requirement testing.

Contains 15 canonical user messages representing real-world user interactions,
including budget expressions (INR / k / lakh / hazaar), multi-turn adjustments,
spelling errors, Hinglish expressions, preference removals, and off-topic queries.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from app.models.product import Category
from app.models.requirements import (
    HardConstraints,
    Message,
    Requirements,
    RequirementsDelta,
)


class ExtractionExample(NamedTuple):
    id: int
    description: str
    message: str
    current: Requirements
    recent: list[Message]
    raw_llm_payload: dict[str, Any]
    expected_delta: RequirementsDelta


EXTRACTION_EXAMPLES: list[ExtractionExample] = [
    # 1. Explicit max budget
    ExtractionExample(
        id=1,
        description="Laptop request with an explicit maximum budget",
        message="I need a laptop under 50000 rupees",
        current=Requirements(),
        recent=[],
        raw_llm_payload={"category": "laptop", "budget_max_inr": 50000},
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            budget_max_inr=50000,
        ),
    ),
    # 2. Budget expressed as 80k
    ExtractionExample(
        id=2,
        description="Budget expressed as 80k",
        message="Looking for a laptop around 80k",
        current=Requirements(),
        recent=[],
        raw_llm_payload={"category": "laptop", "budget_max_inr": 80000},
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            budget_max_inr=80000,
        ),
    ),
    # 3. Budget expressed as 1.2 lakh
    ExtractionExample(
        id=3,
        description="Budget expressed as 1.2 lakh",
        message="Suggest a high-end phone under 1.2 lakh",
        current=Requirements(),
        recent=[],
        raw_llm_payload={"category": "phone", "budget_max_inr": 120000},
        expected_delta=RequirementsDelta(
            category=Category.PHONE,
            budget_max_inr=120000,
        ),
    ),
    # 4. Minimum and maximum budget
    ExtractionExample(
        id=4,
        description="Request with a minimum and maximum budget",
        message="Looking for a monitor between 15000 and 25000",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "monitor",
            "budget_min_inr": 15000,
            "budget_max_inr": 25000,
        },
        expected_delta=RequirementsDelta(
            category=Category.MONITOR,
            budget_min_inr=15000,
            budget_max_inr=25000,
        ),
    ),
    # 5. Programming laptop with minimum RAM requirement
    ExtractionExample(
        id=5,
        description="Programming laptop with minimum RAM requirement",
        message="Need a laptop for programming with at least 16GB RAM",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "laptop",
            "use_cases": ["programming"],
            "hard": {"min_ram_gb": 16},
        },
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            use_cases=["programming"],
            hard=HardConstraints(min_ram_gb=16),
        ),
    ),
    # 6. Machine-learning laptop requiring a GPU
    ExtractionExample(
        id=6,
        description="Machine-learning laptop requiring a GPU",
        message="I want a machine learning laptop with a dedicated GPU",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "laptop",
            "use_cases": ["machine_learning"],
            "hard": {"requires_gpu": True},
            "ml_mode": "local",
        },
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            use_cases=["machine_learning"],
            hard=HardConstraints(requires_gpu=True),
            ml_mode="local",
        ),
    ),
    # 7. Laptop request emphasizing portability
    ExtractionExample(
        id=7,
        description="Laptop request emphasizing portability",
        message="I need a very lightweight laptop for frequent travel",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "laptop",
            "use_cases": ["travel"],
            "preferences": {"portability": 0.9},
        },
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            use_cases=["travel"],
            preferences={"portability": 0.9},
        ),
    ),
    # 8. Smartphone request emphasizing camera quality
    ExtractionExample(
        id=8,
        description="Smartphone request emphasizing camera quality",
        message="Looking for a phone with a great camera under 40k",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "phone",
            "budget_max_inr": 40000,
            "preferences": {"camera_quality": 0.8},
        },
        expected_delta=RequirementsDelta(
            category=Category.PHONE,
            budget_max_inr=40000,
            preferences={"camera_quality": 0.8},
        ),
    ),
    # 9. Headphone request specifying ANC and battery life
    ExtractionExample(
        id=9,
        description="Headphone request specifying ANC and battery life",
        message="I need headphones with ANC and at least 30 hours battery",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "headphones",
            "hard": {"min_battery_hr": 30.0},
            "preferences": {"noise_cancellation": 0.8},
        },
        expected_delta=RequirementsDelta(
            category=Category.HEADPHONES,
            hard=HardConstraints(min_battery_hr=30.0),
            preferences={"noise_cancellation": 0.8},
        ),
    ),
    # 10. Tablet request for studying
    ExtractionExample(
        id=10,
        description="Tablet request for studying",
        message="Need a tablet for college study and taking notes under 30k",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "tablet",
            "use_cases": ["college", "study", "note_taking"],
            "budget_max_inr": 30000,
        },
        expected_delta=RequirementsDelta(
            category=Category.TABLET,
            use_cases=["student", "study", "note_taking"],
            budget_max_inr=30000,
        ),
    ),
    # 11. Common spelling mistakes
    ExtractionExample(
        id=11,
        description="Message containing common spelling mistakes",
        message="Sugest me a gamming lappy under 80k with 16 GB RAM plz",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "lappy",
            "use_cases": ["gamming"],
            "budget_max_inr": 80000,
            "hard": {"min_ram_gb": 16},
        },
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            use_cases=["gaming"],
            budget_max_inr=80000,
            hard=HardConstraints(min_ram_gb=16),
        ),
    ),
    # 12. Mixed Hindi-English request using hazaar or lakh
    ExtractionExample(
        id=12,
        description="Mixed Hindi-English request using hazaar or lakh",
        message="Mujhe coding ke liye laptop chahiye, budget 70 hazaar",
        current=Requirements(),
        recent=[],
        raw_llm_payload={
            "category": "laptop",
            "use_cases": ["coding"],
            "budget_max_inr": 70000,
        },
        expected_delta=RequirementsDelta(
            category=Category.LAPTOP,
            use_cases=["programming"],
            budget_max_inr=70000,
        ),
    ),
    # 13. Changing an existing budget
    ExtractionExample(
        id=13,
        description="Message changing an existing budget",
        message="Actually, increase my budget to 90k",
        current=Requirements(category=Category.LAPTOP, budget_max_inr=70000),
        recent=[
            Message(role="user", content="Looking for laptop under 70k"),
            Message(role="assistant", content="Found models under 70,000 INR."),
        ],
        raw_llm_payload={"budget_max_inr": 90000},
        expected_delta=RequirementsDelta(budget_max_inr=90000),
    ),
    # 14. Explicit request to remove an existing preference
    ExtractionExample(
        id=14,
        description="Explicit request to remove an existing preference",
        message="I don't need the portability preference anymore",
        current=Requirements(category=Category.LAPTOP, preferences={"portability": 0.8}),
        recent=[
            Message(role="user", content="I prefer lightweight"),
            Message(role="assistant", content="Selected lightweight laptops."),
        ],
        raw_llm_payload={"removed_fields": ["portability_preference"]},
        expected_delta=RequirementsDelta(removed_fields=["portability"]),
    ),
    # 15. Unrelated message classified as off-topic
    ExtractionExample(
        id=15,
        description="Unrelated message that should be classified as off-topic",
        message="What is the capital of France and what's the weather today?",
        current=Requirements(),
        recent=[],
        raw_llm_payload={"is_off_topic": True},
        expected_delta=RequirementsDelta(is_off_topic=True),
    ),
]
