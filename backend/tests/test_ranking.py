"""Comprehensive unit tests for Module 8 candidate filtering, scoring, and ranking."""

from __future__ import annotations

import copy
import pytest

from app.agent.ranking import (
    evaluate_candidate,
    filter_candidates,
    rank_candidates,
    score_candidate,
    search_and_rank,
)
from app.models.product import Category, Product
from app.models.ranking import ConstraintStatus
from app.models.requirements import HardConstraints, Requirements


# ---------------------------------------------------------------------------
# Test Helpers and Factories
# ---------------------------------------------------------------------------

def make_test_laptop(
    product_id: str = "LAP-001",
    name: str = "Zenit Air 14",
    brand: str = "Zenit",
    price_inr: int = 60000,
    ram_gb: int = 16,
    storage_gb: int = 512,
    gpu: str = "Intel Iris Xe",
    vram_gb: int = 0,
    weight_kg: float = 1.35,
    battery_hr: float = 10.0,
    intended_use: list[str] | None = None,
    features: list[str] | None = None,
    in_stock: bool = True,
    rating: float = 4.5,
) -> Product:
    """Helper factory for creating validated Laptop products."""
    specs = {
        "cpu": "Intel Core i5-13420H",
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "gpu": gpu,
        "vram_gb": vram_gb,
        "weight_kg": weight_kg,
        "battery_hr": battery_hr,
        "screen_in": 14.0,
        "os": "Windows 11 Home",
    }
    return Product(
        product_id=product_id,
        name=name,
        brand=brand,
        category=Category.LAPTOP,
        price_inr=price_inr,
        specs=specs,
        intended_use=intended_use or ["programming", "general"],
        features=features or ["Backlit Keyboard", "FHD IPS"],
        in_stock=in_stock,
        rating=rating,
    )


def make_test_phone(
    product_id: str = "PHN-001",
    name: str = "Nova Phone 12",
    brand: str = "Nova",
    price_inr: int = 35000,
    ram_gb: int = 8,
    storage_gb: int = 128,
    camera_mp: float = 50.0,
    battery_mah: int = 5000,
    weight_g: float = 185.0,
    in_stock: bool = True,
    rating: float = 4.4,
) -> Product:
    """Helper factory for creating validated Phone products."""
    specs = {
        "chipset": "Snapdragon 778G",
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "camera_mp": camera_mp,
        "battery_mah": battery_mah,
        "screen_in": 6.5,
        "weight_g": weight_g,
        "os": "Android 14",
    }
    return Product(
        product_id=product_id,
        name=name,
        brand=brand,
        category=Category.PHONE,
        price_inr=price_inr,
        specs=specs,
        intended_use=["everyday", "photography"],
        features=["AMOLED 120Hz", "Fast Charging"],
        in_stock=in_stock,
        rating=rating,
    )


# ---------------------------------------------------------------------------
# 1. Hard-Constraint Filtering Tests
# ---------------------------------------------------------------------------

def test_product_within_budget_passes():
    """A product priced within maximum budget passes constraint evaluation."""
    product = make_test_laptop(price_inr=65000)
    req = Requirements(category=Category.LAPTOP, budget_max_inr=70000)

    report = evaluate_candidate(product, req)

    assert report.is_eligible is True
    assert len(report.failed_constraints) == 0
    budget_checks = [c for c in report.passed_constraints if c.name == "budget_max"]
    assert len(budget_checks) == 1
    assert budget_checks[0].status == ConstraintStatus.PASS


def test_product_over_maximum_budget_fails():
    """A product exceeding maximum budget is rejected as ineligible."""
    product = make_test_laptop(price_inr=85000)
    req = Requirements(category=Category.LAPTOP, budget_max_inr=75000)

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    assert len(report.failed_constraints) == 1
    assert report.failed_constraints[0].name == "budget_max"
    assert report.failed_constraints[0].status == ConstraintStatus.FAIL
    assert "exceeds maximum budget" in report.failed_constraints[0].reason


def test_product_below_minimum_budget_fails():
    """A product priced below minimum budget is rejected."""
    product = make_test_laptop(price_inr=45000)
    req = Requirements(category=Category.LAPTOP, budget_min_inr=50000)

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    assert len(report.failed_constraints) == 1
    assert report.failed_constraints[0].name == "budget_min"
    assert report.failed_constraints[0].status == ConstraintStatus.FAIL


def test_minimum_ram_violation_fails():
    """A product with 8GB RAM fails a hard constraint requiring 16GB."""
    product = make_test_laptop(ram_gb=8)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(min_ram_gb=16),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "min_ram_gb"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL
    assert failed[0].actual == 8
    assert failed[0].required == 16


def test_minimum_storage_violation_fails():
    """A product with 256GB storage fails when 512GB is required."""
    product = make_test_laptop(storage_gb=256)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(min_storage_gb=512),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "min_storage_gb"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_minimum_battery_violation_fails():
    """A product with 6hr battery fails when 10hr runtime is required."""
    product = make_test_laptop(battery_hr=6.0)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(min_battery_hr=10.0),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "min_battery_hr"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_maximum_weight_violation_fails():
    """A 2.2kg laptop fails when maximum allowable weight is 1.5kg."""
    product = make_test_laptop(weight_kg=2.2)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(max_weight_kg=1.5),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "max_weight_kg"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_missing_ram_cannot_pass_ram_hard_constraint():
    """Critical Rule: A product lacking RAM spec is marked UNKNOWN and rejected."""
    # Construct a product whose specs dictionary does not contain ram_gb
    p = make_test_laptop()
    specs_without_ram = dict(p.specs)
    del specs_without_ram["ram_gb"]
    object.__setattr__(p, "specs", specs_without_ram)

    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(min_ram_gb=16),
    )

    report = evaluate_candidate(p, req)

    assert report.is_eligible is False
    assert len(report.unverifiable_constraints) == 1
    assert report.unverifiable_constraints[0].name == "min_ram_gb"
    assert report.unverifiable_constraints[0].status == ConstraintStatus.UNKNOWN
    assert "lacks 'ram_gb' specification" in report.unverifiable_constraints[0].reason


def test_missing_gpu_cannot_prove_dedicated_gpu_exists():
    """Critical Rule: Missing GPU spec cannot prove compliance when dedicated GPU is required."""
    p = make_test_laptop()
    specs_without_gpu = dict(p.specs)
    del specs_without_gpu["gpu"]
    object.__setattr__(p, "specs", specs_without_gpu)

    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(requires_gpu=True),
    )

    report = evaluate_candidate(p, req)

    assert report.is_eligible is False
    assert len(report.unverifiable_constraints) == 1
    assert report.unverifiable_constraints[0].name == "requires_gpu"
    assert report.unverifiable_constraints[0].status == ConstraintStatus.UNKNOWN


def test_dedicated_gpu_requirement_excludes_integrated_graphics():
    """When dedicated GPU is required, a verified integrated graphics laptop fails."""
    product = make_test_laptop(gpu="Intel Iris Xe", vram_gb=0)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(requires_gpu=True),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "requires_gpu"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL
    assert "integrated graphics" in failed[0].reason


def test_dedicated_gpu_requirement_accepts_rtx():
    """When dedicated GPU is required, an NVIDIA RTX laptop passes."""
    product = make_test_laptop(gpu="NVIDIA GeForce RTX 4060", vram_gb=8)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(requires_gpu=True),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is True
    passed = [c for c in report.passed_constraints if c.name == "requires_gpu"]
    assert len(passed) == 1
    assert passed[0].status == ConstraintStatus.PASS


def test_minimum_vram_violation_fails():
    """A product with 4GB VRAM fails when 8GB VRAM is required."""
    product = make_test_laptop(gpu="NVIDIA GeForce RTX 3050", vram_gb=4)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(min_vram_gb=8),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "min_vram_gb"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_excluded_brands_match_case_insensitively():
    """Brand exclusions exclude matching brands regardless of letter casing."""
    product = make_test_laptop(brand="Lenovo")
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(exclude_brands=["lenovo", "apple"]),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "exclude_brands"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_in_stock_constraint_enforced():
    """Out of stock product fails when in_stock=True constraint is required."""
    product = make_test_laptop(in_stock=False)
    req = Requirements(
        category=Category.LAPTOP,
        hard=HardConstraints(in_stock=True),
    )

    report = evaluate_candidate(product, req)

    assert report.is_eligible is False
    failed = [c for c in report.failed_constraints if c.name == "in_stock"]
    assert len(failed) == 1
    assert failed[0].status == ConstraintStatus.FAIL


def test_category_specific_constraints_phone_weight():
    """Phone weight in grams is properly converted and validated against max_weight_kg."""
    # 185g phone evaluated against 0.2kg (200g) maximum weight limit -> passes
    phone = make_test_phone(weight_g=185.0)
    req_pass = Requirements(category=Category.PHONE, hard=HardConstraints(max_weight_kg=0.20))
    report_pass = evaluate_candidate(phone, req_pass)
    assert report_pass.is_eligible is True

    # 185g phone evaluated against 0.15kg (150g) maximum weight limit -> fails
    req_fail = Requirements(category=Category.PHONE, hard=HardConstraints(max_weight_kg=0.15))
    report_fail = evaluate_candidate(phone, req_fail)
    assert report_fail.is_eligible is False


def test_unknown_outcomes_distinct_from_verified_failures():
    """CandidateEvaluation clearly separates failed constraints from unverifiable constraints."""
    # Product with both a verified budget failure and an unknown RAM specification
    p = make_test_laptop(price_inr=90000)
    specs = dict(p.specs)
    del specs["ram_gb"]
    object.__setattr__(p, "specs", specs)

    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        hard=HardConstraints(min_ram_gb=16),
    )

    report = evaluate_candidate(p, req)

    assert report.is_eligible is False
    assert len(report.failed_constraints) == 1
    assert report.failed_constraints[0].name == "budget_max"
    assert len(report.unverifiable_constraints) == 1
    assert report.unverifiable_constraints[0].name == "min_ram_gb"


# ---------------------------------------------------------------------------
# 2. Preference-Based Scoring and Ranking Tests
# ---------------------------------------------------------------------------

def test_higher_preference_match_produces_higher_score():
    """When other factors are identical, a lighter laptop scores higher on portability."""
    light_laptop = make_test_laptop(product_id="LAP-A", weight_kg=1.15, price_inr=60000)
    heavy_laptop = make_test_laptop(product_id="LAP-B", weight_kg=2.30, price_inr=60000)

    req = Requirements(
        category=Category.LAPTOP,
        preferences={"portability": 1.0},
    )

    score_light = score_candidate(light_laptop, req)
    score_heavy = score_candidate(heavy_laptop, req)

    assert score_light.overall_score > score_heavy.overall_score
    assert score_light.preference_scores["portability"] > score_heavy.preference_scores["portability"]


def test_multiple_preferences_combined_correctly():
    """Multiple preferences are weighted proportionally without distortion."""
    laptop = make_test_laptop(ram_gb=32, weight_kg=1.20)
    req = Requirements(
        category=Category.LAPTOP,
        preferences={"performance": 0.8, "portability": 0.4},
    )

    score_bd = score_candidate(laptop, req)

    assert 0.0 <= score_bd.overall_score <= 100.0
    assert "performance" in score_bd.weighted_scores
    assert "portability" in score_bd.weighted_scores
    # Performance weight (0.8) is 2x portability weight (0.4), so weighted contribution should be larger
    assert score_bd.weighted_scores["performance"] > score_bd.weighted_scores["portability"]


def test_preference_weights_normalized_safely():
    """Arbitrary preference weight scales (e.g. 0.1 vs 1.0) maintain normalized bounds."""
    laptop = make_test_laptop()
    req1 = Requirements(category=Category.LAPTOP, preferences={"performance": 1.0})
    req2 = Requirements(category=Category.LAPTOP, preferences={"performance": 0.5})

    score1 = score_candidate(laptop, req1)
    score2 = score_candidate(laptop, req2)

    # A single preference with weight 0.5 or 1.0 represents 100% of preference focus
    assert score1.overall_score == score2.overall_score


def test_unspecified_preferences_do_not_influence_scores():
    """Preferences not mentioned by the user do not penalize or distort scores."""
    laptop = make_test_laptop()
    req_no_pref = Requirements(category=Category.LAPTOP, preferences={})

    score_bd = score_candidate(laptop, req_no_pref)

    assert score_bd.weighted_scores == {}
    assert 0.0 <= score_bd.overall_score <= 100.0


def test_hard_filter_failures_never_enter_ranked_shortlist():
    """Hard constraints act as strict gates: ineligible products never appear in rank_candidates."""
    passing_laptop = make_test_laptop(product_id="LAP-PASS", price_inr=50000, ram_gb=16)
    failing_laptop = make_test_laptop(product_id="LAP-FAIL", price_inr=95000, ram_gb=64)  # High specs but over budget

    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        preferences={"performance": 1.0},
    )

    eligible, excluded = filter_candidates([passing_laptop, failing_laptop], req)
    assert len(eligible) == 1
    assert eligible[0].product_id == "LAP-PASS"

    ranked = rank_candidates(eligible, req, top_n=5)
    assert len(ranked) == 1
    assert ranked[0].product_id == "LAP-PASS"


def test_deterministic_tie_breaking():
    """Ties in overall_score are resolved deterministically by price ascending, then product_id."""
    # Two laptops with identical specs and ratings, differing only in price
    laptop1 = make_test_laptop(product_id="LAP-002", price_inr=65000)
    laptop2 = make_test_laptop(product_id="LAP-001", price_inr=55000)

    req = Requirements(category=Category.LAPTOP, preferences={"performance": 0.8})

    ranked = rank_candidates([laptop1, laptop2], req, top_n=5)

    assert len(ranked) == 2
    # LAP-001 should be first because it is cheaper (₹55,000 < ₹65,000)
    assert ranked[0].product_id == "LAP-001"
    assert ranked[1].product_id == "LAP-002"


def test_score_explanations_provided():
    """ScoreBreakdown provides non-empty human-readable explanations for every preference."""
    laptop = make_test_laptop()
    req = Requirements(
        category=Category.LAPTOP,
        preferences={"portability": 0.8, "performance": 0.9},
    )

    score_bd = score_candidate(laptop, req)

    assert "portability" in score_bd.explanations
    assert "performance" in score_bd.explanations
    assert len(score_bd.explanations["portability"]) > 5


# ---------------------------------------------------------------------------
# 3. End-to-End Coordinator (search_and_rank) Tests
# ---------------------------------------------------------------------------

class FakeProductRepo:
    def __init__(self, products: list[Product]) -> None:
        self.products = products

    async def find(self, filters=None, sort="price_inr_asc", page=1, page_size=50):
        res = list(self.products)
        if filters and "category" in filters:
            cat = filters["category"]
            cat_val = cat.value if isinstance(cat, Category) else cat
            res = [p for p in res if p.category.value == cat_val]
        return res[:page_size], len(res)


@pytest.mark.asyncio
async def test_search_and_rank_success():
    """End-to-end flow retrieves, filters, ranks, and returns structured CandidateSearchResult."""
    laptops = [
        make_test_laptop(product_id="LAP-1", price_inr=50000, ram_gb=16),
        make_test_laptop(product_id="LAP-2", price_inr=65000, ram_gb=16),
        make_test_laptop(product_id="LAP-3", price_inr=95000, ram_gb=32),  # Excluded (over budget)
    ]
    repo = FakeProductRepo(laptops)
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        preferences={"portability": 0.8},
    )

    result = await search_and_rank(req, repo, top_n=2)

    assert result.has_matches is True
    assert result.total_retrieved == 3
    assert result.total_passed == 2
    assert result.total_excluded == 1
    assert len(result.ranked_candidates) == 2
    assert result.ranked_candidates[0].product_id in ("LAP-1", "LAP-2")
    assert result.ranked_candidates[1].product_id in ("LAP-1", "LAP-2")
    assert len(result.excluded_candidates) == 1
    assert result.excluded_candidates[0].product_id == "LAP-3"


@pytest.mark.asyncio
async def test_search_and_rank_no_matches_produces_structured_report():
    """When no products satisfy hard constraints, search_and_rank returns structured empty shortlist."""
    laptops = [
        make_test_laptop(product_id="LAP-1", price_inr=80000),
        make_test_laptop(product_id="LAP-2", price_inr=95000),
    ]
    repo = FakeProductRepo(laptops)
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=40000,  # Unattainable budget
    )

    result = await search_and_rank(req, repo)

    assert result.has_matches is False
    assert result.total_retrieved == 2
    assert result.total_passed == 0
    assert result.total_excluded == 2
    assert result.ranked_candidates == []
    assert len(result.excluded_candidates) == 2
    assert all("budget" in r for e in result.excluded_candidates for r in e.rejection_reasons)


@pytest.mark.asyncio
async def test_search_and_rank_does_not_mutate_inputs():
    """search_and_rank leaves requirements state completely unmodified."""
    repo = FakeProductRepo([make_test_laptop()])
    req = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=80000,
        preferences={"performance": 0.8},
        use_cases=["programming"],
    )
    req_copy = copy.deepcopy(req)

    await search_and_rank(req, repo)

    assert req == req_copy
