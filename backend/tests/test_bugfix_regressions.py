"""Regression tests for the fixes made against the SpecSage error report (21 Oct 2026).

Each test names the error number from the report it guards against.
"""

from __future__ import annotations

import pytest

from app.agent.extract import _apply_deterministic_fallbacks
from app.agent.merge import merge
from app.agent.ranking import evaluate_candidate, rank_candidates, score_candidate, search_and_rank
from app.agent.response_validator import (
    build_fallback_product_recommendation,
    check_claim_contradictions,
    validate_recommendation_response,
)
from app.models.product import Category, Product
from app.models.requirements import HardConstraints, Requirements, RequirementsDelta
from app.models.response import LLMProductExplanation, LLMRecommendationPayload


def _monitor(pid: str, name: str, price: int, size: float, hz: int, features: list[str], rating: float = 4.2) -> Product:
    return Product(
        product_id=pid,
        name=name,
        brand=name.split()[0],
        category=Category.MONITOR,
        price_inr=price,
        rating=rating,
        specs={"size_in": size, "resolution": "1920x1080", "panel": "IPS", "refresh_hz": hz, "ports": ["HDMI"]},
        features=features,
        intended_use=["coding", "office"],
        in_stock=True,
    )


def _laptop(**spec_overrides) -> Product:
    specs = {
        "cpu": "AMD Ryzen 5 7640U", "ram_gb": 16, "storage_gb": 512, "gpu": "AMD Radeon 760M",
        "vram_gb": 0, "weight_kg": 1.38, "battery_hr": 12.5, "screen_in": 14.0, "os": "Windows 11 Home",
    }
    specs.update(spec_overrides)
    return Product(
        product_id="LAP-T1", name="Lumina UltraSlate 14", brand="Lumina", category=Category.LAPTOP,
        price_inr=68000, rating=4.3, specs=specs, features=["Backlit Keyboard"],
        intended_use=["student", "programming"], in_stock=True,
    )


def _headphones(anc: bool) -> Product:
    return Product(
        product_id="HP-T1", name="Velo Studio 5", brand="Velo", category=Category.HEADPHONES,
        price_inr=4499, rating=4.1,
        specs={"type": "over-ear", "anc": anc, "battery_hr": 30, "wireless": True, "weight_g": 240, "codec": "AAC"},
        features=["Memory Foam Cushions"], intended_use=["music", "calls"], in_stock=True,
    )


# --- Error 1: requested result count ------------------------------------------------

@pytest.mark.parametrize(
    "message, expected",
    [
        ("Show me the top two monitors under 15000", 2),
        ("best three laptops for coding", 3),
        ("give me 2 options", 2),
        ("suggest five headphones", 5),
        ("I need one laptop for college", None),
        ("laptop with 2 USB ports", None),
        ("I have 2 monitors already, need a laptop", None),
        ("16GB RAM and 512GB SSD under 80k", None),
    ],
)
def test_requested_result_count_handles_number_words(message, expected):
    delta = _apply_deterministic_fallbacks(message, RequirementsDelta())
    assert delta.requested_result_count == expected


# --- Error 27: explanation punctuation and wording ----------------------------------

def test_fallback_reasoning_has_clean_punctuation():
    req = Requirements(category=Category.LAPTOP, budget_max_inr=80000, use_cases=["student", "programming"])
    ranked = rank_candidates([_laptop()], req, top_n=1)
    rec = build_fallback_product_recommendation(ranked[0], req)

    assert ".;" not in rec.reasoning
    assert ".." not in rec.reasoning
    assert "(Lumina)" not in rec.reasoning
    assert "student and programming" in rec.reasoning
    assert ".;" not in rec.match_explanation and ".." not in rec.match_explanation


# --- Error 15: scoring code must not invent certifications --------------------------

def test_eye_comfort_score_quotes_features_without_inventing_certification():
    mon = _monitor("MON-T1", "Ocular CodeView 27", 28999, 27.0, 75, ["Anti-Flicker"])
    req = Requirements(category=Category.MONITOR, preferences={"eye_comfort": 0.9})
    breakdown = score_candidate(mon, req)

    assert "certif" not in breakdown.explanations["eye_comfort"].lower()
    assert "Anti-Flicker" in breakdown.explanations["eye_comfort"]
    assert "certif" not in breakdown.explanations["display_quality"].lower()


# --- Error 12: refresh rate must not inflate display quality when not wanted --------

def test_refresh_rate_bonus_only_when_user_cares():
    fast = _monitor("MON-F", "Zenit Swift 24", 13999, 24.0, 165, [])
    slow = _monitor("MON-S", "Zenit Clear 24", 13999, 24.0, 75, [])
    text_req = Requirements(category=Category.MONITOR, use_cases=["coding"], preferences={"display_quality": 0.9})

    assert score_candidate(fast, text_req).preference_scores["display_quality"] == \
        score_candidate(slow, text_req).preference_scores["display_quality"]

    gaming_req = Requirements(category=Category.MONITOR, use_cases=["gaming"], preferences={"display_quality": 0.9})
    assert score_candidate(fast, gaming_req).preference_scores["display_quality"] > \
        score_candidate(slow, gaming_req).preference_scores["display_quality"]


# --- Errors 15-20: claim validator --------------------------------------------------

def test_validator_ignores_honest_negative_statements():
    req = Requirements(category=Category.HEADPHONES)
    text = "This model does not include active noise cancellation, so it relies on passive isolation."
    assert check_claim_contradictions(text, _headphones(anc=False), req) == []


def test_validator_ignores_unverified_gamut_disclaimer():
    req = Requirements(category=Category.MONITOR)
    mon = _monitor("MON-T2", "Nexa Essential 24", 7999, 24.0, 60, ["VESA Compatible"])
    text = "Exact sRGB coverage is not verified in the catalogue."
    assert check_claim_contradictions(text, mon, req) == []


def test_validator_flags_hyphenated_low_blue_light_certification():
    req = Requirements(category=Category.MONITOR)
    mon = _monitor("MON-T3", "Lumina WorkStation 27", 24999, 27.0, 75, ["Eye-Care Ambient Light Sensor"])
    text = "It carries a low-blue-light certification that reduces eye strain."
    assert any("certification" in c.lower() for c in check_claim_contradictions(text, mon, req))


def test_validator_accepts_certification_listed_in_catalogue():
    req = Requirements(category=Category.MONITOR)
    mon = _monitor("MON-0005", "Zenit OfficeClear 24", 9499, 23.8, 75, ["Low Blue Light Certified"])
    text = "It is low blue light certified."
    assert check_claim_contradictions(text, mon, req) == []


def test_validator_flags_wrong_battery_hours_and_weight():
    req = Requirements(category=Category.LAPTOP, budget_max_inr=80000)
    text = "It offers 15 hours of battery life and weighs just 1.1 kg."
    issues = check_claim_contradictions(text, _laptop(), req)
    assert any("battery" in c.lower() for c in issues)
    assert any("weight" in c.lower() for c in issues)


def test_validator_accepts_correct_battery_and_weight_and_usage_hours():
    req = Requirements(category=Category.LAPTOP, budget_max_inr=80000)
    text = "Ideal for 8 hours of coding daily, with 12.5 hours of battery life at 1.38 kg."
    assert check_claim_contradictions(text, _laptop(), req) == []


def test_validator_storage_units_checked_per_claim():
    req = Requirements(category=Category.LAPTOP)
    laptop = _laptop(storage_gb=1024)
    # Correct 1TB claim alongside an unrelated GB figure must not be flagged.
    text = "Comes with 1TB SSD storage and 16GB RAM."
    assert check_claim_contradictions(text, laptop, req) == []


# --- Errors 4, 10, 11: preference extraction and removal ----------------------------



def test_soft_screen_size_allows_adjectives():
    msg = "I code for eight hours daily, prefer a sharp 24-inch display, and don't need a high refresh rate."
    delta = _apply_deterministic_fallbacks(msg, RequirementsDelta())
    assert delta.hard is not None and delta.hard.target_screen_in == 24.0
    assert "screen_size" in (delta.preferences or {})


def test_unwanted_refresh_rate_clears_preference_and_constraint():
    current = Requirements(
        category=Category.MONITOR,
        preferences={"refresh_rate": 0.8, "display_quality": 0.9},
        hard=HardConstraints(min_refresh_hz=144),
    )
    delta = _apply_deterministic_fallbacks("I don't need a high refresh rate", RequirementsDelta())
    merged, _ = merge(current, delta)
    assert "refresh_rate" not in merged.preferences
    assert merged.hard.min_refresh_hz is None
    assert merged.preferences["display_quality"] == 0.9


def test_removing_anc_clears_hard_constraint_and_preference():
    current = Requirements(
        category=Category.HEADPHONES,
        preferences={"noise_cancellation": 0.9},
        hard=HardConstraints(requires_anc=True),
    )
    merged, _ = merge(current, RequirementsDelta(removed_fields=["anc"]))
    assert merged.hard.requires_anc is None
    assert "noise_cancellation" not in merged.preferences


def test_monitor_ranking_responds_to_size_and_clarity_preferences():
    catalogue = [
        _monitor("MON-0005", "Zenit OfficeClear 24", 9499, 23.8, 75, ["Low Blue Light Certified"]),
        _monitor("MON-0015", "Zenit UltraSlim 27", 14499, 27.0, 100, ["7.5mm Ultra-Slim Body"], rating=4.3),
        _monitor("MON-0007", "Nexa Essential 24", 7999, 24.0, 60, ["VESA Compatible"], rating=4.0),
    ]
    req = Requirements(
        category=Category.MONITOR, budget_max_inr=15000, use_cases=["coding"],
        hard=HardConstraints(target_screen_in=24.0),
        preferences={"screen_size": 0.9, "display_quality": 0.9, "eye_comfort": 0.6},
    )
    ranked = [c.product.name for c in rank_candidates(catalogue, req, top_n=3)]
    assert ranked[-1] == "Zenit UltraSlim 27"


# --- Error 23: stale evidence index -------------------------------------------------

def test_grounding_evidence_never_contradicts_live_record():
    from app.agent.rag import get_hybrid_retriever

    retriever = get_hybrid_retriever()
    live = _laptop()
    live = live.model_copy(update={"product_id": "LAP-0016", "features": ["Changed Feature Text", "Another New Feature"]})
    summary = retriever.get_product_grounding_context(live, Requirements(category=Category.LAPTOP), top_k=8)

    feature_evidence = [e.document.content for e in summary.retrieved_evidence if e.document.source_field.startswith("features")]
    indexed_old_features = ["2.8K 90Hz OLED", "Dual Mics with AI Noise Cancellation"]  # LAP-0016 in data/products.json
    assert not any(old in c for c in feature_evidence for old in indexed_old_features)
    for e in summary.retrieved_evidence:
        if e.document.source_field.startswith("specs."):
            key = e.document.source_field.split(".", 1)[1]
            assert str(live.specs[key]) in e.document.content


# --- Phone scenario follow-up (21 Oct): explanation quality --------------------------



def _phone(pid: str, name: str, price: int, storage: int, weight_g: int, uses: list[str], in_stock: bool = True, mp: float = 64.0) -> Product:
    return Product(
        product_id=pid, name=name, brand=name.split()[0], category=Category.PHONE, price_inr=price, rating=4.3,
        specs={"chipset": "Snapdragon 7s Gen 2", "ram_gb": 8, "storage_gb": storage, "camera_mp": mp,
               "battery_mah": 5000, "screen_in": 6.6, "weight_g": weight_g, "os": "Android 14"},
        features=["Dual SIM"], intended_use=uses, in_stock=in_stock,
    )


_PHONES = [
    _phone("PHN-A", "Titan WorkMobile 10", 41999, 512, 215, ["productivity", "general"], mp=108.0),
    _phone("PHN-B", "Pulse ZoomMaster 12", 48999, 256, 190, ["photography", "gaming"]),
    _phone("PHN-C", "Lumina Clarity 10", 35999, 256, 186, ["photography", "general"]),
    _phone("PHN-D", "Nexa Note Max", 26999, 256, 199, ["photography"], in_stock=False, mp=200.0),
    _phone("PHN-E", "Nova Trigger Red 9", 65999, 512, 229, ["gaming"]),
]
_PHONE_REQ = Requirements(
    category=Category.PHONE, budget_max_inr=50000, use_cases=["photography"],
    preferences={"camera_quality": 0.9}, hard=HardConstraints(min_storage_gb=512),
)


def test_out_of_stock_products_are_never_recommended_by_default():
    req = Requirements(category=Category.PHONE, budget_max_inr=50000)
    assert evaluate_candidate(_PHONES[3], req).is_eligible is False
    allow = Requirements(category=Category.PHONE, budget_max_inr=50000, hard=HardConstraints(in_stock=False))
    assert evaluate_candidate(_PHONES[3], allow).is_eligible is True


def _phone_response(explanation: LLMProductExplanation, uncertainties: list[str] | None = None):
    import asyncio
    sr = asyncio.run(search_and_rank(_PHONE_REQ, _PHONES))
    payload = LLMRecommendationPayload(
        message="Here are the tailored explanations for the phone that meets your criteria.",
        explanations=[explanation], remaining_uncertainties=uncertainties or [],
    )
    return validate_recommendation_response(payload, sr, _PHONE_REQ)


def test_single_match_explains_binding_constraint_and_relaxation():
    resp = _phone_response(LLMProductExplanation(
        product_id="PHN-A", reasoning="Offers a 108 MP camera.", advantages=["108 MP camera"],
        trade_offs=[], match_explanation="Meets storage.", limitations=[],
    ))
    assert "Only 1 of the 5 phones" in resp.message
    assert "minimum storage of 512 GB" in resp.message
    assert "Pulse ZoomMaster 12" in resp.message
    assert "Nexa Note Max" not in resp.message  # out of stock is not a relaxation option


def test_lightweight_claim_on_heavy_phone_is_rejected():
    issues = check_claim_contradictions("215 g lightweight design", _PHONES[0], _PHONE_REQ)
    assert any("lightweight" in c for c in issues)
    assert check_claim_contradictions("A lightweight design.", _PHONES[2], _PHONE_REQ) == []


def test_data_gaps_are_not_trade_offs_and_irrelevant_uncertainties_dropped():
    resp = _phone_response(
        LLMProductExplanation(
            product_id="PHN-A", reasoning="Offers a 108 MP camera.", advantages=["108 MP camera"],
            trade_offs=["No information on camera sensor type or video recording capabilities",
                        "No details on 5G or headphone jack"],
            match_explanation="Meets storage.",
            limitations=["Camera sensor type unknown", "5G connectivity status unknown", "Headphone jack presence unknown"],
        ),
        uncertainties=["Camera sensor type", "Durability and build quality unknown"],
    )
    rec = resp.recommendations[0]
    assert not any(t.lower().startswith("no information") or t.lower().startswith("no details") for t in rec.trade_offs)
    assert any("215" in t for t in rec.trade_offs)
    assert any("productivity" in t for t in rec.trade_offs)
    all_notes = " ".join(rec.limitations + resp.remaining_uncertainties).lower()
    assert "5g" not in all_notes and "headphone jack" not in all_notes and "durability" not in all_notes
    assert sum("camera sensor" in u.lower() for u in resp.remaining_uncertainties) == 1
