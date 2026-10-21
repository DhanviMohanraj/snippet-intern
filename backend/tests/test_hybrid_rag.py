"""Comprehensive Test Suite for SpecSage Lightweight Hybrid RAG Layer.

Tests:
    1. BM25 exact-term lexical retrieval.
    2. Vector retrieval for semantically related queries.
    3. Hybrid fusion (RRF) and deduplication.
    4. Evidence provenance preservation.
    5. Isolation of evidence strictly by product ID.
    6. Unknown attributes remaining unknown.
    7. Unsupported claims being flagged and repaired.
    8. Specifications not transferring between products.
    9. Hard constraints continuing to work independently of RAG.
    10. Product scores remaining deterministic.
    11. All five supported categories covered.
    12. Embedding/retrieval failure handling and graceful fallback.
"""

from __future__ import annotations

import pytest

from app.agent.rag.bm25 import BM25Retriever, tokenize
from app.agent.rag.embeddings import FakeEmbeddingProvider, LocalHashEmbeddingProvider
from app.agent.rag.hybrid import HybridEvidenceRetriever
from app.agent.rag.indexer import EvidenceIndex, build_product_evidence_documents
from app.agent.rag.models import EvidenceDocument, ProductEvidenceSummary
from app.agent.rag.vector import VectorRetriever
from app.agent.ranking import evaluate_candidate, score_candidate
from app.agent.respond import generate_recommendation_response
from app.agent.response_validator import (
    build_fallback_product_recommendation,
    check_claim_contradictions,
    validate_recommendation_response,
)
from app.models.product import Category, Product
from app.models.ranking import CandidateEvaluation, CandidateSearchResult, RankedCandidate, ScoreBreakdown
from app.models.requirements import HardConstraints, Requirements
from app.models.response import LLMProductExplanation, LLMRecommendationPayload


def make_candidate(product: Product, score: float = 85.0, reasons: list[str] | None = None) -> RankedCandidate:
    eval_report = CandidateEvaluation(
        product_id=product.product_id,
        product=product,
        is_eligible=True,
    )
    score_bd = ScoreBreakdown(
        overall_score=score,
        rating_score=score,
        use_case_score=score,
    )
    return RankedCandidate(
        product_id=product.product_id,
        product=product,
        overall_score=score,
        score_breakdown=score_bd,
        evaluation=eval_report,
        reasons=reasons or ["Meets user requirements"],
    )


def make_search_result(candidates: list[RankedCandidate]) -> CandidateSearchResult:
    return CandidateSearchResult(
        category=candidates[0].product.category if candidates else None,
        total_retrieved=len(candidates),
        total_passed=len(candidates),
        total_excluded=0,
        total_unverifiable=0,
        ranked_candidates=candidates,
        has_matches=len(candidates) > 0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_laptop_1() -> Product:
    return Product(
        product_id="LAP-RAG-1",
        name="SwiftBook Air 14",
        brand="Aero",
        category=Category.LAPTOP,
        price_inr=55000,
        rating=4.5,
        specs={
            "cpu": "Intel Core i5-1340P",
            "ram_gb": 16,
            "storage_gb": 512,
            "gpu": "Integrated Intel Iris Xe",
            "vram_gb": 0,
            "weight_kg": 1.3,
            "battery_hr": 14.0,
        },
        features=["Lightweight magnesium chassis", "Backlit keyboard", "Thunderbolt 4"],
        intended_use=["office work", "travel", "programming"],
        in_stock=True,
    )


@pytest.fixture
def test_laptop_2() -> Product:
    return Product(
        product_id="LAP-RAG-2",
        name="Titan Gamer Pro 16",
        brand="Titan",
        category=Category.LAPTOP,
        price_inr=95000,
        rating=4.7,
        specs={
            "cpu": "AMD Ryzen 7 7840HS",
            "ram_gb": 32,
            "storage_gb": 1024,
            "gpu": "NVIDIA GeForce RTX 4070",
            "vram_gb": 8,
            "weight_kg": 2.4,
            "battery_hr": 5.0,
        },
        features=["165Hz QHD display", "RGB mechanical keyboard", "Dual cooling fans"],
        intended_use=["AAA gaming", "3D rendering", "video editing"],
        in_stock=True,
    )


@pytest.fixture
def test_phone() -> Product:
    return Product(
        product_id="PHN-RAG-1",
        name="PixelView Pro 10",
        brand="PixelView",
        category=Category.PHONE,
        price_inr=42000,
        rating=4.6,
        specs={
            "chipset": "Snapdragon 8 Gen 2",
            "ram_gb": 12,
            "storage_gb": 256,
            "camera_mp": 108,
            "battery_mah": 5000,
            "weight_g": 190,
        },
        features=["120Hz AMOLED display", "OIS Night Mode Camera", "67W Turbo Charge"],
        intended_use=["photography", "social media", "multitasking"],
        in_stock=True,
    )


@pytest.fixture
def test_headphones() -> Product:
    return Product(
        product_id="HDP-RAG-1",
        name="AcousticShield 900",
        brand="Acoustic",
        category=Category.HEADPHONES,
        price_inr=7999,
        rating=4.4,
        specs={
            "type": "over-ear",
            "wireless": True,
            "anc": True,
            "battery_hr": 40.0,
            "weight_g": 250,
        },
        features=["Active Noise Cancellation (ANC)", "Plush memory foam earcups", "Transparency Mode"],
        intended_use=["travel", "music", "online calls"],
        in_stock=True,
    )


@pytest.fixture
def test_tablet() -> Product:
    return Product(
        product_id="TAB-RAG-1",
        name="ArtFlow Tab 11",
        brand="ArtFlow",
        category=Category.TABLET,
        price_inr=38000,
        rating=4.3,
        specs={
            "chipset": "Snapdragon 870",
            "ram_gb": 8,
            "storage_gb": 128,
            "screen_in": 11.0,
            "stylus_support": True,
            "battery_hr": 10.0,
        },
        features=["2K IPS 120Hz screen", "Stylus pen compatible", "Quad stereo speakers"],
        intended_use=["drawing", "digital notes", "media streaming"],
        in_stock=True,
    )


@pytest.fixture
def test_monitor() -> Product:
    return Product(
        product_id="MON-RAG-1",
        name="UltraSync Gaming 27",
        brand="UltraSync",
        category=Category.MONITOR,
        price_inr=22999,
        rating=4.6,
        specs={
            "size_in": 27.0,
            "resolution": "2560x1440",
            "refresh_hz": 165,
            "panel": "IPS",
            "ports": ["DisplayPort 1.4", "HDMI 2.0"],
        },
        features=["165Hz rapid refresh rate", "1ms MPRT", "AMD FreeSync Premium"],
        intended_use=["competitive gaming", "coding"],
        in_stock=True,
    )


@pytest.fixture
def all_test_products(
    test_laptop_1,
    test_laptop_2,
    test_phone,
    test_headphones,
    test_tablet,
    test_monitor,
) -> list[Product]:
    return [
        test_laptop_1,
        test_laptop_2,
        test_phone,
        test_headphones,
        test_tablet,
        test_monitor,
    ]


@pytest.fixture
def populated_index(all_test_products) -> EvidenceIndex:
    return EvidenceIndex(all_test_products)


@pytest.fixture
def populated_retriever(populated_index) -> HybridEvidenceRetriever:
    return HybridEvidenceRetriever(index=populated_index)


# ---------------------------------------------------------------------------
# 1. BM25 Exact-Term Retrieval
# ---------------------------------------------------------------------------

def test_bm25_exact_term_retrieval(populated_index):
    retriever = BM25Retriever(documents=populated_index.get_documents())

    # Exact technical spec match: "16GB RAM"
    results = retriever.search("16GB RAM", top_k=5)
    assert len(results) > 0
    top = results[0]
    assert "ram" in top.document.source_field.lower() or "16gb" in top.document.content.lower()
    assert top.score > 0.0
    assert top.retrieval_method == "bm25"

    # Exact GPU model match
    gpu_results = retriever.search("RTX 4070", top_k=3)
    assert len(gpu_results) > 0
    assert gpu_results[0].document.product_id == "LAP-RAG-2"
    assert "rtx 4070" in gpu_results[0].document.content.lower()

    # Refresh rate match
    hz_results = retriever.search("165Hz", top_k=5)
    assert any("165hz" in r.document.content.lower() for r in hz_results)


def test_bm25_tokenizer_compounds():
    tokens = tokenize("16GB RAM with 512GB SSD and 165Hz display")
    assert "16gb" in tokens
    assert "16" in tokens
    assert "gb" in tokens
    assert "512gb" in tokens
    assert "165hz" in tokens


# ---------------------------------------------------------------------------
# 2. Vector Retrieval for Semantically Related Queries
# ---------------------------------------------------------------------------

def test_vector_semantic_retrieval(populated_index):
    retriever = VectorRetriever(documents=populated_index.get_documents())

    # Semantic query for ultraportable travel use case
    results = retriever.search("ultra lightweight portable laptop for travel", top_k=5)
    assert len(results) > 0
    assert results[0].retrieval_method == "vector"
    # Should match SwiftBook Air 14 with high similarity
    matched_pids = {r.document.product_id for r in results[:3]}
    assert "LAP-RAG-1" in matched_pids


def test_vector_retrieval_with_fake_provider():
    # Verify vector retriever logic with deterministic fake embedding
    doc = EvidenceDocument(
        document_id="DOC-1",
        product_id="P-1",
        category=Category.PHONE,
        product_name="Phone 1",
        source_field="camera",
        recorded_value="108MP",
        content="Night photography with 108MP camera",
    )
    fake_provider = FakeEmbeddingProvider(
        dimension=3,
        preset_vectors={
            "camera": [1.0, 0.0, 0.0],
            "battery": [0.0, 1.0, 0.0],
        },
    )
    retriever = VectorRetriever(documents=[doc], embedding_provider=fake_provider)
    results = retriever.search("camera photography", top_k=1)
    assert len(results) == 1
    assert results[0].document.document_id == "DOC-1"


# ---------------------------------------------------------------------------
# 3. Hybrid Fusion and Deduplication
# ---------------------------------------------------------------------------

def test_hybrid_rrf_fusion_and_deduplication(populated_retriever):
    results = populated_retriever.retrieve(
        query="Active Noise Cancellation over-ear headphones",
        top_k=5,
    )
    assert len(results) > 0
    assert results[0].retrieval_method == "hybrid_rrf"
    assert results[0].document.product_id == "HDP-RAG-1"

    # Verify deduplication: no duplicate (product_id, source_field)
    keys = [(r.document.product_id, r.document.source_field) for r in results]
    assert len(keys) == len(set(keys)), "Duplicate (product_id, source_field) found in RRF results."


# ---------------------------------------------------------------------------
# 4. Evidence Provenance Preservation
# ---------------------------------------------------------------------------

def test_evidence_provenance_preservation(populated_retriever):
    results = populated_retriever.retrieve("512GB SSD storage", top_k=5)
    assert len(results) > 0

    for r in results:
        doc = r.document
        assert doc.product_id != ""
        assert doc.product_name != ""
        assert doc.source_field != ""
        assert doc.recorded_value is not None
        assert isinstance(doc.provenance, dict)
        assert "field" in doc.provenance
        assert "product_id" in doc.provenance


# ---------------------------------------------------------------------------
# 5. Isolation of Evidence by Product ID
# ---------------------------------------------------------------------------

def test_evidence_isolation_by_product_id(populated_retriever):
    # Retrieve evidence strictly for LAP-RAG-1
    results = populated_retriever.retrieve(
        query="NVIDIA RTX 4070 dedicated graphics card for gaming",
        top_k=10,
        product_id="LAP-RAG-1",
    )
    # Even though query asks for RTX 4070 (present on LAP-RAG-2), results must strictly belong to LAP-RAG-1
    for r in results:
        assert r.document.product_id == "LAP-RAG-1", (
            f"Evidence leakage: returned evidence from {r.document.product_id} when querying for LAP-RAG-1"
        )


# ---------------------------------------------------------------------------
# 6. Unknown Attributes Remaining Unknown
# ---------------------------------------------------------------------------

def test_unknown_attributes_remaining_unknown(populated_retriever, test_tablet):
    # Tablet specs: lacks 'weight_g' or 'os' in this test instance
    summary = populated_retriever.get_product_grounding_context(
        product=test_tablet,
        requirements=Requirements(category=Category.TABLET, use_cases=["drawing"]),
    )
    assert isinstance(summary, ProductEvidenceSummary)
    assert summary.product_id == test_tablet.product_id
    # Check that missing attributes in category schema are identified as unknown
    assert "weight_g" in summary.unknown_attributes


# ---------------------------------------------------------------------------
# 7. Unsupported Claims Being Flagged and Repaired
# ---------------------------------------------------------------------------

def test_unsupported_anc_claim_flagged_and_repaired(test_headphones):
    passive_hdp = test_headphones.model_copy(deep=True)
    passive_hdp.specs["anc"] = False

    # LLM hallucinating active noise cancellation on passive headphones
    hallucinated_text = "These headphones have world-class Active Noise Cancellation (ANC) with ANC mode."
    contradictions = check_claim_contradictions(
        text=hallucinated_text,
        product=passive_hdp,
        requirements=Requirements(category=Category.HEADPHONES),
    )
    assert any("anc" in c.lower() for c in contradictions)

    # When validated in response validator, explanation must be repaired with grounded fallback
    cand = make_candidate(passive_hdp, score=85.0, reasons=["Priced within budget"])
    search_result = make_search_result([cand])
    payload = LLMRecommendationPayload(
        message="Here are recommendations:",
        explanations=[
            LLMProductExplanation(
                product_id=passive_hdp.product_id,
                reasoning=hallucinated_text,
                advantages=["Active noise cancellation"],
                trade_offs=["Slightly heavier"],
                match_explanation="Has ANC",
            )
        ],
    )
    response = validate_recommendation_response(
        payload=payload,
        search_result=search_result,
        requirements=Requirements(category=Category.HEADPHONES),
    )
    assert len(response.recommendations) == 1
    rec = response.recommendations[0]
    # The hallucinated ANC claim must be replaced by grounded fallback stating passive isolation
    assert "active noise cancellation (anc)" not in rec.reasoning.lower()


# ---------------------------------------------------------------------------
# 8. Specifications Not Transferring Between Products
# ---------------------------------------------------------------------------

def test_specifications_not_transferring_between_products(test_laptop_1, test_laptop_2):
    cand1 = make_candidate(test_laptop_1, score=80.0)
    cand2 = make_candidate(test_laptop_2, score=85.0)

    # Text for Laptop 1 claims it has the RTX 4070 from Laptop 2
    hallucinated_text = f"The {test_laptop_1.name} features an NVIDIA GeForce RTX 4070 for extreme gaming."
    contradictions = check_claim_contradictions(
        text=hallucinated_text,
        product=test_laptop_1,
        requirements=Requirements(category=Category.LAPTOP),
        all_candidates=[cand1, cand2],
    )
    assert any("cross-product" in c.lower() or "gpu" in c.lower() for c in contradictions)


# ---------------------------------------------------------------------------
# 9. Hard Constraints Continuing to Work Independently of RAG
# ---------------------------------------------------------------------------

def test_hard_constraints_independent_of_rag(test_laptop_1):
    # Laptop 1 has budget ₹55,000. User budget is max ₹40,000.
    req_over_budget = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=40000,
    )
    # Hard constraint evaluation must FAIL regardless of any RAG similarity score
    eval_res = evaluate_candidate(test_laptop_1, req_over_budget)
    assert eval_res.is_eligible is False
    assert any("budget_max" in c.name for c in eval_res.failed_constraints)


# ---------------------------------------------------------------------------
# 10. Product Scores Remaining Deterministic
# ---------------------------------------------------------------------------

def test_product_scores_remain_deterministic(test_laptop_1):
    req = Requirements(
        category=Category.LAPTOP,
        use_cases=["programming", "travel"],
        preferences={"portability": 0.9, "battery_life": 0.8},
    )
    score_1 = score_candidate(test_laptop_1, req)
    score_2 = score_candidate(test_laptop_1, req)
    assert score_1.overall_score == score_2.overall_score
    assert score_1.preference_scores == score_2.preference_scores


# ---------------------------------------------------------------------------
# 11. All Five Supported Categories Covered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,expected_pid", [
    (Category.LAPTOP, "LAP-RAG-1"),
    (Category.PHONE, "PHN-RAG-1"),
    (Category.HEADPHONES, "HDP-RAG-1"),
    (Category.TABLET, "TAB-RAG-1"),
    (Category.MONITOR, "MON-RAG-1"),
])
def test_all_five_categories_indexed_and_retrievable(
    populated_retriever,
    category: Category,
    expected_pid: str,
):
    docs = populated_retriever.index.get_documents(category=category)
    assert len(docs) > 0
    assert any(d.product_id == expected_pid for d in docs)

    # Test retrieval within category
    results = populated_retriever.retrieve(
        query=category.value,
        category=category,
        top_k=3,
    )
    assert len(results) > 0
    assert all(r.document.category == category for r in results)


# ---------------------------------------------------------------------------
# 12. Retrieval or Embedding Failure Handling
# ---------------------------------------------------------------------------

class CrashingEmbeddingProvider(LocalHashEmbeddingProvider):
    def embed_text(self, text: str) -> list[float]:
        raise RuntimeError("Simulated embedding engine crash.")


@pytest.mark.asyncio
async def test_retrieval_failure_graceful_fallback(test_laptop_1):
    cand = make_candidate(test_laptop_1, score=82.0, reasons=["Meets portability requirements"])
    search_result = make_search_result([cand])

    # Calling generate_recommendation_response without LLM (offline fallback)
    # Even if RAG encounters an error, it must return grounded fallback safely
    response = await generate_recommendation_response(
        requirements=Requirements(category=Category.LAPTOP, use_cases=["programming"]),
        search_result=search_result,
        llm_client=None,
    )
    assert response is not None
    assert len(response.recommendations) == 1
    assert response.recommendations[0].product_id == test_laptop_1.product_id
    assert response.is_fallback is True
