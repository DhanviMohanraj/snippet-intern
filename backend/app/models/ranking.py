"""Domain models for candidate evaluation, scoring breakdown, and ranking results."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from app.models.product import Category, Product


class ConstraintStatus(str, Enum):
    """Evaluation outcome for a single hard constraint on a candidate product."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ConstraintCheck(BaseModel):
    """Individual constraint verification record."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(description="Name of the evaluated constraint (e.g. 'budget_max', 'min_ram_gb')")
    status: ConstraintStatus = Field(description="Status of evaluation: pass, fail, or unknown")
    required: Any = Field(description="Required threshold or condition from user requirements")
    actual: Any = Field(default=None, description="Actual specification from catalogue record, if available")
    reason: str = Field(description="Human-readable justification for the evaluation outcome")


class CandidateEvaluation(BaseModel):
    """Comprehensive hard-constraint evaluation report for a single product."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    product_id: str = Field(description="Unique product identifier")
    product: Product = Field(description="Underlying validated product domain model")
    is_eligible: bool = Field(description="True if all applicable constraints have PASS status")
    passed_constraints: list[ConstraintCheck] = Field(default_factory=list, description="List of passed constraints")
    failed_constraints: list[ConstraintCheck] = Field(default_factory=list, description="List of failed constraints")
    unverifiable_constraints: list[ConstraintCheck] = Field(
        default_factory=list, description="Constraints that could not be verified due to missing specs"
    )
    rejection_reasons: list[str] = Field(
        default_factory=list, description="Human-readable summary reasons why product is ineligible"
    )


class ScoreBreakdown(BaseModel):
    """Decomposed multi-factor preference score details."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    overall_score: float = Field(ge=0.0, le=100.0, description="Normalized composite score (0.0 to 100.0)")
    preference_scores: dict[str, float] = Field(
        default_factory=dict, description="Raw normalized score (0-100) per evaluated preference"
    )
    weighted_scores: dict[str, float] = Field(
        default_factory=dict, description="Weighted contribution of each preference towards the composite score"
    )
    explanations: dict[str, str] = Field(
        default_factory=dict, description="Human-readable explanation of each preference score"
    )
    unsupported_preferences: list[str] = Field(
        default_factory=list,
        description="Preferences that could not be evaluated due to missing catalogue specifications",
    )
    rating_score: float = Field(ge=0.0, le=100.0, description="Component score derived from user review rating")
    use_case_score: float = Field(ge=0.0, le=100.0, description="Component score derived from persona/use-case match")


class RankedCandidate(BaseModel):
    """An eligible candidate product ranked with score breakdown and selection rationale."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    product_id: str = Field(description="Unique product identifier")
    product: Product = Field(description="Underlying validated product domain model")
    overall_score: float = Field(ge=0.0, le=100.0, description="Composite ranking score (0.0 to 100.0)")
    score_breakdown: ScoreBreakdown = Field(description="Decomposed preference scoring details")
    evaluation: CandidateEvaluation = Field(description="Hard constraint validation report")
    reasons: list[str] = Field(default_factory=list, description="Structured rationale reasons for recommendation")


class CandidateSearchResult(BaseModel):
    """Complete candidate search, filtering, and ranking result for a requirements state."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    category: Category | None = Field(default=None, description="Category queried for candidates")
    total_retrieved: int = Field(ge=0, description="Total candidates retrieved from catalogue")
    total_passed: int = Field(ge=0, description="Total candidates satisfying all hard constraints")
    total_excluded: int = Field(ge=0, description="Total candidates excluded due to constraint failures")
    total_unverifiable: int = Field(ge=0, description="Total candidates excluded due to unverifiable specifications")
    ranked_candidates: list[RankedCandidate] = Field(
        default_factory=list, description="Ordered shortlist of top eligible candidates"
    )
    excluded_candidates: list[CandidateEvaluation] = Field(
        default_factory=list, description="Ineligible candidates with structured exclusion details"
    )
    data_quality_limitations: list[str] = Field(
        default_factory=list, description="Known catalogue data gaps or unverifiable attributes"
    )
    has_matches: bool = Field(description="True if at least one eligible matching product was found")
