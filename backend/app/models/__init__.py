"""Domain, data transfer, and API response models."""

from app.models.api import (
    DependencyStatus,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    OverallHealthStatus,
    ProductListResponse,
)
from app.models.product import Category, CategorySummary, Product
from app.models.ranking import (
    CandidateEvaluation,
    CandidateSearchResult,
    ConstraintCheck,
    ConstraintStatus,
    RankedCandidate,
    ScoreBreakdown,
)
from app.models.requirements import (
    Contradiction,
    HardConstraints,
    Message,
    PolicyAction,
    PolicyDecision,
    RECOGNIZED_PREFERENCES,
    Requirements,
    RequirementsDelta,
    VALID_REMOVABLE_FIELDS,
)
from app.models.response import (
    LLMProductExplanation,
    LLMRecommendationPayload,
    ProductRecommendation,
    RecommendationResponse,
)

from app.models.conversation import (
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationResponse,
    ConversationState,
    ConversationTurnResponse,
    CreateConversationRequest,
    SendMessageRequest,
)

__all__ = [
    "CandidateEvaluation",
    "CandidateSearchResult",
    "Category",
    "CategorySummary",
    "ConstraintCheck",
    "ConstraintStatus",
    "Contradiction",
    "ConversationHistoryResponse",
    "ConversationMessage",
    "ConversationResponse",
    "ConversationState",
    "ConversationTurnResponse",
    "CreateConversationRequest",
    "DependencyStatus",
    "ErrorDetail",
    "ErrorResponse",
    "HardConstraints",
    "HealthResponse",
    "LLMProductExplanation",
    "LLMRecommendationPayload",
    "Message",
    "OverallHealthStatus",
    "PolicyAction",
    "PolicyDecision",
    "Product",
    "ProductListResponse",
    "ProductRecommendation",
    "RECOGNIZED_PREFERENCES",
    "RankedCandidate",
    "RecommendationResponse",
    "Requirements",
    "RequirementsDelta",
    "ScoreBreakdown",
    "SendMessageRequest",
    "VALID_REMOVABLE_FIELDS",
]
