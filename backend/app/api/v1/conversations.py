"""Conversation API endpoints for the Electronics Recommendation Agent.

Provides RESTful operations for:
    - Creating conversation sessions.
    - Sending conversational messages and running the recommendation pipeline.
    - Retrieving current conversation state.
    - Retrieving chronological message history.
    - Deleting conversation sessions and clean state cleanup.
"""

from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, Depends, Header, Path, status

from app.agent.pipeline import run_pipeline
from app.api.products import get_product_repository
from app.core.errors import (
    ConversationNotFoundError,
    InvalidConversationIdError,
)
from app.core.logging import get_logger
from app.db.conversation_repository import (
    ConversationRepository,
    check_conversation_ownership,
    get_conversation_repository,
)
from app.db.repositories import ProductRepository
from app.llm.client import GroqClient, LLMClient
from app.models.conversation import (
    CONVERSATION_ID_REGEX,
    ConversationHistoryResponse,
    ConversationResponse,
    ConversationState,
    ConversationTurnResponse,
    CreateConversationRequest,
    SendMessageRequest,
)
from app.models.requirements import PolicyAction

logger = get_logger("app.api.conversations")

router = APIRouter(prefix="/conversations", tags=["Conversations"])


def get_llm_client() -> LLMClient:
    """Dependency provider returning an active LLMClient instance (GroqClient)."""
    return GroqClient()


def validate_conversation_id_format(conversation_id: str) -> str:
    """Validate syntax of client-supplied conversation_id path parameter."""
    if not isinstance(conversation_id, str) or not CONVERSATION_ID_REGEX.match(conversation_id.strip()):
        raise InvalidConversationIdError(
            f"Invalid conversation ID format: '{conversation_id}'. Must be 8-64 alphanumeric characters, dashes, or underscores."
        )
    return conversation_id.strip()


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
    description="Initialize a new empty conversation state and return its unique identifier.",
)
async def create_conversation(
    request: Optional[CreateConversationRequest] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    conv_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationResponse:
    """Create a new recommendation conversation session."""
    user_id = (request.user_id if request and request.user_id else x_user_id)
    metadata = (request.metadata if request else {})

    conversation = ConversationState(
        user_id=user_id,
        metadata=metadata,
    )
    saved = await conv_repo.create(conversation)
    logger.info("Created conversation '%s' (user_id: %s)", saved.conversation_id, saved.user_id)
    return ConversationResponse.from_state(saved)


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message and run recommendation pipeline",
    description="Process user message through extraction, merging, policy, retrieval, and validation.",
)
async def send_message(
    request: SendMessageRequest,
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    conv_repo: ConversationRepository = Depends(get_conversation_repository),
    product_repo: ProductRepository = Depends(get_product_repository),
    llm_client: LLMClient = Depends(get_llm_client),
) -> ConversationTurnResponse:
    """Process a user conversational message and return the assistant recommendation response."""
    valid_id = validate_conversation_id_format(conversation_id)

    # 1. Fetch current conversation state
    conv = await conv_repo.get(valid_id)
    if conv is None:
        raise ConversationNotFoundError(f"Conversation '{valid_id}' was not found.")

    # 2. Enforce ownership access control
    check_conversation_ownership(conv, x_user_id)

    # 3. Execute agent orchestration pipeline
    pipeline_result = await run_pipeline(
        user_message=request.message,
        conversation=conv,
        llm_client=llm_client,
        product_repo=product_repo,
    )

    # 4. Atomically persist updated conversation state
    await conv_repo.update(pipeline_result.updated_conversation)

    # 5. Format client turn response
    decision = pipeline_result.decision
    requires_clarification = decision.action in (PolicyAction.ASK, PolicyAction.CLARIFY_CONFLICT)
    clarification_question = decision.question if requires_clarification else None

    return ConversationTurnResponse(
        conversation_id=valid_id,
        message=pipeline_result.assistant_message,
        action=decision.action,
        turn_count=pipeline_result.updated_conversation.turn_count,
        requirements=pipeline_result.updated_conversation.requirements,
        requires_clarification=requires_clarification,
        clarification_question=clarification_question,
        unresolved_contradictions=pipeline_result.unresolved_contradictions,
        recommendations=pipeline_result.recommendations,
        recommendation_response=pipeline_result.recommendation_response,
        is_fallback=pipeline_result.is_fallback,
        can_recommend=decision.can_recommend,
        remaining_slots=decision.remaining_slots,
        slot_status=dict(pipeline_result.updated_conversation.slot_status),
        metadata={
            "slot": decision.slot,
            "decision_reason": decision.reason,
        },
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation state details",
    description="Retrieve the current accumulated requirements and metadata of a conversation.",
)
async def get_conversation(
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    conv_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationResponse:
    """Retrieve details and requirements state of a conversation."""
    valid_id = validate_conversation_id_format(conversation_id)
    conv = await conv_repo.get(valid_id)
    if conv is None:
        raise ConversationNotFoundError(f"Conversation '{valid_id}' was not found.")

    check_conversation_ownership(conv, x_user_id)
    return ConversationResponse.from_state(conv)


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation message history",
    description="Retrieve chronological user and assistant message history for a conversation.",
)
async def get_conversation_messages(
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    conv_repo: ConversationRepository = Depends(get_conversation_repository),
) -> ConversationHistoryResponse:
    """Retrieve all chronological messages from a conversation session."""
    valid_id = validate_conversation_id_format(conversation_id)
    conv = await conv_repo.get(valid_id)
    if conv is None:
        raise ConversationNotFoundError(f"Conversation '{valid_id}' was not found.")

    check_conversation_ownership(conv, x_user_id)
    return ConversationHistoryResponse(
        conversation_id=valid_id,
        messages=conv.messages,
        total_messages=len(conv.messages),
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a conversation session",
    description="Permanently remove a conversation session and all associated message history.",
)
async def delete_conversation(
    conversation_id: str = Path(..., description="Unique conversation identifier"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    conv_repo: ConversationRepository = Depends(get_conversation_repository),
) -> dict[str, Any]:
    """Delete a conversation session and verify ownership."""
    valid_id = validate_conversation_id_format(conversation_id)
    conv = await conv_repo.get(valid_id)
    if conv is None:
        raise ConversationNotFoundError(f"Conversation '{valid_id}' was not found.")

    check_conversation_ownership(conv, x_user_id)
    await conv_repo.delete(valid_id)
    return {
        "status": "deleted",
        "conversation_id": valid_id,
    }
