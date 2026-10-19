"""Conversation domain, state persistence, and API schema models.

Defines the structure for conversation sessions, message history,
turn-by-turn interactions, and client requests/responses.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.requirements import Contradiction, PolicyAction, Requirements
from app.models.response import ProductRecommendation, RecommendationResponse

CONVERSATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")


class ConversationMessage(BaseModel):
    """Single persistent chat message in a conversation session."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the message",
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="Author role of the message",
    )
    content: str = Field(
        description="Text content of the message",
    )
    action: PolicyAction | None = Field(
        default=None,
        description="Policy action executed on this turn if assistant message",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when message was created in UTC",
    )
    recommendations: list[ProductRecommendation] = Field(
        default_factory=list,
        description="Product recommendations delivered in this message turn",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Clarification question prompted on this turn if any",
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Detected contradictions active during this turn",
    )
    is_fallback: bool = Field(
        default=False,
        description="Whether this message or recommendation was generated via fallback",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message content must be a non-empty string.")
        return v.strip()


class ConversationState(BaseModel):
    """Persistent state of an active electronics recommendation conversation session."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the conversation session",
    )
    user_id: str | None = Field(
        default=None,
        description="Owner user identifier for authentication and access control",
    )
    requirements: Requirements = Field(
        default_factory=Requirements,
        description="Current accumulated and normalized user requirements",
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Number of conversation turns completed",
    )
    asked_slots: list[str] = Field(
        default_factory=list,
        description="Requirement slot names already asked by policy (e.g. 'category', 'budget')",
    )
    slot_status: dict[str, str] = Field(
        default_factory=dict,
        description="Status tracking for requirement slots (answered, declined, delegated, unknown)",
    )
    last_asked_slot: str | None = Field(
        default=None,
        description="The slot asked on the previous assistant turn",
    )
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Chronological history of conversation messages",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when conversation was created in UTC",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when conversation was last updated in UTC",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary session metadata",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Optimistic locking version counter",
    )

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, v: str) -> str:
        if not isinstance(v, str) or not CONVERSATION_ID_REGEX.match(v.strip()):
            raise ValueError(
                f"Invalid conversation ID '{v}'. Must be 8-64 alphanumeric characters, dashes, or underscores."
            )
        return v.strip()


class CreateConversationRequest(BaseModel):
    """Request payload to initiate a new recommendation conversation."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(
        default=None,
        description="Optional user ID for conversation ownership validation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional client metadata to attach to the conversation",
    )


class ConversationResponse(BaseModel):
    """Response payload detailing a conversation's status and accumulated state."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(description="Unique conversation identifier")
    user_id: str | None = Field(default=None, description="Owner user ID if assigned")
    requirements: Requirements = Field(description="Current normalized requirements")
    turn_count: int = Field(ge=0, description="Completed conversation turn count")
    asked_slots: list[str] = Field(description="List of clarification slots already asked")
    slot_status: dict[str, str] = Field(default_factory=dict, description="Status of requirement slots")
    created_at: datetime = Field(description="Creation timestamp in UTC")
    updated_at: datetime = Field(description="Last update timestamp in UTC")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    @classmethod
    def from_state(cls, state: ConversationState) -> ConversationResponse:
        """Construct ConversationResponse from internal ConversationState."""
        return cls(
            conversation_id=state.conversation_id,
            user_id=state.user_id,
            requirements=state.requirements,
            turn_count=state.turn_count,
            asked_slots=list(state.asked_slots),
            slot_status=dict(state.slot_status),
            created_at=state.created_at,
            updated_at=state.updated_at,
            metadata=dict(state.metadata),
        )


class SendMessageRequest(BaseModel):
    """Request payload containing a user message for a conversation turn."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message: str = Field(
        ...,
        alias="content",
        description="User text message (1-1000 characters)",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("User message cannot be empty or whitespace only.")
        cleaned = v.strip()
        if len(cleaned) > 1000:
            raise ValueError(
                f"User message exceeds maximum permitted length of 1000 characters (received {len(cleaned)})."
            )
        return cleaned


class ConversationTurnResponse(BaseModel):
    """Response payload returned after processing a conversational message turn."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    conversation_id: str = Field(description="Conversation session identifier")
    message: str = Field(description="Assistant response text for the user")
    action: PolicyAction = Field(description="Policy action executed on this turn")
    turn_count: int = Field(ge=0, description="Updated conversation turn count")
    requirements: Requirements = Field(description="Updated normalized user requirements")
    requires_clarification: bool = Field(
        default=False,
        description="Whether conversational clarification or conflict resolution is required",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Clarification question if action is ASK or CLARIFY_CONFLICT",
    )
    unresolved_contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Unresolved contradictions active after this turn",
    )
    recommendations: list[ProductRecommendation] = Field(
        default_factory=list,
        description="Ranked product recommendations when action is RECOMMEND",
    )
    recommendation_response: RecommendationResponse | None = Field(
        default=None,
        description="Full Module 9 recommendation response envelope if action is RECOMMEND",
    )
    is_fallback: bool = Field(
        default=False,
        description="Whether recommendation or extraction used fallback logic",
    )
    can_recommend: bool = Field(
        default=False,
        description="Whether the recommendation gate is unlocked",
    )
    remaining_slots: list[str] = Field(
        default_factory=list,
        description="List of remaining mandatory requirement slots",
    )
    slot_status: dict[str, str] = Field(
        default_factory=dict,
        description="Status tracking for requirement slots",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session context and telemetry metadata",
    )


class ConversationHistoryResponse(BaseModel):
    """Response payload containing chronological message history of a conversation."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(description="Conversation session identifier")
    messages: list[ConversationMessage] = Field(
        default_factory=list,
        description="Chronological list of all messages in the conversation",
    )
    total_messages: int = Field(
        ge=0,
        description="Total message count in the conversation",
    )
