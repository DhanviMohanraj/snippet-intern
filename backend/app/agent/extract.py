"""Asynchronous natural-language requirement extraction layer.

Translates conversational user input into a validated RequirementsDelta model
using the injected LLM client and deterministic normalization.
"""

from __future__ import annotations

from app.agent.normalize import normalize
from app.llm.client import LLMClient
from app.llm.prompts import build_extraction_messages
from app.models.requirements import Message, Requirements, RequirementsDelta

MAX_MESSAGE_LENGTH = 1000


async def extract(
    message: str,
    current: Requirements,
    recent: list[Message],
    llm: LLMClient,
) -> RequirementsDelta:
    """Extract requirements delta from a user message.

    Performs:
        1. Input validation (non-empty, length bounded to <= 1000 chars).
        2. Prompt construction including current state context and <= 6 recent messages.
        3. LLM structured JSON completion using the injected LLMClient.
        4. Pure Python deterministic normalization on extracted values.

    Args:
        message: The latest raw user text message.
        current: Current accumulated session Requirements.
        recent: Recent conversation history (sliced to at most the last 6 messages).
        llm: Injected asynchronous LLM client conforming to LLMClient protocol.

    Returns:
        A normalized and validated RequirementsDelta instance.

    Raises:
        ValueError: If the user message is empty, whitespace, or exceeds max length.
        LLMError: If LLM completion fails after permitted retries.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("User message cannot be empty or whitespace only.")

    cleaned_message = message.strip()
    if len(cleaned_message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"User message exceeds maximum permitted length of {MAX_MESSAGE_LENGTH} characters "
            f"(received {len(cleaned_message)})."
        )

    # Build chat messages with prompt and context
    chat_messages = build_extraction_messages(
        message=cleaned_message,
        current=current,
        recent=recent,
    )

    # Invoke LLM for structured JSON output validated against RequirementsDelta
    raw_delta = await llm.complete_json(
        messages=chat_messages,
        schema=RequirementsDelta,
        max_tokens=1500,
        temperature=0.1,
    )


    # Apply pure Python deterministic normalization
    normalized_delta = normalize(raw_delta)

    return normalized_delta
