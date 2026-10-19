"""Automated offline tests for asynchronous natural-language requirement extraction.

Verifies:
- extract() invocation using FakeLLMClient.
- Proper schema supplied to complete_json().
- Inclusion of current requirements in LLM context.
- Conversation history truncation to at most the last 6 messages.
- Budget modifications without premature state merging.
- Safe handling of empty and oversized input messages.
- Error propagation from LLM failures.
- End-to-end execution of all 15 handwritten extraction scenarios.
"""

from __future__ import annotations

import pytest

from app.agent.examples import EXTRACTION_EXAMPLES
from app.agent.extract import MAX_MESSAGE_LENGTH, extract
from app.llm.errors import LLMAuthenticationError, LLMTimeoutError
from app.llm.fake import FakeLLMClient
from app.models.product import Category
from app.models.requirements import (
    HardConstraints,
    Message,
    Requirements,
    RequirementsDelta,
)


@pytest.mark.asyncio
async def test_extract_basic_laptop_request():
    """extract() calls complete_json with RequirementsDelta and applies normalization."""
    fake_client = FakeLLMClient(
        json_responses=[{
            "category": "lappy",
            "budget_max_inr": 60000,
            "use_cases": ["coding"],
        }]
    )
    current = Requirements()
    recent = []

    delta = await extract(
        message="I need a lightweight laptop for coding under 60k",
        current=current,
        recent=recent,
        llm=fake_client,
    )

    assert isinstance(delta, RequirementsDelta)
    assert delta.category == Category.LAPTOP
    assert delta.budget_max_inr == 60000
    assert delta.use_cases == ["programming"]

    # Verify that fake_client was called with schema=RequirementsDelta
    assert fake_client.call_count == 1
    call_record = fake_client.calls[0]
    assert call_record["method"] == "complete_json"
    assert call_record["schema"] == RequirementsDelta


@pytest.mark.asyncio
async def test_extract_truncates_recent_history_to_at_most_6_messages():
    """extract() ensures at most the last 6 messages from recent history are in context."""
    fake_client = FakeLLMClient(
        json_responses=[{"budget_max_inr": 80000}]
    )

    # Provide 10 recent messages
    ten_messages = [
        Message(role="user", content=f"Message {i}") for i in range(10)
    ]
    current = Requirements(category=Category.LAPTOP)

    await extract(
        message="make it 80k",
        current=current,
        recent=ten_messages,
        llm=fake_client,
    )

    call_messages = fake_client.last_messages
    assert call_messages is not None

    # Find the context system message
    context_msg = next(m["content"] for m in call_messages if "Recent Conversation Context" in m["content"])

    # Messages 0 to 3 should NOT appear; Messages 4 to 9 (last 6) MUST appear
    for i in range(4):
        assert f"Message {i}" not in context_msg
    for i in range(4, 10):
        assert f"Message {i}" in context_msg


@pytest.mark.asyncio
async def test_extract_includes_current_requirements_context():
    """extract() embeds current requirements in prompt context so LLM understands deltas."""
    fake_client = FakeLLMClient(
        json_responses=[{"budget_max_inr": 90000}]
    )
    current = Requirements(
        category=Category.LAPTOP,
        budget_max_inr=70000,
        preferences={"portability": 0.8},
    )

    await extract(
        message="increase my budget to 90k",
        current=current,
        recent=[],
        llm=fake_client,
    )

    call_messages = fake_client.last_messages
    assert call_messages is not None
    context_msg = next(m["content"] for m in call_messages if "Current Accumulated Requirements State" in m["content"])

    assert "laptop" in context_msg
    assert "70000" in context_msg
    assert "portability" in context_msg


@pytest.mark.asyncio
async def test_extract_empty_or_whitespace_message_raises_value_error():
    """extract() rejects empty or whitespace-only messages."""
    fake_client = FakeLLMClient()
    current = Requirements()

    with pytest.raises(ValueError, match="cannot be empty"):
        await extract("", current=current, recent=[], llm=fake_client)

    with pytest.raises(ValueError, match="cannot be empty"):
        await extract("   \n\t  ", current=current, recent=[], llm=fake_client)

    assert fake_client.call_count == 0


@pytest.mark.asyncio
async def test_extract_oversized_message_raises_value_error():
    """extract() rejects messages exceeding MAX_MESSAGE_LENGTH (1000 characters)."""
    fake_client = FakeLLMClient()
    current = Requirements()
    oversized_text = "a" * (MAX_MESSAGE_LENGTH + 1)

    with pytest.raises(ValueError, match="exceeds maximum permitted length"):
        await extract(oversized_text, current=current, recent=[], llm=fake_client)

    assert fake_client.call_count == 0


@pytest.mark.asyncio
async def test_extract_llm_failure_propagates_cleanly():
    """extract() propagates typed LLM errors without suppression."""
    fake_timeout = FakeLLMClient(error_to_raise=LLMTimeoutError("Groq request timed out"))
    current = Requirements()

    with pytest.raises(LLMTimeoutError, match="timed out"):
        await extract("I want a phone", current=current, recent=[], llm=fake_timeout)

    fake_auth = FakeLLMClient(error_to_raise=LLMAuthenticationError("Auth failed"))
    with pytest.raises(LLMAuthenticationError, match="Auth failed"):
        await extract("I want a phone", current=current, recent=[], llm=fake_auth)


@pytest.mark.asyncio
@pytest.mark.parametrize("example", EXTRACTION_EXAMPLES, ids=[f"scenario_{e.id}" for e in EXTRACTION_EXAMPLES])
async def test_extract_across_all_15_examples(example):
    """Verify that extract() handles all 15 canonical scenarios and normalizes outputs."""
    fake_client = FakeLLMClient(json_responses=[example.raw_llm_payload])

    delta = await extract(
        message=example.message,
        current=example.current,
        recent=example.recent,
        llm=fake_client,
    )

    assert isinstance(delta, RequirementsDelta)
    assert delta.category == example.expected_delta.category
    assert delta.use_cases == example.expected_delta.use_cases
    assert delta.budget_max_inr == example.expected_delta.budget_max_inr
    assert delta.budget_min_inr == example.expected_delta.budget_min_inr
    assert delta.hard == example.expected_delta.hard
    assert delta.preferences == example.expected_delta.preferences
    assert delta.ml_mode == example.expected_delta.ml_mode
    assert delta.removed_fields == example.expected_delta.removed_fields
    assert delta.is_off_topic == example.expected_delta.is_off_topic
