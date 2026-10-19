"""Automated offline tests for Module 4 Groq LLM Client & Structured Output Infrastructure.

Covers all 21 required test specifications:
1. Successful text completion.
2. Successful JSON completion.
3. Correct Pydantic schema validation.
4. Malformed JSON handling and bounded recovery/error.
5. Schema validation failure handling and bounded recovery/error.
6. Missing assistant message content.
7. Empty completion response.
8. HTTP 401 handling without repeated retries.
9. HTTP 403 handling without repeated retries.
10. HTTP 429 handling and bounded retry behavior.
11. HTTP 5xx handling and bounded retry behavior.
12. Network connection failure handling.
13. Request timeout handling.
14. Retry exhaustion behavior.
15. Correct exponential backoff delay progression.
16. Correct parsing and handling of Retry-After headers.
17. Missing or placeholder API key validation.
18. Missing model or invalid configuration validation.
19. Prevention of API key/token leakage in exceptions and repr.
20. Deterministic offline FakeLLMClient operations.
21. ping() connectivity checks across success and error conditions.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.llm.client import GroqClient, extract_json_from_text
from app.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMNetworkError,
    LLMPermissionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseFormatError,
    LLMRetryExhaustionError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    sanitize_error_message,
)
from app.llm.fake import FakeLLMClient


# ---------------------------------------------------------------------------
# Test Schemas
# ---------------------------------------------------------------------------

class UserRequirement(BaseModel):
    category: str
    max_budget: float = Field(..., gt=0)
    features: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Helper Fixtures & Builders
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_settings() -> Settings:
    """Settings instance with test credentials."""
    return Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="gsk_valid_mock_api_key_test_12345",
        llm_model="openai/gpt-oss-20b",
        llm_timeout_s=10.0,
        llm_max_retries=2,
    )


def make_groq_response(
    content: str | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    no_choices: bool = False,
    null_choice: bool = False,
    no_message: bool = False,
) -> httpx.Response:
    """Build a mock httpx.Response following Groq's OpenAI-compatible schema."""
    resp_headers = headers or {"Content-Type": "application/json"}
    if status_code != 200:
        return httpx.Response(
            status_code=status_code,
            headers=resp_headers,
            content=json.dumps({"error": {"message": f"Error status {status_code}"}}).encode(),
        )

    if no_choices:
        body: dict[str, Any] = {"choices": []}
    elif null_choice:
        body = {"choices": [None]}
    elif no_message:
        body = {"choices": [{}]}
    else:
        body = {
            "id": "chatcmpl-test-id",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "openai/gpt-oss-20b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

    return httpx.Response(
        status_code=status_code,
        headers=resp_headers,
        content=json.dumps(body).encode(),
    )


# ---------------------------------------------------------------------------
# 1. Successful text completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_success(valid_settings):
    """complete_text sends request and returns raw assistant text."""
    mock_resp = make_groq_response("Here is your laptop recommendation.")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    messages = [{"role": "user", "content": "Recommend a laptop under $1000"}]

    result = await client.complete_text(messages, max_tokens=150)

    assert result == "Here is your laptop recommendation."
    assert mock_client.post.call_count == 1

    # Verify posted payload
    call_kwargs = mock_client.post.call_args[1]
    assert call_kwargs["json"]["model"] == "openai/gpt-oss-20b"
    assert call_kwargs["json"]["messages"] == messages
    assert call_kwargs["json"]["max_tokens"] == 150
    assert "Authorization" in call_kwargs["headers"]
    assert "gsk_valid_mock_api_key_test_12345" in call_kwargs["headers"]["Authorization"]


# ---------------------------------------------------------------------------
# 2 & 3. Successful JSON completion & Pydantic validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_json_success(valid_settings):
    """complete_json validates and parses structured JSON into Pydantic model."""
    valid_data = {
        "category": "laptop",
        "max_budget": 1200.0,
        "features": ["lightweight", "usb-c"],
        "confidence": 0.95,
    }
    mock_resp = make_groq_response(json.dumps(valid_data))
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    messages = [{"role": "user", "content": "Find me a lightweight laptop"}]

    result = await client.complete_json(messages, schema=UserRequirement)

    assert isinstance(result, UserRequirement)
    assert result.category == "laptop"
    assert result.max_budget == 1200.0
    assert result.features == ["lightweight", "usb-c"]
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_complete_json_strips_markdown_fences(valid_settings):
    """complete_json successfully extracts JSON wrapped in ```json ... ``` code blocks."""
    raw_markdown = (
        "```json\n"
        '{"category": "phone", "max_budget": 800.0, "features": ["5g"], "confidence": 0.88}\n'
        "```"
    )
    mock_resp = make_groq_response(raw_markdown)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    result = await client.complete_json([{"role": "user", "content": "phone"}], schema=UserRequirement)

    assert isinstance(result, UserRequirement)
    assert result.category == "phone"
    assert result.max_budget == 800.0


# ---------------------------------------------------------------------------
# 4. Malformed JSON handling & bounded correction retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_json_malformed_json_recovery(valid_settings):
    """complete_json sends correction prompt on malformed JSON and recovers."""
    bad_resp = make_groq_response('{"category": "laptop", "max_budget": 1000, broken...')
    good_resp = make_groq_response('{"category": "laptop", "max_budget": 1000.0, "confidence": 0.9}')

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[bad_resp, good_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)
    result = await client.complete_json([{"role": "user", "content": "laptop"}], schema=UserRequirement)

    assert isinstance(result, UserRequirement)
    assert result.max_budget == 1000.0
    assert mock_client.post.call_count == 2
    assert mock_sleep.call_count == 1


@pytest.mark.asyncio
async def test_complete_json_malformed_json_exhaustion(valid_settings):
    """complete_json raises LLMSchemaValidationError when malformed JSON persists."""
    bad_resp = make_groq_response("{invalid json content}")
    mock_client = AsyncMock()
    # Initial + 2 correction attempts = 3 total calls
    mock_client.post = AsyncMock(side_effect=[bad_resp, bad_resp, bad_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMSchemaValidationError) as exc_info:
        await client.complete_json([{"role": "user", "content": "laptop"}], schema=UserRequirement)

    assert "could not be parsed as JSON" in str(exc_info.value)
    assert mock_client.post.call_count == 3


# ---------------------------------------------------------------------------
# 5. Schema validation failure handling & bounded recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_json_schema_error_recovery(valid_settings):
    """complete_json corrects missing required fields using schema error feedback."""
    # First response misses 'confidence' field and has negative budget
    invalid_data = {"category": "laptop", "max_budget": -50.0}
    valid_data = {"category": "laptop", "max_budget": 500.0, "confidence": 0.9}

    bad_resp = make_groq_response(json.dumps(invalid_data))
    good_resp = make_groq_response(json.dumps(valid_data))

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[bad_resp, good_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)
    result = await client.complete_json([{"role": "user", "content": "laptop"}], schema=UserRequirement)

    assert isinstance(result, UserRequirement)
    assert result.max_budget == 500.0
    assert result.confidence == 0.9
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_complete_json_schema_error_exhaustion(valid_settings):
    """complete_json raises LLMSchemaValidationError when schema error persists."""
    invalid_data = {"category": "laptop"}  # Missing max_budget and confidence
    bad_resp = make_groq_response(json.dumps(invalid_data))

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[bad_resp, bad_resp, bad_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMSchemaValidationError) as exc_info:
        await client.complete_json([{"role": "user", "content": "laptop"}], schema=UserRequirement)

    assert "schema validation" in str(exc_info.value).lower()
    assert mock_client.post.call_count == 3


# ---------------------------------------------------------------------------
# 6 & 7. Missing choices, null message, and empty content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_missing_choices(valid_settings):
    """Raises LLMResponseFormatError when response choices array is empty."""
    mock_resp = make_groq_response(no_choices=True)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMResponseFormatError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])
    assert "no choices" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_complete_text_null_content(valid_settings):
    """Raises LLMResponseFormatError when content is null."""
    mock_resp = make_groq_response(content=None)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMResponseFormatError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])
    assert "null content" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_complete_text_empty_content(valid_settings):
    """Raises LLMResponseFormatError when content is whitespace only."""
    mock_resp = make_groq_response(content="   \n  ")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMResponseFormatError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])
    assert "empty content" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 8 & 9. HTTP 401 & 403 handling (non-retryable)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_auth_error_401_no_retries(valid_settings):
    """HTTP 401 raises LLMAuthenticationError immediately without retrying."""
    mock_resp = make_groq_response(status_code=401)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 401
    assert mock_client.post.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_complete_text_permission_error_403_no_retries(valid_settings):
    """HTTP 403 raises LLMPermissionError immediately without retrying."""
    mock_resp = make_groq_response(status_code=403)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMPermissionError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert exc_info.value.status == 403
    assert mock_client.post.call_count == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 10. HTTP 429 rate limit handling & bounded retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_rate_limit_429_retry_success(valid_settings):
    """HTTP 429 triggers exponential backoff and succeeds on subsequent attempt."""
    rate_limit_resp = make_groq_response(status_code=429)
    success_resp = make_groq_response("Success after rate limit.")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[rate_limit_resp, success_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)
    result = await client.complete_text([{"role": "user", "content": "hi"}])

    assert result == "Success after rate limit."
    assert mock_client.post.call_count == 2
    assert mock_sleep.call_count == 1
    # Check first backoff attempt delay (0.5s)
    assert mock_sleep.call_args[0][0] == 0.5


@pytest.mark.asyncio
async def test_complete_text_rate_limit_exhaustion(valid_settings):
    """Persistent HTTP 429 exhausts retries and raises LLMRetryExhaustionError."""
    rate_limit_resp = make_groq_response(status_code=429)

    mock_client = AsyncMock()
    # max_retries = 2 -> total attempts = 3
    mock_client.post = AsyncMock(side_effect=[rate_limit_resp, rate_limit_resp, rate_limit_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMRetryExhaustionError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert "exhausted" in str(exc_info.value).lower()
    assert mock_client.post.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# 11. HTTP 5xx handling and bounded retry behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_server_error_500_retry_success(valid_settings):
    """HTTP 500 triggers retry and succeeds on second attempt."""
    error_resp = make_groq_response(status_code=500)
    success_resp = make_groq_response("Success after 500.")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[error_resp, success_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)
    result = await client.complete_text([{"role": "user", "content": "hi"}])

    assert result == "Success after 500."
    assert mock_client.post.call_count == 2
    assert mock_sleep.call_count == 1


# ---------------------------------------------------------------------------
# 12 & 13. Network failure and timeout handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_text_network_error_retry_and_exhaustion(valid_settings):
    """httpx.NetworkError triggers bounded retries and raises LLMRetryExhaustionError."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMRetryExhaustionError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert mock_client.post.call_count == 3
    assert mock_sleep.call_count == 2
    assert "network" in str(exc_info.value).lower() or "retries exhausted" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_complete_text_timeout_retry_and_exhaustion(valid_settings):
    """httpx.TimeoutException triggers bounded retries and raises LLMRetryExhaustionError."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Timed out"))
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMRetryExhaustionError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert mock_client.post.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# 14 & 15. Exponential backoff progression verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exponential_backoff_delays(valid_settings):
    """Verify that retries follow exponential progression (0.5s, 1.0s)."""
    server_err = make_groq_response(status_code=503)
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[server_err, server_err, server_err])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)

    with pytest.raises(LLMRetryExhaustionError):
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert mock_sleep.call_count == 2
    delays = [call[0][0] for call in mock_sleep.call_args_list]
    assert delays == [0.5, 1.0]


# ---------------------------------------------------------------------------
# 16. Retry-After header parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_after_header_respected(valid_settings):
    """Verify that Retry-After header overrides default exponential backoff."""
    rate_limit_resp = make_groq_response(
        status_code=429,
        headers={"Content-Type": "application/json", "Retry-After": "3.5"},
    )
    success_resp = make_groq_response("Recovered after retry-after.")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[rate_limit_resp, success_resp])
    mock_sleep = AsyncMock()

    client = GroqClient(settings=valid_settings, http_client=mock_client, sleep_func=mock_sleep)
    result = await client.complete_text([{"role": "user", "content": "hi"}])

    assert result == "Recovered after retry-after."
    assert mock_sleep.call_count == 1
    assert mock_sleep.call_args[0][0] == 3.5


# ---------------------------------------------------------------------------
# 17 & 18. Configuration validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_or_placeholder_api_key():
    """Raises LLMConfigurationError when API key is missing or set to placeholder."""
    bad_settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="replace_me",
    )
    client = GroqClient(settings=bad_settings)
    with pytest.raises(LLMConfigurationError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert "placeholder" in str(exc_info.value).lower() or "not configured" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_missing_model_configuration():
    """Raises LLMConfigurationError when model name is empty."""
    bad_settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="gsk_valid_key",
        llm_model="",
    )
    client = GroqClient(settings=bad_settings)
    with pytest.raises(LLMConfigurationError) as exc_info:
        await client.complete_text([{"role": "user", "content": "hi"}])

    assert "model" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# 19. Privacy & Sanitization: No API key in exceptions or repr
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_api_key_leaked_in_repr_or_exceptions(valid_settings):
    """Assert secret API key is redacted in repr, exceptions, and error scrubbers."""
    raw_secret = "gsk_super_secret_test_token_999888777"
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key=raw_secret,
        llm_model="openai/gpt-oss-20b",
    )
    client = GroqClient(settings=settings)

    # 1. Check repr
    client_repr = repr(client)
    assert raw_secret not in client_repr
    assert "..." in client_repr or "[REDACTED]" in client_repr

    # 2. Check sanitize_error_message
    dirty_text = f"Failed to authenticate with Bearer {raw_secret} on host"
    clean_text = sanitize_error_message(dirty_text)
    assert raw_secret not in clean_text
    assert "[REDACTED_SECRET]" in clean_text

    # 3. Check exception messages
    err = LLMError("CODE", 500, f"Error with token {raw_secret}")
    assert raw_secret not in str(err)


# ---------------------------------------------------------------------------
# 20. FakeLLMClient operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fake_llm_client_operations():
    """Verify FakeLLMClient conforms to LLMClient interface and behaves deterministically."""
    fake = FakeLLMClient(
        text_responses=["First response", "Second response"],
        json_responses=[{"category": "laptop", "max_budget": 1500.0, "confidence": 0.99}],
    )

    # 1. Text completion queue
    t1 = await fake.complete_text([{"role": "user", "content": "q1"}])
    assert t1 == "First response"
    t2 = await fake.complete_text([{"role": "user", "content": "q2"}])
    assert t2 == "Second response"
    t3 = await fake.complete_text([{"role": "user", "content": "q3"}])
    assert t3 == fake.default_text

    # 2. JSON completion queue & validation
    j1 = await fake.complete_json([{"role": "user", "content": "q4"}], schema=UserRequirement)
    assert isinstance(j1, UserRequirement)
    assert j1.max_budget == 1500.0

    # 3. Call history tracking
    assert fake.call_count == 4
    assert fake.last_messages == [{"role": "user", "content": "q4"}]

    # 4. Ping
    assert await fake.ping() is True

    # 5. Simulated failure modes
    fake_err = FakeLLMClient(simulate_invalid_json=True)
    with pytest.raises(LLMSchemaValidationError):
        await fake_err.complete_json([{"role": "user", "content": "hi"}], schema=UserRequirement)

    fake_timeout = FakeLLMClient(error_to_raise=LLMTimeoutError("Simulated timeout"))
    with pytest.raises(LLMTimeoutError):
        await fake_timeout.complete_text([{"role": "user", "content": "hi"}])

    await fake.close()
    assert fake.is_closed is True


# ---------------------------------------------------------------------------
# 21. ping() lightweight connectivity checks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ping_success(valid_settings):
    """ping() returns True when provider models endpoint returns HTTP 200."""
    mock_resp = httpx.Response(status_code=200, content=b'{"data": []}')
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    result = await client.ping()

    assert result is True
    assert mock_client.get.call_count == 1
    call_args = mock_client.get.call_args
    assert "/models" in call_args[0][0]


@pytest.mark.asyncio
async def test_ping_auth_failure_raises_typed_error(valid_settings):
    """ping() raises LLMAuthenticationError on HTTP 401."""
    mock_resp = httpx.Response(status_code=401, content=b'{"error": "Unauthorized"}')
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMAuthenticationError):
        await client.ping()


@pytest.mark.asyncio
async def test_ping_timeout_raises_typed_error(valid_settings):
    """ping() raises LLMTimeoutError on network timeout."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Ping timed out"))

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMTimeoutError):
        await client.ping()


@pytest.mark.asyncio
async def test_ping_network_error_raises_typed_error(valid_settings):
    """ping() raises LLMNetworkError on network failure."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.NetworkError("DNS lookup failure"))

    client = GroqClient(settings=valid_settings, http_client=mock_client)
    with pytest.raises(LLMNetworkError):
        await client.ping()
