"""Reusable asynchronous Groq LLM client infrastructure.

Provides chat-completion text generation, structured JSON output with Pydantic v2
validation, bounded retries with exponential backoff, lightweight connectivity checks,
and safe credential handling.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Protocol, TypeVar, runtime_checkable
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
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

logger = get_logger("app.llm.client")

T = TypeVar("T", bound=BaseModel)

# Regex to strip markdown code fences from JSON output
JSON_CODE_BLOCK_PATTERN = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```$",
    re.DOTALL | re.IGNORECASE,
)


def extract_json_from_text(raw_text: str) -> str:
    """Extract raw JSON string, stripping optional markdown code fences."""
    cleaned = raw_text.strip()
    match = JSON_CODE_BLOCK_PATTERN.match(cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


@runtime_checkable
class LLMClient(Protocol):
    """Reusable asynchronous LLM client protocol."""

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """Send chat messages and return the completion as text."""
        ...

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> T:
        """Send chat messages and return a validated Pydantic model instance."""
        ...

    async def ping(self) -> bool:
        """Perform a lightweight provider connectivity check."""
        ...

    async def close(self) -> None:
        """Close underlying HTTP client resources."""
        ...


class GroqClient:
    """Asynchronous client for Groq Cloud OpenAI-compatible API."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize GroqClient with application settings or custom dependencies.

        Args:
            settings: Application settings instance. Defaults to get_settings().
            http_client: Optional pre-configured httpx.AsyncClient.
            sleep_func: Sleep coroutine for backoff delays (injectable for instant tests).
        """
        self._settings = settings or get_settings()
        self._sleep_func = sleep_func
        self._external_client = http_client
        self._internal_client: httpx.AsyncClient | None = None

        # Extract and normalize configuration
        self.base_url = (self._settings.llm_base_url or "").strip().rstrip("/")
        self.api_key = (self._settings.llm_api_key or "").strip()
        self.model = (self._settings.llm_model or "").strip()
        self.timeout_s = max(1.0, float(self._settings.llm_timeout_s or 30.0))

        # Bound max retries between 0 and 5 to prevent infinite loops or excessive API usage
        raw_retries = int(self._settings.llm_max_retries if self._settings.llm_max_retries is not None else 2)
        self.max_retries = max(0, min(raw_retries, 5))

        # Construct endpoint URLs following Groq OpenAI-compatible path conventions
        if self.base_url.endswith("/v1"):
            self._chat_url = f"{self.base_url}/chat/completions"
            self._models_url = f"{self.base_url}/models"
        else:
            self._chat_url = f"{self.base_url}/v1/chat/completions"
            self._models_url = f"{self.base_url}/v1/models"

    def _validate_config(self) -> None:
        """Validate LLM configuration parameters."""
        if not self.api_key or self.api_key == "replace_me" or self.api_key.startswith("replace"):
            raise LLMConfigurationError(
                "Groq LLM API key is not configured or is set to a development placeholder."
            )
        if not self.model:
            raise LLMConfigurationError("Groq LLM model name is not configured.")
        if not self.base_url:
            raise LLMConfigurationError("Groq LLM base URL is not configured.")

        parsed = urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise LLMConfigurationError(f"Invalid Groq LLM base URL: '{self.base_url}'.")

    def _get_headers(self) -> dict[str, str]:
        """Construct safe request headers including Authorization Bearer token."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtain active HTTP client instance."""
        if self._external_client is not None:
            return self._external_client
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._internal_client

    async def close(self) -> None:
        """Close internally managed HTTP client resources."""
        if self._internal_client is not None and not self._internal_client.is_closed:
            await self._internal_client.aclose()
            self._internal_client = None

    async def __aenter__(self) -> GroqClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        """Parse Retry-After header safely from an HTTP response."""
        header_val = response.headers.get("Retry-After")
        if header_val:
            try:
                seconds = float(header_val.strip())
                if seconds > 0:
                    return min(seconds, 30.0)
            except (ValueError, TypeError):
                pass
        return None

    def _calculate_backoff(self, attempt_index: int, retry_after: float | None = None) -> float:
        """Compute exponential backoff delay bounded between 0.5s and 10.0s."""
        if retry_after is not None:
            return retry_after
        initial_delay = 0.5
        backoff_factor = 2.0
        computed = initial_delay * (backoff_factor ** attempt_index)
        return min(computed, 10.0)

    def _extract_choice_content(self, data: dict[str, Any]) -> str:
        """Safely extract assistant message content from Groq response payload."""
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise LLMResponseFormatError("Groq API response contained no choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMResponseFormatError("Groq API choice element is not a JSON object.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseFormatError("Groq API choice contained no message object.")

        content = message.get("content")
        if content is None:
            raise LLMResponseFormatError("Groq API assistant message contained null content.")

        if not isinstance(content, str) or not content.strip():
            finish_reason = first_choice.get("finish_reason")
            if finish_reason == "length":
                raise LLMResponseFormatError(
                    "Groq API completion stopped due to token length limit before generating output. Increase max_tokens."
                )
            raise LLMResponseFormatError("Groq API assistant message returned empty content.")


        return content.strip()

    async def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute chat completion request with bounded retries and exponential backoff."""
        self._validate_config()
        client = await self._get_client()
        headers = self._get_headers()

        # Total attempts = 1 initial attempt + max_retries
        total_attempts = 1 + self.max_retries
        last_exception: Exception | None = None

        for attempt in range(total_attempts):
            try:
                resp = await client.post(
                    self._chat_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )

                # HTTP 200 OK
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except Exception as err:
                        raise LLMResponseFormatError(
                            f"Failed to decode provider response as JSON: {err}"
                        ) from err

                # Explicit non-retryable client errors
                if resp.status_code == 401:
                    raise LLMAuthenticationError(
                        "Authentication failed with Groq Cloud API. Verify configured API credentials."
                    )
                if resp.status_code == 403:
                    raise LLMPermissionError(
                        "Permission denied by Groq Cloud API for requested model or resource."
                    )

                # HTTP 429 Rate Limit (Retryable)
                if resp.status_code == 429:
                    retry_after = self._parse_retry_after(resp)
                    err = LLMRateLimitError(
                        "Groq Cloud API rate limit exceeded.",
                        retry_after=retry_after,
                    )
                    last_exception = err
                    if attempt < total_attempts - 1:
                        backoff = self._calculate_backoff(attempt, retry_after)
                        logger.warning(
                            "Groq rate limit encountered (attempt %d/%d). Backing off for %.2fs.",
                            attempt + 1,
                            total_attempts,
                            backoff,
                        )
                        await self._sleep_func(backoff)
                        continue
                    raise LLMRetryExhaustionError(
                        f"Rate limit retries exhausted after {total_attempts} attempts."
                    ) from err

                # HTTP 5xx Server Errors (Retryable)
                if 500 <= resp.status_code < 600:
                    err = LLMProviderError(
                        f"Groq Cloud API server error (HTTP {resp.status_code}).",
                        provider_status_code=resp.status_code,
                    )
                    last_exception = err
                    if attempt < total_attempts - 1:
                        backoff = self._calculate_backoff(attempt)
                        logger.warning(
                            "Groq server error HTTP %d (attempt %d/%d). Backing off for %.2fs.",
                            resp.status_code,
                            attempt + 1,
                            total_attempts,
                            backoff,
                        )
                        await self._sleep_func(backoff)
                        continue
                    raise LLMRetryExhaustionError(
                        f"Provider 5xx retries exhausted after {total_attempts} attempts."
                    ) from err

                # Other HTTP 4xx errors (e.g. 400 Bad Request) - non-retryable
                clean_detail = sanitize_error_message(resp.text[:200])
                raise LLMProviderError(
                    f"Groq Cloud API client error (HTTP {resp.status_code}): {clean_detail}",
                    provider_status_code=resp.status_code,
                )

            except (httpx.TimeoutException, TimeoutError) as exc:
                err = LLMTimeoutError(f"Groq API request timed out after {self.timeout_s}s.")
                last_exception = err
                if attempt < total_attempts - 1:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "Groq request timeout (attempt %d/%d). Backing off for %.2fs.",
                        attempt + 1,
                        total_attempts,
                        backoff,
                    )
                    await self._sleep_func(backoff)
                    continue
                raise LLMRetryExhaustionError(
                    f"Timeout retries exhausted after {total_attempts} attempts."
                ) from err

            except (httpx.NetworkError, ConnectionError) as exc:
                err = LLMNetworkError(
                    f"Network error connecting to Groq Cloud API: {sanitize_error_message(str(exc))}"
                )
                last_exception = err
                if attempt < total_attempts - 1:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        "Groq network failure (attempt %d/%d). Backing off for %.2fs.",
                        attempt + 1,
                        total_attempts,
                        backoff,
                    )
                    await self._sleep_func(backoff)
                    continue
                raise LLMRetryExhaustionError(
                    f"Network retries exhausted after {total_attempts} attempts."
                ) from err

            except (LLMAuthenticationError, LLMPermissionError, LLMConfigurationError):
                # Never retry authentication, permission, or configuration errors
                raise

        if last_exception:
            raise LLMRetryExhaustionError(
                f"Retry attempts exhausted after {total_attempts} attempts: {last_exception}"
            ) from last_exception
        raise LLMError("LLM_UNKNOWN_ERROR", 500, "Unexpected failure during chat completion.")

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """Send chat messages and return the generated text.

        Args:
            messages: List of chat message dictionaries with 'role' and 'content'.
            max_tokens: Maximum tokens to generate in response.
            temperature: Sampling temperature (0.0 to 2.0).

        Returns:
            The generated text response.
        """
        if not messages or not isinstance(messages, list):
            raise LLMConfigurationError("Messages must be a non-empty list of message dictionaries.")

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": max(0.0, float(temperature)),
        }

        data = await self._post_chat_completion(payload)
        return self._extract_choice_content(data)

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> T:
        """Send chat messages and return a validated Pydantic v2 model instance.

        Applies JSON mode formatting, parses the output, validates against the schema,
        and executes bounded schema correction retries if the response is malformed.

        Args:
            messages: List of chat message dictionaries.
            schema: Pydantic v2 BaseModel subclass.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Validated instance of the requested Pydantic schema.
        """
        if not issubclass(schema, BaseModel):
            raise LLMConfigurationError(f"Provided schema {schema} is not a Pydantic BaseModel subclass.")
        if not messages or not isinstance(messages, list):
            raise LLMConfigurationError("Messages must be a non-empty list of message dictionaries.")

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        schema_instruction = (
            "You MUST respond ONLY with a valid JSON object strictly matching this JSON Schema:\n"
            f"{schema_json}\n"
            "Do not include any explanation or markdown formatting outside the JSON."
        )

        # Clone messages to prevent modifying caller's data
        conversation = [dict(m) for m in messages]

        # Ensure schema instruction is included in the conversation
        has_system = any(m.get("role") == "system" for m in conversation)
        if has_system:
            # Append schema instruction to existing system message
            for m in conversation:
                if m.get("role") == "system":
                    m["content"] = f"{m.get('content', '')}\n\n{schema_instruction}".strip()
                    break
        else:
            conversation.insert(0, {"role": "system", "content": schema_instruction})

        # Structured output correction loop: at most min(max_retries, 2) correction attempts
        correction_retries = min(self.max_retries, 2)
        total_correction_attempts = 1 + correction_retries
        use_json_mode = True

        for attempt in range(total_correction_attempts):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": conversation,
                "max_tokens": max(1, int(max_tokens)),
                "temperature": max(0.0, float(temperature)),
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                data = await self._post_chat_completion(payload)
            except LLMProviderError as err:
                # Fallback if provider/model rejects response_format parameter
                if use_json_mode and err.provider_status_code == 400:
                    logger.warning(
                        "Groq model rejected response_format={'type': 'json_object'}. "
                        "Retrying with prompt-based JSON enforcement."
                    )
                    use_json_mode = False
                    payload.pop("response_format", None)
                    data = await self._post_chat_completion(payload)
                else:
                    raise

            raw_text = self._extract_choice_content(data)
            json_text = extract_json_from_text(raw_text)

            # Step 1: Parse JSON
            try:
                parsed_json = json.loads(json_text)
            except json.JSONDecodeError as decode_err:
                if attempt < total_correction_attempts - 1:
                    logger.warning(
                        "JSON decode failure (correction attempt %d/%d): %s. Requesting correction.",
                        attempt + 1,
                        total_correction_attempts,
                        decode_err,
                    )
                    conversation.append({"role": "assistant", "content": raw_text})
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"Your previous response was not valid JSON (error: {decode_err}). "
                            "Please output ONLY valid JSON matching the schema."
                        ),
                    })
                    await self._sleep_func(0.5)
                    continue
                raise LLMSchemaValidationError(
                    f"LLM response could not be parsed as JSON: {decode_err}"
                ) from decode_err

            # Step 2: Validate against Pydantic schema
            try:
                validated_model = schema.model_validate(parsed_json)
                return validated_model
            except ValidationError as val_err:
                if attempt < total_correction_attempts - 1:
                    err_summary = ", ".join(
                        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                        for e in val_err.errors()[:3]
                    )
                    logger.warning(
                        "Pydantic validation failure (correction attempt %d/%d): %s. Requesting correction.",
                        attempt + 1,
                        total_correction_attempts,
                        err_summary,
                    )
                    conversation.append({"role": "assistant", "content": raw_text})
                    conversation.append({
                        "role": "user",
                        "content": (
                            f"Your JSON response failed schema validation errors: {err_summary}. "
                            "Please fix these fields and output valid JSON adhering strictly to the schema."
                        ),
                    })
                    await self._sleep_func(0.5)
                    continue
                raise LLMSchemaValidationError(
                    f"LLM response failed schema validation: {val_err}"
                ) from val_err

        raise LLMSchemaValidationError("Structured JSON completion retries exhausted without valid schema output.")

    async def ping(self) -> bool:
        """Perform a lightweight provider connectivity check without generating text.

        Returns:
            True if connectivity to the provider endpoint is confirmed.

        Raises:
            LLMAuthenticationError: When credentials are invalid (HTTP 401).
            LLMPermissionError: When access is forbidden (HTTP 403).
            LLMTimeoutError: When the ping request times out.
            LLMNetworkError: When network connectivity fails.
            LLMProviderError: When provider returns a server error.
            LLMConfigurationError: When configuration is missing or invalid.
        """
        self._validate_config()
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        ping_timeout = min(self.timeout_s, 5.0)

        try:
            resp = await client.get(
                self._models_url,
                headers=headers,
                timeout=ping_timeout,
            )
            if resp.status_code == 200:
                return True
            if resp.status_code == 401:
                raise LLMAuthenticationError(
                    "Authentication failed during ping to Groq Cloud API."
                )
            if resp.status_code == 403:
                raise LLMPermissionError(
                    "Permission denied during ping to Groq Cloud API."
                )
            raise LLMProviderError(
                f"Groq API ping returned non-success status: {resp.status_code}",
                provider_status_code=resp.status_code,
            )
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise LLMTimeoutError(f"Groq API ping timed out after {ping_timeout}s.") from exc
        except (httpx.NetworkError, ConnectionError) as exc:
            raise LLMNetworkError(
                f"Network error during Groq API ping: {sanitize_error_message(str(exc))}"
            ) from exc
        except (LLMAuthenticationError, LLMPermissionError, LLMProviderError, LLMConfigurationError):
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"Unexpected failure during Groq API ping: {sanitize_error_message(str(exc))}"
            ) from exc

    def __repr__(self) -> str:
        """Provide a safe string representation omitting raw secret values."""
        masked_key = (
            f"{self.api_key[:4]}...{self.api_key[-4:]}"
            if len(self.api_key) > 8
            else "[REDACTED]"
        )
        return (
            f"GroqClient(base_url='{self.base_url}', "
            f"model='{self.model}', "
            f"api_key='{masked_key}', "
            f"timeout_s={self.timeout_s}, "
            f"max_retries={self.max_retries})"
        )
