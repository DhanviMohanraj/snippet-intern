"""Typed and sanitized exception hierarchy for LLM operations.

Provides structured error handling, preserving HTTP status mappings while ensuring
API keys, Bearer tokens, and sensitive credentials are never leaked in error messages.
"""

from __future__ import annotations

import re
from typing import Any
from app.core.errors import AppError

# Regex pattern to scrub any accidental token or key patterns
TOKEN_SCRUB_PATTERN = re.compile(r"(gsk_[a-zA-Z0-9_-]+|Bearer\s+[a-zA-Z0-9_\-\.]+)", re.IGNORECASE)


def sanitize_error_message(msg: str) -> str:
    """Scrub potential API keys or authorization tokens from error text."""
    if not isinstance(msg, str):
        return str(msg)
    return TOKEN_SCRUB_PATTERN.sub("[REDACTED_SECRET]", msg)


class LLMError(AppError):
    """Base exception for all LLM client and inference failures."""

    def __init__(self, code: str, status: int, message: str) -> None:
        clean_msg = sanitize_error_message(message)
        super().__init__(code=code, status=status, message=clean_msg)


class LLMConfigurationError(LLMError):
    """Raised when LLM settings are missing, invalid, or using development placeholders."""

    def __init__(
        self,
        message: str = "LLM provider configuration is missing or invalid.",
    ) -> None:
        super().__init__(
            code="LLM_CONFIG_ERROR",
            status=500,
            message=message,
        )


class LLMAuthenticationError(LLMError):
    """Raised on HTTP 401 Unauthorized errors from the LLM provider."""

    def __init__(
        self,
        message: str = "Authentication failed with Groq Cloud API. Verify configured API credentials.",
    ) -> None:
        super().__init__(
            code="LLM_AUTHENTICATION_ERROR",
            status=401,
            message=message,
        )


class LLMPermissionError(LLMError):
    """Raised on HTTP 403 Forbidden errors from the LLM provider."""

    def __init__(
        self,
        message: str = "Permission denied by Groq Cloud API for requested model or resource.",
    ) -> None:
        super().__init__(
            code="LLM_PERMISSION_ERROR",
            status=403,
            message=message,
        )


class LLMRateLimitError(LLMError):
    """Raised on HTTP 429 Too Many Requests responses from the LLM provider."""

    def __init__(
        self,
        message: str = "Groq Cloud API rate limit exceeded.",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            code="LLM_RATE_LIMIT_ERROR",
            status=429,
            message=message,
        )
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Raised when an LLM API request exceeds the configured timeout."""

    def __init__(
        self,
        message: str = "Groq Cloud API request timed out.",
    ) -> None:
        super().__init__(
            code="LLM_TIMEOUT",
            status=504,
            message=message,
        )


class LLMNetworkError(LLMError):
    """Raised on network connection drops, DNS lookup failures, or socket errors."""

    def __init__(
        self,
        message: str = "Failed to establish network connection to Groq Cloud API.",
    ) -> None:
        super().__init__(
            code="LLM_NETWORK_ERROR",
            status=503,
            message=message,
        )


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns a 5xx server-side error."""

    def __init__(
        self,
        message: str = "Groq Cloud API encountered an internal provider error.",
        provider_status_code: int | None = None,
    ) -> None:
        super().__init__(
            code="LLM_PROVIDER_ERROR",
            status=502,
            message=message,
        )
        self.provider_status_code = provider_status_code


class LLMResponseFormatError(LLMError):
    """Raised when the provider response is missing choices, message content, or is empty."""

    def __init__(
        self,
        message: str = "LLM provider returned an empty or malformed completion response.",
    ) -> None:
        super().__init__(
            code="LLM_RESPONSE_FORMAT_ERROR",
            status=502,
            message=message,
        )


class LLMSchemaValidationError(LLMError):
    """Raised when structured JSON output fails JSON decoding or Pydantic validation."""

    def __init__(
        self,
        message: str = "LLM response could not be validated against the requested Pydantic schema.",
    ) -> None:
        super().__init__(
            code="LLM_SCHEMA_VALIDATION_ERROR",
            status=502,
            message=message,
        )


class LLMRetryExhaustionError(LLMError):
    """Raised when transient retries have been exhausted without success."""

    def __init__(
        self,
        message: str = "Maximum retry attempts exhausted for LLM request.",
    ) -> None:
        super().__init__(
            code="LLM_RETRY_EXHAUSTED",
            status=502,
            message=message,
        )
