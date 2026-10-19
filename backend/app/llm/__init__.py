"""LLM client integration and structured output infrastructure.

Exports:
    LLMClient: Interface protocol for asynchronous LLM clients.
    GroqClient: Production client communicating with Groq Cloud OpenAI-compatible API.
    FakeLLMClient: Deterministic offline fake for testing and downstream modules.
    Errors: Typed and sanitized exception hierarchy.
"""

from app.llm.client import GroqClient, LLMClient, extract_json_from_text
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

__all__ = [
    "LLMClient",
    "GroqClient",
    "FakeLLMClient",
    "extract_json_from_text",
    "LLMError",
    "LLMConfigurationError",
    "LLMAuthenticationError",
    "LLMPermissionError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMNetworkError",
    "LLMProviderError",
    "LLMResponseFormatError",
    "LLMSchemaValidationError",
    "LLMRetryExhaustionError",
    "sanitize_error_message",
]
