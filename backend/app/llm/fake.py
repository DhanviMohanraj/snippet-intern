"""Deterministic offline fake LLM client for testing and downstream development.

Conforms to LLMClient protocol without making any outbound network connections.
Supports simulating canned responses, custom schemas, timeouts, and provider errors.
"""

from __future__ import annotations

from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError

from app.llm.client import LLMClient
from app.llm.errors import (
    LLMError,
    LLMSchemaValidationError,
)

T = TypeVar("T", bound=BaseModel)


class FakeLLMClient:
    """Deterministic, offline fake implementation of LLMClient protocol."""

    def __init__(
        self,
        text_responses: list[str] | None = None,
        default_text: str = "This is a deterministic fake response.",
        json_responses: list[dict[str, Any] | BaseModel] | None = None,
        default_json: dict[str, Any] | None = None,
        error_to_raise: Exception | None = None,
        simulate_invalid_json: bool = False,
        simulate_schema_error: bool = False,
        ping_result: bool = True,
    ) -> None:
        """Initialize FakeLLMClient with canned responses or simulated failure modes."""
        self._text_responses = list(text_responses or [])
        self.default_text = default_text
        self._json_responses = list(json_responses or [])
        self.default_json = default_json
        self.error_to_raise = error_to_raise
        self.simulate_invalid_json = simulate_invalid_json
        self.simulate_schema_error = simulate_schema_error
        self.ping_result = ping_result
        self.is_closed = False

        # History tracking for test assertions
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Total number of client method calls made."""
        return len(self.calls)

    @property
    def last_messages(self) -> list[dict[str, str]] | None:
        """Most recent messages payload sent to complete_text or complete_json."""
        if not self.calls:
            return None
        return self.calls[-1].get("messages")

    async def complete_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.2,
    ) -> str:
        """Return canned text response or raise simulated exception."""
        self.calls.append({
            "method": "complete_text",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self._text_responses:
            return self._text_responses.pop(0)

        return self.default_text

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> T:
        """Return validated Pydantic model instance from canned payload or raise simulated exception."""
        self.calls.append({
            "method": "complete_json",
            "messages": messages,
            "schema": schema,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self.simulate_invalid_json:
            raise LLMSchemaValidationError("FakeLLMClient simulated malformed JSON decoding failure.")

        if self.simulate_schema_error:
            raise LLMSchemaValidationError("FakeLLMClient simulated Pydantic schema validation failure.")

        # If canned json_responses provided, pop next
        if self._json_responses:
            item = self._json_responses.pop(0)
            if isinstance(item, schema):
                return item
            if isinstance(item, dict):
                try:
                    return schema.model_validate(item)
                except ValidationError as err:
                    raise LLMSchemaValidationError(
                        f"FakeLLMClient canned payload failed schema validation: {err}"
                    ) from err

        # If default_json is provided
        if self.default_json is not None:
            try:
                return schema.model_validate(self.default_json)
            except ValidationError as err:
                raise LLMSchemaValidationError(
                    f"FakeLLMClient default_json failed schema validation: {err}"
                ) from err

        # Fallback: attempt to instantiate schema with empty dictionary
        try:
            return schema.model_validate({})
        except ValidationError as err:
            raise LLMSchemaValidationError(
                f"FakeLLMClient cannot construct empty instance of {schema.__name__}: {err}. "
                "Provide canned json_responses or default_json."
            ) from err

    async def ping(self) -> bool:
        """Simulate ping connectivity check."""
        self.calls.append({"method": "ping"})
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return self.ping_result

    async def close(self) -> None:
        """Mark fake client as closed."""
        self.is_closed = True

    async def __aenter__(self) -> FakeLLMClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
