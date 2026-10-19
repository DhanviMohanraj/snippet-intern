"""Application error handling and exception definitions.

Provides a unified AppError structure, consistent JSON error responses,
and request ID tracking across the application lifecycle.
"""

from __future__ import annotations

import re
import uuid
from typing import Any
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.logging import get_logger

logger = get_logger("app.errors")

# Regex to validate acceptable client-supplied request IDs (alphanumeric, dashes, underscores)
VALID_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")


class AppError(Exception):
    """Base application exception with machine-readable code and HTTP status."""

    def __init__(self, code: str, status: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        """Convert error to response dictionary format."""
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if request_id:
            data["request_id"] = request_id
        return {"error": data}


class DatabaseConnectionError(AppError):
    """Raised when the database cannot be connected or pinged."""

    def __init__(
        self,
        message: str = "Database connection failed. Ensure local MongoDB is running.",
    ) -> None:
        super().__init__(
            code="DATABASE_UNAVAILABLE",
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
        )


class ConversationNotFoundError(AppError):
    """Raised when a requested conversation ID does not exist."""

    def __init__(self, message: str = "Conversation not found.") -> None:
        super().__init__(
            code="CONVERSATION_NOT_FOUND",
            status=status.HTTP_404_NOT_FOUND,
            message=message,
        )


class ConversationOwnershipError(AppError):
    """Raised when a user attempts to access a conversation they do not own."""

    def __init__(self, message: str = "Access forbidden: you do not own this conversation.") -> None:
        super().__init__(
            code="FORBIDDEN_CONVERSATION_ACCESS",
            status=status.HTTP_403_FORBIDDEN,
            message=message,
        )


class InvalidConversationIdError(AppError):
    """Raised when a conversation ID does not conform to required syntax."""

    def __init__(self, message: str = "Invalid conversation ID format.") -> None:
        super().__init__(
            code="INVALID_CONVERSATION_ID",
            status=status.HTTP_400_BAD_REQUEST,
            message=message,
        )


class InvalidMessageError(AppError):
    """Raised when an incoming user chat message is invalid."""

    def __init__(self, message: str = "User message cannot be empty or exceed length limits.") -> None:
        super().__init__(
            code="INVALID_MESSAGE",
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
        )


class ConcurrentUpdateError(AppError):
    """Raised when a conversation version conflict occurs during concurrent updates."""

    def __init__(self, message: str = "Conversation was modified concurrently by another request. Please retry.") -> None:
        super().__init__(
            code="CONCURRENT_UPDATE_CONFLICT",
            status=status.HTTP_409_CONFLICT,
            message=message,
        )


def get_request_id(request: Request) -> str:
    """Retrieve existing request ID from request state or return a fallback."""
    if hasattr(request.state, "request_id") and request.state.request_id:
        return str(request.state.request_id)
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique request ID to each incoming request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming_id = request.headers.get("x-request-id")
        if incoming_id and VALID_REQUEST_ID_REGEX.match(incoming_id):
            request_id = incoming_id
        else:
            request_id = uuid.uuid4().hex

        try:
            response = await call_next(request)
        except AppError as exc:
            response = await app_error_handler(request, exc)
        except (HTTPException, RequestValidationError):
            raise
        except Exception as exc:
            response = await unhandled_exception_handler(request, exc)

        response.headers["X-Request-ID"] = request_id
        return response


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle custom AppError exceptions."""
    request_id = get_request_id(request)
    logger.warning(
        "Application error occurred: [%s] %s (request_id=%s)",
        exc.code,
        exc.message,
        request_id,
    )
    return JSONResponse(
        status_code=exc.status,
        content=exc.to_dict(request_id=request_id),
        headers={"X-Request-ID": request_id},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected server-side exceptions without leaking tracebacks to clients."""
    request_id = get_request_id(request)
    logger.exception(
        "Unhandled server error encountered (request_id=%s): %s",
        request_id,
        str(exc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
            }
        },
        headers={"X-Request-ID": request_id},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers with the FastAPI application.

    Note: FastAPI's default handlers for HTTPException and RequestValidationError
    are preserved so validation errors and standard HTTP errors function as intended.
    """
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
