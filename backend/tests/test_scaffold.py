"""Automated tests for Module 1 scaffold and configuration.

Verifies:
- Typed Settings parsing and validation
- CORS configuration parsing
- Masking of secrets in logs and representations
- App factory construction without network calls
- HTTP GET / and GET /live endpoints
- AppError response structure and request ID propagation
- FastAPI validation error preservation
- MongoDB connection error handling without credentials
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import FastAPI, Query
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError, DatabaseConnectionError
from app.db.mongo import MongoManager
from app.main import create_app


# ---------------------------------------------------------------------------
# Settings & Configuration Tests
# ---------------------------------------------------------------------------

def test_settings_valid_defaults():
    """Verify default Settings values can be loaded or constructed."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="test_key_12345",
        llm_model="test-model",
    )
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.mongo_db == "test_db"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_api_key == "test_key_12345"
    assert settings.llm_timeout_s == 30.0
    assert settings.llm_max_retries == 2


def test_settings_mongo_uri_validation():
    """Verify invalid or empty mongo_uri raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(mongo_uri="invalid_scheme://localhost:27017")

    with pytest.raises(ValidationError):
        Settings(mongo_uri="")


def test_settings_mongo_db_validation():
    """Verify invalid mongo_db name raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(mongo_db="")

    with pytest.raises(ValidationError):
        Settings(mongo_db="db with spaces")

    with pytest.raises(ValidationError):
        Settings(mongo_db="db.with.dots")


def test_settings_numeric_range_validation():
    """Verify invalid timeout and retry counts raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(llm_timeout_s=0)

    with pytest.raises(ValidationError):
        Settings(llm_timeout_s=-5.0)

    with pytest.raises(ValidationError):
        Settings(llm_max_retries=-1)


def test_settings_url_validation():
    """Verify malformed LLM URLs raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(llm_base_url="not-a-url")

    with pytest.raises(ValidationError):
        Settings(llm_fallback_base_url="ftp://invalid.com")


def test_settings_cors_origins_parsing():
    """Verify CORS_ORIGINS parses JSON array string and python list correctly."""
    # From JSON array string
    s1 = Settings(cors_origins='["http://localhost:3000", "http://localhost:5173"]')
    assert s1.cors_origins == ["http://localhost:3000", "http://localhost:5173"]

    # From Python list
    s2 = Settings(cors_origins=["http://localhost:8080"])
    assert s2.cors_origins == ["http://localhost:8080"]

    # From comma-separated string
    s3 = Settings(cors_origins="http://example.com, https://agent.app")
    assert s3.cors_origins == ["http://example.com", "https://agent.app"]


def test_settings_secret_masking():
    """Verify secrets are masked and not exposed in repr or properties."""
    settings = Settings(
        mongo_uri="mongodb://app_user:super_secret_password@localhost:27017",
        llm_api_key="gsk_1234567890abcdef1234567890",
    )
    # Masked representation checks
    repr_str = repr(settings)
    assert "super_secret_password" not in repr_str
    assert "gsk_1234567890abcdef1234567890" not in repr_str
    assert settings.masked_llm_api_key != "gsk_1234567890abcdef1234567890"
    assert "***" in settings.masked_mongo_uri


# ---------------------------------------------------------------------------
# App Factory & HTTP Endpoint Tests (Offline / No DB ping required)
# ---------------------------------------------------------------------------

@pytest.fixture
def test_app() -> FastAPI:
    """Create test application instance with database connection disabled."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_electronics",
        llm_base_url="https://api.groq.com/openai/v1",
        llm_api_key="test_dummy_key",
        cors_origins=["http://localhost:5173"],
    )
    return create_app(settings=settings, init_db=False)


@pytest.mark.asyncio
async def test_root_endpoint(test_app: FastAPI):
    """GET / must return HTTP 200 with application metadata."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Electronics Recommendation Agent API"
        assert data["version"] == "0.1.0"
        assert data["status"] == "running"
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
async def test_live_probe_endpoint(test_app: FastAPI):
    """GET /live must return HTTP 200 with alive status."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_app_error_handling(test_app: FastAPI):
    """Custom AppError must return structured JSON error with matching status."""
    # Add a temporary test route raising an AppError
    @test_app.get("/test-app-error")
    async def raise_app_error():
        raise AppError(
            code="RESOURCE_NOT_FOUND",
            status=404,
            message="The requested electronic item does not exist.",
        )

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/test-app-error")
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "RESOURCE_NOT_FOUND"
        assert data["error"]["message"] == "The requested electronic item does not exist."
        assert "request_id" in data["error"]


@pytest.mark.asyncio
async def test_request_id_propagation(test_app: FastAPI):
    """Incoming valid X-Request-ID must be preserved across response and body."""
    custom_id = "test-request-id-123456"

    @test_app.get("/test-request-id")
    async def request_id_endpoint():
        return {"status": "ok"}

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/test-request-id",
            headers={"X-Request-ID": custom_id},
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == custom_id


@pytest.mark.asyncio
async def test_validation_error_preserved(test_app: FastAPI):
    """FastAPI RequestValidationError must return HTTP 422 with standard detail."""
    @test_app.get("/test-validation")
    async def validation_endpoint(item_id: int = Query(...)):
        return {"item_id": item_id}

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/test-validation?item_id=not-an-integer")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)


@pytest.mark.asyncio
async def test_unhandled_exception_returns_500(test_app: FastAPI):
    """Unhandled server errors must return HTTP 500 without leaking stack traces."""
    @test_app.get("/test-unhandled-exception")
    async def unhandled_endpoint():
        raise RuntimeError("Secret internal failure detail that must not leak")

    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        response = await client.get("/test-unhandled-exception")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "INTERNAL_ERROR"
        assert data["error"]["message"] == "An unexpected error occurred."
        assert "request_id" in data["error"]
        # Ensure raw exception text is NOT leaked
        assert "Secret internal failure detail" not in json.dumps(data)


# ---------------------------------------------------------------------------
# MongoDB Connection Manager Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mongo_manager_ping_failure():
    """Verify MongoManager raises DatabaseConnectionError when ping fails."""
    manager = MongoManager()
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
    )

    with patch("app.db.mongo.AsyncIOMotorClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(side_effect=Exception("Connection refused"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(DatabaseConnectionError) as exc_info:
            await manager.connect_and_ping(settings=settings, timeout_ms=500)

        assert "Local MongoDB is unreachable" in str(exc_info.value.message) or \
               "Unexpected error" in str(exc_info.value.message)
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_mongo_manager_ping_success_and_close():
    """Verify MongoManager connects, validates ping, and closes cleanly."""
    manager = MongoManager()
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
    )

    with patch("app.db.mongo.AsyncIOMotorClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1.0})
        mock_client_cls.return_value = mock_client

        client, db = await manager.connect_and_ping(settings=settings)
        assert client is mock_client
        assert manager.client is mock_client
        assert manager.db is not None

        await manager.close()
        assert manager.client is None
        assert manager.db is None
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_app_lifespan_startup_ping_failure():
    """Verify application lifespan startup fails when MongoDB ping fails."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
    )
    app = create_app(settings=settings, init_db=True)

    with patch("app.db.mongo.AsyncIOMotorClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(side_effect=Exception("Database unreachable"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(DatabaseConnectionError):
            async with app.router.lifespan_context(app):
                pass


@pytest.mark.asyncio
async def test_app_lifespan_startup_success_and_shutdown():
    """Verify application lifespan registers db on state and closes on exit."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
    )
    app = create_app(settings=settings, init_db=True)

    with patch("app.db.mongo.AsyncIOMotorClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.admin.command = AsyncMock(return_value={"ok": 1.0})
        mock_client_cls.return_value = mock_client

        async with app.router.lifespan_context(app):
            assert hasattr(app.state, "mongo_client")
            assert hasattr(app.state, "mongo_db")

        mock_client.close.assert_called_once()


def test_os_env_overrides_dotenv(monkeypatch):
    """Verify that OS environment variables take precedence over defaults or .env."""
    monkeypatch.setenv("MONGO_DB", "custom_override_db")
    monkeypatch.setenv("LLM_TIMEOUT_S", "45")

    settings = Settings()
    assert settings.mongo_db == "custom_override_db"
    assert settings.llm_timeout_s == 45.0


def test_no_groq_call_during_construction_or_startup():
    """Verify that constructing Settings or app never makes HTTP calls to Groq."""
    with patch("httpx.AsyncClient.post") as mock_post:
        settings = Settings(llm_api_key="placeholder_key")
        app = create_app(settings=settings, init_db=False)
        assert app is not None
        mock_post.assert_not_called()

