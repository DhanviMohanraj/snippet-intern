"""Automated tests for the Health Check API endpoint (/api/v1/health).

Verifies independent reporting of MongoDB and Groq Cloud availability,
mocked failure conditions (auth failure, timeouts, network errors, placeholder keys),
and absence of secret leaks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.health import check_groq_health, check_mongodb_health
from app.core.config import Settings
from app.main import create_app


# ---------------------------------------------------------------------------
# Unit tests for check_mongodb_health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_mongodb_health_success():
    """Returns 'up' when admin ping command succeeds."""
    mock_client = MagicMock()
    mock_client.admin.command = AsyncMock(return_value={"ok": 1.0})
    result = await check_mongodb_health(mock_client)
    assert result == "up"


@pytest.mark.asyncio
async def test_check_mongodb_health_failure():
    """Returns 'down' when admin ping command raises an exception."""
    mock_client = MagicMock()
    mock_client.admin.command = AsyncMock(side_effect=Exception("Connection timed out"))
    result = await check_mongodb_health(mock_client)
    assert result == "down"


@pytest.mark.asyncio
async def test_check_mongodb_health_none_client():
    """Returns 'down' when client is None."""
    result = await check_mongodb_health(None)
    assert result == "down"


# ---------------------------------------------------------------------------
# Unit tests for check_groq_health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_groq_health_placeholder_key_skips_network():
    """Placeholder key must return 'down' without attempting any network call."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="replace_me",
    )
    with patch("app.api.health.httpx.AsyncClient") as mock_client_cls:
        result = await check_groq_health(settings)
        assert result == "down"
        mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_check_groq_health_empty_key_skips_network():
    """Empty key must return 'down' without attempting any network call."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="",
    )
    with patch("app.api.health.httpx.AsyncClient") as mock_client_cls:
        result = await check_groq_health(settings)
        assert result == "down"
        mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_check_groq_health_success_200():
    """Returns 'up' when Groq models endpoint returns HTTP 200."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="gsk_valid_key_12345",
    )
    mock_resp = MagicMock(status_code=200)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.api.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_groq_health(settings)
        assert result == "up"


@pytest.mark.asyncio
async def test_check_groq_health_auth_failure_401():
    """Returns 'down' when Groq returns HTTP 401 Unauthorized."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="gsk_invalid_key",
    )
    mock_resp = MagicMock(status_code=401)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.api.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_groq_health(settings)
        assert result == "down"


@pytest.mark.asyncio
async def test_check_groq_health_timeout():
    """Returns 'down' when Groq connection times out."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="gsk_valid_key_timeout",
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.api.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_groq_health(settings)
        assert result == "down"


@pytest.mark.asyncio
async def test_check_groq_health_network_error():
    """Returns 'down' when a network error occurs."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_db",
        llm_api_key="gsk_valid_key_network",
    )
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.NetworkError("Network unreachable"))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("app.api.health.httpx.AsyncClient", return_value=mock_client):
        result = await check_groq_health(settings)
        assert result == "down"


# ---------------------------------------------------------------------------
# Integration tests for GET /api/v1/health endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def health_app():
    """Test FastAPI application instance for health checks."""
    settings = Settings(
        mongo_uri="mongodb://localhost:27017",
        mongo_db="test_health_db",
        llm_api_key="test_key",
    )
    return create_app(settings=settings, init_db=False)


@pytest.mark.asyncio
async def test_health_endpoint_both_up_status_ok(health_app):
    """When both MongoDB and Groq are up, endpoint returns status='ok'."""
    with (
        patch("app.api.health.check_mongodb_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.health.check_groq_health", new_callable=AsyncMock) as mock_llm,
    ):
        mock_db.return_value = "up"
        mock_llm.return_value = "up"

        async with AsyncClient(transport=ASGITransport(app=health_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["db"] == "up"
            assert data["llm"] == "up"
            assert "X-Request-ID" in resp.headers


@pytest.mark.asyncio
async def test_health_endpoint_mongodb_down_degraded(health_app):
    """When MongoDB is down and Groq is up, endpoint returns status='degraded'."""
    with (
        patch("app.api.health.check_mongodb_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.health.check_groq_health", new_callable=AsyncMock) as mock_llm,
    ):
        mock_db.return_value = "down"
        mock_llm.return_value = "up"

        async with AsyncClient(transport=ASGITransport(app=health_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["db"] == "down"
            assert data["llm"] == "up"


@pytest.mark.asyncio
async def test_health_endpoint_groq_down_degraded(health_app):
    """When Groq is down and MongoDB is up, endpoint returns status='degraded'."""
    with (
        patch("app.api.health.check_mongodb_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.health.check_groq_health", new_callable=AsyncMock) as mock_llm,
    ):
        mock_db.return_value = "up"
        mock_llm.return_value = "down"

        async with AsyncClient(transport=ASGITransport(app=health_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["db"] == "up"
            assert data["llm"] == "down"


@pytest.mark.asyncio
async def test_health_endpoint_both_down_degraded(health_app):
    """When both dependencies are down, endpoint returns status='degraded'."""
    with (
        patch("app.api.health.check_mongodb_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.health.check_groq_health", new_callable=AsyncMock) as mock_llm,
    ):
        mock_db.return_value = "down"
        mock_llm.return_value = "down"

        async with AsyncClient(transport=ASGITransport(app=health_app), base_url="http://test") as ac:
            resp = await ac.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["db"] == "down"
            assert data["llm"] == "down"


@pytest.mark.asyncio
async def test_health_endpoint_no_credential_leaks(health_app):
    """Ensure response payload strictly contains only status fields without secret leakage."""
    async with AsyncClient(transport=ASGITransport(app=health_app), base_url="http://test") as ac:
        resp = await ac.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.text
        assert "password" not in body
        assert "gsk_" not in body
        assert "mongodb://" not in body
        assert set(resp.json().keys()) == {"status", "db", "llm"}
