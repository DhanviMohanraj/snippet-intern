"""Diagnostic health check endpoint for monitoring system dependencies.

Checks local MongoDB ping and Groq Cloud API availability independently,
returning a 200 OK with 'ok' or 'degraded' status.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, Request
import httpx
from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.mongo import mongo_manager
from app.models.api import DependencyStatus, HealthResponse, OverallHealthStatus

logger = get_logger("app.api.health")

router = APIRouter(prefix="/health", tags=["System"])


async def check_mongodb_health(client: Optional[AsyncIOMotorClient]) -> DependencyStatus:
    """Execute admin ping against existing application-owned MongoDB client."""
    if client is None:
        return "down"

    try:
        # Bounded ping check (2.0s timeout)
        await asyncio.wait_for(client.admin.command("ping"), timeout=2.0)
        return "up"
    except Exception as err:
        logger.warning("MongoDB health check ping failed: %s", type(err).__name__)
        return "down"


async def check_groq_health(settings: Settings) -> DependencyStatus:
    """Perform a lightweight, bounded availability check against Groq Cloud API.

    Does not make an outbound request if the API key is empty or still set
    to the development placeholder.
    """
    key = settings.llm_api_key.strip() if settings.llm_api_key else ""
    if not key or key == "replace_me" or key.startswith("replace"):
        logger.debug("Groq health check skipped: placeholder or missing API key.")
        return "down"

    # Construct models endpoint URL without duplicating /v1
    base_url = settings.llm_base_url.rstrip("/")
    if base_url.endswith("/v1"):
        models_url = f"{base_url}/models"
    else:
        models_url = f"{base_url}/v1/models"

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                return "up"
            logger.warning(
                "Groq health check returned non-200 status code: %d",
                resp.status_code,
            )
            return "down"
    except (httpx.TimeoutException, httpx.NetworkError) as err:
        logger.warning("Groq health check network/timeout error: %s", type(err).__name__)
        return "down"
    except Exception as err:
        logger.warning("Groq health check unexpected failure: %s", type(err).__name__)
        return "down"


@router.get(
    "",
    summary="Service dependency health check",
    description=(
        "Diagnostic health endpoint reporting local MongoDB and Groq Cloud API availability. "
        "Returns HTTP 200 with status='ok' when all dependencies are healthy, or status='degraded' "
        "when one or more dependencies are offline."
    ),
    response_model=HealthResponse,
    responses={
        200: {"description": "Health status successfully retrieved."},
    },
)
async def get_health_status(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Evaluate and report system health status."""
    # Obtain application-owned client from app.state or mongo_manager singleton
    mongo_client: Optional[AsyncIOMotorClient] = getattr(
        request.app.state, "mongo_client", None
    ) or mongo_manager.client

    db_status = await check_mongodb_health(mongo_client)
    llm_status = await check_groq_health(settings)

    overall_status: OverallHealthStatus = (
        "ok" if (db_status == "up" and llm_status == "up") else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        db=db_status,
        llm=llm_status,
    )
