"""FastAPI application factory and entry point.

Provides lifecycle management (startup ping and graceful shutdown for MongoDB),
CORS configuration, structured logging, request ID tracking, and core status routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.core.errors import RequestIdMiddleware, register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.db.mongo import mongo_manager

logger = get_logger("app.main")


def create_app(
    settings: Optional[Settings] = None,
    init_db: bool = True,
) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        settings: Optional custom Settings instance (uses get_settings() by default).
        init_db: If True, connects and verifies local MongoDB during startup lifespan.
                 Set to False in tests to avoid live network/database calls.
    """
    app_settings = settings or get_settings()

    # Configure centralized logging
    setup_logging(log_level="INFO")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # --- Startup Phase ---
        logger.info(
            "Starting Electronics Recommendation Agent API v0.1.0 (env: %s)",
            app_settings.mongo_db,
        )

        if init_db:
            logger.info("Initializing local MongoDB connection...")
            client, db = await mongo_manager.connect_and_ping(app_settings)
            app.state.mongo_client = client
            app.state.mongo_db = db
            logger.info("Database initialization completed successfully.")
        else:
            logger.info("Database initialization skipped (testing/offline mode).")

        # Yield control to the running application
        yield

        # --- Shutdown Phase ---
        logger.info("Shutting down Electronics Recommendation Agent API...")
        if init_db:
            await mongo_manager.close()
        logger.info("Application shutdown complete.")

    app = FastAPI(
        title="Electronics Recommendation Agent API",
        version="0.1.0",
        description=(
            "Backend API for the Electronics Recommendation Agent. "
            "Delivers natural language requirement extraction and deterministic catalogue matching."
        ),
        lifespan=lifespan,
    )

    # Attach Request ID middleware
    app.add_middleware(RequestIdMiddleware)

    # Attach CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach centralized error handlers
    register_exception_handlers(app)

    # Register initial scaffold routes
    @app.get(
        "/",
        tags=["System"],
        summary="Root application status",
        response_model=dict,
    )
    async def root_status() -> dict[str, str]:
        """Return basic application metadata and running status."""
        return {
            "name": "Electronics Recommendation Agent API",
            "version": "0.1.0",
            "status": "running",
        }

    @app.get(
        "/live",
        tags=["System"],
        summary="Service liveness probe",
        response_model=dict,
    )
    async def live_probe() -> dict[str, str]:
        """Liveness check indicating the process is alive and receiving traffic."""
        return {"status": "alive"}

    # Prepare placeholder for future /api/v1 router
    api_v1_router = APIRouter(prefix="/api/v1")
    # Sub-routers for later modules will be registered here (Module 2+: catalogue, agent, etc.)
    app.include_router(api_v1_router)

    return app


# Default application instance for Uvicorn
app = create_app()
