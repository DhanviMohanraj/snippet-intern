"""Asynchronous MongoDB connection management using Motor.

Handles client lifecycle, connection validation via admin ping,
safe credential handling, and graceful shutdown.
"""

from __future__ import annotations

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from app.core.config import Settings, get_settings
from app.core.errors import DatabaseConnectionError
from app.core.logging import get_logger

logger = get_logger("app.db.mongo")


class MongoManager:
    """Manages the asynchronous Motor client lifecycle and connection verification."""

    def __init__(self) -> None:
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect_and_ping(
        self,
        settings: Optional[Settings] = None,
        timeout_ms: int = 2500,
    ) -> tuple[AsyncIOMotorClient, AsyncIOMotorDatabase]:
        """Establish connection to local MongoDB and verify connectivity with a ping.

        Does not print credentials in logs or error messages.
        """
        if self.client is not None and self.db is not None:
            return self.client, self.db

        cfg = settings or get_settings()
        masked_uri = cfg.masked_mongo_uri

        logger.info("Initializing connection to MongoDB at %s (database: %s)...", masked_uri, cfg.mongo_db)

        try:
            client = AsyncIOMotorClient(
                cfg.mongo_uri,
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
            )
            # Verify connectivity via admin command
            await client.admin.command("ping")
            database = client[cfg.mongo_db]

            self.client = client
            self.db = database

            logger.info("MongoDB ping succeeded. Connected to database '%s'.", cfg.mongo_db)
            return self.client, self.db

        except (ServerSelectionTimeoutError, ConnectionFailure) as err:
            if "client" in locals():
                client.close()
            error_msg = (
                f"Local MongoDB is unreachable at {masked_uri}. "
                f"Ensure the local MongoDB service is started and listening on the configured port. "
                f"Original error: {type(err).__name__}"
            )
            logger.error("MongoDB connection verification failed: %s", error_msg)
            raise DatabaseConnectionError(message=error_msg) from err
        except PyMongoError as err:
            if "client" in locals():
                client.close()
            error_msg = (
                f"MongoDB error encountered while connecting to {masked_uri}: {type(err).__name__}"
            )
            logger.error("MongoDB connection verification failed: %s", error_msg)
            raise DatabaseConnectionError(message=error_msg) from err
        except Exception as err:
            if "client" in locals():
                client.close()
            error_msg = (
                f"Unexpected error while attempting MongoDB connection at {masked_uri}: {err}"
            )
            logger.error(error_msg)
            raise DatabaseConnectionError(message=error_msg) from err

    async def close(self) -> None:
        """Close the Motor client and clean up database references."""
        if self.client is not None:
            logger.info("Closing MongoDB connection client...")
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB client closed cleanly.")


# Singleton manager instance
mongo_manager = MongoManager()


def get_mongo_client() -> AsyncIOMotorClient:
    """Dependency helper to get the active AsyncIOMotorClient instance."""
    if mongo_manager.client is None:
        raise DatabaseConnectionError(
            message="MongoDB client has not been initialized. Lifespan startup may not have run."
        )
    return mongo_manager.client


def get_mongo_db() -> AsyncIOMotorDatabase:
    """Dependency helper to get the active AsyncIOMotorDatabase instance."""
    if mongo_manager.db is None:
        raise DatabaseConnectionError(
            message="MongoDB database has not been initialized. Lifespan startup may not have run."
        )
    return mongo_manager.db
