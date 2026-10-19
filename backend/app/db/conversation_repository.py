"""Asynchronous repository for conversation session state persistence in MongoDB.

Provides atomic session persistence, optimistic concurrency control via versioning,
clean ownership checks, and an in-memory fallback repository for offline test isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo import ASCENDING, IndexModel
from pymongo.errors import PyMongoError

from app.core.errors import (
    AppError,
    ConcurrentUpdateError,
    ConversationNotFoundError,
    ConversationOwnershipError,
    DatabaseConnectionError,
)
from app.core.logging import get_logger
from app.db.mongo import get_mongo_db
from app.models.conversation import ConversationMessage, ConversationState

logger = get_logger("app.db.conversation_repository")


def check_conversation_ownership(
    conversation: ConversationState,
    requesting_user_id: str | None,
) -> None:
    """Validate ownership of a conversation if the conversation has an assigned owner.

    Args:
        conversation: Target ConversationState to inspect.
        requesting_user_id: User identifier supplied in the incoming request (e.g. from X-User-ID).

    Raises:
        ConversationOwnershipError: If conversation is owned by a user and caller does not match.
    """
    if conversation.user_id is not None:
        if not requesting_user_id or requesting_user_id.strip() != conversation.user_id.strip():
            logger.warning(
                "Conversation ownership violation: conversation '%s' belongs to '%s', attempted access by '%s'.",
                conversation.conversation_id,
                conversation.user_id,
                requesting_user_id,
            )
            raise ConversationOwnershipError(
                f"Access forbidden: conversation '{conversation.conversation_id}' belongs to another user."
            )


class ConversationRepository:
    """Asynchronous MongoDB repository for managing conversation state."""

    def __init__(
        self,
        collection_or_db: AsyncIOMotorCollection | AsyncIOMotorDatabase | None = None,
    ) -> None:
        if isinstance(collection_or_db, AsyncIOMotorCollection):
            self._collection = collection_or_db
        elif isinstance(collection_or_db, AsyncIOMotorDatabase):
            self._collection = collection_or_db["conversations"]
        else:
            self._collection = None

    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Return the active MongoDB collection for conversations."""
        if self._collection is not None:
            return self._collection
        return get_mongo_db()["conversations"]

    async def ensure_indexes(self) -> list[str]:
        """Create primary and secondary indexes on the conversations collection."""
        try:
            indexes = [
                IndexModel([("user_id", ASCENDING)], name="idx_conversations_user_id"),
                IndexModel([("updated_at", ASCENDING)], name="idx_conversations_updated_at"),
            ]
            return await self.collection.create_indexes(indexes)
        except PyMongoError as err:
            logger.error("Failed to create conversation indexes: %s", err)
            raise DatabaseConnectionError("Failed to initialize conversation indexes in MongoDB.") from err

    async def create(self, conversation: ConversationState) -> ConversationState:
        """Persist a new conversation document in MongoDB.

        Args:
            conversation: The new ConversationState to insert.

        Returns:
            The created ConversationState instance.

        Raises:
            DatabaseConnectionError: If database write operation fails.
        """
        doc = conversation.model_dump(mode="json")
        doc["_id"] = conversation.conversation_id

        try:
            await self.collection.insert_one(doc)
            logger.info("Created conversation session '%s'.", conversation.conversation_id)
            return conversation
        except PyMongoError as err:
            logger.error("Failed to create conversation document: %s", err)
            raise DatabaseConnectionError(
                f"Failed to persist new conversation in MongoDB: {type(err).__name__}"
            ) from err

    async def get(self, conversation_id: str) -> ConversationState | None:
        """Retrieve a conversation by its unique identifier.

        Args:
            conversation_id: Unique string identifier for the conversation.

        Returns:
            The validated ConversationState if found, or None.

        Raises:
            DatabaseConnectionError: If MongoDB read operation fails.
        """
        try:
            doc = await self.collection.find_one({"_id": conversation_id})
            if not doc:
                return None
            doc.pop("_id", None)
            return ConversationState.model_validate(doc)
        except PyMongoError as err:
            logger.error("Failed to query conversation '%s': %s", conversation_id, err)
            raise DatabaseConnectionError(
                f"Failed to fetch conversation from MongoDB: {type(err).__name__}"
            ) from err

    async def update(self, conversation: ConversationState) -> ConversationState:
        """Update an existing conversation with optimistic version locking.

        Args:
            conversation: ConversationState with modified attributes to persist.

        Returns:
            The updated ConversationState with incremented version counter.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ConcurrentUpdateError: If concurrent modification version conflict is detected.
            DatabaseConnectionError: If database write fails.
        """
        old_version = conversation.version
        conversation.version += 1
        conversation.updated_at = datetime.now(timezone.utc)
        doc = conversation.model_dump(mode="json")
        doc["_id"] = conversation.conversation_id

        try:
            result = await self.collection.replace_one(
                {"_id": conversation.conversation_id, "version": old_version},
                doc,
            )
            if result.matched_count == 0:
                # Determine if document was missing or if version conflicted
                existing = await self.collection.find_one({"_id": conversation.conversation_id})
                if existing is None:
                    raise ConversationNotFoundError(
                        f"Conversation '{conversation.conversation_id}' does not exist."
                    )
                raise ConcurrentUpdateError(
                    f"Conversation '{conversation.conversation_id}' was updated concurrently by another request."
                )

            logger.info(
                "Updated conversation '%s' (version %d -> %d).",
                conversation.conversation_id,
                old_version,
                conversation.version,
            )
            return conversation
        except AppError:
            raise
        except PyMongoError as err:
            logger.error("Failed to update conversation '%s': %s", conversation.conversation_id, err)
            raise DatabaseConnectionError(
                f"Failed to update conversation in MongoDB: {type(err).__name__}"
            ) from err

    async def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and its message history by conversation_id.

        Args:
            conversation_id: Unique string identifier for the conversation.

        Returns:
            True if conversation was found and deleted.

        Raises:
            ConversationNotFoundError: If conversation was not found.
            DatabaseConnectionError: If database delete operation fails.
        """
        try:
            result = await self.collection.delete_one({"_id": conversation_id})
            if result.deleted_count == 0:
                raise ConversationNotFoundError(f"Conversation '{conversation_id}' does not exist.")
            logger.info("Deleted conversation '%s'.", conversation_id)
            return True
        except AppError:
            raise
        except PyMongoError as err:
            logger.error("Failed to delete conversation '%s': %s", conversation_id, err)
            raise DatabaseConnectionError(
                f"Failed to delete conversation from MongoDB: {type(err).__name__}"
            ) from err

    async def get_messages(self, conversation_id: str) -> list[ConversationMessage] | None:
        """Retrieve the chronological message history for a conversation.

        Args:
            conversation_id: Unique string identifier for the conversation.

        Returns:
            List of ConversationMessages if found, or None.
        """
        conv = await self.get(conversation_id)
        if conv is None:
            return None
        return conv.messages


class InMemoryConversationRepository:
    """In-memory conversation repository for hermetic offline testing and mock execution."""

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, Any]] = {}
        self.simulate_db_failure: bool = False

    async def create(self, conversation: ConversationState) -> ConversationState:
        if self.simulate_db_failure:
            raise DatabaseConnectionError("Simulated database failure during create.")
        if conversation.conversation_id in self._storage:
            raise ValueError(f"Conversation '{conversation.conversation_id}' already exists.")
        doc = conversation.model_dump(mode="json")
        self._storage[conversation.conversation_id] = doc
        return conversation

    async def get(self, conversation_id: str) -> ConversationState | None:
        if self.simulate_db_failure:
            raise DatabaseConnectionError("Simulated database failure during get.")
        doc = self._storage.get(conversation_id)
        if doc is None:
            return None
        return ConversationState.model_validate(doc)

    async def update(self, conversation: ConversationState) -> ConversationState:
        if self.simulate_db_failure:
            raise DatabaseConnectionError("Simulated database failure during update.")
        existing = self._storage.get(conversation.conversation_id)
        if existing is None:
            raise ConversationNotFoundError(f"Conversation '{conversation.conversation_id}' does not exist.")
        if existing.get("version", 1) != conversation.version:
            raise ConcurrentUpdateError(
                f"Conversation '{conversation.conversation_id}' was updated concurrently by another request."
            )
        conversation.version += 1
        conversation.updated_at = datetime.now(timezone.utc)
        self._storage[conversation.conversation_id] = conversation.model_dump(mode="json")
        return conversation

    async def delete(self, conversation_id: str) -> bool:
        if self.simulate_db_failure:
            raise DatabaseConnectionError("Simulated database failure during delete.")
        if conversation_id not in self._storage:
            raise ConversationNotFoundError(f"Conversation '{conversation_id}' does not exist.")
        del self._storage[conversation_id]
        return True

    async def get_messages(self, conversation_id: str) -> list[ConversationMessage] | None:
        conv = await self.get(conversation_id)
        if conv is None:
            return None
        return conv.messages


def get_conversation_repository() -> ConversationRepository:
    """FastAPI dependency provider returning an active ConversationRepository instance."""
    return ConversationRepository()
