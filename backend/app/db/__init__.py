"""Database access, connection management, and repositories."""

from app.db.mongo import get_mongo_client, get_mongo_db, mongo_manager
from app.db.repositories import ProductRepository

__all__ = ["ProductRepository", "get_mongo_client", "get_mongo_db", "mongo_manager"]
