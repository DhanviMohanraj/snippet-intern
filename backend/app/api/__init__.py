"""API package initialization.

Exposes product catalogue and health check routers.
"""

from app.api.health import router as health_router
from app.api.products import router as products_router
from app.api.v1.conversations import router as conversations_router

__all__ = ["conversations_router", "health_router", "products_router"]
