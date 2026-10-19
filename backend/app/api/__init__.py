"""API package initialization.

Exposes product catalogue and health check routers.
"""

from app.api.health import router as health_router
from app.api.products import router as products_router

__all__ = ["health_router", "products_router"]
