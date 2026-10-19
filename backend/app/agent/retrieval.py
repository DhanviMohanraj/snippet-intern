"""Product candidate retrieval engine.

Retrieves initial product candidate records from the MongoDB catalogue using the
existing ProductRepository abstraction. Only products belonging to the requested
category are fetched, ensuring deterministic, bounded, and typed candidate sets.
No network or LLM operations are performed.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.db.repositories import ProductRepository
from app.models.product import Product
from app.models.requirements import Requirements

logger = get_logger("app.agent.retrieval")

# Default upper limit for candidate retrieval per category
DEFAULT_CANDIDATE_LIMIT: int = 50


async def retrieve_candidates(
    requirements: Requirements,
    repository: ProductRepository,
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[Product]:
    """Retrieve candidate products for the category specified in Requirements.

    Queries the existing MongoDB catalogue via ProductRepository. Only products
    matching the validated category enum are retrieved. If requirements.category
    is None, an empty list is immediately returned.

    Args:
        requirements: Current user requirements specifying category and constraints.
        repository: Injected ProductRepository instance.
        limit: Maximum number of products to fetch (clamped to 1..50 per repo constraints).

    Returns:
        List of validated Product objects.

    Raises:
        Exception: Any underlying database or repository error is propagated directly,
                   ensuring database errors are distinguishable from legitimate empty results.
    """
    if requirements.category is None:
        logger.debug("Candidate retrieval skipped: requirements.category is None.")
        return []

    # Clamp page_size safely to repository boundaries (1 to 50)
    clamped_limit = max(1, min(limit, 50))

    filters = {"category": requirements.category}

    logger.debug(
        "Retrieving candidates for category='%s' with limit=%d",
        requirements.category.value,
        clamped_limit,
    )

    products, _ = await repository.find(
        filters=filters,
        sort="price_inr_asc",
        page=1,
        page_size=clamped_limit,
    )

    logger.info(
        "Retrieved %d candidates for category='%s' from catalogue.",
        len(products),
        requirements.category.value,
    )
    return products
