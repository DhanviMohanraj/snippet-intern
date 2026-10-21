"""Catalogue Evidence Indexer for SpecSage Hybrid RAG.

Converts verified catalogue Product records into normalized EvidenceDocument
instances with strict field-level provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from app.agent.rag.models import EvidenceDocument
from app.core.logging import get_logger
from app.models.product import Category, Product

logger = get_logger("app.agent.rag.indexer")

DEFAULT_CATALOGUE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "products.json"


def _format_spec_content(key: str, value: Any) -> str:
    """Format structured specification into unambiguous natural text for indexing."""
    if key == "ram_gb":
        return f"RAM: {value}GB memory"
    if key == "storage_gb":
        return f"Internal Storage: {value}GB SSD"
    if key == "cpu":
        return f"Processor CPU: {value}"
    if key == "gpu":
        return f"Graphics Processor GPU: {value}"
    if key == "vram_gb":
        return f"Dedicated Video Memory (VRAM): {value}GB"
    if key == "battery_hr":
        return f"Battery Runtime: {value} hours"
    if key == "battery_mah":
        return f"Battery Capacity: {value}mAh"
    if key == "weight_kg":
        return f"Weight: {value}kg"
    if key == "weight_g":
        return f"Weight: {value} grams"
    if key == "camera_mp":
        return f"Camera Resolution: {value} Megapixels (MP)"
    if key in ("screen_in", "size_in"):
        return f"Screen Size: {value} inches display"
    if key == "resolution":
        return f"Display Resolution: {value}"
    if key == "refresh_hz":
        return f"Display Refresh Rate: {value}Hz"
    if key == "panel":
        return f"Display Panel Type: {value}"
    if key == "ports":
        port_list = ", ".join(value) if isinstance(value, list) else str(value)
        return f"Connectivity Ports: {port_list}"
    if key == "type":
        return f"Headphone Form Factor: {value}"
    if key == "anc":
        return f"Active Noise Cancellation (ANC): {'Verified Present (Active)' if value else 'Passive noise isolation only (No ANC)'}"
    if key == "wireless":
        return f"Wireless Connectivity: {'Yes (Wireless Bluetooth)' if value else 'No (Wired Only)'}"
    if key == "stylus_support":
        return f"Stylus Pen Support: {'Active stylus input supported' if value else 'No stylus support'}"
    if key == "chipset":
        return f"Chipset Processor: {value}"
    if key == "os":
        return f"Operating System: {value}"
    if key == "codec":
        return f"Supported Audio Codecs: {value}"
    return f"{key.replace('_', ' ').title()}: {value}"


def build_product_evidence_documents(product: Product) -> list[EvidenceDocument]:
    """Convert a single product catalogue record into normalized evidence documents.

    Every piece of evidence tracks its source field and exact recorded value.
    No specifications are generated or assumed if missing from the catalogue.
    """
    docs: list[EvidenceDocument] = []
    pid = product.product_id
    pname = product.name
    cat = product.category

    # 1. Base Metadata
    docs.append(
        EvidenceDocument(
            document_id=f"{pid}:name",
            product_id=pid,
            category=cat,
            product_name=pname,
            source_field="name",
            recorded_value=product.name,
            content=f"Product Model Name: {product.name}",
            provenance={"field": "name", "value": product.name, "product_id": pid},
        )
    )
    docs.append(
        EvidenceDocument(
            document_id=f"{pid}:brand",
            product_id=pid,
            category=cat,
            product_name=pname,
            source_field="brand",
            recorded_value=product.brand,
            content=f"Brand Manufacturer: {product.brand}",
            provenance={"field": "brand", "value": product.brand, "product_id": pid},
        )
    )
    docs.append(
        EvidenceDocument(
            document_id=f"{pid}:price_inr",
            product_id=pid,
            category=cat,
            product_name=pname,
            source_field="price_inr",
            recorded_value=product.price_inr,
            content=f"Official Price: ₹{product.price_inr:,} INR",
            provenance={"field": "price_inr", "value": product.price_inr, "product_id": pid},
        )
    )
    if product.rating is not None:
        docs.append(
            EvidenceDocument(
                document_id=f"{pid}:rating",
                product_id=pid,
                category=cat,
                product_name=pname,
                source_field="rating",
                recorded_value=product.rating,
                content=f"Customer Review Rating: {product.rating} / 5.0 stars",
                provenance={"field": "rating", "value": product.rating, "product_id": pid},
            )
        )
    docs.append(
        EvidenceDocument(
            document_id=f"{pid}:in_stock",
            product_id=pid,
            category=cat,
            product_name=pname,
            source_field="in_stock",
            recorded_value=product.in_stock,
            content=f"Stock Availability: {'In Stock' if product.in_stock else 'Out of Stock'}",
            provenance={"field": "in_stock", "value": product.in_stock, "product_id": pid},
        )
    )

    # 2. Specifications (only populated specs are indexed)
    for spec_k, spec_v in (product.specs or {}).items():
        if spec_v is None:
            continue
        formatted_text = _format_spec_content(spec_k, spec_v)
        docs.append(
            EvidenceDocument(
                document_id=f"{pid}:specs.{spec_k}",
                product_id=pid,
                category=cat,
                product_name=pname,
                source_field=f"specs.{spec_k}",
                recorded_value=spec_v,
                content=formatted_text,
                provenance={
                    "field": f"specs.{spec_k}",
                    "spec_key": spec_k,
                    "recorded_value": spec_v,
                    "product_id": pid,
                    "category": cat.value,
                },
            )
        )

    # 3. Product Features
    for idx, feat in enumerate(product.features or []):
        docs.append(
            EvidenceDocument(
                document_id=f"{pid}:features[{idx}]",
                product_id=pid,
                category=cat,
                product_name=pname,
                source_field=f"features[{idx}]",
                recorded_value=feat,
                content=f"Product Feature: {feat}",
                provenance={
                    "field": f"features[{idx}]",
                    "feature_index": idx,
                    "recorded_value": feat,
                    "product_id": pid,
                },
            )
        )

    # 4. Intended Use Cases
    for idx, uc in enumerate(product.intended_use or []):
        docs.append(
            EvidenceDocument(
                document_id=f"{pid}:intended_use[{idx}]",
                product_id=pid,
                category=cat,
                product_name=pname,
                source_field=f"intended_use[{idx}]",
                recorded_value=uc,
                content=f"Intended Use Case Persona: {uc}",
                provenance={
                    "field": f"intended_use[{idx}]",
                    "use_case_index": idx,
                    "recorded_value": uc,
                    "product_id": pid,
                },
            )
        )

    return docs


class EvidenceIndex:
    """In-memory evidence index with fast lookup by document ID, product ID, and category."""

    def __init__(self, products: Sequence[Product] | None = None) -> None:
        self._documents: list[EvidenceDocument] = []
        self._doc_map: dict[str, EvidenceDocument] = {}
        self._by_product: dict[str, list[EvidenceDocument]] = {}
        self._by_category: dict[Category, list[EvidenceDocument]] = {}

        if products:
            self.build_from_products(products)

    def build_from_products(self, products: Sequence[Product]) -> None:
        """Index a collection of Product domain records into evidence documents."""
        self._documents.clear()
        self._doc_map.clear()
        self._by_product.clear()
        self._by_category.clear()

        for product in products:
            p_docs = build_product_evidence_documents(product)
            self._documents.extend(p_docs)
            self._by_product[product.product_id] = p_docs
            self._by_category.setdefault(product.category, []).extend(p_docs)
            for d in p_docs:
                self._doc_map[d.document_id] = d

        logger.info(
            "EvidenceIndex built successfully: %d evidence documents across %d products.",
            len(self._documents),
            len(products),
        )

    @classmethod
    def from_catalogue_file(cls, path: str | Path | None = None) -> EvidenceIndex:
        """Create an EvidenceIndex directly from a products.json catalogue file."""
        target_path = Path(path) if path else DEFAULT_CATALOGUE_PATH
        if not target_path.exists():
            logger.warning("Catalogue file '%s' not found; initializing empty index.", target_path)
            return cls([])

        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        products = [Product(**item) for item in raw_data]
        return cls(products)

    def get_documents(
        self,
        product_id: str | None = None,
        category: Category | None = None,
    ) -> list[EvidenceDocument]:
        """Retrieve evidence documents with optional product and category filtering."""
        if product_id is not None:
            docs = self._by_product.get(product_id, [])
            if category is not None:
                return [d for d in docs if d.category == category]
            return list(docs)

        if category is not None:
            return list(self._by_category.get(category, []))

        return list(self._documents)

    def get_document(self, document_id: str) -> EvidenceDocument | None:
        """Look up single evidence document by its unique document ID."""
        return self._doc_map.get(document_id)

    def get_product_evidence(self, product_id: str) -> list[EvidenceDocument]:
        """Retrieve all evidence documents strictly belonging to a specific product ID."""
        return list(self._by_product.get(product_id, []))

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def product_count(self) -> int:
        return len(self._by_product)
