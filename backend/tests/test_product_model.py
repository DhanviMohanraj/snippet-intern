"""Unit tests for the Product domain model and specification validations."""

from __future__ import annotations

import math
import pytest
from pydantic import ValidationError

from app.models.product import Category, CategorySummary, Product


def test_valid_laptop_passes():
    """Verify that a properly structured laptop passes validation."""
    data = {
        "product_id": "LAP-0001",
        "name": "Zenit Air 14",
        "brand": "Zenit",
        "category": "laptop",
        "price_inr": 54999,
        "specs": {
            "cpu": "Intel Core i5-13420H",
            "ram_gb": 16,
            "storage_gb": 512,
            "gpu": "Intel Iris Xe Graphics",
            "vram_gb": 0,
            "weight_kg": 1.35,
            "battery_hr": 10.5,
            "screen_in": 14.0,
            "os": "Windows 11 Home",
        },
        "intended_use": ["programming", "student"],
        "features": ["Backlit Keyboard", "Fingerprint Reader"],
        "in_stock": True,
        "rating": 4.5,
    }
    product = Product.model_validate(data)
    assert product.product_id == "LAP-0001"
    assert product.category == Category.LAPTOP
    assert product.price_inr == 54999
    assert product.specs["ram_gb"] == 16


def test_valid_phone_passes():
    """Verify that a properly structured phone passes validation."""
    data = {
        "product_id": "PHN-0001",
        "name": "Velo Ultra 15 Pro",
        "brand": "Velo",
        "category": "phone",
        "price_inr": 119999,
        "specs": {
            "chipset": "Snapdragon 8 Gen 3",
            "ram_gb": 16,
            "storage_gb": 512,
            "camera_mp": 200.0,
            "battery_mah": 5000,
            "screen_in": 6.8,
            "weight_g": 220,
            "os": "Android 14",
        },
        "intended_use": ["photography", "gaming"],
        "features": ["Periscope Zoom", "120Hz Display"],
        "in_stock": True,
        "rating": 4.8,
    }
    product = Product.model_validate(data)
    assert product.category == Category.PHONE
    assert product.specs["camera_mp"] == 200.0


def test_valid_headphone_passes():
    """Verify that a properly structured headphone passes validation."""
    data = {
        "product_id": "HPH-0001",
        "name": "Sonix QuietMaster 100",
        "brand": "Sonix",
        "category": "headphones",
        "price_inr": 29999,
        "specs": {
            "type": "over-ear",
            "anc": True,
            "battery_hr": 35.0,
            "wireless": True,
            "weight_g": 250,
            "codec": "LDAC, AAC",
        },
        "intended_use": ["music", "travel"],
        "features": ["ANC", "Multi-Point"],
        "in_stock": True,
        "rating": 4.7,
    }
    product = Product.model_validate(data)
    assert product.category == Category.HEADPHONES
    assert product.specs["type"] == "over-ear"
    assert product.specs["anc"] is True


def test_valid_tablet_passes():
    """Verify that a properly structured tablet passes validation."""
    data = {
        "product_id": "TAB-0001",
        "name": "Aether Canvas Pro 13",
        "brand": "Aether",
        "category": "tablet",
        "price_inr": 89999,
        "specs": {
            "chipset": "Custom SoC",
            "ram_gb": 16,
            "storage_gb": 512,
            "screen_in": 12.9,
            "stylus_support": True,
            "battery_hr": 11.0,
            "weight_g": 680,
        },
        "intended_use": ["drawing", "note_taking"],
        "features": ["Stylus Included", "120Hz Display"],
        "in_stock": True,
        "rating": 4.8,
    }
    product = Product.model_validate(data)
    assert product.category == Category.TABLET
    assert product.specs["stylus_support"] is True


def test_valid_monitor_passes():
    """Verify that a properly structured monitor passes validation."""
    data = {
        "product_id": "MON-0001",
        "name": "Ocular CodeView 27",
        "brand": "Ocular",
        "category": "monitor",
        "price_inr": 28999,
        "specs": {
            "size_in": 27.0,
            "resolution": "2560x1440",
            "refresh_hz": 75,
            "panel": "IPS",
            "ports": ["HDMI 2.0", "DisplayPort 1.4", "USB-C 65W"],
        },
        "intended_use": ["programming", "office"],
        "features": ["90W USB-C", "Pivot Stand"],
        "in_stock": True,
        "rating": 4.7,
    }
    product = Product.model_validate(data)
    assert product.category == Category.MONITOR
    assert product.specs["refresh_hz"] == 75
    assert len(product.specs["ports"]) == 3


def test_unknown_category_rejected():
    """Verify that unsupported categories raise ValidationError."""
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "UNK-0001",
            "name": "Smart Watch",
            "brand": "Nova",
            "category": "smartwatch",
            "price_inr": 15000,
            "specs": {},
            "intended_use": ["fitness"],
        })


def test_missing_required_fields_rejected():
    """Verify that missing top-level required fields raise ValidationError."""
    # Missing name
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0001",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 50000,
            "intended_use": ["general"],
        })

    # Empty product_id
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "   ",
            "name": "Test Laptop",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 50000,
            "intended_use": ["general"],
        })


def test_negative_and_boolean_price_rejected():
    """Verify that invalid prices (negative, 0, or boolean) are rejected."""
    # Negative price
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0001",
            "name": "Test Laptop",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": -5000,
            "intended_use": ["general"],
        })

    # Zero price
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0001",
            "name": "Test Laptop",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 0,
            "intended_use": ["general"],
        })

    # Boolean price (Pydantic must not coerce True to 1)
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0001",
            "name": "Test Laptop",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": True,
            "intended_use": ["general"],
        })


def test_rating_boundary_validation():
    """Verify that rating must be between 0.0 and 5.0, or None."""
    base_data = {
        "product_id": "LAP-0001",
        "name": "Test Laptop",
        "brand": "Zenit",
        "category": "laptop",
        "price_inr": 50000,
        "intended_use": ["general"],
    }

    # None is allowed
    p1 = Product.model_validate({**base_data, "rating": None})
    assert p1.rating is None

    # 0.0 and 5.0 are allowed
    p2 = Product.model_validate({**base_data, "rating": 0.0})
    assert p2.rating == 0.0
    p3 = Product.model_validate({**base_data, "rating": 5.0})
    assert p3.rating == 5.0

    # > 5.0 rejected
    with pytest.raises(ValidationError):
        Product.model_validate({**base_data, "rating": 5.1})

    # < 0.0 rejected
    with pytest.raises(ValidationError):
        Product.model_validate({**base_data, "rating": -0.1})

    # NaN / Inf rejected
    with pytest.raises(ValidationError):
        Product.model_validate({**base_data, "rating": float("nan")})


def test_missing_specifications_accepted():
    """Verify that partially omitted specifications are accepted without fabrication."""
    p = Product.model_validate({
        "product_id": "LAP-0002",
        "name": "Minimal Laptop",
        "brand": "Voltix",
        "category": "laptop",
        "price_inr": 45000,
        "specs": {
            "cpu": "Intel Core i5-1235U",
            "ram_gb": 8,
            # storage_gb, gpu, vram_gb, battery_hr omitted deliberately
        },
        "intended_use": ["general"],
    })
    assert "cpu" in p.specs
    assert "ram_gb" in p.specs
    assert "storage_gb" not in p.specs


def test_unknown_category_specific_spec_keys_rejected():
    """Verify that category-inappropriate or misspelled spec keys are rejected."""
    # Laptop with phone camera_mp spec
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0003",
            "name": "Invalid Laptop",
            "brand": "Voltix",
            "category": "laptop",
            "price_inr": 50000,
            "specs": {
                "cpu": "Intel i5",
                "camera_mp": 108.0,  # Phone spec
            },
            "intended_use": ["general"],
        })

    # Phone with monitor ports spec
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "PHN-0002",
            "name": "Invalid Phone",
            "brand": "Velo",
            "category": "phone",
            "price_inr": 20000,
            "specs": {
                "chipset": "Snapdragon 680",
                "ports": ["HDMI 2.0"],  # Monitor spec
            },
            "intended_use": ["general"],
        })


def test_invalid_specification_types_rejected():
    """Verify that invalid types in specs (e.g. string for ram_gb) are rejected."""
    # ram_gb as string
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0004",
            "name": "Invalid Laptop",
            "brand": "Voltix",
            "category": "laptop",
            "price_inr": 50000,
            "specs": {"ram_gb": "16"},
            "intended_use": ["general"],
        })

    # anc as string instead of bool
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "HPH-0002",
            "name": "Invalid Headphone",
            "brand": "Sonix",
            "category": "headphones",
            "price_inr": 15000,
            "specs": {"type": "over-ear", "anc": "true"},
            "intended_use": ["music"],
        })

    # headphone type neither over-ear nor in-ear
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "HPH-0003",
            "name": "Invalid Headphone",
            "brand": "Sonix",
            "category": "headphones",
            "price_inr": 15000,
            "specs": {"type": "on-ear"},
            "intended_use": ["music"],
        })


def test_contradictory_laptop_discrete_gpu_vram():
    """Verify that a discrete RTX GPU cannot have vram_gb=0."""
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0005",
            "name": "Contradictory Laptop",
            "brand": "Apex",
            "category": "laptop",
            "price_inr": 90000,
            "specs": {
                "gpu": "NVIDIA GeForce RTX 4060",
                "vram_gb": 0,  # Contradictory: discrete GPU has VRAM
            },
            "intended_use": ["gaming"],
        })


def test_unknown_top_level_fields_forbidden():
    """Verify that extra top-level fields are rejected by extra='forbid'."""
    with pytest.raises(ValidationError):
        Product.model_validate({
            "product_id": "LAP-0006",
            "name": "Extra Field Laptop",
            "brand": "Zenit",
            "category": "laptop",
            "price_inr": 50000,
            "intended_use": ["general"],
            "extra_field_not_allowed": "malicious_or_unknown_value",
        })
