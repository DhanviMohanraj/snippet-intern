"""Requirements domain models for conversational product recommendation.

Defines the core Requirements, RequirementsDelta, HardConstraints, and Message models,
with strict typed validations, non-negative budget controls, bounded preference weights,
and safe removal tracking.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.product import Category

# Recognized soft preference names supported by the domain scoring layer
RECOGNIZED_PREFERENCES: set[str] = {
    "portability",
    "performance",
    "battery_life",
    "camera_quality",
    "display_quality",
    "gpu_power",
    "build_quality",
    "noise_cancellation",
}

CATEGORY_ALIASES: dict[str, Category] = {
    "laptop": Category.LAPTOP,
    "notebook": Category.LAPTOP,
    "lappy": Category.LAPTOP,
    "macbook": Category.LAPTOP,
    "ultrabook": Category.LAPTOP,
    "chromebook": Category.LAPTOP,
    "pc": Category.LAPTOP,
    "phone": Category.PHONE,
    "smartphone": Category.PHONE,
    "mobile": Category.PHONE,
    "cellphone": Category.PHONE,
    "android": Category.PHONE,
    "iphone": Category.PHONE,
    "headphones": Category.HEADPHONES,
    "headphone": Category.HEADPHONES,
    "earphones": Category.HEADPHONES,
    "earphone": Category.HEADPHONES,
    "headset": Category.HEADPHONES,
    "earbuds": Category.HEADPHONES,
    "earbud": Category.HEADPHONES,
    "tws": Category.HEADPHONES,
    "tablet": Category.TABLET,
    "tab": Category.TABLET,
    "ipad": Category.TABLET,
    "monitor": Category.MONITOR,
    "screen": Category.MONITOR,
    "display": Category.MONITOR,
}

PREFERENCE_ALIASES: dict[str, str] = {
    "portability": "portability",
    "portable": "portability",
    "lightweight": "portability",
    "weight": "portability",
    "slim": "portability",
    "performance": "performance",
    "speed": "performance",
    "fast": "performance",
    "battery_life": "battery_life",
    "battery": "battery_life",
    "endurance": "battery_life",
    "camera_quality": "camera_quality",
    "camera": "camera_quality",
    "display_quality": "display_quality",
    "display": "display_quality",
    "screen": "display_quality",
    "screen_quality": "display_quality",
    "gpu_power": "gpu_power",
    "gpu": "gpu_power",
    "graphics": "gpu_power",
    "build_quality": "build_quality",
    "build": "build_quality",
    "durability": "build_quality",
    "noise_cancellation": "noise_cancellation",
    "anc": "noise_cancellation",
}

REMOVAL_FIELD_ALIASES: dict[str, str] = {
    "budget": "budget_max_inr",
    "budget_max": "budget_max_inr",
    "budget_min": "budget_min_inr",
    "budget_limit": "budget_max_inr",
    "max_budget": "budget_max_inr",
    "min_budget": "budget_min_inr",
    "price": "budget_max_inr",
    "price_limit": "budget_max_inr",
    "weight": "portability",
    "lightweight": "portability",
    "portability_preference": "portability",
    "battery": "battery_life",
    "camera": "camera_quality",
    "display": "display_quality",
    "gpu": "requires_gpu",
    "gpu_preference": "gpu_power",
    "ram": "min_ram_gb",
    "storage": "min_storage_gb",
    "anc": "noise_cancellation",
}

# Recognized fields and preference/constraint keys that can be explicitly cleared
VALID_REMOVABLE_FIELDS: set[str] = {
    "category",
    "budget",
    "budget_max",
    "budget_min",
    "budget_max_inr",
    "budget_min_inr",
    "use_cases",
    "hard",
    "preferences",
    "ml_mode",
    "min_ram_gb",
    "min_storage_gb",
    "min_battery_hr",
    "max_weight_kg",
    "requires_gpu",
    "min_vram_gb",
    "exclude_brands",
    "in_stock",
} | RECOGNIZED_PREFERENCES | set(REMOVAL_FIELD_ALIASES.keys())

ML_MODES: set[str] = {"local", "cloud"}


class Message(BaseModel):
    """Single chat message representing conversation history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"] = Field(description="Role of message author")
    content: str = Field(description="Textual content of the message")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message content must be a non-empty string.")
        return v.strip()


class HardConstraints(BaseModel):
    """Explicit, non-negotiable product specifications constraints."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    min_ram_gb: int | None = Field(default=None, description="Minimum RAM in gigabytes", gt=0)
    min_storage_gb: int | None = Field(default=None, description="Minimum storage in gigabytes", gt=0)
    min_battery_hr: float | None = Field(default=None, description="Minimum battery runtime in hours", gt=0.0)
    max_weight_kg: float | None = Field(default=None, description="Maximum product weight in kilograms", gt=0.0)
    requires_gpu: bool | None = Field(default=None, description="Whether a dedicated GPU is strictly required")
    min_vram_gb: int | None = Field(default=None, description="Minimum dedicated VRAM in gigabytes", ge=0)
    exclude_brands: list[str] = Field(default_factory=list, description="Brands explicitly excluded by user")
    in_stock: bool | None = Field(default=None, description="Whether product must currently be in stock")

    @field_validator("exclude_brands")
    @classmethod
    def validate_exclude_brands(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("exclude_brands must be a list of strings.")
        return [b.strip() for b in v if isinstance(b, str) and b.strip()]


class Requirements(BaseModel):
    """Accumulated user requirements state across a session."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    category: Category | None = Field(default=None, description="Selected product category")
    use_cases: list[str] = Field(default_factory=list, description="List of normalized use cases / personas")
    budget_max_inr: int | None = Field(default=None, description="Maximum budget in INR", ge=0)
    budget_min_inr: int | None = Field(default=None, description="Minimum budget in INR", ge=0)
    hard: HardConstraints = Field(default_factory=HardConstraints, description="Hard filter constraints")
    preferences: dict[str, float] = Field(
        default_factory=dict,
        description="Soft preferences mapped to 0.0-1.0 numeric weights",
    )
    ml_mode: str | None = Field(default=None, description="Machine learning execution mode ('local' | 'cloud')")
    asked_slots: list[str] = Field(default_factory=list, description="Requirement slots already queried by policy")

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> Category | None:
        if v is None or v == "":
            return None
        if isinstance(v, Category):
            return v
        s = str(v).strip().lower()
        if s in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[s]
        try:
            return Category(s)
        except ValueError:
            raise ValueError(f"Invalid category '{v}'. Allowed categories are: {[c.value for c in Category]}.")

    @field_validator("ml_mode")
    @classmethod
    def validate_ml_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v_clean = v.strip().lower()
        if v_clean not in ML_MODES:
            raise ValueError(f"ml_mode must be one of {sorted(ML_MODES)} or None, got '{v}'.")
        return v_clean

    @field_validator("preferences", mode="before")
    @classmethod
    def validate_preferences(cls, v: dict[str, float]) -> dict[str, float]:
        if not isinstance(v, dict):
            raise ValueError("preferences must be a dictionary.")
        cleaned: dict[str, float] = {}
        for key, weight in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Preference key must be a non-empty string.")
            k_clean = key.strip().lower()
            canonical_key = PREFERENCE_ALIASES.get(k_clean, k_clean)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"Preference weight for '{key}' must be numeric.")
            w_float = float(weight)
            if not (0.0 <= w_float <= 1.0):
                raise ValueError(f"Preference weight for '{key}' must be between 0.0 and 1.0, got {w_float}.")
            cleaned[canonical_key] = round(w_float, 2)
        return cleaned


class RequirementsDelta(BaseModel):
    """Extracted requirements delta from a single user turn."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    category: Category | None = Field(default=None, description="Extracted product category")
    use_cases: list[str] | None = Field(default=None, description="Extracted use cases")
    budget_max_inr: int | None = Field(default=None, description="Extracted maximum budget in INR", ge=0)
    budget_min_inr: int | None = Field(default=None, description="Extracted minimum budget in INR", ge=0)
    hard: HardConstraints | None = Field(default=None, description="Extracted hard constraints")
    preferences: dict[str, float] | None = Field(
        default=None,
        description="Extracted preferences with 0.0-1.0 numeric weights",
    )
    ml_mode: str | None = Field(default=None, description="Extracted ML mode ('local' | 'cloud')")
    asked_slots: list[str] | None = Field(default=None, description="Asked slots if any")
    removed_fields: list[str] = Field(
        default_factory=list,
        description="Fields or preferences explicitly requested to be removed",
    )
    is_off_topic: bool = Field(default=False, description="True if message is unrelated to electronics shopping")

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> Category | None:
        if v is None or v == "":
            return None
        if isinstance(v, Category):
            return v
        s = str(v).strip().lower()
        if s in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[s]
        try:
            return Category(s)
        except ValueError:
            raise ValueError(f"Invalid category '{v}'. Allowed categories are: {[c.value for c in Category]}.")

    @field_validator("ml_mode")
    @classmethod
    def validate_ml_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v_clean = v.strip().lower()
        if v_clean not in ML_MODES:
            raise ValueError(f"ml_mode must be one of {sorted(ML_MODES)} or None, got '{v}'.")
        return v_clean

    @field_validator("preferences", mode="before")
    @classmethod
    def validate_preferences(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            raise ValueError("preferences must be a dictionary.")
        cleaned: dict[str, float] = {}
        for key, weight in v.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Preference key must be a non-empty string.")
            k_clean = key.strip().lower()
            canonical_key = PREFERENCE_ALIASES.get(k_clean, k_clean)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(f"Preference weight for '{key}' must be numeric.")
            w_float = float(weight)
            if not (0.0 <= w_float <= 1.0):
                raise ValueError(f"Preference weight for '{key}' must be between 0.0 and 1.0, got {w_float}.")
            cleaned[canonical_key] = round(w_float, 2)
        return cleaned

    @field_validator("removed_fields", mode="before")
    @classmethod
    def validate_removed_fields(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("removed_fields must be a list of strings.")
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Each entry in removed_fields must be a non-empty string.")
            norm = item.strip().lower()
            canonical = REMOVAL_FIELD_ALIASES.get(norm, norm)
            if canonical not in VALID_REMOVABLE_FIELDS and norm not in VALID_REMOVABLE_FIELDS:
                raise ValueError(
                    f"Invalid removed_field '{item}'. Must be a recognized field or preference name: "
                    f"{sorted(VALID_REMOVABLE_FIELDS)}"
                )
            cleaned.append(canonical)
        return cleaned


class Contradiction(BaseModel):
    """Represents a logical incompatibility between accumulated requirement constraints."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    code: str = Field(description="Programmatic contradiction identifier (e.g. 'BUDGET_MIN_EXCEEDS_MAX')")
    fields: list[str] = Field(description="Field names directly involved in the contradiction")
    message: str = Field(description="Concise, sanitized explanation of the contradiction")
    severity: Literal["error", "warning"] = Field(default="error", description="Severity level ('error' or 'warning')")
    requires_clarification: bool = Field(default=True, description="Whether conversational clarification is required")

    @field_validator("code", "message")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"'{info.field_name}' must be a non-empty string.")
        return v.strip()

    @field_validator("fields")
    @classmethod
    def validate_fields_list(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("'fields' must be a non-empty list of field names.")
        cleaned = [f.strip() for f in v if isinstance(f, str) and f.strip()]
        if not cleaned:
            raise ValueError("'fields' must contain at least one non-empty field name.")
        return cleaned


class PolicyAction(str, Enum):
    """Next conversational action determined by the clarification policy."""

    ASK = "ASK"
    CLARIFY_CONFLICT = "CLARIFY_CONFLICT"
    REDIRECT = "REDIRECT"
    NO_MATCH = "NO_MATCH"
    RECOMMEND = "RECOMMEND"


class PolicyDecision(BaseModel):
    """Deterministic decision emitted by the clarification and slot-filling policy."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action: PolicyAction = Field(description="Policy action to execute")
    question: str | None = Field(default=None, description="Concise question or explanation for the user")
    slot: str | None = Field(default=None, description="Targeted requirements slot, if action is ASK")
    reason: str | None = Field(default=None, description="Machine-readable code explaining the decision")
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="Relevant contradictions if action is CLARIFY_CONFLICT"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary context for telemetry and downstream integration"
    )
    can_recommend: bool = Field(
        default=False, description="Whether the recommendation gate is unlocked"
    )
    remaining_slots: list[str] = Field(
        default_factory=list, description="Remaining mandatory slots before recommendation is allowed"
    )

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: Any) -> PolicyAction:
        if isinstance(v, PolicyAction):
            return v
        if isinstance(v, str):
            v_clean = v.strip().upper()
            try:
                return PolicyAction(v_clean)
            except ValueError:
                raise ValueError(
                    f"Invalid policy action '{v}'. Allowed actions: {[a.value for a in PolicyAction]}."
                )
        raise ValueError(f"Invalid policy action type: {type(v)}.")

    @property
    def is_ask(self) -> bool:
        return self.action == PolicyAction.ASK

    @property
    def is_recommend(self) -> bool:
        return self.action == PolicyAction.RECOMMEND

    @property
    def is_clarify_conflict(self) -> bool:
        return self.action == PolicyAction.CLARIFY_CONFLICT

    @property
    def is_redirect(self) -> bool:
        return self.action == PolicyAction.REDIRECT

    @property
    def is_no_match(self) -> bool:
        return self.action == PolicyAction.NO_MATCH

