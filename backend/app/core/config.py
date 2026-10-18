"""Application configuration module using Pydantic Settings.

Loads configuration from the root .env file and environment variables,
with environment variables taking precedence over .env file values.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve root directory reliably (backend/app/core -> app -> backend -> root)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BACKEND_DIR.parent
ENV_FILE_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    """Central configuration class for Electronics Recommendation Agent."""

    # Local MongoDB Configuration
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        description="Local MongoDB connection URI",
    )
    mongo_db: str = Field(
        default="electronics_agent",
        description="Name of the MongoDB database",
    )

    # Groq LLM Cloud API Configuration
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for the Groq Cloud API",
    )
    llm_api_key: str = Field(
        default="replace_me",
        description="Groq API key for authentication",
    )
    llm_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Target LLM model name for recommendation reasoning",
    )
    llm_fallback_base_url: str | None = Field(
        default=None,
        description="Optional fallback LLM endpoint base URL",
    )
    llm_fallback_model: str | None = Field(
        default=None,
        description="Optional fallback LLM model identifier",
    )
    llm_timeout_s: float = Field(
        default=30.0,
        description="Timeout for LLM API requests in seconds",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Maximum retry attempts for LLM API requests",
    )

    # CORS Configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:5173"],
        description="List of allowed CORS origins",
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("mongo_uri")
    @classmethod
    def validate_mongo_uri(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("MONGO_URI must be a non-empty string.")
        stripped = v.strip()
        if not (stripped.startswith("mongodb://") or stripped.startswith("mongodb+srv://")):
            raise ValueError("MONGO_URI must start with 'mongodb://' or 'mongodb+srv://'.")
        return stripped

    @field_validator("mongo_db")
    @classmethod
    def validate_mongo_db(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("MONGO_DB must be a non-empty string.")
        stripped = v.strip()
        # Disallow characters forbidden in MongoDB database names
        invalid_chars = {' ', '.', '$', '/', '\\', '\x00', '"'}
        if any(char in stripped for char in invalid_chars):
            raise ValueError(f"MONGO_DB contains invalid characters: {stripped}")
        return stripped

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, v: str) -> str:
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("LLM_BASE_URL must be a non-empty string.")
        parsed = urlparse(v.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"LLM_BASE_URL must be a valid HTTP/HTTPS URL: {v}")
        return v.strip().rstrip("/")

    @field_validator("llm_fallback_base_url", mode="before")
    @classmethod
    def validate_llm_fallback_base_url(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return None
            parsed = urlparse(stripped)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"LLM_FALLBACK_BASE_URL must be a valid HTTP/HTTPS URL: {v}")
            return stripped.rstrip("/")
        raise ValueError("LLM_FALLBACK_BASE_URL must be a string or None.")

    @field_validator("llm_fallback_model", mode="before")
    @classmethod
    def validate_llm_fallback_model(cls, v: Any) -> str | None:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return str(v).strip()

    @field_validator("llm_timeout_s")
    @classmethod
    def validate_llm_timeout_s(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LLM_TIMEOUT_S must be greater than 0.")
        return float(v)

    @field_validator("llm_max_retries")
    @classmethod
    def validate_llm_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("LLM_MAX_RETRIES must be greater than or equal to 0.")
        return int(v)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_and_validate_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return ["*"]
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except json.JSONDecodeError as err:
                    raise ValueError(f"Invalid JSON format for CORS_ORIGINS: {v}") from err
            # Fallback for comma-separated values
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        raise ValueError("CORS_ORIGINS must be a JSON array string or list of strings.")

    @property
    def masked_llm_api_key(self) -> str:
        """Return a securely masked representation of the LLM API key."""
        if not self.llm_api_key or self.llm_api_key == "replace_me":
            return "[NOT_SET_OR_DEFAULT]"
        if len(self.llm_api_key) <= 8:
            return "********"
        return f"{self.llm_api_key[:4]}...{self.llm_api_key[-4:]}"

    @property
    def masked_mongo_uri(self) -> str:
        """Return a sanitized MongoDB URI masking any embedded credentials."""
        try:
            parsed = urlparse(self.mongo_uri)
            if parsed.password:
                sanitized_netloc = f"{parsed.username}:***@{parsed.hostname}"
                if parsed.port:
                    sanitized_netloc += f":{parsed.port}"
                return parsed._replace(netloc=sanitized_netloc).geturl()
            return self.mongo_uri
        except Exception:
            return "mongodb://[sanitized]"

    def __repr__(self) -> str:
        """Provide a safe string representation omitting raw secret values."""
        return (
            f"Settings("
            f"mongo_uri='{self.masked_mongo_uri}', "
            f"mongo_db='{self.mongo_db}', "
            f"llm_base_url='{self.llm_base_url}', "
            f"llm_api_key='{self.masked_llm_api_key}', "
            f"llm_model='{self.llm_model}', "
            f"llm_fallback_base_url={self.llm_fallback_base_url!r}, "
            f"llm_fallback_model={self.llm_fallback_model!r}, "
            f"llm_timeout_s={self.llm_timeout_s}, "
            f"llm_max_retries={self.llm_max_retries}, "
            f"cors_origins={self.cors_origins})"
        )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
