"""Application configuration management using Pydantic Settings.

Loads environment variables from backend/.env or system environment,
validates values, and sets sensible defaults for local development.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine candidate .env locations:
# 1. backend/.env relative to this file
# 2. current working directory backend/.env
# 3. current working directory .env
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ENV_FILES = (
    str(_BACKEND_DIR / ".env"),
    str(Path.cwd() / "backend" / ".env"),
    str(Path.cwd() / ".env"),
)

_ORIGIN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[A-Za-z0-9.\[\]:_-]+$")
_HOST_RE = re.compile(r"^(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)$")


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated setting into trimmed, non-empty items."""
    return [item.strip() for item in value.split(",") if item.strip()]


# Supported AI provider identifiers
SUPPORTED_AI_PROVIDERS: set[str] = {"ollama", "openai", "anthropic", "gemini"}


class Settings(BaseSettings):
    """Central configuration class for AARYX.

    Manages runtime configuration, model endpoints, and credentials.
    Secrets are stored in SecretStr to prevent accidental leaks in logs.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Settings hold API keys and the API token: never echo a rejected value in an error.
        hide_input_in_errors=True,
    )

    # Core Application Settings
    app_name: str = Field(default="AARYX", description="Application name")
    app_env: Literal["development", "production", "test", "staging"] = Field(
        default="development",
        description="Application running environment",
    )
    ai_mode: Literal["local", "cloud"] = Field(
        default="local",
        description="Active AI routing mode (local or cloud)",
    )
    ai_provider: str = Field(
        default="ollama",
        description="Active AI provider identifier (e.g., 'ollama')",
    )
    ai_fallback_enabled: bool = Field(
        default=False,
        description="Whether fallback to a secondary provider is enabled on failures",
    )
    ai_fallback_provider: str | None = Field(
        default=None,
        description="Fallback AI provider identifier (e.g., 'openai', 'anthropic', 'gemini')",
    )
    ai_max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum retry attempts on primary provider for transient failures",
    )
    ai_allow_cloud_fallback: bool = Field(
        default=False,
        description="Explicit opt-in enabling fallback from local Ollama to cloud providers",
    )

    # Local Ollama Settings
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for local Ollama instance",
    )
    ollama_model: str = Field(
        default="qwen3.5:4b",
        description="Target Ollama model name",
    )
    ollama_timeout: float = Field(
        default=180.0,
        description="Timeout in seconds for Ollama LLM requests",
    )

    # Optional Cloud AI Settings (Generic / Fallback)
    cloud_base_url: str | None = Field(
        default=None,
        description="Base URL for cloud provider (OpenAI compatible)",
    )
    cloud_api_key: SecretStr | None = Field(
        default=None,
        description="API key for cloud provider (masked in logs and repr)",
    )
    cloud_model: str | None = Field(
        default=None,
        description="Model name for cloud provider",
    )

    # OpenAI Settings
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (masked in logs and repr)",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Target OpenAI model",
    )
    openai_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for OpenAI requests",
    )

    # Anthropic Settings
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (masked in logs and repr)",
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1",
        description="Anthropic API base URL",
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Target Anthropic model",
    )
    anthropic_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for Anthropic requests",
    )

    # Google Gemini Settings
    gemini_api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key (masked in logs and repr)",
    )
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Google Gemini API base URL",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Target Google Gemini model",
    )
    gemini_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for Gemini requests",
    )

    # Persistent Memory Settings
    memory_enabled: bool = Field(
        default=True,
        description="Whether persistent conversation memory is enabled",
    )
    memory_database_url: str = Field(
        default="sqlite+aiosqlite:///./data/aaryx.db",
        description="Async database connection URL for conversation storage",
    )
    memory_max_history_messages: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum conversation messages retrieved for AI context window",
    )
    memory_extraction_model: str = Field(
        default="qwen3.5:4b",
        description="Local Ollama model identifier used for memory extraction",
    )
    memory_max_context_items: int = Field(
        default=8,
        ge=0,
        le=50,
        description=(
            "Maximum number of active long-term memories added to the prompt each turn. "
            "Set to 0 to stop injecting memories into prompts."
        ),
    )
    memory_suggestions_enabled: bool = Field(
        default=True,
        description=(
            "Whether AARYX may suggest memories inferred from conversation. Suggestions are "
            "always stored as pending 'candidate' memories and only become active after user "
            "approval. When false, only explicit 'remember this' requests create memories."
        ),
    )
    memory_candidate_min_confidence: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for storing a memory suggestion; below this is discarded",
    )

    # Logging and Server Settings
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    api_host: str = Field(
        default="127.0.0.1",
        description=(
            "Address the server binds to when started with `python -m app`. Anything other "
            "than a loopback address requires API_TOKEN."
        ),
    )
    api_port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    api_token: SecretStr | None = Field(
        default=None,
        description=(
            "Optional shared secret. When set, every /api request must send it as "
            "'Authorization: Bearer <token>' (or 'X-API-Key'). Required to bind beyond loopback."
        ),
    )
    api_cors_origins: str = Field(
        default="tauri://localhost,http://tauri.localhost",
        description=(
            "Comma-separated browser origins allowed to call the API (the desktop app). "
            "Empty disables CORS. Wildcards are not accepted."
        ),
    )
    api_allowed_hosts: str = Field(
        default="localhost,127.0.0.1,[::1]",
        description=(
            "Comma-separated Host header values the API answers to. Requests with any other "
            "Host are refused, which blocks DNS-rebinding attacks on the local API."
        ),
    )

    @field_validator("api_token", mode="before")
    @classmethod
    def normalize_api_token(cls, v: Any) -> Any:
        """Treat an empty token as 'not set' and require a reasonably strong secret otherwise."""
        if v is None:
            return None
        raw = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        raw = raw.strip()
        if not raw:
            return None
        if len(raw) < 16:
            raise ValueError(
                "API_TOKEN must be at least 16 characters. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        return raw

    @field_validator("api_host", mode="before")
    @classmethod
    def normalize_api_host(cls, v: Any) -> Any:
        """Strip whitespace around the bind address."""
        return v.strip() if isinstance(v, str) else v

    @field_validator("api_cors_origins")
    @classmethod
    def validate_api_cors_origins(cls, v: str) -> str:
        """Allow only explicit scheme://host[:port] origins; never a wildcard."""
        for origin in _split_csv(v):
            if not _ORIGIN_RE.match(origin):
                raise ValueError(
                    f"API_CORS_ORIGINS entry {origin!r} must look like scheme://host[:port] "
                    "(no wildcard, no path)."
                )
        return v

    @field_validator("api_allowed_hosts")
    @classmethod
    def validate_api_allowed_hosts(cls, v: str) -> str:
        """Require explicit host names (optionally with an IPv6 bracket form)."""
        hosts = _split_csv(v)
        if not hosts:
            raise ValueError("API_ALLOWED_HOSTS must list at least one host.")
        for host in hosts:
            if not _HOST_RE.match(host):
                raise ValueError(
                    f"API_ALLOWED_HOSTS entry {host!r} must be a plain host name or IP address "
                    "(no wildcard, scheme, port or path)."
                )
        return v

    @property
    def cors_origins(self) -> list[str]:
        """Parsed list of allowed browser origins."""
        return _split_csv(self.api_cors_origins)

    @property
    def allowed_hosts(self) -> list[str]:
        """Parsed list of accepted Host header values (lowercase)."""
        return [h.lower() for h in _split_csv(self.api_allowed_hosts)]

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_app_env(cls, v: Any) -> str:
        """Normalize app_env to lowercase stripped string."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("ai_mode", mode="before")
    @classmethod
    def normalize_ai_mode(cls, v: Any) -> str:
        """Normalize ai_mode to lowercase stripped string."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("ai_provider", mode="before")
    @classmethod
    def normalize_ai_provider(cls, v: Any) -> str:
        """Normalize ai_provider to lowercase stripped string."""
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @field_validator("ai_fallback_provider", mode="before")
    @classmethod
    def normalize_ai_fallback_provider(cls, v: Any) -> str | None:
        """Normalize ai_fallback_provider to lowercase stripped string or None."""
        if isinstance(v, str):
            cleaned = v.strip().lower()
            return cleaned or None
        return v

    @field_validator("ai_provider")
    @classmethod
    def validate_ai_provider(cls, v: str) -> str:
        """Ensure ai_provider is supported."""
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("AI_PROVIDER cannot be empty")
        if cleaned not in SUPPORTED_AI_PROVIDERS:
            raise ValueError(
                f"Unsupported AI_PROVIDER '{v}'. Supported providers: {', '.join(sorted(SUPPORTED_AI_PROVIDERS))}"
            )
        return cleaned

    @model_validator(mode="after")
    def validate_routing_and_fallback(self) -> Self:
        """Validate routing configuration, local-first guardrails, and credential requirements."""
        if self.ai_fallback_enabled:
            if not self.ai_fallback_provider:
                raise ValueError(
                    "AI_FALLBACK_PROVIDER must be specified when AI_FALLBACK_ENABLED is true."
                )

            if self.ai_fallback_provider not in SUPPORTED_AI_PROVIDERS:
                raise ValueError(
                    f"Unsupported AI_FALLBACK_PROVIDER '{self.ai_fallback_provider}'. "
                    f"Supported providers: {', '.join(sorted(SUPPORTED_AI_PROVIDERS))}"
                )

            if self.ai_fallback_provider == self.ai_provider:
                raise ValueError(
                    f"Primary provider and fallback provider cannot be identical ('{self.ai_provider}')."
                )

            # Local-first guardrail: Ollama to cloud fallback requires explicit opt-in
            is_primary_local = self.ai_provider == "ollama"
            is_fallback_cloud = self.ai_fallback_provider in ("openai", "anthropic", "gemini")
            if is_primary_local and is_fallback_cloud and not self.ai_allow_cloud_fallback:
                raise ValueError(
                    f"Local-to-cloud fallback to '{self.ai_fallback_provider}' requires explicit opt-in. "
                    "Set AI_ALLOW_CLOUD_FALLBACK=true to authorize sending conversation data to cloud providers."
                )

            # Credential check for active fallback provider
            if self.ai_fallback_provider == "openai" and not self.openai_api_key:
                raise ValueError(
                    "OpenAI configured as fallback provider, but OPENAI_API_KEY is not set."
                )
            if self.ai_fallback_provider == "anthropic" and not self.anthropic_api_key:
                raise ValueError(
                    "Anthropic configured as fallback provider, but ANTHROPIC_API_KEY is not set."
                )
            if self.ai_fallback_provider == "gemini" and not self.gemini_api_key:
                raise ValueError(
                    "Gemini configured as fallback provider, but GEMINI_API_KEY is not set."
                )

        return self

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        """Validate URL scheme and strip trailing slashes."""
        cleaned = v.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(f"Invalid OLLAMA_BASE_URL: '{v}'. Must start with http:// or https://")
        return cleaned

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, v: str) -> str:
        """Ensure Ollama model name is not blank."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("OLLAMA_MODEL cannot be empty")
        return cleaned

    @field_validator(
        "cloud_base_url",
        "openai_base_url",
        "anthropic_base_url",
        "gemini_base_url",
        mode="before",
    )
    @classmethod
    def clean_base_url(cls, v: Any) -> str | None:
        """Strip whitespace and trailing slashes from base URLs."""
        if isinstance(v, str):
            cleaned = v.strip().rstrip("/")
            return cleaned or None
        return v

    @field_validator("cloud_model", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> str | None:
        """Convert empty strings to None for optional model field."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator(
        "cloud_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        mode="before",
    )
    @classmethod
    def empty_key_to_none(cls, v: Any) -> Any:
        """Convert empty API key strings to None."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("memory_database_url")
    @classmethod
    def validate_memory_database_url(cls, v: str) -> str:
        """Validate database URL scheme for local-first storage."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("MEMORY_DATABASE_URL cannot be empty")
        if not (cleaned.startswith(("sqlite+aiosqlite://", "sqlite://"))):
            raise ValueError(
                "MEMORY_DATABASE_URL must use SQLite (sqlite+aiosqlite:// or sqlite://)"
            )
        if cleaned.startswith("sqlite://") and not cleaned.startswith("sqlite+aiosqlite://"):
            cleaned = cleaned.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return cleaned

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level against standard Python log levels."""
        normalized = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(
                f"Invalid LOG_LEVEL '{v}'. Allowed levels: {', '.join(sorted(allowed))}"
            )
        return normalized


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings.

    Use this dependency throughout the FastAPI application.
    For tests, call get_settings.cache_clear() if environment variables change.
    """
    return Settings()
