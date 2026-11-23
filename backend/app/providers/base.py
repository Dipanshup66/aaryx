"""Abstract base classes and schemas for LLM provider adapters.

Defines the contract for interacting with local and cloud AI models.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from app.core.logging import URL_SECRET_PATTERNS

# Supported conversational message roles
MessageRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """Single message in a conversational exchange."""

    role: MessageRole = Field(description="Role of the message author")
    content: str = Field(description="Text content of the message")


class ChatRequest(BaseModel):
    """Normalized chat completion request."""

    messages: list[ChatMessage] = Field(description="Sequence of conversation messages")
    model: str | None = Field(default=None, description="Override model for this request")
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens to generate")


class TokenUsage(BaseModel):
    """Token usage metrics for prompt and generation."""

    prompt_tokens: int = Field(default=0, description="Tokens evaluated from prompt")
    completion_tokens: int = Field(default=0, description="Tokens generated in response")
    total_tokens: int = Field(default=0, description="Total tokens processed")


class ChatResponse(BaseModel):
    """Complete non-streaming chat response."""

    content: str = Field(description="Assistant reply content")
    model: str = Field(description="Model that generated the response")
    finish_reason: str | None = Field(default="stop", description="Reason generation finished")
    usage: TokenUsage | None = Field(default=None, description="Usage metrics if reported")
    thinking: str | None = Field(default=None, description="Thinking trace if reported")
    provider: str | None = Field(
        default=None, description="Name of the provider that produced the completion"
    )


class ChatStreamChunk(BaseModel):
    """Single token or segment in a streaming chat response."""

    delta: str = Field(description="Incremental text chunk")
    model: str | None = Field(default=None, description="Model generating the chunk")
    finish_reason: str | None = Field(default=None, description="Finish reason on final chunk")
    usage: TokenUsage | None = Field(default=None, description="Usage metrics on final chunk")
    thinking: str | None = Field(default=None, description="Thinking chunk if reported")
    provider: str | None = Field(
        default=None, description="Name of the provider emitting the chunk"
    )


class ModelInfo(BaseModel):
    """Metadata describing an available model."""

    name: str = Field(description="Unique model name/tag")
    size: int | None = Field(default=None, description="Model file size in bytes")
    modified_at: str | None = Field(default=None, description="Last modification timestamp")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional model parameters/quantization"
    )


class ProviderHealth(BaseModel):
    """Health check status for an LLM provider."""

    healthy: bool = Field(description="True if provider is reachable and operational")
    provider: str = Field(description="Name of the provider (e.g., 'ollama')")
    details: dict[str, Any] | None = Field(default=None, description="Diagnostics or model summary")
    error: str | None = Field(default=None, description="Error message if unhealthy")


_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_*-]{6,}"), "[REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), "[REDACTED]"),
    (
        re.compile(
            r"""(?i)\b(api[_-]?key|x-api-key|x-goog-api-key|authorization|token|secret)"""
            r"""(["']?\s*[:=]\s*["']?)[^\s"',;}]{4,}"""
        ),
        r"\1\2[REDACTED]",
    ),
)
_MAX_SAFE_MESSAGE_CHARS = 500


def redact_secrets(text: str) -> str:
    """Mask credential-looking substrings (API keys, bearer tokens) in diagnostic text."""
    for pattern, replacement in (*URL_SECRET_PATTERNS, *_SECRET_PATTERNS):
        text = pattern.sub(replacement, text)
    return text


def safe_url(url: str) -> str:
    """Return ``scheme://host[:port]/path`` with credentials, query and fragment removed.

    Safe to log or return in health payloads: userinfo (``user:pass@``), query parameters
    (``?api_key=...``) and fragments are dropped, and the remaining path is redacted too.
    """
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "[invalid-url]"
    if not parts.scheme or not host:
        return redact_secrets(url)[:100]
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parts.scheme, netloc, redact_secrets(parts.path), "", ""))


# Provider-specific exceptions
class ProviderError(Exception):
    """Base exception for all LLM provider failures.

    Attributes:
        provider: Name of the provider that raised the error.
        status_code: HTTP status returned by the provider, when the failure came from an HTTP
            response. Routing decisions (retry / fallback) use this structured field and never
            parse the human-readable message.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    @property
    def safe_message(self) -> str:
        """Error text with credential-like substrings masked and length bounded.

        Use this (not ``str(exc)``) for telemetry and client-facing diagnostics, because
        upstream error bodies can echo fragments of API keys.
        """
        text = redact_secrets(str(self))
        if len(text) > _MAX_SAFE_MESSAGE_CHARS:
            text = text[:_MAX_SAFE_MESSAGE_CHARS] + "…"
        return text


class UnsupportedProviderError(ProviderError):
    """Raised when an unsupported or unregistered provider is requested."""


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration or required credentials are missing or invalid."""


class ProviderConnectionError(ProviderError):
    """Raised when connection to the model provider fails or is refused."""


class ProviderTimeoutError(ProviderError):
    """Raised when a request to the provider times out."""


class ProviderResponseError(ProviderError):
    """Raised when the provider returns an HTTP error code or invalid payload."""


def describe_error(exc: BaseException) -> str:
    """Describe an error for logs and health payloads without exposing sensitive content.

    Provider failures are summarised as class, provider and HTTP status only: their message
    text can carry upstream response bodies, URLs and credential fragments. Configuration
    errors are authored by this application (no upstream content), so their redacted message
    is kept because it tells the operator what to fix.
    """
    if isinstance(exc, (ProviderConfigurationError, UnsupportedProviderError)):
        return f"{exc.__class__.__name__}: {exc.safe_message}"
    if isinstance(exc, ProviderError):
        parts = [exc.__class__.__name__, f"provider={exc.provider}"]
        if exc.status_code is not None:
            parts.append(f"status={exc.status_code}")
        return " ".join(parts)
    return exc.__class__.__name__


class LLMProvider(ABC):
    """Abstract asynchronous interface for LLM provider adapters."""

    @abstractmethod
    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete chat completion."""

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks asynchronously."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List models available on this provider."""

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Check provider connectivity and operational readiness."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release any underlying network connections and resources."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
