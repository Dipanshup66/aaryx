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
        model: Model the failed request asked for, when the adapter knows it. Used to tell the
            user exactly which model to install or fix.
        operation: What was being done when it failed ("list_models"), when that changes
            how the error should be explained; None for an ordinary chat request.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        *,
        status_code: int | None = None,
        model: str | None = None,
        operation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.model = model
        self.operation = operation

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


# ---- Actionable error payloads (what the API returns to clients) -------------------------

_ERROR_TITLES = {
    "provider_unreachable": "Provider connection failed",
    "provider_timeout": "Provider timeout",
    "model_not_found": "Model not available",
    "endpoint_not_found": "Provider endpoint not found",
    "authentication_failed": "Provider authentication failed",
    "rate_limited": "Provider rate limit",
    "provider_response_error": "Provider response error",
    "provider_error": "Provider error",
}

# Settings prefix of each provider ("OLLAMA" -> OLLAMA_BASE_URL, OLLAMA_MODEL, ...).
_ENV_PREFIX = {"ollama": "OLLAMA", "openai": "OPENAI", "anthropic": "ANTHROPIC", "gemini": "GEMINI"}

# A model name is echoed inside a shell command hint, so only plain model-id characters pass.
_SAFE_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}")


def error_code(exc: ProviderError) -> str:
    """Stable machine-readable code for a provider failure (the UI switches on this)."""
    if isinstance(exc, ProviderConnectionError):
        return "provider_unreachable"
    if isinstance(exc, ProviderTimeoutError):
        return "provider_timeout"
    if isinstance(exc, ProviderResponseError):
        if exc.status_code == 404:
            # A 404 on a chat request means an unknown model. But a 404 while listing models,
            # or from an Ollama server that did not name a model, means the address is wrong
            # (some other server answered): advising "install the model" would be misleading.
            if exc.operation == "list_models" or (exc.provider == "ollama" and exc.model is None):
                return "endpoint_not_found"
            return "model_not_found"
        if exc.status_code in (401, 403):
            return "authentication_failed"
        if exc.status_code == 429:
            return "rate_limited"
        return "provider_response_error"
    return "provider_error"


def error_hint(code: str, provider: str, model: str | None = None) -> str | None:
    """One actionable sentence telling the user how to fix the problem, or None.

    ``model`` must already be validated (see ``_safe_model_name``).
    """
    provider = provider.strip().lower()
    env = _ENV_PREFIX.get(provider)
    local = provider == "ollama"

    if code == "provider_unreachable":
        if local:
            return (
                "Start Ollama (run `ollama serve` or open the Ollama app) and check "
                "OLLAMA_BASE_URL in backend/.env."
            )
        where = f"{env}_BASE_URL" if env else "the provider's base URL setting"
        return f"Check your internet connection and {where}."
    if code == "provider_timeout":
        if local:
            return (
                "The model took too long. Try again, use a smaller model, or raise OLLAMA_TIMEOUT."
            )
        return f"Try again. If it keeps happening, raise {env}_TIMEOUT." if env else "Try again."
    if code == "model_not_found":
        if local:
            name = model or "<model>"
            return (
                f"Run `ollama pull {name}` to install it (see installed models with "
                "`ollama list`), or set OLLAMA_MODEL to a model you have."
            )
        setting = f"{env}_MODEL" if env else "the model setting"
        return f"Check the model name: {setting}, or the model sent with the request."
    if code == "endpoint_not_found":
        if local:
            return (
                "The server at OLLAMA_BASE_URL answered 404: it is probably not Ollama, or the "
                "address is wrong. Check OLLAMA_BASE_URL."
            )
        where = f"{env}_BASE_URL" if env else "the provider's base URL setting"
        return f"The server answered 404. Check {where}."
    if code == "authentication_failed":
        if local:
            return (
                "Ollama refused the request. Check any proxy or authentication in front of "
                "OLLAMA_BASE_URL."
            )
        return f"Check {env}_API_KEY in backend/.env." if env else "Check the provider API key."
    if code == "rate_limited":
        return (
            "The provider is limiting requests or your quota is used up. Wait and retry, "
            "or check your plan and billing."
        )
    return None


def _safe_model_name(model: str | None) -> str | None:
    return model if model and _SAFE_MODEL_NAME.fullmatch(model) else None


def error_payload(exc: ProviderError) -> dict[str, str]:
    """JSON body for a failed provider call: title, code, detail, provider, model, hint.

    Contains no secrets: ``detail`` is the redacted ``safe_message``, ``model`` is a validated
    plain model id, and ``hint`` is fixed text built from setting names.
    """
    code = error_code(exc)
    model = _safe_model_name(exc.model)
    payload = {
        "error": _ERROR_TITLES[code],
        "code": code,
        "detail": exc.safe_message,
        "provider": exc.provider,
    }
    if model:
        payload["model"] = model
    hint = error_hint(code, exc.provider, model)
    if hint:
        payload["hint"] = hint
    return payload


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
