"""Configurable Provider Router with fallback routing, retry policy, and dynamic switching.

Provides an LLMProvider implementation that wraps primary and secondary/fallback
providers, enforcing local-first security policies, transient failure retries,
and atomic runtime provider switching without modifying orchestration or endpoint contracts.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.providers.base import (
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    UnsupportedProviderError,
    describe_error,
)
from app.providers.factory import create_provider, get_supported_providers

logger = get_logger("providers.router")


def _safe_error_text(exc: Exception) -> str:
    """Return exception text that is safe to expose in health/diagnostic payloads and logs."""
    return describe_error(exc)


# HTTP statuses worth retrying / failing over on: request timeout, rate limit, any server error.
_TRANSIENT_STATUS_CODES = frozenset({408, 429})


def is_transient_status(status_code: int) -> bool:
    """Return True if an HTTP status indicates a temporary condition (408, 429, 5xx)."""
    return status_code in _TRANSIENT_STATUS_CODES or 500 <= status_code <= 599


def is_transient_error(exc: Exception) -> bool:
    """Determine if an error is transient and eligible for retry or fallback.

    Classification uses the structured exception type and ``ProviderError.status_code``;
    the human-readable message is never parsed.

    * Configuration / unsupported-provider errors are permanent.
    * Timeouts and connection failures are transient.
    * HTTP responses: 408, 429 and 5xx are transient; every other status (400, 401, 403, 404,
      422, ...) is a permanent client-side error that must not be retried or sent to a fallback.
    * A ``ProviderResponseError`` without an HTTP status (malformed payload, truncated stream)
      keeps the pre-existing policy and is treated as transient.
    """
    if isinstance(exc, (ProviderConfigurationError, UnsupportedProviderError)):
        return False

    if isinstance(exc, (ProviderTimeoutError, ProviderConnectionError)):
        return True

    if isinstance(exc, ProviderResponseError):
        if exc.status_code is not None:
            return is_transient_status(exc.status_code)
        return True

    return False


# Model-name prefixes that identify each hosted provider's own model catalogue.
_CLOUD_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "chatgpt-", "o1", "o3", "o4"),
    "anthropic": ("claude-",),
    "gemini": ("gemini-", "models/gemini-"),
}


def is_model_compatible(provider_name: str, model: str) -> bool:
    """Return True if ``model`` is plausibly a valid model identifier for ``provider_name``.

    Hosted providers accept only their own catalogue (recognised by prefix); identifiers that
    contain ``:`` are Ollama-style ``name:tag`` values and are never valid cloud model ids
    (except OpenAI fine-tunes, which start with ``ft:``). Local providers accept anything that
    does not belong to a hosted catalogue.
    """
    name = model.strip().lower()
    if not name:
        return False

    provider = provider_name.strip().lower()
    prefixes = _CLOUD_MODEL_PREFIXES.get(provider)
    if prefixes is not None:
        if name.startswith("ft:"):
            return provider == "openai"
        return ":" not in name and name.startswith(prefixes)

    return not any(is_model_compatible(cloud, name) for cloud in _CLOUD_MODEL_PREFIXES)


class ProviderRouter(LLMProvider):
    """Dynamic LLM provider router managing primary and fallback providers."""

    def __init__(
        self,
        settings: Settings | None = None,
        primary_provider: str | None = None,
        fallback_provider: str | None = None,
        fallback_enabled: bool | None = None,
        max_retries: int | None = None,
        allow_cloud_fallback: bool | None = None,
    ) -> None:
        """Initialize ProviderRouter from configuration settings.

        Args:
            settings: Optional Settings instance. Defaults to get_settings().
            primary_provider: Optional primary provider name override.
            fallback_provider: Optional fallback provider name override.
            fallback_enabled: Optional boolean flag enabling fallback.
            max_retries: Optional maximum retry attempts for transient errors.
            allow_cloud_fallback: Optional boolean authorizing local-to-cloud fallback.
        """
        self.settings = settings or get_settings()

        self._primary_name = (primary_provider or self.settings.ai_provider).strip().lower()

        raw_fallback = (
            fallback_provider
            if fallback_provider is not None
            else self.settings.ai_fallback_provider
        )
        self._fallback_name = raw_fallback.strip().lower() if raw_fallback else None

        self._fallback_enabled = (
            fallback_enabled if fallback_enabled is not None else self.settings.ai_fallback_enabled
        )

        self._max_retries = max_retries if max_retries is not None else self.settings.ai_max_retries

        self._allow_cloud_fallback = (
            allow_cloud_fallback
            if allow_cloud_fallback is not None
            else self.settings.ai_allow_cloud_fallback
        )

        # Mutex protecting runtime provider switching and configuration mutations
        self._lock = asyncio.Lock()

        # Cache of initialized provider adapters to avoid recreation of HTTP clients
        self._provider_pool: dict[str, LLMProvider] = {}

        # Diagnostics and health tracking
        self._failure_counts: dict[str, int] = defaultdict(int)
        self._last_failure: dict[str, Any] | None = None

        logger.info(
            "Initialized ProviderRouter (primary=%s, fallback=%s, fallback_enabled=%s, retries=%d)",
            self._primary_name,
            self._fallback_name,
            self._fallback_enabled,
            self._max_retries,
        )

    @property
    def primary_provider(self) -> str:
        """Name of the active primary provider."""
        return self._primary_name

    @property
    def fallback_provider(self) -> str | None:
        """Name of the configured fallback provider."""
        return self._fallback_name

    @property
    def fallback_enabled(self) -> bool:
        """Whether fallback routing is active."""
        return self._fallback_enabled

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts on primary provider."""
        return self._max_retries

    @property
    def _client(self) -> Any:
        """Convenience property exposing primary provider client for lifecycle assertions."""
        if self._primary_name in self._provider_pool:
            return getattr(self._provider_pool[self._primary_name], "_client", None)
        return None

    def _get_or_create_unlocked(self, normalized: str) -> LLMProvider:
        """Retrieve or instantiate a provider adapter assuming self._lock is already held."""
        if normalized not in self._provider_pool:
            logger.debug("Instantiating provider '%s' into router pool", normalized)
            self._provider_pool[normalized] = create_provider(
                settings=self.settings,
                provider_name=normalized,
            )
        return self._provider_pool[normalized]

    async def get_provider(self, name: str) -> LLMProvider:
        """Retrieve or instantiate a provider adapter from the internal pool."""
        normalized = name.strip().lower()
        if normalized in self._provider_pool:
            return self._provider_pool[normalized]
        async with self._lock:
            return self._get_or_create_unlocked(normalized)

    def _get_routing_snapshot(self) -> tuple[str, bool, str | None, int]:
        """Return an immutable snapshot of active routing parameters for a request."""
        return (
            self._primary_name,
            self._fallback_enabled,
            self._fallback_name,
            self._max_retries,
        )

    def _record_failure(self, provider_name: str, exc: Exception) -> None:
        """Record sanitized failure telemetry."""
        self._failure_counts[provider_name] += 1
        self._last_failure = {
            "provider": provider_name,
            "error_type": exc.__class__.__name__,
            "status_code": getattr(exc, "status_code", None),
            "message": _safe_error_text(exc),
            "transient": is_transient_error(exc),
        }

    def _fallback_request(self, request: ChatRequest, fallback_name: str) -> ChatRequest:
        """Return the request to send to the fallback provider.

        A per-request ``model`` override was chosen for the primary provider. If it is not a
        valid identifier for the fallback provider it is dropped so the fallback uses its own
        configured model instead of failing with a 400/404. Compatible overrides are preserved.
        The caller's request object is never mutated.
        """
        if request.model is None or is_model_compatible(fallback_name, request.model):
            return request
        logger.warning(
            "Model override '%s' is not valid for fallback provider '%s'; "
            "using the fallback provider's configured model",
            request.model,
            fallback_name,
        )
        return request.model_copy(update={"model": None})

    async def generate(self, request: ChatRequest) -> ChatResponse:
        """Generate a complete chat completion via primary provider with optional fallback.

        Executes up to `max_retries` on transient primary failures. If primary continues
        failing and fallback is enabled, routes request to the fallback provider.
        """
        primary_name, fallback_enabled, fallback_name, max_retries = self._get_routing_snapshot()

        last_primary_error: Exception | None = None
        primary_provider = await self.get_provider(primary_name)

        # Attempt primary provider with transient retries
        for attempt in range(max_retries + 1):
            try:
                response = await primary_provider.generate(request)
                if response.provider is None:
                    response.provider = primary_name
                return response
            except (ProviderError, httpx.HTTPError, OSError) as exc:
                last_primary_error = exc
                if not is_transient_error(exc) or attempt == max_retries:
                    break
                logger.warning(
                    "Primary provider '%s' attempt %d/%d failed with transient error: %s. Retrying...",
                    primary_name,
                    attempt + 1,
                    max_retries + 1,
                    describe_error(exc),
                )

        # Record primary provider failure
        assert last_primary_error is not None
        self._record_failure(primary_name, last_primary_error)

        # Check if fallback is eligible
        if not fallback_enabled or not fallback_name:
            logger.error(
                "Primary provider '%s' failed and fallback is disabled: %s",
                primary_name,
                describe_error(last_primary_error),
            )
            raise last_primary_error

        if not is_transient_error(last_primary_error):
            logger.error(
                "Primary provider '%s' failed with permanent error: %s. Fallback aborted.",
                primary_name,
                describe_error(last_primary_error),
            )
            raise last_primary_error

        # Local-first security guardrail check
        if (
            primary_name == "ollama"
            and fallback_name in ("openai", "anthropic", "gemini")
            and not self._allow_cloud_fallback
        ):
            logger.error(
                "Fallback to cloud provider '%s' blocked by local-first security policy",
                fallback_name,
            )
            raise ProviderConfigurationError(
                f"Local-to-cloud fallback to '{fallback_name}' is disabled. "
                "Set AI_ALLOW_CLOUD_FALLBACK=true to authorize cloud data transmission.",
                provider="router",
            )

        logger.warning(
            "Primary provider '%s' failed (%s). Initiating fallback to '%s'",
            primary_name,
            describe_error(last_primary_error),
            fallback_name,
        )

        try:
            fallback = await self.get_provider(fallback_name)
            response = await fallback.generate(self._fallback_request(request, fallback_name))
            if response.provider is None:
                response.provider = fallback_name
            return response
        except (ProviderError, httpx.HTTPError, OSError) as fallback_exc:
            self._record_failure(fallback_name, fallback_exc)
            logger.error(
                "Fallback provider '%s' also failed: %s",
                fallback_name,
                describe_error(fallback_exc),
            )
            raise fallback_exc from last_primary_error

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Stream chat completion chunks with guarded fallback semantics.

        CRITICAL SECURITY & INTEGRITY RULE:
        Fallback is strictly permitted ONLY BEFORE user-visible chunks have been emitted.
        Once any chunk is delivered, fallback is aborted to prevent duplicated/corrupted output.
        """
        primary_name, fallback_enabled, fallback_name, _ = self._get_routing_snapshot()

        chunks_emitted = 0
        primary_failed_before_first_chunk = False
        primary_error: Exception | None = None

        try:
            primary_provider = await self.get_provider(primary_name)
            async for chunk in primary_provider.stream(request):
                chunks_emitted += 1
                if chunk.provider is None:
                    chunk.provider = primary_name
                yield chunk
        except (ProviderError, httpx.HTTPError, OSError) as exc:
            if chunks_emitted > 0:
                logger.error(
                    "Primary stream on '%s' failed after emitting %d chunks: %s. "
                    "Fallback strictly prohibited to prevent duplicate response text.",
                    primary_name,
                    chunks_emitted,
                    describe_error(exc),
                )
                raise
            primary_failed_before_first_chunk = True
            primary_error = exc

        if primary_failed_before_first_chunk and primary_error is not None:
            self._record_failure(primary_name, primary_error)

            if not fallback_enabled or not fallback_name:
                logger.error(
                    "Primary stream on '%s' failed and fallback is disabled: %s",
                    primary_name,
                    describe_error(primary_error),
                )
                raise primary_error

            if not is_transient_error(primary_error):
                logger.error(
                    "Primary stream on '%s' failed with permanent error: %s. Fallback skipped.",
                    primary_name,
                    describe_error(primary_error),
                )
                raise primary_error

            if (
                primary_name == "ollama"
                and fallback_name in ("openai", "anthropic", "gemini")
                and not self._allow_cloud_fallback
            ):
                logger.error("Cloud fallback stream blocked by local-first policy")
                raise ProviderConfigurationError(
                    f"Local-to-cloud stream fallback to '{fallback_name}' is disabled. "
                    "Set AI_ALLOW_CLOUD_FALLBACK=true to authorize cloud data transmission.",
                    provider="router",
                )

            logger.warning(
                "Primary provider '%s' stream failed before first chunk (%s). Falling back to '%s'",
                primary_name,
                describe_error(primary_error),
                fallback_name,
            )

            try:
                fallback = await self.get_provider(fallback_name)
                async for chunk in fallback.stream(self._fallback_request(request, fallback_name)):
                    if chunk.provider is None:
                        chunk.provider = fallback_name
                    yield chunk
            except (ProviderError, httpx.HTTPError, OSError) as fallback_exc:
                self._record_failure(fallback_name, fallback_exc)
                logger.error(
                    "Fallback provider '%s' stream failed: %s",
                    fallback_name,
                    describe_error(fallback_exc),
                )
                raise fallback_exc from primary_error

    async def switch_primary(self, provider_name: str) -> None:
        """Safely switch the active primary provider at runtime.

        Pre-validates provider existence and credentials before committing the switch.
        Concurrent requests remain safe through atomic state mutation.
        """
        normalized = provider_name.strip().lower()
        if normalized not in get_supported_providers():
            raise UnsupportedProviderError(
                f"Cannot switch to unsupported provider: '{provider_name}'",
                provider=provider_name,
            )

        async with self._lock:
            if self._fallback_enabled and self._fallback_name == normalized:
                raise ValueError(
                    f"Primary provider cannot be identical to active fallback provider ('{normalized}')"
                )

            # Pre-validate and warm up provider adapter
            self._get_or_create_unlocked(normalized)

            old_primary = self._primary_name
            self._primary_name = normalized
            logger.info("Switched primary AI provider from '%s' to '%s'", old_primary, normalized)

    async def switch_fallback(
        self,
        provider_name: str | None,
        enabled: bool | None = None,
    ) -> None:
        """Safely configure the fallback provider and its active state at runtime."""
        normalized = provider_name.strip().lower() if provider_name else None
        if normalized and normalized not in get_supported_providers():
            raise UnsupportedProviderError(
                f"Cannot configure unsupported fallback provider: '{provider_name}'",
                provider=provider_name,
            )

        async with self._lock:
            target_enabled = enabled if enabled is not None else self._fallback_enabled

            if target_enabled and normalized:
                if normalized == self._primary_name:
                    raise ValueError(
                        f"Fallback provider cannot be identical to active primary provider ('{normalized}')"
                    )

                if (
                    self._primary_name == "ollama"
                    and normalized in ("openai", "anthropic", "gemini")
                    and not self._allow_cloud_fallback
                ):
                    raise ValueError(
                        f"Local-to-cloud fallback to '{normalized}' requires explicit authorization. "
                        "Set AI_ALLOW_CLOUD_FALLBACK=true before enabling cloud fallback."
                    )

                # Pre-validate credentials and warm up
                self._get_or_create_unlocked(normalized)

            self._fallback_name = normalized
            if enabled is not None:
                self._fallback_enabled = enabled
            logger.info(
                "Updated fallback routing configuration (fallback=%s, enabled=%s)",
                self._fallback_name,
                self._fallback_enabled,
            )

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from active primary provider."""
        primary = await self.get_provider(self._primary_name)
        return await primary.list_models()

    async def health_check(self) -> ProviderHealth:
        """Perform unified health check across primary and fallback providers."""
        primary_health: ProviderHealth | None = None
        try:
            primary = await self.get_provider(self._primary_name)
            primary_health = await primary.health_check()
        except (ProviderError, httpx.HTTPError, OSError) as exc:
            primary_health = ProviderHealth(
                healthy=False,
                provider=self._primary_name,
                error=_safe_error_text(exc),
            )

        fallback_health: ProviderHealth | None = None
        if self._fallback_enabled and self._fallback_name:
            try:
                fallback = await self.get_provider(self._fallback_name)
                fallback_health = await fallback.health_check()
            except (ProviderError, httpx.HTTPError, OSError) as exc:
                fallback_health = ProviderHealth(
                    healthy=False,
                    provider=self._fallback_name,
                    error=_safe_error_text(exc),
                )

        overall_healthy = primary_health.healthy or (
            self._fallback_enabled and fallback_health is not None and fallback_health.healthy
        )

        details = {
            "routing": {
                "primary_provider": self._primary_name,
                "fallback_provider": self._fallback_name,
                "fallback_enabled": self._fallback_enabled,
                "max_retries": self._max_retries,
                "allow_cloud_fallback": self._allow_cloud_fallback,
            },
            "primary": primary_health.model_dump(),
            "fallback": fallback_health.model_dump() if fallback_health else None,
            "failure_counts": dict(self._failure_counts),
            "last_failure": self._last_failure,
        }

        return ProviderHealth(
            healthy=overall_healthy,
            provider="router",
            details=details,
            error=primary_health.error if not overall_healthy else None,
        )

    async def aclose(self) -> None:
        """Release all initialized provider clients in the pool cleanly."""
        async with self._lock:
            for name, provider in self._provider_pool.items():
                logger.debug("Closing router pooled provider '%s'", name)
                await provider.aclose()
            logger.info("ProviderRouter shutdown and resource cleanup complete")
