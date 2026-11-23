"""Unit tests for ProviderRouter (backend/app/providers/router.py).

Validates fallback routing, retry policies, streaming failure guardrails,
local-first privacy policies, and atomic runtime provider switching.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.core.config import Settings
from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChunk,
    LLMProvider,
    ModelInfo,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    TokenUsage,
    UnsupportedProviderError,
)
from app.providers.factory import register_provider, reset_provider_registry
from app.providers.router import ProviderRouter, is_transient_error


class MockRouterTestProvider(LLMProvider):
    """Controllable test double for router testing."""

    def __init__(
        self,
        name: str,
        fail_generate_times: int = 0,
        generate_error: Exception | None = None,
        fail_stream_after_chunks: int | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.generate_attempts = 0
        self.fail_generate_times = fail_generate_times
        self.generate_error = generate_error or ProviderConnectionError(
            f"{name} connection failed", provider=name
        )
        self.stream_attempts = 0
        self.fail_stream_after_chunks = fail_stream_after_chunks
        self.stream_error = stream_error or ProviderConnectionError(
            f"{name} stream connection dropped", provider=name
        )
        self.closed = False

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.generate_attempts += 1
        if self.generate_attempts <= self.fail_generate_times:
            raise self.generate_error
        return ChatResponse(
            content=f"Response from {self.name}",
            model=f"{self.name}-model",
            provider=self.name,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.stream_attempts += 1
        if self.fail_stream_after_chunks == 0:
            raise self.stream_error

        # Emit safe chunks before failure
        limit = self.fail_stream_after_chunks if self.fail_stream_after_chunks is not None else 2
        for i in range(limit):
            yield ChatStreamChunk(
                delta=f"{self.name}_chunk_{i} ", model=f"{self.name}-model", provider=self.name
            )

        if self.fail_stream_after_chunks is not None and self.fail_stream_after_chunks > 0:
            raise self.stream_error

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=f"{self.name}-model")]

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider=self.name)

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="user", content="Hello AARYX")],
        model="test-model",
    )


@pytest.fixture(autouse=True)
def clean_registry():
    reset_provider_registry()
    yield
    reset_provider_registry()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_provider_success(chat_request):
    """1. Verify primary provider succeeds directly when operational."""
    primary = MockRouterTestProvider("mock_primary")
    register_provider("mock_primary", lambda s: primary)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_enabled=False,
    )

    response = await router.generate(chat_request)
    assert response.content == "Response from mock_primary"
    assert response.provider == "mock_primary"
    assert primary.generate_attempts == 1


@pytest.mark.asyncio
async def test_primary_provider_timeout_with_fallback(chat_request):
    """2. Verify primary timeout triggers fallback when enabled."""
    timeout_err = ProviderTimeoutError("Primary timed out after 60s", provider="mock_primary")
    primary = MockRouterTestProvider(
        "mock_primary", fail_generate_times=5, generate_error=timeout_err
    )
    fallback = MockRouterTestProvider("mock_fallback")

    register_provider("mock_primary", lambda s: primary)
    register_provider("mock_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_provider="mock_fallback",
        fallback_enabled=True,
        max_retries=1,
    )

    response = await router.generate(chat_request)
    assert response.content == "Response from mock_fallback"
    assert response.provider == "mock_fallback"
    # Primary attempted once + 1 retry = 2 attempts before fallback
    assert primary.generate_attempts == 2
    assert fallback.generate_attempts == 1


@pytest.mark.asyncio
async def test_primary_provider_connection_failure_with_fallback(chat_request):
    """3. Verify primary connection failure triggers fallback."""
    conn_err = ProviderConnectionError("Connection refused", provider="mock_primary")
    primary = MockRouterTestProvider("mock_primary", fail_generate_times=5, generate_error=conn_err)
    fallback = MockRouterTestProvider("mock_fallback")

    register_provider("mock_primary", lambda s: primary)
    register_provider("mock_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_provider="mock_fallback",
        fallback_enabled=True,
        max_retries=0,
    )

    response = await router.generate(chat_request)
    assert response.content == "Response from mock_fallback"
    assert response.provider == "mock_fallback"
    assert primary.generate_attempts == 1
    assert fallback.generate_attempts == 1


@pytest.mark.asyncio
async def test_fallback_disabled_raises_error(chat_request):
    """4. Verify error is raised when fallback is disabled."""
    conn_err = ProviderConnectionError("Daemon down", provider="mock_primary")
    primary = MockRouterTestProvider("mock_primary", fail_generate_times=5, generate_error=conn_err)
    fallback = MockRouterTestProvider("mock_fallback")

    register_provider("mock_primary", lambda s: primary)
    register_provider("mock_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_provider="mock_fallback",
        fallback_enabled=False,
        max_retries=0,
    )

    with pytest.raises(ProviderConnectionError):
        await router.generate(chat_request)

    assert primary.generate_attempts == 1
    assert fallback.generate_attempts == 0


@pytest.mark.asyncio
async def test_fallback_enabled_success(chat_request):
    """5. Verify fallback succeeds and tracks diagnostics."""
    primary = MockRouterTestProvider("mock_primary", fail_generate_times=1)
    fallback = MockRouterTestProvider("mock_fallback")

    register_provider("mock_primary", lambda s: primary)
    register_provider("mock_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_provider="mock_fallback",
        fallback_enabled=True,
        max_retries=0,
    )

    response = await router.generate(chat_request)
    assert response.provider == "mock_fallback"

    health = await router.health_check()
    assert health.details["last_failure"]["provider"] == "mock_primary"
    assert health.details["failure_counts"]["mock_primary"] == 1


@pytest.mark.asyncio
async def test_cloud_fallback_opt_in_enforcement(chat_request):
    """6. Verify local-to-cloud fallback fails if explicit opt-in is not authorized."""
    # Settings without cloud opt-in authorization
    router = ProviderRouter(
        primary_provider="ollama",
        fallback_provider="openai",
        fallback_enabled=True,
        allow_cloud_fallback=False,
    )

    # Mock Ollama failing
    primary = MockRouterTestProvider("ollama", fail_generate_times=5)
    register_provider("ollama", lambda s: primary)
    register_provider("openai", lambda s: MockRouterTestProvider("openai"))

    with pytest.raises(ProviderConfigurationError) as exc_info:
        await router.generate(chat_request)

    assert "Local-to-cloud fallback to 'openai' is disabled" in str(exc_info.value)


def test_missing_fallback_credentials_rejected_in_config():
    """7. Verify configuration rejects cloud fallback if required API key is missing."""
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        Settings(
            ai_provider="ollama",
            ai_fallback_enabled=True,
            ai_fallback_provider="openai",
            ai_allow_cloud_fallback=True,
            openai_api_key=None,
            _env_file=None,
        )


def test_invalid_provider_configuration_rejected():
    """8. Verify identical primary and fallback configuration is rejected."""
    with pytest.raises(ValueError, match="cannot be identical"):
        Settings(
            ai_provider="ollama",
            ai_fallback_enabled=True,
            ai_fallback_provider="ollama",
            _env_file=None,
        )


@pytest.mark.asyncio
async def test_permanent_authentication_error_does_not_trigger_retry(chat_request):
    """9. Verify 401 Authentication error fails fast without retry or fallback."""
    auth_err = ProviderResponseError(
        "OpenAI authentication error (401): Invalid key",
        provider="mock_primary",
        status_code=401,
    )
    primary = MockRouterTestProvider("mock_primary", fail_generate_times=5, generate_error=auth_err)
    fallback = MockRouterTestProvider("mock_fallback")

    register_provider("mock_primary", lambda s: primary)
    register_provider("mock_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="mock_primary",
        fallback_provider="mock_fallback",
        fallback_enabled=True,
        max_retries=3,
    )

    with pytest.raises(ProviderResponseError) as exc_info:
        await router.generate(chat_request)

    assert "authentication error (401)" in str(exc_info.value).lower()
    # Permanent error must NOT be retried and must NOT trigger fallback
    assert primary.generate_attempts == 1
    assert fallback.generate_attempts == 0


@pytest.mark.asyncio
async def test_transient_errors_follow_retry_policy(chat_request):
    """10. Verify transient error is retried on primary and succeeds before exhausting retries."""
    timeout_err = ProviderTimeoutError("Gateway timeout", provider="mock_primary")
    # Fails 2 times, succeeds on 3rd attempt
    primary = MockRouterTestProvider(
        "mock_primary", fail_generate_times=2, generate_error=timeout_err
    )

    register_provider("mock_primary", lambda s: primary)

    router = ProviderRouter(
        primary_provider="mock_primary",
        max_retries=2,
        fallback_enabled=False,
    )

    response = await router.generate(chat_request)
    assert response.content == "Response from mock_primary"
    assert primary.generate_attempts == 3


@pytest.mark.asyncio
async def test_runtime_provider_switching(chat_request):
    """11. Verify dynamic switching of primary and fallback providers at runtime."""
    p_ollama = MockRouterTestProvider("ollama")
    p_openai = MockRouterTestProvider("openai")
    p_anthropic = MockRouterTestProvider("anthropic")

    register_provider("ollama", lambda s: p_ollama)
    register_provider("openai", lambda s: p_openai)
    register_provider("anthropic", lambda s: p_anthropic)

    router = ProviderRouter(primary_provider="ollama", fallback_enabled=False)

    resp1 = await router.generate(chat_request)
    assert resp1.provider == "ollama"

    # Switch primary to openai
    await router.switch_primary("openai")
    assert router.primary_provider == "openai"

    resp2 = await router.generate(chat_request)
    assert resp2.provider == "openai"

    # Configure fallback to anthropic
    await router.switch_fallback("anthropic", enabled=True)
    assert router.fallback_provider == "anthropic"
    assert router.fallback_enabled is True


@pytest.mark.asyncio
async def test_concurrent_requests_during_switching(chat_request):
    """12. Verify concurrent requests remain thread/async-safe during provider switching."""
    p1 = MockRouterTestProvider("p1")
    p2 = MockRouterTestProvider("p2")

    register_provider("p1", lambda s: p1)
    register_provider("p2", lambda s: p2)

    router = ProviderRouter(primary_provider="p1", fallback_enabled=False)

    async def run_query():
        resp = await router.generate(chat_request)
        return resp.provider

    async def switch():
        await asyncio.sleep(0.01)
        await router.switch_primary("p2")

    # Run 10 concurrent requests while triggering a switch mid-flight
    results = await asyncio.gather(*(run_query() for _ in range(10)), switch())
    providers_used = results[:10]
    # All queries must succeed with either p1 or p2
    assert all(p in ("p1", "p2") for p in providers_used)


@pytest.mark.asyncio
async def test_streaming_failure_before_first_chunk(chat_request):
    """13. Verify streaming failure before first chunk safely triggers fallback."""
    conn_err = ProviderConnectionError("Stream connection failed immediately", provider="p_primary")
    primary = MockRouterTestProvider("p_primary", fail_stream_after_chunks=0, stream_error=conn_err)
    fallback = MockRouterTestProvider("p_fallback")

    register_provider("p_primary", lambda s: primary)
    register_provider("p_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="p_primary",
        fallback_provider="p_fallback",
        fallback_enabled=True,
    )

    chunks = []
    async for chunk in router.stream(chat_request):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].provider == "p_fallback"
    assert "p_fallback_chunk_0" in chunks[0].delta


@pytest.mark.asyncio
async def test_streaming_failure_after_first_chunk_aborts_fallback(chat_request):
    """14. Verify streaming failure AFTER delivering chunks aborts fallback to prevent duplication."""
    drop_err = ProviderConnectionError("Connection dropped mid-stream", provider="p_primary")
    # Primary yields 1 chunk, then drops
    primary = MockRouterTestProvider("p_primary", fail_stream_after_chunks=1, stream_error=drop_err)
    fallback = MockRouterTestProvider("p_fallback")

    register_provider("p_primary", lambda s: primary)
    register_provider("p_fallback", lambda s: fallback)

    router = ProviderRouter(
        primary_provider="p_primary",
        fallback_provider="p_fallback",
        fallback_enabled=True,
    )

    chunks = []
    with pytest.raises(ProviderConnectionError):
        async for chunk in router.stream(chat_request):
            chunks.append(chunk)

    # Exactly 1 chunk was delivered before failure
    assert len(chunks) == 1
    assert chunks[0].provider == "p_primary"
    # Fallback must NOT have been called
    assert fallback.stream_attempts == 0


@pytest.mark.asyncio
async def test_provider_shutdown_cleanup():
    """15. Verify aclose() shuts down all initialized providers in pool."""
    p1 = MockRouterTestProvider("p1")
    p2 = MockRouterTestProvider("p2")

    register_provider("p1", lambda s: p1)
    register_provider("p2", lambda s: p2)

    router = ProviderRouter(primary_provider="p1", fallback_provider="p2", fallback_enabled=True)
    await router.get_provider("p1")
    await router.get_provider("p2")

    assert not p1.closed
    assert not p2.closed

    await router.aclose()

    assert p1.closed
    assert p2.closed


def test_is_transient_error_helper():
    """Verify is_transient_error accurately categorizes errors."""
    assert is_transient_error(ProviderTimeoutError("timeout")) is True
    assert is_transient_error(ProviderConnectionError("conn dropped")) is True
    assert (
        is_transient_error(ProviderResponseError("500 Internal Server Error", status_code=500))
        is True
    )
    assert (
        is_transient_error(ProviderResponseError("429 Rate limit exceeded", status_code=429))
        is True
    )

    # Permanent
    assert is_transient_error(ProviderConfigurationError("Missing key")) is False
    assert is_transient_error(UnsupportedProviderError("Unknown")) is False
    assert (
        is_transient_error(
            ProviderResponseError("401 Unauthorized: Invalid API Key", status_code=401)
        )
        is False
    )
    assert is_transient_error(ProviderResponseError("403 Forbidden", status_code=403)) is False
    assert (
        is_transient_error(ProviderResponseError("400 Bad Request: Invalid model", status_code=400))
        is False
    )
