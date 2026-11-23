"""Regression tests for structured provider error classification (review finding #3).

Retry / fallback decisions must come from `ProviderError.status_code`, never from searching
the exception message, and error text exposed to clients/telemetry must not leak credentials.
"""

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base import (
    ChatMessage,
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
    redact_secrets,
)
from app.providers.factory import register_provider, reset_provider_registry
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter, is_transient_error, is_transient_status

REQUEST = ChatRequest(messages=[ChatMessage(role="user", content="hi")])


# ----------------------------------------------------------------------- pure classification


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529, 599])
def test_transient_statuses(status: int) -> None:
    assert is_transient_status(status)
    assert is_transient_error(ProviderResponseError("x", status_code=status)) is True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 405, 409, 413, 422, 301])
def test_permanent_statuses(status: int) -> None:
    assert not is_transient_status(status)
    assert is_transient_error(ProviderResponseError("x", status_code=status)) is False


def test_status_code_wins_over_misleading_message_text() -> None:
    # Digits inside a message must never influence the decision.
    assert is_transient_error(
        ProviderResponseError("overloaded, request used 4010 tokens", status_code=503)
    )
    assert is_transient_error(ProviderResponseError("error id 4012-400-401", status_code=500))
    assert not is_transient_error(ProviderResponseError("returned 503 once", status_code=404))
    assert not is_transient_error(ProviderResponseError("HTTP 500", status_code=422))


def test_message_is_never_parsed_when_status_is_absent() -> None:
    # No structured status -> legacy policy (transient), regardless of what the text says.
    assert is_transient_error(ProviderResponseError("401 Unauthorized")) is True
    assert is_transient_error(ProviderResponseError("404 not found")) is True


def test_non_http_error_types() -> None:
    assert is_transient_error(ProviderTimeoutError("t"))
    assert is_transient_error(ProviderConnectionError("c"))
    assert not is_transient_error(ProviderConfigurationError("cfg"))
    assert not is_transient_error(UnsupportedProviderError("u"))
    assert not is_transient_error(ValueError("v"))


def test_status_code_defaults_to_none_and_is_keyword_only() -> None:
    assert ProviderError("m", "p").status_code is None
    assert ProviderResponseError("m", provider="p", status_code=418).status_code == 418
    with pytest.raises(TypeError):
        ProviderResponseError("m", "p", 418)  # type: ignore[misc]


# --------------------------------------------------- adapters attach the real HTTP status


def _ollama(handler: Callable[[httpx.Request], httpx.Response]) -> LLMProvider:
    return OllamaProvider(
        base_url="http://mock-ollama:11434",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _openai(handler: Callable[[httpx.Request], httpx.Response]) -> LLMProvider:
    return OpenAIProvider(
        api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _anthropic(handler: Callable[[httpx.Request], httpx.Response]) -> LLMProvider:
    return AnthropicProvider(
        api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def _gemini(handler: Callable[[httpx.Request], httpx.Response]) -> LLMProvider:
    return GeminiProvider(
        api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


ADAPTERS = [_ollama, _openai, _anthropic, _gemini]


@pytest.mark.parametrize("make", ADAPTERS, ids=lambda f: f.__name__.lstrip("_"))
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 503])
async def test_adapter_generate_attaches_status_code(make: Any, status: int) -> None:
    provider = make(lambda request: httpx.Response(status, text="upstream says no"))
    with pytest.raises(ProviderResponseError) as info:
        await provider.generate(REQUEST)
    assert info.value.status_code == status
    assert is_transient_error(info.value) is (status in (429, 500, 503))


@pytest.mark.parametrize("make", ADAPTERS, ids=lambda f: f.__name__.lstrip("_"))
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 500, 503])
async def test_adapter_stream_attaches_status_code(make: Any, status: int) -> None:
    provider = make(lambda request: httpx.Response(status, text="upstream says no"))
    with pytest.raises(ProviderResponseError) as info:
        async for _ in provider.stream(REQUEST):
            pass
    assert info.value.status_code == status
    assert is_transient_error(info.value) is (status in (429, 500, 503))


# ------------------------------------------------------------------ router retry / fallback


class FailingProvider(LLMProvider):
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: ChatRequest) -> ChatResponse:
        self.generate_calls += 1
        if self.error:
            raise self.error
        return ChatResponse(content=f"from {self.name}", model=f"{self.name}-m")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        self.stream_calls += 1
        if self.error:
            raise self.error
        yield ChatStreamChunk(delta=f"from {self.name}", model=f"{self.name}-m")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider=self.name)

    async def aclose(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _registry():
    reset_provider_registry()
    yield
    reset_provider_registry()


def _router(
    status: int, *, allow_cloud: bool = True, retries: int = 2
) -> tuple[ProviderRouter, FailingProvider, FailingProvider]:
    primary = FailingProvider(
        "ollama", ProviderResponseError("Ollama returned HTTP error", "ollama", status_code=status)
    )
    fallback = FailingProvider("openai")
    register_provider("ollama", lambda s: primary)
    register_provider("openai", lambda s: fallback)
    router = ProviderRouter(
        primary_provider="ollama",
        fallback_provider="openai",
        fallback_enabled=True,
        max_retries=retries,
        allow_cloud_fallback=allow_cloud,
    )
    return router, primary, fallback


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_error_no_retry_no_fallback_generate(status: int) -> None:
    router, primary, fallback = _router(status)
    with pytest.raises(ProviderResponseError) as info:
        await router.generate(REQUEST)
    assert info.value.status_code == status
    assert primary.generate_calls == 1  # no retries
    assert fallback.generate_calls == 0  # nothing sent to the cloud


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_permanent_error_no_fallback_stream(status: int) -> None:
    router, primary, fallback = _router(status)
    with pytest.raises(ProviderResponseError):
        async for _ in router.stream(REQUEST):
            pass
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 0


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_transient_error_retries_then_falls_back_generate(status: int) -> None:
    router, primary, fallback = _router(status, retries=2)
    response = await router.generate(REQUEST)
    assert primary.generate_calls == 3  # 1 attempt + 2 retries (existing policy)
    assert fallback.generate_calls == 1
    assert response.content == "from openai"


@pytest.mark.parametrize("status", [429, 503])
async def test_transient_error_falls_back_stream(status: int) -> None:
    router, primary, fallback = _router(status)
    chunks = [c async for c in router.stream(REQUEST)]
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 1
    assert chunks[0].delta == "from openai"


async def test_local_first_policy_still_blocks_cloud_fallback_on_transient_error() -> None:
    router, _, fallback = _router(503, allow_cloud=False, retries=0)
    with pytest.raises(ProviderConfigurationError):
        await router.generate(REQUEST)
    assert fallback.generate_calls == 0


async def test_misleading_digits_in_message_do_not_block_fallback() -> None:
    primary = FailingProvider(
        "ollama",
        ProviderResponseError("overloaded; used 4010 tokens", "ollama", status_code=503),
    )
    fallback = FailingProvider("openai")
    register_provider("ollama", lambda s: primary)
    register_provider("openai", lambda s: fallback)
    router = ProviderRouter(
        primary_provider="ollama",
        fallback_provider="openai",
        fallback_enabled=True,
        max_retries=0,
        allow_cloud_fallback=True,
    )
    assert (await router.generate(REQUEST)).content == "from openai"


# ---------------------------------------------------------------------------- secret hygiene


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("Incorrect API key provided: sk-proj-abcdEFGH1234567890", "abcdEFGH1234567890"),
        ("bad header Authorization: Bearer abc.DEF-123_456xyz", "abc.DEF-123_456xyz"),
        ("key=AIzaSyA1234567890abcdefghijklmnop invalid", "AIzaSyA1234567890abcdefghijklmnop"),
        ('{"api_key": "supersecretvalue123"}', "supersecretvalue123"),
        ("x-api-key: ant-secret-9999", "ant-secret-9999"),
    ],
)
def test_redact_secrets(raw: str, must_not_contain: str) -> None:
    out = redact_secrets(raw)
    assert must_not_contain not in out
    assert "REDACTED" in out


def test_redaction_keeps_ordinary_diagnostic_text() -> None:
    text = "prompt is too long: 4010 tokens > max_tokens: 4096 for model qwen3.5:4b"
    assert redact_secrets(text) == text


def test_safe_message_redacts_and_bounds_length() -> None:
    err = ProviderResponseError(
        "bad key sk-live-ABCDEFGHIJKL " + "x" * 2000, provider="openai", status_code=401
    )
    safe = err.safe_message
    assert "ABCDEFGHIJKL" not in safe
    assert len(safe) <= 501
    assert "sk-live" not in safe


async def test_router_failure_telemetry_is_redacted_and_has_status() -> None:
    primary = FailingProvider(
        "openai",
        ProviderResponseError(
            "Incorrect API key provided: sk-proj-abcdEFGH1234567890", "openai", status_code=401
        ),
    )
    register_provider("openai", lambda s: primary)
    router = ProviderRouter(primary_provider="openai", fallback_enabled=False, max_retries=0)
    with pytest.raises(ProviderResponseError):
        await router.generate(REQUEST)
    health = await router.health_check()
    last = health.details["last_failure"]
    assert last["status_code"] == 401
    assert last["transient"] is False
    assert "abcdEFGH1234567890" not in last["message"]
    assert "abcdEFGH1234567890" not in (health.error or "")
