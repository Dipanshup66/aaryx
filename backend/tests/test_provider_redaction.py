"""Health-check redaction and log sanitization (final audit findings 1 and 2).

Sentinels stand in for things that must never leak: credentials embedded in the base URL,
secret query parameters, API keys echoed by an upstream, and raw upstream response bodies.
What MUST survive is diagnostic: provider name, HTTP status, exception class, host.
"""

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_memory_service, get_orchestrator, get_provider
from app.brain.orchestrator import BrainOrchestrator
from app.main import app
from app.memory.extractor import InferredMemoryExtractor
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
    describe_error,
    redact_secrets,
    safe_url,
)
from app.providers.factory import register_provider, reset_provider_registry
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter

USER = "svc-user"
PASSWORD = "P4ssw0rd-SECRET-77"
QUERY_KEY = "QUERYKEY-SECRET-999"
BODY_MARKER = "UPSTREAM-BODY-MARKER-4242"
BODY_KEY = "sk-live-BODYSECRET1234567890"
HEADER_KEY = "HEADER-API-KEY-SECRET-555"

SECRET_BASE_URL = f"https://{USER}:{PASSWORD}@api.example.test:8443/v1?api_key={QUERY_KEY}&x=1"
SENTINELS = (USER + ":", PASSWORD, QUERY_KEY, BODY_MARKER, BODY_KEY, HEADER_KEY)

REQUEST = ChatRequest(messages=[ChatMessage(role="user", content="hello")])
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


ADAPTERS: dict[str, Callable[[Handler], LLMProvider]] = {
    "openai": lambda h: OpenAIProvider(
        api_key=HEADER_KEY, base_url=SECRET_BASE_URL, client=_client(h)
    ),
    "anthropic": lambda h: AnthropicProvider(
        api_key=HEADER_KEY, base_url=SECRET_BASE_URL, client=_client(h)
    ),
    "gemini": lambda h: GeminiProvider(
        api_key=HEADER_KEY, base_url=SECRET_BASE_URL, client=_client(h)
    ),
    "ollama": lambda h: OllamaProvider(base_url=SECRET_BASE_URL, client=_client(h)),
}

BODY = f'{{"error":{{"message":"{BODY_MARKER} Incorrect API key provided: {BODY_KEY}"}}}}'
URL_IN_MESSAGE = f"cannot reach {SECRET_BASE_URL} ({BODY_KEY})"


def _http(status: int) -> Handler:
    return lambda request: httpx.Response(status, text=BODY)


def _raises(exc: Exception) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


MODELS_OK = {
    "openai": {"data": [{"id": "gpt-4o"}]},
    "anthropic": {"data": [{"id": "claude-x"}]},
    "gemini": {"models": [{"name": "models/gemini-x"}]},
    "ollama": {"models": [{"name": "qwen3.5:4b"}]},
}


def _assert_clean(text: str, where: str) -> None:
    for sentinel in SENTINELS:
        assert sentinel not in text, f"{sentinel!r} leaked into {where}: {text[:300]}"


# =========================================================================== helper units


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (SECRET_BASE_URL, "https://api.example.test:8443/v1"),
        ("https://token123@api.example.test/v1", "https://api.example.test/v1"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("https://api.example.test/v1#frag?x=1", "https://api.example.test/v1"),
        ("http://[::1]:9000/path?key=abc", "http://[::1]:9000/path"),
        ("https://api.example.test/v1/", "https://api.example.test/v1/"),
    ],
)
def test_safe_url_strips_credentials_query_and_fragment(url: str, expected: str) -> None:
    assert safe_url(url) == expected


def test_safe_url_handles_malformed_input_without_raising() -> None:
    assert safe_url("http://host:notaport/x") == "[invalid-url]"
    assert "SECRET" not in safe_url("not a url token=SECRET")
    assert safe_url("") == ""


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        (f"GET {SECRET_BASE_URL} failed", PASSWORD),
        (f"GET {SECRET_BASE_URL} failed", QUERY_KEY),
        ("https://host/p?token=abcdef123456&b=1", "abcdef123456"),
        ("https://host/p?x=1&access_token=abcdef123456", "abcdef123456"),
        ("https://host/p?signature=abcdef123456", "abcdef123456"),
        ("https://onlytoken@host/p", "onlytoken"),
    ],
)
def test_redact_secrets_masks_url_credentials_and_query_secrets(
    raw: str, must_not_contain: str
) -> None:
    assert must_not_contain not in redact_secrets(raw)


def test_redact_secrets_keeps_ordinary_urls_and_text() -> None:
    text = "HTTP 503 from https://api.example.test/v1/models?page=2&limit=10"
    assert redact_secrets(text) == text


def test_describe_error_never_includes_message_text() -> None:
    exc = ProviderResponseError(f"{BODY_MARKER} {BODY_KEY}", provider="openai", status_code=401)
    out = describe_error(exc)
    assert out == "ProviderResponseError provider=openai status=401"
    assert describe_error(ProviderConnectionError(URL_IN_MESSAGE, provider="x")) == (
        "ProviderConnectionError provider=x"
    )
    assert describe_error(ValueError(BODY_KEY)) == "ValueError"
    _assert_clean(describe_error(OSError(URL_IN_MESSAGE)), "describe_error(OSError)")


def test_describe_error_keeps_authored_configuration_messages_redacted() -> None:
    exc = ProviderConfigurationError(f"Bad base URL {SECRET_BASE_URL}", provider="openai")
    out = describe_error(exc)
    assert out.startswith("ProviderConfigurationError: Bad base URL https://")
    _assert_clean(out, "describe_error(config)")


# ======================================================================= #1 health checks


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    "scenario",
    [
        "http_401",
        "http_403",
        "http_500",
        "connect_error",
        "read_timeout",
        "remote_protocol_error",
        "malformed_200",
    ],
)
async def test_health_error_never_exposes_secrets_or_upstream_details(
    adapter: str, scenario: str
) -> None:
    handlers: dict[str, Handler] = {
        "http_401": _http(401),
        "http_403": _http(403),
        "http_500": _http(500),
        "connect_error": _raises(httpx.ConnectError(URL_IN_MESSAGE)),
        "read_timeout": _raises(httpx.ReadTimeout(URL_IN_MESSAGE)),
        "remote_protocol_error": _raises(httpx.RemoteProtocolError(URL_IN_MESSAGE)),
        "malformed_200": lambda request: httpx.Response(200, text=f"<html>{BODY_MARKER}</html>"),
    }
    provider = ADAPTERS[adapter](handlers[scenario])

    health = await provider.health_check()

    assert health.healthy is False
    assert health.provider == adapter
    assert health.error  # still says that something is wrong
    _assert_clean(health.model_dump_json(), f"{adapter} health ({scenario})")
    if scenario.startswith("http_"):
        status = scenario.split("_")[1]
        assert status in health.error  # diagnostic status survives


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_healthy_payload_reports_sanitized_base_url(adapter: str) -> None:
    payload = MODELS_OK[adapter]
    provider = ADAPTERS[adapter](lambda request: httpx.Response(200, json=payload))

    health = await provider.health_check()

    assert health.healthy is True
    assert health.details["base_url"] == "https://api.example.test:8443/v1"
    _assert_clean(health.model_dump_json(), f"{adapter} healthy payload")


async def test_ollama_health_non_200_does_not_return_response_body() -> None:
    provider = ADAPTERS["ollama"](_http(503))
    health = await provider.health_check()
    assert health.error == "HTTP 503"


async def test_ollama_health_keeps_useful_redacted_transport_diagnostic() -> None:
    provider = ADAPTERS["ollama"](
        _raises(httpx.ConnectError(f"Connection refused {URL_IN_MESSAGE}"))
    )
    health = await provider.health_check()
    assert "Connection refused" in health.error
    assert "ConnectError" in health.error
    assert "api.example.test:8443" in health.error  # host is fine, credentials are not
    _assert_clean(health.error, "ollama transport diagnostic")
    assert len(health.error) < 400


@pytest.fixture
def _registry():
    reset_provider_registry()
    yield
    reset_provider_registry()


async def test_router_health_aggregate_never_exposes_secrets(_registry) -> None:
    provider = ADAPTERS["openai"](_http(401))
    register_provider("openai", lambda settings: provider)
    router = ProviderRouter(primary_provider="openai", fallback_enabled=False, max_retries=0)
    with pytest.raises(ProviderResponseError):
        await router.generate(REQUEST)  # populates last_failure

    health = await router.health_check()

    _assert_clean(health.model_dump_json(), "router aggregate health")
    last = health.details["last_failure"]
    assert last["status_code"] == 401 and last["provider"] == "openai"


async def test_health_endpoint_never_exposes_secrets() -> None:
    provider = ADAPTERS["anthropic"](_http(401))
    app.dependency_overrides[get_provider] = lambda: provider
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            res = await client.get("/health")
    finally:
        app.dependency_overrides.clear()
    assert res.status_code == 200
    assert res.json()["status"] == "degraded"
    _assert_clean(res.text, "GET /health")
    assert "status=401" in res.text


# ========================================================================= #2 log hygiene


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog


async def _drain(provider: LLMProvider) -> None:
    async for _ in provider.stream(REQUEST):
        pass


LOG_SCENARIOS: dict[str, Handler] = {
    "http_400": _http(400),
    "http_401": _http(401),
    "http_429": _http(429),
    "http_500": _http(500),
    "connect_error": _raises(httpx.ConnectError(URL_IN_MESSAGE)),
    "read_timeout": _raises(httpx.ReadTimeout(URL_IN_MESSAGE)),
    "remote_protocol_error": _raises(httpx.RemoteProtocolError(URL_IN_MESSAGE)),
    "decoding_error": _raises(httpx.DecodingError(URL_IN_MESSAGE)),
    "malformed_200": lambda request: httpx.Response(200, text=f"{BODY_MARKER} {BODY_KEY}"),
}


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("scenario", LOG_SCENARIOS)
async def test_generate_logs_never_contain_secrets_or_bodies(
    logs: pytest.LogCaptureFixture, adapter: str, scenario: str
) -> None:
    provider = ADAPTERS[adapter](LOG_SCENARIOS[scenario])
    with pytest.raises((ProviderError, httpx.HTTPError)):
        await provider.generate(REQUEST)
    _assert_clean(logs.text, f"{adapter}.generate logs ({scenario})")


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("scenario", LOG_SCENARIOS)
async def test_stream_logs_never_contain_secrets_or_bodies(
    logs: pytest.LogCaptureFixture, adapter: str, scenario: str
) -> None:
    provider = ADAPTERS[adapter](LOG_SCENARIOS[scenario])
    try:
        await _drain(provider)
    except (ProviderError, httpx.HTTPError):
        pass
    _assert_clean(logs.text, f"{adapter}.stream logs ({scenario})")


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_http_failure_logs_keep_the_status_for_diagnosis(
    logs: pytest.LogCaptureFixture, adapter: str
) -> None:
    provider = ADAPTERS[adapter](_http(503))
    with pytest.raises(ProviderResponseError):
        await provider.generate(REQUEST)
    with pytest.raises(ProviderResponseError):
        await _drain(provider)
    assert "503" in logs.text  # status stays visible
    _assert_clean(logs.text, f"{adapter} http_503 logs")


@pytest.mark.parametrize("adapter", ["openai", "anthropic", "gemini", "ollama"])
async def test_malformed_stream_frames_do_not_log_their_content(
    logs: pytest.LogCaptureFixture, adapter: str
) -> None:
    body = f"data: {{{BODY_MARKER} {BODY_KEY} not json\n\n".encode()
    provider = ADAPTERS[adapter](lambda request: httpx.Response(200, content=body))
    try:
        await _drain(provider)
    except (ProviderError, httpx.HTTPError):
        pass  # truncation handling may raise; only the logs matter here
    _assert_clean(logs.text, f"{adapter} malformed-frame logs")


@pytest.mark.parametrize("adapter", ADAPTERS)
async def test_health_check_failure_logs_never_contain_secrets(
    logs: pytest.LogCaptureFixture, adapter: str
) -> None:
    provider = ADAPTERS[adapter](_raises(httpx.ConnectError(URL_IN_MESSAGE)))
    await provider.health_check()
    _assert_clean(logs.text, f"{adapter} health-check logs")


# ----------------------------------------------------------------------------- router logs


class _Raising(LLMProvider):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(self, request: ChatRequest) -> ChatResponse:
        raise self.error

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        raise self.error
        yield  # pragma: no cover

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="x")

    async def aclose(self) -> None:
        pass


def _leaky(status: int, provider: str) -> ProviderResponseError:
    return ProviderResponseError(
        f"{provider} returned HTTP {status}: {BODY_MARKER} {BODY_KEY} {SECRET_BASE_URL}",
        provider=provider,
        status_code=status,
    )


@pytest.mark.parametrize("status", [401, 404, 503])
@pytest.mark.parametrize("mode", ["generate", "stream"])
async def test_router_logs_never_contain_upstream_text(
    logs: pytest.LogCaptureFixture, _registry, status: int, mode: str
) -> None:
    register_provider("ollama", lambda s: _Raising(_leaky(status, "ollama")))
    register_provider("openai", lambda s: _Raising(_leaky(503, "openai")))
    router = ProviderRouter(
        primary_provider="ollama",
        fallback_provider="openai",
        fallback_enabled=True,
        max_retries=1,
        allow_cloud_fallback=True,
    )
    with pytest.raises(ProviderResponseError):
        if mode == "generate":
            await router.generate(REQUEST)
        else:
            await _drain(router)

    _assert_clean(logs.text, f"router {mode} logs (status {status})")
    assert f"status={status}" in logs.text  # useful context kept
    assert "provider=ollama" in logs.text


async def test_router_retry_and_fallback_logs_are_clean(
    logs: pytest.LogCaptureFixture, _registry
) -> None:
    register_provider("ollama", lambda s: _Raising(_leaky(503, "ollama")))
    register_provider("openai", lambda s: _Raising(_leaky(502, "openai")))
    router = ProviderRouter(
        primary_provider="ollama",
        fallback_provider="openai",
        fallback_enabled=True,
        max_retries=2,
        allow_cloud_fallback=True,
    )
    with pytest.raises(ProviderResponseError):
        await router.generate(REQUEST)
    assert "Retrying" in logs.text and "also failed" in logs.text
    _assert_clean(logs.text, "router retry/fallback logs")


# ------------------------------------------------------------------- API + extractor logs


async def test_api_error_handler_logs_and_response_are_sanitized(
    logs: pytest.LogCaptureFixture,
) -> None:
    provider = _Raising(_leaky(502, "openai"))
    orchestrator = BrainOrchestrator(provider=provider)
    app.dependency_overrides[get_provider] = lambda: provider
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_memory_service] = lambda: None
    app.state.provider, app.state.orchestrator, app.state.memory_service = (
        provider,
        orchestrator,
        None,
    )
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
            res = await client.post("/api/chat", json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()

    assert res.status_code == 502
    _assert_clean(logs.text, "API error-handler logs")
    # client-facing detail is redacted for credentials (bodies are the caller's own request)
    for secret in (BODY_KEY, PASSWORD, QUERY_KEY, USER + ":"):
        assert secret not in res.text
    assert "provider=openai" in logs.text and "status=502" in logs.text


async def test_memory_extraction_failure_log_is_sanitized(logs: pytest.LogCaptureFixture) -> None:
    extractor = InferredMemoryExtractor(provider=_Raising(_leaky(500, "ollama")))
    result: list[Any] = await extractor.extract("I really enjoy long hikes in the mountains.")
    assert result == []
    _assert_clean(logs.text, "memory extraction logs")
    assert "status=500" in logs.text
