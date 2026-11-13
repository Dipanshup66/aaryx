"""Unit tests for OpenAIProvider using httpx.MockTransport."""

import json

import httpx
import pytest

from app.providers.base import (
    ChatMessage,
    ChatRequest,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.providers.openai import OpenAIProvider


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="You are a helpful companion."),
            ChatMessage(role="user", content="Hello OpenAI!"),
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=150,
    )


def test_missing_api_key_raises_error():
    """Verify initialization fails when API key is missing."""
    with pytest.raises(ProviderConfigurationError) as exc_info:
        OpenAIProvider(api_key="", client=httpx.AsyncClient())
    assert "OpenAI API key is missing" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_generate_success(sample_chat_request):
    """Verify successful non-streaming chat generation and payload structure."""
    captured_request = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-openai-key"
        captured_request["json"] = json.loads(request.content.decode("utf-8"))

        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenAI!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 14,
                    "completion_tokens": 6,
                    "total_tokens": 20,
                },
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    response = await provider.generate(sample_chat_request)

    assert response.content == "Hello from OpenAI!"
    assert response.model == "gpt-4o-mini"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 14
    assert response.usage.completion_tokens == 6
    assert response.usage.total_tokens == 20

    payload = captured_request["json"]
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 150
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_stream_success(sample_chat_request):
    """Verify streaming response chunks from OpenAI SSE stream."""
    sse_body = (
        b'data: {"id":"1","choices":[{"delta":{"content":"Hello "}}]}\n\n'
        b'data: {"id":"2","choices":[{"delta":{"content":"world!"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(status_code=200, content=sse_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    chunks = []
    async for chunk in provider.stream(sample_chat_request):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].delta == "Hello "
    assert chunks[1].delta == "world!"
    assert chunks[1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_authentication_error(sample_chat_request):
    """Verify 401 Unauthorized raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"error": {"message": "Incorrect API key provided"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="invalid-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "authentication error (401)" in str(exc_info.value).lower()
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_rate_limit_error(sample_chat_request):
    """Verify 429 Too Many Requests raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={"error": {"message": "Rate limit reached"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "rate limit" in str(exc_info.value).lower()
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_timeout_error(sample_chat_request):
    """Verify timeout during request raises ProviderTimeoutError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "timed out" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_connection_error(sample_chat_request):
    """Verify network connection failure raises ProviderConnectionError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    with pytest.raises(ProviderConnectionError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "Cannot connect" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_malformed_response(sample_chat_request):
    """Verify non-JSON or missing choices payload raises ProviderResponseError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, text="<HTML>Not JSON</HTML>")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    with pytest.raises(ProviderResponseError) as exc_info:
        await provider.generate(sample_chat_request)

    assert "Invalid response" in str(exc_info.value)
    assert exc_info.value.provider == "openai"


@pytest.mark.asyncio
async def test_list_models_success():
    """Verify models listing endpoint maps OpenAI format into ModelInfo domain models."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"id": "gpt-4o", "created": 1715368132},
                    {"id": "gpt-4o-mini", "created": 1721172741},
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    models = await provider.list_models()
    assert len(models) == 2
    names = [m.name for m in models]
    assert "gpt-4o" in names
    assert "gpt-4o-mini" in names


@pytest.mark.asyncio
async def test_health_check_healthy():
    """Verify health check returns healthy when models query succeeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=200, json={"data": [{"id": "gpt-4o"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    health = await provider.health_check()
    assert health.healthy is True
    assert health.provider == "openai"
    assert health.details["model_count"] == 1


@pytest.mark.asyncio
async def test_health_check_unhealthy():
    """Verify health check returns unhealthy when models query fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, json={"error": "Unauthorized"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="test-openai-key", client=client)

    health = await provider.health_check()
    assert health.healthy is False
    assert health.provider == "openai"
    assert health.error is not None
